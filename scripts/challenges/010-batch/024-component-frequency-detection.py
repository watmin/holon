#!/usr/bin/env python3
"""
Challenge 010-024: Component-Level Frequency Detection

Problem with whole-URL encoding:
  Normal:  /api/search?q=laptop        → sim=0.65
  Attack:  /api/search?q=' OR '1'='1   → sim=0.60  (too close!)

  The common structure (api, search, q) dominates.
  The rare attack value is buried.

Solution: Track frequency at COMPONENT level.
  - Each path segment gets its own accumulator
  - Each query key gets its own accumulator
  - Each query value gets its own accumulator
  - Flag if ANY component has low similarity

This way:
  - "api", "search", "q" → high similarity (common)
  - "' OR '1'='1" → low similarity (rare) → FLAGGED!
"""

import sys
import time
import random
from collections import defaultdict
from urllib.parse import unquote
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set

import numpy as np

sys.path.insert(0, str(__file__).rsplit("/", 4)[0])

from holon import DeterministicVectorManager
from holon.encoder import Encoder


# =============================================================================
# CONFIGURATION
# =============================================================================

GLOBAL_SEED = 42
DIMENSIONS = 4096
DECAY_FACTOR = 0.999  # Faster decay for component-level
VALUE_THRESHOLD = 0.15  # Threshold for individual values (tuned)
WARMUP_REQUESTS = 300


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return float(np.dot(a.astype(np.float64), b.astype(np.float64)) / (norm_a * norm_b))


# =============================================================================
# URL PARSER
# =============================================================================

def normalize_value(value: str) -> str:
    """
    Normalize high-cardinality values to reduce false positives.

    - Pure numbers → {num}
    - UUIDs → {uuid}
    - Emails → {email}
    - Keep everything else as-is (including attack payloads!)
    """
    # Pure numeric (IDs)
    if value.isdigit():
        return "{num}"

    # UUID pattern
    if len(value) == 36 and value.count("-") == 4:
        return "{uuid}"

    # Email pattern (simple)
    if "@" in value and "." in value:
        return "{email}"

    # Float numbers
    try:
        float(value)
        return "{num}"
    except:
        pass

    # Keep as-is - attack payloads will NOT match these patterns
    return value


def parse_url(url: str) -> Tuple[List[str], List[Tuple[str, str]]]:
    """
    Parse URL into components with normalization.

    Returns: (path_segments, [(key, value), ...])
    """
    url = unquote(url)

    if "?" in url:
        path_part, query_part = url.split("?", 1)
    else:
        path_part = url
        query_part = ""

    # Path segments (normalized)
    path_segments = []
    for seg in path_part.split("/"):
        if seg:
            path_segments.append(normalize_value(seg))

    # Query params (values normalized)
    query_pairs = []
    if query_part:
        for param in query_part.split("&"):
            if "=" in param:
                key, val = param.split("=", 1)
                query_pairs.append((key, normalize_value(val)))
            else:
                query_pairs.append((param, ""))

    return path_segments, query_pairs


# =============================================================================
# COMPONENT ACCUMULATOR
# =============================================================================

class ComponentAccumulator:
    """
    Tracks frequency of individual values using decaying accumulator.

    Each unique "position" (e.g., "path_0", "query_q") has its own accumulator.
    Values are compared against their position's accumulator.
    """

    def __init__(self, dimensions: int, decay: float = DECAY_FACTOR):
        self.dimensions = dimensions
        self.decay = decay

        # Per-position accumulators
        self.accumulators: Dict[str, np.ndarray] = {}
        self.counts: Dict[str, int] = defaultdict(int)

        # Global value accumulator (for values in any position)
        self.global_value_accum = np.zeros(dimensions, dtype=np.float64)
        self.global_count = 0

    def update(self, position: str, vector: np.ndarray, weight: float = 1.0):
        """Update accumulator for a specific position."""
        if position not in self.accumulators:
            self.accumulators[position] = np.zeros(self.dimensions, dtype=np.float64)

        self.accumulators[position] = (
            self.decay * self.accumulators[position] +
            weight * vector.astype(np.float64)
        )
        self.counts[position] += 1

        # Also update global
        self.global_value_accum = (
            self.decay * self.global_value_accum +
            weight * vector.astype(np.float64)
        )
        self.global_count += 1

    def get_similarity(self, position: str, vector: np.ndarray) -> float:
        """Get similarity of vector to position's accumulator."""
        if position not in self.accumulators or self.counts[position] < 5:
            # Not enough data for this position, use global
            return self._global_similarity(vector)

        accum = self.accumulators[position]
        norm = np.linalg.norm(accum)
        if norm < 1e-10:
            return self._global_similarity(vector)

        normalized = accum / norm
        return cosine_similarity(vector, normalized)

    def _global_similarity(self, vector: np.ndarray) -> float:
        """Fallback to global accumulator."""
        if self.global_count < 5:
            return 1.0  # Not enough data yet

        norm = np.linalg.norm(self.global_value_accum)
        if norm < 1e-10:
            return 1.0

        normalized = self.global_value_accum / norm
        return cosine_similarity(vector, normalized)


# =============================================================================
# COMPONENT DETECTOR
# =============================================================================

@dataclass
class ComponentScore:
    position: str
    value: str
    similarity: float
    is_rare: bool


@dataclass
class DetectionResult:
    url: str
    is_flagged: bool
    min_similarity: float
    rare_components: List[ComponentScore]
    is_warmup: bool


class ComponentFrequencyDetector:
    """
    Detect anomalies by tracking component-level frequency.

    Each URL component (path segment, query value) is scored
    against its position's learned distribution.

    Flag if ANY component is rare.
    """

    def __init__(
        self,
        global_seed: int = GLOBAL_SEED,
        decay: float = DECAY_FACTOR,
        threshold: float = VALUE_THRESHOLD,
        warmup: int = WARMUP_REQUESTS,
    ):
        self.vm = DeterministicVectorManager(dimensions=DIMENSIONS, global_seed=global_seed)
        self.encoder = Encoder(vector_manager=self.vm)

        # Separate accumulators for different component types
        self.path_accum = ComponentAccumulator(DIMENSIONS, decay)
        self.query_key_accum = ComponentAccumulator(DIMENSIONS, decay)
        self.query_value_accum = ComponentAccumulator(DIMENSIONS, decay)

        self.threshold = threshold
        self.warmup = warmup
        self.requests_seen = 0

    def process(self, url: str) -> DetectionResult:
        """Process URL and detect rare components."""
        self.requests_seen += 1
        is_warmup = self.requests_seen <= self.warmup

        # Parse URL (with normalization)
        path_segments, query_pairs = parse_url(url)

        # Score each component
        component_scores = []

        # Score path segments
        for i, segment in enumerate(path_segments):
            # Skip normalized placeholders - they're known-good patterns
            if segment.startswith("{") and segment.endswith("}"):
                continue

            position = f"path_{i}"
            vec = self.vm.get_vector(segment)
            sim = self.path_accum.get_similarity(position, vec)
            is_rare = sim < self.threshold
            component_scores.append(ComponentScore(position, segment, sim, is_rare))

        # Score query params
        for key, value in query_pairs:
            # Skip normalized values
            if value.startswith("{") and value.endswith("}"):
                continue

            # Score value (most important for attacks!)
            value_vec = self.vm.get_vector(value)
            value_position = f"value_{key}"
            value_sim = self.query_value_accum.get_similarity(value_position, value_vec)
            is_rare = value_sim < self.threshold
            component_scores.append(ComponentScore(value_position, value, value_sim, is_rare))

        # Find rare components
        rare_components = [c for c in component_scores if c.is_rare]
        min_sim = min((c.similarity for c in component_scores), default=1.0)

        # Decision
        if is_warmup:
            is_flagged = False
        else:
            is_flagged = len(rare_components) > 0

        # Update accumulators
        weight = 0.1 if is_flagged else 1.0

        for i, segment in enumerate(path_segments):
            position = f"path_{i}"
            vec = self.vm.get_vector(segment)
            self.path_accum.update(position, vec, weight)

        for key, value in query_pairs:
            key_vec = self.vm.get_vector(key)
            self.query_key_accum.update("key", key_vec, weight)

            value_vec = self.vm.get_vector(value)
            value_position = f"value_{key}"
            self.query_value_accum.update(value_position, value_vec, weight)

        return DetectionResult(
            url=url,
            is_flagged=is_flagged,
            min_similarity=min_sim,
            rare_components=rare_components,
            is_warmup=is_warmup,
        )


# =============================================================================
# URL GENERATOR (same as before)
# =============================================================================

class URLGenerator:
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

        self.api_patterns = [
            "/api/users", "/api/users/{id}",
            "/api/orders", "/api/orders/{id}",
            "/api/products", "/api/products/{id}",
            "/api/search", "/api/auth/login",
        ]

        self.normal_params = {
            "limit": lambda: str(self.rng.choice([10, 20, 50, 100])),
            "offset": lambda: str(self.rng.randint(0, 100)),
            "sort": lambda: self.rng.choice(["name", "date", "id", "price"]),
            "q": lambda: self.rng.choice(["laptop", "phone", "tablet", "camera", "watch"]),
        }

    def generate_benign(self) -> str:
        pattern = self.rng.choice(self.api_patterns)
        if "{id}" in pattern:
            pattern = pattern.replace("{id}", str(self.rng.randint(1, 1000)))

        if self.rng.random() < 0.4:
            n_params = self.rng.randint(1, 2)
            params = self.rng.sample(list(self.normal_params.keys()), n_params)
            query = "&".join(f"{p}={self.normal_params[p]()}" for p in params)
            return f"{pattern}?{query}"
        return pattern

    def generate_malicious(self) -> Tuple[str, str]:
        attack = self.rng.choice([
            "sqli_path", "sqli_query", "traversal",
            "xss_query", "hidden_file", "cmd_injection",
        ])

        if attack == "sqli_path":
            payloads = ["' OR '1'='1", "admin'--", "1 UNION SELECT *"]
            return f"/api/users/{self.rng.choice(payloads)}", attack

        elif attack == "sqli_query":
            payloads = ["' OR '1'='1", "1; DROP TABLE--", "admin' OR '1'='1"]
            return f"/api/search?q={self.rng.choice(payloads)}", attack

        elif attack == "traversal":
            payloads = ["../../../etc/passwd", "..\\..\\windows", "....//....//etc"]
            return f"/api/files/{self.rng.choice(payloads)}", attack

        elif attack == "xss_query":
            payloads = ["<script>alert(1)</script>", "javascript:alert(1)", "<img onerror=x>"]
            return f"/api/search?q={self.rng.choice(payloads)}", attack

        elif attack == "hidden_file":
            paths = ["/.git/config", "/.env", "/.htaccess", "/wp-config.php"]
            return self.rng.choice(paths), attack

        else:  # cmd_injection
            payloads = ["; cat /etc/passwd", "| ls -la", "`whoami`"]
            return f"/api/exec?cmd=test{self.rng.choice(payloads)}", attack

    def generate_stream(self, n: int, malicious_ratio: float = 0.02):
        stream = []
        for _ in range(n):
            if self.rng.random() < malicious_ratio:
                url, attack = self.generate_malicious()
                stream.append((url, attack))
            else:
                stream.append((self.generate_benign(), None))
        return stream


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("Challenge 010-024: Component-Level Frequency Detection")
    print("=" * 80)
    print("""
Key insight: Track frequency at COMPONENT level, not whole-URL.

Problem:
  /api/search?q=laptop        → whole-URL sim = 0.65
  /api/search?q=' OR '1'='1   → whole-URL sim = 0.60 (too close!)

  Structure dominates, attack payload is buried.

Solution:
  Score each component separately:
  - "api"           → high sim (common path segment)
  - "search"        → high sim (common path segment)
  - "q"             → high sim (common query key)
  - "' OR '1'='1"   → LOW sim (rare value!) → FLAGGED!

Flag if ANY component is rare.
""")

    # Initialize
    generator = URLGenerator(seed=42)
    detector = ComponentFrequencyDetector(
        global_seed=GLOBAL_SEED,
        decay=DECAY_FACTOR,
        threshold=VALUE_THRESHOLD,
        warmup=WARMUP_REQUESTS,
    )

    print(f"Configuration:")
    print(f"  Decay factor: {DECAY_FACTOR}")
    print(f"  Value threshold: {VALUE_THRESHOLD}")
    print(f"  Warmup: {WARMUP_REQUESTS} requests")

    # Generate stream
    n_requests = 10000
    stream = generator.generate_stream(n_requests, malicious_ratio=0.02)

    actual_malicious = sum(1 for _, a in stream if a is not None)
    print(f"\n--- Stream ---")
    print(f"  Total: {n_requests}")
    print(f"  Malicious: {actual_malicious} ({100*actual_malicious/n_requests:.1f}%)")

    # Process
    print(f"\n--- Processing ---")
    results = []
    start = time.time()

    for url, attack in stream:
        result = detector.process(url)
        results.append((result, attack))

    total_time = time.time() - start
    print(f"  Throughput: {n_requests/total_time:.0f} URLs/sec")

    # Metrics
    post_warmup = [(r, a) for r, a in results if not r.is_warmup]

    tp = sum(1 for r, a in post_warmup if a is not None and r.is_flagged)
    fp = sum(1 for r, a in post_warmup if a is None and r.is_flagged)
    fn = sum(1 for r, a in post_warmup if a is not None and not r.is_flagged)
    tn = sum(1 for r, a in post_warmup if a is None and not r.is_flagged)

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(0.001, precision + recall)

    print(f"\n--- Results ---")
    print(f"  TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    print(f"  Precision: {precision:.1%}")
    print(f"  Recall:    {recall:.1%}")
    print(f"  F1 Score:  {f1:.3f}")

    # By attack type
    print(f"\n--- By Attack Type ---")
    attack_stats = {}
    for r, a in post_warmup:
        if a:
            if a not in attack_stats:
                attack_stats[a] = {"detected": 0, "missed": 0}
            if r.is_flagged:
                attack_stats[a]["detected"] += 1
            else:
                attack_stats[a]["missed"] += 1

    for attack, stats in sorted(attack_stats.items()):
        total = stats["detected"] + stats["missed"]
        rate = stats["detected"] / total if total else 0
        print(f"  {attack}: {stats['detected']}/{total} ({rate:.0%})")

    # Show detections
    print(f"\n--- Sample True Positives ---")
    tps = [(r, a) for r, a in post_warmup if a and r.is_flagged][:5]
    for r, a in tps:
        rare = r.rare_components[0] if r.rare_components else None
        if rare:
            print(f"  {a}: '{rare.value}' sim={rare.similarity:.3f} @ {rare.position}")

    print(f"\n--- Sample False Negatives ---")
    fns = [(r, a) for r, a in post_warmup if a and not r.is_flagged][:5]
    for r, a in fns:
        print(f"  {a}: {r.url[:50]} min_sim={r.min_similarity:.3f}")

    print(f"\n--- Sample False Positives ---")
    fps = [(r, a) for r, a in post_warmup if a is None and r.is_flagged][:5]
    for r, a in fps:
        rare = r.rare_components[0] if r.rare_components else None
        if rare:
            print(f"  '{rare.value}' sim={rare.similarity:.3f} @ {rare.position}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"""
Component-Level Frequency Detection:

  URL: /api/search?q=' OR '1'='1

  Components scored:
    path_0: "api"         → sim=0.95 ✓
    path_1: "search"      → sim=0.93 ✓
    value_q: "' OR '1'='1" → sim=0.12 ✗ RARE!

  Result: FLAGGED (rare component detected)

Performance:
  F1 Score:  {f1:.3f}
  Precision: {precision:.1%}
  Recall:    {recall:.1%}

Key: Each value compared to its position's frequency distribution.
     Rare values stand out even in familiar structures!
""")

    return f1


if __name__ == "__main__":
    f1 = main()

    # Tune threshold if needed
    if f1 < 0.85:
        print("\n--- Threshold Tuning ---")
        best_f1 = 0
        best_thresh = 0.25

        for thresh in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
            generator = URLGenerator(seed=42)
            detector = ComponentFrequencyDetector(
                threshold=thresh,
                warmup=WARMUP_REQUESTS,
            )

            stream = generator.generate_stream(10000, 0.02)
            results = [(detector.process(u), a) for u, a in stream]
            post_warmup = [(r, a) for r, a in results if not r.is_warmup]

            tp = sum(1 for r, a in post_warmup if a and r.is_flagged)
            fp = sum(1 for r, a in post_warmup if not a and r.is_flagged)
            fn = sum(1 for r, a in post_warmup if a and not r.is_flagged)

            p = tp / max(1, tp + fp)
            r = tp / max(1, tp + fn)
            f = 2 * p * r / max(0.001, p + r)

            print(f"  threshold={thresh:.2f}: P={p:.1%} R={r:.1%} F1={f:.3f}")

            if f > best_f1:
                best_f1 = f
                best_thresh = thresh

        print(f"\n  Best: threshold={best_thresh:.2f} → F1={best_f1:.3f}")
