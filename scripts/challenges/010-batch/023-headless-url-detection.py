#!/usr/bin/env python3
"""
Challenge 010-023: Headless URL Anomaly Detection

Fully headless approach - NO explicit rules for SQL injection, XSS, etc.

Instead:
1. Parse URL into structured representation
2. Encode structure with Holon
3. Learn "normal" via decaying accumulator
4. Flag anomalies by similarity

URL parsing:
  /some/path?key=val&key2=val2

  → [["some", "path"], [["key", "val"], ["key2", "val2"]]]

The VSA encoding naturally makes attack vectors different because:
- Attack values ("' OR '1'='1") are rare atoms
- Normal values ("users", "123", "api") dominate the accumulator
- Rare atoms → orthogonal vectors → low similarity

No regex. No pattern matching. Pure learned anomaly detection.
"""

import sys
import time
import random
import re
from urllib.parse import urlparse, parse_qs, unquote
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

sys.path.insert(0, str(__file__).rsplit("/", 4)[0])

from holon import DeterministicVectorManager
from holon.encoder import Encoder


# =============================================================================
# CONFIGURATION
# =============================================================================

GLOBAL_SEED = 42
DIMENSIONS = 4096
DECAY_FACTOR = 0.9995
ANOMALY_THRESHOLD = 0.40  # Tuned for headless detection
WARMUP_REQUESTS = 500


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return float(np.dot(a.astype(np.float64), b.astype(np.float64)) / (norm_a * norm_b))


# =============================================================================
# URL PARSER
# =============================================================================

def parse_url_to_structure(url: str) -> List:
    """
    Parse URL into structured representation.

    /some/path?key=val&key2=val2
    → [["some", "path"], [["key", "val"], ["key2", "val2"]]]

    /api/users/123
    → [["api", "users", "123"], []]

    /search?q=hello+world
    → [["search"], [["q", "hello world"]]]
    """
    # Handle URL decoding
    url = unquote(url)

    # Split path and query
    if "?" in url:
        path_part, query_part = url.split("?", 1)
    else:
        path_part = url
        query_part = ""

    # Parse path segments
    path_segments = [seg for seg in path_part.split("/") if seg]

    # Parse query params
    query_pairs = []
    if query_part:
        for param in query_part.split("&"):
            if "=" in param:
                key, val = param.split("=", 1)
                query_pairs.append([key, val])
            else:
                query_pairs.append([param, ""])

    return [path_segments, query_pairs]


def structure_to_string(structure: List) -> str:
    """Convert structure back to readable format for debugging."""
    path_segments, query_pairs = structure
    path = "/" + "/".join(path_segments) if path_segments else "/"
    if query_pairs:
        query = "&".join(f"{k}={v}" for k, v in query_pairs)
        return f"{path}?{query}"
    return path


# =============================================================================
# DECAYING ACCUMULATOR
# =============================================================================

class DecayingAccumulator:
    def __init__(self, dimensions: int, decay: float = DECAY_FACTOR):
        self.dimensions = dimensions
        self.decay = decay
        self.accumulator = np.zeros(dimensions, dtype=np.float64)
        self.count = 0

    def update(self, vector: np.ndarray, weight: float = 1.0):
        self.accumulator = self.decay * self.accumulator + weight * vector.astype(np.float64)
        self.count += 1

    def get_normalized(self) -> np.ndarray:
        norm = np.linalg.norm(self.accumulator)
        if norm < 1e-10:
            return np.zeros(self.dimensions, dtype=np.float32)
        return (self.accumulator / norm).astype(np.float32)


# =============================================================================
# HEADLESS DETECTOR
# =============================================================================

@dataclass
class DetectionResult:
    url: str
    structure: List
    is_flagged: bool
    similarity: float
    is_warmup: bool


class HeadlessURLDetector:
    """
    Fully headless URL anomaly detection.

    No rules. No patterns. Just learned structure.
    """

    def __init__(
        self,
        global_seed: int = GLOBAL_SEED,
        decay: float = DECAY_FACTOR,
        threshold: float = ANOMALY_THRESHOLD,
        warmup: int = WARMUP_REQUESTS,
    ):
        self.vm = DeterministicVectorManager(dimensions=DIMENSIONS, global_seed=global_seed)
        self.encoder = Encoder(vector_manager=self.vm)

        self.accumulator = DecayingAccumulator(DIMENSIONS, decay=decay)
        self.threshold = threshold
        self.warmup = warmup

        self.requests_seen = 0

    def process(self, url: str) -> DetectionResult:
        """Process a URL and detect anomalies."""
        self.requests_seen += 1
        is_warmup = self.requests_seen <= self.warmup

        # Parse URL to structure
        structure = parse_url_to_structure(url)

        # Encode structure with Holon
        vec = self.encoder.encode_data(structure)

        # Get current model
        model = self.accumulator.get_normalized()

        # Compute similarity
        if self.requests_seen <= 1:
            similarity = 1.0
        else:
            similarity = cosine_similarity(vec, model)

        # Decision (pure similarity, no rules)
        if is_warmup:
            is_flagged = False
        else:
            is_flagged = similarity < self.threshold

        # Update model
        if not is_flagged:
            self.accumulator.update(vec)
        else:
            self.accumulator.update(vec, weight=0.1)

        return DetectionResult(
            url=url,
            structure=structure,
            is_flagged=is_flagged,
            similarity=similarity,
            is_warmup=is_warmup,
        )


# =============================================================================
# URL GENERATOR
# =============================================================================

class URLGenerator:
    """Generate realistic URL traffic."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

        # Normal API patterns
        self.api_patterns = [
            "/api/users",
            "/api/users/{id}",
            "/api/orders",
            "/api/orders/{id}",
            "/api/products",
            "/api/products/{id}",
            "/api/auth/login",
            "/api/auth/logout",
            "/api/search",
        ]

        # Normal query params
        self.normal_params = {
            "limit": lambda: str(self.rng.choice([10, 20, 50, 100])),
            "offset": lambda: str(self.rng.randint(0, 100)),
            "sort": lambda: self.rng.choice(["name", "date", "id", "price"]),
            "order": lambda: self.rng.choice(["asc", "desc"]),
            "filter": lambda: self.rng.choice(["active", "pending", "completed"]),
            "q": lambda: self.rng.choice(["laptop", "phone", "tablet", "camera"]),
        }

    def generate_benign(self) -> str:
        """Generate a normal URL."""
        pattern = self.rng.choice(self.api_patterns)

        # Replace {id} with actual ID
        if "{id}" in pattern:
            pattern = pattern.replace("{id}", str(self.rng.randint(1, 10000)))

        # Maybe add query params
        if self.rng.random() < 0.4:
            n_params = self.rng.randint(1, 3)
            params = self.rng.sample(list(self.normal_params.keys()), n_params)
            query = "&".join(f"{p}={self.normal_params[p]()}" for p in params)
            return f"{pattern}?{query}"

        return pattern

    def generate_malicious(self) -> Tuple[str, str]:
        """Generate a malicious URL. Returns (url, attack_type)."""
        attack = self.rng.choice([
            "sqli_path",
            "sqli_query",
            "traversal",
            "xss_query",
            "hidden_file",
            "command_injection",
        ])

        if attack == "sqli_path":
            payloads = [
                "' OR '1'='1",
                "'; DROP TABLE users--",
                "1 OR 1=1",
                "admin'--",
                "' UNION SELECT * FROM users--",
            ]
            return f"/api/users/{self.rng.choice(payloads)}", attack

        elif attack == "sqli_query":
            payloads = [
                "' OR '1'='1",
                "1; DROP TABLE--",
                "admin' OR '1'='1",
            ]
            return f"/api/search?q={self.rng.choice(payloads)}", attack

        elif attack == "traversal":
            payloads = [
                "../../../etc/passwd",
                "..\\..\\..\\windows\\system32",
                "....//....//etc/passwd",
                "%2e%2e%2f%2e%2e%2f",
            ]
            return f"/api/files/{self.rng.choice(payloads)}", attack

        elif attack == "xss_query":
            payloads = [
                "<script>alert(1)</script>",
                "javascript:alert(1)",
                "<img onerror=alert(1)>",
                "'-alert(1)-'",
            ]
            return f"/api/search?q={self.rng.choice(payloads)}", attack

        elif attack == "hidden_file":
            paths = [
                "/.git/config",
                "/.env",
                "/.htaccess",
                "/wp-config.php",
                "/.aws/credentials",
                "/admin/config.php",
            ]
            return self.rng.choice(paths), attack

        else:  # command_injection
            payloads = [
                "; cat /etc/passwd",
                "| ls -la",
                "`whoami`",
                "$(cat /etc/passwd)",
            ]
            return f"/api/exec?cmd=test{self.rng.choice(payloads)}", attack

    def generate_stream(self, n: int, malicious_ratio: float = 0.02) -> List[Tuple[str, Optional[str]]]:
        """Generate stream. Returns [(url, attack_type or None), ...]"""
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
    print("Challenge 010-023: Headless URL Anomaly Detection")
    print("=" * 80)
    print("""
Fully headless approach - NO explicit attack rules!

URL parsing:
  /api/users/123?limit=10&sort=name
  → [["api", "users", "123"], [["limit", "10"], ["sort", "name"]]]

Detection theory:
  - Normal values dominate the accumulator
  - Attack values are rare atoms → orthogonal vectors
  - Low similarity = anomaly

No regex. No pattern matching. Pure learned detection.
""")

    # Initialize
    generator = URLGenerator(seed=42)
    detector = HeadlessURLDetector(
        global_seed=GLOBAL_SEED,
        decay=DECAY_FACTOR,
        threshold=ANOMALY_THRESHOLD,
        warmup=WARMUP_REQUESTS,
    )

    print(f"Configuration:")
    print(f"  Decay factor: {DECAY_FACTOR}")
    print(f"  Anomaly threshold: {ANOMALY_THRESHOLD}")
    print(f"  Warmup: {WARMUP_REQUESTS} requests")

    # Show parsing examples
    print(f"\n--- URL Parsing Examples ---")
    examples = [
        "/api/users/123",
        "/api/search?q=laptop&limit=10",
        "/api/users/' OR '1'='1",
        "/../../../etc/passwd",
    ]
    for url in examples:
        structure = parse_url_to_structure(url)
        print(f"  {url}")
        print(f"    → {structure}")

    # Generate stream
    n_requests = 10000
    malicious_ratio = 0.02
    stream = generator.generate_stream(n_requests, malicious_ratio)

    actual_malicious = sum(1 for _, a in stream if a is not None)
    actual_benign = n_requests - actual_malicious

    print(f"\n--- Stream Stats ---")
    print(f"  Total: {n_requests}")
    print(f"  Benign: {actual_benign} ({100*actual_benign/n_requests:.1f}%)")
    print(f"  Malicious: {actual_malicious} ({100*actual_malicious/n_requests:.1f}%)")

    # Process stream
    print(f"\n--- Processing Stream ---")

    results = []
    start = time.time()

    for url, attack_type in stream:
        result = detector.process(url)
        results.append((result, attack_type))

    total_time = time.time() - start

    print(f"  Processed {n_requests} URLs in {total_time:.2f}s")
    print(f"  Throughput: {n_requests/total_time:.0f} URLs/sec")

    # Metrics (post-warmup)
    post_warmup = [(r, a) for r, a in results if not r.is_warmup]

    tp = sum(1 for r, a in post_warmup if a is not None and r.is_flagged)
    fp = sum(1 for r, a in post_warmup if a is None and r.is_flagged)
    fn = sum(1 for r, a in post_warmup if a is not None and not r.is_flagged)
    tn = sum(1 for r, a in post_warmup if a is None and not r.is_flagged)

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(0.001, precision + recall)

    print(f"\n--- Results (Headless Detection) ---")
    print(f"  Confusion Matrix:")
    print(f"    TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    print(f"\n  Metrics:")
    print(f"    Precision: {precision:.1%}")
    print(f"    Recall:    {recall:.1%}")
    print(f"    F1 Score:  {f1:.3f}")

    # Detection by attack type
    print(f"\n--- Detection by Attack Type ---")
    attack_stats = {}
    for r, a in post_warmup:
        if a is not None:
            if a not in attack_stats:
                attack_stats[a] = {"detected": 0, "missed": 0}
            if r.is_flagged:
                attack_stats[a]["detected"] += 1
            else:
                attack_stats[a]["missed"] += 1

    for attack, stats in sorted(attack_stats.items()):
        total = stats["detected"] + stats["missed"]
        rate = stats["detected"] / total if total > 0 else 0
        print(f"  {attack}: {stats['detected']}/{total} ({rate:.0%})")

    # Similarity distribution
    benign_sims = [r.similarity for r, a in post_warmup if a is None]
    malicious_sims = [r.similarity for r, a in post_warmup if a is not None]

    print(f"\n--- Similarity Distribution ---")
    print(f"  Benign:    min={min(benign_sims):.3f}, mean={np.mean(benign_sims):.3f}, max={max(benign_sims):.3f}")
    print(f"  Malicious: min={min(malicious_sims):.3f}, mean={np.mean(malicious_sims):.3f}, max={max(malicious_sims):.3f}")

    # Separation analysis
    benign_below_threshold = sum(1 for s in benign_sims if s < ANOMALY_THRESHOLD)
    malicious_above_threshold = sum(1 for s in malicious_sims if s >= ANOMALY_THRESHOLD)

    print(f"\n--- Threshold Analysis (threshold={ANOMALY_THRESHOLD}) ---")
    print(f"  Benign below threshold: {benign_below_threshold} ({100*benign_below_threshold/len(benign_sims):.1f}%)")
    print(f"  Malicious above threshold: {malicious_above_threshold} ({100*malicious_above_threshold/len(malicious_sims):.1f}%)")

    # Sample detections
    print(f"\n--- Sample True Positives ---")
    tps = [(r, a) for r, a in post_warmup if a is not None and r.is_flagged][:5]
    for r, a in tps:
        print(f"  sim={r.similarity:.3f} | {a} | {r.url[:60]}")

    print(f"\n--- Sample False Negatives ---")
    fns = [(r, a) for r, a in post_warmup if a is not None and not r.is_flagged][:5]
    for r, a in fns:
        print(f"  sim={r.similarity:.3f} | {a} | {r.url[:60]}")

    print(f"\n--- Sample False Positives ---")
    fps = [(r, a) for r, a in post_warmup if a is None and r.is_flagged][:5]
    for r, a in fps:
        print(f"  sim={r.similarity:.3f} | {r.url[:60]}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: Headless URL Detection")
    print("=" * 80)
    print(f"""
Approach:
  URL → Parse → Structure → Encode → Compare to Accumulator → Flag/Allow

  /api/users/123?limit=10
  → [["api", "users", "123"], [["limit", "10"]]]
  → VSA vector
  → similarity to learned "normal"

Results:
  F1 Score:    {f1:.3f}
  Precision:   {precision:.1%}
  Recall:      {recall:.1%}
  Throughput:  {n_requests/total_time:.0f} URLs/sec

Separation:
  Benign similarity:    {np.mean(benign_sims):.3f} (mean)
  Malicious similarity: {np.mean(malicious_sims):.3f} (mean)
  Gap: {np.mean(benign_sims) - np.mean(malicious_sims):.3f}

Key insight:
  Attack values like "' OR '1'='1" become rare atoms that
  produce orthogonal vectors, naturally lowering similarity.

  NO RULES NEEDED - pure learned detection!
""")

    return f1, precision, recall


if __name__ == "__main__":
    f1, precision, recall = main()

    # Try tuning threshold if needed
    if f1 < 0.9:
        print("\n" + "=" * 80)
        print("TUNING: Searching for optimal threshold...")
        print("=" * 80)

        # Re-run with different thresholds
        best_f1 = 0
        best_threshold = 0.4

        for threshold in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
            generator = URLGenerator(seed=42)
            detector = HeadlessURLDetector(
                global_seed=GLOBAL_SEED,
                decay=DECAY_FACTOR,
                threshold=threshold,
                warmup=WARMUP_REQUESTS,
            )

            stream = generator.generate_stream(10000, 0.02)
            results = []
            for url, attack_type in stream:
                result = detector.process(url)
                results.append((result, attack_type))

            post_warmup = [(r, a) for r, a in results if not r.is_warmup]
            tp = sum(1 for r, a in post_warmup if a is not None and r.is_flagged)
            fp = sum(1 for r, a in post_warmup if a is None and r.is_flagged)
            fn = sum(1 for r, a in post_warmup if a is not None and not r.is_flagged)

            p = tp / max(1, tp + fp)
            r = tp / max(1, tp + fn)
            f = 2 * p * r / max(0.001, p + r)

            print(f"  threshold={threshold:.2f}: P={p:.1%}, R={r:.1%}, F1={f:.3f}")

            if f > best_f1:
                best_f1 = f
                best_threshold = threshold

        print(f"\n  Best: threshold={best_threshold:.2f} → F1={best_f1:.3f}")
