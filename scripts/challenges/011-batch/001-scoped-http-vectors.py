#!/usr/bin/env python3
"""
Challenge 011-001: Scoped HTTP Vectors

Instead of encoding entire requests as single vectors, maintain separate
accumulators per component:

    path_accumulator     → learns normal path patterns
    query_accumulator    → learns normal query patterns
    headers_accumulator  → learns normal header patterns
    body_accumulator     → learns normal body patterns

Detection: Compare each component to its dedicated accumulator.
Aggregation strategies:
    - ANY: Flag if any component is anomalous
    - ALL: Flag only if all components are anomalous
    - VOTING: Flag if N of M components are anomalous
    - WEIGHTED: Weighted average with component importance

Hypothesis: Component-level accumulators provide:
    - Better separation (anomaly in one doesn't mask others)
    - Natural explainability (which component triggered?)
    - Tunable per-component thresholds
"""

import sys
import time
import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from urllib.parse import unquote, parse_qs

import numpy as np

sys.path.insert(0, str(__file__).rsplit("/", 1)[0])  # Add 011-batch to path
sys.path.insert(0, str(__file__).rsplit("/", 4)[0])  # Add repo root to path

from holon import DeterministicVectorManager
from holon.encoder import Encoder


# =============================================================================
# CONFIGURATION
# =============================================================================

GLOBAL_SEED = 42
DIMENSIONS = 4096
DECAY_FACTOR = 0.9995
WARMUP_REQUESTS = 500


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return float(np.dot(a.astype(np.float64), b.astype(np.float64)) / (norm_a * norm_b))


# =============================================================================
# CHARACTER BITMASK (from 010)
# =============================================================================

def char_class_bitmask(s: str) -> int:
    """Compute character class bitmask. Bit 16 = abnormal chars."""
    mask = 0
    normal_special = set("-_./@:,= ")
    for c in s:
        if c.islower():
            mask |= 1
        elif c.isupper():
            mask |= 2
        elif c.isdigit():
            mask |= 4
        elif c in normal_special:
            mask |= 8
        else:
            mask |= 16  # Abnormal
    return mask


def describe_bitmask(mask: int) -> str:
    parts = []
    if mask & 1: parts.append("lower")
    if mask & 2: parts.append("upper")
    if mask & 4: parts.append("digit")
    if mask & 8: parts.append("normal")
    if mask & 16: parts.append("ABNORMAL")
    return "+".join(parts) if parts else "empty"


# =============================================================================
# COMPONENT EXTRACTION
# =============================================================================

def bucket_length(length: int) -> int:
    if length == 0: return 0
    elif length <= 3: return 1
    elif length <= 6: return 2
    elif length <= 12: return 3
    elif length <= 25: return 4
    else: return 5


def bucket_count(count: int) -> int:
    if count == 0: return 0
    elif count <= 2: return 1
    elif count <= 5: return 2
    else: return 3


@dataclass
class RequestComponents:
    """Decomposed request into scoped components."""
    method: dict
    path: dict
    query: dict
    headers: dict
    body: dict

    # Raw data for debugging
    raw_url: str = ""
    raw_method: str = "GET"


def extract_components(
    url: str,
    method: str = "GET",
    headers: Optional[Dict[str, str]] = None,
    body: Optional[Dict[str, Any]] = None,
) -> RequestComponents:
    """
    Extract scoped components from HTTP request.

    Each component becomes a dict that can be independently encoded.
    """
    url = unquote(url)
    headers = headers or {}
    body = body or {}

    # Split URL
    if "?" in url:
        path_part, query_part = url.split("?", 1)
    else:
        path_part, query_part = url, ""

    # --- METHOD component ---
    method_component = {"method": method.upper()}

    # --- PATH component ---
    path_segments = [seg for seg in path_part.split("/") if seg]
    path_bitmasks = [char_class_bitmask(seg) for seg in path_segments]
    path_lengths = [bucket_length(len(seg)) for seg in path_segments]

    path_component = {
        "depth": bucket_count(len(path_segments)),
        "lengths": path_lengths,
        "bitmasks": sorted(set(path_bitmasks)),
        "max_bitmask": max(path_bitmasks) if path_bitmasks else 0,
        "has_parent_ref": ".." in path_part,
        "has_hidden": any(seg.startswith(".") for seg in path_segments),
    }

    # --- QUERY component ---
    query_pairs = []
    if query_part:
        for param in query_part.split("&"):
            if "=" in param:
                k, v = param.split("=", 1)
                query_pairs.append((k, v))
            else:
                query_pairs.append((param, ""))

    query_bitmasks = [char_class_bitmask(v) for k, v in query_pairs]
    query_lengths = [bucket_length(len(v)) for k, v in query_pairs]

    query_component = {
        "count": bucket_count(len(query_pairs)),
        "lengths": query_lengths,
        "bitmasks": sorted(set(query_bitmasks)) if query_bitmasks else [],
        "max_bitmask": max(query_bitmasks) if query_bitmasks else 0,
    }

    # --- HEADERS component ---
    # Track structure, not values (for privacy/cardinality)
    header_names = sorted(headers.keys())
    headers_component = {
        "count": bucket_count(len(headers)),
        "has_auth": "Authorization" in headers or "authorization" in headers,
        "has_content_type": "Content-Type" in headers or "content-type" in headers,
        "has_user_agent": "User-Agent" in headers or "user-agent" in headers,
        "name_lengths": [bucket_length(len(h)) for h in header_names],
    }

    # --- BODY component ---
    # Track structure for JSON bodies
    if body:
        body_keys = sorted(body.keys()) if isinstance(body, dict) else []
        body_bitmasks = []
        for v in (body.values() if isinstance(body, dict) else []):
            if isinstance(v, str):
                body_bitmasks.append(char_class_bitmask(v))

        body_component = {
            "has_body": True,
            "key_count": bucket_count(len(body_keys)),
            "bitmasks": sorted(set(body_bitmasks)) if body_bitmasks else [],
            "max_bitmask": max(body_bitmasks) if body_bitmasks else 0,
        }
    else:
        body_component = {"has_body": False}

    return RequestComponents(
        method=method_component,
        path=path_component,
        query=query_component,
        headers=headers_component,
        body=body_component,
        raw_url=url,
        raw_method=method,
    )


# =============================================================================
# SCOPED ACCUMULATOR
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
# SCOPED DETECTOR
# =============================================================================

@dataclass
class ComponentScore:
    """Score for a single component."""
    name: str
    similarity: float
    threshold: float
    is_anomalous: bool
    features: dict


@dataclass
class ScopedDetectionResult:
    """Full detection result with per-component breakdown."""
    url: str
    method: str
    is_flagged: bool
    aggregation_method: str
    component_scores: List[ComponentScore]
    explanation: str
    is_warmup: bool = False


class ScopedHttpDetector:
    """
    HTTP anomaly detector with per-component accumulators.

    Each component (method, path, query, headers, body) has its own
    accumulator that learns normal patterns independently.
    """

    # Component weights for weighted aggregation
    DEFAULT_WEIGHTS = {
        "method": 0.10,
        "path": 0.35,
        "query": 0.30,
        "headers": 0.10,
        "body": 0.15,
    }

    # Per-component thresholds (can be tuned independently)
    DEFAULT_THRESHOLDS = {
        "method": 0.40,  # Method is low-cardinality, so similarity is always high
        "path": 0.55,
        "query": 0.50,
        "headers": 0.45,
        "body": 0.50,
    }

    def __init__(
        self,
        decay: float = DECAY_FACTOR,
        warmup: int = WARMUP_REQUESTS,
        aggregation: str = "voting",  # any, all, voting, weighted
        voting_threshold: int = 2,  # For voting: flag if >= N anomalous
        weights: Optional[Dict[str, float]] = None,
        thresholds: Optional[Dict[str, float]] = None,
    ):
        self.vm = DeterministicVectorManager(dimensions=DIMENSIONS, global_seed=GLOBAL_SEED)
        self.encoder = Encoder(vector_manager=self.vm)

        self.decay = decay
        self.warmup = warmup
        self.aggregation = aggregation
        self.voting_threshold = voting_threshold
        self.weights = weights or self.DEFAULT_WEIGHTS
        self.thresholds = thresholds or self.DEFAULT_THRESHOLDS

        # Per-component accumulators
        self.accumulators = {
            "method": DecayingAccumulator(DIMENSIONS, decay),
            "path": DecayingAccumulator(DIMENSIONS, decay),
            "query": DecayingAccumulator(DIMENSIONS, decay),
            "headers": DecayingAccumulator(DIMENSIONS, decay),
            "body": DecayingAccumulator(DIMENSIONS, decay),
        }

        self.requests_seen = 0

    def process(
        self,
        url: str,
        method: str = "GET",
        headers: Optional[Dict[str, str]] = None,
        body: Optional[Dict[str, Any]] = None,
    ) -> ScopedDetectionResult:
        """Process request with per-component analysis."""
        self.requests_seen += 1
        is_warmup = self.requests_seen <= self.warmup

        # Extract components
        components = extract_components(url, method, headers, body)

        # Score each component
        component_scores = []
        component_vecs = {}

        for name, features in [
            ("method", components.method),
            ("path", components.path),
            ("query", components.query),
            ("headers", components.headers),
            ("body", components.body),
        ]:
            vec = self.encoder.encode_data(features)
            component_vecs[name] = vec

            model = self.accumulators[name].get_normalized()
            sim = cosine_similarity(vec, model) if self.requests_seen > 1 else 1.0

            threshold = self.thresholds[name]
            is_anomalous = sim < threshold

            component_scores.append(ComponentScore(
                name=name,
                similarity=sim,
                threshold=threshold,
                is_anomalous=is_anomalous,
                features=features,
            ))

        # Aggregate decision
        anomalous_components = [c for c in component_scores if c.is_anomalous]
        n_anomalous = len(anomalous_components)

        if self.aggregation == "any":
            is_flagged = n_anomalous > 0
        elif self.aggregation == "all":
            is_flagged = n_anomalous == len(component_scores)
        elif self.aggregation == "voting":
            is_flagged = n_anomalous >= self.voting_threshold
        elif self.aggregation == "weighted":
            # Weighted average similarity
            weighted_sim = sum(
                c.similarity * self.weights[c.name]
                for c in component_scores
            )
            # Use average threshold as decision boundary
            avg_threshold = sum(
                self.thresholds[name] * self.weights[name]
                for name in self.weights
            )
            is_flagged = weighted_sim < avg_threshold
        else:
            is_flagged = False

        # Don't flag during warmup
        if is_warmup:
            is_flagged = False

        # Generate explanation
        if is_flagged:
            reasons = [f"  - {c.name}: sim={c.similarity:.3f} < {c.threshold:.2f}"
                      for c in anomalous_components]
            explanation = f"FLAGGED ({self.aggregation}): {n_anomalous} anomalous component(s)\n"
            explanation += "\n".join(reasons)
        else:
            explanation = f"ALLOWED: {n_anomalous} anomalous, threshold={self.voting_threshold if self.aggregation == 'voting' else 'N/A'}"

        # Update accumulators
        weight = 0.1 if is_flagged else 1.0
        for name, vec in component_vecs.items():
            self.accumulators[name].update(vec, weight)

        return ScopedDetectionResult(
            url=url,
            method=method,
            is_flagged=is_flagged,
            aggregation_method=self.aggregation,
            component_scores=component_scores,
            explanation=explanation,
            is_warmup=is_warmup,
        )


# =============================================================================
# REQUEST GENERATOR
# =============================================================================

class RequestGenerator:
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

        self.api_patterns = [
            "/api/users", "/api/users/{id}",
            "/api/orders", "/api/orders/{id}",
            "/api/products", "/api/products/{id}",
            "/api/search", "/api/auth/login",
        ]

        self.normal_queries = {
            "limit": ["10", "20", "50"],
            "sort": ["name", "date", "id"],
            "q": ["laptop", "phone", "tablet"],
        }

        self.normal_headers = [
            {"Content-Type": "application/json", "Authorization": "Bearer xxx"},
            {"Content-Type": "application/json"},
            {"Accept": "application/json"},
        ]

    def generate_benign(self) -> Tuple[str, str, dict, dict]:
        """Returns (url, method, headers, body)."""
        pattern = self.rng.choice(self.api_patterns)
        if "{id}" in pattern:
            pattern = pattern.replace("{id}", str(self.rng.randint(1, 999)))

        method = self.rng.choice(["GET", "GET", "GET", "POST"])
        headers = self.rng.choice(self.normal_headers)

        # Maybe add query
        url = pattern
        if self.rng.random() < 0.3:
            key = self.rng.choice(list(self.normal_queries.keys()))
            val = self.rng.choice(self.normal_queries[key])
            url = f"{pattern}?{key}={val}"

        # Maybe add body for POST
        body = {}
        if method == "POST" and self.rng.random() < 0.5:
            body = {"name": "test", "value": str(self.rng.randint(1, 100))}

        return url, method, headers, body

    def generate_malicious(self) -> Tuple[str, str, dict, dict, str]:
        """Returns (url, method, headers, body, attack_type)."""
        attack = self.rng.choice([
            "sqli_path", "sqli_query", "sqli_body",
            "xss_query", "xss_body",
            "traversal", "hidden_file", "cmd_injection",
        ])

        headers = {"User-Agent": "attacker"}
        body = {}

        if attack == "sqli_path":
            payloads = ["' OR '1'='1", "admin'--", "1 UNION SELECT"]
            return f"/api/users/{self.rng.choice(payloads)}", "GET", headers, body, attack

        elif attack == "sqli_query":
            payloads = ["' OR '1'='1", "1; DROP TABLE", "admin' OR '1"]
            return f"/api/search?q={self.rng.choice(payloads)}", "GET", headers, body, attack

        elif attack == "sqli_body":
            payloads = ["admin'--", "' OR '1'='1", "1; DROP TABLE users"]
            body = {"user": self.rng.choice(payloads), "pass": "x"}
            return "/api/auth/login", "POST", headers, body, attack

        elif attack == "xss_query":
            payloads = ["<script>alert(1)</script>", "javascript:alert(1)", "<img onerror=x>"]
            return f"/api/search?q={self.rng.choice(payloads)}", "GET", headers, body, attack

        elif attack == "xss_body":
            payloads = ["<script>alert(1)</script>", "<img src=x onerror=alert(1)>"]
            body = {"comment": self.rng.choice(payloads)}
            return "/api/comments", "POST", headers, body, attack

        elif attack == "traversal":
            payloads = ["../../../etc/passwd", "..\\..\\windows", "....//....//etc"]
            return f"/api/files/{self.rng.choice(payloads)}", "GET", headers, body, attack

        elif attack == "hidden_file":
            paths = ["/.git/config", "/.env", "/.htaccess", "/wp-config.php"]
            return self.rng.choice(paths), "GET", headers, body, attack

        else:  # cmd_injection
            payloads = ["; cat /etc/passwd", "| ls -la", "`whoami`"]
            return f"/api/exec?cmd=test{self.rng.choice(payloads)}", "GET", headers, body, attack

    def generate_stream(self, n: int, malicious_ratio: float = 0.02):
        stream = []
        for _ in range(n):
            if self.rng.random() < malicious_ratio:
                url, method, headers, body, attack = self.generate_malicious()
                stream.append((url, method, headers, body, attack))
            else:
                url, method, headers, body = self.generate_benign()
                stream.append((url, method, headers, body, None))
        return stream


# =============================================================================
# MAIN
# =============================================================================

def evaluate(detector: ScopedHttpDetector, stream: list, name: str):
    """Evaluate detector on stream and print metrics."""
    results = []
    start = time.time()

    for url, method, headers, body, attack in stream:
        result = detector.process(url, method, headers, body)
        results.append((result, attack))

    elapsed = time.time() - start
    throughput = len(stream) / elapsed

    # Post-warmup metrics
    post_warmup = [(r, a) for r, a in results if not r.is_warmup]

    tp = sum(1 for r, a in post_warmup if a is not None and r.is_flagged)
    fp = sum(1 for r, a in post_warmup if a is None and r.is_flagged)
    fn = sum(1 for r, a in post_warmup if a is not None and not r.is_flagged)
    tn = sum(1 for r, a in post_warmup if a is None and not r.is_flagged)

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(0.001, precision + recall)

    print(f"\n--- {name} ---")
    print(f"  Throughput: {throughput:.0f} req/sec")
    print(f"  TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    print(f"  Precision: {precision:.1%}, Recall: {recall:.1%}, F1: {f1:.3f}")

    # By attack type
    attack_stats = {}
    for r, a in post_warmup:
        if a:
            if a not in attack_stats:
                attack_stats[a] = {"detected": 0, "missed": 0}
            if r.is_flagged:
                attack_stats[a]["detected"] += 1
            else:
                attack_stats[a]["missed"] += 1

    print(f"  By attack type:")
    for attack, stats in sorted(attack_stats.items()):
        total = stats["detected"] + stats["missed"]
        rate = stats["detected"] / total if total else 0
        print(f"    {attack}: {stats['detected']}/{total} ({rate:.0%})")

    return f1, precision, recall, throughput


def main():
    print("=" * 80)
    print("Challenge 011-001: Scoped HTTP Vectors")
    print("=" * 80)
    print("""
Per-component accumulators:
  - method_accumulator: learns normal HTTP methods
  - path_accumulator: learns normal path patterns
  - query_accumulator: learns normal query patterns
  - headers_accumulator: learns normal header structures
  - body_accumulator: learns normal body structures

Aggregation strategies:
  - ANY: flag if any component is anomalous
  - ALL: flag only if all components are anomalous
  - VOTING: flag if >= N components are anomalous
  - WEIGHTED: weighted average of component similarities
""")

    # Generate test stream
    generator = RequestGenerator(seed=42)
    stream = generator.generate_stream(10000, malicious_ratio=0.02)

    actual_malicious = sum(1 for _, _, _, _, a in stream if a is not None)
    print(f"Stream: {len(stream)} requests, {actual_malicious} malicious ({100*actual_malicious/len(stream):.1f}%)")

    # Test each aggregation strategy
    results = {}

    for agg in ["any", "voting", "weighted", "all"]:
        detector = ScopedHttpDetector(
            aggregation=agg,
            voting_threshold=2,
            warmup=WARMUP_REQUESTS,
        )
        f1, p, r, tput = evaluate(detector, stream, f"Aggregation: {agg.upper()}")
        results[agg] = {"f1": f1, "precision": p, "recall": r, "throughput": tput}

    # Summary comparison
    print("\n" + "=" * 80)
    print("COMPARISON: Aggregation Strategies")
    print("=" * 80)
    print(f"\n{'Strategy':<15} {'F1':<10} {'Precision':<12} {'Recall':<10} {'Throughput':<12}")
    print("-" * 60)
    for agg, m in results.items():
        print(f"{agg.upper():<15} {m['f1']:<10.3f} {m['precision']:<12.1%} {m['recall']:<10.1%} {m['throughput']:<12.0f}")

    # Show sample detections
    print("\n--- Sample Detections (VOTING) ---")
    detector = ScopedHttpDetector(aggregation="voting", voting_threshold=2)

    # Warmup
    for url, method, headers, body, attack in stream[:WARMUP_REQUESTS]:
        detector.process(url, method, headers, body)

    # Show some malicious
    test_cases = [
        ("/api/users/admin'--", "GET", {}, {}, "sqli_path"),
        ("/api/search?q=<script>alert(1)</script>", "GET", {}, {}, "xss_query"),
        ("/api/auth/login", "POST", {}, {"user": "' OR '1'='1", "pass": "x"}, "sqli_body"),
        ("/api/files/../../../etc/passwd", "GET", {}, {}, "traversal"),
        ("/api/users/123", "GET", {"Content-Type": "application/json"}, {}, None),  # benign
    ]

    for url, method, headers, body, attack in test_cases:
        result = detector.process(url, method, headers, body)
        status = "🚨 FLAGGED" if result.is_flagged else "✅ ALLOWED"
        expected = f"(expected: {'malicious' if attack else 'benign'})"
        print(f"\n{status} {expected}")
        print(f"  URL: {url}")
        print(f"  Body: {body}")
        print(f"  Component scores:")
        for c in result.component_scores:
            flag = "⚠️" if c.is_anomalous else "✓"
            print(f"    {flag} {c.name}: sim={c.similarity:.3f} (thresh={c.threshold:.2f})")

    # Conclusion
    print("\n" + "=" * 80)
    print("LEARNINGS")
    print("=" * 80)

    best = max(results.items(), key=lambda x: x[1]["f1"])
    print(f"""
Best strategy: {best[0].upper()} (F1={best[1]['f1']:.3f})

Key observations:
1. Per-component accumulators allow independent threshold tuning
2. Different components catch different attacks:
   - PATH catches: SQLi in path, traversal, hidden files
   - QUERY catches: SQLi/XSS in query params
   - BODY catches: SQLi/XSS in POST bodies
3. VOTING provides good balance of precision/recall
4. Component-level scoring provides natural explainability

Next: Compare to single-vector baseline from batch 010
""")


if __name__ == "__main__":
    main()
