#!/usr/bin/env python3
"""
Challenge 010-022: Continuous HTTP Anomaly Detection

Same continuous learning approach as PCAP, but for HTTP requests.

Architecture:
- Decaying accumulator for passive learning
- Rule-based boost for known attack patterns
- Similarity threshold for unknown anomalies
- Concept drift adaptation

Key features:
- No batch training required
- Learns "normal" from traffic passively
- Adapts to new API endpoints automatically
- Detects SQL injection, XSS, path traversal, etc.
"""

import sys
import time
import random
import re
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

sys.path.insert(0, str(__file__).rsplit("/", 4)[0])

from holon import DeterministicVectorManager
from holon.encoder import Encoder


# =============================================================================
# CONFIGURATION
# =============================================================================

GLOBAL_SEED = 42
DIMENSIONS = 4096
DECAY_FACTOR = 0.9995  # ~2000 request memory
ANOMALY_THRESHOLD = 0.35
WARMUP_REQUESTS = 500


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return float(np.dot(a.astype(np.float64), b.astype(np.float64)) / (norm_a * norm_b))


# =============================================================================
# DECAYING ACCUMULATOR
# =============================================================================

class DecayingAccumulator:
    """Accumulator with exponential decay for continuous learning."""

    def __init__(self, dimensions: int, decay: float = DECAY_FACTOR):
        self.dimensions = dimensions
        self.decay = decay
        self.accumulator = np.zeros(dimensions, dtype=np.float64)
        self.count = 0

    def update(self, vector: np.ndarray, weight: float = 1.0):
        """Add new observation with decay."""
        self.accumulator = self.decay * self.accumulator + weight * vector.astype(np.float64)
        self.count += 1

    def get_normalized(self) -> np.ndarray:
        """Get unit-normalized representation."""
        norm = np.linalg.norm(self.accumulator)
        if norm < 1e-10:
            return np.zeros(self.dimensions, dtype=np.float32)
        return (self.accumulator / norm).astype(np.float32)

    def get_effective_window(self) -> float:
        return 1.0 / (1.0 - self.decay)


# =============================================================================
# ATTACK PATTERNS (for rule-based detection)
# =============================================================================

# SQL injection patterns
SQL_PATTERNS = [
    r"['\"].*(?:OR|AND).*['\"].*=.*['\"]",
    r"(?:UNION|SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER)\s",
    r"(?:--|#|;)\s*$",
    r"'.*--",
    r"\b(?:1|true)\s*=\s*(?:1|true)\b",
]

# XSS patterns
XSS_PATTERNS = [
    r"<\s*script",
    r"javascript\s*:",
    r"on(?:error|load|click|mouse)\s*=",
    r"<\s*img[^>]+onerror",
    r"<\s*iframe",
]

# Path traversal patterns
TRAVERSAL_PATTERNS = [
    r"\.\./",
    r"\.\.\\",
    r"%2e%2e[/%]",
    r"etc/passwd",
    r"windows/system32",
]

# Suspicious paths
SUSPICIOUS_PATHS = {
    "/.git/config",
    "/.env",
    "/.htaccess",
    "/wp-config.php",
    "/admin",
    "/phpmyadmin",
    "/.aws/credentials",
    "/etc/passwd",
}


def check_sql_injection(text: str) -> bool:
    """Check for SQL injection patterns."""
    text_lower = text.lower()
    for pattern in SQL_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return True
    return False


def check_xss(text: str) -> bool:
    """Check for XSS patterns."""
    text_lower = text.lower()
    for pattern in XSS_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return True
    return False


def check_traversal(text: str) -> bool:
    """Check for path traversal patterns."""
    text_lower = text.lower()
    for pattern in TRAVERSAL_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return True
    return False


# =============================================================================
# HTTP REQUEST GENERATOR
# =============================================================================

class HTTPRequestGenerator:
    """Generate realistic HTTP request traffic."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

        # Common benign patterns
        self.api_endpoints = [
            "/api/users",
            "/api/users/{id}",
            "/api/orders",
            "/api/orders/{id}",
            "/api/products",
            "/api/products/{id}",
            "/api/auth/login",
            "/api/auth/logout",
            "/api/auth/refresh",
            "/api/search",
            "/api/settings",
            "/api/notifications",
        ]

        self.static_paths = [
            "/static/js/app.js",
            "/static/css/style.css",
            "/static/images/logo.png",
            "/favicon.ico",
        ]

        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) Safari/605.1",
            "Mozilla/5.0 (X11; Linux x86_64) Firefox/120.0",
        ]

    def generate_benign(self) -> dict:
        """Generate a benign HTTP request."""
        pattern = self.rng.choices(
            ["api_get", "api_post", "static", "search"],
            weights=[0.4, 0.3, 0.2, 0.1]
        )[0]

        if pattern == "api_get":
            endpoint = self.rng.choice(self.api_endpoints)
            # Replace {id} with actual ID
            if "{id}" in endpoint:
                endpoint = endpoint.replace("{id}", str(self.rng.randint(1, 10000)))

            return {
                "method": "GET",
                "path": endpoint,
                "headers": {
                    "Content-Type": "application/json",
                    "Authorization": "Bearer token123",
                    "User-Agent": self.rng.choice(self.user_agents),
                },
                "query": {},
                "body": None,
            }

        elif pattern == "api_post":
            endpoint = self.rng.choice([
                "/api/users",
                "/api/orders",
                "/api/auth/login",
            ])

            if endpoint == "/api/auth/login":
                body = {"username": f"user{self.rng.randint(1, 100)}", "password": "***"}
            elif endpoint == "/api/users":
                body = {"name": f"User {self.rng.randint(1, 100)}", "email": f"user{self.rng.randint(1, 100)}@example.com"}
            else:
                body = {"items": [self.rng.randint(1, 100) for _ in range(self.rng.randint(1, 5))]}

            return {
                "method": "POST",
                "path": endpoint,
                "headers": {
                    "Content-Type": "application/json",
                    "Authorization": "Bearer token123",
                    "User-Agent": self.rng.choice(self.user_agents),
                },
                "query": {},
                "body": body,
            }

        elif pattern == "static":
            return {
                "method": "GET",
                "path": self.rng.choice(self.static_paths),
                "headers": {
                    "User-Agent": self.rng.choice(self.user_agents),
                },
                "query": {},
                "body": None,
            }

        else:  # search
            return {
                "method": "GET",
                "path": "/api/search",
                "headers": {
                    "Content-Type": "application/json",
                    "User-Agent": self.rng.choice(self.user_agents),
                },
                "query": {"q": f"product{self.rng.randint(1, 100)}"},
                "body": None,
            }

    def generate_malicious(self) -> dict:
        """Generate a malicious HTTP request."""
        attack = self.rng.choice([
            "sqli_path",
            "sqli_body",
            "xss_query",
            "xss_body",
            "traversal",
            "hidden_files",
            "unusual_method",
        ])

        if attack == "sqli_path":
            return {
                "method": "GET",
                "path": f"/api/users/' OR '1'='1",
                "headers": {"User-Agent": self.rng.choice(self.user_agents)},
                "query": {},
                "body": None,
                "attack_type": "sqli_path",
            }

        elif attack == "sqli_body":
            return {
                "method": "POST",
                "path": "/api/auth/login",
                "headers": {
                    "Content-Type": "application/json",
                    "User-Agent": self.rng.choice(self.user_agents),
                },
                "query": {},
                "body": {"username": "admin'--", "password": "x"},
                "attack_type": "sqli_body",
            }

        elif attack == "xss_query":
            return {
                "method": "GET",
                "path": "/api/search",
                "headers": {"User-Agent": self.rng.choice(self.user_agents)},
                "query": {"q": "<script>alert('xss')</script>"},
                "body": None,
                "attack_type": "xss_query",
            }

        elif attack == "xss_body":
            return {
                "method": "POST",
                "path": "/api/users",
                "headers": {
                    "Content-Type": "application/json",
                    "User-Agent": self.rng.choice(self.user_agents),
                },
                "query": {},
                "body": {"name": "<img onerror=alert(1)>", "email": "x@x.com"},
                "attack_type": "xss_body",
            }

        elif attack == "traversal":
            return {
                "method": "GET",
                "path": "/api/files/../../../etc/passwd",
                "headers": {"User-Agent": self.rng.choice(self.user_agents)},
                "query": {},
                "body": None,
                "attack_type": "traversal",
            }

        elif attack == "hidden_files":
            return {
                "method": "GET",
                "path": self.rng.choice(list(SUSPICIOUS_PATHS)),
                "headers": {"User-Agent": self.rng.choice(self.user_agents)},
                "query": {},
                "body": None,
                "attack_type": "hidden_files",
            }

        else:  # unusual_method
            return {
                "method": self.rng.choice(["TRACE", "DEBUG", "TRACK", "OPTIONS"]),
                "path": "/api/users",
                "headers": {"User-Agent": self.rng.choice(self.user_agents)},
                "query": {},
                "body": None,
                "attack_type": "unusual_method",
            }

    def generate_stream(self, n_requests: int, malicious_ratio: float = 0.02) -> List[Tuple[dict, bool]]:
        """Generate a stream of requests."""
        stream = []
        for _ in range(n_requests):
            if self.rng.random() < malicious_ratio:
                stream.append((self.generate_malicious(), True))
            else:
                stream.append((self.generate_benign(), False))
        return stream


# =============================================================================
# CONTINUOUS HTTP DETECTOR
# =============================================================================

@dataclass
class DetectionResult:
    request: dict
    is_flagged: bool
    similarity: float
    rule_triggered: Optional[str]
    is_warmup: bool


class ContinuousHTTPDetector:
    """
    Continuous HTTP anomaly detection with passive learning.

    Combines:
    1. Decaying accumulator (learns normal patterns)
    2. Rule-based detection (known attack patterns)
    3. Similarity threshold (unknown anomalies)
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
        self.flagged_count = 0

    def process(self, request: dict) -> DetectionResult:
        """Process a single HTTP request."""
        self.requests_seen += 1
        is_warmup = self.requests_seen <= self.warmup

        # Normalize and encode
        normalized = self._normalize_request(request)
        vec = self.encoder.encode_data(normalized)

        # Get current model
        model = self.accumulator.get_normalized()

        # Compute similarity
        if self.requests_seen <= 1:
            similarity = 1.0
        else:
            similarity = cosine_similarity(vec, model)

        # Check rules
        rule_triggered = self._check_rules(request)

        # Decision
        if is_warmup:
            is_flagged = False
        elif rule_triggered:
            is_flagged = True
        else:
            is_flagged = similarity < self.threshold

        if is_flagged:
            self.flagged_count += 1

        # Update model (passive learning)
        if not is_flagged:
            self.accumulator.update(vec)
        else:
            # Flagged requests update with reduced weight
            self.accumulator.update(vec, weight=0.1)

        return DetectionResult(
            request=request,
            is_flagged=is_flagged,
            similarity=similarity,
            rule_triggered=rule_triggered,
            is_warmup=is_warmup,
        )

    def _normalize_request(self, request: dict) -> dict:
        """Normalize request for encoding."""
        normalized = {}

        # Method
        normalized["method"] = request.get("method", "GET")

        # Normalize path (replace IDs with placeholders)
        path = request.get("path", "/")
        # Replace numeric IDs
        path_normalized = re.sub(r'/\d+', '/{id}', path)
        # Replace UUIDs
        path_normalized = re.sub(r'/[a-f0-9-]{36}', '/{uuid}', path_normalized)
        normalized["path_pattern"] = path_normalized

        # Path prefix
        parts = path.split("/")
        if len(parts) > 2:
            normalized["path_prefix"] = "/".join(parts[:3])
        else:
            normalized["path_prefix"] = path

        # Headers (just presence, not values)
        headers = request.get("headers", {})
        normalized["has_auth"] = "Authorization" in headers
        normalized["content_type"] = headers.get("Content-Type", "none")

        # Body structure (not values)
        body = request.get("body")
        if body is None:
            normalized["body_type"] = "none"
        elif isinstance(body, dict):
            normalized["body_type"] = "object"
            normalized["body_keys"] = sorted(body.keys())
        else:
            normalized["body_type"] = "other"

        # Query params (just keys)
        query = request.get("query", {})
        if query:
            normalized["query_keys"] = sorted(query.keys())

        return normalized

    def _check_rules(self, request: dict) -> Optional[str]:
        """Check rule-based attack patterns."""
        path = request.get("path", "")
        method = request.get("method", "")
        body = request.get("body", {})
        query = request.get("query", {})

        # Check path
        if check_sql_injection(path):
            return "sqli_path"
        if check_traversal(path):
            return "traversal"
        if path.lower() in {p.lower() for p in SUSPICIOUS_PATHS}:
            return "hidden_files"

        # Check body values
        if isinstance(body, dict):
            for val in body.values():
                if isinstance(val, str):
                    if check_sql_injection(val):
                        return "sqli_body"
                    if check_xss(val):
                        return "xss_body"

        # Check query values
        for val in query.values():
            if isinstance(val, str):
                if check_sql_injection(val):
                    return "sqli_query"
                if check_xss(val):
                    return "xss_query"

        # Check method
        if method.upper() in {"TRACE", "TRACK", "DEBUG"}:
            return "unusual_method"

        return None


# =============================================================================
# MAIN DEMO
# =============================================================================

def main():
    print("=" * 80)
    print("Challenge 010-022: Continuous HTTP Anomaly Detection")
    print("=" * 80)
    print("""
Continuous learning approach for HTTP requests:
- Decaying accumulator learns "normal" passively
- Rule-based detection for known attack patterns
- Similarity threshold for unknown anomalies
- Adapts to new API endpoints automatically
""")

    # Initialize
    generator = HTTPRequestGenerator(seed=42)
    detector = ContinuousHTTPDetector(
        global_seed=GLOBAL_SEED,
        decay=DECAY_FACTOR,
        threshold=ANOMALY_THRESHOLD,
        warmup=WARMUP_REQUESTS,
    )

    print(f"Configuration:")
    print(f"  Decay factor: {DECAY_FACTOR}")
    print(f"  Effective window: ~{detector.accumulator.get_effective_window():.0f} requests")
    print(f"  Anomaly threshold: {ANOMALY_THRESHOLD}")
    print(f"  Warmup period: {WARMUP_REQUESTS} requests")

    # Generate stream
    n_requests = 10000
    malicious_ratio = 0.02
    stream = generator.generate_stream(n_requests, malicious_ratio)

    actual_malicious = sum(1 for _, m in stream if m)
    actual_benign = n_requests - actual_malicious

    print(f"\n--- Stream Stats ---")
    print(f"  Total requests: {n_requests}")
    print(f"  Benign: {actual_benign} ({100*actual_benign/n_requests:.1f}%)")
    print(f"  Malicious: {actual_malicious} ({100*actual_malicious/n_requests:.1f}%)")

    # Process stream
    print(f"\n--- Processing Stream ---")

    results = []
    start = time.time()

    for request, is_malicious in stream:
        result = detector.process(request)
        results.append((result, is_malicious))

    total_time = time.time() - start

    print(f"  Processed {n_requests} requests in {total_time:.2f}s")
    print(f"  Throughput: {n_requests/total_time:.0f} req/sec")

    # Compute metrics (excluding warmup)
    post_warmup = [(r, m) for r, m in results if not r.is_warmup]

    tp = sum(1 for r, m in post_warmup if m and r.is_flagged)
    fp = sum(1 for r, m in post_warmup if not m and r.is_flagged)
    fn = sum(1 for r, m in post_warmup if m and not r.is_flagged)
    tn = sum(1 for r, m in post_warmup if not m and not r.is_flagged)

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(0.001, precision + recall)

    print(f"\n--- Results (post-warmup) ---")
    print(f"  Confusion Matrix:")
    print(f"    TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    print(f"\n  Metrics:")
    print(f"    Precision: {precision:.1%}")
    print(f"    Recall:    {recall:.1%}")
    print(f"    F1 Score:  {f1:.3f}")

    # Detection by attack type
    print(f"\n--- Detection by Attack Type ---")
    attack_types = {}
    for r, m in post_warmup:
        if m:
            attack_type = r.request.get("attack_type", "unknown")
            if attack_type not in attack_types:
                attack_types[attack_type] = {"detected": 0, "missed": 0, "rule": 0}
            if r.is_flagged:
                attack_types[attack_type]["detected"] += 1
                if r.rule_triggered:
                    attack_types[attack_type]["rule"] += 1
            else:
                attack_types[attack_type]["missed"] += 1

    for attack_type, counts in sorted(attack_types.items()):
        total = counts["detected"] + counts["missed"]
        rate = counts["detected"] / total if total > 0 else 0
        rule_pct = counts["rule"] / max(1, counts["detected"])
        print(f"  {attack_type}: {counts['detected']}/{total} ({rate:.0%}) | {rule_pct:.0%} by rules")

    # Detection method breakdown
    rule_detections = sum(1 for r, m in post_warmup if r.is_flagged and r.rule_triggered)
    similarity_detections = sum(1 for r, m in post_warmup if r.is_flagged and not r.rule_triggered)

    print(f"\n--- Detection Method ---")
    print(f"  Rule-based: {rule_detections}")
    print(f"  Similarity: {similarity_detections}")

    # Similarity distribution
    benign_sims = [r.similarity for r, m in post_warmup if not m]
    malicious_sims = [r.similarity for r, m in post_warmup if m]

    print(f"\n--- Similarity Distribution ---")
    print(f"  Benign:    min={min(benign_sims):.3f}, mean={np.mean(benign_sims):.3f}, max={max(benign_sims):.3f}")
    print(f"  Malicious: min={min(malicious_sims):.3f}, mean={np.mean(malicious_sims):.3f}, max={max(malicious_sims):.3f}")

    # Sample flagged requests
    print(f"\n--- Sample Flagged Requests ---")
    flagged = [(r, m) for r, m in post_warmup if r.is_flagged][:10]
    for r, is_mal in flagged:
        status = "MALICIOUS" if is_mal else "FALSE POS"
        attack = r.request.get("attack_type", "n/a")
        rule = r.rule_triggered or "similarity"
        print(f"  [{status}] {r.request.get('method')} {r.request.get('path')[:40]} | {rule}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: Continuous HTTP Detection")
    print("=" * 80)
    print(f"""
Architecture:
  ┌─────────────────┐
  │ HTTP Request    │
  │    Stream       │
  └────────┬────────┘
           │
           ▼
  ┌─────────────────────────────────────────┐
  │      Continuous HTTP Detector           │
  │                                         │
  │  1. Normalize request (path patterns)   │
  │  2. Check rules (SQLi, XSS, traversal)  │
  │  3. Compare to decaying accumulator     │
  │  4. Update accumulator (passive learn)  │
  │                                         │
  │  accum = {DECAY_FACTOR} * accum + vec             │
  └─────────────────────────────────────────┘
           │
           ▼
  ┌─────────────────┐
  │ ALLOW / BLOCK   │
  └─────────────────┘

Performance:
  F1 Score:    {f1:.3f}
  Precision:   {precision:.1%}
  Recall:      {recall:.1%}
  Throughput:  {n_requests/total_time:.0f} req/sec

Detection:
  Rule-based:  {rule_detections} ({100*rule_detections/max(1,tp+fp):.0f}%)
  Similarity:  {similarity_detections} ({100*similarity_detections/max(1,tp+fp):.0f}%)

Key Features:
  ✓ Continuous learning (no batch training)
  ✓ Decaying memory (adapts to traffic changes)
  ✓ Hybrid detection (rules + similarity)
  ✓ Path normalization (reduces cardinality)
""")


def test_concept_drift():
    """Test adaptation to new API endpoints."""
    print("\n" + "=" * 80)
    print("BONUS: Concept Drift Test - New API Endpoint")
    print("=" * 80)
    print("""
Scenario:
  Phase 1: Traffic to existing endpoints (/api/users, /api/orders)
  Phase 2: New endpoint deployed (/api/recommendations)

Question: Does the detector adapt without retraining?
""")

    detector = ContinuousHTTPDetector(
        global_seed=GLOBAL_SEED,
        decay=0.999,  # Faster adaptation
        threshold=0.35,
        warmup=200,
    )

    rng = random.Random(456)

    # Phase 1: Existing endpoints
    print("Phase 1: Existing API endpoints...")
    phase1_flagged = 0
    for i in range(5000):
        request = {
            "method": rng.choice(["GET", "POST"]),
            "path": rng.choice(["/api/users", f"/api/users/{rng.randint(1,100)}", "/api/orders"]),
            "headers": {"Content-Type": "application/json", "Authorization": "Bearer x"},
            "query": {},
            "body": {"data": "x"} if rng.random() > 0.5 else None,
        }
        result = detector.process(request)
        if result.is_flagged and not result.is_warmup:
            phase1_flagged += 1

    print(f"  Flagged: {phase1_flagged} / 4800")

    # Phase 2: New endpoint
    print("\nPhase 2: New endpoint /api/recommendations deployed...")
    phase2_flagged = 0
    phase2_sims = []

    for i in range(5000):
        request = {
            "method": "GET",
            "path": f"/api/recommendations/{rng.randint(1,100)}",
            "headers": {"Content-Type": "application/json", "Authorization": "Bearer x"},
            "query": {"limit": "10"},
            "body": None,
        }
        result = detector.process(request)
        phase2_sims.append(result.similarity)
        if result.is_flagged:
            phase2_flagged += 1

    print(f"  Flagged: {phase2_flagged} / 5000")

    # Check adaptation
    early_sims = phase2_sims[:500]
    late_sims = phase2_sims[-500:]

    print(f"\n  Similarity during transition:")
    print(f"    First 500 /api/recommendations:  mean={np.mean(early_sims):.3f}")
    print(f"    Last 500 /api/recommendations:   mean={np.mean(late_sims):.3f}")

    improvement = np.mean(late_sims) - np.mean(early_sims)

    if improvement > 0.05:
        print(f"\n  ✓ ADAPTATION: Similarity improved by {improvement:.3f}")
        print("    New endpoint learned as normal traffic!")
    else:
        print(f"\n  ⚠ Limited adaptation: {improvement:.3f}")

    print(f"""
Result:
  Phase 1: {phase1_flagged} false positives on known endpoints
  Phase 2: {phase2_flagged} flagged on new endpoint (should decrease over time)
  Adaptation: New endpoint similarity improved from {np.mean(early_sims):.3f} to {np.mean(late_sims):.3f}
""")


if __name__ == "__main__":
    main()
    test_concept_drift()
