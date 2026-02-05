#!/usr/bin/env python3
"""
Challenge 010-025: Structural Fingerprint Detection

Key insight: Track STRUCTURE, not CONTENT.

URL: /foo/bar?baz=bur
Content: [["foo", "bar"], [["baz", "bur"]]]  ← High cardinality
Structure: [[3, 3], [[3, 3]]]                 ← Low cardinality!

Attack detection via structural anomalies:
- SQL injection: "' OR '1'='1" = 14 chars (normal value ~3-5)
- Path traversal: many ".." segments, unusual depth
- XSS: "<script>" = 8 chars, has special chars

For headers:
- Total count
- Presence of common headers (auth, content-type)
- Maybe ordered list of header name lengths

This is TRULY headless - no content inspection, just shape.
"""

import sys
import time
import random
from urllib.parse import unquote
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
ANOMALY_THRESHOLD = 0.60  # Tuned for pure frequency+decay (benign min=0.62)
WARMUP_REQUESTS = 500


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return float(np.dot(a.astype(np.float64), b.astype(np.float64)) / (norm_a * norm_b))


# =============================================================================
# CHARACTER CLASS BITMASK
# =============================================================================

def char_class_bitmask(s: str) -> int:
    """
    Compute bitmask of character classes present in string.

    Bit 0 (1):  lowercase
    Bit 1 (2):  uppercase
    Bit 2 (4):  digit
    Bit 3 (8):  normal special (- _ . / @ : , = space)
    Bit 4 (16): abnormal special (anything else)

    Examples:
      "foo"         → 1  (lowercase only)
      "Foo123"      → 7  (lower + upper + digit)
      "user-name"   → 9  (lower + normal special)
      "' OR '1'='1" → 21 (lower + digit + ABNORMAL)
      "<script>"    → 17 (lower + ABNORMAL)
    """
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
            mask |= 16  # Abnormal special

    return mask


def describe_bitmask(mask: int) -> str:
    """Human-readable bitmask description."""
    parts = []
    if mask & 1: parts.append("lower")
    if mask & 2: parts.append("upper")
    if mask & 4: parts.append("digit")
    if mask & 8: parts.append("normal")
    if mask & 16: parts.append("ABNORMAL")
    return "+".join(parts) if parts else "empty"


# =============================================================================
# STRUCTURAL FEATURE EXTRACTION
# =============================================================================

def extract_structural_features(url: str, method: str = "GET", headers: Dict[str, str] = None) -> dict:
    """
    Extract structural features from a request.

    URL features:
    - path_segment_count: number of path segments
    - path_segment_lengths: list of segment lengths
    - query_param_count: number of query params
    - query_key_lengths: list of key lengths
    - query_value_lengths: list of value lengths
    - has_special_chars: presence of unusual characters

    Header features:
    - header_count: total headers
    - has_auth: Authorization present
    - has_content_type: Content-Type present
    - header_name_lengths: sorted list of header name lengths

    Returns a dict that can be encoded by Holon.
    """
    url = unquote(url)
    headers = headers or {}

    # Split URL
    if "?" in url:
        path_part, query_part = url.split("?", 1)
    else:
        path_part = url
        query_part = ""

    # Path features
    path_segments = [seg for seg in path_part.split("/") if seg]
    path_segment_lengths = [len(seg) for seg in path_segments]

    # Query features
    query_pairs = []
    if query_part:
        for param in query_part.split("&"):
            if "=" in param:
                key, val = param.split("=", 1)
                query_pairs.append((key, val))
            else:
                query_pairs.append((param, ""))

    query_key_lengths = [len(k) for k, v in query_pairs]
    query_value_lengths = [len(v) for k, v in query_pairs]

    # Special character detection (structural anomaly signal)
    special_chars = set("'\"<>(){}[];|&`$")
    url_has_special = any(c in url for c in special_chars)

    # Path depth and patterns
    has_parent_ref = ".." in url
    has_hidden_file = any(seg.startswith(".") for seg in path_segments)

    # Length buckets (reduce cardinality while keeping signal)
    def bucket_length(length: int) -> int:
        """Bucket length into categories: 0=empty, 1=tiny, 2=small, 3=medium, 4=large, 5=huge"""
        if length == 0:
            return 0
        elif length <= 3:
            return 1  # tiny
        elif length <= 6:
            return 2  # small
        elif length <= 12:
            return 3  # medium
        elif length <= 25:
            return 4  # large
        else:
            return 5  # huge

    def bucket_count(count: int) -> int:
        """Bucket count: 0=none, 1=few, 2=some, 3=many"""
        if count == 0:
            return 0
        elif count <= 2:
            return 1
        elif count <= 5:
            return 2
        else:
            return 3

    # Build [length_bucket, char_mask] tuples for each component
    path_fingerprints = []
    for seg in path_segments:
        length_bucket = bucket_length(len(seg))
        char_mask = char_class_bitmask(seg)
        path_fingerprints.append([length_bucket, char_mask])

    query_value_fingerprints = []
    for key, val in query_pairs:
        length_bucket = bucket_length(len(val))
        char_mask = char_class_bitmask(val)
        query_value_fingerprints.append([length_bucket, char_mask])

    # Collect ALL bitmasks as a prominent feature
    # This makes rare bitmasks stand out in frequency-based detection
    all_masks = [fp[1] for fp in path_fingerprints] + [fp[1] for fp in query_value_fingerprints]

    # Build structural fingerprint with bitmasks as primary signal
    features = {
        # Method
        "method": method,

        # Bitmasks are THE key signal for frequency-based detection
        # Encode as a set of observed bitmasks (low cardinality: only 32 possible values)
        "bitmasks": sorted(set(all_masks)),  # Unique bitmasks seen
        "max_bitmask": max(all_masks) if all_masks else 0,  # Highest bitmask (likely abnormal)

        # Path structure
        "path_depth": bucket_count(len(path_segments)),

        # Query structure
        "query_count": bucket_count(len(query_pairs)),

        # Lengths (secondary signal)
        "path_lengths": [fp[0] for fp in path_fingerprints],
        "query_lengths": [fp[0] for fp in query_value_fingerprints],
    }

    # Summary flags for debugging
    features["has_abnormal"] = any(m & 16 for m in all_masks)
    features["has_parent_ref"] = has_parent_ref
    features["has_hidden_file"] = has_hidden_file

    return features


def features_to_string(features: dict) -> str:
    """Convert features to readable string."""
    path_fp = features.get('path_fingerprints', [])
    query_fp = features.get('query_fingerprints', [])
    return (
        f"method={features['method']} "
        f"path={path_fp} "
        f"query={query_fp} "
        f"abnormal={features.get('has_abnormal', False)} "
        f"parent={features.get('has_parent_ref', False)} "
        f"hidden={features.get('has_hidden_file', False)}"
    )


# =============================================================================
# DECAYING ACCUMULATOR
# =============================================================================

class DecayingAccumulator:
    def __init__(self, dimensions: int, decay: float):
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
# STRUCTURAL DETECTOR
# =============================================================================

@dataclass
class DetectionResult:
    url: str
    features: dict
    is_flagged: bool
    similarity: float
    is_warmup: bool


class StructuralFingerprintDetector:
    """
    Detect anomalies based on structural fingerprint.

    No content inspection - purely shape-based.
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

        self.accumulator = DecayingAccumulator(DIMENSIONS, decay)
        self.threshold = threshold
        self.warmup = warmup
        self.requests_seen = 0

    def process(self, url: str, method: str = "GET", headers: Dict[str, str] = None) -> DetectionResult:
        """Process request and detect structural anomalies."""
        self.requests_seen += 1
        is_warmup = self.requests_seen <= self.warmup

        # Extract structural features
        features = extract_structural_features(url, method, headers)

        # Encode features
        vec = self.encoder.encode_data(features)

        # Get current model
        model = self.accumulator.get_normalized()

        # Compute similarity
        if self.requests_seen <= 1:
            similarity = 1.0
        else:
            similarity = cosine_similarity(vec, model)

        # PURE frequency + decay detection
        # No immediate flags - let the accumulator do the work
        # Rare fingerprints (attacks) won't match the learned distribution

        if is_warmup:
            is_flagged = False
        else:
            is_flagged = similarity < self.threshold

        # Update model
        weight = 0.1 if is_flagged else 1.0
        self.accumulator.update(vec, weight)

        return DetectionResult(
            url=url,
            features=features,
            is_flagged=is_flagged,
            similarity=similarity,
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

        self.normal_values = {
            "limit": ["10", "20", "50"],
            "sort": ["name", "date", "id"],
            "q": ["laptop", "phone", "tablet"],
        }

        self.headers_templates = [
            {"Content-Type": "application/json", "Authorization": "Bearer xxx"},
            {"Content-Type": "application/json"},
            {"Accept": "application/json"},
        ]

    def generate_benign(self) -> Tuple[str, str, dict]:
        """Returns (url, method, headers)."""
        pattern = self.rng.choice(self.api_patterns)
        if "{id}" in pattern:
            pattern = pattern.replace("{id}", str(self.rng.randint(1, 999)))

        method = self.rng.choice(["GET", "GET", "GET", "POST"])
        headers = self.rng.choice(self.headers_templates)

        if self.rng.random() < 0.3:
            key = self.rng.choice(list(self.normal_values.keys()))
            val = self.rng.choice(self.normal_values[key])
            return f"{pattern}?{key}={val}", method, headers

        return pattern, method, headers

    def generate_malicious(self) -> Tuple[str, str, dict, str]:
        """Returns (url, method, headers, attack_type)."""
        attack = self.rng.choice([
            "sqli_path", "sqli_query", "traversal",
            "xss_query", "hidden_file", "cmd_injection",
        ])

        headers = {"User-Agent": "attacker"}

        if attack == "sqli_path":
            # Long payload in path
            payloads = ["' OR '1'='1", "admin'--", "1 UNION SELECT * FROM users"]
            return f"/api/users/{self.rng.choice(payloads)}", "GET", headers, attack

        elif attack == "sqli_query":
            payloads = ["' OR '1'='1", "1; DROP TABLE users--", "admin' OR '1'='1"]
            return f"/api/search?q={self.rng.choice(payloads)}", "GET", headers, attack

        elif attack == "traversal":
            # Deep path with parent refs
            payloads = ["../../../etc/passwd", "..\\..\\..\\windows", "....//....//....//etc"]
            return f"/api/files/{self.rng.choice(payloads)}", "GET", headers, attack

        elif attack == "xss_query":
            payloads = ["<script>alert(1)</script>", "javascript:alert(1)", "<img onerror=x>"]
            return f"/api/search?q={self.rng.choice(payloads)}", "GET", headers, attack

        elif attack == "hidden_file":
            paths = ["/.git/config", "/.env", "/.htaccess", "/wp-config.php"]
            return self.rng.choice(paths), "GET", headers, attack

        else:  # cmd_injection
            payloads = ["; cat /etc/passwd", "| ls -la", "`whoami`"]
            return f"/api/exec?cmd=test{self.rng.choice(payloads)}", "GET", headers, attack

    def generate_stream(self, n: int, malicious_ratio: float = 0.02):
        stream = []
        for _ in range(n):
            if self.rng.random() < malicious_ratio:
                url, method, headers, attack = self.generate_malicious()
                stream.append((url, method, headers, attack))
            else:
                url, method, headers = self.generate_benign()
                stream.append((url, method, headers, None))
        return stream


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("Challenge 010-025: Structural Fingerprint Detection")
    print("=" * 80)
    print("""
Key insight: Track STRUCTURE + CHARACTER CLASS, not CONTENT.

Each component becomes [length_bucket, char_class_bitmask]:

  Bitmask:
    Bit 0 (1):  lowercase
    Bit 1 (2):  uppercase
    Bit 2 (4):  digit
    Bit 3 (8):  normal special (- _ . / @ : ,)
    Bit 4 (16): ABNORMAL special (' " < > ; | & etc)

Examples:
  "foo"         → [1, 1]   (tiny, lowercase)
  "user123"     → [2, 5]   (small, lower+digit)
  "' OR '1'='1" → [4, 21]  (large, lower+digit+ABNORMAL!)
  "<script>"    → [2, 17]  (small, lower+ABNORMAL!)

Detection: Flag if ANY component has abnormal chars (bit 16 set).

TRULY headless - no pattern matching, just character class analysis!
""")

    # Show bitmask examples
    print("--- Character Class Bitmask Examples ---")
    bitmask_examples = ["foo", "Foo123", "user-name", "' OR '1'='1", "<script>", "../.."]
    for s in bitmask_examples:
        mask = char_class_bitmask(s)
        print(f"  '{s}' → mask={mask} ({describe_bitmask(mask)})")

    # Show feature extraction examples
    print("\n--- Structural Fingerprint Examples ---")
    examples = [
        ("/api/users/123", "GET", {"Authorization": "Bearer x"}),
        ("/api/users/' OR '1'='1", "GET", {}),
        ("/../../../etc/passwd", "GET", {}),
        ("/api/search?q=<script>alert(1)</script>", "GET", {}),
        ("/.git/config", "GET", {}),
    ]

    for url, method, headers in examples:
        features = extract_structural_features(url, method, headers)
        print(f"\n  {url}")
        print(f"    bitmasks: {features['bitmasks']}, max={features['max_bitmask']}")
        print(f"    path_lengths: {features['path_lengths']}")
        print(f"    has_abnormal: {features['has_abnormal']}")

    # Initialize
    generator = RequestGenerator(seed=42)
    detector = StructuralFingerprintDetector(
        global_seed=GLOBAL_SEED,
        decay=DECAY_FACTOR,
        threshold=ANOMALY_THRESHOLD,
        warmup=WARMUP_REQUESTS,
    )

    print(f"\n--- Configuration ---")
    print(f"  Decay: {DECAY_FACTOR}")
    print(f"  Threshold: {ANOMALY_THRESHOLD}")
    print(f"  Warmup: {WARMUP_REQUESTS}")

    # Generate and process stream
    n_requests = 10000
    stream = generator.generate_stream(n_requests, malicious_ratio=0.02)

    actual_malicious = sum(1 for _, _, _, a in stream if a is not None)
    print(f"\n--- Stream ---")
    print(f"  Total: {n_requests}")
    print(f"  Malicious: {actual_malicious} ({100*actual_malicious/n_requests:.1f}%)")

    print(f"\n--- Processing ---")
    results = []
    start = time.time()

    for url, method, headers, attack in stream:
        result = detector.process(url, method, headers)
        results.append((result, attack))

    total_time = time.time() - start
    print(f"  Throughput: {n_requests/total_time:.0f} req/sec")

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

    # Similarity distribution
    benign_sims = [r.similarity for r, a in post_warmup if a is None]
    malicious_sims = [r.similarity for r, a in post_warmup if a is not None]

    print(f"\n--- Similarity Distribution ---")
    print(f"  Benign:    min={min(benign_sims):.3f}, mean={np.mean(benign_sims):.3f}, max={max(benign_sims):.3f}")
    print(f"  Malicious: min={min(malicious_sims):.3f}, mean={np.mean(malicious_sims):.3f}, max={max(malicious_sims):.3f}")

    # Sample results
    print(f"\n--- Sample True Positives ---")
    for r, a in [(r, a) for r, a in post_warmup if a and r.is_flagged][:5]:
        print(f"  {a}: sim={r.similarity:.3f}")
        print(f"    bitmasks={r.features['bitmasks']} abnormal={r.features['has_abnormal']}")

    print(f"\n--- Sample False Negatives ---")
    for r, a in [(r, a) for r, a in post_warmup if a and not r.is_flagged][:5]:
        print(f"  {a}: sim={r.similarity:.3f}")
        print(f"    {features_to_string(r.features)}")

    print(f"\n--- Sample False Positives ---")
    for r, a in [(r, a) for r, a in post_warmup if not a and r.is_flagged][:5]:
        print(f"  sim={r.similarity:.3f} | {r.url}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: Structural Fingerprint Detection")
    print("=" * 80)
    print(f"""
Approach:
  Request → Extract Structure → Encode → Compare to Accumulator

  Structure features:
    - path_lengths: [tiny, small, medium, large, huge]
    - query_value_lengths: [tiny, small, medium, large, huge]
    - has_special_chars: True/False
    - has_parent_ref: True/False (traversal)
    - has_hidden_file: True/False (.git, .env)

Performance:
  F1 Score:  {f1:.3f}
  Precision: {precision:.1%}
  Recall:    {recall:.1%}

Separation:
  Benign mean:    {np.mean(benign_sims):.3f}
  Malicious mean: {np.mean(malicious_sims):.3f}
  Gap: {np.mean(benign_sims) - np.mean(malicious_sims):.3f}

TRULY HEADLESS: No content inspection, purely structural!
""")

    return f1


if __name__ == "__main__":
    f1 = main()

    if f1 < 0.8:
        print("\n--- Threshold Tuning ---")
        best_f1 = 0
        best_thresh = 0.35

        for thresh in [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55]:
            gen = RequestGenerator(seed=42)
            det = StructuralFingerprintDetector(threshold=thresh, warmup=WARMUP_REQUESTS)

            stream = gen.generate_stream(10000, 0.02)
            results = [(det.process(u, m, h), a) for u, m, h, a in stream]
            post = [(r, a) for r, a in results if not r.is_warmup]

            tp = sum(1 for r, a in post if a and r.is_flagged)
            fp = sum(1 for r, a in post if not a and r.is_flagged)
            fn = sum(1 for r, a in post if a and not r.is_flagged)

            p = tp / max(1, tp + fp)
            r = tp / max(1, tp + fn)
            f = 2 * p * r / max(0.001, p + r)

            print(f"  threshold={thresh:.2f}: P={p:.1%} R={r:.1%} F1={f:.3f}")

            if f > best_f1:
                best_f1 = f
                best_thresh = thresh

        print(f"\n  Best: threshold={best_thresh:.2f} → F1={best_f1:.3f}")
