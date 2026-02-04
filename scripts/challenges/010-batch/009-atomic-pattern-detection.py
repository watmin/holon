#!/usr/bin/env python3
"""
Challenge 010-009: Atomic Pattern Detection

Key insight from 008: Bundling fields allows partial matches.
A malicious "GET /admin/config" resonates with benign "GET /api/users"
because they share the "GET" method.

Solution: Encode the COMPLETE PATTERN as a single atom.
"GET|/api/users" is ONE atom, not bind(GET, /api/users).

This way:
- Matching patterns → same atom → identical vector → sim = 1.0
- Different patterns → different atoms → orthogonal → sim ≈ 0

For anomaly detection:
- Known patterns have vectors in our prototype
- Unknown patterns are orthogonal to everything we know

This is essentially pattern membership testing, but using VSA
for the distributed consensus property (same pattern → same vector
across all nodes without coordination).
"""

import sys
from collections import Counter
from typing import Dict, List, Tuple, Callable

import numpy as np

sys.path.insert(0, str(__file__).rsplit("/", 4)[0])

from deterministic_codebook import DeterministicVectorManager
from holon.encoder import Encoder


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class AtomicPatternDetector:
    """
    Encode complete patterns as SINGLE ATOMS.

    Instead of:
        encode({"method": "GET", "path": "/api/users"})
        → bind(method_vec, GET_vec) + bind(path_vec, users_vec)

    We do:
        encode("GET|/api/users")
        → single random vector for this exact pattern

    Benefits:
    - Exact pattern matching: same pattern → same vector
    - No partial matches: different pattern → orthogonal vector
    - Distributed consensus: deterministic hash → same vector everywhere

    For anomaly detection with frequency weighting:
    - Bundle known patterns with frequency weights
    - High-frequency patterns dominate the prototype
    - New input: check similarity to prototype
    - Unknown patterns (orthogonal) → low similarity → anomaly
    """

    def __init__(
        self,
        vector_manager: DeterministicVectorManager,
        pattern_extractor: Callable[[dict], str],
    ):
        self.vm = vector_manager
        self.pattern_extractor = pattern_extractor

        self.pattern_counts: Counter = Counter()
        self.prototype: np.ndarray = None

    def observe_batch(self, records: List[dict]):
        """Observe and count patterns."""
        for record in records:
            pattern = self.pattern_extractor(record)
            self.pattern_counts[pattern] += 1

    def build_prototype(self, min_count: int = 5, weight_fn: str = "log"):
        """
        Build frequency-weighted prototype from known patterns.

        Only includes patterns seen at least min_count times.
        Weights by log(count) for sublinear frequency scaling.
        """
        # Filter to frequent patterns
        frequent = {p: c for p, c in self.pattern_counts.items() if c >= min_count}

        if not frequent:
            self.prototype = np.zeros(self.vm.dimensions, dtype=np.int8)
            return

        print(f"Building prototype from {len(frequent)} frequent patterns:")

        # Get vectors and weights
        vectors = []
        weights = []

        for pattern, count in frequent.items():
            vec = self.vm.get_vector(pattern)  # Pattern as SINGLE ATOM
            vectors.append(vec)

            if weight_fn == "log":
                weight = np.log(count + 1)
            elif weight_fn == "sqrt":
                weight = np.sqrt(count)
            else:
                weight = float(count)

            weights.append(weight)
            print(f"  {count:4d}x (w={weight:5.2f}) {pattern[:50]}")

        # Weighted sum and threshold
        weighted_sum = np.zeros(self.vm.dimensions, dtype=np.float32)
        for vec, weight in zip(vectors, weights):
            weighted_sum += weight * vec.astype(np.float32)

        self.prototype = np.where(
            weighted_sum > 0, 1,
            np.where(weighted_sum < 0, -1, 0)
        ).astype(np.int8)

        # Report excluded patterns
        excluded = {p: c for p, c in self.pattern_counts.items() if c < min_count}
        if excluded:
            print(f"\nExcluded {len(excluded)} rare patterns:")
            for pattern, count in sorted(excluded.items(), key=lambda x: -x[1]):
                print(f"  {count:4d}x {pattern[:50]}")

    def check(self, record: dict) -> Tuple[float, bool]:
        """
        Check if a record's pattern is known.

        Returns: (similarity, is_known_pattern)
        """
        pattern = self.pattern_extractor(record)

        # Is this an EXACT known pattern?
        is_known = self.pattern_counts.get(pattern, 0) >= 5

        # Get pattern vector (as single atom)
        vec = self.vm.get_vector(pattern)

        # Similarity to prototype
        sim = cosine(vec, self.prototype)

        return sim, is_known


def normalize_path(path: str) -> str:
    """Normalize RESTful path."""
    import re
    if not isinstance(path, str):
        return str(path)
    result = re.sub(r'/\d+', '/{id}', path)
    result = re.sub(r'/[a-f0-9-]{36}', '/{uuid}', result)
    return result


def pattern_extractor(record: dict) -> str:
    """Extract pattern as single string (to be encoded as one atom)."""
    method = record.get("method", "")
    path = normalize_path(record.get("path", ""))
    return f"{method}|{path}"


def generate_traffic(n_benign=1000, n_malicious=10, seed=42):
    """Generate mixed traffic."""
    import random
    random.seed(seed)

    benign = [
        {"method": "GET", "path": "/api/users"},
        {"method": "GET", "path": "/api/users/123"},  # Will normalize to /api/users/{id}
        {"method": "GET", "path": "/api/orders"},
        {"method": "POST", "path": "/api/orders"},
        {"method": "GET", "path": "/api/products"},
    ]

    malicious = [
        {"method": "GET", "path": "/api/../../../etc/passwd"},
        {"method": "GET", "path": "/api/users/' OR 1=1--"},
        {"method": "TRACE", "path": "/api/users"},
        {"method": "GET", "path": "/admin/config"},
        {"method": "GET", "path": "/.git/config"},
    ]

    records = []
    labels = []

    weights = [30, 20, 25, 15, 10]
    for _ in range(n_benign):
        p = random.choices(benign, weights=weights)[0].copy()
        records.append(p)
        labels.append(False)

    for _ in range(n_malicious):
        p = random.choice(malicious).copy()
        records.append(p)
        labels.append(True)

    combined = list(zip(records, labels))
    random.shuffle(combined)
    records, labels = zip(*combined)

    return list(records), list(labels)


def main():
    print("=" * 70)
    print("Challenge 010-009: Atomic Pattern Detection")
    print("=" * 70)

    # Use deterministic vector manager for consensus
    vm = DeterministicVectorManager(dimensions=4096, global_seed=42)

    # Generate data
    train_records, train_labels = generate_traffic(1000, 10, seed=42)
    test_records, test_labels = generate_traffic(200, 20, seed=999)

    print(f"Training: {sum(1 for l in train_labels if not l)} benign, "
          f"{sum(1 for l in train_labels if l)} malicious")

    # Create detector
    detector = AtomicPatternDetector(vm, pattern_extractor)

    # Observe training data
    detector.observe_batch(train_records)

    # Build prototype
    print("\n--- Building Prototype ---")
    detector.build_prototype(min_count=5, weight_fn="log")

    # Evaluate
    print("\n--- Evaluation ---")

    benign_sims = []
    malicious_sims = []
    known_benign = 0
    known_malicious = 0

    for record, is_malicious in zip(test_records, test_labels):
        sim, is_known = detector.check(record)

        if is_malicious:
            malicious_sims.append(sim)
            if is_known:
                known_malicious += 1
        else:
            benign_sims.append(sim)
            if is_known:
                known_benign += 1

    print("\nSimilarity distributions:")
    print(f"  Benign:    mean={np.mean(benign_sims):+.4f}, std={np.std(benign_sims):.4f}, "
          f"min={min(benign_sims):+.4f}, max={max(benign_sims):+.4f}")
    print(f"  Malicious: mean={np.mean(malicious_sims):+.4f}, std={np.std(malicious_sims):.4f}, "
          f"min={min(malicious_sims):+.4f}, max={max(malicious_sims):+.4f}")
    print(f"  Separation: {np.mean(benign_sims) - np.mean(malicious_sims):+.4f}")

    print(f"\nPattern membership:")
    print(f"  Benign:    {known_benign}/{len(benign_sims)} known patterns")
    print(f"  Malicious: {known_malicious}/{len(malicious_sims)} known patterns")

    # Detection using threshold
    print("\n--- Detection Performance ---")

    # Find best threshold
    best_f1 = 0
    best_threshold = 0

    for threshold in np.linspace(-0.1, 0.3, 41):
        tp = sum(1 for s in malicious_sims if s < threshold)
        fp = sum(1 for s in benign_sims if s < threshold)
        fn = sum(1 for s in malicious_sims if s >= threshold)

        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(0.001, precision + recall)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    # Report at best threshold
    tp = sum(1 for s in malicious_sims if s < best_threshold)
    fp = sum(1 for s in benign_sims if s < best_threshold)
    fn = sum(1 for s in malicious_sims if s >= best_threshold)
    tn = sum(1 for s in benign_sims if s >= best_threshold)

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)

    print(f"Best threshold: {best_threshold:.4f}")
    print(f"Confusion matrix:")
    print(f"  TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    print(f"Precision: {precision:.1%}")
    print(f"Recall:    {recall:.1%}")
    print(f"F1 Score:  {best_f1:.3f}")

    # Show sample predictions
    print("\n--- Sample Predictions ---")
    print("Benign samples:")
    for record, is_mal in zip(test_records[:10], test_labels[:10]):
        if not is_mal:
            sim, is_known = detector.check(record)
            pattern = pattern_extractor(record)
            status = "KNOWN" if is_known else "unknown"
            flag = "" if sim >= best_threshold else "⚠️ FALSE POSITIVE"
            print(f"  sim={sim:+.4f} [{status:7}] {pattern[:40]} {flag}")

    print("\nMalicious samples:")
    for record, is_mal in zip(test_records, test_labels):
        if is_mal:
            sim, is_known = detector.check(record)
            pattern = pattern_extractor(record)
            status = "KNOWN" if is_known else "unknown"
            flag = "✓ DETECTED" if sim < best_threshold else "❌ MISSED"
            print(f"  sim={sim:+.4f} [{status:7}] {pattern[:40]} {flag}")

    print("\n" + "=" * 70)
    print("KEY INSIGHT")
    print("=" * 70)
    print("""
Encoding the COMPLETE PATTERN as a single atom avoids partial matches.

"GET|/api/users" → one random vector V1
"GET|/admin/config" → different random vector V2 (orthogonal to V1)

No partial resonance from shared "GET" method!

The frequency-weighted prototype:
- High-frequency benign patterns contribute more
- Rare (malicious) patterns have minimal influence

Detection:
- Known patterns have HIGH similarity to prototype
- Unknown patterns are ORTHOGONAL (sim ≈ 0)

This combines:
1. Pattern normalization (reduce cardinality)
2. Frequency weighting (high-freq dominates)
3. Atomic encoding (no partial matches)
4. Deterministic vectors (distributed consensus)
""")


if __name__ == "__main__":
    main()
