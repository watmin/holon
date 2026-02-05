#!/usr/bin/env python3
"""
Challenge 010-006: Frequency-Based Anomaly Detection

Hypothesis: If training data is predominantly benign with rare malicious,
the prototype will capture the DOMINANT benign pattern. Rare malicious
samples get "outvoted" in the bundling/prototype operation.

Key insight: VSA prototype does MAJORITY VOTING across dimensions.
If 99% of training vectors agree on a dimension, the prototype will
reflect that consensus. Rare anomalies don't affect the prototype.
"""

import sys
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(__file__).rsplit("/", 4)[0])

from holon import DeterministicVectorManager
from holon.encoder import Encoder


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def normalize_path(path: str) -> str:
    """Normalize RESTful paths."""
    import re
    if not isinstance(path, str):
        return str(path)
    result = re.sub(r'/\d+', '/{id}', path)
    return result


class FrequencyBasedDetector:
    """
    Anomaly detection using frequency-weighted prototypes.

    The key insight: prototype() does majority voting.
    If 99% of vectors share a pattern, the prototype captures it.
    Rare deviations don't affect the prototype.
    """

    def __init__(self, encoder: Encoder, relevant_fields: List[str]):
        self.encoder = encoder
        self.relevant_fields = relevant_fields
        self.prototype = None
        self.training_sims = []  # Similarities of training data to prototype
        self.threshold = None

    def _extract_relevant(self, record: dict) -> dict:
        """Extract only relevant fields for encoding."""
        return {f: record.get(f, "") for f in self.relevant_fields}

    def train(self, records: List[dict], contamination: float = 0.01):
        """
        Train on data that may contain some anomalies.

        Args:
            records: Training records (mostly benign, some malicious)
            contamination: Expected fraction of anomalies (for threshold setting)
        """
        print(f"Training on {len(records)} records...")

        # Encode all records
        vectors = []
        for r in records:
            relevant = self._extract_relevant(r)
            vec = self.encoder.encode_data(relevant)
            vectors.append(vec)

        # Build prototype (majority voting)
        # Using a lower threshold to capture more signal
        self.prototype = self.encoder.prototype(vectors, threshold=0.3)

        # Calculate similarities of training data to prototype
        self.training_sims = [cosine(v, self.prototype) for v in vectors]

        # Set threshold based on expected contamination
        # If 1% are anomalies, threshold at 1st percentile
        self.threshold = np.percentile(self.training_sims, contamination * 100)

        print(f"Prototype built")
        print(f"Training similarity: min={min(self.training_sims):.4f}, "
              f"max={max(self.training_sims):.4f}, mean={np.mean(self.training_sims):.4f}")
        print(f"Threshold set at {contamination*100:.1f}th percentile: {self.threshold:.4f}")

    def check(self, record: dict) -> Tuple[bool, float]:
        """
        Check if a record is anomalous.

        Returns: (is_anomaly, similarity_to_prototype)
        """
        relevant = self._extract_relevant(record)
        vec = self.encoder.encode_data(relevant)
        sim = cosine(vec, self.prototype)

        is_anomaly = sim < self.threshold
        return is_anomaly, sim


def generate_mixed_traffic(
    n_benign: int = 10000,
    n_malicious: int = 100,
    seed: int = 42,
) -> Tuple[List[dict], List[bool]]:
    """
    Generate mixed traffic with mostly benign and some malicious.

    Returns: (records, is_malicious_flags)
    """
    import random
    random.seed(seed)

    records = []
    labels = []  # True = malicious

    # Benign patterns (realistic API traffic)
    benign_patterns = [
        # Common CRUD operations
        {"method": "GET", "path": "/api/users/{id}"},
        {"method": "GET", "path": "/api/users"},
        {"method": "POST", "path": "/api/users"},
        {"method": "PUT", "path": "/api/users/{id}"},
        {"method": "DELETE", "path": "/api/users/{id}"},
        {"method": "GET", "path": "/api/orders/{id}"},
        {"method": "GET", "path": "/api/orders"},
        {"method": "POST", "path": "/api/orders"},
        {"method": "GET", "path": "/api/products/{id}"},
        {"method": "GET", "path": "/api/products"},
        {"method": "GET", "path": "/api/search"},
        {"method": "POST", "path": "/api/auth/login"},
        {"method": "POST", "path": "/api/auth/logout"},
    ]

    # Malicious patterns
    malicious_patterns = [
        # Path traversal
        {"method": "GET", "path": "/api/../../../etc/passwd"},
        {"method": "GET", "path": "/api/users/../../admin"},
        # SQL injection
        {"method": "GET", "path": "/api/users/' OR 1=1--"},
        {"method": "POST", "path": "/api/users/1; DROP TABLE users;--"},
        # Command injection
        {"method": "GET", "path": "/api/exec?cmd=ls"},
        {"method": "POST", "path": "/api/run?script=rm -rf /"},
        # Unusual methods
        {"method": "TRACE", "path": "/api/users"},
        {"method": "CONNECT", "path": "/api/proxy"},
        # Encoded attacks
        {"method": "GET", "path": "/api/%2e%2e/etc/passwd"},
        # Unusual paths
        {"method": "GET", "path": "/admin/config"},
        {"method": "GET", "path": "/.git/config"},
        {"method": "GET", "path": "/api/internal/debug"},
    ]

    # Generate benign traffic (with realistic distribution)
    # Some endpoints are more popular than others
    benign_weights = [20, 10, 5, 3, 2, 15, 8, 4, 12, 6, 10, 3, 2]

    for _ in range(n_benign):
        pattern = random.choices(benign_patterns, weights=benign_weights)[0]
        records.append(pattern.copy())
        labels.append(False)

    # Generate malicious traffic
    for _ in range(n_malicious):
        pattern = random.choice(malicious_patterns)
        records.append(pattern.copy())
        labels.append(True)

    # Shuffle
    combined = list(zip(records, labels))
    random.shuffle(combined)
    records, labels = zip(*combined)

    return list(records), list(labels)


def main():
    print("=" * 70)
    print("Challenge 010-006: Frequency-Based Anomaly Detection")
    print("=" * 70)

    # Setup
    vm = DeterministicVectorManager(dimensions=4096, global_seed=42)
    encoder = Encoder(vector_manager=vm)

    # Generate mixed traffic
    print("\n--- Generating Mixed Traffic ---")

    # Training: 10k benign + 100 malicious (1% contamination)
    train_records, train_labels = generate_mixed_traffic(
        n_benign=10000, n_malicious=100, seed=42
    )

    benign_count = sum(1 for l in train_labels if not l)
    malicious_count = sum(1 for l in train_labels if l)
    print(f"Training data: {benign_count} benign, {malicious_count} malicious "
          f"({100*malicious_count/(benign_count+malicious_count):.1f}% contamination)")

    # Test: 1k benign + 50 malicious
    test_records, test_labels = generate_mixed_traffic(
        n_benign=1000, n_malicious=50, seed=999
    )

    benign_test = sum(1 for l in test_labels if not l)
    malicious_test = sum(1 for l in test_labels if l)
    print(f"Test data: {benign_test} benign, {malicious_test} malicious")

    # Train detector
    print("\n--- Training Detector ---")
    detector = FrequencyBasedDetector(
        encoder=encoder,
        relevant_fields=["method", "path"],
    )
    detector.train(train_records, contamination=0.02)  # Expect ~2% anomalies

    # Analyze training similarity distribution
    print("\n--- Training Data Analysis ---")

    # Check which training samples have low similarity (should be the malicious ones)
    train_results = []
    for record, is_malicious in zip(train_records, train_labels):
        is_anomaly, sim = detector.check(record)
        train_results.append((record, is_malicious, is_anomaly, sim))

    # Sort by similarity
    train_results.sort(key=lambda x: x[3])

    print("\nLowest similarity samples (should be malicious):")
    for record, is_malicious, is_anomaly, sim in train_results[:10]:
        actual = "MALICIOUS" if is_malicious else "benign"
        detected = "FLAGGED" if is_anomaly else "passed"
        print(f"  sim={sim:+.4f} | {actual:10} | {detected} | {record['method']:6} {record['path'][:40]}")

    print("\nHighest similarity samples (should be benign):")
    for record, is_malicious, is_anomaly, sim in train_results[-10:]:
        actual = "MALICIOUS" if is_malicious else "benign"
        detected = "FLAGGED" if is_anomaly else "passed"
        print(f"  sim={sim:+.4f} | {actual:10} | {detected} | {record['method']:6} {record['path'][:40]}")

    # Test set evaluation
    print("\n--- Test Set Evaluation ---")

    true_positives = 0  # Malicious correctly flagged
    false_positives = 0  # Benign incorrectly flagged
    true_negatives = 0  # Benign correctly passed
    false_negatives = 0  # Malicious incorrectly passed

    test_results = []
    for record, is_malicious in zip(test_records, test_labels):
        is_anomaly, sim = detector.check(record)
        test_results.append((record, is_malicious, is_anomaly, sim))

        if is_malicious and is_anomaly:
            true_positives += 1
        elif is_malicious and not is_anomaly:
            false_negatives += 1
        elif not is_malicious and is_anomaly:
            false_positives += 1
        else:
            true_negatives += 1

    # Metrics
    precision = true_positives / max(1, true_positives + false_positives)
    recall = true_positives / max(1, true_positives + false_negatives)
    f1 = 2 * precision * recall / max(0.001, precision + recall)

    print(f"\nConfusion Matrix:")
    print(f"                    Predicted")
    print(f"                    Benign    Malicious")
    print(f"  Actual Benign     {true_negatives:6}    {false_positives:6}")
    print(f"  Actual Malicious  {false_negatives:6}    {true_positives:6}")

    print(f"\nMetrics:")
    print(f"  Precision: {precision:.1%} (of flagged, how many were actually malicious)")
    print(f"  Recall:    {recall:.1%} (of malicious, how many were caught)")
    print(f"  F1 Score:  {f1:.3f}")

    # Show similarity distributions
    print("\n--- Similarity Distributions ---")
    benign_sims = [sim for _, is_mal, _, sim in test_results if not is_mal]
    malicious_sims = [sim for _, is_mal, _, sim in test_results if is_mal]

    print(f"Benign:    min={min(benign_sims):.4f}, max={max(benign_sims):.4f}, mean={np.mean(benign_sims):.4f}")
    print(f"Malicious: min={min(malicious_sims):.4f}, max={max(malicious_sims):.4f}, mean={np.mean(malicious_sims):.4f}")
    print(f"Threshold: {detector.threshold:.4f}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
Approach: Frequency-based prototype with threshold detection

Key insight:
  VSA prototype does MAJORITY VOTING across dimensions.
  With 99% benign traffic, the prototype captures the benign pattern.
  Rare malicious samples don't affect the prototype significantly.

Results:
  Training: {benign_count} benign + {malicious_count} malicious ({100*malicious_count/(benign_count+malicious_count):.1f}% contamination)
  Test:     {benign_test} benign + {malicious_test} malicious

  Precision: {precision:.1%}
  Recall:    {recall:.1%}
  F1 Score:  {f1:.3f}

Observations:
  - Benign similarity range: [{min(benign_sims):.4f}, {max(benign_sims):.4f}]
  - Malicious similarity range: [{min(malicious_sims):.4f}, {max(malicious_sims):.4f}]
  - Separation: {np.mean(benign_sims) - np.mean(malicious_sims):.4f} (benign - malicious mean)
""")


if __name__ == "__main__":
    main()
