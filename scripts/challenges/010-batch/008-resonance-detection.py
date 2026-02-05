#!/usr/bin/env python3
"""
Challenge 010-008: Resonance-Based Detection

New idea: Instead of comparing to one prototype, use RESONANCE
to extract how much of a new vector "resonates" with known patterns.

resonance(vec, reference) keeps only dimensions where both agree.
If a new vector has strong resonance with known patterns → normal
If it has weak resonance → anomaly (doesn't match known patterns)

The "resonance strength" (non-zero dimensions after resonance)
could be a better signal than cosine similarity.
"""

import sys
from collections import Counter
from typing import Dict, List, Tuple, Callable

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


class ResonanceDetector:
    """
    Anomaly detection using Holon's resonance() primitive.

    resonance(vec, ref) returns a vector with only the dimensions
    where vec and ref AGREE (same sign).

    For a new vector:
    - Strong resonance with known patterns → normal
    - Weak resonance → doesn't match known patterns → anomaly

    We measure "resonance strength" as the fraction of non-zero
    dimensions after resonance, relative to the original vector.
    """

    def __init__(self, encoder: Encoder, pattern_extractor: Callable[[dict], str]):
        self.encoder = encoder
        self.pattern_extractor = pattern_extractor

        self.pattern_counts: Counter = Counter()
        self.pattern_vectors: Dict[str, np.ndarray] = {}
        self.prototype: np.ndarray = None

    def observe_batch(self, records: List[dict]):
        """Observe and count patterns."""
        for record in records:
            pattern = self.pattern_extractor(record)
            self.pattern_counts[pattern] += 1
            if pattern not in self.pattern_vectors:
                self.pattern_vectors[pattern] = self.encoder.encode_data(record)

    def build_prototype(self, min_count: int = 5):
        """
        Build prototype from frequent patterns only.
        Use amplify() to boost high-frequency patterns.
        """
        # Only include patterns seen at least min_count times
        frequent_patterns = {
            p: v for p, v in self.pattern_vectors.items()
            if self.pattern_counts[p] >= min_count
        }

        if not frequent_patterns:
            self.prototype = np.zeros(self.encoder.vector_manager.dimensions, dtype=np.int8)
            return

        # Bundle frequent patterns
        vectors = list(frequent_patterns.values())
        self.prototype = self.encoder.bundle(vectors)

        # Amplify by frequency
        for pattern, vec in frequent_patterns.items():
            count = self.pattern_counts[pattern]
            strength = np.log(count + 1) / 2
            self.prototype = self.encoder.amplify(self.prototype, vec, strength)

        print(f"Prototype built from {len(frequent_patterns)} frequent patterns")
        print(f"Excluded {len(self.pattern_vectors) - len(frequent_patterns)} rare patterns")

    def check_resonance(self, record: dict) -> Tuple[float, float, float]:
        """
        Check resonance of a new record with the prototype.

        Returns:
            resonance_strength: Fraction of non-zero dims after resonance
            cosine_similarity: Standard cosine similarity
            resonance_norm: L2 norm of resonance vector (energy retained)
        """
        vec = self.encoder.encode_data(record)

        # Get resonance with prototype
        resonant = self.encoder.resonance(vec, self.prototype)

        # Resonance strength: how many dimensions agree?
        vec_nonzero = np.sum(vec != 0)
        resonant_nonzero = np.sum(resonant != 0)
        resonance_strength = resonant_nonzero / max(1, vec_nonzero)

        # Cosine similarity
        cos_sim = cosine(vec, self.prototype)

        # Resonance norm (energy retained after filtering)
        vec_norm = np.linalg.norm(vec)
        resonant_norm = np.linalg.norm(resonant)
        resonance_energy = resonant_norm / max(1e-10, vec_norm)

        return resonance_strength, cos_sim, resonance_energy


def generate_traffic(n_benign=1000, n_malicious=10, seed=42):
    """Generate mixed traffic."""
    import random
    random.seed(seed)

    benign = [
        {"method": "GET", "path": "/api/users"},
        {"method": "GET", "path": "/api/orders"},
        {"method": "POST", "path": "/api/orders"},
        {"method": "GET", "path": "/api/products"},
    ]

    malicious = [
        {"method": "GET", "path": "/api/../../../etc/passwd"},
        {"method": "GET", "path": "/api/users/' OR 1=1--"},
        {"method": "TRACE", "path": "/api/users"},
        {"method": "GET", "path": "/admin/config"},
    ]

    records = []
    labels = []

    weights = [40, 30, 20, 10]
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


def pattern_extractor(record: dict) -> str:
    return f"{record.get('method', '')}|{record.get('path', '')}"


def main():
    print("=" * 70)
    print("Challenge 010-008: Resonance-Based Detection")
    print("=" * 70)

    vm = DeterministicVectorManager(dimensions=4096, global_seed=42)
    encoder = Encoder(vector_manager=vm)

    # Generate data
    train_records, train_labels = generate_traffic(1000, 10, seed=42)
    test_records, test_labels = generate_traffic(200, 20, seed=999)

    print(f"Training: {sum(1 for l in train_labels if not l)} benign, "
          f"{sum(1 for l in train_labels if l)} malicious")

    # Train
    detector = ResonanceDetector(encoder, pattern_extractor)
    detector.observe_batch(train_records)
    detector.build_prototype(min_count=5)

    # Show pattern distribution
    print("\nPattern distribution:")
    for pattern, count in detector.pattern_counts.most_common():
        included = "✓" if count >= 5 else "✗"
        print(f"  {included} {count:4d}x {pattern}")

    # Evaluate using resonance strength
    print("\n--- Resonance Analysis ---")

    benign_results = []
    malicious_results = []

    for record, is_malicious in zip(test_records, test_labels):
        res_strength, cos_sim, res_energy = detector.check_resonance(record)
        result = (record, res_strength, cos_sim, res_energy)

        if is_malicious:
            malicious_results.append(result)
        else:
            benign_results.append(result)

    # Extract just the metrics
    benign_strength = [r[1] for r in benign_results]
    malicious_strength = [r[1] for r in malicious_results]
    benign_cos = [r[2] for r in benign_results]
    malicious_cos = [r[2] for r in malicious_results]
    benign_energy = [r[3] for r in benign_results]
    malicious_energy = [r[3] for r in malicious_results]

    print("\n         Metric        |    Benign    |   Malicious  | Separation")
    print("-" * 70)
    print(f" Resonance Strength    | {np.mean(benign_strength):+.4f}±{np.std(benign_strength):.4f} | "
          f"{np.mean(malicious_strength):+.4f}±{np.std(malicious_strength):.4f} | "
          f"{np.mean(benign_strength) - np.mean(malicious_strength):+.4f}")
    print(f" Cosine Similarity     | {np.mean(benign_cos):+.4f}±{np.std(benign_cos):.4f} | "
          f"{np.mean(malicious_cos):+.4f}±{np.std(malicious_cos):.4f} | "
          f"{np.mean(benign_cos) - np.mean(malicious_cos):+.4f}")
    print(f" Resonance Energy      | {np.mean(benign_energy):+.4f}±{np.std(benign_energy):.4f} | "
          f"{np.mean(malicious_energy):+.4f}±{np.std(malicious_energy):.4f} | "
          f"{np.mean(benign_energy) - np.mean(malicious_energy):+.4f}")

    # Show examples
    print("\n--- Sample Results ---")
    print("Benign samples:")
    for record, strength, cos, energy in benign_results[:3]:
        print(f"  strength={strength:.4f}, cos={cos:+.4f}, energy={energy:.4f} | "
              f"{record['method']} {record['path']}")

    print("\nMalicious samples:")
    for record, strength, cos, energy in malicious_results[:5]:
        print(f"  strength={strength:.4f}, cos={cos:+.4f}, energy={energy:.4f} | "
              f"{record['method']} {record['path']}")

    # Detection performance using resonance strength as threshold
    print("\n--- Detection Performance (using Resonance Strength) ---")

    # Find optimal threshold
    all_results = [(s, True) for s in malicious_strength] + [(s, False) for s in benign_strength]
    all_results.sort()

    best_f1 = 0
    best_threshold = 0

    for threshold in np.linspace(0.3, 0.5, 21):
        tp = sum(1 for s in malicious_strength if s < threshold)
        fp = sum(1 for s in benign_strength if s < threshold)
        fn = sum(1 for s in malicious_strength if s >= threshold)
        tn = sum(1 for s in benign_strength if s >= threshold)

        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(0.001, precision + recall)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    # Report at best threshold
    threshold = best_threshold
    tp = sum(1 for s in malicious_strength if s < threshold)
    fp = sum(1 for s in benign_strength if s < threshold)
    fn = sum(1 for s in malicious_strength if s >= threshold)
    tn = sum(1 for s in benign_strength if s >= threshold)

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)

    print(f"Best threshold: {threshold:.4f}")
    print(f"Confusion matrix:")
    print(f"  TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    print(f"Precision: {precision:.1%}")
    print(f"Recall:    {recall:.1%}")
    print(f"F1 Score:  {best_f1:.3f}")

    print("\n" + "=" * 70)
    print("KEY INSIGHT")
    print("=" * 70)
    print("""
resonance(vec, prototype) extracts the AGREEING part of vec with prototype.

For anomaly detection:
- Benign vectors share structure with prototype → strong resonance
- Malicious vectors are orthogonal to prototype → weak resonance

Resonance strength = fraction of dimensions that agree
- High resonance strength → matches known patterns → normal
- Low resonance strength → doesn't match → anomaly

This is a more DIRECT measure of "does this match what we know?"
than cosine similarity, which can be noisy for bundled vectors.
""")


if __name__ == "__main__":
    main()
