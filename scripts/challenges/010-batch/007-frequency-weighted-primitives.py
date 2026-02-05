#!/usr/bin/env python3
"""
Challenge 010-007: Frequency-Weighted Primitives

Use Holon's primitives to weight high-frequency observations
and decay/deprioritize low-frequency ones.

Key primitives:
- mathematical_bundle(vectors, weights) - weighted sum
- amplify(superposition, component, strength) - boost component
- prototype_add(proto, example, count) - incremental update

New concept: Frequency-weighted prototype
- High frequency patterns get more weight
- Low frequency patterns decay
- Rare anomalies don't affect the baseline
"""

import sys
from collections import Counter, defaultdict
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


class FrequencyWeightedPrototype:
    """
    Build prototypes using frequency-weighted bundling.

    Instead of treating all observations equally, we:
    1. Group by pattern (normalized representation)
    2. Weight each pattern by frequency
    3. Use mathematical_bundle with weights

    Weight functions:
    - linear: weight = count
    - log: weight = log(count + 1)  # Sublinear, diminishing returns
    - sqrt: weight = sqrt(count)     # Moderate decay
    - threshold: weight = 1 if count > k else 0  # Binary cutoff
    """

    def __init__(
        self,
        encoder: Encoder,
        pattern_extractor: Callable[[dict], str],
        weight_function: str = "log",  # linear, log, sqrt, threshold
        threshold_k: int = 5,  # For threshold function
    ):
        self.encoder = encoder
        self.pattern_extractor = pattern_extractor
        self.weight_function = weight_function
        self.threshold_k = threshold_k

        # Pattern tracking
        self.pattern_counts: Counter = Counter()
        self.pattern_vectors: Dict[str, np.ndarray] = {}

        # Computed prototype
        self.prototype: np.ndarray = None

    def _compute_weight(self, count: int) -> float:
        """Compute weight from frequency count."""
        if self.weight_function == "linear":
            return float(count)
        elif self.weight_function == "log":
            return np.log(count + 1)
        elif self.weight_function == "sqrt":
            return np.sqrt(count)
        elif self.weight_function == "threshold":
            return 1.0 if count >= self.threshold_k else 0.0
        else:
            raise ValueError(f"Unknown weight function: {self.weight_function}")

    def observe(self, record: dict):
        """Observe a record and update pattern counts."""
        pattern = self.pattern_extractor(record)
        self.pattern_counts[pattern] += 1

        # Store vector for first occurrence
        if pattern not in self.pattern_vectors:
            self.pattern_vectors[pattern] = self.encoder.encode_data(record)

    def observe_batch(self, records: List[dict]):
        """Observe a batch of records."""
        for record in records:
            self.observe(record)

    def build_prototype(self):
        """
        Build frequency-weighted prototype using mathematical_bundle.

        High-frequency patterns dominate.
        Low-frequency patterns (potential anomalies) have minimal impact.
        """
        if not self.pattern_vectors:
            self.prototype = np.zeros(self.encoder.vector_manager.dimensions, dtype=np.int8)
            return

        vectors = []
        weights = []

        for pattern, count in self.pattern_counts.items():
            if pattern in self.pattern_vectors:
                weight = self._compute_weight(count)
                if weight > 0:  # Skip zero-weight patterns
                    vectors.append(self.pattern_vectors[pattern])
                    weights.append(weight)

        if not vectors:
            self.prototype = np.zeros(self.encoder.vector_manager.dimensions, dtype=np.int8)
            return

        # Use Holon's mathematical_bundle for weighted combination
        self.prototype = self.encoder.mathematical_bundle(vectors, weights)

        return self.prototype

    def similarity(self, record: dict) -> float:
        """Compute similarity of a record to the prototype."""
        vec = self.encoder.encode_data(record)
        return cosine(vec, self.prototype)

    def get_stats(self) -> Dict:
        """Get statistics about observed patterns."""
        total = sum(self.pattern_counts.values())
        return {
            "unique_patterns": len(self.pattern_counts),
            "total_observations": total,
            "top_patterns": self.pattern_counts.most_common(10),
            "weight_function": self.weight_function,
        }


class AmplifyBasedDetector:
    """
    Alternative approach: Start with uniform prototype, then
    AMPLIFY frequently-seen patterns and NEGATE rare ones.

    This uses Holon's amplify() and negate() primitives directly.
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

    def build_prototype(self, amplify_threshold: int = 10, negate_threshold: int = 2):
        """
        Build prototype by:
        1. Start with bundle of ALL patterns
        2. AMPLIFY high-frequency patterns
        3. NEGATE (remove influence of) low-frequency patterns
        """
        if not self.pattern_vectors:
            self.prototype = np.zeros(self.encoder.vector_manager.dimensions, dtype=np.int8)
            return

        # Start with simple bundle of all patterns
        all_vectors = list(self.pattern_vectors.values())
        self.prototype = self.encoder.bundle(all_vectors)

        # Amplify high-frequency patterns
        for pattern, count in self.pattern_counts.items():
            if count >= amplify_threshold:
                vec = self.pattern_vectors[pattern]
                # Strength proportional to log of frequency
                strength = np.log(count) / 2
                self.prototype = self.encoder.amplify(self.prototype, vec, strength)
                print(f"  AMPLIFY: {pattern[:50]} (count={count}, strength={strength:.2f})")

        # Negate low-frequency patterns (potential anomalies)
        for pattern, count in self.pattern_counts.items():
            if count <= negate_threshold:
                vec = self.pattern_vectors[pattern]
                self.prototype = self.encoder.negate(self.prototype, vec)
                print(f"  NEGATE:  {pattern[:50]} (count={count})")

        return self.prototype

    def similarity(self, record: dict) -> float:
        vec = self.encoder.encode_data(record)
        return cosine(vec, self.prototype)


def generate_mixed_traffic(n_benign=1000, n_malicious=10, seed=42):
    """Generate mixed traffic for testing."""
    import random
    random.seed(seed)

    benign_patterns = [
        {"method": "GET", "path": "/api/users"},
        {"method": "GET", "path": "/api/orders"},
        {"method": "POST", "path": "/api/orders"},
        {"method": "GET", "path": "/api/products"},
    ]

    malicious_patterns = [
        {"method": "GET", "path": "/api/../../../etc/passwd"},
        {"method": "GET", "path": "/api/users/' OR 1=1--"},
        {"method": "TRACE", "path": "/api/users"},
    ]

    records = []
    labels = []

    # Benign with frequency distribution
    weights = [40, 30, 20, 10]  # Some patterns more common
    for _ in range(n_benign):
        pattern = random.choices(benign_patterns, weights=weights)[0]
        records.append(pattern.copy())
        labels.append(False)

    # Malicious (rare)
    for _ in range(n_malicious):
        pattern = random.choice(malicious_patterns)
        records.append(pattern.copy())
        labels.append(True)

    combined = list(zip(records, labels))
    random.shuffle(combined)
    records, labels = zip(*combined)

    return list(records), list(labels)


def pattern_extractor(record: dict) -> str:
    """Extract pattern string from record."""
    return f"{record.get('method', '')}|{record.get('path', '')}"


def main():
    print("=" * 70)
    print("Challenge 010-007: Frequency-Weighted Primitives")
    print("=" * 70)

    # Setup
    vm = DeterministicVectorManager(dimensions=4096, global_seed=42)
    encoder = Encoder(vector_manager=vm)

    # Generate data
    print("\n--- Generating Mixed Traffic ---")
    train_records, train_labels = generate_mixed_traffic(
        n_benign=1000, n_malicious=10, seed=42
    )
    print(f"Training: {sum(1 for l in train_labels if not l)} benign, "
          f"{sum(1 for l in train_labels if l)} malicious")

    test_records, test_labels = generate_mixed_traffic(
        n_benign=200, n_malicious=20, seed=999
    )
    print(f"Test: {sum(1 for l in test_labels if not l)} benign, "
          f"{sum(1 for l in test_labels if l)} malicious")

    # Test different weight functions
    print("\n" + "=" * 70)
    print("Testing Different Weight Functions")
    print("=" * 70)

    weight_functions = ["linear", "log", "sqrt", "threshold"]
    results = {}

    for wf in weight_functions:
        print(f"\n--- Weight Function: {wf} ---")

        detector = FrequencyWeightedPrototype(
            encoder=encoder,
            pattern_extractor=pattern_extractor,
            weight_function=wf,
            threshold_k=5,
        )

        # Train
        detector.observe_batch(train_records)
        detector.build_prototype()

        stats = detector.get_stats()
        print(f"Unique patterns: {stats['unique_patterns']}")
        print("Top patterns:")
        for pattern, count in stats['top_patterns'][:5]:
            weight = detector._compute_weight(count)
            print(f"  {count:4d}x (weight={weight:6.2f}) {pattern}")

        # Evaluate
        benign_sims = []
        malicious_sims = []

        for record, is_malicious in zip(test_records, test_labels):
            sim = detector.similarity(record)
            if is_malicious:
                malicious_sims.append(sim)
            else:
                benign_sims.append(sim)

        # Key metric: separation between benign and malicious
        separation = np.mean(benign_sims) - np.mean(malicious_sims)

        print(f"\nSimilarity distribution:")
        print(f"  Benign:    mean={np.mean(benign_sims):.4f}, std={np.std(benign_sims):.4f}")
        print(f"  Malicious: mean={np.mean(malicious_sims):.4f}, std={np.std(malicious_sims):.4f}")
        print(f"  SEPARATION: {separation:.4f}")

        results[wf] = {
            "benign_mean": np.mean(benign_sims),
            "malicious_mean": np.mean(malicious_sims),
            "separation": separation,
        }

    # Try Amplify-based approach
    print("\n" + "=" * 70)
    print("Amplify/Negate Based Approach")
    print("=" * 70)

    amp_detector = AmplifyBasedDetector(encoder, pattern_extractor)
    amp_detector.observe_batch(train_records)

    print("\nBuilding prototype with amplify/negate:")
    amp_detector.build_prototype(amplify_threshold=50, negate_threshold=3)

    benign_sims = []
    malicious_sims = []

    for record, is_malicious in zip(test_records, test_labels):
        sim = amp_detector.similarity(record)
        if is_malicious:
            malicious_sims.append(sim)
        else:
            benign_sims.append(sim)

    separation = np.mean(benign_sims) - np.mean(malicious_sims)

    print(f"\nSimilarity distribution:")
    print(f"  Benign:    mean={np.mean(benign_sims):.4f}, std={np.std(benign_sims):.4f}")
    print(f"  Malicious: mean={np.mean(malicious_sims):.4f}, std={np.std(malicious_sims):.4f}")
    print(f"  SEPARATION: {separation:.4f}")

    results["amplify_negate"] = {
        "benign_mean": np.mean(benign_sims),
        "malicious_mean": np.mean(malicious_sims),
        "separation": separation,
    }

    # Summary
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"\n{'Method':<20} {'Benign Mean':>12} {'Malicious Mean':>14} {'Separation':>12}")
    print("-" * 60)
    for method, r in sorted(results.items(), key=lambda x: -x[1]["separation"]):
        print(f"{method:<20} {r['benign_mean']:>12.4f} {r['malicious_mean']:>14.4f} {r['separation']:>12.4f}")

    best = max(results.items(), key=lambda x: x[1]["separation"])
    print(f"\nBest separation: {best[0]} with {best[1]['separation']:.4f}")

    print("\n" + "=" * 70)
    print("KEY INSIGHT")
    print("=" * 70)
    print("""
Using Holon's primitives for frequency-based detection:

1. mathematical_bundle(vectors, weights)
   - Weight by frequency: log(count), sqrt(count), etc.
   - High-frequency patterns dominate the prototype
   - Low-frequency anomalies have minimal impact

2. amplify(prototype, component, strength)
   - Explicitly boost frequently-seen patterns
   - strength = log(count) gives sublinear boost

3. negate(prototype, component)
   - Remove influence of rare patterns (potential anomalies)
   - Rare patterns that slip through training are de-emphasized

4. PROPOSED NEW PRIMITIVE: frequency_prototype()

   def frequency_prototype(
       self,
       pattern_vectors: Dict[str, np.ndarray],
       pattern_counts: Dict[str, int],
       decay: str = "log",
   ) -> np.ndarray:
       '''
       Build prototype with frequency-based weighting.

       High-frequency patterns contribute more.
       Low-frequency patterns decay toward zero influence.
       '''
       weights = [log(count + 1) for count in pattern_counts.values()]
       return mathematical_bundle(list(pattern_vectors.values()), weights)

This would be a first-class Holon primitive for streaming anomaly detection.
""")


if __name__ == "__main__":
    main()
