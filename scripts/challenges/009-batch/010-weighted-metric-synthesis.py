#!/usr/bin/env python3
"""
Challenge 009-010: Weighted Metric Program Synthesis

Extend program synthesis to learn BOTH:
1. Encoding weights (which fields matter)
2. Metric weights (per-dimension importance in similarity)

This is a more powerful form of training:
- Encoding weights affect vector construction
- Metric weights affect how we compare vectors

Key insight: Instead of just learning to encode well,
we learn to encode AND compare well together.
"""

import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from holon import CPUStore, DistanceEngine, DistanceMetric
from holon.distance import weighted_cosine_similarity

sys.path.insert(0, str(Path(__file__).parent))
from common import compute_accuracy, print_confusion_matrix


def generate_subtle_data(n_samples: int = 600, seed: int = 42):
    """
    Generate data with subtle patterns that benefit from learned weights.

    Key: The discriminative information is spread across multiple fields,
    but some dimensions of the encoded vector are more important than others.
    """
    random.seed(seed)
    np.random.seed(seed)

    items = []
    labels = []

    for i in range(n_samples):
        category = random.choice(["class_a", "class_b", "class_c"])

        # Each class has subtle patterns in different field combinations
        if category == "class_a":
            # Class A: tends toward premium + technical
            type_val = random.choices(
                ["premium", "standard", "basic"],
                weights=[0.6, 0.3, 0.1]
            )[0]
            domain = random.choices(
                ["technical", "billing", "general"],
                weights=[0.5, 0.3, 0.2]
            )[0]

        elif category == "class_b":
            # Class B: tends toward standard + billing
            type_val = random.choices(
                ["premium", "standard", "basic"],
                weights=[0.2, 0.6, 0.2]
            )[0]
            domain = random.choices(
                ["technical", "billing", "general"],
                weights=[0.2, 0.6, 0.2]
            )[0]

        else:  # class_c
            # Class C: tends toward basic + general
            type_val = random.choices(
                ["premium", "standard", "basic"],
                weights=[0.1, 0.3, 0.6]
            )[0]
            domain = random.choices(
                ["technical", "billing", "general"],
                weights=[0.2, 0.2, 0.6]
            )[0]

        # Add noise fields
        item = {
            "type": type_val,
            "domain": domain,
            "noise1": random.choice(["N1", "N2", "N3"]),
            "noise2": random.choice(["X", "Y", "Z"]),
            "noise3": f"id_{random.randint(0, 100)}",
        }

        items.append(item)
        labels.append(category)

    return items, labels


class WeightedMetricEncoder:
    """
    Encoder that learns:
    1. Field weights for encoding
    2. Dimension weights for similarity

    This is more powerful than just field weights because it
    can emphasize certain PARTS of the encoded vector.
    """

    def __init__(
        self,
        store: CPUStore,
        field_weights: Dict[str, float] = None,
        dimension_weights: np.ndarray = None,
        exclude_fields: List[str] = None,
    ):
        self.store = store
        self.encoder = store.encoder
        self.field_weights = field_weights or {}
        self.dimension_weights = dimension_weights
        self.exclude_fields = set(exclude_fields or [])

    def encode(self, item: Dict[str, Any]) -> np.ndarray:
        """Encode with field weights."""
        field_vectors = []
        weights = []

        for key, value in item.items():
            if key in self.exclude_fields:
                continue

            weight = self.field_weights.get(key, 1.0)
            if weight == 0.0:
                continue

            vec = self.encoder.encode_data({key: value})
            field_vectors.append(vec.astype(np.float32))
            weights.append(weight)

        if not field_vectors:
            return np.zeros(self.store.dimensions, dtype=np.int8)

        # Weighted sum
        weighted_sum = np.zeros(self.store.dimensions, dtype=np.float32)
        for vec, weight in zip(field_vectors, weights):
            weighted_sum += weight * vec

        # Threshold to bipolar
        result = np.where(weighted_sum > 0, 1, np.where(weighted_sum < 0, -1, 0))
        return result.astype(np.int8)

    def similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute similarity with optional dimension weights."""
        if self.dimension_weights is not None:
            return weighted_cosine_similarity(vec1, vec2, self.dimension_weights)
        else:
            # Standard cosine
            dot = np.dot(vec1.astype(float), vec2.astype(float))
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return dot / (norm1 * norm2)

    def fit(
        self,
        X_train: List[Dict],
        y_train: List[str],
        learn_dimension_weights: bool = True,
        n_dimension_clusters: int = 16,
        max_iter: int = 50,
        verbose: bool = True,
    ):
        """
        Learn both field weights and dimension weights.

        Args:
            X_train: Training examples
            y_train: Training labels
            learn_dimension_weights: Whether to learn dimension weights
            n_dimension_clusters: Number of dimension clusters to learn
            max_iter: Maximum iterations
            verbose: Print progress
        """
        # Discover fields
        all_fields = set()
        for item in X_train:
            all_fields.update(item.keys())
        all_fields -= self.exclude_fields
        fields = sorted(all_fields)

        if verbose:
            print(f"Learning weights for {len(fields)} fields...")

        # Initialize field weights
        for field in fields:
            self.field_weights[field] = 1.0

        # Initialize dimension weights (cluster-based for efficiency)
        if learn_dimension_weights:
            cluster_size = self.store.dimensions // n_dimension_clusters
            self.dimension_weights = np.ones(self.store.dimensions, dtype=np.float32)
            cluster_weights = np.ones(n_dimension_clusters, dtype=np.float32)
        else:
            self.dimension_weights = None
            cluster_weights = None

        # Joint optimization
        best_acc = self._evaluate(X_train, y_train)
        best_field_weights = self.field_weights.copy()
        best_cluster_weights = cluster_weights.copy() if cluster_weights is not None else None

        if verbose:
            print(f"Initial accuracy: {best_acc:.1%}")

        for iteration in range(max_iter):
            # Perturb field weights
            new_field_weights = {}
            for field in fields:
                new_field_weights[field] = max(0.0, best_field_weights[field] + random.gauss(0, 0.3))
            self.field_weights = new_field_weights

            # Perturb dimension cluster weights
            if cluster_weights is not None:
                new_cluster_weights = np.array([
                    max(0.1, w + random.gauss(0, 0.2))
                    for w in best_cluster_weights
                ])
                # Expand to full dimension weights
                self.dimension_weights = np.repeat(new_cluster_weights, cluster_size)
                if len(self.dimension_weights) < self.store.dimensions:
                    # Pad to full size
                    padding = np.full(
                        self.store.dimensions - len(self.dimension_weights),
                        new_cluster_weights[-1]
                    )
                    self.dimension_weights = np.concatenate([self.dimension_weights, padding])

            # Evaluate
            acc = self._evaluate(X_train, y_train)

            if acc > best_acc:
                best_acc = acc
                best_field_weights = self.field_weights.copy()
                if cluster_weights is not None:
                    best_cluster_weights = new_cluster_weights.copy()
                if verbose:
                    print(f"  Iter {iteration+1}: {acc:.1%} (improved)")

        # Set best weights
        self.field_weights = best_field_weights
        if best_cluster_weights is not None:
            cluster_size = self.store.dimensions // n_dimension_clusters
            self.dimension_weights = np.repeat(best_cluster_weights, cluster_size)
            if len(self.dimension_weights) < self.store.dimensions:
                padding = np.full(
                    self.store.dimensions - len(self.dimension_weights),
                    best_cluster_weights[-1]
                )
                self.dimension_weights = np.concatenate([self.dimension_weights, padding])

        if verbose:
            print(f"\nFinal accuracy: {best_acc:.1%}")
            print(f"Field weights: {self.field_weights}")
            if cluster_weights is not None:
                print(f"Dimension cluster weights: {best_cluster_weights}")

    def _evaluate(self, X: List[Dict], y: List[str]) -> float:
        """Evaluate with prototype-based classification."""
        # Build prototypes
        label_vecs: Dict[str, List[np.ndarray]] = {}
        for item, label in zip(X, y):
            vec = self.encode(item)
            if label not in label_vecs:
                label_vecs[label] = []
            label_vecs[label].append(vec)

        prototypes = {}
        for label, vecs in label_vecs.items():
            stacked = np.stack(vecs)
            mean = np.mean(stacked, axis=0)
            proto = np.where(mean > 0, 1, np.where(mean < 0, -1, 0)).astype(np.int8)
            prototypes[label] = proto

        # Classify
        correct = 0
        for item, true_label in zip(X, y):
            vec = self.encode(item)
            best_label = None
            best_sim = -float('inf')
            for label, proto in prototypes.items():
                sim = self.similarity(vec, proto)
                if sim > best_sim:
                    best_sim = sim
                    best_label = label
            if best_label == true_label:
                correct += 1

        return correct / len(y) if y else 0.0


def main():
    print("=" * 70)
    print("Challenge 009-010: Weighted Metric Program Synthesis")
    print("=" * 70)
    print("""
Learning BOTH:
1. Field weights (which fields matter for encoding)
2. Dimension weights (which parts of the vector matter for comparison)

This is more powerful than just field weights alone.
    """)

    # Generate data
    print("\n1. Generating subtle classification data...")
    items, labels = generate_subtle_data(n_samples=600)

    # Split
    random.seed(42)
    indices = list(range(len(items)))
    random.shuffle(indices)
    split_idx = int(0.8 * len(indices))

    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]

    X_train = [items[i] for i in train_idx]
    y_train = [labels[i] for i in train_idx]
    X_test = [items[i] for i in test_idx]
    y_test = [labels[i] for i in test_idx]

    print(f"   Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"   Classes: {sorted(set(labels))}")

    # Test 1: Field weights only
    print("\n" + "=" * 70)
    print("2. Field Weights Only (baseline)")
    print("=" * 70)

    store = CPUStore(dimensions=4096)
    encoder1 = WeightedMetricEncoder(
        store,
        exclude_fields=["noise1", "noise2", "noise3"],
    )
    encoder1.fit(
        X_train, y_train,
        learn_dimension_weights=False,
        max_iter=50,
        verbose=True,
    )

    # Evaluate on test
    label_vecs = {}
    for item, label in zip(X_train, y_train):
        vec = encoder1.encode(item)
        if label not in label_vecs:
            label_vecs[label] = []
        label_vecs[label].append(vec)

    prototypes = {}
    for label, vecs in label_vecs.items():
        stacked = np.stack(vecs)
        mean = np.mean(stacked, axis=0)
        proto = np.where(mean > 0, 1, np.where(mean < 0, -1, 0)).astype(np.int8)
        prototypes[label] = proto

    correct = 0
    for item, true_label in zip(X_test, y_test):
        vec = encoder1.encode(item)
        best_label = max(prototypes, key=lambda l: encoder1.similarity(vec, prototypes[l]))
        if best_label == true_label:
            correct += 1

    acc1 = correct / len(X_test)
    print(f"\nTest accuracy (field weights only): {acc1:.1%}")

    # Test 2: Field weights + Dimension weights
    print("\n" + "=" * 70)
    print("3. Field Weights + Dimension Weights")
    print("=" * 70)

    encoder2 = WeightedMetricEncoder(
        store,
        exclude_fields=["noise1", "noise2", "noise3"],
    )
    encoder2.fit(
        X_train, y_train,
        learn_dimension_weights=True,
        n_dimension_clusters=16,  # Learn 16 cluster weights
        max_iter=50,
        verbose=True,
    )

    # Evaluate on test
    label_vecs = {}
    for item, label in zip(X_train, y_train):
        vec = encoder2.encode(item)
        if label not in label_vecs:
            label_vecs[label] = []
        label_vecs[label].append(vec)

    prototypes = {}
    for label, vecs in label_vecs.items():
        stacked = np.stack(vecs)
        mean = np.mean(stacked, axis=0)
        proto = np.where(mean > 0, 1, np.where(mean < 0, -1, 0)).astype(np.int8)
        prototypes[label] = proto

    correct = 0
    for item, true_label in zip(X_test, y_test):
        vec = encoder2.encode(item)
        best_label = max(prototypes, key=lambda l: encoder2.similarity(vec, prototypes[l]))
        if best_label == true_label:
            correct += 1

    acc2 = correct / len(X_test)
    print(f"\nTest accuracy (field + dimension weights): {acc2:.1%}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
Results:
  Field weights only:         {acc1:.1%}
  Field + Dimension weights:  {acc2:.1%}
  Improvement:                {acc2 - acc1:+.1%}

The dimension weights allow the model to emphasize certain PARTS
of the encoded vector during comparison, providing an additional
lever for optimization beyond just field encoding weights.

This is similar to attention mechanisms in neural networks,
but implemented as a deterministic, interpretable weight vector.
    """)


if __name__ == "__main__":
    main()
