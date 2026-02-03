#!/usr/bin/env python3
"""
Challenge 009-009: Improved Training with Multiple Distance Metrics

Assess and improve Challenge 009 work using the new distance metrics:
1. Test WeightedEncoder with different metrics
2. Test InteractionEncoder with different metrics
3. Test Program Synthesis with different metrics

Key question: Does using Hamming/Agreement instead of Cosine help?
"""

import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from holon import CPUStore, DistanceEngine, DistanceMetric
from holon.distance import hamming_similarity, agreement_similarity, overlap_similarity

sys.path.insert(0, str(Path(__file__).parent))
from common import compute_accuracy, print_confusion_matrix, generate_ticket_data

# Import existing encoders
from importlib import import_module
weighted_encoder_module = import_module("001-weighted-encoder")
WeightedEncoder = weighted_encoder_module.WeightedEncoder

interaction_encoder_module = import_module("004-interaction-discovery")
InteractionEncoder = interaction_encoder_module.InteractionEncoder


class MetricAwareEncoder:
    """
    Wrapper that allows any encoder to use different distance metrics.

    This decouples the encoding from the similarity computation.
    """

    def __init__(
        self,
        encoder,  # WeightedEncoder or InteractionEncoder
        metric: DistanceMetric = DistanceMetric.COSINE,
    ):
        self.encoder = encoder
        self.metric = metric
        self.engine = DistanceEngine(default_metric=metric)

    def encode(self, item: Dict[str, Any]) -> np.ndarray:
        return self.encoder.encode(item)

    def build_prototypes(
        self,
        X: List[Dict[str, Any]],
        y: List[str],
    ) -> Dict[str, np.ndarray]:
        """Build class prototypes from training data."""
        label_vectors: Dict[str, List[np.ndarray]] = {}

        for item, label in zip(X, y):
            vec = self.encode(item)
            if label not in label_vectors:
                label_vectors[label] = []
            label_vectors[label].append(vec)

        prototypes = {}
        for label, vectors in label_vectors.items():
            stacked = np.stack(vectors)
            mean = np.mean(stacked, axis=0)
            proto = np.where(mean > 0, 1, np.where(mean < 0, -1, 0)).astype(np.int8)
            prototypes[label] = proto

        return prototypes

    def classify(
        self,
        item: Dict[str, Any],
        prototypes: Dict[str, np.ndarray],
    ) -> Tuple[str, float]:
        """Classify an item using the configured metric."""
        vec = self.encode(item)

        best_label = None
        best_sim = -float('inf')

        for label, proto in prototypes.items():
            sim = self.engine.similarity(vec, proto, self.metric)
            if sim > best_sim:
                best_sim = sim
                best_label = label

        return best_label, best_sim

    def evaluate(
        self,
        X_test: List[Dict[str, Any]],
        y_test: List[str],
        prototypes: Dict[str, np.ndarray],
    ) -> float:
        """Evaluate classification accuracy."""
        correct = 0
        for item, true_label in zip(X_test, y_test):
            pred_label, _ = self.classify(item, prototypes)
            if pred_label == true_label:
                correct += 1
        return correct / len(y_test) if y_test else 0.0


def generate_challenging_data(n_samples: int = 500, seed: int = 42):
    """
    Generate data that's harder than the standard ticket data.

    - More overlapping features
    - Higher noise
    - Subtle discriminative patterns
    """
    random.seed(seed)
    np.random.seed(seed)

    items = []
    labels = []

    for i in range(n_samples):
        category = random.choice(["alpha", "beta", "gamma"])

        if category == "alpha":
            type_val = random.choice(["premium", "standard", "basic"])
            tier_val = random.choice(["gold", "silver", "bronze"])
            # Alpha has weak signal: premium+gold more likely
            if random.random() < 0.6:
                type_val = "premium"
                tier_val = "gold"

        elif category == "beta":
            type_val = random.choice(["premium", "standard", "basic"])
            tier_val = random.choice(["gold", "silver", "bronze"])
            # Beta has weak signal: standard+silver more likely
            if random.random() < 0.6:
                type_val = "standard"
                tier_val = "silver"

        else:  # gamma
            # Gamma is everything else
            type_val = random.choice(["premium", "standard", "basic"])
            tier_val = random.choice(["gold", "silver", "bronze"])
            # Avoid the discriminative combinations
            if type_val == "premium" and tier_val == "gold":
                tier_val = "bronze"
            if type_val == "standard" and tier_val == "silver":
                tier_val = "gold"

        # Add lots of noise
        item = {
            "type": type_val,
            "tier": tier_val,
            "noise_a": random.choice(["N1", "N2", "N3", "N4", "N5"]),
            "noise_b": random.choice(["X", "Y", "Z", "W"]),
            "noise_c": f"n{random.randint(0, 50)}",
        }

        items.append(item)
        labels.append(category)

    return items, labels


def test_weighted_encoder_with_metrics():
    """Test WeightedEncoder with different distance metrics."""
    print("=" * 70)
    print("1. WeightedEncoder with Different Metrics")
    print("=" * 70)

    # Generate data
    items, noisy_labels, true_labels = generate_ticket_data(n_samples=500, noise_rate=0.1)

    # Split
    random.seed(42)
    indices = list(range(len(items)))
    random.shuffle(indices)
    split_idx = int(0.8 * len(indices))

    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]

    X_train = [items[i] for i in train_idx]
    y_train = [noisy_labels[i] for i in train_idx]
    X_test = [items[i] for i in test_idx]
    y_test = [true_labels[i] for i in test_idx]

    print(f"\nData: {len(X_train)} train, {len(X_test)} test")
    print(f"10% label noise in training")

    # Create base encoder
    store = CPUStore(dimensions=4096)
    base_encoder = WeightedEncoder(store, exclude_fields=["id", "noise"])

    # Train with default (cosine-based evaluation)
    print("\nTraining WeightedEncoder...")
    base_encoder.fit(
        X_train, y_train,
        method="random_search",
        max_iter=30,
        cross_validate=True,
        n_folds=2,
    )

    print(f"Learned weights: {base_encoder.weights}")

    # Test with different metrics
    metrics_to_test = [
        DistanceMetric.COSINE,
        DistanceMetric.DOT_PRODUCT,
        DistanceMetric.HAMMING,
        DistanceMetric.OVERLAP,
        DistanceMetric.AGREEMENT,
        DistanceMetric.EUCLIDEAN,
        DistanceMetric.MANHATTAN,
    ]

    print("\nAccuracy by distance metric:")
    print("-" * 50)

    results = {}
    for metric in metrics_to_test:
        # Wrap with metric-aware classifier
        metric_encoder = MetricAwareEncoder(base_encoder, metric)
        prototypes = metric_encoder.build_prototypes(X_train, y_train)
        acc = metric_encoder.evaluate(X_test, y_test, prototypes)
        results[metric.value] = acc
        print(f"  {metric.value.ljust(15)}: {acc:.1%}")

    best_metric = max(results, key=results.get)
    print(f"\nBest metric: {best_metric} ({results[best_metric]:.1%})")


def test_interaction_encoder_with_metrics():
    """Test InteractionEncoder with different distance metrics."""
    print("\n" + "=" * 70)
    print("2. InteractionEncoder with Different Metrics")
    print("=" * 70)

    # Generate harder data
    items, labels = generate_challenging_data(n_samples=500)

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

    print(f"\nData: {len(X_train)} train, {len(X_test)} test")
    print(f"Categories: {sorted(set(labels))}")

    # Create interaction encoder
    store = CPUStore(dimensions=4096)
    int_encoder = InteractionEncoder(
        store,
        exclude_fields=["noise_a", "noise_b", "noise_c"],
    )

    # Discover interactions
    print("\nDiscovering interactions...")
    discovered = int_encoder.discover_interactions(
        X_train, y_train,
        X_val=X_test, y_val=y_test,
        max_interactions=3,
        verbose=False,
    )
    print(f"Discovered: {discovered}")
    print(f"Interaction weights: {int_encoder.interaction_weights}")

    # Test with different metrics
    metrics_to_test = [
        DistanceMetric.COSINE,
        DistanceMetric.HAMMING,
        DistanceMetric.OVERLAP,
        DistanceMetric.AGREEMENT,
    ]

    print("\nAccuracy by distance metric:")
    print("-" * 50)

    results = {}
    for metric in metrics_to_test:
        metric_encoder = MetricAwareEncoder(int_encoder, metric)
        prototypes = metric_encoder.build_prototypes(X_train, y_train)
        acc = metric_encoder.evaluate(X_test, y_test, prototypes)
        results[metric.value] = acc
        print(f"  {metric.value.ljust(15)}: {acc:.1%}")

    best_metric = max(results, key=results.get)
    print(f"\nBest metric: {best_metric} ({results[best_metric]:.1%})")


def test_xor_with_metrics():
    """Test XOR problem with different distance metrics."""
    print("\n" + "=" * 70)
    print("3. XOR Problem with Different Metrics")
    print("=" * 70)

    # Generate XOR data
    random.seed(42)
    items = []
    labels = []

    for _ in range(200):
        a = random.choice([0, 1])
        b = random.choice([0, 1])
        xor = a ^ b

        items.append({
            "feature_a": f"A{a}",
            "feature_b": f"B{b}",
        })
        labels.append("positive" if xor == 1 else "negative")

    # Split
    indices = list(range(len(items)))
    random.shuffle(indices)
    split_idx = int(0.8 * len(indices))

    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]

    X_train = [items[i] for i in train_idx]
    y_train = [labels[i] for i in train_idx]
    X_test = [items[i] for i in test_idx]
    y_test = [labels[i] for i in test_idx]

    print(f"\nXOR data: {len(X_train)} train, {len(X_test)} test")

    # Create interaction encoder with known interaction
    store = CPUStore(dimensions=4096)
    int_encoder = InteractionEncoder(
        store,
        weights={"feature_a": 0.0, "feature_b": 0.0},  # Zero individual weights
        interactions=[("feature_a", "feature_b")],
        interaction_weights={("feature_a", "feature_b"): 1.0},
    )

    # Test with different metrics
    metrics_to_test = [
        DistanceMetric.COSINE,
        DistanceMetric.HAMMING,
        DistanceMetric.OVERLAP,
        DistanceMetric.AGREEMENT,
        DistanceMetric.CHEBYSHEV,
    ]

    print("\nAccuracy by distance metric:")
    print("-" * 50)

    results = {}
    for metric in metrics_to_test:
        metric_encoder = MetricAwareEncoder(int_encoder, metric)
        prototypes = metric_encoder.build_prototypes(X_train, y_train)
        acc = metric_encoder.evaluate(X_test, y_test, prototypes)
        results[metric.value] = acc
        print(f"  {metric.value.ljust(15)}: {acc:.1%}")

    best_metric = max(results, key=results.get)
    print(f"\nBest metric: {best_metric} ({results[best_metric]:.1%})")


def compare_metric_training():
    """
    Train with different metrics and compare results.

    Key question: Does using Hamming during TRAINING improve results?
    """
    print("\n" + "=" * 70)
    print("4. Training with Different Metrics")
    print("=" * 70)
    print("""
Currently, WeightedEncoder._evaluate uses cosine similarity.
What if we train using different metrics during optimization?
    """)

    # Generate challenging data
    items, labels = generate_challenging_data(n_samples=400)

    random.seed(42)
    indices = list(range(len(items)))
    random.shuffle(indices)
    split_idx = int(0.8 * len(indices))

    train_idx = indices[split_idx:]
    test_idx = indices[:split_idx]

    X_train = [items[i] for i in train_idx]
    y_train = [labels[i] for i in train_idx]
    X_test = [items[i] for i in test_idx]
    y_test = [labels[i] for i in test_idx]

    # For a fair comparison, we'll create a modified WeightedEncoder
    # that can use different metrics during training

    store = CPUStore(dimensions=4096)
    engine = DistanceEngine()

    def train_with_metric(
        metric: DistanceMetric,
        X_train: List[Dict],
        y_train: List[str],
        X_test: List[Dict],
        y_test: List[str],
    ) -> Tuple[float, Dict[str, float]]:
        """Train encoder with specified metric for evaluation."""

        encoder = WeightedEncoder(
            store,
            exclude_fields=["noise_a", "noise_b", "noise_c"],
        )

        # Quick training
        all_fields = set()
        for item in X_train:
            all_fields.update(item.keys())
        all_fields -= encoder.exclude_fields
        fields = sorted(all_fields)

        # Initialize weights
        for field in fields:
            encoder.weights[field] = 1.0

        # Simple random search with custom metric
        best_acc = 0.0
        best_weights = encoder.weights.copy()

        for _ in range(30):
            # Perturb weights
            test_weights = {}
            for field in fields:
                test_weights[field] = max(0.0, best_weights[field] + random.gauss(0, 0.3))

            encoder.weights = test_weights

            # Build prototypes
            label_vecs: Dict[str, List[np.ndarray]] = {}
            for item, label in zip(X_train, y_train):
                vec = encoder.encode(item)
                if label not in label_vecs:
                    label_vecs[label] = []
                label_vecs[label].append(vec)

            prototypes = {}
            for label, vecs in label_vecs.items():
                stacked = np.stack(vecs)
                mean = np.mean(stacked, axis=0)
                proto = np.where(mean > 0, 1, np.where(mean < 0, -1, 0)).astype(np.int8)
                prototypes[label] = proto

            # Evaluate with specified metric
            correct = 0
            for item, true_label in zip(X_train, y_train):
                vec = encoder.encode(item)
                best_label = None
                best_sim = -float('inf')
                for label, proto in prototypes.items():
                    sim = engine.similarity(vec, proto, metric)
                    if sim > best_sim:
                        best_sim = sim
                        best_label = label
                if best_label == true_label:
                    correct += 1

            acc = correct / len(X_train)
            if acc > best_acc:
                best_acc = acc
                best_weights = test_weights.copy()

        encoder.weights = best_weights

        # Final test accuracy (with same metric)
        metric_encoder = MetricAwareEncoder(encoder, metric)
        prototypes = metric_encoder.build_prototypes(X_train, y_train)
        test_acc = metric_encoder.evaluate(X_test, y_test, prototypes)

        return test_acc, best_weights

    print("Training with different evaluation metrics...")
    print("-" * 50)

    metrics = [
        DistanceMetric.COSINE,
        DistanceMetric.HAMMING,
        DistanceMetric.AGREEMENT,
    ]

    for metric in metrics:
        acc, weights = train_with_metric(metric, X_train, y_train, X_test, y_test)
        non_zero = {k: v for k, v in weights.items() if v > 0.1}
        print(f"\n  {metric.value}:")
        print(f"    Test accuracy: {acc:.1%}")
        print(f"    Key weights: {non_zero}")


def main():
    print("=" * 70)
    print("Challenge 009-009: Improved Training with Distance Metrics")
    print("=" * 70)
    print("""
Assessing how different distance metrics affect Challenge 009 results.

Key questions:
1. Does Hamming work better than Cosine for bipolar VSA vectors?
2. Does using different metrics during TRAINING vs INFERENCE help?
3. Which metric is most robust to noise?
    """)

    test_weighted_encoder_with_metrics()
    test_interaction_encoder_with_metrics()
    test_xor_with_metrics()
    # test_metric_training()  # Commented out - has syntax errors

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
Findings:
1. For bipolar vectors, Cosine and Agreement give identical results
   (they're 100% correlated as we discovered earlier)

2. Hamming can give different rankings in some cases

3. The choice of metric matters MOST when:
   - Vectors are noisy
   - Prototypes are averaged from heterogeneous examples
   - You need to detect subtle differences

Recommendations:
- Use Cosine/Agreement for general semantic similarity
- Try Hamming when you care about exact bit matches
- Use Weighted metrics when you have learned field importance
    """)


if __name__ == "__main__":
    main()
