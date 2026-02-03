#!/usr/bin/env python3
"""
Challenge 009-008: Distance Metrics Evaluation

Test different distance metrics on VSA bipolar vectors to understand:
1. Which metrics work best for bipolar vectors
2. How Qdrant-native vs client-side metrics compare
3. Which metrics are most useful for classification/training

Key insight: For VSA bipolar vectors (-1, 0, +1), Hamming-based metrics
may be more natural than geometric metrics like Euclidean.
"""

import sys
from pathlib import Path

import numpy as np

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from holon import CPUStore
from holon.distance import (
    DistanceEngine,
    DistanceMetric,
    compare_metrics,
    get_recommended_metric,
)


def demo_basic_metrics():
    """Show how each metric behaves on identical/similar/opposite vectors."""
    print("=" * 70)
    print("1. BASIC METRIC BEHAVIOR")
    print("=" * 70)

    # Create test vectors
    np.random.seed(42)
    D = 1000  # 1000 dimensions

    # Random bipolar vector
    vec_a = np.random.choice([-1, 0, 1], size=D, p=[0.33, 0.34, 0.33]).astype(np.int8)

    # Identical
    vec_identical = vec_a.copy()

    # Similar (90% same)
    vec_similar = vec_a.copy()
    flip_idx = np.random.choice(D, size=int(D * 0.1), replace=False)
    vec_similar[flip_idx] *= -1

    # Opposite
    vec_opposite = -vec_a

    # Orthogonal (independent random)
    vec_orthogonal = np.random.choice([-1, 0, 1], size=D, p=[0.33, 0.34, 0.33]).astype(np.int8)

    print("\nVector relationships:")
    print(f"  - vec_a: random bipolar vector (D={D})")
    print(f"  - vec_identical: exact copy of vec_a")
    print(f"  - vec_similar: 90% same as vec_a")
    print(f"  - vec_opposite: -vec_a")
    print(f"  - vec_orthogonal: independent random")

    pairs = [
        ("identical", vec_identical),
        ("similar (90%)", vec_similar),
        ("orthogonal", vec_orthogonal),
        ("opposite", vec_opposite),
    ]

    print("\nSimilarity scores (higher = more similar):")
    print("-" * 70)
    header = "Metric".ljust(20)
    for name, _ in pairs:
        header += name.ljust(15)
    print(header)
    print("-" * 70)

    metrics_to_test = [
        DistanceMetric.COSINE,
        DistanceMetric.DOT_PRODUCT,
        DistanceMetric.HAMMING,
        DistanceMetric.OVERLAP,
        DistanceMetric.AGREEMENT,
        DistanceMetric.EUCLIDEAN,
        DistanceMetric.MANHATTAN,
        DistanceMetric.CHEBYSHEV,
    ]

    engine = DistanceEngine()

    for metric in metrics_to_test:
        row = metric.value.ljust(20)
        for name, vec_b in pairs:
            sim = engine.similarity(vec_a, vec_b, metric)
            row += f"{sim:+.3f}".ljust(15)
        print(row)


def demo_qdrant_compatibility():
    """Show which metrics are Qdrant-native vs client-side."""
    print("\n" + "=" * 70)
    print("2. QDRANT COMPATIBILITY")
    print("=" * 70)

    engine = DistanceEngine()

    print("\nQdrant-Native Metrics (server-side, fast):")
    for metric in DistanceMetric:
        if engine.is_qdrant_native(metric):
            qdrant_name = engine.get_qdrant_distance(metric)
            print(f"  ✓ {metric.value} → Qdrant: {qdrant_name}")

    print("\nClient-Side Metrics (compute after retrieval):")
    for metric in DistanceMetric:
        if not engine.is_qdrant_native(metric):
            print(f"  • {metric.value}")

    print("""
Strategy for non-native metrics:
1. Use Qdrant-native metric for initial retrieval (top-N candidates)
2. Re-rank with client-side metric if needed
3. For training, client-side metrics are fine (no real-time constraint)
    """)


def demo_classification_impact():
    """Test how different metrics affect classification accuracy."""
    print("\n" + "=" * 70)
    print("3. CLASSIFICATION ACCURACY BY METRIC")
    print("=" * 70)

    # Generate classification data
    np.random.seed(42)
    store = CPUStore(dimensions=4096)
    encoder = store.encoder

    # Create class prototypes
    class_data = {
        "billing": [
            {"type": "billing", "issue": "refund"},
            {"type": "billing", "issue": "charge"},
            {"type": "billing", "issue": "invoice"},
        ],
        "technical": [
            {"type": "technical", "issue": "crash"},
            {"type": "technical", "issue": "bug"},
            {"type": "technical", "issue": "error"},
        ],
        "general": [
            {"type": "general", "issue": "info"},
            {"type": "general", "issue": "question"},
            {"type": "general", "issue": "feedback"},
        ],
    }

    # Encode and create prototypes
    prototypes = {}
    for label, examples in class_data.items():
        vecs = [encoder.encode_data(ex) for ex in examples]
        stacked = np.stack(vecs)
        mean = np.mean(stacked, axis=0)
        proto = np.where(mean > 0, 1, np.where(mean < 0, -1, 0)).astype(np.int8)
        prototypes[label] = proto

    # Test items
    test_items = [
        ({"type": "billing", "issue": "payment"}, "billing"),
        ({"type": "billing", "issue": "subscription"}, "billing"),
        ({"type": "technical", "issue": "slow"}, "technical"),
        ({"type": "technical", "issue": "broken"}, "technical"),
        ({"type": "general", "issue": "help"}, "general"),
        ({"type": "general", "issue": "query"}, "general"),
    ]

    metrics_to_test = [
        DistanceMetric.COSINE,
        DistanceMetric.DOT_PRODUCT,
        DistanceMetric.HAMMING,
        DistanceMetric.OVERLAP,
        DistanceMetric.AGREEMENT,
        DistanceMetric.EUCLIDEAN,
        DistanceMetric.MANHATTAN,
    ]

    engine = DistanceEngine()

    print("\nClassification results by metric:")
    print("-" * 50)

    for metric in metrics_to_test:
        correct = 0
        for item, true_label in test_items:
            vec = encoder.encode_data(item)

            # Find most similar prototype
            best_label = None
            best_sim = -float('inf')
            for label, proto in prototypes.items():
                sim = engine.similarity(vec, proto, metric)
                if sim > best_sim:
                    best_sim = sim
                    best_label = label

            if best_label == true_label:
                correct += 1

        acc = correct / len(test_items)
        print(f"  {metric.value.ljust(15)}: {acc:.1%} ({correct}/{len(test_items)})")


def demo_metric_correlation():
    """Show correlation between metrics on random vector pairs."""
    print("\n" + "=" * 70)
    print("4. METRIC CORRELATION ANALYSIS")
    print("=" * 70)

    np.random.seed(42)
    D = 2000

    # Generate random vector pairs
    n_pairs = 100
    pairs = []
    for _ in range(n_pairs):
        v1 = np.random.choice([-1, 0, 1], size=D, p=[0.33, 0.34, 0.33]).astype(np.int8)
        v2 = np.random.choice([-1, 0, 1], size=D, p=[0.33, 0.34, 0.33]).astype(np.int8)
        pairs.append((v1, v2))

    metrics = [
        DistanceMetric.COSINE,
        DistanceMetric.HAMMING,
        DistanceMetric.OVERLAP,
        DistanceMetric.AGREEMENT,
    ]

    engine = DistanceEngine()

    # Compute similarities for all pairs
    results = {m: [] for m in metrics}
    for v1, v2 in pairs:
        for m in metrics:
            results[m].append(engine.similarity(v1, v2, m))

    # Convert to arrays
    for m in metrics:
        results[m] = np.array(results[m])

    # Correlation matrix
    print("\nCorrelation between metrics (on random bipolar vectors):")
    print("-" * 50)

    header = "".ljust(15)
    for m in metrics:
        header += m.value[:10].ljust(12)
    print(header)

    for m1 in metrics:
        row = m1.value[:12].ljust(15)
        for m2 in metrics:
            corr = np.corrcoef(results[m1], results[m2])[0, 1]
            row += f"{corr:.3f}".ljust(12)
        print(row)

    print("""
Key insights:
- Cosine and Agreement are highly correlated for bipolar vectors
- Hamming and Overlap are complementary views of the same thing
- For bipolar VSA, Hamming/Agreement may be more interpretable
    """)


def demo_recommendations():
    """Show recommended metrics for different use cases."""
    print("\n" + "=" * 70)
    print("5. RECOMMENDED METRICS BY USE CASE")
    print("=" * 70)

    use_cases = {
        "semantic": "General semantic similarity, NLP embeddings",
        "bipolar": "VSA bipolar vector comparison",
        "normalized": "Pre-normalized vectors (unit length)",
        "geometric": "Spatial/geometric relationships",
        "outlier": "Outlier-sensitive comparison",
        "weighted": "Feature importance weighting",
        "grid": "Grid-like/lattice spaces",
    }

    print("\nUse Case".ljust(15) + "Metric".ljust(20) + "Description")
    print("-" * 70)

    for use_case, desc in use_cases.items():
        metric = get_recommended_metric(use_case)
        print(f"{use_case.ljust(15)}{metric.value.ljust(20)}{desc}")


def main():
    print("=" * 70)
    print("Challenge 009-008: Distance Metrics for VSA Bipolar Vectors")
    print("=" * 70)
    print("""
This evaluates different distance metrics for Holon's VSA vectors.

Key questions:
1. Which metrics work best for bipolar (-1, 0, +1) vectors?
2. Which metrics are Qdrant-native (fast) vs client-side?
3. How do metrics affect classification accuracy?
    """)

    demo_basic_metrics()
    demo_qdrant_compatibility()
    demo_classification_impact()
    demo_metric_correlation()
    demo_recommendations()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
NEW METRICS ADDED TO HOLON:

Qdrant-Native (fast, server-side):
  ✓ Cosine (already had)
  ✓ Dot Product (already had)
  ✓ Euclidean (already had)
  ✓ Manhattan (already had)

VSA-Specific (client-side, optimal for bipolar):
  + Hamming: Count of bit differences
  + Overlap: Count of matching positions
  + Agreement: (agreements - disagreements) / D

Advanced:
  + Chebyshev: Max difference in any dimension
  + Minkowski: Generalized Lp distance
  + Weighted Cosine: Per-dimension importance
  + Weighted Euclidean: Per-dimension importance

For program synthesis (Challenge 009):
- Hamming is natural for bipolar vectors
- Weighted metrics enable learned importance
- Agreement gives balanced similarity view
    """)


if __name__ == "__main__":
    main()
