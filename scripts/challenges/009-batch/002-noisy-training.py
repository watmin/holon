#!/usr/bin/env python3
"""
Challenge 009-002: Training with Noisy Labels

Test how robust weight learning is to mislabeled training data.
This is crucial for real-world applicability where labels are often imperfect.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from holon import CPUStore

# Import from local modules
sys.path.insert(0, str(Path(__file__).parent))
from common import (
    generate_ticket_data,
    print_confusion_matrix,
    print_training_result,
    split_data,
    compute_accuracy,
)

# Import WeightedEncoder from the main module
from importlib import import_module
weighted_encoder_module = import_module("001-weighted-encoder")
WeightedEncoder = weighted_encoder_module.WeightedEncoder


def run_noise_experiment(noise_rate: float, n_samples: int = 1000):
    """Run training experiment with specified noise rate."""

    print(f"\n{'=' * 60}")
    print(f"NOISE RATE: {noise_rate:.0%}")
    print(f"{'=' * 60}")

    # Generate noisy data (returns both noisy and true labels)
    tickets, noisy_labels, true_labels = generate_ticket_data(
        n_samples=n_samples,
        noise_rate=noise_rate,
        seed=42,
    )

    # Count actual noise
    actual_noise = sum(1 for n, t in zip(noisy_labels, true_labels) if n != t)
    print(f"Train: {n_samples} ({actual_noise} mislabeled = {actual_noise/n_samples:.1%} noise)")

    # Split data
    indices = list(range(n_samples))
    import random
    random.seed(42)
    random.shuffle(indices)
    split_idx = int(0.8 * n_samples)

    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]

    X_train = [tickets[i] for i in train_idx]
    y_train_noisy = [noisy_labels[i] for i in train_idx]
    y_train_true = [true_labels[i] for i in train_idx]

    X_test = [tickets[i] for i in test_idx]
    y_test_true = [true_labels[i] for i in test_idx]

    # Count noise in training set
    train_noise = sum(1 for n, t in zip(y_train_noisy, y_train_true) if n != t)
    print(f"Training set: {len(X_train)} ({train_noise} mislabeled = {train_noise/len(X_train):.1%})")
    print(f"Test set: {len(X_test)} (clean labels)")

    # Create encoder
    store = CPUStore(dimensions=4096)
    encoder = WeightedEncoder(store, exclude_fields=["ticket_id"])

    # Train on NOISY labels (use random_search for speed)
    print("\nTraining with random_search (on noisy labels)...")
    result = encoder.fit(
        X_train, y_train_noisy,  # Train on noisy labels!
        X_val=None,  # Use cross-validation
        method="random_search",
        max_iter=50,  # Reduced iterations
        patience=15,
        regularization=0.15,  # Slightly higher regularization for noisy data
        cross_validate=True,
        n_folds=2,  # Fewer folds for speed
        verbose=False,
    )

    # Evaluate on CLEAN test labels (ground truth)
    prototypes = encoder.build_prototypes(X_train, y_train_noisy)
    y_pred = encoder.predict(X_test, prototypes)
    clean_accuracy = compute_accuracy(y_test_true, y_pred)

    # Also get baseline with uniform weights
    encoder_uniform = WeightedEncoder(store, exclude_fields=["ticket_id"])
    encoder_uniform.weights = {f: 1.0 for f in encoder.weights.keys()}
    prototypes_uniform = encoder_uniform.build_prototypes(X_train, y_train_noisy)
    y_pred_uniform = encoder_uniform.predict(X_test, prototypes_uniform)
    baseline_accuracy = compute_accuracy(y_test_true, y_pred_uniform)

    print(f"\nResults (evaluated on clean test labels):")
    print(f"  Baseline (uniform weights): {baseline_accuracy:.1%}")
    print(f"  Learned weights:            {clean_accuracy:.1%}")
    print(f"  Improvement:                {clean_accuracy - baseline_accuracy:+.1%}")
    print(f"  Training time:              {result.training_time_ms:.0f}ms")

    # Check weight quality
    signal_avg = (result.weights.get("type", 0) + result.weights.get("keywords", 0)) / 2
    noise_avg = (result.weights.get("noise_field_1", 0) + result.weights.get("noise_field_2", 0)) / 2

    print(f"\nWeight quality:")
    print(f"  Signal fields avg: {signal_avg:.2f}")
    print(f"  Noise fields avg:  {noise_avg:.2f}")
    print(f"  Signal/Noise ratio: {signal_avg/max(0.01, noise_avg):.1f}x")

    return {
        "noise_rate": noise_rate,
        "clean_accuracy": clean_accuracy,
        "baseline_accuracy": baseline_accuracy,
        "improvement": clean_accuracy - baseline_accuracy,
        "weights": result.weights,
        "signal_noise_ratio": signal_avg / max(0.01, noise_avg),
    }


def main():
    """Test training robustness across noise levels."""

    print("=" * 60)
    print("Challenge 009-002: Training with Noisy Labels")
    print("=" * 60)
    print("""
Goal: Verify that weight learning is robust to mislabeled training data.

We train on data where X% of labels are incorrect (mislabeled),
and evaluate on clean test data to measure true performance.

The key question: Does learned weighting STILL improve over uniform weights
even when training labels are noisy?
    """)

    # Test across noise levels (smaller samples for speed)
    noise_levels = [0.0, 0.10, 0.20, 0.30]
    results = []

    for noise_rate in noise_levels:
        result = run_noise_experiment(noise_rate, n_samples=500)  # Reduced for speed
        results.append(result)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: Noise Robustness")
    print("=" * 60)

    print("\n| Noise | Baseline | Learned | Improvement | S/N Ratio |")
    print("|-------|----------|---------|-------------|-----------|")

    for r in results:
        print(f"| {r['noise_rate']:>5.0%} | {r['baseline_accuracy']:>8.1%} | {r['clean_accuracy']:>7.1%} | {r['improvement']:>+11.1%} | {r['signal_noise_ratio']:>9.1f}x |")

    # Analysis
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)

    # Find the noise level where improvement drops below 0
    still_improving = [r for r in results if r["improvement"] > 0]

    if len(still_improving) == len(results):
        print("\n✓ Weight learning improves accuracy at ALL noise levels tested!")
        print(f"  Even at {results[-1]['noise_rate']:.0%} noise, still +{results[-1]['improvement']:.1%} improvement")
    else:
        breakpoint_noise = results[len(still_improving)]["noise_rate"]
        print(f"\n⚠ Weight learning stops helping at {breakpoint_noise:.0%} noise")
        print(f"  Beyond this, uniform weights are as good or better")

    # Check weight quality stability
    ratios = [r["signal_noise_ratio"] for r in results]
    print(f"\nWeight quality (Signal/Noise ratio) across noise levels:")
    print(f"  Range: {min(ratios):.1f}x - {max(ratios):.1f}x")
    print(f"  At 0% noise: {results[0]['signal_noise_ratio']:.1f}x")
    print(f"  At 30% noise: {results[-1]['signal_noise_ratio']:.1f}x")

    if results[-1]['signal_noise_ratio'] > 1.0:
        print("\n✓ Signal fields still weighted higher than noise even with 30% label noise!")
    else:
        print("\n⚠ At high noise, optimizer can't distinguish signal from noise fields")


if __name__ == "__main__":
    main()
