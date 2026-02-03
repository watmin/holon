#!/usr/bin/env python3
"""
Challenge 009-003: Hard Categories (Overlapping Classes)

The previous tests used well-separated categories. Real data often has
overlapping classes where the same keywords appear in multiple categories.

This test creates deliberately confusing data to see if weight learning
can find the distinguishing features.
"""

import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from holon import CPUStore

# Import local modules
sys.path.insert(0, str(Path(__file__).parent))
from common import compute_accuracy, print_confusion_matrix

# Import WeightedEncoder
from importlib import import_module
weighted_encoder_module = import_module("001-weighted-encoder")
WeightedEncoder = weighted_encoder_module.WeightedEncoder


def generate_overlapping_data(
    n_samples: int = 500,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Generate data with overlapping categories.

    Categories share many features, making classification harder.
    Only ONE field is truly discriminative.
    """
    random.seed(seed)
    np.random.seed(seed)

    # Categories with heavy overlap
    # The KEY discriminator is the "type" field - everything else is shared
    categories = {
        "alpha": {
            "type": ["alpha_specific", "alpha_only"],  # UNIQUE
            "shared_keyword": ["common", "shared", "both", "overlap"],  # SHARED
            "noise": ["x", "y", "z", "w"],
        },
        "beta": {
            "type": ["beta_specific", "beta_only"],  # UNIQUE
            "shared_keyword": ["common", "shared", "both", "overlap"],  # SHARED
            "noise": ["x", "y", "z", "w"],
        },
        "gamma": {
            "type": ["gamma_specific", "gamma_only"],  # UNIQUE
            "shared_keyword": ["common", "shared", "both", "overlap"],  # SHARED
            "noise": ["x", "y", "z", "w"],
        },
    }

    items = []
    labels = []
    category_names = list(categories.keys())

    for i in range(n_samples):
        cat = random.choice(category_names)
        cat_def = categories[cat]

        item = {
            "type": random.choice(cat_def["type"]),  # Discriminative
            "shared_keyword": random.choice(cat_def["shared_keyword"]),  # NOT discriminative
            "noise_1": random.choice(cat_def["noise"]),  # NOT discriminative
            "noise_2": random.randint(0, 100),  # NOT discriminative
            "noise_3": f"item_{i}",  # Unique per item (NOT discriminative)
            # Add more overlapping fields
            "status": random.choice(["active", "pending", "done"]),  # Shared
            "priority": random.choice(["low", "medium", "high"]),  # Shared
        }

        items.append(item)
        labels.append(cat)

    return items, labels


def run_experiment():
    """Run hard category classification experiment."""

    print("=" * 60)
    print("Challenge 009-003: Hard Categories (Overlapping Classes)")
    print("=" * 60)
    print("""
This test creates data where categories share MOST features.
Only the 'type' field is truly discriminative.

Can weight learning discover that 'type' matters while
'shared_keyword', 'noise_*', 'status', 'priority' don't?
    """)

    # Generate data
    print("\n1. Generating overlapping category data...")
    items, labels = generate_overlapping_data(n_samples=500)

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
    print(f"   Categories: {sorted(set(labels))}")

    # Create store and encoder
    print("\n2. Creating weighted encoder...")
    store = CPUStore(dimensions=4096)
    encoder = WeightedEncoder(
        store,
        exclude_fields=["noise_3"],  # Exclude unique IDs
    )

    # Baseline with uniform weights
    print("\n3. Baseline (uniform weights)...")
    encoder.weights = {}  # Will default to 1.0
    prototypes_uniform = encoder.build_prototypes(X_train, y_train)
    y_pred_uniform = encoder.predict(X_test, prototypes_uniform)
    baseline_acc = compute_accuracy(y_test, y_pred_uniform)
    print(f"   Accuracy: {baseline_acc:.1%}")

    # Train weights
    print("\n4. Training weights...")
    result = encoder.fit(
        X_train, y_train,
        method="coordinate_descent",  # Best for finding specific important fields
        max_iter=100,
        patience=20,
        regularization=0.05,  # Light regularization
        cross_validate=True,
        n_folds=3,
        verbose=True,
    )

    # Evaluate
    print("\n5. Evaluation...")
    prototypes_learned = encoder.build_prototypes(X_train, y_train)
    y_pred_learned = encoder.predict(X_test, prototypes_learned)
    learned_acc = compute_accuracy(y_test, y_pred_learned)

    print(f"\nResults:")
    print(f"  Baseline (uniform): {baseline_acc:.1%}")
    print(f"  Learned weights:    {learned_acc:.1%}")
    print(f"  Improvement:        {learned_acc - baseline_acc:+.1%}")

    # Analyze weights
    print("\n6. Weight Analysis:")
    print("\n   Expected: 'type' should have HIGH weight (discriminative)")
    print("   Expected: other fields should have LOW weight (non-discriminative)")

    sorted_weights = sorted(result.weights.items(), key=lambda x: -x[1])
    print("\n   Learned weights:")
    for field, weight in sorted_weights:
        indicator = "← DISCRIMINATIVE" if field == "type" else ""
        bar = "█" * int(weight * 10)
        print(f"   {field:20} {weight:5.2f} {bar} {indicator}")

    # Check if type got highest weight
    type_weight = result.weights.get("type", 0)
    other_weights = [w for f, w in result.weights.items() if f != "type"]
    max_other = max(other_weights) if other_weights else 0

    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)

    if type_weight > max_other:
        print(f"""
✓ SUCCESS: Weight learning correctly identified 'type' as most important!
  - 'type' weight: {type_weight:.2f}
  - Max other weight: {max_other:.2f}
  - Ratio: {type_weight/max(0.01, max_other):.1f}x

This demonstrates symbolic program synthesis can discover
the discriminative structure in overlapping data.
        """)
    else:
        print(f"""
✗ FAILED: Weight learning did not identify 'type' as most important.
  - 'type' weight: {type_weight:.2f}
  - Max other weight: {max_other:.2f}

The optimizer may need:
  - More iterations
  - Different search method
  - Better regularization
        """)

    # Show confusion matrix
    print("\nConfusion Matrix (learned weights):")
    print_confusion_matrix(y_test, y_pred_learned)

    return {
        "baseline_acc": baseline_acc,
        "learned_acc": learned_acc,
        "type_weight": type_weight,
        "max_other_weight": max_other,
        "success": type_weight > max_other,
    }


if __name__ == "__main__":
    run_experiment()
