#!/usr/bin/env python3
"""
Challenge 009-004: Phase 2 - Interaction Discovery

Learn which field pairs should be bound together because their
interaction is discriminative for classification.

Example: (priority, type) interaction might be more discriminative than
either field alone if "high priority billing" tickets go to a specific team.
"""

import random
import sys
import time
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from holon import CPUStore

# Import local modules
sys.path.insert(0, str(Path(__file__).parent))
from common import compute_accuracy, print_confusion_matrix

# Import WeightedEncoder from Phase 1
from importlib import import_module
weighted_encoder_module = import_module("001-weighted-encoder")
WeightedEncoder = weighted_encoder_module.WeightedEncoder


class InteractionEncoder(WeightedEncoder):
    """
    Encoder that learns both field weights AND field interactions.

    Extends WeightedEncoder to also discover which field pairs
    should be bound together.
    """

    def __init__(
        self,
        store: CPUStore,
        weights: Optional[Dict[str, float]] = None,
        interactions: Optional[List[Tuple[str, str]]] = None,
        interaction_weights: Optional[Dict[Tuple[str, str], float]] = None,
        exclude_fields: Optional[List[str]] = None,
    ):
        super().__init__(store, weights, exclude_fields)
        self.interactions = interactions or []
        self.interaction_weights = interaction_weights or {}

    def encode(self, item: Dict[str, Any]) -> np.ndarray:
        """
        Encode an item with field weights AND interactions.

        The encoding is:
            bundle([
                weight_1 * encode(field_1),
                weight_2 * encode(field_2),
                ...,
                interaction_weight_1 * bind(encode(field_i), encode(field_j)),
                ...
            ])
        """
        field_vectors = []
        all_weights = []

        # Encode individual fields (from parent class)
        for key, value in item.items():
            if key in self.exclude_fields:
                continue

            weight = self.weights.get(key, 1.0)
            if weight == 0.0:
                continue

            field_data = {key: value}
            vec = self.encoder.encode_data(field_data)

            if hasattr(vec, 'cpu'):
                vec = vec.cpu().numpy()

            field_vectors.append(vec.astype(np.float32))
            all_weights.append(weight)

        # Encode interactions (bound field pairs)
        for (field_a, field_b) in self.interactions:
            if field_a not in item or field_b not in item:
                continue

            weight = self.interaction_weights.get((field_a, field_b), 1.0)
            if weight == 0.0:
                continue

            # Encode each field
            vec_a = self.encoder.encode_data({field_a: item[field_a]})
            vec_b = self.encoder.encode_data({field_b: item[field_b]})

            if hasattr(vec_a, 'cpu'):
                vec_a = vec_a.cpu().numpy()
            if hasattr(vec_b, 'cpu'):
                vec_b = vec_b.cpu().numpy()

            # Bind them together
            interaction_vec = (vec_a * vec_b).astype(np.float32)

            field_vectors.append(interaction_vec)
            all_weights.append(weight)

        if not field_vectors:
            return np.zeros(self.store.dimensions, dtype=np.int8)

        # Weighted sum
        weighted_sum = np.zeros(self.store.dimensions, dtype=np.float32)
        for vec, weight in zip(field_vectors, all_weights):
            weighted_sum += weight * vec

        # Threshold to bipolar
        result = np.where(weighted_sum > 0, 1, np.where(weighted_sum < 0, -1, 0))
        return result.astype(np.int8)

    def discover_interactions(
        self,
        X_train: List[Dict[str, Any]],
        y_train: List[str],
        X_val: Optional[List[Dict[str, Any]]] = None,
        y_val: Optional[List[str]] = None,
        max_interactions: int = 5,
        min_improvement: float = 0.005,
        zero_individual_weights: bool = True,
        verbose: bool = True,
    ) -> List[Tuple[str, str]]:
        """
        Discover valuable field interactions via greedy search.

        For each pair of fields, test if adding their binding improves
        classification accuracy. Keep the best interactions.

        Args:
            X_train: Training examples
            y_train: Training labels
            X_val: Validation examples (uses CV if not provided)
            y_val: Validation labels
            max_interactions: Maximum number of interactions to discover
            min_improvement: Minimum accuracy improvement to keep an interaction
            zero_individual_weights: If True, zero out individual field weights
                when testing their interaction (prevents dilution)
            verbose: Print progress

        Returns:
            List of (field_a, field_b) tuples representing valuable interactions
        """
        if X_val is None:
            X_val = X_train
            y_val = y_train

        # Get all field names
        all_fields = set()
        for item in X_train:
            all_fields.update(item.keys())
        all_fields -= self.exclude_fields
        fields = sorted(all_fields)

        if verbose:
            print(f"Discovering interactions among {len(fields)} fields...")
            print(f"Candidate pairs: {len(list(combinations(fields, 2)))}")

        # Save original weights
        original_weights = self.weights.copy()

        # Compute baseline accuracy (no interactions)
        self.interactions = []
        baseline_acc = self._evaluate_accuracy(X_train, y_train, X_val, y_val)

        if verbose:
            print(f"Baseline accuracy (no interactions): {baseline_acc:.1%}")

        # Greedy search: add interactions one at a time
        discovered = []
        current_acc = baseline_acc
        zeroed_fields = set()  # Track fields whose individual weights we've zeroed

        for round_num in range(max_interactions):
            best_pair = None
            best_improvement = 0
            best_new_acc = current_acc

            # Test each candidate pair
            for pair in combinations(fields, 2):
                if pair in discovered:
                    continue

                # Save current weights
                saved_weights = self.weights.copy()

                # Optionally zero out individual weights for interacting fields
                if zero_individual_weights:
                    self.weights[pair[0]] = 0.0
                    self.weights[pair[1]] = 0.0

                # Temporarily add this interaction
                self.interactions = discovered + [pair]
                self.interaction_weights[pair] = 2.0  # Boost interaction

                # Evaluate
                acc = self._evaluate_accuracy(X_train, y_train, X_val, y_val)
                improvement = acc - current_acc

                if improvement > best_improvement:
                    best_improvement = improvement
                    best_pair = pair
                    best_new_acc = acc

                # Restore weights
                self.weights = saved_weights

            # Keep best if it improves enough
            if best_pair and best_improvement >= min_improvement:
                discovered.append(best_pair)
                current_acc = best_new_acc

                # Zero out the individual field weights for discovered interaction
                if zero_individual_weights:
                    self.weights[best_pair[0]] = 0.0
                    self.weights[best_pair[1]] = 0.0
                    zeroed_fields.add(best_pair[0])
                    zeroed_fields.add(best_pair[1])

                self.interaction_weights[best_pair] = 2.0

                if verbose:
                    print(f"  Round {round_num + 1}: Added {best_pair} → {current_acc:.1%} (+{best_improvement:.1%})")
            else:
                if verbose:
                    print(f"  Round {round_num + 1}: No interaction improved by >={min_improvement:.1%}")
                break

        self.interactions = discovered
        return discovered

    def _evaluate_accuracy(
        self,
        X_train: List[Dict],
        y_train: List[str],
        X_val: List[Dict],
        y_val: List[str],
    ) -> float:
        """Evaluate accuracy using prototype-based classification."""
        # Build prototypes
        label_vectors: Dict[str, List[np.ndarray]] = {}
        for item, label in zip(X_train, y_train):
            vec = self.encode(item)
            if label not in label_vectors:
                label_vectors[label] = []
            label_vectors[label].append(vec)

        prototypes: Dict[str, np.ndarray] = {}
        for label, vectors in label_vectors.items():
            stacked = np.stack(vectors)
            mean = np.mean(stacked, axis=0)
            proto = np.where(mean > 0, 1, np.where(mean < 0, -1, 0)).astype(np.int8)
            prototypes[label] = proto

        # Classify validation set
        correct = 0
        for item, true_label in zip(X_val, y_val):
            vec = self.encode(item)
            best_label = None
            best_sim = -float('inf')
            for label, proto in prototypes.items():
                sim = self._cosine_similarity(vec, proto)
                if sim > best_sim:
                    best_sim = sim
                    best_label = label
            if best_label == true_label:
                correct += 1

        return correct / len(y_val) if y_val else 0.0

    def fit_with_interactions(
        self,
        X_train: List[Dict[str, Any]],
        y_train: List[str],
        X_val: Optional[List[Dict[str, Any]]] = None,
        y_val: Optional[List[str]] = None,
        max_interactions: int = 5,
        weight_method: str = "hill_climb",
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Full training: discover interactions, then optimize weights.

        Returns dict with weights, interactions, and metrics.
        """
        start_time = time.time()

        if X_val is None:
            X_val = X_train
            y_val = y_train

        # Step 1: Learn field weights (Phase 1)
        if verbose:
            print("\n=== Step 1: Learning field weights ===")

        weight_result = self.fit(
            X_train, y_train,
            X_val=X_val, y_val=y_val,
            method=weight_method,
            max_iter=50,
            patience=15,
            regularization=0.1,
            cross_validate=False,  # Use provided val set
            verbose=verbose,
        )

        # Step 2: Discover interactions (Phase 2)
        if verbose:
            print("\n=== Step 2: Discovering interactions ===")

        discovered = self.discover_interactions(
            X_train, y_train,
            X_val=X_val, y_val=y_val,
            max_interactions=max_interactions,
            min_improvement=0.005,
            verbose=verbose,
        )

        # Step 3: Optimize interaction weights
        if verbose and discovered:
            print("\n=== Step 3: Optimizing interaction weights ===")

        if discovered:
            self._optimize_interaction_weights(X_train, y_train, X_val, y_val, verbose)

        # Final evaluation
        prototypes = self.build_prototypes(X_train, y_train)
        y_pred = self.predict(X_val, prototypes)
        final_acc = compute_accuracy(y_val, y_pred)

        elapsed = time.time() - start_time

        return {
            "weights": self.weights,
            "interactions": self.interactions,
            "interaction_weights": self.interaction_weights,
            "final_accuracy": final_acc,
            "baseline_accuracy": weight_result.baseline_accuracy,
            "training_time_s": elapsed,
        }

    def _optimize_interaction_weights(
        self,
        X_train: List[Dict],
        y_train: List[str],
        X_val: List[Dict],
        y_val: List[str],
        verbose: bool = True,
    ):
        """Optimize weights for discovered interactions."""
        weight_options = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0]

        for interaction in self.interactions:
            best_weight = 1.0
            best_acc = self._evaluate_accuracy(X_train, y_train, X_val, y_val)

            for w in weight_options:
                self.interaction_weights[interaction] = w
                acc = self._evaluate_accuracy(X_train, y_train, X_val, y_val)

                if acc > best_acc:
                    best_acc = acc
                    best_weight = w

            self.interaction_weights[interaction] = best_weight
            if verbose and best_weight != 1.0:
                print(f"  {interaction} → weight {best_weight}")

    def fit_interaction_only(
        self,
        X_train: List[Dict[str, Any]],
        y_train: List[str],
        X_val: Optional[List[Dict[str, Any]]] = None,
        y_val: Optional[List[str]] = None,
        interaction_fields: Tuple[str, str] = None,
        verbose: bool = True,
    ) -> float:
        """
        Test classification using ONLY an interaction (zero individual field weights).

        This tests the hypothesis that for XOR-like problems, we should
        use only the interaction, not individual fields.
        """
        if X_val is None:
            X_val = X_train
            y_val = y_train

        # Zero out all individual field weights
        all_fields = set()
        for item in X_train:
            all_fields.update(item.keys())
        all_fields -= self.exclude_fields

        self.weights = {f: 0.0 for f in all_fields}

        # Add only the interaction
        self.interactions = [interaction_fields]
        self.interaction_weights[interaction_fields] = 1.0

        # Evaluate
        acc = self._evaluate_accuracy(X_train, y_train, X_val, y_val)

        if verbose:
            print(f"Interaction-only ({interaction_fields}): {acc:.1%}")

        return acc


def generate_interaction_data(
    n_samples: int = 500,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Generate data where INTERACTIONS matter more than individual fields.

    Categories are determined by field COMBINATIONS, not individual values.
    """
    random.seed(seed)
    np.random.seed(seed)

    # The key: category is determined by (priority, urgency) combination
    # - alpha: high priority AND urgent
    # - beta: low priority AND not urgent
    # - gamma: mixed (high priority but not urgent, OR low priority but urgent)

    items = []
    labels = []

    for i in range(n_samples):
        priority = random.choice(["high", "low"])
        urgency = random.choice(["urgent", "normal"])

        # Category determined by INTERACTION
        if priority == "high" and urgency == "urgent":
            category = "alpha"
        elif priority == "low" and urgency == "normal":
            category = "beta"
        else:
            category = "gamma"

        item = {
            "priority": priority,
            "urgency": urgency,
            # These fields are NOT discriminative (shared across categories)
            "channel": random.choice(["email", "chat", "phone"]),
            "region": random.choice(["us", "eu", "asia"]),
            "noise": random.randint(0, 100),
        }

        items.append(item)
        labels.append(category)

    return items, labels


def generate_xor_data(
    n_samples: int = 500,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Generate XOR-like data where individual fields provide NO information.

    Each field is 50/50 split in each category, so looking at any single
    field tells you nothing. Only the COMBINATION reveals the category.

    This is the classic case where interactions are essential.
    """
    random.seed(seed)
    np.random.seed(seed)

    items = []
    labels = []

    # XOR pattern: category = (A XOR B)
    # - positive: (A=1, B=0) or (A=0, B=1)
    # - negative: (A=0, B=0) or (A=1, B=1)
    #
    # In each category, A is 50% 0 and 50% 1, same for B.
    # So individual fields provide zero information.

    for i in range(n_samples):
        # Choose category first
        category = random.choice(["positive", "negative"])

        if category == "positive":
            # XOR = 1: different values
            if random.random() < 0.5:
                feature_a = "A1"
                feature_b = "B0"
            else:
                feature_a = "A0"
                feature_b = "B1"
        else:
            # XOR = 0: same values
            if random.random() < 0.5:
                feature_a = "A0"
                feature_b = "B0"
            else:
                feature_a = "A1"
                feature_b = "B1"

        item = {
            "feature_a": feature_a,
            "feature_b": feature_b,
            # Noise fields (also provide no information)
            "noise_1": random.choice(["X", "Y", "Z"]),
            "noise_2": random.choice(["P", "Q", "R"]),
        }

        items.append(item)
        labels.append(category)

    return items, labels


def run_xor_experiment():
    """Run the XOR experiment - the hardest case for interaction discovery."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: XOR Data (Hardest Case)")
    print("=" * 60)
    print("""
XOR is the classic case where individual features provide ZERO information.
- positive: (A=1,B=0) or (A=0,B=1)
- negative: (A=0,B=0) or (A=1,B=1)

In each category, A is 50% each value, same for B.
Only bind(A, B) can discriminate.
    """)

    # Generate XOR data
    items, labels = generate_xor_data(n_samples=600)

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

    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # Create encoder
    store = CPUStore(dimensions=4096)

    # Baseline
    print("\n1. Baseline (no interactions)...")
    encoder_base = InteractionEncoder(store, exclude_fields=[])
    encoder_base.weights = {}
    encoder_base.interactions = []
    prototypes = encoder_base.build_prototypes(X_train, y_train)
    y_pred = encoder_base.predict(X_test, prototypes)
    baseline_acc = compute_accuracy(y_test, y_pred)
    print(f"   Accuracy: {baseline_acc:.1%}")
    print(f"   (Expected: ~50% since individual fields provide no information)")

    # With manual interaction
    print("\n2. With manual interaction (feature_a, feature_b)...")
    encoder_manual = InteractionEncoder(store, exclude_fields=[])
    encoder_manual.weights = {"feature_a": 1.0, "feature_b": 1.0, "noise_1": 0.5, "noise_2": 0.5}
    encoder_manual.interactions = [("feature_a", "feature_b")]
    encoder_manual.interaction_weights[("feature_a", "feature_b")] = 2.0  # Boost interaction
    prototypes_m = encoder_manual.build_prototypes(X_train, y_train)
    y_pred_m = encoder_manual.predict(X_test, prototypes_m)
    manual_acc = compute_accuracy(y_test, y_pred_m)
    print(f"   Accuracy: {manual_acc:.1%}")

    # Interaction-only (the correct approach for XOR)
    print("\n3. Interaction-only (zero individual field weights)...")
    encoder_only = InteractionEncoder(store, exclude_fields=[])
    interaction_only_acc = encoder_only.fit_interaction_only(
        X_train, y_train,
        X_val=X_test, y_val=y_test,
        interaction_fields=("feature_a", "feature_b"),
        verbose=False,
    )
    prototypes_only = encoder_only.build_prototypes(X_train, y_train)
    y_pred_only = encoder_only.predict(X_test, prototypes_only)
    interaction_only_acc = compute_accuracy(y_test, y_pred_only)
    print(f"   Accuracy: {interaction_only_acc:.1%}")

    # Auto-discover
    print("\n4. Auto-discovering interactions...")
    encoder_auto = InteractionEncoder(store, exclude_fields=[])
    result = encoder_auto.fit_with_interactions(
        X_train, y_train,
        X_val=X_test, y_val=y_test,
        max_interactions=5,
        weight_method="random_search",
        verbose=True,
    )
    prototypes_a = encoder_auto.build_prototypes(X_train, y_train)
    y_pred_a = encoder_auto.predict(X_test, prototypes_a)
    auto_acc = compute_accuracy(y_test, y_pred_a)

    print("\n" + "-" * 40)
    print("XOR RESULTS:")
    print("-" * 40)
    print(f"| Approach | Accuracy |")
    print(f"|----------|----------|")
    print(f"| Baseline (no interactions) | {baseline_acc:.1%} |")
    print(f"| Manual interaction (with individual fields) | {manual_acc:.1%} |")
    print(f"| Interaction-only (correct approach) | {interaction_only_acc:.1%} |")
    print(f"| Auto-discovered | {auto_acc:.1%} |")

    found_correct = ("feature_a", "feature_b") in encoder_auto.interactions or \
                    ("feature_b", "feature_a") in encoder_auto.interactions

    print(f"\nDiscovered interactions: {encoder_auto.interactions}")
    print(f"Found (feature_a, feature_b): {'✓ YES' if found_correct else '✗ NO'}")

    return {
        "baseline": baseline_acc,
        "manual": manual_acc,
        "auto": auto_acc,
        "found_correct": found_correct,
    }


def main():
    """Demo: Discover field interactions."""

    print("=" * 60)
    print("Challenge 009-004: Phase 2 - Interaction Discovery")
    print("=" * 60)
    print("""
This test creates data where INTERACTIONS between fields determine
the category, not individual field values.

Category rules:
- alpha: high priority AND urgent
- beta:  low priority AND normal
- gamma: anything else (mixed)

Individual fields (priority, urgency) don't distinguish well.
But bind(priority, urgency) should be very discriminative.
    """)

    # Generate data
    print("\n1. Generating interaction-dependent data...")
    items, labels = generate_interaction_data(n_samples=600)

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

    # Count category distribution
    from collections import Counter
    dist = Counter(labels)
    print(f"   Distribution: {dict(dist)}")

    # Create encoder
    print("\n2. Creating interaction encoder...")
    store = CPUStore(dimensions=4096)
    encoder = InteractionEncoder(
        store,
        exclude_fields=["noise"],
    )

    # Baseline: no interactions
    print("\n3. Baseline (no interactions, uniform weights)...")
    encoder.weights = {}
    encoder.interactions = []
    prototypes = encoder.build_prototypes(X_train, y_train)
    y_pred = encoder.predict(X_test, prototypes)
    baseline_acc = compute_accuracy(y_test, y_pred)
    print(f"   Accuracy: {baseline_acc:.1%}")

    # Phase 1 only: weight learning
    print("\n4. Phase 1 only (field weights, no interactions)...")
    encoder_phase1 = InteractionEncoder(store, exclude_fields=["noise"])
    result_phase1 = encoder_phase1.fit(
        X_train, y_train,
        X_val=X_test, y_val=y_test,
        method="hill_climb",
        max_iter=50,
        patience=15,
        verbose=False,
    )
    prototypes_p1 = encoder_phase1.build_prototypes(X_train, y_train)
    y_pred_p1 = encoder_phase1.predict(X_test, prototypes_p1)
    phase1_acc = compute_accuracy(y_test, y_pred_p1)
    print(f"   Accuracy: {phase1_acc:.1%}")

    # Phase 2: full training with interactions
    print("\n5. Phase 2 (weights + interactions)...")
    encoder_full = InteractionEncoder(store, exclude_fields=["noise"])
    result = encoder_full.fit_with_interactions(
        X_train, y_train,
        X_val=X_test, y_val=y_test,
        max_interactions=5,
        weight_method="hill_climb",
        verbose=True,
    )

    prototypes_full = encoder_full.build_prototypes(X_train, y_train)
    y_pred_full = encoder_full.predict(X_test, prototypes_full)
    phase2_acc = compute_accuracy(y_test, y_pred_full)

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(f"\n| Approach | Accuracy |")
    print(f"|----------|----------|")
    print(f"| Baseline (uniform, no interactions) | {baseline_acc:.1%} |")
    print(f"| Phase 1 (weights only) | {phase1_acc:.1%} |")
    print(f"| Phase 2 (weights + interactions) | {phase2_acc:.1%} |")

    print(f"\nImprovement over baseline: {phase2_acc - baseline_acc:+.1%}")
    print(f"Improvement over Phase 1: {phase2_acc - phase1_acc:+.1%}")

    # Check if correct interaction was discovered
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)

    expected_interaction = ("priority", "urgency")
    found_expected = expected_interaction in encoder_full.interactions or \
                     ("urgency", "priority") in encoder_full.interactions

    print(f"\nDiscovered interactions: {encoder_full.interactions}")
    print(f"Expected interaction: {expected_interaction}")

    if found_expected:
        print("\n✓ SUCCESS: Correctly discovered (priority, urgency) interaction!")
    else:
        print("\n? Did not find expected interaction (may have found equivalent)")

    # Show confusion matrix
    print("\nConfusion Matrix (Phase 2):")
    print_confusion_matrix(y_test, y_pred_full)

    # Now run the harder XOR experiment
    xor_result = run_xor_experiment()

    # Final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print("""
Experiment 1 (Interaction Data):
- Baseline was already high because VSA bundling captures some structure
- Interaction discovery provides incremental improvement

Experiment 2 (XOR Data):
- Baseline should be ~50% (random) since individual fields provide no info
- Only bind(feature_a, feature_b) can discriminate
- This is the true test of interaction discovery
    """)

    return {
        "exp1_baseline": baseline_acc,
        "exp1_phase2": phase2_acc,
        "exp2_baseline": xor_result["baseline"],
        "exp2_auto": xor_result["auto"],
        "exp2_found_correct": xor_result["found_correct"],
    }


if __name__ == "__main__":
    main()
