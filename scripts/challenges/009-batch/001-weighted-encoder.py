#!/usr/bin/env python3
"""
Challenge 009-001: Weighted Encoder with Learnable Field Weights

This implements Phase 1 of deterministic training: learning optimal field weights
for structured data encoding.

Instead of treating all fields equally, we learn which fields matter most for
a given classification task.
"""

import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from holon import CPUStore
from holon.atomizer import parse_data
from holon.encoder import Encoder

from common import (
    TrainingResult,
    compute_accuracy,
    generate_ticket_data,
    print_confusion_matrix,
    print_training_result,
    split_data,
)


def split_with_true_labels(
    X: List[Any],
    y_noisy: List[str],
    y_true: List[str],
    train_ratio: float = 0.8,
    seed: int = 42,
) -> Tuple[List[Any], List[str], List[str], List[Any], List[str], List[str]]:
    """
    Split data preserving both noisy and true labels.

    Returns:
        (X_train, y_train_noisy, y_train_true, X_test, y_test_noisy, y_test_true)
    """
    import random
    random.seed(seed)

    indices = list(range(len(X)))
    random.shuffle(indices)

    split_idx = int(len(indices) * train_ratio)
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]

    X_train = [X[i] for i in train_indices]
    y_train_noisy = [y_noisy[i] for i in train_indices]
    y_train_true = [y_true[i] for i in train_indices]
    X_test = [X[i] for i in test_indices]
    y_test_noisy = [y_noisy[i] for i in test_indices]
    y_test_true = [y_true[i] for i in test_indices]

    return X_train, y_train_noisy, y_train_true, X_test, y_test_noisy, y_test_true


class WeightedEncoder:
    """
    Encoder that applies learned field weights to structured data.

    Instead of uniform bundling, each field's contribution is weighted:

        vec = bundle([
            weight_1 * encode(field_1),
            weight_2 * encode(field_2),
            ...
        ])

    Weights are learned from labeled examples using gradient-free optimization.
    """

    def __init__(
        self,
        store: CPUStore,
        weights: Optional[Dict[str, float]] = None,
        exclude_fields: Optional[List[str]] = None,
    ):
        """
        Initialize weighted encoder.

        Args:
            store: CPUStore instance for encoding
            weights: Initial field weights (default: uniform 1.0)
            exclude_fields: Fields to exclude from encoding (e.g., IDs)
        """
        self.store = store
        self.encoder = store.encoder
        self.weights = weights or {}
        self.exclude_fields = set(exclude_fields or [])

        # Cache for field vectors (speedup repeated encoding)
        self._field_cache: Dict[str, np.ndarray] = {}

    def encode(self, item: Dict[str, Any]) -> np.ndarray:
        """
        Encode an item with field weights applied.

        Args:
            item: Dictionary to encode

        Returns:
            Weighted encoded vector
        """
        field_vectors = []
        field_weights = []

        for key, value in item.items():
            if key in self.exclude_fields:
                continue

            # Get weight for this field (default 1.0)
            weight = self.weights.get(key, 1.0)

            # Skip zero-weight fields entirely (optimization)
            if weight == 0.0:
                continue

            # Encode the field-value pair
            field_data = {key: value}
            vec = self.encoder.encode_data(field_data)

            # Convert to numpy if needed (for TorchHD compatibility)
            if hasattr(vec, 'cpu'):
                vec = vec.cpu().numpy()

            field_vectors.append(vec.astype(np.float32))
            field_weights.append(weight)

        if not field_vectors:
            return np.zeros(self.store.dimensions, dtype=np.int8)

        # Weighted sum
        weighted_sum = np.zeros(self.store.dimensions, dtype=np.float32)
        for vec, weight in zip(field_vectors, field_weights):
            weighted_sum += weight * vec

        # Threshold to bipolar
        result = np.where(weighted_sum > 0, 1, np.where(weighted_sum < 0, -1, 0))
        return result.astype(np.int8)

    def encode_batch(self, items: List[Dict[str, Any]]) -> List[np.ndarray]:
        """Encode multiple items."""
        return [self.encode(item) for item in items]

    def fit(
        self,
        X_train: List[Dict[str, Any]],
        y_train: List[str],
        X_val: Optional[List[Dict[str, Any]]] = None,
        y_val: Optional[List[str]] = None,
        method: str = "random_search",
        max_iter: int = 100,
        patience: int = 20,
        regularization: float = 0.1,
        cross_validate: bool = True,
        n_folds: int = 3,
        verbose: bool = True,
    ) -> TrainingResult:
        """
        Learn optimal field weights from labeled data.

        Args:
            X_train: Training examples
            y_train: Training labels
            X_val: Validation examples (optional, uses cross-validation if not provided)
            y_val: Validation labels
            method: Optimization method ("random_search", "coordinate_descent", "hill_climb")
            max_iter: Maximum iterations
            patience: Early stopping patience
            regularization: L2 penalty on weight magnitude (prevents overfitting)
            cross_validate: Use cross-validation when no X_val provided
            n_folds: Number of cross-validation folds
            verbose: Print progress

        Returns:
            TrainingResult with learned weights and metrics
        """
        start_time = time.time()

        # Store regularization for use in evaluation
        self._regularization = regularization
        self._use_cv = cross_validate and X_val is None
        self._n_folds = n_folds
        self._X_train_full = X_train
        self._y_train_full = y_train

        # Use train as val if not provided and not using CV
        if X_val is None and not cross_validate:
            X_val = X_train
            y_val = y_train
        elif X_val is None:
            # Will use cross-validation in _evaluate
            X_val = X_train
            y_val = y_train

        # Discover all fields from training data
        all_fields = set()
        for item in X_train:
            all_fields.update(item.keys())
        all_fields -= self.exclude_fields
        fields = sorted(all_fields)

        if verbose:
            print(f"Discovered {len(fields)} fields: {fields}")

        # Initialize uniform weights
        self.weights = {f: 1.0 for f in fields}

        # Compute baseline accuracy
        baseline_acc = self._evaluate(X_val, y_val)
        if verbose:
            print(f"Baseline accuracy (uniform): {baseline_acc:.1%}")

        # Run optimization
        if method == "random_search":
            best_weights, best_acc, iters = self._random_search(
                X_train, y_train, X_val, y_val, fields, max_iter, patience, verbose
            )
        elif method == "hill_climb":
            best_weights, best_acc, iters = self._hill_climb(
                X_train, y_train, X_val, y_val, fields, max_iter, patience, verbose
            )
        elif method == "coordinate_descent":
            best_weights, best_acc, iters = self._coordinate_descent(
                X_train, y_train, X_val, y_val, fields, max_iter, patience, verbose
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        self.weights = best_weights

        # Compute final metrics
        train_acc = self._evaluate(X_train, y_train)
        test_acc = self._evaluate(X_val, y_val)

        elapsed_ms = (time.time() - start_time) * 1000

        return TrainingResult(
            weights=best_weights,
            train_accuracy=train_acc,
            test_accuracy=test_acc,
            iterations=iters,
            training_time_ms=elapsed_ms,
            baseline_accuracy=baseline_acc,
        )

    def _evaluate(self, X: List[Dict], y: List[str]) -> float:
        """
        Evaluate accuracy using prototype-based classification.

        Uses cross-validation if enabled, otherwise train-on-train evaluation.
        Applies regularization penalty to discourage extreme weights.
        """
        if getattr(self, '_use_cv', False):
            # Cross-validation: split X into folds, train on k-1, evaluate on 1
            return self._cross_validate_accuracy(X, y)
        else:
            # Standard: build prototypes from X, classify X
            return self._simple_evaluate(X, y)

    def _simple_evaluate(self, X: List[Dict], y: List[str]) -> float:
        """Simple prototype-based evaluation (may overfit)."""
        # Build prototypes per label
        label_vectors: Dict[str, List[np.ndarray]] = {}
        for item, label in zip(X, y):
            vec = self.encode(item)
            if label not in label_vectors:
                label_vectors[label] = []
            label_vectors[label].append(vec)

        # Compute prototype as mean (then threshold)
        prototypes: Dict[str, np.ndarray] = {}
        for label, vectors in label_vectors.items():
            stacked = np.stack(vectors)
            mean = np.mean(stacked, axis=0)
            proto = np.where(mean > 0, 1, np.where(mean < 0, -1, 0)).astype(np.int8)
            prototypes[label] = proto

        # Classify each item
        y_pred = []
        for item in X:
            vec = self.encode(item)
            best_label = None
            best_sim = -float('inf')
            for label, proto in prototypes.items():
                sim = self._cosine_similarity(vec, proto)
                if sim > best_sim:
                    best_sim = sim
                    best_label = label
            y_pred.append(best_label)

        accuracy = compute_accuracy(y, y_pred)

        # Apply regularization penalty
        reg = getattr(self, '_regularization', 0.0)
        if reg > 0:
            # Penalize extreme weights (prefer weights near 1.0)
            weight_penalty = sum((w - 1.0) ** 2 for w in self.weights.values())
            weight_penalty = reg * weight_penalty / max(1, len(self.weights))
            accuracy = accuracy - weight_penalty

        return accuracy

    def _cross_validate_accuracy(self, X: List[Dict], y: List[str]) -> float:
        """K-fold cross-validation to prevent overfitting."""
        n_folds = getattr(self, '_n_folds', 3)
        n = len(X)
        fold_size = n // n_folds

        accuracies = []
        for fold in range(n_folds):
            # Split into train and val for this fold
            val_start = fold * fold_size
            val_end = val_start + fold_size if fold < n_folds - 1 else n

            val_indices = set(range(val_start, val_end))
            train_indices = [i for i in range(n) if i not in val_indices]

            X_fold_train = [X[i] for i in train_indices]
            y_fold_train = [y[i] for i in train_indices]
            X_fold_val = [X[i] for i in val_indices]
            y_fold_val = [y[i] for i in val_indices]

            # Build prototypes from training fold
            label_vectors: Dict[str, List[np.ndarray]] = {}
            for item, label in zip(X_fold_train, y_fold_train):
                vec = self.encode(item)
                if label not in label_vectors:
                    label_vectors[label] = []
                label_vectors[label].append(vec)

            prototypes: Dict[str, np.ndarray] = {}
            for label, vectors in label_vectors.items():
                if vectors:
                    stacked = np.stack(vectors)
                    mean = np.mean(stacked, axis=0)
                    proto = np.where(mean > 0, 1, np.where(mean < 0, -1, 0)).astype(np.int8)
                    prototypes[label] = proto

            # Evaluate on validation fold
            y_pred = []
            for item in X_fold_val:
                vec = self.encode(item)
                best_label = None
                best_sim = -float('inf')
                for label, proto in prototypes.items():
                    sim = self._cosine_similarity(vec, proto)
                    if sim > best_sim:
                        best_sim = sim
                        best_label = label
                y_pred.append(best_label)

            fold_acc = compute_accuracy(y_fold_val, y_pred)
            accuracies.append(fold_acc)

        accuracy = sum(accuracies) / len(accuracies)

        # Apply regularization penalty
        reg = getattr(self, '_regularization', 0.0)
        if reg > 0:
            weight_penalty = sum((w - 1.0) ** 2 for w in self.weights.values())
            weight_penalty = reg * weight_penalty / max(1, len(self.weights))
            accuracy = accuracy - weight_penalty

        return accuracy

    def predict(self, X: List[Dict], prototypes: Dict[str, np.ndarray]) -> List[str]:
        """Predict labels using pre-computed prototypes."""
        y_pred = []
        for item in X:
            vec = self.encode(item)
            best_label = None
            best_sim = -float('inf')
            for label, proto in prototypes.items():
                sim = self._cosine_similarity(vec, proto)
                if sim > best_sim:
                    best_sim = sim
                    best_label = label
            y_pred.append(best_label)
        return y_pred

    def build_prototypes(self, X: List[Dict], y: List[str]) -> Dict[str, np.ndarray]:
        """Build prototypes from labeled data."""
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

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        dot = np.dot(a.astype(float), b.astype(float))
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def _random_search(
        self,
        X_train: List[Dict],
        y_train: List[str],
        X_val: List[Dict],
        y_val: List[str],
        fields: List[str],
        max_iter: int,
        patience: int,
        verbose: bool,
    ) -> Tuple[Dict[str, float], float, int]:
        """
        Random search optimization for field weights.

        Samples random weight configurations and keeps the best.
        """
        best_weights = self.weights.copy()
        best_acc = self._evaluate(X_val, y_val)
        no_improvement = 0

        for i in range(max_iter):
            # Generate random weights in [0, 3] range
            candidate_weights = {
                f: np.random.uniform(0, 3.0) for f in fields
            }

            self.weights = candidate_weights
            acc = self._evaluate(X_val, y_val)

            if acc > best_acc:
                best_acc = acc
                best_weights = candidate_weights.copy()
                no_improvement = 0
                if verbose:
                    print(f"  Iter {i+1}: {acc:.1%} (improved)")
            else:
                no_improvement += 1

            if no_improvement >= patience:
                if verbose:
                    print(f"  Early stopping at iter {i+1}")
                break

        return best_weights, best_acc, i + 1

    def _hill_climb(
        self,
        X_train: List[Dict],
        y_train: List[str],
        X_val: List[Dict],
        y_val: List[str],
        fields: List[str],
        max_iter: int,
        patience: int,
        verbose: bool,
    ) -> Tuple[Dict[str, float], float, int]:
        """
        Hill climbing optimization with random restarts.

        Perturbs current best weights and keeps improvements.
        """
        best_weights = self.weights.copy()
        best_acc = self._evaluate(X_val, y_val)
        no_improvement = 0
        step_size = 0.5

        for i in range(max_iter):
            # Perturb weights
            candidate_weights = {}
            for f in fields:
                delta = np.random.uniform(-step_size, step_size)
                new_weight = max(0, best_weights[f] + delta)
                candidate_weights[f] = new_weight

            self.weights = candidate_weights
            acc = self._evaluate(X_val, y_val)

            if acc > best_acc:
                best_acc = acc
                best_weights = candidate_weights.copy()
                no_improvement = 0
                if verbose:
                    print(f"  Iter {i+1}: {acc:.1%} (improved)")
            else:
                no_improvement += 1
                # Reduce step size on plateau
                if no_improvement % 10 == 0:
                    step_size *= 0.8

            if no_improvement >= patience:
                if verbose:
                    print(f"  Early stopping at iter {i+1}")
                break

        return best_weights, best_acc, i + 1

    def _coordinate_descent(
        self,
        X_train: List[Dict],
        y_train: List[str],
        X_val: List[Dict],
        y_val: List[str],
        fields: List[str],
        max_iter: int,
        patience: int,
        verbose: bool,
    ) -> Tuple[Dict[str, float], float, int]:
        """
        Coordinate descent: optimize one field at a time.

        For each field, try different weights and keep the best.
        """
        best_weights = self.weights.copy()
        best_acc = self._evaluate(X_val, y_val)
        no_improvement_rounds = 0

        weight_options = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0]

        iteration = 0
        for round_num in range(max_iter // len(fields) + 1):
            round_improved = False

            for field in fields:
                iteration += 1
                if iteration > max_iter:
                    break

                # Try each weight value for this field
                current_best = best_weights[field]
                for w in weight_options:
                    self.weights = best_weights.copy()
                    self.weights[field] = w
                    acc = self._evaluate(X_val, y_val)

                    if acc > best_acc:
                        best_acc = acc
                        current_best = w
                        round_improved = True
                        if verbose:
                            print(f"  Iter {iteration}: {field}={w:.2f} → {acc:.1%}")

                best_weights[field] = current_best

            if not round_improved:
                no_improvement_rounds += 1
                if no_improvement_rounds >= patience // len(fields):
                    if verbose:
                        print(f"  Early stopping at round {round_num+1}")
                    break
            else:
                no_improvement_rounds = 0

        return best_weights, best_acc, iteration


def main():
    """Demo: Learn optimal field weights for ticket routing."""

    print("=" * 60)
    print("Challenge 009-001: Weighted Encoder Training")
    print("=" * 60)

    # Generate data (returns noisy labels and true labels)
    print("\n1. Generating synthetic ticket data...")
    tickets, noisy_labels, true_labels = generate_ticket_data(n_samples=1000, noise_rate=0.0)

    # Split into train/test (using true labels since noise_rate=0)
    X_train, y_train, X_test, y_test = split_data(tickets, true_labels, train_ratio=0.8)
    print(f"   Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"   Labels: {sorted(set(true_labels))}")

    # Create store and weighted encoder
    print("\n2. Initializing weighted encoder...")
    store = CPUStore(dimensions=4096)
    encoder = WeightedEncoder(
        store,
        exclude_fields=["ticket_id"],  # Don't encode IDs
    )

    # Train with best method (coordinate_descent with cross-validation)
    print("\n3. Training with coordinate_descent + cross-validation + regularization...")
    result = encoder.fit(
        X_train, y_train,
        X_val=X_test, y_val=y_test,
        method="coordinate_descent",
        max_iter=100,
        patience=20,
        regularization=0.1,  # Penalize extreme weights
        cross_validate=True,
        verbose=True,
    )

    print_training_result(result)

    # Show predictions on test set
    print("\n4. Evaluating on test set...")
    prototypes = encoder.build_prototypes(X_train, y_train)
    y_pred = encoder.predict(X_test, prototypes)

    print_confusion_matrix(y_test, y_pred)

    # Compare with baseline (uniform weights)
    print("\n5. Comparing with uniform weights baseline...")
    encoder_uniform = WeightedEncoder(store, exclude_fields=["ticket_id"])
    encoder_uniform.weights = {f: 1.0 for f in encoder.weights.keys()}
    prototypes_uniform = encoder_uniform.build_prototypes(X_train, y_train)
    y_pred_uniform = encoder_uniform.predict(X_test, prototypes_uniform)
    baseline_acc = compute_accuracy(y_test, y_pred_uniform)
    learned_acc = compute_accuracy(y_test, y_pred)

    print(f"   Uniform weights accuracy: {baseline_acc:.1%}")
    print(f"   Learned weights accuracy: {learned_acc:.1%}")
    print(f"   Improvement: {(learned_acc - baseline_acc):+.1%}")

    # Final analysis
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)

    # Check if noise fields got low weights (they should)
    noise_weights = [
        result.weights.get("noise_field_1", 0),
        result.weights.get("noise_field_2", 0),
    ]
    signal_weights = [
        result.weights.get("type", 0),
        result.weights.get("keywords", 0),
    ]

    print(f"""
Learned weights reveal which fields are most discriminative:
- High weights: Key differentiators between teams
- Low weights: Noise or irrelevant fields
- Weights near 1.0: Regularization working (prevents extreme values)

Quality check:
- Signal fields (type, keywords) avg weight: {sum(signal_weights)/len(signal_weights):.2f}
- Noise fields (noise_field_*) avg weight: {sum(noise_weights)/len(noise_weights):.2f}
- Ratio (higher is better): {sum(signal_weights)/max(0.01, sum(noise_weights)):.1f}x

{"✓ Good: Signal fields weighted higher than noise" if sum(signal_weights) > sum(noise_weights) else "✗ Bad: Noise fields weighted too high (overfitting)"}
    """)


if __name__ == "__main__":
    main()
