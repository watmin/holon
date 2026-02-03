#!/usr/bin/env python3
"""
Common utilities for Challenge 009: Deterministic Training

Shared code for data generation, evaluation, and reporting.
"""

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np


@dataclass
class TrainingResult:
    """Results from a training run."""

    weights: Dict[str, float]
    train_accuracy: float
    test_accuracy: float
    iterations: int
    training_time_ms: float
    baseline_accuracy: float  # Accuracy with uniform weights


def generate_ticket_data(
    n_samples: int = 1000,
    noise_rate: float = 0.0,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[str], List[str]]:
    """
    Generate synthetic support ticket data for classification.

    Args:
        n_samples: Number of tickets to generate
        noise_rate: Fraction of labels to mislabel (0.0-1.0)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (tickets, noisy_labels, true_labels)
        - noisy_labels: may contain mislabeled examples
        - true_labels: ground truth labels
    """
    random.seed(seed)
    np.random.seed(seed)

    # Team definitions with their characteristic patterns
    teams = {
        "billing": {
            "types": ["refund", "invoice", "payment", "subscription", "pricing"],
            "keywords": ["charge", "bill", "money", "credit", "card", "refund", "price"],
            "priorities": ["low", "medium", "high"],
            "priority_weights": [0.3, 0.5, 0.2],
        },
        "technical": {
            "types": ["bug", "error", "crash", "performance", "integration"],
            "keywords": ["login", "crash", "slow", "error", "broken", "api", "timeout"],
            "priorities": ["low", "medium", "high", "critical"],
            "priority_weights": [0.1, 0.3, 0.4, 0.2],
        },
        "shipping": {
            "types": ["delivery", "tracking", "return", "damage", "lost"],
            "keywords": ["package", "track", "deliver", "ship", "lost", "damage", "return"],
            "priorities": ["low", "medium", "high"],
            "priority_weights": [0.4, 0.4, 0.2],
        },
        "account": {
            "types": ["password", "profile", "settings", "access", "security"],
            "keywords": ["password", "login", "account", "access", "security", "profile"],
            "priorities": ["low", "medium", "high", "critical"],
            "priority_weights": [0.2, 0.4, 0.3, 0.1],
        },
    }

    tickets = []
    noisy_labels = []
    true_labels = []
    team_names = list(teams.keys())

    for i in range(n_samples):
        # Select team (uniform distribution)
        team = random.choice(team_names)
        team_def = teams[team]

        # Generate ticket
        ticket = {
            "type": random.choice(team_def["types"]),
            "keywords": random.sample(team_def["keywords"], k=random.randint(1, 3)),
            "priority": random.choices(
                team_def["priorities"],
                weights=team_def["priority_weights"]
            )[0],
            "customer_tier": random.choice(["free", "pro", "enterprise"]),
            "channel": random.choice(["email", "chat", "phone", "web"]),
            "ticket_id": f"TKT-{i:05d}",
        }

        # Add noise fields with DIFFERENT seed to ensure no correlation with labels
        # Use a completely independent random state
        noise_rng = random.Random(seed + i + 10000)
        ticket["noise_field_1"] = noise_rng.randint(0, 1000)
        ticket["noise_field_2"] = "".join(noise_rng.choices("abcdefgh", k=5))

        tickets.append(ticket)
        true_labels.append(team)

        # Apply label noise if specified
        if noise_rate > 0 and random.random() < noise_rate:
            # Mislabel to a random other team
            wrong_teams = [t for t in team_names if t != team]
            noisy_labels.append(random.choice(wrong_teams))
        else:
            noisy_labels.append(team)

    return tickets, noisy_labels, true_labels


def split_data(
    X: List[Any],
    y: List[str],
    train_ratio: float = 0.8,
    seed: int = 42,
) -> Tuple[List[Any], List[str], List[Any], List[str]]:
    """
    Split data into train and test sets.

    Returns:
        (X_train, y_train, X_test, y_test)
    """
    random.seed(seed)

    indices = list(range(len(X)))
    random.shuffle(indices)

    split_idx = int(len(indices) * train_ratio)
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]

    X_train = [X[i] for i in train_indices]
    y_train = [y[i] for i in train_indices]
    X_test = [X[i] for i in test_indices]
    y_test = [y[i] for i in test_indices]

    return X_train, y_train, X_test, y_test


def compute_accuracy(y_true: List[str], y_pred: List[str]) -> float:
    """Compute classification accuracy."""
    if len(y_true) == 0:
        return 0.0
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return correct / len(y_true)


def print_confusion_matrix(y_true: List[str], y_pred: List[str]) -> None:
    """Print a simple confusion matrix."""
    labels = sorted(set(y_true) | set(y_pred))

    # Count predictions
    matrix = {l1: {l2: 0 for l2 in labels} for l1 in labels}
    for t, p in zip(y_true, y_pred):
        matrix[t][p] += 1

    # Print header
    print("\nConfusion Matrix (rows=true, cols=predicted):")
    header = "          " + " ".join(f"{l:>10}" for l in labels)
    print(header)
    print("-" * len(header))

    # Print rows
    for true_label in labels:
        row = f"{true_label:>10}"
        for pred_label in labels:
            count = matrix[true_label][pred_label]
            row += f" {count:>10}"
        print(row)


def print_training_result(result: TrainingResult) -> None:
    """Pretty-print training results."""
    print("\n" + "=" * 60)
    print("TRAINING RESULTS")
    print("=" * 60)

    print(f"\nBaseline (uniform weights): {result.baseline_accuracy:.1%}")
    print(f"Trained:                    {result.train_accuracy:.1%}")
    print(f"Test accuracy:              {result.test_accuracy:.1%}")
    print(f"Improvement:                {(result.test_accuracy - result.baseline_accuracy):.1%}")

    print(f"\nIterations: {result.iterations}")
    print(f"Training time: {result.training_time_ms:.1f}ms")

    print("\nLearned weights:")
    sorted_weights = sorted(result.weights.items(), key=lambda x: -x[1])
    for field, weight in sorted_weights:
        bar = "█" * int(weight * 10)
        print(f"  {field:20} {weight:5.2f} {bar}")
