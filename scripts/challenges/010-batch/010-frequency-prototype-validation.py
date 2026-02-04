#!/usr/bin/env python3
"""
Challenge 010-010: Frequency Prototype Validation

Before extending Holon, prove the frequency_prototype concept works.

This demo:
1. Implements frequency_prototype as a standalone function
2. Tests on realistic traffic with contamination
3. Compares against baselines (uniform prototype, pattern membership)
4. Validates at different scales and contamination levels
5. Measures actual detection performance

Only if this works well should we consider adding to Holon.
"""

import sys
import time
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(__file__).rsplit("/", 4)[0])

from deterministic_codebook import DeterministicVectorManager


# ============================================================================
# THE PROPOSED PRIMITIVE (standalone implementation for testing)
# ============================================================================

def frequency_prototype(
    vm: DeterministicVectorManager,
    pattern_counts: Dict[str, int],
    min_count: int = 5,
    decay: str = "log",
) -> np.ndarray:
    """
    Build a frequency-weighted prototype from pattern counts.

    This is the proposed primitive we're validating.

    Args:
        vm: Vector manager for getting pattern vectors
        pattern_counts: Dict mapping pattern strings to occurrence counts
        min_count: Minimum count to include pattern (filters rare/suspicious)
        decay: Weight function - "log", "sqrt", "linear", or "threshold"

    Returns:
        Bipolar prototype vector where high-frequency patterns dominate
    """
    vectors = []
    weights = []

    for pattern, count in pattern_counts.items():
        if count >= min_count:
            vec = vm.get_vector(pattern)
            vectors.append(vec)

            if decay == "log":
                weight = np.log(count + 1)
            elif decay == "sqrt":
                weight = np.sqrt(count)
            elif decay == "linear":
                weight = float(count)
            elif decay == "threshold":
                weight = 1.0  # Equal weight for all patterns above min_count
            else:
                raise ValueError(f"Unknown decay function: {decay}")

            weights.append(weight)

    if not vectors:
        return np.zeros(vm.dimensions, dtype=np.int8)

    # Weighted sum
    weighted_sum = np.zeros(vm.dimensions, dtype=np.float32)
    for vec, weight in zip(vectors, weights):
        weighted_sum += weight * vec.astype(np.float32)

    # Threshold to bipolar
    return np.where(
        weighted_sum > 0, 1,
        np.where(weighted_sum < 0, -1, 0)
    ).astype(np.int8)


def uniform_prototype(
    vm: DeterministicVectorManager,
    pattern_counts: Dict[str, int],
    min_count: int = 5,
) -> np.ndarray:
    """Baseline: equal weight for all patterns (no frequency weighting)."""
    return frequency_prototype(vm, pattern_counts, min_count, decay="threshold")


# ============================================================================
# DETECTION METHODS
# ============================================================================

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


@dataclass
class DetectionResult:
    """Results from a detection method."""
    method: str
    precision: float
    recall: float
    f1: float
    threshold: float
    benign_mean: float
    malicious_mean: float
    separation: float


def evaluate_detector(
    vm: DeterministicVectorManager,
    prototype: np.ndarray,
    test_patterns: List[str],
    test_labels: List[bool],  # True = malicious
    method_name: str,
) -> DetectionResult:
    """Evaluate a prototype-based detector."""

    benign_sims = []
    malicious_sims = []

    for pattern, is_malicious in zip(test_patterns, test_labels):
        vec = vm.get_vector(pattern)
        sim = cosine(vec, prototype)

        if is_malicious:
            malicious_sims.append(sim)
        else:
            benign_sims.append(sim)

    # Find best threshold
    best_f1 = 0
    best_threshold = 0
    best_precision = 0
    best_recall = 0

    for threshold in np.linspace(-0.1, 0.3, 81):
        tp = sum(1 for s in malicious_sims if s < threshold)
        fp = sum(1 for s in benign_sims if s < threshold)
        fn = sum(1 for s in malicious_sims if s >= threshold)

        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(0.001, precision + recall)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_precision = precision
            best_recall = recall

    return DetectionResult(
        method=method_name,
        precision=best_precision,
        recall=best_recall,
        f1=best_f1,
        threshold=best_threshold,
        benign_mean=np.mean(benign_sims),
        malicious_mean=np.mean(malicious_sims),
        separation=np.mean(benign_sims) - np.mean(malicious_sims),
    )


def evaluate_membership(
    known_patterns: set,
    test_patterns: List[str],
    test_labels: List[bool],
) -> DetectionResult:
    """Evaluate pattern membership detection."""

    tp = sum(1 for p, l in zip(test_patterns, test_labels) if l and p not in known_patterns)
    fp = sum(1 for p, l in zip(test_patterns, test_labels) if not l and p not in known_patterns)
    fn = sum(1 for p, l in zip(test_patterns, test_labels) if l and p in known_patterns)
    tn = sum(1 for p, l in zip(test_patterns, test_labels) if not l and p in known_patterns)

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(0.001, precision + recall)

    return DetectionResult(
        method="pattern_membership",
        precision=precision,
        recall=recall,
        f1=f1,
        threshold=0,  # N/A
        benign_mean=0,
        malicious_mean=0,
        separation=0,
    )


# ============================================================================
# DATA GENERATION
# ============================================================================

def generate_realistic_traffic(
    n_benign: int,
    n_malicious: int,
    seed: int = 42,
) -> Tuple[List[str], List[bool]]:
    """
    Generate realistic API traffic patterns.

    Returns: (patterns, is_malicious_labels)
    """
    import random
    random.seed(seed)

    # Benign patterns with realistic frequency distribution
    benign_templates = [
        ("GET|/api/users", 100),
        ("GET|/api/users/{id}", 80),
        ("POST|/api/users", 30),
        ("PUT|/api/users/{id}", 20),
        ("DELETE|/api/users/{id}", 10),
        ("GET|/api/orders", 90),
        ("GET|/api/orders/{id}", 70),
        ("POST|/api/orders", 40),
        ("PUT|/api/orders/{id}", 15),
        ("GET|/api/products", 85),
        ("GET|/api/products/{id}", 60),
        ("GET|/api/search", 50),
        ("POST|/api/auth/login", 100),
        ("POST|/api/auth/logout", 40),
        ("GET|/api/profile", 45),
        ("PUT|/api/profile", 20),
        ("GET|/api/notifications", 55),
        ("POST|/api/payments", 25),
    ]

    # Malicious patterns
    malicious_templates = [
        "GET|/api/../../../etc/passwd",
        "GET|/api/users/../../admin/config",
        "GET|/api/users/' OR 1=1--",
        "POST|/api/users/1; DROP TABLE users;--",
        "GET|/api/exec?cmd=ls%20-la",
        "TRACE|/api/users",
        "CONNECT|/api/proxy",
        "GET|/admin/config",
        "GET|/.git/config",
        "GET|/.env",
        "GET|/api/internal/debug",
        "GET|/api/%2e%2e/etc/passwd",
        "POST|/api/upload?file=../../../root/.ssh/authorized_keys",
        "GET|/api/graphql?query={__schema{types{name}}}",
        "OPTIONS|/api/users",  # Unusual but not always malicious
    ]

    patterns = []
    labels = []

    # Generate benign traffic with weighted distribution
    total_weight = sum(w for _, w in benign_templates)
    for _ in range(n_benign):
        r = random.random() * total_weight
        cumulative = 0
        for pattern, weight in benign_templates:
            cumulative += weight
            if r <= cumulative:
                patterns.append(pattern)
                labels.append(False)
                break

    # Generate malicious traffic (uniform distribution)
    for _ in range(n_malicious):
        pattern = random.choice(malicious_templates)
        patterns.append(pattern)
        labels.append(True)

    # Shuffle
    combined = list(zip(patterns, labels))
    random.shuffle(combined)
    patterns, labels = zip(*combined)

    return list(patterns), list(labels)


# ============================================================================
# MAIN VALIDATION
# ============================================================================

def run_validation(
    n_train_benign: int,
    n_train_malicious: int,
    n_test_benign: int,
    n_test_malicious: int,
    train_seed: int = 42,
    test_seed: int = 999,
) -> Dict[str, DetectionResult]:
    """Run validation with specified parameters."""

    vm = DeterministicVectorManager(dimensions=4096, global_seed=42)

    # Generate data
    train_patterns, train_labels = generate_realistic_traffic(
        n_train_benign, n_train_malicious, train_seed
    )
    test_patterns, test_labels = generate_realistic_traffic(
        n_test_benign, n_test_malicious, test_seed
    )

    # Count patterns in training data
    pattern_counts = Counter(train_patterns)

    # Known patterns (for membership test)
    min_count = 5
    known_patterns = {p for p, c in pattern_counts.items() if c >= min_count}

    results = {}

    # Method 1: Frequency-weighted prototype (log decay)
    proto_freq = frequency_prototype(vm, pattern_counts, min_count=min_count, decay="log")
    results["freq_log"] = evaluate_detector(
        vm, proto_freq, test_patterns, test_labels, "frequency_prototype(log)"
    )

    # Method 2: Frequency-weighted prototype (sqrt decay)
    proto_sqrt = frequency_prototype(vm, pattern_counts, min_count=min_count, decay="sqrt")
    results["freq_sqrt"] = evaluate_detector(
        vm, proto_sqrt, test_patterns, test_labels, "frequency_prototype(sqrt)"
    )

    # Method 3: Uniform prototype (no frequency weighting)
    proto_uniform = uniform_prototype(vm, pattern_counts, min_count=min_count)
    results["uniform"] = evaluate_detector(
        vm, proto_uniform, test_patterns, test_labels, "uniform_prototype"
    )

    # Method 4: Pattern membership (baseline)
    results["membership"] = evaluate_membership(
        known_patterns, test_patterns, test_labels
    )

    return results, pattern_counts, known_patterns


def main():
    print("=" * 80)
    print("Challenge 010-010: Frequency Prototype Validation")
    print("=" * 80)
    print("""
Goal: Prove that frequency_prototype() adds value before extending Holon.

We'll test:
1. Does frequency weighting improve detection over uniform weighting?
2. How does it compare to simple pattern membership?
3. Does it work at different scales and contamination levels?
""")

    # Test scenarios
    scenarios = [
        # (train_benign, train_malicious, test_benign, test_malicious, description)
        (1000, 10, 200, 20, "Small scale, 1% contamination"),
        (1000, 50, 200, 20, "Small scale, 5% contamination"),
        (5000, 50, 500, 50, "Medium scale, 1% contamination"),
        (5000, 250, 500, 50, "Medium scale, 5% contamination"),
        (10000, 100, 1000, 100, "Large scale, 1% contamination"),
    ]

    all_results = []

    for train_b, train_m, test_b, test_m, desc in scenarios:
        print(f"\n{'='*80}")
        print(f"Scenario: {desc}")
        print(f"Training: {train_b} benign + {train_m} malicious ({100*train_m/(train_b+train_m):.1f}% contamination)")
        print(f"Testing:  {test_b} benign + {test_m} malicious")
        print("=" * 80)

        start = time.time()
        results, pattern_counts, known_patterns = run_validation(
            train_b, train_m, test_b, test_m
        )
        elapsed = time.time() - start

        print(f"\nUnique patterns in training: {len(pattern_counts)}")
        print(f"Known patterns (count >= 5): {len(known_patterns)}")
        print(f"Evaluation time: {elapsed:.2f}s")

        print(f"\n{'Method':<25} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Separation':>12}")
        print("-" * 70)

        for key in ["freq_log", "freq_sqrt", "uniform", "membership"]:
            r = results[key]
            sep = f"{r.separation:+.4f}" if r.separation != 0 else "N/A"
            print(f"{r.method:<25} {r.precision:>10.1%} {r.recall:>10.1%} {r.f1:>10.3f} {sep:>12}")

        all_results.append((desc, results))

    # Summary comparison
    print("\n" + "=" * 80)
    print("SUMMARY: Does frequency_prototype add value?")
    print("=" * 80)

    freq_log_wins = 0
    freq_sqrt_wins = 0
    uniform_wins = 0
    membership_wins = 0

    print(f"\n{'Scenario':<40} {'Best Method':<25} {'F1':>10}")
    print("-" * 80)

    for desc, results in all_results:
        best = max(results.items(), key=lambda x: x[1].f1)
        print(f"{desc:<40} {best[1].method:<25} {best[1].f1:>10.3f}")

        if best[0] == "freq_log":
            freq_log_wins += 1
        elif best[0] == "freq_sqrt":
            freq_sqrt_wins += 1
        elif best[0] == "uniform":
            uniform_wins += 1
        else:
            membership_wins += 1

    print("\n" + "-" * 80)
    print(f"frequency_prototype(log) wins:  {freq_log_wins}/{len(scenarios)}")
    print(f"frequency_prototype(sqrt) wins: {freq_sqrt_wins}/{len(scenarios)}")
    print(f"uniform_prototype wins:         {uniform_wins}/{len(scenarios)}")
    print(f"pattern_membership wins:        {membership_wins}/{len(scenarios)}")

    # Final verdict
    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)

    freq_wins = freq_log_wins + freq_sqrt_wins

    if freq_wins > len(scenarios) / 2:
        print("""
✓ VALIDATED: frequency_prototype() adds value over uniform weighting.

Recommendation: Add to Holon as a primitive.

The frequency weighting helps because:
- High-frequency patterns contribute more to the prototype
- Rare patterns (potential anomalies) have minimal influence
- This creates better separation between known and unknown patterns
""")
    elif membership_wins >= len(scenarios) / 2:
        print("""
✗ NOT VALIDATED: Pattern membership outperforms frequency_prototype().

Recommendation: Keep pattern membership as primary detection method.
VSA provides distributed consensus, but similarity-based detection
doesn't add value over simple hash lookup.

frequency_prototype() may still be useful for:
- Fuzzy matching when exact pattern not found
- Composing patterns for streaming updates
""")
    else:
        print("""
? INCONCLUSIVE: No clear winner across scenarios.

Recommendation: More investigation needed.
Consider:
- Different decay functions
- Different threshold strategies
- Hybrid approaches (membership + similarity)
""")


if __name__ == "__main__":
    main()
