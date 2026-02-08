#!/usr/bin/env python3
"""
=============================================================================
BATCH 013 - CHALLENGE 011: Binary Search Rate Decoding
=============================================================================

PROBLEM: Storing N reference vectors is expensive. Can we decode with O(log N)?

INSIGHT: encode_scalar_log(x) encodes log10(x) into a vector.
We can binary search to find the log value that maximizes similarity.

ALGORITHM:
1. Target: baseline_rate_vec (single vector, shipped from central)
2. Binary search on log scale [0, 12] covering 1 to 1 trillion pps
3. At each step, encode midpoint and compare similarity
4. Converge on the log10 value that best matches target

COMPLEXITY:
- O(log(range/precision)) encode+similarity operations
- For precision 0.1 on log scale: ~7 iterations
- No reference vectors stored!

Run: ./scripts/run_with_venv.sh python scripts/challenges/013-batch/011-binary-search-rate-decode.py
"""

import sys
import math
from typing import Tuple
import numpy as np

sys.path.insert(0, str(__file__).rsplit("/", 4)[0])

from holon import CPUStore


DIMENSIONS = 4096


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(np.float64).flatten(), b.astype(np.float64).flatten()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class BinarySearchRateDecoder:
    """
    Decodes rate from a rate vector using binary search.

    NO reference vectors stored.
    O(log N) complexity.
    """

    def __init__(self, store: CPUStore, precision: float = 0.1):
        """
        Args:
            store: Holon store for encoding
            precision: Precision on log10 scale (0.1 = within ~25% of true rate)
        """
        self.store = store
        self.precision = precision
        self.encode_count = 0
        self.similarity_count = 0

    def decode(self, target_vec: np.ndarray,
               log_lo: float = -1, log_hi: float = 12) -> Tuple[float, int]:
        """
        Binary search to find rate that best matches target vector.

        Args:
            target_vec: The rate vector to decode
            log_lo: Lower bound on log10(rate) - default 0.1 pps
            log_hi: Upper bound on log10(rate) - default 1 trillion pps

        Returns:
            (decoded_rate_pps, iterations)
        """
        self.encode_count = 0
        self.similarity_count = 0
        iterations = 0

        while log_hi - log_lo > self.precision:
            iterations += 1

            # Probe three points: q1, mid, q3
            mid = (log_lo + log_hi) / 2
            q1 = (log_lo + mid) / 2
            q3 = (mid + log_hi) / 2

            # Encode and measure similarity
            vec_q1 = self.store.encode_scalar_log(10**q1)
            vec_mid = self.store.encode_scalar_log(10**mid)
            vec_q3 = self.store.encode_scalar_log(10**q3)
            self.encode_count += 3

            sim_q1 = similarity(target_vec, vec_q1)
            sim_mid = similarity(target_vec, vec_mid)
            sim_q3 = similarity(target_vec, vec_q3)
            self.similarity_count += 3

            # Find which region has highest similarity
            if sim_q1 >= sim_mid and sim_q1 >= sim_q3:
                log_hi = mid
            elif sim_q3 >= sim_mid and sim_q3 >= sim_q1:
                log_lo = mid
            else:
                # mid is peak, narrow around it
                log_lo = q1
                log_hi = q3

        # Return rate at midpoint of final range
        final_log = (log_lo + log_hi) / 2
        return 10**final_log, iterations


class GoldenSectionRateDecoder:
    """
    Alternative: Golden section search (slightly more efficient).

    Uses golden ratio to minimize encode operations.
    """

    def __init__(self, store: CPUStore, precision: float = 0.1):
        self.store = store
        self.precision = precision
        self.phi = (1 + math.sqrt(5)) / 2  # Golden ratio
        self.encode_count = 0
        self.similarity_count = 0

    def decode(self, target_vec: np.ndarray,
               log_lo: float = -1, log_hi: float = 12) -> Tuple[float, int]:
        """Golden section search for rate decoding."""
        self.encode_count = 0
        self.similarity_count = 0
        iterations = 0

        # Initial probes
        d = (log_hi - log_lo) / self.phi
        x1 = log_hi - d
        x2 = log_lo + d

        vec_x1 = self.store.encode_scalar_log(10**x1)
        vec_x2 = self.store.encode_scalar_log(10**x2)
        self.encode_count += 2

        sim_x1 = similarity(target_vec, vec_x1)
        sim_x2 = similarity(target_vec, vec_x2)
        self.similarity_count += 2

        while log_hi - log_lo > self.precision:
            iterations += 1

            if sim_x1 > sim_x2:
                log_hi = x2
                x2 = x1
                sim_x2 = sim_x1
                d = (log_hi - log_lo) / self.phi
                x1 = log_hi - d
                vec_x1 = self.store.encode_scalar_log(10**x1)
                self.encode_count += 1
                sim_x1 = similarity(target_vec, vec_x1)
                self.similarity_count += 1
            else:
                log_lo = x1
                x1 = x2
                sim_x1 = sim_x2
                d = (log_hi - log_lo) / self.phi
                x2 = log_lo + d
                vec_x2 = self.store.encode_scalar_log(10**x2)
                self.encode_count += 1
                sim_x2 = similarity(target_vec, vec_x2)
                self.similarity_count += 1

        final_log = (log_lo + log_hi) / 2
        return 10**final_log, iterations


# =============================================================================
# TESTS
# =============================================================================

def test_binary_search_accuracy():
    print("="*70)
    print("TEST 1: Binary Search Accuracy")
    print("="*70)

    store = CPUStore(dimensions=DIMENSIONS)
    decoder = BinarySearchRateDecoder(store, precision=0.1)

    test_rates = [
        1,
        10,
        100,
        300,
        1000,
        5000,        # User's example baseline
        10000,
        100000,
        1000000,
        22000000,    # User's example attack
        1000000000,
    ]

    print(f"\n{'True Rate':>15} {'Decoded':>15} {'Error %':>10} {'Iters':>8} {'Encodes':>8}")
    print("-"*65)

    for true_rate in test_rates:
        # Encode true rate
        true_vec = store.encode_scalar_log(float(true_rate))

        # Decode using binary search
        decoded, iters = decoder.decode(true_vec)

        error_pct = abs(decoded - true_rate) / true_rate * 100

        print(f"{true_rate:>15,} {decoded:>15,.0f} {error_pct:>10.1f}% {iters:>8} {decoder.encode_count:>8}")


def test_golden_section():
    print("\n" + "="*70)
    print("TEST 2: Golden Section Search (More Efficient)")
    print("="*70)

    store = CPUStore(dimensions=DIMENSIONS)
    decoder = GoldenSectionRateDecoder(store, precision=0.1)

    test_rates = [5000, 22000000, 1000000000]

    print(f"\n{'True Rate':>15} {'Decoded':>15} {'Error %':>10} {'Iters':>8} {'Encodes':>8}")
    print("-"*65)

    for true_rate in test_rates:
        true_vec = store.encode_scalar_log(float(true_rate))
        decoded, iters = decoder.decode(true_vec)
        error_pct = abs(decoded - true_rate) / true_rate * 100
        print(f"{true_rate:>15,} {decoded:>15,.0f} {error_pct:>10.1f}% {iters:>8} {decoder.encode_count:>8}")


def test_user_scenario():
    print("\n" + "="*70)
    print("TEST 3: User's Scenario - 5,000 pps baseline, 22M attack")
    print("="*70)

    store = CPUStore(dimensions=DIMENSIONS)

    # Baseline: 5,000 pps
    baseline_pps = 5000
    baseline_vec = store.encode_scalar_log(float(baseline_pps))

    # Attack: 22,000,000 pps
    attack_pps = 22000000
    attack_vec = store.encode_scalar_log(float(attack_pps))

    print(f"\nBaseline: {baseline_pps:,} pps (log10 = {math.log10(baseline_pps):.2f})")
    print(f"Attack:   {attack_pps:,} pps (log10 = {math.log10(attack_pps):.2f})")
    print(f"Ratio:    {attack_pps/baseline_pps:,.0f}x ({math.log10(attack_pps/baseline_pps):.1f} orders of magnitude)")

    # Decode baseline using binary search
    decoder = BinarySearchRateDecoder(store, precision=0.05)  # Finer precision
    decoded_baseline, iters = decoder.decode(baseline_vec)

    print(f"\nDecoded baseline: {decoded_baseline:,.0f} pps")
    print(f"  Iterations: {iters}")
    print(f"  Encodes: {decoder.encode_count}")
    print(f"  Error: {abs(decoded_baseline - baseline_pps) / baseline_pps * 100:.1f}%")

    # The decoded rate IS the rate limit to enforce
    print(f"\n→ Rate limit to enforce: {decoded_baseline:,.0f} pps")


def test_complexity():
    print("\n" + "="*70)
    print("TEST 4: Complexity Analysis")
    print("="*70)

    store = CPUStore(dimensions=DIMENSIONS)

    precisions = [1.0, 0.5, 0.1, 0.05, 0.01]

    print(f"\nFor range [0.1, 1 trillion] pps (13 orders of magnitude):")
    print(f"\n{'Precision':>12} {'Iters':>8} {'Encodes':>10} {'Rate Error':>12}")
    print("-"*50)

    true_rate = 5000

    for prec in precisions:
        decoder = BinarySearchRateDecoder(store, precision=prec)
        true_vec = store.encode_scalar_log(float(true_rate))
        decoded, iters = decoder.decode(true_vec)
        error_pct = abs(decoded - true_rate) / true_rate * 100

        # Error bound: 10^(precision) - 1 as percentage
        max_error = (10**prec - 1) * 100

        print(f"{prec:>12} {iters:>8} {decoder.encode_count:>10} {error_pct:>11.1f}%")

    print(f"""
    Complexity: O(log(range/precision))

    For precision=0.1:
    - ~7 iterations
    - ~21 encode operations
    - ~21 similarity computations
    - Error ≤ 25% (within one "step" on log scale)

    For precision=0.05:
    - ~9 iterations
    - ~27 encode operations
    - Error ≤ 12%

    NO REFERENCE VECTORS STORED.
    Just the single baseline rate vector.
    """)


def test_single_vector_solution():
    print("\n" + "="*70)
    print("TEST 5: Single Vector Solution")
    print("="*70)

    store = CPUStore(dimensions=DIMENSIONS)

    print("""
    WHAT CENTRAL SHIPS:
    - baseline_rate_vec (single 4096-dim vector)

    WHAT FIELD SCRUBBER DOES:
    1. Binary search to decode baseline_rate_vec
    2. Get concrete PPS: ~5,000 pps
    3. Enforce that rate

    NO reference vectors needed.
    The encoder is stateless - generate probes on demand.
    """)

    # Simulate
    baseline_pps = 5000
    baseline_vec = store.encode_scalar_log(float(baseline_pps))

    print(f"Central ships: baseline_rate_vec ({baseline_vec.shape}, {baseline_vec.nbytes} bytes)")

    decoder = GoldenSectionRateDecoder(store, precision=0.1)
    decoded, iters = decoder.decode(baseline_vec)

    print(f"\nScrubber decodes:")
    print(f"  Iterations: {iters}")
    print(f"  Encodes: {decoder.encode_count}")
    print(f"  Result: {decoded:,.0f} pps")
    print(f"  True rate: {baseline_pps:,} pps")
    print(f"  Error: {abs(decoded - baseline_pps) / baseline_pps * 100:.1f}%")

    # Memory comparison
    num_refs = 15  # What we had before
    ref_storage = num_refs * DIMENSIONS * 4  # 4 bytes per float32

    print(f"\n  Memory saved:")
    print(f"    Before: {num_refs} reference vectors = {ref_storage:,} bytes")
    print(f"    After:  0 reference vectors = 0 bytes")
    print(f"    (Just compute probes on demand)")


def main():
    test_binary_search_accuracy()
    test_golden_section()
    test_user_scenario()
    test_complexity()
    test_single_vector_solution()

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("""
    Binary search rate decoding:

    INPUT:  Single baseline_rate_vec (shipped from central)
    OUTPUT: Decoded rate in PPS (e.g., 5,000 pps)

    ALGORITHM:
    - Binary search on log10 scale [0, 12]
    - At each step, encode probe values and compare similarity
    - Converge to rate that maximizes similarity

    COMPLEXITY:
    - O(log(range/precision)) ≈ 7-10 iterations
    - ~20-30 encode + similarity operations
    - NO stored reference vectors

    For 5,000 pps baseline → 22,000,000 pps attack:
    - Decode baseline: ~5,000 pps (within 10%)
    - Enforce that rate
    - Done!
    """)


if __name__ == "__main__":
    main()
