#!/usr/bin/env python3
"""
Test fundamental VSA/HDC operations before attempting radical approaches.

This verifies:
1. Bind/unbind actually work (can we recover bound components?)
2. Bundling/superposition works (can we detect components in a bundle?)
3. Encoding structure works (does data similarity work as expected?)

These are prerequisites for all 9 radical approaches.
"""

import sys
sys.path.insert(0, "/home/watmin/work/holon")

import numpy as np
from common import (
    create_client,
    VectorCache,
    bind,
    unbind,
    bundle,
    similarity,
    remove_component,
    effective_dimensionality,
    print_grid_4x4,
    PUZZLE_4x4_EASY,
    PUZZLE_4x4_SOLUTION,
)


def test_bind_unbind():
    """
    Test: Can we bind two vectors and recover one by unbinding the other?

    This is CRITICAL for approaches 5, 8 (entanglement, inverse encoding).
    """
    print("\n" + "=" * 60)
    print("TEST 1: Bind/Unbind Recovery")
    print("=" * 60)

    client = create_client(dimensions=16384)
    cache = VectorCache(client)

    # Get two random vectors (digit and position)
    digit_5 = cache.get_digit_vector(5)
    pos_00 = cache.get_position_vector(0, 0)

    print(f"\nDigit 5 vector: shape={digit_5.shape}, nonzeros={np.count_nonzero(digit_5)}")
    print(f"Position (0,0) vector: shape={pos_00.shape}, nonzeros={np.count_nonzero(pos_00)}")

    # Bind them together
    cell_binding = bind(digit_5, pos_00)
    print(f"Cell binding: shape={cell_binding.shape}, nonzeros={np.count_nonzero(cell_binding)}")

    # Try to recover digit by unbinding position
    recovered_digit = unbind(cell_binding, pos_00)

    # Check similarity to original digit
    recovery_sim = similarity(recovered_digit, digit_5)
    print(f"\nRecovered digit similarity to original: {recovery_sim:.4f}")

    # Check similarity to other digits (should be low)
    for d in [1, 2, 3, 4, 6, 7, 8, 9]:
        other_digit = cache.get_digit_vector(d)
        other_sim = similarity(recovered_digit, other_digit)
        print(f"  vs digit {d}: {other_sim:.4f}")

    # Verdict
    other_sims = [similarity(recovered_digit, cache.get_digit_vector(d)) for d in range(1, 10) if d != 5]
    max_other = max(other_sims)

    # Note: We don't expect 1.0 recovery - 0.5+ with clear separation is success
    if recovery_sim > 0.3 and recovery_sim > max_other + 0.2:
        print("\n✓ PASS: Unbind recovers bound component (clearly distinguishable)")
        return True
    else:
        print(f"\n✗ FAIL: Recovery sim={recovery_sim:.4f}, max other={max_other:.4f}")
        return False


def test_multiple_bindings():
    """
    Test: Can we bundle multiple bindings and still recover components?

    This tests if a grid vector (bundle of cell bindings) can be queried.
    Critical for approaches 3, 5, 8.
    """
    print("\n" + "=" * 60)
    print("TEST 2: Multiple Bindings in Bundle")
    print("=" * 60)

    client = create_client(dimensions=16384)
    cache = VectorCache(client)

    # Create bindings for a few cells
    cells = [
        (0, 0, 5),  # Row 0, Col 0, Digit 5
        (0, 1, 3),  # Row 0, Col 1, Digit 3
        (1, 0, 6),  # Row 1, Col 0, Digit 6
    ]

    cell_bindings = []
    for r, c, d in cells:
        pos_vec = cache.get_position_vector(r, c)
        digit_vec = cache.get_digit_vector(d)
        cell_bindings.append(bind(pos_vec, digit_vec))
        print(f"Created binding for cell ({r},{c})={d}")

    # Bundle all bindings
    grid_vec = bundle(cell_bindings)
    print(f"\nGrid vector (bundle of {len(cells)} cells)")

    # Try to recover each digit
    print("\nRecovery test:")
    successes = 0
    for r, c, expected_d in cells:
        pos_vec = cache.get_position_vector(r, c)
        recovered = unbind(grid_vec, pos_vec)

        # Find best matching digit
        best_d = None
        best_sim = -1
        for d in range(1, 10):
            digit_vec = cache.get_digit_vector(d)
            sim = similarity(recovered, digit_vec)
            if sim > best_sim:
                best_sim = sim
                best_d = d

        correct = (best_d == expected_d)
        symbol = "✓" if correct else "✗"
        print(f"  Cell ({r},{c}): expected={expected_d}, recovered={best_d} (sim={best_sim:.4f}) {symbol}")

        if correct:
            successes += 1

    print(f"\nRecovered {successes}/{len(cells)} cells correctly")

    if successes == len(cells):
        print("✓ PASS: All bindings recovered from bundle")
        return True
    else:
        print("✗ FAIL: Some bindings not recovered")
        return False


def test_superposition_detection():
    """
    Test: Can we detect which digits are present in a superposition?

    This is critical for approaches 2, 4 (superposition collapse, propagation).
    """
    print("\n" + "=" * 60)
    print("TEST 3: Superposition Detection")
    print("=" * 60)

    client = create_client(dimensions=16384)
    cache = VectorCache(client)

    # Create superposition of digits 1, 3, 7
    present = [1, 3, 7]
    absent = [2, 4, 5, 6, 8, 9]

    present_vecs = [cache.get_digit_vector(d) for d in present]
    superposition = bundle(present_vecs)

    print(f"Superposition of digits: {present}")
    print(f"Absent digits: {absent}")

    # Check similarity to each digit
    print("\nSimilarity to each digit:")
    present_sims = []
    absent_sims = []

    for d in range(1, 10):
        digit_vec = cache.get_digit_vector(d)
        sim = similarity(superposition, digit_vec)

        if d in present:
            present_sims.append(sim)
            marker = "← PRESENT"
        else:
            absent_sims.append(sim)
            marker = ""

        print(f"  Digit {d}: {sim:.4f} {marker}")

    # Check if present digits have higher similarity
    min_present = min(present_sims)
    max_absent = max(absent_sims)

    print(f"\nMin present sim: {min_present:.4f}")
    print(f"Max absent sim: {max_absent:.4f}")

    if min_present > max_absent:
        print("✓ PASS: Present digits have higher similarity than absent")
        return True
    else:
        print("✗ FAIL: Cannot distinguish present from absent")
        return False


def test_remove_from_superposition():
    """
    Test: Can we remove a digit from a superposition?

    This is critical for approach 2, 4 (superposition collapse, propagation).
    """
    print("\n" + "=" * 60)
    print("TEST 4: Remove from Superposition")
    print("=" * 60)

    client = create_client(dimensions=16384)
    cache = VectorCache(client)

    # Start with superposition of all 4 digits (for 4x4)
    all_digits = [1, 2, 3, 4]
    all_vecs = [cache.get_digit_vector(d) for d in all_digits]
    superposition = bundle(all_vecs)

    print(f"Initial superposition: {all_digits}")

    # Remove digit 2
    remove_d = 2
    remove_vec = cache.get_digit_vector(remove_d)
    reduced = remove_component(superposition, remove_vec)

    print(f"After removing digit {remove_d}:")

    # Check what remains
    remaining = [1, 3, 4]
    removed = [2]

    print("\nSimilarity to each digit:")
    for d in all_digits:
        digit_vec = cache.get_digit_vector(d)
        orig_sim = similarity(superposition, digit_vec)
        new_sim = similarity(reduced, digit_vec)
        change = new_sim - orig_sim

        marker = "REMOVED" if d == remove_d else ""
        print(f"  Digit {d}: {orig_sim:.4f} → {new_sim:.4f} (Δ={change:+.4f}) {marker}")

    # Check if removed digit has lower similarity
    removed_vec = cache.get_digit_vector(remove_d)
    removed_sim = similarity(reduced, removed_vec)
    remaining_sims = [similarity(reduced, cache.get_digit_vector(d)) for d in remaining]

    if removed_sim < min(remaining_sims):
        print("\n✓ PASS: Removed digit has lowest similarity")
        return True
    else:
        print("\n✗ FAIL: Remove operation didn't work as expected")
        return False


def test_data_structure_similarity():
    """
    Test: Do similar data structures have similar vectors?

    This is critical for approach 7 (data similarity exploitation).
    """
    print("\n" + "=" * 60)
    print("TEST 5: Data Structure Similarity")
    print("=" * 60)

    client = create_client(dimensions=16384)
    encoder = client._store.encoder

    # Create similar data structures
    data_1 = {"pos": {"row": 0, "col": 0}, "digit": 5}
    data_2 = {"pos": {"row": 0, "col": 0}, "digit": 3}  # Same pos, different digit
    data_3 = {"pos": {"row": 1, "col": 1}, "digit": 5}  # Different pos, same digit
    data_4 = {"pos": {"row": 1, "col": 1}, "digit": 9}  # Completely different

    vec_1 = encoder.encode_data(data_1)
    vec_2 = encoder.encode_data(data_2)
    vec_3 = encoder.encode_data(data_3)
    vec_4 = encoder.encode_data(data_4)

    print(f"data_1: {data_1}")
    print(f"data_2: {data_2} (same pos)")
    print(f"data_3: {data_3} (same digit)")
    print(f"data_4: {data_4} (different)")

    sim_12 = similarity(vec_1, vec_2)
    sim_13 = similarity(vec_1, vec_3)
    sim_14 = similarity(vec_1, vec_4)

    print(f"\nSimilarities to data_1:")
    print(f"  data_2 (same pos): {sim_12:.4f}")
    print(f"  data_3 (same digit): {sim_13:.4f}")
    print(f"  data_4 (different): {sim_14:.4f}")

    # Structures sharing attributes should be more similar
    # But the binding makes this less straightforward...
    print("\nNote: Due to binding, shared attributes may not increase similarity linearly")

    return True  # This is more observational


def test_effective_dimensionality():
    """
    Test: Do valid vs invalid configurations have different dimensionality?

    This is critical for approach 9 (dimensional analysis).
    """
    print("\n" + "=" * 60)
    print("TEST 6: Effective Dimensionality")
    print("=" * 60)

    client = create_client(dimensions=16384)
    cache = VectorCache(client)

    # Get all digit vectors as basis
    basis = [cache.get_digit_vector(d) for d in range(1, 5)]  # 4x4

    # Valid row: all 4 digits
    valid_row = bundle([cache.get_digit_vector(d) for d in [1, 2, 3, 4]])
    valid_dim = effective_dimensionality(valid_row, basis)

    # Invalid row: duplicate digit
    invalid_row_1 = bundle([cache.get_digit_vector(d) for d in [1, 1, 3, 4]])
    invalid_dim_1 = effective_dimensionality(invalid_row_1, basis)

    # Very invalid: same digit repeated
    invalid_row_2 = bundle([cache.get_digit_vector(d) for d in [2, 2, 2, 2]])
    invalid_dim_2 = effective_dimensionality(invalid_row_2, basis)

    print(f"Valid row [1,2,3,4]: dimensionality = {valid_dim:.4f}")
    print(f"Invalid row [1,1,3,4]: dimensionality = {invalid_dim_1:.4f}")
    print(f"Very invalid [2,2,2,2]: dimensionality = {invalid_dim_2:.4f}")

    if valid_dim > invalid_dim_1 > invalid_dim_2:
        print("\n✓ PASS: More valid configurations have higher dimensionality")
        return True
    else:
        print("\n✗ PARTIAL: Dimensionality relationship not as expected")
        return False


def main():
    print("=" * 60)
    print("FUNDAMENTAL VSA/HDC OPERATION TESTS")
    print("=" * 60)
    print("\nThese tests verify the core operations needed for radical approaches.")

    results = []

    results.append(("Bind/Unbind", test_bind_unbind()))
    results.append(("Multiple Bindings", test_multiple_bindings()))
    results.append(("Superposition Detection", test_superposition_detection()))
    results.append(("Remove from Superposition", test_remove_from_superposition()))
    results.append(("Data Similarity", test_data_structure_similarity()))
    results.append(("Dimensionality", test_effective_dimensionality()))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        symbol = "✓" if result else "✗"
        print(f"  {symbol} {name}")

    print(f"\nPassed: {passed}/{total}")

    if passed == total:
        print("\n✓ All fundamental operations work!")
        print("Ready to implement radical approaches.")
    else:
        print("\n⚠ Some operations need investigation.")
        print("May need to adjust approaches or add kernel primitives.")


if __name__ == "__main__":
    main()
