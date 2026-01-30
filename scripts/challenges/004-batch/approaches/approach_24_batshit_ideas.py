#!/usr/bin/env python3
"""
Approach 24: Batshit Ideas

Testing remaining unexplored angles:
1. Negative space encoding
2. Multi-scale encoding
3. Symmetry-aware encoding
4. Conflict learning
"""

import sys
sys.path.insert(0, "/home/watmin/work/holon")

from typing import List, Optional, Set, Tuple, Dict
import numpy as np
import time

from holon import CPUStore

from common import (
    similarity,
    count_empty,
    get_available_digits_9x9,
    validate_9x9,
    PUZZLE_9x9_HARD,
)


def create_store(dimensions: int = 16384):
    return CPUStore(dimensions=dimensions)


# =============================================================================
# IDEA 1: NEGATIVE SPACE ENCODING
# =============================================================================

def test_negative_space():
    """
    Encode what ISN'T rather than what IS.

    For each cell, encode which digits are ELIMINATED.
    """
    print("=" * 70)
    print("IDEA 1: NEGATIVE SPACE ENCODING")
    print("=" * 70)

    store = create_store()
    grid = [[cell for cell in row] for row in PUZZLE_9x9_HARD]

    # For each cell, get eliminated digits
    def get_eliminated(g, r, c):
        if g[r][c] is not None:
            return set(range(1, 10)) - {g[r][c]}  # All except placed
        else:
            available = get_available_digits_9x9(g, r, c)
            return set(range(1, 10)) - available

    # Encode negative space
    digit_vecs = {d: store.vector_manager.get_vector(f"NOT_{d}") for d in range(1, 10)}

    def encode_negative_space(g):
        """Encode what's eliminated from each cell."""
        vectors = []
        for r in range(9):
            for c in range(9):
                eliminated = get_eliminated(g, r, c)
                if eliminated:
                    pos_vec = store.vector_manager.get_vector(f"neg_pos_{r}_{c}")
                    elim_bundle = np.sum([digit_vecs[d] for d in eliminated], axis=0)
                    vectors.append(pos_vec * elim_bundle)

        if vectors:
            bundled = np.sum(vectors, axis=0)
            return np.where(bundled > 0, 1, np.where(bundled < 0, -1, 0)).astype(np.int8)
        return np.zeros(store.dimensions, dtype=np.int8)

    # Test: does negative space distinguish choices?
    neg_vec = encode_negative_space(grid)
    print(f"\nNegative space encoding norm: {np.linalg.norm(neg_vec):.1f}")

    # Find decision point
    for r in range(9):
        for c in range(9):
            if grid[r][c] is None:
                options = list(get_available_digits_9x9(grid, r, c))
                if len(options) > 1:
                    print(f"\nDecision point: ({r},{c}) options={options}")

                    # Score by negative space change
                    for digit in options:
                        test_grid = [[cell for cell in row] for row in grid]
                        test_grid[r][c] = digit

                        new_neg = encode_negative_space(test_grid)
                        delta = np.linalg.norm(new_neg - neg_vec)
                        sim = similarity(neg_vec, new_neg)

                        print(f"  Digit {digit}: delta={delta:.1f}, sim={sim:.4f}")

                    return


# =============================================================================
# IDEA 2: MULTI-SCALE ENCODING
# =============================================================================

def test_multi_scale():
    """
    Encode at multiple scales: cell, block, row, grid.
    """
    print("\n" + "=" * 70)
    print("IDEA 2: MULTI-SCALE ENCODING")
    print("=" * 70)

    store = create_store()
    grid = [[cell for cell in row] for row in PUZZLE_9x9_HARD]

    digit_vecs = {d: store.vector_manager.get_vector(str(d)) for d in range(1, 10)}
    complete = np.sum([digit_vecs[d] for d in range(1, 10)], axis=0)
    complete = np.where(complete > 0, 1, np.where(complete < 0, -1, 0)).astype(np.int8)

    def encode_set(digits):
        if not digits:
            return np.zeros(store.dimensions, dtype=np.int8)
        vecs = [digit_vecs[d] for d in digits]
        bundled = np.sum(vecs, axis=0)
        return np.where(bundled > 0, 1, np.where(bundled < 0, -1, 0)).astype(np.int8)

    def multi_scale_score(g, r, c, digit):
        """Score at multiple scales."""
        # Cell scale: just the digit
        cell_vec = digit_vecs[digit]

        # Row scale
        row_digits = {g[r][cc] for cc in range(9) if g[r][cc] is not None} | {digit}
        row_vec = encode_set(row_digits)
        row_score = similarity(row_vec, complete)

        # Column scale
        col_digits = {g[rr][c] for rr in range(9) if g[rr][c] is not None} | {digit}
        col_vec = encode_set(col_digits)
        col_score = similarity(col_vec, complete)

        # Block scale
        br, bc = (r // 3) * 3, (c // 3) * 3
        block_digits = {digit}
        for rr in range(br, br + 3):
            for cc in range(bc, bc + 3):
                if g[rr][cc] is not None:
                    block_digits.add(g[rr][cc])
        block_vec = encode_set(block_digits)
        block_score = similarity(block_vec, complete)

        # Grid scale: how complete is the whole grid?
        all_filled = sum(1 for rr in range(9) for cc in range(9) if g[rr][cc] is not None)
        grid_progress = (all_filled + 1) / 81.0

        return {
            'row': row_score,
            'col': col_score,
            'block': block_score,
            'grid': grid_progress,
            'total': row_score + col_score + block_score + grid_progress
        }

    # Test
    for r in range(9):
        for c in range(9):
            if grid[r][c] is None:
                options = list(get_available_digits_9x9(grid, r, c))
                if len(options) > 1:
                    print(f"\nDecision point: ({r},{c}) options={options}")

                    for digit in options:
                        scores = multi_scale_score(grid, r, c, digit)
                        print(f"\n  Digit {digit}:")
                        print(f"    Row:   {scores['row']:.4f}")
                        print(f"    Col:   {scores['col']:.4f}")
                        print(f"    Block: {scores['block']:.4f}")
                        print(f"    Total: {scores['total']:.4f}")

                    return


# =============================================================================
# IDEA 3: SYMMETRY-AWARE ENCODING
# =============================================================================

def test_symmetry():
    """
    Sudoku has symmetries. Can we exploit them?

    Symmetries:
    - 9! digit relabeling
    - 6 row band permutations
    - 6 column stack permutations
    - 6 row permutations within bands
    - 6 column permutations within stacks
    - 2 transpose
    - 4 rotations

    Total: massive symmetry group
    """
    print("\n" + "=" * 70)
    print("IDEA 3: SYMMETRY-AWARE ENCODING")
    print("=" * 70)

    store = create_store()
    grid = [[cell for cell in row] for row in PUZZLE_9x9_HARD]

    # Compute symmetry-invariant features
    def get_invariant_features(g):
        """Features that are invariant under symmetry."""
        # Count of each digit
        digit_counts = {d: 0 for d in range(1, 10)}
        for r in range(9):
            for c in range(9):
                if g[r][c] is not None:
                    digit_counts[g[r][c]] += 1

        # Sorted counts (invariant to digit relabeling)
        sorted_counts = tuple(sorted(digit_counts.values(), reverse=True))

        # Constraint saturation: how full are rows/cols/blocks
        row_counts = [sum(1 for c in range(9) if g[r][c] is not None) for r in range(9)]
        col_counts = [sum(1 for r in range(9) if g[r][c] is not None) for c in range(9)]
        block_counts = []
        for b in range(9):
            br, bc = (b // 3) * 3, (b % 3) * 3
            count = sum(1 for r in range(br, br+3) for c in range(bc, bc+3) if g[r][c] is not None)
            block_counts.append(count)

        # Sorted (invariant to permutation)
        sorted_rows = tuple(sorted(row_counts, reverse=True))
        sorted_cols = tuple(sorted(col_counts, reverse=True))
        sorted_blocks = tuple(sorted(block_counts, reverse=True))

        return {
            'digit_distribution': sorted_counts,
            'row_saturation': sorted_rows,
            'col_saturation': sorted_cols,
            'block_saturation': sorted_blocks,
        }

    features = get_invariant_features(grid)
    print(f"\nSymmetry-invariant features:")
    print(f"  Digit distribution: {features['digit_distribution']}")
    print(f"  Row saturation: {features['row_saturation']}")
    print(f"  Col saturation: {features['col_saturation']}")
    print(f"  Block saturation: {features['block_saturation']}")

    # Encode invariant features
    def encode_invariant(g):
        features = get_invariant_features(g)

        # Encode each distribution
        vecs = []
        for i, count in enumerate(features['digit_distribution']):
            vecs.append(store.vector_manager.get_vector(f"dig_rank_{i}_{count}"))
        for i, count in enumerate(features['row_saturation']):
            vecs.append(store.vector_manager.get_vector(f"row_rank_{i}_{count}"))

        if vecs:
            bundled = np.sum(vecs, axis=0)
            return np.where(bundled > 0, 1, np.where(bundled < 0, -1, 0)).astype(np.int8)
        return np.zeros(store.dimensions, dtype=np.int8)

    inv_vec = encode_invariant(grid)
    print(f"\nInvariant encoding norm: {np.linalg.norm(inv_vec):.1f}")


# =============================================================================
# IDEA 4: CONFLICT LEARNING (CDCL-inspired)
# =============================================================================

def test_conflict_learning():
    """
    When we backtrack, encode WHY we failed.
    Avoid repeating the same mistake.
    """
    print("\n" + "=" * 70)
    print("IDEA 4: CONFLICT LEARNING")
    print("=" * 70)

    store = create_store()

    # Conflict database
    conflicts = []

    def encode_partial_assignment(g):
        """Encode the current partial assignment."""
        digit_vecs = {d: store.vector_manager.get_vector(str(d)) for d in range(1, 10)}
        vectors = []
        for r in range(9):
            for c in range(9):
                if g[r][c] is not None:
                    pos_vec = store.vector_manager.get_vector(f"pos_{r}_{c}")
                    vectors.append(pos_vec * digit_vecs[g[r][c]])

        if vectors:
            bundled = np.sum(vectors, axis=0)
            return np.where(bundled > 0, 1, np.where(bundled < 0, -1, 0)).astype(np.int8)
        return np.zeros(store.dimensions, dtype=np.int8)

    def propagate(g):
        changed = True
        while changed:
            changed = False
            for r in range(9):
                for c in range(9):
                    if g[r][c] is None:
                        opts = list(get_available_digits_9x9(g, r, c))
                        if not opts:
                            return False
                        if len(opts) == 1:
                            g[r][c] = opts[0]
                            changed = True
        return True

    def solve_with_conflict_learning(puzzle):
        """Solve and collect conflict patterns."""
        grid = [[cell for cell in row] for row in puzzle]
        propagate(grid)

        backtracks = [0]

        def solve_rec(g, depth):
            if count_empty(g) == 0:
                return g

            best = None
            best_count = 10
            for r in range(9):
                for c in range(9):
                    if g[r][c] is None:
                        opts = list(get_available_digits_9x9(g, r, c))
                        if not opts:
                            return None
                        if len(opts) < best_count:
                            best_count = len(opts)
                            best = (r, c, opts)

            if best is None:
                return g

            r, c, options = best

            for digit in options:
                test_g = [[cell for cell in row] for row in g]
                test_g[r][c] = digit

                if not propagate(test_g):
                    # CONFLICT! Encode this state
                    conflict_vec = encode_partial_assignment(test_g)
                    conflicts.append(conflict_vec)
                    continue

                result = solve_rec(test_g, depth + 1)
                if result:
                    return result

                # Backtrack - also a conflict
                backtracks[0] += 1
                conflict_vec = encode_partial_assignment(test_g)
                conflicts.append(conflict_vec)

            return None

        result = solve_rec(grid, 0)
        return result, backtracks[0]

    # Solve
    result, bt = solve_with_conflict_learning(PUZZLE_9x9_HARD)
    print(f"\nSolved: {result is not None}")
    print(f"Backtracks: {bt}")
    print(f"Conflicts collected: {len(conflicts)}")

    if conflicts:
        # Build conflict prototype
        conflict_bundle = np.sum(conflicts, axis=0)
        conflict_proto = np.where(conflict_bundle > 0, 1, np.where(conflict_bundle < 0, -1, 0)).astype(np.int8)
        print(f"Conflict prototype norm: {np.linalg.norm(conflict_proto):.1f}")

        # Could use this to AVOID similar states in future solving


# =============================================================================
# MAIN
# =============================================================================

def main():
    test_negative_space()
    test_multi_scale()
    test_symmetry()
    test_conflict_learning()


if __name__ == "__main__":
    main()
