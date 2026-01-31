#!/usr/bin/env python3
"""
Approach 42: Violation Pre-computation

Instead of learning from good/bad choices,
PRE-COMPUTE violation patterns and use them for fast rejection.

THE IDEA:
1. Encode all possible VIOLATION patterns (duplicates in row/col/block)
2. Before making a choice, check if it would create a pattern SIMILAR to violations
3. Reject choices that score high against violation patterns

This is like a "violation database" for fast lookup.
"""

import sys
sys.path.insert(0, "/home/watmin/work/holon")

from typing import List, Optional, Set, Tuple, Dict
import numpy as np
import time
from itertools import combinations

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
# EXPERIMENT 1: Pre-compute Violation Patterns
# =============================================================================

def test_precompute_violations():
    """
    Pre-compute all violation patterns for fast lookup.
    """
    print("=" * 70)
    print("EXPERIMENT 1: Pre-compute Violation Patterns")
    print("=" * 70)

    store = create_store()

    digit_vecs = {d: store.vector_manager.get_vector(f"d{d}") for d in range(1, 10)}
    pos_vecs = {(r, c): store.vector_manager.get_vector(f"p{r}{c}")
                for r in range(9) for c in range(9)}

    # A violation is: same digit in two cells of the same constraint unit

    def encode_row_violation(r, c1, c2, d):
        """Encode: digit d at both (r,c1) and (r,c2)."""
        p1 = store.bind(pos_vecs[(r, c1)], digit_vecs[d])
        p2 = store.bind(pos_vecs[(r, c2)], digit_vecs[d])
        return store.bundle([p1, p2])

    def encode_col_violation(c, r1, r2, d):
        """Encode: digit d at both (r1,c) and (r2,c)."""
        p1 = store.bind(pos_vecs[(r1, c)], digit_vecs[d])
        p2 = store.bind(pos_vecs[(r2, c)], digit_vecs[d])
        return store.bundle([p1, p2])

    def encode_block_violation(block_idx, pos1, pos2, d):
        """Encode: digit d at two positions in same block."""
        br, bc = (block_idx // 3) * 3, (block_idx % 3) * 3
        r1, c1 = br + pos1 // 3, bc + pos1 % 3
        r2, c2 = br + pos2 // 3, bc + pos2 % 3
        p1 = store.bind(pos_vecs[(r1, c1)], digit_vecs[d])
        p2 = store.bind(pos_vecs[(r2, c2)], digit_vecs[d])
        return store.bundle([p1, p2])

    # Generate all violations
    print("\nGenerating violation patterns...")

    row_violations = []
    for r in range(9):
        for c1, c2 in combinations(range(9), 2):
            for d in range(1, 10):
                row_violations.append(encode_row_violation(r, c1, c2, d))

    print(f"  Row violations: {len(row_violations)}")

    col_violations = []
    for c in range(9):
        for r1, r2 in combinations(range(9), 2):
            for d in range(1, 10):
                col_violations.append(encode_col_violation(c, r1, r2, d))

    print(f"  Column violations: {len(col_violations)}")

    block_violations = []
    for b in range(9):
        for p1, p2 in combinations(range(9), 2):
            for d in range(1, 10):
                block_violations.append(encode_block_violation(b, p1, p2, d))

    print(f"  Block violations: {len(block_violations)}")

    total = len(row_violations) + len(col_violations) + len(block_violations)
    print(f"  Total: {total}")

    # Create bundled violation databases
    row_db = store.bundle(row_violations)
    col_db = store.bundle(col_violations)
    block_db = store.bundle(block_violations)

    # Combined
    all_violations = store.bundle([row_db, col_db, block_db])

    print(f"\nViolation database norms:")
    print(f"  Row: {np.linalg.norm(row_db):.1f}")
    print(f"  Col: {np.linalg.norm(col_db):.1f}")
    print(f"  Block: {np.linalg.norm(block_db):.1f}")
    print(f"  All: {np.linalg.norm(all_violations):.1f}")

    return row_db, col_db, block_db, all_violations


# =============================================================================
# EXPERIMENT 2: Test Violation Detection
# =============================================================================

def test_violation_detection(row_db, col_db, block_db):
    """
    Test if we can detect violations using similarity.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Violation Detection")
    print("=" * 70)

    store = create_store()

    digit_vecs = {d: store.vector_manager.get_vector(f"d{d}") for d in range(1, 10)}
    pos_vecs = {(r, c): store.vector_manager.get_vector(f"p{r}{c}")
                for r in range(9) for c in range(9)}

    def encode_state(placements: List[Tuple[int, int, int]]):
        """Encode a set of placements."""
        if not placements:
            return np.zeros(store.dimensions, dtype=np.int8)
        components = [store.bind(pos_vecs[(r, c)], digit_vecs[d])
                      for r, c, d in placements]
        return store.bundle(components)

    # Valid state (no duplicates)
    valid_state = encode_state([
        (0, 0, 1), (0, 1, 2), (0, 2, 3),  # Row 0: 1, 2, 3
        (1, 0, 4), (1, 1, 5),              # Row 1: 4, 5
    ])

    # Invalid state (duplicate 1 in row 0)
    invalid_state = encode_state([
        (0, 0, 1), (0, 1, 1), (0, 2, 3),  # Row 0: 1, 1, 3 - VIOLATION!
        (1, 0, 4), (1, 1, 5),
    ])

    print(f"\nSimilarity to row violation database:")
    print(f"  Valid state:   {similarity(valid_state, row_db):.4f}")
    print(f"  Invalid state: {similarity(invalid_state, row_db):.4f}")

    # Test specific violation pattern
    specific_violation = encode_state([
        (0, 0, 1), (0, 1, 1),  # Just the violation
    ])

    print(f"\nSpecific violation pattern (0,0)=1, (0,1)=1:")
    print(f"  sim to row_db: {similarity(specific_violation, row_db):.4f}")

    # Valid 2-cell pattern
    valid_2cell = encode_state([
        (0, 0, 1), (0, 1, 2),  # Different digits
    ])
    print(f"  Valid pattern (0,0)=1, (0,1)=2:")
    print(f"  sim to row_db: {similarity(valid_2cell, row_db):.4f}")


# =============================================================================
# EXPERIMENT 3: Violation-Aware Solver
# =============================================================================

def test_violation_aware_solver(row_db, col_db, block_db, all_violations):
    """
    Solver that penalizes choices similar to violation patterns.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Violation-Aware Solver")
    print("=" * 70)

    store = create_store()

    digit_vecs = {d: store.vector_manager.get_vector(f"d{d}") for d in range(1, 10)}
    pos_vecs = {(r, c): store.vector_manager.get_vector(f"p{r}{c}")
                for r in range(9) for c in range(9)}
    complete_template = store.bundle([digit_vecs[d] for d in range(1, 10)])

    class ViolationAwareSolver:
        def __init__(self, violation_db, violation_weight=1.0):
            self.violation_db = violation_db
            self.violation_weight = violation_weight
            self.backtracks = 0

        def score_template(self, grid, r, c, digit):
            """Template matching score (from Approach 22)."""
            total = 0.0

            # Row
            row_digits = {grid[r][cc] for cc in range(9) if grid[r][cc] is not None}
            row_digits.add(digit)
            row_vec = store.bundle([digit_vecs[d] for d in row_digits])
            total += similarity(row_vec, complete_template)

            # Column
            col_digits = {grid[rr][c] for rr in range(9) if grid[rr][c] is not None}
            col_digits.add(digit)
            col_vec = store.bundle([digit_vecs[d] for d in col_digits])
            total += similarity(col_vec, complete_template)

            # Block
            br, bc = (r // 3) * 3, (c // 3) * 3
            block_digits = {grid[rr][cc] for rr in range(br, br+3) for cc in range(bc, bc+3)
                            if grid[rr][cc] is not None}
            block_digits.add(digit)
            block_vec = store.bundle([digit_vecs[d] for d in block_digits])
            total += similarity(block_vec, complete_template)

            return total

        def score_violation(self, grid, r, c, digit):
            """Penalty based on similarity to violation patterns."""
            # Encode the new placement with existing placements
            placements = [(r, c, digit)]

            # Add nearby existing placements that could form violations
            # Row
            for cc in range(9):
                if grid[r][cc] is not None:
                    placements.append((r, cc, grid[r][cc]))

            # Col
            for rr in range(9):
                if grid[rr][c] is not None:
                    placements.append((rr, c, grid[rr][c]))

            # Block
            br, bc = (r // 3) * 3, (c // 3) * 3
            for rr in range(br, br + 3):
                for cc in range(bc, bc + 3):
                    if grid[rr][cc] is not None:
                        placements.append((rr, cc, grid[rr][cc]))

            # Encode state
            state = store.bundle([store.bind(pos_vecs[(pr, pc)], digit_vecs[pd])
                                  for pr, pc, pd in placements])

            return similarity(state, self.violation_db)

        def score_choice(self, grid, r, c, digit):
            template_score = self.score_template(grid, r, c, digit)
            violation_penalty = self.score_violation(grid, r, c, digit)
            return template_score - self.violation_weight * violation_penalty

        def solve(self, puzzle):
            grid = [[cell for cell in row] for row in puzzle]
            return self._solve_rec(grid)

        def _solve_rec(self, grid):
            changed = True
            while changed:
                changed = False
                for r in range(9):
                    for c in range(9):
                        if grid[r][c] is None:
                            opts = list(get_available_digits_9x9(grid, r, c))
                            if not opts:
                                return None
                            if len(opts) == 1:
                                grid[r][c] = opts[0]
                                changed = True

            if count_empty(grid) == 0:
                return grid

            best = None
            best_count = 10
            for r in range(9):
                for c in range(9):
                    if grid[r][c] is None:
                        opts = list(get_available_digits_9x9(grid, r, c))
                        if len(opts) < best_count:
                            best_count = len(opts)
                            best = (r, c, opts)

            if best is None:
                return None

            r, c, options = best

            scores = [(self.score_choice(grid, r, c, d), d) for d in options]
            scores.sort(reverse=True)
            ordered = [d for _, d in scores]

            for digit in ordered:
                test_grid = [[cell for cell in row] for row in grid]
                test_grid[r][c] = digit

                result = self._solve_rec(test_grid)
                if result is not None:
                    return result

                self.backtracks += 1

            return None

    # Test with different violation weights
    print("\nTesting different violation weights:")
    results = []

    for weight in [0.0, 0.1, 0.5, 1.0, 2.0]:
        solver = ViolationAwareSolver(all_violations, violation_weight=weight)
        result = solver.solve(PUZZLE_9x9_HARD)
        valid = result is not None and validate_9x9(result)
        results.append((weight, solver.backtracks, valid))
        print(f"  weight={weight}: {solver.backtracks} backtracks, valid={valid}")

    best = min(results, key=lambda x: x[1])
    print(f"\nBest: weight={best[0]}, backtracks={best[1]}")

    return best[1]


# =============================================================================
# EXPERIMENT 4: Compare Approaches
# =============================================================================

def test_compare_approaches(all_violations):
    """
    Compare violation-aware vs template-only vs baseline.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Approach Comparison")
    print("=" * 70)

    store = create_store()
    digit_vecs = {d: store.vector_manager.get_vector(f"d{d}") for d in range(1, 10)}
    complete_template = store.bundle([digit_vecs[d] for d in range(1, 10)])
    pos_vecs = {(r, c): store.vector_manager.get_vector(f"p{r}{c}")
                for r in range(9) for c in range(9)}

    # Baseline solver (no guidance)
    def solve_baseline(puzzle):
        grid = [[cell for cell in row] for row in puzzle]
        backtracks = [0]

        def rec(g):
            changed = True
            while changed:
                changed = False
                for r in range(9):
                    for c in range(9):
                        if g[r][c] is None:
                            opts = list(get_available_digits_9x9(g, r, c))
                            if not opts:
                                return None
                            if len(opts) == 1:
                                g[r][c] = opts[0]
                                changed = True
            if count_empty(g) == 0:
                return g

            best = None
            best_count = 10
            for r in range(9):
                for c in range(9):
                    if g[r][c] is None:
                        opts = list(get_available_digits_9x9(g, r, c))
                        if len(opts) < best_count:
                            best_count = len(opts)
                            best = (r, c, opts)
            if best is None:
                return None

            r, c, options = best
            for digit in options:
                test = [[cell for cell in row] for row in g]
                test[r][c] = digit
                result = rec(test)
                if result:
                    return result
                backtracks[0] += 1
            return None

        rec(grid)
        return backtracks[0]

    # Template-only solver
    def solve_template(puzzle):
        grid = [[cell for cell in row] for row in puzzle]
        backtracks = [0]

        def score(g, r, c, d):
            total = 0.0
            for get_digits in [
                lambda: {g[r][cc] for cc in range(9) if g[r][cc]},
                lambda: {g[rr][c] for rr in range(9) if g[rr][c]},
                lambda: {g[rr][cc] for rr in range((r//3)*3, (r//3)*3+3)
                                   for cc in range((c//3)*3, (c//3)*3+3) if g[rr][cc]},
            ]:
                digits = get_digits()
                digits.add(d)
                vec = store.bundle([digit_vecs[x] for x in digits])
                total += similarity(vec, complete_template)
            return total

        def rec(g):
            changed = True
            while changed:
                changed = False
                for r in range(9):
                    for c in range(9):
                        if g[r][c] is None:
                            opts = list(get_available_digits_9x9(g, r, c))
                            if not opts:
                                return None
                            if len(opts) == 1:
                                g[r][c] = opts[0]
                                changed = True
            if count_empty(g) == 0:
                return g

            best = None
            best_count = 10
            for r in range(9):
                for c in range(9):
                    if g[r][c] is None:
                        opts = list(get_available_digits_9x9(g, r, c))
                        if len(opts) < best_count:
                            best_count = len(opts)
                            best = (r, c, opts)
            if best is None:
                return None

            r, c, options = best
            scores = [(score(g, r, c, d), d) for d in options]
            scores.sort(reverse=True)
            ordered = [x[1] for x in scores]

            for digit in ordered:
                test = [[cell for cell in row] for row in g]
                test[r][c] = digit
                result = rec(test)
                if result:
                    return result
                backtracks[0] += 1
            return None

        rec(grid)
        return backtracks[0]

    base_bt = solve_baseline(PUZZLE_9x9_HARD)
    template_bt = solve_template(PUZZLE_9x9_HARD)

    print(f"\n| Approach | Backtracks | vs Baseline |")
    print(f"|----------|-----------|-------------|")
    print(f"| Baseline | {base_bt} | - |")
    print(f"| Template | {template_bt} | {(1 - template_bt/max(1,base_bt))*100:+.1f}% |")


# =============================================================================
# MAIN
# =============================================================================

def main():
    row_db, col_db, block_db, all_violations = test_precompute_violations()
    test_violation_detection(row_db, col_db, block_db)
    best_bt = test_violation_aware_solver(row_db, col_db, block_db, all_violations)
    test_compare_approaches(all_violations)

    print("\n" + "=" * 70)
    print("VIOLATION PRE-COMPUTATION SUMMARY")
    print("=" * 70)
    print(f"""
APPROACH:
1. Pre-compute ALL violation patterns (duplicate in row/col/block)
2. Bundle into violation databases
3. Penalize choices that score high against violation patterns

RESULTS:
- Total violation patterns: 9 * C(9,2) * 9 * 3 = 8,748 per type
- Violation detection: see similarity scores above
- Best violation weight: see results above

KEY INSIGHT:
The violation database captures WHAT TO AVOID, but constraints already
prevent violations. The signal from template matching (what to PURSUE)
is more valuable than violation avoidance.

Template matching: 52 backtracks (from Approach 22)
Best violation-aware: {best_bt} backtracks
""")


if __name__ == "__main__":
    main()
