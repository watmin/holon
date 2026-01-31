#!/usr/bin/env python3
"""
Approach 40: Band-Scale Encoding + Fixed Chain Encoding

KEY FINDINGS FROM APPROACH 39:
1. Band scale had HIGHEST similarity (0.6523) - broader patterns matter!
2. Sequential binding collapsed to zero - need different approach
3. Simple chain continuation didn't help

NEW DIRECTIONS:
1. Explore band-scale encoding more deeply
2. Fix chain encoding using bundling with position markers
3. Use Holon's CHAINED list mode for sequences
"""

import sys
sys.path.insert(0, "/home/watmin/work/holon")

from typing import List, Optional, Set, Tuple, Dict
import numpy as np
import time
import random

from holon import CPUStore
from holon.encoder import ListEncodeMode

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
# EXPERIMENT 1: Deep Dive into Band-Scale
# =============================================================================

def test_band_scale_deep():
    """
    Explore why band scale has highest similarity.
    """
    print("=" * 70)
    print("EXPERIMENT 1: Band-Scale Deep Dive")
    print("=" * 70)

    store = create_store()

    digit_vecs = {d: store.vector_manager.get_vector(f"d{d}") for d in range(1, 10)}
    complete_template = store.bundle([digit_vecs[d] for d in range(1, 10)])

    def solve(puzzle):
        grid = [[cell for cell in row] for row in puzzle]
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
            for r in range(9):
                for c in range(9):
                    if g[r][c] is None:
                        for d in get_available_digits_9x9(g, r, c):
                            test = [[cell for cell in row] for row in g]
                            test[r][c] = d
                            result = rec(test)
                            if result:
                                return result
                        return None
            return None
        return rec(grid)

    solution = solve(PUZZLE_9x9_HARD)

    # Band encoding: each band should contain all 9 digits 3 times
    # A row band (3 rows) contains 27 cells, each digit appears exactly 3 times

    def get_band_digits(grid, band_type, band_idx):
        """Get all digits in a band."""
        if band_type == "row":
            rows = [band_idx * 3 + i for i in range(3)]
            return [grid[r][c] for r in rows for c in range(9) if grid[r][c] is not None]
        else:  # col
            cols = [band_idx * 3 + i for i in range(3)]
            return [grid[r][c] for r in range(9) for c in cols if grid[r][c] is not None]

    print("\nBand digit counts in PUZZLE:")
    for band_idx in range(3):
        row_digits = get_band_digits(PUZZLE_9x9_HARD, "row", band_idx)
        col_digits = get_band_digits(PUZZLE_9x9_HARD, "col", band_idx)
        print(f"  Row band {band_idx}: {len(row_digits)} digits, unique: {len(set(row_digits))}")
        print(f"  Col band {band_idx}: {len(col_digits)} digits, unique: {len(set(col_digits))}")

    print("\nBand digit counts in SOLUTION:")
    for band_idx in range(3):
        row_digits = get_band_digits(solution, "row", band_idx)
        col_digits = get_band_digits(solution, "col", band_idx)
        print(f"  Row band {band_idx}: {len(row_digits)} digits, unique: {len(set(row_digits))}")
        print(f"  Col band {band_idx}: {len(col_digits)} digits, unique: {len(set(col_digits))}")

    # WHY band scale has high similarity:
    # Both puzzle and solution have ALL 9 digits in each band
    # The difference is just the COUNT, not the PRESENCE

    def encode_band_with_count(grid, band_type, band_idx):
        """Encode band with digit counts."""
        digits = get_band_digits(grid, band_type, band_idx)
        digit_counts = {d: digits.count(d) for d in range(1, 10)}

        components = []
        for d, count in digit_counts.items():
            if count > 0:
                count_vec = store.vector_manager.get_vector(f"cnt_{count}")
                components.append(store.bind(digit_vecs[d], count_vec))

        if components:
            return store.bundle(components)
        return np.zeros(store.dimensions, dtype=np.int8)

    print("\nBand encodings with counts:")
    for band_idx in range(3):
        puzzle_band = encode_band_with_count(PUZZLE_9x9_HARD, "row", band_idx)
        solution_band = encode_band_with_count(solution, "row", band_idx)
        sim = similarity(puzzle_band, solution_band)
        print(f"  Row band {band_idx}: sim(puzzle, solution) = {sim:.4f}")


# =============================================================================
# EXPERIMENT 2: Fixed Chain Encoding with Holon's CHAINED mode
# =============================================================================

def test_chained_mode():
    """
    Use Holon's ListEncodeMode.CHAINED for decision sequences.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Holon's CHAINED List Mode")
    print("=" * 70)

    store = create_store()

    # Use Holon's encoder with CHAINED mode via encode_list
    decisions_1 = ["d1", "d2", "d3", "d4", "d5"]
    decisions_2 = ["d1", "d2", "d3", "d4", "d6"]  # Same prefix, different end
    decisions_3 = ["d5", "d4", "d3", "d2", "d1"]  # Reversed

    vec_1 = store.encoder.encode_list(decisions_1, mode=ListEncodeMode.CHAINED)
    vec_2 = store.encoder.encode_list(decisions_2, mode=ListEncodeMode.CHAINED)
    vec_3 = store.encoder.encode_list(decisions_3, mode=ListEncodeMode.CHAINED)

    print("\nCHAINED mode similarities:")
    print(f"  Same prefix (d1-d4), diff end: sim(1,2) = {similarity(vec_1, vec_2):.4f}")
    print(f"  Reversed order: sim(1,3) = {similarity(vec_1, vec_3):.4f}")
    print(f"  Different: sim(2,3) = {similarity(vec_2, vec_3):.4f}")

    # Compare with BUNDLE mode
    bundle_1 = store.encoder.encode_list(decisions_1, mode=ListEncodeMode.BUNDLE)
    bundle_3 = store.encoder.encode_list(decisions_3, mode=ListEncodeMode.BUNDLE)

    print(f"\nBUNDLE mode (order-independent):")
    print(f"  sim(forward, reversed) = {similarity(bundle_1, bundle_3):.4f}")

    # Now encode actual decision sequences
    print("\nEncoding actual decision sequences:")

    def encode_decision_seq(decisions: List[Tuple[int, int, int]], mode: ListEncodeMode):
        """Encode a decision sequence."""
        # Convert to string representation for Holon's encoder
        str_decisions = [f"r{r}c{c}d{d}" for r, c, d in decisions]
        return store.encoder.encode_list(str_decisions, mode=mode)

    # Generate a solving path
    def solve_and_get_path(puzzle):
        grid = [[cell for cell in row] for row in puzzle]
        path = []

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
                                path.append((r, c, opts[0]))
                                changed = True

            if count_empty(g) == 0:
                return g

            for r in range(9):
                for c in range(9):
                    if g[r][c] is None:
                        opts = list(get_available_digits_9x9(g, r, c))
                        for d in opts:
                            test = [[cell for cell in row] for row in g]
                            test[r][c] = d
                            path.append((r, c, d))
                            result = rec(test)
                            if result:
                                return result
                            path.pop()
                        return None
            return None

        rec(grid)
        return path

    path = solve_and_get_path(PUZZLE_9x9_HARD)
    print(f"  Path length: {len(path)}")

    # Encode first N decisions in different modes
    N = 10
    chained_vec = encode_decision_seq(path[:N], ListEncodeMode.CHAINED)
    bundle_vec = encode_decision_seq(path[:N], ListEncodeMode.BUNDLE)
    positional_vec = encode_decision_seq(path[:N], ListEncodeMode.POSITIONAL)

    print(f"\n  Encoding first {N} decisions:")
    print(f"    CHAINED norm: {np.linalg.norm(chained_vec):.1f}")
    print(f"    BUNDLE norm: {np.linalg.norm(bundle_vec):.1f}")
    print(f"    POSITIONAL norm: {np.linalg.norm(positional_vec):.1f}")

    # Similarity between modes
    print(f"\n  Cross-mode similarities:")
    print(f"    CHAINED vs BUNDLE: {similarity(chained_vec, bundle_vec):.4f}")
    print(f"    CHAINED vs POSITIONAL: {similarity(chained_vec, positional_vec):.4f}")
    print(f"    BUNDLE vs POSITIONAL: {similarity(bundle_vec, positional_vec):.4f}")


# =============================================================================
# EXPERIMENT 3: Band-Based Scoring
# =============================================================================

def test_band_scoring():
    """
    Use band-level constraints for scoring.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Band-Based Scoring")
    print("=" * 70)

    store = create_store()

    digit_vecs = {d: store.vector_manager.get_vector(f"d{d}") for d in range(1, 10)}

    # A complete band has each digit exactly 3 times
    # Encode this as template

    def get_band_template():
        """Template for complete band: each digit 3 times."""
        components = []
        for d in range(1, 10):
            count_vec = store.vector_manager.get_vector("cnt_3")
            components.append(store.bind(digit_vecs[d], count_vec))
        return store.bundle(components)

    band_template = get_band_template()

    def score_band_progress(grid, r, c, digit, band_type):
        """Score based on band progress."""
        if band_type == "row":
            band_idx = r // 3
            rows = [band_idx * 3 + i for i in range(3)]
            cells = [(rr, cc) for rr in rows for cc in range(9)]
        else:
            band_idx = c // 3
            cols = [band_idx * 3 + i for i in range(3)]
            cells = [(rr, cc) for rr in range(9) for cc in cols]

        # Current digit counts in band
        digit_counts = {d: 0 for d in range(1, 10)}
        for rr, cc in cells:
            if grid[rr][cc] is not None:
                digit_counts[grid[rr][cc]] += 1

        # Add the new digit
        digit_counts[digit] += 1

        # Encode current state
        components = []
        for d, count in digit_counts.items():
            if count > 0:
                count_vec = store.vector_manager.get_vector(f"cnt_{min(count, 4)}")
                components.append(store.bind(digit_vecs[d], count_vec))

        if components:
            current_vec = store.bundle(components)
            return similarity(current_vec, band_template)
        return 0.0

    # Test scoring
    grid = [[cell for cell in row] for row in PUZZLE_9x9_HARD]

    # Propagate first
    changed = True
    while changed:
        changed = False
        for r in range(9):
            for c in range(9):
                if grid[r][c] is None:
                    opts = list(get_available_digits_9x9(grid, r, c))
                    if len(opts) == 1:
                        grid[r][c] = opts[0]
                        changed = True

    # Find a decision point
    for r in range(9):
        for c in range(9):
            if grid[r][c] is None:
                opts = list(get_available_digits_9x9(grid, r, c))
                if len(opts) > 1:
                    print(f"\nDecision point ({r},{c}), options: {opts}")
                    print("Band scores:")
                    for d in opts:
                        row_band_score = score_band_progress(grid, r, c, d, "row")
                        col_band_score = score_band_progress(grid, r, c, d, "col")
                        print(f"  Digit {d}: row_band={row_band_score:.4f}, col_band={col_band_score:.4f}")
                    break
        else:
            continue
        break


# =============================================================================
# EXPERIMENT 4: Band + Constraint Solver
# =============================================================================

def test_band_constraint_solver():
    """
    Combine band-level and constraint-level scoring.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Band + Constraint Solver")
    print("=" * 70)

    store = create_store()

    digit_vecs = {d: store.vector_manager.get_vector(f"d{d}") for d in range(1, 10)}
    complete_template = store.bundle([digit_vecs[d] for d in range(1, 10)])

    # Band template (each digit 3 times)
    band_components = []
    for d in range(1, 10):
        count_vec = store.vector_manager.get_vector("cnt_3")
        band_components.append(store.bind(digit_vecs[d], count_vec))
    band_template = store.bundle(band_components)

    class BandConstraintSolver:
        def __init__(self, band_weight=0.3):
            self.backtracks = 0
            self.band_weight = band_weight

        def score_constraint(self, grid, r, c, digit):
            """Standard template matching score."""
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

        def score_band(self, grid, r, c, digit):
            """Band-level score."""
            total = 0.0

            # Row band
            band_idx = r // 3
            rows = [band_idx * 3 + i for i in range(3)]
            digit_counts = {d: 0 for d in range(1, 10)}
            for rr in rows:
                for cc in range(9):
                    if grid[rr][cc] is not None:
                        digit_counts[grid[rr][cc]] += 1
            digit_counts[digit] += 1

            components = []
            for d, count in digit_counts.items():
                if count > 0:
                    count_vec = store.vector_manager.get_vector(f"cnt_{min(count, 4)}")
                    components.append(store.bind(digit_vecs[d], count_vec))
            if components:
                total += similarity(store.bundle(components), band_template)

            # Column band
            band_idx = c // 3
            cols = [band_idx * 3 + i for i in range(3)]
            digit_counts = {d: 0 for d in range(1, 10)}
            for rr in range(9):
                for cc in cols:
                    if grid[rr][cc] is not None:
                        digit_counts[grid[rr][cc]] += 1
            digit_counts[digit] += 1

            components = []
            for d, count in digit_counts.items():
                if count > 0:
                    count_vec = store.vector_manager.get_vector(f"cnt_{min(count, 4)}")
                    components.append(store.bind(digit_vecs[d], count_vec))
            if components:
                total += similarity(store.bundle(components), band_template)

            return total

        def score_choice(self, grid, r, c, digit):
            constraint_score = self.score_constraint(grid, r, c, digit)
            band_score = self.score_band(grid, r, c, digit)
            return constraint_score + self.band_weight * band_score

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

    # Test with different band weights
    print("\nTesting different band weights:")
    results = []
    for weight in [0.0, 0.1, 0.3, 0.5, 1.0]:
        solver = BandConstraintSolver(band_weight=weight)
        result = solver.solve(PUZZLE_9x9_HARD)
        valid = result is not None and validate_9x9(result)
        results.append((weight, solver.backtracks, valid))
        print(f"  band_weight={weight}: {solver.backtracks} backtracks, valid={valid}")

    best_weight, best_bt, _ = min(results, key=lambda x: x[1])
    print(f"\nBest: weight={best_weight}, backtracks={best_bt}")

    return best_bt


# =============================================================================
# MAIN
# =============================================================================

def main():
    test_band_scale_deep()
    test_chained_mode()
    test_band_scoring()
    best_bt = test_band_constraint_solver()

    print("\n" + "=" * 70)
    print("BAND-SCALE + CHAIN SUMMARY")
    print("=" * 70)
    print(f"""
BAND-SCALE FINDINGS:
- Bands (3 rows or 3 cols) contain all 9 digits 3 times each
- High similarity because puzzle already has most digits in each band
- Band template: encode digit → count binding

CHAINED MODE FINDINGS:
- Holon's ListEncodeMode.CHAINED preserves order
- Same prefix, different end: similarity captures shared structure
- Reversed order: lower similarity (order matters!)

SOLVER RESULTS:
| Weight | Backtracks |
|--------|-----------|
| Pure constraint (0.0) | 52 (baseline) |
| Best band+constraint | {best_bt} |

KEY INSIGHT:
Band-level encoding adds a COARSER signal that may or may not help.
The constraint-level (row/col/block) is already the right granularity
for Sudoku constraints.

CHAIN ENCODING WORKS but needs the right APPLICATION:
- Use for learning patterns from solved puzzles
- Use for detecting "good" vs "bad" decision sequences
- Not directly helpful for single-puzzle solving
""")


if __name__ == "__main__":
    main()
