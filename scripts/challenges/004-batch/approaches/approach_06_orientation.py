#!/usr/bin/env python3
"""
Approach 6: Constraint Orientation Space

HYPOTHESIS:
Each cell's valid options define a region/orientation in hyperspace.
The INTERSECTION of valid regions across row/col/block constraints
is the solution space.

KEY INSIGHT:
Your intuition: "orientation in hyperspace that can be exploited to isolate solutions"

Instead of scoring individual digits, we can:
1. Compute the "valid orientation" for each constraint
2. Find where all constraints agree (intersection)
3. The solution is where all orientations align

MECHANISM:
1. For each cell, compute what row constraint "wants" (valid digits bundled)
2. Compute what column constraint "wants"
3. Compute what block constraint "wants"
4. Find the common component (if any) via projection/similarity

This is fundamentally different because we're looking at
CONSTRAINT AGREEMENT rather than scoring individual options.
"""

import sys
sys.path.insert(0, "/home/watmin/work/holon")

from typing import List, Optional, Set, Tuple, Dict
import numpy as np

from common import (
    create_client,
    VectorCache,
    bundle,
    similarity,
    project,
    ApproachResult,
    Timer,
    print_grid_4x4,
    print_grid_9x9,
    validate_4x4,
    validate_9x9,
    count_empty,
    get_available_digits_4x4,
    get_available_digits_9x9,
    PUZZLE_4x4_EASY,
    PUZZLE_9x9_EASY,
    PUZZLE_9x9_HARD,
)


class OrientationSolver:
    """
    Sudoku solver using constraint orientation analysis.

    Key idea: Find where all constraint orientations intersect.
    """

    def __init__(self, size: int = 9, dimensions: int = 16384, verbose: bool = True):
        self.size = size
        self.block_size = 2 if size == 4 else 3
        self.digits = list(range(1, size + 1))
        self.verbose = verbose

        self.client = create_client(dimensions=dimensions)
        self.cache = VectorCache(self.client)

        # Pre-cache vectors
        self.digit_vectors = {d: self.cache.get_digit_vector(d) for d in self.digits}

        self.notes: List[str] = []
        self.iterations = 0

    def log(self, msg: str):
        if self.verbose:
            print(msg)
        self.notes.append(msg)

    def get_available(self, grid, row, col) -> Set[int]:
        if self.size == 4:
            return get_available_digits_4x4(grid, row, col)
        return get_available_digits_9x9(grid, row, col)

    def get_row_used(self, grid: List[List[Optional[int]]], row: int) -> Set[int]:
        """Get digits already used in row."""
        return {grid[row][c] for c in range(self.size) if grid[row][c] is not None}

    def get_col_used(self, grid: List[List[Optional[int]]], col: int) -> Set[int]:
        """Get digits already used in column."""
        return {grid[r][col] for r in range(self.size) if grid[r][col] is not None}

    def get_block_used(self, grid: List[List[Optional[int]]], row: int, col: int) -> Set[int]:
        """Get digits already used in block."""
        br = (row // self.block_size) * self.block_size
        bc = (col // self.block_size) * self.block_size
        used = set()
        for r in range(br, br + self.block_size):
            for c in range(bc, bc + self.block_size):
                if grid[r][c] is not None:
                    used.add(grid[r][c])
        return used

    def constraint_orientation(self, available: Set[int]) -> np.ndarray:
        """
        Compute the orientation vector for a constraint.

        This is the "direction" pointing toward valid options.
        """
        if not available:
            return np.zeros_like(self.digit_vectors[1])

        # Bundle all available digit vectors
        vecs = [self.digit_vectors[d] for d in available]
        return bundle(vecs)

    def find_intersection_digit(self, row_orient: np.ndarray,
                                 col_orient: np.ndarray,
                                 block_orient: np.ndarray,
                                 available: Set[int]) -> Tuple[Optional[int], float]:
        """
        Find the digit that best aligns with all three orientations.

        This is the "intersection" of constraint spaces.
        """
        if not available:
            return None, 0.0

        best_digit = None
        best_score = -1

        for d in available:
            digit_vec = self.digit_vectors[d]

            # How well does this digit align with each constraint?
            row_align = similarity(digit_vec, row_orient)
            col_align = similarity(digit_vec, col_orient)
            block_align = similarity(digit_vec, block_orient)

            # The "intersection" score: how much do all constraints agree?
            # Use minimum (all must agree) or product (amplifies agreement)
            score = min(row_align, col_align, block_align)

            if score > best_score:
                best_score = score
                best_digit = d

        return best_digit, best_score

    def compute_constraint_agreement(self, grid: List[List[Optional[int]]],
                                     row: int, col: int) -> Dict[int, float]:
        """
        Compute how much each available digit aligns with all constraints.

        Returns scores for each digit.
        """
        available = self.get_available(grid, row, col)
        if not available:
            return {}

        # What does each constraint "want"?
        row_available = set(self.digits) - self.get_row_used(grid, row)
        col_available = set(self.digits) - self.get_col_used(grid, col)
        block_available = set(self.digits) - self.get_block_used(grid, row, col)

        row_orient = self.constraint_orientation(row_available)
        col_orient = self.constraint_orientation(col_available)
        block_orient = self.constraint_orientation(block_available)

        # Score each digit
        scores = {}
        for d in available:
            digit_vec = self.digit_vectors[d]

            row_align = similarity(digit_vec, row_orient)
            col_align = similarity(digit_vec, col_orient)
            block_align = similarity(digit_vec, block_orient)

            # Agreement score
            scores[d] = row_align + col_align + block_align

        return scores

    def solve(self, puzzle: List[List[Optional[int]]]) -> Tuple[bool, List[List[int]]]:
        """
        Solve by finding constraint intersection for each cell.
        """
        self.log(f"\n{'='*60}")
        self.log("APPROACH 6: CONSTRAINT ORIENTATION")
        self.log(f"{'='*60}")
        self.log(f"Empty cells: {count_empty(puzzle)}")

        grid = [[cell for cell in row] for row in puzzle]
        max_iterations = self.size * self.size

        while self.iterations < max_iterations:
            self.iterations += 1

            # Find cell with strongest constraint agreement
            best_cell = None
            best_digit = None
            best_score = -1
            best_gap = 0

            for r in range(self.size):
                for c in range(self.size):
                    if grid[r][c] is not None:
                        continue

                    scores = self.compute_constraint_agreement(grid, r, c)
                    if not scores:
                        continue

                    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                    top_d, top_s = sorted_scores[0]
                    second_s = sorted_scores[1][1] if len(sorted_scores) > 1 else 0
                    gap = top_s - second_s

                    if gap > best_gap:
                        best_cell = (r, c)
                        best_digit = top_d
                        best_score = top_s
                        best_gap = gap

            if best_cell is None:
                empty = count_empty(grid)
                if empty == 0:
                    self.log(f"\n✓ All cells filled")
                else:
                    self.log(f"\n⚠ Stuck with {empty} cells")
                break

            r, c = best_cell
            self.log(f"  [Orient] ({r},{c}) → {best_digit} (agreement={best_score:.3f}, gap={best_gap:.3f})")
            grid[r][c] = best_digit

        return count_empty(grid) == 0, grid


def analyze_orientation_space():
    """Analyze how orientation space behaves."""
    print("\n" + "=" * 60)
    print("ANALYSIS: Orientation Space")
    print("=" * 60)

    solver = OrientationSolver(size=9, verbose=False)

    # Test: When constraints agree vs disagree
    print("\n1. Constraint Agreement Test:")
    print("-" * 40)

    # All constraints allow same digits
    available_all = {1, 2, 3}
    orient_all = solver.constraint_orientation(available_all)

    for d in range(1, 10):
        sim = similarity(solver.digit_vectors[d], orient_all)
        marker = "✓ ALLOWED" if d in available_all else ""
        print(f"  Digit {d}: sim={sim:.4f} {marker}")

    print("\n2. Overlapping Constraints:")
    print("-" * 40)

    row_available = {1, 2, 3, 4, 5}
    col_available = {3, 4, 5, 6, 7}
    block_available = {4, 5, 6, 7, 8}

    # Intersection: {4, 5}
    print(f"  Row allows: {row_available}")
    print(f"  Col allows: {col_available}")
    print(f"  Block allows: {block_available}")
    print(f"  Intersection: {row_available & col_available & block_available}")

    row_orient = solver.constraint_orientation(row_available)
    col_orient = solver.constraint_orientation(col_available)
    block_orient = solver.constraint_orientation(block_available)

    print("\n  Digit alignments:")
    for d in range(1, 10):
        digit_vec = solver.digit_vectors[d]
        r_sim = similarity(digit_vec, row_orient)
        c_sim = similarity(digit_vec, col_orient)
        b_sim = similarity(digit_vec, block_orient)
        total = r_sim + c_sim + b_sim
        in_intersection = d in (row_available & col_available & block_available)
        marker = "← INTERSECTION" if in_intersection else ""
        print(f"    {d}: R={r_sim:.3f} C={c_sim:.3f} B={b_sim:.3f} Total={total:.3f} {marker}")


def test_4x4():
    print("\n" + "=" * 60)
    print("TEST: 4x4 with Orientation")
    print("=" * 60)

    print("\nInput:")
    print_grid_4x4(PUZZLE_4x4_EASY)

    solver = OrientationSolver(size=4, verbose=True)
    with Timer() as timer:
        solved, grid = solver.solve(PUZZLE_4x4_EASY)

    print("\nResult:")
    print_grid_4x4(grid)

    valid, msg = validate_4x4(grid)
    print(f"\nValid: {valid} - {msg}")
    return valid


def test_9x9_hard():
    print("\n" + "=" * 60)
    print("TEST: 9x9 HARD with Orientation")
    print("=" * 60)

    print("\nInput:")
    print_grid_9x9(PUZZLE_9x9_HARD)

    solver = OrientationSolver(size=9, verbose=True)
    with Timer() as timer:
        solved, grid = solver.solve(PUZZLE_9x9_HARD)

    print("\nResult:")
    print_grid_9x9(grid)

    valid, msg = validate_9x9(grid)
    cells = 58 - count_empty(grid)
    print(f"\nCells: {cells}/58, Valid: {valid}")
    print(f"Time: {timer.elapsed:.2f}s")
    return valid


def main():
    print("=" * 60)
    print("APPROACH 6: CONSTRAINT ORIENTATION SPACE")
    print("=" * 60)
    print("\nHypothesis: Constraint orientations intersect at solutions.")

    # Analyze the orientation space
    analyze_orientation_space()

    # Test puzzles
    result_4x4 = test_4x4()
    result_hard = test_9x9_hard()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"4x4: {'✓' if result_4x4 else '✗'}")
    print(f"9x9 Hard: {'✓' if result_hard else '✗'}")


if __name__ == "__main__":
    main()
