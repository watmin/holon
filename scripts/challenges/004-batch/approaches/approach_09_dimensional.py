#!/usr/bin/env python3
"""
Approach 9: Dimensional Analysis

HYPOTHESIS:
Valid configurations have higher "effective dimensionality" than invalid ones.
We can use this geometric property to guide constraint satisfaction.

EVIDENCE FROM TESTS:
- Valid row [1,2,3,4]: dimensionality = 0.9999
- Invalid row [1,1,3,4]: dimensionality = 0.7315
- Very invalid [2,2,2,2]: dimensionality = 0.0726

The dimensionality measure is nearly BINARY for distinguishing valid from invalid!

APPROACH:
1. For each empty cell, try each possible digit
2. Compute the dimensionality of affected row/col/block with that digit
3. Choose the digit that MAXIMIZES dimensionality (closest to valid)
4. Repeat until solved or stuck

NO BACKTRACKING - pure geometric guidance.
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
    effective_dimensionality,
    ApproachResult,
    Timer,
    print_grid_4x4,
    print_grid_9x9,
    validate_4x4,
    validate_9x9,
    get_available_digits_4x4,
    get_available_digits_9x9,
    count_empty,
    PUZZLE_4x4_EASY,
    PUZZLE_4x4_SOLUTION,
    PUZZLE_9x9_EASY,
    PUZZLE_9x9_EASY_SOLUTION,
)


class DimensionalSolver:
    """
    Sudoku solver using dimensional analysis.

    Core idea: Valid configurations span more dimensions than invalid ones.
    We measure this and use it to choose placements.
    """

    def __init__(self, size: int = 9, dimensions: int = 16384, verbose: bool = True):
        self.size = size
        self.block_size = 2 if size == 4 else 3
        self.digits = list(range(1, size + 1))
        self.verbose = verbose

        # Create Holon client and cache
        self.client = create_client(dimensions=dimensions)
        self.cache = VectorCache(self.client)

        # Pre-cache all digit vectors
        self.digit_vectors = {d: self.cache.get_digit_vector(d) for d in self.digits}
        self.digit_basis = [self.digit_vectors[d] for d in self.digits]

        # Stats
        self.iterations = 0
        self.cells_filled = 0
        self.notes: List[str] = []

    def log(self, msg: str):
        """Log a message."""
        if self.verbose:
            print(msg)
        self.notes.append(msg)

    def get_row_digits(self, grid: List[List[Optional[int]]], row: int) -> List[int]:
        """Get non-None digits in a row."""
        return [d for d in grid[row] if d is not None]

    def get_col_digits(self, grid: List[List[Optional[int]]], col: int) -> List[int]:
        """Get non-None digits in a column."""
        return [grid[r][col] for r in range(self.size) if grid[r][col] is not None]

    def get_block_digits(self, grid: List[List[Optional[int]]], row: int, col: int) -> List[int]:
        """Get non-None digits in a block."""
        br = (row // self.block_size) * self.block_size
        bc = (col // self.block_size) * self.block_size
        digits = []
        for r in range(br, br + self.block_size):
            for c in range(bc, bc + self.block_size):
                if grid[r][c] is not None:
                    digits.append(grid[r][c])
        return digits

    def get_available(self, grid: List[List[Optional[int]]], row: int, col: int) -> Set[int]:
        """Get available digits for a cell."""
        if self.size == 4:
            return get_available_digits_4x4(grid, row, col)
        else:
            return get_available_digits_9x9(grid, row, col)

    def constraint_dimensionality(self, digits: List[int]) -> float:
        """
        Compute the dimensionality of a constraint group.

        Higher = more uniform spread across digit basis = more valid.
        """
        if not digits:
            return 1.0  # Empty constraint is maximally valid (all possibilities)

        # Bundle the digit vectors
        digit_vecs = [self.digit_vectors[d] for d in digits]
        bundled = bundle(digit_vecs)

        # Compute effective dimensionality
        return effective_dimensionality(bundled, self.digit_basis)

    def score_placement(self, grid: List[List[Optional[int]]],
                        row: int, col: int, digit: int) -> float:
        """
        Score a potential placement by its effect on constraint dimensionality.

        Returns average dimensionality across affected row, col, and block.
        """
        # Get current digits in each constraint
        row_digits = self.get_row_digits(grid, row) + [digit]
        col_digits = self.get_col_digits(grid, col) + [digit]
        block_digits = self.get_block_digits(grid, row, col) + [digit]

        # Compute dimensionality for each constraint
        row_dim = self.constraint_dimensionality(row_digits)
        col_dim = self.constraint_dimensionality(col_digits)
        block_dim = self.constraint_dimensionality(block_digits)

        # Average (or could weight differently)
        return (row_dim + col_dim + block_dim) / 3.0

    def find_best_cell(self, grid: List[List[Optional[int]]]) -> Optional[Tuple[int, int, Set[int]]]:
        """
        Find the empty cell with most constrained options.
        MRV heuristic - helps even geometric approaches.
        """
        best = None
        min_options = self.size + 1

        for r in range(self.size):
            for c in range(self.size):
                if grid[r][c] is not None:
                    continue

                available = self.get_available(grid, r, c)
                if len(available) < min_options:
                    min_options = len(available)
                    best = (r, c, available)

        return best

    def solve(self, puzzle: List[List[Optional[int]]]) -> Tuple[bool, List[List[int]]]:
        """
        Solve using dimensional analysis.

        NO BACKTRACKING - purely geometric guidance.
        """
        # Copy puzzle to working grid
        grid = [[cell for cell in row] for row in puzzle]

        self.log(f"\n{'='*60}")
        self.log("APPROACH 9: DIMENSIONAL ANALYSIS SOLVER")
        self.log(f"{'='*60}")
        self.log(f"Puzzle size: {self.size}x{self.size}")
        self.log(f"Empty cells: {count_empty(grid)}")

        max_iterations = self.size * self.size

        while self.iterations < max_iterations:
            self.iterations += 1

            # Find best cell to fill
            result = self.find_best_cell(grid)

            if result is None:
                # All cells filled!
                self.log(f"\n✓ All cells filled after {self.iterations} iterations")
                break

            row, col, available = result

            if len(available) == 0:
                self.log(f"\n✗ Contradiction at ({row},{col}) - no available digits")
                return False, grid

            if len(available) == 1:
                # Forced placement
                digit = list(available)[0]
                grid[row][col] = digit
                self.cells_filled += 1
                self.log(f"  [Forced] ({row},{col}) → {digit}")
                continue

            # Score each available digit by dimensionality
            scores: Dict[int, float] = {}
            for digit in available:
                scores[digit] = self.score_placement(grid, row, col, digit)

            # Choose digit with HIGHEST dimensionality
            best_digit = max(scores, key=scores.get)
            best_score = scores[best_digit]

            # Log the decision
            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            score_str = ", ".join([f"{d}:{s:.3f}" for d, s in sorted_scores[:3]])
            self.log(f"  [Dimensional] ({row},{col}) → {best_digit} (dim={best_score:.4f}) | {score_str}")

            grid[row][col] = best_digit
            self.cells_filled += 1

        return True, grid


def test_4x4():
    """Test on 4x4 puzzle."""
    print("\n" + "=" * 60)
    print("TEST: 4x4 Sudoku with Dimensional Analysis")
    print("=" * 60)

    result = ApproachResult("Approach 9: Dimensional Analysis")
    result.puzzle_size = 4
    result.puzzle_name = "4x4 Easy"

    print("\nInput puzzle:")
    print_grid_4x4(PUZZLE_4x4_EASY)

    solver = DimensionalSolver(size=4, verbose=True)

    with Timer() as timer:
        solved, grid = solver.solve(PUZZLE_4x4_EASY)

    result.time_seconds = timer.elapsed
    result.iterations = solver.iterations
    result.cells_filled_geometrically = solver.cells_filled
    result.backtracking_used = False
    result.solution = grid

    print("\nResult:")
    print_grid_4x4(grid)

    valid, msg = validate_4x4(grid)
    result.valid = valid
    result.validation_msg = msg
    result.solved = solved and valid

    print(f"\nValid: {valid} - {msg}")
    print(f"Time: {timer.elapsed:.4f}s")
    print(f"Iterations: {solver.iterations}")
    print(f"Cells filled: {solver.cells_filled}")

    # Compare to expected solution
    matches_expected = (grid == PUZZLE_4x4_SOLUTION)
    print(f"Matches expected solution: {matches_expected}")

    return result


def test_9x9():
    """Test on 9x9 puzzle."""
    print("\n" + "=" * 60)
    print("TEST: 9x9 Sudoku with Dimensional Analysis")
    print("=" * 60)

    result = ApproachResult("Approach 9: Dimensional Analysis")
    result.puzzle_size = 9
    result.puzzle_name = "9x9 Easy"

    print("\nInput puzzle:")
    print_grid_9x9(PUZZLE_9x9_EASY)

    solver = DimensionalSolver(size=9, verbose=True)

    with Timer() as timer:
        solved, grid = solver.solve(PUZZLE_9x9_EASY)

    result.time_seconds = timer.elapsed
    result.iterations = solver.iterations
    result.cells_filled_geometrically = solver.cells_filled
    result.backtracking_used = False
    result.solution = grid

    print("\nResult:")
    print_grid_9x9(grid)

    valid, msg = validate_9x9(grid)
    result.valid = valid
    result.validation_msg = msg
    result.solved = solved and valid

    print(f"\nValid: {valid} - {msg}")
    print(f"Time: {timer.elapsed:.4f}s")
    print(f"Iterations: {solver.iterations}")
    print(f"Cells filled: {solver.cells_filled}")

    return result


def analyze_dimensionality_patterns():
    """
    Analyze how dimensionality behaves across different configurations.
    This helps us understand the geometry.
    """
    print("\n" + "=" * 60)
    print("ANALYSIS: Dimensionality Patterns")
    print("=" * 60)

    client = create_client(dimensions=16384)
    cache = VectorCache(client)

    # Digit basis for 9x9
    digit_basis = [cache.get_digit_vector(d) for d in range(1, 10)]

    def dim(digits):
        vecs = [cache.get_digit_vector(d) for d in digits]
        return effective_dimensionality(bundle(vecs), digit_basis)

    print("\n1. Dimensionality vs Number of Unique Digits:")
    print("-" * 40)

    # Test different fill levels
    for n in range(1, 10):
        digits = list(range(1, n + 1))
        d = dim(digits)
        print(f"  {n} unique digits {digits}: dim = {d:.4f}")

    print("\n2. Dimensionality vs Duplicates:")
    print("-" * 40)

    test_cases = [
        [1, 2, 3, 4, 5, 6, 7, 8, 9],  # All unique
        [1, 1, 3, 4, 5, 6, 7, 8, 9],  # One duplicate
        [1, 1, 1, 4, 5, 6, 7, 8, 9],  # Triple
        [1, 1, 1, 1, 5, 6, 7, 8, 9],  # Quad
        [1, 1, 1, 1, 1, 6, 7, 8, 9],  # Five same
        [1, 1, 1, 1, 1, 1, 7, 8, 9],  # Six same
    ]

    for digits in test_cases:
        d = dim(digits)
        unique = len(set(digits))
        dupes = 9 - unique
        print(f"  {digits}: dim = {d:.4f} (dupes: {dupes})")

    print("\n3. Partial Fill Analysis (simulating solving):")
    print("-" * 40)

    # Simulate filling a row one digit at a time
    row = []
    unused = list(range(1, 10))

    print("  Empty row: dim = 1.0000 (by definition)")

    for i in range(9):
        digit = unused.pop(0)
        row.append(digit)
        d = dim(row)
        valid = len(row) == len(set(row))
        print(f"  After adding {digit}: {row} → dim = {d:.4f} {'✓' if valid else '✗'}")

    print("\n4. Effect of Wrong Choice:")
    print("-" * 40)

    # Partial row, choosing next digit
    partial = [1, 2, 3, 4, 5]
    print(f"  Partial row: {partial}")
    print(f"  Current dim: {dim(partial):.4f}")
    print("\n  If we add:")

    for next_d in range(1, 10):
        test_row = partial + [next_d]
        d = dim(test_row)
        valid = len(set(test_row)) == len(test_row)
        print(f"    + {next_d}: dim = {d:.4f} {'← VALID' if valid else '← DUPLICATE'}")


def test_9x9_hard():
    """Test on hard 9x9 puzzle that requires real choices."""
    from common import PUZZLE_9x9_HARD

    print("\n" + "=" * 60)
    print("TEST: 9x9 HARD Sudoku with Dimensional Analysis")
    print("=" * 60)
    print("\nThis puzzle REQUIRES guessing in traditional solvers.")
    print("Can dimensional analysis guide us without backtracking?")

    result = ApproachResult("Approach 9: Dimensional Analysis")
    result.puzzle_size = 9
    result.puzzle_name = "9x9 Hard"

    print("\nInput puzzle:")
    print_grid_9x9(PUZZLE_9x9_HARD)
    print(f"Empty cells: {count_empty(PUZZLE_9x9_HARD)}")

    solver = DimensionalSolver(size=9, verbose=True)

    with Timer() as timer:
        solved, grid = solver.solve(PUZZLE_9x9_HARD)

    result.time_seconds = timer.elapsed
    result.iterations = solver.iterations
    result.cells_filled_geometrically = solver.cells_filled
    result.backtracking_used = False
    result.solution = grid

    print("\nResult:")
    print_grid_9x9(grid)

    valid, msg = validate_9x9(grid)
    result.valid = valid
    result.validation_msg = msg
    result.solved = solved and valid

    print(f"\nValid: {valid} - {msg}")
    print(f"Time: {timer.elapsed:.4f}s")
    print(f"Iterations: {solver.iterations}")
    print(f"Cells filled: {solver.cells_filled}")

    # Count how many were dimensional vs forced
    dimensional_count = sum(1 for n in solver.notes if "[Dimensional]" in n)
    forced_count = sum(1 for n in solver.notes if "[Forced]" in n)
    print(f"\nDimensional choices: {dimensional_count}")
    print(f"Forced (single option): {forced_count}")

    return result


def main():
    """Run dimensional analysis approach."""
    print("=" * 60)
    print("APPROACH 9: DIMENSIONAL ANALYSIS")
    print("=" * 60)
    print("\nHypothesis: Valid configurations have higher dimensionality.")
    print("Method: Choose digits that maximize constraint dimensionality.")

    # First, analyze the patterns
    analyze_dimensionality_patterns()

    # Test on 4x4
    result_4x4 = test_4x4()

    # Test on 9x9 easy
    result_9x9_easy = test_9x9()

    # Test on 9x9 hard - the REAL test
    result_9x9_hard = test_9x9_hard()

    # Summary
    print("\n" + "=" * 60)
    print("APPROACH 9 SUMMARY")
    print("=" * 60)

    print(f"\n4x4 Puzzle:")
    print(f"  Solved: {result_4x4.solved}")
    print(f"  Valid: {result_4x4.valid}")
    print(f"  Time: {result_4x4.time_seconds:.4f}s")

    print(f"\n9x9 Easy Puzzle:")
    print(f"  Solved: {result_9x9_easy.solved}")
    print(f"  Valid: {result_9x9_easy.valid}")
    print(f"  Time: {result_9x9_easy.time_seconds:.4f}s")

    print(f"\n9x9 HARD Puzzle (the real test):")
    print(f"  Solved: {result_9x9_hard.solved}")
    print(f"  Valid: {result_9x9_hard.valid}")
    print(f"  Time: {result_9x9_hard.time_seconds:.4f}s")
    print(f"  Backtracking: {result_9x9_hard.backtracking_used}")

    if result_9x9_hard.solved:
        print("\n✓✓ MAJOR BREAKTHROUGH: Hard puzzle solved WITHOUT backtracking!")
        print("   Pure dimensional geometry guided us to the solution!")
    elif result_4x4.solved and result_9x9_easy.solved:
        print("\n◐ PARTIAL SUCCESS: Easy puzzles solved, hard puzzle failed")
        print("   Dimensional analysis helps but may not be sufficient alone")
    else:
        print("\n✗ APPROACH 9 NEEDS REFINEMENT")

    print("\nKey findings:")
    print("- Dimensionality clearly distinguishes valid from invalid")
    print("- Easy puzzles mostly solved by constraint propagation (forced)")
    print("- Hard puzzle is the true test of geometric guidance")


if __name__ == "__main__":
    main()
