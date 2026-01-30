#!/usr/bin/env python3
"""
Approach 8: Inverse Encoding

HYPOTHESIS:
Instead of encoding cells as position⊙digit, what if we encode
the INVERSE: what position SHOULD have this digit?

KEY IDEA:
- Normal encoding: cell = bind(position, digit)
- Inverse encoding: digit_location = bind(digit, position)

This encodes "digit 5 is at position (2,3)" differently than
"position (2,3) contains digit 5".

WHY THIS MIGHT HELP:
- Different geometric perspective on the same information
- Might make certain patterns more apparent
- Could enable different query patterns

APPROACH:
1. Encode constraints from digit perspective ("digit X must be in row Y")
2. Query which position each digit should occupy
3. See if this gives different/better signal
"""

import sys
sys.path.insert(0, "/home/watmin/work/holon")

from typing import List, Optional, Set, Tuple, Dict
import numpy as np

from common import (
    create_client,
    VectorCache,
    bind,
    unbind,
    bundle,
    similarity,
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


class InverseSolver:
    """
    Solve using inverse encoding (digit→position instead of position→digit).
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
        self.position_vectors = {}
        for r in range(size):
            for c in range(size):
                self.position_vectors[(r, c)] = self.cache.get_position_vector(r, c)

        self.notes: List[str] = []

    def log(self, msg: str):
        if self.verbose:
            print(msg)
        self.notes.append(msg)

    def get_available(self, grid, row, col) -> Set[int]:
        if self.size == 4:
            return get_available_digits_4x4(grid, row, col)
        return get_available_digits_9x9(grid, row, col)

    def encode_digit_locations(self, grid: List[List[Optional[int]]]) -> Dict[int, np.ndarray]:
        """
        Encode where each digit is placed (inverse encoding).

        Returns dict mapping digit -> vector encoding its known positions.
        """
        digit_locs = {d: [] for d in self.digits}

        for r in range(self.size):
            for c in range(self.size):
                if grid[r][c] is not None:
                    d = grid[r][c]
                    # Inverse: bind digit to position (digit is the "key")
                    binding = bind(self.digit_vectors[d], self.position_vectors[(r, c)])
                    digit_locs[d].append(binding)

        # Bundle all known positions for each digit
        result = {}
        for d in self.digits:
            if digit_locs[d]:
                result[d] = bundle(digit_locs[d])
            else:
                result[d] = np.zeros_like(self.digit_vectors[1])

        return result

    def query_digit_position(self, digit_vec: np.ndarray,
                             row: int, col: int) -> float:
        """
        Query how strongly a digit is associated with a position.

        Unbind the digit from its location encoding to see if this position matches.
        """
        pos_vec = self.position_vectors[(row, col)]
        return similarity(digit_vec, pos_vec)

    def encode_row_constraint(self, row: int, digit: int,
                               grid: List[List[Optional[int]]]) -> np.ndarray:
        """
        Encode: "digit D must go in one of the empty cells in row R"

        This is a superposition of possible positions for this digit in this row.
        """
        possible_positions = []
        for c in range(self.size):
            if grid[row][c] is None:
                available = self.get_available(grid, row, c)
                if digit in available:
                    possible_positions.append(self.position_vectors[(row, c)])

        if possible_positions:
            return bundle(possible_positions)
        return np.zeros_like(self.digit_vectors[1])

    def score_by_inverse(self, grid: List[List[Optional[int]]],
                         row: int, col: int) -> Dict[int, float]:
        """
        Score digits by how strongly they "want" to be at this position.

        For each available digit, encode its constraints and see how strongly
        this position is indicated.
        """
        available = self.get_available(grid, row, col)
        if not available:
            return {}

        scores = {}
        for d in available:
            # Get row constraint for this digit
            row_constraint = self.encode_row_constraint(row, d, grid)

            # How strongly does this constraint indicate this position?
            pos_vec = self.position_vectors[(row, col)]
            score = similarity(row_constraint, pos_vec)
            scores[d] = score

        return scores

    def solve(self, puzzle: List[List[Optional[int]]]) -> Tuple[bool, List[List[int]]]:
        """
        Solve using inverse encoding approach.
        """
        self.log(f"\n{'='*60}")
        self.log("APPROACH 8: INVERSE ENCODING")
        self.log(f"{'='*60}")
        self.log(f"Empty cells: {count_empty(puzzle)}")

        grid = [[cell for cell in row] for row in puzzle]
        max_iterations = self.size * self.size

        for iteration in range(max_iterations):
            # Find best placement
            best_cell = None
            best_digit = None
            best_gap = -1

            for r in range(self.size):
                for c in range(self.size):
                    if grid[r][c] is not None:
                        continue

                    scores = self.score_by_inverse(grid, r, c)
                    if not scores:
                        continue

                    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                    top_d, top_s = sorted_scores[0]
                    second_s = sorted_scores[1][1] if len(sorted_scores) > 1 else 0
                    gap = top_s - second_s

                    if gap > best_gap:
                        best_cell = (r, c)
                        best_digit = top_d
                        best_gap = gap

            if best_cell is None:
                empty = count_empty(grid)
                if empty == 0:
                    self.log(f"\n✓ All cells filled")
                else:
                    self.log(f"\n⚠ Stuck with {empty} cells")
                break

            r, c = best_cell
            self.log(f"  [Inverse] ({r},{c}) → {best_digit} (gap={best_gap:.4f})")
            grid[r][c] = best_digit

        return count_empty(grid) == 0, grid


def analyze_inverse_encoding():
    """Analyze how inverse encoding behaves."""
    print("\n" + "=" * 60)
    print("ANALYSIS: Inverse Encoding")
    print("=" * 60)

    solver = InverseSolver(size=9, verbose=False)

    # Create a partial grid
    grid = [[None for _ in range(9)] for _ in range(9)]
    grid[0][0] = 5
    grid[0][1] = 3
    grid[0][4] = 7

    print("\nPartial grid row 0: [5, 3, _, _, 7, _, _, _, _]")

    # Encode digit locations
    digit_locs = solver.encode_digit_locations(grid)

    print("\nDigit location encodings:")
    for d in range(1, 10):
        norm = np.linalg.norm(digit_locs[d])
        print(f"  Digit {d}: norm = {norm:.4f} {'(has position)' if norm > 0 else ''}")

    # Query row constraint
    print("\nRow 0 constraint for digit 1 (where can 1 go in row 0?):")
    row_constraint = solver.encode_row_constraint(0, 1, grid)

    for c in range(9):
        if grid[0][c] is None:
            sim = similarity(row_constraint, solver.position_vectors[(0, c)])
            print(f"  Position (0,{c}): sim = {sim:.4f}")


def test_4x4():
    print("\n" + "=" * 60)
    print("TEST: 4x4 with Inverse Encoding")
    print("=" * 60)

    print("\nInput:")
    print_grid_4x4(PUZZLE_4x4_EASY)

    solver = InverseSolver(size=4, verbose=True)
    solved, grid = solver.solve(PUZZLE_4x4_EASY)

    print("\nResult:")
    print_grid_4x4(grid)

    valid, msg = validate_4x4(grid)
    print(f"\nValid: {valid}")
    return valid


def test_9x9_hard():
    print("\n" + "=" * 60)
    print("TEST: 9x9 HARD with Inverse Encoding")
    print("=" * 60)

    print("\nInput:")
    print_grid_9x9(PUZZLE_9x9_HARD)

    solver = InverseSolver(size=9, verbose=True)
    with Timer() as timer:
        solved, grid = solver.solve(PUZZLE_9x9_HARD)

    print("\nResult:")
    print_grid_9x9(grid)

    valid, _ = validate_9x9(grid)
    cells = 58 - count_empty(grid)
    print(f"\nCells: {cells}/58, Valid: {valid}")
    print(f"Time: {timer.elapsed:.2f}s")
    return valid


def main():
    print("=" * 60)
    print("APPROACH 8: INVERSE ENCODING")
    print("=" * 60)
    print("\nHypothesis: Encoding digit→position instead of position→digit")
    print("might reveal different patterns.")

    analyze_inverse_encoding()
    test_4x4()
    test_9x9_hard()


if __name__ == "__main__":
    main()
