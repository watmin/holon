#!/usr/bin/env python3
"""Compare our approach to a standard backtracking solver."""

import sys
sys.path.insert(0, "/home/watmin/work/holon")

from typing import List, Optional, Set
import time

from common import (
    PUZZLE_9x9_HARD,
    print_grid_9x9,
    validate_9x9,
    get_available_digits_9x9,
    count_empty,
)


class StandardBacktracker:
    """Plain MRV backtracking - no geometric simulation."""

    def __init__(self):
        self.backtracks = 0

    def get_available(self, grid, row, col) -> Set[int]:
        return get_available_digits_9x9(grid, row, col)

    def solve(self, grid: List[List[Optional[int]]]) -> bool:
        # Find MRV cell
        best_cell = None
        min_opts = 10

        for r in range(9):
            for c in range(9):
                if grid[r][c] is None:
                    opts = self.get_available(grid, r, c)
                    if len(opts) == 0:
                        return False
                    if len(opts) < min_opts:
                        min_opts = len(opts)
                        best_cell = (r, c, list(opts))

        if best_cell is None:
            return True  # Solved

        r, c, opts = best_cell

        for d in opts:
            grid[r][c] = d
            if self.solve(grid):
                return True
            self.backtracks += 1

        grid[r][c] = None
        return False


def test_standard():
    print("=" * 50)
    print("STANDARD BACKTRACKER (MRV only, no simulation)")
    print("=" * 50)

    grid = [[cell for cell in row] for row in PUZZLE_9x9_HARD]
    solver = StandardBacktracker()

    start = time.time()
    solved = solver.solve(grid)
    elapsed = time.time() - start

    print(f"Solved: {solved}")
    print(f"Backtracks: {solver.backtracks}")
    print(f"Time: {elapsed:.4f}s")

    if solved:
        valid, _ = validate_9x9(grid)
        print(f"Valid: {valid}")
        print_grid_9x9(grid)


if __name__ == "__main__":
    test_standard()
