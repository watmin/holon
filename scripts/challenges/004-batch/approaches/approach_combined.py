#!/usr/bin/env python3
"""
Combined Approach: Best Insights from All Tests

KEY INSIGHTS FROM TESTING:
1. Approach 5 (Row Completion): 54/58 - Best greedy approach
2. Approach 6 (Orientation): Correctly identifies constraint intersection
3. Approach 7 (Similarity): Duplicate detection via negative similarity change

COMBINED STRATEGY:
1. Score using row completion (best signal)
2. Verify with duplicate detection (reject bad placements)
3. Use constraint intersection as tie-breaker
4. Be conservative when scores are close

The goal is to AVOID wrong decisions rather than make more placements.
"""

import sys
sys.path.insert(0, "/home/watmin/work/holon")

from typing import List, Optional, Set, Tuple, Dict
import numpy as np

from common import (
    create_client,
    VectorCache,
    bind,
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


class CombinedSolver:
    """
    Combined solver using best insights from all approaches.
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

        # Complete row template (for similarity checking)
        self.complete_template = bundle([self.digit_vectors[d] for d in self.digits])

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

    def encode_row_state(self, grid: List[List[Optional[int]]], row: int) -> np.ndarray:
        """Encode current digits in row."""
        present = []
        for c in range(self.size):
            if grid[row][c] is not None:
                present.append(self.digit_vectors[grid[row][c]])
        if present:
            return bundle(present)
        return np.zeros_like(self.complete_template)

    def encode_col_state(self, grid: List[List[Optional[int]]], col: int) -> np.ndarray:
        """Encode current digits in column."""
        present = []
        for r in range(self.size):
            if grid[r][col] is not None:
                present.append(self.digit_vectors[grid[r][col]])
        if present:
            return bundle(present)
        return np.zeros_like(self.complete_template)

    def encode_block_state(self, grid: List[List[Optional[int]]], row: int, col: int) -> np.ndarray:
        """Encode current digits in block."""
        br = (row // self.block_size) * self.block_size
        bc = (col // self.block_size) * self.block_size
        present = []
        for r in range(br, br + self.block_size):
            for c in range(bc, bc + self.block_size):
                if grid[r][c] is not None:
                    present.append(self.digit_vectors[grid[r][c]])
        if present:
            return bundle(present)
        return np.zeros_like(self.complete_template)

    def check_duplicate_penalty(self, grid: List[List[Optional[int]]],
                                 row: int, col: int, digit: int) -> Tuple[bool, float]:
        """
        Check if placing this digit would cause a "duplicate penalty".

        Returns (has_penalty, penalty_score)
        A penalty occurs when adding the digit DECREASES similarity to complete template.
        """
        digit_vec = self.digit_vectors[digit]

        penalties = []

        # Check row
        row_state = self.encode_row_state(grid, row)
        row_sim_before = similarity(row_state, self.complete_template)
        row_sim_after = similarity(bundle([row_state, digit_vec]), self.complete_template)
        row_delta = row_sim_after - row_sim_before
        if row_delta < 0:
            penalties.append(("row", row_delta))

        # Check column
        col_state = self.encode_col_state(grid, col)
        col_sim_before = similarity(col_state, self.complete_template)
        col_sim_after = similarity(bundle([col_state, digit_vec]), self.complete_template)
        col_delta = col_sim_after - col_sim_before
        if col_delta < 0:
            penalties.append(("col", col_delta))

        # Check block
        block_state = self.encode_block_state(grid, row, col)
        block_sim_before = similarity(block_state, self.complete_template)
        block_sim_after = similarity(bundle([block_state, digit_vec]), self.complete_template)
        block_delta = block_sim_after - block_sim_before
        if block_delta < 0:
            penalties.append(("block", block_delta))

        has_penalty = len(penalties) > 0
        total_penalty = sum(p[1] for p in penalties) if penalties else 0

        return has_penalty, total_penalty

    def score_completion(self, grid: List[List[Optional[int]]],
                         row: int, col: int, digit: int) -> float:
        """
        Score a digit based on how much it improves completion similarity.
        Combined across row, col, and block.
        """
        digit_vec = self.digit_vectors[digit]
        total_delta = 0

        # Row contribution
        row_state = self.encode_row_state(grid, row)
        row_sim_before = similarity(row_state, self.complete_template)
        row_sim_after = similarity(bundle([row_state, digit_vec]), self.complete_template)
        total_delta += row_sim_after - row_sim_before

        # Column contribution
        col_state = self.encode_col_state(grid, col)
        col_sim_before = similarity(col_state, self.complete_template)
        col_sim_after = similarity(bundle([col_state, digit_vec]), self.complete_template)
        total_delta += col_sim_after - col_sim_before

        # Block contribution
        block_state = self.encode_block_state(grid, row, col)
        block_sim_before = similarity(block_state, self.complete_template)
        block_sim_after = similarity(bundle([block_state, digit_vec]), self.complete_template)
        total_delta += block_sim_after - block_sim_before

        return total_delta

    def solve(self, puzzle: List[List[Optional[int]]],
              min_gap: float = 0.01,
              require_no_penalty: bool = True) -> Tuple[bool, List[List[int]]]:
        """
        Solve using combined approach.

        Args:
            min_gap: Minimum score gap required to place (higher = more conservative)
            require_no_penalty: If True, reject placements that would cause penalties
        """
        self.log(f"\n{'='*60}")
        self.log(f"COMBINED APPROACH (gap={min_gap}, no_penalty={require_no_penalty})")
        self.log(f"{'='*60}")
        self.log(f"Empty cells: {count_empty(puzzle)}")

        grid = [[cell for cell in row] for row in puzzle]
        max_iterations = self.size * self.size * 2
        self.iterations = 0

        while self.iterations < max_iterations:
            self.iterations += 1

            # Find best placement
            best_cell = None
            best_digit = None
            best_score = -999
            best_gap = -1

            for r in range(self.size):
                for c in range(self.size):
                    if grid[r][c] is not None:
                        continue

                    available = self.get_available(grid, r, c)
                    if not available:
                        continue

                    # Score each available digit
                    scores = {}
                    for d in available:
                        # Check for penalty
                        has_penalty, penalty = self.check_duplicate_penalty(grid, r, c, d)

                        if require_no_penalty and has_penalty:
                            continue  # Skip this digit

                        # Score based on completion
                        scores[d] = self.score_completion(grid, r, c, d)

                    if not scores:
                        continue

                    # Find best and gap
                    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                    top_d, top_s = sorted_scores[0]
                    second_s = sorted_scores[1][1] if len(sorted_scores) > 1 else -999
                    gap = top_s - second_s

                    # Must meet minimum gap
                    if gap >= min_gap and gap > best_gap:
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
            self.log(f"  [Combined] ({r},{c}) → {best_digit} (score={best_score:.4f}, gap={best_gap:.4f})")
            grid[r][c] = best_digit

        return count_empty(grid) == 0, grid


def test_hard_puzzle():
    """Test on hard puzzle with different settings."""
    print("=" * 60)
    print("COMBINED APPROACH - TESTING DIFFERENT SETTINGS")
    print("=" * 60)

    print("\nInput:")
    print_grid_9x9(PUZZLE_9x9_HARD)
    print(f"Empty cells: {count_empty(PUZZLE_9x9_HARD)}")

    # Test different configurations
    configs = [
        {"min_gap": 0.0, "require_no_penalty": False, "name": "Greedy (baseline)"},
        {"min_gap": 0.0, "require_no_penalty": True, "name": "No penalty"},
        {"min_gap": 0.01, "require_no_penalty": True, "name": "Conservative"},
        {"min_gap": 0.02, "require_no_penalty": True, "name": "Very conservative"},
    ]

    results = []
    for config in configs:
        solver = CombinedSolver(size=9, verbose=False)
        with Timer() as timer:
            solved, grid = solver.solve(
                PUZZLE_9x9_HARD,
                min_gap=config["min_gap"],
                require_no_penalty=config["require_no_penalty"]
            )
        cells = 58 - count_empty(grid)
        valid, _ = validate_9x9(grid)
        results.append((config["name"], cells, valid, timer.elapsed))
        print(f"\n{config['name']}: {cells}/58 cells, Valid: {valid}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("\n| Configuration | Cells | Valid | Time |")
    print("|---------------|-------|-------|------|")
    for name, cells, valid, time in results:
        print(f"| {name:20} | {cells}/58 | {'✓' if valid else '✗'} | {time:.2f}s |")

    # Show best grid
    best_config = max(results, key=lambda x: x[1])
    if best_config[2]:  # If valid
        print(f"\n✓ SOLVED with: {best_config[0]}")
    else:
        # Show best grid
        print(f"\nBest result: {best_config[0]} with {best_config[1]}/58 cells")

        # Regenerate best grid
        solver = CombinedSolver(size=9, verbose=False)
        _, grid = solver.solve(
            PUZZLE_9x9_HARD,
            min_gap=0.0,
            require_no_penalty=(best_config[0] != "Greedy (baseline)")
        )
        print("\nBest grid:")
        print_grid_9x9(grid)


def main():
    print("=" * 60)
    print("COMBINED APPROACH")
    print("=" * 60)
    print("\nUsing insights from all approaches:")
    print("- Row/Col/Block completion scoring (Approach 5/6)")
    print("- Duplicate penalty detection (Approach 7)")
    print("- Conservative gap thresholds")

    # First test 4x4
    print("\n" + "=" * 60)
    print("4x4 TEST")
    print("=" * 60)

    print("\nInput:")
    print_grid_4x4(PUZZLE_4x4_EASY)

    solver = CombinedSolver(size=4, verbose=True)
    solved, grid = solver.solve(PUZZLE_4x4_EASY)

    print("\nResult:")
    print_grid_4x4(grid)

    valid, msg = validate_4x4(grid)
    print(f"Valid: {valid}")

    # Test hard puzzle
    test_hard_puzzle()


if __name__ == "__main__":
    main()
