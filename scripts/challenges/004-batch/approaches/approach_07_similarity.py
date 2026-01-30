#!/usr/bin/env python3
"""
Approach 7: Data Similarity Exploitation

HYPOTHESIS:
Similar partial configurations should lead to similar solutions.
If we encode a partial row and compare to known complete rows,
the most similar complete row might indicate the correct digit.

KEY IDEA:
- Encode ALL valid complete rows (9! = 362880 is too many, but 9 is not)
- Actually, encode the "canonical" valid row [1,2,3,4,5,6,7,8,9]
- Compare partial row to rotations/permutations of canonical

ALTERNATIVE APPROACH:
- Encode the DIFFERENCE between what we have and what we need
- A complete row has all 9 digits
- A partial row is missing some digits
- The "completion" should be the missing digits in the right positions

This approach focuses on WHAT'S MISSING rather than what's present.
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
    remove_component,
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


class SimilaritySolver:
    """
    Sudoku solver using data similarity patterns.

    Key insight: Encode what's MISSING and find best way to fill it.
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

        # Pre-compute "complete row" encoding
        # This represents what a valid complete row looks like
        self.complete_row_template = self._encode_complete_row()

        self.notes: List[str] = []
        self.iterations = 0

    def log(self, msg: str):
        if self.verbose:
            print(msg)
        self.notes.append(msg)

    def _encode_complete_row(self) -> np.ndarray:
        """
        Encode the concept of a "complete row" - all positions filled with unique digits.

        This is a template that any valid row should be similar to (after normalization).
        """
        # Bundle all digit vectors (represents "has all digits")
        return bundle([self.digit_vectors[d] for d in self.digits])

    def encode_row_state(self, grid: List[List[Optional[int]]], row: int) -> np.ndarray:
        """Encode the current state of a row (what digits are present)."""
        present = []
        for c in range(self.size):
            if grid[row][c] is not None:
                present.append(self.digit_vectors[grid[row][c]])

        if present:
            return bundle(present)
        return np.zeros_like(self.complete_row_template)

    def encode_missing_digits(self, grid: List[List[Optional[int]]], row: int) -> np.ndarray:
        """Encode what's missing from a row."""
        present = {grid[row][c] for c in range(self.size) if grid[row][c] is not None}
        missing = set(self.digits) - present

        if missing:
            return bundle([self.digit_vectors[d] for d in missing])
        return np.zeros_like(self.complete_row_template)

    def get_available(self, grid, row, col) -> Set[int]:
        if self.size == 4:
            return get_available_digits_4x4(grid, row, col)
        return get_available_digits_9x9(grid, row, col)

    def score_by_completion_similarity(self, grid: List[List[Optional[int]]],
                                        row: int, col: int) -> Dict[int, float]:
        """
        Score digits by how much they help complete the row.

        Idea: Adding a digit should make the row more "complete".
        """
        available = self.get_available(grid, row, col)
        if not available:
            return {}

        current_state = self.encode_row_state(grid, row)
        current_sim = similarity(current_state, self.complete_row_template)

        scores = {}
        for d in available:
            # What if we add this digit?
            test_state = bundle([current_state, self.digit_vectors[d]])
            test_sim = similarity(test_state, self.complete_row_template)

            # Score is the improvement in similarity
            scores[d] = test_sim - current_sim

        return scores

    def score_by_missing_match(self, grid: List[List[Optional[int]]],
                                row: int, col: int) -> Dict[int, float]:
        """
        Score digits by how well they match what's missing.

        Idea: The digit that's most strongly represented in the "missing" encoding
        is the one we should place.
        """
        available = self.get_available(grid, row, col)
        if not available:
            return {}

        missing_vec = self.encode_missing_digits(grid, row)

        scores = {}
        for d in available:
            # How strongly is this digit represented in "missing"?
            scores[d] = similarity(self.digit_vectors[d], missing_vec)

        return scores

    def solve_completion(self, puzzle: List[List[Optional[int]]]) -> Tuple[bool, List[List[int]]]:
        """Solve by maximizing completion similarity."""
        self.log(f"\n{'='*60}")
        self.log("APPROACH 7A: COMPLETION SIMILARITY")
        self.log(f"{'='*60}")
        self.log(f"Empty cells: {count_empty(puzzle)}")

        grid = [[cell for cell in row] for row in puzzle]
        max_iterations = self.size * self.size

        while self.iterations < max_iterations:
            self.iterations += 1

            best_cell = None
            best_digit = None
            best_gap = -1

            for r in range(self.size):
                for c in range(self.size):
                    if grid[r][c] is not None:
                        continue

                    scores = self.score_by_completion_similarity(grid, r, c)
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
            self.log(f"  [Complete] ({r},{c}) → {best_digit} (gap={best_gap:.4f})")
            grid[r][c] = best_digit

        return count_empty(grid) == 0, grid

    def solve_missing(self, puzzle: List[List[Optional[int]]]) -> Tuple[bool, List[List[int]]]:
        """Solve by matching what's missing."""
        self.log(f"\n{'='*60}")
        self.log("APPROACH 7B: MISSING DIGIT MATCH")
        self.log(f"{'='*60}")
        self.log(f"Empty cells: {count_empty(puzzle)}")

        grid = [[cell for cell in row] for row in puzzle]
        max_iterations = self.size * self.size
        self.iterations = 0

        while self.iterations < max_iterations:
            self.iterations += 1

            best_cell = None
            best_digit = None
            best_gap = -1

            for r in range(self.size):
                for c in range(self.size):
                    if grid[r][c] is not None:
                        continue

                    scores = self.score_by_missing_match(grid, r, c)
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
            self.log(f"  [Missing] ({r},{c}) → {best_digit} (gap={best_gap:.4f})")
            grid[r][c] = best_digit

        return count_empty(grid) == 0, grid


def analyze_similarity_patterns():
    """Analyze how similarity patterns work."""
    print("\n" + "=" * 60)
    print("ANALYSIS: Similarity Patterns")
    print("=" * 60)

    solver = SimilaritySolver(size=9, verbose=False)

    print("\n1. Complete Row Template:")
    print("-" * 40)
    template = solver.complete_row_template
    for d in range(1, 10):
        sim = similarity(solver.digit_vectors[d], template)
        print(f"  Digit {d}: sim to complete = {sim:.4f}")

    print("\n2. Partial Row Similarity:")
    print("-" * 40)

    # Simulate a partial row [1, 2, 3, None, None, None, None, None, None]
    partial_state = bundle([solver.digit_vectors[d] for d in [1, 2, 3]])
    partial_sim = similarity(partial_state, template)
    print(f"  Row with [1,2,3]: sim = {partial_sim:.4f}")

    # Add digit 4
    test_state = bundle([partial_state, solver.digit_vectors[4]])
    test_sim = similarity(test_state, template)
    print(f"  After adding 4: sim = {test_sim:.4f} (Δ = {test_sim - partial_sim:.4f})")

    # Add duplicate (bad)
    test_state_bad = bundle([partial_state, solver.digit_vectors[1]])
    test_sim_bad = similarity(test_state_bad, template)
    print(f"  After adding 1 (dup): sim = {test_sim_bad:.4f} (Δ = {test_sim_bad - partial_sim:.4f})")

    print("\n3. Missing Digit Detection:")
    print("-" * 40)

    # Row with [1,2,3,4,5] - missing [6,7,8,9]
    present = [1, 2, 3, 4, 5]
    missing_vec = bundle([solver.digit_vectors[d] for d in [6, 7, 8, 9]])

    print(f"  Row has: {present}, missing: [6,7,8,9]")
    print("  Similarity to missing encoding:")
    for d in range(1, 10):
        sim = similarity(solver.digit_vectors[d], missing_vec)
        in_missing = d in [6, 7, 8, 9]
        marker = "← MISSING" if in_missing else ""
        print(f"    Digit {d}: {sim:.4f} {marker}")


def test_4x4():
    print("\n" + "=" * 60)
    print("TEST: 4x4 with Similarity")
    print("=" * 60)

    print("\nInput:")
    print_grid_4x4(PUZZLE_4x4_EASY)

    # Test completion approach
    solver = SimilaritySolver(size=4, verbose=True)
    with Timer() as timer:
        solved, grid = solver.solve_completion(PUZZLE_4x4_EASY)

    print("\nResult:")
    print_grid_4x4(grid)

    valid, msg = validate_4x4(grid)
    print(f"\nValid: {valid}")
    return valid


def test_9x9_hard():
    print("\n" + "=" * 60)
    print("TEST: 9x9 HARD with Similarity")
    print("=" * 60)

    # Test both approaches
    for name, method in [("Completion", "solve_completion"), ("Missing", "solve_missing")]:
        print(f"\n--- {name} Approach ---")
        solver = SimilaritySolver(size=9, verbose=False)
        with Timer() as timer:
            solved, grid = getattr(solver, method)(PUZZLE_9x9_HARD)
        cells = 58 - count_empty(grid)
        valid, _ = validate_9x9(grid)
        print(f"  Cells: {cells}/58, Valid: {valid}, Time: {timer.elapsed:.2f}s")

    return valid


def main():
    print("=" * 60)
    print("APPROACH 7: DATA SIMILARITY EXPLOITATION")
    print("=" * 60)

    analyze_similarity_patterns()
    test_4x4()
    test_9x9_hard()


if __name__ == "__main__":
    main()
