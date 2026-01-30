#!/usr/bin/env python3
"""
Approach 5: Structural Entanglement

HYPOTHESIS:
Holon's recursive encoding creates ENTANGLED representations.
Binding preserves the relationship such that unbinding can recover it.
We can use this to query unknown values from encoded structures.

KEY INSIGHT:
If we encode: cell = bind(position, digit)
Then: unbind(cell, position) ≈ digit

Can we exploit this for solving?
1. Encode all known placements
2. Encode constraint "templates"
3. Query unknowns by unbinding

TEST QUESTIONS:
1. Does unbinding from bundled structure preserve individual bindings?
2. Can constraints inform what digit should be at a position?
3. Is there a "completion" operation that fills unknowns?
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
    PUZZLE_4x4_SOLUTION,
    PUZZLE_9x9_EASY,
    PUZZLE_9x9_EASY_SOLUTION,
    PUZZLE_9x9_HARD,
)


class EntanglementSolver:
    """
    Sudoku solver exploiting entanglement properties.

    Core idea: Encode structures such that unbinding reveals solutions.
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
        self.iterations = 0
        self.entanglement_queries = 0

    def log(self, msg: str):
        if self.verbose:
            print(msg)
        self.notes.append(msg)

    def get_available(self, grid, row, col) -> Set[int]:
        if self.size == 4:
            return get_available_digits_4x4(grid, row, col)
        return get_available_digits_9x9(grid, row, col)

    def encode_cell(self, row: int, col: int, digit: int) -> np.ndarray:
        """Encode a single cell placement as position ⊙ digit."""
        return bind(self.position_vectors[(row, col)], self.digit_vectors[digit])

    def encode_grid(self, grid: List[List[Optional[int]]]) -> np.ndarray:
        """Encode all known cells as bundled cell bindings."""
        cell_bindings = []
        for r in range(self.size):
            for c in range(self.size):
                if grid[r][c] is not None:
                    cell_bindings.append(self.encode_cell(r, c, grid[r][c]))

        if cell_bindings:
            return bundle(cell_bindings)
        return np.zeros_like(self.digit_vectors[1])

    def encode_ideal_row(self, row: int) -> np.ndarray:
        """
        Encode the ideal row constraint.

        An ideal row has ALL 9 digits at positions (row, 0) through (row, 8).
        We encode this as a bundle of all possible valid row completions...
        but that's 9! = 362880 combinations.

        Simpler: encode the STRUCTURE of a valid row (superposition of all digits).
        """
        # Bundle all position-digit bindings for this row
        # Each column should have some digit
        row_bindings = []
        for c in range(self.size):
            # Superposition: this position could have any digit
            pos = self.position_vectors[(row, c)]
            digit_super = bundle([self.digit_vectors[d] for d in self.digits])
            row_bindings.append(bind(pos, digit_super))

        return bundle(row_bindings)

    def query_digit_at_position(self, grid_vec: np.ndarray,
                                 row: int, col: int) -> Tuple[int, float]:
        """
        Query what digit is at a position by unbinding.

        Unbind the position from the grid vector.
        The result should be close to the digit vector if it's known.
        """
        pos_vec = self.position_vectors[(row, col)]
        recovered = unbind(grid_vec, pos_vec)

        # Find best matching digit
        best_digit = None
        best_sim = -1
        for d in self.digits:
            sim = similarity(recovered, self.digit_vectors[d])
            if sim > best_sim:
                best_sim = sim
                best_digit = d

        self.entanglement_queries += 1
        return best_digit, best_sim

    def query_row_completion(self, grid: List[List[Optional[int]]],
                             row: int, col: int) -> Dict[int, float]:
        """
        Use row structure to inform what digit should go at (row, col).

        Idea: Encode the partial row, compare to ideal row,
        find what digit would make it more similar to ideal.
        """
        ideal_row = self.encode_ideal_row(row)

        # Current row encoding (known cells)
        known_bindings = []
        for c in range(self.size):
            if grid[row][c] is not None:
                known_bindings.append(self.encode_cell(row, c, grid[row][c]))

        if known_bindings:
            current_row = bundle(known_bindings)
        else:
            current_row = np.zeros_like(ideal_row)

        # Score each possible digit for this cell
        scores = {}
        available = self.get_available(grid, row, col)

        for d in available:
            # What if we add this digit?
            test_binding = self.encode_cell(row, col, d)
            test_row = bundle([current_row, test_binding])

            # How similar to ideal?
            sim = similarity(test_row, ideal_row)
            scores[d] = sim

        return scores

    def solve_via_entanglement_query(self, puzzle: List[List[Optional[int]]]) -> Tuple[bool, List[List[int]]]:
        """
        Attempt 1: Solve by querying encoded grid.

        Encode all known cells, then query unknown positions.
        """
        self.log(f"\n{'='*60}")
        self.log("APPROACH 5A: ENTANGLEMENT QUERY")
        self.log(f"{'='*60}")

        grid = [[cell for cell in row] for row in puzzle]
        self.log(f"Empty cells: {count_empty(grid)}")

        max_iterations = self.size * self.size

        while self.iterations < max_iterations:
            self.iterations += 1

            # Encode current grid state
            grid_vec = self.encode_grid(grid)

            # Find an empty cell
            found = False
            for r in range(self.size):
                for c in range(self.size):
                    if grid[r][c] is not None:
                        continue

                    # Query what digit is at this position
                    digit, conf = self.query_digit_at_position(grid_vec, r, c)
                    available = self.get_available(grid, r, c)

                    if digit in available:
                        self.log(f"  [Query] ({r},{c}) → {digit} (conf={conf:.4f})")
                        grid[r][c] = digit
                        found = True
                        break
                    else:
                        # Query returned invalid digit
                        self.log(f"  [Query] ({r},{c}) → {digit} INVALID (not in {available})")

                if found:
                    break

            if not found:
                # Check if solved or stuck
                empty = count_empty(grid)
                if empty == 0:
                    self.log(f"\n✓ All cells filled")
                    break
                else:
                    self.log(f"\n⚠ Stuck with {empty} cells empty")
                    break

        return count_empty(grid) == 0, grid

    def solve_via_row_completion(self, puzzle: List[List[Optional[int]]]) -> Tuple[bool, List[List[int]]]:
        """
        Attempt 2: Solve by comparing to ideal row structure.

        For each empty cell, find digit that makes row most similar to ideal.
        """
        self.log(f"\n{'='*60}")
        self.log("APPROACH 5B: ROW COMPLETION")
        self.log(f"{'='*60}")

        grid = [[cell for cell in row] for row in puzzle]
        self.log(f"Empty cells: {count_empty(grid)}")

        max_iterations = self.size * self.size

        while self.iterations < max_iterations:
            self.iterations += 1

            # Find best placement across all empty cells
            best_cell = None
            best_digit = None
            best_score = -1
            best_gap = 0

            for r in range(self.size):
                for c in range(self.size):
                    if grid[r][c] is not None:
                        continue

                    scores = self.query_row_completion(grid, r, c)

                    if not scores:
                        continue

                    # Find best and second-best
                    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                    top_d, top_s = sorted_scores[0]
                    second_s = sorted_scores[1][1] if len(sorted_scores) > 1 else 0
                    gap = top_s - second_s

                    # Prefer cells with clear winner (larger gap)
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
                    self.log(f"\n⚠ No valid placements found, {empty} cells empty")
                break

            r, c = best_cell
            self.log(f"  [RowComplete] ({r},{c}) → {best_digit} (sim={best_score:.4f}, gap={best_gap:.4f})")
            grid[r][c] = best_digit

        return count_empty(grid) == 0, grid


def test_unbind_from_bundle():
    """Test: Can we recover bindings from a bundled structure?"""
    print("\n" + "=" * 60)
    print("TEST: Unbinding from Bundled Structure")
    print("=" * 60)

    solver = EntanglementSolver(size=9, verbose=False)

    # Create a partial grid with known values
    cells = [(0, 0, 5), (0, 1, 3), (0, 4, 7), (1, 0, 6)]

    # Encode as bundled bindings
    bindings = [solver.encode_cell(r, c, d) for r, c, d in cells]
    grid_vec = bundle(bindings)

    print(f"\nEncoded {len(cells)} cell bindings into grid vector")

    # Try to recover each digit
    print("\nRecovery test:")
    for r, c, expected in cells:
        digit, conf = solver.query_digit_at_position(grid_vec, r, c)
        match = "✓" if digit == expected else "✗"
        print(f"  ({r},{c}): expected={expected}, recovered={digit} (conf={conf:.4f}) {match}")

    # Also test an unknown position
    print("\nUnknown position test:")
    for pos in [(0, 2), (1, 1), (2, 0)]:
        digit, conf = solver.query_digit_at_position(grid_vec, pos[0], pos[1])
        print(f"  {pos}: recovered={digit} (conf={conf:.4f}) - THIS IS NOISE")


def test_4x4():
    """Test entanglement approach on 4x4."""
    print("\n" + "=" * 60)
    print("TEST: 4x4 with Entanglement Query")
    print("=" * 60)

    print("\nInput puzzle:")
    print_grid_4x4(PUZZLE_4x4_EASY)

    solver = EntanglementSolver(size=4, verbose=True)

    with Timer() as timer:
        solved, grid = solver.solve_via_entanglement_query(PUZZLE_4x4_EASY)

    print("\nResult:")
    print_grid_4x4(grid)

    valid, msg = validate_4x4(grid)
    print(f"\nSolved: {solved}, Valid: {valid} - {msg}")
    print(f"Entanglement queries: {solver.entanglement_queries}")

    return solved and valid


def test_9x9_easy():
    """Test entanglement on 9x9 easy."""
    print("\n" + "=" * 60)
    print("TEST: 9x9 Easy with Entanglement Query")
    print("=" * 60)

    print("\nInput puzzle:")
    print_grid_9x9(PUZZLE_9x9_EASY)

    solver = EntanglementSolver(size=9, verbose=True)

    with Timer() as timer:
        solved, grid = solver.solve_via_entanglement_query(PUZZLE_9x9_EASY)

    print("\nResult:")
    print_grid_9x9(grid)

    valid, msg = validate_9x9(grid)
    print(f"\nSolved: {solved}, Valid: {valid} - {msg}")

    return solved and valid


def test_row_completion():
    """Test row completion approach."""
    print("\n" + "=" * 60)
    print("TEST: Row Completion Approach")
    print("=" * 60)

    print("\nInput puzzle (4x4):")
    print_grid_4x4(PUZZLE_4x4_EASY)

    solver = EntanglementSolver(size=4, verbose=True)

    with Timer() as timer:
        solved, grid = solver.solve_via_row_completion(PUZZLE_4x4_EASY)

    print("\nResult:")
    print_grid_4x4(grid)

    valid, msg = validate_4x4(grid)
    print(f"\nSolved: {solved}, Valid: {valid} - {msg}")

    return solved and valid


def test_row_completion_9x9():
    """Test row completion on 9x9 puzzles."""
    print("\n" + "=" * 60)
    print("TEST: 9x9 Easy with Row Completion")
    print("=" * 60)

    print("\nInput puzzle:")
    print_grid_9x9(PUZZLE_9x9_EASY)

    solver = EntanglementSolver(size=9, verbose=True)

    with Timer() as timer:
        solved, grid = solver.solve_via_row_completion(PUZZLE_9x9_EASY)

    print("\nResult:")
    print_grid_9x9(grid)

    valid, msg = validate_9x9(grid)
    print(f"\nSolved: {solved}, Valid: {valid} - {msg}")
    print(f"Time: {timer.elapsed:.4f}s")

    return solved and valid


def test_row_completion_9x9_hard():
    """Test row completion on 9x9 hard puzzle."""
    print("\n" + "=" * 60)
    print("TEST: 9x9 HARD with Row Completion")
    print("=" * 60)

    print("\nInput puzzle:")
    print_grid_9x9(PUZZLE_9x9_HARD)
    print(f"Empty cells: {count_empty(PUZZLE_9x9_HARD)}")

    solver = EntanglementSolver(size=9, verbose=True)

    with Timer() as timer:
        solved, grid = solver.solve_via_row_completion(PUZZLE_9x9_HARD)

    print("\nResult:")
    print_grid_9x9(grid)

    valid, msg = validate_9x9(grid)
    print(f"\nSolved: {solved}, Valid: {valid} - {msg}")
    print(f"Time: {timer.elapsed:.4f}s")

    return solved and valid


def main():
    print("=" * 60)
    print("APPROACH 5: STRUCTURAL ENTANGLEMENT")
    print("=" * 60)
    print("\nHypothesis: Binding creates entanglement that can be queried.")

    # First test the fundamental operation
    test_unbind_from_bundle()

    # Test approaches
    print("\n" + "=" * 60)
    print("SOLVING TESTS")
    print("=" * 60)

    result_4x4_query = test_4x4()
    result_4x4_row = test_row_completion()
    result_9x9_easy = test_row_completion_9x9()
    result_9x9_hard = test_row_completion_9x9_hard()

    print("\n" + "=" * 60)
    print("APPROACH 5 SUMMARY")
    print("=" * 60)

    print(f"\n4x4 Entanglement Query: {'✓' if result_4x4_query else '✗'}")
    print(f"4x4 Row Completion: {'✓' if result_4x4_row else '✗'}")
    print(f"9x9 Easy Row Completion: {'✓' if result_9x9_easy else '✗'}")
    print(f"9x9 Hard Row Completion: {'✓' if result_9x9_hard else '✗'}")

    print("\nKey observations:")
    print("- Unbinding from bundle recovers KNOWN bindings")
    print("- Querying UNKNOWN positions returns noise")
    print("- Row completion uses similarity to ideal row structure")

    if result_9x9_hard:
        print("\n✓✓ BREAKTHROUGH: Row completion solved hard puzzle!")
    elif result_9x9_easy and result_4x4_row:
        print("\n◐ PARTIAL: Works on easy, needs enhancement for hard")
    else:
        print("\n✗ Needs refinement")


if __name__ == "__main__":
    main()
