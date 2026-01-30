#!/usr/bin/env python3
"""
Approach 3: Direct Geometric Decoding

HYPOTHESIS:
If we encode a complete valid grid, we can decode it to extract solutions.
The question: can we CONSTRUCT the solution vector without knowing the solution?

KEY IDEA:
- A valid Sudoku grid has a specific geometric signature
- This signature is a point in hyperspace
- Can we find this point through constraint equations?

APPROACH:
1. Encode each constraint (row must have all digits) as a vector
2. The solution is the "intersection" of all constraint vectors
3. Try to find the solution by combining constraint information

This is exploring whether the solution can be "computed" geometrically
rather than searched for.
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


class DecodingSolver:
    """
    Attempt to directly decode solutions from constraint geometry.
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

    def encode_solution(self, grid: List[List[int]]) -> np.ndarray:
        """Encode a complete solution grid."""
        bindings = []
        for r in range(self.size):
            for c in range(self.size):
                if grid[r][c] is not None:
                    cell_binding = bind(
                        self.position_vectors[(r, c)],
                        self.digit_vectors[grid[r][c]]
                    )
                    bindings.append(cell_binding)
        return bundle(bindings)

    def decode_cell(self, solution_vec: np.ndarray, row: int, col: int) -> Tuple[int, float]:
        """Attempt to decode a cell from a solution vector."""
        pos_vec = self.position_vectors[(row, col)]
        recovered = unbind(solution_vec, pos_vec)

        best_d = None
        best_sim = -1
        for d in self.digits:
            sim = similarity(recovered, self.digit_vectors[d])
            if sim > best_sim:
                best_sim = sim
                best_d = d

        return best_d, best_sim

    def test_encoding_decoding(self):
        """Test if we can encode a solution and decode it back."""
        self.log("\n" + "=" * 60)
        self.log("TEST: Encode-Decode Round Trip")
        self.log("=" * 60)

        # Use known solution
        if self.size == 4:
            solution = PUZZLE_4x4_SOLUTION
        else:
            solution = PUZZLE_9x9_EASY_SOLUTION

        # Encode full solution
        solution_vec = self.encode_solution(solution)

        # Try to decode each cell
        correct = 0
        total = self.size * self.size

        for r in range(self.size):
            for c in range(self.size):
                decoded, conf = self.decode_cell(solution_vec, r, c)
                expected = solution[r][c]
                match = decoded == expected
                if match:
                    correct += 1
                if self.verbose and not match:
                    self.log(f"  ({r},{c}): expected {expected}, got {decoded} (conf={conf:.3f})")

        accuracy = correct / total * 100
        self.log(f"\nDecode accuracy: {correct}/{total} = {accuracy:.1f}%")

        return accuracy > 90

    def construct_from_constraints(self, puzzle: List[List[Optional[int]]]) -> np.ndarray:
        """
        Attempt to construct a solution vector from constraint information.

        Idea: Bundle all known cell bindings and see if we can infer unknowns.
        """
        # Encode known cells
        known_bindings = []
        for r in range(self.size):
            for c in range(self.size):
                if puzzle[r][c] is not None:
                    cell_binding = bind(
                        self.position_vectors[(r, c)],
                        self.digit_vectors[puzzle[r][c]]
                    )
                    known_bindings.append(cell_binding)

        if known_bindings:
            return bundle(known_bindings)
        return np.zeros_like(self.digit_vectors[1])

    def solve_via_decoding(self, puzzle: List[List[Optional[int]]]) -> Tuple[bool, List[List[int]]]:
        """
        Attempt to solve by constructing and decoding solution vector.
        """
        self.log(f"\n{'='*60}")
        self.log("APPROACH 3: DIRECT GEOMETRIC DECODING")
        self.log(f"{'='*60}")
        self.log(f"Empty cells: {count_empty(puzzle)}")

        # Construct partial solution vector
        partial_vec = self.construct_from_constraints(puzzle)

        # Try to decode all cells
        grid = [[None for _ in range(self.size)] for _ in range(self.size)]

        for r in range(self.size):
            for c in range(self.size):
                if puzzle[r][c] is not None:
                    grid[r][c] = puzzle[r][c]
                else:
                    digit, conf = self.decode_cell(partial_vec, r, c)
                    grid[r][c] = digit
                    self.log(f"  Decoded ({r},{c}) → {digit} (conf={conf:.3f})")

        return True, grid


def test_4x4():
    print("\n" + "=" * 60)
    print("TEST: 4x4 with Direct Decoding")
    print("=" * 60)

    solver = DecodingSolver(size=4, verbose=True)

    # First test encode-decode
    solver.test_encoding_decoding()

    # Then try solving
    print("\nInput:")
    print_grid_4x4(PUZZLE_4x4_EASY)

    _, grid = solver.solve_via_decoding(PUZZLE_4x4_EASY)

    print("\nResult:")
    print_grid_4x4(grid)

    valid, msg = validate_4x4(grid)
    print(f"\nValid: {valid}")
    return valid


def test_9x9_hard():
    print("\n" + "=" * 60)
    print("TEST: 9x9 HARD with Direct Decoding")
    print("=" * 60)

    solver = DecodingSolver(size=9, verbose=True)

    # First verify encode-decode works
    print("\nVerifying encode-decode on known solution...")
    solver.test_encoding_decoding()

    # Then try on puzzle
    print("\nInput puzzle:")
    print_grid_9x9(PUZZLE_9x9_HARD)

    _, grid = solver.solve_via_decoding(PUZZLE_9x9_HARD)

    print("\nResult:")
    print_grid_9x9(grid)

    valid, _ = validate_9x9(grid)
    cells_correct = sum(1 for r in range(9) for c in range(9)
                       if grid[r][c] is not None)
    print(f"\nValid: {valid}")
    return valid


def main():
    print("=" * 60)
    print("APPROACH 3: DIRECT GEOMETRIC DECODING")
    print("=" * 60)
    print("\nHypothesis: Can we decode solutions from partial encodings?")

    test_4x4()
    test_9x9_hard()


if __name__ == "__main__":
    main()
