#!/usr/bin/env python3
"""
Approach 2: Superposition Collapse

HYPOTHESIS:
Represent unknown cells as superposition of all possible digits.
Apply constraints by "removing" digits from superpositions.
Cells automatically collapse when one digit dominates.

This is constraint propagation implemented in hyperspace.

MECHANISM:
1. Empty cell = bundle(all digit vectors) = superposition
2. When digit is placed, remove it from all peers' superpositions
3. Check each cell: if one digit has much higher similarity, collapse
4. Repeat until solved or stable

KEY OPERATIONS:
- remove_component(superposition, digit_vec) - removes a digit
- similarity(superposition, digit_vec) - checks for dominance
- collapse detection - when one digit clearly dominates
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
    remove_component,
    ApproachResult,
    Timer,
    print_grid_4x4,
    print_grid_9x9,
    validate_4x4,
    validate_9x9,
    count_empty,
    PUZZLE_4x4_EASY,
    PUZZLE_9x9_EASY,
    PUZZLE_9x9_HARD,
)


class SuperpositionSolver:
    """
    Sudoku solver using superposition collapse.

    Each cell is represented as a vector:
    - Known cell: digit vector
    - Unknown cell: superposition of possible digits

    Constraints propagate by removing digits from superpositions.
    """

    def __init__(self, size: int = 9, dimensions: int = 16384, verbose: bool = True):
        self.size = size
        self.block_size = 2 if size == 4 else 3
        self.digits = list(range(1, size + 1))
        self.verbose = verbose

        # Create Holon client and cache
        self.client = create_client(dimensions=dimensions)
        self.cache = VectorCache(self.client)

        # Pre-cache digit vectors
        self.digit_vectors = {d: self.cache.get_digit_vector(d) for d in self.digits}

        # Full superposition (unknown cell)
        self.full_superposition = bundle([self.digit_vectors[d] for d in self.digits])

        # Collapse threshold - how much higher must dominant digit be?
        self.collapse_threshold = 0.15  # Dominant must be 0.15 higher than next

        # Stats
        self.iterations = 0
        self.collapses = 0
        self.propagations = 0
        self.notes: List[str] = []

    def log(self, msg: str):
        if self.verbose:
            print(msg)
        self.notes.append(msg)

    def get_peers(self, row: int, col: int) -> List[Tuple[int, int]]:
        """Get all cells that share a constraint with this cell."""
        peers = set()

        # Row peers
        for c in range(self.size):
            if c != col:
                peers.add((row, c))

        # Column peers
        for r in range(self.size):
            if r != row:
                peers.add((r, col))

        # Block peers
        br = (row // self.block_size) * self.block_size
        bc = (col // self.block_size) * self.block_size
        for r in range(br, br + self.block_size):
            for c in range(bc, bc + self.block_size):
                if (r, c) != (row, col):
                    peers.add((r, c))

        return list(peers)

    def digit_similarities(self, cell_vec: np.ndarray) -> Dict[int, float]:
        """Compute similarity to each digit vector."""
        return {d: similarity(cell_vec, self.digit_vectors[d]) for d in self.digits}

    def is_collapsed(self, cell_vec: np.ndarray, remaining: Set[int]) -> Tuple[bool, Optional[int]]:
        """
        Check if a cell has collapsed to a single digit.

        Args:
            cell_vec: The cell's vector state
            remaining: Set of digits not yet removed from this cell

        Returns (is_collapsed, digit or None)
        """
        # Case 1: Only one digit remains - immediate collapse
        if len(remaining) == 1:
            return True, list(remaining)[0]

        # Case 2: Check if one remaining digit is clearly dominant
        sims = {d: similarity(cell_vec, self.digit_vectors[d]) for d in remaining}

        if len(sims) < 2:
            return False, None

        sorted_sims = sorted(sims.items(), key=lambda x: x[1], reverse=True)
        top_digit, top_sim = sorted_sims[0]
        second_digit, second_sim = sorted_sims[1]

        # Top is clearly dominant
        if top_sim - second_sim > self.collapse_threshold:
            return True, top_digit

        return False, None

    def extract_digit(self, cell_vec: np.ndarray) -> int:
        """Extract the dominant digit from a cell vector."""
        sims = self.digit_similarities(cell_vec)
        return max(sims, key=sims.get)

    def remove_digit_from_cell(self, cell_vec: np.ndarray, digit: int) -> np.ndarray:
        """Remove a digit from a cell's superposition."""
        return remove_component(cell_vec, self.digit_vectors[digit])

    def solve(self, puzzle: List[List[Optional[int]]]) -> Tuple[bool, List[List[int]]]:
        """
        Solve using superposition collapse.
        """
        self.log(f"\n{'='*60}")
        self.log("APPROACH 2: SUPERPOSITION COLLAPSE SOLVER")
        self.log(f"{'='*60}")
        self.log(f"Puzzle size: {self.size}x{self.size}")
        self.log(f"Empty cells: {count_empty(puzzle)}")

        # Initialize cell states
        # Track removed digits per cell (to avoid double-removal)
        cell_states: Dict[Tuple[int, int], np.ndarray] = {}
        collapsed: Dict[Tuple[int, int], Optional[int]] = {}
        removed_digits: Dict[Tuple[int, int], Set[int]] = {}

        for r in range(self.size):
            for c in range(self.size):
                if puzzle[r][c] is not None:
                    # Known cell - use digit vector directly
                    digit = puzzle[r][c]
                    cell_states[(r, c)] = self.digit_vectors[digit].copy()
                    collapsed[(r, c)] = digit
                    removed_digits[(r, c)] = set()  # Not relevant for collapsed
                else:
                    # Unknown cell - start with full superposition
                    cell_states[(r, c)] = self.full_superposition.copy()
                    collapsed[(r, c)] = None
                    removed_digits[(r, c)] = set()

        # Initial propagation from given digits
        self.log("\nInitial propagation from given digits:")
        for r in range(self.size):
            for c in range(self.size):
                if collapsed[(r, c)] is not None:
                    digit = collapsed[(r, c)]
                    self._propagate(cell_states, collapsed, removed_digits, r, c, digit)

        # Main loop: check for collapses, propagate, repeat
        max_iterations = self.size * self.size * 2
        stable = False

        while self.iterations < max_iterations and not stable:
            self.iterations += 1
            stable = True

            # Check each uncollapsed cell for potential collapse
            for r in range(self.size):
                for c in range(self.size):
                    if collapsed[(r, c)] is not None:
                        continue

                    remaining = set(self.digits) - removed_digits[(r, c)]
                    is_col, digit = self.is_collapsed(cell_states[(r, c)], remaining)

                    if is_col and digit is not None:
                        # Collapse this cell!
                        self.log(f"  [Collapse] ({r},{c}) → {digit}")
                        collapsed[(r, c)] = digit
                        cell_states[(r, c)] = self.digit_vectors[digit].copy()
                        self.collapses += 1
                        stable = False

                        # Propagate the collapse
                        self._propagate(cell_states, collapsed, removed_digits, r, c, digit)

            # Check for contradictions (all digits removed)
            for r in range(self.size):
                for c in range(self.size):
                    if collapsed[(r, c)] is None:
                        # Contradiction if all digits have been removed
                        if len(removed_digits[(r, c)]) >= self.size:
                            self.log(f"\n✗ Contradiction at ({r},{c}) - all {self.size} digits removed")
                            return False, self._build_grid(collapsed)

        # Check if solved
        all_collapsed = all(collapsed[(r, c)] is not None
                          for r in range(self.size) for c in range(self.size))

        if all_collapsed:
            self.log(f"\n✓ All cells collapsed after {self.iterations} iterations")
        else:
            uncollapsed = sum(1 for r in range(self.size) for c in range(self.size)
                             if collapsed[(r, c)] is None)
            self.log(f"\n⚠ Stable but {uncollapsed} cells not collapsed")

            # Log state of uncollapsed cells
            for r in range(self.size):
                for c in range(self.size):
                    if collapsed[(r, c)] is None:
                        sims = self.digit_similarities(cell_states[(r, c)])
                        top_3 = sorted(sims.items(), key=lambda x: x[1], reverse=True)[:3]
                        sim_str = ", ".join([f"{d}:{s:.3f}" for d, s in top_3])
                        self.log(f"    ({r},{c}): {sim_str}")

        grid = self._build_grid(collapsed)
        return all_collapsed, grid

    def _propagate(self, cell_states: Dict, collapsed: Dict,
                   removed_digits: Dict, row: int, col: int, digit: int):
        """Propagate a collapse to all peers."""
        for pr, pc in self.get_peers(row, col):
            if collapsed[(pr, pc)] is None:
                # Only remove if not already removed
                if digit not in removed_digits[(pr, pc)]:
                    old_state = cell_states[(pr, pc)]
                    new_state = self.remove_digit_from_cell(old_state, digit)
                    cell_states[(pr, pc)] = new_state
                    removed_digits[(pr, pc)].add(digit)
                    self.propagations += 1

    def _build_grid(self, collapsed: Dict) -> List[List[Optional[int]]]:
        """Build a grid from collapsed state."""
        grid = [[None for _ in range(self.size)] for _ in range(self.size)]
        for r in range(self.size):
            for c in range(self.size):
                grid[r][c] = collapsed[(r, c)]
        return grid


def test_4x4():
    """Test on 4x4 puzzle."""
    print("\n" + "=" * 60)
    print("TEST: 4x4 Sudoku with Superposition Collapse")
    print("=" * 60)

    result = ApproachResult("Approach 2: Superposition Collapse")
    result.puzzle_size = 4
    result.puzzle_name = "4x4 Easy"

    print("\nInput puzzle:")
    print_grid_4x4(PUZZLE_4x4_EASY)

    solver = SuperpositionSolver(size=4, verbose=True)

    with Timer() as timer:
        solved, grid = solver.solve(PUZZLE_4x4_EASY)

    result.time_seconds = timer.elapsed
    result.iterations = solver.iterations
    result.backtracking_used = False

    print("\nResult:")
    print_grid_4x4(grid)

    valid, msg = validate_4x4(grid)
    result.valid = valid
    result.validation_msg = msg
    result.solved = solved and valid

    print(f"\nSolved: {solved}")
    print(f"Valid: {valid} - {msg}")
    print(f"Time: {timer.elapsed:.4f}s")
    print(f"Iterations: {solver.iterations}")
    print(f"Collapses: {solver.collapses}")
    print(f"Propagations: {solver.propagations}")

    return result


def test_9x9_easy():
    """Test on 9x9 easy puzzle."""
    print("\n" + "=" * 60)
    print("TEST: 9x9 Easy Sudoku with Superposition Collapse")
    print("=" * 60)

    result = ApproachResult("Approach 2: Superposition Collapse")
    result.puzzle_size = 9
    result.puzzle_name = "9x9 Easy"

    print("\nInput puzzle:")
    print_grid_9x9(PUZZLE_9x9_EASY)

    solver = SuperpositionSolver(size=9, verbose=True)

    with Timer() as timer:
        solved, grid = solver.solve(PUZZLE_9x9_EASY)

    result.time_seconds = timer.elapsed
    result.iterations = solver.iterations
    result.backtracking_used = False

    print("\nResult:")
    print_grid_9x9(grid)

    valid, msg = validate_9x9(grid)
    result.valid = valid
    result.validation_msg = msg
    result.solved = solved and valid

    print(f"\nSolved: {solved}")
    print(f"Valid: {valid} - {msg}")
    print(f"Time: {timer.elapsed:.4f}s")
    print(f"Iterations: {solver.iterations}")
    print(f"Collapses: {solver.collapses}")
    print(f"Propagations: {solver.propagations}")

    return result


def test_9x9_hard():
    """Test on 9x9 hard puzzle."""
    print("\n" + "=" * 60)
    print("TEST: 9x9 HARD Sudoku with Superposition Collapse")
    print("=" * 60)
    print("\nThis puzzle requires guessing in traditional solvers.")
    print("Can superposition collapse solve it purely geometrically?")

    result = ApproachResult("Approach 2: Superposition Collapse")
    result.puzzle_size = 9
    result.puzzle_name = "9x9 Hard"

    print("\nInput puzzle:")
    print_grid_9x9(PUZZLE_9x9_HARD)
    print(f"Empty cells: {count_empty(PUZZLE_9x9_HARD)}")

    solver = SuperpositionSolver(size=9, verbose=True)

    with Timer() as timer:
        solved, grid = solver.solve(PUZZLE_9x9_HARD)

    result.time_seconds = timer.elapsed
    result.iterations = solver.iterations
    result.backtracking_used = False

    print("\nResult:")
    print_grid_9x9(grid)

    valid, msg = validate_9x9(grid)
    result.valid = valid
    result.validation_msg = msg
    result.solved = solved and valid

    print(f"\nSolved: {solved}")
    print(f"Valid: {valid} - {msg}")
    print(f"Time: {timer.elapsed:.4f}s")
    print(f"Iterations: {solver.iterations}")
    print(f"Collapses: {solver.collapses}")
    print(f"Propagations: {solver.propagations}")

    return result


def analyze_superposition_behavior():
    """Analyze how superposition collapse behaves."""
    print("\n" + "=" * 60)
    print("ANALYSIS: Superposition Collapse Behavior")
    print("=" * 60)

    client = create_client(dimensions=16384)
    cache = VectorCache(client)

    # Get digit vectors
    digit_vecs = {d: cache.get_digit_vector(d) for d in range(1, 10)}

    # Create full superposition
    full_super = bundle(list(digit_vecs.values()))

    print("\n1. Initial superposition similarities:")
    for d in range(1, 10):
        sim = similarity(full_super, digit_vecs[d])
        print(f"   Digit {d}: {sim:.4f}")

    print("\n2. After removing digits one by one:")
    current = full_super.copy()
    for remove_d in [1, 2, 3, 4, 5]:
        current = remove_component(current, digit_vecs[remove_d])
        print(f"\n   After removing {remove_d}:")
        for d in range(1, 10):
            sim = similarity(current, digit_vecs[d])
            marker = " ← REMOVED" if d <= remove_d else ""
            print(f"      Digit {d}: {sim:.4f}{marker}")

    print("\n3. Single remaining digit detection:")
    # Start fresh and remove all but one
    current = full_super.copy()
    keep_digit = 7
    for d in range(1, 10):
        if d != keep_digit:
            current = remove_component(current, digit_vecs[d])

    print(f"   Kept only digit {keep_digit}:")
    for d in range(1, 10):
        sim = similarity(current, digit_vecs[d])
        marker = " ← KEPT" if d == keep_digit else ""
        print(f"      Digit {d}: {sim:.4f}{marker}")


def main():
    print("=" * 60)
    print("APPROACH 2: SUPERPOSITION COLLAPSE")
    print("=" * 60)
    print("\nHypothesis: Unknown cells as superpositions, constraints as removal.")
    print("Method: Propagate collapses until all cells settle.")

    # Analyze behavior first
    analyze_superposition_behavior()

    # Test puzzles
    result_4x4 = test_4x4()
    result_9x9_easy = test_9x9_easy()
    result_9x9_hard = test_9x9_hard()

    # Summary
    print("\n" + "=" * 60)
    print("APPROACH 2 SUMMARY")
    print("=" * 60)

    print(f"\n4x4: {'✓ SOLVED' if result_4x4.solved else '✗ FAILED'}")
    print(f"9x9 Easy: {'✓ SOLVED' if result_9x9_easy.solved else '✗ FAILED'}")
    print(f"9x9 Hard: {'✓ SOLVED' if result_9x9_hard.solved else '✗ FAILED'}")

    if result_9x9_hard.solved:
        print("\n✓✓ BREAKTHROUGH: Hard puzzle solved via superposition collapse!")
    elif result_9x9_easy.solved:
        print("\n◐ PARTIAL: Easy puzzles work, hard needs enhancement")
    else:
        print("\n✗ Superposition collapse needs refinement")


if __name__ == "__main__":
    main()
