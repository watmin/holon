#!/usr/bin/env python3
"""
Approach 1: Hopfield-Style Constraint Resonance

HYPOTHESIS:
Treat the Sudoku grid as an energy-based system.
Each cell's state is a vector that evolves toward valid configurations.
Constraints act as "attractors" pulling cells toward consistency.

KEY IDEA:
- Hopfield networks find stable states via energy minimization
- Valid Sudoku configurations are energy minima
- Cells iteratively update based on constraint satisfaction

MECHANISM:
1. Each cell starts as superposition of possible digits
2. Define "energy" based on constraint violations
3. Iteratively update cells to reduce energy
4. Cells should "settle" into valid configuration

This is fundamentally different from greedy approaches because
it allows cells to influence each other iteratively until
the system reaches a stable (hopefully valid) state.
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


class HopfieldSolver:
    """
    Sudoku solver using Hopfield-style energy minimization.

    Key difference from greedy: cells influence each other iteratively.
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

        # Full superposition
        self.full_super = bundle([self.digit_vectors[d] for d in self.digits])

        self.notes: List[str] = []
        self.iterations = 0

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

    def compute_energy(self, cell_states: Dict[Tuple[int, int], np.ndarray]) -> float:
        """
        Compute total energy of the system.

        Energy is based on constraint violations:
        - Lower energy = more consistent state
        - Valid solution = minimum energy

        Energy terms:
        1. Peer similarity penalty: if two peers have similar states, that's bad
        2. Digit clarity reward: if a cell clearly represents one digit, that's good
        """
        energy = 0.0

        # Peer similarity penalty
        for r in range(self.size):
            for c in range(self.size):
                cell_vec = cell_states[(r, c)]
                for pr, pc in self.get_peers(r, c):
                    peer_vec = cell_states[(pr, pc)]
                    sim = similarity(cell_vec, peer_vec)
                    # High similarity between peers = bad = higher energy
                    energy += max(0, sim) ** 2

        # Digit clarity reward (negative energy for clear states)
        for r in range(self.size):
            for c in range(self.size):
                cell_vec = cell_states[(r, c)]
                # Find max similarity to any digit
                max_sim = max(similarity(cell_vec, self.digit_vectors[d])
                             for d in self.digits)
                # Clear state = high max_sim = lower energy
                energy -= max_sim

        return energy

    def extract_digit(self, cell_vec: np.ndarray) -> Tuple[int, float]:
        """Extract the most likely digit from a cell vector."""
        best_d = None
        best_sim = -1
        for d in self.digits:
            sim = similarity(cell_vec, self.digit_vectors[d])
            if sim > best_sim:
                best_sim = sim
                best_d = d
        return best_d, best_sim

    def update_cell(self, cell_states: Dict, row: int, col: int,
                    fixed: Set[Tuple[int, int]], learning_rate: float = 0.5) -> np.ndarray:
        """
        Update a cell based on its peers' states.

        The update rule: move away from what peers represent.
        If peers strongly represent digit X, this cell should move away from X.
        """
        if (row, col) in fixed:
            return cell_states[(row, col)]

        current = cell_states[(row, col)]

        # Collect what peers represent
        peer_influence = np.zeros_like(current)
        for pr, pc in self.get_peers(row, col):
            peer_vec = cell_states[(pr, pc)]
            peer_influence += peer_vec

        # Normalize
        peer_norm = np.linalg.norm(peer_influence)
        if peer_norm > 0:
            peer_influence = peer_influence / peer_norm

        # Move AWAY from peer influence
        # New state = current - learning_rate * peer_influence
        new_state = current - learning_rate * peer_influence

        # Normalize to unit length
        norm = np.linalg.norm(new_state)
        if norm > 0:
            new_state = new_state / norm

        return new_state

    def solve(self, puzzle: List[List[Optional[int]]],
              max_iterations: int = 100,
              learning_rate: float = 0.3) -> Tuple[bool, List[List[int]]]:
        """
        Solve using Hopfield-style iterative relaxation.
        """
        self.log(f"\n{'='*60}")
        self.log("APPROACH 1: HOPFIELD CONSTRAINT RESONANCE")
        self.log(f"{'='*60}")
        self.log(f"Puzzle size: {self.size}x{self.size}")
        self.log(f"Empty cells: {count_empty(puzzle)}")
        self.log(f"Max iterations: {max_iterations}, Learning rate: {learning_rate}")

        # Initialize cell states
        cell_states: Dict[Tuple[int, int], np.ndarray] = {}
        fixed: Set[Tuple[int, int]] = set()

        for r in range(self.size):
            for c in range(self.size):
                if puzzle[r][c] is not None:
                    # Fixed cell - use digit vector
                    cell_states[(r, c)] = self.digit_vectors[puzzle[r][c]].copy()
                    fixed.add((r, c))
                else:
                    # Unknown cell - start with superposition
                    cell_states[(r, c)] = self.full_super.copy()

        # Initial energy
        initial_energy = self.compute_energy(cell_states)
        self.log(f"\nInitial energy: {initial_energy:.4f}")

        # Iterative relaxation
        prev_energy = initial_energy
        stable_count = 0

        for iteration in range(max_iterations):
            self.iterations += 1

            # Update all non-fixed cells
            new_states = {}
            for r in range(self.size):
                for c in range(self.size):
                    new_states[(r, c)] = self.update_cell(
                        cell_states, r, c, fixed, learning_rate
                    )

            cell_states = new_states

            # Compute new energy
            energy = self.compute_energy(cell_states)

            if iteration % 10 == 0:
                self.log(f"  Iteration {iteration}: energy = {energy:.4f}")

            # Check for stability
            if abs(energy - prev_energy) < 0.001:
                stable_count += 1
                if stable_count >= 5:
                    self.log(f"\n✓ Stable after {iteration} iterations")
                    break
            else:
                stable_count = 0

            prev_energy = energy

        # Extract solution
        self.log(f"\nFinal energy: {energy:.4f}")
        self.log(f"Energy reduction: {initial_energy - energy:.4f}")

        grid = [[None for _ in range(self.size)] for _ in range(self.size)]
        for r in range(self.size):
            for c in range(self.size):
                digit, conf = self.extract_digit(cell_states[(r, c)])
                grid[r][c] = digit
                if (r, c) not in fixed:
                    self.log(f"  ({r},{c}) → {digit} (conf={conf:.3f})")

        return True, grid


def test_4x4():
    print("\n" + "=" * 60)
    print("TEST: 4x4 with Hopfield")
    print("=" * 60)

    print("\nInput:")
    print_grid_4x4(PUZZLE_4x4_EASY)

    solver = HopfieldSolver(size=4, verbose=True)
    with Timer() as timer:
        solved, grid = solver.solve(PUZZLE_4x4_EASY, max_iterations=50)

    print("\nResult:")
    print_grid_4x4(grid)

    valid, msg = validate_4x4(grid)
    print(f"\nValid: {valid} - {msg}")
    print(f"Time: {timer.elapsed:.2f}s")

    return valid


def test_9x9_hard():
    print("\n" + "=" * 60)
    print("TEST: 9x9 HARD with Hopfield")
    print("=" * 60)

    print("\nInput:")
    print_grid_9x9(PUZZLE_9x9_HARD)

    solver = HopfieldSolver(size=9, verbose=True)
    with Timer() as timer:
        solved, grid = solver.solve(PUZZLE_9x9_HARD, max_iterations=100)

    print("\nResult:")
    print_grid_9x9(grid)

    valid, msg = validate_9x9(grid)
    cells = sum(1 for r in range(9) for c in range(9) if grid[r][c] is not None)
    print(f"\nCells filled: {cells}/81")
    print(f"Valid: {valid}")
    print(f"Time: {timer.elapsed:.2f}s")

    return valid


def main():
    print("=" * 60)
    print("APPROACH 1: HOPFIELD CONSTRAINT RESONANCE")
    print("=" * 60)
    print("\nHypothesis: Iterative energy minimization finds global consistency.")
    print("Method: Cells update based on peers, settling into valid configuration.")

    result_4x4 = test_4x4()
    result_hard = test_9x9_hard()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"4x4: {'✓' if result_4x4 else '✗'}")
    print(f"9x9 Hard: {'✓' if result_hard else '✗'}")


if __name__ == "__main__":
    main()
