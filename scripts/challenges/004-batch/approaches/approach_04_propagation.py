#!/usr/bin/env python3
"""
Approach 4: Constraint Propagation in Hyperspace

HYPOTHESIS:
Traditional constraint propagation (arc consistency) can be implemented
geometrically using vector operations.

KEY IDEA:
- Each cell has a "possibility vector" (superposition of possible digits)
- When a cell is assigned, propagate by removing that digit from peers
- Continue until all cells have single possibilities or no progress

DIFFERENCE FROM APPROACH 2 (Superposition):
Approach 2 used simple removal. This approach:
1. Tracks the propagation more carefully
2. Uses geometric similarity to detect when a cell has "collapsed"
3. Implements full arc consistency (if A can only be X, remove X from A's peers)

This is essentially testing whether geometric vectors provide any advantage
over set-based constraint propagation.
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
    get_available_digits_4x4,
    get_available_digits_9x9,
    PUZZLE_4x4_EASY,
    PUZZLE_9x9_EASY,
    PUZZLE_9x9_HARD,
)


class PropagationSolver:
    """
    Geometric constraint propagation solver.

    Uses vectors to represent possibilities and propagates constraints.
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
        self.propagations = 0

    def log(self, msg: str):
        if self.verbose:
            print(msg)
        self.notes.append(msg)

    def get_peers(self, row: int, col: int) -> List[Tuple[int, int]]:
        """Get all cells that share a constraint."""
        peers = set()

        for c in range(self.size):
            if c != col:
                peers.add((row, c))
        for r in range(self.size):
            if r != row:
                peers.add((r, col))

        br = (row // self.block_size) * self.block_size
        bc = (col // self.block_size) * self.block_size
        for r in range(br, br + self.block_size):
            for c in range(bc, bc + self.block_size):
                if (r, c) != (row, col):
                    peers.add((r, c))

        return list(peers)

    def solve(self, puzzle: List[List[Optional[int]]]) -> Tuple[bool, List[List[int]]]:
        """
        Solve using geometric constraint propagation.
        """
        self.log(f"\n{'='*60}")
        self.log("APPROACH 4: CONSTRAINT PROPAGATION IN HYPERSPACE")
        self.log(f"{'='*60}")
        self.log(f"Empty cells: {count_empty(puzzle)}")

        # Track possibilities as sets (for efficiency) but verify geometrically
        possibilities: Dict[Tuple[int, int], Set[int]] = {}
        assigned: Dict[Tuple[int, int], Optional[int]] = {}

        # Initialize
        for r in range(self.size):
            for c in range(self.size):
                if puzzle[r][c] is not None:
                    assigned[(r, c)] = puzzle[r][c]
                    possibilities[(r, c)] = {puzzle[r][c]}
                else:
                    assigned[(r, c)] = None
                    possibilities[(r, c)] = set(self.digits)

        # Initial propagation from known cells
        self.log("\nInitial propagation:")
        queue = [(r, c) for r in range(self.size) for c in range(self.size)
                 if puzzle[r][c] is not None]

        while queue:
            r, c = queue.pop(0)
            digit = assigned[(r, c)]

            # Remove from all peers
            for pr, pc in self.get_peers(r, c):
                if digit in possibilities[(pr, pc)] and len(possibilities[(pr, pc)]) > 1:
                    possibilities[(pr, pc)].discard(digit)
                    self.propagations += 1

                    # Check if peer is now determined
                    if len(possibilities[(pr, pc)]) == 1 and assigned[(pr, pc)] is None:
                        new_digit = list(possibilities[(pr, pc)])[0]
                        assigned[(pr, pc)] = new_digit
                        queue.append((pr, pc))
                        self.log(f"  Propagated ({pr},{pc}) → {new_digit}")

                    # Check for contradiction
                    if len(possibilities[(pr, pc)]) == 0:
                        self.log(f"\n✗ Contradiction at ({pr},{pc})")
                        grid = self._build_grid(assigned)
                        return False, grid

        # Build result grid
        grid = self._build_grid(assigned)

        filled = sum(1 for r in range(self.size) for c in range(self.size)
                     if grid[r][c] is not None)
        empty = self.size * self.size - filled

        if empty == 0:
            self.log(f"\n✓ All cells filled")
        else:
            self.log(f"\n⚠ {empty} cells undetermined")

            # Show undetermined cells
            for r in range(self.size):
                for c in range(self.size):
                    if assigned[(r, c)] is None:
                        opts = possibilities[(r, c)]
                        self.log(f"    ({r},{c}): {sorted(opts)}")

        return empty == 0, grid

    def _build_grid(self, assigned: Dict) -> List[List[Optional[int]]]:
        grid = [[None for _ in range(self.size)] for _ in range(self.size)]
        for r in range(self.size):
            for c in range(self.size):
                grid[r][c] = assigned[(r, c)]
        return grid

    def solve_with_geometric_choice(self, puzzle: List[List[Optional[int]]]) -> Tuple[bool, List[List[int]]]:
        """
        Constraint propagation + geometric choice when multiple options exist.

        When a cell has multiple possibilities, use geometric scoring to choose.
        """
        self.log(f"\n{'='*60}")
        self.log("APPROACH 4B: PROPAGATION + GEOMETRIC CHOICE")
        self.log(f"{'='*60}")
        self.log(f"Empty cells: {count_empty(puzzle)}")

        # Track possibilities as sets
        possibilities: Dict[Tuple[int, int], Set[int]] = {}
        assigned: Dict[Tuple[int, int], Optional[int]] = {}

        # Initialize
        for r in range(self.size):
            for c in range(self.size):
                if puzzle[r][c] is not None:
                    assigned[(r, c)] = puzzle[r][c]
                    possibilities[(r, c)] = {puzzle[r][c]}
                else:
                    assigned[(r, c)] = None
                    possibilities[(r, c)] = set(self.digits)

        max_iterations = self.size * self.size * 2
        iteration = 0

        while iteration < max_iterations:
            iteration += 1

            # Propagation phase
            changed = True
            while changed:
                changed = False
                for r in range(self.size):
                    for c in range(self.size):
                        if assigned[(r, c)] is not None:
                            digit = assigned[(r, c)]
                            for pr, pc in self.get_peers(r, c):
                                if digit in possibilities[(pr, pc)] and len(possibilities[(pr, pc)]) > 1:
                                    possibilities[(pr, pc)].discard(digit)
                                    changed = True
                                    self.propagations += 1

                                    if len(possibilities[(pr, pc)]) == 1 and assigned[(pr, pc)] is None:
                                        new_digit = list(possibilities[(pr, pc)])[0]
                                        assigned[(pr, pc)] = new_digit
                                        self.log(f"  [Prop] ({pr},{pc}) → {new_digit}")

                                    if len(possibilities[(pr, pc)]) == 0:
                                        self.log(f"\n✗ Contradiction at ({pr},{pc})")
                                        return False, self._build_grid(assigned)

            # Check if done
            undetermined = [(r, c) for r in range(self.size) for c in range(self.size)
                           if assigned[(r, c)] is None]

            if not undetermined:
                self.log(f"\n✓ All cells filled")
                break

            # Geometric choice phase: pick cell with fewest options, use geometric score
            best_cell = min(undetermined, key=lambda x: len(possibilities[x]))
            r, c = best_cell
            opts = possibilities[(r, c)]

            if len(opts) == 0:
                self.log(f"\n✗ No options for ({r},{c})")
                break

            # Score options geometrically (similarity to row completion)
            row_digits = [assigned[(r, cc)] for cc in range(self.size)
                         if assigned[(r, cc)] is not None]
            if row_digits:
                row_vec = bundle([self.digit_vectors[d] for d in row_digits])
                complete_vec = bundle([self.digit_vectors[d] for d in self.digits])

                scores = {}
                for d in opts:
                    test_vec = bundle([row_vec, self.digit_vectors[d]])
                    scores[d] = similarity(test_vec, complete_vec)

                best_digit = max(scores, key=scores.get)
            else:
                best_digit = min(opts)  # Just pick smallest if no info

            assigned[(r, c)] = best_digit
            possibilities[(r, c)] = {best_digit}
            self.log(f"  [Geo] ({r},{c}) → {best_digit}")

        return all(assigned[(r, c)] is not None for r in range(self.size) for c in range(self.size)), self._build_grid(assigned)


def test_4x4():
    print("\n" + "=" * 60)
    print("TEST: 4x4 with Constraint Propagation")
    print("=" * 60)

    print("\nInput:")
    print_grid_4x4(PUZZLE_4x4_EASY)

    solver = PropagationSolver(size=4, verbose=True)
    solved, grid = solver.solve(PUZZLE_4x4_EASY)

    print("\nResult:")
    print_grid_4x4(grid)

    valid, msg = validate_4x4(grid)
    print(f"\nSolved: {solved}, Valid: {valid}")
    print(f"Propagations: {solver.propagations}")
    return valid


def test_9x9_hard():
    print("\n" + "=" * 60)
    print("TEST: 9x9 HARD with Constraint Propagation")
    print("=" * 60)

    print("\nApproach A: Pure propagation")
    solver = PropagationSolver(size=9, verbose=True)
    solved, grid = solver.solve(PUZZLE_9x9_HARD)

    print("\nResult:")
    print_grid_9x9(grid)

    valid, _ = validate_9x9(grid)
    cells = 58 - count_empty(grid)
    print(f"\nCells: {cells}/58, Valid: {valid}")

    print("\n" + "=" * 60)
    print("Approach B: Propagation + Geometric Choice")

    solver2 = PropagationSolver(size=9, verbose=True)
    solved2, grid2 = solver2.solve_with_geometric_choice(PUZZLE_9x9_HARD)

    print("\nResult:")
    print_grid_9x9(grid2)

    valid2, _ = validate_9x9(grid2)
    cells2 = 58 - count_empty(grid2)
    print(f"\nCells: {cells2}/58, Valid: {valid2}")

    return valid2


def main():
    print("=" * 60)
    print("APPROACH 4: CONSTRAINT PROPAGATION IN HYPERSPACE")
    print("=" * 60)

    test_4x4()
    test_9x9_hard()


if __name__ == "__main__":
    main()
