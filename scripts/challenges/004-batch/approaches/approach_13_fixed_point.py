#!/usr/bin/env python3
"""
Approach 13: Solution as Fixed Point

RADICAL IDEA:
What if the solution is a FIXED POINT of an operation F(x) = x?

We define an operation that, when applied repeatedly:
- Moves partial states toward the solution
- Leaves the solution unchanged

This is inspired by:
1. Hopfield networks (energy minimization)
2. Belief propagation (message passing)
3. Power iteration (finding eigenvectors)

THE OPERATION:
For each cell, compute a "pressure" from all constraints.
Update the cell's state to reduce conflict with constraints.
Repeat until stable.

KEY INSIGHT:
The solution is the unique configuration where all constraints agree.
We're looking for the state where "constraint pressure" is zero everywhere.
"""

import sys
sys.path.insert(0, "/home/watmin/work/holon")

from typing import List, Optional, Set, Tuple, Dict
import numpy as np
from collections import defaultdict

from common import (
    create_client,
    VectorCache,
    bind,
    unbind,
    bundle,
    similarity,
    Timer,
    print_grid_9x9,
    validate_9x9,
    count_empty,
    get_available_digits_9x9,
    PUZZLE_9x9_HARD,
)


SOLUTION_9x9_HARD = [
    [5, 8, 1, 6, 7, 2, 4, 3, 9],
    [7, 9, 2, 8, 4, 3, 6, 5, 1],
    [3, 6, 4, 5, 9, 1, 7, 8, 2],
    [4, 3, 8, 9, 5, 7, 2, 1, 6],
    [2, 5, 6, 1, 8, 4, 9, 7, 3],
    [1, 7, 9, 3, 2, 6, 8, 4, 5],
    [8, 4, 5, 2, 1, 9, 3, 6, 7],
    [9, 1, 3, 7, 6, 8, 5, 2, 4],
    [6, 2, 7, 4, 3, 5, 1, 9, 8],
]


class FixedPointSolver:
    """
    Find solution as fixed point of constraint satisfaction operation.
    """

    def __init__(self, dimensions: int = 16384):
        self.dimensions = dimensions
        self.client = create_client(dimensions=dimensions)
        self.cache = VectorCache(self.client)

        self.digits = list(range(1, 10))
        self.digit_vectors = {d: self.cache.get_digit_vector(d) for d in self.digits}

    def initialize_state(self, puzzle: List[List[Optional[int]]]) -> Dict[Tuple[int, int], np.ndarray]:
        """
        Initialize each cell as a probability vector over digits.

        Known cells: sharp vector (just that digit)
        Unknown cells: superposition of available digits
        """
        state = {}

        for r in range(9):
            for c in range(9):
                if puzzle[r][c] is not None:
                    # Known: sharp
                    state[(r, c)] = self.digit_vectors[puzzle[r][c]].copy()
                else:
                    # Unknown: superposition of available
                    available = get_available_digits_9x9(puzzle, r, c)
                    if available:
                        state[(r, c)] = bundle([self.digit_vectors[d] for d in available])
                    else:
                        state[(r, c)] = np.zeros(self.dimensions)

        return state

    def get_constraint_cells(self, row: int, col: int) -> List[List[Tuple[int, int]]]:
        """Get cells in each constraint (row, col, block) for position (row, col)."""
        # Row constraint
        row_cells = [(row, c) for c in range(9) if c != col]

        # Column constraint
        col_cells = [(r, col) for r in range(9) if r != row]

        # Block constraint
        br, bc = (row // 3) * 3, (col // 3) * 3
        block_cells = [(r, c) for r in range(br, br+3) for c in range(bc, bc+3)
                       if (r, c) != (row, col)]

        return [row_cells, col_cells, block_cells]

    def compute_constraint_message(self, state: Dict, constraint_cells: List[Tuple[int, int]]) -> np.ndarray:
        """
        Compute what this constraint "says" about what's available.

        If cells in constraint have certain digits, those digits are NOT available.
        The message is a superposition of what IS available.
        """
        # Find which digits are "taken" by this constraint
        taken_digits = []
        for (r, c) in constraint_cells:
            cell_vec = state[(r, c)]
            # Find the dominant digit in this cell
            sims = {d: similarity(cell_vec, self.digit_vectors[d]) for d in self.digits}
            max_sim = max(sims.values())
            if max_sim > 0.5:  # Confident enough
                best_d = max(sims, key=lambda x: sims[x])
                taken_digits.append(best_d)

        # Available = all - taken
        available = [d for d in self.digits if d not in taken_digits]

        if available:
            return bundle([self.digit_vectors[d] for d in available])
        else:
            return np.zeros(self.dimensions)

    def iterate(self, state: Dict, fixed_cells: Set[Tuple[int, int]],
                learning_rate: float = 0.3) -> Dict:
        """
        One iteration: update each non-fixed cell based on constraint messages.
        """
        new_state = {k: v.copy() for k, v in state.items()}

        for r in range(9):
            for c in range(9):
                if (r, c) in fixed_cells:
                    continue

                constraints = self.get_constraint_cells(r, c)

                # Get message from each constraint
                messages = []
                for constraint_cells in constraints:
                    msg = self.compute_constraint_message(state, constraint_cells)
                    if np.any(msg != 0):
                        messages.append(msg)

                if not messages:
                    continue

                # Combined message = intersection of what all constraints allow
                combined = messages[0]
                for msg in messages[1:]:
                    # Element-wise product emphasizes agreement
                    combined = combined * msg

                # Normalize
                norm = np.linalg.norm(combined)
                if norm > 0:
                    combined = combined / norm

                # Update: move toward combined message
                new_state[(r, c)] = (1 - learning_rate) * state[(r, c)] + learning_rate * combined

        return new_state

    def decode_state(self, state: Dict) -> List[List[Optional[int]]]:
        """Decode state into grid by picking max-similarity digit for each cell."""
        grid = [[None for _ in range(9)] for _ in range(9)]

        for r in range(9):
            for c in range(9):
                cell_vec = state[(r, c)]
                sims = {d: similarity(cell_vec, self.digit_vectors[d]) for d in self.digits}
                best_d = max(sims, key=lambda x: sims[x])
                grid[r][c] = best_d

        return grid

    def compute_entropy(self, state: Dict) -> float:
        """Measure how "collapsed" the state is. Lower = more decided."""
        total_entropy = 0
        for (r, c), vec in state.items():
            sims = np.array([similarity(vec, self.digit_vectors[d]) for d in self.digits])
            # Convert to probabilities
            sims = np.maximum(sims, 0)  # Clamp negatives
            if np.sum(sims) > 0:
                probs = sims / np.sum(sims)
                # Shannon entropy
                entropy = -np.sum(probs * np.log(probs + 1e-10))
                total_entropy += entropy
        return total_entropy

    def solve(self, puzzle: List[List[Optional[int]]], max_iters: int = 100,
              verbose: bool = True) -> Tuple[bool, List[List[int]]]:
        """
        Iterate until fixed point.
        """
        if verbose:
            print(f"\n{'='*60}")
            print("FIXED POINT SOLVER")
            print(f"{'='*60}")

        # Initialize
        state = self.initialize_state(puzzle)
        fixed_cells = {(r, c) for r in range(9) for c in range(9) if puzzle[r][c] is not None}

        if verbose:
            print(f"Fixed cells: {len(fixed_cells)}")
            print(f"Initial entropy: {self.compute_entropy(state):.2f}")

        # Iterate
        prev_entropy = float('inf')
        for i in range(max_iters):
            state = self.iterate(state, fixed_cells)
            entropy = self.compute_entropy(state)

            if verbose and i % 10 == 0:
                print(f"  Iter {i}: entropy = {entropy:.2f}")

            # Check for convergence
            if abs(entropy - prev_entropy) < 0.01:
                if verbose:
                    print(f"  Converged at iteration {i}")
                break

            prev_entropy = entropy

        # Decode
        grid = self.decode_state(state)

        # Validate
        valid, msg = validate_9x9(grid)

        if verbose:
            print(f"\nFinal entropy: {entropy:.2f}")
            print(f"Valid: {valid}")
            if not valid:
                print(f"Error: {msg}")

        return valid, grid


class BeliefPropagationSolver:
    """
    Alternative: Belief propagation style message passing.
    """

    def __init__(self, dimensions: int = 16384):
        self.dimensions = dimensions
        self.client = create_client(dimensions=dimensions)
        self.cache = VectorCache(self.client)

        self.digits = list(range(1, 10))
        self.digit_vectors = {d: self.cache.get_digit_vector(d) for d in self.digits}

    def solve(self, puzzle: List[List[Optional[int]]], max_iters: int = 50,
              verbose: bool = True) -> Tuple[bool, List[List[int]]]:
        """
        Message passing: each cell sends messages to its constraint neighbors.
        """
        if verbose:
            print(f"\n{'='*60}")
            print("BELIEF PROPAGATION SOLVER")
            print(f"{'='*60}")

        # Initialize beliefs: probability distribution over digits
        beliefs = {}
        for r in range(9):
            for c in range(9):
                if puzzle[r][c] is not None:
                    # Known: certain
                    beliefs[(r, c)] = {d: 1.0 if d == puzzle[r][c] else 0.0 for d in self.digits}
                else:
                    # Unknown: uniform over available
                    available = get_available_digits_9x9(puzzle, r, c)
                    beliefs[(r, c)] = {d: 1.0/len(available) if d in available else 0.0
                                       for d in self.digits}

        fixed_cells = {(r, c) for r in range(9) for c in range(9) if puzzle[r][c] is not None}

        # Messages: from cell to constraint
        # We'll simplify: just update beliefs based on constraints

        for iteration in range(max_iters):
            changed = False

            for r in range(9):
                for c in range(9):
                    if (r, c) in fixed_cells:
                        continue

                    old_beliefs = beliefs[(r, c)].copy()

                    # Get constraints
                    row_cells = [(r, cc) for cc in range(9) if cc != c]
                    col_cells = [(rr, c) for rr in range(9) if rr != r]
                    br, bc = (r // 3) * 3, (c // 3) * 3
                    block_cells = [(rr, cc) for rr in range(br, br+3) for cc in range(bc, bc+3)
                                   if (rr, cc) != (r, c)]

                    # For each digit, compute probability it's allowed
                    for d in self.digits:
                        # Probability d is NOT taken by constraint
                        prob_row = 1.0 - sum(beliefs[cell].get(d, 0) for cell in row_cells)
                        prob_col = 1.0 - sum(beliefs[cell].get(d, 0) for cell in col_cells)
                        prob_block = 1.0 - sum(beliefs[cell].get(d, 0) for cell in block_cells)

                        # Combined: must be allowed by all
                        prob_allowed = max(0, prob_row) * max(0, prob_col) * max(0, prob_block)
                        beliefs[(r, c)][d] *= prob_allowed

                    # Normalize
                    total = sum(beliefs[(r, c)].values())
                    if total > 0:
                        for d in self.digits:
                            beliefs[(r, c)][d] /= total

                    # Check if beliefs changed
                    for d in self.digits:
                        if abs(beliefs[(r, c)][d] - old_beliefs[d]) > 0.01:
                            changed = True

            if verbose and iteration % 10 == 0:
                # Count cells that are "decided" (one digit has prob > 0.9)
                decided = sum(1 for cell in beliefs if max(beliefs[cell].values()) > 0.9)
                print(f"  Iter {iteration}: {decided}/81 cells decided")

            if not changed:
                if verbose:
                    print(f"  Converged at iteration {iteration}")
                break

        # Decode: pick max probability digit
        grid = [[None for _ in range(9)] for _ in range(9)]
        for r in range(9):
            for c in range(9):
                best_d = max(self.digits, key=lambda d: beliefs[(r, c)][d])
                grid[r][c] = best_d

        valid, msg = validate_9x9(grid)

        if verbose:
            print(f"Valid: {valid}")
            if not valid:
                print(f"Error: {msg}")

        return valid, grid


def main():
    print("=" * 70)
    print("APPROACH 13: SOLUTION AS FIXED POINT")
    print("=" * 70)
    print("\nIdea: The solution is where 'constraint pressure' is zero.")
    print("Iterate until stable.\n")

    print_grid_9x9(PUZZLE_9x9_HARD)

    # Test fixed point solver
    solver1 = FixedPointSolver(dimensions=16384)
    valid1, grid1 = solver1.solve(PUZZLE_9x9_HARD, verbose=True)

    print("\nResult:")
    print_grid_9x9(grid1)

    # Count correct
    correct = sum(1 for r in range(9) for c in range(9) if grid1[r][c] == SOLUTION_9x9_HARD[r][c])
    print(f"\nCorrect cells: {correct}/81")

    # Test belief propagation
    solver2 = BeliefPropagationSolver(dimensions=16384)
    valid2, grid2 = solver2.solve(PUZZLE_9x9_HARD, verbose=True)

    print("\nResult:")
    print_grid_9x9(grid2)

    correct2 = sum(1 for r in range(9) for c in range(9) if grid2[r][c] == SOLUTION_9x9_HARD[r][c])
    print(f"\nCorrect cells: {correct2}/81")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    if valid1 or valid2:
        print("✓ Fixed point iteration found the solution!")
    else:
        print("""
Neither solver found a valid solution via iteration alone.

This confirms: the constraint satisfaction landscape has LOCAL MINIMA.
The iteration gets stuck before reaching the global solution.

The fixed point IS the solution, but there are other fixed points
(local minima) that the iteration can converge to instead.
""")


if __name__ == "__main__":
    main()
