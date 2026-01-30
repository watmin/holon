#!/usr/bin/env python3
"""
Approach 17: Prototype-Guided Solving

THE QUESTION:
Can abstract prototypes learned from solved puzzles
actually REDUCE backtracking on new puzzles?

THE EXPERIMENT:
1. Build abstract "good decision" prototypes from puzzle 1
2. Use prototypes to ORDER choices when solving puzzle 2
3. Compare backtrack count with/without guidance
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

# Additional test puzzles
PUZZLE_2 = [
    [None, 2, None, None, None, None, None, None, None],
    [None, None, None, 6, None, None, None, None, 3],
    [None, 7, 4, None, 8, None, None, None, None],
    [None, None, None, None, None, 3, None, None, 2],
    [None, 8, None, None, 4, None, None, 1, None],
    [6, None, None, 5, None, None, None, None, None],
    [None, None, None, None, 1, None, 7, 8, None],
    [5, None, None, None, None, 9, None, None, None],
    [None, None, None, None, None, None, None, 4, None],
]

PUZZLE_3 = [
    [None, None, None, None, None, None, None, 1, 2],
    [None, None, None, None, 3, 5, None, None, None],
    [None, None, None, 6, None, None, None, 7, None],
    [7, None, None, None, None, None, 3, None, None],
    [None, None, None, 4, None, None, 8, None, None],
    [1, None, None, None, None, None, None, None, None],
    [None, None, None, 1, 2, None, None, None, None],
    [None, 8, None, None, None, None, None, 4, None],
    [None, 5, None, None, None, None, 6, None, None],
]


class AbstractFeatureEncoder:
    """Encode decisions using abstract, transferable features."""

    def __init__(self, dimensions: int = 16384):
        self.dimensions = dimensions

        # Create feature basis vectors
        np.random.seed(42)

        # Number of options at decision point (1-9)
        self.NUM_OPTIONS = {i: self._random_vector(f'num_options_{i}') for i in range(1, 10)}

        # Block type (corner, edge, center of 3x3 block)
        self.BLOCK_TYPE = {
            'corner': self._random_vector('corner'),
            'edge': self._random_vector('edge'),
            'center': self._random_vector('center'),
        }

        # Constraint tightness levels (0-27)
        self.TIGHTNESS = {i: self._random_vector(f'tight_{i}') for i in range(28)}

        # Row/col position type
        self.POS_TYPE = {
            'top': self._random_vector('top'),
            'middle': self._random_vector('middle'),
            'bottom': self._random_vector('bottom'),
            'left': self._random_vector('left'),
            'center_h': self._random_vector('center_h'),
            'right': self._random_vector('right'),
        }

    def _random_vector(self, seed: str) -> np.ndarray:
        np.random.seed(hash(seed) % (2**32))
        return np.random.choice([-1.0, 1.0], size=self.dimensions)

    def get_block_type(self, r: int, c: int) -> str:
        br, bc = r % 3, c % 3
        if (br, bc) in [(0, 0), (0, 2), (2, 0), (2, 2)]:
            return 'corner'
        elif (br, bc) == (1, 1):
            return 'center'
        else:
            return 'edge'

    def get_row_type(self, r: int) -> str:
        if r < 3:
            return 'top'
        elif r < 6:
            return 'middle'
        else:
            return 'bottom'

    def get_col_type(self, c: int) -> str:
        if c < 3:
            return 'left'
        elif c < 6:
            return 'center_h'
        else:
            return 'right'

    def count_tightness(self, grid: List[List[Optional[int]]], r: int, c: int) -> int:
        filled = 0
        for cc in range(9):
            if grid[r][cc] is not None:
                filled += 1
        for rr in range(9):
            if grid[rr][c] is not None:
                filled += 1
        br, bc = (r // 3) * 3, (c // 3) * 3
        for rr in range(br, br + 3):
            for cc in range(bc, bc + 3):
                if grid[rr][cc] is not None:
                    filled += 1
        return min(filled, 27)

    def encode(self, grid: List[List[Optional[int]]], r: int, c: int,
               num_options: int) -> np.ndarray:
        """Encode a decision point using abstract features."""
        features = []

        # Number of options
        features.append(self.NUM_OPTIONS[num_options])

        # Block type
        features.append(self.BLOCK_TYPE[self.get_block_type(r, c)])

        # Row/col type
        features.append(self.POS_TYPE[self.get_row_type(r)])
        features.append(self.POS_TYPE[self.get_col_type(c)])

        # Constraint tightness
        tightness = self.count_tightness(grid, r, c)
        features.append(self.TIGHTNESS[tightness])

        return bundle(features)


class PrototypeLearner:
    """Learn good/bad decision prototypes from solved puzzles."""

    def __init__(self, encoder: AbstractFeatureEncoder):
        self.encoder = encoder
        self.good_decisions = []
        self.bad_decisions = []
        self.good_prototype = None
        self.bad_prototype = None

    def learn_from_puzzle(self, puzzle: List[List[Optional[int]]]):
        """Solve puzzle and learn from the decision tree."""
        decision_log = []

        def solve_and_log(grid, depth=0):
            # Propagate forced moves
            for _ in range(81):
                changed = False
                for r in range(9):
                    for c in range(9):
                        if grid[r][c] is None:
                            available = get_available_digits_9x9(grid, r, c)
                            if len(available) == 0:
                                return False
                            if len(available) == 1:
                                grid[r][c] = list(available)[0]
                                changed = True
                if not changed:
                    break

            empty = count_empty(grid)
            if empty == 0:
                valid, _ = validate_9x9(grid)
                return valid

            # Find MRV cell
            best_cell = None
            min_opts = 10
            for r in range(9):
                for c in range(9):
                    if grid[r][c] is None:
                        opts = get_available_digits_9x9(grid, r, c)
                        if len(opts) < min_opts:
                            min_opts = len(opts)
                            best_cell = (r, c, list(opts))

            if best_cell is None:
                return False

            r, c, options = best_cell
            original_grid = [[cell for cell in row] for row in grid]

            for digit in options:
                test_grid = [[cell for cell in row] for row in grid]
                test_grid[r][c] = digit

                result = solve_and_log(test_grid, depth + 1)

                # Encode this decision
                abstract_vec = self.encoder.encode(original_grid, r, c, len(options))

                if result:
                    self.good_decisions.append(abstract_vec)
                else:
                    self.bad_decisions.append(abstract_vec)

                if result:
                    return True

            return False

        grid = [[cell for cell in row] for row in puzzle]
        solve_and_log(grid)

        # Build prototypes
        if self.good_decisions:
            self.good_prototype = bundle(self.good_decisions)
        if self.bad_decisions:
            self.bad_prototype = bundle(self.bad_decisions)

    def score_decision(self, grid: List[List[Optional[int]]], r: int, c: int,
                       num_options: int) -> float:
        """Score a decision point using learned prototypes."""
        if self.good_prototype is None or self.bad_prototype is None:
            return 0.0

        abstract_vec = self.encoder.encode(grid, r, c, num_options)

        sim_good = similarity(abstract_vec, self.good_prototype)
        sim_bad = similarity(abstract_vec, self.bad_prototype)

        # Return difference: positive = more like good, negative = more like bad
        return sim_good - sim_bad


class GuidedSolver:
    """Solve Sudoku with prototype guidance."""

    def __init__(self, learner: PrototypeLearner = None):
        self.learner = learner
        self.backtracks = 0
        self.decisions = 0

    def solve(self, puzzle: List[List[Optional[int]]], use_guidance: bool = True
              ) -> Tuple[bool, List[List[int]]]:
        """
        Solve puzzle, optionally using prototype guidance to order choices.
        """
        self.backtracks = 0
        self.decisions = 0

        grid = [[cell for cell in row] for row in puzzle]

        def solve_recursive(depth=0):
            # Propagate forced moves
            for _ in range(81):
                changed = False
                for r in range(9):
                    for c in range(9):
                        if grid[r][c] is None:
                            available = get_available_digits_9x9(grid, r, c)
                            if len(available) == 0:
                                return False
                            if len(available) == 1:
                                grid[r][c] = list(available)[0]
                                changed = True
                if not changed:
                    break

            empty = count_empty(grid)
            if empty == 0:
                valid, _ = validate_9x9(grid)
                return valid

            # Find MRV cell
            best_cell = None
            min_opts = 10
            for r in range(9):
                for c in range(9):
                    if grid[r][c] is None:
                        opts = get_available_digits_9x9(grid, r, c)
                        if len(opts) < min_opts:
                            min_opts = len(opts)
                            best_cell = (r, c, list(opts))

            if best_cell is None:
                return False

            r, c, options = best_cell
            self.decisions += 1

            # Order options
            if use_guidance and self.learner and len(options) > 1:
                # Score this decision point
                score = self.learner.score_decision(grid, r, c, len(options))

                # If score is positive (more like good), try in standard order
                # If negative (more like bad), we might want to be more careful
                # For now, just use standard order but could randomize based on score
                pass  # Standard order for now

            for digit in options:
                grid[r][c] = digit

                if solve_recursive(depth + 1):
                    return True

                self.backtracks += 1

            grid[r][c] = None
            return False

        success = solve_recursive()
        return success, grid


def experiment_guided_vs_unguided():
    """
    Main experiment: Compare solving with and without guidance.
    """
    print("=" * 70)
    print("EXPERIMENT: Prototype-Guided Solving")
    print("=" * 70)

    encoder = AbstractFeatureEncoder(dimensions=16384)

    # Learn from puzzle 1
    print("\n1. Learning prototypes from PUZZLE 1 (hard)...")
    learner = PrototypeLearner(encoder)
    with Timer() as t:
        learner.learn_from_puzzle(PUZZLE_9x9_HARD)
    print(f"   Good decisions: {len(learner.good_decisions)}")
    print(f"   Bad decisions: {len(learner.bad_decisions)}")
    print(f"   Learning time: {t.elapsed:.2f}s")

    # Test puzzles
    test_puzzles = [
        ("PUZZLE 2", PUZZLE_2),
        ("PUZZLE 3", PUZZLE_3),
        ("PUZZLE 1 (same)", PUZZLE_9x9_HARD),  # Sanity check
    ]

    print("\n2. Testing on multiple puzzles...\n")
    print(f"{'Puzzle':<20} {'Guided Backtracks':<20} {'Unguided Backtracks':<20} {'Reduction'}")
    print("-" * 70)

    for name, puzzle in test_puzzles:
        # Without guidance
        solver_no = GuidedSolver()
        _, _ = solver_no.solve(puzzle, use_guidance=False)
        bt_no = solver_no.backtracks

        # With guidance (using learned prototypes)
        solver_yes = GuidedSolver(learner)
        solved, grid = solver_yes.solve(puzzle, use_guidance=True)
        bt_yes = solver_yes.backtracks

        reduction = (bt_no - bt_yes) / bt_no * 100 if bt_no > 0 else 0

        print(f"{name:<20} {bt_yes:<20} {bt_no:<20} {reduction:+.1f}%")


def experiment_ordering_by_prototype():
    """
    Experiment: Use prototype scores to ORDER digit choices, not just accept/reject.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT: Ordering Choices by Prototype Similarity")
    print("=" * 70)

    encoder = AbstractFeatureEncoder(dimensions=16384)

    # Learn from puzzle 1
    learner = PrototypeLearner(encoder)
    learner.learn_from_puzzle(PUZZLE_9x9_HARD)

    # New idea: for each choice at a decision point, encode the choice itself
    # and see which digit leads to a more "good-like" abstract state

    def solve_with_ordering(puzzle):
        grid = [[cell for cell in row] for row in puzzle]
        backtracks = 0

        def solve_recursive(depth=0):
            nonlocal backtracks

            # Propagate
            for _ in range(81):
                changed = False
                for r in range(9):
                    for c in range(9):
                        if grid[r][c] is None:
                            available = get_available_digits_9x9(grid, r, c)
                            if len(available) == 0:
                                return False
                            if len(available) == 1:
                                grid[r][c] = list(available)[0]
                                changed = True
                if not changed:
                    break

            empty = count_empty(grid)
            if empty == 0:
                valid, _ = validate_9x9(grid)
                return valid

            # Find MRV cell
            best_cell = None
            min_opts = 10
            for r in range(9):
                for c in range(9):
                    if grid[r][c] is None:
                        opts = get_available_digits_9x9(grid, r, c)
                        if len(opts) < min_opts:
                            min_opts = len(opts)
                            best_cell = (r, c, list(opts))

            if best_cell is None:
                return False

            r, c, options = best_cell

            # Score each option by simulating one step and checking abstract features
            scored_options = []
            for digit in options:
                test_grid = [[cell for cell in row] for row in grid]
                test_grid[r][c] = digit

                # Count forced moves after this choice
                forced = 0
                contradiction = False
                for _ in range(81):
                    changed = False
                    for rr in range(9):
                        for cc in range(9):
                            if test_grid[rr][cc] is None:
                                available = get_available_digits_9x9(test_grid, rr, cc)
                                if len(available) == 0:
                                    contradiction = True
                                    break
                                if len(available) == 1:
                                    test_grid[rr][cc] = list(available)[0]
                                    forced += 1
                                    changed = True
                        if contradiction:
                            break
                    if not changed or contradiction:
                        break

                if contradiction:
                    score = -1000  # Bad choice
                else:
                    # More forced moves = more deterministic = potentially good
                    score = forced

                scored_options.append((digit, score))

            # Sort by score (higher first)
            scored_options.sort(key=lambda x: x[1], reverse=True)
            ordered = [d for d, s in scored_options]

            for digit in ordered:
                grid[r][c] = digit

                if solve_recursive(depth + 1):
                    return True

                backtracks += 1

            grid[r][c] = None
            return False

        success = solve_recursive()
        return success, backtracks

    # Compare with standard ordering
    test_puzzles = [
        ("PUZZLE 2", PUZZLE_2),
        ("PUZZLE 3", PUZZLE_3),
    ]

    print(f"\n{'Puzzle':<20} {'Ordered Backtracks':<22} {'Standard Backtracks':<22} {'Reduction'}")
    print("-" * 75)

    for name, puzzle in test_puzzles:
        # Standard
        solver_std = GuidedSolver()
        _, _ = solver_std.solve(puzzle, use_guidance=False)
        bt_std = solver_std.backtracks

        # Ordered by forced moves
        _, bt_ord = solve_with_ordering(puzzle)

        reduction = (bt_std - bt_ord) / bt_std * 100 if bt_std > 0 else 0
        print(f"{name:<20} {bt_ord:<22} {bt_std:<22} {reduction:+.1f}%")


def main():
    experiment_guided_vs_unguided()
    experiment_ordering_by_prototype()

    print("\n" + "=" * 70)
    print("CONCLUSIONS")
    print("=" * 70)


if __name__ == "__main__":
    main()
