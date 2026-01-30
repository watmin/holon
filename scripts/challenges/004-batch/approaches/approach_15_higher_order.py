#!/usr/bin/env python3
"""
Approach 15: Higher-Order Binding

THE ATTACK:
Standard binding encodes FACTS: bind(position, digit)
Higher-order binding encodes IMPLICATIONS: bind(choice, consequences)

THE HYPOTHESIS:
If we encode the chain of consequences that follow from a choice,
the CORRECT choice will have a geometrically distinguishable signature.

WHAT WE ENCODE:
1. The choice itself: bind(pos, digit)
2. Immediate consequences: what peers can no longer be
3. Forced moves: what cells become determined
4. Propagation chain: cascade of forced moves
5. Contradiction potential: does the chain hit a dead end?

THE QUESTION:
Can we distinguish correct from incorrect choices by their chain structure?
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
    effective_dimensionality,
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


def get_peers(row: int, col: int) -> Set[Tuple[int, int]]:
    """Get all cells that share a constraint with (row, col)."""
    peers = set()
    # Row
    for c in range(9):
        if c != col:
            peers.add((row, c))
    # Column
    for r in range(9):
        if r != row:
            peers.add((r, col))
    # Block
    br, bc = (row // 3) * 3, (col // 3) * 3
    for r in range(br, br + 3):
        for c in range(bc, bc + 3):
            if (r, c) != (row, col):
                peers.add((r, c))
    return peers


class HigherOrderEncoder:
    """
    Encode choices with their consequences using higher-order binding.
    """

    def __init__(self, dimensions: int = 16384):
        self.dimensions = dimensions
        self.client = create_client(dimensions=dimensions)
        self.cache = VectorCache(self.client)

        self.digits = list(range(1, 10))
        self.digit_vectors = {d: self.cache.get_digit_vector(d) for d in self.digits}
        self.pos_vectors = {}
        for r in range(9):
            for c in range(9):
                self.pos_vectors[(r, c)] = self.cache.get_position_vector(r, c)

        # Special vectors for encoding consequences
        self.FORCES = self._random_vector("FORCES")  # "this forces that"
        self.ELIMINATES = self._random_vector("ELIMINATES")  # "this removes option"
        self.CONTRADICTION = self._random_vector("CONTRADICTION")  # "dead end"
        self.VALID = self._random_vector("VALID")  # "leads to valid state"

    def _random_vector(self, seed: str) -> np.ndarray:
        np.random.seed(hash(seed) % (2**32))
        return np.random.choice([-1.0, 1.0], size=self.dimensions)

    def copy_grid(self, grid):
        return [[cell for cell in row] for row in grid]

    # =========================================================================
    # ENCODING LEVEL 1: Immediate consequences
    # =========================================================================

    def encode_immediate(self, grid: List[List[Optional[int]]],
                         row: int, col: int, digit: int) -> np.ndarray:
        """
        Encode choice + immediate consequences (what peers lose this option).
        """
        # The choice itself: pos ⊙ digit
        choice = bind(self.pos_vectors[(row, col)], self.digit_vectors[digit])

        # Immediate eliminations: for each peer, encode that 'digit' is eliminated
        eliminations = []
        for (pr, pc) in get_peers(row, col):
            if grid[pr][pc] is None:
                # peer loses 'digit' as option
                elim = bind(self.pos_vectors[(pr, pc)], self.digit_vectors[digit])
                elim = bind(elim, self.ELIMINATES)
                eliminations.append(elim)

        if eliminations:
            consequence = bundle(eliminations)
            return bind(choice, consequence)
        return choice

    # =========================================================================
    # ENCODING LEVEL 2: Forced moves chain
    # =========================================================================

    def find_forced_moves(self, grid: List[List[Optional[int]]]) -> List[Tuple[int, int, int]]:
        """Find cells with exactly one option (naked singles)."""
        forced = []
        for r in range(9):
            for c in range(9):
                if grid[r][c] is None:
                    available = get_available_digits_9x9(grid, r, c)
                    if len(available) == 1:
                        forced.append((r, c, list(available)[0]))
                    elif len(available) == 0:
                        forced.append((r, c, -1))  # Contradiction marker
        return forced

    def encode_chain(self, grid: List[List[Optional[int]]],
                     row: int, col: int, digit: int,
                     max_depth: int = 5) -> Tuple[np.ndarray, Dict]:
        """
        Encode the choice and the chain of forced moves that follow.

        Returns:
        - chain_vec: The encoded chain
        - info: Dictionary with chain statistics
        """
        # Start with the choice
        chain_vec = bind(self.pos_vectors[(row, col)], self.digit_vectors[digit])

        test_grid = self.copy_grid(grid)
        test_grid[row][col] = digit

        info = {
            'depth': 0,
            'forced_moves': 0,
            'contradiction': False,
            'cells_filled': 1
        }

        for depth in range(max_depth):
            forced = self.find_forced_moves(test_grid)

            if not forced:
                break  # No more forced moves

            for (r, c, d) in forced:
                if d == -1:
                    # Contradiction!
                    info['contradiction'] = True
                    chain_vec = bind(chain_vec, self.CONTRADICTION)
                    return chain_vec, info

                # Encode this forced move
                forced_vec = bind(self.pos_vectors[(r, c)], self.digit_vectors[d])
                forced_vec = bind(forced_vec, self.FORCES)
                chain_vec = bind(chain_vec, forced_vec)

                test_grid[r][c] = d
                info['forced_moves'] += 1
                info['cells_filled'] += 1

            info['depth'] = depth + 1

        # If we got here without contradiction, mark as valid (so far)
        chain_vec = bind(chain_vec, self.VALID)

        return chain_vec, info

    # =========================================================================
    # ENCODING LEVEL 3: Constraint tightening
    # =========================================================================

    def compute_tightening(self, grid: List[List[Optional[int]]],
                           row: int, col: int, digit: int) -> Dict:
        """
        Measure how much this choice constrains the remaining puzzle.
        """
        # Before
        before_options = 0
        for r in range(9):
            for c in range(9):
                if grid[r][c] is None:
                    before_options += len(get_available_digits_9x9(grid, r, c))

        # After
        test_grid = self.copy_grid(grid)
        test_grid[row][col] = digit

        after_options = 0
        min_options = 10
        cells_with_zero = 0

        for r in range(9):
            for c in range(9):
                if test_grid[r][c] is None:
                    opts = len(get_available_digits_9x9(test_grid, r, c))
                    after_options += opts
                    min_options = min(min_options, opts)
                    if opts == 0:
                        cells_with_zero += 1

        return {
            'before_options': before_options,
            'after_options': after_options,
            'options_removed': before_options - after_options,
            'min_options': min_options,
            'cells_with_zero': cells_with_zero,
            'causes_contradiction': cells_with_zero > 0
        }

    # =========================================================================
    # SCORING: Compare choices via their chain structure
    # =========================================================================

    def score_choice(self, grid: List[List[Optional[int]]],
                     row: int, col: int, digit: int,
                     method: str = "chain") -> Tuple[float, Dict]:
        """
        Score a choice by its higher-order properties.
        """
        if method == "immediate":
            vec = self.encode_immediate(grid, row, col, digit)
            return np.linalg.norm(vec), {'method': 'immediate'}

        elif method == "chain":
            chain_vec, info = self.encode_chain(grid, row, col, digit)

            # Score: penalize contradiction, reward depth
            if info['contradiction']:
                score = -1000
            else:
                score = info['forced_moves'] * 10 + info['depth']

            return score, info

        elif method == "tightening":
            tightening = self.compute_tightening(grid, row, col, digit)

            if tightening['causes_contradiction']:
                score = -1000
            else:
                # More tightening = more constrained = potentially good
                # But also riskier...
                score = tightening['options_removed'] + tightening['min_options'] * 5

            return score, tightening

        elif method == "combined":
            # Combine chain and tightening
            chain_vec, chain_info = self.encode_chain(grid, row, col, digit)
            tightening = self.compute_tightening(grid, row, col, digit)

            if chain_info['contradiction'] or tightening['causes_contradiction']:
                score = -1000
            else:
                score = (
                    chain_info['forced_moves'] * 10 +
                    chain_info['depth'] * 5 +
                    tightening['min_options'] * 2
                )

            return score, {**chain_info, **tightening}

        else:
            raise ValueError(f"Unknown method: {method}")


class HigherOrderSolver:
    """
    Solve using higher-order binding to score choices.
    """

    def __init__(self, dimensions: int = 16384, verbose: bool = True):
        self.encoder = HigherOrderEncoder(dimensions)
        self.verbose = verbose

    def solve(self, puzzle: List[List[Optional[int]]],
              method: str = "chain") -> Tuple[bool, List[List[int]], Dict]:
        """
        Solve by choosing digits with best higher-order scores.
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"HIGHER-ORDER SOLVER (method={method})")
            print(f"{'='*60}")

        grid = [[cell for cell in row] for row in puzzle]
        stats = {
            'correct_predictions': 0,
            'total_predictions': 0,
            'contradictions_avoided': 0
        }

        max_iters = 81

        for iteration in range(max_iters):
            # Find cells with options
            candidates = []
            for r in range(9):
                for c in range(9):
                    if grid[r][c] is None:
                        available = get_available_digits_9x9(grid, r, c)
                        if len(available) == 0:
                            if self.verbose:
                                print(f"  Contradiction at ({r},{c})")
                            return False, grid, stats
                        if len(available) == 1:
                            # Forced - no choice needed
                            grid[r][c] = list(available)[0]
                            continue
                        candidates.append((r, c, available))

            if not candidates:
                break  # All filled

            # Pick cell with fewest options (MRV)
            candidates.sort(key=lambda x: len(x[2]))
            r, c, available = candidates[0]

            # Score each option using higher-order encoding
            scores = {}
            infos = {}
            for d in available:
                score, info = self.encoder.score_choice(grid, r, c, d, method)
                scores[d] = score
                infos[d] = info

            # Pick best score
            best_digit = max(scores, key=lambda x: scores[x])
            best_score = scores[best_digit]

            # Track if we avoided a contradiction
            contradictions = sum(1 for d in available if scores[d] < -500)
            if contradictions > 0:
                stats['contradictions_avoided'] += 1

            # Track accuracy (if we know the solution)
            correct = SOLUTION_9x9_HARD[r][c]
            stats['total_predictions'] += 1
            if best_digit == correct:
                stats['correct_predictions'] += 1

            if self.verbose and iteration < 15:
                print(f"  ({r},{c}): {sorted(available)} → {best_digit} "
                      f"(score={best_score:.0f}, correct={correct})")

            grid[r][c] = best_digit

        valid, msg = validate_9x9(grid)

        if self.verbose:
            print(f"\nStats:")
            print(f"  Correct predictions: {stats['correct_predictions']}/{stats['total_predictions']}")
            print(f"  Contradictions avoided: {stats['contradictions_avoided']}")
            print(f"  Valid: {valid}")

        return valid, grid, stats


def test_chain_distinguishing():
    """
    Test: Can chain encoding distinguish correct from incorrect choices?
    """
    print("=" * 70)
    print("TEST: Can higher-order chains distinguish correct choices?")
    print("=" * 70)

    encoder = HigherOrderEncoder(dimensions=16384)

    # Find decision points
    decision_points = []
    for r in range(9):
        for c in range(9):
            if PUZZLE_9x9_HARD[r][c] is None:
                available = get_available_digits_9x9(PUZZLE_9x9_HARD, r, c)
                if len(available) > 1:
                    correct = SOLUTION_9x9_HARD[r][c]
                    decision_points.append((r, c, available, correct))

    print(f"\nFound {len(decision_points)} decision points\n")

    # Test each method
    methods = ['chain', 'tightening', 'combined']

    for method in methods:
        correct_wins = 0
        total = 0

        for r, c, available, correct in decision_points[:20]:
            scores = {}
            for d in available:
                score, info = encoder.score_choice(PUZZLE_9x9_HARD, r, c, d, method)
                scores[d] = score

            best = max(scores, key=lambda x: scores[x])
            if best == correct:
                correct_wins += 1
            total += 1

        pct = 100 * correct_wins / total if total > 0 else 0
        print(f"Method '{method}': {correct_wins}/{total} = {pct:.1f}% correct")


def test_solver():
    """Test the full solver."""
    print("\n" + "=" * 70)
    print("TEST: Higher-Order Solver")
    print("=" * 70)

    print_grid_9x9(PUZZLE_9x9_HARD)

    for method in ['chain', 'tightening', 'combined']:
        print(f"\n--- Method: {method} ---")
        solver = HigherOrderSolver(verbose=False)
        with Timer() as timer:
            valid, grid, stats = solver.solve(PUZZLE_9x9_HARD, method=method)

        cells = 81 - count_empty(grid)
        accuracy = 100 * stats['correct_predictions'] / stats['total_predictions']
        print(f"  Cells filled: {cells}/81, Valid: {valid}")
        print(f"  Prediction accuracy: {accuracy:.1f}%")
        print(f"  Contradictions avoided: {stats['contradictions_avoided']}")
        print(f"  Time: {timer.elapsed:.2f}s")

        if valid:
            print("\n✓✓ SOLVED!")
            print_grid_9x9(grid)
            return True

    return False


def simulate_until_end(grid, row, col, digit, max_steps=100):
    """
    Simulate placing digit and following forced moves until:
    1. Contradiction (return 'contradiction', steps)
    2. No more forced moves (return 'stuck', steps)
    3. Solved (return 'solved', steps)
    """
    test_grid = [[cell for cell in r] for r in grid]
    test_grid[row][col] = digit
    steps = 1

    for _ in range(max_steps):
        # Find forced moves
        forced = []
        has_contradiction = False

        for r in range(9):
            for c in range(9):
                if test_grid[r][c] is None:
                    available = get_available_digits_9x9(test_grid, r, c)
                    if len(available) == 0:
                        has_contradiction = True
                        break
                    if len(available) == 1:
                        forced.append((r, c, list(available)[0]))
            if has_contradiction:
                break

        if has_contradiction:
            return 'contradiction', steps

        if not forced:
            # Check if solved
            empty = sum(1 for r in range(9) for c in range(9) if test_grid[r][c] is None)
            if empty == 0:
                return 'solved', steps
            return 'stuck', steps

        # Apply forced moves
        for (r, c, d) in forced:
            test_grid[r][c] = d
            steps += 1

    return 'timeout', steps


def analyze_deep():
    """
    DEEP ANALYSIS: Simulate until contradiction or stuck.
    """
    print("\n" + "=" * 70)
    print("DEEP ANALYSIS: Simulate until end for each choice")
    print("=" * 70)

    correct_leads_to = {'contradiction': 0, 'stuck': 0, 'solved': 0}
    wrong_leads_to = {'contradiction': 0, 'stuck': 0, 'solved': 0}

    decision_points = []
    for r in range(9):
        for c in range(9):
            if PUZZLE_9x9_HARD[r][c] is None:
                available = get_available_digits_9x9(PUZZLE_9x9_HARD, r, c)
                if len(available) >= 2:
                    correct = SOLUTION_9x9_HARD[r][c]
                    decision_points.append((r, c, available, correct))

    print(f"\nAnalyzing {len(decision_points)} decision points...")
    print("\nFirst 10 detailed results:\n")

    for i, (r, c, available, correct) in enumerate(decision_points):
        wrong_options = [d for d in available if d != correct]

        # Simulate correct
        outcome_correct, steps_correct = simulate_until_end(
            PUZZLE_9x9_HARD, r, c, correct
        )
        correct_leads_to[outcome_correct] += 1

        # Simulate wrong (first wrong option)
        outcome_wrong, steps_wrong = simulate_until_end(
            PUZZLE_9x9_HARD, r, c, wrong_options[0]
        )
        wrong_leads_to[outcome_wrong] += 1

        if i < 10:
            print(f"({r},{c}) options={sorted(available)}, correct={correct}")
            print(f"  Correct → {outcome_correct} in {steps_correct} steps")
            print(f"  Wrong   → {outcome_wrong} in {steps_wrong} steps")

            if outcome_correct != outcome_wrong:
                print(f"  *** DISTINGUISHABLE! ***")
            print()

    print("\n" + "=" * 70)
    print("SUMMARY: Where do correct vs wrong choices lead?")
    print("=" * 70)
    print(f"\nCorrect choices lead to:")
    for outcome, count in correct_leads_to.items():
        print(f"  {outcome}: {count}")

    print(f"\nWrong choices lead to:")
    for outcome, count in wrong_leads_to.items():
        print(f"  {outcome}: {count}")

    # Key question: How often does wrong lead to contradiction where correct doesn't?
    print("\n" + "=" * 70)
    print("KEY QUESTION: Can we detect wrong choices via contradiction?")
    print("=" * 70)

    detectable = 0
    for r, c, available, correct in decision_points:
        outcome_correct, _ = simulate_until_end(PUZZLE_9x9_HARD, r, c, correct)

        for wrong in available:
            if wrong == correct:
                continue
            outcome_wrong, _ = simulate_until_end(PUZZLE_9x9_HARD, r, c, wrong)

            if outcome_wrong == 'contradiction' and outcome_correct != 'contradiction':
                detectable += 1

    total_wrong = sum(len(avail) - 1 for _, _, avail, _ in decision_points)
    print(f"\nWrong choices that lead to contradiction: {detectable}/{total_wrong}")
    print(f"Detection rate: {100 * detectable / total_wrong:.1f}%")


def analyze_chain_structure():
    """
    Analyze: What does the chain structure look like for correct vs incorrect?
    """
    print("\n" + "=" * 70)
    print("ANALYSIS: Chain structure of correct vs incorrect choices")
    print("=" * 70)

    encoder = HigherOrderEncoder(dimensions=16384)

    # Find a decision point
    for r in range(9):
        for c in range(9):
            if PUZZLE_9x9_HARD[r][c] is None:
                available = get_available_digits_9x9(PUZZLE_9x9_HARD, r, c)
                if len(available) >= 2:
                    correct = SOLUTION_9x9_HARD[r][c]
                    wrong = [d for d in available if d != correct][0]

                    print(f"\nDecision point: ({r},{c})")
                    print(f"  Available: {available}")
                    print(f"  Correct: {correct}, Wrong: {wrong}")

                    # Analyze correct choice
                    print(f"\n  CORRECT ({correct}):")
                    _, info_correct = encoder.encode_chain(PUZZLE_9x9_HARD, r, c, correct, max_depth=10)
                    print(f"    Depth: {info_correct['depth']}")
                    print(f"    Forced moves: {info_correct['forced_moves']}")
                    print(f"    Cells filled: {info_correct['cells_filled']}")
                    print(f"    Contradiction: {info_correct['contradiction']}")

                    tight_correct = encoder.compute_tightening(PUZZLE_9x9_HARD, r, c, correct)
                    print(f"    Options removed: {tight_correct['options_removed']}")
                    print(f"    Min options remaining: {tight_correct['min_options']}")

                    # Analyze wrong choice
                    print(f"\n  WRONG ({wrong}):")
                    _, info_wrong = encoder.encode_chain(PUZZLE_9x9_HARD, r, c, wrong, max_depth=10)
                    print(f"    Depth: {info_wrong['depth']}")
                    print(f"    Forced moves: {info_wrong['forced_moves']}")
                    print(f"    Cells filled: {info_wrong['cells_filled']}")
                    print(f"    Contradiction: {info_wrong['contradiction']}")

                    tight_wrong = encoder.compute_tightening(PUZZLE_9x9_HARD, r, c, wrong)
                    print(f"    Options removed: {tight_wrong['options_removed']}")
                    print(f"    Min options remaining: {tight_wrong['min_options']}")

                    # Compare
                    print(f"\n  COMPARISON:")
                    if info_correct['contradiction'] and not info_wrong['contradiction']:
                        print("    ✗ Correct leads to contradiction, wrong doesn't!")
                    elif info_wrong['contradiction'] and not info_correct['contradiction']:
                        print("    ✓ Wrong leads to contradiction, correct doesn't!")
                    elif info_correct['forced_moves'] > info_wrong['forced_moves']:
                        print("    ~ Correct forces more moves")
                    elif info_wrong['forced_moves'] > info_correct['forced_moves']:
                        print("    ~ Wrong forces more moves")
                    else:
                        print("    ~ Chains are similar")

                    # Only analyze first few
                    if r > 2:
                        break
        if r > 2:
            break


def analyze_at_depth():
    """
    Key insight: Early choices don't cause contradiction.
    What if we're further along? Do choices become distinguishable?
    """
    print("\n" + "=" * 70)
    print("ANALYSIS: Do choices become distinguishable later in solving?")
    print("=" * 70)

    # Fill some cells correctly first
    partial_grid = [[cell for cell in row] for row in PUZZLE_9x9_HARD]

    # Add some correct values
    cells_to_add = [
        (0, 0, 5), (0, 1, 8), (0, 2, 1), (0, 4, 7), (0, 5, 2),
        (1, 1, 9), (1, 2, 2), (1, 3, 8), (1, 4, 4),
        (2, 0, 3), (2, 1, 6), (2, 2, 4), (2, 3, 5),
    ]

    for r, c, d in cells_to_add:
        partial_grid[r][c] = d

    print(f"\nPartially filled grid ({len(cells_to_add)} cells added correctly):")
    print_grid_9x9(partial_grid)

    # Now find decision points and analyze
    decision_points = []
    for r in range(9):
        for c in range(9):
            if partial_grid[r][c] is None:
                available = get_available_digits_9x9(partial_grid, r, c)
                if len(available) >= 2:
                    correct = SOLUTION_9x9_HARD[r][c]
                    decision_points.append((r, c, available, correct))

    print(f"\nDecision points remaining: {len(decision_points)}")

    # Test detection rate
    detectable = 0
    total_wrong = 0

    for r, c, available, correct in decision_points:
        outcome_correct, steps_correct = simulate_until_end(partial_grid, r, c, correct)

        for wrong in available:
            if wrong == correct:
                continue
            total_wrong += 1
            outcome_wrong, steps_wrong = simulate_until_end(partial_grid, r, c, wrong)

            if outcome_wrong == 'contradiction' and outcome_correct != 'contradiction':
                detectable += 1
                print(f"  DETECTABLE: ({r},{c}) correct={correct} OK, wrong={wrong} CONTRADICTION")

    print(f"\nWrong choices detectable: {detectable}/{total_wrong}")
    print(f"Detection rate: {100 * detectable / total_wrong:.1f}%" if total_wrong > 0 else "No data")

    # Try with more cells filled
    print("\n" + "-" * 50)
    print("Adding more cells...")

    more_cells = [
        (0, 7, 3), (0, 8, 9),
        (1, 7, 5), (1, 8, 1),
        (2, 6, 7), (2, 8, 2),
        (3, 0, 4), (3, 1, 3), (3, 2, 8), (3, 3, 9), (3, 4, 5), (3, 5, 7),
    ]

    for r, c, d in more_cells:
        partial_grid[r][c] = d

    print(f"\nAfter adding {len(more_cells)} more cells:")
    print_grid_9x9(partial_grid)

    # Re-analyze
    decision_points = []
    for r in range(9):
        for c in range(9):
            if partial_grid[r][c] is None:
                available = get_available_digits_9x9(partial_grid, r, c)
                if len(available) >= 2:
                    correct = SOLUTION_9x9_HARD[r][c]
                    decision_points.append((r, c, available, correct))

    print(f"\nDecision points: {len(decision_points)}")

    detectable = 0
    total_wrong = 0

    for r, c, available, correct in decision_points:
        outcome_correct, steps_correct = simulate_until_end(partial_grid, r, c, correct)

        for wrong in available:
            if wrong == correct:
                continue
            total_wrong += 1
            outcome_wrong, steps_wrong = simulate_until_end(partial_grid, r, c, wrong)

            if outcome_wrong == 'contradiction' and outcome_correct != 'contradiction':
                detectable += 1
                print(f"  DETECTABLE: ({r},{c}) correct={correct} OK, wrong={wrong} CONTRADICTION")

    print(f"\nDetection rate: {100 * detectable / total_wrong:.1f}%" if total_wrong > 0 else "No wrong choices")


def main():
    print("=" * 70)
    print("APPROACH 15: HIGHER-ORDER BINDING")
    print("=" * 70)
    print("\nEncoding choices with their consequences, not just facts.")
    print("The question: Can chain structure distinguish correct choices?\n")

    # The key analysis: simulate until we hit contradiction or get stuck
    analyze_deep()

    # Test at different depths
    analyze_at_depth()

    # Also test the solver
    test_solver()


if __name__ == "__main__":
    main()
