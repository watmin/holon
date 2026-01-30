#!/usr/bin/env python3
"""
Approach 19: Opportunistic Guessing with Fast Negation

KEY INSIGHT: We found that contradiction detection:
- 0% at start (no signal)
- 85% when grid is full (strong signal)

STRATEGY: Race ahead on each choice path:
- Choose paths that reach verdict FASTEST
- "Lucky" paths = many forced moves = fast verdict
- "Unlucky" paths = ambiguity = slow verdict
- Fast negation = detect bad paths before they diverge

This exploits the LATE signal by getting there FASTER.
"""

import sys
sys.path.insert(0, "/home/watmin/work/holon")

from typing import List, Optional, Set, Tuple, Dict
from dataclasses import dataclass
import numpy as np
import time

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


@dataclass
class PathStats:
    """Statistics about a choice path."""
    choice: Tuple[int, int, int]  # (row, col, digit)
    forced_moves: int  # How many moves were forced
    total_filled: int  # Total cells filled before verdict
    contradicted: bool  # Did it hit contradiction?
    ambiguity_points: int  # How many times had to guess
    depth_to_verdict: int  # How deep before contradiction or stall


def propagate_forced(grid: List[List[Optional[int]]]) -> Tuple[bool, int]:
    """
    Propagate all forced moves. Returns (success, count).
    """
    count = 0
    changed = True

    while changed:
        changed = False
        for r in range(9):
            for c in range(9):
                if grid[r][c] is not None:
                    continue

                options = get_available_digits_9x9(grid, r, c)

                if not options:
                    return False, count  # Contradiction

                options_list = list(options)
                if len(options_list) == 1:
                    grid[r][c] = options_list[0]
                    count += 1
                    changed = True

    return True, count


def race_to_verdict(grid: List[List[Optional[int]]], row: int, col: int, digit: int,
                    max_depth: int = 15) -> PathStats:
    """
    Race ahead on this choice path and see how fast we reach a verdict.

    Returns PathStats about this path:
    - How many moves were forced (good = more)
    - How deep before contradiction or stall (good = shallower)
    - How many ambiguity points (good = fewer)
    """
    # Make the choice
    test_grid = [[cell for cell in row] for row in grid]
    test_grid[row][col] = digit

    # Propagate forced moves
    success, forced_count = propagate_forced(test_grid)

    if not success:
        # Immediate contradiction - FAST NEGATION
        return PathStats(
            choice=(row, col, digit),
            forced_moves=forced_count,
            total_filled=forced_count + 1,
            contradicted=True,
            ambiguity_points=0,
            depth_to_verdict=0
        )

    # Continue racing - simulate further choices
    depth = 0
    ambiguity_points = 0

    while depth < max_depth:
        # Find cell with fewest options (MRV)
        best_cell = None
        best_count = 10

        for r in range(9):
            for c in range(9):
                if test_grid[r][c] is None:
                    options = get_available_digits_9x9(test_grid, r, c)
                    if len(options) < best_count:
                        best_count = len(options)
                        best_cell = (r, c, options)

        if best_cell is None:
            # Solved!
            return PathStats(
                choice=(row, col, digit),
                forced_moves=forced_count,
                total_filled=81 - count_empty(test_grid),
                contradicted=False,
                ambiguity_points=ambiguity_points,
                depth_to_verdict=depth
            )

        r, c, options_set = best_cell
        options = list(options_set)

        if not options:
            # Contradiction at depth
            return PathStats(
                choice=(row, col, digit),
                forced_moves=forced_count,
                total_filled=81 - count_empty(test_grid),
                contradicted=True,
                ambiguity_points=ambiguity_points,
                depth_to_verdict=depth
            )

        if len(options) == 1:
            # Forced move - keep going
            test_grid[r][c] = options[0]
            forced_count += 1
        else:
            # Ambiguity - pick first and continue
            ambiguity_points += 1
            test_grid[r][c] = options[0]

        # Propagate
        success, new_forced = propagate_forced(test_grid)
        forced_count += new_forced

        if not success:
            return PathStats(
                choice=(row, col, digit),
                forced_moves=forced_count,
                total_filled=81 - count_empty(test_grid),
                contradicted=True,
                ambiguity_points=ambiguity_points,
                depth_to_verdict=depth
            )

        depth += 1

    # Reached max depth without verdict
    return PathStats(
        choice=(row, col, digit),
        forced_moves=forced_count,
        total_filled=81 - count_empty(test_grid),
        contradicted=False,
        ambiguity_points=ambiguity_points,
        depth_to_verdict=depth
    )


def score_path_opportunistically(stats: PathStats) -> float:
    """
    Score a path for opportunistic selection.

    KEY INSIGHT: Contradiction during racing is NOT definitive rejection!
    The contradiction comes from arbitrary choices at ambiguity points.

    What we ACTUALLY learn:
    - More forced moves = more deterministic path = MORE PROMISING
    - Deeper contradiction = more "runway" before trouble
    - We're measuring PATH QUALITY, not correctness

    REVISED SCORING:
    1. Forced moves is the PRIMARY signal (lucky chains)
    2. Depth to verdict is secondary (more runway = more potential)
    3. Contradiction is NOT penalized (it's from arbitrary ambiguity choices)
    """
    # Primary: forced moves (lucky chain length)
    chain_score = stats.forced_moves * 10

    # Secondary: total progress (how far we got)
    progress_score = stats.total_filled

    # Tertiary: ratio of forced to ambiguous (determinism)
    if stats.ambiguity_points == 0:
        determinism_bonus = 50  # Fully deterministic is great
    else:
        determinism_bonus = stats.forced_moves / stats.ambiguity_points

    return chain_score + progress_score + determinism_bonus


class OpportunisticSolver:
    """
    Solver that uses opportunistic path racing.

    Strategy:
    1. At each decision point, RACE ahead on all options
    2. Score paths by "luck" (forced moves) and "speed" (fast verdict)
    3. Try paths in order of luck score
    4. Fast negation: skip paths that contradict quickly
    """

    def __init__(self, race_depth: int = 15, verbose: bool = True):
        self.race_depth = race_depth
        self.verbose = verbose
        self.backtracks = 0
        self.fast_rejections = 0

    def solve(self, puzzle: List[List[Optional[int]]]) -> Tuple[bool, List[List[int]]]:
        grid = [[cell for cell in row] for row in puzzle]

        # Initial propagation
        success, _ = propagate_forced(grid)
        if not success:
            return False, [[0]*9 for _ in range(9)]

        result = self.solve_recursive(grid, depth=0)

        if result:
            return True, result
        else:
            return False, [[0]*9 for _ in range(9)]

    def solve_recursive(self, grid: List[List[Optional[int]]], depth: int) -> Optional[List[List[int]]]:
        # Check if solved
        if count_empty(grid) == 0:
            return grid

        # Find cell with fewest options
        best_cell = None
        best_count = 10

        for r in range(9):
            for c in range(9):
                if grid[r][c] is None:
                    options = get_available_digits_9x9(grid, r, c)
                    if len(options) == 0:
                        return None  # Contradiction
                    if len(options) < best_count:
                        best_count = len(options)
                        best_cell = (r, c, options)

        if best_cell is None:
            return grid

        r, c, options_set = best_cell
        options = list(options_set)

        # RACE ahead on each option
        path_stats = []
        for digit in options:
            stats = race_to_verdict(grid, r, c, digit, self.race_depth)
            path_stats.append(stats)

        # Score and sort by opportunistic measure
        scored = [(score_path_opportunistically(s), i, s) for i, s in enumerate(path_stats)]
        scored.sort(key=lambda x: x[0], reverse=True)  # Best first

        if self.verbose and depth < 3:
            print(f"  Depth {depth}: Cell ({r},{c}) options: {options}")
            for score, _, stats in scored:
                status = "CONTRADICT" if stats.contradicted else "viable"
                print(f"    Digit {stats.choice[2]}: score={score:.1f}, "
                      f"forced={stats.forced_moves}, ambig={stats.ambiguity_points}, "
                      f"status={status}")

        # Try in order of luck (most forced moves first)
        for score, _, stats in scored:

            digit = stats.choice[2]

            test_grid = [[cell for cell in row] for row in grid]
            test_grid[r][c] = digit

            success, _ = propagate_forced(test_grid)
            if not success:
                self.fast_rejections += 1
                continue

            result = self.solve_recursive(test_grid, depth + 1)
            if result:
                return result

            self.backtracks += 1

        return None


class HybridSolver:
    """
    HYBRID: Combines simulation rejection with chain-length ordering.

    Key insight:
    - Simulation rejection (approach 10) is PROVEN to reduce backtracks
    - Chain length might help ORDER the surviving options
    """

    def __init__(self, sim_depth: int = 10, verbose: bool = True):
        self.sim_depth = sim_depth
        self.verbose = verbose
        self.backtracks = 0
        self.simulation_rejections = 0

    def solve(self, puzzle: List[List[Optional[int]]]) -> Tuple[bool, List[List[int]]]:
        grid = [[cell for cell in row] for row in puzzle]

        success, _ = propagate_forced(grid)
        if not success:
            return False, [[0]*9 for _ in range(9)]

        result = self.solve_recursive(grid, depth=0)

        if result:
            return True, result
        else:
            return False, [[0]*9 for _ in range(9)]

    def simulate_survives(self, grid: List[List[Optional[int]]],
                          row: int, col: int, digit: int,
                          depth: int = 10) -> Tuple[bool, int]:
        """
        Simulate ahead and return (survives, forced_count).
        """
        test_grid = [[cell for cell in r] for r in grid]
        test_grid[row][col] = digit

        forced_count = 0

        for _ in range(depth):
            success, count = propagate_forced(test_grid)
            forced_count += count

            if not success:
                return False, forced_count

            # Find next cell with fewest options
            best = None
            best_count = 10

            for r in range(9):
                for c in range(9):
                    if test_grid[r][c] is None:
                        opts = get_available_digits_9x9(test_grid, r, c)
                        if not opts:
                            return False, forced_count
                        if len(opts) < best_count:
                            best_count = len(opts)
                            best = (r, c, list(opts))

            if best is None:
                return True, forced_count  # Solved

            if best_count == 1:
                # Forced move
                r, c, opts = best
                test_grid[r][c] = opts[0]
                forced_count += 1
            else:
                # Ambiguity - stop simulation
                return True, forced_count

        return True, forced_count

    def solve_recursive(self, grid: List[List[Optional[int]]], depth: int) -> Optional[List[List[int]]]:
        if count_empty(grid) == 0:
            return grid

        # Find cell with fewest options
        best_cell = None
        best_count = 10

        for r in range(9):
            for c in range(9):
                if grid[r][c] is None:
                    options = get_available_digits_9x9(grid, r, c)
                    if len(options) == 0:
                        return None
                    if len(options) < best_count:
                        best_count = len(options)
                        best_cell = (r, c, list(options))

        if best_cell is None:
            return grid

        r, c, options = best_cell

        # Simulate each option
        viable = []
        for digit in options:
            survives, forced_count = self.simulate_survives(grid, r, c, digit, self.sim_depth)
            if survives:
                viable.append((forced_count, digit))
            else:
                self.simulation_rejections += 1

        # Sort by forced_count (DESCENDING - more forced = better chain)
        viable.sort(reverse=True)

        if not viable:
            # All rejected by simulation - try all anyway
            viable = [(0, d) for d in options]

        for forced_count, digit in viable:
            test_grid = [[cell for cell in row] for row in grid]
            test_grid[r][c] = digit

            success, _ = propagate_forced(test_grid)
            if not success:
                continue

            result = self.solve_recursive(test_grid, depth + 1)
            if result:
                return result

            self.backtracks += 1

        return None


def compare_solvers():
    """Compare opportunistic vs standard backtracking."""
    print("=" * 70)
    print("OPPORTUNISTIC SOLVER COMPARISON")
    print("=" * 70)

    puzzles = {
        "hard": PUZZLE_9x9_HARD,
        "easy": [
            [5,3,None,None,7,None,None,None,None],
            [6,None,None,1,9,5,None,None,None],
            [None,9,8,None,None,None,None,6,None],
            [8,None,None,None,6,None,None,None,3],
            [4,None,None,8,None,3,None,None,1],
            [7,None,None,None,2,None,None,None,6],
            [None,6,None,None,None,None,2,8,None],
            [None,None,None,4,1,9,None,None,5],
            [None,None,None,None,8,None,None,7,9],
        ],
    }

    for name, puzzle in puzzles.items():
        print(f"\n{'='*50}")
        print(f"PUZZLE: {name}")
        print(f"{'='*50}")

        # Standard backtracking
        from approach_10_global import SimulationGuidedSolver

        standard = SimulationGuidedSolver(verbose=False)
        start = time.time()
        success1, result1 = standard.solve(puzzle)
        time1 = time.time() - start

        # Opportunistic
        opportunistic = OpportunisticSolver(race_depth=15, verbose=False)
        start = time.time()
        success2, result2 = opportunistic.solve(puzzle)
        time2 = time.time() - start

        # Hybrid
        hybrid = HybridSolver(sim_depth=10, verbose=False)
        start = time.time()
        success3, result3 = hybrid.solve(puzzle)
        time3 = time.time() - start

        print(f"\nStandard (simulation-guided):")
        print(f"  Solved: {success1}")
        print(f"  Backtracks: {standard.backtracks}")
        print(f"  Time: {time1:.3f}s")

        print(f"\nOpportunistic (race ahead):")
        print(f"  Solved: {success2}")
        print(f"  Backtracks: {opportunistic.backtracks}")
        print(f"  Fast rejections: {opportunistic.fast_rejections}")
        print(f"  Time: {time2:.3f}s")
        if success2:
            print(f"  Valid: {validate_9x9(result2)}")

        print(f"\nHYBRID (simulate + order by chain length):")
        print(f"  Solved: {success3}")
        print(f"  Backtracks: {hybrid.backtracks}")
        print(f"  Simulation rejections: {hybrid.simulation_rejections}")
        print(f"  Time: {time3:.3f}s")
        if success3:
            print(f"  Valid: {validate_9x9(result3)}")


def analyze_path_racing():
    """Analyze what path racing reveals."""
    print("=" * 70)
    print("PATH RACING ANALYSIS")
    print("=" * 70)

    grid = [[cell for cell in row] for row in PUZZLE_9x9_HARD]
    propagate_forced(grid)

    print(f"\nInitial grid after propagation:")
    print(f"Empty cells: {count_empty(grid)}")

    # Find first decision point
    for r in range(9):
        for c in range(9):
            if grid[r][c] is None:
                options = get_available_digits_9x9(grid, r, c)
                if len(options) > 1:
                    print(f"\nFirst decision point: ({r},{c}) with options {options}")

                    for digit in options:
                        stats = race_to_verdict(grid, r, c, digit, max_depth=20)
                        score = score_path_opportunistically(stats)

                        print(f"\n  Digit {digit}:")
                        print(f"    Forced moves: {stats.forced_moves}")
                        print(f"    Total filled: {stats.total_filled}")
                        print(f"    Contradicted: {stats.contradicted}")
                        print(f"    Ambiguity points: {stats.ambiguity_points}")
                        print(f"    Depth to verdict: {stats.depth_to_verdict}")
                        print(f"    LUCK SCORE: {score:.1f}")

                    return

    print("No decision points found (all forced)")


def main():
    analyze_path_racing()
    print("\n")
    compare_solvers()


if __name__ == "__main__":
    main()
