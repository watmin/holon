#!/usr/bin/env python3
"""
Approach 21: Constraint Landscape Encoding

Key insight from opportunistic approach:
- Choices that force more moves are better
- This means the "constraint landscape" changes MORE after good choices

Encode the CHANGE in constraint landscape:
- Before choice: total options across all cells
- After choice: total options (after propagation)
- The DELTA tells us how "constraining" the choice is

Hypothesis: Good choices cause RAPID constraint collapse
Bad choices leave the landscape "flat"
"""

import sys
sys.path.insert(0, "/home/watmin/work/holon")

from typing import List, Optional, Set, Tuple, Dict
from dataclasses import dataclass
import numpy as np
import time

from common import (
    create_client,
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


def count_total_options(grid: List[List[Optional[int]]]) -> int:
    """Count total options across all empty cells."""
    total = 0
    for r in range(9):
        for c in range(9):
            if grid[r][c] is None:
                total += len(get_available_digits_9x9(grid, r, c))
    return total


def get_option_vector(grid: List[List[Optional[int]]]) -> List[int]:
    """Get the number of options for each empty cell."""
    options = []
    for r in range(9):
        for c in range(9):
            if grid[r][c] is None:
                options.append(len(get_available_digits_9x9(grid, r, c)))
    return options


def propagate_forced(grid: List[List[Optional[int]]]) -> bool:
    """Propagate forced moves."""
    changed = True
    while changed:
        changed = False
        for r in range(9):
            for c in range(9):
                if grid[r][c] is None:
                    opts = list(get_available_digits_9x9(grid, r, c))
                    if not opts:
                        return False
                    if len(opts) == 1:
                        grid[r][c] = opts[0]
                        changed = True
    return True


@dataclass
class ConstraintDelta:
    """The change in constraint landscape after a choice."""
    digit: int
    total_before: int
    total_after: int
    delta: int  # How much total options decreased
    cells_filled: int  # How many cells were filled by propagation
    min_options_before: int
    min_options_after: int
    contradicted: bool


def analyze_choice_constraint_delta(grid: List[List[Optional[int]]],
                                    row: int, col: int, digit: int) -> ConstraintDelta:
    """
    Analyze how a choice changes the constraint landscape.
    """
    # Before
    total_before = count_total_options(grid)
    options_before = get_option_vector(grid)
    min_before = min(options_before) if options_before else 0
    empty_before = count_empty(grid)

    # After
    test_grid = [[cell for cell in r] for r in grid]
    test_grid[row][col] = digit

    success = propagate_forced(test_grid)

    if not success:
        return ConstraintDelta(
            digit=digit,
            total_before=total_before,
            total_after=0,
            delta=total_before,  # All options gone
            cells_filled=empty_before - count_empty(test_grid),
            min_options_before=min_before,
            min_options_after=0,
            contradicted=True
        )

    total_after = count_total_options(test_grid)
    options_after = get_option_vector(test_grid)
    min_after = min(options_after) if options_after else 0
    empty_after = count_empty(test_grid)

    return ConstraintDelta(
        digit=digit,
        total_before=total_before,
        total_after=total_after,
        delta=total_before - total_after,
        cells_filled=empty_before - empty_after,
        min_options_before=min_before,
        min_options_after=min_after,
        contradicted=False
    )


def score_by_constraint_collapse(delta: ConstraintDelta) -> float:
    """
    Score a choice by how much it collapses the constraint space.

    HIGHER is better:
    - More delta (options removed) = more constraining
    - More cells filled = more progress
    - NOT contradicted = viable
    """
    if delta.contradicted:
        return -1000

    # Primary: options removed per cell filled
    if delta.cells_filled > 0:
        efficiency = delta.delta / delta.cells_filled
    else:
        efficiency = 0

    # Secondary: absolute progress (cells filled)
    progress = delta.cells_filled * 10

    return efficiency + progress


class ConstraintCollapseSolver:
    """
    Solver that orders choices by constraint collapse rate.
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.backtracks = 0
        self.rejections = 0

    def solve(self, puzzle: List[List[Optional[int]]]) -> Tuple[bool, List[List[int]]]:
        grid = [[cell for cell in row] for row in puzzle]

        if not propagate_forced(grid):
            return False, [[0]*9 for _ in range(9)]

        result = self.solve_recursive(grid, depth=0)

        if result:
            return True, result
        else:
            return False, [[0]*9 for _ in range(9)]

    def solve_recursive(self, grid: List[List[Optional[int]]], depth: int) -> Optional[List[List[int]]]:
        if count_empty(grid) == 0:
            return grid

        # Find cell with fewest options
        best = None
        best_count = 10

        for r in range(9):
            for c in range(9):
                if grid[r][c] is None:
                    opts = list(get_available_digits_9x9(grid, r, c))
                    if not opts:
                        return None
                    if len(opts) < best_count:
                        best_count = len(opts)
                        best = (r, c, opts)

        if best is None:
            return grid

        r, c, options = best

        # Analyze constraint delta for each option
        deltas = []
        for digit in options:
            delta = analyze_choice_constraint_delta(grid, r, c, digit)
            score = score_by_constraint_collapse(delta)
            deltas.append((score, delta))

        # Sort by score (highest first)
        deltas.sort(key=lambda x: x[0], reverse=True)

        if self.verbose and depth < 3:
            print(f"  Depth {depth}: Cell ({r},{c}) options: {options}")
            for score, d in deltas:
                status = "CONTRA" if d.contradicted else f"filled={d.cells_filled}, delta={d.delta}"
                print(f"    Digit {d.digit}: score={score:.1f}, {status}")

        # Try in order
        for score, delta in deltas:
            if delta.contradicted:
                self.rejections += 1
                continue

            digit = delta.digit

            test_grid = [[cell for cell in row] for row in grid]
            test_grid[r][c] = digit

            if not propagate_forced(test_grid):
                self.rejections += 1
                continue

            result = self.solve_recursive(test_grid, depth + 1)
            if result:
                return result

            self.backtracks += 1

        return None


class UltimateSolver:
    """
    ULTIMATE: Combines ALL successful techniques:
    1. Simulation rejection (skip paths that contradict)
    2. Chain length ordering (more forced = better)
    3. Constraint collapse (higher delta = more constraining)
    """

    def __init__(self, sim_depth: int = 10, verbose: bool = True):
        self.sim_depth = sim_depth
        self.verbose = verbose
        self.backtracks = 0
        self.simulation_rejections = 0

    def solve(self, puzzle: List[List[Optional[int]]]) -> Tuple[bool, List[List[int]]]:
        grid = [[cell for cell in row] for row in puzzle]

        if not propagate_forced(grid):
            return False, [[0]*9 for _ in range(9)]

        result = self.solve_recursive(grid, depth=0)

        if result:
            return True, result
        else:
            return False, [[0]*9 for _ in range(9)]

    def simulate_and_score(self, grid: List[List[Optional[int]]],
                           row: int, col: int, digit: int) -> Tuple[bool, float]:
        """
        Simulate and return (survives, combined_score).

        Combined score = forced_moves * 10 + constraint_delta
        """
        test_grid = [[cell for cell in r] for r in grid]
        test_grid[row][col] = digit

        # Before propagation
        total_before = count_total_options(grid)

        forced_count = 0

        for _ in range(self.sim_depth):
            if not propagate_forced(test_grid):
                return False, 0

            # Count new forced
            new_empty = count_empty(test_grid)

            # Find next cell with fewest options
            best = None
            best_count = 10

            for r in range(9):
                for c in range(9):
                    if test_grid[r][c] is None:
                        opts = list(get_available_digits_9x9(test_grid, r, c))
                        if not opts:
                            return False, 0
                        if len(opts) < best_count:
                            best_count = len(opts)
                            best = (r, c, opts)

            if best is None:
                # Solved!
                return True, 1000 + forced_count * 10

            if best_count == 1:
                r, c, opts = best
                test_grid[r][c] = opts[0]
                forced_count += 1
            else:
                break  # Hit ambiguity

        # Calculate constraint delta
        total_after = count_total_options(test_grid)
        delta = total_before - total_after

        # Combined score
        score = forced_count * 10 + delta

        return True, score

    def solve_recursive(self, grid: List[List[Optional[int]]], depth: int) -> Optional[List[List[int]]]:
        if count_empty(grid) == 0:
            return grid

        # Find cell with fewest options (MRV)
        best = None
        best_count = 10

        for r in range(9):
            for c in range(9):
                if grid[r][c] is None:
                    opts = list(get_available_digits_9x9(grid, r, c))
                    if not opts:
                        return None
                    if len(opts) < best_count:
                        best_count = len(opts)
                        best = (r, c, opts)

        if best is None:
            return grid

        r, c, options = best

        # Score each option
        scored = []
        for digit in options:
            survives, score = self.simulate_and_score(grid, r, c, digit)
            if survives:
                scored.append((score, digit))
            else:
                self.simulation_rejections += 1

        # Sort by score (descending)
        scored.sort(reverse=True)

        if not scored:
            # All rejected - try anyway
            scored = [(0, d) for d in options]

        for score, digit in scored:
            test_grid = [[cell for cell in row] for row in grid]
            test_grid[r][c] = digit

            if not propagate_forced(test_grid):
                continue

            result = self.solve_recursive(test_grid, depth + 1)
            if result:
                return result

            self.backtracks += 1

        return None


def compare_all_solvers():
    """Compare all solver variants."""
    print("=" * 70)
    print("CONSTRAINT LANDSCAPE ANALYSIS")
    print("=" * 70)

    grid = [[cell for cell in row] for row in PUZZLE_9x9_HARD]
    propagate_forced(grid)

    print(f"\nInitial state: {count_empty(grid)} empty cells")
    print(f"Total options: {count_total_options(grid)}")

    # Find first decision point
    for r in range(9):
        for c in range(9):
            if grid[r][c] is None:
                opts = list(get_available_digits_9x9(grid, r, c))
                if len(opts) > 1:
                    print(f"\nDecision point: ({r},{c}) with options {opts}")

                    for digit in opts:
                        delta = analyze_choice_constraint_delta(grid, r, c, digit)
                        score = score_by_constraint_collapse(delta)

                        print(f"\n  Digit {digit}:")
                        print(f"    Options before: {delta.total_before}")
                        print(f"    Options after:  {delta.total_after}")
                        print(f"    Delta:          {delta.delta}")
                        print(f"    Cells filled:   {delta.cells_filled}")
                        print(f"    Contradicted:   {delta.contradicted}")
                        print(f"    SCORE:          {score:.1f}")

                    break
            else:
                continue
            break

    # Compare solvers
    print("\n" + "=" * 70)
    print("SOLVER COMPARISON")
    print("=" * 70)

    from approach_10_global import SimulationGuidedSolver
    from approach_19_opportunistic import HybridSolver

    solvers = [
        ("Standard (sim-guided)", SimulationGuidedSolver(verbose=False)),
        ("Hybrid (sim + chain)", HybridSolver(sim_depth=10, verbose=False)),
        ("Constraint Collapse", ConstraintCollapseSolver(verbose=False)),
        ("ULTIMATE (sim + chain + delta)", UltimateSolver(sim_depth=10, verbose=False)),
    ]

    for name, solver in solvers:
        start = time.time()
        success, result = solver.solve(PUZZLE_9x9_HARD)
        elapsed = time.time() - start

        print(f"\n{name}:")
        print(f"  Solved: {success}")
        print(f"  Backtracks: {solver.backtracks}")

        if hasattr(solver, 'rejections'):
            print(f"  Rejections: {solver.rejections}")
        if hasattr(solver, 'simulation_rejections'):
            print(f"  Sim rejections: {solver.simulation_rejections}")

        print(f"  Time: {elapsed:.3f}s")

        if success:
            print(f"  Valid: {validate_9x9(result)}")


def main():
    compare_all_solvers()


if __name__ == "__main__":
    main()
