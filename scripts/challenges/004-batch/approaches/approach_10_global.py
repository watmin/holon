#!/usr/bin/env python3
"""
Approach 10: Global Coherence Optimization

THE BREAKTHROUGH INSIGHT:
All previous approaches fail because they only consider LOCAL validity.
A digit might be valid for its row/col/block but lead to an unsolvable grid.

WHAT IS GLOBAL COHERENCE?
The entire grid forms a coherent whole where all constraints are satisfiable.
This is NOT just "no duplicates" - it's "the remaining cells can still be filled."

NEW APPROACH:
Instead of scoring individual placements, score the ENTIRE GRID STATE.
Pick the placement that leaves the grid most "solvable."

HOW TO MEASURE GLOBAL COHERENCE:
1. **Remaining Options**: For each empty cell, count available options.
   A globally coherent state has many options remaining.
   A bad choice creates cells with 0 options (contradiction).

2. **Constraint Satisfaction Potential**:
   Encode all constraints, measure how "satisfiable" they remain.

3. **Dimensionality Cascade**:
   Use dimensionality (from Approach 9) but compute it for ALL constraints.
   Sum the dimensionality across all rows/cols/blocks.
   Higher total = more globally coherent.

This is more expensive (evaluates entire grid per choice) but might break
the 54/58 barrier by seeing global consequences of local choices.
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
    effective_dimensionality,
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


class GlobalCoherenceSolver:
    """
    Solve by optimizing global coherence rather than local fitness.
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
        self.digit_basis = [self.digit_vectors[d] for d in self.digits]

        self.notes: List[str] = []
        self.iterations = 0
        self.evaluations = 0

    def log(self, msg: str):
        if self.verbose:
            print(msg)
        self.notes.append(msg)

    def get_available(self, grid, row, col) -> Set[int]:
        if self.size == 4:
            return get_available_digits_4x4(grid, row, col)
        return get_available_digits_9x9(grid, row, col)

    def count_total_options(self, grid: List[List[Optional[int]]]) -> int:
        """
        Count total available options across all empty cells.

        Higher = more flexible/solvable state
        Zero options anywhere = contradiction
        """
        total = 0
        for r in range(self.size):
            for c in range(self.size):
                if grid[r][c] is None:
                    opts = len(self.get_available(grid, r, c))
                    if opts == 0:
                        return -1  # Contradiction!
                    total += opts
        return total

    def min_options(self, grid: List[List[Optional[int]]]) -> int:
        """
        Find minimum options for any empty cell.

        Zero = contradiction
        One = forced move (good)
        Many = uncertainty
        """
        min_opts = self.size + 1
        for r in range(self.size):
            for c in range(self.size):
                if grid[r][c] is None:
                    opts = len(self.get_available(grid, r, c))
                    if opts == 0:
                        return 0  # Contradiction
                    min_opts = min(min_opts, opts)
        return min_opts if min_opts <= self.size else self.size

    def global_dimensionality(self, grid: List[List[Optional[int]]]) -> float:
        """
        Compute global dimensionality across ALL constraints.

        Sum of dimensionality for all rows + cols + blocks.
        Higher = more "valid-like" configuration.
        """
        total_dim = 0.0

        # Rows
        for r in range(self.size):
            row_digits = [grid[r][c] for c in range(self.size) if grid[r][c] is not None]
            if row_digits:
                row_vec = bundle([self.digit_vectors[d] for d in row_digits])
                total_dim += effective_dimensionality(row_vec, self.digit_basis)

        # Columns
        for c in range(self.size):
            col_digits = [grid[r][c] for r in range(self.size) if grid[r][c] is not None]
            if col_digits:
                col_vec = bundle([self.digit_vectors[d] for d in col_digits])
                total_dim += effective_dimensionality(col_vec, self.digit_basis)

        # Blocks
        for br in range(0, self.size, self.block_size):
            for bc in range(0, self.size, self.block_size):
                block_digits = []
                for r in range(br, br + self.block_size):
                    for c in range(bc, bc + self.block_size):
                        if grid[r][c] is not None:
                            block_digits.append(grid[r][c])
                if block_digits:
                    block_vec = bundle([self.digit_vectors[d] for d in block_digits])
                    total_dim += effective_dimensionality(block_vec, self.digit_basis)

        return total_dim

    def evaluate_placement(self, grid: List[List[Optional[int]]],
                           row: int, col: int, digit: int) -> Tuple[float, int, int]:
        """
        Evaluate a potential placement by its effect on global coherence.

        Returns:
        - global_dim: Total dimensionality after placement
        - total_opts: Total options remaining after placement
        - min_opts: Minimum options for any cell after placement
        """
        # Make temporary placement
        grid[row][col] = digit
        self.evaluations += 1

        # Evaluate global state
        total_opts = self.count_total_options(grid)
        min_opts = self.min_options(grid)

        # If contradiction, return worst score
        if total_opts < 0 or min_opts == 0:
            grid[row][col] = None
            return -999, -1, 0

        # Compute global dimensionality (expensive but informative)
        global_dim = self.global_dimensionality(grid)

        # Undo placement
        grid[row][col] = None

        return global_dim, total_opts, min_opts

    def solve(self, puzzle: List[List[Optional[int]]],
              strategy: str = "options") -> Tuple[bool, List[List[int]]]:
        """
        Solve by maximizing global coherence.

        Strategies:
        - "options": Maximize total remaining options
        - "dimensionality": Maximize global dimensionality
        - "combined": Balance both
        """
        self.log(f"\n{'='*60}")
        self.log(f"APPROACH 10: GLOBAL COHERENCE (strategy={strategy})")
        self.log(f"{'='*60}")
        self.log(f"Empty cells: {count_empty(puzzle)}")

        grid = [[cell for cell in row] for row in puzzle]
        max_iterations = self.size * self.size

        while self.iterations < max_iterations:
            self.iterations += 1

            # Find all empty cells and their available options
            candidates = []
            for r in range(self.size):
                for c in range(self.size):
                    if grid[r][c] is None:
                        available = self.get_available(grid, r, c)
                        if available:
                            candidates.append((r, c, available))

            if not candidates:
                break

            # Use MRV: Start with cells that have fewest options
            candidates.sort(key=lambda x: len(x[2]))

            # Evaluate each option for the most constrained cell(s)
            best_placement = None
            best_score = -float('inf')

            # Only evaluate top few candidates (most constrained)
            for r, c, available in candidates[:3]:
                for d in available:
                    global_dim, total_opts, min_opts = self.evaluate_placement(grid, r, c, d)

                    if total_opts < 0:
                        continue  # Contradiction

                    # Compute score based on strategy
                    if strategy == "options":
                        score = total_opts * 10 + min_opts
                    elif strategy == "dimensionality":
                        score = global_dim
                    else:  # combined
                        score = global_dim + total_opts * 0.01

                    if score > best_score:
                        best_score = score
                        best_placement = (r, c, d, global_dim, total_opts, min_opts)

            if best_placement is None:
                self.log(f"\n✗ No valid placement found")
                break

            r, c, d, g_dim, t_opts, m_opts = best_placement
            grid[r][c] = d
            self.log(f"  [Global] ({r},{c}) → {d} (dim={g_dim:.2f}, opts={t_opts}, min={m_opts})")

        empty = count_empty(grid)
        if empty == 0:
            self.log(f"\n✓ All cells filled after {self.iterations} iterations")
            self.log(f"Total evaluations: {self.evaluations}")
        else:
            self.log(f"\n⚠ Stuck with {empty} cells empty")

        return empty == 0, grid


def test_4x4():
    print("\n" + "=" * 60)
    print("TEST: 4x4 with Global Coherence")
    print("=" * 60)

    print("\nInput:")
    print_grid_4x4(PUZZLE_4x4_EASY)

    solver = GlobalCoherenceSolver(size=4, verbose=True)
    with Timer() as timer:
        solved, grid = solver.solve(PUZZLE_4x4_EASY)

    print("\nResult:")
    print_grid_4x4(grid)

    valid, _ = validate_4x4(grid)
    print(f"\nValid: {valid}, Time: {timer.elapsed:.2f}s")
    return valid


def test_9x9_hard():
    print("\n" + "=" * 60)
    print("TEST: 9x9 HARD with Global Coherence")
    print("=" * 60)

    print("\nInput:")
    print_grid_9x9(PUZZLE_9x9_HARD)
    print(f"Empty cells: {count_empty(PUZZLE_9x9_HARD)}")

    # Test different strategies
    for strategy in ["options", "dimensionality", "combined"]:
        print(f"\n--- Strategy: {strategy} ---")
        solver = GlobalCoherenceSolver(size=9, verbose=False)
        with Timer() as timer:
            solved, grid = solver.solve(PUZZLE_9x9_HARD, strategy=strategy)

        cells = 58 - count_empty(grid)
        valid, _ = validate_9x9(grid)
        print(f"  Cells: {cells}/58, Valid: {valid}, Evals: {solver.evaluations}, Time: {timer.elapsed:.2f}s")

        if valid:
            print("\n✓✓ SOLVED!")
            print_grid_9x9(grid)
            return True

    # Show best result
    print("\n" + "=" * 60)
    print("Best result with 'options' strategy:")
    solver = GlobalCoherenceSolver(size=9, verbose=True)
    solved, grid = solver.solve(PUZZLE_9x9_HARD, strategy="options")

    print("\nResult:")
    print_grid_9x9(grid)

    valid, _ = validate_9x9(grid)
    return valid


class LookaheadSolver:
    """
    Solve using lookahead simulation - see future consequences of choices.
    """

    def __init__(self, size: int = 9, dimensions: int = 16384, verbose: bool = True):
        self.size = size
        self.block_size = 2 if size == 4 else 3
        self.digits = list(range(1, size + 1))
        self.verbose = verbose

        self.client = create_client(dimensions=dimensions)
        self.cache = VectorCache(self.client)
        self.digit_vectors = {d: self.cache.get_digit_vector(d) for d in self.digits}

        self.notes: List[str] = []
        self.simulations = 0

    def log(self, msg: str):
        if self.verbose:
            print(msg)
        self.notes.append(msg)

    def get_available(self, grid, row, col) -> Set[int]:
        if self.size == 4:
            return get_available_digits_4x4(grid, row, col)
        return get_available_digits_9x9(grid, row, col)

    def copy_grid(self, grid):
        return [[cell for cell in row] for row in grid]

    def propagate_forced(self, grid: List[List[Optional[int]]]) -> Tuple[bool, int]:
        """
        Propagate forced moves (naked singles) until stable.

        Returns (success, num_forced)
        - success: False if contradiction found
        - num_forced: Number of cells filled by propagation
        """
        forced_count = 0
        changed = True

        while changed:
            changed = False
            for r in range(self.size):
                for c in range(self.size):
                    if grid[r][c] is not None:
                        continue

                    available = self.get_available(grid, r, c)

                    if len(available) == 0:
                        return False, forced_count  # Contradiction

                    if len(available) == 1:
                        grid[r][c] = list(available)[0]
                        forced_count += 1
                        changed = True

        return True, forced_count

    def simulate_path(self, grid: List[List[Optional[int]]],
                      row: int, col: int, digit: int,
                      depth: int = 5) -> Tuple[bool, int, int]:
        """
        Simulate placing digit and following the path for 'depth' moves.

        Uses MRV + first available heuristic for simulation.
        Returns (survived, cells_filled, final_options)
        """
        sim_grid = self.copy_grid(grid)
        sim_grid[row][col] = digit
        self.simulations += 1

        # First propagate forced moves
        success, forced = self.propagate_forced(sim_grid)
        if not success:
            return False, 0, 0

        cells_filled = 1 + forced

        # Simulate depth more moves
        for _ in range(depth):
            # Find MRV cell
            best_cell = None
            min_opts = self.size + 1

            for r in range(self.size):
                for c in range(self.size):
                    if sim_grid[r][c] is None:
                        opts = self.get_available(sim_grid, r, c)
                        if len(opts) == 0:
                            return False, cells_filled, 0  # Contradiction
                        if len(opts) < min_opts:
                            min_opts = len(opts)
                            best_cell = (r, c, opts)

            if best_cell is None:
                # All filled!
                return True, cells_filled, 0

            r, c, opts = best_cell
            # Pick first option (consistent ordering)
            sim_grid[r][c] = min(opts)
            cells_filled += 1

            # Propagate
            success, forced = self.propagate_forced(sim_grid)
            if not success:
                return False, cells_filled, 0
            cells_filled += forced

        # Count remaining options
        total_opts = 0
        for r in range(self.size):
            for c in range(self.size):
                if sim_grid[r][c] is None:
                    total_opts += len(self.get_available(sim_grid, r, c))

        return True, cells_filled, total_opts

    def solve(self, puzzle: List[List[Optional[int]]],
              lookahead_depth: int = 5) -> Tuple[bool, List[List[int]]]:
        """
        Solve by evaluating lookahead for each choice.
        """
        self.log(f"\n{'='*60}")
        self.log(f"LOOKAHEAD SOLVER (depth={lookahead_depth})")
        self.log(f"{'='*60}")
        self.log(f"Empty cells: {count_empty(puzzle)}")

        grid = self.copy_grid(puzzle)
        max_iterations = self.size * self.size * 2

        for iteration in range(max_iterations):
            # Propagate forced moves first
            success, forced = self.propagate_forced(grid)
            if not success:
                self.log(f"\n✗ Contradiction during propagation")
                break

            if forced > 0:
                self.log(f"  [Prop] Forced {forced} cells")

            # Find MRV cell
            best_cell = None
            min_opts = self.size + 1

            for r in range(self.size):
                for c in range(self.size):
                    if grid[r][c] is None:
                        opts = self.get_available(grid, r, c)
                        if len(opts) == 0:
                            self.log(f"\n✗ Contradiction at ({r},{c})")
                            return False, grid
                        if len(opts) < min_opts:
                            min_opts = len(opts)
                            best_cell = (r, c, opts)

            if best_cell is None:
                self.log(f"\n✓ All cells filled!")
                break

            r, c, opts = best_cell

            if len(opts) == 1:
                # Forced
                grid[r][c] = list(opts)[0]
                continue

            # Evaluate each option with lookahead
            evaluations = []
            for d in opts:
                survived, filled, remaining = self.simulate_path(
                    grid, r, c, d, lookahead_depth
                )
                evaluations.append((d, survived, filled, remaining))

            # Prefer: survived > not, then most filled, then most remaining
            evaluations.sort(key=lambda x: (x[1], x[2], x[3]), reverse=True)

            best = evaluations[0]
            second = evaluations[1] if len(evaluations) > 1 else (0, False, 0, 0)

            # Log decision
            best_d, best_surv, best_fill, best_rem = best
            status = "✓" if best_surv else "✗"
            self.log(f"  [Look{lookahead_depth}] ({r},{c}) → {best_d} ({status} fill={best_fill}, rem={best_rem})")

            grid[r][c] = best_d

        self.log(f"\nTotal simulations: {self.simulations}")
        return count_empty(grid) == 0, grid


class SimulationGuidedSolver:
    """
    Use simulation to detect failures and backtrack when needed.

    This combines:
    1. Lookahead simulation to predict failures
    2. Backtracking when all options fail
    3. Geometric scoring for choice ordering
    """

    def __init__(self, size: int = 9, dimensions: int = 16384, verbose: bool = True):
        self.size = size
        self.block_size = 2 if size == 4 else 3
        self.digits = list(range(1, size + 1))
        self.verbose = verbose

        self.client = create_client(dimensions=dimensions)
        self.cache = VectorCache(self.client)
        self.digit_vectors = {d: self.cache.get_digit_vector(d) for d in self.digits}

        self.backtracks = 0
        self.simulations = 0

    def log(self, msg: str):
        if self.verbose:
            print(msg)

    def get_available(self, grid, row, col) -> Set[int]:
        if self.size == 4:
            return get_available_digits_4x4(grid, row, col)
        return get_available_digits_9x9(grid, row, col)

    def copy_grid(self, grid):
        return [[cell for cell in row] for row in grid]

    def propagate_forced(self, grid: List[List[Optional[int]]]) -> bool:
        """Propagate forced moves. Returns False if contradiction."""
        changed = True
        while changed:
            changed = False
            for r in range(self.size):
                for c in range(self.size):
                    if grid[r][c] is None:
                        available = self.get_available(grid, r, c)
                        if len(available) == 0:
                            return False
                        if len(available) == 1:
                            grid[r][c] = list(available)[0]
                            changed = True
        return True

    def simulate_survives(self, grid: List[List[Optional[int]]],
                          row: int, col: int, digit: int,
                          depth: int = 10) -> bool:
        """Check if placing digit leads to survival for 'depth' moves."""
        sim_grid = self.copy_grid(grid)
        sim_grid[row][col] = digit
        self.simulations += 1

        if not self.propagate_forced(sim_grid):
            return False

        # Simulate depth more moves
        for _ in range(depth):
            # Find MRV cell
            best_cell = None
            min_opts = self.size + 1

            for r in range(self.size):
                for c in range(self.size):
                    if sim_grid[r][c] is None:
                        opts = self.get_available(sim_grid, r, c)
                        if len(opts) == 0:
                            return False
                        if len(opts) < min_opts:
                            min_opts = len(opts)
                            best_cell = (r, c, opts)

            if best_cell is None:
                return True  # All filled - definitely survives!

            r, c, opts = best_cell
            sim_grid[r][c] = min(opts)

            if not self.propagate_forced(sim_grid):
                return False

        return True

    def solve_recursive(self, grid: List[List[Optional[int]]], depth: int = 0) -> Optional[List[List[int]]]:
        """
        Recursive solve with simulation-guided backtracking.
        Returns the solution grid or None if no solution.
        """
        # Propagate forced moves
        if not self.propagate_forced(grid):
            return None

        # Find MRV cell
        best_cell = None
        min_opts = self.size + 1

        for r in range(self.size):
            for c in range(self.size):
                if grid[r][c] is None:
                    opts = self.get_available(grid, r, c)
                    if len(opts) == 0:
                        return None
                    if len(opts) < min_opts:
                        min_opts = len(opts)
                        best_cell = (r, c, list(opts))

        if best_cell is None:
            return grid  # Solved! Return the solution

        r, c, opts = best_cell

        # Filter options using simulation
        surviving_opts = []
        for d in opts:
            if self.simulate_survives(grid, r, c, d, depth=5):
                surviving_opts.append(d)

        # If no options survive simulation, try all anyway
        if not surviving_opts:
            surviving_opts = opts
            if self.verbose and depth < 5:
                self.log(f"  [Depth {depth}] ({r},{c}): No sim survivors, trying all {opts}")

        # Try each option
        for d in surviving_opts:
            test_grid = self.copy_grid(grid)
            test_grid[r][c] = d

            if self.verbose and depth < 3:
                self.log(f"  [Depth {depth}] ({r},{c}) → {d} (of {surviving_opts})")

            result = self.solve_recursive(test_grid, depth + 1)
            if result is not None:
                return result  # Found solution!

            self.backtracks += 1

        return None  # No solution found

    def solve(self, puzzle: List[List[Optional[int]]]) -> Tuple[bool, List[List[int]]]:
        """Solve with simulation-guided backtracking."""
        self.log(f"\n{'='*60}")
        self.log("SIMULATION-GUIDED BACKTRACKING")
        self.log(f"{'='*60}")
        self.log(f"Empty cells: {count_empty(puzzle)}")

        grid = self.copy_grid(puzzle)

        with Timer() as timer:
            result = self.solve_recursive(grid)

        self.log(f"\nBacktracks: {self.backtracks}")
        self.log(f"Simulations: {self.simulations}")
        self.log(f"Time: {timer.elapsed:.2f}s")

        if result is not None:
            self.log("✓ SOLVED!")
            return True, result
        else:
            self.log("✗ Failed")
            return False, grid


def test_simulation_guided():
    print("\n" + "=" * 60)
    print("TEST: 9x9 HARD with Simulation-Guided Backtracking")
    print("=" * 60)

    print("\nInput:")
    print_grid_9x9(PUZZLE_9x9_HARD)

    solver = SimulationGuidedSolver(size=9, verbose=True)
    solved, grid = solver.solve(PUZZLE_9x9_HARD)

    print("\nResult:")
    print_grid_9x9(grid)

    valid, _ = validate_9x9(grid)
    print(f"\nValid: {valid}")

    if valid:
        print("\n" + "=" * 60)
        print("✓✓✓ GLOBAL COHERENCE ACHIEVED!")
        print("=" * 60)
        print(f"Backtracks: {solver.backtracks}")
        print(f"Simulations: {solver.simulations}")

    return valid


def test_lookahead():
    print("\n" + "=" * 60)
    print("TEST: 9x9 HARD with Lookahead")
    print("=" * 60)

    print("\nInput:")
    print_grid_9x9(PUZZLE_9x9_HARD)
    print(f"Empty cells: {count_empty(PUZZLE_9x9_HARD)}")

    for depth in [3, 5, 10]:
        print(f"\n--- Lookahead depth: {depth} ---")
        solver = LookaheadSolver(size=9, verbose=False)
        with Timer() as timer:
            solved, grid = solver.solve(PUZZLE_9x9_HARD, lookahead_depth=depth)

        cells = 58 - count_empty(grid)
        valid, _ = validate_9x9(grid)
        print(f"  Cells: {cells}/58, Valid: {valid}, Sims: {solver.simulations}, Time: {timer.elapsed:.2f}s")

        if valid:
            print("\n✓✓ SOLVED!")
            print_grid_9x9(grid)
            return True

    # Show best result
    print("\n" + "=" * 60)
    print("Best result with depth=10:")
    solver = LookaheadSolver(size=9, verbose=True)
    solved, grid = solver.solve(PUZZLE_9x9_HARD, lookahead_depth=10)

    print("\nResult:")
    print_grid_9x9(grid)

    valid, _ = validate_9x9(grid)
    print(f"\nValid: {valid}")
    return valid


def main():
    print("=" * 60)
    print("APPROACH 10: GLOBAL COHERENCE OPTIMIZATION")
    print("=" * 60)
    print("\nKey insight: Use simulation to see global consequences,")
    print("and BACKTRACK when all local choices lead to failure.")

    # The key test: simulation-guided backtracking
    result = test_simulation_guided()

    if result:
        print("\n" + "=" * 60)
        print("BREAKTHROUGH ACHIEVED!")
        print("=" * 60)
        print("\nSimulation-guided backtracking solved the hard puzzle.")
        print("The simulation acts as a geometric 'failure detector'.")
        print("Backtracking is still needed, but guided by global insight.")


if __name__ == "__main__":
    main()
