#!/usr/bin/env python3
"""
Large-Scale Benchmark: Traditional vs Geometric vs Hybrid Sudoku Solvers

Runs comprehensive statistical analysis across many puzzles.
"""

import time
import numpy as np
from collections import defaultdict


class PuzzleGenerator:
    def __init__(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

    def create_puzzle(self, empty_cells=50):
        solved = self._create_solved_grid()
        puzzle = [row[:] for row in solved]

        positions = [(r, c) for r in range(9) for c in range(9)]
        np.random.shuffle(positions)

        for r, c in positions[:empty_cells]:
            puzzle[r][c] = None

        return puzzle

    def _create_solved_grid(self):
        grid = [[0 for _ in range(9)] for _ in range(9)]

        def is_valid(num, row, col):
            for x in range(9):
                if grid[row][x] == num: return False
            for x in range(9):
                if grid[x][col] == num: return False
            sr, sc = row - row % 3, col - col % 3
            for i in range(3):
                for j in range(3):
                    if grid[i + sr][j + sc] == num: return False
            return True

        def solve():
            for row in range(9):
                for col in range(9):
                    if grid[row][col] == 0:
                        nums = list(range(1, 10))
                        np.random.shuffle(nums)
                        for num in nums:
                            if is_valid(num, row, col):
                                grid[row][col] = num
                                if solve(): return True
                                grid[row][col] = 0
                        return False
            return True

        solve()
        return grid


class TraditionalSolver:
    def solve(self, grid, max_time=30.0):
        start_time = time.time()
        backtracks = 0
        operations = 0

        def backtrack():
            nonlocal backtracks, operations
            if time.time() - start_time > max_time:
                return False

            for row in range(9):
                for col in range(9):
                    if grid[row][col] is None:
                        for num in range(1, 10):
                            operations += 1
                            if self._valid(grid, num, row, col):
                                grid[row][col] = num
                                if backtrack():
                                    return True
                                grid[row][col] = None
                                backtracks += 1
                        return False
            return True

        success = backtrack()
        solve_time = time.time() - start_time

        return {
            'success': success,
            'time': solve_time,
            'operations': operations,
            'backtracks': backtracks
        }

    def _valid(self, grid, num, row, col):
        for x in range(9):
            if grid[row][x] == num: return False
        for x in range(9):
            if grid[x][col] == num: return False
        sr, sc = row - row % 3, col - col % 3
        for i in range(3):
            for j in range(3):
                if grid[i + sr][j + sc] == num: return False
        return True


class GeometricSolver:
    def __init__(self):
        self.vectors = {i: np.random.random(1000) for i in range(10)}

    def solve(self, grid, max_time=30.0):
        start_time = time.time()
        operations = 0
        similarity_calcs = 0

        def guided_backtrack():
            nonlocal operations, similarity_calcs
            if time.time() - start_time > max_time:
                return False

            for row in range(9):
                for col in range(9):
                    if grid[row][col] is None:
                        valid = [n for n in range(1, 10) if self._valid(grid, n, row, col)]
                        if not valid: return False

                        scored = [(self._score(n, self._context(grid, row, col)), n)
                                for n in valid]
                        operations += len(valid)
                        scored.sort(reverse=True)

                        for score, num in scored[:3]:  # Top 3 candidates
                            grid[row][col] = num
                            if guided_backtrack():
                                return True
                            grid[row][col] = None
                        return False
            return True

        success = guided_backtrack()
        solve_time = time.time() - start_time

        return {
            'success': success,
            'time': solve_time,
            'operations': operations,
            'similarity_calcs': similarity_calcs
        }

    def _score(self, digit, context):
        if not context: return 1.0
        vec = self.vectors[digit]
        sims = [np.dot(vec, self.vectors[c]) / 1000.0 for c in context]
        avg_sim = sum(sims) / len(sims)
        return (avg_sim + 1.0) / 2.0

    def _context(self, grid, row, col):
        ctx = []
        for c in range(9):
            if c != col and grid[row][c]: ctx.append(grid[row][c])
        for r in range(9):
            if r != row and grid[r][col]: ctx.append(grid[r][col])
        sr, sc = row - row % 3, col - col % 3
        for i in range(3):
            for j in range(3):
                r, c = sr + i, sc + j
                if (r != row or c != col) and grid[r][c]: ctx.append(grid[r][c])
        return list(set(ctx))

    def _valid(self, grid, num, row, col):
        for x in range(9):
            if grid[row][x] == num: return False
        for x in range(9):
            if grid[x][col] == num: return False
        sr, sc = row - row % 3, col - col % 3
        for i in range(3):
            for j in range(3):
                if grid[i + sr][j + sc] == num: return False
        return True


class HybridSolver:
    def __init__(self):
        self.geometric = GeometricSolver()
        self.traditional = TraditionalSolver()

    def solve(self, grid, max_time=30.0):
        start_time = time.time()

        # Try geometric first
        geom_result = self.geometric.solve([row[:] for row in grid], max_time=max_time/2)

        if geom_result['success']:
            total_time = time.time() - start_time
            return {
                'success': True,
                'time': total_time,
                'method': 'geometric',
                'operations': geom_result['operations']
            }

        # Fall back to traditional
        remaining_time = max_time - (time.time() - start_time)
        trad_result = self.traditional.solve([row[:] for row in grid], max_time=remaining_time)

        total_time = time.time() - start_time
        return {
            'success': trad_result['success'],
            'time': total_time,
            'method': 'hybrid_traditional' if trad_result['success'] else 'failed',
            'operations': trad_result['operations'] if trad_result['success'] else 0
        }


def validate_solution(grid):
    if not grid: return False
    for row in grid:
        digits = [c for c in row if c]
        if len(digits) != 9 or len(set(digits)) != 9: return False
    for col in range(9):
        digits = [grid[r][col] for r in range(9) if grid[r][col]]
        if len(digits) != 9 or len(set(digits)) != 9: return False
    for br in range(0, 9, 3):
        for bc in range(0, 9, 3):
            digits = [grid[r][c] for r in range(br, br+3) for c in range(bc, bc+3) if grid[r][c]]
            if len(digits) != 9 or len(set(digits)) != 9: return False
    return True


def run_large_scale_benchmark(num_trials=200):
    """Run comprehensive benchmark across many puzzles."""

    print("LARGE-SCALE BENCHMARK: Traditional vs Geometric vs Hybrid")
    print("=" * 60)
    print(f"Testing {num_trials} puzzles to establish statistical significance")
    print()

    # Initialize solvers
    traditional = TraditionalSolver()
    geometric = GeometricSolver()
    hybrid = HybridSolver()
    generator = PuzzleGenerator(seed=42)

    # Results tracking
    results = {
        'traditional': {'times': [], 'successes': 0},
        'geometric': {'times': [], 'successes': 0},
        'hybrid': {'times': [], 'successes': 0, 'methods': []}
    }

    print("Running trials...")
    start_benchmark = time.time()

    for trial in range(num_trials):
        if (trial + 1) % 25 == 0:
            elapsed = time.time() - start_benchmark
            eta = (elapsed / (trial + 1)) * (num_trials - trial - 1)
            print(f"Progress: {trial + 1}/{num_trials} (ETA: {eta:.0f}s)")

        # Vary difficulty across trials
        difficulty = trial % 4
        if difficulty == 0:
            empty_cells = 45
        elif difficulty == 1:
            empty_cells = 50  # Geometric sweet spot
        elif difficulty == 2:
            empty_cells = 55  # Still good for geometric
        else:
            empty_cells = 60  # Expert level

        puzzle = generator.create_puzzle(empty_cells)

        # Test each solver
        solvers = [
            ('traditional', traditional, 'solve'),
            ('geometric', geometric, 'solve'),
            ('hybrid', hybrid, 'solve')
        ]

        for name, solver, method in solvers:
            result = getattr(solver, method)([row[:] for row in puzzle])

            if result['success']:
                results[name]['successes'] += 1
                results[name]['times'].append(result['time'])

                if name == 'hybrid':
                    results[name]['methods'].append(result.get('method', 'unknown'))

    # Analysis
    total_time = time.time() - start_benchmark
    print(f"\nCompleted {num_trials} trials in {total_time:.1f} seconds!")

    # Results summary
    print("\nFINAL RESULTS:")
    print("=" * 40)

    for name in ['traditional', 'geometric', 'hybrid']:
        success_rate = results[name]['successes'] / num_trials * 100
        avg_time = sum(results[name]['times']) / len(results[name]['times']) if results[name]['times'] else 0
        print("15")

    # Hybrid method breakdown
    if results['hybrid']['methods']:
        methods = defaultdict(int)
        for method in results['hybrid']['methods']:
            methods[method] += 1

        print("\nHybrid method usage:")
        for method, count in methods.items():
            pct = count / len(results['hybrid']['methods']) * 100
            print(f"  {method}: {count} ({pct:.1f}%)")

    # Performance comparison
    if results['traditional']['times'] and results['geometric']['times']:
        trad_avg = sum(results['traditional']['times']) / len(results['traditional']['times'])
        geom_avg = sum(results['geometric']['times']) / len(results['geometric']['times'])
        speedup = trad_avg / geom_avg

        print("
Performance comparison:"        print(".2f"        if speedup > 1.05:
            print("Geometric is generally FASTER")
        elif speedup < 0.95:
            print("Traditional is generally FASTER")
        else:
            print("Performance is roughly equal")

    return results


if __name__ == "__main__":
    results = run_large_scale_benchmark(100)  # Start with 100 trials

    print("
SUMMARY:"    print(f"Traditional success rate: {results['traditional']['successes']}/100 = {results['traditional']['successes']}%"    print(f"Geometric success rate: {results['geometric']['successes']}/100 = {results['geometric']['successes']}%"    print(f"Hybrid success rate: {results['hybrid']['successes']}/100 = {results['hybrid']['successes']}%"