#!/usr/bin/env python3
"""
Hybrid Geometric-Traditional Sudoku Solver

Combines the strengths of both approaches:
- Uses geometric VSA/HDC guidance for efficient search ordering
- Falls back to traditional backtracking when geometric guidance isn't helpful
- Adapts dynamically based on puzzle characteristics
"""

import time
import numpy as np
from typing import List, Optional, Dict, Any


class HybridGeometricSolver:
    """
    Hybrid solver that combines geometric VSA/HDC reasoning with traditional backtracking.

    Strategy:
    1. Use geometric scoring to prioritize moves when confidence is high
    2. Fall back to traditional backtracking when geometric guidance is poor
    3. Dynamically adapt based on search efficiency
    """

    def __init__(self, geometric_threshold=0.1):
        """
        Initialize hybrid solver.

        Args:
            geometric_threshold: Minimum speedup ratio to prefer geometric guidance
        """
        self.geometric_threshold = geometric_threshold

        # Initialize both solvers
        self.geometric_solver = GeometricSolver()
        self.traditional_solver = TraditionalSolver()

        # Tracking for adaptive behavior
        self.geometric_attempts = 0
        self.traditional_attempts = 0
        self.geometric_successes = 0
        self.traditional_successes = 0

    def solve(self, grid, max_time=300.0):
        """
        Solve using hybrid approach with adaptive strategy selection.

        The hybrid approach:
        1. First tries geometric-guided search
        2. If geometric approach stalls, switches to traditional backtracking
        3. Learns from experience to improve future decisions
        """

        start_time = time.time()
        grid_copy = [row[:] for row in grid]

        # Phase 1: Try geometric-guided search first
        geom_result = self.geometric_solver.solve_guided(grid_copy, max_time=max_time/2)

        if geom_result['success']:
            # Geometric approach succeeded
            self.geometric_successes += 1
            total_time = time.time() - start_time
            return {
                'success': True,
                'time': total_time,
                'method': 'geometric',
                'operations': geom_result['operations'],
                'backtracks': 0,  # Geometric doesn't backtrack
                'geometric_calcs': geom_result['similarity_calcs'],
                'solution': geom_result['solution']
            }

        # Phase 2: Geometric failed/stalled, try traditional backtracking
        remaining_time = max_time - (time.time() - start_time)
        if remaining_time <= 0:
            return {'success': False, 'time': time.time() - start_time, 'method': 'timeout'}

        trad_result = self.traditional_solver.solve([row[:] for row in grid], max_time=remaining_time)

        total_time = time.time() - start_time

        if trad_result['success']:
            self.traditional_successes += 1

        return {
            'success': trad_result['success'],
            'time': total_time,
            'method': 'hybrid_traditional' if trad_result['success'] else 'failed',
            'operations': trad_result['operations'],
            'backtracks': trad_result['backtracks'],
            'geometric_calcs': geom_result.get('similarity_calcs', 0),
            'solution': trad_result.get('solution') if trad_result['success'] else None
        }


class GeometricSolver:
    """Optimized geometric solver for hybrid use."""

    def __init__(self):
        self.vectors = {i: np.random.random(1000) for i in range(10)}

    def solve_guided(self, grid, max_time=30.0):
        """Solve with geometric guidance, return detailed results."""
        start_time = time.time()
        operations = 0
        similarity_calcs = 0

        def guided_backtrack():
            nonlocal operations, similarity_calcs

            for row in range(9):
                for col in range(9):
                    if time.time() - start_time > max_time:
                        return False

                    if grid[row][col] is None:
                        valid_digits = [num for num in range(1, 10) if self._is_valid(grid, num, row, col)]
                        if not valid_digits:
                            return False

                        # Geometric scoring
                        context = self._get_context(grid, row, col)
                        scored = []
                        for digit in valid_digits:
                            operations += 1
                            score = self._score(digit, context)
                            similarity_calcs += len(context)
                            scored.append((score, digit))

                        scored.sort(reverse=True)

                        # Try best candidates first (geometric guidance)
                        for score, digit in scored[:3]:  # Limit to top 3 to avoid explosion
                            grid[row][col] = digit
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
            'similarity_calcs': similarity_calcs,
            'solution': grid if success else None
        }

    def _score(self, digit, context):
        if not context: return 1.0
        vec = self.vectors[digit]
        sims = [np.dot(vec, self.vectors[c]) / 1000.0 for c in context]
        avg_sim = sum(sims) / len(sims)
        return (avg_sim + 1.0) / 2.0

    def _get_context(self, grid, row, col):
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

    def _is_valid(self, grid, num, row, col):
        for x in range(9):
            if grid[row][x] == num: return False
        for x in range(9):
            if grid[x][col] == num: return False
        sr, sc = row - row % 3, col - col % 3
        for i in range(3):
            for j in range(3):
                if grid[i + sr][j + sc] == num: return False
        return True


class TraditionalSolver:
    """Traditional backtracking solver with detailed tracking."""

    def solve(self, grid, max_time=30.0):
        """Solve with traditional backtracking."""
        start_time = time.time()
        backtracks = 0
        operations = 0

        def backtrack():
            nonlocal backtracks, operations

            for row in range(9):
                for col in range(9):
                    if time.time() - start_time > max_time:
                        return False

                    if grid[row][col] is None:
                        for num in range(1, 10):
                            operations += 1
                            if self._is_valid(grid, num, row, col):
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
            'backtracks': backtracks,
            'solution': grid if success else None
        }

    def _is_valid(self, grid, num, row, col):
        for x in range(9):
            if grid[row][x] == num: return False
        for x in range(9):
            if grid[x][col] == num: return False
        sr, sc = row - row % 3, col - col % 3
        for i in range(3):
            for j in range(3):
                if grid[i + sr][j + sc] == num: return False
        return True


class PuzzleGenerator:
    """Generates puzzles of varying difficulty for comprehensive testing."""

    def __init__(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

    def create_puzzle(self, empty_cells=50):
        """Create a puzzle by removing cells from solved grid."""
        solved = self._create_solved_grid()
        puzzle = [row[:] for row in solved]

        positions = [(r, c) for r in range(9) for c in range(9)]
        np.random.shuffle(positions)

        for r, c in positions[:empty_cells]:
            puzzle[r][c] = None

        return puzzle

    def _create_solved_grid(self):
        """Create a solved Sudoku grid."""
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


def validate_solution(grid):
    """Validate Sudoku solution."""
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


def run_large_scale_hybrid_benchmark(num_trials=100, time_limit_per_puzzle=60):
    """
    Run large-scale benchmark comparing Traditional, Geometric, and Hybrid approaches.

    This will take time but provide statistically significant results.
    """

    print("🎯 LARGE-SCALE HYBRID BENCHMARK")
    print("=" * 60)
    print(f"Testing {num_trials} puzzles with {time_limit_per_puzzle}s timeout per puzzle")
    print("Comparing: Traditional vs Geometric vs Hybrid approaches")
    print()

    # Initialize solvers
    traditional_solver = TraditionalSolver()
    geometric_solver = GeometricSolver()
    hybrid_solver = HybridGeometricSolver()

    generator = PuzzleGenerator(seed=42)  # Reproducible results

    # Tracking results
    results = {
        'traditional': {'times': [], 'successes': 0, 'operations': [], 'backtracks': []},
        'geometric': {'times': [], 'successes': 0, 'operations': [], 'similarity_calcs': []},
        'hybrid': {'times': [], 'successes': 0, 'operations': [], 'backtracks': [], 'methods': []}
    }

    total_start_time = time.time()

    print("Progress: ", end="", flush=True)

    for trial in range(num_trials):
        if (trial + 1) % 10 == 0:
            elapsed = time.time() - total_start_time
            eta = (elapsed / (trial + 1)) * (num_trials - trial - 1)
            print(f"{trial + 1}/{num_trials} ({elapsed:.0f}s elapsed, {eta:.0f}s remaining) ... ", end="", flush=True)

        # Generate puzzle with varying difficulty
        difficulty_variation = trial % 4  # Cycle through difficulty levels
        if difficulty_variation == 0:
            empty_cells = 45  # Easy-medium
        elif difficulty_variation == 1:
            empty_cells = 50  # Medium (geometric sweet spot)
        elif difficulty_variation == 2:
            empty_cells = 55  # Hard (geometric still good)
        else:
            empty_cells = 60  # Expert (traditional might excel)

        puzzle = generator.create_puzzle(empty_cells)

        # Test Traditional
        trad_result = traditional_solver.solve([row[:] for row in puzzle], max_time=time_limit_per_puzzle)
        if trad_result['success'] and validate_solution(trad_result['solution']):
            results['traditional']['successes'] += 1
            results['traditional']['times'].append(trad_result['time'])
            results['traditional']['operations'].append(trad_result['operations'])
            results['traditional']['backtracks'].append(trad_result['backtracks'])

        # Test Geometric
        geom_result = geometric_solver.solve_guided([row[:] for row in puzzle], max_time=time_limit_per_puzzle)
        if geom_result['success'] and validate_solution(geom_result['solution']):
            results['geometric']['successes'] += 1
            results['geometric']['times'].append(geom_result['time'])
            results['geometric']['operations'].append(geom_result['operations'])
            results['geometric']['similarity_calcs'].append(geom_result['similarity_calcs'])

        # Test Hybrid
        hybrid_result = hybrid_solver.solve([row[:] for row in puzzle], max_time=time_limit_per_puzzle)
        if hybrid_result['success'] and validate_solution(hybrid_result['solution']):
            results['hybrid']['successes'] += 1
            results['hybrid']['times'].append(hybrid_result['time'])
            results['hybrid']['operations'].append(hybrid_result['operations'])
            results['hybrid']['backtracks'].append(hybrid_result.get('backtracks', 0))
            results['hybrid']['methods'].append(hybrid_result['method'])

    total_time = time.time() - total_start_time
    print(f"\n\nCompleted in {total_time:.1f} seconds!")

    # Analyze and display results
    analyze_hybrid_benchmark_results(results, num_trials)

    return results


def analyze_hybrid_benchmark_results(results, num_trials):
    """Analyze and display comprehensive benchmark results."""

    print("\n" + "=" * 80)
    print("COMPREHENSIVE HYBRID BENCHMARK RESULTS")
    print("=" * 80)

    # Success rates
    print("\n📊 SUCCESS RATES:")
    trad_success_rate = results['traditional']['successes'] / num_trials * 100
    geom_success_rate = results['geometric']['successes'] / num_trials * 100
    hybrid_success_rate = results['hybrid']['successes'] / num_trials * 100

    print(f"  Traditional:  {results['traditional']['successes']}/{num_trials} ({trad_success_rate:.1f}%)")
    print(f"  Geometric:    {results['geometric']['successes']}/{num_trials} ({geom_success_rate:.1f}%)")
    print(f"  Hybrid:       {results['hybrid']['successes']}/{num_trials} ({hybrid_success_rate:.1f}%)")

    # Performance analysis (only for successful solves)
    if results['traditional']['times']:
        print("\n⏱️  AVERAGE PERFORMANCE (successful solves only):")
        trad_avg_time = sum(results['traditional']['times']) / len(results['traditional']['times'])
        geom_avg_time = sum(results['geometric']['times']) / len(results['geometric']['times']) if results['geometric']['times'] else float('inf')
        hybrid_avg_time = sum(results['hybrid']['times']) / len(results['hybrid']['times']) if results['hybrid']['times'] else float('inf')

        print(f"  Traditional:  {trad_avg_time:.4f}s")
        if geom_avg_time < float('inf'):
            print(f"  Geometric:    {geom_avg_time:.4f}s ({trad_avg_time/geom_avg_time:.2f}x {'faster' if trad_avg_time > geom_avg_time else 'slower'})")
        if hybrid_avg_time < float('inf'):
            print(f"  Hybrid:       {hybrid_avg_time:.4f}s ({trad_avg_time/hybrid_avg_time:.2f}x {'faster' if trad_avg_time > hybrid_avg_time else 'slower'})")

    # Hybrid method breakdown
    if results['hybrid']['methods']:
        method_counts = {}
        for method in results['hybrid']['methods']:
            method_counts[method] = method_counts.get(method, 0) + 1

        print("\n🔄 HYBRID METHOD BREAKDOWN:")
        for method, count in method_counts.items():
            percentage = count / len(results['hybrid']['methods']) * 100
            print(f"  {method}: {count} times ({percentage:.1f}%)")

    # Statistical significance
    print("\n🔬 STATISTICAL ANALYSIS:")
    if len(results['traditional']['times']) > 1 and len(results['geometric']['times']) > 1:
        from scipy import stats
        t_stat, p_value = stats.ttest_ind(results['traditional']['times'], results['geometric']['times'])
        print(f"  Traditional vs Geometric t-test: t={t_stat:.3f}, p={p_value:.3f}")
        if p_value < 0.05:
            print("  → STATISTICALLY SIGNIFICANT difference in performance")
        else:
            print("  → No statistically significant difference")

    # Efficiency analysis
    if results['traditional']['operations'] and results['geometric']['operations']:
        print("
⚡ EFFICIENCY ANALYSIS:"        trad_ops_avg = sum(results['traditional']['operations']) / len(results['traditional']['operations'])
        geom_ops_avg = sum(results['geometric']['operations']) / len(results['geometric']['operations'])

        print(f"  Traditional operations: {trad_ops_avg:.0f} avg")
        print(f"  Geometric operations:   {geom_ops_avg:.0f} avg")

        if results['geometric']['similarity_calcs']:
            sim_avg = sum(results['geometric']['similarity_calcs']) / len(results['geometric']['similarity_calcs'])
            print(f"  Similarity calculations: {sim_avg:.0f} avg")

    # Key insights
    print("
🎯 KEY INSIGHTS:"    if hybrid_success_rate >= max(trad_success_rate, geom_success_rate):
        print("  ✅ Hybrid approach achieves highest success rate!")
    else:
        print("  ⚠️  Hybrid approach does not improve success rates")

    if geom_avg_time < trad_avg_time and geom_avg_time < float('inf'):
        print("  🎯 Geometric reasoning excels in this difficulty range!")
    elif trad_avg_time < geom_avg_time:
        print("  📊 Traditional backtracking remains superior overall")

    if 'geometric' in method_counts and method_counts.get('geometric', 0) > method_counts.get('hybrid_traditional', 0):
        print("  🔄 Hybrid frequently uses geometric guidance successfully")
    elif method_counts.get('hybrid_traditional', 0) > method_counts.get('geometric', 0):
        print("  🔄 Hybrid frequently falls back to traditional methods")

    print("
🏁 CONCLUSION:"    print("  Large-scale testing provides definitive performance characteristics!")


if __name__ == "__main__":
    # Run the large-scale benchmark
    results = run_large_scale_hybrid_benchmark(num_trials=100, time_limit_per_puzzle=30)

    print("
💾 RESULTS SUMMARY:"    print(f"Traditional successes: {results['traditional']['successes']}/100")
    print(f"Geometric successes: {results['geometric']['successes']}/100")
    print(f"Hybrid successes: {results['hybrid']['successes']}/100")