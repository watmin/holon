#!/usr/bin/env python3
"""
Approach 43: Best Combination - Template Matching + Simulation Rejection

Our two best heuristics:
1. Template Matching (Approach 22): 52 backtracks
2. Simulation Rejection (Approach 10): 10x reduction

THE IDEA:
Combine them properly:
1. Use template matching for ORDERING choices
2. Use simulation to REJECT choices that lead to contradictions
3. Skip simulation for choices that template strongly prefers

This should give us the best of both worlds.
"""

import sys
sys.path.insert(0, "/home/watmin/work/holon")

from typing import List, Optional, Set, Tuple, Dict
import numpy as np
import time

from holon import CPUStore

from common import (
    similarity,
    count_empty,
    get_available_digits_9x9,
    validate_9x9,
    PUZZLE_9x9_HARD,
)


def create_store(dimensions: int = 16384):
    return CPUStore(dimensions=dimensions)


# =============================================================================
# Pure Template Matching Solver (Baseline)
# =============================================================================

def solve_template_only(puzzle, store, digit_vecs, complete_template):
    """Template matching only - our current best."""
    grid = [[cell for cell in row] for row in puzzle]
    backtracks = [0]

    def score(g, r, c, d):
        total = 0.0
        # Row
        row_digits = {g[r][cc] for cc in range(9) if g[r][cc] is not None}
        row_digits.add(d)
        row_vec = store.bundle([digit_vecs[x] for x in row_digits])
        total += similarity(row_vec, complete_template)

        # Col
        col_digits = {g[rr][c] for rr in range(9) if g[rr][c] is not None}
        col_digits.add(d)
        col_vec = store.bundle([digit_vecs[x] for x in col_digits])
        total += similarity(col_vec, complete_template)

        # Block
        br, bc = (r // 3) * 3, (c // 3) * 3
        block_digits = {g[rr][cc] for rr in range(br, br+3) for cc in range(bc, bc+3)
                        if g[rr][cc] is not None}
        block_digits.add(d)
        block_vec = store.bundle([digit_vecs[x] for x in block_digits])
        total += similarity(block_vec, complete_template)

        return total

    def rec(g):
        changed = True
        while changed:
            changed = False
            for r in range(9):
                for c in range(9):
                    if g[r][c] is None:
                        opts = list(get_available_digits_9x9(g, r, c))
                        if not opts:
                            return None
                        if len(opts) == 1:
                            g[r][c] = opts[0]
                            changed = True

        if count_empty(g) == 0:
            return g

        best = None
        best_count = 10
        for r in range(9):
            for c in range(9):
                if g[r][c] is None:
                    opts = list(get_available_digits_9x9(g, r, c))
                    if len(opts) < best_count:
                        best_count = len(opts)
                        best = (r, c, opts)

        if best is None:
            return None

        r, c, options = best

        scores = [(score(g, r, c, d), d) for d in options]
        scores.sort(reverse=True)
        ordered = [x[1] for x in scores]

        for digit in ordered:
            test = [[cell for cell in row] for row in g]
            test[r][c] = digit
            result = rec(test)
            if result:
                return result
            backtracks[0] += 1

        return None

    result = rec(grid)
    return backtracks[0], result


# =============================================================================
# Simulation Rejection
# =============================================================================

def simulate_choice(grid, r, c, digit, max_depth=10):
    """
    Simulate making a choice and propagating.
    Return True if no contradiction found within max_depth.
    """
    test_grid = [[cell for cell in row] for row in grid]
    test_grid[r][c] = digit

    depth = 0
    changed = True
    while changed and depth < max_depth:
        changed = False
        depth += 1
        for rr in range(9):
            for cc in range(9):
                if test_grid[rr][cc] is None:
                    opts = list(get_available_digits_9x9(test_grid, rr, cc))
                    if not opts:
                        return False  # Contradiction!
                    if len(opts) == 1:
                        test_grid[rr][cc] = opts[0]
                        changed = True

    return True  # No contradiction found


# =============================================================================
# Combined Solver: Template + Simulation
# =============================================================================

def solve_template_simulation(puzzle, store, digit_vecs, complete_template, sim_threshold=0.05):
    """
    Template matching for ordering + simulation for rejection.

    sim_threshold: if top choice is better than second by this margin,
                   skip simulation (template is confident enough)
    """
    grid = [[cell for cell in row] for row in puzzle]
    backtracks = [0]
    simulations = [0]

    def score(g, r, c, d):
        total = 0.0
        row_digits = {g[r][cc] for cc in range(9) if g[r][cc] is not None}
        row_digits.add(d)
        total += similarity(store.bundle([digit_vecs[x] for x in row_digits]), complete_template)

        col_digits = {g[rr][c] for rr in range(9) if g[rr][c] is not None}
        col_digits.add(d)
        total += similarity(store.bundle([digit_vecs[x] for x in col_digits]), complete_template)

        br, bc = (r // 3) * 3, (c // 3) * 3
        block_digits = {g[rr][cc] for rr in range(br, br+3) for cc in range(bc, bc+3)
                        if g[rr][cc] is not None}
        block_digits.add(d)
        total += similarity(store.bundle([digit_vecs[x] for x in block_digits]), complete_template)

        return total

    def rec(g):
        changed = True
        while changed:
            changed = False
            for r in range(9):
                for c in range(9):
                    if g[r][c] is None:
                        opts = list(get_available_digits_9x9(g, r, c))
                        if not opts:
                            return None
                        if len(opts) == 1:
                            g[r][c] = opts[0]
                            changed = True

        if count_empty(g) == 0:
            return g

        best = None
        best_count = 10
        for r in range(9):
            for c in range(9):
                if g[r][c] is None:
                    opts = list(get_available_digits_9x9(g, r, c))
                    if len(opts) < best_count:
                        best_count = len(opts)
                        best = (r, c, opts)

        if best is None:
            return None

        r, c, options = best

        # Score all options
        scores = [(score(g, r, c, d), d) for d in options]
        scores.sort(reverse=True)

        # Check if we need simulation
        use_simulation = True
        if len(scores) >= 2:
            margin = scores[0][0] - scores[1][0]
            if margin > sim_threshold:
                use_simulation = False  # Template is confident

        # Filter by simulation if needed
        if use_simulation:
            filtered = []
            for sc, d in scores:
                simulations[0] += 1
                if simulate_choice(g, r, c, d):
                    filtered.append((sc, d))
            if filtered:
                scores = filtered

        ordered = [x[1] for x in scores]

        for digit in ordered:
            test = [[cell for cell in row] for row in g]
            test[r][c] = digit
            result = rec(test)
            if result:
                return result
            backtracks[0] += 1

        return None

    result = rec(grid)
    return backtracks[0], simulations[0], result


# =============================================================================
# Main Experiment
# =============================================================================

def main():
    print("=" * 70)
    print("EXPERIMENT: Template Matching + Simulation Rejection")
    print("=" * 70)

    store = create_store()
    digit_vecs = {d: store.vector_manager.get_vector(f"d{d}") for d in range(1, 10)}
    complete_template = store.bundle([digit_vecs[d] for d in range(1, 10)])

    # Test template only
    print("\n1. Template Matching Only:")
    start = time.time()
    template_bt, result = solve_template_only(PUZZLE_9x9_HARD, store, digit_vecs, complete_template)
    elapsed = time.time() - start
    valid = result is not None and validate_9x9(result)
    print(f"   Backtracks: {template_bt}")
    print(f"   Time: {elapsed:.3f}s")
    print(f"   Valid: {valid}")

    # Test different thresholds
    print("\n2. Template + Simulation (varying threshold):")
    print(f"   {'Threshold':<12} {'Backtracks':<12} {'Simulations':<12} {'Time':<10}")
    print("   " + "-" * 46)

    results = []
    for threshold in [0.0, 0.02, 0.05, 0.1, 0.2, 1.0]:
        start = time.time()
        bt, sims, result = solve_template_simulation(
            PUZZLE_9x9_HARD, store, digit_vecs, complete_template, sim_threshold=threshold
        )
        elapsed = time.time() - start
        valid = result is not None and validate_9x9(result)
        results.append((threshold, bt, sims, elapsed, valid))
        print(f"   {threshold:<12} {bt:<12} {sims:<12} {elapsed:<10.3f}")

    # Find best
    best = min(results, key=lambda x: x[1])
    print(f"\n   Best: threshold={best[0]}, backtracks={best[1]}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
| Approach | Backtracks | Notes |
|----------|-----------|-------|
| Template only | {template_bt} | Current best |
| Template + Sim (best) | {best[1]} | threshold={best[0]} |

ANALYSIS:
- threshold=0.0: Always simulate (expensive but thorough)
- threshold=1.0: Never simulate (fast but may miss rejections)
- Sweet spot: Where template confidence reduces simulation overhead

KEY INSIGHT:
Simulation catches contradictions that template matching misses.
Template matching orders choices correctly most of the time.
Combining them should give the best of both.

Best result: {best[1]} backtracks with {best[2]} simulations
""")


if __name__ == "__main__":
    main()
