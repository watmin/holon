#!/usr/bin/env python3
"""
Approach 29: Ultimate Combination

Combine ALL our best findings:
1. Template matching (52 backtracks) - base approach
2. Simulation rejection (fast negation)
3. Chain length bonus (opportunistic)
4. Prototype learning (from good/bad paths)

The goal: Beat 52 backtracks!
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


class UltimateSolver:
    """
    Combines:
    1. Template matching for base scoring
    2. Simulation for fast rejection
    3. Chain length for tie-breaking
    4. Learned prototypes for additional signal
    """

    def __init__(self, store):
        self.store = store
        self.digits = {d: store.vector_manager.get_vector(f"d{d}") for d in range(1, 10)}
        self.complete = store.bundle([self.digits[d] for d in range(1, 10)])
        self.backtracks = 0

        # Learned prototypes (will be populated during solving)
        self.good_patterns = []
        self.bad_patterns = []
        self.good_proto = None
        self.bad_proto = None

    def template_score(self, grid, r, c, digit):
        """Score by template matching (Approach 22 style)."""
        row_digits = {grid[r][cc] for cc in range(9) if grid[r][cc] is not None}
        col_digits = {grid[rr][c] for rr in range(9) if grid[rr][c] is not None}
        br, bc = (r // 3) * 3, (c // 3) * 3
        block_digits = {grid[rr][cc] for rr in range(br, br+3)
                        for cc in range(bc, bc+3) if grid[rr][cc] is not None}

        total = 0
        for used in [row_digits, col_digits, block_digits]:
            new_set = used | {digit}
            set_vec = self.store.bundle([self.digits[d] for d in new_set])
            total += similarity(set_vec, self.complete)

        return total

    def simulate_choice(self, grid, r, c, digit):
        """
        Simulate a choice and return:
        - contradicts: bool
        - chain_length: int (forced moves)
        - filled_after: int
        """
        test_grid = [[cell for cell in row] for row in grid]
        test_grid[r][c] = digit

        chain = 0
        changed = True
        while changed:
            changed = False
            for rr in range(9):
                for cc in range(9):
                    if test_grid[rr][cc] is None:
                        opts = list(get_available_digits_9x9(test_grid, rr, cc))
                        if not opts:
                            return True, 0, 0  # Contradiction
                        if len(opts) == 1:
                            test_grid[rr][cc] = opts[0]
                            chain += 1
                            changed = True

        filled = sum(1 for rr in range(9) for cc in range(9) if test_grid[rr][cc] is not None)
        return False, chain, filled

    def encode_context(self, grid, r, c, digit):
        """Encode the context of a choice for prototype matching."""
        # Count constraint saturation
        row_filled = sum(1 for cc in range(9) if grid[r][cc] is not None)
        col_filled = sum(1 for rr in range(9) if grid[rr][c] is not None)
        br, bc = (r // 3) * 3, (c // 3) * 3
        block_filled = sum(1 for rr in range(br, br+3)
                           for cc in range(bc, bc+3) if grid[rr][cc] is not None)

        vecs = [
            self.store.vector_manager.get_vector(f"row_sat_{row_filled}"),
            self.store.vector_manager.get_vector(f"col_sat_{col_filled}"),
            self.store.vector_manager.get_vector(f"block_sat_{block_filled}"),
            self.digits[digit],
        ]
        return self.store.bundle(vecs)

    def prototype_score(self, grid, r, c, digit):
        """Score by prototype matching if available."""
        if self.good_proto is None or self.bad_proto is None:
            return 0

        context = self.encode_context(grid, r, c, digit)
        good_sim = similarity(context, self.good_proto)
        bad_sim = similarity(context, self.bad_proto)

        return good_sim - bad_sim  # Higher = more like good, less like bad

    def combined_score(self, grid, r, c, digit):
        """
        Combine all scoring methods.
        """
        # 1. Check for contradiction (fast reject)
        contradicts, chain, filled = self.simulate_choice(grid, r, c, digit)
        if contradicts:
            return -10000  # Reject immediately

        # 2. Template matching (primary signal)
        template = self.template_score(grid, r, c, digit)

        # 3. Chain length bonus (tie-breaker)
        chain_bonus = chain * 0.01

        # 4. Prototype matching (learned signal)
        proto = self.prototype_score(grid, r, c, digit) * 0.1

        return template + chain_bonus + proto

    def solve(self, puzzle):
        grid = [[cell for cell in row] for row in puzzle]
        return self._solve_rec(grid, depth=0)

    def _solve_rec(self, grid, depth):
        # Propagate
        changed = True
        while changed:
            changed = False
            for r in range(9):
                for c in range(9):
                    if grid[r][c] is None:
                        opts = list(get_available_digits_9x9(grid, r, c))
                        if not opts:
                            return None
                        if len(opts) == 1:
                            grid[r][c] = opts[0]
                            changed = True

        if count_empty(grid) == 0:
            return grid

        # Find MRV
        best = None
        best_count = 10
        for r in range(9):
            for c in range(9):
                if grid[r][c] is None:
                    opts = list(get_available_digits_9x9(grid, r, c))
                    if len(opts) < best_count:
                        best_count = len(opts)
                        best = (r, c, opts)

        if best is None:
            return None

        r, c, options = best

        # Score and order
        scores = [(self.combined_score(grid, r, c, d), d) for d in options]
        scores.sort(reverse=True)
        ordered = [d for score, d in scores if score > -10000]  # Filter contradictions

        if not ordered:
            return None

        for digit in ordered:
            context = self.encode_context(grid, r, c, digit)

            test_grid = [[cell for cell in row] for row in grid]
            test_grid[r][c] = digit

            result = self._solve_rec(test_grid, depth + 1)
            if result is not None:
                # Learn: this was a good choice
                self.good_patterns.append(context)
                if len(self.good_patterns) >= 5:
                    self.good_proto = self.store.prototype(self.good_patterns[-20:], threshold=0.3)
                return result

            # Learn: this was a bad choice
            self.bad_patterns.append(context)
            if len(self.bad_patterns) >= 5:
                self.bad_proto = self.store.prototype(self.bad_patterns[-20:], threshold=0.3)

            self.backtracks += 1

        return None


def main():
    print("=" * 70)
    print("ULTIMATE COMBINATION SOLVER")
    print("=" * 70)
    print("""
Combining:
1. Template matching (52 backtracks baseline)
2. Simulation rejection (fast negation)
3. Chain length bonus (opportunistic)
4. Online prototype learning

Goal: Beat 52 backtracks!
""")

    store = create_store()
    solver = UltimateSolver(store)

    start = time.time()
    result = solver.solve(PUZZLE_9x9_HARD)
    elapsed = time.time() - start

    print(f"Result: {'SOLVED' if result else 'FAILED'}")
    print(f"Backtracks: {solver.backtracks}")
    print(f"Time: {elapsed:.3f}s")
    print(f"\nGood patterns learned: {len(solver.good_patterns)}")
    print(f"Bad patterns learned: {len(solver.bad_patterns)}")

    if result:
        valid = validate_9x9(result)
        print(f"Valid: {valid}")

    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print(f"""
| Solver | Backtracks |
|--------|-----------|
| Standard backtracking | ~2500 |
| Simulation-guided | 249 |
| Template matching | 52 |
| Ultimate combo | {solver.backtracks} |
""")


if __name__ == "__main__":
    main()
