#!/usr/bin/env python3
"""
Approach 39: Multi-Scale Encoding + Decision Chain Encoding

TWO DIRECTIONS TO EXPLORE:

1. MULTI-SCALE ENCODING
   - Cell level: individual placements
   - Constraint level: row/col/block states
   - Band level: top/middle/bottom, left/center/right
   - Full grid: global state

2. SEQUENCE/CHAIN ENCODING
   - Encode decision PATHS, not just states
   - Use ListEncodeMode.CHAINED for sequential dependencies
   - Capture: "this choice led to that choice led to..."
   - Learn patterns of GOOD decision sequences
"""

import sys
sys.path.insert(0, "/home/watmin/work/holon")

from typing import List, Optional, Set, Tuple, Dict
import numpy as np
import time
import random

from holon import CPUStore
from holon.encoder import ListEncodeMode

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
# PART 1: MULTI-SCALE ENCODING
# =============================================================================

def test_multiscale_encoding():
    """
    Encode puzzle at multiple scales simultaneously.
    """
    print("=" * 70)
    print("EXPERIMENT 1: Multi-Scale Encoding")
    print("=" * 70)

    store = create_store()

    digit_vecs = {d: store.vector_manager.get_vector(f"d{d}") for d in range(1, 10)}

    # Scale markers
    cell_scale = store.vector_manager.get_vector("SCALE_CELL")
    constraint_scale = store.vector_manager.get_vector("SCALE_CONSTRAINT")
    band_scale = store.vector_manager.get_vector("SCALE_BAND")
    grid_scale = store.vector_manager.get_vector("SCALE_GRID")

    def encode_at_cell_scale(grid):
        """Individual cell placements."""
        cells = []
        for r in range(9):
            for c in range(9):
                if grid[r][c] is not None:
                    pos = store.vector_manager.get_vector(f"cell_{r}_{c}")
                    cells.append(store.bind(pos, digit_vecs[grid[r][c]]))
        if cells:
            return store.bind(store.bundle(cells), cell_scale)
        return np.zeros(store.dimensions, dtype=np.int8)

    def encode_at_constraint_scale(grid):
        """Row/col/block digit sets."""
        constraints = []
        # Rows
        for r in range(9):
            digits = {grid[r][c] for c in range(9) if grid[r][c] is not None}
            if digits:
                row_marker = store.vector_manager.get_vector(f"row_{r}")
                digit_bundle = store.bundle([digit_vecs[d] for d in digits])
                constraints.append(store.bind(row_marker, digit_bundle))
        # Cols
        for c in range(9):
            digits = {grid[r][c] for r in range(9) if grid[r][c] is not None}
            if digits:
                col_marker = store.vector_manager.get_vector(f"col_{c}")
                digit_bundle = store.bundle([digit_vecs[d] for d in digits])
                constraints.append(store.bind(col_marker, digit_bundle))
        # Blocks
        for b in range(9):
            br, bc = (b // 3) * 3, (b % 3) * 3
            digits = {grid[rr][cc] for rr in range(br, br+3) for cc in range(bc, bc+3)
                      if grid[rr][cc] is not None}
            if digits:
                block_marker = store.vector_manager.get_vector(f"block_{b}")
                digit_bundle = store.bundle([digit_vecs[d] for d in digits])
                constraints.append(store.bind(block_marker, digit_bundle))

        if constraints:
            return store.bind(store.bundle(constraints), constraint_scale)
        return np.zeros(store.dimensions, dtype=np.int8)

    def encode_at_band_scale(grid):
        """Top/middle/bottom row bands, left/center/right column bands."""
        bands = []
        # Row bands (3 rows each)
        for band_idx, rows in enumerate([(0,1,2), (3,4,5), (6,7,8)]):
            digits = set()
            for r in rows:
                for c in range(9):
                    if grid[r][c] is not None:
                        digits.add(grid[r][c])
            if digits:
                band_marker = store.vector_manager.get_vector(f"rowband_{band_idx}")
                digit_bundle = store.bundle([digit_vecs[d] for d in digits])
                bands.append(store.bind(band_marker, digit_bundle))

        # Column bands (3 cols each)
        for band_idx, cols in enumerate([(0,1,2), (3,4,5), (6,7,8)]):
            digits = set()
            for r in range(9):
                for c in cols:
                    if grid[r][c] is not None:
                        digits.add(grid[r][c])
            if digits:
                band_marker = store.vector_manager.get_vector(f"colband_{band_idx}")
                digit_bundle = store.bundle([digit_vecs[d] for d in digits])
                bands.append(store.bind(band_marker, digit_bundle))

        if bands:
            return store.bind(store.bundle(bands), band_scale)
        return np.zeros(store.dimensions, dtype=np.int8)

    def encode_at_grid_scale(grid):
        """Global grid features."""
        # Count of each digit
        digit_counts = {d: 0 for d in range(1, 10)}
        for r in range(9):
            for c in range(9):
                if grid[r][c] is not None:
                    digit_counts[grid[r][c]] += 1

        features = []
        for d, count in digit_counts.items():
            if count > 0:
                count_vec = store.vector_manager.get_vector(f"count_{count}")
                features.append(store.bind(digit_vecs[d], count_vec))

        if features:
            return store.bind(store.bundle(features), grid_scale)
        return np.zeros(store.dimensions, dtype=np.int8)

    def encode_multiscale(grid):
        """Combine all scales into one vector."""
        cell_vec = encode_at_cell_scale(grid)
        constraint_vec = encode_at_constraint_scale(grid)
        band_vec = encode_at_band_scale(grid)
        grid_vec = encode_at_grid_scale(grid)

        return store.bundle([cell_vec, constraint_vec, band_vec, grid_vec])

    # Test: encode puzzle
    puzzle_multiscale = encode_multiscale(PUZZLE_9x9_HARD)
    print(f"\nMulti-scale encoding norm: {np.linalg.norm(puzzle_multiscale):.1f}")

    # Solve puzzle and encode solution
    def solve(puzzle):
        grid = [[cell for cell in row] for row in puzzle]
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
            for r in range(9):
                for c in range(9):
                    if g[r][c] is None:
                        for d in get_available_digits_9x9(g, r, c):
                            test = [[cell for cell in row] for row in g]
                            test[r][c] = d
                            result = rec(test)
                            if result:
                                return result
                        return None
            return None
        return rec(grid)

    solution = solve(PUZZLE_9x9_HARD)
    solution_multiscale = encode_multiscale(solution)

    print(f"Solution multi-scale encoding norm: {np.linalg.norm(solution_multiscale):.1f}")
    print(f"sim(puzzle, solution) at multi-scale: {similarity(puzzle_multiscale, solution_multiscale):.4f}")

    # Check each scale's contribution
    print(f"\nPer-scale similarities (puzzle → solution):")
    print(f"  Cell scale:       {similarity(encode_at_cell_scale(PUZZLE_9x9_HARD), encode_at_cell_scale(solution)):.4f}")
    print(f"  Constraint scale: {similarity(encode_at_constraint_scale(PUZZLE_9x9_HARD), encode_at_constraint_scale(solution)):.4f}")
    print(f"  Band scale:       {similarity(encode_at_band_scale(PUZZLE_9x9_HARD), encode_at_band_scale(solution)):.4f}")
    print(f"  Grid scale:       {similarity(encode_at_grid_scale(PUZZLE_9x9_HARD), encode_at_grid_scale(solution)):.4f}")

    return encode_multiscale


# =============================================================================
# PART 2: DECISION CHAIN ENCODING
# =============================================================================

def test_chain_encoding():
    """
    Encode decision sequences as chains.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Decision Chain Encoding")
    print("=" * 70)

    store = create_store()

    digit_vecs = {d: store.vector_manager.get_vector(f"d{d}") for d in range(1, 10)}
    pos_vecs = {(r, c): store.vector_manager.get_vector(f"p{r}{c}")
                for r in range(9) for c in range(9)}

    def encode_decision(r, c, digit):
        """Encode a single decision."""
        return store.bind(pos_vecs[(r, c)], digit_vecs[digit])

    def encode_decision_chain(decisions: List[Tuple[int, int, int]]):
        """
        Encode a sequence of decisions as a CHAIN.
        Each decision is bound to the previous, creating temporal structure.
        """
        if not decisions:
            return np.zeros(store.dimensions, dtype=np.int8)

        # Method 1: Sequential binding (creates unique signature)
        chain = encode_decision(*decisions[0])
        for r, c, d in decisions[1:]:
            decision = encode_decision(r, c, d)
            chain = store.bind(chain, decision)
        return chain

    def encode_decision_chain_bundled(decisions: List[Tuple[int, int, int]]):
        """
        Encode decisions with position markers then bundle.
        Preserves all decisions but marks their order.
        """
        if not decisions:
            return np.zeros(store.dimensions, dtype=np.int8)

        components = []
        for i, (r, c, d) in enumerate(decisions):
            decision = encode_decision(r, c, d)
            order_marker = store.vector_manager.get_vector(f"order_{i}")
            components.append(store.bind(decision, order_marker))

        return store.bundle(components)

    # Generate a solving trace
    def solve_with_trace(puzzle):
        """Solve and record the decision sequence."""
        grid = [[cell for cell in row] for row in puzzle]
        trace = []

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
                                trace.append((r, c, opts[0], "forced"))
                                changed = True

            if count_empty(g) == 0:
                return g

            for r in range(9):
                for c in range(9):
                    if g[r][c] is None:
                        opts = list(get_available_digits_9x9(g, r, c))
                        for d in opts:
                            test = [[cell for cell in row] for row in g]
                            test[r][c] = d
                            trace.append((r, c, d, "choice"))
                            result = rec(test)
                            if result:
                                return result
                            trace.append((r, c, d, "backtrack"))
                        return None
            return None

        solution = rec(grid)
        return solution, trace

    solution, trace = solve_with_trace(PUZZLE_9x9_HARD)

    print(f"\nSolving trace: {len(trace)} events")
    forced = sum(1 for t in trace if t[3] == "forced")
    choices = sum(1 for t in trace if t[3] == "choice")
    backtracks = sum(1 for t in trace if t[3] == "backtrack")
    print(f"  Forced: {forced}, Choices: {choices}, Backtracks: {backtracks}")

    # Extract just the successful path (no backtracks)
    successful_path = []
    for r, c, d, event in trace:
        if event == "backtrack":
            if successful_path and successful_path[-1][:3] == (r, c, d):
                successful_path.pop()
        else:
            successful_path.append((r, c, d))

    print(f"  Successful path length: {len(successful_path)}")

    # Encode the successful path
    chain_vec = encode_decision_chain(successful_path[:20])  # First 20 decisions
    bundled_vec = encode_decision_chain_bundled(successful_path[:20])

    print(f"\nEncoded chain (first 20 decisions):")
    print(f"  Chain (bound) norm: {np.linalg.norm(chain_vec):.1f}")
    print(f"  Bundled norm: {np.linalg.norm(bundled_vec):.1f}")
    print(f"  sim(chain, bundled): {similarity(chain_vec, bundled_vec):.4f}")

    return successful_path


# =============================================================================
# PART 3: Chain Pattern Learning
# =============================================================================

def test_chain_pattern_learning():
    """
    Learn patterns from decision chains.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Chain Pattern Learning")
    print("=" * 70)

    store = create_store()

    digit_vecs = {d: store.vector_manager.get_vector(f"d{d}") for d in range(1, 10)}
    pos_vecs = {(r, c): store.vector_manager.get_vector(f"p{r}{c}")
                for r in range(9) for c in range(9)}

    def encode_decision(r, c, digit):
        return store.bind(pos_vecs[(r, c)], digit_vecs[digit])

    # Solve multiple puzzles to collect chains
    # (We'll just use variations of our puzzle for demo)
    def solve_and_get_path(puzzle):
        grid = [[cell for cell in row] for row in puzzle]
        path = []

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
                                path.append((r, c, opts[0]))
                                changed = True

            if count_empty(g) == 0:
                return g

            for r in range(9):
                for c in range(9):
                    if g[r][c] is None:
                        opts = list(get_available_digits_9x9(g, r, c))
                        for d in opts:
                            test = [[cell for cell in row] for row in g]
                            test[r][c] = d
                            path.append((r, c, d))
                            result = rec(test)
                            if result:
                                return result
                            path.pop()
                        return None
            return None

        rec(grid)
        return path

    # Collect paths from solving
    path = solve_and_get_path(PUZZLE_9x9_HARD)
    print(f"Collected path with {len(path)} decisions")

    # Encode subsequences (n-grams of decisions)
    def encode_ngram(path, start, n):
        """Encode n consecutive decisions."""
        if start + n > len(path):
            return None
        components = []
        for i in range(n):
            r, c, d = path[start + i]
            decision = encode_decision(r, c, d)
            order = store.vector_manager.get_vector(f"pos_{i}")
            components.append(store.bind(decision, order))
        return store.bundle(components)

    # Create prototype from all 3-grams
    n = 3
    ngrams = []
    for i in range(len(path) - n + 1):
        ngram = encode_ngram(path, i, n)
        if ngram is not None:
            ngrams.append(ngram)

    if ngrams:
        # Use prototype to find common patterns
        prototype = store.prototype(ngrams)

        print(f"\nCreated prototype from {len(ngrams)} {n}-grams")
        print(f"Prototype norm: {np.linalg.norm(prototype):.1f}")

        # Check which n-grams are most similar to prototype
        sims = [(similarity(ng, prototype), i) for i, ng in enumerate(ngrams)]
        sims.sort(reverse=True)

        print(f"\nTop 5 n-grams most similar to prototype:")
        for sim, i in sims[:5]:
            segment = path[i:i+n]
            print(f"  sim={sim:.3f}: {segment}")


# =============================================================================
# PART 4: Multi-Scale + Chain Solver
# =============================================================================

def test_multiscale_chain_solver():
    """
    Combine multi-scale encoding with chain-based scoring.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Multi-Scale + Chain Solver")
    print("=" * 70)

    store = create_store()

    digit_vecs = {d: store.vector_manager.get_vector(f"d{d}") for d in range(1, 10)}

    # Complete template for constraint scoring
    complete_template = store.bundle([digit_vecs[d] for d in range(1, 10)])

    # Scale weights (learned or tuned)
    scale_weights = {
        "constraint": 1.0,  # Main signal
        "cell": 0.1,        # Minor signal
        "chain": 0.5,       # Chain continuation
    }

    class MultiScaleChainSolver:
        def __init__(self):
            self.backtracks = 0
            self.decision_history = []  # Track decisions made

        def score_constraint_scale(self, grid, r, c, digit):
            """Score based on constraint satisfaction (template matching)."""
            total = 0.0

            # Row
            row_digits = {grid[r][cc] for cc in range(9) if grid[r][cc] is not None}
            row_digits.add(digit)
            row_vec = store.bundle([digit_vecs[d] for d in row_digits])
            total += similarity(row_vec, complete_template)

            # Column
            col_digits = {grid[rr][c] for rr in range(9) if grid[rr][c] is not None}
            col_digits.add(digit)
            col_vec = store.bundle([digit_vecs[d] for d in col_digits])
            total += similarity(col_vec, complete_template)

            # Block
            br, bc = (r // 3) * 3, (c // 3) * 3
            block_digits = {grid[rr][cc] for rr in range(br, br+3) for cc in range(bc, bc+3)
                            if grid[rr][cc] is not None}
            block_digits.add(digit)
            block_vec = store.bundle([digit_vecs[d] for d in block_digits])
            total += similarity(block_vec, complete_template)

            return total

        def score_chain_continuation(self, r, c, digit):
            """Score based on continuation of decision chain."""
            if len(self.decision_history) < 2:
                return 0.0

            # Recent decisions
            recent = self.decision_history[-3:]

            # Does this decision "continue" the pattern?
            # E.g., same row, same column, same block as recent decisions
            continuation_score = 0.0

            for prev_r, prev_c, prev_d in recent:
                # Same row continuation
                if r == prev_r:
                    continuation_score += 0.2
                # Same column continuation
                if c == prev_c:
                    continuation_score += 0.2
                # Same block continuation
                if (r // 3 == prev_r // 3) and (c // 3 == prev_c // 3):
                    continuation_score += 0.1
                # Consecutive digits
                if abs(digit - prev_d) == 1:
                    continuation_score += 0.1

            return continuation_score / len(recent)

        def score_choice(self, grid, r, c, digit):
            """Combined multi-scale score."""
            constraint_score = self.score_constraint_scale(grid, r, c, digit)
            chain_score = self.score_chain_continuation(r, c, digit)

            return (scale_weights["constraint"] * constraint_score +
                    scale_weights["chain"] * chain_score)

        def solve(self, puzzle):
            grid = [[cell for cell in row] for row in puzzle]
            self.decision_history = []
            return self._solve_rec(grid)

        def _solve_rec(self, grid):
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
                                self.decision_history.append((r, c, opts[0]))
                                changed = True

            if count_empty(grid) == 0:
                return grid

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

            scores = [(self.score_choice(grid, r, c, d), d) for d in options]
            scores.sort(reverse=True)
            ordered = [d for _, d in scores]

            for digit in ordered:
                test_grid = [[cell for cell in row] for row in grid]
                test_grid[r][c] = digit
                history_len = len(self.decision_history)
                self.decision_history.append((r, c, digit))

                result = self._solve_rec(test_grid)
                if result is not None:
                    return result

                self.decision_history = self.decision_history[:history_len]
                self.backtracks += 1

            return None

    solver = MultiScaleChainSolver()
    start = time.time()
    result = solver.solve(PUZZLE_9x9_HARD)
    elapsed = time.time() - start

    print(f"\nMulti-Scale + Chain solver:")
    print(f"  Result: {'SOLVED' if result else 'FAILED'}")
    print(f"  Backtracks: {solver.backtracks}")
    print(f"  Time: {elapsed:.3f}s")

    if result and validate_9x9(result):
        print("  Valid solution!")

    return solver.backtracks


# =============================================================================
# MAIN
# =============================================================================

def main():
    encode_multiscale = test_multiscale_encoding()
    successful_path = test_chain_encoding()
    test_chain_pattern_learning()
    bt = test_multiscale_chain_solver()

    print("\n" + "=" * 70)
    print("MULTI-SCALE + CHAIN SUMMARY")
    print("=" * 70)
    print(f"""
MULTI-SCALE ENCODING:
Different scales capture different aspects:
- Cell scale: exact placements
- Constraint scale: digit sets (what template matching uses!)
- Band scale: broader region patterns
- Grid scale: global digit distribution

CHAIN ENCODING:
Decision sequences as temporal patterns:
- Sequential binding creates unique path signatures
- Bundled with order markers preserves all decisions
- N-grams can capture local decision patterns
- Prototype extracts common patterns

SOLVER RESULT:
Multi-Scale + Chain: {bt} backtracks
Template matching baseline: 52 backtracks

KEY INSIGHT:
Chain continuation scoring adds a "momentum" heuristic:
- Prefer choices that continue recent patterns
- Same row/col/block = continuing a "thread"
- This is like "opportunistic" solving we found earlier!

WHAT THIS REVEALS:
The constraint scale IS the most informative - it's what
template matching uses. Other scales add minor signal.

Chain encoding captures PROCESS, not just STATE.
This could be powerful for learning from solved puzzles.
""")


if __name__ == "__main__":
    main()
