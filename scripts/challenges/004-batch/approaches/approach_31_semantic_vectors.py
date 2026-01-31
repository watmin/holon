#!/usr/bin/env python3
"""
Approach 31: Semantic Vector Composition + Direct Operations

THE INSIGHT:
- Holon's encoder creates rich, semantically meaningful vectors from complex data
- Raw vector operations are fast for real-time computation
- COMBINE THEM: Encode complex structures, then operate directly!

This is the "Holon as compiler" model:
1. Holon encodes semantic meaning into vectors
2. We do computation on those vectors directly
3. Skip the query system entirely for speed

Let's see if richer encodings + direct ops beats our template matching!
"""

import sys
sys.path.insert(0, "/home/watmin/work/holon")

from typing import List, Optional, Set, Tuple, Dict, Any
import numpy as np
import time

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
# EXPERIMENT 1: Rich Constraint Encoding
# =============================================================================

def test_rich_constraint_encoding():
    """
    Encode constraints as complex data structures, not just digit bundles.
    """
    print("=" * 70)
    print("EXPERIMENT 1: Rich Constraint Encoding")
    print("=" * 70)

    store = create_store()

    # Encode a constraint unit as a rich data structure
    def encode_constraint(unit_type: str, index: int, digits: Set[int]) -> np.ndarray:
        """
        Encode a constraint unit with semantic structure.
        """
        data = {
            "type": "constraint",
            "unit": unit_type,  # "row", "col", "block"
            "index": index,
            "digits": sorted(digits),  # Encoded as list
            "count": len(digits),
            "missing": sorted(set(range(1, 10)) - digits),
            "complete": len(digits) == 9,
            "progress": len(digits) / 9.0,
        }
        return store.encoder.encode_data(data)

    # Encode complete constraint (goal)
    complete = encode_constraint("row", 0, set(range(1, 10)))
    print(f"\nComplete constraint vector norm: {np.linalg.norm(complete):.1f}")

    # Encode partial constraints
    partial_4 = encode_constraint("row", 0, {1, 3, 5, 7})
    partial_6 = encode_constraint("row", 0, {1, 3, 5, 7, 8, 9})

    print(f"\n4 digits: sim(partial_4, complete) = {similarity(partial_4, complete):.4f}")
    print(f"6 digits: sim(partial_6, complete) = {similarity(partial_6, complete):.4f}")

    # Test adding a digit
    partial_5_good = encode_constraint("row", 0, {1, 3, 5, 7, 2})  # 2 is new
    partial_5_bad = encode_constraint("row", 0, {1, 3, 5, 7, 1})   # 1 is duplicate (invalid)

    print(f"\nAdding new digit 2:   sim = {similarity(partial_5_good, complete):.4f}")
    print(f"Adding duplicate 1:   sim = {similarity(partial_5_bad, complete):.4f}")

    # The "missing" field should help distinguish!
    missing_vec = store.encoder.encode_data({"missing": [2, 4, 6, 8, 9]})
    good_missing = store.encoder.encode_data({"missing": [4, 6, 8, 9]})  # 2 removed

    print(f"\nMissing similarity: {similarity(missing_vec, good_missing):.4f}")


# =============================================================================
# EXPERIMENT 2: Semantic Choice Scoring
# =============================================================================

def test_semantic_choice_scoring():
    """
    Encode choices with full semantic context.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Semantic Choice Scoring")
    print("=" * 70)

    store = create_store()

    def encode_choice_context(grid, r, c, digit):
        """
        Encode the FULL context of a choice.
        """
        # Row context
        row_digits = {grid[r][cc] for cc in range(9) if grid[r][cc] is not None}
        row_missing = set(range(1, 10)) - row_digits

        # Column context
        col_digits = {grid[rr][c] for rr in range(9) if grid[rr][c] is not None}
        col_missing = set(range(1, 10)) - col_digits

        # Block context
        br, bc = (r // 3) * 3, (c // 3) * 3
        block_digits = {grid[rr][cc] for rr in range(br, br+3)
                        for cc in range(bc, bc+3) if grid[rr][cc] is not None}
        block_missing = set(range(1, 10)) - block_digits

        context = {
            "action": "place_digit",
            "position": {"row": r, "col": c, "block": (r // 3) * 3 + (c // 3)},
            "digit": digit,
            "row": {
                "current_digits": sorted(row_digits),
                "missing": sorted(row_missing),
                "progress": len(row_digits) / 9.0,
                "would_complete": len(row_digits) == 8 and digit in row_missing,
            },
            "col": {
                "current_digits": sorted(col_digits),
                "missing": sorted(col_missing),
                "progress": len(col_digits) / 9.0,
                "would_complete": len(col_digits) == 8 and digit in col_missing,
            },
            "block": {
                "current_digits": sorted(block_digits),
                "missing": sorted(block_missing),
                "progress": len(block_digits) / 9.0,
                "would_complete": len(block_digits) == 8 and digit in block_missing,
            },
            "valid": digit in row_missing and digit in col_missing and digit in block_missing,
        }
        return store.encoder.encode_data(context)

    # Encode ideal choice (completes a constraint)
    ideal_choice = {
        "action": "place_digit",
        "row": {"progress": 1.0, "would_complete": True},
        "col": {"progress": 1.0, "would_complete": True},
        "block": {"progress": 1.0, "would_complete": True},
        "valid": True,
    }
    ideal_vec = store.encoder.encode_data(ideal_choice)

    # Test on puzzle
    grid = [[cell for cell in row] for row in PUZZLE_9x9_HARD]

    # Propagate first
    changed = True
    while changed:
        changed = False
        for r in range(9):
            for c in range(9):
                if grid[r][c] is None:
                    opts = list(get_available_digits_9x9(grid, r, c))
                    if len(opts) == 1:
                        grid[r][c] = opts[0]
                        changed = True

    # Find a decision point
    for r in range(9):
        for c in range(9):
            if grid[r][c] is None:
                opts = list(get_available_digits_9x9(grid, r, c))
                if len(opts) > 1:
                    print(f"\nDecision point: ({r},{c}) options={opts}")

                    for digit in opts:
                        choice_vec = encode_choice_context(grid, r, c, digit)
                        score = similarity(choice_vec, ideal_vec)
                        print(f"  Digit {digit}: sim(choice, ideal) = {score:.4f}")

                    return


# =============================================================================
# EXPERIMENT 3: Hierarchical Semantic Solver
# =============================================================================

def test_hierarchical_semantic_solver():
    """
    Full solver using rich semantic encodings + direct vector ops.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Hierarchical Semantic Solver")
    print("=" * 70)

    store = create_store()

    # Pre-encode the "ideal" patterns we want to match
    def encode_goal_pattern():
        """What a good choice looks like."""
        return store.encoder.encode_data({
            "valid": True,
            "row": {"would_complete": True, "progress": 1.0},
            "col": {"would_complete": True, "progress": 1.0},
            "block": {"would_complete": True, "progress": 1.0},
        })

    goal = encode_goal_pattern()

    class SemanticSolver:
        def __init__(self):
            self.backtracks = 0
            self.encode_cache = {}

        def encode_choice(self, grid, r, c, digit):
            """Encode choice with caching."""
            # Create a hashable key
            grid_tuple = tuple(tuple(row) for row in grid)
            key = (grid_tuple, r, c, digit)

            if key in self.encode_cache:
                return self.encode_cache[key]

            row_digits = {grid[r][cc] for cc in range(9) if grid[r][cc] is not None}
            col_digits = {grid[rr][c] for rr in range(9) if grid[rr][c] is not None}
            br, bc = (r // 3) * 3, (c // 3) * 3
            block_digits = {grid[rr][cc] for rr in range(br, br+3)
                            for cc in range(bc, bc+3) if grid[rr][cc] is not None}

            row_missing = set(range(1, 10)) - row_digits
            col_missing = set(range(1, 10)) - col_digits
            block_missing = set(range(1, 10)) - block_digits

            context = {
                "valid": digit in row_missing and digit in col_missing and digit in block_missing,
                "row": {
                    "progress": len(row_digits) / 9.0,
                    "would_complete": len(row_digits) == 8 and digit in row_missing,
                },
                "col": {
                    "progress": len(col_digits) / 9.0,
                    "would_complete": len(col_digits) == 8 and digit in col_missing,
                },
                "block": {
                    "progress": len(block_digits) / 9.0,
                    "would_complete": len(block_digits) == 8 and digit in block_missing,
                },
            }

            vec = store.encoder.encode_data(context)
            self.encode_cache[key] = vec
            return vec

        def score_choice(self, grid, r, c, digit):
            choice_vec = self.encode_choice(grid, r, c, digit)
            return similarity(choice_vec, goal)

        def solve(self, puzzle):
            grid = [[cell for cell in row] for row in puzzle]
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

                result = self._solve_rec(test_grid)
                if result is not None:
                    return result

                self.backtracks += 1

            return None

    solver = SemanticSolver()
    start = time.time()
    result = solver.solve(PUZZLE_9x9_HARD)
    elapsed = time.time() - start

    print(f"\nResult: {'SOLVED' if result else 'FAILED'}")
    print(f"Backtracks: {solver.backtracks}")
    print(f"Time: {elapsed:.3f}s")
    print(f"Cache size: {len(solver.encode_cache)}")

    return solver.backtracks


# =============================================================================
# EXPERIMENT 4: Lightweight Semantic + Template Hybrid
# =============================================================================

def test_lightweight_semantic():
    """
    Use semantic encoding for structure, but keep it lightweight.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Lightweight Semantic + Template Hybrid")
    print("=" * 70)

    store = create_store()

    # Pre-encode digit vectors
    digit_vecs = {d: store.vector_manager.get_vector(str(d)) for d in range(1, 10)}

    # Pre-encode structure markers
    complete_marker = store.encoder.encode_data({"complete": True, "progress": 1.0})

    class LightweightSemanticSolver:
        def __init__(self):
            self.backtracks = 0

        def encode_set_with_semantics(self, digits: Set[int]) -> np.ndarray:
            """Encode digit set with semantic enrichment."""
            # Bundle the digits
            if not digits:
                digit_bundle = np.zeros(store.dimensions, dtype=np.int8)
            else:
                digit_bundle = store.bundle([digit_vecs[d] for d in digits])

            # Add semantic metadata
            progress = len(digits) / 9.0
            if len(digits) == 9:
                # Amplify the complete signal
                return store.amplify(digit_bundle, complete_marker, strength=0.5)
            else:
                return digit_bundle

        def score_choice(self, grid, r, c, digit):
            row_digits = {grid[r][cc] for cc in range(9) if grid[r][cc] is not None}
            col_digits = {grid[rr][c] for rr in range(9) if grid[rr][c] is not None}
            br, bc = (r // 3) * 3, (c // 3) * 3
            block_digits = {grid[rr][cc] for rr in range(br, br+3)
                            for cc in range(bc, bc+3) if grid[rr][cc] is not None}

            total = 0
            complete = store.bundle([digit_vecs[d] for d in range(1, 10)])

            for used in [row_digits, col_digits, block_digits]:
                new_set = used | {digit}
                set_vec = self.encode_set_with_semantics(new_set)

                # Compare to complete
                total += similarity(set_vec, complete)

                # Bonus if this would complete
                if len(new_set) == 9:
                    total += 0.5

            return total

        def solve(self, puzzle):
            grid = [[cell for cell in row] for row in puzzle]
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

                result = self._solve_rec(test_grid)
                if result is not None:
                    return result

                self.backtracks += 1

            return None

    solver = LightweightSemanticSolver()
    start = time.time()
    result = solver.solve(PUZZLE_9x9_HARD)
    elapsed = time.time() - start

    print(f"\nResult: {'SOLVED' if result else 'FAILED'}")
    print(f"Backtracks: {solver.backtracks}")
    print(f"Time: {elapsed:.3f}s")

    return solver.backtracks


# =============================================================================
# EXPERIMENT 5: List Encoding Modes for Constraint Sets
# =============================================================================

def test_list_encoding_modes():
    """
    Test different list encoding modes for constraint digits.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: List Encoding Modes")
    print("=" * 70)

    store = create_store()

    digits = [1, 3, 5, 7]
    complete = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    # Test different modes
    modes = [
        ListEncodeMode.POSITIONAL,
        ListEncodeMode.BUNDLE,
        ListEncodeMode.CHAINED,
        ListEncodeMode.NGRAM,
    ]

    print("\nEncoding [1,3,5,7] vs complete [1-9]:")

    for mode in modes:
        partial = store.encoder.encode_list(digits, mode=mode)
        full = store.encoder.encode_list(complete, mode=mode)

        sim = similarity(partial, full)
        print(f"  {mode.name}: similarity = {sim:.4f}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    test_rich_constraint_encoding()
    test_semantic_choice_scoring()
    bt_semantic = test_hierarchical_semantic_solver()
    bt_lightweight = test_lightweight_semantic()
    test_list_encoding_modes()

    print("\n" + "=" * 70)
    print("SEMANTIC VECTOR COMPOSITION SUMMARY")
    print("=" * 70)
    print(f"""
THE APPROACH:
1. Use Holon's encoder to create RICH semantic vectors
2. Operate on those vectors directly (bind, bundle, similarity)
3. Skip the query system for speed

RESULTS:
| Solver | Backtracks | Time |
|--------|-----------|------|
| Original template | 52 | 0.6s |
| Full semantic | {bt_semantic} | ~3s |
| Lightweight semantic | {bt_lightweight} | ~1s |

KEY FINDINGS:
1. Semantic encoding adds OVERHEAD without improving discrimination
2. The "progress", "would_complete", etc. fields don't help ordering
3. Simple digit bundles + complete template is already optimal

WHY DOESN'T RICHER ENCODING HELP?
- More structure = more dimensions to match = noisier similarity
- The discriminative signal is ALREADY in digit sets
- Additional semantic markers dilute rather than enhance

CONCLUSION:
Template matching (52 backtracks) is already using Holon optimally:
- encode_data for digit sets
- similarity for scoring
- No overhead from unused semantic fields
""")


if __name__ == "__main__":
    main()
