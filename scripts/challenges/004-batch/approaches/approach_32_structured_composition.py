#!/usr/bin/env python3
"""
Approach 32: Structured Composition with Position-Value Mapping

THE USER'S IDEA:
1. Encode each constraint unit as: {0: val, 1: val, 2: val, ...}
2. Tag with orientation: {orientation: "row", index: 0, values: {...}}
3. Use $any for unknown positions
4. Use $not to negate illegal values
5. Pre-compute viable states and query against them

This is DIFFERENT from what we've been doing:
- We bundled digits: bundle([d1, d3, d5])
- User suggests structured dicts with position keys

Let's implement and test this!
"""

import sys
sys.path.insert(0, "/home/watmin/work/holon")

from typing import List, Optional, Set, Tuple, Dict, Any
import numpy as np
import time
import json

from holon import CPUStore, HolonClient
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
# EXPERIMENT 1: Position-Value Dict Encoding
# =============================================================================

def test_position_value_encoding():
    """
    Encode rows as {position: value} dicts.
    """
    print("=" * 70)
    print("EXPERIMENT 1: Position-Value Dict Encoding")
    print("=" * 70)

    store = create_store()

    # Encode a row as {0: digit, 1: digit, ...}
    def encode_row_as_dict(row_values: List[Optional[int]]) -> Dict:
        """Convert row to position-value dict."""
        return {str(i): v for i, v in enumerate(row_values) if v is not None}

    def encode_row_vector(row_values: List[Optional[int]]) -> np.ndarray:
        """Encode row dict to vector."""
        row_dict = encode_row_as_dict(row_values)
        if not row_dict:
            return np.zeros(store.dimensions, dtype=np.int8)
        return store.encoder.encode_data(row_dict)

    # Test: complete row
    complete_row = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    complete_vec = encode_row_vector(complete_row)

    print(f"\nComplete row: {encode_row_as_dict(complete_row)}")
    print(f"Vector norm: {np.linalg.norm(complete_vec):.1f}")

    # Test: partial rows
    partial_4 = [1, None, 3, None, 5, None, 7, None, None]
    partial_6 = [1, 2, 3, None, 5, 6, 7, None, None]

    vec_4 = encode_row_vector(partial_4)
    vec_6 = encode_row_vector(partial_6)

    print(f"\nPartial (4 filled): {encode_row_as_dict(partial_4)}")
    print(f"  sim(partial_4, complete) = {similarity(vec_4, complete_vec):.4f}")

    print(f"\nPartial (6 filled): {encode_row_as_dict(partial_6)}")
    print(f"  sim(partial_6, complete) = {similarity(vec_6, complete_vec):.4f}")

    # Test: adding a digit
    partial_5_good = [1, 2, 3, None, 5, None, 7, None, None]  # Added 2
    partial_5_bad = [1, None, 3, None, 5, None, 7, None, 1]   # Added duplicate 1

    vec_5_good = encode_row_vector(partial_5_good)
    vec_5_bad = encode_row_vector(partial_5_bad)

    print(f"\nAdding digit 2 (good):   sim = {similarity(vec_5_good, complete_vec):.4f}")
    print(f"Adding duplicate 1 (bad): sim = {similarity(vec_5_bad, complete_vec):.4f}")


# =============================================================================
# EXPERIMENT 2: Orientation-Tagged Encoding
# =============================================================================

def test_orientation_encoding():
    """
    Encode with orientation tag: {orientation: "row", index: 0, values: {...}}
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Orientation-Tagged Encoding")
    print("=" * 70)

    store = create_store()

    def encode_constraint_unit(orientation: str, index: int, values: Dict[int, int]) -> np.ndarray:
        """
        Encode a constraint unit with orientation tag.

        orientation: "row", "col", or "block"
        index: 0-8
        values: {position: digit}
        """
        data = {
            "orientation": orientation,
            "index": index,
            "values": {str(k): v for k, v in values.items()},
            "count": len(values),
        }
        return store.encoder.encode_data(data)

    # Complete row template
    complete_row = encode_constraint_unit("row", 0, {i: i+1 for i in range(9)})

    # Complete column template (same digits, different orientation)
    complete_col = encode_constraint_unit("col", 0, {i: i+1 for i in range(9)})

    print(f"\nRow vs Column (same digits, different orientation):")
    print(f"  sim(row, col) = {similarity(complete_row, complete_col):.4f}")

    # Partial row
    partial_row = encode_constraint_unit("row", 0, {0: 1, 2: 3, 4: 5, 6: 7})

    print(f"\nPartial row {'{0:1, 2:3, 4:5, 6:7}'}:")
    print(f"  sim(partial, complete_row) = {similarity(partial_row, complete_row):.4f}")

    # Different row index
    row_0 = encode_constraint_unit("row", 0, {0: 1, 1: 2, 2: 3})
    row_5 = encode_constraint_unit("row", 5, {0: 1, 1: 2, 2: 3})

    print(f"\nSame values, different row index:")
    print(f"  sim(row_0, row_5) = {similarity(row_0, row_5):.4f}")


# =============================================================================
# EXPERIMENT 3: Pre-compute Valid States and Query
# =============================================================================

def test_precompute_and_query():
    """
    Pre-compute all valid constraint states and query against them.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Pre-compute Valid States")
    print("=" * 70)

    store = create_store()
    client = HolonClient(local_store=store)

    # Pre-compute valid partial row states
    # A valid partial row has unique digits in each position

    print("\nPre-computing valid row states...")

    from itertools import permutations

    # We can't store all 9! = 362880 complete rows
    # But we can store patterns for "N positions filled"

    valid_patterns = []

    # Store patterns for 3-digit rows (more manageable)
    for positions in [(0, 1, 2), (3, 4, 5), (6, 7, 8)]:
        for digits in permutations(range(1, 10), 3):
            pattern = {
                "type": "valid_row_partial",
                "positions": list(positions),
                "values": {str(positions[i]): digits[i] for i in range(3)},
                "digit_set": sorted(digits),
            }
            valid_patterns.append(pattern)

    print(f"Generated {len(valid_patterns)} valid 3-digit patterns")

    # Insert a sample
    for p in valid_patterns[:100]:
        client.insert_json(p)

    print(f"Inserted 100 sample patterns")

    # Query: find valid patterns similar to current state
    current_state = {
        "type": "valid_row_partial",
        "positions": [0, 1, 2],
        "values": {"0": 5, "1": 3, "2": 7},
        "digit_set": [3, 5, 7],
    }

    results = client.search_json(probe=current_state, limit=5, threshold=0.3)

    print(f"\nQuery for state with digits [3,5,7] at positions [0,1,2]:")
    for r in results[:3]:
        print(f"  Score {r['score']:.3f}: {r['data']['values']}")


# =============================================================================
# EXPERIMENT 4: Composite Puzzle Encoding
# =============================================================================

def test_composite_puzzle():
    """
    Encode entire puzzle as composition of row, column, block units.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Composite Puzzle Encoding")
    print("=" * 70)

    store = create_store()

    def encode_puzzle_composite(grid: List[List[Optional[int]]]) -> Dict[str, np.ndarray]:
        """
        Encode puzzle as composite of all constraint units.
        Returns dict of vectors for each unit.
        """
        result = {}

        # Encode all 9 rows
        for r in range(9):
            row_values = {c: grid[r][c] for c in range(9) if grid[r][c] is not None}
            data = {"orientation": "row", "index": r, "values": {str(k): v for k, v in row_values.items()}}
            result[f"row_{r}"] = store.encoder.encode_data(data)

        # Encode all 9 columns
        for c in range(9):
            col_values = {r: grid[r][c] for r in range(9) if grid[r][c] is not None}
            data = {"orientation": "col", "index": c, "values": {str(k): v for k, v in col_values.items()}}
            result[f"col_{c}"] = store.encoder.encode_data(data)

        # Encode all 9 blocks
        for b in range(9):
            br, bc = (b // 3) * 3, (b % 3) * 3
            block_values = {}
            for i, r in enumerate(range(br, br + 3)):
                for j, c in enumerate(range(bc, bc + 3)):
                    if grid[r][c] is not None:
                        pos = i * 3 + j
                        block_values[pos] = grid[r][c]
            data = {"orientation": "block", "index": b, "values": {str(k): v for k, v in block_values.items()}}
            result[f"block_{b}"] = store.encoder.encode_data(data)

        return result

    # Encode complete template (all 9 digits in proper positions)
    complete_row_template = store.encoder.encode_data({
        "orientation": "row",
        "values": {str(i): i+1 for i in range(9)}
    })

    # Encode puzzle
    puzzle_vecs = encode_puzzle_composite(PUZZLE_9x9_HARD)

    print(f"\nPuzzle composite encoding: {len(puzzle_vecs)} vectors")

    # Check similarity of each row to complete
    print("\nRow similarities to complete template:")
    for r in range(9):
        sim = similarity(puzzle_vecs[f"row_{r}"], complete_row_template)
        filled = sum(1 for c in range(9) if PUZZLE_9x9_HARD[r][c] is not None)
        print(f"  Row {r} ({filled}/9 filled): sim = {sim:.4f}")


# =============================================================================
# EXPERIMENT 5: Structured Solver with Negation
# =============================================================================

def test_structured_solver():
    """
    Solver using structured encoding + direct vector ops.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: Structured Solver")
    print("=" * 70)

    store = create_store()

    # Pre-encode complete templates for each orientation
    complete_row = store.encoder.encode_data({
        "orientation": "row",
        "values": {str(i): i+1 for i in range(9)}
    })
    complete_col = store.encoder.encode_data({
        "orientation": "col",
        "values": {str(i): i+1 for i in range(9)}
    })
    complete_block = store.encoder.encode_data({
        "orientation": "block",
        "values": {str(i): i+1 for i in range(9)}
    })

    class StructuredSolver:
        def __init__(self):
            self.backtracks = 0

        def encode_constraint(self, orientation: str, values: Dict[int, int]) -> np.ndarray:
            data = {
                "orientation": orientation,
                "values": {str(k): v for k, v in values.items()}
            }
            return store.encoder.encode_data(data)

        def score_choice(self, grid, r, c, digit):
            # Get row values after adding digit
            row_values = {cc: grid[r][cc] for cc in range(9) if grid[r][cc] is not None}
            row_values[c] = digit
            row_vec = self.encode_constraint("row", row_values)

            # Get column values
            col_values = {rr: grid[rr][c] for rr in range(9) if grid[rr][c] is not None}
            col_values[r] = digit
            col_vec = self.encode_constraint("col", col_values)

            # Get block values
            br, bc = (r // 3) * 3, (c // 3) * 3
            block_values = {}
            for i, rr in enumerate(range(br, br + 3)):
                for j, cc in enumerate(range(bc, bc + 3)):
                    if grid[rr][cc] is not None:
                        block_values[i * 3 + j] = grid[rr][cc]
            local_pos = (r - br) * 3 + (c - bc)
            block_values[local_pos] = digit
            block_vec = self.encode_constraint("block", block_values)

            # Score: sum of similarities to complete templates
            score = (
                similarity(row_vec, complete_row) +
                similarity(col_vec, complete_col) +
                similarity(block_vec, complete_block)
            )

            return score

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

    solver = StructuredSolver()
    start = time.time()
    result = solver.solve(PUZZLE_9x9_HARD)
    elapsed = time.time() - start

    print(f"\nResult: {'SOLVED' if result else 'FAILED'}")
    print(f"Backtracks: {solver.backtracks}")
    print(f"Time: {elapsed:.3f}s")

    return solver.backtracks


# =============================================================================
# EXPERIMENT 6: Compare Encoding Approaches
# =============================================================================

def test_compare_encodings():
    """
    Compare structured vs flat encoding approaches.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 6: Encoding Comparison")
    print("=" * 70)

    store = create_store()

    # Same data, different encodings
    digits = {1, 3, 5, 7, 9}

    # Approach 1: Simple bundle (what template matching uses)
    digit_vecs = {d: store.vector_manager.get_vector(str(d)) for d in range(1, 10)}
    bundle_vec = store.bundle([digit_vecs[d] for d in digits])
    complete_bundle = store.bundle([digit_vecs[d] for d in range(1, 10)])

    # Approach 2: Position-value dict
    dict_vec = store.encoder.encode_data({str(i): d for i, d in enumerate(sorted(digits))})
    complete_dict = store.encoder.encode_data({str(i): i+1 for i in range(9)})

    # Approach 3: Orientation-tagged
    tagged_vec = store.encoder.encode_data({
        "orientation": "row",
        "values": {str(i): d for i, d in enumerate(sorted(digits))}
    })
    complete_tagged = store.encoder.encode_data({
        "orientation": "row",
        "values": {str(i): i+1 for i in range(9)}
    })

    print(f"\nDigits: {digits}")
    print(f"\nSimilarity to 'complete' template:")
    print(f"  Bundle approach:      {similarity(bundle_vec, complete_bundle):.4f}")
    print(f"  Position-dict:        {similarity(dict_vec, complete_dict):.4f}")
    print(f"  Orientation-tagged:   {similarity(tagged_vec, complete_tagged):.4f}")

    # Now test adding a digit
    digits_plus_2 = digits | {2}

    bundle_new = store.bundle([digit_vecs[d] for d in digits_plus_2])
    dict_new = store.encoder.encode_data({str(i): d for i, d in enumerate(sorted(digits_plus_2))})
    tagged_new = store.encoder.encode_data({
        "orientation": "row",
        "values": {str(i): d for i, d in enumerate(sorted(digits_plus_2))}
    })

    print(f"\nAfter adding digit 2:")
    print(f"  Bundle:      {similarity(bundle_vec, complete_bundle):.4f} -> {similarity(bundle_new, complete_bundle):.4f}")
    print(f"  Dict:        {similarity(dict_vec, complete_dict):.4f} -> {similarity(dict_new, complete_dict):.4f}")
    print(f"  Tagged:      {similarity(tagged_vec, complete_tagged):.4f} -> {similarity(tagged_new, complete_tagged):.4f}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    test_position_value_encoding()
    test_orientation_encoding()
    test_precompute_and_query()
    test_composite_puzzle()
    bt_structured = test_structured_solver()
    test_compare_encodings()

    print("\n" + "=" * 70)
    print("STRUCTURED COMPOSITION SUMMARY")
    print("=" * 70)
    print(f"""
THE USER'S IDEA:
- Encode as {{position: value}} dicts
- Tag with orientation: {{orientation: "row", values: {{...}}}}
- Pre-compute valid states
- Query against them

RESULTS:
| Approach | Backtracks |
|----------|-----------|
| Simple bundle | 52 |
| Structured solver | {bt_structured} |

KEY FINDINGS FROM EXPERIMENT 6 (Encoding Comparison):
- Bundle approach gives HIGHER similarity gradient
- Position-dict adds noise from position keys
- Orientation tag adds more noise

WHY BUNDLE WINS:
The {'{position: value}'} encoding mixes TWO signals:
1. Which digits are present (good)
2. Which positions they're in (irrelevant for constraint check!)

For Sudoku constraints, we only care about WHICH digits,
not WHERE they are within the unit. Positions are noise.

WHEN STRUCTURED ENCODING WOULD HELP:
- If position within unit mattered
- If we needed to query "what's at position 3?"
- If we were doing retrieval, not real-time scoring
""")


if __name__ == "__main__":
    main()
