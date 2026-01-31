#!/usr/bin/env python3
"""
Approach 36: Cell-Level Query with $any and Negations

Instead of querying for full row completions,
query at CELL level with negations for constraints.

For each empty cell:
1. Store all 9 possible values
2. Query with $any
3. Use negations to exclude row/col/block values
4. Get valid candidates via query
"""

import sys
sys.path.insert(0, "/home/watmin/work/holon")

from typing import List, Optional, Set, Dict
import numpy as np
import time

from holon import CPUStore, HolonClient

from common import (
    similarity,
    count_empty,
    get_available_digits_9x9,
    validate_9x9,
    PUZZLE_9x9_HARD,
)


def create_client(dimensions: int = 8192):
    store = CPUStore(dimensions=dimensions)
    return HolonClient(local_store=store), store


# =============================================================================
# EXPERIMENT 1: Cell-Level Digit Query
# =============================================================================

def test_cell_digit_query():
    """
    Store digits and query with negations for cell constraints.
    """
    print("=" * 70)
    print("EXPERIMENT 1: Cell-Level Digit Query")
    print("=" * 70)

    client, store = create_client()

    # Store all 9 digits
    for d in range(1, 10):
        client.insert_json({"type": "digit", "value": d})

    print("Stored 9 digit entries")

    # For a specific cell, get constraints
    grid = [[cell for cell in row] for row in PUZZLE_9x9_HARD]
    r, c = 0, 0

    # What's NOT allowed at this cell?
    row_used = {grid[r][cc] for cc in range(9) if grid[r][cc] is not None}
    col_used = {grid[rr][c] for rr in range(9) if grid[rr][c] is not None}
    br, bc = (r // 3) * 3, (c // 3) * 3
    block_used = {grid[rr][cc] for rr in range(br, br+3) for cc in range(bc, bc+3)
                  if grid[rr][cc] is not None}

    all_forbidden = row_used | col_used | block_used
    expected = set(range(1, 10)) - all_forbidden

    print(f"\nCell ({r},{c}):")
    print(f"  Row forbids: {row_used}")
    print(f"  Col forbids: {col_used}")
    print(f"  Block forbids: {block_used}")
    print(f"  All forbidden: {all_forbidden}")
    print(f"  Expected valid: {expected}")

    # Query all digits, filter out forbidden
    # Note: Holon's search doesn't support multiple OR negations directly
    # We'll query all and filter
    all_results = client.search_json(probe={"type": "digit"}, limit=10, threshold=0.0)
    valid_results = [r for r in all_results if r['data']['value'] not in all_forbidden]

    print(f"\n  Query results (after filtering): {[r['data']['value'] for r in valid_results]}")


# =============================================================================
# EXPERIMENT 2: Constraint Units as Query Targets
# =============================================================================

def test_constraint_unit_query():
    """
    Store constraint units (row/col/block states) and query for matches.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Constraint Unit Query")
    print("=" * 70)

    client, store = create_client()

    # For each constraint, store what values are already present
    # Then query to find "gaps"

    grid = [[cell for cell in row] for row in PUZZLE_9x9_HARD]

    # Store constraint states
    for r in range(9):
        row_vals = [grid[r][c] for c in range(9) if grid[r][c] is not None]
        client.insert_json({
            "type": "constraint",
            "unit": "row",
            "index": r,
            "values": sorted(row_vals),
            "count": len(row_vals),
        })

    for c in range(9):
        col_vals = [grid[r][c] for r in range(9) if grid[r][c] is not None]
        client.insert_json({
            "type": "constraint",
            "unit": "col",
            "index": c,
            "values": sorted(col_vals),
            "count": len(col_vals),
        })

    for b in range(9):
        br, bc = (b // 3) * 3, (b % 3) * 3
        block_vals = [grid[rr][cc] for rr in range(br, br+3) for cc in range(bc, bc+3)
                      if grid[rr][cc] is not None]
        client.insert_json({
            "type": "constraint",
            "unit": "block",
            "index": b,
            "values": sorted(block_vals),
            "count": len(block_vals),
        })

    print("Stored 27 constraint unit states")

    # Query for constraints affecting cell (0,0)
    r, c = 0, 0
    block_idx = (r // 3) * 3 + (c // 3)

    print(f"\nConstraints for cell ({r},{c}):")

    row_result = client.search_json(
        probe={"type": "constraint", "unit": "row", "index": r},
        limit=1, threshold=0.0
    )
    if row_result:
        print(f"  Row {r}: values = {row_result[0]['data']['values']}")

    col_result = client.search_json(
        probe={"type": "constraint", "unit": "col", "index": c},
        limit=1, threshold=0.0
    )
    if col_result:
        print(f"  Col {c}: values = {col_result[0]['data']['values']}")

    block_result = client.search_json(
        probe={"type": "constraint", "unit": "block", "index": block_idx},
        limit=1, threshold=0.0
    )
    if block_result:
        print(f"  Block {block_idx}: values = {block_result[0]['data']['values']}")


# =============================================================================
# EXPERIMENT 3: Cell-Context Encoding with $any
# =============================================================================

def test_cell_context_any():
    """
    Encode cell with its context, use $any for unknowns.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Cell Context with $any")
    print("=" * 70)

    client, store = create_client()

    # Store "valid cell configurations"
    # {cell: (r,c), value: d, row_has: [...], col_has: [...], block_has: [...]}

    # For now, just store some examples
    configs = [
        {"cell_r": 0, "cell_c": 0, "value": 5, "row_has": [4, 6], "col_has": [7, 9]},
        {"cell_r": 0, "cell_c": 0, "value": 1, "row_has": [4, 6], "col_has": [7, 9]},
        {"cell_r": 0, "cell_c": 0, "value": 2, "row_has": [4, 6], "col_has": [7, 9]},
    ]

    for cfg in configs:
        client.insert_json(cfg)

    # Query with $any for unknown parts
    query = {
        "cell_r": 0,
        "cell_c": 0,
        "value": {"$any": True},  # We want ANY valid value
        "row_has": {"$any": True},  # Match any row state
    }

    results = client.search_json(probe=query, limit=10, threshold=0.0)

    print(f"Query for cell (0,0) with value=$any:")
    for r in results:
        print(f"  Value {r['data']['value']}: score {r['score']:.3f}")


# =============================================================================
# EXPERIMENT 4: Hybrid Query + Vector Scoring
# =============================================================================

def test_hybrid_solver():
    """
    Combine query for candidates with vector scoring for ordering.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Hybrid Query + Vector Scoring")
    print("=" * 70)

    client, store = create_client()

    # Pre-compute digit vectors
    digit_vecs = {d: store.vector_manager.get_vector(str(d)) for d in range(1, 10)}
    complete_template = store.bundle([digit_vecs[d] for d in range(1, 10)])

    class HybridSolver:
        def __init__(self):
            self.backtracks = 0

        def get_candidates(self, grid, r, c):
            """Get valid candidates for a cell using constraint logic."""
            return get_available_digits_9x9(grid, r, c)

        def score_choice(self, grid, r, c, digit):
            """Score a choice using template matching."""
            # Row score
            row_digits = {grid[r][cc] for cc in range(9) if grid[r][cc] is not None}
            row_digits.add(digit)
            row_vec = store.bundle([digit_vecs[d] for d in row_digits])
            row_score = similarity(row_vec, complete_template)

            # Col score
            col_digits = {grid[rr][c] for rr in range(9) if grid[rr][c] is not None}
            col_digits.add(digit)
            col_vec = store.bundle([digit_vecs[d] for d in col_digits])
            col_score = similarity(col_vec, complete_template)

            # Block score
            br, bc = (r // 3) * 3, (c // 3) * 3
            block_digits = {grid[rr][cc] for rr in range(br, br+3) for cc in range(bc, bc+3)
                            if grid[rr][cc] is not None}
            block_digits.add(digit)
            block_vec = store.bundle([digit_vecs[d] for d in block_digits])
            block_score = similarity(block_vec, complete_template)

            return row_score + col_score + block_score

        def solve(self, puzzle):
            grid = [[cell for cell in row] for row in puzzle]
            return self._solve_rec(grid)

        def _solve_rec(self, grid):
            # Propagate
            changed = True
            while changed:
                changed = False
                for r in range(9):
                    for c in range(9):
                        if grid[r][c] is None:
                            opts = list(self.get_candidates(grid, r, c))
                            if not opts:
                                return None
                            if len(opts) == 1:
                                grid[r][c] = opts[0]
                                changed = True

            if count_empty(grid) == 0:
                return grid

            # Find best cell
            best = None
            best_count = 10
            for r in range(9):
                for c in range(9):
                    if grid[r][c] is None:
                        opts = list(self.get_candidates(grid, r, c))
                        if len(opts) < best_count:
                            best_count = len(opts)
                            best = (r, c, opts)

            if best is None:
                return None

            r, c, options = best

            # Score and order choices
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

    solver = HybridSolver()
    start = time.time()
    result = solver.solve(PUZZLE_9x9_HARD)
    elapsed = time.time() - start

    print(f"\nHybrid solver:")
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
    test_cell_digit_query()
    test_constraint_unit_query()
    test_cell_context_any()
    bt = test_hybrid_solver()

    print("\n" + "=" * 70)
    print("CELL-LEVEL QUERY SUMMARY")
    print("=" * 70)
    print(f"""
APPROACH:
Query at CELL level instead of ROW level.
Use constraint logic to get candidates, vector scoring to order them.

RESULT:
| Approach | Backtracks |
|----------|-----------|
| Template matching (baseline) | 52 |
| Hybrid query+vector | {bt} |

KEY INSIGHT:
The $any and negation queries are most useful for:
1. DISCOVERING what's stored
2. RETRIEVING matching patterns
3. FILTERING by constraints

But for SOLVING, the constraint logic (row/col/block intersection)
is simple arithmetic that's faster computed directly than queried.

WHAT HOLON ADDS:
The SCORING via template matching - geometric signal for ordering choices.
This is where the VSA/HDC magic happens.

RECOMMENDATION:
Use queries for retrieval/discovery, direct vector ops for solving.
""")


if __name__ == "__main__":
    main()
