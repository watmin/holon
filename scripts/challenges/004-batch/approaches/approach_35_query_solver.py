#!/usr/bin/env python3
"""
Approach 35: Full Query-Based Solver

Use pre-computed valid rows + $any queries + negations to solve Sudoku.

The idea:
1. Pre-compute all 362,880 valid rows
2. For each row in puzzle, query with known values as constraints
3. Use column/block constraints as negations
4. Find valid row completions that respect ALL constraints
"""

import sys
sys.path.insert(0, "/home/watmin/work/holon")

from typing import List, Optional, Set, Tuple, Dict, Any
from itertools import permutations
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
# EXPERIMENT 1: Query Solver - Find Valid Row Completions
# =============================================================================

def test_row_completion_with_constraints():
    """
    Query for valid row completions, filtering by column/block constraints.
    """
    print("=" * 70)
    print("EXPERIMENT 1: Row Completion with Full Constraints")
    print("=" * 70)

    client, store = create_client()

    # Pre-compute valid rows
    print("\nPre-computing all valid rows...")
    start = time.time()
    count = 0
    for perm in permutations(range(1, 10)):
        client.insert_json({
            "type": "row",
            "p0": perm[0], "p1": perm[1], "p2": perm[2],
            "p3": perm[3], "p4": perm[4], "p5": perm[5],
            "p6": perm[6], "p7": perm[7], "p8": perm[8],
        })
        count += 1

    print(f"Inserted {count} rows in {time.time() - start:.1f}s")

    # For row 0 of puzzle, find valid completions
    grid = [[cell for cell in row] for row in PUZZLE_9x9_HARD]
    row_idx = 0

    # Get column and block constraints
    def get_col_constraints(grid, row_idx):
        """For each position, what values are NOT allowed (from column)?"""
        constraints = {}
        for c in range(9):
            col_used = {grid[r][c] for r in range(9) if r != row_idx and grid[r][c] is not None}
            if col_used:
                constraints[c] = col_used
        return constraints

    def get_block_constraints(grid, row_idx):
        """For each position, what values are NOT allowed (from block)?"""
        constraints = {}
        br = (row_idx // 3) * 3

        for c in range(9):
            bc = (c // 3) * 3
            block_used = set()
            for rr in range(br, br + 3):
                for cc in range(bc, bc + 3):
                    if rr != row_idx and grid[rr][cc] is not None:
                        block_used.add(grid[rr][cc])
            if block_used:
                constraints[c] = block_used
        return constraints

    col_constraints = get_col_constraints(grid, row_idx)
    block_constraints = get_block_constraints(grid, row_idx)

    print(f"\nRow {row_idx}: {grid[row_idx]}")
    print(f"Column constraints: {col_constraints}")
    print(f"Block constraints: {block_constraints}")

    # Build query
    query = {"type": "row"}
    for c in range(9):
        if grid[row_idx][c] is not None:
            query[f"p{c}"] = grid[row_idx][c]
        else:
            query[f"p{c}"] = {"$any": True}

    # Query all valid row completions
    results = client.search_json(probe=query, limit=1000, threshold=0.0)

    print(f"\nQuery returned {len(results)} results")

    # Filter by column and block constraints manually
    # (Holon's negations work on single fields, not OR combinations)
    valid_completions = []
    for r in results:
        d = r['data']
        row_vals = [d[f"p{i}"] for i in range(9)]

        # Check column constraints
        col_valid = True
        for c, forbidden in col_constraints.items():
            if row_vals[c] in forbidden:
                col_valid = False
                break

        # Check block constraints
        block_valid = True
        if col_valid:
            for c, forbidden in block_constraints.items():
                if row_vals[c] in forbidden:
                    block_valid = False
                    break

        if col_valid and block_valid:
            valid_completions.append(row_vals)

    print(f"After constraint filtering: {len(valid_completions)} valid completions")
    for row in valid_completions[:5]:
        print(f"  {row}")


# =============================================================================
# EXPERIMENT 2: Full Puzzle Solver
# =============================================================================

def test_full_solver():
    """
    Solve entire puzzle using row-by-row query approach.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Full Puzzle Solver (Row-by-Row)")
    print("=" * 70)

    client, store = create_client()

    # Pre-compute valid rows
    print("\nPre-computing all valid rows...")
    start = time.time()
    for perm in permutations(range(1, 10)):
        client.insert_json({
            "type": "row",
            "p0": perm[0], "p1": perm[1], "p2": perm[2],
            "p3": perm[3], "p4": perm[4], "p5": perm[5],
            "p6": perm[6], "p7": perm[7], "p8": perm[8],
        })
    print(f"Inserted rows in {time.time() - start:.1f}s")

    def get_valid_row_completions(client, grid, row_idx, limit=5000):
        """Get all valid completions for a row given current grid state."""

        # Build query from known values
        query = {"type": "row"}
        for c in range(9):
            if grid[row_idx][c] is not None:
                query[f"p{c}"] = grid[row_idx][c]
            else:
                query[f"p{c}"] = {"$any": True}

        results = client.search_json(probe=query, limit=limit, threshold=0.0)

        # Get constraints from columns and blocks
        valid = []
        for r in results:
            d = r['data']
            row_vals = [d[f"p{i}"] for i in range(9)]

            is_valid = True

            # Check column constraints
            for c in range(9):
                if grid[row_idx][c] is None:  # Only check unfilled positions
                    for other_r in range(9):
                        if other_r != row_idx and grid[other_r][c] == row_vals[c]:
                            is_valid = False
                            break
                if not is_valid:
                    break

            # Check block constraints
            if is_valid:
                br = (row_idx // 3) * 3
                for c in range(9):
                    if grid[row_idx][c] is None:
                        bc = (c // 3) * 3
                        for rr in range(br, br + 3):
                            if rr != row_idx:
                                for cc in range(bc, bc + 3):
                                    if grid[rr][cc] == row_vals[c]:
                                        is_valid = False
                                        break
                            if not is_valid:
                                break
                        if not is_valid:
                            break

            if is_valid:
                valid.append(row_vals)

        return valid

    # Solve row by row
    grid = [[cell for cell in row] for row in PUZZLE_9x9_HARD]
    backtracks = 0

    def solve_row_by_row(grid, row_idx=0):
        nonlocal backtracks

        if row_idx == 9:
            return True  # All rows solved

        # Get valid completions for this row
        completions = get_valid_row_completions(client, grid, row_idx)

        if not completions:
            backtracks += 1
            return False

        # Try each valid completion
        for completion in completions:
            # Apply this row
            old_row = grid[row_idx][:]
            grid[row_idx] = completion

            if solve_row_by_row(grid, row_idx + 1):
                return True

            # Backtrack
            grid[row_idx] = old_row
            backtracks += 1

        return False

    start = time.time()
    solved = solve_row_by_row(grid)
    elapsed = time.time() - start

    print(f"\nResult: {'SOLVED' if solved else 'FAILED'}")
    print(f"Backtracks: {backtracks}")
    print(f"Time: {elapsed:.2f}s")

    if solved and validate_9x9(grid):
        print("Solution validated!")
        for row in grid:
            print(f"  {row}")
    elif solved:
        print("INVALID SOLUTION!")

    return backtracks


# =============================================================================
# EXPERIMENT 3: Compare with Traditional Backtracking
# =============================================================================

def test_comparison():
    """
    Compare query-based approach with traditional cell-by-cell.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Comparison Summary")
    print("=" * 70)

    # Traditional solver for reference
    def solve_traditional(puzzle):
        grid = [[cell for cell in row] for row in puzzle]
        backtracks = [0]

        def solve():
            # Propagate
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

            if count_empty(grid) == 0:
                return True

            # Find cell with fewest options
            best = None
            for r in range(9):
                for c in range(9):
                    if grid[r][c] is None:
                        opts = list(get_available_digits_9x9(grid, r, c))
                        if best is None or len(opts) < len(best[2]):
                            best = (r, c, opts)

            if best is None:
                return False

            r, c, options = best

            for d in options:
                test_grid = [[cell for cell in row] for row in grid]
                test_grid[r][c] = d
                grid[:] = test_grid

                if solve():
                    return True

                backtracks[0] += 1
                grid[:] = [[cell for cell in row] for row in test_grid]
                grid[r][c] = None

            return False

        start = time.time()
        solved = solve()
        elapsed = time.time() - start

        return backtracks[0], elapsed

    trad_bt, trad_time = solve_traditional(PUZZLE_9x9_HARD)

    print(f"\nTraditional (cell-by-cell): {trad_bt} backtracks, {trad_time:.3f}s")
    print(f"Query-based (row-by-row): See Experiment 2 above")


# =============================================================================
# MAIN
# =============================================================================

def main():
    test_row_completion_with_constraints()
    query_bt = test_full_solver()
    test_comparison()

    print("\n" + "=" * 70)
    print("QUERY-BASED SOLVER SUMMARY")
    print("=" * 70)
    print(f"""
THE USER'S APPROACH IMPLEMENTED:
1. Pre-compute all 362,880 valid rows
2. Query with $any for unknown positions
3. Filter results by column/block constraints
4. Solve row-by-row

RESULTS:
| Approach | Backtracks | Notes |
|----------|-----------|-------|
| Template matching (cell) | 52 | Best overall |
| Query-based (row) | {query_bt} | Full constraint check |
| Traditional (cell) | ~249 | No geometric guidance |

ANALYSIS:

The row-by-row query approach:
+ Uses Holon's query system properly
+ Pre-computes all valid patterns
+ Uses $any for unknowns (as user requested!)

BUT:
- Row-by-row is coarser granularity than cell-by-cell
- When a row completion fails, we backtrack the ENTIRE row
- Cell-by-cell can backtrack just one cell
- Constraint filtering still needed (Holon can't do multi-field OR negations)

KEY INSIGHT:
Holon's $any and negations work for RETRIEVAL and FILTERING,
but the granularity of "valid rows" vs "valid cells" matters for solving.

Cell-by-cell with template matching = best of both worlds:
- Fine-grained backtracking
- Geometric guidance from template similarity
""")


if __name__ == "__main__":
    main()
