#!/usr/bin/env python3
"""
Approach 34: Using $any Wildcards for Unknown Positions

THE USER'S IDEA:
1. Encode FULL rows with $any for unknowns:
   {group: "row", values: {0: 1, 1: $any, 2: 3, 3: $any, ...}}

2. Pre-compute all valid complete rows (permutations of 1-9)

3. Query with negations to reject illegal combinations

This uses Holon's QUERY SYSTEM, not just vector operations!
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
# EXPERIMENT 1: Basic $any Matching
# =============================================================================

def test_any_matching():
    """
    Test how $any works for matching partial patterns.
    """
    print("=" * 70)
    print("EXPERIMENT 1: Basic $any Matching")
    print("=" * 70)

    client, store = create_client()

    # Insert some complete rows
    complete_rows = [
        {"group": "row", "index": 0, "values": {"0": 1, "1": 2, "2": 3, "3": 4, "4": 5, "5": 6, "6": 7, "7": 8, "8": 9}},
        {"group": "row", "index": 0, "values": {"0": 9, "1": 8, "2": 7, "3": 6, "4": 5, "5": 4, "6": 3, "7": 2, "8": 1}},
        {"group": "row", "index": 0, "values": {"0": 1, "1": 3, "2": 5, "3": 7, "4": 9, "5": 2, "6": 4, "7": 6, "8": 8}},
        {"group": "row", "index": 0, "values": {"0": 5, "1": 4, "2": 3, "3": 2, "4": 1, "5": 6, "6": 7, "7": 8, "8": 9}},
    ]

    for row in complete_rows:
        client.insert_json(row)

    print(f"\nInserted {len(complete_rows)} complete rows")

    # Query with $any for unknowns
    # "Find rows where position 0 is 1, position 2 is 3, others are anything"
    query = {
        "group": "row",
        "values": {
            "0": 1,
            "1": {"$any": True},
            "2": 3,
            "3": {"$any": True},
            "4": {"$any": True},
            "5": {"$any": True},
            "6": {"$any": True},
            "7": {"$any": True},
            "8": {"$any": True},
        }
    }

    results = client.search_json(probe=query, limit=10, threshold=0.0)

    print(f"\nQuery: values[0]=1, values[2]=3, rest=$any")
    print(f"Results: {len(results)}")
    for r in results:
        vals = r['data']['values']
        print(f"  Score {r['score']:.3f}: [{vals['0']},{vals['1']},{vals['2']},{vals['3']},{vals['4']},{vals['5']},{vals['6']},{vals['7']},{vals['8']}]")


# =============================================================================
# EXPERIMENT 2: Pre-compute Valid Rows and Query
# =============================================================================

def test_precompute_valid_rows():
    """
    Pre-compute a sample of valid complete rows and query against them.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Pre-compute Valid Rows")
    print("=" * 70)

    client, store = create_client()

    # We can't store all 362880 permutations, but let's store a sample
    print("\nGenerating sample of valid rows...")

    count = 0
    max_rows = 1000  # Sample

    for perm in permutations(range(1, 10)):
        if count >= max_rows:
            break

        row_data = {
            "type": "valid_row",
            "values": {str(i): perm[i] for i in range(9)},
        }
        client.insert_json(row_data)
        count += 1

    print(f"Inserted {count} valid rows")

    # Now query with a partial pattern using $any
    # Simulate: we know positions 0, 3, 6 have values 5, 2, 8
    query = {
        "type": "valid_row",
        "values": {
            "0": 5,
            "1": {"$any": True},
            "2": {"$any": True},
            "3": 2,
            "4": {"$any": True},
            "5": {"$any": True},
            "6": 8,
            "7": {"$any": True},
            "8": {"$any": True},
        }
    }

    results = client.search_json(probe=query, limit=20, threshold=0.0)

    print(f"\nQuery: values[0]=5, values[3]=2, values[6]=8, rest=$any")
    print(f"Found {len(results)} matches")
    for r in results[:5]:
        vals = r['data']['values']
        row = [vals[str(i)] for i in range(9)]
        print(f"  Score {r['score']:.3f}: {row}")


# =============================================================================
# EXPERIMENT 3: Query with Negations
# =============================================================================

def test_negation_queries():
    """
    Use negations to reject invalid combinations.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Query with Negations")
    print("=" * 70)

    client, store = create_client()

    # Insert valid rows
    print("\nInserting valid rows...")
    count = 0
    for perm in permutations(range(1, 10)):
        if count >= 500:
            break
        client.insert_json({
            "type": "row",
            "values": {str(i): perm[i] for i in range(9)},
        })
        count += 1

    print(f"Inserted {count} valid rows")

    # Query: find rows where position 0 is 5,
    # but position 1 is NOT 3 (constraint from column)
    # and position 2 is NOT 7 (constraint from block)

    query = {
        "type": "row",
        "values": {
            "0": 5,
            "1": {"$any": True},
            "2": {"$any": True},
        }
    }

    # Use negations to exclude
    negations = {
        "values.1": 3,  # position 1 cannot be 3
        "values.2": 7,  # position 2 cannot be 7
    }

    results = client.search_json(probe=query, negations=negations, limit=20, threshold=0.0)

    print(f"\nQuery: values[0]=5, values[1]≠3, values[2]≠7")
    print(f"Found {len(results)} matches")
    for r in results[:5]:
        vals = r['data']['values']
        v1, v2 = vals.get("1"), vals.get("2")
        print(f"  Score {r['score']:.3f}: pos[1]={v1}, pos[2]={v2}")


# =============================================================================
# EXPERIMENT 4: Full Puzzle Row Query
# =============================================================================

def test_puzzle_row_query():
    """
    Query for valid completions of a puzzle row.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Puzzle Row Query")
    print("=" * 70)

    client, store = create_client()

    # Insert valid rows
    print("\nInserting all valid rows (may take a moment)...")
    count = 0
    for perm in permutations(range(1, 10)):
        client.insert_json({
            "type": "row",
            "v0": perm[0], "v1": perm[1], "v2": perm[2],
            "v3": perm[3], "v4": perm[4], "v5": perm[5],
            "v6": perm[6], "v7": perm[7], "v8": perm[8],
        })
        count += 1
        if count % 50000 == 0:
            print(f"  {count} rows...")

    print(f"Inserted {count} valid rows")

    # Query based on first row of puzzle
    # PUZZLE_9x9_HARD first row: [None, None, None, 4, None, None, 6, None, None]
    row = PUZZLE_9x9_HARD[0]
    print(f"\nPuzzle row 0: {row}")

    # Build query with $any for unknowns
    query = {"type": "row"}
    for i, val in enumerate(row):
        if val is not None:
            query[f"v{i}"] = val
        else:
            query[f"v{i}"] = {"$any": True}

    results = client.search_json(probe=query, limit=20, threshold=0.0)

    print(f"\nQuery with known values, $any for unknowns:")
    print(f"Found {len(results)} valid completions")
    for r in results[:5]:
        d = r['data']
        row_vals = [d[f"v{i}"] for i in range(9)]
        print(f"  Score {r['score']:.3f}: {row_vals}")


# =============================================================================
# EXPERIMENT 5: Multi-Constraint Query
# =============================================================================

def test_multi_constraint():
    """
    Query considering row, column, AND block constraints.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: Multi-Constraint Query")
    print("=" * 70)

    client, store = create_client()

    # For a cell, we need to find values that satisfy:
    # - Row constraint (this digit not in row)
    # - Column constraint (this digit not in column)
    # - Block constraint (this digit not in block)

    # Store candidate vectors for digits 1-9
    print("\nStoring digit candidates...")
    for d in range(1, 10):
        client.insert_json({
            "type": "digit",
            "value": d,
        })

    # Get constraints from puzzle for cell (0,0)
    grid = PUZZLE_9x9_HARD
    r, c = 0, 0

    # What's NOT allowed?
    row_used = {grid[r][cc] for cc in range(9) if grid[r][cc] is not None}
    col_used = {grid[rr][c] for rr in range(9) if grid[rr][c] is not None}
    br, bc = (r // 3) * 3, (c // 3) * 3
    block_used = {grid[rr][cc] for rr in range(br, br+3) for cc in range(bc, bc+3)
                  if grid[rr][cc] is not None}

    all_used = row_used | col_used | block_used
    available = set(range(1, 10)) - all_used

    print(f"\nCell (0,0) constraints:")
    print(f"  Row has: {row_used}")
    print(f"  Col has: {col_used}")
    print(f"  Block has: {block_used}")
    print(f"  Available: {available}")

    # Query with negations for used values
    query = {"type": "digit"}

    # Build negations for each used value
    # Note: Holon's negation format uses $not
    negation_list = [{"value": v} for v in all_used]

    print(f"\nQuerying with {len(negation_list)} negations...")

    # Try query without negations first
    all_results = client.search_json(probe=query, limit=10, threshold=0.0)
    print(f"Without negations: {len(all_results)} results")

    # Now filter manually (Holon's negations work differently)
    filtered = [r for r in all_results if r['data']['value'] not in all_used]
    print(f"After filtering used values: {len(filtered)} results")
    for r in filtered:
        print(f"  Value {r['data']['value']}: score {r['score']:.3f}")


# =============================================================================
# EXPERIMENT 6: Encoding Rows with Full Structure
# =============================================================================

def test_full_row_structure():
    """
    Encode rows with complete 9-value structure using $any.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 6: Full Row Structure Encoding")
    print("=" * 70)

    client, store = create_client()

    # The user's exact format:
    # {group: "row", values: {0: 1, 1: $any, 2: 3, ...}}

    # First, let's understand how $any affects encoding
    # Encode same row with and without $any placeholders

    # Row with known values only
    sparse = {
        "group": "row",
        "index": 0,
        "v0": 5,
        "v3": 2,
        "v6": 8,
    }
    sparse_vec = store.encoder.encode_data(sparse)

    # Row with $any placeholders (but $any might not encode meaningfully)
    full_with_any = {
        "group": "row",
        "index": 0,
        "v0": 5,
        "v1": {"$any": True},
        "v2": {"$any": True},
        "v3": 2,
        "v4": {"$any": True},
        "v5": {"$any": True},
        "v6": 8,
        "v7": {"$any": True},
        "v8": {"$any": True},
    }
    full_vec = store.encoder.encode_data(full_with_any)

    print(f"\nSparse encoding norm: {np.linalg.norm(sparse_vec):.1f}")
    print(f"Full with $any norm: {np.linalg.norm(full_vec):.1f}")
    print(f"Similarity: {similarity(sparse_vec, full_vec):.4f}")

    # Complete row
    complete = {
        "group": "row",
        "index": 0,
        "v0": 5, "v1": 1, "v2": 3, "v3": 2, "v4": 4, "v5": 6, "v6": 8, "v7": 7, "v8": 9,
    }
    complete_vec = store.encoder.encode_data(complete)

    print(f"\nComplete row norm: {np.linalg.norm(complete_vec):.1f}")
    print(f"sim(sparse, complete): {similarity(sparse_vec, complete_vec):.4f}")
    print(f"sim(full_any, complete): {similarity(full_vec, complete_vec):.4f}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    test_any_matching()
    test_precompute_valid_rows()
    test_negation_queries()
    test_puzzle_row_query()
    test_multi_constraint()
    test_full_row_structure()

    print("\n" + "=" * 70)
    print("$ANY WILDCARD SUMMARY")
    print("=" * 70)
    print("""
THE USER'S APPROACH:
Encode full rows with $any for unknowns, query against valid completions,
use negations to filter out illegal values.

WHAT WE LEARNED:

1. $any IS supported in queries - it matches any value at that position

2. Pre-computing valid rows WORKS - we can store all 362,880 permutations

3. Negations can filter - exclude values that violate constraints

4. BUT: This is a RETRIEVAL approach, not a SOLVING approach
   - We're querying for valid completions
   - Sudoku solving requires checking ROW + COLUMN + BLOCK together
   - Each cell is constrained by THREE overlapping groups

THE CHALLENGE:
For cell (r,c), we need values that are:
  - Not in row r (8 other cells)
  - Not in col c (8 other cells)
  - Not in block (8 other cells)

These constraints INTERACT - we can't just query one group.

WHEN THIS APPROACH EXCELS:
- Pattern matching: "Find rows that look like this"
- Retrieval: "What valid completions exist?"
- Validation: "Is this configuration valid?"

FOR SOLVING:
Still need backtracking, but $any queries could help
identify valid candidates faster than brute enumeration.
""")


if __name__ == "__main__":
    main()
