#!/usr/bin/env python3
"""
Approach 30: Using Holon's Query System

We've been doing everything with raw vector operations!
Holon supports rich querying:
- Similarity search with probes
- Guards for filtering
- $or for disjunctions
- $not for negations
- $any for wildcards
- Batch operations
- Threshold-based retrieval

Let's actually USE this for Sudoku!

Ideas:
1. Store valid digit patterns for each constraint type
2. Query to find which patterns match current state
3. Use guards to filter by constraint satisfaction
4. Use negations to exclude invalid combinations
"""

import sys
sys.path.insert(0, "/home/watmin/work/holon")

import json
from typing import List, Optional, Set, Tuple, Dict
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


# =============================================================================
# EXPERIMENT 1: Store Constraint Patterns and Query for Matches
# =============================================================================

def test_pattern_store():
    """
    Pre-compute valid constraint patterns and query them.
    """
    print("=" * 70)
    print("EXPERIMENT 1: Pattern Store and Query")
    print("=" * 70)

    store = CPUStore()
    client = HolonClient(local_store=store)

    # Insert valid constraint patterns
    # A "valid row pattern" is any permutation of {1-9}
    # But that's 9! = 362880, too many
    # Instead, let's store patterns for "partial validity"

    # Pattern: row with N unique digits
    print("\nInserting constraint patterns...")

    patterns_inserted = 0

    # Store patterns for "N digits present, all unique"
    for n in range(1, 10):
        for digits_combo in _combinations(range(1, 10), n):
            pattern = {
                "type": "valid_partial",
                "count": n,
                "digits": list(digits_combo),
                "complete": n == 9
            }
            client.insert_json(pattern)
            patterns_inserted += 1

    print(f"Inserted {patterns_inserted} patterns")

    # Now query: given current row digits, find similar valid patterns
    current_row = {1, 3, 5, 7}  # Current digits in a row
    probe = {
        "type": "valid_partial",
        "count": len(current_row),
        "digits": sorted(current_row)
    }

    results = client.search_json(
        probe=probe,
        limit=10,
        threshold=0.5
    )

    print(f"\nQuery: current row has {current_row}")
    print(f"Found {len(results)} similar patterns:")
    for r in results[:5]:
        print(f"  Score {r['score']:.3f}: {r['data']}")

    # Query with guard: find patterns that could extend current
    print("\n\nQuery with guard: patterns that are 'complete'")
    results = client.search_json(
        probe={"type": "valid_partial"},
        guard={"complete": True},
        limit=5
    )

    for r in results[:3]:
        print(f"  Complete pattern: {r['data']['digits']}")


def _combinations(iterable, r):
    """Generate combinations."""
    from itertools import combinations
    return combinations(iterable, r)


# =============================================================================
# EXPERIMENT 2: Store Sudoku States and Query for Similar
# =============================================================================

def test_state_store():
    """
    Store intermediate Sudoku states and query for guidance.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Sudoku State Store")
    print("=" * 70)

    store = CPUStore()
    client = HolonClient(local_store=store)

    # Represent a state as: filled cells count + constraint satisfaction
    def encode_state(grid):
        filled = sum(1 for r in range(9) for c in range(9) if grid[r][c] is not None)

        # Count unique digits per row/col/block
        row_counts = []
        for r in range(9):
            digits = {grid[r][c] for c in range(9) if grid[r][c] is not None}
            row_counts.append(len(digits))

        return {
            "type": "sudoku_state",
            "filled": filled,
            "progress": filled / 81.0,
            "row_avg": sum(row_counts) / 9,
            "min_row": min(row_counts),
            "max_row": max(row_counts),
        }

    # Store some states from solving
    print("\nStoring states from solving process...")

    grid = [[cell for cell in row] for row in PUZZLE_9x9_HARD]
    states_stored = 0

    # Propagate and store states
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

                        # Store this state
                        state = encode_state(grid)
                        state["outcome"] = "forced"  # This was a forced move
                        client.insert_json(state)
                        states_stored += 1

    print(f"Stored {states_stored} intermediate states")

    # Query: find states similar to current
    current = encode_state(grid)
    print(f"\nCurrent state: {current}")

    results = client.search_json(
        probe=current,
        limit=5,
        threshold=0.5
    )

    print(f"\nSimilar states found: {len(results)}")
    for r in results[:3]:
        print(f"  Score {r['score']:.3f}: progress={r['data']['progress']:.2f}")


# =============================================================================
# EXPERIMENT 3: Decision Database with Guards
# =============================================================================

def test_decision_database():
    """
    Store decisions and their outcomes, query to guide future decisions.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Decision Database")
    print("=" * 70)

    store = CPUStore()
    client = HolonClient(local_store=store)

    # Collect decisions during solving
    decisions = []

    def solve_and_record(puzzle):
        grid = [[cell for cell in row] for row in puzzle]

        # Propagate
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

        def solve_rec(g, depth):
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

            # Find MRV
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

            for digit in options:
                # Record decision context
                filled_before = sum(1 for rr in range(9) for cc in range(9) if g[rr][cc] is not None)

                test_g = [[cell for cell in row] for row in g]
                test_g[r][c] = digit

                result = solve_rec(test_g, depth + 1)

                decision = {
                    "type": "decision",
                    "depth": depth,
                    "options_count": len(options),
                    "digit_chosen": digit,
                    "filled_before": filled_before,
                    "outcome": "success" if result else "backtrack"
                }
                decisions.append(decision)

                if result:
                    return result

            return None

        return solve_rec(grid, 0)

    print("Solving and recording decisions...")
    result = solve_and_record(PUZZLE_9x9_HARD)
    print(f"Solved: {result is not None}")

    # Insert all decisions
    for d in decisions:
        client.insert_json(d)

    print(f"Recorded {len(decisions)} decisions")

    # Query: find successful decisions at similar depth
    print("\n\nQuerying for successful decisions:")

    results = client.search_json(
        probe={"type": "decision", "depth": 3},
        guard={"outcome": "success"},
        limit=10
    )

    success_count = len(results)
    print(f"  Successful decisions at depth ~3: {success_count}")

    # Query with negation: exclude backtracks
    results = client.search_json(
        probe={"type": "decision"},
        negations={"outcome": {"$not": "backtrack"}},
        limit=100
    )

    print(f"  Non-backtrack decisions: {len(results)}")

    # Query with $or: success OR depth < 2
    results = client.search_json(
        probe={"type": "decision"},
        guard={"$or": [{"outcome": "success"}, {"depth": 0}]},
        limit=100
    )

    print(f"  Success OR depth 0: {len(results)}")


# =============================================================================
# EXPERIMENT 4: Constraint Violation Detection via Query
# =============================================================================

def test_violation_detection():
    """
    Use queries to detect constraint violations.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Constraint Violation Detection")
    print("=" * 70)

    store = CPUStore()
    client = HolonClient(local_store=store)

    # Store "violation patterns" - what a bad state looks like
    print("\nStoring violation patterns...")

    # Pattern: duplicate digit in constraint unit
    for d in range(1, 10):
        violation = {
            "type": "violation",
            "violation_type": "duplicate",
            "digit": d,
            "count": 2  # Same digit appears twice
        }
        client.insert_json(violation)

    # Pattern: empty cell with no options
    violation = {
        "type": "violation",
        "violation_type": "empty_no_options",
        "digit": 0,
        "count": 0
    }
    client.insert_json(violation)

    print("Stored violation patterns")

    # Now check a state for violations
    def check_for_violations(grid):
        # Check each row for duplicates
        for r in range(9):
            digits = [grid[r][c] for c in range(9) if grid[r][c] is not None]
            for d in range(1, 10):
                if digits.count(d) > 1:
                    # Query: is this a known violation pattern?
                    results = client.search_json(
                        probe={
                            "type": "violation",
                            "violation_type": "duplicate",
                            "digit": d
                        },
                        threshold=0.8
                    )
                    if results:
                        return True, f"Row {r} has duplicate {d}"

        return False, "No violations"

    # Test on valid partial grid
    grid = [[cell for cell in row] for row in PUZZLE_9x9_HARD]
    has_violation, msg = check_for_violations(grid)
    print(f"\nCheck puzzle: {msg}")

    # Create an invalid grid
    bad_grid = [[cell for cell in row] for row in grid]
    # Force a duplicate
    bad_grid[0][0] = 6  # 6 is already in row 0
    has_violation, msg = check_for_violations(bad_grid)
    print(f"Check bad grid: {msg}")


# =============================================================================
# EXPERIMENT 5: Full Query-Based Solver
# =============================================================================

def test_query_solver():
    """
    Build a solver that uses queries for guidance.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: Query-Based Solver")
    print("=" * 70)

    store = CPUStore()
    client = HolonClient(local_store=store)

    # Pre-populate with "good choice" patterns
    print("\nPre-populating with choice patterns...")

    # Good: choice leads to many forced moves
    # Bad: choice leads to contradiction

    patterns_count = 0

    # Store patterns for each (options_count, digit_in_options_position)
    for opt_count in range(2, 6):
        for position in range(opt_count):
            pattern = {
                "type": "choice_context",
                "options_count": opt_count,
                "digit_position": position,  # Position in sorted options
                "expected_quality": 1.0 / opt_count  # Rough prior
            }
            client.insert_json(pattern)
            patterns_count += 1

    print(f"Stored {patterns_count} choice patterns")

    class QueryGuidedSolver:
        def __init__(self):
            self.backtracks = 0

        def query_choice_quality(self, options_count, digit, options):
            """Query for expected quality of this choice."""
            sorted_opts = sorted(options)
            position = sorted_opts.index(digit)

            results = client.search_json(
                probe={
                    "type": "choice_context",
                    "options_count": options_count,
                    "digit_position": position
                },
                limit=1,
                threshold=0.3
            )

            if results:
                return results[0]['data'].get('expected_quality', 0.5)
            return 0.5

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

            # Query for choice quality
            scores = []
            for d in options:
                quality = self.query_choice_quality(len(options), d, options)
                scores.append((quality, d))

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

    solver = QueryGuidedSolver()
    start = time.time()
    result = solver.solve(PUZZLE_9x9_HARD)
    elapsed = time.time() - start

    print(f"\nResult: {'SOLVED' if result else 'FAILED'}")
    print(f"Backtracks: {solver.backtracks}")
    print(f"Time: {elapsed:.3f}s")

    return solver.backtracks


# =============================================================================
# MAIN
# =============================================================================

def main():
    test_pattern_store()
    test_state_store()
    test_decision_database()
    test_violation_detection()
    backtracks = test_query_solver()

    print("\n" + "=" * 70)
    print("QUERY-BASED APPROACH SUMMARY")
    print("=" * 70)
    print(f"""
We've been ignoring Holon's query system!

HOLON QUERY FEATURES:
- Similarity search with probes
- Guards for exact filtering
- $or for disjunctions
- $not for negations
- $any for wildcards
- Batch insert/search
- Threshold-based retrieval

WHAT WE TESTED:
1. Pattern Store: Store valid constraint patterns, query for matches
2. State Store: Store intermediate states, find similar
3. Decision Database: Record outcomes, query for guidance
4. Violation Detection: Query for known bad patterns
5. Query-Based Solver: {backtracks} backtracks (vs 52 for template)

KEY INSIGHT:
The query system is great for retrieval-based reasoning, but
Sudoku solving needs real-time constraint checking, not retrieval.
The query overhead doesn't pay off for this domain.

BETTER USE CASES FOR QUERIES:
- Finding similar past problems
- Retrieving relevant examples
- Pattern matching in databases
- Semantic search applications
""")


if __name__ == "__main__":
    main()
