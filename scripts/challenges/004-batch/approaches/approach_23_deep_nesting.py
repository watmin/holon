#!/usr/bin/env python3
"""
Approach 23: Deep Nested Data Structures

Explore different ways to represent Sudoku as nested data that
Holon's recursive encoder can exploit.

REPRESENTATIONS TO TRY:

1. CONSTRAINT-CENTRIC: What each constraint has and needs
   {"row_0": {"has": [5,3,6], "needs": [1,2,4,7,8,9]}, ...}

2. DIGIT-CENTRIC: Where is each digit?
   {"digit_1": {"positions": [(r,c), ...], "missing_from": [...]}, ...}

3. CELL-WITH-CONTEXT: Each cell knows its constraints
   {"cell_0_0": {"value": 5, "row": [...], "col": [...], "block": [...]}, ...}

4. HIERARCHICAL GROUPS: 3x3 band/stack structure
   {"band_0": {"rows": [0,1,2], "blocks": [0,1,2], ...}, ...}

5. SATISFACTION STATE: What's complete, what's pending
   {"complete": ["row_3", "col_5"], "pending": [...], ...}
"""

import sys
sys.path.insert(0, "/home/watmin/work/holon")

from typing import List, Optional, Dict, Tuple, Any, Set
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
# REPRESENTATION 1: CONSTRAINT-CENTRIC
# =============================================================================

def to_constraint_centric(grid: List[List[Optional[int]]]) -> Dict:
    """
    Represent grid by what each constraint HAS and NEEDS.

    Structure:
    {
        "rows": {
            "r0": {"has": [5,3,6], "needs": [1,2,4,7,8,9]},
            ...
        },
        "cols": {...},
        "blocks": {...}
    }
    """
    result = {"rows": {}, "cols": {}, "blocks": {}}

    for r in range(9):
        has = [grid[r][c] for c in range(9) if grid[r][c] is not None]
        needs = [d for d in range(1, 10) if d not in has]
        result["rows"][f"r{r}"] = {"has": has, "needs": needs}

    for c in range(9):
        has = [grid[r][c] for r in range(9) if grid[r][c] is not None]
        needs = [d for d in range(1, 10) if d not in has]
        result["cols"][f"c{c}"] = {"has": has, "needs": needs}

    for b in range(9):
        br, bc = (b // 3) * 3, (b % 3) * 3
        has = []
        for r in range(br, br + 3):
            for c in range(bc, bc + 3):
                if grid[r][c] is not None:
                    has.append(grid[r][c])
        needs = [d for d in range(1, 10) if d not in has]
        result["blocks"][f"b{b}"] = {"has": has, "needs": needs}

    return result


# =============================================================================
# REPRESENTATION 2: DIGIT-CENTRIC
# =============================================================================

def to_digit_centric(grid: List[List[Optional[int]]]) -> Dict:
    """
    Represent grid by where each digit appears.

    Structure:
    {
        "d1": {"placed": [(r,c), ...], "count": 3},
        "d2": {...},
        ...
    }
    """
    result = {}

    for d in range(1, 10):
        positions = []
        for r in range(9):
            for c in range(9):
                if grid[r][c] == d:
                    positions.append({"row": r, "col": c})

        result[f"d{d}"] = {
            "placed": positions,
            "count": len(positions),
            "complete": len(positions) == 9
        }

    return result


# =============================================================================
# REPRESENTATION 3: CELL-WITH-CONTEXT
# =============================================================================

def to_cell_context(grid: List[List[Optional[int]]]) -> Dict:
    """
    Each cell knows its value and constraint context.

    Structure:
    {
        "c_0_0": {
            "value": 5,
            "row_peers": [3, 6, ...],  # other digits in same row
            "col_peers": [6, 8, ...],
            "block_peers": [3, 6, ...]
        },
        ...
    }
    """
    result = {}

    for r in range(9):
        for c in range(9):
            row_peers = [grid[r][cc] for cc in range(9) if cc != c and grid[r][cc] is not None]
            col_peers = [grid[rr][c] for rr in range(9) if rr != r and grid[rr][c] is not None]

            br, bc = (r // 3) * 3, (c // 3) * 3
            block_peers = []
            for rr in range(br, br + 3):
                for cc in range(bc, bc + 3):
                    if (rr, cc) != (r, c) and grid[rr][cc] is not None:
                        block_peers.append(grid[rr][cc])

            cell_data = {
                "row_peers": row_peers,
                "col_peers": col_peers,
                "block_peers": block_peers
            }

            if grid[r][c] is not None:
                cell_data["value"] = grid[r][c]

            result[f"c_{r}_{c}"] = cell_data

    return result


# =============================================================================
# REPRESENTATION 4: HIERARCHICAL BANDS/STACKS
# =============================================================================

def to_hierarchical_bands(grid: List[List[Optional[int]]]) -> Dict:
    """
    Sudoku has 3x3 band/stack structure.

    Structure:
    {
        "bands": {  # horizontal groups of 3 rows
            "band_0": {
                "rows": [row0_data, row1_data, row2_data],
                "blocks": [block0, block1, block2]
            },
            ...
        },
        "stacks": {  # vertical groups of 3 columns
            ...
        }
    }
    """
    result = {"bands": {}, "stacks": {}}

    # Bands (horizontal)
    for band in range(3):
        band_rows = []
        for r in range(band * 3, band * 3 + 3):
            row_data = [grid[r][c] for c in range(9)]
            band_rows.append(row_data)

        band_blocks = []
        for b in range(band * 3, band * 3 + 3):
            br, bc = (b // 3) * 3, (b % 3) * 3
            block_data = []
            for r in range(br, br + 3):
                for c in range(bc, bc + 3):
                    block_data.append(grid[r][c])
            band_blocks.append(block_data)

        result["bands"][f"band_{band}"] = {
            "rows": band_rows,
            "blocks": band_blocks
        }

    # Stacks (vertical)
    for stack in range(3):
        stack_cols = []
        for c in range(stack * 3, stack * 3 + 3):
            col_data = [grid[r][c] for r in range(9)]
            stack_cols.append(col_data)

        result["stacks"][f"stack_{stack}"] = {
            "cols": stack_cols
        }

    return result


# =============================================================================
# REPRESENTATION 5: SATISFACTION STATE
# =============================================================================

def to_satisfaction_state(grid: List[List[Optional[int]]]) -> Dict:
    """
    Track what's complete vs pending.

    Structure:
    {
        "complete_rows": [3, 5],
        "complete_cols": [1, 7],
        "complete_blocks": [4],
        "pending_cells": [(0,0), (0,2), ...],
        "progress": 0.28  # 23/81 cells filled
    }
    """
    complete_rows = []
    complete_cols = []
    complete_blocks = []
    pending_cells = []

    for r in range(9):
        row_digits = set(grid[r][c] for c in range(9) if grid[r][c] is not None)
        if len(row_digits) == 9:
            complete_rows.append(r)

    for c in range(9):
        col_digits = set(grid[r][c] for r in range(9) if grid[r][c] is not None)
        if len(col_digits) == 9:
            complete_cols.append(c)

    for b in range(9):
        br, bc = (b // 3) * 3, (b % 3) * 3
        block_digits = set()
        for r in range(br, br + 3):
            for c in range(bc, bc + 3):
                if grid[r][c] is not None:
                    block_digits.add(grid[r][c])
        if len(block_digits) == 9:
            complete_blocks.append(b)

    for r in range(9):
        for c in range(9):
            if grid[r][c] is None:
                pending_cells.append({"row": r, "col": c})

    filled = 81 - len(pending_cells)

    return {
        "complete_rows": complete_rows,
        "complete_cols": complete_cols,
        "complete_blocks": complete_blocks,
        "pending_cells": pending_cells,
        "filled": filled,
        "progress": filled / 81.0
    }


# =============================================================================
# COMPARE REPRESENTATIONS
# =============================================================================

def compare_representations():
    """Compare how different representations encode the same puzzle."""
    print("=" * 70)
    print("COMPARING NESTED DATA REPRESENTATIONS")
    print("=" * 70)

    store = create_store()
    grid = [[cell for cell in row] for row in PUZZLE_9x9_HARD]

    representations = [
        ("Constraint-Centric", to_constraint_centric),
        ("Digit-Centric", to_digit_centric),
        ("Cell-Context", to_cell_context),
        ("Hierarchical Bands", to_hierarchical_bands),
        ("Satisfaction State", to_satisfaction_state),
    ]

    vectors = {}

    print(f"\nPuzzle: {81 - count_empty(grid)} filled cells")

    for name, converter in representations:
        data = converter(grid)
        vec = store.encoder.encode_data(data)

        vectors[name] = vec
        print(f"\n{name}:")
        print(f"  Vector norm: {np.linalg.norm(vec):.1f}")
        print(f"  Non-zero: {np.sum(vec != 0)}")

    # Compare similarities between representations
    print("\n" + "-" * 50)
    print("Cross-representation similarities:")
    names = list(vectors.keys())
    for i, n1 in enumerate(names):
        for n2 in names[i+1:]:
            sim = similarity(vectors[n1], vectors[n2])
            print(f"  {n1[:15]:15s} vs {n2[:15]:15s}: {sim:.4f}")


def test_representation_for_choice_scoring():
    """Test which representation best scores choices."""
    print("\n" + "=" * 70)
    print("CHOICE SCORING WITH DIFFERENT REPRESENTATIONS")
    print("=" * 70)

    store = create_store()
    grid = [[cell for cell in row] for row in PUZZLE_9x9_HARD]

    # Find first decision point
    for r in range(9):
        for c in range(9):
            if grid[r][c] is None:
                options = list(get_available_digits_9x9(grid, r, c))
                if len(options) > 1:
                    print(f"\nDecision point: ({r},{c}) with options {options}")

                    # For each representation, see how it distinguishes choices
                    representations = [
                        ("Constraint-Centric", to_constraint_centric),
                        ("Digit-Centric", to_digit_centric),
                        ("Satisfaction State", to_satisfaction_state),
                    ]

                    for rep_name, converter in representations:
                        print(f"\n  {rep_name}:")

                        # Encode current state
                        current_data = converter(grid)
                        current_vec = store.encoder.encode_data(current_data)

                        # Encode after each choice
                        scores = {}
                        for digit in options:
                            test_grid = [[cell for cell in row] for row in grid]
                            test_grid[r][c] = digit

                            new_data = converter(test_grid)
                            new_vec = store.encoder.encode_data(new_data)

                            # How much did it change?
                            delta = np.linalg.norm(new_vec - current_vec)
                            sim = similarity(current_vec, new_vec)

                            scores[digit] = (delta, sim)

                        for digit in options:
                            delta, sim = scores[digit]
                            print(f"    Digit {digit}: delta={delta:.1f}, sim={sim:.4f}")

                    return


def test_constraint_centric_templates():
    """
    Use constraint-centric representation with template matching.

    Idea: A "needs" list that's EMPTY means constraint is satisfied.
    A "has" list that matches {1-9} is complete.
    """
    print("\n" + "=" * 70)
    print("CONSTRAINT-CENTRIC TEMPLATE MATCHING")
    print("=" * 70)

    store = create_store()

    # Build "complete constraint" template
    complete_has = store.encoder.encode_data({"has": list(range(1, 10)), "needs": []})
    print(f"\nComplete constraint template norm: {np.linalg.norm(complete_has):.1f}")

    # Test puzzle
    grid = [[cell for cell in row] for row in PUZZLE_9x9_HARD]
    constraint_data = to_constraint_centric(grid)

    print("\nRow similarities to complete template:")
    for r in range(9):
        row_data = constraint_data["rows"][f"r{r}"]
        row_vec = store.encoder.encode_data(row_data)
        sim = similarity(row_vec, complete_has)
        has_count = len(row_data["has"])
        print(f"  Row {r}: sim={sim:.4f}, has={has_count}/9")


def test_needs_list_for_scoring():
    """
    Use the "needs" list from constraint-centric representation.

    Idea: A choice is good if it REDUCES the "needs" lists.
    We can measure this by encoding the needs and checking similarity.
    """
    print("\n" + "=" * 70)
    print("USING 'NEEDS' LISTS FOR SCORING")
    print("=" * 70)

    store = create_store()
    grid = [[cell for cell in row] for row in PUZZLE_9x9_HARD]

    # Find decision point
    for r in range(9):
        for c in range(9):
            if grid[r][c] is None:
                options = list(get_available_digits_9x9(grid, r, c))
                if len(options) > 1:
                    print(f"\nDecision point: ({r},{c}) with options {options}")

                    # Current needs
                    current = to_constraint_centric(grid)
                    row_needs = set(current["rows"][f"r{r}"]["needs"])
                    col_needs = set(current["cols"][f"c{c}"]["needs"])

                    br, bc = (r // 3) * 3, (c // 3) * 3
                    block_idx = (r // 3) * 3 + (c // 3)
                    block_needs = set(current["blocks"][f"b{block_idx}"]["needs"])

                    print(f"\n  Row {r} needs: {sorted(row_needs)}")
                    print(f"  Col {c} needs: {sorted(col_needs)}")
                    print(f"  Block {block_idx} needs: {sorted(block_needs)}")

                    # Score each choice by how it impacts "needs"
                    print(f"\n  Choice impact on needs:")
                    for digit in options:
                        # After placing digit, it's removed from all three needs lists
                        new_row_needs = row_needs - {digit}
                        new_col_needs = col_needs - {digit}
                        new_block_needs = block_needs - {digit}

                        # Total reduction
                        reduction = (
                            (1 if digit in row_needs else 0) +
                            (1 if digit in col_needs else 0) +
                            (1 if digit in block_needs else 0)
                        )

                        # Encode the new needs lists
                        new_needs = {
                            "row": list(new_row_needs),
                            "col": list(new_col_needs),
                            "block": list(new_block_needs)
                        }
                        new_vec = store.encoder.encode_data(new_needs)

                        print(f"    Digit {digit}: reduces needs by {reduction}, new_needs_norm={np.linalg.norm(new_vec):.1f}")

                    return


def test_deep_nested_structure():
    """
    Test REALLY deep nesting to see if it helps.

    Structure:
    {
        "puzzle": {
            "band_0": {
                "row_0": {
                    "cell_0": {"value": 5, "options": []},
                    "cell_1": {"value": 3, "options": []},
                    "cell_2": {"value": None, "options": [1,2,4,7]},
                    ...
                },
                ...
            },
            ...
        }
    }
    """
    print("\n" + "=" * 70)
    print("DEEP NESTED STRUCTURE")
    print("=" * 70)

    store = create_store()
    grid = [[cell for cell in row] for row in PUZZLE_9x9_HARD]

    def to_deep_nested(g):
        puzzle = {"bands": {}}

        for band in range(3):
            band_data = {"rows": {}}

            for row_in_band in range(3):
                r = band * 3 + row_in_band
                row_data = {"cells": {}}

                for c in range(9):
                    if g[r][c] is not None:
                        cell_data = {"value": g[r][c]}
                    else:
                        opts = list(get_available_digits_9x9(g, r, c))
                        cell_data = {"options": opts, "option_count": len(opts)}

                    row_data["cells"][f"c{c}"] = cell_data

                band_data["rows"][f"r{r}"] = row_data

            puzzle["bands"][f"band_{band}"] = band_data

        return {"puzzle": puzzle}

    deep_data = to_deep_nested(grid)
    deep_vec = store.encoder.encode_data(deep_data)

    print(f"\nDeep nested encoding:")
    print(f"  Norm: {np.linalg.norm(deep_vec):.1f}")
    print(f"  Non-zero: {np.sum(deep_vec != 0)}")

    # Test unbinding through layers
    print("\nUnbinding through layers:")

    # puzzle -> bands -> band_0 -> rows -> r0 -> cells -> c0 -> value
    keys = ["puzzle", "bands", "band_0", "rows", "r0", "cells", "c0", "value"]

    current = deep_vec
    for key in keys:
        key_vec = store.vector_manager.get_vector(key)
        current = current * key_vec  # unbind
        print(f"  After unbind '{key}': norm={np.linalg.norm(current):.1f}")

    # Check similarity to expected value
    expected_value = grid[0][0]  # Should be None for empty or digit
    if expected_value is not None:
        digit_vec = store.vector_manager.get_vector(str(expected_value))
        sim = similarity(current, digit_vec)
        print(f"\n  Similarity to {expected_value}: {sim:.4f}")


class GoalOrientedConstraintSolver:
    """
    Uses constraint-centric representation but scores toward a GOAL state.

    Goal: All constraints should have has=[1-9], needs=[]
    Score choices by how much they move us toward this goal.
    """

    def __init__(self, dimensions: int = 16384, verbose: bool = True):
        self.store = create_store(dimensions)
        self.verbose = verbose
        self.backtracks = 0

        # Build goal template: each constraint is complete
        self.complete_constraint = self.store.encoder.encode_data({
            "has": list(range(1, 10)),
            "needs": []
        })

    def propagate(self, grid):
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
        return True

    def score_choice(self, grid, r, c, digit):
        """Score by similarity of affected constraints to complete template."""
        test_grid = [[cell for cell in row] for row in grid]
        test_grid[r][c] = digit

        new_state = to_constraint_centric(test_grid)

        # Score the three affected constraints
        row_data = new_state['rows'][f'r{r}']
        col_data = new_state['cols'][f'c{c}']
        block_idx = (r // 3) * 3 + (c // 3)
        block_data = new_state['blocks'][f'b{block_idx}']

        row_vec = self.store.encoder.encode_data(row_data)
        col_vec = self.store.encoder.encode_data(col_data)
        block_vec = self.store.encoder.encode_data(block_data)

        row_sim = similarity(row_vec, self.complete_constraint)
        col_sim = similarity(col_vec, self.complete_constraint)
        block_sim = similarity(block_vec, self.complete_constraint)

        return row_sim + col_sim + block_sim

    def solve(self, puzzle):
        grid = [[cell for cell in row] for row in puzzle]
        if not self.propagate(grid):
            return False, [[0]*9 for _ in range(9)]

        result = self.solve_recursive(grid, 0)
        if result:
            return True, result
        return False, [[0]*9 for _ in range(9)]

    def solve_recursive(self, grid, depth):
        if count_empty(grid) == 0:
            return grid

        best = None
        best_count = 10
        for r in range(9):
            for c in range(9):
                if grid[r][c] is None:
                    opts = list(get_available_digits_9x9(grid, r, c))
                    if not opts:
                        return None
                    if len(opts) < best_count:
                        best_count = len(opts)
                        best = (r, c, opts)

        if best is None:
            return grid

        r, c, options = best

        # Score each option toward goal
        scored = [(self.score_choice(grid, r, c, d), d) for d in options]
        scored.sort(reverse=True)  # Higher = closer to complete

        for score, digit in scored:
            test_grid = [[cell for cell in row] for row in grid]
            test_grid[r][c] = digit
            if not self.propagate(test_grid):
                continue

            result = self.solve_recursive(test_grid, depth + 1)
            if result:
                return result
            self.backtracks += 1

        return None


class ConstraintCentricSolver:
    """
    Solver using constraint-centric representation.

    Key insight: The constraint-centric encoding captures
    "what we have" and "what we need" for each constraint.

    Adding a digit changes both lists, and the encoding
    change varies based on how the digit interacts with
    existing digits.
    """

    def __init__(self, dimensions: int = 16384, verbose: bool = True):
        self.store = create_store(dimensions)
        self.verbose = verbose
        self.backtracks = 0

    def propagate(self, grid):
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
        return True

    def score_choice(self, grid, r, c, digit):
        """Score by how the constraint-centric encoding changes."""
        # Current encoding
        current_data = to_constraint_centric(grid)
        current_vec = self.store.encoder.encode_data(current_data)

        # After choice
        test_grid = [[cell for cell in row] for row in grid]
        test_grid[r][c] = digit

        new_data = to_constraint_centric(test_grid)
        new_vec = self.store.encoder.encode_data(new_data)

        # Higher similarity = less change = more "natural" fit?
        return similarity(current_vec, new_vec)

    def solve(self, puzzle):
        grid = [[cell for cell in row] for row in puzzle]
        if not self.propagate(grid):
            return False, [[0]*9 for _ in range(9)]

        result = self.solve_recursive(grid, 0)
        if result:
            return True, result
        return False, [[0]*9 for _ in range(9)]

    def solve_recursive(self, grid, depth):
        if count_empty(grid) == 0:
            return grid

        best = None
        best_count = 10
        for r in range(9):
            for c in range(9):
                if grid[r][c] is None:
                    opts = list(get_available_digits_9x9(grid, r, c))
                    if not opts:
                        return None
                    if len(opts) < best_count:
                        best_count = len(opts)
                        best = (r, c, opts)

        if best is None:
            return grid

        r, c, options = best

        # Score each option
        scored = [(self.score_choice(grid, r, c, d), d) for d in options]
        scored.sort(reverse=True)  # Higher similarity first

        for score, digit in scored:
            test_grid = [[cell for cell in row] for row in grid]
            test_grid[r][c] = digit
            if not self.propagate(test_grid):
                continue

            result = self.solve_recursive(test_grid, depth + 1)
            if result:
                return result
            self.backtracks += 1

        return None


class HybridNestedSolver:
    """
    Hybrid: Use nested structure for CONTEXT but simple sets for SCORING.

    Encode as nested:
    {
        "constraints": {
            "row_0": set([filled digits]),
            "col_0": set([filled digits]),
            "block_0": set([filled digits]),
            ...
        }
    }

    Score by: similarity of (row_set ∪ digit) to complete, summed for all 3.
    """

    def __init__(self, dimensions: int = 16384, verbose: bool = True):
        self.store = create_store(dimensions)
        self.verbose = verbose
        self.backtracks = 0

        # Cache digit vectors
        self.digit_vecs = {d: self.store.vector_manager.get_vector(str(d)) for d in range(1, 10)}

        # Complete template
        complete = np.sum([self.digit_vecs[d] for d in range(1, 10)], axis=0)
        self.complete = np.where(complete > 0, 1, np.where(complete < 0, -1, 0)).astype(np.int8)

    def encode_set(self, digits: Set[int]) -> np.ndarray:
        if not digits:
            return np.zeros(self.store.dimensions, dtype=np.int8)
        vecs = [self.digit_vecs[d] for d in digits]
        bundled = np.sum(vecs, axis=0)
        return np.where(bundled > 0, 1, np.where(bundled < 0, -1, 0)).astype(np.int8)

    def propagate(self, grid):
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
        return True

    def score_choice(self, grid, r, c, digit):
        """Score by template matching on all 27 constraints."""
        # Get all constraint sets AFTER the choice
        test_grid = [[cell for cell in row] for row in grid]
        test_grid[r][c] = digit

        total_score = 0.0

        # All 9 rows
        for row in range(9):
            row_digits = {test_grid[row][col] for col in range(9) if test_grid[row][col] is not None}
            row_vec = self.encode_set(row_digits)
            total_score += similarity(row_vec, self.complete)

        # All 9 cols
        for col in range(9):
            col_digits = {test_grid[row][col] for row in range(9) if test_grid[row][col] is not None}
            col_vec = self.encode_set(col_digits)
            total_score += similarity(col_vec, self.complete)

        # All 9 blocks
        for b in range(9):
            br, bc = (b // 3) * 3, (b % 3) * 3
            block_digits = set()
            for rr in range(br, br + 3):
                for cc in range(bc, bc + 3):
                    if test_grid[rr][cc] is not None:
                        block_digits.add(test_grid[rr][cc])
            block_vec = self.encode_set(block_digits)
            total_score += similarity(block_vec, self.complete)

        return total_score

    def solve(self, puzzle):
        grid = [[cell for cell in row] for row in puzzle]
        if not self.propagate(grid):
            return False, [[0]*9 for _ in range(9)]

        result = self.solve_recursive(grid, 0)
        if result:
            return True, result
        return False, [[0]*9 for _ in range(9)]

    def solve_recursive(self, grid, depth):
        if count_empty(grid) == 0:
            return grid

        best = None
        best_count = 10
        for r in range(9):
            for c in range(9):
                if grid[r][c] is None:
                    opts = list(get_available_digits_9x9(grid, r, c))
                    if not opts:
                        return None
                    if len(opts) < best_count:
                        best_count = len(opts)
                        best = (r, c, opts)

        if best is None:
            return grid

        r, c, options = best

        scored = [(self.score_choice(grid, r, c, d), d) for d in options]
        scored.sort(reverse=True)

        for score, digit in scored:
            test_grid = [[cell for cell in row] for row in grid]
            test_grid[r][c] = digit
            if not self.propagate(test_grid):
                continue

            result = self.solve_recursive(test_grid, depth + 1)
            if result:
                return result
            self.backtracks += 1

        return None


class PathAwareSolver:
    """
    Encode the decision PATH using chained mode.

    Idea: Good paths have characteristic "shapes".
    Use chained encoding to capture path structure.
    """

    def __init__(self, dimensions: int = 16384, verbose: bool = True):
        self.store = create_store(dimensions)
        self.verbose = verbose
        self.backtracks = 0

        # For template matching
        self.digit_vecs = {d: self.store.vector_manager.get_vector(str(d)) for d in range(1, 10)}
        complete = np.sum([self.digit_vecs[d] for d in range(1, 10)], axis=0)
        self.complete = np.where(complete > 0, 1, np.where(complete < 0, -1, 0)).astype(np.int8)

        # For path encoding
        self.path_history = []
        self.good_path_prototype = None

    def encode_set(self, digits: Set[int]) -> np.ndarray:
        if not digits:
            return np.zeros(self.store.dimensions, dtype=np.int8)
        vecs = [self.digit_vecs[d] for d in digits]
        bundled = np.sum(vecs, axis=0)
        return np.where(bundled > 0, 1, np.where(bundled < 0, -1, 0)).astype(np.int8)

    def encode_decision(self, r: int, c: int, digit: int, option_count: int) -> Dict:
        """Encode a decision as a structured object."""
        return {
            "position": {"row": r, "col": c},
            "digit": digit,
            "options": option_count,
            "block": (r // 3) * 3 + (c // 3)
        }

    def propagate(self, grid):
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
        return True

    def template_score(self, grid, r, c, digit):
        """Score by template matching on 3 affected constraints."""
        row_digits = {grid[r][cc] for cc in range(9) if grid[r][cc] is not None} | {digit}
        col_digits = {grid[rr][c] for rr in range(9) if grid[rr][c] is not None} | {digit}

        br, bc = (r // 3) * 3, (c // 3) * 3
        block_digits = {digit}
        for rr in range(br, br + 3):
            for cc in range(bc, bc + 3):
                if grid[rr][cc] is not None:
                    block_digits.add(grid[rr][cc])

        row_sim = similarity(self.encode_set(row_digits), self.complete)
        col_sim = similarity(self.encode_set(col_digits), self.complete)
        block_sim = similarity(self.encode_set(block_digits), self.complete)

        return row_sim + col_sim + block_sim

    def solve(self, puzzle):
        grid = [[cell for cell in row] for row in puzzle]
        if not self.propagate(grid):
            return False, [[0]*9 for _ in range(9)]

        self.path_history = []
        result = self.solve_recursive(grid, 0)

        if result:
            # Encode the successful path
            if self.path_history:
                path_data = [self.encode_decision(r, c, d, n) for r, c, d, n in self.path_history]
                self.good_path_prototype = self.store.encoder.encode_list(
                    path_data, mode=ListEncodeMode.CHAINED
                )
                if self.verbose:
                    print(f"  Good path length: {len(self.path_history)}")
                    print(f"  Good path prototype norm: {np.linalg.norm(self.good_path_prototype):.1f}")

            return True, result
        return False, [[0]*9 for _ in range(9)]

    def solve_recursive(self, grid, depth):
        if count_empty(grid) == 0:
            return grid

        best = None
        best_count = 10
        for r in range(9):
            for c in range(9):
                if grid[r][c] is None:
                    opts = list(get_available_digits_9x9(grid, r, c))
                    if not opts:
                        return None
                    if len(opts) < best_count:
                        best_count = len(opts)
                        best = (r, c, opts)

        if best is None:
            return grid

        r, c, options = best

        scored = [(self.template_score(grid, r, c, d), d) for d in options]
        scored.sort(reverse=True)

        for score, digit in scored:
            test_grid = [[cell for cell in row] for row in grid]
            test_grid[r][c] = digit
            if not self.propagate(test_grid):
                continue

            # Record this decision
            self.path_history.append((r, c, digit, len(options)))

            result = self.solve_recursive(test_grid, depth + 1)
            if result:
                return result

            # Backtrack - remove from path
            self.path_history.pop()
            self.backtracks += 1

        return None


def compare_all_deep_solvers():
    """Compare constraint-centric solver with others."""
    print("\n" + "=" * 70)
    print("DEEP NESTED SOLVER COMPARISON")
    print("=" * 70)

    from approach_22_hierarchical import TemplateMatchingSolver

    solvers = [
        ("Template Matching (3 constraints)", TemplateMatchingSolver(verbose=False)),
        ("Path-Aware (chained encoding)", PathAwareSolver(verbose=True)),
    ]

    for name, solver in solvers:
        start = time.time()
        success, result = solver.solve(PUZZLE_9x9_HARD)
        elapsed = time.time() - start

        print(f"\n{name}:")
        print(f"  Solved: {success}")
        print(f"  Backtracks: {solver.backtracks}")
        print(f"  Time: {elapsed:.3f}s")
        if success:
            print(f"  Valid: {validate_9x9(result)}")


def analyze_why_constraint_centric_discriminates():
    """Understand why constraint-centric gives different scores."""
    print("\n" + "=" * 70)
    print("WHY CONSTRAINT-CENTRIC DISCRIMINATES")
    print("=" * 70)

    store = create_store()
    grid = [[cell for cell in row] for row in PUZZLE_9x9_HARD]

    # Find decision point
    for r in range(9):
        for c in range(9):
            if grid[r][c] is None:
                options = list(get_available_digits_9x9(grid, r, c))
                if len(options) > 1:
                    print(f"\nDecision point: ({r},{c}) with options {options}")

                    # Current state
                    current = to_constraint_centric(grid)

                    print(f"\nCurrent constraint state:")
                    print(f"  Row {r} has: {current['rows'][f'r{r}']['has']}")
                    print(f"  Row {r} needs: {current['rows'][f'r{r}']['needs']}")

                    block_idx = (r // 3) * 3 + (c // 3)
                    print(f"  Block {block_idx} has: {current['blocks'][f'b{block_idx}']['has']}")

                    # After each choice
                    print(f"\nAfter each choice:")
                    for digit in options:
                        test_grid = [[cell for cell in row] for row in grid]
                        test_grid[r][c] = digit

                        new_state = to_constraint_centric(test_grid)

                        # Show changes
                        new_row_has = new_state['rows'][f'r{r}']['has']
                        new_block_has = new_state['blocks'][f'b{block_idx}']['has']

                        print(f"\n  Digit {digit}:")
                        print(f"    Row has: {sorted(new_row_has)}")
                        print(f"    Block has: {sorted(new_block_has)}")

                        # The key: how does the encoding differ?
                        current_vec = store.encoder.encode_data(current)
                        new_vec = store.encoder.encode_data(new_state)

                        # Component-wise analysis
                        diff = new_vec - current_vec
                        pos_changes = np.sum(diff > 0)
                        neg_changes = np.sum(diff < 0)

                        print(f"    Encoding: +{pos_changes} components, -{neg_changes} components")

                    return


def main():
    compare_representations()
    test_representation_for_choice_scoring()
    test_constraint_centric_templates()
    test_needs_list_for_scoring()
    test_deep_nested_structure()
    analyze_why_constraint_centric_discriminates()
    compare_all_deep_solvers()


if __name__ == "__main__":
    main()
