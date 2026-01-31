#!/usr/bin/env python3
"""
Approach 33: Candidate Encoding

A different interpretation of the user's idea:
Instead of encoding {position: value}, encode {position: candidates}

For Sudoku, this captures:
- What digits COULD go at each position
- NOT just what's there, but what's POSSIBLE

This is more like constraint propagation encoded as data!
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
# EXPERIMENT 1: Encode Candidates Per Cell
# =============================================================================

def test_candidate_encoding():
    """
    Encode what's POSSIBLE at each cell, not just what's there.
    """
    print("=" * 70)
    print("EXPERIMENT 1: Candidate Encoding Per Cell")
    print("=" * 70)

    store = create_store()

    def get_candidates(grid, r, c):
        """Get candidate digits for a cell."""
        if grid[r][c] is not None:
            return {grid[r][c]}  # Already filled
        return get_available_digits_9x9(grid, r, c)

    def encode_cell_candidates(candidates: Set[int]) -> np.ndarray:
        """Encode the candidate set for a cell."""
        if not candidates:
            return np.zeros(store.dimensions, dtype=np.int8)
        digit_vecs = [store.vector_manager.get_vector(str(d)) for d in candidates]
        return store.bundle(digit_vecs)

    def encode_puzzle_candidates(grid) -> np.ndarray:
        """Encode entire puzzle as candidate sets per cell."""
        cell_encodings = []
        for r in range(9):
            for c in range(9):
                cands = get_candidates(grid, r, c)
                pos = store.vector_manager.get_vector(f"cell_{r}_{c}")
                cand_vec = encode_cell_candidates(cands)
                cell_encodings.append(store.bind(pos, cand_vec))

        return store.bundle(cell_encodings)

    # Encode puzzle
    grid = [[cell for cell in row] for row in PUZZLE_9x9_HARD]
    puzzle_vec = encode_puzzle_candidates(grid)

    print(f"\nPuzzle candidate encoding norm: {np.linalg.norm(puzzle_vec):.1f}")

    # Encode "solved" puzzle (all cells have exactly 1 candidate)
    # This is our goal state
    solved_template_cells = []
    for r in range(9):
        for c in range(9):
            pos = store.vector_manager.get_vector(f"cell_{r}_{c}")
            # In solved state, each cell has exactly 1 option
            single = store.vector_manager.get_vector("single_candidate")
            solved_template_cells.append(store.bind(pos, single))

    solved_template = store.bundle(solved_template_cells)

    print(f"Solved template norm: {np.linalg.norm(solved_template):.1f}")
    print(f"sim(puzzle, solved_template) = {similarity(puzzle_vec, solved_template):.4f}")

    # After propagation
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

    propagated_vec = encode_puzzle_candidates(grid)
    print(f"\nAfter propagation: sim = {similarity(propagated_vec, solved_template):.4f}")


# =============================================================================
# EXPERIMENT 2: Constraint Entropy Encoding
# =============================================================================

def test_entropy_encoding():
    """
    Encode constraints by their "entropy" - how constrained they are.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Constraint Entropy Encoding")
    print("=" * 70)

    store = create_store()

    def constraint_entropy(grid, unit_type: str, index: int) -> float:
        """
        Calculate entropy of a constraint unit.
        Lower = more constrained = closer to solved.
        """
        if unit_type == "row":
            cells = [(index, c) for c in range(9)]
        elif unit_type == "col":
            cells = [(r, index) for r in range(9)]
        else:  # block
            br, bc = (index // 3) * 3, (index % 3) * 3
            cells = [(br + i, bc + j) for i in range(3) for j in range(3)]

        total_options = 0
        for r, c in cells:
            if grid[r][c] is None:
                opts = get_available_digits_9x9(grid, r, c)
                total_options += len(opts)
            # Filled cells contribute 0

        return total_options

    def encode_grid_entropy(grid) -> np.ndarray:
        """Encode grid by constraint entropy levels."""
        entropy_vecs = []

        for r in range(9):
            ent = constraint_entropy(grid, "row", r)
            level = min(9, ent // 5)  # Bucket into levels
            vec = store.vector_manager.get_vector(f"row_entropy_{level}")
            entropy_vecs.append(vec)

        for c in range(9):
            ent = constraint_entropy(grid, "col", c)
            level = min(9, ent // 5)
            vec = store.vector_manager.get_vector(f"col_entropy_{level}")
            entropy_vecs.append(vec)

        for b in range(9):
            ent = constraint_entropy(grid, "block", b)
            level = min(9, ent // 5)
            vec = store.vector_manager.get_vector(f"block_entropy_{level}")
            entropy_vecs.append(vec)

        return store.bundle(entropy_vecs)

    # Encode solved state (all entropy = 0)
    solved_entropy_vecs = []
    for r in range(9):
        solved_entropy_vecs.append(store.vector_manager.get_vector("row_entropy_0"))
    for c in range(9):
        solved_entropy_vecs.append(store.vector_manager.get_vector("col_entropy_0"))
    for b in range(9):
        solved_entropy_vecs.append(store.vector_manager.get_vector("block_entropy_0"))
    solved_vec = store.bundle(solved_entropy_vecs)

    # Test on puzzle
    grid = [[cell for cell in row] for row in PUZZLE_9x9_HARD]
    puzzle_vec = encode_grid_entropy(grid)

    print(f"\nOriginal puzzle entropy similarity: {similarity(puzzle_vec, solved_vec):.4f}")

    # After propagation
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

    propagated_vec = encode_grid_entropy(grid)
    print(f"After propagation: {similarity(propagated_vec, solved_vec):.4f}")


# =============================================================================
# EXPERIMENT 3: Choice Impact Encoding
# =============================================================================

def test_choice_impact():
    """
    Encode choices by their IMPACT on constraint state.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Choice Impact Encoding")
    print("=" * 70)

    store = create_store()
    digit_vecs = {d: store.vector_manager.get_vector(str(d)) for d in range(1, 10)}

    def encode_impact(grid, r, c, digit):
        """
        Encode the IMPACT of placing a digit.
        This captures what changes, not just the resulting state.
        """
        # What gets eliminated from row/col/block
        row_elim = []
        for cc in range(9):
            if cc != c and grid[r][cc] is None:
                opts = get_available_digits_9x9(grid, r, cc)
                if digit in opts:
                    row_elim.append(cc)

        col_elim = []
        for rr in range(9):
            if rr != r and grid[rr][c] is None:
                opts = get_available_digits_9x9(grid, rr, c)
                if digit in opts:
                    col_elim.append(rr)

        br, bc = (r // 3) * 3, (c // 3) * 3
        block_elim = []
        for rr in range(br, br + 3):
            for cc in range(bc, bc + 3):
                if (rr, cc) != (r, c) and grid[rr][cc] is None:
                    opts = get_available_digits_9x9(grid, rr, cc)
                    if digit in opts:
                        block_elim.append((rr, cc))

        # Encode as structured impact
        impact = {
            "digit": digit,
            "row_eliminations": len(row_elim),
            "col_eliminations": len(col_elim),
            "block_eliminations": len(block_elim),
            "total_eliminations": len(row_elim) + len(col_elim) + len(block_elim),
        }

        return store.encoder.encode_data(impact)

    # "Good" impact: eliminates options, constrains puzzle
    good_impact = store.encoder.encode_data({
        "digit": 5,
        "row_eliminations": 3,
        "col_eliminations": 3,
        "block_eliminations": 2,
        "total_eliminations": 8,
    })

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

    # Find decision point and score by impact
    for r in range(9):
        for c in range(9):
            if grid[r][c] is None:
                opts = list(get_available_digits_9x9(grid, r, c))
                if len(opts) > 1:
                    print(f"\nDecision point ({r},{c}), options={opts}")

                    for digit in opts:
                        impact_vec = encode_impact(grid, r, c, digit)
                        score = similarity(impact_vec, good_impact)
                        print(f"  Digit {digit}: impact_score = {score:.4f}")

                    return


# =============================================================================
# EXPERIMENT 4: Constraint Network Encoding
# =============================================================================

def test_constraint_network():
    """
    Encode the constraint NETWORK - how cells relate to each other.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Constraint Network Encoding")
    print("=" * 70)

    store = create_store()

    def encode_cell_context(grid, r, c):
        """Encode a cell's constraint context."""
        if grid[r][c] is not None:
            # Filled cell: just its value
            return store.vector_manager.get_vector(f"filled_{grid[r][c]}")

        # Empty cell: encode its options and neighbors
        opts = get_available_digits_9x9(grid, r, c)

        # Encode: {options_count, row_fill, col_fill, block_fill}
        row_fill = sum(1 for cc in range(9) if grid[r][cc] is not None)
        col_fill = sum(1 for rr in range(9) if grid[rr][c] is not None)
        br, bc = (r // 3) * 3, (c // 3) * 3
        block_fill = sum(1 for rr in range(br, br+3) for cc in range(bc, bc+3)
                         if grid[rr][cc] is not None)

        context = {
            "options": len(opts),
            "row_fill": row_fill,
            "col_fill": col_fill,
            "block_fill": block_fill,
            "constraint_pressure": (row_fill + col_fill + block_fill) / 27.0,
        }

        return store.encoder.encode_data(context)

    # Test on puzzle
    grid = [[cell for cell in row] for row in PUZZLE_9x9_HARD]

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

    # "Solved" cell template
    solved_cell = store.encoder.encode_data({
        "options": 1,
        "row_fill": 9,
        "col_fill": 9,
        "block_fill": 9,
        "constraint_pressure": 1.0,
    })

    # Check similarities
    print("\nCell context similarities to 'solved' template:")
    for r in range(3):  # Just first few rows
        for c in range(9):
            cell_vec = encode_cell_context(grid, r, c)
            sim = similarity(cell_vec, solved_cell)
            status = "F" if grid[r][c] is not None else "E"
            print(f"  ({r},{c})[{status}]: {sim:.3f}", end="")
        print()


# =============================================================================
# MAIN
# =============================================================================

def main():
    test_candidate_encoding()
    test_entropy_encoding()
    test_choice_impact()
    test_constraint_network()

    print("\n" + "=" * 70)
    print("CANDIDATE ENCODING SUMMARY")
    print("=" * 70)
    print("""
ALTERNATIVE INTERPRETATIONS OF STRUCTURED ENCODING:

1. CANDIDATE ENCODING: {cell: possible_digits}
   - Captures WHAT COULD GO there, not just what's there
   - More like constraint propagation state

2. ENTROPY ENCODING: {constraint: entropy_level}
   - Captures HOW CONSTRAINED each unit is
   - Lower entropy = closer to solved

3. IMPACT ENCODING: {digit: eliminations_caused}
   - Captures the EFFECT of each choice
   - More eliminations = more constraining

4. NETWORK ENCODING: {cell: neighbor_fill_levels}
   - Captures the LOCAL CONTEXT of each cell
   - Higher pressure = more constrained

KEY INSIGHT:
All these structured encodings capture DIFFERENT aspects of Sudoku state.
But for SOLVING, what matters is simple: does this digit complete the set?

Template matching answers that directly:
  sim(current_digits ∪ {new}, complete) → score

The structured approaches encode more information, but that information
doesn't improve the core discrimination needed for ordering choices.
""")


if __name__ == "__main__":
    main()
