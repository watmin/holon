#!/usr/bin/env python3
"""
Approach 38: Orthogonal Spaces via Binding

THE PROBLEM WE DISCOVERED:
- Bundle(8 digits) ≈ Bundle(9 digits) - too similar!
- Violation manifold collapsed onto valid manifold

THE INSIGHT:
Binding creates ORTHOGONAL vectors: bind(A, B) ⊥ A and ⊥ B
Maybe we can create orthogonal "validity" and "violation" spaces.

NEW APPROACH:
1. Define a VALIDITY AXIS using binding
2. Define a VIOLATION AXIS using binding
3. Project candidates onto these axes
4. Valid choices should project HIGH on validity, LOW on violation
"""

import sys
sys.path.insert(0, "/home/watmin/work/holon")

from typing import List, Optional, Set, Dict
import numpy as np
import time
import random

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


# =============================================================================
# EXPERIMENT 1: Creating Orthogonal Validity/Violation Axes
# =============================================================================

def test_orthogonal_axes():
    """
    Use binding to create orthogonal subspaces.
    """
    print("=" * 70)
    print("EXPERIMENT 1: Orthogonal Validity/Violation Axes")
    print("=" * 70)

    store = create_store()

    # Create semantic axes
    validity_axis = store.vector_manager.get_vector("VALID")
    violation_axis = store.vector_manager.get_vector("VIOLATION")

    print(f"Validity ⊥ Violation: {similarity(validity_axis, violation_axis):.4f}")

    # Create digit representations bound to each axis
    digit_vecs = {d: store.vector_manager.get_vector(f"d{d}") for d in range(1, 10)}

    # Valid: each digit bound to validity axis
    valid_digits = {d: store.bind(digit_vecs[d], validity_axis) for d in range(1, 10)}

    # Violation patterns: duplicates bound to violation axis
    violation_patterns = {}
    for d in range(1, 10):
        # A duplicate is: d appears twice, bound to violation
        dup = store.bind(store.bind(digit_vecs[d], digit_vecs[d]), violation_axis)
        violation_patterns[d] = dup

    # Check orthogonality
    print(f"\nValid digit vs Violation pattern for same digit:")
    for d in [1, 5, 9]:
        sim = similarity(valid_digits[d], violation_patterns[d])
        print(f"  Digit {d}: sim(valid, violation) = {sim:.4f}")

    # Complete valid set
    complete_valid = store.bundle([valid_digits[d] for d in range(1, 10)])

    # Some violation sets
    some_violations = store.bundle([violation_patterns[d] for d in range(1, 10)])

    print(f"\nComplete sets:")
    print(f"  sim(all_valid, all_violations) = {similarity(complete_valid, some_violations):.4f}")


# =============================================================================
# EXPERIMENT 2: State Encoding with Validity Binding
# =============================================================================

def test_validity_bound_encoding():
    """
    Encode grid state with validity binding.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Validity-Bound State Encoding")
    print("=" * 70)

    store = create_store()

    validity_axis = store.vector_manager.get_vector("VALID")
    violation_axis = store.vector_manager.get_vector("VIOLATION")
    digit_vecs = {d: store.vector_manager.get_vector(f"d{d}") for d in range(1, 10)}
    pos_vecs = {(r, c): store.vector_manager.get_vector(f"p{r}{c}")
                for r in range(9) for c in range(9)}

    def encode_valid_placement(r, c, d):
        """Encode a valid digit placement."""
        return store.bind(store.bind(pos_vecs[(r, c)], digit_vecs[d]), validity_axis)

    def encode_violation(r, c, d):
        """Encode a violation (e.g., duplicate in constraint)."""
        return store.bind(store.bind(pos_vecs[(r, c)], digit_vecs[d]), violation_axis)

    def encode_grid_with_validity(grid, is_valid_func):
        """Encode grid, marking valid/invalid placements."""
        components = []
        for r in range(9):
            for c in range(9):
                if grid[r][c] is not None:
                    if is_valid_func(grid, r, c):
                        components.append(encode_valid_placement(r, c, grid[r][c]))
                    else:
                        components.append(encode_violation(r, c, grid[r][c]))
        if components:
            return store.bundle(components)
        return np.zeros(store.dimensions, dtype=np.int8)

    def check_validity(grid, r, c):
        """Check if placement at (r,c) is valid."""
        d = grid[r][c]
        # Check for duplicates in row
        for cc in range(9):
            if cc != c and grid[r][cc] == d:
                return False
        # Check col
        for rr in range(9):
            if rr != r and grid[rr][c] == d:
                return False
        # Check block
        br, bc = (r // 3) * 3, (c // 3) * 3
        for rr in range(br, br + 3):
            for cc in range(bc, bc + 3):
                if (rr, cc) != (r, c) and grid[rr][cc] == d:
                    return False
        return True

    # Test on valid grid
    valid_grid = [[cell for cell in row] for row in PUZZLE_9x9_HARD]
    valid_vec = encode_grid_with_validity(valid_grid, check_validity)

    # Create invalid grid
    invalid_grid = [[cell for cell in row] for row in PUZZLE_9x9_HARD]
    # Add a duplicate
    for c in range(9):
        if invalid_grid[0][c] is None:
            for cc in range(9):
                if invalid_grid[0][cc] is not None:
                    invalid_grid[0][c] = invalid_grid[0][cc]
                    break
            break

    invalid_vec = encode_grid_with_validity(invalid_grid, check_validity)

    # Project onto axes
    valid_projection = similarity(valid_vec, validity_axis)
    violation_projection = similarity(valid_vec, violation_axis)

    print(f"\nValid grid projections:")
    print(f"  onto VALIDITY axis: {valid_projection:.4f}")
    print(f"  onto VIOLATION axis: {violation_projection:.4f}")

    valid_projection2 = similarity(invalid_vec, validity_axis)
    violation_projection2 = similarity(invalid_vec, violation_axis)

    print(f"\nInvalid grid projections:")
    print(f"  onto VALIDITY axis: {valid_projection2:.4f}")
    print(f"  onto VIOLATION axis: {violation_projection2:.4f}")


# =============================================================================
# EXPERIMENT 3: Constraint Satisfaction as Geometric Distance
# =============================================================================

def test_constraint_geometry():
    """
    Encode constraints geometrically.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Constraint Satisfaction Geometry")
    print("=" * 70)

    store = create_store()

    digit_vecs = {d: store.vector_manager.get_vector(f"d{d}") for d in range(1, 10)}

    # IDEA: Instead of bundling digits, BIND them in sequence
    # This creates a unique "signature" for each combination

    def encode_digit_sequence(digits):
        """Bind digits in sequence to create unique signature."""
        if not digits:
            return np.zeros(store.dimensions, dtype=np.int8)
        sorted_digits = sorted(digits)
        result = digit_vecs[sorted_digits[0]].copy()
        for d in sorted_digits[1:]:
            result = store.bind(result, digit_vecs[d])
        return result

    # Complete sequence (1,2,3,4,5,6,7,8,9)
    complete = encode_digit_sequence(range(1, 10))
    print(f"Complete sequence norm: {np.linalg.norm(complete):.1f}")

    # Missing one digit
    missing_9 = encode_digit_sequence(range(1, 9))
    missing_5 = encode_digit_sequence([1,2,3,4,6,7,8,9])

    # Binding creates VERY different vectors
    print(f"\nBinding creates orthogonal sequences:")
    print(f"  sim(complete, missing_9) = {similarity(complete, missing_9):.4f}")
    print(f"  sim(complete, missing_5) = {similarity(complete, missing_5):.4f}")
    print(f"  sim(missing_9, missing_5) = {similarity(missing_9, missing_5):.4f}")

    # Problem: binding is TOO orthogonal - no gradient!
    # The 8-digit sequence is completely unrelated to 9-digit

    # Alternative: use binding with a "completion" marker
    completion_marker = store.vector_manager.get_vector("COMPLETE")

    def encode_with_completion_marker(digits):
        """Bundle digits and bind with completion if complete."""
        vec = store.bundle([digit_vecs[d] for d in digits])
        if len(digits) == 9 and set(digits) == set(range(1, 10)):
            vec = store.bind(vec, completion_marker)
        return vec

    complete_marked = encode_with_completion_marker(range(1, 10))
    incomplete = encode_with_completion_marker(range(1, 9))

    print(f"\nWith completion marker:")
    print(f"  sim(complete_marked, incomplete) = {similarity(complete_marked, incomplete):.4f}")
    print(f"  sim(complete_marked, completion_marker) = {similarity(complete_marked, completion_marker):.4f}")
    print(f"  sim(incomplete, completion_marker) = {similarity(incomplete, completion_marker):.4f}")


# =============================================================================
# EXPERIMENT 4: Hierarchical Binding Structure
# =============================================================================

def test_hierarchical_binding():
    """
    Create hierarchical binding structure for constraints.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Hierarchical Binding Structure")
    print("=" * 70)

    store = create_store()

    digit_vecs = {d: store.vector_manager.get_vector(f"d{d}") for d in range(1, 10)}
    row_marker = store.vector_manager.get_vector("ROW")
    col_marker = store.vector_manager.get_vector("COL")
    block_marker = store.vector_manager.get_vector("BLOCK")

    def encode_constraint_state(unit_type, digits):
        """Encode a constraint unit's state."""
        marker = {"row": row_marker, "col": col_marker, "block": block_marker}[unit_type]
        digit_bundle = store.bundle([digit_vecs[d] for d in digits])
        # Bind digit set with unit type and count
        count_vec = store.vector_manager.get_vector(f"count_{len(digits)}")
        return store.bind(store.bind(digit_bundle, marker), count_vec)

    # Test: differentiate by completeness
    complete_row = encode_constraint_state("row", range(1, 10))
    partial_row_8 = encode_constraint_state("row", range(1, 9))
    partial_row_5 = encode_constraint_state("row", range(1, 6))

    print(f"Hierarchically bound constraint states:")
    print(f"  sim(complete, 8-digit) = {similarity(complete_row, partial_row_8):.4f}")
    print(f"  sim(complete, 5-digit) = {similarity(complete_row, partial_row_5):.4f}")
    print(f"  sim(8-digit, 5-digit) = {similarity(partial_row_8, partial_row_5):.4f}")

    # Check if different unit types are distinguished
    complete_col = encode_constraint_state("col", range(1, 10))
    complete_block = encode_constraint_state("block", range(1, 10))

    print(f"\nSame digits, different units:")
    print(f"  sim(row, col) = {similarity(complete_row, complete_col):.4f}")
    print(f"  sim(row, block) = {similarity(complete_row, complete_block):.4f}")


# =============================================================================
# EXPERIMENT 5: Projective Scoring
# =============================================================================

def test_projective_scoring():
    """
    Score choices by projection onto validity dimension.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: Projective Scoring")
    print("=" * 70)

    store = create_store()

    digit_vecs = {d: store.vector_manager.get_vector(f"d{d}") for d in range(1, 10)}

    # Define GOAL vector: all 9 digits present
    goal = store.bundle([digit_vecs[d] for d in range(1, 10)])

    # Define ANTI-GOAL: missing a digit (we'll create one per missing digit)
    anti_goals = {
        missing: store.bundle([digit_vecs[d] for d in range(1, 10) if d != missing])
        for missing in range(1, 10)
    }

    # Create SEPARATING hyperplane using negate
    # The idea: project goal - anti_goal to get direction toward goal

    def get_direction_toward_goal(current_digits, missing):
        """Get vector pointing toward adding the missing digit."""
        current_vec = store.bundle([digit_vecs[d] for d in current_digits])

        # The "missing direction" is: goal - current
        # But we can't do direct subtraction well in VSA

        # Alternative: similarity to goal with missing digit added
        with_missing = store.bundle([digit_vecs[d] for d in current_digits] + [digit_vecs[missing]])

        return similarity(with_missing, goal) - similarity(current_vec, goal)

    # Test: score adding different digits
    current = {1, 3, 5, 7}
    print(f"\nCurrent digits: {current}")
    print(f"Goal: {{1,2,3,4,5,6,7,8,9}}")
    print(f"\nScore for adding each missing digit:")

    for d in range(1, 10):
        if d not in current:
            score = get_direction_toward_goal(current, d)
            print(f"  Add {d}: score = {score:.4f}")


# =============================================================================
# EXPERIMENT 6: Solver with Orthogonal Scoring
# =============================================================================

def test_orthogonal_solver():
    """
    Solver using orthogonal projection scoring.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 6: Orthogonal Projection Solver")
    print("=" * 70)

    store = create_store()

    digit_vecs = {d: store.vector_manager.get_vector(f"d{d}") for d in range(1, 10)}
    goal = store.bundle([digit_vecs[d] for d in range(1, 10)])

    class OrthogonalSolver:
        def __init__(self):
            self.backtracks = 0

        def score_choice(self, grid, r, c, digit):
            """Score by improvement in goal projection."""
            total = 0.0

            # For each constraint unit, measure improvement
            for get_unit_digits in [
                lambda: {grid[r][cc] for cc in range(9) if grid[r][cc] is not None},  # row
                lambda: {grid[rr][c] for rr in range(9) if grid[rr][c] is not None},  # col
                lambda: {grid[rr][cc] for rr in range((r//3)*3, (r//3)*3+3)
                                      for cc in range((c//3)*3, (c//3)*3+3)
                         if grid[rr][cc] is not None},  # block
            ]:
                current = get_unit_digits()
                current_vec = store.bundle([digit_vecs[d] for d in current]) if current else np.zeros(store.dimensions, dtype=np.int8)
                new_vec = store.bundle([digit_vecs[d] for d in current] + [digit_vecs[digit]])

                # Improvement in goal similarity
                current_sim = similarity(current_vec, goal) if np.any(current_vec) else 0
                new_sim = similarity(new_vec, goal)
                total += new_sim - current_sim

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

    solver = OrthogonalSolver()
    start = time.time()
    result = solver.solve(PUZZLE_9x9_HARD)
    elapsed = time.time() - start

    print(f"\nOrthogonal solver:")
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
    test_orthogonal_axes()
    test_validity_bound_encoding()
    test_constraint_geometry()
    test_hierarchical_binding()
    test_projective_scoring()
    bt = test_orthogonal_solver()

    print("\n" + "=" * 70)
    print("ORTHOGONAL SPACES SUMMARY")
    print("=" * 70)
    print(f"""
THE PROBLEM WITH APPROACH 37:
- Bundle(8 digits) ≈ Bundle(9 digits) - similarity ~0.98
- Valid and violation manifolds COLLAPSED together
- No separation in the bundling space

WHAT BINDING PROVIDES:
- bind(A, B) is orthogonal to A and B
- Creates genuinely distinct subspaces
- But: TOO orthogonal - no gradient between partial and complete!

THE FUNDAMENTAL TENSION:
- We want partial → complete to be GRADUAL (for scoring)
- But we want valid ⊥ invalid (for separation)
- These goals CONFLICT in VSA

WHAT ACTUALLY WORKS:
The template matching approach (Approach 22) measures:
  Δsim = sim(current ∪ {{new}}, goal) - sim(current, goal)

This IS the "direction toward goal" projection.
It's essentially what the user envisioned, just expressed as
similarity rather than orthogonal axes.

RESULT:
Orthogonal solver backtracks: {bt}
Template matching baseline: 52

KEY INSIGHT:
The hyperspace geometry IS being exploited - just via similarity gradients,
not orthogonal projections. The geometry of "approaching the complete set"
is the same as "projecting toward the goal manifold."
""")


if __name__ == "__main__":
    main()
