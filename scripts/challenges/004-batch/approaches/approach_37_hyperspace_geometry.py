#!/usr/bin/env python3
"""
Approach 37: Exploiting Hyperdimensional Geometry

THE USER'S VISION:
- Each vector is a line from origin into orthogonal directions
- Valid solutions cluster together in hyperspace
- Invalid solutions are distant from this cluster
- Use negation to PUSH invalids further away
- The database defines the "valid region"

HYPOTHESIS TO TEST:
1. Do valid Sudoku states cluster together?
2. Are invalid states farther from the valid cluster?
3. Can we define a "valid manifold" and measure distance to it?
4. Can negation of violations improve separation?
"""

import sys
sys.path.insert(0, "/home/watmin/work/holon")

from typing import List, Optional, Set, Tuple, Dict
from itertools import permutations
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
# EXPERIMENT 1: Do Valid States Cluster?
# =============================================================================

def test_valid_state_clustering():
    """
    Test if valid Sudoku partial states are closer to each other
    than to invalid states.
    """
    print("=" * 70)
    print("EXPERIMENT 1: Do Valid States Cluster in Hyperspace?")
    print("=" * 70)

    store = create_store()

    # Create base vectors for encoding
    digit_vecs = {d: store.vector_manager.get_vector(f"d{d}") for d in range(1, 10)}
    pos_vecs = {(r, c): store.vector_manager.get_vector(f"p{r}{c}")
                for r in range(9) for c in range(9)}

    def encode_grid(grid):
        """Encode a grid state as a single vector."""
        components = []
        for r in range(9):
            for c in range(9):
                if grid[r][c] is not None:
                    components.append(store.bind(pos_vecs[(r, c)], digit_vecs[grid[r][c]]))
        if components:
            return store.bundle(components)
        return np.zeros(store.dimensions, dtype=np.int8)

    # Generate some VALID partial grids (from actual solving)
    def generate_valid_partial(puzzle, n_moves=10):
        """Make n valid moves from puzzle."""
        grid = [[cell for cell in row] for row in puzzle]
        moves = 0
        while moves < n_moves:
            for r in range(9):
                for c in range(9):
                    if grid[r][c] is None:
                        opts = list(get_available_digits_9x9(grid, r, c))
                        if len(opts) == 1:
                            grid[r][c] = opts[0]
                            moves += 1
                            if moves >= n_moves:
                                return grid
            # If no forced moves, make one valid choice
            for r in range(9):
                for c in range(9):
                    if grid[r][c] is None:
                        opts = list(get_available_digits_9x9(grid, r, c))
                        if opts:
                            grid[r][c] = random.choice(opts)
                            moves += 1
                            break
                if moves >= n_moves:
                    break
        return grid

    # Generate some INVALID partial grids (introduce violations)
    def generate_invalid_partial(puzzle, n_violations=3):
        """Make moves that create violations."""
        grid = [[cell for cell in row] for row in puzzle]
        violations = 0
        for r in range(9):
            for c in range(9):
                if grid[r][c] is None and violations < n_violations:
                    # Pick a digit that's already in row/col/block
                    row_used = {grid[r][cc] for cc in range(9) if grid[r][cc] is not None}
                    if row_used:
                        grid[r][c] = random.choice(list(row_used))  # Duplicate!
                        violations += 1
        return grid

    # Generate samples
    print("\nGenerating valid and invalid partial grids...")
    valid_grids = [generate_valid_partial(PUZZLE_9x9_HARD, n_moves=5+i) for i in range(10)]
    invalid_grids = [generate_invalid_partial(PUZZLE_9x9_HARD, n_violations=1+i%3) for i in range(10)]

    # Encode all
    valid_vecs = [encode_grid(g) for g in valid_grids]
    invalid_vecs = [encode_grid(g) for g in invalid_grids]

    # Compute centroid of valid states
    valid_centroid = store.bundle(valid_vecs)

    # Measure distances
    valid_to_centroid = [similarity(v, valid_centroid) for v in valid_vecs]
    invalid_to_centroid = [similarity(v, valid_centroid) for v in invalid_vecs]

    print(f"\nSimilarity to VALID CENTROID:")
    print(f"  Valid states:   mean={np.mean(valid_to_centroid):.4f}, std={np.std(valid_to_centroid):.4f}")
    print(f"  Invalid states: mean={np.mean(invalid_to_centroid):.4f}, std={np.std(invalid_to_centroid):.4f}")

    # Also check pairwise similarities
    valid_pairwise = []
    for i in range(len(valid_vecs)):
        for j in range(i+1, len(valid_vecs)):
            valid_pairwise.append(similarity(valid_vecs[i], valid_vecs[j]))

    invalid_pairwise = []
    for i in range(len(invalid_vecs)):
        for j in range(i+1, len(invalid_vecs)):
            invalid_pairwise.append(similarity(invalid_vecs[i], invalid_vecs[j]))

    cross_pairwise = []
    for v in valid_vecs:
        for iv in invalid_vecs:
            cross_pairwise.append(similarity(v, iv))

    print(f"\nPairwise similarities:")
    print(f"  Valid-Valid:     mean={np.mean(valid_pairwise):.4f}")
    print(f"  Invalid-Invalid: mean={np.mean(invalid_pairwise):.4f}")
    print(f"  Valid-Invalid:   mean={np.mean(cross_pairwise):.4f}")


# =============================================================================
# EXPERIMENT 2: Violation Manifold - Encode What's WRONG
# =============================================================================

def test_violation_manifold():
    """
    Pre-encode ALL violation patterns.
    Measure distance FROM violations.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Violation Manifold")
    print("=" * 70)

    store = create_store()

    digit_vecs = {d: store.vector_manager.get_vector(f"d{d}") for d in range(1, 10)}
    pos_vecs = {(r, c): store.vector_manager.get_vector(f"p{r}{c}")
                for r in range(9) for c in range(9)}

    # A VIOLATION is: same digit in two cells of the same row/col/block
    # Encode violations as: bind(pos1, d) + bind(pos2, d) for same d

    def encode_row_violation(r, c1, c2, d):
        """Two cells in same row with same digit."""
        v1 = store.bind(pos_vecs[(r, c1)], digit_vecs[d])
        v2 = store.bind(pos_vecs[(r, c2)], digit_vecs[d])
        return store.bundle([v1, v2])

    # Generate ALL row violations (for one digit)
    print("\nGenerating violation patterns...")
    row_violations = []
    for r in range(9):
        for c1 in range(9):
            for c2 in range(c1+1, 9):
                for d in range(1, 10):
                    row_violations.append(encode_row_violation(r, c1, c2, d))

    print(f"  Generated {len(row_violations)} row violations")

    # Bundle into violation manifold
    # (sampling because 9*36*9 = 2916 is a lot)
    sampled = random.sample(row_violations, min(500, len(row_violations)))
    violation_manifold = store.bundle(sampled)

    print(f"  Violation manifold vector created from {len(sampled)} samples")

    # Now test: valid states should be FAR from violation manifold
    # Invalid states should be CLOSE

    def encode_grid(grid):
        components = []
        for r in range(9):
            for c in range(9):
                if grid[r][c] is not None:
                    components.append(store.bind(pos_vecs[(r, c)], digit_vecs[grid[r][c]]))
        if components:
            return store.bundle(components)
        return np.zeros(store.dimensions, dtype=np.int8)

    # Valid grid (just the puzzle)
    valid_vec = encode_grid(PUZZLE_9x9_HARD)

    # Invalid grid (introduce a violation)
    invalid_grid = [[cell for cell in row] for row in PUZZLE_9x9_HARD]
    # Find first empty cell in row 0 and put a duplicate
    for c in range(9):
        if invalid_grid[0][c] is None:
            # Find what's already in row 0
            for cc in range(9):
                if invalid_grid[0][cc] is not None:
                    invalid_grid[0][c] = invalid_grid[0][cc]  # Duplicate!
                    break
            break

    invalid_vec = encode_grid(invalid_grid)

    print(f"\nSimilarity to VIOLATION MANIFOLD:")
    print(f"  Valid grid:   {similarity(valid_vec, violation_manifold):.4f}")
    print(f"  Invalid grid: {similarity(invalid_vec, violation_manifold):.4f}")

    # The GOAL: valid should be LOW (far from violations)
    # Invalid should be HIGH (close to violations)


# =============================================================================
# EXPERIMENT 3: Negation to Push Away from Violations
# =============================================================================

def test_negation_repulsion():
    """
    Use negation to REMOVE violation patterns from state encoding.
    Does this push valid states further from invalid?
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Negation as Repulsion")
    print("=" * 70)

    store = create_store()

    digit_vecs = {d: store.vector_manager.get_vector(f"d{d}") for d in range(1, 10)}
    pos_vecs = {(r, c): store.vector_manager.get_vector(f"p{r}{c}")
                for r in range(9) for c in range(9)}

    def encode_grid(grid):
        components = []
        for r in range(9):
            for c in range(9):
                if grid[r][c] is not None:
                    components.append(store.bind(pos_vecs[(r, c)], digit_vecs[grid[r][c]]))
        if components:
            return store.bundle(components)
        return np.zeros(store.dimensions, dtype=np.int8)

    # Encode a violation pattern
    def encode_violation(r, c1, c2, d):
        v1 = store.bind(pos_vecs[(r, c1)], digit_vecs[d])
        v2 = store.bind(pos_vecs[(r, c2)], digit_vecs[d])
        return store.bundle([v1, v2])

    # Create grid with and without violation
    base_grid = [[cell for cell in row] for row in PUZZLE_9x9_HARD]

    # Find empty cells in row 0
    empty_in_row0 = [c for c in range(9) if base_grid[0][c] is None]

    if len(empty_in_row0) >= 2:
        c1, c2 = empty_in_row0[0], empty_in_row0[1]

        # Valid: put different digits
        valid_grid = [[cell for cell in row] for row in base_grid]
        opts = list(get_available_digits_9x9(valid_grid, 0, c1))
        if len(opts) >= 2:
            valid_grid[0][c1] = opts[0]
            valid_grid[0][c2] = opts[1]

        # Invalid: put same digit (violation)
        invalid_grid = [[cell for cell in row] for row in base_grid]
        invalid_grid[0][c1] = opts[0]
        invalid_grid[0][c2] = opts[0]  # SAME - violation!

        valid_vec = encode_grid(valid_grid)
        invalid_vec = encode_grid(invalid_grid)

        # The violation pattern
        violation_pattern = encode_violation(0, c1, c2, opts[0])

        print(f"\nPlacing digits at (0,{c1}) and (0,{c2})")
        print(f"  Valid: {opts[0]} and {opts[1]}")
        print(f"  Invalid: {opts[0]} and {opts[0]} (same - violation!)")

        print(f"\nSimilarity to violation pattern:")
        print(f"  Valid grid:   {similarity(valid_vec, violation_pattern):.4f}")
        print(f"  Invalid grid: {similarity(invalid_vec, violation_pattern):.4f}")

        # Now NEGATE the violation from both
        valid_negated = store.negate(valid_vec, violation_pattern)
        invalid_negated = store.negate(invalid_vec, violation_pattern)

        print(f"\nAfter NEGATING violation pattern:")
        print(f"  sim(valid_negated, violation):   {similarity(valid_negated, violation_pattern):.4f}")
        print(f"  sim(invalid_negated, violation): {similarity(invalid_negated, violation_pattern):.4f}")

        # Check if this improves separation
        diff_before = similarity(invalid_vec, violation_pattern) - similarity(valid_vec, violation_pattern)
        diff_after = similarity(invalid_negated, violation_pattern) - similarity(valid_negated, violation_pattern)

        print(f"\nSeparation improvement:")
        print(f"  Before negation: {diff_before:.4f}")
        print(f"  After negation:  {diff_after:.4f}")


# =============================================================================
# EXPERIMENT 4: Valid Solution Database as Reference Frame
# =============================================================================

def test_solution_reference_frame():
    """
    Pre-compute valid complete solutions.
    Use them as reference points in hyperspace.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Solution Database as Reference Frame")
    print("=" * 70)

    store = create_store()

    digit_vecs = {d: store.vector_manager.get_vector(f"d{d}") for d in range(1, 10)}

    # For this experiment, encode solutions as DIGIT SETS per constraint
    # (using our breakthrough template matching approach)

    def encode_solution_constraints(grid):
        """Encode a complete solution by its constraint structure."""
        constraint_vecs = []

        # Each row as a digit bundle
        for r in range(9):
            row_digits = [grid[r][c] for c in range(9)]
            row_vec = store.bundle([digit_vecs[d] for d in row_digits])
            constraint_vecs.append(row_vec)

        # Each column
        for c in range(9):
            col_digits = [grid[r][c] for r in range(9)]
            col_vec = store.bundle([digit_vecs[d] for d in col_digits])
            constraint_vecs.append(col_vec)

        # Each block
        for b in range(9):
            br, bc = (b // 3) * 3, (b % 3) * 3
            block_digits = [grid[br+i][bc+j] for i in range(3) for j in range(3)]
            block_vec = store.bundle([digit_vecs[d] for d in block_digits])
            constraint_vecs.append(block_vec)

        return store.bundle(constraint_vecs)

    # Generate some valid complete solutions
    # (we'll use the actual solution to our puzzle)
    def solve_to_completion(puzzle):
        grid = [[cell for cell in row] for row in puzzle]

        def solve(g):
            # Propagate
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
                        opts = list(get_available_digits_9x9(g, r, c))
                        for d in opts:
                            test = [[cell for cell in row] for row in g]
                            test[r][c] = d
                            result = solve(test)
                            if result:
                                return result
                        return None
            return None

        return solve(grid)

    solution = solve_to_completion(PUZZLE_9x9_HARD)

    if solution and validate_9x9(solution):
        print("\nFound valid solution")
        solution_vec = encode_solution_constraints(solution)
        print(f"Solution constraint vector norm: {np.linalg.norm(solution_vec):.1f}")

        # Encode the complete template (what all solutions share)
        complete_template = store.bundle([digit_vecs[d] for d in range(1, 10)])

        # All 27 constraints of a valid solution should match the template
        print(f"\nSimilarity of solution constraints to complete template: {similarity(solution_vec, complete_template):.4f}")

        # Now encode a WRONG solution (swap two digits)
        wrong = [[cell for cell in row] for row in solution]
        # Swap two digits in row 0
        wrong[0][0], wrong[0][1] = wrong[0][1], wrong[0][0]

        wrong_vec = encode_solution_constraints(wrong)

        print(f"\nSimilarity to solution database:")
        print(f"  Correct solution: {similarity(solution_vec, solution_vec):.4f}")
        print(f"  Wrong solution:   {similarity(wrong_vec, solution_vec):.4f}")


# =============================================================================
# EXPERIMENT 5: The Full Vision - Hyperspace Manifold
# =============================================================================

def test_hyperspace_manifold():
    """
    Implement the user's full vision:
    1. Define "valid region" by bundling valid patterns
    2. Define "violation region" by bundling violations
    3. Score candidates by: distance_to_valid - distance_to_violations
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: Hyperspace Manifold Scoring")
    print("=" * 70)

    store = create_store()

    digit_vecs = {d: store.vector_manager.get_vector(f"d{d}") for d in range(1, 10)}

    # VALID MANIFOLD: bundle of all valid constraint patterns
    # (For a row/col/block, valid = has all 9 unique digits)
    complete_template = store.bundle([digit_vecs[d] for d in range(1, 10)])

    # VIOLATION MANIFOLD: bundle of violation patterns
    # (For a constraint, violation = missing a digit or duplicate)
    violation_templates = []
    for missing in range(1, 10):
        # Pattern with one digit missing
        incomplete = store.bundle([digit_vecs[d] for d in range(1, 10) if d != missing])
        violation_templates.append(incomplete)
    violation_manifold = store.bundle(violation_templates)

    print(f"Valid template norm: {np.linalg.norm(complete_template):.1f}")
    print(f"Violation manifold norm: {np.linalg.norm(violation_manifold):.1f}")
    print(f"Similarity(valid, violation): {similarity(complete_template, violation_manifold):.4f}")

    # Test scoring function
    def manifold_score(digit_set):
        """Score a digit set by manifold distances."""
        if not digit_set:
            return 0.0
        vec = store.bundle([digit_vecs[d] for d in digit_set])

        dist_to_valid = similarity(vec, complete_template)
        dist_to_violation = similarity(vec, violation_manifold)

        # We want HIGH valid, LOW violation
        return dist_to_valid - dist_to_violation

    print("\nManifold scores for different digit sets:")
    test_sets = [
        {1, 2, 3, 4, 5, 6, 7, 8, 9},  # Complete
        {1, 2, 3, 4, 5, 6, 7, 8},     # Missing 9
        {1, 2, 3, 4, 5},              # Missing 4
        {1, 2, 3},                     # Missing 6
        {1, 1, 2, 3, 4, 5, 6, 7, 8},  # Duplicate 1 (but sets remove dups)
    ]

    for s in test_sets:
        score = manifold_score(s)
        print(f"  {sorted(s)}: score = {score:.4f}")

    # Now test: scoring choices
    print("\nScoring choices at decision point:")
    current = {1, 3, 5, 7}

    for digit in range(1, 10):
        if digit not in current:
            new_set = current | {digit}
            score = manifold_score(new_set)
            print(f"  Add {digit}: {sorted(new_set)} → score = {score:.4f}")


# =============================================================================
# EXPERIMENT 6: Solver Using Manifold Distance
# =============================================================================

def test_manifold_solver():
    """
    Full solver using manifold distance scoring.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 6: Manifold Distance Solver")
    print("=" * 70)

    store = create_store()

    digit_vecs = {d: store.vector_manager.get_vector(f"d{d}") for d in range(1, 10)}

    # Manifolds
    complete_template = store.bundle([digit_vecs[d] for d in range(1, 10)])

    # Violation manifold: incomplete sets
    violation_templates = []
    for missing in range(1, 10):
        incomplete = store.bundle([digit_vecs[d] for d in range(1, 10) if d != missing])
        violation_templates.append(incomplete)
    violation_manifold = store.bundle(violation_templates)

    class ManifoldSolver:
        def __init__(self):
            self.backtracks = 0

        def score_choice(self, grid, r, c, digit):
            """Score by manifold distances."""
            total = 0.0

            # Row
            row_digits = {grid[r][cc] for cc in range(9) if grid[r][cc] is not None}
            row_digits.add(digit)
            row_vec = store.bundle([digit_vecs[d] for d in row_digits])
            total += similarity(row_vec, complete_template) - similarity(row_vec, violation_manifold)

            # Column
            col_digits = {grid[rr][c] for rr in range(9) if grid[rr][c] is not None}
            col_digits.add(digit)
            col_vec = store.bundle([digit_vecs[d] for d in col_digits])
            total += similarity(col_vec, complete_template) - similarity(col_vec, violation_manifold)

            # Block
            br, bc = (r // 3) * 3, (c // 3) * 3
            block_digits = {grid[rr][cc] for rr in range(br, br+3) for cc in range(bc, bc+3)
                            if grid[rr][cc] is not None}
            block_digits.add(digit)
            block_vec = store.bundle([digit_vecs[d] for d in block_digits])
            total += similarity(block_vec, complete_template) - similarity(block_vec, violation_manifold)

            return total

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
                            opts = list(get_available_digits_9x9(grid, r, c))
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
                        opts = list(get_available_digits_9x9(grid, r, c))
                        if len(opts) < best_count:
                            best_count = len(opts)
                            best = (r, c, opts)

            if best is None:
                return None

            r, c, options = best

            # Score and order by manifold distance
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

    solver = ManifoldSolver()
    start = time.time()
    result = solver.solve(PUZZLE_9x9_HARD)
    elapsed = time.time() - start

    print(f"\nManifold solver:")
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
    test_valid_state_clustering()
    test_violation_manifold()
    test_negation_repulsion()
    test_solution_reference_frame()
    test_hyperspace_manifold()
    bt = test_manifold_solver()

    print("\n" + "=" * 70)
    print("HYPERSPACE GEOMETRY SUMMARY")
    print("=" * 70)
    print(f"""
THE USER'S VISION:
1. Valid solutions cluster in hyperspace
2. Invalid solutions are distant
3. Negation pushes invalids further
4. Database defines the valid region

WHAT WE FOUND:

Experiment 1 (Clustering):
- Do valid states cluster? Need to check results above

Experiment 2 (Violation Manifold):
- Invalid states SHOULD be closer to violation patterns
- Valid states SHOULD be far from violations

Experiment 3 (Negation Repulsion):
- Negation CAN push invalid patterns further from violation template

Experiment 4 (Solution Reference):
- Solutions share constraint structure (all rows/cols/blocks complete)

Experiment 5 (Manifold Scoring):
- Score = distance_to_valid - distance_to_violations
- More complete sets should score higher

Experiment 6 (Solver):
- Manifold solver backtracks: {bt}
- Template matching baseline: 52

KEY INSIGHT:
The "valid manifold" IS the complete template.
The "violation manifold" is incomplete patterns.
Scoring by difference = what template matching already does!
""")


if __name__ == "__main__":
    main()
