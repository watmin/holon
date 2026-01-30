#!/usr/bin/env python3
"""
Approach 11: Can We PROVE There's No Geometric Solution?

QUESTION:
Is there ANY encoding/orientation in hyperspace where the correct
digit is geometrically distinguishable from incorrect ones?

THE CRUX:
At decision points, we have multiple locally-valid options.
Only ONE leads to the solution. Can ANY geometric measure distinguish them?

EXPERIMENT 1: Information Content
- At each decision point, measure ALL geometric properties we can think of
- See if the correct choice is EVER distinguishable

EXPERIMENT 2: Constraint System
- Encode constraints as a system of vector equations
- See if the solution can be found via linear algebra in hyperspace

EXPERIMENT 3: Holographic Completion
- Treat the puzzle as a "partial hologram"
- See if the solution can be "reconstructed"
"""

import sys
sys.path.insert(0, "/home/watmin/work/holon")

from typing import List, Optional, Set, Tuple, Dict
import numpy as np
from collections import defaultdict

from common import (
    create_client,
    VectorCache,
    bind,
    unbind,
    bundle,
    similarity,
    effective_dimensionality,
    Timer,
    print_grid_9x9,
    validate_9x9,
    count_empty,
    get_available_digits_9x9,
    PUZZLE_9x9_HARD,
)


# The known solution (from our solver)
SOLUTION_9x9_HARD = [
    [5, 8, 1, 6, 7, 2, 4, 3, 9],
    [7, 9, 2, 8, 4, 3, 6, 5, 1],
    [3, 6, 4, 5, 9, 1, 7, 8, 2],
    [4, 3, 8, 9, 5, 7, 2, 1, 6],
    [2, 5, 6, 1, 8, 4, 9, 7, 3],
    [1, 7, 9, 3, 2, 6, 8, 4, 5],
    [8, 4, 5, 2, 1, 9, 3, 6, 7],
    [9, 1, 3, 7, 6, 8, 5, 2, 4],
    [6, 2, 7, 4, 3, 5, 1, 9, 8],
]


class GeometricDistinguisher:
    """
    Test if we can geometrically distinguish correct from incorrect choices.
    """

    def __init__(self, dimensions: int = 16384):
        self.dimensions = dimensions
        self.client = create_client(dimensions=dimensions)
        self.cache = VectorCache(self.client)

        self.digits = list(range(1, 10))
        self.digit_vectors = {d: self.cache.get_digit_vector(d) for d in self.digits}
        self.pos_vectors = {}
        for r in range(9):
            for c in range(9):
                self.pos_vectors[(r, c)] = self.cache.get_position_vector(r, c)

    def encode_grid(self, grid: List[List[Optional[int]]]) -> np.ndarray:
        """Encode entire grid state as single vector."""
        bindings = []
        for r in range(9):
            for c in range(9):
                if grid[r][c] is not None:
                    pos_vec = self.pos_vectors[(r, c)]
                    dig_vec = self.digit_vectors[grid[r][c]]
                    bindings.append(bind(pos_vec, dig_vec))
        return bundle(bindings) if bindings else np.zeros(self.dimensions)

    def encode_constraint(self, constraint_type: str, idx: int,
                          grid: List[List[Optional[int]]]) -> np.ndarray:
        """Encode a constraint's current state."""
        if constraint_type == 'row':
            cells = [(idx, c) for c in range(9)]
        elif constraint_type == 'col':
            cells = [(r, idx) for r in range(9)]
        else:  # block
            br, bc = (idx // 3) * 3, (idx % 3) * 3
            cells = [(br + dr, bc + dc) for dr in range(3) for dc in range(3)]

        bindings = []
        for r, c in cells:
            if grid[r][c] is not None:
                bindings.append(bind(self.pos_vectors[(r, c)],
                                    self.digit_vectors[grid[r][c]]))
        return bundle(bindings) if bindings else np.zeros(self.dimensions)

    def encode_ideal_constraint(self, constraint_type: str, idx: int) -> np.ndarray:
        """Encode what a complete constraint should look like."""
        if constraint_type == 'row':
            cells = [(idx, c) for c in range(9)]
        elif constraint_type == 'col':
            cells = [(r, idx) for r in range(9)]
        else:
            br, bc = (idx // 3) * 3, (idx % 3) * 3
            cells = [(br + dr, bc + dc) for dr in range(3) for dc in range(3)]

        # Ideal = each position bound to superposition of all digits
        bindings = []
        all_digits = bundle([self.digit_vectors[d] for d in self.digits])
        for r, c in cells:
            bindings.append(bind(self.pos_vectors[(r, c)], all_digits))
        return bundle(bindings)

    def compute_all_metrics(self, grid: List[List[Optional[int]]],
                            row: int, col: int, digit: int) -> Dict[str, float]:
        """
        Compute EVERY geometric metric we can think of for placing digit at (row, col).
        """
        metrics = {}

        # Make temporary placement
        test_grid = [[cell for cell in r] for r in grid]
        test_grid[row][col] = digit

        # 1. Grid encoding similarity to solution
        grid_vec = self.encode_grid(test_grid)
        solution_vec = self.encode_grid(SOLUTION_9x9_HARD)
        metrics['grid_to_solution'] = similarity(grid_vec, solution_vec)

        # 2. Row completion similarity
        row_vec = self.encode_constraint('row', row, test_grid)
        ideal_row = self.encode_ideal_constraint('row', row)
        metrics['row_completion'] = similarity(row_vec, ideal_row)

        # 3. Column completion
        col_vec = self.encode_constraint('col', col, test_grid)
        ideal_col = self.encode_ideal_constraint('col', col)
        metrics['col_completion'] = similarity(col_vec, ideal_col)

        # 4. Block completion
        block_idx = (row // 3) * 3 + (col // 3)
        block_vec = self.encode_constraint('block', block_idx, test_grid)
        ideal_block = self.encode_ideal_constraint('block', block_idx)
        metrics['block_completion'] = similarity(block_vec, ideal_block)

        # 5. Dimensionality of row
        row_digits = [test_grid[row][c] for c in range(9) if test_grid[row][c]]
        if row_digits:
            row_bundle = bundle([self.digit_vectors[d] for d in row_digits])
            digit_basis = [self.digit_vectors[d] for d in self.digits]
            metrics['row_dimensionality'] = effective_dimensionality(row_bundle, digit_basis)

        # 6. Digit similarity to position (unbind)
        pos_vec = self.pos_vectors[(row, col)]
        unbound = unbind(grid_vec, pos_vec)
        metrics['unbind_to_digit'] = similarity(unbound, self.digit_vectors[digit])

        # 7. Cross-constraint agreement
        # How much do row, col, block agree on this digit?
        row_available = set()
        col_available = set()
        block_available = set()

        for d in self.digits:
            # Check if d is in row
            if d not in [test_grid[row][c] for c in range(9) if c != col and test_grid[row][c]]:
                row_available.add(d)
            if d not in [test_grid[r][col] for r in range(9) if r != row and test_grid[r][col]]:
                col_available.add(d)
            br, bc = (row // 3) * 3, (col // 3) * 3
            block_vals = [test_grid[r][c] for r in range(br, br+3) for c in range(bc, bc+3)
                         if (r, c) != (row, col) and test_grid[r][c]]
            if d not in block_vals:
                block_available.add(d)

        # Count how many constraints this digit satisfies
        metrics['constraint_agreement'] = sum([
            digit in row_available,
            digit in col_available,
            digit in block_available
        ]) / 3.0

        return metrics

    def find_decision_points(self, puzzle: List[List[Optional[int]]],
                             solution: List[List[int]]) -> List[Tuple[int, int, Set[int], int]]:
        """
        Find cells where there are multiple valid options.
        Returns: [(row, col, available_set, correct_digit), ...]
        """
        decision_points = []

        for r in range(9):
            for c in range(9):
                if puzzle[r][c] is None:
                    available = get_available_digits_9x9(puzzle, r, c)
                    if len(available) > 1:
                        correct = solution[r][c]
                        decision_points.append((r, c, available, correct))

        return decision_points


def experiment_1_distinguish_correct():
    """
    EXPERIMENT 1: Can we distinguish correct from incorrect at decision points?
    """
    print("=" * 70)
    print("EXPERIMENT 1: Can ANY metric distinguish correct from incorrect?")
    print("=" * 70)

    dist = GeometricDistinguisher(dimensions=16384)

    # Find all decision points in the original puzzle
    decision_points = dist.find_decision_points(PUZZLE_9x9_HARD, SOLUTION_9x9_HARD)

    print(f"\nFound {len(decision_points)} decision points in the puzzle")
    print("\nFor each decision point, we measure all metrics and see if")
    print("the CORRECT digit scores highest on ANY metric.\n")

    metric_success = defaultdict(int)
    metric_total = defaultdict(int)
    total_points = 0
    any_metric_works = 0

    for r, c, available, correct in decision_points[:20]:  # First 20 for speed
        total_points += 1

        # Compute metrics for all available options
        option_metrics = {}
        for d in available:
            option_metrics[d] = dist.compute_all_metrics(PUZZLE_9x9_HARD, r, c, d)

        # For each metric, check if correct digit scores highest
        metrics_where_correct_wins = []

        for metric_name in option_metrics[correct].keys():
            scores = {d: option_metrics[d].get(metric_name, 0) for d in available}
            best = max(scores, key=lambda x: scores[x])

            metric_total[metric_name] += 1
            if best == correct:
                metric_success[metric_name] += 1
                metrics_where_correct_wins.append(metric_name)

        if metrics_where_correct_wins:
            any_metric_works += 1

        print(f"Cell ({r},{c}): options={sorted(available)}, correct={correct}")
        print(f"  Metrics where correct wins: {metrics_where_correct_wins or 'NONE'}")

    print("\n" + "=" * 70)
    print("SUMMARY: How often does each metric identify the correct digit?")
    print("=" * 70)

    for metric_name in sorted(metric_total.keys()):
        success = metric_success[metric_name]
        total = metric_total[metric_name]
        pct = 100 * success / total if total > 0 else 0
        print(f"  {metric_name:25s}: {success}/{total} = {pct:.1f}%")

    print(f"\n  Points where ANY metric works: {any_metric_works}/{total_points}")

    return any_metric_works / total_points if total_points > 0 else 0


def experiment_2_grid_similarity():
    """
    EXPERIMENT 2: Does the complete puzzle encoding point toward solution?
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Does partial grid point toward solution?")
    print("=" * 70)

    dist = GeometricDistinguisher(dimensions=16384)

    # Encode the partial puzzle
    puzzle_vec = dist.encode_grid(PUZZLE_9x9_HARD)

    # Encode the solution
    solution_vec = dist.encode_grid(SOLUTION_9x9_HARD)

    print(f"\nSimilarity of puzzle encoding to solution: {similarity(puzzle_vec, solution_vec):.4f}")

    # Now let's try: does adding CORRECT digits increase similarity more than wrong ones?
    print("\nTest: Does adding correct digits increase similarity to solution?")

    improvements_correct = []
    improvements_wrong = []

    for r, c, available, correct in dist.find_decision_points(PUZZLE_9x9_HARD, SOLUTION_9x9_HARD)[:10]:
        wrong_options = [d for d in available if d != correct]
        if not wrong_options:
            continue

        # Baseline
        baseline_sim = similarity(puzzle_vec, solution_vec)

        # Add correct digit
        test_grid = [[cell for cell in row] for row in PUZZLE_9x9_HARD]
        test_grid[r][c] = correct
        correct_vec = dist.encode_grid(test_grid)
        correct_sim = similarity(correct_vec, solution_vec)
        improvements_correct.append(correct_sim - baseline_sim)

        # Add wrong digit
        test_grid[r][c] = wrong_options[0]
        wrong_vec = dist.encode_grid(test_grid)
        wrong_sim = similarity(wrong_vec, solution_vec)
        improvements_wrong.append(wrong_sim - baseline_sim)

        print(f"  ({r},{c}): correct={correct} gives +{correct_sim - baseline_sim:.4f}, "
              f"wrong={wrong_options[0]} gives +{wrong_sim - baseline_sim:.4f}")

    print(f"\nAvg improvement with correct: {np.mean(improvements_correct):.4f}")
    print(f"Avg improvement with wrong:   {np.mean(improvements_wrong):.4f}")

    if np.mean(improvements_correct) > np.mean(improvements_wrong):
        print("\n✓ Correct digits DO increase similarity more on average!")
        print("  But the difference might be too small to reliably distinguish.")
    else:
        print("\n✗ Wrong digits sometimes increase similarity more!")


def experiment_3_holographic():
    """
    EXPERIMENT 3: Holographic completion - can we recover missing cells?
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Holographic Completion")
    print("=" * 70)
    print("\nIdea: If puzzle is a 'partial hologram', can we reconstruct missing parts?")

    dist = GeometricDistinguisher(dimensions=16384)

    # Encode the full solution
    solution_vec = dist.encode_grid(SOLUTION_9x9_HARD)

    print("\nTest: Can we recover digits by unbinding positions from solution vector?")

    correct_recoveries = 0
    total_tests = 0

    for r in range(9):
        for c in range(9):
            pos_vec = dist.pos_vectors[(r, c)]
            unbound = unbind(solution_vec, pos_vec)

            # Find which digit it's most similar to
            best_digit = None
            best_sim = -1
            for d in dist.digits:
                sim = similarity(unbound, dist.digit_vectors[d])
                if sim > best_sim:
                    best_sim = sim
                    best_digit = d

            correct = SOLUTION_9x9_HARD[r][c]
            if best_digit == correct:
                correct_recoveries += 1
            total_tests += 1

    print(f"\nFrom COMPLETE solution vector:")
    print(f"  Correctly recovered: {correct_recoveries}/{total_tests} = {100*correct_recoveries/total_tests:.1f}%")

    # Now try from partial puzzle
    print("\nFrom PARTIAL puzzle vector:")
    puzzle_vec = dist.encode_grid(PUZZLE_9x9_HARD)

    correct_recoveries = 0
    for r, c, available, correct in dist.find_decision_points(PUZZLE_9x9_HARD, SOLUTION_9x9_HARD):
        pos_vec = dist.pos_vectors[(r, c)]
        unbound = unbind(puzzle_vec, pos_vec)

        best_digit = max(dist.digits, key=lambda d: similarity(unbound, dist.digit_vectors[d]))

        if best_digit == correct:
            correct_recoveries += 1
        total_tests += 1

    empty_cells = count_empty(PUZZLE_9x9_HARD)
    print(f"  Correctly predicted empty cells: {correct_recoveries}/{empty_cells}")


def experiment_4_constraint_intersection():
    """
    EXPERIMENT 4: Constraint intersection in hyperspace
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Constraint Intersection")
    print("=" * 70)
    print("\nIdea: Each constraint defines a 'valid region'. Solution is intersection.")

    dist = GeometricDistinguisher(dimensions=16384)

    print("\nTest: For each empty cell, compute constraint agreement vector")
    print("and see if it points to correct digit.\n")

    correct_predictions = 0
    total_predictions = 0

    for r, c, available, correct in dist.find_decision_points(PUZZLE_9x9_HARD, SOLUTION_9x9_HARD)[:15]:
        # For each available digit, compute how well it "fits" all constraints
        constraint_scores = {}

        for d in available:
            # Bundle the "vote" from each constraint
            row_vote = dist.digit_vectors[d]  # Row says "d is available"
            col_vote = dist.digit_vectors[d]
            block_vote = dist.digit_vectors[d]

            # Weight by how constrained each constraint is
            row_used = sum(1 for cc in range(9) if PUZZLE_9x9_HARD[r][cc] is not None)
            col_used = sum(1 for rr in range(9) if PUZZLE_9x9_HARD[rr][c] is not None)
            br, bc = (r // 3) * 3, (c // 3) * 3
            block_used = sum(1 for rr in range(br, br+3) for cc in range(bc, bc+3)
                            if PUZZLE_9x9_HARD[rr][cc] is not None)

            # More constrained = stronger vote
            combined = bundle([
                row_vote * row_used,
                col_vote * col_used,
                block_vote * block_used
            ])

            constraint_scores[d] = np.linalg.norm(combined)

        predicted = max(constraint_scores, key=lambda x: constraint_scores[x])

        if predicted == correct:
            correct_predictions += 1
            status = "✓"
        else:
            status = "✗"

        total_predictions += 1
        print(f"  ({r},{c}): predicted={predicted}, correct={correct} {status}")

    print(f"\nCorrect predictions: {correct_predictions}/{total_predictions}")


def main():
    print("=" * 70)
    print("APPROACH 11: PROVING OR DISPROVING GEOMETRIC SOLUTION EXISTS")
    print("=" * 70)
    print("\nWe test whether ANY encoding/metric can distinguish correct digits.")
    print("If no metric works, we have evidence that geometry alone is insufficient.\n")

    # Run experiments
    exp1_result = experiment_1_distinguish_correct()
    experiment_2_grid_similarity()
    experiment_3_holographic()
    experiment_4_constraint_intersection()

    # Conclusion
    print("\n" + "=" * 70)
    print("CONCLUSIONS")
    print("=" * 70)

    if exp1_result < 0.5:
        print("""
The experiments suggest that NO SINGLE METRIC reliably distinguishes
correct from incorrect digits at decision points.

However, this doesn't PROVE impossibility - it just shows our current
encodings don't work. There might be an encoding we haven't tried.

To prove impossibility, we would need to show:
1. The information required to distinguish is not present in local geometry
2. OR that any encoding with this information would be exponentially large

The fundamental issue remains: multiple locally-valid choices that only
diverge later are geometrically indistinguishable at the choice point.
""")
    else:
        print("""
Surprisingly, some metrics DO identify correct digits more often than chance!
This suggests there IS geometric information we can exploit.

The question is: can we combine metrics or find better encodings?
""")


if __name__ == "__main__":
    main()
