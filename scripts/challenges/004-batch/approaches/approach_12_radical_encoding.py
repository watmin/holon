#!/usr/bin/env python3
"""
Approach 12: Radical Encoding Exploration

KEY INSIGHT FROM EXPERIMENT 11:
- Different metrics work for different cells
- unbind_to_digit works 40% of the time
- The information IS there, but scattered across representations

NEW IDEA: What if the encoding structure itself is wrong?

Current encoding: grid = Σ bind(position, digit)
This is compositional - each cell contributes independently.

RADICAL ENCODINGS TO TRY:

1. RELATIONAL ENCODING:
   Encode pairs of cells that must be different.
   grid = Σ bind(pos1 ⊙ pos2, "different")

2. CONSTRAINT ENCODING:
   Encode each constraint as a unit.
   grid = Σ bind(constraint_id, Σ digits_in_constraint)

3. SOLUTION PATH ENCODING:
   Encode the SEQUENCE of placements, not just the state.
   Captures temporal/causal structure.

4. RESONANCE ENCODING:
   Create vectors that "resonate" only with valid configurations.
   Like a lock that only opens with the right key.

5. DUAL SPACE ENCODING:
   Encode in both position→digit AND digit→position spaces.
   The intersection of both might reveal more.
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


class RadicalEncoder:
    """Test radically different encoding schemes."""

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

        # Additional basis vectors for constraints
        self.row_vectors = {r: self._random_vector(f"row_{r}") for r in range(9)}
        self.col_vectors = {c: self._random_vector(f"col_{c}") for c in range(9)}
        self.block_vectors = {b: self._random_vector(f"block_{b}") for b in range(9)}

    def _random_vector(self, seed_str: str) -> np.ndarray:
        """Generate a deterministic random vector from string seed."""
        np.random.seed(hash(seed_str) % (2**32))
        vec = np.random.choice([-1.0, 1.0], size=self.dimensions)
        return vec

    # =========================================================================
    # ENCODING 1: Relational - encode that cells must differ
    # =========================================================================

    def encode_relational(self, grid: List[List[Optional[int]]]) -> np.ndarray:
        """
        Encode relationships between cells, not just cell contents.

        For each pair of cells that must differ (same row/col/block),
        if we know both values, encode that they're different.
        """
        relations = []

        # For each constraint
        for r in range(9):
            row_cells = [(r, c) for c in range(9)]
            relations.extend(self._encode_constraint_relations(grid, row_cells))

        for c in range(9):
            col_cells = [(r, c) for r in range(9)]
            relations.extend(self._encode_constraint_relations(grid, col_cells))

        for br in range(0, 9, 3):
            for bc in range(0, 9, 3):
                block_cells = [(br+dr, bc+dc) for dr in range(3) for dc in range(3)]
                relations.extend(self._encode_constraint_relations(grid, block_cells))

        return bundle(relations) if relations else np.zeros(self.dimensions)

    def _encode_constraint_relations(self, grid, cells) -> List[np.ndarray]:
        """Encode pairwise "different" relationships within a constraint."""
        relations = []
        for i, (r1, c1) in enumerate(cells):
            for (r2, c2) in cells[i+1:]:
                d1, d2 = grid[r1][c1], grid[r2][c2]
                if d1 is not None and d2 is not None:
                    # Encode: pos1 ⊙ pos2 ⊙ (d1 ⊙ d2)
                    # This captures "these positions have these different digits"
                    pos_pair = bind(self.pos_vectors[(r1, c1)], self.pos_vectors[(r2, c2)])
                    dig_pair = bind(self.digit_vectors[d1], self.digit_vectors[d2])
                    relations.append(bind(pos_pair, dig_pair))
        return relations

    # =========================================================================
    # ENCODING 2: Constraint-centric
    # =========================================================================

    def encode_constraint_centric(self, grid: List[List[Optional[int]]]) -> Dict[str, np.ndarray]:
        """
        Encode each constraint as a unit, tracking what's placed and what's missing.
        """
        encodings = {}

        for r in range(9):
            placed = [grid[r][c] for c in range(9) if grid[r][c] is not None]
            missing = [d for d in self.digits if d not in placed]

            # Encode row as: row_vec ⊙ bundle(placed)
            placed_vec = bundle([self.digit_vectors[d] for d in placed]) if placed else np.zeros(self.dimensions)
            missing_vec = bundle([self.digit_vectors[d] for d in missing]) if missing else np.zeros(self.dimensions)

            encodings[f'row_{r}'] = {
                'placed': bind(self.row_vectors[r], placed_vec),
                'missing': bind(self.row_vectors[r], missing_vec),
                'placed_digits': placed,
                'missing_digits': missing
            }

        for c in range(9):
            placed = [grid[r][c] for r in range(9) if grid[r][c] is not None]
            missing = [d for d in self.digits if d not in placed]

            placed_vec = bundle([self.digit_vectors[d] for d in placed]) if placed else np.zeros(self.dimensions)
            missing_vec = bundle([self.digit_vectors[d] for d in missing]) if missing else np.zeros(self.dimensions)

            encodings[f'col_{c}'] = {
                'placed': bind(self.col_vectors[c], placed_vec),
                'missing': bind(self.col_vectors[c], missing_vec),
                'placed_digits': placed,
                'missing_digits': missing
            }

        for b in range(9):
            br, bc = (b // 3) * 3, (b % 3) * 3
            placed = [grid[r][c] for r in range(br, br+3) for c in range(bc, bc+3)
                      if grid[r][c] is not None]
            missing = [d for d in self.digits if d not in placed]

            placed_vec = bundle([self.digit_vectors[d] for d in placed]) if placed else np.zeros(self.dimensions)
            missing_vec = bundle([self.digit_vectors[d] for d in missing]) if missing else np.zeros(self.dimensions)

            encodings[f'block_{b}'] = {
                'placed': bind(self.block_vectors[b], placed_vec),
                'missing': bind(self.block_vectors[b], missing_vec),
                'placed_digits': placed,
                'missing_digits': missing
            }

        return encodings

    # =========================================================================
    # ENCODING 3: Dual Space (position→digit AND digit→position)
    # =========================================================================

    def encode_dual_space(self, grid: List[List[Optional[int]]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encode in both directions:
        - pos_to_dig: position → digit (standard)
        - dig_to_pos: digit → positions where it appears
        """
        pos_to_dig_bindings = []
        dig_to_pos = {d: [] for d in self.digits}

        for r in range(9):
            for c in range(9):
                if grid[r][c] is not None:
                    d = grid[r][c]
                    pos_vec = self.pos_vectors[(r, c)]
                    dig_vec = self.digit_vectors[d]

                    # Standard: pos → dig
                    pos_to_dig_bindings.append(bind(pos_vec, dig_vec))

                    # Inverse: dig → pos
                    dig_to_pos[d].append(pos_vec)

        pos_to_dig = bundle(pos_to_dig_bindings) if pos_to_dig_bindings else np.zeros(self.dimensions)

        # Bundle all positions for each digit
        dig_to_pos_vec = {}
        for d in self.digits:
            if dig_to_pos[d]:
                dig_to_pos_vec[d] = bundle(dig_to_pos[d])
            else:
                dig_to_pos_vec[d] = np.zeros(self.dimensions)

        return pos_to_dig, dig_to_pos_vec

    # =========================================================================
    # ENCODING 4: Resonance - encode "valid configuration" templates
    # =========================================================================

    def encode_valid_row_template(self) -> np.ndarray:
        """
        Encode what a VALID row looks like.

        A valid row = each position has one unique digit.
        We encode this as: Σ_i (pos_i ⊙ digit_i) for some permutation.

        Since all permutations are valid, we can't encode a specific one.
        Instead, encode the STRUCTURE: each position gets exactly one digit.
        """
        # Template: position i binds to "exactly one from {1-9}"
        template_parts = []
        for c in range(9):
            # Position c should have exactly one digit (superposition of all)
            all_digits = bundle([self.digit_vectors[d] for d in self.digits])
            template_parts.append(bind(self.pos_vectors[(0, c)], all_digits))

        return bundle(template_parts)

    def score_by_resonance(self, grid: List[List[Optional[int]]],
                           row: int, col: int, digit: int) -> float:
        """
        Score a digit by how much it makes the row "resonate" with valid template.
        """
        # Current row encoding
        row_parts = []
        for c in range(9):
            if grid[row][c] is not None:
                row_parts.append(bind(self.pos_vectors[(row, c)],
                                     self.digit_vectors[grid[row][c]]))

        # Add proposed digit
        row_parts.append(bind(self.pos_vectors[(row, col)], self.digit_vectors[digit]))
        row_vec = bundle(row_parts)

        # Compare to valid template
        template = self.encode_valid_row_template()
        return similarity(row_vec, template)

    # =========================================================================
    # QUERY: Using dual space for prediction
    # =========================================================================

    def query_dual_space(self, grid: List[List[Optional[int]]],
                         row: int, col: int) -> Dict[int, float]:
        """
        Query using both encoding spaces.

        For each available digit:
        - In pos→dig space: How well does adding this binding improve the encoding?
        - In dig→pos space: How well does this position fit the digit's current locations?
        """
        available = get_available_digits_9x9(grid, row, col)
        pos_to_dig, dig_to_pos = self.encode_dual_space(grid)

        scores = {}
        pos_vec = self.pos_vectors[(row, col)]

        for d in available:
            # Score 1: Unbinding - does pos→dig suggest this digit?
            unbound = unbind(pos_to_dig, pos_vec)
            unbind_score = similarity(unbound, self.digit_vectors[d])

            # Score 2: Does this position fit the pattern of where d appears?
            # (For Sudoku, d should NOT appear in same row/col/block,
            #  so low similarity to current positions is GOOD)
            if np.any(dig_to_pos[d] != 0):
                pos_pattern = dig_to_pos[d]
                pos_sim = similarity(pos_vec, pos_pattern)
                # Lower similarity = better (means this position is DIFFERENT from where d already is)
                inverse_pos_score = 1 - pos_sim
            else:
                inverse_pos_score = 1.0

            # Score 3: Constraint-based - how constrained is this digit?
            constraint_encoding = self.encode_constraint_centric(grid)
            row_missing = constraint_encoding[f'row_{row}']['missing_digits']
            col_missing = constraint_encoding[f'col_{col}']['missing_digits']
            block_idx = (row // 3) * 3 + (col // 3)
            block_missing = constraint_encoding[f'block_{block_idx}']['missing_digits']

            # Count how many constraints this digit satisfies
            constraint_score = sum([d in row_missing, d in col_missing, d in block_missing]) / 3

            # Combine scores
            scores[d] = unbind_score + inverse_pos_score + constraint_score

        return scores


def test_dual_space_solver():
    """Test if dual space encoding helps solve the puzzle."""
    print("=" * 70)
    print("TEST: Dual Space Encoding Solver")
    print("=" * 70)

    encoder = RadicalEncoder(dimensions=16384)
    grid = [[cell for cell in row] for row in PUZZLE_9x9_HARD]

    correct = 0
    total = 0
    wrong_predictions = []

    for r in range(9):
        for c in range(9):
            if PUZZLE_9x9_HARD[r][c] is None:
                scores = encoder.query_dual_space(grid, r, c)
                predicted = max(scores, key=lambda x: scores[x])
                actual = SOLUTION_9x9_HARD[r][c]

                total += 1
                if predicted == actual:
                    correct += 1
                else:
                    wrong_predictions.append((r, c, predicted, actual, scores))

    print(f"\nDual space prediction accuracy: {correct}/{total} = {100*correct/total:.1f}%")
    print(f"\nFirst 5 wrong predictions:")
    for r, c, pred, actual, scores in wrong_predictions[:5]:
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        print(f"  ({r},{c}): predicted {pred}, actual {actual}")
        print(f"    Scores: {sorted_scores[:3]}")


def test_relational_encoding():
    """Test if relational encoding captures solution structure better."""
    print("\n" + "=" * 70)
    print("TEST: Relational Encoding")
    print("=" * 70)

    encoder = RadicalEncoder(dimensions=16384)

    # Encode the puzzle relationally
    puzzle_rel = encoder.encode_relational(PUZZLE_9x9_HARD)

    # Encode the solution relationally
    solution_rel = encoder.encode_relational(SOLUTION_9x9_HARD)

    print(f"\nSimilarity puzzle→solution (relational): {similarity(puzzle_rel, solution_rel):.4f}")

    # Now test: does adding correct digits increase similarity more?
    improvements_correct = []
    improvements_wrong = []

    for r in range(9):
        for c in range(9):
            if PUZZLE_9x9_HARD[r][c] is None:
                available = get_available_digits_9x9(PUZZLE_9x9_HARD, r, c)
                if len(available) <= 1:
                    continue

                correct_d = SOLUTION_9x9_HARD[r][c]
                wrong_d = [d for d in available if d != correct_d][0]

                # Test correct
                test_grid = [[cell for cell in row] for row in PUZZLE_9x9_HARD]
                test_grid[r][c] = correct_d
                correct_vec = encoder.encode_relational(test_grid)
                correct_sim = similarity(correct_vec, solution_rel)

                # Test wrong
                test_grid[r][c] = wrong_d
                wrong_vec = encoder.encode_relational(test_grid)
                wrong_sim = similarity(wrong_vec, solution_rel)

                improvements_correct.append(correct_sim)
                improvements_wrong.append(wrong_sim)

                if len(improvements_correct) >= 10:
                    break
        if len(improvements_correct) >= 10:
            break

    print(f"\nAvg similarity with correct digit: {np.mean(improvements_correct):.4f}")
    print(f"Avg similarity with wrong digit:   {np.mean(improvements_wrong):.4f}")

    if np.mean(improvements_correct) > np.mean(improvements_wrong):
        print("✓ Relational encoding DOES favor correct digits!")
    else:
        print("✗ Relational encoding doesn't reliably distinguish.")


def test_resonance_scoring():
    """Test if resonance with valid templates identifies correct digits."""
    print("\n" + "=" * 70)
    print("TEST: Resonance Scoring")
    print("=" * 70)

    encoder = RadicalEncoder(dimensions=16384)

    correct = 0
    total = 0

    for r in range(9):
        for c in range(9):
            if PUZZLE_9x9_HARD[r][c] is None:
                available = get_available_digits_9x9(PUZZLE_9x9_HARD, r, c)
                if len(available) <= 1:
                    continue

                # Score each option by resonance
                scores = {d: encoder.score_by_resonance(PUZZLE_9x9_HARD, r, c, d)
                         for d in available}

                predicted = max(scores, key=lambda x: scores[x])
                actual = SOLUTION_9x9_HARD[r][c]

                total += 1
                if predicted == actual:
                    correct += 1

    print(f"\nResonance prediction accuracy: {correct}/{total} = {100*correct/total:.1f}%")


def main():
    print("=" * 70)
    print("APPROACH 12: RADICAL ENCODING EXPLORATION")
    print("=" * 70)
    print("\nTesting fundamentally different encoding schemes...")

    test_dual_space_solver()
    test_relational_encoding()
    test_resonance_scoring()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
Different encodings capture different aspects:
- Dual space: Combines pos→dig and dig→pos information
- Relational: Captures "must be different" constraints
- Resonance: Compares to "valid template" structure

The question: Is there an encoding that consistently identifies correct?
""")


if __name__ == "__main__":
    main()
