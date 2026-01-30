#!/usr/bin/env python3
"""
Approach 5: Structural Entanglement

HYPOTHESIS:
Holon's recursive encoding creates ENTANGLED representations.
Binding preserves the relationship such that unbinding can recover it.
We can use this to query unknown values from encoded structures.

KEY INSIGHT:
If we encode: cell = bind(position, digit)
Then: unbind(cell, position) ≈ digit

Can we exploit this for solving?
1. Encode all known placements
2. Encode constraint "templates"
3. Query unknowns by unbinding

TEST QUESTIONS:
1. Does unbinding from bundled structure preserve individual bindings?
2. Can constraints inform what digit should be at a position?
3. Is there a "completion" operation that fills unknowns?
"""

import sys
sys.path.insert(0, "/home/watmin/work/holon")

from typing import List, Optional, Set, Tuple, Dict
import numpy as np

from common import (
    create_client,
    VectorCache,
    bind,
    unbind,
    bundle,
    similarity,
    ApproachResult,
    Timer,
    print_grid_4x4,
    print_grid_9x9,
    validate_4x4,
    validate_9x9,
    count_empty,
    get_available_digits_4x4,
    get_available_digits_9x9,
    PUZZLE_4x4_EASY,
    PUZZLE_4x4_SOLUTION,
    PUZZLE_9x9_EASY,
    PUZZLE_9x9_EASY_SOLUTION,
    PUZZLE_9x9_HARD,
)


class EntanglementSolver:
    """
    Sudoku solver exploiting entanglement properties.

    Core idea: Encode structures such that unbinding reveals solutions.
    """

    def __init__(self, size: int = 9, dimensions: int = 16384, verbose: bool = True):
        self.size = size
        self.block_size = 2 if size == 4 else 3
        self.digits = list(range(1, size + 1))
        self.verbose = verbose

        self.client = create_client(dimensions=dimensions)
        self.cache = VectorCache(self.client)

        # Pre-cache vectors
        self.digit_vectors = {d: self.cache.get_digit_vector(d) for d in self.digits}
        self.position_vectors = {}
        for r in range(size):
            for c in range(size):
                self.position_vectors[(r, c)] = self.cache.get_position_vector(r, c)

        self.notes: List[str] = []
        self.iterations = 0
        self.entanglement_queries = 0

    def log(self, msg: str):
        if self.verbose:
            print(msg)
        self.notes.append(msg)

    def get_available(self, grid, row, col) -> Set[int]:
        if self.size == 4:
            return get_available_digits_4x4(grid, row, col)
        return get_available_digits_9x9(grid, row, col)

    def encode_cell(self, row: int, col: int, digit: int) -> np.ndarray:
        """Encode a single cell placement as position ⊙ digit."""
        return bind(self.position_vectors[(row, col)], self.digit_vectors[digit])

    def encode_grid(self, grid: List[List[Optional[int]]]) -> np.ndarray:
        """Encode all known cells as bundled cell bindings."""
        cell_bindings = []
        for r in range(self.size):
            for c in range(self.size):
                if grid[r][c] is not None:
                    cell_bindings.append(self.encode_cell(r, c, grid[r][c]))

        if cell_bindings:
            return bundle(cell_bindings)
        return np.zeros_like(self.digit_vectors[1])

    def encode_ideal_row(self, row: int) -> np.ndarray:
        """
        Encode the ideal row constraint.

        An ideal row has ALL 9 digits at positions (row, 0) through (row, 8).
        We encode this as a bundle of all possible valid row completions...
        but that's 9! = 362880 combinations.

        Simpler: encode the STRUCTURE of a valid row (superposition of all digits).
        """
        # Bundle all position-digit bindings for this row
        # Each column should have some digit
        row_bindings = []
        for c in range(self.size):
            # Superposition: this position could have any digit
            pos = self.position_vectors[(row, c)]
            digit_super = bundle([self.digit_vectors[d] for d in self.digits])
            row_bindings.append(bind(pos, digit_super))

        return bundle(row_bindings)

    def encode_ideal_col(self, col: int) -> np.ndarray:
        """Encode the ideal column constraint."""
        col_bindings = []
        for r in range(self.size):
            pos = self.position_vectors[(r, col)]
            digit_super = bundle([self.digit_vectors[d] for d in self.digits])
            col_bindings.append(bind(pos, digit_super))
        return bundle(col_bindings)

    def encode_ideal_block(self, row: int, col: int) -> np.ndarray:
        """Encode the ideal block constraint for the block containing (row, col)."""
        br = (row // self.block_size) * self.block_size
        bc = (col // self.block_size) * self.block_size

        block_bindings = []
        for r in range(br, br + self.block_size):
            for c in range(bc, bc + self.block_size):
                pos = self.position_vectors[(r, c)]
                digit_super = bundle([self.digit_vectors[d] for d in self.digits])
                block_bindings.append(bind(pos, digit_super))
        return bundle(block_bindings)

    def query_digit_at_position(self, grid_vec: np.ndarray,
                                 row: int, col: int) -> Tuple[int, float]:
        """
        Query what digit is at a position by unbinding.

        Unbind the position from the grid vector.
        The result should be close to the digit vector if it's known.
        """
        pos_vec = self.position_vectors[(row, col)]
        recovered = unbind(grid_vec, pos_vec)

        # Find best matching digit
        best_digit = None
        best_sim = -1
        for d in self.digits:
            sim = similarity(recovered, self.digit_vectors[d])
            if sim > best_sim:
                best_sim = sim
                best_digit = d

        self.entanglement_queries += 1
        return best_digit, best_sim

    def query_row_completion(self, grid: List[List[Optional[int]]],
                             row: int, col: int) -> Dict[int, float]:
        """
        Use row structure to inform what digit should go at (row, col).

        Idea: Encode the partial row, compare to ideal row,
        find what digit would make it more similar to ideal.
        """
        ideal_row = self.encode_ideal_row(row)

        # Current row encoding (known cells)
        known_bindings = []
        for c in range(self.size):
            if grid[row][c] is not None:
                known_bindings.append(self.encode_cell(row, c, grid[row][c]))

        if known_bindings:
            current_row = bundle(known_bindings)
        else:
            current_row = np.zeros_like(ideal_row)

        # Score each possible digit for this cell
        scores = {}
        available = self.get_available(grid, row, col)

        for d in available:
            # What if we add this digit?
            test_binding = self.encode_cell(row, col, d)
            test_row = bundle([current_row, test_binding])

            # How similar to ideal?
            sim = similarity(test_row, ideal_row)
            scores[d] = sim

        return scores

    def query_full_completion(self, grid: List[List[Optional[int]]],
                              row: int, col: int,
                              strategy: str = "min") -> Dict[int, float]:
        """
        Use ALL THREE constraint types (row, column, block) to score digits.

        Strategies:
        - "avg": Average of three (original, dilutes signal)
        - "min": Minimum of three (all must agree)
        - "max": Maximum of three (any strong signal wins)
        - "product": Product of three (amplifies agreement)
        """
        available = self.get_available(grid, row, col)
        if not available:
            return {}

        # Get ideals
        ideal_row = self.encode_ideal_row(row)
        ideal_col = self.encode_ideal_col(col)
        ideal_block = self.encode_ideal_block(row, col)

        # Encode current state for each constraint
        # Row
        row_bindings = []
        for c in range(self.size):
            if grid[row][c] is not None:
                row_bindings.append(self.encode_cell(row, c, grid[row][c]))
        current_row = bundle(row_bindings) if row_bindings else np.zeros_like(ideal_row)

        # Column
        col_bindings = []
        for r in range(self.size):
            if grid[r][col] is not None:
                col_bindings.append(self.encode_cell(r, col, grid[r][col]))
        current_col = bundle(col_bindings) if col_bindings else np.zeros_like(ideal_col)

        # Block
        br = (row // self.block_size) * self.block_size
        bc = (col // self.block_size) * self.block_size
        block_bindings = []
        for r in range(br, br + self.block_size):
            for c in range(bc, bc + self.block_size):
                if grid[r][c] is not None:
                    block_bindings.append(self.encode_cell(r, c, grid[r][c]))
        current_block = bundle(block_bindings) if block_bindings else np.zeros_like(ideal_block)

        # Score each available digit
        scores = {}
        for d in available:
            test_binding = self.encode_cell(row, col, d)

            # Row similarity
            test_row = bundle([current_row, test_binding])
            row_sim = similarity(test_row, ideal_row)

            # Column similarity
            test_col = bundle([current_col, test_binding])
            col_sim = similarity(test_col, ideal_col)

            # Block similarity
            test_block = bundle([current_block, test_binding])
            block_sim = similarity(test_block, ideal_block)

            # Combine based on strategy
            if strategy == "avg":
                scores[d] = (row_sim + col_sim + block_sim) / 3.0
            elif strategy == "min":
                scores[d] = min(row_sim, col_sim, block_sim)
            elif strategy == "max":
                scores[d] = max(row_sim, col_sim, block_sim)
            elif strategy == "product":
                scores[d] = row_sim * col_sim * block_sim

        return scores

    def solve_via_full_completion(self, puzzle: List[List[Optional[int]]],
                                   strategy: str = "avg") -> Tuple[bool, List[List[int]]]:
        """
        Solve using all three constraint types for pattern completion.
        """
        self.log(f"\n{'='*60}")
        self.log(f"APPROACH 5C: FULL CONSTRAINT COMPLETION (strategy={strategy})")
        self.log(f"{'='*60}")

        grid = [[cell for cell in row] for row in puzzle]
        self.log(f"Empty cells: {count_empty(grid)}")

        max_iterations = self.size * self.size
        self.iterations = 0

        while self.iterations < max_iterations:
            self.iterations += 1

            # Find best placement across all empty cells
            best_cell = None
            best_digit = None
            best_score = -1
            best_gap = 0

            for r in range(self.size):
                for c in range(self.size):
                    if grid[r][c] is not None:
                        continue

                    scores = self.query_full_completion(grid, r, c, strategy=strategy)

                    if not scores:
                        continue

                    # Find best and second-best
                    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                    top_d, top_s = sorted_scores[0]
                    second_s = sorted_scores[1][1] if len(sorted_scores) > 1 else 0
                    gap = top_s - second_s

                    # Prefer cells with clear winner (larger gap)
                    if gap > best_gap:
                        best_cell = (r, c)
                        best_digit = top_d
                        best_score = top_s
                        best_gap = gap

            if best_cell is None:
                empty = count_empty(grid)
                if empty == 0:
                    self.log(f"\n✓ All cells filled after {self.iterations} iterations")
                else:
                    self.log(f"\n⚠ No valid placements found, {empty} cells empty")
                break

            r, c = best_cell
            self.log(f"  [{strategy}] ({r},{c}) → {best_digit} (score={best_score:.4f}, gap={best_gap:.4f})")
            grid[r][c] = best_digit

        return count_empty(grid) == 0, grid

    def solve_via_entanglement_query(self, puzzle: List[List[Optional[int]]]) -> Tuple[bool, List[List[int]]]:
        """
        Attempt 1: Solve by querying encoded grid.

        Encode all known cells, then query unknown positions.
        """
        self.log(f"\n{'='*60}")
        self.log("APPROACH 5A: ENTANGLEMENT QUERY")
        self.log(f"{'='*60}")

        grid = [[cell for cell in row] for row in puzzle]
        self.log(f"Empty cells: {count_empty(grid)}")

        max_iterations = self.size * self.size

        while self.iterations < max_iterations:
            self.iterations += 1

            # Encode current grid state
            grid_vec = self.encode_grid(grid)

            # Find an empty cell
            found = False
            for r in range(self.size):
                for c in range(self.size):
                    if grid[r][c] is not None:
                        continue

                    # Query what digit is at this position
                    digit, conf = self.query_digit_at_position(grid_vec, r, c)
                    available = self.get_available(grid, r, c)

                    if digit in available:
                        self.log(f"  [Query] ({r},{c}) → {digit} (conf={conf:.4f})")
                        grid[r][c] = digit
                        found = True
                        break
                    else:
                        # Query returned invalid digit
                        self.log(f"  [Query] ({r},{c}) → {digit} INVALID (not in {available})")

                if found:
                    break

            if not found:
                # Check if solved or stuck
                empty = count_empty(grid)
                if empty == 0:
                    self.log(f"\n✓ All cells filled")
                    break
                else:
                    self.log(f"\n⚠ Stuck with {empty} cells empty")
                    break

        return count_empty(grid) == 0, grid

    def query_col_completion(self, grid: List[List[Optional[int]]],
                             row: int, col: int) -> Dict[int, float]:
        """Score digits based on column completion."""
        ideal_col = self.encode_ideal_col(col)

        col_bindings = []
        for r in range(self.size):
            if grid[r][col] is not None:
                col_bindings.append(self.encode_cell(r, col, grid[r][col]))
        current_col = bundle(col_bindings) if col_bindings else np.zeros_like(ideal_col)

        scores = {}
        available = self.get_available(grid, row, col)

        for d in available:
            test_binding = self.encode_cell(row, col, d)
            test_col = bundle([current_col, test_binding])
            scores[d] = similarity(test_col, ideal_col)

        return scores

    def query_block_completion(self, grid: List[List[Optional[int]]],
                               row: int, col: int) -> Dict[int, float]:
        """Score digits based on block completion."""
        ideal_block = self.encode_ideal_block(row, col)

        br = (row // self.block_size) * self.block_size
        bc = (col // self.block_size) * self.block_size
        block_bindings = []
        for r in range(br, br + self.block_size):
            for c in range(bc, bc + self.block_size):
                if grid[r][c] is not None:
                    block_bindings.append(self.encode_cell(r, c, grid[r][c]))
        current_block = bundle(block_bindings) if block_bindings else np.zeros_like(ideal_block)

        scores = {}
        available = self.get_available(grid, row, col)

        for d in available:
            test_binding = self.encode_cell(row, col, d)
            test_block = bundle([current_block, test_binding])
            scores[d] = similarity(test_block, ideal_block)

        return scores

    def query_voting(self, grid: List[List[Optional[int]]],
                     row: int, col: int) -> Tuple[Dict[int, float], int]:
        """
        Each constraint type votes for its preferred digit.
        Return combined scores and number of unanimous votes.
        """
        available = self.get_available(grid, row, col)
        if not available:
            return {}, 0

        row_scores = self.query_row_completion(grid, row, col)
        col_scores = self.query_col_completion(grid, row, col)
        block_scores = self.query_block_completion(grid, row, col)

        # Find winner for each
        row_winner = max(row_scores, key=row_scores.get) if row_scores else None
        col_winner = max(col_scores, key=col_scores.get) if col_scores else None
        block_winner = max(block_scores, key=block_scores.get) if block_scores else None

        # Count votes for each digit
        votes = {d: 0 for d in available}
        vote_scores = {d: 0.0 for d in available}

        if row_winner and row_winner in votes:
            votes[row_winner] += 1
            vote_scores[row_winner] += row_scores[row_winner]
        if col_winner and col_winner in votes:
            votes[col_winner] += 1
            vote_scores[col_winner] += col_scores[col_winner]
        if block_winner and block_winner in votes:
            votes[block_winner] += 1
            vote_scores[block_winner] += block_scores[block_winner]

        # Combine: weight by number of votes and total score
        combined = {}
        for d in available:
            combined[d] = votes[d] * 10 + vote_scores[d]  # Votes matter more

        max_votes = max(votes.values())
        return combined, max_votes

    def solve_via_voting(self, puzzle: List[List[Optional[int]]]) -> Tuple[bool, List[List[int]]]:
        """
        Solve using constraint voting.
        Place digit only when all 3 constraints agree (unanimous vote).
        """
        self.log(f"\n{'='*60}")
        self.log("APPROACH 5D: CONSTRAINT VOTING")
        self.log(f"{'='*60}")

        grid = [[cell for cell in row] for row in puzzle]
        self.log(f"Empty cells: {count_empty(grid)}")

        max_iterations = self.size * self.size * 2
        self.iterations = 0

        while self.iterations < max_iterations:
            self.iterations += 1

            # Find cell with unanimous agreement (3 votes)
            best_cell = None
            best_digit = None
            best_score = -1

            for r in range(self.size):
                for c in range(self.size):
                    if grid[r][c] is not None:
                        continue

                    scores, max_votes = self.query_voting(grid, r, c)

                    if not scores:
                        continue

                    top_d = max(scores, key=scores.get)
                    top_s = scores[top_d]

                    # Prefer cells with unanimous votes, then by score
                    if top_s > best_score:
                        best_cell = (r, c)
                        best_digit = top_d
                        best_score = top_s

            if best_cell is None:
                empty = count_empty(grid)
                if empty == 0:
                    self.log(f"\n✓ All cells filled")
                else:
                    self.log(f"\n⚠ Stuck, {empty} cells empty")
                break

            r, c = best_cell
            votes = int(best_score / 10)
            self.log(f"  [Vote{votes}] ({r},{c}) → {best_digit} (score={best_score:.4f})")
            grid[r][c] = best_digit

        return count_empty(grid) == 0, grid

    def would_cause_contradiction(self, grid: List[List[Optional[int]]],
                                    row: int, col: int, digit: int) -> bool:
        """
        Check if placing digit at (row, col) would cause immediate contradiction.

        A contradiction occurs if any peer cell has no valid options left.
        """
        # Temporarily place
        grid[row][col] = digit

        # Check all peers
        contradiction = False
        for r in range(self.size):
            for c in range(self.size):
                if grid[r][c] is not None:
                    continue
                available = self.get_available(grid, r, c)
                if len(available) == 0:
                    contradiction = True
                    break
            if contradiction:
                break

        # Undo
        grid[row][col] = None
        return contradiction

    def solve_via_row_lookahead(self, puzzle: List[List[Optional[int]]]) -> Tuple[bool, List[List[int]]]:
        """
        Row completion with lookahead to avoid contradictions.

        Use geometric scoring but verify placement won't cause immediate contradiction.
        """
        self.log(f"\n{'='*60}")
        self.log("APPROACH 5F: ROW + LOOKAHEAD")
        self.log(f"{'='*60}")

        grid = [[cell for cell in row] for row in puzzle]
        self.log(f"Empty cells: {count_empty(grid)}")

        max_iterations = self.size * self.size * 2
        self.iterations = 0

        while self.iterations < max_iterations:
            self.iterations += 1

            # Find best placement that doesn't cause contradiction
            best_cell = None
            best_digit = None
            best_gap = -1

            for r in range(self.size):
                for c in range(self.size):
                    if grid[r][c] is not None:
                        continue

                    scores = self.query_row_completion(grid, r, c)
                    if not scores:
                        continue

                    # Sort by score
                    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

                    # Find best digit that doesn't cause contradiction
                    for d, s in sorted_scores:
                        if not self.would_cause_contradiction(grid, r, c, d):
                            second_s = sorted_scores[1][1] if len(sorted_scores) > 1 else 0
                            gap = s - second_s

                            if gap > best_gap:
                                best_cell = (r, c)
                                best_digit = d
                                best_gap = gap
                            break  # Found valid digit for this cell

            if best_cell is None:
                empty = count_empty(grid)
                if empty == 0:
                    self.log(f"\n✓ All cells filled")
                else:
                    self.log(f"\n⚠ Stuck, {empty} cells empty")
                break

            r, c = best_cell
            self.log(f"  [Lookahead] ({r},{c}) → {best_digit} (gap={best_gap:.4f})")
            grid[r][c] = best_digit

        return count_empty(grid) == 0, grid

    def solve_via_row_adaptive(self, puzzle: List[List[Optional[int]]],
                                min_gap: float = 0.0) -> Tuple[bool, List[List[int]]]:
        """
        Row completion with adaptive confidence threshold.

        Only place when gap between best and second-best exceeds min_gap.
        This is more conservative and may avoid early wrong decisions.
        """
        self.log(f"\n{'='*60}")
        self.log(f"APPROACH 5E: ROW ADAPTIVE (min_gap={min_gap})")
        self.log(f"{'='*60}")

        grid = [[cell for cell in row] for row in puzzle]
        self.log(f"Empty cells: {count_empty(grid)}")

        max_iterations = self.size * self.size * 2
        self.iterations = 0
        stuck_count = 0

        while self.iterations < max_iterations:
            self.iterations += 1

            # Find placement with largest gap above threshold
            best_cell = None
            best_digit = None
            best_score = -1
            best_gap = -1

            for r in range(self.size):
                for c in range(self.size):
                    if grid[r][c] is not None:
                        continue

                    scores = self.query_row_completion(grid, r, c)

                    if not scores:
                        continue

                    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                    top_d, top_s = sorted_scores[0]
                    second_s = sorted_scores[1][1] if len(sorted_scores) > 1 else 0
                    gap = top_s - second_s

                    # Only consider if gap exceeds threshold
                    if gap >= min_gap and gap > best_gap:
                        best_cell = (r, c)
                        best_digit = top_d
                        best_score = top_s
                        best_gap = gap

            if best_cell is None:
                empty = count_empty(grid)
                if empty == 0:
                    self.log(f"\n✓ All cells filled")
                    break
                else:
                    # No placement above threshold - lower threshold or give up
                    stuck_count += 1
                    if stuck_count > 10:
                        self.log(f"\n⚠ Stuck, {empty} cells empty")
                        break
                    # Skip this iteration, try again
                    continue

            r, c = best_cell
            self.log(f"  [Row] ({r},{c}) → {best_digit} (gap={best_gap:.4f})")
            grid[r][c] = best_digit
            stuck_count = 0  # Reset

        return count_empty(grid) == 0, grid

    def solve_via_row_completion(self, puzzle: List[List[Optional[int]]]) -> Tuple[bool, List[List[int]]]:
        """
        Attempt 2: Solve by comparing to ideal row structure.

        For each empty cell, find digit that makes row most similar to ideal.
        """
        self.log(f"\n{'='*60}")
        self.log("APPROACH 5B: ROW COMPLETION")
        self.log(f"{'='*60}")

        grid = [[cell for cell in row] for row in puzzle]
        self.log(f"Empty cells: {count_empty(grid)}")

        max_iterations = self.size * self.size

        while self.iterations < max_iterations:
            self.iterations += 1

            # Find best placement across all empty cells
            best_cell = None
            best_digit = None
            best_score = -1
            best_gap = 0

            for r in range(self.size):
                for c in range(self.size):
                    if grid[r][c] is not None:
                        continue

                    scores = self.query_row_completion(grid, r, c)

                    if not scores:
                        continue

                    # Find best and second-best
                    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                    top_d, top_s = sorted_scores[0]
                    second_s = sorted_scores[1][1] if len(sorted_scores) > 1 else 0
                    gap = top_s - second_s

                    # Prefer cells with clear winner (larger gap)
                    if gap > best_gap:
                        best_cell = (r, c)
                        best_digit = top_d
                        best_score = top_s
                        best_gap = gap

            if best_cell is None:
                empty = count_empty(grid)
                if empty == 0:
                    self.log(f"\n✓ All cells filled")
                else:
                    self.log(f"\n⚠ No valid placements found, {empty} cells empty")
                break

            r, c = best_cell
            self.log(f"  [RowComplete] ({r},{c}) → {best_digit} (sim={best_score:.4f}, gap={best_gap:.4f})")
            grid[r][c] = best_digit

        return count_empty(grid) == 0, grid


def test_unbind_from_bundle():
    """Test: Can we recover bindings from a bundled structure?"""
    print("\n" + "=" * 60)
    print("TEST: Unbinding from Bundled Structure")
    print("=" * 60)

    solver = EntanglementSolver(size=9, verbose=False)

    # Create a partial grid with known values
    cells = [(0, 0, 5), (0, 1, 3), (0, 4, 7), (1, 0, 6)]

    # Encode as bundled bindings
    bindings = [solver.encode_cell(r, c, d) for r, c, d in cells]
    grid_vec = bundle(bindings)

    print(f"\nEncoded {len(cells)} cell bindings into grid vector")

    # Try to recover each digit
    print("\nRecovery test:")
    for r, c, expected in cells:
        digit, conf = solver.query_digit_at_position(grid_vec, r, c)
        match = "✓" if digit == expected else "✗"
        print(f"  ({r},{c}): expected={expected}, recovered={digit} (conf={conf:.4f}) {match}")

    # Also test an unknown position
    print("\nUnknown position test:")
    for pos in [(0, 2), (1, 1), (2, 0)]:
        digit, conf = solver.query_digit_at_position(grid_vec, pos[0], pos[1])
        print(f"  {pos}: recovered={digit} (conf={conf:.4f}) - THIS IS NOISE")


def test_4x4():
    """Test entanglement approach on 4x4."""
    print("\n" + "=" * 60)
    print("TEST: 4x4 with Entanglement Query")
    print("=" * 60)

    print("\nInput puzzle:")
    print_grid_4x4(PUZZLE_4x4_EASY)

    solver = EntanglementSolver(size=4, verbose=True)

    with Timer() as timer:
        solved, grid = solver.solve_via_entanglement_query(PUZZLE_4x4_EASY)

    print("\nResult:")
    print_grid_4x4(grid)

    valid, msg = validate_4x4(grid)
    print(f"\nSolved: {solved}, Valid: {valid} - {msg}")
    print(f"Entanglement queries: {solver.entanglement_queries}")

    return solved and valid


def test_9x9_easy():
    """Test entanglement on 9x9 easy."""
    print("\n" + "=" * 60)
    print("TEST: 9x9 Easy with Entanglement Query")
    print("=" * 60)

    print("\nInput puzzle:")
    print_grid_9x9(PUZZLE_9x9_EASY)

    solver = EntanglementSolver(size=9, verbose=True)

    with Timer() as timer:
        solved, grid = solver.solve_via_entanglement_query(PUZZLE_9x9_EASY)

    print("\nResult:")
    print_grid_9x9(grid)

    valid, msg = validate_9x9(grid)
    print(f"\nSolved: {solved}, Valid: {valid} - {msg}")

    return solved and valid


def test_row_completion():
    """Test row completion approach."""
    print("\n" + "=" * 60)
    print("TEST: Row Completion Approach")
    print("=" * 60)

    print("\nInput puzzle (4x4):")
    print_grid_4x4(PUZZLE_4x4_EASY)

    solver = EntanglementSolver(size=4, verbose=True)

    with Timer() as timer:
        solved, grid = solver.solve_via_row_completion(PUZZLE_4x4_EASY)

    print("\nResult:")
    print_grid_4x4(grid)

    valid, msg = validate_4x4(grid)
    print(f"\nSolved: {solved}, Valid: {valid} - {msg}")

    return solved and valid


def test_row_completion_9x9():
    """Test row completion on 9x9 puzzles."""
    print("\n" + "=" * 60)
    print("TEST: 9x9 Easy with Row Completion")
    print("=" * 60)

    print("\nInput puzzle:")
    print_grid_9x9(PUZZLE_9x9_EASY)

    solver = EntanglementSolver(size=9, verbose=True)

    with Timer() as timer:
        solved, grid = solver.solve_via_row_completion(PUZZLE_9x9_EASY)

    print("\nResult:")
    print_grid_9x9(grid)

    valid, msg = validate_9x9(grid)
    print(f"\nSolved: {solved}, Valid: {valid} - {msg}")
    print(f"Time: {timer.elapsed:.4f}s")

    return solved and valid


def test_row_completion_9x9_hard():
    """Test row completion on 9x9 hard puzzle."""
    print("\n" + "=" * 60)
    print("TEST: 9x9 HARD with Row Completion")
    print("=" * 60)

    print("\nInput puzzle:")
    print_grid_9x9(PUZZLE_9x9_HARD)
    print(f"Empty cells: {count_empty(PUZZLE_9x9_HARD)}")

    solver = EntanglementSolver(size=9, verbose=True)

    with Timer() as timer:
        solved, grid = solver.solve_via_row_completion(PUZZLE_9x9_HARD)

    print("\nResult:")
    print_grid_9x9(grid)

    valid, msg = validate_9x9(grid)
    print(f"\nSolved: {solved}, Valid: {valid} - {msg}")
    print(f"Time: {timer.elapsed:.4f}s")

    return solved and valid


def test_full_completion_4x4():
    """Test full constraint completion on 4x4."""
    print("\n" + "=" * 60)
    print("TEST: 4x4 with FULL Constraint Completion")
    print("=" * 60)

    print("\nInput puzzle:")
    print_grid_4x4(PUZZLE_4x4_EASY)

    solver = EntanglementSolver(size=4, verbose=True)

    with Timer() as timer:
        solved, grid = solver.solve_via_full_completion(PUZZLE_4x4_EASY)

    print("\nResult:")
    print_grid_4x4(grid)

    valid, msg = validate_4x4(grid)
    print(f"\nSolved: {solved}, Valid: {valid} - {msg}")
    print(f"Time: {timer.elapsed:.4f}s")

    return solved and valid


def test_full_completion_9x9_hard():
    """Test full constraint completion on 9x9 hard with all strategies."""
    print("\n" + "=" * 60)
    print("TEST: 9x9 HARD - Comparing Combination Strategies")
    print("=" * 60)
    print("\nPrevious row-only: 54/58 cells")
    print("Testing: avg, min, max, product strategies")

    strategies = ["avg", "min", "max", "product"]
    results = {}

    for strategy in strategies:
        print(f"\n--- Strategy: {strategy} ---")
        solver = EntanglementSolver(size=9, verbose=False)

        with Timer() as timer:
            solved, grid = solver.solve_via_full_completion(PUZZLE_9x9_HARD, strategy=strategy)

        cells_filled = 58 - count_empty(grid)
        valid, msg = validate_9x9(grid)

        results[strategy] = {
            "cells": cells_filled,
            "solved": solved and valid,
            "time": timer.elapsed,
            "grid": grid
        }
        print(f"  Cells: {cells_filled}/58, Valid: {valid}, Time: {timer.elapsed:.2f}s")

    # Find best
    best_strategy = max(results, key=lambda s: results[s]["cells"])
    best = results[best_strategy]

    print("\n" + "=" * 60)
    print(f"BEST STRATEGY: {best_strategy}")
    print("=" * 60)
    print(f"Cells filled: {best['cells']}/58")

    print("\nBest result grid:")
    print_grid_9x9(best["grid"])

    return best["solved"]


def test_voting_9x9_hard():
    """Test voting approach on hard puzzle."""
    print("\n" + "=" * 60)
    print("TEST: 9x9 HARD with VOTING")
    print("=" * 60)
    print("\nEach constraint votes independently for best digit.")
    print("Prefer unanimous agreement (3 votes).")

    solver = EntanglementSolver(size=9, verbose=True)

    with Timer() as timer:
        solved, grid = solver.solve_via_voting(PUZZLE_9x9_HARD)

    print("\nResult:")
    print_grid_9x9(grid)

    cells_filled = 58 - count_empty(grid)
    valid, msg = validate_9x9(grid)
    print(f"\nCells filled: {cells_filled}/58")
    print(f"Solved: {solved}, Valid: {valid} - {msg}")
    print(f"Time: {timer.elapsed:.4f}s")

    return solved and valid


def main():
    print("=" * 60)
    print("APPROACH 5: STRUCTURAL ENTANGLEMENT")
    print("=" * 60)
    print("\nTesting ROW + LOOKAHEAD (geometric with contradiction avoidance)")

    # Test the lookahead approach
    print("\n" + "=" * 60)
    print("ROW + LOOKAHEAD TEST")
    print("=" * 60)

    solver = EntanglementSolver(size=9, verbose=True)
    with Timer() as timer:
        solved, grid = solver.solve_via_row_lookahead(PUZZLE_9x9_HARD)

    print("\nResult:")
    print_grid_9x9(grid)

    cells = 58 - count_empty(grid)
    valid, msg = validate_9x9(grid)

    print(f"\nCells filled: {cells}/58")
    print(f"Solved: {solved}, Valid: {valid}")
    print(f"Time: {timer.elapsed:.2f}s")

    # Compare to baseline
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print("| Approach | Cells | Valid | Notes |")
    print("|----------|-------|-------|-------|")
    print("| Row-only (greedy) | 54/58 | ✗ | Contradictions |")
    print(f"| Row + Lookahead | {cells}/58 | {'✓' if valid else '✗'} | Avoids contradictions |")

    if solved and valid:
        print("\n" + "=" * 60)
        print("✓✓✓ SOLVED! Geometric guidance + lookahead works!")
        print("=" * 60)


if __name__ == "__main__":
    main()
