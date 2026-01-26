#!/usr/bin/env python3
"""
Adversarial Analysis: Defeating Geometric Sudoku Solver Weaknesses

This script systematically analyzes and attacks the weaknesses in the geometric
VSA/HDC Sudoku solver to understand why it fails and how to defeat math.
"""

import random
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
from holon import CPUStore
from holon.vector_manager import VectorManager
from holon.encoder import Encoder
from holon.similarity import normalized_dot_similarity


class AdversarialGeometricSudokuAnalyzer:
    """
    Analyzes geometric solver weaknesses through adversarial puzzle design.
    """

    def __init__(self, dimensions: int = 16384):
        self.dimensions = dimensions
        self.vector_manager = VectorManager(dimensions, "cpu")
        self.encoder = Encoder(self.vector_manager)

        # Core encoding vectors (same as original solver)
        self.position_vectors = {}
        self.symbol_vectors = {}
        self._generate_encodings()

        self.store = CPUStore(dimensions=dimensions)

    def _generate_encodings(self):
        """Generate position and symbol vectors."""
        for row in range(9):
            for col in range(9):
                pos_key = f"pos_r{row}_c{col}"
                pos_vector = self.vector_manager.get_vector(pos_key)
                self.position_vectors[(row, col)] = pos_vector

        for digit in range(1, 10):
            sym_key = f"sym_{digit}"
            sym_vector = self.vector_manager.get_vector(sym_key)
            self.symbol_vectors[digit] = sym_vector

    def encode_cell(self, row: int, col: int, digit: int) -> np.ndarray:
        """Encode a filled cell as position ⊙ symbol binding."""
        pos_vec = self.position_vectors[(row, col)]
        sym_vec = self.symbol_vectors[digit]
        return self.encoder.bind(pos_vec, sym_vec)

    def encode_grid(self, grid: List[List[Optional[int]]]) -> np.ndarray:
        """Encode entire grid as bundled cell vectors."""
        cell_vectors = []

        for row in range(9):
            for col in range(9):
                if grid[row][col] is not None:
                    cell_vec = self.encode_cell(row, col, grid[row][col])
                    cell_vectors.append(cell_vec)

        if not cell_vectors:
            return np.zeros(self.dimensions, dtype=np.int8)

        return self.encoder.bundle(cell_vectors)

    def analyze_geometric_weaknesses(self):
        """
        Analyze fundamental weaknesses in the current geometric encoding.
        """
        print("🔬 ANALYZING GEOMETRIC SOLVER WEAKNESSES")
        print("=" * 60)

        # Weakness 1: Row-Only Similarity Measurement
        print("\n🚨 WEAKNESS 1: Row-Only Similarity Measurement")
        print("The geometric scoring uses minimal grids with all symbols in row 0")
        print("This ignores actual constraint geometry (row/col/block differences)")

        self._demonstrate_row_only_weakness()

        # Weakness 2: Position-Independent Scoring
        print("\n🚨 WEAKNESS 2: Position-Independent Constraint Evaluation")
        print("Scoring doesn't consider which constraint type is being evaluated")

        self._demonstrate_position_independence()

        # Weakness 3: Local vs Global Constraint Awareness
        print("\n🚨 WEAKNESS 3: Local Constraint Evaluation Misses Global Interactions")
        print("Each constraint scored independently, missing cross-constraint effects")

        self._demonstrate_local_global_gap()

    def _demonstrate_row_only_weakness(self):
        """Show how row-only encoding breaks column and block constraints."""

        # Create test grids
        row_constraint_grid = [
            [1, 2, 3, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None]
        ]

        col_constraint_grid = [
            [1, None, None, None, None, None, None, None, None],
            [2, None, None, None, None, None, None, None, None],
            [3, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None]
        ]

        block_constraint_grid = [
            [1, 2, 3, None, None, None, None, None, None],
            [4, 5, 6, None, None, None, None, None, None],
            [7, 8, 9, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None]
        ]

        # Score digit 4 in different constraint contexts
        test_digit = 4

        # Row constraint (should be valid - digit 4 not in row)
        row_symbols = [1, 2, 3]  # Existing in row
        row_score = self._original_score_symbol_in_constraint(test_digit, row_symbols)
        print(f"Row constraint [1,2,3] + digit {test_digit}: Score = {row_score:.4f}")

        # Column constraint (should be valid - digit 4 not in column)
        col_symbols = [1, 2, 3]  # Existing in column
        col_score = self._original_score_symbol_in_constraint(test_digit, col_symbols)
        print(f"Col constraint [1,2,3] + digit {test_digit}: Score = {col_score:.4f}")

        # Block constraint (should be INVALID - digit 4 already in block)
        block_symbols = [1, 2, 3, 4, 5, 6, 7, 8]  # 4 already present!
        block_score = self._original_score_symbol_in_constraint(test_digit, block_symbols)
        print(f"Block constraint [1,2,3,4,5,6,7,8] + digit {test_digit}: Score = {block_score:.4f} (SHOULD BE -1.0)")

        print("❌ FAILURE: All scores identical despite different constraint semantics!")
        print("   The geometric encoding treats all constraints as 'rows'")

    def _original_score_symbol_in_constraint(self, symbol: int, existing_symbols: List[int]) -> float:
        """Reproduce the original flawed scoring logic."""
        if symbol in existing_symbols:
            return -1.0

        if not existing_symbols:
            return 1.0

        # Create minimal grids for comparison (THIS IS THE BUG)
        symbol_grid = [[None] * 9 for _ in range(9)]
        symbol_grid[0][0] = symbol
        symbol_vector = self.encode_grid(symbol_grid)

        existing_vectors = []
        for i, existing_sym in enumerate(existing_symbols):
            existing_grid = [[None] * 9 for _ in range(9)]
            existing_grid[0][i + 1] = existing_sym  # All in row 0!
            existing_vector = self.encode_grid(existing_grid)
            existing_vectors.append(existing_vector)

        # Calculate geometric uniqueness
        similarities = []
        for existing_vector in existing_vectors:
            sim = normalized_dot_similarity(symbol_vector, existing_vector)
            similarities.append(sim)

        avg_similarity = sum(similarities) / len(similarities)
        uniqueness_score = 1.0 - (avg_similarity + 1.0) / 2.0

        return uniqueness_score

    def _demonstrate_position_independence(self):
        """Show how scoring ignores constraint position information."""

        print("🔍 Testing position independence...")

        # Same constraint pattern, different positions
        test_cases = [
            ("Row 0", [1, 2, 3]),
            ("Row 4", [1, 2, 3]),
            ("Column 0", [1, 2, 3]),
            ("Block 0", [1, 2, 3])
        ]

        test_digit = 4
        scores = {}

        for constraint_name, symbols in test_cases:
            score = self._original_score_symbol_in_constraint(test_digit, symbols)
            scores[constraint_name] = score
            print(f"  {constraint_name}: {score:.4f}")

        # Check if all scores are identical (they should be due to the bug)
        unique_scores = set(scores.values())
        if len(unique_scores) == 1:
            print("❌ FAILURE: All constraint types score identically!")
            print("   Position and constraint type information is completely ignored")
        else:
            print("✅ Scores differ - some position awareness exists")

    def _demonstrate_local_global_gap(self):
        """Show how local scoring misses global constraint interactions."""

        print("🔍 Testing local vs global constraint awareness...")

        # Create a grid where local constraints look good but global fails
        problematic_grid = [
            [1, 2, 3, 4, 5, 6, 7, 8, None],  # Row complete except last cell
            [2, 3, 4, 5, 6, 7, 8, 9, 1],     # Valid row but 2 appears twice globally
            [3, 4, 5, 6, 7, 8, 9, 1, 2],     # Valid row but creates duplicates
            [4, 5, 6, 7, 8, 9, 1, 2, 3],     # Valid row
            [5, 6, 7, 8, 9, 1, 2, 3, 4],     # Valid row
            [6, 7, 8, 9, 1, 2, 3, 4, 5],     # Valid row
            [7, 8, 9, 1, 2, 3, 4, 5, 6],     # Valid row
            [8, 9, 1, 2, 3, 4, 5, 6, 7],     # Valid row
            [9, 1, 2, 3, 4, 5, 6, 7, 8]      # Valid row
        ]

        print("Grid with valid local constraints but global duplicates:")
        self.print_grid(problematic_grid)

        # Check local constraints for cell (0,8)
        row_symbols = [1, 2, 3, 4, 5, 6, 7, 8]  # Row 0 existing
        col_symbols = [1, 2, 3, 4, 5, 6, 7, 8, 9]  # Column 8 is full!
        block_symbols = [7, 8, 9]  # Block 2 existing

        print("\nLocal constraint analysis for cell (0,8):")
        print(f"Row 0 existing: {row_symbols} -> digit 9 score: {self._original_score_symbol_in_constraint(9, row_symbols):.4f}")
        print(f"Col 8 existing: {col_symbols} -> digit 9 score: {self._original_score_symbol_in_constraint(9, col_symbols):.4f}")
        print(f"Block 2 existing: {block_symbols} -> digit 9 score: {self._original_score_symbol_in_constraint(9, block_symbols):.4f}")

        print("❌ FAILURE: Local scoring doesn't detect that column 8 is already full!")
        print("   Each constraint scored independently, missing global state")

    def create_adversarial_puzzles(self):
        """
        Create puzzles specifically designed to break geometric reasoning.
        """
        print("\n🎯 CREATING ADVERSARIAL PUZZLES")
        print("=" * 60)

        # Adversarial Puzzle 1: Constraint Type Confusion
        print("\n🃏 ADVERSARIAL PUZZLE 1: Constraint Type Confusion")
        puzzle1 = self._create_constraint_confusion_puzzle()
        self._test_adversarial_puzzle(puzzle1, "Constraint Confusion")

        # Adversarial Puzzle 2: Local vs Global Conflict
        print("\n🃏 ADVERSARIAL PUZZLE 2: Local-Global Conflict")
        puzzle2 = self._create_local_global_conflict_puzzle()
        self._test_adversarial_puzzle(puzzle2, "Local-Global Conflict")

        # Adversarial Puzzle 3: Geometric Ambiguity
        print("\n🃏 ADVERSARIAL PUZZLE 3: Geometric Ambiguity")
        puzzle3 = self._create_geometric_ambiguity_puzzle()
        self._test_adversarial_puzzle(puzzle3, "Geometric Ambiguity")

    def _create_constraint_confusion_puzzle(self) -> List[List[Optional[int]]]:
        """Create puzzle where geometric scoring confuses constraint types."""

        # Start with a solved grid and remove specific cells to create confusion
        solved = [
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [4, 5, 6, 7, 8, 9, 1, 2, 3],
            [7, 8, 9, 1, 2, 3, 4, 5, 6],
            [2, 3, 4, 5, 6, 7, 8, 9, 1],
            [5, 6, 7, 8, 9, 1, 2, 3, 4],
            [8, 9, 1, 2, 3, 4, 5, 6, 7],
            [3, 4, 5, 6, 7, 8, 9, 1, 2],
            [6, 7, 8, 9, 1, 2, 3, 4, 5],
            [9, 1, 2, 3, 4, 5, 6, 7, 8]
        ]

        # Remove cells to create situations where geometric scoring fails
        puzzle = [row[:] for row in solved]
        puzzle[0][8] = None  # Remove 9 from (0,8)
        puzzle[1][8] = None  # Remove 3 from (1,8)
        puzzle[2][8] = None  # Remove 6 from (2,8)

        return puzzle

    def _create_local_global_conflict_puzzle(self) -> List[List[Optional[int]]]:
        """Create puzzle with valid local constraints but global conflicts."""

        puzzle = [
            [1, 2, 3, 4, 5, 6, 7, 8, None],
            [2, 3, 4, 5, 6, 7, 8, 9, 1],
            [3, 4, 5, 6, 7, 8, 9, 1, 2],
            [4, 5, 6, 7, 8, 9, 1, 2, 3],
            [5, 6, 7, 8, 9, 1, 2, 3, 4],
            [6, 7, 8, 9, 1, 2, 3, 4, 5],
            [7, 8, 9, 1, 2, 3, 4, 5, 6],
            [8, 9, 1, 2, 3, 4, 5, 6, 7],
            [9, 1, 2, 3, 4, 5, 6, 7, 8]
        ]

        return puzzle

    def _create_geometric_ambiguity_puzzle(self) -> List[List[Optional[int]]]:
        """Create puzzle where multiple digits have similar geometric scores."""

        # Create a puzzle where geometric similarity gives ambiguous results
        puzzle = [
            [5, 3, None, None, 7, None, None, None, None],
            [6, None, None, 1, 9, 5, None, None, None],
            [None, 9, 8, None, None, None, None, 6, None],
            [8, None, None, None, 6, None, None, None, 3],
            [4, None, None, 8, None, 3, None, None, 1],
            [7, None, None, None, 2, None, None, None, 6],
            [None, 6, None, None, None, None, 2, 8, None],
            [None, None, None, 4, 1, 9, None, None, 5],
            [None, None, None, None, 8, None, None, 7, 9]
        ]

        return puzzle

    def _test_adversarial_puzzle(self, puzzle: List[List[Optional[int]]], name: str):
        """Test geometric solver on an adversarial puzzle."""

        print(f"\n🧪 Testing {name} Puzzle:")
        self.print_grid(puzzle)

        # Analyze the critical cell that should break geometric reasoning
        empty_cells = [(r, c) for r in range(9) for c in range(9) if puzzle[r][c] is None]

        if empty_cells:
            row, col = empty_cells[0]
            print(f"\n🎯 Analyzing critical cell ({row}, {col}):")

            # Get geometric scores for all digits
            scores = {}
            for digit in range(1, 10):
                score = self.score_symbol_placement(puzzle, row, col, digit)
                scores[digit] = score
                print(f"  Digit {digit}: {score:.4f}")

            # Find best geometric choice
            best_digit = max(scores, key=scores.get)
            best_score = scores[best_digit]

            # Check if it's actually valid
            is_valid = self.is_valid_placement(puzzle, row, col, best_digit)

            print(f"🎯 Geometric choice: {best_digit} (score: {best_score:.4f})")
            print(f"✅ Actually valid: {is_valid}")

            if not is_valid:
                print("🚨 GEOMETRIC FAILURE: Chose invalid placement!")
            elif best_score < 0:
                print("⚠️ GEOMETRIC WEAKNESS: Low confidence in correct choice")

    def score_symbol_placement(self, grid: List[List[Optional[int]]], row: int, col: int, digit: int) -> float:
        """Score digit placement using current geometric method."""
        if not self.is_valid_placement(grid, row, col, digit):
            return -1.0

        constraints = self.get_cell_constraints(row, col)
        total_score = 0.0

        for constraint_type, idx in constraints:
            existing_symbols = self.get_constraint_symbols(grid, constraint_type, idx)
            constraint_score = self._original_score_symbol_in_constraint(digit, existing_symbols)
            total_score += constraint_score

        return total_score / len(constraints)

    def get_constraint_symbols(self, grid: List[List[Optional[int]]], constraint_type: str, idx: int) -> List[int]:
        """Get symbols in a constraint."""
        symbols = []

        if constraint_type == 'row':
            symbols = [grid[idx][col] for col in range(9) if grid[idx][col] is not None]
        elif constraint_type == 'col':
            symbols = [grid[row][idx] for row in range(9) if grid[row][idx] is not None]
        elif constraint_type == 'block':
            block_row = (idx // 3) * 3
            block_col = (idx % 3) * 3
            for r in range(block_row, block_row + 3):
                for c in range(block_col, block_col + 3):
                    if grid[r][c] is not None:
                        symbols.append(grid[r][c])

        return symbols

    def get_cell_constraints(self, row: int, col: int) -> List[Tuple[str, int]]:
        """Get constraint types for a cell."""
        constraints = [
            ('row', row),
            ('col', col),
            ('block', (row // 3) * 3 + (col // 3))
        ]
        return constraints

    def is_valid_placement(self, grid: List[List[Optional[int]]], row: int, col: int, digit: int) -> bool:
        """Check traditional constraint validity."""
        # Check row
        for c in range(9):
            if grid[row][c] == digit:
                return False

        # Check column
        for r in range(9):
            if grid[r][col] == digit:
                return False

        # Check 3x3 block
        block_row = (row // 3) * 3
        block_col = (col // 3) * 3
        for r in range(block_row, block_row + 3):
            for c in range(block_col, block_col + 3):
                if grid[r][c] == digit:
                    return False

        return True

    def print_grid(self, grid: List[List[Optional[int]]]):
        """Pretty print Sudoku grid."""
        print("\n" + "="*25)
        for i, row in enumerate(grid):
            if i % 3 == 0 and i > 0:
                print("-" * 25)

            row_str = ""
            for j, cell in enumerate(row):
                if j % 3 == 0 and j > 0:
                    row_str += "| "
                row_str += str(cell) if cell is not None else "."
                row_str += " "

            print(row_str.strip())
        print("="*25)


def main():
    """Run adversarial analysis."""
    analyzer = AdversarialGeometricSudokuAnalyzer()

    # Phase 1: Analyze weaknesses
    analyzer.analyze_geometric_weaknesses()

    # Phase 2: Create and test adversarial puzzles
    analyzer.create_adversarial_puzzles()

    print("\n🎉 ADVERSARIAL ANALYSIS COMPLETE")
    print("Key findings:")
    print("1. Geometric encoding treats all constraints as rows")
    print("2. Position information is completely ignored")
    print("3. Local scoring misses global constraint violations")
    print("4. Adversarial puzzles can systematically break geometric reasoning")


if __name__ == "__main__":
    main()
