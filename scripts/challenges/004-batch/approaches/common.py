#!/usr/bin/env python3
"""
Common utilities for radical geometric Sudoku approaches.

This module provides:
- Test puzzles (4x4 and 9x9)
- Validation functions
- Holon client setup
- Timing utilities
- Result reporting
"""

import time
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from holon import CPUStore, HolonClient


# =============================================================================
# HOLON CLIENT SETUP
# =============================================================================

def create_client(dimensions: int = 16384) -> HolonClient:
    """Create a fresh Holon client with specified dimensions."""
    store = CPUStore(dimensions=dimensions, backend="cpu")
    return HolonClient(local_store=store)


def get_encoder(client: HolonClient):
    """Get the encoder from a client."""
    return client._store.encoder


def get_vector_manager(client: HolonClient):
    """Get the vector manager from a client."""
    return client._store.vector_manager


# =============================================================================
# VECTOR OPERATIONS (potential kernel primitives to test)
# =============================================================================

def bind(*vectors: np.ndarray) -> np.ndarray:
    """Bind vectors together (element-wise multiplication)."""
    result = vectors[0].copy()
    for v in vectors[1:]:
        result = result * v
    return result


def unbind(composite: np.ndarray, component: np.ndarray) -> np.ndarray:
    """
    Unbind a component from a composite vector.

    For bipolar vectors, unbinding is the same as binding
    (since x * x = 1 for x in {-1, 1}).
    """
    return bind(composite, component)


def bundle(vectors: List[np.ndarray], threshold: bool = True) -> np.ndarray:
    """Bundle vectors into superposition (sum + optional threshold)."""
    if not vectors:
        return np.zeros_like(vectors[0])

    result = np.sum(vectors, axis=0)

    if threshold:
        # Convert to bipolar: positive → 1, negative → -1, zero → 0
        result = np.where(result > 0, 1, np.where(result < 0, -1, 0)).astype(np.int8)

    return result


def similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute cosine similarity between vectors."""
    v1_float = v1.astype(np.float64)
    v2_float = v2.astype(np.float64)

    dot = np.dot(v1_float, v2_float)
    norm1 = np.linalg.norm(v1_float)
    norm2 = np.linalg.norm(v2_float)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(dot / (norm1 * norm2))


def project(vector: np.ndarray, onto: np.ndarray) -> np.ndarray:
    """Project vector onto another vector's direction."""
    onto_float = onto.astype(np.float64)
    vector_float = vector.astype(np.float64)

    dot = np.dot(vector_float, onto_float)
    norm_sq = np.dot(onto_float, onto_float)

    if norm_sq == 0:
        return np.zeros_like(vector)

    scale = dot / norm_sq
    return (scale * onto_float).astype(np.int8)


def remove_component(vector: np.ndarray, component: np.ndarray) -> np.ndarray:
    """
    Remove a component's contribution from a vector.

    For bundled superpositions, this subtracts the component and re-thresholds.
    """
    # Convert to float for subtraction
    vector_f = vector.astype(np.float64)
    component_f = component.astype(np.float64)

    # Direct subtraction (component was added via bundling)
    result = vector_f - component_f

    # Re-threshold to bipolar
    return np.where(result > 0, 1, np.where(result < 0, -1, 0)).astype(np.int8)


def effective_dimensionality(vector: np.ndarray, basis_vectors: List[np.ndarray]) -> float:
    """
    Measure effective dimensionality as uniformity of projections onto basis.

    High uniformity = vector spans many basis dimensions equally
    Low uniformity = vector concentrated in few dimensions
    """
    projections = [abs(similarity(vector, bv)) for bv in basis_vectors]

    if not projections or max(projections) == 0:
        return 0.0

    # Normalize projections
    total = sum(projections)
    if total == 0:
        return 0.0

    probs = [p / total for p in projections]

    # Entropy-based measure (higher = more uniform)
    entropy = -sum(p * np.log(p + 1e-10) for p in probs if p > 0)
    max_entropy = np.log(len(basis_vectors))

    return entropy / max_entropy if max_entropy > 0 else 0.0


# =============================================================================
# TEST PUZZLES
# =============================================================================

# 4x4 Sudoku (digits 1-4, 2x2 blocks)
PUZZLE_4x4_EASY = [
    [1, None, None, 4],
    [None, 4, 1, None],
    [None, 1, 4, None],
    [4, None, None, 1]
]

PUZZLE_4x4_SOLUTION = [
    [1, 2, 3, 4],
    [3, 4, 1, 2],
    [2, 1, 4, 3],
    [4, 3, 2, 1]
]

# 9x9 Sudoku - Easy (classic example)
PUZZLE_9x9_EASY = [
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

PUZZLE_9x9_EASY_SOLUTION = [
    [5, 3, 4, 6, 7, 8, 9, 1, 2],
    [6, 7, 2, 1, 9, 5, 3, 4, 8],
    [1, 9, 8, 3, 4, 2, 5, 6, 7],
    [8, 5, 9, 7, 6, 1, 4, 2, 3],
    [4, 2, 6, 8, 5, 3, 7, 9, 1],
    [7, 1, 3, 9, 2, 4, 8, 5, 6],
    [9, 6, 1, 5, 3, 7, 2, 8, 4],
    [2, 8, 7, 4, 1, 9, 6, 3, 5],
    [3, 4, 5, 2, 8, 6, 1, 7, 9]
]

# 9x9 Sudoku - Hard (requires guessing in traditional solvers)
PUZZLE_9x9_HARD = [
    [None, None, None, 6, None, None, 4, None, None],
    [7, None, None, None, None, 3, 6, None, None],
    [None, None, None, None, 9, 1, None, 8, None],
    [None, None, None, None, None, None, None, None, None],
    [None, 5, None, 1, 8, None, None, None, 3],
    [None, None, None, 3, None, 6, None, 4, 5],
    [None, 4, None, 2, None, None, None, 6, None],
    [9, None, 3, None, None, None, None, None, None],
    [None, 2, None, None, None, None, 1, None, None]
]


# =============================================================================
# VALIDATION
# =============================================================================

def validate_4x4(grid: List[List[int]]) -> Tuple[bool, str]:
    """Validate a 4x4 Sudoku solution."""
    # Check rows
    for r in range(4):
        if set(grid[r]) != {1, 2, 3, 4}:
            return False, f"Row {r} invalid: {grid[r]}"

    # Check columns
    for c in range(4):
        col = [grid[r][c] for r in range(4)]
        if set(col) != {1, 2, 3, 4}:
            return False, f"Column {c} invalid: {col}"

    # Check 2x2 blocks
    for br in range(2):
        for bc in range(2):
            block = []
            for r in range(br * 2, br * 2 + 2):
                for c in range(bc * 2, bc * 2 + 2):
                    block.append(grid[r][c])
            if set(block) != {1, 2, 3, 4}:
                return False, f"Block ({br},{bc}) invalid: {block}"

    return True, "Valid"


def validate_9x9(grid: List[List[int]]) -> Tuple[bool, str]:
    """Validate a 9x9 Sudoku solution."""
    # Check rows
    for r in range(9):
        if set(grid[r]) != set(range(1, 10)):
            return False, f"Row {r} invalid: {grid[r]}"

    # Check columns
    for c in range(9):
        col = [grid[r][c] for r in range(9)]
        if set(col) != set(range(1, 10)):
            return False, f"Column {c} invalid: {col}"

    # Check 3x3 blocks
    for br in range(3):
        for bc in range(3):
            block = []
            for r in range(br * 3, br * 3 + 3):
                for c in range(bc * 3, bc * 3 + 3):
                    block.append(grid[r][c])
            if set(block) != set(range(1, 10)):
                return False, f"Block ({br},{bc}) invalid: {block}"

    return True, "Valid"


def count_empty(grid: List[List[Optional[int]]]) -> int:
    """Count empty cells in a puzzle."""
    return sum(1 for row in grid for cell in row if cell is None)


def get_available_digits_4x4(grid: List[List[Optional[int]]], row: int, col: int) -> Set[int]:
    """Get available digits for a cell in 4x4 Sudoku."""
    used = set()

    # Row
    for c in range(4):
        if grid[row][c] is not None:
            used.add(grid[row][c])

    # Column
    for r in range(4):
        if grid[r][col] is not None:
            used.add(grid[r][col])

    # Block
    br, bc = (row // 2) * 2, (col // 2) * 2
    for r in range(br, br + 2):
        for c in range(bc, bc + 2):
            if grid[r][c] is not None:
                used.add(grid[r][c])

    return {1, 2, 3, 4} - used


def get_available_digits_9x9(grid: List[List[Optional[int]]], row: int, col: int) -> Set[int]:
    """Get available digits for a cell in 9x9 Sudoku."""
    used = set()

    # Row
    for c in range(9):
        if grid[row][c] is not None:
            used.add(grid[row][c])

    # Column
    for r in range(9):
        if grid[r][col] is not None:
            used.add(grid[r][col])

    # Block
    br, bc = (row // 3) * 3, (col // 3) * 3
    for r in range(br, br + 3):
        for c in range(bc, bc + 3):
            if grid[r][c] is not None:
                used.add(grid[r][c])

    return set(range(1, 10)) - used


# =============================================================================
# DISPLAY
# =============================================================================

def print_grid_4x4(grid: List[List[Optional[int]]]):
    """Pretty print a 4x4 grid."""
    print("┌─────┬─────┐")
    for r in range(4):
        if r == 2:
            print("├─────┼─────┤")
        row_str = "│"
        for c in range(4):
            if c == 2:
                row_str += "│"
            val = grid[r][c]
            row_str += f" {val if val else '.'} "
        row_str += "│"
        print(row_str)
    print("└─────┴─────┘")


def print_grid_9x9(grid: List[List[Optional[int]]]):
    """Pretty print a 9x9 grid."""
    print("┌───────┬───────┬───────┐")
    for r in range(9):
        if r > 0 and r % 3 == 0:
            print("├───────┼───────┼───────┤")
        row_str = "│"
        for c in range(9):
            if c > 0 and c % 3 == 0:
                row_str += "│"
            val = grid[r][c]
            row_str += f" {val if val else '.'} "
        row_str += "│"
        print(row_str)
    print("└───────┴───────┴───────┘")


# =============================================================================
# TIMING AND REPORTING
# =============================================================================

class ApproachResult:
    """Result from testing an approach."""

    def __init__(self, approach_name: str):
        self.approach_name = approach_name
        self.puzzle_size = None
        self.puzzle_name = None
        self.solved = False
        self.solution = None
        self.valid = False
        self.validation_msg = ""
        self.time_seconds = 0.0
        self.iterations = 0
        self.backtracking_used = False
        self.cells_filled_geometrically = 0
        self.notes = []

    def add_note(self, note: str):
        self.notes.append(note)

    def report(self):
        """Print a summary report."""
        print(f"\n{'=' * 60}")
        print(f"APPROACH: {self.approach_name}")
        print(f"{'=' * 60}")
        print(f"Puzzle: {self.puzzle_name} ({self.puzzle_size}x{self.puzzle_size})")
        print(f"Solved: {'✓' if self.solved else '✗'}")
        print(f"Valid: {'✓' if self.valid else '✗'} - {self.validation_msg}")
        print(f"Time: {self.time_seconds:.4f}s")
        print(f"Iterations: {self.iterations}")
        print(f"Backtracking: {'Yes' if self.backtracking_used else 'No'}")
        print(f"Geometric fills: {self.cells_filled_geometrically}")

        if self.notes:
            print("\nNotes:")
            for note in self.notes:
                print(f"  - {note}")

        return self


class Timer:
    """Context manager for timing."""

    def __init__(self):
        self.elapsed = 0.0

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.start


# =============================================================================
# VECTOR CACHE (for consistent atom vectors across approaches)
# =============================================================================

class VectorCache:
    """Cache for consistent vector representations."""

    def __init__(self, client: HolonClient):
        self.client = client
        self.encoder = client._store.encoder
        self._cache: Dict[str, np.ndarray] = {}

    def get_digit_vector(self, digit: int) -> np.ndarray:
        """Get or create vector for a digit."""
        key = f"digit_{digit}"
        if key not in self._cache:
            self._cache[key] = self.encoder.encode_data({"digit": digit})
        return self._cache[key]

    def get_position_vector(self, row: int, col: int) -> np.ndarray:
        """Get or create vector for a position."""
        key = f"pos_{row}_{col}"
        if key not in self._cache:
            self._cache[key] = self.encoder.encode_data({"pos": {"row": row, "col": col}})
        return self._cache[key]

    def get_cell_vector(self, row: int, col: int, digit: int) -> np.ndarray:
        """Get or create vector for a cell placement (position bound to digit)."""
        key = f"cell_{row}_{col}_{digit}"
        if key not in self._cache:
            pos_vec = self.get_position_vector(row, col)
            digit_vec = self.get_digit_vector(digit)
            self._cache[key] = bind(pos_vec, digit_vec)
        return self._cache[key]

    def get_superposition(self, digits: List[int]) -> np.ndarray:
        """Get superposition of multiple digit vectors."""
        key = f"super_{'_'.join(map(str, sorted(digits)))}"
        if key not in self._cache:
            vectors = [self.get_digit_vector(d) for d in digits]
            self._cache[key] = bundle(vectors)
        return self._cache[key]


# =============================================================================
# APPROACH TEMPLATE
# =============================================================================

def approach_template(puzzle, size=9) -> ApproachResult:
    """
    Template for implementing an approach.

    Copy this and fill in the approach-specific logic.
    """
    result = ApproachResult("Template Approach")
    result.puzzle_size = size
    result.puzzle_name = "test_puzzle"

    # Create client and vector cache
    client = create_client()
    cache = VectorCache(client)

    with Timer() as timer:
        # TODO: Implement approach-specific logic here
        pass

    result.time_seconds = timer.elapsed

    # Validate result
    if size == 4:
        result.valid, result.validation_msg = validate_4x4(result.solution)
    else:
        result.valid, result.validation_msg = validate_9x9(result.solution)

    result.solved = result.valid

    return result.report()
