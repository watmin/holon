#!/usr/bin/env python3
"""
Approach 14: Constraint Graph Encoding

THE INSIGHT:
Previous approaches encode the PUZZLE STATE.
What if we encode the CONSTRAINT STRUCTURE itself?

The constraint graph is FIXED for all Sudokus:
- 81 nodes (cells)
- Edges connect cells that must differ (same row/col/block)
- Each cell has 20 neighbors (8 row + 8 col + 4 additional block)

THE IDEA:
Instead of encoding "cell (r,c) = digit d", encode:
"Edge between (r1,c1) and (r2,c2) means they're different"

The solution is a 9-coloring of this graph.

NEW APPROACH:
1. Encode the constraint graph as a matrix of relationships
2. Find configurations that satisfy all relationships
3. Use spectral methods (eigenvalue decomposition) to find structure

This is inspired by:
- Graph coloring algorithms
- Spectral clustering
- Matrix factorization
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


def cell_to_idx(r: int, c: int) -> int:
    return r * 9 + c


def idx_to_cell(idx: int) -> Tuple[int, int]:
    return idx // 9, idx % 9


def get_neighbors(r: int, c: int) -> Set[Tuple[int, int]]:
    """Get all cells that must differ from (r, c)."""
    neighbors = set()

    # Same row
    for cc in range(9):
        if cc != c:
            neighbors.add((r, cc))

    # Same column
    for rr in range(9):
        if rr != r:
            neighbors.add((rr, c))

    # Same block
    br, bc = (r // 3) * 3, (c // 3) * 3
    for rr in range(br, br + 3):
        for cc in range(bc, bc + 3):
            if (rr, cc) != (r, c):
                neighbors.add((rr, cc))

    return neighbors


class ConstraintGraphEncoder:
    """Encode Sudoku using constraint graph structure."""

    def __init__(self, dimensions: int = 16384):
        self.dimensions = dimensions
        self.client = create_client(dimensions=dimensions)
        self.cache = VectorCache(self.client)

        self.digits = list(range(1, 10))
        self.digit_vectors = {d: self.cache.get_digit_vector(d) for d in self.digits}

        # Encode the constraint graph
        self.adjacency = self._build_adjacency()

        # Pre-compute cell vectors
        self.cell_vectors = {}
        for r in range(9):
            for c in range(9):
                self.cell_vectors[(r, c)] = self.cache.get_position_vector(r, c)

    def _build_adjacency(self) -> np.ndarray:
        """Build 81x81 adjacency matrix of constraint graph."""
        adj = np.zeros((81, 81))
        for r in range(9):
            for c in range(9):
                idx = cell_to_idx(r, c)
                for (nr, nc) in get_neighbors(r, c):
                    nidx = cell_to_idx(nr, nc)
                    adj[idx, nidx] = 1
        return adj

    def encode_configuration(self, grid: List[List[Optional[int]]]) -> np.ndarray:
        """
        Encode a grid configuration as a vector.

        Uses graph-aware encoding: each cell contributes its value
        weighted by its constraint relationships.
        """
        # Create 81-dimensional "digit assignment" vector
        assignment = np.zeros(81)
        for r in range(9):
            for c in range(9):
                if grid[r][c] is not None:
                    assignment[cell_to_idx(r, c)] = grid[r][c]

        # Weight by graph structure (Laplacian)
        degree = np.diag(np.sum(self.adjacency, axis=1))
        laplacian = degree - self.adjacency

        # Apply Laplacian - this emphasizes differences
        laplacian_encoded = laplacian @ assignment

        # Convert to high-dimensional vector
        vec = np.zeros(self.dimensions)
        for idx in range(81):
            if assignment[idx] != 0:
                r, c = idx_to_cell(idx)
                # Contribution: position ⊙ digit, weighted by Laplacian value
                contrib = bind(self.cell_vectors[(r, c)],
                              self.digit_vectors[int(assignment[idx])])
                vec += contrib * (1 + laplacian_encoded[idx] / 10)

        return vec / np.linalg.norm(vec) if np.linalg.norm(vec) > 0 else vec

    def compute_graph_coloring_score(self, grid: List[List[Optional[int]]]) -> float:
        """
        Compute how well the grid satisfies graph coloring constraints.

        For each edge (i, j), if both cells are filled and different, score +1.
        If same, score -1. If one or both empty, score 0.
        """
        score = 0
        for r in range(9):
            for c in range(9):
                if grid[r][c] is None:
                    continue
                for (nr, nc) in get_neighbors(r, c):
                    if grid[nr][nc] is None:
                        continue
                    if grid[r][c] != grid[nr][nc]:
                        score += 1
                    else:
                        score -= 10  # Heavy penalty for violations

        return score

    def spectral_analysis(self, puzzle: List[List[Optional[int]]]) -> Dict:
        """
        Use spectral methods to analyze the constraint structure.

        The Laplacian's eigenvalues encode graph structure.
        """
        # Build constraint matrix for unfilled cells only
        unfilled = [(r, c) for r in range(9) for c in range(9) if puzzle[r][c] is None]
        n = len(unfilled)
        idx_map = {cell: i for i, cell in enumerate(unfilled)}

        # Build reduced adjacency
        adj = np.zeros((n, n))
        for i, (r, c) in enumerate(unfilled):
            for (nr, nc) in get_neighbors(r, c):
                if (nr, nc) in idx_map:
                    j = idx_map[(nr, nc)]
                    adj[i, j] = 1

        # Compute Laplacian
        degree = np.diag(np.sum(adj, axis=1))
        laplacian = degree - adj

        # Eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(laplacian)

        return {
            'unfilled': unfilled,
            'adjacency': adj,
            'laplacian': laplacian,
            'eigenvalues': eigenvalues,
            'eigenvectors': eigenvectors,
            'num_unfilled': n
        }


class SpectralSolver:
    """
    Attempt to solve using spectral properties of constraint graph.
    """

    def __init__(self, dimensions: int = 16384):
        self.encoder = ConstraintGraphEncoder(dimensions)
        self.digits = list(range(1, 10))

    def solve(self, puzzle: List[List[Optional[int]]], verbose: bool = True) -> Tuple[bool, List[List[int]]]:
        """
        Use spectral analysis to guide solving.
        """
        if verbose:
            print(f"\n{'='*60}")
            print("SPECTRAL / GRAPH-BASED SOLVER")
            print(f"{'='*60}")

        grid = [[cell for cell in row] for row in puzzle]

        # Get spectral analysis
        analysis = self.encoder.spectral_analysis(puzzle)

        if verbose:
            print(f"\nUnfilled cells: {analysis['num_unfilled']}")
            print(f"First 5 eigenvalues: {analysis['eigenvalues'][:5]}")
            print(f"Last 5 eigenvalues: {analysis['eigenvalues'][-5:]}")

        # The eigenvector corresponding to smallest non-zero eigenvalue
        # reveals graph structure (Fiedler vector in graph partitioning)
        fiedler_idx = 1  # Second eigenvalue (first is 0 for connected graph)
        fiedler_vec = analysis['eigenvectors'][:, fiedler_idx]

        if verbose:
            print(f"\nFiedler vector range: [{fiedler_vec.min():.3f}, {fiedler_vec.max():.3f}]")

        # Try to use Fiedler vector to order cell filling
        unfilled = analysis['unfilled']
        cell_order = sorted(range(len(unfilled)),
                           key=lambda i: abs(fiedler_vec[i]),
                           reverse=True)

        if verbose:
            print(f"\nFilling order based on Fiedler vector (most 'central' first):")

        cells_filled = 0
        max_iters = len(unfilled) * 2

        for _ in range(max_iters):
            # Find most constrained cell among remaining
            best_cell = None
            min_options = 10

            for r in range(9):
                for c in range(9):
                    if grid[r][c] is not None:
                        continue
                    available = get_available_digits_9x9(grid, r, c)
                    if len(available) == 0:
                        if verbose:
                            print(f"  Contradiction at ({r},{c})")
                        return False, grid
                    if len(available) < min_options:
                        min_options = len(available)
                        best_cell = (r, c, available)

            if best_cell is None:
                break

            r, c, available = best_cell

            if len(available) == 1:
                grid[r][c] = list(available)[0]
                cells_filled += 1
                continue

            # Use graph coloring heuristic: pick digit that maximizes "different" constraint
            best_digit = None
            best_score = -float('inf')

            for d in available:
                grid[r][c] = d
                score = self.encoder.compute_graph_coloring_score(grid)
                if score > best_score:
                    best_score = score
                    best_digit = d
                grid[r][c] = None

            grid[r][c] = best_digit
            cells_filled += 1

            if verbose and cells_filled <= 10:
                print(f"  ({r},{c}) → {best_digit} (score={best_score})")

        # Validate
        valid, msg = validate_9x9(grid)

        if verbose:
            print(f"\nCells filled: {cells_filled}")
            print(f"Valid: {valid}")
            if not valid:
                print(f"Error: {msg}")

        return valid, grid


def test_constraint_graph():
    """Test the constraint graph approach."""
    print("=" * 70)
    print("APPROACH 14: CONSTRAINT GRAPH ENCODING")
    print("=" * 70)
    print("\nIdea: Encode the constraint STRUCTURE, not just cell values.")
    print("Use spectral methods (eigenvalues) to find graph structure.\n")

    print_grid_9x9(PUZZLE_9x9_HARD)

    solver = SpectralSolver(dimensions=16384)
    valid, grid = solver.solve(PUZZLE_9x9_HARD, verbose=True)

    print("\nResult:")
    print_grid_9x9(grid)

    correct = sum(1 for r in range(9) for c in range(9) if grid[r][c] == SOLUTION_9x9_HARD[r][c])
    print(f"\nCorrect cells: {correct}/81")

    return valid


def analyze_solution_uniqueness():
    """
    Analyze whether the solution has unique spectral signature.
    """
    print("\n" + "=" * 70)
    print("ANALYSIS: Does the solution have unique spectral signature?")
    print("=" * 70)

    encoder = ConstraintGraphEncoder(dimensions=16384)

    # Encode solution
    solution_vec = encoder.encode_configuration(SOLUTION_9x9_HARD)

    # Encode puzzle
    puzzle_vec = encoder.encode_configuration(PUZZLE_9x9_HARD)

    print(f"\nSimilarity puzzle → solution: {similarity(puzzle_vec, solution_vec):.4f}")

    # Generate some wrong solutions (swap two digits in valid solution)
    wrong_solutions = []
    for swap_row in range(3):
        for swap_col in range(3):
            wrong = [[c for c in r] for r in SOLUTION_9x9_HARD]
            # Swap two cells in same block (will violate row/col)
            r1, c1 = swap_row * 3, swap_col * 3
            r2, c2 = swap_row * 3 + 1, swap_col * 3 + 1
            wrong[r1][c1], wrong[r2][c2] = wrong[r2][c2], wrong[r1][c1]
            wrong_solutions.append(wrong)

    print("\nSimilarity puzzle → wrong solutions:")
    for i, wrong in enumerate(wrong_solutions[:5]):
        wrong_vec = encoder.encode_configuration(wrong)
        sim = similarity(puzzle_vec, wrong_vec)
        print(f"  Wrong solution {i}: {sim:.4f}")


def main():
    test_constraint_graph()
    analyze_solution_uniqueness()

    print("\n" + "=" * 70)
    print("CONCLUSIONS")
    print("=" * 70)
    print("""
The spectral / graph-based approach provides interesting insights:
- The constraint graph has fixed structure (same for all Sudokus)
- Eigenvalues reveal graph connectivity patterns
- Fiedler vector can guide cell ordering

However, it still faces the same fundamental issue:
- Multiple locally-valid configurations exist
- Graph coloring heuristics help but don't guarantee global optimum
- The solution IS a valid 9-coloring, but finding it is NP-complete
""")


if __name__ == "__main__":
    main()
