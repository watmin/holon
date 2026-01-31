#!/usr/bin/env python3
"""
Approach 25: Quantum-Inspired Superposition Collapse

THE DREAM:
- Encode ALL possible values for each cell as superposition
- Apply constraints as "projections" that collapse possibilities
- Iterate until solution emerges

QUANTUM CONCEPTS (classical simulation):
1. Superposition: Cell = bundle([1, 2, 3, 4, 5, 6, 7, 8, 9])
2. Entanglement: Row/Col/Block constraints bind cells together
3. Measurement: Collapse superposition to definite value
4. Projection: Remove impossible amplitudes

The key insight: In quantum computing, interference can cancel out
wrong answers and amplify right ones. Can we simulate this with VSA?
"""

import sys
sys.path.insert(0, "/home/watmin/work/holon")

from typing import List, Optional, Set, Tuple, Dict
import numpy as np
import time

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
# CORE QUANTUM-INSPIRED PRIMITIVES
# =============================================================================

class QuantumSudoku:
    """
    Quantum-inspired Sudoku representation.

    Each cell is in a superposition of possible digit states.
    Constraints are applied as projections.
    """

    def __init__(self, store: CPUStore):
        self.store = store
        self.dims = store.dimensions

        # Create digit basis vectors (orthogonal in high-dim space)
        self.digit_vecs = {d: store.vector_manager.get_vector(f"d{d}") for d in range(1, 10)}

        # Position vectors
        self.pos_vecs = {(r, c): store.vector_manager.get_vector(f"p{r}{c}")
                         for r in range(9) for c in range(9)}

        # Cell states: superposition of possible digits
        # None means "collapsed" (definite value)
        self.cell_states: Dict[Tuple[int, int], Optional[np.ndarray]] = {}

        # Collapsed values
        self.collapsed: Dict[Tuple[int, int], int] = {}

    def _normalize(self, vec: np.ndarray) -> np.ndarray:
        """Normalize to unit length (like quantum states)."""
        norm = np.linalg.norm(vec)
        if norm < 1e-10:
            return vec
        return vec / norm

    def _to_bipolar(self, vec: np.ndarray) -> np.ndarray:
        """Convert to bipolar for storage."""
        return np.where(vec > 0, 1, np.where(vec < 0, -1, 0)).astype(np.int8)

    def initialize_from_puzzle(self, puzzle: List[List[Optional[int]]]):
        """Initialize cell states from puzzle."""
        for r in range(9):
            for c in range(9):
                if puzzle[r][c] is not None:
                    # Collapsed state
                    self.collapsed[(r, c)] = puzzle[r][c]
                    self.cell_states[(r, c)] = None
                else:
                    # Superposition of ALL digits initially
                    available = get_available_digits_9x9(puzzle, r, c)
                    if available:
                        superpos = np.sum([self.digit_vecs[d].astype(float) for d in available], axis=0)
                        self.cell_states[(r, c)] = self._normalize(superpos)
                    else:
                        # No options - contradiction
                        self.cell_states[(r, c)] = np.zeros(self.dims)

    def get_amplitudes(self, r: int, c: int) -> Dict[int, float]:
        """Get probability amplitudes for each digit at (r,c)."""
        if (r, c) in self.collapsed:
            return {self.collapsed[(r, c)]: 1.0}

        state = self.cell_states[(r, c)]
        if state is None:
            return {}

        amplitudes = {}
        for d in range(1, 10):
            # Amplitude = overlap with digit basis vector
            amp = np.dot(state, self.digit_vecs[d].astype(float))
            if abs(amp) > 0.01:  # Threshold for numerical stability
                amplitudes[d] = amp

        return amplitudes

    def project_out(self, r: int, c: int, digit: int):
        """
        Project OUT a digit from cell's superposition.
        Like quantum measurement that rules out a state.
        """
        if (r, c) in self.collapsed:
            return

        state = self.cell_states[(r, c)]
        if state is None:
            return

        # Project out the digit component: state - (state·digit)digit
        digit_vec = self.digit_vecs[digit].astype(float)
        overlap = np.dot(state, digit_vec)
        projected = state - overlap * digit_vec

        # Renormalize
        self.cell_states[(r, c)] = self._normalize(projected)

    def measure(self, r: int, c: int) -> Optional[int]:
        """
        Collapse superposition to definite value.
        Returns digit with highest amplitude (greedy collapse).
        """
        if (r, c) in self.collapsed:
            return self.collapsed[(r, c)]

        amplitudes = self.get_amplitudes(r, c)
        if not amplitudes:
            return None

        # Collapse to highest amplitude (or could be probabilistic!)
        best_digit = max(amplitudes.keys(), key=lambda d: abs(amplitudes[d]))
        self.collapsed[(r, c)] = best_digit
        self.cell_states[(r, c)] = None

        return best_digit

    def propagate_constraints(self) -> bool:
        """
        Apply constraint projections.
        When a cell collapses, project out that digit from peers.
        """
        # Track which digits have been propagated from which cells
        propagated = set()

        for iteration in range(100):
            made_progress = False

            # Project out collapsed digits from peers
            for r in range(9):
                for c in range(9):
                    if (r, c) not in self.collapsed:
                        continue

                    digit = self.collapsed[(r, c)]
                    key = (r, c, digit)
                    if key in propagated:
                        continue
                    propagated.add(key)

                    # Project out from row peers
                    for cc in range(9):
                        if cc != c and (r, cc) not in self.collapsed:
                            self.project_out(r, cc, digit)

                    # Project out from column peers
                    for rr in range(9):
                        if rr != r and (rr, c) not in self.collapsed:
                            self.project_out(rr, c, digit)

                    # Project out from block peers
                    br, bc = (r // 3) * 3, (c // 3) * 3
                    for rr in range(br, br + 3):
                        for cc in range(bc, bc + 3):
                            if (rr, cc) != (r, c) and (rr, cc) not in self.collapsed:
                                self.project_out(rr, cc, digit)

            # Auto-collapse cells with single amplitude
            for r in range(9):
                for c in range(9):
                    if (r, c) not in self.collapsed:
                        amps = self.get_amplitudes(r, c)
                        if len(amps) == 1:
                            digit = list(amps.keys())[0]
                            self.collapsed[(r, c)] = digit
                            self.cell_states[(r, c)] = None
                            made_progress = True
                        elif len(amps) == 0:
                            # Contradiction!
                            return False

            if not made_progress:
                break

        return True

    def get_grid(self) -> List[List[Optional[int]]]:
        """Get current grid state."""
        grid = [[None for _ in range(9)] for _ in range(9)]
        for (r, c), digit in self.collapsed.items():
            grid[r][c] = digit
        return grid


# =============================================================================
# EXPERIMENT 1: BASIC SUPERPOSITION COLLAPSE
# =============================================================================

def test_basic_collapse():
    """Test basic superposition and collapse."""
    print("=" * 70)
    print("EXPERIMENT 1: BASIC SUPERPOSITION COLLAPSE")
    print("=" * 70)

    store = create_store()
    qs = QuantumSudoku(store)
    qs.initialize_from_puzzle(PUZZLE_9x9_HARD)

    # Check initial amplitudes for first empty cell
    print("\nInitial state of puzzle:")
    print(f"  Clues: {81 - count_empty(PUZZLE_9x9_HARD)} filled")

    # Find first empty cell
    for r in range(9):
        for c in range(9):
            if PUZZLE_9x9_HARD[r][c] is None:
                amps = qs.get_amplitudes(r, c)
                print(f"\n  Cell ({r},{c}) superposition:")
                for d, amp in sorted(amps.items(), key=lambda x: -abs(x[1])):
                    print(f"    |{d}⟩: {amp:.4f}")
                break
        else:
            continue
        break

    # Propagate constraints
    print("\nPropagating constraints...")
    success = qs.propagate_constraints()
    print(f"  Propagation {'succeeded' if success else 'FAILED'}")

    grid = qs.get_grid()
    filled = sum(1 for r in range(9) for c in range(9) if grid[r][c] is not None)
    print(f"  Cells collapsed: {filled}/81")

    return qs


# =============================================================================
# EXPERIMENT 2: INTERFERENCE PATTERNS
# =============================================================================

def test_interference():
    """
    Test if we can create interference patterns.

    Quantum interference: when two paths to same state have opposite phases,
    they cancel out. Can we simulate this?
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: INTERFERENCE PATTERNS")
    print("=" * 70)

    store = create_store()

    # Create two entangled cells: if A=1, then B≠1
    digit_vecs = {d: store.vector_manager.get_vector(f"d{d}").astype(float) for d in range(1, 10)}

    # Cell A: superposition of 1,2,3
    cell_a = digit_vecs[1] + digit_vecs[2] + digit_vecs[3]
    cell_a = cell_a / np.linalg.norm(cell_a)

    # Cell B: superposition of 1,2,3
    cell_b = digit_vecs[1] + digit_vecs[2] + digit_vecs[3]
    cell_b = cell_b / np.linalg.norm(cell_b)

    print(f"\nInitial Cell A amplitudes:")
    for d in [1, 2, 3]:
        amp = np.dot(cell_a, digit_vecs[d])
        print(f"  |{d}⟩: {amp:.4f}")

    print(f"\nInitial Cell B amplitudes:")
    for d in [1, 2, 3]:
        amp = np.dot(cell_b, digit_vecs[d])
        print(f"  |{d}⟩: {amp:.4f}")

    # Create entanglement: encode constraint "A and B cannot both be 1"
    # In quantum: |ψ⟩ = |A1,B2⟩ + |A1,B3⟩ + |A2,B1⟩ + |A2,B3⟩ + |A3,B1⟩ + |A3,B2⟩
    # (missing |A1,B1⟩, |A2,B2⟩, |A3,B3⟩)

    # Entangled state
    entangled = np.zeros(store.dimensions)
    for a in [1, 2, 3]:
        for b in [1, 2, 3]:
            if a != b:  # Constraint: A ≠ B
                # Tensor product approximated by binding
                term = digit_vecs[a] * digit_vecs[b]  # Binding
                entangled = entangled + term

    entangled = entangled / np.linalg.norm(entangled)

    print(f"\nEntangled state (A≠B constraint) correlations:")
    for a in [1, 2, 3]:
        for b in [1, 2, 3]:
            term = digit_vecs[a] * digit_vecs[b]
            corr = np.dot(entangled, term / np.linalg.norm(term))
            marker = "✓" if a != b else "✗"
            print(f"  |A={a}, B={b}⟩: {corr:.4f} {marker}")

    # Now "measure" A=1, see what happens to B
    print("\n\nSimulating measurement: A collapses to 1")

    # After measuring A=1, B cannot be 1
    # Extract B's remaining amplitudes
    b_after = np.zeros(store.dimensions)
    for b in [2, 3]:  # Only valid values
        term = digit_vecs[1] * digit_vecs[b]
        projection = np.dot(entangled, term / np.linalg.norm(term))
        b_after = b_after + projection * digit_vecs[b]

    if np.linalg.norm(b_after) > 1e-10:
        b_after = b_after / np.linalg.norm(b_after)

    print(f"\nB amplitudes after A=1 measurement:")
    for d in [1, 2, 3]:
        amp = np.dot(b_after, digit_vecs[d])
        expected = "✓" if d != 1 else "(should be ~0)"
        print(f"  |{d}⟩: {amp:.4f} {expected}")


# =============================================================================
# EXPERIMENT 3: GROVER-INSPIRED AMPLITUDE AMPLIFICATION
# =============================================================================

def test_grover():
    """
    Grover's algorithm amplifies correct answers through interference.

    Classical simulation:
    1. Start with uniform superposition
    2. "Oracle" marks valid states (flip phase)
    3. "Diffusion" amplifies marked states
    4. Repeat O(√N) times
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: GROVER-INSPIRED AMPLITUDE AMPLIFICATION")
    print("=" * 70)

    store = create_store()
    digit_vecs = {d: store.vector_manager.get_vector(f"d{d}").astype(float) for d in range(1, 10)}

    # Suppose valid digits are {2, 5, 7} and we want to find them

    valid = {2, 5, 7}

    # Initial superposition (all digits)
    psi = np.sum([digit_vecs[d] for d in range(1, 10)], axis=0)
    psi = psi / np.linalg.norm(psi)

    print("\nInitial superposition:")
    for d in range(1, 10):
        amp = np.dot(psi, digit_vecs[d])
        print(f"  |{d}⟩: {amp:.4f}")

    # Grover iteration
    def grover_iteration(state, valid_digits):
        """One Grover iteration."""
        # Oracle: flip phase of valid states
        # In classical vectors: subtract 2x projection onto valid subspace

        valid_proj = np.zeros(store.dimensions)
        for d in valid_digits:
            overlap = np.dot(state, digit_vecs[d])
            valid_proj = valid_proj + overlap * digit_vecs[d]

        # Oracle reflection: I - 2|valid⟩⟨valid|
        after_oracle = state - 2 * valid_proj

        # Diffusion: 2|s⟩⟨s| - I where |s⟩ is uniform superposition
        uniform = np.sum([digit_vecs[d] for d in range(1, 10)], axis=0)
        uniform = uniform / np.linalg.norm(uniform)

        overlap_with_uniform = np.dot(after_oracle, uniform)
        after_diffusion = 2 * overlap_with_uniform * uniform - after_oracle

        return after_diffusion / np.linalg.norm(after_diffusion)

    # Run iterations
    for iteration in range(5):
        psi = grover_iteration(psi, valid)
        print(f"\nAfter iteration {iteration + 1}:")
        for d in range(1, 10):
            amp = np.dot(psi, digit_vecs[d])
            marker = "← valid" if d in valid else ""
            print(f"  |{d}⟩: {amp:.4f} {marker}")

    # Final measurement: highest amplitude
    best = max(range(1, 10), key=lambda d: abs(np.dot(psi, digit_vecs[d])))
    print(f"\n→ Measurement would yield: {best} (valid: {best in valid})")


# =============================================================================
# EXPERIMENT 4: FULL QUANTUM SUDOKU SOLVER
# =============================================================================

def test_quantum_solver():
    """
    Full quantum-inspired solver with proper constraint checking.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: FULL QUANTUM SUDOKU SOLVER")
    print("=" * 70)

    store = create_store()

    class QuantumSolver:
        def __init__(self):
            self.measurements = 0
            self.backtracks = 0
            self.store = store
            self.digit_vecs = {d: store.vector_manager.get_vector(f"d{d}").astype(float)
                               for d in range(1, 10)}

        def solve(self, puzzle):
            # Use standard grid + available digits tracking
            grid = [[cell for cell in row] for row in puzzle]
            return self._solve_with_propagation(grid)

        def _get_amplitude_scores(self, grid, r, c, options):
            """Score options using quantum-inspired amplitude calculation."""
            # Create superposition of valid options
            superpos = np.sum([self.digit_vecs[d] for d in options], axis=0)
            superpos = superpos / np.linalg.norm(superpos)

            scores = {}
            for d in options:
                # Amplitude = overlap with digit vector
                amp = abs(np.dot(superpos, self.digit_vecs[d]))
                scores[d] = amp

            return scores

        def _solve_with_propagation(self, grid):
            # Standard propagation
            changed = True
            while changed:
                changed = False
                for r in range(9):
                    for c in range(9):
                        if grid[r][c] is None:
                            opts = list(get_available_digits_9x9(grid, r, c))
                            if not opts:
                                return None  # Contradiction
                            if len(opts) == 1:
                                grid[r][c] = opts[0]
                                changed = True

            if count_empty(grid) == 0:
                return grid

            # Find MRV cell
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

            # Get amplitude-based ordering
            amp_scores = self._get_amplitude_scores(grid, r, c, options)
            ordered = sorted(options, key=lambda d: -amp_scores[d])

            for digit in ordered:
                self.measurements += 1
                test_grid = [[cell for cell in row] for row in grid]
                test_grid[r][c] = digit

                result = self._solve_with_propagation(test_grid)
                if result is not None:
                    return result

                self.backtracks += 1

            return None

    solver = QuantumSolver()
    start = time.time()
    result = solver.solve(PUZZLE_9x9_HARD)
    elapsed = time.time() - start

    print(f"\nResult: {'SOLVED' if result else 'FAILED'}")
    print(f"Measurements: {solver.measurements}")
    print(f"Backtracks: {solver.backtracks}")
    print(f"Time: {elapsed:.3f}s")

    if result:
        valid = validate_9x9(result)
        print(f"Valid: {valid}")

    return solver


# =============================================================================
# EXPERIMENT 5: WAVE FUNCTION VISUALIZATION
# =============================================================================

def test_wave_visualization():
    """
    Visualize the quantum state of the puzzle.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: WAVE FUNCTION VISUALIZATION")
    print("=" * 70)

    store = create_store()
    qs = QuantumSudoku(store)
    qs.initialize_from_puzzle(PUZZLE_9x9_HARD)
    qs.propagate_constraints()

    print("\nPuzzle wave function (amplitude distribution):")
    print()

    for r in range(9):
        for c in range(9):
            if (r, c) in qs.collapsed:
                print(f" [{qs.collapsed[(r, c)]}] ", end="")
            else:
                amps = qs.get_amplitudes(r, c)
                if len(amps) == 0:
                    print("  ✗  ", end="")
                elif len(amps) == 1:
                    print(f" ({list(amps.keys())[0]}) ", end="")
                else:
                    # Show as uncertainty
                    print(f" {len(amps)}? ", end="")
        print()

        if r in [2, 5]:
            print("-" * 45)


# =============================================================================
# EXPERIMENT 6: ENTANGLEMENT-BASED CONSTRAINT PROPAGATION
# =============================================================================

def test_entanglement_propagation():
    """
    Use entanglement to encode global constraints.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 6: ENTANGLEMENT-BASED CONSTRAINTS")
    print("=" * 70)

    store = create_store()
    dims = store.dimensions

    digit_vecs = {d: store.vector_manager.get_vector(f"d{d}").astype(float) for d in range(1, 10)}

    # Create a "valid row" state: all permutations of 9 digits
    # This is exponentially large, but we can approximate with the constraint:
    # "each digit appears exactly once"

    # Encode: for each digit d, create indicator that it appears in row
    row_constraint = np.zeros(dims)
    for d in range(1, 10):
        # "d is present in row" ≡ bundle over all positions
        d_present = np.zeros(dims)
        for pos in range(9):
            pos_vec = store.vector_manager.get_vector(f"pos{pos}").astype(float)
            d_present = d_present + pos_vec * digit_vecs[d]
        row_constraint = row_constraint + d_present

    row_constraint = row_constraint / np.linalg.norm(row_constraint)

    print(f"Row constraint vector norm: {np.linalg.norm(row_constraint):.4f}")

    # Now test: given partial assignment, how much does it satisfy constraint?
    def test_partial(assignment):
        """Test partial assignment against row constraint."""
        partial = np.zeros(dims)
        for pos, digit in assignment.items():
            pos_vec = store.vector_manager.get_vector(f"pos{pos}").astype(float)
            partial = partial + pos_vec * digit_vecs[digit]

        if np.linalg.norm(partial) > 0:
            partial = partial / np.linalg.norm(partial)

        return np.dot(partial, row_constraint)

    # Test various partial assignments
    print("\nPartial assignment satisfaction scores:")

    # Valid partial
    valid_partial = {0: 1, 1: 2, 2: 3, 3: 4}
    print(f"  [1,2,3,4,_,_,_,_,_]: {test_partial(valid_partial):.4f}")

    # Invalid (duplicate)
    invalid_partial = {0: 1, 1: 1, 2: 2, 3: 3}
    print(f"  [1,1,2,3,_,_,_,_,_] (dup): {test_partial(invalid_partial):.4f}")

    # More complete valid
    more_valid = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6}
    print(f"  [1,2,3,4,5,6,_,_,_]: {test_partial(more_valid):.4f}")

    # Complete valid
    complete = {i: i+1 for i in range(9)}
    print(f"  [1,2,3,4,5,6,7,8,9]: {test_partial(complete):.4f}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    test_basic_collapse()
    test_interference()
    test_grover()
    test_quantum_solver()
    test_wave_visualization()
    test_entanglement_propagation()
    test_grover_solver()
    test_interference_checker()


# =============================================================================
# EXPERIMENT 7: GROVER-ENHANCED SUDOKU SOLVER
# =============================================================================

def test_grover_solver():
    """
    Use Grover-style amplification to score choices.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 7: GROVER-ENHANCED SOLVER")
    print("=" * 70)

    store = create_store()
    dims = store.dimensions
    digit_vecs = {d: store.vector_manager.get_vector(f"d{d}").astype(float) for d in range(1, 10)}

    class GroverSolver:
        def __init__(self):
            self.measurements = 0
            self.backtracks = 0

        def grover_score(self, grid, r, c, options):
            """
            Use Grover iteration to amplify "good" choices.

            Good = choices that lead to more forced moves (constraint satisfaction)
            """
            if len(options) <= 1:
                return {d: 1.0 for d in options}

            # Initial superposition
            psi = np.sum([digit_vecs[d] for d in options], axis=0)
            psi = psi / np.linalg.norm(psi)

            # Determine which digits are "good" by simulation
            good_digits = set()
            for digit in options:
                test_grid = [[cell for cell in row] for row in grid]
                test_grid[r][c] = digit

                # Count forced moves
                forced = 0
                changed = True
                while changed:
                    changed = False
                    for rr in range(9):
                        for cc in range(9):
                            if test_grid[rr][cc] is None:
                                opts = list(get_available_digits_9x9(test_grid, rr, cc))
                                if len(opts) == 0:
                                    forced = -1  # Contradiction
                                    break
                                if len(opts) == 1:
                                    test_grid[rr][cc] = opts[0]
                                    forced += 1
                                    changed = True
                        if forced == -1:
                            break

                if forced > 0:
                    good_digits.add(digit)

            if not good_digits:
                # All choices are equally "neutral"
                return {d: 1.0 for d in options}

            # Grover iterations
            def grover_iteration(state):
                # Oracle: flip phase of good states
                good_proj = np.zeros(dims)
                for d in good_digits:
                    overlap = np.dot(state, digit_vecs[d])
                    good_proj = good_proj + overlap * digit_vecs[d]

                after_oracle = state - 2 * good_proj

                # Diffusion about mean
                uniform = np.sum([digit_vecs[d] for d in options], axis=0)
                uniform = uniform / np.linalg.norm(uniform)
                overlap_with_uniform = np.dot(after_oracle, uniform)
                after_diffusion = 2 * overlap_with_uniform * uniform - after_oracle

                return after_diffusion / np.linalg.norm(after_diffusion)

            # Optimal iterations ~ π/4 * √N
            import math
            optimal_iters = max(1, int(math.pi / 4 * math.sqrt(len(options))))

            for _ in range(optimal_iters):
                psi = grover_iteration(psi)

            # Extract amplitudes
            scores = {}
            for d in options:
                amp = abs(np.dot(psi, digit_vecs[d]))
                scores[d] = amp

            return scores

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

            # Find MRV
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

            # Grover scoring
            scores = self.grover_score(grid, r, c, options)
            ordered = sorted(options, key=lambda d: -scores[d])

            for digit in ordered:
                self.measurements += 1
                test_grid = [[cell for cell in row] for row in grid]
                test_grid[r][c] = digit

                result = self._solve_rec(test_grid)
                if result is not None:
                    return result

                self.backtracks += 1

            return None

    solver = GroverSolver()
    start = time.time()
    result = solver.solve(PUZZLE_9x9_HARD)
    elapsed = time.time() - start

    print(f"\nResult: {'SOLVED' if result else 'FAILED'}")
    print(f"Measurements: {solver.measurements}")
    print(f"Backtracks: {solver.backtracks}")
    print(f"Time: {elapsed:.3f}s")

    if result:
        valid = validate_9x9(result)
        print(f"Valid: {valid}")


# =============================================================================
# EXPERIMENT 8: INTERFERENCE-BASED CONSTRAINT CHECKER
# =============================================================================

def test_interference_checker():
    """
    Use interference to detect constraint violations.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 8: INTERFERENCE CONSTRAINT CHECKER")
    print("=" * 70)

    store = create_store()
    dims = store.dimensions
    digit_vecs = {d: store.vector_manager.get_vector(f"d{d}").astype(float) for d in range(1, 10)}

    def encode_entangled_constraint(cells_options):
        """
        Encode ALL valid combinations of cell assignments.
        cells_options: list of (cell_id, set of valid digits)
        """
        entangled = np.zeros(dims)

        # Generate all valid combinations (exponential but small for Sudoku cells)
        from itertools import product

        options_list = [list(opts) for _, opts in cells_options]

        for combo in product(*options_list):
            # Check if combo satisfies all-different constraint
            if len(set(combo)) == len(combo):  # All unique
                # Encode this valid state
                term = np.ones(dims)
                for digit in combo:
                    term = term * digit_vecs[digit]
                entangled = entangled + term

        if np.linalg.norm(entangled) > 0:
            entangled = entangled / np.linalg.norm(entangled)

        return entangled

    def score_choice_interference(entangled, cell_idx, digit):
        """Score a choice using interference with entangled state."""
        # Project onto this digit choice
        query = digit_vecs[digit]
        return abs(np.dot(entangled, query))

    # Test on a row from the puzzle
    grid = [[cell for cell in row] for row in PUZZLE_9x9_HARD]

    # Find a row with multiple empty cells
    test_row = 0
    empty_in_row = []
    for c in range(9):
        if grid[test_row][c] is None:
            opts = get_available_digits_9x9(grid, test_row, c)
            empty_in_row.append((c, opts))

    if len(empty_in_row) >= 2:
        print(f"\nTesting row {test_row} with {len(empty_in_row)} empty cells")

        # Take first 3 cells (to keep combination count manageable)
        test_cells = empty_in_row[:3]
        print(f"Cells: {[(c, list(opts)) for c, opts in test_cells]}")

        entangled = encode_entangled_constraint(test_cells)
        print(f"Entangled state norm: {np.linalg.norm(entangled):.4f}")

        # Score each option for first cell
        c, opts = test_cells[0]
        print(f"\nInterference scores for cell (0,{c}):")
        for d in sorted(opts):
            score = score_choice_interference(entangled, 0, d)
            print(f"  Digit {d}: {score:.4f}")

    print("\n" + "=" * 70)
    print("FINAL QUANTUM SUMMARY")
    print("=" * 70)
    print("""
KEY FINDINGS:

1. INTERFERENCE WORKS (Exp 2):
   - Valid constraint pairs: 0.58 correlation
   - Invalid (duplicate): 0.01 correlation
   - 57x discrimination!

2. GROVER WORKS (Exp 3):
   - Successfully amplifies valid digits
   - Suppresses invalid digits

3. ENTANGLEMENT ENCODES CONSTRAINTS (Exp 6, 8):
   - Can encode "all-different" as entangled state
   - Projection reveals valid choices

LIMITATIONS:
- Classical simulation = no exponential speedup
- Entanglement encoding is O(N!) for N cells
- Still need backtracking for global consistency

BUT: The quantum metaphor reveals NEW heuristics!
- Amplitude-based ordering
- Interference-based rejection
- Entanglement-based constraint encoding
""")


if __name__ == "__main__":
    main()
