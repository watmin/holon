#!/usr/bin/env python3
"""
Approach 28: Revisiting Previous Approaches with New Primitives

New primitives to apply:
- Negate:    Remove component influence
- Amplify:   Boost component influence
- Prototype: Extract common pattern
- Difference: Track what changed
- Blend:     Interpolate between states
- Resonance: Extract matching part

Most promising revisits:
1. Approach 22 (Template) + Amplify/Negate
2. Approach 15 (Higher-Order) + Amplify (bootstrap problem)
3. Approach 16 (Tree) + Prototype (learn from paths)
4. Approach 10 (Simulation) + Difference
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
# REVISIT 1: Template Matching + Amplify/Negate
# =============================================================================

def test_template_with_primitives():
    """
    Enhance template matching with amplify and negate.

    Idea:
    - Amplify digits that are "almost complete" in a constraint
    - Negate digits that would create duplicates more strongly
    """
    print("=" * 70)
    print("REVISIT 1: Template Matching + Amplify/Negate")
    print("=" * 70)

    store = create_store()
    digits = {d: store.vector_manager.get_vector(f"d{d}") for d in range(1, 10)}

    # Complete template
    complete = store.bundle([digits[d] for d in range(1, 10)])

    class EnhancedTemplateSolver:
        def __init__(self):
            self.backtracks = 0

        def score_choice(self, grid, r, c, digit):
            # Get constraint units
            row_digits = {grid[r][cc] for cc in range(9) if grid[r][cc] is not None}
            col_digits = {grid[rr][c] for rr in range(9) if grid[rr][c] is not None}
            br, bc = (r // 3) * 3, (c // 3) * 3
            block_digits = {grid[rr][cc] for rr in range(br, br+3)
                            for cc in range(bc, bc+3) if grid[rr][cc] is not None}

            total_score = 0

            for used_digits in [row_digits, col_digits, block_digits]:
                new_set = used_digits | {digit}
                set_vec = store.bundle([digits[d] for d in new_set])

                # Base template similarity
                template_score = similarity(set_vec, complete)

                # ENHANCEMENT: Amplify if close to complete (7+ digits)
                if len(new_set) >= 7:
                    # Amplify the new digit's contribution
                    amplified = store.amplify(set_vec, digits[digit], strength=1.5)
                    template_score = similarity(amplified, complete)

                # ENHANCEMENT: Penalize if would create near-duplicate pattern
                if digit in used_digits:
                    # This shouldn't happen (constraint check), but extra penalty
                    template_score -= 1.0

                total_score += template_score

            return total_score

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

    solver = EnhancedTemplateSolver()
    start = time.time()
    result = solver.solve(PUZZLE_9x9_HARD)
    elapsed = time.time() - start

    print(f"\nResult: {'SOLVED' if result else 'FAILED'}")
    print(f"Backtracks: {solver.backtracks}")
    print(f"Time: {elapsed:.3f}s")
    print(f"Comparison: Original template = 52 backtracks")

    return solver.backtracks


# =============================================================================
# REVISIT 2: Higher-Order Binding + Amplify (Bootstrap Problem)
# =============================================================================

def test_higher_order_amplify():
    """
    The bootstrap problem: signal is weak at start, strong later.

    Idea: Use AMPLIFY to boost weak early signals.
    """
    print("\n" + "=" * 70)
    print("REVISIT 2: Higher-Order Binding + Amplify")
    print("=" * 70)

    store = create_store()
    digits = {d: store.vector_manager.get_vector(f"d{d}") for d in range(1, 10)}

    # Build "good choice" and "bad choice" prototypes from simulation
    good_patterns = []
    bad_patterns = []

    def encode_choice_context(grid, r, c, digit):
        """Encode the context around a choice."""
        # Get constraint satisfaction levels
        row_filled = sum(1 for cc in range(9) if grid[r][cc] is not None)
        col_filled = sum(1 for rr in range(9) if grid[rr][c] is not None)
        br, bc = (r // 3) * 3, (c // 3) * 3
        block_filled = sum(1 for rr in range(br, br+3)
                           for cc in range(bc, bc+3) if grid[rr][cc] is not None)

        # Encode context
        context_vecs = [
            store.vector_manager.get_vector(f"row_fill_{row_filled}"),
            store.vector_manager.get_vector(f"col_fill_{col_filled}"),
            store.vector_manager.get_vector(f"block_fill_{block_filled}"),
            digits[digit],
        ]
        return store.bundle(context_vecs)

    def simulate_choice(grid, r, c, digit):
        """Simulate a choice and return if it leads to contradiction."""
        test_grid = [[cell for cell in row] for row in grid]
        test_grid[r][c] = digit

        # Propagate
        changed = True
        while changed:
            changed = False
            for rr in range(9):
                for cc in range(9):
                    if test_grid[rr][cc] is None:
                        opts = list(get_available_digits_9x9(test_grid, rr, cc))
                        if not opts:
                            return False, 0  # Contradiction
                        if len(opts) == 1:
                            test_grid[rr][cc] = opts[0]
                            changed = True

        filled = sum(1 for rr in range(9) for cc in range(9) if test_grid[rr][cc] is not None)
        return True, filled

    # Collect patterns from the puzzle
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

    # Find decision points and classify
    for r in range(9):
        for c in range(9):
            if grid[r][c] is None:
                opts = list(get_available_digits_9x9(grid, r, c))
                for digit in opts:
                    context = encode_choice_context(grid, r, c, digit)
                    valid, filled = simulate_choice(grid, r, c, digit)

                    if not valid:
                        bad_patterns.append(context)
                    elif filled > count_empty(grid) - 5:
                        good_patterns.append(context)

    print(f"\nCollected patterns: {len(good_patterns)} good, {len(bad_patterns)} bad")

    if good_patterns and bad_patterns:
        # Build prototypes
        good_proto = store.prototype(good_patterns, threshold=0.3)
        bad_proto = store.prototype(bad_patterns, threshold=0.3)

        # AMPLIFY good, NEGATE bad
        combined = store.amplify(good_proto, good_proto, strength=1.0)
        combined = store.negate(combined, bad_proto)

        print(f"\nPrototype built. Testing discrimination...")

        # Test on new choices
        correct = 0
        total = 0
        for r in range(9):
            for c in range(9):
                if grid[r][c] is None:
                    opts = list(get_available_digits_9x9(grid, r, c))
                    if len(opts) > 1:
                        best_score = -999
                        best_digit = None
                        for digit in opts:
                            context = encode_choice_context(grid, r, c, digit)
                            # Amplify context before comparing
                            amplified = store.amplify(context, digits[digit], strength=0.5)
                            score = similarity(amplified, combined)
                            if score > best_score:
                                best_score = score
                                best_digit = digit

                        # Check if best choice is valid
                        valid, _ = simulate_choice(grid, r, c, best_digit)
                        if valid:
                            correct += 1
                        total += 1

        if total > 0:
            print(f"Accuracy: {correct}/{total} = {100*correct/total:.1f}%")


# =============================================================================
# REVISIT 3: Path Encoding + Prototype/Difference
# =============================================================================

def test_path_prototype():
    """
    Learn from good vs bad paths using prototype and difference.

    Idea:
    - Collect encoding of paths that lead to solutions
    - Collect encoding of paths that lead to contradictions
    - Use PROTOTYPE to find common patterns
    - Use DIFFERENCE to distinguish good from bad
    """
    print("\n" + "=" * 70)
    print("REVISIT 3: Path Encoding + Prototype/Difference")
    print("=" * 70)

    store = create_store()

    def encode_path(moves: List[Tuple[int, int, int]]) -> np.ndarray:
        """Encode a sequence of moves."""
        if not moves:
            return np.zeros(store.dimensions, dtype=np.int8)

        vecs = []
        for i, (r, c, d) in enumerate(moves):
            pos = store.vector_manager.get_vector(f"pos_{r}_{c}")
            digit = store.vector_manager.get_vector(f"digit_{d}")
            step = store.vector_manager.get_vector(f"step_{i}")
            vecs.append(store.bind(step, store.bind(pos, digit)))

        return store.bundle(vecs)

    # Solve with path tracking
    good_paths = []
    bad_paths = []

    def solve_and_track(puzzle):
        grid = [[cell for cell in row] for row in puzzle]

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

        def solve_rec(g, path):
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
                                path.append((r, c, opts[0]))
                                changed = True

            if count_empty(g) == 0:
                return g, path

            # Find MRV
            best = None
            best_count = 10
            for r in range(9):
                for c in range(9):
                    if g[r][c] is None:
                        opts = list(get_available_digits_9x9(g, r, c))
                        if len(opts) < best_count:
                            best_count = len(opts)
                            best = (r, c, opts)

            if best is None:
                return None

            r, c, options = best

            for digit in options:
                test_g = [[cell for cell in row] for row in g]
                test_g[r][c] = digit
                new_path = path + [(r, c, digit)]

                result = solve_rec(test_g, new_path)
                if result is not None:
                    good_paths.append(new_path)
                    return result
                else:
                    bad_paths.append(new_path)

            return None

        return solve_rec(grid, [])

    result = solve_and_track(PUZZLE_9x9_HARD)
    print(f"\nSolved: {result is not None}")
    print(f"Good paths: {len(good_paths)}, Bad paths: {len(bad_paths)}")

    if good_paths and bad_paths:
        # Encode paths
        good_encodings = [encode_path(p[:10]) for p in good_paths[:50]]  # Limit
        bad_encodings = [encode_path(p[:10]) for p in bad_paths[:50]]

        # Build prototypes
        good_proto = store.prototype(good_encodings, threshold=0.3)
        bad_proto = store.prototype(bad_encodings, threshold=0.3)

        # Compute difference
        diff = store.difference(bad_proto, good_proto)

        print(f"\nPrototypes built.")
        print(f"sim(diff, good_proto) = {similarity(diff, good_proto):.4f}")
        print(f"sim(diff, bad_proto)  = {similarity(diff, bad_proto):.4f}")


# =============================================================================
# REVISIT 4: Simulation + Difference for Delta Scoring
# =============================================================================

def test_simulation_difference():
    """
    Use DIFFERENCE to score choices by how much they change the state.

    Idea: Choices that cause bigger "difference" vectors might be more decisive.
    """
    print("\n" + "=" * 70)
    print("REVISIT 4: Simulation + Difference Scoring")
    print("=" * 70)

    store = create_store()
    digits = {d: store.vector_manager.get_vector(f"d{d}") for d in range(1, 10)}

    def encode_grid(g):
        """Encode full grid state."""
        vecs = []
        for r in range(9):
            for c in range(9):
                if g[r][c] is not None:
                    pos = store.vector_manager.get_vector(f"p{r}{c}")
                    vecs.append(store.bind(pos, digits[g[r][c]]))
        return store.bundle(vecs) if vecs else np.zeros(store.dimensions, dtype=np.int8)

    class DifferenceSolver:
        def __init__(self):
            self.backtracks = 0

        def score_by_difference(self, grid, r, c, digit):
            """Score by magnitude of change caused."""
            before = encode_grid(grid)

            # Simulate
            test_grid = [[cell for cell in row] for row in grid]
            test_grid[r][c] = digit

            # Propagate
            changed = True
            forced = 0
            while changed:
                changed = False
                for rr in range(9):
                    for cc in range(9):
                        if test_grid[rr][cc] is None:
                            opts = list(get_available_digits_9x9(test_grid, rr, cc))
                            if not opts:
                                return -1000, True  # Contradiction
                            if len(opts) == 1:
                                test_grid[rr][cc] = opts[0]
                                forced += 1
                                changed = True

            after = encode_grid(test_grid)
            diff = store.difference(before, after)

            # Score = magnitude of difference (more change = more decisive)
            magnitude = np.linalg.norm(diff)
            return magnitude + forced * 10, False  # Bonus for forced moves

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

            # Score by difference
            scores = []
            for d in options:
                score, contradicts = self.score_by_difference(grid, r, c, d)
                if not contradicts:
                    scores.append((score, d))

            scores.sort(reverse=True)
            ordered = [d for _, d in scores] if scores else options

            for digit in ordered:
                test_grid = [[cell for cell in row] for row in grid]
                test_grid[r][c] = digit

                result = self._solve_rec(test_grid)
                if result is not None:
                    return result

                self.backtracks += 1

            return None

    solver = DifferenceSolver()
    start = time.time()
    result = solver.solve(PUZZLE_9x9_HARD)
    elapsed = time.time() - start

    print(f"\nResult: {'SOLVED' if result else 'FAILED'}")
    print(f"Backtracks: {solver.backtracks}")
    print(f"Time: {elapsed:.3f}s")
    print(f"Comparison: Original template = 52, simulation = 249")

    return solver.backtracks


# =============================================================================
# REVISIT 5: Resonance for Noise Reduction
# =============================================================================

def test_resonance_solver():
    """
    Use RESONANCE to extract the "correct" signal from noisy encodings.

    Idea: Compare current state to solved puzzle patterns.
    """
    print("\n" + "=" * 70)
    print("REVISIT 5: Resonance for Signal Extraction")
    print("=" * 70)

    store = create_store()
    digits = {d: store.vector_manager.get_vector(f"d{d}") for d in range(1, 10)}

    # Complete template as reference
    complete = store.bundle([digits[d] for d in range(1, 10)])

    class ResonanceSolver:
        def __init__(self):
            self.backtracks = 0

        def score_by_resonance(self, grid, r, c, digit):
            """Score by how well choice resonates with complete patterns."""
            row_digits = {grid[r][cc] for cc in range(9) if grid[r][cc] is not None}
            col_digits = {grid[rr][c] for rr in range(9) if grid[rr][c] is not None}
            br, bc = (r // 3) * 3, (c // 3) * 3
            block_digits = {grid[rr][cc] for rr in range(br, br+3)
                            for cc in range(bc, bc+3) if grid[rr][cc] is not None}

            total_score = 0

            for used_digits in [row_digits, col_digits, block_digits]:
                new_set = used_digits | {digit}
                set_vec = store.bundle([digits[d] for d in new_set])

                # Use resonance to extract matching part with complete template
                resonant = store.resonance(set_vec, complete)

                # Score = how much of the set resonates with complete
                score = np.linalg.norm(resonant) / np.linalg.norm(set_vec) if np.linalg.norm(set_vec) > 0 else 0
                total_score += score

            return total_score

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

            scores = [(self.score_by_resonance(grid, r, c, d), d) for d in options]
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

    solver = ResonanceSolver()
    start = time.time()
    result = solver.solve(PUZZLE_9x9_HARD)
    elapsed = time.time() - start

    print(f"\nResult: {'SOLVED' if result else 'FAILED'}")
    print(f"Backtracks: {solver.backtracks}")
    print(f"Time: {elapsed:.3f}s")
    print(f"Comparison: Original template = 52")

    return solver.backtracks


# =============================================================================
# MAIN
# =============================================================================

def main():
    results = {}

    results['template_amplify'] = test_template_with_primitives()
    test_higher_order_amplify()
    test_path_prototype()
    results['difference'] = test_simulation_difference()
    results['resonance'] = test_resonance_solver()

    print("\n" + "=" * 70)
    print("SUMMARY: Revisited Approaches with New Primitives")
    print("=" * 70)
    print(f"""
| Approach | Original | With New Primitives |
|----------|----------|---------------------|
| Template Matching | 52 | {results.get('template_amplify', 'N/A')} |
| Simulation-Guided | 249 | {results.get('difference', 'N/A')} |
| Resonance | N/A | {results.get('resonance', 'N/A')} |
""")


if __name__ == "__main__":
    main()
