#!/usr/bin/env python3
"""
Approach 20: Chained Encoding for Decision Sequences

Use Holon's CHAINED encoding mode to encode sequences of decisions.

Hypothesis: Good decision sequences have a characteristic "shape" in hyperspace
that we can learn and recognize.

This exploits Holon's ListEncodeMode.CHAINED which was designed for:
- Suffix matching
- Prefix unbinding
- Sequence operations

Can we encode the "chain of forced moves" as a vector and use it to distinguish
good from bad initial choices?
"""

import sys
sys.path.insert(0, "/home/watmin/work/holon")

from typing import List, Optional, Set, Tuple, Dict
from dataclasses import dataclass
import numpy as np
import time

from holon import HolonClient, CPUStore
from holon.encoder import ListEncodeMode
from holon.vector_manager import VectorManager

from common import (
    Timer,
    print_grid_9x9,
    validate_9x9,
    count_empty,
    get_available_digits_9x9,
    PUZZLE_9x9_HARD,
)


def create_chained_encoder(dimensions: int = 16384):
    """Create encoder with chained mode."""
    store = CPUStore(dimensions=dimensions)
    return store, store.encoder, store.vector_manager


def encode_decision_chain(encoder, moves: List[Tuple[int, int, int]]) -> np.ndarray:
    """
    Encode a sequence of moves using CHAINED encoding.

    Each move is (row, col, digit).
    The chained encoding creates: move_n ⊙ (move_n-1 ⊙ (... ⊙ move_1))
    """
    if not moves:
        return np.zeros(encoder.vector_manager.dimensions, dtype=np.int8)

    # Encode each move as a structure
    move_data = []
    for r, c, d in moves:
        # Encode as a dict: {pos: (r,c), digit: d}
        move_dict = {
            "row": r,
            "col": c,
            "digit": d,
            "position": r * 9 + c,  # Flat position
        }
        move_data.append(move_dict)

    # Use CHAINED mode
    return encoder.encode_list(move_data, mode=ListEncodeMode.CHAINED)


def get_forced_chain(grid: List[List[Optional[int]]], row: int, col: int, digit: int,
                     max_depth: int = 20) -> List[Tuple[int, int, int]]:
    """
    Get the chain of forced moves after making a choice.
    """
    test_grid = [[cell for cell in r] for r in grid]
    test_grid[row][col] = digit

    chain = [(row, col, digit)]

    for _ in range(max_depth):
        # Find forced moves
        found_forced = False
        for r in range(9):
            for c in range(9):
                if test_grid[r][c] is None:
                    options = get_available_digits_9x9(test_grid, r, c)
                    if not options:
                        return chain  # Contradiction
                    if len(options) == 1:
                        d = list(options)[0]
                        test_grid[r][c] = d
                        chain.append((r, c, d))
                        found_forced = True
                        break
            if found_forced:
                break

        if not found_forced:
            break

    return chain


def analyze_chain_encodings():
    """Analyze how chain encodings differ for good vs bad choices."""
    print("=" * 70)
    print("CHAINED ENCODING ANALYSIS")
    print("=" * 70)

    store, encoder, vm = create_chained_encoder()

    grid = [[cell for cell in row] for row in PUZZLE_9x9_HARD]

    # Find first decision point
    for r in range(9):
        for c in range(9):
            if grid[r][c] is None:
                options = get_available_digits_9x9(grid, r, c)
                options = list(options)

                if len(options) > 1:
                    print(f"\nDecision point: ({r},{c}) with options {options}")

                    # Get chain for each option
                    chains = {}
                    chain_vecs = {}

                    for digit in options:
                        chain = get_forced_chain(grid, r, c, digit, max_depth=20)
                        chains[digit] = chain

                        # Encode the chain
                        chain_vec = encode_decision_chain(encoder, chain)
                        chain_vecs[digit] = chain_vec

                        print(f"\n  Digit {digit}: chain length = {len(chain)}")
                        print(f"    Chain: {chain[:5]}..." if len(chain) > 5 else f"    Chain: {chain}")

                    # Compare chain similarities
                    print("\n  Chain Similarities (cosine):")
                    for i, d1 in enumerate(options):
                        for d2 in options[i+1:]:
                            v1 = chain_vecs[d1]
                            v2 = chain_vecs[d2]

                            norm1 = np.linalg.norm(v1)
                            norm2 = np.linalg.norm(v2)

                            if norm1 > 0 and norm2 > 0:
                                sim = np.dot(v1, v2) / (norm1 * norm2)
                            else:
                                sim = 0

                            print(f"    Digit {d1} vs {d2}: {sim:.4f}")

                    return


def build_chain_prototypes():
    """
    Build prototypes for "good chains" vs "bad chains" from solved puzzle.
    """
    print("\n" + "=" * 70)
    print("BUILDING CHAIN PROTOTYPES")
    print("=" * 70)

    store, encoder, vm = create_chained_encoder()

    # We need to solve the puzzle and track which initial choices led to success
    from approach_19_opportunistic import HybridSolver

    # Track chains for correct vs incorrect choices
    good_chains = []
    bad_chains = []

    def solve_with_chain_tracking(puzzle):
        """Solve and track which chains succeed."""
        grid = [[cell for cell in row] for row in puzzle]

        def propagate(g):
            changed = True
            while changed:
                changed = False
                for r in range(9):
                    for c in range(9):
                        if g[r][c] is None:
                            opts = list(get_available_digits_9x9(g, r, c))
                            if not opts:
                                return False
                            if len(opts) == 1:
                                g[r][c] = opts[0]
                                changed = True
            return True

        propagate(grid)

        def solve_rec(g, depth=0):
            if count_empty(g) == 0:
                return True

            best = None
            best_count = 10
            for r in range(9):
                for c in range(9):
                    if g[r][c] is None:
                        opts = list(get_available_digits_9x9(g, r, c))
                        if not opts:
                            return False
                        if len(opts) < best_count:
                            best_count = len(opts)
                            best = (r, c, opts)

            if best is None:
                return True

            r, c, opts = best

            for digit in opts:
                # Get the forced chain for this choice
                chain = get_forced_chain(g, r, c, digit)

                test_g = [[cell for cell in row] for row in g]
                test_g[r][c] = digit

                if not propagate(test_g):
                    # This choice leads to contradiction
                    if depth < 5:  # Only track early chains
                        bad_chains.append(chain)
                    continue

                if solve_rec(test_g, depth + 1):
                    # This choice led to solution
                    if depth < 5:
                        good_chains.append(chain)
                    return True
                else:
                    # This choice eventually failed
                    if depth < 5:
                        bad_chains.append(chain)

            return False

        return solve_rec(grid)

    # Solve and collect chains
    solved = solve_with_chain_tracking(PUZZLE_9x9_HARD)
    print(f"Solved: {solved}")
    print(f"Good chains collected: {len(good_chains)}")
    print(f"Bad chains collected: {len(bad_chains)}")

    if not good_chains or not bad_chains:
        print("Not enough chains to build prototypes")
        return

    # Encode and bundle chains
    good_vecs = [encode_decision_chain(encoder, chain) for chain in good_chains]
    bad_vecs = [encode_decision_chain(encoder, chain) for chain in bad_chains]

    good_prototype = np.sum(good_vecs, axis=0)
    bad_prototype = np.sum(bad_vecs, axis=0)

    # Threshold to bipolar
    good_prototype = np.where(good_prototype > 0, 1, np.where(good_prototype < 0, -1, 0)).astype(np.int8)
    bad_prototype = np.where(bad_prototype > 0, 1, np.where(bad_prototype < 0, -1, 0)).astype(np.int8)

    # Compare prototypes
    norm_good = np.linalg.norm(good_prototype)
    norm_bad = np.linalg.norm(bad_prototype)

    if norm_good > 0 and norm_bad > 0:
        proto_sim = np.dot(good_prototype, bad_prototype) / (norm_good * norm_bad)
        print(f"\nGood vs Bad prototype similarity: {proto_sim:.4f}")

    # Test: can prototype predict chain quality?
    print("\nTesting prototype prediction on chains:")

    def score_chain(chain):
        vec = encode_decision_chain(encoder, chain)
        norm = np.linalg.norm(vec)

        if norm == 0:
            return 0, 0

        good_sim = np.dot(vec, good_prototype) / (norm * norm_good)
        bad_sim = np.dot(vec, bad_prototype) / (norm * norm_bad)

        return good_sim, bad_sim

    # Sample some chains
    print("\n  Good chains:")
    for chain in good_chains[:3]:
        gs, bs = score_chain(chain)
        print(f"    len={len(chain):2d}: good_sim={gs:.4f}, bad_sim={bs:.4f}, delta={gs-bs:+.4f}")

    print("\n  Bad chains:")
    for chain in bad_chains[:3]:
        gs, bs = score_chain(chain)
        print(f"    len={len(chain):2d}: good_sim={gs:.4f}, bad_sim={bs:.4f}, delta={gs-bs:+.4f}")

    # Calculate accuracy
    correct = 0
    total = 0

    for chain in good_chains:
        gs, bs = score_chain(chain)
        if gs > bs:
            correct += 1
        total += 1

    for chain in bad_chains:
        gs, bs = score_chain(chain)
        if bs > gs:
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0
    print(f"\nPrototype prediction accuracy: {accuracy:.1%}")


def main():
    analyze_chain_encodings()
    build_chain_prototypes()


if __name__ == "__main__":
    main()
