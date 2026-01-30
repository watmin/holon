#!/usr/bin/env python3
"""
Approach 16: Search Tree Encoding

THE RADICAL IDEA:
Encode the SEARCH TREE itself, not just the puzzle state.

The tree contains:
- All possible paths from puzzle to solution/contradiction
- Which choices lead where
- The structure of the solution space

QUESTIONS:
1. Can we encode the tree efficiently?
2. Does the encoding reveal patterns?
3. Can patterns transfer across puzzles?
4. Can we compress/shortcut the tree?
"""

import sys
sys.path.insert(0, "/home/watmin/work/holon")

from typing import List, Optional, Set, Tuple, Dict, Any
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field

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


@dataclass
class TreeNode:
    """A node in the search tree."""
    row: int
    col: int
    digit: int
    outcome: str  # 'solution', 'contradiction', 'branch', 'stuck'
    children: List['TreeNode'] = field(default_factory=list)
    depth: int = 0
    path_length: int = 0  # Steps from here to leaf


class TreeBuilder:
    """Build and analyze search trees for Sudoku."""

    def __init__(self, max_depth: int = 10, max_nodes: int = 1000):
        self.max_depth = max_depth
        self.max_nodes = max_nodes
        self.nodes_created = 0

    def copy_grid(self, grid):
        return [[cell for cell in row] for row in grid]

    def propagate(self, grid: List[List[Optional[int]]]) -> Tuple[bool, int]:
        """
        Apply forced moves until stable.
        Returns (success, forced_count)
        """
        forced = 0
        changed = True
        while changed:
            changed = False
            for r in range(9):
                for c in range(9):
                    if grid[r][c] is None:
                        available = get_available_digits_9x9(grid, r, c)
                        if len(available) == 0:
                            return False, forced  # Contradiction
                        if len(available) == 1:
                            grid[r][c] = list(available)[0]
                            forced += 1
                            changed = True
        return True, forced

    def build_tree(self, puzzle: List[List[Optional[int]]],
                   depth: int = 0) -> Optional[TreeNode]:
        """
        Build search tree from current state.
        Returns root node of subtree.
        """
        if self.nodes_created >= self.max_nodes or depth >= self.max_depth:
            return None

        grid = self.copy_grid(puzzle)

        # Propagate forced moves
        success, forced = self.propagate(grid)
        if not success:
            return TreeNode(-1, -1, -1, 'contradiction', depth=depth)

        # Check if solved
        empty = count_empty(grid)
        if empty == 0:
            valid, _ = validate_9x9(grid)
            if valid:
                return TreeNode(-1, -1, -1, 'solution', depth=depth)
            else:
                return TreeNode(-1, -1, -1, 'contradiction', depth=depth)

        # Find MRV cell
        best_cell = None
        min_opts = 10
        for r in range(9):
            for c in range(9):
                if grid[r][c] is None:
                    opts = get_available_digits_9x9(grid, r, c)
                    if len(opts) < min_opts:
                        min_opts = len(opts)
                        best_cell = (r, c, list(opts))

        if best_cell is None or min_opts == 0:
            return TreeNode(-1, -1, -1, 'stuck', depth=depth)

        r, c, options = best_cell

        # Create branch node
        node = TreeNode(r, c, 0, 'branch', depth=depth)
        self.nodes_created += 1

        # Explore each option
        for digit in options:
            test_grid = self.copy_grid(grid)
            test_grid[r][c] = digit

            child = self.build_tree(test_grid, depth + 1)
            if child is not None:
                child.digit = digit
                child.row = r
                child.col = c
                node.children.append(child)

        return node


class TreeEncoder:
    """Encode search trees in hyperspace."""

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

        # Outcome vectors
        self.SOLUTION = self._random_vector("SOLUTION")
        self.CONTRADICTION = self._random_vector("CONTRADICTION")
        self.BRANCH = self._random_vector("BRANCH")
        self.STUCK = self._random_vector("STUCK")

    def _random_vector(self, seed: str) -> np.ndarray:
        np.random.seed(hash(seed) % (2**32))
        return np.random.choice([-1.0, 1.0], size=self.dimensions)

    def encode_node(self, node: TreeNode) -> np.ndarray:
        """Encode a single node."""
        if node.row < 0:
            # Leaf node - just outcome
            if node.outcome == 'solution':
                return self.SOLUTION.copy()
            elif node.outcome == 'contradiction':
                return self.CONTRADICTION.copy()
            else:
                return self.STUCK.copy()

        # Choice node: bind position, digit, outcome type
        pos_vec = self.pos_vectors[(node.row, node.col)]
        if node.digit > 0:
            dig_vec = self.digit_vectors[node.digit]
            choice_vec = bind(pos_vec, dig_vec)
        else:
            choice_vec = pos_vec

        if node.outcome == 'solution':
            return bind(choice_vec, self.SOLUTION)
        elif node.outcome == 'contradiction':
            return bind(choice_vec, self.CONTRADICTION)
        else:
            return bind(choice_vec, self.BRANCH)

    def encode_tree_recursive(self, node: TreeNode) -> np.ndarray:
        """Encode entire tree recursively."""
        if node is None:
            return np.zeros(self.dimensions)

        node_vec = self.encode_node(node)

        if not node.children:
            return node_vec

        # Bundle children
        child_vecs = [self.encode_tree_recursive(child) for child in node.children]
        children_vec = bundle(child_vecs)

        # Bind node with children
        return bind(node_vec, children_vec)

    def encode_paths(self, node: TreeNode, prefix: List[Tuple[int, int, int]] = None
                     ) -> List[Tuple[List[Tuple[int, int, int]], str, np.ndarray]]:
        """
        Encode all paths from node to leaves.
        Returns list of (path, outcome, path_vec)
        """
        if prefix is None:
            prefix = []

        if node is None:
            return []

        current_path = prefix + [(node.row, node.col, node.digit)]

        if not node.children:
            # Leaf - encode full path
            path_vec = self._encode_path(current_path, node.outcome)
            return [(current_path, node.outcome, path_vec)]

        # Recurse
        all_paths = []
        for child in node.children:
            all_paths.extend(self.encode_paths(child, current_path))

        return all_paths

    def _encode_path(self, path: List[Tuple[int, int, int]], outcome: str) -> np.ndarray:
        """Encode a single path."""
        # Sequential binding of choices
        vec = np.ones(self.dimensions)
        for (r, c, d) in path:
            if r >= 0 and d > 0:
                choice = bind(self.pos_vectors[(r, c)], self.digit_vectors[d])
                vec = bind(vec, choice)

        # Bind outcome
        if outcome == 'solution':
            vec = bind(vec, self.SOLUTION)
        elif outcome == 'contradiction':
            vec = bind(vec, self.CONTRADICTION)

        return vec


def analyze_tree_structure():
    """Analyze the structure of the search tree."""
    print("=" * 70)
    print("ANALYZING SEARCH TREE STRUCTURE")
    print("=" * 70)

    builder = TreeBuilder(max_depth=8, max_nodes=500)
    encoder = TreeEncoder(dimensions=16384)

    print("\nBuilding search tree (limited to 500 nodes, depth 8)...")
    with Timer() as timer:
        tree = builder.build_tree(PUZZLE_9x9_HARD)

    print(f"Nodes created: {builder.nodes_created}")
    print(f"Time: {timer.elapsed:.2f}s")

    # Count outcomes
    def count_outcomes(node, counts=None):
        if counts is None:
            counts = {'solution': 0, 'contradiction': 0, 'branch': 0, 'stuck': 0}
        if node is None:
            return counts
        counts[node.outcome] += 1
        for child in node.children:
            count_outcomes(child, counts)
        return counts

    outcomes = count_outcomes(tree)
    print(f"\nTree composition:")
    for outcome, count in outcomes.items():
        print(f"  {outcome}: {count}")

    # Extract and analyze paths
    print("\n" + "-" * 50)
    print("Encoding paths...")

    paths = encoder.encode_paths(tree)
    print(f"Total paths extracted: {len(paths)}")

    solution_paths = [p for p in paths if p[1] == 'solution']
    contradiction_paths = [p for p in paths if p[1] == 'contradiction']
    print(f"  Solution paths: {len(solution_paths)}")
    print(f"  Contradiction paths: {len(contradiction_paths)}")

    if solution_paths and contradiction_paths:
        # Compare solution vs contradiction path encodings
        print("\n" + "-" * 50)
        print("Comparing path encodings...")

        sol_vecs = [p[2] for p in solution_paths[:10]]
        con_vecs = [p[2] for p in contradiction_paths[:10]]

        # Similarity within solution paths
        sol_sims = []
        for i, v1 in enumerate(sol_vecs):
            for v2 in sol_vecs[i+1:]:
                sol_sims.append(similarity(v1, v2))

        # Similarity within contradiction paths
        con_sims = []
        for i, v1 in enumerate(con_vecs):
            for v2 in con_vecs[i+1:]:
                con_sims.append(similarity(v1, v2))

        # Cross similarity
        cross_sims = []
        for v1 in sol_vecs[:5]:
            for v2 in con_vecs[:5]:
                cross_sims.append(similarity(v1, v2))

        print(f"  Similarity within solution paths: {np.mean(sol_sims):.4f}" if sol_sims else "  No solution paths")
        print(f"  Similarity within contradiction paths: {np.mean(con_sims):.4f}" if con_sims else "  No contradiction paths")
        print(f"  Similarity across (sol vs con): {np.mean(cross_sims):.4f}" if cross_sims else "  No cross comparison")

        # Key test: can we distinguish solution paths from contradiction paths?
        print("\n" + "-" * 50)
        print("KEY TEST: Can we identify solution paths geometrically?")

        if sol_vecs and con_vecs:
            # Create "prototype" vectors
            sol_prototype = bundle(sol_vecs)
            con_prototype = bundle(con_vecs)

            print(f"\nSimilarity of solution prototype to:")
            for i, (path, outcome, vec) in enumerate(paths[:10]):
                sim_to_sol = similarity(vec, sol_prototype)
                sim_to_con = similarity(vec, con_prototype)
                pred = "SOL" if sim_to_sol > sim_to_con else "CON"
                correct = "✓" if (pred == "SOL" and outcome == "solution") or (pred == "CON" and outcome == "contradiction") else "✗"
                print(f"  Path {i}: actual={outcome[:3]}, pred={pred}, {correct}")


def explore_tree_compression():
    """Explore compression opportunities in the tree."""
    print("\n" + "=" * 70)
    print("EXPLORING TREE COMPRESSION")
    print("=" * 70)

    builder = TreeBuilder(max_depth=6, max_nodes=200)

    print("\nBuilding tree...")
    tree = builder.build_tree(PUZZLE_9x9_HARD)

    # Find repeated subtrees (states that appear multiple times)
    # This requires hashing grid states

    def collect_states(node, grid, states=None):
        if states is None:
            states = defaultdict(list)
        if node is None or node.row < 0:
            return states

        # Hash the current decision point
        state_key = (node.row, node.col, tuple(sorted(
            [(r, c, grid[r][c]) for r in range(9) for c in range(9) if grid[r][c] is not None]
        )))

        states[state_key].append(node)

        for child in node.children:
            if child.row >= 0 and child.digit > 0:
                new_grid = [[cell for cell in row] for row in grid]
                new_grid[child.row][child.col] = child.digit
                collect_states(child, new_grid, states)

        return states

    states = collect_states(tree, PUZZLE_9x9_HARD)
    repeated = {k: v for k, v in states.items() if len(v) > 1}

    print(f"Unique decision points: {len(states)}")
    print(f"Repeated states (compression opportunity): {len(repeated)}")

    if repeated:
        print("\nExample repeated states:")
        for key, nodes in list(repeated.items())[:3]:
            print(f"  State at ({key[0]},{key[1]}): appears {len(nodes)} times")


def test_transfer():
    """Test if tree patterns transfer across puzzles."""
    print("\n" + "=" * 70)
    print("TESTING PATTERN TRANSFER")
    print("=" * 70)

    # A slightly different puzzle (same difficulty level)
    puzzle2 = [
        [None, None, None, None, None, 2, None, None, None],
        [None, 8, None, None, None, None, None, None, None],
        [None, None, None, 6, None, None, None, None, 4],
        [None, None, 8, None, None, None, 1, None, None],
        [None, 4, None, None, None, None, None, None, None],
        [None, None, 3, None, None, None, None, 7, 6],
        [None, None, None, None, None, None, 2, None, None],
        [None, 9, None, None, None, 7, None, None, None],
        [3, None, 2, None, None, None, None, 8, None],
    ]

    encoder = TreeEncoder(dimensions=16384)
    builder1 = TreeBuilder(max_depth=5, max_nodes=100)
    builder2 = TreeBuilder(max_depth=5, max_nodes=100)

    print("\nBuilding trees for two different puzzles...")

    tree1 = builder1.build_tree(PUZZLE_9x9_HARD)
    tree2 = builder2.build_tree(puzzle2)

    # Encode both trees
    tree1_vec = encoder.encode_tree_recursive(tree1)
    tree2_vec = encoder.encode_tree_recursive(tree2)

    sim = similarity(tree1_vec, tree2_vec)
    print(f"\nSimilarity between tree encodings: {sim:.4f}")

    if sim > 0.1:
        print("  ✓ Trees share some structural similarity!")
    else:
        print("  ✗ Trees are structurally different")

    # Compare path patterns
    paths1 = encoder.encode_paths(tree1)
    paths2 = encoder.encode_paths(tree2)

    print(f"\nPaths in tree1: {len(paths1)}")
    print(f"Paths in tree2: {len(paths2)}")


def build_full_tree_with_backtracking():
    """
    Build tree using actual backtracking to find solutions.
    Record the structure as we go.
    """
    print("\n" + "=" * 70)
    print("BUILDING FULL TREE (with solutions)")
    print("=" * 70)

    encoder = TreeEncoder(dimensions=16384)

    # Track all decision points and their outcomes
    decision_log = []  # [(depth, row, col, digit, led_to_solution)]

    def solve_and_log(grid, depth=0):
        """Solve with backtracking, logging decisions."""
        # Propagate
        for _ in range(81):
            changed = False
            for r in range(9):
                for c in range(9):
                    if grid[r][c] is None:
                        available = get_available_digits_9x9(grid, r, c)
                        if len(available) == 0:
                            return False  # Contradiction
                        if len(available) == 1:
                            grid[r][c] = list(available)[0]
                            changed = True
            if not changed:
                break

        # Check if solved
        empty = count_empty(grid)
        if empty == 0:
            valid, _ = validate_9x9(grid)
            return valid

        # Find MRV
        best_cell = None
        min_opts = 10
        for r in range(9):
            for c in range(9):
                if grid[r][c] is None:
                    opts = get_available_digits_9x9(grid, r, c)
                    if len(opts) < min_opts:
                        min_opts = len(opts)
                        best_cell = (r, c, list(opts))

        if best_cell is None:
            return False

        r, c, options = best_cell

        for digit in options:
            test_grid = [[cell for cell in row] for row in grid]
            test_grid[r][c] = digit

            result = solve_and_log(test_grid, depth + 1)

            # Log this decision
            decision_log.append({
                'depth': depth,
                'row': r,
                'col': c,
                'digit': digit,
                'led_to_solution': result,
                'num_options': len(options)
            })

            if result:
                return True

        return False

    # Solve and log
    grid = [[cell for cell in row] for row in PUZZLE_9x9_HARD]
    print("\nSolving with logging...")
    with Timer() as timer:
        solved = solve_and_log(grid)

    print(f"Solved: {solved}")
    print(f"Decision points logged: {len(decision_log)}")
    print(f"Time: {timer.elapsed:.2f}s")

    # Analyze decisions
    solution_decisions = [d for d in decision_log if d['led_to_solution']]
    fail_decisions = [d for d in decision_log if not d['led_to_solution']]

    print(f"\nDecisions leading to solution: {len(solution_decisions)}")
    print(f"Decisions leading to failure: {len(fail_decisions)}")

    # Key question: at decision points, are there patterns?
    print("\n" + "-" * 50)
    print("PATTERN ANALYSIS: Good vs Bad decisions")

    # Group by (row, col)
    by_position = defaultdict(list)
    for d in decision_log:
        by_position[(d['row'], d['col'])].append(d)

    print(f"\nUnique decision positions: {len(by_position)}")

    # Find positions where we have both good and bad outcomes
    mixed_positions = {k: v for k, v in by_position.items()
                       if any(d['led_to_solution'] for d in v) and any(not d['led_to_solution'] for d in v)}

    print(f"Positions with mixed outcomes: {len(mixed_positions)}")

    if mixed_positions:
        print("\nExample mixed position:")
        pos = list(mixed_positions.keys())[0]
        decisions = mixed_positions[pos]
        print(f"  Position ({pos[0]}, {pos[1]}):")
        for d in decisions:
            status = "✓ SOL" if d['led_to_solution'] else "✗ FAIL"
            print(f"    digit={d['digit']}, depth={d['depth']}: {status}")

    # Encode the decision patterns
    print("\n" + "-" * 50)
    print("ENCODING DECISION PATTERNS")

    # Encode each decision
    good_vecs = []
    bad_vecs = []

    for d in decision_log[:100]:  # Limit for speed
        pos_vec = encoder.pos_vectors[(d['row'], d['col'])]
        dig_vec = encoder.digit_vectors[d['digit']]
        choice_vec = bind(pos_vec, dig_vec)

        if d['led_to_solution']:
            good_vecs.append(choice_vec)
        else:
            bad_vecs.append(choice_vec)

    if good_vecs and bad_vecs:
        good_proto = bundle(good_vecs)
        bad_proto = bundle(bad_vecs)

        print(f"\nGood decisions encoded: {len(good_vecs)}")
        print(f"Bad decisions encoded: {len(bad_vecs)}")

        # Similarity between prototypes
        proto_sim = similarity(good_proto, bad_proto)
        print(f"Similarity between good/bad prototypes: {proto_sim:.4f}")

        # Test: can we predict outcome from encoding?
        print("\nTesting prediction on held-out decisions:")
        correct = 0
        total = 0
        for d in decision_log[100:150]:
            pos_vec = encoder.pos_vectors[(d['row'], d['col'])]
            dig_vec = encoder.digit_vectors[d['digit']]
            choice_vec = bind(pos_vec, dig_vec)

            sim_good = similarity(choice_vec, good_proto)
            sim_bad = similarity(choice_vec, bad_proto)

            pred = sim_good > sim_bad
            actual = d['led_to_solution']

            if pred == actual:
                correct += 1
            total += 1

        if total > 0:
            print(f"  Prediction accuracy: {correct}/{total} = {100*correct/total:.1f}%")


def main():
    print("=" * 70)
    print("APPROACH 16: SEARCH TREE ENCODING")
    print("=" * 70)
    print("\nEncoding the search tree itself in hyperspace.")
    print("Looking for patterns that can shortcut or transfer.\n")

    # First: shallow analysis
    analyze_tree_structure()

    # Second: full tree with solutions
    build_full_tree_with_backtracking()

    print("\n" + "=" * 70)
    print("CONCLUSIONS")
    print("=" * 70)


if __name__ == "__main__":
    main()
