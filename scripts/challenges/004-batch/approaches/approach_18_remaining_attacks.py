#!/usr/bin/env python3
"""
Approach 18: Evaluate Remaining Attack Vectors

Quickly assess each remaining attack:
1. Pre-compute and cache
2. Probabilistic bounds
3. Different domains
"""

import sys
sys.path.insert(0, "/home/watmin/work/holon")

from typing import List, Optional, Set, Tuple, Dict
import numpy as np

from common import (
    create_client,
    VectorCache,
    bind,
    bundle,
    similarity,
    Timer,
    print_grid_9x9,
    validate_9x9,
    count_empty,
    get_available_digits_9x9,
    PUZZLE_9x9_HARD,
)


# ============================================================================
# ATTACK 1: Pre-compute and Cache
# ============================================================================

def assess_precompute():
    """
    Can we pre-compute "contradiction signatures" that help new puzzles?

    Idea: Build a database of (abstract_state → outcome) mappings.
    For new puzzles, check if state matches known contradictions.
    """
    print("=" * 70)
    print("ATTACK 1: Pre-compute and Cache")
    print("=" * 70)

    print("""
IDEA:
- Build database of abstract states → outcomes from solved puzzles
- For new puzzles, lookup similar states to predict outcomes

WHAT WE KNOW:
- Abstract features transfer (0.76 similarity)
- Contradiction detection works within same puzzle (85% when grid is full)
- But contradiction detection fails at start (0%) - too early to tell

ASSESSMENT:
The problem is that contradictions only become detectable LATER.
Caching early states doesn't help because they all look similar.
Caching late states doesn't help because they're puzzle-specific.

VIABLE USE CASE:
- Cache "constraint violation patterns" (e.g., duplicate in row)
- These are simple rules, not learned patterns
- Already implemented as constraint propagation

CONCLUSION: PRE-COMPUTE HAS LIMITED VALUE
- Abstract features transfer but don't predict outcomes
- Contradiction patterns are either too early (no signal) or too specific
- Better to use simulation (Approach 10) which computes on-demand
""")

    return "LIMITED VALUE - simulation is better"


# ============================================================================
# ATTACK 2: Probabilistic Bounds
# ============================================================================

def assess_probabilistic():
    """
    What accuracy can we achieve with pure geometric methods?
    Can we bound this theoretically or empirically?
    """
    print("\n" + "=" * 70)
    print("ATTACK 2: Probabilistic Bounds")
    print("=" * 70)

    print("""
EMPIRICAL BOUNDS FROM OUR EXPERIMENTS:

| Method | Cells Filled | Accuracy | Valid Solution? |
|--------|--------------|----------|-----------------|
| Random guessing | ~35/58 | ~40% | No |
| Pure geometric (best) | 54/58 | 93% | No |
| Simulation-guided | 58/58 | 100% | YES |

THE 93% BARRIER:
- Achieved by Approach 5 (row completion)
- Fills cells where geometric signal is strong
- Gets stuck when multiple options look identical

WHAT DETERMINES THE BOUND?
1. Number of truly ambiguous decision points
2. How early ambiguity appears
3. Depth of search needed to resolve ambiguity

FOR OUR HARD PUZZLE:
- 58 empty cells
- ~10 decision points with 2+ options that look similar
- Each wrong choice requires backtracking
- Pure greedy can fill 54, then gets trapped

THEORETICAL INSIGHT:
The "probabilistic bound" is puzzle-dependent.
Easy puzzles: geometric gets 100% (all forced moves)
Hard puzzles: geometric gets ~90-95% (ambiguity at end)
Very hard: geometric gets <50% (early ambiguity)

CONCLUSION: 93% IS SOFT CEILING FOR HARD PUZZLES
- Actual bound depends on puzzle structure
- Can estimate by counting decision points
- Not a universal constant
""")

    # Quick empirical test
    print("EMPIRICAL CHECK:")
    print("How many decision points have 2+ similarly-scored options?")

    from approach_05_entanglement import EntanglementSolver

    solver = EntanglementSolver(size=9, dimensions=16384, verbose=False)
    grid = [[cell for cell in row] for row in PUZZLE_9x9_HARD]

    ambiguous = 0
    clear = 0

    for r in range(9):
        for c in range(9):
            if grid[r][c] is None:
                scores = solver.query_row_completion(grid, r, c)
                if len(scores) >= 2:
                    sorted_scores = sorted(scores.values(), reverse=True)
                    gap = sorted_scores[0] - sorted_scores[1]
                    if gap < 0.01:  # Very similar
                        ambiguous += 1
                    else:
                        clear += 1

    print(f"  Ambiguous decision points (gap < 0.01): {ambiguous}")
    print(f"  Clear decision points: {clear}")

    return f"~93% for hard puzzles, {ambiguous} ambiguous points"


# ============================================================================
# ATTACK 3: Different Domains
# ============================================================================

def assess_different_domains():
    """
    Where might geometric methods work BETTER than Sudoku?
    """
    print("\n" + "=" * 70)
    print("ATTACK 3: Different Domains")
    print("=" * 70)

    print("""
SUDOKU CHARACTERISTICS (why it's hard for geometry):
- Discrete: 9 possible values per cell
- Global constraints: choices propagate across entire grid
- Multiple valid solutions exist for partial states
- No gradient: correct vs incorrect are equally "valid" locally

DOMAINS WHERE GEOMETRY MIGHT WORK BETTER:

1. APPROXIMATE MATCHING (already works!)
   - Find similar items in database
   - No exact solution required
   - VSA/HDC excels here

2. CLASSIFICATION
   - Threshold-based decisions
   - One "best" answer, not exact match
   - Geometric similarity is meaningful

3. SEQUENCE COMPLETION
   - Predict next item in pattern
   - Similarity to known patterns
   - Partial matches are useful

4. CONSTRAINT SATISFACTION (easy cases)
   - When constraints are local, not global
   - Tree-structured problems
   - No backtracking needed

DOMAINS THAT WOULD BE SIMILARLY HARD:

1. SAT (Boolean satisfiability)
   - Same NP-complete structure
   - Local validity ≠ global solution

2. GRAPH COLORING
   - Same as Sudoku (9-coloring of constraint graph)

3. SCHEDULING
   - Many local solutions, few global ones

4. INTEGER PROGRAMMING
   - Discrete optimization with constraints

THE PATTERN:
- Geometry works when local signals correlate with global solutions
- Fails when valid local choices diverge globally
- NP-complete problems have this divergence by definition

BEST DOMAIN FOR GEOMETRY:
Problems where "approximately correct" is acceptable.
Geometry gives great heuristics, just not exact solutions.
""")

    return "Approximate matching, classification, easy CSPs"


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("EVALUATING REMAINING ATTACK VECTORS")
    print("=" * 70)

    results = {}

    results['precompute'] = assess_precompute()

    try:
        results['probabilistic'] = assess_probabilistic()
    except ImportError:
        results['probabilistic'] = "Could not import (93% empirical bound)"

    results['domains'] = assess_different_domains()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY OF REMAINING ATTACKS")
    print("=" * 70)

    print(f"""
| Attack | Assessment |
|--------|------------|
| Pre-compute/cache | {results['precompute']} |
| Probabilistic bounds | {results['probabilistic']} |
| Different domains | {results['domains']} |

OVERALL CONCLUSION:
The remaining attacks don't offer breakthrough potential for hard CSPs.

- Pre-compute: Simulation is better (computes on-demand, always fresh)
- Probabilistic: 93% ceiling is puzzle-dependent, not breakable
- Different domains: Geometry works where approximate is OK

THE ANSWER TO "CAN WE SOLVE NP-HARD PURELY GEOMETRICALLY?" IS:
NO - but geometry provides valuable 10x speedup for guided search.
""")


if __name__ == "__main__":
    main()
