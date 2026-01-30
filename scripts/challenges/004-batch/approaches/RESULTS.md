# Radical Approaches - Experimental Results

## Approach 9: Dimensional Analysis

**Status:** PARTIAL SUCCESS

### Hypothesis
Valid configurations have higher "effective dimensionality" than invalid ones.

### Results

| Puzzle | Solved | Time | Notes |
|--------|--------|------|-------|
| 4x4 Easy | ✓ | 0.016s | 1 dimensional choice, 7 forced |
| 9x9 Easy | ✓ | 0.005s | 0 dimensional choices, all forced |
| 9x9 Hard | ✗ | 0.042s | 4 dimensional choices → contradiction |

### Key Findings

**What Works:**
1. Dimensionality perfectly discriminates valid from invalid rows:
   - Valid [1,2,3,4,5,6,7,8,9]: dim = 0.9999
   - Invalid [1,1,3,4,5,6,7,8,9]: dim = 0.9280
   - Very invalid [2,2,2,2,2,6,7,8,9]: dim = 0.5263

2. Adding ANY valid digit increases dimensionality:
   - Partial [1,2,3,4,5] + 6 → 0.8284 (valid)
   - Partial [1,2,3,4,5] + 1 → 0.7245 (duplicate)

3. Easy puzzles solved without backtracking (but mostly via forced moves)

**What Doesn't Work:**
1. **Cannot distinguish between multiple valid options:**
   - Cell (2,1): digit 3 → 0.583, digit 6 → 0.581
   - Both are valid! Dimensionality can't tell which leads to global solution.

2. **Greedy local optimality ≠ global optimality**
   - The highest-dimensionality choice at each step doesn't guarantee a solution
   - This is fundamental - not a bug in implementation

### Conclusion

Dimensionality is a powerful **validity detector** but not a sufficient **solution finder**.

**Possible enhancements:**
- Combine with other geometric signals (orientation, similarity)
- Use as part of hybrid approach (geometric + backtracking)
- Consider lookahead (dimensionality after multiple steps)

---

## Approach 2: Superposition Collapse

**Status:** PARTIAL SUCCESS

### Hypothesis
Represent unknown cells as superposition of digit vectors.
Apply constraints by removing digits from superpositions.
Cells collapse when one digit remains or dominates.

### Results

| Puzzle | Solved | Time | Notes |
|--------|--------|------|-------|
| 4x4 Easy | ✗ | 0.004s | Stuck with 2 options per cell (2 vs 3) |
| 9x9 Easy | ✓ | ~0.1s | All cells reduced to 1 option via propagation |
| 9x9 Hard | ✗ | 0.07s | 0 collapses, cells have 3-6 options each |

### Key Findings

**What Works:**
1. Remove operation successfully excludes digits (goes negative in similarity)
2. Tracking removed digits prevents double-removal
3. Cells with 1 remaining option collapse automatically
4. Easy puzzles (all naked singles) solve completely

**What Doesn't Work:**
1. **Cannot distinguish between multiple valid options:**
   - Cell with [2, 3] remaining: sim 0.338 vs 0.333
   - Difference is noise-level (0.005)

2. **Superposition collapse = constraint propagation**
   - It's implementing arc consistency in hyperspace
   - No additional insight beyond what set-based propagation provides

3. **Hard puzzles have no naked singles**
   - Multiple valid options at each cell
   - No geometric signal to guide choice

### Conclusion

Superposition collapse is a valid hyperspace implementation of constraint propagation,
but provides **no advantage over traditional propagation** for hard puzzles.

---

## Approach 1: Hopfield Constraint Resonance

**Status:** FAILED

### Results
| Puzzle | Result | Notes |
|--------|--------|-------|
| 4x4 | ✗ | All cells → digit 2 |
| 9x9 Hard | ✗ | Converged to invalid state with repeated digits |

### Key Findings
- Energy minimization converges to local minima
- Cells "avoid" each other but settle to similar patterns
- Confidence values very low (0.006-0.054)
- Classic Hopfield problem: stable ≠ valid

---

## Approach 6: Constraint Orientation

**Status:** PARTIAL SUCCESS

### Results
| Puzzle | Cells | Valid | Notes |
|--------|-------|-------|-------|
| 4x4 | 4/4 | ✓ | Solved! |
| 9x9 Hard | 53/58 | ✗ | Similar to row-only |

### Key Findings
- **Correctly identifies intersection digits!**
- When Row allows {1,2,3,4,5}, Col allows {3,4,5,6,7}, Block allows {4,5,6,7,8}:
  - Intersection {4,5} has highest total score (1.186, 1.245)
  - Non-intersection digits score lower
- Still makes wrong early decisions

---

## Approach 7: Data Similarity

**Status:** PARTIAL SUCCESS

### Results
| Puzzle | Cells | Valid | Notes |
|--------|-------|-------|-------|
| 4x4 | 4/4 | ✓ | Solved! |
| 9x9 Hard (Completion) | 52/58 | ✗ | |
| 9x9 Hard (Missing) | 52/58 | ✗ | 10x faster |

### Key Findings
- **Adding valid digit increases similarity** (+0.055)
- **Adding duplicate DECREASES similarity** (-0.053) ← Natural validity detector!
- Missing digit detection works perfectly:
  - Missing digits: ~0.44 similarity
  - Present digits: ~0.00 similarity

---

## Approach 5: Structural Entanglement

**Status:** PARTIAL SUCCESS (most promising so far!)

### Hypothesis
Binding creates entanglement that can be queried/exploited.
Use structure comparison to guide placements.

### Results

| Puzzle | Solved | Time | Notes |
|--------|--------|------|-------|
| 4x4 Easy | ✓ | ~0.01s | Row completion works! |
| 9x9 Easy | ✓ | ~0.5s | All cells placed correctly |
| 9x9 Hard | ✗ | 8.5s | 54/58 cells (93%), then stuck |

### Key Findings

**What Works:**
1. **Unbinding recovers known bindings** (100% accuracy from bundle)
2. **Row completion approach**: Compare partial row to ideal row structure
3. **Clear score gaps** when obvious choice exists
4. **Solved both easy puzzles WITHOUT backtracking!**

**What Doesn't Work:**
1. **Querying unknown positions** returns noise
2. **Row-only comparison** doesn't consider column/block constraints
3. **Hard puzzle** makes wrong choices that lead to contradiction

### Why Row Completion Works Better

The key insight: instead of asking "what digit IS here?" (which returns noise),
we ask "which digit SHOULD be here to make row more ideal?"

This frames the problem as **pattern completion** rather than **information extraction**.

### Extensive Strategy Testing (January 2026)

Tested multiple strategies to improve on row-only:

| Strategy | Cells | Valid | Notes |
|----------|-------|-------|-------|
| Row-only (greedy) | 54/58 | ✗ | Best coverage but wrong decisions |
| Full avg (row+col+block) | 51/58 | ✗ | Averaging dilutes signal |
| Full min | 51/58 | ✗ | Too conservative |
| Full max | 51/58 | ✗ | Not helpful |
| Full product | 52/58 | ✗ | Slightly better than avg |
| Voting (unanimous) | 46/58 | ✗ | Too conservative |
| Row + Lookahead | 44/58 | ✗ | Avoids contradictions, fewer placements |

**Key Finding:** The row-only approach is actually the BEST. Adding more constraints or being more conservative HURTS performance.

### Why Row-Only is Best

1. Row constraint gives the clearest geometric signal
2. Adding column/block adds noise rather than clarity
3. All strategies make fundamentally wrong decisions due to lack of global consistency encoding

### The Fundamental Limitation

**Geometric similarity to "ideal constraint" doesn't encode global consistency.**

When choosing between digits 3 and 6 at a cell:
- Both are valid for the row (not duplicates)
- Both increase similarity to ideal row by similar amounts
- But only ONE leads to a globally consistent solution
- The geometric signal cannot distinguish between them

### Comparison to Other Approaches

| Approach | 4x4 | 9x9 Easy | 9x9 Hard | Key Insight |
|----------|-----|----------|----------|-------------|
| Approach 9 (Dimensional) | ✓ | ✓ | 6/58 | Detects validity, not global consistency |
| Approach 2 (Superposition) | ✗ | ✓ | 0/58 | Constraint propagation in hyperspace |
| **Approach 5 (Entanglement)** | ✓ | ✓ | **54/58** | **Best greedy geometric approach** |

Approach 5 gets MUCH further on the hard puzzle (54 cells vs 0-6 for others), but still cannot achieve a valid solution without backtracking.

---

## Combined Approach

**Status:** WORSE THAN SIMPLE

### Results
| Configuration | Cells | Valid |
|---------------|-------|-------|
| Greedy (row+col+block) | 51/58 | ✗ |
| No penalty | 51/58 | ✗ |
| Conservative | 51/58 | ✗ |
| Very conservative | 50/58 | ✗ |

### Key Finding
**Combining signals HURTS performance.** Simple row-only (54/58) beats all combined approaches (50-51/58).

---

---

## Approach 3: Direct Geometric Decoding

**Status:** FAILED

### Results
| Puzzle | Result | Notes |
|--------|--------|-------|
| 4x4 | ✗ | Decodes noise from partial encoding |
| 9x9 Hard | ✗ | All cells filled with wrong digits |

### Key Findings
- Encode-decode works for COMPLETE solutions (90%+ accuracy)
- **Decoding from PARTIAL encoding returns noise**
- Confidence values low (0.1-0.2)
- Cannot infer unknown cells from known cells via unbinding

---

## Approach 4: Constraint Propagation in Hyperspace

**Status:** PARTIAL

### Results
| Puzzle | Method | Cells | Valid | Notes |
|--------|--------|-------|-------|-------|
| 4x4 | Pure Prop | 0/8 | ✗ | No naked singles |
| 9x9 Hard | Pure Prop | 0/58 | ✗ | No naked singles |
| 9x9 Hard | Prop + Geo | 58/58 | ✗ | Fills all but creates duplicates |

### Key Findings
- Pure propagation = set-based arc consistency (no geometric advantage)
- Adding geometric choice fills all cells but makes wrong decisions
- Geometric scoring doesn't prevent constraint violations

---

## Approach 8: Inverse Encoding

**Status:** PARTIAL SUCCESS

### Results
| Puzzle | Cells | Valid | Notes |
|--------|-------|-------|-------|
| 4x4 | 4/4 | ✓ | Solved! |
| 9x9 Hard | 52/58 | ✗ | Similar to other approaches |

### Key Findings
- Inverse encoding (digit→position) gives different perspective
- Works on simple puzzles
- **Still hits the same barrier** on hard puzzles

---

# FINAL SUMMARY

## All 9 Approaches Tested

| # | Approach | 4x4 | 9x9 Hard | Key Insight |
|---|----------|-----|----------|-------------|
| 1 | Hopfield | ✗ | ✗ (invalid) | Converges to local minima |
| 2 | Superposition | ✗ | 0/58 | = Constraint propagation |
| 3 | Direct Decoding | ✗ | ✗ (invalid) | Partial encoding → noise |
| 4 | Propagation | ✗ | 0/58* | No naked singles |
| **5** | **Entanglement (Row-only)** | ✓ | **54/58** | **BEST RESULT** |
| 6 | Orientation | ✓ | 53/58 | Correctly finds intersection |
| 7 | Similarity | ✓ | 52/58 | Detects duplicates |
| 8 | Inverse Encoding | ✓ | 52/58 | Different perspective, same barrier |
| 9 | Dimensional | ✓ | 6/58 | Detects validity only |
| - | Combined | ✓ | 51/58 | Worse than simple! |

*Approach 4 with geometric choice fills 58/58 but creates invalid solution

## Conclusions

### What Works
1. **Simpler is better**: Row-only completion beats all combinations
2. **Pattern completion framing**: "What should be here?" not "What is here?"
3. **Constraint intersection detection**: Correctly identifies valid digits
4. **Duplicate detection**: Geometric natural validity check

### What Doesn't Work
1. **Combining signals**: Dilutes rather than strengthens
2. **Hopfield energy minimization**: Local minima ≠ valid solutions
3. **Conservative approaches**: Just make fewer placements

### The Fundamental Limitation

**Geometric similarity cannot encode global consistency.**

When choosing between digits 3 and 6 for a cell:
- Both are valid for row/col/block
- Both increase similarity to "ideal" patterns
- But only ONE leads to a globally consistent solution
- The geometric signal is noise-level for this distinction

### The 54/58 Barrier

Pure geometric approaches plateau at ~93% (54/58) on hard puzzles.
The remaining cells require information that local geometric comparisons cannot provide.

**Options to break through:**
1. Hybrid: Geometric guidance + minimal backtracking (implemented in 001-solution.py)
2. Solution encoding: Pre-encode known solutions and match (defeats the purpose)
3. Global consistency mechanism: Not yet discovered

---
