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

## Approach 5: Structural Entanglement

**Status:** TO BE TESTED

---
