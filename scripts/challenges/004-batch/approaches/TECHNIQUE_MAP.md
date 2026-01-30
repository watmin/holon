# Technique Map: Understanding What We Tried and Where the Impossibles Live

## The Core VSA/HDC Operations

| Operation | What It Does | Mathematical View |
|-----------|--------------|-------------------|
| **Binding (⊙)** | Creates associations: `pos ⊙ digit` | Element-wise multiply, preserves orthogonality |
| **Bundling (Σ)** | Superposes items: `Σ items` | Vector addition, creates "set" representation |
| **Unbinding** | Recovers bound element: `bundle ⊙ pos⁻¹` | Inverse of binding, noisy recovery |
| **Similarity** | Measures closeness: `cos(A, B)` | Dot product / norms, geometric distance |
| **Dimensionality** | Measures orthogonality spread | How many independent components |

---

## The 14 Approaches: Techniques and Barriers

### ENCODING STRATEGIES

#### 1. Position-Digit Binding (Standard)
```
grid_vec = Σ bind(pos, digit) for all filled cells
```
**Barrier:** Compositional - each cell independent, no global structure

#### 2. Relational Encoding
```
grid_vec = Σ bind(pos1 ⊙ pos2, "different") for constraint pairs
```
**Barrier:** Still local - encodes pairwise, not global chains

#### 3. Dual Space Encoding
```
pos→digit: standard encoding
digit→position: bundle(positions where digit appears)
```
**Barrier:** Both views are local; neither captures global propagation

#### 4. Constraint-Centric Encoding
```
constraint_vec = bind(constraint_id, bundle(digits in constraint))
```
**Barrier:** Each constraint encoded separately, no inter-constraint info

---

### RETRIEVAL STRATEGIES

#### 5. Direct Unbinding (Holographic)
```
To find digit at (r,c): unbind(grid_vec, pos(r,c)) → digit
```
**Barrier:** Works for known cells, NOISE for unknown cells
**Why:** Unbinding from partial information is inherently lossy

#### 6. Similarity to Ideal
```
Score digit by: similarity(grid_with_digit, ideal_template)
```
**Barrier:** "Ideal" is ambiguous - many valid completions exist
**Why:** Template doesn't encode WHICH valid completion

#### 7. Pattern Completion
```
Partial row + digit → how similar to "complete row"?
```
**SUCCESS (partial):** Got 54/58 cells
**Barrier:** Can't distinguish between locally-valid choices

---

### ITERATIVE STRATEGIES

#### 8. Hopfield/Energy Minimization
```
Energy = Σ (peer_similarity_penalty + clarity_reward)
Iterate: move each cell to reduce energy
```
**Barrier:** Non-convex landscape → local minima traps
**Why:** Multiple configurations have zero constraint violations

#### 9. Fixed Point Iteration
```
F(state) = apply_constraint_pressure(state)
Find x where F(x) = x
```
**Barrier:** Multiple fixed points exist (local minima)
**Why:** The solution is A fixed point, not THE ONLY one

#### 10. Belief Propagation
```
Messages flow between cells and constraints
Update beliefs based on neighbor messages
```
**Barrier:** Works for tree graphs, Sudoku has LOOPS
**Why:** Loopy BP doesn't guarantee convergence to truth

---

### SPECTRAL STRATEGIES

#### 11. Constraint Graph Eigenanalysis
```
Build adjacency matrix of "must differ" constraints
Analyze eigenvalues/eigenvectors
```
**Barrier:** Graph structure is FIXED for all Sudokus
**Why:** Eigenstructure doesn't depend on the specific puzzle

#### 12. Fiedler Vector Ordering
```
Use second eigenvector to order cell importance
Fill "most central" cells first
```
**Barrier:** Centrality ≠ most constrained ≠ correct order
**Why:** Graph centrality doesn't encode puzzle clues

---

### PREDICTION STRATEGIES

#### 13. Multi-Metric Voting
```
For each metric, pick the digit that scores highest
Vote across metrics
```
**Barrier:** Different metrics win for different cells - no pattern
**Why:** Each metric captures different incomplete information

#### 14. Simulation/Lookahead
```
Before choosing, simulate N moves ahead
Reject choices that lead to contradiction
```
**SUCCESS:** 10x backtrack reduction
**Why it works:** Converts local choice to limited global view

---

## The Barrier Taxonomy

### Barrier 1: LOCAL vs GLOBAL
```
What we CAN see:   Does digit X fit at position (r,c)?
What we CAN'T see: Does digit X lead to a solvable grid?
```
**The gap:** Global consequences of local choices

### Barrier 2: MULTIPLE VALID EXTENSIONS
```
At decision point: digits 3 and 6 both fit locally
Reality: Only one leads to solution
Geometry: Both look identical
```
**The gap:** No local signal distinguishes them

### Barrier 3: SIMILARITY IS MISLEADING
```
Experiment 14 showed:
  similarity(puzzle, correct_solution) = 0.2118
  similarity(puzzle, wrong_solution)   = 0.2521  ← HIGHER!
```
**The gap:** Wrong can be "closer" in representation space

### Barrier 4: INFORMATION GAP
```
Empty cells: 58
Bits needed to specify solution: ~66 bits
Bits in local geometry: ~0 for truly ambiguous cells
```
**The gap:** Required information doesn't exist locally

### Barrier 5: NON-CONVEX LANDSCAPE
```
Iterative methods assume: move downhill → reach minimum
Reality: Multiple minima, some wrong
```
**The gap:** No gradient points to global optimum

---

## The "Impossibles" - Where to Attack

### IMPOSSIBLE 1: Pure Geometric Retrieval
**Claim:** Retrieve solution in O(poly) time via similarity
**Why impossible:** NP-completeness; would prove P=NP
**Attack vector:** Can we get CLOSE? (93% achieved)

### IMPOSSIBLE 2: Local → Global Inference
**Claim:** Infer global consistency from local similarity
**Why impossible:** Information gap - global state not encoded locally
**Attack vector:** Can we ENCODE global state somehow?

### IMPOSSIBLE 3: Guaranteed Fixed Point
**Claim:** Iterate to unique correct fixed point
**Why impossible:** Multiple fixed points (local minima)
**Attack vector:** Can we make landscape more convex?

### IMPOSSIBLE 4: Perfect Similarity Ordering
**Claim:** Correct choice always has highest similarity
**Why impossible:** Wrong solutions can be more similar
**Attack vector:** Can we define BETTER similarity?

---

## Potential Attack Vectors

### Attack 1: DEEPER LOOKAHEAD
```
Current: Simulate 5 moves
Result: 10x backtrack reduction
Question: What if we simulate 10, 20, 50 moves?
Trade-off: Exponential cost for diminishing returns
```

### Attack 2: LEARNED ORIENTATION
```
Idea: Train a model to predict "which metric works here"
Input: Cell position, puzzle state, available options
Output: Best scoring function for this decision
Question: Does pattern exist across puzzles?
```

### Attack 3: SOLUTION DATABASE
```
Idea: Pre-compute vectors for many known solutions
Query: Find most similar solution to partial puzzle
Trade-off: Space vs. time; may not generalize
Question: Can we compress solution space?
```

### Attack 4: APPROXIMATE SOLUTIONS
```
Idea: Accept 93% accuracy for 1000x speed
Application: Puzzle hints, difficulty estimation
Question: What accuracy is achievable deterministically?
```

### Attack 5: CONSTRAINT SPACE ENCODING
```
Current: Encode grid STATE
Alternative: Encode CONSTRAINT SATISFACTION STRUCTURE
Idea: The solution is intersection of constraint manifolds
Question: Can we find intersection geometrically?
```

### Attack 6: HIGHER-ORDER BINDINGS
```
Current: Binary binding (pos ⊙ digit)
Alternative: Ternary+ binding (pos ⊙ digit ⊙ constraint)
Idea: Encode relationships, not just associations
Question: Does higher-order capture more global structure?
```

### Attack 7: TEMPORAL/CAUSAL ENCODING
```
Current: Static grid state
Alternative: Encode SEQUENCE of deductions
Idea: Capture logical chain, not just end state
Question: Can we encode "if A then B then C"?
```

### Attack 8: DIFFERENT PROBLEM DOMAINS
```
Sudoku: NP-complete, discrete, exact solution required
Question: Where might pure geometry work BETTER?
Candidates:
  - Approximate matching (already works!)
  - Continuous optimization
  - Classification (threshold ok)
  - Pattern detection
```

---

## Summary: The Landscape

```
                    SOLVABLE BY GEOMETRY
                           ↓
    ┌──────────────────────────────────────────┐
    │  Pattern matching, similarity search     │ ← VSA/HDC excels
    │  Approximate retrieval, classification   │
    └──────────────────────────────────────────┘
                           ↓
    ┌──────────────────────────────────────────┐
    │  Constraint satisfaction (easy)          │ ← Works via propagation
    │  Tree-structured problems                │
    └──────────────────────────────────────────┘
                           ↓
    ┌──────────────────────────────────────────┐
    │  BARRIER: Local vs Global                │ ← We hit this
    │  BARRIER: Multiple valid extensions      │
    └──────────────────────────────────────────┘
                           ↓
    ┌──────────────────────────────────────────┐
    │  NP-complete problems (Sudoku, SAT)      │ ← Need search
    │  Geometry can GUIDE but not REPLACE      │
    └──────────────────────────────────────────┘
                           ↓
                    REQUIRES SEARCH
```

The boundary is REAL. But finding exactly where it is, and pushing it, is the game.
