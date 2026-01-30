# Radical Approaches: Genuine Geometric Constraint Satisfaction

This document outlines 9 approaches for solving constraints through hyperspace geometry rather than augmenting traditional search.

## The Goal

```
puzzle_vector = encode(incomplete_puzzle)
solution_vector = geometric_solve(puzzle_vector)  # No backtracking
solution = decode(solution_vector)
```

The solution should "fall out" of the geometry, not be searched for.

---

## Approach 1: Constraint Resonance (Hopfield-Style)

**Status:** To be tested

### Concept
Encode the puzzle such that the valid solution is an **attractor state** in hyperspace. The system "settles" into the solution through iterative resonance.

### Key Idea
- Each constraint is a "resonator" vector
- Binding with constraint, then unbinding, should "clean up" the state toward validity
- Valid solutions are fixed points under constraint application

### Implementation
```python
def resonate(state, resonator):
    projection = bind(state, resonator)
    cleaned = threshold(projection)
    return unbind(cleaned, resonator)
```

### Research Questions
1. How to encode constraints such that valid solutions are stable?
2. What is the basin of attraction?
3. Does dimensionality affect convergence?

---

## Approach 2: Superposition Collapse

**Status:** To be tested

### Concept
Each cell starts as a **superposition of all 9 digits**. Constraints are applied as **unbinding operations** that progressively collapse each cell to a single digit.

### Key Idea
- Unknown cell = bundle of all 9 digit vectors
- When a digit is placed, "subtract" it from peers' superpositions
- Cells automatically collapse when one digit dominates

### Implementation
```python
def apply_constraint(states, group):
    for cell in group:
        if is_collapsed(states[cell]):
            digit = extract_digit(states[cell])
            for peer in group:
                if peer != cell:
                    states[peer] = remove_digit(states[peer], digit)
```

### Research Questions
1. Does unbinding properly "remove" a digit from superposition?
2. How to handle conflicts (two cells want same digit)?
3. What dimensionality is needed for 9 distinguishable digits?

---

## Approach 3: Direct Geometric Decoding

**Status:** To be tested

### Concept
Encode the constraint structure such that the solution can be **directly extracted** by unbinding operations.

### Key Idea
- Instead of encoding all solutions, encode what MAKES a solution valid
- The constraint structure implicitly defines all valid solutions
- The puzzle + constraints should uniquely determine the solution

### Implementation
```python
def encode_constraint_structure():
    # Each row must have all 9 digits - encode as structure
    row_constraints = bundle([
        bind(position_vec(r, c), digit_superposition)
        for r in range(9) for c in range(9)
    ])
    # Intersection of all constraints IS the solution space
```

### Research Questions
1. Can constraint structure be encoded compactly?
2. Does binding/unbinding preserve constraint relationships?
3. How to decode unique solution from constraint intersection?

---

## Approach 4: Constraint Propagation in Hyperspace

**Status:** To be tested

### Concept
Implement classic constraint propagation (like AC-3) but using vector operations instead of set operations.

### Key Idea
- Each cell has a "domain vector" (superposition of possible digits)
- Propagation removes digits from domains using vector operations
- Vector subtraction instead of set subtraction

### Implementation
```python
def remove_digit_from_superposition(superposition, digit_vec):
    # Options:
    # 1. Unbind: superposition ⊙ digit_vec^(-1)
    # 2. Subtract: superposition - digit_vec
    # 3. Project: superposition - project(superposition, digit_vec)
    pass
```

### Research Questions
1. What's the vector equivalent of set subtraction?
2. Does propagation converge in hyperspace?
3. Can we detect "naked pairs" geometrically?

---

## Approach 5: Structural Entanglement

**Status:** To be tested

### Concept
Exploit Holon's recursive encoding to create **entangled** representations where unbinding extracts the answer.

### Key Idea
- A cell placement is encoded as: `bind(position, digit)`
- To query what digit is at a position, unbind the position
- Build a grid vector where every position unbinds to the correct digit

### Implementation
```python
# Encode a cell
cell_00_has_5 = encode({"pos": {"row": 0, "col": 0}, "digit": 5})

# Query: unbind position to get digit
pos_00 = encode({"pos": {"row": 0, "col": 0}})
digit_at_00 = unbind(cell_00_has_5, pos_00)  # Should recover digit 5's vector

# Solution: the vector where ALL positions unbind correctly
```

### Research Questions
1. Does unbinding actually recover the bound value reliably?
2. Can we build a composite vector with multiple entangled bindings?
3. Does bundling interfere with unbinding accuracy?

---

## Approach 6: Constraint Orientation Space

**Status:** To be tested

### Concept
Each constraint defines a **direction** in hyperspace. The solution is the unique orientation that aligns with ALL constraint directions simultaneously.

### Key Idea
- "Row 0 must have all digits" → specific geometric orientation
- A valid solution's row projection aligns with ideal orientation
- Find the vector that aligns with all 27 constraint directions

### Implementation
```python
# Each constraint has an "ideal" orientation
row_0_ideal = bundle([
    bind(encode({"row": 0, "col": c}), encode({"digit": d}))
    for c, d in some_valid_row_arrangement
])

# Solution aligns with ALL constraint ideals
def constraint_alignment_score(solution, constraints):
    return sum(similarity(solution, c) for c in constraints)
```

### Research Questions
1. Do valid solutions have measurably different orientations than invalid ones?
2. Can we "average" constraint directions to find the solution?
3. Is there a unique orientation that satisfies all constraints?

---

## Approach 7: Data Similarity Exploitation

**Status:** To be tested

### Concept
Exploit Holon's **data** similarity (not text similarity). Partial solutions that share structure are geometrically close.

### Key Idea
- Two grids sharing placements have high similarity
- The solution is "maximally similar" to puzzle while being complete and valid
- Navigate toward the solution by maximizing data overlap

### Implementation
```python
# Grids that share placements are similar
grid_1 = encode({"cells": [(0,0,5), (0,1,3)]})
grid_2 = encode({"cells": [(0,0,5), (0,1,3), (0,2,4)]})
similarity(grid_1, grid_2)  # High - grid_2 extends grid_1

# Solution: maximize similarity to puzzle while satisfying constraints
def find_solution(puzzle):
    puzzle_vec = encode(puzzle)
    # Find the "valid" vector closest to puzzle_vec
```

### Research Questions
1. How similar are partial solutions that share placements?
2. Can we use similarity gradient to guide toward solution?
3. Is the solution the "closest valid point" to the puzzle?

---

## Approach 8: Inverse Encoding

**Status:** To be tested

### Concept
Exploit the **reversibility of binding** to extract unknown values from the constraint structure.

### Key Idea
- If `solution_vec = bundle([bind(pos_i, digit_i) for each cell])`
- Then `unbind(solution_vec, pos_i)` should give `digit_i`
- Can we compute `solution_vec` from constraints + puzzle?

### Implementation
```python
# Build query for unknowns
for empty_cell in empty_cells:
    position_vec = encode({"pos": empty_cell})

    # Unbind from constraint structure to get digit
    cell_vec = unbind(constraint_structure, position_vec)

    # Decode: which digit vector is closest?
    digit = argmax([similarity(cell_vec, digit_vec[d]) for d in range(1, 10)])
```

### Research Questions
1. What should `constraint_structure` contain for this to work?
2. Does unbinding from bundled structure preserve individual bindings?
3. Can we incrementally build the solution by successive unbindings?

---

## Approach 9: Dimensional Analysis

**Status:** To be tested

### Concept
Valid and invalid configurations have **measurably different geometric properties** like effective dimensionality.

### Key Idea
- A valid row bundles 9 orthogonal vectors → spans 9 dimensions
- An invalid row (with duplicates) spans fewer dimensions
- Use dimensional properties to detect and enforce validity

### Implementation
```python
def row_validity_score(row_vector, digit_vectors):
    # Project onto each digit's direction
    projections = [np.dot(row_vector, dv) for dv in digit_vectors]

    # Valid row: all projections roughly equal (uniform spread)
    # Invalid row: some projections much larger (duplicates)
    uniformity = 1.0 - np.std(projections) / np.mean(np.abs(projections))
    return uniformity
```

### Research Questions
1. Can we reliably distinguish valid from invalid rows by dimension?
2. What geometric measure best captures "all different" constraint?
3. Can this guide placement decisions without backtracking?

---

## Experimental Strategy

### Phase 1: Validate Fundamentals (All Approaches)
For each approach, test the core assumption:
1. Does the key operation work as expected?
2. On trivial examples (4x4 Sudoku)?
3. What are the failure modes?

### Phase 2: Identify Viable Approaches
Based on Phase 1:
- Which approaches work on simple cases?
- Which fail fundamentally?
- Which need refinement?

### Phase 3: Scale Up Viable Approaches
For approaches that pass Phase 1:
- Test on 9x9 easy puzzles
- Test on 9x9 hard puzzles
- Measure success rate and failure modes

### Phase 4: Hybrid Integration
Combine successful geometric approaches:
- Use geometry as far as it goes
- Identify minimal fallback needed
- Measure what percentage is truly geometric

---

## Directory Structure

```
scripts/challenges/004-batch/
├── 001-solution.py              # Current hyperspace-guided backtracking
├── 002-solution-http.py         # HTTP API demo
├── approaches/
│   ├── __init__.py
│   ├── common.py                # Shared utilities (puzzles, validation, etc.)
│   ├── approach_01_resonance.py
│   ├── approach_02_superposition.py
│   ├── approach_03_direct_decode.py
│   ├── approach_04_propagation.py
│   ├── approach_05_entanglement.py
│   ├── approach_06_orientation.py
│   ├── approach_07_similarity.py
│   ├── approach_08_inverse.py
│   └── approach_09_dimensional.py
└── run_all_approaches.py        # Benchmark runner
```

---

## Success Criteria

**Truly Radical Success:**
- Solve >50% of easy puzzles with ZERO backtracking
- Geometric operations only (bind, bundle, unbind, similarity)
- Solution "falls out" of the geometry

**Partial Success:**
- Geometric approach handles more than just ordering
- Constraint propagation works in hyperspace
- Measurably less search than current approach

**Learning Success:**
- Understand exactly why/where each approach fails
- Document the fundamental limitations
- Identify if kernel primitives are missing

---

## Kernel Primitives Available

From `holon/encoder.py`:
- `encode_data(data)` - Recursive structural encoding
- `bind(vec1, vec2)` - Element-wise multiplication
- `bundle(vectors)` - Sum + threshold to bipolar
- `mathematical_bind(*vectors)` - Bind with thresholding
- `mathematical_bundle(vectors, weights)` - Weighted bundle

From `holon/similarity.py`:
- `find_similar_vectors(probe, store, top_k, threshold)`
- Cosine similarity

---

## Fundamental Operations Validated (January 2026)

All core operations have been tested and work:

| Operation | Result | Notes |
|-----------|--------|-------|
| **Bind/Unbind** | ✓ | Recovery sim ~0.54 vs ~0.01 for others - clearly distinguishable |
| **Multiple Bindings** | ✓ | Can recover cells from bundled grid (0.34-0.38 sim) |
| **Superposition Detection** | ✓ | Present digits ~0.52 vs absent ~0.01 |
| **Remove from Superposition** | ✓ | Removed digit goes from +0.44 to -0.43 |
| **Data Similarity** | ✓ | Same digit → 0.60, same pos → 0.35, different → ~0 |
| **Dimensionality** | ✓ | **Valid=0.9999, invalid=0.73, very invalid=0.07** |

**Key Finding:** The dimensionality measure perfectly discriminates valid from invalid configurations. This is extremely promising for Approach 9.

**Helper Primitives (in `approaches/common.py`):**
- `unbind(composite, component)` - Same as bind for bipolar vectors
- `remove_component(vector, component)` - Subtract and re-threshold
- `effective_dimensionality(vector, basis)` - Entropy-based uniformity measure
- `project(vector, onto)` - Project onto direction
- `similarity(v1, v2)` - Cosine similarity

These are userland implementations using Holon kernel primitives.
