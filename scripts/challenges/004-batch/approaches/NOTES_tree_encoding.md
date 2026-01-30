# Encoding the Search Tree in Hyperspace

## The Radical Idea

Instead of encoding the PUZZLE STATE, encode the SEARCH TREE itself.

```
Standard encoding:
  puzzle_vec = encode(grid_state)

Tree encoding:
  tree_vec = encode(all possible paths from puzzle to solution/contradiction)
```

## Why This Might Work

The search tree contains ALL the information:
- Which choices lead to contradiction (bad branches)
- Which choices lead to solution (good branches)
- Which choices are equivalent (symmetry)
- Which patterns repeat (compression opportunity)

## The Structure

```
                    [Initial Puzzle]
                    /       |       \
                [A=1]    [A=2]    [A=3]
               /    \      |        \
           [B=4]  [B=5]  [B=4]    [STUCK]
             |      |      |
          [SOL]  [BAD]  [SOL]
```

Each node: a grid state
Each edge: a choice (position, digit)
Leaves: SOLUTION, CONTRADICTION, or STUCK

## Encoding Options

### Option 1: Path Encoding
Encode each path from root to leaf as a sequence:
```
path_vec = bind(bind(bind(choice1, choice2), choice3), outcome)
```

Bundle all paths:
```
tree_vec = bundle([path_vec for path in all_paths])
```

**Problem:** Exponential number of paths

### Option 2: Decision Point Encoding
Encode decision points with their outcomes:
```
decision_vec = bind(state_vec, bundle([
    bind(choice1, outcome1),
    bind(choice2, outcome2),
    ...
]))
```

**Advantage:** Captures the branching structure

### Option 3: Outcome-Weighted Encoding
Encode each choice weighted by its outcome quality:
```
choice_vec = bind(position, digit) * outcome_score
tree_vec = bundle(all_weighted_choices)
```

**Advantage:** Good choices dominate the representation

### Option 4: Hierarchical Encoding
Encode at multiple levels:
```
level0 = puzzle_vec
level1 = bundle([encode(state_after_choice) for choice in level0])
level2 = bundle([encode(state_after_choice) for choice in level1])
...
```

**Problem:** Still exponential in depth

## Key Questions

1. Can we encode the tree WITHOUT fully exploring it?
2. Can partial tree encoding still reveal good choices?
3. Can we transfer tree structure across similar puzzles?
4. Can we find compression patterns in the tree?

## Potential Compression Strategies

### 1. Prune Symmetric Branches
Many branches are equivalent up to permutation.
Encode one representative, map others to it.

### 2. Memoize Common Subtrees
If two paths reach the same state, they share a subtree.
Encode once, reference multiple times.

### 3. Encode "Shape" Not Content
The tree STRUCTURE might be more important than specific values.
Encode branching patterns, not cell values.

### 4. Learn Prototypical Trees
Cluster puzzles by their tree structure.
Encode prototype trees, match new puzzles to prototypes.

## The Experiment

Let's try:
1. Build (partial) search tree for our hard puzzle
2. Encode the tree structure
3. See if the encoding reveals patterns
4. Test if patterns transfer to similar puzzles

---

## EXPERIMENTAL RESULTS

### Raw (Position, Digit) Encoding

**Does not transfer:**
- Prediction accuracy across puzzles: 61.5% (baseline: 93.5%)
- Good prototype similarity: 0.2036 (low)

**Why?** Different puzzles have completely different solution paths.
The specific (position, digit) pairs are puzzle-specific.

### Abstract Feature Encoding

**TRANSFERS WELL:**
- Prediction accuracy: 75.7%
- Good prototype similarity: **0.7586** (high!)

**Abstract features used:**
1. Number of options at decision point
2. Block type (corner, edge, center)
3. Constraint tightness (how many cells already filled)

**Key insight:** The PATTERN of good decisions is similar:
- Good decisions tend to happen at certain constraint levels
- Good decisions have similar "shapes" in abstract space

### Implications for Compression

We CAN compress the tree by:
1. Encoding abstract features instead of raw values
2. Learning a "good decision prototype" from solved puzzles
3. Using the prototype to guide new puzzles

**This is essentially learned heuristics!**

The abstract encoding captures "what makes a decision good" in a
puzzle-independent way. This is compressible and transferable.

---

## Open Questions

1. Can we improve abstract features for better transfer?
2. Can we beat the baseline (always predict bad) for good predictions?
3. Can abstract prototypes guide solving (not just predict)?
