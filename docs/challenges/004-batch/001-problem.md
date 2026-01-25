# Geometric VSA/HDC Sudoku Solver using Holon

Goal: Build a proof-of-concept Sudoku solver that leverages Holon's vector-symbolic / hyperdimensional computing (VSA/HDC) geometry instead of traditional backtracking.
The core idea is to encode the puzzle and constraints as a single high-dimensional vector (or a small set of them), then use similarity search, binding, bundling, guards, and negations to make valid completions "fall out" geometrically — i.e., candidates with the highest cosine similarity to ideal constraint patterns are chosen iteratively.

Focus for this first version: 9×9 classic Sudoku (symbols 1–9, standard row/column/3×3 block constraints).

### Puzzle Data Format
Use a simple nested list-of-lists structure (Python-friendly) or EDN-style map. Example partial puzzle (famous "easy" one):

The grid (rows as lists, nil/None for empty cells):

5 3 . . 7 . . . .
6 . . 1 9 5 . . .
. 9 8 . . . . 6 .
8 . . . 6 . . . 3
4 . . 8 . 3 . . 1
7 . . . 2 . . . 6
. 6 . . . . 2 8 .
. . . 4 1 9 . . 5
. . . . 8 . . 7 9

As a dict for Holon ingestion (recommended):

{
  "size": 9,
  "grid": [
    [5, 3, null, null, 7, null, null, null, null],
    [6, null, null, 1, 9, 5, null, null, null],
    [null, 9, 8, null, null, null, null, 6, null],
    [8, null, null, null, 6, null, null, null, 3],
    [4, null, null, 8, null, 3, null, null, 1],
    [7, null, null, null, 2, null, null, null, 6],
    [null, 6, null, null, null, null, 2, 8, null],
    [null, null, null, 4, 1, 9, null, null, 5],
    [null, null, null, null, 8, null, null, 7, 9]
  ]
}

(Use None/null for empties — convert to your preferred format during ingestion.)

### Requirements & Geometric Strategy

1. Encoding Ideas (your choice — explain your reasoning!)
   - Positions: Use permutations to encode (row, col) pairs (e.g. permute a base vector by row then column, or progressive permutation)
   - Symbols: 9 random nearly-orthogonal hypervectors (one per digit 1–9)
   - Cells: Bind position HV ⊙ symbol HV for filled cells
   - Grid: Bundle (sum/average) all cell HVs into one holistic grid vector
   - Constraints:
     - Precompute "ideal" row/column/block vectors (bundled superposition of all 9 symbols, possibly sequenced/permuted to reduce interference)
     - Use Holon guards to enforce uniqueness (no symbol appears twice → low dot product between same-symbol bindings in same constraint group)

2. Solving Approach (geometric / similarity-driven, not backtracking)
   - Start with partial grid vector
   - Iteratively:
     - Find the "most constrained" empty cell (e.g., fewest legal symbols via quick similarity probes)
     - Probe for each possible symbol 1–9:
       - Temporarily bind it into the grid
       - Recompute affected row/col/block projections
       - Score how well the new grid aligns with ideal constraint vectors (cosine similarity)
     - Choose the symbol with highest geometric alignment score
     - Commit if above threshold, or collect top-k candidates
   - Use negations ("NOT this symbol already in row") and guards (numeric thresholds on similarity) to prune
   - Goal: solutions should emerge as high-similarity attractors in hyperspace

3. Success Criteria
   - Successfully solve the example puzzle above
   - Show at least partial convergence geometrically (print similarity scores per iteration)
   - Handle a few empty cells correctly via similarity alone (even if not full solve on first try)
   - Bonus stretch goals (if it works well):
     - Solve a harder puzzle (more empties)
     - Show how the same encoding might scale toward 16×16 or 25×25 with higher dimensionality

### Deliverables
Please provide:
- Imports & Holon setup (e.g. CPUStore with suggested dimensionality 8192–16384?)
- Encoding strategy explanation (how positions, symbols, cells, constraints are turned into HVs)
- Code to ingest the puzzle (from dict/list → Holon memory)
- Iterative solving loop with similarity-based decision making
- Final grid result + similarity/confidence scores
- Any observations about where the geometry shines or struggles

Let's see how far pure VSA/HDC geometry can take us on Sudoku before we fall back to symbolic cleanup!
