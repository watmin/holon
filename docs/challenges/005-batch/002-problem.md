# Challenge: Max-SAT – Superposed Satisfiability

## Problem
Given a CNF formula, find an assignment that satisfies as many clauses as possible.
NP-hard (even approximating within certain ratios is hard). We target "very good" solutions quickly.

## Holon Attack Strategy
- Encode each literal as variable_vector bound to polarity_vector (+1 or -1).
- Each clause = bundled sum of its literals. Full formula = bundled all clauses.
- Candidate assignment = bundled chosen literals (one per variable).
- Similarity score = dot-product overlap with formula vector after cleaning — approximates fraction of satisfied clauses.
- Use Holon query with $or for alternative literal choices per variable, $not to exclude known bad partial assignments.
- Search: random restart + local flips, or beam search keeping top-k bundled assignments, querying Holon for fitness each step.
- Add a small violation penalty vector (bundled negated clauses) for guided refinement.

## Expected Outcome
- On medium SAT instances (100–500 vars, 1000 clauses): 95%+ clauses satisfied in under 10 seconds.
- Outperforms basic random walk, competitive with simple local-search solvers.
- Fuzzy retrieval finds "near-miss" assignments that exact solvers would miss early.

## Quick Setup
Parse DIMACS .cnf file into Holon inserts.
Run beam/refinement loop querying similarity to formula vector.
Compare satisfied clause count to WalkSAT baseline.

Perfect for seeing how Holon turns symbolic satisfiability into analogical vector search.
