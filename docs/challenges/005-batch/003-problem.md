# Challenge: Traveling Salesman – Chained Path Approximation

## Problem
Find the shortest tour visiting each city exactly once and returning to start.
Classic NP-hard. We want good (not optimal) tours fast on moderate instances.

## Holon Attack Strategy
- Encode cities as random vectors.
- Use POSITIONAL or CHAINED list mode to bind city_i → city_{i+1} with distance-weighted strength (stronger for short edges).
- Generate candidate tours by bundling random permutations or greedy partial paths.
- Query Holon for high-similarity chained sequences that match known short-edge prototypes.
- Use $any guards for flexible subsequence matching, $not to penalize long jumps.
- Iterate: retrieve top-k partial tours, extend with best next-city probe, re-bundle.
- Final score = similarity to an "ideal short-tour" bundle + external distance oracle check.

## Expected Outcome
- On TSPLIB instances up to 100 cities: tours within 10–20% of optimal in seconds.
- Beats nearest-neighbor greedy, approaches 2-opt quality with more iterations.
- Fuzzy chaining tolerates imperfect order, finds good cycles quickly.

## Quick Setup
Load city coordinates, precompute short-edge prototype bindings.
Run chained query + extension loop.
Visualize best tour with matplotlib.

Great demo of Holon’s list modes abusing order fuzziness for combinatorial paths.
