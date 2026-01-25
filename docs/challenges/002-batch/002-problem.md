# VSA/HDC Geometric Approach to Graph Matching

Goal: Implement graph isomorphism/subgraph matching using Holon's VSA/HDC for geometric search in high-D space. Classically NP-hard, this becomes tractable via vector encoding: nodes as atomic vectors, edges as bindings (rotations), graphs as bundled superpositions. Matching reduces to cosine similarity (angle alignment) with fuzzy tolerances.

Data Structure (EDN for relational maps/sets):
- :graph-id (string)
- :nodes (set of keywords e.g. #{:A :B :C})
- :edges (set of maps e.g. #{{:from :A :to :B :label :connects} {:from :B :to :C :label :depends}})
- :type (keyword e.g. :directed :undirected)
- :attributes (map e.g. {:A {:color :red :weight 1} :B {:color :blue}})
- :subgraphs (optional set of subgraph IDs for hierarchical graphs)

Requirements:
1. Generate and ingest 15–30 synthetic graphs (small, 5–10 nodes/edges, varied structures like cycles, trees, or random).
2. Encode graphs to preserve geometry: Bind edges (from ⊙ to ⊙ label), bundle all into one HV per graph for holistic comparison.
3. Support queries like:
   - "Find graphs isomorphic to this one" (probe full structure, high similarity threshold >0.9 for near-orthogonality match)
   - "Subgraphs similar to {:edges #{{:from :X :to :Y}}}" but NOT directed (negation on type)
   - "Graphs with guard: at least 3 nodes, attribute color :red on any node (wildcard $any)"
   - "Disjunction: graphs like cycle OR tree structures" ($or on types/patterns)
4. Demonstrate geometric insight: For subgraph matching, use partial probes; show how noise tolerance (fuzzy) handles approximate isomorphisms.
5. Bonus: Detect anomalies (e.g., query for "graphs NOT similar to known good ones").

Show full code: imports, store setup, encoding strategy (permutations for sequences in edges, binding for associations), ingestion of EDN strings, and several query demos with results (interpret high scores as geometric alignment, low as orthogonality).
