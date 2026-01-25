# Geometric VSA/HDC Solution for Raven's Progressive Matrices

Goal: Use Holon's VSA/HDC memory to encode and solve simplified Raven's Progressive Matrices (RPM)—a classic abstract reasoning task. RPM involves inferring rules in a 3x3 grid of panels (e.g., shape progressions) to find the missing panel. Geometrically, this turns combinatorial rule search into vector similarity: encode the matrix as bundled/bound structures, then query for the "best-aligned" completion in hyperspace.

Data Structure (EDN preferred for sets and keywords):
- :matrix-id (string)
- :panels (map of positions to panel maps, e.g. {:row1-col1 {:shapes #{:circle :triangle}, :count 2, :progression :add-one, :color :black}})
- :rule (keyword e.g. :progression, :xor, :union) – for training/supervision
- :attributes (set of keywords e.g. #{:shape :size :count})
- :missing-position (keyword e.g. :row3-col3)

Requirements:
1. Generate and ingest 10–20 synthetic RPM matrices (3x3 grids with simple rules like shape count increasing by 1 per row, or XOR on presence).
2. Encode panels as nested structures to leverage Holon's binding (geometric association) and bundling (superposition for rows/columns).
3. Support queries like:
   - "Find the missing panel for this incomplete matrix" (probe with partial structure, use similarity to align rules)
   - "Matrices with progression rule but NOT xor" (negation on rule)
   - "Similar to this matrix but with guard on attribute :color = :black"
   - "Wildcard: any matrix where row2 has shapes similar to #{:square *}" (fuzzy wildcard on sets)
4. Demonstrate geometric tractability: Compute "transformation" via query similarity (e.g., unbind row deltas), select top-3 candidates.
5. Bonus: Simulate solving an unseen matrix by querying for nearest geometric analog.

Show full code: imports, CPUStore init (high dims e.g. 16000), encoding strategy (how to bind positions/attributes, bundle rows), ingestion, and 3–4 query examples with ranked results (cosine scores as geometric alignment measure).
