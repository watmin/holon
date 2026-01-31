# 002-Batch Challenge Learnings

## Overview

This batch contained two challenges:
1. **001 - Raven's Progressive Matrices (RPM)**: Visual reasoning puzzles
2. **002 - Graph Matching**: Finding isomorphic/similar graph structures

Both challenges benefited significantly from the new kernel primitives added during the 004-batch Sudoku exploration.

---

## New Kernel Primitives Applied

### 1. `prototype(vectors, threshold)` - Pattern Extraction

**What it does**: Extracts the common pattern from a set of vectors by keeping dimensions that are consistent across examples.

**RPM Application**:
- Created rule prototypes (progression, xor, union) from example matrices
- Enabled rule classification without explicit labels
- Result: Progression prototype matched its family at 0.58 vs 0.19 for others

**Graph Matching Application**:
- Learned topology family signatures (star, cycle, tree, chain, complete)
- **100% classification accuracy** across 15 test graphs
- Enables unsupervised graph clustering

**Key Insight**: Prototype learning is extremely powerful for unsupervised classification. Given enough examples of a category, the prototype captures the "essence" of that category.

---

### 2. `difference(before, after)` - Change Detection

**What it does**: Computes what changed between two states, returning a vector highlighting additions and removals.

**RPM Application**:
- Extracted row-to-row transformations
- Captures "what changes" between matrix positions
- Could identify if transformation rules are consistent

**Graph Matching Application**:
- Computed structural changes between graph sizes (star_4 → star_5)
- Showed that star growth differs from cycle growth (0.03 similarity)
- Extracted "hub pattern" by computing star minus chain

**Key Insight**: Difference vectors can represent abstract concepts like "add a node" or "hub structure" even though these aren't explicitly encoded.

---

### 3. `blend(vec1, vec2, alpha)` - Weighted Interpolation

**What it does**: Creates a weighted combination of two vectors, enabling "in-between" queries.

**RPM Application**:
- Created hybrid rule patterns (progression-union blend)
- Useful for finding matrices with mixed characteristics

**Graph Matching Application**:
- Star-tree hybrid query returned both star and tree graphs at top
- Enables "find graphs that are EITHER star-like OR tree-like"

**Key Insight**: Blend enables fuzzy categorical queries. Instead of hard OR, you get soft similarity to both patterns.

---

### 4. `amplify(superposition, component, strength)` - Signal Boosting

**What it does**: Strengthens a component's presence in a superposition.

**RPM Application**:
- Boosted weak "progression" query with learned prototype
- Improved matching: 0.22 → 0.59 (2.6x improvement)
- Non-matching categories improved less (0.01 → 0.15)

**Graph Matching Application**:
- Star detection improved more than non-star detection
- Stars: +0.21, Cycles: +0.08, Trees: +0.05

**Key Insight**: Amplification provides proportional improvement - relevant items improve more than irrelevant ones, increasing precision.

---

### 5. `negate(superposition, component, method)` - Pattern Removal

**What it does**: Removes a component's influence from a superposition.

**RPM Application**:
- Excluded XOR patterns from general query
- XOR matrices scored -0.46, others scored -0.10
- Successfully separates "everything except X"

**Graph Matching Application**:
- Anomaly detection: Created "normal pattern" from all prototypes
- Anomaly graph ranked #1 as most unusual (score: 0.06 vs 0.25+ for normal)

**Key Insight**: Negation enables anomaly detection. By encoding "what's normal" and measuring distance from it, outliers naturally emerge.

---

## Architectural Observations

### Local vs HTTP Trade-offs

| Capability | Local | HTTP | Notes |
|------------|-------|------|-------|
| Basic insert/search | ✅ | ✅ | Both work well |
| Prototype learning | ✅ | ⚠️ | Requires vector access |
| Difference computation | ✅ | ⚠️ | Requires vector access |
| Blend queries | ✅ | ⚠️ | Requires vector access |
| Amplification | ✅ | ⚠️ | Requires vector access |
| Negation | ✅ | ⚠️ | Requires vector access |

### HTTP API Now Supports All Primitives ✅

After adding the new endpoints, HTTP provides full access to all primitives:

| Endpoint | Purpose | Status |
|----------|---------|--------|
| `POST /api/v1/vectors/encode` | Data → vector | ✅ Existing |
| `POST /api/v1/vectors/prototype` | Pattern extraction | ✅ Added |
| `POST /api/v1/vectors/difference` | Change detection | ✅ Added |
| `POST /api/v1/vectors/blend` | Fuzzy queries | ✅ Added |
| `POST /api/v1/vectors/amplify` | Signal boosting | ✅ Added |
| `POST /api/v1/vectors/negate` | Pattern removal | ✅ Added |
| `POST /api/v1/search/by-vector` | Vector search | ✅ Existing |
| `POST /api/v1/vectors/similarity` | Compare vectors | ✅ Existing |

### HTTP Usage Example

```python
# 1. Encode data to vectors
star_vecs = [encode({"topology": "star", ...}) for star_graph in stars]
cycle_vecs = [encode({"topology": "cycle", ...}) for cycle_graph in cycles]

# 2. Learn prototypes
star_proto = prototype(star_vecs, threshold=0.5)
cycle_proto = prototype(cycle_vecs, threshold=0.5)

# 3. Create fuzzy query
hybrid = blend(star_proto, cycle_proto, alpha=0.5)

# 4. Amplify signal
strong_query = amplify(weak_query, star_proto, strength=2.0)

# 5. Anomaly detection
normal_pattern = blend(star_proto, cycle_proto, 0.5)
anomaly_score = similarity(unknown_vec, normal_pattern)  # Low = anomaly

# 6. Search with computed vector
results = search_by_vector(hybrid, top_k=10)
```

### HTTP Test Results

All primitives verified working via HTTP:

- **Prototype Classification**: 100% accuracy (7/7 graphs)
- **Blend**: Star-cycle hybrid returns both families at top
- **Amplify**: Star scores 0.48 → 0.74 (+55%), others unchanged
- **Anomaly Detection**: Anomaly ranked #1 most unusual
- **Difference**: Successfully computes structural changes

---

## Performance Observations

| Operation | Time (local) | Notes |
|-----------|-------------|-------|
| Prototype from 4 vectors | <1ms | Very fast |
| Classification (15 graphs) | ~50ms | 100% accuracy |
| Blend query | <1ms | Instant |
| Anomaly detection | ~100ms | Full dataset scan |

---

## Recommendations

### For Local Use
The new primitives are immediately useful and powerful. Use them for:
- Unsupervised classification (prototype)
- Anomaly detection (negate from normal)
- Fuzzy queries (blend)
- Query refinement (amplify)

### For HTTP Use
Two options:

**Option A: Expose Vector Operations**
- Add endpoints for each primitive
- Client manages vectors
- Maximum flexibility, more network round-trips

**Option B: Server-Side Composition**
- Add high-level query modifiers
- Server manages vectors internally
- Less flexibility, fewer round-trips

**Recommendation**: Start with Option A for power users, then add Option B convenience endpoints based on common patterns.

---

## Summary

The 002-batch challenges demonstrated that:

1. **Prototype learning** enables unsupervised classification with high accuracy
2. **Difference vectors** capture abstract structural changes
3. **Blend** enables fuzzy categorical queries
4. **Amplify** improves precision proportionally
5. **Negate** enables powerful anomaly detection

For HTTP support, the server needs vector operation endpoints or server-side composition. The current HTTP API is sufficient for basic use cases but lacks the power of the new primitives.
