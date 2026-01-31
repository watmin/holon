# Code Structure Search Findings

Structural code search using Holon's VSA encoding and `$or` disjunctions.

## What Works

### Multi-Pattern Queries
Single query returns multiple node types:

```python
client.search_json(probe={
    "$or": [
        {"_type": "ClassDef"},
        {"_type": "FunctionDef"},
        {"_type": "Import"},
        {"_type": "For"},
        {"_type": "Try"}
    ]
})
```

This replaces 5 separate grep commands with one query.

### Context-Aware Filtering
Find methods across multiple classes:

```python
client.search_json(probe={
    "$or": [
        {"_type": "FunctionDef", "_in_class": "CPUStore"},
        {"_type": "FunctionDef", "_in_class": "Encoder"},
        {"_type": "FunctionDef", "_in_class": "HolonClient"}
    ]
})
```

### Method Call Pattern Matching
Find all call sites for specific methods:

```python
client.search_json(probe={
    "$or": [
        {"_type": "Call", "func": {"attr": "encode"}},
        {"_type": "Call", "func": {"attr": "query"}},
        {"_type": "Call", "func": {"attr": "insert"}}
    ]
})
```

### Accurate Coordinates
Every result includes verified `file:line:col` that points to actual source.

## Performance

| Metric | Value |
|--------|-------|
| Ingestion speed | ~800 nodes/sec |
| Structure size | ~180 bytes/node (flat encoding) |
| Holon codebase | 2,127 nodes from 12 files |
| Single probe query | ~300ms (brute force, no ANN) |
| **Bundled $or query** | **148ms for 3-way OR (23x faster than naive loop)** |

### $or Uses VSA Superposition

Instead of running N separate queries, `$or` branches are bundled into a single probe:

```python
# All branches encoded and superimposed into ONE vector
branch_vectors = [encode({"_type": "ClassDef"}),
                  encode({"_type": "FunctionDef"}),
                  encode({"_type": "Import"})]
bundled = sum(branch_vectors) / norm  # Single probe!
```

| Approach | Time | Speedup |
|----------|------|---------|
| 3 separate queries | 3484ms | baseline |
| Bundled superposition | 148ms | **23x faster** |

### Bulk Mode Required
For large codebases, must use bulk mode to defer ANN indexing:

```python
store.start_bulk_insert()
ingest_directory(client, 'path/')
store.end_bulk_insert()
```

Without bulk mode: ~100 nodes/sec (ANN rebuild per insert).

## What Doesn't Work

### Deep Nested Structure Matching
We deliberately flatten AST structures. This query won't work:

```python
# WON'T WORK - we don't store full nested AST
{"_type": "FunctionDef", "body": [{"_type": "Return", "value": {"_type": "Call"}}]}
```

The flattening is intentional - deep structures cause:
- 411KB+ per node (vs 180 bytes)
- 100 nodes/sec ingestion (vs 800)
- Memory exhaustion on large codebases

### Exact Text Matching
This is structural search, not text search. For exact strings:

```python
# Use grep for this, not Holon
grep -r "def encode" holon/
```

### Cross-File Analysis
Each node is independent. Can't query "functions that call X which is defined in Y".

### Ranking by Relevance
All matches have similarity ~0.0 because we're doing exact structural matching.
The `$or` returns union of matches, not ranked results.

## Comparison to grep/ripgrep

| Feature | grep/rg | Holon |
|---------|---------|-------|
| Exact text | ✓ Fast | ✗ Not designed for |
| Regex | ✓ | ✗ |
| Multi-pattern OR | Multiple runs | Single query |
| Structural matching | ✗ | ✓ |
| Context filtering | ✗ | ✓ `_in_class`, `_in_function` |
| AST-aware | ✗ | ✓ |

**Use grep when**: You know the exact text.
**Use Holon when**: You want structural patterns across node types with context filtering.

## Honest Assessment

### This is cool for:
1. "Find all methods in these 3 classes" - one query
2. "Find all Try blocks and their ExceptHandlers" - structural
3. "Find calls to .encode() inside any Encoder method" - context-aware
4. Building code navigation tools that understand structure

### This is NOT:
1. A replacement for grep/ripgrep for text search
2. Fast enough for real-time IDE integration (sub-100ms)
3. Able to understand semantic meaning of code
4. Cross-reference analysis (call graphs, data flow)

### The real value:
Expressive `$or` queries that would require multiple grep runs + post-processing.
One query, multiple patterns, context-aware filtering, verified coordinates.

## Known Optimization Opportunities

### Vectorized Similarity (Not Yet Implemented)
Current bottleneck: `normalized_dot_similarity` does per-item type conversion.

```python
# Current (slow): 2135 individual calls with .astype()
for item in stored_vectors:
    similarity = normalized_dot_similarity(probe, item)

# Better: Matrix multiplication
all_vectors = np.vstack(stored_vectors)  # Pre-stack once
similarities = all_vectors @ probe  # Single vectorized op
```

Estimated speedup: 10-50x for brute-force search.

## Future Potential

1. **Vectorized similarity**: Matrix multiplication instead of per-item loops
2. **Clojure/EDN support**: Same approach for S-expressions
3. **Pattern wildcards**: `{"_type": "Call", "func": {"attr": "$any"}}`
4. **Cross-file relations**: Index imports to enable "who uses this module"
5. **Incremental indexing**: Only re-index changed files

## Running the Demo

```bash
./scripts/run_with_venv.sh python scripts/demos/code_structure_search.py
```
