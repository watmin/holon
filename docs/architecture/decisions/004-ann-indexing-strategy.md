# ADR 004: FAISS ANN Indexing with Brute Force Fallback

## Status
Accepted

## Context
VSA/HDC similarity queries are computationally expensive:
- Cosine similarity: O(dimensions) per comparison
- Brute force: O(n × dimensions) for n items
- Target: Sub-millisecond queries for thousands of items

Need to maintain accuracy while achieving high performance.

## Decision
Implement hybrid indexing strategy:
- **FAISS ANN**: Approximate Nearest Neighbor for >1000 items
- **Brute Force**: Exact similarity for smaller datasets
- **Consistency Validation**: Automatic ANN vs brute force verification
- **Dynamic Switching**: Automatic ANN enable/disable based on dataset size

## Implementation Details

### ANN Configuration
- **Library**: FAISS (Facebook AI Similarity Search)
- **Algorithm**: Inner Product (cosine similarity)
- **Threshold**: 1000 items triggers ANN indexing
- **Dimensions**: Configurable (default 16,000)

### Consistency Assurance
- **Parallel Computation**: Both ANN and brute force run on queries
- **Result Comparison**: Scores must match within 0.001 tolerance
- **Failure Handling**: Fallback to brute force on ANN failures
- **Performance Monitoring**: Speedup ratios tracked and reported

### Memory Management
- **Lazy Building**: ANN index built only when needed
- **Invalidation**: Index cleared on new insertions (rebuilds on next query)
- **Bulk Mode**: Defers ANN rebuilds during rapid insertions

## Performance Results

### Benchmarks (Latest)
- **ANN Speedup**: 260x faster than brute force
- **Accuracy**: 100% ANN/brute force consistency
- **Threshold**: 1000 items optimal switching point
- **Memory**: ~70KB per stored item

### Scaling Characteristics
- **Small Datasets** (<1000): Brute force faster (no index overhead)
- **Large Datasets** (>1000): ANN essential for performance
- **Insertion Impact**: ANN rebuild cost amortized over queries

## Consequences

### Positive
- **Performance**: Enables real-time queries on large datasets
- **Accuracy**: Guaranteed consistency through validation
- **Scalability**: Linear growth with dataset size
- **Reliability**: Automatic fallback mechanisms

### Negative
- **Dependency**: FAISS library requirement
- **Memory**: Additional RAM for index storage
- **Insertion Penalty**: Index rebuilding on data changes
- **Complexity**: Dual code paths to maintain

### Mitigations
- **Optional Dependency**: Graceful degradation without FAISS
- **Bulk Operations**: Batch insertions to minimize rebuilds
- **Monitoring**: Built-in performance tracking
- **Testing**: Comprehensive ANN validation in test suite

## Configuration

```python
# Auto-selection (recommended)
store = CPUStore(dimensions=16000)  # Uses ANN when >1000 items

# Manual control
store.ann_index = None  # Force brute force
store._build_ann_index()  # Force ANN rebuild
```

## References
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [ANN Performance Tests](../../scripts/tests/performance/extreme_query_challenge.py)
- [Integration Tests](../../scripts/tests/integration/test_full_pipeline.py)
