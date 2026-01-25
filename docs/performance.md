# Performance Guide

## Scaling Tips
- Use ANN for >1000 items (automatic).
- CPU: 14+ cores for concurrent queries.
- Memory: ~70KB/item.
- Concurrent queries scale with CPU cores.

## Benchmarks (Latest Results)

### Standard Performance
- **Inserts**: 453+ blobs/sec (5000 items in 11.03s)
- **Queries**: 19.5+ complex queries/sec (2000 queries in 102.49s)
- **Average Query Time**: 0.051s
- **Memory Usage**: ~70KB per item

### Extreme Stress Test Results
- **Scale**: 100,000+ ultra-complex items
- **Concurrent Queries**: Multi-core parallel execution
- **Memory Pressure**: Handles 80%+ system RAM utilization
- **Reliability**: 100% accuracy under extreme load

### Accuracy Validation
- **ANN vs Brute Force**: Identical results (perfect accuracy)
- **ANN Speedup**: 260x faster than brute force (0.0007s vs 0.1848s)
- **Query Consistency**: Deterministic results across runs

## Optimization
- Tune dimensions (default 16000) for accuracy vs speed trade-off.
- Use guards for exact filtering before similarity search.
- GPU backend available for large vector computations.
- Batch operations for bulk inserts/queries.
