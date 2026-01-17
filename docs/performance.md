# Performance Guide

## Scaling Tips
- Use ANN for >1000 items (automatic).
- CPU: 14+ cores for concurrent queries.
- Memory: ~70KB/item.

## Benchmarks
- Inserts: 500+/sec
- Queries: 80+/sec
- Extreme: 5000 items, 2000 queries in ~21s

## Optimization
- Tune dimensions (default 16000).
- Use guards for exact filtering.
- GPU backend for large vectors.
