# Performance Guide

See the main [README](../README.md) for overview and quick start.

## Hardware Context
**Battle-tested on**: Intel Ultra 7 (22 cores), 54GB RAM, Ubuntu 22.04

All benchmarks validated on production hardware with real-world data patterns and concurrent workloads.

## Scaling Tips
- Use ANN indexing for >1000 items (automatic activation)
- CPU: 14+ cores recommended for concurrent query scaling
- Memory: ~70KB per item with intelligent vector management
- GPU: RTX 4090+ for large-scale vector computations (CuPy backend)
- Concurrent queries scale linearly with CPU cores

## Battle-Tested Performance Benchmarks

### Core Performance Metrics
- **Inserts**: 453+ items/second (5000 items in 11.03s)
- **Queries**: 19.5+ complex queries/second (2000 queries in 102.49s)
- **Query Latency**: 0.051s average response time
- **Memory Usage**: ~70KB per item (int8 vectors, 16KB each)

### Extreme Stress Validation
- **Scale Tested**: 100,000+ ultra-complex items under memory pressure
- **Concurrent Load**: Handles 2000+ simultaneous queries without degradation
- **Memory Pressure**: Maintains performance with 80%+ system RAM utilization
- **Large Datasets**: Successfully processes datasets with 100K+ items
- **Fault Tolerance**: 100% accuracy maintained under extreme load conditions

### ANN Indexing Performance
- **Automatic Activation**: Switches to FAISS when >1000 items
- **Speedup**: 260x faster than brute-force similarity search
- **Accuracy**: 100% consistency with brute-force results
- **Query Time**: 0.0007s vs 0.1848s (brute-force)
- **Memory**: Efficient scaling for 10k-100k+ item datasets

### Scaling Characteristics

| Dataset Size | Query Performance | Memory Usage | Recommended Backend |
|-------------|-------------------|-------------|-------------------|
| < 1,000 items | Sub-millisecond | Minimal | CPU (brute-force) |
| 1k-10k items | Millisecond range | Moderate | CPU + ANN |
| 10k-100k items | Millisecond range | High | GPU + ANN |
| 100k-1M items | 15-20ms range | Very High | Qdrant |
| 1M+ items | 15-20ms range | Disk-based | Qdrant |

## Qdrant Backend (Persistent Storage)

For production workloads beyond in-memory limits, use `QdrantStore`:

### Performance at 1 Million Records

| Metric | Value |
|--------|-------|
| **Query Rate** | 59-64 queries/sec |
| **Query Latency** | 15-17 ms |
| **Storage Size** | 16 GB (4096d vectors) |
| **Bytes per Record** | ~16 KB |
| **Index Status** | HNSW (automatic) |

### Comparison: CPUStore vs QdrantStore at 1M Scale

| Backend | Query Rate | Latency | Storage |
|---------|-----------|---------|---------|
| CPUStore (brute-force) | 0.07 q/s | 14.7s | In-memory only |
| **QdrantStore (HNSW)** | **59-64 q/s** | **16ms** | Persistent disk |

**Qdrant is ~850x faster at 1M scale** due to HNSW indexing.

### Insert Performance

| Phase | Rate | Notes |
|-------|------|-------|
| Early (1K-10K) | ~260 items/sec | Before HNSW overhead |
| Mid (100K-500K) | ~200-260 items/sec | Index building begins |
| Late (500K-1M) | ~120-300 items/sec | Concurrent indexing |

### Storage Scaling

| Records | Storage (4096d) | Storage (1024d) |
|---------|-----------------|-----------------|
| 100K | ~1.6 GB | ~0.4 GB |
| 500K | ~8 GB | ~2 GB |
| 1M | ~16 GB | ~4 GB |
| 10M | ~160 GB | ~40 GB |

### Usage

```python
from holon import QdrantStore, HolonClient

# Persistent storage with namespacing
store = QdrantStore(
    collection="my_app",  # Namespace for isolation
    dimensions=4096,
    url="http://localhost:6333"
)
client = HolonClient(local_store=store)

# Same API as CPUStore
client.insert_json({"name": "Alice"})
results = client.search_json(probe={"name": "Alice"})

# Easy cleanup
store.drop_collection()  # Wipe entire namespace
```

### Running Qdrant

```bash
# Start with Docker Compose
docker-compose up -d

# Check status
curl http://localhost:6333/collections
```

### Accuracy Validation
- **ANN Consistency**: 100% identical results between ANN and brute-force
- **Deterministic Results**: Reproducible outputs across multiple runs
- **Query Validation**: All results pass threshold and guard filters correctly
- **Data Integrity**: Perfect fidelity for stored/retrieved data

## Advanced Optimization Techniques

### Vector Configuration
- **Dimensions**: Default 16,000 for optimal accuracy/speed balance
- **Dtype**: Int8 vectors (8x memory reduction vs float64)
- **Backend Auto-selection**: GPU when available, CPU fallback

### Query Optimization
- **Guards First**: Apply exact filters before expensive similarity search
- **Top-k Limiting**: Reduces ANN search scope for better performance
- **Threshold Tuning**: Filter low-similarity results early in pipeline
- **Negation Efficiency**: Vector-level exclusions avoid post-processing overhead

### Bulk Operations
- **Batch Inserts**: Defer ANN index rebuilds during bulk operations
- **Parallel Encoding**: Multi-core processing for large datasets
- **Memory Batching**: Process items in chunks to manage memory usage
- **Index Optimization**: Single ANN rebuild after bulk completion

### Hardware Acceleration
- **GPU Support**: CuPy backend for RTX 4090+ acceleration
- **SIMD Utilization**: Automatic CPU vector instruction usage
- **Memory Management**: Intelligent CPU/GPU vector transfer
- **Concurrent Scaling**: Multi-core query parallelism

### Production Deployment Tips
- **Pre-warm ANN**: Build indexes during low-traffic periods
- **Monitor Memory**: Track vector storage growth patterns
- **Batch Inserts**: Use bulk operations for data ingestion pipelines
- **Query Limits**: Set reasonable top_k limits (default: 10, max: 100)
- **Health Checks**: Monitor backend type and item counts via `/health`

### Performance Profiling
Use the built-in benchmarking tools:
```bash
# Quick performance test
./scripts/run_with_venv.sh python scripts/demos/test_accuracy.py

# Extreme stress testing
./scripts/run_with_venv.sh python scripts/tests/performance/extreme_query_challenge.py

# HTTP API stress test
./scripts/run_with_venv.sh python scripts/tests/performance/extreme_http_test.py
```

For more information about optimizing queries and batch operations, see the [API Reference](api_reference.md).
