# Holon: Neural Memory System - Project Context

## Project Overview
Holon is a high-performance implementation of Vector Symbolic Architectures (VSA) and Hyperdimensional Computing (HDC) for structured data storage and similarity search. It provides neural-inspired memory capabilities with efficient similarity-based querying.

## Core Architecture
- **VSA/HDC Foundation**: Bipolar vector representations with binding/bundling operations
- **Structural Encoding**: Recursive encoding preserving data relationships (maps, sequences, sets)
- **Similarity Search**: Cosine/dot product similarity with optimized algorithms
- **Backend Support**: CPU (NumPy) and GPU (CuPy) with auto-detection

## Key Components Implemented

### Core Engine (holon/)
- **vector_manager.py**: High-dimensional vector allocation and caching
- **encoder.py**: Structural encoding with binding operations
- **similarity.py**: Optimized similarity search with heap selection
- **cpu_store.py**: Main storage interface with CPU/GPU backend support
- **atomizer.py**: Data parsing and atomization for JSON/EDN

### API Layer (scripts/)
- **holon_server.py**: FastAPI REST API for HTTP access
- **performance_test.py**: Benchmarking and optimization testing

### Testing & Examples
- **tests/**: Unit tests with pytest
- **examples/**: JSON and EDN usage examples
- **scripts/**: Performance testing and API clients

## Performance Characteristics

### Benchmarks (16k dimensions)
- **Memory**: 70KB per item (int8 vectors, 16KB each)
- **Insertion**: 200-250 items/sec (parallel processing)
- **Query**: 0.002-0.01s per query (heap optimization)
- **Scalability**: Practical for 10k-50k items

### Optimizations Implemented
- **Heap Selection**: O(N log K) for top-k queries
- **Parallel Processing**: Multi-core encoding with ProcessPoolExecutor
- **Int8 Vectors**: 8x memory reduction vs float64
- **SIMD Operations**: NumPy leverages CPU vector instructions
- **GPU Ready**: CuPy integration for RTX 4090 acceleration

## Current State
- ✅ **Core VSA/HDC implementation** complete
- ✅ **CPU performance optimizations** implemented
- ✅ **GPU acceleration framework** ready
- ✅ **HTTP REST API** with FastAPI
- ✅ **JSON/EDN support** with type conversion
- ✅ **Comprehensive testing** and documentation
- ✅ **Repository organization** clean and documented

## Architecture Decisions Made

### Vector Representation
- **Bipolar {-1, 0, 1}** for HDC properties
- **16k dimensions** as default (configurable)
- **Int8 dtype** for memory efficiency

### Encoding Strategy
- **Recursive structural encoding** preserving relationships
- **Map binding**: key * value operations
- **Sequence bundling**: sum of encoded items
- **Set handling**: item aggregation with indicators

### Query Optimization
- **Heap-based top-k** for efficient selection
- **Parallel processing** for large datasets
- **Dot product similarity** with normalization

### API Design
- **RESTful endpoints**: /insert, /batch_insert, /query, /health
- **Content negotiation**: JSON/EDN input/output
- **Query limits**: Configurable top_k with system maximum
- **Error handling**: Comprehensive validation

## Scaling Limits Identified
- **10k items**: Excellent performance (milliseconds queries)
- **50k items**: Good performance with optimizations
- **100k+ items**: Requires ANN indexing for practicality
- **Memory bound**: 70KB/item with current encoding

## Technology Stack
- **Python 3.11+**
- **NumPy**: CPU vector operations
- **CuPy**: GPU acceleration
- **FastAPI**: HTTP API framework
- **pytest**: Testing framework

## Key Insights & Learnings

### Performance Bottlenecks
- **Encoding complexity**: Recursive operations on nested data
- **Memory allocation**: Frequent vector creations
- **Query scaling**: O(N) brute force becomes limiting

### Optimization Successes
- **Parallel processing**: 2.5x insertion speedup
- **Heap selection**: Massive query time reduction
- **Int8 vectors**: 8x memory efficiency
- **SIMD utilization**: Automatic CPU vector instruction usage

### VSA/HDC Validation
- **Structural preservation**: Relationships maintained in vectors
- **Similarity effectiveness**: Partial matching works for substructures
- **Scalability challenges**: Distributed representations need indexing at scale

## Future Development Path

### Immediate Next Steps
- **ANN Indexing**: Implement HNSW/FAISS for large datasets ✅ COMPLETED
- **GPU Optimization**: Advanced CuPy kernels
- **Durable Storage**: Database persistence layer

### Long-term Vision
- **Distributed Processing**: Multi-node VSA operations
- **Hybrid Approaches**: VSA + traditional indexing
- **Advanced Encoding**: Hierarchical and manifold-aware representations

## Proposed Improvements for AI Memory Use Case

### Enhanced Partial Matching
- **Description**: Boost substructure detection for snippet-based queries. Ensure "what do I remember about <snippet>" reliably finds containing blobs.
- **Implementation**: Modify `encoder.py` to weight partial overlaps higher; add query modes like "subquery" for subset matching.
- **Benefit**: Deeper entanglement for partial data retrieval.

### Hierarchical Encoding
- **Description**: Encode data at multiple levels (whole, sections, snippets) for better partial retrieval.
- **Implementation**: Create "summary vectors" for sub-parts in `encoder.py`, stored alongside main vectors. Multi-level similarity checks in queries.
- **Benefit**: Tiny snippets reliably pull full contexts.

### Memory API Primitives
- **Description**: Simple, composable API for AI memory operations.
- **Implementation**: Wrap Holon in functions like `remember(data)`, `recall(snippet)`, `forget(id)`. Make it lambda-friendly for deterministic workflows.
- **Benefit**: Easier integration with LLMs/agents for composable AI.

### Integration with LLMs
- **Description**: Use Holon as a "memory layer" for LLMs to enable deterministic, high-fidelity recall.
- **Implementation**: Hybrid mode: Store LLM interactions, retrieve similar past ones for consistency. Feed exact blobs to LLMs.
- **Benefit**: Fixes LLM stochasticity; enables reliable, composable AI memory.

## Advanced Probe Features (To Implement)

### Negation in Probes
- **Description**: Vector-level negations via subtraction, supports nested structures. Probe `{"user": "alice"}` with negations `{"meta": {"status": "failed"}}` subtracts nested pattern.
- **Why?**: Guards check presence; negations use HDC algebra for deep pattern exclusion.
- **Implementation**: Encode full negations dict, subtract from probe_vector; recursive data matching.
- **Benefit**: Advanced HDC operations at any depth, prepares for distributed filtering.

### Wildcard Probes ($any)
- **Description**: Use `$any` for wildcards. E.g., `{"user": "alice", "action": "$any"}` matches any action for alice.
- **Implementation**: Skip `$any` keys during encoding.
- **Benefit**: Flexible queries without enumerating values.

### Time Indexing and Queries
- **Description**: Support time-based queries (e.g., "events after timestamp T").
- **Implementation**: Add time fields to data; use two encoders: linear for absolute time (range queries), circular for periodic (daily/weekly patterns). Encode time separately, combine with data vector.
- **Benefit**: Temporal memory—query by recency or patterns (e.g., "what happened last week").

### Other Potential Features
- **Range Queries**: Support numerical ranges (e.g., age between 20-30).
- **Compound Logic**: AND/OR for complex conditions.
- **Substructure Boosting**: Weight partial matches higher for better snippet retrieval.

## Repository Structure
```
holon/              # Core package
├── __init__.py     # Package exports
├── cpu_store.py    # Storage interface
├── encoder.py      # Encoding engine
├── vector_manager.py # Vector management
├── atomizer.py     # Data parsing
└── similarity.py   # Query optimization

scripts/            # Utilities
├── holon_server.py # HTTP API
└── performance_test.py # Benchmarking

docs/               # Documentation
├── README.md       # API & usage
├── architecture.md # Technical design
├── api_design.md   # API specification
└── limits_and_performance.md # Performance guide

examples/           # Usage examples
tests/              # Unit tests
```

## Critical Files to Understand
- **holon/cpu_store.py**: Main interface and backend logic
- **holon/encoder.py**: Core VSA/HDC encoding implementation
- **holon/similarity.py**: Query optimization algorithms
- **scripts/holon_server.py**: HTTP API implementation
- **docs/limits_and_performance.md**: Performance characteristics

## Current Working State
- **All core functionality** implemented and tested
- **Performance optimizations** active and effective
- **API fully functional** with proper error handling
- **Documentation complete** with usage guides
- **Repository clean** and ready for collaboration

This context file ensures continuity if development sessions are interrupted, providing complete project state and technical foundation.