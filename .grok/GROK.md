# Holon: Neural Memory System - Grok Context

## Project Overview
Holon is a high-performance implementation of Vector Symbolic Architectures (VSA) and Hyperdimensional Computing (HDC) for structured data storage and similarity search. It provides neural-inspired memory capabilities with efficient similarity-based querying, scaling to large datasets via ANN integration.

## Key Features
- **VSA/HDC Core**: Bipolar vector representations with binding/bundling operations for structural encoding.
- **ANN Scaling**: FAISS-based approximate nearest neighbor search for 100k+ items with 100% accuracy preservation.
- **Advanced Querying**:
  - Fuzzy similarity search.
  - Guards: Schema-based filtering (presence checks).
  - Negations: Value-based exclusions, supporting lists and nested structures.
  - User-specified expressions: $any wildcards, $or disjunctions, $not markers.
- **Ephemeral Mode**: In-memory storage (Redis-like) with plans for durable (MongoDB+Qdrant-like) persistence.
- **HTTP API**: RESTful FastAPI server with JSON/EDN support.
- **Performance**: 70KB/item memory, 200-500 inserts/sec, sub-second queries.

## Architecture
- **holon/**: Core package (encoder, store, similarity, vector management).
- **scripts/**: Utilities, tests, server.
- **docs/**: Documentation, context.
- **tests/**: Unit and integration tests.
- **examples/**: Usage examples.

## Getting Started
1. **Install**: `pip install -e .`
2. **Run Server**: `python scripts/holon_server.py`
3. **API Test**: `curl -X POST http://localhost:8000/insert -H "Content-Type: application/json" -d '{"data": "{\"user\": \"alice\"}"}'`

## Running Tests
- **Unit Tests**: `python -m pytest tests/ -v`
- **Integration Tests**: `python scripts/run_all_tests.py`
- **Individual Scripts**:
  - Accuracy: `python scripts/test_accuracy.py`
  - Guards: `python scripts/test_guards_filtering.py`
  - Negation: `python scripts/test_negation.py`
  - Vector Tricks: `python scripts/test_vector_tricks.py`

## Development Notes
- **Encoding**: HDC preserves structure; vectors are bipolar int8 for efficiency.
- **Query Language**: JSON-based with special markers ($any, $not, $or) for advanced patterns.
- **Scaling**: ANN switches on at 1000+ items; vector operations for speed.
- **Future**: Time indexing, persistent storage, Prolog-like unification.
- **Constraints**: Arbitrary user data supported; special markers user-defined to avoid conflicts.

## Recent Changes
- ANN integration with FAISS.
- Guards and negations with vector-level ops.
- Advanced probes ($any, $or, structured $not).
- HTTP API with loopback testing.
- Comprehensive test suite.

This context ensures continuity for future Grok interactions on Holon development.