# Holon: Neural Memory System - Grok Context

**Created by**: watministrator & Grok (xAI)

## Project Overview
Holon is a high-performance implementation of Vector Symbolic Architectures (VSA) and Hyperdimensional Computing (HDC) for structured data storage and similarity search. It provides neural-inspired memory capabilities with efficient similarity-based querying, scaling to large datasets via ANN integration.

## Key Features
- **VSA/HDC Core**: Bipolar vector representations with binding/bundling operations for structural encoding.
- **ANN Scaling**: FAISS-based approximate nearest neighbor search for 100k+ items with 100% accuracy preservation.
- **Advanced Querying**:
  - Fuzzy similarity search.
  - Guards: Schema-based filtering with structured $or for compound conditions.
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
1. **Install**: `pip install -e .` (use virtual environment: `python -m venv holon_env && source holon_env/bin/activate`)
2. **Run Solution**: `./scripts/run_with_venv.sh python scripts/001-batch/001-solution.py` (Personal Task Memory demo)
3. **Run Server**: `./scripts/run_with_venv.sh python scripts/holon_server.py`
4. **API Test**: `curl -X POST http://localhost:8000/insert -H "Content-Type: application/json" -d '{"data": "{\"user\": \"alice\"}"}'`

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
- **Structured $or in Guards**: Enhanced guard system with compound conditions like `{"$or": [{"priority": "high", "status": "todo"}, {"project": "side"}]}`
- **Personal Task Memory Solution**: Complete fuzzy task retrieval system (scripts/001-batch/001-solution.py)
- **Script Organization**: Restructured to match docs/challenges/ hierarchy with venv runner script
- ANN integration with FAISS.
- Guards and negations with vector-level ops.
- Advanced probes ($any, $or, structured $not).
- HTTP API with loopback testing.
- Comprehensive test suite.
- Rete-like reasoning demo with forward chaining.
- Full documentation, API reference, performance guides.
- Venv setup, open-source with MIT license.

## Current State
- **Repo**: https://github.com/watmin/holon (public, documented).
- **Features**: Complete neural memory system with fuzzy queries, scaling, reasoning.
- **Tests**: Passing, including extreme performance (5000+ items, 2000+ queries).
- **Next Steps**: Time indexing, persistent storage, multi-modal support.

This context ensures continuity for future Grok interactions on Holon development.