# Holon: Programmatic Neural Memory

**Authors**: watministrator & Grok (xAI)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Coverage](https://img.shields.io/badge/coverage-83%25-green.svg)](https://github.com/watmin/holon)
[![Tests](https://img.shields.io/badge/tests-136%2F138%20passed-brightgreen.svg)](https://github.com/watmin/holon)
[![Documentation](https://img.shields.io/badge/docs-available-blue.svg)](docs/)
[![Research](https://img.shields.io/badge/research-VSA%2FHDC-red.svg)](docs/rpm_geometric_solution_findings.md)

<div align="center">
<img src="assets/superposition-incantation.gif" alt="Superposition Incantation Demo">

<em>Reality doesn't fold itself. We make it fold.</em>
</div>

## Inspiration

Holon draws heavy inspiration from Vector Symbolic Architectures (VSA) — hyperdimensional computing that binds and bundles symbols in high dimensions, much like the brain's probabilistic representations.

This talk by Carin Meier was the spark that got this project started:

[![Vector Symbolic Architectures In Clojure by Carin Meier](https://img.youtube.com/vi/j7ygjfbBJD0/maxresdefault.jpg)](https://www.youtube.com/watch?v=j7ygjfbBJD0)

*Watching this felt like seeing the algebra of thought laid bare. I wanted that elegance in Clojure — but chose Python for Grok collaboration and broader adoption, with EDN support as my functional programming compromise.*

## How Holon Works

Holon encodes structured JSON/EDN into high-dimensional bipolar vectors using recursive binding (keys × values) and bundling (sum + threshold).
Supports positional, chained, n-gram, and bundle list modes, guards, negations, wildcards, and mathematical primitives.

**Mathematical Primitives**: Core VSA/HDC capabilities for mathematical pattern recognition including convergence rates, iteration complexity, frequency domains, amplitude scales, power-law exponents, clustering coefficients, topological distance, and self-similarity measures.

<div align="center">
<img src="assets/time-bending-lattices.gif" alt="Time-Bending Lattices Demo">

<em>We don't just store data. We weave it into something that remembers itself.</em>
</div>

## Overview

Holon is a Python library implementing a "programmatic neural memory" system using Vector Symbolic Architectures (VSA) and Hyperdimensional Computing (HDC). It allows for the ingestion, encoding, and querying of structured data (JSON and EDN strings) through vector-based representations, enabling efficient similarity-based retrieval.

EDN (Extensible Data Notation) is fully supported, providing richer data structures than JSON including keywords (`:keyword`), sets (`#{...}`), symbols, booleans, nil, and complex nested structures. Every scalar value in EDN forms is atomized and encoded into the vector representation.

The core concept uses Vector Symbolic Architectures to encode structured data by preserving relationships: maps bind keys to values, sequences bundle items, sets aggregate with indicators. Each scalar gets a unique high-dimensional vector, and structural operations (binding/bundling) create representations that enable similarity-based querying of partial structures.

For example, `{"user": "alice", "action": "login"}` and `{:user "alice" :actions ["login"]}` are encoded with structural fidelity.

### Mathematical Primitives

Beyond structural encoding, Holon provides fundamental mathematical primitives for pattern recognition through the unified client interface:

```python
# Mathematical primitives
vector = client.encode_mathematical("convergence_rate", 0.85)

# Mathematical composition (bind/bundle operations)
composed_vector = client.compose_vectors("bind", [vector1, vector2])

# Structural data encoding (original functionality)
vector = client.encode_vectors_json({"user": "alice"})
```

**Available Primitives:**
- **Convergence Analysis**: `convergence_rate` - Mathematical stability
- **Computational Complexity**: `iteration_complexity` - Algorithmic depth
- **Wave Phenomena**: `frequency_domain` - Physical frequency
- **Energy Scales**: `amplitude_scale` - Magnitude properties
- **Network Topologies**: `power_law_exponent` - Scale-free properties
- **Local Connectivity**: `clustering_coefficient` - Neighborhood structure
- **Path Properties**: `topological_distance` - Distance metrics
- **Fractal Properties**: `self_similarity` - Self-similar patterns

These primitives enable semantic similarity for mathematical patterns, going beyond structural matching to understand mathematical relationships.

### Why "Holon"?
Named after Arthur Koestler's concept of a "holon"—a self-contained whole that is simultaneously a part of a larger whole. In Holon, each data item is a holon: independent yet entangled in the memory system through vector relationships, reflecting the interdependent, hierarchical nature of knowledge and memory.

## Research Validation

Comprehensive validation results:
- **RPM geometric reasoning**: 100% accuracy on implemented rules ([RPM Findings](docs/rpm_geometric_solution_findings.md))
- **Graph topology recognition**: 100% family clustering ([Challenge 2 Assessment](docs/challenge_2_gaps_and_improvements.md))
- **Mathematical primitives**: 8 primitives for semantic encoding ([Mathematical Primitives Findings](docs/challenge_2_mathematical_primitives_findings.md))
- **Hybrid text search**: 75% F1 score ([Quote Finder Improvements](docs/quote_finder_improvements.md))
- **Geometric constraint satisfaction**: Sudoku solver planned ([Sudoku Findings](docs/sudoku_geometric_solution_findings.md))

## Use Cases

VSA/HDC systems like Holon enable:
- **Semantic search** through vector similarity rather than keyword matching
- **Geometric reasoning** for pattern completion and rule learning
- **Hybrid AI systems** combining symbolic and neural approaches
- **Memory augmentation** for LLM-based agents

## Implementation Status

Challenge solutions demonstrating capabilities:
- **Task memory system** (fuzzy retrieval with complex filtering) - ✅ Production-ready
- **RPM solver** (geometric rule learning, 100% accuracy on 4 implemented rules) - ✅ Complete
- **Graph matching** (VSA/HDC topology recognition, 100% family clustering) - ✅ Complete
- **Quote finder** (hybrid VSA + traditional text search, 75% F1 score) - ✅ Working hybrid system
- **Sudoku solver** (geometric constraint satisfaction) - 📋 Planned implementation

See [docs/](docs/) for detailed documentation, API reference, and [architecture decisions](docs/architecture/decisions/). Check [examples/](examples/) for runnable code samples.


## Quick Start

### Basic Usage
```python
from holon import CPUStore, HolonClient

# Create store and client
store = CPUStore()
client = HolonClient(local_store=store)

# Insert and query data
client.insert_json({"name": "Alice", "role": "developer"})
results = client.search_json(probe={"role": "developer"})
print(f"Found {len(results)} developers")
```

This inserts JSON data like `{"name": "Alice", "role": "developer"}` and queries for similar structures.

### Advanced Queries
```python
# Wildcards - match any value for a field
client.search_json(probe={"role": {"$any": true}})  # Any role

# Guards - exact post-query filtering
client.search_json(probe={"name": "Alice"}, guard={"role": "developer"})

# Negations - exclude specific values
client.search_json(probe={"role": "developer"}, negations={"name": {"$not": "Alice"}})

# Disjunctions - OR logic in query probe
client.search_json(probe={"$or": [{"role": "developer"}, {"role": "designer"}]})
```

### Complex Guard Syntax with $or Logic

Guards support sophisticated compound conditions using structured `$or` for powerful filtering:

```python
# Complex OR conditions in guards
results = client.search_json(probe={}, guard={
    "$or": [
        {"priority": "high", "status": "todo"},     # High priority TODO items
        {"project": "side", "category": "urgent"}   # OR urgent side projects
    ]
})

# Nested OR conditions for hierarchical filtering
results = client.search_json(probe={"project": "work"}, guard={
    "status": "active",
    "tags": {
        "$or": [
            {"$any": True},  # Any tagged items
            ["urgent"]       # OR items with urgent tag array
        ]
    }
})

# Combined with negations for precise filtering
results = client.search_json(
    probe={"project": "side"},
    guard={
        "$or": [
            {"priority": "high"},
            {"priority": "medium", "status": "todo"}
        ]
    },
    negations={"status": {"$not": "waiting"}}
)
```

### Query Pattern Examples
```python
# Fuzzy similarity search
client.search_json(probe={"title": "prepare presentation"})

# Exact structural matching with guards
client.search_json(probe={"role": "developer"}, guard={"status": "active"})

# Complex compound conditions
client.search_json(probe={}, guard={
    "$or": [
        {"priority": "high", "status": "todo"},
        {"project": "personal", "due": "2026-01-25"}
    ]
})

# Context-aware filtering
client.search_json(probe={"context": ["computer"]}, guard={"priority": "high"})
```

### EDN Support

Holon fully supports **EDN (Extensible Data Notation)** for richer, more expressive data structures beyond JSON:

```python
# Keywords (prefixed with :) - self-evaluating identifiers
client.insert('{:user "alice" :role :admin}', data_type='edn')

# Sets - unique collections with #{} syntax
client.insert('{:name "alice" :skills #{"clojure" "python" "ml"}}', data_type='edn')

# Symbols - unquoted identifiers for domain-specific meanings
client.insert('{:event login :timestamp (java.util.Date.)}', data_type='edn')

# Rich nested structures
client.insert('''
{:user {:name "alice" :id 123}
 :actions [{:type :login :time "2024-01-01"}
           {:type :edit :resource :profile}]
 :metadata {:source "web" :version "1.2"}}
''', data_type='edn')

# Query with EDN syntax
results = client.search(probe='{:user {:name "alice"}}', data_type='edn')
```

**EDN Advantages over JSON:**
- **Keywords**: `:user`, `:admin` - domain-specific identifiers without string overhead
- **Sets**: `#{:clojure :python :ml}` - unique collections with semantic meaning
- **Symbols**: `login`, `edit` - unquoted identifiers for cleaner syntax
- **Rich Primitives**: Built-in support for dates, UUIDs, and custom tagged elements
- **Complex Nesting**: Maps within maps, sets within vectors, etc.

All EDN data is atomized and encoded into the same high-dimensional vector space as JSON, enabling cross-format similarity queries.

### HTTP API
```bash
# Start server
python scripts/server/holon_server.py

# Insert
curl -X POST http://localhost:8000/insert -H "Content-Type: application/json" -d '{"data": "{\"event\": \"login\"}"}'

# Query
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" -d '{"probe": "{\"event\": \"login\"}"}'
```

### Complex Use Case: AI Agent Memory
```python
# Store conversation history
client.insert_json({"session": "123", "user": "alice", "message": "How does HDC work?"})
client.insert_json({"session": "123", "user": "bot", "message": "HDC uses high-dim vectors"})

# Find bot responses in session 123
results = client.search_json(probe={"user": "bot"}, guard={"session": "123"})
print(f"Bot responses in session 123: {len(results)}")
```

### Advanced Guards with Structured OR
```python
# Complex compound conditions using structured $or
results = client.search_json(probe={}, guard={
    "$or": [
        {"priority": "high", "status": "todo"},  # High priority tasks that are todo
        {"project": "side", "category": "urgent"} # OR urgent side projects
    ]
})
print(f"High priority todo OR urgent side projects: {len(results)}")
```

### Task Memory Demo
```bash
./scripts/run_with_venv.sh python scripts/challenges/001-batch/001-solution.py
```

## Command Reference

### Installation
```bash
git clone https://github.com/watmin/holon.git
cd holon
python -m venv holon_env
source holon_env/bin/activate  # Linux/Mac
pip install -e .
```

### Running Tests
```bash
# All tests (use venv runner)
./scripts/run_with_venv.sh python -m pytest tests/

# Run personal task memory solution
./scripts/run_with_venv.sh python scripts/challenges/001-batch/001-solution.py

# Other scripts
./scripts/run_with_venv.sh python scripts/demos/test_accuracy.py
./scripts/run_with_venv.sh python scripts/tests/integration/run_all_tests.py
./scripts/run_with_venv.sh python scripts/tests/performance/extreme_query_challenge.py
```

### Code Quality (Pre-commit Hooks)
```bash
# Install development dependencies (use venv script)
./scripts/run_with_venv.sh pip install -r requirements-dev.txt

# Install pre-commit hooks (automatic code quality)
./scripts/run_with_venv.sh pre-commit install

# Run quality checks on all files
./scripts/run_with_venv.sh pre-commit run --all-files

# Or run on just staged files (happens automatically on commit)
./scripts/run_with_venv.sh pre-commit run
```

### Starting the Server
```bash
python scripts/server/holon_server.py
# API available at http://localhost:8000
```

### Running Benchmarks
Experience Holon's performance on your machine:

```bash
# Quick accuracy test
./scripts/run_with_venv.sh python scripts/demos/test_accuracy.py

# Comprehensive test suite
./scripts/run_with_venv.sh python scripts/tests/integration/run_all_tests.py

# Extreme stress test (5000 items, 1000 queries)
./scripts/run_with_venv.sh python scripts/tests/performance/extreme_query_challenge.py

# HTTP API stress test (2000 inserts/queries)
./scripts/run_with_venv.sh python scripts/tests/performance/extreme_http_test.py
```


### Extending Holon
- **Add Encoders**: Subclass `Encoder` in `holon/encoder.py` for new data types.
- **Custom Markers**: Modify parsing in `cpu_store.py` for new query operators.
- **Vector Ops**: Extend `vector_manager.py` for custom algebras.
- **API Endpoints**: Add routes in `scripts/server/holon_server.py`.
- **Tests**: Add to `tests/` following pytest conventions.

See [docs/](docs/) for API docs and examples.

## Key Concepts

- **Structural Encoding**: Data structures are encoded recursively preserving relationships (maps bind keys to values, sequences bundle items, etc.).
- **Atomic Vectors**: Each scalar value (strings, numbers, keywords, etc.) gets a unique high-dimensional vector (16k dimensions, bipolar values {-1, 0, 1}).
- **Binding**: Element-wise multiplication combines related vectors (e.g., key * value).
- **Bundling**: Vector summation aggregates multiple vectors with thresholding to maintain bipolarity.
- **Similarity Querying**: Encode probe structures and compute cosine similarity against stored data vectors for partial/substructure matching.

## Long-Term Goals

1. **Local CPU Mode**: In-memory operations using CPU for vector computations.
2. **Local GPU Mode**: Accelerated in-memory operations using GPU for large-scale vector processing.
3. **Remote Service Mode**: Distributed backend using traditional databases (e.g., MongoDB) and vector stores (e.g., Qdrant) via an HTTP layer.

From the user's perspective, Holon provides an abstract "store" interface for inserting and querying data, abstracting away the underlying backend implementation.

### Development
```bash
# Install development dependencies
./scripts/run_with_venv.sh pip install -r requirements-dev.txt

# Install pre-commit hooks
./scripts/run_with_venv.sh pre-commit install

# Run tests
./scripts/run_with_venv.sh python -m pytest tests/
```

## Getting Started

### Installation

Clone and install in a virtual environment:

```bash
git clone https://github.com/watmin/holon.git
cd holon
python -m venv holon_env
. holon_env/bin/activate  # Linux/Mac
pip install -r requirements.txt
pip install -e .
```

### Basic Usage

```python
from holon import CPUStore, HolonClient

# Create store and client (works locally or remotely)
store = CPUStore(dimensions=16000)
client = HolonClient(local_store=store)

# Insert JSON data
data_id = client.insert_json({"name": "Alice", "age": 30})

# Query similar JSON data
results = client.search_json(probe={"name": "Alice"})
for result in results:
    print(f"Score: {result['score']}, Data: {result['data']}")

# Insert EDN data with richer structures
edn_id = client.insert('{:name "Bob", :skills #{"clojure" "python"}}', data_type="edn")

# Query EDN data
edn_results = client.search(probe='{:skills #{"python"}}', data_type="edn")
```

See `examples/basic_usage.py` for JSON examples and `examples/edn_usage.py` for EDN examples.

## Architecture

See [architecture.md](architecture.md) for detailed data flow and design.

## API Design

See [api_design.md](api_design.md) for the abstract store interface specification.

## Research

Comprehensive validation results available in docs:
- [RPM Geometric Reasoning](docs/rpm_geometric_solution_findings.md) - 100% accuracy on geometric rule learning
- [Graph Topology Recognition](docs/challenge_2_gaps_and_improvements.md) - 100% family clustering breakthrough
- [Hybrid Text Search](docs/quote_finder_improvements.md) - 75% F1 score validation
- [Geometric Constraint Satisfaction](docs/sudoku_geometric_solution_findings.md) - Analysis and planned implementation
- [Comprehensive Assessment](docs/comprehensive_challenge_assessment.md) - All challenges evaluated

## Challenge Implementations

Challenge solutions available in scripts/challenges/:
- **Batch 1**: Production-ready fuzzy task retrieval system
- **Batch 2**: Complete RPM geometric reasoning (100% accuracy) + VSA/HDC graph matching (100% topology recognition)
- **Batch 3**: Hybrid quote finder (75% F1 score)
- **Batch 4**: Not yet implemented (will use Holon for advanced reasoning)
- **Batch 5-6**: Additional challenge implementations

MIT licensed. FastAPI server included.

<div align="center">
<img src="assets/forbidden-binding-spell.gif" alt="Vector Operations Demo">

<em>From mystical runes to mathematical vectors. The power endures.</em>
</div>

## Performance & Limits

See [limits_and_performance.md](limits_and_performance.md) for detailed performance analysis, scaling limits, and probe effectiveness findings.
