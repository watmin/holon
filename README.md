# Holon: Programmatic Neural Memory

**Authors**: watministrator & Grok (xAI)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Overview

Holon is a Python library implementing a "programmatic neural memory" system using Vector Symbolic Architectures (VSA) and Hyperdimensional Computing (HDC). It allows for the ingestion, encoding, and querying of structured data (JSON and EDN strings) through vector-based representations, enabling efficient similarity-based retrieval.

EDN (Extensible Data Notation) is fully supported, providing richer data structures than JSON including keywords (:keyword), sets (#{...}), symbols, booleans, nil, and complex nested structures. Every scalar value in EDN forms is atomized and encoded into the vector representation.

The core concept uses Vector Symbolic Architectures to encode structured data by preserving relationships: maps bind keys to values, sequences bundle items, sets aggregate with indicators. Each scalar gets a unique high-dimensional vector, and structural operations (binding/bundling) create representations that enable similarity-based querying of partial structures.

### Why "Holon"?
Named after Arthur Koestler's concept of a "holon"—a self-contained whole that is simultaneously a part of a larger whole. In Holon, each data item is a holon: independent yet entangled in the memory system through vector relationships, reflecting the interdependent, hierarchical nature of knowledge and memory.

See [docs/](docs/) for additional documentation and API reference. Check [examples/](examples/) for runnable code samples.

## Why Holon is Cool

- **Neural-Inspired**: Uses brain-like vector operations for memory that "entangles" data—partial cues retrieve wholes.
- **Blazing Fast**: ANN scaling makes 5000+ items query in milliseconds (500+ inserts/sec, 80+ queries/sec).
- **Flexible Queries**: Fuzzy search + guards/negations/wildcards/$or for precise, composable retrieval.
- **AI-Ready**: Perfect for LLM memory—deterministic, no hallucinations.
- **Unique Algebra**: Vector subtraction for exclusions—pure HDC magic.
- **Scale Tested**: 5000 blobs, 2000 concurrent queries with 100% accuracy.

## Quick Start

### Basic Usage
```python
from holon import CPUStore

store = CPUStore()
store.insert('{"name": "Alice", "role": "developer"}')
results = store.query('{"role": "developer"}')
print(f"Found {len(results)} developers")
```

### Advanced Queries
```python
# Wildcards
store.query('{"role": {"$any": true}}')  # Any role

# Guards
store.query('{"name": "Alice"}', guard={"role": "developer"})

# Negations
store.query('{"role": "developer"}', negations={"name": {"$not": "Alice"}})

# Disjunctions
store.query('{"$or": [{"role": "developer"}, {"role": "designer"}]}')
```

### EDN Support
```python
store.insert('{:user "alice" :actions [:login :edit]}', data_type='edn')
results = store.query('{:user "alice"}', data_type='edn')
```

### HTTP API
```bash
# Start server
python scripts/holon_server.py

# Insert
curl -X POST http://localhost:8000/insert -H "Content-Type: application/json" -d '{"data": "{\"event\": \"login\"}"}'

# Query
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" -d '{"probe": "{\"event\": \"login\"}"}'
```

### Complex Use Case: AI Agent Memory
```python
# Store conversation history
store.insert('{"session": "123", "user": "alice", "message": "How does HDC work?"}')
store.insert('{"session": "123", "user": "bot", "message": "HDC uses high-dim vectors"}')

# Find bot responses in session 123
results = store.query('{"user": "bot"}', guard={"session": "123"})
print(f"Bot responses in session 123: {len(results)}")
```

## Command Reference

### Installation
```bash
git clone https://github.com/watmin/holon.git
cd holon
pip install -e .
```

### Running Tests
```bash
# All tests
python -m pytest tests/

# Specific test
python scripts/test_accuracy.py

# Comprehensive test suite
python scripts/run_all_tests.py

# Extreme performance test
python scripts/extreme_query_challenge.py
```

### Starting the Server
```bash
python scripts/holon_server.py
# API available at http://localhost:8000
```

### Running Benchmarks
Experience Holon's performance on your machine:

```bash
# Quick accuracy test
python scripts/test_accuracy.py

# Comprehensive test suite
python scripts/run_all_tests.py

# Extreme stress test (5000 items, 1000 queries)
python scripts/extreme_query_challenge.py

# HTTP API stress test (2000 inserts/queries)
python scripts/extreme_http_test.py
```

**Sample Results** (on Intel Ultra 7, 54GB RAM):
- Extreme Challenge: 5000 items inserted in 9.79s (510/sec), 1000 queries in 12.22s (81/sec), avg 0.012s/query.

### Extending Holon
- **Add Encoders**: Subclass `Encoder` in `holon/encoder.py` for new data types.
- **Custom Markers**: Modify parsing in `cpu_store.py` for new query operators.
- **Vector Ops**: Extend `vector_manager.py` for custom algebras.
- **API Endpoints**: Add routes in `scripts/holon_server.py`.
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

## Getting Started

### Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/watmin/holon.git
cd holon
pip install -r requirements.txt
```

For development, install in editable mode:

```bash
pip install -e .
```

### Basic Usage

```python
from holon import CPUStore

# Create a store
store = CPUStore(dimensions=16000)

# Insert JSON data
data_id = store.insert('{"name": "Alice", "age": 30}', 'json')

# Query similar JSON data
results = store.query('{"name": "Alice"}', 'json')
for id, score, data in results:
    print(f"Score: {score}, Data: {data}")

# Insert EDN data with richer structures
edn_id = store.insert('{:name "Bob", :skills #{"clojure" "python"}}', 'edn')

# Query EDN data
edn_results = store.query('{:skills #{"python"}}', 'edn')
```

See `examples/basic_usage.py` for JSON examples and `examples/edn_usage.py` for EDN examples.

## Architecture

See [architecture.md](architecture.md) for detailed data flow and design.

## API Design

See [api_design.md](api_design.md) for the abstract store interface specification.

## Applications & Use Cases

Holon's VSA/HDC architecture enables powerful applications in data understanding, search, and AI systems:

### 🔍 **Semantic Search & Retrieval**
- **Document Similarity:** Find related documents by structural/content similarity
- **Code Search:** Locate functions/classes by behavioral patterns
- **Knowledge Discovery:** Uncover hidden relationships in data

### 🧠 **AI Memory Systems**
- **Long-term Memory:** Persistent storage of learned patterns
- **Contextual Recall:** Retrieve information based on partial cues
- **Memory Augmentation:** Enhance LLMs with structured memory

### 📊 **Data Integration & Analysis**
- **Schema Matching:** Automatically align different data structures
- **Anomaly Detection:** Identify structural outliers in datasets
- **Data Fusion:** Merge information from heterogeneous sources

### 🎯 **Recommendation & Personalization**
- **Content Recommendation:** Suggest items based on structural similarity
- **User Profiling:** Understand preferences through data patterns
- **Dynamic Adaptation:** Learn and adapt to user behavior patterns

### 🔬 **Research & Development**
- **Cognitive Architectures:** Build systems that learn like humans
- **Neural-Symbolic Integration:** Combine symbolic reasoning with neural processing
- **Explainable AI:** Provide interpretable similarity reasoning

## Performance & Limits

See [limits_and_performance.md](limits_and_performance.md) for detailed performance analysis, scaling limits, and probe effectiveness findings.