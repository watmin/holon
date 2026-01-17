# Holon: Programmatic Neural Memory

**Authors**: watministrator & Grok (xAI)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Overview

Holon is a Python library implementing a "programmatic neural memory" system using Vector Symbolic Architectures (VSA) and Hyperdimensional Computing (HDC). It allows for the ingestion, encoding, and querying of structured data (JSON and EDN strings) through vector-based representations, enabling efficient similarity-based retrieval.

EDN (Extensible Data Notation) is fully supported, providing richer data structures than JSON including keywords (:keyword), sets (#{...}), symbols, booleans, nil, and complex nested structures. Every scalar value in EDN forms is atomized and encoded into the vector representation.

The core concept uses Vector Symbolic Architectures to encode structured data by preserving relationships: maps bind keys to values, sequences bundle items, sets aggregate with indicators. Each scalar gets a unique high-dimensional vector, and structural operations (binding/bundling) create representations that enable similarity-based querying of partial structures.

See [docs/](docs/) for additional documentation, API reference, and examples.

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