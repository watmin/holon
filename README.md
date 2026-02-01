# Holon: Hyperdimensional Memory for Structured Data

**Authors**: watministrator, Grok (xAI), & Claude (Anthropic)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

<div align="center">
<img src="assets/superposition-incantation.gif" alt="Superposition Incantation Demo">

*Reality doesn't fold itself. We make it fold.*
</div>

## What is Holon?

Holon is a Python library implementing **Vector Symbolic Architectures (VSA)** and **Hyperdimensional Computing (HDC)** for structured data. It encodes JSON/EDN into high-dimensional vectors, enabling:

- **Semantic similarity search** - Find similar structures, not just keywords
- **Fuzzy matching** - Partial matches and substructure queries
- **Prototype classification** - Learn categories from examples
- **Memory augmentation** - Persistent context for LLM-based agents

### Inspiration

Inspired by [Carin Meier's VSA talk](https://www.youtube.com/watch?v=j7ygjfbBJD0) on hyperdimensional computing in Clojure.

<div align="center">
<img src="assets/time-bending-lattices.gif" alt="Time-Bending Lattices Demo">

*Structured data encoded into geometry. Similarity becomes distance.*
</div>

## Quick Start

```python
from holon import CPUStore, HolonClient

# Create store and client (4096 dimensions recommended for most use cases)
store = CPUStore(dimensions=4096)
client = HolonClient(local_store=store)

# Insert data
client.insert_json({"name": "Alice", "role": "developer", "skills": ["python", "ml"]})
client.insert_json({"name": "Bob", "role": "designer", "skills": ["figma", "css"]})

# Similarity search
results = client.search_json(probe={"role": "developer"}, limit=5)
# → Finds Alice (high similarity) and Bob (lower similarity)

# Fuzzy matching with guards
results = client.search_json(
    probe={"skills": ["python"]},
    guard={"role": "developer"}  # Exact filter
)
```

## Installation

```bash
git clone https://github.com/watmin/holon.git
cd holon
python -m venv holon_env
source holon_env/bin/activate
pip install -e .
```

## Configuration

### Dimension Selection

| Use Case | Dimensions | Records/GB |
|----------|------------|------------|
| Simple documents (<20 fields) | 1024 | ~817K |
| Complex documents, time encoding | 4096 | ~233K |
| Very complex (100+ fields) | 8192 | ~119K |

**Key findings**:
- 512 dimensions fail at 100+ field documents. Use 1024+ for production.
- For **prototype classification**, lower dimensions (~1024) often outperform higher dimensions due to better generalization. Common wisdom (10K+) applies to storage, not classification.

See [Dimension Selection Guide](docs/dimension_selection.md) for benchmarks and capacity planning.

### Backend Selection

Holon supports multiple backends with different trade-offs:

| Backend | Speed | Accuracy | Best For |
|---------|-------|----------|----------|
| `cpu` (default) | **11K ops/sec** | Good | General use |
| `gpu` (CuPy) | **40x batch speedup** | Good | Large batch operations |
| `torchhd` | 300 ops/sec | **Best** | Accuracy-critical classification |

```python
# Default is CPU (fastest for typical workloads)
store = CPUStore(dimensions=4096)

# Explicit backend selection
store = CPUStore(backend="cpu")      # Default - fastest individual ops
store = CPUStore(backend="gpu")      # CuPy - for batch operations
store = CPUStore(backend="torchhd")  # Best accuracy (Level embeddings)

# Environment variable override
# HOLON_BACKEND=torchhd python my_script.py
```

**TorchHD Backend** (`pip install torch-hd`):
- Uses Level embeddings for numeric fields (200 ≈ 201, 200 ≠ 500)
- Better precision on classification tasks (85% vs 70% on API anomaly detection)
- **39x slower** than CPU due to GPU transfer overhead per operation
- Use when accuracy matters more than throughput

**CuPy GPU Backend** (`pip install cupy-cuda12x`):
- Fast for batch matrix operations (40x speedup)
- Same accuracy as CPU
- Individual ops slower due to transfer overhead
- Use for large-scale prototype learning, batch similarity

**The honest truth**: For typical insert/query workloads, CPU is fastest. GPU only helps with batch operations (1000+ items at once).

## Core Features

### Query Operators

```python
# Wildcards - match any value
client.search_json(probe={"role": {"$any": True}})

# Negations - exclude values
client.search_json(probe={}, negations={"role": {"$not": "admin"}})

# Disjunctions - OR logic
client.search_json(probe={"$or": [{"role": "dev"}, {"role": "designer"}]})

# Guards - exact post-query filtering
client.search_json(probe={"name": "Alice"}, guard={"status": "active"})

# Guard operators for complex filtering
client.search_json(
    probe={"type": "user"},
    guard={
        "age": {"$gte": 18, "$lt": 65},        # Numeric ranges
        "bio": {"$contains": "developer"},     # Substring match
        "role": {"$in": ["admin", "mod"]},     # Membership
        "$exists": {"email": True}             # Field presence
    }
)
```

### N-gram Encoding for Text

```python
# Fuzzy text matching with n-grams
client.insert_json({
    "content": {
        "_encode_mode": "ngram",
        "sequence": ["the", "quick", "brown", "fox"]
    }
})

# Partial phrase matches
results = client.search_json(probe={
    "content": {"_encode_mode": "ngram", "sequence": ["quick", "fox"]}
})
```

### Time Encoding

Holon can encode timestamps with circular (periodic) and positional (linear) components:

```python
# Use $time marker for temporal awareness
client.insert_json({
    "order_id": "12345",
    "customer": {"tier": "platinum"},
    "created_at": {"$time": 1706500000}  # Unix timestamp
})

# ISO strings also work
client.insert_json({
    "event": "login",
    "occurred_at": {"$time": "2024-01-29T10:30:00Z"}
})

# Control resolution (second, minute, hour, day)
client.insert_json({
    "log": {"$time": 1706500000, "$time_resolution": "minute"}
})

# Query with time similarity
results = client.search_json(probe={
    "customer": {"tier": "platinum"},
    "created_at": {"$time": 1706503600}  # ~1 hour later
})
# → Finds platinum orders from around the same time
```

Time encoding captures:
- **Circular patterns**: Same hour different days are similar, December wraps to January
- **Positional proximity**: Recent times score higher than old times
- **Combined queries**: Structure + time in one query

### EDN Support

```python
# Rich data types beyond JSON
client.insert('{:user "alice" :skills #{:python :ml}}', data_type='edn')
client.search(probe='{:skills #{:python}}', data_type='edn')
```

## Advanced Primitives

Holon provides kernel primitives for sophisticated vector operations:

### Pattern Learning

```python
# Learn prototypes from examples
star_graphs = [encode(g) for g in star_graph_examples]
star_prototype = store.prototype(star_graphs, threshold=0.5)

# Classify new graphs
similarity = normalized_dot_similarity(new_graph_vec, star_prototype)
```

### Vector Operations

```python
# Blend - fuzzy queries across categories
hybrid = store.blend(star_proto, tree_proto, alpha=0.5)

# Amplify - boost specific signals
boosted = store.amplify(query_vec, topic_proto, strength=2.0)

# Negate - "X but NOT Y" queries
filtered = store.negate(include_proto, exclude_proto)

# Difference - extract what changed
delta = store.difference(before_vec, after_vec)
```

## HTTP API

```bash
# Start server
./scripts/run_with_venv.sh python scripts/server/holon_server.py
```

### Endpoints (v1 API)

```bash
# Insert
curl -X POST http://localhost:8000/api/v1/items \
  -H "Content-Type: application/json" \
  -d '{"data": "{\"name\": \"Alice\"}", "data_type": "json"}'

# Search
curl -X POST http://localhost:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{"probe": "{\"name\": \"Alice\"}", "top_k": 5}'

# Vector operations
curl -X POST http://localhost:8000/api/v1/vectors/encode \
  -d '{"data": "{\"topic\": \"calculus\"}", "data_type": "json"}'

curl -X POST http://localhost:8000/api/v1/vectors/prototype \
  -d '{"vectors": [[1,-1,0,...], [0,1,-1,...]], "threshold": 0.5}'
```

See [docs/api_reference.md](docs/api_reference.md) for complete API documentation.

## Challenge Solutions

Validated implementations with honest assessments:

| Challenge | Description | Result |
|-----------|-------------|--------|
| **001-batch** | Task memory with fuzzy retrieval | ✅ Production-ready |
| **002-batch** | RPM geometric reasoning + Graph matching | ✅ 100% accuracy |
| **003-batch** | Quote finder with vector bootstrapping | ✅ 100% validation |
| **004-batch** | Sudoku constraint satisfaction | ❌ [VSA fundamentally cannot solve](scripts/challenges/004-batch/approaches/FINAL_ASSESSMENT.md) |
| **005-batch** | NP-hard optimization (3-coloring, SAT, TSP) | ⏳ Expected to fail (same reasons as Sudoku) |
| **006-batch** | LLM memory augmentation | ✅ [Ideal use case - 82% token savings](docs/challenges/006-batch/LEARNINGS.md) |
| **007-batch** | Multi-domain demonstrations (7 solutions) | ✅ [7/7 working, 100% fraud detection](scripts/challenges/007-batch/FINAL_STATUS.md) |
| **008-batch** | Production patterns (GPU, API anomaly detection) | ⏳ [1/6 complete, 95.9% precision](docs/challenges/008-batch/CHALLENGES.md) |

### Batch 008: Primitive Composition

Key finding from API anomaly detection challenge:

```python
# Basic prototype learning: 74% precision
attack_proto = store.prototype(attack_vectors)

# With primitive composition: 95.9% precision
attack_diff = store.difference(normal_proto, attack_proto)  # Extract unique features
enhanced = store.amplify(attack_proto, attack_diff, 0.5)    # Boost distinguishing signals
```

**Takeaway**: The algebraic operations (`difference`, `amplify`, `negate`, `blend`) aren't just utilities—they're essential for precision-critical classification.

### Batch 007 Highlights

Seven real-world use cases demonstrating Holon as a remote service:

| Solution | Domain | Key Result |
|----------|--------|------------|
| Rete Rule Engine | Business rules | Exact + fuzzy matching with truth maintenance |
| Code Understanding | Developer tools | 160 items indexed, AST + metadata search |
| Hierarchical Docs | Legal tech | Section hierarchy, cross-references, amendments |
| Event Sequences | Fraud detection | **100% accuracy** (5/5 k-NN classification) |
| Knowledge Graph | Information retrieval | Entity relations, influence queries |
| Medical Records | Healthcare | Symptom search, severity filtering |
| Scale Experiments | Capacity planning | See scale results below |

**Scale experiment results (HTTP-compatible, no local numpy):**
- Category saturation: 30% accuracy at 50 categories (k-NN struggles with many similar categories)
- Needle in haystack: **Rank 1 among 500 similar items** ✅
- Binding depth: No degradation up to depth 6 ✅
- Field dilution: 100% retention up to 100 fields ✅

### What Works vs What Doesn't

**Holon excels at:**
- Fuzzy similarity search over structured data
- Top-k retrieval with semantic ranking
- Prototype learning and classification
- "Find similar to X" and "X but not Y" queries
- Finding needles in haystacks (rank 1 among 500+ similar items)
- Deep nesting without signal loss (6+ levels)
- High field counts without dilution (100+ fields)

**Holon cannot solve:**
- Constraint satisfaction (Sudoku, SAT, graph coloring)
- Problems requiring global coherence across all parts
- Exact matching where "close enough" isn't acceptable
- NP-hard optimization (despite creative attempts)
- Many-category classification via k-NN (30% at 50 categories)

## Documentation

- [Project Assessment](docs/ASSESSMENT.md) - What works, what doesn't, what we learned
- [API Reference](docs/api_reference.md) - Complete HTTP and Python API
- [Encoding Guide](docs/encoding_guide.md) - How data becomes vectors
- [Architecture Decisions](docs/architecture/decisions/) - Design rationale
- [Contributing](docs/contributing.md) - Development setup

## Development

```bash
# Install dev dependencies
./scripts/run_with_venv.sh pip install -r requirements-dev.txt

# Run tests
./scripts/run_with_venv.sh python -m pytest tests/

# Pre-commit hooks
./scripts/run_with_venv.sh pre-commit install
```

## Key Concepts

- **Atomic Vectors**: Each scalar value gets a unique 16k-dimensional bipolar vector
- **Binding**: Element-wise multiplication combines related vectors (key × value)
- **Bundling**: Vector summation aggregates multiple vectors
- **Similarity**: Cosine similarity enables partial/substructure matching

## Why "Holon"?

Named after Arthur Koestler's concept - a self-contained whole that is simultaneously part of a larger whole. Each data item is independent yet entangled through vector relationships.

<div align="center">
<img src="assets/forbidden-binding-spell.gif" alt="Vector Operations Demo">

*From mystical runes to mathematical vectors. The power endures.*
</div>

---

MIT Licensed | [Documentation](docs/) | [Examples](examples/)
