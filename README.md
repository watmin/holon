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

# Create store and client
store = CPUStore(dimensions=16000)
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

### What Works vs What Doesn't

**Holon excels at:**
- Fuzzy similarity search over structured data
- Top-k retrieval with semantic ranking
- Prototype learning and classification
- "Find similar to X" and "X but not Y" queries

**Holon cannot solve:**
- Constraint satisfaction (Sudoku, SAT, graph coloring)
- Problems requiring global coherence across all parts
- Exact matching where "close enough" isn't acceptable
- NP-hard optimization (despite creative attempts)

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
