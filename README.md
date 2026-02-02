# Holon: Hyperdimensional Memory for Structured Data

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

<div align="center">
<img src="assets/superposition-incantation.gif" alt="Superposition Incantation">

*Reality doesn't fold itself. We make it fold.*

Inspired by [Carin Meier's VSA talk](https://www.youtube.com/watch?v=j7ygjfbBJD0) on hyperdimensional computing.
</div>

Holon encodes **JSON structure into vectors**, enabling similarity search over structured data. Unlike semantic embeddings that capture meaning, Holon captures *structure* - keys, nesting, relationships become geometry.

## Quick Start

```python
from holon import CPUStore, HolonClient

store = CPUStore(dimensions=4096)
client = HolonClient(local_store=store)

# Insert structured data
client.insert_json({"name": "Alice", "role": "developer", "skills": ["python", "ml"]})
client.insert_json({"name": "Bob", "role": "designer", "skills": ["figma", "css"]})

# Similarity search - finds structurally similar documents
results = client.search_json(probe={"role": "developer"}, limit=5)
# → Alice (high similarity), Bob (lower)

# Fuzzy matching with guards
results = client.search_json(
    probe={"skills": ["python"]},
    guard={"role": "developer"},      # Exact filter
    negations={"status": "inactive"}  # Exclude
)

# Time-aware search - "documents from around that time"
client.insert_json({"event": "deploy", "at": {"$time": 1706500000}})
results = client.search_json(probe={"at": {"$time": 1706503600}})  # ~1hr later
# → Finds events from around the same time (similarity, not range query)

# Sequence encoding for event patterns
client.insert_json({
    "session": "abc",
    "events": {"$mode": "chained", "sequence": ["login", "transfer", "logout"]}
})
results = client.search_json(probe={
    "events": {"$mode": "chained", "sequence": ["login", "transfer"]}
})
# → Finds sessions with similar event patterns
```

<div align="center">
<img src="assets/time-bending-lattices.gif" alt="Time-Bending Lattices">

*Structured data encoded into geometry. Similarity becomes distance.*
</div>

## What Makes Holon Different

| Traditional Vector DB | Holon |
|----------------------|-------|
| Semantic embeddings (meaning) | Structural embeddings (shape) |
| "Find similar text" | "Find similar JSON structures" |
| Requires ML models | Pure math (no models) |
| Opaque vectors | Composable primitives |

**The genuine insight**: VSA encodes *structure* as *geometry*. You can `difference()` two configs and get the delta as a vector. You can `negate()` expected changes. You can `amplify()` security fields. These operations *compose mathematically*.

## Installation

```bash
git clone https://github.com/watmin/holon.git
cd holon && pip install -e .
```

## Core Primitives

Everything in Holon is built from these kernel operations:

| Category | Primitives |
|----------|-----------|
| **Encoding** | `encode_data(json)`, `encode_sequence(items, mode)` |
| **VSA Ops** | `bind(a,b)`, `unbind(ab,a)`, `bundle([vecs])`, `permute(v,k)` |
| **Learning** | `prototype([examples])`, `prototype_add(p,ex,n)`, `cleanup(noisy,codebook)` |
| **Manipulation** | `difference(a,b)`, `amplify(v,sig,str)`, `negate(v,x)`, `blend(a,b,α)`, `resonance(v,ref)` |

### Quick Examples

```python
# Learn a prototype from examples
dev_vecs = [store.encoder.encode_data(d) for d in developer_profiles]
dev_prototype = store.prototype(dev_vecs)

# Classify new data
new_vec = store.encoder.encode_data(new_profile)
is_developer = similarity(new_vec, dev_prototype) > 0.5

# Find what changed between versions
v1 = store.encoder.encode_data(config_v1)
v2 = store.encoder.encode_data(config_v2)
delta = store.difference(v1, v2)  # The change is a vector!

# "X but NOT Y" queries
all_errors = store.encoder.encode_data({"type": "error"})
known_bugs = store.encoder.encode_data({"type": "error", "known": True})
unknown_errors = store.negate(all_errors, known_bugs)

# Boost specific signals
base_query = store.encoder.encode_data({"topic": "security"})
priority_signal = store.encoder.encode_data({"severity": "critical"})
boosted = store.amplify(base_query, priority_signal, strength=2.0)
```

### Config Drift Detection (The Coolest Thing We Built)

```python
# Encode configs as vectors
golden = store.encoder.encode_data(golden_config)
actual = store.encoder.encode_data(server_config)

# The drift is a vector
drift = store.difference(golden, actual)

# Remove expected changes
expected = store.encoder.encode_data({"version": "2.0"})
unexpected = store.negate(drift, expected, method="orthogonalize")

# Amplify security-related drift
security = store.encoder.encode_data({"tls": {}, "auth": {}})
security_drift = store.amplify(unexpected, security, 2.0)

# Find servers with similar security drift
results = client.search_by_vector(security_drift, limit=10)
```

This isn't possible with traditional search. The drift, the expected changes, the amplification - they're all *vectors* that compose.

## Markers

Special `$`-prefixed keys control encoding behavior:

| Marker | Purpose | Example |
|--------|---------|---------|
| `$time` | Temporal similarity | `{"created": {"$time": 1706500000}}` |
| `$mode` | Sequence encoding | `{"events": {"$mode": "ngram", "sequence": [...]}}` |
| `$any` | Wildcard | `{"role": {"$any": True}}` |
| `$or` | Disjunction | `{"$or": [{"a": 1}, {"b": 2}]}` |

### Sequence Modes

| Mode | Use Case | Example |
|------|----------|---------|
| `positional` | Ordered lists | Event sequences |
| `chained` | Prefix/suffix matching | Transaction chains |
| `ngram` | Fuzzy substring | Text search |
| `bundle` | Unordered sets | Tags, categories |

### Guards (Post-Query Filtering)

```python
results = client.search_json(
    probe={"type": "user"},
    guard={
        "age": {"$gte": 18, "$lt": 65},
        "role": {"$in": ["admin", "mod"]},
        "bio": {"$contains": "developer"},
        "$exists": {"email": True}
    },
    negations={"status": "inactive"}
)
```

### Marker Prefix

If your data uses `$` keys, configure a different prefix:

```python
store = CPUStore(dimensions=4096, marker_prefix="@@")
# Now use @@time, @@mode, @@any, etc.
```

## Configuration

### Backends

| Backend | Speed | Best For |
|---------|-------|----------|
| `cpu` (default) | 11K ops/sec | General use |
| `torchhd` | 300 ops/sec | Accuracy-critical (Level embeddings: 200 ≈ 201 ≠ 500) |
| `gpu` | 40x batch | Large batch operations (1000+ items) |

```python
store = CPUStore(backend="torchhd")  # Best accuracy for numeric fields
```

### Dimensions

| Use Case | Dimensions | Records/GB |
|----------|------------|------------|
| Simple documents (<20 fields) | 1024 | ~817K |
| Complex + time encoding | 4096 | ~233K |
| Very complex (100+ fields) | 8192 | ~119K |

## HTTP API

```bash
./scripts/run_with_venv.sh python scripts/server/holon_server.py
```

```bash
# Insert
curl -X POST http://localhost:8000/api/v1/items \
  -d '{"data": "{\"name\": \"Alice\"}", "data_type": "json"}'

# Search
curl -X POST http://localhost:8000/api/v1/search \
  -d '{"probe": "{\"name\": \"Alice\"}", "top_k": 5}'

# Encode vector
curl -X POST http://localhost:8000/api/v1/vectors/encode \
  -d '{"data": "{\"topic\": \"security\"}", "data_type": "json"}'

# Prototype
curl -X POST http://localhost:8000/api/v1/vectors/prototype \
  -d '{"vectors": [[1,-1,0,...], [0,1,-1,...]], "threshold": 0.5}'
```

See [API Reference](docs/api_reference.md) for complete documentation.

## Honest Assessment

**What Holon does well:**
- Fuzzy similarity search over structured data
- Prototype learning and classification  
- "Find similar to X" and "X but not Y" queries
- Deep nesting (6+ levels), high field counts (100+ fields)
- Composable vector operations that work over HTTP
- Finding needles in haystacks (rank 1 among 500+ similar items)

**What Holon cannot do:**
- Constraint satisfaction (Sudoku, SAT, graph coloring)
- NP-hard optimization
- Exact matching where "close enough" isn't acceptable
- Many-category k-NN (30% accuracy at 50 categories)

**Honest caveats:**
- Scale tested to 10K items. Behavior at 1M+ is unknown.
- "100% accuracy" claims are on synthetic data with clean separation.
- Sub-1ms queries are numpy array ops, not magic.
- Never benchmarked against Elasticsearch/Algolia.
- TorchHD accuracy gains come at 39x speed cost.

## Challenges & Examples

| Batch | Description | Status |
|-------|-------------|--------|
| [008](docs/challenges/008-batch/CHALLENGES.md) | Full primitive showcase (7 solutions) | ✅ 92-100% |
| [007](scripts/challenges/007-batch/FINAL_STATUS.md) | Multi-domain demos | ✅ 7/7 |
| [006](docs/challenges/006-batch/LEARNINGS.md) | LLM memory augmentation | ✅ 82% token savings |
| [004](scripts/challenges/004-batch/approaches/FINAL_ASSESSMENT.md) | Sudoku (VSA limitation) | ❌ Cannot solve |

```bash
# Run the primitives demo
./scripts/run_with_venv.sh python examples/primitives_demo.py
```

## Documentation

- [API Reference](docs/api_reference.md) - Complete HTTP and Python API
- [Encoding Guide](docs/encoding_guide.md) - How data becomes vectors
- [Examples](examples/) - Working code samples

## Key Concepts

- **Binding**: `key × value` - element-wise multiplication combines related vectors
- **Bundling**: `a + b + c` - vector superposition aggregates multiple vectors
- **Similarity**: Cosine distance enables partial/substructure matching

Named after Arthur Koestler's holon - a self-contained whole that is simultaneously part of a larger whole.

<div align="center">
<img src="assets/forbidden-binding-spell.gif" alt="Vector Operations">

*From mystical runes to mathematical vectors. The power endures.*
</div>

---

MIT Licensed
