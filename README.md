# Holon: Hyperdimensional Memory for Structured Data

**Authors**: watministrator, Grok (xAI), & Claude (Anthropic)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Rust](https://img.shields.io/badge/rust_version-holon--rs-orange.svg)](https://github.com/watmin/holon-rs)

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
cd holon
python -m venv holon_env
source holon_env/bin/activate
pip install -e .
```

All scripts use `./scripts/run_with_venv.sh` to ensure venv activation:

```bash
./scripts/run_with_venv.sh python examples/basic_usage.py
./scripts/run_with_venv.sh pytest tests/
```

## Core Primitives

Everything in Holon is built from these kernel operations:

| Category | Primitives |
|----------|-----------|
| **Encoding** | `encode_data(json)`, `encode_sequence(items, mode)` |
| **Continuous** | `encode_scalar(v, mode)`, `encode_scalar_log(v)` - linear, circular, log-scale |
| **VSA Ops** | `bind(a,b)`, `unbind(ab,a)`, `bundle([vecs])`, `permute(v,k)` |
| **Learning** | `prototype([examples])`, `prototype_add(p,ex,n)`, `cleanup(noisy,codebook)` |
| **Streaming** | `create_accumulator()`, `accumulate(acc,v)`, `decay(acc,f)`, `normalize_accumulator(acc)` |
| **Manipulation** | `difference(a,b)`, `amplify(v,sig,str)`, `negate(v,x)`, `blend(a,b,α)`, `resonance(v,ref)` |
| **Extended** | `attend(q,m,s,mode)`, `analogy(a,b,c)`, `project(v,subspace)`, `conditional_bind(a,b,gate)` |
| **Analysis** | `similarity_profile(a,b)`, `complexity(v)`, `segment(stream,w,t)`, `invert(v,codebook)` |
| **Similarity** | `similarity(a,b,metric)` - cosine, hamming, overlap, agreement, euclidean, manhattan |

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

### Extended Primitives (Challenge 014)

New primitives for explainable anomaly forensics:

```python
from holon.primitives import segment, invert, complexity, attend, analogy

# Find WHEN behavior changed in a stream
breakpoints = segment(packet_vectors, window=100, threshold=0.3)
# → [0, 145, 312] - traffic changed at indices 145 and 312

# Measure HOW "mixed" a vector is (0.0 = clean, 1.0 = superposition)
c = complexity(anomaly_vec)  # → 0.45 (fairly clean signal)

# Decompose WHAT patterns explain an anomaly
codebook = [("normal", normal_proto), ("scan", scan_proto), ("exfil", exfil_proto)]
components = invert(anomaly_vec, codebook, top_k=3)
# → [("scan", 0.78), ("normal", 0.35)] - "78% scan, 35% normal"

# Soft attention: focus on attack-relevant dimensions
attended = attend(attack_signature, packet_vec, strength=2.0, mode="soft")

# Pattern transfer: A is to B as C is to ?
port_443_exfil = analogy(port_80_scan, port_443_scan, port_80_exfil)
# Generalizes attack patterns across contexts
```

### Continuous Value Encoding (Challenge 012)

Encode continuous values where similar values produce similar vectors:

```python
# Log-scale encoding: equal ratios = equal similarity
rate_100 = store.encode_scalar_log(100)
rate_1000 = store.encode_scalar_log(1000)
rate_10000 = store.encode_scalar_log(10000)

# 100→1000 similarity ≈ 1000→10000 similarity (both 10x)
store.similarity(rate_100, rate_1000)   # ~0.94
store.similarity(rate_1000, rate_10000) # ~0.92

# Linear encoding for positions, temperatures, etc.
temp_vec = store.encode_scalar(72.5, mode="linear")

# Circular encoding for angles, hours (wraps around)
hour_vec = store.encode_scalar(23.5, mode="circular", period=24.0)
# hour 23.5 is similar to hour 0.5 (they're close on the clock)
```

**Why it matters**: Eliminates hardcoded discretization like `{rate: "high"}`. A rate of 100 pps is naturally similar to 150 pps and dissimilar from 100,000 pps.

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
| `cpu` (default) | 3.9K/sec (1 core), 11K/sec (10 cores) | General use |
| `torchhd` | 300 ops/sec | Accuracy-critical (Level embeddings: 200 ≈ 201 ≠ 500) |
| `gpu` | 40x batch | Large batch operations (1000+ items) |

```python
store = CPUStore(backend="torchhd")  # Best accuracy for numeric fields
```

For parallel encoding at scale, see [011-large-scale-stress-test.py](scripts/challenges/009-batch/011-large-scale-stress-test.py).

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

## Scale Testing (Challenge 009)

We stress-tested Holon at realistic scale (14-core, 54GB RAM machine):

| Samples | Categories | Accuracy | Encode Rate | Time | Memory |
|---------|------------|----------|-------------|------|--------|
| 1M | 100 | **94.5%** | 25,581/sec | 44s | 3.9 GB |
| 1M | 1,000 | **84.5%** | 29,561/sec | 68s | 3.9 GB |
| **5M** | **1,000** | **84.4%** | 23,322/sec | **7.5 min** | **19.5 GB** |

**Key finding**: Accuracy depends on samples per category, not dimensions.

| Factor | Impact on Accuracy |
|--------|-------------------|
| 10 samples/cat → 100 | +13 points |
| Signal 70% → 90% | +14 points |
| 1024D → 8192D | +6 points |

At 1000 categories, 84% accuracy = 840x random baseline. The approach: encode records → average by category → find nearest prototype. No neural networks, no GPU, no training loop.

See [Challenge 009 Learnings](docs/challenges/009-batch/LEARNINGS.md) for details.

## Network Anomaly Detection (Challenge 010)

We built a complete streaming anomaly detection system with **deterministic consensus**:

### HTTP Request Detection

| Metric | Value |
|--------|-------|
| **F1 Score** | **1.000** |
| **Throughput** | **8,339 req/sec** |
| **Latency** | **0.12 ms** |

Detects SQL injection, XSS, path traversal, command injection - using either:
- **Rule-based**: Explicit patterns (~95% of detections)
- **Headless**: Pure frequency+decay on character class bitmasks (no attack knowledge)

### DDoS Detection (Two-Phase)

| Attack | Detected | Classified | Throughput |
|--------|----------|------------|------------|
| SYN Flood | ✅ | ✅ | 3,193/sec |
| DNS Reflection | ✅ | ✅ | 3,962/sec |
| NTP Amplification | ✅ | ✅ | 1,752/sec |
| ICMP Flood | ✅ | ✅ | 6,258/sec |

**Key insight**: DDoS = variance drop + mean similarity rise (homogeneous attack traffic).

### Distributed Consensus

All nodes with the same `global_seed` generate identical vectors:

```python
# Node A (Tokyo)
vm = DeterministicVectorManager(global_seed=42)
vec_a = vm.get_vector("admin'--")

# Node B (NYC)
vm = DeterministicVectorManager(global_seed=42)
vec_b = vm.get_vector("admin'--")

assert np.array_equal(vec_a, vec_b)  # ✅ Always true
```

No synchronization needed - mathematical agreement enables parallel stream processing.

### Accumulator Primitives

New primitives for frequency-preserving learning:

```python
# Create accumulator (float64, no thresholding)
accum = encoder.create_accumulator()

# Stream observations
for request in stream:
    vec = encoder.encode_data(request)
    accum = encoder.accumulate(accum, vec)

# Get normalized for similarity queries
prototype = encoder.normalize_accumulator(accum)
```

Why it works: `prototype_add()` thresholds → loses frequency. Accumulator preserves frequency → 99% benign dominates → F1=1.000.

See [Challenge 010 Learnings](docs/challenges/010-batch/LEARNINGS.md) for complete details.

## Packet Analysis & Structural Detection (Challenge 011)

Building on Challenge 010, we discovered **the critical importance of structural encoding**:

### The Key Insight: Use Holon's Structural Encoding

```python
# WRONG: Naive atom bundling (loses structure)
atoms = ["proto:tcp", "dst_port:80", "flags:PA"]
vec = sum(vm.get_vector(atom) for atom in atoms)
# Result: F1 = 0.368 ❌

# RIGHT: Structural encoding (preserves structure)
structure = {"l4": {"proto": "tcp", "dst_port": 80, "flags": "PA"}}
vec = encoder.encode_data(structure)
# Result: F1 = 1.000 ✅
```

**Why it works**: Role-filler binding (`key ⊛ value`) preserves that `{dst_port: 80}` is different from `{src_port: 80}` - they share "80" but are bound to different roles.

### Three-Dimensional Detection

| Dimension | Purpose | F1 Score |
|-----------|---------|----------|
| **Transition** | Attack beginning/ending | **0.936** |
| **Classification** | SYN flood, DNS reflection, etc. | **0.998** |
| **Binary** | Is this an attack? | **1.000** |

### Knowledge Composition

```python
# Three knowledge sources work together
prior_sim = 0.96    # Frozen baseline (survives attacks)
recent_sim = 0.84   # Adaptive with decay (tracks current traffic)
divergence = 0.30   # Prior/recent divergence (regime change signal)
```

| Phase | Prior Similarity | Divergence |
|-------|------------------|------------|
| Normal | 0.96 | 0.99 (stable) |
| Attack | 0.16 | 0.30 (regime change!) |
| Recovery | 0.75 | 0.93 (returning) |

### Cross-Pollination: Best of Both Batches

The integrated detector combines:

| From Batch 010 | From Batch 011 |
|----------------|----------------|
| Port bucketing | Structural encoding |
| IP prefix levels | Prior/recent separation |
| Payload bitmask | State machine transitions |
| Rule-based detection | Sample-based signatures |
| Variance-based DDoS | Culprit identification |

**Result**: F1 = 1.000, Classification = 100%

### Mitigation Synthesis: Closing the Loop

Using vector operations to derive actionable firewall rules:

```python
# What makes attacks different from normal?
attack_delta = store.difference(attack_signature, baseline)

# Which features contribute most?
for feature in features:
    importance = similarity(encode({feature: value}), attack_delta)
    if importance > threshold:
        rules.append(f"DROP if {feature}={value}")
```

**Generated Rules (F1 = 1.000):**
```bash
# DNS reflection
iptables -A INPUT -p udp --sport 53 --dport 1024:65535 -j DROP

# SYN flood
iptables -A INPUT -p tcp --tcp-flags ALL SYN -m limit --limit 10/s -j ACCEPT
```

The complete pipeline: **Learn → Detect → Identify → Mitigate**

```bash
# Run the wrap-up demo
./scripts/run_with_venv.sh python scripts/challenges/011-batch/DEMO-batch-011-wrapup.py
```

See [Challenge 011 Learnings](docs/challenges/011-batch/LEARNINGS.md) for complete details.

## Honest Assessment

**What Holon does well:**
- Fuzzy similarity search over structured data
- Prototype learning and classification (94.5% at 100 categories)
- "Find similar to X" and "X but not Y" queries
- Deep nesting (6+ levels), high field counts (100+ fields)
- Composable vector operations that work over HTTP
- Finding needles in haystacks (rank 1 among 500+ similar items)
- Scales to 500k records at 11k encodes/sec
- Streaming anomaly detection (F1=1.000 at 8k req/sec)
- DDoS detection with attack classification (100% accuracy)
- Distributed consensus without synchronization
- Three-dimensional detection: transition (0.936), classification (0.998), binary (1.000)
- Knowledge composition: prior/recent/divergence tracks regime changes
- Mitigation synthesis: vector-derived firewall rules from attack signatures
- Zero-hardcode detection: 100% recall, 4% FP without domain knowledge (Challenge 012)
- Continuous value encoding: log-scale rates, circular angles with smooth similarity

**What Holon cannot do:**
- Constraint satisfaction (Sudoku, SAT, graph coloring)
- NP-hard optimization
- Exact matching where "close enough" isn't acceptable

**Brutal honesty:**
- **All benchmarks use synthetic data** - Real-world accuracy is unproven
- **Never compared to baselines** - No TF-IDF, neural embedding, or traditional ML comparisons
- **81.7% at 1000 categories** - Still on planted signal, real data may be harder
- **HNSW index build is the bottleneck** - 132s for 80k vectors (~30 min for 1M)
- **17k/sec encoding required 10 cores** - Single-threaded is ~4k/sec

## Challenges & Examples

| Batch | Description | Status |
|-------|-------------|--------|
| [012](docs/challenges/012-batch/LEARNINGS.md) | Zero-hardcode anomaly detection | ✅ 100% recall, 4% FP |
| [011](docs/challenges/011-batch/LEARNINGS.md) | Structural detection & cross-pollination | ✅ F1=1.000 |
| [010](docs/challenges/010-batch/LEARNINGS.md) | Network anomaly & DDoS detection | ✅ 100% detection |
| [009](docs/challenges/009-batch/LEARNINGS.md) | Deterministic training at scale (500k records) | ✅ 94.5% accuracy |
| [008](docs/challenges/008-batch/CHALLENGES.md) | Full primitive showcase (7 solutions) | ✅ 92-100% |
| [007](scripts/challenges/007-batch/FINAL_STATUS.md) | Multi-domain demos | ✅ 7/7 |
| [006](docs/challenges/006-batch/LEARNINGS.md) | LLM memory augmentation | ✅ 82% token savings |
| [004](scripts/challenges/004-batch/approaches/FINAL_ASSESSMENT.md) | Sudoku (VSA limitation) | ❌ Cannot solve |

```bash
# Run the primitives demo
./scripts/run_with_venv.sh python examples/primitives_demo.py
```

## Documentation

- [Use Cases](docs/USE_CASES.md) - When and why to use Holon
- [API Reference](docs/api_reference.md) - Complete HTTP and Python API
- [Encoding Guide](docs/encoding_guide.md) - How data becomes vectors
- [Examples](examples/) - Working code samples

## Key Concepts

- **Binding**: `key × value` - element-wise multiplication combines related vectors
- **Bundling**: `a + b + c` - vector superposition aggregates multiple vectors
- **Similarity**: Cosine distance enables partial/substructure matching

## Why "Holon"?

Named after Arthur Koestler's concept - a *holon* is a self-contained whole that is simultaneously part of a larger whole. Each data item in Holon is independent yet entangled through vector relationships. A document is complete on its own, but its vector connects it to every similar document in the space.

<div align="center">
<img src="assets/forbidden-binding-spell.gif" alt="Vector Operations">

*From mystical runes to mathematical vectors. The power endures.*
</div>

## See Also

- **[holon-rs](https://github.com/watmin/holon-rs)** — Rust implementation with 12x performance boost and SIMD acceleration. Same VSA foundations, same API patterns, faster execution.

---

MIT Licensed
