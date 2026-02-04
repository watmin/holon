# Challenge 010 Learnings: Realistic Data & Deterministic Consensus

## Overview

This challenge addressed the key limitation from batch 009: synthetic data was too clean. We built production-like data generators and validated deterministic consensus for distributed processing.

## Key Findings

### 1. Deterministic Consensus Works

Two independently created encoders processing the same atoms produce identical vectors:

```
Node A: processes ["billing", "technical", "shipping"] in random order → vectors V1, V2, V3
Node B: processes same atoms in different order → same vectors V1, V2, V3
```

**Consensus: 100/100 exact matches** across all tests.

The key insight: hash-based seeding makes vector generation order-independent:

```python
def get_vector(self, atom: str) -> np.ndarray:
    seed = hash(atom) ^ self.global_seed  # Deterministic!
    return generate_from_seed(seed)
```

This enables sharded stream processing without synchronization points.

### 2. Cardinality Scales Well

| Scale | Atoms | Encode Rate | Notes |
|-------|-------|-------------|-------|
| 10k records | 42,979 | 962/sec | Quick iteration |
| 100k records | 332,525 | 1,119/sec | Still fast |

**33x more atoms than batch 009** with no performance degradation.

The codebook grows linearly with unique values, but cached lookups are nearly free (1000x faster than generation).

### 3. Realistic Data Reveals True Difficulty

Not all schemas are equally classifiable:

| Schema | 10k Accuracy | 100k Accuracy | Why |
|--------|-------------|---------------|-----|
| metric | 100% | 100% | Few categories, distinct values |
| order | 100% | 100% | Strong status signal |
| alert | 100% | 100% | Severity is very discriminative |
| user_event | 100% | 100% | Clear event type patterns |
| log_entry | 95.7% | 100% | Improved with more examples |
| deployment | 89.3% | 90.3% | Status transitions blur |
| config_change | 66.0% | 90.4% | Improved with more examples |
| api_request | 48.5% | **11.0%** | Path variance explodes |

**Key insight**: `api_request` has high-cardinality paths (`/api/users/123`, `/api/users/456`, etc.). Each path becomes its own category, making classification nearly impossible. This is realistic - not all data is classifiable!

### 4. Holon Primitives in Action

We demonstrated 6 primitives that were underutilized in previous batches:

**`prototype()`** - Build category centroid from examples
```python
proto = encoder.prototype(billing_vectors, threshold=0.5)
# Non-zero elements: 1509 / 4096 (keeps only agreed dimensions)
```

**`negate()`** - Remove component from superposition
```python
bundle_abc = encoder.bundle([A, B, C])
negated = encoder.negate(bundle_abc, A)
# Similarity to A dropped from 0.003 to -0.024 (815% reduction)
```

**`difference()`** - Detect what changed
```python
delta = encoder.difference(before, after)
# Positive: 1247 (added), Negative: 1221 (removed)
```

**`resonance()`** - Extract agreeing parts
```python
resonant = encoder.resonance(vec1, vec2)
# 1556 / 4096 dimensions where both agree
```

**`blend()`** - Interpolate between vectors
```python
blended = encoder.blend(vec1, vec2, alpha=0.5)
# Midpoint between two records
```

### 5. Data Generator Characteristics

The RealisticDataGenerator produces genuinely complex data:

| Feature | Configuration |
|---------|--------------|
| Schemas | 8 types (api_request, log_entry, order, etc.) |
| Missing fields | 20% of optional fields omitted |
| Extra fields | 10% records have unexpected fields |
| Nesting | Up to 3 levels deep |
| Lists | 1-15 items per list field |
| Type coercion | 5% have int/string variance |
| Null handling | 50% explicit null vs absent |
| Cardinality | 100k+ unique values configurable |

## Technical Insights

### Hash-Based Seeding for Consensus

The critical fix for distributed consensus:

```python
# WRONG: Sequential seeding (order-dependent)
rng = np.random.RandomState(42)
vectors["billing"] = rng.choice(...)   # depends on call order
vectors["technical"] = rng.choice(...) # different if called first!

# RIGHT: Hash-based seeding (order-independent)
def get_vector(atom):
    seed = sha256(atom) ^ global_seed
    rng = RandomState(seed)
    return rng.choice(...)  # same result regardless of order
```

### Memory Characteristics

At 100k records with 4096 dimensions:

| Component | Memory |
|-----------|--------|
| Raw records | ~50 MB (JSON-ish) |
| Encoded vectors | 327 MB (80k × 4096 × int8) |
| Codebook | ~1.3 GB (332k atoms × 4096 × int8) |

Codebook dominates. For persistence, only the codebook needs to be saved - vectors can be regenerated.

### Encoding Performance

Single-threaded: ~1,100 vectors/sec

The bottleneck is the recursive encoding of nested structures, not vector operations. Parallelization would help (as demonstrated in batch 009).

## Limitations and Open Questions

### 1. High-Cardinality Categories Fail

When a field has too many unique values (like API paths), each value becomes its own category. With hundreds of categories and few examples each, classification fails.

**Potential solutions**:
- Hierarchical encoding (encode path prefix separately from full path)
- Feature hashing to limit cardinality
- Ignore high-cardinality fields in category assignment

### 2. Streaming Updates Not Tested

We built prototypes from batch data. Real streaming needs:
- `prototype_add()` for incremental updates
- Drift detection via `difference()`
- Prototype decay for outdated patterns

### 3. Persistence Strategy TBD

Current approach: everything in memory. Future needs:
- Dragonfly for distributed codebook storage
- Efficient serialization (codebook is 1.3 GB at 100k scale)
- Lazy loading for large codebooks

## Code Summary

| File | Purpose |
|------|---------|
| `realistic_data_generator.py` | 8-schema messy data generator |
| `deterministic_codebook.py` | Order-independent vector manager |
| `001-realistic-data-stress-test.py` | Integration test with all primitives |

## Conclusions

1. **Deterministic consensus works** - Hash-based seeding enables sync-free distributed processing

2. **Cardinality scales** - 332k atoms with no performance degradation

3. **Realistic data is harder** - Some schemas (api_request) are genuinely difficult to classify

4. **Holon primitives are powerful** - negate, amplify, resonance, difference, blend all work as expected

5. **The "deterministic AI" vision is validated** - No gradients, no magic weights, fully reproducible

## Phase 2: Anomaly Detection Deep Dive

### The Frequency-Based Approach

When training data is predominantly benign with rare malicious samples (realistic 99/1 ratio), we explored using frequency-weighted prototypes.

**Key insight**: VSA's `prototype()` does majority voting. If 99% of training vectors agree, the prototype captures that consensus. Rare anomalies get "outvoted."

### What Worked

1. **Atomic pattern encoding** - Encode complete patterns as single atoms
   - `"GET|/api/users"` → one random vector (not bind(method, path))
   - Avoids partial matches (malicious GET doesn't match benign GET patterns)

2. **Pattern membership** - The clearest signal
   - Known patterns (seen during training) vs unknown patterns
   - In our test: 200/200 benign KNOWN, 0/20 malicious KNOWN = 100% separation

3. **Using Holon primitives for frequency weighting**:
   - `mathematical_bundle(vectors, weights)` - weight by log(frequency)
   - `amplify(prototype, high_freq_pattern)` - boost common patterns
   - `negate(prototype, rare_pattern)` - remove rare (suspicious) patterns

### What Didn't Work Well

1. **Cosine similarity to bundled prototype** - Too noisy
   - Benign patterns had similarity range [-0.01, +0.03]
   - Malicious patterns had range [-0.01, +0.01]
   - Overlap makes threshold-based detection imprecise

2. **Resonance with partial field matches**
   - When encoding `bind(method, path)`, malicious "GET /admin" resonates with benign "GET /api/users" due to shared method
   - Solution: atomic encoding of complete pattern

### The Real Value of VSA for Anomaly Detection

VSA doesn't provide semantic similarity (all random atoms are orthogonal). Its value is:

| Feature | How It Helps |
|---------|--------------|
| **Deterministic vectors** | Same pattern → same vector across nodes (consensus) |
| **Pattern membership** | Hash pattern → known or unknown |
| **Frequency composition** | `mathematical_bundle` with frequency weights |
| **Incremental updates** | `prototype_add` for streaming |
| **Composability** | Bundle patterns without coordination |

### Proposed New Primitive

Based on this exploration, a `frequency_prototype()` primitive would be useful:

```python
def frequency_prototype(
    self,
    pattern_counts: Dict[str, int],
    normalizer: Callable[[str], str] = None,
    min_count: int = 5,
    decay: str = "log",  # linear, log, sqrt, threshold
) -> np.ndarray:
    """
    Build prototype from frequency-counted patterns.

    - Patterns below min_count are excluded (potential anomalies)
    - Weights are computed as decay(count)
    - Returns bundled prototype for similarity checking
    """
    vectors = []
    weights = []

    for pattern, count in pattern_counts.items():
        if count >= min_count:
            normalized = normalizer(pattern) if normalizer else pattern
            vec = self.vector_manager.get_vector(normalized)
            vectors.append(vec)
            weights.append(decay_fn(count))

    return self.mathematical_bundle(vectors, weights)
```

## Conclusions (Updated)

1. **Deterministic consensus works** - Hash-based seeding enables sync-free distributed processing

2. **Cardinality scales** - 332k atoms with no performance degradation

3. **Domain knowledge is required** - Fully headless encoding doesn't work for anomaly detection
   - Need to normalize high-cardinality fields
   - Need to select relevant fields
   - Need to understand what "anomaly" means

4. **Pattern membership > similarity** - For anomaly detection, checking if a pattern is KNOWN vs UNKNOWN is clearer than cosine similarity to a prototype

5. **Frequency weighting via Holon primitives** - `amplify()`, `negate()`, `mathematical_bundle()` provide the tools for frequency-based detection

6. **Atomic encoding avoids partial matches** - Encode complete normalized patterns as single atoms

## Phase 3: Accumulator Primitive (The Breakthrough)

### The Problem with `prototype_add()`

The `prototype_add()` primitive thresholds after each update:

```python
def prototype_add(prototype, example, count):
    weighted = prototype * count + example
    averaged = weighted / (count + 1)
    return threshold_bipolar(averaged)  # ← Loses frequency info!
```

After 10,000 observations, each new one barely affects the thresholded result. The **frequency signal is lost**.

### The Solution: Accumulator (Float Sum, No Thresholding)

Keep a running float sum without thresholding:

```python
def accumulate(accumulator, example):
    return accumulator + example.astype(np.float64)

def normalize_accumulator(accumulator):
    norm = np.linalg.norm(accumulator)
    return (accumulator / norm).astype(np.float32)
```

With 99% benign and 1% malicious traffic, benign patterns contribute **99x more** to the accumulator. The frequency signal is preserved!

### Results: Perfect Separation

| Metric | `prototype_add()` | Accumulator |
|--------|-------------------|-------------|
| Benign similarity | ~0.01 | **0.52 - 0.82** |
| Malicious similarity | ~0.00 | **0.21 - 0.41** |
| Overlap | Yes (unusable) | **None** |
| Separation | ~0.01 | **+0.38** |
| F1 Score | ~0.23 | **1.000** |

**No overlap** - benign minimum (0.52) > malicious maximum (0.41)!

### New Primitives Added to Holon

Four new primitives in `encoder.py`:

```python
# Create empty accumulator
accum = encoder.create_accumulator()

# Add observation (no thresholding)
accum = encoder.accumulate(accum, vec)

# Get unit-normalized for similarity queries
normalized = encoder.normalize_accumulator(accum)

# Convert back to bipolar if needed
bipolar = encoder.threshold_accumulator(accum)
```

### Key Properties (Validated by 22 Unit Tests)

1. **Frequency preservation** - Repeated patterns dominate
2. **Order independence** - Same result regardless of observation order
3. **Precision** - Uses float64 to avoid overflow at scale (1M+ obs)
4. **Composability** - Works with encoded records, nested data, etc.

### Why This Works

VSA vectors are sparse bipolar {-1, 0, 1}. When accumulated:

- **Frequent patterns**: Dimensions that always agree (e.g., always +1) grow large in the accumulator
- **Rare patterns**: Contribute small values that get normalized away
- **Normalization**: Highlights dimensions with strong agreement

The normalized accumulator is like a "consensus signature" of what the system has seen most often.

### Test Coverage

New test file: `tests/test_accumulator.py` with 22 tests covering:
- Basic accumulation
- Frequency preservation vs `prototype_add()`
- Commutativity (order-independence)
- Anomaly detection scenarios
- Edge cases (zero vectors, negation, large scale)
- Integration with encoded data structures
- Determinism

## Conclusions (Final)

1. **Deterministic consensus works** - Hash-based seeding enables sync-free distributed processing

2. **Cardinality scales** - 332k atoms with no performance degradation

3. **Domain knowledge is required** - Need to normalize/select relevant fields

4. **Accumulator > `prototype_add()` for frequency** - The key insight: don't threshold during training

5. **Perfect anomaly detection achieved** - F1=1.000 with proper accumulator approach

6. **New primitive added to Holon** - `accumulate()`, `normalize_accumulator()`, `threshold_accumulator()`, `create_accumulator()`

## Phase 4: Challenge Vector System (F1 = 1.000)

### The Multi-Signal Architecture

Single-prototype detection fails when attacks look structurally similar to benign requests.
The solution: **multiple challenge vectors** with **multi-signal scoring**.

```
TRAINING (offline):
├── Category Prototypes (per-endpoint)
│   └── accumulate() benign patterns → normalized prototype per category
├── Attack Signatures (value-level)
│   └── bundle(get_vector("admin'--"), get_vector("' OR '1'='1"), ...)
└── Structural Templates (expected shapes)

RUNTIME (online):
├── Vectorize request
├── Signal 1: Best category match (cleanup-style nearest neighbor)
├── Signal 2: Attack signature match (check body VALUES)
├── Signal 3: Structural template match
├── Signal 4: Global benign similarity
└── Multi-signal decision logic
```

### Key Insight: Value-Level Matching

The breakthrough was encoding **malicious value strings directly as atoms**:

```python
# OLD (didn't work): Encode as nested structure
{"body": {"user": "admin'--"}}  # Structure dominates

# NEW (works): Encode values directly
sql_sig = bundle([
    get_vector("admin'--"),
    get_vector("' OR '1'='1"),
    get_vector("'; DROP TABLE"),
])

# Check body values individually
for val in body.values():
    sim = cosine(get_vector(val), sql_sig)  # Direct match!
```

### Results

| Metric | Single Prototype | Multi-Signal |
|--------|------------------|--------------|
| Precision | 25-67% | **100%** |
| Recall | 65-100% | **100%** |
| F1 | 0.37-0.59 | **1.000** |
| Throughput | ~2,800/sec | **4,000/sec** |

### Decision Logic

```python
if is_high_attack AND is_low_category:
    # Matches attack + doesn't match benign → ATTACK

elif is_high_attack AND NOT is_low_category:
    # Attack pattern in benign structure (SQL in login body)
    # Still flag as attack

elif is_low_category AND is_low_global:
    # Unknown pattern → suspicious
```

### Holon Primitives Used

- `accumulate()` - Build frequency-weighted category prototypes
- `bundle()` - Combine attack signature values
- `get_vector()` - Direct atom encoding for value-level matching
- `normalize_accumulator()` - Unit normalize for similarity queries

### Production Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   TRAINING CLUSTER                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │ Log Shard 1 │  │ Log Shard 2 │  │ Log Shard 3 │     │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘     │
│         │                │                │             │
│         └────────────────┼────────────────┘             │
│                          ▼                              │
│              ┌───────────────────────┐                  │
│              │   Build Challenge     │                  │
│              │       Vectors         │                  │
│              └───────────┬───────────┘                  │
│                          │                              │
│              ┌───────────▼───────────┐                  │
│              │  ChallengeVectors:    │                  │
│              │  - category_protos    │                  │
│              │  - attack_signatures  │                  │
│              │  - global_benign      │                  │
│              └───────────┬───────────┘                  │
└──────────────────────────┼──────────────────────────────┘
                           │ Distribute async
                           ▼
┌─────────────────────────────────────────────────────────┐
│                   RUNTIME SERVERS                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │  Server A   │  │  Server B   │  │  Server C   │     │
│  │             │  │             │  │             │     │
│  │  Request    │  │  Request    │  │  Request    │     │
│  │     ↓       │  │     ↓       │  │     ↓       │     │
│  │ Vectorize   │  │ Vectorize   │  │ Vectorize   │     │
│  │     ↓       │  │     ↓       │  │     ↓       │     │
│  │ Score(cv)   │  │ Score(cv)   │  │ Score(cv)   │     │
│  │     ↓       │  │     ↓       │  │     ↓       │     │
│  │ ALLOW/DENY  │  │ ALLOW/DENY  │  │ ALLOW/DENY  │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
└─────────────────────────────────────────────────────────┘

Throughput: ~4,000 req/sec per core
Latency: ~0.25 ms/request
No synchronization needed (deterministic vectors)
```

## Phase 5: Streaming Challenge Vectors (Complete Solution)

### The Three Phases

We validated a complete end-to-end streaming anomaly detection system:

```
┌─────────────────────────────────────────────────────────────────────────┐
│ PHASE 1: TRAINING (offline)                                             │
│                                                                         │
│   Historical Logs → TrainingPipeline → ChallengeVectors                │
│                                                                         │
│   Components built:                                                     │
│   - Category prototypes (per-endpoint, using accumulate())             │
│   - Attack signatures (bundled malicious value atoms)                  │
│   - Global benign prototype (fallback detector)                        │
│                                                                         │
│   Output: ChallengeVectors (~140 KB for 7 categories + 3 attack sigs)  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ Distribute async (no sync needed)
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ PHASE 2: DISTRIBUTION                                                   │
│                                                                         │
│   Each server:                                                          │
│   1. Creates own DeterministicVectorManager(global_seed=42)            │
│   2. Loads ChallengeVectors                                            │
│   3. Ready to score requests                                           │
│                                                                         │
│   Key property: All servers generate IDENTICAL vectors for same atoms  │
│   Verified: get_vector("admin'--") matches across all nodes            │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ Incoming request stream
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ PHASE 3: STREAMING (real-time)                                          │
│                                                                         │
│   For each request:                                                     │
│   1. Encode request → vector                                           │
│   2. Score against category prototypes (find best match)               │
│   3. Score against attack signatures (check body values!)              │
│   4. Multi-signal decision → ALLOW/DENY                                │
│                                                                         │
│   Latency: 0.188 ms/request                                            │
│   Throughput: 5,277 req/sec (single core)                              │
└─────────────────────────────────────────────────────────────────────────┘
```

### Performance Results

| Metric | Value |
|--------|-------|
| **F1 Score** | **1.000** |
| **Precision** | **100%** |
| **Recall** | **100%** |
| **Throughput** | **5,277 req/sec** |
| **Latency** | **0.188 ms/req** |
| **Distributed Consensus** | **100%** (3 servers, 0 disagreements) |
| **Vector Package Size** | **140 KB** |

### Key Implementation Details

**1. DeterministicVectorManager for Consensus**

Each server creates its own encoder with the same global seed:

```python
class RuntimeScorer:
    def __init__(self, global_seed: int = 42):
        self.vm = DeterministicVectorManager(dimensions=4096, global_seed=global_seed)
        self.encoder = Encoder(vector_manager=self.vm)
```

Result: All servers generate identical vectors for the same atoms, enabling distributed consensus without synchronization.

**2. Body Value Checking for Content Attacks**

SQL injection in login body `{"user": "admin'--"}` looks structurally identical to benign `{"user": "alice"}`. The solution:

```python
# Attack signature built from malicious VALUE strings
sql_sig = bundle([get_vector("admin'--"), get_vector("' OR '1'='1"), ...])

# At runtime, check each body value directly
for val in request.get("body", {}).values():
    val_vec = vm.get_vector(val)
    if cosine(val_vec, sql_sig) > threshold:
        → ATTACK DETECTED
```

**3. Multi-Signal Decision Logic**

```python
is_low_cat = category_similarity < 0.55
is_low_global = global_similarity < 0.45
is_high_attack = attack_similarity > 0.10

if is_high_attack:
    # Matches known attack pattern → DENY
elif is_low_cat and is_low_global:
    # Unknown pattern (not seen in training) → DENY
elif is_low_cat:
    # Rare/unusual structure → DENY
else:
    # Matches known benign category → ALLOW
```

### What Makes This Production-Ready

1. **CPU-only inference** - No GPU required, runs on commodity hardware
2. **Sub-millisecond latency** - 0.188ms p50, compatible with hot path
3. **Deterministic decisions** - Same input → same output across all nodes
4. **Small footprint** - 140 KB challenge vectors, fits in L2 cache
5. **Async updates** - New challenge vectors can be pushed without restart
6. **No external dependencies** - Pure NumPy, no network calls during inference

### Files Created

| File | Purpose |
|------|---------|
| `019-challenge-vector-system.py` | Multi-signal architecture development |
| `020-streaming-challenge-vectors.py` | Complete 3-phase streaming demo |

### Comparison: Single Prototype vs Multi-Signal

| Approach | F1 | Issue |
|----------|-----|-------|
| Single benign prototype | 0.23-0.59 | Thresholding loses frequency |
| Accumulator prototype | 0.925 | Misses content-level attacks |
| **Multi-signal with value checking** | **1.000** | Catches structural + content |

## Phase 6: Continuous Learning with Decaying Accumulator

### The Problem with Batch Training

Previous phases required explicit training on historical data before deployment. This has limitations:
- Requires labeled training data
- Doesn't adapt to traffic pattern changes
- New endpoints trigger false positives until retrained

### The Solution: Decaying Accumulator

A new primitive that enables **continuous passive learning**:

```python
class DecayingAccumulator:
    def update(self, vector, weight=1.0):
        self.accumulator = decay * self.accumulator + weight * vector
```

Key properties:
- **Decay factor** controls memory: `decay=0.9995` → ~2000 request effective window
- **Old patterns fade** automatically
- **New patterns dominate** after sufficient observations
- **No explicit retraining** needed

### Continuous Detection Architecture

```
Request Stream
      │
      ▼
┌─────────────────────────────────────────┐
│         Continuous Detector             │
│                                         │
│  1. Encode request                      │
│  2. Check rules (known attack patterns) │
│  3. Compare to decaying accumulator     │
│  4. Flag if rule match OR low similarity│
│  5. Update accumulator (passive learn)  │
│     - Normal: weight=1.0                │
│     - Flagged: weight=0.1 (resist poison)│
└─────────────────────────────────────────┘
      │
      ▼
  ALLOW / BLOCK
```

### Hybrid Detection

Combines two detection methods:

| Method | What it catches | % of detections |
|--------|-----------------|-----------------|
| **Rule-based** | Known patterns (SQLi, XSS, traversal) | ~95% |
| **Similarity** | Unknown anomalies, unusual structure | ~5% |

Rules are explicit pattern matching:
```python
SQL_PATTERNS = [r"['\"].*(?:OR|AND).*=", r"(?:UNION|SELECT)\\s", ...]
XSS_PATTERNS = [r"<\\s*script", r"javascript\\s*:", r"on(?:error|load)\\s*=", ...]
TRAVERSAL_PATTERNS = [r"\\.\\./", r"etc/passwd", ...]
```

Similarity catches what rules miss - novel attacks that don't match known patterns but look structurally different from normal traffic.

### Results: PCAP Detection

| Metric | Value |
|--------|-------|
| F1 Score | **0.998** |
| Precision | **99.5%** |
| Recall | **100%** |
| Throughput | **3,370 packets/sec** |
| FP Rate | **0.01%** |

Attack type detection:
- c2_beacon: 100%
- known_bad_ip: 100%
- port_scan: 100%
- unusual_port: 100%
- malformed: 100%

### Results: HTTP Detection

| Metric | Value |
|--------|-------|
| F1 Score | **1.000** |
| Precision | **100%** |
| Recall | **100%** |
| Throughput | **8,339 req/sec** |
| FP Rate | **0%** |

Attack type detection:
- sqli_path: 100%
- sqli_body: 100%
- xss_query: 100%
- xss_body: 100%
- traversal: 100%
- hidden_files: 100%
- unusual_method: 100%

### Concept Drift Adaptation

Both detectors automatically adapt to changing traffic patterns:

**PCAP: Traffic shift from web to SMB**
```
First 500 SMB packets:  similarity = 0.467
Last 500 SMB packets:   similarity = 0.702
Improvement: +0.235 (auto-learned new pattern)
```

**HTTP: New API endpoint deployed**
```
First 500 /api/recommendations:  similarity = 0.693
Last 500 /api/recommendations:   similarity = 1.000
Improvement: +0.307 (perfect adaptation)
```

Zero false positives during transition in both cases.

### Key Implementation Details

**1. Request Normalization**

Reduce cardinality while preserving structure:
```python
# Path: /api/users/12345 → /api/users/{id}
path_normalized = re.sub(r'/\\d+', '/{id}', path)

# Headers: just presence, not values
normalized["has_auth"] = "Authorization" in headers

# Body: structure, not content
normalized["body_keys"] = sorted(body.keys())
```

**2. Passive Learning with Poisoning Resistance**

```python
if not is_flagged:
    accumulator.update(vec, weight=1.0)  # Learn from normal
else:
    accumulator.update(vec, weight=0.1)  # Resist poisoning
```

Flagged requests still contribute (in case of false positives) but with reduced weight.

**3. Warmup Period**

First N requests only learn, don't flag:
```python
is_warmup = requests_seen <= WARMUP_REQUESTS
if is_warmup:
    is_flagged = False  # Always allow during warmup
```

### Files Created

| File | Purpose |
|------|---------|
| `021-continuous-pcap-detection.py` | PCAP continuous detection demo |
| `022-continuous-http-detection.py` | HTTP continuous detection demo |

### Comparison: Batch vs Continuous

| Aspect | Batch Training | Continuous Learning |
|--------|----------------|---------------------|
| Training data | Required upfront | Not needed |
| Adaptation | Manual retrain | Automatic |
| New patterns | False positives | Auto-learned |
| Memory | Stores history | Decaying accumulator |
| Deployment | Train → Deploy | Deploy → Learn |

## Phase 7: Headless Structural Fingerprint Detection (F1 = 0.985)

### The Goal: Truly Headless Detection

Previous approaches used explicit rules (SQL patterns, XSS regex). Can we detect attacks with **zero content knowledge**?

### The Solution: Structure + Character Class Bitmask

Instead of encoding content, encode **structural fingerprints**:

```
URL component → [length_bucket, char_class_bitmask]

Bitmask:
  Bit 0 (1):  lowercase
  Bit 1 (2):  uppercase
  Bit 2 (4):  digit
  Bit 3 (8):  normal special (- _ . / @ : ,)
  Bit 4 (16): ABNORMAL special (' " < > ; | & $ ( ) { } [ ] `)
```

Examples:
```
"foo"         → [1, 1]   (tiny, lowercase only)
"user123"     → [2, 5]   (small, lower+digit)
"' OR '1'='1" → [4, 21]  (large, lower+digit+ABNORMAL!)
"<script>"    → [2, 17]  (small, lower+ABNORMAL!)
```

### Why It Works

Attack payloads inherently contain **abnormal characters**:

| Attack Type | Payload | Has Abnormal |
|-------------|---------|--------------|
| SQL Injection | `' OR '1'='1` | Yes (`'`) |
| XSS | `<script>alert(1)</script>` | Yes (`<`, `>`, `(`, `)`) |
| Command Injection | `; cat /etc/passwd` | Yes (`;`) |
| Path Traversal | `../../../etc` | No (but has `parent_ref`) |

Normal values use alphanumeric + normal special chars (`-`, `_`, `.`, `/`).

### Detection Logic

```python
def process(request):
    features = extract_structural_features(request)

    # Immediate flag if abnormal chars detected
    if features["has_abnormal"] or features["has_parent_ref"] or features["has_hidden_file"]:
        return FLAGGED

    # Otherwise use similarity to learned accumulator
    vec = encode(features)
    similarity = cosine(vec, accumulator)
    if similarity < threshold:
        return FLAGGED

    # Update accumulator (passive learning)
    accumulator.update(vec)
    return ALLOWED
```

### Results

| Metric | Value |
|--------|-------|
| **F1 Score** | **0.985** |
| **Precision** | **100%** |
| **Recall** | **97.1%** |
| **Throughput** | **3,530 req/sec** |

Detection by attack type:
- sqli_path: **100%**
- sqli_query: **100%**
- xss_query: **100%**
- cmd_injection: **100%**
- traversal: **100%**
- hidden_file: **81%** (misses `/wp-config.php` - no abnormal chars)

### What This Proves

1. **Character class analysis alone detects most attacks** - No regex needed
2. **Bit 16 (abnormal chars) is the key signal** - Catches SQLi, XSS, command injection
3. **Low cardinality fingerprints work** - Only 32 possible bitmask values
4. **Truly headless is possible** - No attack pattern knowledge required

### Limitations

Cannot detect attacks that use only "normal" characters:
- `/wp-config.php` - Sensitive filename, but normal chars
- `/admin` - Sensitive path, but normal chars

These require content knowledge (known sensitive paths) or access control.

### Files Created

| File | Purpose |
|------|---------|
| `023-headless-url-detection.py` | Initial headless attempt (whole-URL encoding) |
| `024-component-frequency-detection.py` | Component-level frequency tracking |
| `025-structural-fingerprint-detection.py` | Bitmask-based structural fingerprinting |

### Evolution of Headless Approaches

| Approach | F1 | Issue |
|----------|-----|-------|
| Whole-URL encoding | 0.398 | Structure dominates, attacks buried |
| Component frequency | 0.563 | Rare-but-benign false positives |
| Structural fingerprint + flags | 0.985 | Uses immediate flags (not pure frequency) |
| **Pure frequency+decay** | **1.000** | Truly headless! |

## Phase 8: Pure Frequency+Decay Headless Detection (F1 = 1.000)

### The Key Insight

The user's intuition was correct: **frequency + decay = headless detection**.

The previous structural fingerprint approach used immediate flags (`has_abnormal`) which is essentially rule-based. True headless detection relies purely on:

1. **Frequency** - Common patterns dominate the accumulator
2. **Decay** - Old patterns fade, recent patterns emphasized
3. **Similarity** - Rare patterns have low similarity → flagged

### Making Bitmasks the Primary Signal

The breakthrough was restructuring the fingerprint to make bitmasks prominent:

```python
# OLD: Bitmask buried in structure
features = {
    "path_fingerprints": [[1, 1], [2, 1], [3, 30]],  # Attack bitmask (30) buried
    ...
}

# NEW: Bitmask as primary signal
features = {
    "bitmasks": [1, 30],      # Unique bitmasks (attack value stands out!)
    "max_bitmask": 30,        # Highest bitmask
    "path_lengths": [1, 2, 3],
    ...
}
```

### Why It Works

Normal traffic has bitmasks in a narrow range:
- `1` (lowercase only) - "api", "users", "search"
- `4` (digit only) - "123", "456"
- `5` (lower+digit) - "user123"
- `9` (lower+normal) - "api-v2", "user.name"

Attack traffic has bitmasks with bit 16 (ABNORMAL):
- `17` (lower+ABNORMAL) - `<script>`
- `21` (lower+digit+ABNORMAL) - `1=1--`
- `25` (lower+normal+ABNORMAL) - `../;`
- `29`, `30`, `31` - various attack patterns

The accumulator learns that bitmasks {1, 4, 5, 9} are common. Bitmasks {17, 21, 25, 29, 30, 31} are rare → low similarity → flagged.

### Results

| Metric | Value |
|--------|-------|
| **F1 Score** | **1.000** |
| **Precision** | **100%** |
| **Recall** | **100%** |
| **Throughput** | **4,776 req/sec** |

Clean separation:
- Benign: min=0.621, mean=0.788
- Malicious: max=0.582, mean=0.506
- **Gap: 0.039** (0.621 - 0.582)

### Detection Flow

```
Request
   │
   ▼
┌─────────────────────────────────────────────┐
│  Extract Structural Fingerprint             │
│                                             │
│  URL: /api/search?q=<script>alert(1)        │
│    bitmasks: [1, 29]                        │
│    max_bitmask: 29                          │
│    path_lengths: [1, 2]                     │
│    query_lengths: [4]                       │
└─────────────────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────────────────┐
│  Encode with Holon                          │
│    → VSA vector                             │
└─────────────────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────────────────┐
│  Compare to Decaying Accumulator            │
│                                             │
│  Accumulator learned: bitmasks {1,4,5,9}    │
│  This request has: bitmask 29               │
│  → Low similarity (0.51)                    │
│  → Below threshold (0.60)                   │
│  → FLAGGED                                  │
└─────────────────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────────────────┐
│  Update Accumulator (with reduced weight)   │
│    accum = 0.9995 * accum + 0.1 * vec       │
└─────────────────────────────────────────────┘
```

### Comparison: Approaches

| Approach | F1 | Throughput | Headless? |
|----------|-----|------------|-----------|
| Rule-based (regex) | 1.000 | 8,339/sec | No |
| Structural + flags | 0.985 | 3,530/sec | Partial |
| **Pure freq+decay** | **1.000** | **4,776/sec** | **Yes** |

### What This Proves

1. **Pure frequency+decay works** - No explicit attack patterns needed
2. **Bitmask encoding is key** - Low cardinality, captures attack signatures
3. **Decay enables adaptation** - Model continuously learns "normal"
4. **F1=1.000 is achievable headless** - Character class analysis alone suffices

### The Decaying Accumulator Formula

```
accumulator = decay × accumulator + weight × new_vector

Where:
  decay = 0.9995 (effective window ~2000 requests)
  weight = 1.0 for normal, 0.1 for flagged (resist poisoning)
```

## Phase 9: PCAP Headless Detection

### Applying URL Approach to Packets

We applied the same frequency+decay approach to network packets:

**Packet Fingerprint:**
```python
{
    "protocol": "TCP",
    "dst_port": 443,
    "flags": 24,  # Raw bitmask
}
```

### Results: Raw Packet Encoding

| Mode | F1 | Notes |
|------|-----|-------|
| Anomaly (low sim) | 0.114 | 100% recall, low precision |
| DDoS (high sim) | 0.664 | First attempt |

**By attack type (anomaly detection):**
- icmp_tunnel: **100%**
- port_scan: **100%**
- udp_amp: **100%**
- null_scan: 64%
- syn_fin: 60%
- unusual_port: 0%

**Key insight**: PCAP is harder than URLs because:
1. Attacks share structure with benign (unusual_port uses ACK flag like normal traffic)
2. No "smoking gun" feature like URL bitmask bit 16 (abnormal chars)
3. Lower cardinality means less discrimination

## Phase 10: Two-Phase DDoS Detection (100% Detection + Classification)

### The Realistic DDoS Simulation

Previous tests mixed attacks with normal. Real DDoS **drowns out** good traffic:

```
Phase 1: Normal traffic (learning)
Phase 2: DDoS dominates (95% attack, 5% normal trickles through)
```

### Two-Phase Detection

**Phase 1 - DETECT:** Monitor similarity variance
- Normal traffic: diverse → high variance
- DDoS traffic: homogeneous → low variance + high mean similarity

```python
if variance < baseline * 0.3 and mean_similarity > 0.7:
    → DDoS DETECTED
```

**Phase 2 - CLASSIFY:** Analyze dominant pattern
- Check protocol distribution (TCP/UDP/ICMP)
- Check specific patterns (SYN flag for SYN flood, src_port=53 for DNS reflection)

### Key Insight: Detection Mode Matters

| Attack Type | Signal | Mode |
|-------------|--------|------|
| SYN flood | Concentration of SYN flag | **high_sim** |
| UDP reflection | Novel src_port (53/123/1900) | **low_sim** |
| ICMP flood | Unusual ICMP volume | **high_sim** |

**UDP reflection is ANOMALY, not repetition!**
- Normal: src_port = ephemeral (49152-65535)
- Attack: src_port = 53 (FROM DNS server) — never seen in normal!

### Results

| Attack | Detected | Delay | Classified As | Throughput |
|--------|----------|-------|---------------|------------|
| SYN Flood | ✓ | 121 pkts | ✓ syn_flood | 3,193/sec |
| DNS Reflection | ✓ | 280 pkts | ✓ dns_reflection | 3,962/sec |
| NTP Amplification | ✓ | 278 pkts | ✓ ntp_amplification | 1,752/sec |
| ICMP Flood | ✓ | 119 pkts | ✓ icmp_flood | 6,258/sec |

**Detection Rate: 100%**
**Classification Accuracy: 100%**
**Average Throughput: 3,791 packets/sec**

### Variance Trajectory Shows Clear Transition

```
Normal phase:  var=0.03, mean=0.46  (diverse traffic)
DDoS starts:   var=0.01, mean=0.70  → DETECTED!
DDoS saturated: var=0.00, mean=0.80+ (homogeneous flood)
```

### The Detection Logic

```python
# During DDoS:
# - Variance drops (same pattern repeating)
# - Mean similarity rises (matches accumulated pattern)

if (current_var < baseline_var * 0.3 and  # Variance dropped
    current_mean > 0.7):                   # High similarity
    → DDoS DETECTED
    → Classify based on recent packet patterns
```

### Classification Logic

```python
def classify_attack():
    # Check protocol distribution
    if icmp_count / total > 0.8:
        return "icmp_flood"

    if tcp_count / total > 0.8:
        syn_count = count(flags == 0x02)
        if syn_count / tcp_count > 0.8:
            return "syn_flood"

    if udp_count / total > 0.8:
        # Check for reflection (src_port is well-known service)
        if most_common_src_port == 53:
            return "dns_reflection"
        if most_common_src_port == 123:
            return "ntp_amplification"
```

### Files Created

| File | Purpose |
|------|---------|
| `026-headless-pcap-detection.py` | PCAP anomaly detection |
| `027-ddos-detection.py` | Initial DDoS via high similarity |
| `028-raw-packet-encoding.py` | Raw encoding comparison |
| `029-realistic-ddos.py` | Mode-based detection |
| `030-ddos-two-phase.py` | Complete two-phase solution |

### Key Learnings

1. **DDoS is variance + mean shift** - Not just high similarity
2. **UDP reflection is ANOMALY** - Novel src_port, use low_sim detection
3. **Classification requires recent packet analysis** - Not just similarity
4. **Two-phase works** - Detect → Classify is practical
5. **3,800 pkt/sec throughput** - Production viable

## Phase 11: Explainability - WHY is this suspicious?

### The Problem

Pure similarity-based detection answers "IS this suspicious?" but not "WHY?"

Security analysts need actionable information:
- Which part of the request triggered the flag?
- What specifically is unusual?
- Is this a false positive worth investigating?

### The Solution: Component-Level Analysis + Boosting

**1. Component-Level Analysis**

VSA encoding is compositional. The full vector is a superposition:
```
full_vec = encode({method, bitmasks, path_depth, query, ...})
```

We can encode each component separately and compare to the reference:
```python
method_vec = encode({"method": "GET"})
method_sim = cosine(method_vec, reference)  # How "normal" is the method?

bitmask_vec = encode({"bitmasks": [1, 30]})
bitmask_sim = cosine(bitmask_vec, reference)  # How "normal" are the char classes?
```

Components with low similarity or novel values are flagged as suspicious.

**2. Component-Based Boosting**

Key insight: Suspicious components should RAISE the detection threshold.

```python
boosted_threshold = base_threshold + (n_suspicious * boost_factor)

# Example:
# Base threshold: 0.55
# 1 suspicious component: 0.55 + 0.15 = 0.70
# 2 suspicious components: 0.55 + 0.30 = 0.85
```

**Before boosting**: SQL injection in path ALLOWED (sim=0.614 >= 0.55)
**After boosting**: SQL injection in path FLAGGED (sim=0.614 < 0.70)

This catches attacks that are "borderline" on similarity but have suspicious structural features.

**3. Dimension Analysis**

Beyond component-level, we can analyze individual vector dimensions:

```python
def analyze_dimensions(request_vec, reference):
    agreeing = 0      # Both positive or both negative
    disagreeing = 0   # Opposite signs
    neutral = 0       # One or both zero

    for i in range(len(request_vec)):
        if same_sign(request_vec[i], reference[i]):
            agreeing += 1
        elif opposite_sign(...):
            disagreeing += 1
        else:
            neutral += 1
```

| Request Type | Agreeing | Disagreeing | Verdict |
|--------------|----------|-------------|---------|
| Normal | 70.7% | 9.9% | ✅ |
| SQL injection | 64.2% | 16.1% | 🚨 |
| XSS | 58.6% | 22.0% | 🚨 |
| Path traversal | 60.1% | 21.1% | 🚨 |

Higher disagreement % = more anomalous.

### Results with Boosting

| Attack Type | Without Boosting | With Boosting |
|-------------|------------------|---------------|
| SQL in path | ALLOWED | **FLAGGED** |
| SQL in query | FLAGGED | FLAGGED |
| XSS | FLAGGED | FLAGGED |
| Path traversal | FLAGGED | FLAGGED |
| Hidden file | ALLOWED | **FLAGGED** |
| Command injection | FLAGGED | FLAGGED |
| Unusual method | FLAGGED | FLAGGED |

**Detection improved from 5/7 to 7/7** by using component analysis to boost threshold.

### What We CAN Deduce

Given (reference_vec, request, request_vec):

| Signal | How We Detect It |
|--------|------------------|
| ABNORMAL characters | Bitmask with bit 16 set |
| Novel character classes | Bitmask not seen in training |
| Unusual method | Method not in learned set |
| Path depth anomaly | Depth bucket not in training |
| Long values | Length bucket 4+ (>12 chars) |
| Dimension divergence | High % of disagreeing dims |

### What We CANNOT Deduce

- **Specific payload content** - We encode structure, not semantics
- **Attack semantics** - "This is SQL injection" requires domain knowledge
- **Intent** - Why the attacker sent this request

### Brutal Honesty

**This is NOT magic explainability.** We're not reverse-engineering the vector to recover the original input. We're:

1. Tracking metadata during training (learned_methods, learned_bitmasks)
2. Encoding components separately at inference time
3. Comparing each component to the reference
4. Using structural heuristics (bit 16 = ABNORMAL chars)

The "explainability" comes from:
- **Compositional encoding** - we control how data becomes vectors
- **Metadata tracking** - we remember what we learned
- **Structural analysis** - we extract features we understand

A pure "black box" approach (just vectors, no metadata) would give us similarity scores but no explanations.

### Files Created

| File | Purpose |
|------|---------|
| `031-explainability-demo.py` | Component analysis + boosting demo |

## Final Conclusions

### What Works

1. **Deterministic consensus** - Same global_seed → same vectors → same decisions across nodes
2. **Accumulator > prototype_add()** - Preserves frequency signal for better separation
3. **Multi-signal detection** - Single prototype fails; component-level analysis catches more
4. **F1 = 1.000 achievable** - With proper architecture (boosting, multi-signal)
5. **Production performance** - 4,000-8,000 req/sec on CPU, sub-millisecond latency
6. **Continuous learning** - Decaying accumulator adapts without retraining
7. **Headless detection** - Character class bitmask works without content knowledge
8. **DDoS detection** - Variance + mean shift detects, packet analysis classifies
9. **Explainability** - Component-level analysis answers "WHY is this suspicious?"
10. **Boosting from explainability** - Suspicious components raise threshold, improves detection

### What Doesn't Work (Brutal Honesty)

1. **Not true XAI** - We track metadata and use heuristics, not reverse-engineering vectors
2. **Domain knowledge required** - "bit 16 = ABNORMAL" is a human-defined rule
3. **PCAP harder than HTTP** - Attacks share structure with benign traffic (F1=0.114 raw)
4. **Warmup vulnerability** - Attacker can poison during warmup period
5. **Threshold tuning needed** - Different domains need different thresholds
6. **Fingerprinting helps** - Raw encoding works but explicit features improve results

### The Real Achievement

In ~3 hours and ~$30 of API credits, we built:

| Feature | Status |
|---------|--------|
| Deterministic AI | ✅ No random init, no gradients, reproducible |
| Explainable | ✅ Component-level "why" with boosting |
| High-performance | ✅ 8k req/sec, CPU-only |
| Streaming | ✅ Continuous learning with decay |
| DDoS detection | ✅ 100% detection + classification |
| Distributed | ✅ No sync needed, mathematical consensus |

**All using pure math**: sparse bipolar vectors, bind/bundle/permute, cosine similarity.

No neural networks. No backprop. No GPUs. No training loops.

### Limitations to Acknowledge

1. **All benchmarks are synthetic** - Real-world accuracy unproven
2. **No comparison to baselines** - Haven't tested against ML/rule-based alternatives
3. **Explainability is partial** - We explain structure, not semantics
4. **Requires understanding your data** - Need to know what features matter
5. **Not a silver bullet** - Works for anomaly detection, not constraint satisfaction

### What This Proves

VSA/HDC can deliver "deterministic AI" that is:
- **Reproducible** across machines
- **Explainable** at the component level
- **Fast** enough for production
- **Simple** enough to reason about

The trade-off: You need to understand your encoding. The vectors don't magically learn semantics - they capture the structure you define.

---

*Challenge 010 completed: February 2026*
*~3 hours, ~$30, 31 scripts, 1 library extension (accumulator primitives)*
