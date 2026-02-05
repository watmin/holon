# Challenge 011 Learnings: Scoped Vectors & Adaptive Dynamics

## Overview

This batch explored **component-level detection** and **adaptive temporal dynamics** building on batch 010's success with accumulators and continuous learning.

## Key Experiments

### 001: Scoped HTTP Vectors

**Concept**: Separate accumulators per HTTP component (path, query, headers, body).

**Results**:

| Aggregation | F1    | Precision | Recall |
|-------------|-------|-----------|--------|
| ANY         | 0.068 | 3.5%      | 82.7%  |
| VOTING (2)  | 0.063 | 3.5%      | 29.1%  |
| WEIGHTED    | 0.000 | 0.0%      | 0.0%   |
| ALL         | 0.000 | 0.0%      | 0.0%   |

**Findings**:

1. **Headers component triggers too often** - threshold (0.45) too aggressive for header variety
2. **Path SQLi missed** - similarity 0.614 above threshold 0.55 because structural features similar
3. **Body/Query attacks detected well** - similarity drops to 0.12 for SQLi/XSS in these components
4. **Per-component detection provides natural explainability** - "query component anomalous"

**Key insight**: Scoped vectors work but need:
- Per-component threshold tuning
- Bitmask-based boosting from batch 010
- Exclude or de-weight noisy components (headers)

### 002: Scoped PCAP Fields

**Concept**: Separate accumulators per packet field (src_addr, dst_port, protocol, flags, etc.)

**Results**:

| Threshold | F1    | Precision | Recall |
|-----------|-------|-----------|--------|
| 1         | 0.320 | 19.0%     | 100%   |
| 2         | 0.363 | 22.2%     | 100%   |
| 3         | 0.739 | 68.3%     | 80.5%  |

**Attack detection by field triggers**:

| Attack Type | Triggering Fields |
|-------------|-------------------|
| DNS reflection | src_addr, src_port, protocol |
| NTP amplification | src_addr, src_port, protocol |
| SYN flood | src_addr, size, flags |
| Port scan | src_addr, size, flags |
| ICMP flood | src_addr, src_port, protocol |
| Unusual port | size, flags (only 2) |

**Key insights**:

1. **src_port is KEY for reflection** - normal traffic uses ephemeral (49152+), reflection uses 53/123
2. **flags catches SYN floods** - pure SYN (0x02) vs normal mix
3. **Different attacks have distinct field signatures** - enables classification from explainability
4. **Threshold=3 misses unusual_port** - only 2 fields trigger, need adaptive thresholds

### 003: N-gram Text Encoding

**Concept**: Character and token n-grams to capture attack sequences.

**Results**:

| Config | F1    | Precision | Recall |
|--------|-------|-----------|--------|
| char=2, token=2 | 0.191 | 10.5% | 100% |
| char=3, token=2 | 0.191 | 10.5% | 100% |
| char=4, token=3 | 0.097 | 5.1%  | 100% |

**Similarity distribution**:
- Benign: min=0.007, mean=0.414, max=0.626
- Malicious: min=0.128, mean=0.271, max=0.334

**Key insights**:

1. **100% recall on all attack types** - n-grams DO capture attack patterns
2. **Low precision** - many false positives, threshold needs tuning
3. **Good separation exists** - benign mean (0.414) vs malicious mean (0.271)
4. **Attack signatures visible**:
   - SQL: `"' O", " OR", "R '"` character n-grams
   - XSS: `"<sc", "scr", "ipt"` character n-grams
   - Token: `"admin_'", "'_--"` semantic structure

**Recommendation**: Combine n-grams with structural fingerprints:
- Fingerprints: fast rejection based on char class
- N-grams: detailed pattern matching for borderline cases

### 004: Adaptive Decay Mechanics

**Concept**: Explore decay strategies for variable-rate environments.

**Fixed decay impact at 100k pkt/sec DDoS**:

| Decay | Half-life |
|-------|-----------|
| 0.99  | 0.7 ms    |
| 0.995 | 1.4 ms    |
| 0.999 | 6.9 ms    |
| 0.9995| 13.9 ms   |
| 0.9999| 69.3 ms   |

**Multi-horizon divergence during attack**:

| Time | Divergence | Phase |
|------|------------|-------|
| 0.95 | 0.003      | normal |
| 1.10 | 0.167      | attack <<<|
| 1.25 | 0.211      | attack <<<|
| 1.75 | 0.029      | normal |

**Key insights**:

1. **Time-based decay** provides consistent memory regardless of rate
   - `decay = exp(-ln(2) * elapsed / half_life_seconds)`

2. **Rate-adaptive decay** automatically adjusts to traffic rate
   - Target window in seconds, compute decay from rate

3. **Multi-horizon best for change detection**:
   - Fast (100 obs half-life): current traffic
   - Slow (2000 obs half-life): baseline
   - Divergence = attack transition signal

4. **Divergence clearly signals attack**:
   - Normal: ~0.003
   - Attack: ~0.17-0.21 (5-70x higher)

### 005: Smart Checkpoints

**Concept**: Frozen "known good" states for recovery detection.

**Results**:
- Attack detected at packet 2277 (delay: 1277 from attack start)
- Recovery detection triggered prematurely at 2278 (during attack)

**Issues identified**:
- State machine conflict: homogeneous attack traffic has low variance, triggering both attack AND recovery conditions
- Need hysteresis or separate conditions for each phase

**Key insights**:

1. **Checkpoint mechanism works** - can save and compare to frozen state
2. **Recovery detection needs refinement**:
   - Don't use variance alone (attack is homogeneous = low variance)
   - Use divergence from checkpoint + divergence from slow accumulator
   - Add hysteresis (require sustained condition)

3. **Checkpoint strategies validated**:
   - Periodic: every N observations during stability
   - Stability-triggered: when variance low for extended period
   - Post-recovery: after returning to baseline

## Summary: What Works

### Scoped Vectors
- ✅ Natural explainability ("src_port anomaly" = likely reflection)
- ✅ Per-component thresholds for tuning
- ✅ Field-level attack signatures
- ⚠️ Need threshold tuning per component
- ⚠️ Some components (headers) too noisy

### Adaptive Decay
- ✅ Multi-horizon clearly signals transitions
- ✅ Time-based decay for variable rates
- ✅ Rate-adaptive for unknown environments
- ⚠️ Need to pick right strategy for use case

### N-grams
- ✅ Captures attack sequences
- ✅ 100% recall possible
- ⚠️ High false positive rate without tuning
- ⚠️ Higher cardinality than structural fingerprints

### Checkpoints
- ✅ Enables recovery detection
- ✅ Resists poisoning
- ⚠️ State machine logic needs refinement
- ⚠️ Need hysteresis for stable transitions

## Recommendations for Production

### HTTP Detection
```python
# Combine scoped vectors + bitmask boosting
detector = ScopedHttpDetector(
    components=["path", "query", "body"],  # Exclude headers
    aggregation="voting",
    voting_threshold=2,
    boost_on_abnormal_bitmask=True,  # From batch 010
)
```

### PCAP Detection
```python
# Scoped fields with attack-specific thresholds
detector = ScopedPacketDetector(
    voting_threshold=2,
    # Lower threshold for reflection-key fields
    thresholds={
        "src_port": 0.35,  # Key for reflection
        "flags": 0.40,     # Key for SYN flood
        "dst_port": 0.50,
        ...
    },
)
```

### Decay Strategy
```python
# For DDoS environments with variable rates
detector = MultiHorizonDetector(
    fast_half_life=100,   # Recent trend
    slow_half_life=2000,  # Baseline
    divergence_threshold=0.10,  # Attack signal
)
```

### Checkpoints
```python
# With hysteresis for stable transitions
checkpoint_mgr = CheckpointManager(
    periodic_interval=1000,
    stability_window=100,
    attack_to_recovery_delay=500,  # Require sustained recovery
)
```

## Files Created

| File | Purpose |
|------|---------|
| `001-scoped-http-vectors.py` | Per-component HTTP accumulators |
| `002-scoped-pcap-fields.py` | Per-field packet accumulators |
| `003-ngram-text-encoding.py` | Character/token n-gram encoding |
| `004-adaptive-decay.py` | Decay strategies comparison |
| `005-smart-checkpoints.py` | Checkpoint-based recovery detection |

## Phase 2: Raw Packet Analysis (006-007)

### 006: Multi-Perspective Raw Packet Analysis

Used scapy for real packet crafting with 6 perspectives:
- L3: IP prefix patterns
- L4-ports: Port usage
- L4-flags: TCP flags / ICMP type
- Payload-positional: Byte at position N
- Payload-ngrams: Byte sequences
- Payload-structure: Spread, printability

**Key findings**:

| Attack | Key Anomalous Perspectives |
|--------|---------------------------|
| SYN flood | L4-flags (pure SYN = 0.320 sim) |
| DNS reflection | L4-ports, payload (src_port=53) |
| Random payload | payload_ngram, payload_struct |

**Cardinality as detection signal**:
- Normal: src_prefix 0.82 (diverse)
- SYN flood: dst_prefix 0.01 (single target)
- DNS reflection: src_prefix 0.01 (single source)

### 007: Deviation Detection & Cardinality Fusion

**Surprise metric** (1 - similarity):

| Traffic | L3 Surprise | L4 Surprise | Payload Surprise |
|---------|-------------|-------------|------------------|
| Normal | 0.00 | 0.00 | 0.00 |
| SYN flood | 0.95 | 0.78 | 1.00 |
| DNS reflection | 0.89 | 0.94 | 0.95 |

**Agreement ratio** (entropy-like without info theory):

| Traffic | L3 Agreement | L4 Agreement | Payload Agreement |
|---------|--------------|--------------|-------------------|
| Normal | 1.00 | 1.00 | 1.00 |
| SYN flood | 0.57 | 0.57 | 0.17 |
| DNS reflection | 0.49 | 0.42 | 0.47 |

**Detection rate**: 100% for both attack types

### Key Insights from Raw Packet Analysis

1. **Multi-perspective is essential** - Different attacks trigger different perspectives
2. **Agreement ratio = entropy without math** - Just count agreeing dimensions
3. **Cardinality drop = focused attack** - The CHANGE is the signal
4. **Payload spread captures content entropy** - unique_bytes is encodable

## Architecture Recommendation

```python
# The full detection stack
detector = MultiPerspectiveDetector(
    perspectives=[
        L3Perspective(prefix_levels=[16, 24]),
        L4Perspective(track_flags=True, track_reflection_ports=True),
        PayloadPerspective(positional=True, ngrams=True, structure=True),
    ],
    accumulators={
        "fast": decay_for_half_life(100),   # Recent trend
        "slow": decay_for_half_life(2000),  # Baseline
    },
    cardinality_tracker=CardinalityAnalyzer(window=100),
    byte_tracker=BytePositionTracker(max_positions=64),
)

# Detection signals
def is_anomalous(report):
    return (
        any(surprise > 0.7 for surprise in report.surprises)
        or any(cardinality_dropped(field) for field in report.fields)
        or agreement_ratio < 0.5
    )
```

## Phase 3: Realistic Streaming Deployment (008)

### 008: Production-Ready Streaming Architecture

Built a deployment-ready system with:

**Prior Knowledge (frozen, from training)**:
- Baseline vectors per perspective (L3, L4, Payload)
- Expected field values and frequencies
- Expected byte distributions per position
- Serializable to disk (`pickle`)

**Recent Knowledge (adaptive, from stream)**:
- Decaying accumulators track recent patterns
- Sliding window cardinality per field
- Reduced update weight for anomalies (resist poisoning)

**Culprit Identification**:
- Pinpoints specific unusual fields/values
- Shows expected vs observed
- Severity rating (high/medium/low)
- Human-readable explanations

**Results**:

| Traffic Type | Detection Rate | False Positive Rate |
|--------------|----------------|---------------------|
| Normal | N/A | 0% (after warmup) |
| SYN Flood | 100% | N/A |
| DNS Reflection | 100% | N/A |
| Unknown Protocol | 100% | N/A |

**Culprit Examples**:

```
SYN Flood packet:
  🚨 ANOMALY (score: 1.00)
  Culprits:
    [high] src_prefix=172.16 (Expected: ['192.168'])
    [high] flags=2 (Expected: ['24'])

DNS Reflection:
  🚨 ANOMALY (score: 1.00)
  Culprits:
    [high] src_port=53 is a well-known port (possible reflection attack)
    [high] dst_port=59584 (Expected: ['80', '53', '443'])

Unknown Protocol (port 31337):
  🚨 ANOMALY (score: 0.92)
  Culprits:
    [high] dst_port=31337 (Expected: ['80', '53', '443'])
    [medium] payload_byte_0=0xde (Expected: ['0x47', '0x16', '0x20'])
```

**Key Design Decisions**:

1. **Skip high-cardinality fields for culprits** - ephemeral ports are expected to vary
2. **Special case for reflection** - well-known ports as SOURCE = suspicious
3. **Weight perspectives by reliability** - L4 and Payload more reliable than L3
4. **Boost score for high-severity culprits** - specific findings increase confidence

## Library Promotion: Deterministic Vector Generation

During this batch, we identified that `DeterministicVectorManager` was being copied between
challenge directories (010 → 011). This class provides critical functionality:

- **Order-independent vector generation**: Hash-based seeding ensures the same atom always
  produces the same vector, regardless of when or where it's first seen
- **Distributed consensus**: Multiple nodes with the same `global_seed` produce identical
  vectors without coordination
- **Reproducibility**: Results are deterministic across runs, machines, and time

**Decision**: Merged into `VectorManager` with `deterministic=True` as the default.

**Key Changes**:
1. `VectorManager` now defaults to `deterministic=True` (hash-based, order-independent)
2. `deterministic=False` is deprecated and emits a warning
3. `DeterministicVectorManager` is now a thin subclass for backwards compatibility

**Usage**:
```python
from holon import VectorManager

# Default: deterministic, order-independent (RECOMMENDED)
vm = VectorManager(global_seed=42)  # dimensions=16000, deterministic=True
vec = vm.get_vector("some_atom")  # Same vector every time, everywhere

# Legacy: order-dependent (DEPRECATED)
vm_legacy = VectorManager(deterministic=False)  # Emits DeprecationWarning

# Backwards compatibility alias (uses VectorManager under the hood)
from holon import DeterministicVectorManager
vm_compat = DeterministicVectorManager(dimensions=4096, global_seed=42)
```

**Implementation Notes**:
- Default dimensions remain 16000 (matching original VectorManager) for backwards compatibility
- Vector distribution uses `choice([-1, 0, 1])` (equal 1/3 probability) matching original
- Hash-based seeding: `SHA256(atom) XOR global_seed` for order-independence

**Test Coverage**: 31 unit tests covering determinism, order-independence, distributed
consensus, caching, and backwards compatibility.

## Phase 4: Production Readiness (Experiments 009-012)

### Experiment 009: Attack Classification

Automatic mapping of culprit patterns to specific attack types using hybrid rule-based
and VSA-based signature matching.

**Results:**
- 5/6 attack types correctly classified (83% accuracy)
- SYN Flood, UDP Flood, DNS Amplification, Port Scan, IP Sweep all detected
- Normal traffic misclassified (needs "none of the above" threshold)

**Key Differentiators:**
| Attack Type | Key Indicators |
|-------------|----------------|
| Reflection attacks | Well-known port as SOURCE (port < 1024) |
| Flood attacks | High source IP cardinality |
| Scans | High destination port/IP cardinality |
| Amplification | Large payload + reflection |

### Experiment 010: Prior Knowledge Update Mechanism

Safe update strategies to prevent accepting attack traffic as new baseline.

**Strategy Comparison:**

| Strategy | Attack Blocked | Evolution Accepted | Score |
|----------|----------------|-------------------|-------|
| REPLACE | ✗ | ✓ | 1/2 |
| GRADUAL_BLEND | ✗ | ✓ | 1/2 |
| VALIDATION_WINDOW | ✓ | ✓ | 2/2 |
| SIMILARITY_GATE | ✓ | ✗ | 1/2 |

**Best Practice:** VALIDATION_WINDOW (reject updates with high anomaly rate during
validation period) achieves best balance of security and adaptability.

### Experiment 011: Multi-Node Consensus

Distributed detection with shared priors across multiple sensors.

**Results:**
- 100% vector agreement across all nodes (same `global_seed` = identical vectors)
- No coordination needed for atom → vector mapping
- Accumulator merging works (post-merge similarity = 100%)

**Architecture:**
```
[Prior Knowledge] ← shared, frozen (generated once)
       ↓
┌──────┴──────┬──────┴──────┐
│             │             │
Node A      Node B      Node C   ← each with own recent knowledge
│             │             │
└──────┬──────┴──────┬──────┘
       ↓
[Periodic Merge] ← optional convergence
```

### Experiment 012: Performance Benchmark

Throughput testing at scale.

**Results:**
```
Best throughput: 14,604 pps (dim=2048)
10k pps target: ✓ ACHIEVABLE

Component throughput (dim=4096):
  Parsing (Scapy):     48,397 pps
  Encoding (VSA):      20,103 pps
  Similarity (cosine): 150,439 pps
  Full detector:       12,198 pps
```

**Bottleneck:** VSA encoding (hash + vector lookup + accumulation)

**Optimization Strategies:**
- Lower dimensions (2048 vs 4096) for ~25% speedup
- Raw packet parsing instead of Scapy for ~2-3x improvement
- C/Rust implementation for 10-100x improvement

## Complete File List

```
scripts/challenges/011-batch/
├── 001-scoped-http-vectors.py      # Per-component HTTP accumulators
├── 002-scoped-pcap-fields.py       # Per-field packet accumulators
├── 003-ngram-text-encoding.py      # Character/token n-grams
├── 004-adaptive-decay.py           # Decay strategies
├── 005-smart-checkpoints.py        # Recovery detection
├── 006-raw-packet-multiperspective.py  # Scapy multi-perspective
├── 007-deviation-detection.py      # Surprise/agreement metrics
├── 008-streaming-deployment.py     # Production architecture
├── 009-attack-classification.py    # Automatic attack type detection
├── 010-prior-update.py             # Safe baseline update strategies
├── 011-multi-node-consensus.py     # Distributed detection
└── 012-performance-benchmark.py    # Throughput testing
```

## Phase 5: F1 Optimization & Three-Dimensional Detection (013-015)

### Critical Discovery: Use Holon's Structural Encoding

Experiments 013-014 revealed a **fundamental mistake**: we were using naive atom bundling:

```python
# WRONG: Naive atom bundling (loses structure)
atoms = ["proto:tcp", "dst_port:80", "flags:PA"]
vec = sum(vm.get_vector(atom) for atom in atoms)
# Result: F1 = 0.368
```

Instead, Holon's power comes from **structural encoding with role-filler binding**:

```python
# CORRECT: Structural encoding (preserves structure)
structure = {"l4": {"proto": "tcp", "dst_port": 80, "flags": "PA"}}
vec = encoder.encode_data(structure)
# Result: F1 = 1.000
```

**Why structural encoding works:**
- `{dst_port: 80}` and `{src_port: 80}` share "80" but are bound to different roles
- Nested structure preserved: `l4.proto`, `l4.flags` are separate bindings
- The encoder handles `key × value` binding and bundling automatically

### Experiment 014: Structural Detection Results

Converting packets to nested dicts and using `encoder.encode_data()`:

| Metric | Naive Bundling | Structural Encoding |
|--------|----------------|---------------------|
| Binary F1 | 0.368 | **1.000** |
| Normal→Baseline Similarity | 0.355 | **0.867** |
| Attack→Baseline Similarity | 0.161 | **0.092** |

The structural encoding creates **massive separation** (0.77 gap vs 0.19 gap).

### Experiment 015: Three-Dimensional Detection

Focused on user's three key dimensions:

**Dimension 1: Transition Detection (beginning/ending)**
- `stable_normal → attack_beginning → stable_attack → attack_ending → stable_normal`
- State machine driven by anomaly/normal streaks
- **F1 = 0.936**

**Dimension 2: Attack Classification**
- SYN flood, UDP flood, DNS reflection, port scan, ICMP flood
- Build signatures from actual attack samples (not handcrafted)
- **F1 = 0.998**

**Dimension 3: Knowledge Composition**
- Prior: Frozen baseline (0.82 normal, 0.12 attack similarity)
- Recent: Decaying accumulator tracking current traffic
- Compositional: Prior/recent divergence (0.97 normal, 0.91 attack)
- **Binary F1 = 1.000**

### Final Results

| Dimension | F1 Score |
|-----------|----------|
| Transition Detection | **0.936** |
| Attack Classification | **0.998** |
| Binary Detection | **1.000** |

### Key Learnings

1. **Always use `encoder.encode_data()` for structured data** - don't roll your own atom bundling
2. **Build attack signatures from samples** - not handcrafted structures
3. **Realistic transition labeling matters** - `attack_ending` = first normal packets AFTER attack
4. **Prior knowledge provides massive separation** - 0.82 vs 0.12 similarity is definitive
5. **Compositional signal (prior/recent divergence)** detects regime changes

### Updated Architecture Recommendation

```python
from holon import CPUStore

# 1. Use Holon's encoder (role-filler binding)
store = CPUStore(dimensions=4096)
encoder = store.encoder

# 2. Convert packets to nested structure
def packet_to_structure(pkt):
    return {
        "l3": {"src_net": "192.168", "dst_net": "10.0"},
        "l4": {"proto": "tcp", "dst_port": 80, "flags": "PA"},
        "payload": {"present": True, "size_class": "small"},
    }

# 3. Build baseline from normal samples
baseline = encoder.create_accumulator()
for pkt in normal_samples:
    vec = encoder.encode_data(packet_to_structure(pkt))
    baseline = encoder.accumulate(baseline, vec)

# 4. Build attack signatures from attack samples (not handcrafted!)
signatures = {}
for attack_type in AttackType:
    acc = encoder.create_accumulator()
    for pkt in attack_samples[attack_type]:
        vec = encoder.encode_data(packet_to_structure(pkt))
        acc = encoder.accumulate(acc, vec)
    signatures[attack_type] = encoder.normalize_accumulator(acc)

# 5. Detect using similarity to prior
def detect(pkt):
    vec = encoder.encode_data(packet_to_structure(pkt))
    prior_sim = cosine(vec, encoder.normalize_accumulator(baseline))
    return prior_sim < 0.4  # Anomaly threshold
```

## Phase 6: Cross-Pollination (016-017)

### Experiment 016: Cross-Pollination Tests

Tested techniques from batch 010 applied to 011 and vice versa:

| Experiment | Finding | Improvement |
|------------|---------|-------------|
| A: Smart Normalization | Port bucketing + IP prefixes | Gap: 0.091 → 0.656 **(+0.565)** |
| B: Payload Bitmask | Binary detection via bitmask | Gap: 0.487 |
| C: Variance Detection | Detects DDoS transitions | Works with tuning |
| D: Sample Signatures | 100% classification from samples | Confirms 011 approach |
| E: Prior/Recent Separation | Divergence tracks regime change | 1.0 → 0.48 → 0.91 |

**Key finding**: Smart normalization from 010 dramatically improves 011's detection.

### Experiment 017: Integrated Detector

Combined best techniques from both batches:

**FROM BATCH 010:**
- Port bucketing (wellknown/registered/ephemeral)
- Multi-level IP prefixes (/8, /16)
- Payload character bitmask (256 possible values)
- Rule-based detection (reflection, SYN flood, binary payload)
- Variance-based anomaly detection

**FROM BATCH 011:**
- Structural encoding via `encoder.encode_data()`
- Prior/recent knowledge separation
- State machine for transitions
- Sample-based attack signatures
- Per-field culprit identification

**Results:**

| Metric | Score |
|--------|-------|
| Binary Detection Precision | **1.000** |
| Binary Detection Recall | **1.000** |
| Binary Detection F1 | **1.000** |
| Classification Accuracy | **100%** |

The integrated detector achieves perfect detection by combining:
1. **Rules** for known patterns (fast, precise)
2. **Similarity** for unknown anomalies (adaptive)
3. **Variance** for DDoS detection (homogeneous traffic)
4. **State machine** for transition tracking

## Complete File List

```
scripts/challenges/011-batch/
├── 001-scoped-http-vectors.py      # Per-component HTTP accumulators
├── 002-scoped-pcap-fields.py       # Per-field packet accumulators
├── 003-ngram-text-encoding.py      # Character/token n-grams
├── 004-adaptive-decay.py           # Decay strategies
├── 005-smart-checkpoints.py        # Recovery detection
├── 006-raw-packet-multiperspective.py  # Scapy multi-perspective
├── 007-deviation-detection.py      # Surprise/agreement metrics
├── 008-streaming-deployment.py     # Production architecture
├── 009-attack-classification.py    # Automatic attack type detection
├── 010-prior-update.py             # Safe baseline update strategies
├── 011-multi-node-consensus.py     # Distributed detection
├── 012-performance-benchmark.py    # Throughput testing
├── 013-unified-detection-f1.py     # F1 analysis (naive bundling - don't use)
├── 014-structural-detection.py     # Structural encoding (correct approach)
├── 015-three-dimensions.py         # Three-dimensional detection
├── 016-cross-pollination.py        # Cross-batch technique testing
├── 017-integrated-detector.py      # Combined best of 010 + 011 (F1=1.000)
└── DEMO-batch-011-wrapup.py        # Comprehensive wrap-up demo
```

## Wrap-Up Demo

The `DEMO-batch-011-wrapup.py` showcases all key achievements from this batch in five demonstrations:

```bash
./scripts/run_with_venv.sh python scripts/challenges/011-batch/DEMO-batch-011-wrapup.py
```

### Demo 1: Structural Encoding Discovery
Shows the 94% improvement in separation by using `encoder.encode_data()` with nested structures vs naive atom bundling.

### Demo 2: Three-Dimensional Detection
Simulates normal → attack → recovery phases, demonstrating:
- 100% detection rate during attacks
- 100% classification accuracy
- Proper state machine transitions

### Demo 3: Knowledge Composition
Visualizes prior/recent/divergence during a DDoS scenario:
- Prior similarity drops to 0.16 during attack (baseline unchanged)
- Divergence drops to 0.30 (regime change signal)
- Both recover after attack ends

### Demo 4: Cross-Pollination
Demonstrates the combined power of Batch 010 + 011 techniques achieving F1=1.000.

### Demo 5: Live Detection Simulation
Real-time streaming detection with:
- Multiple attack types (SYN flood, DNS reflection)
- State machine tracking (NORMAL → ATTACK_START → UNDER_ATTACK → RECOVERING)
- Classification on each packet

## Key Achievements Summary

| Achievement | Result |
|-------------|--------|
| Structural encoding discovery | F1: 0.368 → 1.000 |
| Three-dimensional detection | Transition=0.936, Class=0.998, Binary=1.000 |
| Knowledge composition | Prior/recent/divergence separation |
| Cross-pollination | Combined best of 010 + 011 |
| VectorManager refactoring | Deterministic mode now default |
| Integrated detector | F1=1.000, Classification=100% |

## Open Questions for Future Batches

1. **Real PCAP testing**: Validate with actual network captures
2. **GPU acceleration**: Test CuPy backend for higher throughput
3. **Attack signature learning**: Train signatures from labeled attack data
4. **Adaptive thresholds**: Auto-tune detection thresholds per deployment

---

*Challenge 011 completed: February 2026*
*Final experiments 014-015: F1 optimization achieving 0.936/0.998/1.000 across three dimensions*
