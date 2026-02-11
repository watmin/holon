# Challenge Batch 014: Extended Primitives for Explainable VSA

## Philosophy

**Detection is not enough. We need explanation.**

Batches 012-013 achieved zero-hardcode detection with vector-derived rate limits. But when an alert fires, analysts ask:

- **WHEN** did behavior change?
- **WHAT** patterns are present in the anomaly?
- **WHERE** do the vectors differ?
- **WHY** is this anomalous?

This batch introduces 9 new algebraic primitives that answer these questions.

## The Extended Primitives

### 1. `unbind(bound, key) → value`

Explicit API for retrieving values from bound vectors.

```python
# Store: key → value binding
stored = bind(key_vec, value_vec)

# Retrieve: unbind to get value back
retrieved = unbind(stored, key_vec)
# similarity(retrieved, value_vec) ≈ 1.0
```

**Why explicit?** For bipolar vectors, unbind = bind (self-inverse). But making it explicit helps users discover the algebra.

### 2. `similarity_profile(A, B) → vector`

Returns similarity as a VECTOR, not a scalar.

```python
profile = similarity_profile(attack_vec, normal_vec)
# profile[i] = +1 where both agree
# profile[i] = -1 where they disagree
# profile[i] = 0 where either is zero

# "Which dimensions distinguish attack from normal?"
disagree_dims = np.where(profile < 0)[0]
```

**Use case**: Localize WHERE two patterns differ dimension-by-dimension.

### 3. `attend(query, memory, strength, mode) → vector`

Soft attention in VSA algebra.

```python
# Modes:
# - "hard": Binary resonance (same as resonance())
# - "soft": Smooth weighting based on agreement
# - "amplify": Boost agreeing dimensions

attended = attend(attack_signature, packet_vec, strength=2.0, mode="soft")
# Emphasizes dimensions where attack_signature and packet_vec agree
```

**Use case**: Focus analysis on attack-relevant features.

### 4. `segment(stream, window, threshold, method) → [indices]`

Find structural breakpoints in a vector stream.

```python
breakpoints = segment(packet_vectors, window=100, threshold=0.3)
# Returns: [0, 145, 312, 489, ...]
# "Traffic changed significantly at indices 145, 312, 489"
```

**Methods**:
- `"prototype"`: Compare to running prototype (default)
- `"diff"`: Compare consecutive vectors
- `"accumulator"`: Compare to decaying accumulator

**Use case**: Answer "WHEN did behavior change?"

### 5. `analogy(A, B, C) → D`

Relational transfer: A is to B as C is to ?

```python
# If we know how port_80_scan relates to port_443_scan...
# ...we can predict what port_443_exfil looks like from port_80_exfil

port_443_exfil = analogy(port_80_scan, port_443_scan, port_80_exfil)
```

**Use case**: Generalize attack patterns across contexts.

### 6. `project(vec, subspace, orthogonalize) → vector`

Project vector onto subspace defined by exemplars.

```python
# Define attack subspace from known attack prototypes
attack_subspace = [scan_proto, cred_stuff_proto, exfil_proto]

# How much of this packet is in the attack subspace?
projected = project(packet_vec, attack_subspace)
attack_component = norm(projected) / norm(packet_vec)
```

**Use case**: Classify by measuring projection onto known subspaces.

### 7. `complexity(vec) → float`

Measure the "mixedness" of a vector (0.0 to 1.0).

```python
c = complexity(vec)
# Low complexity: Clean signal, single pattern
# High complexity: Superposition of many patterns
```

**Use case**: Detect when traffic becomes "noisy" (attack mixing with normal).

### 8. `conditional_bind(A, B, gate, mode) → vector`

Bind only where gate condition is met.

```python
# Bind source_ip to behavior, but ONLY when status is 4xx
gated = conditional_bind(ip_vec, behavior_vec, status_vec, mode="positive")
# Only dimensions where status_vec > 0 participate in binding
```

**Modes**:
- `"positive"`: Bind where gate > 0
- `"negative"`: Bind where gate < 0
- `"nonzero"`: Bind where gate ≠ 0
- `"strong"`: Bind where |gate| > 75th percentile

**Use case**: Create context-aware, sparse representations.

### 9. `invert(vec, codebook, top_k, threshold) → [(name, sim)]`

Reconstruct components from a vector using a codebook.

```python
# Build codebook of known patterns
codebook = [
    ("normal", normal_proto),
    ("scan", scan_proto),
    ("credential_stuffing", cred_proto),
    ("exfiltration", exfil_proto),
]

# Invert an anomalous vector
components = invert(anomaly_vec, codebook, top_k=3)
# Returns: [("scan", 0.78), ("normal", 0.35), ("exfil", 0.12)]
# "This anomaly is 78% scan, 35% normal, 12% exfil"
```

**Use case**: Answer "WHAT patterns explain this anomaly?"

## The Forensics Workflow

```
┌─────────────────────────────────────────────────────────────────────┐
│  DETECTION          │  INVESTIGATION       │  EXPLANATION          │
├─────────────────────┼──────────────────────┼───────────────────────┤
│  segment()          │  invert()            │  similarity_profile() │
│  Find WHEN          │  Find WHAT           │  Show WHERE           │
│                     │                      │                       │
│  complexity()       │  project()           │  analogy()            │
│  Measure entropy    │  Classify subspace   │  Generalize patterns  │
│                     │                      │                       │
│                     │  attend()            │  conditional_bind()   │
│                     │  Focus analysis      │  Context-aware encode │
└─────────────────────┴──────────────────────┴───────────────────────┘
```

## Key Insights

### VSA as Forensic Algebra

Traditional ML models are black boxes. You get a score, not an explanation.

VSA provides algebraic operations that preserve interpretability:
- **Decomposable**: `invert()` breaks vectors into known components
- **Localizable**: `similarity_profile()` shows dimension-wise differences
- **Temporal**: `segment()` identifies when behavior changed
- **Transferable**: `analogy()` generalizes across contexts

### Complexity as Attack Signal

Attack traffic often has different complexity than normal:
- **Scans**: Low complexity (repetitive probes)
- **Credential stuffing**: Medium complexity (varied IPs, same action)
- **Exfiltration**: High complexity (mixed with normal to evade)

### Sparse, Gated Representations

`conditional_bind()` creates representations that only activate in specific contexts:
```python
# "This IP is suspicious WHEN it accesses /admin AND gets 403"
suspicious_ip = conditional_bind(
    conditional_bind(ip_vec, path_vec, admin_gate),
    status_vec,
    forbidden_gate
)
```

## Experiments

### 001: Explainable Anomaly Forensics

**Goal**: Complete forensics workflow from detection to explanation.

**Workflow**:
1. Generate traffic with attack phases
2. Use `segment()` to find phase transitions
3. Use `complexity()` to characterize each phase
4. Use `invert()` to attribute anomalies to known patterns
5. Use `similarity_profile()` to localize differences
6. Use `attend()` to focus on attack-relevant dimensions
7. Use `project()` to classify into attack subspace
8. Use `analogy()` to generalize patterns
9. Use `conditional_bind()` for context-aware encoding

**Success criteria**:
- `segment()` detects phase transitions within ±5 events
- `invert()` correctly identifies top-1 pattern for each attack type
- `similarity_profile()` shows higher disagreement for more different attacks

### Future Experiments

- **002**: Real-time segment detection (streaming breakpoints)
- **003**: Attack family clustering via complexity + projection
- **004**: Zero-shot attack detection via analogy transfer
- **005**: Explainability dashboard for SOC analysts

## What We're Building Toward

The ultimate goal: **Explainable, algebraic security analysis**.

When an alert fires, the analyst sees:

```
ALERT: Anomaly detected at 14:32:17

WHEN: Traffic changed at 14:31:45 (segment index 1245)
WHAT: 78% scan_probe + 22% unknown pattern
WHERE: Disagreement in dimensions [method, path_prefix, user_agent]
WHY: Complexity dropped from 0.82 to 0.45 (became repetitive)

SIMILAR TO:
  - 2024-01-15 incident (port scan from 45.33.x.x)
  - 2024-02-22 incident (admin probe attempt)

SUGGESTED ACTION: Rate limit src_ip=45.33.32.156 to 10% of baseline
```

All derived from vector operations. No black box. Pure algebra.

---

*Created: February 2026*
