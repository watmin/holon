# Challenge 012 Learnings: Zero-Hardcode Significance Detection

## Overview

This batch eliminated hardcoded domain knowledge from anomaly detection. Instead of rules like `if src_port == 53 then dns_reflection`, we detect when ANY field value BECOMES significant.

## The Core Insight

**Traditional approach** (batches 010-011):
```python
# Hardcoded domain knowledge
if pkt.src_port == 53:
    return "dns_reflection"
if pkt.flags == "S":
    return "syn_flood"
```

**Zero-hardcode approach** (batch 012):
```python
# Observe what BECOMES significant
prior_concentration = 1%  # src_port rarely on any single value
recent_concentration = 96%  # src_port now 96% on value 53

alert("src_port CONCENTRATED on NEW value 53 (96%, was 1%)")
# Operator decides what 53 means
```

## Experiment Results

### 001: Per-Field Significance Detection

**Concept**: Maintain per-field accumulators tracking value distributions.

**Key Innovation**: Compare CHANGE from baseline, not absolute values.

```python
# If prior was 80% TCP, then 80% TCP now is NOT anomalous
# But if prior was 80% TCP and now it's 95% TCP with flags=S, that's a change
concentration_delta = recent_concentration - prior_concentration
dominant_value_changed = (dominant_value != prior_dominant_value)
```

**Results by Attack Type**:

| Attack | F1 | Key Alert |
|--------|-----|-----------|
| SYN Flood | 0.800 | `tcp_flags CONCENTRATED on NEW value S (98%, was 42% on PA)` |
| DNS Reflection | 0.706 | `src_port CONCENTRATED on NEW value 53 (96%, was 1%)` |
| NTP Amplification | 0.695 | `src_port CONCENTRATED on NEW value 123 (94%, was 1%)` |
| Port Scan | 0.763 | `src_port CONCENTRATED on NEW value 45000 (100%, was 1%)` |
| **Average** | **0.741** | |

**Key Learnings**:
1. **Divergence is the signal** - `1 - cosine_similarity(prior, recent)` detects shifts
2. **Concentration change matters** - 60% → 96% is significant; 80% → 82% is not
3. **Novel values + divergence = strong signal** - new dominant value with high divergence

### 002: Difference Vector Explainability

**Concept**: Use holon's `difference(prior, recent)` primitive for explanations.

```python
# What changed between prior and recent?
difference = store.difference(prior_norm, recent_norm)

# For each field-value, how much does it contribute to what changed?
importance = cosine_similarity(field_value_vec, difference)

# Low prior similarity = novel
prior_sim = cosine_similarity(field_value_vec, prior_norm)
```

**Results by Attack Type**:

| Attack | F1 | Top Contributors |
|--------|-----|------------------|
| SYN Flood | 0.601 | `dst_port=80: novel (imp=0.23)` |
| DNS Reflection | 0.849 | `src_port=53: novel (imp=0.41), protocol=UDP (imp=0.30)` |
| NTP Amplification | 0.851 | `src_port=123: novel (imp=0.41), protocol=UDP (imp=0.31)` |
| Port Scan | 0.850 | `src_port=45000: novel (imp=0.35)` |
| ICMP Flood | 0.854 | `protocol=ICMP: novel (imp=0.39), icmp_type=8 (imp=0.37)` |
| **Average** | **0.801** | |

**Key Learnings**:
1. **`difference()` captures regime change** - the vector represents "what's new"
2. **Importance = similarity to difference** - high similarity means "part of what changed"
3. **Prior similarity distinguishes novel vs shifted** - low prior_sim = truly new value

### 003: Unified Significance Detector

**Concept**: Combine per-field tracking with difference explanations.

**Detection Signals**:
1. Field divergence > 0.20 (any field's distribution shifted)
2. Traffic divergence > 0.30 (overall pattern shifted)
3. Multiple significant fields (≥2 fields triggered)

**Adaptive Thresholds**:
```python
# Compute baseline variability during warmup
baseline_mean = np.mean(warmup_similarities)
baseline_std = np.std(warmup_similarities)

# Anomaly if packet is > 2 std below baseline mean
sim_threshold = baseline_mean - 2 * baseline_std
```

**Results by Attack Type**:

| Attack | F1 | Precision | Recall | Detection Delay |
|--------|-----|-----------|--------|-----------------|
| SYN Flood | 0.769 | 73.3% | 80.9% | 98 packets |
| DNS Reflection | 0.858 | 76.3% | 98.0% | 10 packets |
| NTP Amplification | 0.860 | 76.7% | 98.0% | 9 packets |
| Port Scan | 0.757 | 72.1% | 79.7% | 99 packets |
| ICMP Flood | 0.863 | 77.1% | 98.0% | 10 packets |
| **Average** | **0.822** | 75.1% | 90.9% | |

**Sample Alerts**:
```
[511] ALERT: src_port has 291 novel values
  → src_port=53: novel (imp=0.13)
  → protocol=UDP: shifted (imp=0.07)

[599] ALERT: src_port has 387 novel values
  → dst_port=80: novel (imp=0.13)
```

## What Zero-Hardcode Achieves

### Detection Without Domain Knowledge

The detector doesn't know:
- Port 53 is DNS
- Port 123 is NTP
- Flags "S" means SYN
- ICMP type 8 is echo request

It only knows:
- **src_port** just became 96% concentrated on value **53**
- This is NOVEL (prior similarity = 1%)
- This is a SIGNIFICANT CHANGE (divergence = 0.40)

### Actionable Operator Alerts

Instead of: `ALERT: DNS reflection attack detected`

We provide: `ALERT: src_port CONCENTRATED on NEW value 53 (96%, was 1% on 54160)`

The operator:
1. Sees the specific field and value causing the anomaly
2. Uses their domain knowledge to interpret (port 53 = DNS)
3. Decides on mitigation (block incoming UDP from port 53)

### Extensibility to Unknown Attacks

If a new attack uses port 12345:
- **Hardcoded approach**: Misses it (no rule for port 12345)
- **Zero-hardcode approach**: Detects `src_port CONCENTRATED on value 12345`

## Holon Primitives That Made This Work

### 1. Structural Encoding
```python
# Role-filler binding preserves structure
vec = encoder.encode_data({"src_port": 53})
# This is different from {"dst_port": 53}
```

### 2. Frequency-Preserving Accumulators
```python
# Accumulator tracks actual frequencies, not just presence
accum = encoder.create_accumulator()
accum = encoder.accumulate(accum, vec)  # Adds to running sum
normalized = encoder.normalize_accumulator(accum)  # Unit vector
```

### 3. Difference Operation
```python
# Captures "what changed" between two states
difference = store.difference(prior_norm, recent_norm)
# High similarity to difference = part of what's new
```

### 4. Cosine Similarity
```python
# Measures distribution divergence
divergence = 1.0 - cosine_similarity(prior_norm, recent_norm)
# High divergence = significant shift
```

## Comparison with Hardcoded Approaches

| Aspect | Hardcoded (010-011) | Zero-Hardcode (012) |
|--------|---------------------|---------------------|
| **Average F1** | 0.85-1.00 | 0.822 |
| **Domain knowledge required** | Yes | No |
| **Novel attack detection** | No | Yes |
| **Explainability** | "DNS reflection" | "src_port → 53 (96%)" |
| **Maintenance** | Add rules per attack | None |
| **False confidence** | Can be spoofed | Less vulnerable |

## Recommendations

### When to Use Zero-Hardcode

1. **Unknown environments**: Don't know what's "normal"
2. **Novel attack detection**: Want to catch unknown attacks
3. **Reduced maintenance**: Don't want to update rules
4. **Operator-in-the-loop**: Humans interpret alerts

### When to Keep Hardcoded Rules

1. **Known attack signatures**: High-confidence detection
2. **Automated response**: Need specific action per attack type
3. **High precision required**: Can't afford false positives
4. **Regulatory compliance**: Must detect specific threats

### Hybrid Approach

```python
# Combine both approaches
def detect(packet):
    # Fast path: known signatures (hardcoded)
    if matches_known_signature(packet):
        return KnownAttackResult(...)

    # Slow path: significance detection (zero-hardcode)
    significance = significance_detector.process(packet)
    if significance.is_anomalous:
        return NovelAnomalyResult(
            explanation=significance.explanation,
            # Operator decides what it means
        )
```

### 004: Attack Lifecycle Validation

**Concept**: Test the full attack lifecycle - learn baseline, detect attack, recover when attack drains, re-detect when attack returns (5 consecutive waves).

**Test Structure**:
```
WARMUP → NORMAL → ATTACK1 → DRAIN1 → ATTACK2 → DRAIN2 → ... → ATTACK5 → FINAL
```

**Key Questions Tested**:
1. Do we detect attacks when they arrive?
2. Do we stop alerting when attacks drain?
3. Can we re-detect attacks when they return (multiple times)?
4. Does detection degrade over repeated attack/drain cycles?

**Mixed Traffic**: Attack waves are 80-92% attack traffic with 8-20% normal traffic mixed in.

**Results by Attack Type (5 waves)**:

| Attack | F1 | Attack Recall | Normal FP | Drain Recovery |
|--------|-----|---------------|-----------|----------------|
| SYN Flood | 0.615 | 45.9% | 2.0% | All ✓ CLEAN |
| DNS Reflection | 0.963 | 96.7% | 2.2% | All ✓ CLEAN |
| NTP Amplification | 0.968 | 97.3% | 1.3% | All ✓ CLEAN |
| ICMP Flood | 0.954 | 96.1% | 2.8% | All ✓ CLEAN |
| **Average** | **0.875** | 84.0% | 2.1% | |

**Key Observations**:

1. **Consistent Re-detection**: All 5 attack waves detected without degradation
2. **Fast Recovery**: Drain phases immediately return to ✓ CLEAN (1-7% FP)
3. **Mixed Traffic Handled**: Normal packets during attack don't cause confusion
4. **No Signature Wear-off**: Attack detection doesn't "tire" across cycles

**SYN Flood Challenge**: Lower recall (45.9%) because individual SYN packets look like normal TCP traffic. The attack IS detected (see alerts in all 5 waves) but not every single attack packet. This is expected without hardcoded `flags == "S"` rules.

**Sample Detection Across Waves** (DNS Reflection):
```
ATTACK-1: 97% recall → drain-1: 4% FP ✓
ATTACK-2: 97% recall → drain-2: 1% FP ✓
ATTACK-3: 97% recall → drain-3: 4% FP ✓
ATTACK-4: 96% recall → drain-4: 3% FP ✓
ATTACK-5: 95% recall → final:   2% FP ✓
```

**Key Learnings**:

1. **Decay rate matters**: Faster decay (0.98) enables quicker recovery after attack drains
2. **Packet-level + pattern-level detection**: Need both for comprehensive coverage
   - Packet novelty catches DNS/NTP/ICMP (individual packets are unusual)
   - Pattern concentration catches SYN flood (individual packets look normal)
3. **History smoothing is essential**: Without anomaly rate filtering, detection is noisy

### 005: Mitigation Signal Emission

**Concept**: Emit structured mitigation signals that downstream consumers can act on. Pure data - no actual firewall rules.

**Signal Structure**:
```json
{
  "timestamp": 510,
  "signal_type": "novelty",
  "action": "block",
  "scope": "exact",
  "severity": 0.85,
  "field": "src_port",
  "value": 53,
  "concentration": 0.96,
  "prior_concentration": 0.01,
  "divergence": 0.40,
  "reason": "Novel value 53 appeared (never seen in baseline)"
}
```

**Signal Types**:
- **CONCENTRATION**: Field became concentrated on specific value
- **NOVELTY**: Novel values appeared in field
- **VOLUMETRIC**: Overall traffic pattern shifted

**Action Types**:
- **BLOCK**: High severity (>0.7) + high concentration (>0.8)
- **RATE_LIMIT**: Medium severity (>0.4)
- **MONITOR**: Low severity (watch only)
- **CLEAR**: Remove previous mitigation when traffic normalizes

**Results by Attack Type**:

| Attack | Signals | BLOCKs | CLEARs | Key Signal |
|--------|---------|--------|--------|------------|
| DNS Reflection | 43 | 23 | 4 | `BLOCK src_port=53 (novelty)` |
| SYN Flood | 33 | 2 | 8 | `RATE_LIMIT dst_port=80 (concentration)` |

**Key Design Decisions**:

1. **Signal consolidation**: Deduplicate signals (cooldown = 50 packets)
2. **Ephemeral port filtering**: Skip src_port >= 49152 (always "novel", not actionable)
3. **Severity calculation**: Combines concentration, change, and divergence
4. **CLEAR emission**: When field divergence drops below 15%

**Sample Signal Flow (DNS Reflection)**:
```
[510] RATE_LIMIT src_port=53 (sev=0.66, novelty)    # Attack starting
[590] BLOCK      GLOBAL (sev=0.67, volumetric)       # Attack intensifying
[610] BLOCK      src_port=53 (sev=1.00, novelty)     # High severity
[792] CLEAR      protocol=UDP (divergence=15%)        # Attack draining
[822] CLEAR      dst_port=443 (divergence=1%)         # Back to normal
```

**Consumer Responsibility**:
- Whether to act on the signal
- How to implement (iptables, ACL, null route, API call)
- How long to maintain mitigation
- Whether to escalate or de-escalate

### 006: Volumetric Attack with Synthetic Timing

**Concept**: Detect volumetric attacks using rate ratio as the primary signal, not pattern divergence.

**Timeline Simulation**:
```
calm-1 (5 min)  →  ATTACK-1 (30 sec)  →  calm-2 (2.5 min)  →  ATTACK-2 (1 min)  →  calm-3 (3 min)
   100 pps            100,000 pps            100 pps            50,000 pps           100 pps
                         1000x                                     500x
```

**Key Insight**: Rate ratio is the defining characteristic of volumetric attacks.
- During attack: rate is 500-1000x normal
- After attack: rate immediately returns to normal
- Pattern divergence takes time to recover (residue, not attack)

**Detection Logic**:
```python
is_volumetric = (
    rate_ratio > 50 or                    # 50x rate = definite attack
    (rate_ratio > 5 and divergence > 0.25)  # 5x + pattern shift
)
# Don't trigger on divergence alone - that's residue, not active attack
```

**Results**:

| Attack Type | Attack Recall | Normal FP | Calm Recovery |
|-------------|---------------|-----------|---------------|
| DNS Reflection | 100% | 3% | ✓ CLEAN |
| SYN Flood | 100% | 3% | ✓ CLEAN |
| UDP Flood | 100% | 3% | ✓ CLEAN |

**Key Learnings**:

1. **Rate is the primary signal**: Volumetric attacks BY DEFINITION have high rate
2. **Pattern divergence is secondary**: Useful during attack, but causes FPs during recovery
3. **Divergence residue**: After flood, pattern takes time to recover - don't alert on this
4. **Fast recovery**: Shorter smoothing window (20 packets) enables quick return to normal

**Unknown Attack Detection**:
```
What we DON'T hardcode:        What we DO observe:
- Port 53 = DNS                - "Rate is 1000x normal"
- Port 123 = NTP               - "Divergence is 44%"
- Protocol semantics           - "Rate dropped, back to normal"
```

This enables detection of novel/zero-day volumetric attacks - any traffic spike triggers detection.

### 007: Vectorized Rate Detection

**Concept**: Eliminate magic numbers like `rate > 50x` by encoding rate itself as a vector.

**Rate Encoding**:
```python
def encode_rate(pps):
    return encoder.encode_data({
        "rate_magnitude": int(log10(pps)),      # 2 for 100pps, 5 for 100000pps
        "rate_fraction": int((log10(pps) % 1) * 10),
        "rate_band": get_band(pps),              # "moderate", "extreme", etc.
    })
```

**Detection Logic**:
```python
# Learn baseline
baseline_rate = accumulate(warmup_rate_vectors)
baseline_sim_mean, baseline_sim_std = compute_stats(warmup_similarities)

# Detect
rate_vec = encode_rate(current_pps)
similarity = cosine_sim(rate_vec, baseline_rate)
threshold = baseline_sim_mean - 2.5 * baseline_sim_std  # LEARNED, not hardcoded

if similarity < threshold:
    is_volumetric = True
```

**Results**:

| Attack Type | Attack Recall | Normal FP | Rate Anomalies |
|-------------|---------------|-----------|----------------|
| DNS Reflection | 100% | 8% | 100% of attack pkts |
| SYN Flood | 100% | 7% | 100% of attack pkts |
| UDP Flood | 100% | 8% | 100% of attack pkts |

**Key Insight**: During warmup at 100 pps, rate vectors are consistent:
```
Learned: sim_mean=1.000, sim_std=0.000
```

During attack at 100,000 pps, rate vectors are DISSIMILAR to baseline.

**What Changed from 006 to 007**:

| Aspect | 006 (Hardcoded) | 007 (Vectorized) |
|--------|-----------------|------------------|
| Detection | `rate_ratio > 50` | `similarity < learned_threshold` |
| Thresholds | Magic numbers | Learned from baseline |
| Rate representation | Scalar | Vector |
| Adaptability | Fixed | Adapts to environment |

**Trade-off**: Slightly higher FP (8% vs 3%) because the vectorized approach is more sensitive. But it eliminates ALL hardcoded thresholds.

### 008: Pure Vector Rate Detection (No Magic Numbers)

**Concept**: Fix the remaining magic numbers in 007 by using continuous positional encoding.

**Issues in 007**:
```python
# 007 still had hardcoded bands
def _get_rate_band(self, pps: float) -> str:
    if pps < 10: return "trickle"
    elif pps < 100: return "low"     # MAGIC NUMBER
    elif pps < 1000: return "moderate"
    ...

# 007 reimplemented cosine similarity
def cosine_similarity(vec1, vec2):  # Should use store.similarity()
```

**008 Fixes**:
```python
# Use Holon's built-in similarity
sim = store.similarity(vec1, vec2, metric="cosine")

# Use positional encoding for continuous rate
log_rate = log10(pps)           # 100 pps → 2.0, 100000 pps → 5.0
rate_vec = positional_encode(log_rate)  # Smooth, continuous

# Similar rates → similar vectors (NO discretization)
```

**Positional Encoding for Rate**:
```python
def _positional_encode(self, value: float) -> np.ndarray:
    indices = np.arange(self.dimensions)
    freqs = 1 / (self.scale ** (indices / self.dimensions))

    values = np.where(
        indices % 2 == 0,
        np.sin(value * freqs),
        np.cos(value * freqs),
    )
    return np.sign(values).astype(np.int8)
```

**Rate Similarity Matrix** (demonstrating continuous nature):
```
Rate           10       100      1000     10000    100000
--------------------------------------------------------------
10           1.00      0.96      0.91      0.83      0.77
100          0.96      1.00      0.94      0.86      0.79
1000         0.91      0.94      1.00      0.92      0.85
10000        0.83      0.86      0.92      1.00      0.93
100000       0.77      0.79      0.85      0.93      1.00
```

Key: Similar rates have similar vectors, dissimilar rates have dissimilar vectors. Smooth gradient, not stepped categories.

**Results**:

| Attack Type | Attack Recall | Normal FP | Rate Anomalies |
|-------------|---------------|-----------|----------------|
| DNS Reflection | 100% | 8% | 100% |
| SYN Flood | 100% | 7% | 100% |
| UDP Flood | 100% | 8% | 100% |

**What's Different from 007 to 008**:

| Aspect | 007 | 008 |
|--------|-----|-----|
| Rate encoding | Structured `{magnitude, band}` | Positional encoding of log(rate) |
| Rate bands | Hardcoded ("low", "extreme") | None (continuous) |
| Similarity | Custom `cosine_similarity()` | `store.similarity(..., metric="cosine")` |
| Threshold | `mean - 2.5 * std` (same) | `mean - 2.5 * std` (same) |

**The Only "Numbers" in 008**:
1. `2.5` std below mean - **statistical** threshold (not domain-specific)
2. `0.4` smoothing threshold - **sensitivity** parameter (tunable)
3. `DECAY = 0.98` - **forgetting rate** (tunable)

These are **statistical/tuning** parameters, not **domain knowledge**.

## Future Directions

1. **Multi-timescale tracking**: Fast + slow accumulators for trend detection
2. **Automatic signature learning**: When anomaly ends, extract difference as signature
3. **Field correlation**: Detect when COMBINATIONS become significant
4. **Rate trending**: Detect gradual ramp-up attacks (slow rate increase)
5. **SYN-specific improvement**: Track flag+port combinations for pattern-based attacks

## Files Created

| File | Lines | Purpose |
|------|-------|---------|
| `001-significance-detection.py` | ~450 | Per-field tracking with concentration/divergence |
| `002-difference-explainability.py` | ~350 | Vector difference for explanations |
| `003-unified-significance.py` | ~400 | Combined approach with adaptive thresholds |
| `004-attack-lifecycle.py` | ~450 | Multi-wave attack/drain cycle validation |
| `005-mitigation-signals.py` | ~550 | Structured signal emission for mitigation |
| `006-volumetric-timing.py` | ~450 | Rate-based volumetric detection with timing |
| `007-vectorized-rate.py` | ~600 | Rate encoded as vector (still has band magic) |
| `008-pure-vector-rate.py` | ~400 | Positional encoding for rate (zero magic) |
| `009-accuracy-improvements.py` | ~550 | Ensemble voting (residual, multiscale, etc.) |
| `014-fast-recovery.py` | ~350 | Best: Gated + fast recovery (4% FP) |

## NEW: Holon Continuous Encoding API

This batch led to exposing continuous encoding as a **core Holon primitive**:

```python
# Encode rate on log scale - equal ratios = equal similarity
rate_vec = store.encode_scalar_log(pps)

# Similarity matrix (log-scale):
#   Rate        100    1000   10000  100000
#   100        1.00    0.94    0.86    0.79
#   1000       0.94    1.00    0.92    0.85
#   10000      0.86    0.92    1.00    0.93
#   100000     0.79    0.85    0.93    1.00
```

**Key property**: 100→1000 similarity ≈ 1000→10000 similarity (equal ratios = equal similarity)

**API Methods**:
| Method | Purpose |
|--------|---------|
| `store.encode_scalar(value, mode="linear")` | Linear position encoding |
| `store.encode_scalar(value, mode="circular", period=360)` | Wrap-around (angles, hours) |
| `store.encode_scalar_log(value)` | Log scale for multiplicative quantities |

**Unit test coverage**: 19 tests in `tests/test_continuous_encoding.py`

## Final Demo: Zero-Hardcode Detection

The batch culminates in `DEMO-zero-hardcode-detection.py`:

```bash
./scripts/run_with_venv.sh python scripts/challenges/012-batch/DEMO-zero-hardcode-detection.py
```

**Results**:
```
ATTACK RECALL                              100%
FALSE POSITIVE RATE                          4%
```

**Techniques showcased**:
1. `store.encode_scalar_log(pps)` - Log-scale rate encoding (NEW API!)
2. `encoder.encode_data(packet)` - Structured packet encoding
3. Frozen z-score baselines - Learned thresholds
4. Gated detection - Pattern requires rate confirmation
5. Fast recovery - Quick return to normal after attacks

**What's NOT hardcoded**:
- No port meanings (53=DNS, 123=NTP)
- No protocol semantics (TCP vs UDP)
- No rate thresholds (if rate > 1000x)
- No attack signatures

The detector learned what's "normal" and detects deviations. All interpretation is left to the human operator.

## Summary

Batch 012 demonstrated that **zero-hardcode anomaly detection is viable** with holon's primitives:

| Metric | Best Result |
|--------|-------------|
| Attack Recall | **100%** |
| False Positive Rate | **4%** |
| Domain knowledge required | **None** |
| Explainability | **Field + value + z-score** |
| Lifecycle support | **5+ waves, all recovered** |

**Key insights**:
1. **Track what BECOMES significant, not what IS significant**
2. **Faster decay enables faster recovery after attacks drain**
3. **Gated detection** - Pattern anomalies should require rate confirmation
4. **Continuous encoding** (new Holon API) eliminates discretization magic

**Holon library contribution**: `encode_scalar()` and `encode_scalar_log()` methods now exposed as public API for continuous value encoding.

---

*Challenge 012 completed: February 2026*
