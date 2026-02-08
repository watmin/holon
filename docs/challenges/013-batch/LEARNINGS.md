# Challenge 013 Learnings: Vector-Derived Rate Limiting

## Overview

This batch built on batch 012's zero-hardcode anomaly detection to emit **actionable rate limits** derived from vector operations.

**Key Question**: Can we tell an enforcer "rate limit src_port=53 to 100 pps" where BOTH the pattern and the rate come from vectors?

**Answer**: Yes!

## The Journey

### Experiment 001: Similarity as Rate Factor

**Hypothesis**: `rate_factor = similarity(packet, baseline)` is the rate limit.

**Result**: Partial success.
- DNS reflection, ICMP flood: Good separation (attacks at 0.09-0.12)
- SYN flood: FAILED - attacks got HIGHER similarity (0.42) than normal (0.39)

**Problem**: Individual packet similarity doesn't work because:
- Normal traffic is diverse (each packet ≠ aggregate baseline)
- Attack traffic is homogeneous (consistent pattern = higher similarity to accumulated baseline)

### Experiment 002: Drift as Rate Factor

**Hypothesis**: `rate_factor = similarity(prior_accum, recent_accum)`

**Result**: SUCCESS!

| Attack | Normal | Attack Min | Recovery |
|--------|--------|------------|----------|
| SYN flood | 0.99 | 0.51 | 0.98 |
| DNS reflection | 0.99 | 0.19 | 0.96 |
| ICMP flood | 0.99 | 0.12 | 0.96 |

**Key insight**: The drift between accumulators IS the rate signal.
- High similarity = normal traffic = allow
- Low similarity = traffic shifted = throttle
- Recovery is automatic as accumulators converge

### Experiment 003: Difference as Pattern

**Hypothesis**: `difference(recent, prior)` gives us the anomaly signature.

**Result**: Not as useful as expected.
- Attack packets got HIGHER rates than normal packets during attack
- The difference vector captures "what's new" but doesn't separate individual packets well

**Learning**: Difference is useful for explanation, not for per-packet rate limiting.

### Experiment 004: Per-Field Rate Vectors

**Concept**: Track rate per field, not globally.

**Result**: Global rate encoding gave same rate_factor for all fields (0.865 for 100x rate increase).

**Problem**: We're comparing global rate, not the rate of specific anomalous patterns.

### Experiment 005: Accumulator Magnitude

**Hypothesis**: `magnitude_ratio = ||recent|| / ||prior||` reflects rate.

**Result**: Magnitude dropped below 1.0 during attack (opposite of expected).

**Problem**: With decay, magnitude doesn't directly reflect rate.

### Experiment 006: Rate Vector Decoding (BREAKTHROUGH)

**Concept**: Use `encode_scalar_log(pps)` during warmup, then DECODE by comparing to reference rates.

**Method**:
1. Create reference rate vectors: `encode_scalar_log(10)`, `encode_scalar_log(100)`, etc.
2. Learn baseline rate as accumulated rate vectors
3. Decode by finding highest-similarity reference

**Result**: PERFECT decoding!
- 100 pps → decoded as 100 pps
- 10000 pps → decoded as 10000 pps

**Key insight**: The rate vector IS a lookup key. We query it against known rates to decode.

### Experiment 007: Unified Rate Limit Signal

**Concept**: Combine batch 012 field detection with rate decoding.

**Result**: Complete, actionable signals:

```json
{
  "match": {"src_port": 53},
  "action": "rate_limit",
  "rate_pps": 100,
  "reason": "src_port=53 anomalous (baseline: 100 pps)"
}
```

## What Works

### 1. Drift for Global Rate Signal
```python
rate_factor = similarity(prior_accum, recent_accum)
# 1.0 = normal, 0.2 = attack, recovers automatically
```

### 2. Rate Vector Decoding for Concrete PPS
```python
# During warmup
baseline_rate_vec = accumulate(encode_scalar_log(observed_pps))

# To decode
for ref_pps in [10, 100, 1000, 10000]:
    sim = similarity(baseline_rate_vec, encode_scalar_log(ref_pps))
# Highest sim = decoded rate
```

### 3. Per-Field Pattern Tracking (from batch 012)
```python
# Pattern divergence tells us WHICH field is anomalous
divergence = 1 - similarity(prior_pattern, recent_pattern)
if divergence > 0.15:
    # This field shifted
```

### 4. Combined Signal
```python
signal = {
    "field": field_name,              # From pattern tracking
    "value": dominant_value,          # From concentration
    "is_novel": value not in baseline,
    "baseline_rate_pps": decode(prior_rate_vec),  # From rate decoding
    "enforce_rate_pps": baseline_rate_pps,        # = baseline
}
```

## What Doesn't Work

### 1. Individual Packet Similarity
- Normal traffic is diverse → low similarity to aggregate
- Attack traffic is homogeneous → high similarity to aggregate
- Separation is inverted for some attacks

### 2. Accumulator Magnitude as Rate
- With decay, magnitude doesn't directly track rate
- Magnitude can decrease during high-rate attacks

### 3. Per-Packet Rate Factors
- Can't reliably distinguish attack vs normal packets during attack
- Global drift works; per-packet similarity doesn't

## Final Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DETECTOR                                  │
│                                                                 │
│  For each monitored field:                                      │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Pattern Accumulator                                     │   │
│  │  - prior_pattern (frozen baseline)                       │   │
│  │  - recent_pattern (decaying)                             │   │
│  │  - divergence = 1 - sim(prior, recent)                   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Rate Accumulator                                        │   │
│  │  - prior_rate = accumulate(encode_scalar_log(pps))       │   │
│  │  - baseline_pps = decode(prior_rate)                     │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Concentration Tracker                                   │   │
│  │  - recent_counts: {value → count}                        │   │
│  │  - dominant_value = max(counts)                          │   │
│  │  - is_novel = value not in baseline                      │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     RATE LIMIT SIGNAL                            │
│                                                                 │
│  {                                                              │
│    "match": {"src_port": 53},                                   │
│    "action": "rate_limit",                                      │
│    "rate_pps": 100,                                             │
│    "reason": "src_port=53 (novel) at 96% concentration"         │
│  }                                                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        ENFORCER                                  │
│                                                                 │
│  For each rule:                                                 │
│    - Match packets by field=value                               │
│    - Apply rate limit (e.g., token bucket at rate_pps)          │
│    - Allow up to baseline rate, drop excess                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Key Insights

### 1. State Lives in Vectors
- The accumulators ARE the state
- No explicit counters needed (except for concentration)
- Recovery happens naturally as accumulators converge

### 2. Rates Can Be Encoded AND Decoded
- `encode_scalar_log(pps)` creates a rate vector
- Accumulating rate vectors creates a "rate fingerprint"
- Decoding by comparing to reference rates

### 3. Separation of Concerns
- Pattern tracking: WHAT is anomalous
- Rate tracking: HOW MUCH to limit
- Both from vectors, combined in signal

### 4. No Magic Thresholds
- Divergence threshold (0.15) is sensitivity tuning
- Reference rates are the log10 scale
- No hardcoded attack knowledge

## Files Created

| File | Purpose |
|------|---------|
| `001-similarity-as-rate.py` | Individual packet similarity (partial success) |
| `002-drift-as-rate.py` | Accumulator drift (SUCCESS) |
| `003-difference-as-pattern.py` | Difference vector for pattern matching |
| `004-per-field-rate-vectors.py` | Per-field rate tracking |
| `005-accumulator-magnitude-as-rate.py` | Magnitude-based rate (failed) |
| `006-rate-vector-decode.py` | Rate decoding (BREAKTHROUGH) |
| `007-unified-rate-limit-signal.py` | Complete unified signal |

## Sample Output

**DNS Reflection Attack:**
```
src_port=53
  Status:           NOVEL
  Pattern shift:    103%
  Baseline rate:    100 pps
  Current rate:     10000 pps
  → ENFORCE:        100 pps
```

**SYN Flood Attack:**
```
flags=S
  Status:           92% concentration
  Pattern shift:    77%
  Baseline rate:    100 pps
  Current rate:     10000 pps
  → ENFORCE:        100 pps
```

**ICMP Flood Attack:**
```
protocol=ICMP
  Status:           88% concentration
  Pattern shift:    85%
  Baseline rate:    100 pps
  Current rate:     10000 pps
  → ENFORCE:        100 pps
```

### Experiment 008: Learned Rate References (ZERO MAGIC NUMBERS)

**Problem**: Hardcoded `REFERENCE_RATES = [1, 5, 10, 50, 100, ...]` is magic.

Different apps have different baselines:
- Low-traffic service: ~10 pps
- High-traffic API: ~50,000 pps

**Solution**: Learn reference rates from observation during warmup.

**Method**:
1. Observe actual PPS values during warmup
2. Compute percentiles: p10, p25, p50, p75, p90
3. Project attack scales: 2x, 5x, 10x, 50x, 100x median
4. Build reference vectors from learned rates

**Result**: Works for ANY baseline!

| Baseline | Learned References (sample) |
|----------|---------------------------|
| 10 pps | 1, 5, 9, 10, 20, 50, 100, 500, 1000 |
| 500 pps | 50, 250, 500, 1000, 2500, 5000, 25000, 50000 |
| 50,000 pps | 5000, 25000, 50000, 100000, 250000, 500000, 2.5M, 5M |

**Key insight**: The same approach we used for ports/protocols/flags works for rates.
- Observe during warmup
- Build references from observations
- No hardcoded knowledge needed

### Experiment 009: Field Scrubber Architecture

**Question**: Can a field scrubber operate with ONLY shipped vectors?

**Answer**: Yes!

**Shipped from Central**:
- `prior_pattern_norm` (vector) - baseline fingerprint
- `prior_rate_norm` (vector) - baseline rate fingerprint
- `decay` (scalar constant)

**Local State at Scrubber**:
- `recent_pattern` (vector) - decaying accumulator
- Encoder (stateless)

**No scalar counters for detection**. The pattern drift itself signals anomalies.

### Experiment 010: Extreme Rate Handling

**Scenario**: 300 pps baseline → 1 billion pps attack → recovery

**Results**:
- Log scale encodes any rate (0.1 to 1 trillion+)
- Frozen baseline protects against corruption
- Recovery in <1 second at 300 pps after attack ends
- Attack signal washes out: 0.98^230 ≈ 0.01 (99% gone)

### Experiment 011: Binary Search Rate Decode (NO REFERENCE VECTORS)

**Problem**: Storing N reference vectors is expensive.

**Solution**: Binary search on log10 scale.

**Algorithm**:
1. Target: single `baseline_rate_vec`
2. Binary search on [0, 12] (covering 1 to 1 trillion pps)
3. At each step, encode probe and compare similarity
4. Converge in O(log N) iterations

**Results**:
- 5,000 pps baseline → decoded as 4,980 pps (0.4% error)
- 11 iterations, 13 encode operations
- NO stored reference vectors
- Memory saved: 245KB → 0 bytes

## Files Created

| File | Purpose |
|------|---------|
| `001-similarity-as-rate.py` | Individual packet similarity (partial success) |
| `002-drift-as-rate.py` | Accumulator drift (SUCCESS) |
| `003-difference-as-pattern.py` | Difference vector for pattern matching |
| `004-per-field-rate-vectors.py` | Per-field rate tracking |
| `005-accumulator-magnitude-as-rate.py` | Magnitude-based rate (failed) |
| `006-rate-vector-decode.py` | Rate decoding (BREAKTHROUGH) |
| `007-unified-rate-limit-signal.py` | Complete unified signal |
| `008-learned-rate-references.py` | Zero magic numbers (SUCCESS) |
| `009-field-scrubber.py` | Field scrubber architecture |
| `010-extreme-rate-test.py` | Extreme rate handling |
| `011-binary-search-rate-decode.py` | O(log N) rate decode |
| `DEMO-rate-limit-mitigation.py` | Complete demo |

## Future Work

1. **Per-Field-Value Rate Tracking**: Track rate for specific values, not just global
2. **Multi-Field Matching**: Emit rules matching multiple fields (src_port=53 AND protocol=UDP)
3. **Graceful Degradation**: Emit partial throttle during ramp-up, not immediate full throttle
4. **Enforcer Feedback**: Learn from enforcer effectiveness to tune signals
5. **Continuous Rate Interpolation**: Interpolate between reference rates for finer granularity

---

*Challenge 013 completed: February 2026*
