# Challenge Batch 012: Zero-Hardcode Significance Detection

## Overview

Building on batches 010-011's successful DDoS detection, this batch tackles a fundamental limitation: **eliminating hardcoded domain knowledge**.

Previous batches relied on rules like:
- `if src_port == 53 then dns_reflection`
- `if flags == "S" then syn_flood`
- `port < 1024 is wellknown`

This batch asks: **Can we detect anomalies by observing what BECOMES significant, without knowing in advance what to look for?**

## The Problem

Hardcoded indicators have drawbacks:
1. **Maintenance burden**: New attacks require new rules
2. **Domain expertise required**: Must know that port 53 = DNS
3. **Blind spots**: Unknown attacks don't match any rules
4. **False confidence**: Matched rules feel "certain" but may be spoofed

## The Approach

Instead of encoding domain knowledge, we encode **field-value pairs as structured data** and track their **distribution changes** over time:

```python
# NOT this (string atoms)
"src_port:53"

# THIS (structured data with role-filler binding)
{"src_port": 53}
```

Key insight: The value `53` in `{"src_port": 53}` is bound to the `src_port` role, making it distinct from `{"dst_port": 53}`.

## Detection Modes

### 1. CONCENTRATION
A field that was diverse just became concentrated:
- **Normal**: src_port varies across ephemeral range (49152-65535)
- **Attack**: src_port is now 96% concentrated on value 53

### 2. DIVERSIFICATION
A field that was stable just became diverse:
- **Normal**: dst_port is mostly 443/80
- **Attack**: dst_port scanning across many ports

### 3. NOVELTY
Field values we've never seen before appeared:
- **Prior**: src_port always ephemeral (>= 1024)
- **Attack**: src_port is now 53, 123, 1900 (never seen before)

### 4. VOLUMETRIC
Traffic pattern drowned out prior knowledge:
- **Detected via**: `similarity(prior, recent)` drops sharply
- **Signal**: Something is flooding the network

## Experiments

### 001: Per-Field Significance Detection
Track each monitored field with its own accumulator:
- `prior_accum`: Frozen baseline from warmup
- `recent_accum`: Decaying recent window

Compare distributions to detect when any field's distribution SHIFTS from baseline.

### 002: Difference Vector Explainability
Use holon's `difference(prior, recent)` primitive to explain anomalies:
- Compute what changed between baseline and current
- For each field-value, measure similarity to difference vector
- High similarity = this field-value is part of what's new

### 003: Unified Significance Detector
Combine per-field tracking with difference explanations:
- Adaptive thresholds based on baseline variability
- Multiple detection signals (field divergence, traffic divergence, novelty)
- Actionable explanations

### 004: Attack Lifecycle Validation
Test the full attack lifecycle with 5 consecutive waves:
- Learn baseline → Detect attack → Recover when attack drains → Re-detect when attack returns
- 80-92% attack traffic mixed with 8-20% normal traffic
- Validates recovery and re-detection across multiple cycles

### 005: Mitigation Signal Emission
Emit structured mitigation signals for downstream consumers:
- Signal types: CONCENTRATION, NOVELTY, VOLUMETRIC
- Actions: BLOCK, RATE_LIMIT, MONITOR, CLEAR
- Consolidated signals (deduplication, ephemeral port filtering)
- Pure data output - consumer decides implementation

### 006: Volumetric Attack with Synthetic Timing
Simulate realistic volumetric attacks with time-based rate differentials:
- Calm periods: 100 packets/second
- Attack periods: 50,000-100,000 packets/second (500-1000x)
- Rate ratio as primary detection signal
- Pattern divergence as secondary signal

### 007: Vectorized Rate Detection
Eliminate magic numbers by encoding rate as vectors:
- Rate encoded as: `{rate_magnitude: 2, rate_band: "moderate"}`
- Baseline learned during warmup
- Detection via vector similarity, not `rate > 50x`
- Thresholds computed from baseline variance

### 008: Pure Vector Rate Detection (No Magic)
Fixes remaining issues from 007:
- Uses `store.similarity()` instead of custom cosine function
- Uses positional encoding for continuous rate (no discretization)
- NO hardcoded rate bands ("low", "moderate", "extreme")
- Thresholds are purely statistical (mean - 2.5*std)

## Monitored Fields

No domain knowledge about what values mean:

| Field | What We Track | Not Hardcoded |
|-------|---------------|---------------|
| `src_port` | Value distribution | Port 53 = DNS |
| `dst_port` | Value distribution | Port 443 = HTTPS |
| `protocol` | Value distribution | TCP vs UDP meaning |
| `tcp_flags` | Value distribution | SYN = connection start |
| `icmp_type` | Value distribution | Type 8 = echo request |
| `payload_size` | Size buckets | What sizes are "normal" |

## Holon Primitives Used

| Primitive | Purpose |
|-----------|---------|
| `create_accumulator()` | Initialize frequency-preserving tracker |
| `accumulate()` | Add observations without losing frequency |
| `normalize_accumulator()` | Get unit vector for similarity queries |
| `difference(before, after)` | Compute what changed |
| `encode_data()` | Structural encoding with role-filler binding |
| `store.similarity()` | Measure distribution divergence (multiple metrics) |
| `encode_scalar(value, mode)` | **NEW** Linear/circular continuous encoding |
| `encode_scalar_log(value)` | **NEW** Log-scale continuous encoding (rates, sizes) |

## NEW Holon API: Continuous Scalar Encoding

This batch led to exposing continuous encoding as a core Holon primitive:

```python
# Encode rate on log scale - equal ratios = equal similarity
rate_vec = store.encode_scalar_log(pps)

# 100→1000 similarity ≈ 1000→10000 similarity
store.similarity(store.encode_scalar_log(100), store.encode_scalar_log(1000))  # ~0.94
store.similarity(store.encode_scalar_log(1000), store.encode_scalar_log(10000)) # ~0.92

# Also works directly on encoder
encoder.encode_scalar(value, mode="linear")  # Linear position
encoder.encode_scalar(value, mode="circular", period=360.0)  # Wrap-around (angles, hours)
encoder.encode_scalar_log(value)  # Log scale for multiplicative quantities
```

**Why this matters**: Eliminates hardcoded discretization like `{rate_band: "low"}` - continuous values get smooth similarity gradients.

## Implementation Files

```
scripts/challenges/012-batch/
├── 001-significance-detection.py     # Per-field tracking
├── 002-difference-explainability.py  # Vector difference explanations
├── 003-unified-significance.py       # Combined approach
├── 004-attack-lifecycle.py           # Multi-wave lifecycle validation
├── 005-mitigation-signals.py         # Structured signal emission
├── 006-volumetric-timing.py          # Rate-based volumetric detection
├── 007-vectorized-rate.py            # Rate as vector (still has magic bands)
├── 008-pure-vector-rate.py           # Pure positional encoding (no magic)
├── 009-015: Accuracy experiments     # See LEARNINGS.md for details
└── DEMO-zero-hardcode-detection.py   # Final demo showcasing best techniques
```

## Success Criteria

1. **Zero hardcoding**: No domain knowledge about ports, flags, or protocols
2. **Detection**: Identify all attack types from 010-011 batches
3. **Explainability**: Output like "src_port CONCENTRATED on value 53"
4. **F1 > 0.75**: Competitive with hardcoded approaches
5. **Lifecycle**: Recover after attacks drain, re-detect when they return
6. **Mitigation signals**: Actionable data for downstream consumers
7. **Volumetric**: Detect rate spikes (100-1000x normal)

## Results Summary

| Experiment | Metric | Key Finding |
|------------|--------|-------------|
| 001: Per-Field Tracking | F1=0.741 | Concentration detection works |
| 002: Difference Explainability | F1=0.801 | `difference()` primitive powerful |
| 003: Unified Approach | F1=0.822 | Combined approach best |
| 004: Attack Lifecycle | F1=0.875 | 5-wave detection, clean recovery |
| 005: Mitigation Signals | F1=0.785 | Actionable BLOCK/RATE_LIMIT/CLEAR |
| 006: Volumetric Timing | 100% Recall, 3% FP | Rate ratio is key signal |
| 007: Vectorized Rate | 100% Recall, 8% FP | Rate as vector (still has band magic) |
| 008: Pure Vector Rate | 100% Recall, 8% FP | Positional encoding, zero magic |
| 009-013: Accuracy Experiments | Various | Ensemble voting, frozen baselines |
| **014: Fast Recovery** | **100% Recall, 4% FP** | Gated detection + fast recovery |
| **DEMO** | **100% Recall, 4% FP** | Best composition of techniques |

See `LEARNINGS.md` for detailed analysis of experiments 009-015.

## Final Demo

The batch culminates in `DEMO-zero-hardcode-detection.py` which showcases:

1. **New `store.encode_scalar_log()` API** - Rate encoding with equal-ratio similarity
2. **Frozen z-score baselines** - Learned thresholds that don't adapt to attacks
3. **Gated detection** - Pattern anomalies require rate confirmation
4. **Fast recovery** - Quick return to normal after attacks end
5. **Zero domain knowledge** - No port meanings, protocol semantics, or rate thresholds

Run the demo:
```bash
./scripts/run_with_venv.sh python scripts/challenges/012-batch/DEMO-zero-hardcode-detection.py
```

---

*Created: February 2026*
*Completed: February 2026*
