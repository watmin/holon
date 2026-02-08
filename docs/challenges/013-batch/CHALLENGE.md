# Challenge Batch 013: Vector-Derived Rate Limiting

## Philosophy

**Everything is a vector. State lives in vectors. Rate limits emerge from vector math.**

Batch 012 achieved zero-hardcode detection. But the mitigation signals still used scalar state:
- `severity: 0.85` - a scalar
- `packet_count += 1` - a counter
- `if severity > 0.7 then BLOCK` - discrete rules

This batch asks: **Can we derive rate limits purely from vector operations?**

## Key Achievements

1. **Drift as Rate Signal**: `similarity(prior_accum, recent_accum)` tells us how much traffic shifted
2. **Rate Vector Decoding**: Baseline rate learned as vector, decoded by probing against reference rates
3. **Unified Signals**: Combine batch 012 field detection with concrete PPS rate limits
4. **Actionable Output**: Enforcer receives `{"match": {"src_port": 53}, "rate_pps": 100}`

## The Core Insight

In VSA, similarity is a continuous value from -1 to +1. Instead of:
```python
# Discrete: scalar state → discrete action
if anomaly_score > 0.7:
    action = "BLOCK"
elif anomaly_score > 0.4:
    action = "RATE_LIMIT"
```

What if:
```python
# Continuous: vector similarity → continuous rate
rate_factor = similarity(packet_vec, baseline_vec)
# rate_factor ∈ [0, 1]
# 1.0 = allow at full rate
# 0.5 = allow at 50% rate
# 0.0 = block entirely
```

The rate limit IS the similarity. No thresholds. No discrete categories.

## What "State in Vectors" Means

### Traditional Approach (Batch 012)
```python
# Explicit scalar state
packet_count = 0
anomaly_history = deque(maxlen=30)
baseline_mean = 0.7
baseline_std = 0.1

def process(packet):
    packet_count += 1
    if packet_count > warmup:
        ...
```

### Vector-First Approach (Batch 013)
```python
# State IS the accumulated vector
prior_accum = accumulator(warmup_traffic)
recent_accum = accumulator(recent_traffic, decay=0.98)

def process(packet):
    packet_vec = encode(packet)

    # "How normal is this packet?" - no counters, just similarity
    normalcy = similarity(packet_vec, prior_accum)

    # "Has traffic shifted?" - no explicit tracking, just vector divergence
    drift = 1.0 - similarity(prior_accum, recent_accum)

    # "What rate should this flow at?" - derived from similarity
    rate_factor = normalcy  # or some function of normalcy and drift

    # Update recent state (decay handles "forgetting")
    recent_accum = decay * recent_accum + packet_vec
```

No `packet_count`. No `anomaly_history`. The vectors ARE the history.

## Experiments

### 001: Similarity-Based Rate Factor

**Hypothesis**: `rate_factor = similarity(packet, baseline)` is a meaningful rate limit.

**Test**:
1. Build baseline from warmup traffic
2. For each packet: `rate_factor = similarity(packet_vec, baseline_norm)`
3. Observe distribution of rate_factor for normal vs attack traffic
4. Validate that normal traffic gets high rate_factor, attacks get low

**Key question**: Is the similarity distribution separable enough to be useful as a rate?

### 002: Rate Vectors

**Concept**: Encode rate itself as a vector, use vector operations to derive limits.

```python
# Baseline rate (learned during warmup)
baseline_rate_vec = encode_scalar_log(normal_pps)  # e.g., 100 pps

# Current rate (observed)
current_rate_vec = encode_scalar_log(current_pps)  # e.g., 100,000 pps

# Rate anomaly as vector divergence
rate_divergence = 1.0 - similarity(current_rate_vec, baseline_rate_vec)

# Suggested rate limit = project back to baseline
# If we're 1000x over baseline, rate_factor should be ~0.001
```

**Key question**: Can we derive a rate limit from the vector divergence without knowing the actual PPS values?

### 003: Compositional Rate Limits

**Concept**: Different fields contribute different rate factors.

```python
# Per-field rate factors
protocol_factor = similarity(encode({"protocol": pkt.protocol}), baseline)
port_factor = similarity(encode({"dst_port": pkt.dst_port}), baseline)
flags_factor = similarity(encode({"flags": pkt.flags}), baseline)

# Composite rate factor
# Option A: Minimum (most restrictive field wins)
rate_factor = min(protocol_factor, port_factor, flags_factor)

# Option B: Product (compound effect)
rate_factor = protocol_factor * port_factor * flags_factor

# Option C: Use resonance (agreement across dimensions)
composite = resonance(packet_vec, baseline)
rate_factor = count_nonzero(composite) / dimensions
```

**Key question**: Which composition method gives best separation?

### 004: Difference-Derived Rate Limits

**Concept**: The `difference()` primitive tells us "what's new" - use it to inform rate.

```python
# What's changed from baseline?
delta = difference(baseline, recent_traffic)

# How much does this packet match "what's new"?
novelty = similarity(packet_vec, delta)

# High novelty = new/anomalous = low rate
# Low novelty = matches baseline = high rate
rate_factor = 1.0 - novelty
```

**Key question**: Does `difference()` provide actionable rate information?

### 005: Enforcer Communication via Vectors

**Concept**: Instead of JSON signals, send vectors to the enforcer.

```python
# The rate limit IS a vector
rate_limit_vec = blend(baseline, current_pattern, alpha=rate_factor)

# Enforcer receives vectors, not scalars
enforcer.apply(
    match_pattern=anomaly_signature_vec,  # What to match
    rate_limit=rate_factor,                # How much to allow
    reference=baseline,                    # For comparison
)
```

**Key question**: Can an enforcer meaningfully use vector-based rate limits?

### 006: Recovery Detection via Convergence

**Concept**: Recovery = when recent_accum converges back to prior_accum.

```python
# No explicit "recovery counter" or "state machine"
# Just measure convergence
convergence = similarity(prior_accum, recent_accum)

# High convergence = we've recovered
# Low convergence = still under attack or drifted

# Rate limit relaxation tied to convergence
rate_factor = convergence  # As we recover, rate limit naturally relaxes
```

**Key question**: Does accumulator convergence reliably indicate recovery?

### 007: Field-Specific Rate Vectors

**Concept**: Build rate limit vectors per field, not per packet.

```python
# For each field, track normal vs current distribution
field_rate_limits = {}
for field in ["src_port", "dst_port", "protocol", "flags"]:
    prior = field_accumulators[field]["prior"]
    recent = field_accumulators[field]["recent"]

    # Divergence = how much this field has shifted
    divergence = 1.0 - similarity(prior, recent)

    # Rate factor = inverse of divergence
    field_rate_limits[field] = 1.0 - divergence

# Emit field-level rate guidance
# "src_port traffic: allow at 10% (0.10)"
# "dst_port traffic: allow at 95% (0.95)"
# "protocol traffic: allow at 100% (1.00)"
```

**Key question**: Can field-level rate factors drive meaningful mitigation?

### 008: Unified Rate Signal

**Concept**: Emit a single rate signal that combines all vector-derived factors.

```python
# Combine multiple signals
packet_normalcy = similarity(packet_vec, baseline)
traffic_convergence = similarity(prior_accum, recent_accum)
rate_normalcy = similarity(current_rate_vec, baseline_rate_vec)

# Unified rate factor
# Option A: Minimum (most concerning signal wins)
unified_rate = min(packet_normalcy, traffic_convergence, rate_normalcy)

# Option B: Gated (rate only matters if traffic diverged)
if traffic_convergence < 0.8:  # Traffic has shifted
    unified_rate = packet_normalcy * rate_normalcy
else:
    unified_rate = 1.0  # Normal conditions

# Option C: Pure vector
unified_vec = resonance(packet_vec, prior_accum)
unified_rate = magnitude(unified_vec) / magnitude(prior_accum)
```

## What We're NOT Doing

- ❌ Counting packets
- ❌ Maintaining anomaly histories as lists
- ❌ Discrete thresholds like `if X > 0.7`
- ❌ State machines with explicit transitions
- ❌ Scalar severity scores that aren't derived from vectors

## What We ARE Doing

- ✅ State lives in accumulated vectors
- ✅ Rate limits emerge from similarity/divergence
- ✅ Recovery is detected via convergence
- ✅ Thresholds (if any) are properties of vector relationships
- ✅ Continuous outputs, not discrete categories

## Success Criteria

1. **Zero scalar state**: No counters, no explicit histories
2. **Rate as similarity**: Rate factor directly derived from vector similarity
3. **Meaningful separation**: Normal traffic gets ~1.0, attacks get ~0.0-0.3
4. **Smooth recovery**: Rate limits naturally relax as accumulators converge
5. **Enforcer-friendly**: Output can be consumed by a rate limiting enforcer

## The Key Question

Can we communicate to an enforcer:

> "Rate limit traffic matching THIS pattern to THIS factor of baseline"

...where both "THIS pattern" and "THIS factor" are derived purely from vector operations?

If yes, we've achieved **vector-native rate limiting** - the holy grail of deterministic traffic shaping.

---

*Created: February 2026*
