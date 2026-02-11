# Challenge Batch 15: Magnitude-Aware Numeric Encoding

## Philosophy

**Numbers have meaning beyond their string representation.**

When we encode `{"rate_pps": 1000}` and `{"rate_pps": 1100}`, they should be *similar* - a 10% difference in packet rate is semantically close. But with string encoding, "1000" and "1100" are quasi-orthogonal random vectors with ~0 similarity.

This batch explores the new `$log` and `$linear` markers that enable **magnitude-aware numeric encoding**:
- `{"$log": value}` - Log10 encoding where equal *ratios* have equal similarity drops
- `{"$linear": value}` - Positional encoding where equal *differences* have equal similarity drops

## Key Questions

1. **Can we cluster traffic by magnitude?** Group 10k-50k pps attacks separately from 100k-500k pps attacks without explicit thresholds
2. **Can we find "similar intensity" events?** Query for attacks with comparable severity regardless of exact values
3. **Does log encoding improve anomaly detection?** When rates vary over orders of magnitude, does log encoding give more useful similarity signals?
4. **When should we use linear vs log?** What domains favor each encoding?

## Experiments

### 001: String vs Log Encoding Comparison
**Hypothesis**: Log encoding clusters similar magnitudes while string encoding doesn't.

Show side-by-side that:
- String encoding: 100, 200, 1000, 10000 all have ~0 similarity
- Log encoding: 100↔200 is high, 100↔10000 is lower but proportional

### 002: Traffic Magnitude Clustering
**Hypothesis**: Network traffic can be clustered by rate magnitude without explicit thresholds.

Create traffic samples spanning multiple orders of magnitude and show:
- Low-rate scanners (10-100 pps) cluster together
- Medium traffic (1k-10k pps) clusters together
- High-volume attacks (100k+ pps) cluster together

### 003: Similar Intensity Attack Discovery
**Hypothesis**: Given an attack signature, we can find attacks of "similar intensity" across heterogeneous data.

Demonstrate finding attacks where:
- Rate is within same order of magnitude
- Byte count is proportionally similar
- Without hard-coding thresholds

### 004: Response Time Anomaly Detection
**Hypothesis**: Linear encoding catches latency anomalies better than string or log encoding.

For response times:
- 10ms → 20ms (absolute +10ms) should look similar to 100ms → 110ms
- But log would say 10→20 (2x) is worse than 100→110 (1.1x)
- Linear captures "added delay" regardless of baseline

### 005: Multi-Field Magnitude Correlation
**Hypothesis**: Combining log-encoded fields reveals correlated magnitude patterns.

Detect when multiple metrics spike together:
- CPU% goes from 10% → 80%
- Memory goes from 20% → 90%
- Network goes from 1k → 100k pps

The *proportional* changes correlate even though absolute values differ.

### 003: Categorical vs Log Encoding Comparison (COMPLETED)
**Hypothesis**: $log encoding improves detection over categorical buckets.

**Result**: **Hybrid wins** (F1=0.895) > Categorical (0.887) > Log (0.844)

Key insight: Categorical works well when buckets align with attack types.
Log wins on cluster separation (0.563 vs 0.538).
Hybrid combines both signals for best overall performance.

See `003-categorical-vs-log-encoding.py` for details.

## What We're NOT Doing

- Replacing all numeric encoding with log (would break IDs, ports, status codes)
- Claiming one encoding is universally better
- Hard-coding magnitude thresholds

## What We ARE Doing

- Providing opt-in magnitude awareness via `$log` and `$linear` markers
- Demonstrating when each encoding mode is appropriate
- Showing practical applications in traffic analysis

## Success Criteria

1. Demonstrate clustering by magnitude with zero threshold configuration
2. Show "find similar" queries that work across orders of magnitude
3. Validate that appropriate encoding choice improves detection quality
4. Provide clear guidance on when to use each encoding mode

## Implementation Notes

New markers added to `encoder.py`:
- `{"$log": value}` - Uses `encode_scalar_log()` from scalar.py
- `{"$linear": value}` - Uses `encode_positional()` from scalar.py
- `{"$log": value, "$scale": 500}` - Custom decay rate

See `holon/scalar.py` for the underlying encoding functions.
