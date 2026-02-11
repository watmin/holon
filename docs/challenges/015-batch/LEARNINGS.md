# Batch 15 Learnings: Magnitude-Aware Numeric Encoding

## What Worked

### 1. Log Encoding for Magnitude Clustering
The `$log` marker successfully clusters similar magnitudes:
- Within-tier similarity (e.g., 10-100 pps scanners): ~0.67-0.68
- Cross-tier similarity (scanners ↔ attacks): ~0.32
- Clear separation without threshold configuration

### 2. Incident Classification by Behavior Profile
Log encoding enables behavior-based matching:
- DDoS incidents (high network): cluster at 0.99+
- Mining incidents (high CPU): cluster together
- Exfiltration (high disk+network): distinct cluster
- New incidents correctly classified to nearest behavior type

### 3. String vs Log Tradeoff
Clear separation of use cases:
- **String encoding**: Identical values → 1.0, different → ~0.0
- **Log encoding**: Similar magnitudes → 0.95+, 10x difference → 0.90

### 4. Equal Ratios → Equal Similarity Drops
The log encoding preserves proportional relationships:
- 100 → 1000 (10x): similarity 0.94
- 1000 → 10000 (10x): similarity 0.92
- Consistent regardless of absolute scale

## Observations

### Similarity Values with Small Dimensions
Tests with 1000 dimensions show lower absolute similarities than 4096:
- 4096 dims: 100 vs 200 log similarity = 0.98
- 1000 dims: same comparison lower
- For production, higher dimensions recommended for cleaner separation

### Anomaly Scores
With log encoding, anomaly detection naturally captures magnitude:
- Normal traffic (baseline): anomaly ~0.0
- Correlated spikes: anomaly ~0.05
- CPU-only spikes: anomaly ~0.01
- Network floods: anomaly ~0.05

The scores are relatively compressed because log encoding compresses ranges.

### Delta Vectors
Using `difference()` on log-encoded states captures the *type* of change but may not perfectly preserve ratio similarity. Consider encoding the ratio directly if that's the primary signal.

## When to Use Each Encoding

| Field Type | Encoding | Rationale |
|------------|----------|-----------|
| Packet rates (pps) | `$log` | Orders of magnitude matter |
| Byte counts | `$log` | Wide range, proportional growth |
| CPU/Memory % | `$log` | 10% → 50% is same "jump" as 50% → 100% |
| Latency (ms) | `$linear` | Added delay is added delay |
| Temperature | `$linear` | Absolute difference matters |
| Port numbers | string (default) | Exact match needed |
| Status codes | string (default) | Exact match needed |
| User IDs | string (default) | No magnitude relationship |

## Challenge 003: Categorical vs Log Encoding Comparison

### Experiment Results

| Approach | F1 Score | Attribution | Cluster Sep | Magnitude Corr |
|----------|----------|-------------|-------------|----------------|
| Categorical | **0.887** | **0.940** | 0.538 | -0.028 |
| Log ($log) | 0.844 | 0.931 | **0.563** | -0.030 |
| Hybrid | **0.895** | **0.940** | 0.557 | -0.018 |

### Key Finding: Both Approaches Have Strengths

**Categorical wins on F1** because:
- Bucket boundaries align with attack types (SYN=zero, DNS=large, exfil=huge)
- Creates crisp decision boundaries for classification

**Log wins on cluster separation** because:
- Same-type traffic is more cohesive (0.904 vs 0.849 within-cluster)
- No arbitrary bucket boundaries creating false dissimilarity

### Detailed Magnitude Comparison

| Size Comparison | Categorical | Log | Insight |
|-----------------|-------------|-----|---------|
| 100 vs 150 (1.5x) | 1.000 | 0.997 | Both handle well (same bucket) |
| 100 vs 500 (5x, crosses bucket) | 0.760 | 0.989 | **Log preserves similarity** |
| 100 vs 1000 (10x) | 0.760 | 0.986 | Log maintains proportionality |
| 1000 vs 10000 (10x) | 0.777 | 0.976 | Same ratio = similar similarity |
| 100 vs 100000 (1000x) | 0.776 | 0.937 | Log still higher but proportional |

### Recommendations

1. **Use categorical** when attack types naturally align with size buckets
2. **Use $log** when you need:
   - "Find similar magnitude" queries
   - No arbitrary boundaries
   - Proportional magnitude relationships
3. **Consider hybrid** for maximum signal richness (best F1: 0.895)

### Hybrid Encoding Pattern

```python
# Best of both worlds
{
    "size_class": "large",           # Categorical for crisp boundaries
    "payload_size": {"$log": 5000},  # Log for magnitude preservation
}
```

## Future Work

1. **Hybrid encoding**: Some fields might benefit from both log and linear components
2. **Automatic mode detection**: Heuristics to suggest encoding based on field name/values
3. **Scale auto-tuning**: Learn optimal $scale from data distribution
4. **Circular log**: For periodic values that also span magnitudes (e.g., daily traffic cycles)

## Implementation Notes

New markers in `encoder.py`:
- `_log_marker`: `$log`
- `_linear_marker`: `$linear`
- `_scale_marker`: `$scale`

Helper method `_is_numeric_scalar_marker()` detects marker presence.

Encoding delegated to `scalar.py`:
- `encode_scalar_log()` for `$log`
- `encode_positional()` for `$linear`

## Test Coverage

14 new tests added in `test_encoder_coverage.py`:
- Basic encoding validation
- Magnitude similarity properties
- Equal ratios property
- In-record context
- Custom scale parameter
- Edge cases (zero, small values)
- Detection helper
- Error handling
