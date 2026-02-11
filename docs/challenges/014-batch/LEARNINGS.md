# Learnings: Batch 014 - Extended Primitives

## Implementation Notes

### Python Module Reorganization

Refactored Python codebase to mirror Rust structure:

| Module | Lines | Purpose |
|--------|-------|---------|
| `primitives.py` | 660 | Core VSA algebra (all primitives) |
| `accumulator.py` | 167 | Streaming operations |
| `scalar.py` | 147 | Continuous value encoding |
| `encoder.py` | 922 | Data encoding (thin wrapper) |

**Key insight**: Encoder methods now delegate to module functions, maintaining backward compatibility while improving organization.

### Rust Port

Added 9 primitives to `holon-rs/src/primitives.rs`:
- `unbind`, `similarity_profile`, `attend`
- `analogy`, `project`, `conditional_bind`
- `complexity`, `invert`, `segment`

New enums: `AttendMode`, `GateMode`, `SegmentMethod`

### Test Coverage

- Python: 497 tests passing
- Rust: 74 tests passing (15 in primitives module)

## Observations from 001-explainable-anomaly-forensics.py

### segment() Behavior

Detected 20 breakpoints in 450-event stream. Breakpoints cluster at phase transitions but also trigger on individual outliers within phases.

**Tuning needed**: `window` and `threshold` parameters significantly affect sensitivity. Window=20, threshold=0.4 worked well for this scenario.

### complexity() Consistency

Complexity scores were surprisingly consistent across phases (~0.81-0.83):
- Normal: 0.816
- Scan: 0.826
- Credential stuffing: 0.828
- Exfiltration: 0.812

**Insight**: The current complexity formula (0.4*density + 0.3*balance + 0.3*entropy) may need tuning. Attack types don't differentiate as expected.

**Future work**: Consider per-field complexity or comparison to baseline complexity.

### invert() Accuracy

Correctly identified top-1 pattern for all test samples:
- Scan sample → scan_probe: 0.778
- Cred sample → credential_stuffing: 0.872
- Exfil sample → exfiltration: 1.000

**Key finding**: `invert()` is highly effective for pattern attribution when codebook is well-constructed.

### attend() Mode Differences

Hard attention dramatically reduces complexity (0.81 → 0.45) by zeroing non-resonant dimensions. Soft and amplify modes preserve original structure while weighting.

### project() Subspace Ratios

All samples showed >100% projection ratio onto attack subspace, including normal samples. This suggests the attack subspace (3 prototypes) is too broad.

**Fix needed**: Use smaller, more discriminative subspaces, or compare projection ratios between attack and normal subspaces.

### analogy() Transfer Quality

0.570 similarity when predicting port_443_exfil from port_80 patterns. Reasonable but not high.

**Insight**: Analogy works best when the relational structure is preserved (port change is a clean transformation).

### conditional_bind() Sparsity

Gated binding (229/4096 dims) vs full binding (1065/4096 dims) - 78% sparsity reduction.

**Use case validated**: Context-aware encoding creates sparser, more targeted representations.

## Open Questions

1. **How to tune segment() parameters for real-world traffic?**
   - Should window scale with traffic rate?
   - Is threshold=0.3 generally applicable?

2. **Can complexity() differentiate attack types with better formulation?**
   - Per-field complexity?
   - Delta-complexity from baseline?

3. **How to construct effective subspaces for project()?**
   - Too many exemplars = everything projects
   - Too few = misses nuances

4. **What's the optimal codebook size for invert()?**
   - Diminishing returns after N patterns?
   - Hierarchical codebooks (attack_family → specific_attack)?

## Observations from 002-improved-detection-attribution.py

### Disagreement Ratio as Primary Signal

The `similarity_profile()` derived disagreement ratio provides the clearest separation:

| Traffic Type | Disagreement Ratio |
|--------------|-------------------|
| Normal | 0.043 |
| DNS Reflection | 0.444 |
| SYN Flood | 0.115 |
| NTP Amplification | 0.426 |

**Key insight**: Threshold at 0.10 gives excellent separation between normal and attack traffic.

### Attribution Accuracy Improvement

With proper feature encoding (including direction hint for amplification pattern):
- DNS reflection: 100% (up from 34.7% in first iteration)
- SYN flood: 100%
- NTP amplification: 100%

**Fix that worked**: Adding `src_port_band` feature to differentiate DNS vs NTP (both were being confused as "UDP from wellknown port").

### Detection Metrics

| Metric | Batch 012 Best | Batch 014 |
|--------|----------------|-----------|
| F1 | 0.875 | 0.897 |
| Precision | 0.82 | 0.858 |
| Recall | 0.93 | 0.940 |

**Improvement**: 2.5% F1 gain while adding attribution capability.

## Observations from 003-targeted-rate-limiting.py

### The Key Insight: Scale Similarity by Anomalous Ratio

The original approach (suppress anomalous dimensions) didn't work - it shrank vector magnitude.

**Winning approach**: Use anomalous_ratio from `similarity_profile()` to SCALE the baseline similarity:

```python
rate_factor = baseline_similarity * (1 - anomalous_ratio)
```

This creates multiplicative penalty:
- Low anomalous (8%): similarity * 0.92 (small penalty)
- High anomalous (25%): similarity * 0.75 (larger penalty)

### 2x Wider Separation Gap

| Metric | Old Approach | Targeted Approach |
|--------|--------------|-------------------|
| Gap (min_legit - max_attack) | 0.171 | **0.337** |
| Attacks blocked (< 0.3) | 0% | **100%** |

The targeted approach creates 2x wider separation between legitimate and attack traffic.

### Why This Matters

With wider separation:
- Easier to find a threshold that blocks attacks but allows legitimate traffic
- Less tuning required for different traffic mixes
- More robust to edge cases

## Observations from 004-attack-variant-detection.py

### Zero-Shot Detection via analogy()

**Key result**: Training on DNS reflection alone enables detection of UNSEEN amplification variants:

| Variant | Training | Detection | Analogy Similarity |
|---------|----------|-----------|-------------------|
| NTP | No | 100% | 0.771 |
| SSDP | No | 100% | 0.488 |
| CHARGEN | No | 100% | 0.775 |

### Why Analogy Works for Amplification

All amplification attacks share structure:
```
src=wellknown_port, dst=ephemeral, protocol=UDP, payload=large
```

Analogy transfers this structure: `DNS_attack - DNS_port + NTP_port = NTP_attack`

The inferred NTP attack vector captures the "amplification shape" without specific NTP training data.

### SSDP Lower Similarity

SSDP (0.488) has lower similarity than NTP/CHARGEN (0.77) because:
- SSDP (port 1900) is not encoded as "wellknown" in our scheme
- The analogy reasoning is slightly weaker

**Fix**: Expand port classification or add "known_amplification_port" feature.

## Updated Open Questions

1. ~~**How to tune segment() for real-world traffic?**~~ → Threshold=0.4, window=50 works for simulated traffic

2. ~~**Can complexity() differentiate attack types?**~~ → Not with current formula; not a priority given other signals work well

3. **Can we combine analogy + attend for targeted variant detection?**
   - Use analogy to infer variant structure
   - Use attend to focus on variant-specific dimensions

4. **What's the coverage limit of analogy-based detection?**
   - Works well for same-structure attacks (all amplification)
   - Unknown for attacks with different structures (scan vs exfil)

## Completed Experiments

- [x] **002**: Improved detection with pattern attribution (F1=0.897)
- [x] **003**: Targeted rate limiting with attend()
- [x] **004**: Zero-shot variant detection via analogy (100% detection)

## Future Experiments

- [ ] **005**: SOC analyst explainability dashboard
- [ ] **006**: Cross-structure analogy (can scan knowledge help detect exfil?)
- [ ] **007**: Real-time streaming segment detection

---

*Updated: February 2026*
