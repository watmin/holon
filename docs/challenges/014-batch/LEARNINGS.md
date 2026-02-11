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

## Future Experiments

- [ ] **002**: Streaming segment detection (real-time phase changes)
- [ ] **003**: Attack clustering via complexity + projection
- [ ] **004**: Zero-shot detection via analogy transfer
- [ ] **005**: SOC analyst explainability dashboard

---

*Updated: February 2026*
