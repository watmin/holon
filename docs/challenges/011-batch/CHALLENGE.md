# Challenge Batch 011: Scoped Vectors & Adaptive Dynamics

## Overview

Building on batch 010's success with multi-signal detection and continuous learning, this batch explores **scoped vector architectures** and **adaptive temporal dynamics**.

## Key Experiments

### 1. Scoped Vectors by Component

Instead of encoding entire requests/packets as single vectors, maintain separate accumulators per component:

```
HTTP Request:
├── path_accumulator     → path similarity
├── query_accumulator    → query similarity
├── headers_accumulator  → headers similarity
└── body_accumulator     → body similarity

Packet:
├── src_addr_accumulator → source pattern
├── dst_addr_accumulator → destination pattern
├── port_accumulator     → port usage pattern
├── proto_accumulator    → protocol pattern
└── size_accumulator     → payload size pattern
```

**Hypothesis**: Component-level anomaly detection provides:
- Better explainability (which component triggered?)
- Finer-grained detection (one unusual field doesn't mask normal ones)
- Natural aggregation strategies (voting, weighted, min-of-N)

### 2. Black Box Packet Analysis

Treat packets as opaque byte sequences:
- Positional encoding (byte at position N)
- N-gram analysis on raw bytes
- No protocol knowledge required

**Hypothesis**: Can detect novel protocols, tunneling, and encoding tricks that field-based analysis misses.

### 3. N-gram / Sequence Encoders

For HTTP text analysis:
- Character-level n-grams (catch `'--`, `<script>`)
- Token-level n-grams (catch `SELECT FROM`)
- Position-weighted patterns

**Hypothesis**: Sequence structure contains attack signatures that structural fingerprints miss.

### 4. Prefix Tracking for Reflection

Track destination IP prefixes to detect:
- Amplification attacks (spray to many /24s)
- Reconnaissance (scanning patterns)
- Exfiltration (unusual destination concentration)

### 5. Adaptive Decay Mechanics

Current decay=0.9995 gives ~2000 observation window. Explore:
- Rate-adaptive decay (adjust based on traffic volume)
- Time-based decay (half-life in seconds, not observations)
- Multi-horizon accumulators (fast + slow for trend detection)

### 6. Smart Checkpoints

Save "known good" reference states to:
- Detect attack exit (return to baseline)
- Resist long-running poisoning
- Enable before/after comparison

## Implementation Files

```
scripts/challenges/011-batch/
├── 001-scoped-http-vectors.py      # Per-component HTTP accumulators
├── 002-scoped-pcap-fields.py       # Per-field packet accumulators
├── 003-ngram-text-encoding.py      # Character/token n-grams for HTTP
├── 004-adaptive-decay.py           # Rate/time-based decay strategies
├── 005-smart-checkpoints.py        # Pre-attack reference management
├── 006-raw-packet-multiperspective.py  # Scapy-based multi-perspective analysis
├── 007-deviation-detection.py      # Surprise/agreement metrics
└── deterministic_codebook.py       # Copied from 010-batch
```

## Results Summary

| Experiment | Best F1 | Key Finding |
|------------|---------|-------------|
| Scoped HTTP | 0.068 | Per-component works but needs threshold tuning |
| Scoped PCAP | 0.739 | Clear field signatures per attack type |
| N-gram Text | 0.191 | 100% recall but low precision |
| Adaptive Decay | N/A | Multi-horizon divergence clearly signals attacks |
| Checkpoints | N/A | Concept works, state machine needs refinement |

## Key Learnings

1. **Scoped vectors provide natural explainability** - "src_port anomaly = likely reflection"
2. **Multi-horizon divergence is THE signal** - 0.003 normal vs 0.17-0.21 during attack
3. **N-grams capture attack sequences** - but need combination with structural features
4. **Checkpoints enable recovery detection** - need hysteresis for stable transitions

See `LEARNINGS.md` for detailed analysis.

## Open Questions (Updated)

1. **Hybrid detector**: How to combine fingerprints + n-grams + scoped vectors?
2. **Attack classification**: Can we classify attack type from triggering fields?
3. **Checkpoint state machine**: What hysteresis prevents oscillation?
4. **Real traffic**: All experiments used synthetic data

---

*Created: February 2026*
*Updated after experiments: February 2026*
