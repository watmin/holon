# Challenge Batch 018: Eigenvalue-as-Probe — Window-Level Pattern Matching

## Philosophy

**Batch 017 learned patterns from individual vectors. Batch 018 asks whether
we can match patterns from the *shape of a time window*.**

Every approach so far scores one vector at a time against a stored engram.
That works well for single-packet detection and eager activation. But it
requires encoding each packet, computing a residual against each engram, and
acting on the result — per packet, per engram.

There's a different question you can ask: instead of "does this packet look
like a known attack?", ask "does this *30-second window of traffic* look like
a known attack?" If the shape of the traffic distribution over a window
resembles the eigenvalue fingerprint of a known pattern, you can detect it
without scoring individual packets at all.

The mechanism: `match_spectrum`. It compares the eigenvalue signature of a
probe subspace — trained over a live window — against the stored eigenvalue
signatures in the library using cosine similarity. No full residual computation.
No per-packet encoding in the matching step. Just shape-to-shape comparison.

This opens three new capabilities:

1. **Early warning**: Attack patterns distort the eigenvalue spectrum before
   enough packets accumulate for per-packet matching. Eigenvalue shift can be
   the *first* signal, not a confirmation signal.

2. **Blind detection**: You can detect that *something is wrong* without
   knowing which packets are anomalous — the window as a whole has an unusual
   shape, even if individual packets pass residual thresholds.

3. **Temporal fingerprinting**: Different attack types produce different
   eigenvalue distortions. A slow exfiltration has a different spectrum shift
   than a SYN flood. The shape of a window can classify the attack type before
   per-packet analysis confirms it.

## The Primitive

### `match_spectrum` on `EngramLibrary`

```python
matches = library.match_spectrum(probe_eigenvalues, top_k=5)
# → [(name, cosine_similarity), ...] sorted descending
```

**What it computes**: Cosine similarity between the probe's eigenvalue vector
and each stored engram's eigenvalue signature. Returns the top-k matches.

**What it skips**: Full residual computation. No per-packet scoring. The
matching cost is O(n·k) where n is the number of engrams and k is the number
of eigenvalue components — orders of magnitude cheaper than full residual
scoring at O(n·k·dim).

**How to get a probe**: Train a short-window `OnlineSubspace` over a sliding
window of recent traffic, then call `subspace.eigenvalues` (or equivalent)
to get the eigenvalue vector to probe with.

```python
window_sub = OnlineSubspace(dim=4096, k=64)
for packet in recent_window:
    window_sub.update(encode(packet))

matches = library.match_spectrum(window_sub.eigenvalues, top_k=3)
```

## Experiments

### 001: Eigenvalue Shift as Early Warning

**Goal**: Demonstrate that eigenvalue spectrum distortion occurs earlier in
an attack than per-packet residual hits. The eigenvalue signal leads; the
residual signal confirms.

**Approach**:
1. Train a baseline subspace on 500 normal packets
2. Begin a slow-ramp SYN flood (traffic mixed 90/10 normal/attack, shifting
   to 50/50 over 200 packets, then 100% attack)
3. Track two signals over time:
   - Cosine similarity of live window eigenvalues to stored attack engram
   - Per-packet residual hit rate (fraction exceeding threshold)
4. Identify the packet number at which each signal first exceeds its threshold

**Success criteria**:
- Eigenvalue similarity signal exceeds 0.7 at least 20 packets before
  per-packet hit rate exceeds 50%
- Both signals converge to the same attack type by packet 100
- On pure normal traffic: eigenvalue similarity to attack engrams < 0.5

---

### 002: Attack Type Classification from Window Shape Alone

**Goal**: Classify attack type using only the eigenvalue spectrum of a
traffic window — no per-packet scoring, no residual computation.

**Approach**:
1. Build an engram library from 4 attack types:
   - DNS amplification (high volume, uniform dst_port=53, large bytes)
   - SYN flood (high volume, randomized src_ip, uniform dst_port, small bytes)
   - Slow exfiltration (low volume, long duration, distinctive payload sizes)
   - Credential stuffing (moderate volume, structured paths, varied status codes)
2. For each attack type, generate 10 fresh traffic windows (200 packets each)
3. Train a short-window subspace per window, extract eigenvalues
4. Run `match_spectrum` against the library for each window
5. Measure: does the top match correspond to the correct attack type?

**Success criteria**:
- Overall classification accuracy > 75% from eigenvalue shape alone
- Each attack type correct > 60% of the time
- Wrong matches are at least plausible (similar traffic structure)
- Normal traffic windows: top match score < 0.6 for all engrams

---

### 003: Window Size Sensitivity

**Goal**: Find the minimum window size (number of packets) at which
eigenvalue matching becomes reliable.

**Approach**:
1. Use the 4-engram library from experiment 002
2. For each attack type, sweep window sizes: 10, 25, 50, 100, 200, 500 packets
3. At each window size, train a subspace and run `match_spectrum`
4. Measure classification accuracy vs window size

**Success criteria**:
- Reliable matching (>70% accuracy) achievable in ≤ 100 packets
- Clear accuracy knee: minimal gain above some window size threshold
- Accuracy curve is monotonically non-decreasing with window size

---

### 004: Blind Anomaly Detection — Unknown Attack Shape

**Goal**: `match_spectrum` returns low similarity for all engrams when a
window contains a novel attack type not in the library. The low similarity
itself is the anomaly signal — no match is a warning.

**Approach**:
1. Build library from 3 known attack types (DNS amp, SYN flood, exfiltration)
2. Generate windows containing a 4th type not in the library (credential stuffing)
3. Also generate: pure normal, known attack types, and 50/50 normal+unknown
4. For each window, record max cosine similarity across all engrams
5. Threshold: max similarity < 0.5 → "anomalous but unrecognized"

**Success criteria**:
- Normal windows: max similarity < 0.4
- Known attack windows: max similarity > 0.7 (correct match)
- Unknown attack windows: max similarity < 0.5 (rejected — anomalous but not matched)
- Mixed (50/50 normal + unknown): max similarity between 0.4 and 0.65
  (degraded match signal, distinct from both normal and known)

---

### 005: Eigenvalue Matching as Pre-Filter for Per-Packet Scoring

**Goal**: Use `match_spectrum` to identify the most likely engram before
running full residual scoring. Measure accuracy cost vs compute cost of
the two-stage approach vs brute-force full scoring.

**Approach**:
1. Build a library of 8 engrams (4 attack types × 2 parameter variants each)
2. For each probe window:
   a. Run `match_spectrum` → top-2 candidate engrams
   b. Score individual packets only against those 2 candidates (not all 8)
   c. Compare: does the two-stage approach produce the same final match
      as scoring against all 8?
3. Measure: accuracy loss of two-stage vs brute-force
4. Measure: per-packet compute cost (O(2·k·dim) vs O(8·k·dim))

**Success criteria**:
- Two-stage accuracy ≥ 95% of brute-force (at most 5% degradation)
- 4× compute reduction (scoring 2 instead of 8 engrams per packet)
- Failure cases are identifiable: when does the pre-filter pick the wrong candidate?

---

### 006: Temporal Evolution of Eigenvalue Fingerprint

**Goal**: Characterize how the eigenvalue spectrum evolves during an attack
lifecycle — onset, peak, and subsidence. Different phases should produce
different spectra; the evolution itself is a temporal fingerprint.

**Approach**:
1. Simulate a complete attack lifecycle:
   - Phase 0: 200 normal packets
   - Phase 1: attack ramp-up (100 packets, 20%→80% attack)
   - Phase 2: full attack (200 packets, 100% attack)
   - Phase 3: subsidence (100 packets, 80%→0% attack)
   - Phase 4: 200 normal packets (recovery)
2. Every 25 packets, train a fresh 50-packet window subspace, extract eigenvalues
3. Plot cosine similarity of each window eigenvalue vector to:
   - The stored attack engram
   - The normal baseline engram

**Success criteria**:
- Similarity to attack engram peaks during phase 2, near 0 in phases 0 and 4
- Similarity to normal engram inversely tracks attack similarity
- Phase transitions (onset, subsidence) are detectable as slope changes
- Clear hysteresis: subsidence phase returns to normal slower than onset rises

---

### 007: Cross-Implementation Eigenvalue Consistency

**Goal**: Verify that eigenvalue signatures produced by the Python
`OnlineSubspace` and the Rust `OnlineSubspace` (holon-rs) are comparable
— i.e., that `match_spectrum` can work across implementations if needed.

**Note**: This is informational. We expect the eigenvalue spectra to be
similar in shape (same algorithm, same data) but not bitwise identical
(different floating-point order of operations). The question is whether the
cosine similarity between Python and Rust eigenvalue vectors is high enough
to be useful.

**Approach**:
1. Generate 500 normal traffic packets; encode with Python holon
2. Train Python `OnlineSubspace`, extract eigenvalues
3. Encode the same 500 packets with holon-rs (same schema, same seed)
4. Train Rust `OnlineSubspace`, extract eigenvalues
5. Compute cosine similarity between Python and Rust eigenvalue vectors

**Success criteria**:
- Cosine similarity between Python and Rust eigenvalue vectors > 0.95
- If < 0.95: document the divergence, understand whether it's algorithmic
  or numerical, and note it as a cross-implementation constraint

---

## What We're Building Toward

```
                    PER-PACKET SCORING        WINDOW-LEVEL SCORING
                    ──────────────────        ────────────────────
Mechanism:          residual(vec)             match_spectrum(eigenvalues)
                    vs each engram            vs each engram signature

Cost:               O(n·k·dim) per packet     O(n·k) per window

Signal timing:      confirmation              early warning

Granularity:        which packet              which window

Use case:           single-packet triggers    trend detection
                    eager activation          temporal fingerprinting
                    field attribution         blind anomaly detection

Combined:           match_spectrum identifies candidate engram →
                    per-packet residual scoring against candidate only →
                    field attribution on confirmed anomalous packets
```

Window-level and packet-level scoring are complementary layers. Window
matching is cheap, early, and operates without per-packet labels. Packet
matching is more expensive, later, and provides field-level attribution.
The ideal pipeline runs both: the window signal narrows the candidate engram,
the packet signal confirms and attributes.

---

*Created: February 2026*
