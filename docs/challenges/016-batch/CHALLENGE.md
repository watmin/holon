# Challenge Batch 016: Advanced Vector Properties for DDoS Detection

## Philosophy

**Traditional VSA exploits one property of vectors: cosine similarity to a reference.
We exploit six more.**

Batches 010–015 proved that VSA/HDC can detect, classify, rate-limit, and explain
DDoS attacks. But the entire detection pipeline depends on one thing: a baseline.
Accumulate normal traffic, measure drift from that baseline, alert when drift
exceeds a threshold.

This works — until it doesn't:

- **Cold start**: No baseline exists. First N seconds are blind.
- **Baseline corruption (boiling frog)**: Attacker ramps slowly over hours.
  The baseline adapts. Drift stays low. Attack succeeds.
- **Magic thresholds**: "Alert when similarity < 0.5" — why 0.5? Why not 0.4?
  The threshold is a guess, not a derivation.
- **Binary detection**: Similarity dropped. But HOW did it drop? Flash flood
  or gradual shift? Same alert, completely different response.
- **Layered attacks**: Two concurrent attacks. You detect the obvious one.
  Is there a second one hiding underneath?

This batch introduces operations that attack each of these problems by
exploiting **mathematical properties of vectors that the VSA literature ignores**.

## The New Vector Properties

### 1. Pairwise Similarity Distribution (Coherence)

**Property**: Not "how similar is X to Y?" but "how similar are X₁, X₂, ..., Xₙ
to each other?"

**Operation**: `coherence(vectors)` — mean pairwise cosine similarity.

**Why it matters**: Normal traffic is diverse (many different packet types).
Attack traffic is homogeneous (same pattern repeated). Coherence measures
this without ANY reference to a baseline. The window's internal structure
IS the signal.

```python
window_vecs = [encode(pkt) for pkt in recent_packets]
c = coherence(window_vecs)
# Normal: c ≈ 0.12 (diverse, nearly orthogonal)
# Attack: c ≈ 0.50 (repetitive, high pairwise similarity)
```

**Solves**: Cold start, baseline corruption.

### 2. Statistical Geometry (Significance)

**Property**: In high dimensions, random vectors are nearly orthogonal with
cosine similarity ≈ N(0, 1/√d). A raw similarity score means nothing without
knowing the dimensionality.

**Operation**: `significance(similarity, dimensions)` → z-score.

**Why it matters**: A cosine similarity of 0.05 is noise at d=64 (z=0.4) but
highly significant at d=4096 (z=3.2). This replaces magic thresholds with
principled, dimension-aware statistical tests.

```python
z = significance(sim, 4096)
# z > 3.0 → p < 0.003, not random chance
# z > 5.0 → essentially certain
```

**Solves**: Magic thresholds.

### 3. Temporal Derivative (Drift Rate)

**Property**: Not "how different is current from baseline?" (state) but
"how fast is the difference changing?" (dynamics).

**Operation**: `drift_rate(stream, window)` → rate of similarity change.

**Why it matters**: A flash flood and a gradual organic growth both produce
"low similarity to baseline." But a flash flood has a massive negative spike
in drift rate, while organic growth has a slow steady decline. Same state,
completely different dynamics, completely different response.

```python
rates = drift_rate(window_vectors, window=1)
# Flash flood: min(rates) < -0.5 (sudden spike)
# Ramp-up:    consecutive negative rates (accelerating decline)
# Organic:    rates ≈ 0 (stable)
# Pulsed:     alternating ±0.3 (oscillation)
```

**Solves**: Binary detection ("something happened" vs "what kind of thing").

### 4. Per-Dimension Agreement Strength (Confidence Margins)

**Property**: When bundling N vectors, each dimension has a vote. The margin
of victory (512-to-512 vs 1000-to-24) is discarded by standard `bundle()`.

**Operation**: `bundle_with_confidence(vectors)` → (bundled, margins).

**Why it matters**: Feed margins into `weighted_cosine_similarity` and drift
detection focuses on dimensions the baseline is CONFIDENT about, ignoring
dimensions that are inherently noisy. This is learned feature importance
without machine learning.

```python
baseline, margins = bundle_with_confidence(baseline_packets)
# margins[i] = 1.0 → all packets agreed on dim i (trust it)
# margins[i] = 0.0 → perfect tie (ignore it)

drift = 1.0 - weighted_cosine(current, baseline, margins)
# Only trusted dimensions contribute to drift
```

**Solves**: False positives from noisy dimensions.

### 5. Subspace Residuals (Reject)

**Property**: `project()` tells you what's IN a subspace. `reject()` tells you
what's NOT in it — the orthogonal complement, the unexplained residual.

**Operation**: `reject(vec, subspace)` → residual vector.

**Why it matters**: After identifying a known attack, reject it to see if
there's a SECOND attack hiding underneath. This is signal separation —
peeling known signals to expose hidden ones.

```python
# Peel known DNS attack
peeled = negate(traffic, dns_attack_profile)
# Reject all known attack types
residual = reject(peeled, [dns_profile, syn_profile, ntp_profile])
# residual = novel attack vector (if any)
```

**Solves**: Layered attacks, novel attack discovery.

### 6. Scalar Round-Trip (Decode)

**Property**: Encoding goes value → vector. Decoding goes vector → value.
The round-trip through vector space produces the **consensus** of accumulated
observations.

**Operation**: `decode_scalar_log(vec)` → scalar value.

**Why it matters**: Closes the loop on vector-derived rate limiting (Batch 013).
Encode baseline rates → accumulate → decode the accumulated vector → get an
actual PPS number for rate limiting rules. No stored state, no counters — the
rate limit emerges from the algebra.

```python
# Encode rate observations
for rate in observed_rates:
    accumulate(acc, encode_scalar_log(rate))
# Decode consensus rate
effective_rate = decode_scalar_log(normalize(acc))
# → the frequency-weighted average rate
```

**Solves**: Rate limit derivation without counters.

## Experiments

### 001: Coherence as Baseline-Free Attack Detection

**Goal**: Detect attacks with NO baseline — cold start scenario.

**Approach**:
1. Generate normal traffic (diverse sources, protocols, ports)
2. Generate two attack types (DNS amplification, botnet SYN flood)
3. Measure coherence of each window
4. Compare with significance z-scores

**Success criteria**:
- Coherence separates normal from attack by > 2× ratio
- No false positives on normal traffic
- Works for both attack types without tuning

**Follow-up experiments**:
- Sensitivity: minimum attack fraction for detection
- Window size: does separation hold at small windows?
- Comparison: coherence vs similarity-to-baseline

### 002: Drift Rate for Attack Onset Classification

**Goal**: Classify attack TYPE from the shape of the drift rate timeseries.

**Approach**:
1. Generate four stream types: stable, flash flood, ramp-up, pulsed
2. Compute drift_rate() for each
3. Classify from drift shape features (max spike, acceleration, oscillation)

**Success criteria**:
- Flash flood: negative spike > 0.5
- Ramp-up: consecutive negative rates
- Pulsed: high sign-change frequency
- Stable: max|drift| < 0.1

**Follow-up experiments**:
- Windowed smoothing effect on classification
- Combined drift_rate + coherence for richer signal

### 003: Confidence-Weighted Drift Detection

**Goal**: Reduce false positives by weighting drift detection with per-dimension
confidence margins.

**Approach**:
1. Build baseline with `bundle_with_confidence()` → get margins
2. Compare unweighted cosine vs weighted cosine drift
3. Test three scenarios: normal traffic, attack on noisy dims, attack on stable dims

**Success criteria**:
- Normal traffic: weighted drift < unweighted drift (fewer false positives)
- Attack on stable dims: weighted drift ≥ unweighted drift (no sensitivity loss)
- Clear margin distribution: some high-confidence, some low-confidence dims

**Follow-up experiments**:
- Inverted weights (1 - margin) for novelty detection in noisy dimensions
- Sensitivity curve: minimum attack fraction for detection with both metrics

### 004: Reject for Novel Attack Isolation

**Goal**: Discover unknown attack vectors after peeling known attacks.

**Approach**:
1. Build known attack profiles (DNS amp, SYN flood, NTP amp)
2. Generate layered traffic: known attack + novel SSDP attack + normal
3. Project onto known attacks → measure explained component
4. Peel known attack with `negate()`
5. Reject known attack subspace → examine residual
6. Check if residual correlates with the novel attack

**Success criteria**:
- Peeling reduces similarity to known attack to < 0
- Residual has positive similarity to novel attack pattern
- Pipeline discovers the novel attack without prior SSDP knowledge

**Follow-up experiments**:
- Iterative peeling: peel → detect → peel → detect → ...
- Reject baseline (not attacks) to isolate anomalous component

### 005: Baseline-Free Multi-Signal Detection

**Goal**: Combine coherence, complexity, and purity for robust baseline-free
detection. Test against the "boiling frog" attack.

**Approach**:
1. Measure all three signals across attack fractions (0% to 100%)
2. Determine which signals trigger first
3. Simulate boiling frog: adaptive baseline + slow ramp attack
4. Compare baseline-dependent (similarity) vs baseline-free (coherence) detection
5. Compute signal correlations to assess independence

**Success criteria**:
- At least one signal detects at ≤ 30% attack fraction
- Boiling frog: baseline-free signal detects where similarity-to-baseline fails
- Signals are not fully correlated (< 0.8 pairwise)

### 006: Top-K Coherence for Low Attack Fraction Detection

**Goal**: Detect attacks at 2–10% traffic fraction, not 50%+.

**Approach**:
1. Compute ALL pairwise similarities (not just mean)
2. Compare distribution statistics: mean, median, P90, P95, P99, max, top-10 mean
3. Find which statistic triggers earliest at each attack fraction
4. Test stability across trials and window sizes

**Success criteria**:
- At least one statistic detects at ≤ 10% attack fraction
- No false positives on pure normal traffic (20 trials)
- Detection is stable (CV < 15% across trials)

**Constraint**: Stateless, per-window, unidirectional traffic only.

### 007: Per-Field Coherence Spectrum via Unbinding

**Goal**: Fingerprint attack TYPE by which fields are homogeneous.

**Approach**:
1. Unbind each packet vector by each field's role vector
2. Compute per-field coherence → produces a "spectrum"
3. Compare spectra across attack types (DNS amp, SYN flood, NTP, SSDP, ICMP, port scan)
4. Test if spectra can classify attack type at 20% attack fraction

**Success criteria**:
- Each attack type produces a distinctive spectrum shape
- Attack classification from spectrum matches expected field patterns
- Works at mixed traffic (80% normal, 20% attack)

**Constraint**: Stateless, per-window, unidirectional traffic only.

### 008: Improved Drift Rate Classification

**Goal**: Fix the 25% accuracy from experiment 002 using proper feature extraction.

**Approach**:
1. Extract statistical features from drift rate timeseries (peak count, spike ratio,
   mean absolute drift, crossing rate, trend correlation, energy ratio)
2. Build a decision tree from feature distributions
3. Evaluate over 50 trials per onset type

**Success criteria**:
- Classification accuracy > 75% (up from 25%)
- No misclassification between "attack present" and "no attack" (stable vs. others)
- Feature distributions show clear inter-class separation

**Constraint**: Stateless, per-window, unidirectional traffic only.

### 009: Adversarial Robustness — Evasion vs Detection

**Goal**: Determine if an attacker can evade coherence detection without breaking
the attack.

**Approach**:
1. Progressive evasion: randomize fields one by one (src_ip → pkt_len → dst_port → proto → dst_ip)
2. At each level, measure coherence AND attack effectiveness
3. Compare mean coherence vs P99 (top-k) detection
4. Analyze the fundamental trade-off between evasion and effectiveness

**Success criteria**:
- Coherence detects all levels where the attack is still effective
- Detection failure only occurs when the attack is no longer viable
- Demonstrate the attacker's dilemma: hide → stop attacking

**Constraint**: Stateless, per-window, unidirectional traffic only.

## What We're Building Toward

The detection pipeline evolves from single-signal to multi-property:

```
                    BASELINE-DEPENDENT          BASELINE-FREE
                    ──────────────────          ─────────────
Detection:          similarity to baseline      coherence of window
                                                purity of accumulator

Dynamics:           —                           drift_rate (onset type)

Thresholds:         magic numbers               significance (z-score)

Sensitivity:        unweighted cosine           confidence-weighted cosine

Investigation:      invert, similarity_profile  reject + negate (peeling)

Rate limiting:      magnitude ratio             decode_scalar_log (round-trip)
```

The ultimate goal: **a detection system with no warmup, no magic thresholds,
no blind spots for slow ramps, and the ability to peel apart concurrent
attacks.** Every property of the vector carries signal — direction, magnitude,
pairwise distribution, temporal derivative, per-dimension confidence, and
subspace residuals.

### 010: Payload Anomaly Detection — VSA-to-XDP Byte Match Pipeline

**Goal**: Detect attacks that are structurally normal (correct headers) but carry
unfamiliar payload bytes. Generate surgical `(l4-match)` byte match rules to
enforce in the XDP filter.

**Scenario**: UDP game server. Legit and attack traffic share identical headers.
The only signal is unfamiliar payload content at high velocity.

**Approach**:
1. Encode first 32 bytes of payload positionally (pos_i → byte_hex atoms)
2. Accumulate legitimate payloads into a familiar-traffic baseline
3. Score new payloads against baseline — low similarity = unfamiliar
4. Adaptive threshold: legit_mean - 3σ (data-driven, not magic number)
5. Among flagged outliers: measure coherence (coordinated attack cluster?)
6. If coherent: prototype the cluster, extract consensus byte positions
7. Filter candidate rules against legit traffic (drop rules with >10% false positive)
8. Output `(l4-match offset hex-match hex-mask)` rules for XDP enforcement

**Success criteria**:
- Payload similarity separates legit from attack traffic (>1.5× ratio)
- Zero false positives at adaptive threshold
- Coherence among flagged outliers confirms coordinated attack (>0.5)
- Generated byte match rules: 0% false positive, >90% true positive
- Full pipeline: detect → cluster → extract → rule, end-to-end

**Attack types tested**:
- Attack A: Buffer overflow (NOP sled + padding, no protocol header)
- Attack B: Spoofed game header (valid magic bytes, invalid type + garbage body)
- Attack C: Random flood (completely random bytes)

**What this proves**:
Header-agnostic detection. When attackers send to the right IP:port with the right
protocol, the payload IS the signal. VSA can detect, cluster, and characterize
unfamiliar payload patterns, and the byte match rules feed directly into the
XDP filter's `(l4-match)` capability — closing the loop from detection to enforcement.

**Constraints**: Stateless, per-window, unidirectional. No flow tracking.

### 011: Per-Position Payload Analysis — Surgical Byte Localization

**Goal**: Eliminate the cross-talk problem (experiment 007) for payload analysis by
using independent accumulators per byte position. Pinpoint exactly WHICH byte positions
are anomalous.

**Approach**:
1. Two-tier system: Tier 1 (whole-payload vector) for fast screening, Tier 2
   (per-position accumulators) for surgical localization
2. Train on 500 legitimate game protocol packets
3. Score each byte position independently against its position-specific accumulator
4. Generate l4-match rules from consistently anomalous positions with consensus bytes
5. Full pipeline: mixed window → Tier 1 detect → Tier 2 locate → consensus rule

**Attack types tested**:
- Overflow (NOP sled + padding)
- Spoofed header (valid magic bytes, wrong message type + high-byte garbage)
- Subtle (valid header, unusual body byte range 0x70-0x7F)

**Success criteria**:
- Per-position scores eliminate cross-talk (unfamiliar at position X doesn't affect Y)
- Protocol structure is recoverable from accumulator state
- Consensus rules: 0% false positive, >80% true positive
- Full pipeline works end-to-end on mixed traffic

**Constraints**: Stateless, per-window, unidirectional. No flow tracking.

### 012: Unified Payload Baseline — Zero New Machinery

**Goal**: Prove that simply adding payload bytes (`p0`, `p1`, ...) to the existing
packet encoding map is sufficient. No new accumulators, no new data structures.
The existing baseline + drill-down handles everything.

**Approach**:
1. Encode packets as: `{"src_ip": ..., "proto": ..., "p0": "0x47", "p1": "0x4d", ...}`
2. Same accumulator, same similarity check, same drill-down as header-only detection
3. Drill-down reveals unfamiliar positions via negative similarity to baseline
4. Extract rules directly from drill-down analysis
5. Test progressive learning: add new legitimate message type, verify adaptation

**Success criteria**:
- Payload similarity separates legit from attack (>2× ratio)
- Drill-down correctly identifies payload byte positions as unfamiliar (not headers)
- Rule extraction: 0% FP, >90% TP
- Progressive learning: new legitimate patterns become familiar after accumulation

**What this proves**:
The simplest approach works. No architectural changes needed for payload detection —
the same VSA primitives that handle headers handle payload bytes identically.

**Constraints**: Stateless, per-window, unidirectional. No flow tracking.

### 013: Optimal Byte Match Rule Selection — Resource-Conscious Enforcement

**Goal**: From VSA drill-down candidates, find the single most effective and
resource-efficient `l4-match` rule. Score every option by TP rate, FP rate, and
resource cost.

**Approach**:
1. VSA drill-down identifies candidate positions (unfamiliar bytes)
2. For each position: evaluate exact match (0xFF mask), masked match (0xFC, 0xF0, etc.),
   and multi-byte contiguous combinations
3. Rank all candidates by coverage (highest TP) with zero FP constraint
4. Select the winner that uses the least resources (1 custom dim slot preferred)

**Resource constraints** (from eBPF filter):
- 7 custom dim slots (1-4 byte matches, O(1) cost) — PRECIOUS
- 32 byte matches per destination scope
- 65,536 pattern guard entries (5-64 byte matches)

**Attack types tested**:
- Overflow (uniform NOP sled)
- Overflow (varied: NOP variants 0x90-0x93)
- Polymorphic (only 2 stable bytes, rest varies)

**Success criteria**:
- Best rule: 100% TP, 0% FP
- Masked matches catch attack variation without false positives
- Total resource usage: ≤3 custom dim slots for 3 attack types
- Polymorphic attacks are handled by multi-byte rules

**Constraints**: Stateless, per-window, unidirectional. No flow tracking.

### 014: Sparse Byte Match — Non-Contiguous Anomaly Regions

**Goal**: When anomalous bytes are separated by normal-looking bytes in the middle,
generate ONE rule that spans the full range with `mask=0x00` for the familiar
middle positions.

**Scenario**: Attack payload with exploit header (positions 0-3), valid game traffic
in the middle (positions 4-13), and shellcode stub (positions 14-19). Instead of
two separate rules, one sparse rule covers everything.

**Approach**:
1. VSA drill-down identifies anomalous and familiar regions
2. For each position in the span: anomalous → find best mask, familiar → mask=0x00
3. Build single rule spanning first-to-last anomalous position
4. Compare resource cost: 1 sparse rule vs. N separate rules

**Success criteria**:
- Single rule with zeros in the mask for familiar middle positions
- 0% false positive rate
- Resource cost: 1 pattern guard instead of 2+ separate rules
- VSA drill-down correctly distinguishes anomalous from familiar zones

**Constraints**: Stateless, per-window, unidirectional. No flow tracking.

### 015: Sparse Mask Clean — End-to-End Generic Detection-to-Enforcement

**Goal**: Clean demonstration of the complete pipeline: stream traffic, learn
baseline, detect anomalous payload, drill down to identify unfamiliar positions,
generate sparse byte match rule — all with zero domain knowledge.

**Scenario**:
```
Normal traffic: each position has ~6 familiar byte values
Attack payload: [FF, AC, FB, CA, 01, 02, 03, FF, FA, CB]
                 ^^^^^^^^^^^^^^  ^^^^^^^^  ^^^^^^^^^^^^
                  unfamiliar     familiar   unfamiliar
```

**Approach**:
1. Learn 500 normal payloads (diverse byte values per position)
2. Attack arrives with mostly unfamiliar bytes, but positions 4-6 happen to be familiar
3. VSA drill-down identifies familiar vs unfamiliar at each position
4. Familiar positions → mask=0x00 (skip). Unfamiliar → best mask for coverage.
5. For varied attacks: automatic mask loosening (0xFF → 0xFC) to catch byte variation
6. Output single l4-match rule with zeros for "don't care" middle bytes

**Success criteria**:
- 100% TP, 0% FP for both uniform and varied attacks
- Masks automatically adapt to attacker byte variation
- 1 pattern guard total cost
- Fully generic: no signatures, no threat intel, no labels, no domain knowledge

**What this proves**:
The entire chain — from "I've never seen these bytes before" to "here's a surgical
kernel filter rule" — requires zero knowledge of the attack or the protocol. The
system learns what's familiar from traffic and generates enforcement rules from
the unfamiliar. This is the culmination of the batch.

**Constraints**: Stateless, per-window, unidirectional. No flow tracking.

---

*Created: February 2026*
