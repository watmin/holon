# Challenge Batch 017: Online Subspace Learning (HyperBox)

## Philosophy

**Bundles measure the center. Subspaces measure the shape.**

Every detection primitive so far — accumulators, prototypes, cosine drift — reduces
a set of familiar vectors to a single point (the centroid) and measures distance from
it. This works when anomalies are radially separated from the centroid. It fails when:

- **Non-radial anomalies**: The anomaly is the same distance from the centroid as
  normal data, but in a direction the normal data never explores. Two clusters of
  normal traffic (API calls and web browsing) have a centroid between them. An anomaly
  equidistant from the centroid but off-manifold is invisible to cosine drift.

- **High-dimensional sparsity**: In 4096D, "distance from center" is a single scalar.
  The subspace that normal data actually spans might be 50-100 dimensions out of 4096.
  The other 3900+ dimensions are noise. A centroid-based detector gives equal weight
  to all dimensions, diluting the signal.

- **Streaming adaptation**: Prototypes are averages. They can't distinguish "the
  distribution shifted" from "the distribution broadened." A subspace can — its
  dimensionality (number of significant eigenvalues) grows when the distribution
  broadens and stays constant when it shifts.

This batch introduces `OnlineSubspace`, a CCIPCA-based primitive that learns the
low-dimensional manifold from streaming hypervectors and uses the reconstruction
residual as an anomaly score.

## The Primitive

### OnlineSubspace (CCIPCA)

**What it learns**: The k-dimensional subspace (k << 4096) that familiar vectors
span. Tracks both the subspace orientation (basis vectors) and the variance along
each axis (eigenvalues).

**Algorithm**: Candid Covariance-Free Incremental PCA (Weng et al., 2003).
Per-vector updates, O(k*d) per step, no covariance matrix, no batch SVD.

**Key parameters**:
- `k`: Number of components (subspace rank). 32-128 typical for 4096D.
- `amnesia`: Forgetting exponent. Higher = adapts faster but forgets faster.
- `sigma_mult`: Threshold sensitivity. 3.5 = conservative, 2.0 = aggressive.

**Outputs**:
- `residual(x)`: Reconstruction error. High = out-of-subspace = anomalous.
- `project(x)`: Low-dimensional embedding (k coefficients). For clustering/viz.
- `threshold`: Adaptive cutoff from EMA of residual statistics.
- `eigenvalues`: Approximate spectrum. Shape reveals intrinsic dimensionality.
- `anomalous_component(x)`: The part of x that doesn't belong (vector, not scalar).

**Integration with existing primitives**:
- Accumulators learn the *mean* of traffic. Subspace learns the *shape*.
- `coherence()` measures within-window homogeneity. Subspace measures
  within-subspace membership across all history.
- `anomalous_component()` produces a vector that can be fed to `invert()` for
  attribution, or to `similarity_profile()` for dimension-wise analysis.

```python
from holon import HolonClient

client = HolonClient()
sub = client.create_subspace(k=64)

# Learn from stream
for record in normal_traffic:
    sub.update(client.encode(record))

# Detect anomalies and attribute
for record in mixed_traffic:
    vec = client.encode(record)
    if sub.residual(vec) > sub.threshold:
        # Which fields are surprising?
        fp = client.surprise_fingerprint(vec, sub,
            fields=["src_ip", "dst_port", "proto", "ttl"])
        # → {"ttl": 44.9, "dst_port": 44.4, "proto": 42.9, ...}

        # Take top-k surprising fields + their values from the record
        # → ((and (= ttl 245) (= dst_port 53)) => (drop))
```

Also available directly:

```python
from holon.subspace import OnlineSubspace

sub = OnlineSubspace(dim=4096, k=64)
```

## Experiments

### 001: CCIPCA Convergence on Structured Encodings

**Goal**: Show that CCIPCA converges to a meaningful subspace from holon-encoded
structured data, and that the eigenvalue spectrum reveals intrinsic dimensionality.

**Approach**:
1. Encode 2000 "normal" packet-like dicts with holon (varied IPs, ports, protocols)
2. Feed into OnlineSubspace one-by-one
3. Track residual, eigenvalue spectrum, and explained variance over time
4. Verify convergence: residual stabilizes, top eigenvalues dominate

**Success criteria**:
- Residual decreases monotonically during warmup (first ~200 vectors)
- Eigenvalue spectrum shows clear knee (few large, long tail of small)
- Explained variance > 0.5 with k=64 on 4096D structured data
- Convergence within 500 vectors (residual CV < 10% over last 100)

### 002: Anomaly Separation — Residual as Detector

**Goal**: Show that subspace residuals cleanly separate in-distribution from
out-of-distribution vectors, and that the adaptive threshold works.

**Approach**:
1. Train OnlineSubspace on 1000 normal traffic vectors
2. Score 200 normal (holdout) and 200 anomalous vectors
3. Anomalies: different schemas (e.g., DNS amplification vs normal web traffic)
4. Plot residual distributions, measure separation (AUC-equivalent)
5. Verify adaptive threshold: catches anomalies, no false positives on holdout

**Success criteria**:
- Residual distributions non-overlapping (or minimal overlap)
- Zero false positives on normal holdout at adaptive threshold
- > 95% true positive rate on anomalies at adaptive threshold
- Clear separation ratio: mean(anomaly_residuals) / mean(normal_residuals) > 2×

### 003: Subspace vs Centroid — Non-Radial Anomalies

**Goal**: Demonstrate the key advantage of subspace over cosine-to-centroid:
detecting anomalies that are equidistant from the center but off-manifold.

**Approach**:
1. Generate two clusters of normal traffic (e.g., API traffic + web traffic)
2. Compute centroid (prototype/bundle)
3. Craft an anomaly that has the same cosine distance to the centroid as
   normal data but lies in an unexplored direction
4. Compare: centroid misses it, subspace catches it

**Success criteria**:
- Cosine-to-centroid: anomaly score within normal range (false negative)
- Subspace residual: anomaly score above threshold (true positive)
- Clear demonstration that shape > center for multi-modal normal distributions

### 004: Streaming Drift Adaptation

**Goal**: Show the amnesia parameter allows the subspace to track concept drift
(evolving normal traffic) while still detecting anomalies.

**Approach**:
1. Phase 1: Normal traffic type A (e.g., web browsing patterns)
2. Phase 2: Gradual transition to type B (e.g., API-heavy patterns)
3. Phase 3: Steady type B
4. Throughout: inject anomalies and check detection
5. Compare amnesia=1.0 (no forgetting) vs amnesia=2.0 vs amnesia=3.0

**Success criteria**:
- amnesia=2.0: adapts to drift within ~200 vectors, still detects anomalies
- amnesia=1.0: fails to adapt (type B flagged as anomalous even after 500 vectors)
- amnesia=3.0: adapts too fast (unstable threshold, potential false negatives)
- Sweet spot exists where drift is absorbed but attacks are caught

### 005: DDoS Detection — Subspace vs Accumulator Pipeline

**Goal**: Full DDoS scenario comparing three detection approaches: cosine-to-baseline
(current), coherence (batch 016), and subspace residual (this batch).

**Approach**:
1. Normal traffic warmup (500 packets)
2. Attack onset (DNS amplification, SYN flood, or slow ramp)
3. Recovery to normal
4. Measure detection latency, false positive rate, and signal richness for each method
5. Show subspace projection coordinates cluster attack packets separately from normal

**Success criteria**:
- All three methods detect the attack
- Subspace provides additional signal: projection coordinates separate
  attack from normal in low-D space (visual cluster separation)
- At least one scenario where subspace detects earlier or with fewer
  false positives than the other two methods
- Subspace eigenvalue spectrum shifts during attack (new variance direction)

### 006: Feature Isolation via Anomalous Component

**Goal**: Show that unbinding field role vectors from `anomalous_component()`
isolates which fields make a vector anomalous — sharper than centroid drill-down.

**Approach**:
1. Train subspace on normal traffic (7 fields)
2. Generate three attack types with known "surprising" fields
3. For each attack: unbind each field's role vector from anomalous component
4. Rank fields by anomaly magnitude
5. Compare ranking quality (precision@k) to centroid drill-down

**Success criteria**:
- Subspace attribution P@3 > 50% (top-3 fields include ground truth)
- Subspace ranking quality >= centroid drill-down
- Signal contrast: surprising fields have >1.5× anomaly magnitude vs unsurprising

### 007: Surprise Fingerprint Generation

**Goal**: Show that per-field anomaly scores form a compact, consistent,
distinctive fingerprint that can classify attack type.

**Approach**:
1. Compute per-field anomaly magnitudes for each attack sample → fingerprint vector (7D)
2. Measure within-type consistency (same attack → similar fingerprint)
3. Measure between-type separation (different attack → different fingerprint)
4. Classify attack type by nearest-fingerprint matching

**Success criteria**:
- Within-type similarity > 0.8 (fingerprints are consistent)
- Within-type > between-type (fingerprints are distinctive)
- Classification accuracy > 75% from fingerprint alone
- Each attack has different dominant fields

### 008: End-to-End Subspace → Mitigation Rule Pipeline

**Goal**: Full pipeline from traffic stream to actionable mitigation rule.
Detect → extract anomalous component → identify surprising fields → find
consensus values → generate EDN-style rule predicate.

**Approach**:
1. Learn subspace from normal traffic
2. Detect attack traffic (residual > threshold)
3. Compute surprise fingerprint for detected packets
4. Top-K surprising fields → extract consensus field values
5. Generate rule: `((and (= field1 val1) (= field2 val2)) => (drop))`
6. Test rule: TP rate on attack holdout, FP rate on normal holdout

**Success criteria**:
- Detection rate > 90%
- Generated rule: TP > 50% on holdout attacks
- Generated rule: FP = 0% on holdout normal traffic
- Rule uses only surprising fields (not familiar ones like status=200)

### 009: Eigenvalue Spectrum Fingerprinting

**Goal**: Classify attack type from the eigenvalue spectrum shift alone.
Different attacks distort the subspace differently — the delta is a fingerprint.

**Approach**:
1. Snapshot eigenvalues before and after each attack type
2. Compute delta vector (after - before)
3. Characterize delta shape: concentrated (volumetric) vs spread (diversified)
4. Classify attack type by nearest delta fingerprint

**Success criteria**:
- DNS amp concentrates in PC1 (>40% of total delta)
- Attack deltas are distinguishable (mean between-type similarity < 0.95)
- Classification accuracy > 60% from eigenvalue delta alone

### 010: Adversarial Evasion

**Goal**: Test whether an attacker who knows the subspace can craft packets
that evade the residual detector while still carrying malicious field values.

**Approach**:
1. Naive attack (no evasion) — baseline detection
2. Minimal evasion — change only 2 fields from a normal template
3. Projection-aware evasion — send `reconstruct(attack_vec)` (zero residual by construction)
4. Check if attack field=value pairs survive projection (unbind + cosine match)
5. Blend strategies — mix original and projected at various ratios

**Success criteria**:
- Naive attack detected (100%)
- Projection evasion defeats residual detector (0% detection)
- Attack-exclusive fields (dst_port=53, path=dns, ttl=245) destroyed by projection
- Evasion is self-defeating: the anomalous component IS the attack

### 011: Real Traffic Intrinsic Dimensionality

**Goal**: Measure the intrinsic dimensionality of 19-field sidecar encodings
and find the optimal k for deployment.

**Approach**:
1. Generate 2000 mixed TCP/UDP packets using all 19 sidecar fields
2. Sweep k from 8 to 256, measure residual convergence
3. Analyze eigenvalue spectrum to find knee points
4. Compare 7-field vs 19-field intrinsic dimensionality
5. Test detection quality across k values

**Success criteria**:
- 19-field intrinsic dim > 7-field
- k=32 still achieves >90% TP
- Clear eigenvalue knee below 50% of max k
- Diminishing returns above the knee

### 012: Multi-Attack Separation via Subspace Peeling

**Goal**: When two attack types occur simultaneously, separate them using
fingerprint clustering and prototype subtraction ("peeling").

**Approach**:
1. Train on normal, then feed mixed DNS amp + credential stuffing
2. Extract anomalous components for all detected packets
3. Cluster by surprise fingerprint (k-means on 7D fingerprint vectors)
4. Build prototype of dominant cluster, subtract from other cluster
5. Verify separated fingerprints match known attack signatures

**Success criteria**:
- Both attack types detected (>90% each)
- Fingerprint clustering >80% accuracy
- Per-cluster fingerprints have distinct shapes

### 013: Subspace + Coherence Combined Detector

**Goal**: Evaluate whether fusing subspace residual with coherence
improves detection beyond either signal alone.

**Approach**:
1. Calibrate per-window thresholds from normal baseline (mean + 3σ)
2. Four scenarios: DNS amp, slow exfiltration, credential stuffing, stealth scan
3. Compare: residual-only, coherence-only, combined (OR)
4. Measure TP and FP for each detector configuration

**Success criteria**:
- Residual catches all attack types (>80% TP)
- Zero false positives across all detectors
- Combined never worse than best single signal
- Coherence contributes for volumetric attacks

### 014: Attack Manifold Capture

**Goal**: Prove we can train a *separate* subspace on attack-only traffic and it
converges to a meaningful manifold that discriminates same-type attacks from normal
traffic and other attack types.

**Approach**:
1. Train 4 attack subspaces (DNS amp, SYN flood, credential stuffing, exfiltration)
2. Cross-test each subspace against all traffic types
3. Compare eigenvalue spectra across attack types
4. Verify convergence via eigenvalue stability over training

**Success criteria**:
- Each attack subspace has lowest residual for its own type
- Normal traffic rejected (ratio > 1.5x)
- Eigenvalue spectrum converges (cos@200 vs @500 > 0.95)

### 015: Engram Library Matching

**Goal**: An EngramLibrary storing 4 attack types can correctly match new instances
to the right engram. Two-tier matching (eigenvalue pre-filter + full residual)
produces the same correct result as brute-force.

**Approach**:
1. Train 4 attack subspaces, mint engrams, add to library
2. Match fresh attack traffic → verify correct engram is top match
3. Verify normal traffic does not match well
4. Test eigenvalue spectrum pre-filter accuracy
5. Save/load round-trip preserves matching behavior

**Success criteria**:
- Overall matching accuracy > 90%
- Each attack type correctly matched > 80%
- Normal traffic has higher residual than attack traffic
- Spectrum pre-filter accuracy >= 75%
- Persistence round-trip preserves exact scores

### 016: Eager Activation (Few-Shot Matching)

**Goal**: Measure how many anomalous packets are needed before a confident
library match. If we can match within 1-5 packets, eager activation is practical.

**Approach**:
1. Build engram library from 4 attack types
2. Simulate attack onset, feed packets one at a time
3. Majority vote over sliding window of N packets
4. Sweep N=1,2,3,5,10,20,50
5. Measure false activation rate on normal traffic windows

**Success criteria**:
- Single-packet matching > 80%
- 5-packet window > 90%
- 10-packet window > 95%
- Zero false activations on normal traffic (N=10)

### 017: Variant Resilience

**Goal**: Same attack type returns with different parameters (different source IPs,
different target ports, different TTLs). The stored engram should still match because
the subspace captures structure, not specific values.

**Approach**:
1. Train engrams from "original" DNS amp and SYN flood attacks
2. Generate 5 variants of each with progressively more changes
3. Variant 1-3: single field change; Variant 4-5: multi-field and stress test
4. Match variants against library, compare residuals to original and normal

**Success criteria**:
- Baseline (original) matches 100%
- All variants match correctly (>80%)
- Worst variant still has lower residual than normal traffic

### 018: Full Lifecycle — Detection, Minting, Re-detection

**Goal**: Validate the complete engram lifecycle end-to-end: normal baseline →
first attack (unknown) → learn manifold → mint engram with EDN rule → calm period →
same attack returns → instant library match → stored rule deployed.

**Approach**:
1. Learn detection subspace from 1000 normal packets
2. First attack: detect anomalies, library miss, build attack subspace
3. Generate EDN rule from surprise fingerprint
4. Mint engram with rule + metadata
5. Calm period: verify zero false library hits
6. Second attack: library match within 1 packet, deploy stored rule

**Success criteria**:
- Detection catches first attack (>90%)
- Library miss on first encounter (correct — no prior engrams)
- Attack subspace converges (n > 100)
- EDN rule generated with constraints and actions
- Zero false library hits during calm
- Second attack matched in <= 2 packets
- Instant match rate > 90%

---

## What We're Building Toward

```
                    SCALAR SCORE          SUBSPACE ANALYSIS
                    ────────────          ─────────────────
Detection:          cosine drift          residual (off-manifold distance)
                    coherence             eigenvalue shift

Sensitivity:        single threshold      per-component thresholds
                                          (eigenvalue-weighted)

Attribution:        invert(diff_vec)      invert(anomalous_component)
                                          → what SPECIFICALLY doesn't belong

Visualization:      1D similarity plot    k-D projection plot
                                          (cluster structure visible)

Adaptation:         accumulator decay     amnesia (controlled forgetting)
                                          + growing k (new variance directions)
```

The subspace doesn't replace the accumulator pipeline — it complements it.
The accumulator says "this is different from normal." The subspace says
"this is outside the SPACE of all normal variations." When both agree,
confidence is high. When they disagree, the disagreement itself is
informative.

If this works in Python, the Rust port is straightforward: CCIPCA is just
dot products, scalar multiplications, and vector additions — the same
operations holon-rs already optimizes.

---

*Created: February 2026*
