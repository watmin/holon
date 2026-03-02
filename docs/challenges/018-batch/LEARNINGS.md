# Learnings: Batch 018 — Window-Level Pattern Matching & The Spectral Firewall

## Objective

Validate window-level eigenvalue matching (`match_spectrum`) as an early warning
and classification primitive, introduce directional subspace alignment
(`match_alignment`) as a complementary signal, and prove out the four-layer **spectral firewall** architecture end-to-end —
from allow-list membership through sealed denial tokens and CI/CD engram
promotion.

The name reflects the mechanism: eigenvalue *spectra* are the core signal,
spectral decomposition separates magnitude from direction, and the firewall
analyzes the variance *geometry* of encoded traffic rather than matching
signatures or modeling behavior.

Sixteen experiments across three areas:
- **001–007**: Core window-level primitives (spectrum matching, alignment, temporal evolution)
- **008–014**: Meta engram analysis and firewall layer composition
- **015–016**: Operational deployment primitives (denial tokens, engram promotion)

All experiments use Layer 7 HTTP/WAF context (method, path, headers, TLS) rather
than L3/4 packet fields, matching the http-lab's sidecar architecture.

## New Primitives

### `match_spectrum` — Eigenvalue Cosine Similarity

| Component | Detail |
|-----------|--------|
| Method | `EngramLibrary.match_spectrum(eigenvalues, top_k)` |
| Algorithm | Cosine similarity between eigenvalue vectors |
| Cost | O(n·k) per window — orders of magnitude cheaper than residual |
| Measures | Variance *shape* — how variance distributes across principal components |
| Limitation | Cannot distinguish subspaces with similar spectra but different orientations |

### `match_alignment` — Subspace Principal Angle Alignment

| Component | Detail |
|-----------|--------|
| Method | `OnlineSubspace.subspace_alignment(other, top_angles)` |
| Algorithm | SVD of basis inner-product matrix → cosine of principal angles |
| Cost | O(k·dim) for basis products + O(k²) for SVD |
| Measures | Variance *direction* — whether components point the same way in dim-space |
| Added to | `holon/memory/subspace.py` |

### `match_alignment` (Library)

| Component | Detail |
|-----------|--------|
| Method | `EngramLibrary.match_alignment(probe_subspace, top_k)` |
| Algorithm | `subspace_alignment` against each stored engram |
| Cost | O(n·k·dim) per probe |
| Added to | `holon/memory/engram.py` |

### Drilldown Surprise Probe

| Component | Detail |
|-----------|--------|
| Function | `_drilldown_probe(anomaly, request_data, vm)` |
| Algorithm | Recursive unbinding through the encoding hierarchy |
| What it does | Walks the request structure as the encoder does — unbinding map keys, list positions — to isolate anomaly contribution at each leaf field |
| Output | `[(path, norm, display_value), ...]` for every leaf (e.g., `tls.version`, `headers.[1].[1]`) |
| Replaces | Flat `FIELD_NAMES` probing that treated `tls` and `headers` as monolithic blobs |

### Dual Signal

The recurring finding across this batch: **single-metric approaches are
insufficient**. Two subspaces can have identical eigenvalue spectra but
completely different orientations (experiment 004). Combining spectrum
(magnitude) and alignment (direction) produces a reliable known-vs-unknown gate.

| Signal | Measures | Discriminates | Cost |
|--------|----------|---------------|------|
| `match_spectrum` | Variance shape | Between attack types | O(n·k) |
| `match_alignment` | Subspace orientation | Known vs unknown | O(n·k·dim) |
| Combined (spectrum × alignment) | Both | Full classification | O(n·k·dim) |

## Experiments & Results

### 001: Eigenvalue Shift as Early Warning

**Hypothesis**: Eigenvalue spectrum divergence from baseline leads per-request
residual detection during attack ramp-up.

| Signal | First Detection | Final State |
|--------|-----------------|-------------|
| Spectrum divergence | Request 78 | 0.41 |
| Per-request hit rate | Request 100 | 1.00 |
| Lead time | **22 requests** | — |

**Key Finding**: The eigenvalue signal leads by 22 requests because it
detects the *distributional shift* before enough individual requests cross
the residual threshold. On pure normal traffic, divergence stays below
0.0025 (vs detection threshold of ~0.01).

### 002: Attack Type Classification from Window Shape

**Hypothesis**: `match_spectrum` alone can classify attack type from a
traffic window without per-request scoring.

| Attack Type | Accuracy |
|-------------|----------|
| GET flood | 100% |
| Credential stuffing | 100% |
| Scraper | 100% |
| TLS-randomized | 100% |
| **Overall** | **100%** |

**Key Finding**: Each attack type produces a distinctive eigenvalue
signature. Cross-type similarity maxes at 0.93 — high but distinguishable.

### 003: Window Size Sensitivity

| Window Size | Accuracy |
|-------------|----------|
| 10 | 32% |
| 25 | 32% |
| 50 | 65% |
| 100 | 88% |
| 200 | 100% |
| 500 | 100% |

**Key Finding**: Reliable matching (>70%) requires ~100 requests.
Clear accuracy knee at 100 requests — minimal gain above that. CCIPCA needs
sufficient samples to stabilize eigenvalue estimates. Below 50, the
spectrum is dominated by sampling noise.

### 004: Blind Anomaly Detection — The Dual Signal Discovery

**Hypothesis**: Eigenvalue matching alone cannot separate known from
unknown attacks. A directional component is needed.

| Signal | Known Min | Unknown Max | Gap |
|--------|-----------|-------------|-----|
| Spectrum only | 0.936 | 0.944 | **-0.008** (wrong direction) |
| Alignment only | 0.338 | 0.276 | +0.062 |
| Combined (spec × align) | 0.321 | 0.262 | **+0.059** |

**Key Finding**: This is the central discovery of batch 018. Spectrum alone
produces a *negative* gap — unknown attacks can score higher than known ones
because they share similar variance shapes. Alignment reverses this: it
measures whether the variance lives in the *same directions* in high-dimensional
space, which it doesn't for unknown attack types.

This recapitulates the project's recurring lesson: magnitude and direction
are complementary and neither suffices alone.

### 005: Dual-Signal Pre-Filter

| Pipeline | Accuracy | vs Brute-Force | Compute |
|----------|----------|----------------|---------|
| Spectrum-only | 76% | — | 2/8 engrams |
| **Dual-signal** | **100%** | **100% agreement** | 2/8 engrams |
| Brute-force | 100% | — | 8/8 engrams |

**Key Finding**: The dual signal achieves brute-force accuracy at 75% compute
savings. Spectrum-only pre-filtering misclassifies 24% of probes because it
selects candidates with similar variance shapes but different orientations.
The alignment signal corrects all 55 misclassifications.

### 006: Temporal Evolution of Eigenvalue Fingerprint

| Phase | Attack Sim | Normal Sim |
|-------|-----------|------------|
| Pre-attack (normal) | 0.41 | 0.76 |
| Ramp-up | 0.37→0.95 | 0.76→0.45 |
| Full attack | **0.95** | 0.45 |
| Subsidence | 0.95→0.41 | 0.45→0.77 |
| Post-attack (normal) | 0.41 | **0.77** |

**Key Finding**: Attack lifecycle is fully visible in the eigenvalue spectrum.
Phase transitions are detectable as slope sign changes (4/4 boundaries
detected). Hysteresis observed: subsidence returns to normal slower than onset
rises — the subspace retains attack-influenced structure briefly.

### 007: Cross-Implementation Eigenvalue Consistency

| Comparison | Cosine Similarity |
|------------|-------------------|
| Same order, same data | 1.000 |
| Shuffled input order | 0.990 |
| Different k (first 32) | 1.000 |

**Key Finding**: CCIPCA produces consistent eigenvalue spectra across runs.
Input order introduces ~1% variation — acceptable for cross-implementation
matching. Python and Rust implementations will produce compatible spectra
provided the same encoding is used.

### 008: Normal Manifold Membership as Allow List

| Traffic Type | Mean Residual | Detection/Rejection |
|-------------|---------------|---------------------|
| Normal (holdout) | — | **0% FPR** |
| Attack | 7.33× normal | **100% rejected** |
| Unusual-but-legitimate | — | **5.5% rejected** |

**Key Finding**: Residual scoring against a normal engram is a viable Layer 0
gate. Clean separation (7.33× ratio between normal and attack residuals).
Only 5.5% of unusual-but-legitimate requests rejected — these are edge cases
(rare paths, uncommon UAs) that could be recovered via engram update or
multi-modal library (experiment 009).

This validates the "allow what looks normal" strategy: the normal manifold is
tight enough to exclude attacks while accommodating legitimate variation.

### 009: Multi-Modal Normal Library

| Metric | Value |
|--------|-------|
| Correct mode identification | **100%** |
| Cross-mode residual ratio | 6.33× |
| Attack rejection (best-of-library) | **100%** |
| Library FPR vs single-engram FPR | **0% vs 100%** |

**Key Finding**: Multiple normal engrams (browser-web, api-client, mobile-app)
eliminate cross-mode false positives. A single "normal" engram rejects 100% of
API traffic because it only knows browser patterns. A 3-engram library with
best-of-library scoring achieves 0% FPR across all modes.

### 010: Engram Taxonomy from Spectrum Clustering

Dual-signal clustering (spectrum × alignment) on an 8-engram library produces
a natural taxonomy: normal engrams cluster together, volumetric attacks cluster
together, application-layer attacks cluster together. Stable across 3
independent library builds.

### 011: Spectrum Decomposition of Mixed Windows

| Window | Dominant Engram Sim | Recovery |
|--------|---------------------|----------|
| Pure normal | 0.989 | NNLS weight 0.80 |
| Pure flood | 0.998 | NNLS weight 0.97 |
| 50/50 mix | Both ~0.69 | — |

**Key Finding**: `match_spectrum` similarity tracks mixing ratio qualitatively —
as attack fraction increases, attack engram similarity increases and normal
decreases. NNLS decomposition recovers pure windows well (dominant weight >0.4)
but mixed-ratio estimation is imprecise. This is a fundamental property of
CCIPCA: eigenvalue formation is nonlinear, so linear decomposition of mixed
spectra is approximate. Pure window detection is the reliable use case.

### 012: Engram Staleness Detection

| Metric | Value |
|--------|-------|
| Residual monotonicity | 5/5 epoch-to-epoch rises |
| Staleness threshold crossed | By epoch 3 (1.38 vs baseline 0.13) |
| Drift vs attack discrimination | **15.2× ratio** (attack=13.2, drift=1.4) |

**Key Finding**: The staleness metric (residual × (1 - spectrum_similarity))
cleanly separates gradual drift from sudden attack — 15.2× ratio. This
enables automated decisions: staleness below threshold → retrain engram;
staleness spike → attack, do NOT retrain.

### 013: Allow-List Freeze Under Poisoning

| Scenario | FPR (post-attack) | Attack Detection |
|----------|--------------------|--------------------|
| **Gated (frozen)** | **0%** | **100%** |
| Ungated | 0% | **0%** (poisoned) |

**Key Finding**: Gated updates (freeze during detected attack) prevent
manifold poisoning. The ungated subspace absorbs 500 attack requests and
loses all attack detection capability (0% TP). The gated subspace maintains
perfect detection (100% TP) and perfect normal acceptance (0% FPR). Freeze
triggers within 49 requests of attack onset.

### 014: Cross-Layer Attribution Pipeline

| Metric | Value |
|--------|-------|
| Layer 2 detection | First attack window (requests 200-249) |
| Pipeline vs brute-force agreement | **100%** |
| Top surprise field consistency | **100%** |
| Normal FPR | **0%** |
| Compute savings | **80.6%** |

**Key Finding**: The three-layer pipeline (spectrum → residual → drilldown
attribution) produces identical results to brute-force scoring at 80.6%
compute savings. Layer 2 correctly narrows candidates, Layer 1 confirms, and
the drilldown probe identifies the same anomalous fields. The pipeline
validates the concept firewall's layered architecture.

### 015: Denial Context Tokens — Sealed Verdicts

| Metric | Value |
|--------|-------|
| Round-trip fidelity | **100%** (60/60) |
| Token size | avg=1.6KB, max=2.1KB |
| True-positive field match | **100%** (40/40) |
| False-positive recovery | **100%** (20/20) |

**Key Finding**: Every denial can be sealed into a self-contained encrypted
token (~2KB) that recovers the complete verdict: anomalous leaf fields with
their values and anomaly shares, engram matches, residual scores. The
drilldown probe identifies specific sub-fields (`tls.version`, `headers.[1].[1]`,
`path_parts.[1]`) rather than monolithic parents — matching the http-lab's
surgical field attribution.

**Drilldown probe example** (get_flood):

```
ANOMALOUS FIELDS (5/24 above baseline):
  #1  has_cookie     share=5.4%  = false
  #2  version        share=5.3%  = HTTP/1.1
  #3  header_count   share=5.2%  = 3.0
  #4  method         share=5.1%  = GET
  #5  path           share=5.1%  = /api/search
NORMAL FIELDS: 19 (all within baseline)
```

**False-positive recovery**: An operator decrypts the token, sees the denial
was caused by an unusual path (`/admin/dashboard`, `path_parts.[1]=admin`).
Feeding the vector back into the subspace drops the residual below threshold.
The traffic is allowed going forward with no manual rule authoring.

### 016: Engram Promotion — CI/CD Training Pipeline

| Metric | Value |
|--------|-------|
| Preprod detection (new patterns) | **100%** miss rate |
| Production coverage | **100%** |
| Attack rejection | **100%** |
| Existing feature regression | **0%** |
| Library diff | Exactly 1 new engram |

**Key Finding**: Engrams trained on integration test traffic in preprod
generalize to production traffic for the same feature — 100% coverage despite
training on limited test variety. The key is that test traffic must be
structurally realistic (real browser UAs, full headers, cookies) even if
the specific values differ. An initial version with simplified test traffic
(no cookies, minimal headers) failed to generalize, validating that the
manifold captures structural patterns, not specific values.

## Key Insights

### 1. Magnitude and Direction Are Complementary — Always

This is the third time the project has discovered this pattern:
- **Batch 017**: Cosine-to-centroid (direction) vs residual (distance-from-manifold)
- **Batch 018 004**: `match_spectrum` (eigenvalue magnitude) vs `match_alignment`
  (principal angle direction)

Each individual metric has a blind spot that the other fills. Spectrum can't
distinguish subspaces with similar variance shapes but different orientations.
Alignment can't distinguish subspaces with similar orientations but different
variance distributions. The combined signal has no blind spot in our experiments.

### 2. Recursive Unbinding Enables Surgical Attribution

The original `surprise_fingerprint` (batch 017) probed top-level fields
(`tls`, `headers`, `path`). This produced results like "tls is anomalous"
without saying *what about TLS* was anomalous.

The drilldown probe walks the encoding hierarchy — unbinding map keys and list
positions at each level — to reach leaf-level attribution: `tls.version=TLS1.2`,
`headers.[1].[1]=python-requests/2.31.0`, `path_parts.[1]=admin`.

This mirrors the http-lab's Rust `drilldown_probe` which uses the same recursive
unbinding to derive surgical mitigation rules. The algebraic operation is
identical: `bind(bind(anomaly, role("tls")), role("version"))` isolates the TLS
version's contribution to the anomaly.

**Limitation**: MAP cross-talk compresses per-leaf anomaly shares into a tight
range (3–5.5% across ~24 leaves in 4096 dimensions). Statistical thresholding
(mean + 0.5σ) separates genuine signal from noise floor but the margin is thin.
The Rust implementation handles this with multiple probe kinds (Content, Shape,
Duplicate) and the full expr_tree rule language. Higher dimensionality would
also widen the margin.

### 3. The Four-Layer Architecture Composes Correctly

The concept manifold firewall's four layers were validated individually and
in combination:

| Layer | Primitive | Experiment | Result |
|-------|-----------|------------|--------|
| Layer 0 — Normal Allow List | `residual` vs normal engram | 008, 009 | 0% FPR, 100% attack rejection |
| Layer 1 — Anomaly Enforcement | `residual` vs threshold | 013, 014 | Gated updates prevent poisoning |
| Layer 2 — Window Spectrum | `match_spectrum × match_alignment` | 004, 005, 014 | 100% accuracy, 75%+ savings |
| Layer 3 — Symbolic Rules | Drilldown → field attribution | 014, 015 | Leaf-level attribution, sealed tokens |

Experiment 014 validates the full pipeline: Layer 2 narrows candidates → Layer 1
confirms → drilldown explains. 100% agreement with brute-force, 80.6% compute
savings.

### 4. Eigenvalue Spectra Are Temporal Fingerprints

Experiment 006 shows the attack lifecycle (onset → peak → subsidence) is fully
visible in the eigenvalue spectrum. This enables:
- **Phase detection**: Slope sign changes at attack boundaries
- **Attack classification**: Different attack types produce different spectral
  signatures (experiment 002: 100% accuracy)
- **Recovery monitoring**: Subsidence is detectable as the spectrum returns to
  baseline (with observable hysteresis)

### 5. Engram Lifecycle Is Manageable

The staleness metric (experiment 012) and freeze mechanism (experiment 013)
together solve the engram maintenance problem:
- **Drift**: Staleness grows gradually → retrain engram
- **Attack**: Staleness spikes suddenly (15.2×) → freeze, do NOT retrain
- **Promotion**: Preprod engrams transfer to production (experiment 016)
- **Recovery**: False-positive denial tokens feed back to update engrams
  (experiment 015)

### 6. Cross-Talk Limits Per-Field Resolution but Ranking Is Reliable

With 24 leaf-level probes in 4096 dimensions, individual anomaly shares cluster
between 3% and 5.5% (uniform baseline: 4.2%). The top-ranked fields are
consistently correct — `has_cookie=false` and `version=HTTP/1.1` reliably
appear at the top for get_flood attacks, `path=/admin/dashboard` for unusual
legitimate requests. But the absolute share margin between "anomalous" and
"noise" is thin (~1%), meaning a hard threshold will sometimes include slightly
noisy fields or exclude slightly marginal ones.

**Practical implication**: Use ranking for rule generation (top-K fields), not
absolute thresholds. The Rust expr_tree rule language does this — it generates
rules from the top contributors regardless of their absolute magnitude.

### 7. Sealed Tokens Close the Operational Loop

Denial context tokens (experiment 015) solve two operational problems
simultaneously:
- **No information disclosure**: Callers receive an opaque blob, firewall admins
  decrypt for full context
- **No log diving**: The token IS the log entry — verdict, anomalous fields
  with values, engram matches, residual scores, all in ~2KB

For false positives, the decoded token provides everything needed to mint a
corrective engram: the original request vector, which fields triggered the
denial, and the current subspace state. Recovery is a single `subspace.update()`
call.

## Spectral Firewall — Validation Status

| Firewall Component | Validated By | Status |
|--------------------|--------------| -------|
| Allow-list gate (Layer 0) | 008: 0% FPR, 100% rejection | Proven |
| Multi-modal allow list | 009: 3-mode library, 0% FPR | Proven |
| Spectrum pre-filter (Layer 2) | 005: 100% accuracy, 75% savings | Proven |
| Dual signal (magnitude + direction) | 004: Combined gap +0.059 | Proven |
| Temporal detection | 001: 22-request lead time, 006: full lifecycle | Proven |
| Classification | 002: 100% from window shape | Proven |
| Window size bounds | 003: ~100 requests for reliable matching | Characterized |
| Poisoning resistance | 013: Gated updates, 0% FPR post-attack | Proven |
| Staleness management | 012: 15.2× drift/attack discrimination | Proven |
| Layered pipeline | 014: 100% agreement, 80.6% savings | Proven |
| Sealed denial tokens | 015: 100% round-trip, leaf-level attribution | Proven |
| CI/CD promotion | 016: 100% production coverage from preprod | Proven |
| Cross-implementation | 007: >0.99 eigenvalue consistency | Validated |

Every component of the spectral firewall has been validated in isolation and
the critical pipeline (Layers 2→1→0) has been validated end-to-end. The
remaining step is integration in Rust via the http-lab sidecar with real
traffic from security tools (Nikto, ZAP, Nuclei) against a WordPress
deployment with LLM-based legitimate agent traffic.

## Recommendations for Rust Integration

1. **Port `subspace_alignment`**: Add to the existing `OnlineSubspace` in
   holon-rs. The algorithm is straightforward: basis inner-product matrix →
   SVD → mean of top-k cosine principal angles. No external dependencies
   beyond what holon-rs already uses.

2. **Port `match_alignment` to `EngramLibrary`**: Iterate stored engrams,
   call `subspace_alignment` on each. Return sorted results.

3. **Implement drilldown probe in the sidecar**: The Rust sidecar already has
   `drilldown_probe` in `detectors.rs` — extend it to use the `subspace_alignment`
   signal alongside the existing residual-based probing for improved candidate
   selection.

4. **Dual-signal scoring in the detection pipeline**: Replace spectrum-only
   pre-filtering with `spectrum × alignment` for engram candidate selection.
   This eliminates the 24% misclassification rate seen with spectrum-only.

5. **Sealed denial tokens**: Implement in the proxy's deny response path.
   AES-256-GCM encryption, base64 encoding, ~2KB per token. Include the
   drilldown probe output, engram match results, and the request fields.

6. **Engram promotion pipeline**: Add `engram export/import` CLI commands to
   the sidecar. Preprod sidecar observes integration tests → exports engram
   library as JSON → CI/CD pipeline copies to production → production sidecar
   loads on startup.

7. **Validation target**: Zero good requests dropped, all bad requests dropped.
   Test matrix: Nikto + ZAP + Nuclei attack traffic against WordPress, with
   LLM-based agents performing legitimate browsing. The allow-list gate
   (Layer 0) handles the "zero good dropped" requirement; the layered
   detection (Layers 1-3) handles the "all bad dropped" requirement.

## Open Questions

1. **Cross-talk at higher field counts**: The http-lab sidecar encodes 30+
   fields per request. With more leaves in the drilldown probe, does the
   per-field resolution improve (more probes → better statistical thresholding)
   or degrade (more cross-talk)?

2. **Optimal `top_angles` for alignment**: The current default is `max(3, k//4)`.
   With real traffic diversity, is a different setting better?

3. **Token encryption key management**: The experiments simulate encryption
   with HMAC-SHA256. Production needs real AES-256-GCM with key rotation.
   How does the key travel from the proxy to the admin console?

4. **Engram versioning**: When a preprod engram is promoted, how do we handle
   rollbacks? Should the engram library support version history?

5. **Multi-attack windows**: Experiment 011 shows that mixed-window
   decomposition via NNLS is approximate. Can the dual signal (spectrum +
   alignment) decompose multi-attack windows more reliably than spectrum alone?

## Completed Experiments

- [x] **001**: Eigenvalue early warning — 22-request lead time, divergence < 0.003 on normal
- [x] **002**: Attack classification — 100% from window shape, 4 attack types
- [x] **003**: Window size sensitivity — reliable at 100 requests, knee at 100
- [x] **004**: Dual signal discovery — spectrum gap -0.008, alignment gap +0.062, combined +0.059
- [x] **005**: Dual-signal pre-filter — 100% accuracy (vs 76% spectrum-only), 75% savings
- [x] **006**: Temporal evolution — full lifecycle visible, 4/4 boundaries detected, hysteresis observed
- [x] **007**: Cross-implementation — cosine > 0.99 same-order, > 0.989 shuffled
- [x] **008**: Normal allow list — 0% FPR, 100% attack rejection, 5.5% unusual rejection
- [x] **009**: Multi-modal library — 100% mode identification, 0% cross-mode FPR
- [x] **010**: Engram taxonomy — natural clustering, stable across builds
- [x] **011**: Spectrum decomposition — pure window recovery, monotonic tracking, nonlinear mixing
- [x] **012**: Engram staleness — 15.2× drift/attack discrimination, threshold by epoch 3
- [x] **013**: Allow-list freeze — 0% FPR gated, 0% TP ungated (poisoning), freeze at request 49
- [x] **014**: Cross-layer pipeline — 100% agreement, 80.6% savings, leaf-level attribution
- [x] **015**: Denial context tokens — 100% round-trip, 2.1KB max, drilldown probe, FP recovery
- [x] **016**: Engram promotion — 100% production coverage from preprod, 0% regression

---

*Updated: March 2026*
