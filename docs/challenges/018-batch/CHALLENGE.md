# Challenge Batch 018: Eigenvalue-as-Probe — Window-Level Pattern Matching

## Philosophy

**Batch 017 learned patterns from individual vectors. Batch 018 asks whether
we can match patterns from the *shape of a time window*.**

Every approach so far scores one vector at a time against a stored engram.
That works well for single-request detection and eager activation. But it
requires encoding each request, computing a residual against each engram, and
acting on the result — per request, per engram.

There's a different question you can ask: instead of "does this request look
like a known attack?", ask "does this *30-second window of traffic* look like
one?" If the shape of the traffic distribution over a window resembles the
eigenvalue fingerprint of a known pattern, you can detect it without scoring
individual requests at all.

The mechanism: `match_spectrum`. It compares the eigenvalue signature of a
probe subspace — trained over a live window — against the stored eigenvalue
signatures in the library using cosine similarity. No full residual computation.
No per-request encoding in the matching step. Just shape-to-shape comparison.

This opens three new capabilities:

1. **Early warning**: Attack patterns distort the eigenvalue spectrum before
   enough requests accumulate for per-request matching. Eigenvalue shift can
   be the *first* signal, not a confirmation signal.

2. **Blind detection**: You can detect that *something is wrong* without
   knowing which requests are anomalous — the window as a whole has an unusual
   shape, even if individual requests pass residual thresholds.

3. **Temporal fingerprinting**: Different attack types produce different
   eigenvalue distortions. A GET flood has a different spectrum shift than
   credential stuffing. The shape of a window can classify the attack type
   before per-request analysis confirms it.

### HTTP/WAF Context

These experiments simulate traffic at Layer 7 — HTTP requests with TLS
context — matching the http-lab's WAF architecture. The field set reflects
what a TLS-terminating reverse proxy observes:

**Request fields**: `method`, `path`, `http_version`, `user_agent`, `host`,
`content_type`, `header_count`, `has_cookie`

**TLS fields**: `tls_version`, `tls_cipher`, `tls_ext_types`

**Attack types** (from http-lab generator profiles):
- **GET flood**: high-volume GET /api/search, curl/bot UA, TLS 1.2,
  limited cipher suites
- **Credential stuffing**: POST /api/v1/auth/login, python-requests UA,
  repeated paths, 401 status
- **Scraper**: GET /products/{random}, Scrapy UA, high path cardinality
- **TLS-randomized flood**: GET with shuffled cipher ordering, bot UA,
  designed to evade TLS fingerprinting

**Normal traffic**: browser UAs (Chrome, Firefox), TLS 1.3, varied paths,
cookies present, standard header ordering.

## The Primitives

### `match_spectrum` — Magnitude Signal

```python
matches = library.match_spectrum(probe_eigenvalues, top_k=5)
# → [(name, cosine_similarity), ...] sorted descending
```

**What it computes**: Cosine similarity between the probe's eigenvalue vector
and each stored engram's eigenvalue signature. Measures variance *shape* —
how variance distributes across principal components.

**What it skips**: Full residual computation. No per-request scoring. The
matching cost is O(n·k) where n is the number of engrams and k is the number
of eigenvalue components — orders of magnitude cheaper than full residual
scoring at O(n·k·dim).

**Limitation**: Two subspaces can have identical eigenvalue spectra but
completely different principal component orientations. Spectrum alone cannot
reliably separate known from unknown traffic types (see experiment 004).

### `match_alignment` — Directional Signal

```python
matches = library.match_alignment(probe_subspace, top_k=5)
# → [(name, alignment_score), ...] sorted descending
```

**What it computes**: Subspace alignment via principal angles (SVD of the
basis inner-product matrix). Measures whether variance lives in the *same
directions* in dim-space, not just the same amounts.

**Cost**: O(n·k·dim) for basis products + O(n·k²) for SVDs. More expensive
than `match_spectrum` but still much cheaper than full residual scoring.

### Dual Signal

The two primitives are complementary:

| Signal | Measures | Discriminates | Cost |
|--------|----------|---------------|------|
| `match_spectrum` | Variance shape (magnitude) | Between attack types | O(n·k) |
| `match_alignment` | Subspace orientation (direction) | Known vs unknown | O(n·k·dim) |

Combined scoring (`spectrum × alignment`) requires BOTH magnitude and
directional agreement to classify traffic as "known."

**How to get a probe**: Train a short-window `OnlineSubspace` over a sliding
window of recent traffic.

```python
window_sub = OnlineSubspace(dim=4096, k=64)
for request in recent_window:
    window_sub.update(encode(request))

# Magnitude: eigenvalue shape matching
spec_matches = library.match_spectrum(window_sub.eigenvalues, top_k=3)

# Direction: principal angle alignment
align_matches = library.match_alignment(window_sub, top_k=3)
```

## Experiments

### 001: Eigenvalue Shift as Early Warning

**Goal**: Demonstrate that eigenvalue spectrum divergence from baseline
occurs earlier in an attack than per-request residual hits. The divergence
signal leads; the residual signal confirms.

The signal is **differential**: we track how much the live window's
eigenvalue spectrum diverges from the stored baseline spectrum. Attack
traffic shifts the variance structure away from normal. Raw eigenvalue
similarity can't discriminate (normal and attack may have similar variance
shapes), but the *divergence from a known baseline* does.

**Approach**:
1. Train a baseline subspace on 500 normal HTTP requests (browser traffic)
2. Store the baseline eigenvalue spectrum as the reference
3. Begin a slow-ramp GET flood (traffic mixed 90/10 normal/attack, shifting
   to 50/50 over 200 requests, then 100% attack)
4. Track two signals over time:
   - Spectrum divergence: `1 - cosine_sim(window_eigenvalues, baseline_eigenvalues)`
   - Per-request residual hit rate (fraction exceeding threshold)
5. Identify the request number at which each signal first exceeds its threshold

**Success criteria**:
- Divergence signal exceeds its threshold at least 20 requests before
  per-request hit rate exceeds 50%
- Both signals are elevated (trending toward detection) by request 100
- On pure normal traffic: divergence stays below the detection threshold

---

### 002: Attack Type Classification from Window Shape Alone

**Goal**: Classify attack type using only the eigenvalue spectrum of a
traffic window — no per-request scoring, no residual computation.

**Approach**:
1. Build an engram library from 4 attack types:
   - GET flood (high volume, curl UA, TLS 1.2, uniform path /api/search)
   - Credential stuffing (POST /api/v1/auth/login, python-requests UA, 401s)
   - Scraper (GET /products/{random}, Scrapy UA, high path cardinality)
   - TLS-randomized flood (shuffled cipher ordering, bot UA)
2. For each attack type, generate 10 fresh traffic windows (200 requests each)
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

**Goal**: Find the minimum window size (number of requests) at which
eigenvalue matching becomes reliable.

**Approach**:
1. Use the 4-engram library from experiment 002
2. For each attack type, sweep window sizes: 10, 25, 50, 100, 200, 500 requests
3. At each window size, train a subspace and run `match_spectrum`
4. Measure classification accuracy vs window size

**Success criteria**:
- Reliable matching (>70% accuracy) achievable in ≤ 100 requests
- Clear accuracy knee: minimal gain above some window size threshold
- Accuracy curve is monotonically non-decreasing with window size

---

### 004: Blind Anomaly Detection — Dual Signal (Spectrum + Alignment)

**Goal**: Demonstrate that eigenvalue spectrum matching alone cannot
separate known from unknown attacks (the magnitude problem), but adding
a **directional** signal — subspace alignment via principal angles —
provides the missing discriminator.

This recapitulates the project's recurring finding: single-metric
approaches are insufficient. `match_spectrum` measures variance *shape*
(how much variance on each axis). `match_alignment` measures variance
*direction* (which directions in dim-space those axes point). Together
they compose into a reliable known-vs-unknown gate.

**Approach**:
1. Build library from 3 known attack types (GET flood, credential
   stuffing, scraper)
2. Generate windows containing a 4th type not in the library
   (TLS-randomized flood)
3. Also generate: known attack types and 50/50 normal+unknown mixes
4. For each window, record three signals:
   - **Spectrum**: max eigenvalue cosine similarity (magnitude)
   - **Alignment**: max principal-angle alignment score (direction)
   - **Combined**: spectrum × alignment (both required for "known")
5. Compare gap between known minimum and unknown maximum for each signal

**Success criteria**:
- Spectrum alone has near-zero or negative gap (motivation for dual signal)
- Alignment provides positive separation where spectrum doesn't
- Alignment gap exceeds 3× the spectrum gap (meaningful improvement)
- Combined signal correctly orders known > unknown
- Combined gap is positive and actionable (> 0.02)

---

### 005: Dual-Signal Pre-Filter for Per-Request Scoring

**Goal**: Use the dual signal (spectrum × alignment) to identify the most
likely engram before running full residual scoring. Compare against
spectrum-only pre-filtering to show the directional signal improves
candidate selection. Measure accuracy and compute cost vs brute-force.

**Approach**:
1. Build a library of 8 engrams (4 attack types × 2 parameter variants each)
2. For each probe window, run three pipelines:
   a. **Spectrum-only**: `match_spectrum` → top-2 → residual
   b. **Dual-signal**: `match_spectrum × match_alignment` → top-2 → residual
   c. **Brute-force**: residual against all 8
3. Compare: does each pre-filter pick the same best-match as brute-force?
4. Measure: accuracy of dual vs spectrum-only vs brute-force

**Success criteria**:
- Dual-signal accuracy ≥ 95% of brute-force
- Dual-signal accuracy ≥ spectrum-only accuracy
- Dual-signal agreement with brute-force > 90%
- 75% compute reduction (scoring 2 instead of 8 engrams per request)

---

### 006: Temporal Evolution of Eigenvalue Fingerprint

**Goal**: Characterize how the eigenvalue spectrum evolves during an attack
lifecycle — onset, peak, and subsidence. Different phases should produce
different spectra; the evolution itself is a temporal fingerprint.

**Approach**:
1. Simulate a complete attack lifecycle:
   - Phase 0: 200 normal requests (browser traffic)
   - Phase 1: attack ramp-up (100 requests, 20%→80% GET flood)
   - Phase 2: full attack (200 requests, 100% GET flood)
   - Phase 3: subsidence (100 requests, 80%→0% GET flood)
   - Phase 4: 200 normal requests (recovery)
2. Every 25 requests, train a fresh 50-request window subspace, extract
   eigenvalues
3. Plot cosine similarity of each window eigenvalue vector to:
   - The stored attack engram
   - The normal baseline engram

**Success criteria**:
- Similarity to attack engram peaks during phase 2, near 0 in phases 0
  and 4
- Similarity to normal engram inversely tracks attack similarity
- Phase transitions (onset, subsidence) are detectable as slope changes
- Clear hysteresis: subsidence phase returns to normal slower than onset
  rises

---

### 007: Cross-Implementation Eigenvalue Consistency

**Goal**: Verify that eigenvalue signatures produced by the Python
`OnlineSubspace` and the Rust `OnlineSubspace` (holon-rs) are comparable
— i.e., that `match_spectrum` can work across implementations if needed.

**Note**: This is informational. We expect the eigenvalue spectra to be
similar in shape (same algorithm, same data) but not bitwise identical
(different floating-point order of operations). The question is whether
the cosine similarity between Python and Rust eigenvalue vectors is high
enough to be useful.

**Approach**:
1. Generate 500 normal HTTP requests; encode with Python holon
2. Train Python `OnlineSubspace`, extract eigenvalues
3. Encode the same 500 requests with holon-rs (same schema, same seed)
4. Train Rust `OnlineSubspace`, extract eigenvalues
5. Compute cosine similarity between Python and Rust eigenvalue vectors

**Success criteria**:
- Cosine similarity between Python and Rust eigenvalue vectors > 0.95
- If < 0.95: document the divergence, understand whether it's algorithmic
  or numerical, and note it as a cross-implementation constraint

---

## Meta Engram Analysis — Toward the Concept Manifold Firewall

Experiments 001–007 validate `match_spectrum` as a window-level detection
primitive. But the concept manifold firewall (see
`http-lab/docs/CONCEPT-MANIFOLD-FIREWALL.md`) requires more: the engram
library itself becomes a first-class object of analysis, and the allow-list
defense (Layer 0) needs proving out alongside the anomaly layers.

The following experiments bridge that gap. They address three areas that
001–007 leave untouched:

1. **Layer 0 — Normal Allow List**: Can manifold membership serve as a
   pass/fail gate for legitimate traffic? What are the FPR/FNR
   characteristics?

2. **Meta Engram Analysis**: What can you learn by analyzing the engram
   library itself — relationships between engrams, coverage gaps, staleness,
   decomposition of mixed signals?

3. **Layer Composition**: Do the four firewall layers compose correctly?
   Does information flow from spectrum matching → residual scoring →
   field attribution as the concept doc describes?

### 008: Normal Manifold Membership as Allow List

**Goal**: Use residual scoring against *normal* engrams as a pass/fail gate.
Characterize the false-positive and false-negative rates. This is the
foundation for Layer 0 of the concept firewall — "allow what looks normal."

**Approach**:
1. Train a normal engram from 1000 requests of typical browser traffic
2. Mint it into the library as a "normal" engram
3. Score 500 new normal requests (should pass — low residual)
4. Score 500 attack requests of various types (should fail — high residual)
5. Score 200 "unusual but legitimate" requests (edge cases — rare paths,
   uncommon user agents, but structurally valid browser requests)
6. Sweep the membership threshold; plot FPR vs FNR curve

**Success criteria**:
- At some threshold: FPR < 5% AND FNR < 10%
- Attack requests: > 90% rejected (residual above threshold)
- Legitimate-but-unusual requests: < 30% rejected (most should pass)
- Clear separation between normal and attack residual distributions

---

### 009: Multi-Modal Normal Library

**Goal**: Build multiple normal engrams for different traffic modes
(endpoint patterns, client types) and show that best-match scoring across
the library correctly identifies which mode an incoming request belongs to.

**Approach**:
1. Train 3 normal engrams:
   - "browser-web": Chrome/Firefox UAs, TLS 1.3, varied paths, cookies
   - "api-client": application/json, Authorization headers, /api/v2 paths
   - "mobile-app": mobile UAs, TLS 1.3, /api/mobile paths, compact headers
2. Mint all three into the library
3. For each mode, generate 200 test requests and score against all 3 engrams
4. The correct engram should have the lowest residual for its own traffic
5. Score attack traffic — all 3 normal engrams should reject it

**Success criteria**:
- Correct mode identified (lowest residual) > 80% of the time
- Cross-mode residuals are measurably higher than same-mode residuals
- Attack traffic rejected by all 3 normal engrams (residual > threshold
  for all)
- A "best-of-library" scoring function works: `min(residual across
  normal engrams)`

---

### 010: Engram Taxonomy from Spectrum Clustering

**Goal**: Given an engram library containing both normal and attack engrams,
cluster them by eigenvalue spectrum similarity. Show that a natural taxonomy
emerges — normal engrams cluster together, attack subtypes cluster by
similarity, and the clustering is stable.

**Approach**:
1. Build a library with 8+ engrams:
   - 3 normal engrams (from experiment 009's modes)
   - 5 attack engrams (GET flood, credential stuffing, scraper,
     TLS-randomized flood, POST flood)
2. Compute pairwise `match_spectrum` similarity between all engrams
3. Hierarchical cluster the similarity matrix
4. Verify: normal engrams form one cluster, attacks form another
5. Within the attack cluster: volumetric attacks (GET flood, POST flood)
   should cluster separately from application-layer attacks (credential
   stuffing, scraper)

**Success criteria**:
- Normal vs attack clusters have < 0.5 mean inter-cluster similarity
- Intra-cluster similarity > 0.6 for both normal and attack groups
- Volumetric attacks cluster together (similarity > 0.5)
- Application-layer attacks cluster together (similarity > 0.5)
- Clustering is stable across 3 independent library builds (same structure)

---

### 011: Spectrum Decomposition of Mixed Windows

**Goal**: When a traffic window contains a mix of two patterns (e.g.,
70% normal + 30% attack, or two attack types overlapping), decompose
the window's eigenvalue spectrum into weighted contributions from known
engrams.

**Approach**:
1. Build a library with 3 engrams: normal, GET flood, credential stuffing
2. Generate mixed windows at known ratios:
   - 100% normal, 80/20 normal/flood, 50/50 normal/flood, 20/80
     normal/flood, 100% flood, 50/50 flood/cred_stuff
3. For each window, run `match_spectrum` and record similarity to each engram
4. Fit: can the similarity scores recover the mixing ratio?
5. Test with non-negative least squares:
   `eigenvalues ≈ Σ wᵢ · engram_eigenvaluesᵢ`

**Success criteria**:
- Monotonic relationship: as attack fraction increases, attack engram
  similarity increases and normal similarity decreases
- At 50/50 mix: both engram similarities within 0.2 of each other
- NNLS decomposition recovers mixing ratios within ±15%
- Pure windows: dominant engram similarity > 0.7, others < 0.5

---

### 012: Engram Staleness Detection

**Goal**: Measure how an engram's match quality degrades as the underlying
traffic distribution drifts. Define a staleness metric that signals when
an engram should be refreshed.

**Approach**:
1. Train a normal engram on "epoch 0" traffic (specific browser versions,
   paths, TLS profiles)
2. Evolve the traffic through 5 epochs, each shifting one aspect:
   - Epoch 1: new paths added (/api/v2 alongside /api/v1)
   - Epoch 2: user agent distribution shifts (new Chrome version)
   - Epoch 3: TLS profile changes (new cipher suites)
   - Epoch 4: new content types (application/graphql)
   - Epoch 5: multiple simultaneous shifts
3. At each epoch, score 200 requests against the original engram
4. Track: mean residual, fraction above threshold, match_spectrum
   similarity between window eigenvalues and the original engram

**Success criteria**:
- Residual increases monotonically across epochs
- match_spectrum similarity to original engram decreases monotonically
- Staleness metric (residual × (1 - spectrum_similarity)) crosses a
  detectable threshold by epoch 3
- The staleness metric correctly distinguishes "drift" from "attack"
  (drift is gradual, attack is sudden)

---

### 013: Allow-List Freeze Under Poisoning

**Goal**: Demonstrate that freezing normal engrams during detected attacks
prevents the allow list from being poisoned. Without freeze, attack traffic
shifts the normal manifold; with freeze, the manifold stays clean.

**Approach**:
1. Train a normal engram on 1000 clean browser requests
2. Simulate an attack scenario (500 attack requests mixed with 500 normal)
3. Path A (unfrozen): continue updating the normal subspace during attack
4. Path B (frozen): stop updates when anomaly rate exceeds a trigger
   threshold
5. After the attack, score 500 clean normal requests against both versions
6. The frozen version should still correctly pass normal traffic; the
   unfrozen version may have drifted

**Success criteria**:
- Unfrozen engram: FPR on post-attack normal traffic > 10% (it now
  thinks some normal traffic is anomalous because the manifold shifted)
- Frozen engram: FPR on post-attack normal traffic < 5% (manifold
  unchanged, still recognizes normal)
- Both detect attack traffic during the attack (residual > threshold)
- Freeze trigger fires within the first 50 attack requests

---

### 014: Cross-Layer Attribution Pipeline

**Goal**: Wire up three layers — spectrum matching (Layer 2) → residual
scoring (Layer 1) → field attribution (Layer 0/drilldown) — and show
that information flows correctly. Layer 2 narrows the candidate set,
Layer 1 confirms, attribution explains.

**Approach**:
1. Build a library with 4 attack engrams and 2 normal engrams
2. Simulate a traffic stream: 500 normal → 200 attack → 300 normal
3. Every 50 requests, run Layer 2: `match_spectrum` on the window
   - If attack spectrum detected: record candidate engram(s)
4. For individual requests flagged by Layer 1 (residual > threshold):
   - Score only against Layer 2's candidate engrams (not full library)
   - Run `surprise_fingerprint` for field-level attribution
5. Measure: does the pipeline produce the same final answer as brute-force
   scoring against all engrams? Is the attribution consistent?

**Success criteria**:
- Layer 2 correctly identifies the attack type within the first 50
  attack requests (before per-request scoring accumulates enough hits)
- Pipeline final answer matches brute-force in > 95% of cases
- Attribution (top surprise field) is consistent between pipeline and
  brute-force scoring
- Compute cost reduced by > 50% vs brute-force (fewer engrams scored
  per request)

---

## Operational Primitives — Deployment-Ready Firewall Foundations

The firewall isn't just a detection system — it's an operational system.
Experiments 015–016 prove two capabilities that are prerequisites for
real deployment: explainable denial without information disclosure, and
engram promotion through CI/CD pipelines.

### 015: Denial Context Tokens — Sealed Verdicts

**Goal**: When the firewall denies a request, produce a self-contained,
cryptographically sealed token that encodes the *complete* denial reason
as data — the surprise fingerprint, residual score, matched engrams,
field attribution, raw fields — everything. The caller receives an opaque
blob (no information disclosure). A firewall admin decrypts the token for
instant, full-context explainability without a log dive. Critically: the
decoded context is actionable — if the denial is a false positive, the
admin can feed it back to mint a corrective engram that allows the traffic.

**Approach**:
1. Build a trained allow-list firewall (normal engram + detection subspace)
2. Generate traffic that gets denied (both true attacks and false positives)
3. For each denial, build a denial context structure:
   - `residual_score`: the anomaly score that triggered denial
   - `threshold`: the active threshold at denial time
   - `top_engram_matches`: [(name, score), ...] from library matching
   - `surprise_fingerprint`: per-field attribution {field: magnitude}
   - `request_fields`: the raw fields of the denied request
   - `timestamp`: when the denial occurred
   - `engram_eigenvalues`: the probe window's eigenvalue snapshot
4. Serialize to JSON, encrypt with AES-256-GCM, base64-encode
5. Verify round-trip: decrypt → deserialize → all fields recovered exactly
6. For false-positive denials: use the decoded context to encode the
   request, update the normal engram, and verify the same request now passes
7. For true-positive denials: verify the decoded context correctly
   identifies the attack type and top contributing fields

**Success criteria**:
- Round-trip fidelity: decoded token recovers 100% of denial context
  (all fields, scores, and attribution intact after decrypt)
- Zero information leakage: token is indistinguishable from random bytes
  (no plaintext fragments, no structure visible in base64)
- False-positive recovery: corrective engram minted from decoded context
  allows the originally-denied traffic pattern (residual drops below
  threshold)
- True-positive clarity: decoded context's top surprise field matches the
  field that actually differs between attack and normal traffic
- Token size: < 4KB base64 for a typical denial (practical for HTTP
  response headers)

---

### 016: Engram Promotion — CI/CD Training Pipeline

**Goal**: Prove that engrams can be learned from integration/manual test
traffic in a pre-production environment and deployed as artifacts alongside
application code. On Day 1 in production, the firewall already knows what
"normal" looks like for the new feature — no cold-start learning period,
no blind window where novel legitimate patterns get denied.

**Approach**:
1. Simulate a "preprod" environment:
   - Existing baseline: normal engrams for /api/users, /api/search (v1)
   - New feature: /api/v2/recommendations endpoint with new request shape
2. Run "integration tests" in preprod: 200 structured test requests to
   the new endpoint (limited variety, controlled parameters)
3. The preprod sidecar observes the test flows, detects new patterns
   (not in existing engram library), and mints a new engram
4. "Deploy" to production: load the preprod engram library
5. Score production-like traffic for the new endpoint:
   - 500 varied production requests (realistic diversity, not just test
     patterns)
   - 200 attack requests targeting the new endpoint
6. Measure: does the preprod engram correctly allow diverse production
   traffic for the same feature? Does it still reject attacks?
7. Engram diff: compare the pre-deploy and post-deploy libraries, verify
   the diff corresponds exactly to the new feature

**Success criteria**:
- Preprod engram detects new patterns: the new endpoint's traffic does not
  match existing engrams (library miss → new engram minted)
- Production coverage: preprod engram allows > 90% of production traffic
  for the new endpoint (despite only training on test flows)
- Attack rejection: preprod engram does NOT accidentally allow attack
  traffic targeting the new endpoint (residual > threshold for attacks)
- Existing features unaffected: traffic for /api/users and /api/search
  still matches the original engrams (no regression)
- Engram diff is meaningful: exactly 1 new engram in the library,
  corresponding to exactly 1 new feature
- Staleness signal: if the production traffic drifts significantly from
  the test-trained engram, the staleness metric (from experiment 012)
  detects it and signals for retraining

---

## What We're Building Toward — The Spectral Firewall

```
                    PER-REQUEST SCORING       WINDOW-LEVEL SCORING
                    ───────────────────       ────────────────────
Mechanism:          residual(vec)             match_spectrum(eigenvalues)
                    vs each engram            vs each engram signature

Cost:               O(n·k·dim) per request    O(n·k) per window

Signal timing:      confirmation              early warning

Granularity:        which request             which window

Use case:           single-request triggers   trend detection
                    eager activation          temporal fingerprinting
                    field attribution         blind anomaly detection

Combined:           match_spectrum identifies candidate engram →
                    per-request residual scoring against candidate only →
                    field attribution on confirmed anomalous requests
```

Window-level and request-level scoring are complementary layers. Window
matching is cheap, early, and operates without per-request labels. Request
matching is more expensive, later, and provides field-level attribution.
The ideal pipeline runs both: the window signal narrows the candidate engram,
the request signal confirms and attributes.

The meta engram experiments (008–014) extend this from "use engrams" to
"reason about engrams." The allow-list experiments prove that normal
manifold membership is a viable primary defense. The taxonomy and staleness
experiments prove that the library itself can be managed algebraically.
The pipeline experiment proves the layers compose.

The operational experiments (015–016) prove the system is deployable.
Sealed denial tokens (015) give operators instant explainability and
false-positive recovery without information disclosure to callers.
Engram promotion (016) eliminates the cold-start problem by baking
learned traffic patterns into the deployment pipeline. Together with
001–014, this batch validates every primitive needed for the concept
manifold firewall — from detection algebra through operational deployment.

---

*Created: February 2026*
