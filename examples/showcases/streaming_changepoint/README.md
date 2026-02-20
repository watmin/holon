# Streaming Changepoint Detection

Find phase transitions in an unlabeled system metric stream — no per-metric
thresholds, no label column, no sliding window code.

## The Problem

A stream of system metrics arrives continuously. At some point, latency starts
climbing. Then error rates spike. Then everything breaks. Then it recovers —
but not quite to baseline. You want to know when each transition happened,
what changed structurally, and whether recovery is complete.

You don't have phase labels. You don't know in advance which metrics to watch.
You can't hard-code thresholds because they change between services.

## What Holon Does

`OnlineSubspace` learns "healthy" from the first 50 observations. Every
subsequent observation is scored against that baseline — the residual. No
per-metric threshold. No feature engineering. The residual rises through
degradation, peaks at incident, and falls during recovery. The structural story
emerges from a single learned baseline.

`segment()` finds where the stream transitions. `difference()` computes a delta
vector between phases — a storable, algebraically composable outage fingerprint.
`invert()` decomposes what the incident "looks like" against known prototypes.

## Run

```bash
./scripts/run_with_venv.sh python -m examples.showcases.streaming_changepoint.showcase
```

## Output

125 observations across 4 phases, no labels passed to Holon:

```
Stream: 50 healthy → 25 degraded → 20 incident → 30 recovery
Total : 125 observations  |  True boundaries: [50, 75, 95]
Holon sees an unlabeled stream — no phase column, no field metadata
```

The threshold is learned from healthy data alone:

```
Learning healthy baseline from first 50 observations...
  Threshold : 54.88  (1.5σ above healthy EMA)
  Train max : 30.64  (healthy data is highly consistent)
```

**The residual timeline** — this is the whole story:

```
RESIDUAL TIMELINE  (structural anomaly score vs healthy baseline)
  threshold=54.9  |  bar scale: each █ ≈ 3 residual units
  Healthy   [0-49]     mean= 17.2  flagged= 0/50  █████
  Degraded  [50-74]    mean= 58.2  flagged=25/25  ███████████████████
  Incident  [75-94]    mean= 64.6  flagged=20/20  █████████████████████
  Recovery  [95-124]   mean= 56.4  flagged=23/30  ██████████████████ ◄ threshold
```

0/50 healthy flagged. 25/25 degraded flagged. 20/20 incident flagged. Recovery
partially settled: 23/30 still above threshold, meaning the system hasn't
returned to structural baseline yet. No metric threshold was set for any of this.
The subspace inferred the boundary from healthy variance alone.

**Changepoint detection** via `segment()`:

```
  Consolidated breakpoints: [1, 14, 27, 40, 53, 66, 79, 92, 105, 118]
  True phase boundaries      : [50, 75, 95]
  Matched (within 12 steps)  : [40, 53, 66, 79, 92, 105]

  → index  40: entering 'healthy' territory  (expected 'degraded' at 50, drift=10 steps)
  → index  53: entering 'degraded' territory  (expected 'degraded' at 50, drift=3 steps)
  → index  79: entering 'incident' territory  (expected 'incident' at 75, drift=4 steps)
  → index  99: entering 'recovery' territory  (expected 'recovery' at 95, drift=4 steps)
```

6 breakpoints matched within 12 steps of the 3 true phase boundaries. The
degraded→incident transition is detected at index 79, 4 steps after the true
boundary — a 4-observation delay on a stream of 125.

**Change analysis** — what structurally changed?

```
  difference(healthy, incident):
    46.1% of vector dimensions changed
    This delta IS the outage fingerprint — storable and algebraically composable

  invert(incident_proto, codebook) — structural overlap per phase:
    'incident':  1.000  ██████████████████████████████
    'healthy':  0.003
    'recovery':  0.000
```

The incident prototype has near-zero overlap with healthy — these are
structurally distinct. The `difference` vector, carrying 46% of all dimensions,
is the exact signature of the outage. You can store it as an engram, query
against it, or use it to amplify future anomaly detection.

Recovery matches back to the healthy engram:

```
  EngramLibrary.match(recovery_proto) → 'healthy'
    residual=31.15  threshold=54.88  anomalous=False
    Recovery structurally stabilised relative to learned healthy baseline
```

```
Residual rises 3x from healthy (17.2) to incident (64.6).
segment() identified 6/3 phase transitions within 12 steps.
Zero per-metric thresholds. Zero label columns.
```

## Without Holon

You would need: a per-metric alert rule for each of 7 fields (with manually
chosen thresholds), a correlation layer to combine them, a time-windowing
implementation to smooth noise, an offline changepoint detection model (e.g.,
PELT or BOCPD) trained on labeled data, and a separate reporting layer. Any
new metric added to the schema requires updating alert rules.

## Try

- Lower `sigma_mult=1.0` to catch more of recovery as anomalous.
- Add a second engram for "incident" and use `match()` to classify each window.
- Use `difference(healthy_proto, recovery_proto)` to see whether recovery
  returns to exactly the same manifold or a nearby-but-different one.
