# Config Drift Remediation

Detect config drift, name the exact changed field, and produce a corrective
vector — on arbitrary nested JSON with no schema annotations.

## The Problem

Your configs are validated by schema. All types are correct, all required fields
are present. But `pool_size` drifted from 15 to 8. Or someone swapped `db.host`
to an external address. Or two fields changed at once. Schema validation passes
all of these. Structural memory doesn't.

## What Holon Does

`OnlineSubspace` learns the golden manifold from 20 stable config versions
streamed 10 times each. Multi-pass training tightens the subspace until training
residuals approach zero. Any config that doesn't fit gets a residual above the
learned threshold. Attribution is residual-swap over every nested leaf field:
revert each field to golden, measure the residual drop, largest drop names the
culprit. `difference` + `amplify` produce a corrective vector. Verification
encodes the actual correct config and confirms it scores below threshold.

## Run

```bash
./scripts/run_with_venv.sh python -m examples.showcases.config_drift_remediation.showcase
```

## Output

Training converges the subspace to near-zero residuals on all 20 stable configs:

```
Learning golden config manifold (20 configs × 10 passes)...
  Threshold  : 15.34
  Train mean : 0.0282  max: 0.0612  (near-zero = tight convergence)
```

A training max of 0.06 means the learned manifold fits every golden config almost
perfectly. Anything above 15.34 is structurally foreign.

Single-field drift — caught and attributed:

```
  Config  : Single field: db.host -> external
  Residual: 21.95  (threshold 15.34)  drift=True
  Cause 1 : 'db.host' (residual drop 21.92 when reverted)
  Fix     : amplify(golden, Δ) → residual 19.04  (was 21.95)
  Verify  : actual correct config residual = 0.03  (below threshold: True)
```

The residual drop of 21.92 when `db.host` is reverted to golden — nearly the
entire anomaly score — is definitive. The verified correct config scores 0.03,
confirming the remediation direction is structurally sound.

Subtle drift — a value that passes any schema validator:

```
  Config  : Subtle drift: pool_size 15 -> 8 (still 'valid', just wrong)
  Note    : 8 is a positive integer — any schema validator passes this
  Residual: 20.39  (threshold 15.34)  drift=True
  Cause 1 : 'db.pool_size' (residual drop 20.37 when reverted)
  Fix     : amplify(golden, Δ) → residual 20.61  (was 20.39)
  Verify  : actual correct config residual = 9.19  (below threshold: True)
```

`pool_size=8` is a perfectly valid integer. No schema catches it. The subspace
does, because it learned the distribution of `pool_size` values across 20
golden configs and 10 passes — the structure says this is wrong.

Multi-field drift — two simultaneous changes, both attributed:

```
  Config  : Multi-field: db compromised + rate limit blown
  Note    : Two simultaneous changes — cascading attribution finds both
  Residual: 31.68  (threshold 15.34)  drift=True
  Cause 1 : 'api.rate_limit' (residual drop 9.47 when reverted)
  Cause 2 : 'db.host' (residual drop 7.85 when reverted)
  Fix     : amplify(golden, Δ) → residual 23.44  (was 31.68)
  Verify  : actual correct config residual = 0.04  (below threshold: True)
```

The residual-swap loop runs over every leaf field independently. When two fields
change, both show large drops — you get an ordered list of culprits, not just one.

```
Detected 5/5 drifts — including multi-field and subtle in-range changes.
No schema annotations. No field-specific rules. Pure structure.
```

## Without Holon

You would need: a per-field allowlist or range spec (maintained as configs evolve),
explicit rules for multi-field correlations (e.g., "if db.host changes, check
db.port"), and a separate alerting pipeline. Subtle in-range changes require
statistical baselines per field, maintained separately. There is no generic
attribution — you need to know which fields to check.

## Try

- Add a per-environment engram (staging vs prod) to catch environment mix-ups.
- Inject a two-field drift and run `attribute_drift` iteratively: fix the top
  culprit, re-check, find the second.
- Lower `sigma_mult=2.0` to tighten the threshold and see if subtle drift is
  caught more aggressively.
