# Log Anomaly Memory

Catching what per-field rules can't: combinatorial anomalies.

## The Problem

You have an allowlist of valid users, valid actions, and valid status codes. It
passes everything. The insider threat has valid credentials and uses a valid
action — just one they've never used before. The compromised account has the
right user, the right hour, the right status code — the combination is what's
wrong.

Per-field rules are blind to correlation. Holon isn't.

## What Holon Does

`OnlineSubspace` learns the *joint manifold* of normal behavior from 600 log
lines — which users do which actions, at which hours, with which response codes.
No schema, no feature engineering, no label column. When an observation doesn't
fit the learned structure, the residual spikes. Residual-swap attribution then
identifies which specific field broke the pattern.

## Run

```bash
./scripts/run_with_venv.sh python -m examples.showcases.log_anomaly_memory.showcase
```

## Output

After training on 600 correlated normal logs (users with behavioral fingerprints),
the subspace threshold is learned automatically from the EMA of training residuals:

```
Learning normal behavior (600 logs, correlated user profiles)...
  Threshold  : 16.97
  Explained  : 97.7%
  (adaptive threshold: EMA of training residuals + 2.5σ — zero hand-tuning)
```

97.7% explained variance means the subspace has captured the dominant structure of
normal behavior. The threshold of 16.97 was set entirely from the training data —
no human chose it.

Now four anomalies, each with every field individually valid:

```
  [ANOMALY] view-only user does 'edit'
      Note    : u_001 only ever views/exports — 'edit' is a valid system action, never from this user
      Residual: 20.30  (threshold 16.97)
      Culprit : 'action' (residual drops 8.63 when field is normalised)
      Overlap : 62.7% structural similarity to learned normal
      Engram  : 'normal_ops'  (nearest memory, fit 20.30)

  [ANOMALY] export-only user does 'login'
      Note    : u_007 only exports — 'login' is a valid system action, u_007 never uses it
      Residual: 20.74  (threshold 16.97)
      Culprit : 'action' (residual drops 11.43 when field is normalised)
      Overlap : 62.4% structural similarity to learned normal
      Engram  : 'normal_ops'  (nearest memory, fit 20.74)

  [ANOMALY] edit action returns unexpected 301 redirect
      Note    : u_005 edits always return 200 — 301 is a valid HTTP code, wrong for this pattern
      Residual: 26.66  (threshold 16.97)
      Culprit : 'status' (residual drops 15.34 when field is normalised)
      Overlap : 46.0% structural similarity to learned normal
      Engram  : 'normal_ops'  (nearest memory, fit 26.66)

  [ANOMALY] login-only user suddenly does 'export'
      Note    : u_009 does login/edit — 'export' is in the system, just not this user's pattern
      Residual: 19.63  (threshold 16.97)
      Culprit : 'user' (residual drops 13.05 when field is normalised)
      Overlap : 66.7% structural similarity to learned normal
      Engram  : 'normal_ops'  (nearest memory, fit 19.63)
```

All four anomalies are detected. The `Culprit` field is identified by
residual-swap: temporarily normalising each field and measuring how much the
residual drops. The field with the largest drop is the structural violation.

The `Overlap` percentage shows that each anomaly is mostly similar to normal (62-66%)
— these are not obvious outliers. They are right on the edge of the manifold.

Genuinely normal logs stay well below threshold:

```
  [normal ] user=u_006  action=edit  status=200
      Residual: 14.06  (threshold 16.97)

  [normal ] user=u_006  action=edit  status=200
      Residual: 13.57  (threshold 16.97)

  [normal ] user=u_010  action=login  status=200
      Residual: 8.04  (threshold 16.97)
```

```
Per-field allowlists see 0 violations in the anomaly set.
Holon catches 4/4 combinatorial anomalies — zero rules written.
```

## Without Holon

You would need: a per-user behavioral model (trained on labeled sequences), a
feature extraction pipeline (user × action × hour cross-features), a classifier
(trained with positive and negative examples), and a threshold chosen manually
per field or feature. Reconfigure when any user changes their behavior pattern.

## Try

- Lower `sigma_mult=1.5` for more sensitivity (will flag more, may increase false positives).
- Add a second `EngramLibrary` engram for a known attack pattern and see `match()` rank it above `normal_ops` on anomalies.
- Swap the `attribute_field` loop to also test `timestamp` to see if off-hours detection works.
