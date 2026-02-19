# Config Drift Remediation

## Problem
15 stable config versions (nested JSON: db, redis, api, features) streamed
through a subspace until it converges.  Five variants inject controlled drift
(wrong host, wrong port, disabled feature, rate-limit explosion).  Detect each,
name the changed key, produce a remediation vector.

## Why Holon Wins
Multi-pass streaming converges the subspace tightly around golden behaviour
(max training residual → ~0.04).  Field attribution via residual-swap requires
no schema annotations — it works on arbitrary nested JSON.  `difference` +
`amplify` produce a corrective vector without rule tables or if/else chains.

## Run

```bash
./scripts/run_with_venv.sh python -m examples.showcases.config_drift_remediation.showcase
```

## Expected Output (abbreviated)

```
CONFIG DRIFT REMEDIATION
...
  Config  : db.host→external
  Residual: 24.15  (threshold 20.40)  drift=True
  Cause   : 'db.host' (residual drop 24.13 when reverted)
  Fix     : amplify(golden, Δ) → new residual 21.97  (was 24.15)
...
Detected 5/5 drifts.
```

## Key Concepts
- Multi-pass `OnlineSubspace` training — tight convergence on small datasets
- Residual-swap attribution — works on arbitrary nested JSON without schema
- `difference` + `amplify` — VSA remediation without rule tables
