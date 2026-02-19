# Config Drift Remediation

Stream 15 stable config versions through a subspace until it converges
(training residual → ~0). Any production config that exceeds the threshold
is drift; the culprit key is found by reverting fields one-at-a-time.
A remediation vector is computed via `difference(golden, drifted)` + `amplify`.

**Why Holon:** nested JSON → one vector. Drift detection, field attribution,
and remediation direction all fall out of the same algebra.

```bash
./scripts/run_with_venv.sh python -m examples.showcases.config_drift_remediation.showcase
```

Expected output: `Detected 5/5 drifts.`
