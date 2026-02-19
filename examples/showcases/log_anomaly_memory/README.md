# Log Anomaly Memory

Learn normal log behavior as a subspace. When a new log line falls outside
the manifold, attribute the anomaly to a specific field by swapping each
field back to a normal value and measuring the residual drop.

**Why Holon:** no rules, no regex — the normal distribution *is* the model.
Field attribution comes from the geometry, not from hand-coded logic.

```bash
./scripts/run_with_venv.sh python -m examples.showcases.log_anomaly_memory.showcase
```

Expected output: `Anomaly type detected: action='exfiltrate' … cause: 'action' (residual drop 6.40 when normalised)`
