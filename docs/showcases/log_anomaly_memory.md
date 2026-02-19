# Log Anomaly Memory

## Problem
500 JSON log lines, 90% normal traffic, 10% injected anomalies (bad actions,
error statuses, unknown users).  Find the anomalies, name the culprit field,
recall the matching learned engram.

## Why Holon Wins
One `OnlineSubspace` encodes the normal manifold from 400 examples.  No labels,
no separate model per anomaly type.  Field attribution is a single residual-swap
loop — no gradient computation.  The `EngramLibrary` links every detection back
to the named memory that flagged it.

## Run

```bash
./scripts/run_with_venv.sh python -m examples.showcases.log_anomaly_memory.showcase
```

## Expected Output (abbreviated)

```
LOG ANOMALY MEMORY
...
  Anomaly type detected: action='exfiltrate'  status='200'
    cause   : 'action' (residual drop 6.40 when normalised)
    residual: 44.85  (threshold 39.52)
    engram  : 'normal_ops'  (fit 44.85)
...
Detected 10 anomalies in 100 test logs (5 distinct types).
```

## Key Concepts
- `OnlineSubspace` — online, label-free manifold learning
- Residual-swap attribution — field-level root-cause without backprop
- `EngramLibrary` — named memory linking detections to learned patterns
- `{"$time": unix_ts}` — circular timestamp encoding (hour-of-day, day-of-week)
