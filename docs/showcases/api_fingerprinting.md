# API Fingerprinting

## Problem
Synthetic HTTP request/response pairs cover 5 logical endpoints.  New requests
arrive with extra headers, changed user-agents, or additional body fields.
Match each to the correct endpoint without a routing table, and flag structurally
novel requests as anomalies.

## Why Holon Wins
One `OnlineSubspace` per endpoint learns its structural manifold from 10
training requests.  Variant resilience is emergent — the subspace smooths
over irrelevant surface variation automatically.  Anomaly detection is a
single residual threshold check; no regex, no schema validator.

## Run

```bash
./scripts/run_with_venv.sh python -m examples.showcases.api_fingerprinting.showcase
```

## Expected Output (abbreviated)

```
API FINGERPRINTING
...
  variant  : POST /api/v1/users  (+1 extra fields)
  matched  : 'create_user'  residual=26.94  correct=True

  DELETE /admin/purge  (never seen before)
    1. 'create_user'  residual=69.97
  avg residual=70.66  → uniformly high: no match
```

## Key Concepts
- Per-endpoint `OnlineSubspace` — structural fingerprint without a routing table
- Variant resilience — extra headers/fields averaged out by the manifold
- `EngramLibrary.match` — ranked retrieval across all fingerprints in one call
