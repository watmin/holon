# Time-Series Recall & Forecast

## Problem
Synthetic sensor streams follow one of three patterns (gradual rise from warm,
spike+recovery, steady state).  Given only the first half of a new stream,
identify the pattern and forecast the next 3 states.

## Why Holon Wins
Each pattern's subspace captures its manifold from 10 training examples ×
5 passes (no label field — discrimination comes from value trajectory alone).
`{"$time": unix_ts}` encodes timestamps with circular hour-of-day structure.
Matching is pure residual comparison — no RNN, no window buffering.

## Run

```bash
./scripts/run_with_venv.sh python -m examples.showcases.timeseries_recall_forecast.showcase
```

## Expected Output (abbreviated)

```
TIME-SERIES RECALL & FORECAST
...
  Input   : 'spike_recovery'  (current state=critical)
  Matches :
    1. 'spike_recovery'  residual=62.58  forecast=['normal', 'normal', 'normal'] ← best
    2. 'gradual_rise'    residual=62.94  forecast=['critical', 'critical', 'critical']
    3. 'steady_state'    residual=63.57  forecast=['normal', 'normal', 'normal']
  Forecast: ['normal', 'normal', 'normal']   actual=[...]   correct=True
```

## Key Concepts
- `{"$time": unix_ts}` — circular timestamp encoding baked into each step
- `encode_list(..., POSITIONAL)` — order-aware sequence encoding
- Per-pattern `OnlineSubspace` + `EngramLibrary` — multi-model recall in one structure
- Engram metadata — lightweight forecast storage alongside the subspace
