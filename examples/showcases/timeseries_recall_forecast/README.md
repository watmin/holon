# Time-Series Recall & Forecast

Encode sensor sequences with positional time markers. Train one subspace per
named pattern (gradual rise, spike+recovery, steady state). Feed a partial
sequence → the engram with the lowest residual is the match; its stored
metadata gives the forecast.

**Why Holon:** no sliding windows, no DTW, no model training. Pattern recall
is a single residual comparison across the engram library.

```bash
./scripts/run_with_venv.sh python -m examples.showcases.timeseries_recall_forecast.showcase
```

Expected output: `Forecast: ['critical', 'critical', 'critical']   actual=[...]   correct=True`
