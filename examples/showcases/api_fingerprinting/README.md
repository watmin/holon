# API Fingerprinting

Mint one engram per logical endpoint from 10 example requests each. New
requests with extra headers, different user-agents, or added body fields
still match the correct endpoint. A structurally unknown endpoint gets
a uniformly high residual across all engrams — no match.

**Why Holon:** structural encoding means variant-resilient matching with
zero schema definitions or regex patterns.

```bash
./scripts/run_with_venv.sh python -m examples.showcases.api_fingerprinting.showcase
```

Expected output: `correct=True` for all 3 variants; high avg residual for unknown endpoint.
