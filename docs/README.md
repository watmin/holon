# Holon Documentation

See the main [README.md](../README.md) for overview and quick start.

## Project Assessment

**[ASSESSMENT.md](ASSESSMENT.md)** - Honest evaluation of what works, what doesn't, and what we learned.

## Core Documentation

| Document | Description |
|----------|-------------|
| [API Reference](api_reference.md) | Complete HTTP and Python API documentation |
| [Encoding Guide](encoding_guide.md) | How structured data becomes vectors |
| [Walkable Extension](walkable-extension.md) | Zero-serialization encoding for custom types |
| [Dimension Selection](dimension_selection.md) | Choosing optimal vector dimensions for accuracy vs memory |
| [Performance Guide](performance.md) | Scaling, benchmarks, optimization |
| [Similarity Methods](similarity_methods.md) | Similarity algorithms and when to use them |
| [Contributing](contributing.md) | Development setup and extending Holon |

## Architecture

| Document | Description |
|----------|-------------|
| [Architecture Overview](ARCHITECTURE.md) | **NEW**: Layered design (kernel/memory/highlevel), import patterns, and design rationale |
| [System Context](holon_context.md) | Overall architecture and data flow |
| [Architecture Decisions](architecture/decisions/) | Key design choices (ADRs) |

## Challenge Solutions

Each batch contains problem statements and learnings:

| Batch | Challenge | Status | Key Learning |
|-------|-----------|--------|--------------|
| [001](challenges/001-batch/) | Task Memory, Recipes, Bugs, D&D | Complete | Fuzzy retrieval with guards/negations |
| [002](challenges/002-batch/) | RPM Reasoning, Graph Matching | Complete | 100% classification with prototypes |
| [003](challenges/003-batch/) | Quote Finder | Complete | N-gram encoding, vector bootstrapping |
| [004](challenges/004-batch/) | Sudoku | Complete | VSA cannot solve CSPs ([honest assessment](challenges/004-batch/LEARNINGS.md)) |
| [005](challenges/005-batch/) | NP-hard optimization | Not attempted | Expected to fail (same reasons as Sudoku) |
| [006](challenges/006-batch/) | LLM Memory Augmentation | Complete | [Ideal use case - 82% token savings](challenges/006-batch/LEARNINGS.md) |

## Showcases (Non-Networking)

End-to-end demos of Holon's generic power — no networking stack required.

| Showcase | Description |
|----------|-------------|
| [Log Anomaly Memory](showcases/log_anomaly_memory.md) | Detect and attribute log anomalies; recall matching engram |
| [Config Drift Remediation](showcases/config_drift_remediation.md) | Detect config drift, attribute culprit field, remediate with `difference`+`amplify` |
| [Time-Series Recall & Forecast](showcases/timeseries_recall_forecast.md) | Match partial sensor sequences, forecast next 3 states |
| [API Fingerprinting](showcases/api_fingerprinting.md) | Variant-resilient endpoint fingerprinting; anomaly on unknown structure |

Run any with:
```bash
./scripts/run_with_venv.sh python -m examples.showcases.<name>.showcase
```

## Examples

| Example | Description |
|---------|-------------|
| [basic_usage.py](../examples/basic_usage.py) | Getting started with JSON/EDN |
| [walkable_demo.py](../examples/walkable_demo.py) | Zero-serialization encoding with custom types |
| [advanced_queries.py](../examples/advanced_queries.py) | Guards, negations, $or logic |
| [bulk_operations.py](../examples/bulk_operations.py) | Efficient large-scale handling |
| [geometric_reasoning.py](../examples/geometric_reasoning.py) | Pattern completion |
| [http_api_example.py](../examples/http_api_example.py) | REST API usage |
| [edn_usage.py](../examples/edn_usage.py) | Rich data types and keywords |

## Archive

[Legacy documentation](archive/) - Older files and working notes preserved for reference.
