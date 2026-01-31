# Holon Documentation

See the main [README.md](../README.md) for overview and quick start.

## Core Documentation

| Document | Description |
|----------|-------------|
| [API Reference](api_reference.md) | Complete HTTP and Python API documentation |
| [Encoding Guide](encoding_guide.md) | How structured data becomes vectors |
| [Performance Guide](performance.md) | Scaling, benchmarks, optimization |
| [Similarity Methods](similarity_methods.md) | Similarity algorithms and when to use them |
| [Contributing](contributing.md) | Development setup and extending Holon |

## Architecture

| Document | Description |
|----------|-------------|
| [System Context](holon_context.md) | Overall architecture and data flow |
| [Architecture Decisions](architecture/decisions/) | Key design choices (ADRs) |

## Challenge Solutions

Each batch contains problem statements and learnings:

| Batch | Challenge | Status | Key Learning |
|-------|-----------|--------|--------------|
| [001](challenges/001-batch/) | Task Memory, Recipes, Bugs, D&D | Complete | Fuzzy retrieval with guards/negations |
| [002](challenges/002-batch/) | RPM Reasoning, Graph Matching | Complete | 100% classification with prototypes |
| [003](challenges/003-batch/) | Quote Finder | Complete | N-gram encoding, vector bootstrapping |
| [004](challenges/004-batch/) | Sudoku | Complete | VSA as heuristic layer ([honest assessment](challenges/004-batch/LEARNINGS.md)) |
| [005](challenges/005-batch/) | Additional challenges | Available | - |
| [006](challenges/006-batch/) | Additional challenges | Available | - |

## Examples

| Example | Description |
|---------|-------------|
| [basic_usage.py](../examples/basic_usage.py) | Getting started with JSON/EDN |
| [advanced_queries.py](../examples/advanced_queries.py) | Guards, negations, $or logic |
| [bulk_operations.py](../examples/bulk_operations.py) | Efficient large-scale handling |
| [geometric_reasoning.py](../examples/geometric_reasoning.py) | Pattern completion |
| [http_api_example.py](../examples/http_api_example.py) | REST API usage |
| [edn_usage.py](../examples/edn_usage.py) | Rich data types and keywords |

## Archive

[Legacy documentation](archive/) - Older files and working notes preserved for reference.
