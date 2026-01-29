# Holon Scripts Organization

## Directory Structure

```
scripts/
├── run_with_venv.sh          # Virtual environment helper
├── server/                   # HTTP API server
│   └── holon_server.py
├── challenges/               # Challenge solutions (organized by batch)
│   ├── 001-batch/           # Personal task memory
│   ├── 002-batch/           # RPM geometric solver
│   ├── 003-batch/           # PDF quote finder
│   └── 004-batch/           # Sudoku solver (hyperspace-guided backtracking)
├── tests/                   # Testing scripts
│   ├── performance/         # Stress and performance tests
│   ├── integration/         # End-to-end pipeline tests
│   └── api/                 # API feature tests
├── demos/                   # Demonstration scripts
└── utils/                   # Utility and validation scripts
```

## Usage

Always use the virtual environment to avoid polluting your system Python:

```bash
# Start the server
./scripts/run_with_venv.sh python scripts/server/holon_server.py

# Run performance tests
./scripts/run_with_venv.sh python scripts/tests/performance/extreme_query_challenge.py

# Run integration tests
./scripts/run_with_venv.sh python scripts/tests/integration/test_comprehensive.py

# Run challenge solutions
./scripts/run_with_venv.sh python scripts/challenges/001-batch/001-solution.py

# Run unit tests
./scripts/run_with_venv.sh pytest tests/
```

## Challenge Solutions

### ✅ 001-batch: Personal Task Memory
- **001-solution.py**: Fuzzy task retrieval with guards, negations, and wildcards
- **002-solution.py**: HTTP-integrated task management
- **003-solution.py**: Advanced querying with hierarchical tasks
- **004-solution.py**: Complete task management system

### ✅ 002-batch: Raven's Progressive Matrices
- **001-solution.py**: Basic RPM geometric encoding
- **002-solution.py**: Statistical validation of geometric learning

### ✅ 003-batch: PDF Quote Finder
- **quote_finder_app.py**: Full PDF indexing and search system
- **pdf_content_indexer.py**: Document structure extraction
- **vector bootstrapping**: API for custom search term encoding

### ✅ 004-batch: Sudoku Solver (Hyperspace-Guided Backtracking)
- **001-solution.py**: Local solver with VSA/HDC-guided backtracking
- **002-solution-http.py**: HTTP API version demonstrating Holon-as-a-service
- **LEARNINGS.md**: Honest assessment of what VSA/HDC actually does here
- **FUTURE_RADICAL_APPROACHES.md**: Ideas for genuinely geometric solutions

**Honest Note**: VSA/HDC provides ordering heuristics for backtracking, not geometric solutions. Valuable API improvements (batch search) but not the "radical" approach originally envisioned.

## Testing Strategy

- **Unit Tests**: `tests/` directory (core functionality)
- **Integration Tests**: `tests/integration/` (full pipelines)
- **Performance Tests**: `tests/performance/` (stress testing)
- **API Tests**: `tests/api/` (HTTP endpoints and features)

## Development Workflow

1. **Use venv**: Always run through `./scripts/run_with_venv.sh`
2. **Organize by purpose**: Place scripts in appropriate subdirectories
3. **Test thoroughly**: Run relevant tests before committing
4. **Document**: Update this README when adding new functionality
5. **Challenge solutions**: Keep in `challenges/` with batch organization
