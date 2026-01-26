# Contributing to Holon

See the main [README](../README.md) for overview and project details.

## Development Setup
1. Clone: `git clone https://github.com/watmin/holon.git`
2. Create venv: `python -m venv holon_env`
3. Activate: `. holon_env/bin/activate` (Linux/Mac) or `holon_env\Scripts\activate` (Windows)
4. Install: `./scripts/run_with_venv.sh pip install -r requirements.txt && ./scripts/run_with_venv.sh pip install -e .`
5. Install dev tools: `./scripts/run_with_venv.sh pip install -r requirements-dev.txt`
6. Setup pre-commit: `./scripts/run_with_venv.sh pre-commit install`
7. Test: `./scripts/run_with_venv.sh python -m pytest tests/`

## Adding Features
- **Encoders**: Subclass `Encoder` for new data types.
- **Queries**: Extend parsing in `cpu_store.py`.
- **API**: Add routes in `scripts/server/holon_server.py`.
- **Tests**: Add to `tests/` with pytest.

## Testing & Quality Assurance

- **Test Coverage**: 138 test cases across unit, integration, and performance suites
- **Pass Rate**: 136/138 tests pass (2 GPU-related tests appropriately skipped)
- **Code Quality**: Pre-commit hooks handle formatting (Black), import sorting (isort), and style checks (flake8)
- **Type Hints**: Full type annotation coverage for better IDE support
- **Docstrings**: Comprehensive documentation for all public APIs

### Running Tests
```bash
# All tests
./scripts/run_with_venv.sh python -m pytest tests/

# Specific test categories
./scripts/run_with_venv.sh python -m pytest tests/test_cpu_store.py
./scripts/run_with_venv.sh python -m pytest tests/test_similarity_coverage.py

# Performance benchmarks
./scripts/run_with_venv.sh python scripts/demos/test_accuracy.py
```

## Code Style
- Python 3.8+
- Type hints encouraged.
- Docstrings for all functions.
- PEP 8 compliance with automated formatting.

## Submitting Changes
1. Fork & branch.
2. Add tests.
3. Ensure CI passes.
4. PR with description.

Thanks for contributing!
