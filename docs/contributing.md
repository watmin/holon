# Contributing to Holon

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

## Code Style
- Python 3.8+
- Type hints encouraged.
- Docstrings for all functions.
- **Pre-commit hooks** handle formatting (Black), import sorting (isort), and style checks (flake8).

## Submitting Changes
1. Fork & branch.
2. Add tests.
3. Ensure CI passes.
4. PR with description.

Thanks for contributing! 🚀
