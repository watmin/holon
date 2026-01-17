# Contributing to Holon

## Development Setup
1. Clone: `git clone https://github.com/watmin/holon.git`
2. Create venv: `python -m venv holon_env`
3. Activate: `. holon_env/bin/activate` (Linux/Mac) or `holon_env\Scripts\activate` (Windows)
4. Install: `pip install -r requirements.txt && pip install -e .`
5. Test: `python -m pytest tests/`

## Adding Features
- **Encoders**: Subclass `Encoder` for new data types.
- **Queries**: Extend parsing in `cpu_store.py`.
- **API**: Add routes in `holon_server.py`.
- **Tests**: Add to `tests/` with pytest.

## Code Style
- Python 3.8+
- Type hints encouraged.
- Docstrings for all functions.

## Submitting Changes
1. Fork & branch.
2. Add tests.
3. Ensure CI passes.
4. PR with description.

Thanks for contributing! 🚀
