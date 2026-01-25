# Holon Scripts Organization

## Directory Structure

Scripts are organized to match the challenge structure in `docs/challenges/`:

```
scripts/
├── run_with_venv.sh          # Helper script to run commands with venv
├── 001-batch/                # Solutions for first batch of challenges
│   ├── 001-solution.py       # Personal Task Memory system
│   └── ...
├── 002-batch/                # Solutions for second batch
│   └── ...
└── [other utility scripts]    # General testing/performance scripts
```

## Usage

Always use the virtual environment to avoid polluting your system Python:

```bash
# Run a solution script
./scripts/run_with_venv.sh python scripts/001-batch/001-solution.py

# Run tests
./scripts/run_with_venv.sh pytest tests/

# Run any Python command
./scripts/run_with_venv.sh python -m mymodule
```

## Solutions by Batch

### 001-batch: Neural Memory Foundations
- **001-solution.py**: Personal Task Memory - Fuzzy task retrieval with guards, negations, and wildcards

### Future Batches
- **002-batch**: Graph algorithms using VSA/HDC
- **003-batch**: Book quote indexing and search
- **004-batch**: Sudoku solver with geometric constraints

## Development Workflow

1. **Use venv**: Always run Python commands through `./scripts/run_with_venv.sh`
2. **Organize by batch**: Place solutions in matching batch directories
3. **Test thoroughly**: Run full test suite before committing
4. **Document**: Update this README when adding new batches/solutions