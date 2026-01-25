#!/bin/bash

# Helper script to run Python commands with the Holon virtual environment
# Usage: ./scripts/run_with_venv.sh python script.py [args...]
#        ./scripts/run_with_venv.sh pytest tests/ [args...]

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Activate the virtual environment
source "$PROJECT_ROOT/holon_env/bin/activate"

# Run the command with all arguments
"$@"

# Deactivate (optional, since this script exits)
deactivate