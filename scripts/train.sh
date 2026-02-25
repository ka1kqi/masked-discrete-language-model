#!/bin/bash
# Train the diffusion LM with default config
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_DIR/.venv"

cleanup() {
    echo ""
    echo "Deactivating virtual environment..."
    deactivate 2>/dev/null || true
    echo "Done. Venv preserved at $VENV_DIR (run 'rm -rf $VENV_DIR' to remove)"
}
trap cleanup EXIT

# Create venv if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment at $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
fi

# Activate
source "$VENV_DIR/bin/activate"

# Install deps if needed
if [ ! -f "$VENV_DIR/.installed" ]; then
    echo "Installing dependencies..."
    pip install -e "$PROJECT_DIR[dev]"
    touch "$VENV_DIR/.installed"
fi

# Train
cd "$PROJECT_DIR"
python -m src.train --config "${1:-configs/default.yaml}"
