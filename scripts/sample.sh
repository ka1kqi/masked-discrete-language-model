#!/bin/bash
# Generate samples from a trained checkpoint
# Usage: ./scripts/sample.sh [checkpoint] [--num-samples N] [--steps N] [--temperature F]
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

# Sample
cd "$PROJECT_DIR"
CHECKPOINT=${1:-checkpoints/checkpoint_final.pt}
shift 2>/dev/null || true
python -m src.sample "$CHECKPOINT" --num-samples 4 --steps 50 --temperature 0.8 "$@"
