#!/bin/bash
# Setup script for GPU training rig (RunPod deployment)
#
# Usage: bash scripts/setup_training.sh
#
# Assumes:
# - RunPod or similar with CUDA pre-installed
# - Repository cloned to /workspace/ventriloquist
# - data/ folder copied from macOS machine

set -e  # Exit on error

echo "=== GPU Training Rig Setup ==="

# System updates
echo "[1/7] Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Build tools and monitoring
echo "[2/7] Installing build tools..."
sudo apt install -y cmake libncurses5-dev libncursesw5-dev git

echo "[3/7] Installing nvtop for GPU monitoring..."
sudo apt install -y nvtop

echo "[4/7] Installing screen for decoupled running..."
sudo apt install -y screen

# Install uv if not present
echo "[5/7] Setting up uv..."
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Sync Python dependencies
echo "[6/7] Syncing Python dependencies..."
cd /workspace/ventriloquist
uv sync

# Optional: wandb login reminder
echo "[7/7] Final setup..."
if [ -z "$WANDB_API_KEY" ]; then
    echo ""
    echo "NOTE: WANDB_API_KEY not set. Run one of:"
    echo "  export WANDB_API_KEY=your_key"
    echo "  wandb login"
    echo ""
fi

# Verify setup
echo "=== Verifying Setup ==="
echo "Python: $(uv run python --version)"
echo "CUDA available: $(uv run python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU: $(uv run python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")')"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "  1. Copy data/ folder from macOS: rsync -avz data/ runpod:/workspace/ventriloquist/data/"
echo "  2. Run preprocessing: uv run python main.py 2"
echo "  3. Start training: uv run python main.py 3"
echo ""
echo "Useful commands:"
echo "  nvtop                    # Monitor GPU usage"
echo "  uv run python main.py    # Interactive menu"
