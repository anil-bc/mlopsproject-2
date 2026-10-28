#!/usr/bin/env bash
# =============================================================================
# One-time setup for macOS (Apple clang needs libomp for OpenMP)
# Requires Homebrew: https://brew.sh
# Usage (from the repo root):   bash setup/install_macos.sh
# =============================================================================
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo ">>> [1/3] Installing libomp and Open MPI with Homebrew"
brew install libomp open-mpi

echo ">>> [2/3] Creating Python virtual environment in $ROOT/.venv"
python3 -m venv "$ROOT/.venv"
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"
pip install --upgrade pip
pip install -r "$ROOT/requirements.txt"

echo ">>> [3/3] Verifying the toolchain"
bash "$ROOT/setup/check_env.sh"
echo
echo "Setup complete. Next:  source .venv/bin/activate   then   ./run_all.sh"
