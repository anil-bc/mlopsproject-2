#!/usr/bin/env bash
# =============================================================================
# One-time setup for Ubuntu / Debian / WSL2-Ubuntu (recommended on Windows)
#   - C compiler with OpenMP (gcc), make
#   - Open MPI runtime + headers (mpirun, mpicc)
#   - Python venv with numpy, mpi4py, matplotlib
# Usage (from the repo root):   bash setup/install_ubuntu_wsl.sh
# =============================================================================
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo ">>> [1/3] Installing system packages (needs sudo)"
sudo apt-get update
sudo apt-get install -y build-essential openmpi-bin libopenmpi-dev \
                        python3 python3-venv python3-dev python3-pip

echo ">>> [2/3] Creating Python virtual environment in $ROOT/.venv"
python3 -m venv "$ROOT/.venv"
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"
pip install --upgrade pip
pip install -r "$ROOT/requirements.txt"     # numpy, mpi4py (compiles against Open MPI), matplotlib

echo ">>> [3/3] Verifying the toolchain"
bash "$ROOT/setup/check_env.sh"
echo
echo "Setup complete. Next:  source .venv/bin/activate   then   ./run_all.sh"
