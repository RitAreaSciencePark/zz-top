#!/bin/bash
#SBATCH --job-name=zztop-gpu-tests
#SBATCH --partition=DGX                    # <-- adjust for your cluster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:A100:1                  # <-- adjust GPU type
#SBATCH --time=00:30:00
#SBATCH --output=slurm-zztop-gpu-%j.out
#SBATCH --error=slurm-zztop-gpu-%j.err

# ============================================================
# zztop GPU test runner  (TEMPLATE — edit paths below)
#
# This script:
#   1. Activates a conda env with the GPU-enabled GUDHI fork
#   2. Installs zztop (editable) into the env
#   3. Runs the full test suite (GPU parity tests should no
#      longer be skipped)
#
# Before first use, set CONDA_BASE, CONDA_ENV, ZZTOP_DIR,
# and the CUDA module version to match your environment.
# ============================================================

set -euo pipefail

# ---- USER CONFIG ----
CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"      # <-- path to conda
CONDA_ENV="${CONDA_ENV:-topofisher_gudhi_gpu}"     # <-- env with GPU GUDHI
ZZTOP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"  # repo root
CUDA_MODULE="${CUDA_MODULE:-cuda/12.6}"            # <-- CUDA module to load
# ---------------------

echo "=== Job info ==="
echo "Node:       $(hostname)"
echo "Date:       $(date)"
echo "Job ID:     ${SLURM_JOB_ID}"
echo "Partition:  ${SLURM_JOB_PARTITION}"
nvidia-smi 2>/dev/null || echo "(nvidia-smi not available)"
echo ""

# --- Conda setup ---
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

echo "=== Environment ==="
echo "Python:     $(python --version)"
echo "Python path: $(which python)"
echo "GUDHI:      $(python -c 'import gudhi; print(gudhi.__version__)')"
echo ""

# --- Check GPU backend before install ---
echo "=== GPU backend probe (before zztop install) ==="
python -c "
from gudhi._pers_cub_low_dim_gpu_ext import __backend__
print(f'Backend tag: {__backend__}')
" || echo "WARNING: GPU ext not loadable"
echo ""

# --- Load CUDA module if needed ---
module load "${CUDA_MODULE}" 2>/dev/null || true

# --- Install zztop (editable, no deps — deps already in env) ---
echo "=== Installing zztop from ${ZZTOP_DIR} ==="
pip install --no-deps -e "${ZZTOP_DIR}" 2>&1 | tail -3
echo ""

# --- Run the full test suite ---
echo "=== Running tests ==="
cd "${ZZTOP_DIR}"
python -m pytest tests/ -v --tb=short 2>&1

echo ""
echo "=== Done ==="
date
