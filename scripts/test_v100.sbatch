#!/bin/bash
#SBATCH --job-name=zztop-v100
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:V100:1
#SBATCH --time=01:00:00
#SBATCH --output=slurm-zztop-v100-%j.out
#SBATCH --error=slurm-zztop-v100-%j.err

# ============================================================
# Rebuild GUDHI GPU fork (with 3D kernels) + run zztop tests
# + real-data 3D benchmark
# ============================================================

set -euo pipefail

CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
CONDA_ENV="${CONDA_ENV:-topofisher_gudhi_gpu}"
GUDHI_SRC="/u/area/mbiagetti/Codes/gudhi-devel"
ZZTOP_DIR="/u/area/mbiagetti/Codes/ZigZagPipeline/zztop"
BUILD_DIR="${GUDHI_SRC}/build_gpu_kernels"
DATA_DIR="/orfeo/scratch/area/ygardinazzi/sensorium_2026/derivatives/grid-15x15x10_norm-by_minmax/dynamic29156-11-10-Video-8744edeac3b4d1ce16b680916b5267ce/trials"

echo "=== Job info ==="
echo "Node:       $(hostname)"
echo "Date:       $(date)"
echo "Job ID:     ${SLURM_JOB_ID}"
echo "Partition:  ${SLURM_JOB_PARTITION}"
nvidia-smi 2>/dev/null || echo "(nvidia-smi not available)"
echo ""

# --- Load CUDA ---
# Try multiple strategies to find nvcc:
# 1. /usr/local/cuda-* (DGX nodes)
# 2. module load cuda (GPU partition nodes)
# 3. conda env (some envs bundle cuda-toolkit)
CUDA_FOUND=""
for cuda_dir in /usr/local/cuda-12.8 /usr/local/cuda-12.6 /usr/local/cuda-12.2 /usr/local/cuda-12 /usr/local/cuda; do
    if [[ -x "${cuda_dir}/bin/nvcc" ]]; then
        export PATH="${cuda_dir}/bin:${PATH}"
        export LD_LIBRARY_PATH="${cuda_dir}/lib64:${LD_LIBRARY_PATH:-}"
        CUDA_FOUND="${cuda_dir}"
        break
    fi
done

if [[ -z "${CUDA_FOUND}" ]]; then
    # Try module system
    if type module &>/dev/null; then
        echo "Probing module system for CUDA..."
        for mod in cuda/12.6 cuda/12.2 cuda/12.0 cuda/11.8 cuda; do
            module load "$mod" 2>/dev/null && break
        done
    fi
    if command -v nvcc &>/dev/null; then
        CUDA_FOUND="$(dirname $(dirname $(which nvcc)))"
    fi
fi

if [[ -z "${CUDA_FOUND}" ]]; then
    echo "ERROR: no nvcc found (tried /usr/local/cuda-*, module load, PATH)"
    echo "Available modules:"
    module avail cuda 2>&1 | head -20 || true
    echo "Contents of /usr/local:"
    ls /usr/local/ | grep -i cuda || echo "(none)"
    echo "Contents of /opt:"
    ls /opt/programs/ 2>/dev/null | grep -i cuda || echo "(none)"
    exit 1
fi
echo "CUDA dir:   ${CUDA_FOUND}"
echo "nvcc:       $(nvcc --version 2>/dev/null | tail -1 || echo 'NOT FOUND')"

# --- Conda setup ---
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

echo "=== Environment ==="
echo "Python:     $(python --version)"
echo "Python path: $(which python)"
echo "nvcc:       $(nvcc --version 2>/dev/null | tail -1 || echo 'not found')"
echo "GUDHI:      $(python -c 'import gudhi; print(gudhi.__version__)')"
echo ""

# ============================================================
# STEP 1 — Rebuild GPU kernels .so from source
# ============================================================
echo "=== Step 1: Rebuild GPU kernel extension (with 3D support) ==="
rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"

# Copy the CMakeLists.txt into the build dir (avoid stale CMakeCache conflicts)
cp "${GUDHI_SRC}/archives/build_gpu_kernels/CMakeLists.txt" "${BUILD_DIR}/CMakeLists.txt"

cd "${BUILD_DIR}"

cmake -DGUDHI_SRC="${GUDHI_SRC}" \
      -DCMAKE_CUDA_ARCHITECTURES="70;80;90" \
      -DPython_EXECUTABLE="$(which python)" \
      -Dnanobind_DIR="$(python -c 'import nanobind; print(nanobind.cmake_dir())')" \
      -DCMAKE_BUILD_TYPE=Release \
      .

make -j"${SLURM_CPUS_PER_TASK}" VERBOSE=1 2>&1 | tail -30
echo ""

# --- Copy the compiled .so into the conda GUDHI package ---
SO_FILE=$(find "${BUILD_DIR}" -name "_pers_cub_low_dim_gpu_kernels_ext*.so" -maxdepth 1 | head -1)
GUDHI_PKG=$(python -c "import gudhi, os; print(os.path.dirname(gudhi.__file__))")
if [[ -z "${SO_FILE}" ]]; then
    echo "ERROR: .so not found after build"
    exit 1
fi
echo "Copying ${SO_FILE} -> ${GUDHI_PKG}/"
cp -v "${SO_FILE}" "${GUDHI_PKG}/"

# Also copy the updated Python wrapper (with 3D function definitions)
echo "Copying updated _pers_cub_low_dim_gpu_ext.py -> ${GUDHI_PKG}/"
cp -v "${GUDHI_SRC}/src/python/gudhi/_pers_cub_low_dim_gpu_ext.py" "${GUDHI_PKG}/"
echo ""

# --- Verify the new extension loads with CUDA backend and 3D functions ---
echo "=== Step 1b: Verify GPU extension ==="
python -c "
from gudhi._pers_cub_low_dim_gpu_ext import __backend__
print(f'Backend tag: {__backend__}')
from gudhi._pers_cub_low_dim_gpu_ext import (
    _persistence_on_rectangle_from_top_cells_gpu,
    _persistence_on_box_from_top_cells_gpu,
    _persistence_on_boxes_from_top_cells_gpu_batched,
)
print('2D single: OK')
print('3D single: OK')
print('3D batched: OK')
print('All GPU functions loaded successfully')
"
echo ""

# ============================================================
# STEP 2 — Install zztop + run test suite
# ============================================================
echo "=== Step 2: Install zztop ==="
cd "${ZZTOP_DIR}"
pip install --no-deps -e . 2>&1 | tail -3
echo ""

echo "=== Step 3: zztop GPU detection ==="
python -c "
from zztop import gpu_available, gpu_info, gpu_supports_3d
info = gpu_info()
print(f'GPU available:    {gpu_available()}')
print(f'Backend tag:      {info[\"backend_tag\"]}')
print(f'3D supported:     {gpu_supports_3d()}')
print(f'Batched 2D fn:    {info[\"batched_top_cells_fn\"] is not None}')
print(f'Batched 3D fn:    {info[\"batched_top_cells_3d_fn\"] is not None}')
"
echo ""

echo "=== Step 4: Run full test suite ==="
python -m pytest tests/ -v --tb=short 2>&1
echo ""

# ============================================================
# STEP 5 — Real data 3D benchmark (CPU vs GPU)
# ============================================================
echo "=== Step 5: Real-data 3D benchmark ==="
python -c "
import numpy as np
import time
import glob
from zztop import run_cubical_persistence_gpu, gpu_supports_3d

# Load one trial
files = sorted(glob.glob('${DATA_DIR}/*.npy'))
if not files:
    print('ERROR: no .npy files found')
    exit(1)

data = np.load(files[0])
print(f'Loaded: {files[0]}')
print(f'Shape: {data.shape}, dtype: {data.dtype}')
print(f'Value range: [{data.min():.4f}, {data.max():.4f}]')

# Use first 20 frames for the benchmark (enough to show GPU batching advantage)
n_bench = min(20, data.shape[-1])
grid = data[..., :n_bench]  # (15, 15, 10, n_bench)
print(f'Benchmark: {n_bench} frames of {grid.shape[:-1]} 3D grid')
print()

# --- CPU benchmark ---
t0 = time.perf_counter()
cpu_dgms = run_cubical_persistence_gpu(grid, backend='cpu', min_persistence=0.0)
t_cpu = time.perf_counter() - t0
print(f'CPU time ({n_bench} frames): {t_cpu:.3f}s')
for dim in range(3):
    bars = sum(dgm[dim].shape[0] for dgm in cpu_dgms if dim in dgm)
    print(f'  H{dim}: {bars} total bars across all frames')

# --- GPU benchmark (if 3D available) ---
if gpu_supports_3d():
    t0 = time.perf_counter()
    gpu_dgms = run_cubical_persistence_gpu(grid, backend='gpu', min_persistence=0.0)
    t_gpu = time.perf_counter() - t0
    print(f'GPU time ({n_bench} frames): {t_gpu:.3f}s')
    print(f'Speedup: {t_cpu/t_gpu:.2f}x')
    for dim in range(3):
        bars = sum(dgm[dim].shape[0] for dgm in gpu_dgms if dim in dgm)
        print(f'  H{dim}: {bars} total bars across all frames')
else:
    print('3D GPU not available — skipping GPU benchmark')

# --- Full trial timing ---
print()
print('--- Full trial (all frames) ---')
t0 = time.perf_counter()
full_dgms = run_cubical_persistence_gpu(data, backend='cpu', min_persistence=0.0)
t_full = time.perf_counter() - t0
print(f'CPU: {data.shape[-1]} frames: {t_full:.3f}s')
for dim in range(3):
    bars = sum(dgm[dim].shape[0] for dgm in full_dgms if dim in dgm)
    print(f'  H{dim}: {bars} total bars')

if gpu_supports_3d():
    t0 = time.perf_counter()
    full_gpu = run_cubical_persistence_gpu(data, backend='gpu', min_persistence=0.0)
    t_full_gpu = time.perf_counter() - t0
    print(f'GPU: {data.shape[-1]} frames: {t_full_gpu:.3f}s  (speedup {t_full/t_full_gpu:.2f}x)')
    for dim in range(3):
        bars = sum(dgm[dim].shape[0] for dgm in full_gpu if dim in dgm)
        print(f'  H{dim}: {bars} total bars')
"
echo ""

echo "=== All done ==="
date
