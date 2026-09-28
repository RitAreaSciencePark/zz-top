#!/bin/bash
#SBATCH --job-name=zztop-vectorize
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:V100:1
#SBATCH --time=02:00:00
#SBATCH --output=slurm-zztop-vectorize-%j.out
#SBATCH --error=slurm-zztop-vectorize-%j.err

# ============================================================
# Compute zigzag + GPU static persistence on sensorium trials
# Save results for the vectorization demo notebook
# ============================================================

set -euo pipefail

CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
CONDA_ENV="${CONDA_ENV:-topofisher_gudhi_gpu}"
export GUDHI_SRC="/u/area/mbiagetti/Codes/gudhi-devel"
export ZZTOP_DIR="/u/area/mbiagetti/Codes/ZigZagPipeline/zztop"
BUILD_DIR="${GUDHI_SRC}/build_gpu_kernels"
export DATA_DIR="/orfeo/scratch/area/ygardinazzi/sensorium_2026/derivatives/grid-15x15x10_norm-by_minmax/dynamic29156-11-10-Video-8744edeac3b4d1ce16b680916b5267ce/trials"
export OUT_DIR="${ZZTOP_DIR}/experiments"

# Number of trials to process
export N_TRIALS=5

echo "=== Job info ==="
echo "Node:       $(hostname)"
echo "Date:       $(date)"
echo "Job ID:     ${SLURM_JOB_ID}"
echo "Partition:  ${SLURM_JOB_PARTITION}"
nvidia-smi 2>/dev/null || echo "(nvidia-smi not available)"
echo ""

# --- Load CUDA ---
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
echo "CUDA dir:   ${CUDA_FOUND:-NOT FOUND}"
echo "nvcc:       $(nvcc --version 2>/dev/null | tail -1 || echo 'NOT FOUND')"

# --- Conda setup ---
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

echo "=== Environment ==="
echo "Python:     $(python --version)"
echo "NumPy:      $(python -c 'import numpy; print(numpy.__version__)')"
echo ""

# --- Ensure GUDHI GPU extension is built ---
if ! python -c "import gudhi._pers_cub_low_dim_gpu_ext" 2>/dev/null; then
    echo "=== Building GUDHI GPU kernels ==="
    mkdir -p "${BUILD_DIR}"
    cd "${BUILD_DIR}"
    cmake .. \
        -DWITH_CUDA=ON \
        -DCUDA_ARCHITECTURES="70;80;90" \
        -DCMAKE_BUILD_TYPE=Release \
        -DPython_EXECUTABLE="$(which python)" 2>&1 | tail -5
    make -j$(nproc) _pers_cub_low_dim_gpu_kernels_ext 2>&1 | tail -10
    # Install into site-packages
    EXT=$(find . -name "_pers_cub_low_dim_gpu_ext*" -name "*.so" | head -1)
    if [[ -n "$EXT" ]]; then
        GUDHI_PKG=$(python -c "import gudhi, os; print(os.path.dirname(gudhi.__file__))")
        cp -v "$EXT" "$GUDHI_PKG/"
    fi
fi

# --- Install zztop ---
cd "${ZZTOP_DIR}"
pip install -e . --quiet 2>&1 | tail -3

# --- Quick GPU check ---
python -c "
from zztop import gpu_available, gpu_info
info = gpu_info()
print(f'GPU available: {gpu_available()}')
print(f'Backend tag:   {info[\"backend_tag\"]}')
print(f'3D support:    {info.get(\"batched_top_cells_3d_fn\") is not None}')
"

echo ""
echo "=== Computing persistence on ${N_TRIALS} trials (300 frames each) ==="
mkdir -p "${OUT_DIR}"

# --- Main computation ---
python << 'PYEOF'
import sys, os, glob, re, time, pickle
import numpy as np

sys.path.insert(0, os.environ["ZZTOP_DIR"])
from zztop import run_cubical_zigzag, run_cubical_persistence_gpu, gpu_available

DATA_DIR = os.environ["DATA_DIR"]
OUT_DIR = os.environ["OUT_DIR"]
N_TRIALS = int(os.environ["N_TRIALS"])

# Discover trial files, pick first N
all_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.npy")))
trial_nums = sorted(set(
    int(re.search(r'trial-(\d+)', f).group(1)) for f in all_files
))[:N_TRIALS]

print(f"Selected trials: {trial_nums}")
print(f"GPU available: {gpu_available()}")

results = {
    "trial_nums": trial_nums,
    "data_dir": DATA_DIR,
    "grid_shape": None,
    "n_frames": None,
    # Per-trial results
    "zigzag_bars": {},       # trial_num → list of (dim, birth, death)
    "zigzag_times": {},
    "static_diagrams": {},   # trial_num → list of {dim: ndarray}
    "static_times": {},
    "static_backend": None,
}

use_gpu = gpu_available()
results["static_backend"] = "gpu" if use_gpu else "cpu"
backend = "gpu" if use_gpu else "cpu"

for trial_num in trial_nums:
    # Find file
    pattern = f"*trial-{trial_num}.npy"
    matches = glob.glob(os.path.join(DATA_DIR, pattern))
    if not matches:
        print(f"  WARNING: trial-{trial_num} not found, skipping")
        continue
    fpath = matches[0]
    data = np.load(fpath)  # (15, 15, 10, 300)
    results["grid_shape"] = data.shape[:3]
    results["n_frames"] = data.shape[3]

    print(f"\n--- Trial {trial_num} ---")
    print(f"  Shape: {data.shape}, range: [{data.min():.4f}, {data.max():.4f}]")

    # 1) Cubical zigzag (CPU, all 300 frames)
    print(f"  Running cubical zigzag on {data.shape}...")
    t0 = time.time()
    bars = run_cubical_zigzag(data, threshold=0.0)
    dt_zz = time.time() - t0
    results["zigzag_bars"][trial_num] = bars
    results["zigzag_times"][trial_num] = dt_zz
    from collections import Counter
    c = Counter(b[0] for b in bars)
    print(f"  Zigzag: {len(bars)} bars in {dt_zz:.1f}s — dims: {dict(sorted(c.items()))}")

    # 2) Static per-frame persistence (GPU if available)
    print(f"  Running static persistence ({backend}) on {data.shape}...")
    t0 = time.time()
    diagrams = run_cubical_persistence_gpu(data, backend=backend)
    dt_st = time.time() - t0
    results["static_diagrams"][trial_num] = diagrams
    results["static_times"][trial_num] = dt_st
    n_bars_total = sum(d[dim].shape[0] for d in diagrams for dim in d)
    print(f"  Static: {n_bars_total} total bars across {len(diagrams)} frames in {dt_st:.1f}s")

# --- Save results ---
out_path = os.path.join(OUT_DIR, "persistence_results.pkl")
with open(out_path, "wb") as f:
    pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
print(f"\n=== Results saved to {out_path} ===")
print(f"  Trials:         {len(trial_nums)}")
print(f"  Grid shape:     {results['grid_shape']}")
print(f"  Frames/trial:   {results['n_frames']}")
print(f"  Static backend: {results['static_backend']}")

# Summary table
print(f"\n{'Trial':>6} {'ZZ bars':>10} {'ZZ time':>10} {'Static bars':>12} {'Static time':>12}")
print("-" * 55)
for tn in trial_nums:
    zz_n = len(results["zigzag_bars"].get(tn, []))
    zz_t = results["zigzag_times"].get(tn, 0)
    st_n = sum(d[dim].shape[0] for d in results["static_diagrams"].get(tn, []) for dim in d)
    st_t = results["static_times"].get(tn, 0)
    print(f"{tn:>6} {zz_n:>10} {zz_t:>9.1f}s {st_n:>12} {st_t:>11.1f}s")
PYEOF

echo ""
echo "=== Done ==="
echo "Results at: ${OUT_DIR}/persistence_results.pkl"
date
