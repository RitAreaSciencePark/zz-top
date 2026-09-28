#!/bin/bash
#SBATCH --job-name=zzparity3d
#SBATCH --partition=EPYC
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --output=<campaign dir>/parity_3d.slurm.out
#SBATCH --error=<campaign dir>/parity_3d.slurm.err
set -o pipefail
export OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=''
PY=~/miniconda3/envs/topofisher_gudhi_gpu/bin/python
cd <campaign dir> || { echo "FATAL: cannot cd to <campaign dir>" >&2; exit 2; }
$PY -u parity_fzz_3d.py <campaign dir>/zz-top-parity 3
echo "DONE rc=$?"
