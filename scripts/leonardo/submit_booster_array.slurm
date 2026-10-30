#!/bin/bash
# ==============================================================================
# CINECA Leonardo Booster: SLURM Job Array for Candidate Frame DFT Labeling
# ==============================================================================
# Usage:
#   sbatch --array=0-39%16 scripts/leonardo/submit_booster_array.slurm
# ==============================================================================
#SBATCH --job-name=hydeph_array
#SBATCH --account=${CINECA_ACCOUNT:-cin_staff}
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/array_%A_%a.out
#SBATCH --error=logs/array_%A_%a.err

set -e

# 1. Environment & Module Setup
module purge
module load profile/chem-phys
module load q-e/7.2
module load cuda/11.8
module load openmpi/4.1.4

ENV_DIR="${WORK:-$HOME}/hydeph_env"
if [ -d "${ENV_DIR}" ]; then
    eval "$(conda shell.bash hook 2>/dev/null || true)"
    conda activate "${ENV_DIR}"
fi

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

mkdir -p logs

echo "[HPC Array] Processing Task $SLURM_ARRAY_TASK_ID on $(hostname) with GPU: $CUDA_VISIBLE_DEVICES"

# 2. Execute HyDePh Array Worker for the assigned candidate frame
python -m hydeph.hpc.array_worker \
    --config "${CONFIG_PATH:-config.yaml}" \
    --task-id "$SLURM_ARRAY_TASK_ID" \
    --chunk-size "${CHUNK_SIZE:-1}"

echo "[HPC Array] Task $SLURM_ARRAY_TASK_ID finished at $(date)"
