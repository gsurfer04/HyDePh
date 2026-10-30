#!/bin/bash
# ==============================================================================
# CINECA Leonardo DCGP: Master HyDePh End-to-End Workflow Execution (CPU)
# ==============================================================================
#SBATCH --job-name=hydeph_dcgp
#SBATCH --account=${CINECA_ACCOUNT:-cin_staff}
#SBATCH --partition=dcgp_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=7
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=hydeph_dcgp_%j.out
#SBATCH --error=hydeph_dcgp_%j.err

set -e

echo "=============================================================================="
echo " Starting HyDePh on CINECA Leonardo DCGP (CPU)"
echo " Host: $(hostname) | Job ID: $SLURM_JOB_ID | Start: $(date)"
echo "=============================================================================="

# 1. Environment & Module Setup
module purge
module load profile/chem-phys
module load q-e/7.2
module load intel-oneapi-mkl
module load openmpi/4.1.4

ENV_DIR="${WORK:-$HOME}/hydeph_env"
if [ -d "${ENV_DIR}" ]; then
    eval "$(conda shell.bash hook 2>/dev/null || true)"
    conda activate "${ENV_DIR}"
fi

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

WORKSPACE_DIR="${SCRATCH:-.}/hydeph_run_${SLURM_JOB_ID}"
mkdir -p "${WORKSPACE_DIR}"

# 2. Launch HyDePh Workflow across 7 concurrent workers (16 cores each = 112 cores)
python -m hydeph \
    --config config.yaml \
    --workspace "${WORKSPACE_DIR}" \
    --n-workers-qe 7 \
    --n-procs-qe 1

echo "=============================================================================="
echo " HyDePh Finished Successfully on Leonardo DCGP at $(date)"
echo "=============================================================================="
