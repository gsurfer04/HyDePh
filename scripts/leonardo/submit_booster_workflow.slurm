#!/bin/bash
# ==============================================================================
# CINECA Leonardo Booster: Master HyDePh End-to-End Workflow Execution
# ==============================================================================
#SBATCH --job-name=hydeph_booster
#SBATCH --account=${CINECA_ACCOUNT:-cin_staff}
#SBATCH --partition=boost_usr_prod
# Note: --qos is omitted by default so Slurm uses your account's default QoS (e.g. boost_qos_lprod).
# For short debug jobs (<= 30 min), use: #SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --time=06:00:00
#SBATCH --output=hydeph_%j.out
#SBATCH --error=hydeph_%j.err

set -e

echo "=============================================================================="
echo " Starting HyDePh on CINECA Leonardo Booster"
echo " Host: $(hostname) | Job ID: $SLURM_JOB_ID | Start: $(date)"
echo "=============================================================================="

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
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

# 2. Set scratch workspace
WORKSPACE_DIR="${SCRATCH:-.}/hydeph_run_${SLURM_JOB_ID}"
mkdir -p "${WORKSPACE_DIR}"

echo "[+] Using workspace: ${WORKSPACE_DIR}"
echo "[+] Visible GPUs:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# 3. Launch HyDePh Workflow with 4 concurrent GPU workers
python -m hydeph \
    --config config.yaml \
    --workspace "${WORKSPACE_DIR}" \
    --n-workers-qe 4 \
    --n-procs-qe 1

echo "=============================================================================="
echo " HyDePh Finished Successfully on Leonardo Booster at $(date)"
echo "=============================================================================="
