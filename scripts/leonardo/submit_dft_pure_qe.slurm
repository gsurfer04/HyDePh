#!/bin/bash
# ==============================================================================
# Standalone Quantum ESPRESSO DFT Runner on CINECA Leonardo Booster
# ==============================================================================
# Executes candidate frame DFT calculations across 4x NVIDIA A100 GPUs.
# This script does NOT require Python or any Conda environment on compute nodes!
# ==============================================================================
#SBATCH --job-name=hydeph_pw_dft
#SBATCH --account=${CINECA_ACCOUNT:-cin_staff}
#SBATCH --partition=boost_usr_prod
# Note: --qos is omitted by default so Slurm uses your account default QoS (e.g. boost_qos_lprod).
# For short debug jobs (<= 30 min), use: #SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00
#SBATCH --output=hydeph_dft_%j.out
#SBATCH --error=hydeph_dft_%j.err

set -e

echo "=============================================================================="
echo " Starting Pure Quantum ESPRESSO Calculations on CINECA Leonardo Booster"
echo " Host: $(hostname) | Job ID: $SLURM_JOB_ID | Start: $(date)"
echo "=============================================================================="

# Load native CINECA modules - NO Python or Conda needed!
module purge
module load profile/chem-phys
module load q-e/7.2
module load cuda/11.8

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

PW_CMD="pw.x"

# Collect all frame directories containing pw.in
DIRS=()
if [ -d "dft_calculations" ]; then
    while IFS= read -r d; do
        DIRS+=("$d")
    done < <(find dft_calculations -mindepth 1 -maxdepth 1 -type d | sort)
fi
if [ -d "dft_calculations_es" ]; then
    while IFS= read -r d; do
        DIRS+=("$d")
    done < <(find dft_calculations_es -mindepth 1 -maxdepth 1 -type d | sort)
fi

TOTAL=${#DIRS[@]}
echo "[+] Discovered ${TOTAL} candidate frame calculations."

if [ $TOTAL -eq 0 ]; then
    echo "[!] Error: No candidate frame directories found with pw.in!"
    echo "    Please generate input decks on the login node first via:"
    echo "        python -m hydeph run config.yaml --to-stage 2"
    exit 1
fi

MAX_CONCURRENT=4
RUNNING=0

for i in "${!DIRS[@]}"; do
    DIR="${DIRS[$i]}"
    IN_FILE="$DIR/pw.in"
    OUT_FILE="$DIR/pw.out"

    if [ ! -f "$IN_FILE" ]; then
        continue
    fi

    # Skip already converged calculations
    if [ -f "$OUT_FILE" ] && grep -q "JOB DONE." "$OUT_FILE" 2>/dev/null && ! grep -q "convergence NOT achieved" "$OUT_FILE" 2>/dev/null; then
        echo "[SKIP] Frame $((i+1))/$TOTAL ($DIR) is already converged."
        continue
    fi

    GPU_ID=$(( i % MAX_CONCURRENT ))
    echo "[RUN]  Frame $((i+1))/$TOTAL ($DIR) on GPU $GPU_ID..."

    (
        export CUDA_VISIBLE_DEVICES=$GPU_ID
        cd "$DIR"
        $PW_CMD -in pw.in > pw.out 2>&1
    ) &

    RUNNING=$((RUNNING + 1))
    if [ $RUNNING -ge $MAX_CONCURRENT ]; then
        wait -n 2>/dev/null || wait
        RUNNING=$((RUNNING - 1))
    fi
done

wait
echo "=============================================================================="
echo " All DFT calculations finished at $(date)"
echo " Next step: On the login node, resume post-processing with:"
echo "     python -m hydeph run config.yaml --from-stage 4"
echo "=============================================================================="
