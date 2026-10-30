#!/bin/bash
# ==============================================================================
# CINECA Leonardo Environment Setup & Verification Script for HyDePh
# ==============================================================================
# Usage:
#   source scripts/leonardo/env_leonardo.sh
# ==============================================================================

echo "=============================================================================="
echo "          CINECA Leonardo Environment Configuration for HyDePh               "
echo "=============================================================================="

# 1. Module purge & loading
echo "[-] Purging existing modules and loading CINECA Leonardo profiles..."
module purge
module load profile/chem-phys
module load q-e/7.2
module load cuda/11.8
module load openmpi/4.1.4

echo "[+] Loaded modules:"
module list

# 2. Check GPU availability (Booster nodes)
if command -v nvidia-smi &> /dev/null; then
    echo -e "\n[+] NVIDIA GPUs detected on host $(hostname):"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
else
    echo -e "\n[i] No GPU detected directly on login node (normal; compute nodes feature 4x A100 GPUs)."
fi

# 3. Setup Python / Conda environment
ENV_DIR="${WORK:-$HOME}/hydeph_env"
echo -e "\n[-] Checking Conda environment in: ${ENV_DIR}..."

if [ ! -d "${ENV_DIR}" ]; then
    echo "[!] Environment not found. Creating conda environment in ${ENV_DIR}..."
    module load python/3.11 || true
    conda create -p "${ENV_DIR}" python=3.11 -y
    eval "$(conda shell.bash hook)"
    conda activate "${ENV_DIR}"
    echo "[-] Installing dependencies..."
    pip install --upgrade pip
    pip install numpy scipy matplotlib ase calorine pyyaml tqdm
    pip install -e .
    echo "[+] HyDePh environment successfully installed!"
else
    echo "[+] Found existing environment at ${ENV_DIR}. Activating..."
    eval "$(conda shell.bash hook)"
    conda activate "${ENV_DIR}"
fi

# 4. Set OpenMP and MPI variables
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8

echo -e "\n[+] Leonardo environment ready for HyDePh workflows!"
echo "=============================================================================="
