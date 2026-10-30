#!/bin/bash
# ==============================================================================
# CINECA Leonardo Booster: Interactive Resource Allocation via salloc
# ==============================================================================
# CINECA officially recommends salloc over srun for interactive sessions:
# https://docs.hpc.cineca.it/hpc/hpc_scheduler.html
# https://docs.hpc.cineca.it/hpc/leonardo.html
#
# Usage:
#   bash scripts/leonardo/interactive_booster.sh [--debug] [time_limit]
#
# Examples:
#   bash scripts/leonardo/interactive_booster.sh             # 1 hour standard production
#   bash scripts/leonardo/interactive_booster.sh --debug     # 30 min debug queue (fast launch)
#   bash scripts/leonardo/interactive_booster.sh 04:00:00    # 4 hours production
# ==============================================================================

ACCOUNT="${CINECA_ACCOUNT:-cin_staff}"
TIME_LIMIT="01:00:00"
QOS="boost_qos_prod"

for arg in "$@"; do
    if [ "$arg" == "--debug" ] || [ "$arg" == "-d" ]; then
        QOS="boost_qos_dbg"
        TIME_LIMIT="00:30:00"
    elif [[ "$arg" =~ ^[0-9]{2}:[0-9]{2}:[0-9]{2}$ ]]; then
        TIME_LIMIT="$arg"
    fi
done

echo "=============================================================================="
echo " Requesting Interactive Leonardo Booster Allocation (salloc)"
echo " Account:   ${ACCOUNT}"
echo " Partition: boost_usr_prod"
echo " QOS:       ${QOS}"
echo " Resources: 1 Node, 4x NVIDIA A100 GPUs, 32 CPUs (8 per GPU)"
echo " Walltime:  ${TIME_LIMIT}"
echo "=============================================================================="
echo "Note: Once granted, you will receive an interactive shell inside your allocation."
echo "      To run HyDePh, source your environment and execute python:"
echo "        source scripts/leonardo/env_leonardo.sh"
echo "        python -m hydeph --config config.yaml"
echo "=============================================================================="

salloc \
    --account="${ACCOUNT}" \
    --partition=boost_usr_prod \
    --qos="${QOS}" \
    --nodes=1 \
    --ntasks-per-node=4 \
    --cpus-per-task=8 \
    --gres=gpu:4 \
    --time="${TIME_LIMIT}"
