#!/bin/bash
# ==============================================================================
# CINECA Leonardo DCGP: Interactive Resource Allocation via salloc (CPU)
# ==============================================================================
# Usage:
#   bash scripts/leonardo/interactive_dcgp.sh [time_limit]
# ==============================================================================

ACCOUNT="${CINECA_ACCOUNT:-cin_staff}"
TIME_LIMIT="${1:-01:00:00}"

echo "=============================================================================="
echo " Requesting Interactive Leonardo DCGP Allocation (salloc)"
echo " Account:   ${ACCOUNT}"
echo " Partition: dcgp_usr_prod"
echo " Resources: 1 Node, 112 CPU cores (7 tasks x 16 cpus)"
echo " Walltime:  ${TIME_LIMIT}"
echo "=============================================================================="

salloc \
    --account="${ACCOUNT}" \
    --partition=dcgp_usr_prod \
    --nodes=1 \
    --ntasks-per-node=7 \
    --cpus-per-task=16 \
    --time="${TIME_LIMIT}"
