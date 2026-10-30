#!/bin/bash
# ==============================================================================
# Helper Script to Submit Unattended Batch Job on CINECA Leonardo Booster
# ==============================================================================
# This does NOT require keeping the SSH connection open. You can disconnect
# immediately after running this script.
#
# Usage:
#   bash scripts/leonardo/submit_booster_batch.sh [--debug] [time_limit] [-A=<account>]
# ==============================================================================

ACCOUNT="${CINECA_ACCOUNT:-cin_staff}"
TIME_LIMIT="06:00:00"
QOS_OPT=""

for arg in "$@"; do
    if [ "$arg" == "--debug" ] || [ "$arg" == "-d" ]; then
        QOS_OPT="--qos=boost_qos_dbg"
        TIME_LIMIT="00:30:00"
    elif [[ "$arg" =~ ^--qos= ]]; then
        QOS_OPT="$arg"
    elif [[ "$arg" =~ ^[0-9]{2}:[0-9]{2}:[0-9]{2}$ ]]; then
        TIME_LIMIT="$arg"
    elif [[ "$arg" =~ ^-A= ]] || [[ "$arg" =~ ^--account= ]]; then
        ACCOUNT="${arg#*=}"
    fi
done

echo "=============================================================================="
echo " Submitting Unattended HyDePh Batch Job to Leonardo Booster"
echo " Account:   ${ACCOUNT}"
echo " Partition: boost_usr_prod (QOS: ${QOS_OPT:-default})"
echo " Resources: 1 Node, 4x NVIDIA A100 GPUs, 32 CPU cores"
echo " Walltime:  ${TIME_LIMIT}"
echo "=============================================================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB_ID=$(sbatch \
    --account="${ACCOUNT}" \
    ${QOS_OPT} \
    --time="${TIME_LIMIT}" \
    --parsable \
    "${SCRIPT_DIR}/submit_booster_workflow.slurm")

if [ $? -eq 0 ] && [ -n "$JOB_ID" ]; then
    echo "[+] Successfully submitted SLURM Job ID: ${JOB_ID}"
    echo ""
    echo "You may now safely close your SSH session or laptop."
    echo ""
    echo "Useful monitoring commands:"
    echo "  • Check job status:      squeue -j ${JOB_ID}"
    echo "  • Monitor live output:   tail -f hydeph_${JOB_ID}.out"
    echo "  • Cancel job if needed:  scancel ${JOB_ID}"
    echo "=============================================================================="
else
    echo "[!] Submission failed. Please verify your account ($ACCOUNT) and partition access."
fi
