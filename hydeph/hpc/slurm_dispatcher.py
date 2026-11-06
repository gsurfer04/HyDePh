"""SLURM Job Array dispatch, multi-GPU pinning, and Lustre-safe atomic synchronization."""

from __future__ import annotations

import os
import sys
import json
import time
import socket
import logging
from typing import Optional, Dict, Any, List, Tuple
import ase.io
from ase import Atoms

from .cluster_profiles import ClusterProfile, get_cluster_profile

logger = logging.getLogger(__name__)


def is_slurm_active() -> bool:
    """Returns True if the current process is running inside a SLURM job allocation."""
    return bool(os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_ARRAY_TASK_ID"))


def get_slurm_job_id() -> Optional[str]:
    """Returns the current SLURM job ID, or None if not running in SLURM."""
    return os.environ.get("SLURM_JOB_ID")


def get_array_task_id() -> Optional[int]:
    """Returns the current SLURM array task ID ($SLURM_ARRAY_TASK_ID) as an int, or None."""
    val = os.environ.get("SLURM_ARRAY_TASK_ID")
    if val is not None and val.isdigit():
        return int(val)
    return None


def get_pinned_worker_env(
    worker_idx: int,
    gpus_per_node: int = 4,
    cpus_per_worker: int = 8,
    base_env: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """Generates an environment dictionary with CUDA and OpenMP pinning for a specific worker.

    On Leonardo Booster, each compute node possesses 4 NVIDIA A100 GPUs and 32 CPU cores.
    Each concurrent worker (0..3) is pinned to a dedicated GPU (0..3) and allocated
    8 OpenMP threads.

    Parameters
    ----------
    worker_idx : int
        Index of the concurrent worker process (0, 1, ...).
    gpus_per_node : int
        Total physical GPUs available on the compute node (default: 4).
    cpus_per_worker : int
        Number of CPU cores allocated per worker (default: 8).
    base_env : Dict[str, str], optional
        Base environment dictionary. Defaults to os.environ.copy().

    Returns
    -------
    Dict[str, str]
        Updated environment dictionary with CUDA_VISIBLE_DEVICES and OMP_NUM_THREADS set.
    """
    env = dict(os.environ if base_env is None else base_env)

    if gpus_per_node > 0:
        gpu_id = int(worker_idx % gpus_per_node)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env["GPU_DEVICE_ORDINAL"] = str(gpu_id)
        logger.debug(f"[hpc] Worker {worker_idx} pinned to GPU {gpu_id} (CUDA_VISIBLE_DEVICES={gpu_id})")

    env["OMP_NUM_THREADS"] = str(cpus_per_worker)
    env["MKL_NUM_THREADS"] = str(cpus_per_worker)
    env["OPENBLAS_NUM_THREADS"] = str(cpus_per_worker)
    return env


def mark_frame_done(calc_dir: str, metadata: Optional[Dict[str, Any]] = None) -> None:
    """Atomically creates a `.done` marker file in calc_dir to certify calculation completion.

    This prevents race conditions on shared parallel filesystems (such as Lustre on Leonardo)
    when multiple array tasks or worker processes are polling calculation status.

    Parameters
    ----------
    calc_dir : str
        Path to calculation directory.
    metadata : Dict[str, Any], optional
        Optional execution metadata (walltime, hostname, job ID, energy) saved to `.done`.
    """
    os.makedirs(calc_dir, exist_ok=True)
    done_file = os.path.join(calc_dir, ".done")
    tmp_file = os.path.join(calc_dir, f".done.tmp.{os.getpid()}.{time.time()}")

    payload = {
        "timestamp": time.time(),
        "hostname": socket.gethostname(),
        "slurm_job_id": get_slurm_job_id(),
        "slurm_array_task_id": get_array_task_id(),
        "metadata": metadata or {},
    }

    try:
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        # Atomic rename on POSIX filesystems
        os.replace(tmp_file, done_file)
    except Exception as exc:
        logger.warning(f"Could not write atomic .done marker in {calc_dir}: {exc}")
        if os.path.exists(tmp_file):
            try:
                os.remove(tmp_file)
            except OSError:
                pass


def is_frame_done(calc_dir: str) -> bool:
    """Checks whether a calculation in calc_dir has successfully finished with a `.done` marker.

    Parameters
    ----------
    calc_dir : str
        Path to calculation directory.

    Returns
    -------
    bool
        True if the .done marker exists and is valid.
    """
    done_file = os.path.join(calc_dir, ".done")
    return os.path.isfile(done_file)


class SlurmDispatcher:
    """SLURM job array generator and multi-GPU task manager for CINECA Leonardo."""

    def __init__(
        self,
        profile: Optional[ClusterProfile] = None,
        account: Optional[str] = None,
        workspace_dir: str = "./hydeph_run",
        slurm_modules: Optional[List[str]] = None,
    ):
        self.profile = profile or get_cluster_profile("leonardo_booster")
        self.account = (
            account
            or os.environ.get("CINECA_ACCOUNT")
            or os.environ.get("SLURM_ACCOUNT")
            or "cin_staff"
        )
        self.workspace_dir = os.path.abspath(workspace_dir)
        self.slurm_modules = list(slurm_modules) if slurm_modules is not None else None

    def generate_job_array_script(
        self,
        config_path: str,
        n_frames: int,
        chunk_size: int = 1,
        max_concurrent: int = 16,
        time_limit: str = "02:00:00",
        script_output_path: Optional[str] = None,
        conda_env_name: Optional[str] = "hydeph",
    ) -> str:
        """Generates a production-ready SLURM Job Array bash script for parallel frame labeling.

        Parameters
        ----------
        config_path : str
            Path to the JSON or YAML configuration file.
        n_frames : int
            Total number of candidate frames to label.
        chunk_size : int
            Number of frames evaluated sequentially by a single array task (default: 1).
        max_concurrent : int
            Maximum number of concurrent SLURM array tasks (e.g. %16).
        time_limit : str
            SLURM walltime per task (e.g. '02:00:00').
        script_output_path : str, optional
            File path to save the generated script.
        conda_env_name : str, optional
            Name or path of the conda environment to activate.

        Returns
        -------
        str
            Full contents of the generated SLURM job array script.
        """
        n_tasks = (n_frames + chunk_size - 1) // chunk_size
        max_idx = max(0, n_tasks - 1)

        mods = self.slurm_modules if self.slurm_modules is not None else []
        if not mods:
            logger.warning(
                "[!] Warning: No cluster environment modules specified ('modules' or 'slurm_modules' is empty). "
                "Cluster jobs may fail to find required binaries unless loaded in the shell environment."
            )
        module_lines = "\n".join([f"module load {str(m).strip()}" for m in mods if str(m).strip()])
        qos_line = f"#SBATCH --qos={self.profile.qos}" if self.profile.qos else ""
        gres_line = f"#SBATCH --gres=gpu:1" if self.profile.has_gpu else ""

        script_content = f"""#!/bin/bash
# ==============================================================================
# SLURM Job Array for HyDePh Candidate Frame Labeling
# Generated automatically by HyDePh SlurmDispatcher
# Profile: {self.profile.description}
# ==============================================================================
#SBATCH --job-name=hydeph_array
#SBATCH --account={self.account}
#SBATCH --partition={self.profile.partition}
{qos_line}
#SBATCH --array=0-{max_idx}%{max_concurrent}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={self.profile.default_cpus_per_task}
{gres_line}
#SBATCH --time={time_limit}
#SBATCH --output={self.workspace_dir}/logs/array_%A_%a.out
#SBATCH --error={self.workspace_dir}/logs/array_%A_%a.err

set -e

# 1. Environment & Module Setup
module purge
{module_lines}

# 2. Activate Conda Environment
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate {conda_env_name}
elif [ -d "$WORK/{conda_env_name}" ]; then
    source "$WORK/{conda_env_name}/bin/activate"
fi

export OMP_NUM_THREADS={self.profile.default_cpus_per_task}
export MKL_NUM_THREADS={self.profile.default_cpus_per_task}
mkdir -p "{self.workspace_dir}/logs"

echo "[HPC Array] Task $SLURM_ARRAY_TASK_ID on $(hostname) with GPU: $CUDA_VISIBLE_DEVICES"

# 3. Execute HyDePh Array Worker for the assigned frame(s)
python -m hydeph.hpc.array_worker \\
    --config "{os.path.abspath(config_path)}" \\
    --task-id "$SLURM_ARRAY_TASK_ID" \\
    --chunk-size "{chunk_size}" \\
    --total-frames "{n_frames}"

echo "[HPC Array] Task $SLURM_ARRAY_TASK_ID completed successfully at $(date)"
"""
        if script_output_path:
            os.makedirs(os.path.dirname(os.path.abspath(script_output_path)), exist_ok=True)
            with open(script_output_path, "w", encoding="utf-8") as f:
                f.write(script_content)
            logger.info(f"[+] Saved SLURM job array script to: {script_output_path}")

        return script_content

    def generate_salloc_command(
        self,
        nodes: int = 1,
        time_limit: str = "01:00:00",
        debug: bool = False,
    ) -> str:
        """Generates an interactive resource allocation command using salloc for the active profile."""
        return generate_salloc_command(
            profile=self.profile,
            account=self.account,
            nodes=nodes,
            time_limit=time_limit,
            debug=debug,
        )


def generate_salloc_command(
    profile: Optional[ClusterProfile] = None,
    account: Optional[str] = None,
    nodes: int = 1,
    time_limit: str = "01:00:00",
    debug: bool = False,
) -> str:
    """Generates an interactive resource allocation command using salloc, adhering to CINECA Leonardo guidance.

    On CINECA clusters like Leonardo, salloc is officially recommended over srun for interactive sessions.
    The scheduler allocates the requested compute nodes; commands executed inside the allocation session
    subsequently dispatch parallel tasks to compute nodes using srun.

    Parameters
    ----------
    profile : ClusterProfile, optional
        Target cluster profile (default: LEONARDO_BOOSTER).
    account : str, optional
        CINECA project account identifier (e.g. 'cin_staff').
    nodes : int
        Number of compute nodes to reserve (default: 1).
    time_limit : str
        Walltime allocation limit (e.g. '01:00:00' or '00:30:00').
    debug : bool
        If True, selects the short debug QOS ('boost_qos_dbg' on Booster).

    Returns
    -------
    str
        The formatted salloc command string.
    """
    prof = profile or get_cluster_profile("leonardo_booster")
    acct = account or os.environ.get("CINECA_ACCOUNT") or os.environ.get("SLURM_ACCOUNT") or "cin_staff"

    qos = "boost_qos_dbg" if debug and "booster" in prof.name else prof.qos
    qos_part = f"--qos={qos} " if qos else ""
    gres_part = f"--gres=gpu:{prof.gpus_per_node} " if prof.has_gpu else ""
    ntasks_node = prof.gpus_per_node if prof.has_gpu else max(1, prof.cpus_per_node // prof.default_cpus_per_task)

    return (
        f"salloc -A {acct} -p {prof.partition} {qos_part}"
        f"--nodes={nodes} --ntasks-per-node={ntasks_node} "
        f"--cpus-per-task={prof.default_cpus_per_task} {gres_part}"
        f"--time={time_limit}"
    ).strip()

