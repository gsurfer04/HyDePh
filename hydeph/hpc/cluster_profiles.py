"""Hardware and environment profiles for HPC clusters, specifically CINECA Leonardo."""

from __future__ import annotations

import os
import sys
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)


@dataclass
class ClusterProfile:
    """Hardware and scheduler configuration profile for a high-performance computing system.

    Parameters
    ----------
    name : str
        Canonical profile identifier (e.g. 'leonardo_booster', 'leonardo_dcgp', 'generic_slurm').
    description : str
        Human-readable description of the system architecture.
    partition : str
        Default SLURM partition name.
    qos : Optional[str]
        Default Quality of Service (QOS) identifier.
    gpus_per_node : int
        Number of GPUs per compute node (e.g. 4 for Leonardo Booster, 0 for DCGP).
    cpus_per_node : int
        Number of physical CPU cores per compute node.
    default_modules : List[str]
        List of environment modules to load before execution.
    gpu_type : Optional[str]
        Model identifier of the GPU accelerators (e.g. 'NVIDIA A100-SXM4-64GB').
    default_cpus_per_task : int
        Recommended CPU cores allocated per MPI/GPU worker task.
    memory_per_node_gb : int
        Total system RAM per node in GB.
    scratch_env_var : str
        Environment variable pointing to high-performance parallel scratch filesystem (e.g. 'SCRATCH').
    """

    name: str
    description: str
    partition: str
    qos: Optional[str] = None
    gpus_per_node: int = 0
    cpus_per_node: int = 32
    default_modules: List[str] = field(default_factory=list)
    gpu_type: Optional[str] = None
    default_cpus_per_task: int = 8
    memory_per_node_gb: int = 512
    scratch_env_var: str = "SCRATCH"

    @property
    def has_gpu(self) -> bool:
        """Returns True if this cluster profile features GPU acceleration."""
        return self.gpus_per_node > 0


# --- Pre-configured Profiles for CINECA Leonardo ---

LEONARDO_BOOSTER = ClusterProfile(
    name="leonardo_booster",
    description="CINECA Leonardo Booster: Atos BullSequana X2135, 1x Intel Xeon 8358 (32 cores) + 4x NVIDIA A100 64GB",
    partition="boost_usr_prod",
    qos=None,  # Default: omit --qos so Slurm automatically uses the account's configured QoS (or boost_qos_lprod)
    gpus_per_node=4,
    cpus_per_node=32,
    default_modules=[
        "profile/chem-phys",
        "q-e/7.2",
        "cuda/11.8",
        "openmpi/4.1.4",
    ],
    gpu_type="NVIDIA A100-SXM4-64GB",
    default_cpus_per_task=8,  # 32 cores / 4 GPUs = 8 cores per GPU worker
    memory_per_node_gb=512,
    scratch_env_var="SCRATCH",
)

LEONARDO_BOOSTER_DEBUG = ClusterProfile(
    name="leonardo_booster_debug",
    description="CINECA Leonardo Booster Debug Queue (max 2 nodes, 30 min walltime)",
    partition="boost_usr_prod",
    qos="boost_qos_dbg",
    gpus_per_node=4,
    cpus_per_node=32,
    default_modules=[
        "profile/chem-phys",
        "q-e/7.2",
        "cuda/11.8",
        "openmpi/4.1.4",
    ],
    gpu_type="NVIDIA A100-SXM4-64GB",
    default_cpus_per_task=8,
    memory_per_node_gb=512,
    scratch_env_var="SCRATCH",
)

LEONARDO_DCGP = ClusterProfile(
    name="leonardo_dcgp",
    description="CINECA Leonardo DCGP: 2x Intel Xeon Platinum 8480+ (112 cores total), CPU-only",
    partition="dcgp_usr_prod",
    qos=None,
    gpus_per_node=0,
    cpus_per_node=112,
    default_modules=[
        "profile/chem-phys",
        "q-e/7.2",
        "intel-oneapi-mkl",
        "openmpi/4.1.4",
    ],
    gpu_type=None,
    default_cpus_per_task=16,
    memory_per_node_gb=512,
    scratch_env_var="SCRATCH",
)

GENERIC_SLURM = ClusterProfile(
    name="generic_slurm",
    description="Generic SLURM cluster profile with standard resource allocation",
    partition="standard",
    qos=None,
    gpus_per_node=0,
    cpus_per_node=16,
    default_modules=[],
    gpu_type=None,
    default_cpus_per_task=4,
    memory_per_node_gb=64,
    scratch_env_var="SCRATCH",
)

_REGISTRY: Dict[str, ClusterProfile] = {
    "leonardo_booster": LEONARDO_BOOSTER,
    "leonardo": LEONARDO_BOOSTER,  # Alias: default Leonardo targets Booster
    "booster": LEONARDO_BOOSTER,
    "leonardo_booster_debug": LEONARDO_BOOSTER_DEBUG,
    "booster_dbg": LEONARDO_BOOSTER_DEBUG,
    "leonardo_dcgp": LEONARDO_DCGP,
    "dcgp": LEONARDO_DCGP,
    "generic_slurm": GENERIC_SLURM,
    "slurm": GENERIC_SLURM,
}


def get_cluster_profile(name: Optional[str] = None) -> ClusterProfile:
    """Retrieves a registered cluster profile by name with fallback to auto-detection.

    Parameters
    ----------
    name : str, optional
        Profile name (e.g. 'leonardo_booster', 'leonardo_dcgp'). If omitted or 'auto',
        attempts to infer the system from environment variables.

    Returns
    -------
    ClusterProfile
        The matching cluster profile.
    """
    if name and name.lower() != "auto":
        clean = name.strip().lower()
        if clean in _REGISTRY:
            return _REGISTRY[clean]
        logger.warning(f"Unknown cluster profile {name!r}. Falling back to LEONARDO_BOOSTER.")
        return LEONARDO_BOOSTER

    # Auto-detection from environment
    detected = detect_cluster_environment()
    if detected is not None:
        return detected

    return LEONARDO_BOOSTER


def detect_cluster_environment() -> Optional[ClusterProfile]:
    """Detects whether execution is running on CINECA Leonardo based on hostnames and environment."""
    hostname = os.environ.get("HOSTNAME", "").lower()
    slurm_cluster = os.environ.get("SLURM_CLUSTER_NAME", "").lower()

    if "leonardo" in hostname or "leonardo" in slurm_cluster or "cineca" in hostname:
        # Check if running in a booster or dcgp partition
        partition = os.environ.get("SLURM_JOB_PARTITION", "").lower()
        if "dcgp" in partition:
            return LEONARDO_DCGP
        qos = os.environ.get("SLURM_JOB_QOS", "").lower()
        if "dbg" in qos:
            return LEONARDO_BOOSTER_DEBUG
        return LEONARDO_BOOSTER

    # Check for CINECA specific environment indicators
    if "CINECA_SCRATCH" in os.environ or "CINECA_WORK" in os.environ:
        return LEONARDO_BOOSTER

    return None
