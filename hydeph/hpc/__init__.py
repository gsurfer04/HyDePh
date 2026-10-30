"""High-Performance Computing (HPC) and parallelization subsystem for HyDePh."""

from .cluster_profiles import (
    ClusterProfile,
    LEONARDO_BOOSTER,
    LEONARDO_BOOSTER_DEBUG,
    LEONARDO_DCGP,
    GENERIC_SLURM,
    get_cluster_profile,
    detect_cluster_environment,
)
from .slurm_dispatcher import (
    SlurmDispatcher,
    is_slurm_active,
    get_slurm_job_id,
    get_array_task_id,
    get_pinned_worker_env,
    mark_frame_done,
    is_frame_done,
)
from .parallel_vib import ParallelVibrations

__all__ = [
    "ClusterProfile",
    "LEONARDO_BOOSTER",
    "LEONARDO_BOOSTER_DEBUG",
    "LEONARDO_DCGP",
    "GENERIC_SLURM",
    "get_cluster_profile",
    "detect_cluster_environment",
    "SlurmDispatcher",
    "is_slurm_active",
    "get_slurm_job_id",
    "get_array_task_id",
    "get_pinned_worker_env",
    "mark_frame_done",
    "is_frame_done",
    "ParallelVibrations",
]
