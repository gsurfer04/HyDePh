"""Worker entrypoint executed by individual SLURM array tasks."""

from __future__ import annotations

import os
import sys
import argparse
import logging
from typing import Optional, List
import ase.io

from ..config import ToolkitConfig
from ..qe_runner import QERunner
from .slurm_dispatcher import mark_frame_done, is_frame_done, get_array_task_id

logger = logging.getLogger("hydeph.hpc.array_worker")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="HyDePh SLURM Array Worker for candidate frame labeling.")
    parser.add_argument("--config", required=True, help="Path to resolved HyDePh configuration file.")
    parser.add_argument("--task-id", type=int, default=None, help="SLURM array task ID (defaults to $SLURM_ARRAY_TASK_ID).")
    parser.add_argument("--chunk-size", type=int, default=1, help="Number of frames processed by this task.")
    parser.add_argument("--total-frames", type=int, default=None, help="Total candidate frames in pool.")

    args = parser.parse_args(argv)

    task_id = args.task_id if args.task_id is not None else get_array_task_id()
    if task_id is None:
        logger.error("No valid task ID provided via --task-id or $SLURM_ARRAY_TASK_ID.")
        return 1

    # Load configuration
    cfg = ToolkitConfig.load_from_yaml(args.config) if args.config.endswith((".yaml", ".yml")) else ToolkitConfig.load_from_json(args.config)
    workspace = cfg.workspace_dir
    cand_file = os.path.join(workspace, "candidate_frames.xyz")

    if not os.path.isfile(cand_file):
        logger.error(f"Candidate frames file not found: {cand_file}")
        return 1

    frames = ase.io.read(cand_file, index=":")
    if not isinstance(frames, list):
        frames = [frames]

    total_frames = args.total_frames or len(frames)
    start_idx = task_id * args.chunk_size
    end_idx = min(len(frames), (task_id + 1) * args.chunk_size)

    if start_idx >= len(frames):
        logger.info(f"Task ID {task_id} out of bounds for {len(frames)} frames. Exiting cleanly.")
        return 0

    target_frames = list(range(start_idx, end_idx))
    logger.info(f"[hpc worker] Task ID {task_id} processing frame indices: {target_frames} of {total_frames}")

    qe = QERunner(
        cfg,
        n_procs=cfg.n_procs_qe,
        pw_exe=cfg.pw_exe,
        mpi_command=cfg.mpi_command,
        mpi_args=cfg.mpi_args,
        n_workers=1,  # Inside an array task, single dedicated worker
        timeout_s=cfg.qe_timeout_s,
    )

    dft_dir = os.path.join(workspace, "dft_runs")
    os.makedirs(dft_dir, exist_ok=True)

    for idx in target_frames:
        frame = frames[idx]
        frame_dir = os.path.join(dft_dir, f"frame_{idx:04d}")

        # 1. Ground state calculation
        if is_frame_done(frame_dir):
            logger.info(f"Frame {idx} GS already completed with .done marker. Skipping.")
        else:
            logger.info(f"Executing Ground State DFT for frame {idx}...")
            at_res = qe.run_scf(frame, frame_dir)
            mark_frame_done(frame_dir, {"energy_eV": at_res.info.get("energy", 0.0), "mode": "gs"})

        # 2. Excited state calculation (if paired Delta-NEP enabled)
        if cfg.paired_delta_nep:
            es_dir = os.path.join(dft_dir, f"frame_{idx:04d}_es")
            if is_frame_done(es_dir):
                logger.info(f"Frame {idx} ES already completed with .done marker. Skipping.")
            else:
                logger.info(f"Executing Optical Excited State Delta-SCF for frame {idx}...")
                at_es_res = qe.run_delta_scf(frame, es_dir)
                mark_frame_done(es_dir, {"energy_eV": at_es_res.info.get("energy", 0.0), "mode": "es"})

    logger.info(f"[+] Task {task_id} successfully completed all assigned frames.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
