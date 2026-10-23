"""End-to-end Delta-NEP training and optical lineshape pipeline."""

from __future__ import annotations

import os
import sys
import time
import argparse
import logging
from typing import Optional, Dict, Any, List
import numpy as np
import ase.io

from .config import ToolkitConfig
from .defect_builder import DefectBuilder
from .frame_sampler import FrameSampler
from .qe_runner import QERunner
from .delta_trainer import DeltaTrainer
from .calculator import HybridDefectCalculator
from .lineshape import LineshapeEngine

logger = logging.getLogger(__name__)


def run_pipeline(
    config: ToolkitConfig,
    max_dft_frames: Optional[int] = None,
    n_procs_qe: Optional[int] = None,
    n_workers_qe: Optional[int] = None,
) -> Dict[str, Any]:
    """Runs the complete Delta-NEP defect workflow and returns a summary dictionary."""
    t_start = time.time()
    eff_max_dft = max_dft_frames if max_dft_frames is not None else config.max_dft_frames
    eff_n_procs = n_procs_qe if n_procs_qe is not None else config.n_procs_qe
    eff_n_workers = n_workers_qe if n_workers_qe is not None else config.n_workers_qe

    logger.info("=" * 78)
    logger.info("      AUTOMATED COLOR-CENTER DELTA-NEP WORKFLOW PIPELINE")
    logger.info("=" * 78)
    logger.info(f"Workspace: {os.path.abspath(config.workspace_dir)}")
    logger.info(f"Foundation Potential: {config.foundation_potential_path}")
    logger.info(f"Temperatures of interest: {config.temperatures} K")
    logger.info(f"Pseudopotentials: {config.pseudopotentials}")
    logger.info("=" * 78)

    config.validate()

    logger.info("\n[STEP 1/6] Defect Construction & Geometry Optimization...")
    builder = DefectBuilder(config)
    shared_foundation = builder.calc_foundation
    pristine_sc, defect_sc = builder.build_supercell_and_defect()
    at_gs = builder.relax_ground_state(defect_sc)
    at_es, dq_fc = builder.generate_excited_state_geometry(at_gs)

    logger.info("\n[STEP 2/6] Representative Frame Generation Protocol...")
    sampler = FrameSampler(config, calc_foundation=shared_foundation)
    candidate_frames = sampler.assemble_candidate_dataset(at_gs, at_es)

    logger.info("\n[STEP 3/6] High-Fidelity DFT Labeling (Quantum ESPRESSO)...")
    qe = QERunner(
        config,
        n_procs=eff_n_procs,
        pw_exe=config.pw_exe,
        mpi_command=config.mpi_command,
        mpi_args=config.mpi_args,
        n_workers=eff_n_workers,
        timeout_s=config.qe_timeout_s,
    )
    labeled_frames = qe.label_all_frames(
        candidate_frames,
        max_frames=eff_max_dft,
        sampler=sampler,
        at_gs=at_gs,
        n_workers=eff_n_workers,
    )

    # Attach DFT ground-state energy and Kohn-Sham optical transition properties to at_gs and at_es
    gs_cand = next(
        (f for f in labeled_frames if f.info.get("sample_type") == "franck_condon" and abs(f.info.get("lambda", 1.0)) < 1e-4),
        None,
    )
    if gs_cand is None and labeled_frames:
        gs_cand = labeled_frames[0]
    if gs_cand is not None:
        if "energy" in gs_cand.info:
            at_gs.info["energy"] = float(gs_cand.info["energy"])
        for k in ("dft_gap_eV", "dft_homo_eV", "dft_lumo_eV", "dft_fermi_eV"):
            if k in gs_cand.info:
                at_gs.info[k] = float(gs_cand.info[k])

    es_cand = next(
        (f for f in labeled_frames if f.info.get("sample_type") == "franck_condon" and abs(f.info.get("lambda", 0.0) - 1.0) < 1e-4),
        None,
    )
    if es_cand is not None and "energy" in es_cand.info:
        at_es.info["energy"] = float(es_cand.info["energy"])

    logger.info("\n[STEP 4/6] Delta Baseline Extraction & GPU GPUMD Training...")
    trainer = DeltaTrainer(config, calc_foundation=shared_foundation)
    delta_frames = trainer.compute_delta_dataset(labeled_frames)
    delta_pot_path = trainer.train_delta_nep(delta_frames)

    logger.info("\n[STEP 5/6] Assembling Hybrid Defect Potential (V_base + Delta_V)...")
    hybrid_calc = HybridDefectCalculator(
        config.foundation_potential_path, delta_pot_path, calc_base=shared_foundation
    )

    at_test = at_gs.copy()
    at_test.calc = hybrid_calc
    e_hybrid = at_test.get_potential_energy()
    f_hybrid = at_test.get_forces()
    max_f = float(np.abs(f_hybrid).max())
    logger.info(f"[+] Hybrid Potential verified on relaxed GS: E = {e_hybrid:.4f} eV, Max F = {max_f:.4f} eV/A")

    if max_f > 0.015:
        logger.info("[-] Refining Ground State equilibrium on Hybrid Potential...")
        from ase.optimize import BFGS
        at_gs_hybrid = at_gs.copy()
        at_gs_hybrid.calc = hybrid_calc
        opt = BFGS(at_gs_hybrid, logfile=None)
        opt.run(fmax=0.01, steps=50)
        at_gs_hybrid.info = dict(at_gs.info)
        dr_fc = mic_displacement(at_es.positions, at_gs.positions, at_gs.cell)
        at_es_hybrid = at_gs_hybrid.copy()
        at_es_hybrid.positions = at_gs_hybrid.positions + dr_fc
        at_es_hybrid.info = dict(at_es.info)
        e_opt = at_gs_hybrid.get_potential_energy()
        f_opt = float(np.abs(at_gs_hybrid.get_forces()).max())
        logger.info(f"[+] Refined GS equilibrium on Hybrid Potential: E = {e_opt:.4f} eV, Max F = {f_opt:.4f} eV/A")
        at_gs = at_gs_hybrid
        at_es = at_es_hybrid

    logger.info("\n[STEP 6/6] Normal Modes & Multi-Temperature Optical Lineshapes...")
    engine = LineshapeEngine(config, hybrid_calc)
    lineshapes = engine.calculate_spectral_density_and_lineshapes(at_gs, at_es)

    total_time = time.time() - t_start
    logger.info("\n" + "=" * 78)
    logger.info(f"      WORKFLOW COMPLETE (Total Walltime: {total_time:.1f} s / {total_time/60.0:.2f} min)")
    logger.info("=" * 78)

    cc_analysis = lineshapes.get("cc_analysis") if isinstance(lineshapes, dict) else None
    cc_figure = cc_analysis.get("figure_path") if cc_analysis else None
    failed_dft = [e["frame_id"] for e in getattr(qe, "last_failed_entries", [])]

    summary = {
        "workspace_dir": config.workspace_dir,
        "delta_potential": delta_pot_path,
        "n_dft_frames": len(labeled_frames),
        "n_failed_dft_frames": len(failed_dft),
        "failed_dft_frames": failed_dft,
        "dq_fc": dq_fc,
        "temperatures": config.temperatures,
        "total_walltime_s": total_time,
        "lineshapes": lineshapes,
        "stokes_analysis": cc_analysis,
        "cc_figure": cc_figure,
    }
    return summary


def main(argv: Optional[List[str]] = None) -> int:
    """Command-line interface entry point for executing HyDePh workflows.
    
    Usage:
        hydeph config.yaml
        hydeph config.toml
    """
    parser = argparse.ArgumentParser(
        prog="hydeph",
        description="HyDePh: High-Definition Hybrid Defect Photoluminescence & Anharmonic Spectroscopy",
    )
    parser.add_argument(
        "config_path",
        help="Path to YAML, TOML, or JSON configuration file (e.g. config.yaml)",
    )
    parser.add_argument(
        "--max-dft",
        type=int,
        default=None,
        help="Optional override for maximum DFT frames to label",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Optional override for number of concurrent DFT workers",
    )
    args = parser.parse_args(argv)

    cfg = ToolkitConfig.from_file(args.config_path)
    if args.max_dft is not None:
        cfg.max_dft_frames = args.max_dft
    if args.workers is not None:
        cfg.n_workers_qe = args.workers

    run_pipeline(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
