"""End-to-end Delta-NEP training and optical lineshape pipeline."""

import os
import time
import numpy as np
import ase.io

from .config import ToolkitConfig
from .defect_builder import DefectBuilder
from .frame_sampler import FrameSampler
from .qe_runner import QERunner
from .delta_trainer import DeltaTrainer
from .calculator import HybridDefectCalculator
from .lineshape import LineshapeEngine


def run_pipeline(config: ToolkitConfig, max_dft_frames: int = None, n_procs_qe: int = 2, n_workers_qe: int = None):
    """Runs the Delta-NEP defect workflow and returns a summary dictionary."""
    t_start = time.time()
    print("=" * 78)
    print("      AUTOMATED COLOR-CENTER DELTA-NEP WORKFLOW PIPELINE")
    print("=" * 78)
    print(f"Workspace: {os.path.abspath(config.workspace_dir)}")
    print(f"Foundation Potential: {config.foundation_potential_path}")
    print(f"Temperatures of interest: {config.temperatures} K")
    print(f"Pseudopotentials: {config.pseudopotentials}")
    print("=" * 78)

    config.validate()

    print("\n[STEP 1/6] Defect Construction & Geometry Optimization...")
    builder = DefectBuilder(config)
    shared_foundation = builder.calc_foundation
    pristine_sc, defect_sc = builder.build_supercell_and_defect()
    at_gs = builder.relax_ground_state(defect_sc)
    at_es, dq_fc = builder.generate_excited_state_geometry(at_gs)

    print("\n[STEP 2/6] Representative Frame Generation Protocol...")
    sampler = FrameSampler(config, calc_foundation=shared_foundation)
    candidate_frames = sampler.assemble_candidate_dataset(at_gs, at_es)

    print("\n[STEP 3/6] High-Fidelity DFT Labeling (Quantum ESPRESSO)...")
    qe = QERunner(
        config,
        n_procs=n_procs_qe,
        pw_exe=getattr(config, "pw_exe", None),
        mpi_command=getattr(config, "mpi_command", None),
        mpi_args=getattr(config, "mpi_args", None),
        n_workers=n_workers_qe,
        timeout_s=getattr(config, "qe_timeout_s", None),
    )
    dft_limit = max_dft_frames if max_dft_frames is not None else getattr(config, "max_dft_frames", None)
    labeled_frames = qe.label_all_frames(
        candidate_frames,
        max_frames=dft_limit,
        sampler=sampler,
        at_gs=at_gs,
        n_workers=n_workers_qe,
    )

    print("\n[STEP 4/6] Delta Baseline Extraction & GPU GPUMD Training...")
    trainer = DeltaTrainer(config, calc_foundation=shared_foundation)
    delta_frames = trainer.compute_delta_dataset(labeled_frames)
    delta_pot_path = trainer.train_delta_nep(delta_frames)

    print("\n[STEP 5/6] Assembling Hybrid Defect Potential (V_base + Delta_V)...")
    hybrid_calc = HybridDefectCalculator(
        config.foundation_potential_path, delta_pot_path, calc_base=shared_foundation
    )

    at_test = at_gs.copy()
    at_test.calc = hybrid_calc
    e_hybrid = at_test.get_potential_energy()
    f_hybrid = at_test.get_forces()
    print(f"[+] Hybrid Potential verified on relaxed GS: E = {e_hybrid:.4f} eV, Max F = {np.abs(f_hybrid).max():.4f} eV/A")

    print("\n[STEP 6/6] Normal Modes & Multi-Temperature Optical Lineshapes...")
    engine = LineshapeEngine(config, hybrid_calc)
    lineshapes = engine.calculate_spectral_density_and_lineshapes(at_gs, at_es)

    total_time = time.time() - t_start
    print("\n" + "=" * 78)
    print(f"      WORKFLOW COMPLETE (Total Walltime: {total_time:.1f} s / {total_time/60.0:.2f} min)")
    print("=" * 78)

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
