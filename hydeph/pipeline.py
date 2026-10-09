"""
defect_delta_nep.pipeline
=========================
Top-level turnkey workflow orchestrating the automated color-center Delta-NEP
training and multi-temperature lineshape generation.

Core Inputs:
  1. pristine_cell_path
  2. defect_cell_path (or automatic defect generation specification)
  3. foundation_potential_path (e.g., nep89_20250409.txt)
  4. pseudopotentials (dict of element -> UPF path)
  5. temperatures (list of temperatures in K)
"""

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


def run_pipeline(config: ToolkitConfig, max_dft_frames: int = None, n_procs_qe: int = 2):
    """
    Executes the end-to-end automated Delta-NEP color center workflow.
    
    Parameters
    ----------
    config : ToolkitConfig
        Validated configuration specifying the 5 inputs and workflow parameters.
    max_dft_frames : int, optional
        Maximum number of representative frames to label with DFT (useful for rapid demo).
    n_procs_qe : int, default=2
        Number of MPI processes for Quantum ESPRESSO pw.x.
        
    Returns
    -------
    dict
        Comprehensive summary of workflow execution, metrics, and generated artifacts.
    """
    t_start = time.time()
    print("=" * 78)
    print("      AUTOMATED COLOR-CENTER DELTA-NEP WORKFLOW PIPELINE")
    print("=" * 78)
    print(f"Workspace: {os.path.abspath(config.workspace_dir)}")
    print(f"Foundation Potential: {config.foundation_potential_path}")
    print(f"Temperatures of interest: {config.temperatures} K")
    print(f"Pseudopotentials: {config.pseudopotentials}")
    print("=" * 78)

    # Step 0: Validate configuration
    config.validate()

    # Step 1: Defect Construction & Foundation Potential Geometry Optimizations
    print("\n[STEP 1/6] Defect Construction & Geometry Optimization...")
    builder = DefectBuilder(config)
    pristine_sc, defect_sc = builder.build_supercell_and_defect()
    at_gs = builder.relax_ground_state(defect_sc)
    at_es, dq_fc = builder.generate_excited_state_geometry(at_gs)

    # Step 2: 4-Phase Representative Frame Sampling
    print("\n[STEP 2/6] Representative Frame Generation Protocol...")
    sampler = FrameSampler(config)
    candidate_frames = sampler.assemble_candidate_dataset(at_gs, at_es)

    # Step 3: First-Principles Quantum ESPRESSO Labeling
    print("\n[STEP 3/6] High-Fidelity DFT Labeling (Quantum ESPRESSO pw.x)...")
    qe = QERunner(config, n_procs=n_procs_qe)
    labeled_frames = qe.label_all_frames(candidate_frames, max_frames=max_dft_frames)

    # Step 4: Foundation Baseline Extraction & Delta-NEP Training
    print("\n[STEP 4/6] Delta Baseline Extraction & GPU GPUMD Training...")
    trainer = DeltaTrainer(config)
    delta_frames = trainer.compute_delta_dataset(labeled_frames)
    delta_pot_path = trainer.train_delta_nep(delta_frames)

    # Step 5: Hybrid Potential Assembly
    print("\n[STEP 5/6] Assembling Hybrid Defect Potential (V_base + Delta_V)...")
    hybrid_calc = HybridDefectCalculator(config.foundation_potential_path, delta_pot_path)
    
    # Test hybrid evaluation on ground state
    at_test = at_gs.copy()
    at_test.calc = hybrid_calc
    e_hybrid = at_test.get_potential_energy()
    f_hybrid = at_test.get_forces()
    print(f"[+] Hybrid Potential verified on relaxed GS: E = {e_hybrid:.4f} eV, Max F = {np.abs(f_hybrid).max():.4f} eV/A")

    # Step 6: Multi-Temperature Optical Lineshapes
    print("\n[STEP 6/6] Normal Modes & Multi-Temperature Optical Lineshapes...")
    engine = LineshapeEngine(config, hybrid_calc)
    lineshapes = engine.calculate_spectral_density_and_lineshapes(at_gs, at_es)

    total_time = time.time() - t_start
    print("\n" + "=" * 78)
    print(f"      WORKFLOW COMPLETE (Total Walltime: {total_time:.1f} s / {total_time/60.0:.2f} min)")
    print("=" * 78)

    summary = {
        "workspace_dir": config.workspace_dir,
        "delta_potential": delta_pot_path,
        "n_dft_frames": len(labeled_frames),
        "dq_fc": dq_fc,
        "temperatures": config.temperatures,
        "total_walltime_s": total_time,
        "lineshapes": lineshapes
    }
    return summary
