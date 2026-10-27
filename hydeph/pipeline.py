"""End-to-end Delta-NEP training and optical lineshape pipeline."""

from __future__ import annotations

import os
import sys
import time
import argparse
import logging
from typing import Optional, Dict, Any, List, Set, Union
import numpy as np
import ase.io
from ase import Atoms

from .config import ToolkitConfig
from .defect_builder import DefectBuilder, mic_displacement
from .frame_sampler import FrameSampler
from .qe_runner import QERunner
from .delta_trainer import DeltaTrainer, evaluate_delta_nep_quality_control
from .calculator import HybridDefectCalculator, PairedDefectCalculator
from .lineshape import LineshapeEngine

logger = logging.getLogger(__name__)


def setup_logging(
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    console: bool = True,
) -> logging.Logger:
    """Configures centralized logging for HyDePh to console and an automatic log file.

    Parameters
    ----------
    log_file : str, optional
        Path to output log file (e.g. hydeph.log in workspace).
    level : int
        Logging level (default: logging.INFO).
    console : bool
        Whether to attach a stdout console handler.

    Returns
    -------
    logging.Logger
        The root 'hydeph' package logger.
    """
    pkg_logger = logging.getLogger("hydeph")
    pkg_logger.setLevel(level)

    # Console stream handler
    if console and not any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in pkg_logger.handlers
    ):
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(logging.Formatter("%(message)s"))
        pkg_logger.addHandler(ch)

    # File handler
    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        norm_target = os.path.normcase(os.path.abspath(log_file))
        already_attached = False
        for h in pkg_logger.handlers:
            if isinstance(h, logging.FileHandler):
                if os.path.normcase(os.path.abspath(h.baseFilename)) == norm_target:
                    already_attached = True
                    break
        if not already_attached:
            fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
            fh.setLevel(level)
            fh.setFormatter(
                logging.Formatter(
                    "%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )
            pkg_logger.addHandler(fh)

    return pkg_logger


def parse_stage_arg(val: Optional[Union[int, str, List[Any], Set[Any]]]) -> Set[int]:
    """Parses stage arguments (int, string name, comma-separated list) into a set of stage integers (1..6)."""
    if val is None:
        return set()
    if isinstance(val, (int, float)):
        return {int(val)}
    if isinstance(val, (list, tuple, set)):
        res: Set[int] = set()
        for item in val:
            res.update(parse_stage_arg(item))
        return res

    stage_map = {
        "defect_construction": 1,
        "supercell": 1,
        "relaxation": 1,
        "frame_sampling": 2,
        "sampling": 2,
        "dft_labeling": 3,
        "dft": 3,
        "labeling": 3,
        "delta_training": 4,
        "training": 4,
        "delta_nep": 4,
        "hybrid_assembly": 5,
        "assembly": 5,
        "optical_lineshapes": 6,
        "lineshapes": 6,
    }
    res = set()
    for token in str(val).split(","):
        t = token.strip().lower()
        if t.isdigit():
            res.add(int(t))
        elif t in stage_map:
            res.add(stage_map[t])
    return res


def run_pipeline(
    config: ToolkitConfig,
    max_dft_frames: Optional[int] = None,
    n_procs_qe: Optional[int] = None,
    n_workers_qe: Optional[int] = None,
    from_stage: int = 1,
    force_stage: Optional[Union[int, str, List[Any]]] = None,
) -> Dict[str, Any]:
    """Runs the complete Delta-NEP defect workflow and returns a summary dictionary.

    Parameters
    ----------
    config : ToolkitConfig
        Active workflow configuration.
    max_dft_frames : int, optional
        Override maximum DFT frames to label.
    n_procs_qe : int, optional
        Override MPI processes per QE calculation.
    n_workers_qe : int, optional
        Override concurrent QE worker pools.
    from_stage : int
        Stage to resume execution from (1..6, default 1).
    force_stage : int or str or list, optional
        Stages to force re-execution even if cached artifacts exist.

    Returns
    -------
    Dict[str, Any]
        Summary dictionary containing paths, energies, provenance, and lineshapes.
    """
    t_start = time.time()
    eff_max_dft = max_dft_frames if max_dft_frames is not None else config.max_dft_frames
    eff_n_procs = n_procs_qe if n_procs_qe is not None else config.n_procs_qe
    eff_n_workers = n_workers_qe if n_workers_qe is not None else config.n_workers_qe
    from_stage = int(from_stage) if from_stage is not None else 1
    force_stages = parse_stage_arg(force_stage)

    # Configure centralized logging to file and console
    log_path = os.path.join(config.workspace_dir, "hydeph.log")
    setup_logging(log_file=log_path)

    config.validate()
    config.export_resolved_manifest()

    logger.info("=" * 78)
    logger.info("      AUTOMATED COLOR-CENTER DELTA-NEP WORKFLOW PIPELINE")
    logger.info("=" * 78)
    logger.info(f"Workspace: {os.path.abspath(config.workspace_dir)}")
    logger.info(f"Log file: {os.path.abspath(log_path)}")
    logger.info(f"Foundation Potential: {config.foundation_potential_path}")
    logger.info(f"Temperatures of interest: {config.temperatures} K")
    logger.info(f"Pseudopotentials: {config.pseudopotentials}")
    if from_stage > 1:
        logger.info(f"Resuming workflow from Stage {from_stage}")
    if force_stages:
        logger.info(f"Forcing re-execution of Stage(s): {sorted(list(force_stages))}")
    logger.info("=" * 78)

    qe = QERunner(
        config,
        n_procs=eff_n_procs,
        pw_exe=config.pw_exe,
        mpi_command=config.mpi_command,
        mpi_args=config.mpi_args,
        n_workers=eff_n_workers,
        timeout_s=config.qe_timeout_s,
    )
    builder = DefectBuilder(config, calc_qe=qe)
    shared_foundation = builder.calc_foundation

    # -------------------------------------------------------------------------
    # [STEP 1/6] Defect Construction & Geometry Optimization
    # -------------------------------------------------------------------------
    gs_xyz = os.path.join(config.workspace_dir, "relaxed_gs.xyz")
    es_xyz = os.path.join(config.workspace_dir, "relaxed_es.xyz")

    if from_stage <= 1 or 1 in force_stages:
        logger.info("\n[STEP 1/6] Defect Construction & Geometry Optimization...")
        pristine_sc, defect_sc = builder.build_supercell_and_defect()
        at_gs = builder.relax_ground_state(defect_sc, force_relax=(1 in force_stages))
        at_es, dq_fc = builder.generate_excited_state_geometry(at_gs, force_relax=(1 in force_stages))
    else:
        logger.info(f"\n[STEP 1/6] Skipping Defect Construction (resuming from stage {from_stage})...")
        if not os.path.isfile(gs_xyz):
            raise FileNotFoundError(
                f"Cannot resume from stage {from_stage}: Relaxed ground state '{gs_xyz}' not found. "
                "Run stage 1 first or set --from-stage 1."
            )
        at_gs = ase.io.read(gs_xyz)
        if os.path.isfile(es_xyz):
            at_es = ase.io.read(es_xyz)
        else:
            logger.info("    [i] relaxed_es.xyz not found; generating from relaxed_gs.xyz...")
            at_es, _ = builder.generate_excited_state_geometry(at_gs)
        dq_vec = mic_displacement(at_es.positions, at_gs.positions, at_gs.cell)
        m = at_gs.get_masses()
        dq_fc = float(np.sqrt(np.sum(m[:, None] * (dq_vec ** 2))))
        logger.info(f"[+] Loaded relaxed GS ({len(at_gs)} atoms) and ES (Delta Q = {dq_fc:.4f} amu^(1/2)*A)")

    # -------------------------------------------------------------------------
    # [STEP 2/6] Representative Frame Generation Protocol
    # -------------------------------------------------------------------------
    cand_xyz = os.path.join(config.workspace_dir, "candidate_frames.xyz")
    sampler = FrameSampler(config, calc_foundation=shared_foundation)

    if from_stage <= 2 or 2 in force_stages:
        logger.info("\n[STEP 2/6] Representative Frame Generation Protocol...")
        candidate_frames = sampler.assemble_candidate_dataset(
            at_gs, at_es, force_resample=(2 in force_stages)
        )
    else:
        logger.info(f"\n[STEP 2/6] Skipping Frame Sampling (resuming from stage {from_stage})...")
        if not os.path.isfile(cand_xyz):
            raise FileNotFoundError(
                f"Cannot resume from stage {from_stage}: Candidate frames file '{cand_xyz}' not found. "
                "Run stage 2 first or set --from-stage 2."
            )
        candidate_frames = ase.io.read(cand_xyz, index=":")
        logger.info(f"[+] Loaded {len(candidate_frames)} candidate frames from {cand_xyz}")

    # -------------------------------------------------------------------------
    # [STEP 3/6] High-Fidelity DFT Labeling (Quantum ESPRESSO)
    # -------------------------------------------------------------------------
    labeled_xyz = os.path.join(config.workspace_dir, "labeled_dft_frames.xyz")
    if not os.path.isfile(labeled_xyz) and os.path.isfile(os.path.join(config.workspace_dir, "labeled_frames.xyz")):
        labeled_xyz = os.path.join(config.workspace_dir, "labeled_frames.xyz")
    labeled_es_xyz = os.path.join(config.workspace_dir, "labeled_frames_es.xyz")

    if from_stage <= 3 or 3 in force_stages:
        logger.info("\n[STEP 3/6] High-Fidelity DFT Labeling (Quantum ESPRESSO)...")
        labeled_frames = qe.label_all_frames(
            candidate_frames,
            max_frames=eff_max_dft,
            sampler=sampler,
            at_gs=at_gs,
            n_workers=eff_n_workers,
            force_resample=(3 in force_stages),
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

        # High-Fidelity Optical Excited State DFT Labeling (Delta-SCF) along FC path
        labeled_frames_es: List[Atoms] = []
        fc_candidates = [f for f in candidate_frames if f.info.get("sample_type") == "franck_condon"]
        if getattr(config, "paired_delta_nep", True) and getattr(config, "excited_state_method", "delta_scf") == "delta_scf":
            logger.info("\n[-] High-Fidelity Optical Excited State DFT Labeling (Delta-SCF)...")
            labeled_frames_es = qe.label_fc_excited_state(fc_candidates, at_gs=at_gs)
            ase.io.write(labeled_es_xyz, labeled_frames_es, format="extxyz")
            logger.info(f"[+] Labeled {len(labeled_frames_es)} optical excited state frames -> {labeled_es_xyz}")

            # Extract exact adiabatic ZPL, absorption, emission, and Stokes shift
            gs_0 = next((f for f in labeled_frames if f.info.get("sample_type") == "franck_condon" and abs(f.info.get("lambda", 1.0)) < 1e-4), None)
            gs_1 = next((f for f in labeled_frames if f.info.get("sample_type") == "franck_condon" and abs(f.info.get("lambda", 0.0) - 1.0) < 1e-4), None)
            es_0 = next((f for f in labeled_frames_es if abs(f.info.get("lambda", 1.0)) < 1e-4), None)
            es_1 = next((f for f in labeled_frames_es if abs(f.info.get("lambda", 0.0) - 1.0) < 1e-4), None)

            if gs_0 is not None and es_1 is not None and "energy" in gs_0.info and "energy" in es_1.info:
                e_gs_0 = float(gs_0.info["energy"])
                e_es_1 = float(es_1.info["energy"])
                e_zpl_exact = float(e_es_1 - e_gs_0)

                at_gs.info["energy"] = e_gs_0
                at_es.info["energy"] = e_es_1
                at_gs.info["e_zpl_exact_eV"] = e_zpl_exact
                at_gs.info["zpl_source"] = "delta_scf_adiabatic"

                logger.info("\n" + "-" * 60)
                logger.info("[+] EXACT ADIABATIC DFT ENERGETICS (Delta-SCF):")
                logger.info(f"    • Ground State Minimum E_GS(lambda=0): {e_gs_0:.4f} eV")
                logger.info(f"    • Excited State Minimum E_ES(lambda=1): {e_es_1:.4f} eV")
                logger.info(f"    • Exact Adiabatic ZPL E_ZPL: {e_zpl_exact:.4f} eV")

                if gs_1 is not None and es_0 is not None and "energy" in gs_1.info and "energy" in es_0.info:
                    e_gs_1 = float(gs_1.info["energy"])
                    e_es_0 = float(es_0.info["energy"])
                    e_abs_exact = float(e_es_0 - e_gs_0)
                    e_em_exact = float(e_es_1 - e_gs_1)
                    erel_gs = float(e_gs_1 - e_gs_0)
                    erel_es = float(e_es_0 - e_es_1)
                    stokes_exact = float(e_abs_exact - e_em_exact)
                    logger.info(f"    • Vertical Absorption E_abs: {e_abs_exact:.4f} eV")
                    logger.info(f"    • Vertical Emission E_em: {e_em_exact:.4f} eV")
                    logger.info(f"    • GS Relaxation Energy Delta E_rel,GS: {erel_gs*1000:.1f} meV ({erel_gs:.4f} eV)")
                    logger.info(f"    • ES Relaxation Energy Delta E_rel,ES: {erel_es*1000:.1f} meV ({erel_es:.4f} eV)")
                    logger.info(f"    • Exact Stokes Shift Delta E_Stokes: {stokes_exact*1000:.1f} meV ({stokes_exact:.4f} eV)")
                logger.info("-" * 60 + "\n")
    else:
        logger.info(f"\n[STEP 3/6] Skipping DFT Labeling (resuming from stage {from_stage})...")
        if not os.path.isfile(labeled_xyz):
            raise FileNotFoundError(
                f"Cannot resume from stage {from_stage}: Labeled frames file '{labeled_xyz}' not found. "
                "Run stage 3 first or set --from-stage 3."
            )
        labeled_frames = ase.io.read(labeled_xyz, index=":")
        logger.info(f"[+] Loaded {len(labeled_frames)} DFT-labeled GS frames from {labeled_xyz}")
        if os.path.isfile(labeled_es_xyz):
            labeled_frames_es = ase.io.read(labeled_es_xyz, index=":")
            logger.info(f"[+] Loaded {len(labeled_frames_es)} DFT-labeled ES frames from {labeled_es_xyz}")
        else:
            labeled_frames_es = []

    # -------------------------------------------------------------------------
    # [STEP 4/6] Delta Baseline Extraction & GPU GPUMD Training
    # -------------------------------------------------------------------------
    delta_pot_path = os.path.join(config.workspace_dir, "nep_delta_gs.txt")
    delta_pot_es_path = os.path.join(config.workspace_dir, "nep_delta_es.txt") if labeled_frames_es else None

    if from_stage <= 4 or 4 in force_stages:
        logger.info("\n[STEP 4/6] Delta Baseline Extraction & GPU GPUMD Training...")
        trainer = DeltaTrainer(config, calc_foundation=shared_foundation)
        delta_frames = trainer.compute_delta_dataset(labeled_frames)
        delta_pot_path = trainer.train_delta_nep(
            delta_frames, pot_filename="nep_delta_gs.txt", subdir="delta_nep_training_gs"
        )

        if labeled_frames_es:
            logger.info("\n[-] Extracting Excited-State Delta Baselines & Training ES Delta-NEP...")
            delta_frames_es = trainer.compute_delta_dataset(labeled_frames_es)
            delta_pot_es_path = trainer.train_delta_nep(
                delta_frames_es, pot_filename="nep_delta_es.txt", subdir="delta_nep_training_es"
            )
    else:
        logger.info(f"\n[STEP 4/6] Skipping Delta Training (resuming from stage {from_stage})...")
        if not os.path.isfile(delta_pot_path):
            legacy_pot = os.path.join(config.workspace_dir, "nep_delta.txt")
            if os.path.isfile(legacy_pot):
                delta_pot_path = legacy_pot
            else:
                raise FileNotFoundError(
                    f"Cannot resume from stage {from_stage}: Delta potential '{delta_pot_path}' not found. "
                    "Run stage 4 first or set --from-stage 4."
                )
        logger.info(f"[+] Reusing existing trained GS Delta-NEP: {delta_pot_path}")
        if delta_pot_es_path and os.path.isfile(delta_pot_es_path):
            logger.info(f"[+] Reusing existing trained ES Delta-NEP: {delta_pot_es_path}")
        elif getattr(config, "paired_delta_nep", True) and os.path.isfile(os.path.join(config.workspace_dir, "nep_delta_es.txt")):
            delta_pot_es_path = os.path.join(config.workspace_dir, "nep_delta_es.txt")
            logger.info(f"[+] Reusing existing trained ES Delta-NEP: {delta_pot_es_path}")
        else:
            delta_pot_es_path = None

    # -------------------------------------------------------------------------
    # [STEP 5/6] Assembling Hybrid Defect Potential (V_base + Delta_V)
    # -------------------------------------------------------------------------
    logger.info("\n[STEP 5/6] Assembling Hybrid Defect Potential (V_base + Delta_V)...")
    committee_size = int(getattr(config, "nep_committee_size", 1))
    pot_base_gs, pot_ext_gs = os.path.splitext(delta_pot_path)
    gs_committee_files = [
        os.path.join(config.workspace_dir, f"{os.path.basename(pot_base_gs)}_{m}{pot_ext_gs}")
        for m in range(committee_size)
    ]
    if committee_size > 1 and all(os.path.isfile(f) for f in gs_committee_files):
        hybrid_calc = HybridDefectCalculator(
            config.foundation_potential_path,
            delta_pot_paths=gs_committee_files,
            calc_base=shared_foundation,
        )
        logger.info(f"[+] Paired with GS Committee Ensemble ({committee_size} members)")
    else:
        hybrid_calc = HybridDefectCalculator(
            config.foundation_potential_path, delta_pot_path, calc_base=shared_foundation
        )

    hybrid_calc_es = None
    paired_calc = None
    if delta_pot_es_path and os.path.isfile(delta_pot_es_path):
        pot_base_es, pot_ext_es = os.path.splitext(delta_pot_es_path)
        es_committee_files = [
            os.path.join(config.workspace_dir, f"{os.path.basename(pot_base_es)}_{m}{pot_ext_es}")
            for m in range(committee_size)
        ]
        if committee_size > 1 and all(os.path.isfile(f) for f in es_committee_files):
            hybrid_calc_es = HybridDefectCalculator(
                config.foundation_potential_path,
                delta_pot_paths=es_committee_files,
                calc_base=shared_foundation,
            )
            logger.info(f"[+] Paired with ES Committee Ensemble ({committee_size} members)")
        else:
            hybrid_calc_es = HybridDefectCalculator(
                config.foundation_potential_path, delta_pot_es_path, calc_base=shared_foundation
            )
        paired_calc = PairedDefectCalculator(hybrid_calc, hybrid_calc_es)
        logger.info(f"[+] Paired Hybrid Potential assembled: GS ({delta_pot_path}) and ES ({delta_pot_es_path})")

    at_test = at_gs.copy()
    at_test.calc = hybrid_calc
    e_hybrid = at_test.get_potential_energy()
    f_hybrid = at_test.get_forces()
    max_f = float(np.abs(f_hybrid).max())
    logger.info(f"[+] Ground State evaluated on Hybrid Potential: E = {e_hybrid:.4f} eV, Max F = {max_f:.4f} eV/A")

    if max_f > 0.015 or 5 in force_stages:
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

    # Evaluate Delta-NEP Quality Control & Gate Verification
    qc_report = evaluate_delta_nep_quality_control(
        calc_hybrid=hybrid_calc,
        labeled_frames=labeled_frames,
        at_gs=at_gs,
        workspace_dir=config.workspace_dir,
        config=config,
    )

    # -------------------------------------------------------------------------
    # [STEP 6/6] Normal Modes & Multi-Temperature Optical Lineshapes
    # -------------------------------------------------------------------------
    logger.info("\n[STEP 6/6] Normal Modes & Multi-Temperature Optical Lineshapes...")
    engine = LineshapeEngine(config, hybrid_calc)
    if hybrid_calc_es is not None:
        engine.calc_es = hybrid_calc_es
        engine.paired_calc = paired_calc
    lineshapes = engine.calculate_spectral_density_and_lineshapes(at_gs, at_es)

    total_time = time.time() - t_start
    logger.info("\n" + "=" * 78)
    logger.info(f"      WORKFLOW COMPLETE (Total Walltime: {total_time:.1f} s / {total_time/60.0:.2f} min)")
    logger.info("=" * 78)

    cc_analysis = lineshapes.get("cc_analysis") if isinstance(lineshapes, dict) else None
    cc_figure = cc_analysis.get("figure_path") if cc_analysis else None
    failed_dft = [e["frame_id"] for e in getattr(qe, "last_failed_entries", [])]

    from .provenance import generate_provenance_manifest, write_provenance

    manifest = generate_provenance_manifest(
        config=config,
        at_gs=at_gs,
        at_es=at_es,
        cc_results=cc_analysis,
        delta_pot_path=delta_pot_path,
        delta_pot_es_path=delta_pot_es_path,
        total_time_s=total_time,
    )
    prov_path = write_provenance(config.workspace_dir, manifest)
    logger.info(f"[+] Provenance manifest recorded: {prov_path}")

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
        "qc_report": qc_report,
        "provenance_path": prov_path,
        "provenance": manifest,
    }
    return summary


def main(argv: Optional[List[str]] = None) -> int:
    """Command-line interface entry point for executing HyDePh workflows.

    Usage:
        hydeph config.yaml
        hydeph run config.yaml [--from-stage 1..6] [--force-stage 1..6]
        hydeph validate config.yaml
    """
    raw_args = list(argv if argv is not None else sys.argv[1:])

    parser = argparse.ArgumentParser(
        prog="hydeph",
        description="HyDePh: High-Definition Hybrid Defect Photoluminescence & Anharmonic Spectroscopy",
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="Available subcommands")

    # validate subcommand
    p_validate = subparsers.add_parser(
        "validate",
        help="Perform pre-flight dry-run validation of configuration and physical parameters",
    )
    p_validate.add_argument("config_path", help="Path to YAML, TOML, or JSON configuration file")

    # run subcommand
    p_run = subparsers.add_parser("run", help="Run the end-to-end HyDePh pipeline")
    p_run.add_argument("config_path", help="Path to YAML, TOML, or JSON configuration file")
    p_run.add_argument(
        "--from-stage",
        type=int,
        default=1,
        choices=range(1, 7),
        help="Resume pipeline execution from specified stage (1..6)",
    )
    p_run.add_argument(
        "--force-stage",
        type=str,
        default=None,
        help="Force re-execution of specified stage(s) (e.g. 3 or 'dft_labeling')",
    )
    p_run.add_argument(
        "--max-dft",
        type=int,
        default=None,
        help="Optional override for maximum DFT frames to label",
    )
    p_run.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Optional override for number of concurrent DFT workers",
    )

    # Route `hydeph config.yaml [options]` without explicit 'run' subcommand
    if raw_args and raw_args[0] not in ("run", "validate", "-h", "--help"):
        raw_args.insert(0, "run")

    args = parser.parse_args(raw_args)

    if args.subcommand == "validate":
        setup_logging(console=True)
        from .validator import validate_workflow
        cfg = ToolkitConfig.from_file(args.config_path)
        try:
            validate_workflow(cfg, print_report=True)
            return 0
        except Exception as exc:
            logger.error(f"[X] Validation Failed: {exc}")
            return 1

    elif args.subcommand == "run":
        setup_logging(console=True)
        cfg = ToolkitConfig.from_file(args.config_path)
        if args.max_dft is not None:
            cfg.max_dft_frames = args.max_dft
        if args.workers is not None:
            cfg.n_workers_qe = args.workers

        run_pipeline(
            cfg,
            from_stage=args.from_stage,
            force_stage=args.force_stage,
        )
        return 0

    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
