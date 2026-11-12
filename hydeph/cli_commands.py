"""Standalone CLI subcommands for HyDePh: status, qc, lineshape, and converge."""

from __future__ import annotations

import os
import sys
import json
import logging
from typing import Optional, Any, Tuple
import numpy as np
import ase.io

from .config import ToolkitConfig
from .provenance import write_stage_manifest, update_incremental_provenance

logger = logging.getLogger("hydeph.cli")


def _resolve_workspace_and_config(target: str) -> Tuple[str, Optional[ToolkitConfig]]:
    """Resolves workspace directory and optional ToolkitConfig from target path."""
    if os.path.isfile(target):
        try:
            cfg = ToolkitConfig.from_file(target)
            return os.path.abspath(cfg.workspace_dir), cfg
        except Exception:
            return os.path.abspath(os.path.dirname(target)), None
    elif os.path.isdir(target):
        # Look for config in directory
        for cand in ("config.yaml", "config.yml", "config.json", "config.toml"):
            p = os.path.join(target, cand)
            if os.path.isfile(p):
                try:
                    cfg = ToolkitConfig.from_file(p)
                    return os.path.abspath(target), cfg
                except Exception:
                    pass
        return os.path.abspath(target), None
    else:
        raise FileNotFoundError(f"Target path does not exist: {target}")


def cli_status(target: str) -> int:
    """Prints a structured stage manifest table, artifact counts, QC status, and next steps."""
    try:
        ws, cfg = _resolve_workspace_and_config(target)
    except Exception as e:
        print(f"[X] Error: {e}", file=sys.stderr)
        return 1

    print("=" * 80)
    print("                      HyDePh Workspace Status")
    print("=" * 80)
    print(f"Workspace: {ws}")
    if cfg is not None:
        print(f"Defect:    {getattr(cfg, 'defect_type', 'unknown')} in {os.path.basename(cfg.pristine_cell_path)}")
    print("-" * 80)

    stages_info = [
        (1, "Pristine & Defect Relaxation", "relaxed_gs.xyz", "qe_gs_relax"),
        (2, "Configuration Coordinate", "configuration_coordinate_data.json", "cc_curve"),
        (3, "Active Framing & DFT Labels", "labeled_dft_frames.xyz", "dft_calculations"),
        (4, "Delta-NEP & ES QC", "delta_nep_parity.json", "nep_delta.txt"),
        (5, "Anharmonic CC Validation", "anharmonic_lineshapes.png", "configuration_coordinate.png"),
        (6, "Spectral Density & Lineshapes", "optical_lineshapes.csv", "spectral_density.csv"),
    ]

    last_completed = 0
    stage_statuses = []

    for stage_idx, name, primary_art, alt_art in stages_info:
        manifest_file = os.path.join(ws, f"stage_{stage_idx}_manifest.json")
        if not os.path.isfile(manifest_file):
            manifest_file = os.path.join(ws, f"manifest_stage_{stage_idx}.json")
        primary_path = os.path.join(ws, primary_art)
        alt_path = os.path.join(ws, alt_art)

        status = "PENDING"
        details = ""

        if os.path.isfile(manifest_file):
            try:
                with open(manifest_file, "r", encoding="utf-8") as f:
                    mdata = json.load(f)
                status = "COMPLETED"
                last_completed = max(last_completed, stage_idx)
                outputs = mdata.get("outputs", {})
                if stage_idx == 1:
                    details = "Relaxed GS and ES geometries generated"
                elif stage_idx == 2:
                    zpl = outputs.get("zpl_energy_eV")
                    dq = outputs.get("delta_q")
                    details = f"ZPL={zpl:.3f} eV, dQ={dq:.3f} amu^(1/2) A" if zpl and dq else "CC curve mapped"
                elif stage_idx == 3:
                    n_frames = outputs.get("n_labeled_frames", len(outputs.get("frames", [])))
                    details = f"{n_frames} DFT frames labeled"
                elif stage_idx == 4:
                    f_rmse = outputs.get("force_rmse_ev_a")
                    qc_ok = outputs.get("qc_gates_passed", True)
                    gate_str = "PASSED" if qc_ok else "FAILED"
                    details = f"F_RMSE={f_rmse:.3f} eV/A, QC: {gate_str}" if f_rmse else f"QC: {gate_str}"
                elif stage_idx == 5:
                    details = "Anharmonic potential validated on CC path"
                elif stage_idx == 6:
                    s_tot = outputs.get("total_S_discrete")
                    details = f"Total S={s_tot:.3f}, Lineshapes evaluated" if s_tot else "Spectral density evaluated"
            except Exception:
                status = "COMPLETED"
                last_completed = max(last_completed, stage_idx)
        elif os.path.isfile(primary_path) or os.path.isdir(alt_path) or os.path.isfile(alt_path):
            status = "COMPLETED (unmanifested)"
            last_completed = max(last_completed, stage_idx)
            details = f"Found {primary_art}"

        stage_statuses.append((stage_idx, name, status, details))

    print(f"{'Stage':<6} {'Name':<32} {'Status':<16} {'Details'}")
    print("-" * 80)
    for s_idx, s_name, s_stat, s_det in stage_statuses:
        print(f"{s_idx:<6} {s_name:<32} {s_stat:<16} {s_det}")
    print("-" * 80)

    # Check for in-flight DFT calculations in Stage 3
    dft_dir = os.path.join(ws, "dft_calculations")
    has_pending_dft = False
    if os.path.isdir(dft_dir):
        subdirs = [d for d in os.listdir(dft_dir) if os.path.isdir(os.path.join(dft_dir, d)) and d.startswith("frame_")]
        if subdirs:
            converged = sum(1 for d in subdirs if os.path.isfile(os.path.join(dft_dir, d, "pw.out")))
            print(f"DFT Calculations Progress: {converged}/{len(subdirs)} frames converged in {dft_dir}")
            if converged < len(subdirs):
                has_pending_dft = True

    # Print QC status summary if available
    qc_file = os.path.join(ws, "delta_nep_qc_report.json")
    if os.path.isfile(qc_file):
        try:
            with open(qc_file, "r", encoding="utf-8") as f:
                qc_data = json.load(f)
            gate_stat = "PASSED" if qc_data.get("all_passed", False) else "FAILED"
            f_rmse = qc_data.get("test_force_rmse_eV_A")
            e_rmse = qc_data.get("test_energy_rmse_meV_atom")
            print(f"Ground-State Delta-NEP QC Gate: {gate_stat} (Energy RMSE: {e_rmse:.2f} meV/at, Force RMSE: {f_rmse:.4f} eV/A)")
        except Exception:
            pass

    print("-" * 80)
    print("Recommended next step:")
    cfg_cmd = target if target.endswith((".yaml", ".yml", ".json", ".toml")) else "config.yaml"
    if last_completed == 0:
        print(f"  hydeph run {cfg_cmd} --from-stage 1")
    elif has_pending_dft:
        print("  Submit/complete cluster DFT calculations (e.g. sbatch submit_dft_pure_qe.slurm), then resume:")
        print(f"  hydeph run {cfg_cmd} --from-stage 4")
    elif last_completed < 6:
        next_st = last_completed + 1
        print(f"  hydeph run {cfg_cmd} --from-stage {next_st}")
        if next_st == 6:
            print(f"  or execute lineshapes directly: hydeph lineshape {cfg_cmd}")
    else:
        print(f"  All workflow stages completed! Results saved in {ws}.")
        print("  To sweep smearing, dephasing, or temperatures without rerunning stages 1-5:")
        print(f"  hydeph lineshape {cfg_cmd} --smearing 5.0 --dephasing 1.0")
    print("=" * 80)
    return 0


def cli_qc(target: str) -> int:
    """Reads and prints the comprehensive Ground-State and Excited-State QC reports."""
    try:
        ws, _ = _resolve_workspace_and_config(target)
    except Exception as e:
        print(f"[X] Error: {e}", file=sys.stderr)
        return 1

    qc_gs_file = os.path.join(ws, "delta_nep_qc_report.json")
    qc_es_file = os.path.join(ws, "delta_nep_es_qc_report.json")

    if not os.path.isfile(qc_gs_file) and not os.path.isfile(qc_es_file):
        print(f"[!] No QC reports found in workspace: {ws}")
        print("    Run Stage 4 (Delta-NEP Training) to generate QC reports.")
        return 1

    all_ok = True

    if os.path.isfile(qc_gs_file):
        print("=" * 80)
        print("                  Ground-State Delta-NEP Quality Control (QC)")
        print("=" * 80)
        with open(qc_gs_file, "r", encoding="utf-8") as f:
            gs_qc = json.load(f)

        gates = gs_qc.get("gates", {})
        all_passed = gs_qc.get("all_passed", False)
        if not all_passed:
            all_ok = False

        print(f"{'Metric':<35} {'Value':<18} {'Threshold':<15} {'Status'}")
        print("-" * 80)
        for g_name, g_info in gates.items():
            val = g_info.get("value")
            thresh = g_info.get("threshold")
            passed = g_info.get("passed", False)
            stat = "PASS" if passed else "FAIL"
            val_str = f"{val:.4f}" if isinstance(val, (int, float)) else str(val)
            thresh_str = f"{thresh:.4f}" if isinstance(thresh, (int, float)) else str(thresh)
            print(f"{g_name:<35} {val_str:<18} {thresh_str:<15} {stat}")
        print("-" * 80)
        print(f"Overall Ground-State Gate: {'PASSED' if all_passed else 'FAILED'}")

    if os.path.isfile(qc_es_file):
        print("\n" + "=" * 80)
        print("                  Excited-State Delta-NEP Quality Control (QC)")
        print("=" * 80)
        with open(qc_es_file, "r", encoding="utf-8") as f:
            es_qc = json.load(f)

        gates = es_qc.get("gates", {})
        all_passed = es_qc.get("all_passed", False)
        if not all_passed:
            all_ok = False

        print(f"{'Metric':<35} {'Value':<18} {'Threshold':<15} {'Status'}")
        print("-" * 80)
        for g_name, g_info in gates.items():
            val = g_info.get("value")
            thresh = g_info.get("threshold")
            passed = g_info.get("passed", False)
            stat = "PASS" if passed else "FAIL"
            val_str = f"{val:.4f}" if isinstance(val, (int, float)) else str(val)
            thresh_str = f"{thresh:.4f}" if isinstance(thresh, (int, float)) else str(thresh)
            print(f"{g_name:<35} {val_str:<18} {thresh_str:<15} {stat}")
        print("-" * 80)
        print(f"Overall Excited-State Gate: {'PASSED' if all_passed else 'FAILED'}")

    print("=" * 80)
    return 0 if all_ok else 1


def cli_lineshape(config_path: str, args: Any) -> int:
    """Executes Stage 6 directly from existing relaxed structures and trained potentials."""
    cfg = ToolkitConfig.from_file(config_path)

    # Apply CLI parameter overrides
    if getattr(args, "smearing", None) is not None:
        cfg.phonon_smearing_meV = float(args.smearing)
    if getattr(args, "dephasing", None) is not None:
        cfg.dephasing_gamma_meV = float(args.dephasing)
    if getattr(args, "tau", None) is not None:
        cfg.time_window_tau_fs = float(args.tau)
    if getattr(args, "embedding", None) is not None:
        cfg.phonon_embedding_method = str(args.embedding)
    if getattr(args, "temperatures", None) is not None:
        cfg.temperatures = [float(t.strip()) for t in args.temperatures.split(",")]

    ws = os.path.abspath(cfg.workspace_dir)
    relaxed_gs_path = os.path.join(ws, "relaxed_gs.xyz")
    relaxed_es_path = os.path.join(ws, "relaxed_es.xyz")

    if not os.path.isfile(relaxed_gs_path) or not os.path.isfile(relaxed_es_path):
        print(f"[X] Error: relaxed_gs.xyz or relaxed_es.xyz missing in {ws}.", file=sys.stderr)
        print("    Run stages 1-5 first or resume with 'hydeph run config.yaml'.", file=sys.stderr)
        return 1

    at_gs = ase.io.read(relaxed_gs_path)
    at_es = ase.io.read(relaxed_es_path)

    from .calculator import HybridDefectCalculator
    from .lineshape import LineshapeEngine

    logger.info(f"[-] Initializing Hybrid Calculator from workspace {ws}...")
    hybrid_calc = HybridDefectCalculator(cfg)
    calc_es = None
    paired_calc = None

    delta_es_file = os.path.join(ws, "nep_delta_es.txt")
    if os.path.isfile(delta_es_file):
        try:
            calc_es = HybridDefectCalculator(cfg, delta_potential_file=delta_es_file)
            from .delta_trainer import PairedDefectCalculator
            paired_calc = PairedDefectCalculator(hybrid_calc, calc_es)
            logger.info(f"[+] Paired Excited-State Hybrid Calculator initialized ({delta_es_file}).")
        except Exception as e:
            logger.warning(f"[!] Warning: Could not initialize excited-state calculator: {e}")

    engine = LineshapeEngine(cfg, hybrid_calc, calc_es=calc_es)
    if paired_calc is not None:
        engine.paired_calc = paired_calc

    logger.info("[-] Computing spectral density and lineshapes (Stage 6)...")
    res = engine.calculate_spectral_density_and_lineshapes(at_gs, at_es)

    # Committee uncertainty if present
    pot_dir = os.path.join(ws, "delta_nep_committee")
    if os.path.isdir(pot_dir):
        committee_files = [os.path.join(pot_dir, f) for f in os.listdir(pot_dir) if f.endswith(".txt")]
        if len(committee_files) >= 2:
            try:
                engine.evaluate_committee_lineshape_uncertainty(at_gs, at_es, committee_files)
            except Exception as e:
                logger.warning(f"[!] Warning: Committee uncertainty calculation failed: {e}")

    res_meta = {k: v for k, v in res.items() if not isinstance(v, np.ndarray)}
    write_stage_manifest(ws, 6, cfg.to_dict(), res_meta, output_files=["optical_lineshapes.csv", "spectral_density.csv"])
    update_incremental_provenance(ws, 6, res, cfg)
    logger.info(f"[+] Lineshape spectroscopy completed successfully -> {ws}")
    return 0


def cli_converge(config_path: str, args: Any) -> int:
    """Convenience helper scanning phonon supercell sizes and smearing parameters."""
    cfg = ToolkitConfig.from_file(config_path)
    ws = os.path.abspath(cfg.workspace_dir)

    relaxed_gs_path = os.path.join(ws, "relaxed_gs.xyz")
    relaxed_es_path = os.path.join(ws, "relaxed_es.xyz")
    if not os.path.isfile(relaxed_gs_path) or not os.path.isfile(relaxed_es_path):
        print(f"[X] Error: relaxed_gs.xyz or relaxed_es.xyz missing in {ws}.", file=sys.stderr)
        return 1

    at_gs = ase.io.read(relaxed_gs_path)
    at_es = ase.io.read(relaxed_es_path)

    from .calculator import HybridDefectCalculator
    from .lineshape import LineshapeEngine

    hybrid_calc = HybridDefectCalculator(cfg)
    engine = LineshapeEngine(cfg, hybrid_calc)

    supercell_list = [(1, 1, 1), (2, 2, 2)]
    if getattr(args, "supercells", None):
        supercell_list = []
        for s in args.supercells.split(";"):
            dims = [int(x.strip()) for x in s.split(",")]
            supercell_list.append(tuple(dims))

    smearings = [2.0, 5.0, 10.0]
    if getattr(args, "smearings", None):
        smearings = [float(x.strip()) for x in args.smearings.split(",")]

    results = []
    print(f"[-] Running convergence scan across {len(supercell_list)} supercells and {len(smearings)} smearing values...")

    for sc in supercell_list:
        cfg.phonon_supercell_repeat = sc
        for sm in smearings:
            cfg.phonon_smearing_meV = sm
            try:
                res = engine.calculate_spectral_density_and_lineshapes(at_gs, at_es)
                s_tot = res.get("total_S_discrete", 0.0)
                s_integ = res.get("total_S_integrated", 0.0)
                results.append({
                    "supercell": f"{sc[0]}x{sc[1]}x{sc[2]}",
                    "smearing_meV": sm,
                    "total_S_discrete": s_tot,
                    "total_S_integrated": s_integ,
                })
            except Exception as e:
                logger.warning(f"[!] Scan failed for {sc} with smearing {sm}: {e}")

    csv_path = os.path.join(ws, "convergence_summary.csv")
    import csv
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["supercell", "smearing_meV", "total_S_discrete", "total_S_integrated"])
        writer.writeheader()
        writer.writerows(results)

    print(f"[+] Convergence summary written to {csv_path}")
    return 0
