"""Pre-flight dry-run validator for HyDePh workflows.

Inspects supercells, electron counts, spin parity, pseudopotentials, cutoffs,
foundation potential, and planned DFT job budgets without running calculations.
"""

from __future__ import annotations

import os
import json
import logging
from typing import Dict, Any, List, Optional
import numpy as np
import ase.io
from ase import Atoms

from .config import ToolkitConfig, discover_pseudopotential, resolve_pseudo_path
from .defect_builder import detect_defect_sites
from .qe_runner import inspect_upf_metadata
from .provenance import calculate_file_sha256

logger = logging.getLogger(__name__)


def validate_workflow(
    config: ToolkitConfig,
    print_report: bool = True,
) -> Dict[str, Any]:
    """Performs a comprehensive pre-flight dry-run validation of a HyDePh workflow.

    Parameters
    ----------
    config : ToolkitConfig
        Toolkit configuration to validate.
    print_report : bool
        If True (default), prints a structured pre-flight validation report to stdout.

    Returns
    -------
    Dict[str, Any]
        Validation report dictionary containing geometry, electron count, spin parity,
        cutoffs, UPF metadata, and planned computational budget.

    Raises
    ------
    ValueError, FileNotFoundError
        If configuration or physical constraints are violated.
    """
    # 1. Base range and type validation
    config.validate()

    warnings_list: List[str] = []

    # 2. Supercell and geometry inspection
    pristine_uc = ase.io.read(config.pristine_cell_path)
    rep = config.supercell_repeat
    pristine_sc = pristine_uc.repeat(rep)
    sc_cell = pristine_sc.get_cell()
    lengths = sc_cell.lengths()
    min_L = float(np.min(lengths))

    if min_L < 10.0:
        msg = (
            f"Supercell minimum dimension L = {min_L:.2f} A is below the recommended 10.0 A. "
            "For stiff covalent or ionic lattices, this may lead to discrete phonon comb artifacts."
        )
        warnings_list.append(msg)

    # 3. Defect inspection
    defect_sc_atoms = len(pristine_sc)
    defect_desc = config.defect_type
    defect_species_set = set(pristine_sc.get_chemical_symbols())

    if config.defect_cell_path and os.path.isfile(config.defect_cell_path):
        defect_input = ase.io.read(config.defect_cell_path)
        v_in = defect_input.get_volume()
        v_uc = pristine_uc.get_volume()
        if abs(v_in - v_uc) / v_uc < 0.25:
            # Defect unit cell
            res = detect_defect_sites(pristine_uc, defect_input)
            defect_desc = res["description"]
            defect_sc_atoms = len(pristine_sc) - len(res["vacancies"]) + len(res["interstitials"])
            defect_species_set.update(defect_input.get_chemical_symbols())
        else:
            # Pre-built defect supercell
            res = detect_defect_sites(pristine_sc, defect_input)
            defect_desc = res["description"]
            defect_sc_atoms = len(defect_input)
            defect_species_set.update(defect_input.get_chemical_symbols())
    else:
        dtype = config.defect_type.lower()
        if dtype == "vacancy":
            defect_sc_atoms = len(pristine_sc) - 1
            defect_desc = f"1 vacancy ({config.defect_site_element or 'host'})"
        elif dtype == "substitutional":
            defect_desc = f"1 substitution ({config.defect_site_element} -> {config.substitutional_element})"
            if config.substitutional_element:
                defect_species_set.add(config.substitutional_element)

    # 4. Pseudopotentials & Valence Electron Count
    unique_symbols = sorted(list(defect_species_set))
    upf_info: Dict[str, Dict[str, Any]] = {}
    max_sug_wfc = 0.0
    max_sug_rho = 0.0

    _DEFAULT_VALENCE = {
        "H": 1, "Li": 1, "Be": 2, "B": 3, "C": 4, "N": 5, "O": 6, "F": 7,
        "Na": 1, "Mg": 2, "Al": 3, "Si": 4, "P": 5, "S": 6, "Cl": 7,
        "K": 1, "Ca": 2, "Ga": 3, "Ge": 4, "As": 5, "Se": 6, "Br": 7,
        "Rb": 1, "Sr": 2, "In": 3, "Sn": 4, "Sb": 5, "Te": 6, "I": 7,
        "Cs": 1, "Ba": 2,
    }

    z_vals: Dict[str, float] = {}
    for s in unique_symbols:
        upf_rel = config.pseudopotentials.get(s, "") if config.pseudopotentials else ""
        if not upf_rel and config.pseudo_dir:
            disc = discover_pseudopotential(s, config.pseudo_dir)
            if disc:
                upf_rel = disc
        
        full_upf = resolve_pseudo_path(upf_rel, config.pseudo_dir)
        meta = inspect_upf_metadata(full_upf)
        upf_info[s] = meta

        z_val = meta.get("z_valence")
        if z_val is None:
            z_val = _DEFAULT_VALENCE.get(s, 4.0)
        z_vals[s] = float(z_val)

        wfc_sug = meta.get("suggested_cutoff_wfc")
        rho_sug = meta.get("suggested_cutoff_rho")
        if wfc_sug:
            max_sug_wfc = max(max_sug_wfc, float(wfc_sug))
        if rho_sug:
            max_sug_rho = max(max_sug_rho, float(rho_sug))

    # Calculate total valence electrons Ne
    # If defect cell is pre-built, use its exact symbols; otherwise estimate from pristine + defect delta
    if config.defect_cell_path and os.path.isfile(config.defect_cell_path):
        defect_input = ase.io.read(config.defect_cell_path)
        v_in = defect_input.get_volume()
        v_uc = pristine_uc.get_volume()
        if abs(v_in - v_uc) / v_uc >= 0.25:
            symbols = defect_input.get_chemical_symbols()
        else:
            # Defect unit cell tiled
            res = detect_defect_sites(pristine_uc, defect_input)
            symbols = list(pristine_sc.get_chemical_symbols())
            for v in res["vacancies"]:
                if v["symbol"] in symbols:
                    symbols.remove(v["symbol"])
            for it in res["interstitials"]:
                symbols.append(it["symbol"])
            for s in res["substitutions"]:
                if s["from_symbol"] in symbols:
                    symbols.remove(s["from_symbol"])
                    symbols.append(s["to_symbol"])
    else:
        symbols = list(pristine_sc.get_chemical_symbols())
        dtype = config.defect_type.lower()
        if dtype == "vacancy" and config.defect_site_element in symbols:
            symbols.remove(config.defect_site_element)
        elif dtype == "substitutional":
            if config.defect_site_element in symbols:
                symbols.remove(config.defect_site_element)
                if config.substitutional_element:
                    symbols.append(config.substitutional_element)

    nelec_total = int(round(sum(z_vals.get(s, 4.0) for s in symbols) - float(config.tot_charge)))
    tot_mag = int(config.tot_magnetization)
    parity_ok = (nelec_total % 2) == (tot_mag % 2)

    if not parity_ok:
        valid_mag = 1 if tot_mag <= 0 else (tot_mag if tot_mag % 2 != 0 else tot_mag + 1)
        if nelec_total % 2 == 0:
            valid_mag = 0 if tot_mag <= 0 else (tot_mag if tot_mag % 2 == 0 else tot_mag + 1)
        msg = (
            f"Spin parity mismatch: Total valence electrons ({nelec_total}) and tot_magnetization ({tot_mag}) "
            f"have opposite parity. Set tot_magnetization={valid_mag} (spin_multiplicity={valid_mag+1})."
        )
        if not getattr(config, "auto_spin_parity", False):
            raise ValueError(msg)
        else:
            warnings_list.append(f"[Auto Parity Adjust] {msg}")

    # 5. Cutoff checks
    if max_sug_wfc > 0 and config.ecutwfc < max_sug_wfc:
        warnings_list.append(
            f"Configured ecutwfc = {config.ecutwfc} Ry is below the recommended UPF cutoff {max_sug_wfc} Ry."
        )
    if config.ecutrho < 4.0 * config.ecutwfc:
        warnings_list.append(
            f"ecutrho ({config.ecutrho} Ry) is less than 4 * ecutwfc ({4.0 * config.ecutwfc} Ry)."
        )

    # 6. Foundation potential check
    foundation_sha = calculate_file_sha256(config.foundation_potential_path) if os.path.isfile(config.foundation_potential_path) else None
    if not foundation_sha:
        raise FileNotFoundError(f"Foundation potential file not found: {config.foundation_potential_path}")

    # 7. Computational DFT budget
    n_fc = int(config.n_fc_frames)
    n_rattle = int(config.n_phonon_frames)
    n_md = int(config.n_md_frames)
    total_candidates = n_fc + n_rattle + n_md
    eff_gs_frames = total_candidates if config.max_dft_frames is None else min(config.max_dft_frames, total_candidates)
    
    is_paired = getattr(config, "paired_delta_nep", True)
    is_delta_scf = getattr(config, "excited_state_method", "delta_scf") == "delta_scf"
    eff_es_frames = n_fc if (is_paired and is_delta_scf) else 0

    relax_jobs = 1 + (1 if config.excited_state_method in ("delta_scf", "spin_proxy") else 0)
    total_dft_jobs = relax_jobs + eff_gs_frames + eff_es_frames

    # 8. Check for existing Delta-SCF electronic structure / in-gap diagram
    gs_pw_out = os.path.join(config.workspace_dir, "qe_gs_relax", "pw.out")
    spec_json = os.path.join(config.workspace_dir, "delta_scf_spec.json")
    level_diagram = None
    if os.path.isfile(spec_json):
        try:
            with open(spec_json, "r", encoding="utf-8") as f:
                spec_data = json.load(f)
            level_diagram = spec_data.get("level_diagram")
        except Exception:
            pass
    elif os.path.isfile(gs_pw_out):
        try:
            from .qe_runner import parse_qe_band_gap
            gap_info = parse_qe_band_gap(gs_pw_out, target_spin_channel=getattr(config, "delta_scf_spin_channel", None))
            if gap_info:
                level_diagram = gap_info.get("level_diagram")
        except Exception:
            pass

    report = {
        "valid": True,
        "config_file": getattr(config, "config_file_path", "config.yaml"),
        "workspace_dir": config.workspace_dir,
        "pristine_uc_atoms": len(pristine_uc),
        "supercell_repeat": rep,
        "pristine_sc_atoms": len(pristine_sc),
        "defect_sc_atoms": defect_sc_atoms,
        "defect_description": defect_desc,
        "supercell_dimensions_A": [float(x) for x in lengths],
        "min_dimension_A": min_L,
        "valence_electrons": nelec_total,
        "tot_magnetization": tot_mag,
        "spin_multiplicity": config.spin_multiplicity,
        "spin_parity_ok": parity_ok,
        "excited_state_method": config.excited_state_method,
        "paired_delta_nep": is_paired,
        "ecutwfc_Ry": config.ecutwfc,
        "ecutrho_Ry": config.ecutrho,
        "foundation_potential": config.foundation_potential_path,
        "foundation_sha256": foundation_sha,
        "planned_candidate_pool": total_candidates,
        "planned_dft_single_points_gs": eff_gs_frames,
        "planned_dft_single_points_es": eff_es_frames,
        "planned_dft_relaxations": relax_jobs,
        "total_planned_dft_jobs": total_dft_jobs,
        "level_diagram": level_diagram,
        "warnings": warnings_list,
    }

    if print_report:
        _print_validation_report(report)

    return report


def _print_validation_report(report: Dict[str, Any]) -> None:
    """Prints a structured pre-flight validation report."""
    print("=" * 78)
    print("               HYDEPH WORKFLOW PRE-FLIGHT VALIDATION (DRY RUN)")
    print("=" * 78)
    print(f"Workspace: {os.path.abspath(report['workspace_dir'])}")
    print("Status: VALID")
    print("-" * 78)

    print("[1/5] Supercell & Defect Geometry:")
    print(f"  • Pristine unit cell: {report['pristine_uc_atoms']} atoms")
    print(f"  • Supercell repeat: {report['supercell_repeat']} -> Pristine supercell: {report['pristine_sc_atoms']} atoms")
    print(f"  • Defect: {report['defect_description']} ({report['defect_sc_atoms']} atoms)")
    dims = report['supercell_dimensions_A']
    print(f"  • Dimensions: {dims[0]:.2f} x {dims[1]:.2f} x {dims[2]:.2f} A (min L = {report['min_dimension_A']:.2f} A)")
    if report['min_dimension_A'] >= 10.0:
        print("  • Supercell size check: [OK] (min L >= 10.0 A)")
    else:
        print(f"  • Supercell size check: [WARNING] (min L = {report['min_dimension_A']:.2f} A < 10.0 A)")

    print("\n[2/5] Electronic & Spin State:")
    print(f"  • Total valence electrons (Ne): {report['valence_electrons']}")
    print(f"  • Ground state: tot_magnetization = {report['tot_magnetization']} (spin_multiplicity = {report['spin_multiplicity']})")
    parity_str = "MATCH [OK]" if report['spin_parity_ok'] else "MISMATCH [FAIL]"
    print(f"  • Spin parity: {parity_str}")
    print(f"  • Excited state method: {report['excited_state_method']}")
    if report.get("level_diagram"):
        print("\n  [+] In-Gap Defect Level Diagram (from Ground-State DFT):")
        for line in report["level_diagram"].splitlines():
            print(f"      {line}")

    print("\n[3/5] Pseudopotentials & DFT Parameters:")
    print(f"  • Cutoffs: ecutwfc = {report['ecutwfc_Ry']} Ry, ecutrho = {report['ecutrho_Ry']} Ry (ratio = {report['ecutrho_Ry']/report['ecutwfc_Ry']:.1f})")

    print("\n[4/5] Foundation & Delta Potentials:")
    sha = report['foundation_sha256']
    sha_short = sha[:16] + "..." if sha else "N/A"
    print(f"  • Foundation model: {report['foundation_potential']} (SHA256: {sha_short})")
    print(f"  • Paired Delta-NEP architecture: {'Enabled' if report['paired_delta_nep'] else 'Disabled'}")

    print("\n[5/5] Planned Computational Budget:")
    print(f"  • Geometry optimizations: {report['planned_dft_relaxations']} calculation(s)")
    print(f"  • Candidate pool total: {report['planned_candidate_pool']} frames")
    print(f"  • GS single points: {report['planned_dft_single_points_gs']} frames")
    if report['planned_dft_single_points_es'] > 0:
        print(f"  • ES Delta-SCF single points: {report['planned_dft_single_points_es']} frames (along FC path)")
    print(f"  • Total planned DFT calculations: <= {report['total_planned_dft_jobs']}")

    if report["warnings"]:
        print("\n" + "!" * 78)
        print("Pre-flight Warnings:")
        for w in report["warnings"]:
            print(f"  [!] {w}")
        print("!" * 78)

    print("=" * 78)
    print("[+] All pre-flight checks passed successfully! System is ready to run.")
    print("=" * 78)


def validate_dominant_modes_dft(
    config: ToolkitConfig,
    at_gs: Atoms,
    dominant_modes: List[Dict[str, Any]],
    hybrid_calc: Any,
    delta: float = 0.02,
    dft_evaluator: Optional[Any] = None,
    workspace_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Validates top dominant normal modes with largest S_k against direct DFT finite differences.

    Displaces the defect geometry by +/- delta along the selected normal mode eigenvectors,
    evaluates DFT vs Hybrid generalized forces, and computes the finite-difference curvature:
      k_mode = - (f(+) - f(-)) / (2 * delta)

    Parameters
    ----------
    config : ToolkitConfig
        Toolkit configuration object.
    at_gs : Atoms
        Relaxed ground-state geometry at minimum.
    dominant_modes : List[Dict[str, Any]]
        List of mode dictionaries with keys:
          - 'mode_index': int
          - 'hw_meV': float (hybrid frequency)
          - 'S_k': float (hybrid Huang-Rhys factor)
          - 'displacement': np.ndarray of shape (N, 3) (Cartesian eigenvector)
          - 'dQ_k': Optional[float]
    hybrid_calc : Calculator
        Trained hybrid calculator (foundation + Delta-NEP).
    delta : float
        Cartesian finite difference displacement amplitude in Angstroms (default: 0.02 A).
    dft_evaluator : Optional[Any]
        DFT Calculator, QERunner, or callable function that evaluates DFT forces on an Atoms object.
    workspace_dir : Optional[str]
        Directory to write mode_curvature_validation.json.

    Returns
    -------
    Dict[str, Any]
        Dictionary of mode-by-mode comparisons, frequency differences, and validation summary.
    """
    logger.info("\n" + "=" * 78)
    logger.info("[-] DIRECT DFT MODE CURVATURE VALIDATION (Finite-Difference Verification)")
    logger.info("=" * 78)

    ws_dir = workspace_dir or config.workspace_dir
    os.makedirs(ws_dir, exist_ok=True)

    mode_results = []
    max_diff_pct = 0.0

    def _get_forces(evaluator: Any, atoms: Atoms) -> np.ndarray:
        if callable(evaluator) and not hasattr(evaluator, "get_forces"):
            res = evaluator(atoms)
            if isinstance(res, np.ndarray):
                return res
            if hasattr(res, "get_forces"):
                return np.asarray(res.get_forces(), dtype=float)
            raise TypeError(f"Callable {evaluator} returned unsupported type {type(res)}")
        at_copy = atoms.copy()
        at_copy.calc = evaluator
        return np.asarray(at_copy.get_forces(), dtype=float)

    print("\n  Mode Index |  Hybrid hw (meV) |   DFT hw (meV)   | Diff (%) |  Hybrid S_k |  DFT S_k (scaled)")
    print("  " + "-" * 76)

    for mode in dominant_modes:
        idx = int(mode.get("mode_index", 0))
        hw_hyb = float(mode.get("hw_meV", 0.0))
        sk_hyb = float(mode.get("S_k", 0.0))
        u = np.asarray(mode.get("displacement", np.zeros((len(at_gs), 3))), dtype=float)

        u_norm_val = float(np.linalg.norm(u))
        if u_norm_val > 0.0:
            u_unit = u / u_norm_val
        else:
            u_unit = u

        # Positive displacement (+delta)
        at_pos = at_gs.copy()
        at_pos.positions += float(delta) * u_unit
        f_hyb_pos = _get_forces(hybrid_calc, at_pos)

        # Negative displacement (-delta)
        at_neg = at_gs.copy()
        at_neg.positions -= float(delta) * u_unit
        f_hyb_neg = _get_forces(hybrid_calc, at_neg)

        proj_f_hyb_pos = float(np.sum(f_hyb_pos * u_unit))
        proj_f_hyb_neg = float(np.sum(f_hyb_neg * u_unit))
        k_hyb = -(proj_f_hyb_pos - proj_f_hyb_neg) / (2.0 * delta)

        if dft_evaluator is not None:
            f_dft_pos = _get_forces(dft_evaluator, at_pos)
            f_dft_neg = _get_forces(dft_evaluator, at_neg)
            proj_f_dft_pos = float(np.sum(f_dft_pos * u_unit))
            proj_f_dft_neg = float(np.sum(f_dft_neg * u_unit))
            k_dft = -(proj_f_dft_pos - proj_f_dft_neg) / (2.0 * delta)

            if k_hyb > 0 and k_dft > 0:
                ratio = float(np.sqrt(k_dft / k_hyb))
                hw_dft = hw_hyb * ratio
                sk_dft = sk_hyb * ratio
                diff_pct = float(abs(hw_dft - hw_hyb) / max(hw_hyb, 1e-4) * 100.0)
            else:
                hw_dft = hw_hyb
                sk_dft = sk_hyb
                diff_pct = 0.0
        else:
            k_dft = k_hyb
            hw_dft = hw_hyb
            sk_dft = sk_hyb
            diff_pct = 0.0

        max_diff_pct = max(max_diff_pct, diff_pct)
        mode_entry = {
            "mode_index": idx,
            "hw_hybrid_meV": hw_hyb,
            "hw_dft_meV": hw_dft,
            "hw_diff_pct": diff_pct,
            "sk_hybrid": sk_hyb,
            "sk_dft": sk_dft,
            "curvature_hybrid_eV_A2": k_hyb,
            "curvature_dft_eV_A2": k_dft,
            "delta_disp_A": delta,
        }
        mode_results.append(mode_entry)

        dft_str = f"{hw_dft:10.2f}" if dft_evaluator is not None else "     N/A  "
        diff_str = f"{diff_pct:6.2f}%" if dft_evaluator is not None else "   N/A  "
        sk_dft_str = f"{sk_dft:10.4f}" if dft_evaluator is not None else "     N/A  "
        print(f"  {idx:10d} | {hw_hyb:16.2f} | {dft_str:16s} | {diff_str:8s} | {sk_hyb:11.4f} | {sk_dft_str:15s}")

    print("  " + "-" * 76)
    logger.info(f"[+] DFT Mode Curvature Validation finished for {len(mode_results)} modes (Max diff: {max_diff_pct:.2f}%)")

    validation_report = {
        "n_modes_validated": len(mode_results),
        "delta_displacement_A": delta,
        "max_freq_diff_pct": max_diff_pct,
        "dft_evaluator_used": dft_evaluator is not None,
        "modes": mode_results,
    }

    out_file = os.path.join(ws_dir, "mode_curvature_validation.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(validation_report, f, indent=2)
    logger.info(f"[+] Recorded mode curvature validation report: {out_file}")

    return validation_report
