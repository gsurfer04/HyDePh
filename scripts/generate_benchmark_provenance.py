#!/usr/bin/env python3
"""Generates structured provenance.json manifests for each benchmark system."""

import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
import ase.io
from hydeph.provenance import calculate_file_sha256
import hydeph

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
benchmarks_dir = os.path.join(repo_root, "benchmarks")
systems = ["diamond-nvminus", "4H-SiC", "NaCl"]

for sys_name in systems:
    sdir = os.path.join(benchmarks_dir, sys_name)
    if not os.path.isdir(sdir):
        continue

    cfg_obj = None
    cfg_py = os.path.join(sdir, "config.py")
    if os.path.isfile(cfg_py):
        prev_cwd = os.getcwd()
        try:
            os.chdir(sdir)
            spec = importlib.util.spec_from_file_location("bench_cfg_prov", cfg_py)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            cfg_obj = getattr(mod, "config", None)
        except Exception:
            cfg_obj = None
        finally:
            os.chdir(prev_cwd)

    gs_path = os.path.join(sdir, "relaxed_gs.xyz")
    es_path = os.path.join(sdir, "relaxed_es.xyz")
    at_gs = ase.io.read(gs_path) if os.path.isfile(gs_path) else None

    sum_path = os.path.join(sdir, "lineshape_summary.json")
    summary = {}
    if os.path.isfile(sum_path):
        with open(sum_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

    cc_path = os.path.join(sdir, "configuration_coordinate_data.json")
    cc_data = {}
    if os.path.isfile(cc_path):
        with open(cc_path, "r", encoding="utf-8") as f:
            cc_data = json.load(f)

    files_to_hash = [
        "pristine.xyz",
        "defect.xyz",
        "relaxed_gs.xyz",
        "relaxed_es.xyz",
        "nep_delta_gs.txt",
        "nep_delta_es.txt",
        "optical_lineshapes.csv",
        "spectral_density.csv",
        "configuration_coordinate_curve.csv",
    ]
    file_hashes = {}
    for fn in files_to_hash:
        fp = os.path.join(sdir, fn)
        if os.path.isfile(fp):
            file_hashes[fn] = {
                "bytes": os.path.getsize(fp),
                "sha256": calculate_file_sha256(fp),
            }

    prov = {
        "system": sys_name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hydeph_version": hydeph.__version__,
        "python_version": sys.version.split()[0],
        "ase_version": getattr(ase, "__version__", "unknown"),
        "allow_fallbacks": getattr(cfg_obj, "allow_fallbacks", False),
        "excited_state_method": getattr(cfg_obj, "excited_state_method", "delta_scf"),
        "paired_delta_nep": getattr(cfg_obj, "paired_delta_nep", True),
        "phonon_supercell_repeat": str(getattr(cfg_obj, "phonon_supercell_repeat", "auto")),
        "hessian_embedding_mode": getattr(cfg_obj, "hessian_embedding_mode", "auto"),
        "phonon_force_state": getattr(cfg_obj, "phonon_force_state", "auto"),
        "anharmonic_md_surface": getattr(cfg_obj, "anharmonic_md_surface", "auto"),
        "anharmonic_n_trajectories": getattr(cfg_obj, "anharmonic_n_trajectories", 16),
        "anharmonic_md_steps": getattr(cfg_obj, "anharmonic_md_steps", 3000),
        "num_atoms_dft_supercell": len(at_gs) if at_gs else None,
        "n_normal_modes": summary.get("n_normal_modes"),
        "chemical_formula": at_gs.get_chemical_formula() if at_gs else None,
        "total_huang_rhys_S": summary.get("total_S"),
        "dwf_zero_temp": summary.get("dwf_zero_temp"),
        "zpl_energy_eV": cc_data.get("e_zpl_eV", cc_data.get("zpl_energy_eV")),
        "stokes_shift_eV": cc_data.get("stokes_shift_eV"),
        "data_files": file_hashes,
    }

    out_prov = os.path.join(sdir, "provenance.json")
    with open(out_prov, "w", encoding="utf-8") as f:
        json.dump(prov, f, indent=2)
    print(f"[+] Provenance manifest created: {out_prov}")
