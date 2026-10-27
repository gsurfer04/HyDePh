#!/usr/bin/env python3
"""Generates structured provenance.json manifests for each benchmark system."""

import json
import os
import sys
from datetime import datetime, timezone
import ase.io
from hydeph.provenance import calculate_file_sha256
import hydeph

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
benchmarks_dir = os.path.join(repo_root, "benchmarks")

systems = ["diamond-nvminus", "CsI", "4H-SiC", "NaCl"]

for sys_name in systems:
    sdir = os.path.join(benchmarks_dir, sys_name)
    if not os.path.isdir(sdir):
        continue

    # Load structures
    gs_path = os.path.join(sdir, "relaxed_gs.xyz")
    es_path = os.path.join(sdir, "relaxed_es.xyz")
    at_gs = ase.io.read(gs_path) if os.path.isfile(gs_path) else None
    at_es = ase.io.read(es_path) if os.path.isfile(es_path) else None

    # Load summary
    sum_path = os.path.join(sdir, "lineshape_summary.json")
    summary = {}
    if os.path.isfile(sum_path):
        with open(sum_path, "r") as f:
            summary = json.load(f)

    # Load CC data
    cc_path = os.path.join(sdir, "configuration_coordinate_data.json")
    cc_data = {}
    if os.path.isfile(cc_path):
        with open(cc_path, "r") as f:
            cc_data = json.load(f)

    # Check potential and CSVs
    files_to_hash = [
        "optical_lineshapes.csv",
        "spectral_density.csv",
        "configuration_coordinate_curve.csv",
        "relaxed_gs.xyz",
        "relaxed_es.xyz",
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
        "allow_fallbacks": False,
        "excited_state_method": "spin_proxy",
        "paired_delta_nep": True,
        "num_atoms": len(at_gs) if at_gs else None,
        "chemical_formula": at_gs.get_chemical_formula() if at_gs else None,
        "total_huang_rhys_S": summary.get("total_S"),
        "dwf_zero_temp": summary.get("dwf_zero_temp"),
        "zpl_energy_eV": cc_data.get("zpl_energy_eV"),
        "stokes_shift_eV": cc_data.get("stokes_shift_eV"),
        "data_files": file_hashes,
    }

    out_prov = os.path.join(sdir, "provenance.json")
    with open(out_prov, "w", encoding="utf-8") as f:
        json.dump(prov, f, indent=2)
    print(f"[+] Provenance manifest created: {out_prov}")
