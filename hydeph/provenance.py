"""Provenance tracking for HyDePh calculations and scientific reproducibility."""

from __future__ import annotations

import os
import sys
import json
import hashlib
from datetime import datetime, timezone
from typing import Dict, Any, Optional

import ase
import hydeph


def calculate_file_sha256(filepath: str) -> Optional[str]:
    """Calculates SHA256 checksum for a file if it exists, else returns None."""
    if not filepath or not os.path.isfile(filepath):
        return None
    h = hashlib.sha256()
    try:
        with open(filepath, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


compute_file_sha256 = calculate_file_sha256


def generate_provenance_manifest(
    config,
    at_gs=None,
    at_es=None,
    cc_results: Optional[Dict[str, Any]] = None,
    delta_pot_path: Optional[str] = None,
    delta_pot_es_path: Optional[str] = None,
    total_time_s: Optional[float] = None,
) -> Dict[str, Any]:
    """Assembles a full provenance dictionary capturing software versions, input hashes, and physics origins."""
    now_iso = datetime.now(timezone.utc).isoformat()

    input_files: Dict[str, Any] = {}
    for attr in ("pristine_cell_path", "defect_cell_path", "excited_cell_path", "foundation_potential_path"):
        val = getattr(config, attr, None)
        if val:
            input_files[attr] = {
                "path": str(val),
                "sha256": calculate_file_sha256(str(val)),
            }

    pseudos = getattr(config, "pseudopotentials", {}) or {}
    pseudo_hashes: Dict[str, Any] = {}
    for elem, p_path in pseudos.items():
        if os.path.isfile(p_path):
            pseudo_hashes[elem] = {
                "path": str(p_path),
                "sha256": calculate_file_sha256(str(p_path)),
            }
        elif getattr(config, "pseudo_dir", None) and os.path.isdir(config.pseudo_dir):
            full_p = os.path.join(config.pseudo_dir, p_path)
            if os.path.isfile(full_p):
                pseudo_hashes[elem] = {
                    "path": full_p,
                    "sha256": calculate_file_sha256(full_p),
                }

    gs_source = "unknown"
    if at_gs is not None:
        gs_source = at_gs.info.get("gs_source", "dft" if at_gs.info.get("energy") else "unknown")

    es_source = "unknown"
    if at_es is not None:
        es_source = at_es.info.get("es_source", "unknown")

    zpl_source = "unknown"
    zpl_eV = None
    if cc_results is not None:
        zpl_source = cc_results.get("zpl_source", "unknown")
        zpl_eV = cc_results.get("e_zpl_eV")
    elif getattr(config, "zpl_energy_eV", None) is not None:
        zpl_source = "user_configured"
        zpl_eV = float(config.zpl_energy_eV)

    manifest: Dict[str, Any] = {
        "timestamp_utc": now_iso,
        "hydeph_version": hydeph.__version__,
        "python_version": sys.version,
        "ase_version": getattr(ase, "__version__", "unknown"),
        "allow_fallbacks": getattr(config, "allow_fallbacks", False),
        "total_walltime_s": total_time_s,
        "origins": {
            "gs_source": gs_source,
            "es_source": es_source,
            "zpl_source": zpl_source,
            "zpl_energy_eV": zpl_eV,
        },
        "input_files": input_files,
        "pseudopotentials": pseudo_hashes,
        "dft_parameters": {
            "ecutwfc": getattr(config, "ecutwfc", None),
            "ecutrho": getattr(config, "ecutrho", None),
            "conv_thr": getattr(config, "conv_thr", None),
            "spin_multiplicity": getattr(config, "spin_multiplicity", None),
            "tot_charge": getattr(config, "tot_charge", None),
            "kpoints": list(getattr(config, "kpoints", (1, 1, 1))),
        },
        "models": {
            "foundation_potential": {
                "path": getattr(config, "foundation_potential_path", None),
                "sha256": calculate_file_sha256(getattr(config, "foundation_potential_path", "")),
            },
            "delta_pot_gs": {
                "path": delta_pot_path,
                "sha256": calculate_file_sha256(delta_pot_path) if delta_pot_path else None,
            },
            "delta_pot_es": {
                "path": delta_pot_es_path,
                "sha256": calculate_file_sha256(delta_pot_es_path) if delta_pot_es_path else None,
            },
        },
    }
    return manifest


def write_provenance(
    out_dir: str,
    manifest: Dict[str, Any],
    filename: str = "provenance.json",
) -> str:
    """Writes the provenance manifest to a JSON file."""
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return out_path
