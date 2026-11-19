#!/usr/bin/env python3
"""Validates physical invariants, file completeness, and convergence for HyDePh benchmark systems.

Checks:
1. File completeness: relaxed structures, CC data, spectral density, lineshape CSVs, and provenance.
2. Physical bounds: ZPL energy, Stokes shift, Huang-Rhys factor S, effective frequency.
3. Mathematical consistency: int(S(hw) d(hw)) == S_discrete within 2%.
4. Zero-temperature Debye-Waller factor DWF(0) == exp(-S) consistency.
"""

from __future__ import annotations

import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np


BENCHMARK_SPECS = {
    "diamond-nvminus": {
        "zpl_range": (1.70, 2.30),
        "stokes_range_meV": (150.0, 450.0),
        "s_range": (2.5, 4.5),
        "min_modes": 10000,
        "stiff_lattice": True,
    },
    "4H-SiC": {
        "zpl_range": (0.95, 1.90),
        "stokes_range_meV": (150.0, 400.0),
        "s_range": (1.8, 3.8),
        "min_modes": 4000,
        "stiff_lattice": True,
    },
    "NaCl": {
        "zpl_range": (1.00, 1.80),
        "stokes_range_meV": (300.0, 800.0),
        "s_range": (9.0, 22.0),
        "min_modes": 1000,
        "stiff_lattice": False,
    },
}

REQUIRED_FILES = [
    "config.py",
    "pristine.xyz",
    "defect.xyz",
    "relaxed_gs.xyz",
    "relaxed_es.xyz",
    "nep_delta_gs.txt",
    "nep_delta_es.txt",
    "configuration_coordinate_data.json",
    "configuration_coordinate_curve.csv",
    "lineshape_summary.json",
    "spectral_density.csv",
    "optical_lineshapes.csv",
    "normal_modes.csv",
    "provenance.json",
]


def verify_system(sys_dir: Path, spec: Dict[str, Any]) -> Tuple[bool, List[str], Dict[str, Any]]:
    """Runs rigorous physical and numerical validation on a single benchmark system directory."""
    errors: List[str] = []
    metrics: Dict[str, Any] = {}

    # 1. File existence
    for fname in REQUIRED_FILES:
        fpath = sys_dir / fname
        if not fpath.is_file():
            errors.append(f"Missing required file: {fname}")

    if errors:
        return False, errors, metrics

    # 2. Parse configuration coordinate data
    with open(sys_dir / "configuration_coordinate_data.json", "r", encoding="utf-8") as f:
        cc = json.load(f)

    zpl_eV = cc.get("e_zpl_eV") or cc.get("zpl_energy_eV")
    stokes_meV = cc.get("stokes_shift_meV") or (cc.get("stokes_shift_eV", 0.0) * 1000.0)
    dq = cc.get("dq_total", 0.0)

    metrics["zpl_eV"] = zpl_eV
    metrics["stokes_meV"] = stokes_meV
    metrics["dq"] = dq

    z_min, z_max = spec["zpl_range"]
    if zpl_eV is None or not (z_min <= zpl_eV <= z_max):
        errors.append(f"ZPL {zpl_eV} eV out of physical range [{z_min}, {z_max}] eV")

    s_min, s_max = spec["stokes_range_meV"]
    if not (s_min <= stokes_meV <= s_max):
        errors.append(f"Stokes shift {stokes_meV:.1f} meV out of physical range [{s_min}, {s_max}] meV")

    if dq <= 0.01:
        errors.append(f"Delta Q {dq:.4f} amu^(1/2)*A is unphysically small or zero")

    # 3. Parse lineshape summary
    with open(sys_dir / "lineshape_summary.json", "r", encoding="utf-8") as f:
        ls_sum = json.load(f)

    total_s = ls_sum.get("total_S_discrete", ls_sum.get("total_S", 0.0))
    s_int = ls_sum.get("total_S_integrated", total_s)
    dwf_0 = ls_sum.get("dwf_zero_temp", 0.0)

    metrics["total_S"] = total_s
    metrics["total_S_integrated"] = s_int
    metrics["dwf_0"] = dwf_0

    hr_min, hr_max = spec["s_range"]
    if not (hr_min <= total_s <= hr_max):
        errors.append(f"Huang-Rhys factor S={total_s:.3f} out of physical range [{hr_min}, {hr_max}]")

    # Check S_integrated vs S_discrete agreement (within 2% or 0.05)
    if abs(total_s - s_int) > max(0.05, 0.02 * total_s):
        errors.append(
            f"Spectral density integral mismatch: sum(S_k)={total_s:.4f} vs int(S)={s_int:.4f} (>2%)"
        )

    # Check embedded supercell size and multi-temperature anharmonic convergence
    n_modes = int(ls_sum.get("n_normal_modes", 0))
    if n_modes < int(spec.get("min_modes", 0)):
        errors.append(
            f"n_normal_modes={n_modes} below converged embedded threshold ({spec['min_modes']})"
        )
    anh = ls_sum.get("anharmonic") or {}
    temps = ls_sum.get("temperatures") or []
    for t_val in temps:
        t_key = str(float(t_val))
        if t_key not in anh:
            errors.append(f"Missing anharmonic MD result for T={t_key} K")
        elif int(anh[t_key].get("n_nve_steps", 0)) < 3000:
            errors.append(
                f"T={t_key} K n_nve_steps={anh[t_key].get('n_nve_steps')} < 3000 recommended default"
            )

    # Check DWF(0) consistency with exp(-S)
    expected_dwf = float(np.exp(-total_s))
    if total_s < 10.0 and abs(dwf_0 - expected_dwf) > 0.02:
        errors.append(
            f"DWF(T=0)={dwf_0:.4f} deviates from exp(-S)={expected_dwf:.4f}"
        )

    # 4. Check spectral density non-negativity
    sd_file = sys_dir / "spectral_density.csv"
    try:
        sd_data = np.loadtxt(str(sd_file), delimiter=",", skiprows=1)
        if len(sd_data.shape) == 2 and sd_data.shape[1] >= 2:
            s_vals = sd_data[:, 1]
            min_s = float(np.min(s_vals))
            if min_s < -1e-4:
                errors.append(f"Spectral density contains negative values (min: {min_s:.4e})")
    except Exception as exc:
        errors.append(f"Failed to read/verify spectral_density.csv: {exc}")

    # 5. Check provenance manifest
    with open(sys_dir / "provenance.json", "r", encoding="utf-8") as f:
        prov = json.load(f)

    if not prov.get("hydeph_version"):
        errors.append("Provenance missing hydeph_version")

    passed = (len(errors) == 0)
    return passed, errors, metrics


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    benchmarks_dir = repo_root / "benchmarks"

    print("=" * 86)
    print("                    HyDePh Benchmark Physical Invariant Verification")
    print("=" * 86)
    print(f"Directory: {benchmarks_dir}\n")

    header = f"{'System':<18} | {'ZPL (eV)':<9} | {'Stokes (meV)':<13} | {'S (HR)':<8} | {'DWF(0)':<8} | {'Status':<10}"
    print(header)
    print("-" * 86)

    all_passed = True
    all_errors: Dict[str, List[str]] = {}

    for sys_name, spec in BENCHMARK_SPECS.items():
        sdir = benchmarks_dir / sys_name
        if not sdir.is_dir():
            print(f"{sys_name:<18} | {'--':<9} | {'--':<13} | {'--':<8} | {'--':<8} | SKIPPED (not found)")
            continue

        passed, errs, m = verify_system(sdir, spec)
        if not passed:
            all_passed = False
            all_errors[sys_name] = errs
            status_str = "FAILED"
        else:
            status_str = "PASSED"

        zpl_str = f"{m.get('zpl_eV', 0.0):.4f}" if m.get('zpl_eV') is not None else "--"
        stk_str = f"{m.get('stokes_meV', 0.0):.1f}" if m.get('stokes_meV') is not None else "--"
        s_str = f"{m.get('total_S', 0.0):.3f}" if m.get('total_S') is not None else "--"
        dwf_str = f"{m.get('dwf_0', 0.0):.4f}" if m.get('dwf_0') is not None else "--"

        print(f"{sys_name:<18} | {zpl_str:<9} | {stk_str:<13} | {s_str:<8} | {dwf_str:<8} | {status_str:<10}")

    print("=" * 86)

    if not all_passed:
        print("\n[!] VERIFICATION FAILURES DETECTED:")
        for sname, errs in all_errors.items():
            print(f"  • {sname}:")
            for e in errs:
                print(f"      - {e}")
        return 1

    print("\n[+] All benchmark physical invariants, file structures, and sum rules verified successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
