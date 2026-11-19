#!/usr/bin/env python3
"""Unified benchmark execution runner for HyDePh color-center systems.

Runs benchmark pipelines across benchmark directories (diamond-nvminus, 4H-SiC, NaCl),
refreshes provenance manifests, and validates physical invariants.

Usage:
    python scripts/run_benchmarks.py [--systems diamond-nvminus 4H-SiC NaCl] [--from-stage 6] [--verify]
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from typing import List


AVAILABLE_SYSTEMS = ["diamond-nvminus", "4H-SiC", "NaCl"]


def _load_benchmark_config(cfg_py: Path):
    spec = importlib.util.spec_from_file_location("hydeph_benchmark_cfg", str(cfg_py))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load benchmark config module: {cfg_py}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.config


def run_system_benchmark(sys_name: str, benchmarks_dir: Path, from_stage: int = 6, to_stage: int = 6) -> bool:
    """Executes the pipeline (or self-contained Stage 6) for a single benchmark system."""
    sdir = benchmarks_dir / sys_name
    if not sdir.is_dir():
        print(f"[-] Benchmark system directory '{sys_name}' not found at {sdir}")
        return False

    cfg_py = sdir / "config.py"
    if not cfg_py.is_file():
        print(f"[-] Benchmark config.py not found in {sdir}")
        return False

    repo_root = benchmarks_dir.parent
    sys.path.insert(0, str(repo_root))

    import ase.io
    from hydeph import run_pipeline
    from hydeph.calculator import HybridDefectCalculator, PairedDefectCalculator
    from hydeph.lineshape import LineshapeEngine

    print(f"\n{'='*78}")
    print(f"  RUNNING BENCHMARK: {sys_name} (Stages {from_stage}..{to_stage})")
    print(f"{'='*78}")

    prev_cwd = os.getcwd()
    try:
        os.chdir(str(sdir))
        cfg = _load_benchmark_config(cfg_py)
        cfg.workspace_dir = str(sdir)
        # Allow environment/local fallback for the large nep89 foundation potential file
        env_fpot = os.environ.get("HYDEPH_FOUNDATION_POTENTIAL")
        if env_fpot and os.path.isfile(env_fpot):
            cfg.foundation_potential_path = env_fpot
        elif not os.path.isfile(str(cfg.foundation_potential_path)):
            for cand in (
                Path("/home/jamesc/HyDePhTest/nep89_20250409.txt"),
                repo_root / "nep89_20250409.txt",
            ):
                if cand.is_file():
                    cfg.foundation_potential_path = str(cand)
                    break

        labeled_xyz = sdir / "labeled_dft_frames.xyz"
        if from_stage < 6 or labeled_xyz.is_file():
            run_pipeline(cfg, from_stage=from_stage, to_stage=to_stage)
        else:
            at_gs = ase.io.read(str(sdir / "relaxed_gs.xyz"))
            at_es = ase.io.read(str(sdir / "relaxed_es.xyz"))
            gs_pot = sdir / "nep_delta_gs.txt"
            es_pot = sdir / "nep_delta_es.txt"
            calc_gs = HybridDefectCalculator(
                cfg.foundation_potential_path,
                str(gs_pot) if gs_pot.is_file() else None,
            )
            calc_es = None
            paired_calc = None
            if es_pot.is_file():
                calc_es = HybridDefectCalculator(
                    cfg.foundation_potential_path,
                    str(es_pot),
                    calc_base=calc_gs.calc_base,
                )
                paired_calc = PairedDefectCalculator(calc_gs, calc_es)
            engine = LineshapeEngine(cfg, calc_gs, calc_es=calc_es)
            if paired_calc is not None:
                engine.paired_calc = paired_calc
            engine.calculate_spectral_density_and_lineshapes(at_gs, at_es)
        print(f"[+] Benchmark {sys_name} completed successfully.")
        return True
    finally:
        os.chdir(prev_cwd)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run HyDePh color-center benchmarks.")
    parser.add_argument(
        "--systems",
        nargs="+",
        default=AVAILABLE_SYSTEMS,
        choices=AVAILABLE_SYSTEMS,
        help="Benchmark systems to execute.",
    )
    parser.add_argument(
        "--from-stage",
        type=int,
        default=6,
        help="Stage to start from (default: 6 for lineshape & CC refresh).",
    )
    parser.add_argument(
        "--to-stage",
        type=int,
        default=6,
        help="Stage to finish at (default: 6).",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip running verify_benchmarks.py after completion.",
    )

    args = parser.parse_args(argv)
    repo_root = Path(__file__).resolve().parent.parent
    benchmarks_dir = repo_root / "benchmarks"

    successes = []
    failures = []

    for sys_name in args.systems:
        ok = run_system_benchmark(sys_name, benchmarks_dir, from_stage=args.from_stage, to_stage=args.to_stage)
        if ok:
            successes.append(sys_name)
        else:
            failures.append(sys_name)

    prov_script = repo_root / "scripts" / "generate_benchmark_provenance.py"
    if prov_script.is_file():
        print("\n[-] Refreshing benchmark provenance manifests...")
        subprocess.run([sys.executable, str(prov_script)], cwd=str(repo_root), check=False)

    if not args.no_verify:
        verify_script = repo_root / "scripts" / "verify_benchmarks.py"
        if verify_script.is_file():
            print("\n[-] Executing benchmark physical verification...")
            v_res = subprocess.run([sys.executable, str(verify_script)], cwd=str(repo_root))
            if v_res.returncode != 0:
                print("[!] Benchmark verification reported errors.")
                return 1

    print(f"\n[+] Benchmark execution complete: {len(successes)} succeeded, {len(failures)} failed.")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
