#!/usr/bin/env python3
"""Unified benchmark execution runner for HyDePh color-center systems.

Runs benchmark pipelines across benchmark directories (diamond-nvminus, 4H-SiC, NaCl),
refreshes provenance manifests, and validates physical invariants.

Usage:
    python scripts/run_benchmarks.py [--systems diamond-nvminus 4H-SiC NaCl] [--from-stage 6] [--verify]
"""

from __future__ import annotations

import sys
import argparse
import subprocess
from pathlib import Path
from typing import List


AVAILABLE_SYSTEMS = ["diamond-nvminus", "4H-SiC", "NaCl"]


def run_system_benchmark(sys_name: str, benchmarks_dir: Path, from_stage: int = 6, to_stage: int = 6) -> bool:
    """Executes the pipeline for a single benchmark system."""
    sdir = benchmarks_dir / sys_name
    if not sdir.is_dir():
        print(f"[-] Benchmark system directory '{sys_name}' not found at {sdir}")
        return False

    cfg_py = sdir / "config.py"
    cfg_yaml = sdir / "config.yaml"

    print(f"\n{'='*78}")
    print(f"  RUNNING BENCHMARK: {sys_name} (Stages {from_stage}..{to_stage})")
    print(f"{'='*78}")

    if cfg_py.is_file():
        cmd = [
            sys.executable,
            "-c",
            f"""
import sys
sys.path.insert(0, '{sdir.parent.parent}')
from hydeph import run_pipeline
from benchmarks.{sys_name.replace('-', '_') if '-' in sys_name else sys_name}.config import config
config.workspace_dir = '{sdir}'
summary = run_pipeline(config, from_stage={from_stage}, to_stage={to_stage})
print('[+] Benchmark run completed successfully.')
""",
        ]
    elif cfg_yaml.is_file():
        cmd = [
            sys.executable,
            "-m",
            "hydeph.pipeline",
            "run",
            str(cfg_yaml),
            f"--from-stage={from_stage}",
            f"--to-stage={to_stage}",
        ]
    else:
        print(f"[-] Neither config.py nor config.yaml found in {sdir}")
        return False

    res = subprocess.run(cmd, cwd=str(sdir))
    return res.returncode == 0


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
        "--verify",
        action="store_true",
        default=True,
        help="Run verify_benchmarks.py after completing benchmark runs.",
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

    # Refresh provenance manifests
    prov_script = repo_root / "scripts" / "generate_benchmark_provenance.py"
    if prov_script.is_file():
        print("\n[-] Refreshing benchmark provenance manifests...")
        subprocess.run([sys.executable, str(prov_script)], cwd=str(repo_root))

    if args.verify:
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
