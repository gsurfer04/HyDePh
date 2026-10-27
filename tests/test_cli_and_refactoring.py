"""Tests for CLI, Logging, and Code Refactoring (Batch 6)."""

import os
import sys
import numpy as np
import pytest
import ase.io
from ase import Atoms

from hydeph.config import ToolkitConfig
from hydeph.pipeline import setup_logging, parse_stage_arg, main
from hydeph.validator import validate_workflow
from hydeph.defect_builder import detect_defect_sites
from hydeph.frame_sampler import get_rattle_sigma
from hydeph.qe_runner import run_monitored_qe_subprocess


def test_defect_detection_order_invariance(tmp_path):
    """Verify that detect_defect_sites is completely invariant to atom ordering."""
    cell = [5.0, 5.0, 5.0]
    # Pristine: 4 C atoms
    pristine = Atoms("C4", positions=[[0, 0, 0], [1.5, 1.5, 0], [1.5, 0, 1.5], [0, 1.5, 1.5]], cell=cell, pbc=True)

    # Defect: 1 C substituted by N at [1.5, 1.5, 0], but atoms permuted in array
    defect_permuted = Atoms(
        symbols=["C", "N", "C", "C"],
        positions=[[0, 1.5, 1.5], [1.5, 1.5, 0], [0, 0, 0], [1.5, 0, 1.5]],
        cell=cell,
        pbc=True,
    )

    res = detect_defect_sites(pristine, defect_permuted, tol=0.5)
    assert len(res["substitutions"]) == 1
    assert res["substitutions"][0]["from_symbol"] == "C"
    assert res["substitutions"][0]["to_symbol"] == "N"
    assert np.allclose(res["substitutions"][0]["position"], [1.5, 1.5, 0])
    assert len(res["vacancies"]) == 0
    assert len(res["interstitials"]) == 0


def test_defect_detection_multi_defect_complex(tmp_path):
    """Verify detection of complex defects (e.g. 1 vacancy + 1 substitution)."""
    cell = [6.0, 6.0, 6.0]
    pristine = Atoms("C4", positions=[[0, 0, 0], [2, 2, 0], [2, 0, 2], [0, 2, 2]], cell=cell, pbc=True)
    # Defect has vacancy at [0, 0, 0] and substitution C->N at [2, 2, 0]
    defect = Atoms("NC2", positions=[[2, 2, 0], [2, 0, 2], [0, 2, 2]], cell=cell, pbc=True)

    res = detect_defect_sites(pristine, defect, tol=0.5)
    assert len(res["vacancies"]) == 1
    assert res["vacancies"][0]["symbol"] == "C"
    assert np.allclose(res["vacancies"][0]["position"], [0, 0, 0])

    assert len(res["substitutions"]) == 1
    assert res["substitutions"][0]["from_symbol"] == "C"
    assert res["substitutions"][0]["to_symbol"] == "N"
    assert np.allclose(res["substitutions"][0]["position"], [2, 2, 0])


def test_standardized_rattle_sigma():
    """Verify that rattle sigma is standardized to 0.08 A and 0.14 A."""
    assert get_rattle_sigma(0) == 0.08
    assert get_rattle_sigma(1) == 0.14
    assert get_rattle_sigma(2) == 0.08
    assert get_rattle_sigma(3) == 0.14


def test_stage_argument_parser():
    """Verify parsing of integer, string name, and list stage arguments."""
    assert parse_stage_arg(None) == set()
    assert parse_stage_arg(3) == {3}
    assert parse_stage_arg("3") == {3}
    assert parse_stage_arg("dft_labeling") == {3}
    assert parse_stage_arg("1, 3, 5") == {1, 3, 5}
    assert parse_stage_arg(["sampling", 4]) == {2, 4}


def test_centralized_logging_to_file(tmp_path):
    """Verify that setup_logging configures root package logger and writes to hydeph.log."""
    log_file = tmp_path / "sub" / "hydeph.log"
    pkg_logger = setup_logging(log_file=str(log_file), console=False)

    test_msg = "HyDePh pre-flight check initialized"
    pkg_logger.info(test_msg)

    # Force flush handlers
    for h in pkg_logger.handlers:
        h.flush()

    assert os.path.isfile(str(log_file))
    with open(str(log_file), "r", encoding="utf-8") as f:
        content = f.read()
    assert test_msg in content
    assert "[INFO]" in content


def test_dry_run_validation(tmp_path):
    """Verify hydeph validate pre-flight dry-run inspection."""
    # Create mock pristine cell and UPF
    p_file = tmp_path / "pristine.xyz"
    pristine = Atoms("C8", cell=[3.567, 3.567, 3.567], pbc=True)
    pristine.set_positions([
        [0.0, 0.0, 0.0], [0.89, 0.89, 0.89],
        [1.78, 1.78, 0.0], [2.67, 2.67, 0.89],
        [1.78, 0.0, 1.78], [2.67, 0.89, 2.67],
        [0.0, 1.78, 1.78], [0.89, 2.67, 2.67],
    ])
    ase.io.write(str(p_file), pristine)

    upf_file = tmp_path / "C.upf"
    upf_file.write_text(
        '<UPF version="2.0.1">\n'
        '  <PP_HEADER\n'
        '     element="C"\n'
        '     z_valence="4.00"\n'
        '     wfc_cutoff="40.0"\n'
        '     rho_cutoff="160.0"\n'
        '  />\n'
        '</UPF>\n'
    )

    pot_file = tmp_path / "nep.txt"
    pot_file.write_text("dummy foundation potential\n")

    cfg = ToolkitConfig(
        pristine_cell_path=str(p_file),
        supercell_repeat=(3, 3, 3),  # 8 * 27 = 216 atoms -> L = 10.7 A
        defect_type="vacancy",
        defect_site_element="C",
        pseudopotentials={"C": str(upf_file)},
        foundation_potential_path=str(pot_file),
        charge=0,
        spin_multiplicity=1,  # 215 atoms * 4 = 860 electrons (even), spin mult 1 (tot_mag=0, even) -> parity match
        max_dft_frames=20,
        workspace_dir=str(tmp_path / "ws"),
    )

    report = validate_workflow(cfg, print_report=True)
    assert report["valid"] is True
    assert report["pristine_sc_atoms"] == 216
    assert report["defect_sc_atoms"] == 215
    assert report["valence_electrons"] == 860
    assert report["spin_parity_ok"] is True
    assert report["min_dimension_A"] > 10.0
    assert report["planned_dft_single_points_gs"] == 20
    assert report["planned_dft_single_points_es"] == 7
    assert report["total_planned_dft_jobs"] == 29  # 2 relaxations + 20 GS + 7 ES


def test_dry_run_cli_entry_point(tmp_path):
    """Verify CLI dry run execution via main(['validate', ...])."""
    p_file = tmp_path / "pristine.xyz"
    pristine = Atoms("C8", cell=[3.567, 3.567, 3.567], pbc=True)
    ase.io.write(str(p_file), pristine)

    upf_file = tmp_path / "C.upf"
    upf_file.write_text(
        '<UPF version="2.0.1">\n'
        '  <PP_HEADER element="C" z_valence="4.00" />\n'
        '</UPF>\n'
    )
    pot_file = tmp_path / "nep.txt"
    pot_file.write_text("dummy\n")

    cfg_yaml = tmp_path / "config.yaml"
    cfg_yaml.write_text(
        f"pristine_cell_path: '{p_file}'\n"
        f"supercell_repeat: [3, 3, 3]\n"
        f"defect_type: 'vacancy'\n"
        f"defect_site_element: 'C'\n"
        f"pseudopotentials:\n  C: '{upf_file}'\n"
        f"foundation_potential_path: '{pot_file}'\n"
        f"spin_multiplicity: 1\n"
        f"workspace_dir: '{tmp_path / 'ws_cli'}'\n"
    )

    exit_code = main(["validate", str(cfg_yaml)])
    assert exit_code == 0


def test_run_monitored_qe_subprocess_timeout(tmp_path):
    """Verify that run_monitored_qe_subprocess raises RuntimeError on timeout."""
    pw_out = tmp_path / "pw.out"
    # Execute a python command that sleeps 5 seconds with a 0.2s timeout
    cmd = [sys.executable, "-c", "import time; time.sleep(5)"]

    with pytest.raises(RuntimeError, match="timed out"):
        run_monitored_qe_subprocess(
            cmd_args=cmd,
            calc_dir=str(tmp_path),
            pw_out=str(pw_out),
            env=os.environ.copy(),
            timeout=0.2,
            mode="scf",
            allow_fallbacks=False,
        )


def test_pipeline_stage_resumption_missing_artifacts(tmp_path):
    """Verify that resuming from stage > 1 raises FileNotFoundError if required artifacts are absent."""
    p_file = tmp_path / "pristine.xyz"
    pristine = Atoms("C2", cell=[3, 3, 3], positions=[[0, 0, 0], [1, 1, 1]], pbc=True)
    ase.io.write(str(p_file), pristine)

    pot_file = tmp_path / "nep.txt"
    pot_file.write_text("dummy\n")

    pw_exe = tmp_path / "pw.exe"
    pw_exe.write_text("dummy pw\n")

    cfg = ToolkitConfig(
        pristine_cell_path=str(p_file),
        foundation_potential_path=str(pot_file),
        pw_exe=str(pw_exe),
        workspace_dir=str(tmp_path / "ws_empty"),
    )

    from hydeph.pipeline import run_pipeline
    from unittest.mock import patch

    with patch("hydeph.calculator.CPUNEP"):
        with pytest.raises(FileNotFoundError, match="Cannot resume from stage 2"):
            run_pipeline(cfg, from_stage=2)
