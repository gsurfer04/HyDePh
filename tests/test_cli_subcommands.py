"""Tests for HyDePh CLI subcommands: status, qc, lineshape, and converge."""

import json
from unittest.mock import patch, MagicMock
import numpy as np
import ase.io
from ase import Atoms

from hydeph.config import ToolkitConfig
from hydeph.pipeline import main


def test_cli_routing_backward_compatibility():
    """Verify that 'hydeph config.yaml' automatically routes to 'run' subcommand."""
    with patch("hydeph.pipeline.run_pipeline") as mock_run:
        with patch("hydeph.config.ToolkitConfig.from_file") as mock_cfg:
            mock_cfg.return_value = ToolkitConfig(
                pristine_cell_path="dummy.xyz",
                foundation_potential_path="dummy.txt",
                workspace_dir="ws",
            )
            ret = main(["my_config.yaml", "--stage", "3"])
            assert ret == 0
            assert mock_run.called
            _, kwargs = mock_run.call_args
            assert kwargs["from_stage"] == 3
            assert kwargs["to_stage"] == 3


def test_cli_status(tmp_path, capsys):
    """Verify 'hydeph status' displays stage table, manifests, and next step recommendations."""
    ws = tmp_path / "ws_status"
    ws.mkdir()
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(f"""
pristine_cell_path: dummy.xyz
foundation_potential_path: dummy.txt
workspace_dir: {ws}
spin_multiplicity: 1
""", encoding="utf-8")

    # Write stage 1 manifest
    m1 = ws / "manifest_stage_1.json"
    m1.write_text(json.dumps({
        "stage_index": 1,
        "stage_name": "pristine_and_defect_relaxation",
        "outputs": {"relaxed_gs": "relaxed_gs.xyz", "relaxed_es": "relaxed_es.xyz"},
    }), encoding="utf-8")

    # Run status command pointing to config
    ret = main(["status", str(cfg_path)])
    assert ret == 0
    captured = capsys.readouterr()
    assert "HyDePh Workspace Status" in captured.out
    assert "Pristine & Defect Relaxation" in captured.out
    assert "COMPLETED" in captured.out
    assert "Recommended next step:" in captured.out
    assert "--from-stage 2" in captured.out

    # Run status command pointing directly to workspace dir
    ret = main(["status", str(ws)])
    assert ret == 0


def test_cli_qc(tmp_path, capsys):
    """Verify 'hydeph qc' displays Ground-State and Excited-State QC gate tables."""
    ws = tmp_path / "ws_qc"
    ws.mkdir()

    # Write mock QC reports
    qc_gs = ws / "delta_nep_qc_report.json"
    qc_gs.write_text(json.dumps({
        "all_passed": True,
        "test_energy_rmse_meV_atom": 1.25,
        "test_force_rmse_eV_A": 0.045,
        "gates": {
            "energy_rmse": {"value": 1.25, "threshold": 2.0, "passed": True},
            "force_rmse": {"value": 0.045, "threshold": 0.08, "passed": True},
        },
    }), encoding="utf-8")

    qc_es = ws / "delta_nep_es_qc_report.json"
    qc_es.write_text(json.dumps({
        "all_passed": True,
        "gates": {
            "es_force_rmse": {"value": 0.062, "threshold": 0.10, "passed": True},
        },
    }), encoding="utf-8")

    ret = main(["qc", str(ws)])
    assert ret == 0
    captured = capsys.readouterr()
    assert "Ground-State Delta-NEP Quality Control" in captured.out
    assert "Excited-State Delta-NEP Quality Control" in captured.out
    assert "energy_rmse" in captured.out
    assert "es_force_rmse" in captured.out
    assert "PASSED" in captured.out


def test_cli_lineshape_direct_execution(tmp_path):
    """Verify 'hydeph lineshape' runs Stage 6 without executing stages 1-5."""
    ws = tmp_path / "ws_lineshape"
    ws.mkdir()

    # Write mock relaxed_gs.xyz and relaxed_es.xyz
    at_gs = Atoms("C2", positions=[[0, 0, 0], [1.5, 0, 0]], cell=[5, 5, 5], pbc=True)
    at_es = Atoms("C2", positions=[[0, 0, 0], [1.55, 0, 0]], cell=[5, 5, 5], pbc=True)
    ase.io.write(str(ws / "relaxed_gs.xyz"), at_gs)
    ase.io.write(str(ws / "relaxed_es.xyz"), at_es)

    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(f"""
pristine_cell_path: dummy.xyz
foundation_potential_path: dummy.txt
workspace_dir: {ws}
spin_multiplicity: 1
phonon_smearing_meV: 2.0
""", encoding="utf-8")

    with patch("hydeph.calculator.HybridDefectCalculator") as mock_calc_cls:
        mock_calc = MagicMock()
        mock_calc_cls.return_value = mock_calc

        with patch("hydeph.lineshape.LineshapeEngine.calculate_spectral_density_and_lineshapes") as mock_calc_line:
            mock_calc_line.return_value = {
                "total_S_discrete": 1.45,
                "total_S_integrated": 1.45,
                "hw_grid": np.linspace(1, 100, 50),
                "spectral_density": np.ones(50),
            }
            ret = main([
                "lineshape",
                str(cfg_file),
                "--smearing", "4.5",
                "--dephasing", "1.2",
            ])
            assert ret == 0
            assert mock_calc_line.called

            # Check that stage 6 manifest was written
            m6 = ws / "stage_6_manifest.json"
            assert m6.is_file()
