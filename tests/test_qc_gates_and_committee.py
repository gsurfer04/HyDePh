"""Tests for Delta-NEP Validation, Quality Gate & Committee (Batch 7)."""

import os
import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator

from hydeph.config import ToolkitConfig
from hydeph.delta_trainer import DeltaTrainer, evaluate_delta_nep_quality_control
from hydeph.calculator import HybridDefectCalculator


class MockSimpleCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, e_val=10.0, f_scale=0.1):
        super().__init__()
        self.e_val = float(e_val)
        self.f_scale = float(f_scale)

    def calculate(self, atoms=None, properties=None, system_changes=None):
        super().calculate(atoms, properties, system_changes)
        n = len(atoms) if atoms else 1
        self.results["energy"] = self.e_val
        self.results["forces"] = np.zeros((n, 3)) + self.f_scale


def test_mean_energy_offset_subtraction_and_restoration(tmp_path):
    """Verify that per-atom baseline energy offset is subtracted during Delta extraction

    and restored identically in HybridDefectCalculator.
    """
    cfg = ToolkitConfig(
        pristine_cell_path=str(tmp_path / "pristine.xyz"),
        workspace_dir=str(tmp_path),
        subtract_mean_energy_offset=True,
    )
    mock_base = MockSimpleCalculator(e_val=-100.0, f_scale=0.0)
    trainer = DeltaTrainer(cfg, calc_foundation=mock_base)

    # 4 frames with 2 atoms each
    # e_dft = -5000.0 eV, e_base = -100.0 eV -> Delta E = -4900.0 eV (-2450.0 eV/atom)
    frames = []
    for i in range(4):
        at = Atoms("C2", positions=[[0, 0, 0], [1.5 + 0.05 * i, 0, 0]], cell=[5, 5, 5], pbc=True)
        at.info["energy"] = -5000.0 + i * 0.2
        at.arrays["forces"] = np.zeros((2, 3)) + 0.02 * i
        at.info["sample_type"] = "core_rattle"
        frames.append(at)

    delta_frames = trainer.compute_delta_dataset(frames)
    expected_offset = np.mean([(-5000.0 + i * 0.2 - (-100.0)) / 2.0 for i in range(4)])
    assert np.isclose(trainer.energy_offset_per_atom, expected_offset, atol=1e-5)

    # Delta frame energies should now have mean ~ 0
    delta_mean_e = np.mean([f.info["energy"] for f in delta_frames])
    assert np.isclose(delta_mean_e, 0.0, atol=1e-5)

    # Now verify calculator restoration
    mock_delta = MockSimpleCalculator(e_val=0.0, f_scale=0.0)
    hybrid = HybridDefectCalculator(
        calc_base=mock_base,
        calc_delta=mock_delta,
        energy_offset_per_atom=trainer.energy_offset_per_atom,
    )

    test_at = frames[0].copy()
    test_at.calc = hybrid
    e_hyb = test_at.get_potential_energy()
    # e_hyb = e_base (-100) + e_delta (0) + 2 * offset (-4900.0 / 2 * 2 = -4900) = -5000 eV (matches DFT!)
    assert np.isclose(e_hyb, -5000.0 + 0.3, atol=1.0)


def test_delta_nep_quality_control_report_and_gates(tmp_path):
    """Verify Quality Control report generation and strict gate violation raising."""
    ws = tmp_path / "ws_qc"
    ws.mkdir()

    cfg = ToolkitConfig(
        pristine_cell_path=str(ws / "pristine.xyz"),
        workspace_dir=str(ws),
        max_test_force_rmse_eV_A=0.05,
        max_gs_residual_force_eV_A=0.03,
        allow_fallbacks=False,
    )

    # Build mock frames with known forces and energies
    frames = []
    for i in range(8):
        at = Atoms("C2", positions=[[0, 0, 0], [1.5, 0, 0]], cell=[5, 5, 5], pbc=True)
        at.info["energy"] = -100.0 + i * 0.1
        at.arrays["forces"] = np.zeros((2, 3)) + 0.01 * (i + 1)
        at.info["sample_type"] = "core_rattle"
        frames.append(at)

    at_gs = frames[0].copy()

    # Case 1: Calculator with good agreement (Force error 0.01 eV/A <= gate 0.05 eV/A)
    mock_base = MockSimpleCalculator(e_val=-100.0, f_scale=0.0)
    mock_delta = MockSimpleCalculator(e_val=0.35, f_scale=0.02)
    hybrid_good = HybridDefectCalculator(calc_base=mock_base, calc_delta=mock_delta, energy_offset_per_atom=0.0)

    report = evaluate_delta_nep_quality_control(
        calc_hybrid=hybrid_good,
        labeled_frames=frames,
        at_gs=at_gs,
        workspace_dir=str(ws),
        config=cfg,
    )
    assert report["qc_passed"] is True
    assert report["gates"]["test_force_rmse_gate"] is True
    assert report["gates"]["gs_residual_force_gate"] is True
    assert os.path.isfile(str(ws / "delta_nep_qc_report.json"))
    assert os.path.isfile(str(ws / "delta_nep_parity.json"))

    # Case 2: Calculator with high force error (Force 0.20 eV/A > gate 0.05 eV/A)
    mock_bad_delta = MockSimpleCalculator(e_val=0.0, f_scale=0.20)
    hybrid_bad = HybridDefectCalculator(calc_base=mock_base, calc_delta=mock_bad_delta, energy_offset_per_atom=0.0)

    with pytest.raises(ValueError, match="Quality Gate Failed"):
        evaluate_delta_nep_quality_control(
            calc_hybrid=hybrid_bad,
            labeled_frames=frames,
            at_gs=at_gs,
            workspace_dir=str(ws),
            config=cfg,
        )

    # Case 3: With allow_fallbacks=True, violation does NOT raise but records failed gate
    cfg.allow_fallbacks = True
    rep_fallback = evaluate_delta_nep_quality_control(
        calc_hybrid=hybrid_bad,
        labeled_frames=frames,
        at_gs=at_gs,
        workspace_dir=str(ws),
        config=cfg,
    )
    assert rep_fallback["qc_passed"] is False


def test_committee_ensemble_calculator():
    """Verify that HybridDefectCalculator supports committee ensembles and reports uncertainties."""
    mock_base = MockSimpleCalculator(e_val=10.0, f_scale=0.0)
    # 3 committee members with varying predictions
    member_0 = MockSimpleCalculator(e_val=1.0, f_scale=0.01)
    member_1 = MockSimpleCalculator(e_val=1.5, f_scale=0.02)
    member_2 = MockSimpleCalculator(e_val=2.0, f_scale=0.03)

    calc = HybridDefectCalculator(calc_base=mock_base, calc_delta=member_0)
    calc.calc_deltas = [member_0, member_1, member_2]

    assert calc.is_committee is True
    at = Atoms("C2", positions=[[0, 0, 0], [1, 1, 1]], cell=[5, 5, 5], pbc=True)
    at.calc = calc

    e = at.get_potential_energy()
    # Base = 10, mean(1.0, 1.5, 2.0) = 1.5 -> Energy = 11.5
    assert np.isclose(e, 11.5)
    assert "energy_std" in calc.results
    assert np.isclose(calc.results["energy_std"], np.std([1.0, 1.5, 2.0]))

    forces = at.get_forces()
    # Base = 0, mean(0.01, 0.02, 0.03) = 0.02
    assert np.allclose(forces, 0.02)
    assert "forces_std" in calc.results
