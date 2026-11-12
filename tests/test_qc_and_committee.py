"""Unit tests for Phase 5: QC Gates, ES QC, Bootstrap Committee Ensembles, and DFT Mode Validation."""

import os
import pytest
import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from hydeph.config import ToolkitConfig
from hydeph.delta_trainer import (
    DeltaTrainer,
    evaluate_delta_nep_quality_control,
    plot_delta_nep_parity,
)
from hydeph.lineshape import LineshapeEngine
from hydeph.validator import validate_dominant_modes_dft


class ToyDisplacedHarmonicCalc(Calculator):
    """Toy harmonic potential with configurable forces and energies."""
    implemented_properties = ["energy", "forces"]

    def __init__(self, k: float = 10.0, shift: float = 0.0, force_bias: float = 0.0):
        super().__init__()
        self.k = k
        self.shift = shift
        self.force_bias = force_bias

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        pos = atoms.positions
        dx = pos[:, 0] - self.shift
        e = 0.5 * self.k * np.sum(dx ** 2)
        f = -self.k * np.zeros_like(pos)
        f[:, 0] = -self.k * dx + self.force_bias
        self.results["energy"] = float(e)
        self.results["forces"] = f


class ToyHybridDefectCalculator:
    """Mock HybridDefectCalculator implementing basic ASE calculator interface and committee properties."""
    def __init__(self, energy_val=0.0, force_arr=None, calc_deltas=None, energy_offset=0.0):
        self.energy_val = energy_val
        self.force_arr = force_arr
        self.calc_deltas = calc_deltas or []
        self.calc_delta = self.calc_deltas[0] if self.calc_deltas else None
        self.calc_base = None
        self.energy_offset_per_atom = energy_offset
        self.results = {}

    @property
    def is_committee(self) -> bool:
        return len(self.calc_deltas) > 1

    def get_potential_energy(self, atoms=None):
        return self.energy_val

    def get_forces(self, atoms=None):
        if self.force_arr is not None:
            return np.asarray(self.force_arr, dtype=float)
        n = len(atoms) if atoms is not None else 2
        return np.zeros((n, 3))


def _make_mock_frames(n_frames=10, has_fc=True):
    frames = []
    cell = np.eye(3) * 10.0
    for i in range(n_frames):
        at = Atoms("C2", positions=[[0, 0, 0], [1.5, 0, 0]], cell=cell, pbc=True)
        if has_fc and i < 5:
            at.info["sample_type"] = "franck_condon"
            lams = [-0.2, 0.0, 0.5, 1.0, 1.2]
            at.info["lambda"] = lams[i]
            at.info["frame_id"] = f"fc_{i:02d}"
            at.info["energy"] = float(10.0 + lams[i] * 2.0)
            at.arrays["forces"] = np.ones((2, 3)) * (0.1 * lams[i])
        else:
            at.info["sample_type"] = "rattle"
            at.info["frame_id"] = f"rattle_{i:02d}"
            # Small displacement for some rattle frames
            disp = 0.02 * (i - 5)
            at.positions += disp
            at.info["energy"] = float(10.0 + disp ** 2)
            at.arrays["forces"] = np.ones((2, 3)) * 0.05
        frames.append(at)
    return frames


def _make_cfg(tmp_path, **kwargs):
    default_kwargs = {
        "pristine_cell_path": "dummy.xyz",
        "foundation_potential_path": "dummy.txt",
        "workspace_dir": str(tmp_path),
        "pseudopotentials": {"C": "C.upf"},
    }
    default_kwargs.update(kwargs)
    return ToolkitConfig(**default_kwargs)


def test_qc_lambda_0_reference(tmp_path):
    """Test that lambda=0.0 is selected as reference f0 rather than lambda=-0.2."""
    cfg = _make_cfg(
        tmp_path,
        allow_fallbacks=True,
    )
    frames = _make_mock_frames(n_frames=8, has_fc=True)
    at_gs = frames[1].copy()  # lambda = 0.0 frame

    calc_hyb = ToyHybridDefectCalculator(energy_val=10.0, force_arr=np.zeros((2, 3)))
    qc = evaluate_delta_nep_quality_control(
        calc_hybrid=calc_hyb,
        labeled_frames=frames,
        at_gs=at_gs,
        workspace_dir=str(tmp_path),
        config=cfg,
        test_fraction=0.25,
        write_files=True,
    )

    # In our mock dataset, at lambda=0.0: energy=10.0. At lambda=1.0: energy=12.0.
    # Therefore Delta E_rel (DFT) = 12.0 - 10.0 = 2.0 eV.
    # If lambda=-0.2 (energy=9.6) were used, Delta E_rel would have been 12.0 - 9.6 = 2.4 eV.
    assert abs(qc["delta_erel_dft_eV"] - 2.0) < 1e-4
    assert qc["state_label"] == "ground_state"
    assert os.path.isfile(tmp_path / "delta_nep_qc_report.json")
    assert os.path.isfile(tmp_path / "delta_nep_parity.json")


def test_qc_pre_refinement_force_gate(tmp_path):
    """Test that pre-refinement force gate fails loud when unrefined residual force is too high."""
    cfg = _make_cfg(
        tmp_path,
        max_gs_residual_force_eV_A=0.03,
        allow_fallbacks=False,
    )
    frames = _make_mock_frames(n_frames=6, has_fc=True)
    at_gs = frames[1].copy()

    calc_hyb = ToyHybridDefectCalculator(energy_val=10.0, force_arr=np.zeros((2, 3)))

    # Passing pre_refinement_force = 0.06 eV/A exceeds threshold 0.03 eV/A
    with pytest.raises(ValueError, match="Delta-NEP Quality Gate Failed"):
        evaluate_delta_nep_quality_control(
            calc_hybrid=calc_hyb,
            labeled_frames=frames,
            at_gs=at_gs,
            workspace_dir=str(tmp_path),
            config=cfg,
            pre_refinement_force=0.06,
            write_files=False,
        )


def test_qc_refinement_displacement_gate(tmp_path):
    """Test that excessive hybrid refinement displacement fails the gate when allow_fallbacks=False."""
    cfg = _make_cfg(
        tmp_path,
        max_refinement_displacement_A=0.04,
        allow_fallbacks=False,
    )
    frames = _make_mock_frames(n_frames=6, has_fc=True)
    at_gs = frames[1].copy()
    calc_hyb = ToyHybridDefectCalculator(energy_val=10.0, force_arr=np.zeros((2, 3)))

    # Refinement displacement 0.08 A exceeds 0.04 A
    with pytest.raises(ValueError, match="Delta-NEP Quality Gate Failed"):
        evaluate_delta_nep_quality_control(
            calc_hybrid=calc_hyb,
            labeled_frames=frames,
            at_gs=at_gs,
            workspace_dir=str(tmp_path),
            config=cfg,
            refinement_displacement=0.08,
            write_files=False,
        )


def test_shared_baseline_energy_offset(tmp_path):
    """Test that DeltaTrainer applies a shared baseline energy offset to ES dataset."""
    cfg = _make_cfg(
        tmp_path,
        subtract_mean_energy_offset=True,
    )
    trainer = DeltaTrainer(cfg)

    calc_base = ToyDisplacedHarmonicCalc(k=5.0)
    trainer.calc_foundation = calc_base

    frames_gs = _make_mock_frames(n_frames=4, has_fc=False)
    _ = trainer.compute_delta_dataset(frames_gs)
    gs_offset = trainer.energy_offset_per_atom

    frames_es = _make_mock_frames(n_frames=4, has_fc=False)
    delta_es = trainer.compute_delta_dataset(frames_es, shared_offset_per_atom=gs_offset)

    assert trainer.energy_offset_per_atom == gs_offset
    for fr in delta_es:
        assert fr.info["delta_energy_offset_per_atom"] == gs_offset


def test_dual_state_qc_and_parity_plot(tmp_path):
    """Test dual-state QC evaluation and plot_delta_nep_parity generation."""
    cfg = _make_cfg(
        tmp_path,
        allow_fallbacks=True,
    )
    frames_gs = _make_mock_frames(n_frames=8, has_fc=True)
    frames_es = _make_mock_frames(n_frames=8, has_fc=True)
    at_gs = frames_gs[1].copy()
    at_es = frames_es[3].copy()

    calc_gs = ToyHybridDefectCalculator(energy_val=10.0, force_arr=np.zeros((2, 3)))
    calc_es = ToyHybridDefectCalculator(energy_val=12.0, force_arr=np.zeros((2, 3)))

    qc_gs = evaluate_delta_nep_quality_control(
        calc_hybrid=calc_gs,
        labeled_frames=frames_gs,
        at_gs=at_gs,
        workspace_dir=str(tmp_path),
        config=cfg,
        state_label="ground_state",
        write_files=True,
    )

    qc_es = evaluate_delta_nep_quality_control(
        calc_hybrid=calc_es,
        labeled_frames=frames_es,
        at_gs=at_es,
        workspace_dir=str(tmp_path),
        config=cfg,
        state_label="excited_state",
        write_files=True,
    )

    plot_path = tmp_path / "delta_nep_parity.png"
    out_path = plot_delta_nep_parity(qc_gs, qc_es, save_path=str(plot_path))
    assert os.path.isfile(out_path)
    assert os.path.isfile(tmp_path / "delta_nep_qc_report.json")
    assert os.path.isfile(tmp_path / "delta_nep_qc_report_excited_state.json")


def test_committee_uncertainty_in_lineshape(tmp_path):
    """Test that LineshapeEngine computes uncertainty bands when calculator is a committee."""
    cfg = _make_cfg(
        tmp_path,
        temperatures=[10.0, 300.0],
        enable_anharmonic=False,
        export_normal_modes=False,
    )
    # 3-member committee
    members = [ToyDisplacedHarmonicCalc(k=8.0), ToyDisplacedHarmonicCalc(k=10.0), ToyDisplacedHarmonicCalc(k=12.0)]
    committee_calc = ToyHybridDefectCalculator(energy_val=5.0, force_arr=np.zeros((2, 3)), calc_deltas=members)

    engine = LineshapeEngine(cfg, committee_calc)

    at_gs = Atoms("C2", positions=[[0, 0, 0], [1.5, 0, 0]], cell=np.eye(3) * 10.0, pbc=True)
    at_es = at_gs.copy()
    at_es.positions[0, 0] += 0.05

    res = engine.calculate_spectral_density_and_lineshapes(at_gs, at_es)

    assert "total_S_std" in res
    assert res["total_S_std"] is not None
    assert res["total_S_std"] >= 0.0
    assert "s_cont_std" in res
    assert res["s_cont_std"] is not None
    assert "lineshape_std" in res
    assert 10.0 in res["lineshape_std"]

    # Verify CSV headers include standard deviation
    with open(tmp_path / "spectral_density.csv", "r") as f:
        header = f.readline()
        assert "S_harmonic_std_per_eV" in header

    with open(tmp_path / "optical_lineshapes.csv", "r") as f:
        header = f.readline()
        assert "A_harmonic_std_10.0K" in header


def test_validate_dominant_modes_dft(tmp_path):
    """Test validate_dominant_modes_dft helper."""
    cfg = _make_cfg(
        tmp_path,
    )
    at_gs = Atoms("C2", positions=[[0, 0, 0], [1.5, 0, 0]], cell=np.eye(3) * 10.0, pbc=True)
    dominant_modes = [
        {
            "mode_index": 3,
            "hw_meV": 45.0,
            "S_k": 0.85,
            "displacement": np.array([[0.1, 0, 0], [-0.1, 0, 0]]),
        },
        {
            "mode_index": 4,
            "hw_meV": 60.0,
            "S_k": 1.20,
            "displacement": np.array([[0, 0.1, 0], [0, -0.1, 0]]),
        },
    ]

    calc_hyb = ToyDisplacedHarmonicCalc(k=10.0)
    calc_dft = ToyDisplacedHarmonicCalc(k=10.5)

    rep = validate_dominant_modes_dft(
        config=cfg,
        at_gs=at_gs,
        dominant_modes=dominant_modes,
        hybrid_calc=calc_hyb,
        delta=0.02,
        dft_evaluator=calc_dft,
        workspace_dir=str(tmp_path),
    )

    assert rep["n_modes_validated"] == 2
    assert rep["dft_evaluator_used"] is True
    assert len(rep["modes"]) == 2
    assert os.path.isfile(tmp_path / "mode_curvature_validation.json")
