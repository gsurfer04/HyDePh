"""Tests for Batch 8: Large-Supercell Phonon Embedding, Omega^3 PL, Consistent S, and Normal Modes Export."""

import json
import csv
import pytest
import numpy as np
from unittest.mock import patch
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from hydeph.config import ToolkitConfig
from hydeph.lineshape import (
    compute_quantum_lineshape,
    compute_photoluminescence_spectrum,
    LineshapeEngine,
    embed_phonon_supercell,
    KB_MEV_K,
)


class MockHarmonicCalc(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, k=5.0, r0=None, **kwargs):
        super().__init__(**kwargs)
        self.k = k
        self.r0 = r0

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        pos = atoms.positions
        if self.r0 is None:
            self.r0 = pos.copy()
        ref = self.r0
        dr = pos - ref
        self.results["energy"] = float(0.5 * self.k * np.sum(dr ** 2))
        self.results["forces"] = -self.k * dr


def test_photoluminescence_omega_cubed_prefactor():
    r"""Verify I_PL(hbar omega) \propto (hbar omega)^3 * A(E) on absolute photon energy axis."""
    e_zpl_eV = 2.0
    e_rel_meV = np.linspace(-300.0, 50.0, 500)
    # Mock Gaussian lineshape centered at -100 meV
    a_norm = np.exp(-0.5 * ((e_rel_meV + 100.0) / 20.0) ** 2)
    trapz_fn = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    a_norm /= trapz_fn(a_norm, e_rel_meV)

    photon_hw_eV, i_pl = compute_photoluminescence_spectrum(e_rel_meV, a_norm, e_zpl_eV)

    assert len(photon_hw_eV) == len(e_rel_meV)
    assert np.allclose(photon_hw_eV, 2.0 + e_rel_meV / 1000.0)

    # Area under I_PL should be normalized to 1.0
    area_pl = float(trapz_fn(i_pl, photon_hw_eV))
    assert abs(area_pl - 1.0) < 1e-4

    # Peak of I_PL should be slightly blueshifted relative to A(E) due to (hbar omega)^3 weight
    peak_idx_a = int(np.argmax(a_norm))
    peak_idx_pl = int(np.argmax(i_pl))
    assert photon_hw_eV[peak_idx_pl] >= photon_hw_eV[peak_idx_a]


def test_finite_temperature_debye_waller_factor():
    """Verify DWF(T) = exp(-sum S_k (2 n_k + 1)) strictly decreases with T."""
    hw_modes = np.array([15.0, 30.0, 50.0])
    sk_modes = np.array([0.5, 0.8, 0.4])
    total_S = float(np.sum(sk_modes))

    # T = 0 K
    dwf_0 = float(np.exp(-total_S))

    # Finite temperatures
    temps = [10.0, 77.0, 150.0, 300.0]
    dwf_values = []
    for T in temps:
        n_bose = 1.0 / (np.exp(hw_modes / (KB_MEV_K * T)) - 1.0)
        dwf_T = float(np.exp(-np.sum(sk_modes * (2.0 * n_bose + 1.0))))
        dwf_values.append(dwf_T)

    assert dwf_0 == pytest.approx(np.exp(-1.7), rel=1e-5)
    # Strictly decreasing: dwf_0 > dwf(10) > dwf(77) > dwf(150) > dwf(300)
    for i in range(len(dwf_values) - 1):
        assert dwf_values[i] > dwf_values[i + 1]
    assert dwf_0 > dwf_values[0]


def test_consistent_s_continuous_normalization(tmp_path):
    """Verify that continuous S(hbar omega) integrates strictly to total discrete S."""
    pristine_file = tmp_path / "pristine.xyz"
    pristine = Atoms("C2", positions=[[0, 0, 0], [1.5, 1.5, 1.5]], cell=[4, 4, 4], pbc=True)
    import ase.io
    ase.io.write(str(pristine_file), pristine)

    cfg = ToolkitConfig(
        pristine_cell_path=str(pristine_file),
        foundation_potential_path=str(tmp_path / "pot.txt"),
        workspace_dir=str(tmp_path / "ws"),
        finite_size_acoustic_cutoff=True,
        phonon_smearing_meV=5.0,
    )
    mock_calc = MockHarmonicCalc(k=10.0)
    engine = LineshapeEngine(cfg, mock_calc)

    at_gs = pristine.copy()
    at_es = pristine.copy()
    at_es.positions[1] += [0.05, 0.05, 0.05]

    # Mock vibrations energies and modes
    mock_energies = np.array([0.0001, 0.0002, 0.0003, 0.025, 0.035, 0.045], dtype=complex)  # 3 acoustic, 3 optical
    mock_mode = np.array([[0.0, 0.0, 0.0], [0.1, 0.1, 0.1]])

    with patch("hydeph.lineshape.Vibrations") as mock_vib_cls:
        vib_inst = mock_vib_cls.return_value
        vib_inst.get_energies.return_value = mock_energies
        vib_inst.get_mode.return_value = mock_mode

        res = engine.calculate_spectral_density_and_lineshapes(at_gs, at_es)

        total_S_disc = res["total_S_discrete"]
        total_S_integ = res["total_S_integrated"]
        assert total_S_disc > 0
        assert total_S_integ == pytest.approx(total_S_disc, rel=1e-3)

        # Check that lineshape_summary.json recorded both
        summary_file = tmp_path / "ws" / "lineshape_summary.json"
        with open(summary_file, "r") as f:
            data = json.load(f)
        assert data["total_S_discrete"] == pytest.approx(total_S_disc, rel=1e-3)
        assert data["total_S_integrated"] == pytest.approx(total_S_disc, rel=1e-3)
        assert "dwf_temperatures" in data


def test_imaginary_modes_detection_and_guard(tmp_path):
    """Verify that imaginary phonon modes are detected and raise when exceeding threshold."""
    pristine_file = tmp_path / "pristine.xyz"
    pristine = Atoms("C2", positions=[[0, 0, 0], [1.5, 1.5, 1.5]], cell=[4, 4, 4], pbc=True)
    import ase.io
    ase.io.write(str(pristine_file), pristine)

    cfg = ToolkitConfig(
        pristine_cell_path=str(pristine_file),
        foundation_potential_path=str(tmp_path / "pot.txt"),
        workspace_dir=str(tmp_path / "ws"),
        max_imaginary_modes=0,  # strict fail-closed
    )
    mock_calc = MockHarmonicCalc(k=10.0)
    engine = LineshapeEngine(cfg, mock_calc)

    at_gs = pristine.copy()
    at_es = pristine.copy()

    # Mode 3 has imaginary energy (e = 0.005j eV = 5.0i meV)
    mock_energies = np.array([0.0001, 0.0002, 0.0003, 0.005j, 0.035, 0.045])

    with patch("hydeph.lineshape.Vibrations") as mock_vib_cls:
        vib_inst = mock_vib_cls.return_value
        vib_inst.get_energies.return_value = mock_energies
        vib_inst.get_mode.return_value = np.zeros((2, 3))

        with pytest.raises(RuntimeError, match="Hessian calculation produced 1 imaginary mode"):
            engine.calculate_spectral_density_and_lineshapes(at_gs, at_es)

        # Now allow 1 imaginary mode
        cfg.max_imaginary_modes = 1
        res = engine.calculate_spectral_density_and_lineshapes(at_gs, at_es)
        assert res["n_imaginary_modes"] == 1


def test_normal_modes_csv_export_and_ipr(tmp_path):
    """Verify normal_modes.csv is exported with mode properties, IPR, and participation ratio."""
    pristine_file = tmp_path / "pristine.xyz"
    pristine = Atoms("C2", positions=[[0, 0, 0], [1.5, 1.5, 1.5]], cell=[4, 4, 4], pbc=True)
    import ase.io
    ase.io.write(str(pristine_file), pristine)

    cfg = ToolkitConfig(
        pristine_cell_path=str(pristine_file),
        foundation_potential_path=str(tmp_path / "pot.txt"),
        workspace_dir=str(tmp_path / "ws"),
        export_normal_modes=True,
    )
    mock_calc = MockHarmonicCalc(k=10.0)
    engine = LineshapeEngine(cfg, mock_calc)

    at_gs = pristine.copy()
    at_es = pristine.copy()
    at_es.positions[1] += [0.05, 0.05, 0.05]

    mock_energies = np.array([0.0001, 0.0002, 0.0003, 0.025, 0.035, 0.045], dtype=complex)
    # Mode localized completely on atom 1
    mock_mode = np.array([[0.0, 0.0, 0.0], [1.0 / np.sqrt(12.011), 0.0, 0.0]])

    with patch("hydeph.lineshape.Vibrations") as mock_vib_cls:
        vib_inst = mock_vib_cls.return_value
        vib_inst.get_energies.return_value = mock_energies
        vib_inst.get_mode.return_value = mock_mode

        engine.calculate_spectral_density_and_lineshapes(at_gs, at_es)

        csv_path = tmp_path / "ws" / "normal_modes.csv"
        assert csv_path.is_file()

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 3  # 3 optical modes
        for row in rows:
            ipr = float(row["IPR"])
            pr = float(row["participation_ratio"])
            assert 0.5 <= ipr <= 1.0  # N=2 atoms -> 1/N = 0.5 <= IPR <= 1.0
            assert pr > 0.0
            assert float(row["hw_meV"]) > 0.0


def test_configurable_dephasing_and_time_damping():
    """Verify default instrument-only broadening and configurable Gaussian time window."""
    hw_grid = np.linspace(1.0, 80.0, 200)
    s_cont = np.exp(-0.5 * ((hw_grid - 40.0) / 10.0) ** 2) * 1000.0

    # Default: time_window_tau_fs=None, dephasing_gamma_meV=1.5
    eg1, a1 = compute_quantum_lineshape(
        hw_grid, s_cont, temp_k=10.0, dephasing_gamma_meV=1.5, time_window_tau_fs=None
    )
    # With aggressive time window tau = 500 fs
    eg2, a2 = compute_quantum_lineshape(
        hw_grid, s_cont, temp_k=10.0, dephasing_gamma_meV=1.5, time_window_tau_fs=500.0
    )

    assert len(eg1) == len(eg2)
    assert np.allclose(eg1, eg2)
    # Apodized window broadens lineshape peak
    assert np.max(a1) > np.max(a2)


def test_large_supercell_phonon_embedding(tmp_path):
    """Verify defect embedding into a larger pristine cell for phonon calculations."""
    pristine_file = tmp_path / "pristine.xyz"
    pristine = Atoms("C2", positions=[[0, 0, 0], [1.5, 1.5, 1.5]], cell=[3.0, 3.0, 3.0], pbc=True)
    import ase.io
    ase.io.write(str(pristine_file), pristine)

    cfg = ToolkitConfig(
        pristine_cell_path=str(pristine_file),
        foundation_potential_path=str(tmp_path / "pot.txt"),
        workspace_dir=str(tmp_path / "ws"),
        supercell_repeat=(1, 1, 1),
        phonon_supercell_repeat=(2, 2, 2),  # Expand 2x2x2
        defect_type="vacancy",
        defect_site_index=1,
    )
    mock_calc = MockHarmonicCalc(k=5.0)

    at_gs = pristine.copy()
    del at_gs[1]  # 1 atom
    at_gs.info["defect_center"] = np.array([1.5, 1.5, 1.5])
    at_gs.info["defect_positions"] = [np.array([1.5, 1.5, 1.5])]

    at_es = at_gs.copy()
    at_es.positions[0] += [0.05, 0.05, 0.05]

    at_gs_large, dr_large, f_es_large = embed_phonon_supercell(
        at_gs, at_es, cfg, calc_gs=mock_calc, calc_es=None
    )

    # Pristine 2x2x2 is 16 atoms; 1 vacancy -> 15 atoms
    assert len(at_gs_large) == 15
    assert len(dr_large) == 15
    # Active displaced atom exists
    active_disps = np.linalg.norm(dr_large, axis=1)
    assert np.count_nonzero(active_disps > 1e-4) >= 1


def test_smart_phonon_limits_scaling_with_anharmonic_peak(tmp_path):
    """Verify that get_smart_phonon_limits dynamically scales y-axis when anharmonic peak exceeds 1.0."""
    pristine_file = tmp_path / "pristine.xyz"
    pristine = Atoms("C2", positions=[[0, 0, 0], [1.5, 1.5, 1.5]], cell=[3.0, 3.0, 3.0], pbc=True)
    import ase.io
    ase.io.write(str(pristine_file), pristine)

    cfg = ToolkitConfig(
        pristine_cell_path=str(pristine_file),
        foundation_potential_path=str(tmp_path / "pot.txt"),
        workspace_dir=str(tmp_path / "ws"),
    )
    mock_calc = MockHarmonicCalc(k=5.0)
    engine = LineshapeEngine(cfg, mock_calc)

    hw_grid = np.linspace(1.0, 100.0, 200)
    s_cont = np.exp(-0.5 * ((hw_grid - 70.0) / 5.0) ** 2) * 45.0  # Peak = 45
    s_0k_max = float(np.max(s_cont))
    s_norm = s_cont / s_0k_max

    # 1. Harmonic only: y_max should be default headroom ~ 1.18
    xlim_harm, ylim_harm = engine.get_smart_phonon_limits(hw_grid, s_norm)
    assert ylim_harm[0] == 0.0
    assert ylim_harm[1] == pytest.approx(1.18, rel=1e-3)

    # 2. Anharmonic with sharp thermal peak at 150 K (e.g. 165 eV^-1 -> dens = 3.66)
    s_anh_150K = np.exp(-0.5 * ((hw_grid - 72.0) / 2.0) ** 2) * 165.0
    anharmonic_dict = {
        150.0: {
            "s_anh": s_anh_150K,
            "vdos": np.ones_like(hw_grid),
        }
    }
    xlim_anh, ylim_anh = engine.get_smart_phonon_limits(
        hw_grid, s_norm, anharmonic_dict=anharmonic_dict, s_0k_max=s_0k_max
    )
    expected_peak_dens = float(np.max(s_anh_150K)) / s_0k_max
    assert ylim_anh[0] == 0.0
    # y_max should accommodate the peak with ~15% headroom (> 4.0)
    assert ylim_anh[1] > expected_peak_dens
    assert ylim_anh[1] == pytest.approx(expected_peak_dens * 1.15, rel=1e-3)

    # 3. User override via config
    cfg_user = ToolkitConfig(
        pristine_cell_path=str(pristine_file),
        foundation_potential_path=str(tmp_path / "pot.txt"),
        workspace_dir=str(tmp_path / "ws"),
        phonon_y_min=0.0,
        phonon_y_max=5.0,
    )
    engine_user = LineshapeEngine(cfg_user, mock_calc)
    _, ylim_user = engine_user.get_smart_phonon_limits(
        hw_grid, s_norm, anharmonic_dict=anharmonic_dict, s_0k_max=s_0k_max
    )
    assert ylim_user == (0.0, 5.0)


def test_acoustic_sum_rule_and_translation_projection():
    """Verify that Acoustic Sum Rule symmetrization and projection enforces translational invariance."""
    from hydeph.lineshape import apply_acoustic_sum_rule_and_projection
    from ase.vibrations.data import VibrationsData

    # 4-atom chain or cubic cell
    atoms = Atoms("C4", positions=[[0, 0, 0], [1.5, 0, 0], [3.0, 0, 0], [4.5, 0, 0]], cell=[6, 6, 6], pbc=True)
    natoms = len(atoms)

    # Synthetic Hessian with random translational violation (row sums != 0)
    rng = np.random.RandomState(42)
    H_rand = rng.normal(0.0, 1.0, (3 * natoms, 3 * natoms))
    H_raw = 0.5 * (H_rand + H_rand.T) + 5.0 * np.eye(3 * natoms)

    # Initial row sums violate ASR
    initial_row_sums = np.sum(H_raw.reshape(natoms, 3, natoms, 3), axis=2)
    assert np.max(np.abs(initial_row_sums)) > 0.5

    # Apply ASR + projection
    H_clean = apply_acoustic_sum_rule_and_projection(H_raw, atoms)

    # 1. Row sums are now zero within numerical precision
    clean_row_sums = np.sum(H_clean.reshape(natoms, 3, natoms, 3), axis=2)
    assert np.allclose(clean_row_sums, 0.0, atol=1e-12)

    # 2. Diagonalize: exactly 3 acoustic translation zero modes (0.0 meV)
    vdata = VibrationsData(atoms, H_clean.reshape(natoms, 3, natoms, 3))
    energies = vdata.get_energies()
    # Sort by real energy
    re_energies = np.sort(np.real(energies) * 1000.0)
    # The 3 smallest modes must be exactly 0.0 meV
    assert np.allclose(re_energies[:3], 0.0, atol=1e-6)


def test_rattle_mode_strategies_hybrid_core_whole(tmp_path):
    """Verify that rattle_mode ('hybrid', 'core', 'whole') operates as configured."""
    from hydeph.frame_sampler import FrameSampler

    pristine_file = tmp_path / "pristine.xyz"
    # Create 8-atom supercell where atom 0 is at origin and atom 7 is far
    atoms = Atoms("C8", positions=[
        [0, 0, 0], [1.5, 0, 0], [0, 1.5, 0], [1.5, 1.5, 0],
        [0, 0, 1.5], [1.5, 0, 1.5], [0, 1.5, 1.5], [5.0, 5.0, 5.0]
    ], cell=[8, 8, 8], pbc=True)
    atoms.info["defect_center"] = [0.0, 0.0, 0.0]
    import ase.io
    ase.io.write(str(pristine_file), atoms)

    # 1. "core" mode: atom 7 (far away at r > 6 A) is heavily dampened
    cfg_core = ToolkitConfig(
        pristine_cell_path=str(pristine_file),
        foundation_potential_path=str(tmp_path / "pot.txt"),
        workspace_dir=str(tmp_path / "ws_core"),
        rattle_mode="core",
        rattle_cutoff=3.0,
        n_phonon_frames=6,
    )
    sampler_core = FrameSampler(cfg_core)
    frames_core = sampler_core.sample_defect_core_rattling(atoms, n_frames=6)
    assert len(frames_core) == 6
    disp_far_core = [np.linalg.norm(f.positions[7] - atoms.positions[7]) for f in frames_core]

    # 2. "whole" mode: atom 7 rattles uniformly with full amplitude
    cfg_whole = ToolkitConfig(
        pristine_cell_path=str(pristine_file),
        foundation_potential_path=str(tmp_path / "pot.txt"),
        workspace_dir=str(tmp_path / "ws_whole"),
        rattle_mode="whole",
        n_phonon_frames=6,
    )
    sampler_whole = FrameSampler(cfg_whole)
    frames_whole = sampler_whole.sample_defect_core_rattling(atoms, n_frames=6)
    disp_far_whole = [np.linalg.norm(f.positions[7] - atoms.positions[7]) for f in frames_whole]
    assert np.mean(disp_far_whole) > 2.0 * np.mean(disp_far_core)

    # 3. "hybrid" mode (default): frame 0 (i % 3 == 0) rattles uniformly, frame 1 dampens outer
    cfg_hybrid = ToolkitConfig(
        pristine_cell_path=str(pristine_file),
        foundation_potential_path=str(tmp_path / "pot.txt"),
        workspace_dir=str(tmp_path / "ws_hybrid"),
        rattle_mode="hybrid",
        rattle_cutoff=3.0,
        n_phonon_frames=6,
    )
    sampler_hybrid = FrameSampler(cfg_hybrid)
    frames_hybrid = sampler_hybrid.sample_defect_core_rattling(atoms, n_frames=6)
    disp_0 = np.linalg.norm(frames_hybrid[0].positions[7] - atoms.positions[7])  # whole
    disp_1 = np.linalg.norm(frames_hybrid[1].positions[7] - atoms.positions[7])  # core
    assert disp_0 > disp_1


def test_foundation_bulk_stability_diagnostic():
    """Verify check_foundation_bulk_stability detects imaginary modes on unstable bulk."""
    from hydeph.pipeline import check_foundation_bulk_stability
    from ase.calculators.calculator import Calculator, all_changes

    atoms = Atoms("C2", positions=[[0, 0, 0], [1.5, 0, 0]], cell=[4, 4, 4], pbc=True)

    # 1. Stable harmonic potential: all real frequencies
    class StableCalc(Calculator):
        implemented_properties = ["energy", "forces"]
        def calculate(self, atoms=None, properties=["energy", "forces"], system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            self.results["energy"] = 0.0
            # Restoring spring forces: F = -k * (r - r0)
            dr = atoms.positions[1] - atoms.positions[0] - np.array([1.5, 0, 0])
            f = np.zeros((2, 3))
            f[0] = 5.0 * dr
            f[1] = -5.0 * dr
            self.results["forces"] = f

    n_im, worst = check_foundation_bulk_stability(atoms, StableCalc())
    assert n_im == 0
    assert worst == 0.0

    # 2. Unstable inverted potential: negative curvature / imaginary mode
    class UnstableCalc(Calculator):
        implemented_properties = ["energy", "forces"]
        def calculate(self, atoms=None, properties=["energy", "forces"], system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            self.results["energy"] = 0.0
            dr = atoms.positions[1] - atoms.positions[0] - np.array([1.5, 0, 0])
            # Inverted repulsive spring: F = +k * dr (negative curvature)
            f = np.zeros((2, 3))
            f[0] = -5.0 * dr
            f[1] = 5.0 * dr
            self.results["forces"] = f

    n_im_unstable, worst_unstable = check_foundation_bulk_stability(atoms, UnstableCalc())
    assert n_im_unstable > 0
    assert worst_unstable > 0.0


