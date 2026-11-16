"""Tests for physical anharmonic sampling and classical-to-quantum relations in HyDePh."""

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from hydeph.config import ToolkitConfig
from hydeph.lineshape import LineshapeEngine, HBAR_EV_FS, KB_MEV_K
from hydeph.calculator import PairedDefectCalculator


class SpyHarmonicCalculator(Calculator):
    """Harmonic displaced-oscillator calculator that logs every call to get_potential_energy."""

    implemented_properties = ["energy", "forces"]

    def __init__(self, k: float = 10.0, r0: float = 0.0, offset_eV: float = 0.0, label_tag: str = "calc"):
        super().__init__()
        self.k = k
        self.r0 = r0
        self.offset_eV = offset_eV
        self.label_tag = label_tag
        self.call_count = 0

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.call_count += 1
        pos = atoms.positions
        dr = pos - self.r0
        e = 0.5 * self.k * np.sum(dr ** 2) + self.offset_eV
        f = -self.k * dr
        self.results["energy"] = float(e)
        self.results["forces"] = np.asarray(f, dtype=float)


def test_excited_state_calculator_is_called_in_emission_md(tmp_path):
    """Verify that emission MD runs on the excited-state potential and evaluates PairedDefectCalculator."""
    pristine_file = tmp_path / "pristine.xyz"
    pristine = Atoms("C2", positions=[[0, 0, 0], [1.5, 1.5, 1.5]], cell=[6, 6, 6], pbc=True)
    import ase.io
    ase.io.write(str(pristine_file), pristine)

    cfg = ToolkitConfig(
        pristine_cell_path=str(pristine_file),
        foundation_potential_path=str(tmp_path / "pot.txt"),
        workspace_dir=str(tmp_path / "ws"),
        temperatures=[150.0],
        anharmonic_md_steps=30,
        anharmonic_thermalization_steps=10,
        anharmonic_n_trajectories=1,
    )

    calc_gs = SpyHarmonicCalculator(k=15.0, r0=0.0, offset_eV=0.0, label_tag="GS")
    calc_es = SpyHarmonicCalculator(k=15.0, r0=0.05, offset_eV=2.5, label_tag="ES")
    paired_calc = PairedDefectCalculator(calc_gs, calc_es)

    engine = LineshapeEngine(cfg, calc_gs)
    engine.calc_es = calc_es
    engine.paired_calc = paired_calc

    at_gs = pristine.copy()
    at_es = pristine.copy()
    at_es.positions += 0.05

    hw_grid = np.linspace(1.0, 100.0, 100)
    s_cont = np.exp(-0.5 * ((hw_grid - 40.0) / 10.0) ** 2) * 1000.0

    res = engine.run_anharmonic_sampling(at_gs, at_es, hw_grid, s_cont, total_S=1.8)

    # Verify ES calculator was actively called during MD
    assert calc_es.call_count > 10, f"Expected calc_es to be called during MD, got {calc_es.call_count}"
    assert calc_gs.call_count > 10, f"Expected calc_gs to be called during Delta U evaluation, got {calc_gs.call_count}"
    assert 150.0 in res
    assert res[150.0]["is_physical_paired"] is True
    assert "total_S_anh" in res[150.0]


def test_analytic_classical_fluctuation_dissipation_recovery():
    """Verify that S_MD(hw) = \tilde{C}_cl(hw) / (2 * kB * T * hw) recovers analytic S_k."""
    HBAR_MEV_FS = HBAR_EV_FS * 1000.0
    T = 300.0
    kbt = KB_MEV_K * T

    modes = [(25.0, 0.8), (75.0, 1.2)]
    dt_fs = 2.0
    t = np.arange(0, 6000.0, dt_fs)

    C_cl = np.zeros_like(t)
    for hw_k, S_k in modes:
        wk_fs = (hw_k / 1000.0) / HBAR_EV_FS
        C_cl += 2.0 * S_k * hw_k * kbt * np.cos(wk_fs * t)

    gamma_acf = 2.0
    gamma_acf_fs = (gamma_acf / 1000.0) / HBAR_EV_FS
    acf_window = np.exp(-0.5 * (gamma_acf_fs * t) ** 2)
    hann = 0.5 * (1.0 + np.cos(np.pi * t / t[-1]))
    weights = C_cl * acf_window * hann * dt_fs
    weights[0] *= 0.5
    weights[-1] *= 0.5

    hw_grid = np.linspace(1.0, 150.0, 1500)
    w_grid_rad_fs = (hw_grid / 1000.0) / HBAR_EV_FS
    j_cl = weights @ np.cos(np.outer(t, w_grid_rad_fs))

    filter_acoustic = 1.0 - np.exp(- (hw_grid / 5.0) ** 4)
    s_md_meV = (2.0 / (np.pi * HBAR_MEV_FS)) * (j_cl / (2.0 * kbt * hw_grid)) * filter_acoustic

    # Integrated S around mode 1 (25 meV) and mode 2 (75 meV)
    trapz_fn = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    mask_1 = (hw_grid >= 15.0) & (hw_grid <= 35.0)
    mask_2 = (hw_grid >= 65.0) & (hw_grid <= 85.0)
    s1_rec = float(trapz_fn(s_md_meV[mask_1], hw_grid[mask_1]))
    s2_rec = float(trapz_fn(s_md_meV[mask_2], hw_grid[mask_2]))

    assert pytest.approx(0.8, rel=0.05) == s1_rec
    assert pytest.approx(1.2, rel=0.05) == s2_rec
    # Ratio between modes should match 0.8 / 1.2 = 0.667 without frequency bias
    assert pytest.approx(0.8 / 1.2, rel=0.03) == s1_rec / s2_rec


def test_paired_defect_calculator_skips_shared_foundation_in_energy_gap():
    """Verify PairedDefectCalculator cancels shared calc_base in get_energy_gap identically without calling it."""
    from hydeph.calculator import HybridDefectCalculator

    at = Atoms("Si2", positions=[[0, 0, 0], [1.5, 1.5, 1.5]], cell=[5, 5, 5], pbc=True)
    shared_base = SpyHarmonicCalculator(k=10.0, r0=0.0, offset_eV=-100.0)
    delta_gs = SpyHarmonicCalculator(k=1.0, r0=0.0, offset_eV=-1.0)
    delta_es = SpyHarmonicCalculator(k=2.0, r0=0.05, offset_eV=1.5)

    h_gs = HybridDefectCalculator(calc_base=shared_base, calc_delta=delta_gs, energy_offset_per_atom=0.25)
    h_es = HybridDefectCalculator(calc_base=shared_base, calc_delta=delta_es, energy_offset_per_atom=-0.50)
    paired = PairedDefectCalculator(h_gs, h_es)

    calls_before = shared_base.call_count
    du = paired.get_energy_gap(at)
    assert shared_base.call_count == calls_before

    # Verify exact numerical equivalence with V_ES(R) - V_GS(R)
    du_full = float(h_es.get_potential_energy(at) - h_gs.get_potential_energy(at))
    assert du == pytest.approx(du_full, abs=1e-12)


def test_paired_defect_calculator_skips_shared_foundation_by_matching_path(tmp_path, monkeypatch):
    """Verify PairedDefectCalculator cancels distinct CPUNEP calc_base objects sharing the same potential path."""
    import hydeph.calculator as calc_mod
    from hydeph.calculator import HybridDefectCalculator

    pot_file = tmp_path / "nep_foundation.txt"
    pot_file.write_text("dummy", encoding="utf-8")

    created_bases = []

    class _SpyCPUNEP(SpyHarmonicCalculator):
        def __init__(self, path):
            super().__init__(k=10.0, r0=0.0, offset_eV=-100.0)
            self.path = path
            created_bases.append(self)

    monkeypatch.setattr(calc_mod, "CPUNEP", _SpyCPUNEP)
    # Also ensure load_foundation_calculator sees our patched CPUNEP
    at = Atoms("Si2", positions=[[0, 0, 0], [1.5, 1.5, 1.5]], cell=[5, 5, 5], pbc=True)
    delta_gs = SpyHarmonicCalculator(k=1.0, r0=0.0, offset_eV=-1.0)
    delta_es = SpyHarmonicCalculator(k=2.0, r0=0.05, offset_eV=1.5)

    h_gs = HybridDefectCalculator(
        foundation_pot_path=str(pot_file),
        calc_delta=delta_gs,
        energy_offset_per_atom=0.25,
    )
    h_es = HybridDefectCalculator(
        foundation_pot_path=str(pot_file),
        calc_delta=delta_es,
        energy_offset_per_atom=-0.50,
    )
    assert len(created_bases) == 2
    assert h_gs.calc_base is not h_es.calc_base

    paired = PairedDefectCalculator(h_gs, h_es)
    calls_gs_before = h_gs.calc_base.call_count
    calls_es_before = h_es.calc_base.call_count
    du = paired.get_energy_gap(at)
    assert h_gs.calc_base.call_count == calls_gs_before
    assert h_es.calc_base.call_count == calls_es_before

    du_full = float(h_es.get_potential_energy(at) - h_gs.get_potential_energy(at))
    assert du == pytest.approx(du_full, abs=1e-12)


def test_run_anharmonic_sampling_preserves_low_energy_acoustic_mode(tmp_path, monkeypatch):
    """Verify MD-ACF acoustic filter preserves ~6 meV low-energy modes even when omega_0 ~ 50 meV."""
    import hydeph.lineshape as ls_mod

    p_file = tmp_path / "p.xyz"
    at = Atoms("Si2", positions=[[0, 0, 0], [1.5, 1.5, 1.5]], cell=[5, 5, 5], pbc=True)
    import ase.io
    ase.io.write(str(p_file), at)

    cfg = ToolkitConfig(
        pristine_cell_path=str(p_file),
        foundation_potential_path=str(tmp_path / "f.txt"),
        workspace_dir=str(tmp_path),
        enable_anharmonic=True,
        temperatures=[100.0],
        anharmonic_n_trajectories=1,
        anharmonic_md_steps=3000,
        anharmonic_dt_fs=1.0,
        anharmonic_coupling_regime="cumulant",
        parallel_hessian=False,
    )
    calc_gs = SpyHarmonicCalculator(k=10.0)
    engine = LineshapeEngine(cfg, calc_gs)

    # S_cont centered around 50 meV so omega_0 = 50 meV (previously clamped hw_cutoff = 8.0 meV)
    hw_grid = np.linspace(0.5, 120.0, 600)
    s_cont = np.exp(-0.5 * ((hw_grid - 50.0) / 4.0) ** 2) * 100.0

    # Mock single trajectory to emit exact classical fluctuation for a 6 meV mode (S=1.0) and 50 meV mode (S=1.0)
    T_k = 100.0
    kbt = KB_MEV_K * T_k
    dt_obs = 2.0
    t_arr = np.arange(1500, dtype=float) * dt_obs

    def fake_single_traj(**kwargs):
        rng = np.random.default_rng(kwargs.get("seed", 0))
        du_series = np.zeros_like(t_arr)
        for hw_k, S_k in [(6.0, 1.0), (50.0, 1.0)]:
            wk_fs = (hw_k / 1000.0) / HBAR_EV_FS
            phi = rng.uniform(0.0, 2.0 * np.pi)
            amp = np.sqrt(4.0 * S_k * hw_k * kbt)
            du_series += amp * np.cos(wk_fs * t_arr + phi)
        return du_series, np.zeros_like(t_arr), T_k

    monkeypatch.setattr(ls_mod, "_run_single_anharmonic_trajectory", fake_single_traj)
    res = engine.run_anharmonic_sampling(at, at, hw_grid, s_cont, total_S=1.0, e_min=-200.0, e_max=40.0, n_pts=400)
    s_anh = res[100.0]["s_anh"]
    trapz_fn = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    m_low = (hw_grid >= 3.5) & (hw_grid <= 10.0)
    s_low = float(trapz_fn(s_anh[m_low] / 1000.0, hw_grid[m_low]))
    # Previously ~0.24 when hw_cutoff=8.0 suppressed 6 meV; with hw_cutoff<=5.2 should retain > 0.55
    assert s_low > 0.55


def test_run_anharmonic_sampling_detrends_linear_gap_drift(tmp_path, monkeypatch):
    """Verify slow linear drift in Delta U(t) is detrended before ACF evaluation."""
    import hydeph.lineshape as ls_mod

    p_file = tmp_path / "p2.xyz"
    at = Atoms("Si2", positions=[[0, 0, 0], [1.5, 1.5, 1.5]], cell=[5, 5, 5], pbc=True)
    import ase.io
    ase.io.write(str(p_file), at)

    cfg = ToolkitConfig(
        pristine_cell_path=str(p_file),
        foundation_potential_path=str(tmp_path / "f.txt"),
        workspace_dir=str(tmp_path),
        enable_anharmonic=True,
        temperatures=[300.0],
        anharmonic_n_trajectories=1,
        anharmonic_md_steps=3000,
        anharmonic_dt_fs=1.0,
        anharmonic_coupling_regime="cumulant",
        parallel_hessian=False,
    )
    engine = LineshapeEngine(cfg, SpyHarmonicCalculator(k=10.0))

    hw_grid = np.linspace(0.5, 120.0, 600)
    s_cont = np.exp(-0.5 * ((hw_grid - 40.0) / 4.0) ** 2) * 100.0
    T_k = 300.0
    kbt = KB_MEV_K * T_k
    t_arr = np.arange(1500, dtype=float) * 2.0
    wk_fs = (40.0 / 1000.0) / HBAR_EV_FS
    base_osc = np.sqrt(4.0 * 1.0 * 40.0 * kbt) * np.cos(wk_fs * t_arr)

    def make_traj(drift_meV):
        def _fake(**kwargs):
            ramp = np.linspace(-0.5 * drift_meV, 0.5 * drift_meV, len(t_arr))
            return base_osc + ramp, np.zeros_like(t_arr), T_k
        return _fake

    monkeypatch.setattr(ls_mod, "_run_single_anharmonic_trajectory", make_traj(0.0))
    res_clean = engine.run_anharmonic_sampling(at, at, hw_grid, s_cont, total_S=1.0, e_min=-200.0, e_max=40.0, n_pts=400)

    monkeypatch.setattr(ls_mod, "_run_single_anharmonic_trajectory", make_traj(400.0))
    res_drift = engine.run_anharmonic_sampling(at, at, hw_grid, s_cont, total_S=1.0, e_min=-200.0, e_max=40.0, n_pts=400)

    assert np.allclose(res_clean[300.0]["s_anh"], res_drift[300.0]["s_anh"], rtol=1e-4, atol=1e-4)


def test_embedded_anharmonic_uses_core_force_gap_avoiding_bulk_noise(tmp_path):
    """Embedded supercells with f_es_mode use core-force Delta U(t), ignoring bulk gap noise."""

    class NoisyBulkGapCalc(Calculator):
        implemented_properties = ["energy", "forces"]

        def __init__(self, k_bond: float = 8.0, bulk_noise: float = 0.0):
            super().__init__()
            self.k_bond = k_bond
            self.bulk_noise = bulk_noise

        def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            pos = atoms.positions
            forces = np.zeros_like(pos)
            # Pair (0,1) core oscillator along x around dx0 = 1.5 A
            dx = pos[1, 0] - pos[0, 0] - 1.5
            e = 0.5 * self.k_bond * (dx ** 2)
            forces[0, 0] = self.k_bond * dx
            forces[1, 0] = -self.k_bond * dx
            # Pair (2,3) harmonic bath oscillator along x around dx0 = 1.5 A
            dx_b = pos[3, 0] - pos[2, 0] - 1.5
            e += 0.5 * self.k_bond * (dx_b ** 2)
            forces[2, 0] = self.k_bond * dx_b
            forces[3, 0] = -self.k_bond * dx_b
            if self.bulk_noise != 0.0:
                # Huge unphysical energy bias solely from outer/bulk atoms (2, 3)
                e += self.bulk_noise * np.sum((pos[2:] - np.array([[5.0, 0.0, 0.0], [6.5, 0.0, 0.0]])) ** 2)
            self.results["energy"] = float(e)
            self.results["forces"] = forces

    at_gs = Atoms(
        "Si4",
        positions=[[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [5.0, 0.0, 0.0], [6.5, 0.0, 0.0]],
        cell=[20.0, 20.0, 20.0],
        pbc=True,
    )
    at_gs.info["is_embedded"] = True
    at_es = at_gs.copy()
    at_es.positions[1, 0] += 0.08
    at_es.positions[0, 0] -= 0.08

    f_es_mode = np.array(
        [[-0.8, 0.0, 0.0], [0.8, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        dtype=float,
    )
    hw_grid = np.linspace(1.0, 150.0, 300)
    s_cont = np.zeros_like(hw_grid)

    pristine_path = tmp_path / "pristine.xyz"
    at_gs.write(str(pristine_path))

    cfg = ToolkitConfig(
        pristine_cell_path=str(pristine_path),
        foundation_potential_path=str(tmp_path / "pot.txt"),
        workspace_dir=str(tmp_path / "out"),
        temperatures=[100.0],
        anharmonic_dt_fs=1.0,
        anharmonic_thermalization_steps=10,
        anharmonic_md_steps=40,
        anharmonic_n_trajectories=1,
        parallel_hessian=False,
        dephasing_gamma_meV=1.0,
        zpl_energy_eV=1.5,
    )

    calc_gs = NoisyBulkGapCalc(bulk_noise=0.0)
    calc_es_clean = NoisyBulkGapCalc(bulk_noise=0.0)
    calc_es_noisy = NoisyBulkGapCalc(bulk_noise=50.0)

    eng_clean = LineshapeEngine(cfg, calc_gs)
    eng_clean.calc_es = calc_es_clean
    res_clean = eng_clean.run_anharmonic_sampling(
        at_gs, at_es, hw_grid, s_cont, total_S=1.0, e_min=-100.0, e_max=100.0, n_pts=201, f_es_mode=f_es_mode
    )

    eng_noisy = LineshapeEngine(cfg, calc_gs)
    eng_noisy.calc_es = calc_es_noisy
    res_noisy = eng_noisy.run_anharmonic_sampling(
        at_gs, at_es, hw_grid, s_cont, total_S=1.0, e_min=-100.0, e_max=100.0, n_pts=201, f_es_mode=f_es_mode
    )

    assert np.allclose(res_clean[100.0]["s_anh"], res_noisy[100.0]["s_anh"], rtol=1e-6, atol=1e-8)



def test_long_acf_window_skips_redundant_hann_broadening(tmp_path, monkeypatch):
    """When Gaussian ACF window decays <5% at t_max, skip redundant Hann double-apodization."""
    import hydeph.lineshape as ls_mod

    dt_vacf = 2.0
    n_samples = 1500  # t_max = 1500 fs -> exp(-0.5 * (2.0/658.2 * 1500)^2) ~ 3e-5 < 0.05
    t_series = np.arange(n_samples, dtype=float) * dt_vacf
    hw_0 = 40.0
    w_rad = (hw_0 / 1000.0) / HBAR_EV_FS
    du_series = 25.0 * np.cos(w_rad * t_series)
    v_opt_series = np.zeros_like(t_series)

    monkeypatch.setattr(
        ls_mod,
        "_run_single_anharmonic_trajectory",
        lambda **kwargs: (du_series.copy(), v_opt_series.copy(), 300.0),
    )

    pristine_path = tmp_path / "pristine.xyz"
    at_gs = Atoms("C2", positions=[[0, 0, 0], [1.5, 0, 0]], cell=[6, 6, 6], pbc=True)
    at_gs.write(str(pristine_path))

    cfg = ToolkitConfig(
        pristine_cell_path=str(pristine_path),
        foundation_potential_path=str(tmp_path / "pot.txt"),
        workspace_dir=str(tmp_path / "out"),
        temperatures=[300.0],
        anharmonic_dt_fs=1.0,
        anharmonic_thermalization_steps=10,
        anharmonic_md_steps=2 * n_samples,
        anharmonic_n_trajectories=1,
        anharmonic_acf_damping_meV=2.0,
        parallel_hessian=False,
    )
    eng = LineshapeEngine(cfg, SpyHarmonicCalculator())
    hw_grid = np.linspace(1.0, 100.0, 991)
    s_cont = np.zeros_like(hw_grid)
    res = eng.run_anharmonic_sampling(at_gs, at_gs.copy(), hw_grid, s_cont, total_S=1.0, e_min=-80.0, e_max=80.0, n_pts=201)
    s_anh = res[300.0]["s_anh"]
    above_half = hw_grid[s_anh >= 0.5 * np.max(s_anh)]
    fwhm = float(above_half[-1] - above_half[0])
    # Pure gamma_acf=2.0 meV Gaussian has FWHM = 2*sqrt(2*ln2)*2.0 = 4.71 meV (< 5.2 meV);
    # with redundant Hann on 1500 fs it widens noticeably.
    assert 4.3 <= fwhm <= 5.2



def test_variable_dispersion_smearing_conserves_total_s_and_smooths_acoustic():
    """Verify linear dispersion-matched smearing conserves integral S and broadens low modes."""
    from hydeph.lineshape import _smear_harmonic_spectral_density

    hw_grid = np.linspace(0.5, 130.0, 2600)
    hw_modes = np.array([25.0, 95.0], dtype=float)
    sk_modes = np.array([1.0, 1.0], dtype=float)
    s_var = _smear_harmonic_spectral_density(hw_grid, hw_modes, sk_modes, sigma_smear=1.93, variable_dispersion=True)

    trapz_fn = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    assert pytest.approx(2.0, rel=1e-2) == float(trapz_fn(s_var, hw_grid))
    pk_low = float(np.max(s_var[(hw_grid >= 15.0) & (hw_grid <= 35.0)]))
    pk_high = float(np.max(s_var[(hw_grid >= 85.0) & (hw_grid <= 105.0)]))
    # Acoustic peak uses broader sigma (~3.0 meV) than high-energy optical peak (~1.8 meV).
    assert pk_high > pk_low * 1.4



def test_embedded_acf_uses_dispersion_matched_damping(tmp_path, monkeypatch):
    """Embedded supercells with default smearing use broader low-hw than high-hw ACF damping."""
    import hydeph.lineshape as ls_mod

    dt_vacf = 2.0
    n_samples = 1500
    t_series = np.arange(n_samples, dtype=float) * dt_vacf
    w_lo = (20.0 / 1000.0) / HBAR_EV_FS
    w_hi = (90.0 / 1000.0) / HBAR_EV_FS
    du_series = 20.0 * np.cos(w_lo * t_series) + 20.0 * np.cos(w_hi * t_series)
    v_opt_series = np.zeros_like(t_series)

    monkeypatch.setattr(
        ls_mod,
        "_run_single_anharmonic_trajectory",
        lambda **kwargs: (du_series.copy(), v_opt_series.copy(), 300.0),
    )

    pristine_path = tmp_path / "pristine.xyz"
    at_gs = Atoms("C2", positions=[[0, 0, 0], [1.5, 0, 0]], cell=[6, 6, 6], pbc=True)
    at_gs.info["is_embedded"] = True
    at_gs.write(str(pristine_path))

    cfg = ToolkitConfig(
        pristine_cell_path=str(pristine_path),
        foundation_potential_path=str(tmp_path / "pot.txt"),
        workspace_dir=str(tmp_path / "out"),
        temperatures=[300.0],
        anharmonic_dt_fs=1.0,
        anharmonic_thermalization_steps=10,
        anharmonic_md_steps=2 * n_samples,
        anharmonic_n_trajectories=1,
        anharmonic_acf_damping_meV=2.0,
        phonon_smearing_meV=None,
        parallel_hessian=False,
    )
    eng = LineshapeEngine(cfg, SpyHarmonicCalculator())
    hw_grid = np.linspace(1.0, 120.0, 1191)
    res = eng.run_anharmonic_sampling(at_gs, at_gs.copy(), hw_grid, np.zeros_like(hw_grid), total_S=1.0, e_min=-80.0, e_max=80.0, n_pts=201)
    s_anh = res[300.0]["s_anh"]

    def fwhm_around(center, hw_arr, s_arr):
        m = (hw_arr >= center - 12.0) & (hw_arr <= center + 12.0)
        sub_h = hw_arr[m]
        sub_s = s_arr[m]
        above = sub_h[sub_s >= 0.5 * np.max(sub_s)]
        return float(above[-1] - above[0])

    assert fwhm_around(20.0, hw_grid, s_anh) > fwhm_around(90.0, hw_grid, s_anh) + 1.5
