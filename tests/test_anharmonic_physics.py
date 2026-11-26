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



def test_anharmonic_md_surface_gs_skips_es_dynamics(tmp_path):
    """Verify anharmonic_md_surface='gs' runs MD on calc_gs even when calc_es is attached."""
    pristine_path = tmp_path / "pristine.xyz"
    at_gs = Atoms("C2", positions=[[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], cell=[6.0, 6.0, 6.0], pbc=True)
    at_gs.info["is_embedded"] = True
    at_gs.write(str(pristine_path))

    cfg = ToolkitConfig(
        pristine_cell_path=str(pristine_path),
        foundation_potential_path=str(tmp_path / "pot.txt"),
        workspace_dir=str(tmp_path / "out"),
        temperatures=[100.0],
        anharmonic_md_steps=20,
        anharmonic_thermalization_steps=10,
        anharmonic_n_trajectories=1,
        anharmonic_md_surface="gs",
        parallel_hessian=False,
    )
    calc_gs = SpyHarmonicCalculator(k=10.0)
    calc_es = SpyHarmonicCalculator(k=10.0, offset_eV=2.0)
    eng = LineshapeEngine(cfg, calc_gs)
    eng.calc_es = calc_es
    hw_grid = np.linspace(1.0, 100.0, 100)
    f_core = np.array([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=float)
    eng.run_anharmonic_sampling(at_gs, at_gs.copy(), hw_grid, np.zeros_like(hw_grid), total_S=1.0, f_es_mode=f_core)
    assert calc_gs.call_count > 10
    assert calc_es.call_count == 0


def test_sub_acoustic_cutoff_and_bandwidth_scaled_acf(tmp_path, monkeypatch):
    """Verify hw_min_phonon suppresses sub-lattice 1/omega spikes and hw_max_phonon scales ACF damping."""
    import hydeph.lineshape as ls_mod

    p_file = tmp_path / "p_diamond.xyz"
    at = Atoms("C2", positions=[[0, 0, 0], [0.89, 0.89, 0.89]], cell=[3.57, 3.57, 3.57], pbc=True)
    at.info["is_embedded"] = True
    import ase.io
    ase.io.write(str(p_file), at)

    cfg = ToolkitConfig(
        pristine_cell_path=str(p_file),
        foundation_potential_path=str(tmp_path / "f.txt"),
        workspace_dir=str(tmp_path / "out"),
        enable_anharmonic=True,
        temperatures=[300.0],
        anharmonic_n_trajectories=1,
        anharmonic_md_steps=2000,
        anharmonic_dt_fs=1.0,
        anharmonic_coupling_regime="cumulant",
        parallel_hessian=False,
    )
    calc_gs = SpyHarmonicCalculator(k=10.0)
    engine = LineshapeEngine(cfg, calc_gs)
    hw_grid = np.linspace(0.5, 175.0, 700)
    s_cont = np.exp(-0.5 * ((hw_grid - 70.0) / 8.0) ** 2) * 80.0
    T_k = 300.0
    kbt = KB_MEV_K * T_k
    t_arr = np.arange(1000, dtype=float) * 2.0

    def fake_traj(**kwargs):
        du_series = np.zeros_like(t_arr)
        # Spurious 7.5 meV thermostat leak + real 35 meV and 70 meV lattice modes
        for hw_k, s_k in [(7.5, 0.25), (35.0, 0.8), (70.0, 2.0)]:
            wk_fs = (hw_k / 1000.0) / HBAR_EV_FS
            amp = np.sqrt(4.0 * s_k * hw_k * kbt)
            du_series += amp * np.cos(wk_fs * t_arr)
        return du_series, np.zeros_like(t_arr), T_k

    monkeypatch.setattr(ls_mod, "_run_single_anharmonic_trajectory", fake_traj)
    res_def = engine.run_anharmonic_sampling(at, at, hw_grid, s_cont, total_S=3.0, e_min=-300.0, e_max=50.0, n_pts=300)
    res_scaled = engine.run_anharmonic_sampling(
        at, at, hw_grid, s_cont, total_S=3.0, e_min=-300.0, e_max=50.0, n_pts=300,
        hw_min_phonon=26.0, hw_max_phonon=162.0,
    )
    trapz_fn = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    m_sub = (hw_grid >= 3.0) & (hw_grid <= 14.0)
    s_sub_def = float(trapz_fn(res_def[300.0]["s_anh"][m_sub] / 1000.0, hw_grid[m_sub]))
    s_sub_scaled = float(trapz_fn(res_scaled[300.0]["s_anh"][m_sub] / 1000.0, hw_grid[m_sub]))
    assert s_sub_scaled < 0.25 * s_sub_def

    res_narrow = engine.run_anharmonic_sampling(
        at, at, hw_grid, s_cont, total_S=3.0, e_min=-300.0, e_max=50.0, n_pts=300,
        hw_min_phonon=4.0, hw_max_phonon=26.0,
    )
    # Check wider phonon bandwidth broadens the 35 meV line smoothly relative to narrow-band crystals
    m_35 = (hw_grid >= 20.0) & (hw_grid <= 50.0)
    def fwhm(arr):
        sub_h = hw_grid[m_35]
        sub_s = arr[m_35]
        ab = sub_h[sub_s >= 0.5 * np.max(sub_s)]
        return float(ab[-1] - ab[0])

    assert fwhm(res_scaled[300.0]["s_anh"]) > fwhm(res_narrow[300.0]["s_anh"]) + 5.0


def test_anharmonic_planar_md_clamps_z_motion(tmp_path):
    """Verify anharmonic_planar_md=True clamps z-motion while preserving 2D thermal dynamics."""
    from hydeph.lineshape import _run_single_anharmonic_trajectory

    p_file = tmp_path / "p_sheet.xyz"
    at = Atoms(
        "C2",
        positions=[[0.0, 0.0, 10.0], [1.4, 0.0, 10.0]],
        cell=[8.0, 8.0, 20.0],
        pbc=True,
    )
    at.info["is_embedded"] = True
    import ase.io
    ase.io.write(str(p_file), at)
    pot_file = tmp_path / "pot.txt"
    pot_file.write_text("dummy")

    cfg = ToolkitConfig(
        pristine_cell_path=str(p_file),
        foundation_potential_path=str(pot_file),
        workspace_dir=str(tmp_path / "out"),
        spin_multiplicity=1,
        temperatures=[200.0],
        anharmonic_md_steps=20,
        anharmonic_thermalization_steps=10,
        anharmonic_n_trajectories=1,
        anharmonic_planar_md=True,
        parallel_hessian=False,
    )
    cfg.validate()

    cfg_bad = ToolkitConfig(
        pristine_cell_path=str(p_file),
        foundation_potential_path=str(pot_file),
        workspace_dir=str(tmp_path / "out"),
        spin_multiplicity=1,
        anharmonic_planar_md="yes",
    )
    with pytest.raises(ValueError, match="anharmonic_planar_md"):
        cfg_bad.validate()

    calc_gs = SpyHarmonicCalculator(k=10.0)
    z_seen = []

    class RecordingHarmonic(SpyHarmonicCalculator):
        def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
            z_seen.append(atoms.positions[:, 2].copy())
            super().calculate(atoms=atoms, properties=properties, system_changes=system_changes)

    rec_calc = RecordingHarmonic(k=10.0)
    u_opt = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=float)
    dr_fc = np.zeros((2, 3), dtype=float)
    f_core = np.array([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=float)

    du, _, teq = _run_single_anharmonic_trajectory(
        at_start=at,
        T=200.0,
        dt_fs=1.0,
        n_therm=10,
        n_steps=20,
        u_opt=u_opt,
        dr_fc=dr_fc,
        calc_gs=rec_calc,
        calc_es=None,
        paired_calc=None,
        seed=123,
        show_progress=False,
        f_core=f_core,
        r_ref=at.positions.copy(),
        planar_2d=True,
    )
    assert len(du) == 11
    assert teq > 50.0
    assert np.allclose(np.array(z_seen), 10.0, atol=1e-12)

    eng = LineshapeEngine(cfg, calc_gs)
    hw_grid = np.linspace(1.0, 100.0, 100)
    res = eng.run_anharmonic_sampling(at, at.copy(), hw_grid, np.zeros_like(hw_grid), total_S=1.0, f_es_mode=f_core)
    assert 200.0 in res


def test_detect_stacked_layers_and_layer_com_constraint(tmp_path):
    """Verify _detect_stacked_layers targets vdW multi-layer sheets and _LayerCOMConstraint pins sheet COM."""
    from ase.build import bulk
    from hydeph.lineshape import _detect_stacked_layers, _run_single_anharmonic_trajectory

    # 1. 3D cubic crystal (diamond 2x2x2) must not trigger stacked layer detection
    diam = bulk("C", "diamond", a=3.57, cubic=True).repeat((2, 2, 2))
    layers_d, ax_d = _detect_stacked_layers(diam)
    assert layers_d == [] and ax_d is None

    # 1b. 3D ionic rocksalt (NaCl, plane spacing 2.82 A == bond length) must not trigger detection
    nacl = bulk("NaCl", "rocksalt", a=5.64, cubic=True).repeat((2, 2, 2))
    layers_nacl, ax_nacl = _detect_stacked_layers(nacl)
    assert layers_nacl == [] and ax_nacl is None

    # 2. Single 2D monolayer in vacuum must not trigger stacked layer detection (< 2 sheets)
    mono = Atoms(
        "C8",
        positions=[[ix * 1.4, iy * 1.4, 10.0] for ix in range(2) for iy in range(4)],
        cell=[8.0, 8.0, 20.0],
        pbc=True,
    )
    layers_m, _ = _detect_stacked_layers(mono)
    assert layers_m == []

    # 3. 2-layer stacked 2D crystal (separated by 3.33 A) must be detected with 2 sheets along axis 2
    bilayer = Atoms(
        "B4N4B4N4",
        positions=[
            [x, y, z]
            for z in (0.0, 3.33)
            for x in (0.0, 1.25, 2.5, 3.75)
            for y in (0.0, 2.5)
        ],
        cell=[5.0, 5.0, 6.66],
        pbc=True,
    )
    layers_b, ax_b = _detect_stacked_layers(bilayer)
    assert ax_b == 2
    assert len(layers_b) == 2
    assert sorted(len(cl) for cl in layers_b) == [8, 8]

    # Run short trajectory with layer_com_axes=(0, 1) + planar_2d=True; sheet COM (x, y) must stay fixed
    m = bilayer.get_masses()
    ref_xy = [
        np.sum(bilayer.positions[cl, :2] * m[cl][:, None], axis=0) / np.sum(m[cl])
        for cl in layers_b
    ]
    seen_pos = []

    class RecordingHarmonic(SpyHarmonicCalculator):
        def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
            seen_pos.append(atoms.positions.copy())
            super().calculate(atoms=atoms, properties=properties, system_changes=system_changes)

    rec = RecordingHarmonic(k=10.0)
    u_opt = np.ones((len(bilayer), 3)) / np.sqrt(3.0 * len(bilayer))
    du, _, teq = _run_single_anharmonic_trajectory(
        at_start=bilayer,
        T=250.0,
        dt_fs=0.5,
        n_therm=10,
        n_steps=20,
        u_opt=u_opt,
        dr_fc=np.zeros((len(bilayer), 3)),
        calc_gs=rec,
        calc_es=None,
        paired_calc=None,
        seed=7,
        show_progress=False,
        planar_2d=True,
        layer_com_axes=(0, 1),
    )
    assert len(du) == 11 and teq > 50.0
    for pos in seen_pos:
        for cl, c0 in zip(layers_b, ref_xy):
            com_xy = np.sum(pos[cl, :2] * m[cl][:, None], axis=0) / np.sum(m[cl])
            assert np.allclose(com_xy, c0, atol=1e-10)
        assert np.allclose(pos[:, 2], bilayer.positions[:, 2], atol=1e-12)


def test_config_validation_and_dft_md_supercell_option(tmp_path):
    """Verify validation of anharmonic_layer_com_constraint / anharmonic_md_supercell and dft MD cell."""
    import ase.io
    from unittest.mock import patch

    p_file = tmp_path / "p.xyz"
    at = Atoms("C2", positions=[[0, 0, 0], [1.4, 0, 0]], cell=[6.0, 6.0, 6.0], pbc=True)
    ase.io.write(str(p_file), at)
    pot = tmp_path / "pot.txt"
    pot.write_text("dummy")

    with pytest.raises(ValueError, match="anharmonic_layer_com_constraint"):
        ToolkitConfig(
            pristine_cell_path=str(p_file),
            foundation_potential_path=str(pot),
            workspace_dir=str(tmp_path / "ws1"),
            spin_multiplicity=1,
            anharmonic_layer_com_constraint="bogus",
        ).validate()

    with pytest.raises(ValueError, match="anharmonic_md_supercell"):
        ToolkitConfig(
            pristine_cell_path=str(p_file),
            foundation_potential_path=str(pot),
            workspace_dir=str(tmp_path / "ws2"),
            spin_multiplicity=1,
            anharmonic_md_supercell="invalid",
        ).validate()

    cfg = ToolkitConfig(
        pristine_cell_path=str(p_file),
        foundation_potential_path=str(pot),
        workspace_dir=str(tmp_path / "ws3"),
        spin_multiplicity=1,
        supercell_repeat=[2, 2, 2],
        phonon_supercell_repeat=[3, 3, 3],
        enable_anharmonic=True,
        anharmonic_md_supercell="dft",
        parallel_hessian=False,
    )
    cfg.validate()

    at_dft = at.repeat((2, 2, 2))
    del at_dft[0]
    at_large = at.repeat((3, 3, 3))
    del at_large[0]
    at_large.info["is_embedded"] = True
    calc = SpyHarmonicCalculator(k=10.0)
    eng = LineshapeEngine(cfg, calc)

    seen_md_natoms = []
    with patch("hydeph.lineshape.embed_phonon_supercell", return_value=(at_large, np.zeros((len(at_large), 3)), None)):
        with patch.object(eng, "_run_vibrations_for_calculator", return_value=(np.full(3 * len(at_large), 0.05), lambda i: np.zeros((len(at_large), 3)))):
            def spy_md(g, e, *a, **kw):
                seen_md_natoms.append(len(g))
                return {}
            with patch.object(eng, "run_anharmonic_sampling", side_effect=spy_md):
                eng.calculate_spectral_density_and_lineshapes(at_dft, at_dft.copy())
    assert seen_md_natoms == [len(at_dft)]


def test_load_dft_optical_forces_rejects_local_geometry_mismatch(tmp_path):
    """Verify load_dft_optical_forces checks max per-atom displacement, not just whole-cell RMSD."""
    from unittest.mock import patch
    from hydeph.system_profile import load_dft_optical_forces

    ws = tmp_path / "ws"
    (ws / "qe_es_relax").mkdir(parents=True)
    (ws / "qe_gs_relax").mkdir(parents=True)
    (ws / "qe_es_relax" / "pw.out").write_text("dummy")
    (ws / "qe_gs_relax" / "pw.out").write_text("dummy")

    # 32 atoms: moving one atom by 0.07 A keeps whole-cell RMSD ~0.012 A (< 0.02),
    # which must still be rejected by the per-atom max-displacement guard.
    at_gs = Atoms("C32", positions=[[i * 2.0, 0.0, 0.0] for i in range(32)], cell=[64.0, 10.0, 10.0], pbc=True)
    at_es0 = at_gs.copy()
    at_es0.positions[0, 0] += 0.07

    from ase.calculators.singlepoint import SinglePointCalculator
    at_gs.calc = SinglePointCalculator(at_gs, forces=np.zeros((32, 3)))
    at_es0.calc = SinglePointCalculator(at_es0, forces=np.ones((32, 3)))

    def fake_read(path, index=-1):
        return at_es0 if "qe_es_relax" in str(path) else at_gs

    with patch("ase.io.read", side_effect=fake_read):
        res = load_dft_optical_forces(at_gs, at_gs.copy(), str(ws), force_state="absorption")
    assert res is None
