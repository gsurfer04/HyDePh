"""Tests for Batch 9: Physical Paired Cumulant Anharmonic Module (A6)."""

import pytest
import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from hydeph.config import ToolkitConfig
from hydeph.lineshape import (
    LineshapeEngine,
    KB_MEV_K,
)


class MockPairedCalc(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, k=5.0, offset_eV=2.0, **kwargs):
        super().__init__(**kwargs)
        self.k = k
        self.offset_eV = offset_eV
        self.r0 = None

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        pos = atoms.positions
        if self.r0 is None:
            self.r0 = pos.copy()
        dr = pos - self.r0
        self.results["energy"] = float(0.5 * self.k * np.sum(dr ** 2) + self.offset_eV)
        self.results["forces"] = -self.k * dr


def test_harmonic_quantum_correction_factor():
    """Verify symmetric harmonic quantum correction Q(hw, T) properties."""
    T = 100.0
    hw_grid = np.array([0.001, 1.0, 10.0, 50.0, 100.0])

    x = hw_grid / (KB_MEV_K * T)
    q_factor = x / (2.0 * np.tanh(0.5 * x))

    # Low frequency limit: Q -> 1
    assert q_factor[0] == pytest.approx(1.0, rel=1e-3)

    # High frequency limit: Q -> x / 2 = hw / (2 kB T)
    assert q_factor[-1] == pytest.approx(x[-1] / 2.0, rel=1e-2)

    # Monotonically increasing with frequency
    for i in range(len(q_factor) - 1):
        assert q_factor[i] < q_factor[i + 1]


def test_nve_production_dynamics_and_paired_delta_u(tmp_path):
    """Verify NVT thermalization -> NVE production and physical Delta U(t) = V_ES - V_GS."""
    pristine_file = tmp_path / "pristine.xyz"
    pristine = Atoms("C2", positions=[[0, 0, 0], [1.5, 1.5, 1.5]], cell=[4, 4, 4], pbc=True)
    import ase.io
    ase.io.write(str(pristine_file), pristine)

    cfg = ToolkitConfig(
        pristine_cell_path=str(pristine_file),
        foundation_potential_path=str(tmp_path / "pot.txt"),
        workspace_dir=str(tmp_path / "ws"),
        temperatures=[77.0],
        anharmonic_md_steps=20,
        anharmonic_thermalization_steps=10,
        anharmonic_coupling_regime="cumulant",
    )

    calc_gs = MockPairedCalc(k=5.0, offset_eV=0.0)
    calc_es = MockPairedCalc(k=5.0, offset_eV=2.2)  # Vertical offset 2.2 eV = 2200 meV

    engine = LineshapeEngine(cfg, calc_gs)
    engine.calc_es = calc_es

    at_gs = pristine.copy()
    at_es = pristine.copy()
    at_es.positions[1] += [0.05, 0.05, 0.05]

    hw_grid = np.linspace(1.0, 80.0, 100)
    s_cont = np.exp(-0.5 * ((hw_grid - 30.0) / 10.0) ** 2) * 1000.0

    res = engine.run_anharmonic_sampling(at_gs, at_es, hw_grid, s_cont, total_S=1.5)

    assert 77.0 in res
    data_77 = res[77.0]
    assert data_77["regime"] == "cumulant"
    assert data_77["method"] == "cumulant_energy_gap"
    assert data_77["is_physical_paired"] is True
    assert "sigma_du" in data_77
    assert "lineshape" in data_77
    assert len(data_77["lineshape"]) > 0


def test_backward_compatible_regimes(tmp_path):
    """Verify that legacy 'quantum' and 'semiclassical' regimes still execute cleanly."""
    pristine_file = tmp_path / "pristine.xyz"
    pristine = Atoms("C2", positions=[[0, 0, 0], [1.5, 1.5, 1.5]], cell=[4, 4, 4], pbc=True)
    import ase.io
    ase.io.write(str(pristine_file), pristine)

    cfg = ToolkitConfig(
        pristine_cell_path=str(pristine_file),
        foundation_potential_path=str(tmp_path / "pot.txt"),
        workspace_dir=str(tmp_path / "ws"),
        temperatures=[50.0],
        anharmonic_md_steps=15,
        anharmonic_thermalization_steps=5,
        anharmonic_coupling_regime="quantum",
    )

    calc = MockPairedCalc(k=5.0, offset_eV=0.0)
    engine = LineshapeEngine(cfg, calc)

    at_gs = pristine.copy()
    at_es = pristine.copy()
    at_es.positions[1] += [0.05, 0.05, 0.05]

    hw_grid = np.linspace(1.0, 80.0, 100)
    s_cont = np.exp(-0.5 * ((hw_grid - 30.0) / 10.0) ** 2) * 1000.0

    res_quant = engine.run_anharmonic_sampling(at_gs, at_es, hw_grid, s_cont, total_S=1.5)
    assert res_quant[50.0]["regime"] == "quantum"

    cfg.anharmonic_coupling_regime = "semiclassical"
    engine_semi = LineshapeEngine(cfg, calc)
    res_semi = engine_semi.run_anharmonic_sampling(at_gs, at_es, hw_grid, s_cont, total_S=1.5)
    assert res_semi[50.0]["regime"] == "semiclassical"


def test_multi_trajectory_and_linderalv_acf_windowing(tmp_path):
    """Verify multi-trajectory ensemble sampling and Linderälv Gaussian ACF windowing."""
    pristine_file = tmp_path / "pristine.xyz"
    pristine = Atoms("C2", positions=[[0, 0, 0], [1.5, 1.5, 1.5]], cell=[4, 4, 4], pbc=True)
    import ase.io
    ase.io.write(str(pristine_file), pristine)

    cfg = ToolkitConfig(
        pristine_cell_path=str(pristine_file),
        foundation_potential_path=str(tmp_path / "pot.txt"),
        workspace_dir=str(tmp_path / "ws"),
        temperatures=[77.0, 300.0],
        anharmonic_md_steps=20,
        anharmonic_thermalization_steps=10,
        anharmonic_n_trajectories=2,
        anharmonic_acf_damping_meV=2.0,
        dephasing_temperature_coefficient=5.5e-5,
    )

    calc_gs = MockPairedCalc(k=5.0, offset_eV=0.0)
    calc_es = MockPairedCalc(k=5.0, offset_eV=2.0)
    engine = LineshapeEngine(cfg, calc_gs)
    engine.calc_es = calc_es

    at_gs = pristine.copy()
    at_es = pristine.copy()
    at_es.positions[1] += [0.05, 0.05, 0.05]

    hw_grid = np.linspace(1.0, 100.0, 100)
    s_cont = np.exp(-0.5 * ((hw_grid - 50.0) / 10.0) ** 2) * 1000.0

    res = engine.run_anharmonic_sampling(at_gs, at_es, hw_grid, s_cont, total_S=2.0)
    assert 77.0 in res and 300.0 in res
    assert res[77.0]["n_trajectories"] == 2
    assert res[300.0]["n_trajectories"] == 2
    assert "s_anh" in res[77.0] and res[77.0]["s_anh"] is not None
    a_300 = res[300.0]["lineshape"]
    eg = res[300.0]["energies"]
    zpl_idx = np.argmin(np.abs(eg))
    assert a_300[zpl_idx] > 0.0


def test_anharmonic_uses_dft_cell_relaxes_es_and_computes_contiguous_fwhm(tmp_path):
    """Verify embedded runs pass the DFT cell to anharmonic MD, pre-relax ES, and avoid ZPL-to-PSB FWHM span."""
    import ase.io
    from unittest.mock import patch
    from ase import Atoms
    from ase.calculators.calculator import Calculator, all_changes
    from hydeph.config import ToolkitConfig
    from hydeph.lineshape import LineshapeEngine

    class SimpleHarmonic(Calculator):
        implemented_properties = ["energy", "forces"]

        def __init__(self, k=8.0, shift=0.0):
            super().__init__()
            self.k = k
            self.shift = shift

        def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            disp = atoms.positions - self.shift
            self.results["energy"] = float(0.5 * self.k * np.sum(disp ** 2))
            self.results["forces"] = -self.k * disp

    # 1. Verify contiguous FWHM ignores an isolated ZPL spike separated from a dominant PSB peak
    p_file = tmp_path / "pristine.xyz"
    uc = Atoms("C2", positions=[[0, 0, 0], [1.4, 1.4, 1.4]], cell=[2.8, 2.8, 2.8], pbc=True)
    ase.io.write(str(p_file), uc)
    cfg = ToolkitConfig(
        pristine_cell_path=str(p_file),
        foundation_potential_path="dummy.txt",
        workspace_dir=str(tmp_path / "ws"),
        temperatures=[10.0],
        anharmonic_md_steps=16,
        anharmonic_thermalization_steps=6,
        anharmonic_n_trajectories=1,
        supercell_repeat=(1, 1, 1),
        phonon_supercell_repeat=(2, 2, 2),
        defect_type="vacancy",
        defect_site_index=0,
    )
    engine = LineshapeEngine(cfg, SimpleHarmonic(k=8.0, shift=0.0))
    engine.calc_es = SimpleHarmonic(k=8.0, shift=0.05)

    at_dft = Atoms("C2", positions=[[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]], cell=[2.8, 2.8, 2.8], pbc=True)
    hw_grid = np.linspace(5.0, 120.0, 80)
    s_cont = np.exp(-0.5 * ((hw_grid - 40.0) / 10.0) ** 2) * 100.0

    # Pre-MD relaxation should bring initial force down on at_es without raising
    res_md = engine.run_anharmonic_sampling(
        at_dft,
        Atoms("C2", positions=[[0.15, 0.0, 0.0], [1.4, 0.0, 0.0]], cell=[2.8, 2.8, 2.8], pbc=True),
        hw_grid,
        s_cont,
        total_S=1.0,
    )
    assert res_md[10.0]["fwhm_meV"] >= 0.0

    # 2. Verify separated ZPL + PSB lineshape gives contiguous PSB FWHM, not ZPL-to-PSB distance (~200 meV)
    eg = np.linspace(-400.0, 50.0, 451)
    a_two_peak = np.exp(-0.5 * ((eg + 200.0) / 25.0) ** 2) + 0.7 * np.exp(-0.5 * (eg / 1.5) ** 2)
    with patch("hydeph.lineshape.compute_quantum_lineshape", return_value=(eg, a_two_peak)):
        res_fwhm = engine.run_anharmonic_sampling(at_dft, at_dft.copy(), hw_grid, s_cont, total_S=1.0)
    assert res_fwhm[10.0]["fwhm_meV"] < 80.0

    # 3. Verify calculate_spectral_density_and_lineshapes passes embedded phonon supercell to MD
    seen_natoms = []
    orig_anh = engine.run_anharmonic_sampling

    def spy_anh(g_cell, e_cell, *args, **kwargs):
        seen_natoms.append((len(g_cell), len(e_cell)))
        return orig_anh(g_cell, e_cell, *args, **kwargs)

    engine_harm_vib_modes = np.array([0.04, 0.05, 0.06])
    with patch.object(engine, "_run_vibrations_for_calculator", return_value=(engine_harm_vib_modes, lambda i: np.zeros((15, 3)))):
        with patch.object(engine, "run_anharmonic_sampling", side_effect=spy_anh):
            with patch.object(engine, "plot_all", return_value={}):
                engine.calculate_spectral_density_and_lineshapes(at_dft, at_dft.copy())
    assert seen_natoms == [(15, 15)]


def test_e_zpl_ev_propagation_and_consistent_dephasing_defaults(tmp_path):
    """Verify e_zpl_eV from CC propagates to summary/I_PL CSV and anharmonic uses consistent default gamma."""
    import csv
    import json
    import ase.io
    from unittest.mock import patch
    from ase import Atoms
    from ase.calculators.calculator import Calculator, all_changes
    from hydeph.config import ToolkitConfig
    from hydeph.lineshape import LineshapeEngine

    class SimpleHarmonic(Calculator):
        implemented_properties = ["energy", "forces"]

        def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            self.results["energy"] = float(4.0 * np.sum(atoms.positions ** 2))
            self.results["forces"] = -8.0 * atoms.positions

    p_file = tmp_path / "pristine.xyz"
    uc = Atoms("C2", positions=[[0, 0, 0], [1.4, 1.4, 1.4]], cell=[2.8, 2.8, 2.8], pbc=True)
    ase.io.write(str(p_file), uc)
    ws = tmp_path / "ws_zpl"
    ws.mkdir()
    cfg = ToolkitConfig(
        pristine_cell_path=str(p_file),
        foundation_potential_path="dummy.txt",
        workspace_dir=str(ws),
        temperatures=[10.0],
        enable_anharmonic=True,
        anharmonic_md_steps=8,
        anharmonic_thermalization_steps=4,
        anharmonic_n_trajectories=1,
        supercell_repeat=(1, 1, 1),
        phonon_supercell_repeat=(1, 1, 1),
        defect_type="vacancy",
        defect_site_index=0,
        zpl_energy_eV=None,
    )
    engine = LineshapeEngine(cfg, SimpleHarmonic())
    at_gs = uc.copy()
    at_es = uc.copy()
    at_es.positions[1, 0] += 0.05

    seen_gammas = []
    from hydeph import lineshape as ls_mod

    orig_q_ls = ls_mod.compute_quantum_lineshape

    def spy_q_ls(*args, **kwargs):
        seen_gammas.append(kwargs.get("dephasing_gamma_meV"))
        return orig_q_ls(*args, **kwargs)

    with patch.object(engine, "_run_vibrations_for_calculator", return_value=(np.array([0.04, 0.05]), lambda i: np.zeros((2, 3)))):
        with patch.object(engine, "plot_configuration_coordinate", return_value={"e_zpl_eV": 1.25}):
            with patch.object(engine, "plot_all", return_value={}):
                with patch("hydeph.lineshape.compute_quantum_lineshape", side_effect=spy_q_ls):
                    res = engine.calculate_spectral_density_and_lineshapes(at_gs, at_es)

    # Harmonic + anharmonic both default to base_gamma = 1.0 + 0.0 * T^2 = 1.0 meV
    assert len(seen_gammas) >= 3
    assert seen_gammas[0] == pytest.approx(1.0)
    assert seen_gammas[-1] == pytest.approx(1.0)
    assert res["pl_spectrum"] and 10.0 in res["pl_spectrum"]

    summary = json.loads((ws / "lineshape_summary.json").read_text(encoding="utf-8"))
    assert summary["zpl_energy_eV"] == pytest.approx(1.25)

    with open(ws / "optical_lineshapes.csv", "r", encoding="utf-8") as f:
        header = next(csv.reader(f))
    assert any("I_PL_" in col for col in header)
    assert "photon_energy_eV" in header


def test_anharmonic_t_equil_measured_from_nve_production(tmp_path):
    """Verify equilibrated_T_K reflects actual NVE production kinetic temperature rather than NVT target."""
    from ase import Atoms
    from ase.calculators.calculator import Calculator, all_changes
    try:
        from ase.md.velocitydistribution import thermalize_momenta as _init_vel
    except ImportError:
        from ase.md.velocitydistribution import MaxwellBoltzmannDistribution as _init_vel
    from hydeph.config import ToolkitConfig
    from hydeph.lineshape import LineshapeEngine

    class SimpleHarmonic(Calculator):
        implemented_properties = ["energy", "forces"]

        def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            self.results["energy"] = float(4.0 * np.sum(atoms.positions ** 2))
            self.results["forces"] = -8.0 * atoms.positions

    cfg = ToolkitConfig(
        pristine_cell_path=str(tmp_path),
        foundation_potential_path="dummy.txt",
        workspace_dir=str(tmp_path / "ws_nve"),
        temperatures=[10.0],
        enable_anharmonic=True,
        anharmonic_md_steps=16,
        anharmonic_thermalization_steps=0,
        anharmonic_n_trajectories=2,
    )
    engine = LineshapeEngine(cfg, SimpleHarmonic())

    # Initialize velocities corresponding to ~200 K while config target temperature is 10 K and n_therm=0
    at_gs = Atoms("C4", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], cell=[6, 6, 6], pbc=True)
    _init_vel(at_gs, temperature_K=200.0, rng=np.random.default_rng(7))
    at_es = at_gs.copy()
    at_es.positions[0, 0] += 0.05

    hw_grid = np.linspace(5.0, 100.0, 40)
    s_cont = np.exp(-0.5 * ((hw_grid - 40.0) / 10.0) ** 2) * 50.0
    res = engine.run_anharmonic_sampling(at_gs, at_es, hw_grid, s_cont, total_S=1.0)
    # Must reflect NVE kinetic energy (~50-150 K), not fallback target T=10.0 K
    assert res[10.0]["equilibrated_T_K"] > 25.0


def test_cli_lineshape_wiring(tmp_path, monkeypatch):
    """Verify cli_lineshape wires HybridDefectCalculator, PairedDefectCalculator, and LineshapeEngine cleanly."""
    import types
    import ase.io
    from unittest.mock import patch
    from ase import Atoms
    from hydeph.cli_commands import cli_lineshape

    ws = tmp_path / "ws_cli"
    ws.mkdir()
    uc = Atoms("C2", positions=[[0, 0, 0], [1.4, 1.4, 1.4]], cell=[2.8, 2.8, 2.8], pbc=True)
    ase.io.write(str(ws / "relaxed_gs.xyz"), uc)
    ase.io.write(str(ws / "relaxed_es.xyz"), uc)
    (ws / "nep_delta_gs.txt").write_text("dummy", encoding="utf-8")
    (ws / "nep_delta_es.txt").write_text("dummy", encoding="utf-8")
    pot = tmp_path / "nep_base.txt"
    pot.write_text("dummy", encoding="utf-8")
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        f"pristine_cell_path: {tmp_path / 'p.xyz'}\n"
        f"foundation_potential_path: {pot}\n"
        f"workspace_dir: {ws}\n"
        f"spin_multiplicity: 3\n",
        encoding="utf-8",
    )

    init_calls = []

    class DummyHybrid:
        def __init__(self, foundation_pot_path=None, delta_pot_path=None, calc_base=None, **kwargs):
            init_calls.append((foundation_pot_path, delta_pot_path, calc_base))
            self.calc_base = calc_base or object()

    class DummyPaired:
        def __init__(self, gs, es):
            self.gs = gs
            self.es = es

    class DummyEngine:
        def __init__(self, cfg, calc, calc_es=None):
            self.cfg = cfg
            self.calc = calc
            self.calc_es = calc_es
            self.paired_calc = None

        def calculate_spectral_density_and_lineshapes(self, at_gs, at_es):
            assert self.paired_calc is not None
            assert self.calc_es is not None
            return {"total_S": 1.0}

    import hydeph.calculator as calc_mod
    import hydeph.lineshape as ls_mod

    monkeypatch.setattr(calc_mod, "HybridDefectCalculator", DummyHybrid)
    monkeypatch.setattr(calc_mod, "PairedDefectCalculator", DummyPaired)
    monkeypatch.setattr(ls_mod, "LineshapeEngine", DummyEngine)

    with patch("hydeph.cli_commands.write_stage_manifest"), patch("hydeph.cli_commands.update_incremental_provenance"):
        rc = cli_lineshape(str(cfg_path), types.SimpleNamespace())
    assert rc == 0
    assert len(init_calls) == 2
    assert init_calls[0][1] == str(ws / "nep_delta_gs.txt")
    assert init_calls[1][1] == str(ws / "nep_delta_es.txt")
    assert init_calls[1][2] is not None


def test_cc_paired_potential_surface_zpl(tmp_path):
    """Verify CC derives ZPL from V_ES(R_ES) - V_GS(R_GS) when HybridDefectCalculator calc_es is provided."""
    from ase import Atoms
    from ase.calculators.calculator import Calculator, all_changes
    from hydeph.calculator import HybridDefectCalculator
    from hydeph.cc_diagram import ConfigurationCoordinateEngine
    from hydeph.config import ToolkitConfig

    class OffsetHarmonic(Calculator):
        implemented_properties = ["energy", "forces"]

        def __init__(self, shift=0.0, offset=0.0):
            super().__init__()
            self.shift = shift
            self.offset = offset

        def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            disp = atoms.positions - self.shift
            self.results["energy"] = float(4.0 * np.sum(disp ** 2) + self.offset)
            self.results["forces"] = -8.0 * disp

    at_gs = Atoms("C2", positions=[[0, 0, 0], [1.4, 0.0, 0.0]], cell=[5.0, 5.0, 5.0], pbc=True)
    at_es = at_gs.copy()
    at_es.positions[1, 0] += 0.1

    base = OffsetHarmonic(shift=at_gs.positions, offset=-10.0)
    h_gs = HybridDefectCalculator(
        calc_base=base,
        calc_delta=OffsetHarmonic(shift=at_gs.positions, offset=0.0),
        energy_offset_per_atom=0.0,
    )
    # Use small parabolic minimum aligned at at_es on calc_es so Delta E_rel,ES > 0.005 eV
    class EsShiftCalc(Calculator):
        implemented_properties = ["energy", "forces"]

        def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            d = atoms.positions - at_es.positions
            self.results["energy"] = float(4.0 * np.sum(d ** 2) + 1.8)
            self.results["forces"] = -8.0 * d

    h_es = HybridDefectCalculator(calc_base=base, calc_delta=EsShiftCalc(), energy_offset_per_atom=0.0)

    cfg = ToolkitConfig(
        pristine_cell_path=str(tmp_path),
        workspace_dir=str(tmp_path / "ws_cc_paired"),
        zpl_energy_eV=None,
        allow_fallbacks=False,
    )
    engine = ConfigurationCoordinateEngine(cfg, h_gs)
    with pytest.warns(UserWarning, match="Auto-derived ZPL energy"):
        res = engine.calculate_cc_curve(at_gs, at_es, n_points=5, calc_es=h_es)
    assert res["zpl_source"] == "paired_potential_surface"
    assert res["e_zpl_eV"] > 1.7
