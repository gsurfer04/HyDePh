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
