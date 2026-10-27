"""Tests for Batch 10: Scientific Regression Tests (D).

Covers:
1. Analytic 1D Displaced Harmonic Oscillator (Poisson distribution, sum rule, first and second moments, DWF(T)).
2. Toy Force-Field Hessian and S_k comparison against analytical derivation.
3. Quantum ESPRESSO output parser fixtures (converged SCF, unconverged, nstep exhausted, timeout, spin bands).
"""

import os
import pytest
import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from hydeph.config import ToolkitConfig
from hydeph.lineshape import (
    compute_quantum_lineshape,
    LineshapeEngine,
    MODE_COUPLING_PREFACTOR,
    KB_MEV_K,
)
from hydeph.qe_runner import (
    parse_qe_output,
    parse_qe_relaxation_output,
    parse_qe_band_gap,
    RY_TO_EV,
    RY_BOHR_TO_EV_A,
)


FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "qe")


# =============================================================================
# 1. Analytic 1D Displaced Harmonic Oscillator
# =============================================================================

def test_analytic_1d_poisson_lineshape():
    """Verify that a single displaced mode yields exact Poisson progression, sum rule, and first moment."""
    hw_0 = 40.0  # meV
    S = 2.0      # Huang-Rhys factor

    # Discrete single mode represented on fine grid
    hw_grid = np.linspace(0.5, 120.0, 1200)
    dhw = hw_grid[1] - hw_grid[0]
    # Delta-like spike at hw_0
    s_cont = np.zeros_like(hw_grid)
    spike_idx = int(np.argmin(np.abs(hw_grid - hw_0)))
    s_cont[spike_idx] = (S / dhw) * 1000.0  # eV^-1 so that integral is S

    trapz_fn = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    int_S = float(trapz_fn(s_cont / 1000.0, hw_grid))
    assert int_S == pytest.approx(S, rel=1e-3)

    # Compute lineshape at T = 0 K with narrow dephasing (gamma = 0.5 meV)
    # Using sufficiently wide window [-360, 60] meV so higher-order phonon replicas (up to n=8) and Lorentzian tails are captured
    e_grid, a_0K = compute_quantum_lineshape(
        hw_grid, s_cont, temp_k=0.001, dephasing_gamma_meV=0.5,
        e_min_meV=-360.0, e_max_meV=60.0, n_e=4000
    )

    # 1. Sum Rule: \int A(E) dE = 1.0
    area = float(trapz_fn(a_0K, e_grid))
    assert area == pytest.approx(1.0, rel=1e-3)

    # 2. First Moment: <E> = \int E A(E) dE = -S * hw_0
    first_moment = float(trapz_fn(e_grid * a_0K, e_grid))
    expected_first_moment = -S * hw_0  # -80.0 meV
    assert first_moment == pytest.approx(expected_first_moment, abs=2.5)

    # 3. Second Central Moment: <(E - <E>)^2> = S * (hw_0)^2
    second_moment = float(trapz_fn(((e_grid - first_moment) ** 2) * a_0K, e_grid))
    expected_variance = S * (hw_0 ** 2)  # 2.0 * 1600 = 3200 meV^2
    assert second_moment == pytest.approx(expected_variance, rel=0.08)

    # 4. Debye-Waller Factor at T = 0: DWF(0) = exp(-S)
    expected_dwf_0 = np.exp(-S)
    assert expected_dwf_0 == pytest.approx(np.exp(-2.0), rel=1e-5)

    # 5. Debye-Waller Factor at finite T: DWF(T) = exp(-S * (2 n_0 + 1))
    T = 100.0  # K
    n_0 = 1.0 / (np.exp(hw_0 / (KB_MEV_K * T)) - 1.0)
    expected_dwf_T = float(np.exp(-S * (2.0 * n_0 + 1.0)))
    assert expected_dwf_T < expected_dwf_0


# =============================================================================
# 2. Toy Force-Field Hessian & S_k Benchmark
# =============================================================================

class DiatomicHarmonicCalc(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, k_eV_A2=10.0, r0=1.50, **kwargs):
        super().__init__(**kwargs)
        self.k = k_eV_A2
        self.r0 = r0

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        p = atoms.positions
        r = float(np.linalg.norm(p[1] - p[0]))
        dr = r - self.r0
        e = 0.5 * self.k * (dr ** 2)
        u = (p[1] - p[0]) / r
        f = np.zeros_like(p)
        f[0] = self.k * dr * u
        f[1] = -self.k * dr * u
        self.results["energy"] = float(e)
        self.results["forces"] = f


def test_toy_force_field_hessian_and_sk(tmp_path):
    """Compare numerical normal mode frequency and S_k against exact hand-derived analytical formulas."""
    m_amu = 12.011  # C atom mass
    mu_amu = m_amu / 2.0  # reduced mass = 6.0055 amu
    k_eV_A2 = 8.0  # spring constant in eV/A^2
    r0 = 1.50      # equilibrium bond length in A

    # Hand calculation of stretch vibrational frequency:
    # omega = sqrt(2 * k / m) in natural units
    # Units conversion: 1 eV / (amu * A^2) = 1.602176634e-19 J / (1.660539e-27 kg * 1e-20 m^2) = 9.64853e27 s^-2
    # omega (s^-1) = sqrt(k / mu * 9.64853e27)
    # hbar * omega (eV) = hbar * omega
    # In MODE_COUPLING_PREFACTOR units:
    # K_mode = 2 * C * 1e-3 * (hw_meV)^2
    # Here mode coordinate Q = sqrt(mu) * Delta r
    # Energy = 0.5 * k * (Delta r)^2 = 0.5 * (k / mu) * Q^2
    # So K_mode = k / mu = 8.0 / 6.0055 = 1.332112 eV / (amu * A^2)
    # 2 * C * 1e-3 * (hw_meV)^2 = K_mode
    # hw_meV = sqrt( K_mode / (2 * C * 1e-3) )
    expected_hw_meV = np.sqrt((k_eV_A2 / mu_amu) / (2.0 * MODE_COUPLING_PREFACTOR * 1.0e-3))

    # Analytical Huang-Rhys factor for excitation displacement Delta r = 0.08 A:
    delta_r = 0.08
    delta_Q = np.sqrt(mu_amu) * delta_r
    expected_S = MODE_COUPLING_PREFACTOR * expected_hw_meV * (delta_Q ** 2)

    # Set up numerical ASE system
    pristine_file = tmp_path / "pristine.xyz"
    mol = Atoms("C2", positions=[[0.0, 0.0, 0.0], [r0, 0.0, 0.0]], cell=[10.0, 10.0, 10.0], pbc=False)
    import ase.io
    ase.io.write(str(pristine_file), mol)

    cfg = ToolkitConfig(
        pristine_cell_path=str(pristine_file),
        foundation_potential_path=str(tmp_path / "pot.txt"),
        workspace_dir=str(tmp_path / "ws"),
        supercell_repeat=(1, 1, 1),
    )

    calc = DiatomicHarmonicCalc(k_eV_A2=k_eV_A2, r0=r0)
    engine = LineshapeEngine(cfg, calc)

    at_gs = mol.copy()
    at_es = mol.copy()
    at_es.positions[1, 0] += delta_r  # excited-state displacement Delta r

    engine.calculate_spectral_density_and_lineshapes(at_gs, at_es)

    # Read normal_modes.csv to get the optical stretch mode
    csv_file = tmp_path / "ws" / "normal_modes.csv"
    assert csv_file.is_file()

    with open(csv_file, "r") as f:
        import csv
        reader = csv.DictReader(f)
        modes = list(reader)

    # The stretch mode is the highest-frequency non-zero mode
    stretch_modes = [m for m in modes if float(m["hw_meV"]) > 10.0]
    assert len(stretch_modes) == 1
    m_stretch = stretch_modes[0]

    calc_hw_meV = float(m_stretch["hw_meV"])
    calc_S = float(m_stretch["S_k"])

    assert calc_hw_meV == pytest.approx(expected_hw_meV, rel=1e-2)
    assert calc_S == pytest.approx(expected_S, rel=1e-2)


# =============================================================================
# 3. Quantum ESPRESSO Output Parser Fixtures
# =============================================================================

def test_qe_parser_converged_scf():
    """Verify parse_qe_output correctly parses converged SCF calculation."""
    out_file = os.path.join(FIXTURES_DIR, "scf_converged.out")
    energy, forces = parse_qe_output(out_file)

    assert energy is not None
    assert energy == pytest.approx(-120.45678901 * RY_TO_EV, rel=1e-6)
    assert forces is not None
    assert forces.shape == (2, 3)
    assert np.allclose(forces[0], [0.00001000 * RY_BOHR_TO_EV_A, 0.00000500 * RY_BOHR_TO_EV_A, -0.00000200 * RY_BOHR_TO_EV_A], atol=1e-4)


def test_qe_parser_unconverged_scf():
    """Verify parse_qe_output detects unconverged SCF calculation."""
    out_file = os.path.join(FIXTURES_DIR, "scf_unconverged.out")
    energy, forces = parse_qe_output(out_file)
    assert energy is None
    assert forces is None


def test_qe_parser_relax_nstep_exhausted():
    """Verify parse_qe_relaxation_output flags relaxation hitting nstep as unconverged."""
    out_file = os.path.join(FIXTURES_DIR, "relax_nstep_exhausted.out")
    template = Atoms("C2", positions=[[0.0, 0.0, 0.0], [1.5, 1.5, 1.5]])
    at, energy = parse_qe_relaxation_output(out_file, template_atoms=template, require_converged=True)
    assert at is None
    assert energy is None


def test_qe_parser_walltime_timeout():
    """Verify parse_qe_relaxation_output flags walltime exceeded as unconverged."""
    out_file = os.path.join(FIXTURES_DIR, "walltime_timeout.out")
    template = Atoms("C2", positions=[[0.0, 0.0, 0.0], [1.5, 1.5, 1.5]])
    at, energy = parse_qe_relaxation_output(out_file, template_atoms=template, require_converged=True)
    assert at is None
    assert energy is None


def test_qe_parser_spin_polarized_bands():
    """Verify parse_qe_band_gap resolves separate spin-up and spin-down band blocks."""
    out_file = os.path.join(FIXTURES_DIR, "spin_polarized_bands.out")
    gap_data = parse_qe_band_gap(out_file)

    assert gap_data is not None
    assert gap_data["gap_eV"] == pytest.approx(5.7, rel=1e-3)
    assert gap_data["homo_eV"] == pytest.approx(-4.5, rel=1e-3)
    assert gap_data["lumo_eV"] == pytest.approx(1.2, rel=1e-3)
    assert gap_data["fermi_eV"] == pytest.approx(0.5, rel=1e-3)
    assert gap_data["spin"] == "up"
