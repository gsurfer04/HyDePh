"""Tests for phonon supercell embedding conservation and force vs displacement agreement."""

import pytest
import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from hydeph.config import ToolkitConfig
from hydeph.lineshape import embed_phonon_supercell, LineshapeEngine


class ToyHarmonic1DCalc(Calculator):
    """Harmonic spring calculator for an atom shifted along x from equilibrium x0."""
    implemented_properties = ["energy", "forces"]

    def __init__(self, k: float = 10.0, x0: float = 1.4, e0: float = 0.0):
        super().__init__()
        self.k = k
        self.x0 = x0
        self.e0 = e0

    def calculate(self, atoms=None, properties=["energy", "forces"], system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        forces = np.zeros((len(atoms), 3))
        if len(atoms) >= 2:
            x = atoms.positions[1, 0] - atoms.positions[0, 0]
            dx = x - self.x0
            fx = -self.k * dx
            forces[0, 0] = -fx
            forces[1, 0] = fx
            energy = self.e0 + 0.5 * self.k * (dx ** 2)
        else:
            energy = self.e0
        self.results["energy"] = energy
        self.results["forces"] = forces


def test_force_vs_displacement_embedding_harmonic_agreement(tmp_path):
    """In the purely harmonic limit, displacement and force embedding yield identical dQ and S."""
    ws = tmp_path / "ws_embed"
    ws.mkdir()
    pristine_file = tmp_path / "pristine.xyz"
    pristine = Atoms("C2", positions=[[0, 0, 0], [1.5, 0, 0]], cell=[6, 6, 6], pbc=True)
    pristine.write(str(pristine_file))

    dx = 0.08
    at_gs = pristine.copy()
    at_es = pristine.copy()
    at_es.positions[1, 0] += dx

    at_gs.info["energy"] = -10.0
    at_es.info["energy"] = -8.0
    at_gs.info["dft_gap_eV"] = 2.0
    at_es.info["dft_gap_eV"] = 2.0

    k_spring = 20.0
    calc_gs = ToyHarmonic1DCalc(k=k_spring, x0=1.5, e0=-10.0)
    calc_es = ToyHarmonic1DCalc(k=k_spring, x0=1.5 + dx, e0=-8.0)

    # 1. Displacement embedding
    cfg_disp = ToolkitConfig(
        pristine_cell_path=str(pristine_file),
        foundation_potential_path="dummy.txt",
        workspace_dir=str(ws / "disp"),
        phonon_embedding_method="displacement",
        supercell_repeat=(1, 1, 1),
        phonon_supercell_repeat=(1, 1, 1),
        finite_size_acoustic_cutoff=False,
        acoustic_sum_rule=False,
    )
    engine_disp = LineshapeEngine(cfg_disp, calc_gs, calc_es=calc_es)
    res_disp = engine_disp.calculate_spectral_density_and_lineshapes(at_gs, at_es)

    # 2. Force embedding
    cfg_force = ToolkitConfig(
        pristine_cell_path=str(pristine_file),
        foundation_potential_path="dummy.txt",
        workspace_dir=str(ws / "force"),
        phonon_embedding_method="force",
        supercell_repeat=(1, 1, 1),
        phonon_supercell_repeat=(1, 1, 1),
        finite_size_acoustic_cutoff=False,
        acoustic_sum_rule=False,
    )
    engine_force = LineshapeEngine(cfg_force, calc_gs, calc_es=calc_es)
    res_force = engine_force.calculate_spectral_density_and_lineshapes(at_gs, at_es)

    # In harmonic limit, displacement and force embedding must agree
    assert res_disp["delta_q"] == pytest.approx(res_force["delta_q"], rel=1e-2)
    assert res_disp["total_S_discrete"] == pytest.approx(res_force["total_S_discrete"], rel=1e-2)
    assert res_disp["total_S_discrete"] > 0.0


class ZeroForceCalc(Calculator):
    """Toy calculator with zero forces to keep structures at target geometries."""
    implemented_properties = ["energy", "forces"]

    def calculate(self, atoms=None, properties=["energy", "forces"], system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results["energy"] = -10.0 * len(atoms)
        self.results["forces"] = np.zeros((len(atoms), 3))


def test_embedding_supercell_delta_q_conservation(tmp_path):
    """Embedding defect displacement into a larger pristine supercell conserves total delta_Q."""
    pristine_file = tmp_path / "pristine.xyz"
    pristine = Atoms("C2", positions=[[0, 0, 0], [1.4, 0, 0]], cell=[5, 5, 5], pbc=True)
    pristine.write(str(pristine_file))

    # Single-atom vacancy cell: atom at [1.4, 0, 0] remaining, vacancy at [0, 0, 0]
    at_gs = Atoms("C", positions=[[1.4, 0, 0]], cell=[5, 5, 5], pbc=True)
    at_es = Atoms("C", positions=[[1.4 + 0.06, 0, 0]], cell=[5, 5, 5], pbc=True)

    m = at_gs.get_masses()
    dr_orig = at_es.positions - at_gs.positions
    dq_orig = float(np.sqrt(np.sum(m[:, None] * (dr_orig ** 2))))

    cfg = ToolkitConfig(
        pristine_cell_path=str(pristine_file),
        foundation_potential_path="dummy.txt",
        workspace_dir=str(tmp_path),
        supercell_repeat=(1, 1, 1),
        phonon_supercell_repeat=(2, 2, 2),
        defect_type="vacancy",
        defect_site_index=0,
    )

    calc = ZeroForceCalc()
    at_large, dr_large, _ = embed_phonon_supercell(at_gs, at_es, cfg, calc)

    m_large = at_large.get_masses()
    dq_embedded = float(np.sqrt(np.sum(m_large[:, None] * (dr_large ** 2))))

    # Total delta Q must be strictly conserved within 0.1%
    assert dq_embedded == pytest.approx(dq_orig, rel=1e-3)
