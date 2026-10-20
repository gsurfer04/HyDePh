"""Tests for HyDePh 0.5.0 upgraded code review, efficiency, and UX recommendations."""

import os
import sys
import json
import warnings
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from hydeph.config import ToolkitConfig
from hydeph.calculator import HybridDefectCalculator
from hydeph.defect_builder import DefectBuilder
from hydeph.frame_sampler import FrameSampler, farthest_point_sampling
from hydeph.qe_runner import write_qe_input, parse_qe_output, QERunner
from hydeph.delta_trainer import (
    DeltaTrainer,
    stratified_train_test_split,
    write_delta_nep_in,
    _compute_training_digest,
)
from hydeph.cc_diagram import ConfigurationCoordinateEngine, NV_ZPL_EV
from hydeph.lineshape import LineshapeEngine
from hydeph.pipeline import main as cli_main


class MockLinearCalc(Calculator):
    """Mock ASE calculator returning energy = sum(pos) and forces = -1."""
    implemented_properties = ["energy", "forces"]

    def __init__(self, energy_offset=0.0):
        super().__init__()
        self.energy_offset = energy_offset

    def calculate(self, atoms=None, properties=["energy", "forces"], system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        if "energy" in properties:
            self.results["energy"] = float(np.sum(atoms.positions)) + self.energy_offset
        if "forces" in properties:
            self.results["forces"] = np.ones_like(atoms.positions) * -0.5


def test_hybrid_defect_calculator_property_gating():
    """HybridDefectCalculator only calculates requested properties (energy-only vs forces)."""
    calc_base = MockLinearCalc(energy_offset=10.0)
    calc_delta = MockLinearCalc(energy_offset=2.5)

    hybrid = HybridDefectCalculator(calc_base=calc_base, calc_delta=calc_delta)
    at = Atoms("LiCl", positions=[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], cell=[5, 5, 5], pbc=True)

    # 1. Request only energy
    hybrid.calculate(at, properties=["energy"])
    assert "energy" in hybrid.results
    assert hybrid.results["energy"] == pytest.approx(float(np.sum(at.positions)) * 2 + 12.5)
    assert "forces" not in hybrid.results

    # 2. Request forces and energy
    hybrid.calculate(at, properties=["energy", "forces"])
    assert "energy" in hybrid.results
    assert "forces" in hybrid.results
    assert np.allclose(hybrid.results["forces"], -1.0)


def test_defect_builder_vectorized_embedding_matches_exact_positions(tmp_path):
    """Vectorized embed_defect_unitcell generates identical geometry to standard supercell expansion."""
    uc_pristine = Atoms("Si2", positions=[[0.0, 0.0, 0.0], [1.35, 1.35, 1.35]], cell=[5.43, 5.43, 5.43], pbc=True)
    uc_defect = uc_pristine[:-1].copy()  # 1 vacancy

    pristine_file = tmp_path / "pristine.xyz"
    ase_write = __import__("ase.io").io.write
    ase_write(str(pristine_file), uc_pristine)

    cfg = ToolkitConfig(
        pristine_cell_path=str(pristine_file),
        supercell_repeat=(3, 3, 3),
        workspace_dir=str(tmp_path / "ws_sc"),
    )
    builder = DefectBuilder(cfg)
    sc, pos = builder.embed_defect_unitcell(uc_pristine, uc_defect, rep=(3, 3, 3))

    # 27 cells * 2 atoms - 1 vacancy = 53 atoms
    assert len(sc) == 53
    assert np.allclose(sc.cell.diagonal(), [5.43 * 3, 5.43 * 3, 5.43 * 3])
    assert pos is not None


def test_fps_incremental_speedup():
    """farthest_point_sampling returns valid and unique points using incremental running min."""
    rng = np.random.RandomState(42)
    descriptors = rng.normal(size=(50, 8))
    selected = farthest_point_sampling(descriptors, n_select=10)

    assert len(selected) == 10
    assert len(set(selected)) == 10  # All selections are unique
    assert all(0 <= idx < 50 for idx in selected)


def test_delta_trainer_chemical_symbol_union(tmp_path):
    """DeltaTrainer.train_delta_nep takes the union of chemical elements across all frames."""
    fr1 = Atoms("LiCl", positions=[[0, 0, 0], [1, 1, 1]], cell=[5, 5, 5], pbc=True)
    fr1.info = {"sample_type": "franck_condon", "frame_id": "fc_00", "lambda": 0.0, "energy": 1.0}
    fr1.arrays["forces"] = np.zeros((2, 3))

    # Frame 2 has an additional dopant element 'F'
    fr2 = Atoms("LiClF", positions=[[0, 0, 0], [1, 1, 1], [2, 2, 2]], cell=[5, 5, 5], pbc=True)
    fr2.info = {"sample_type": "core_rattle", "frame_id": "rattle_00", "energy": 2.0}
    fr2.arrays["forces"] = np.zeros((3, 3))

    mock_nep = tmp_path / "mock_nep.py"
    mock_nep.write_text(
        f"#!{sys.executable}\n"
        "with open('nep.txt', 'w') as f:\n"
        "    f.write('mock\\n')\n"
    )

    cfg = ToolkitConfig(
        pristine_cell_path=str(tmp_path),
        workspace_dir=str(tmp_path / "ws_union"),
        nep_exe=str(mock_nep),
        nep_cutoff_radial=4.8,
        nep_cutoff_angular=3.8,
        nep_n_max_radial=7,
        nep_n_max_angular=5,
    )
    trainer = DeltaTrainer(cfg)
    pot = trainer.train_delta_nep([fr1, fr2])
    assert os.path.isfile(pot)

    nep_in_text = (tmp_path / "ws_union" / "delta_nep_training" / "nep.in").read_text()
    assert "type 3 Cl F Li" in nep_in_text
    assert "cutoff 4.8 3.8" in nep_in_text
    assert "n_max 7 5" in nep_in_text


def test_stratified_train_test_split_edge_cases():
    """Single frame emits a UserWarning; multiple frames partition correctly."""
    fr = Atoms("LiCl", positions=[[0, 0, 0], [1, 1, 1]], cell=[5, 5, 5], pbc=True)
    fr.info = {"sample_type": "franck_condon", "frame_id": "fc_00", "lambda": 0.0}

    with pytest.warns(UserWarning, match="only 1 frame"):
        train, test = stratified_train_test_split([fr], test_fraction=0.2)
        assert len(train) == 1
        assert len(test) == 1


def test_configuration_coordinate_zpl_derivation_and_warning(tmp_path):
    """ZPL is derived from GS/ES energies when available; warns when falling back to default."""
    at_gs = Atoms("C2", positions=[[0, 0, 0], [1.5, 0, 0]], cell=[5, 5, 5], pbc=True)
    at_es = at_gs.copy()
    at_es.positions[1, 0] += 0.2

    cfg = ToolkitConfig(
        pristine_cell_path=str(tmp_path),
        workspace_dir=str(tmp_path / "ws_cc"),
        zpl_energy_eV=None,
    )
    calc = MockLinearCalc()
    engine = ConfigurationCoordinateEngine(cfg, calc)

    # 1. Neither user nor atoms.info energy specified -> emits warning and uses canonical reference
    with pytest.warns(UserWarning, match="Falling back to reference canonical NV- ZPL"):
        res = engine.calculate_cc_curve(at_gs, at_es, n_points=5)
        assert res["e_zpl_eV"] == NV_ZPL_EV

    # 2. Energies present in atoms.info -> auto-derived
    at_gs.info["energy"] = -100.0
    at_es.info["energy"] = -97.85  # ZPL = 2.15 eV
    with pytest.warns(UserWarning, match="Auto-derived ZPL energy: 2.1500 eV"):
        res2 = engine.calculate_cc_curve(at_gs, at_es, n_points=5)
        assert pytest.approx(res2["e_zpl_eV"], rel=1e-4) == 2.15

    # 3. Explicit config override takes precedence without warning
    cfg_explicit = ToolkitConfig(
        pristine_cell_path=str(tmp_path),
        workspace_dir=str(tmp_path / "ws_cc"),
        zpl_energy_eV=3.45,
    )
    engine_explicit = ConfigurationCoordinateEngine(cfg_explicit, calc)
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        res3 = engine_explicit.calculate_cc_curve(at_gs, at_es, n_points=5)
        assert res3["e_zpl_eV"] == 3.45
        assert len(recorded) == 0


def test_export_numerical_results(tmp_path):
    """LineshapeEngine and ConfigurationCoordinateEngine export CSV and JSON alongside figures."""
    at_gs = Atoms("C2", positions=[[0, 0, 0], [1.5, 0, 0]], cell=[5, 5, 5], pbc=True)
    at_es = at_gs.copy()
    at_es.positions[1, 0] += 0.15

    out_dir = tmp_path / "export_test"
    out_dir.mkdir()
    cfg = ToolkitConfig(
        pristine_cell_path=str(tmp_path),
        workspace_dir=str(out_dir),
        zpl_energy_eV=2.0,
    )
    calc = MockLinearCalc()

    # CC export
    cc_engine = ConfigurationCoordinateEngine(cfg, calc)
    cc_data = cc_engine.calculate_cc_curve(at_gs, at_es, n_points=11)
    paths_cc = cc_engine.export_results(cc_data, output_dir=str(out_dir))
    assert os.path.isfile(paths_cc["json"])
    assert os.path.isfile(paths_cc["csv"])

    # Lineshape export
    ls_engine = LineshapeEngine(cfg, calc)
    fake_results = {
        "total_S": 1.25,
        "hw_grid": np.linspace(1, 50, 50),
        "s_cont": np.ones(50) * 10.0,
        "harmonic": {
            10.0: (np.linspace(-100, 20, 40), np.ones(40) * 0.05),
            300.0: (np.linspace(-100, 20, 40), np.ones(40) * 0.02),
        },
    }
    paths_ls = ls_engine.export_results(fake_results, output_dir=str(out_dir))
    assert os.path.isfile(paths_ls["summary"])
    assert os.path.isfile(paths_ls["spectral_density"])
    assert os.path.isfile(paths_ls["lineshapes"])


def test_toolkit_config_repr_and_file_io(tmp_path):
    """ToolkitConfig.__repr__ displays only modified fields and loads YAML/JSON."""
    cfg = ToolkitConfig(
        pristine_cell_path="dummy.xyz",
        temperatures=[77.0],
        spin_multiplicity=3,
    )
    r = repr(cfg)
    assert "temperatures=[77.0]" in r
    assert "spin_multiplicity=3" in r
    assert "random_seed" not in r  # default field omitted

    # Test dictionary round-trip
    d = cfg.to_dict()
    assert d["spin_multiplicity"] == 3
    reloaded = ToolkitConfig.from_dict(d)
    assert reloaded.spin_multiplicity == 3
    assert reloaded.temperatures == [77.0]

    # Test JSON file load
    json_file = tmp_path / "cfg.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(d, f)
    cfg_loaded = ToolkitConfig.from_file(str(json_file))
    assert cfg_loaded.spin_multiplicity == 3


def test_qe_gamma_kpoints_and_universal_masses(tmp_path):
    """write_qe_input writes 'K_POINTS gamma' for (1,1,1) and universal ASE masses."""
    # Test unusual elements like Arsenic (As, mass ~ 74.9216) and Gallium (Ga, mass ~ 69.723)
    at = Atoms("GaAs", positions=[[0, 0, 0], [1.4, 1.4, 1.4]], cell=[5.65, 5.65, 5.65], pbc=True)
    pw_in = tmp_path / "pw.in"
    cfg = ToolkitConfig(
        pristine_cell_path=str(tmp_path),
        kpoints=(1, 1, 1),
        pseudopotentials={"Ga": "Ga.UPF", "As": "As.UPF"},
    )
    write_qe_input(at, str(pw_in), cfg)
    text = pw_in.read_text()

    assert "K_POINTS gamma" in text
    assert "Ga   69.7230" in text or "Ga    69.723" in text
    assert "As   74.9216" in text or "As    74.922" in text


def test_single_pass_parse_qe_output(tmp_path):
    """parse_qe_output parses energy and forces in a single pass and returns None on non-convergence."""
    pw_out = tmp_path / "pw.out"
    pw_out.write_text("""
!    total energy              =     -52.1234 Ry
     Forces acting on atoms (cartesian axes, Ry/au):
     atom    1 type 1   force =     0.0100    0.0200    0.0300
     total force = 0.05
     JOB DONE.
""")
    e, f = parse_qe_output(str(pw_out))
    assert e is not None
    assert f is not None
    assert f.shape == (1, 3)

    # If convergence NOT achieved occurs, returns None, None
    pw_unconv = tmp_path / "pw_unconv.out"
    pw_unconv.write_text("""
!    total energy              =     -52.1234 Ry
     Forces acting on atoms (cartesian axes, Ry/au):
     atom    1 type 1   force =     0.0100    0.0200    0.0300
     convergence NOT achieved
""")
    e_un, f_un = parse_qe_output(str(pw_unconv))
    assert e_un is None
    assert f_un is None


