"""Unit tests for automated spin-proxy excited-state relaxation in DefectBuilder and QERunner."""

import os
import shutil
import tempfile
import pytest
import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from hydeph.config import ToolkitConfig
from hydeph.qe_runner import write_qe_input, parse_qe_relaxation_output, QERunner
from hydeph.defect_builder import DefectBuilder


class DummyHarmonicCalc(Calculator):
    """Minimal dummy calculator for testing without external potential binaries."""
    implemented_properties = ["energy", "forces"]

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        pos = self.atoms.positions
        self.results["energy"] = float(0.5 * np.sum((pos - 1.0) ** 2))
        self.results["forces"] = -(pos - 1.0)


class MockRelaxationQERunner:
    """Mock QERunner that records calls and returns relaxed structures."""

    def __init__(self, config: ToolkitConfig, delta_d: float = 0.065):
        self.config = config
        self.delta_d = delta_d
        self.calls = []

    def run_relaxation(
        self,
        atoms: Atoms,
        calc_dir: str,
        tot_magnetization=None,
        nspin=None,
        timeout_s=None,
        force_rerun=False,
    ) -> Atoms:
        self.calls.append({
            "calc_dir": calc_dir,
            "tot_magnetization": tot_magnetization,
            "nspin": nspin,
            "timeout_s": timeout_s,
            "force_rerun": force_rerun,
        })
        os.makedirs(calc_dir, exist_ok=True)
        relaxed = atoms.copy()
        # If excited state relaxation (target_mag > 0), stretch first two atoms
        if tot_magnetization and tot_magnetization >= 2 and len(relaxed) >= 2:
            relaxed.positions[1, 0] += self.delta_d
        relaxed.info["energy"] = -100.0 - (tot_magnetization or 0)
        return relaxed


def test_toolkit_config_spin_proxy_fields():
    """Tests default configuration fields and spin multiplicity derivations."""
    cfg = ToolkitConfig(pristine_cell_path="test.xyz")
    assert cfg.excited_state_method == "spin_proxy"
    assert cfg.ground_state_relax_method == "auto"
    assert cfg.relax_forc_conv_thr == 1.0e-3
    assert cfg.relax_nstep == 100
    assert cfg.excited_state_tot_magnetization is None
    assert cfg.excited_state_spin_multiplicity is None

    # Derive spin multiplicity from tot_magnetization
    cfg2 = ToolkitConfig(pristine_cell_path="test.xyz", excited_state_tot_magnetization=2)
    assert cfg2.excited_state_spin_multiplicity == 3
    assert cfg2.excited_state_tot_magnetization == 2

    # Derive tot_magnetization from spin multiplicity
    cfg3 = ToolkitConfig(pristine_cell_path="test.xyz", excited_state_spin_multiplicity=4)
    assert cfg3.excited_state_tot_magnetization == 3
    assert cfg3.excited_state_spin_multiplicity == 4


def test_toolkit_config_validation_spin_proxy(tmp_path):
    """Tests validation of spin proxy and relaxation parameters."""
    dummy_pristine = tmp_path / "pristine.xyz"
    dummy_pristine.write_text("1\ncell\nC 0 0 0\n")
    dummy_nep = tmp_path / "nep.txt"
    dummy_nep.write_text("nep")

    cfg = ToolkitConfig(
        pristine_cell_path=str(dummy_pristine),
        foundation_potential_path=str(dummy_nep),
        excited_state_method="invalid",
    )
    with pytest.raises(ValueError, match="Invalid excited_state_method"):
        cfg.validate()

    cfg_gs = ToolkitConfig(
        pristine_cell_path=str(dummy_pristine),
        foundation_potential_path=str(dummy_nep),
        ground_state_relax_method="invalid",
    )
    with pytest.raises(ValueError, match="Invalid ground_state_relax_method"):
        cfg_gs.validate()

    cfg_thr = ToolkitConfig(
        pristine_cell_path=str(dummy_pristine),
        foundation_potential_path=str(dummy_nep),
        relax_forc_conv_thr=-1.0,
    )
    with pytest.raises(ValueError, match="Invalid relax_forc_conv_thr"):
        cfg_thr.validate()

    cfg_step = ToolkitConfig(
        pristine_cell_path=str(dummy_pristine),
        foundation_potential_path=str(dummy_nep),
        relax_nstep=0,
    )
    with pytest.raises(ValueError, match="Invalid relax_nstep"):
        cfg_step.validate()


def test_write_qe_input_relaxation_and_spin_override(tmp_path):
    """Tests write_qe_input for calculation='relax' with &IONS and tot_magnetization override."""
    atoms = Atoms("BN", positions=[[0, 0, 0], [1.45, 0, 0]], cell=[5, 5, 5], pbc=True)
    cfg = ToolkitConfig(
        pristine_cell_path="test.xyz",
        pseudopotentials={"B": "B.upf", "N": "N.upf"},
        spin_multiplicity=1,  # Ground state singlet (tot_magnetization = 0)
        relax_forc_conv_thr=2.0e-3,
        relax_nstep=80,
    )

    pw_in = str(tmp_path / "pw_relax.in")
    write_qe_input(
        atoms,
        pw_in,
        cfg,
        calculation="relax",
        tot_magnetization=2,  # Overridden triplet proxy
        nspin=2,
    )

    with open(pw_in, "r") as f:
        content = f.read()

    assert "calculation = 'relax'" in content
    assert "forc_conv_thr = 0.002" in content
    assert "nstep = 80" in content
    assert "nosym = .true." in content
    assert "tot_magnetization = 2" in content
    assert "nspin = 2" in content
    assert "&IONS" in content
    assert "ion_dynamics = 'bfgs'" in content


def test_parse_qe_relaxation_output_with_final_coordinates(tmp_path):
    """Tests parsing QE relaxation output with 'Begin final coordinates' block."""
    atoms = Atoms("BN", positions=[[0, 0, 0], [1.45, 0, 0]], cell=[5, 5, 5], pbc=True)
    mock_out = str(tmp_path / "pw_relax.out")

    sample_output = """
     End of BFGS Geometry Optimization

     Final energy             =    -105.12345678 Ry

Begin final coordinates

ATOMIC_POSITIONS (angstrom)
B                0.0100000000        0.0050000000        0.0020000000
N                1.5150000000        0.0020000000       -0.0010000000
End final coordinates

     JOB DONE.
"""
    with open(mock_out, "w") as f:
        f.write(sample_output)

    at_rel, e_ev = parse_qe_relaxation_output(mock_out, atoms)
    assert at_rel is not None
    assert len(at_rel) == 2
    assert e_ev is not None
    assert np.isclose(e_ev, -105.12345678 * 13.6056980659, atol=1e-3)
    assert np.isclose(at_rel.positions[0, 0], 0.010, atol=1e-4)
    assert np.isclose(at_rel.positions[1, 0], 1.515, atol=1e-4)


def test_parse_qe_relaxation_output_fallback_last_block(tmp_path):
    """Tests parsing QE relaxation output when 'Begin final coordinates' is missing."""
    atoms = Atoms("BN", positions=[[0, 0, 0], [1.45, 0, 0]], cell=[5, 5, 5], pbc=True)
    mock_out = str(tmp_path / "pw_relax_fallback.out")

    sample_output = """
ATOMIC_POSITIONS (angstrom)
B                0.0000000000        0.0000000000        0.0000000000
N                1.4500000000        0.0000000000        0.0000000000

!    total energy              =    -100.00000000 Ry

ATOMIC_POSITIONS (angstrom)
B                0.0050000000        0.0010000000        0.0000000000
N                1.5200000000       -0.0020000000        0.0010000000

!    total energy              =    -100.50000000 Ry
"""
    with open(mock_out, "w") as f:
        f.write(sample_output)

    at_rel, e_ev = parse_qe_relaxation_output(mock_out, atoms)
    assert at_rel is not None
    assert len(at_rel) == 2
    assert np.isclose(at_rel.positions[0, 0], 0.005, atol=1e-4)
    assert np.isclose(at_rel.positions[1, 0], 1.520, atol=1e-4)
    assert np.isclose(e_ev, -100.5 * 13.6056980659, atol=1e-3)


def test_defect_builder_automated_spin_proxy_singlet(tmp_path):
    """Tests DefectBuilder spin_proxy method auto-selecting triplet (M=2) for singlet ground state (M=0)."""
    pristine_path = str(tmp_path / "pristine.xyz")
    pristine = Atoms("B2N2", positions=[[0, 0, 0], [1.45, 0, 0], [0, 2.5, 0], [1.45, 2.5, 0]], cell=[5, 5, 10], pbc=True)
    pristine.write(pristine_path)

    ws = str(tmp_path / "run_singlet")
    cfg = ToolkitConfig(
        pristine_cell_path=pristine_path,
        defect_type="substitutional",
        defect_site_element="B->C",
        supercell_repeat=(1, 1, 1),
        workspace_dir=ws,
        spin_polarized=True,
        spin_multiplicity=1,  # Ground state singlet: tot_magnetization = 0
        excited_state_method="spin_proxy",
        ground_state_relax_method="auto",
    )

    mock_qe = MockRelaxationQERunner(cfg, delta_d=0.08)
    builder = DefectBuilder(cfg, calc_foundation=DummyHarmonicCalc(), calc_qe=mock_qe)

    _, defect_sc = builder.build_supercell_and_defect()
    at_gs = builder.relax_ground_state(defect_sc)
    at_es, dq_fc = builder.generate_excited_state_geometry(at_gs)

    # Verify Mock QE was invoked for both GS and ES
    assert len(mock_qe.calls) == 2
    gs_call = mock_qe.calls[0]
    es_call = mock_qe.calls[1]

    # GS should be relaxed with tot_magnetization = 0
    assert gs_call["tot_magnetization"] == 0
    # ES should automatically be relaxed with tot_magnetization = 0 + 2 = 2 (triplet proxy)
    assert es_call["tot_magnetization"] == 2

    # Check that relaxed_gs.xyz and relaxed_es.xyz were written
    assert os.path.isfile(os.path.join(ws, "relaxed_gs.xyz"))
    assert os.path.isfile(os.path.join(ws, "relaxed_es.xyz"))
    assert dq_fc > 0.0


def test_defect_builder_automated_spin_proxy_doublet(tmp_path):
    """Tests DefectBuilder spin_proxy method auto-selecting quartet (M=3) for doublet ground state (M=1)."""
    pristine_path = str(tmp_path / "pristine.xyz")
    pristine = Atoms("B2N2", positions=[[0, 0, 0], [1.45, 0, 0], [0, 2.5, 0], [1.45, 2.5, 0]], cell=[5, 5, 10], pbc=True)
    pristine.write(pristine_path)

    ws = str(tmp_path / "run_doublet")
    cfg = ToolkitConfig(
        pristine_cell_path=pristine_path,
        defect_type="vacancy",
        defect_site_element="B",
        supercell_repeat=(1, 1, 1),
        workspace_dir=ws,
        spin_polarized=True,
        spin_multiplicity=2,  # Ground state doublet: tot_magnetization = 1
        excited_state_method="spin_proxy",
        ground_state_relax_method="auto",
    )

    mock_qe = MockRelaxationQERunner(cfg, delta_d=0.05)
    builder = DefectBuilder(cfg, calc_foundation=DummyHarmonicCalc(), calc_qe=mock_qe)

    _, defect_sc = builder.build_supercell_and_defect()
    at_gs = builder.relax_ground_state(defect_sc)
    at_es, dq_fc = builder.generate_excited_state_geometry(at_gs)

    assert len(mock_qe.calls) == 2
    gs_call = mock_qe.calls[0]
    es_call = mock_qe.calls[1]

    # GS doublet: tot_magnetization = 1
    assert gs_call["tot_magnetization"] == 1
    # ES quartet proxy: tot_magnetization = 1 + 2 = 3
    assert es_call["tot_magnetization"] == 3
    assert dq_fc > 0.0


def test_defect_builder_spin_proxy_fallback_to_synthetic(tmp_path):
    """Tests DefectBuilder falling back to synthetic breathing mode if QE relaxation fails."""
    pristine_path = str(tmp_path / "pristine.xyz")
    pristine = Atoms("B2N2", positions=[[0, 0, 0], [1.45, 0, 0], [0, 2.5, 0], [1.45, 2.5, 0]], cell=[5, 5, 10], pbc=True)
    pristine.write(pristine_path)

    ws = str(tmp_path / "run_fallback")
    cfg = ToolkitConfig(
        pristine_cell_path=pristine_path,
        defect_type="substitutional",
        defect_site_element="B->C",
        supercell_repeat=(1, 1, 1),
        workspace_dir=ws,
        excited_state_method="spin_proxy",
        ground_state_relax_method="foundation",
    )

    class FailingQERunner:
        def run_relaxation(self, *args, **kwargs):
            raise RuntimeError("Simulated QE failure or missing binary")

    builder = DefectBuilder(cfg, calc_foundation=DummyHarmonicCalc(), calc_qe=FailingQERunner())

    _, defect_sc = builder.build_supercell_and_defect()
    at_gs = builder.relax_ground_state(defect_sc)
    # generate_excited_state_geometry should catch the error and fall back to synthetic breathing mode
    at_es, dq_fc = builder.generate_excited_state_geometry(at_gs)

    assert os.path.isfile(os.path.join(ws, "relaxed_es.xyz"))
    assert dq_fc > 0.0
