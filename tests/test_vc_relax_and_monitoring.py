"""Unit tests for Quantum ESPRESSO vc-relax (variable-cell relaxation) and BFGS monitoring."""

import os
import sys
import numpy as np
import pytest
from ase import Atoms

from hydeph.config import ToolkitConfig
from hydeph.qe_runner import (
    write_qe_input,
    parse_qe_relaxation_output,
    run_monitored_qe_subprocess,
    BOHR_TO_ANGSTROM,
    RY_TO_EV,
)


@pytest.fixture
def dummy_pseudo(tmp_path):
    """Creates mock UPF pseudopotential files."""
    p_dir = tmp_path / "pseudo"
    p_dir.mkdir()
    for s in ("Cs", "I", "C", "N"):
        upf = p_dir / f"{s}.upf"
        upf.write_text(f"<UPF version=\"2.0.1\">\n<PP_HEADER\nz_valence=\"{1.0 if s in ('Cs', 'I') else 4.0}\"\n/>\n</UPF>\n")
    return str(p_dir)


def test_config_vc_relax_options_and_validation(tmp_path, dummy_pseudo):
    """Verifies ToolkitConfig supports relaxation_calculation and cell options with strict validation."""
    cell_file = tmp_path / "cell.xyz"
    Atoms("CsI", cell=[4.6, 4.6, 4.6], pbc=True).write(str(cell_file))
    nep_file = tmp_path / "nep89.txt"
    nep_file.write_text("dummy nep model")

    # 1. Defaults
    cfg = ToolkitConfig(
        pristine_cell_path=str(cell_file),
        foundation_potential_path=str(nep_file),
        pseudo_dir=dummy_pseudo,
        pseudopotentials={"Cs": "Cs.upf", "I": "I.upf"},
    )
    assert cfg.relaxation_calculation == "relax"
    assert cfg.cell_dynamics == "bfgs"
    assert cfg.cell_dofree == "all"
    assert cfg.press_kbar == 0.0
    assert cfg.cell_factor == 2.0

    # 2. Configured vc-relax
    cfg_vc = ToolkitConfig(
        pristine_cell_path=str(cell_file),
        foundation_potential_path=str(nep_file),
        pseudo_dir=dummy_pseudo,
        pseudopotentials={"Cs": "Cs.upf", "I": "I.upf"},
        relaxation_calculation="vc-relax",
        cell_dynamics="bfgs",
        cell_dofree="shape",
        press_kbar=10.0,
        cell_factor=2.5,
    )
    cfg_vc.validate()
    assert cfg_vc.relaxation_calculation == "vc-relax"
    assert cfg_vc.cell_dofree == "shape"
    assert cfg_vc.press_kbar == 10.0

    # 3. Validation failures
    with pytest.raises(ValueError, match="Invalid relaxation_calculation"):
        cfg_bad = ToolkitConfig(
            pristine_cell_path=str(cell_file),
            foundation_potential_path=str(nep_file),
            pseudo_dir=dummy_pseudo,
            pseudopotentials={"Cs": "Cs.upf", "I": "I.upf"},
            relaxation_calculation="md",
        )
        cfg_bad.validate()

    with pytest.raises(ValueError, match="Invalid cell_dynamics"):
        cfg_bad_dyn = ToolkitConfig(
            pristine_cell_path=str(cell_file),
            foundation_potential_path=str(nep_file),
            pseudo_dir=dummy_pseudo,
            pseudopotentials={"Cs": "Cs.upf", "I": "I.upf"},
            cell_dynamics="invalid_dynamics",
        )
        cfg_bad_dyn.validate()

    with pytest.raises(ValueError, match="Invalid cell_dofree"):
        cfg_bad_dof = ToolkitConfig(
            pristine_cell_path=str(cell_file),
            foundation_potential_path=str(nep_file),
            pseudo_dir=dummy_pseudo,
            pseudopotentials={"Cs": "Cs.upf", "I": "I.upf"},
            cell_dofree="unsupported_axis",
        )
        cfg_bad_dof.validate()

    # 4. Aliases in from_dict
    data = {
        "pristine_cell_path": str(cell_file),
        "foundation_potential_path": str(nep_file),
        "pseudo_dir": dummy_pseudo,
        "pseudopotentials": {"Cs": "Cs.upf", "I": "I.upf"},
        "relaxation_mode": "vc-relax",
        "press": 5.0,
    }
    cfg_from_dict = ToolkitConfig.from_dict(data)
    assert cfg_from_dict.relaxation_calculation == "vc-relax"
    assert cfg_from_dict.press_kbar == 5.0


def test_write_qe_input_vc_relax_namelist(tmp_path, dummy_pseudo):
    """Verifies write_qe_input produces calculation='vc-relax', tstress=.true., &IONS, and &CELL namelists."""
    cell_file = tmp_path / "cell.xyz"
    atoms = Atoms("CsI", positions=[[0, 0, 0], [2.3, 2.3, 2.3]], cell=[4.6, 4.6, 4.6], pbc=True)
    atoms.write(str(cell_file))

    cfg = ToolkitConfig(
        pristine_cell_path=str(cell_file),
        pseudo_dir=dummy_pseudo,
        pseudopotentials={"Cs": "Cs.upf", "I": "I.upf"},
        spin_polarized=False,
        tot_magnetization=0,
        relaxation_calculation="vc-relax",
        cell_dynamics="bfgs",
        cell_dofree="volume",
        press_kbar=2.5,
        cell_factor=2.2,
    )

    pw_in = str(tmp_path / "pw.in")
    write_qe_input(atoms, pw_in, cfg, calculation="vc-relax", tot_magnetization=0, nspin=1)

    with open(pw_in, "r") as f:
        content = f.read()

    assert "calculation = 'vc-relax'" in content
    assert "tstress = .true." in content
    assert "&IONS" in content
    assert "ion_dynamics = 'bfgs'" in content
    assert "&CELL" in content
    assert "cell_dynamics = 'bfgs'" in content
    assert "press = 2.5000" in content
    assert "cell_dofree = 'volume'" in content
    assert "cell_factor = 2.20" in content


def test_parse_qe_relaxation_output_vc_relax_angstrom(tmp_path):
    """Verifies parse_qe_relaxation_output extracts CELL_PARAMETERS (angstrom) and updates Atoms cell."""
    template = Atoms("CsI", positions=[[0, 0, 0], [2.3, 2.3, 2.3]], cell=[4.6, 4.6, 4.6], pbc=True)
    out_file = str(tmp_path / "pw_vc.out")

    sample_output = """
     Program PWSCF v.7.2

     lattice parameter (alat)  =       8.7600  a.u.
     End of BFGS Geometry Optimization
     Final energy             =    -120.50000000 Ry

Begin final coordinates

CELL_PARAMETERS (angstrom)
   4.750000000   0.000000000   0.000000000
   0.000000000   4.750000000   0.000000000
   0.000000000   0.000000000   4.750000000

ATOMIC_POSITIONS (angstrom)
Cs   0.010000000   0.010000000   0.010000000
I    2.380000000   2.380000000   2.380000000

End final coordinates
"""
    with open(out_file, "w") as f:
        f.write(sample_output)

    at_rel, e_ev = parse_qe_relaxation_output(out_file, template, require_converged=True)
    assert at_rel is not None
    assert np.allclose(np.diag(at_rel.cell), [4.75, 4.75, 4.75], atol=1e-5)
    assert np.allclose(at_rel.positions[1], [2.38, 2.38, 2.38], atol=1e-5)
    assert np.isclose(e_ev, -120.5 * RY_TO_EV, atol=1e-3)


def test_parse_qe_relaxation_output_vc_relax_crystal_alat(tmp_path):
    """Verifies parse_qe_relaxation_output converts ATOMIC_POSITIONS (crystal) with CELL_PARAMETERS (alat=...)."""
    template = Atoms("CsI", positions=[[0, 0, 0], [2.3, 2.3, 2.3]], cell=[4.6, 4.6, 4.6], pbc=True)
    out_file = str(tmp_path / "pw_vc_alat.out")

    sample_output = """
     lattice parameter (alat)  =       8.7600  a.u.
     bfgs converged in   5 scf cycles and   4 bfgs steps
     End of BFGS Geometry Optimization
     Final energy             =    -125.00000000 Ry

Begin final coordinates

CELL_PARAMETERS (alat=  9.00000000)
   1.000000000   0.000000000   0.000000000
   0.000000000   1.000000000   0.000000000
   0.000000000   0.000000000   1.000000000

ATOMIC_POSITIONS (crystal)
Cs   0.000000000   0.000000000   0.000000000
I    0.500000000   0.500000000   0.500000000

End final coordinates
"""
    with open(out_file, "w") as f:
        f.write(sample_output)

    at_rel, e_ev = parse_qe_relaxation_output(out_file, template, require_converged=True)
    assert at_rel is not None
    expected_alat = 9.0 * BOHR_TO_ANGSTROM
    assert np.allclose(np.diag(at_rel.cell), [expected_alat, expected_alat, expected_alat], atol=1e-5)
    assert np.allclose(at_rel.positions[1], [expected_alat * 0.5, expected_alat * 0.5, expected_alat * 0.5], atol=1e-5)
    assert np.isclose(e_ev, -125.0 * RY_TO_EV, atol=1e-3)


def test_parse_qe_relaxation_output_vc_relax_fallback_pass2(tmp_path):
    """Verifies Pass 2 fallback correctly recovers CELL_PARAMETERS and positions when Begin final coordinates is absent."""
    template = Atoms("CsI", positions=[[0, 0, 0], [2.3, 2.3, 2.3]], cell=[4.6, 4.6, 4.6], pbc=True)
    out_file = str(tmp_path / "pw_vc_fallback.out")

    sample_output = """
     lattice parameter (alat)  =       8.7600  a.u.

CELL_PARAMETERS (angstrom)
   4.700000000   0.000000000   0.000000000
   0.000000000   4.700000000   0.000000000
   0.000000000   0.000000000   4.700000000

ATOMIC_POSITIONS (angstrom)
Cs   0.000000000   0.000000000   0.000000000
I    2.350000000   2.350000000   2.350000000

!    total energy              =    -122.00000000 Ry

CELL_PARAMETERS (angstrom)
   4.720000000   0.000000000   0.000000000
   0.000000000   4.720000000   0.000000000
   0.000000000   0.000000000   4.720000000

ATOMIC_POSITIONS (angstrom)
Cs   0.005000000   0.005000000   0.005000000
I    2.360000000   2.360000000   2.360000000

!    total energy              =    -122.50000000 Ry

     bfgs converged in   6 scf cycles and   2 bfgs steps
"""
    with open(out_file, "w") as f:
        f.write(sample_output)

    at_rel, e_ev = parse_qe_relaxation_output(out_file, template, require_converged=True)
    assert at_rel is not None
    assert np.allclose(np.diag(at_rel.cell), [4.72, 4.72, 4.72], atol=1e-5)
    assert np.allclose(at_rel.positions[1], [2.36, 2.36, 2.36], atol=1e-5)
    assert np.isclose(e_ev, -122.5 * RY_TO_EV, atol=1e-3)


def test_monitored_subprocess_bfgs_realtime_logging(tmp_path, monkeypatch, caplog):
    """Verifies run_monitored_qe_subprocess logs SCF iterations, SCF convergence, and BFGS step finishes."""
    import logging
    calc_dir = str(tmp_path / "calc")
    os.makedirs(calc_dir, exist_ok=True)
    pw_out = os.path.join(calc_dir, "pw.out")

    # Command script that writes mock relaxation output to stdout
    mock_script = tmp_path / "mock_qe.py"
    mock_script.write_text("""
import sys
sys.stdout.write("     iteration #  1     ecut=    40.00 Ry\\n")
sys.stdout.write("     iteration #  2     ecut=    40.00 Ry\\n")
sys.stdout.write("     convergence has been achieved in   2 iterations\\n")
sys.stdout.write("     Total force =     0.050000     Total SCF correction =     0.001000\\n")
sys.stdout.write("     Energy error            =      2.5E-02 Ry\\n")
sys.stdout.write("     Gradient error          =      1.8E-02 Ry/Bohr\\n")
sys.stdout.write("     number of scf cycles    =   1\\n")
sys.stdout.write("     number of bfgs steps    =   0\\n")
sys.stdout.write("     energy             new  =   -2449.81500000 Ry\\n")
sys.stdout.write("     iteration #  1     ecut=    40.00 Ry\\n")
sys.stdout.write("     convergence has been achieved in   1 iterations\\n")
sys.stdout.write("     Total force =     0.005000     Total SCF correction =     0.000100\\n")
sys.stdout.write("     Energy error            =      1.2E-04 Ry\\n")
sys.stdout.write("     Gradient error          =      4.5E-04 Ry/Bohr\\n")
sys.stdout.write("     number of scf cycles    =   2\\n")
sys.stdout.write("     number of bfgs steps    =   1\\n")
sys.stdout.write("     energy             new  =   -2449.85000000 Ry\\n")
sys.stdout.write("     bfgs converged in   2 scf cycles and   1 bfgs steps\\n")
sys.stdout.flush()
""")

    # Run dummy quick command with monitoring
    with caplog.at_level(logging.INFO):
        run_monitored_qe_subprocess(
            cmd_args=[sys.executable, str(mock_script)],
            calc_dir=calc_dir,
            pw_out=pw_out,
            env=os.environ.copy(),
            mode="relax",
        )

    log_text = caplog.text
    assert "BFGS Step 0 finished!" in log_text
    assert "BFGS Step 1 finished!" in log_text
    assert "BFGS converged in 2 SCF cycles and 1 BFGS steps!" in log_text
