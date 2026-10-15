"""
Tests for HyDePh Windows and Cross-Platform MPI / Quantum ESPRESSO support (v0.3.0).
"""

import os
import sys
import tempfile
import pytest
from ase import Atoms
from ase.units import Ry, Bohr

from hydeph import __version__
from hydeph.config import ToolkitConfig
from hydeph.qe_runner import (
    is_windows,
    resolve_pw_executable,
    resolve_mpi_runner,
    build_qe_command,
    write_qe_input,
    parse_qe_output,
    QERunner,
)
from hydeph.delta_trainer import resolve_nep_executable, write_delta_nep_in


def test_version_bump():
    """Verify version 0.3.0."""
    assert __version__ == "0.3.0"


def test_is_windows():
    """Verify is_windows matches sys.platform."""
    expected = sys.platform.startswith("win")
    assert is_windows() == expected


def test_resolve_pw_executable_explicit_file(tmp_path):
    """Verify resolve_pw_executable with an existing explicit file."""
    fake_exe = tmp_path / ("pw.exe" if is_windows() else "pw.x")
    fake_exe.write_text("fake binary")
    resolved = resolve_pw_executable(str(fake_exe))
    assert os.path.abspath(str(fake_exe)) == resolved


def test_resolve_pw_executable_env_var(monkeypatch, tmp_path):
    """Verify resolve_pw_executable using PW_EXE environment variable."""
    fake_exe = tmp_path / "custom_pw.exe"
    fake_exe.write_text("fake binary")
    monkeypatch.setenv("PW_EXE", str(fake_exe))
    resolved = resolve_pw_executable(None)
    assert resolved == str(fake_exe.resolve())


def test_resolve_pw_executable_windows_not_assumed_raises(monkeypatch):
    """On Windows, when pw_exe is not supplied and not in env or PATH, ValueError is raised."""
    monkeypatch.setattr("hydeph.qe_runner.is_windows", lambda: True)
    monkeypatch.delenv("PW_EXE", raising=False)
    monkeypatch.delenv("QE_PW_EXE", raising=False)
    monkeypatch.delenv("QE_BIN_DIR", raising=False)
    monkeypatch.delenv("QE_PATH", raising=False)
    monkeypatch.setattr("shutil.which", lambda name: None)

    with pytest.raises(ValueError) as excinfo:
        resolve_pw_executable(None)
    assert "not installed as a standard system program" in str(excinfo.value)
    assert "not assumed" in str(excinfo.value)


def test_resolve_mpi_runner_serial():
    """When n_procs <= 1, MPI runner should be None (serial execution)."""
    runner, flag = resolve_mpi_runner(n_procs=1)
    assert runner is None
    assert flag is None

    runner, flag = resolve_mpi_runner(n_procs=0)
    assert runner is None
    assert flag is None


def test_resolve_mpi_runner_explicit():
    """Explicit mpi_command should be respected with appropriate flag."""
    runner, flag = resolve_mpi_runner(mpi_command="mpiexec", n_procs=4)
    assert runner == "mpiexec"
    assert flag in ("-n", "-np")

    runner, flag = resolve_mpi_runner(mpi_command="mpirun", n_procs=4)
    assert runner == "mpirun"
    assert flag == "-np"


def test_resolve_mpi_runner_windows_msmpi(monkeypatch):
    """Verify MS-MPI detection logic with MSMPI_BIN environment variable."""
    with tempfile.TemporaryDirectory() as td:
        fake_mpiexec = os.path.join(td, "mpiexec.exe")
        with open(fake_mpiexec, "w") as f:
            f.write("fake mpiexec")

        monkeypatch.setattr("hydeph.qe_runner.is_windows", lambda: True)
        monkeypatch.setenv("MSMPI_BIN", td)

        runner, flag = resolve_mpi_runner(n_procs=4)
        assert runner == fake_mpiexec
        assert flag == "-n"


def test_build_qe_command_serial():
    """Serial execution builds command without MPI wrapper."""
    cmd = build_qe_command("pw.exe", pw_in="pw.in", n_procs=1)
    assert cmd == ["pw.exe", "-in", "pw.in"]


def test_build_qe_command_with_mpi():
    """MPI execution includes mpi runner and process count flag."""
    cmd = build_qe_command("pw.exe", pw_in="pw.in", n_procs=4, mpi_command="mpiexec")
    assert cmd[0] == "mpiexec"
    assert cmd[1] in ("-n", "-np")
    assert cmd[2] == "4"
    assert cmd[3] == "pw.exe"
    assert cmd[4:] == ["-in", "pw.in"]


def test_build_qe_command_with_extra_mpi_args():
    """Extra MPI args (e.g. MS-MPI env vars) are passed directly to launcher."""
    extra = ["-env", "OMP_NUM_THREADS", "1"]
    cmd = build_qe_command("pw.exe", pw_in="pw.in", n_procs=4, mpi_command="mpiexec", mpi_args=extra)
    assert cmd[0] == "mpiexec"
    assert cmd[3:6] == ["-env", "OMP_NUM_THREADS", "1"]
    assert "pw.exe" in cmd
    assert cmd[-2:] == ["-in", "pw.in"]


def test_write_qe_input_windows_path_normalization(tmp_path):
    """Verify Windows backslashes in pseudo_dir are converted to forward slashes."""
    atoms = Atoms("LiCl", positions=[[0, 0, 0], [2.5, 2.5, 2.5]], cell=[5, 5, 5], pbc=True)
    pw_in = tmp_path / "pw.in"
    
    # Create fake UPF files
    pseudo_dir = tmp_path / "pseudo"
    pseudo_dir.mkdir()
    (pseudo_dir / "Li.UPF").write_text("fake upf")
    (pseudo_dir / "Cl.UPF").write_text("fake upf")

    cfg = ToolkitConfig(
        pristine_cell_path=str(tmp_path),
        pseudo_dir=str(pseudo_dir),  # May have backslashes on Windows
        pseudopotentials={"Li": "Li.UPF", "Cl": "Cl.UPF"}
    )
    
    write_qe_input(atoms, str(pw_in), cfg)
    content = pw_in.read_text()
    
    # Check that pseudo_dir in &CONTROL has forward slashes
    assert "\\" not in [line for line in content.splitlines() if "pseudo_dir =" in line][0]
    assert "outdir = './tmp'" in content
    assert "CELL_PARAMETERS" in content
    assert "ATOMIC_POSITIONS" in content


def test_parse_qe_output(tmp_path):
    """Verify parsing of total energy and forces from QE output."""
    sample_out = """
     Program PWSCF v.7.5 starts on 12Oct2026
     ...
!    total energy              =     -47.99240624 Ry

     Forces acting on atoms (cartesian axes, Ry/au):

     atom    1 type  1   force =     0.00120000    -0.00240000     0.00360000
     atom    2 type  2   force =    -0.00120000     0.00240000    -0.00360000

     total force = 0.00587874
     JOB DONE.
"""
    pw_out = tmp_path / "pw.out"
    pw_out.write_text(sample_out)
    
    energy_ev, forces = parse_qe_output(str(pw_out))
    assert energy_ev is not None
    assert pytest.approx(energy_ev, rel=1e-5) == -47.99240624 * Ry
    assert forces is not None
    assert forces.shape == (2, 3)
    assert pytest.approx(forces[0, 0], rel=1e-5) == 0.0012 * (Ry / Bohr)


def test_resolve_nep_executable_explicit(tmp_path):
    """Verify resolve_nep_executable with explicit path."""
    fake_nep = tmp_path / ("nep.exe" if is_windows() else "nep")
    fake_nep.write_text("fake binary")
    resolved = resolve_nep_executable(str(fake_nep))
    assert resolved == str(fake_nep.resolve())


def test_resolve_nep_executable_fallback():
    """Verify fallback name based on platform."""
    res = resolve_nep_executable(None)
    assert isinstance(res, str)
    if is_windows():
        assert res.endswith(".exe") or res == "nep"
    else:
        assert "nep" in res


def test_write_delta_nep_in(tmp_path):
    """Verify nep.in content generation."""
    nep_in_file = tmp_path / "nep.in"
    write_delta_nep_in(str(nep_in_file), ["Cl", "Li"], neurons=25, batch=1000, steps=15000)
    text = nep_in_file.read_text()
    assert "type 2 Cl Li" in text
    assert "neuron 25" in text
    assert "batch 1000" in text
    assert "generation 15000" in text


def test_toolkit_config_validation(tmp_path):
    """Verify ToolkitConfig validation and custom executable attributes."""
    pristine_file = tmp_path / "pristine.xyz"
    pristine_file.write_text("fake atoms")
    
    pot_file = tmp_path / "nep.txt"
    pot_file.write_text("fake potential")
    
    pseudo_dir = tmp_path / "pseudo"
    pseudo_dir.mkdir()
    (pseudo_dir / "Li.UPF").write_text("fake upf")

    cfg = ToolkitConfig(
        pristine_cell_path=str(pristine_file),
        foundation_potential_path=str(pot_file),
        pseudo_dir=str(pseudo_dir),
        pseudopotentials={"Li": "Li.UPF"},
        pw_exe="pw.exe",
        mpi_command="mpiexec",
        mpi_args=["-n", "4"],
        nep_exe="nep.exe"
    )
    assert cfg.validate() is True
    assert cfg.pw_exe == "pw.exe"
    assert cfg.mpi_command == "mpiexec"
    assert cfg.mpi_args == ["-n", "4"]
    assert cfg.nep_exe == "nep.exe"
