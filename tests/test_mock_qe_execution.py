"""
Tests for QERunner caching, execution handling, and error messages.
"""

import sys
import pytest
from ase import Atoms
from unittest.mock import patch, MagicMock

from hydeph.config import ToolkitConfig
from hydeph.qe_runner import QERunner


def test_qe_runner_cache_hit(tmp_path):
    """When converged pw.out already exists with JOB DONE, QERunner reuses it without invoking subprocess."""
    calc_dir = tmp_path / "calc_frame_0"
    calc_dir.mkdir()
    
    pw_out = calc_dir / "pw.out"
    pw_out.write_text("""
!    total energy              =     -50.1234 Ry

     Forces acting on atoms (cartesian axes, Ry/au):

     atom    1 type  1   force =     0.0010    0.0020    0.0030
     atom    2 type  2   force =    -0.0010   -0.0020   -0.0030
     total force = 0.007

     JOB DONE.
""")

    atoms = Atoms("LiCl", positions=[[0, 0, 0], [2.5, 2.5, 2.5]], cell=[5, 5, 5], pbc=True)
    cfg = ToolkitConfig(
        pristine_cell_path=str(tmp_path),
        pseudo_dir=str(tmp_path),
        pseudopotentials={"Li": "fake.UPF", "Cl": "fake.UPF"}
    )
    from hydeph.qe_runner import write_qe_input
    write_qe_input(atoms, str(calc_dir / "pw.in"), cfg)
    
    runner = QERunner(cfg, n_procs=2, pw_exe="pw.exe")
    
    with patch("subprocess.Popen") as mock_popen:
        labeled = runner.label_single_frame(atoms, str(calc_dir))
        mock_popen.assert_not_called()
        assert "energy" in labeled.info
        assert labeled.info["energy"] is not None
        assert "forces" in labeled.arrays
        assert len(labeled.arrays["forces"]) == 2


def test_qe_runner_error_includes_guidance(tmp_path):
    """Failed calculation raises RuntimeError containing Windows/MS-MPI guidance."""
    calc_dir = tmp_path / "calc_fail"
    calc_dir.mkdir()
    
    atoms = Atoms("LiCl", positions=[[0, 0, 0], [2.5, 2.5, 2.5]], cell=[5, 5, 5], pbc=True)
    cfg = ToolkitConfig(
        pristine_cell_path=str(tmp_path),
        pseudo_dir=str(tmp_path),
        pseudopotentials={"Li": "fake.UPF", "Cl": "fake.UPF"}
    )
    from hydeph.qe_runner import write_qe_input
    write_qe_input(atoms, str(calc_dir / "pw.in"), cfg)
    
    runner = QERunner(cfg, n_procs=2, pw_exe="pw.exe")
    
    # Mock subprocess to exit immediately without generating valid pw.out
    mock_proc = MagicMock()
    mock_proc.poll.return_value = 1
    mock_proc.wait.return_value = 1
    mock_proc.returncode = 1
    
    with patch("subprocess.Popen", return_value=mock_proc):
        with pytest.raises(RuntimeError) as exc_info:
            runner.label_single_frame(atoms, str(calc_dir))
        
        err_msg = str(exc_info.value)
        assert "QE calculation failed" in err_msg
        if sys.platform.startswith("win"):
            assert "QMatSuite" in err_msg or "Microsoft MPI" in err_msg
