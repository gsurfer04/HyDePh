"""
Tests for smart resumption, frame labeling, cache validation, and DFT budget allocation.
"""

import os
import numpy as np
from unittest.mock import patch, MagicMock
from ase import Atoms
import ase.io

from hydeph.config import ToolkitConfig
from hydeph.frame_sampler import FrameSampler
from hydeph.qe_runner import QERunner, check_pw_in_matches_atoms


def test_frame_sampler_labels(tmp_path):
    """Test that all sampling phases generate explicit, persistent labels and metadata."""
    atoms_gs = Atoms("Li2Cl2", positions=[[0, 0, 0], [2, 0, 0], [0, 2, 0], [2, 2, 2]], cell=[4, 4, 4], pbc=True)
    atoms_es = atoms_gs.copy()
    atoms_es.positions[0] += [0.1, 0.1, 0.1]
    
    cfg = ToolkitConfig(
        pristine_cell_path=str(tmp_path),
        pseudo_dir=str(tmp_path),
        pseudopotentials={"Li": "Li.upf", "Cl": "Cl.upf"},
        workspace_dir=str(tmp_path / "ws"),
        n_fc_frames=3,
        n_phonon_frames=3,
        n_md_frames=3,
        random_seed=42,
    )
    
    sampler = FrameSampler(cfg)
    
    # Phase 1: FC frames
    fc_frames = sampler.sample_franck_condon_path(atoms_gs, atoms_es, n_points=3)
    assert len(fc_frames) == 3
    for i, fr in enumerate(fc_frames):
        assert fr.info["sample_type"] == "franck_condon"
        assert fr.info["frame_id"] == f"fc_{i:02d}"
        assert "lambda" in fr.info
        
    # Phase 2: Rattle frames
    rattle_frames = sampler.sample_defect_core_rattling(atoms_gs, n_frames=3)
    assert len(rattle_frames) == 3
    for i, fr in enumerate(rattle_frames):
        assert fr.info["sample_type"] == "core_rattle"
        assert fr.info["frame_id"] == f"rattle_{i:02d}"
        assert "rattle_sigma" in fr.info


def test_candidate_dataset_caching(tmp_path):
    """When candidate_frames.xyz exists, assemble_candidate_dataset reuses it instead of regenerating."""
    ws = tmp_path / "ws_cache"
    ws.mkdir(parents=True)
    
    cand_path = ws / "candidate_frames.xyz"
    existing_frame = Atoms("LiCl", positions=[[0, 0, 0], [1.5, 1.5, 1.5]], cell=[3, 3, 3], pbc=True)
    existing_frame.info["sample_type"] = "franck_condon"
    existing_frame.info["frame_id"] = "fc_00"
    ase.io.write(str(cand_path), [existing_frame], format="extxyz")
    
    cfg = ToolkitConfig(
        pristine_cell_path=str(tmp_path),
        pseudo_dir=str(tmp_path),
        pseudopotentials={"Li": "Li.upf", "Cl": "Cl.upf"},
        workspace_dir=str(ws),
        force_resample=False,
    )
    sampler = FrameSampler(cfg)
    
    atoms_dummy = Atoms("LiCl", positions=[[0, 0, 0], [2, 2, 2]], cell=[3, 3, 3], pbc=True)
    loaded = sampler.assemble_candidate_dataset(atoms_dummy, atoms_dummy)
    
    assert len(loaded) == 1
    # Positions should match existing_frame (1.5), NOT atoms_dummy (2.0)
    assert np.allclose(loaded[0].positions, existing_frame.positions)
    assert loaded[0].info["frame_id"] == "fc_00"


def test_check_pw_in_matches_atoms(tmp_path):
    """Geometry check accurately verifies atomic positions between pw.in and Atoms object."""
    pw_in = tmp_path / "pw.in"
    pw_in.write_text("""
&CONTROL
/
&SYSTEM
/
&ELECTRONS
/
ATOMIC_POSITIONS {angstrom}
Li   0.000000   0.000000   0.000000
Cl   2.500000   2.500000   2.500000
K_POINTS {gamma}
""")
    
    matching_atoms = Atoms("LiCl", positions=[[0.0, 0.0, 0.0], [2.5, 2.5, 2.5]])
    mismatched_atoms = Atoms("LiCl", positions=[[0.0, 0.0, 0.0], [2.8, 2.5, 2.5]])
    
    assert check_pw_in_matches_atoms(str(pw_in), matching_atoms) is True
    assert check_pw_in_matches_atoms(str(pw_in), mismatched_atoms) is False


def test_smart_resumption_and_budget_allocation(tmp_path):
    """
    Simulates resuming an incomplete run:
    2 FC frames already completed in dft_calculations.
    Candidate dataset contains 2 FC frames, 2 Rattle frames, 2 MD frames.
    When max_dft_frames=4 is requested:
      - The 2 completed FC frames are reused via cache.
      - 2 additional frames are selected from Rattle & MD (underrepresented).
      - No duplicate DFT execution occurs on the FC frames.
    """
    ws = tmp_path / "ws_resume"
    dft_dir = ws / "dft_calculations"
    dft_dir.mkdir(parents=True)
    
    # Prepare 6 candidate frames: 2 FC, 2 Rattle, 2 MD
    candidates = []
    types_and_ids = [
        ("fc_00", "franck_condon"),
        ("rattle_00", "core_rattle"),
        ("md_fps_00", "foundation_md_fps"),
        ("fc_01", "franck_condon"),
        ("rattle_01", "core_rattle"),
        ("md_fps_01", "foundation_md_fps"),
    ]
    for idx, (fid, stype) in enumerate(types_and_ids):
        at = Atoms("LiCl", positions=[[0, 0, 0], [2.0 + 0.1 * idx, 2, 2]], cell=[4, 4, 4], pbc=True)
        at.info["frame_id"] = fid
        at.info["label"] = fid
        at.info["sample_type"] = stype
        at.info["candidate_index"] = idx
        candidates.append(at)
        
    # Pre-populate converged DFT calculation for fc_00 (index 0) and fc_01 (index 3)
    def create_converged_calc(dirname, atoms, energy):
        cdir = dft_dir / dirname
        cdir.mkdir(parents=True)
        (cdir / "pw.out").write_text(f"""
!    total energy              =     {energy} Ry

     Forces acting on atoms (cartesian axes, Ry/au):

     atom    1 type  1   force =     0.0010    0.0020    0.0030
     atom    2 type  2   force =    -0.0010   -0.0020   -0.0030
     total force = 0.007

     JOB DONE.
""")
        # Matching pw.in
        (cdir / "pw.in").write_text(f"""
&CONTROL
/
&SYSTEM
/
&ELECTRONS
/
ATOMIC_POSITIONS (angstrom)
Li   {atoms.positions[0][0]:14.8f} {atoms.positions[0][1]:14.8f} {atoms.positions[0][2]:14.8f}
Cl   {atoms.positions[1][0]:14.8f} {atoms.positions[1][1]:14.8f} {atoms.positions[1][2]:14.8f}
K_POINTS (automatic)
  1 1 1 0 0 0
""")

    create_converged_calc("frame_000_fc_00", candidates[0], -45.10)
    create_converged_calc("frame_003_fc_01", candidates[3], -45.20)
    
    cfg = ToolkitConfig(
        pristine_cell_path=str(tmp_path),
        pseudo_dir=str(tmp_path),
        pseudopotentials={"Li": "Li.upf", "Cl": "Cl.upf"},
        workspace_dir=str(ws),
        max_dft_frames=4,
        dft_resume_mode="target_total",
    )
    
    runner = QERunner(cfg, n_procs=1, pw_exe="pw.exe")
    
    # Mock subprocess.Popen for any newly executed frame so it succeeds by writing to f_out
    def mock_popen_side_effect(*args, **kwargs):
        stdout_f = kwargs.get("stdout")
        if stdout_f:
            stdout_f.write(b"""
!    total energy              =     -46.0000 Ry

     Forces acting on atoms (cartesian axes, Ry/au):

     atom    1 type  1   force =     0.0010    0.0000    0.0000
     atom    2 type  2   force =    -0.0010    0.0000    0.0000
     total force = 0.002

     JOB DONE.
""")
            stdout_f.flush()
        mock_p = MagicMock()
        mock_p.poll.return_value = 0
        mock_p.wait.return_value = 0
        mock_p.returncode = 0
        return mock_p

    with patch("subprocess.Popen", side_effect=mock_popen_side_effect) as mock_exec:
        labeled = runner.label_all_frames(candidates, max_frames=4)
        
        # We wanted 4 frames total. 2 were already done (fc_00, fc_01).
        # Therefore, exactly 2 NEW calculations should have been executed!
        assert mock_exec.call_count == 2
        assert len(labeled) == 4
        
        # Verify that all 4 frames in the dataset are unique and labeled
        labels_in_result = [fr.info["frame_id"] for fr in labeled]
        assert "fc_00" in labels_in_result
        assert "fc_01" in labels_in_result
        
        # Verify that the 2 new calculations were selected from the missing sample types
        types_in_result = [fr.info["sample_type"] for fr in labeled]
        assert "core_rattle" in types_in_result
        assert "foundation_md_fps" in types_in_result
        
        # Verify incremental output was written
        out_xyz = ws / "labeled_dft_frames.xyz"
        assert out_xyz.is_file()
        frames_on_disk = ase.io.read(str(out_xyz), ":")
        assert len(frames_on_disk) == 4


def test_resume_cleans_incomplete_scratch(tmp_path):
    """When resuming an incomplete calculation, tmp scratch directory is purged before restarting."""
    calc_dir = tmp_path / "frame_incomplete"
    calc_dir.mkdir()
    
    # Incomplete pw.out (no JOB DONE)
    (calc_dir / "pw.out").write_text("iteration # 1 ... CRASH\n")
    
    # Leftover scratch files
    tmp_scratch = calc_dir / "tmp"
    tmp_scratch.mkdir()
    (tmp_scratch / "prefix.wfc1").write_text("corrupted wavefunctions")
    (tmp_scratch / "prefix.save").mkdir()
    (tmp_scratch / "prefix.save" / "data.xml").write_text("corrupted data")
    
    atoms = Atoms("LiCl", positions=[[0, 0, 0], [2.5, 2.5, 2.5]], cell=[5, 5, 5], pbc=True)
    cfg = ToolkitConfig(
        pristine_cell_path=str(tmp_path),
        pseudo_dir=str(tmp_path),
        pseudopotentials={"Li": "Li.upf", "Cl": "Cl.upf"},
        workspace_dir=str(tmp_path),
    )
    runner = QERunner(cfg, n_procs=1, pw_exe="pw.exe")
    
    def mock_popen_side_effect(*args, **kwargs):
        # By the time subprocess is called, tmp_scratch should be empty!
        assert len(os.listdir(str(tmp_scratch))) == 0
        stdout_f = kwargs.get("stdout")
        if stdout_f:
            stdout_f.write(b"""
!    total energy              =     -50.0000 Ry

     Forces acting on atoms (cartesian axes, Ry/au):

     atom    1 type  1   force =     0.0000    0.0000    0.0000
     atom    2 type  2   force =     0.0000    0.0000    0.0000
     total force = 0.000

     JOB DONE.
""")
            stdout_f.flush()
        mock_p = MagicMock()
        mock_p.poll.return_value = 0
        mock_p.wait.return_value = 0
        mock_p.returncode = 0
        return mock_p

    with patch("subprocess.Popen", side_effect=mock_popen_side_effect):
        labeled = runner.label_single_frame(atoms, str(calc_dir))
        assert "energy" in labeled.info


def _write_converged_qe_stdout(stdout_f, energy_ry=-46.0):
    if stdout_f:
        stdout_f.write(
            f"""
!    total energy              =     {energy_ry:.4f} Ry

     Forces acting on atoms (cartesian axes, Ry/au):

     atom    1 type  1   force =     0.0010    0.0000    0.0000
     atom    2 type  2   force =    -0.0010    0.0000    0.0000
     total force = 0.002

     JOB DONE.
""".encode("utf-8")
        )
        stdout_f.flush()


def _write_failed_qe_stdout(stdout_f):
    if stdout_f:
        stdout_f.write(
            b"""
     iteration # 100     ecut=    40.00 Ry     beta= 0.40
     convergence NOT achieved after 100 iterations: stopping
     JOB DONE.
"""
        )
        stdout_f.flush()


def test_qe_retry_after_rest_succeeds(tmp_path):
    """Failed frame is tagged, deferred until remaining frames finish, and succeeds on retry."""
    ws = tmp_path / "ws_retry_ok"
    ws.mkdir(parents=True)

    candidates = []
    for idx, (fid, stype) in enumerate([
        ("fc_00", "franck_condon"),
        ("rattle_00", "core_rattle"),
        ("md_fps_00", "foundation_md_fps"),
    ]):
        at = Atoms("LiCl", positions=[[0, 0, 0], [2.0 + 0.1 * idx, 2, 2]], cell=[4, 4, 4], pbc=True)
        at.info["frame_id"] = fid
        at.info["label"] = fid
        at.info["sample_type"] = stype
        at.info["candidate_index"] = idx
        candidates.append(at)

    cfg = ToolkitConfig(
        pristine_cell_path=str(tmp_path),
        pseudo_dir=str(tmp_path),
        pseudopotentials={"Li": "Li.upf", "Cl": "Cl.upf"},
        workspace_dir=str(ws),
        retry_failed_dft=True,
        replace_failed_frames=False,
    )
    runner = QERunner(cfg, n_procs=1, pw_exe="pw.exe")

    call_order = []
    rattle_attempts = {"count": 0}

    def mock_popen(*args, **kwargs):
        cwd = os.path.basename(kwargs.get("cwd", ""))
        call_order.append(cwd)
        stdout_f = kwargs.get("stdout")
        if "rattle_00" in cwd:
            rattle_attempts["count"] += 1
            if rattle_attempts["count"] == 1:
                _write_failed_qe_stdout(stdout_f)
            else:
                _write_converged_qe_stdout(stdout_f, -46.5)
        else:
            _write_converged_qe_stdout(stdout_f, -46.0)
        mock_p = MagicMock()
        mock_p.poll.return_value = 0
        mock_p.wait.return_value = 0
        mock_p.returncode = 0
        return mock_p

    with patch("subprocess.Popen", side_effect=mock_popen):
        labeled = runner.label_all_frames(candidates)

    assert len(labeled) == 3
    assert len(runner.last_failed_entries) == 0
    assert call_order == [
        "frame_000_fc_00",
        "frame_001_rattle_00",
        "frame_002_md_fps_00",
        "frame_001_rattle_00",
    ]


def test_qe_second_failure_ignored_when_replace_disabled(tmp_path):
    """When a frame fails twice and replace_failed_frames=False, it is ignored and remaining frames are returned."""
    ws = tmp_path / "ws_ignore"
    ws.mkdir(parents=True)

    candidates = []
    for idx, (fid, stype) in enumerate([
        ("fc_00", "franck_condon"),
        ("rattle_00", "core_rattle"),
        ("md_fps_00", "foundation_md_fps"),
    ]):
        at = Atoms("LiCl", positions=[[0, 0, 0], [2.0 + 0.1 * idx, 2, 2]], cell=[4, 4, 4], pbc=True)
        at.info["frame_id"] = fid
        at.info["label"] = fid
        at.info["sample_type"] = stype
        at.info["candidate_index"] = idx
        candidates.append(at)

    cfg = ToolkitConfig(
        pristine_cell_path=str(tmp_path),
        pseudo_dir=str(tmp_path),
        pseudopotentials={"Li": "Li.upf", "Cl": "Cl.upf"},
        workspace_dir=str(ws),
        retry_failed_dft=True,
        replace_failed_frames=False,
    )
    runner = QERunner(cfg, n_procs=1, pw_exe="pw.exe")

    def mock_popen(*args, **kwargs):
        cwd = os.path.basename(kwargs.get("cwd", ""))
        stdout_f = kwargs.get("stdout")
        if "rattle_00" in cwd:
            _write_failed_qe_stdout(stdout_f)
        else:
            _write_converged_qe_stdout(stdout_f, -46.0)
        mock_p = MagicMock()
        mock_p.poll.return_value = 0
        mock_p.wait.return_value = 0
        mock_p.returncode = 0
        return mock_p

    with patch("subprocess.Popen", side_effect=mock_popen):
        labeled = runner.label_all_frames(candidates)

    assert [fr.info["frame_id"] for fr in labeled] == ["fc_00", "md_fps_00"]
    assert len(runner.last_failed_entries) == 1
    assert runner.last_failed_entries[0]["frame_id"] == "rattle_00"


def test_qe_second_failure_replaced_when_enabled(tmp_path):
    """When a rattle or MD frame fails twice and replace_failed_frames=True, a replacement frame fulfills the quota."""
    ws = tmp_path / "ws_replace"
    ws.mkdir(parents=True)

    candidates = []
    for idx, (fid, stype) in enumerate([
        ("fc_00", "franck_condon"),
        ("rattle_00", "core_rattle"),
        ("md_fps_00", "foundation_md_fps"),
    ]):
        at = Atoms("LiCl", positions=[[0, 0, 0], [2.0 + 0.1 * idx, 2, 2]], cell=[4, 4, 4], pbc=True)
        at.info["frame_id"] = fid
        at.info["label"] = fid
        at.info["sample_type"] = stype
        at.info["candidate_index"] = idx
        candidates.append(at)

    cfg = ToolkitConfig(
        pristine_cell_path=str(tmp_path),
        pseudo_dir=str(tmp_path),
        pseudopotentials={"Li": "Li.upf", "Cl": "Cl.upf"},
        workspace_dir=str(ws),
        retry_failed_dft=True,
        replace_failed_frames=True,
    )
    runner = QERunner(cfg, n_procs=1, pw_exe="pw.exe")

    def mock_popen(*args, **kwargs):
        cwd = os.path.basename(kwargs.get("cwd", ""))
        stdout_f = kwargs.get("stdout")
        if "rattle_00" in cwd or "md_fps_00" in cwd:
            _write_failed_qe_stdout(stdout_f)
        else:
            _write_converged_qe_stdout(stdout_f, -46.0)
        mock_p = MagicMock()
        mock_p.poll.return_value = 0
        mock_p.wait.return_value = 0
        mock_p.returncode = 0
        return mock_p

    with patch("subprocess.Popen", side_effect=mock_popen):
        labeled = runner.label_all_frames(candidates)

    assert len(labeled) == 3
    ids = [fr.info["frame_id"] for fr in labeled]
    types = [fr.info["sample_type"] for fr in labeled]
    assert ids == ["fc_00", "rattle_01", "md_fps_01"]
    assert types == ["franck_condon", "core_rattle", "foundation_md_fps"]

