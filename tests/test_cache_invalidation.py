"""Tests for cache invalidation, stage manifests, and fail-closed checks."""

import os
import pytest
from ase import Atoms

from hydeph.config import ToolkitConfig
from hydeph.qe_runner import check_pw_in_matches_atoms, extract_and_attach_adiabatic_energetics
from hydeph.provenance import compute_atoms_hash, write_stage_manifest, verify_stage_manifest


def test_check_pw_in_matches_atoms_invalidation_cases(tmp_path):
    """Verify that check_pw_in_matches_atoms detects discrepancies across all DFT parameters."""
    atoms = Atoms("Si2", positions=[[0, 0, 0], [1.35, 1.35, 1.35]], cell=[5.43, 5.43, 5.43], pbc=True)
    pw_in_path = tmp_path / "pw.in"

    base_pw_in = """&CONTROL
    calculation = 'scf',
    pseudo_dir = './pseudo',
/
&SYSTEM
    ibrav = 0,
    nat = 2,
    ntyp = 1,
    ecutwfc = 60.0,
    ecutrho = 240.0,
    tot_charge = 0.0,
    tot_magnetization = 0,
    occupations = 'fixed',
    nbnd = 16,
/
&ELECTRONS
    conv_thr = 1.0d-8,
/
ATOMIC_SPECIES
  Si  28.0855  Si_ONCV_PBE-1.2.upf
ATOMIC_POSITIONS crystal
  Si  0.0 0.0 0.0
  Si  0.25 0.25 0.25
CELL_PARAMETERS angstrom
  5.43 0.00 0.00
  0.00 5.43 0.00
  0.00 0.00 5.43
K_POINTS gamma
"""
    pw_in_path.write_text(base_pw_in, encoding="utf-8")

    # Base matching config
    cfg = ToolkitConfig(
        pristine_cell_path="dummy.xyz",
        foundation_potential_path="dummy.txt",
        workspace_dir=str(tmp_path),
        pseudopotentials={"Si": "Si_ONCV_PBE-1.2.upf"},
        ecutwfc=60.0,
        ecutrho=240.0,
        tot_charge=0.0,
        spin_polarized=False,
        tot_magnetization=0,
        conv_thr=1.0e-8,
        kpoints=(1, 1, 1),
    )
    # Target positions in Cartesian matching crystal
    atoms.positions = [[0.0, 0.0, 0.0], [1.3575, 1.3575, 1.3575]]

    assert check_pw_in_matches_atoms(str(pw_in_path), atoms, config=cfg) is True

    # 1. kpoints change
    cfg.kpoints = (2, 2, 2)
    assert check_pw_in_matches_atoms(str(pw_in_path), atoms, config=cfg) is False
    cfg.kpoints = (1, 1, 1)

    # 2. input_dft (functional change)
    cfg.input_dft = "hse"
    assert check_pw_in_matches_atoms(str(pw_in_path), atoms, config=cfg) is False
    cfg.input_dft = None

    # 3. Pseudopotential change
    cfg.pseudopotentials = {"Si": "Si_PBE_new.upf"}
    assert check_pw_in_matches_atoms(str(pw_in_path), atoms, config=cfg) is False
    cfg.pseudopotentials = {"Si": "Si_ONCV_PBE-1.2.upf"}

    # 4. conv_thr change
    cfg.conv_thr = 1.0e-6
    assert check_pw_in_matches_atoms(str(pw_in_path), atoms, config=cfg) is False
    cfg.conv_thr = 1.0e-8

    # 5. ecutwfc change
    cfg.ecutwfc = 80.0
    assert check_pw_in_matches_atoms(str(pw_in_path), atoms, config=cfg) is False
    cfg.ecutwfc = 60.0


def test_stage_manifest_invalidation(tmp_path):
    """Verify that stage manifest detects geometry and config changes and prevents stale cache reuse."""
    ws = tmp_path / "ws"
    ws.mkdir()
    at1 = Atoms("C2", positions=[[0, 0, 0], [1.5, 1.5, 1.5]], cell=[5, 5, 5], pbc=True)
    at2 = at1.copy()
    at2.positions[1] += 0.05

    cfg_dict = {"n_fc_frames": 5, "rattle_mode": "hybrid", "random_seed": 42}
    input_hashes = {"at_gs": compute_atoms_hash(at1), "at_es": compute_atoms_hash(at2)}

    # Write dummy output file
    out_file = ws / "candidate_frames.xyz"
    out_file.write_text("dummy", encoding="utf-8")

    manifest_p = write_stage_manifest(str(ws), stage_idx=2, config_dict=cfg_dict, input_hashes=input_hashes, output_files=["candidate_frames.xyz"])
    assert os.path.isfile(manifest_p)

    # Valid check
    assert verify_stage_manifest(str(ws), stage_idx=2, config_dict=cfg_dict, input_hashes=input_hashes, required_outputs=["candidate_frames.xyz"]) is True

    # Geometry change in at_es
    at2_mod = at2.copy()
    at2_mod.positions[1] += 0.10
    mod_hashes = {"at_gs": compute_atoms_hash(at1), "at_es": compute_atoms_hash(at2_mod)}
    assert verify_stage_manifest(str(ws), stage_idx=2, config_dict=cfg_dict, input_hashes=mod_hashes, required_outputs=["candidate_frames.xyz"]) is False

    # Config change
    mod_cfg = {"n_fc_frames": 10, "rattle_mode": "hybrid", "random_seed": 42}
    assert verify_stage_manifest(str(ws), stage_idx=2, config_dict=mod_cfg, input_hashes=input_hashes, required_outputs=["candidate_frames.xyz"]) is False


def test_extract_and_attach_adiabatic_energetics_fails_loud():
    """Verify that extract_and_attach_adiabatic_energetics fails when lambda=0 or lambda=1 is missing."""
    f_other = Atoms("C2", positions=[[0, 0, 0], [1.5, 1.5, 1.5]])
    f_other.info = {"sample_type": "core_rattle", "energy": -20.0}

    # Missing lambda=0 and lambda=1
    with pytest.raises(RuntimeError, match="Exact adiabatic Delta-SCF energetics require converged lambda=0"):
        extract_and_attach_adiabatic_energetics([f_other], [f_other], allow_fallbacks=False)

    # With allow_fallbacks=True returns empty dict without raising
    res = extract_and_attach_adiabatic_energetics([f_other], [f_other], allow_fallbacks=True)
    assert res == {}
