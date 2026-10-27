"""Tests for Batch 5: Strict configuration, cache invalidation, and manifest integrity."""

import os
import json
import pytest
import numpy as np
from ase import Atoms
from hydeph.config import ToolkitConfig
from hydeph.qe_runner import (
    check_pw_in_matches_atoms,
    parse_qe_relaxation_output,
    write_qe_input,
)
from hydeph.delta_trainer import _compute_training_digest


def test_check_pw_in_matches_atoms_fail_closed(tmp_path):
    """Verify check_pw_in_matches_atoms is fail-closed and checks cell, species, positions, and config."""
    # 1. Non-existent file must return False (not True!)
    atoms = Atoms("C2", positions=[[0, 0, 0], [1.5, 0, 0]], cell=[5, 5, 5], pbc=True)
    assert check_pw_in_matches_atoms(str(tmp_path / "missing.in"), atoms) is False

    # 2. Malformed file must return False
    malformed = tmp_path / "malformed.in"
    malformed.write_text("CORRUPTED JUNK\n", encoding="utf-8")
    assert check_pw_in_matches_atoms(str(malformed), atoms) is False

    # 3. Create valid pw.in
    pw_valid = tmp_path / "valid.in"
    cfg = ToolkitConfig(
        pristine_cell_path="dummy.xyz",
        pseudopotentials={"C": "C.upf"},
        ecutwfc=60.0,
        tot_charge=0.0,
        spin_multiplicity=1,
    )
    write_qe_input(atoms, str(pw_valid), cfg)

    # Valid matching input returns True
    assert check_pw_in_matches_atoms(str(pw_valid), atoms, config=cfg) is True

    # 4. Mismatched species returns False
    atoms_diff_elem = Atoms("Si2", positions=[[0, 0, 0], [1.5, 0, 0]], cell=[5, 5, 5], pbc=True)
    assert check_pw_in_matches_atoms(str(pw_valid), atoms_diff_elem, config=cfg) is False

    # 5. Mismatched cell returns False
    atoms_diff_cell = Atoms("C2", positions=[[0, 0, 0], [1.5, 0, 0]], cell=[6, 6, 6], pbc=True)
    assert check_pw_in_matches_atoms(str(pw_valid), atoms_diff_cell, config=cfg) is False

    # 6. Mismatched positions returns False
    atoms_diff_pos = Atoms("C2", positions=[[0, 0, 0], [1.8, 0, 0]], cell=[5, 5, 5], pbc=True)
    assert check_pw_in_matches_atoms(str(pw_valid), atoms_diff_pos, config=cfg) is False

    # 7. Mismatched config cutoffs / charge returns False
    cfg_diff_cutoff = ToolkitConfig(
        pristine_cell_path="dummy.xyz",
        pseudopotentials={"C": "C.upf"},
        ecutwfc=80.0,  # Modified from 60.0
        tot_charge=0.0,
        spin_multiplicity=1,
    )
    assert check_pw_in_matches_atoms(str(pw_valid), atoms, config=cfg_diff_cutoff) is False


def test_strict_qe_relaxation_caching(tmp_path):
    """Verify parse_qe_relaxation_output strictly enforces convergence markers and rejects unconverged runs."""
    template = Atoms("C2", positions=[[0, 0, 0], [1.5, 0, 0]], cell=[5, 5, 5], pbc=True)

    # 1. Output with 'Begin final coordinates' is accepted
    out_converged = tmp_path / "converged.out"
    out_converged.write_text(
        "JOB DONE.\n"
        "!    total energy              =    -50.00000000 Ry\n"
        "Begin final coordinates\n"
        "ATOMIC_POSITIONS (angstrom)\n"
        "C   0.000000   0.000000   0.000000\n"
        "C   1.540000   0.000000   0.000000\n"
        "End final coordinates\n",
        encoding="utf-8",
    )
    at_rel, e = parse_qe_relaxation_output(str(out_converged), template, require_converged=True)
    assert at_rel is not None
    assert np.isclose(at_rel.positions[1, 0], 1.54)

    # 2. Output with 'bfgs converged in' is accepted
    out_bfgs = tmp_path / "bfgs.out"
    out_bfgs.write_text(
        "JOB DONE.\n"
        "bfgs converged in   8 scf cycles and   4 bfgs steps\n"
        "!    total energy              =    -50.20000000 Ry\n"
        "ATOMIC_POSITIONS (angstrom)\n"
        "C   0.000000   0.000000   0.000000\n"
        "C   1.530000   0.000000   0.000000\n",
        encoding="utf-8",
    )
    at_rel2, e2 = parse_qe_relaxation_output(str(out_bfgs), template, require_converged=True)
    assert at_rel2 is not None
    assert np.isclose(at_rel2.positions[1, 0], 1.53)

    # 3. Output with nstep exhaustion ('The maximum number of steps has been reached') is rejected
    out_nstep = tmp_path / "nstep.out"
    out_nstep.write_text(
        "JOB DONE.\n"
        "The maximum number of steps has been reached.\n"
        "!    total energy              =    -49.80000000 Ry\n"
        "ATOMIC_POSITIONS (angstrom)\n"
        "C   0.000000   0.000000   0.000000\n"
        "C   1.520000   0.000000   0.000000\n",
        encoding="utf-8",
    )
    at_nstep, _ = parse_qe_relaxation_output(str(out_nstep), template, require_converged=True)
    assert at_nstep is None

    # 4. Incomplete/timed-out output (missing both markers) is rejected
    out_timeout = tmp_path / "timeout.out"
    out_timeout.write_text(
        "iteration #  10\n"
        "!    total energy              =    -49.00000000 Ry\n"
        "ATOMIC_POSITIONS (angstrom)\n"
        "C   0.000000   0.000000   0.000000\n"
        "C   1.510000   0.000000   0.000000\n",
        encoding="utf-8",
    )
    at_timeout, _ = parse_qe_relaxation_output(str(out_timeout), template, require_converged=True)
    assert at_timeout is None


def test_complete_delta_nep_digest_hashing(tmp_path):
    """Verify that changing any architectural hyperparameter or foundation potential changes the digest."""
    nep_file = tmp_path / "nep.txt"
    nep_file.write_text("model_version_1\n", encoding="utf-8")

    atoms = Atoms("C2", positions=[[0, 0, 0], [1.5, 0, 0]], cell=[5, 5, 5], pbc=True)
    atoms.info["energy"] = -10.0
    atoms.arrays["forces"] = np.zeros((2, 3))
    frames = [atoms]

    cfg_base = ToolkitConfig(
        pristine_cell_path="dummy.xyz",
        foundation_potential_path=str(nep_file),
        nep_cutoff_radial=4.5,
        nep_cutoff_angular=3.5,
        nep_n_max_radial=6,
        nep_basis_size=8,
    )
    base_digest = _compute_training_digest(frames, neurons=30, batch=2000, steps=1000, config=cfg_base)

    # 1. Changing nep_cutoff_radial changes digest
    cfg_rcut = ToolkitConfig(
        pristine_cell_path="dummy.xyz",
        foundation_potential_path=str(nep_file),
        nep_cutoff_radial=5.0,  # Changed
        nep_cutoff_angular=3.5,
        nep_n_max_radial=6,
        nep_basis_size=8,
    )
    assert _compute_training_digest(frames, neurons=30, batch=2000, steps=1000, config=cfg_rcut) != base_digest

    # 2. Changing nep_basis_size changes digest
    cfg_basis = ToolkitConfig(
        pristine_cell_path="dummy.xyz",
        foundation_potential_path=str(nep_file),
        nep_cutoff_radial=4.5,
        nep_cutoff_angular=3.5,
        nep_n_max_radial=6,
        nep_basis_size=12,  # Changed
    )
    assert _compute_training_digest(frames, neurons=30, batch=2000, steps=1000, config=cfg_basis) != base_digest

    # 3. Modifying foundation potential content changes digest
    nep_file2 = tmp_path / "nep2.txt"
    nep_file2.write_text("model_version_2_MODIFIED\n", encoding="utf-8")
    cfg_found = ToolkitConfig(
        pristine_cell_path="dummy.xyz",
        foundation_potential_path=str(nep_file2),
        nep_cutoff_radial=4.5,
        nep_cutoff_angular=3.5,
        nep_n_max_radial=6,
        nep_basis_size=8,
    )
    assert _compute_training_digest(frames, neurons=30, batch=2000, steps=1000, config=cfg_found) != base_digest


def test_from_dict_rejects_unknown_keys_with_suggestions():
    """Verify ToolkitConfig.from_dict rejects unknown keys and suggests close matches."""
    # Typo: ecutwfcc -> ecutwfc
    with pytest.raises(ValueError, match="Did you mean 'ecutwfc'"):
        ToolkitConfig.from_dict({"pristine_cell_path": "dummy.xyz", "ecutwfcc": 80.0})

    # Typo: pseudopotential -> pseudopotentials
    with pytest.raises(ValueError, match="Did you mean 'pseudopotentials'"):
        ToolkitConfig.from_dict({"pristine_cell_path": "dummy.xyz", "pseudopotential": {"C": "C.upf"}})

    # Completely unknown key
    with pytest.raises(ValueError, match="Unknown configuration key: 'completely_unknown_parameter'"):
        ToolkitConfig.from_dict({"pristine_cell_path": "dummy.xyz", "completely_unknown_parameter": 123})


def test_comprehensive_range_validation(tmp_path):
    """Verify ToolkitConfig.validate() enforces strict parameter ranges and detects conflicts."""
    pristine_file = tmp_path / "pristine.xyz"
    pristine_file.write_text("1\ncell\nC 0 0 0\n", encoding="utf-8")
    nep_file = tmp_path / "nep.txt"
    nep_file.write_text("version 1\n", encoding="utf-8")
    upf_file = tmp_path / "C.upf"
    upf_file.write_text("<UPF>\n", encoding="utf-8")

    def make_valid_config(**kwargs):
        base = {
            "pristine_cell_path": str(pristine_file),
            "foundation_potential_path": str(nep_file),
            "pseudopotentials": {"C": str(upf_file)},
        }
        base.update(kwargs)
        return ToolkitConfig(**base)

    # 1. Temperatures <= 0
    with pytest.raises(ValueError, match="Invalid temperatures"):
        make_valid_config(temperatures=[-10.0, 300.0]).validate()

    with pytest.raises(ValueError, match="Invalid temperatures"):
        make_valid_config(temperatures=[]).validate()

    # 2. Repeat < 1
    with pytest.raises(ValueError, match="Invalid supercell_repeat"):
        make_valid_config(supercell_repeat=(2, 0, 2)).validate()

    # 3. K-points < 1
    with pytest.raises(ValueError, match="Invalid kpoints"):
        make_valid_config(kpoints=(1, -1, 1)).validate()

    # 4. Cutoff <= 0
    with pytest.raises(ValueError, match="Invalid ecutwfc"):
        make_valid_config(ecutwfc=-10.0).validate()

    # 5. ecutrho < 4 * ecutwfc
    with pytest.raises(ValueError, match="Invalid ecutrho"):
        make_valid_config(ecutwfc=50.0, ecutrho=150.0).validate()

    # 6. n_fc_frames < 2
    with pytest.raises(ValueError, match="Invalid n_fc_frames"):
        make_valid_config(n_fc_frames=1).validate()

    # 7. Conflicting charge specifications
    with pytest.raises(ValueError, match="Conflicting charge specifications"):
        ToolkitConfig(pristine_cell_path="dummy.xyz", charge=-1.0, tot_charge=1.0)

    # 8. Conflicting spin specifications
    with pytest.raises(ValueError, match="Conflicting spin specifications"):
        ToolkitConfig(pristine_cell_path="dummy.xyz", spin_multiplicity=3, tot_magnetization=0)


def test_export_resolved_manifest(tmp_path):
    """Verify export_resolved_manifest writes config_resolved.json containing hashes and versions."""
    pristine_file = tmp_path / "pristine.xyz"
    pristine_file.write_text("1\ncell\nC 0 0 0\n", encoding="utf-8")
    nep_file = tmp_path / "nep.txt"
    nep_file.write_text("foundation_weights\n", encoding="utf-8")
    upf_file = tmp_path / "C.upf"
    upf_file.write_text("<UPF_data>\n", encoding="utf-8")

    ws = tmp_path / "workspace"
    cfg = ToolkitConfig(
        pristine_cell_path=str(pristine_file),
        foundation_potential_path=str(nep_file),
        pseudopotentials={"C": str(upf_file)},
        workspace_dir=str(ws),
    )
    cfg.validate()

    manifest_path = cfg.export_resolved_manifest()
    assert os.path.isfile(manifest_path)
    assert os.path.basename(manifest_path) == "config_resolved.json"

    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "software_versions" in data
    assert "hydeph" in data["software_versions"]
    assert "input_hashes" in data
    assert data["input_hashes"]["pristine_cell_path"]["sha256"] is not None
    assert data["input_hashes"]["foundation_potential_path"]["sha256"] is not None
    assert "C" in data["input_hashes"]["pseudopotentials"]
    assert data["config"]["workspace_dir"] == str(ws)
