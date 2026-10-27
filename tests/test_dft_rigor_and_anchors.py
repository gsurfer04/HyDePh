"""Tests for Batch 4: DFT rigor, spin parity bug fix, hybrid functionals, and anchor frame priority."""

import pytest
import numpy as np
import logging
from ase import Atoms
from hydeph.config import ToolkitConfig
from hydeph.qe_runner import (
    write_qe_input,
    inspect_upf_metadata,
    QERunner,
)


def test_fixed_occupations_default(tmp_path):
    """Verify write_qe_input defaults to fixed occupations for defect calculations."""
    atoms = Atoms("C2", positions=[[0, 0, 0], [1.5, 0, 0]], cell=[5, 5, 5], pbc=True)
    pw_in = tmp_path / "pw_fixed.in"
    cfg = ToolkitConfig(
        pristine_cell_path="dummy.xyz",
        pseudopotentials={"C": "C.upf"},
    )
    assert cfg.qe_occupations == "fixed"
    write_qe_input(atoms, str(pw_in), cfg)
    content = pw_in.read_text(encoding="utf-8")
    assert "occupations = 'fixed'" in content
    assert "smearing" not in content

    # Verify explicit smearing works when configured
    cfg_smear = ToolkitConfig(
        pristine_cell_path="dummy.xyz",
        pseudopotentials={"C": "C.upf"},
        qe_occupations="smearing",
        smearing="fermi-dirac",
        degauss=0.01,
    )
    pw_smear = tmp_path / "pw_smear.in"
    write_qe_input(atoms, str(pw_smear), cfg_smear)
    content_smear = pw_smear.read_text(encoding="utf-8")
    assert "occupations = 'smearing'" in content_smear
    assert "smearing = 'fermi-dirac'" in content_smear
    assert "degauss = 0.01" in content_smear


def test_atom_scaled_conv_thr(tmp_path):
    """Verify default conv_thr scales inversely with atom count (1e-10 Ry/atom)."""
    # 10-atom cell: conv_thr should be 1.0d-09
    atoms10 = Atoms("C10", positions=np.zeros((10, 3)), cell=[10, 10, 10], pbc=True)
    pw10 = tmp_path / "pw10.in"
    cfg = ToolkitConfig(
        pristine_cell_path="dummy.xyz",
        pseudopotentials={"C": "C.upf"},
    )
    assert cfg.conv_thr is None
    write_qe_input(atoms10, str(pw10), cfg)
    content10 = pw10.read_text(encoding="utf-8")
    assert "conv_thr = 1.0d-09," in content10

    # 200-atom cell: conv_thr should be 2.0d-08
    atoms200 = Atoms("C200", positions=np.zeros((200, 3)), cell=[20, 20, 20], pbc=True)
    pw200 = tmp_path / "pw200.in"
    write_qe_input(atoms200, str(pw200), cfg)
    content200 = pw200.read_text(encoding="utf-8")
    assert "conv_thr = 2.0d-08," in content200

    # Explicit threshold overrides auto-scaling
    cfg_explicit = ToolkitConfig(
        pristine_cell_path="dummy.xyz",
        pseudopotentials={"C": "C.upf"},
        conv_thr="5.0d-6",
    )
    pw_exp = tmp_path / "pw_exp.in"
    write_qe_input(atoms200, str(pw_exp), cfg_explicit)
    content_exp = pw_exp.read_text(encoding="utf-8")
    assert "conv_thr = 5.0d-6," in content_exp


def test_upf_metadata_cutoffs_and_norm_conserving_ecutrho(tmp_path, caplog):
    """Verify UPF suggested cutoff parsing, NC ecutrho enforcement, and low cutoff warnings."""
    upf_c = tmp_path / "C.upf"
    upf_c.write_text(
        '<UPF version="2.0.1">\n'
        '  <PP_HEADER element="C" pseudo_type="NC" is_norm_conserving="T" '
        'suggested_cutoff_wfc="80.0" suggested_cutoff_rho="320.0" z_valence="4.00"/>\n'
        '</UPF>\n',
        encoding="utf-8",
    )
    meta = inspect_upf_metadata(str(upf_c))
    assert meta["is_norm_conserving"] is True
    assert meta["z_valence"] == 4.0
    assert meta["suggested_cutoff_wfc"] == 80.0
    assert meta["suggested_cutoff_rho"] == 320.0

    # Test NC dual rule: ecutrho = 4 * ecutwfc
    cfg = ToolkitConfig(
        pristine_cell_path="dummy.xyz",
        pseudopotentials={"C": str(upf_c)},
        ecutwfc=50.0,
        spin_multiplicity=1,
    )
    pw_in = tmp_path / "pw_nc.in"
    atoms = Atoms("C2", positions=np.zeros((2, 3)), cell=[5, 5, 5], pbc=True)

    with caplog.at_level(logging.WARNING):
        write_qe_input(atoms, str(pw_in), cfg)

    content = pw_in.read_text(encoding="utf-8")
    # For NC, ecutrho = 4 * 50 = 200.0
    assert "ecutrho = 200.0," in content
    # Caplog should capture warning that ecutwfc=50 is below suggested 80.0 and < 60 for hard element C
    assert any("lower than UPF suggested cutoff 80.0" in r.message for r in caplog.records)


def test_hybrid_dft_hse_support(tmp_path):
    """Verify hybrid functional parameters (input_dft, exx_fraction, nqx) in config and pw.in."""
    atoms = Atoms("C2", positions=np.zeros((2, 3)), cell=[5, 5, 5], pbc=True)
    pw_in = tmp_path / "pw_hybrid.in"
    pristine_file = tmp_path / "pristine.xyz"
    pristine_file.write_text("2\ncell\nC 0 0 0\nC 1 1 1\n", encoding="utf-8")
    nep_file = tmp_path / "nep.txt"
    nep_file.write_text("version 1\n", encoding="utf-8")
    upf_file = tmp_path / "C.upf"
    upf_file.write_text("<UPF>\n", encoding="utf-8")

    cfg = ToolkitConfig(
        pristine_cell_path=str(pristine_file),
        foundation_potential_path=str(nep_file),
        pseudopotentials={"C": str(upf_file)},
        input_dft="hse",
        exx_fraction=0.25,
        nqx=(1, 1, 1),
        ecutfock=80.0,
    )
    cfg.validate()

    write_qe_input(atoms, str(pw_in), cfg)
    content = pw_in.read_text(encoding="utf-8")
    assert "input_dft = 'hse'," in content
    assert "exx_fraction = 0.25," in content
    assert "nqx1 = 1," in content
    assert "nqx2 = 1," in content
    assert "nqx3 = 1," in content
    assert "ecutfock = 80.0," in content

    # Validation errors on invalid parameters
    cfg_bad_exx = ToolkitConfig(
        pristine_cell_path=str(pristine_file),
        foundation_potential_path=str(nep_file),
        pseudopotentials={"C": str(upf_file)},
        input_dft="hse",
        exx_fraction=1.5,
    )
    with pytest.raises(ValueError, match="Invalid exx_fraction"):
        cfg_bad_exx.validate()

    cfg_bad_nqx = ToolkitConfig(
        pristine_cell_path=str(pristine_file),
        foundation_potential_path=str(nep_file),
        pseudopotentials={"C": str(upf_file)},
        input_dft="hse",
        nqx=(1, 0, 1),
    )
    with pytest.raises(ValueError, match="Invalid nqx"):
        cfg_bad_nqx.validate()


def test_spin_parity_error_and_auto_adjust_bug_fix(tmp_path, caplog):
    """Verify that spin-parity mismatch raises ValueError by default, and auto-adjust fixes odd electron tot_mag=0."""
    upf_c = tmp_path / "C.upf"
    upf_c.write_text(
        '<UPF version="2.0.1">\n'
        '  <PP_HEADER element="C" pseudo_type="NC" z_valence="4.00"/>\n'
        '</UPF>\n',
        encoding="utf-8",
    )

    # 9 atoms * 4 valence - 1 charge = 35 electrons (odd electron count)
    atoms_odd = Atoms("C9", positions=np.zeros((9, 3)), cell=[10, 10, 10], pbc=True)
    pw_in = tmp_path / "pw_parity.in"

    # Default auto_spin_parity=False and spin_multiplicity=1 (tot_magnetization=0, even)
    cfg_fail = ToolkitConfig(
        pristine_cell_path="dummy.xyz",
        pseudopotentials={"C": str(upf_c)},
        tot_charge=1.0,
        spin_multiplicity=1,  # tot_magnetization = 0 (mismatches odd 35 electrons)
    )
    assert cfg_fail.auto_spin_parity is False

    with pytest.raises(ValueError, match="Spin parity mismatch"):
        write_qe_input(atoms_odd, str(pw_in), cfg_fail)

    # With auto_spin_parity=True: tot_mag=0 should auto-adjust to 1 (not stay 0 as in old bug!)
    cfg_auto = ToolkitConfig(
        pristine_cell_path="dummy.xyz",
        pseudopotentials={"C": str(upf_c)},
        tot_charge=1.0,
        spin_multiplicity=1,  # tot_magnetization = 0
        auto_spin_parity=True,
    )
    with caplog.at_level(logging.WARNING):
        write_qe_input(atoms_odd, str(pw_in), cfg_auto)

    content = pw_in.read_text(encoding="utf-8")
    assert "tot_magnetization = 1" in content
    assert cfg_auto.tot_magnetization == 1
    assert cfg_auto.spin_multiplicity == 2
    assert any("Auto-adjusting tot_magnetization -> 1" in r.message for r in caplog.records)


def test_anchor_frame_priority_in_dft_budget(tmp_path, monkeypatch, caplog):
    """Verify that lambda=0 and lambda=1 anchor frames are always calculated first regardless of sample type."""
    # Build candidate frames: 5 FC frames, 5 rattle frames, 5 MD frames
    frames = []
    # 5 FC frames with lambda: [0.0, 1.0, 0.5, -0.2, 1.2]
    lams = [0.0, 1.0, 0.5, -0.2, 1.2]
    for i, lam in enumerate(lams):
        at = Atoms("C2", positions=np.zeros((2, 3)), cell=[5, 5, 5], pbc=True)
        at.info["sample_type"] = "franck_condon"
        at.info["frame_id"] = f"fc_{i:02d}"
        at.info["lambda"] = float(lam)
        at.info["candidate_index"] = i
        frames.append(at)

    for i in range(5):
        at = Atoms("C2", positions=np.zeros((2, 3)), cell=[5, 5, 5], pbc=True)
        at.info["sample_type"] = "core_rattle"
        at.info["frame_id"] = f"rattle_{i:02d}"
        at.info["candidate_index"] = 5 + i
        frames.append(at)

    for i in range(5):
        at = Atoms("C2", positions=np.zeros((2, 3)), cell=[5, 5, 5], pbc=True)
        at.info["sample_type"] = "foundation_md_fps"
        at.info["frame_id"] = f"md_{i:02d}"
        at.info["candidate_index"] = 10 + i
        frames.append(at)

    cfg = ToolkitConfig(
        pristine_cell_path="dummy.xyz",
        workspace_dir=str(tmp_path),
        pseudopotentials={"C": "C.upf"},
        max_dft_frames=2,  # Budget only allows 2 frames!
    )

    runner = QERunner(cfg, pw_exe="pw.x")

    # Mock label_single_frame to return atoms with dummy energy
    calculated_labels = []

    def mock_label_single_frame(at, calc_dir, retry=False, **kwargs):
        calculated_labels.append(at.info["frame_id"])
        at_copy = at.copy()
        at_copy.info = dict(at.info)
        at_copy.info["energy"] = -100.0
        at_copy.arrays["forces"] = np.zeros((len(at), 3))
        return at_copy

    monkeypatch.setattr(runner, "label_single_frame", mock_label_single_frame)

    with caplog.at_level(logging.WARNING):
        runner.label_all_frames(frames, max_frames=2)

    # Exactly 2 frames calculated, and they MUST be fc_00 (lambda=0) and fc_01 (lambda=1)
    assert len(calculated_labels) == 2
    assert calculated_labels == ["fc_00", "fc_01"]

    # Budget warning check: total budget 2 < 20
    assert any("fewer than 20 frames" in r.message for r in caplog.records)
