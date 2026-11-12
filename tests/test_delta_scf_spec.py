"""Tests for Delta-SCF spec resolution, in-gap level diagram, and state continuity checks."""

import json
import pytest
from ase import Atoms

from hydeph.config import ToolkitConfig
from hydeph.qe_runner import (
    resolve_delta_scf_spec,
    parse_qe_band_gap,
    QERunner,
)


def test_resolve_delta_scf_spec_strict_upf_valence(tmp_path):
    """Verify that resolve_delta_scf_spec raises ValueError when UPF is missing or lacks z_valence."""
    atoms = Atoms("Si2", positions=[[0, 0, 0], [1.35, 1.35, 1.35]], cell=[5.43, 5.43, 5.43], pbc=True)
    cfg = ToolkitConfig(
        pristine_cell_path="dummy.xyz",
        foundation_potential_path="dummy.txt",
        workspace_dir=str(tmp_path),
        pseudopotentials={"Si": "Si_missing.upf"},
    )
    # Missing UPF file on disk
    with pytest.raises((ValueError, FileNotFoundError)):
        resolve_delta_scf_spec(atoms, cfg, workspace_dir=str(tmp_path))

    # UPF file exists but has no z_valence
    bad_upf = tmp_path / "Si_bad.upf"
    bad_upf.write_text("<UPF version=\"2.0.1\">\n</UPF>\n", encoding="utf-8")
    cfg.pseudopotentials = {"Si": str(bad_upf)}
    with pytest.raises(ValueError, match="Could not parse valence electron count"):
        resolve_delta_scf_spec(atoms, cfg, workspace_dir=str(tmp_path))


def test_resolve_delta_scf_spec_valid(tmp_path):
    """Verify that resolve_delta_scf_spec writes delta_scf_spec.json and constructs exact occupations."""
    atoms = Atoms("C2", positions=[[0, 0, 0], [1.5, 1.5, 1.5]], cell=[5, 5, 5], pbc=True)
    upf = tmp_path / "C.upf"
    upf.write_text('<PP_HEADER\nz_valence="4.000000000000000E+00"\n/>\n', encoding="utf-8")

    cfg = ToolkitConfig(
        pristine_cell_path="dummy.xyz",
        foundation_potential_path="dummy.txt",
        workspace_dir=str(tmp_path),
        pseudopotentials={"C": str(upf)},
        tot_charge=0.0,
        spin_polarized=True,
        spin_multiplicity=1,  # singlet
        delta_scf_homo_band=4,
        delta_scf_lumo_band=5,
        delta_scf_spin_channel=2,
        delta_scf_lumo_degeneracy=2,
    )

    spec = resolve_delta_scf_spec(atoms, cfg, workspace_dir=str(tmp_path))
    assert spec["nelec"] == 8
    assert spec["tot_magnetization"] == 0
    assert spec["spin_channel"] == 2
    assert spec["homo_band"] == 4
    assert spec["lumo_band"] == 5
    assert spec["lumo_degeneracy"] == 2
    assert spec["occupations_down"][3] == 0.0  # promoted hole
    assert spec["occupations_down"][4] == 0.5  # degenerate fraction 1/2
    assert spec["occupations_down"][5] == 0.5

    spec_file = tmp_path / "delta_scf_spec.json"
    assert spec_file.is_file()
    with open(spec_file, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["nelec"] == 8


def test_parse_qe_band_gap_level_diagram(tmp_path):
    """Verify that parse_qe_band_gap produces an in-gap level diagram."""
    pw_out = tmp_path / "pw.out"
    sample_text = """
     number of electrons       =        8.00 (up:   4.00, down:   4.00)
     total magnetization       =        0.00 Bohr mag/cell

 ------ SPIN UP ------------
          k =   0.0000 0.0000 0.0000 (   100 PWs)   bands (ev):
   -15.2000 -10.5000  -5.2000  -1.2000   3.5000   4.2000   5.1000   6.0000

 ------ SPIN DOWN ----------
          k =   0.0000 0.0000 0.0000 (   100 PWs)   bands (ev):
   -15.2000 -10.5000  -5.2000  -1.4000   1.8000   1.8020   5.1000   6.0000

     the spin up/dw Fermi energies are    -0.5000   -0.6000 ev
"""
    pw_out.write_text(sample_text, encoding="utf-8")

    res = parse_qe_band_gap(str(pw_out))
    assert res is not None
    assert "level_diagram" in res
    assert "Band" in res["level_diagram"]
    assert "Candidate Optical Transitions:" in res["level_diagram"]
    assert res["homo_band"] == 4
    assert res["lumo_band"] == 5
    assert res["spin_channel"] == 2  # down channel has smaller gap (3.2 eV vs 4.7 eV)
    assert res["lumo_degeneracy"] == 2  # 1.8000 and 1.8020 within 5 meV


def test_es_continuity_outlier_detection(tmp_path, monkeypatch):
    """Verify that label_excited_state_frames flags state-crossing outliers."""
    atoms = Atoms("C2", positions=[[0, 0, 0], [1.5, 1.5, 1.5]], cell=[5, 5, 5], pbc=True)
    upf = tmp_path / "C.upf"
    upf.write_text('<PP_HEADER\nz_valence="4.000000000000000E+00"\n/>\n', encoding="utf-8")

    cfg = ToolkitConfig(
        pristine_cell_path="dummy.xyz",
        foundation_potential_path="dummy.txt",
        workspace_dir=str(tmp_path),
        pseudopotentials={"C": str(upf)},
        tot_charge=0.0,
        spin_polarized=False,
    )
    runner = QERunner(cfg)

    # Mock label_single_frame to return synthetic energies
    energies_es = [-18.0, -18.1, -22.0, -18.05]  # frame 2 has inverted/anomalous energy
    call_idx = 0

    def mock_label_single_frame(atoms, calc_dir, retry=False, occupations_mode=None, constrained_occupations=None, worker_idx=0):
        nonlocal call_idx
        at = atoms.copy()
        at.info["energy"] = energies_es[call_idx % len(energies_es)]
        at.arrays["forces"] = np.zeros((len(atoms), 3))
        call_idx += 1
        return at

    import numpy as np
    monkeypatch.setattr(runner, "label_single_frame", mock_label_single_frame)

    # Create 4 candidate frames with ground state energies of -20.0 eV
    # Normal delta_e_vert = +2.0 eV. Frame 2: -22.0 - (-20.0) = -2.0 eV (sign reversal!)
    frames = []
    for i in range(4):
        f = atoms.copy()
        f.info = {
            "candidate_index": i,
            "frame_id": f"fc_{i:02d}",
            "sample_type": "franck_condon",
            "energy": -20.0,
        }
        frames.append(f)

    # Pass pre-set constrained occupations to bypass DFT gs lookup
    labeled = runner.label_excited_state_frames(
        frames=frames,
        at_gs=atoms,
        constrained_occupations=([1.0]*4, [1.0]*4),
        n_workers=1,
    )
    assert len(labeled) == 4
    assert labeled[0].info["delta_e_vert_eV"] == pytest.approx(2.0)
    assert labeled[2].info["delta_e_vert_eV"] == pytest.approx(-2.0)
    assert labeled[2].info.get("delta_scf_state_crossing_outlier") is True
    assert labeled[0].info.get("delta_scf_state_crossing_outlier") is None
