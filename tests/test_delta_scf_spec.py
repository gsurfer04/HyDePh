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


def test_attach_gap_info_skips_multiline_diagram_for_extxyz_roundtrip(tmp_path):
    """Verify QERunner omits multiline level_diagram and band arrays from atoms.info so extxyz round-trips."""
    import ase.io
    from hydeph.qe.io import write_qe_input

    atoms = Atoms("C2", positions=[[0.0, 0.0, 0.0], [1.5, 1.5, 1.5]], cell=[5.0, 5.0, 5.0], pbc=True)
    upf = tmp_path / "C.upf"
    upf.write_text('<PP_HEADER\nz_valence="4.000000000000000E+00"\n/>\n', encoding="utf-8")

    cfg = ToolkitConfig(
        pristine_cell_path="dummy.xyz",
        foundation_potential_path="dummy.txt",
        workspace_dir=str(tmp_path),
        pseudopotentials={"C": str(upf)},
        tot_charge=0.0,
        spin_polarized=True,
        spin_multiplicity=3,
    )
    runner = QERunner(cfg)
    calc_dir = tmp_path / "calc"
    calc_dir.mkdir()
    write_qe_input(atoms, str(calc_dir / "pw.in"), cfg, tot_magnetization=2)

    (calc_dir / "pw.out").write_text(
        """
     number of electrons       =        8.00 (up:   5.00, down:   3.00)
     total magnetization       =        2.00 Bohr mag/cell

 ------ SPIN UP ------------
          k =   0.0000 0.0000 0.0000 (   100 PWs)   bands (ev):
   -15.2000 -10.5000  -5.2000  -2.0000  -1.2000   3.5000   4.2000   5.1000

 ------ SPIN DOWN ----------
          k =   0.0000 0.0000 0.0000 (   100 PWs)   bands (ev):
   -15.2000 -10.5000  -5.2000   1.8000   2.5000   3.9000   4.5000   5.2000

     the spin up/dw Fermi energies are    -0.5000   -0.6000 ev
!    total energy              =     -20.00000000 Ry

     Forces acting on atoms (cartesian axes, Ry/au):
     atom    1 type  1   force =     0.00000000     0.00000000     0.00000000
     atom    2 type  1   force =     0.00000000     0.00000000     0.00000000

     JOB DONE.
""",
        encoding="utf-8",
    )

    at_lab = runner.label_single_frame(atoms, str(calc_dir))
    assert "dft_gap_eV" in at_lab.info
    assert "dft_level_diagram" not in at_lab.info
    assert "dft_u_bands" not in at_lab.info
    assert "dft_d_bands" not in at_lab.info

    xyz_path = tmp_path / "labeled.xyz"
    ase.io.write(str(xyz_path), [at_lab], format="extxyz")
    reloaded = ase.io.read(str(xyz_path), index=":")
    assert len(reloaded) == 1
    assert reloaded[0].info["dft_gap_eV"] == pytest.approx(at_lab.info["dft_gap_eV"])


def test_parse_qe_band_gap_prefers_ingap_minority_lumo_over_cbm(tmp_path):
    """When majority spin LUMO sits at host CBM while minority spin LUMO is an in-gap defect orbital, select minority spin."""
    pw_out = tmp_path / "pw_vv0.out"
    pw_out.write_text(
        """
     number of electrons       =       10.00 (up:   6.00, down:   4.00)
     total magnetization       =        2.00 Bohr mag/cell

 ------ SPIN UP ------------
          k =   0.0000 0.0000 0.0000 (   100 PWs)   bands (ev):
    2.0000   4.0000   6.0000   8.5000   9.9469   9.9480  11.5431  12.7000

 ------ SPIN DOWN ----------
          k =   0.0000 0.0000 0.0000 (   100 PWs)   bands (ev):
    2.0000   4.0000   6.0000   9.2458  11.0803  11.0820  11.5560  12.7000

     the spin up/dw Fermi energies are    10.4000   10.2000 ev
""",
        encoding="utf-8",
    )

    res = parse_qe_band_gap(str(pw_out))
    assert res is not None
    assert res["spin"] == "down"
    assert res["spin_channel"] == 2
    assert res["homo_band"] == 4
    assert res["lumo_band"] == 5
    assert res["gap_eV"] == pytest.approx(1.8345, abs=1e-3)


def test_extract_and_attach_adiabatic_energetics_reads_calculator_energies(tmp_path):
    """Verify XYZ frames whose energies live on SinglePointCalculator (not .info['energy']) recover ZPL and relaxation."""
    import ase.io
    from ase.calculators.singlepoint import SinglePointCalculator
    from hydeph.qe_runner import extract_and_attach_adiabatic_energetics

    def make_frame(sample_type, lam, energy):
        fr = Atoms("SiC", positions=[[0, 0, 0], [1.5, 0, 0]], cell=[6, 6, 6], pbc=True)
        fr.info = {"sample_type": sample_type, "lambda": lam}
        fr.calc = SinglePointCalculator(fr, energy=energy)
        return fr

    gs_0 = make_frame("franck_condon", 0.0, -100.0)
    gs_1 = make_frame("franck_condon", 1.0, -99.8)
    es_0 = make_frame("franck_condon", 0.0, -98.5)
    es_1 = make_frame("franck_condon", 1.0, -98.8)

    xyz_gs = tmp_path / "labeled_gs.xyz"
    xyz_es = tmp_path / "labeled_es.xyz"
    ase.io.write(str(xyz_gs), [gs_0, gs_1], format="extxyz")
    ase.io.write(str(xyz_es), [es_0, es_1], format="extxyz")

    reloaded_gs = ase.io.read(str(xyz_gs), index=":")
    reloaded_es = ase.io.read(str(xyz_es), index=":")
    assert "energy" not in reloaded_gs[0].info

    at_gs = Atoms("SiC", positions=[[0, 0, 0], [1.5, 0, 0]], cell=[6, 6, 6], pbc=True)
    at_es = Atoms("SiC", positions=[[0, 0, 0], [1.6, 0, 0]], cell=[6, 6, 6], pbc=True)
    res = extract_and_attach_adiabatic_energetics(reloaded_gs, reloaded_es, at_gs=at_gs, at_es=at_es)
    assert res["e_zpl_exact_eV"] == pytest.approx(1.2)
    assert res["erel_gs_eV"] == pytest.approx(0.2)
    assert res["erel_es_eV"] == pytest.approx(0.3)
    assert at_gs.info["e_zpl_exact_eV"] == pytest.approx(1.2)
