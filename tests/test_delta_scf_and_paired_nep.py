"""Tests for Batch 3: Real Delta-SCF excited states and paired Delta-NEP architecture."""

import os
import pytest
import numpy as np
from unittest.mock import MagicMock
from ase import Atoms
from hydeph.config import ToolkitConfig
from hydeph.qe_runner import write_qe_input, build_delta_scf_occupations, parse_qe_band_gap
from hydeph.calculator import HybridDefectCalculator, PairedDefectCalculator
from hydeph.defect_builder import DefectBuilder
from hydeph.cc_diagram import ConfigurationCoordinateEngine


class DummyCalc:
    """Mock ASE calculator."""
    def __init__(self, energy_val=-100.0):
        self.energy_val = energy_val
        self.atoms = None

    def get_potential_energy(self, atoms=None, *args, **kwargs):
        return self.energy_val

    def get_forces(self, atoms=None, *args, **kwargs):
        if atoms is not None:
            return np.zeros((len(atoms), 3))
        return np.zeros((len(self.atoms), 3))


def test_build_delta_scf_occupations():
    """Verify build_delta_scf_occupations creates valid constrained cards with strict particle/spin conservation."""
    # 1. Triplet system (e.g. NV-): nelec=14, tot_mag=2 (n_up=8, n_dw=6), nbnd=12
    # Promote electron in spin down from HOMO (band 6) to LUMO (band 7)
    occ_up, occ_dw = build_delta_scf_occupations(
        nelec=14, tot_magnetization=2, nbnd=12, homo_band=6, lumo_band=7, spin_channel=2
    )
    assert len(occ_up) == 12
    assert len(occ_dw) == 12
    assert sum(occ_up) == 8.0
    assert sum(occ_dw) == 6.0
    assert occ_dw[5] == 0.0  # Band 6 demoted
    assert occ_dw[6] == 1.0  # Band 7 promoted

    # 2. Spin-up promotion in doublet: nelec=7, tot_mag=1 (n_up=4, n_dw=3), nbnd=8
    occ_up2, occ_dw2 = build_delta_scf_occupations(
        nelec=7, tot_magnetization=1, nbnd=8, homo_band=4, lumo_band=5, spin_channel=1
    )
    assert sum(occ_up2) == 4.0
    assert sum(occ_dw2) == 3.0
    assert occ_up2[3] == 0.0
    assert occ_up2[4] == 1.0

    # 3. Incompatible parity raises ValueError
    with pytest.raises(ValueError, match="incompatible parity"):
        build_delta_scf_occupations(nelec=8, tot_magnetization=1, nbnd=10, homo_band=4, lumo_band=5)

    # 4. Invalid spin channel raises ValueError
    with pytest.raises(ValueError, match="Invalid spin_channel"):
        build_delta_scf_occupations(nelec=14, tot_magnetization=2, nbnd=12, homo_band=6, lumo_band=7, spin_channel=3)


def test_write_qe_input_from_input_occupations(tmp_path):
    """Verify write_qe_input writes occupations='from_input' and OCCUPATIONS card."""
    atoms = Atoms("C2", positions=[[0, 0, 0], [1.5, 0, 0]], cell=[5, 5, 5], pbc=True)
    upf_file = tmp_path / "C.upf"
    upf_file.write_text("<UPF version=\"2.0.1\">\n  <PP_HEADER z_valence=\"4.0\" element=\"C\"/>\n</UPF>\n", encoding="utf-8")

    cfg = ToolkitConfig(
        pristine_cell_path=str(tmp_path / "pristine.xyz"),
        pseudopotentials={"C": str(upf_file)},
        spin_multiplicity=1,
    )

    pw_in = tmp_path / "pw.in"
    occ_up = [1.0, 1.0, 1.0, 1.0, 0.0, 0.0]
    occ_dw = [1.0, 1.0, 0.0, 1.0, 0.0, 0.0]

    write_qe_input(
        atoms,
        str(pw_in),
        cfg,
        occupations_mode="from_input",
        constrained_occupations=(occ_up, occ_dw),
    )

    content = pw_in.read_text(encoding="utf-8")
    assert "occupations = 'from_input'" in content
    assert "nbnd = 6," in content
    assert "OCCUPATIONS" in content
    assert "1.0  1.0  1.0  1.0  0.0  0.0" in content
    assert "1.0  1.0  0.0  1.0  0.0  0.0" in content


def test_parse_qe_band_gap_detects_defect_orbital(tmp_path):
    """Verify parse_qe_band_gap extracts spin channel, homo_band, and lumo_band."""
    pw_out = tmp_path / "pw.out"
    sample_text = """
     the spin up/dw Fermi energies are    -1.5000   -1.5000 ev

     ------ SPIN UP ------------
     bands (ev):
        -12.500  -8.400  -4.200  -2.100   2.500   3.800

     ------ SPIN DOWN ----------
     bands (ev):
        -12.400  -8.300  -4.100  -1.800   0.400   3.900

     the spin
    """
    pw_out.write_text(sample_text, encoding="utf-8")

    gap_info = parse_qe_band_gap(str(pw_out))
    assert gap_info is not None
    # Down gap is (-1.800 to 0.400) = 2.2 eV; Up gap is (-2.100 to 2.500) = 4.6 eV
    # The smaller gap is the in-gap defect channel: spin down (channel 2)
    assert gap_info["spin"] == "down"
    assert gap_info["spin_channel"] == 2
    assert gap_info["homo_band"] == 4
    assert gap_info["lumo_band"] == 5
    assert pytest.approx(gap_info["gap_eV"], rel=1e-3) == 2.2


def test_parse_qe_band_gap_fixed_occupations_spin_polarized(tmp_path):
    """Verify parse_qe_band_gap extracts orbital promotion for fixed occupations (no Fermi output)."""
    pw_out = tmp_path / "pw_fixed.out"
    sample_text = """
     number of electrons       =         9.00 (up: 5.00, down: 4.00)
     number of Kohn-Sham states=            6

 ------ SPIN UP ------------
          k = 0.0000 0.0000 0.0000 ( 21250 PWs)   bands (ev):
    -5.0000  -4.0000  -3.0000   0.8591   4.8644   6.2577

 ------ SPIN DOWN ----------
          k = 0.0000 0.0000 0.0000 ( 21250 PWs)   bands (ev):
    -5.0000  -4.0000  -3.0000   0.8619   5.4425   6.5451

     highest occupied, lowest unoccupied level (ev):     4.8644    5.4425
    """
    pw_out.write_text(sample_text, encoding="utf-8")

    gap_info = parse_qe_band_gap(str(pw_out))
    assert gap_info is not None
    assert gap_info["spin"] == "up"
    assert gap_info["spin_channel"] == 1
    assert gap_info["homo_band"] == 5
    assert gap_info["lumo_band"] == 6
    assert pytest.approx(gap_info["gap_eV"], rel=1e-3) == 1.3933
    assert pytest.approx(gap_info["homo_eV"], rel=1e-3) == 4.8644
    assert pytest.approx(gap_info["lumo_eV"], rel=1e-3) == 6.2577


def test_defect_builder_delta_scf_relaxation(tmp_path):
    """Verify DefectBuilder invokes Delta-SCF relaxation with constrained occupations."""
    pristine_path = str(tmp_path / "pristine.xyz")
    pristine = Atoms("C8", positions=np.eye(8, 3) * 1.5, cell=[6, 6, 6], pbc=True)
    pristine.write(pristine_path)

    cfg = ToolkitConfig(
        pristine_cell_path=pristine_path,
        workspace_dir=str(tmp_path / "ws_delta_scf"),
        excited_state_method="delta_scf",
        ground_state_relax_method="foundation",
        spin_multiplicity=1,
        delta_scf_homo_band=126,
        delta_scf_lumo_band=127,
        delta_scf_spin_channel=2,
    )

    mock_qe = MagicMock()
    def mock_relax(atoms, *args, **kwargs):
        res = atoms.copy()
        res.positions[0] += np.array([0.15, 0.0, 0.0])
        return res
    mock_qe.run_relaxation.side_effect = mock_relax

    builder = DefectBuilder(cfg, calc_foundation=DummyCalc(-200.0), calc_qe=mock_qe)
    _, defect_sc = builder.build_supercell_and_defect()
    at_gs = builder.relax_ground_state(defect_sc)

    at_es, dq_fc = builder.generate_excited_state_geometry(at_gs)
    assert at_es.info.get("es_source") == "delta_scf"
    assert dq_fc > 0.0

    # Verify mock_qe was called with occupations_mode="from_input"
    call_kwargs = mock_qe.run_relaxation.call_args[1]
    assert call_kwargs.get("occupations_mode") == "from_input"
    assert call_kwargs.get("constrained_occupations") is not None


def test_paired_defect_calculator():
    """Verify PairedDefectCalculator computes independent GS/ES surfaces and energy gap."""
    mock_base = MagicMock()
    mock_base.results = {"energy": -100.0, "forces": np.zeros((2, 3))}
    mock_base.implemented_properties = ["energy", "forces"]

    mock_delta_gs = MagicMock()
    mock_delta_gs.results = {"energy": -1.5, "forces": np.zeros((2, 3))}
    mock_delta_gs.implemented_properties = ["energy", "forces"]

    mock_delta_es = MagicMock()
    mock_delta_es.results = {"energy": 0.8, "forces": np.zeros((2, 3))}
    mock_delta_es.implemented_properties = ["energy", "forces"]

    calc_gs = HybridDefectCalculator(calc_base=mock_base, calc_delta=mock_delta_gs)
    calc_es = HybridDefectCalculator(calc_base=mock_base, calc_delta=mock_delta_es)

    paired = PairedDefectCalculator(calc_gs, calc_es)

    atoms = Atoms("C2", positions=[[0, 0, 0], [1.5, 0, 0]], cell=[5, 5, 5], pbc=True)

    e_gs = paired.get_potential_energy_gs(atoms)
    e_es = paired.get_potential_energy_es(atoms)
    gap = paired.get_energy_gap(atoms)

    assert pytest.approx(e_gs) == -101.5
    assert pytest.approx(e_es) == -99.2
    assert pytest.approx(gap) == 2.3  # Delta U = (-99.2) - (-101.5) = 2.3 eV


def test_configuration_coordinate_with_paired_calc(tmp_path):
    """Verify ConfigurationCoordinateEngine evaluates real excited PES when calc_es is provided."""
    at_gs = Atoms("C2", positions=[[0, 0, 0], [1.5, 0, 0]], cell=[5, 5, 5], pbc=True)
    at_es = at_gs.copy()
    at_es.positions[1, 0] += 0.2

    cfg = ToolkitConfig(
        pristine_cell_path=str(tmp_path),
        workspace_dir=str(tmp_path / "ws_cc_paired"),
        zpl_energy_eV=2.10,
    )

    # GS: harmonic centered at lambda = 0
    class GSCalc:
        def get_potential_energy(self, atoms=None, *args, **kwargs):
            pos = atoms.positions if atoms is not None else at_gs.positions
            dx = pos[1, 0] - 1.5
            return 0.5 * 20.0 * (dx ** 2)

    # ES: harmonic centered at lambda = 1 (pos[1, 0] = 1.7) with stiffer curvature k = 30.0 + offset 2.1 eV
    class ESCalc:
        def get_potential_energy(self, atoms=None, *args, **kwargs):
            pos = atoms.positions if atoms is not None else at_es.positions
            dx = pos[1, 0] - 1.7
            return 2.10 + 0.5 * 30.0 * (dx ** 2)

    engine = ConfigurationCoordinateEngine(cfg, GSCalc())
    res = engine.calculate_cc_curve(at_gs, at_es, n_points=11, calc_es=ESCalc())

    assert res["e_zpl_eV"] == 2.10
    # ES at lambda=1 should sit at 2.10 eV
    idx_1 = int(np.argmin(np.abs(res["lambda_grid"] - 1.0)))
    assert pytest.approx(res["e_es"][idx_1], rel=1e-4) == 2.10
    # ES relaxation energy should reflect true ESCalc curvature, not GSCalc curvature
    assert res["delta_e_rel_es_eV"] > res["delta_e_rel_gs_eV"]


def test_dft_es_candidates_config(tmp_path):
    """Verify dft_es_candidates validation in ToolkitConfig."""
    cfg_default = ToolkitConfig(pristine_cell_path=str(tmp_path))
    assert cfg_default.dft_es_candidates == "all"

    cfg_fc = ToolkitConfig(pristine_cell_path=str(tmp_path), dft_es_candidates="fc_path")
    assert cfg_fc.dft_es_candidates == "fc_path"

    cfg_case = ToolkitConfig(pristine_cell_path=str(tmp_path), dft_es_candidates="ALL")
    assert cfg_case.dft_es_candidates == "all"

    with pytest.raises(ValueError, match="Invalid dft_es_candidates"):
        ToolkitConfig(pristine_cell_path=str(tmp_path), dft_es_candidates="random_mode")


def test_label_excited_state_frames_caching(tmp_path, monkeypatch):
    """Verify label_excited_state_frames labels candidate frames and correctly reuses cache."""
    import ase.io
    from hydeph.qe_runner import QERunner

    at_gs = Atoms("C2", positions=[[0, 0, 0], [1.5, 0, 0]], cell=[5, 5, 5], pbc=True)
    at_1 = at_gs.copy()
    at_1.info["candidate_index"] = 0
    at_1.info["frame_id"] = "fc_00"
    at_1.info["sample_type"] = "franck_condon"
    at_1.info["lambda"] = 0.0

    at_2 = at_gs.copy()
    at_2.positions[1, 0] += 0.1
    at_2.info["candidate_index"] = 1
    at_2.info["frame_id"] = "rattle_00"
    at_2.info["sample_type"] = "core_rattle"

    upf = tmp_path / "C.upf"
    upf.write_text("<UPF version=\"2.0.1\">\n  <PP_HEADER z_valence=\"4.0\" element=\"C\"/>\n</UPF>\n", encoding="utf-8")

    cfg = ToolkitConfig(
        pristine_cell_path=str(tmp_path / "pristine.xyz"),
        workspace_dir=str(tmp_path / "ws_es_label"),
        pseudopotentials={"C": str(upf)},
        spin_multiplicity=1,
    )
    qe = QERunner(cfg)

    call_count = [0]
    def mock_label_single(atoms, calc_dir, retry=False, occupations_mode=None, constrained_occupations=None):
        call_count[0] += 1
        res = atoms.copy()
        res.info["energy"] = -150.0 + call_count[0]
        res.arrays["forces"] = np.zeros((len(atoms), 3))
        os.makedirs(calc_dir, exist_ok=True)
        with open(os.path.join(calc_dir, "pw.out"), "w") as f:
            f.write("JOB DONE.\n!    total energy              =  -150.00000000 Ry\nForces acting on atoms (cartesian axes, Ry/au):\n     atom    1 type  1   force =   0.0  0.0  0.0\n     atom    2 type  1   force =   0.0  0.0  0.0\n")
        from hydeph.qe_runner import write_qe_input
        write_qe_input(
            atoms,
            os.path.join(calc_dir, "pw.in"),
            cfg,
            occupations_mode=occupations_mode,
            constrained_occupations=constrained_occupations,
        )
        return res

    monkeypatch.setattr(qe, "label_single_frame", mock_label_single)

    dummy_occ = ([1.0, 1.0, 0.0, 0.0], [1.0, 0.0, 1.0, 0.0])
    labeled = qe.label_excited_state_frames([at_1, at_2], at_gs=at_gs, constrained_occupations=dummy_occ)

    assert len(labeled) == 2
    assert labeled[0].info["sample_type"] == "franck_condon"
    assert labeled[0].info["electronic_state"] == "excited"
    assert labeled[1].info["sample_type"] == "core_rattle"
    assert labeled[1].info["electronic_state"] == "excited"
    assert call_count[0] == 2

    # Verify labeled_frames_es.xyz was written
    out_es_xyz = os.path.join(cfg.workspace_dir, "labeled_frames_es.xyz")
    assert os.path.isfile(out_es_xyz)
    read_back = ase.io.read(out_es_xyz, index=":")
    assert len(read_back) == 2

    # Re-running with cache should reload converged calculation
    call_count[0] = 0
    labeled_cached = qe.label_excited_state_frames([at_1, at_2], at_gs=at_gs, constrained_occupations=dummy_occ)
    assert len(labeled_cached) == 2
    assert call_count[0] == 2

