"""Tests for the unified property-driven workflow (auto supercell, Hessian splicing, force & surface routing)."""

from pathlib import Path
import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.emt import EMT
from ase.vibrations import Vibrations

from hydeph.config import ToolkitConfig
from hydeph.lineshape import (
    LineshapeEngine,
    apply_acoustic_sum_rule_and_projection,
    embed_phonon_supercell,
)
from hydeph.provenance import generate_provenance_manifest
from hydeph.system_profile import (
    assess_relaxation_asymmetry,
    build_spliced_hessian,
    load_dft_optical_forces,
    resolve_phonon_supercell_repeat,
)


def _make_base_cfg(tmp_path, pristine, **kwargs):
    p_file = tmp_path / "pristine.xyz"
    import ase.io
    ase.io.write(str(p_file), pristine)
    pot_file = tmp_path / "pot.txt"
    pot_file.write_text("dummy")
    defaults = {
        "pristine_cell_path": str(p_file),
        "foundation_potential_path": str(pot_file),
        "workspace_dir": str(tmp_path / "ws"),
        "spin_multiplicity": 1,
    }
    defaults.update(kwargs)
    return ToolkitConfig(**defaults)


def test_resolve_phonon_supercell_repeat_by_host_class():
    """Auto supercell size maps stiff, polar, and soft hosts to (8,8,8), (8,8,3), and (4,4,4)."""
    uc_diam = Atoms("C8", positions=np.zeros((8, 3)), cell=[3.57, 3.57, 3.57], pbc=True)
    uc_sic = Atoms("Si4C4", positions=np.zeros((8, 3)), cell=[3.09, 3.09, 10.13], pbc=True)
    uc_nacl = Atoms("Na4Cl4", positions=np.zeros((8, 3)), cell=[5.69, 5.69, 5.69], pbc=True)

    assert resolve_phonon_supercell_repeat(uc_diam, hw_max_meV=165.0) == (8, 8, 8)
    assert resolve_phonon_supercell_repeat(uc_sic, hw_max_meV=118.0) == (8, 8, 3)
    assert resolve_phonon_supercell_repeat(uc_nacl, hw_max_meV=32.0) == (4, 4, 4)


def test_config_and_provenance_accept_auto_phonon_repeat(tmp_path):
    """Verify 'auto' phonon_supercell_repeat validates and serializes cleanly."""
    uc = bulk("Cu", "fcc", a=3.6, cubic=True)
    cfg = _make_base_cfg(
        tmp_path,
        uc,
        phonon_supercell_repeat="auto",
        hessian_embedding_mode="auto",
        phonon_force_state="auto",
        anharmonic_md_surface="auto",
    )
    cfg.validate()
    assert cfg.phonon_supercell_repeat == "auto"
    assert cfg.finite_size_acoustic_cutoff is False

    at = uc.copy()
    prov = generate_provenance_manifest(cfg, at, at.copy())
    assert prov["lineshape_parameters"]["phonon_supercell_repeat"] == "auto"


def test_build_spliced_hessian_matches_full_vibrations(tmp_path):
    """Alkauskas bulk+core spliced Hessian reproduces full Vibrations frequencies accurately."""
    uc = bulk("Al", "fcc", a=4.05, cubic=True)
    rep = (3, 3, 3)
    at_large = uc.repeat(rep)
    at_large.info["defect_center"] = at_large.cell.sum(axis=0) / 2.0
    calc = EMT()

    # Exact full Vibrations Hessian with ASR
    at_v = at_large.copy()
    at_v.calc = calc
    vib = Vibrations(at_v, name=str(tmp_path / "full_vib"), delta=0.01)
    vib.run()
    H_full = apply_acoustic_sum_rule_and_projection(vib.get_vibrations().get_hessian_2d(), at_v)

    H_spl = build_spliced_hessian(
        pristine_uc=uc,
        at_large=at_large,
        rep_large=rep,
        calc_gs=calc,
        calc_bulk=calc,
        core_radius_A=5.0,
        delta=0.01,
        n_workers=2,
        cache_dir=str(tmp_path / "cache"),
        fingerprint="test_al",
    )
    assert H_spl.shape == H_full.shape
    assert np.allclose(H_spl, H_spl.T, atol=1e-6)
    max_diff = float(np.max(np.abs(H_full - H_spl)))
    assert max_diff < 5e-3, f"Spliced Hessian drifted from full Vibrations: max_diff={max_diff}"


class AsymmetricHarmonicCalc(Calculator):
    """Analytic pair-mode calculator where ES and GS have different spring constants around central atoms."""

    implemented_properties = ["energy", "forces"]

    def __init__(self, k, idx_a=0, idx_b=1, x0_a=0.0, x0_b=2.5, e0=0.0):
        super().__init__()
        self.k = float(k)
        self.idx_a = int(idx_a)
        self.idx_b = int(idx_b)
        self.x0_a = float(x0_a)
        self.x0_b = float(x0_b)
        self.e0 = float(e0)

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        da = atoms.positions[self.idx_a, 0] - self.x0_a
        db = atoms.positions[self.idx_b, 0] - self.x0_b
        self.results["energy"] = self.e0 + 0.5 * self.k * float(da ** 2 + db ** 2)
        forces = np.zeros_like(atoms.positions)
        forces[self.idx_a, 0] = -self.k * da
        forces[self.idx_b, 0] = -self.k * db
        self.results["forces"] = forces


def test_auto_force_state_switches_to_emission_on_strong_asymmetry(tmp_path):
    """When Delta E_rel,ES / Delta E_rel,GS >= 1.8, auto selects F_GS(R_ES) - F_ES(R_ES) over absorption F."""
    uc = Atoms("NaCl", positions=[[0, 0, 0], [2.5, 2.5, 2.5]], cell=[5.0, 5.0, 5.0], pbc=True)
    cfg = _make_base_cfg(
        tmp_path,
        uc,
        supercell_repeat=(2, 2, 2),
        phonon_supercell_repeat=(3, 3, 3),
        phonon_embedding_method="force",
        phonon_force_state="auto",
        anharmonic_md_steps=8,
        anharmonic_n_trajectories=1,
    )
    at_gs = uc.repeat((2, 2, 2))
    v_pos = at_gs.positions[1].copy()  # Cl anion vacancy at [2.5, 2.5, 2.5] matches build_supercell_and_defect
    del at_gs[1]
    at_gs.info["defect_center"] = v_pos
    cell_len = 10.0
    rel = (((at_gs.positions - v_pos) + cell_len / 2.0) % cell_len) - cell_len / 2.0
    order = np.argsort(np.linalg.norm(rel, axis=1))
    ia, ib = int(order[0]), int(order[1])
    x_a0, x_b0 = float(at_gs.positions[ia, 0]), float(at_gs.positions[ib, 0])

    at_es = at_gs.copy()
    at_es.positions[ia, 0] += 0.2
    at_es.positions[ib, 0] -= 0.2

    # k_ES = 20 vs k_GS = 5 -> Delta_E_rel,ES / Delta_E_rel,GS = 4.0 >= 1.8!
    c_gs = AsymmetricHarmonicCalc(k=5.0, idx_a=ia, idx_b=ib, x0_a=x_a0, x0_b=x_b0, e0=0.0)
    c_es = AsymmetricHarmonicCalc(k=20.0, idx_a=ia, idx_b=ib, x0_a=x_a0 + 0.2, x0_b=x_b0 - 0.2, e0=1.2)
    _, _, ratio = assess_relaxation_asymmetry(at_gs, at_es, c_gs, c_es)
    assert ratio == pytest.approx(4.0, rel=1e-4)

    _, _, f_auto = embed_phonon_supercell(at_gs, at_es, cfg, c_gs, calc_es=c_es)
    # F_GS(R_ES) = ±(5.0 * 0.2) = ±1.0 eV/A (vs absorption F_ES(R_GS) = ±4.0 eV/A)
    max_f = float(np.max(np.abs(f_auto)))
    assert max_f == pytest.approx(1.0, abs=0.2), f"Expected emission force ~1.0 eV/A, got {max_f}"


def test_dft_core_force_override_and_embedded_md_surface_auto(tmp_path):
    """Exact DFT core forces populate Wigner-Seitz mask and auto MD surface chooses 'gs' when embedded."""
    uc = bulk("C", "diamond", a=3.57, cubic=True)
    cfg = _make_base_cfg(
        tmp_path,
        uc,
        supercell_repeat=(2, 2, 2),
        phonon_supercell_repeat=(3, 3, 3),
        phonon_embedding_method="force",
        phonon_force_state="absorption",
        use_dft_core_forces=True,
        anharmonic_md_surface="auto",
        anharmonic_md_steps=16,
        anharmonic_thermalization_steps=4,
        anharmonic_n_trajectories=1,
    )
    at_gs = uc.repeat((2, 2, 2))
    del at_gs[0]
    at_gs.info["defect_center"] = np.array([0.0, 0.0, 0.0])
    # Zero-sum DFT core optical force signature on first 4 neighbors
    f_dft = np.zeros_like(at_gs.positions)
    f_dft[0] = [1.5, 0.0, 0.0]
    f_dft[1] = [-1.5, 0.0, 0.0]
    at_gs.info["dft_optical_forces_abs"] = f_dft
    at_es = at_gs.copy()

    loaded = load_dft_optical_forces(at_gs, at_es, str(tmp_path / "ws"), force_state="absorption")
    assert loaded is not None and np.allclose(loaded, f_dft)

    c_gs = AsymmetricHarmonicCalc(k=10.0, idx_a=0, idx_b=1, x0_a=at_gs.positions[0, 0], x0_b=at_gs.positions[1, 0], e0=0.0)
    c_es = AsymmetricHarmonicCalc(k=10.0, idx_a=0, idx_b=1, x0_a=at_gs.positions[0, 0] + 0.01, x0_b=at_gs.positions[1, 0] - 0.01, e0=2.0)

    at_large, _, f_large = embed_phonon_supercell(at_gs, at_es, cfg, c_gs, calc_es=c_es)
    assert at_large.info.get("is_embedded") is True
    # Must contain injected ~1.5 eV/A signature, not tiny 0.1 eV/A from c_es
    assert float(np.max(np.abs(f_large))) > 1.0
    assert np.allclose(np.sum(f_large, axis=0), 0.0, atol=1e-6)

    # Verify auto MD surface skips calc_es dynamics on embedded supercell
    class SpyES(AsymmetricHarmonicCalc):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.calls = 0

        def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
            self.calls += 1
            super().calculate(atoms, properties, system_changes)

    spy_es = SpyES(k=10.0, idx_a=0, idx_b=1, x0_a=at_gs.positions[0, 0] + 0.01, x0_b=at_gs.positions[1, 0] - 0.01, e0=2.0)
    eng = LineshapeEngine(cfg, c_gs, calc_es=spy_es)
    hw_grid = np.linspace(1.0, 100.0, 200)
    s_cont = np.exp(-0.5 * ((hw_grid - 50.0) / 5.0) ** 2)
    eng.run_anharmonic_sampling(at_large, at_large.copy(), hw_grid, s_cont, total_S=1.0, f_es_mode=f_large)
    assert spy_es.calls == 0


def test_qe_gs_on_es_conjugate_scf_loads_emission_forces_and_slurm_hook(tmp_path):
    """Verify qe_gs_on_es/pw.out + qe_es_relax/pw.out loads exact emission forces and SLURM runs [3/3]."""
    from hydeph.qe.runner import QERunner

    uc = bulk("C", "diamond", a=3.57, cubic=True)
    ws = tmp_path / "ws"
    ws.mkdir(parents=True, exist_ok=True)
    pdir = tmp_path / "pseudo"
    pdir.mkdir(parents=True, exist_ok=True)
    (pdir / "C.upf").write_text('<UPF version="2.0.1"><PP_HEADER element="C" pseudo_type="NC" z_valence="4.00"/></UPF>')
    cfg = _make_base_cfg(
        tmp_path,
        uc,
        workspace_dir=str(ws),
        pseudo_dir=str(pdir),
        pseudopotentials={"C": "C.upf"},
        supercell_repeat=(1, 1, 1),
        phonon_supercell_repeat=(2, 2, 2),
        phonon_force_state="auto",
        use_dft_core_forces=True,
        spin_multiplicity=3,
    )
    at_es = uc.copy()
    # Stub minimal QE pw.out outputs in qe_gs_on_es and qe_es_relax via monkeypatching ase.io.read isn't needed
    # if we test load_dft_optical_forces with valid extxyz/pw mock or direct QE output snippet:
    from unittest.mock import patch

    at_gs_es = at_es.copy()
    f_gs_es = np.zeros((len(at_es), 3))
    f_gs_es[0, 0] = 0.8
    f_gs_es[1, 0] = -0.8
    at_gs_es.info["dft_optical_forces_em"] = f_gs_es

    loaded = load_dft_optical_forces(at_es.copy(), at_gs_es, str(ws), force_state="emission")
    assert loaded is not None and np.allclose(loaded, f_gs_es)

    # Test load_dft_optical_forces reading qe_gs_on_es/pw.out - qe_es_relax/pw.out
    (ws / "qe_gs_on_es").mkdir()
    (ws / "qe_es_relax").mkdir()
    pw_gs_on_es = ws / "qe_gs_on_es" / "pw.out"
    pw_es_relax = ws / "qe_es_relax" / "pw.out"
    pw_gs_on_es.write_text("stub")
    pw_es_relax.write_text("stub")

    at_a = at_es.copy()
    at_b = at_es.copy()
    from ase.calculators.singlepoint import SinglePointCalculator

    f_a = np.zeros((len(at_es), 3))
    f_a[0, 0] = 1.25
    f_a[1, 0] = -1.25
    at_a.calc = SinglePointCalculator(at_a, energy=-10.0, forces=f_a)
    f_b = np.zeros((len(at_es), 3))
    f_b[0, 0] = 0.05
    f_b[1, 0] = -0.05
    at_b.calc = SinglePointCalculator(at_b, energy=-9.0, forces=f_b)

    at_es_clean = at_es.copy()
    with patch("hydeph.system_profile.ase.io.read", side_effect=lambda p, **kw: at_a if "qe_gs_on_es/pw.out" in str(p) else at_b):
        df_em = load_dft_optical_forces(at_es.copy(), at_es_clean, str(ws), force_state="emission")
    assert df_em is not None
    assert np.allclose(df_em, f_a - f_b)

    # Verify SLURM prepare_relaxation_inputs includes [3/3] qe_gs_on_es
    runner = QERunner(cfg)
    prep = runner.prepare_relaxation_inputs(at_es)
    slurm_txt = Path(prep["slurm_script"]).read_text(encoding="utf-8")
    assert "qe_gs_on_es" in slurm_txt
    assert "[3/3] Launching Conjugate Ground-State SCF at R_ES" in slurm_txt
