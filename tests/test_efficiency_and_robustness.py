"""Tests for package-wide efficiency and robustness improvements in HyDePh."""

import os
import sys
import time
import pytest
import numpy as np
import ase.build
import ase.io
from ase import Atoms
from ase.calculators.emt import EMT
from unittest.mock import patch

from hydeph.config import ToolkitConfig
from hydeph.defect_builder import DefectBuilder, mic_displacement
from hydeph.qe_runner import QERunner
from hydeph.delta_trainer import DeltaTrainer, stratified_train_test_split
from hydeph.lineshape import LineshapeEngine, compute_quantum_lineshape
from hydeph.cc_diagram import ConfigurationCoordinateEngine


def test_pbc_minimum_image_invariance(tmp_path):
    """Wrapping an excited-state atom across a periodic cell boundary preserves Delta Q and CC curves."""
    at_gs = ase.build.bulk("Cu", "fcc", a=3.6, cubic=True).repeat((2, 2, 2))
    del at_gs[0]
    at_es = at_gs.copy()
    at_es.positions[0] += np.array([0.12, -0.08, 0.05])

    at_es_wrapped = at_es.copy()
    # Shift atom 0 by +1 full supercell lattice vector along x and -1 along y
    at_es_wrapped.positions[0] += at_gs.cell[0] - at_gs.cell[1]

    dr_unwrapped = mic_displacement(at_es.positions, at_gs.positions, at_gs.cell)
    dr_wrapped = mic_displacement(at_es_wrapped.positions, at_gs.positions, at_gs.cell)
    np.testing.assert_allclose(dr_unwrapped, dr_wrapped, atol=1e-10)

    cfg = ToolkitConfig(pristine_cell_path=str(tmp_path), workspace_dir=str(tmp_path), zpl_energy_eV=2.0)
    cc_engine = ConfigurationCoordinateEngine(cfg, EMT())
    res1 = cc_engine.calculate_cc_curve(at_gs, at_es, n_points=11)
    res2 = cc_engine.calculate_cc_curve(at_gs, at_es_wrapped, n_points=11)

    assert pytest.approx(res1["dq_total"], rel=1e-7) == res2["dq_total"]
    np.testing.assert_allclose(res1["e_gs"], res2["e_gs"], atol=1e-8)


def test_substitutional_and_site_index_defect_builder(tmp_path):
    """DefectBuilder supports substitutional defects and explicit defect_site_index."""
    uc = ase.build.bulk("LiCl", "rocksalt", a=5.14, cubic=True)
    uc_path = tmp_path / "licl.xyz"
    ase.io.write(str(uc_path), uc)

    cfg = ToolkitConfig(
        pristine_cell_path=str(uc_path),
        workspace_dir=str(tmp_path / "ws_sub"),
        supercell_repeat=(1, 1, 1),
        defect_type="substitutional",
        defect_site_element="Cl->F",
        defect_site_index=1,
    )
    builder = DefectBuilder(cfg)
    pristine_sc, defect_sc = builder.build_supercell_and_defect()

    assert len(defect_sc) == len(pristine_sc)
    assert defect_sc[1].symbol == "F"
    assert "defect_center" in defect_sc.info


def test_hermitian_fft_quantum_lineshape_matches_direct_integral():
    """Hermitian + FFT compute_quantum_lineshape matches direct double-sided integration to high accuracy."""
    hw_grid = np.linspace(0.5, 70.0, 600)
    s_cont = 1000.0 * (
        0.18 * np.exp(-0.5 * ((hw_grid - 22.0) / 3.0) ** 2)
        + 0.12 * np.exp(-0.5 * ((hw_grid - 41.0) / 3.5) ** 2)
    )
    temp_k = 77.0
    gamma = 2.0

    t0 = time.perf_counter()
    eg, a_fft = compute_quantum_lineshape(
        hw_grid, s_cont, temp_k=temp_k, dephasing_gamma_meV=gamma,
        e_min_meV=-250.0, e_max_meV=60.0, n_e=800
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    # Reference direct integration on [-6000, 6000] fs
    hbar_ev_fs = 0.6582119514
    kb_mev_k = 8.617333262e-2
    mask = (hw_grid > 0.1) & (s_cont > 0.0)
    hw = hw_grid[mask]
    s_val = s_cont[mask] / 1000.0
    dhw = hw_grid[1] - hw_grid[0]
    n_bose = 1.0 / (np.exp(hw / (kb_mev_k * temp_k)) - 1.0)
    t_fs = np.linspace(-6000.0, 6000.0, 16385)
    dt = t_fs[1] - t_fs[0]
    wt = np.outer(hw / (1000.0 * hbar_ev_fs), t_fs)
    integrand = (n_bose[:, None] + 1.0) * np.exp(1j * wt) + n_bose[:, None] * np.exp(-1j * wt) - (2.0 * n_bose[:, None] + 1.0)
    g_t = np.sum(s_val[:, None] * integrand * dhw, axis=0)
    damping = np.exp(-((gamma / 1000.0) / hbar_ev_fs) * np.abs(t_fs) - 0.5 * (t_fs / 2000.0) ** 2)
    w_out = (eg / 1000.0) / hbar_ev_fs
    a_ref = np.maximum(0.0, np.real(np.exp(1j * np.outer(w_out, t_fs)) @ (np.exp(g_t) * damping)) * dt / (2.0 * np.pi * hbar_ev_fs))
    trapz_fn = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    a_ref /= trapz_fn(a_ref, eg)

    rel_l1 = trapz_fn(np.abs(a_fft - a_ref), eg)
    assert rel_l1 < 2e-3
    assert elapsed_ms < 250.0


def test_stale_vibrations_cache_invalidation(tmp_path):
    """LineshapeEngine invalidates stale vib/ cache files when the underlying potential changes."""
    from ase.calculators.calculator import Calculator, all_changes

    class SpringCalc(Calculator):
        implemented_properties = ["energy", "forces"]

        def __init__(self, k_spring, eq_pos):
            super().__init__()
            self.k_spring = k_spring
            self.eq_pos = eq_pos.copy()

        def calculate(self, atoms=None, properties=["energy", "forces"], system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            disp = atoms.positions - self.eq_pos
            self.results["energy"] = 0.5 * self.k_spring * float(np.sum(disp ** 2))
            self.results["forces"] = -self.k_spring * disp

    at_gs = Atoms("Cu2", positions=[[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]], cell=[8.0, 8.0, 8.0], pbc=True)
    at_es = at_gs.copy()
    at_es.positions[0, 0] += 0.10

    cfg = ToolkitConfig(
        pristine_cell_path=str(tmp_path),
        workspace_dir=str(tmp_path / "ws_vib"),
        temperatures=[10.0],
        enable_anharmonic=False,
        plot_configuration_coordinate=False,
    )

    engine1 = LineshapeEngine(cfg, SpringCalc(k_spring=1.0, eq_pos=at_gs.positions))
    res1 = engine1.calculate_spectral_density_and_lineshapes(at_gs, at_es)

    # Stiffer spring constant in the same workspace must trigger vib.clean() and produce a different S
    engine2 = LineshapeEngine(cfg, SpringCalc(k_spring=4.0, eq_pos=at_gs.positions))
    res2 = engine2.calculate_spectral_density_and_lineshapes(at_gs, at_es)

    # S_k scales as omega_k ~ sqrt(k_spring), so doubling sqrt(k) doubles total_S
    assert pytest.approx(res2["total_S"], rel=1e-2) == 2.0 * res1["total_S"]


def test_stratified_split_and_training_manifest_cache(tmp_path):
    """Anchor FC endpoints (lambda=0, 1) stay in train.xyz and unchanged datasets reuse nep_delta.txt."""
    frames = []
    for i, (stype, lam) in enumerate([
        ("franck_condon", 0.0),
        ("franck_condon", 1.0),
        ("franck_condon", 0.5),
        ("core_rattle", None),
        ("core_rattle", None),
        ("foundation_md_fps", None),
        ("foundation_md_fps", None),
    ]):
        at = Atoms("LiCl", positions=[[0.0, 0.0, 0.0], [2.5 + 0.02 * i, 2.5, 2.5]], cell=[5.14, 5.14, 5.14], pbc=True)
        at.info["sample_type"] = stype
        at.info["frame_id"] = f"{stype}_{i:02d}"
        if lam is not None:
            at.info["lambda"] = lam
        at.info["energy"] = -1.0 - 0.1 * i
        at.arrays["forces"] = np.full((2, 3), 0.01 * i)
        frames.append(at)

    train_f, test_f = stratified_train_test_split(frames, test_fraction=0.30, seed=7)
    train_ids = {f.info["frame_id"] for f in train_f}
    assert "franck_condon_00" in train_ids
    assert "franck_condon_01" in train_ids

    mock_nep = tmp_path / "mock_nep.py"
    mock_nep.write_text(
        f"#!{sys.executable}\n"
        "with open('nep.txt', 'w') as f:\n"
        "    f.write('nep4_mock_potential\\n')\n"
    )
    os.chmod(mock_nep, 0o755)

    cfg = ToolkitConfig(
        pristine_cell_path=str(tmp_path),
        workspace_dir=str(tmp_path / "ws_train"),
        nep_exe=str(mock_nep),
    )
    trainer = DeltaTrainer(cfg)
    with patch("subprocess.run", wraps=__import__("subprocess").run) as spy_run:
        pot1 = trainer.train_delta_nep(frames)
        assert spy_run.call_count == 1
        # Second call with identical frames & hyperparameters reuses cached nep_delta.txt
        pot2 = trainer.train_delta_nep(frames)
        assert spy_run.call_count == 1
        assert pot1 == pot2


def test_qe_runner_parallel_workers_and_timeout(tmp_path):
    """QERunner supports concurrent workers (n_workers_qe=2) and recovers from hung calculations via qe_timeout_s."""
    ws = tmp_path / "ws_parallel"
    ws.mkdir()
    pseudo_dir = tmp_path / "pseudo"
    pseudo_dir.mkdir()
    (pseudo_dir / "Li.UPF").write_text("UPF")
    (pseudo_dir / "Cl.UPF").write_text("UPF")

    mock_pw = tmp_path / "mock_pw_timeout.py"
    mock_pw.write_text(
        f"#!{sys.executable}\n"
        "import os, time\n"
        "cwd = os.path.basename(os.getcwd())\n"
        "with open('pw.in') as f:\n"
        "    pw_in = f.read()\n"
        "is_retry = 'electron_maxstep = 150' in pw_in or 'local-TF' in pw_in\n"
        "if 'rattle_00' in cwd and not is_retry:\n"
        "    time.sleep(2.0)\n"
        "print('!    total energy              =     -48.00000000 Ry')\n"
        "print('     Forces acting on atoms (cartesian axes, Ry/au):')\n"
        "for i in range(1, 3):\n"
        "    print(f'     atom {i:4d} type 1   force =     0.00100000    0.00100000    0.00100000')\n"
        "print('     total force = 0.002000')\n"
        "print('     JOB DONE.')\n"
    )
    os.chmod(mock_pw, 0o755)

    frames = []
    for idx, (fid, stype) in enumerate([
        ("fc_00", "franck_condon"),
        ("rattle_00", "core_rattle"),
        ("md_fps_00", "foundation_md_fps"),
    ]):
        at = Atoms("LiCl", positions=[[0, 0, 0], [2.5 + 0.03 * idx, 2.5, 2.5]], cell=[5.14, 5.14, 5.14], pbc=True)
        at.info["frame_id"] = fid
        at.info["sample_type"] = stype
        at.info["candidate_index"] = idx
        frames.append(at)

    cfg = ToolkitConfig(
        pristine_cell_path=str(tmp_path),
        workspace_dir=str(ws),
        pseudo_dir=str(pseudo_dir),
        pseudopotentials={"Li": "Li.UPF", "Cl": "Cl.UPF"},
        pw_exe=str(mock_pw),
        n_workers_qe=2,
        qe_timeout_s=0.25,
        retry_failed_dft=True,
    )
    runner = QERunner(cfg, n_procs=1)
    labeled = runner.label_all_frames(frames)

    assert len(labeled) == 3
    by_id = {f.info["frame_id"]: f for f in labeled}
    assert by_id["rattle_00"].info["qe_status"] == "converged_on_retry"
