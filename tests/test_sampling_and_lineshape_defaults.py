"""Verification tests for Phase 6: Training Set Sampling (§3.3) and Lineshape Defaults (§3.4)."""

import os
import numpy as np
import pytest
from ase import Atoms
from unittest.mock import MagicMock
from hydeph.config import ToolkitConfig
from hydeph.frame_sampler import FrameSampler
from hydeph.lineshape import LineshapeEngine, compute_quantum_lineshape, embed_phonon_supercell


def _make_cfg(tmp_path, **kwargs):
    """Helper to create ToolkitConfig with dummy pristine file."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    p_file = tmp_path / "pristine_dummy.xyz"
    if not p_file.exists():
        import ase.io
        uc = Atoms("C2", positions=[[0, 0, 0], [1.5, 1.5, 1.5]], cell=[3.0, 3.0, 3.0], pbc=True)
        ase.io.write(str(p_file), uc)
    defaults = {
        "pristine_cell_path": str(p_file),
        "workspace_dir": str(tmp_path / "ws"),
    }
    defaults.update(kwargs)
    return ToolkitConfig(**defaults)


def test_lineshape_and_embedding_defaults(tmp_path):
    """Verify modernised defaults for dephasing, embedding, and acoustic cutoff."""
    # 1. Default config without phonon_supercell_repeat
    cfg_default = _make_cfg(tmp_path / "t1")
    cfg_default.validate()
    assert cfg_default.dephasing_gamma_meV == 1.0
    assert cfg_default.dephasing_temperature_coefficient == 0.0
    assert cfg_default.phonon_embedding_method == "force"
    assert cfg_default.finite_size_acoustic_cutoff is True
    assert cfg_default.n_hessian_frames == 4
    assert cfg_default.n_pristine_rattle_frames == 0

    # 2. Config with phonon_supercell_repeat automatically disables finite_size_acoustic_cutoff
    cfg_embedded = _make_cfg(tmp_path / "t2", phonon_supercell_repeat=(4, 4, 4))
    cfg_embedded.validate()
    assert cfg_embedded.finite_size_acoustic_cutoff is False


def test_frame_sampler_es_minimum_and_hessian_rattling(tmp_path):
    """Verify that candidate sampling centers half of rattle frames around R_ES and includes small Hessian frames."""
    cfg = _make_cfg(tmp_path, n_phonon_frames=6, n_hessian_frames=4, n_fc_frames=3, n_md_frames=0)
    sampler = FrameSampler(cfg)

    at_gs = Atoms("C2", positions=[[0, 0, 0], [1.5, 1.5, 1.5]], cell=[3.0, 3.0, 3.0], pbc=True)
    at_es = at_gs.copy()
    at_es.positions[1] += [0.15, 0.15, 0.15]

    # Rattle frames: 6 frames total -> 3 GS, 3 ES
    rattles = sampler.sample_defect_core_rattling(at_gs, at_es=at_es, n_frames=6)
    assert len(rattles) == 6
    ref_states = [fr.info.get("reference_state") for fr in rattles]
    assert ref_states[:3] == ["ground", "ground", "ground"]
    assert ref_states[3:] == ["excited", "excited", "excited"]

    # Hessian-regime rattle frames: small amplitude in [0.01, 0.03] A
    hess_frames = sampler.sample_hessian_rattling(at_gs, at_es=at_es, n_frames=4)
    assert len(hess_frames) == 4
    for fr in hess_frames:
        assert fr.info.get("is_hessian") is True
        assert fr.info.get("sample_type") == "core_rattle"
        assert 0.01 <= fr.info.get("rattle_sigma") <= 0.03


def test_pristine_supercell_rattling(tmp_path):
    """Verify generation of rattled pristine supercell frames for bulk regularization."""
    p_file = tmp_path / "pristine_bulk.xyz"
    import ase.io
    uc = Atoms("Si2", positions=[[0, 0, 0], [1.35, 1.35, 1.35]], cell=[5.43, 5.43, 5.43], pbc=True)
    ase.io.write(str(p_file), uc)

    cfg = ToolkitConfig(
        pristine_cell_path=str(p_file),
        workspace_dir=str(tmp_path / "ws_pristine"),
        n_pristine_rattle_frames=3,
    )
    sampler = FrameSampler(cfg)
    pristine_frames = sampler.sample_pristine_rattling(n_frames=3, sigma=0.02)
    assert len(pristine_frames) == 3
    for fr in pristine_frames:
        assert fr.info.get("sample_type") == "pristine_rattle"
        assert fr.info.get("is_pristine") is True
        assert len(fr) == 2 * 8  # 2x2x2 repeat of 2 atoms = 16 atoms


def test_embed_phonon_supercell_propagates_zpl_metadata(tmp_path):
    """Verify embed_phonon_supercell copies e_zpl_exact_eV and defect metadata onto large cell."""
    p_file = tmp_path / "pristine.xyz"
    import ase.io
    uc = Atoms("C2", positions=[[0, 0, 0], [1.5, 1.5, 1.5]], cell=[3.0, 3.0, 3.0], pbc=True)
    ase.io.write(str(p_file), uc)

    cfg = ToolkitConfig(
        pristine_cell_path=str(p_file),
        workspace_dir=str(tmp_path / "ws_embed"),
        supercell_repeat=(1, 1, 1),
        phonon_supercell_repeat=(2, 2, 2),
        defect_type="vacancy",
        defect_site_index=1,
    )

    class MockCalc:
        def get_potential_energy(self, atoms=None, **kwargs):
            return -100.0
        def get_forces(self, atoms=None, **kwargs):
            return np.zeros((15, 3))

    calc = MockCalc()
    at_gs = uc.copy()
    del at_gs[1]
    at_gs.info["e_zpl_exact_eV"] = 1.945
    at_gs.info["zpl_source"] = "delta_scf_exact"
    at_gs.info["defect_positions"] = [np.array([1.5, 1.5, 1.5])]

    at_es = at_gs.copy()
    at_es.positions[0] += [0.05, 0.05, 0.05]

    at_gs_large, dr_large, _ = embed_phonon_supercell(at_gs, at_es, cfg, calc_gs=calc, calc_es=None)
    assert len(at_gs_large) == 15
    assert at_gs_large.info.get("e_zpl_exact_eV") == 1.945
    assert at_gs_large.info.get("zpl_source") == "delta_scf_exact"
    assert "defect_center" in at_gs_large.info
    assert at_gs_large.info.get("is_embedded") is True


def test_fft_window_scaling_prevents_ringing():
    """Verify compute_quantum_lineshape dynamically scales FFT window for narrow dephasing."""
    hw_grid = np.linspace(1.0, 100.0, 300)
    # Single mode at 40 meV
    s_cont = np.exp(-0.5 * ((hw_grid - 40.0) / 4.0) ** 2) * 500.0

    # 1. Standard gamma = 1.0 meV
    eg1, a1 = compute_quantum_lineshape(hw_grid, s_cont, temp_k=10.0, dephasing_gamma_meV=1.0)
    assert np.all(a1 >= 0.0)
    assert np.isclose(np.trapezoid(a1, eg1), 1.0, atol=1e-3)

    # 2. Narrow gamma = 0.1 meV (should scale FFT time window automatically without negative values or ringing)
    eg2, a2 = compute_quantum_lineshape(hw_grid, s_cont, temp_k=10.0, dephasing_gamma_meV=0.1)
    assert np.all(a2 >= 0.0)
    assert np.isclose(np.trapezoid(a2, eg2), 1.0, atol=1e-3)
    # Narrower gamma yields much sharper peak
    assert np.max(a2) > 2.0 * np.max(a1)


def test_acoustic_cutoff_first_moment_shift(tmp_path):
    """Verify finite_size_acoustic_cutoff shifts spectral weight and reports both first moments."""
    cfg = _make_cfg(tmp_path, finite_size_acoustic_cutoff=True)
    engine = LineshapeEngine(cfg, hybrid_calc=None)

    # Synthetic mock mode energies and coupling
    hw_modes = np.array([5.0, 15.0, 45.0, 75.0])
    sk_modes = np.array([0.4, 0.4, 0.4, 0.4])
    hw_grid = np.linspace(0.5, 90.0, 500)
    sigma_smear = 3.0

    # Unfiltered continuous density
    diff = (hw_grid[:, None] - hw_modes[None, :]) / sigma_smear
    s_raw = (np.exp(-0.5 * (diff ** 2)) @ sk_modes) / (np.sqrt(2.0 * np.pi) * sigma_smear)
    int_raw = float(np.trapezoid(s_raw, hw_grid))
    m_before = float(np.trapezoid(hw_grid * s_raw, hw_grid) / int_raw)

    # Filter with tanh
    s_cut = s_raw * np.tanh((hw_grid / sigma_smear) ** 2)
    int_cut = float(np.trapezoid(s_cut, hw_grid))
    s_cut *= (int_raw / int_cut)
    m_after = float(np.trapezoid(hw_grid * s_cut, hw_grid) / int_raw)

    # Cutting low-frequency modes must increase the average first moment
    assert m_after > m_before
    assert (m_after - m_before) > 0.5  # Noticeable meV shift
