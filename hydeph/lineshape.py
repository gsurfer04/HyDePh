"""Normal modes, electron-phonon spectral density S(hbar omega), Huang-Rhys factor S, and optical lineshapes A(E, T)."""

from __future__ import annotations

import os
import csv
import json
import time
import hashlib
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from typing import Optional, Dict, Any, Tuple, List
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ase import Atoms
from ase.vibrations import Vibrations as _AseVibrations
from scipy.signal import fftconvolve, correlate

from .config import ToolkitConfig
from .defect_builder import mic_displacement
from .calculator import HybridDefectCalculator

from tqdm import tqdm

Vibrations = _AseVibrations

logger = logging.getLogger(__name__)

# Module-level physical constants and parameters
HBAR_EV_FS = 0.6582119514             # ℏ in eV·fs
KB_EV_K = 8.617333262e-5              # Boltzmann constant in eV/K
KB_MEV_K = 8.617333262e-2             # Boltzmann constant in meV/K
MODE_COUPLING_PREFACTOR = 0.1196127   # Prefactor: THz²·Å² to meV (meV^-1 amu^-1 A^-2)
DEFAULT_THERMALIZATION_STEPS = 300    # MD thermalisation steps
FFT_TIME_POINTS = 8193                # Time grid points for Hermitian FFT
FFT_N = 131072                        # Zero-pad FFT size

# Canonical deduplicated color cycle for temperature plots
_COLOR_CYCLE = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728', '#9467bd', '#8c564b', '#e377c2']


def compute_quantum_lineshape(
    hw_grid_meV: np.ndarray,
    s_cont_per_eV: np.ndarray,
    temp_k: float,
    dephasing_gamma_meV: float = 1.0,
    time_window_tau_fs: Optional[float] = None,
    e_min_meV: Optional[float] = None,
    e_max_meV: Optional[float] = None,
    n_e: int = 1500,
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes normalized emission lineshape A(E - E_ZPL, T) via the Hermitian time-domain generating function."""
    hw_grid_meV = np.asarray(hw_grid_meV, dtype=float)
    s_cont_per_eV = np.asarray(s_cont_per_eV, dtype=float)
    mask = (hw_grid_meV > 0.1) & (s_cont_per_eV > 0.0)
    hw = hw_grid_meV[mask]
    s_val = s_cont_per_eV[mask] / 1000.0
    dhw = float(hw_grid_meV[1] - hw_grid_meV[0]) if len(hw_grid_meV) > 1 else 1.0

    if e_min_meV is None or e_max_meV is None:
        total_s = float(np.sum(s_val * dhw))
        mean_hw = float(np.sum(hw * s_val * dhw) / total_s) if total_s > 0 else 35.0
        erel = total_s * mean_hw
        sigma_fc = np.sqrt(max(1.0, total_s)) * mean_hw
        if e_min_meV is None:
            e_min_meV = -max(250.0, erel + 4.5 * sigma_fc)
        if e_max_meV is None:
            e_max_meV = max(60.0, 2.5 * mean_hw)

    energy_grid = np.linspace(e_min_meV, e_max_meV, n_e)
    if len(hw) == 0:
        return energy_grid, np.zeros_like(energy_grid)

    if temp_k < 0.1:
        n_bose = np.zeros_like(hw)
    else:
        n_bose = 1.0 / (np.exp(np.clip(hw / (KB_MEV_K * temp_k), 1e-9, 500.0)) - 1.0)

    # Scale FFT time window with 1/gamma to prevent truncation ringing for narrow lines
    gamma_eff = max(1e-4, float(dephasing_gamma_meV))
    tau_decay_fs = 1000.0 * HBAR_EV_FS / gamma_eff
    max_t_fs = float(np.clip(max(6000.0, 6.0 * tau_decay_fs), 6000.0, 100000.0))

    t_pos = np.linspace(0.0, max_t_fs, FFT_TIME_POINTS)
    dt = float(t_pos[1] - t_pos[0])

    w_fs = hw / (1000.0 * HBAR_EV_FS)
    wt = np.outer(w_fs, t_pos)
    w_re = (s_val * (2.0 * n_bose + 1.0)) * dhw
    w_im = s_val * dhw

    g_re = (w_re @ np.cos(wt)) - float(np.sum(w_re))
    g_im = w_im @ np.sin(wt)

    gamma_fs_inv = (dephasing_gamma_meV / 1000.0) / HBAR_EV_FS
    if time_window_tau_fs is not None and time_window_tau_fs > 0:
        damping = np.exp(-gamma_fs_inv * t_pos - 0.5 * (t_pos / time_window_tau_fs) ** 2)
    else:
        damping = np.exp(-gamma_fs_inv * t_pos - 0.5 * (t_pos / max_t_fs) ** 2)
    G_pos = np.exp(g_re + 1j * g_im) * damping

    # Zero-padded 1D FFT evaluation of 2 * Re \int_0^T dt G(t) exp(i * w_out * t)
    n_fft = FFT_N
    h_pos = G_pos.copy()
    h_pos[0] *= 0.5
    spec_fft = np.real(np.fft.ifft(h_pos, n=n_fft)) * (2.0 * n_fft * dt / (2.0 * np.pi * HBAR_EV_FS))
    freqs_rad_fs = 2.0 * np.pi * np.fft.fftfreq(n_fft, d=dt)
    order = np.argsort(freqs_rad_fs)
    e_fft_meV = freqs_rad_fs[order] * (1000.0 * HBAR_EV_FS)
    spec_sorted = spec_fft[order]

    a_unnorm = np.interp(energy_grid, e_fft_meV, spec_sorted, left=0.0, right=0.0)
    a_unnorm = np.maximum(0.0, a_unnorm)

    trapz_fn = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    total_area = float(trapz_fn(a_unnorm, energy_grid))
    a_norm = a_unnorm / total_area if total_area > 0 else a_unnorm
    return energy_grid, a_norm


def compute_photoluminescence_spectrum(
    e_rel_meV: np.ndarray,
    a_norm: np.ndarray,
    e_zpl_eV: float,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Computes normalized photoluminescence spectrum I_PL(hbar omega) \propto (hbar omega)^3 * A(hbar omega - E_ZPL).

    Parameters
    ----------
    e_rel_meV : np.ndarray
        Energy grid relative to ZPL (meV).
    a_norm : np.ndarray
        Normalized lineshape A(E - E_ZPL).
    e_zpl_eV : float
        Zero-Phonon Line energy in eV.

    Returns
    -------
    photon_energy_eV : np.ndarray
        Absolute photon energy grid (eV).
    i_pl_norm : np.ndarray
        Normalized luminescence intensity (1/eV).
    """
    trapz_fn = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    photon_energy_eV = e_zpl_eV + (np.asarray(e_rel_meV, dtype=float) / 1000.0)
    mask = photon_energy_eV > 0.0
    i_pl = np.zeros_like(photon_energy_eV)
    if np.any(mask):
        i_pl[mask] = (photon_energy_eV[mask] ** 3) * a_norm[mask]
        area = float(trapz_fn(i_pl, photon_energy_eV))
        if area > 0:
            i_pl = i_pl / area
    return photon_energy_eV, i_pl


_ANHARM_WORKER_STATE: Dict[str, Any] = {}


def _build_worker_calculator(spec: Dict[str, Any], shared_base: Optional[Any] = None) -> Any:
    """Reconstructs a file-backed HybridDefectCalculator inside a worker process."""
    return HybridDefectCalculator(
        foundation_pot_path=spec["foundation_pot_path"],
        delta_pot_paths=spec["delta_pot_paths"],
        calc_base=shared_base,
        energy_offset_per_atom=spec["energy_offset_per_atom"],
    )


def _init_anharmonic_worker(worker_spec: Dict[str, Any]) -> None:
    """Initializes single-threaded NEP calculators once per anharmonic MD worker process."""
    for env_var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[env_var] = "1"

    from .calculator import PairedDefectCalculator

    gs_spec = worker_spec["gs"]
    es_spec = worker_spec.get("es")
    calc_gs = _build_worker_calculator(gs_spec)
    calc_es = None
    paired_calc = None
    if es_spec is not None:
        same_base = (
            gs_spec["foundation_pot_path"] is not None
            and es_spec["foundation_pot_path"] is not None
            and os.path.abspath(gs_spec["foundation_pot_path"])
            == os.path.abspath(es_spec["foundation_pot_path"])
        )
        calc_es = _build_worker_calculator(
            es_spec, shared_base=calc_gs.calc_base if same_base else None
        )
        paired_calc = PairedDefectCalculator(calc_gs, calc_es)

    _ANHARM_WORKER_STATE.clear()
    _ANHARM_WORKER_STATE.update(
        {
            "calc_gs": calc_gs,
            "calc_es": calc_es,
            "paired_calc": paired_calc,
        }
    )


def _is_real_cpunep(calc_obj: Any) -> bool:
    """Returns True only for an actual calorine CPUNEP instance (never test mocks)."""
    return (
        calc_obj is not None
        and type(calc_obj).__name__ == "CPUNEP"
        and type(calc_obj).__module__.startswith("calorine")
    )


def _extract_hybrid_calc_spec(calc: Any) -> Optional[Dict[str, Any]]:
    """Returns a serializable file-backed spec for HybridDefectCalculator, or None."""
    if not isinstance(calc, HybridDefectCalculator):
        return None
    calc_deltas = list(getattr(calc, "calc_deltas", []) or [])
    if not calc_deltas or not all(_is_real_cpunep(d) for d in calc_deltas):
        return None
    delta_paths = list(getattr(calc, "delta_pot_paths", []) or [])
    if len(delta_paths) != len(calc_deltas) or not all(
        isinstance(p, str) and os.path.isfile(p) for p in delta_paths
    ):
        return None
    f_path = getattr(calc, "foundation_pot_path", None)
    calc_base = getattr(calc, "calc_base", None)
    if calc_base is not None:
        if not _is_real_cpunep(calc_base) or not (isinstance(f_path, str) and os.path.isfile(f_path)):
            return None
    return {
        "foundation_pot_path": f_path,
        "delta_pot_paths": delta_paths,
        "energy_offset_per_atom": float(getattr(calc, "energy_offset_per_atom", 0.0)),
    }


def _run_single_anharmonic_trajectory(
    at_start: Atoms,
    T: float,
    dt_fs: float,
    n_therm: int,
    n_steps: int,
    u_opt: np.ndarray,
    dr_fc: np.ndarray,
    calc_gs: Any,
    calc_es: Optional[Any],
    paired_calc: Optional[Any],
    seed: Optional[int] = None,
    show_progress: bool = True,
    desc_str: str = "Anharmonic MD",
    f_core: Optional[np.ndarray] = None,
    r_ref: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Runs NVT Langevin thermalization followed by NVE VelocityVerlet production."""
    from ase import units
    from ase.md.langevin import Langevin
    from ase.md.verlet import VelocityVerlet
    from ase.md.velocitydistribution import Stationary, ZeroRotation

    has_paired = calc_es is not None
    use_core_force_gap = f_core is not None and r_ref is not None and len(f_core) == len(at_start)
    core_idx = np.flatnonzero(np.linalg.norm(f_core, axis=1) > 1e-12) if use_core_force_gap else np.array([], dtype=int)
    use_core_force_gap = use_core_force_gap and len(core_idx) > 0
    f_core_nz = np.asarray(f_core[core_idx], dtype=float) if use_core_force_gap else None
    r_ref_nz = np.asarray(r_ref[core_idx], dtype=float) if use_core_force_gap else None

    at = at_start.copy()
    at.calc = calc_es if has_paired else calc_gs
    at_disp = None
    if not has_paired and not use_core_force_gap:
        at_disp = at_start.copy()
        at_disp.calc = calc_gs

    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()

    t_equil_nvt = float(T)
    if n_therm > 0:
        try:
            from ase.md.velocitydistribution import thermalize_momenta

            thermalize_momenta(at, temperature_K=T, rng=rng)
        except (ImportError, TypeError):
            from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

            MaxwellBoltzmannDistribution(at, temperature_K=T, rng=rng)
        Stationary(at)
        if not np.any(at.pbc):
            ZeroRotation(at)
        dyn_therm = Langevin(
            at,
            dt_fs * units.fs,
            temperature_K=T,
            friction=0.01 / units.fs,
            fixcm=False,
            rng=rng,
            logfile=None,
        )
        t_therm_samples: List[float] = []
        dyn_therm.attach(
            lambda: t_therm_samples.append(float(at.get_temperature())),
            interval=max(1, n_therm // 10),
        )
        dyn_therm.run(n_therm)
        if t_therm_samples:
            t_equil_nvt = float(np.mean(t_therm_samples[-5:]))

    Stationary(at)
    if not np.any(at.pbc):
        ZeroRotation(at)
    if n_therm > 0 and use_core_force_gap:
        # Gentle thermal coupling keeps localized core modes canonically equilibrated across multi-ps MD.
        dyn = Langevin(
            at,
            dt_fs * units.fs,
            temperature_K=T,
            friction=0.0025 / units.fs,
            fixcm=False,
            rng=rng,
            logfile=None,
        )
    else:
        dyn = VelocityVerlet(at, dt_fs * units.fs, logfile=None)

    du_list: List[float] = []
    v_opt_list: List[float] = []
    t_nve_list: List[float] = []

    def step_observer() -> None:
        v = at.get_velocities()
        v_opt_list.append(float(np.sum(v * u_opt)))
        t_nve_list.append(float(at.get_temperature()))

        if use_core_force_gap:
            disp = mic_displacement(at.positions[core_idx], r_ref_nz, at.cell)
            du = float(-np.sum(f_core_nz * disp) * 1000.0)
        elif has_paired:
            if paired_calc is not None:
                du = float(paired_calc.get_energy_gap(at) * 1000.0)
            else:
                ue = at.get_potential_energy()
                at_gs_eval = at.copy()
                at_gs_eval.calc = calc_gs
                ug = at_gs_eval.get_potential_energy()
                du = float((ue - ug) * 1000.0)
        else:
            ug = at.get_potential_energy()
            at_disp.positions = at.positions + dr_fc
            at_disp.cell = at.cell
            ue = at_disp.get_potential_energy()
            du = float((ue - ug) * 1000.0)
        du_list.append(du)

    dyn.attach(step_observer, interval=2)
    if show_progress:
        pbar = tqdm(total=n_steps, desc=desc_str, unit="step", leave=False)
        step_chunk = 50
        steps_done = 0
        while steps_done < n_steps:
            this_chunk = min(step_chunk, n_steps - steps_done)
            dyn.run(this_chunk)
            steps_done += this_chunk
            pbar.update(this_chunk)
        pbar.close()
    else:
        dyn.run(n_steps)

    t_equil = float(np.mean(t_nve_list)) if t_nve_list else t_equil_nvt
    return np.array(du_list, dtype=float), np.array(v_opt_list, dtype=float), t_equil


def _run_anharmonic_window_worker(
    task_args: Tuple[Any, ...]
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Worker entry point executing one NVT->NVE trajectory using cached worker calculators."""
    if len(task_args) == 10:
        at_start, T, dt_fs, n_therm, n_steps, u_opt, dr_fc, seed, f_core, r_ref = task_args
    else:
        at_start, T, dt_fs, n_therm, n_steps, u_opt, dr_fc, seed = task_args
        f_core, r_ref = None, None
    return _run_single_anharmonic_trajectory(
        at_start=at_start,
        T=T,
        dt_fs=dt_fs,
        n_therm=n_therm,
        n_steps=n_steps,
        u_opt=u_opt,
        dr_fc=dr_fc,
        calc_gs=_ANHARM_WORKER_STATE["calc_gs"],
        calc_es=_ANHARM_WORKER_STATE.get("calc_es"),
        paired_calc=_ANHARM_WORKER_STATE.get("paired_calc"),
        seed=seed,
        show_progress=False,
        f_core=f_core,
        r_ref=r_ref,
    )


def _wigner_seitz_radius(cell: Any) -> float:
    """Returns the inscribed Wigner-Seitz sphere radius 0.5 * min_{n!=0} ||n @ cell||."""
    cell_arr = np.asarray(cell, dtype=float)
    if cell_arr.shape != (3, 3):
        return float("inf")
    combos = np.array(
        [
            [i, j, k]
            for i in (-1, 0, 1)
            for j in (-1, 0, 1)
            for k in (-1, 0, 1)
            if (i, j, k) != (0, 0, 0)
        ],
        dtype=float,
    )
    norms = np.linalg.norm(combos @ cell_arr, axis=1)
    pos_norms = norms[norms > 1e-8]
    return 0.5 * float(np.min(pos_norms)) if len(pos_norms) > 0 else float("inf")


def _smear_harmonic_spectral_density(
    hw_grid: np.ndarray,
    hw_modes: np.ndarray,
    sk_modes: np.ndarray,
    sigma_smear: float,
    variable_dispersion: bool = False,
) -> np.ndarray:
    """Broadens discrete Huang-Rhys sticks S_k using constant or linear frequency-dependent sigma."""
    if len(hw_modes) == 0:
        return np.zeros_like(hw_grid)
    if variable_dispersion:
        max_hw = max(1e-6, float(np.max(hw_modes)))
        sigma_low = float(np.clip(1.8 * sigma_smear, 3.2, 6.0))
        sigma_high = float(np.clip(0.78 * sigma_smear, 1.2, 3.0))
        sigmas = sigma_low + (sigma_high - sigma_low) * (np.asarray(hw_modes, dtype=float) / max_hw)
        sigmas = sigmas[None, :]
    else:
        sigmas = float(sigma_smear)
    diff = (hw_grid[:, None] - hw_modes[None, :]) / sigmas
    gauss = np.exp(-0.5 * (diff ** 2)) / (np.sqrt(2.0 * np.pi) * sigmas)
    return gauss @ sk_modes


def embed_phonon_supercell(
    at_gs: Atoms,
    at_es: Atoms,
    config: ToolkitConfig,
    calc_gs,
    calc_es=None,
) -> Tuple[Atoms, np.ndarray, Optional[np.ndarray]]:
    """Embeds a relaxed defect from the DFT supercell into a larger pristine supercell for phonon calculations.

    Parameters
    ----------
    at_gs : Atoms
        Ground-state defect supercell.
    at_es : Atoms
        Excited-state defect supercell.
    config : ToolkitConfig
        Toolkit configuration containing phonon_supercell_repeat and phonon_embedding_method.
    calc_gs : Calculator
        Hybrid defect calculator for ground state.
    calc_es : Calculator, optional
        Hybrid defect calculator for excited state.

    Returns
    -------
    at_gs_large : Atoms
        Relaxed ground state in the large supercell.
    dr_large : np.ndarray
        Optical displacement field Delta R mapped to the large supercell.
    f_es_large : Optional[np.ndarray]
        Excited-state forces evaluated on at_gs_large (if calc_es provided and force method used).
    """
    import copy
    from ase.optimize import BFGS, LBFGS
    import ase.io
    from .defect_builder import DefectBuilder, detect_defect_sites

    rep_large = config.phonon_supercell_repeat
    logger.info(f"\n[-] Embedding defect into large phonon supercell (repeat={rep_large})...")

    pristine_uc = ase.io.read(config.pristine_cell_path)
    pristine_large = pristine_uc.repeat(rep_large)

    pristine_small = pristine_uc.repeat(config.supercell_repeat)
    defect_info = detect_defect_sites(pristine_small, at_gs)
    c_small = defect_info["centroid"]

    if config.defect_cell_path and os.path.isfile(config.defect_cell_path):
        defect_input = ase.io.read(config.defect_cell_path)
        v_input = defect_input.get_volume()
        v_uc = pristine_uc.get_volume()
        if abs(v_input - v_uc) / v_uc < 0.25:
            temp_builder = DefectBuilder(config, calc_foundation=calc_gs)
            at_gs_large, detected_pos = temp_builder.embed_defect_unitcell(pristine_uc, defect_input, rep_large)
            c_large = np.asarray(detected_pos, dtype=float)
        else:
            at_gs_large = pristine_large.copy()
            c_large = at_gs_large.get_cell().sum(axis=0) / 2.0
            _apply_defect_sites_to_supercell(pristine_small, at_gs, at_gs_large, c_small, c_large)
    else:
        temp_cfg = copy.copy(config)
        temp_cfg.supercell_repeat = rep_large
        temp_builder = DefectBuilder(temp_cfg, calc_foundation=calc_gs)
        _, at_gs_large = temp_builder.build_supercell_and_defect()
        c_large = np.asarray(
            at_gs_large.info.get("defect_center", at_gs_large.get_cell().sum(axis=0) / 2.0),
            dtype=float,
        )

    logger.info(f"    [-] Relaxing large phonon supercell ({len(at_gs_large)} atoms) on hybrid potential...")
    at_gs_large.calc = calc_gs
    opt = LBFGS(at_gs_large, logfile=None) if len(at_gs_large) > 400 else BFGS(at_gs_large, logfile=None)
    fmax_thr = getattr(config, "relax_forc_conv_thr", 0.01)
    opt.run(fmax=fmax_thr, steps=100)
    e_large = at_gs_large.get_potential_energy()
    f_large_max = float(np.abs(at_gs_large.get_forces()).max())
    logger.info(f"    [+] Large phonon supercell relaxed: E = {e_large:.4f} eV, max |F| = {f_large_max:.4f} eV/A")

    dr_small = mic_displacement(at_es.positions, at_gs.positions, at_gs.cell)
    dr_large = np.zeros_like(at_gs_large.positions)
    core_mask = np.zeros(len(at_gs_large), dtype=bool)
    r_from_center = np.full(len(at_gs_large), float("inf"), dtype=float)
    ri_rels = mic_displacement(at_gs.positions, c_small, at_gs.cell)

    for j, at_j in enumerate(at_gs_large):
        rj_rel = mic_displacement([at_j.position], c_large, at_gs_large.cell)[0]
        r_from_center[j] = float(np.linalg.norm(rj_rel))
        dists = np.linalg.norm(ri_rels - rj_rel, axis=1)
        min_i = int(np.argmin(dists))
        if dists[min_i] < 0.25 and at_gs[min_i].symbol == at_j.symbol:
            dr_large[j] = dr_small[min_i]
            core_mask[j] = True

    dq_fc_large = float(np.sqrt(np.sum(at_gs_large.get_masses()[:, None] * (dr_large ** 2))))
    logger.info(
        f"    [+] Embedded optical displacement field: Delta Q = {dq_fc_large:.4f} amu^(1/2)*A "
        f"({np.count_nonzero(np.linalg.norm(dr_large, axis=1) > 1e-4)} active atoms)"
    )

    f_es_large = None
    if calc_es is not None and getattr(config, "phonon_embedding_method", "force").lower() == "force":
        logger.info("    [-] Evaluating excited-state forces on relaxed ground-state large supercell (Alkauskas/Razinkovas)...")
        at_gs_large.calc = calc_gs
        f_gs_large = at_gs_large.get_forces().copy()
        at_gs_large.calc = calc_es
        f_es_large = at_gs_large.get_forces().copy() - f_gs_large
        at_gs_large.calc = calc_gs
        if len(at_gs_large) > len(at_gs):
            force_mask = core_mask
            r_ws = _wigner_seitz_radius(at_gs.cell) if len(at_gs) > 8 else float("inf")
            if np.isfinite(r_ws):
                # Exclude DFT periodic-boundary corners outside the Wigner-Seitz inscribed sphere.
                ws_mask = core_mask & (r_from_center <= r_ws)
                if int(np.count_nonzero(ws_mask)) >= 4:
                    force_mask = ws_mask
            f_es_large[~force_mask] = 0.0
            n_core = int(np.count_nonzero(force_mask))
            if n_core > 1 and float(np.max(np.ptp(f_es_large[force_mask], axis=0))) > 1e-6:
                f_es_large[force_mask] -= np.mean(f_es_large[force_mask], axis=0)
            ws_note = f" within R_WS = {r_ws:.2f} A" if np.isfinite(r_ws) else ""
            logger.info(
                f"    [+] Masked ES-GS force difference to {n_core} DFT core atoms{ws_note} "
                f"(zeroed {int(np.count_nonzero(~force_mask))} outer/bulk atoms)."
            )

    # Propagate DFT energetics, ZPL metadata, and defect coordinates onto embedded cell
    for k in (
        "e_zpl_exact_eV",
        "zpl_source",
        "energy",
        "defect_positions",
        "spin_multiplicity",
        "delta_scf_spec",
    ):
        if k in at_gs.info and k not in at_gs_large.info:
            at_gs_large.info[k] = at_gs.info[k]
    at_gs_large.info["defect_center"] = c_large
    at_gs_large.info["is_embedded"] = True
    at_gs_large.info["dft_supercell_natoms"] = len(at_gs)

    return at_gs_large, dr_large, f_es_large


def _apply_defect_sites_to_supercell(pristine_small, at_gs, at_large, c_small, c_large):
    """Applies vacancies and substitutions from small cell to large supercell."""
    from .defect_builder import detect_defect_sites, mic_distances
    res = detect_defect_sites(pristine_small, at_gs)
    del_indices = []
    for v in res["vacancies"]:
        r_rel = mic_displacement([v["position"]], c_small, pristine_small.cell)[0]
        target_pos = c_large + r_rel
        dists = mic_distances(at_large.positions, target_pos, at_large.cell)
        min_idx = int(np.argmin(dists))
        if dists[min_idx] < 0.3:
            del_indices.append(min_idx)
    for idx in sorted(del_indices, reverse=True):
        del at_large[idx]
    for s in res["substitutions"]:
        r_rel = mic_displacement([s["position"]], c_small, pristine_small.cell)[0]
        target_pos = c_large + r_rel
        dists = mic_distances(at_large.positions, target_pos, at_large.cell)
        min_idx = int(np.argmin(dists))
        if dists[min_idx] < 0.3:
            at_large[min_idx].symbol = s["to_symbol"]


def _compute_hessian_fingerprint(atoms, calc, delta=0.01) -> str:
    """Computes a SHA256 fingerprint of geometry and calculator forces to detect stale Vibrations caches."""
    h = hashlib.sha256()
    h.update(f"delta={delta};nat={len(atoms)}".encode("utf-8"))
    h.update(np.asarray(atoms.positions, dtype=np.float64).tobytes())
    h.update(np.asarray(atoms.cell, dtype=np.float64).tobytes())
    h.update(",".join(atoms.get_chemical_symbols()).encode("utf-8"))

    try:
        at_probe = atoms.copy()
        if len(at_probe) > 0:
            at_probe.positions[0, 0] += float(delta)
        at_probe.calc = calc
        e_val = float(at_probe.get_potential_energy())
        f_val = np.asarray(at_probe.get_forces(), dtype=np.float64)
        h.update(np.float64(e_val).tobytes())
        h.update(f_val.tobytes())
    except Exception:
        pass
    return h.hexdigest()


def apply_acoustic_sum_rule_and_projection(H_2d: np.ndarray, atoms: Atoms) -> np.ndarray:
    """Enforces Acoustic Sum Rule (ASR) translational invariance and projects zero modes.

    1. Symmetrizes the 4D Cartesian force constant tensor Phi_ij^{a,b}.
    2. Corrects diagonal on-site blocks Phi_ii^{a,b} = - sum_{j != i} Phi_ij^{a,b} so row sums vanish.
    3. Projects out the 3 rigid center-of-mass translation eigenvectors from the mass-weighted Hessian,
       guaranteeing exact zero frequencies for acoustic translations and suppressing spurious acoustic drift.

    Parameters
    ----------
    H_2d : np.ndarray
        2D Hessian matrix of shape (3N, 3N) in eV / Angstrom^2.
    atoms : Atoms
        Atomic geometry with masses.

    Returns
    -------
    np.ndarray
        Cleaned, ASR-symmetrized, and projection-filtered 2D Hessian of shape (3N, 3N).
    """
    natoms = len(atoms)
    masses = atoms.get_masses()
    if not np.all(masses > 0):
        return H_2d

    H_4d = H_2d.reshape(natoms, 3, natoms, 3).copy()

    # Hermitian symmetrization in 4D: Phi_ij^{a,b} = Phi_ji^{b,a}
    H_4d = 0.5 * (H_4d + H_4d.transpose(2, 3, 0, 1))

    # Tensorial Acoustic Sum Rule (ASR): sum_j Phi_ij^{a,b} = 0
    row_sums = np.sum(H_4d, axis=2)
    for i in range(natoms):
        H_4d[i, :, i, :] -= row_sums[i]
    H_4d = 0.5 * (H_4d + H_4d.transpose(2, 3, 0, 1))
    H_clean = H_4d.reshape(3 * natoms, 3 * natoms)

    # Rigid translation projection on mass-weighted dynamical matrix
    m_inv_sqrt = 1.0 / np.sqrt(np.repeat(masses, 3))
    m_sqrt = np.sqrt(np.repeat(masses, 3))
    m_tot = float(np.sum(masses))

    D = H_clean * np.outer(m_inv_sqrt, m_inv_sqrt)

    T = np.zeros((3, 3 * natoms))
    for ax in range(3):
        for i in range(natoms):
            T[ax, 3 * i + ax] = np.sqrt(masses[i] / m_tot)

    P = np.eye(3 * natoms) - T.T @ T
    D_proj = P @ D @ P
    D_proj = 0.5 * (D_proj + D_proj.T)

    return D_proj * np.outer(m_sqrt, m_sqrt)


class LineshapeEngine:
    """Calculates phonon normal modes, spectral densities S(hw), Huang-Rhys factor, and lineshapes."""

    def __init__(self, config: ToolkitConfig, hybrid_calc, calc_es: Optional[Any] = None, vibrations_cls: Optional[Any] = None):
        self.config = config
        self.calc = hybrid_calc
        self.calc_es = calc_es
        self.paired_calc = None
        self.vibrations_cls = vibrations_cls

    def _run_vibrations_for_calculator(
        self,
        at_gs: Atoms,
        calc: Any,
        vib_subname: str = "vib",
    ):
        """Runs finite-difference Vibrations and optional ASR projection for a given calculator."""
        vib_dir = os.path.join(self.config.workspace_dir, "vib")
        os.makedirs(vib_dir, exist_ok=True)
        vib_path = os.path.join(vib_dir, vib_subname)
        state_file = f"{vib_path}_state.json"

        at = at_gs.copy()
        at.calc = calc

        use_parallel = getattr(self.config, "parallel_hessian", True)
        n_workers_vib = getattr(self.config, "parallel_hessian_workers", None) or 4
        delta_vib = getattr(self.config, "displacement_delta", 0.01)
        vib_cls = self.vibrations_cls if self.vibrations_cls is not None else Vibrations
        if use_parallel and n_workers_vib > 1 and vib_cls is _AseVibrations and Vibrations is _AseVibrations:
            from .hpc.parallel_vib import ParallelVibrations
            vib = ParallelVibrations(at, name=vib_path, delta=delta_vib, n_workers=n_workers_vib)
        else:
            vib = vib_cls(at, name=vib_path, delta=delta_vib)

        current_fp = _compute_hessian_fingerprint(at_gs, calc, delta=delta_vib)
        cached_fp = None
        if os.path.isfile(state_file):
            try:
                with open(state_file, "r", encoding="utf-8") as f:
                    cached_fp = json.load(f).get("fingerprint")
            except Exception:
                cached_fp = None

        if cached_fp != current_fp:
            vib.clean()

        orig_calc = vib.calculate
        n_disps = 6 * len(at)
        pbar_vib = tqdm(total=n_disps, desc=f"Normal modes ({vib_subname})", unit="disp", leave=False)

        def _wrapped_calculate(atoms, disp):
            res = orig_calc(atoms, disp)
            pbar_vib.update(1)
            return res

        vib.calculate = _wrapped_calculate
        try:
            vib.run()
        finally:
            pbar_vib.close()

        try:
            with open(state_file, "w", encoding="utf-8") as f:
                json.dump({"fingerprint": current_fp}, f)
        except Exception:
            pass

        use_asr = getattr(self.config, "acoustic_sum_rule", True)
        _get_mode_func = None
        if use_asr and hasattr(vib, "get_vibrations"):
            try:
                vdata = vib.get_vibrations()
                H_raw = getattr(vdata, "get_hessian_2d", lambda: None)()
                if isinstance(H_raw, np.ndarray):
                    H_clean = apply_acoustic_sum_rule_and_projection(H_raw, at)
                    from ase.vibrations.data import VibrationsData
                    vdata_clean = VibrationsData(at, H_clean.reshape(len(at), 3, len(at), 3))
                    energies_ev = vdata_clean.get_energies()
                    all_modes_arr = vdata_clean.get_modes()

                    def _get_mode_clean(idx):
                        return all_modes_arr[idx]

                    _get_mode_func = _get_mode_clean
                else:
                    energies_ev = vib.get_energies()
                    _get_mode_func = vib.get_mode
            except Exception as e_asr:
                if not getattr(self.config, "allow_fallbacks", False):
                    raise RuntimeError(
                        f"Acoustic sum rule projection failed ({e_asr}). Set allow_fallbacks=True to proceed with raw Vibrations."
                    )
                logger.warning(
                    f"[!] Warning: Acoustic sum rule application encountered error ({e_asr}); falling back to standard Vibrations."
                )
                energies_ev = vib.get_energies()
                _get_mode_func = vib.get_mode
        else:
            energies_ev = vib.get_energies()
            _get_mode_func = vib.get_mode

        return energies_ev, _get_mode_func

    def calculate_spectral_density_and_lineshapes(self, at_gs: Atoms, at_es: Atoms) -> Dict[str, Any]:
        """Calculates normal modes, spectral density S(hw), and multi-T lineshapes."""
        logger.info("\n[-] Evaluating Vibrational Normal Modes on Hybrid Potential...")
        phonon_rep = getattr(self.config, "phonon_supercell_repeat", None)
        supercell_rep = getattr(self.config, "supercell_repeat", (2, 2, 2))
        f_es_mode = None
        is_embedded = phonon_rep is not None and tuple(phonon_rep) != tuple(supercell_rep)

        if is_embedded:
            calc_es_opt = getattr(self, "calc_es", None)
            at_gs, dr, f_es_mode = embed_phonon_supercell(
                at_gs, at_es, self.config, self.calc, calc_es=calc_es_opt
            )
            at_es = at_gs.copy()
            at_es.positions += dr
            m = at_gs.get_masses()
            dq_total = float(np.sqrt(np.sum(m[:, None] * (dr ** 2))))
        else:
            dr = mic_displacement(at_es.positions, at_gs.positions, at_gs.cell)
            m = at_gs.get_masses()
            dq_total = float(np.sqrt(np.sum(m[:, None] * (dr ** 2))))
            if (
                getattr(self.config, "phonon_embedding_method", "displacement").lower() == "force"
                and hasattr(self, "calc_es")
                and self.calc_es is not None
            ):
                at_probe = at_gs.copy()
                at_probe.calc = self.calc_es
                f_es_mode = at_probe.get_forces().copy()

        energies_ev, _get_mode_func = self._run_vibrations_for_calculator(at_gs, self.calc, "vib")

        # 1. Detect imaginary and acoustic modes
        imaginary_modes = []
        acoustic_modes = []
        for i, e in enumerate(energies_ev):
            e_im = float(np.abs(np.imag(e)) * 1000.0)
            e_re = float(np.real(e) * 1000.0)
            if e_im > 1.0:
                imaginary_modes.append((i, e_im))
            elif e_re <= 1.0:
                acoustic_modes.append(i)

        max_imag = getattr(self.config, "max_imaginary_modes", 0)
        if len(imaginary_modes) > 0:
            masses = at_gs.get_masses()
            symbols = at_gs.get_chemical_symbols()
            nat = len(at_gs)
            im_details = []
            for idx, freq in imaginary_modes:
                mode_disp = _get_mode_func(idx)
                disp_norms = np.linalg.norm(mode_disp, axis=1)
                p = (disp_norms ** 2) * masses
                tot_p = float(np.sum(p))
                p = p / tot_p if tot_p > 0 else np.ones(nat) / nat
                ipr = float(np.sum(p ** 2))
                pr = float(1.0 / (nat * ipr)) if ipr > 0 else 1.0
                top_atoms = np.argsort(disp_norms)[::-1][:2]
                top_str = "/".join([f"{symbols[a]}{a}" for a in top_atoms])
                im_details.append(f"mode {idx} ({freq:.2f}i meV, PR={pr:.2f}, top:{top_str})")

            mode_desc = ", ".join(im_details[:8])
            if len(imaginary_modes) > 8:
                mode_desc += f"... (+{len(imaginary_modes) - 8} more)"

            if len(imaginary_modes) > max_imag:
                raise RuntimeError(
                    f"Hessian calculation produced {len(imaginary_modes)} imaginary mode(s) "
                    f"(threshold: max_imaginary_modes={max_imag}).\n"
                    f"Imaginary modes breakdown: {mode_desc}\n"
                    f"DIAGNOSTIC GUIDANCE:\n"
                    f"  1. If imaginary modes are localized near defect atoms (low PR, top atoms are defect neighbors),\n"
                    f"     the ground state structure is at a saddle point and needs tighter relaxation or symmetry-breaking.\n"
                    f"  2. If imaginary modes are distributed across bulk host atoms (high PR or distant bulk atoms),\n"
                    f"     the underlying foundation potential may have negative curvature/instability on the pristine host crystal\n"
                    f"     at this cell volume (e.g. tensile strain beyond the foundation model's stability limit).\n"
                    f"     Consider setting `max_imaginary_modes={len(imaginary_modes)}` to filter them out,\n"
                    f"     using whole-supercell rattling (`rattle_mode='hybrid'`), or relaxing the pristine unit cell."
                )
            else:
                logger.warning(
                    f"[!] Notice: Filtered out {len(imaginary_modes)} imaginary mode(s) "
                    f"(within allowed threshold max_imaginary_modes={max_imag}): {mode_desc}. "
                    f"Proceeding with {len(energies_ev) - len(imaginary_modes) - len(acoustic_modes)} valid optical modes."
                )

        # 2. Collect valid optical modes
        valid_indices = [
            i for i, e in enumerate(energies_ev)
            if float(np.real(e) * 1000.0) > 1.0 and float(np.abs(np.imag(e)) * 1000.0) <= 1.0
        ]

        natoms = len(at_gs)
        defect_indices = []
        if "defect_positions" in at_gs.info:
            from .defect_builder import mic_distances
            d_pos = np.array(at_gs.info["defect_positions"])
            for p in d_pos:
                dists = mic_distances(at_gs.positions, p, at_gs.cell)
                min_i = int(np.argmin(dists))
                if dists[min_i] < 0.3:
                    defect_indices.append(min_i)

        hw_modes = []
        sk_list = []
        dqk_list = []
        ipr_list = []
        pr_list = []
        def_wt_list = []

        mass_dr = m[:, None] * dr
        use_force = (
            f_es_mode is not None
            and getattr(self.config, "phonon_embedding_method", "displacement").lower() == "force"
        )
        if use_force:
            logger.info("    [+] Using force-based mode coupling projection Delta Q_k = F_k / omega_k^2 (Alkauskas/Razinkovas)")
        else:
            logger.info("    [+] Using displacement mode coupling projection Delta Q_k = sum(m * Delta R * u_k)")

        for idx in valid_indices:
            hw_k = float(np.real(energies_ev[idx]) * 1000.0)
            mode_disp = _get_mode_func(idx)

            if use_force:
                fq_k = float(np.sum(f_es_mode * mode_disp))
                dq_k = fq_k / (2.0 * MODE_COUPLING_PREFACTOR * 1.0e-3 * (hw_k ** 2))
                s_k = float(MODE_COUPLING_PREFACTOR * hw_k * (dq_k ** 2))
            else:
                dq_k = float(np.sum(mass_dr * mode_disp))
                s_k = float(MODE_COUPLING_PREFACTOR * hw_k * (dq_k ** 2))

            weights = m * np.sum(mode_disp ** 2, axis=1)
            tot_w = float(np.sum(weights))
            p_i = weights / tot_w if tot_w > 0 else np.ones(natoms) / natoms
            ipr_k = float(np.sum(p_i ** 2))
            pr_k = float(1.0 / (natoms * ipr_k)) if ipr_k > 0 else 1.0
            def_wt = float(np.sum(p_i[defect_indices])) if defect_indices else 0.0

            hw_modes.append(hw_k)
            sk_list.append(s_k)
            dqk_list.append(dq_k)
            ipr_list.append(ipr_k)
            pr_list.append(pr_k)
            def_wt_list.append(def_wt)

        hw_modes = np.array(hw_modes, dtype=float)
        sk_modes = np.array(sk_list, dtype=float)
        total_S = float(np.sum(sk_modes))
        logger.info(f"[+] Total Huang-Rhys Factor S: {total_S:.3f} (Franck-Condon Delta Q = {dq_total:.4f} amu^(1/2)*A)")

        is_embedded = at_gs.info.get("is_embedded", False) or (
            phonon_rep is not None and tuple(phonon_rep) != tuple(supercell_rep)
        )
        s_outside_fraction = 0.0
        s_outside_total = 0.0
        if is_embedded:
            c_center = np.asarray(at_gs.info.get("defect_center", at_gs.get_cell().sum(axis=0) / 2.0), dtype=float)
            from .defect_builder import mic_distances
            dists_to_center = mic_distances(at_gs.positions, c_center, at_gs.cell)
            dft_natoms = at_gs.info.get("dft_supercell_natoms", len(at_gs))
            sorted_indices = np.argsort(dists_to_center)
            dft_core_mask = np.zeros(len(at_gs), dtype=bool)
            dft_core_mask[sorted_indices[:dft_natoms]] = True
            outside_mask = ~dft_core_mask

            if len(valid_indices) > 0 and np.any(outside_mask):
                outside_S_modes = []
                for k, idx in enumerate(valid_indices):
                    mode_disp = _get_mode_func(idx)
                    weights = m * np.sum(mode_disp ** 2, axis=1)
                    tot_w = float(np.sum(weights))
                    p_i = weights / tot_w if tot_w > 0 else np.ones(natoms) / natoms
                    p_outside = float(np.sum(p_i[outside_mask]))
                    outside_S_modes.append(sk_modes[k] * p_outside)
                s_outside_total = float(np.sum(outside_S_modes))
                s_outside_fraction = (s_outside_total / total_S) if total_S > 0 else 0.0
                logger.info(
                    f"    [+] Fraction of S originating from atoms outside DFT supercell radius: "
                    f"{s_outside_fraction * 100.0:.2f}% (S_outside = {s_outside_total:.4f} of {total_S:.4f})"
                )

        if getattr(self.config, "export_normal_modes", True):
            normal_modes_csv = os.path.join(self.config.workspace_dir, "normal_modes.csv")
            with open(normal_modes_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "mode_index",
                    "hw_meV",
                    "S_k",
                    "dQ_k_amu12_A",
                    "IPR",
                    "participation_ratio",
                    "is_localized",
                    "defect_weight",
                ])
                for k, idx in enumerate(valid_indices):
                    is_loc = 1 if ipr_list[k] > (3.0 / natoms) else 0
                    writer.writerow([
                        idx,
                        f"{hw_modes[k]:.4f}",
                        f"{sk_list[k]:.6f}",
                        f"{dqk_list[k]:.6f}",
                        f"{ipr_list[k]:.6f}",
                        f"{pr_list[k]:.4f}",
                        is_loc,
                        f"{def_wt_list[k]:.4f}",
                    ])
            logger.info(f"[+] Exported per-mode normal mode analysis ({len(valid_indices)} modes) -> {normal_modes_csv}")

        max_mode = float(np.max(hw_modes)) if len(hw_modes) > 0 else 65.0
        user_hw_max = self.config.phonon_hw_max
        hw_max = user_hw_max if user_hw_max is not None else max(70.0, max_mode * 1.05)
        n_hw_pts = max(600, int(np.ceil(hw_max * 10)))
        hw_grid = np.linspace(0.5, hw_max, n_hw_pts)

        user_smear = self.config.phonon_smearing_meV
        if user_smear is not None:
            sigma_smear = float(user_smear)
        else:
            dft_natoms = at_gs.info.get("dft_supercell_natoms", len(at_gs))
            linear_scale = (len(at_gs) / max(1, dft_natoms)) ** (1.0 / 3.0) if is_embedded else 1.0
            active_mask = (
                sk_modes > 0.005 * np.max(sk_modes)
                if (len(sk_modes) > 0 and np.max(sk_modes) > 0)
                else np.ones(len(sk_modes), dtype=bool)
            )
            active_modes = np.sort(hw_modes[active_mask])
            scale_smear = (0.035 * hw_max) / max(1.0, linear_scale)
            if len(active_modes) > 2:
                gaps = np.diff(active_modes)
                valid_gaps = gaps[gaps > 0.1]
                med_gap = float(np.median(valid_gaps)) if len(valid_gaps) > 0 else scale_smear
                target_smear = max(med_gap * 1.5, scale_smear)
            else:
                target_smear = scale_smear

            if is_embedded:
                sigma_smear = float(np.clip(target_smear, 0.8, 3.5))
            else:
                sigma_smear = float(np.clip(target_smear, 1.5, 8.5))

        use_var_smear = (user_smear is None) and is_embedded
        if use_var_smear and len(hw_modes) > 0:
            s_lo = float(np.clip(1.8 * sigma_smear, 3.2, 6.0))
            s_hi = float(np.clip(0.78 * sigma_smear, 1.2, 3.0))
            logger.info(
                f"[+] Phonon Spectral Smearing (Dispersion-Matched): sigma = {s_lo:.2f} -> {s_hi:.2f} meV"
            )
        else:
            logger.info(f"[+] Phonon Spectral Smearing (Finite-Size Correction): sigma = {sigma_smear:.2f} meV")
        s_cont = _smear_harmonic_spectral_density(
            hw_grid, hw_modes, sk_modes, sigma_smear, variable_dispersion=use_var_smear
        )

        trapz_fn = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
        moment_before = None
        moment_after = None
        if self.config.finite_size_acoustic_cutoff:
            int_s_raw = float(trapz_fn(s_cont, hw_grid))
            moment_before = float(trapz_fn(hw_grid * s_cont, hw_grid) / int_s_raw) if int_s_raw > 0 else 0.0

            s_cont *= np.tanh((hw_grid / max(1.0, sigma_smear)) ** 2)

            int_s_cut = float(trapz_fn(s_cont, hw_grid))
            if int_s_cut > 0 and int_s_raw > 0:
                s_cont *= (int_s_raw / int_s_cut)

            moment_after = float(trapz_fn(hw_grid * s_cont, hw_grid) / int_s_raw) if int_s_raw > 0 else 0.0
            logger.info(
                f"    [i] Acoustic cutoff applied: <hbar*omega> first moment shifted from "
                f"{moment_before:.2f} meV to {moment_after:.2f} meV (delta = {moment_after - moment_before:+.2f} meV)"
            )

        s_cont *= 1000.0  # eV^-1

        int_s_before = float(trapz_fn(s_cont / 1000.0, hw_grid))
        if int_s_before > 0 and total_S > 0:
            s_cont *= (total_S / int_s_before)
        int_s_after = float(trapz_fn(s_cont / 1000.0, hw_grid))
        logger.info(
            f"[+] Normalised continuous S(hw): integrated S = {int_s_after:.4f} strictly matches sum(S_k) = {total_S:.4f}"
        )

        e_min = self.config.lineshape_e_min
        e_max = self.config.lineshape_e_max
        n_pts = self.config.lineshape_n_points
        if e_min is None or e_max is None:
            mean_hw = float(np.sum(hw_modes * sk_modes) / total_S) if total_S > 0 else 35.0
            erel = total_S * mean_hw
            sigma_fc = np.sqrt(max(1.0, total_S)) * mean_hw
            if e_min is None:
                e_min = -max(250.0, erel + 4.5 * sigma_fc)
            if e_max is None:
                e_max = max(60.0, 2.5 * mean_hw)

        # Finite-T Debye-Waller factor
        dwf_dict = {}
        for T in self.config.temperatures:
            if T < 0.1:
                dwf_dict[T] = float(np.exp(-total_S))
            else:
                n_bose_k = 1.0 / (np.exp(np.clip(hw_modes / (KB_MEV_K * T), 1e-9, 500.0)) - 1.0)
                dwf_dict[T] = float(np.exp(-np.sum(sk_modes * (2.0 * n_bose_k + 1.0))))

        logger.info(
            f"\n[-] Computing Harmonic Optical Lineshapes across temperatures: {self.config.temperatures} K "
            f"(Range: [{e_min:.1f}, {e_max:.1f}] meV)..."
        )
        lineshape_dict = {}
        pl_dict = {}
        zpl_eV = self.config.zpl_energy_eV

        for T in tqdm(self.config.temperatures, desc="Harmonic lineshapes", unit="T", leave=False):
            user_gamma = self.config.dephasing_gamma_meV
            base_gamma = 1.0 if user_gamma is None else float(user_gamma)
            coeff = getattr(self.config, "dephasing_temperature_coefficient", 0.0)
            if coeff > 0.0 and T == self.config.temperatures[0]:
                logger.info(
                    f"    [i] User-specified thermal dephasing active: gamma(T) = {base_gamma:.2f} + {coeff:.2e} * T^2 meV (phenomenological model)"
                )
            gamma = base_gamma + coeff * (T ** 2)

            eg, a_norm = compute_quantum_lineshape(
                hw_grid,
                s_cont,
                temp_k=T,
                dephasing_gamma_meV=gamma,
                time_window_tau_fs=self.config.time_window_tau_fs,
                e_min_meV=e_min,
                e_max_meV=e_max,
                n_e=n_pts,
            )
            lineshape_dict[T] = (eg, a_norm)
            dwf_t = dwf_dict[T]
            logger.info(
                f"    [+] Harmonic T = {T:5.1f} K | Peak A(E) = {np.max(a_norm):.4f} | "
                f"ZPL DWF(T) = {dwf_t*100.0:.2f}% (T=0: {np.exp(-total_S)*100.0:.2f}%)"
            )
            if zpl_eV is not None and getattr(self.config, "calculate_pl_intensity", True):
                hw_phot, i_pl = compute_photoluminescence_spectrum(eg, a_norm, zpl_eV)
                pl_dict[T] = (hw_phot, i_pl)

        cc_dict = None
        if self.config.plot_configuration_coordinate:
            try:
                calc_es_opt = getattr(self, "calc_es", None)
                cc_dict = self.plot_configuration_coordinate(at_gs, at_es, zpl_energy_eV=zpl_eV, calc_es=calc_es_opt)
                if zpl_eV is None and cc_dict:
                    raw_zpl = cc_dict.get("e_zpl_eV", cc_dict.get("zpl_energy_eV"))
                    if raw_zpl is not None and float(raw_zpl) > 0:
                        zpl_eV = float(raw_zpl)
                        logger.info(f"[+] Derived ZPL energy from CC analysis: {zpl_eV:.4f} eV")
                        if getattr(self.config, "calculate_pl_intensity", True):
                            for T, (eg, a_norm) in lineshape_dict.items():
                                hw_phot, i_pl = compute_photoluminescence_spectrum(eg, a_norm, zpl_eV)
                                pl_dict[T] = (hw_phot, i_pl)
            except Exception as e:
                logger.warning(f"[!] Warning: Configuration coordinate calculation encountered an issue: {e}")

        total_S_std = None
        s_cont_std = None
        lineshape_std_dict = {}

        if getattr(self.calc, "is_committee", False):
            committee_size = len(getattr(self.calc, "calc_deltas", []))
            if committee_size > 1:
                logger.info(
                    f"\n[+] Propagating Delta-NEP Committee Ensemble ({committee_size} members) to evaluate uncertainties..."
                )
                member_S = []
                member_s_cont = []
                member_lineshapes = {T: [] for T in self.config.temperatures}

                for m_idx in range(committee_size):
                    if isinstance(self.calc, HybridDefectCalculator):
                        m_calc = HybridDefectCalculator(
                            calc_base=self.calc.calc_base,
                            calc_delta=self.calc.calc_deltas[m_idx],
                            energy_offset_per_atom=self.calc.energy_offset_per_atom,
                        )
                    else:
                        m_calc = self.calc.calc_deltas[m_idx]

                        m_f_es = f_es_mode
                        if use_force and hasattr(self, "calc_es") and self.calc_es is not None:
                            if getattr(self.calc_es, "is_committee", False):
                                es_idx = min(m_idx, len(self.calc_es.calc_deltas) - 1)
                                m_es_delta = self.calc_es.calc_deltas[es_idx]
                            else:
                                m_es_delta = getattr(self.calc_es, "calc_delta", self.calc_es)

                            if isinstance(self.calc_es, HybridDefectCalculator):
                                m_calc_es = HybridDefectCalculator(
                                    calc_base=self.calc_es.calc_base,
                                    calc_delta=m_es_delta,
                                    energy_offset_per_atom=getattr(
                                        self.calc_es, "energy_offset_per_atom", self.calc.energy_offset_per_atom
                                    ),
                                )
                            else:
                                m_calc_es = m_es_delta

                            at_probe = at_gs.copy()
                            at_probe.calc = m_calc_es
                            m_f_es = at_probe.get_forces().copy()

                        m_energies, m_mode_fn = self._run_vibrations_for_calculator(
                            at_gs, m_calc, f"vib_member_{m_idx}"
                        )
                        m_sk_list = []
                        m_hw_list = []
                        for idx_v in valid_indices:
                            if idx_v < len(m_energies):
                                hw_v = float(np.real(m_energies[idx_v]) * 1000.0)
                                if hw_v > 1.0:
                                    u_v = m_mode_fn(idx_v)
                                    if use_force and m_f_es is not None:
                                        fq_v = float(np.sum(m_f_es * u_v))
                                        dq_v = fq_v / (2.0 * MODE_COUPLING_PREFACTOR * 1.0e-3 * (hw_v ** 2))
                                        sk_v = float(MODE_COUPLING_PREFACTOR * hw_v * (dq_v ** 2))
                                    else:
                                        dq_v = float(np.sum(mass_dr * u_v))
                                        sk_v = float(MODE_COUPLING_PREFACTOR * hw_v * (dq_v ** 2))
                                    m_hw_list.append(hw_v)
                                    m_sk_list.append(sk_v)

                        m_hw_arr = np.array(m_hw_list, dtype=float)
                        m_sk_arr = np.array(m_sk_list, dtype=float)
                        m_tot_s = float(np.sum(m_sk_arr)) if len(m_sk_arr) > 0 else total_S
                        member_S.append(m_tot_s)

                        if len(m_hw_arr) > 0:
                            m_s_cont = _smear_harmonic_spectral_density(
                                hw_grid, m_hw_arr, m_sk_arr, sigma_smear, variable_dispersion=use_var_smear
                            )
                            if self.config.finite_size_acoustic_cutoff:
                                m_s_cont *= np.tanh((hw_grid / max(1.0, sigma_smear)) ** 2)
                            m_s_cont *= 1000.0
                            int_m = float(trapz_fn(m_s_cont / 1000.0, hw_grid))
                            if int_m > 0 and m_tot_s > 0:
                                m_s_cont *= (m_tot_s / int_m)
                        else:
                            m_s_cont = np.zeros_like(hw_grid)
                        member_s_cont.append(m_s_cont)

                        for T in self.config.temperatures:
                            user_gamma = self.config.dephasing_gamma_meV
                            base_gamma = 1.0 if user_gamma is None else float(user_gamma)
                            coeff = getattr(self.config, "dephasing_temperature_coefficient", 0.0)
                            gamma = base_gamma + coeff * (T ** 2)
                            _, m_a = compute_quantum_lineshape(
                                hw_grid,
                                m_s_cont,
                                temp_k=T,
                                dephasing_gamma_meV=gamma,
                                time_window_tau_fs=self.config.time_window_tau_fs,
                                e_min_meV=e_min,
                                e_max_meV=e_max,
                                n_e=n_pts,
                            )
                            member_lineshapes[T].append(m_a)

                total_S_std = float(np.std(member_S))
                s_cont_std = np.std(member_s_cont, axis=0)
                lineshape_std_dict = {
                    T: np.std(member_lineshapes[T], axis=0)
                    for T in self.config.temperatures
                }
                logger.info(
                    f"[+] Committee Uncertainty Summary: Total S = {total_S:.3f} +/- {total_S_std:.3f} (spread over {committee_size} models)"
                )

        anharmonic_dict = None
        if self.config.enable_anharmonic:
            if is_embedded:
                logger.info(
                    f"    [-] Running anharmonic MD on embedded {len(at_gs)}-atom phonon supercell "
                    f"to resolve Brillouin-zone acoustic modes."
                )
            anharmonic_dict = self.run_anharmonic_sampling(
                at_gs, at_es, hw_grid, s_cont, total_S=total_S, e_min=e_min, e_max=e_max, n_pts=n_pts,
                f_es_mode=f_es_mode,
            )
            self.plot_all(
                lineshape_dict,
                hw_grid,
                s_cont,
                total_S,
                anharmonic_dict=anharmonic_dict,
                zpl_energy_eV=zpl_eV,
                s_cont_std=s_cont_std,
                lineshape_std_dict=lineshape_std_dict,
            )
        else:
            self.plot_all(
                lineshape_dict,
                hw_grid,
                s_cont,
                total_S,
                anharmonic_dict=None,
                zpl_energy_eV=zpl_eV,
                s_cont_std=s_cont_std,
                lineshape_std_dict=lineshape_std_dict,
            )

        res = {
            "harmonic": lineshape_dict,
            "anharmonic": anharmonic_dict,
            "pl_spectrum": pl_dict,
            "total_S": total_S,
            "total_S_std": total_S_std,
            "total_S_discrete": total_S,
            "total_S_integrated": int_s_after,
            "delta_q": dq_total,
            "s_outside_dft_fraction": float(s_outside_fraction),
            "s_outside_dft_total": float(s_outside_total),
            "acoustic_cutoff_moment_before": moment_before,
            "acoustic_cutoff_moment_after": moment_after,
            "dwf_temperatures": dwf_dict,
            "n_normal_modes": len(valid_indices),
            "n_imaginary_modes": len(imaginary_modes),
            "zpl_energy_eV": zpl_eV,
            "hw_grid": hw_grid,
            "s_cont": s_cont,
            "s_cont_std": s_cont_std,
            "lineshape_std": lineshape_std_dict,
            "cc_analysis": cc_dict,
        }

        # Export numerical results
        self.export_results(res)
        return res

    def export_results(self, results: Dict[str, Any], output_dir: Optional[str] = None) -> Dict[str, str]:
        """Exports numerical results (spectral densities, lineshapes, summary) to CSV and JSON."""
        out_dir = output_dir or self.config.workspace_dir
        os.makedirs(out_dir, exist_ok=True)

        summary_json = os.path.join(out_dir, "lineshape_summary.json")
        spectral_csv = os.path.join(out_dir, "spectral_density.csv")
        lineshape_csv = os.path.join(out_dir, "optical_lineshapes.csv")

        # 1. Summary JSON
        total_S = float(results.get("total_S", 0.0))
        zpl_eV = results.get("zpl_energy_eV")
        dwf_dict = results.get("dwf_temperatures", {})
        summary_data = {
            "total_S": total_S,
            "total_S_std": float(results["total_S_std"]) if results.get("total_S_std") is not None else None,
            "total_S_discrete": float(results.get("total_S_discrete", total_S)),
            "total_S_integrated": float(results.get("total_S_integrated", total_S)),
            "s_outside_dft_fraction": float(results["s_outside_dft_fraction"]) if "s_outside_dft_fraction" in results else None,
            "s_outside_dft_total": float(results["s_outside_dft_total"]) if "s_outside_dft_total" in results else None,
            "acoustic_cutoff_applied": bool(self.config.finite_size_acoustic_cutoff),
            "acoustic_cutoff_moment_before_meV": float(results["acoustic_cutoff_moment_before"]) if results.get("acoustic_cutoff_moment_before") is not None else None,
            "acoustic_cutoff_moment_after_meV": float(results["acoustic_cutoff_moment_after"]) if results.get("acoustic_cutoff_moment_after") is not None else None,
            "dwf_zero_temp": float(np.exp(-total_S)),
            "dwf_temperatures": {str(T): float(dwf_dict.get(T, np.exp(-total_S))) for T in self.config.temperatures},
            "n_normal_modes": int(results.get("n_normal_modes", 0)),
            "n_imaginary_modes": int(results.get("n_imaginary_modes", 0)),
            "zpl_energy_eV": float(zpl_eV) if zpl_eV is not None else None,
            "temperatures": self.config.temperatures,
        }
        anharmonic_dict = results.get("anharmonic")
        if anharmonic_dict:
            summary_data["anharmonic"] = {}
            for T, data in anharmonic_dict.items():
                summary_data["anharmonic"][str(T)] = {
                    "regime": data.get("regime"),
                    "md_strategy": data.get("md_strategy", "NVT_thermalization_to_NVE_production"),
                    "n_nvt_steps": int(data.get("n_nvt_steps", 0)),
                    "n_nve_steps": int(data.get("n_nve_steps", 0)),
                    "target_T_K": float(data.get("target_T_K", float(T))),
                    "equilibrated_T_K": float(data.get("equilibrated_T_K", float(T))),
                    "fwhm_meV": float(data.get("fwhm_meV", 0.0)),
                    "sigma_du": float(data.get("sigma_du", 0.0)),
                    "alpha_soft": float(data.get("alpha_soft", 1.0)),
                    "omega_T": float(data.get("omega_T", 0.0)),
                }

        with open(summary_json, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2)

        # 2. Spectral density CSV
        hw_grid = results.get("hw_grid")
        s_cont = results.get("s_cont")
        s_cont_std = results.get("s_cont_std")
        if hw_grid is not None and s_cont is not None:
            with open(spectral_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                headers = ["hw_meV", "S_harmonic_per_eV"]
                if s_cont_std is not None:
                    headers.append("S_harmonic_std_per_eV")
                anh_temps = [T for T in self.config.temperatures if anharmonic_dict and T in anharmonic_dict]
                for T in anh_temps:
                    headers.append(f"S_anharmonic_{T}K_per_eV")
                    headers.append(f"VDOS_{T}K")
                writer.writerow(headers)

                for idx, (hw, s) in enumerate(zip(hw_grid, s_cont)):
                    row = [f"{hw:.4f}", f"{s:.6f}"]
                    if s_cont_std is not None:
                        row.append(f"{s_cont_std[idx]:.6f}")
                    for T in anh_temps:
                        s_t = anharmonic_dict[T].get("s_anh")
                        s_val = s_t[idx] if s_t is not None else s
                        vdos = anharmonic_dict[T]["vdos"]
                        row.append(f"{s_val:.6f}")
                        row.append(f"{vdos[idx]:.6f}")
                    writer.writerow(row)

        # 3. Optical lineshapes CSV
        lineshape_dict = results.get("harmonic", {})
        pl_dict = results.get("pl_spectrum", {})
        lineshape_std = results.get("lineshape_std", {})
        if lineshape_dict:
            first_T = self.config.temperatures[0]
            if first_T in lineshape_dict:
                eg = lineshape_dict[first_T][0]
                with open(lineshape_csv, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    headers = ["energy_rel_meV"]
                    if zpl_eV is not None:
                        headers.append("photon_energy_eV")
                    for T in self.config.temperatures:
                        if T in lineshape_dict:
                            headers.append(f"A_harmonic_{T}K")
                            if lineshape_std and T in lineshape_std:
                                headers.append(f"A_harmonic_std_{T}K")
                        if pl_dict and T in pl_dict:
                            headers.append(f"I_PL_{T}K_norm")
                        if anharmonic_dict and T in anharmonic_dict:
                            headers.append(f"A_anharmonic_{T}K")
                    writer.writerow(headers)

                    for idx, e_rel in enumerate(eg):
                        row = [f"{e_rel:.4f}"]
                        if zpl_eV is not None:
                            row.append(f"{zpl_eV + e_rel / 1000.0:.6f}")
                        for T in self.config.temperatures:
                            if T in lineshape_dict:
                                row.append(f"{lineshape_dict[T][1][idx]:.8e}")
                                if lineshape_std and T in lineshape_std:
                                    row.append(f"{lineshape_std[T][idx]:.8e}")
                            if pl_dict and T in pl_dict:
                                row.append(f"{pl_dict[T][1][idx]:.8e}")
                            if anharmonic_dict and T in anharmonic_dict:
                                row.append(f"{anharmonic_dict[T]['lineshape'][idx]:.8e}")
                        writer.writerow(row)

        logger.info(f"[+] Numerical lineshape results exported: {summary_json}, {spectral_csv}, {lineshape_csv}")
        return {"summary": summary_json, "spectral_density": spectral_csv, "lineshapes": lineshape_csv}

    def run_anharmonic_sampling(
        self, at_gs, at_es, hw_grid, s_cont, total_S=None, e_min=None, e_max=None, n_pts=1500,
        f_es_mode=None,
    ):
        """Runs paired molecular dynamics across configured temperatures to evaluate anharmonic lineshapes."""
        logger.info("\n[-] Running Finite-Temperature Anharmonic Molecular Dynamics Sampling...")
        dr_fc = mic_displacement(at_es.positions, at_gs.positions, at_gs.cell)
        m = at_gs.get_masses()
        dt_fs = self.config.anharmonic_dt_fs
        n_steps = self.config.anharmonic_md_steps
        trapz_fn = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

        if total_S is None:
            dhw = hw_grid[1] - hw_grid[0] if len(hw_grid) > 1 else 1.0
            total_S = float(np.sum(s_cont * dhw) / 1000.0)

        regime_opt = self.config.anharmonic_coupling_regime.lower()
        is_cumulant = regime_opt in ("cumulant", "auto")
        is_quantum = regime_opt in ("quantum", "legacy_quantum")
        is_semiclassical = regime_opt in ("semiclassical", "legacy_semiclassical", "marcus")

        if is_quantum or is_semiclassical:
            logger.warning(
                f"[!] Anharmonic coupling regime '{regime_opt}' is a legacy method. "
                f"The energy-gap cumulant ('cumulant' or 'auto') is the standard physical method."
            )

        norm_dr = np.sqrt(np.sum(m[:, None] * (dr_fc ** 2)))
        if norm_dr > 1e-8:
            u_opt = (m[:, None] * dr_fc) / norm_dr
        else:
            u_opt = np.ones_like(dr_fc) / np.sqrt(len(dr_fc) * 3)

        active_mask = (s_cont > 0.01 * np.max(s_cont)) if (len(s_cont) > 0 and np.max(s_cont) > 0) else np.ones(len(hw_grid), dtype=bool)
        if np.any(active_mask) and trapz_fn(s_cont[active_mask], hw_grid[active_mask]) > 1e-8:
            omega_0 = float(
                trapz_fn(hw_grid[active_mask] * s_cont[active_mask], hw_grid[active_mask])
                / trapz_fn(s_cont[active_mask], hw_grid[active_mask])
            )
        else:
            omega_0 = 35.0

        eg_base, a_quant_0K = compute_quantum_lineshape(
            hw_grid, s_cont, temp_k=0.01, dephasing_gamma_meV=1.0,
            e_min_meV=e_min, e_max_meV=e_max, n_e=n_pts
        )
        dE = eg_base[1] - eg_base[0] if len(eg_base) > 1 else 1.0
        w_rad_fs = (hw_grid / 1000.0) / HBAR_EV_FS

        calc_es_opt = getattr(self, "calc_es", None)
        paired_calc = getattr(self, "paired_calc", None)
        md_surface = str(getattr(self.config, "anharmonic_md_surface", "es")).lower()
        has_paired = (calc_es_opt is not None) and (md_surface != "gs")
        is_embedded_cell = bool(at_gs.info.get("is_embedded", False))
        use_embedded_core_force = (
            is_embedded_cell
            and f_es_mode is not None
            and len(f_es_mode) == len(at_gs)
            and bool(np.any(np.abs(f_es_mode) > 1e-12))
        )
        f_core_pass = np.asarray(f_es_mode, dtype=float) if use_embedded_core_force else None
        r_ref_pass = np.asarray(at_gs.positions, dtype=float).copy() if use_embedded_core_force else None

        if use_embedded_core_force:
            n_active_core = int(np.count_nonzero(np.linalg.norm(f_core_pass, axis=1) > 1e-12))
            pot_label = "V_ES(R)" if has_paired else "V_GS(R)"
            logger.info(
                f"    [+] Embedded anharmonic Delta U(t) localized to {n_active_core} core atoms via F_core ({pot_label}), "
                f"suppressing bulk Delta-NEP quadratic noise."
            )
        elif has_paired:
            if paired_calc is None:
                from .calculator import PairedDefectCalculator
                paired_calc = PairedDefectCalculator(self.calc, calc_es_opt)
            logger.info("    [+] Anharmonic emission Delta U(t) driven by physical paired model: trajectories run on V_ES(R), Delta U = V_ES(R) - V_GS(R)")
        else:
            logger.info("    [i] Ground-state MD sampling: Delta U(t) driven on V_GS(R) trajectory")

        anharmonic_dict = {}
        n_trajectories = max(1, int(getattr(self.config, "anharmonic_n_trajectories", 4)))

        at_es_eq = at_es.copy()
        at_gs_eq = at_gs.copy()
        if has_paired and len(at_es_eq) > 0:
            at_es_eq.calc = calc_es_opt
            try:
                f_es_init = float(np.abs(at_es_eq.get_forces()).max())
                if f_es_init > 0.02:
                    from ase.optimize import BFGS, LBFGS
                    logger.info(
                        f"    [-] Relaxing excited-state MD starting geometry (initial max |F| = {f_es_init:.4f} eV/A)..."
                    )
                    opt_es = LBFGS(at_es_eq, logfile=None) if len(at_es_eq) > 400 else BFGS(at_es_eq, logfile=None)
                    opt_es.run(fmax=0.01, steps=40)
                    f_es_after = float(np.abs(at_es_eq.get_forces()).max())
                    logger.info(f"    [+] Pre-MD ES geometry relaxed: max |F| = {f_es_after:.4f} eV/A")
            except Exception as exc:
                logger.warning(f"    [!] Pre-MD ES relaxation skipped ({exc}).")
        elif not has_paired and len(at_gs_eq) > 0:
            at_gs_eq.calc = self.calc
            try:
                f_gs_init = float(np.abs(at_gs_eq.get_forces()).max())
                if f_gs_init > 0.02:
                    from ase.optimize import BFGS, LBFGS
                    opt_gs = LBFGS(at_gs_eq, logfile=None) if len(at_gs_eq) > 400 else BFGS(at_gs_eq, logfile=None)
                    opt_gs.run(fmax=0.01, steps=40)
            except Exception:
                pass

        n_therm = getattr(self.config, "anharmonic_thermalization_steps", 400)
        at_start_bare = (at_es_eq if has_paired else at_gs_eq).copy()
        at_start_bare.calc = None

        use_parallel = bool(getattr(self.config, "parallel_hessian", True)) and n_trajectories > 1
        gs_spec = _extract_hybrid_calc_spec(self.calc) if use_parallel else None
        es_spec = _extract_hybrid_calc_spec(calc_es_opt) if (use_parallel and has_paired) else None
        can_parallel = use_parallel and gs_spec is not None and (not has_paired or es_spec is not None)

        executor = None
        total_md_tasks = len(self.config.temperatures) * n_trajectories
        if can_parallel:
            max_cfg = getattr(self.config, "parallel_hessian_workers", None) or os.cpu_count() or 4
            n_workers = max(1, min(int(max_cfg), total_md_tasks, os.cpu_count() or 1))
            if n_workers > 1:
                worker_spec = {"gs": gs_spec, "es": es_spec}
                executor = ProcessPoolExecutor(
                    max_workers=n_workers,
                    mp_context=multiprocessing.get_context("spawn"),
                    initializer=_init_anharmonic_worker,
                    initargs=(worker_spec,),
                )
                logger.info(
                    f"    [+] Using {n_workers} parallel single-threaded MD workers across "
                    f"{total_md_tasks} total trajectories ({len(self.config.temperatures)} temperatures x {n_trajectories})."
                )

        md_results_by_T: Dict[float, Tuple[List[np.ndarray], List[np.ndarray], float, float]] = {}
        try:
            if executor is not None:
                t_md_all0 = time.time()
                for T in self.config.temperatures:
                    logger.info(
                        f"    [MD T = {T:g} K] Queuing NVT ({n_therm} steps) -> NVE ({n_steps} steps, {n_trajectories} trajectories)..."
                    )
                all_tasks = [
                    (
                        at_start_bare,
                        float(T),
                        dt_fs,
                        n_therm,
                        n_steps,
                        u_opt,
                        dr_fc,
                        int(round(T * 100.0)) * 1000 + traj_idx,
                        f_core_pass,
                        r_ref_pass,
                    )
                    for T in self.config.temperatures
                    for traj_idx in range(n_trajectories)
                ]
                all_traj_results = list(executor.map(_run_anharmonic_window_worker, all_tasks))
                per_t_elapsed = (time.time() - t_md_all0) / max(1, len(self.config.temperatures))
                for t_idx, T in enumerate(self.config.temperatures):
                    sl = all_traj_results[t_idx * n_trajectories : (t_idx + 1) * n_trajectories]
                    t_equil_val = float(np.mean([r[2] for r in sl])) if sl else float(T)
                    md_results_by_T[T] = (
                        [r[0] for r in sl],
                        [r[1] for r in sl],
                        t_equil_val,
                        per_t_elapsed,
                    )
            else:
                for T in self.config.temperatures:
                    t_md0 = time.time()
                    logger.info(
                        f"    [MD T = {T:g} K] Running NVT ({n_therm} steps) -> NVE ({n_steps} steps, {n_trajectories} trajectories)..."
                    )
                    traj_results = [
                        _run_single_anharmonic_trajectory(
                            at_start=at_start_bare,
                            T=float(T),
                            dt_fs=dt_fs,
                            n_therm=n_therm,
                            n_steps=n_steps,
                            u_opt=u_opt,
                            dr_fc=dr_fc,
                            calc_gs=self.calc,
                            calc_es=calc_es_opt if has_paired else None,
                            paired_calc=paired_calc if has_paired else None,
                            seed=int(round(T * 100.0)) * 1000 + traj_idx,
                            show_progress=True,
                            desc_str=(
                                f"Anharmonic MD ({T:g} K)"
                                if n_trajectories == 1
                                else f"Anharmonic MD ({T:g} K traj {traj_idx+1}/{n_trajectories})"
                            ),
                            f_core=f_core_pass,
                            r_ref=r_ref_pass,
                        )
                        for traj_idx in range(n_trajectories)
                    ]
                    t_equil_val = float(np.mean([r[2] for r in traj_results])) if traj_results else float(T)
                    md_elapsed_s = time.time() - t_md0
                    md_results_by_T[T] = (
                        [r[0] for r in traj_results],
                        [r[1] for r in traj_results],
                        t_equil_val,
                        md_elapsed_s,
                    )
        finally:
            if executor is not None:
                executor.shutdown(wait=True)

        for T in self.config.temperatures:
            t_post0 = time.time()
            all_du_series, all_v_opt_series, t_equil, md_elapsed_s = md_results_by_T[T]

            # Combine energy gap fluctuations across trajectories
            all_du_fluc = []
            for du_arr in all_du_series:
                if len(du_arr) > 10:
                    from scipy.signal import detrend
                    du_fl = detrend(du_arr)
                    mad = np.median(np.abs(du_fl - np.median(du_fl)))
                    valid = np.abs(du_fl) < 4.0 * max(mad, 10.0)
                    all_du_fluc.append(du_fl[valid] if np.sum(valid) > 10 else du_fl)
                elif len(du_arr) > 1:
                    all_du_fluc.append(du_arr - np.mean(du_arr))
                else:
                    all_du_fluc.append(du_arr)
            sigma_du = float(np.std(np.concatenate(all_du_fluc))) if all_du_fluc else 0.0

            dt_vacf = 2.0 * dt_fs

            # Compute VACF VDOS for multi-panel spectrum with trajectory averaging
            acf_v_list = []
            for v_opt_arr in all_v_opt_series:
                n_v = len(v_opt_arr)
                n_half = max(1, n_v // 2)
                acf_full = correlate(v_opt_arr, v_opt_arr, mode="full")[n_v - 1 : n_v - 1 + n_half]
                vacf_sub = acf_full / np.arange(n_v, n_v - n_half, -1, dtype=float)
                vacf_sub /= (vacf_sub[0] + 1e-12)
                acf_v_list.append(vacf_sub)

            vacf = np.mean(acf_v_list, axis=0) if acf_v_list else np.zeros(1)
            t_vacf = np.arange(len(vacf), dtype=float) * dt_vacf
            # Linderälv et al. (npj Comput. Mater. 11, 101, 2025, Eq. 12) Gaussian ACF damping window:
            gamma_acf = float(getattr(self.config, "anharmonic_acf_damping_meV", 2.0))
            gamma_acf_fs = (gamma_acf / 1000.0) / HBAR_EV_FS
            acf_window_v = np.exp(-0.5 * (gamma_acf_fs * t_vacf) ** 2)
            # Only add Hann boundary taper when the Gaussian window has not already decayed below 5%.
            denom = t_vacf[-1] if len(t_vacf) > 1 and t_vacf[-1] > 0 else 1.0
            hann = (
                0.5 * (1.0 + np.cos(np.pi * t_vacf / denom))
                if len(acf_window_v) > 0 and float(acf_window_v[-1]) >= 0.05
                else 1.0
            )

            weights_t = vacf * acf_window_v * hann * dt_vacf
            if len(weights_t) > 1:
                weights_t[0] *= 0.5
                weights_t[-1] *= 0.5
            vdos = np.maximum(0.0, weights_t @ np.cos(np.outer(t_vacf, w_rad_fs)))
            if np.max(vdos) > 0:
                vdos /= np.max(vdos)

            if is_cumulant:
                # Energy-gap cumulant approach: C(tau) = <delta Delta U(tau) delta Delta U(0)>
                acf_du_list = []
                for du_arr in all_du_series:
                    if len(du_arr) > 10:
                        from scipy.signal import detrend
                        du_demean = detrend(du_arr)
                    else:
                        du_demean = du_arr - np.mean(du_arr)
                    n_du = len(du_demean)
                    n_half_du = max(1, n_du // 2)
                    acf_full = correlate(du_demean, du_demean, mode="full")[n_du - 1 : n_du - 1 + n_half_du]
                    acf_norm = acf_full / np.arange(n_du, n_du - n_half_du, -1, dtype=float)
                    acf_du_list.append(acf_norm)

                acf_du = np.mean(acf_du_list, axis=0) if acf_du_list else np.zeros(1)
                t_du = np.arange(len(acf_du), dtype=float) * dt_vacf

                # Linderälv et al. (npj Comput. Mater. 11, 101, 2025, Eq. 12) Gaussian ACF damping window:
                # For embedded cells with default smearing/damping, use dispersion-matched damping (3.0 -> 1.5 meV).
                use_var_acf = (
                    is_embedded_cell
                    and getattr(self.config, "phonon_smearing_meV", None) is None
                    and abs(gamma_acf - 2.0) < 1e-6
                    and len(hw_grid) > 1
                )
                denom_du = t_du[-1] if len(t_du) > 1 and t_du[-1] > 0 else 1.0
                gamma_hw = (
                    3.5 - 2.0 * (hw_grid / max(1e-6, float(hw_grid[-1])))
                    if use_var_acf
                    else np.full_like(hw_grid, gamma_acf, dtype=float)
                )
                gamma_fs_mat = (np.asarray(gamma_hw, dtype=float)[None, :] / 1000.0) / HBAR_EV_FS
                acf_window_mat = np.exp(-0.5 * (gamma_fs_mat * t_du[:, None]) ** 2)
                hann_du = (
                    0.5 * (1.0 + np.cos(np.pi * t_du / denom_du))[:, None]
                    if len(acf_window_mat) > 0 and float(np.max(acf_window_mat[-1, :])) >= 0.05
                    else 1.0
                )
                weights_mat = acf_du[:, None] * acf_window_mat * hann_du * dt_vacf
                if len(weights_mat) > 1:
                    weights_mat[0, :] *= 0.5
                    weights_mat[-1, :] *= 0.5
                j_cl = np.maximum(0.0, np.sum(weights_mat * np.cos(np.outer(t_du, w_rad_fs)), axis=0))

                HBAR_MEV_FS = HBAR_EV_FS * 1000.0
                kbt_eff = max(1e-4, KB_MEV_K * t_equil)

                # Acoustic suppression filter: suppress hw -> 0 drift/1/omega divergence while preserving >= 6 meV acoustic phonons:
                hw_cutoff = max(3.0, min(5.2, 0.11 * omega_0))
                filter_acoustic = 1.0 - np.exp(- (hw_grid / hw_cutoff) ** 4)

                # Exact classical fluctuation-dissipation relation:
                # S_MD(hw) = \tilde{C}_cl(hw) / (2 * kB * T_equil * hw)
                # where \tilde{C}_cl(hw) = 2 * j_cl(hw). The factor of 2 cancels.
                # In meV^-1: (2 / (pi * hbar_meV_fs)) * (j_cl / (2 * kbt_eff * hw))
                s_md_meV = (2.0 / (np.pi * HBAR_MEV_FS)) * (j_cl / (2.0 * kbt_eff * hw_grid)) * filter_acoustic
                s_md = s_md_meV * 1000.0  # convert to eV^-1 for compute_quantum_lineshape

                # Physical observable: do not artificially renormalize or blend away with harmonic S
                s_anh = s_md
                int_s_anh = float(trapz_fn(s_anh / 1000.0, hw_grid))

                # Below ~5 K, classical MD lacks thermal fluctuations; fallback to harmonic S(hw)
                if T < 5.0 or int_s_anh < 1e-4:
                    logger.info(f"    [i] T = {T:g} K: classical thermal fluctuations frozen; using harmonic spectral density.")
                    s_anh = s_cont
                    int_s_anh = float(trapz_fn(s_anh / 1000.0, hw_grid))

                user_gamma = self.config.dephasing_gamma_meV
                base_gamma = 1.0 if user_gamma is None else float(user_gamma)
                coeff = getattr(self.config, "dephasing_temperature_coefficient", 0.0)
                gamma = base_gamma + coeff * (T ** 2)

                eg_out, a_anh = compute_quantum_lineshape(
                    hw_grid, s_anh, temp_k=T, dephasing_gamma_meV=gamma,
                    time_window_tau_fs=self.config.time_window_tau_fs,
                    e_min_meV=e_min, e_max_meV=e_max, n_e=n_pts
                )
                regime_tag = f"Energy-Gap Cumulant (NVT->NVE, T_equil={t_equil:.1f} K, sigma_DeltaU={sigma_du:.1f} meV, S_anh={int_s_anh:.3f})"
                alpha_soft = 1.0
                omega_T = omega_0
            elif is_quantum:
                # Legacy quantum regime
                mask_active = (hw_grid >= max(10.0, 0.4 * omega_0)) & (hw_grid <= min(float(hw_grid[-1]), 1.8 * omega_0))
                if np.sum(vdos[mask_active]) > 1e-4:
                    omega_T = float(trapz_fn(hw_grid[mask_active] * vdos[mask_active], hw_grid[mask_active]) / trapz_fn(vdos[mask_active], hw_grid[mask_active]))
                    alpha_soft = float(np.clip(omega_T / omega_0, 0.80, 1.20))
                else:
                    omega_T = omega_0
                    alpha_soft = 1.0

                hw_shifted = hw_grid / alpha_soft
                s_anh = np.interp(hw_shifted, hw_grid, s_cont, left=0.0, right=0.0) / alpha_soft

                user_gamma = self.config.dephasing_gamma_meV
                gamma_base = 1.0 if user_gamma is None else float(user_gamma)
                delta_gamma_anh = 0.3 * np.sqrt(max(1.0, T) / 10.0)
                gamma_total = gamma_base + delta_gamma_anh

                eg_out, a_anh = compute_quantum_lineshape(
                    hw_grid, s_anh, temp_k=T, dephasing_gamma_meV=gamma_total,
                    e_min_meV=e_min, e_max_meV=e_max, n_e=n_pts
                )
                regime_tag = f"Quantum GenFn (alpha_soft={alpha_soft:.3f}, gamma={gamma_total:.2f} meV)"
            else:
                # Legacy semiclassical regime
                erel_est = abs(eg_base[np.argmax(a_quant_0K)])
                sig_scale = np.sqrt(2.0 * max(1.0, KB_MEV_K * T) * max(50.0, erel_est))
                sigma_du = min(max(sigma_du, 1.0), 2.0 * sig_scale)

                kw = max(5.0 * sigma_du, 50.0)
                n_pts_k = int(np.ceil(kw / dE))
                tau_grid = np.arange(-n_pts_k, n_pts_k + 1) * dE
                kernel = np.exp(-0.5 * (tau_grid / max(1.0, sigma_du)) ** 2)
                kernel /= np.sum(kernel)

                a_anh = fftconvolve(a_quant_0K, kernel, mode='same')
                total_area = trapz_fn(a_anh, eg_base)
                if total_area > 0:
                    a_anh /= total_area
                eg_out = eg_base
                alpha_soft = 1.0
                omega_T = omega_0
                regime_tag = f"Semiclassical Marcus (sigma={sigma_du:.1f} meV)"

            if len(a_anh) > 1 and np.max(a_anh) > 0:
                pk_idx = int(np.argmax(a_anh))
                half_max = 0.5 * float(a_anh[pk_idx])
                left_idx = pk_idx
                while left_idx > 0 and a_anh[left_idx - 1] >= half_max:
                    left_idx -= 1
                right_idx = pk_idx
                while right_idx < len(a_anh) - 1 and a_anh[right_idx + 1] >= half_max:
                    right_idx += 1
                fwhm_meV = float(eg_out[right_idx] - eg_out[left_idx])
            else:
                fwhm_meV = 0.0

            anharmonic_dict[T] = {
                "energies": eg_out,
                "lineshape": a_anh,
                "sigma_du": sigma_du,
                "fwhm_meV": fwhm_meV,
                "vdos": vdos,
                "s_anh": s_anh if is_cumulant else None,
                "total_S_anh": float(int_s_anh) if is_cumulant else None,
                "total_S_harm": float(total_S) if total_S is not None else None,
                "omega_T": omega_T,
                "alpha_soft": alpha_soft,
                "regime": "cumulant" if is_cumulant else ("quantum" if is_quantum else "semiclassical"),
                "method": "cumulant_energy_gap" if is_cumulant else "legacy_vdos",
                "is_physical_paired": calc_es_opt is not None,
                "md_strategy": "NVT_thermalization_to_NVE_production",
                "n_nvt_steps": n_therm,
                "n_nve_steps": n_steps,
                "n_trajectories": n_trajectories,
                "target_T_K": float(T),
                "equilibrated_T_K": float(t_equil),
                "walltime_s": md_elapsed_s + (time.time() - t_post0),
            }
            logger.info(
                f"    [+] Anharmonic MD T = {T:5.1f} K | {regime_tag} | Lineshape FWHM = {fwhm_meV:5.1f} meV "
                f"({anharmonic_dict[T]['walltime_s']:.1f} s)"
            )

        return anharmonic_dict

    def get_smart_phonon_limits(self, hw_grid, s_norm, anharmonic_dict=None, s_0k_max=None):
        """Determines optimal [x_min, x_max] and [y_min, y_max] for phonon density plots."""
        user_xmin = getattr(self.config, "phonon_hw_min", None)
        user_xmax = getattr(self.config, "phonon_hw_max", None)
        user_ymin = getattr(self.config, "phonon_y_min", None)
        user_ymax = getattr(self.config, "phonon_y_max", None)

        peak_y = float(np.max(s_norm)) if (s_norm is not None and len(s_norm) > 0 and np.max(s_norm) > 0) else 1.0
        active = hw_grid[s_norm >= 0.008] if s_norm is not None else np.array([])

        if anharmonic_dict:
            ref_max = s_0k_max if (s_0k_max is not None and s_0k_max > 0) else 1.0
            for T, data in anharmonic_dict.items():
                s_t = data.get("s_anh")
                if s_t is not None and len(s_t) == len(hw_grid):
                    dens_t = s_t / ref_max
                    peak_y = max(peak_y, float(np.max(dens_t)))
                    active = np.concatenate([active, hw_grid[dens_t >= 0.008]])
                vdos = data.get("vdos")
                if vdos is not None and len(vdos) == len(hw_grid):
                    dens_v = vdos / np.max(vdos) if np.max(vdos) > 0 else vdos
                    peak_y = max(peak_y, float(np.max(dens_v)))
                    active = np.concatenate([active, hw_grid[dens_v >= 0.008]])

        if not self.config.smart_axis_scaling:
            x_min = float(user_xmin) if user_xmin is not None else float(hw_grid[0])
            x_max = float(user_xmax) if user_xmax is not None else float(hw_grid[-1])
            y_min = float(user_ymin) if user_ymin is not None else 0.0
            y_max = float(user_ymax) if user_ymax is not None else max(1.05, 1.05 * peak_y)
            return (x_min, x_max), (y_min, y_max)

        x_min = float(user_xmin) if user_xmin is not None else 0.0

        if user_xmax is not None:
            x_max = float(user_xmax)
        else:
            if len(active) > 0:
                high_val = float(np.max(active))
                pad = max(6.0, high_val * 0.06)
                candidate_max = min(float(hw_grid[-1]), high_val + pad)
                if candidate_max > 100:
                    candidate_max = np.ceil(candidate_max / 25.0) * 25.0
                elif candidate_max > 40:
                    candidate_max = np.ceil(candidate_max / 10.0) * 10.0
                else:
                    candidate_max = np.ceil(candidate_max / 5.0) * 5.0
                x_max = min(float(hw_grid[-1]), candidate_max)
            else:
                x_max = float(hw_grid[-1])

        y_min = float(user_ymin) if user_ymin is not None else 0.0
        y_max = float(user_ymax) if user_ymax is not None else max(1.18, peak_y * 1.15)
        return (x_min, x_max), (y_min, y_max)

    def get_smart_lineshape_limits(self, lineshape_dict, anharmonic_dict=None, active_curve_subset=None):
        """Determines optimal [e_min, e_max] and [y_min, y_max] for lineshape plots."""
        all_curves = []
        for T, (eg, an_harm) in lineshape_dict.items():
            all_curves.append((eg, an_harm))
        if anharmonic_dict:
            for T, data in anharmonic_dict.items():
                if "energies" in data and "lineshape" in data:
                    all_curves.append((data["energies"], data["lineshape"]))

        if not all_curves:
            return (-250.0, 50.0), (0.0, 1.0)

        global_max_I = max(float(np.max(a)) for (_, a) in all_curves)
        user_emin = self.config.lineshape_e_min
        user_emax = self.config.lineshape_e_max
        user_ymin = self.config.lineshape_y_min
        user_ymax = self.config.lineshape_y_max

        if not self.config.smart_axis_scaling:
            grid_min = min(float(eg[0]) for (eg, _) in all_curves)
            grid_max = max(float(eg[-1]) for (eg, _) in all_curves)
            e_min_final = float(user_emin) if user_emin is not None else grid_min
            e_max_final = float(user_emax) if user_emax is not None else grid_max
            y_min_final = float(user_ymin) if user_ymin is not None else 0.0
            y_max_final = float(user_ymax) if user_ymax is not None else 1.05 * global_max_I
            return (e_min_final, e_max_final), (y_min_final, y_max_final)

        # 1. Smart X Limits (Energy relative to ZPL)
        if user_emin is not None and user_emax is not None:
            e_min_final = float(user_emin)
            e_max_final = float(user_emax)
        else:
            curve_scope_x = active_curve_subset if active_curve_subset else all_curves
            active_mins = []
            active_maxs = []
            for eg, a in curve_scope_x:
                peak = float(np.max(a))
                if peak <= 0:
                    continue
                thresh = 0.02 * peak
                mask = a >= thresh
                if np.any(mask):
                    active_mins.append(float(eg[mask].min()))
                    active_maxs.append(float(eg[mask].max()))

            if active_mins and active_maxs:
                raw_min = min(active_mins)
                raw_max = max(active_maxs)
                span = raw_max - raw_min
                pad = max(20.0, span * 0.06)

                smart_min = raw_min - pad
                if abs(smart_min) > 400:
                    smart_min = np.floor(smart_min / 50.0) * 50.0
                elif abs(smart_min) > 150:
                    smart_min = np.floor(smart_min / 25.0) * 25.0
                else:
                    smart_min = np.floor(smart_min / 10.0) * 10.0

                smart_max = max(raw_max + pad * 0.4, 35.0)
                if smart_max > 100:
                    smart_max = np.ceil(smart_max / 25.0) * 25.0
                else:
                    smart_max = np.ceil(smart_max / 10.0) * 10.0

                grid_min = min(float(eg[0]) for (eg, _) in curve_scope_x)
                grid_max = max(float(eg[-1]) for (eg, _) in curve_scope_x)
                smart_min = max(grid_min, smart_min)
                smart_max = min(grid_max, smart_max)
            else:
                smart_min, smart_max = -250.0, 50.0

            e_min_final = float(user_emin) if user_emin is not None else smart_min
            e_max_final = float(user_emax) if user_emax is not None else smart_max

        # 2. Smart Y Limits
        curve_scope = active_curve_subset if active_curve_subset else all_curves
        peak_I = max(float(np.max(a)) for (_, a) in curve_scope)
        if peak_I <= 0:
            peak_I = 1.0
        y_min_final = float(user_ymin) if user_ymin is not None else (-0.015 * peak_I)
        y_max_final = float(user_ymax) if user_ymax is not None else (1.15 * peak_I)
        if y_max_final <= y_min_final:
            y_max_final = y_min_final + 1.0

        return (e_min_final, e_max_final), (y_min_final, y_max_final)

    def plot_all(
        self,
        lineshape_dict,
        hw_grid,
        s_cont,
        total_S,
        anharmonic_dict=None,
        zpl_energy_eV=None,
        s_cont_std=None,
        lineshape_std_dict=None,
    ):
        """Dispatches combined and/or individual temperature plots according to `temperature_plot_mode`."""
        plot_mode = self.config.temperature_plot_mode.lower()

        if plot_mode in ("combined", "both"):
            self.plot_combined(
                lineshape_dict,
                hw_grid,
                s_cont,
                total_S,
                anharmonic_dict=anharmonic_dict,
                zpl_energy_eV=zpl_energy_eV,
                s_cont_std=s_cont_std,
                lineshape_std_dict=lineshape_std_dict,
            )

        if plot_mode in ("individual", "both"):
            self.plot_individual(
                lineshape_dict,
                hw_grid,
                s_cont,
                total_S,
                anharmonic_dict=anharmonic_dict,
                zpl_energy_eV=zpl_energy_eV,
                s_cont_std=s_cont_std,
                lineshape_std_dict=lineshape_std_dict,
            )

    def plot_combined(
        self,
        lineshape_dict,
        hw_grid,
        s_cont,
        total_S,
        anharmonic_dict=None,
        zpl_energy_eV=None,
        s_cont_std=None,
        lineshape_std_dict=None,
    ):
        """Plots a combined multi-temperature figure of phonon densities and lineshapes."""
        fig, axs = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
        show_harmonic = self.config.show_harmonic_in_anharmonic

        temp_colors = {T: _COLOR_CYCLE[i % len(_COLOR_CYCLE)] for i, T in enumerate(self.config.temperatures)}

        ax = axs[0]
        s_0k_max = float(np.max(s_cont)) if np.max(s_cont) > 0 else 1.0
        s_norm = s_cont / s_0k_max

        if anharmonic_dict:
            for T in self.config.temperatures:
                if T in anharmonic_dict:
                    col = temp_colors.get(T, '#1f77b4')
                    s_t = anharmonic_dict[T].get("s_anh")
                    t_str = f"{int(T)} K" if float(T).is_integer() else f"{T} K"
                    if s_t is not None:
                        dens = s_t / s_0k_max
                        lbl = f'Thermal $S(\\hbar\\omega)$ ({t_str})'
                    else:
                        vdos = anharmonic_dict[T]["vdos"]
                        dens = vdos / np.max(vdos) if np.max(vdos) > 0 else vdos
                        lbl = f'Anharmonic VDOS ({t_str})'
                    ax.fill_between(hw_grid, 0, dens, color=col, alpha=0.18, label=lbl)
                    ax.plot(hw_grid, dens, color=col, lw=2.0)

            if show_harmonic:
                ax.plot(hw_grid, s_norm, color='black', lw=2.2, linestyle='--', label=r'Harmonic $S(\hbar\omega)$ ($0\ \mathrm{K}$)')
        else:
            ax.fill_between(hw_grid, 0, s_norm, color='#2196F3', alpha=0.25, label=r'Harmonic $S(\hbar\omega)$ ($0\ \mathrm{K}$)')
            ax.plot(hw_grid, s_norm, color='#1976D2', lw=2.2)

        if s_cont_std is not None:
            s_std_norm = s_cont_std / s_0k_max
            ax.fill_between(
                hw_grid,
                np.maximum(0, s_norm - s_std_norm),
                s_norm + s_std_norm,
                color='gray',
                alpha=0.25,
                label=r'Committee $\pm 1\sigma$',
            )

        hw_xlim, hw_ylim = self.get_smart_phonon_limits(
            hw_grid, s_norm, anharmonic_dict=anharmonic_dict, s_0k_max=s_0k_max
        )
        ax.set_xlim(hw_xlim)
        ax.set_ylim(hw_ylim)

        ax.set_xlabel(r'Phonon Frequency $\hbar\omega$ (meV)', fontsize=11)
        ax.set_ylabel(r'Normalized Density / Coupling', fontsize=11)
        ax.set_title(r'(a) Electron-Phonon Coupling & Thermal VDOS', loc='left', fontweight='bold', fontsize=12)
        ax.grid(True, linestyle=':', alpha=0.5)
        ax.legend(frameon=True, loc='upper right', framealpha=0.9, fontsize=8.8)

        ax2 = axs[1]
        for T in self.config.temperatures:
            if T not in lineshape_dict:
                continue
            eg, an_harm = lineshape_dict[T]
            col = temp_colors.get(T, '#1f77b4')
            t_str = f"{int(T)} K" if float(T).is_integer() else f"{T} K"

            if anharmonic_dict and T in anharmonic_dict:
                an_anh = anharmonic_dict[T]["lineshape"]
                fwhm = anharmonic_dict[T].get("fwhm_meV", 0.0)
                ax2.plot(eg, an_anh, color=col, lw=2.4, linestyle='-', label=f'Anharmonic {t_str} (FWHM = {fwhm:.1f} meV)')
                ax2.fill_between(eg, 0, an_anh, color=col, alpha=0.10)
                if show_harmonic:
                    ax2.plot(eg, an_harm, color=col, lw=1.5, linestyle='--', alpha=0.6, label=f'Harmonic {t_str}')
            else:
                ax2.plot(eg, an_harm, color=col, lw=2.2, label=f'Harmonic {t_str}')
                ax2.fill_between(eg, 0, an_harm, color=col, alpha=0.12)

            if lineshape_std_dict and T in lineshape_std_dict:
                a_std = lineshape_std_dict[T]
                ax2.fill_between(
                    eg,
                    np.maximum(0, an_harm - a_std),
                    an_harm + a_std,
                    color=col,
                    alpha=0.20,
                )

        e_xlim, e_ylim = self.get_smart_lineshape_limits(lineshape_dict, anharmonic_dict=anharmonic_dict)
        ax2.set_xlim(e_xlim)
        if self.config.lineshape_y_scale.lower() == "log":
            ax2.set_yscale("log")
            ax2.set_ylim(bottom=max(1e-6, e_ylim[1] * 1e-4), top=e_ylim[1] * 1.5)
        else:
            ax2.set_ylim(e_ylim)

        ax2.axvline(0.0, color='gray', linestyle=':', lw=1.2, label=r'ZPL ($E=E_{\mathrm{ZPL}}$)')
        ax2.set_xlabel(r'Energy Relative to ZPL: $E - E_{\mathrm{ZPL}}$ (meV)', fontsize=11)
        ax2.set_ylabel(r'Normalized Emission Intensity', fontsize=11)
        ax2.set_title(r'(b) Multi-Temperature Optical Lineshapes $A(E, T)$', loc='left', fontweight='bold', fontsize=12)
        ax2.grid(True, linestyle=':', alpha=0.5)
        ax2.legend(frameon=True, loc='upper left', framealpha=0.9, fontsize=8.8, ncol=2 if len(lineshape_dict) > 2 else 1)

        if zpl_energy_eV is not None and zpl_energy_eV > 0:
            try:
                secax = ax2.secondary_xaxis(
                    'top',
                    functions=(lambda e: zpl_energy_eV + e / 1000.0, lambda hw: (hw - zpl_energy_eV) * 1000.0)
                )
                secax.set_xlabel(r'Absolute Photon Energy $\hbar\omega$ (eV)', fontsize=11)
            except Exception:
                pass

        plt.tight_layout()
        fig_name = self.config.anharmonic_figure_name if anharmonic_dict else self.config.lineshape_figure_name
        out_fig = os.path.join(self.config.workspace_dir, fig_name)
        plt.savefig(out_fig, dpi=300)
        plt.close(fig)
        logger.info(f"[+] Combined lineshape figure saved: {out_fig} (X-limits: {e_xlim[0]:.1f} to {e_xlim[1]:.1f} meV)")

    def plot_individual(
        self,
        lineshape_dict,
        hw_grid,
        s_cont,
        total_S,
        anharmonic_dict=None,
        zpl_energy_eV=None,
        s_cont_std=None,
        lineshape_std_dict=None,
    ):
        """Plots per-temperature figures of phonon densities and lineshapes."""
        show_harmonic = self.config.show_harmonic_in_anharmonic
        prefix = self.config.individual_figure_prefix
        s_0k_max = float(np.max(s_cont)) if np.max(s_cont) > 0 else 1.0
        s_norm = s_cont / s_0k_max

        temp_colors = {T: _COLOR_CYCLE[i % len(_COLOR_CYCLE)] for i, T in enumerate(self.config.temperatures)}

        global_e_xlim, _ = self.get_smart_lineshape_limits(lineshape_dict, anharmonic_dict=anharmonic_dict)

        for T in self.config.temperatures:
            if T not in lineshape_dict:
                continue
            eg, an_harm = lineshape_dict[T]
            col = temp_colors.get(T, '#1f77b4')
            t_str = f"{int(T)} K" if float(T).is_integer() else f"{T} K"

            fig, axs = plt.subplots(1, 2, figsize=(14, 5.0), dpi=300)

            ax = axs[0]
            vdos_sub = {T: anharmonic_dict[T]} if (anharmonic_dict and T in anharmonic_dict) else None
            hw_xlim, hw_ylim = self.get_smart_phonon_limits(
                hw_grid, s_norm, anharmonic_dict=vdos_sub, s_0k_max=s_0k_max
            )

            if anharmonic_dict and T in anharmonic_dict:
                s_t = anharmonic_dict[T].get("s_anh")
                if s_t is not None:
                    dens = s_t / s_0k_max
                    lbl = f'Thermal $S(\\hbar\\omega)$ ({t_str})'
                else:
                    vdos = anharmonic_dict[T]["vdos"]
                    dens = vdos / np.max(vdos) if np.max(vdos) > 0 else vdos
                    lbl = f'Anharmonic VDOS ({t_str})'
                ax.fill_between(hw_grid, 0, dens, color=col, alpha=0.25, label=lbl)
                ax.plot(hw_grid, dens, color=col, lw=2.2)
                if show_harmonic:
                    ax.plot(hw_grid, s_norm, color='black', lw=2.0, linestyle='--', label=r'Harmonic $S(\hbar\omega)$ ($0\ \mathrm{K}$)')
            else:
                ax.fill_between(hw_grid, 0, s_norm, color=col, alpha=0.25, label=r'Harmonic $S(\hbar\omega)$')
                ax.plot(hw_grid, s_norm, color=col, lw=2.2)

            if s_cont_std is not None:
                s_std_norm = s_cont_std / s_0k_max
                ax.fill_between(
                    hw_grid,
                    np.maximum(0, s_norm - s_std_norm),
                    s_norm + s_std_norm,
                    color='gray',
                    alpha=0.25,
                    label=r'Committee $\pm 1\sigma$',
                )

            ax.set_xlim(hw_xlim)
            ax.set_ylim(hw_ylim)

            ax.set_xlabel(r'Phonon Frequency $\hbar\omega$ (meV)', fontsize=11)
            ax.set_ylabel(r'Normalized Density / Coupling', fontsize=11)
            ax.set_title(f'(a) Phonon Density at {t_str}', loc='left', fontweight='bold', fontsize=11.5)
            ax.grid(True, linestyle=':', alpha=0.5)
            ax.legend(frameon=True, loc='upper right', framealpha=0.9, fontsize=9.2)

            ax2 = axs[1]
            curves_T = [(eg, an_harm)]
            if anharmonic_dict and T in anharmonic_dict:
                an_anh = anharmonic_dict[T]["lineshape"]
                curves_T.append((eg, an_anh))
                fwhm = anharmonic_dict[T].get("fwhm_meV", 0.0)
                ax2.plot(eg, an_anh, color=col, lw=2.4, linestyle='-', label=f'Anharmonic MD ({t_str}, FWHM = {fwhm:.1f} meV)')
                ax2.fill_between(eg, 0, an_anh, color=col, alpha=0.15)
                if show_harmonic:
                    ax2.plot(eg, an_harm, color='gray', lw=1.8, linestyle='--', label=f'Harmonic Theory ({t_str})')
            else:
                ax2.plot(eg, an_harm, color=col, lw=2.2, linestyle='-', label=f'Harmonic Theory ({t_str})')
                ax2.fill_between(eg, 0, an_harm, color=col, alpha=0.15)

            if lineshape_std_dict and T in lineshape_std_dict:
                a_std = lineshape_std_dict[T]
                ax2.fill_between(
                    eg,
                    np.maximum(0, an_harm - a_std),
                    an_harm + a_std,
                    color=col,
                    alpha=0.20,
                    label=r'Harmonic $\pm 1\sigma$',
                )

            e_xlim_T, e_ylim_T = self.get_smart_lineshape_limits(
                lineshape_dict, anharmonic_dict=anharmonic_dict, active_curve_subset=curves_T
            )
            share_x = self.config.individual_shared_x
            ax2.set_xlim(global_e_xlim if share_x else e_xlim_T)
            if self.config.lineshape_y_scale.lower() == "log":
                ax2.set_yscale("log")
                ax2.set_ylim(bottom=max(1e-6, e_ylim_T[1] * 1e-4), top=e_ylim_T[1] * 1.5)
            else:
                ax2.set_ylim(e_ylim_T)

            ax2.axvline(0.0, color='gray', linestyle=':', lw=1.2, label=r'ZPL ($E=E_{\mathrm{ZPL}}$)')
            ax2.set_xlabel(r'Energy Relative to ZPL: $E - E_{\mathrm{ZPL}}$ (meV)', fontsize=11)
            ax2.set_ylabel(r'Normalized Emission Intensity', fontsize=11)
            ax2.set_title(f'(b) Optical Lineshape at {t_str}', loc='left', fontweight='bold', fontsize=11.5)
            ax2.grid(True, linestyle=':', alpha=0.5)
            ax2.legend(frameon=True, loc='upper left', framealpha=0.9, fontsize=9.0)

            if zpl_energy_eV is not None and zpl_energy_eV > 0:
                try:
                    secax = ax2.secondary_xaxis(
                        'top',
                        functions=(lambda e: zpl_energy_eV + e / 1000.0, lambda hw: (hw - zpl_energy_eV) * 1000.0)
                    )
                    secax.set_xlabel(r'Absolute Photon Energy $\hbar\omega$ (eV)', fontsize=11)
                except Exception:
                    pass

            plt.tight_layout()
            t_tag = f"{int(T)}K" if float(T).is_integer() else f"{T}K"
            out_file = os.path.join(self.config.workspace_dir, f"{prefix}_{t_tag}.png")
            plt.savefig(out_file, dpi=300)
            plt.close(fig)
            xlim_used = global_e_xlim if share_x else e_xlim_T
            logger.info(f"[+] Individual lineshape figure saved: {out_file} (X-limits: {xlim_used[0]:.1f} to {xlim_used[1]:.1f} meV)")

    def plot_anharmonic_comparison(self, lineshape_dict, anharmonic_dict, hw_grid, s_cont, total_S):
        """Plots harmonic and anharmonic lineshape comparisons."""
        self.plot_all(lineshape_dict, hw_grid, s_cont, total_S, anharmonic_dict=anharmonic_dict)

    def plot_lineshapes(self, lineshape_dict, hw_grid, s_cont, total_S):
        """Plots harmonic lineshapes."""
        self.plot_all(lineshape_dict, hw_grid, s_cont, total_S, anharmonic_dict=None)

    def calculate_configuration_coordinate(
        self, at_gs, at_es, zpl_energy_eV=None, lambda_min=None, lambda_max=None, n_points=None
    ):
        """Calculates 1D potential energy curves and Stokes shift parameters along the configuration coordinate."""
        from .cc_diagram import ConfigurationCoordinateEngine
        engine = ConfigurationCoordinateEngine(self.config, self.calc)
        calc_es = getattr(self, "calc_es", None)
        return engine.calculate_cc_curve(
            at_gs, at_es,
            lambda_min=lambda_min,
            lambda_max=lambda_max,
            n_points=n_points,
            zpl_energy_eV=zpl_energy_eV,
            calc_es=calc_es,
        )

    def plot_configuration_coordinate(
        self, at_gs, at_es, output_path=None, zpl_energy_eV=None, system_name=None, calc_es=None
    ):
        """Calculates 1D configuration coordinate curves and renders the 2-panel diagram."""
        from .cc_diagram import ConfigurationCoordinateEngine
        engine = ConfigurationCoordinateEngine(self.config, self.calc)
        eff_calc_es = calc_es if calc_es is not None else getattr(self, "calc_es", None)
        cc_data = engine.calculate_cc_curve(at_gs, at_es, zpl_energy_eV=zpl_energy_eV, calc_es=eff_calc_es)
        fig_path = engine.plot_configuration_coordinate(cc_data, output_path=output_path, system_name=system_name)
        cc_data["figure_path"] = fig_path
        return cc_data
