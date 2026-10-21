"""Normal modes, electron-phonon spectral density S(hbar omega), Huang-Rhys factor S, and optical lineshapes A(E, T)."""

from __future__ import annotations

import os
import csv
import json
import time
import hashlib
import logging
from typing import Optional, Dict, Any, Tuple, List
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ase.vibrations import Vibrations
from scipy.signal import fftconvolve, correlate

from .config import ToolkitConfig
from .defect_builder import mic_displacement

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable

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
    dephasing_gamma_meV: float = 2.0,
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

    # Exploit Hermitian symmetry G(-t) = conj(G(t)) over t in [0, 6000] fs (FFT_TIME_POINTS points)
    t_pos = np.linspace(0.0, 6000.0, FFT_TIME_POINTS)
    dt = float(t_pos[1] - t_pos[0])

    w_fs = hw / (1000.0 * HBAR_EV_FS)
    wt = np.outer(w_fs, t_pos)
    w_re = (s_val * (2.0 * n_bose + 1.0)) * dhw
    w_im = s_val * dhw

    g_re = (w_re @ np.cos(wt)) - float(np.sum(w_re))
    g_im = w_im @ np.sin(wt)

    gamma_fs_inv = (dephasing_gamma_meV / 1000.0) / HBAR_EV_FS
    damping = np.exp(-gamma_fs_inv * t_pos - 0.5 * (t_pos / 2000.0) ** 2)
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


class LineshapeEngine:
    """Calculates phonon normal modes, spectral densities S(hw), Huang-Rhys factor, and lineshapes."""

    def __init__(self, config: ToolkitConfig, hybrid_calc):
        self.config = config
        self.calc = hybrid_calc

    def calculate_spectral_density_and_lineshapes(self, at_gs: Atoms, at_es: Atoms) -> Dict[str, Any]:
        """Calculates normal modes, spectral density S(hw), and multi-T lineshapes."""
        logger.info("\n[-] Evaluating Vibrational Normal Modes on Hybrid Potential...")
        dr = mic_displacement(at_es.positions, at_gs.positions, at_gs.cell)
        m = at_gs.get_masses()
        dq_total = float(np.sqrt(np.sum(m[:, None] * (dr ** 2))))

        vib_dir = os.path.join(self.config.workspace_dir, "vib")
        os.makedirs(vib_dir, exist_ok=True)
        state_file = os.path.join(vib_dir, "hessian_state.json")

        at = at_gs.copy()
        at.calc = self.calc
        vib = Vibrations(at, name=os.path.join(vib_dir, "vib"), delta=0.01)

        current_fp = _compute_hessian_fingerprint(at_gs, self.calc, delta=0.01)
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
        pbar_vib = tqdm(total=n_disps, desc="Normal modes (Hessian)", unit="disp", leave=False)

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

        energies_ev = vib.get_energies()

        valid_indices = [
            i for i, e in enumerate(energies_ev)
            if np.real(e) * 1000.0 > 1.0 and abs(np.imag(e)) < 1e-4
        ]
        hw_modes = []
        sk_list = []
        mass_dr = m[:, None] * dr
        for idx in valid_indices:
            hw_k = float(np.real(energies_ev[idx]) * 1000.0)
            mode_disp = vib.get_mode(idx)
            dq_k = float(np.sum(mass_dr * mode_disp))
            s_k = MODE_COUPLING_PREFACTOR * hw_k * (dq_k ** 2)
            hw_modes.append(hw_k)
            sk_list.append(s_k)

        hw_modes = np.array(hw_modes, dtype=float)
        sk_modes = np.array(sk_list, dtype=float)
        total_S = float(np.sum(sk_modes))
        logger.info(f"[+] Total Huang-Rhys Factor S: {total_S:.3f} (Franck-Condon Delta Q = {dq_total:.4f} amu^(1/2)*A)")

        max_mode = float(np.max(hw_modes)) if len(hw_modes) > 0 else 65.0
        user_hw_max = self.config.phonon_hw_max
        hw_max = user_hw_max if user_hw_max is not None else max(70.0, max_mode * 1.05)
        n_hw_pts = max(600, int(np.ceil(hw_max * 10)))
        hw_grid = np.linspace(0.5, hw_max, n_hw_pts)

        user_smear = self.config.phonon_smearing_meV
        if user_smear is not None:
            sigma_smear = float(user_smear)
        else:
            active_mask = sk_modes > 0.005 * np.max(sk_modes) if (len(sk_modes) > 0 and np.max(sk_modes) > 0) else np.ones(len(sk_modes), dtype=bool)
            active_modes = np.sort(hw_modes[active_mask])
            scale_smear = 0.035 * hw_max
            if len(active_modes) > 2:
                gaps = np.diff(active_modes)
                valid_gaps = gaps[gaps > 0.1]
                med_gap = float(np.median(valid_gaps)) if len(valid_gaps) > 0 else scale_smear
                sigma_smear = float(np.clip(max(med_gap * 1.5, scale_smear), 1.5, 8.5))
            else:
                sigma_smear = float(np.clip(scale_smear, 1.5, 8.5))

        logger.info(f"[+] Phonon Spectral Smearing (Finite-Size Correction): sigma = {sigma_smear:.2f} meV")
        if len(hw_modes) > 0:
            diff = (hw_grid[:, None] - hw_modes[None, :]) / sigma_smear
            s_cont = (np.exp(-0.5 * (diff ** 2)) @ sk_modes) / (np.sqrt(2.0 * np.pi) * sigma_smear)
        else:
            s_cont = np.zeros_like(hw_grid)

        if self.config.finite_size_acoustic_cutoff:
            s_cont *= np.tanh((hw_grid / max(1.0, sigma_smear)) ** 2)

        s_cont *= 1000.0  # eV^-1

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

        logger.info(f"\n[-] Computing Harmonic Optical Lineshapes across temperatures: {self.config.temperatures} K (Range: [{e_min:.1f}, {e_max:.1f}] meV)...")
        lineshape_dict = {}
        for T in tqdm(self.config.temperatures, desc="Harmonic lineshapes", unit="T", leave=False):
            user_gamma = self.config.dephasing_gamma_meV
            if user_gamma is not None:
                gamma = float(user_gamma)
            else:
                gamma = 1.5 + 2.0e-4 * (T ** 2)
            eg, a_norm = compute_quantum_lineshape(
                hw_grid, s_cont, temp_k=T, dephasing_gamma_meV=gamma,
                e_min_meV=e_min, e_max_meV=e_max, n_e=n_pts
            )
            lineshape_dict[T] = (eg, a_norm)
            logger.info(f"    [+] Harmonic T = {T:5.1f} K | Peak A(E) = {np.max(a_norm):.4f} | ZPL DWF = {np.exp(-total_S)*100.0:.2f}%")

        anharmonic_dict = None
        if self.config.enable_anharmonic:
            anharmonic_dict = self.run_anharmonic_sampling(
                at_gs, at_es, hw_grid, s_cont, total_S=total_S, e_min=e_min, e_max=e_max, n_pts=n_pts
            )
            self.plot_all(lineshape_dict, hw_grid, s_cont, total_S, anharmonic_dict=anharmonic_dict)
        else:
            self.plot_all(lineshape_dict, hw_grid, s_cont, total_S, anharmonic_dict=None)

        cc_dict = None
        if self.config.plot_configuration_coordinate:
            try:
                cc_dict = self.plot_configuration_coordinate(at_gs, at_es)
            except Exception as e:
                logger.warning(f"[!] Warning: Configuration coordinate calculation encountered an issue: {e}")

        res = {
            "harmonic": lineshape_dict,
            "anharmonic": anharmonic_dict,
            "total_S": total_S,
            "hw_grid": hw_grid,
            "s_cont": s_cont,
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
        summary_data = {
            "total_S": total_S,
            "dwf_zero_temp": float(np.exp(-total_S)),
            "temperatures": self.config.temperatures,
        }
        anharmonic_dict = results.get("anharmonic")
        if anharmonic_dict:
            summary_data["anharmonic"] = {}
            for T, data in anharmonic_dict.items():
                summary_data["anharmonic"][str(T)] = {
                    "regime": data.get("regime"),
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
        if hw_grid is not None and s_cont is not None:
            with open(spectral_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                headers = ["hw_meV", "S_harmonic_per_eV"]
                anh_temps = [T for T in self.config.temperatures if anharmonic_dict and T in anharmonic_dict]
                for T in anh_temps:
                    headers.append(f"VDOS_{T}K")
                writer.writerow(headers)

                for idx, (hw, s) in enumerate(zip(hw_grid, s_cont)):
                    row = [f"{hw:.4f}", f"{s:.6f}"]
                    for T in anh_temps:
                        vdos = anharmonic_dict[T]["vdos"]
                        row.append(f"{vdos[idx]:.6f}")
                    writer.writerow(row)

        # 3. Optical lineshapes CSV
        lineshape_dict = results.get("harmonic", {})
        if lineshape_dict:
            first_T = self.config.temperatures[0]
            if first_T in lineshape_dict:
                eg = lineshape_dict[first_T][0]
                with open(lineshape_csv, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    headers = ["energy_rel_meV"]
                    for T in self.config.temperatures:
                        if T in lineshape_dict:
                            headers.append(f"A_harmonic_{T}K")
                        if anharmonic_dict and T in anharmonic_dict:
                            headers.append(f"A_anharmonic_{T}K")
                    writer.writerow(headers)

                    for idx, e_rel in enumerate(eg):
                        row = [f"{e_rel:.4f}"]
                        for T in self.config.temperatures:
                            if T in lineshape_dict:
                                row.append(f"{lineshape_dict[T][1][idx]:.8e}")
                            if anharmonic_dict and T in anharmonic_dict:
                                row.append(f"{anharmonic_dict[T]['lineshape'][idx]:.8e}")
                        writer.writerow(row)

        logger.info(f"[+] Numerical lineshape results exported: {summary_json}, {spectral_csv}, {lineshape_csv}")
        return {"summary": summary_json, "spectral_density": spectral_csv, "lineshapes": lineshape_csv}

    def run_anharmonic_sampling(
        self, at_gs, at_es, hw_grid, s_cont, total_S=None, e_min=None, e_max=None, n_pts=1500
    ):
        """Runs Langevin MD across configured temperatures to evaluate anharmonic VDOS and lineshapes."""
        from ase.md.langevin import Langevin
        from ase import units

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
        if regime_opt == "auto":
            is_quantum = (total_S < 10.0)
        elif regime_opt == "quantum":
            is_quantum = True
        else:
            is_quantum = False

        norm_dr = np.sqrt(np.sum(m[:, None] * (dr_fc ** 2)))
        if norm_dr > 1e-8:
            u_opt = (m[:, None] * dr_fc) / norm_dr
        else:
            u_opt = np.ones_like(dr_fc) / np.sqrt(len(dr_fc) * 3)

        active_mask = s_cont > 0.01 * np.max(s_cont) if np.max(s_cont) > 0 else np.ones(len(hw_grid), dtype=bool)
        if np.any(active_mask):
            omega_0 = float(trapz_fn(hw_grid[active_mask] * s_cont[active_mask], hw_grid[active_mask]) / trapz_fn(s_cont[active_mask], hw_grid[active_mask]))
        else:
            omega_0 = 35.0

        eg_base, a_quant_0K = compute_quantum_lineshape(
            hw_grid, s_cont, temp_k=0.01, dephasing_gamma_meV=1.0,
            e_min_meV=e_min, e_max_meV=e_max, n_e=n_pts
        )
        dE = eg_base[1] - eg_base[0]
        w_rad_fs = (hw_grid / 1000.0) / HBAR_EV_FS

        at_disp = at_gs.copy()
        at_disp.calc = self.calc
        du_stride = 5 if is_quantum else 1

        anharmonic_dict = {}
        for T in self.config.temperatures:
            t0 = time.time()
            at = at_gs.copy()
            at.calc = self.calc
            dyn = Langevin(at, dt_fs * units.fs, temperature_K=T, friction=0.01 / units.fs, logfile=None)

            dyn.run(DEFAULT_THERMALIZATION_STEPS)

            du_list = []
            v_opt_list = []
            obs_count = [0]

            def step_observer():
                v = at.get_velocities()
                v_opt = float(np.sum(v * u_opt))
                v_opt_list.append(v_opt)
                obs_count[0] += 1

                if (obs_count[0] - 1) % du_stride == 0:
                    ug = at.get_potential_energy()
                    at_disp.set_positions(at.positions + dr_fc)
                    ue = at_disp.get_potential_energy()
                    du_list.append((ue - ug) * 1000.0)

            dyn.attach(step_observer, interval=2)
            pbar = tqdm(total=n_steps, desc=f"Anharmonic MD ({T:g} K)", unit="step", leave=False)
            step_chunk = 50
            steps_done = 0
            while steps_done < n_steps:
                this_chunk = min(step_chunk, n_steps - steps_done)
                dyn.run(this_chunk)
                steps_done += this_chunk
                pbar.update(this_chunk)
            pbar.close()

            du = np.array(du_list, dtype=float)
            if len(du) > 10:
                from scipy.signal import detrend
                du_fluc = detrend(du)
                mad = np.median(np.abs(du_fluc - np.median(du_fluc)))
                valid = np.abs(du_fluc) < 4.0 * max(mad, 10.0)
                if np.sum(valid) > 10:
                    sigma_du = float(np.std(du_fluc[valid]))
                else:
                    sigma_du = float(np.std(du_fluc))
            elif len(du) > 1:
                sigma_du = float(np.std(du))
            else:
                sigma_du = 0.0

            v_arr = np.array(v_opt_list, dtype=float)
            n_v = len(v_arr)
            n_half = max(1, n_v // 2)
            acf_full = correlate(v_arr, v_arr, mode="full")[n_v - 1 : n_v - 1 + n_half]
            vacf = acf_full / np.arange(n_v, n_v - n_half, -1, dtype=float)
            vacf /= (vacf[0] + 1e-12)

            dt_vacf = 2.0 * dt_fs
            t_vacf = np.arange(len(vacf), dtype=float) * dt_vacf
            denom = t_vacf[-1] if len(t_vacf) > 1 and t_vacf[-1] > 0 else 1.0
            hann = 0.5 * (1.0 + np.cos(np.pi * t_vacf / denom))

            weights_t = vacf * hann * dt_vacf
            if len(weights_t) > 1:
                weights_t[0] *= 0.5
                weights_t[-1] *= 0.5
            vdos = np.maximum(0.0, weights_t @ np.cos(np.outer(t_vacf, w_rad_fs)))
            if np.max(vdos) > 0:
                vdos /= np.max(vdos)

            if is_quantum:
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
                if user_gamma is not None:
                    gamma_base = float(user_gamma)
                else:
                    gamma_base = 1.5 + 2.0e-4 * (T ** 2)

                delta_gamma_anh = 0.3 * np.sqrt(max(1.0, T) / 10.0)
                gamma_total = gamma_base + delta_gamma_anh

                eg_out, a_anh = compute_quantum_lineshape(
                    hw_grid, s_anh, temp_k=T, dephasing_gamma_meV=gamma_total,
                    e_min_meV=e_min, e_max_meV=e_max, n_e=n_pts
                )
                regime_tag = f"Quantum GenFn (alpha_soft={alpha_soft:.3f}, gamma={gamma_total:.2f} meV)"
            else:
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

            fwhm_idx = np.where(a_anh >= 0.5 * np.max(a_anh))[0]
            fwhm_meV = eg_out[fwhm_idx[-1]] - eg_out[fwhm_idx[0]] if len(fwhm_idx) > 1 else 0.0

            anharmonic_dict[T] = {
                "energies": eg_out,
                "lineshape": a_anh,
                "sigma_du": sigma_du,
                "fwhm_meV": fwhm_meV,
                "vdos": vdos,
                "omega_T": omega_T,
                "alpha_soft": alpha_soft,
                "regime": "quantum" if is_quantum else "semiclassical",
                "walltime_s": time.time() - t0,
            }
            logger.info(f"    [+] Anharmonic MD T = {T:5.1f} K | {regime_tag} | Lineshape FWHM = {fwhm_meV:5.1f} meV ({time.time()-t0:.1f} s)")

        return anharmonic_dict

    def get_smart_phonon_limits(self, hw_grid, s_norm, anharmonic_dict=None):
        """Determines optimal [x_min, x_max] and [y_min, y_max] for phonon density plots."""
        if not self.config.smart_axis_scaling:
            return (float(hw_grid[0]), float(hw_grid[-1])), (0.0, 1.05)

        user_xmin = self.config.phonon_hw_min
        user_xmax = self.config.phonon_hw_max

        x_min = float(user_xmin) if user_xmin is not None else 0.0

        if user_xmax is not None:
            x_max = float(user_xmax)
        else:
            active = hw_grid[s_norm >= 0.008]
            if anharmonic_dict:
                for T, data in anharmonic_dict.items():
                    vdos = data.get("vdos")
                    if vdos is not None and len(vdos) == len(hw_grid):
                        active = np.concatenate([active, hw_grid[vdos >= 0.008]])

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

        y_min = 0.0
        y_max = 1.18  # Headroom for legend
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
        y_min_final = float(user_ymin) if user_ymin is not None else (-0.015 * peak_I)
        y_max_final = float(user_ymax) if user_ymax is not None else (1.15 * peak_I)

        return (e_min_final, e_max_final), (y_min_final, y_max_final)

    def plot_all(self, lineshape_dict, hw_grid, s_cont, total_S, anharmonic_dict=None):
        """Dispatches combined and/or individual temperature plots according to `temperature_plot_mode`."""
        plot_mode = self.config.temperature_plot_mode.lower()

        if plot_mode in ("combined", "both"):
            self.plot_combined(lineshape_dict, hw_grid, s_cont, total_S, anharmonic_dict=anharmonic_dict)

        if plot_mode in ("individual", "both"):
            self.plot_individual(lineshape_dict, hw_grid, s_cont, total_S, anharmonic_dict=anharmonic_dict)

    def plot_combined(self, lineshape_dict, hw_grid, s_cont, total_S, anharmonic_dict=None):
        """Plots a combined multi-temperature figure of phonon densities and lineshapes."""
        fig, axs = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
        show_harmonic = self.config.show_harmonic_in_anharmonic

        temp_colors = {T: _COLOR_CYCLE[i % len(_COLOR_CYCLE)] for i, T in enumerate(self.config.temperatures)}

        ax = axs[0]
        s_norm = s_cont / np.max(s_cont) if np.max(s_cont) > 0 else s_cont

        if anharmonic_dict:
            for T in self.config.temperatures:
                if T in anharmonic_dict:
                    col = temp_colors.get(T, '#1f77b4')
                    vdos = anharmonic_dict[T]["vdos"]
                    t_str = f"{int(T)} K" if float(T).is_integer() else f"{T} K"
                    ax.fill_between(hw_grid, 0, vdos, color=col, alpha=0.22, label=f'Anharmonic VDOS ({t_str})')
                    ax.plot(hw_grid, vdos, color=col, lw=2.0)

            if show_harmonic:
                ax.plot(hw_grid, s_norm, color='black', lw=2.2, linestyle='--', label=r'Harmonic $S(\hbar\omega)$ ($0\ \mathrm{K}$)')
        else:
            ax.fill_between(hw_grid, 0, s_norm, color='#2196F3', alpha=0.25, label=r'Harmonic $S(\hbar\omega)$ ($0\ \mathrm{K}$)')
            ax.plot(hw_grid, s_norm, color='#1976D2', lw=2.2)

        hw_xlim, hw_ylim = self.get_smart_phonon_limits(hw_grid, s_norm, anharmonic_dict=anharmonic_dict)
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

        plt.tight_layout()
        fig_name = self.config.anharmonic_figure_name if anharmonic_dict else self.config.lineshape_figure_name
        out_fig = os.path.join(self.config.workspace_dir, fig_name)
        plt.savefig(out_fig, dpi=300)
        plt.close(fig)
        logger.info(f"[+] Combined lineshape figure saved: {out_fig} (X-limits: {e_xlim[0]:.1f} to {e_xlim[1]:.1f} meV)")

    def plot_individual(self, lineshape_dict, hw_grid, s_cont, total_S, anharmonic_dict=None):
        """Plots per-temperature figures of phonon densities and lineshapes."""
        show_harmonic = self.config.show_harmonic_in_anharmonic
        prefix = self.config.individual_figure_prefix
        s_norm = s_cont / np.max(s_cont) if np.max(s_cont) > 0 else s_cont

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
            hw_xlim, hw_ylim = self.get_smart_phonon_limits(hw_grid, s_norm, anharmonic_dict=vdos_sub)

            if anharmonic_dict and T in anharmonic_dict:
                vdos = anharmonic_dict[T]["vdos"]
                ax.fill_between(hw_grid, 0, vdos, color=col, alpha=0.28, label=f'Anharmonic VDOS ({t_str})')
                ax.plot(hw_grid, vdos, color=col, lw=2.2)
                if show_harmonic:
                    ax.plot(hw_grid, s_norm, color='black', lw=2.0, linestyle='--', label=r'Harmonic $S(\hbar\omega)$ ($0\ \mathrm{K}$)')
            else:
                ax.fill_between(hw_grid, 0, s_norm, color=col, alpha=0.25, label=r'Harmonic $S(\hbar\omega)$')
                ax.plot(hw_grid, s_norm, color=col, lw=2.2)

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
        return engine.calculate_cc_curve(
            at_gs, at_es,
            lambda_min=lambda_min,
            lambda_max=lambda_max,
            n_points=n_points,
            zpl_energy_eV=zpl_energy_eV
        )

    def plot_configuration_coordinate(
        self, at_gs, at_es, output_path=None, zpl_energy_eV=None, system_name=None
    ):
        """Calculates 1D configuration coordinate curves and renders the 2-panel diagram."""
        from .cc_diagram import ConfigurationCoordinateEngine
        engine = ConfigurationCoordinateEngine(self.config, self.calc)
        cc_data = engine.calculate_cc_curve(at_gs, at_es, zpl_energy_eV=zpl_energy_eV)
        fig_path = engine.plot_configuration_coordinate(cc_data, output_path=output_path, system_name=system_name)
        cc_data["figure_path"] = fig_path
        return cc_data
