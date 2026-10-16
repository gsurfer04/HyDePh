"""
hydeph.lineshape
================
Computes normal modes, electron-phonon spectral density S(hbar omega),
total Huang-Rhys factor S, and optical lineshapes A(E, T) across user temperatures.
Includes both harmonic generating-function lineshapes and finite-temperature
anharmonic molecular dynamics free energy sampling.
"""

import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ase.vibrations import Vibrations
from ase.units import _hbar, _e, _amu, _me
from scipy.signal import fftconvolve

HBAR_eV_fs = 0.6582119514
KB_eV_K = 8.617333262e-5
KB_meV_K = KB_eV_K * 1000.0


def compute_quantum_lineshape(
    hw_grid_meV,
    s_cont_per_eV,
    temp_k,
    dephasing_gamma_meV=2.0,
    e_min_meV=None,
    e_max_meV=None,
    n_e=1500,
):
    """Computes normalized emission lineshape A(E - E_ZPL, T) via generating function."""
    mask = (hw_grid_meV > 0.1) & (s_cont_per_eV > 0.0)
    hw = hw_grid_meV[mask]
    s_val = s_cont_per_eV[mask] / 1000.0
    dhw = hw[1] - hw[0] if len(hw) > 1 else 1.0

    # Adaptively determine energy range if not explicitly provided
    if e_min_meV is None or e_max_meV is None:
        total_s = np.sum(s_val * dhw)
        mean_hw = np.sum(hw * s_val * dhw) / total_s if total_s > 0 else 35.0
        erel = total_s * mean_hw
        sigma_fc = np.sqrt(max(1.0, total_s)) * mean_hw
        if e_min_meV is None:
            e_min_meV = -max(250.0, erel + 4.5 * sigma_fc)
        if e_max_meV is None:
            e_max_meV = max(60.0, 2.5 * mean_hw)

    if temp_k < 0.1:
        n_bose = np.zeros_like(hw)
    else:
        n_bose = 1.0 / (np.exp(hw / (KB_meV_K * temp_k)) - 1.0)

    t_fs = np.linspace(-6000.0, 6000.0, 16385)
    dt = t_fs[1] - t_fs[0]

    w_fs = hw / (1000.0 * HBAR_eV_fs)
    wt = np.outer(w_fs, t_fs)
    integrand = (
        (n_bose[:, None] + 1.0) * np.exp(1j * wt)
        + n_bose[:, None] * np.exp(-1j * wt)
        - (2.0 * n_bose[:, None] + 1.0)
    )
    g_t = np.sum(s_val[:, None] * integrand * dhw, axis=0)
    G_t = np.exp(g_t)

    gamma_fs_inv = (dephasing_gamma_meV / 1000.0) / HBAR_eV_fs
    damping = np.exp(-gamma_fs_inv * np.abs(t_fs) - 0.5 * (t_fs / 2000.0)**2)

    energy_grid = np.linspace(e_min_meV, e_max_meV, n_e)
    w_out = (energy_grid / 1000.0) / HBAR_eV_fs

    integrand_ft = G_t * damping
    ft_matrix = np.exp(1j * np.outer(w_out, t_fs))
    a_unnorm = np.real(ft_matrix @ integrand_ft) * dt / (2.0 * np.pi * HBAR_eV_fs)
    a_unnorm = np.maximum(0.0, a_unnorm)

    total_area = np.trapezoid(a_unnorm, energy_grid) if hasattr(np, "trapezoid") else np.trapz(a_unnorm, energy_grid)
    a_norm = a_unnorm / total_area if total_area > 0 else a_unnorm
    return energy_grid, a_norm


class LineshapeEngine:
    def __init__(self, config, hybrid_calc):
        self.config = config
        self.calc = hybrid_calc

    def calculate_spectral_density_and_lineshapes(self, at_gs, at_es):
        """Calculates normal modes, spectral density S(hw), and multi-T lineshapes."""
        print("\n[-] Evaluating Vibrational Normal Modes on Hybrid Potential...")
        dr = at_es.positions - at_gs.positions
        m = at_gs.get_masses()
        dq_total = np.sqrt(np.sum(np.repeat(m, 3) * dr.flatten()**2))
        
        # Finite-difference Hessian around ground state
        vib_dir = os.path.join(self.config.workspace_dir, "vib")
        os.makedirs(vib_dir, exist_ok=True)
        
        at = at_gs.copy()
        at.calc = self.calc
        vib = Vibrations(at, name=os.path.join(vib_dir, "vib"), delta=0.01)
        vib.run()
        
        energies_ev = vib.get_energies()
        
        # Filter real positive vibrational modes and project mass-weighted displacements
        valid_indices = [i for i, e in enumerate(energies_ev) if np.real(e) * 1000.0 > 1.0 and abs(np.imag(e)) < 1e-4]
        hw_modes = []
        sk_list = []
        for idx in valid_indices:
            hw_k = np.real(energies_ev[idx]) * 1000.0
            mode_disp = vib.get_mode(idx)
            dq_k = np.sum(m[:, None] * dr * mode_disp)
            # Exact formula: S_k = [omega_k / (2*hbar)] * (Delta Q_k)^2
            # With hw_k in meV, factor = 0.1196127 meV^-1 * amu^-1 * A^-2
            s_k = 0.1196127 * hw_k * (dq_k ** 2)
            hw_modes.append(hw_k)
            sk_list.append(s_k)
            
        hw_modes = np.array(hw_modes)
        sk_modes = np.array(sk_list)
        total_S = np.sum(sk_modes)
        print(f"[+] Total Huang-Rhys Factor S: {total_S:.3f} (Franck-Condon Delta Q = {dq_total:.4f} amu^(1/2)*A)")
        
        # Build continuous spectral density S(hbar omega)
        max_mode = float(np.max(hw_modes)) if len(hw_modes) > 0 else 65.0
        user_hw_max = getattr(self.config, "phonon_hw_max", None)
        hw_max = user_hw_max if user_hw_max is not None else max(70.0, max_mode * 1.05)
        n_hw_pts = max(600, int(np.ceil(hw_max * 10)))
        hw_grid = np.linspace(0.5, hw_max, n_hw_pts)
        # Finite-size smearing width to bridge discrete supercell mode spacing
        user_smear = getattr(self.config, "phonon_smearing_meV", None)
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

        print(f"[+] Phonon Spectral Smearing (Finite-Size Correction): sigma = {sigma_smear:.2f} meV")
        s_cont = np.zeros_like(hw_grid)
        for w, s in zip(hw_modes, sk_modes):
            s_cont += s * np.exp(-0.5 * ((hw_grid - w) / sigma_smear)**2) / (np.sqrt(2.0 * np.pi) * sigma_smear)

        # Enforce physical acoustic scaling: S(hw) -> 0 as hw -> 0 to avoid boundary artifacts
        if getattr(self.config, "finite_size_acoustic_cutoff", True):
            s_cont *= np.tanh((hw_grid / max(1.0, sigma_smear))**2)

        s_cont *= 1000.0 # to eV^-1
        
        # Determine energy window for lineshapes
        e_min = getattr(self.config, "lineshape_e_min", None)
        e_max = getattr(self.config, "lineshape_e_max", None)
        n_pts = getattr(self.config, "lineshape_n_points", 1500)
        if e_min is None or e_max is None:
            mean_hw = np.sum(hw_modes * sk_modes) / total_S if total_S > 0 else 35.0
            erel = total_S * mean_hw
            sigma_fc = np.sqrt(max(1.0, total_S)) * mean_hw
            if e_min is None:
                e_min = -max(250.0, erel + 4.5 * sigma_fc)
            if e_max is None:
                e_max = max(60.0, 2.5 * mean_hw)

        # Compute harmonic lineshapes across user temperatures
        print(f"\n[-] Computing Harmonic Optical Lineshapes across temperatures: {self.config.temperatures} K (Range: [{e_min:.1f}, {e_max:.1f}] meV)...")
        lineshape_dict = {}
        for T in self.config.temperatures:
            user_gamma = getattr(self.config, "dephasing_gamma_meV", None)
            if user_gamma is not None:
                gamma = float(user_gamma)
            else:
                gamma = 1.5 + 2.0e-4 * (T**2)
            eg, a_norm = compute_quantum_lineshape(
                hw_grid, s_cont, temp_k=T, dephasing_gamma_meV=gamma,
                e_min_meV=e_min, e_max_meV=e_max, n_e=n_pts
            )
            lineshape_dict[T] = (eg, a_norm)
            print(f"    [+] Harmonic T = {T:5.1f} K | Peak A(E) = {np.max(a_norm):.4f} | ZPL DWF = {np.exp(-total_S)*100.0:.2f}%")
            
        # Optional Anharmonic Sampling
        anharmonic_dict = None
        if getattr(self.config, "enable_anharmonic", True):
            anharmonic_dict = self.run_anharmonic_sampling(
                at_gs, at_es, hw_grid, s_cont, total_S=total_S, e_min=e_min, e_max=e_max, n_pts=n_pts
            )
            self.plot_all(lineshape_dict, hw_grid, s_cont, total_S, anharmonic_dict=anharmonic_dict)
        else:
            self.plot_all(lineshape_dict, hw_grid, s_cont, total_S, anharmonic_dict=None)

        # Optional 1D Configuration Coordinate & Stokes Shift Analysis
        cc_dict = None
        if getattr(self.config, "plot_configuration_coordinate", True):
            try:
                cc_dict = self.plot_configuration_coordinate(at_gs, at_es)
            except Exception as e:
                print(f"[!] Warning: Configuration coordinate calculation encountered an issue: {e}")

        return {
            "harmonic": lineshape_dict,
            "anharmonic": anharmonic_dict,
            "total_S": total_S,
            "hw_grid": hw_grid,
            "s_cont": s_cont,
            "cc_analysis": cc_dict,
        }

    def run_anharmonic_sampling(
        self, at_gs, at_es, hw_grid, s_cont, total_S=None, e_min=None, e_max=None, n_pts=1500
    ):
        """
        Runs finite-temperature Langevin MD on the hybrid potential across user temperatures.

        For intermediate/weak coupling (S < 10, e.g. SiC divacancies, diamond NV):
          - Projects velocities onto the optical Franck-Condon reaction coordinate u_opt.
          - Computes projected VDOS g_opt(hw, T) to capture mode softening alpha_soft(T)
            and intrinsic anharmonic dephasing without bulk optical phonon artifacts.
          - Evaluates optical lineshape via the finite-temperature quantum generating function,
            preserving the sharp ZPL and structured phonon sidebands observed experimentally.

        For strong coupling (S >= 10, e.g. F-centers with massive lattice relaxation):
          - Semiclassical Marcus energy-gap fluctuation convolution is applied for broad classical bands.
        """
        from ase.md.langevin import Langevin
        from ase import units

        print("\n[-] Running Finite-Temperature Anharmonic Molecular Dynamics Sampling...")
        dr_fc = at_es.positions - at_gs.positions
        m = at_gs.get_masses()
        dt_fs = getattr(self.config, "anharmonic_dt_fs", 1.0)
        n_steps = getattr(self.config, "anharmonic_md_steps", 2000)
        trapz_fn = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

        if total_S is None:
            dhw = hw_grid[1] - hw_grid[0] if len(hw_grid) > 1 else 1.0
            total_S = float(np.sum(s_cont * dhw) / 1000.0)

        # Coupling regime: 'auto' (< 10 -> quantum, >= 10 -> semiclassical), 'quantum', or 'semiclassical'
        regime_opt = getattr(self.config, "anharmonic_coupling_regime", "auto").lower()
        if regime_opt == "auto":
            is_quantum = (total_S < 10.0)
        elif regime_opt == "quantum":
            is_quantum = True
        else:
            is_quantum = False

        # Mass-weighted optical projection unit vector:
        # u_opt,i = (M_i * Delta R_FC,i) / sqrt(sum_j M_j * |Delta R_FC,j|^2)
        norm_dr = np.sqrt(np.sum(m[:, None] * (dr_fc**2)))
        if norm_dr > 1e-8:
            u_opt = (m[:, None] * dr_fc) / norm_dr
        else:
            u_opt = np.ones_like(dr_fc) / np.sqrt(len(dr_fc) * 3)

        # Baseline harmonic effective frequency Omega_0
        active_mask = s_cont > 0.01 * np.max(s_cont) if np.max(s_cont) > 0 else np.ones(len(hw_grid), dtype=bool)
        if np.any(active_mask):
            omega_0 = float(trapz_fn(hw_grid[active_mask] * s_cont[active_mask], hw_grid[active_mask]) / trapz_fn(s_cont[active_mask], hw_grid[active_mask]))
        else:
            omega_0 = 35.0

        # Baseline zero-temperature quantum lineshape
        eg_base, a_quant_0K = compute_quantum_lineshape(
            hw_grid, s_cont, temp_k=0.01, dephasing_gamma_meV=1.0,
            e_min_meV=e_min, e_max_meV=e_max, n_e=n_pts
        )
        dE = eg_base[1] - eg_base[0]

        anharmonic_dict = {}
        for T in self.config.temperatures:
            t0 = time.time()
            at = at_gs.copy()
            at.calc = self.calc
            dyn = Langevin(at, dt_fs * units.fs, temperature_K=T, friction=0.01 / units.fs, logfile=None)
            
            # Equilibrate
            dyn.run(300)
            
            du_list = []
            v_opt_list = []
            def step_observer():
                # Projected velocity along optical coordinate
                v = at.get_velocities()
                v_opt = float(np.sum(v * u_opt))
                v_opt_list.append(v_opt)

                # Energy gap Delta U(t)
                ug = at.get_potential_energy()
                at_temp = at.copy()
                at_temp.positions += dr_fc
                at_temp.calc = self.calc
                ue = at_temp.get_potential_energy()
                du_list.append((ue - ug) * 1000.0) # meV

            dyn.attach(step_observer, interval=2)
            dyn.run(n_steps)

            du = np.array(du_list)
            # Remove any low-frequency trajectory drift to isolate high-frequency thermal fluctuations
            if len(du) > 10:
                from scipy.signal import detrend
                du_fluc = detrend(du)
                mad = np.median(np.abs(du_fluc - np.median(du_fluc)))
                valid = np.abs(du_fluc) < 4.0 * max(mad, 10.0)
                if np.sum(valid) > 10:
                    sigma_du = float(np.std(du_fluc[valid]))
                else:
                    sigma_du = float(np.std(du_fluc))
            else:
                sigma_du = float(np.std(du))

            # Optical-projected VACF and VDOS along u_opt
            v_arr = np.array(v_opt_list)
            n_v = len(v_arr)
            vacf = np.zeros(n_v // 2)
            for lag in range(len(vacf)):
                vacf[lag] = np.mean(v_arr[:n_v - lag] * v_arr[lag:])
            vacf /= (vacf[0] + 1e-12)

            t_vacf = np.arange(len(vacf)) * (2.0 * dt_fs) # dt between samples = 2 fs
            hann = 0.5 * (1.0 + np.cos(np.pi * t_vacf / t_vacf[-1]))
            vdos = []
            for w_mev in hw_grid:
                w_rad_fs = (w_mev / 1000.0) / HBAR_eV_fs
                val = trapz_fn(vacf * hann * np.cos(w_rad_fs * t_vacf), t_vacf)
                vdos.append(max(0.0, val))
            vdos = np.array(vdos)
            if np.max(vdos) > 0:
                vdos /= np.max(vdos)

            if is_quantum:
                # 1. Mode softening ratio alpha_soft(T) = Omega(T) / Omega_0
                mask_active = (hw_grid >= max(10.0, 0.4 * omega_0)) & (hw_grid <= min(float(hw_grid[-1]), 1.8 * omega_0))
                if np.sum(vdos[mask_active]) > 1e-4:
                    omega_T = float(trapz_fn(hw_grid[mask_active] * vdos[mask_active], hw_grid[mask_active]) / trapz_fn(vdos[mask_active], hw_grid[mask_active]))
                    alpha_soft = float(np.clip(omega_T / omega_0, 0.80, 1.20))
                else:
                    omega_T = omega_0
                    alpha_soft = 1.0

                # Scale spectral density by alpha_soft (conserving total Huang-Rhys factor)
                hw_shifted = hw_grid / alpha_soft
                s_anh = np.interp(hw_shifted, hw_grid, s_cont, left=0.0, right=0.0) / alpha_soft

                # Thermal dephasing
                user_gamma = getattr(self.config, "dephasing_gamma_meV", None)
                if user_gamma is not None:
                    gamma_base = float(user_gamma)
                else:
                    gamma_base = 1.5 + 2.0e-4 * (T**2)

                # Anharmonic phonon-phonon scattering dephasing
                delta_gamma_anh = 0.3 * np.sqrt(max(1.0, T) / 10.0)
                gamma_total = gamma_base + delta_gamma_anh

                eg_out, a_anh = compute_quantum_lineshape(
                    hw_grid, s_anh, temp_k=T, dephasing_gamma_meV=gamma_total,
                    e_min_meV=e_min, e_max_meV=e_max, n_e=n_pts
                )
                regime_tag = f"Quantum GenFn (alpha_soft={alpha_soft:.3f}, gamma={gamma_total:.2f} meV)"
            else:
                # Semiclassical Gaussian convolution with energy gap fluctuations sigma_du
                erel_est = abs(eg_base[np.argmax(a_quant_0K)])
                sig_scale = np.sqrt(2.0 * max(1.0, KB_meV_K * T) * max(50.0, erel_est))
                sigma_du = min(max(sigma_du, 1.0), 2.0 * sig_scale)
                
                kw = max(5.0 * sigma_du, 50.0)
                n_pts_k = int(np.ceil(kw / dE))
                tau_grid = np.arange(-n_pts_k, n_pts_k + 1) * dE
                kernel = np.exp(-0.5 * (tau_grid / max(1.0, sigma_du))**2)
                kernel /= np.sum(kernel)

                a_anh = fftconvolve(a_quant_0K, kernel, mode='same')
                total_area = trapz_fn(a_anh, eg_base)
                if total_area > 0:
                    a_anh /= total_area
                eg_out = eg_base
                alpha_soft = 1.0
                omega_T = omega_0
                regime_tag = f"Semiclassical Marcus (sigma={sigma_du:.1f} meV)"

            # Calculate FWHM
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
                "walltime_s": time.time() - t0
            }
            print(f"    [+] Anharmonic MD T = {T:5.1f} K | {regime_tag} | Lineshape FWHM = {fwhm_meV:5.1f} meV ({time.time()-t0:.1f} s)")

        return anharmonic_dict

    def get_smart_phonon_limits(self, hw_grid, s_norm, anharmonic_dict=None):
        """Determines optimal [x_min, x_max] and [y_min, y_max] for phonon density plots."""
        if not getattr(self.config, "smart_axis_scaling", True):
            return (float(hw_grid[0]), float(hw_grid[-1])), (0.0, 1.05)

        user_xmin = getattr(self.config, "phonon_hw_min", None)
        user_xmax = getattr(self.config, "phonon_hw_max", None)

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
        user_emin = getattr(self.config, "lineshape_e_min", None)
        user_emax = getattr(self.config, "lineshape_e_max", None)
        user_ymin = getattr(self.config, "lineshape_y_min", None)
        user_ymax = getattr(self.config, "lineshape_y_max", None)

        if not getattr(self.config, "smart_axis_scaling", True):
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
            # Active threshold: 2% of peak for each evaluated curve to filter asymptotic dephasing tails
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

                # Anti-Stokes side (right, E > 0): always leave room for ZPL (+35 to +60 meV)
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
        """Unified plotting dispatcher honoring temperature_plot_mode ('combined', 'individual', 'both')."""
        plot_mode = getattr(self.config, "temperature_plot_mode", "both").lower()
        
        if plot_mode in ("combined", "both"):
            self.plot_combined(lineshape_dict, hw_grid, s_cont, total_S, anharmonic_dict=anharmonic_dict)
            
        if plot_mode in ("individual", "both"):
            self.plot_individual(lineshape_dict, hw_grid, s_cont, total_S, anharmonic_dict=anharmonic_dict)

    def plot_combined(self, lineshape_dict, hw_grid, s_cont, total_S, anharmonic_dict=None):
        """Plots combined multi-temperature figure (phonon densities and lineshapes) with smart scaling."""
        fig, axs = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
        show_harmonic = getattr(self.config, "show_harmonic_in_anharmonic", True)

        color_cycle = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728', '#9467bd', '#8c564b', '#e377c2']
        temp_colors = {T: color_cycle[i % len(color_cycle)] for i, T in enumerate(self.config.temperatures)}

        # Panel (a): Phonon Density & Electron-Phonon Coupling
        ax = axs[0]
        s_norm = s_cont / np.max(s_cont) if np.max(s_cont) > 0 else s_cont

        if anharmonic_dict:
            # Anharmonic: curve is filled
            for T in self.config.temperatures:
                if T in anharmonic_dict:
                    col = temp_colors.get(T, '#1f77b4')
                    vdos = anharmonic_dict[T]["vdos"]
                    t_str = f"{int(T)} K" if float(T).is_integer() else f"{T} K"
                    ax.fill_between(hw_grid, 0, vdos, color=col, alpha=0.22, label=f'Anharmonic VDOS ({t_str})')
                    ax.plot(hw_grid, vdos, color=col, lw=2.0)
            
            # Harmonic data optionally graphed as dashed line
            if show_harmonic:
                ax.plot(hw_grid, s_norm, color='black', lw=2.2, linestyle='--', label=r'Harmonic $S(\hbar\omega)$ ($0\ \mathrm{K}$)')
        else:
            # Harmonic only: filled curve
            ax.plot(hw_grid, s_norm, color='#1f77b4', lw=2.2, label=r'Harmonic $S(\hbar\omega)$ ($0\ \mathrm{K}$)')
            ax.fill_between(hw_grid, 0, s_norm, color='#1f77b4', alpha=0.25)

        # Smart scaling for Panel (a)
        hw_xlim, hw_ylim = self.get_smart_phonon_limits(hw_grid, s_norm, anharmonic_dict=anharmonic_dict)
        ax.set_xlim(hw_xlim)
        ax.set_ylim(hw_ylim)

        ax.set_xlabel(r'Phonon Frequency $\hbar\omega$ (meV)', fontsize=11)
        ax.set_ylabel(r'Normalized Density / Coupling', fontsize=11)
        ax.set_title(r'(a) Phonon Density & Electron-Phonon Coupling', loc='left', fontweight='bold', fontsize=11.5)
        ax.grid(True, linestyle=':', alpha=0.5)
        ax.legend(frameon=True, loc='upper right', framealpha=0.9, fontsize=9.2)

        # Panel (b): Optical Emission Lineshapes
        ax2 = axs[1]
        for T, (eg, an_harm) in lineshape_dict.items():
            col = temp_colors.get(T, '#1f77b4')
            t_str = f"{int(T)} K" if float(T).is_integer() else f"{T} K"
            
            if anharmonic_dict and T in anharmonic_dict:
                an_anh = anharmonic_dict[T]["lineshape"]
                ax2.plot(eg, an_anh, color=col, lw=2.2, linestyle='-', label=f'Anh. {t_str}')
                if show_harmonic:
                    ax2.plot(eg, an_harm, color=col, lw=1.5, linestyle='--', label=f'Harm. {t_str}')
            else:
                ax2.plot(eg, an_harm, color=col, lw=2.0, linestyle='-', label=f'{t_str}')

        # Smart scaling for Panel (b)
        e_xlim, e_ylim = self.get_smart_lineshape_limits(lineshape_dict, anharmonic_dict=anharmonic_dict)
        ax2.set_xlim(e_xlim)
        if getattr(self.config, "lineshape_y_scale", "linear").lower() == "log":
            ax2.set_yscale("log")
            ax2.set_ylim(bottom=max(1e-6, e_ylim[1] * 1e-4), top=e_ylim[1] * 1.5)
        else:
            ax2.set_ylim(e_ylim)

        ax2.axvline(0.0, color='gray', linestyle=':', lw=1.2, label=r'ZPL ($E=E_{\mathrm{ZPL}}$)')
        ax2.set_xlabel(r'Energy Relative to ZPL: $E - E_{\mathrm{ZPL}}$ (meV)', fontsize=11)
        ax2.set_ylabel(r'Normalized Emission Intensity', fontsize=11)
        ax2.set_title(r'(b) Multi-Temperature Optical Emission Lineshapes $A(E - E_{\mathrm{ZPL}}, T)$', loc='left', fontweight='bold', fontsize=11.5)
        ax2.grid(True, linestyle=':', alpha=0.5)
        ax2.legend(frameon=True, loc='upper left', framealpha=0.9, fontsize=8.8, ncol=2 if len(lineshape_dict) > 2 else 1)

        plt.tight_layout()
        fig_name = getattr(self.config, "anharmonic_figure_name", "anharmonic_lineshapes.png") if anharmonic_dict else getattr(self.config, "lineshape_figure_name", "lineshapes.png")
        out_fig = os.path.join(self.config.workspace_dir, fig_name)
        plt.savefig(out_fig, dpi=300)
        plt.close(fig)
        print(f"[+] Combined lineshape figure saved: {out_fig} (X-limits: {e_xlim[0]:.1f} to {e_xlim[1]:.1f} meV)")

    def plot_individual(self, lineshape_dict, hw_grid, s_cont, total_S, anharmonic_dict=None):
        """Plots individual clean figures for each temperature with smart scaling."""
        show_harmonic = getattr(self.config, "show_harmonic_in_anharmonic", True)
        prefix = getattr(self.config, "individual_figure_prefix", "lineshape")
        s_norm = s_cont / np.max(s_cont) if np.max(s_cont) > 0 else s_cont

        color_cycle = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728', '#9467bd', '#8c564b', '#e377c2']
        temp_colors = {T: color_cycle[i % len(color_cycle)] for i, T in enumerate(self.config.temperatures)}

        # Global X-range so all temperature figures share the exact same energy scale
        global_e_xlim, _ = self.get_smart_lineshape_limits(lineshape_dict, anharmonic_dict=anharmonic_dict)

        for T in self.config.temperatures:
            if T not in lineshape_dict:
                continue
            eg, an_harm = lineshape_dict[T]
            col = temp_colors.get(T, '#1f77b4')
            t_str = f"{int(T)} K" if float(T).is_integer() else f"{T} K"

            fig, axs = plt.subplots(1, 2, figsize=(14, 5.0), dpi=300)

            # Panel (a): Phonon Density at T
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

            # Panel (b): Optical Lineshape at T
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

            # Smart limits for temperature T
            e_xlim_T, e_ylim_T = self.get_smart_lineshape_limits(
                lineshape_dict, anharmonic_dict=anharmonic_dict, active_curve_subset=curves_T
            )
            share_x = getattr(self.config, "individual_shared_x", False)
            ax2.set_xlim(global_e_xlim if share_x else e_xlim_T)
            if getattr(self.config, "lineshape_y_scale", "linear").lower() == "log":
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
            print(f"[+] Individual lineshape figure saved: {out_file} (X-limits: {xlim_used[0]:.1f} to {xlim_used[1]:.1f} meV)")


    def plot_anharmonic_comparison(self, lineshape_dict, anharmonic_dict, hw_grid, s_cont, total_S):
        """Backward-compatible alias for plot_all with anharmonic results."""
        self.plot_all(lineshape_dict, hw_grid, s_cont, total_S, anharmonic_dict=anharmonic_dict)

    def plot_lineshapes(self, lineshape_dict, hw_grid, s_cont, total_S):
        """Backward-compatible alias for plot_all with harmonic results."""
        self.plot_all(lineshape_dict, hw_grid, s_cont, total_S, anharmonic_dict=None)

    def calculate_configuration_coordinate(
        self, at_gs, at_es, zpl_energy_eV=None, lambda_min=None, lambda_max=None, n_points=None
    ):
        """
        Calculates 1D potential energy curves along the configuration coordinate.
        
        Returns
        -------
        dict
            Dictionary of coordinates, energies, relaxation parameters, and Stokes shift metrics.
        """
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
        """
        Calculates 1D CC potential energy curves and renders publication-grade 2-panel figure.
        
        Returns
        -------
        dict
            Dictionary containing CC parameters and 'figure_path' key.
        """
        from .cc_diagram import ConfigurationCoordinateEngine
        engine = ConfigurationCoordinateEngine(self.config, self.calc)
        cc_data = engine.calculate_cc_curve(at_gs, at_es, zpl_energy_eV=zpl_energy_eV)
        fig_path = engine.plot_configuration_coordinate(cc_data, output_path=output_path, system_name=system_name)
        cc_data["figure_path"] = fig_path
        return cc_data

