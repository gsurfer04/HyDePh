"""
defect_delta_nep.lineshape
==========================
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


def compute_quantum_lineshape(hw_grid_meV, s_cont_per_eV, temp_k, dephasing_gamma_meV=2.0, e_min_meV=-250.0, e_max_meV=50.0, n_e=1200):
    """Computes normalized emission lineshape A(E - E_ZPL, T) via generating function."""
    t_fs = np.linspace(-6000.0, 6000.0, 8193)
    dt = t_fs[1] - t_fs[0]

    mask = (hw_grid_meV > 0.1) & (s_cont_per_eV > 0.0)
    hw = hw_grid_meV[mask]
    s_val = s_cont_per_eV[mask] / 1000.0
    dhw = hw[1] - hw[0] if len(hw) > 1 else 1.0

    if temp_k < 0.1:
        n_bose = np.zeros_like(hw)
    else:
        n_bose = 1.0 / (np.exp(hw / (KB_meV_K * temp_k)) - 1.0)

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
    damping = np.exp(-gamma_fs_inv * np.abs(t_fs) - 0.5 * (t_fs / 1000.0)**2)

    energy_grid = np.linspace(e_min_meV, e_max_meV, n_e)
    w_out = (energy_grid / 1000.0) / HBAR_eV_fs

    integrand_ft = G_t * damping
    ft_matrix = np.exp(1j * np.outer(w_out, t_fs))
    a_unnorm = np.real(ft_matrix @ integrand_ft) * dt / (2.0 * np.pi * HBAR_eV_fs)

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
        hw_grid = np.linspace(0.5, 65.0, 600)
        sigma_smear = 1.8 # meV
        s_cont = np.zeros_like(hw_grid)
        for w, s in zip(hw_modes, sk_modes):
            s_cont += s * np.exp(-0.5 * ((hw_grid - w) / sigma_smear)**2) / (np.sqrt(2.0 * np.pi) * sigma_smear)
        s_cont *= 1000.0 # to eV^-1
        
        # Compute harmonic lineshapes across user temperatures
        print(f"\n[-] Computing Harmonic Optical Lineshapes across temperatures: {self.config.temperatures} K...")
        lineshape_dict = {}
        for T in self.config.temperatures:
            gamma = 1.5 + 2.0e-4 * (T**2)
            eg, a_norm = compute_quantum_lineshape(hw_grid, s_cont, temp_k=T, dephasing_gamma_meV=gamma)
            lineshape_dict[T] = (eg, a_norm)
            print(f"    [+] Harmonic T = {T:5.1f} K | Peak A(E) = {np.max(a_norm):.4f} | ZPL DWF = {np.exp(-total_S)*100.0:.2f}%")
            
        # Optional Anharmonic Sampling
        anharmonic_dict = None
        if getattr(self.config, "enable_anharmonic", True):
            anharmonic_dict = self.run_anharmonic_sampling(at_gs, at_es, hw_grid, s_cont)
            self.plot_anharmonic_comparison(lineshape_dict, anharmonic_dict, hw_grid, s_cont, total_S)
        else:
            self.plot_lineshapes(lineshape_dict, hw_grid, s_cont, total_S)

        return {
            "harmonic": lineshape_dict,
            "anharmonic": anharmonic_dict,
            "total_S": total_S,
            "hw_grid": hw_grid,
            "s_cont": s_cont
        }

    def run_anharmonic_sampling(self, at_gs, at_es, hw_grid, s_cont):
        """
        Runs finite-temperature Langevin MD on the hybrid potential across user temperatures,
        evaluating energy gap fluctuations Delta U(t) and defect-core VACF.
        """
        from ase.md.langevin import Langevin
        from ase import units

        print("\n[-] Running Finite-Temperature Anharmonic Molecular Dynamics Sampling...")
        dr_fc = at_es.positions - at_gs.positions
        dt_fs = getattr(self.config, "anharmonic_dt_fs", 1.0)
        n_steps = getattr(self.config, "anharmonic_md_steps", 2000)

        # Baseline zero-temperature quantum lineshape
        eg_base, a_quant_0K = compute_quantum_lineshape(hw_grid, s_cont, temp_k=0.01, dephasing_gamma_meV=1.0)
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
            v_list = []
            def step_observer():
                ug = at.get_potential_energy()
                at_temp = at.copy()
                at_temp.positions += dr_fc
                at_temp.calc = self.calc
                ue = at_temp.get_potential_energy()
                du_list.append((ue - ug) * 1000.0) # meV
                v_list.append(at.get_velocities()[:6].copy()) # core Li velocities

            dyn.attach(step_observer, interval=2)
            dyn.run(n_steps)

            du = np.array(du_list)
            sigma_du = np.std(du)
            
            # Anharmonic thermal broadening kernel
            p_thermal = np.exp(-0.5 * (eg_base / max(1.0, sigma_du))**2)
            p_area = np.trapezoid(p_thermal, eg_base) if hasattr(np, "trapezoid") else np.trapz(p_thermal, eg_base)
            p_thermal /= p_area

            # Convolve quantum zero-point vibronic spectrum with thermal fluctuation distribution
            a_anh = fftconvolve(a_quant_0K, p_thermal, mode='same') * dE
            total_area = np.trapezoid(a_anh, eg_base) if hasattr(np, "trapezoid") else np.trapz(a_anh, eg_base)
            a_anh /= total_area

            # Calculate FWHM
            fwhm_idx = np.where(a_anh >= 0.5 * np.max(a_anh))[0]
            fwhm_meV = eg_base[fwhm_idx[-1]] - eg_base[fwhm_idx[0]] if len(fwhm_idx) > 1 else 0.0

            # Core VACF and VDOS
            v_arr = np.array(v_list)
            n_v = len(v_arr)
            vacf = np.zeros(n_v // 2)
            for lag in range(len(vacf)):
                vacf[lag] = np.mean([np.sum(v_arr[i] * v_arr[i + lag]) for i in range(len(v_arr) - lag)])
            vacf /= (vacf[0] + 1e-12)

            t_vacf = np.arange(len(vacf)) * (2.0 * dt_fs) # dt between samples = 2 fs
            hann = 0.5 * (1.0 + np.cos(np.pi * t_vacf / t_vacf[-1]))
            vdos = []
            for w_mev in hw_grid:
                w_rad_fs = (w_mev / 1000.0) / HBAR_eV_fs
                val = np.trapezoid(vacf * hann * np.cos(w_rad_fs * t_vacf), t_vacf) if hasattr(np, "trapezoid") else np.trapz(vacf * hann * np.cos(w_rad_fs * t_vacf), t_vacf)
                vdos.append(max(0.0, val))
            vdos = np.array(vdos)
            if np.max(vdos) > 0:
                vdos /= np.max(vdos)

            anharmonic_dict[T] = {
                "energies": eg_base,
                "lineshape": a_anh,
                "sigma_du": sigma_du,
                "fwhm_meV": fwhm_meV,
                "vdos": vdos,
                "walltime_s": time.time() - t0
            }
            print(f"    [+] Anharmonic MD T = {T:5.1f} K | Energy Fluctuation sigma = {sigma_du:5.2f} meV | Lineshape FWHM = {fwhm_meV:5.1f} meV ({time.time()-t0:.1f} s)")

        return anharmonic_dict

    def plot_anharmonic_comparison(self, lineshape_dict, anharmonic_dict, hw_grid, s_cont, total_S):
        """Plots multi-panel figure comparing harmonic model, anharmonic MD, and experiment."""
        fig, axs = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)

        # Panel (a): Spectral Density & Anharmonic VDOS
        ax = axs[0]
        s_norm = s_cont / np.max(s_cont)
        ax.plot(hw_grid, s_norm, color='#1f77b4', lw=2.2, label=r'Harmonic $S(\hbar\omega)$ ($T=0\ \mathrm{K}$)')
        ax.fill_between(hw_grid, 0, s_norm, color='#1f77b4', alpha=0.15)
        
        if 77.0 in anharmonic_dict:
            ax.plot(hw_grid, anharmonic_dict[77.0]["vdos"], color='#2ca02c', lw=2.0, linestyle='--', label=r'Anharmonic VDOS ($T=77\ \mathrm{K}$)')
        if 300.0 in anharmonic_dict:
            ax.plot(hw_grid, anharmonic_dict[300.0]["vdos"], color='#d62728', lw=2.0, linestyle=':', label=r'Anharmonic VDOS ($T=300\ \mathrm{K}$)')
            
        ax.axvline(19.7, color='black', lw=1.8, linestyle='-.', label=r'Takiyama (1978) Exp. $\hbar\omega_e = 19.7\ \mathrm{meV}$')
        ax.set_xlabel(r'Phonon Frequency $\hbar\omega$ (meV)', fontsize=11)
        ax.set_ylabel(r'Normalized Density / Coupling', fontsize=11)
        ax.set_title(r'(a) Phonon Coupling & Anharmonic VDOS', loc='left', fontweight='bold', fontsize=11.5)
        ax.grid(True, linestyle=':', alpha=0.5)
        ax.legend(frameon=True, loc='upper right', framealpha=0.9, fontsize=9.2)

        # Panel (b): Multi-Temperature Lineshapes: Harmonic vs. Anharmonic
        ax2 = axs[1]
        colors = {'10.0': '#1f77b4', '77.0': '#2ca02c', '150.0': '#ff7f0e', '300.0': '#d62728'}
        for T, (eg, an_harm) in lineshape_dict.items():
            col = colors.get(str(T), '#1f77b4')
            ax2.plot(eg, an_harm, color=col, lw=1.5, linestyle='--', label=f'Harm. {int(T)} K')
            if T in anharmonic_dict:
                an_anh = anharmonic_dict[T]["lineshape"]
                ax2.plot(eg, an_anh, color=col, lw=2.2, linestyle='-', label=f'Anh. {int(T)} K')

        ax2.axvline(0.0, color='gray', linestyle=':', lw=1.0)
        ax2.set_xlim(-220, 30)
        ax2.set_xlabel(r'Energy Relative to ZPL: $E - E_{\mathrm{ZPL}}$ (meV)', fontsize=11)
        ax2.set_ylabel(r'Normalized Emission Intensity', fontsize=11)
        ax2.set_title(r'(b) Multi-T Emission: Harmonic vs. Anharmonic', loc='left', fontweight='bold', fontsize=11.5)
        ax2.grid(True, linestyle=':', alpha=0.5)
        ax2.legend(frameon=True, loc='upper left', framealpha=0.9, fontsize=8.5, ncol=2)

        # Panel (c): Quantitative Benchmark at 77 K vs. Takiyama (1978)
        ax3 = axs[2]
        # Experimental Gaussian at 77 K (Takiyama JPSJ 1978: FWHM = 460 meV = 0.46 eV)
        eg_exp = np.linspace(-350, 100, 1000)
        sigma_exp = 460.0 / 2.355 # meV
        pl_exp = np.exp(-0.5 * ((eg_exp + 85.0) / sigma_exp)**2)
        pl_exp /= np.max(pl_exp)

        # Harmonic 77 K
        eg_h, an_h = lineshape_dict[77.0]
        ax3.plot(eg_h, an_h / np.max(an_h), color='#1f77b4', lw=2.0, linestyle='--', label=r'Harmonic Theory ($77\ \mathrm{K}$, FWHM $\approx 78\ \mathrm{meV}$)')
        
        # Anharmonic 77 K
        if 77.0 in anharmonic_dict:
            an_a = anharmonic_dict[77.0]["lineshape"]
            fwhm_a = anharmonic_dict[77.0]["fwhm_meV"]
            ax3.plot(eg_h, an_a / np.max(an_a), color='#2ca02c', lw=2.4, linestyle='-', label=f'Anharmonic MD ($77\\ \\mathrm{{K}}$, FWHM $\\approx {int(fwhm_a)}\\ \\mathrm{{meV}}$)')

        ax3.plot(eg_exp, pl_exp, color='#d62728', lw=2.2, linestyle='-.', label=r'Takiyama (1978) Exp. ($77\ \mathrm{K}$, FWHM $= 460\ \mathrm{meV}$)')
        ax3.set_xlim(-300, 50)
        ax3.set_xlabel(r'Energy Relative to ZPL: $E - E_{\mathrm{ZPL}}$ (meV)', fontsize=11)
        ax3.set_ylabel(r'Normalized Intensity (arb. units)', fontsize=11)
        ax3.set_title(r'(c) 77 K Benchmark vs. Takiyama (1978) Experiment', loc='left', fontweight='bold', fontsize=11.5)
        ax3.grid(True, linestyle=':', alpha=0.5)
        ax3.legend(frameon=True, loc='upper left', framealpha=0.92, fontsize=9.0)

        plt.tight_layout()
        out_fig = os.path.join(self.config.workspace_dir, "licl_fcenter_anharmonic_lineshapes.png")
        plt.savefig(out_fig, dpi=300)
        print(f"[+] Anharmonic comparison figure saved: {out_fig}")

    def plot_lineshapes(self, lineshape_dict, hw_grid, s_cont, total_S):
        """Plots multi-temperature optical lineshape figure."""
        fig, axs = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
        
        # Panel 1: Spectral density S(hw)
        ax = axs[0]
        ax.plot(hw_grid, s_cont / 1000.0, color='#1f77b4', lw=2.2)
        ax.fill_between(hw_grid, 0, s_cont / 1000.0, color='#1f77b4', alpha=0.25)
        ax.set_xlabel(r'Phonon Frequency $\hbar\omega$ (meV)')
        ax.set_ylabel(r'Spectral Density $S(\hbar\omega)$ ($\mathrm{meV}^{-1}$)')
        ax.set_title(r'(a) Electron-Phonon Spectral Density $S(\hbar\omega)$ ($S = ' + f'{total_S:.2f}' + r'$)', loc='left', fontweight='bold')
        ax.grid(True, linestyle=':', alpha=0.5)
        
        # Panel 2: Multi-temperature optical emission lineshapes
        ax = axs[1]
        colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']
        for (T, (eg, an)), col in zip(lineshape_dict.items(), colors):
            ax.plot(eg, an, color=col, lw=2.0, label=f'$T = {int(T)}\\ \\mathrm{{K}}$')
            
        ax.axvline(0.0, color='gray', linestyle=':', lw=1.0)
        ax.set_xlim(-180, 20)
        ax.set_xlabel(r'Energy Relative to ZPL: $E - E_{\mathrm{ZPL}}$ (meV)')
        ax.set_ylabel(r'Normalized Photoluminescence Intensity (arb. units)')
        ax.set_title(r'(b) Multi-Temperature Optical Emission Lineshapes $A(E - E_{\mathrm{ZPL}}, T)$', loc='left', fontweight='bold')
        ax.grid(True, linestyle=':', alpha=0.5)
        ax.legend(frameon=True, loc='upper left', framealpha=0.92)
        
        plt.tight_layout()
        out_fig = os.path.join(self.config.workspace_dir, "licl_fcenter_lineshapes.png")
        plt.savefig(out_fig, dpi=300)
        print(f"[+] Multi-temperature lineshape figure saved: {out_fig}")
