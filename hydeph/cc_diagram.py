"""
hydeph.cc_diagram
=================
Computes potential energy curves along the one-dimensional Franck-Condon configuration
coordinate from ground state to excited state coordinates, deriving relaxation energies,
Franck-Condon parameters, and the total Stokes shift.
Generates publication-quality 2-panel configuration coordinate diagrams and optical spectra.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any
from ase import Atoms


# Unit conversion constant for effective phonon frequency:
# hbar * Omega (meV) = 91.43476 * sqrt(Delta E_rel (eV) / (Delta Q (amu^(1/2) A))^2)
# Derived from hbar = 6.58211957e-16 eV*s and 1 eV / (amu * A^2) = 9.648533e27 s^-2.
# Matches exactly the lineshape coupling constant 0.1196127 meV^-1 amu^-1 A^-2.
CC_HBAR_OMEGA_PREFACTOR = 91.43476


class ConfigurationCoordinateEngine:
    """
    Evaluates 1D potential energy surfaces along the generalized optical coordinate
    R(lambda) = R_GS + lambda * (R_ES - R_GS) using the trained hybrid defect potential,
    and extracts all spectroscopic relaxation and Stokes shift parameters.
    """

    def __init__(self, config, calculator):
        """
        Parameters
        ----------
        config : ToolkitConfig
            Toolkit configuration object.
        calculator : Calculator
            Trained HybridDefectCalculator (or ASE calculator) evaluated on defect frames.
        """
        self.config = config
        self.calc = calculator

    def calculate_cc_curve(
        self,
        at_gs: Atoms,
        at_es: Atoms,
        lambda_min: Optional[float] = None,
        lambda_max: Optional[float] = None,
        n_points: Optional[int] = None,
        zpl_energy_eV: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Calculates ground- and excited-state potential energy curves along the 1D CC path.

        Parameters
        ----------
        at_gs : ase.Atoms
            Relaxed ground-state defect supercell.
        at_es : ase.Atoms
            Excited-state defect supercell geometry.
        lambda_min : float, optional
            Minimum coordinate lambda (default from config or -0.30).
        lambda_max : float, optional
            Maximum coordinate lambda (default from config or 1.30).
        n_points : int, optional
            Number of grid points along the linear CC path (default from config or 81).
        zpl_energy_eV : float, optional
            Zero-Phonon Line energy in eV. If None, uses config.zpl_energy_eV or defaults to 1.945 eV.

        Returns
        -------
        dict
            Dictionary of coordinates, energies, relaxation parameters, and Stokes shift metrics.
        """
        l_min = lambda_min if lambda_min is not None else getattr(self.config, "cc_lambda_min", -0.30)
        l_max = lambda_max if lambda_max is not None else getattr(self.config, "cc_lambda_max", 1.30)
        n_pts = n_points if n_points is not None else getattr(self.config, "cc_n_points", 81)

        # Mass-weighted displacement vector Delta Q
        m = at_gs.get_masses()
        dr = at_es.positions - at_gs.positions
        dq_total = float(np.sqrt(np.sum(np.repeat(m, 3) * dr.flatten()**2)))

        lambdas = np.linspace(l_min, l_max, n_pts)
        q_grid = lambdas * dq_total

        # Evaluate Ground State baseline reference energy at lambda = 0
        at_ref = at_gs.copy()
        at_ref.calc = self.calc
        v_ref = float(at_ref.get_potential_energy())

        # Sweep along linear path R(lambda)
        e_gs_list = []
        for lam in lambdas:
            at_lam = at_gs.copy()
            at_lam.positions = at_gs.positions + lam * dr
            at_lam.calc = self.calc
            e_val = float(at_lam.get_potential_energy()) - v_ref
            e_gs_list.append(e_val)

        e_gs = np.array(e_gs_list)

        # Ground-state Franck-Condon relaxation energy (at lambda = 1.0, R = R_ES)
        at_1 = at_gs.copy()
        at_1.positions = at_es.positions
        at_1.calc = self.calc
        e_rel_gs_eV = float(at_1.get_potential_energy()) - v_ref
        e_rel_gs_meV = e_rel_gs_eV * 1000.0

        # Zero-phonon line energy
        user_zpl = zpl_energy_eV if zpl_energy_eV is not None else getattr(self.config, "zpl_energy_eV", None)
        if user_zpl is not None:
            e_zpl = float(user_zpl)
            zpl_source = "user_configured"
        else:
            e_zpl = 1.945  # Representative canonical default (e.g. Diamond NV-)
            zpl_source = "default_canonical_1.945eV"

        # Excited-state potential energy surface in standard equal-curvature Franck-Condon model:
        # E_ES(lambda) = E_ZPL + Delta E_rel,ES * (1 - lambda)^2
        # where Delta E_rel,ES = Delta E_rel,GS (equal curvature approximation)
        e_rel_es_eV = e_rel_gs_eV
        e_rel_es_meV = e_rel_gs_meV
        e_es = e_zpl + e_rel_es_eV * (1.0 - lambdas)**2

        # Spectroscopic transition energies:
        # Vertical absorption from GS equilibrium (lambda = 0) to ES:
        e_abs_eV = e_zpl + e_rel_es_eV
        # Vertical emission (luminescence) from ES equilibrium (lambda = 1) to GS:
        e_em_eV = e_zpl - e_rel_gs_eV
        # Stokes shift Delta E_Stokes = E_abs - E_em = Delta E_rel,ES + Delta E_rel,GS
        stokes_shift_eV = e_abs_eV - e_em_eV
        stokes_shift_meV = stokes_shift_eV * 1000.0

        # Effective effective phonon frequency and 1D Huang-Rhys factor:
        if dq_total > 1e-6 and e_rel_gs_eV > 0:
            hbar_omega_meV = float(CC_HBAR_OMEGA_PREFACTOR * np.sqrt(e_rel_gs_eV) / dq_total)
            s_factor_1d = float(e_rel_gs_meV / hbar_omega_meV)
        else:
            hbar_omega_meV = 50.0
            s_factor_1d = 0.0

        # Fit quadratic curvature around GS minimum (lambda in [-0.2, 0.2])
        fit_mask = np.abs(lambdas) <= 0.25
        if np.sum(fit_mask) >= 3:
            p_fit = np.polyfit(q_grid[fit_mask], e_gs[fit_mask], 2)
            k_fit_eV_amu_A2 = float(2.0 * p_fit[0])
        else:
            k_fit_eV_amu_A2 = float(2.0 * e_rel_gs_eV / (dq_total**2)) if dq_total > 0 else 0.0

        results = {
            "lambda_grid": lambdas,
            "q_grid": q_grid,
            "e_gs": e_gs,
            "e_es": e_es,
            "dq_total": dq_total,
            "delta_e_rel_gs_eV": e_rel_gs_eV,
            "delta_e_rel_gs_meV": e_rel_gs_meV,
            "delta_e_rel_es_eV": e_rel_es_eV,
            "delta_e_rel_es_meV": e_rel_es_meV,
            "e_zpl_eV": e_zpl,
            "zpl_source": zpl_source,
            "e_abs_eV": e_abs_eV,
            "e_em_eV": e_em_eV,
            "stokes_shift_eV": stokes_shift_eV,
            "stokes_shift_meV": stokes_shift_meV,
            "hbar_omega_meV": hbar_omega_meV,
            "s_factor_1d": s_factor_1d,
            "k_curvature_gs": k_fit_eV_amu_A2,
        }

        print(f"[+] 1D Configuration Coordinate analysis complete:")
        print(f"    • Mass-weighted displacement Delta Q: {dq_total:.4f} amu^(1/2) A")
        print(f"    • Zero-Phonon Line E_ZPL: {e_zpl:.4f} eV ({zpl_source})")
        print(f"    • GS Relaxation Energy Delta E_rel,GS: {e_rel_gs_meV:.1f} meV ({e_rel_gs_eV:.4f} eV)")
        print(f"    • ES Relaxation Energy Delta E_rel,ES: {e_rel_es_meV:.1f} meV ({e_rel_es_eV:.4f} eV)")
        print(f"    • Vertical Absorption E_abs: {e_abs_eV:.4f} eV")
        print(f"    • Vertical Emission E_em: {e_em_eV:.4f} eV")
        print(f"    • Total Stokes Shift Delta E_Stokes: {stokes_shift_meV:.1f} meV ({stokes_shift_eV:.4f} eV)")
        print(f"    • Effective Phonon Frequency hbar*Omega: {hbar_omega_meV:.1f} meV (S_1D = {s_factor_1d:.2f})")

        return results

    def plot_configuration_coordinate(
        self,
        cc_results: Dict[str, Any],
        output_path: Optional[str] = None,
        title: Optional[str] = None,
        system_name: Optional[str] = None,
    ) -> str:
        """
        Renders a publication-grade 2-panel diagram showing:
          - Panel (a): 1D Configuration Coordinate potential energy curves, optical vertical
                       transitions (absorption and luminescence emission), non-radiative
                       relaxations, and ZPL origin.
          - Panel (b): Normalized Optical Absorption and Luminescence Emission spectra
                       separated by the derived Stokes shift.

        Parameters
        ----------
        cc_results : dict
            Result dictionary returned by calculate_cc_curve().
        output_path : str, optional
            Path to save figure. Defaults to config.workspace_dir / config.cc_figure_name.
        title : str, optional
            Optional overall figure suptitle.
        system_name : str, optional
            Optional system descriptor for plot headers (e.g. 'Diamond NV^-', '4H-SiC V_C-V_Si').

        Returns
        -------
        str
            Absolute path to saved figure.
        """
        if output_path is None:
            fig_name = getattr(self.config, "cc_figure_name", "configuration_coordinate.png")
            ws_dir = getattr(self.config, "workspace_dir", "./hydeph_run")
            output_path = os.path.join(ws_dir, fig_name)

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        q_grid = cc_results["q_grid"]
        lambdas = cc_results["lambda_grid"]
        e_gs = cc_results["e_gs"]
        e_es = cc_results["e_es"]
        dq_total = cc_results["dq_total"]
        e_rel_gs_eV = cc_results["delta_e_rel_gs_eV"]
        e_rel_gs_meV = cc_results["delta_e_rel_gs_meV"]
        e_rel_es_eV = cc_results["delta_e_rel_es_eV"]
        e_rel_es_meV = cc_results["delta_e_rel_es_meV"]
        e_zpl = cc_results["e_zpl_eV"]
        e_abs = cc_results["e_abs_eV"]
        e_em = cc_results["e_em_eV"]
        stokes_shift_eV = cc_results["stokes_shift_eV"]
        stokes_shift_meV = cc_results["stokes_shift_meV"]
        hbar_omega_meV = cc_results["hbar_omega_meV"]
        s_factor = cc_results["s_factor_1d"]

        fig, axs = plt.subplots(1, 2, figsize=(13.6, 5.8), dpi=300)

        # =========================================================================
        # PANEL (a): 1D Configuration Coordinate Diagram
        # =========================================================================
        ax = axs[0]
        ax.plot(q_grid, e_gs, color='#0b559f', lw=2.6, label=r'Ground State $E_{\mathrm{GS}}(Q)$')
        ax.plot(q_grid, e_es, color='#b2182b', lw=2.6, label=r'Excited State $E_{\mathrm{ES}}(Q)$')

        # Subtle fill under curves
        ax.fill_between(q_grid, e_gs, -0.5, color='#0b559f', alpha=0.04)

        # Minima markers
        ax.plot(0.0, 0.0, marker='o', markersize=7.5, color='#0b559f', zorder=5)
        ax.plot(dq_total, e_zpl, marker='o', markersize=7.5, color='#b2182b', zorder=5)

        # Vertical guide lines at Q = 0 and Q = Delta Q
        ax.axvline(0.0, color='gray', linestyle=':', lw=1.2, alpha=0.7)
        ax.axvline(dq_total, color='gray', linestyle=':', lw=1.2, alpha=0.7)

        # Zero-Phonon Line guide
        ax.plot([0.0, dq_total], [e_zpl, e_zpl], color='#2ca02c', linestyle='--', lw=1.6,
                label=r'Zero-Phonon Line ($E_{\mathrm{ZPL}} = ' + f'{e_zpl:.3f}' + r'\ \mathrm{eV}$)')

        # Vertical absorption arrow at Q = 0
        e_abs_val = e_abs
        ax.annotate('', xy=(0.0, e_abs_val), xytext=(0.0, 0.0),
                    arrowprops=dict(arrowstyle="->", color='#2b5c8f', lw=2.4, mutation_scale=15))
        ax.text(-0.03 * max(0.5, dq_total), e_abs_val * 0.48,
                r'$E_{\mathrm{abs}} = ' + f'{e_abs:.3f}' + r'\ \mathrm{eV}$' + '\n(Absorption)',
                color='#2b5c8f', fontsize=9.5, fontweight='bold', ha='right', va='center')

        # Vertical luminescence emission arrow at Q = Delta Q
        ax.annotate('', xy=(dq_total, e_rel_gs_eV), xytext=(dq_total, e_zpl),
                    arrowprops=dict(arrowstyle="->", color='#990000', lw=2.4, mutation_scale=15))
        ax.text(dq_total + 0.03 * max(0.5, dq_total), (e_zpl + e_rel_gs_eV) * 0.5,
                r'$E_{\mathrm{em}} = ' + f'{e_em:.3f}' + r'\ \mathrm{eV}$' + '\n(Emission)',
                color='#990000', fontsize=9.5, fontweight='bold', ha='left', va='center')

        # Relaxation arrows
        # Non-radiative relaxation in Excited State:
        ax.annotate('', xy=(dq_total * 0.82, e_zpl + e_rel_es_eV * 0.16), xytext=(0.06 * dq_total, e_abs_val - 0.03),
                    arrowprops=dict(arrowstyle="->", color='#b2182b', lw=1.8, linestyle='--'))
        ax.text(dq_total * 0.42, e_zpl + e_rel_es_eV * 0.80,
                r'$\Delta E_{\mathrm{rel,ES}} = ' + f'{e_rel_es_meV:.0f}' + r'\ \mathrm{meV}$',
                color='#b2182b', fontsize=9.5, ha='center', va='bottom')

        # Non-radiative relaxation in Ground State:
        ax.annotate('', xy=(dq_total * 0.18, e_rel_gs_eV * 0.16), xytext=(dq_total * 0.94, e_rel_gs_eV - 0.03),
                    arrowprops=dict(arrowstyle="->", color='#0b559f', lw=1.8, linestyle='--'))
        ax.text(dq_total * 0.58, e_rel_gs_eV * 0.70,
                r'$\Delta E_{\mathrm{rel,GS}} = ' + f'{e_rel_gs_meV:.0f}' + r'\ \mathrm{meV}$',
                color='#0b559f', fontsize=9.5, ha='center', va='bottom')

        # X-limits with padding
        q_pad = 0.08 * (q_grid[-1] - q_grid[0])
        ax.set_xlim(q_grid[0] - q_pad, q_grid[-1] + q_pad)

        # Dual top axis for dimensionless coordinate lambda
        ax_top = ax.twiny()
        ax_top.set_xlim(ax.get_xlim()[0] / dq_total, ax.get_xlim()[1] / dq_total)
        ax_top.set_xlabel(r'Dimensionless Coordinate $\lambda$ ($0 = \mathbf{R}_{\mathrm{GS}},\ 1 = \mathbf{R}_{\mathrm{ES}}$)', fontsize=10.5)

        # Dynamic Y-limits with ample headroom for summary box
        max_e_val = max(float(np.max(e_es)), float(np.max(e_gs)), e_abs)
        min_e_val = min(float(np.min(e_gs)), 0.0)
        e_span = max_e_val - min_e_val
        ax.set_ylim(min_e_val - 0.10 * e_span, max_e_val + 0.28 * e_span)

        ax.set_xlabel(r'Mass-Weighted Configuration Coordinate $Q$ ($\mathrm{amu}^{1/2}\mathrm{\AA}$)', fontsize=11)
        ax.set_ylabel(r'Potential Energy $E$ (eV)', fontsize=11)
        sys_str = f" - {system_name}" if system_name else ""
        ax.set_title(f'(a) 1D Configuration Coordinate Diagram{sys_str}', loc='left', fontweight='bold', fontsize=12)
        ax.grid(True, linestyle=':', alpha=0.5)
        ax.legend(frameon=True, loc='upper left', framealpha=0.92, fontsize=8.8)

        # Parameter summary text card in upper right of Panel (a)
        summary_text = (
            "Stokes & Franck-Condon Parameters:\n"
            f"  • $\\Delta Q = {dq_total:.4f}\\ \\mathrm{{amu}}^{{1/2}}\\mathrm{{\\AA}}$\n"
            f"  • $E_{{\\mathrm{{ZPL}}}} = {e_zpl:.3f}\\ \\mathrm{{eV}}$\n"
            f"  • $\\Delta E_{{\\mathrm{{rel,GS}}}} = {e_rel_gs_meV:.1f}\\ \\mathrm{{meV}}$\n"
            f"  • $\\Delta E_{{\\mathrm{{rel,ES}}}} = {e_rel_es_meV:.1f}\\ \\mathrm{{meV}}$\n"
            f"  • $\\Delta E_{{\\mathrm{{Stokes}}}} = {stokes_shift_meV:.1f}\\ \\mathrm{{meV}}$\n"
            f"  • $\\hbar\\Omega = {hbar_omega_meV:.1f}\\ \\mathrm{{meV}}$, $S = {s_factor:.2f}$"
        )
        ax.text(0.97, 0.97, summary_text, transform=ax.transAxes,
                fontsize=8.6, va='top', ha='right',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa', edgecolor='#cccccc', alpha=0.95))

        # =========================================================================
        # PANEL (b): Optical Absorption & Luminescence Emission Spectra
        # =========================================================================
        ax2 = axs[1]

        # Spectral lineshape envelope from 1D Franck-Condon model:
        # Gaussian width sigma = hbar * Omega * sqrt(S * (2*n_bose + 1))
        sigma_eV = max(0.04, (hbar_omega_meV * np.sqrt(max(1.0, s_factor))) / 1000.0)
        e_margin = max(0.5, 3.5 * sigma_eV)
        e_spec = np.linspace(e_em - e_margin, e_abs + e_margin, 1000)

        spec_em = np.exp(-0.5 * ((e_spec - e_em) / sigma_eV)**2)
        spec_abs = np.exp(-0.5 * ((e_spec - e_abs) / sigma_eV)**2)

        spec_em /= np.max(spec_em)
        spec_abs /= np.max(spec_abs)

        ax2.plot(e_spec, spec_em, color='#b2182b', lw=2.4,
                 label=r'Luminescence Emission ($E_{\mathrm{em}} = ' + f'{e_em:.3f}' + r'\ \mathrm{eV}$)')
        ax2.fill_between(e_spec, 0, spec_em, color='#b2182b', alpha=0.18)

        ax2.plot(e_spec, spec_abs, color='#0b559f', lw=2.4,
                 label=r'Optical Absorption ($E_{\mathrm{abs}} = ' + f'{e_abs:.3f}' + r'\ \mathrm{eV}$)')
        ax2.fill_between(e_spec, 0, spec_abs, color='#0b559f', alpha=0.18)

        # Zero-Phonon Line marker
        ax2.axvline(e_zpl, color='#2ca02c', linestyle='--', lw=1.8,
                    label=r'ZPL ($E_{\mathrm{ZPL}} = ' + f'{e_zpl:.3f}' + r'\ \mathrm{eV}$)')

        # Stokes shift dimension bracket between absorption and emission peaks
        peak_y = 1.05
        ax2.annotate('', xy=(e_abs, peak_y), xytext=(e_em, peak_y),
                     arrowprops=dict(arrowstyle="<->", color='black', lw=1.8, mutation_scale=12))
        stokes_label = f"Stokes Shift: {stokes_shift_meV:.0f} meV"
        ax2.text((e_em + e_abs) * 0.5, peak_y + 0.035, stokes_label,
                 ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='black',
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none', alpha=0.9))

        ax2.set_ylim(-0.02, 1.30)
        ax2.set_xlim(e_spec[0], e_spec[-1])
        ax2.set_xlabel(r'Photon Energy (eV)', fontsize=11)
        ax2.set_ylabel(r'Normalized Optical Intensity', fontsize=11)
        ax2.set_title(r'(b) Optical Spectra & Derived Stokes Shift', loc='left', fontweight='bold', fontsize=12)
        ax2.grid(True, linestyle=':', alpha=0.5)
        ax2.legend(frameon=True, loc='upper right', framealpha=0.92, fontsize=8.5)

        if title:
            fig.suptitle(title, fontsize=13, fontweight='bold', y=0.99)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close(fig)

        print(f"[+] Publication-grade Configuration Coordinate diagram saved: {output_path}")
        return os.path.abspath(output_path)


def plot_configuration_coordinate(
    config,
    calculator,
    at_gs: Atoms,
    at_es: Atoms,
    output_path: Optional[str] = None,
    zpl_energy_eV: Optional[float] = None,
    system_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Convenience functional interface to compute 1D configuration coordinate curves
    and generate the publication figure in one step.
    """
    engine = ConfigurationCoordinateEngine(config, calculator)
    cc_data = engine.calculate_cc_curve(at_gs, at_es, zpl_energy_eV=zpl_energy_eV)
    fig_path = engine.plot_configuration_coordinate(cc_data, output_path=output_path, system_name=system_name)
    cc_data["figure_path"] = fig_path
    return cc_data
