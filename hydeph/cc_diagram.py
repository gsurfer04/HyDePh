"""1D Franck-Condon configuration coordinate potential energy curves and Stokes shift analysis."""

from __future__ import annotations

import os
import json
import csv
import warnings
import logging
from typing import Optional, Dict, Any, Tuple
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ase import Atoms

from .config import ToolkitConfig
from .defect_builder import mic_displacement

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable

logger = logging.getLogger(__name__)

# Module-level physical constants and prefactors
NV_ZPL_EV = 1.945  # Reference canonical NV- defect ZPL energy in eV
CC_HBAR_OMEGA_PREFACTOR = 91.43476  # Prefactor: 91.43476 * sqrt(Delta E_rel (eV)) / Delta Q (amu^(1/2) A)


class ConfigurationCoordinateEngine:
    """Evaluates 1D potential surfaces along R(lambda) = R_GS + lambda * (R_ES - R_GS) and extracts Stokes shift parameters."""

    def __init__(self, config: ToolkitConfig, calculator):
        self.config = config
        self.calc = calculator
        self.last_results: Optional[Dict[str, Any]] = None

    def _derive_zpl(
        self, at_gs: Atoms, at_es: Atoms, e_rel_es_eV: float = 0.0
    ) -> Tuple[Optional[float], Optional[str]]:
        """Derives ZPL energy from Kohn-Sham defect band gap, Delta-SCF energies, or stored CC results."""
        # 1. Kohn-Sham defect eigenvalue transition gap from DFT
        if "dft_gap_eV" in at_gs.info and at_gs.info["dft_gap_eV"] is not None:
            gap_val = float(at_gs.info["dft_gap_eV"])
            # ZPL = vertical transition energy - excited state relaxation energy
            e_zpl = max(0.05, gap_val - float(e_rel_es_eV))
            return e_zpl, "kohn_sham_eigenvalues"

        # 2. Delta-SCF excited-state energy difference
        if "energy" in at_es.info and "energy" in at_gs.info:
            diff = float(at_es.info["energy"] - at_gs.info["energy"])
            # If difference significantly exceeds ground-state relaxation, treat as electronic Delta-SCF
            if diff > float(e_rel_es_eV) * 1.5 and diff > 0.1:
                return diff, "delta_scf"

        # 3. Stored CC results cache
        if self.last_results is not None and "e_zpl_eV" in self.last_results:
            return float(self.last_results["e_zpl_eV"]), "stored_cc_results"

        return None, None

    def calculate_cc_curve(
        self,
        at_gs: Atoms,
        at_es: Atoms,
        lambda_min: Optional[float] = None,
        lambda_max: Optional[float] = None,
        n_points: Optional[int] = None,
        zpl_energy_eV: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Calculates ground- and excited-state potential energy curves along the 1D configuration coordinate."""
        l_min = lambda_min if lambda_min is not None else self.config.cc_lambda_min
        l_max = lambda_max if lambda_max is not None else self.config.cc_lambda_max
        n_pts = n_points if n_points is not None else self.config.cc_n_points

        m = at_gs.get_masses()
        pos_gs = at_gs.positions.copy()
        dr = mic_displacement(at_es.positions, pos_gs, at_gs.cell)
        dq_total = float(np.sqrt(np.sum(m[:, None] * (dr ** 2))))

        lambdas = np.linspace(l_min, l_max, n_pts)
        q_grid = lambdas * dq_total

        at_eval = at_gs.copy()
        at_eval.calc = self.calc

        # Check for residual force along CC displacement vector dr
        try:
            f0 = at_eval.get_forces()
            proj_grad = -float(np.sum(f0 * dr))  # dE/dlambda at lambda=0
            if proj_grad < -0.01:
                from ase.optimize import BFGS
                opt = BFGS(at_eval, logfile=None)
                opt.run(fmax=0.01, steps=30)
                pos_gs = at_eval.positions.copy()
                dr = mic_displacement(at_es.positions, pos_gs, at_gs.cell)
                dq_total = float(np.sqrt(np.sum(m[:, None] * (dr ** 2))))
                q_grid = lambdas * dq_total
        except Exception:
            pass

        v_ref = float(at_eval.get_potential_energy())

        e_gs_list = []
        pbar = tqdm(lambdas, desc="CC curve evaluation", unit="pt", leave=False)
        for lam in pbar:
            at_eval.set_positions(pos_gs + lam * dr)
            e_val = float(at_eval.get_potential_energy()) - v_ref
            e_gs_list.append(e_val)

        e_gs = np.array(e_gs_list, dtype=float)

        at_eval.set_positions(pos_gs + dr)
        e_rel_gs_eV = float(at_eval.get_potential_energy()) - v_ref
        e_rel_gs_meV = e_rel_gs_eV * 1000.0

        user_zpl = zpl_energy_eV if zpl_energy_eV is not None else self.config.zpl_energy_eV
        if user_zpl is not None and user_zpl > 0:
            e_zpl = float(user_zpl)
            zpl_source = "user_configured"
        else:
            derived, src = self._derive_zpl(at_gs, at_es, e_rel_es_eV=e_rel_gs_eV)
            if derived is not None and derived > 0:
                e_zpl = derived
                zpl_source = src
                if src == "kohn_sham_eigenvalues":
                    logger.info(
                        f"      [+] Auto-derived Zero-Phonon Line E_ZPL: {e_zpl:.4f} eV from Kohn-Sham defect gap "
                        f"({at_gs.info['dft_gap_eV']:.4f} eV vertical minus {e_rel_gs_eV:.4f} eV relaxation)"
                    )
                else:
                    logger.info(f"      [+] Auto-derived Zero-Phonon Line E_ZPL: {e_zpl:.4f} eV ({src})")
                warnings.warn(
                    f"Auto-derived ZPL energy: {e_zpl:.4f} eV from {src}.",
                    UserWarning,
                    stacklevel=2,
                )
            else:
                e_zpl = NV_ZPL_EV
                zpl_source = f"default_reference_{NV_ZPL_EV}eV"
                warnings.warn(
                    f"No ZPL energy specified in ToolkitConfig and could not be derived from DFT energies. "
                    f"Falling back to reference canonical NV- ZPL ({NV_ZPL_EV} eV). "
                    "For accurate defect-specific diagrams, specify `ToolkitConfig.zpl_energy_eV`.",
                    UserWarning,
                    stacklevel=2,
                )

        # Equal-curvature Franck-Condon excited state: E_ES(lambda) = E_ZPL + Delta E_rel,ES * (1 - lambda)^2
        e_rel_es_eV = e_rel_gs_eV
        e_rel_es_meV = e_rel_gs_meV
        e_es = e_zpl + e_rel_es_eV * (1.0 - lambdas)**2

        e_abs_eV = e_zpl + e_rel_es_eV
        e_em_eV = e_zpl - e_rel_gs_eV
        stokes_shift_eV = e_abs_eV - e_em_eV
        stokes_shift_meV = stokes_shift_eV * 1000.0

        if dq_total > 1e-6 and e_rel_gs_eV > 0:
            hbar_omega_meV = float(CC_HBAR_OMEGA_PREFACTOR * np.sqrt(e_rel_gs_eV) / dq_total)
            s_factor_1d = float(e_rel_gs_meV / hbar_omega_meV)
        else:
            hbar_omega_meV = 50.0
            s_factor_1d = 0.0

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
            "dft_gap_eV": at_gs.info.get("dft_gap_eV"),
        }

        self.last_results = results

        logger.info("[+] 1D Configuration Coordinate analysis complete:")
        logger.info(f"    • Mass-weighted displacement Delta Q: {dq_total:.4f} amu^(1/2) A")
        logger.info(f"    • Zero-Phonon Line E_ZPL: {e_zpl:.4f} eV ({zpl_source})")
        logger.info(f"    • GS Relaxation Energy Delta E_rel,GS: {e_rel_gs_meV:.1f} meV ({e_rel_gs_eV:.4f} eV)")
        logger.info(f"    • ES Relaxation Energy Delta E_rel,ES: {e_rel_es_meV:.1f} meV ({e_rel_es_eV:.4f} eV)")
        logger.info(f"    • Vertical Absorption E_abs: {e_abs_eV:.4f} eV")
        logger.info(f"    • Vertical Emission E_em: {e_em_eV:.4f} eV")
        logger.info(f"    • Total Stokes Shift Delta E_Stokes: {stokes_shift_meV:.1f} meV ({stokes_shift_eV:.4f} eV)")
        logger.info(f"    • Effective Phonon Frequency hbar*Omega: {hbar_omega_meV:.1f} meV (S_1D = {s_factor_1d:.2f})")

        return results

    def export_results(self, cc_results: Dict[str, Any], output_dir: Optional[str] = None) -> Dict[str, str]:
        """Exports numerical results to JSON and CSV files."""
        out_dir = output_dir or self.config.workspace_dir
        os.makedirs(out_dir, exist_ok=True)

        json_path = os.path.join(out_dir, "configuration_coordinate_data.json")
        csv_path = os.path.join(out_dir, "configuration_coordinate_curve.csv")

        # Scalar JSON export
        scalars = {
            k: float(v) if isinstance(v, (np.floating, float, np.integer, int)) else v
            for k, v in cc_results.items()
            if not isinstance(v, np.ndarray) and k != "figure_path"
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(scalars, f, indent=2)

        # Tabular CSV export
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["lambda", "Q_amu_A", "E_GS_eV", "E_ES_eV"])
            for lam, q, e_g, e_e in zip(
                cc_results["lambda_grid"],
                cc_results["q_grid"],
                cc_results["e_gs"],
                cc_results["e_es"],
            ):
                writer.writerow([f"{lam:.6f}", f"{q:.6f}", f"{e_g:.6f}", f"{e_e:.6f}"])

        logger.info(f"[+] Numerical CC results exported: {json_path} and {csv_path}")
        return {"json": json_path, "csv": csv_path}

    def plot_configuration_coordinate(
        self,
        cc_results: Dict[str, Any],
        output_path: Optional[str] = None,
        title: Optional[str] = None,
        system_name: Optional[str] = None,
    ) -> str:
        """Renders a 2-panel configuration coordinate and optical spectra diagram, returning the saved file path."""
        if output_path is None:
            fig_name = self.config.cc_figure_name
            ws_dir = self.config.workspace_dir
            output_path = os.path.join(ws_dir, fig_name)

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        q_grid = cc_results["q_grid"]
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
        stokes_shift_meV = cc_results["stokes_shift_meV"]
        hbar_omega_meV = cc_results["hbar_omega_meV"]
        s_factor = cc_results["s_factor_1d"]

        fig, axs = plt.subplots(1, 2, figsize=(13.6, 5.8), dpi=300)

        ax = axs[0]
        ax.plot(q_grid, e_gs, color='#0b559f', lw=2.6, label=r'Ground State $E_{\mathrm{GS}}(Q)$')
        ax.plot(q_grid, e_es, color='#b2182b', lw=2.6, label=r'Excited State $E_{\mathrm{ES}}(Q)$')
        ax.fill_between(q_grid, e_gs, -0.5, color='#0b559f', alpha=0.04)

        ax.plot(0.0, 0.0, marker='o', markersize=7.5, color='#0b559f', zorder=5)
        ax.plot(dq_total, e_zpl, marker='o', markersize=7.5, color='#b2182b', zorder=5)

        ax.axvline(0.0, color='gray', linestyle=':', lw=1.2, alpha=0.7)
        ax.axvline(dq_total, color='gray', linestyle=':', lw=1.2, alpha=0.7)

        ax.plot([0.0, dq_total], [e_zpl, e_zpl], color='#2ca02c', linestyle='--', lw=1.6,
                label=r'Zero-Phonon Line ($E_{\mathrm{ZPL}} = ' + f'{e_zpl:.3f}' + r'\ \mathrm{eV}$)')

        e_abs_val = e_abs
        ax.annotate('', xy=(0.0, e_abs_val), xytext=(0.0, 0.0),
                    arrowprops=dict(arrowstyle="->", color='#2b5c8f', lw=2.4, mutation_scale=15))
        ax.text(-0.03 * max(0.5, dq_total), e_abs_val * 0.48,
                r'$E_{\mathrm{abs}} = ' + f'{e_abs:.3f}' + r'\ \mathrm{eV}$' + '\n(Absorption)',
                color='#2b5c8f', fontsize=9.5, fontweight='bold', ha='right', va='center')

        ax.annotate('', xy=(dq_total, e_rel_gs_eV), xytext=(dq_total, e_zpl),
                    arrowprops=dict(arrowstyle="->", color='#990000', lw=2.4, mutation_scale=15))
        ax.text(dq_total + 0.03 * max(0.5, dq_total), (e_zpl + e_rel_gs_eV) * 0.5,
                r'$E_{\mathrm{em}} = ' + f'{e_em:.3f}' + r'\ \mathrm{eV}$' + '\n(Emission)',
                color='#990000', fontsize=9.5, fontweight='bold', ha='left', va='center')

        dy_es = min(0.03, 0.15 * max(0.005, e_rel_es_eV))
        dy_gs = min(0.03, 0.15 * max(0.005, e_rel_gs_eV))
        ax.annotate('', xy=(dq_total * 0.82, e_zpl + e_rel_es_eV * 0.16), xytext=(0.06 * dq_total, e_abs_val - dy_es),
                    arrowprops=dict(arrowstyle="->", color='#b2182b', lw=1.8, linestyle='--'))
        ax.text(dq_total * 0.42, e_zpl + e_rel_es_eV * 0.80,
                r'$\Delta E_{\mathrm{rel,ES}} = ' + f'{e_rel_es_meV:.0f}' + r'\ \mathrm{meV}$',
                color='#b2182b', fontsize=9.5, ha='center', va='bottom')

        ax.annotate('', xy=(dq_total * 0.18, e_rel_gs_eV * 0.16), xytext=(dq_total * 0.94, e_rel_gs_eV - dy_gs),
                    arrowprops=dict(arrowstyle="->", color='#0b559f', lw=1.8, linestyle='--'))
        ax.text(dq_total * 0.58, e_rel_gs_eV * 0.70,
                r'$\Delta E_{\mathrm{rel,GS}} = ' + f'{e_rel_gs_meV:.0f}' + r'\ \mathrm{meV}$',
                color='#0b559f', fontsize=9.5, ha='center', va='bottom')

        q_pad = 0.08 * (q_grid[-1] - q_grid[0])
        ax.set_xlim(q_grid[0] - q_pad, q_grid[-1] + q_pad)

        ax_top = ax.twiny()
        ax_top.set_xlim(ax.get_xlim()[0] / dq_total, ax.get_xlim()[1] / dq_total)
        ax_top.set_xlabel(r'Dimensionless Coordinate $\lambda$ ($0 = \mathbf{R}_{\mathrm{GS}},\ 1 = \mathbf{R}_{\mathrm{ES}}$)', fontsize=10.5)

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

        ax2 = axs[1]

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

        ax2.axvline(e_zpl, color='#2ca02c', linestyle='--', lw=1.8,
                    label=r'ZPL ($E_{\mathrm{ZPL}} = ' + f'{e_zpl:.3f}' + r'\ \mathrm{eV}$)')

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

        logger.info(f"[+] Publication-grade Configuration Coordinate diagram saved: {output_path}")
        self.export_results(cc_results, os.path.dirname(os.path.abspath(output_path)))
        return os.path.abspath(output_path)


def plot_configuration_coordinate(
    config: ToolkitConfig,
    calculator,
    at_gs: Atoms,
    at_es: Atoms,
    output_path: Optional[str] = None,
    zpl_energy_eV: Optional[float] = None,
    system_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Computes 1D configuration coordinate curves and renders the 2-panel diagram."""
    engine = ConfigurationCoordinateEngine(config, calculator)
    cc_data = engine.calculate_cc_curve(at_gs, at_es, zpl_energy_eV=zpl_energy_eV)
    fig_path = engine.plot_configuration_coordinate(cc_data, output_path=output_path, system_name=system_name)
    cc_data["figure_path"] = fig_path
    return cc_data
