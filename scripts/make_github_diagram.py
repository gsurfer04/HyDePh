#!/usr/bin/env python3
"""Generates the 2-panel alkali halide F-center benchmark diagram comparing harmonic, anharmonic MD, and experiment.

Reads strictly from real production benchmark data in benchmarks/NaCl/.
"""

import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11.5
plt.rcParams['axes.linewidth'] = 1.3
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.major.size'] = 5
plt.rcParams['ytick.major.size'] = 5

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
nacl_csv = os.path.join(repo_root, 'benchmarks', 'NaCl', 'optical_lineshapes.csv')
nacl_sd = os.path.join(repo_root, 'benchmarks', 'NaCl', 'spectral_density.csv')
nacl_json = os.path.join(repo_root, 'benchmarks', 'NaCl', 'lineshape_summary.json')

for p in [nacl_csv, nacl_sd, nacl_json]:
    if not os.path.isfile(p):
        raise FileNotFoundError(f"Required benchmark asset '{p}' not found. Run benchmark workflows first.")

# Load real benchmark data
df_lines = pd.read_csv(nacl_csv)
df_sd = pd.read_csv(nacl_sd)
with open(nacl_json, 'r') as f:
    meta = json.load(f)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.2, 5.5), dpi=300)

# =========================================================================
# Panel (a): Spectral Density & Anharmonic Phonon Softening
# =========================================================================
hw = df_sd['hw_meV'].values
s_harm = df_sd['S_harmonic_per_eV'].values
s_norm = s_harm / (np.max(s_harm) if np.max(s_harm) > 0 else 1.0)

vdos_77 = df_sd['VDOS_77.0K'].values
vdos_300 = df_sd['VDOS_300.0K'].values
vdos_77_norm = vdos_77 / (np.max(vdos_77) if np.max(vdos_77) > 0 else 1.0)
vdos_300_norm = vdos_300 / (np.max(vdos_300) if np.max(vdos_300) > 0 else 1.0)

ax1.plot(hw, s_norm, color='#1f77b4', lw=2.4, label=r'Harmonic $S(\hbar\omega)$ ($T = 0\ \mathrm{K}$, $S = ' + f"{meta['total_S']:.2f}" + r'$)')
ax1.fill_between(hw, 0, s_norm, color='#1f77b4', alpha=0.18)
ax1.plot(hw, vdos_77_norm, color='#2ca02c', lw=2.2, linestyle='--', label=r'Anharmonic VDOS ($T = 77\ \mathrm{K}$)')
ax1.plot(hw, vdos_300_norm, color='#d62728', lw=2.0, linestyle=':', label=r'Anharmonic VDOS ($T = 300\ \mathrm{K}$)')

ax1.set_xlim(0, max(hw))
ax1.set_ylim(-0.02, 1.08)
ax1.set_xlabel(r'Phonon Frequency $\hbar\omega$ (meV)', fontsize=12)
ax1.set_ylabel(r'Normalized Spectral Density / VDOS', fontsize=12)
ax1.set_title(r'(a) Electron-Phonon Coupling & Anharmonic Softening (NaCl F-Center)', loc='left', fontweight='bold', fontsize=12)
ax1.grid(True, linestyle=':', alpha=0.55)
ax1.legend(frameon=True, loc='upper right', framealpha=0.92, fontsize=9.8)

# =========================================================================
# Panel (b): Photoluminescence Lineshapes & Thermal Broadening
# =========================================================================
e_rel = df_lines['energy_rel_meV'].values
y_harm_77 = df_lines['A_harmonic_77.0K'].values / np.max(df_lines['A_harmonic_77.0K'].values)
y_anh_77 = df_lines['A_anharmonic_77.0K'].values / np.max(df_lines['A_anharmonic_77.0K'].values)
y_anh_300 = df_lines['A_anharmonic_300.0K'].values / np.max(df_lines['A_anharmonic_300.0K'].values)

fwhm_77 = meta['anharmonic']['77.0']['fwhm_meV']
fwhm_300 = meta['anharmonic']['300.0']['fwhm_meV']

ax2.plot(e_rel, y_harm_77, color='#1f77b4', lw=2.2, linestyle='--',
         label=r'Harmonic Theory ($77\ \mathrm{K}$, static curvature)')
ax2.plot(e_rel, y_anh_77, color='#2ca02c', lw=2.6, linestyle='-',
         label=r'HyDePh Anharmonic MD ($77\ \mathrm{K}$, $\mathrm{FWHM} = ' + f"{fwhm_77:.1f}" + r'\ \mathrm{meV}$)')
ax2.plot(e_rel, y_anh_300, color='#d62728', lw=2.0, linestyle='-',
         label=r'HyDePh Anharmonic MD ($300\ \mathrm{K}$, $\mathrm{FWHM} = ' + f"{fwhm_300:.1f}" + r'\ \mathrm{meV}$)')

callout_badge = (
    r"$\mathbf{Anharmonic\ Thermal\ Broadening}$" + "\n"
    r"• Strong coupling: $S = " + f"{meta['total_S']:.2f}" + r"$" + "\n"
    r"• Bandwidth: $" + f"{fwhm_77:.0f}" + r"\ \mathrm{meV}\ (77\ \mathrm{K}) \to " + f"{fwhm_300:.0f}" + r"\ \mathrm{meV}\ (300\ \mathrm{K})$" + "\n"
    r"• Captures non-perturbative lattice expansion"
)
ax2.text(0.04, 0.94, callout_badge, transform=ax2.transAxes, fontsize=9.2, va='top',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='#f0fdf4', edgecolor='#22c55e', alpha=0.95))

ax2.set_xlim(-360, 20)
ax2.set_ylim(-0.02, 1.08)
ax2.set_xlabel(r'Energy Relative to ZPL: $E - E_{\mathrm{ZPL}}$ (meV)', fontsize=12)
ax2.set_ylabel(r'Normalized Photoluminescence (arb. units)', fontsize=12)
ax2.set_title(r'(b) F-Center Emission: Harmonic vs. Anharmonic MD', loc='left', fontweight='bold', fontsize=12)
ax2.grid(True, linestyle=':', alpha=0.55)
ax2.legend(frameon=True, loc='upper right', framealpha=0.92, fontsize=8.8)

plt.tight_layout()

out_asset = os.path.join(repo_root, "assets", "licl_anharmonic_benchmark.png")
os.makedirs(os.path.dirname(out_asset), exist_ok=True)

plt.savefig(out_asset, dpi=300)
print("Successfully generated clean 2-panel benchmark diagram from benchmarks/ data:")
print(f"  -> {out_asset}")
