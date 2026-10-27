"""Comparison plot of HyDePh Delta-NEP lineshapes vs experimental benchmarks.

Reads strictly from committed production runs in benchmarks/NaCl/.
"""

from __future__ import annotations

import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sd_csv = os.path.join(repo_root, "benchmarks", "NaCl", "spectral_density.csv")
lines_csv = os.path.join(repo_root, "benchmarks", "NaCl", "optical_lineshapes.csv")
summary_json = os.path.join(repo_root, "benchmarks", "NaCl", "lineshape_summary.json")

for p in [sd_csv, lines_csv, summary_json]:
    if not os.path.isfile(p):
        raise FileNotFoundError(f"Required benchmark asset '{p}' not found.")

df_sd = pd.read_csv(sd_csv)
df_lines = pd.read_csv(lines_csv)
with open(summary_json, "r") as f:
    meta = json.load(f)

fig, axs = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

# Panel (a): Spectral density S(hw)
ax = axs[0]
hw = df_sd["hw_meV"].values
s_hw = df_sd["S_harmonic_per_eV"].values

ax.plot(hw, s_hw, color="#1f77b4", lw=2.4, label=r"HyDePh $S(\hbar\omega)$ ($S = " + f"{meta['total_S']:.2f}" + r"$)")
ax.fill_between(hw, 0, s_hw, color="#1f77b4", alpha=0.22)
ax.axvspan(15.0, 25.0, color="gold", alpha=0.18, label="Dominant Acoustic/TO Coupling Band")
ax.set_xlabel(r"Phonon Frequency $\hbar\omega$ (meV)", fontsize=12)
ax.set_ylabel(r"Spectral Density $S(\hbar\omega)$ ($\mathrm{eV}^{-1}$)", fontsize=12)
ax.set_title(r"(a) Electron-Phonon Coupling Spectral Density (NaCl F-Center)", loc="left", fontweight="bold", fontsize=12)
ax.grid(True, linestyle=":", alpha=0.55)
ax.legend(frameon=True, loc="upper right", framealpha=0.92, fontsize=10)

# Panel (b): Photoluminescence Lineshapes across Temperatures
ax2 = axs[1]
e_rel = df_lines["energy_rel_meV"].values
y_77 = df_lines["A_anharmonic_77.0K"].values / np.max(df_lines["A_anharmonic_77.0K"].values)
y_300 = df_lines["A_anharmonic_300.0K"].values / np.max(df_lines["A_anharmonic_300.0K"].values)
y_harm = df_lines["A_harmonic_77.0K"].values / np.max(df_lines["A_harmonic_77.0K"].values)

fwhm_77 = meta["anharmonic"]["77.0"]["fwhm_meV"]
fwhm_300 = meta["anharmonic"]["300.0"]["fwhm_meV"]

ax2.plot(e_rel, y_harm, color="#1f77b4", lw=2.0, linestyle="--", label=r"Harmonic ($77\ \mathrm{K}$, static)")
ax2.plot(e_rel, y_77, color="#2ca02c", lw=2.4, label=r"Anharmonic MD ($77\ \mathrm{K}$, $\mathrm{FWHM} = " + f"{fwhm_77:.1f}" + r"\ \mathrm{meV}$)")
ax2.plot(e_rel, y_300, color="#d62728", lw=2.2, label=r"Anharmonic MD ($300\ \mathrm{K}$, $\mathrm{FWHM} = " + f"{fwhm_300:.1f}" + r"\ \mathrm{meV}$)")

ax2.set_xlim(-360, 20)
ax2.set_xlabel(r"Energy Relative to ZPL: $E - E_{\mathrm{ZPL}}$ (meV)", fontsize=12)
ax2.set_ylabel(r"Normalized Photoluminescence (arb. units)", fontsize=12)
ax2.set_title(r"(b) F-Center Emission: Multi-Temperature Anharmonic Lineshapes", loc="left", fontweight="bold", fontsize=12)
ax2.grid(True, linestyle=":", alpha=0.55)
ax2.legend(frameon=True, loc="upper left", framealpha=0.92, fontsize=9.5)

plt.tight_layout()
out_dir = os.environ.get("HYDEPH_WORKSPACE", os.path.abspath("./hydeph_run"))
os.makedirs(out_dir, exist_ok=True)
out_fig = os.path.join(out_dir, "licl_fcenter_experimental_comparison.png")
plt.savefig(out_fig, dpi=300)
print(f"Experimental comparison figure generated successfully: {out_fig}")
