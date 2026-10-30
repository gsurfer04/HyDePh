import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# 1. Load HyDePh optical lineshapes
data_path = "/home/jamesc/HyDePhTest/SiC-4H-kk/hydeph_run_c1h_consistent/optical_lineshapes.csv"
if not os.path.exists(data_path):
    data_path = "hydeph_run_c1h_consistent/optical_lineshapes.csv"
df_hydeph = pd.read_csv(data_path)

# 2. Load Jin et al. digitized data
ref_path = "/mnt/h/Il mio Drive/HyDePh/jin_et_al_digitized_kk_10K.csv"
if not os.path.exists(ref_path):
    ref_path = "jin_et_al_digitized_kk_10K.csv"
df_ref = pd.read_csv(ref_path)

# Extract 10 K and 300 K data
e_rel = df_hydeph["energy_rel_meV"].values
i_10k = df_hydeph["I_PL_10.0K_norm"].values
i_300k = df_hydeph["I_PL_300.0K_norm"].values

# Filter window [-250, +30] meV
mask_10k = (e_rel >= -250) & (e_rel <= 30)
e_win = e_rel[mask_10k]
i_win_10k = i_10k[mask_10k]
i_win_300k = i_300k[mask_10k]

# Normalize by max in window
norm_10k = i_win_10k / np.max(i_win_10k)
norm_300k = i_win_300k / np.max(i_win_300k)

# Find peaks in 10 K lineshape (excluding ZPL at 0)
psb_mask = (e_win >= -200) & (e_win <= -10)
peaks, props = find_peaks(norm_10k[psb_mask], height=0.01, distance=15)
psb_e = e_win[psb_mask]
psb_i = norm_10k[psb_mask]

print("=== HyDePh 10 K Emission Spectrum Peaks (PSB) ===")
for p in peaks:
    print(f"Peak at {psb_e[p]:.2f} meV | Relative Intensity = {psb_i[p]:.4f}")

# Digitize normalization for reference data
ref_e = df_ref["energy_rel_meV"].values
ref_pbe = df_ref["intensity_pbe"].values
ref_expt = df_ref["intensity_expt"].values

ref_mask = (ref_e >= -250) & (ref_e <= 30)
ref_e_win = ref_e[ref_mask]
ref_pbe_win = ref_pbe[ref_mask] / np.max(ref_pbe[ref_mask])
ref_expt_win = ref_expt[ref_mask] / np.max(ref_expt[ref_mask])

# Create publication-quality figure
fig, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=True, gridspec_kw={'hspace': 0.15})

# Panel 1: 10 K Comparison
ax1 = axes[0]
ax1.plot(e_win, norm_10k, color="#1f77b4", lw=2.2, label="HyDePh (This Work, 10 K)")
ax1.plot(ref_e_win, ref_pbe_win, color="#d62728", lw=1.8, linestyle="--", label="Jin et al. (PBE, 10 K)")
ax1.plot(ref_e_win, ref_expt_win, color="#2ca02c", lw=1.5, linestyle=":", alpha=0.85, label="Jin et al. (Expt, 10 K)")

# Annotate the 3 attested peaks
# Peak 1: ~ -34 meV
ax1.annotate("Peak 1 (-34 meV)\nSi breathing", xy=(-34.0, norm_10k[np.argmin(np.abs(e_win - (-34.0)))]),
             xytext=(-45, 0.45), arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.5),
             fontsize=10, fontweight="bold", color="#1f77b4")

# Peak 2: ~ -72 meV
ax1.annotate("Peak 2 (-72 meV)\nC mode", xy=(-72.5, norm_10k[np.argmin(np.abs(e_win - (-72.5)))]),
             xytext=(-95, 0.35), arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.5),
             fontsize=10, fontweight="bold", color="#1f77b4")

# Peak 3: ~ -97 meV
ax1.annotate("Peak 3 (-97 meV)\nLocal mode", xy=(-97.0, norm_10k[np.argmin(np.abs(e_win - (-97.0)))]),
             xytext=(-140, 0.25), arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.5),
             fontsize=10, fontweight="bold", color="#1f77b4")

# Annotate ZPL
ax1.annotate("ZPL (1.096 eV)", xy=(0, 1.0), xytext=(5, 0.9),
             arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
             fontsize=10, fontweight="bold")

ax1.set_ylabel("Normalized PL Intensity (a.u.)", fontsize=12, fontweight="bold")
ax1.set_title("4H-SiC kk-Divacancy Emission Spectrum (10 K): HyDePh vs. Jin et al. (PRMaterials 2021)", fontsize=13, fontweight="bold")
ax1.legend(loc="upper left", frameon=True, fontsize=10)
ax1.grid(True, linestyle="--", alpha=0.5)
ax1.set_ylim(-0.02, 1.08)

# Panel 2: 300 K Lineshape demonstrating distinct persistent ZPL
ax2 = axes[1]
ax2.plot(e_win, norm_300k, color="#ff7f0e", lw=2.2, label="HyDePh (300 K)")
ax2.plot(e_win, norm_10k, color="#1f77b4", lw=1.2, linestyle="--", alpha=0.6, label="HyDePh (10 K reference)")

# Annotate 300 K ZPL
zpl_300_idx = np.argmin(np.abs(e_win - 0.0))
ax2.annotate("Distinct ZPL at 300 K", xy=(0, norm_300k[zpl_300_idx]), xytext=(10, 0.8),
             arrowprops=dict(arrowstyle="->", color="#ff7f0e", lw=1.5),
             fontsize=10, fontweight="bold", color="#ff7f0e")

ax2.set_xlabel("Energy relative to ZPL (meV)", fontsize=12, fontweight="bold")
ax2.set_ylabel("Normalized PL Intensity (a.u.)", fontsize=12, fontweight="bold")
ax2.set_title("Thermal Evolution & Persistent ZPL at 300 K", fontsize=13, fontweight="bold")
ax2.legend(loc="upper left", frameon=True, fontsize=10)
ax2.grid(True, linestyle="--", alpha=0.5)
ax2.set_xlim(-220, 25)
ax2.set_ylim(-0.02, 1.08)

out_fig = "/mnt/h/Il mio Drive/HyDePh/benchmarks/4H-SiC/hydeph_vs_jin_benchmark.png"
os.makedirs(os.path.dirname(out_fig), exist_ok=True)
plt.savefig(out_fig, dpi=300, bbox_inches="tight")
print(f"[+] Saved comparison figure to: {out_fig}")

# Also copy to artifacts dir
import shutil
artifact_dir = "C:/Users/gsurf/.gemini/antigravity-cli/brain/88b9fb1f-2111-4d53-bf65-47950df9c321"
if os.path.exists(artifact_dir):
    shutil.copy2(out_fig, os.path.join(artifact_dir, "hydeph_vs_jin_benchmark.png"))
