#!/usr/bin/env python3
"""Generate 2-panel 4H-SiC kk-VV0 benchmark comparison against Jin et al. (PRM 5, 084603, 2021)."""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.size"] = 10.5
plt.rcParams["axes.linewidth"] = 1.2
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"

sic_dir = Path(__file__).resolve().parents[1]
repo_root = sic_dir.parents[1]

df_h = pd.read_csv(sic_dir / "optical_lineshapes.csv")
df_r = pd.read_csv(sic_dir / "jin_et_al_digitized_kk_10K.csv")

e_rel = df_h["energy_rel_meV"].values
E_ph = df_h["photon_energy_eV"].values

win_mask = (e_rel >= -235.0) & (e_rel <= 25.0)
e_win = e_rel[win_mask]
psb_mask_h = (e_win >= -200.0) & (e_win <= -15.0)

def _ipl(col: str) -> np.ndarray:
    arr = df_h[col].values[win_mask]
    if col.startswith("A_"):
        return (np.maximum(0.05, E_ph[win_mask]) ** 3) * arr
    return arr

i_harm_10 = _ipl("I_PL_10.0K_norm")
i_anh_10 = _ipl("A_anharmonic_10.0K")
i_harm_10 /= float(np.max(i_harm_10[psb_mask_h]))
i_anh_10 /= float(np.max(i_anh_10[psb_mask_h]))

# Digitized Jin et al. 10 K spectra, excluding edge-border artifacts outside [-235, 10] meV
ref_e = df_r["energy_rel_meV"].values
ref_mask = (ref_e >= -235.0) & (ref_e <= 12.0)
ref_e_w = ref_e[ref_mask]
ref_psb = (ref_e_w >= -200.0) & (ref_e_w <= -15.0)
ref_pbe = df_r["intensity_pbe"].values[ref_mask]
ref_expt = df_r["intensity_expt"].values[ref_mask]
ref_pbe = ref_pbe / float(np.max(ref_pbe[ref_psb]))
ref_expt = ref_expt / float(np.max(ref_expt[ref_psb]))

fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(10.5, 8.6), sharex=True, dpi=300, gridspec_kw={"hspace": 0.16}
)

# Panel (a): 10 K comparison vs Jin et al.
ax1.fill_between(
    ref_e_w, 0.0, ref_expt, color="#bdbdbd", alpha=0.55, label="Jin et al. Experiment (10 K)"
)
ax1.plot(
    ref_e_w, ref_pbe, color="#2ca02c", lw=1.8, linestyle="--", label="Jin et al. PBE Theory (10 K)"
)
ax1.plot(
    e_win,
    i_anh_10,
    color="#1f77b4",
    lw=2.1,
    label=r"HyDePh Anharmonic MD-ACF (10 K, $8\times 8\times 3$ Spliced)",
)
ax1.plot(
    e_win,
    i_harm_10,
    color="#ff7f0e",
    lw=1.7,
    linestyle=":",
    label=r"HyDePh Harmonic GenFn (10 K, $8\times 8\times 3$ Spliced)",
)

for x_pk, lbl_txt, x_txt in [
    (-93.0, "Peak 3\n(-93 meV)", -93.0),
    (-72.2, "Peak 2\n(-72 meV)", -67.5),
    (-36.3, "Peak 1\n(-36 meV)", -36.3),
    (0.0, "ZPL\n(1.096 eV)", 11.5),
]:
    ax1.axvline(x_pk, color="gray", linestyle="-.", lw=0.9, alpha=0.7)
    ax1.text(
        x_txt,
        1.18,
        lbl_txt,
        ha="center",
        va="bottom",
        fontsize=8.5,
        fontweight="bold",
        color="#222222",
    )

ax1.set_xlim(-235.0, 22.0)
ax1.set_ylim(0.0, 1.44)
ax1.set_ylabel("PSB-Normalized Intensity (arb. units)", fontsize=11, fontweight="bold")
ax1.set_title(
    r"(a) 4H-SiC $kk$-$\mathrm{VV}^0$ Emission at 10 K: HyDePh vs. Jin et al. (PRM 5, 084603)",
    fontsize=12,
    fontweight="bold",
)
ax1.legend(loc="upper left", frameon=True, framealpha=0.92, fontsize=8.4)
ax1.grid(True, linestyle=":", alpha=0.55)

# Panel (b): Multi-temperature PL evolution
temps = [
    ("10.0", "#1f77b4", "10 K"),
    ("77.0", "#2ca02c", "77 K"),
    ("150.0", "#ff7f0e", "150 K"),
    ("300.0", "#d62728", "300 K"),
]
# Normalize consistently to the 300 K PSB maximum so relative ZPL attenuation is directly visible
scale_ref = float(np.max(_ipl("A_anharmonic_300.0K")[psb_mask_h]))
for T_str, color, lbl in temps:
    y_t = _ipl(f"A_anharmonic_{T_str}K") / scale_ref
    ax2.plot(e_win, y_t, color=color, lw=2.0, label=f"HyDePh {lbl} (16×3000 MD-ACF)")

ax2.axvline(0.0, color="gray", linestyle=":", lw=1.0, alpha=0.8)
ax2.set_xlabel(r"Photon Energy Relative to ZPL: $E - E_{\mathrm{ZPL}}$ (meV)", fontsize=11, fontweight="bold")
ax2.set_ylabel(r"PL Intensity $I_{\mathrm{PL}}(E)$ (300 K PSB = 1)", fontsize=11, fontweight="bold")
ax2.set_title(
    "(b) Multi-Temperature Photoluminescence Evolution (ZPL Resolved up to 300 K)",
    fontsize=12,
    fontweight="bold",
)
ax2.legend(loc="upper left", frameon=True, framealpha=0.92, fontsize=9.2)
ax2.grid(True, linestyle=":", alpha=0.55)
ax2.set_ylim(0.0, None)

out_fig = repo_root / "assets" / "hydeph_vs_jin_benchmark.png"
out_fig.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(str(out_fig), dpi=300, bbox_inches="tight")
plt.close()
print(f"[+] Saved updated 4H-SiC vs Jin comparison figure to: {out_fig}")
