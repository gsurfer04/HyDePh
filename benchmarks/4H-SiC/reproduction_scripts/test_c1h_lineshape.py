import os
import time
import numpy as np
import pandas as pd
import ase.io
from hydeph import ToolkitConfig
from hydeph.calculator import load_foundation_calculator
from hydeph.lineshape import LineshapeEngine

print("==============================================================================")
print("  4H-SiC kk DIVACANCY: C1h JIN ET AL. TRUE GEOMETRY LINESHAPE TEST")
print("==============================================================================")

at_gs = ase.io.read("relaxed_gs_dft.xyz")
at_es = ase.io.read("relaxed_es_c1h_consistent.xyz")
print(f"[+] Loaded relaxed GS ({len(at_gs)} atoms) and consistent C1h ES ({len(at_es)} atoms)")

calc_dir = "hydeph_run_c1h_consistent"
os.makedirs(calc_dir, exist_ok=True)

# Copy cached vib from test_embed_552
import shutil
os.makedirs(os.path.join(calc_dir, "vib"), exist_ok=True)
src_vib = "test_embed_552/vib"
dst_vib = os.path.join(calc_dir, "vib")
for f in os.listdir(src_vib):
    src_f = os.path.join(src_vib, f)
    dst_f = os.path.join(dst_vib, f)
    if os.path.isfile(src_f) and not os.path.exists(dst_f):
        shutil.copy2(src_f, dst_f)
    elif os.path.isdir(src_f) and not os.path.exists(dst_f):
        shutil.copytree(src_f, dst_f)

print("[+] Reusing cached ground-state (5x5x2) Hessian from test_embed_552.")

cfg = ToolkitConfig(
    pristine_cell_path="pristine.xyz",
    defect_cell_path="defect.xyz",
    supercell_repeat=(3, 3, 2),
    foundation_potential_path="/home/jamesc/HyDePhTest/nep89_20250409.txt",
    workspace_dir=calc_dir,
    phonon_supercell_repeat=(5, 5, 2),
    phonon_embedding_method="displacement",
    zpl_energy_eV=1.096,  # Experimental ZPL anchor from Jin et al.
    temperatures=[10.0, 77.0, 150.0, 300.0],
    max_imaginary_modes=5,
    enable_anharmonic=False,  # Generating function approach matching Jin et al.
    allow_fallbacks=False,
)

calc = load_foundation_calculator("/home/jamesc/HyDePhTest/nep89_20250409.txt")
engine = LineshapeEngine(cfg, calc)

t0 = time.time()
print("\n[-] Running LineshapeEngine with true C1h excited-state geometry...")
res = engine.calculate_spectral_density_and_lineshapes(at_gs, at_es)
walltime = time.time() - t0
print(f"\n[+] Lineshape calculation finished in {walltime:.1f} s ({walltime/60.0:.2f} min)!")

df = pd.read_csv(os.path.join(calc_dir, "normal_modes.csv"))
S_tot = float(df["S_k"].sum())
dwf_0 = float(np.exp(-S_tot) * 100.0)

print(f"\n=======================================================")
print(f"  Total Huang-Rhys Factor S = {S_tot:.3f} (Jin et al. PBE Table II: 2.51)")
print(f"  Debye-Waller Factor DWF_0 = {dwf_0:.2f}% (Jin et al. PBE Table III: 7.10%)")
print(f"  Number of phonon modes    = {len(df)}")
print(f"=======================================================")

top_modes = df.sort_values(by="S_k", ascending=False).head(20)
print("\nTop 20 normal modes by S_k:")
print(top_modes[["mode_index", "hw_meV", "S_k", "dQ_k_amu12_A", "IPR", "is_localized"]].to_string())

df_spec = pd.read_csv(os.path.join(calc_dir, "spectral_density.csv"))
from scipy.signal import find_peaks
hw = df_spec["hw_meV"].values
s_harm = df_spec["S_harmonic_per_eV"].values
peaks, props = find_peaks(s_harm, height=0.5, distance=10)
print("\nPeaks in continuous S_harmonic(hw):")
for p in peaks:
    print(f"  hw = {hw[p]:.2f} meV | S = {s_harm[p]:.4f} eV^-1")
