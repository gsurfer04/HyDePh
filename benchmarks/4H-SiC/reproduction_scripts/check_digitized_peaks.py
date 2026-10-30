import pandas as pd
from scipy.signal import find_peaks

df = pd.read_csv("/mnt/c/Users/gsurf/.gemini/antigravity-cli/brain/88b9fb1f-2111-4d53-bf65-47950df9c321/jin_et_al_digitized_kk_10K.csv")
sub = df[(df["energy_rel_meV"] >= -120) & (df["energy_rel_meV"] <= 10)]

p_pbe, _ = find_peaks(sub["intensity_pbe"], height=20, distance=5)
p_exp, _ = find_peaks(sub["intensity_expt"], height=20, distance=5)

print("PBE peaks (Jin et al.):")
for p in p_pbe:
    row = sub.iloc[p]
    print(f"  E = {row.energy_rel_meV:.2f} meV | intensity = {row.intensity_pbe:.1f}")

print("\nExpt peaks (Jin et al.):")
for p in p_exp:
    row = sub.iloc[p]
    print(f"  E = {row.energy_rel_meV:.2f} meV | intensity = {row.intensity_expt:.1f}")
