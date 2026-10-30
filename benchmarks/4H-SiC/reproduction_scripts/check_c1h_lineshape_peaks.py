import pandas as pd
from scipy.signal import find_peaks

df = pd.read_csv("hydeph_run_c1h/optical_lineshapes.csv")
print("Columns in optical_lineshapes.csv:", df.columns)

for col in ["I_PL_10.0K_norm", "I_PL_77.0K_norm", "I_PL_150.0K_norm", "I_PL_300.0K_norm"]:
    if col in df.columns:
        p, _ = find_peaks(df[col], height=0.01, distance=5)
        print(f"\nPeaks in {col}:")
        for idx in p:
            hw = df["photon_energy_eV"].iloc[idx]
            rel_meV = df["energy_rel_meV"].iloc[idx]
            print(f"  E = {hw:.4f} eV ({rel_meV:6.1f} meV rel) | intensity = {df[col].iloc[idx]:.4f}")
