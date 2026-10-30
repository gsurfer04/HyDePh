import pandas as pd
import numpy as np

df = pd.read_csv("/home/jamesc/HyDePhTest/SiC-4H-kk/hydeph_run_c1h_consistent/optical_lineshapes.csv")
zpl = df.iloc[(df["energy_rel_meV"]).abs().argmin()]
print(f"ZPL 10K:  rel_E = {zpl['energy_rel_meV']:.2f} meV | Intensity = {zpl['I_PL_10.0K_norm']:.4e}")
print(f"ZPL 300K: rel_E = {zpl['energy_rel_meV']:.2f} meV | Intensity = {zpl['I_PL_300.0K_norm']:.4e}")

psb_10k = df[(df["energy_rel_meV"] < -15) & (df["energy_rel_meV"] > -200)]["I_PL_10.0K_norm"].max()
print(f"Max PSB 10K:  {psb_10k:.4e} | Ratio PSB/ZPL = {psb_10k / zpl['I_PL_10.0K_norm']:.4f}")

psb_300k = df[(df["energy_rel_meV"] < -15) & (df["energy_rel_meV"] > -200)]["I_PL_300.0K_norm"].max()
print(f"Max PSB 300K: {psb_300k:.4e} | Ratio PSB/ZPL = {psb_300k / zpl['I_PL_300.0K_norm']:.4f}")
