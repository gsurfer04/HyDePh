import pandas as pd
df = pd.read_csv("test_embed_552/normal_modes.csv")
sub = df[(df["hw_meV"] >= 30.0) & (df["hw_meV"] <= 38.0)]
print(f"Modes between 30 and 38 meV (Total {len(sub)} modes):")
print(sub[["mode_index", "hw_meV", "S_k", "dQ_k_amu12_A", "IPR", "is_localized"]].to_string())
