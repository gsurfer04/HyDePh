import pandas as pd
df = pd.read_csv("test_embed_552/normal_modes.csv")
top = df.sort_values(by="S_k", ascending=False).head(25)
print(top[["mode_index", "hw_meV", "S_k", "dQ_k_amu12_A", "IPR", "is_localized"]].to_string())
print(f"\nTotal S = {df['S_k'].sum():.4f}")
