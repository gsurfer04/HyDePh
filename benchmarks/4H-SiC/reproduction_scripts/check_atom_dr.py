import numpy as np
import ase.io
from hydeph.defect_builder import mic_displacement

gs = ase.io.read("relaxed_gs_dft.xyz")
es = ase.io.read("relaxed_es_delta_scf.xyz")
cell = gs.cell

dr = mic_displacement(es.positions, gs.positions, cell)
m = gs.get_masses()
dq_per_atom = np.sqrt(m[:, None] * (dr**2))

c3 = [13, 29, 77]
si3 = [8, 56, 72]

print("In relaxed_es_delta_scf.xyz:")
print("C atoms (13, 29, 77):")
for idx in c3:
    print(f"  Atom {idx:2d} (C): dr = {dr[idx]}, |dr| = {np.linalg.norm(dr[idx]):.4f} A, dQ = {np.linalg.norm(dq_per_atom[idx]):.4f}")

print("\nSi atoms (8, 56, 72):")
for idx in si3:
    print(f"  Atom {idx:2d} (Si): dr = {dr[idx]}, |dr| = {np.linalg.norm(dr[idx]):.4f} A, dQ = {np.linalg.norm(dq_per_atom[idx]):.4f}")
