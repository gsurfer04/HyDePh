import numpy as np
import ase.io

at_gs = ase.io.read("relaxed_gs_dft.xyz")
pris = ase.io.read("pristine.xyz")
defect = ase.io.read("defect.xyz")

print(f"Pristine atoms: {len(pris)}, Defect atoms: {len(defect)}, GS atoms: {len(at_gs)}")

# Find missing atoms in defect vs pristine
# Vacancies:
pris_pos = pris.get_positions()
defect_pos = defect.get_positions()
# Let's find the vacancy positions by checking which pristine atoms are not in defect
missing = []
for i, p in enumerate(pris_pos):
    # check min dist to any defect atom
    dists = np.linalg.norm(defect_pos - p, axis=1)
    if np.min(dists) > 0.5:
        missing.append((i, pris.get_chemical_symbols()[i], p))

print("Missing atoms (vacancies):")
for m in missing:
    print(f"  Pristine index {m[0]}: {m[1]} at {m[2]}")

v_si = [m[2] for m in missing if m[1] == 'Si'][0]
v_c = [m[2] for m in missing if m[1] == 'C'][0]

print(f"V_Si at {v_si}")
print(f"V_C at {v_c}")

# In relaxed_gs_dft, find closest C atoms to V_Si and closest Si atoms to V_C
gs_pos = at_gs.get_positions()
syms = np.array(at_gs.get_chemical_symbols())

c_indices = np.where(syms == 'C')[0]
si_indices = np.where(syms == 'Si')[0]

d_c_vsi = np.linalg.norm(gs_pos[c_indices] - v_si, axis=1)
nearest_c = c_indices[np.argsort(d_c_vsi)[:6]]

d_si_vc = np.linalg.norm(gs_pos[si_indices] - v_c, axis=1)
nearest_si = si_indices[np.argsort(d_si_vc)[:6]]

print("\nNearest C atoms to V_Si:")
for idx in nearest_c:
    p = gs_pos[idx]
    d = np.linalg.norm(p - v_si)
    print(f"  Atom {idx} (C): pos={p}, dist={d:.4f} A")

print("\nNearest Si atoms to V_C:")
for idx in nearest_si:
    p = gs_pos[idx]
    d = np.linalg.norm(p - v_c)
    print(f"  Atom {idx} (Si): pos={p}, dist={d:.4f} A")
