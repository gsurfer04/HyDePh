import numpy as np
import ase.io
from hydeph.defect_builder import mic_displacement
from ase.build import make_supercell

unit_pris = ase.io.read("pristine.xyz")
pris_super = make_supercell(unit_pris, np.diag([3, 3, 2]))
gs = ase.io.read("relaxed_gs_dft.xyz")

print(f"Pristine supercell: {len(pris_super)} atoms, GS: {len(gs)} atoms")

pris_pos = pris_super.get_positions()
cell = gs.cell

missing = []
for i, p in enumerate(pris_pos):
    dr = mic_displacement(gs.positions, p[None, :], cell)
    d = np.min(np.linalg.norm(dr, axis=1))
    if d > 0.8:
        sym = pris_super.get_chemical_symbols()[i]
        missing.append((i, sym, p))

for m in missing:
    print(f"Missing {m[1]} at {m[2]}")

v_si = [m[2] for m in missing if m[1] == 'Si'][0]
v_c = [m[2] for m in missing if m[1] == 'C'][0]

dr_vsi = mic_displacement(gs.positions, v_si[None, :], cell)
dist_vsi = np.linalg.norm(dr_vsi, axis=1)

dr_vc = mic_displacement(gs.positions, v_c[None, :], cell)
dist_vc = np.linalg.norm(dr_vc, axis=1)

syms = np.array(gs.get_chemical_symbols())
c_idx = np.where(syms == 'C')[0]
si_idx = np.where(syms == 'Si')[0]

c_around_vsi = c_idx[np.argsort(dist_vsi[c_idx])[:5]]
si_around_vc = si_idx[np.argsort(dist_vc[si_idx])[:5]]

print("\nClosest C atoms to V_Si:")
for idx in c_around_vsi:
    print(f"  Atom {idx:3d}: dist = {dist_vsi[idx]:.4f} A, pos = {gs.positions[idx]}")

print("\nClosest Si atoms to V_C:")
for idx in si_around_vc:
    print(f"  Atom {idx:3d}: dist = {dist_vc[idx]:.4f} A, pos = {gs.positions[idx]}")

print("\nDistances between 3 closest C atoms in GS:")
c3 = c_around_vsi[:3]
for i in range(3):
    for j in range(i+1, 3):
        dr = mic_displacement(gs.positions[c3[i]:c3[i]+1], gs.positions[c3[j]:c3[j]+1], cell)
        d = np.linalg.norm(dr)
        print(f"  d(C{c3[i]} - C{c3[j]}) = {d:.4f} A")

print("\nDistances between 3 closest Si atoms in GS:")
si3 = si_around_vc[:3]
for i in range(3):
    for j in range(i+1, 3):
        dr = mic_displacement(gs.positions[si3[i]:si3[i]+1], gs.positions[si3[j]:si3[j]+1], cell)
        d = np.linalg.norm(dr)
        print(f"  d(Si{si3[i]} - Si{si3[j]}) = {d:.4f} A")
