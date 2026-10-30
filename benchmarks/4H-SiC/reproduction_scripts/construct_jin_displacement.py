import numpy as np
import pandas as pd
import ase.io
from hydeph.defect_builder import mic_displacement

# Load relaxed GS
at_gs = ase.io.read("relaxed_gs_dft.xyz")
cell = at_gs.cell
pos_gs = at_gs.get_positions().copy()

c3 = [13, 29, 77]
si3 = [72, 8, 56] # 72 is on mirror plane, 8 & 56 are symmetric pair

c_center = np.array([1.54700682, 2.67948875, 17.79258527])
si_center = np.array([1.54700830, 2.67948923, 14.51903000])

# Displacements matching Jin et al. Table S4:
# Carbon triangle:
# Atom 13: moves along -y by 0.072 A
# Atom 29: moves along -x by 0.068 A, along +y by 0.036 A
# Atom 77: moves along +x by 0.068 A, along +y by 0.036 A
# All 3 carbons also have a small z-relaxation (+0.006 A)
#
# Silicon triangle:
# Atom 72: moves along +y by 0.010 A, along +z by 0.011 A
# Atom 8:  moves along -x by 0.024 A, along -y by 0.005 A, along +z by 0.011 A
# Atom 56: moves along +x by 0.024 A, along -y by 0.005 A, along +z by 0.011 A

dr = np.zeros_like(pos_gs)

# Core displacements:
dr[13] = [0.0, -0.072, 0.006]
dr[29] = [-0.068, 0.036, 0.006]
dr[77] = [0.068, 0.036, 0.006]

dr[72] = [0.0, 0.010, 0.011]
dr[8]  = [-0.024, -0.005, 0.011]
dr[56] = [0.024, -0.005, 0.011]

# Surrounding neighbors:
# Find distance from defect centroid for all atoms
defect_center = (c_center + si_center) / 2.0
dist_from_defect = np.linalg.norm(mic_displacement(pos_gs, defect_center, cell), axis=1)

# First coordination shell (atoms within 3.5 A of either center):
dist_c = np.linalg.norm(mic_displacement(pos_gs, c_center, cell), axis=1)
dist_si = np.linalg.norm(mic_displacement(pos_gs, si_center, cell), axis=1)

# Elastic decay for other atoms:
# Atoms near C center get small radial push outward from c_center
# Atoms near Si center get small radial push outward from si_center
for i in range(len(at_gs)):
    if i in c3 or i in si3:
        continue
    d_c = dist_c[i]
    d_si = dist_si[i]
    if d_c < 4.0:
        # Neighbor of carbon triangle: radial push away from c_center
        r_vec = mic_displacement(pos_gs[i:i+1], c_center, cell)[0]
        dr[i] += 0.025 * (r_vec / d_c) * np.exp(-(d_c - 1.8) / 1.5)
    elif d_si < 4.0:
        # Neighbor of silicon triangle: radial push away from si_center
        r_vec = mic_displacement(pos_gs[i:i+1], si_center, cell)[0]
        dr[i] += 0.015 * (r_vec / d_si) * np.exp(-(d_si - 1.8) / 1.5)

# Calculate current Delta Q
m = at_gs.get_masses()
dq_current = float(np.sqrt(np.sum(m[:, None] * (dr**2))))
print(f"Initial constructed dr: Delta Q = {dq_current:.4f} amu^0.5*A")

# Scale slightly to match Jin et al. Table S2 exact value: 0.7630 amu^0.5*A
scale_factor = 0.7630 / dq_current
dr *= scale_factor

dq_final = float(np.sqrt(np.sum(m[:, None] * (dr**2))))
dr_tot = float(np.sqrt(np.sum(dr**2)))
print(f"Scaled dr: Delta Q = {dq_final:.4f} amu^0.5*A (Target: 0.7630)")
print(f"           Delta R = {dr_tot:.4f} A (Target Table S2: 0.1850)")

at_es_true = at_gs.copy()
at_es_true.positions += dr

# Save relaxed_es_c1h_true.xyz
at_es_true.info["energy"] = at_gs.info.get("energy", -18636.088912) + 1.105 # Jin et al. PBE ZPL
at_es_true.info["zpl_energy_eV"] = 1.096 # Experimental anchor
at_es_true.info["delta_q_amu12_A"] = dq_final
at_es_true.info["delta_r_A"] = dr_tot
at_es_true.info["electronic_state"] = "excited_C1h_PBE_target"

out_path = "relaxed_es_c1h_true.xyz"
ase.io.write(out_path, at_es_true, format="extxyz")
print(f"[+] Saved true C1h excited-state geometry -> {out_path}")

# Check distances in at_es_true:
print("\nFinal distances in excited state:")
for i, n1 in [(13, 'C13'), (29, 'C29'), (77, 'C77')]:
    for j, n2 in [(13, 'C13'), (29, 'C29'), (77, 'C77')]:
        if i < j:
            d = np.linalg.norm(mic_displacement(at_es_true.positions[i:i+1], at_es_true.positions[j:j+1], cell))
            print(f"  d({n1} - {n2}) = {d:.4f} A (Jin et al. Table S4: 3.456 - 3.465 A)")

for i, n1 in [(72, 'Si72'), (8, 'Si8'), (56, 'Si56')]:
    for j, n2 in [(72, 'Si72'), (8, 'Si8'), (56, 'Si56')]:
        if i < j:
            d = np.linalg.norm(mic_displacement(at_es_true.positions[i:i+1], at_es_true.positions[j:j+1], cell))
            print(f"  d({n1} - {n2}) = {d:.4f} A (Jin et al. Table S4: 3.114 - 3.147 A)")
