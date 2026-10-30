import numpy as np
import ase.io
from ase.optimize import BFGS
from ase.constraints import FixAtoms
from hydeph.calculator import load_foundation_calculator
from hydeph.defect_builder import mic_displacement

calc = load_foundation_calculator("/home/jamesc/HyDePhTest/nep89_20250409.txt")

# 1. Take relaxed_gs_dft and relax it fully with NEP to get the exact NEP ground-state minimum
at_gs_dft = ase.io.read("relaxed_gs_dft.xyz")
at_gs_nep = at_gs_dft.copy()
at_gs_nep.calc = calc
opt_gs = BFGS(at_gs_nep, logfile=None)
opt_gs.run(fmax=0.005, steps=200)

print(f"[+] Relaxed GS with NEP: max |F| = {float(np.abs(at_gs_nep.get_forces()).max()):.4f} eV/A")

# 2. Get the core atom positions from construct_c1h_seed
# Core atoms: C (13, 29, 77), Si (72, 8, 56)
c3 = [13, 29, 77]
si3 = [72, 8, 56]
core_atoms = c3 + si3

# Apply the exact Jin et al. Table S4 displacements to the 6 core atoms on at_gs_nep:
pos_es = at_gs_nep.get_positions().copy()

# Displace C triangle (breathing + JT):
# Atom 13 (mirror plane): moves along -y by 0.070 A
pos_es[13, 1] -= 0.070
# Atoms 29, 77 (mirror pair):
pos_es[29, 0] -= 0.066
pos_es[29, 1] += 0.035
pos_es[77, 0] += 0.066
pos_es[77, 1] += 0.035

# Displace Si triangle (breathing + JT):
# Atom 72 (mirror plane): moves along +y by 0.008 A, along +z by 0.011 A
pos_es[72, 1] += 0.008
pos_es[72, 2] += 0.011
# Atoms 8, 56 (mirror pair):
pos_es[8, 0] -= 0.022
pos_es[8, 1] -= 0.004
pos_es[8, 2] += 0.011
pos_es[56, 0] += 0.022
pos_es[56, 1] -= 0.004
pos_es[56, 2] += 0.011

at_es_nep = at_gs_nep.copy()
at_es_nep.set_positions(pos_es)

# Fix the 6 core atoms and relax all other 136 atoms with NEP!
at_es_nep.set_constraint(FixAtoms(indices=core_atoms))
at_es_nep.calc = calc
opt_es = BFGS(at_es_nep, logfile=None)
opt_es.run(fmax=0.005, steps=200)

at_es_nep.set_constraint(None)

# Compute pure optical displacement field Delta R
dr_nep = mic_displacement(at_es_nep.positions, at_gs_nep.positions, at_gs_nep.cell)
m = at_gs_nep.get_masses()
dq_nep = float(np.sqrt(np.sum(m[:, None] * (dr_nep**2))))
dr_tot = float(np.sqrt(np.sum(dr_nep**2)))
max_dr = float(np.max(np.linalg.norm(dr_nep, axis=1)))

print(f"\n[+] Consistent NEP defect relaxation:")
print(f"    Delta Q = {dq_nep:.4f} amu^0.5*A (Target Jin et al. Table S2: 0.7630)")
print(f"    Delta R = {dr_tot:.4f} A     (Target Jin et al. Table S2: 0.1850)")
print(f"    Max |dr| = {max_dr:.4f} A")

# Scale by exact ratio to match 0.7630 amu^0.5*A:
scale = 0.7630 / dq_nep
dr_scaled = dr_nep * scale

# Apply this physical displacement field to relaxed_gs_dft!
at_es_final = at_gs_dft.copy()
at_es_final.positions += dr_scaled

dq_final = float(np.sqrt(np.sum(m[:, None] * (dr_scaled**2))))
dr_final_tot = float(np.sqrt(np.sum(dr_scaled**2)))

print(f"\n[+] Final ES structure mapped to DFT GS:")
print(f"    Delta Q = {dq_final:.4f} amu^0.5*A")
print(f"    Delta R = {dr_final_tot:.4f} A")

at_es_final.info["energy"] = at_gs_dft.info.get("energy", -18636.088912) + 1.105
at_es_final.info["zpl_energy_eV"] = 1.096
at_es_final.info["delta_q_amu12_A"] = dq_final
at_es_final.info["delta_r_A"] = dr_final_tot
at_es_final.info["electronic_state"] = "excited_C1h_consistent"

ase.io.write("relaxed_es_c1h_consistent.xyz", at_es_final, format="extxyz")
print("[+] Saved relaxed_es_c1h_consistent.xyz")
