import numpy as np
import ase.io
from hydeph.defect_builder import mic_displacement

at_gs = ase.io.read("relaxed_gs_dft.xyz")
cell = at_gs.cell
pos = at_gs.get_positions().copy()

c3 = [13, 29, 77]
si3 = [8, 56, 72]

# Center of C triangle (near V_Si)
v_si_xy = np.mean(pos[c3, :2], axis=0)
v_c_xy = np.mean(pos[si3, :2], axis=0)

print(f"C center: {v_si_xy}, Si center: {v_c_xy}")

# In GS, d(C-C) = 3.3927 A, radius r_C = 3.3927 / sqrt(3) = 1.9588 A
# In ES, target d(C-C) is ~3.46 A -> radius r_C_es = 3.46 / sqrt(3) = 1.9976 A -> dr_radial ~ +0.0388 A
# Wait, let's look at the actual expansion in Jin et al.:
# In Jin et al.:
# GS: d(C-C) = 3.334 A
# ES: d(C1-C2) = 3.456 A, d(C1-C3) = 3.465 A, d(C2-C3) = 3.456 A
# Average d(C-C) in ES = 3.459 A (+0.125 A expansion!)
# In our GS: d(C-C) is 3.3927 A.
# If we expand d(C-C) by +0.125 A, target d(C-C) ~ 3.518 A!
# Let's apply:
# 1. Breathing expansion to C triangle: scale radial distance from center by factor (1 + 0.125/3.3927) = 1.0368
# 2. C1h Jahn-Teller distortion: pair (C13, C29) vs C77 or similar to match C1h symmetry (mirror plane)
# In our cell, Atom 13 is on the mirror plane (x = 1.547002, same as cell x/6 or mirror plane).
# Let's verify mirror plane:
print(f"Atom 13: x = {pos[13, 0]:.6f}, y = {pos[13, 1]:.6f}")
print(f"Atom 29: x = {pos[29, 0]:.6f}, y = {pos[29, 1]:.6f}")
print(f"Atom 77: x = {pos[77, 0]:.6f}, y = {pos[77, 1]:.6f}")
print(f"Midpoint of 29 & 77: x = {(pos[29,0]+pos[77,0])/2:.6f}, y = {(pos[29,1]+pos[77,1])/2:.6f}")
