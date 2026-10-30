import numpy as np
import ase.io
from hydeph.defect_builder import mic_displacement

at_gs = ase.io.read("relaxed_gs_dft.xyz")
cell = at_gs.cell
pos_gs = at_gs.get_positions().copy()

# GS distances:
# C triangle: 13, 29, 77
# Si triangle: 72, 8, 56
# Center of C triangle:
c_center = np.array([1.54700682, 2.67948875, 17.79258527])
# Center of Si triangle:
si_center = np.array([1.5470083, 2.67948923, 14.51903000])

# In Jin et al. Table S4:
# GS PBE:
# d(C-C) = 3.334 A
# ES PBE:
# d(C1-C2) = 3.456 A, d(C1-C3) = 3.465 A, d(C2-C3) = 3.456 A
# Average d(C-C) expansion = +0.125 A (ratio = 3.459 / 3.334 = 1.0375)
# JT splitting: d(C1-C3) - d(C1-C2) = +0.009 A
#
# GS Si-Si: 3.103 A
# ES Si-Si:
# d(Si1-Si2) = 3.114 A, d(Si1-Si3) = 3.147 A, d(Si2-Si3) = 3.114 A
# Average d(Si-Si) expansion = +0.022 A (ratio = 3.125 / 3.103 = 1.0071)
# JT splitting: d(Si1-Si3) - d(Si1-Si2) = +0.033 A

# Let's compute the displacement vectors in our cell:
# For C triangle:
# We want:
# d(C29-C77) = 3.3927 + 0.125 + 0.006 = 3.524 A
# d(C13-C29) = 3.3927 + 0.125 - 0.003 = 3.515 A
# d(C13-C77) = 3.3927 + 0.125 - 0.003 = 3.515 A

# Let's adjust pos of 13, 29, 77 precisely to match these exact target distances in our cell!
# Atom 13 is on mirror plane: pos[13, 0] = c_center[0], pos[13, 1] = c_center[1] - y13
# Atoms 29, 77: pos[29, 0] = c_center[0] - x_pair, pos[77, 0] = c_center[0] + x_pair
# pos[29, 1] = pos[77, 1] = c_center[1] + y_pair
# Then:
# d(29-77) = 2 * x_pair = 3.524 A -> x_pair = 1.762 A (was 1.696 A, delta_x = +0.066 A)
# d(13-29)^2 = x_pair^2 + (y13 + y_pair)^2 = 3.515^2
# (y13 + y_pair)^2 = 3.515^2 - 1.762^2 = 12.355 - 3.105 = 9.250 -> y13 + y_pair = 3.041 A
# In GS: y13 = 1.9588 A, y_pair = 0.9794 A, sum = 2.9382 A (height of equilateral triangle)
# Target height = 3.041 A (expansion of 0.103 A).
# Distribute height expansion between y13 and y_pair proportionally (2:1):
# delta_y13 = 0.103 * (2/3) = 0.069 A
# delta_y_pair = 0.103 * (1/3) = 0.034 A

pos_c1h = pos_gs.copy()
# Atom 13: moves along -y by 0.069 A
pos_c1h[13, 1] -= 0.069
# Atom 29: moves along -x by 0.066 A and along +y by 0.034 A
pos_c1h[29, 0] -= 0.066
pos_c1h[29, 1] += 0.034
# Atom 77: moves along +x by 0.066 A and along +y by 0.034 A
pos_c1h[77, 0] += 0.066
pos_c1h[77, 1] += 0.034

# For Si triangle:
# Atom 72 is on mirror plane (y > center):
# In GS: d(Si-Si) = 3.1601 A
# Target:
# d(Si8-Si56) = 3.1601 + 0.044 = 3.204 A -> x_pair = 3.204 / 2 = 1.602 A (was 1.580 A, delta_x = +0.022 A)
# d(Si72-Si8) = 3.1601 + 0.011 = 3.171 A
# height^2 = 3.171^2 - 1.602^2 = 10.055 - 2.566 = 7.489 -> height = 2.737 A (was 2.7368 A)
# delta height is very small (~0.001 A)
pos_c1h[8, 0] -= 0.022
pos_c1h[56, 0] += 0.022

# Let's check distances with these positions:
at_test = at_gs.copy()
at_test.set_positions(pos_c1h)

print("Distances in constructed C1h structure:")
c3 = [13, 29, 77]
si3 = [72, 8, 56]
for i, name in [(13, 'C13'), (29, 'C29'), (77, 'C77')]:
    for j, name2 in [(13, 'C13'), (29, 'C29'), (77, 'C77')]:
        if i < j:
            dr = mic_displacement(at_test.positions[i:i+1], at_test.positions[j:j+1], cell)
            print(f"  d({name} - {name2}) = {np.linalg.norm(dr):.4f} A")

for i, name in [(72, 'Si72'), (8, 'Si8'), (56, 'Si56')]:
    for j, name2 in [(72, 'Si72'), (8, 'Si8'), (56, 'Si56')]:
        if i < j:
            dr = mic_displacement(at_test.positions[i:i+1], at_test.positions[j:j+1], cell)
            print(f"  d({name} - {name2}) = {np.linalg.norm(dr):.4f} A")

# Check Delta Q between at_test and at_gs:
dr_test = mic_displacement(at_test.positions, at_gs.positions, cell)
m = at_gs.get_masses()
dq_test = float(np.sqrt(np.sum(m[:, None] * (dr_test**2))))
max_dr = float(np.max(np.linalg.norm(dr_test, axis=1)))
print(f"\nDelta Q = {dq_test:.4f} amu^0.5*A, max |dr| = {max_dr:.4f} A")
print(f"Literature Table S2: Delta Q = 0.763 amu^0.5*A, Delta R = 0.185 A")
