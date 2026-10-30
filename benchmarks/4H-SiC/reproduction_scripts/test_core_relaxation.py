import numpy as np
import ase.io
from ase.optimize import BFGS
from hydeph.calculator import load_foundation_calculator
from hydeph.defect_builder import mic_displacement

# Load relaxed GS
at_gs = ase.io.read("relaxed_gs_dft.xyz")
cell = at_gs.cell

# From construct_c1h_seed.py, we have pos_c1h for the 6 core atoms
from construct_c1h_seed import pos_c1h

at_es_restrained = at_gs.copy()
at_es_restrained.set_positions(pos_c1h)

# Now, let's fix the 6 core atoms (13, 29, 77, 8, 56, 72) and relax all other 136 atoms with NEP!
from ase.constraints import FixAtoms

c3 = [13, 29, 77]
si3 = [8, 56, 72]
core_atoms = c3 + si3
fix_constraint = FixAtoms(indices=core_atoms)
at_es_restrained.set_constraint(fix_constraint)

calc = load_foundation_calculator("/home/jamesc/HyDePhTest/nep89_20250409.txt")
at_es_restrained.calc = calc

opt = BFGS(at_es_restrained, logfile=None)
opt.run(fmax=0.005, steps=200)

# Remove constraint
at_es_restrained.set_constraint(None)

dr = mic_displacement(at_es_restrained.positions, at_gs.positions, cell)
m = at_gs.get_masses()
dq = float(np.sqrt(np.sum(m[:, None] * (dr**2))))
dr_tot = float(np.sqrt(np.sum(dr**2)))
max_dr = float(np.max(np.linalg.norm(dr, axis=1)))

print(f"Relaxed with fixed core atoms matching Jin et al. Table S4:")
print(f"  Delta Q = {dq:.4f} amu^0.5*A (Literature Table S2: 0.763)")
print(f"  Delta R = {dr_tot:.4f} A     (Literature Table S2: 0.185)")
print(f"  Max |dr| = {max_dr:.4f} A")
