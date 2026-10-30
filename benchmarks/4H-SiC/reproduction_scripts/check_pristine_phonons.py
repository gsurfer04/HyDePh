import ase.io
import numpy as np
from hydeph.calculator import load_foundation_calculator
from hydeph.lineshape import PhononCalculator

calc = load_foundation_calculator("/home/jamesc/HyDePhTest/nep89_20250409.txt")
pris = ase.io.read("pristine.xyz")
ph_calc = PhononCalculator(calc)
supercell_matrix = np.diag([3, 3, 2])

freqs, evs = ph_calc.calculate_modes(pris, supercell_matrix=supercell_matrix)
freqs_mev = freqs * 1000.0
print(f"Total phonon modes in pristine 3x3x2: {len(freqs_mev)}")
print(f"Min freq: {freqs_mev[3]:.2f} meV")
print(f"Max freq: {freqs_mev[-1]:.2f} meV")

# Check optical and acoustic phonons
import scipy.stats
# Print density of states / histogram
counts, bins = np.histogram(freqs_mev[3:], bins=20)
print("\nPhonon DOS bins (meV):")
for i in range(len(counts)):
    print(f"  {bins[i]:.1f} - {bins[i+1]:.1f} meV: {counts[i]} modes")
