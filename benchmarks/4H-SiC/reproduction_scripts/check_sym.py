import ase.io
import spglib

at = ase.io.read("relaxed_gs_dft.xyz")
sg = spglib.get_spacegroup((at.cell, at.get_scaled_positions(), at.numbers), symprec=1e-3)
print(f"GS spacegroup (symprec=1e-3): {sg}")

sg2 = spglib.get_spacegroup((at.cell, at.get_scaled_positions(), at.numbers), symprec=1e-2)
print(f"GS spacegroup (symprec=1e-2): {sg2}")
