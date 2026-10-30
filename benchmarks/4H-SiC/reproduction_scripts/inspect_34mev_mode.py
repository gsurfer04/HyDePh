import os
import numpy as np
import ase.io
from ase.vibrations import Vibrations
from hydeph.calculator import load_foundation_calculator

at_gs = ase.io.read("relaxed_gs_dft.xyz")
vib_dir = "test_embed_552/vib"

from hydeph import ToolkitConfig
from hydeph.lineshape import embed_phonon_supercell

cfg = ToolkitConfig(
    pristine_cell_path="pristine.xyz",
    defect_cell_path="defect.xyz",
    supercell_repeat=(3, 3, 2),
    phonon_supercell_repeat=(5, 5, 2),
    workspace_dir="test_embed_552",
)
calc = load_foundation_calculator("/home/jamesc/HyDePhTest/nep89_20250409.txt")
at_es = ase.io.read("relaxed_es_delta_scf.xyz")
at_large, dr, _ = embed_phonon_supercell(at_gs, at_es, cfg, calc)

at_large.calc = calc
vib = Vibrations(at_large, name=os.path.join(vib_dir, "vib"), delta=0.01)

energies = vib.get_energies()
print(f"Total vibrations: {len(energies)}")

# Find modes near 34.4 meV
e_mev = np.real(energies) * 1000.0
idx_34 = np.where(np.abs(e_mev - 34.4) < 0.3)[0]
print(f"Modes near 34.4 meV: {idx_34}")
for idx in idx_34:
    m = vib.get_mode(idx) # (N, 3)
    norms = np.linalg.norm(m, axis=1)
    top_atoms = np.argsort(norms)[::-1][:6]
    print(f"\nMode {idx} (E = {e_mev[idx]:.2f} meV):")
    for at in top_atoms:
        sym = at_large.get_chemical_symbols()[at]
        pos = at_large.positions[at]
        print(f"  Atom {at:3d} ({sym}): |u| = {norms[at]:.4f}, u = {m[at]}")
