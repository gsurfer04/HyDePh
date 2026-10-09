"""
hydeph.run_licl_fcenter
=======================
Turnkey execution script demonstrating the HyDePh toolkit
on the F-center (neutral chlorine vacancy V_Cl^0) in Lithium Chloride (LiCl).

User Inputs:
  1. Pristine bulk unit cell: Rock-salt LiCl (a = 5.14 A, 8 atoms)
  2. Pre-made defect unit cell: 7 atoms (4 Li, 3 Cl with 1 Cl vacancy)
  3. Foundation potential: Universal NEP (nep89_20250409.txt)
  4. Pseudopotentials: SSSP PBE UPF files for Li and Cl
  5. Temperatures of interest: [10.0, 77.0, 150.0, 300.0] K
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ase.build
import ase.io
from hydeph.config import ToolkitConfig
from hydeph.pipeline import run_pipeline


def main():
    workspace = os.path.abspath("./licl_fcenter_run")
    os.makedirs(workspace, exist_ok=True)
    
    # 1. Generate pristine unit cell (conventional rock-salt LiCl, 8 atoms, a = 5.14 A)
    pristine_unitcell_path = os.path.join(workspace, "licl_pristine_unitcell.xyz")
    pristine_cell = ase.build.bulk("LiCl", "rocksalt", a=5.14, cubic=True)
    ase.io.write(pristine_unitcell_path, pristine_cell)
    print(f"[+] Pristine LiCl unit cell created: {pristine_unitcell_path} ({len(pristine_cell)} atoms)")

    # 2. Generate pre-made defect unit cell (7 atoms: 4 Li, 3 Cl with 1 Cl vacancy)
    defect_unitcell_path = os.path.join(workspace, "licl_defect_unitcell.xyz")
    defect_cell = pristine_cell.copy()
    cl_idx = [i for i, s in enumerate(defect_cell.get_chemical_symbols()) if s == "Cl"][0]
    del defect_cell[cl_idx]
    ase.io.write(defect_unitcell_path, defect_cell)
    print(f"[+] Pre-made LiCl defect unit cell created: {defect_unitcell_path} ({len(defect_cell)} atoms)")

    # 3. Configure Toolkit with the 5 core inputs
    config = ToolkitConfig(
        # Input 1: Pristine unit cell
        pristine_cell_path=pristine_unitcell_path,
        
        # Input 2: Pre-made defect unit cell (surrounded by pristine cells in supercell)
        defect_cell_path=defect_unitcell_path,
        supercell_repeat=(2, 2, 2),
        
        # Input 3: General-purpose foundation potential
        foundation_potential_path="/home/jamesc/GPUMD/potentials/nep/nep89_20250409/nep89_20250409.txt",
        
        # Input 4: Pseudopotentials
        pseudopotentials={
            "Li": "li_pbe_v1.4.uspp.F.UPF",
            "Cl": "cl_pbe_v1.4.uspp.F.UPF",
        },
        pseudo_dir="/home/jamesc/quantumespresso/pseudo",
        
        # Input 5: Temperatures of interest (K)
        temperatures=[10.0, 77.0, 150.0, 300.0],
        
        # DFT & training parameters
        spin_polarized=True,
        spin_multiplicity=2, # S = 1/2 for neutral F-center
        tot_charge=0.0,
        ecutwfc=40.0,
        ecutrho=320.0,
        kpoints=(1, 1, 1),
        
        # Sampling & training
        n_fc_frames=5,
        n_phonon_frames=6,
        n_md_frames=10,
        nep_steps=25000,
        workspace_dir=workspace
    )

    # 3. Execute turnkey pipeline with representative DFT subset (4 frames for demonstration)
    summary = run_pipeline(config, max_dft_frames=4, n_procs_qe=2)
    return summary


if __name__ == "__main__":
    main()
