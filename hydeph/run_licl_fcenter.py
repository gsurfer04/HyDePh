"""Benchmark driver for the neutral chlorine vacancy F-center (V_Cl^0) in rock-salt LiCl."""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ase.build
import ase.io
from hydeph.config import ToolkitConfig
from hydeph.pipeline import run_pipeline


def main():
    workspace = os.path.abspath("./hydeph_run")
    os.makedirs(workspace, exist_ok=True)

    pristine_unitcell_path = os.path.join(workspace, "licl_pristine_unitcell.xyz")
    pristine_cell = ase.build.bulk("LiCl", "rocksalt", a=5.14, cubic=True)
    ase.io.write(pristine_unitcell_path, pristine_cell)
    print(f"[+] Pristine LiCl unit cell created: {pristine_unitcell_path} ({len(pristine_cell)} atoms)")

    defect_unitcell_path = os.path.join(workspace, "licl_defect_unitcell.xyz")
    defect_cell = pristine_cell.copy()
    cl_idx = [i for i, s in enumerate(defect_cell.get_chemical_symbols()) if s == "Cl"][0]
    del defect_cell[cl_idx]
    ase.io.write(defect_unitcell_path, defect_cell)
    print(f"[+] Pre-made LiCl defect unit cell created: {defect_unitcell_path} ({len(defect_cell)} atoms)")

    config = ToolkitConfig(
        pristine_cell_path=pristine_unitcell_path,
        defect_cell_path=defect_unitcell_path,
        supercell_repeat=(2, 2, 2),
        foundation_potential_path=os.environ.get(
            "NEP_FOUNDATION_PATH",
            os.path.join(workspace, "nep89_20250409.txt")
        ),
        pseudopotentials={
            "Li": "li_pbe_v1.4.uspp.F.UPF",
            "Cl": "cl_pbe_v1.4.uspp.F.UPF",
        },
        pseudo_dir=os.environ.get("ESPRESSO_PSEUDO", os.path.join(workspace, "pseudo")),
        temperatures=[10.0, 77.0, 150.0, 300.0],
        spin_polarized=True,
        spin_multiplicity=2,  # S = 1/2
        tot_charge=0.0,
        ecutwfc=40.0,
        ecutrho=320.0,
        kpoints=(1, 1, 1),
        n_fc_frames=5,
        n_phonon_frames=6,
        n_md_frames=10,
        nep_steps=25000,
        workspace_dir=workspace
    )

    summary = run_pipeline(config, max_dft_frames=4, n_procs_qe=2)
    return summary


if __name__ == "__main__":
    main()
