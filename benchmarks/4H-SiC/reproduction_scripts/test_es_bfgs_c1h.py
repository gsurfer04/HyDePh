import os
import sys
import time
import numpy as np
import ase.io
from hydeph import ToolkitConfig
from hydeph.qe_runner import QERunner, build_delta_scf_occupations
from construct_c1h_seed import pos_c1h

print("==============================================================================")
print("  TEST: BFGS EXCITED-STATE C1h RELAXATION WITH SEED GEOMETRY")
print("==============================================================================")

at_gs = ase.io.read("relaxed_gs_dft.xyz")
at_seed = at_gs.copy()
at_seed.set_positions(pos_c1h)

cfg = ToolkitConfig(
    pristine_cell_path="pristine.xyz",
    defect_cell_path="defect.xyz",
    supercell_repeat=(3, 3, 3),
    foundation_potential_path="/home/jamesc/HyDePhTest/nep89_20250409.txt",
    pseudopotentials={"Si": "Si_ONCV_PBE-1.2.upf", "C": "C_ONCV_PBE-1.2.upf"},
    pseudo_dir="/home/jamesc/HyDePhTest/pseudo",
    workspace_dir="test_es_bfgs_c1h",
    charge=0,
    spin_multiplicity=3,
    ecutwfc=65.0,
    ecutrho=260.0,
    mixing_beta=0.15,
    pw_exe="/home/jamesc/.local/bin/pw.x",
    n_procs_qe=1,
    n_workers_qe=1,
    qe_timeout_s=7200.0,
    allow_fallbacks=False,
)

qe = QERunner(cfg, n_procs=1, pw_exe="/home/jamesc/.local/bin/pw.x", timeout_s=7200.0)

# Physical minority-spin transition: homo=283 -> lumo=284, deg=1
occ_up, occ_dw = build_delta_scf_occupations(
    nelec=568,
    tot_magnetization=2,
    nbnd=300,
    homo_band=283,
    lumo_band=284,
    spin_channel=2,
    lumo_degeneracy=1,
)

calc_dir = "test_es_bfgs_c1h"
os.makedirs(calc_dir, exist_ok=True)

print("\n[-] Launching QE BFGS Relaxation from C1h seed structure...")
t0 = time.time()
try:
    at_es = qe.run_relaxation(
        at_seed,
        calc_dir=calc_dir,
        tot_magnetization=2,
        nspin=2,
        occupations_mode="from_input",
        constrained_occupations=(occ_up, occ_dw),
        force_rerun=True,
        timeout_s=3600.0,
    )
    walltime = time.time() - t0
    print(f"\n[+] Finished in {walltime:.1f} s ({walltime/60.0:.2f} min)!")
    e_es = at_es.info.get("energy", 0.0)
    print(f"Energy = {e_es:.6f} eV")
except Exception as e:
    print(f"Failed with exception: {e}")
