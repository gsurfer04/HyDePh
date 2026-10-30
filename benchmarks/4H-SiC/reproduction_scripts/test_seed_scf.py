import os
import time
import ase.io
from hydeph import ToolkitConfig
from hydeph.qe_runner import QERunner
from construct_c1h_seed import pos_c1h

at_gs = ase.io.read("relaxed_gs_dft.xyz")
at_seed = at_gs.copy()
at_seed.set_positions(pos_c1h)

cfg = ToolkitConfig(
    pristine_cell_path="pristine.xyz",
    defect_cell_path="defect.xyz",
    supercell_repeat=(3, 3, 2),
    foundation_potential_path="/home/jamesc/HyDePhTest/nep89_20250409.txt",
    pseudopotentials={"Si": "Si_ONCV_PBE-1.2.upf", "C": "C_ONCV_PBE-1.2.upf"},
    pseudo_dir="/home/jamesc/HyDePhTest/pseudo",
    workspace_dir="test_seed_calc",
    charge=0,
    spin_multiplicity=3,
    ecutwfc=65.0,
    ecutrho=260.0,
    mixing_beta=0.20,
    pw_exe="/home/jamesc/.local/bin/pw.x",
    n_procs_qe=1,
    n_workers_qe=1,
    qe_timeout_s=7200.0,
    allow_fallbacks=False,
)

qe = QERunner(cfg, n_procs=1, pw_exe="/home/jamesc/.local/bin/pw.x", timeout_s=7200.0)
calc_dir = "test_seed_calc"
os.makedirs(calc_dir, exist_ok=True)

print("[-] Running SCF on C1h distorted seed structure...")
t0 = time.time()
res = qe.label_single_frame(
    at_seed,
    calc_dir=calc_dir,
    timeout_s=3600.0,
)
walltime = time.time() - t0
print(f"SCF finished in {walltime:.1f} s ({walltime/60.0:.2f} min)!")
print(f"Energy = {res.info.get('energy', 0.0):.6f} eV")
