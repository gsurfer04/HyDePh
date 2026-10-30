import os
import ase.io
from hydeph import ToolkitConfig
from hydeph.qe_runner import write_qe_input, build_delta_scf_occupations
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
    workspace_dir="test_sym_calc",
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

occ_up, occ_dw = build_delta_scf_occupations(
    nelec=568,
    tot_magnetization=2,
    nbnd=300,
    homo_band=283,
    lumo_band=284,
    spin_channel=2,
    lumo_degeneracy=1,
)

calc_dir = "test_sym_calc"
os.makedirs(calc_dir, exist_ok=True)
pw_in_path = os.path.join(calc_dir, "pw.in")

write_qe_input(
    at_seed,
    pw_in_path,
    config=cfg,
    calculation="scf",
    occupations_mode="from_input",
    constrained_occupations=(occ_up, occ_dw),
)

with open(pw_in_path, "r") as f:
    text = f.read()

# Change nosym = .true. to nosym = .false.
text = text.replace("nosym = .true.", "nosym = .false.")
with open(pw_in_path, "w") as f:
    f.write(text)

print("Generated pw.in with nosym = .false.")
