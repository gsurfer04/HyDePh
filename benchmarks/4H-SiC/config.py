from hydeph import ToolkitConfig, run_pipeline

config = ToolkitConfig(
    # INPUT
    pristine_cell_path="pristine.xyz",
    defect_cell_path="defect.xyz",
    supercell_repeat=(3, 3, 2),
    foundation_potential_path=r"/home/jamesc/HyDePhTest/nep89_20250409.txt",
    pseudopotentials={
        "Si": "Si_ONCV_PBE-1.2.upf",
        "C": "C_ONCV_PBE-1.2.upf",
    },
    pseudo_dir=r"/home/jamesc/HyDePhTest/pseudo",
    workspace_dir="hydeph_run_65ry",
    allow_fallbacks=False,
    
    # ELECTRONIC STRUCTURE & GEOMETRY (HIGH CONVERGENCE 65 Ry)
    charge=0,
    spin_multiplicity=3,
    ecutwfc=65.0,
    ecutrho=260.0,
    mixing_beta=0.20,
    
    # DELTA-SCF EXCITATION & PAIRED DELTA-NEP
    excited_state_method="delta_scf",
    paired_delta_nep=True,
    dft_es_candidates="all",
    delta_scf_homo_band=285,
    delta_scf_lumo_band=286,
    delta_scf_lumo_degeneracy=2,
    delta_scf_spin_channel=1,
    
    # SAMPLING & RESUMPTION
    temperatures=[10.0, 77.0, 150.0, 300.0],
    max_dft_frames=40,
    enable_anharmonic=True,
    anharmonic_md_steps=1500,
    anharmonic_acf_damping_meV=2.0,
    dephasing_temperature_coefficient=5.5e-5,
    max_test_force_rmse_eV_A=0.15,
    
    # DNEP ARCHITECTURE HYPERPARAMETERS
    nep_steps=15000,
    
    # SPECTROSCOPY, GRAPHING & 1D CONFIGURATION COORDINATE
    temperature_plot_mode="both",
    show_harmonic_in_anharmonic=True,
)

if __name__ == "__main__":
    summary = run_pipeline(config)
