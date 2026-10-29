from hydeph import ToolkitConfig, run_pipeline

# Configure with the 5 core inputs for color-center calculation
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
    workspace_dir="hydeph_run",
    allow_fallbacks=False,
    
    # ELECTRONIC STRUCTURE & GEOMETRY
    charge=0,
    spin_multiplicity=3,
    
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
    zpl_energy_eV=1.027412,
)

if __name__ == "__main__":
    # Run end-to-end turnkey pipeline
    summary = run_pipeline(config)
