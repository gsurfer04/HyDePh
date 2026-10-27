from hydeph import ToolkitConfig, run_pipeline

# Configure with the 5 core inputs for LiF color-center calculation
config = ToolkitConfig(
    # INPUT
    pristine_cell_path="pristine.xyz",    
    defect_cell_path="defect.xyz",
    supercell_repeat=(3, 3, 3),
    foundation_potential_path=r"/home/jamesc/HyDePhtest/nep89_20250409.txt",
    pseudopotentials={
        "C": "C_ONCV_PBE-1.2.upf",
        "N": "N_ONCV_PBE-1.2.upf",
    },
    pseudo_dir=r"/home/jamesc/HyDePhtest/pseudo",
    
    # ELECTRONIC STRUCTURE & GEOMETRY
    charge=-1,
    spin_polarized=True,
    spin_multiplicity=3,
    excited_state_tot_magnetization=2,
    
    # SAMPLING & RESUMPTION
    max_dft_frames=40,
    temperatures=[10.0, 77.0, 150.0, 300.0],
    n_workers_qe=1,
    
    temperature_plot_mode="both",
    show_harmonic_in_anharmonic=True,
    
    # Enable finite-temperature anharmonic MD sampling
    enable_anharmonic=True,
    anharmonic_md_steps=2000,
)

if __name__ == "__main__":
    # Run end-to-end turnkey pipeline
    summary = run_pipeline(config, n_procs_qe=4)
