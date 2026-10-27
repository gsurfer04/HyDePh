from hydeph import ToolkitConfig, run_pipeline

# Configure with the 5 core inputs for color-center calculation
config = ToolkitConfig(
    # 1. Pristine bulk unit cell
    pristine_cell_path="pristine.xyz",
    
    # 2. Pre-made defect unit cell (surrounded by pristine bulk in supercell)
    defect_cell_path="defect.xyz",
    supercell_repeat=(3, 3, 3),
    
    # 3. Foundation universal potential
    foundation_potential_path=r"/home/jamesc/HyDePhTest/nep89_20250409.txt",
    
    # 4. Pseudopotentials for each element
    pseudopotentials={
        "Na": "Na_ONCV_PBE-1.2.upf",
        "Cl": "Cl_ONCV_PBE-1.2.upf",
    },
    pseudo_dir=r"/home/jamesc/HyDePhTest/pseudo",
    
    n_procs_qe=1,
    diagonalization='cg',
    
    # 5. Temperatures of interest (K)
    temperatures=[10.0, 77.0, 150.0, 300.0],
    
    # Graphing options: "combined", "individual", or "both"
    temperature_plot_mode="both",
    show_harmonic_in_anharmonic=True,
    
    # Enable finite-temperature anharmonic MD sampling
    enable_anharmonic=True,
    anharmonic_md_steps=1500,

    # Frame sampling and DFT budget
    n_fc_frames=10,
    n_phonon_frames=10,
    n_md_frames=20,
    max_dft_frames=40,  # Rapid test run budget (set to None for all frames)
    nep_steps=15000,
)

if __name__ == "__main__":
    # Run end-to-end turnkey pipeline
    summary = run_pipeline(config)
