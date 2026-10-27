from hydeph import ToolkitConfig, run_pipeline

# Configure with the DFT-relaxed excited state for CBCN color-center calculation
config = ToolkitConfig(
    # INPUT
    pristine_cell_path="pristine.xyz",
    defect_cell_path="defect.xyz",
    supercell_repeat=(2, 2, 2),
    foundation_potential_path=r"/home/jamesc/HyDePhTest/nep89_20250409.txt",
    #pseudopotentials={
    #    "Cs": "Cs_ONCV_PBE-1.2.upf",
    #    "I": "I_ONCV_PBE-1.2.upf",
    #},
    pseudo_dir=r"/home/jamesc/HyDePhTest/pseudo",
    workspace_dir="hydeph_run_dft_es",
    
    # ELECTRONIC STRUCTURE & GEOMETRY
    charge=0,
    n_procs_qe=1,
    qe_timeout_s=300.0,
    excited_state_method='spin_proxy',
    
    # SAMPLING & RESUMPTION
    temperatures=[10.0, 77.0, 150.0, 300.0],
    max_dft_frames=40,
    enable_anharmonic=True,
    anharmonic_md_steps=1500,
    
    # DNEP ARCHITECTURE HYPERPARAMETERS
    nep_steps=15000,
    
    # SPECTROSCOPY, GRAPHING & 1D CONFIGURATION COORDINATE
    temperature_plot_mode="both",
    show_harmonic_in_anharmonic=True,
    zpl_energy_eV=None
)

if __name__ == "__main__":
    # Run end-to-end turnkey pipeline with ab initio relaxed excited state
    summary = run_pipeline(config)
