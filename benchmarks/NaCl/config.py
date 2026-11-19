from hydeph import ToolkitConfig, run_pipeline

config = ToolkitConfig(
    pristine_cell_path="pristine.xyz",
    defect_cell_path="defect.xyz",
    supercell_repeat=(2, 2, 2),
    foundation_potential_path="/home/jamesc/HyDePhTest/nep89_20250409.txt",
    pseudopotentials={
        "Na": "Na_ONCV_PBE-1.2.upf",
        "Cl": "Cl_ONCV_PBE-1.2.upf",
    },
    pseudo_dir="/home/jamesc/HyDePhTest/pseudo",
    workspace_dir=".",
    allow_fallbacks=False,
    charge=0,
    spin_multiplicity=2,
    excited_state_method="delta_scf",
    paired_delta_nep=True,
    # Unified property-aware Stage 6 workflow
    phonon_supercell_repeat="auto",
    phonon_embedding_method="force",
    hessian_embedding_mode="auto",
    spliced_bulk_potential="auto",
    phonon_force_state="auto",
    use_dft_core_forces="auto",
    anharmonic_md_surface="auto",
    temperatures=[10.0, 77.0, 150.0, 300.0],
    enable_anharmonic=True,
    anharmonic_md_steps=3000,
    anharmonic_n_trajectories=16,
    anharmonic_thermalization_steps=400,
    anharmonic_acf_damping_meV=2.0,
    dephasing_gamma_meV=1.0,
    dephasing_temperature_coefficient=0.0,
    temperature_plot_mode="both",
    show_harmonic_in_anharmonic=True,
    plot_configuration_coordinate=True,
    zpl_energy_eV=1.272897,
    parallel_hessian=True,
    parallel_hessian_workers=16,
)

if __name__ == "__main__":
    summary = run_pipeline(config)
