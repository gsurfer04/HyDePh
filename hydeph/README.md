# `hydeph`: Hybrid Defect Photoluminescence Package

Python package for training paired $\Delta$-Neuroevolution Potentials ($\Delta V_{\mathrm{GS}}$ and $\Delta V_{\mathrm{ES}}$) on top of universal foundation models (`nep89`) and computing first-principles multi-temperature optical spectra for point defects in solids:

$$
V_{\mathrm{hybrid}, S}(\mathbf{R}) = V_{\mathrm{base}}(\mathbf{R}) + N_{\mathrm{atoms}}\bar{\Delta e}_S + \Delta V_{\mathrm{NEP}, S}(\mathbf{R}), \qquad \mathbf{F}_{\mathrm{hybrid}, S}(\mathbf{R}) = \mathbf{F}_{\mathrm{base}}(\mathbf{R}) + \Delta \mathbf{F}_{\mathrm{NEP}, S}(\mathbf{R})
$$

See the root [`README.md`](../README.md) for full configuration tables and [`docs/THEORY_WALKTHROUGH.md`](../docs/THEORY_WALKTHROUGH.md) for the theoretical formulation.

---

## Core Inputs

1. **Pristine Unit Cell**: Bulk unit cell structure (`.xyz`, `.cif`, or `POSCAR`).
2. **Defect Unit Cell**: Defect unit cell embedded into the supercell center.
3. **Foundation Potential**: GPUMD universal NEP model (e.g., `nep89_20250409.txt`).
4. **Pseudopotentials**: Element-to-UPF mapping for Quantum ESPRESSO.
5. **Temperatures**: Target temperatures in Kelvin (e.g., `[10.0, 77.0, 150.0, 300.0]`).

---

## Modules

- `config.py`: `ToolkitConfig` dataclass, range verification, and UPF discovery.
- `defect_builder.py`: Vectorized supercell tiling, minimum-image defect detection, and ground/excited-state DFT optimization.
- `frame_sampler.py`: 4-phase representative frame generation (Franck-Condon scan, core rattling, foundation MD + FPS, anchor frame priority).
- `qe_runner.py`: Quantum ESPRESSO runner with fixed occupations, atom-scaled convergence thresholds, single-pass parsing, and constrained $\Delta\text{SCF}$ occupations.
- `delta_trainer.py`: Baseline subtraction, mean energy offset subtraction ($\bar{\Delta e}_S$), automated quality control gates, and GPUMD training.
- `calculator.py`: `HybridDefectCalculator` and `PairedDefectCalculator` combining foundation potentials with paired ground- and optical excited-state $\Delta$-NEPs.
- `lineshape.py`: Large-supercell phonon embedding, $(\hbar\omega)^3$ PL spectra, finite-$T$ DWF, normal mode decomposition, and paired cumulant NVE dynamics.
- `cc_diagram.py`: Paired 1D configuration coordinate curves, vertical transition energies, and Stokes shift analysis.
- `provenance.py`: Environmental provenance, input SHA256 hashes, and manifest tracking for scientific reproducibility.
- `validator.py`: Dry-run CLI structure, valence, and spin validator.
- `pipeline.py`: Pipeline orchestrator with stage resumption (`--from-stage 1..6`).

---

## Quickstart

```python
from hydeph import ToolkitConfig, run_pipeline

config = ToolkitConfig(
    # Core Inputs & Paths
    pristine_cell_path="nacl_pristine_unitcell.xyz",
    defect_cell_path="nacl_defect_unitcell.xyz",
    foundation_potential_path="nep89_20250409.txt",
    pseudopotentials={
        "Na": "Na_ONCV_PBE-1.2.upf",
        "Cl": "Cl_ONCV_PBE-1.2.upf",
    },
    pseudo_dir="./pseudo",
    workspace_dir="./nacl_run",
    allow_fallbacks=False,

    # Electronic Structure & Geometry
    supercell_repeat=(2, 2, 2),
    charge=0.0,
    spin_polarized=True,
    spin_multiplicity=2,
    excited_state_method="delta_scf",
    paired_delta_nep=True,

    # Sampling & Resumption
    max_dft_frames=30,
    n_procs_qe=1,
    n_workers_qe=1,

    # Delta-NEP Architecture Hyperparameters
    nep_steps=20000,

    # Spectroscopy, Graphing & 1D Configuration Coordinate
    temperatures=[10.0, 77.0, 150.0, 300.0],
    enable_anharmonic=True,
    anharmonic_md_steps=1500,
)

summary = run_pipeline(config)
```

---

## Workspace Outputs

- `relaxed_gs.xyz`, `relaxed_es.xyz`: Relaxed ground- and excited-state supercells.
- `candidate_frames.xyz`: 4-phase candidate frame pool.
- `labeled_dft_frames.xyz`, `labeled_frames_es.xyz`: DFT-labeled structures with energies and forces.
- `nep_delta_gs.txt`, `nep_delta_es.txt`: Trained ground- and optical excited-state $\Delta$-NEP models.
- `delta_nep_qc_report.json`, `delta_nep_parity.png`: Test set parity and quality control gate evaluations.
- `normal_modes.csv`: Vibrational frequencies $\hbar\omega_k$, Huang-Rhys factors $S_k$, and localization $\mathrm{IPR}_k$.
- `optical_lineshapes.csv`, `spectral_density.csv`: Numerical PL emission and spectral density curves across temperatures.
- `lineshapes.png`, `anharmonic_lineshapes.png`: Multi-temperature optical emission figures.
- `configuration_coordinate.png`: 2-panel 1D configuration coordinate and Stokes shift diagram.
- `provenance.json`, `config_resolved.json`: Cryptographic input hashes, software environments, and execution metadata.
