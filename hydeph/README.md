# `hydeph`: Hybrid Defect Photoluminescence Package

Python package for training $\Delta$-Neuroevolution Potentials ($\Delta$-NEP) on top of universal foundation models (`nep89`) and computing multi-temperature optical spectra for point defects in solids:

$$
V_{\mathrm{hybrid}}(\mathbf{R}) = V_{\mathrm{base}}(\mathbf{R}) + \Delta V_{\mathrm{NEP}}(\mathbf{R}), \qquad \mathbf{F}_{\mathrm{hybrid}}(\mathbf{R}) = \mathbf{F}_{\mathrm{base}}(\mathbf{R}) + \Delta \mathbf{F}_{\mathrm{NEP}}(\mathbf{R})
$$

See the root [`README.md`](../README.md) for full configuration tables and [`docs/THEORY_WALKTHROUGH.md`](../docs/THEORY_WALKTHROUGH.md) for the theoretical formulation.

---

## Core Inputs

1. **Pristine Unit Cell**: Bulk unit cell structure (`.xyz`, `.cif`, `POSCAR`).
2. **Defect Unit Cell**: Pre-made defect unit cell embedded into the supercell center.
3. **Foundation Potential**: GPUMD universal NEP model (e.g., `nep89_20250409.txt`).
4. **Pseudopotentials**: Element-to-UPF file mapping for Quantum ESPRESSO.
5. **Temperatures**: Target temperatures in Kelvin (e.g., `[10.0, 77.0, 150.0, 300.0]`).

---

## Modules

- `config.py`: `ToolkitConfig` dataclass, fail-loud validation, and UPF discovery.
- `defect_builder.py`: Vectorized supercell tiling, defect site detection, and ground/excited-state DFT relaxation (spin-proxy and $\Delta\text{SCF}$).
- `frame_sampler.py`: 4-phase representative frame generation (Franck-Condon scan, core rattling, foundation MD + FPS, anchor frame priority).
- `qe_runner.py`: Quantum ESPRESSO runner with fixed occupations, atom-scaled thresholds, single-pass parsing, and caching.
- `delta_trainer.py`: Residual extraction, mean energy offset subtraction, automated quality control gates, and GPUMD training.
- `calculator.py`: `HybridDefectCalculator` and `PairedDefectCalculator` combining foundation potential with ground- and optical excited-state $\Delta$-NEPs.
- `lineshape.py`: Large-supercell phonon embedding, $(\hbar\omega)^3$ PL spectra, finite-$T$ DWF, normal mode decomposition, and paired cumulant NVE dynamics.
- `cc_diagram.py`: 1D configuration coordinate curves, vertical transition energies, and Stokes shift analysis.
- `provenance.py`: Environmental provenance, input SHA256 hashes, and manifest tracking for scientific reproducibility.
- `pipeline.py`: Turnkey `run_pipeline` workflow orchestrator with stage resumption.

---

## Quickstart

```python
from hydeph import ToolkitConfig, run_pipeline

config = ToolkitConfig(
    # Core Inputs & Paths
    pristine_cell_path="pristine_unitcell.xyz",
    defect_cell_path="defect_unitcell.xyz",
    foundation_potential_path="nep89_20250409.txt",
    pseudopotentials={
        "Li": "li_pbe_v1.4.uspp.F.UPF",
        "Cl": "cl_pbe_v1.4.uspp.F.UPF",
    },
    pseudo_dir="./pseudo",
    workspace_dir="./hydeph_run",

    # Electronic Structure & Geometry
    supercell_repeat=(2, 2, 2),
    charge=0.0,
    spin_polarized=True,
    spin_multiplicity=2,
    excited_state_method="spin_proxy",

    # Sampling & Resumption
    max_dft_frames=30,
    n_procs_qe=4,

    # Delta-NEP Architecture Hyperparameters
    nep_steps=30000,

    # Spectroscopy, Graphing & 1D Configuration Coordinate
    temperatures=[10.0, 77.0, 150.0, 300.0],
    enable_anharmonic=True,
)

summary = run_pipeline(config)
```

---

## Workspace Outputs

- `relaxed_gs.xyz`, `relaxed_es.xyz`: Relaxed ground- and excited-state supercells.
- `candidate_frames.xyz`: 4-phase candidate frame pool.
- `labeled_dft_frames.xyz`: DFT-labeled structures with energies and forces.
- `nep_delta.txt`: Trained $\Delta$-NEP model.
- `lineshapes.png`, `anharmonic_lineshapes.png`: Spectral density $S(\hbar\omega)$, VDOS, and multi-temperature emission spectra.
- `configuration_coordinate.png`: 2-panel 1D configuration coordinate and Stokes shift diagram.
