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

- `config.py`: `ToolkitConfig` dataclass and input validation.
- `defect_builder.py`: Supercell tiling, defect site detection, and ground/excited-state relaxation (foundation potential or spin-proxy DFT).
- `frame_sampler.py`: 4-phase frame sampling (Franck-Condon scan, core rattling, foundation MD + FPS, interleaved queue).
- `qe_runner.py`: Quantum ESPRESSO (`pw.x` / `pw.exe`) runner with single-point labeling, BFGS relaxation, caching, and type-balanced resumption.
- `delta_trainer.py`: Residual extraction ($\Delta E$, $\Delta \mathbf{F}$) and GPUMD `nep` training.
- `calculator.py`: `HybridDefectCalculator` combining $V_{\mathrm{base}}$ and $\Delta V_{\mathrm{NEP}}$.
- `lineshape.py`: Normal modes, Huang-Rhys factors, quantum generating functions, and anharmonic Langevin MD.
- `cc_diagram.py`: 1D configuration coordinate potential surfaces and Stokes shift analysis.
- `pipeline.py`: End-to-end `run_pipeline` orchestrator.

---

## Quickstart

```python
from hydeph import ToolkitConfig, run_pipeline

config = ToolkitConfig(
    pristine_cell_path="pristine_unitcell.xyz",
    defect_cell_path="defect_unitcell.xyz",
    supercell_repeat=(2, 2, 2),
    foundation_potential_path="nep89_20250409.txt",
    pseudopotentials={
        "Li": "li_pbe_v1.4.uspp.F.UPF",
        "Cl": "cl_pbe_v1.4.uspp.F.UPF",
    },
    pseudo_dir="./pseudo",
    temperatures=[10.0, 77.0, 150.0, 300.0],
    workspace_dir="./hydeph_run",
)

summary = run_pipeline(config, n_procs_qe=4)
```

---

## Workspace Outputs

- `relaxed_gs.xyz`, `relaxed_es.xyz`: Relaxed ground- and excited-state supercells.
- `candidate_frames.xyz`: 4-phase candidate frame pool.
- `labeled_dft_frames.xyz`: DFT-labeled structures with energies and forces.
- `nep_delta.txt`: Trained $\Delta$-NEP model.
- `lineshapes.png`, `anharmonic_lineshapes.png`: Spectral density $S(\hbar\omega)$, VDOS, and multi-temperature emission spectra.
- `configuration_coordinate.png`: 2-panel 1D configuration coordinate and Stokes shift diagram.
