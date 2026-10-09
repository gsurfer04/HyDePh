# `defect_delta_nep`: Turnkey Color-Center $\Delta$-NEP Toolkit

A modular Python framework for training ultra-fast, high-fidelity $\Delta$-Machine Learning Potentials ($\Delta$-NEP) on top of universal foundation models (such as `nep89`) for color centers in solids.

---

## 1. Overview

Universal machine learning potentials (e.g. `nep89`) provide broad lattice stability and elastic properties across the periodic table, but lack defect-specific many-body correlation corrections (such as deep defect state localization, Jahn-Teller distortions, and Franck-Condon relaxation forces). 

`defect_delta_nep` automates the construction of a targeted $\Delta$-correction potential:
$$V_{\mathrm{hybrid}}(\mathbf{R}) = V_{\mathrm{base}}(\mathbf{R}) + \Delta V_{\mathrm{NEP}}(\mathbf{R})$$
$$\mathbf{F}_{\mathrm{hybrid}}(\mathbf{R}) = \mathbf{F}_{\mathrm{base}}(\mathbf{R}) + \Delta \mathbf{F}_{\mathrm{NEP}}(\mathbf{R})$$

where $\Delta V$ is learned from a compact set of first-principles DFT calculations (Quantum ESPRESSO) localized around the defect manifold.

---

## 2. The 5 Core User Inputs

The toolkit requires only 5 primary inputs from the user:

1. **Pristine Unit Cell**: ASE-readable structure (`.xyz`, `.cif`, `POSCAR`) for the bulk unit cell.
2. **Defect Unit Cell**: Pre-made defect unit cell (containing the defect vacancy, substitution, or core). The toolkit automatically embeds it into the center block of the supercell, surrounded by pristine bulk unit cells, and auto-detects the defect coordinate.
3. **Foundation Potential**: Path to GPUMD universal NEP potential (e.g., `nep89_20250409.txt`).
4. **Pseudopotentials**: Dictionary mapping element symbols to UPF pseudopotential file paths (e.g., SSSP PBE library).
5. **Temperatures of Interest**: List of operating temperatures in Kelvin (e.g., `[10.0, 77.0, 150.0, 300.0]`).

---

## 3. Four-Phase Representative Sampling Protocol

To ensure $\Delta$-NEP learns accurate defect energetics with minimal DFT calls, candidate training frames are collected via 4 complementary physical phases:

| Phase | Strategy | Purpose |
|---|---|---|
| **Phase 1: Franck-Condon Path** | Linear scan $\mathbf{R}(\lambda) = \mathbf{R}_{\mathrm{GS}} + \lambda (\mathbf{R}_{\mathrm{ES}} - \mathbf{R}_{\mathrm{GS}})$ ($\lambda \in [-0.2, 1.2]$) | Captures optical relaxation coordinate and potential curvature |
| **Phase 2: Defect-Core Rattling** | Local thermal Gaussian displacements ($r < 6.0\ \text{Å}$, $\sigma \in [0.08, 0.14]\ \text{Å}$) | Samples frozen-phonon distortions around the defect cavity |
| **Phase 3: Foundation MD + FPS** | Fast NVT MD with `nep89` ($T = 500\ \text{K}$) followed by Farthest Point Sampling | Explores high-temperature configurational diversity without expensive DFT |
| **Phase 4: Interleaved Priority** | Balances ground, excited, and thermal configurations from frame 1 | Guarantees balanced representation even for tight DFT budgets |

---

## 4. Package Architecture

```
defect_delta_nep/
├── __init__.py           # Package exports
├── config.py             # ToolkitConfig dataclass & input validator
├── defect_builder.py     # Supercell expansion, vacancy creation, GS/ES relaxation
├── frame_sampler.py      # 4-Phase representative frame sampler (FC, rattle, MD+FPS)
├── qe_runner.py          # MPI-accelerated Quantum ESPRESSO pw.x runner with caching
├── delta_trainer.py      # Delta property extraction & GPU GPUMD NEP trainer
├── calculator.py         # HybridDefectCalculator (V_base + Delta_V)
├── lineshape.py          # Normal modes, Huang-Rhys S, S(hw), & multi-T lineshapes
├── pipeline.py           # Top-level turnkey pipeline orchestrator
└── run_licl_fcenter.py   # Turnkey benchmark driver for LiCl F-center
```

---

## 5. Quickstart Example

```python
from defect_delta_nep import ToolkitConfig, run_pipeline

# 1. Configure the 5 inputs
config = ToolkitConfig(
    pristine_cell_path="licl_unitcell.xyz",
    defect_type="vacancy",
    defect_site_element="Cl",
    supercell_repeat=(2, 2, 2),
    foundation_potential_path="/path/to/nep89_20250409.txt",
    pseudopotentials={
        "Li": "li_pbe_v1.4.uspp.F.UPF",
        "Cl": "cl_pbe_v1.4.uspp.F.UPF"
    },
    temperatures=[10.0, 77.0, 150.0, 300.0]
)

# 2. Run automated pipeline
summary = run_pipeline(config)
```

---

## 6. Physical Outputs

- `relaxed_gs.xyz` & `relaxed_es.xyz`: Ground and excited equilibrium geometries.
- `candidate_frames.xyz`: 4-phase representative frame ensemble.
- `labeled_dft_frames.xyz`: First-principles DFT energies and atomic forces.
- `nep_delta.txt`: Trained defect $\Delta$-NEP potential.
- `licl_fcenter_lineshapes.png`: Publication-grade electron-phonon spectral density $S(\hbar\omega)$ and multi-temperature emission lineshapes $A(E - E_{\mathrm{ZPL}}, T)$.
