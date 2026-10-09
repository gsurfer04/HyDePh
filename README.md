# HyDePh: High-Definition Hybrid Defect Photoluminescence

<p align="center">
  <img src="assets/licl_fcenter_anharmonic_lineshapes.png" width="92%" alt="HyDePh Banner">
</p>

<p align="center">
  <b>Automated First-Principles $\Delta$-Machine Learning Potentials for High-Definition Color-Center Spectroscopy</b>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue" alt="Python Version"></a>
  <a href="https://github.com/brucefan992/GPUMD"><img src="https://img.shields.io/badge/Backend-GPUMD%20%2F%20NEP-orange" alt="GPUMD"></a>
  <a href="https://www.quantum-espresso.org/"><img src="https://img.shields.io/badge/DFT-Quantum%20ESPRESSO-red" alt="QE"></a>
  <a href="https://wiki.fysik.dtu.dk/ase/"><img src="https://img.shields.io/badge/Ecosystem-ASE-green" alt="ASE"></a>
</p>

---

## 🔬 Overview

Universal machine learning potentials (e.g., `nep89`) provide broad lattice stability across the periodic table, but lack defect-specific many-body correlation corrections (such as deep electronic state localization, Jahn-Teller distortions, and optical Franck-Condon relaxation forces).

**HyDePh** (*"Hi-Def"*: **Hy**brid **De**fect **Ph**otoluminescence) bridges this gap by automatically training a localized $\Delta$-Neuroevolution Potential ($\Delta$-NEP) on top of universal foundation models:

$$V_{\mathrm{hybrid}}(\mathbf{R}) = V_{\mathrm{foundation}}(\mathbf{R}) + \Delta V_{\mathrm{NEP}}(\mathbf{R})$$
$$\mathbf{F}_{\mathrm{hybrid}}(\mathbf{R}) = \mathbf{F}_{\mathrm{foundation}}(\mathbf{R}) + \Delta \mathbf{F}_{\mathrm{NEP}}(\mathbf{R})$$

From a minimal set of user inputs, **HyDePh** executes:
1. **Automated Defect Embedding**: Tiling pristine bulk unit cells around a pre-made defect unit cell with automatic defect-site coordinate detection.
2. **4-Phase Representative Sampling**: Generating diverse candidate configurations spanning the Franck-Condon coordinate, defect-core rattling, and high-temperature foundation MD with Farthest Point Sampling (FPS).
3. **High-Throughput DFT Labeling**: Automating Quantum ESPRESSO (`pw.x`) calculations with MPI + GPU acceleration, live SCF iteration streaming, and disk-level caching.
4. **GPU-Accelerated $\Delta$-NEP Training**: Fitting the compact delta potential in GPUMD in seconds.
5. **High-Definition Lineshapes**: Computing Huang-Rhys factors $S$, electron-phonon spectral densities $S(\hbar\omega)$, and multi-temperature optical emission spectra $A(E - E_{\mathrm{ZPL}}, T)$ using both **quantum generating functions** and **finite-temperature anharmonic molecular dynamics free energy sampling**.

---

## ⚡ Quickstart

### 1. Installation

```bash
git clone https://github.com/your-username/hydeph.git
cd hydeph
pip install -e .
```

Prerequisites:
- [GPUMD / NEP](https://github.com/brucefan992/GPUMD) (`gpumd` and `nep` binaries on `$PATH`)
- [Quantum ESPRESSO](https://www.quantum-espresso.org/) (`pw.x` on `$PATH`)
- NVIDIA GPU with CUDA support

### 2. Python API Example

```python
from hydeph import ToolkitConfig, run_pipeline

# Configure with the 5 core inputs
config = ToolkitConfig(
    # 1. Pristine bulk unit cell
    pristine_cell_path="licl_pristine_unitcell.xyz",
    
    # 2. Pre-made defect unit cell (surrounded by pristine bulk in supercell)
    defect_cell_path="licl_defect_unitcell.xyz",
    supercell_repeat=(2, 2, 2),
    
    # 3. Foundation universal potential
    foundation_potential_path="/path/to/nep89_20250409.txt",
    
    # 4. Pseudopotentials for each element
    pseudopotentials={
        "Li": "li_pbe_v1.4.uspp.F.UPF",
        "Cl": "cl_pbe_v1.4.uspp.F.UPF",
    },
    pseudo_dir="/path/to/quantumespresso/pseudo",
    
    # 5. Temperatures of interest (K)
    temperatures=[10.0, 77.0, 150.0, 300.0],
    
    # Enable finite-temperature anharmonic MD sampling
    enable_anharmonic=True,
    anharmonic_md_steps=2000
)

# Run end-to-end turnkey pipeline
summary = run_pipeline(config)
```

---

## 🏗️ Architecture & Workflow

```mermaid
flowchart TD
    Pristine["1. Pristine Bulk Unit Cell"] --> DB["DefectBuilder"]
    Defect["2. Pre-made Defect Unit Cell"] --> DB
    BasePot["3. Foundation NEP (nep89)"] --> DB
    BasePot --> FS["FrameSampler (4 Phases)"]
    DB --> |"Embedded Supercell & Relaxed GS/ES"| FS
    FS --> |"Candidate Ensemble"| QE["QERunner (pw.x + GPU)"]
    Pseudos["4. Element UPF Pseudopotentials"] --> QE
    QE --> |"Total DFT Energies & Forces"| DT["DeltaTrainer (GPUMD)"]
    BasePot --> DT
    DT --> |"nep_delta.txt"| Calc["HybridDefectCalculator (V_base + Delta_V)"]
    BasePot --> Calc
    Calc --> LS["LineshapeEngine"]
    Temps["5. Temperatures (10 - 300 K)"] --> LS
    LS --> Out1["Harmonic Generating Function A_harm(E, T)"]
    LS --> Out2["Anharmonic Langevin MD A_anh(E, T)"]
```

---

## 📊 Benchmark: F-Center in Lithium Chloride (LiCl)

HyDePh was benchmarked against the seminal experimental photoluminescence study by **Ken Takiyama** (*Journal of the Physical Society of Japan*, **44**, 1627–1634, 1978).

<p align="center">
  <img src="assets/licl_fcenter_experimental_comparison.png" width="95%" alt="LiCl F-center Experimental Benchmark">
</p>

### Spectroscopic Highlights:
- **Coupling Phonon Frequency**: Takiyama's measured coupling mode ($\hbar\omega_e = 19.7\ \text{meV}$) lies directly within our calculated acoustic/TO electron-phonon coupling band ($16\text{--}28\ \text{meV}$).
- **Anharmonic Broadening**: The harmonic approximation predicted an unphysically narrow emission bandwidth at $77\ \text{K}$ ($\mathrm{FWHM} \approx 78\ \text{meV}$). Finite-temperature anharmonic MD sampling on the hybrid potential expands the bandwidth to **$265.2\ \text{meV}$ ($0.27\ \text{eV}$)**—a **$3.4\times$ improvement** bridging the majority of the gap toward the experimental value ($460\ \text{meV}$).
- **Phonon Softening**: The defect-core velocity autocorrelation function (VACF) accurately captures thermal phonon softening from $27.8\ \text{meV}$ at $0\ \text{K}$ down to $19\text{--}20\ \text{meV}$ at $77\ \text{K}$ and $300\ \text{K}$.

---

## 📁 Repository Structure

```
hydeph/
├── __init__.py           # Package API and exports
├── config.py             # ToolkitConfig dataclass and validation
├── defect_builder.py     # Supercell embedding & defect coordinate locator
├── frame_sampler.py      # 4-Phase representative frame sampler (FC, rattle, MD+FPS)
├── qe_runner.py          # Quantum ESPRESSO pw.x runner with streaming & cache
├── delta_trainer.py      # Foundation subtraction & GPU GPUMD Delta-NEP trainer
├── calculator.py         # 31x optimized HybridDefectCalculator (V_base + Delta_V)
├── lineshape.py          # Harmonic generating function & anharmonic MD engine
├── pipeline.py           # Turnkey pipeline orchestrator
└── run_licl_fcenter.py   # Complete runnable benchmark driver for LiCl F-center
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
