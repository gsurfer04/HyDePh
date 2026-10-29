# HyDePh: High-Definition Hybrid Defect Photoluminescence

<p align="center">
  <img src="assets/hydeph_stiff_vs_soft_benchmark.png" width="96%" alt="HyDePh Stiff vs Soft Benchmark Banner">
</p>

<p align="center">
  <b>First-Principles &Delta;-Machine Learning Potentials for Color-Center Spectroscopy</b>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue" alt="Python Version"></a>
  <a href="https://github.com/brucefan992/GPUMD"><img src="https://img.shields.io/badge/Backend-GPUMD%20%2F%20NEP-orange" alt="GPUMD"></a>
  <a href="https://www.quantum-espresso.org/"><img src="https://img.shields.io/badge/DFT-Quantum%20ESPRESSO%20(GPU%2FCPU)-red" alt="QE"></a>
  <a href="https://learn.microsoft.com/en-us/windows/wsl/"><img src="https://img.shields.io/badge/Windows-WSL2%20(CUDA%20Accelerated)-7057ff" alt="WSL2"></a>
  <a href="https://github.com/QMatSuite/quantum-espresso-windows-exe"><img src="https://img.shields.io/badge/Windows-Native%20QE%20(QMatSuite)-2F81F7" alt="Windows QE"></a>
  <a href="https://wiki.fysik.dtu.dk/ase/"><img src="https://img.shields.io/badge/Ecosystem-ASE-green" alt="ASE"></a>
</p>

---

## Overview

Universal machine learning potentials (`nep89`) capture bulk lattice elasticity across the periodic table, but lack defect-specific electronic excitation states, Jahn-Teller distortions, and Franck-Condon relaxation forces.

**HyDePh** (**Hy**brid **De**fect **Ph**otoluminescence) trains paired localized $\Delta$-Neuroevolution Potentials ($\Delta V_{\mathrm{GS}}$ and $\Delta V_{\mathrm{ES}}$) on top of a universal foundation model:

$$
\begin{aligned}
V_{\mathrm{hybrid,GS}}(\mathbf{R}) &= V_{\mathrm{base}}(\mathbf{R}) + N_{\mathrm{atoms}}\bar{\Delta e}_{\mathrm{GS}} + \Delta V_{\mathrm{NEP,GS}}(\mathbf{R}) \\
V_{\mathrm{hybrid,ES}}(\mathbf{R}) &= V_{\mathrm{base}}(\mathbf{R}) + N_{\mathrm{atoms}}\bar{\Delta e}_{\mathrm{ES}} + \Delta V_{\mathrm{NEP,ES}}(\mathbf{R}) \\
\mathbf{F}_{\mathrm{hybrid}, S}(\mathbf{R}) &= \mathbf{F}_{\mathrm{base}}(\mathbf{R}) + \Delta \mathbf{F}_{\mathrm{NEP}, S}(\mathbf{R})
\end{aligned}
$$

Key capabilities:
1. **Defect Embedding**: Vectorized tiling of pristine bulk unit cells around a defect unit cell and automated defect coordinate detection via minimum-image distance matching.
2. **4-Phase Active Sampling**: Generates candidate configurations across the Franck-Condon coordinate, defect-core rattling, and high-temperature foundation MD with incremental Farthest Point Sampling (FPS).
3. **DFT Labeling & Resumption**: Drives Quantum ESPRESSO (`pw.x` / `pw.exe`) with $\Gamma$-point real wavefunctions (`K_POINTS gamma`), fixed occupations (`occupations = 'fixed'`), atom-scaled convergence thresholds, and fail-closed disk caching.
4. **Constrained $\Delta\text{SCF}$ Excited States**: Automatically detects in-gap defect frontier orbitals from ground-state Kohn-Sham eigenvalues, symmetrizes degenerate LUMO occupations, and extracts exact adiabatic Zero-Phonon Line ($E_{\mathrm{ZPL}}$), absorption, emission, and Stokes shift energies.
5. **Paired GPU $\Delta$-NEP Training**: Fits independent ground-state and optical excited-state residuals in GPUMD with mean baseline offset subtraction ($\bar{\Delta e}_S$), automated quality control gates, and optional committee ensembles.
6. **Large-Supercell Phonon Embedding**: Embeds localized defect vibrations and forces into large bulk phonon supercells ($L \ge 10\text{--}12\ \text{Å}$) via force or displacement projection, eliminating discrete mode-comb artifacts.
7. **Paired Cumulant Anharmonic Dynamics**: Computes multi-temperature emission lineshapes using NVT thermalization followed by unperturbed NVE microcanonical molecular dynamics with the physical energy-gap cumulant generating function and Linderälv Gaussian time-domain windowing.
8. **1D Configuration Coordinate & Stokes Shift**: Evaluates paired potential surfaces along $\mathbf{R}(\lambda) = \mathbf{R}_{\mathrm{GS}} + \lambda (\mathbf{R}_{\mathrm{ES}} - \mathbf{R}_{\mathrm{GS}})$ to extract $\Delta Q$, asymmetric relaxation energies $\Delta E_{\mathrm{rel,GS}} \ne \Delta E_{\mathrm{rel,ES}}$, effective phonon energies $\hbar\Omega_{S}$, and Stokes shift $\Delta E_{\mathrm{Stokes}}$.
9. **Provenance & Numerical Data Export**: Exports publication-ready figures, CSVs, `normal_modes.csv`, and `provenance.json` recording software versions, input SHA256 hashes, and execution metrics.

See the **[Theoretical Walkthrough](docs/THEORY_WALKTHROUGH.md)** for detailed mathematical derivations, **[CHANGELOG.md](CHANGELOG.md)** for release history, and **[CONTRIBUTING.md](CONTRIBUTING.md)** for developer setup.

---

## Quickstart

### 1. Installation

```bash
git clone https://github.com/gsurfer04/HyDePh.git
cd HyDePh
pip install -e .
```

For development dependencies (testing, linting, formatting):
```bash
pip install -e ".[dev]"
```

Prerequisites:
- [GPUMD / NEP](https://github.com/brucefan992/GPUMD) (`gpumd` and `nep` / `nep.exe` with CUDA support)
- [Quantum ESPRESSO](https://www.quantum-espresso.org/):
  - **Linux / Windows WSL2 (Recommended)**: GPU-accelerated `pw.x` (CUDA/OpenACC). Running inside WSL2 delivers an order-of-magnitude speedup for supercell DFT labeling over CPU execution.
  - **Native Windows (CPU Fallback)**: `pw.exe` from [QMatSuite/quantum-espresso-windows-exe](https://github.com/QMatSuite/quantum-espresso-windows-exe) with Microsoft MPI.
- NVIDIA GPU with CUDA support (or CPU fallback for potential evaluation)

### 2. Command-Line Interface (CLI)

Run end-to-end calculations directly from a YAML or TOML configuration file:

```bash
hydeph config.yaml
```

Optional flags:
```bash
hydeph config.yaml --from-stage 3 --max-dft 30 --workers 1
```

Validate configuration and dry-run defect setup without running DFT:
```bash
hydeph validate config.yaml
```

### 3. Python API

```python
from hydeph import ToolkitConfig, run_pipeline

config = ToolkitConfig(
    # Core Inputs & Paths
    pristine_cell_path="licl_pristine_unitcell.xyz",
    defect_cell_path="licl_defect_unitcell.xyz",
    foundation_potential_path="/path/to/nep89_20250409.txt",
    pseudopotentials={
        "Li": "li_pbe_v1.4.uspp.F.UPF",
        "Cl": "cl_pbe_v1.4.uspp.F.UPF",
    },
    pseudo_dir="/path/to/quantumespresso/pseudo",
    workspace_dir="./hydeph_run",
    allow_fallbacks=False,

    # Electronic Structure & Geometry
    supercell_repeat=(2, 2, 2),
    charge=0.0,
    spin_polarized=True,
    spin_multiplicity=2,  # S = 1/2 (doublet)
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

### 4. Charged Defects & Spin Polarization (e.g., $\text{NV}^-$ in Diamond)

```python
from hydeph import ToolkitConfig, run_pipeline

config = ToolkitConfig(
    # Core Inputs & Paths
    pristine_cell_path="diamond_unitcell.xyz",
    defect_cell_path="diamond_nv_unitcell.xyz",
    foundation_potential_path="nep89_20250409.txt",
    pseudopotentials={
        "C": "C_ONCV_PBE-1.2.upf",
        "N": "N_ONCV_PBE-1.2.upf",
    },
    pseudo_dir="./pseudo",
    workspace_dir="./nv_diamond_run",
    allow_fallbacks=False,

    # Electronic Structure & Geometry
    supercell_repeat=(3, 3, 3),
    charge=-1.0,
    spin_polarized=True,
    spin_multiplicity=3,  # Triplet (3A2, S=1 -> tot_magnetization=2)
    excited_state_method="delta_scf",
    paired_delta_nep=True,
    ecutwfc=65.0,
    ecutrho=260.0,

    # Sampling & Resumption
    max_dft_frames=40,
    n_procs_qe=1,
    n_workers_qe=1,

    # Delta-NEP Architecture Hyperparameters
    nep_steps=25000,

    # Spectroscopy, Graphing & 1D Configuration Coordinate
    temperatures=[10.0, 77.0, 150.0, 300.0],
    enable_anharmonic=True,
    anharmonic_acf_damping_meV=2.0,
    dephasing_temperature_coefficient=5.5e-5,
)

summary = run_pipeline(config)
```

---

## Configuration Reference

### Core Inputs & Paths

| Parameter | Type | Default | Description |
|---|---|---|---|
| `pristine_cell_path` | `str` | *(required)* | Path to bulk pristine unit cell (`.xyz`, `.cif`, or `POSCAR`). |
| `defect_cell_path` | `str` | `None` | Path to pre-constructed defect unit cell or supercell. |
| `foundation_potential_path` | `str` | `"nep89_20250409.txt"` | Path to GPUMD universal foundation potential. |
| `pseudopotentials` | `dict` | `None` | Element-to-pseudopotential filename map (auto-discovered if omitted). |
| `pseudo_dir` | `str` | `None` | Directory containing Quantum ESPRESSO UPF pseudopotential files. |
| `workspace_dir` | `str` | `"./hydeph_run"` | Workspace directory for outputs, caches, and scratch files. |
| `allow_fallbacks` | `bool` | `False` | Fail-closed guardrail: raises `RuntimeError` immediately upon DFT failure rather than substituting unrelaxed geometries. |

### Electronic Structure & Geometry

| Parameter | Type | Default | Description |
|---|---|---|---|
| `supercell_repeat` | `tuple of int` | `(2, 2, 2)` | Supercell expansion multipliers `(nx, ny, nz)`. Physical length $L \ge 10\text{--}12\ \text{Å}$ is recommended. |
| `charge` / `tot_charge` | `float` | `0.0` | Net supercell charge in units of $e$ (`&SYSTEM tot_charge` in QE). |
| `spin_polarized` | `bool` | `True` | Spin-unrestricted calculation (`nspin=2` in QE). |
| `spin_multiplicity` | `int` | `2` | Spin multiplicity $2S+1$ ($1$: singlet, $2$: doublet, $3$: triplet). |
| `tot_magnetization` | `int` | `None` | Unpaired electrons $2S$; synchronized with `spin_multiplicity`. |
| `excited_state_method` | `str` | `"delta_scf"` | Method for excited state: `"delta_scf"` (constrained occupations), `"spin_proxy"`, or `"synthetic"`. |
| `paired_delta_nep` | `bool` | `True` | Trains independent $\Delta V_{\mathrm{GS}}$ and $\Delta V_{\mathrm{ES}}$ models for separate ground and excited surfaces. |
| `dft_es_candidates` | `str` | `"all"` | Candidate subset for excited-state DFT: `"all"` or `"fc_path"`. |
| `delta_scf_homo_band` | `int` | `None` | Explicit HOMO band index for $\Delta\text{SCF}$; auto-detected from ground-state eigenvalues when `None`. |
| `delta_scf_lumo_band` | `int` | `None` | Explicit LUMO band index for $\Delta\text{SCF}$; auto-detected from ground-state eigenvalues when `None`. |
| `delta_scf_lumo_degeneracy` | `int` | `None` | Degeneracy of LUMO state ($1$ for singlet, $2$ for doublet, etc.) for symmetric orbital filling. |
| `delta_scf_spin_channel` | `int` | `None` | Spin channel for excitation ($1$: up, $2$: down); auto-detected when `None`. |
| `ecutwfc` | `float` | `None` | Kinetic energy cutoff for wavefunctions in Ry. |
| `ecutrho` | `float` | `None` | Kinetic energy cutoff for charge density in Ry (defaults to $4 \times \text{ecutwfc}$ for norm-conserving UPFs). |
| `conv_thr` | `str` | `None` | SCF convergence threshold in QE (defaults to $10^{-10}\ \text{Ry/atom}$). |
| `mixing_beta` | `float` | `0.4` | SCF charge density mixing factor. |
| `diagonalization` | `str` | `"david"` | Diagonalization method in QE (`'david'`, `'cg'`). |

### Sampling & Resumption

| Parameter | Type | Default | Description |
|---|---|---|---|
| `random_seed` | `int` | `42` | RNG seed for rattling displacements and MD velocities. |
| `force_resample` | `bool` | `False` | Reuses existing `relaxed_gs.xyz`, `relaxed_es.xyz`, and `candidate_frames.xyz` when `False`; regenerates when `True`. |
| `force_retrain` | `bool` | `False` | Reuses `nep_delta_gs.txt` / `nep_delta_es.txt` when cache manifest matches; retrains when `True`. |
| `max_dft_frames` | `int` | `None` | Target total converged DFT frames. Reuses converged frames and executes remaining quota. |
| `max_new_dft_frames` | `int` | `None` | Maximum new DFT calculations to execute in the current run. |
| `dft_resume_mode` | `str` | `"target_total"` | `"target_total"` (reach `max_dft_frames` total) or `"max_new"` (run up to `max_new_dft_frames` new frames). |
| `retry_failed_dft` | `bool` | `True` | Retries failed or non-converged frames with conservative mixing (`mixing_beta=0.20`, `electron_maxstep=150`). |
| `replace_failed_frames` | `bool` | `False` | Replaces a permanently failed rattle or MD frame with a new candidate to fulfill quota. |
| `n_procs_qe` | `int` | `1` | Number of MPI processes per Quantum ESPRESSO worker. |
| `n_workers_qe` | `int` | `1` | Number of independent `pw.x` calculations to execute concurrently. |
| `qe_timeout_s` | `float` | `None` | Wall-clock timeout (seconds) per `pw.x` run before terminating and queuing for retry. |

### Delta-NEP Architecture & Quality Gates

| Parameter | Type | Default | Description |
|---|---|---|---|
| `nep_neurons` | `int` | `30` | Neurons in Delta-NEP hidden layer. |
| `nep_batch` | `int` | `2000` | Batch size for Delta-NEP GPUMD training. |
| `nep_steps` | `int` | `30000` | Training generations for Delta-NEP. |
| `nep_cutoff_radial` | `float` | `4.5` | Radial cutoff in Å. |
| `nep_cutoff_angular` | `float` | `3.5` | Angular cutoff in Å. |
| `subtract_mean_energy_offset` | `bool` | `True` | Subtracts baseline energy offset $\bar{\Delta e}$ prior to training and restores during potential evaluation. |
| `max_test_force_rmse_eV_A` | `float` | `0.15` | Maximum allowable test force RMSE gate before raising a validation error. |
| `nep_committee_size` | `int` | `1` | Size of committee ensemble for epistemic uncertainty quantification. |

### Spectroscopy & Optical Lineshapes

| Parameter | Type | Default | Description |
|---|---|---|---|
| `temperatures` | `list of float` | `[10.0, 77.0, 300.0]` | Emission temperatures in Kelvin. |
| `enable_anharmonic` | `bool` | `True` | Evaluates multi-temperature anharmonic lineshapes via NVE cumulant dynamics. |
| `anharmonic_equil_steps` | `int` | `500` | Langevin equilibration steps prior to NVE production. |
| `anharmonic_md_steps` | `int` | `1500` | NVE production steps per trajectory for energy-gap autocorrelation sampling. |
| `anharmonic_n_trajectories` | `int` | `1` | Number of independent NVE trajectories for ensemble averaging. |
| `anharmonic_acf_damping_meV` | `float` | `2.0` | Gaussian time-domain window width $\gamma_{\mathrm{acf}}$ for smoothing autocorrelation truncation noise. |
| `normalize_anharmonic_s` | `bool` | `True` | Normalizes $\int S_{\mathrm{anh}}(\hbar\omega)\,d(\hbar\omega) = S$ to preserve Franck-Condon sum rule while capturing anharmonic mode shifts. |
| `dephasing_gamma_meV` | `float` | `1.5` | Base ZPL dephasing width $\gamma_0$ in meV. |
| `dephasing_temperature_coefficient` | `float` | `5.5e-5` | Quadratic acoustic Raman dephasing coefficient $\alpha_{\mathrm{deph}}$ ($\text{meV/K}^2$). |
| `phonon_supercell_repeat` | `tuple of int` | `None` | Supercell repeat for expanded phonon embedding calculations. |
| `phonon_embedding_method` | `str` | `"displacement"` | Embedding mode projection: `"displacement"` or `"force"`. |
| `plot_configuration_coordinate` | `bool` | `True` | Generates 2-panel 1D configuration coordinate and Stokes shift figure. |

---

## Supercell Size Guidelines

Under periodic boundary conditions, the physical supercell edge length **$L \ge 10\text{--}12\ \text{Å}$**, rather than an arbitrary integer repeat tuple, dictates convergence:

1. **Brillouin-Zone Phonon Sampling**: The grid spacing in reciprocal space is $\Delta q = \frac{2\pi}{L}$. In small cells ($L < 10\ \text{Å}$), acoustic branches cannot disperse, and localized defect modes cannot hybridize with the host continuum. Electron-phonon coupling is artificially compressed into isolated modes, producing spurious multi-phonon spike combs.
2. **Elastic Image Strain**: Defect-defect image interactions decay elastically as $L^{-3}$ and electrostatically as $L^{-1}$. Supercells with $L \ge 10\ \text{Å}$ isolate the defect cavity.
3. **Lattice Constant Differences**:
   - In **compact covalent hosts** (diamond, $a = 3.57\ \text{Å}$), a $2\times 2\times 2$ cell has $L = 7.15\ \text{Å}$ (insufficient), requiring `(3, 3, 3)` ($L = 10.72\ \text{Å}$, 215 atoms).
   - In **large-lattice ionic hosts** (rocksalt NaCl, $a = 5.69\ \text{Å}$), a **$2\times 2\times 2$ cell is already $L = 11.38\ \text{Å}$** (63 atoms), fully converging the vibrational continuum with $< 1\ \text{GB}$ VRAM.

| Host Material Platform | Crystal Type | Recommended Supercell | Atom Count | Minimum Dimension $L$ | Guidelines & Expected Features |
|---|---|---|---|---|---|
| **Diamond** ($\text{NV}^-$, $\text{SiV}^-$, $\text{GeV}^-$) | Covalent ($\Theta_D \approx 2200\ \mathrm{K}$) | `(3, 3, 3)` | $215$ atoms | $\ge 10.7\ \text{Å}$ | Use $\ge 215$ atoms. $2\times 2\times 2$ (63 atoms) produces artificial $73\ \text{meV}$ multi-phonon spike combs. |
| **4H-SiC** ($kk$-$\mathrm{VV}^0$, $V_{\mathrm{Si}}$) | Polar covalent ($\Theta_D \approx 1200\ \mathrm{K}$) | `(3, 3, 2)` or `(4, 4, 2)` | $142\text{--}254$ atoms | $\ge 9.2\text{--}12.3\ \text{Å}$ | Separates defect images across basal and $c$-axis hexagonal layers. |
| **3C-SiC** | Polar covalent ($\Theta_D \approx 1200\ \mathrm{K}$) | `(3, 3, 3)` | $215$ atoms | $\ge 13.1\ \text{Å}$ | Captures smooth quasi-local breathing mode dispersion. |
| **Alkali Halides** (NaCl, LiCl) | Rocksalt ($\Theta_D \approx 320\text{--}420\ \mathrm{K}$) | `(2, 2, 2)` | $63$ atoms | $\ge 11.4\ \text{Å}$ | $a \approx 5.69\ \text{Å}$ means `(2, 2, 2)` already exceeds $L \ge 11\ \text{Å}$ with $< 1\ \text{GB}$ VRAM usage. |
| **Fluorites** (LiF, $\mathrm{CaF}_2$) | Fluorite ($\Theta_D \approx 730\ \mathrm{K}$) | `(2, 2, 2)` | $95\text{--}189$ atoms | $\ge 8.0\text{--}10.9\ \text{Å}$ | Strong coupling ($S > 10$); Marcus Gaussian envelope dominates. |

---

## Benchmark Systems

| System | Host ($\Theta_D$) | Supercell ($N_{\mathrm{atoms}}$) | $S$ | $\Delta Q$ ($\mathrm{amu}^{1/2}\text{Å}$) | Adiabatic $E_{\mathrm{ZPL}}$ | Key Spectroscopic Result |
|---|---|---|---|---|---|---|
| **Diamond $\mathrm{NV}^-$** | Covalent ($2200\ \mathrm{K}$) | $3\times 3\times 3$ ($215$) | $2.91$ | $0.562$ | $1.716\ \mathrm{eV}$ | Sharp ZPL ($\mathrm{DWF}_0 = 5.46\%$) visible up to $300\ \mathrm{K}$; $64.6\ \mathrm{meV}$ $a_1$ resonance & smooth sideband wing |
| **4H-SiC $kk$-$\mathrm{VV}^0$** | Polar covalent ($1200\ \mathrm{K}$) | $3\times 3\times 2$ ($142$) | $4.77$ | $0.812$ | $1.10\ \mathrm{eV}$ | Sharp ZPL ($\mathrm{DWF}_0 = 0.85\%$), resolved sidebands at $-38, -76\ \mathrm{meV}$ across basal/$c$-axis modes |
| **LiCl F-Center** | Rocksalt ($420\ \mathrm{K}$) | $2\times 2\times 2$ ($63$) | $4.00$ | $1.632$ | $2.15\ \mathrm{eV}$ | Mode softens $27.8 \to 19.8\ \mathrm{meV}$ ($19.7\ \mathrm{meV}$ exp. [13]); anharmonic broadening to $265\ \mathrm{meV}$ |
| **NaCl F-Center** | Rocksalt ($320\ \mathrm{K}$) | $2\times 2\times 2$ ($63$) | $11.58$ | $2.220$ | $1.274\ \mathrm{eV}$ | Strong coupling ($\Delta E_{\mathrm{Stokes}} = 0.671\ \mathrm{eV}$); smooth FWHM broadening $163.8 \to 263.1\ \mathrm{meV}$ ($10\text{--}300\ \mathrm{K}$) |
| **LiF F-Center** | Rocksalt ($730\ \mathrm{K}$) | $2\times 2\times 2$ ($63$) | $23.93$ | $2.391$ | $3.42\ \mathrm{eV}$ | Deep Marcus regime ($\mathrm{DWF} \approx 0$); broad Gaussian band ($\mathrm{FWHM} = 490\text{--}1101\ \mathrm{meV}$) |

---

## Execution Platforms (WSL2 & Windows)

### 1. Linux & Windows WSL2 (CUDA Accelerated)
For workstations with an NVIDIA GPU, running in WSL2 (Ubuntu) provides end-to-end GPU acceleration across both the Quantum ESPRESSO DFT labeling stage (`pw.x`) and the GPUMD $\Delta$-NEP training stage (`gpumd` / `nep`).

```bash
# In WSL2 terminal
export OMP_NUM_THREADS=16
python HyDePh.py
```

### 2. Native Windows (Intel oneAPI / MS-MPI)
For environments without GPU support, HyDePh automatically detects Windows paths, Microsoft MPI (`mpiexec.exe`), and Colocated DLLs when using [QMatSuite Quantum ESPRESSO binaries](https://github.com/QMatSuite/quantum-espresso-windows-exe).

---

## Repository Structure

```
hydeph/
├── __init__.py                    # Package exports and version metadata
├── __main__.py                    # CLI entrypoint (python -m hydeph)
├── config.py                      # ToolkitConfig dataclass, file I/O & validation
├── defect_builder.py              # Supercell embedding & defect coordinate locator
├── frame_sampler.py               # 4-phase frame sampler (FC, rattle, streamed MD+FPS)
├── qe_runner.py                   # Quantum ESPRESSO runner (Gamma-point, Delta-SCF)
├── delta_trainer.py               # Baseline subtraction, GPUMD training & QC gates
├── calculator.py                  # HybridDefectCalculator & PairedDefectCalculator
├── lineshape.py                   # Quantum generating functions & NVE cumulant dynamics
├── cc_diagram.py                  # Paired 1D configuration coordinate & Stokes shift engine
├── provenance.py                  # Environmental provenance & SHA256 manifest tracking
├── validator.py                   # Dry-run structure, valence, and spin validator
└── pipeline.py                    # Pipeline orchestrator with stage resumption
benchmarks/
├── diamond-nvminus/               # Diamond NV- benchmark data & figures
├── 4H-SiC/                        # 4H-SiC kk divacancy benchmark data & figures
└── NaCl/                          # NaCl F-center benchmark data & figures
docs/
└── THEORY_WALKTHROUGH.md          # Theoretical formulation & literature citations
tests/                             # Automated test suite (128 unit & regression tests)
CHANGELOG.md                       # Release notes and history
CONTRIBUTING.md                    # Developer and contribution guidelines
pyproject.toml                     # Packaging metadata, CLI scripts & dependencies
```

## License

[MIT License](LICENSE)
