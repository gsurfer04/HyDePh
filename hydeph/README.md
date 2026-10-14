# `hydeph`: High-Definition Hybrid Defect Photoluminescence Toolkit

A modular Python framework for training ultra-fast, high-fidelity $\Delta$-Machine Learning Potentials ($\Delta$-NEP) on top of universal foundation models (such as `nep89`) for color centers in solids.

---

## 1. Overview

Universal machine learning potentials (e.g. `nep89`) provide broad lattice stability and elastic properties across the periodic table, but lack defect-specific many-body correlation corrections (such as deep defect state localization, Jahn-Teller distortions, and Franck-Condon relaxation forces). 

`hydeph` automates the construction of a targeted $\Delta$-correction potential:
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
hydeph/
├── __init__.py           # Package exports (v0.2.0)
├── config.py             # ToolkitConfig dataclass & cross-platform input validator
├── defect_builder.py     # Supercell expansion, defect embedding, GS/ES relaxation
├── frame_sampler.py      # 4-Phase representative frame sampler (FC, rattle, MD+FPS)
├── qe_runner.py          # Quantum ESPRESSO runner supporting Windows MS-MPI (QMatSuite) & Linux OpenMPI
├── delta_trainer.py      # Delta property extraction & GPU GPUMD NEP trainer (nep / nep.exe)
├── calculator.py         # HybridDefectCalculator (V_base + Delta_V)
├── lineshape.py          # Normal modes, Huang-Rhys S, S(hw), & multi-T lineshapes
├── pipeline.py           # Top-level turnkey pipeline orchestrator
└── run_licl_fcenter.py   # Turnkey benchmark driver for LiCl F-center
```

---

## 5. Quickstart Example

```python
from hydeph import ToolkitConfig, run_pipeline

# 1. Configure inputs (works on both Linux and Windows)
config = ToolkitConfig(
    pristine_cell_path="pristine_unitcell.xyz",
    defect_cell_path="defect_unitcell.xyz",
    supercell_repeat=(2, 2, 2),
    foundation_potential_path="nep89_20250409.txt",
    pseudopotentials={
        "Li": "li_pbe_v1.4.uspp.F.UPF",
        "Cl": "cl_pbe_v1.4.uspp.F.UPF"
    },
    pseudo_dir="./pseudo",
    temperatures=[10.0, 77.0, 150.0, 300.0],
    workspace_dir="./hydeph_run",
    
    # Optional Windows / custom binary overrides:
    # pw_exe=r"C:\QE\bin\pw.exe",       # Precompiled QE Windows (QMatSuite)
    # mpi_command="mpiexec",            # Microsoft MPI on Windows
    # nep_exe=r"C:\GPUMD\bin\nep.exe"   # GPUMD Windows binary
)

# 2. Run automated pipeline
summary = run_pipeline(config, n_procs_qe=4)
```

### 5.1 Charged Defects & Spin States (e.g., $\text{NV}^-$ in Diamond)

For charged defect systems such as the negatively charged nitrogen-vacancy center ($\text{NV}^-$) in diamond, specify `charge` (or `tot_charge`) and `spin_multiplicity`:

```python
from hydeph import ToolkitConfig, run_pipeline

config = ToolkitConfig(
    pristine_cell_path="diamond_unitcell.xyz",
    defect_cell_path="diamond_nv_unitcell.xyz",
    supercell_repeat=(3, 3, 3),
    foundation_potential_path="nep89_20250409.txt",
    pseudopotentials={
        "C": "C.upf",
        "N": "N.upf"
    },
    pseudo_dir="./pseudo",

    # --- System Charge & Spin State ---
    charge=-1,                 # Net charge in units of e (tot_charge = -1.0)
    spin_polarized=True,       # Spin-polarized calculation (nspin = 2)
    spin_multiplicity=3,       # 2S+1: Triplet ground state (3A2, S=1 -> tot_magnetization = 2)

    temperatures=[10.0, 77.0, 300.0],
    enable_anharmonic=True,
    workspace_dir="./nv_diamond_run"
)

summary = run_pipeline(config)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `charge` / `tot_charge` | `float` | `0.0` | Net supercell charge in units of $e$ (e.g. `-1` for $\text{NV}^-$ in diamond). Injected directly into Quantum ESPRESSO `&SYSTEM tot_charge`. |
| `spin_polarized` | `bool` | `True` | Enables spin-unrestricted calculations (`nspin=2` in QE). |
| `spin_multiplicity` | `int` | `2` | Spin multiplicity $2S+1$ ($1$ for singlet $S=0$, $2$ for doublet $S=1/2$, $3$ for triplet $S=1$ like $\text{NV}^-$). |
| `tot_magnetization` | `int` | `None` | Number of unpaired electrons $2S$ (e.g. `2` for $\text{NV}^-$). Synchronized with `spin_multiplicity`. |
| `excited_cell_path` | `str` | `None` | Path to explicit excited-state cell or supercell. If omitted, procedural defect coordination shell breathing is used. |
| `excited_state_displacement` | `float` | `None` | Radial displacement ($\text{Å}$) applied to coordinating atoms for procedural excited-state generation (defaults to $0.08\ \text{Å}$ for diamond NV, $0.15\ \text{Å}$ for ionic halides). |
| `phonon_hw_max` | `float` | `None` | Maximum frequency ($\text{meV}$) for electron-phonon spectral density $S(\hbar\omega)$. Defaults to adapting automatically to the highest normal mode (e.g. $175\ \text{meV}$ in diamond). |
| `smart_axis_scaling` | `bool` | `True` | Automatically determines tight, optimal axis limits based on active vibrational modes and emission envelopes. |
| `individual_shared_x` | `bool` | `False` | For individual plots, `False` frames each temperature to its active envelope; `True` uses a shared uniform energy range. |
| `lineshape_y_scale` | `str` | `"linear"` | Scale for lineshape intensity axis (`"linear"` or `"log"`). |
| `phonon_hw_min` / `phonon_hw_max` | `float` | `None` | Manual overrides for phonon frequency axis bounds ($\text{meV}$). |
| `lineshape_e_min` / `lineshape_e_max` | `float` | `None` | Manual overrides for emission energy range relative to ZPL ($\text{meV}$). |
| `lineshape_y_min` / `lineshape_y_max` | `float` | `None` | Manual overrides for emission intensity axis bounds. |
| `phonon_smearing_meV` | `float` | `6.0` | Gaussian smearing ($\text{meV}$) for $S(\hbar\omega)$ to bridge supercell discrete mode gaps (finite-size correction). If `None`, auto-detects from median mode gap. |
| `dephasing_gamma_meV` | `float` | `None` | ZPL dephasing damping ($\text{meV}$). Defaults to $\gamma(T) = 1.5 + 2.0\times 10^{-4} T^2\ \text{meV}$. |
| `finite_size_acoustic_cutoff` | `bool` | `True` | Enforces acoustic phonon scaling $S(\hbar\omega) \to 0$ as $\omega \to 0$ to prevent finite-size boundary artifacts. |
| `random_seed` | `int` | `42` | Deterministic RNG seed for rattling displacements and MD velocities across runs. |
| `force_resample` | `bool` | `False` | Reuses existing `candidate_frames.xyz` in workspace when `False`; regenerates fresh pool when `True`. |
| `max_dft_frames` | `int` | `None` | Target total number of converged DFT frames. Resumes pending frames prioritizing missing sample types without re-running completed frames. |
| `max_new_dft_frames` | `int` | `None` | Optional limit on new DFT calculations to execute per run. |
| `dft_resume_mode` | `str` | `"target_total"` | Resumption budget mode (`"target_total"` or `"max_new"`). |



---

## 6. Windows Support & MPI (QMatSuite)

For Windows users, precompiled native Quantum ESPRESSO builds are distributed by [QMatSuite/quantum-espresso-windows-exe](https://github.com/QMatSuite/quantum-espresso-windows-exe).
- **No WSL/MinGW required**: Colocated binaries under `bin\` with Intel MKL, OpenMP, and Microsoft MPI.
- **MPI Differences handled automatically by HyDePh**:
  - Automatically switches between `mpiexec` (Microsoft MPI syntax: `mpiexec -n 4 pw.exe -in pw.in`) on Windows and `mpirun` (`mpirun -np 4 pw.x -in pw.in`) on Linux.
  - Automatically sets subprocess environment `PATH` to resolve colocated Intel MKL & MS-MPI DLLs.
  - Sanitizes Windows file paths in Quantum ESPRESSO input cards (`&CONTROL pseudo_dir`).

---

## 7. Physical Outputs

Generated inside `workspace_dir` (`./hydeph_run` by default):

- `relaxed_gs.xyz` & `relaxed_es.xyz`: Ground and excited equilibrium geometries.
- `candidate_frames.xyz`: 4-phase representative frame ensemble.
- `labeled_dft_frames.xyz`: First-principles DFT energies and atomic forces.
- `nep_delta.txt`: Trained defect $\Delta$-NEP potential.
- `lineshapes.png`: Multi-temperature optical emission lineshapes and electron-phonon spectral density.
- `anharmonic_lineshapes.png`: Finite-temperature anharmonic MD lineshapes and defect-core VDOS.
