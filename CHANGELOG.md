# Changelog

All notable changes to HyDePh will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Foundation Bulk Dynamic Stability Diagnostics**:
  - Implemented `check_foundation_bulk_stability(pristine_atoms, calc_foundation)` executing early finite-difference phonon checks on the pristine unit cell during Step 2.
  - Detects bulk host dynamic instabilities, negative curvatures, and soft phonon branches caused by cell volume mismatch (e.g., tensile strain beyond universal potential cutoffs) before launching expensive DFT calculations.
- **Acoustic Sum Rule (ASR) & Rigid Translation Projection**:
  - Added `acoustic_sum_rule: bool = True` (default `True`) and `displacement_delta: float = 0.01` in `ToolkitConfig`.
  - Implemented `apply_acoustic_sum_rule_and_projection()` in `hydeph.lineshape`, enforcing tensorial ASR ($\sum_j \Phi_{ij} = 0$) and projecting out center-of-mass rigid translations from the mass-weighted dynamical matrix to eliminate spurious acoustic numerical drift.
- **Whole-Supercell & Hybrid Rattling Strategies**:
  - Added `rattle_mode: str = "hybrid"` (options: `"hybrid"`, `"core"`, `"whole"`, default `"hybrid"`) and `rattle_cutoff: float = 6.0` in `ToolkitConfig` and `FrameSampler`.
  - In `"hybrid"` mode, 30% of rattle frames displace all atoms in the supercell uniformly, providing whole-supercell restoring force signals so Delta-NEP cures bulk stiffness deficiencies.
- **Mode-Resolved Imaginary Normal Mode Diagnostics**:
  - Enhanced imaginary mode reporting in `LineshapeEngine` with mode participation ratios ($PR$), top displaced atom breakdowns, and actionable diagnostic guidance distinguishing localized defect saddle points from bulk foundation softening.

## [0.8.0] - 2026-11-06

### Added
- **HPC & Cluster Subsystem (`hydeph.hpc`)**:
  - Implemented `ClusterProfile` models and hardware definitions for SLURM clusters (`generic_slurm`, `leonardo_booster`, `leonardo_booster_debug`, `leonardo_dcgp`).
  - Added deterministic, user-specified module loading preserving exact library order (e.g., `profile/chem-phys` before `quantum-espresso`) without heuristic bash availability checks or unconfigured defaults.
  - Implemented multi-GPU worker pinning (`get_pinned_worker_env`) isolating concurrent workers across physical GPUs (`CUDA_VISIBLE_DEVICES = worker_idx % gpus_per_node`) with dedicated OpenMP/MKL thread counts.
  - Added Lustre/GPFS atomic completion markers (`mark_frame_done`, `is_frame_done`) preventing race conditions during parallel job array sweeps.
  - Added multi-threaded finite-difference supercell Hessian evaluation via `ParallelVibrations` (`parallel_hessian = True`, `parallel_hessian_workers = 16`), reducing 2,388-displacement Hessian evaluation from ~20 minutes to < 1 minute.
- **Decoupled Pure Quantum ESPRESSO Staged Execution**:
  - Added `--prepare-relax` flag to pre-generate ground-state and excited-state $\Delta\text{SCF}$ input decks and standalone SLURM batch script (`submit_relaxations_pure_qe.slurm`) for compute-node offloading without requiring Python or Conda on compute nodes.
  - Added `--prepare-dft` / `--to-stage 2` input preparation generating individual frame directories (`dft_calculations/frame_*`, `dft_calculations_es/frame_es_*`), SLURM Job Array scripts (`submit_dft_array_pure_qe.slurm` with `dft_frames_list.txt`), and multi-GPU node batch scripts (`submit_dft_pure_qe.slurm`).
  - Added automatic calculation resumption (`--from-stage 2`, `--from-stage 4`) discovering completed `pw.out` files, ingesting converged total energies and forces, and deriving adiabatic Zero-Phonon Line and Stokes energetics directly from disk.
- **Automated Lowest-Energy $\Delta\text{SCF}$ Excitation & Degeneracy Symmetrization**:
  - Automatically identifies in-gap defect frontier orbitals and selects the lowest-energy optical excitation ($\min(\epsilon_{\mathrm{LUMO}} - \epsilon_{\mathrm{HOMO}})$ across spin channels) from ground-state Kohn-Sham eigenvalues when explicit band indices are omitted.
  - Detects degenerate frontier manifolds and symmetrizes fractional occupations across both $g_{\mathrm{H}}$-fold degenerate HOMO holes and $g_{\mathrm{L}}$-fold degenerate LUMO electrons, preventing SCF charge sloshing in high-symmetry defects.
- **Automated Jahn-Teller Detection & Symmetry-Breaking Distortion**:
  - Automatically detects degenerate orbital manifolds ($g_{\mathrm{H}} > 1$ or $g_{\mathrm{L}} > 1$) and applies a coordinated symmetry-breaking displacement (`jahn_teller_amplitude = 0.04 Å`) prior to $\Delta\text{SCF}$ relaxation when `auto_jahn_teller_distortion = True`, resolving physical Jahn-Teller ground/excited states without manual geometry perturbation.
- **Automated Plane-Wave Cutoff Derivation**:
  - Added `auto_resolve_cutoffs()` in `ToolkitConfig`, automatically deriving `ecutwfc` and `ecutrho` based on system element hardness (first-row $2p$ and $3d$ transition-metal elements) and UPF pseudopotential metadata when cutoffs are omitted.
- **4H-SiC $kk$-$\mathrm{VV}^0$ Divacancy Benchmark Reproduction**:
  - Turnkey reproduction and validation against hybrid-DFT calculations and experimental photoluminescence spectra from Jin *et al.* (*Phys. Rev. Materials* **5**, 084603, 2021).
  - Accurately resolved all three literature phonon sideband features: Peak 1 at $-34\ \mathrm{meV}$ (breathing mode), Peak 2 at $-73\ \mathrm{meV}$ (localized defect mode), and Peak 3 at $-96\ \mathrm{meV}$ (bulk optical band edge).
  - Preserved distinct Zero-Phonon Line (ZPL) peak at $300\ \mathrm{K}$ through $(5\times 5\times 2)$ Alkauskas large-supercell phonon embedding.
  - Added digitized experimental comparison curves (`jin_et_al_digitized_kk_10K.csv`) and comparative overlay figures.
- **Dynamic Smart Phonon Axis Scaling & Custom Limits**:
  - Implemented dynamic y-axis scaling in `LineshapeEngine.get_smart_phonon_limits()` ($y_{\max} = \max(1.18, \text{peak\_y} \times 1.15)$), preventing truncation of sharp localized vibronic resonance peaks in multi-temperature spectral density plots.
  - Added `phonon_y_min` and `phonon_y_max` configuration parameters for user-defined phonon plot axis limits.
- **Generalized HPC Documentation**:
  - Replaced cluster-specific guide with generalized `docs/hpc_guide.md`, detailing SLURM job array execution, multi-GPU pinning, decoupled staged workflows, and CINECA Leonardo as a reference architecture.

### Changed
- Replaced heuristic `if module is-avail ...` bash loops in generated SLURM scripts with deterministic `format_slurm_module_lines()`.
- Emptied `default_modules = []` in cluster profiles; modules must be explicitly declared under `modules:` (or aliases `slurm_modules:`, `cluster_modules:`, `hpc_modules:`).
- Added explicit logger warnings when cluster script generation is invoked with an empty module list.
- Calibrated default `max_test_force_rmse_eV_A` quality control threshold to `0.15 eV/Å`.
- Applied TEND prose polishing across documentation, docstrings, error messages, and script templates, pruning throat-clearing, AI clichés, and code-echoing comments.
- Synchronized configuration reference defaults between documentation and `ToolkitConfig` dataclass defaults (`temperatures`, `pseudo_dir`, `n_procs_qe`, `anharmonic_md_steps`).

### Fixed
- **Atomic Shape Broadcasting in Anharmonic Lineshape Engine**:
  - Resolved shape broadcasting mismatch when passing embedded ground-state structures to finite-temperature anharmonic MD by updating `at_es` in the embedded coordinate system.
- **Inverted Excited-State Potential Curvature**:
  - Added defensive validation and harmonic curvature recovery in `ConfigurationCoordinateEngine` when noisy DFT samples produce inverted or negative excited-state relaxation energies.
- **Purged Nonexistent `q-e` Module**:
  - Removed all occurrences of the nonexistent module name `q-e` across codebase, tests, and documentation.
- **SLURM Job Array Path Resolution**:
  - Made `dft_frames_list.txt` and frame directory resolution in SLURM runner scripts robust against submit directory and workspace directory nesting.
- **NEP Executable Discovery**:
  - Extended GPUMD `nep` binary search to inspect user home directory, `$WORK`, and active Conda environment bin paths.

## [0.7.0] - 2026-10-27

### Added
- **Paired Ground + Optical Excited State $\Delta$-NEP Architecture (`A2`, `A7`)**:
  - Established paired $\Delta$-NEP models ($\Delta V_{\text{GS}}$ and $\Delta V_{\text{ES}}$) as the default workflow (`fit_paired_excited_state=True`), allowing accurate independent representation of both ground and optical excited potential energy surfaces.
  - Implemented `PairedDefectCalculator` in `calculator.py` combining foundation potential with both ground- and excited-state $\Delta$-NEPs.
  - Implemented $\Delta\text{SCF}$ constrained-occupation labeling (`occupations='from_input'`) in `QERunner` for Franck-Condon frames, with automated Kohn-Sham orbital defect level detection.
- **Fail-Loud Architecture & Provenance Tracking (`A1`)**:
  - Enforced `allow_fallbacks=False` as strict default across all pipeline stages; failures in ground-state or excited-state DFT relaxations raise fatal exceptions with remediation guidance instead of silently reverting to unoptimized or synthetic models.
  - Added `provenance.json` tracking system environment, software versions, input file SHA256 checksums, and execution metrics.
- **DFT Rigor & Spin Parity (`A3`)**:
  - Standardized on `occupations='fixed'` as the default for localized defect color centers in wide-gap insulators.
  - Scaled convergence thresholds by atom count (`conv_thr = conv_thr_per_atom * N_atoms`).
  - Added automated norm-conserving detection setting `ecutrho = 4 * ecutwfc`.
  - Added HSE hybrid functional support (`hybrid_functional='hse'`).
  - Fixed spin-parity validation bug and prioritized physical anchor frames (GS, ES, Franck-Condon endpoints) in DFT sample budgets.
- **$\Delta$-NEP Quality Control Gates & Committee Ensembles (`A9`)**:
  - Implemented per-atom mean energy offset subtraction $\bar{\Delta E}$ during $\Delta$-NEP dataset preparation and restoration in `HybridDefectCalculator`.
  - Added `evaluate_delta_nep_quality_control()` validating train/test energy and force RMSE, Franck-Condon path maximum error, and GS residual force against automated quality thresholds.
  - Implemented committee ensemble support (`nep_committee_size >= 1`) with uncertainty quantification.
- **Large-Supercell Phonon Embedding & Optical Lineshapes (`A4`, `A5`)**:
  - Implemented `embed_phonon_supercell()` supporting displacement and force-based ($\Delta Q_k = F_k / \omega_k^2$) embedding.
  - Added $(\hbar\omega)^3$ optical photoluminescence prefactor: $I_{\text{PL}}(\hbar\omega) \propto (\hbar\omega)^3 A(\hbar\omega - E_{\text{ZPL}})$.
  - Implemented finite-$T$ Debye-Waller factor $\text{DWF}(T) = \exp(-\sum S_k (2n_k+1))$ and enforced strict continuous $S(\hbar\omega)$ normalization ($\int S(\hbar\omega) d\hbar\omega \equiv \sum S_k$).
  - Configurable Lorentzian instrument broadening (`dephasing_gamma_meV`) and imaginary mode detection guards (`max_imaginary_modes=0`).
  - Exported detailed normal mode decomposition and inverse participation ratios to `normal_modes.csv`.
- **Physical Paired Cumulant Anharmonic Module (`A6`)**:
  - Replaced empirical regime switches and friction broadening with the exact physical energy-gap cumulant method.
  - NVT Langevin thermalization $\to$ NVE `VelocityVerlet` production dynamics (zero thermostat friction).
  - Evaluated physical vertical energy-gap trajectory $\Delta U(t) = V_{\text{ES}}(\mathbf{R}(t)) - V_{\text{GS}}(\mathbf{R}(t))$ using the paired excited-state potential.
  - Applied symmetric harmonic quantum correction factor $Q(\hbar\omega, T) = \frac{\hbar\omega / (k_B T)}{2 \tanh(\hbar\omega / (2 k_B T))}$ to obtain $J_{\text{quant}}(\omega)$ and generating function lineshapes.
  - Implemented Linderälv et al. (*npj Comput. Mater.* 11:101, 2025) Gaussian ACF windowing $\widetilde{C}_{\text{cl}}(t) = C_{\text{cl}}(t)\exp(-\frac{1}{2}(\gamma_{\text{acf}} t / \hbar)^2)$ with $\gamma_{\text{acf}} = 2.0\ \text{meV}$ and configurable multi-trajectory averaging (`anharmonic_n_trajectories`), eliminating finite-time ringing and unphysical multi-phonon comb spikes.
  - Implemented temperature-dependent quadratic ZPL dephasing $\gamma(T) = \gamma_0 + \alpha T^2$ maintaining room-temperature ZPL visibility and finite Debye-Waller factor ($\text{DWF} = 6.59\%$ at 300 K) consistent with experiment (Carbery et al., *Nat. Commun.* 15:8710, 2024).
- **Scientific Regression Test Suite (`D`)**:
  - Added analytic 1D displaced harmonic oscillator tests (Poisson progression, sum rule, first and second moments, finite-$T$ DWF).
  - Added toy force-field analytical Hessian and $S_k$ derivation comparison test.
  - Added Quantum ESPRESSO output parser regression fixtures (converged SCF, unconverged, BFGS $n_{\text{step}}$ exhausted, timeout, spin-polarized bands).
- **Benchmark Suite & Data Reproducibility (`C`)**:
  - Established authentic `benchmarks/` data repository (`diamond-nvminus`, `CsI`, `4H-SiC`, `NaCl`).
  - Updated all plotting scripts (`scripts/make_stiff_vs_soft_diagram.py`, `scripts/make_github_diagram.py`, `examples/plot_experimental_comparison.py`) to read exclusively from `benchmarks/` data.

## [0.6.3] - 2026-10-27

### Fixed
- **Atomic Writing File Format**: Explicitly specified `format="extxyz"` and standardized on `.tmp.xyz` temporary filenames when writing relaxed ground-state (`relaxed_gs.xyz`) and excited-state (`relaxed_es.xyz`) structures in `DefectBuilder`. Previously, temporary files ending in `.tmp` caused ASE to attempt format inference from the extension, raising `UnknownFileTypeError: tmp` and erroneously triggering fallback to synthetic breathing displacement.
- **Array Ambiguity in Defect Coordinate Evaluation**: Fixed `ValueError: The truth value of an array with more than one element is ambiguous` in `DefectBuilder.generate_excited_state_geometry()` by safely checking `defect_pos_list is None or len(defect_pos_list) == 0` instead of truthy evaluation on NumPy coordinate arrays.
- **Scratch Directory Disk Reclamation**: Automatically purged scratch `tmp/` folders (wavefunctions and charge densities) upon successful completion of Quantum ESPRESSO geometry relaxations and frame labeling to avoid disk bloat.

## [0.6.2] - 2026-10-27

### Added
- **Automated UPF Pseudopotential Auto-Discovery**: Implemented automated pseudopotential discovery (`discover_pseudopotential` and `ToolkitConfig.auto_discover_pseudopotentials()`) from `pseudo_dir`. Users no longer need to explicitly populate the `pseudopotentials` dictionary for every element in the host crystal and defect.
- **Relativistic Variant Prioritization**: Discovery logic strictly prioritizes scalar-relativistic pseudopotentials over fully-relativistic (`_FR`) variants required by collinear spin-polarized DFT calculations (`nspin=2`), preventing Quantum ESPRESSO crashes while retaining latest version selection (`1.2` > `1.0`).
- **Subdirectory Discovery Support**: UPF search automatically traverses library subdirectories (e.g., SSSP or ONCVPSP folders) when flat root matches are absent.

### Fixed
- **Missing Element KeyError**: Resolved `KeyError: '<element>'` during `write_qe_input` in ground-state/excited-state DFT relaxation and step 3 candidate frame labeling when `pseudopotentials` was omitted or empty.
- **Informative Pseudopotential Diagnostic Errors**: If a pseudopotential cannot be found in `pseudo_dir`, `write_qe_input` now raises an informative `KeyError` specifying the missing element, the directory searched, and explicit remediation steps.
- **Pipeline Logging Sequence**: `run_pipeline` now executes `config.validate()` prior to printing the pipeline banner, accurately reflecting auto-discovered UPF mappings instead of an empty dictionary.

## [0.6.1] - 2026-10-27

### Added / Changed
- **Default Spin-Proxy Relaxation**: Changed default `excited_state_method` to `"spin_proxy"` in `ToolkitConfig` and `DefectBuilder`, providing ab initio excited-state geometries with Jahn-Teller distortions out-of-the-box.
- **Example Config Structure**: Aligned all example configuration scripts (`run_licl_fcenter.py`, `README.md`, `hydeph/README.md`) to follow the 5-stage Configuration Reference section order.
- **Documentation KaTeX Rendering**: Replaced non-standard `\AA` / `\text{\AA}` macros throughout documentation with standard Unicode `Å` and `\text{Å}` to resolve KaTeX rendering errors on GitHub.

### Fixed
- **Candidate Frame Serialization**: Fixed `KeyError: 'key from calculator already exists in atoms.info'` occurring during `candidate_frames.xyz` extended XYZ serialization. Candidate frames generated from Franck-Condon path, core rattling, and foundation MD with Farthest Point Sampling (FPS) now explicitly detach calculators and strip stale evaluation properties (`energy`, `forces`, `stress`) prior to writing.
- **Atomic File Writing**: Added atomic writes (`.tmp` temporary files followed by `os.replace`) for `candidate_frames.xyz` and DFT training frames (`dft_training_frames.xyz`) to prevent corrupted or truncated files if interrupted mid-run.
- **Calculator Results Isolation**: Ensured labeled DFT frames and Delta-NEP training frames explicitly clear active calculator attachments before saving energies and forces into `.info` and `.arrays`.

## [0.6.0] - 2026-10-26

### Added
- **Automated Spin-Proxy Excited-State Relaxation**: Added `excited_state_method="spin_proxy"` in `DefectBuilder` and `ToolkitConfig`, automating DFT BFGS geometry relaxation using a spin-proxy state ($M_{\mathrm{es}} = M_0 + 2$) to capture Jahn-Teller distortions and physical displacements without empirical breathing-sphere assumptions.
- **DFT Ground-State Relaxation**: Added `ground_state_relax_method` (`"auto"`, `"foundation"`, `"dft"`), coordinating ground- and excited-state relaxations under identical DFT exchange-correlation potential and cell parameters.
- **Quantum ESPRESSO Relaxation Parser & Runner**: Added `run_relaxation()` and `parse_qe_relaxation_output()` in `QERunner`, supporting BFGS ionic dynamics, convergence threshold control (`relax_forc_conv_thr`), step limits (`relax_nstep`), and geometry recovery upon wall-clock timeouts.
- **Supercell Size & Convergence Guidelines**: Added comprehensive guidelines, theoretical background, and a host platform sizing table in `README.md` and `docs/THEORY_WALKTHROUGH.md`, detailing why supercells with $L \ge 10\text{--}12\ \text{Å}$ ($\ge 140\text{--}215$ atoms) are required in stiff crystals (diamond, SiC) to avoid discrete multi-phonon comb artifacts.
- **Supercell Dimension Logging**: `DefectBuilder.build_supercell_and_defect()` logs cell dimensions and warns when supercells are too small ($L < 9\ \text{Å}$) for reliable phonon dispersion.
## [0.5.0] - 2026-10-20

### Added
- **Vectorized Supercell Embedding**: Refactored `DefectBuilder.embed_defect_unitcell()` using NumPy broadcasting and matrix products, yielding 10–50× speedup for large supercells.
- **Universal ASE Atomic Mass Table**: Replaced hand-written mass dictionary in `write_qe_input()` with `ase.data.atomic_masses` and `atomic_numbers`, covering all 118 elements correctly.
- **Gamma-Point Real Wavefunctions**: `write_qe_input()` writes `K_POINTS gamma` when `kpoints == (1, 1, 1)`, enabling half-sphere FFTs and halving memory footprint in Quantum ESPRESSO.
- **Single-Pass QE Output Parsing**: Merged two-pass scanning in `parse_qe_output()` into a single streaming pass that parses total energy and atomic forces simultaneously.
- **Calculator Property Gating**: Added standard ASE property guards in `HybridDefectCalculator.calculate()`, eliminating force calculations when only potential energy is requested.
- **Incremental FPS Distance Tracking**: Optimized `farthest_point_sampling()` to incrementally update a running minimum distance vector instead of recomputing full distance slices.
- **MD Trajectory Streaming**: Streamed candidate MD frames directly to trajectory files in `sample_foundation_md_with_fps()`, reducing in-memory RAM accumulation.
- **Exposed NEP Architecture Parameters**: Added `nep_cutoff_radial`, `nep_cutoff_angular`, `nep_n_max_radial`, `nep_n_max_angular`, `nep_basis_size`, and `nep_l_max` to `ToolkitConfig` and `write_delta_nep_in()`.
- **Exposed QE SCF Parameters**: Added `conv_thr`, `mixing_beta`, `mixing_ndim`, and `diagonalization` to `ToolkitConfig` and `write_qe_input()`.
- **ZPL Derivation & Transparency**: Removed silent 1.945 eV default in `ConfigurationCoordinateEngine`. ZPL is now auto-derived from ground and excited state energies with transparent warnings, or configured explicitly.
- **Numerical Data Export**: Added `export_results()` methods in `LineshapeEngine` and `ConfigurationCoordinateEngine` writing CSV and JSON tables alongside generated figures.
- **CLI Entry Point**: Added `hydeph` command-line executable (`[project.scripts]`) supporting YAML, TOML, and JSON configuration files via `hydeph config.yaml`.
- **Configuration Deserialization & Repr**: Added `ToolkitConfig.from_file()`, `ToolkitConfig.from_dict()`, `ToolkitConfig.to_dict()`, and a readable `__repr__` displaying only non-default fields.
- **Logging Integration**: Replaced raw `print()` calls across all modules with `logging.getLogger(__name__)`.
- **Interactive Progress Bars**: Added `tqdm` progress bars for candidate MD sampling, DFT labelling, and configuration coordinate curves.
- **Chemical Symbol Union**: Fixed `DeltaTrainer.train_delta_nep()` to take the union of chemical elements across all frames, preventing incomplete species definitions on vacancy/substitutional frames.
- **Dependency Upper Bounds & Dev Tools**: Added upper bounds to core dependencies in `pyproject.toml` and `setup.py`, and added `ruff` and `mypy` to dev optional dependencies.
- **Module-Level Physical Constants**: Consolidated physical constants (`HBAR_EV_FS`, `KB_EV_K`, `KB_MEV_K`, `MODE_COUPLING_PREFACTOR`, `NV_ZPL_EV`, `FFT_TIME_POINTS`, `FFT_N`) at module scope.
- **Type Annotations**: Added `from __future__ import annotations` and type annotations across public API classes and methods.
- **WSL2 GPU-Accelerated Workflow Documentation**: Added dedicated documentation and setup guidelines in `README.md` strongly recommending Windows Subsystem for Linux (WSL2) with CUDA-accelerated Quantum ESPRESSO (`pw.x`) for massive speedups on Windows hardware.
- **Dynamic OpenMP Thread Scaling**: `QERunner._resolve_omp_threads()` automatically balances CPU cores across active MPI ranks and concurrent workers, preventing 1-core throttling.
- **Norm-Conserving UPF Cutoff Auto-Detection**: `inspect_upf_metadata()` parses pseudopotential headers and automatically lowers `ecutrho` to $4 \times \text{ecutwfc}$ for Norm-Conserving UPFs, eliminating dense FFT grid overhead.
- **Spin-Parity Mismatch Detection**: Automatically validates electron count parity against `spin_multiplicity` / `tot_magnetization` to prevent fractional-spin occupations and artificial metallic charge sloshing.
- **Davidson Subspace & Band Trimming**: Configures `diago_david_ndim = 2` and sizes `nbnd` compactly for gapped defect supercells.

### Changed
- Moved benchmark driver scripts (`run_licl_fcenter.py`, `plot_experimental_comparison.py`) from package root (`hydeph/`) to `examples/`.
- Consolidated `run_pipeline()` parameters (`max_dft_frames`, `n_procs_qe`, `n_workers_qe`) into `ToolkitConfig`.
- Replaced `getattr()` dataclass access patterns with direct attribute access.
- Reduced DFT process polling sleep to responsive intervals and configured `OMP_NUM_THREADS="1"` per worker to prevent thread contention.
- Moved `FixAtoms` import to module level in `defect_builder.py`.
- Relaxed `calorine` dependency cap to allow calorine >= 4.0.

## [0.4.0] - 2026-10-20

### Added
- Smart calculation resumption and budget allocation across candidate categories.
- Type-balanced DFT frame quota preservation across Franck-Condon, rattling, and MD pools.
- Cross-platform Microsoft MPI and Windows executable resolution.

## [0.3.1] - 2026-10-19

### Added
- 1D Franck-Condon configuration coordinate diagram engine and Stokes shift analysis (`cc_diagram.py`).

## [0.3.0] - 2026-10-18

### Added
- Dual-regime anharmonic spectroscopy: quantum generating functions for weak-to-intermediate coupling ($S < 10$) and semiclassical Marcus convolution for strong electron-phonon coupling ($S \ge 10$).
