# Changelog

All notable changes to HyDePh will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
