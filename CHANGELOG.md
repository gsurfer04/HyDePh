# Changelog

All notable changes to HyDePh will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

### Changed
- Moved benchmark driver scripts (`run_licl_fcenter.py`, `plot_experimental_comparison.py`) from package root (`hydeph/`) to `examples/`.
- Consolidated `run_pipeline()` parameters (`max_dft_frames`, `n_procs_qe`, `n_workers_qe`) into `ToolkitConfig`.
- Replaced `getattr()` dataclass access patterns with direct attribute access.
- Reduced DFT process polling sleep to responsive intervals and configured `OMP_NUM_THREADS="1"` per worker to prevent thread contention.
- Moved `FixAtoms` import to module level in `defect_builder.py`.

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
