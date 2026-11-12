"""Quantum ESPRESSO integration package for HyDePh."""

from .io import (
    is_windows,
    resolve_pw_executable,
    resolve_mpi_runner,
    build_qe_command,
    build_delta_scf_occupations,
    write_qe_input,
    check_pw_in_matches_atoms,
    discover_pseudopotential,
    inspect_upf_metadata,
    HARD_ELEMENT_CUTOFFS,
    resolve_pseudo_path,
    RY_TO_EV,
    RY_BOHR_TO_EV_A,
    BOHR_TO_ANGSTROM,
)
from .parsers import (
    parse_qe_output,
    parse_qe_relaxation_output,
    parse_qe_band_gap,
    resolve_delta_scf_spec,
    find_frame_calc_dir,
    extract_and_attach_adiabatic_energetics,
)
from .slurm import (
    format_slurm_module_lines,
    run_monitored_qe_subprocess,
)
from .runner import QERunner

__all__ = [
    "is_windows",
    "resolve_pw_executable",
    "resolve_mpi_runner",
    "build_qe_command",
    "build_delta_scf_occupations",
    "write_qe_input",
    "check_pw_in_matches_atoms",
    "discover_pseudopotential",
    "inspect_upf_metadata",
    "HARD_ELEMENT_CUTOFFS",
    "resolve_pseudo_path",
    "RY_TO_EV",
    "RY_BOHR_TO_EV_A",
    "BOHR_TO_ANGSTROM",
    "parse_qe_output",
    "parse_qe_relaxation_output",
    "parse_qe_band_gap",
    "resolve_delta_scf_spec",
    "find_frame_calc_dir",
    "extract_and_attach_adiabatic_energetics",
    "format_slurm_module_lines",
    "run_monitored_qe_subprocess",
    "QERunner",
]
