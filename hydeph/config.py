"""Configuration dataclass and input validation for HyDePh workflows."""

from __future__ import annotations

import os
import json
import logging
import re
from dataclasses import dataclass, field, fields
from typing import List, Dict, Optional, Tuple, Any, Iterable, Set

logger = logging.getLogger(__name__)


def discover_pseudopotential(element: str, pseudo_dir: str) -> Optional[str]:
    """Auto-discovers the best matching UPF pseudopotential for an element in pseudo_dir.

    Prioritizes scalar-relativistic pseudopotentials over fully-relativistic (FR)
    variants (as collinear spin-polarized DFT calculations require scalar-relativistic
    potentials) and selects the latest version available.

    Parameters
    ----------
    element : str
        Chemical element symbol (e.g. 'Cs', 'I', 'C', 'Si').
    pseudo_dir : str
        Directory path to search for UPF files.

    Returns
    -------
    str or None
        Filename of the best matching UPF file, or None if not found.
    """
    if not pseudo_dir or not os.path.isdir(pseudo_dir):
        return None

    clean_elem = element.strip()
    if not clean_elem:
        return None

    # Matches filenames like 'C.upf', 'C_ONCV_PBE-1.2.upf', 'Cs_ONCV_PBE-1.2.upf'
    # but ensures element boundary: 'C' will NOT match 'Cs', 'Cl', 'Ca', etc.
    pattern = re.compile(rf"^{re.escape(clean_elem)}([_\.\-].*)?\.upf$", re.IGNORECASE)
    candidates: List[str] = []

    try:
        filenames = os.listdir(pseudo_dir)
        for fname in filenames:
            if pattern.match(fname):
                full_path = os.path.join(pseudo_dir, fname)
                if os.path.isfile(full_path):
                    candidates.append(fname)
    except OSError:
        pass

    # If not found directly in pseudo_dir, search subdirectories (e.g., SSSP, ONCVPSP)
    if not candidates:
        try:
            for root, _dirs, files in os.walk(pseudo_dir):
                for fname in files:
                    if pattern.match(fname):
                        rel = os.path.relpath(os.path.join(root, fname), pseudo_dir)
                        candidates.append(rel)
        except OSError:
            pass

    if not candidates:
        return None

    def is_fr(f: str) -> bool:
        base = os.path.basename(f)
        return bool(re.search(r"([_\.\-]fr([_\.\-]|$))", base, re.IGNORECASE))

    def natural_sort_key(s: str):
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)]

    non_fr = [c for c in candidates if not is_fr(c)]
    pool = non_fr if non_fr else candidates
    pool.sort(key=natural_sort_key)
    return pool[-1]


def inspect_upf_metadata(upf_path: str) -> Dict[str, Any]:
    """Reads lightweight header metadata (`is_norm_conserving`, `z_valence`, suggested cutoffs) from a UPF file."""
    info: Dict[str, Any] = {
        "is_norm_conserving": None,
        "z_valence": None,
        "suggested_cutoff_wfc": None,
        "suggested_cutoff_rho": None,
    }
    if not upf_path or not os.path.isfile(upf_path):
        return info
    try:
        header_lines = []
        with open(upf_path, "r", encoding="utf-8", errors="replace") as f:
            for _ in range(200):
                line = f.readline()
                if not line:
                    break
                header_lines.append(line)
                if "<PP_MESH" in line or "<PP_R " in line or "</PP_HEADER>" in line:
                    break
        text = "".join(header_lines)

        # Parse z_valence (UPF v2 attribute or UPF v1 line)
        m_z2 = re.search(r'z_valence\s*=\s*"\s*([-\d.Ee+]+)\s*"', text, re.IGNORECASE)
        if m_z2:
            info["z_valence"] = float(m_z2.group(1))
        else:
            m_z1 = re.search(r"([-\d.Ee+]+)\s+Z\s+valence", text, re.IGNORECASE)
            if m_z1:
                info["z_valence"] = float(m_z1.group(1))

        # Parse suggested cutoffs (UPF v2 attributes or comments)
        m_wfc = re.search(r'(?:suggested_cutoff_wfc|wfc_cutoff)\s*=\s*"\s*([-\d.Ee+]+)\s*"', text, re.IGNORECASE)
        if not m_wfc:
            m_wfc = re.search(r'(?:suggested|recommended)?\s*cutoff(?:\s*for)?\s*wavefunctions?\s*[:=]\s*([-\d.Ee+]+)', text, re.IGNORECASE)
        if m_wfc:
            try:
                info["suggested_cutoff_wfc"] = float(m_wfc.group(1))
            except ValueError:
                pass

        m_rho = re.search(r'(?:suggested_cutoff_rho|rho_cutoff)\s*=\s*"\s*([-\d.Ee+]+)\s*"', text, re.IGNORECASE)
        if not m_rho:
            m_rho = re.search(r'(?:suggested|recommended)?\s*cutoff(?:\s*for)?\s*(?:charge\s*density|density|rho)\s*[:=]\s*([-\d.Ee+]+)', text, re.IGNORECASE)
        if m_rho:
            try:
                val = float(m_rho.group(1))
                # Only treat as energy cutoff (Ry) if >= 50.0; small values (e.g. 6.0 Bohr) are radial mesh radii
                if val >= 50.0:
                    info["suggested_cutoff_rho"] = val
            except ValueError:
                pass

        # Parse pseudopotential type (Norm-Conserving vs Ultrasoft/PAW)
        m_us = re.search(r'is_ultrasoft\s*=\s*"\s*([TFtf])\s*"', text)
        m_paw = re.search(r'is_paw\s*=\s*"\s*([TFtf])\s*"', text)
        m_type = re.search(r'pseudo_type\s*=\s*"\s*([A-Za-z0-9_-]+)\s*"', text)
        if m_us and m_us.group(1).upper() == "T":
            info["is_norm_conserving"] = False
        elif m_paw and m_paw.group(1).upper() == "T":
            info["is_norm_conserving"] = False
        elif m_type:
            ptype = m_type.group(1).upper()
            if ptype in ("NC", "SL", "NCPP"):
                info["is_norm_conserving"] = True
            elif ptype in ("US", "USPP", "PAW"):
                info["is_norm_conserving"] = False
        elif m_us and m_us.group(1).upper() == "F":
            info["is_norm_conserving"] = True
    except Exception:
        pass
    return info


# Recommended plane-wave kinetic energy cutoffs (ecutwfc in Ry) for elements:
# Mapping: element -> (paw_or_uspp_wfc, norm_conserving_wfc)
# Hard elements: 1st-row 1s/2p (C, N, O, F, B, Be, H) and 3d transition metals (Sc to Zn)
HARD_ELEMENT_CUTOFFS: Dict[str, Tuple[float, float]] = {
    # 1st-row 2p and 1s elements (extreme localization, nodeless valence orbitals)
    "C": (65.0, 80.0),
    "N": (65.0, 80.0),
    "O": (65.0, 80.0),
    "F": (70.0, 85.0),
    "B": (55.0, 65.0),
    "Be": (50.0, 65.0),
    "H": (50.0, 60.0),
    "He": (50.0, 60.0),
    "Li": (45.0, 60.0),
    "Ne": (65.0, 80.0),
    # 3d transition metals (localized 3d valence orbitals)
    "Sc": (50.0, 65.0),
    "Ti": (50.0, 65.0),
    "V":  (55.0, 70.0),
    "Cr": (55.0, 70.0),
    "Mn": (55.0, 70.0),
    "Fe": (55.0, 70.0),
    "Co": (55.0, 70.0),
    "Ni": (55.0, 70.0),
    "Cu": (55.0, 70.0),
    "Zn": (55.0, 70.0),
}


@dataclass
class ToolkitConfig:
    """Central configuration dataclass for the HyDePh workflow.

    Controls supercell construction, candidate frame sampling, Quantum ESPRESSO
    labelling, Delta-NEP training, and multi-temperature optical lineshapes.

    Parameters
    ----------
    pristine_cell_path : str
        Path to pristine unit cell file (e.g., .xyz, POSCAR, .cif). Required.
    defect_cell_path : str, optional
        Path to pre-constructed defect cell or unit cell.
    excited_cell_path : str, optional
        Path to pre-constructed or DFT-relaxed excited-state geometry.
    excited_state_method : str
        Method for constructing the optical excited-state geometry: "delta_scf" (automated constrained
        occupation Delta-SCF DFT relaxation, default), "spin_proxy" (legacy spin-proxy proxy relaxation),
        "synthetic" (isotropic breathing mode), or "file".
    delta_scf_homo_band : int, optional
        Band index of the occupied defect state from which to promote an electron in Delta-SCF mode.
        If omitted, auto-detected from ground-state Kohn-Sham eigenvalues.
    delta_scf_lumo_band : int, optional
        Band index of the unoccupied defect state into which to promote an electron in Delta-SCF mode.
        If omitted, auto-detected from ground-state Kohn-Sham eigenvalues.
    delta_scf_spin_channel : int, optional
        Spin channel (1=up, 2=down) for the optical transition promotion in Delta-SCF mode.
        If omitted, auto-detected from the active in-gap defect channel.
    paired_delta_nep : bool
        Whether to train paired Delta-NEP potential surfaces (Delta V_GS and Delta V_ES). Default is True.
    dft_es_candidates : str
        Candidate frame selection for optical excited state Delta-SCF DFT labeling.
        "all" (default) labels all candidate frames (Franck-Condon, rattle, and MD frames) to train
        a full 3D paired excited-state Delta-NEP potential matching the ground-state dataset size.
        "fc_path" labels only the 1D Franck-Condon path frames (fast mode).
    excited_state_tot_magnetization : int, optional
        Target spin magnetization (2S) for excited-state relaxation in "spin_proxy" mode.
        Defaults to ground state tot_magnetization + 2.
    excited_state_spin_multiplicity : int, optional
        Target spin multiplicity (2S+1) for excited state. Derived from excited_state_tot_magnetization + 1 if omitted.
    ground_state_relax_method : str
        Ground-state geometry optimization method: "auto", "foundation", or "dft".
        In "auto" (default), selects "dft" when excited_state_method is "spin_proxy" and "foundation" otherwise.
    relax_forc_conv_thr : float
        Force convergence threshold (Ry/Bohr) for Quantum ESPRESSO BFGS relaxations. Default is 1.0e-3.
    relax_nstep : int
        Maximum ionic optimization steps for Quantum ESPRESSO BFGS relaxations. Default is 100.
    excited_state_displacement : float, optional
        Breathing displacement magnitude (Angstrom) for synthetic excited state generation.
    defect_type : str
        Type of defect: "vacancy", "substitutional", or "file". Default is "vacancy".
    defect_site_element : str, optional
        Element to substitute or vacate (e.g. "C", or "Si->C" notation).
    substitutional_element : str, optional
        Target substitutional element symbol.
    defect_site_index : int, optional
        Explicit 0-based atom index in the supercell for defect creation.
    allow_fallbacks : bool
        If False (default), raises an immediate RuntimeError upon DFT relaxation failure or
        unresolvable ZPL. If True, permits heuristic recovery with prominent warnings.
    supercell_repeat : tuple of int
        Supercell expansion multipliers (nx, ny, nz). Default is (2, 2, 2).
        For stiff covalent hosts (e.g. Diamond, SiC), (3, 3, 3) or >= 140-200 atoms
        is strongly recommended for converged, continuous phonon sidebands.

    foundation_potential_path : str
        Path to the pre-trained foundation NEP model (e.g. nep89_20250409.txt).
    pseudopotentials : dict of str to str
        Mapping of element symbols to pseudopotential UPF file paths.
    pseudo_dir : str
        Directory containing UPF pseudopotential files.

    temperatures : list of float
        Temperatures (K) at which to calculate optical lineshapes. Default is [10.0, 77.0, 150.0, 300.0].

    pw_exe : str, optional
        Path to Quantum ESPRESSO pw.x executable.
    n_procs_qe : int
        Number of MPI processes per Quantum ESPRESSO worker. Default is 2.
    n_workers_qe : int
        Number of concurrent Quantum ESPRESSO worker processes. Default is 1.
    omp_num_threads : int, optional
        OpenMP threads per MPI process. If omitted, automatically computed from CPU cores / (n_procs_qe * n_workers_qe).
    mpi_command : str, optional
        Custom MPI runner command (e.g. "mpirun", "mpiexec").
    mpi_args : list of str, optional
        Extra arguments passed directly to the MPI runner.
    qe_timeout_s : float, optional
        Timeout per Quantum ESPRESSO calculation in seconds.
    conv_thr : str, optional
        SCF convergence threshold in Quantum ESPRESSO. If omitted (None, default),
        automatically scales with supercell size ($1.0\\times 10^{-10}\\ \\mathrm{Ry/atom}$)
        for accurate atomic forces and Hessians.
    mixing_beta : float
        SCF charge density mixing factor. Default is 0.4.
    mixing_ndim : int
        Number of iterations used in mixing. Default is 8.
    diagonalization : str
        Diagonalization method ('david', 'cg', etc.). Default is "david".
    diago_david_ndim : int
        Subspace dimension multiplier for Davidson diagonalization (2 or 4). Default is 2.
    nbnd : int, optional
        Number of Kohn-Sham bands. If omitted, automatically sized from valence electron count when UPF metadata is available.
    auto_spin_parity : bool
        If False (default), raises an explicit ValueError when total valence electrons and tot_magnetization
        have opposite parity (which forces unphysical fractional spin occupations in QE).
        If True, automatically adjusts tot_magnetization to the nearest valid integer parity.
    qe_occupations : str
        Electronic occupations method in Quantum ESPRESSO: "fixed" (default for gapped defect insulators/semiconductors),
        "smearing", or "from_input".
    smearing : str
        Smearing type if qe_occupations="smearing" (e.g. "gaussian", "marzari-vanderbilt", "methfessel-paxton", "fermi-dirac").
        Default is "gaussian".
    degauss : float
        Smearing width in Ry if qe_occupations="smearing". Default is 0.005.

    nep_exe : str, optional
        Path to GPUMD nep executable.
    charge : float, optional
        Net defect system charge in elementary charge units e.
    tot_charge : float
        Quantum ESPRESSO total charge alias.
    spin_polarized : bool
        Whether to perform spin-polarized (nspin=2) DFT calculations. Default is True.
    spin_multiplicity : int
        Spin multiplicity (2S+1). Default is 2.
    tot_magnetization : int, optional
        Target magnetization (2S). Derived from spin_multiplicity - 1 if omitted.
    ecutwfc : float, optional
        Plane-wave cutoff energy for wavefunctions (Ry). If omitted, automatically
        derived from system element hardness and UPF header recommendations.
    ecutrho : float, optional
        Plane-wave cutoff energy for charge density (Ry). If omitted, automatically
        derived (defaults to 4x ecutwfc for norm-conserving, 8x for ultrasoft/PAW).
    kpoints : tuple of int
        Monkhorst-Pack k-point grid. Default is (1, 1, 1) (Gamma-point).
    input_dft : str, optional
        Exchange-correlation functional tag in Quantum ESPRESSO (e.g. "hse", "hse06", "pbe0")
        for hybrid functional DFT calculations.
    exx_fraction : float, optional
        Fraction of exact exchange for hybrid functionals (e.g. 0.25).
    nqx : tuple of int, optional
        Monkhorst-Pack q-point grid for exact exchange evaluations (e.g. (1, 1, 1)).
    ecutfock : float, optional
        Plane-wave cutoff energy for exact exchange operator (Ry).

    n_fc_frames : int
        Number of Franck-Condon 1D path frames to sample. Default is 7.
    n_phonon_frames : int
        Number of localized core rattle frames to sample. Default is 12.
    n_md_frames : int
        Number of foundation MD frames to sample via FPS. Default is 20.
    md_temp_exploration : float
        Temperature (K) for foundation MD candidate sampling. Default is 500.0.
    random_seed : int
        Random seed for reproducible rattle, MD velocities, and train/test splits.
    force_resample : bool
        If True, ignores existing candidate_frames.xyz and resamples from scratch.
    force_retrain : bool
        If True, retrains Delta-NEP even if cached potential exists.
    max_dft_frames : int, optional
        Maximum total DFT calculations to execute.
    max_new_dft_frames : int, optional
        Maximum new DFT calculations to launch in the current session.
    dft_resume_mode : str
        Resumption mode: "target_total" or "max_new". Default is "target_total".
    retry_failed_dft : bool
        Whether to retry unconverged DFT calculations with conservative SCF parameters.
    replace_failed_frames : bool
        Whether to replace persistently unconverged frames with backup samples.

    nep_neurons : int
        Number of neurons in Delta-NEP hidden layer. Default is 30.
    nep_batch : int
        Batch size for Delta-NEP training. Default is 2000.
    nep_steps : int
        Number of training generations for Delta-NEP. Default is 30000.
    nep_cutoff_radial : float
        Radial cutoff for Delta-NEP in Angstroms. Default is 4.5.
    nep_cutoff_angular : float
        Angular cutoff for Delta-NEP in Angstroms. Default is 3.5.
    nep_n_max_radial : int
        Radial expansion basis size for Delta-NEP. Default is 6.
    nep_n_max_angular : int
        Angular expansion basis size for Delta-NEP. Default is 4.
    nep_basis_size : int
        Descriptor basis size for Delta-NEP. Default is 8.
    nep_l_max : int
        Maximum angular momentum for Delta-NEP expansion. Default is 2.

    enable_anharmonic : bool
        Whether to compute anharmonic MD velocity-autocorrelation spectra.
    anharmonic_md_steps : int
        Number of MD steps for anharmonic VACF propagation. Default is 2500.
    anharmonic_dt_fs : float
        Time step (fs) for anharmonic MD propagation. Default is 1.0.
    anharmonic_coupling_regime : str
        "auto", "quantum", or "semiclassical". Default is "auto".

    phonon_y_min : float, optional
        Minimum y-axis limit for phonon spectral density plots.
    phonon_y_max : float, optional
        Maximum y-axis limit for phonon spectral density plots.
    cluster_profile : str
        HPC hardware profile name ("leonardo_booster", "leonardo_dcgp", or "generic_slurm").
    slurm_modules : list of str
        List of environment modules to load in order on HPC clusters (default is empty).
    workspace_dir : str
        Output and scratch working directory. Default is "./hydeph_run".
    plot_configuration_coordinate : bool
        Whether to compute and plot the 1D CC curve and Stokes shifts. Default is True.
    zpl_energy_eV : float, optional
        Zero-Phonon Line energy in eV. If omitted, derived from DFT/CC minima.
    """

    pristine_cell_path: str
    defect_cell_path: Optional[str] = None
    excited_cell_path: Optional[str] = None
    excited_state_seed_path: Optional[str] = None
    excited_state_method: str = "delta_scf"  # "delta_scf", "spin_proxy", "synthetic", or "file"
    delta_scf_homo_band: Optional[int] = None
    delta_scf_homo_degeneracy: Optional[int] = None
    delta_scf_lumo_band: Optional[int] = None
    delta_scf_lumo_degeneracy: Optional[int] = None
    delta_scf_spin_channel: Optional[int] = None
    auto_jahn_teller_distortion: bool = True
    jahn_teller_amplitude: float = 0.04
    paired_delta_nep: bool = True
    dft_es_candidates: str = "all"  # "all" (paired with ground state dataset) or "fc_path"
    excited_state_tot_magnetization: Optional[int] = None
    excited_state_spin_multiplicity: Optional[int] = None
    ground_state_relax_method: str = "auto"  # "auto", "foundation", or "dft"
    prepare_relaxation_inputs: bool = False
    prepare_dft_inputs: bool = False
    relax_forc_conv_thr: float = 1.0e-3
    relax_nstep: int = 100
    excited_state_displacement: Optional[float] = None  # Angstrom
    defect_type: str = "vacancy"  # "vacancy", "substitutional", or "file"
    defect_site_element: Optional[str] = None
    substitutional_element: Optional[str] = None
    defect_site_index: Optional[int] = None
    supercell_repeat: Tuple[int, int, int] = (2, 2, 2)
    allow_fallbacks: bool = False

    foundation_potential_path: str = field(
        default_factory=lambda: os.environ.get("NEP_FOUNDATION_PATH", "nep89_20250409.txt")
    )

    pseudopotentials: Dict[str, str] = field(default_factory=dict)
    pseudo_dir: str = field(
        default_factory=lambda: os.environ.get("ESPRESSO_PSEUDO", "./pseudo")
    )

    temperatures: List[float] = field(default_factory=lambda: [10.0, 77.0, 150.0, 300.0])

    pw_exe: Optional[str] = None
    n_procs_qe: int = 2
    n_workers_qe: int = 1
    omp_num_threads: Optional[int] = None
    mpi_command: Optional[str] = None
    mpi_args: Optional[List[str]] = None
    nep_exe: Optional[str] = None
    qe_timeout_s: Optional[float] = None

    conv_thr: Optional[str] = None
    mixing_mode: Optional[str] = None
    mixing_beta: float = 0.4
    mixing_ndim: int = 8
    diagonalization: str = "david"
    diago_david_ndim: int = 2
    nbnd: Optional[int] = None
    auto_spin_parity: bool = False
    qe_occupations: str = "fixed"
    smearing: str = "gaussian"
    degauss: float = 0.005

    charge: Optional[float] = None  # Net charge in e
    tot_charge: float = 0.0         # QE alias for charge
    spin_polarized: bool = True     # nspin=2 in QE
    spin_multiplicity: Optional[int] = None  # 2S+1 (defaults to 2 if tot_magnetization omitted)
    tot_magnetization: Optional[int] = None  # 2S = spin_multiplicity - 1
    ecutwfc: Optional[float] = None  # Ry (auto-derived for hard elements / UPFs if omitted)
    ecutrho: Optional[float] = None  # Ry (auto-derived from dual rule if omitted)
    _user_specified_ecutwfc: bool = field(default=False, repr=False)
    _user_specified_ecutrho: bool = field(default=False, repr=False)
    kpoints: Tuple[int, int, int] = (1, 1, 1)
    input_dft: Optional[str] = None
    exx_fraction: Optional[float] = None
    nqx: Optional[Tuple[int, int, int]] = None
    ecutfock: Optional[float] = None

    n_fc_frames: int = 7
    n_phonon_frames: int = 12
    n_md_frames: int = 20
    md_temp_exploration: float = 500.0
    random_seed: int = 42
    force_resample: bool = False
    force_retrain: bool = False
    max_dft_frames: Optional[int] = None
    max_new_dft_frames: Optional[int] = None
    dft_resume_mode: str = "target_total"  # "target_total" or "max_new"
    retry_failed_dft: bool = True
    replace_failed_frames: bool = False

    delta_nep_generation: int = 1
    nep_neurons: int = 30
    nep_batch: int = 2000
    nep_steps: int = 30000
    nep_cutoff_radial: float = 4.5
    nep_cutoff_angular: float = 3.5
    nep_n_max_radial: int = 6
    nep_n_max_angular: int = 4
    nep_basis_size: int = 8
    nep_l_max: int = 2
    nep_committee_size: int = 1
    max_test_force_rmse_eV_A: float = 0.15
    max_gs_residual_force_eV_A: float = 0.05
    subtract_mean_energy_offset: bool = True

    enable_anharmonic: bool = True
    anharmonic_md_steps: int = 2500
    anharmonic_dt_fs: float = 1.0
    anharmonic_coupling_regime: str = "cumulant"  # "cumulant", "auto", "quantum", or "semiclassical"
    anharmonic_thermalization_steps: int = 400

    workspace_dir: str = "./hydeph_run"
    lineshape_figure_name: str = "lineshapes.png"
    anharmonic_figure_name: str = "anharmonic_lineshapes.png"

    temperature_plot_mode: str = "both"  # "combined", "individual", or "both"
    show_harmonic_in_anharmonic: bool = True
    individual_figure_prefix: str = "lineshape"
    smart_axis_scaling: bool = True
    phonon_hw_min: Optional[float] = None    # meV
    phonon_hw_max: Optional[float] = None    # meV
    phonon_y_min: Optional[float] = None
    phonon_y_max: Optional[float] = None
    lineshape_e_min: Optional[float] = None  # meV relative to ZPL
    lineshape_e_max: Optional[float] = None  # meV relative to ZPL
    lineshape_y_min: Optional[float] = None
    lineshape_y_max: Optional[float] = None
    lineshape_y_scale: str = "linear"        # "linear" or "log"
    individual_shared_x: bool = False
    phonon_smearing_meV: Optional[float] = None
    dephasing_gamma_meV: Optional[float] = 1.5
    dephasing_temperature_coefficient: float = 5.5e-5
    time_window_tau_fs: Optional[float] = None
    max_imaginary_modes: int = 0
    phonon_supercell_repeat: Optional[Tuple[int, int, int]] = None
    phonon_embedding_method: str = "displacement"  # "displacement" or "force"
    export_normal_modes: bool = True
    anharmonic_n_trajectories: int = 1
    anharmonic_acf_damping_meV: float = 2.0
    calculate_pl_intensity: bool = True
    finite_size_acoustic_cutoff: bool = True
    lineshape_n_points: int = 1500

    plot_configuration_coordinate: bool = True
    zpl_energy_eV: Optional[float] = None
    cc_figure_name: str = "configuration_coordinate.png"
    cc_lambda_min: float = -0.30
    cc_lambda_max: float = 1.30
    cc_n_points: int = 81

    # --- HPC & CINECA Leonardo Configuration ---
    hpc_scheduler: str = "local"  # "local", "slurm"
    cluster_profile: str = "auto"  # "auto", "leonardo_booster", "leonardo_dcgp", "generic_slurm"
    slurm_account: Optional[str] = field(
        default_factory=lambda: os.environ.get("CINECA_ACCOUNT", os.environ.get("SLURM_ACCOUNT", None))
    )
    slurm_partition: Optional[str] = None
    slurm_qos: Optional[str] = None
    slurm_nodes: int = 1
    slurm_gpus_per_node: int = 4
    slurm_cpus_per_task: int = 8
    slurm_modules: List[str] = field(default_factory=list)
    gpu_pinning: bool = True
    parallel_hessian: bool = True
    parallel_hessian_workers: Optional[int] = None
    scratch_dir: Optional[str] = field(
        default_factory=lambda: os.environ.get("SCRATCH", None)
    )

    def __post_init__(self):
        if str(self.dft_es_candidates).lower() not in ("all", "fc_path"):
            raise ValueError(
                f"Invalid dft_es_candidates: {self.dft_es_candidates!r}. Must be 'all' or 'fc_path'."
            )
        self.dft_es_candidates = str(self.dft_es_candidates).lower()

        if self.excited_state_method == "spin_proxy":
            logger.warning(
                "[!] Notice: 'spin_proxy' excited state method optimizes an artificial spin state "
                "rather than the true optical state. Consider 'delta_scf' (default) for physical optical transitions."
            )

        if self.charge is not None and self.tot_charge != 0.0 and abs(float(self.charge) - float(self.tot_charge)) > 1e-5:
            raise ValueError(
                f"Conflicting charge specifications: charge={self.charge} but tot_charge={self.tot_charge}. "
                "Specify only one or ensure they agree."
            )
        if self.charge is not None:
            self.tot_charge = float(self.charge)
        else:
            self.charge = float(self.tot_charge)

        if self.spin_multiplicity is not None and self.tot_magnetization is not None:
            if int(self.tot_magnetization) != int(self.spin_multiplicity) - 1:
                raise ValueError(
                    f"Conflicting spin specifications: spin_multiplicity={self.spin_multiplicity} "
                    f"(implies tot_magnetization={self.spin_multiplicity - 1}) but tot_magnetization={self.tot_magnetization}. "
                    "Specify only one or ensure they agree."
                )

        if self.tot_magnetization is not None:
            self.spin_multiplicity = int(self.tot_magnetization) + 1
        elif self.spin_multiplicity is not None:
            self.tot_magnetization = max(0, int(self.spin_multiplicity) - 1)
        else:
            self.spin_multiplicity = 2
            self.tot_magnetization = 1

        if self.excited_state_tot_magnetization is not None:
            self.excited_state_spin_multiplicity = int(self.excited_state_tot_magnetization) + 1
        elif self.excited_state_spin_multiplicity is not None:
            self.excited_state_tot_magnetization = max(0, int(self.excited_state_spin_multiplicity) - 1)

        if self.phonon_supercell_repeat is not None:
            self.phonon_supercell_repeat = tuple(int(x) for x in self.phonon_supercell_repeat)

        if self.zpl_energy_eV is not None and self.zpl_energy_eV <= 0:
            logger.info(
                f"[i] zpl_energy_eV={self.zpl_energy_eV} <= 0; auto-deriving ZPL from potential surfaces."
            )
            self.zpl_energy_eV = None

        # Normalize slurm_modules
        if isinstance(self.slurm_modules, str):
            if "," in self.slurm_modules:
                self.slurm_modules = [m.strip() for m in self.slurm_modules.split(",") if m.strip()]
            else:
                self.slurm_modules = [m.strip() for m in self.slurm_modules.split() if m.strip()]
        elif isinstance(self.slurm_modules, (list, tuple)):
            self.slurm_modules = [str(m).strip() for m in self.slurm_modules if str(m).strip()]

        # Resolve HPC & Leonardo cluster defaults
        if (
            str(self.hpc_scheduler).lower() == "slurm"
            or str(self.cluster_profile).lower() != "auto"
            or self.prepare_relaxation_inputs
            or self.prepare_dft_inputs
        ):
            from .hpc.cluster_profiles import get_cluster_profile
            prof = get_cluster_profile(self.cluster_profile)
            if self.slurm_partition is None and prof:
                self.slurm_partition = prof.partition
            if self.slurm_qos is None and prof and prof.qos:
                self.slurm_qos = prof.qos

        if self.parallel_hessian and self.parallel_hessian_workers is None:
            self.parallel_hessian_workers = max(1, min(16, os.cpu_count() or 4))

        # Track whether ecutwfc and ecutrho were explicitly supplied by the caller
        if self.ecutwfc is not None:
            self._user_specified_ecutwfc = True
        if self.ecutrho is not None:
            self._user_specified_ecutrho = True

        # Auto-resolve cutoffs if not specified
        if self.ecutwfc is None or self.ecutrho is None:
            self.auto_resolve_cutoffs()

    def __repr__(self) -> str:
        """Returns a readable representation highlighting only non-default fields."""
        defaults = ToolkitConfig(pristine_cell_path="")
        diffs = {}
        for f in fields(self):
            if f.name.startswith("_"):
                continue
            val = getattr(self, f.name)
            def_val = getattr(defaults, f.name)
            if f.name == "pristine_cell_path" or val != def_val:
                diffs[f.name] = val
        diff_str = ", ".join(f"{k}={v!r}" for k, v in diffs.items())
        return f"ToolkitConfig({diff_str})"

    def to_dict(self) -> Dict[str, Any]:
        """Serializes configuration fields to a dictionary."""
        d = {}
        for f in fields(self):
            if f.name.startswith("_"):
                continue
            val = getattr(self, f.name)
            if isinstance(val, tuple):
                d[f.name] = list(val)
            else:
                d[f.name] = val
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ToolkitConfig:
        """Constructs a ToolkitConfig from a dictionary, rejecting unknown keys with helpful suggestions."""
        # Normalize common user-friendly aliases for decoupled execution
        alias_map = {
            "prepare_dft": "prepare_dft_inputs",
            "prepare_scf": "prepare_dft_inputs",
            "prepare_scf_inputs": "prepare_dft_inputs",
            "prepare_sample_frames": "prepare_dft_inputs",
            "prepare_frames": "prepare_dft_inputs",
            "prepare_relax": "prepare_relaxation_inputs",
            "prepare_relaxation": "prepare_relaxation_inputs",
            "modules": "slurm_modules",
            "cluster_modules": "slurm_modules",
            "hpc_modules": "slurm_modules",
            "hpc_account": "slurm_account",
            "account": "slurm_account",
        }
        normalized_data = {alias_map.get(k, k): v for k, v in data.items()}

        valid_keys = {f.name for f in fields(cls) if not f.name.startswith("_")}
        unknown_keys = [k for k in normalized_data.keys() if k not in valid_keys]
        if unknown_keys:
            import difflib
            errors = []
            for k in sorted(unknown_keys):
                matches = difflib.get_close_matches(k, list(valid_keys), n=1, cutoff=0.6)
                if matches:
                    errors.append(f"Unknown configuration key: '{k}'. Did you mean '{matches[0]}'?")
                else:
                    errors.append(f"Unknown configuration key: '{k}'.")
            raise ValueError("\n".join(errors))

        filtered = {}
        for k, v in normalized_data.items():
            if k in ("supercell_repeat", "kpoints", "nqx") and isinstance(v, list):
                filtered[k] = tuple(v)
            else:
                filtered[k] = v
        return cls(**filtered)

    @classmethod
    def from_file(cls, path: str) -> ToolkitConfig:
        """Loads configuration from a YAML, TOML, or JSON file."""
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Config file not found: {path}")

        ext = os.path.splitext(path)[1].lower()
        if ext in (".yaml", ".yml"):
            try:
                import yaml
                with open(path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
            except ImportError:
                raise ImportError("PyYAML is required to read .yaml config files. Install via `pip install pyyaml`.")
        elif ext == ".toml":
            try:
                import tomllib
            except ImportError:
                try:
                    import tomli as tomllib
                except ImportError:
                    raise ImportError("tomllib or tomli is required to read .toml config files.")
            with open(path, "rb") as f:
                data = tomllib.load(f)
        elif ext == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            # Fallback: try yaml first, then json
            try:
                import yaml
                with open(path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
            except Exception:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)

        return cls.from_dict(data)

    @classmethod
    def load_from_yaml(cls, path: str) -> ToolkitConfig:
        """Loads configuration from a YAML file (convenience alias for from_file)."""
        return cls.from_file(path)

    @classmethod
    def from_yaml(cls, path: str) -> ToolkitConfig:
        """Loads configuration from a YAML file (convenience alias for from_file)."""
        return cls.from_file(path)

    @classmethod
    def load_from_json(cls, path: str) -> ToolkitConfig:
        """Loads configuration from a JSON file (convenience alias for from_file)."""
        return cls.from_file(path)

    @classmethod
    def from_json(cls, path: str) -> ToolkitConfig:
        """Loads configuration from a JSON file (convenience alias for from_file)."""
        return cls.from_file(path)

    def get_system_elements(self) -> List[str]:
        """Detect all unique chemical elements present in configured structures and defect settings."""
        elements: Set[str] = set()

        for path in (self.pristine_cell_path, self.defect_cell_path, self.excited_cell_path):
            if path and os.path.isfile(path):
                try:
                    import ase.io
                    atoms = ase.io.read(path)
                    elements.update(atoms.get_chemical_symbols())
                except Exception as e:
                    logger.debug(f"Could not read structure from '{path}' for element discovery: {e}")

        if self.substitutional_element:
            elements.add(self.substitutional_element.strip())

        if self.defect_site_element:
            cleaned = self.defect_site_element.replace("->", " ").replace(":", " ").replace("-", " ")
            for part in cleaned.split():
                if part.isalpha():
                    elements.add(part.strip())

        elements.update(self.pseudopotentials.keys())
        return sorted(list(elements))

    def auto_discover_pseudopotentials(
        self, elements: Optional[Iterable[str]] = None
    ) -> Dict[str, str]:
        """Auto-discovers UPF pseudopotentials for elements from pseudo_dir.

        Parameters
        ----------
        elements : iterable of str, optional
            List of element symbols to discover. If None, auto-detects elements
            from pristine/defect structures and defect specifications.

        Returns
        -------
        dict of str to str
            Mapping of newly discovered element symbols to pseudopotential paths.
        """
        if not self.pseudo_dir or not os.path.isdir(self.pseudo_dir):
            return {}

        targets = list(elements) if elements is not None else self.get_system_elements()
        discovered: Dict[str, str] = {}
        for elem in targets:
            if not elem:
                continue
            # Do not overwrite user-specified pseudopotentials
            if elem in self.pseudopotentials and self.pseudopotentials[elem]:
                continue
            upf_file = discover_pseudopotential(elem, self.pseudo_dir)
            if upf_file:
                full_upf = (
                    upf_file
                    if os.path.isabs(upf_file)
                    else os.path.join(self.pseudo_dir, upf_file)
                )
                self.pseudopotentials[elem] = full_upf
                discovered[elem] = full_upf
                logger.info(f"[+] Auto-discovered pseudopotential for '{elem}': {os.path.basename(full_upf)}")

        return discovered

    def auto_resolve_cutoffs(
        self, elements: Optional[Iterable[str]] = None, force: bool = False
    ) -> Tuple[float, float]:
        """Automatically derives ecutwfc and ecutrho for system elements if not specified.

        Takes into account:
        - Element hardness (first-row elements C, N, O, F, B, Be, H and 3d transition metals).
        - Pseudopotential type (Norm-Conserving vs. Ultrasoft/PAW).
        - UPF header suggested cutoffs (suggested_cutoff_wfc, suggested_cutoff_rho).

        Parameters
        ----------
        elements : iterable of str, optional
            System element symbols. If None, auto-detected from configured structures and defects.
        force : bool
            If True, overrides user-specified cutoffs.

        Returns
        -------
        tuple of (ecutwfc, ecutrho)
        """
        targets = list(elements) if elements is not None else self.get_system_elements()
        if not targets:
            if not self._user_specified_ecutwfc or force or self.ecutwfc is None:
                self.ecutwfc = 40.0
            if not self._user_specified_ecutrho or force or self.ecutrho is None:
                self.ecutrho = 320.0
            return (float(self.ecutwfc), float(self.ecutrho))

        rec_wfcs: List[float] = []
        nc_flags: List[bool] = []
        max_sug_rho = 0.0

        for elem in targets:
            if not elem:
                continue
            upf_path = self.pseudopotentials.get(elem, "")
            if not upf_path and self.pseudo_dir:
                disc = discover_pseudopotential(elem, self.pseudo_dir)
                if disc:
                    upf_path = disc if os.path.isabs(disc) else os.path.join(self.pseudo_dir, disc)
            full_upf = upf_path if os.path.isabs(upf_path) else os.path.join(self.pseudo_dir or ".", upf_path)
            meta = inspect_upf_metadata(full_upf) if os.path.isfile(full_upf) else {}

            is_nc = meta.get("is_norm_conserving")
            if is_nc is None:
                base_low = os.path.basename(full_upf).lower()
                if "oncv" in base_low or "nc" in base_low or "sg15" in base_low:
                    is_nc = True
                else:
                    is_nc = False
            nc_flags.append(is_nc)

            paw_wfc, nc_wfc = HARD_ELEMENT_CUTOFFS.get(elem, (40.0, 50.0))
            wfc_base = nc_wfc if is_nc else paw_wfc

            wfc_sug = meta.get("suggested_cutoff_wfc")
            if wfc_sug is not None:
                wfc_base = max(wfc_base, float(wfc_sug))
            rec_wfcs.append(wfc_base)

            rho_sug = meta.get("suggested_cutoff_rho")
            if rho_sug is not None:
                max_sug_rho = max(max_sug_rho, float(rho_sug))

        all_nc = bool(nc_flags) and all(nc_flags)
        system_wfc = max(rec_wfcs) if rec_wfcs else (50.0 if all_nc else 40.0)

        if not self._user_specified_ecutwfc or force or self.ecutwfc is None:
            self.ecutwfc = system_wfc

        if not self._user_specified_ecutrho or force or self.ecutrho is None:
            if all_nc:
                self.ecutrho = 4.0 * float(self.ecutwfc)
            else:
                self.ecutrho = max(320.0, 8.0 * float(self.ecutwfc))
            if max_sug_rho > 0:
                self.ecutrho = max(float(self.ecutrho), max_sug_rho)

        hard_found = [e for e in targets if e in HARD_ELEMENT_CUTOFFS]
        if hard_found:
            logger.info(
                f"[+] Auto-configured ecutwfc = {self.ecutwfc} Ry, ecutrho = {self.ecutrho} Ry "
                f"(detected hard elements: {sorted(list(set(hard_found)))}, norm_conserving={all_nc})."
            )
        else:
            logger.info(
                f"[+] Auto-configured ecutwfc = {self.ecutwfc} Ry, ecutrho = {self.ecutrho} Ry "
                f"for system elements {sorted(list(set(targets)))}."
            )

        return (float(self.ecutwfc), float(self.ecutrho))

    def validate(self):
        """Validate input files and configuration options."""
        if not os.path.isfile(self.pristine_cell_path):
            raise FileNotFoundError(f"Pristine cell file not found: {self.pristine_cell_path}")
        
        if self.defect_cell_path and not os.path.isfile(self.defect_cell_path):
            raise FileNotFoundError(f"Defect cell file not found: {self.defect_cell_path}")

        if self.excited_cell_path and not os.path.isfile(self.excited_cell_path):
            raise FileNotFoundError(f"Excited state cell file not found: {self.excited_cell_path}")
        
        if not os.path.isfile(self.foundation_potential_path):
            raise FileNotFoundError(
                f"Foundation potential not found: {self.foundation_potential_path}. "
                "Provide a valid path to a GPUMD foundation NEP model file (e.g. nep89_20250409.txt)."
            )

        # Auto-discover any missing pseudopotentials for system elements
        self.auto_discover_pseudopotentials()

        # Auto-resolve cutoffs with discovered UPFs if cutoffs were not user-specified
        if not self._user_specified_ecutwfc or not self._user_specified_ecutrho or self.ecutwfc is None or self.ecutrho is None:
            self.auto_resolve_cutoffs()
        
        for elem, upf in self.pseudopotentials.items():
            full_upf = upf if os.path.isabs(upf) else os.path.join(self.pseudo_dir, upf)
            if not os.path.isfile(full_upf):
                raise FileNotFoundError(
                    f"Pseudopotential for {elem} not found: {full_upf}. "
                    f"Check that pseudo_dir ('{self.pseudo_dir}') contains the UPF file."
                )
            self.pseudopotentials[elem] = full_upf

        if self.pw_exe and (os.path.isabs(self.pw_exe) or ("/" in self.pw_exe or "\\" in self.pw_exe)):
            if not os.path.isfile(self.pw_exe):
                raise FileNotFoundError(f"Specified Quantum ESPRESSO executable not found: {self.pw_exe}")

        if self.nep_exe and (os.path.isabs(self.nep_exe) or ("/" in self.nep_exe or "\\" in self.nep_exe)):
            if not os.path.isfile(self.nep_exe):
                raise FileNotFoundError(f"Specified GPUMD NEP executable not found: {self.nep_exe}")

        if self.temperature_plot_mode.lower() not in ("combined", "individual", "both"):
            raise ValueError(
                f"Invalid temperature_plot_mode: '{self.temperature_plot_mode}'. "
                "Must be 'combined', 'individual', or 'both'."
            )

        if self.excited_state_method.lower() not in ("delta_scf", "synthetic", "spin_proxy", "file"):
            raise ValueError(
                f"Invalid excited_state_method: '{self.excited_state_method}'. "
                "Must be 'delta_scf', 'synthetic', 'spin_proxy', or 'file'."
            )

        if self.ground_state_relax_method.lower() not in ("auto", "foundation", "dft"):
            raise ValueError(
                f"Invalid ground_state_relax_method: '{self.ground_state_relax_method}'. "
                "Must be 'auto', 'foundation', or 'dft'."
            )

        if float(self.relax_forc_conv_thr) <= 0.0:
            raise ValueError(f"Invalid relax_forc_conv_thr: {self.relax_forc_conv_thr}. Must be > 0.")

        if int(self.relax_nstep) < 1:
            raise ValueError(f"Invalid relax_nstep: {self.relax_nstep}. Must be >= 1.")

        if self.lineshape_y_scale.lower() not in ("linear", "log"):
            raise ValueError(
                f"Invalid lineshape_y_scale: '{self.lineshape_y_scale}'. "
                "Must be 'linear' or 'log'."
            )

        if self.dft_resume_mode.lower() not in ("target_total", "max_new"):
            raise ValueError(
                f"Invalid dft_resume_mode: '{self.dft_resume_mode}'. "
                "Must be 'target_total' or 'max_new'."
            )

        if self.anharmonic_coupling_regime.lower() not in ("auto", "quantum", "semiclassical", "cumulant"):
            raise ValueError(
                f"Invalid anharmonic_coupling_regime: '{self.anharmonic_coupling_regime}'. "
                "Must be 'cumulant', 'auto', 'quantum', or 'semiclassical'."
            )

        if int(self.anharmonic_thermalization_steps) < 0:
            raise ValueError(
                f"Invalid anharmonic_thermalization_steps: {self.anharmonic_thermalization_steps}. Must be >= 0."
            )

        if int(self.n_workers_qe) < 1:
            raise ValueError(f"Invalid n_workers_qe: {self.n_workers_qe}. Must be >= 1.")

        if self.qe_timeout_s is not None and float(self.qe_timeout_s) <= 0.0:
            raise ValueError(f"Invalid qe_timeout_s: {self.qe_timeout_s}. Must be > 0.")

        if self.qe_occupations.lower() not in ("fixed", "smearing", "from_input"):
            raise ValueError(
                f"Invalid qe_occupations: '{self.qe_occupations}'. "
                "Must be 'fixed', 'smearing', or 'from_input'."
            )

        if self.qe_occupations.lower() == "smearing":
            if float(self.degauss) <= 0.0:
                raise ValueError(f"Invalid degauss: {self.degauss}. Must be > 0 when qe_occupations='smearing'.")

        if self.input_dft is not None:
            if self.exx_fraction is not None and not (0.0 <= float(self.exx_fraction) <= 1.0):
                raise ValueError(f"Invalid exx_fraction: {self.exx_fraction}. Must be between 0.0 and 1.0.")
            if self.nqx is not None:
                if len(self.nqx) != 3 or any(int(k) < 1 for k in self.nqx):
                    raise ValueError(f"Invalid nqx: {self.nqx}. Must be a 3-tuple of positive integers.")
            if self.ecutfock is not None and float(self.ecutfock) <= 0.0:
                raise ValueError(f"Invalid ecutfock: {self.ecutfock}. Must be > 0.")

        if self.conv_thr is not None and not str(self.conv_thr).strip():
            raise ValueError("conv_thr cannot be an empty string. Provide a valid threshold or None for auto-scaling.")

        # Temperatures validation
        if not self.temperatures or any(float(t) <= 0.0 for t in self.temperatures):
            raise ValueError(
                f"Invalid temperatures: {self.temperatures}. Must be a non-empty list of positive temperatures (K)."
            )

        # Supercell repeat validation
        if len(self.supercell_repeat) != 3 or any(int(r) < 1 for r in self.supercell_repeat):
            raise ValueError(
                f"Invalid supercell_repeat: {self.supercell_repeat}. Must be a 3-tuple of integers >= 1."
            )

        # K-points validation
        if len(self.kpoints) != 3 or any(int(k) < 1 for k in self.kpoints):
            raise ValueError(
                f"Invalid kpoints: {self.kpoints}. Must be a 3-tuple of positive integers."
            )

        # Plane-wave cutoffs validation
        if self.ecutwfc is not None and float(self.ecutwfc) <= 0.0:
            raise ValueError(f"Invalid ecutwfc: {self.ecutwfc}. Must be > 0 Ry.")
        if self.ecutrho is not None and float(self.ecutrho) <= 0.0:
            raise ValueError(f"Invalid ecutrho: {self.ecutrho}. Must be > 0 Ry.")
        if self.ecutwfc is not None and self.ecutrho is not None and float(self.ecutrho) < 4.0 * float(self.ecutwfc):
            raise ValueError(
                f"Invalid ecutrho: {self.ecutrho} Ry is less than 4 * ecutwfc ({4.0 * self.ecutwfc} Ry). "
                "Charge density cutoff must be at least 4 times wavefunction cutoff."
            )

        # Frame numbers validation
        if int(self.n_fc_frames) < 2:
            raise ValueError(f"Invalid n_fc_frames: {self.n_fc_frames}. Must be >= 2 to include lambda=0 and lambda=1.")
        if int(self.n_phonon_frames) < 0:
            raise ValueError(f"Invalid n_phonon_frames: {self.n_phonon_frames}. Must be >= 0.")
        if int(self.n_md_frames) < 0:
            raise ValueError(f"Invalid n_md_frames: {self.n_md_frames}. Must be >= 0.")

        # NEP hyperparameters validation
        if int(self.nep_steps) < 1:
            raise ValueError(f"Invalid nep_steps: {self.nep_steps}. Must be >= 1.")
        if int(self.nep_neurons) < 1:
            raise ValueError(f"Invalid nep_neurons: {self.nep_neurons}. Must be >= 1.")
        if int(self.nep_committee_size) < 1:
            raise ValueError(f"Invalid nep_committee_size: {self.nep_committee_size}. Must be >= 1.")
        if float(self.max_test_force_rmse_eV_A) <= 0.0:
            raise ValueError(f"Invalid max_test_force_rmse_eV_A: {self.max_test_force_rmse_eV_A}. Must be > 0.")
        if float(self.max_gs_residual_force_eV_A) <= 0.0:
            raise ValueError(f"Invalid max_gs_residual_force_eV_A: {self.max_gs_residual_force_eV_A}. Must be > 0.")
        if int(self.nep_batch) < 1:
            raise ValueError(f"Invalid nep_batch: {self.nep_batch}. Must be >= 1.")
        if float(self.nep_cutoff_radial) <= 0.0 or float(self.nep_cutoff_angular) <= 0.0:
            raise ValueError("nep_cutoff_radial and nep_cutoff_angular must be > 0 Angstrom.")
        if int(self.nep_n_max_radial) < 1 or int(self.nep_n_max_angular) < 1 or int(self.nep_basis_size) < 1:
            raise ValueError("nep_n_max_radial, nep_n_max_angular, and nep_basis_size must be >= 1.")

        # Phonon and lineshape parameters validation
        if self.phonon_supercell_repeat is not None:
            if len(self.phonon_supercell_repeat) != 3 or any(int(r) < 1 for r in self.phonon_supercell_repeat):
                raise ValueError(
                    f"Invalid phonon_supercell_repeat: {self.phonon_supercell_repeat}. Must be a 3-tuple of integers >= 1."
                )
        if self.phonon_embedding_method.lower() not in ("displacement", "force"):
            raise ValueError(
                f"Invalid phonon_embedding_method: '{self.phonon_embedding_method}'. Must be 'displacement' or 'force'."
            )
        if self.dephasing_gamma_meV is not None and float(self.dephasing_gamma_meV) < 0.0:
            raise ValueError(f"Invalid dephasing_gamma_meV: {self.dephasing_gamma_meV}. Must be >= 0.")
        if float(self.dephasing_temperature_coefficient) < 0.0:
            raise ValueError(
                f"Invalid dephasing_temperature_coefficient: {self.dephasing_temperature_coefficient}. Must be >= 0."
            )
        if int(self.anharmonic_n_trajectories) < 1:
            raise ValueError(f"Invalid anharmonic_n_trajectories: {self.anharmonic_n_trajectories}. Must be >= 1.")
        if float(self.anharmonic_acf_damping_meV) <= 0.0:
            raise ValueError(f"Invalid anharmonic_acf_damping_meV: {self.anharmonic_acf_damping_meV}. Must be > 0.")
        if self.time_window_tau_fs is not None and float(self.time_window_tau_fs) <= 0.0:
            raise ValueError(f"Invalid time_window_tau_fs: {self.time_window_tau_fs}. Must be > 0 fs.")
        if int(self.max_imaginary_modes) < 0:
            raise ValueError(f"Invalid max_imaginary_modes: {self.max_imaginary_modes}. Must be >= 0.")

        os.makedirs(self.workspace_dir, exist_ok=True)
        return True

    def export_resolved_manifest(self, workspace_dir: Optional[str] = None) -> str:
        """Saves config_resolved.json containing fully validated configuration, versions, and hashes."""
        ws = workspace_dir or self.workspace_dir
        os.makedirs(ws, exist_ok=True)
        manifest_path = os.path.join(ws, "config_resolved.json")

        from .provenance import compute_file_sha256
        import sys
        import time
        from . import __version__

        upf_hashes = {}
        for elem, path in self.pseudopotentials.items():
            full_p = path if os.path.isabs(path) else os.path.join(self.pseudo_dir, path)
            h = compute_file_sha256(full_p) if os.path.isfile(full_p) else None
            upf_hashes[elem] = {"path": path, "sha256": h}

        manifest = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "software_versions": {
                "hydeph": __version__,
                "python": sys.version.split()[0],
                "pw_exe": self.pw_exe,
                "nep_exe": self.nep_exe,
            },
            "input_hashes": {
                "pristine_cell_path": {
                    "path": self.pristine_cell_path,
                    "sha256": compute_file_sha256(self.pristine_cell_path) if os.path.isfile(self.pristine_cell_path) else None,
                },
                "defect_cell_path": {
                    "path": self.defect_cell_path,
                    "sha256": compute_file_sha256(self.defect_cell_path) if self.defect_cell_path and os.path.isfile(self.defect_cell_path) else None,
                },
                "foundation_potential_path": {
                    "path": self.foundation_potential_path,
                    "sha256": compute_file_sha256(self.foundation_potential_path) if os.path.isfile(self.foundation_potential_path) else None,
                },
                "pseudopotentials": upf_hashes,
            },
            "config": self.to_dict(),
        }

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        logger.info(f"[+] Exported resolved configuration manifest to {manifest_path}")
        return manifest_path
