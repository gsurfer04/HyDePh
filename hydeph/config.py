"""Configuration dataclass and input validation for HyDePh workflows."""

from __future__ import annotations

import os
import json
import logging
from dataclasses import dataclass, field, fields
from typing import List, Dict, Optional, Tuple, Any

logger = logging.getLogger(__name__)


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
        Path to pre-constructed or DFT-relaxed excited state geometry.
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
    supercell_repeat : tuple of int
        Supercell expansion multipliers (nx, ny, nz). Default is (2, 2, 2).

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
    mpi_command : str, optional
        Custom MPI runner command (e.g. "mpirun", "mpiexec").
    mpi_args : list of str, optional
        Extra arguments passed directly to the MPI runner.
    qe_timeout_s : float, optional
        Timeout per Quantum ESPRESSO calculation in seconds.
    conv_thr : str
        SCF convergence threshold in Quantum ESPRESSO. Default is "1.0d-5".
    mixing_beta : float
        SCF charge density mixing factor. Default is 0.4.
    mixing_ndim : int
        Number of iterations used in mixing. Default is 8.
    diagonalization : str
        Diagonalization method ('david', 'cg', etc.). Default is "david".

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
    ecutwfc : float
        Plane-wave cutoff energy for wavefunctions (Ry). Default is 40.0.
    ecutrho : float
        Plane-wave cutoff energy for charge density (Ry). Default is 320.0.
    kpoints : tuple of int
        Monkhorst-Pack k-point grid. Default is (1, 1, 1) (Gamma-point).

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
    excited_state_displacement: Optional[float] = None  # Angstrom
    defect_type: str = "vacancy"  # "vacancy", "substitutional", or "file"
    defect_site_element: Optional[str] = None
    substitutional_element: Optional[str] = None
    defect_site_index: Optional[int] = None
    supercell_repeat: Tuple[int, int, int] = (2, 2, 2)

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
    mpi_command: Optional[str] = None
    mpi_args: Optional[List[str]] = None
    nep_exe: Optional[str] = None
    qe_timeout_s: Optional[float] = None

    conv_thr: str = "1.0d-5"
    mixing_beta: float = 0.4
    mixing_ndim: int = 8
    diagonalization: str = "david"

    charge: Optional[float] = None  # Net charge in e
    tot_charge: float = 0.0         # QE alias for charge
    spin_polarized: bool = True     # nspin=2 in QE
    spin_multiplicity: int = 2      # 2S+1
    tot_magnetization: Optional[int] = None  # 2S = spin_multiplicity - 1
    ecutwfc: float = 40.0           # Ry
    ecutrho: float = 320.0          # Ry
    kpoints: Tuple[int, int, int] = (1, 1, 1)

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
    reuse_gs_density: bool = False

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

    enable_anharmonic: bool = True
    anharmonic_md_steps: int = 2500
    anharmonic_dt_fs: float = 1.0
    anharmonic_coupling_regime: str = "auto"  # "auto", "quantum", or "semiclassical"

    workspace_dir: str = "./hydeph_run"
    lineshape_figure_name: str = "lineshapes.png"
    anharmonic_figure_name: str = "anharmonic_lineshapes.png"

    temperature_plot_mode: str = "both"  # "combined", "individual", or "both"
    show_harmonic_in_anharmonic: bool = True
    individual_figure_prefix: str = "lineshape"
    smart_axis_scaling: bool = True
    phonon_hw_min: Optional[float] = None    # meV
    phonon_hw_max: Optional[float] = None    # meV
    lineshape_e_min: Optional[float] = None  # meV relative to ZPL
    lineshape_e_max: Optional[float] = None  # meV relative to ZPL
    lineshape_y_min: Optional[float] = None
    lineshape_y_max: Optional[float] = None
    lineshape_y_scale: str = "linear"        # "linear" or "log"
    individual_shared_x: bool = False
    phonon_smearing_meV: Optional[float] = None
    dephasing_gamma_meV: Optional[float] = None
    finite_size_acoustic_cutoff: bool = True
    lineshape_n_points: int = 1500

    plot_configuration_coordinate: bool = True
    zpl_energy_eV: Optional[float] = None
    cc_figure_name: str = "configuration_coordinate.png"
    cc_lambda_min: float = -0.30
    cc_lambda_max: float = 1.30
    cc_n_points: int = 81

    def __post_init__(self):
        if self.charge is not None:
            self.tot_charge = float(self.charge)
        else:
            self.charge = float(self.tot_charge)

        if self.tot_magnetization is not None:
            self.spin_multiplicity = int(self.tot_magnetization) + 1
        else:
            self.tot_magnetization = max(0, int(self.spin_multiplicity) - 1)

    def __repr__(self) -> str:
        """Returns a readable representation highlighting only non-default fields."""
        defaults = ToolkitConfig(pristine_cell_path="")
        diffs = {}
        for f in fields(self):
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
            val = getattr(self, f.name)
            if isinstance(val, tuple):
                d[f.name] = list(val)
            else:
                d[f.name] = val
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ToolkitConfig:
        """Constructs a ToolkitConfig from a dictionary, filtering unknown keys."""
        valid_keys = {f.name for f in fields(cls)}
        filtered = {}
        for k, v in data.items():
            if k in valid_keys:
                if k in ("supercell_repeat", "kpoints") and isinstance(v, list):
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

        if self.anharmonic_coupling_regime.lower() not in ("auto", "quantum", "semiclassical"):
            raise ValueError(
                f"Invalid anharmonic_coupling_regime: '{self.anharmonic_coupling_regime}'. "
                "Must be 'auto', 'quantum', or 'semiclassical'."
            )

        if int(self.n_workers_qe) < 1:
            raise ValueError(f"Invalid n_workers_qe: {self.n_workers_qe}. Must be >= 1.")

        if self.qe_timeout_s is not None and float(self.qe_timeout_s) <= 0.0:
            raise ValueError(f"Invalid qe_timeout_s: {self.qe_timeout_s}. Must be > 0.")

        os.makedirs(self.workspace_dir, exist_ok=True)
        return True
