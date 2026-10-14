"""
hydeph.config
=============
Defines the input specification and configuration validator for the automated
color center Delta-NEP toolkit.

The 5 core user inputs:
  1. pristine_cell: Path to pristine unit cell (or ASE Atoms)
  2. defect_cell: Path to defect cell (or defect creation specification)
  3. foundation_potential: Path to general-purpose NEP (e.g. nep89_20250409.txt)
  4. pseudopotentials: Dict of {Element: path_to_UPF}
  5. temperatures: List of temperatures in Kelvin (e.g. [10.0, 77.0, 150.0, 300.0])
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import ase.io
from ase import Atoms


@dataclass
class ToolkitConfig:
    # 1. Pristine unit cell
    pristine_cell_path: str
    
    # 2. Defect cell or specification
    defect_cell_path: Optional[str] = None
    excited_cell_path: Optional[str] = None  # Optional explicit excited-state cell or supercell
    excited_state_displacement: Optional[float] = None  # Optional displacement magnitude (A) for procedural ES
    defect_type: str = "vacancy"  # "vacancy", "substitutional", or "file"
    defect_site_element: Optional[str] = None
    defect_site_index: Optional[int] = None
    supercell_repeat: Tuple[int, int, int] = (2, 2, 2)
    
    # 3. Foundation potential
    foundation_potential_path: str = field(
        default_factory=lambda: os.environ.get("NEP_FOUNDATION_PATH", "nep89_20250409.txt")
    )
    
    # 4. Pseudopotentials
    pseudopotentials: Dict[str, str] = field(default_factory=dict)
    pseudo_dir: str = field(
        default_factory=lambda: os.environ.get("ESPRESSO_PSEUDO", "./pseudo")
    )
    
    # 5. Temperatures of interest
    temperatures: List[float] = field(default_factory=lambda: [10.0, 77.0, 150.0, 300.0])
    
    # Binary executable & MPI configuration (supports Windows QMatSuite / MS-MPI and Linux)
    pw_exe: Optional[str] = None
    mpi_command: Optional[str] = None
    mpi_args: Optional[List[str]] = None
    nep_exe: Optional[str] = None

    # Electronic structure & defect state parameters
    charge: Optional[float] = None  # Net system charge in e (e.g. -1 for NV- in diamond, 0 for neutral)
    tot_charge: float = 0.0         # Net system charge in e (Quantum ESPRESSO parameter alias for charge)
    spin_polarized: bool = True     # Spin-unrestricted calculation (nspin=2 in QE)
    spin_multiplicity: int = 2      # 2S+1: 1 for singlet (S=0), 2 for doublet (S=1/2), 3 for triplet (S=1, e.g. NV-)
    tot_magnetization: Optional[int] = None # Unpaired electrons (2S = spin_multiplicity - 1, e.g. 2 for NV-)
    ecutwfc: float = 40.0           # Plane wave cutoff (Ry)
    ecutrho: float = 320.0          # Charge density cutoff (Ry)
    kpoints: Tuple[int, int, int] = (1, 1, 1) # Gamma point for defect supercells

    def __post_init__(self):
        # Synchronize charge and tot_charge
        if self.charge is not None:
            self.tot_charge = float(self.charge)
        else:
            self.charge = float(self.tot_charge)

        # Synchronize spin_multiplicity and tot_magnetization
        if self.tot_magnetization is not None:
            self.spin_multiplicity = int(self.tot_magnetization) + 1
        else:
            self.tot_magnetization = max(0, int(self.spin_multiplicity) - 1)
    
    # Frame generation parameters
    n_fc_frames: int = 7            # Franck-Condon path frames
    n_phonon_frames: int = 12       # Frozen phonon / thermal rattle frames
    n_md_frames: int = 20           # Diverse frames selected via FPS from nep89 MD
    md_temp_exploration: float = 500.0 # Elevated temperature for exploratory MD
    max_dft_frames: Optional[int] = None # Optional limit on number of DFT frames to calculate
    
    # Delta-NEP training parameters
    delta_nep_generation: int = 1
    nep_neurons: int = 30           # Compact ANN architecture for delta correction
    nep_batch: int = 2000
    nep_steps: int = 30000          # Fast convergence on delta forces
    
    # Anharmonic MD & Free Energy Sampling parameters
    enable_anharmonic: bool = True
    anharmonic_md_steps: int = 2500 # MD steps per temperature for thermal sampling
    anharmonic_dt_fs: float = 1.0   # Timestep in fs
    
    # Output workspace and figure naming
    workspace_dir: str = "./hydeph_run"
    lineshape_figure_name: str = "lineshapes.png"
    anharmonic_figure_name: str = "anharmonic_lineshapes.png"
    
    # Graphing and temperature display options
    temperature_plot_mode: str = "both"  # "combined", "individual", or "both"
    show_harmonic_in_anharmonic: bool = True  # graph harmonic curve as dashed line in anharmonic plots
    individual_figure_prefix: str = "lineshape"
    smart_axis_scaling: bool = True          # Automatically frame axes tightly around active spectral features
    phonon_hw_min: Optional[float] = None    # meV, lower cutoff for phonon frequency axis (None for auto)
    phonon_hw_max: Optional[float] = None    # meV, upper cutoff for phonon frequency axis (None for adaptive)
    lineshape_e_min: Optional[float] = None  # meV, relative to ZPL (None for smart adaptive range)
    lineshape_e_max: Optional[float] = None  # meV, relative to ZPL (None for smart adaptive range)
    lineshape_y_min: Optional[float] = None  # Lower limit for lineshape intensity axis (None for auto)
    lineshape_y_max: Optional[float] = None  # Upper limit for lineshape intensity axis (None for auto)
    lineshape_y_scale: str = "linear"        # "linear" or "log"
    individual_shared_x: bool = False        # If True, individual plots share global x-axis; if False, each frames its temperature
    phonon_smearing_meV: Optional[float] = None # meV, Gaussian smearing for discrete supercell modes (None for material-adaptive)
    dephasing_gamma_meV: Optional[float] = None # meV, ZPL dephasing width (None for temperature-dependent default)
    finite_size_acoustic_cutoff: bool = True # Enforces physical acoustic scaling S(hw) -> 0 as hw -> 0
    lineshape_n_points: int = 1500

    def validate(self):
        """Validate input existence and parameters."""
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

        os.makedirs(self.workspace_dir, exist_ok=True)
        return True
