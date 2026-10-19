"""Configuration and input validation for HyDePh."""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import ase.io
from ase import Atoms


@dataclass
class ToolkitConfig:
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
    mpi_command: Optional[str] = None
    mpi_args: Optional[List[str]] = None
    nep_exe: Optional[str] = None
    n_workers_qe: int = 1
    qe_timeout_s: Optional[float] = None

    charge: Optional[float] = None  # Net charge in e
    tot_charge: float = 0.0         # QE alias for charge
    spin_polarized: bool = True     # nspin=2 in QE
    spin_multiplicity: int = 2      # 2S+1
    tot_magnetization: Optional[int] = None  # 2S = spin_multiplicity - 1
    ecutwfc: float = 40.0           # Ry
    ecutrho: float = 320.0          # Ry
    kpoints: Tuple[int, int, int] = (1, 1, 1)

    def __post_init__(self):
        if self.charge is not None:
            self.tot_charge = float(self.charge)
        else:
            self.charge = float(self.tot_charge)

        if self.tot_magnetization is not None:
            self.spin_multiplicity = int(self.tot_magnetization) + 1
        else:
            self.tot_magnetization = max(0, int(self.spin_multiplicity) - 1)

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

    enable_anharmonic: bool = True
    anharmonic_md_steps: int = 2500
    anharmonic_dt_fs: float = 1.0
    anharmonic_coupling_regime: str = "auto"  # "auto" (S < 10 quantum, S >= 10 semiclassical), "quantum", or "semiclassical"

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
