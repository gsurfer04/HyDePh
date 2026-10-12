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
    defect_type: str = "vacancy"  # "vacancy", "substitutional", or "file"
    defect_site_element: Optional[str] = None
    defect_site_index: Optional[int] = None
    supercell_repeat: Tuple[int, int, int] = (2, 2, 2)
    
    # 3. Foundation potential
    foundation_potential_path: str = "/home/jamesc/GPUMD/potentials/nep/nep89_20250409/nep89_20250409.txt"
    
    # 4. Pseudopotentials
    pseudopotentials: Dict[str, str] = field(default_factory=dict)
    pseudo_dir: str = "/home/jamesc/quantumespresso/pseudo"
    
    # 5. Temperatures of interest
    temperatures: List[float] = field(default_factory=lambda: [10.0, 77.0, 150.0, 300.0])
    
    # Electronic structure & sampling parameters
    spin_polarized: bool = True
    spin_multiplicity: int = 2      # S = 1/2 (2S+1 = 2) for neutral F-center V_Cl^0
    tot_charge: float = 0.0         # Net charge
    ecutwfc: float = 40.0           # Plane wave cutoff (Ry)
    ecutrho: float = 320.0          # Charge density cutoff (Ry)
    kpoints: Tuple[int, int, int] = (1, 1, 1) # Gamma point for defect supercells
    
    # Frame generation parameters
    n_fc_frames: int = 7            # Franck-Condon path frames
    n_phonon_frames: int = 12       # Frozen phonon / thermal rattle frames
    n_md_frames: int = 20           # Diverse frames selected via FPS from nep89 MD
    md_temp_exploration: float = 500.0 # Elevated temperature for exploratory MD
    
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

    def validate(self):
        """Validate input existence and parameters."""
        if not os.path.isfile(self.pristine_cell_path):
            raise FileNotFoundError(f"Pristine cell file not found: {self.pristine_cell_path}")
        
        if not os.path.isfile(self.foundation_potential_path):
            raise FileNotFoundError(f"Foundation potential not found: {self.foundation_potential_path}")
        
        for elem, upf in self.pseudopotentials.items():
            full_upf = upf if os.path.isabs(upf) else os.path.join(self.pseudo_dir, upf)
            if not os.path.isfile(full_upf):
                raise FileNotFoundError(f"Pseudopotential for {elem} not found: {full_upf}")
            self.pseudopotentials[elem] = full_upf

        os.makedirs(self.workspace_dir, exist_ok=True)
        return True
