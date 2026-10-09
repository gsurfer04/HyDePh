"""
HyDePh: High-Definition Hybrid Defect Photoluminescence
======================================================
Automated framework for training Delta-Machine Learning Potentials (Delta-NEP)
on top of universal foundation models (e.g., nep89) to compute high-definition
optical photoluminescence emission spectra for color centers in solids.

Inputs:
  1. Pristine bulk unit cell
  2. Pre-made defect unit cell
  3. Foundation potential (e.g., nep89_20250409.txt)
  4. Pseudopotentials (UPF files)
  5. Temperatures of interest (K)

Outputs:
  - Trained Delta-NEP potential (nep_delta.txt)
  - Hybrid potential calculator (HybridDefectCalculator)
  - Electron-phonon spectral density S(hbar omega) & Huang-Rhys factor S
  - Multi-temperature harmonic & anharmonic emission lineshapes A(E - E_ZPL, T)
"""

__version__ = "0.1.0"

from .config import ToolkitConfig
from .defect_builder import DefectBuilder
from .frame_sampler import FrameSampler
from .qe_runner import QERunner
from .delta_trainer import DeltaTrainer
from .calculator import HybridDefectCalculator
from .lineshape import LineshapeEngine
from .pipeline import run_pipeline

__all__ = [
    "ToolkitConfig",
    "DefectBuilder",
    "FrameSampler",
    "QERunner",
    "DeltaTrainer",
    "HybridDefectCalculator",
    "LineshapeEngine",
    "run_pipeline",
]
