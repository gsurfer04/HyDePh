"""
HyDePh: Hybrid Defect Photoluminescence.
Trains Delta-NEP potentials on top of foundation models to compute
multi-temperature optical spectra for point defects in solids.
"""

__version__ = "0.3.2"

from .config import ToolkitConfig
from .defect_builder import DefectBuilder
from .frame_sampler import FrameSampler
from .qe_runner import QERunner
from .delta_trainer import DeltaTrainer
from .calculator import HybridDefectCalculator
from .lineshape import LineshapeEngine
from .cc_diagram import ConfigurationCoordinateEngine, plot_configuration_coordinate
from .pipeline import run_pipeline

__all__ = [
    "ToolkitConfig",
    "DefectBuilder",
    "FrameSampler",
    "QERunner",
    "DeltaTrainer",
    "HybridDefectCalculator",
    "LineshapeEngine",
    "ConfigurationCoordinateEngine",
    "plot_configuration_coordinate",
    "run_pipeline",
]
