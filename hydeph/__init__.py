"""
HyDePh: Hybrid Defect Photoluminescence.
Trains Delta-NEP potentials on top of foundation models to compute
multi-temperature optical spectra for point defects in solids.
"""

from __future__ import annotations

import logging

from .config import ToolkitConfig, discover_pseudopotential
from .defect_builder import DefectBuilder, detect_defect_sites, apply_jahn_teller_distortion
from .frame_sampler import FrameSampler, get_rattle_sigma
from .qe_runner import QERunner, extract_and_attach_adiabatic_energetics
from .delta_trainer import DeltaTrainer, evaluate_delta_nep_quality_control
from .calculator import HybridDefectCalculator, PairedDefectCalculator, load_foundation_calculator
from .lineshape import LineshapeEngine
from .cc_diagram import ConfigurationCoordinateEngine, plot_configuration_coordinate
from .pipeline import run_pipeline, setup_logging
from .validator import validate_workflow

__version__ = "0.8.0"

# Package-level null logger so library use does not emit unconfigured warnings
logging.getLogger("hydeph").addHandler(logging.NullHandler())

__all__ = [
    "ToolkitConfig",
    "discover_pseudopotential",
    "DefectBuilder",
    "detect_defect_sites",
    "apply_jahn_teller_distortion",
    "FrameSampler",
    "get_rattle_sigma",
    "QERunner",
    "extract_and_attach_adiabatic_energetics",
    "DeltaTrainer",
    "evaluate_delta_nep_quality_control",
    "HybridDefectCalculator",
    "PairedDefectCalculator",
    "load_foundation_calculator",
    "LineshapeEngine",
    "ConfigurationCoordinateEngine",
    "plot_configuration_coordinate",
    "run_pipeline",
    "setup_logging",
    "validate_workflow",
]
