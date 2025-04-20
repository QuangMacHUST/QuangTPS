"""
QuangTPS: A Modern Radiotherapy Treatment Planning System

QuangTPS is a Python-based radiotherapy treatment planning system
that provides tools for contouring, planning, dose calculation,
optimization, and evaluation.
"""

__version__ = "0.2.0"
__author__ = "QuangTPS Team"
__license__ = "MIT"

# Import commonly used modules for easier access
from quangtps.core.logging import get_logger, setup_logging

import os
import sys
import logging
from pathlib import Path
from enum import Enum

# Setup logging
logger = logging.getLogger(__name__)


# Define MLCType enum here to avoid import issues
class MLCType(Enum):
    STATIC = "static"
    DYNAMIC = "dynamic"
    STEP_AND_SHOOT = "step_and_shoot"


# Root directory
ROOT_DIR = Path(__file__).parent.absolute()

# Add module directories to path if needed
for module_dir in ["beams", "dose", "structures", "planning", "ui"]:
    module_path = ROOT_DIR / module_dir
    if module_path.exists() and str(module_path) not in sys.path:
        sys.path.insert(0, str(module_path))

# Import key components
try:
    # Core modules
    from quangtps.core import types, exceptions, services

    # Beam-related modules - import directly from treatment.beams to avoid circular imports
    from quangtps.treatment.beams.beam import Beam, BeamType
    from quangtps.treatment.beams.beam_geometry import BeamGeometry
    from quangtps.treatment.beams.beam_modifiers import Wedge, Block, Bolus, Compensator
    from quangtps.planning.beam_set import BeamSet
    from quangtps.planning.mlc import MLC

    # Structure-related modules
    from quangtps.structures.structure import Structure
    from quangtps.structures.structure_set import StructureSet

    # Planning modules
    from quangtps.planning.plan import Plan, PlanCollection

    # Dose calculation
    from quangtps.dose.dose_calculator import DoseCalculator

    # UI components (lazy-loaded to avoid overhead)
    def get_main_window(*args, **kwargs):
        from quangtps.ui.main_window import MainWindow

        return MainWindow(*args, **kwargs)

except ImportError as e:
    logger.warning(f"Error importing modules: {str(e)}")
    logger.warning("Some features may not be available")

# Define public API
__all__ = [
    # Core
    "services",
    "types",
    "exceptions",
    # Beams
    "Beam",
    "BeamSet",
    "BeamType",
    "BeamGeometry",
    "Wedge",
    "Block",
    "Bolus",
    "Compensator",
    "MLC",
    "MLCType",
    # Structures
    "Structure",
    "StructureSet",
    # Planning
    "Plan",
    "PlanCollection",
    # Dose
    "DoseCalculator",
    # UI
    "get_main_window",
    # Constants
    "__version__",
    "ROOT_DIR",
]
