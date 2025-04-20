"""
Beams package for QuangTPS.

This package provides a convenient import interface for beam-related modules.
It redirects imports to the actual implementation files located in quangtps.treatment.beams.
"""

# Import beam-related modules from their actual location
from quangtps.treatment.beams.beam import *
from quangtps.treatment.beams.beam_geometry import *
from quangtps.treatment.beams.beam_modifiers import *
from quangtps.treatment.beams.beam_library import *
from quangtps.treatment.beams.beam_sequence_generator import *

# Import BeamSet from planning
from quangtps.planning.beam_set import BeamSet

# Re-export commonly used classes
__all__ = [
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
]
