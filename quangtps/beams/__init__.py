"""
Initialization for the beams module.

This module provides classes and functions for managing radiotherapy beams,
including beam parameters, geometry, and modifiers.
"""

import logging

# Forward declarations to avoid circular imports
from quangtps.beams.beam import Beam, BeamType, BeamSet
from quangtps.beams.beam_geometry import BeamGeometry
from quangtps.beams.beam_modifiers import Wedge, Block, Bolus, Compensator

__all__ = [
    'Beam',
    'BeamType',
    'BeamSet',
    'BeamGeometry',
    'Wedge',
    'Block',
    'Bolus',
    'Compensator'
] 