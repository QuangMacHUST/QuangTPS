#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Beams module for QuangTPS.

This is a redirection module that forwards imports to the actual implementation
in quangtps.treatment.beams to maintain backward compatibility and resolve import issues.
"""

import logging
from quangtps.core.logging import get_logger

logger = get_logger(__name__)

# Import beam-related classes from treatment.beams
try:
    from quangtps.treatment.beams.beam import (
        Beam,
        BeamType,
        BeamEnergy,
        BeamParameters,
        PhotonBeam,
        ElectronBeam,
        ProtonBeam,
    )

    # Other beam-related imports
    from quangtps.treatment.beams.beam_modifier import (
        BeamModifier,
        Wedge,
        Block,
        Compensator,
        Bolus,
    )

    # MLC-related imports if available
    try:
        from quangtps.treatment.beams.mlc import (
            MLCModel,
            MLCType,
            MLCLeaf,
            MLCPosition,
        )
    except ImportError as e:
        logger.warning(f"Could not import MLC classes: {e}")

    # Forward all imported names to maintain backward compatibility
    __all__ = [
        # Beam classes
        "Beam",
        "BeamType",
        "BeamEnergy",
        "BeamParameters",
        "PhotonBeam",
        "ElectronBeam",
        "ProtonBeam",
        # Modifiers
        "BeamModifier",
        "Wedge",
        "Block",
        "Compensator",
        "Bolus",
        # MLC
        "MLCModel",
        "MLCType",
        "MLCLeaf",
        "MLCPosition",
    ]

except ImportError as e:
    logger.warning(f"Error importing beam classes from treatment.beams: {e}")

    # Define placeholder classes when actual implementations can't be imported
    class Beam:
        """Placeholder for Beam class."""

        def __init__(
            self,
            name="",
            beam_type=None,
            energy=None,
            gantry_angle=0,
            collimator_angle=0,
            couch_angle=0,
            isocenter=(0, 0, 0),
        ):
            self.name = name
            self.beam_type = beam_type
            self.energy = energy
            self.gantry_angle = gantry_angle
            self.collimator_angle = collimator_angle
            self.couch_angle = couch_angle
            self.isocenter = isocenter
            self.monitor_units = 0
            self.weight = 1.0

    class BeamType:
        """Placeholder for BeamType enum."""

        PHOTON = "PHOTON"
        ELECTRON = "ELECTRON"
        PROTON = "PROTON"

    class BeamEnergy:
        """Placeholder for BeamEnergy class."""

        def __init__(self, value=0, unit="MV"):
            self.value = value
            self.unit = unit

    class BeamParameters:
        """Placeholder for BeamParameters class."""

        def __init__(self):
            pass

    # Add placeholder classes to __all__
    __all__ = ["Beam", "BeamType", "BeamEnergy", "BeamParameters"]

# Version info
__version__ = "0.2.0"
