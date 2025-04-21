#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Beam module for QuangTPS.

This is a forwarding module that redirects to the actual implementation
in quangtps.treatment.beams.beam to maintain backward compatibility.
"""

import logging
from quangtps.core.logging import get_logger

logger = get_logger(__name__)

# Try to import beam classes from their actual location
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

    # Set __all__ to forward all imported names
    __all__ = [
        "Beam",
        "BeamType",
        "BeamEnergy",
        "BeamParameters",
        "PhotonBeam",
        "ElectronBeam",
        "ProtonBeam",
    ]

except ImportError as e:
    logger.warning(f"Error importing beam classes from treatment.beams.beam: {e}")

    # Define placeholder classes
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
            self.beam_type = beam_type or BeamType.PHOTON
            self.energy = energy or BeamEnergy()
            self.gantry_angle = gantry_angle
            self.collimator_angle = collimator_angle
            self.couch_angle = couch_angle
            self.isocenter = isocenter
            self.monitor_units = 0
            self.weight = 1.0

    class PhotonBeam(Beam):
        """Placeholder for PhotonBeam class."""

        def __init__(self, name="", energy=None, **kwargs):
            super().__init__(
                name=name, beam_type=BeamType.PHOTON, energy=energy, **kwargs
            )

    class ElectronBeam(Beam):
        """Placeholder for ElectronBeam class."""

        def __init__(self, name="", energy=None, **kwargs):
            super().__init__(
                name=name, beam_type=BeamType.ELECTRON, energy=energy, **kwargs
            )

    class ProtonBeam(Beam):
        """Placeholder for ProtonBeam class."""

        def __init__(self, name="", energy=None, **kwargs):
            super().__init__(
                name=name, beam_type=BeamType.PROTON, energy=energy, **kwargs
            )

    # Set __all__ to forward the placeholder classes
    __all__ = [
        "Beam",
        "BeamType",
        "BeamEnergy",
        "BeamParameters",
        "PhotonBeam",
        "ElectronBeam",
        "ProtonBeam",
    ]
