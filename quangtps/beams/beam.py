#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Beam module for QuangTPS.

This is a forwarding module that redirects to the actual implementation
in quangtps.treatment.beams.beam to maintain backward compatibility.
"""

import logging
from quangtps.core.logging import get_logger
import sys
from typing import Tuple, Dict, Any, Optional, List, Union
from enum import Enum, auto
import numpy as np

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
    class BeamType(str, Enum):
        """Enum for different types of beams used in radiation therapy."""

        PHOTON = "PHOTON"  # Standard photon beam
        ELECTRON = "ELECTRON"  # Electron beam
        PROTON = "PROTON"  # Proton beam

        # Delivery technique types
        STATIC = "STATIC"  # Static field delivery
        DYNAMIC = "DYNAMIC"  # Dynamic delivery (e.g. for IMRT)
        IMRT = "IMRT"  # Intensity Modulated Radiation Therapy
        VMAT = "VMAT"  # Volumetric Modulated Arc Therapy
        ARC = "ARC"  # Arc therapy
        CONFORMAL = "CONFORMAL"  # 3D conformal
        SBRT = "SBRT"  # Stereotactic Body Radiation Therapy
        SRS = "SRS"  # Stereotactic Radiosurgery

        # Mixed or special types
        MIXED = "MIXED"  # Mixed beam types
        UNKNOWN = "UNKNOWN"  # Unknown beam type

    class BeamEnergy:
        """Class representing beam energy with value and unit."""

        def __init__(self, value=0, unit="MV"):
            self.value = value
            self.unit = unit  # MV, MeV, etc.

    class BeamParameters:
        """Container for additional beam parameters."""

        def __init__(self):
            self.sad = 1000.0  # Source-Axis Distance (mm)
            self.ssd = None  # Source-Surface Distance (mm)
            self.technique = None
            self.modality = None
            self.dose_rate = None
            self.machine = "TrueBeam"
            self.tolerance_table = "Default"
            self.meta_data = {}  # Additional metadata

    class Beam:
        """Base class for all beam types."""

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
            self.beam_type = beam_type if beam_type else BeamType.PHOTON
            self.energy = energy if energy else BeamEnergy()
            self.gantry_angle = gantry_angle
            self.collimator_angle = collimator_angle
            self.couch_angle = couch_angle
            self.isocenter = isocenter
            self.weight = 1.0
            self.mu = 0.0
            self.parameters = BeamParameters()
            self.control_points = []
            self.id = ""
            self.description = ""
            self.enabled = True

        def __str__(self):
            return f"Beam: {self.name}, Type: {self.beam_type}, Energy: {self.energy.value}{self.energy.unit}"

    class PhotonBeam(Beam):
        """Photon beam class with photon-specific properties."""

        def __init__(self, name="", energy=None, **kwargs):
            super().__init__(name, BeamType.PHOTON, energy, **kwargs)
            self.flattening_filter = True  # True for standard, False for FFF beams

    class ElectronBeam(Beam):
        """Electron beam class with electron-specific properties."""

        def __init__(self, name="", energy=None, **kwargs):
            super().__init__(name, BeamType.ELECTRON, energy, **kwargs)
            self.applicator = None  # Electron applicator info

    class ProtonBeam(Beam):
        """Proton beam class with proton-specific properties."""

        def __init__(self, name="", energy=None, **kwargs):
            super().__init__(name, BeamType.PROTON, energy, **kwargs)
            self.scanning_mode = "PBS"  # Pencil Beam Scanning by default
            self.field_layers = []

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
