#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Package for structure handling in QuangTPS.

This package provides functionality for working with anatomical structures
in the radiotherapy treatment planning system.
"""

import logging
from quangtps.core.logging import get_logger

# Import directly from local modules - avoid circular imports
from quangtps.structures.structure import Structure
from quangtps.structures.structure_set import StructureSet

# Import structure utilities
try:
    from quangtps.structures.structure_utils import (
        calculate_volume,
        calculate_centroid,
        calculate_overlap,
        calculate_distance,
        find_closest_structure,
        calculate_surface_area,
        calculate_dice_coefficient,
    )
except ImportError as e:
    logging.getLogger(__name__).warning(f"Failed to import structure utilities: {e}")

# Setup basic logging
logger = get_logger(__name__)

# Define structure types for usage throughout the package
from enum import Enum, auto


class StructureType(Enum):
    """Enumeration of structure types."""

    PTV = "PTV"
    CTV = "CTV"
    GTV = "GTV"
    OAR = "OAR"  # Organ at Risk
    EXTERNAL = "EXTERNAL"  # Body contour
    PRV = "PRV"  # Planning organ at Risk Volume
    CONTROL = "CONTROL"  # Control structure for optimization
    CUSTOM = "CUSTOM"  # Custom structure
    UNDEFINED = "UNDEFINED"  # Undefined structure type


__all__ = [
    "Structure",
    "StructureSet",
    "StructureType",
    "calculate_volume",
    "calculate_centroid",
    "calculate_overlap",
    "calculate_distance",
    "find_closest_structure",
    "calculate_surface_area",
    "calculate_dice_coefficient",
]

__version__ = "0.2.0"
