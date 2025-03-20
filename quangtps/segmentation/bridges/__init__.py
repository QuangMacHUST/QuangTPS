"""
Bridges module for converting between different representations of structures.

This module provides utilities for converting between different module's
structure representations, ensuring consistency across the system.
"""

from quangtps.segmentation.bridges.structures_bridge import (
    imaging_to_segmentation_structure,
    segmentation_to_imaging_structure,
    imaging_to_segmentation_structure_set,
    segmentation_to_imaging_structure_set
)

__all__ = [
    'imaging_to_segmentation_structure',
    'segmentation_to_imaging_structure',
    'imaging_to_segmentation_structure_set',
    'segmentation_to_imaging_structure_set'
]
