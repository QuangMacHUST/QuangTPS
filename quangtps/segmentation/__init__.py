#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Segmentation Module for QuangTPS.

This module provides segmentation functionality for the QuangTPS treatment planning system,
including contour operations, margin generation, and auto-segmentation capabilities.
"""

import logging

from quangtps.segmentation.contour.boolean_operations import BooleanOperations, BooleanOperation
from quangtps.segmentation.contour.margin import MarginGenerator, MarginType
from quangtps.segmentation.contour.contour_tools import ContourTool
from quangtps.segmentation.contour.contour_editor import ContourEditor

# Import structures
from quangtps.segmentation.structures.structure_set import StructureSet
from quangtps.segmentation.structures.structure_library import StructureLibrary
from quangtps.segmentation.structures.structure_templates import StructureTemplate

# Import auto-segmentation components
from quangtps.segmentation.auto_segmentation.atlas import AtlasSegmentor
from quangtps.segmentation.auto_segmentation.unet import UNetSegmentor

logger = logging.getLogger(__name__)

__all__ = [
    # Contour operations
    'BooleanOperations',
    'BooleanOperation',
    'MarginGenerator',
    'MarginType',
    'ContourTool',
    'ContourEditor',
    
    # Structure components
    'StructureSet',
    'StructureLibrary',
    
    # Auto-segmentation components
    'AtlasSegmentor',
    'UNetSegmentor'
]