#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Contour Module for QuangTPS.

This module provides functionality for contour manipulation in radiotherapy treatment planning.
It includes tools for contour creation, editing, boolean operations, margin generation,
and format conversion.
"""

import logging

from quangtps.segmentation.contour.contour_tools import (
    ContourType, ContourTool, BrushTool, ThresholdTool, RegionGrowingTool, WatershedTool
)
from quangtps.segmentation.contour.boolean_operations import BooleanOperator
from quangtps.segmentation.contour.margin import MarginGenerator
from quangtps.segmentation.contour.contour_editor import ContourEditor
from quangtps.segmentation.contour.contour_format_converter import ContourFormatConverter

logger = logging.getLogger(__name__)

__all__ = [
    'ContourType',
    'ContourTool',
    'BrushTool',
    'ThresholdTool',
    'RegionGrowingTool',
    'WatershedTool',
    'BooleanOperator',
    'MarginGenerator',
    'ContourEditor',
    'ContourFormatConverter'
]