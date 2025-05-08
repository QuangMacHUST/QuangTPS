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
    ContourType,
    ContourTool,
    BrushTool,
    ThresholdTool,
    RegionGrowingTool,
    WatershedTool,
)
from quangtps.segmentation.contour.boolean_operations import (
    BooleanOperation,
    BooleanOperations,
)
from quangtps.segmentation.contour.margin import MarginTool, MarginType
from quangtps.segmentation.contour.contour_editor import ContourEditor
from quangtps.segmentation.contour.contour_format_converter import (
    ContourFormatConverter,
)
from quangtps.segmentation.contour.contour_utils import ContourUtils
from quangtps.segmentation.contour.dice import (
    calculate_dice_coefficient,
    calculate_jaccard_index,
    calculate_hausdorff_distance,
    calculate_volume_overlap_metrics,
    batch_calculate_dice,
)

logger = logging.getLogger(__name__)

__all__ = [
    "ContourType",
    "ContourTool",
    "BrushTool",
    "ThresholdTool",
    "RegionGrowingTool",
    "WatershedTool",
    "BooleanOperation",
    "BooleanOperations",
    "MarginTool",
    "MarginType",
    "ContourEditor",
    "ContourFormatConverter",
    "ContourUtils",
    "calculate_dice_coefficient",
    "calculate_jaccard_index",
    "calculate_hausdorff_distance",
    "calculate_volume_overlap_metrics",
    "batch_calculate_dice",
]
