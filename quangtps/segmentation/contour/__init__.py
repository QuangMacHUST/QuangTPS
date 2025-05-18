#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module xử lý contour trong QuangTPS.

Module này cung cấp các công cụ để tạo, chỉnh sửa và xử lý contour
cho các cấu trúc giải phẫu trong hệ thống lập kế hoạch xạ trị.
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
    calculate_volumetric_similarity,
    calculate_hausdorff_distance,
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
    "calculate_volumetric_similarity",
    "calculate_hausdorff_distance",
]

try:
    from quangtps.segmentation.contour.contour_data import ContourData, ContourSet

    __all__.extend(["ContourData", "ContourSet"])
except ImportError:
    pass
