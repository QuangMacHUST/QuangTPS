"""
Manual Segmentation Tools

This package contains tools for manual contouring and segmentation
in the QuangTPS treatment planning system.
"""

# Import tool classes for easy access
from quangtps.segmentation.manual_segmentation.freehand_tool import FreehandTool, FreehandToolWidget
from quangtps.segmentation.manual_segmentation.polygon_tool import PolygonTool, PolygonToolWidget
from quangtps.segmentation.manual_segmentation.threshold_tool import ThresholdTool, ThresholdToolWidget

__all__ = [
    'FreehandTool', 'FreehandToolWidget',
    'PolygonTool', 'PolygonToolWidget',
    'ThresholdTool', 'ThresholdToolWidget',
] 