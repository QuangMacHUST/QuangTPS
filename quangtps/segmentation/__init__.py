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
from quangtps.segmentation.structures.structure_analysis import StructureAnalyzer

# Import auto-segmentation components
from quangtps.segmentation.auto_segmentation.atlas import AtlasSegmentor
from quangtps.segmentation.auto_segmentation.unet import UNetSegmentor
from quangtps.segmentation.auto_segmentation.tumor_segmentation import TumorSegmentor
from quangtps.segmentation.auto_segmentation.oar_segmentation import OARSegmentor
from quangtps.segmentation.auto_segmentation.model_loader import ModelLoader
from quangtps.segmentation.auto_segmentation.cyclegan import CycleGAN
from quangtps.segmentation.auto_segmentation.semi_automatic import (
    ThresholdSegmenter,
    RegionGrowingSegmenter,
    WatershedSegmenter,
    create_threshold_segmenter,
    create_region_growing_segmenter,
    create_watershed_segmenter
)
from quangtps.segmentation.auto_segmentation.active_contour import (
    ActiveContourSegmenter,
    GVFSnake,
    create_active_contour_segmenter
)
from quangtps.segmentation.auto_segmentation.level_set import (
    LevelSetSegmenter,
    MorphologicalLevelSet,
    GeodesicLevelSet,
    ChanVeseLevelSet,
    SimpleLevelSet,
    create_morphological_level_set,
    create_geodesic_level_set,
    create_chan_vese_level_set,
    create_simple_level_set
)

# Import manual segmentation components
from quangtps.segmentation.manual_segmentation.drawing_tools import DrawingTool
from quangtps.segmentation.manual_segmentation.manual_editor import ManualSegmentationEditor

# Import validation components
from quangtps.segmentation.validation.metrics import SegmentationMetrics, calculate_comprehensive_metrics
from quangtps.segmentation.validation.validator import SegmentationValidator
from quangtps.segmentation.validation.refinement import SegmentationRefinement

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
    'StructureTemplate',
    'StructureAnalyzer',
    
    # Auto-segmentation components
    'AtlasSegmentor',
    'UNetSegmentor',
    'TumorSegmentor',
    'OARSegmentor',
    'ModelLoader',
    'CycleGAN',
    
    # Semi-automatic segmentation components
    'ThresholdSegmenter',
    'RegionGrowingSegmenter',
    'WatershedSegmenter',
    'create_threshold_segmenter',
    'create_region_growing_segmenter',
    'create_watershed_segmenter',
    
    # Active contour components
    'ActiveContourSegmenter',
    'GVFSnake',
    'create_active_contour_segmenter',
    
    # Level set components
    'LevelSetSegmenter',
    'MorphologicalLevelSet',
    'GeodesicLevelSet',
    'ChanVeseLevelSet',
    'SimpleLevelSet',
    'create_morphological_level_set',
    'create_geodesic_level_set', 
    'create_chan_vese_level_set',
    'create_simple_level_set',
    
    # Manual segmentation components
    'DrawingTool',
    'ManualSegmentationEditor',
    
    # Validation components
    'SegmentationMetrics',
    'calculate_comprehensive_metrics',
    'SegmentationValidator',
    'SegmentationRefinement'
]