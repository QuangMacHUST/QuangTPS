#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Structures package for QuangTPS.

This package provides classes and functionality for managing structures (contours)
used in radiotherapy treatment planning.
"""

__all__ = [
    'Structure', 
    'StructureSet', 
    'StructureTemplate', 
    'StructureLibrary',
    'AnatomicalSite', 
    'StructureType', 
    'StructureAnalyzer', 
    'DoseVolumeAnalyzer', 
    'TreatmentPlanEvaluator'
]

from quangtps.segmentation.structures.structure import Structure
from quangtps.segmentation.structures.structure_set import StructureSet
from quangtps.segmentation.structures.structure_templates import StructureTemplate
from quangtps.segmentation.structures.structure_library import (
    StructureLibrary, 
    AnatomicalSite, 
    StructureType
)
from quangtps.segmentation.structures.structure_analysis import (
    StructureAnalyzer,
    DoseVolumeAnalyzer,
    TreatmentPlanEvaluator
)