#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module for structure management and segmentation.

This package provides classes and functions for managing anatomical structures
in radiotherapy treatment planning, including structure definition, manipulation,
and analysis.
"""

from quangtps.segmentation.structures.structure import Structure
from quangtps.segmentation.structures.structure_set import StructureSet, StructureType, StructurePriority
from quangtps.segmentation.structures.geometry import Point, Contour
from quangtps.segmentation.structures.structure_templates import StructureTemplate
from quangtps.segmentation.structures.structure_library import (
    StructureLibrary,
    StandardStructureLibrary,
    UserStructureLibrary
)
from quangtps.segmentation.structures.structure_evaluator import StructureEvaluator
from quangtps.segmentation.structures.dvh_analyzer import DVHAnalyzer
from quangtps.segmentation.structures.treatment_plan_evaluator import TreatmentPlanEvaluator

__all__ = [
    'Structure', 
    'StructureSet',
    'StructureType',
    'StructurePriority',
    'Point', 
    'Contour', 
    'StructureTemplate', 
    'StructureLibrary',
    'StandardStructureLibrary',
    'UserStructureLibrary',
    'StructureEvaluator',
    'DVHAnalyzer',
    'TreatmentPlanEvaluator'
]