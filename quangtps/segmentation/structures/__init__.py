#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module chứa các đối tượng cấu trúc cho phân đoạn trong QuangTPS.

Module này chứa các định nghĩa về cấu trúc phân đoạn như Point (điểm), 
Contour (đường bao), Structure (cấu trúc), StructureSet (tập cấu trúc),
StructureTemplate (mẫu cấu trúc), và các đối tượng liên quan.
"""

from quangtps.segmentation.structures.geometry import Point, Contour
from quangtps.segmentation.structures.structure import Structure, StructureType, StructurePriority
from quangtps.segmentation.structures.structure_set import StructureSet
from quangtps.segmentation.structures.structure_templates import StructureTemplate, template_library

__all__ = [
    'Point',
    'Contour',
    'Structure',
    'StructureType',
    'StructurePriority',
    'StructureSet',
    'StructureTemplate',
    'template_library',
]