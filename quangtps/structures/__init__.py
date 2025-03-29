#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Package for structure handling in QuangTPS.

This package provides functionality for working with anatomical structures
in the radiotherapy treatment planning system. It re-exports the basic 
structures from core.structures for backward compatibility.
"""

# Re-export from core.structures for backward compatibility
from quangtps.core.structures import Structure, StructureSet, StructureType

# Import specialized structure handling
try:
    from quangtps.segmentation.structures.structure import StructureData
    from quangtps.segmentation.structures.structure_set import StructureSetData
    from quangtps.segmentation.structures.structure_templates import StructureTemplate
    from quangtps.segmentation.structures.structure_library import StructureLibrary
    from quangtps.segmentation.structures.structure_analysis import StructureAnalyzer
except ImportError as e:
    import logging
    logging.getLogger(__name__).warning(f"Could not import specialized structure modules: {e}")

__all__ = [
    'Structure',
    'StructureSet',
    'StructureType',
    'StructureData',
    'StructureSetData',
    'StructureTemplate',
    'StructureLibrary',
    'StructureAnalyzer'
]

__version__ = '0.1.0' 