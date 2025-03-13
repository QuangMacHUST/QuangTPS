"""
Beam module của QuangTPS.
Cung cấp các công cụ để quản lý chùm tia trong hệ thống lập kế hoạch xạ trị.
"""

from quangtps.treatment.beams.beam import Beam, BeamType, DoseSpecificationPoint
from quangtps.treatment.beams.beam_geometry import BeamGeometry
from quangtps.treatment.beams.beam_library import BeamLibrary, BeamTemplate, BeamArrangementTemplate
from quangtps.treatment.beams.beam_modifiers import (
    BeamModifier, ModifierType, Wedge, Block, Bolus, Compensator
)

__all__ = [
    'Beam', 'BeamType', 'DoseSpecificationPoint',
    'BeamGeometry',
    'BeamLibrary', 'BeamTemplate', 'BeamArrangementTemplate',
    'BeamModifier', 'ModifierType', 'Wedge', 'Block', 'Bolus', 'Compensator'
]