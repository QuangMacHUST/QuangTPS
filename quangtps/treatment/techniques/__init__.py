#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Package các kỹ thuật điều trị xạ trị.

Package này chứa các module định nghĩa các kỹ thuật điều trị xạ trị khác nhau,
như 3DCRT, IMRT, VMAT, v.v.
"""

from enum import Enum, auto

# Import các module kỹ thuật
from quangtps.treatment.techniques.technique_interface import BaseTreatmentTechnique, TechniqueCategory
from quangtps.treatment.techniques.conformal import Conformal3DRT
from quangtps.treatment.techniques.imrt import IMRT
from quangtps.treatment.techniques.vmat import VMAT
from quangtps.treatment.techniques.dcat import DCAT
from quangtps.treatment.techniques.tomotherapy import Tomotherapy
from quangtps.treatment.techniques.stereotactic import SBRT, SRS
from quangtps.treatment.techniques.brachytherapy import Brachytherapy
from quangtps.treatment.techniques.proton import Proton
from quangtps.treatment.techniques.electron import Electron
from quangtps.treatment.techniques.carbon import Carbon, CarbonIonTherapy
from quangtps.treatment.techniques.tbi import TBI
from quangtps.treatment.techniques.igrt import IGRT
from quangtps.treatment.techniques.adaptive import AdaptiveRT, AdaptiveRadiotherapy
from quangtps.treatment.techniques.flash import FLASH, FLASHRadiotherapy, FLASHTherapy
from quangtps.treatment.techniques.pencil_beam_scanning import PencilBeamScanning
from quangtps.treatment.techniques.bnct import BNCT
from quangtps.treatment.techniques.crt_manager import CRTManager
from quangtps.treatment.techniques.crt_visualizer import CRTVisualizer

# Danh sách kỹ thuật điều trị xạ trị có sẵn
AVAILABLE_TECHNIQUES = {
    'Conformal3DRT': Conformal3DRT,
    'IMRT': IMRT,
    'VMAT': VMAT,
    'DCAT': DCAT,
    'Tomotherapy': Tomotherapy,
    'SBRT': SBRT,
    'SRS': SRS,
    'Brachytherapy': Brachytherapy,
    'Proton': Proton,
    'Electron': Electron,
    'Carbon': Carbon,
    'CarbonIonTherapy': CarbonIonTherapy,
    'TBI': TBI,
    'IGRT': IGRT,
    'AdaptiveRT': AdaptiveRT,
    'AdaptiveRadiotherapy': AdaptiveRadiotherapy,
    'FLASH': FLASH,
    'FLASHRadiotherapy': FLASHRadiotherapy,
    'FLASHTherapy': FLASHTherapy,
    'PencilBeamScanning': PencilBeamScanning,
    'BNCT': BNCT
}

__all__ = [
    'BaseTreatmentTechnique', 'TechniqueCategory',
    'Conformal3DRT', 'IMRT', 'VMAT', 'DCAT', 'Tomotherapy',
    'SBRT', 'SRS', 'Brachytherapy', 'Proton', 'Electron',
    'Carbon', 'CarbonIonTherapy', 'TBI', 'IGRT', 'AdaptiveRT', 
    'AdaptiveRadiotherapy', 'FLASH', 'FLASHRadiotherapy', 'FLASHTherapy', 'PencilBeamScanning', 'BNCT', 
    'CRTManager', 'CRTVisualizer', 'AVAILABLE_TECHNIQUES'
]