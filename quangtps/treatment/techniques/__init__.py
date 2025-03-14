"""
Techniques module của QuangTPS.
Cung cấp các công cụ để định nghĩa và quản lý các kỹ thuật xạ trị khác nhau.
"""

from quangtps.treatment.techniques.conformal import Conformal3DRT
from quangtps.treatment.techniques.imrt import IMRT
from quangtps.treatment.techniques.vmat import VMAT
from quangtps.treatment.techniques.dcat import DCAT
from quangtps.treatment.techniques.stereotactic import SRS, SBRT
from quangtps.treatment.techniques.tbi import TBI, TSI
from quangtps.treatment.techniques.proton import ProtonTherapy, PencilBeamScanning, PassiveScattering
from quangtps.treatment.techniques.carbon import CarbonIonTherapy
from quangtps.treatment.techniques.adaptive import AdaptiveRadiotherapy
from quangtps.treatment.techniques.electron import ElectronTherapy
from quangtps.treatment.techniques.igrt import IGRT
from quangtps.treatment.techniques.flash import FLASHRadiotherapy
from quangtps.treatment.techniques.bnct import BNCT

__all__ = [
    'Conformal3DRT',
    'IMRT',
    'VMAT',
    'DCAT',
    'SRS',
    'SBRT',
    'TBI',
    'TSI',
    'ProtonTherapy',
    'PencilBeamScanning',
    'PassiveScattering',
    'CarbonIonTherapy',
    'AdaptiveRadiotherapy',
    'ElectronTherapy',
    'IGRT',
    'FLASHRadiotherapy',
    'BNCT'
]