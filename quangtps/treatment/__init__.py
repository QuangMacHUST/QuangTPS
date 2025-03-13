"""
Module treatment của QuangTPS.
Cung cấp các công cụ để quản lý và áp dụng các kỹ thuật xạ trị khác nhau.
"""

# Nhập các module từ beams
from quangtps.treatment.beams.beam import Beam
from quangtps.treatment.beams.beam_geometry import BeamGeometry
from quangtps.treatment.beams.beam_library import BeamLibrary, BeamTemplate, BeamArrangementTemplate
from quangtps.treatment.beams.beam_modifiers import BeamModifier, Wedge, Block, Bolus, Compensator

# Nhập module fractionation
from quangtps.treatment.fractionation import Fractionation, FractionationScheme, FractionationType

# Nhập các module từ machine
from quangtps.treatment.machine.accelerator import Accelerator
from quangtps.treatment.machine.linac import Linac
from quangtps.treatment.machine.machine_library import MachineLibrary
from quangtps.treatment.machine.machine_specs import MachineSpecification

# Nhập các module từ mlc
from quangtps.treatment.mlc.mlc_controller import MLCController
from quangtps.treatment.mlc.mlc_model import MLCModel
from quangtps.treatment.mlc.mlc_simulation import MLCSimulation
from quangtps.treatment.mlc.mlc_viewer import MLCViewer

# Nhập các kỹ thuật xạ trị
from quangtps.treatment.techniques.conformal import Conformal3DRT
from quangtps.treatment.techniques.imrt import IMRT
from quangtps.treatment.techniques.vmat import VMAT
from quangtps.treatment.techniques.dcat import DCAT
from quangtps.treatment.techniques.stereotactic import SRS, SBRT
from quangtps.treatment.techniques.tbi import TBI, TSI
from quangtps.treatment.techniques.proton import ProtonTherapy, PencilBeamScanning
from quangtps.treatment.techniques.carbon import CarbonIonTherapy
from quangtps.treatment.techniques.adaptive import AdaptiveRadiotherapy
from quangtps.treatment.techniques.electron import ElectronTherapy
from quangtps.treatment.techniques.igrt import IGRT
from quangtps.treatment.techniques.flash import FLASHRadiotherapy
from quangtps.treatment.techniques.bnct import BNCT

# Định nghĩa các thành phần được export
__all__ = [
    # Beam modules
    'Beam', 'BeamGeometry', 'BeamLibrary', 'BeamTemplate', 'BeamArrangementTemplate',
    'BeamModifier', 'Wedge', 'Block', 'Bolus', 'Compensator',
    
    # Fractionation modules
    'Fractionation', 'FractionationScheme', 'FractionationType',
    
    # Machine modules
    'Accelerator', 'Linac', 'MachineLibrary', 'MachineSpecification',
    
    # MLC modules
    'MLCController', 'MLCModel', 'MLCSimulation', 'MLCViewer',
    
    # Technique modules
    'Conformal3DRT', 'IMRT', 'VMAT', 'DCAT', 'SRS', 'SBRT',
    'TBI', 'TSI', 'ProtonTherapy', 'PencilBeamScanning', 'CarbonIonTherapy',
    'AdaptiveRadiotherapy', 'ElectronTherapy', 'IGRT', 'FLASHRadiotherapy', 'BNCT'
]