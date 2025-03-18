"""
Physics module for QuangTPS.

This module provides physics models and calculations for radiation transport
and dose deposition in radiotherapy treatment planning.
"""

from quangtps.physics.particle import Particle, ParticleType, ParticleHistory
from quangtps.physics.material import Material, MaterialProperties
from quangtps.physics.interaction import PhotonInteraction, ElectronInteraction
from quangtps.physics.source import RadiationSource, PhotonSource, ElectronSource
from quangtps.physics.boron import BoronCompoundType, BoronDistributionModel
from quangtps.physics.boron import BPAModel, BSHModel, BoronophenylalanineModel, GenericBoronModel

__all__ = [
    'Particle', 'ParticleType', 'ParticleHistory',
    'Material', 'MaterialProperties',
    'PhotonInteraction', 'ElectronInteraction',
    'RadiationSource', 'PhotonSource', 'ElectronSource',
    'BoronCompoundType', 'BoronDistributionModel',
    'BPAModel', 'BSHModel', 'BoronophenylalanineModel', 'GenericBoronModel'
]
