"""
Core package for QuangTPS.

This package contains the core models and functionality for the QuangTPS system.
"""

from quangtps.core.plan import Plan
from quangtps.core.structures import Structure, StructureType
from quangtps.core.beams import Beam, BeamType, MachineType
from quangtps.core.prescriptions import Prescription, PrescriptionType
from quangtps.core.logging import get_logger, setup_logging

__all__ = [
    'Plan',
    'Structure', 'StructureType',
    'Beam', 'BeamType', 'MachineType',
    'Prescription', 'PrescriptionType',
    'get_logger', 'setup_logging'
]

__version__ = '0.1.0'

from quangtps.core.config import Config
from quangtps.core.constants import Constants
from quangtps.core.exceptions import QuangTPSError, ValidationError, IOError
from quangtps.core.utils import Timer, get_memory_usage, create_unique_id
from quangtps.core.services import ServiceRegistry, ServiceBase, ServiceManager
