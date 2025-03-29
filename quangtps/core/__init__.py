"""
Module cốt lõi của QuangTPS.
Chứa các thành phần cơ bản và tiện ích dùng chung cho toàn bộ hệ thống.
"""

__version__ = '0.1.0'

from quangtps.core.config import Config
from quangtps.core.constants import Constants
from quangtps.core.logging import setup_logger, get_logger
from quangtps.core.exceptions import QuangTPSError, ValidationError, IOError
from quangtps.core.utils import Timer, get_memory_usage, create_unique_id
from quangtps.core.structures import Structure, StructureSet, StructureType
from quangtps.core.services import ServiceRegistry, ServiceBase, ServiceManager
