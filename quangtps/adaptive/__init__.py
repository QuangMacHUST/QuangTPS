"""
Module chứa các chức năng điều chỉnh kế hoạch xạ trị thích ứng.
Hỗ trợ cả xạ trị thích ứng trực tuyến (online) và ngoại tuyến (offline).
"""

from enum import Enum, auto
import logging

logger = logging.getLogger(__name__)

class AdaptiveStrategy(Enum):
    """Các chiến lược thích ứng khác nhau được hỗ trợ trong hệ thống"""
    ADAPT_TO_POSITION = auto()  # Điều chỉnh vị trí isocenter
    ADAPT_TO_SHAPE = auto()     # Tối ưu lại kế hoạch dựa trên thay đổi hình dạng
    PLAN_LIBRARY = auto()       # Sử dụng thư viện kế hoạch đã chuẩn bị trước
    DOSE_TRACKING = auto()      # Theo dõi liều tích lũy và điều chỉnh nếu cần

from .adaptive_planning import AdaptivePlanner, PlanAdaptationSession
from .dose_accumulation import DoseAccumulator, AccumulatedDose
from .deformation import DeformableRegistration, RigidRegistration
from .four_d import FourDHandler, RespiratoryMotionModel
from .setup_error import SetupErrorEstimator, SetupCorrectionStrategy
from .temporal_analysis import TemporalChangeDetector, AnatomicalChangeMetrics

__all__ = [
    'AdaptiveStrategy',
    'AdaptivePlanner',
    'PlanAdaptationSession',
    'DoseAccumulator',
    'AccumulatedDose',
    'DeformableRegistration',
    'RigidRegistration',
    'FourDHandler',
    'RespiratoryMotionModel',
    'SetupErrorEstimator',
    'SetupCorrectionStrategy',
    'TemporalChangeDetector',
    'AnatomicalChangeMetrics',
]
