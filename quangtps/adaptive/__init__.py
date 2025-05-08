#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module thích ứng của QuangTPS.

Module này cung cấp các chức năng để phân tích, dự đoán và thích ứng kế hoạch
xạ trị theo thời gian.
"""

from enum import Enum, auto
from typing import List, Dict, Tuple, Optional, Union, Any
import logging

logger = logging.getLogger(__name__)


class AdaptiveStrategy(Enum):
    """Các chiến lược thích ứng khác nhau được hỗ trợ trong hệ thống"""

    ADAPT_TO_POSITION = auto()  # Điều chỉnh vị trí isocenter
    ADAPT_TO_SHAPE = auto()  # Tối ưu lại kế hoạch dựa trên thay đổi hình dạng
    PLAN_LIBRARY = auto()  # Sử dụng thư viện kế hoạch đã chuẩn bị trước
    DOSE_TRACKING = auto()  # Theo dõi liều tích lũy và điều chỉnh nếu cần
    ROBUST_ADAPTATION = auto()  # Lập kế hoạch thích ứng bền vững


from quangtps.adaptive.adaptive_planning import (
    AdaptivePlan,
    AdaptivePlanner,
    create_adaptive_plan,
)
from quangtps.adaptive.dose_accumulation import DoseAccumulator, AccumulatedDose
from quangtps.adaptive.deformation import DeformableRegistration, RigidRegistration
from quangtps.adaptive.four_d import FourDHandler, RespiratoryMotionModel
from quangtps.adaptive.setup_error import SetupErrorEstimator, SetupCorrectionStrategy
from quangtps.adaptive.temporal_analysis import (
    TemporalAnalyzer,
    TemporalAnalysisResult,
    analyze_temporal_changes,
)
from quangtps.adaptive.prediction import (
    AnatomyPrediction,
    AnatomyPredictor,
    PredictionMethod,
    predict_anatomy_changes,
)
from quangtps.adaptive.robust_adaptive_planning import (
    AdaptationTrigger,
    AdaptationType,
    RobustAdaptivePlan,
    RobustAdaptivePlanner,
)

__all__ = [
    "AdaptiveStrategy",
    "AdaptivePlan",
    "AdaptivePlanner",
    "create_adaptive_plan",
    "DoseAccumulator",
    "AccumulatedDose",
    "DeformableRegistration",
    "RigidRegistration",
    "FourDHandler",
    "RespiratoryMotionModel",
    "SetupErrorEstimator",
    "SetupCorrectionStrategy",
    "TemporalAnalyzer",
    "TemporalAnalysisResult",
    "analyze_temporal_changes",
    "AnatomyPrediction",
    "AnatomyPredictor",
    "PredictionMethod",
    "predict_anatomy_changes",
    "AdaptationTrigger",
    "AdaptationType",
    "RobustAdaptivePlan",
    "RobustAdaptivePlanner",
]
