#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module xử lý tối ưu hóa kế hoạch thích ứng trong QuangTPS.

Module này cung cấp các lớp và hàm để tối ưu hóa kế hoạch thích ứng dựa trên
sự thay đổi giải phẫu của bệnh nhân theo thời gian, đảm bảo kế hoạch xạ trị
luôn phù hợp với giải phẫu hiện tại của bệnh nhân.
"""

from quangtps.adaptive.optimization.real_time_adaptive_planning import (
    RealTimeAdaptivePlanner,
    RealTimeAdaptiveSession,
    AdaptationPriority,
    AdaptationStatus,
    create_real_time_adaptive_planner,
)

from quangtps.adaptive.optimization.anatomy_prediction_integration import (
    AnatomyPredictionIntegrator,
    AdaptationStrategy,
    PredictionConfidenceLevel,
)

__all__ = [
    "RealTimeAdaptivePlanner",
    "RealTimeAdaptiveSession",
    "AdaptationPriority",
    "AdaptationStatus",
    "create_real_time_adaptive_planner",
    "AnatomyPredictionIntegrator",
    "AdaptationStrategy",
    "PredictionConfidenceLevel",
]
