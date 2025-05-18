#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Các metrics đánh giá trong hệ thống lập kế hoạch QuangTPS.

Module này chứa các chỉ số và công cụ đánh giá khác nhau bao gồm
phân tích gamma, chỉ số đồng nhất, chỉ số đánh giá, và phân tích DVH.
"""

from typing import Dict, List, Any, Optional, Union, Tuple

# Import các lớp và hàm chính
try:
    from .clinical_metrics import calculate_clinical_metrics
except ImportError:
    pass

try:
    from .quality_metrics import calculate_quality_metrics
except ImportError:
    pass

try:
    from .radiobiological import calculate_radiobiological_metrics
except ImportError:
    pass

try:
    from .plan_metrics import (
        calculate_plan_metrics,
        compare_plan_metrics,
        analyze_plan_robustness,
        plot_plan_comparison,
        generate_plan_metrics_report,
    )
except ImportError:
    pass

# Import các module metrics
try:
    from quangtps.evaluation.metrics.gamma_analysis import (
        calculate_gamma_3d,
        calculate_gamma_2d,
        calculate_gamma_3d_gpu,
        gamma_pass_rate,
        get_gamma_statistics,
        analyze_gamma_by_dose_regions,
        plot_gamma_results,
    )

    HAS_GAMMA_MODULE = True

    # Kiểm tra khả năng sử dụng GPU
    try:
        import cupy as cp

        HAS_GAMMA_GPU = True
    except (ImportError, ModuleNotFoundError):
        HAS_GAMMA_GPU = False
except ImportError:
    HAS_GAMMA_MODULE = False
    HAS_GAMMA_GPU = False

__all__ = [
    "calculate_clinical_metrics",
    "calculate_quality_metrics",
    "calculate_radiobiological_metrics",
    "calculate_plan_metrics",
    "compare_plan_metrics",
    "analyze_plan_robustness",
    "plot_plan_comparison",
    "generate_plan_metrics_report",
]

if HAS_GAMMA_MODULE:
    __all__.extend(
        [
            "calculate_gamma_3d",
            "calculate_gamma_2d",
            "gamma_pass_rate",
            "get_gamma_statistics",
            "analyze_gamma_by_dose_regions",
            "plot_gamma_results",
        ]
    )

    # Thêm hàm GPU nếu có hỗ trợ
    if HAS_GAMMA_GPU:
        __all__.append("calculate_gamma_3d_gpu")


def get_available_metrics() -> List[str]:
    """
    Trả về danh sách các metrics đã đăng ký và sẵn sàng sử dụng.

    Returns
    -------
    List[str]
        Danh sách tên các metrics có sẵn
    """
    metrics = []

    if HAS_GAMMA_MODULE:
        metrics.append("gamma_analysis")
        if HAS_GAMMA_GPU:
            metrics.append("gamma_analysis_gpu")

    return metrics
