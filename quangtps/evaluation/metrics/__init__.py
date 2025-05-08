#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Metrics Module for QuangTPS.

This module provides functionality for calculating and evaluating
metrics for radiotherapy treatment plans.
"""

from typing import Dict, List, Any, Optional, Union

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
