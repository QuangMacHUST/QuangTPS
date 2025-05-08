"""
Module chứa các phương pháp tối ưu hóa khác nhau cho QuangTPS.

Module này cung cấp các thuật toán tối ưu hóa khác nhau để sử dụng trong
hệ thống lập kế hoạch xạ trị, bao gồm cả các phương pháp tối ưu hóa bền vững.
"""

from typing import Dict, List, Optional, Any, Union, Tuple

try:
    from .objective_based import ObjectiveBasedOptimizer
except ImportError:
    pass

try:
    from .gradient_based import GradientBasedOptimizer
except ImportError:
    pass

try:
    from .robust_optimizer import (
        RobustOptimizer,
        UncertaintyScenario,
        create_robust_objective,
        optimize_robust_plan,
    )
except ImportError:
    pass

try:
    from .vmat_optimization import VMATOptimizer
except ImportError:
    pass

try:
    from .auto_planning import AutoPlanningEngine
except ImportError:
    pass

__all__ = [
    "ObjectiveBasedOptimizer",
    "GradientBasedOptimizer",
    "RobustOptimizer",
    "UncertaintyScenario",
    "create_robust_objective",
    "optimize_robust_plan",
    "VMATOptimizer",
    "AutoPlanningEngine",
]
