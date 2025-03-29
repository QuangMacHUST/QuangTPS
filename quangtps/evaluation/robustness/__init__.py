"""
Robustness evaluation module for QuangTPS treatment planning system.

This module provides tools for evaluating and optimizing treatment plans
considering setup and range uncertainties to ensure robust treatment delivery.
"""

from typing import Dict, List, Optional, Any, Union, Tuple

# Import robustness analyzer if it exists
try:
    from .robustness_analyzer import (
        RobustnessAnalyzer, RobustnessResult, ScenarioResult,
        UncertaintyType, analyze_plan_robustness
    )
except ImportError:
    pass

# Import robust optimizer if it exists
try:
    from .robust_optimizer import (
        RobustOptimizer, optimize_robust_plan, create_robust_objective
    )
except ImportError:
    pass

__all__ = [
    'RobustnessAnalyzer',
    'RobustnessResult',
    'ScenarioResult',
    'UncertaintyType',
    'analyze_plan_robustness',
    'RobustOptimizer', 
    'optimize_robust_plan',
    'create_robust_objective'
] 