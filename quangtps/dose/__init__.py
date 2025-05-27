"""
Module containing dose calculation functionality for QuangTPS.

This module provides classes and functions for calculating dose distributions
from radiation therapy beams.
"""

import logging

logger = logging.getLogger(__name__)


# Lazy import function to avoid circular dependencies
def _lazy_import():
    """Lazy import các thành phần dose calculation để tránh circular import"""
    try:
        from quangtps.dose.dose_calculator import (
            DoseAlgorithmBase,
            SimpleRayTracingAlgorithm,
            PencilBeamAlgorithm,
            CollapsedConeAlgorithm,
            MonteCarloAlgorithm,
            DoseCalculator,
        )

        return {
            "DoseCalculator": DoseCalculator,
            "DoseAlgorithmBase": DoseAlgorithmBase,
            "SimpleRayTracingAlgorithm": SimpleRayTracingAlgorithm,
            "PencilBeamAlgorithm": PencilBeamAlgorithm,
            "CollapsedConeAlgorithm": CollapsedConeAlgorithm,
            "MonteCarloAlgorithm": MonteCarloAlgorithm,
        }
    except ImportError as e:
        logger.warning(f"Cannot import dose calculator classes: {e}")
        return {}


# Import dose grid class
try:
    from quangtps.dose.dose_grid import DoseGrid
except ImportError as e:
    logger.warning(f"Cannot import DoseGrid: {e}")

    class DoseGrid:
        """Fallback DoseGrid class"""

        def __init__(self, *args, **kwargs):
            pass


# Import dose visualization
try:
    from quangtps.dose.dose_visualization import DoseColorwash as DoseVisualizer
except ImportError as e:
    logger.warning(f"Cannot import DoseVisualizer: {e}")

    class DoseVisualizer:
        """Fallback DoseVisualizer class"""

        def __init__(self, *args, **kwargs):
            pass


# Import dose constraints
try:
    from quangtps.dose.dose_constraints import (
        DoseVolumeConstraint as DoseConstraint,
        ConstraintType as DoseConstraintType,
    )
except ImportError as e:
    logger.warning(f"Cannot import dose constraints: {e}")

    class DoseConstraint:
        """Fallback DoseConstraint class"""

        def __init__(self, *args, **kwargs):
            pass

    class DoseConstraintType:
        """Fallback DoseConstraintType class"""

        def __init__(self, *args, **kwargs):
            pass


# Module-level getter functions để tránh circular import
def get_dose_calculator():
    """Lấy DoseCalculator class một cách an toàn"""
    classes = _lazy_import()
    return classes.get("DoseCalculator")


def get_dose_algorithm_base():
    """Lấy DoseAlgorithmBase class một cách an toàn"""
    classes = _lazy_import()
    return classes.get("DoseAlgorithmBase")


def get_available_algorithms():
    """Lấy danh sách các thuật toán khả dụng"""
    classes = _lazy_import()
    return [name for name in classes.keys() if "Algorithm" in name]


# Khởi tạo lazy loading cho compatibility
_dose_classes = {}


def __getattr__(name):
    """Dynamic attribute access for lazy loading"""
    if name in [
        "DoseCalculator",
        "DoseAlgorithmBase",
        "SimpleRayTracingAlgorithm",
        "PencilBeamAlgorithm",
        "CollapsedConeAlgorithm",
        "MonteCarloAlgorithm",
    ]:
        if name not in _dose_classes:
            classes = _lazy_import()
            _dose_classes.update(classes)
        return _dose_classes.get(name)

    # Return basic classes if not in dose calculator imports
    if name == "DoseGrid":
        return DoseGrid
    elif name == "DoseVisualizer":
        return DoseVisualizer
    elif name == "DoseConstraint":
        return DoseConstraint
    elif name == "DoseConstraintType":
        return DoseConstraintType

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


# Export chỉ những gì thực sự có sẵn
__all__ = [
    # Core functions
    "get_dose_calculator",
    "get_dose_algorithm_base",
    "get_available_algorithms",
    # Fallback classes
    "DoseGrid",
    "DoseVisualizer",
    "DoseConstraint",
    "DoseConstraintType",
]
