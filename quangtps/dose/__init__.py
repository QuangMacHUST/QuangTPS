"""
Module containing dose calculation functionality for QuangTPS.

This module provides classes and functions for calculating dose distributions
from radiation therapy beams.
"""

# Import key dose calculation components
from quangtps.dose.dose_calculator import (
    DoseAlgorithmBase,
    SimpleRayTracingAlgorithm,
    PencilBeamAlgorithm,
    CollapsedConeAlgorithm,
    MonteCarloAlgorithm,
    DoseCalculator,
)

# Import dose grid class
from quangtps.dose.dose_grid import DoseGrid

# Import dose visualization
from quangtps.dose.dose_visualization import DoseVisualizer

# Import dose constraints
from quangtps.dose.dose_constraints import DoseConstraint, DoseConstraintType

__all__ = [
    # Dose calculator and algorithms
    "DoseCalculator",
    "DoseAlgorithmBase",
    "SimpleRayTracingAlgorithm",
    "PencilBeamAlgorithm",
    "CollapsedConeAlgorithm",
    "MonteCarloAlgorithm",
    # Dose grid
    "DoseGrid",
    # Dose visualization
    "DoseVisualizer",
    # Dose constraints
    "DoseConstraint",
    "DoseConstraintType",
]
