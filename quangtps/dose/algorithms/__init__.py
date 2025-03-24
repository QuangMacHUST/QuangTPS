"""
Module containing dose calculation algorithms.

This module provides classes for calculating dose distributions
using various algorithms for radiotherapy treatment planning.
"""

from quangtps.dose.algorithms.pencil_beam import PencilBeamAlgorithm
from quangtps.dose.algorithms.collapsed_cone import CollapsedConeAlgorithm
from quangtps.dose.algorithms.monte_carlo import MonteCarloAlgorithm
from quangtps.dose.algorithms.aaa import AAAImplementer
from quangtps.dose.algorithms.acuros import AcurosXBImplementer

# Register available algorithms
AVAILABLE_ALGORITHMS = {
    "PENCIL_BEAM": PencilBeamAlgorithm,
    "COLLAPSED_CONE": CollapsedConeAlgorithm,
    "MONTE_CARLO": MonteCarloAlgorithm,
    "AAA": AAAImplementer,
    "ACUROS": AcurosXBImplementer
}
