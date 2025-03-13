"""
Module chứa các thuật toán tính toán liều.

Module này cung cấp các lớp triển khai cụ thể cho các thuật toán tính toán liều khác nhau,
bao gồm CCC, Pencil Beam, AAA, và có thể mở rộng với các thuật toán khác như
Acuros XB, Convolution Superposition, Monte Carlo và GBBS.
"""

from quangtps.dose.algorithms.ccc import CollapsedConeImplementer
from quangtps.dose.algorithms.pencil_beam import PencilBeamImplementer
from quangtps.dose.algorithms.aaa import AAAImplementer

__all__ = [
    'CollapsedConeImplementer',
    'PencilBeamImplementer',
    'AAAImplementer'
]
