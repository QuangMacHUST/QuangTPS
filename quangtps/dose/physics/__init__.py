"""
Module chứa các mô hình vật lý liên quan đến tính toán liều.

Module này cung cấp các hàm và lớp để mô phỏng các hiện tượng vật lý trong quá trình 
tính toán liều xạ trị, bao gồm TERMA, tán xạ, hiệu chỉnh không đồng nhất, 
Linear Energy Transfer (LET), và Relative Biological Effectiveness (RBE).
"""

from quangtps.dose.physics.terma import calculate_terma, calculate_pdd, calculate_oar, get_beam_spectrum
from quangtps.dose.physics.scatter import calculate_scatter, calculate_scatter_kernel
from quangtps.dose.physics.heterogeneity import apply_heterogeneity_correction
from quangtps.dose.physics.let import calculate_let
from quangtps.dose.physics.rbe import calculate_rbe

__all__ = [
    'calculate_terma',
    'calculate_pdd',
    'calculate_oar',
    'get_beam_spectrum',
    'calculate_scatter',
    'calculate_scatter_kernel',
    'apply_heterogeneity_correction',
    'calculate_let',
    'calculate_rbe'
]
