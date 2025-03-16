"""
Module đánh giá DVH (Dose Volume Histogram) cho hệ thống lập kế hoạch xạ trị QuangTPS.

Module này cung cấp các công cụ để tính toán, phân tích và hiển thị Biểu đồ Liều-Thể tích 
(DVH) từ dữ liệu phân bố liều và cấu trúc của bệnh nhân, giúp đánh giá chất lượng kế hoạch xạ trị.

Các tính năng chính:
- Tính toán DVH tích lũy và vi phân
- Phân tích DVH với các chỉ số đánh giá tiêu chuẩn (D95, V95, HI, CI, ...)
- Vẽ biểu đồ DVH đẹp và tùy chỉnh linh hoạt
- So sánh nhiều DVH để đánh giá các kế hoạch xạ trị khác nhau
- Xuất dữ liệu ra nhiều định dạng (CSV, PNG, PDF)
"""

# Import từ dvh_calculator
from quangtps.evaluation.dvh.dvh_calculator import (
    DVHPoint,
    DVHData,
    DVHCalculator,
    DVHMetrics
)

# Import từ dvh_calculation
from quangtps.evaluation.dvh.dvh_calculation import (
    calculate_dvh,
    _get_dose_at_volume,
    _get_volume_at_dose,
    calculate_dvh_metrics,
    calculate_dvh_from_dose_grid
)

from quangtps.evaluation.dvh.dvh_analysis import DVHAnalysis

from quangtps.evaluation.dvh.dvh_visualization import (
    plot_dvh,
    plot_multiple_dvh,
    create_dvh_report,
    plot_dvh_bands,
    export_dvh_to_csv
)

__all__ = [
    # Từ dvh_calculator.py
    'DVHPoint',
    'DVHData',
    'DVHCalculator',
    'DVHMetrics',
    
    # Từ dvh_calculation.py
    'calculate_dvh',
    'calculate_dvh_metrics',
    'calculate_dvh_from_dose_grid',
    
    # Từ dvh_analysis.py
    'DVHAnalysis',
    
    # Từ dvh_visualization.py
    'plot_dvh',
    'plot_multiple_dvh',
    'create_dvh_report',
    'plot_dvh_bands',
    'export_dvh_to_csv',
]
