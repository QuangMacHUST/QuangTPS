"""
Module đảm bảo chất lượng (Quality Assurance - QA) cho hệ thống lập kế hoạch xạ trị QuangTPS.

Module này cung cấp các công cụ để thực hiện đảm bảo chất lượng cho các kế hoạch xạ trị,
bao gồm phân tích Gamma, so sánh liều và đánh giá hiệu suất kế hoạch.
"""

# Nhập các lớp và hàm từ các module con
from quangtps.evaluation.qa.gamma_analysis import (
    GammaParameters,
    GammaAnalysis,
    perform_gamma_analysis
)

from quangtps.evaluation.qa.dose_comparison import (
    ComparisonMetricType,
    DoseComparisonParameters,
    DoseComparison,
    compare_dose_distributions
)

# Nhập các chức năng đảm bảo chất lượng kế hoạch xạ trị từ treatment_qa
from quangtps.evaluation.qa.treatment_qa import (
    QATestType,
    QAProtocol,
    MetricResult,
    TreatmentQATest,
    TreatmentQAManager,
    TreatmentQAResult,
    TreatmentQAReport,
    PlanQualityMetrics,
    perform_treatment_qa,
    evaluate_plan_quality
)

__all__ = [
    # Từ gamma_analysis.py
    'GammaParameters',
    'GammaAnalysis',
    'perform_gamma_analysis',
    
    # Từ dose_comparison.py
    'ComparisonMetricType',
    'DoseComparisonParameters',
    'DoseComparison',
    'compare_dose_distributions',
    
    # Từ treatment_qa.py
    'QATestType',
    'QAProtocol',
    'MetricResult',
    'TreatmentQATest',
    'TreatmentQAManager',
    'TreatmentQAResult',
    'TreatmentQAReport',
    'PlanQualityMetrics',
    'perform_treatment_qa',
    'evaluate_plan_quality'
]