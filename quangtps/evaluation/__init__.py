"""
Trình đánh giá kế hoạch điều trị phóng xạ.

Module này cung cấp các công cụ để đánh giá kế hoạch điều trị, bao gồm:
- Phân tích và hiển thị phân bố liều
- Tính toán và vẽ Biểu đồ Thể tích-Liều (DVH)
- Tính toán chỉ số đánh giá kế hoạch như CI, HI, GI
- Mô hình tác động sinh học: TCP, NTCP, EQD2
- Tạo báo cáo đánh giá kế hoạch
- So sánh kế hoạch xạ trị
"""

from quangtps.evaluation.dose_analysis import DoseAnalysis
from quangtps.evaluation.biological.tcp import (
    calculate_tcp_lq_poisson,
    calculate_tcp_lq_poisson_dvh,
    calculate_tcp_niemierko,
    calculate_tcp_logistic,
    calculate_tcp_webb
)
from quangtps.evaluation.biological.ntcp import (
    calculate_ntcp_lkb,
    calculate_ntcp_relative_seriality,
    calculate_ntcp_logit,
    calculate_ntcp_poisson,
    calculate_ntcp_for_dvh,
    get_ntcp_constraints
)
from quangtps.evaluation.biological.eqd2 import (
    calculate_eqd2,
    calculate_bed,
    calculate_eqd2_for_volume,
    get_alpha_beta_ratio
)
from quangtps.evaluation.evaluation_report import EvaluationReport
from quangtps.evaluation.plan_comparison import PlanComparison

# Import từ module DVH
from quangtps.evaluation.dvh import (
    calculate_dvh,
    calculate_dvh_metrics,
    calculate_dvh_from_dose_grid,
    plot_dvh,
    plot_multiple_dvh,
    create_dvh_report
)

# Import plan evaluation module
from quangtps.evaluation.plan_evaluation import (
    PlanEvaluation,
    DVHCalculator,
    evaluate_plan
)

# Import plan quality module
from quangtps.evaluation.plan_quality import (
    ClinicalGoal,
    PlanQualityEvaluator
)

# Import clinical protocols module
from quangtps.evaluation.clinical_protocols import (
    ClinicalProtocolManager,
    get_protocol,
    select_protocol_dialog
)

# Import robustness evaluation if available
try:
    from .robustness import (
        RobustnessAnalyzer, RobustnessResult, ScenarioResult,
        analyze_plan_robustness, RobustOptimizer, optimize_robust_plan
    )
except ImportError:
    pass

__all__ = [
    # Phân tích liều
    'DoseAnalysis',
    
    # TCP (Tumor Control Probability)
    'calculate_tcp_lq_poisson',
    'calculate_tcp_lq_poisson_dvh',
    'calculate_tcp_niemierko',
    'calculate_tcp_logistic',
    'calculate_tcp_webb',
    
    # NTCP (Normal Tissue Complication Probability)
    'calculate_ntcp_lkb',
    'calculate_ntcp_relative_seriality',
    'calculate_ntcp_logit',
    'calculate_ntcp_poisson',
    'calculate_ntcp_for_dvh',
    'get_ntcp_constraints',
    
    # EQD2 (Equivalent Dose in 2 Gy fractions)
    'calculate_eqd2',
    'calculate_bed',
    'calculate_eqd2_for_volume',
    'get_alpha_beta_ratio',
    
    # DVH (Dose Volume Histogram)
    'calculate_dvh',
    'calculate_dvh_metrics',
    'calculate_dvh_from_dose_grid',
    'plot_dvh',
    'plot_multiple_dvh',
    'create_dvh_report',
    
    # Đánh giá kế hoạch
    'PlanEvaluation',
    'DVHCalculator',
    'evaluate_plan',
    
    # Đánh giá chất lượng kế hoạch
    'ClinicalGoal',
    'PlanQualityEvaluator',
    'ClinicalProtocolManager',
    'get_protocol',
    'select_protocol_dialog',
    
    # Đánh giá và so sánh kế hoạch
    'EvaluationReport',
    'PlanComparison',
    'RobustnessAnalyzer', 'RobustnessResult', 'analyze_plan_robustness'
]
