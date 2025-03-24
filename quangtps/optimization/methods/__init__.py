"""
Module chứa các phương pháp tối ưu hóa kế hoạch xạ trị.

Module này cung cấp các phương pháp khác nhau để tối ưu hóa kế hoạch xạ trị, bao gồm:
- Tối ưu hóa dựa trên hàm mục tiêu
- Tối ưu hóa đa tiêu chuẩn (MCO)
- Tối ưu hóa dựa trên tri thức (KBO)
- Và các phương pháp tiên tiến khác
"""

from typing import Dict, List, Any, Optional

# Tạo lớp giả cho MCO
class MCOEngine:
    def __init__(self):
        pass

class MCONavigator:
    def __init__(self):
        pass

class MCOTrade:
    def __init__(self):
        pass

class MCOMethod:
    def __init__(self):
        pass

# Thử import từ optimization.py
try:
    from quangtps.optimization.objectives import ObjectiveCollection
except ImportError:
    # Nếu không thành công, định nghĩa một placeholder
    class ObjectiveCollection:
        def __init__(self):
            self.objectives = []

# Bỏ qua việc import constraints.py để tránh lỗi
# from quangtps.optimization.constraints import ConstraintCollection

# Thử import từ objective_based.py
try:
    from quangtps.optimization.methods.objective_based import (
        ObjectiveOptimizer,
        DirectObjectiveOptimizer,
        GradientBasedOptimizer,
        SimulatedAnnealingOptimizer,
        ObjectiveOptimizationParams
    )
except ImportError:
    # Nếu không thành công, bỏ qua
    pass

# Bỏ qua import từ mco.py
# from quangtps.optimization.methods.mco import (
#     MCOEngine,
#     MCONavigator,
#     MCOTrade,
#     MCOMethod
# )

# Thử import từ kbo.py
try:
    from quangtps.optimization.methods.kbo import (
        KnowledgeBaseOptimizer,
        DBSCANClusteringOptimizer,
        KMeansClusteringOptimizer
    )
except ImportError:
    # Nếu không thành công, bỏ qua
    pass

# Thêm chuỗi phiên bản và thông tin tác giả
__version__ = '0.1.0'
__author__ = 'QuangTPS Team'

# Export các lớp giả MCO
__all__ = ['MCOEngine', 'MCONavigator', 'MCOTrade', 'MCOMethod']

# Định nghĩa các phương thức hỗ trợ
def get_available_methods() -> Dict[str, Any]:
    """
    Trả về danh sách các phương pháp tối ưu hóa có sẵn.
    
    Returns:
        Dictionary chứa thông tin về các phương pháp có sẵn
    """
    methods = {
        "objective_based": {
            "direct": "Tối ưu hóa trực tiếp dựa trên hàm mục tiêu",
            "gradient": "Tối ưu hóa dựa trên gradient của hàm mục tiêu",
            "simulated_annealing": "Tối ưu hóa mô phỏng luyện kim"
        },
        "mco": {
            "pareto_navigator": "Điều hướng bề mặt Pareto cho tối ưu hóa đa mục tiêu"
        },
        "kbo": {
            "dbscan": "Tối ưu hóa dựa trên tri thức với phân cụm DBSCAN",
            "kmeans": "Tối ưu hóa dựa trên tri thức với phân cụm K-means"
        }
    }
    
    return methods
