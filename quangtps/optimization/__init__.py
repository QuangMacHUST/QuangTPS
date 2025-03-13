"""
Module tối ưu hóa của hệ thống lập kế hoạch xạ trị QuangTPS.

Module này cung cấp các công cụ và thuật toán để tối ưu hóa kế hoạch xạ trị, bao gồm
định nghĩa các hàm mục tiêu, ràng buộc, động cơ tối ưu hóa và các thuật toán giải.

Các tính năng chính:
- Hàm mục tiêu và ràng buộc cho kế hoạch xạ trị
- Động cơ tối ưu hóa với các thuật toán khác nhau
- Tối ưu hóa dựa trên DVH và chỉ số lâm sàng
- Phương pháp tối ưu hóa hướng tri thức (KBP)
"""

# Import từ objectives.py
from quangtps.optimization.objectives import (
    ObjectiveBase,
    MinDose,
    MaxDose,
    UniformDose,
    MeanDose,
    DoseVolume,
    ConformityIndex,
    HomogeneityIndex,
    GradientIndex,
    EUDObjective,
    FalloffObjective,
    ObjectiveCollection,
    create_objective,
    OBJECTIVE_TYPES
)

# Import từ constraints.py
from quangtps.optimization.constraints import (
    ConstraintBase,
    MaxDoseConstraint,
    MinDoseConstraint,
    MeanDoseConstraint,
    DoseVolumeConstraint,
    HomogeneityConstraint,
    ConformityConstraint,
    ConstraintCollection,
    create_constraint,
    get_default_constraints_for_structure,
    get_organ_specific_constraints,
    CONSTRAINT_TYPES
)

# Import từ optimization_engine.py
from quangtps.optimization.optimization_engine import (
    OptimizationStatus,
    OptimizationEvent,
    OptimizationParameters,
    OptimizationResults,
    OptimizationEngine,
    create_engine
)

# Import từ solver.py
from quangtps.optimization.solver import (
    OptimizerBase,
    GradientDescentOptimizer,
    LBFGSOptimizer,
    SimulatedAnnealingOptimizer,
    create_optimizer,
    optimize_plan
)

# Thêm chuỗi phiên bản và thông tin tác giả
__version__ = '0.1.0'
__author__ = 'QuangTPS Team'

# Danh sách export
__all__ = [
    # Objectives
    'ObjectiveBase',
    'MinDose',
    'MaxDose',
    'UniformDose',
    'MeanDose',
    'DoseVolume',
    'ConformityIndex',
    'HomogeneityIndex',
    'GradientIndex',
    'EUDObjective',
    'FalloffObjective',
    'ObjectiveCollection',
    'create_objective',
    'OBJECTIVE_TYPES',
    
    # Constraints
    'ConstraintBase',
    'MaxDoseConstraint',
    'MinDoseConstraint',
    'MeanDoseConstraint',
    'DoseVolumeConstraint',
    'HomogeneityConstraint',
    'ConformityConstraint',
    'ConstraintCollection',
    'create_constraint',
    'get_default_constraints_for_structure',
    'get_organ_specific_constraints',
    'CONSTRAINT_TYPES',
    
    # Optimization Engine
    'OptimizationStatus',
    'OptimizationEvent',
    'OptimizationParameters',
    'OptimizationResults',
    'OptimizationEngine',
    'create_engine',
    
    # Solvers
    'OptimizerBase',
    'GradientDescentOptimizer',
    'LBFGSOptimizer',
    'SimulatedAnnealingOptimizer',
    'create_optimizer',
    'optimize_plan'
]
