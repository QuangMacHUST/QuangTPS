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


# Tạo các lớp giả làm placeholder cho constraints
class ConstraintBase:
    def __init__(self, structure_name, **kwargs):
        self.structure_name = structure_name
        self.priority = kwargs.get("priority", 1)
        self.is_enabled = kwargs.get("is_enabled", True)
        self.constraint_type = kwargs.get("constraint_type", "None")
        self.is_hard_constraint = kwargs.get("is_hard_constraint", False)


class MaxDoseConstraint(ConstraintBase):
    def __init__(self, structure_name, dose_limit, **kwargs):
        super().__init__(structure_name, **kwargs)
        self.dose_limit = dose_limit
        self.constraint_type = "MaxDose"


class MinDoseConstraint(ConstraintBase):
    def __init__(self, structure_name, dose_limit, **kwargs):
        super().__init__(structure_name, **kwargs)
        self.dose_limit = dose_limit
        self.constraint_type = "MinDose"


class MeanDoseConstraint(ConstraintBase):
    def __init__(self, structure_name, dose_limit, **kwargs):
        super().__init__(structure_name, **kwargs)
        self.dose_limit = dose_limit
        self.constraint_type = "MeanDose"


class DoseVolumeConstraint(ConstraintBase):
    def __init__(self, structure_name, dose, volume_percent, **kwargs):
        super().__init__(structure_name, **kwargs)
        self.dose = dose
        self.volume_percent = volume_percent
        self.direction = kwargs.get("direction", "upper")
        self.constraint_type = "DoseVolume"


class HomogeneityConstraint(ConstraintBase):
    def __init__(self, structure_name, prescription_dose, **kwargs):
        super().__init__(structure_name, **kwargs)
        self.prescription_dose = prescription_dose
        self.max_hi = kwargs.get("max_hi", 0.15)
        self.constraint_type = "Homogeneity"


class ConformityConstraint(ConstraintBase):
    def __init__(self, structure_name, reference_dose, **kwargs):
        super().__init__(structure_name, **kwargs)
        self.reference_dose = reference_dose
        self.min_ci = kwargs.get("min_ci", 0.8)
        self.constraint_type = "Conformity"


class ConstraintCollection:
    def __init__(self):
        self.constraints = []

    def add_constraint(self, constraint):
        self.constraints.append(constraint)

    def remove_constraint(self, index):
        if 0 <= index < len(self.constraints):
            del self.constraints[index]

    def check_all(self, dose_grid, structures):
        return {"summary": {"violations_count": 0}}


# Hàm giả cho constraints
def create_constraint(constraint_type, **kwargs):
    CONSTRAINT_TYPES = {
        "MaxDose": MaxDoseConstraint,
        "MinDose": MinDoseConstraint,
        "MeanDose": MeanDoseConstraint,
        "DoseVolume": DoseVolumeConstraint,
        "Homogeneity": HomogeneityConstraint,
        "Conformity": ConformityConstraint,
    }
    return CONSTRAINT_TYPES[constraint_type](**kwargs)


def get_default_constraints_for_structure(
    structure_name, structure_type, prescription_dose=None
):
    return []


def get_organ_specific_constraints(structure_name, prescription_dose):
    return []


# Hằng số cho constraints
CONSTRAINT_TYPES = {
    "MaxDose": MaxDoseConstraint,
    "MinDose": MinDoseConstraint,
    "MeanDose": MeanDoseConstraint,
    "DoseVolume": DoseVolumeConstraint,
    "Homogeneity": HomogeneityConstraint,
    "Conformity": ConformityConstraint,
}

# Import từ objectives.py
try:
    from quangtps.optimization.objectives import (
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
        OBJECTIVE_TYPES,
    )
except (TypeError, ImportError):
    # Nếu có lỗi trong file objectives, bỏ qua
    pass

# Import thuật toán tối ưu hóa mới
try:
    from quangtps.optimization.gradient_descent import GradientDescent
    from quangtps.optimization.simulated_annealing import SimulatedAnnealing
    from quangtps.optimization.genetic_algorithm import GeneticAlgorithm
except (TypeError, ImportError) as e:
    import logging

    logging.getLogger(__name__).error(f"Error importing optimization algorithms: {e}")

try:
    # Import từ optimization_engine.py
    from quangtps.optimization.optimization_engine import (
        OptimizationStatus,
        OptimizationEvent,
        OptimizationParameters,
        OptimizationResults,
        OptimizationEngine,
        create_engine,
    )
except (TypeError, ImportError):
    # Nếu có lỗi trong file optimization_engine, bỏ qua
    pass

try:
    # Import từ solver.py
    from quangtps.optimization.solver import (
        OptimizerBase,
        GradientDescentOptimizer,
        LBFGSOptimizer,
        SimulatedAnnealingOptimizer,
        optimize_plan,
    )
except (TypeError, ImportError):
    # Nếu có lỗi trong file solver, bỏ qua
    pass

# Thêm chuỗi phiên bản và thông tin tác giả
__version__ = "0.1.0"
__author__ = "QuangTPS Team"

# Danh sách export
__all__ = [
    # Objectives
    "MinDose",
    "MaxDose",
    "UniformDose",
    "MeanDose",
    "DoseVolume",
    "ConformityIndex",
    "HomogeneityIndex",
    "GradientIndex",
    "EUDObjective",
    "FalloffObjective",
    "ObjectiveCollection",
    "create_objective",
    "OBJECTIVE_TYPES",
    # Constraints
    "ConstraintBase",
    "MaxDoseConstraint",
    "MinDoseConstraint",
    "MeanDoseConstraint",
    "DoseVolumeConstraint",
    "HomogeneityConstraint",
    "ConformityConstraint",
    "ConstraintCollection",
    "create_constraint",
    "get_default_constraints_for_structure",
    "get_organ_specific_constraints",
    "CONSTRAINT_TYPES",
    # Optimization Engine
    "OptimizationStatus",
    "OptimizationEvent",
    "OptimizationParameters",
    "OptimizationResults",
    "OptimizationEngine",
    "create_engine",
    # Solvers
    "OptimizerBase",
    "GradientDescentOptimizer",
    "LBFGSOptimizer",
    "SimulatedAnnealingOptimizer",
    "optimize_plan",
    # New Optimization Algorithms
    "GradientDescent",
    "SimulatedAnnealing",
    "GeneticAlgorithm",
]

# Import các lớp cơ sở từ các modules
from quangtps.optimization.base import OptimizerBase
from quangtps.optimization.optimizer import (
    PlanOptimizer,
    DoseFunctionBasedOptimizer,
    MCOOptimizer,
)
from quangtps.optimization.objectives import (
    ObjectiveType,
    Objective,
    ObjectiveFunction,
    DoseObjective,
    register_objective,
    get_objective_by_id,
)
from quangtps.optimization.constraints import (
    ConstraintType,
    ConstraintBase,
    DoseConstraint,
)
from quangtps.optimization.methods.vmat_optimization import (
    VMATOptimizer,
    VMATParameters,
    VMATOptimizationObjective,
    VMATConstraintType,
)


# Các hàm tiện ích
def create_optimizer(method_name="standard", **kwargs):
    """
    Tạo đối tượng optimizer dựa trên phương pháp được chỉ định.

    Args:
        method_name: Tên phương pháp tối ưu hóa ("standard", "mco", "vmat", ...)
        **kwargs: Các tham số bổ sung cho optimizer

    Returns:
        Đối tượng optimizer phù hợp với phương pháp
    """
    method_name = method_name.lower()

    if method_name == "standard":
        return PlanOptimizer(**kwargs)
    elif method_name == "dose_function":
        return DoseFunctionBasedOptimizer(**kwargs)
    elif method_name == "mco":
        return MCOOptimizer(**kwargs)
    elif method_name == "vmat":
        if "params" in kwargs:
            return VMATOptimizer(params=kwargs["params"])
        else:
            return VMATOptimizer()
    else:
        raise ValueError(f"Phương pháp tối ưu hóa không được hỗ trợ: {method_name}")


# Xuất các symbols chính
__all__ = [
    "OptimizerBase",
    "PlanOptimizer",
    "DoseFunctionBasedOptimizer",
    "MCOOptimizer",
    "VMATOptimizer",
    "VMATParameters",
    "ObjectiveType",
    "Objective",
    "ObjectiveFunction",
    "DoseObjective",
    "ConstraintType",
    "ConstraintBase",
    "DoseConstraint",
    "register_objective",
    "get_objective_by_id",
    "create_optimizer",
]
