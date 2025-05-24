#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module quản lý tối ưu hóa kế hoạch xạ trị (Optimization).

Module này cung cấp các lớp và phương thức để định nghĩa và quản lý các thiết lập tối ưu hóa,
mục tiêu tối ưu hóa, và ràng buộc tối ưu hóa cho kế hoạch xạ trị.
"""

import logging
import uuid
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Tuple

logger = logging.getLogger(__name__)


class OptimizationType(str, Enum):
    """Enum cho các loại tối ưu hóa."""

    FLUENCE = "Fluence"
    APERTURE = "Aperture"
    DIRECT_MACHINE_PARAMETER = "DirectMachineParameter"
    MONTE_CARLO = "MonteCarlo"
    HYBRID = "Hybrid"
    MULTI_CRITERIA = "MultiCriteria"


class OptimizationAlgorithm(str, Enum):
    """Enum cho các thuật toán tối ưu hóa."""

    GRADIENT_DESCENT = "GradientDescent"  # Thuật toán gradient descent
    SIMULATED_ANNEALING = "SimulatedAnnealing"  # Thuật toán mô phỏng luyện kim
    GENETIC = "Genetic"  # Thuật toán di truyền
    PARTICLE_SWARM = "ParticleSwarm"  # Thuật toán đàn bầy
    NEWTON = "Newton"  # Phương pháp Newton
    QUASI_NEWTON = "QuasiNewton"  # Phương pháp Quasi-Newton
    SEQUENTIAL = "Sequential"  # Quy hoạch tuần tự
    BRANCH_AND_BOUND = "BranchAndBound"  # Nhánh và cận
    MIXED_INTEGER = "MixedInteger"  # Số nguyên hỗn hợp
    COLUMN_GENERATION = "ColumnGeneration"  # Tạo cột


class OptimizationObjectiveType(str, Enum):
    """Enum cho các loại mục tiêu tối ưu hóa."""

    MIN_DOSE = "MinDose"  # Liều tối thiểu
    MAX_DOSE = "MaxDose"  # Liều tối đa
    UNIFORM_DOSE = "UniformDose"  # Liều đồng đều
    MEAN_DOSE = "MeanDose"  # Liều trung bình
    EUD = "EUD"  # Liều tương đương đồng đều (Equivalent Uniform Dose)
    DVH = "DVH"  # Dose Volume Histogram
    CONFORMITY = "Conformity"  # Độ phù hợp
    HOMOGENEITY = "Homogeneity"  # Độ đồng đều
    GRADIENT = "Gradient"  # Độ dốc (gradient)


class OptimizationConstraintType(str, Enum):
    """Enum cho các loại ràng buộc tối ưu hóa."""

    MIN_DOSE = "MinDose"  # Liều tối thiểu
    MAX_DOSE = "MaxDose"  # Liều tối đa
    MEAN_DOSE = "MeanDose"  # Liều trung bình
    VOLUME_AT_DOSE = "VolumeAtDose"  # Thể tích tại liều
    DOSE_AT_VOLUME = "DoseAtVolume"  # Liều tại thể tích
    MAX_EUD = "MaxEUD"  # EUD tối đa
    MIN_EUD = "MinEUD"  # EUD tối thiểu


class OptimizationObjective:
    """
    Lớp đại diện cho một mục tiêu tối ưu hóa.

    Lớp này chứa thông tin về một mục tiêu tối ưu hóa, bao gồm loại mục tiêu,
    cấu trúc liên quan, giá trị mục tiêu và trọng số.
    """

    def __init__(
        self,
        structure_id: str,
        objective_type: OptimizationObjectiveType,
        dose_value: float,
        weight: float = 1.0,
        volume_value: Optional[float] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ):
        """
        Khởi tạo một mục tiêu tối ưu hóa.

        Parameters
        ----------
        structure_id : str
            ID của cấu trúc liên quan đến mục tiêu
        objective_type : OptimizationObjectiveType
            Loại mục tiêu tối ưu hóa
        dose_value : float
            Giá trị liều mục tiêu (Gy)
        weight : float, optional
            Trọng số của mục tiêu (default = 1.0)
        volume_value : float, optional
            Giá trị thể tích (%) cho các mục tiêu liên quan đến DVH
        parameters : Dict[str, Any], optional
            Các tham số bổ sung cho mục tiêu
        """
        self.obj_id = str(uuid.uuid4())
        self.structure_id = structure_id
        self.objective_type = objective_type
        self.dose_value = dose_value
        self.weight = weight
        self.volume_value = volume_value
        self.parameters = parameters or {}

    def set_weight(self, weight: float):
        """
        Đặt trọng số cho mục tiêu tối ưu hóa.

        Parameters
        ----------
        weight : float
            Trọng số mới
        """
        self.weight = weight

    def to_dict(self) -> Dict[str, Any]:
        """
        Chuyển đổi đối tượng mục tiêu tối ưu hóa thành dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary chứa thông tin mục tiêu tối ưu hóa
        """
        return {
            "obj_id": self.obj_id,
            "structure_id": self.structure_id,
            "objective_type": self.objective_type.value,
            "dose_value": self.dose_value,
            "weight": self.weight,
            "volume_value": self.volume_value,
            "parameters": self.parameters,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OptimizationObjective":
        """
        Tạo đối tượng OptimizationObjective từ dictionary.

        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary chứa thông tin mục tiêu tối ưu hóa

        Returns
        -------
        OptimizationObjective
            Đối tượng OptimizationObjective được tạo từ dữ liệu
        """
        obj = cls(
            structure_id=data["structure_id"],
            objective_type=OptimizationObjectiveType(data["objective_type"]),
            dose_value=data["dose_value"],
            weight=data.get("weight", 1.0),
            volume_value=data.get("volume_value"),
        )

        obj.obj_id = data.get("obj_id", str(uuid.uuid4()))
        obj.parameters = data.get("parameters", {})

        return obj

    def __str__(self) -> str:
        """Biểu diễn chuỗi của mục tiêu tối ưu hóa."""
        result = (
            f"{self.objective_type.value} for {self.structure_id}: {self.dose_value} Gy"
        )
        if self.volume_value is not None:
            result += f" at {self.volume_value}%"
        result += f" (weight: {self.weight})"
        return result


class OptimizationConstraint:
    """
    Lớp đại diện cho một ràng buộc tối ưu hóa.

    Lớp này chứa thông tin về một ràng buộc tối ưu hóa, bao gồm loại ràng buộc,
    cấu trúc liên quan, giá trị ràng buộc và mức độ ưu tiên.
    """

    def __init__(
        self,
        structure_id: str,
        constraint_type: OptimizationConstraintType,
        dose_value: float,
        priority: int = 1,
        volume_value: Optional[float] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ):
        """
        Khởi tạo một ràng buộc tối ưu hóa.

        Parameters
        ----------
        structure_id : str
            ID của cấu trúc liên quan đến ràng buộc
        constraint_type : OptimizationConstraintType
            Loại ràng buộc tối ưu hóa
        dose_value : float
            Giá trị liều ràng buộc (Gy)
        priority : int, optional
            Mức độ ưu tiên của ràng buộc (default = 1)
        volume_value : float, optional
            Giá trị thể tích (%) cho các ràng buộc liên quan đến DVH
        parameters : Dict[str, Any], optional
            Các tham số bổ sung cho ràng buộc
        """
        self.constraint_id = str(uuid.uuid4())
        self.structure_id = structure_id
        self.constraint_type = constraint_type
        self.dose_value = dose_value
        self.priority = priority
        self.volume_value = volume_value
        self.parameters = parameters or {}

    def set_priority(self, priority: int):
        """
        Đặt mức độ ưu tiên cho ràng buộc tối ưu hóa.

        Parameters
        ----------
        priority : int
            Mức độ ưu tiên mới
        """
        self.priority = priority

    def to_dict(self) -> Dict[str, Any]:
        """
        Chuyển đổi đối tượng ràng buộc tối ưu hóa thành dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary chứa thông tin ràng buộc tối ưu hóa
        """
        return {
            "constraint_id": self.constraint_id,
            "structure_id": self.structure_id,
            "constraint_type": self.constraint_type.value,
            "dose_value": self.dose_value,
            "priority": self.priority,
            "volume_value": self.volume_value,
            "parameters": self.parameters,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OptimizationConstraint":
        """
        Tạo đối tượng OptimizationConstraint từ dictionary.

        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary chứa thông tin ràng buộc tối ưu hóa

        Returns
        -------
        OptimizationConstraint
            Đối tượng OptimizationConstraint được tạo từ dữ liệu
        """
        constraint = cls(
            structure_id=data["structure_id"],
            constraint_type=OptimizationConstraintType(data["constraint_type"]),
            dose_value=data["dose_value"],
            priority=data.get("priority", 1),
            volume_value=data.get("volume_value"),
        )

        constraint.constraint_id = data.get("constraint_id", str(uuid.uuid4()))
        constraint.parameters = data.get("parameters", {})

        return constraint

    def __str__(self) -> str:
        """Biểu diễn chuỗi của ràng buộc tối ưu hóa."""
        result = f"{self.constraint_type.value} for {self.structure_id}: {self.dose_value} Gy"
        if self.volume_value is not None:
            result += f" at {self.volume_value}%"
        result += f" (priority: {self.priority})"
        return result


class OptimizationSettings:
    """
    Lớp đại diện cho thiết lập tối ưu hóa kế hoạch xạ trị.

    Lớp này chứa thông tin về thiết lập tối ưu hóa, bao gồm loại tối ưu hóa,
    mục tiêu, ràng buộc, và các tham số tối ưu hóa khác.
    """

    def __init__(
        self,
        optimization_type: OptimizationType = OptimizationType.FLUENCE,
        max_iterations: int = 100,
        convergence_threshold: float = 0.001,
        parameters: Optional[Dict[str, Any]] = None,
    ):
        """
        Khởi tạo thiết lập tối ưu hóa.

        Parameters
        ----------
        optimization_type : OptimizationType, optional
            Loại tối ưu hóa
        max_iterations : int, optional
            Số lần lặp tối đa
        convergence_threshold : float, optional
            Ngưỡng hội tụ
        parameters : Dict[str, Any], optional
            Các tham số bổ sung cho tối ưu hóa
        """
        self.optimization_type = optimization_type
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.parameters = parameters or {}

        self.objectives = []
        self.constraints = []

        # Các thông số phân tích
        self.current_iteration = 0
        self.current_value = 0.0
        self.history = []

    def add_objective(self, objective: OptimizationObjective):
        """
        Thêm mục tiêu tối ưu hóa.

        Parameters
        ----------
        objective : OptimizationObjective
            Đối tượng mục tiêu tối ưu hóa
        """
        self.objectives.append(objective)

    def add_constraint(self, constraint: OptimizationConstraint):
        """
        Thêm ràng buộc tối ưu hóa.

        Parameters
        ----------
        constraint : OptimizationConstraint
            Đối tượng ràng buộc tối ưu hóa
        """
        self.constraints.append(constraint)

    def remove_objective(self, objective_id: str) -> bool:
        """
        Xóa mục tiêu tối ưu hóa.

        Parameters
        ----------
        objective_id : str
            ID của mục tiêu cần xóa

        Returns
        -------
        bool
            True nếu xóa thành công, False nếu không tìm thấy
        """
        for i, obj in enumerate(self.objectives):
            if obj.obj_id == objective_id:
                self.objectives.pop(i)
                return True
        return False

    def remove_constraint(self, constraint_id: str) -> bool:
        """
        Xóa ràng buộc tối ưu hóa.

        Parameters
        ----------
        constraint_id : str
            ID của ràng buộc cần xóa

        Returns
        -------
        bool
            True nếu xóa thành công, False nếu không tìm thấy
        """
        for i, constraint in enumerate(self.constraints):
            if constraint.constraint_id == constraint_id:
                self.constraints.pop(i)
                return True
        return False

    def get_objectives_for_structure(
        self, structure_id: str
    ) -> List[OptimizationObjective]:
        """
        Lấy danh sách các mục tiêu tối ưu hóa cho một cấu trúc cụ thể.

        Parameters
        ----------
        structure_id : str
            ID của cấu trúc

        Returns
        -------
        List[OptimizationObjective]
            Danh sách các mục tiêu tối ưu hóa
        """
        return [obj for obj in self.objectives if obj.structure_id == structure_id]

    def get_constraints_for_structure(
        self, structure_id: str
    ) -> List[OptimizationConstraint]:
        """
        Lấy danh sách các ràng buộc tối ưu hóa cho một cấu trúc cụ thể.

        Parameters
        ----------
        structure_id : str
            ID của cấu trúc

        Returns
        -------
        List[OptimizationConstraint]
            Danh sách các ràng buộc tối ưu hóa
        """
        return [
            constraint
            for constraint in self.constraints
            if constraint.structure_id == structure_id
        ]

    def reset_optimization(self):
        """Đặt lại trạng thái tối ưu hóa."""
        self.current_iteration = 0
        self.current_value = 0.0
        self.history = []

    def to_dict(self) -> Dict[str, Any]:
        """
        Chuyển đổi đối tượng thiết lập tối ưu hóa thành dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary chứa thông tin thiết lập tối ưu hóa
        """
        return {
            "optimization_type": self.optimization_type.value,
            "max_iterations": self.max_iterations,
            "convergence_threshold": self.convergence_threshold,
            "parameters": self.parameters,
            "objectives": [obj.to_dict() for obj in self.objectives],
            "constraints": [constraint.to_dict() for constraint in self.constraints],
            "current_iteration": self.current_iteration,
            "current_value": self.current_value,
            "history": self.history,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OptimizationSettings":
        """
        Tạo đối tượng OptimizationSettings từ dictionary.

        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary chứa thông tin thiết lập tối ưu hóa

        Returns
        -------
        OptimizationSettings
            Đối tượng OptimizationSettings được tạo từ dữ liệu
        """
        settings = cls(
            optimization_type=OptimizationType(
                data.get("optimization_type", OptimizationType.FLUENCE.value)
            ),
            max_iterations=data.get("max_iterations", 100),
            convergence_threshold=data.get("convergence_threshold", 0.001),
            parameters=data.get("parameters", {}),
        )

        # Tạo các mục tiêu
        if "objectives" in data:
            for obj_data in data["objectives"]:
                settings.objectives.append(OptimizationObjective.from_dict(obj_data))

        # Tạo các ràng buộc
        if "constraints" in data:
            for constraint_data in data["constraints"]:
                settings.constraints.append(
                    OptimizationConstraint.from_dict(constraint_data)
                )

        # Cập nhật trạng thái tối ưu hóa
        settings.current_iteration = data.get("current_iteration", 0)
        settings.current_value = data.get("current_value", 0.0)
        settings.history = data.get("history", [])

        return settings


class PlanOptimizer:
    """
    Lớp chính để thực hiện tối ưu hóa kế hoạch xạ trị.

    Lớp này tích hợp các thuật toán tối ưu hóa khác nhau và quản lý
    quá trình tối ưu hóa kế hoạch điều trị.
    """

    def __init__(
        self, algorithm: OptimizationAlgorithm = OptimizationAlgorithm.GRADIENT_DESCENT
    ):
        """
        Khởi tạo PlanOptimizer.

        Parameters
        ----------
        algorithm : OptimizationAlgorithm
            Thuật toán tối ưu hóa sử dụng
        """
        self.algorithm = algorithm
        self.settings = OptimizationSettings()
        self.is_running = False
        self.current_iteration = 0
        self.convergence_history = []

        logger.info(f"Khởi tạo PlanOptimizer với thuật toán {algorithm.value}")

    def set_optimization_settings(self, settings: OptimizationSettings):
        """
        Đặt cài đặt tối ưu hóa.

        Parameters
        ----------
        settings : OptimizationSettings
            Cài đặt tối ưu hóa
        """
        self.settings = settings
        logger.info("Đã cập nhật cài đặt tối ưu hóa")

    def optimize_plan(self, plan_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Thực hiện tối ưu hóa kế hoạch.

        Parameters
        ----------
        plan_data : Dict[str, Any]
            Dữ liệu kế hoạch cần tối ưu hóa

        Returns
        -------
        Dict[str, Any]
            Kết quả tối ưu hóa
        """
        try:
            self.is_running = True
            self.current_iteration = 0
            self.convergence_history = []

            logger.info(
                f"Bắt đầu tối ưu hóa kế hoạch với {len(self.settings.objectives)} mục tiêu"
            )

            # Giả lập quá trình tối ưu hóa
            for iteration in range(self.settings.max_iterations):
                self.current_iteration = iteration + 1

                # Tính toán objective function (giả lập)
                objective_value = self._calculate_objective_function(plan_data)
                self.convergence_history.append(objective_value)

                # Kiểm tra hội tụ
                if self._check_convergence():
                    logger.info(
                        f"Tối ưu hóa hội tụ tại iteration {self.current_iteration}"
                    )
                    break

                # Cập nhật parameters (giả lập)
                plan_data = self._update_plan_parameters(plan_data)

            self.is_running = False

            result = {
                "optimized_plan": plan_data,
                "iterations": self.current_iteration,
                "final_objective": self.convergence_history[-1]
                if self.convergence_history
                else 0,
                "convergence_history": self.convergence_history,
                "status": "converged"
                if self._check_convergence()
                else "max_iterations_reached",
            }

            logger.info(
                f"Hoàn thành tối ưu hóa sau {self.current_iteration} iterations"
            )
            return result

        except Exception as e:
            self.is_running = False
            logger.error(f"Lỗi trong quá trình tối ưu hóa: {e}")
            raise

    def _calculate_objective_function(self, plan_data: Dict[str, Any]) -> float:
        """
        Tính toán giá trị hàm mục tiêu.

        Parameters
        ----------
        plan_data : Dict[str, Any]
            Dữ liệu kế hoạch

        Returns
        -------
        float
            Giá trị hàm mục tiêu
        """
        # Giả lập tính toán objective function
        import random

        base_value = 100.0
        noise = random.uniform(-5, 5)
        decay = self.current_iteration * 0.1
        return max(0, base_value - decay + noise)

    def _check_convergence(self) -> bool:
        """
        Kiểm tra điều kiện hội tụ.

        Returns
        -------
        bool
            True nếu đã hội tụ
        """
        if len(self.convergence_history) < 2:
            return False

        recent_change = abs(self.convergence_history[-1] - self.convergence_history[-2])
        return recent_change < self.settings.convergence_threshold

    def _update_plan_parameters(self, plan_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Cập nhật parameters của kế hoạch.

        Parameters
        ----------
        plan_data : Dict[str, Any]
            Dữ liệu kế hoạch hiện tại

        Returns
        -------
        Dict[str, Any]
            Dữ liệu kế hoạch đã cập nhật
        """
        # Giả lập cập nhật parameters
        updated_plan = plan_data.copy()

        # Thêm thông tin iteration
        updated_plan["optimization_iteration"] = self.current_iteration
        updated_plan["algorithm"] = self.algorithm.value

        return updated_plan

    def stop_optimization(self):
        """Dừng quá trình tối ưu hóa."""
        self.is_running = False
        logger.info("Đã dừng quá trình tối ưu hóa")

    def get_optimization_status(self) -> Dict[str, Any]:
        """
        Lấy trạng thái hiện tại của quá trình tối ưu hóa.

        Returns
        -------
        Dict[str, Any]
            Thông tin trạng thái tối ưu hóa
        """
        return {
            "is_running": self.is_running,
            "current_iteration": self.current_iteration,
            "max_iterations": self.settings.max_iterations,
            "algorithm": self.algorithm.value,
            "num_objectives": len(self.settings.objectives),
            "num_constraints": len(self.settings.constraints),
            "convergence_history": self.convergence_history,
        }
