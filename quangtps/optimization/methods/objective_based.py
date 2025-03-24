"""
Module định nghĩa các phương pháp tối ưu hóa dựa trên hàm mục tiêu.

Module này cung cấp các thuật toán và chiến lược tối ưu hóa kế hoạch xạ trị
sử dụng các hàm mục tiêu để hướng dẫn quá trình tối ưu hóa.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field

from quangtps.dose.dose_grid import DoseGrid
from quangtps.optimization.objectives import ObjectiveCollection
from quangtps.optimization.constraints import ConstraintCollection
from quangtps.optimization.optimization_engine import OptimizationParameters, OptimizationEngine, OptimizationResults

logger = logging.getLogger(__name__)

class ObjectiveBasedMethod:
    """
    Lớp cơ sở cho các phương pháp tối ưu hóa dựa trên hàm mục tiêu.
    """
    def __init__(
        self,
        objectives: ObjectiveCollection,
        constraints: Optional[ConstraintCollection] = None,
        parameters: Optional[OptimizationParameters] = None
    ):
        """
        Khởi tạo phương pháp tối ưu hóa.
        
        Args:
            objectives: Collection chứa các hàm mục tiêu
            constraints: Collection chứa các ràng buộc (nếu có)
            parameters: Các tham số tối ưu hóa
        """
        self.objectives = objectives
        self.constraints = constraints or ConstraintCollection()
        self.parameters = parameters or OptimizationParameters()
        
    def create_engine(self, solver_name: str = "gradient_descent") -> OptimizationEngine:
        """
        Tạo đối tượng tối ưu hóa engine với các tham số đã được thiết lập.
        
        Args:
            solver_name: Tên của thuật toán giải cần sử dụng
            
        Returns:
            Đối tượng OptimizationEngine đã được cấu hình
        """
        from quangtps.optimization import create_engine
        
        return create_engine(
            objectives=self.objectives,
            constraints=self.constraints,
            parameters=self.parameters,
            solver_name=solver_name
        )
    
    def optimize(
        self,
        dose_grid: DoseGrid,
        structures: Dict[str, np.ndarray],
        solver_name: str = "gradient_descent"
    ) -> Tuple[DoseGrid, OptimizationResults]:
        """
        Thực hiện tối ưu hóa với phân bố liều và cấu trúc đã cho.
        
        Args:
            dose_grid: Phân bố liều ban đầu
            structures: Dictionary chứa các mặt nạ cấu trúc
            solver_name: Tên của thuật toán giải cần sử dụng
            
        Returns:
            Tuple[DoseGrid, OptimizationResults]: Phân bố liều tối ưu và kết quả tối ưu hóa
        """
        # Tạo engine
        engine = self.create_engine(solver_name)
        
        # Thiết lập trạng thái ban đầu
        engine.set_initial_state(dose_grid, structures)
        
        # Tối ưu hóa
        results = engine.optimize()
        
        return results.final_dose_grid, results
    
    def update_objectives(self, new_objectives: ObjectiveCollection) -> None:
        """
        Cập nhật collection hàm mục tiêu.
        
        Args:
            new_objectives: Collection hàm mục tiêu mới
        """
        self.objectives = new_objectives
        
    def update_constraints(self, new_constraints: ConstraintCollection) -> None:
        """
        Cập nhật collection ràng buộc.
        
        Args:
            new_constraints: Collection ràng buộc mới
        """
        self.constraints = new_constraints
        
    def update_parameters(self, new_parameters: OptimizationParameters) -> None:
        """
        Cập nhật tham số tối ưu hóa.
        
        Args:
            new_parameters: Tham số tối ưu hóa mới
        """
        self.parameters = new_parameters

class WeightedSumMethod(ObjectiveBasedMethod):
    """
    Phương pháp tối ưu hóa sử dụng tổng có trọng số của các hàm mục tiêu.
    Đây là phương pháp phổ biến nhất trong lập kế hoạch xạ trị.
    """
    def __init__(
        self,
        objectives: ObjectiveCollection,
        constraints: Optional[ConstraintCollection] = None,
        parameters: Optional[OptimizationParameters] = None
    ):
        """
        Khởi tạo phương pháp tối ưu hóa tổng có trọng số.
        
        Args:
            objectives: Collection chứa các hàm mục tiêu
            constraints: Collection chứa các ràng buộc (nếu có)
            parameters: Các tham số tối ưu hóa
        """
        super().__init__(objectives, constraints, parameters)
    
    def scale_objective_weights(self, scaling_factor: float) -> None:
        """
        Nhân tất cả trọng số của các mục tiêu với một hệ số.
        
        Args:
            scaling_factor: Hệ số nhân cho trọng số
        """
        for objective in self.objectives:
            objective.weight *= scaling_factor
    
    def set_objective_weights(self, weights: Dict[int, float]) -> None:
        """
        Thiết lập trọng số cụ thể cho các mục tiêu dựa trên chỉ mục.
        
        Args:
            weights: Dictionary ánh xạ từ chỉ mục mục tiêu sang trọng số
        """
        for idx, weight in weights.items():
            if idx < len(self.objectives):
                self.objectives[idx].weight = weight

class LexicographicMethod(ObjectiveBasedMethod):
    """
    Phương pháp tối ưu hóa theo thứ tự từ điển, tối ưu hóa tuần tự từng mục tiêu
    theo thứ tự ưu tiên.
    """
    def __init__(
        self,
        objectives: ObjectiveCollection,
        constraints: Optional[ConstraintCollection] = None,
        parameters: Optional[OptimizationParameters] = None,
        tolerance: float = 0.05
    ):
        """
        Khởi tạo phương pháp tối ưu hóa theo thứ tự từ điển.
        
        Args:
            objectives: Collection chứa các hàm mục tiêu
            constraints: Collection chứa các ràng buộc (nếu có)
            parameters: Các tham số tối ưu hóa
            tolerance: Dung sai cho phép khi tối ưu hóa mục tiêu cao hơn
        """
        super().__init__(objectives, constraints, parameters)
        self.tolerance = tolerance
        self.objective_order: List[int] = list(range(len(objectives)))
    
    def set_objective_order(self, order: List[int]) -> None:
        """
        Thiết lập thứ tự ưu tiên của các mục tiêu.
        
        Args:
            order: Danh sách chỉ mục của các mục tiêu theo thứ tự ưu tiên giảm dần
        """
        if not set(order) == set(range(len(self.objectives))):
            raise ValueError("Danh sách thứ tự phải chứa tất cả các chỉ mục mục tiêu")
        self.objective_order = order
    
    def optimize(
        self,
        dose_grid: DoseGrid,
        structures: Dict[str, np.ndarray],
        solver_name: str = "gradient_descent"
    ) -> Tuple[DoseGrid, OptimizationResults]:
        """
        Thực hiện tối ưu hóa theo thứ tự từ điển.
        
        Args:
            dose_grid: Phân bố liều ban đầu
            structures: Dictionary chứa các mặt nạ cấu trúc
            solver_name: Tên của thuật toán giải cần sử dụng
            
        Returns:
            Tuple[DoseGrid, OptimizationResults]: Phân bố liều tối ưu và kết quả tối ưu hóa
        """
        current_dose_grid = dose_grid.copy()
        all_results = []
        optimal_values = []
        
        # Lưu trạng thái ban đầu của các mục tiêu
        original_states = [(obj.is_enabled, obj.weight) for obj in self.objectives]
        
        try:
            # Tắt tất cả các mục tiêu
            for obj in self.objectives:
                obj.is_enabled = False
            
            # Tối ưu hóa từng mục tiêu theo thứ tự
            for idx in self.objective_order:
                logger.info(f"Tối ưu hóa mục tiêu #{idx}: {self.objectives[idx].objective_type} cho {self.objectives[idx].structure_name}")
                
                # Bật mục tiêu hiện tại và đặt trọng số cao
                self.objectives[idx].is_enabled = True
                self.objectives[idx].weight = 1.0
                
                # Tạo engine
                engine = self.create_engine(solver_name)
                
                # Thiết lập trạng thái ban đầu
                engine.set_initial_state(current_dose_grid, structures)
                
                # Tối ưu hóa
                results = engine.optimize()
                all_results.append(results)
                
                # Cập nhật phân bố liều hiện tại
                current_dose_grid = results.final_dose_grid
                
                # Lưu giá trị tối ưu
                optimal_values.append(results.final_objective_value)
                
                # Chuyển mục tiêu hiện tại thành ràng buộc cho các lần tiếp theo
                if idx < len(self.objective_order) - 1:  # Không cần thêm ràng buộc cho mục tiêu cuối cùng
                    obj = self.objectives[idx]
                    optimal_value = results.final_objective_value
                    
                    # Thêm ràng buộc: giá trị hàm mục tiêu này không được vượt quá optimal_value * (1 + tolerance)
                    # cho các lần tối ưu hóa tiếp theo
                    self.constraints.add_constraint(
                        LexicographicConstraint(
                            objective_index=idx,
                            objective_collection=self.objectives,
                            max_value=optimal_value * (1 + self.tolerance)
                        )
                    )
                
                # Tắt mục tiêu hiện tại cho lần tiếp theo
                self.objectives[idx].is_enabled = False
            
            # Bật lại tất cả các mục tiêu cho lần đánh giá cuối cùng
            for idx, (is_enabled, weight) in enumerate(original_states):
                self.objectives[idx].is_enabled = is_enabled
                self.objectives[idx].weight = weight
            
            # Tạo kết quả tổng hợp
            final_result = all_results[-1]  # Lấy kết quả của lần tối ưu hóa cuối cùng
            final_result.lexicographic_results = all_results
            final_result.lexicographic_values = optimal_values
            
            return current_dose_grid, final_result
            
        finally:
            # Khôi phục trạng thái ban đầu của các mục tiêu
            for idx, (is_enabled, weight) in enumerate(original_states):
                self.objectives[idx].is_enabled = is_enabled
                self.objectives[idx].weight = weight

@dataclass
class LexicographicConstraint:
    """Ràng buộc cho phương pháp tối ưu hóa theo thứ tự từ điển."""
    objective_index: int
    objective_collection: ObjectiveCollection
    max_value: float
    is_enabled: bool = True
    priority: int = 1
    constraint_type: str = "Lexicographic"
    is_hard_constraint: bool = True
    
    def check(self, dose_grid: DoseGrid, structures: Dict[str, np.ndarray]) -> Tuple[bool, float]:
        """
        Kiểm tra xem ràng buộc có được thỏa mãn không.
        
        Args:
            dose_grid: Phân bố liều hiện tại trong kế hoạch
            structures: Dictionary chứa các mặt nạ cấu trúc
            
        Returns:
            Tuple[bool, float]: (Có thỏa mãn không, Mức độ vi phạm)
        """
        if not self.is_enabled:
            return True, 0.0
        
        # Lưu trạng thái hiện tại của mục tiêu
        obj = self.objective_collection.objectives[self.objective_index]
        original_enabled = obj.is_enabled
        
        try:
            # Bật tạm thời để đánh giá
            obj.is_enabled = True
            
            # Đánh giá giá trị mục tiêu
            current_value = obj.evaluate(dose_grid, structures)
            
            # Kiểm tra xem có thỏa mãn ràng buộc không
            is_satisfied = current_value <= self.max_value
            violation = max(0, current_value - self.max_value)
            
            return is_satisfied, violation
            
        finally:
            # Khôi phục trạng thái
            obj.is_enabled = original_enabled
    
    def get_info(self) -> Dict[str, Any]:
        """Trả về thông tin mô tả về ràng buộc."""
        obj = self.objective_collection.objectives[self.objective_index]
        return {
            "structure_name": obj.structure_name,
            "type": self.constraint_type,
            "priority": self.priority,
            "is_enabled": self.is_enabled,
            "is_hard_constraint": self.is_hard_constraint,
            "objective_type": obj.objective_type,
            "max_value": self.max_value
        }
    
    def get_description(self) -> str:
        """Trả về mô tả bằng văn bản của ràng buộc."""
        obj = self.objective_collection.objectives[self.objective_index]
        return f"Lexicographic: {obj.objective_type} cho {obj.structure_name} ≤ {self.max_value:.4f}"

class GoalProgrammingMethod(ObjectiveBasedMethod):
    """
    Phương pháp tối ưu hóa lập trình mục tiêu, tập trung vào việc đạt được
    các mục tiêu cụ thể thay vì tối ưu hóa chúng.
    """
    def __init__(
        self,
        objectives: ObjectiveCollection,
        constraints: Optional[ConstraintCollection] = None,
        parameters: Optional[OptimizationParameters] = None,
        goal_values: Optional[Dict[int, float]] = None
    ):
        """
        Khởi tạo phương pháp tối ưu hóa lập trình mục tiêu.
        
        Args:
            objectives: Collection chứa các hàm mục tiêu
            constraints: Collection chứa các ràng buộc (nếu có)
            parameters: Các tham số tối ưu hóa
            goal_values: Dictionary ánh xạ từ chỉ mục mục tiêu sang giá trị mục tiêu
        """
        super().__init__(objectives, constraints, parameters)
        self.goal_values = goal_values or {}
    
    def set_goal_values(self, goal_values: Dict[int, float]) -> None:
        """
        Thiết lập giá trị mục tiêu cho các hàm mục tiêu.
        
        Args:
            goal_values: Dictionary ánh xạ từ chỉ mục mục tiêu sang giá trị mục tiêu
        """
        self.goal_values = goal_values
    
    def optimize(
        self,
        dose_grid: DoseGrid,
        structures: Dict[str, np.ndarray],
        solver_name: str = "gradient_descent"
    ) -> Tuple[DoseGrid, OptimizationResults]:
        """
        Thực hiện tối ưu hóa lập trình mục tiêu.
        
        Args:
            dose_grid: Phân bố liều ban đầu
            structures: Dictionary chứa các mặt nạ cấu trúc
            solver_name: Tên của thuật toán giải cần sử dụng
            
        Returns:
            Tuple[DoseGrid, OptimizationResults]: Phân bố liều tối ưu và kết quả tối ưu hóa
        """
        # Lưu trạng thái ban đầu của các mục tiêu
        original_objectives = self.objectives
        
        try:
            # Tạo các mục tiêu mới dựa trên các giá trị mục tiêu
            goal_objectives = ObjectiveCollection()
            
            for idx, goal_value in self.goal_values.items():
                if idx >= len(self.objectives):
                    logger.warning(f"Chỉ mục mục tiêu {idx} vượt quá số lượng mục tiêu hiện có")
                    continue
                
                original_obj = self.objectives[idx]
                
                # Tạo mục tiêu mới là sự khác biệt bình phương giữa giá trị hiện tại và giá trị mục tiêu
                goal_obj = GoalObjective(
                    original_objective=original_obj,
                    goal_value=goal_value,
                    weight=original_obj.weight
                )
                
                goal_objectives.add_objective(goal_obj)
            
            # Thay thế tạm thời các mục tiêu
            self.objectives = goal_objectives
            
            # Tạo engine
            engine = self.create_engine(solver_name)
            
            # Thiết lập trạng thái ban đầu
            engine.set_initial_state(dose_grid, structures)
            
            # Tối ưu hóa
            results = engine.optimize()
            
            return results.final_dose_grid, results
            
        finally:
            # Khôi phục các mục tiêu ban đầu
            self.objectives = original_objectives

@dataclass
class GoalObjective:
    """Mục tiêu cho phương pháp lập trình mục tiêu."""
    original_objective: Any  # ObjectiveBase
    goal_value: float
    weight: float = 1.0
    is_enabled: bool = True
    objective_type: str = "Goal"
    
    @property
    def structure_name(self) -> str:
        """Lấy tên cấu trúc từ mục tiêu gốc."""
        return self.original_objective.structure_name
    
    def evaluate(self, dose_grid: DoseGrid, structures: Dict[str, np.ndarray]) -> float:
        """
        Đánh giá hàm mục tiêu với phân bố liều và cấu trúc hiện tại.
        
        Args:
            dose_grid: Phân bố liều hiện tại trong kế hoạch
            structures: Dictionary chứa các mặt nạ cấu trúc
            
        Returns:
            Giá trị của hàm mục tiêu (cost)
        """
        if not self.is_enabled:
            return 0.0
            
        # Đánh giá mục tiêu gốc
        current_value = self.original_objective.evaluate(dose_grid, structures)
        
        # Tính độ lệch bình phương so với giá trị mục tiêu
        deviation = (current_value - self.goal_value)**2
        
        return deviation * self.weight
    
    def get_info(self) -> Dict[str, Any]:
        """Trả về thông tin mô tả về hàm mục tiêu."""
        return {
            "structure_name": self.structure_name,
            "type": self.objective_type,
            "weight": self.weight,
            "is_enabled": self.is_enabled,
            "original_objective_type": self.original_objective.objective_type,
            "goal_value": self.goal_value
        }

def create_objective_based_method(
    method_type: str,
    objectives: ObjectiveCollection,
    constraints: Optional[ConstraintCollection] = None,
    parameters: Optional[OptimizationParameters] = None,
    **kwargs
) -> ObjectiveBasedMethod:
    """
    Tạo đối tượng phương pháp tối ưu hóa dựa trên loại phương pháp.
    
    Args:
        method_type: Loại phương pháp tối ưu hóa ("weighted_sum", "lexicographic", "goal_programming")
        objectives: Collection chứa các hàm mục tiêu
        constraints: Collection chứa các ràng buộc (nếu có)
        parameters: Các tham số tối ưu hóa
        **kwargs: Các tham số bổ sung cho từng loại phương pháp
        
    Returns:
        Đối tượng phương pháp tối ưu hóa
    """
    if method_type == "weighted_sum":
        return WeightedSumMethod(objectives, constraints, parameters)
    elif method_type == "lexicographic":
        tolerance = kwargs.get("tolerance", 0.05)
        method = LexicographicMethod(objectives, constraints, parameters, tolerance)
        
        if "objective_order" in kwargs:
            method.set_objective_order(kwargs["objective_order"])
            
        return method
    elif method_type == "goal_programming":
        method = GoalProgrammingMethod(objectives, constraints, parameters)
        
        if "goal_values" in kwargs:
            method.set_goal_values(kwargs["goal_values"])
            
        return method
    else:
        raise ValueError(f"Không hỗ trợ loại phương pháp: {method_type}")
