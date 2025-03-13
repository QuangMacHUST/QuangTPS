"""
Module định nghĩa các phương pháp tối ưu hóa dựa trên ràng buộc.

Module này cung cấp các thuật toán và chiến lược tối ưu hóa kế hoạch xạ trị
sử dụng các ràng buộc để giới hạn không gian tìm kiếm và đảm bảo các ràng buộc lâm sàng
được thỏa mãn trong quá trình tối ưu hóa.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field

from quangtps.dose.dose_grid import DoseGrid
from quangtps.optimization.objectives import ObjectiveCollection, ObjectiveBase
from quangtps.optimization.constraints import ConstraintCollection, ConstraintBase
from quangtps.optimization.optimization_engine import OptimizationParameters, OptimizationEngine, OptimizationResults

logger = logging.getLogger(__name__)

class ConstraintBasedMethod:
    """
    Lớp cơ sở cho các phương pháp tối ưu hóa dựa trên ràng buộc.
    """
    def __init__(
        self,
        objectives: ObjectiveCollection,
        constraints: ConstraintCollection,
        parameters: Optional[OptimizationParameters] = None
    ):
        """
        Khởi tạo phương pháp tối ưu hóa dựa trên ràng buộc.
        
        Args:
            objectives: Collection chứa các hàm mục tiêu
            constraints: Collection chứa các ràng buộc
            parameters: Các tham số tối ưu hóa
        """
        self.objectives = objectives
        self.constraints = constraints
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

class ConstraintSatisfactionMethod(ConstraintBasedMethod):
    """
    Phương pháp thỏa mãn ràng buộc, tập trung vào việc đảm bảo tất cả các ràng buộc
    được thỏa mãn mà không quan tâm đến việc tối ưu hóa bất kỳ hàm mục tiêu nào.
    """
    def __init__(
        self,
        constraints: ConstraintCollection,
        objectives: Optional[ObjectiveCollection] = None,
        parameters: Optional[OptimizationParameters] = None,
        feasibility_threshold: float = 1e-4
    ):
        """
        Khởi tạo phương pháp thỏa mãn ràng buộc.
        
        Args:
            constraints: Collection chứa các ràng buộc
            objectives: Collection chứa các hàm mục tiêu (nếu cần)
            parameters: Các tham số tối ưu hóa
            feasibility_threshold: Ngưỡng vi phạm ràng buộc được chấp nhận là khả thi
        """
        # Tạo hàm mục tiêu mặc định nếu không được cung cấp
        if objectives is None:
            objectives = ObjectiveCollection()
            objectives.add_objective(ConstraintViolationObjective(constraints))
        
        super().__init__(objectives, constraints, parameters)
        self.feasibility_threshold = feasibility_threshold
    
    def optimize(
        self,
        dose_grid: DoseGrid,
        structures: Dict[str, np.ndarray],
        solver_name: str = "gradient_descent"
    ) -> Tuple[DoseGrid, OptimizationResults]:
        """
        Thực hiện tối ưu hóa để tìm phân bố liều thỏa mãn các ràng buộc.
        
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
        
        # Kiểm tra xem tất cả các ràng buộc có được thỏa mãn không
        constraint_results = self.constraints.check_all(results.final_dose_grid, structures)
        all_satisfied = all(info["is_satisfied"] for info in constraint_results.values())
        
        if all_satisfied:
            logger.info("Tất cả các ràng buộc đều được thỏa mãn")
        else:
            # Lọc các ràng buộc không thỏa mãn
            unsatisfied = {name: info for name, info in constraint_results.items() if not info["is_satisfied"]}
            logger.warning(f"Có {len(unsatisfied)} ràng buộc không được thỏa mãn")
            for name, info in unsatisfied.items():
                logger.warning(f"- {name}: {info['description']}, vi phạm: {info['violation']:.4f}")
        
        return results.final_dose_grid, results

class PenaltyMethod(ConstraintBasedMethod):
    """
    Phương pháp tối ưu hóa sử dụng hàm phạt để xử lý các ràng buộc.
    """
    def __init__(
        self,
        objectives: ObjectiveCollection,
        constraints: ConstraintCollection,
        parameters: Optional[OptimizationParameters] = None,
        penalty_weight: float = 10.0,
        adaptive_weights: bool = True
    ):
        """
        Khởi tạo phương pháp hàm phạt.
        
        Args:
            objectives: Collection chứa các hàm mục tiêu
            constraints: Collection chứa các ràng buộc
            parameters: Các tham số tối ưu hóa
            penalty_weight: Trọng số ban đầu cho hàm phạt
            adaptive_weights: Có tăng trọng số phạt trong quá trình tối ưu hóa không
        """
        super().__init__(objectives, constraints, parameters)
        self.penalty_weight = penalty_weight
        self.adaptive_weights = adaptive_weights
        self.penalty_increase_factor = 2.0
        self.max_penalty_weight = 1e6
    
    def optimize(
        self,
        dose_grid: DoseGrid,
        structures: Dict[str, np.ndarray],
        solver_name: str = "gradient_descent"
    ) -> Tuple[DoseGrid, OptimizationResults]:
        """
        Thực hiện tối ưu hóa sử dụng phương pháp hàm phạt.
        
        Args:
            dose_grid: Phân bố liều ban đầu
            structures: Dictionary chứa các mặt nạ cấu trúc
            solver_name: Tên của thuật toán giải cần sử dụng
            
        Returns:
            Tuple[DoseGrid, OptimizationResults]: Phân bố liều tối ưu và kết quả tối ưu hóa
        """
        current_dose_grid = dose_grid.copy()
        current_penalty_weight = self.penalty_weight
        all_results = []
        
        if not self.adaptive_weights:
            # Nếu không sử dụng trọng số thích ứng, thực hiện tối ưu hóa một lần
            return super().optimize(dose_grid, structures, solver_name)
        
        # Thêm mục tiêu phạt
        penalty_objective = ConstraintViolationObjective(
            constraints=self.constraints,
            weight=current_penalty_weight
        )
        self.objectives.add_objective(penalty_objective)
        
        max_iterations = 5  # Số lần lặp tối đa cho việc tăng trọng số
        
        for i in range(max_iterations):
            logger.info(f"Vòng lặp hàm phạt {i+1}/{max_iterations}, trọng số phạt: {current_penalty_weight}")
            
            # Cập nhật trọng số cho hàm phạt
            penalty_objective.weight = current_penalty_weight
            
            # Tối ưu hóa với trọng số hiện tại
            final_dose_grid, results = super().optimize(current_dose_grid, structures, solver_name)
            all_results.append(results)
            
            # Cập nhật phân bố liều hiện tại
            current_dose_grid = final_dose_grid
            
            # Kiểm tra vi phạm ràng buộc
            constraint_results = self.constraints.check_all(final_dose_grid, structures)
            all_satisfied = all(info["is_satisfied"] for info in constraint_results.values())
            
            if all_satisfied:
                logger.info("Tất cả các ràng buộc đều được thỏa mãn, dừng vòng lặp hàm phạt")
                break
            
            # Tăng trọng số phạt
            current_penalty_weight = min(
                current_penalty_weight * self.penalty_increase_factor,
                self.max_penalty_weight
            )
            
            # Kiểm tra xem đã đạt đến trọng số tối đa chưa
            if current_penalty_weight >= self.max_penalty_weight:
                logger.warning("Đã đạt đến trọng số phạt tối đa, không thể tăng thêm")
                break
        
        # Gộp kết quả
        final_result = all_results[-1]
        final_result.penalty_results = all_results
        
        # Loại bỏ mục tiêu phạt khỏi danh sách để tránh ảnh hưởng đến các lần gọi sau
        self.objectives.remove_objective(len(self.objectives) - 1)
        
        return current_dose_grid, final_result

class AugmentedLagrangianMethod(ConstraintBasedMethod):
    """
    Phương pháp tối ưu hóa sử dụng Lagrangian tăng cường để xử lý các ràng buộc.
    """
    def __init__(
        self,
        objectives: ObjectiveCollection,
        constraints: ConstraintCollection,
        parameters: Optional[OptimizationParameters] = None,
        initial_penalty: float = 1.0,
        initial_multipliers: Optional[Dict[int, float]] = None
    ):
        """
        Khởi tạo phương pháp Lagrangian tăng cường.
        
        Args:
            objectives: Collection chứa các hàm mục tiêu
            constraints: Collection chứa các ràng buộc
            parameters: Các tham số tối ưu hóa
            initial_penalty: Hệ số phạt ban đầu
            initial_multipliers: Dictionary ánh xạ từ chỉ mục ràng buộc sang nhân tử Lagrange ban đầu
        """
        super().__init__(objectives, constraints, parameters)
        self.penalty = initial_penalty
        self.multipliers = initial_multipliers or {}
        self.penalty_increase_factor = 10.0
        self.max_penalty = 1e6
        self.tolerance = 1e-4
    
    def optimize(
        self,
        dose_grid: DoseGrid,
        structures: Dict[str, np.ndarray],
        solver_name: str = "gradient_descent"
    ) -> Tuple[DoseGrid, OptimizationResults]:
        """
        Thực hiện tối ưu hóa sử dụng phương pháp Lagrangian tăng cường.
        
        Args:
            dose_grid: Phân bố liều ban đầu
            structures: Dictionary chứa các mặt nạ cấu trúc
            solver_name: Tên của thuật toán giải cần sử dụng
            
        Returns:
            Tuple[DoseGrid, OptimizationResults]: Phân bố liều tối ưu và kết quả tối ưu hóa
        """
        current_dose_grid = dose_grid.copy()
        current_penalty = self.penalty
        all_results = []
        previous_max_violation = float('inf')  # Khởi tạo giá trị ban đầu cho vi phạm lớn nhất
        
        # Khởi tạo nhân tử Lagrange cho tất cả các ràng buộc nếu chưa có
        for i in range(len(self.constraints)):
            if i not in self.multipliers:
                self.multipliers[i] = 0.0
        
        # Thêm mục tiêu Lagrangian tăng cường
        lagrangian_objective = AugmentedLagrangianObjective(
            constraints=self.constraints,
            multipliers=self.multipliers,
            penalty=current_penalty
        )
        self.objectives.add_objective(lagrangian_objective)
        
        max_outer_iterations = 10  # Số lần lặp tối đa cho vòng lặp ngoài
        
        for i in range(max_outer_iterations):
            logger.info(f"Vòng lặp Lagrangian tăng cường {i+1}/{max_outer_iterations}, hệ số phạt: {current_penalty}")
            
            # Cập nhật hệ số phạt cho mục tiêu Lagrangian
            lagrangian_objective.penalty = current_penalty
            lagrangian_objective.multipliers = self.multipliers
            
            # Tối ưu hóa với các tham số hiện tại
            final_dose_grid, results = super().optimize(current_dose_grid, structures, solver_name)
            all_results.append(results)
            
            # Cập nhật phân bố liều hiện tại
            current_dose_grid = final_dose_grid
            
            # Kiểm tra vi phạm ràng buộc
            constraint_results = self.constraints.check_all(final_dose_grid, structures)
            constraint_violations = {i: info["violation"] for i, (_, info) in enumerate(constraint_results.items())}
            
            # Kiểm tra điều kiện dừng
            max_violation = max(constraint_violations.values()) if constraint_violations else 0
            if max_violation < self.tolerance:
                logger.info(f"Đạt đến độ chính xác yêu cầu (vi phạm lớn nhất: {max_violation:.6f}), dừng vòng lặp")
                break
            
            # Cập nhật nhân tử Lagrange
            for i, violation in constraint_violations.items():
                self.multipliers[i] = max(0, self.multipliers[i] + current_penalty * violation)
            
            # Tăng hệ số phạt nếu vi phạm chưa giảm đủ nhanh
            if i > 0 and max_violation > 0.25 * previous_max_violation:
                current_penalty = min(current_penalty * self.penalty_increase_factor, self.max_penalty)
                if current_penalty >= self.max_penalty:
                    logger.warning("Đã đạt đến hệ số phạt tối đa, không thể tăng thêm")
            
            # Lưu lại vi phạm lớn nhất cho vòng lặp tiếp theo
            previous_max_violation = max_violation
        
        # Gộp kết quả
        final_result = all_results[-1]
        final_result.lagrangian_results = all_results
        final_result.final_multipliers = self.multipliers.copy()
        
        # Loại bỏ mục tiêu Lagrangian khỏi danh sách để tránh ảnh hưởng đến các lần gọi sau
        self.objectives.remove_objective(len(self.objectives) - 1)
        
        return current_dose_grid, final_result

class ConstraintViolationObjective(ObjectiveBase):
    """Hàm mục tiêu để tối thiểu hóa vi phạm ràng buộc (dùng cho phương pháp hàm phạt)."""
    def __init__(
        self,
        constraints: ConstraintCollection,
        weight: float = 10.0,
        structure_name: str = "all_structures"
    ):
        """
        Khởi tạo hàm mục tiêu vi phạm ràng buộc.
        
        Args:
            constraints: Collection chứa các ràng buộc cần kiểm tra
            weight: Trọng số cho hàm mục tiêu
            structure_name: Tên giả định cho cấu trúc
        """
        super().__init__(structure_name=structure_name, weight=weight, objective_type="ConstraintViolation")
        self.constraints = constraints
    
    def _calculate_cost(self, dose_grid: DoseGrid, structure_mask: np.ndarray) -> float:
        """
        Tính toán tổng bình phương của vi phạm ràng buộc.
        
        Args:
            dose_grid: Phân bố liều hiện tại
            structure_mask: Không được sử dụng trong trường hợp này
            
        Returns:
            Tổng bình phương của vi phạm ràng buộc
        """
        # Lưu ý: structure_mask không được sử dụng vì chúng ta cần kiểm tra với tất cả các cấu trúc
        # và DoseGrid đã chứa đầy đủ thông tin.
        
        # Giả định là các cấu trúc đã được cung cấp bởi engine khi kiểm tra ràng buộc
        # Chúng ta cần truy cập vào trạng thái của engine để lấy structures
        engine = self._get_current_engine()
        if not engine or not hasattr(engine, "structures"):
            logger.warning("Không thể truy cập vào cấu trúc từ engine, không thể đánh giá vi phạm ràng buộc")
            return 0.0
        
        # Kiểm tra tất cả các ràng buộc
        constraint_results = self.constraints.check_all(dose_grid, engine.structures)
        
        # Tính tổng bình phương của vi phạm
        total_violation = sum(info["violation"]**2 for info in constraint_results.values())
        
        return total_violation
    
    def _get_current_engine(self) -> Optional[Any]:
        """
        Cố gắng lấy đối tượng engine hiện tại từ ngữ cảnh.
        
        Returns:
            OptimizationEngine hoặc None nếu không tìm thấy
        """
        # Đây là một hàm phụ trợ để cố gắng truy cập vào engine hiện tại
        # Trong thực tế, sẽ cần thiết kế lại cấu trúc để truyền engine hoặc structures khi cần
        
        # Giả định là engine đã được lưu ở đâu đó trong ngữ cảnh toàn cục
        # Đây không phải là cách tối ưu và chỉ nên được sử dụng như một giải pháp tạm thời
        
        import inspect
        frame = inspect.currentframe()
        
        try:
            # Tìm kiếm đối tượng engine trong các frame gọi
            while frame:
                if "engine" in frame.f_locals:
                    return frame.f_locals["engine"]
                frame = frame.f_back
        finally:
            # Đảm bảo frame được giải phóng để tránh tham chiếu vòng tròn
            del frame
        
        return None

class AugmentedLagrangianObjective(ObjectiveBase):
    """Hàm mục tiêu cho phương pháp Lagrangian tăng cường."""
    def __init__(
        self,
        constraints: ConstraintCollection,
        multipliers: Dict[int, float],
        penalty: float = 1.0,
        structure_name: str = "all_structures"
    ):
        """
        Khởi tạo hàm mục tiêu Lagrangian tăng cường.
        
        Args:
            constraints: Collection chứa các ràng buộc cần kiểm tra
            multipliers: Dictionary ánh xạ từ chỉ mục ràng buộc sang nhân tử Lagrange
            penalty: Hệ số phạt cho phần bình phương
            structure_name: Tên giả định cho cấu trúc
        """
        super().__init__(structure_name=structure_name, weight=1.0, objective_type="AugmentedLagrangian")
        self.constraints = constraints
        self.multipliers = multipliers
        self.penalty = penalty
    
    def _calculate_cost(self, dose_grid: DoseGrid, structure_mask: np.ndarray) -> float:
        """
        Tính toán giá trị của hàm Lagrangian tăng cường.
        
        Args:
            dose_grid: Phân bố liều hiện tại
            structure_mask: Không được sử dụng trong trường hợp này
            
        Returns:
            Giá trị của hàm Lagrangian tăng cường
        """
        # Tương tự như ConstraintViolationObjective, cần truy cập vào engine
        engine = self._get_current_engine()
        if not engine or not hasattr(engine, "structures"):
            logger.warning("Không thể truy cập vào cấu trúc từ engine, không thể đánh giá Lagrangian")
            return 0.0
        
        # Tính giá trị hàm mục tiêu gốc
        original_objective = 0.0
        for obj in engine.objectives:
            if obj != self:  # Tránh đệ quy vô hạn
                original_objective += obj.evaluate(dose_grid, engine.structures)
        
        # Kiểm tra tất cả các ràng buộc
        constraint_results = {}
        for i, constraint in enumerate(self.constraints):
            is_satisfied, violation = constraint.check(dose_grid, engine.structures)
            constraint_results[i] = violation
        
        # Tính giá trị Lagrangian tăng cường
        lagrangian_term = 0.0
        for i, violation in constraint_results.items():
            multiplier = self.multipliers.get(i, 0.0)
            # L(x, λ) = f(x) + λg(x) + (ρ/2)(g(x))²
            lagrangian_term += multiplier * violation + (self.penalty / 2) * (violation**2)
        
        return original_objective + lagrangian_term
    
    def _get_current_engine(self) -> Optional[Any]:
        """
        Cố gắng lấy đối tượng engine hiện tại từ ngữ cảnh.
        
        Returns:
            OptimizationEngine hoặc None nếu không tìm thấy
        """
        import inspect
        frame = inspect.currentframe()
        
        try:
            while frame:
                if "engine" in frame.f_locals:
                    return frame.f_locals["engine"]
                frame = frame.f_back
        finally:
            del frame
        
        return None

def create_constraint_based_method(
    method_type: str,
    objectives: ObjectiveCollection,
    constraints: ConstraintCollection,
    parameters: Optional[OptimizationParameters] = None,
    **kwargs
) -> ConstraintBasedMethod:
    """
    Tạo đối tượng phương pháp tối ưu hóa dựa trên loại phương pháp.
    
    Args:
        method_type: Loại phương pháp tối ưu hóa ("constraint_satisfaction", "penalty", "augmented_lagrangian")
        objectives: Collection chứa các hàm mục tiêu
        constraints: Collection chứa các ràng buộc
        parameters: Các tham số tối ưu hóa
        **kwargs: Các tham số bổ sung cho từng loại phương pháp
        
    Returns:
        Đối tượng phương pháp tối ưu hóa
    """
    if method_type == "constraint_satisfaction":
        feasibility_threshold = kwargs.get("feasibility_threshold", 1e-4)
        return ConstraintSatisfactionMethod(constraints, objectives, parameters, feasibility_threshold)
    elif method_type == "penalty":
        penalty_weight = kwargs.get("penalty_weight", 10.0)
        adaptive_weights = kwargs.get("adaptive_weights", True)
        return PenaltyMethod(objectives, constraints, parameters, penalty_weight, adaptive_weights)
    elif method_type == "augmented_lagrangian":
        initial_penalty = kwargs.get("initial_penalty", 1.0)
        initial_multipliers = kwargs.get("initial_multipliers", None)
        return AugmentedLagrangianMethod(objectives, constraints, parameters, initial_penalty, initial_multipliers)
    else:
        raise ValueError(f"Không hỗ trợ loại phương pháp: {method_type}")
