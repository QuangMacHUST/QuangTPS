"""
Module tối ưu hóa đa tiêu chí (Multi-Criteria Optimization - MCO) cho hệ thống QuangTPS.

Module này cung cấp các lớp và hàm để thực hiện quá trình tối ưu hóa kế hoạch xạ trị 
dựa trên nhiều tiêu chí khác nhau, giúp bác sĩ cân bằng giữa các mục tiêu cạnh tranh như 
tối đa hóa liều cho khối u và giảm thiểu liều cho các cơ quan nguy cấp.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
import threading
import queue
from enum import Enum, auto
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, Delaunay
from scipy.optimize import minimize

from quangtps.optimization.objectives import ObjectiveBase, ObjectiveCollection
from quangtps.optimization.constraints import ConstraintBase, ConstraintCollection
from quangtps.optimization.optimization_engine import OptimizationEngine, OptimizationParameters, OptimizationResults
from quangtps.dose.dose_grid import DoseGrid
from quangtps.evaluation.dvh import calculate_dvh

logger = logging.getLogger(__name__)

class ParetoSolutionStatus(Enum):
    """Trạng thái của một điểm nghiệm trong không gian Pareto."""
    OPTIMAL = auto()      # Điểm nằm trên mặt Pareto
    DOMINATED = auto()    # Điểm bị chi phối bởi các điểm khác
    INFEASIBLE = auto()   # Điểm vi phạm các ràng buộc
    CANDIDATE = auto()    # Điểm đang được xem xét
    SELECTED = auto()     # Điểm được người dùng lựa chọn

@dataclass
class ParetoSolution:
    """Đại diện cho một điểm nghiệm trong không gian Pareto."""
    objective_values: Dict[str, float]
    weights: Dict[str, float]
    status: ParetoSolutionStatus = ParetoSolutionStatus.CANDIDATE
    control_parameters: Optional[np.ndarray] = None
    dose_grid: Optional[DoseGrid] = None
    optimization_results: Optional[OptimizationResults] = None
    dvh_data: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_dominated_by(self, other: 'ParetoSolution') -> bool:
        """
        Kiểm tra xem nghiệm này có bị chi phối bởi một nghiệm khác không.
        
        Một nghiệm bị chi phối nếu tất cả các mục tiêu của nghiệm khác tốt hơn hoặc bằng nó,
        và ít nhất một mục tiêu của nghiệm khác tốt hơn nó.
        """
        if self.status == ParetoSolutionStatus.INFEASIBLE:
            return True
        
        if other.status == ParetoSolutionStatus.INFEASIBLE:
            return False
        
        # Giả sử giá trị mục tiêu càng thấp càng tốt
        all_better_or_equal = True
        any_better = False
        
        for obj_name, obj_value in self.objective_values.items():
            if obj_name in other.objective_values:
                other_value = other.objective_values[obj_name]
                if other_value > obj_value:
                    all_better_or_equal = False
                    break
                if other_value < obj_value:
                    any_better = True
        
        return all_better_or_equal and any_better
    
    def distance_to(self, other: 'ParetoSolution') -> float:
        """Tính khoảng cách Euclidean đến một nghiệm khác trong không gian mục tiêu."""
        squared_diffs = []
        for obj_name, obj_value in self.objective_values.items():
            if obj_name in other.objective_values:
                diff = obj_value - other.objective_values[obj_name]
                squared_diffs.append(diff ** 2)
        
        if not squared_diffs:
            return float('inf')
        
        return np.sqrt(sum(squared_diffs))
    
    def get_summary(self) -> Dict[str, Any]:
        """Trả về tóm tắt ngắn gọn thông tin của nghiệm."""
        return {
            "objectives": self.objective_values,
            "weights": self.weights,
            "status": self.status.name,
            "metadata": self.metadata
        }

class MCOEngine:
    """
    Động cơ tối ưu hóa đa tiêu chí (MCO) cho kế hoạch xạ trị.
    
    Lớp này cho phép tạo và quản lý tập hợp các kế hoạch tối ưu theo từng tiêu chí riêng biệt,
    cho phép bác sĩ khám phá không gian Pareto và lựa chọn kế hoạch phù hợp nhất.
    """
    
    def __init__(
        self,
        objectives: ObjectiveCollection,
        constraints: Optional[ConstraintCollection] = None,
        parameters: Optional[OptimizationParameters] = None,
        base_optimizer: str = "gradient_descent"
    ):
        """
        Khởi tạo động cơ tối ưu hóa đa tiêu chí.
        
        Args:
            objectives: Tập hợp các hàm mục tiêu để tối ưu hóa
            constraints: Tập hợp các ràng buộc phải thỏa mãn
            parameters: Tham số tối ưu hóa
            base_optimizer: Thuật toán tối ưu hóa nền tảng
        """
        self.objectives = objectives
        self.constraints = constraints or ConstraintCollection()
        self.parameters = parameters or OptimizationParameters()
        self.base_optimizer = base_optimizer
        
        # Quản lý các nghiệm Pareto
        self.pareto_solutions: List[ParetoSolution] = []
        self.anchor_solutions: Dict[str, ParetoSolution] = {}
        self.selected_solution: Optional[ParetoSolution] = None
        
        # Dữ liệu kế hoạch
        self.structures = {}
        self.initial_dose_grid = None
        self.initial_control_parameters = None
        
        # Trạng thái
        self.is_initialized = False
        self.current_navigating_weights = {}

    def initialize(
        self,
        structures: Dict[str, np.ndarray],
        initial_dose_grid: Optional[DoseGrid] = None,
        initial_control_parameters: Optional[np.ndarray] = None
    ) -> None:
        """
        Khởi tạo động cơ MCO với dữ liệu cần thiết.
        
        Args:
            structures: Dictionary chứa các mặt nạ cấu trúc
            initial_dose_grid: Phân bố liều ban đầu (tùy chọn)
            initial_control_parameters: Tham số điều khiển ban đầu (tùy chọn)
        """
        self.structures = structures
        self.initial_dose_grid = initial_dose_grid
        self.initial_control_parameters = initial_control_parameters
        self.is_initialized = True
        
        # Xóa các nghiệm cũ
        self.pareto_solutions = []
        self.anchor_solutions = {}
        self.selected_solution = None
        
        # Khởi tạo các trọng số ban đầu bằng nhau cho tất cả các mục tiêu
        objective_info = self.objectives.get_objectives_info()
        equal_weight = 1.0 / len(objective_info) if objective_info else 0.0
        self.current_navigating_weights = {obj["id"]: equal_weight for obj in objective_info}
        
        logger.info("MCO Engine đã được khởi tạo thành công với %d cấu trúc", len(structures))

    def generate_anchor_plans(self) -> Dict[str, ParetoSolution]:
        """
        Tạo các kế hoạch neo (anchor plans) cho mỗi mục tiêu.
        
        Mỗi kế hoạch neo tối ưu hóa một mục tiêu cụ thể và bỏ qua các mục tiêu khác.
        Các kế hoạch này định nghĩa các điểm cực trị trong không gian Pareto.
        
        Returns:
            Dictionary chứa các kế hoạch neo, với khóa là ID của mục tiêu
        """
        if not self.is_initialized:
            raise RuntimeError("MCO Engine chưa được khởi tạo")
        
        self.anchor_solutions = {}
        objective_info = self.objectives.get_objectives_info()
        
        for obj in objective_info:
            # Tạo một bản sao của tập mục tiêu và chỉ bật một mục tiêu
            obj_collection = ObjectiveCollection()
            for other_obj in objective_info:
                original_obj = self.objectives.objectives[other_obj["index"]]
                new_obj = type(original_obj)(**{k: getattr(original_obj, k) for k in original_obj.__dataclass_fields__})
                new_obj.is_enabled = (other_obj["id"] == obj["id"])
                new_obj.weight = 1.0 if new_obj.is_enabled else 0.0
                obj_collection.add_objective(new_obj)
            
            # Tạo động cơ tối ưu hóa chỉ cho mục tiêu này
            optimizer = OptimizationEngine(
                objectives=obj_collection,
                constraints=self.constraints,
                parameters=self.parameters,
                solver_name=self.base_optimizer
            )
            
            # Thực hiện tối ưu hóa
            logger.info("Đang tạo kế hoạch neo cho mục tiêu: %s", obj["id"])
            optimizer.set_structures(self.structures)
            
            if self.initial_dose_grid is not None:
                optimizer.set_dose_grid(self.initial_dose_grid)
                
            if self.initial_control_parameters is not None:
                optimizer.set_control_parameters(self.initial_control_parameters)
            
            results = optimizer.optimize()
            
            # Lưu trữ kết quả như một nghiệm Pareto
            weights = {other_obj["id"]: 0.0 for other_obj in objective_info}
            weights[obj["id"]] = 1.0
            
            objective_values = {}
            for o in objective_info:
                o_obj = self.objectives.objectives[o["index"]]
                value = o_obj.evaluate(optimizer.dose_grid, self.structures)
                objective_values[o["id"]] = value
            
            solution = ParetoSolution(
                objective_values=objective_values,
                weights=weights,
                status=ParetoSolutionStatus.OPTIMAL,
                control_parameters=optimizer.control_parameters.copy() if optimizer.control_parameters is not None else None,
                dose_grid=optimizer.dose_grid.copy() if optimizer.dose_grid is not None else None,
                optimization_results=results,
                metadata={"type": "anchor", "obj_id": obj["id"]}
            )
            
            self.anchor_solutions[obj["id"]] = solution
            self.pareto_solutions.append(solution)
            
            logger.info("Đã tạo kế hoạch neo cho %s với giá trị: %s", 
                        obj["id"], {k: round(v, 3) for k, v in objective_values.items()})
        
        return self.anchor_solutions

    def generate_pareto_surface(self, num_samples: int = 10) -> List[ParetoSolution]:
        """
        Tạo mẫu không gian Pareto bằng cách lấy mẫu ngẫu nhiên các trọng số.
        
        Args:
            num_samples: Số lượng mẫu để tạo
            
        Returns:
            Danh sách các nghiệm Pareto được tạo
        """
        if not self.is_initialized:
            raise RuntimeError("MCO Engine chưa được khởi tạo")
        
        if not self.anchor_solutions:
            logger.info("Không tìm thấy các kế hoạch neo, đang tạo kế hoạch neo tự động...")
            self.generate_anchor_plans()
        
        objective_info = self.objectives.get_objectives_info()
        new_solutions = []
        
        for _ in range(num_samples):
            # Tạo trọng số ngẫu nhiên có tổng bằng 1
            weights = np.random.dirichlet(np.ones(len(objective_info)))
            weights_dict = {obj["id"]: w for obj, w in zip(objective_info, weights)}
            
            # Tạo bản sao của tập mục tiêu với trọng số mới
            obj_collection = ObjectiveCollection()
            for i, obj in enumerate(objective_info):
                original_obj = self.objectives.objectives[obj["index"]]
                new_obj = type(original_obj)(**{k: getattr(original_obj, k) for k in original_obj.__dataclass_fields__})
                new_obj.is_enabled = True
                new_obj.weight = weights[i]
                obj_collection.add_objective(new_obj)
            
            # Tạo động cơ tối ưu hóa với trọng số mới
            optimizer = OptimizationEngine(
                objectives=obj_collection,
                constraints=self.constraints,
                parameters=self.parameters,
                solver_name=self.base_optimizer
            )
            
            # Thực hiện tối ưu hóa
            logger.info("Đang tạo điểm nghiệm Pareto với trọng số: %s", 
                        {k: round(v, 3) for k, v in weights_dict.items()})
            optimizer.set_structures(self.structures)
            
            if self.initial_dose_grid is not None:
                optimizer.set_dose_grid(self.initial_dose_grid)
                
            if self.initial_control_parameters is not None:
                optimizer.set_control_parameters(self.initial_control_parameters)
            
            results = optimizer.optimize()
            
            # Lưu trữ kết quả như một nghiệm Pareto
            objective_values = {}
            for obj in objective_info:
                o_obj = self.objectives.objectives[obj["index"]]
                value = o_obj.evaluate(optimizer.dose_grid, self.structures)
                objective_values[obj["id"]] = value
            
            solution = ParetoSolution(
                objective_values=objective_values,
                weights=weights_dict,
                status=ParetoSolutionStatus.CANDIDATE,
                control_parameters=optimizer.control_parameters.copy() if optimizer.control_parameters is not None else None,
                dose_grid=optimizer.dose_grid.copy() if optimizer.dose_grid is not None else None,
                optimization_results=results,
                metadata={"type": "sampled"}
            )
            
            new_solutions.append(solution)
            self.pareto_solutions.append(solution)
            
            logger.info("Đã tạo điểm nghiệm Pareto với giá trị: %s", 
                        {k: round(v, 3) for k, v in objective_values.items()})
        
        # Cập nhật trạng thái các nghiệm
        self._update_pareto_status()
        
        return new_solutions

    def _update_pareto_status(self) -> None:
        """
        Cập nhật trạng thái của tất cả các nghiệm trong không gian Pareto.
        
        Xác định các nghiệm nào nằm trên bề mặt Pareto (không bị chi phối bởi bất kỳ
        nghiệm nào khác) và các nghiệm nào bị chi phối.
        """
        if not self.pareto_solutions:
            return
        
        for i, solution in enumerate(self.pareto_solutions):
            # Bỏ qua các nghiệm đã được chọn hoặc không khả thi
            if solution.status in [ParetoSolutionStatus.SELECTED, ParetoSolutionStatus.INFEASIBLE]:
                continue
            
            is_dominated = False
            for j, other_solution in enumerate(self.pareto_solutions):
                if i == j:
                    continue
                
                if solution.is_dominated_by(other_solution):
                    is_dominated = True
                    break
            
            if is_dominated:
                solution.status = ParetoSolutionStatus.DOMINATED
            else:
                solution.status = ParetoSolutionStatus.OPTIMAL
        
        logger.info("Đã cập nhật trạng thái Pareto: %d tối ưu, %d bị chi phối", 
                   sum(1 for s in self.pareto_solutions if s.status == ParetoSolutionStatus.OPTIMAL),
                   sum(1 for s in self.pareto_solutions if s.status == ParetoSolutionStatus.DOMINATED))

    def get_pareto_front(self) -> List[ParetoSolution]:
        """
        Trả về tất cả các nghiệm nằm trên bề mặt Pareto.
        
        Returns:
            Danh sách các nghiệm Pareto tối ưu
        """
        self._update_pareto_status()
        return [s for s in self.pareto_solutions if s.status == ParetoSolutionStatus.OPTIMAL]

    def navigate_pareto_surface(self, weights: Dict[str, float]) -> ParetoSolution:
        """
        Tạo một kế hoạch mới dựa trên nội suy giữa các kế hoạch Pareto hiện có.
        
        Args:
            weights: Trọng số của các mục tiêu cho kế hoạch mới
            
        Returns:
            Nghiệm Pareto mới được tạo ra
        """
        if not self.is_initialized:
            raise RuntimeError("MCO Engine chưa được khởi tạo")
        
        if not self.pareto_solutions:
            logger.warning("Không có nghiệm Pareto nào để định hướng. Tạo các nghiệm Pareto trước.")
            self.generate_pareto_surface(num_samples=5)
        
        # Chuẩn hóa trọng số
        total_weight = sum(weights.values())
        if total_weight == 0:
            raise ValueError("Tổng trọng số phải lớn hơn 0")
        normalized_weights = {k: v / total_weight for k, v in weights.items()}
        
        # Tạo bản sao của tập mục tiêu với trọng số mới
        objective_info = self.objectives.get_objectives_info()
        obj_collection = ObjectiveCollection()
        
        for obj in objective_info:
            original_obj = self.objectives.objectives[obj["index"]]
            new_obj = type(original_obj)(**{k: getattr(original_obj, k) for k in original_obj.__dataclass_fields__})
            new_obj.is_enabled = True
            new_obj.weight = normalized_weights.get(obj["id"], 0.0)
            obj_collection.add_objective(new_obj)
        
        # Tìm nghiệm Pareto gần nhất để bắt đầu
        closest_solution = None
        min_distance = float('inf')
        
        for solution in self.pareto_solutions:
            if solution.status == ParetoSolutionStatus.INFEASIBLE:
                continue
            
            # Tính khoảng cách dựa trên trọng số
            distance = sum((normalized_weights.get(k, 0.0) - v) ** 2 for k, v in solution.weights.items())
            
            if distance < min_distance:
                min_distance = distance
                closest_solution = solution
        
        # Tạo động cơ tối ưu hóa với trọng số mới
        optimizer = OptimizationEngine(
            objectives=obj_collection,
            constraints=self.constraints,
            parameters=self.parameters,
            solver_name=self.base_optimizer
        )
        
        # Thực hiện tối ưu hóa
        logger.info("Đang tạo kế hoạch mới với trọng số: %s", 
                    {k: round(v, 3) for k, v in normalized_weights.items()})
        optimizer.set_structures(self.structures)
        
        # Bắt đầu từ nghiệm Pareto gần nhất nếu có
        if closest_solution is not None and closest_solution.dose_grid is not None:
            optimizer.set_dose_grid(closest_solution.dose_grid)
            
        if closest_solution is not None and closest_solution.control_parameters is not None:
            optimizer.set_control_parameters(closest_solution.control_parameters)
        else:
            if self.initial_dose_grid is not None:
                optimizer.set_dose_grid(self.initial_dose_grid)
                
            if self.initial_control_parameters is not None:
                optimizer.set_control_parameters(self.initial_control_parameters)
        
        results = optimizer.optimize()
        
        # Lưu trữ kết quả như một nghiệm Pareto
        objective_values = {}
        for obj in objective_info:
            o_obj = self.objectives.objectives[obj["index"]]
            value = o_obj.evaluate(optimizer.dose_grid, self.structures)
            objective_values[obj["id"]] = value
        
        solution = ParetoSolution(
            objective_values=objective_values,
            weights=normalized_weights,
            status=ParetoSolutionStatus.CANDIDATE,
            control_parameters=optimizer.control_parameters.copy() if optimizer.control_parameters is not None else None,
            dose_grid=optimizer.dose_grid.copy() if optimizer.dose_grid is not None else None,
            optimization_results=results,
            metadata={"type": "navigated"}
        )
        
        self.pareto_solutions.append(solution)
        self.current_navigating_weights = normalized_weights
        self._update_pareto_status()
        
        logger.info("Đã tạo kế hoạch mới với giá trị: %s", 
                    {k: round(v, 3) for k, v in objective_values.items()})
        
        return solution

    def select_solution(self, solution_index: int) -> Optional[ParetoSolution]:
        """
        Chọn một nghiệm Pareto cụ thể làm nghiệm được chọn.
        
        Args:
            solution_index: Chỉ số của nghiệm trong danh sách pareto_solutions
            
        Returns:
            Nghiệm được chọn hoặc None nếu chỉ số không hợp lệ
        """
        if not self.pareto_solutions or solution_index < 0 or solution_index >= len(self.pareto_solutions):
            logger.error("Chỉ số nghiệm không hợp lệ: %d", solution_index)
            return None
        
        # Đặt lại trạng thái của nghiệm được chọn trước đó nếu có
        if self.selected_solution is not None:
            self.selected_solution.status = ParetoSolutionStatus.CANDIDATE
            self._update_pareto_status()
        
        # Cập nhật nghiệm được chọn mới
        self.selected_solution = self.pareto_solutions[solution_index]
        self.selected_solution.status = ParetoSolutionStatus.SELECTED
        
        logger.info("Đã chọn nghiệm #%d với giá trị: %s", 
                    solution_index, {k: round(v, 3) for k, v in self.selected_solution.objective_values.items()})
        
        return self.selected_solution

    def interpolate_solutions(self, solution_indices: List[int], weights: List[float]) -> Optional[ParetoSolution]:
        """
        Nội suy giữa các nghiệm Pareto để tạo ra một nghiệm mới.
        
        Args:
            solution_indices: Chỉ số của các nghiệm được sử dụng để nội suy
            weights: Trọng số tương ứng cho từng nghiệm
            
        Returns:
            Nghiệm Pareto mới được tạo bằng nội suy hoặc None nếu có lỗi
        """
        if not self.pareto_solutions:
            logger.warning("Không có nghiệm Pareto nào để nội suy")
            return None
        
        if len(solution_indices) != len(weights):
            logger.error("Số lượng chỉ số nghiệm và trọng số phải bằng nhau")
            return None
        
        if abs(sum(weights) - 1.0) > 1e-6:
            logger.warning("Tổng trọng số phải bằng 1.0. Đang chuẩn hóa trọng số.")
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]
        
        # Lấy các nghiệm Pareto được chọn để nội suy
        solutions = []
        for idx in solution_indices:
            if idx < 0 or idx >= len(self.pareto_solutions):
                logger.error("Chỉ số nghiệm không hợp lệ: %d", idx)
                return None
            solutions.append(self.pareto_solutions[idx])
        
        # Nội suy tham số điều khiển và phân bố liều
        interpolated_control_params = None
        if all(s.control_parameters is not None for s in solutions):
            interpolated_control_params = np.zeros_like(solutions[0].control_parameters)
            for solution, w in zip(solutions, weights):
                interpolated_control_params += solution.control_parameters * w
        
        interpolated_dose = None
        if all(s.dose_grid is not None for s in solutions):
            interpolated_dose = solutions[0].dose_grid.copy()
            interpolated_dose.dose_array = np.zeros_like(solutions[0].dose_grid.dose_array)
            for solution, w in zip(solutions, weights):
                interpolated_dose.dose_array += solution.dose_grid.dose_array * w
        
        # Tính toán giá trị mục tiêu cho nghiệm nội suy mới
        objective_values = {}
        objective_info = self.objectives.get_objectives_info()
        
        for obj in objective_info:
            if interpolated_dose is not None:
                o_obj = self.objectives.objectives[obj["index"]]
                value = o_obj.evaluate(interpolated_dose, self.structures)
                objective_values[obj["id"]] = value
            else:
                # Nội suy giá trị mục tiêu nếu không có phân bố liều
                value = 0.0
                for solution, w in zip(solutions, weights):
                    if obj["id"] in solution.objective_values:
                        value += solution.objective_values[obj["id"]] * w
                objective_values[obj["id"]] = value
        
        # Tính trọng số hàm mục tiêu nội suy
        interp_weights = {}
        for obj in objective_info:
            value = 0.0
            for solution, w in zip(solutions, weights):
                if obj["id"] in solution.weights:
                    value += solution.weights[obj["id"]] * w
            interp_weights[obj["id"]] = value
        
        # Tạo nghiệm mới
        solution = ParetoSolution(
            objective_values=objective_values,
            weights=interp_weights,
            status=ParetoSolutionStatus.CANDIDATE,
            control_parameters=interpolated_control_params,
            dose_grid=interpolated_dose,
            optimization_results=None,
            metadata={"type": "interpolated", "source_indices": solution_indices, "interp_weights": weights}
        )
        
        self.pareto_solutions.append(solution)
        self._update_pareto_status()
        
        logger.info("Đã tạo kế hoạch nội suy mới với giá trị: %s", 
                    {k: round(v, 3) for k, v in objective_values.items()})
        
        return solution

    def get_objective_ranges(self) -> Dict[str, Tuple[float, float]]:
        """
        Lấy phạm vi giá trị của mỗi mục tiêu trên bề mặt Pareto.
        
        Returns:
            Dictionary chứa phạm vi (min, max) của mỗi mục tiêu
        """
        if not self.pareto_solutions:
            return {}
        
        ranges = {}
        objective_ids = set()
        
        # Lấy tất cả các ID mục tiêu
        for solution in self.pareto_solutions:
            for obj_id in solution.objective_values.keys():
                objective_ids.add(obj_id)
        
        # Tính min/max cho mỗi mục tiêu
        for obj_id in objective_ids:
            values = [solution.objective_values.get(obj_id, float('nan')) 
                     for solution in self.pareto_solutions 
                     if obj_id in solution.objective_values]
            
            values = [v for v in values if not np.isnan(v)]
            if values:
                ranges[obj_id] = (min(values), max(values))
        
        return ranges

    def visualize_pareto_front(self, objective_x: str, objective_y: str, 
                               ax=None, highlight_selected: bool = True,
                               show_dominated: bool = False) -> plt.Figure:
        """
        Trực quan hóa bề mặt Pareto trong không gian 2D của hai mục tiêu.
        
        Args:
            objective_x: ID của mục tiêu trên trục X
            objective_y: ID của mục tiêu trên trục Y
            ax: Matplotlib Axes để vẽ (nếu None, tạo mới)
            highlight_selected: Có làm nổi bật nghiệm được chọn hay không
            show_dominated: Có hiển thị các nghiệm bị chi phối hay không
            
        Returns:
            Matplotlib Figure chứa đồ thị
        """
        if not self.pareto_solutions:
            logger.warning("Không có nghiệm Pareto nào để trực quan hóa")
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.set_xlabel(f"Mục tiêu: {objective_x}")
            ax.set_ylabel(f"Mục tiêu: {objective_y}")
            ax.set_title("Bề mặt Pareto (không có dữ liệu)")
            return fig
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        else:
            fig = ax.figure
        
        # Lọc các nghiệm có đủ dữ liệu cho cả hai mục tiêu
        valid_solutions = [s for s in self.pareto_solutions 
                          if objective_x in s.objective_values and objective_y in s.objective_values]
        
        if not valid_solutions:
            logger.warning("Không có nghiệm nào chứa cả hai mục tiêu %s và %s", objective_x, objective_y)
            ax.set_xlabel(f"Mục tiêu: {objective_x}")
            ax.set_ylabel(f"Mục tiêu: {objective_y}")
            ax.set_title("Bề mặt Pareto (không có dữ liệu)")
            return fig
        
        # Phân loại nghiệm theo trạng thái
        optimal_sols = [s for s in valid_solutions if s.status == ParetoSolutionStatus.OPTIMAL]
        dominated_sols = [s for s in valid_solutions if s.status == ParetoSolutionStatus.DOMINATED]
        selected_sol = next((s for s in valid_solutions if s.status == ParetoSolutionStatus.SELECTED), None)
        
        # Vẽ các nghiệm tối ưu (bề mặt Pareto)
        if optimal_sols:
            x_values = [s.objective_values[objective_x] for s in optimal_sols]
            y_values = [s.objective_values[objective_y] for s in optimal_sols]
            ax.scatter(x_values, y_values, c='blue', s=80, alpha=0.7, label='Pareto Front')
            
            # Nối các điểm Pareto bằng đường
            if len(optimal_sols) > 1:
                try:
                    # Sắp xếp các điểm theo trục X để nối đúng
                    points = np.array(list(zip(x_values, y_values)))
                    sorted_indices = np.argsort(points[:, 0])
                    sorted_points = points[sorted_indices]
                    ax.plot(sorted_points[:, 0], sorted_points[:, 1], 'b--', alpha=0.5)
                except:
                    logger.warning("Không thể nối các điểm Pareto")
        
        # Vẽ các nghiệm bị chi phối
        if show_dominated and dominated_sols:
            x_values = [s.objective_values[objective_x] for s in dominated_sols]
            y_values = [s.objective_values[objective_y] for s in dominated_sols]
            ax.scatter(x_values, y_values, c='gray', s=50, alpha=0.5, label='Dominated Solutions')
        
        # Làm nổi bật nghiệm được chọn
        if highlight_selected and selected_sol is not None:
            ax.scatter(selected_sol.objective_values[objective_x], 
                      selected_sol.objective_values[objective_y],
                      c='red', s=100, edgecolors='black', linewidth=2, label='Selected Solution')
        
        # Thêm các chú thích và tiêu đề
        ax.set_xlabel(f"Mục tiêu: {objective_x}")
        ax.set_ylabel(f"Mục tiêu: {objective_y}")
        ax.set_title("Bề mặt Pareto")
        ax.grid(True, alpha=0.3)
        
        if optimal_sols or (show_dominated and dominated_sols) or (highlight_selected and selected_sol is not None):
            ax.legend()
        
        return fig

    def visualize_radar_chart(self, solution_indices: List[int] = None, 
                              normalize: bool = True, ax=None) -> plt.Figure:
        """
        Trực quan hóa các nghiệm Pareto dướidạng biểu đồ radar.
        
        Args:
            solution_indices: Chỉ số của các nghiệm cần hiển thị (nếu None, hiển thị tất cả nghiệm tối ưu)
            normalize: Có chuẩn hóa giá trị mục tiêu hay không
            ax: Matplotlib Axes để vẽ (nếu None, tạo mới)
            
        Returns:
            Matplotlib Figure chứa biểu đồ radar
        """
        if not self.pareto_solutions:
            logger.warning("Không có nghiệm Pareto nào để trực quan hóa")
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.set_title("Biểu đồ Radar (không có dữ liệu)")
            return fig
        
        # Lấy các nghiệm cần hiển thị
        if solution_indices is None:
            solutions = [s for s in self.pareto_solutions if s.status == ParetoSolutionStatus.OPTIMAL]
            if self.selected_solution is not None:
                solutions.append(self.selected_solution)
        else:
            solutions = [self.pareto_solutions[i] for i in solution_indices 
                        if 0 <= i < len(self.pareto_solutions)]
        
        if not solutions:
            logger.warning("Không có nghiệm nào để hiển thị")
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.set_title("Biểu đồ Radar (không có dữ liệu)")
            return fig
        
        # Lấy tất cả các ID mục tiêu từ các nghiệm
        objective_ids = set()
        for solution in solutions:
            objective_ids.update(solution.objective_values.keys())
        objective_ids = sorted(list(objective_ids))
        
        if not objective_ids:
            logger.warning("Không có mục tiêu nào để hiển thị")
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.set_title("Biểu đồ Radar (không có dữ liệu)")
            return fig
        
        # Chuẩn bị dữ liệu cho biểu đồ radar
        if normalize:
            # Tính min/max cho mỗi mục tiêu
            ranges = self.get_objective_ranges()
            
            # Chuẩn hóa dữ liệu (0: worst, 1: best) giả sử giá trị thấp hơn tốt hơn
            data = []
            for solution in solutions:
                values = []
                for obj_id in objective_ids:
                    if obj_id in solution.objective_values and obj_id in ranges:
                        min_val, max_val = ranges[obj_id]
                        if max_val > min_val:
                            # Đảo ngược thang đo vì giá trị thấp hơn tốt hơn
                            norm_val = 1 - (solution.objective_values[obj_id] - min_val) / (max_val - min_val)
                            values.append(max(0, min(1, norm_val)))
                        else:
                            values.append(0.5)
                    else:
                        values.append(0)
                data.append(values)
        else:
            # Sử dụng giá trị nguyên thủy
            data = []
            for solution in solutions:
                values = []
                for obj_id in objective_ids:
                    if obj_id in solution.objective_values:
                        values.append(solution.objective_values[obj_id])
                    else:
                        values.append(0)
                data.append(values)
        
        # Tạo biểu đồ radar
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        else:
            fig = ax.figure
        
        # Số lượng biến
        N = len(objective_ids)
        
        # Góc cho mỗi trục
        angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
        angles += angles[:1]  # Khép vòng tròn
        
        # Màu sắc và kiểu đường cho mỗi nghiệm
        colors = plt.cm.jet(np.linspace(0, 1, len(solutions)))
        
        # Vẽ cho từng nghiệm
        for i, (values, solution) in enumerate(zip(data, solutions)):
            values = values + values[:1]  # Khép vòng tròn
            
            label = f"Nghiệm #{i}" 
            if "type" in solution.metadata:
                label += f" ({solution.metadata['type']})"
            if solution.status == ParetoSolutionStatus.SELECTED:
                label += " (Đã chọn)"
            
            line_style = '-' if solution.status == ParetoSolutionStatus.SELECTED else '--'
            linewidth = 2 if solution.status == ParetoSolutionStatus.SELECTED else 1.5
            ax.plot(angles, values, linewidth=linewidth, linestyle=line_style, color=colors[i], label=label)
            ax.fill(angles, values, color=colors[i], alpha=0.1)
        
        # Thiết lập các trục và nhãn
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(objective_ids)
        
        # Thêm lưới và chú thích
        ax.grid(True)
        ax.set_title("So sánh các Nghiệm Pareto")
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))
        
        return fig

    def save_pareto_solutions(self, filename: str) -> bool:
        """
        Lưu tất cả các nghiệm Pareto vào file.
        
        Args:
            filename: Đường dẫn đến file để lưu
            
        Returns:
            True nếu lưu thành công, False nếu thất bại
        """
        try:
            import pickle
            
            # Lọc ra các thông tin cần lưu
            save_data = {
                "objectives_info": self.objectives.get_objectives_info(),
                "solutions": []
            }
            
            for solution in self.pareto_solutions:
                solution_data = {
                    "objective_values": solution.objective_values,
                    "weights": solution.weights,
                    "status": solution.status.name,
                    "metadata": solution.metadata
                }
                save_data["solutions"].append(solution_data)
            
            with open(filename, 'wb') as f:
                pickle.dump(save_data, f)
            
            logger.info("Đã lưu %d nghiệm Pareto vào %s", len(self.pareto_solutions), filename)
            return True
            
        except Exception as e:
            logger.error("Lỗi khi lưu nghiệm Pareto: %s", str(e))
            return False

    def load_pareto_solutions(self, filename: str) -> bool:
        """
        Tải các nghiệm Pareto từ file.
        
        Args:
            filename: Đường dẫn đến file để tải
            
        Returns:
            True nếu tải thành công, False nếu thất bại
        """
        try:
            import pickle
            
            with open(filename, 'rb') as f:
                save_data = pickle.load(f)
            
            # Khôi phục các nghiệm
            self.pareto_solutions = []
            for solution_data in save_data["solutions"]:
                solution = ParetoSolution(
                    objective_values=solution_data["objective_values"],
                    weights=solution_data["weights"],
                    status=ParetoSolutionStatus[solution_data["status"]],
                    metadata=solution_data["metadata"]
                )
                self.pareto_solutions.append(solution)
            
            logger.info("Đã tải %d nghiệm Pareto từ %s", len(self.pareto_solutions), filename)
            return True
            
        except Exception as e:
            logger.error("Lỗi khi tải nghiệm Pareto: %s", str(e))
            return False


def create_mco_engine(
    objectives: ObjectiveCollection,
    constraints: Optional[ConstraintCollection] = None,
    parameters: Optional[OptimizationParameters] = None,
    base_optimizer: str = "gradient_descent"
) -> MCOEngine:
    """
    Tạo một động cơ MCO mới.
    
    Args:
        objectives: Tập hợp các hàm mục tiêu để tối ưu hóa
        constraints: Tập hợp các ràng buộc phải thỏa mãn
        parameters: Tham số tối ưu hóa
        base_optimizer: Thuật toán tối ưu hóa nền tảng
        
    Returns:
        MCOEngine: Đối tượng động cơ MCO mới
    """
    return MCOEngine(objectives, constraints, parameters, base_optimizer)