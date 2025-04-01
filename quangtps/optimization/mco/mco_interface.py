#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module giao diện cho Multi-Criteria Optimization (MCO) của QuangTPS.

Module này triển khai giao diện và các chức năng tối ưu đa tiêu chí,
mô phỏng theo tính năng MCO của Eclipse. Cho phép người dùng tối ưu hóa
kế hoạch xạ trị dựa trên nhiều tiêu chí khác nhau và khám phá
không gian lời giải trực quan.
"""

import os
import sys
import logging
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Any, Set, Union

# Import from QuangTPS modules
from quangtps.core.services import ServiceRegistry
from quangtps.planning.plan import Plan
from quangtps.optimization.objectives import ObjectiveFunction, ObjectiveCollection
from quangtps.optimization.constraints import Constraint, ConstraintCollection
from quangtps.segmentation.structures.structure import Structure
from quangtps.evaluation.dvh.dvh_calculation import DVHCalculator
from quangtps.evaluation.metrics.conformity import ConformityIndex
from quangtps.evaluation.metrics.homogeneity import HomogeneityIndex
from quangtps.evaluation.metrics.gradient import GradientIndex

logger = logging.getLogger(__name__)

class MCOSolution:
    """
    Lớp đại diện cho một lời giải trong không gian lời giải MCO.
    
    Lưu trữ thông tin về kế hoạch, giá trị của các hàm mục tiêu,
    trạng thái ràng buộc, và các thông tin liên quan khác.
    """
    
    def __init__(self, plan: Plan, objectives: Dict[str, float], 
                 constraints_satisfied: bool = True, metadata: Dict = None):
        """
        Khởi tạo một lời giải MCO.
        
        Parameters
        ----------
        plan : Plan
            Kế hoạch xạ trị
        objectives : Dict[str, float]
            Từ điển các giá trị hàm mục tiêu
        constraints_satisfied : bool, optional
            Có thỏa mãn các ràng buộc hay không
        metadata : Dict, optional
            Thông tin bổ sung về lời giải
        """
        self.plan = plan
        self.objectives = objectives
        self.constraints_satisfied = constraints_satisfied
        self.metadata = metadata if metadata else {}
        self.generation_time = None
        self.is_pareto_optimal = None
        self.rank = None
    
    def get_objective_value(self, objective_name: str) -> float:
        """
        Lấy giá trị của một hàm mục tiêu.
        
        Parameters
        ----------
        objective_name : str
            Tên của hàm mục tiêu
            
        Returns
        -------
        float
            Giá trị của hàm mục tiêu
        """
        return self.objectives.get(objective_name, float('nan'))
    
    def is_dominated_by(self, other_solution) -> bool:
        """
        Kiểm tra xem lời giải này có bị lời giải khác chi phối không.
        
        Parameters
        ----------
        other_solution : MCOSolution
            Lời giải khác
            
        Returns
        -------
        bool
            True nếu lời giải này bị chi phối, ngược lại False
        """
        # Chỉ xem xét các hàm mục tiêu có trong cả hai lời giải
        common_objectives = set(self.objectives.keys()) & set(other_solution.objectives.keys())
        
        # Kiểm tra xem other_solution có tốt hơn hoặc bằng ở tất cả các mục tiêu
        all_better_or_equal = True
        at_least_one_better = False
        
        for obj_name in common_objectives:
            # Giả sử giá trị thấp hơn là tốt hơn cho tất cả các mục tiêu
            # Có thể tùy chỉnh điều này dựa trên loại mục tiêu
            if other_solution.objectives[obj_name] > self.objectives[obj_name]:
                all_better_or_equal = False
                break
            if other_solution.objectives[obj_name] < self.objectives[obj_name]:
                at_least_one_better = True
        
        return all_better_or_equal and at_least_one_better


class MCOObjectiveSpace:
    """
    Lớp quản lý không gian mục tiêu trong MCO.
    
    Cung cấp phương thức để khám phá không gian Pareto,
    tạo biểu đồ trực quan, và tương tác với các lời giải.
    """
    
    def __init__(self):
        """Khởi tạo không gian mục tiêu."""
        self.solutions = []
        self.pareto_front = []
        self.objectives = {}  # Ánh xạ tên mục tiêu -> metadata
        self.current_solution_index = None
    
    def add_solution(self, solution: MCOSolution):
        """
        Thêm một lời giải vào không gian mục tiêu.
        
        Parameters
        ----------
        solution : MCOSolution
            Lời giải cần thêm
        """
        self.solutions.append(solution)
        self._update_pareto_front()
    
    def add_objective(self, name: str, metadata: Dict = None):
        """
        Thêm một hàm mục tiêu vào không gian mục tiêu.
        
        Parameters
        ----------
        name : str
            Tên của hàm mục tiêu
        metadata : Dict, optional
            Thông tin bổ sung về hàm mục tiêu
        """
        self.objectives[name] = metadata if metadata else {}
    
    def _update_pareto_front(self):
        """Cập nhật tập Pareto dựa trên tất cả các lời giải."""
        self.pareto_front = []
        
        # Đánh dấu các lời giải Pareto
        for i, solution in enumerate(self.solutions):
            is_pareto = True
            
            for other_solution in self.solutions:
                if other_solution is solution:
                    continue
                
                if solution.is_dominated_by(other_solution):
                    is_pareto = False
                    break
            
            solution.is_pareto_optimal = is_pareto
            if is_pareto:
                self.pareto_front.append(solution)
    
    def get_solution(self, index: int) -> Optional[MCOSolution]:
        """
        Lấy lời giải theo chỉ số.
        
        Parameters
        ----------
        index : int
            Chỉ số của lời giải
            
        Returns
        -------
        MCOSolution or None
            Lời giải nếu tồn tại, ngược lại None
        """
        if 0 <= index < len(self.solutions):
            return self.solutions[index]
        return None
    
    def get_current_solution(self) -> Optional[MCOSolution]:
        """
        Lấy lời giải hiện tại.
        
        Returns
        -------
        MCOSolution or None
            Lời giải hiện tại nếu tồn tại, ngược lại None
        """
        if self.current_solution_index is not None:
            return self.get_solution(self.current_solution_index)
        return None
    
    def set_current_solution(self, index: int):
        """
        Đặt lời giải hiện tại.
        
        Parameters
        ----------
        index : int
            Chỉ số của lời giải
        """
        if 0 <= index < len(self.solutions):
            self.current_solution_index = index
    
    def plot_pareto_front(self, objective_x: str, objective_y: str, 
                          ax=None, highlight_current=True):
        """
        Vẽ không gian Pareto 2D dựa trên hai hàm mục tiêu.
        
        Parameters
        ----------
        objective_x : str
            Tên của hàm mục tiêu trên trục x
        objective_y : str
            Tên của hàm mục tiêu trên trục y
        ax : matplotlib.axes.Axes, optional
            Trục để vẽ, nếu không sẽ tạo mới
        highlight_current : bool, optional
            Đánh dấu lời giải hiện tại
            
        Returns
        -------
        matplotlib.axes.Axes
            Trục đã vẽ
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 6))
        
        # Vẽ tất cả các lời giải
        x_values = []
        y_values = []
        pareto_x = []
        pareto_y = []
        
        for solution in self.solutions:
            x = solution.get_objective_value(objective_x)
            y = solution.get_objective_value(objective_y)
            
            if solution.is_pareto_optimal:
                pareto_x.append(x)
                pareto_y.append(y)
            else:
                x_values.append(x)
                y_values.append(y)
        
        # Vẽ các điểm không thuộc Pareto
        ax.scatter(x_values, y_values, color='gray', label='Non-Pareto', alpha=0.5)
        
        # Vẽ các điểm Pareto
        ax.scatter(pareto_x, pareto_y, color='blue', label='Pareto Front')
        
        # Nối các điểm Pareto
        if len(pareto_x) > 1:
            # Sắp xếp theo x để vẽ đường
            points = list(zip(pareto_x, pareto_y))
            points.sort(key=lambda p: p[0])
            sorted_x, sorted_y = zip(*points)
            ax.plot(sorted_x, sorted_y, 'b--', alpha=0.5)
        
        # Đánh dấu lời giải hiện tại
        if highlight_current and self.current_solution_index is not None:
            current = self.get_current_solution()
            if current:
                x = current.get_objective_value(objective_x)
                y = current.get_objective_value(objective_y)
                ax.scatter([x], [y], color='red', s=100, label='Current Solution', zorder=10)
        
        # Thiết lập trục
        ax.set_xlabel(objective_x)
        ax.set_ylabel(objective_y)
        ax.set_title(f"Pareto Front: {objective_x} vs {objective_y}")
        ax.grid(True)
        ax.legend()
        
        return ax


class MCONavigator:
    """
    Lớp điều hướng và tương tác với không gian lời giải MCO.
    
    Cung cấp phương thức để khám phá các lời giải khác nhau,
    tương tác với biểu đồ, và chọn lời giải tối ưu.
    """
    
    def __init__(self, objective_space: MCOObjectiveSpace):
        """
        Khởi tạo điều hướng MCO.
        
        Parameters
        ----------
        objective_space : MCOObjectiveSpace
            Không gian mục tiêu để điều hướng
        """
        self.objective_space = objective_space
        self.navigation_history = []
        self.current_history_index = -1
    
    def select_solution(self, index: int):
        """
        Chọn một lời giải làm lời giải hiện tại.
        
        Parameters
        ----------
        index : int
            Chỉ số của lời giải
        """
        solution = self.objective_space.get_solution(index)
        if solution:
            self.objective_space.set_current_solution(index)
            self._add_to_history(index)
    
    def interpolate_solutions(self, index1: int, index2: int, weight: float) -> Optional[int]:
        """
        Nội suy giữa hai lời giải để tạo lời giải mới.
        
        Parameters
        ----------
        index1 : int
            Chỉ số của lời giải thứ nhất
        index2 : int
            Chỉ số của lời giải thứ hai
        weight : float
            Trọng số từ 0 đến 1, 0 = hoàn toàn index1, 1 = hoàn toàn index2
            
        Returns
        -------
        int or None
            Chỉ số của lời giải mới nếu thành công, ngược lại None
        """
        # Triển khai phương thức nội suy trong MCO
        pass
    
    def move_slider(self, objective_name: str, target_value: float) -> Optional[int]:
        """
        Di chuyển thanh trượt để đạt giá trị mục tiêu mong muốn.
        
        Parameters
        ----------
        objective_name : str
            Tên của hàm mục tiêu
        target_value : float
            Giá trị mục tiêu mong muốn
            
        Returns
        -------
        int or None
            Chỉ số của lời giải mới nếu thành công, ngược lại None
        """
        # Triển khai phương thức điều chỉnh thanh trượt trong MCO
        pass
    
    def _add_to_history(self, index: int):
        """
        Thêm một lời giải vào lịch sử điều hướng.
        
        Parameters
        ----------
        index : int
            Chỉ số của lời giải
        """
        # Cắt bỏ lịch sử sau chỉ số hiện tại nếu đã có
        if self.current_history_index < len(self.navigation_history) - 1:
            self.navigation_history = self.navigation_history[:self.current_history_index + 1]
        
        self.navigation_history.append(index)
        self.current_history_index = len(self.navigation_history) - 1
    
    def undo(self) -> Optional[int]:
        """
        Quay lại lời giải trước đó trong lịch sử.
        
        Returns
        -------
        int or None
            Chỉ số của lời giải trước đó nếu tồn tại, ngược lại None
        """
        if self.current_history_index > 0:
            self.current_history_index -= 1
            index = self.navigation_history[self.current_history_index]
            self.objective_space.set_current_solution(index)
            return index
        return None
    
    def redo(self) -> Optional[int]:
        """
        Tiến tới lời giải tiếp theo trong lịch sử.
        
        Returns
        -------
        int or None
            Chỉ số của lời giải tiếp theo nếu tồn tại, ngược lại None
        """
        if self.current_history_index < len(self.navigation_history) - 1:
            self.current_history_index += 1
            index = self.navigation_history[self.current_history_index]
            self.objective_space.set_current_solution(index)
            return index
        return None


class MCOEngine:
    """
    Lớp chính cho Multi-Criteria Optimization trong QuangTPS.
    
    Kết hợp các thuật toán tối ưu, không gian mục tiêu, và
    giao diện điều hướng để cung cấp giải pháp MCO hoàn chỉnh.
    """
    
    def __init__(self, plan: Plan):
        """
        Khởi tạo MCO Engine.
        
        Parameters
        ----------
        plan : Plan
            Kế hoạch xạ trị cần tối ưu hóa
        """
        self.plan = plan
        self.objectives = ObjectiveCollection()
        self.constraints = ConstraintCollection()
        self.objective_space = MCOObjectiveSpace()
        self.navigator = MCONavigator(self.objective_space)
        
        # Lấy các dịch vụ cần thiết
        self.service_registry = ServiceRegistry.get_instance()
        self.optimization_engine = self.service_registry.get_service("OptimizationEngine")
        self.dvh_calculator = self.service_registry.get_service("DVHCalculator")
    
    def add_objective(self, objective: ObjectiveFunction, weight: float = 1.0, 
                      name: str = None, metadata: Dict = None):
        """
        Thêm một hàm mục tiêu vào MCO.
        
        Parameters
        ----------
        objective : ObjectiveFunction
            Hàm mục tiêu cần thêm
        weight : float, optional
            Trọng số ban đầu của hàm mục tiêu
        name : str, optional
            Tên của hàm mục tiêu, nếu None sẽ dùng tên của hàm
        metadata : Dict, optional
            Thông tin bổ sung về hàm mục tiêu
        """
        obj_name = name if name else objective.get_name()
        self.objectives.add_objective(objective, weight)
        
        # Thêm vào không gian mục tiêu
        metadata = metadata if metadata else {}
        metadata['weight'] = weight
        self.objective_space.add_objective(obj_name, metadata)
    
    def add_constraint(self, constraint: Constraint):
        """
        Thêm một ràng buộc vào MCO.
        
        Parameters
        ----------
        constraint : Constraint
            Ràng buộc cần thêm
        """
        self.constraints.add_constraint(constraint)
    
    def generate_initial_solutions(self, num_solutions: int = 10) -> List[MCOSolution]:
        """
        Tạo tập lời giải ban đầu cho MCO.
        
        Parameters
        ----------
        num_solutions : int, optional
            Số lời giải cần tạo
            
        Returns
        -------
        List[MCOSolution]
            Danh sách các lời giải ban đầu
        """
        # Triển khai thuật toán tạo lời giải ban đầu cho không gian Pareto
        solutions = []
        
        # Placeholder - cần triển khai thuật toán thực tế
        for i in range(num_solutions):
            # Tạo lời giải với các trọng số khác nhau
            weight_vector = self._generate_weight_vector(i, num_solutions)
            solution = self._generate_solution(weight_vector)
            if solution:
                solutions.append(solution)
                self.objective_space.add_solution(solution)
        
        return solutions
    
    def _generate_weight_vector(self, index: int, total: int) -> Dict[str, float]:
        """
        Tạo vector trọng số cho một lời giải.
        
        Parameters
        ----------
        index : int
            Chỉ số của lời giải
        total : int
            Tổng số lời giải
            
        Returns
        -------
        Dict[str, float]
            Từ điển ánh xạ tên hàm mục tiêu -> trọng số
        """
        # Tạo các trọng số khác nhau dựa trên chỉ số
        objective_names = list(self.objectives.get_objectives().keys())
        n_objectives = len(objective_names)
        
        if n_objectives == 0:
            return {}
        
        # Placeholder - cách đơn giản để tạo trọng số khác nhau
        weights = {}
        
        if n_objectives == 1:
            # Chỉ có một mục tiêu, luôn luôn trọng số 1.0
            weights[objective_names[0]] = 1.0
        elif n_objectives == 2:
            # Hai mục tiêu, thay đổi tuyến tính
            alpha = index / (total - 1) if total > 1 else 0.5
            weights[objective_names[0]] = 1.0 - alpha
            weights[objective_names[1]] = alpha
        else:
            # Nhiều mục tiêu, sử dụng phương pháp ngẫu nhiên
            raw_weights = np.random.random(n_objectives)
            sum_weights = sum(raw_weights)
            normalized_weights = raw_weights / sum_weights
            
            for i, name in enumerate(objective_names):
                weights[name] = normalized_weights[i]
        
        return weights
    
    def _generate_solution(self, weight_vector: Dict[str, float]) -> Optional[MCOSolution]:
        """
        Tạo một lời giải dựa trên vector trọng số.
        
        Parameters
        ----------
        weight_vector : Dict[str, float]
            Từ điển ánh xạ tên hàm mục tiêu -> trọng số
            
        Returns
        -------
        MCOSolution or None
            Lời giải nếu thành công, ngược lại None
        """
        # Thiết lập trọng số cho các hàm mục tiêu
        for name, weight in weight_vector.items():
            objective = self.objectives.get_objective(name)
            if objective:
                self.objectives.set_weight(name, weight)
        
        # Tối ưu hóa kế hoạch với vector trọng số hiện tại
        try:
            # Clone kế hoạch để không ảnh hưởng đến kế hoạch gốc
            optimized_plan = self.plan.clone()
            
            # Tối ưu hóa
            success = self.optimization_engine.optimize(
                plan=optimized_plan,
                objectives=self.objectives,
                constraints=self.constraints
            )
            
            if not success:
                logger.warning("Optimization failed for weight vector")
                return None
            
            # Tính toán giá trị của các hàm mục tiêu
            objective_values = {}
            constraints_satisfied = True
            
            for name in self.objectives.get_objectives().keys():
                objective = self.objectives.get_objective(name)
                if objective:
                    value = objective.evaluate(optimized_plan)
                    objective_values[name] = value
            
            # Kiểm tra các ràng buộc
            for constraint in self.constraints.get_constraints():
                if not constraint.is_satisfied(optimized_plan):
                    constraints_satisfied = False
                    break
            
            # Tạo lời giải MCO
            solution = MCOSolution(
                plan=optimized_plan,
                objectives=objective_values,
                constraints_satisfied=constraints_satisfied,
                metadata={"weights": weight_vector}
            )
            
            return solution
            
        except Exception as e:
            logger.error(f"Error generating MCO solution: {e}")
            return None
    
    def interpolate_solutions(self, solution1: MCOSolution, solution2: MCOSolution, 
                             alpha: float) -> Optional[MCOSolution]:
        """
        Nội suy giữa hai lời giải để tạo lời giải mới.
        
        Parameters
        ----------
        solution1 : MCOSolution
            Lời giải thứ nhất
        solution2 : MCOSolution
            Lời giải thứ hai
        alpha : float
            Tham số nội suy từ 0 đến 1, 0 = solution1, 1 = solution2
            
        Returns
        -------
        MCOSolution or None
            Lời giải nội suy nếu thành công, ngược lại None
        """
        # Triển khai phương thức nội suy trong MCO
        pass
    
    def navigate_by_dose_sculpting(self, structure: Structure, dose_level: float) -> Optional[MCOSolution]:
        """
        Điều hướng không gian lời giải bằng cách điều chỉnh liều.
        
        Parameters
        ----------
        structure : Structure
            Cấu trúc cần điều chỉnh liều
        dose_level : float
            Mức liều mong muốn
            
        Returns
        -------
        MCOSolution or None
            Lời giải mới nếu thành công, ngược lại None
        """
        # Triển khai phương thức điều hướng bằng cách điều chỉnh liều
        pass


def calculate_mco_metrics(solution: MCOSolution) -> Dict[str, float]:
    """
    Tính toán các chỉ số chất lượng cho một lời giải MCO.
    
    Parameters
    ----------
    solution : MCOSolution
        Lời giải cần tính toán chỉ số
        
    Returns
    -------
    Dict[str, float]
        Từ điển chứa các chỉ số chất lượng
    """
    metrics = {}
    
    # Lấy PTV và OARs từ kế hoạch
    plan = solution.plan
    
    # Tính toán các chỉ số conformity, homogeneity, gradient
    try:
        conformity_index = ConformityIndex()
        homogeneity_index = HomogeneityIndex()
        gradient_index = GradientIndex()
        
        # Tìm PTV chính
        ptv = None
        for structure in plan.get_structures():
            if "PTV" in structure.name.upper():
                ptv = structure
                break
        
        if ptv and hasattr(plan, 'dose_grid') and plan.dose_grid is not None:
            metrics['CI'] = conformity_index.calculate(plan, ptv)
            metrics['HI'] = homogeneity_index.calculate(plan, ptv)
            metrics['GI'] = gradient_index.calculate(plan, ptv)
    except Exception as e:
        logger.error(f"Error calculating MCO metrics: {e}")
    
    return metrics 