"""
Module triển khai các thuật toán tối ưu hóa cho kế hoạch xạ trị trong hệ thống QuangTPS.

Module này cung cấp các bộ giải (solvers) khác nhau cho quá trình tối ưu hóa kế hoạch xạ trị,
bao gồm các thuật toán gradient-based và các phương pháp tìm kiếm toàn cục.
"""

import numpy as np
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from scipy.optimize import minimize, basinhopping
from dataclasses import dataclass, field

from quangtps.optimization.objectives import ObjectiveCollection
from quangtps.optimization.constraints import ConstraintCollection
from quangtps.dose.dose_grid import DoseGrid

logger = logging.getLogger(__name__)

class OptimizerBase(ABC):
    """Lớp cơ sở cho các thuật toán tối ưu hóa."""
    
    def __init__(
        self,
        objectives: ObjectiveCollection,
        constraints: ConstraintCollection,
        learning_rate: float = 0.01,
        max_iterations: int = 100,
        convergence_threshold: float = 1e-5,
        verbose: bool = True
    ):
        """
        Khởi tạo bộ tối ưu hóa.
        
        Args:
            objectives: Tập hợp các hàm mục tiêu
            constraints: Tập hợp các ràng buộc
            learning_rate: Tốc độ học cho các phương pháp gradient-based
            max_iterations: Số lần lặp tối đa
            convergence_threshold: Ngưỡng hội tụ
            verbose: Có in log chi tiết không
        """
        self.objectives = objectives
        self.constraints = constraints
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.verbose = verbose
        
        # Trạng thái tối ưu hóa
        self.current_iteration = 0
        self.best_objective_value = float('inf')
        self.current_objective_value = float('inf')
        self.objective_values_history = []
        self.best_parameters = None
        self.parameters = None
        
        # Callback và trạng thái dừng
        self.callbacks = []
        self.stop_flag = False
    
    def register_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """
        Đăng ký callback được gọi sau mỗi lần lặp.
        
        Args:
            callback: Hàm callback nhận context làm tham số
        """
        self.callbacks.append(callback)
    
    def _trigger_callbacks(self, context: Dict[str, Any]):
        """
        Gọi tất cả các callback đã đăng ký.
        
        Args:
            context: Ngữ cảnh cho callbacks
        """
        for callback in self.callbacks:
            try:
                callback(context)
            except Exception as e:
                logger.error(f"Lỗi trong callback: {e}")
    
    def set_parameters(self, parameters: np.ndarray):
        """
        Thiết lập tham số điều khiển ban đầu.
        
        Args:
            parameters: Mảng tham số điều khiển
        """
        self.parameters = parameters.copy()
        self.best_parameters = parameters.copy()
    
    def evaluate(self, dose_grid: DoseGrid, structures: Dict[str, np.ndarray]) -> float:
        """
        Đánh giá hàm mục tiêu và các ràng buộc.
        
        Args:
            dose_grid: Phân bố liều
            structures: Dictionary chứa các mặt nạ cấu trúc
            
        Returns:
            float: Giá trị hàm mục tiêu (bao gồm cả penalty từ ràng buộc)
        """
        # Tính giá trị hàm mục tiêu
        objective_results = self.objectives.evaluate_all(dose_grid, structures)
        objective_value = objective_results.get("total_cost", 0.0)
        
        # Kiểm tra các ràng buộc và tính penalty
        constraint_results = self.constraints.check_all(dose_grid, structures)
        constraint_penalty = 0.0
        
        for key, value in constraint_results.items():
            if key == "summary":
                continue
            
            if isinstance(value, dict) and not value.get("is_satisfied", True):
                violation = value.get("violation", 0.0)
                is_hard = value.get("is_hard_constraint", False)
                
                if is_hard:
                    # Áp dụng trọng số cao hơn cho ràng buộc bắt buộc
                    constraint_penalty += violation * 100.0
                else:
                    constraint_penalty += violation * 10.0
        
        # Tổng giá trị = objective + constraint_penalty
        return objective_value + constraint_penalty
    
    @abstractmethod
    def optimize(
        self,
        dose_grid: DoseGrid,
        structures: Dict[str, np.ndarray],
        initial_parameters: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, float, List[float]]:
        """
        Thực hiện quá trình tối ưu hóa.
        
        Args:
            dose_grid: Phân bố liều ban đầu
            structures: Dictionary chứa các mặt nạ cấu trúc
            initial_parameters: Tham số điều khiển ban đầu
            
        Returns:
            Tuple[np.ndarray, float, List[float]]: Tham số tối ưu, giá trị hàm mục tiêu, lịch sử mục tiêu
        """
        pass
    
    def stop(self):
        """Dừng quá trình tối ưu hóa."""
        self.stop_flag = True

class GradientDescentOptimizer(OptimizerBase):
    """Thuật toán Gradient Descent cho tối ưu hóa kế hoạch xạ trị."""
    
    def __init__(
        self,
        objectives: ObjectiveCollection,
        constraints: ConstraintCollection,
        learning_rate: float = 0.01,
        momentum: float = 0.9,
        adaptive_learning_rate: bool = True,
        learning_rate_decay: float = 0.95,
        min_learning_rate: float = 1e-6,
        max_iterations: int = 100,
        convergence_threshold: float = 1e-5,
        verbose: bool = True
    ):
        """
        Khởi tạo bộ tối ưu hóa Gradient Descent.
        
        Args:
            objectives: Tập hợp các hàm mục tiêu
            constraints: Tập hợp các ràng buộc
            learning_rate: Tốc độ học ban đầu
            momentum: Hệ số momentum
            adaptive_learning_rate: Có sử dụng tốc độ học thích ứng không
            learning_rate_decay: Hệ số giảm tốc độ học
            min_learning_rate: Tốc độ học tối thiểu
            max_iterations: Số lần lặp tối đa
            convergence_threshold: Ngưỡng hội tụ
            verbose: Có in log chi tiết không
        """
        super().__init__(
            objectives=objectives,
            constraints=constraints,
            learning_rate=learning_rate,
            max_iterations=max_iterations,
            convergence_threshold=convergence_threshold,
            verbose=verbose
        )
        self.momentum = momentum
        self.adaptive_learning_rate = adaptive_learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.min_learning_rate = min_learning_rate
        
        # Trạng thái tối ưu hóa
        self.velocity = None
        self.current_learning_rate = learning_rate
    
    def optimize(
        self,
        dose_grid: DoseGrid,
        structures: Dict[str, np.ndarray],
        initial_parameters: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, float, List[float]]:
        """
        Thực hiện tối ưu hóa bằng thuật toán Gradient Descent.
        
        Args:
            dose_grid: Phân bố liều ban đầu
            structures: Dictionary chứa các mặt nạ cấu trúc
            initial_parameters: Tham số điều khiển ban đầu
            
        Returns:
            Tuple[np.ndarray, float, List[float]]: Tham số tối ưu, giá trị hàm mục tiêu, lịch sử mục tiêu
        """
        # Khởi tạo
        if initial_parameters is not None:
            self.set_parameters(initial_parameters)
        elif self.parameters is None:
            # Tạo tham số mặc định nếu chưa được thiết lập
            self.set_parameters(np.ones_like(dose_grid.dose_array) * 0.5)
        
        # Khởi tạo velocity cho momentum
        self.velocity = np.zeros_like(self.parameters)
        
        # Khởi tạo trạng thái
        self.current_iteration = 0
        self.current_learning_rate = self.learning_rate
        self.objective_values_history = []
        self.stop_flag = False
        
        # Đánh giá giá trị mục tiêu ban đầu
        updated_dose_grid = dose_grid.copy()
        updated_dose_grid.dose_array = self.parameters.copy()
        self.current_objective_value = self.evaluate(updated_dose_grid, structures)
        self.best_objective_value = self.current_objective_value
        self.objective_values_history.append(self.current_objective_value)
        
        if self.verbose:
            logger.info(f"Bắt đầu tối ưu hóa Gradient Descent: cost ban đầu = {self.current_objective_value:.6f}")
        
        # Vòng lặp chính
        while self.current_iteration < self.max_iterations and not self.stop_flag:
            # Lưu giá trị mục tiêu trước đó
            previous_objective_value = self.current_objective_value
            
            # Tính gradient
            gradient = self._calculate_gradient(updated_dose_grid, structures)
            
            # Cập nhật velocity (momentum)
            self.velocity = self.momentum * self.velocity - self.current_learning_rate * gradient
            
            # Cập nhật tham số
            self.parameters += self.velocity
            
            # Đảm bảo tham số trong khoảng hợp lệ
            self.parameters = np.clip(self.parameters, 0, None)
            
            # Cập nhật dose_grid
            updated_dose_grid.dose_array = self.parameters.copy()
            
            # Đánh giá giá trị mục tiêu mới
            self.current_objective_value = self.evaluate(updated_dose_grid, structures)
            self.objective_values_history.append(self.current_objective_value)
            
            # Lưu trạng thái tốt nhất
            if self.current_objective_value < self.best_objective_value:
                self.best_objective_value = self.current_objective_value
                self.best_parameters = self.parameters.copy()
            
            # Gọi callbacks
            self._trigger_callbacks({
                "iteration": self.current_iteration,
                "previous_objective_value": previous_objective_value,
                "current_objective_value": self.current_objective_value,
                "best_objective_value": self.best_objective_value,
                "learning_rate": self.current_learning_rate,
                "gradient_norm": np.linalg.norm(gradient),
                "parameters": self.parameters
            })
            
            # In thông tin nếu cần
            if self.verbose and (self.current_iteration % 10 == 0 or self.current_iteration < 10):
                logger.info(f"Lần lặp {self.current_iteration}: cost = {self.current_objective_value:.6f}, "
                           f"lr = {self.current_learning_rate:.6f}")
            
            # Cập nhật tốc độ học nếu sử dụng adaptive learning rate
            if self.adaptive_learning_rate:
                self._update_learning_rate(previous_objective_value)
            
            # Kiểm tra điều kiện hội tụ
            if abs(previous_objective_value - self.current_objective_value) < self.convergence_threshold:
                if self.verbose:
                    logger.info(f"Hội tụ đạt được sau {self.current_iteration} lần lặp")
                break
            
            # Tăng số lần lặp
            self.current_iteration += 1
        
        # Kết thúc tối ưu hóa
        if self.verbose:
            logger.info(f"Tối ưu hóa kết thúc sau {self.current_iteration} lần lặp")
            logger.info(f"Giá trị mục tiêu tốt nhất: {self.best_objective_value:.6f}")
        
        # Sử dụng tham số tốt nhất
        self.parameters = self.best_parameters.copy()
        updated_dose_grid.dose_array = self.parameters.copy()
        
        return self.parameters, self.best_objective_value, self.objective_values_history
    
    def _calculate_gradient(self, dose_grid: DoseGrid, structures: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Tính gradient của hàm mục tiêu.
        
        Args:
            dose_grid: Phân bố liều hiện tại
            structures: Dictionary chứa các mặt nạ cấu trúc
            
        Returns:
            np.ndarray: Gradient
        """
        # Đây là ví dụ đơn giản sử dụng phương pháp sai phân hữu hạn
        # Trong thực tế, nên sử dụng gradient chính xác từ đạo hàm phân tích hoặc autodiff
        
        epsilon = 1e-6
        gradient = np.zeros_like(self.parameters)
        base_cost = self.current_objective_value
        
        # Trong thực tế, việc tính gradient đầy đủ cho mảng lớn là không hiệu quả
        # Đây chỉ là ví dụ đơn giản, cần triển khai phương pháp hiệu quả hơn
        
        # Giả lập gradient - thay thế bằng công thức thực tế
        gradient = np.random.normal(0, 1, self.parameters.shape)
        gradient = gradient / (np.linalg.norm(gradient) + 1e-10)
        
        return gradient
    
    def _update_learning_rate(self, previous_objective_value: float):
        """
        Cập nhật tốc độ học dựa trên sự thay đổi của giá trị mục tiêu.
        
        Args:
            previous_objective_value: Giá trị mục tiêu ở lần lặp trước
        """
        # Giảm tốc độ học nếu giá trị mục tiêu không giảm
        if self.current_objective_value >= previous_objective_value:
            self.current_learning_rate *= self.learning_rate_decay
            self.current_learning_rate = max(self.current_learning_rate, self.min_learning_rate)
        # Có thể thêm logic để tăng tốc độ học nếu cần

class LBFGSOptimizer(OptimizerBase):
    """Thuật toán L-BFGS cho tối ưu hóa kế hoạch xạ trị."""
    
    def __init__(
        self,
        objectives: ObjectiveCollection,
        constraints: ConstraintCollection,
        memory_size: int = 10,
        max_iterations: int = 100,
        convergence_threshold: float = 1e-5,
        verbose: bool = True
    ):
        """
        Khởi tạo bộ tối ưu hóa L-BFGS.
        
        Args:
            objectives: Tập hợp các hàm mục tiêu
            constraints: Tập hợp các ràng buộc
            memory_size: Số lượng vector gradient và vị trí lưu trữ
            max_iterations: Số lần lặp tối đa
            convergence_threshold: Ngưỡng hội tụ
            verbose: Có in log chi tiết không
        """
        super().__init__(
            objectives=objectives,
            constraints=constraints,
            max_iterations=max_iterations,
            convergence_threshold=convergence_threshold,
            verbose=verbose
        )
        self.memory_size = memory_size
    
    def optimize(
        self,
        dose_grid: DoseGrid,
        structures: Dict[str, np.ndarray],
        initial_parameters: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, float, List[float]]:
        """
        Thực hiện tối ưu hóa bằng thuật toán L-BFGS.
        
        Args:
            dose_grid: Phân bố liều ban đầu
            structures: Dictionary chứa các mặt nạ cấu trúc
            initial_parameters: Tham số điều khiển ban đầu
            
        Returns:
            Tuple[np.ndarray, float, List[float]]: Tham số tối ưu, giá trị hàm mục tiêu, lịch sử mục tiêu
        """
        # Khởi tạo
        if initial_parameters is not None:
            self.set_parameters(initial_parameters)
        elif self.parameters is None:
            # Tạo tham số mặc định nếu chưa được thiết lập
            self.set_parameters(np.ones_like(dose_grid.dose_array) * 0.5)
        
        # Tạo hàm mục tiêu và gradient cho scipy.optimize.minimize
        def objective_function(x):
            # Reshape tham số từ 1D thành dạng của dose_grid
            x_reshaped = x.reshape(dose_grid.dose_array.shape)
            
            # Cập nhật dose_grid
            updated_dose_grid = dose_grid.copy()
            updated_dose_grid.dose_array = x_reshaped
            
            # Tính giá trị mục tiêu
            return self.evaluate(updated_dose_grid, structures)
        
        def gradient_function(x):
            # Reshape tham số từ 1D thành dạng của dose_grid
            x_reshaped = x.reshape(dose_grid.dose_array.shape)
            
            # Cập nhật dose_grid
            updated_dose_grid = dose_grid.copy()
            updated_dose_grid.dose_array = x_reshaped
            
            # Tính gradient
            gradient = self._calculate_gradient(updated_dose_grid, structures)
            
            # Flatten gradient để phù hợp với scipy.optimize.minimize
            return gradient.flatten()
        
        # Khởi tạo lịch sử mục tiêu
        self.objective_values_history = []
        
        # Đánh giá giá trị mục tiêu ban đầu
        updated_dose_grid = dose_grid.copy()
        updated_dose_grid.dose_array = self.parameters.copy()
        self.current_objective_value = self.evaluate(updated_dose_grid, structures)
        self.best_objective_value = self.current_objective_value
        self.objective_values_history.append(self.current_objective_value)
        
        if self.verbose:
            logger.info(f"Bắt đầu tối ưu hóa L-BFGS: cost ban đầu = {self.current_objective_value:.6f}")
        
        # Đặt callback để cập nhật lịch sử và kiểm tra dừng
        def callback(x):
            # Reshape tham số từ 1D thành dạng của dose_grid
            x_reshaped = x.reshape(dose_grid.dose_array.shape)
            
            # Cập nhật dose_grid
            updated_dose_grid = dose_grid.copy()
            updated_dose_grid.dose_array = x_reshaped
            
            # Lưu giá trị mục tiêu trước đó
            previous_objective_value = self.current_objective_value
            
            # Tính giá trị mục tiêu mới
            self.current_objective_value = self.evaluate(updated_dose_grid, structures)
            self.objective_values_history.append(self.current_objective_value)
            
            # Lưu trạng thái tốt nhất
            if self.current_objective_value < self.best_objective_value:
                self.best_objective_value = self.current_objective_value
                self.best_parameters = x_reshaped.copy()
            
            # Cập nhật số lần lặp
            self.current_iteration += 1
            
            # Gọi callbacks
            self._trigger_callbacks({
                "iteration": self.current_iteration,
                "previous_objective_value": previous_objective_value,
                "current_objective_value": self.current_objective_value,
                "best_objective_value": self.best_objective_value,
                "parameters": x_reshaped
            })
            
            # In thông tin nếu cần
            if self.verbose and (self.current_iteration % 10 == 0 or self.current_iteration < 10):
                logger.info(f"Lần lặp {self.current_iteration}: cost = {self.current_objective_value:.6f}")
            
            # Kiểm tra dừng
            return self.stop_flag
        
        # Thực hiện tối ưu hóa bằng L-BFGS-B
        try:
            result = minimize(
                fun=objective_function,
                x0=self.parameters.flatten(),
                method='L-BFGS-B',
                jac=gradient_function,
                bounds=[(0, None) for _ in range(np.prod(self.parameters.shape))],
                options={
                    'maxiter': self.max_iterations,
                    'ftol': self.convergence_threshold,
                    'maxcor': self.memory_size
                },
                callback=callback
            )
            
            # Reshape kết quả thành dạng của dose_grid
            optimal_parameters = result.x.reshape(dose_grid.dose_array.shape)
            
            # Cập nhật parameters
            self.parameters = optimal_parameters.copy()
            
            if self.verbose:
                logger.info(f"Tối ưu hóa L-BFGS kết thúc: {result.message}")
                logger.info(f"Số lần lặp: {result.nit}")
                logger.info(f"Giá trị mục tiêu cuối cùng: {result.fun:.6f}")
            
            return optimal_parameters, result.fun, self.objective_values_history
        
        except Exception as e:
            logger.error(f"Lỗi trong quá trình tối ưu hóa L-BFGS: {e}")
            return self.best_parameters, self.best_objective_value, self.objective_values_history
    
    def _calculate_gradient(self, dose_grid: DoseGrid, structures: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Tính gradient của hàm mục tiêu.
        
        Args:
            dose_grid: Phân bố liều hiện tại
            structures: Dictionary chứa các mặt nạ cấu trúc
            
        Returns:
            np.ndarray: Gradient
        """
        # Giống như trong GradientDescentOptimizer
        epsilon = 1e-6
        gradient = np.zeros_like(dose_grid.dose_array)
        base_cost = self.evaluate(dose_grid, structures)
        
        # Giả lập gradient - thay thế bằng công thức thực tế
        gradient = np.random.normal(0, 1, dose_grid.dose_array.shape)
        gradient = gradient / (np.linalg.norm(gradient) + 1e-10)
        
        return gradient

class SimulatedAnnealingOptimizer(OptimizerBase):
    """Thuật toán Simulated Annealing cho tối ưu hóa kế hoạch xạ trị."""
    
    def __init__(
        self,
        objectives: ObjectiveCollection,
        constraints: ConstraintCollection,
        initial_temperature: float = 100.0,
        cooling_rate: float = 0.95,
        min_temperature: float = 1e-6,
        max_iterations: int = 1000,
        steps_per_temp: int = 10,
        step_size: float = 0.1,
        verbose: bool = True
    ):
        """
        Khởi tạo bộ tối ưu hóa Simulated Annealing.
        
        Args:
            objectives: Tập hợp các hàm mục tiêu
            constraints: Tập hợp các ràng buộc
            initial_temperature: Nhiệt độ ban đầu
            cooling_rate: Tốc độ làm mát (0 < cooling_rate < 1)
            min_temperature: Nhiệt độ tối thiểu
            max_iterations: Số lần lặp tối đa
            steps_per_temp: Số bước ở mỗi nhiệt độ
            step_size: Kích thước bước
            verbose: Có in log chi tiết không
        """
        super().__init__(
            objectives=objectives,
            constraints=constraints,
            max_iterations=max_iterations,
            verbose=verbose
        )
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.min_temperature = min_temperature
        self.steps_per_temp = steps_per_temp
        self.step_size = step_size
    
    def optimize(
        self,
        dose_grid: DoseGrid,
        structures: Dict[str, np.ndarray],
        initial_parameters: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, float, List[float]]:
        """
        Thực hiện tối ưu hóa bằng thuật toán Simulated Annealing.
        
        Args:
            dose_grid: Phân bố liều ban đầu
            structures: Dictionary chứa các mặt nạ cấu trúc
            initial_parameters: Tham số điều khiển ban đầu
            
        Returns:
            Tuple[np.ndarray, float, List[float]]: Tham số tối ưu, giá trị hàm mục tiêu, lịch sử mục tiêu
        """
        # Khởi tạo
        if initial_parameters is not None:
            self.set_parameters(initial_parameters)
        elif self.parameters is None:
            # Tạo tham số mặc định nếu chưa được thiết lập
            self.set_parameters(np.ones_like(dose_grid.dose_array) * 0.5)
        
        # Khởi tạo trạng thái
        self.current_iteration = 0
        self.objective_values_history = []
        self.stop_flag = False
        temperature = self.initial_temperature
        
        # Đánh giá giá trị mục tiêu ban đầu
        updated_dose_grid = dose_grid.copy()
        updated_dose_grid.dose_array = self.parameters.copy()
        self.current_objective_value = self.evaluate(updated_dose_grid, structures)
        self.best_objective_value = self.current_objective_value
        self.objective_values_history.append(self.current_objective_value)
        
        if self.verbose:
            logger.info(f"Bắt đầu tối ưu hóa Simulated Annealing: cost ban đầu = {self.current_objective_value:.6f}")
        
        # Vòng lặp chính
        while temperature > self.min_temperature and self.current_iteration < self.max_iterations and not self.stop_flag:
            # Thực hiện các bước ở nhiệt độ hiện tại
            for step in range(self.steps_per_temp):
                # Tạo giải pháp mới
                new_parameters = self._generate_neighbor(self.parameters, temperature)
                
                # Đảm bảo tham số trong khoảng hợp lệ
                new_parameters = np.clip(new_parameters, 0, None)
                
                # Cập nhật dose_grid
                updated_dose_grid.dose_array = new_parameters.copy()
                
                # Đánh giá giá trị mục tiêu mới
                new_objective_value = self.evaluate(updated_dose_grid, structures)
                
                # Tính delta
                delta = new_objective_value - self.current_objective_value
                
                # Quyết định chấp nhận giải pháp mới
                if delta < 0 or np.random.random() < np.exp(-delta / temperature):
                    # Chấp nhận giải pháp mới
                    self.parameters = new_parameters.copy()
                    self.current_objective_value = new_objective_value
                    
                    # Cập nhật trạng thái tốt nhất
                    if new_objective_value < self.best_objective_value:
                        self.best_objective_value = new_objective_value
                        self.best_parameters = new_parameters.copy()
                
                # Lưu giá trị mục tiêu hiện tại vào lịch sử
                self.objective_values_history.append(self.current_objective_value)
                
                # Tăng số lần lặp
                self.current_iteration += 1
                
                # Gọi callbacks
                self._trigger_callbacks({
                    "iteration": self.current_iteration,
                    "current_objective_value": self.current_objective_value,
                    "best_objective_value": self.best_objective_value,
                    "temperature": temperature,
                    "parameters": self.parameters
                })
                
                # Kiểm tra điều kiện dừng
                if self.current_iteration >= self.max_iterations or self.stop_flag:
                    break
            
            # In thông tin nếu cần
            if self.verbose and (self.current_iteration % 10 == 0 or self.current_iteration < 10):
                logger.info(f"Lần lặp {self.current_iteration}: cost = {self.current_objective_value:.6f}, "
                           f"T = {temperature:.6f}")
            
            # Giảm nhiệt độ
            temperature *= self.cooling_rate
        
        # Kết thúc tối ưu hóa
        if self.verbose:
            logger.info(f"Tối ưu hóa kết thúc sau {self.current_iteration} lần lặp")
            logger.info(f"Giá trị mục tiêu tốt nhất: {self.best_objective_value:.6f}")
        
        # Sử dụng tham số tốt nhất
        self.parameters = self.best_parameters.copy()
        updated_dose_grid.dose_array = self.parameters.copy()
        
        return self.parameters, self.best_objective_value, self.objective_values_history
    
    def _generate_neighbor(self, current_parameters: np.ndarray, temperature: float) -> np.ndarray:
        """
        Tạo giải pháp lân cận.
        
        Args:
            current_parameters: Tham số hiện tại
            temperature: Nhiệt độ hiện tại
            
        Returns:
            np.ndarray: Tham số mới
        """
        # Kích thước bước tỷ lệ với nhiệt độ
        step_size = self.step_size * temperature / self.initial_temperature
        
        # Tạo nhiễu ngẫu nhiên
        noise = np.random.normal(0, step_size, current_parameters.shape)
        
        # Tạo tham số mới
        new_parameters = current_parameters + noise
        
        return new_parameters

def create_optimizer(
    optimizer_type: str,
    objectives: ObjectiveCollection,
    constraints: ConstraintCollection,
    **kwargs
) -> OptimizerBase:
    """
    Tạo bộ tối ưu hóa dựa trên loại.
    
    Args:
        optimizer_type: Loại bộ tối ưu hóa ('gradient_descent', 'lbfgs', 'simulated_annealing')
        objectives: Tập hợp các hàm mục tiêu
        constraints: Tập hợp các ràng buộc
        **kwargs: Các tham số bổ sung
        
    Returns:
        OptimizerBase: Đối tượng bộ tối ưu hóa
        
    Raises:
        ValueError: Nếu optimizer_type không được hỗ trợ
    """
    if optimizer_type == "gradient_descent":
        return GradientDescentOptimizer(objectives, constraints, **kwargs)
    elif optimizer_type == "lbfgs":
        return LBFGSOptimizer(objectives, constraints, **kwargs)
    elif optimizer_type == "simulated_annealing":
        return SimulatedAnnealingOptimizer(objectives, constraints, **kwargs)
    else:
        raise ValueError(f"Loại bộ tối ưu hóa không được hỗ trợ: {optimizer_type}")

def optimize_plan(
    dose_grid: DoseGrid,
    structures: Dict[str, np.ndarray],
    objectives: ObjectiveCollection,
    constraints: ConstraintCollection,
    optimizer_type: str = "gradient_descent",
    optimizer_params: Dict[str, Any] = None,
    initial_parameters: Optional[np.ndarray] = None
) -> Tuple[DoseGrid, Dict[str, Any]]:
    """
    Thực hiện tối ưu hóa kế hoạch xạ trị.
    
    Args:
        dose_grid: Phân bố liều ban đầu
        structures: Dictionary chứa các mặt nạ cấu trúc
        objectives: Tập hợp các hàm mục tiêu
        constraints: Tập hợp các ràng buộc
        optimizer_type: Loại bộ tối ưu hóa
        optimizer_params: Tham số cho bộ tối ưu hóa
        initial_parameters: Tham số điều khiển ban đầu
        
    Returns:
        Tuple[DoseGrid, Dict[str, Any]]: Phân bố liều tối ưu và thông tin bổ sung
    """
    # Thiết lập tham số mặc định nếu không được cung cấp
    if optimizer_params is None:
        optimizer_params = {}
    
    # Tạo bộ tối ưu hóa
    optimizer = create_optimizer(optimizer_type, objectives, constraints, **optimizer_params)
    
    # Thực hiện tối ưu hóa
    optimal_parameters, optimal_value, objective_history = optimizer.optimize(
        dose_grid, structures, initial_parameters
    )
    
    # Tạo phân bố liều tối ưu
    optimal_dose_grid = dose_grid.copy()
    optimal_dose_grid.dose_array = optimal_parameters
    
    # Thông tin bổ sung
    info = {
        "optimal_value": optimal_value,
        "objective_history": objective_history,
        "num_iterations": len(objective_history),
        "optimizer_type": optimizer_type,
        "optimizer_params": optimizer_params,
        "constraint_violations": constraints.check_all(optimal_dose_grid, structures)
    }
    
    return optimal_dose_grid, info
