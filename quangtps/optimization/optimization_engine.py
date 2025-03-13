"""
Module điều phối quá trình tối ưu hóa kế hoạch xạ trị trong hệ thống QuangTPS.

Module này cung cấp các lớp và hàm để quản lý và thực hiện quá trình tối ưu hóa kế hoạch
xạ trị, sử dụng các mục tiêu và ràng buộc để tạo ra kế hoạch tối ưu.
"""

import numpy as np
import logging
import time
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
import threading
import queue
from enum import Enum, auto

from quangtps.optimization.objectives import ObjectiveBase, ObjectiveCollection
from quangtps.optimization.constraints import ConstraintBase, ConstraintCollection
from quangtps.dose.dose_grid import DoseGrid
from quangtps.evaluation.dvh import calculate_dvh

logger = logging.getLogger(__name__)

class OptimizationStatus(Enum):
    """Trạng thái của quá trình tối ưu hóa."""
    READY = auto()
    RUNNING = auto()
    PAUSED = auto()
    CONVERGED = auto()
    STOPPED = auto()
    FAILED = auto()
    MAX_ITERATIONS_REACHED = auto()
    MAX_TIME_REACHED = auto()

class OptimizationEvent(Enum):
    """Sự kiện trong quá trình tối ưu hóa."""
    ITERATION_COMPLETED = auto()
    OBJECTIVE_UPDATED = auto()
    OBJECTIVE_VALUE_REDUCED = auto()
    CONSTRAINT_SATISFIED = auto()
    CONSTRAINT_VIOLATED = auto()
    PARAMETER_CHANGED = auto()
    CONVERGENCE_REACHED = auto()
    ERROR_OCCURRED = auto()

@dataclass
class OptimizationParameters:
    """Tham số cho quá trình tối ưu hóa."""
    # Tham số chung
    max_iterations: int = 100
    max_time_seconds: float = 3600.0  # 1 giờ
    convergence_threshold: float = 1e-5
    checkpoint_interval: int = 10
    verbose: bool = True
    
    # Tham số thuật toán gradient descent
    learning_rate: float = 0.01
    momentum: float = 0.9
    use_adaptive_learning_rate: bool = True
    min_learning_rate: float = 1e-6
    learning_rate_decay: float = 0.95
    
    # Tham số BFGS/L-BFGS
    use_lbfgs: bool = False
    lbfgs_memory_size: int = 10
    
    # Tham số Simulated Annealing
    use_simulated_annealing: bool = False
    initial_temperature: float = 100.0
    cooling_rate: float = 0.95
    
    # Tham số DVH-specific
    dvh_constraint_weight: float = 10.0
    
    # Tham số VMAT-specific
    vmat_smoothness_weight: float = 1.0
    
    def __post_init__(self):
        """Xác thực các tham số sau khi khởi tạo."""
        if self.max_iterations <= 0:
            raise ValueError(f"max_iterations phải là số dương, nhận được: {self.max_iterations}")
        
        if self.max_time_seconds <= 0:
            raise ValueError(f"max_time_seconds phải là số dương, nhận được: {self.max_time_seconds}")
        
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate phải là số dương, nhận được: {self.learning_rate}")
        
        if self.momentum < 0 or self.momentum >= 1:
            raise ValueError(f"momentum phải nằm trong khoảng [0, 1), nhận được: {self.momentum}")
        
        if self.min_learning_rate <= 0:
            raise ValueError(f"min_learning_rate phải là số dương, nhận được: {self.min_learning_rate}")
        
        if self.learning_rate_decay <= 0 or self.learning_rate_decay >= 1:
            raise ValueError(f"learning_rate_decay phải nằm trong khoảng (0, 1), nhận được: {self.learning_rate_decay}")
        
        if self.convergence_threshold <= 0:
            raise ValueError(f"convergence_threshold phải là số dương, nhận được: {self.convergence_threshold}")

@dataclass
class OptimizationResults:
    """Kết quả của quá trình tối ưu hóa."""
    # Giá trị hàm mục tiêu
    final_objective_value: float
    initial_objective_value: float
    objective_values_history: List[float]
    
    # Thông tin quá trình
    num_iterations: int
    elapsed_time: float
    convergence_reached: bool
    termination_reason: str
    
    # Thông tin đánh giá
    constraint_violations: Dict[str, Any]
    dvh_values: Dict[str, Any]
    final_dose_grid: Optional[DoseGrid] = None
    
    # Thông tin về tham số điều khiển
    final_control_parameters: Optional[np.ndarray] = None
    
    def get_improvement_percentage(self) -> float:
        """Tính phần trăm cải thiện của giá trị hàm mục tiêu."""
        if self.initial_objective_value == 0:
            return 0.0
        
        improvement = self.initial_objective_value - self.final_objective_value
        return (improvement / self.initial_objective_value) * 100.0
    
    def get_summary(self) -> Dict[str, Any]:
        """Trả về tóm tắt ngắn gọn kết quả tối ưu hóa."""
        return {
            "initial_cost": self.initial_objective_value,
            "final_cost": self.final_objective_value,
            "improvement_percent": self.get_improvement_percentage(),
            "num_iterations": self.num_iterations,
            "elapsed_time": self.elapsed_time,
            "convergence_reached": self.convergence_reached,
            "termination_reason": self.termination_reason,
            "constraint_violations_count": len([v for v in self.constraint_violations.values() 
                                               if isinstance(v, dict) and v.get("is_satisfied") is False])
        }

class OptimizationEngine:
    """
    Động cơ tối ưu hóa cho kế hoạch xạ trị.
    
    Lớp này điều phối quá trình tối ưu hóa sử dụng các hàm mục tiêu và ràng buộc
    để tìm ra kế hoạch tối ưu.
    """
    
    def __init__(
        self,
        objectives: Optional[ObjectiveCollection] = None,
        constraints: Optional[ConstraintCollection] = None,
        parameters: Optional[OptimizationParameters] = None,
        solver_name: str = "gradient_descent"
    ):
        """
        Khởi tạo động cơ tối ưu hóa.
        
        Args:
            objectives: Tập hợp các hàm mục tiêu
            constraints: Tập hợp các ràng buộc
            parameters: Tham số tối ưu hóa
            solver_name: Tên thuật toán tối ưu hóa
        """
        self.objectives = objectives or ObjectiveCollection()
        self.constraints = constraints or ConstraintCollection()
        self.parameters = parameters or OptimizationParameters()
        self.solver_name = solver_name
        
        # Trạng thái tối ưu hóa
        self.status = OptimizationStatus.READY
        self.current_iteration = 0
        self.current_objective_value = float('inf')
        self.best_objective_value = float('inf')
        self.start_time = 0.0
        self.elapsed_time = 0.0
        
        # Chọn thuật toán tối ưu hóa
        self._select_optimizer()
        
        # Callback và event handlers
        self.callbacks = {}
        self.stop_flag = threading.Event()
        self.pause_flag = threading.Event()
        self.event_queue = queue.Queue()
        
        # Dữ liệu tối ưu hóa
        self.structures = {}
        self.dose_grid = None
        self.best_dose_grid = None
        self.control_parameters = None
        self.best_control_parameters = None
        
        # Lịch sử
        self.objective_values_history = []
        self.constraint_violations_history = []
    
    def _select_optimizer(self):
        """Chọn thuật toán tối ưu hóa dựa trên tên."""
        if self.solver_name == "gradient_descent":
            self._update_parameters = self._gradient_descent_update
        elif self.solver_name == "lbfgs":
            if not self.parameters.use_lbfgs:
                logger.warning("Đã chọn L-BFGS nhưng use_lbfgs=False. Bật tùy chọn.")
                self.parameters.use_lbfgs = True
            self._update_parameters = self._lbfgs_update
        elif self.solver_name == "simulated_annealing":
            if not self.parameters.use_simulated_annealing:
                logger.warning("Đã chọn Simulated Annealing nhưng use_simulated_annealing=False. Bật tùy chọn.")
                self.parameters.use_simulated_annealing = True
            self._update_parameters = self._simulated_annealing_update
        else:
            raise ValueError(f"Thuật toán không được hỗ trợ: {self.solver_name}")
    
    def set_initial_state(
        self,
        dose_grid: DoseGrid,
        structures: Dict[str, np.ndarray],
        control_parameters: Optional[np.ndarray] = None
    ):
        """
        Thiết lập trạng thái ban đầu cho quá trình tối ưu hóa.
        
        Args:
            dose_grid: Phân bố liều ban đầu
            structures: Dictionary chứa các mặt nạ cấu trúc
            control_parameters: Mảng tham số điều khiển (nếu có)
        """
        self.dose_grid = dose_grid.copy()
        self.best_dose_grid = dose_grid.copy()
        self.structures = structures.copy()
        
        if control_parameters is None:
            # Tạo tham số điều khiển ban đầu - điều này tùy thuộc vào ứng dụng cụ thể
            # (ví dụ: trọng số beam, fluence map, v.v.)
            # Đây chỉ là ví dụ đơn giản
            shape = self.dose_grid.dose_array.shape
            self.control_parameters = np.ones(shape) * 0.5
        else:
            self.control_parameters = control_parameters.copy()
        
        self.best_control_parameters = self.control_parameters.copy()
        
        # Reset trạng thái tối ưu hóa
        self.status = OptimizationStatus.READY
        self.current_iteration = 0
        self.objective_values_history = []
        self.constraint_violations_history = []
        
        # Tính giá trị hàm mục tiêu ban đầu
        self.current_objective_value = self._evaluate_objective()
        self.best_objective_value = self.current_objective_value
        self.objective_values_history.append(self.current_objective_value)
        
        # Tính mức vi phạm ràng buộc ban đầu
        self.constraint_violations_history.append(
            self.constraints.check_all(self.dose_grid, self.structures)
        )
        
        logger.info(f"Trạng thái ban đầu: cost = {self.current_objective_value:.6f}")
    
    def register_callback(self, event_type: OptimizationEvent, callback: Callable):
        """
        Đăng ký callback cho một loại sự kiện.
        
        Args:
            event_type: Loại sự kiện
            callback: Hàm callback, nhận context làm tham số
        """
        if event_type not in self.callbacks:
            self.callbacks[event_type] = []
        
        self.callbacks[event_type].append(callback)
    
    def _trigger_event(self, event_type: OptimizationEvent, context: Dict[str, Any] = None):
        """
        Kích hoạt một sự kiện và gọi các callback liên quan.
        
        Args:
            event_type: Loại sự kiện
            context: Ngữ cảnh sự kiện (các dữ liệu liên quan)
        """
        if context is None:
            context = {}
        
        # Thêm thông tin cơ bản vào context
        context.update({
            "iteration": self.current_iteration,
            "elapsed_time": time.time() - self.start_time,
            "objective_value": self.current_objective_value,
            "best_objective_value": self.best_objective_value,
            "status": self.status
        })
        
        # Đưa sự kiện vào queue
        self.event_queue.put((event_type, context))
        
        # Gọi các callback đã đăng ký
        if event_type in self.callbacks:
            for callback in self.callbacks[event_type]:
                try:
                    callback(context)
                except Exception as e:
                    logger.error(f"Lỗi trong callback cho {event_type}: {e}")
    
    def optimize(self) -> OptimizationResults:
        """
        Chạy quá trình tối ưu hóa.
        
        Returns:
            OptimizationResults: Kết quả tối ưu hóa
        """
        if self.status != OptimizationStatus.READY:
            logger.warning(f"Trạng thái hiện tại là {self.status}, reset trước khi tối ưu hóa.")
            return self._get_current_results("Trạng thái không hợp lệ")
        
        if self.dose_grid is None:
            raise ValueError("Chưa thiết lập trạng thái ban đầu.")
        
        # Khởi tạo
        self.status = OptimizationStatus.RUNNING
        self.stop_flag.clear()
        self.pause_flag.clear()
        self.start_time = time.time()
        initial_objective_value = self.current_objective_value
        
        logger.info("Bắt đầu tối ưu hóa...")
        
        # Vòng lặp tối ưu hóa chính
        while self.status == OptimizationStatus.RUNNING or self.status == OptimizationStatus.PAUSED:
            # Kiểm tra điều kiện dừng
            if self.stop_flag.is_set():
                self.status = OptimizationStatus.STOPPED
                break
            
            # Xử lý tạm dừng
            if self.pause_flag.is_set():
                self.status = OptimizationStatus.PAUSED
                time.sleep(0.1)  # Tránh chiếm dụng CPU quá nhiều
                continue
            
            # Kiểm tra điều kiện dừng theo thời gian
            current_time = time.time()
            self.elapsed_time = current_time - self.start_time
            if self.elapsed_time >= self.parameters.max_time_seconds:
                self.status = OptimizationStatus.MAX_TIME_REACHED
                break
            
            # Kiểm tra điều kiện dừng theo số lần lặp
            if self.current_iteration >= self.parameters.max_iterations:
                self.status = OptimizationStatus.MAX_ITERATIONS_REACHED
                break
            
            # Cập nhật tham số điều khiển
            try:
                self._update_parameters()
            except Exception as e:
                logger.error(f"Lỗi trong cập nhật tham số: {e}")
                self.status = OptimizationStatus.FAILED
                self._trigger_event(OptimizationEvent.ERROR_OCCURRED, {"error": str(e)})
                break
            
            # Cập nhật phân bố liều
            self._update_dose_grid()
            
            # Đánh giá hàm mục tiêu
            previous_objective_value = self.current_objective_value
            self.current_objective_value = self._evaluate_objective()
            self.objective_values_history.append(self.current_objective_value)
            
            # Kiểm tra ràng buộc
            constraint_results = self.constraints.check_all(self.dose_grid, self.structures)
            self.constraint_violations_history.append(constraint_results)
            
            # Trigger sự kiện hoàn thành lần lặp
            self._trigger_event(OptimizationEvent.ITERATION_COMPLETED, {
                "previous_value": previous_objective_value,
                "current_value": self.current_objective_value,
                "constraint_results": constraint_results
            })
            
            # Lưu trạng thái tốt nhất
            if self.current_objective_value < self.best_objective_value:
                self.best_objective_value = self.current_objective_value
                self.best_dose_grid = self.dose_grid.copy()
                self.best_control_parameters = self.control_parameters.copy()
                
                self._trigger_event(OptimizationEvent.OBJECTIVE_VALUE_REDUCED, {
                    "reduction": previous_objective_value - self.current_objective_value
                })
            
            # In thông tin nếu cần
            if self.parameters.verbose and (self.current_iteration % 10 == 0 or self.current_iteration < 10):
                logger.info(f"Lần lặp {self.current_iteration}: cost = {self.current_objective_value:.6f}")
            
            # Kiểm tra điều kiện hội tụ
            if (previous_objective_value - self.current_objective_value) < self.parameters.convergence_threshold:
                self.status = OptimizationStatus.CONVERGED
                self._trigger_event(OptimizationEvent.CONVERGENCE_REACHED, {})
                break
            
            # Tăng số lần lặp
            self.current_iteration += 1
        
        # Thời gian kết thúc
        end_time = time.time()
        self.elapsed_time = end_time - self.start_time
        
        # Sử dụng kết quả tốt nhất
        self.dose_grid = self.best_dose_grid
        self.control_parameters = self.best_control_parameters
        self.current_objective_value = self.best_objective_value
        
        # Tạo thông tin kết quả
        termination_reason = self._get_termination_reason()
        logger.info(f"Tối ưu hóa kết thúc: {termination_reason}")
        logger.info(f"  Số lần lặp: {self.current_iteration}")
        logger.info(f"  Thời gian: {self.elapsed_time:.2f} giây")
        logger.info(f"  Cải thiện: {initial_objective_value:.6f} -> {self.best_objective_value:.6f}")
        
        # Trả về kết quả
        return self._get_current_results(termination_reason)
    
    def stop(self):
        """Dừng quá trình tối ưu hóa."""
        self.stop_flag.set()
    
    def pause(self):
        """Tạm dừng quá trình tối ưu hóa."""
        self.pause_flag.set()
    
    def resume(self):
        """Tiếp tục quá trình tối ưu hóa sau khi tạm dừng."""
        self.pause_flag.clear()
        self.status = OptimizationStatus.RUNNING
    
    def _get_termination_reason(self) -> str:
        """Trả về lý do kết thúc dựa trên trạng thái hiện tại."""
        if self.status == OptimizationStatus.CONVERGED:
            return "Hội tụ đạt được"
        elif self.status == OptimizationStatus.MAX_ITERATIONS_REACHED:
            return f"Đạt số lần lặp tối đa ({self.parameters.max_iterations})"
        elif self.status == OptimizationStatus.MAX_TIME_REACHED:
            return f"Đạt thời gian tối đa ({self.parameters.max_time_seconds}s)"
        elif self.status == OptimizationStatus.STOPPED:
            return "Bị dừng bởi người dùng"
        elif self.status == OptimizationStatus.FAILED:
            return "Tối ưu hóa thất bại do lỗi"
        else:
            return f"Lý do không xác định: {self.status}"
    
    def _evaluate_objective(self) -> float:
        """
        Đánh giá hàm mục tiêu tổng thể.
        
        Returns:
            float: Giá trị hàm mục tiêu
        """
        # Sử dụng ObjectiveCollection để đánh giá tất cả các hàm mục tiêu
        objective_results = self.objectives.evaluate_all(self.dose_grid, self.structures)
        
        # Kiểm tra xem có vi phạm ràng buộc không và thêm vào penalty nếu cần
        constraint_results = self.constraints.check_all(self.dose_grid, self.structures)
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
        
        # Tổng giá trị mục tiêu = objective + constraint_penalty
        return objective_results.get("total_cost", 0.0) + constraint_penalty
    
    def _update_dose_grid(self):
        """
        Cập nhật phân bố liều từ tham số điều khiển.
        
        Phương pháp cụ thể phụ thuộc vào ứng dụng. Đây là phiên bản đơn giản.
        """
        # Đây là ví dụ đơn giản, trong thực tế cần triển khai phương pháp cụ thể
        # dựa trên mô hình vật lý, cách tạo liều từ các tham số điều khiển, v.v.
        self.dose_grid.dose_array = self.control_parameters
    
    def _gradient_descent_update(self):
        """Cập nhật tham số điều khiển bằng thuật toán gradient descent."""
        # Tính gradient
        gradient = self._calculate_gradient()
        
        # Áp dụng gradient descent
        learning_rate = self.parameters.learning_rate
        self.control_parameters -= learning_rate * gradient
        
        # Đảm bảo tham số trong khoảng hợp lệ
        self.control_parameters = np.clip(self.control_parameters, 0, None)
    
    def _lbfgs_update(self):
        """Cập nhật tham số điều khiển bằng thuật toán L-BFGS."""
        # Phiên bản đơn giản, thực tế cần triển khai đầy đủ thuật toán L-BFGS
        # Đây chỉ là giả lập cho mục đích minh họa
        gradient = self._calculate_gradient()
        self.control_parameters -= self.parameters.learning_rate * gradient
        self.control_parameters = np.clip(self.control_parameters, 0, None)
    
    def _simulated_annealing_update(self):
        """Cập nhật tham số điều khiển bằng thuật toán Simulated Annealing."""
        # Phiên bản đơn giản, thực tế cần triển khai đầy đủ thuật toán SA
        # Đây chỉ là giả lập cho mục đích minh họa
        temperature = self.parameters.initial_temperature * (self.parameters.cooling_rate ** self.current_iteration)
        random_perturbation = np.random.normal(0, temperature, self.control_parameters.shape)
        self.control_parameters += random_perturbation
        self.control_parameters = np.clip(self.control_parameters, 0, None)
    
    def _calculate_gradient(self) -> np.ndarray:
        """
        Tính gradient của hàm mục tiêu theo tham số điều khiển.
        
        Returns:
            np.ndarray: Gradient
        """
        # Đây là ví dụ đơn giản sử dụng phương pháp sai phân hữu hạn
        # Trong thực tế, nên sử dụng gradient chính xác từ đạo hàm phân tích hoặc autodiff
        
        epsilon = 1e-6
        gradient = np.zeros_like(self.control_parameters)
        base_cost = self._evaluate_objective()
        
        # Tính gradient cho từng tham số
        # Lưu ý: Phương pháp này không hiệu quả cho tham số nhiều chiều
        # Trong thực tế, nên sử dụng phương pháp hiệu quả hơn
        
        # Giả lập gradient - thay thế bằng công thức thực tế
        gradient = np.random.normal(0, 1, self.control_parameters.shape)
        gradient = gradient / (np.linalg.norm(gradient) + 1e-10)
        
        return gradient
    
    def _get_current_results(self, termination_reason: str) -> OptimizationResults:
        """
        Tạo đối tượng OptimizationResults từ trạng thái hiện tại.
        
        Args:
            termination_reason: Lý do kết thúc
            
        Returns:
            OptimizationResults: Kết quả tối ưu hóa
        """
        # Tính DVH cho các cấu trúc
        dvh_values = {}
        for name, mask in self.structures.items():
            dvh = calculate_dvh(
                dose_array=self.dose_grid.dose_array,
                structure_mask=mask,
                volume_type='relative'
            )
            dvh_values[name] = dvh
        
        # Kiểm tra ràng buộc
        constraint_violations = self.constraints.check_all(self.dose_grid, self.structures)
        
        # Trả về kết quả
        return OptimizationResults(
            final_objective_value=self.current_objective_value,
            initial_objective_value=self.objective_values_history[0] if self.objective_values_history else float('inf'),
            objective_values_history=self.objective_values_history,
            num_iterations=self.current_iteration,
            elapsed_time=self.elapsed_time,
            convergence_reached=self.status == OptimizationStatus.CONVERGED,
            termination_reason=termination_reason,
            constraint_violations=constraint_violations,
            dvh_values=dvh_values,
            final_dose_grid=self.dose_grid,
            final_control_parameters=self.control_parameters
        )

def create_engine(
    objectives: Optional[ObjectiveCollection] = None,
    constraints: Optional[ConstraintCollection] = None,
    parameters: Optional[OptimizationParameters] = None,
    solver_name: str = "gradient_descent"
) -> OptimizationEngine:
    """
    Tạo một đối tượng OptimizationEngine mới.
    
    Args:
        objectives: Tập hợp các hàm mục tiêu
        constraints: Tập hợp các ràng buộc
        parameters: Tham số tối ưu hóa
        solver_name: Tên thuật toán tối ưu hóa
        
    Returns:
        OptimizationEngine: Đối tượng động cơ tối ưu hóa mới
    """
    return OptimizationEngine(objectives, constraints, parameters, solver_name)
