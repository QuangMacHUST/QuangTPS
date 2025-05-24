#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module tối ưu hóa kế hoạch xạ trị cho QuangTPS.

Module này triển khai các thuật toán để tối ưu hóa kế hoạch xạ trị
dựa trên các mục tiêu và ràng buộc lâm sàng.
"""

import logging
import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
import copy

from quangtps.optimization.objectives import ObjectiveFunction, Objective
from quangtps.dose.dose_calculation import DoseCalculator

logger = logging.getLogger(__name__)


class PlanOptimizer:
    """
    Lớp tối ưu hóa kế hoạch điều trị.

    Lớp này triển khai các thuật toán tối ưu hóa kế hoạch xạ trị
    dựa trên các mục tiêu lâm sàng.
    """

    def __init__(self, plan=None, dose_calculator=None):
        """
        Khởi tạo bộ tối ưu hóa kế hoạch.

        Args:
            plan: Kế hoạch điều trị cần tối ưu (tùy chọn)
            dose_calculator: Bộ tính toán liều (tùy chọn)
        """
        self.plan = plan
        self.dose_calculator = dose_calculator or DoseCalculator()

        # Khởi tạo danh sách mục tiêu
        self.objectives = []

        # Theo dõi tiến độ
        self.progress_callback = None

        # Theo dõi lịch sử tối ưu hóa
        self.iteration_history = []
        self.objective_history = []

        # Trạng thái
        self.is_optimizing = False
        self.iteration = 0
        self.best_iteration = 0
        self.best_objective_value = float("inf")

        logger.info("Khởi tạo PlanOptimizer")

    def set_plan(self, plan) -> None:
        """
        Thiết lập kế hoạch cần tối ưu.

        Args:
            plan: Kế hoạch điều trị
        """
        self.plan = plan

    def add_objective(self, objective: Union[ObjectiveFunction, Objective]) -> None:
        """
        Thêm một mục tiêu tối ưu hóa.

        Args:
            objective: Mục tiêu tối ưu hóa
        """
        if not isinstance(objective, (ObjectiveFunction, Objective)):
            logger.error(f"Loại mục tiêu không hợp lệ: {type(objective)}")
            return

        self.objectives.append(objective)
        logger.info(f"Đã thêm mục tiêu: {objective}")

    def set_objectives(
        self, objectives: List[Union[ObjectiveFunction, Objective]]
    ) -> None:
        """
        Đặt lại toàn bộ danh sách mục tiêu.

        Args:
            objectives: Danh sách mục tiêu tối ưu hóa
        """
        self.objectives = []
        for obj in objectives:
            self.add_objective(obj)

        logger.info(f"Đã thiết lập {len(self.objectives)} mục tiêu")

    def get_objectives(self) -> List[Union[ObjectiveFunction, Objective]]:
        """
        Lấy danh sách các mục tiêu hiện tại.

        Returns:
            Danh sách mục tiêu
        """
        return self.objectives

    def set_progress_callback(
        self, callback: Callable[[int, Dict[str, float]], None]
    ) -> None:
        """
        Thiết lập callback theo dõi tiến độ.

        Args:
            callback: Hàm callback nhận (iteration, metrics)
        """
        self.progress_callback = callback

    def optimize(
        self, max_iterations: int = 100, convergence_threshold: float = 1e-4
    ) -> bool:
        """
        Tối ưu hóa kế hoạch dựa trên các mục tiêu.

        Args:
            max_iterations: Số lần lặp tối đa
            convergence_threshold: Ngưỡng hội tụ

        Returns:
            True nếu tối ưu hóa thành công, False nếu không
        """
        if not self.plan:
            logger.error("Không có kế hoạch nào để tối ưu hóa")
            return False

        if not self.objectives:
            logger.error("Không có mục tiêu nào để tối ưu hóa")
            return False

        # Đánh dấu bắt đầu
        self.is_optimizing = True
        start_time = time.time()

        try:
            # Khởi tạo lịch sử
            self.iteration_history = []
            self.objective_history = []

            # Trạng thái
            self.iteration = 0
            self.best_iteration = 0
            self.best_objective_value = float("inf")

            # Tính toán liều ban đầu
            if not self.plan.has_dose():
                logger.info("Tính toán liều ban đầu...")
                self.dose_calculator.calculate(self.plan)

            # Triển khai quá trình tối ưu lặp
            prev_objective_value = float("inf")

            for i in range(max_iterations):
                self.iteration = i + 1

                # Cập nhật tham số kế hoạch
                self._update_plan_parameters()

                # Tính toán lại liều
                logger.debug(f"Tính toán lại liều ở lần lặp {self.iteration}...")
                self.dose_calculator.calculate(self.plan)

                # Đánh giá hàm mục tiêu
                objective_value, objective_values = self._evaluate_objectives()

                # Lưu vào lịch sử
                self.iteration_history.append(i + 1)
                self.objective_history.append(objective_value)

                # Kiểm tra giá trị tốt nhất
                if objective_value < self.best_objective_value:
                    self.best_objective_value = objective_value
                    self.best_iteration = i + 1
                    # Lưu trạng thái tốt nhất
                    self._save_best_state()

                # Gọi callback
                if self.progress_callback:
                    metrics = {
                        "objective_value": objective_value,
                        "best_value": self.best_objective_value,
                        **objective_values,
                    }
                    self.progress_callback(i + 1, metrics)

                # Kiểm tra hội tụ
                improvement = abs(prev_objective_value - objective_value)
                relative_improvement = improvement / (prev_objective_value + 1e-10)

                if relative_improvement < convergence_threshold and i > 10:
                    logger.info(f"Tối ưu hóa hội tụ sau {i + 1} lần lặp")
                    break

                prev_objective_value = objective_value

            # Khôi phục trạng thái tốt nhất
            self._restore_best_state()

            # Tính toán liều cuối cùng
            self.dose_calculator.calculate(self.plan)

            elapsed = time.time() - start_time
            logger.info(
                f"Tối ưu hóa hoàn thành sau {elapsed:.1f}s, {self.iteration} lần lặp"
            )

            return True

        except Exception as e:
            logger.error(f"Lỗi trong quá trình tối ưu hóa: {e}")
            return False

        finally:
            self.is_optimizing = False

    def _update_plan_parameters(self) -> None:
        """
        Cập nhật các tham số kế hoạch dựa trên gradient.

        Trong thuật toán tối ưu thực tế, đây là nơi ta cập nhật
        các trọng số của lá, hướng chùm tia, v.v.
        """
        # Placeholder - trong triển khai thực tế sẽ cập nhật
        # các tham số của kế hoạch dựa trên gradient

        # Ví dụ, nếu kế hoạch có các trọng số của lá
        if hasattr(self.plan, "leaf_weights"):
            # Tính gradient
            gradient = self._calculate_gradient()

            # Cập nhật trọng số (gradient descent với learning rate 0.01)
            learning_rate = 0.01
            self.plan.leaf_weights -= learning_rate * gradient

            # Đảm bảo các trọng số nằm trong khoảng hợp lệ [0, 1]
            self.plan.leaf_weights = np.clip(self.plan.leaf_weights, 0, 1)

    def _calculate_gradient(self) -> np.ndarray:
        """
        Tính gradient của hàm mục tiêu đối với các tham số kế hoạch.

        Returns:
            Ma trận gradient
        """
        # Placeholder - trong triển khai thực tế sẽ tính toán
        # gradient thực sự dựa trên các mục tiêu

        # Ví dụ đơn giản, tạo gradient ngẫu nhiên
        if hasattr(self.plan, "leaf_weights"):
            return np.random.randn(*self.plan.leaf_weights.shape) * 0.1

        return np.array([])

    def _evaluate_objectives(self) -> Tuple[float, Dict[str, float]]:
        """
        Đánh giá tất cả các hàm mục tiêu.

        Returns:
            Tuple (tổng giá trị mục tiêu, từ điển các giá trị riêng)
        """
        total_value = 0.0
        objective_values = {}

        for i, obj in enumerate(self.objectives):
            if hasattr(obj, "evaluate"):
                value = obj.evaluate(self.plan.dose)
            else:
                # Sử dụng mô phỏng cho mục tiêu không có phương thức evaluate
                value = 100.0 / (self.iteration + 1) * np.random.random()

            # Tên mục tiêu dựa trên loại và cấu trúc
            obj_name = f"{obj.type.name if hasattr(obj, 'type') else 'Unknown'}_{getattr(obj, 'structure_name', f'Obj{i}')}"
            objective_values[obj_name] = value

            # Thêm vào tổng
            total_value += value

        return total_value, objective_values

    def _save_best_state(self) -> None:
        """Lưu trạng thái tốt nhất của kế hoạch."""
        # Trong triển khai thực tế, ta sẽ lưu toàn bộ trạng thái kế hoạch
        self.best_plan_state = copy.deepcopy(self.plan)

    def _restore_best_state(self) -> None:
        """Khôi phục trạng thái tốt nhất của kế hoạch."""
        # Trong triển khai thực tế, ta sẽ khôi phục toàn bộ trạng thái
        if hasattr(self, "best_plan_state"):
            # Khôi phục các tham số quan trọng
            if hasattr(self.best_plan_state, "leaf_weights") and hasattr(
                self.plan, "leaf_weights"
            ):
                self.plan.leaf_weights = copy.deepcopy(
                    self.best_plan_state.leaf_weights
                )


class DoseFunctionBasedOptimizer(PlanOptimizer):
    """
    Bộ tối ưu hóa dựa trên hàm liều.

    Lớp con này triển khai thuật toán tối ưu hóa sử dụng
    ma trận hàm liều để tăng tốc độ tính toán.
    """

    def __init__(self, plan=None, dose_calculator=None):
        """Khởi tạo bộ tối ưu hóa hàm liều."""
        super().__init__(plan, dose_calculator)
        self.dose_matrix = None

    def optimize(
        self, max_iterations: int = 100, convergence_threshold: float = 1e-4
    ) -> bool:
        """
        Tối ưu hóa sử dụng ma trận hàm liều.

        Args:
            max_iterations: Số lần lặp tối đa
            convergence_threshold: Ngưỡng hội tụ

        Returns:
            True nếu tối ưu hóa thành công, False nếu không
        """
        if not self.plan:
            logger.error("Không có kế hoạch nào để tối ưu hóa")
            return False

        # Tính toán ma trận hàm liều
        logger.info("Tính toán ma trận hàm liều...")
        self._calculate_dose_influence_matrix()

        # Gọi thuật toán tối ưu của lớp cha
        return super().optimize(max_iterations, convergence_threshold)

    def _calculate_dose_influence_matrix(self) -> None:
        """Tính toán ma trận hàm liều."""
        # Placeholder - trong triển khai thực tế sẽ tính toán
        # ma trận hàm liều cho kế hoạch
        self.dose_matrix = np.random.rand(100, 100, 100)

    def _update_plan_parameters(self) -> None:
        """
        Cập nhật tham số kế hoạch sử dụng ma trận hàm liều.
        """
        # Tính gradient sử dụng ma trận hàm liều
        gradient = self._calculate_gradient_with_dose_matrix()

        # Cập nhật tham số
        if hasattr(self.plan, "leaf_weights"):
            learning_rate = 0.01
            self.plan.leaf_weights -= learning_rate * gradient
            self.plan.leaf_weights = np.clip(self.plan.leaf_weights, 0, 1)

    def _calculate_gradient_with_dose_matrix(self) -> np.ndarray:
        """
        Tính gradient sử dụng ma trận hàm liều.

        Returns:
            Gradient
        """
        # Placeholder - sẽ triển khai thuật toán thực tế
        if hasattr(self.plan, "leaf_weights"):
            return np.random.randn(*self.plan.leaf_weights.shape) * 0.05

        return np.array([])


class MCOOptimizer(PlanOptimizer):
    """
    Bộ tối ưu hóa đa tiêu chí (MCO).

    Lớp này triển khai các thuật toán để tạo các lời giải
    Pareto-optimal cho tối ưu hóa đa tiêu chí.
    """

    def __init__(self, plan=None, dose_calculator=None):
        """Khởi tạo bộ tối ưu hóa MCO."""
        super().__init__(plan, dose_calculator)
        self.weight_vectors = []
        self.pareto_solutions = []

    def set_weight_vectors(self, weight_vectors: List[np.ndarray]) -> None:
        """
        Thiết lập các vector trọng số.

        Args:
            weight_vectors: Danh sách các vector trọng số
        """
        self.weight_vectors = weight_vectors

    def generate_pareto_solutions(
        self, num_solutions: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Tạo các lời giải Pareto-optimal.

        Args:
            num_solutions: Số lượng lời giải cần tạo

        Returns:
            Danh sách các lời giải Pareto
        """
        # Tạo các vector trọng số
        self._generate_weight_vectors(num_solutions)

        # Tạo lời giải cho mỗi vector trọng số
        self.pareto_solutions = []

        for i, weights in enumerate(self.weight_vectors):
            # Cập nhật trọng số cho các mục tiêu
            self._set_objective_weights(weights)

            # Tối ưu hóa với trọng số này
            success = self.optimize(max_iterations=50)

            if success:
                # Lưu lời giải
                solution = {
                    "plan": copy.deepcopy(self.plan),
                    "weights": weights,
                    "objective_values": self._get_objective_values(),
                }
                self.pareto_solutions.append(solution)

            # Gọi callback nếu có
            if self.progress_callback:
                progress = (i + 1) / len(self.weight_vectors)
                self.progress_callback(
                    i + 1,
                    {
                        "progress": progress,
                        "current_solution": i + 1,
                        "total_solutions": len(self.weight_vectors),
                    },
                )

        return self.pareto_solutions

    def _generate_weight_vectors(self, num_vectors: int) -> None:
        """
        Tạo các vector trọng số phân bố đều.

        Args:
            num_vectors: Số lượng vector trọng số cần tạo
        """
        # Số lượng mục tiêu
        n_objectives = len(self.objectives)

        if n_objectives <= 1:
            self.weight_vectors = [np.array([1.0])]
            return

        if n_objectives == 2:
            # Phân bố đều trên đoạn [0,1]
            weights = []
            for i in range(num_vectors):
                w1 = i / (num_vectors - 1) if num_vectors > 1 else 0.5
                w2 = 1.0 - w1
                weights.append(np.array([w1, w2]))

            self.weight_vectors = weights
            return

        # Trường hợp >2 mục tiêu
        try:
            from scipy.stats.qmc import LatinHypercube

            sampler = LatinHypercube(d=n_objectives)
            sample = sampler.random(n=num_vectors)

            # Chuẩn hóa để tổng = 1
            weights = []
            for s in sample:
                w = s / np.sum(s)
                weights.append(w)

            self.weight_vectors = weights

        except ImportError:
            # Fallback nếu không có scipy
            logger.warning("Không thể import scipy, sử dụng sampling ngẫu nhiên")

            weights = []
            for _ in range(num_vectors):
                w = np.random.random(n_objectives)
                w = w / np.sum(w)  # Chuẩn hóa
                weights.append(w)

            self.weight_vectors = weights

    def _set_objective_weights(self, weights: np.ndarray) -> None:
        """
        Thiết lập trọng số cho các mục tiêu.

        Args:
            weights: Vector trọng số
        """
        if len(weights) != len(self.objectives):
            logger.error(
                f"Kích thước vector trọng số ({len(weights)}) không khớp với số mục tiêu ({len(self.objectives)})"
            )
            return

        for i, obj in enumerate(self.objectives):
            if hasattr(obj, "weight"):
                obj.weight = float(weights[i])

    def _get_objective_values(self) -> Dict[str, float]:
        """
        Lấy giá trị của tất cả các mục tiêu.

        Returns:
            Từ điển ánh xạ tên mục tiêu đến giá trị
        """
        _, objective_values = self._evaluate_objectives()
        return objective_values


class OptimizationParameters:
    """
    Lớp chứa các tham số cho quá trình tối ưu hóa.

    Lớp này định nghĩa các tham số cần thiết để điều khiển
    quá trình tối ưu hóa kế hoạch xạ trị.
    """

    def __init__(
        self,
        max_iterations: int = 100,
        convergence_threshold: float = 1e-4,
        step_size: float = 0.01,
        momentum: float = 0.9,
        learning_rate_decay: float = 0.95,
        regularization_weight: float = 0.001,
        use_adaptive_learning: bool = True,
        tolerance: float = 1e-6,
        verbose: bool = True,
    ):
        """
        Khởi tạo OptimizationParameters.

        Parameters
        ----------
        max_iterations : int, optional
            Số lần lặp tối đa, mặc định 100
        convergence_threshold : float, optional
            Ngưỡng hội tụ, mặc định 1e-4
        step_size : float, optional
            Kích thước bước, mặc định 0.01
        momentum : float, optional
            Hệ số momentum, mặc định 0.9
        learning_rate_decay : float, optional
            Tỷ lệ giảm learning rate, mặc định 0.95
        regularization_weight : float, optional
            Trọng số regularization, mặc định 0.001
        use_adaptive_learning : bool, optional
            Sử dụng adaptive learning rate, mặc định True
        tolerance : float, optional
            Độ dung sai, mặc định 1e-6
        verbose : bool, optional
            In thông tin chi tiết, mặc định True
        """
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.step_size = step_size
        self.momentum = momentum
        self.learning_rate_decay = learning_rate_decay
        self.regularization_weight = regularization_weight
        self.use_adaptive_learning = use_adaptive_learning
        self.tolerance = tolerance
        self.verbose = verbose

        # Tham số bổ sung cho các thuật toán đặc biệt
        self.algorithm_specific_params = {}

        logger.info("Khởi tạo OptimizationParameters")

    def set_algorithm_parameter(self, name: str, value: Any):
        """
        Thiết lập tham số đặc biệt cho thuật toán.

        Parameters
        ----------
        name : str
            Tên tham số
        value : Any
            Giá trị tham số
        """
        self.algorithm_specific_params[name] = value

    def get_algorithm_parameter(self, name: str, default=None):
        """
        Lấy tham số đặc biệt của thuật toán.

        Parameters
        ----------
        name : str
            Tên tham số
        default : Any, optional
            Giá trị mặc định nếu không tìm thấy

        Returns
        -------
        Any
            Giá trị tham số
        """
        return self.algorithm_specific_params.get(name, default)

    def to_dict(self) -> Dict[str, Any]:
        """
        Chuyển đổi parameters thành dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary chứa các tham số
        """
        return {
            "max_iterations": self.max_iterations,
            "convergence_threshold": self.convergence_threshold,
            "step_size": self.step_size,
            "momentum": self.momentum,
            "learning_rate_decay": self.learning_rate_decay,
            "regularization_weight": self.regularization_weight,
            "use_adaptive_learning": self.use_adaptive_learning,
            "tolerance": self.tolerance,
            "verbose": self.verbose,
            "algorithm_specific_params": self.algorithm_specific_params,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OptimizationParameters":
        """
        Tạo OptimizationParameters từ dictionary.

        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary chứa các tham số

        Returns
        -------
        OptimizationParameters
            Instance được tạo từ dictionary
        """
        params = cls(
            max_iterations=data.get("max_iterations", 100),
            convergence_threshold=data.get("convergence_threshold", 1e-4),
            step_size=data.get("step_size", 0.01),
            momentum=data.get("momentum", 0.9),
            learning_rate_decay=data.get("learning_rate_decay", 0.95),
            regularization_weight=data.get("regularization_weight", 0.001),
            use_adaptive_learning=data.get("use_adaptive_learning", True),
            tolerance=data.get("tolerance", 1e-6),
            verbose=data.get("verbose", True),
        )

        params.algorithm_specific_params = data.get("algorithm_specific_params", {})
        return params

    def __str__(self) -> str:
        return (
            f"OptimizationParameters(max_iter={self.max_iterations}, "
            f"threshold={self.convergence_threshold}, step={self.step_size})"
        )

    def __repr__(self) -> str:
        return self.__str__()


class Optimizer:
    """
    Lớp tối ưu hóa chung cho kế hoạch xạ trị.

    Đây là wrapper class cung cấp interface thống nhất
    cho các thuật toán tối ưu hóa khác nhau.
    """

    def __init__(
        self,
        algorithm: str = "gradient_descent",
        parameters: Optional[OptimizationParameters] = None,
    ):
        """
        Khởi tạo Optimizer.

        Parameters
        ----------
        algorithm : str, optional
            Tên thuật toán tối ưu hóa, mặc định "gradient_descent"
        parameters : OptimizationParameters, optional
            Tham số tối ưu hóa
        """
        self.algorithm = algorithm
        self.parameters = parameters or OptimizationParameters()

        # Trạng thái tối ưu hóa
        self.is_initialized = False
        self.current_iteration = 0
        self.best_score = float("inf")
        self.convergence_history = []

        # Callbacks
        self.progress_callback = None
        self.iteration_callback = None

        logger.info(f"Khởi tạo Optimizer với thuật toán: {algorithm}")

    def initialize(self, plan, objectives, constraints=None):
        """
        Khởi tạo optimizer với kế hoạch và mục tiêu.

        Parameters
        ----------
        plan : Plan
            Kế hoạch cần tối ưu hóa
        objectives : List[Objective]
            Danh sách mục tiêu
        constraints : List[Constraint], optional
            Danh sách ràng buộc
        """
        self.plan = plan
        self.objectives = objectives
        self.constraints = constraints or []

        # Khởi tạo optimizer nội bộ dựa trên thuật toán
        self._initialize_internal_optimizer()

        self.is_initialized = True
        logger.info("Optimizer đã được khởi tạo")

    def _initialize_internal_optimizer(self):
        """Khởi tạo optimizer nội bộ dựa trên thuật toán được chọn."""
        if self.algorithm == "gradient_descent":
            from quangtps.optimization.optimizers import GradientDescentOptimizer

            self._internal_optimizer = GradientDescentOptimizer()
        elif self.algorithm == "simulated_annealing":
            from quangtps.optimization.optimizers import SimulatedAnnealingOptimizer

            self._internal_optimizer = SimulatedAnnealingOptimizer()
        elif self.algorithm == "genetic_algorithm":
            from quangtps.optimization.optimizers import GeneticAlgorithmOptimizer

            self._internal_optimizer = GeneticAlgorithmOptimizer()
        else:
            # Fallback to PlanOptimizer
            self._internal_optimizer = PlanOptimizer()

    def optimize(self) -> Dict[str, Any]:
        """
        Thực hiện tối ưu hóa.

        Returns
        -------
        Dict[str, Any]
            Kết quả tối ưu hóa
        """
        if not self.is_initialized:
            raise ValueError("Optimizer chưa được khởi tạo. Gọi initialize() trước.")

        logger.info(f"Bắt đầu tối ưu hóa với thuật toán {self.algorithm}")

        # Reset trạng thái
        self.current_iteration = 0
        self.best_score = float("inf")
        self.convergence_history = []

        try:
            # Gọi optimizer nội bộ
            if hasattr(self._internal_optimizer, "optimize"):
                success = self._internal_optimizer.optimize(
                    max_iterations=self.parameters.max_iterations,
                    convergence_threshold=self.parameters.convergence_threshold,
                )
            else:
                success = self._fallback_optimize()

            # Tạo kết quả
            result = {
                "success": success,
                "algorithm": self.algorithm,
                "iterations": self.current_iteration,
                "best_score": self.best_score,
                "convergence_history": self.convergence_history,
                "final_plan": self.plan,
            }

            logger.info(
                f"Tối ưu hóa hoàn thành: {'thành công' if success else 'thất bại'}"
            )
            return result

        except Exception as e:
            logger.error(f"Lỗi trong quá trình tối ưu hóa: {e}")
            return {
                "success": False,
                "error": str(e),
                "algorithm": self.algorithm,
                "iterations": self.current_iteration,
            }

    def _fallback_optimize(self) -> bool:
        """Phương thức tối ưu hóa dự phòng."""
        logger.warning("Sử dụng phương thức tối ưu hóa dự phòng")

        for i in range(self.parameters.max_iterations):
            self.current_iteration = i + 1

            # Tính toán score giả lập
            score = self.best_score * (1 - 0.01 * i / self.parameters.max_iterations)
            self.convergence_history.append(score)

            if score < self.best_score:
                self.best_score = score

            # Kiểm tra hội tụ
            if (
                i > 10
                and abs(self.convergence_history[-1] - self.convergence_history[-2])
                < self.parameters.convergence_threshold
            ):
                break

        return True

    def set_progress_callback(self, callback: Callable[[int, float], None]):
        """
        Thiết lập callback theo dõi tiến độ.

        Parameters
        ----------
        callback : Callable[[int, float], None]
            Function nhận (iteration, score)
        """
        self.progress_callback = callback

    def set_iteration_callback(self, callback: Callable[[int, Dict[str, Any]], None]):
        """
        Thiết lập callback cho mỗi iteration.

        Parameters
        ----------
        callback : Callable[[int, Dict[str, Any]], None]
            Function nhận (iteration, data)
        """
        self.iteration_callback = callback

    def get_convergence_data(self) -> List[float]:
        """
        Lấy dữ liệu hội tụ.

        Returns
        -------
        List[float]
            Lịch sử giá trị hàm mục tiêu
        """
        return self.convergence_history.copy()

    def get_best_solution(self) -> Dict[str, Any]:
        """
        Lấy giải pháp tốt nhất.

        Returns
        -------
        Dict[str, Any]
            Thông tin giải pháp tốt nhất
        """
        return {
            "plan": self.plan,
            "score": self.best_score,
            "iteration": self.current_iteration,
            "algorithm": self.algorithm,
        }

    def save_state(self, filepath: str):
        """
        Lưu trạng thái optimizer.

        Parameters
        ----------
        filepath : str
            Đường dẫn file để lưu
        """
        import json

        state = {
            "algorithm": self.algorithm,
            "parameters": self.parameters.to_dict(),
            "current_iteration": self.current_iteration,
            "best_score": self.best_score,
            "convergence_history": self.convergence_history,
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)

        logger.info(f"Đã lưu trạng thái optimizer vào {filepath}")

    @classmethod
    def load_state(cls, filepath: str) -> "Optimizer":
        """
        Tải trạng thái optimizer từ file.

        Parameters
        ----------
        filepath : str
            Đường dẫn file để tải

        Returns
        -------
        Optimizer
            Instance optimizer được tải
        """
        import json

        with open(filepath, "r", encoding="utf-8") as f:
            state = json.load(f)

        parameters = OptimizationParameters.from_dict(state["parameters"])
        optimizer = cls(algorithm=state["algorithm"], parameters=parameters)

        optimizer.current_iteration = state["current_iteration"]
        optimizer.best_score = state["best_score"]
        optimizer.convergence_history = state["convergence_history"]

        logger.info(f"Đã tải trạng thái optimizer từ {filepath}")
        return optimizer

    def __str__(self) -> str:
        return (
            f"Optimizer(algorithm={self.algorithm}, initialized={self.is_initialized})"
        )

    def __repr__(self) -> str:
        return self.__str__()
