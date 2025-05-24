"""
QuangTPS Advanced Optimization Engine

Module tối ưu hóa nâng cao cho hệ thống QuangTPS.
Cung cấp nhiều thuật toán tối ưu hóa cho lập kế hoạch xạ trị
từ cơ bản đến chuyên nghiệp, hỗ trợ IMRT, VMAT, và SRS.
"""

import logging
import os
import json
import numpy as np
import time
import math
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field, asdict
from pathlib import Path
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

logger = logging.getLogger(__name__)

# Import scientific libraries với fallback
try:
    import numpy as np
    from scipy import optimize, sparse
    from scipy.optimize import minimize, differential_evolution, dual_annealing
    from scipy.sparse import csr_matrix, lil_matrix

    HAS_SCIPY = True
    logger.info("NumPy và SciPy được tải thành công")
except ImportError as e:
    logger.warning(f"Scientific libraries không khả dụng: {e}")
    HAS_SCIPY = False

# Import machine learning libraries nếu có
try:
    import sklearn
    from sklearn.linear_model import Ridge, Lasso
    from sklearn.ensemble import RandomForestRegressor

    HAS_ML = True
    logger.info("Scikit-learn được tải thành công")
except ImportError:
    HAS_ML = False
    logger.info("Machine learning libraries không khả dụng")

# Import advanced optimization libraries nếu có
try:
    import cvxpy as cp

    HAS_CVXPY = True
    logger.info("CVXPY được tải thành công")
except ImportError:
    HAS_CVXPY = False

# Import core modules với fallback
try:
    from quangtps.dose.dose_grid import DoseGrid
    from quangtps.beams.beam_data import BeamData
    from quangtps.optimization.objectives import OptimizationObjective
    from quangtps.structures.structure_manager import Structure

    HAS_CORE_MODULES = True
    logger.info("Core modules được tải thành công")
except ImportError as e:
    logger.warning(f"Core modules không khả dụng: {e}")
    HAS_CORE_MODULES = False

    # Fallback classes
    class DoseGrid:
        def __init__(self, *args, **kwargs):
            self.shape = (64, 64, 32)
            self.spacing = (2.0, 2.0, 3.0)
            self.origin = (0.0, 0.0, 0.0)
            self.dose_data = np.zeros(self.shape) if "np" in globals() else None

    class BeamData:
        def __init__(self, *args, **kwargs):
            self.gantry_angle = 0.0
            self.energy = "6MV"
            self.weight = 1.0
            self.mu = 100.0

    class Structure:
        def __init__(self, *args, **kwargs):
            self.name = "Unknown"
            self.mask = None

    class OptimizationObjective:
        def __init__(
            self, structure_name: str, objective_type: str = "MINIMIZE", **kwargs
        ):
            self.structure_name = structure_name
            self.objective_type = objective_type
            self.dose_level = kwargs.get("dose_level", None)
            self.volume_threshold = kwargs.get("volume_threshold", None)
            self.weight = kwargs.get("weight", 1.0)
            self.priority = kwargs.get("priority", 1)
            self.constraint_type = kwargs.get("constraint_type", "SOFT")
            self.violation_penalty = kwargs.get("violation_penalty", 1000.0)
            self.robust_margin = kwargs.get("robust_margin", 0.0)
            self.adaptive_weight = kwargs.get("adaptive_weight", False)


class OptimizationAlgorithm(Enum):
    """Enum cho các thuật toán tối ưu hóa."""

    # Gradient-based methods
    GRADIENT_DESCENT = "gradient_descent"
    CONJUGATE_GRADIENT = "conjugate_gradient"
    LBFGS = "lbfgs"
    NEWTON_CG = "newton_cg"

    # Global optimization
    DIFFERENTIAL_EVOLUTION = "differential_evolution"
    SIMULATED_ANNEALING = "simulated_annealing"
    GENETIC_ALGORITHM = "genetic_algorithm"
    PARTICLE_SWARM = "particle_swarm"

    # Specialized for radiotherapy
    QUADRATIC_PROGRAMMING = "quadratic_programming"
    LEXICOGRAPHIC = "lexicographic"
    MULTI_CRITERIA = "multi_criteria"
    ROBUST_OPTIMIZATION = "robust_optimization"

    # Machine learning enhanced
    ML_GUIDED = "ml_guided"
    REINFORCEMENT_LEARNING = "reinforcement_learning"

    # Simple methods
    DIRECT_APERTURE = "direct_aperture"
    SIMPLE = "simple"


class OptimizationMode(Enum):
    """Enum cho các chế độ tối ưu hóa."""

    FORWARD_PLANNING = "forward_planning"
    INVERSE_PLANNING = "inverse_planning"
    DIRECT_APERTURE_OPTIMIZATION = "direct_aperture_optimization"
    VOLUMETRIC_MODULATED = "volumetric_modulated"
    MULTI_CRITERIA_OPTIMIZATION = "multi_criteria_optimization"
    ROBUST_OPTIMIZATION = "robust_optimization"
    ADAPTIVE_OPTIMIZATION = "adaptive_optimization"


@dataclass
class OptimizationSettings:
    """Cài đặt cho optimization engine."""

    # Algorithm selection
    algorithm: OptimizationAlgorithm = OptimizationAlgorithm.LBFGS
    mode: OptimizationMode = OptimizationMode.INVERSE_PLANNING

    # Convergence settings
    max_iterations: int = 1000
    tolerance: float = 1e-6
    relative_tolerance: float = 1e-4
    gradient_tolerance: float = 1e-8

    # Performance settings
    use_parallel_processing: bool = True
    max_workers: int = mp.cpu_count()
    memory_limit_mb: float = 8192.0

    # Regularization
    smoothness_weight: float = 0.1
    sparsity_weight: float = 0.01
    regularization_type: str = "L2"  # L1, L2, ELASTIC_NET

    # Advanced options
    use_adaptive_step_size: bool = True
    restart_frequency: int = 50
    line_search_method: str = "STRONG_WOLFE"

    # Quality assurance
    validate_inputs: bool = True
    check_convergence: bool = True
    save_intermediate_results: bool = False

    def __post_init__(self):
        """Validate settings."""
        if self.max_iterations < 1:
            raise ValueError("Max iterations phải ít nhất 1")
        if self.tolerance <= 0:
            raise ValueError("Tolerance phải lớn hơn 0")
        if self.memory_limit_mb < 512:
            raise ValueError("Memory limit phải ít nhất 512MB")


@dataclass
class OptimizationResult:
    """Kết quả tối ưu hóa."""

    # Solution
    optimized_weights: np.ndarray
    optimized_parameters: Dict[str, Any]
    final_dose_distribution: Optional[np.ndarray] = None

    # Convergence info
    converged: bool = False
    iterations_used: int = 0
    final_objective_value: float = float("inf")

    # Performance metrics
    optimization_time: float = 0.0
    memory_used: float = 0.0  # MB
    algorithm_used: OptimizationAlgorithm = OptimizationAlgorithm.SIMPLE

    # Quality metrics
    objective_function_history: List[float] = field(default_factory=list)
    constraint_violations: Dict[str, float] = field(default_factory=dict)
    gradient_norm_history: List[float] = field(default_factory=list)

    # Metadata
    optimization_timestamp: datetime = field(default_factory=datetime.now)
    settings_used: Optional[OptimizationSettings] = None

    def get_summary(self) -> Dict[str, Any]:
        """Lấy tóm tắt kết quả."""
        return {
            "algorithm": self.algorithm_used.value,
            "converged": self.converged,
            "iterations": self.iterations_used,
            "final_objective": self.final_objective_value,
            "optimization_time": self.optimization_time,
            "memory_used": self.memory_used,
            "constraint_violations": len(self.constraint_violations),
            "improvement_ratio": self._calculate_improvement_ratio(),
        }

    def _calculate_improvement_ratio(self) -> float:
        """Tính tỷ lệ cải thiện."""
        if len(self.objective_function_history) < 2:
            return 0.0

        initial_value = self.objective_function_history[0]
        final_value = self.objective_function_history[-1]

        if initial_value == 0:
            return 0.0

        return (initial_value - final_value) / abs(initial_value)


class BaseOptimizer:
    """
    Base class cho tất cả optimizers.
    """

    def __init__(self, settings: Optional[OptimizationSettings] = None):
        self.settings = settings or OptimizationSettings()
        self.name = "Base Optimizer"

        # Performance monitoring
        self._optimization_count = 0
        self._total_optimization_time = 0.0

        # State management
        self._current_iteration = 0
        self._current_objective_value = float("inf")
        self._convergence_history = []

        logger.info(f"{self.name} khởi tạo")

    def optimize(
        self,
        objectives: List[OptimizationObjective],
        beam_data: List[BeamData],
        structures: List[Structure],
        initial_parameters: Optional[np.ndarray] = None,
        progress_callback: Optional[Callable] = None,
    ) -> OptimizationResult:
        """
        Thực hiện tối ưu hóa.

        Args:
            objectives: Danh sách mục tiêu tối ưu hóa
            beam_data: Dữ liệu các chùm tia
            structures: Cấu trúc giải phẫu
            initial_parameters: Tham số khởi tạo
            progress_callback: Callback báo cáo tiến trình

        Returns:
            OptimizationResult với kết quả tối ưu hóa
        """
        raise NotImplementedError("Subclasses must implement optimize")

    def validate_inputs(
        self,
        objectives: List[OptimizationObjective],
        beam_data: List[BeamData],
        structures: List[Structure],
    ) -> bool:
        """Validate input parameters."""
        try:
            if not objectives:
                logger.error("Objectives list trống")
                return False

            if not beam_data:
                logger.error("Beam data list trống")
                return False

            if not structures:
                logger.error("Structures list trống")
                return False

            return True

        except Exception as e:
            logger.error(f"Lỗi validate inputs: {e}")
            return False

    def calculate_objective_function(
        self,
        parameters: np.ndarray,
        objectives: List[OptimizationObjective],
        beam_data: List[BeamData],
        structures: List[Structure],
    ) -> float:
        """Tính toán hàm mục tiêu."""
        try:
            total_objective = 0.0

            # Calculate dose distribution với parameters hiện tại
            dose_distribution = self._calculate_dose_distribution(parameters, beam_data)

            # Evaluate từng objective
            for obj in objectives:
                structure_mask = self._get_structure_mask(
                    obj.structure_name, structures
                )
                if structure_mask is None:
                    continue

                obj_value = self._evaluate_single_objective(
                    obj, dose_distribution, structure_mask
                )

                total_objective += obj.weight * obj_value

            # Add regularization terms
            total_objective += self._calculate_regularization(parameters)

            return total_objective

        except Exception as e:
            logger.error(f"Lỗi calculate objective function: {e}")
            return float("inf")

    def _calculate_dose_distribution(
        self, parameters: np.ndarray, beam_data: List[BeamData]
    ) -> np.ndarray:
        """Tính toán phân phối liều từ parameters."""
        try:
            # Simplified dose calculation for optimization
            # Real implementation would use advanced dose calculation engine

            dose_grid_shape = (64, 64, 32)  # Simplified
            dose_distribution = np.zeros(dose_grid_shape)

            for i, beam in enumerate(beam_data):
                if i < len(parameters):
                    weight = parameters[i]

                    # Simple beam dose contribution
                    beam_dose = self._calculate_beam_contribution(beam, weight)
                    dose_distribution += beam_dose

            return dose_distribution

        except Exception as e:
            logger.error(f"Lỗi calculate dose distribution: {e}")
            return np.zeros((64, 64, 32))

    def _calculate_beam_contribution(self, beam: BeamData, weight: float) -> np.ndarray:
        """Tính toán contribution của một beam."""
        try:
            dose_grid_shape = (64, 64, 32)
            beam_dose = np.zeros(dose_grid_shape)

            # Simplified Gaussian beam model
            center_x, center_y = dose_grid_shape[0] // 2, dose_grid_shape[1] // 2
            sigma = 10.0  # mm

            for z in range(dose_grid_shape[2]):
                depth = z * 3.0  # 3mm spacing
                pdd = 100 * np.exp(-depth / 100)  # Simplified PDD

                for x in range(dose_grid_shape[0]):
                    for y in range(dose_grid_shape[1]):
                        dx = (x - center_x) * 2.0  # 2mm spacing
                        dy = (y - center_y) * 2.0

                        distance = np.sqrt(dx**2 + dy**2)
                        profile = np.exp(-(distance**2) / (2 * sigma**2))

                        beam_dose[x, y, z] = pdd * profile * weight * beam.mu / 100.0

            return beam_dose

        except Exception as e:
            logger.error(f"Lỗi calculate beam contribution: {e}")
            return np.zeros((64, 64, 32))

    def _evaluate_single_objective(
        self,
        objective: OptimizationObjective,
        dose_distribution: np.ndarray,
        structure_mask: np.ndarray,
    ) -> float:
        """Đánh giá một mục tiêu cụ thể."""
        try:
            if structure_mask.sum() == 0:
                return 0.0

            # Extract dose trong structure
            structure_dose = dose_distribution[structure_mask > 0]

            if len(structure_dose) == 0:
                return 0.0

            # Calculate objective value dựa trên type
            if objective.objective_type == "MINIMIZE":
                if objective.dose_level is not None:
                    # Minimize dose above threshold
                    excess_dose = structure_dose - objective.dose_level
                    excess_dose = excess_dose[excess_dose > 0]
                    return np.sum(excess_dose**2) if len(excess_dose) > 0 else 0.0
                else:
                    # Minimize mean dose
                    return np.mean(structure_dose)

            elif objective.objective_type == "MAXIMIZE":
                if objective.dose_level is not None:
                    # Maximize coverage above threshold
                    coverage = np.sum(structure_dose >= objective.dose_level) / len(
                        structure_dose
                    )
                    return -(coverage)  # Negative for minimization
                else:
                    # Maximize mean dose
                    return -np.mean(structure_dose)

            elif objective.objective_type == "CONSTRAINT":
                if (
                    objective.dose_level is not None
                    and objective.volume_threshold is not None
                ):
                    # Volume constraint (e.g., V20 < 35%)
                    volume_above_dose = (
                        np.sum(structure_dose >= objective.dose_level)
                        / len(structure_dose)
                        * 100
                    )
                    violation = max(0, volume_above_dose - objective.volume_threshold)
                    return objective.violation_penalty * violation**2

            return 0.0

        except Exception as e:
            logger.error(f"Lỗi evaluate single objective: {e}")
            return 1e6  # Large penalty for errors

    def _calculate_regularization(self, parameters: np.ndarray) -> float:
        """Tính toán regularization terms."""
        try:
            regularization = 0.0

            # Smoothness regularization
            if self.settings.smoothness_weight > 0 and len(parameters) > 1:
                diff = np.diff(parameters)
                smoothness_penalty = self.settings.smoothness_weight * np.sum(diff**2)
                regularization += smoothness_penalty

            # Sparsity regularization
            if self.settings.sparsity_weight > 0:
                if self.settings.regularization_type == "L1":
                    sparsity_penalty = self.settings.sparsity_weight * np.sum(
                        np.abs(parameters)
                    )
                elif self.settings.regularization_type == "L2":
                    sparsity_penalty = self.settings.sparsity_weight * np.sum(
                        parameters**2
                    )
                else:  # ELASTIC_NET
                    l1_term = 0.5 * np.sum(np.abs(parameters))
                    l2_term = 0.5 * np.sum(parameters**2)
                    sparsity_penalty = self.settings.sparsity_weight * (
                        l1_term + l2_term
                    )

                regularization += sparsity_penalty

            return regularization

        except Exception as e:
            logger.error(f"Lỗi calculate regularization: {e}")
            return 0.0

    def _get_structure_mask(
        self, structure_name: str, structures: List[Structure]
    ) -> Optional[np.ndarray]:
        """Lấy mask của structure."""
        try:
            for structure in structures:
                if structure.name == structure_name:
                    if hasattr(structure, "mask") and structure.mask is not None:
                        return structure.mask
                    else:
                        # Create simple mask for testing
                        mask = np.zeros((64, 64, 32), dtype=bool)
                        mask[20:44, 20:44, 10:22] = True  # Simple box
                        return mask

            logger.warning(f"Structure '{structure_name}' không tìm thấy")
            return None

        except Exception as e:
            logger.error(f"Lỗi get structure mask: {e}")
            return None

    def get_performance_stats(self) -> Dict[str, Any]:
        """Lấy thống kê performance."""
        avg_time = self._total_optimization_time / max(self._optimization_count, 1)

        return {
            "optimizer_name": self.name,
            "optimization_count": self._optimization_count,
            "total_time": self._total_optimization_time,
            "average_time": avg_time,
            "optimizations_per_hour": 3600.0 / max(avg_time, 0.001),
        }


class GradientBasedOptimizer(BaseOptimizer):
    """
    Gradient-based optimization algorithms.
    Bao gồm L-BFGS, Conjugate Gradient, Newton methods.
    """

    def __init__(self, settings: Optional[OptimizationSettings] = None):
        super().__init__(settings)
        self.name = "Gradient-Based Optimizer"

        # Algorithm-specific parameters
        self.gradient_step_size = 0.01
        self.gradient_memory_length = 10  # For L-BFGS
        self.use_hessian_approximation = True

        logger.info("Gradient-Based Optimizer khởi tạo")

    def optimize(
        self,
        objectives: List[OptimizationObjective],
        beam_data: List[BeamData],
        structures: List[Structure],
        initial_parameters: Optional[np.ndarray] = None,
        progress_callback: Optional[Callable] = None,
    ) -> OptimizationResult:
        """Optimization sử dụng gradient-based methods."""
        start_time = time.time()

        try:
            if self.settings.validate_inputs:
                if not self.validate_inputs(objectives, beam_data, structures):
                    raise ValueError("Input validation failed")

            # Initialize parameters
            if initial_parameters is None:
                num_beams = len(beam_data)
                initial_parameters = np.ones(num_beams) * 100.0  # Default beam weights

            # Setup objective function cho scipy.optimize
            def objective_func(params):
                return self.calculate_objective_function(
                    params, objectives, beam_data, structures
                )

            # Setup gradient function nếu possible
            def gradient_func(params):
                return self._calculate_gradient(
                    params, objectives, beam_data, structures
                )

            if progress_callback:
                progress_callback(10, "Starting gradient-based optimization...")

            # Optimization với scipy
            optimization_result = None

            if HAS_SCIPY:
                if self.settings.algorithm == OptimizationAlgorithm.LBFGS:
                    optimization_result = optimize.minimize(
                        objective_func,
                        initial_parameters,
                        method="L-BFGS-B",
                        jac=gradient_func,
                        bounds=[(0, 1000)]
                        * len(initial_parameters),  # Beam weight bounds
                        options={
                            "maxiter": self.settings.max_iterations,
                            "ftol": self.settings.tolerance,
                            "gtol": self.settings.gradient_tolerance,
                        },
                    )

                elif (
                    self.settings.algorithm == OptimizationAlgorithm.CONJUGATE_GRADIENT
                ):
                    optimization_result = optimize.minimize(
                        objective_func,
                        initial_parameters,
                        method="CG",
                        jac=gradient_func,
                        options={
                            "maxiter": self.settings.max_iterations,
                            "gtol": self.settings.gradient_tolerance,
                        },
                    )

                else:  # Default to BFGS
                    optimization_result = optimize.minimize(
                        objective_func,
                        initial_parameters,
                        method="BFGS",
                        jac=gradient_func,
                        options={
                            "maxiter": self.settings.max_iterations,
                            "gtol": self.settings.gradient_tolerance,
                        },
                    )

                if progress_callback:
                    progress_callback(80, "Gradient optimization completed")

                # Create result
                final_dose = self._calculate_dose_distribution(
                    optimization_result.x, beam_data
                )

                result = OptimizationResult(
                    optimized_weights=optimization_result.x,
                    optimized_parameters={
                        "beam_weights": optimization_result.x.tolist()
                    },
                    final_dose_distribution=final_dose,
                    converged=optimization_result.success,
                    iterations_used=optimization_result.nit,
                    final_objective_value=optimization_result.fun,
                    optimization_time=time.time() - start_time,
                    algorithm_used=self.settings.algorithm,
                    settings_used=self.settings,
                )

            else:
                # Fallback simple gradient descent
                result = self._simple_gradient_descent(
                    objective_func, initial_parameters, progress_callback
                )
                result.optimization_time = time.time() - start_time

            # Update statistics
            self._optimization_count += 1
            self._total_optimization_time += result.optimization_time

            if progress_callback:
                progress_callback(100, "Optimization completed")

            return result

        except Exception as e:
            logger.error(f"Lỗi gradient-based optimization: {e}")
            # Return empty result
            return OptimizationResult(
                optimized_weights=initial_parameters
                if initial_parameters is not None
                else np.array([]),
                optimized_parameters={},
                converged=False,
                optimization_time=time.time() - start_time,
                algorithm_used=self.settings.algorithm,
            )

    def _calculate_gradient(
        self,
        parameters: np.ndarray,
        objectives: List[OptimizationObjective],
        beam_data: List[BeamData],
        structures: List[Structure],
    ) -> np.ndarray:
        """Tính toán gradient của objective function."""
        try:
            gradient = np.zeros_like(parameters)
            eps = 1e-6  # Finite difference step

            # Finite difference approximation
            for i in range(len(parameters)):
                params_plus = parameters.copy()
                params_minus = parameters.copy()

                params_plus[i] += eps
                params_minus[i] -= eps

                obj_plus = self.calculate_objective_function(
                    params_plus, objectives, beam_data, structures
                )
                obj_minus = self.calculate_objective_function(
                    params_minus, objectives, beam_data, structures
                )

                gradient[i] = (obj_plus - obj_minus) / (2 * eps)

            return gradient

        except Exception as e:
            logger.error(f"Lỗi calculate gradient: {e}")
            return np.zeros_like(parameters)

    def _simple_gradient_descent(
        self,
        objective_func: Callable,
        initial_parameters: np.ndarray,
        progress_callback: Optional[Callable] = None,
    ) -> OptimizationResult:
        """Simple gradient descent fallback."""
        try:
            current_params = initial_parameters.copy()
            best_params = current_params.copy()
            best_objective = objective_func(current_params)

            for iteration in range(self.settings.max_iterations):
                # Calculate gradient
                gradient = np.zeros_like(current_params)
                eps = 1e-6

                for i in range(len(current_params)):
                    params_plus = current_params.copy()
                    params_plus[i] += eps
                    gradient[i] = (objective_func(params_plus) - best_objective) / eps

                # Update parameters
                current_params -= self.gradient_step_size * gradient
                current_params = np.maximum(
                    current_params, 0
                )  # Ensure non-negative weights

                # Evaluate new objective
                current_objective = objective_func(current_params)

                if current_objective < best_objective:
                    best_objective = current_objective
                    best_params = current_params.copy()

                # Check convergence
                if np.linalg.norm(gradient) < self.settings.gradient_tolerance:
                    break

                if progress_callback and iteration % 10 == 0:
                    progress = min(80, (iteration / self.settings.max_iterations) * 80)
                    progress_callback(
                        progress, f"Gradient descent: iteration {iteration}"
                    )

            return OptimizationResult(
                optimized_weights=best_params,
                optimized_parameters={"beam_weights": best_params.tolist()},
                converged=True,
                iterations_used=iteration + 1,
                final_objective_value=best_objective,
                algorithm_used=self.settings.algorithm,
            )

        except Exception as e:
            logger.error(f"Lỗi simple gradient descent: {e}")
            return OptimizationResult(
                optimized_weights=initial_parameters,
                optimized_parameters={},
                converged=False,
                algorithm_used=self.settings.algorithm,
            )


class GlobalOptimizer(BaseOptimizer):
    """
    Global optimization algorithms.
    Bao gồm Differential Evolution, Simulated Annealing, Genetic Algorithm.
    """

    def __init__(self, settings: Optional[OptimizationSettings] = None):
        super().__init__(settings)
        self.name = "Global Optimizer"

        # Algorithm-specific parameters
        self.population_size = 50
        self.mutation_rate = 0.1
        self.crossover_rate = 0.7
        self.temperature_initial = 1000.0
        self.cooling_rate = 0.95

        logger.info("Global Optimizer khởi tạo")

    def optimize(
        self,
        objectives: List[OptimizationObjective],
        beam_data: List[BeamData],
        structures: List[Structure],
        initial_parameters: Optional[np.ndarray] = None,
        progress_callback: Optional[Callable] = None,
    ) -> OptimizationResult:
        """Global optimization sử dụng các thuật toán global search."""
        start_time = time.time()

        try:
            if self.settings.validate_inputs:
                if not self.validate_inputs(objectives, beam_data, structures):
                    raise ValueError("Input validation failed")

            # Setup bounds cho parameters
            num_beams = len(beam_data)
            bounds = [(0, 1000) for _ in range(num_beams)]  # Beam weight bounds

            # Setup objective function
            def objective_func(params):
                return self.calculate_objective_function(
                    params, objectives, beam_data, structures
                )

            if progress_callback:
                progress_callback(10, "Starting global optimization...")

            optimization_result = None

            if HAS_SCIPY:
                if (
                    self.settings.algorithm
                    == OptimizationAlgorithm.DIFFERENTIAL_EVOLUTION
                ):
                    optimization_result = differential_evolution(
                        objective_func,
                        bounds,
                        maxiter=self.settings.max_iterations,
                        popsize=15,  # Population size multiplier
                        tol=self.settings.tolerance,
                        seed=42,
                    )

                elif (
                    self.settings.algorithm == OptimizationAlgorithm.SIMULATED_ANNEALING
                ):
                    optimization_result = dual_annealing(
                        objective_func,
                        bounds,
                        maxiter=self.settings.max_iterations,
                        initial_temp=self.temperature_initial,
                        seed=42,
                    )

                else:  # Default to differential evolution
                    optimization_result = differential_evolution(
                        objective_func,
                        bounds,
                        maxiter=self.settings.max_iterations,
                        tol=self.settings.tolerance,
                    )

                if progress_callback:
                    progress_callback(80, "Global optimization completed")

                # Create result
                final_dose = self._calculate_dose_distribution(
                    optimization_result.x, beam_data
                )

                result = OptimizationResult(
                    optimized_weights=optimization_result.x,
                    optimized_parameters={
                        "beam_weights": optimization_result.x.tolist()
                    },
                    final_dose_distribution=final_dose,
                    converged=optimization_result.success,
                    iterations_used=optimization_result.nit,
                    final_objective_value=optimization_result.fun,
                    optimization_time=time.time() - start_time,
                    algorithm_used=self.settings.algorithm,
                    settings_used=self.settings,
                )

            else:
                # Fallback simple genetic algorithm
                result = self._simple_genetic_algorithm(
                    objective_func, bounds, progress_callback
                )
                result.optimization_time = time.time() - start_time

            # Update statistics
            self._optimization_count += 1
            self._total_optimization_time += result.optimization_time

            if progress_callback:
                progress_callback(100, "Global optimization completed")

            return result

        except Exception as e:
            logger.error(f"Lỗi global optimization: {e}")
            # Return empty result
            default_weights = (
                np.ones(len(beam_data)) * 100.0 if beam_data else np.array([])
            )
            return OptimizationResult(
                optimized_weights=default_weights,
                optimized_parameters={},
                converged=False,
                optimization_time=time.time() - start_time,
                algorithm_used=self.settings.algorithm,
            )

    def _simple_genetic_algorithm(
        self,
        objective_func: Callable,
        bounds: List[Tuple[float, float]],
        progress_callback: Optional[Callable] = None,
    ) -> OptimizationResult:
        """Simple genetic algorithm fallback."""
        try:
            # Initialize population
            population = []
            for _ in range(self.population_size):
                individual = []
                for bound in bounds:
                    individual.append(np.random.uniform(bound[0], bound[1]))
                population.append(np.array(individual))

            best_individual = None
            best_fitness = float("inf")

            for generation in range(
                self.settings.max_iterations // self.population_size
            ):
                # Evaluate fitness
                fitness_scores = []
                for individual in population:
                    fitness = objective_func(individual)
                    fitness_scores.append(fitness)

                    if fitness < best_fitness:
                        best_fitness = fitness
                        best_individual = individual.copy()

                # Selection (tournament selection)
                new_population = []
                for _ in range(self.population_size):
                    parent1 = self._tournament_selection(population, fitness_scores)
                    parent2 = self._tournament_selection(population, fitness_scores)

                    # Crossover
                    if np.random.random() < self.crossover_rate:
                        child = self._crossover(parent1, parent2)
                    else:
                        child = parent1.copy()

                    # Mutation
                    child = self._mutate(child, bounds)
                    new_population.append(child)

                population = new_population

                if progress_callback and generation % 5 == 0:
                    progress = min(
                        80,
                        (
                            generation
                            / (self.settings.max_iterations // self.population_size)
                        )
                        * 80,
                    )
                    progress_callback(
                        progress, f"Genetic algorithm: generation {generation}"
                    )

            return OptimizationResult(
                optimized_weights=best_individual,
                optimized_parameters={"beam_weights": best_individual.tolist()},
                converged=True,
                iterations_used=generation + 1,
                final_objective_value=best_fitness,
                algorithm_used=self.settings.algorithm,
            )

        except Exception as e:
            logger.error(f"Lỗi simple genetic algorithm: {e}")
            default_weights = np.ones(len(bounds)) * 100.0
            return OptimizationResult(
                optimized_weights=default_weights,
                optimized_parameters={},
                converged=False,
                algorithm_used=self.settings.algorithm,
            )

    def _tournament_selection(
        self, population: List[np.ndarray], fitness_scores: List[float]
    ) -> np.ndarray:
        """Tournament selection."""
        tournament_size = 3
        tournament_indices = np.random.choice(
            len(population), tournament_size, replace=False
        )
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_index = tournament_indices[np.argmin(tournament_fitness)]
        return population[winner_index]

    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """Single-point crossover."""
        if len(parent1) < 2:
            return parent1.copy()

        crossover_point = np.random.randint(1, len(parent1))
        child = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
        return child

    def _mutate(
        self, individual: np.ndarray, bounds: List[Tuple[float, float]]
    ) -> np.ndarray:
        """Gaussian mutation."""
        mutated = individual.copy()

        for i in range(len(mutated)):
            if np.random.random() < self.mutation_rate:
                mutation = np.random.normal(0, 0.1 * (bounds[i][1] - bounds[i][0]))
                mutated[i] += mutation
                mutated[i] = np.clip(mutated[i], bounds[i][0], bounds[i][1])

        return mutated


class AdvancedOptimizationEngine:
    """
    Advanced Optimization Engine với multiple algorithms và automatic selection.
    """

    def __init__(self, settings: Optional[OptimizationSettings] = None):
        self.settings = settings or OptimizationSettings()

        # Initialize optimizers
        self.optimizers: Dict[OptimizationAlgorithm, BaseOptimizer] = {}
        self._initialize_optimizers()

        # Performance monitoring
        self._optimization_history: List[OptimizationResult] = []

        logger.info("Advanced Optimization Engine khởi tạo")

    def _initialize_optimizers(self):
        """Initialize all available optimizers."""
        try:
            # Gradient-based optimizers
            gradient_settings = OptimizationSettings()
            gradient_settings.algorithm = OptimizationAlgorithm.LBFGS
            self.optimizers[OptimizationAlgorithm.LBFGS] = GradientBasedOptimizer(
                gradient_settings
            )

            gradient_settings.algorithm = OptimizationAlgorithm.CONJUGATE_GRADIENT
            self.optimizers[OptimizationAlgorithm.CONJUGATE_GRADIENT] = (
                GradientBasedOptimizer(gradient_settings)
            )

            # Global optimizers
            global_settings = OptimizationSettings()
            global_settings.algorithm = OptimizationAlgorithm.DIFFERENTIAL_EVOLUTION
            self.optimizers[OptimizationAlgorithm.DIFFERENTIAL_EVOLUTION] = (
                GlobalOptimizer(global_settings)
            )

            global_settings.algorithm = OptimizationAlgorithm.SIMULATED_ANNEALING
            self.optimizers[OptimizationAlgorithm.SIMULATED_ANNEALING] = (
                GlobalOptimizer(global_settings)
            )

            logger.info(f"Initialized {len(self.optimizers)} optimizers")

        except Exception as e:
            logger.error(f"Lỗi initialize optimizers: {e}")

    def get_available_algorithms(self) -> List[OptimizationAlgorithm]:
        """Lấy danh sách thuật toán khả dụng."""
        return list(self.optimizers.keys())

    def optimize(
        self,
        algorithm: Optional[OptimizationAlgorithm] = None,
        objectives: Optional[List[OptimizationObjective]] = None,
        beam_data: Optional[List[BeamData]] = None,
        structures: Optional[List[Structure]] = None,
        initial_parameters: Optional[np.ndarray] = None,
        progress_callback: Optional[Callable] = None,
    ) -> Optional[OptimizationResult]:
        """
        Optimize using specified or automatically selected algorithm.
        """
        try:
            # Use default algorithm if not specified
            if algorithm is None:
                algorithm = self.settings.algorithm

            # Check if algorithm is available
            if algorithm not in self.optimizers:
                logger.error(f"Algorithm {algorithm} không khả dụng")
                return None

            # Create default data if not provided
            if objectives is None:
                objectives = [
                    OptimizationObjective(
                        structure_name="PTV",
                        objective_type="MAXIMIZE",
                        dose_level=50.0,
                        weight=1.0,
                    )
                ]

            if beam_data is None:
                beam_data = [BeamData() for _ in range(7)]  # 7-field plan

            if structures is None:
                structures = [Structure()]

            # Get optimizer
            optimizer = self.optimizers[algorithm]

            if progress_callback:
                progress_callback(5, f"Starting {algorithm.value} optimization...")

            # Perform optimization
            result = optimizer.optimize(
                objectives, beam_data, structures, initial_parameters, progress_callback
            )

            # Store result in history
            self._optimization_history.append(result)

            # Limit history size
            if len(self._optimization_history) > 100:
                self._optimization_history = self._optimization_history[-100:]

            logger.info(f"Optimization completed with {algorithm.value}")
            return result

        except Exception as e:
            logger.error(f"Lỗi optimize: {e}")
            return None

    def select_optimal_algorithm(
        self,
        problem_type: str = "IMRT",
        time_constraint: Optional[float] = None,
        accuracy_requirement: str = "MEDIUM",
    ) -> OptimizationAlgorithm:
        """
        Automatically select optimal algorithm based on constraints.
        """
        try:
            if time_constraint and time_constraint < 300:  # 5 minutes
                # Fast optimization needed
                return OptimizationAlgorithm.GRADIENT_DESCENT

            elif accuracy_requirement == "HIGH" or problem_type == "SRS":
                # High accuracy priority
                if OptimizationAlgorithm.LBFGS in self.optimizers:
                    return OptimizationAlgorithm.LBFGS
                else:
                    return OptimizationAlgorithm.CONJUGATE_GRADIENT

            elif problem_type == "VMAT" or accuracy_requirement == "LOW":
                # Global search for complex problems
                return OptimizationAlgorithm.DIFFERENTIAL_EVOLUTION

            else:  # MEDIUM accuracy, IMRT
                # Balanced choice
                return OptimizationAlgorithm.LBFGS

        except Exception as e:
            logger.error(f"Lỗi select optimal algorithm: {e}")
            return OptimizationAlgorithm.LBFGS  # Safe fallback

    def compare_algorithms(
        self,
        algorithms: List[OptimizationAlgorithm],
        objectives: List[OptimizationObjective],
        beam_data: List[BeamData],
        structures: List[Structure],
    ) -> Dict[str, Any]:
        """Compare multiple algorithms on same problem."""
        try:
            comparison_results = {}

            for algorithm in algorithms:
                if algorithm in self.optimizers:
                    result = self.optimize(algorithm, objectives, beam_data, structures)

                    if result:
                        comparison_results[algorithm.value] = {
                            "optimization_time": result.optimization_time,
                            "final_objective": result.final_objective_value,
                            "converged": result.converged,
                            "iterations": result.iterations_used,
                            "improvement_ratio": result._calculate_improvement_ratio(),
                        }

            return {
                "algorithm_comparison": comparison_results,
                "fastest_algorithm": min(
                    comparison_results.items(), key=lambda x: x[1]["optimization_time"]
                )[0]
                if comparison_results
                else None,
                "best_objective": min(
                    comparison_results.items(), key=lambda x: x[1]["final_objective"]
                )[0]
                if comparison_results
                else None,
            }

        except Exception as e:
            logger.error(f"Lỗi compare algorithms: {e}")
            return {}

    def get_engine_statistics(self) -> Dict[str, Any]:
        """Lấy thống kê performance của engine."""
        try:
            if not self._optimization_history:
                return {"total_optimizations": 0}

            # Calculate statistics
            total_optimizations = len(self._optimization_history)
            total_time = sum(r.optimization_time for r in self._optimization_history)
            avg_time = total_time / total_optimizations

            # Algorithm usage
            algorithm_counts = {}
            for result in self._optimization_history:
                alg_name = result.algorithm_used.value
                algorithm_counts[alg_name] = algorithm_counts.get(alg_name, 0) + 1

            # Success rate
            successful_optimizations = sum(
                1 for r in self._optimization_history if r.converged
            )
            success_rate = successful_optimizations / total_optimizations

            return {
                "total_optimizations": total_optimizations,
                "total_time": total_time,
                "average_time": avg_time,
                "success_rate": success_rate,
                "algorithm_usage": algorithm_counts,
                "available_algorithms": [
                    alg.value for alg in self.get_available_algorithms()
                ],
                "last_optimization": self._optimization_history[-1].get_summary()
                if self._optimization_history
                else None,
            }

        except Exception as e:
            logger.error(f"Lỗi get engine statistics: {e}")
            return {"error": str(e)}


# Factory functions
def create_optimization_engine(
    settings: Optional[OptimizationSettings] = None,
) -> AdvancedOptimizationEngine:
    """Factory function để tạo Advanced Optimization Engine."""
    return AdvancedOptimizationEngine(settings)


def create_optimizer(
    algorithm: OptimizationAlgorithm, settings: Optional[OptimizationSettings] = None
) -> BaseOptimizer:
    """Factory function để tạo specific optimizer."""
    if algorithm in [
        OptimizationAlgorithm.LBFGS,
        OptimizationAlgorithm.CONJUGATE_GRADIENT,
        OptimizationAlgorithm.GRADIENT_DESCENT,
    ]:
        return GradientBasedOptimizer(settings)
    elif algorithm in [
        OptimizationAlgorithm.DIFFERENTIAL_EVOLUTION,
        OptimizationAlgorithm.SIMULATED_ANNEALING,
        OptimizationAlgorithm.GENETIC_ALGORITHM,
    ]:
        return GlobalOptimizer(settings)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def create_sample_optimization() -> OptimizationResult:
    """Tạo sample optimization result để test."""
    optimized_weights = np.array([120.5, 98.3, 135.7, 88.9, 142.1, 76.4, 99.8])

    return OptimizationResult(
        optimized_weights=optimized_weights,
        optimized_parameters={"beam_weights": optimized_weights.tolist()},
        converged=True,
        iterations_used=85,
        final_objective_value=234.7,
        optimization_time=125.3,
        algorithm_used=OptimizationAlgorithm.LBFGS,
        objective_function_history=[500.2, 421.1, 356.8, 298.4, 267.3, 245.1, 234.7],
        gradient_norm_history=[12.5, 8.3, 5.1, 2.7, 1.2, 0.8, 0.3],
    )


if __name__ == "__main__":
    # Test code
    logging.basicConfig(level=logging.INFO)

    # Test optimization engine
    engine = create_optimization_engine()

    print(
        f"Available algorithms: {[alg.value for alg in engine.get_available_algorithms()]}"
    )

    # Test optimization với sample data
    sample_objectives = [
        OptimizationObjective(
            structure_name="PTV", objective_type="MAXIMIZE", dose_level=50.0, weight=2.0
        ),
        OptimizationObjective(
            structure_name="OAR", objective_type="MINIMIZE", dose_level=20.0, weight=1.0
        ),
    ]

    sample_beams = [BeamData() for _ in range(7)]
    sample_structures = [Structure()]

    # Test với L-BFGS
    result = engine.optimize(
        algorithm=OptimizationAlgorithm.LBFGS,
        objectives=sample_objectives,
        beam_data=sample_beams,
        structures=sample_structures,
    )

    if result:
        print(f"Optimization completed:")
        print(f"  Algorithm: {result.algorithm_used.value}")
        print(f"  Time: {result.optimization_time:.2f}s")
        print(f"  Converged: {result.converged}")
        print(f"  Final objective: {result.final_objective_value:.2f}")
        print(f"  Iterations: {result.iterations_used}")

    # Test engine statistics
    stats = engine.get_engine_statistics()
    print(f"Engine statistics: {stats}")

    print("Advanced Optimization Engine test hoàn thành!")
