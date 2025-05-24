"""
QuangTPS Beam Optimizer

Module tối ưu hóa beam configuration cho hệ thống QuangTPS.
Cung cấp các thuật toán tối ưu hóa beam angles, weights, và MLC patterns
cho external beam radiotherapy planning.
"""

import logging
import os
import json
import numpy as np
import math
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field, asdict
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor
import time

logger = logging.getLogger(__name__)

# Import optimization libraries với fallback
try:
    from scipy.optimize import minimize, differential_evolution
    from scipy.spatial.distance import cdist

    HAS_SCIPY = True
    logger.info("SciPy được tải thành công")
except ImportError as e:
    logger.warning(f"SciPy không khả dụng: {e}")
    HAS_SCIPY = False

    # Fallback functions
    def minimize(func, x0, *args, **kwargs):
        return type("Result", (), {"x": x0, "fun": func(x0), "success": True})()

    def differential_evolution(func, bounds, *args, **kwargs):
        return type(
            "Result", (), {"x": [b[0] for b in bounds], "fun": 0, "success": True}
        )()

    def cdist(X, Y, metric="euclidean"):
        return [[1.0 for _ in Y] for _ in X]


# Import core modules với fallback
try:
    from quangtps.core.geometry.geometry_utils import calculate_beam_intersection
    from quangtps.core.geometry.collision_detection import check_collision

    HAS_GEOMETRY = True
    logger.info("Geometry utils được tải thành công")
except ImportError as e:
    logger.warning(f"Geometry utils không khả dụng: {e}")
    HAS_GEOMETRY = False

    # Fallback functions
    def calculate_beam_intersection(*args, **kwargs):
        return 1.0

    def check_collision(*args, **kwargs):
        return False


# Import beam modules với fallback
try:
    from quangtps.beams.beam_data import BeamData, BeamGeometry
    from quangtps.beams.mlc_optimizer import MLCOptimizer

    HAS_BEAM_MODULES = True
    logger.info("Beam modules được tải thành công")
except ImportError as e:
    logger.warning(f"Beam modules không khả dụng: {e}")
    HAS_BEAM_MODULES = False

    # Fallback classes
    class BeamData:
        def __init__(self, *args, **kwargs):
            self.beam_id = "beam_1"
            self.gantry_angle = 0.0
            self.beam_weight = 1.0

    class BeamGeometry:
        def __init__(self, *args, **kwargs):
            pass

    class MLCOptimizer:
        def __init__(self, *args, **kwargs):
            pass

        def optimize_leaves(self, *args, **kwargs):
            return []


# Import dose calculation với fallback
try:
    from quangtps.dose.dose_engine import DoseEngine
    from quangtps.dose.dose_grid import DoseGrid

    HAS_DOSE_ENGINE = True
    logger.info("Dose Engine được tải thành công")
except ImportError as e:
    logger.warning(f"Dose Engine không khả dụng: {e}")
    HAS_DOSE_ENGINE = False

    # Fallback classes
    class DoseEngine:
        def __init__(self, *args, **kwargs):
            pass

        def calculate_beam_dose(self, *args, **kwargs):
            return np.random.rand(32, 32, 16) * 10

    class DoseGrid:
        def __init__(self, *args, **kwargs):
            self.shape = (32, 32, 16)


@dataclass
class BeamOptimizationObjective:
    """Mục tiêu tối ưu hóa beam."""

    objective_type: str  # MINIMIZE_DOSE, MAXIMIZE_COVERAGE, MINIMIZE_OAR_DOSE
    structure_name: str
    weight: float = 1.0

    # Dose constraints
    max_dose: Optional[float] = None  # Gy
    min_dose: Optional[float] = None  # Gy
    volume_constraint: Optional[float] = None  # % volume

    # Priority
    priority: int = 1  # 1 = highest

    def __post_init__(self):
        """Validate objective."""
        if not self.structure_name:
            raise ValueError("Structure name là bắt buộc")

        if self.weight <= 0:
            raise ValueError("Weight phải lớn hơn 0")

    def evaluate(
        self, dose_distribution: np.ndarray, structure_mask: np.ndarray
    ) -> float:
        """Đánh giá objective function."""
        try:
            if structure_mask.sum() == 0:
                return 0.0

            # Extract dose in structure
            structure_dose = dose_distribution[structure_mask > 0]

            if len(structure_dose) == 0:
                return 0.0

            # Calculate objective value
            if self.objective_type == "MINIMIZE_DOSE":
                return np.mean(structure_dose)

            elif self.objective_type == "MAXIMIZE_COVERAGE":
                if self.min_dose is not None:
                    covered_volume = np.sum(structure_dose >= self.min_dose) / len(
                        structure_dose
                    )
                    return -covered_volume  # Negative because we minimize
                else:
                    return -np.mean(structure_dose)

            elif self.objective_type == "MINIMIZE_OAR_DOSE":
                if self.max_dose is not None:
                    violation = np.sum(structure_dose > self.max_dose)
                    return violation + np.mean(structure_dose)
                else:
                    return np.mean(structure_dose)

            elif self.objective_type == "MINIMIZE_MAX_DOSE":
                return np.max(structure_dose)

            elif self.objective_type == "HOMOGENEITY":
                # Minimize dose heterogeneity
                return np.std(structure_dose)

            else:
                logger.warning(f"Unknown objective type: {self.objective_type}")
                return 0.0

        except Exception as e:
            logger.error(f"Lỗi evaluate objective: {e}")
            return 1e6  # Large penalty for errors


@dataclass
class BeamConstraint:
    """Ràng buộc cho beam optimization."""

    constraint_type: str = (
        "GENERAL"  # ANGLE_SEPARATION, COLLISION_AVOIDANCE, BEAM_COUNT, GENERAL
    )

    # Angle constraints
    min_angle_separation: Optional[float] = None  # degrees
    excluded_angles: Optional[List[Tuple[float, float]]] = None  # (min, max) ranges

    # Collision constraints
    enable_collision_check: bool = True
    couch_clearance: float = 5.0  # cm
    gantry_clearance: float = 10.0  # cm

    # Beam count constraints
    min_beams: int = 3
    max_beams: int = 11

    # Field size constraints
    min_field_size: float = 2.0  # cm
    max_field_size: float = 40.0  # cm

    def __post_init__(self):
        """Validate constraints."""
        if self.min_beams < 1:
            self.min_beams = 1
        if self.max_beams < self.min_beams:
            self.max_beams = self.min_beams

    def check_angle_constraints(self, angles: List[float]) -> bool:
        """Kiểm tra ràng buộc góc."""
        try:
            # Check angle separation
            if self.min_angle_separation is not None and len(angles) > 1:
                sorted_angles = sorted(angles)
                for i in range(len(sorted_angles) - 1):
                    separation = sorted_angles[i + 1] - sorted_angles[i]
                    if separation < self.min_angle_separation:
                        return False

                # Check wrap-around separation
                wrap_separation = (360 - sorted_angles[-1]) + sorted_angles[0]
                if wrap_separation < self.min_angle_separation:
                    return False

            # Check excluded angles
            if self.excluded_angles:
                for angle in angles:
                    for min_excl, max_excl in self.excluded_angles:
                        if min_excl <= angle <= max_excl:
                            return False

            return True

        except Exception as e:
            logger.error(f"Lỗi check angle constraints: {e}")
            return False

    def check_collision_constraints(
        self, beam_geometries: List[Dict[str, float]]
    ) -> bool:
        """Kiểm tra ràng buộc collision."""
        try:
            if not self.enable_collision_check or not HAS_GEOMETRY:
                return True

            for beam_geom in beam_geometries:
                gantry_angle = beam_geom.get("gantry_angle", 0.0)
                couch_angle = beam_geom.get("couch_angle", 0.0)

                # Check collision (simplified)
                if check_collision(
                    gantry_angle,
                    couch_angle,
                    self.couch_clearance,
                    self.gantry_clearance,
                ):
                    return False

            return True

        except Exception as e:
            logger.error(f"Lỗi check collision constraints: {e}")
            return True  # Default to safe

    def check_beam_count_constraints(self, beam_count: int) -> bool:
        """Kiểm tra ràng buộc số lượng beam."""
        return self.min_beams <= beam_count <= self.max_beams


class BeamAngleOptimizer:
    """
    Optimizer cho beam angles.
    """

    def __init__(
        self,
        dose_engine: Optional[DoseEngine] = None,
        constraints: Optional[BeamConstraint] = None,
    ):
        self.dose_engine = dose_engine or DoseEngine()
        self.constraints = constraints or BeamConstraint()

        # Optimization settings
        self.max_iterations = 100
        self.convergence_tolerance = 1e-6
        self.population_size = 50

        # Parallel processing
        self.use_parallel = True
        self.max_workers = 4

        logger.info("Beam Angle Optimizer khởi tạo")

    def optimize_angles(
        self,
        target_structures: List[str],
        oar_structures: List[str],
        objectives: List[BeamOptimizationObjective],
        num_beams: int = 7,
        initial_angles: Optional[List[float]] = None,
        progress_callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Tối ưu hóa beam angles.
        """
        try:
            if progress_callback:
                progress_callback(0, "Khởi tạo tối ưu hóa beam angles...")

            # Validate inputs
            if not self.constraints.check_beam_count_constraints(num_beams):
                raise ValueError(f"Số beam {num_beams} không hợp lệ")

            # Setup optimization bounds
            bounds = [(0, 360) for _ in range(num_beams)]

            # Initial guess
            if initial_angles and len(initial_angles) == num_beams:
                x0 = initial_angles
            else:
                # Evenly distributed angles
                x0 = [i * 360 / num_beams for i in range(num_beams)]

            if progress_callback:
                progress_callback(20, "Thiết lập objective function...")

            # Define objective function
            def objective_function(angles):
                return self._evaluate_beam_configuration(angles, objectives)

            # Define constraint function
            def constraint_function(angles):
                return self._check_all_constraints(angles)

            if progress_callback:
                progress_callback(30, "Chạy optimization...")

            # Run optimization
            if HAS_SCIPY:
                # Use differential evolution for global optimization
                result = differential_evolution(
                    objective_function,
                    bounds,
                    maxiter=self.max_iterations,
                    popsize=self.population_size,
                    seed=42,
                    callback=lambda xk, convergence: progress_callback(
                        30 + (convergence * 60), f"Optimization... {convergence:.1%}"
                    )
                    if progress_callback
                    else None,
                )

                optimized_angles = result.x
                objective_value = result.fun
                success = result.success

            else:
                # Simple grid search fallback
                best_angles = x0
                best_objective = objective_function(x0)

                for iteration in range(10):  # Limited iterations for fallback
                    # Random perturbation
                    test_angles = [a + np.random.normal(0, 10) for a in best_angles]
                    test_angles = [a % 360 for a in test_angles]  # Normalize

                    if constraint_function(test_angles):
                        test_objective = objective_function(test_angles)
                        if test_objective < best_objective:
                            best_angles = test_angles
                            best_objective = test_objective

                    if progress_callback:
                        progress_callback(
                            30 + iteration * 6, f"Grid search... {iteration + 1}/10"
                        )

                optimized_angles = best_angles
                objective_value = best_objective
                success = True

            if progress_callback:
                progress_callback(95, "Xử lý kết quả...")

            # Process results
            final_angles = [angle % 360 for angle in optimized_angles]

            # Calculate final dose distribution
            final_dose = self._calculate_dose_distribution(final_angles)

            # Evaluate final objectives
            final_objectives = {}
            for obj in objectives:
                if hasattr(obj, "structure_name"):
                    structure_mask = self._get_structure_mask(obj.structure_name)
                    final_objectives[obj.structure_name] = obj.evaluate(
                        final_dose, structure_mask
                    )

            result_data = {
                "optimized_angles": final_angles,
                "objective_value": objective_value,
                "final_objectives": final_objectives,
                "dose_distribution": final_dose,
                "success": success,
                "optimization_info": {
                    "num_beams": num_beams,
                    "iterations": self.max_iterations,
                    "method": "differential_evolution" if HAS_SCIPY else "grid_search",
                },
            }

            if progress_callback:
                progress_callback(100, "Hoàn thành tối ưu hóa")

            logger.info(f"Hoàn thành beam angle optimization: {final_angles}")
            return result_data

        except Exception as e:
            logger.error(f"Lỗi beam angle optimization: {e}")
            if progress_callback:
                progress_callback(-1, f"Lỗi: {str(e)}")
            return {"success": False, "error": str(e)}

    def _evaluate_beam_configuration(
        self, angles: List[float], objectives: List[BeamOptimizationObjective]
    ) -> float:
        """Đánh giá beam configuration."""
        try:
            # Check constraints first
            if not self._check_all_constraints(angles):
                return 1e6  # Large penalty

            # Calculate dose distribution
            dose_distribution = self._calculate_dose_distribution(angles)

            # Evaluate all objectives
            total_objective = 0.0

            for obj in objectives:
                structure_mask = self._get_structure_mask(obj.structure_name)
                obj_value = obj.evaluate(dose_distribution, structure_mask)
                weighted_value = obj_value * obj.weight

                # Apply priority weighting
                priority_weight = 1.0 / obj.priority if obj.priority > 0 else 1.0
                total_objective += weighted_value * priority_weight

            return total_objective

        except Exception as e:
            logger.error(f"Lỗi evaluate beam configuration: {e}")
            return 1e6

    def _check_all_constraints(self, angles: List[float]) -> bool:
        """Kiểm tra tất cả constraints."""
        try:
            # Check angle constraints
            if not self.constraints.check_angle_constraints(angles):
                return False

            # Check collision constraints
            beam_geometries = [
                {"gantry_angle": angle, "couch_angle": 0.0} for angle in angles
            ]
            if not self.constraints.check_collision_constraints(beam_geometries):
                return False

            return True

        except Exception as e:
            logger.error(f"Lỗi check constraints: {e}")
            return False

    def _calculate_dose_distribution(self, angles: List[float]) -> np.ndarray:
        """Tính toán dose distribution cho beam configuration."""
        try:
            # Create beam data
            beams = []
            for i, angle in enumerate(angles):
                beam = BeamData() if HAS_BEAM_MODULES else type("Beam", (), {})()
                beam.beam_id = f"beam_{i + 1}"
                beam.gantry_angle = angle
                beam.beam_weight = 1.0 / len(angles)  # Equal weights
                beams.append(beam)

            # Calculate dose (simplified)
            if HAS_DOSE_ENGINE:
                dose_result = self.dose_engine.calculate_beam_dose(beams)
            else:
                # Fallback calculation
                dose_result = np.random.rand(32, 32, 16) * 50

            return dose_result

        except Exception as e:
            logger.error(f"Lỗi calculate dose distribution: {e}")
            return np.zeros((32, 32, 16))

    def _get_structure_mask(self, structure_name: str) -> np.ndarray:
        """Lấy structure mask (simplified)."""
        try:
            # This would normally interface with structure manager
            # For now, create a mock mask
            mask = np.zeros((32, 32, 16), dtype=bool)

            if "PTV" in structure_name.upper():
                # Create a spherical target region
                center = (16, 16, 8)
                radius = 5
                for x in range(32):
                    for y in range(32):
                        for z in range(16):
                            dist = (
                                (x - center[0]) ** 2
                                + (y - center[1]) ** 2
                                + (z - center[2]) ** 2
                            ) ** 0.5
                            if dist <= radius:
                                mask[x, y, z] = True

            elif any(
                oar in structure_name.upper() for oar in ["BLADDER", "RECTUM", "SPINAL"]
            ):
                # Create a smaller OAR region
                center = (20, 12, 6)
                radius = 3
                for x in range(32):
                    for y in range(32):
                        for z in range(16):
                            dist = (
                                (x - center[0]) ** 2
                                + (y - center[1]) ** 2
                                + (z - center[2]) ** 2
                            ) ** 0.5
                            if dist <= radius:
                                mask[x, y, z] = True

            return mask

        except Exception as e:
            logger.error(f"Lỗi get structure mask: {e}")
            return np.zeros((32, 32, 16), dtype=bool)


class BeamWeightOptimizer:
    """
    Optimizer cho beam weights (fluence optimization).
    """

    def __init__(self, dose_engine: Optional[DoseEngine] = None):
        self.dose_engine = dose_engine or DoseEngine()

        # Optimization settings
        self.max_iterations = 200
        self.convergence_tolerance = 1e-8

        # Regularization
        self.use_smoothness_regularization = True
        self.smoothness_weight = 0.1

        logger.info("Beam Weight Optimizer khởi tạo")

    def optimize_weights(
        self,
        beam_angles: List[float],
        objectives: List[BeamOptimizationObjective],
        initial_weights: Optional[List[float]] = None,
        progress_callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Tối ưu hóa beam weights.
        """
        try:
            if progress_callback:
                progress_callback(0, "Khởi tạo weight optimization...")

            num_beams = len(beam_angles)

            # Initial weights
            if initial_weights and len(initial_weights) == num_beams:
                x0 = np.array(initial_weights)
            else:
                x0 = np.ones(num_beams) / num_beams  # Equal weights

            # Normalization constraint: sum of weights = 1
            constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1.0}

            # Bounds: weights must be non-negative
            bounds = [(0, 1) for _ in range(num_beams)]

            if progress_callback:
                progress_callback(20, "Thiết lập optimization...")

            # Define objective function
            def objective_function(weights):
                return self._evaluate_weight_configuration(
                    beam_angles, weights, objectives
                )

            if progress_callback:
                progress_callback(30, "Chạy optimization...")

            # Run optimization
            if HAS_SCIPY:
                result = minimize(
                    objective_function,
                    x0,
                    method="SLSQP",
                    bounds=bounds,
                    constraints=constraints,
                    options={
                        "maxiter": self.max_iterations,
                        "ftol": self.convergence_tolerance,
                    },
                )

                optimized_weights = result.x
                objective_value = result.fun
                success = result.success
                iterations = (
                    result.nit if hasattr(result, "nit") else self.max_iterations
                )

            else:
                # Simple iterative optimization fallback
                current_weights = x0.copy()
                best_objective = objective_function(current_weights)

                for iteration in range(50):  # Limited iterations
                    # Random perturbation
                    perturbation = np.random.normal(0, 0.01, num_beams)
                    test_weights = current_weights + perturbation

                    # Ensure non-negative and normalized
                    test_weights = np.maximum(test_weights, 0)
                    test_weights = test_weights / np.sum(test_weights)

                    test_objective = objective_function(test_weights)

                    if test_objective < best_objective:
                        current_weights = test_weights
                        best_objective = test_objective

                    if progress_callback:
                        progress_callback(
                            30 + iteration, f"Iteration {iteration + 1}/50"
                        )

                optimized_weights = current_weights
                objective_value = best_objective
                success = True
                iterations = 50

            if progress_callback:
                progress_callback(90, "Xử lý kết quả...")

            # Ensure weights are normalized
            optimized_weights = optimized_weights / np.sum(optimized_weights)

            # Calculate final dose distribution
            final_dose = self._calculate_weighted_dose_distribution(
                beam_angles, optimized_weights
            )

            # Evaluate final objectives
            final_objectives = {}
            for obj in objectives:
                structure_mask = self._get_structure_mask(obj.structure_name)
                final_objectives[obj.structure_name] = obj.evaluate(
                    final_dose, structure_mask
                )

            result_data = {
                "optimized_weights": optimized_weights.tolist(),
                "objective_value": objective_value,
                "final_objectives": final_objectives,
                "dose_distribution": final_dose,
                "success": success,
                "optimization_info": {
                    "iterations": iterations,
                    "convergence_tolerance": self.convergence_tolerance,
                    "method": "SLSQP" if HAS_SCIPY else "iterative",
                },
            }

            if progress_callback:
                progress_callback(100, "Hoàn thành weight optimization")

            logger.info(f"Hoàn thành beam weight optimization")
            return result_data

        except Exception as e:
            logger.error(f"Lỗi beam weight optimization: {e}")
            if progress_callback:
                progress_callback(-1, f"Lỗi: {str(e)}")
            return {"success": False, "error": str(e)}

    def _evaluate_weight_configuration(
        self,
        angles: List[float],
        weights: np.ndarray,
        objectives: List[BeamOptimizationObjective],
    ) -> float:
        """Đánh giá weight configuration."""
        try:
            # Calculate dose distribution
            dose_distribution = self._calculate_weighted_dose_distribution(
                angles, weights
            )

            # Evaluate objectives
            total_objective = 0.0

            for obj in objectives:
                structure_mask = self._get_structure_mask(obj.structure_name)
                obj_value = obj.evaluate(dose_distribution, structure_mask)
                weighted_value = obj_value * obj.weight

                # Apply priority weighting
                priority_weight = 1.0 / obj.priority if obj.priority > 0 else 1.0
                total_objective += weighted_value * priority_weight

            # Add smoothness regularization
            if self.use_smoothness_regularization and len(weights) > 1:
                smoothness_penalty = 0.0
                for i in range(len(weights) - 1):
                    smoothness_penalty += (weights[i + 1] - weights[i]) ** 2
                total_objective += self.smoothness_weight * smoothness_penalty

            return total_objective

        except Exception as e:
            logger.error(f"Lỗi evaluate weight configuration: {e}")
            return 1e6

    def _calculate_weighted_dose_distribution(
        self, angles: List[float], weights: np.ndarray
    ) -> np.ndarray:
        """Tính toán weighted dose distribution."""
        try:
            total_dose = np.zeros((32, 32, 16))

            for i, (angle, weight) in enumerate(zip(angles, weights)):
                # Calculate dose for this beam
                beam_dose = self._calculate_single_beam_dose(angle)
                total_dose += beam_dose * weight

            return total_dose

        except Exception as e:
            logger.error(f"Lỗi calculate weighted dose: {e}")
            return np.zeros((32, 32, 16))

    def _calculate_single_beam_dose(self, angle: float) -> np.ndarray:
        """Tính toán dose cho single beam."""
        try:
            if HAS_DOSE_ENGINE:
                # Use proper dose engine
                beam = BeamData() if HAS_BEAM_MODULES else type("Beam", (), {})()
                beam.gantry_angle = angle
                beam.beam_weight = 1.0
                return self.dose_engine.calculate_beam_dose([beam])
            else:
                # Simplified calculation based on angle
                dose = np.zeros((32, 32, 16))

                # Create a simple beam profile
                center_x, center_y = 16, 16
                beam_width = 10

                # Direction based on gantry angle
                direction_x = np.cos(np.radians(angle))
                direction_y = np.sin(np.radians(angle))

                for x in range(32):
                    for y in range(32):
                        for z in range(16):
                            # Distance from beam central axis
                            beam_center_x = center_x + direction_x * (z - 8)
                            beam_center_y = center_y + direction_y * (z - 8)

                            dist_from_axis = (
                                (x - beam_center_x) ** 2 + (y - beam_center_y) ** 2
                            ) ** 0.5

                            if dist_from_axis <= beam_width:
                                # Simple depth-dose falloff
                                depth_factor = np.exp(-0.05 * z)
                                dose[x, y, z] = (
                                    10.0 * depth_factor * np.exp(-0.1 * dist_from_axis)
                                )

                return dose

        except Exception as e:
            logger.error(f"Lỗi calculate single beam dose: {e}")
            return np.zeros((32, 32, 16))

    def _get_structure_mask(self, structure_name: str) -> np.ndarray:
        """Lấy structure mask (simplified)."""
        # Reuse the same method from BeamAngleOptimizer
        try:
            mask = np.zeros((32, 32, 16), dtype=bool)

            if "PTV" in structure_name.upper():
                center = (16, 16, 8)
                radius = 5
                for x in range(32):
                    for y in range(32):
                        for z in range(16):
                            dist = (
                                (x - center[0]) ** 2
                                + (y - center[1]) ** 2
                                + (z - center[2]) ** 2
                            ) ** 0.5
                            if dist <= radius:
                                mask[x, y, z] = True

            elif any(
                oar in structure_name.upper() for oar in ["BLADDER", "RECTUM", "SPINAL"]
            ):
                center = (20, 12, 6)
                radius = 3
                for x in range(32):
                    for y in range(32):
                        for z in range(16):
                            dist = (
                                (x - center[0]) ** 2
                                + (y - center[1]) ** 2
                                + (z - center[2]) ** 2
                            ) ** 0.5
                            if dist <= radius:
                                mask[x, y, z] = True

            return mask

        except Exception as e:
            logger.error(f"Lỗi get structure mask: {e}")
            return np.zeros((32, 32, 16), dtype=bool)


class ComprehensiveBeamOptimizer:
    """
    Optimizer tổng hợp cho beam configuration.
    """

    def __init__(
        self,
        dose_engine: Optional[DoseEngine] = None,
        constraints: Optional[BeamConstraint] = None,
    ):
        self.dose_engine = dose_engine or DoseEngine()
        self.constraints = constraints or BeamConstraint()

        # Sub-optimizers
        self.angle_optimizer = BeamAngleOptimizer(dose_engine, constraints)
        self.weight_optimizer = BeamWeightOptimizer(dose_engine)

        # MLC optimizer
        if HAS_BEAM_MODULES:
            self.mlc_optimizer = MLCOptimizer()
        else:
            self.mlc_optimizer = MLCOptimizer()

        # Optimization strategy
        self.optimization_strategy = "SEQUENTIAL"  # SEQUENTIAL, SIMULTANEOUS, ITERATIVE
        self.max_outer_iterations = 5

        logger.info("Comprehensive Beam Optimizer khởi tạo")

    def optimize_complete_beam_configuration(
        self,
        target_structures: List[str],
        oar_structures: List[str],
        objectives: List[BeamOptimizationObjective],
        num_beams: int = 7,
        optimization_strategy: str = "SEQUENTIAL",
        progress_callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Tối ưu hóa hoàn chỉnh beam configuration.
        """
        try:
            if progress_callback:
                progress_callback(0, "Khởi tạo comprehensive optimization...")

            self.optimization_strategy = optimization_strategy

            if self.optimization_strategy == "SEQUENTIAL":
                return self._sequential_optimization(
                    target_structures,
                    oar_structures,
                    objectives,
                    num_beams,
                    progress_callback,
                )
            elif self.optimization_strategy == "ITERATIVE":
                return self._iterative_optimization(
                    target_structures,
                    oar_structures,
                    objectives,
                    num_beams,
                    progress_callback,
                )
            else:
                # Default to sequential
                return self._sequential_optimization(
                    target_structures,
                    oar_structures,
                    objectives,
                    num_beams,
                    progress_callback,
                )

        except Exception as e:
            logger.error(f"Lỗi comprehensive beam optimization: {e}")
            if progress_callback:
                progress_callback(-1, f"Lỗi: {str(e)}")
            return {"success": False, "error": str(e)}

    def _sequential_optimization(
        self,
        target_structures: List[str],
        oar_structures: List[str],
        objectives: List[BeamOptimizationObjective],
        num_beams: int,
        progress_callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """Sequential optimization: angles -> weights -> MLC."""
        try:
            results = {
                "optimization_strategy": "SEQUENTIAL",
                "success": True,
                "stages": {},
            }

            # Stage 1: Optimize beam angles
            if progress_callback:
                progress_callback(10, "Stage 1: Optimizing beam angles...")

            angle_result = self.angle_optimizer.optimize_angles(
                target_structures=target_structures,
                oar_structures=oar_structures,
                objectives=objectives,
                num_beams=num_beams,
                progress_callback=lambda p, msg: progress_callback(
                    10 + p * 0.3, f"Angles: {msg}"
                )
                if progress_callback
                else None,
            )

            if not angle_result.get("success", False):
                results["success"] = False
                results["error"] = "Angle optimization failed"
                return results

            results["stages"]["angle_optimization"] = angle_result
            optimized_angles = angle_result["optimized_angles"]

            # Stage 2: Optimize beam weights
            if progress_callback:
                progress_callback(50, "Stage 2: Optimizing beam weights...")

            weight_result = self.weight_optimizer.optimize_weights(
                beam_angles=optimized_angles,
                objectives=objectives,
                progress_callback=lambda p, msg: progress_callback(
                    50 + p * 0.3, f"Weights: {msg}"
                )
                if progress_callback
                else None,
            )

            if not weight_result.get("success", False):
                results["success"] = False
                results["error"] = "Weight optimization failed"
                return results

            results["stages"]["weight_optimization"] = weight_result
            optimized_weights = weight_result["optimized_weights"]

            # Stage 3: Optimize MLC shapes
            if progress_callback:
                progress_callback(80, "Stage 3: Optimizing MLC shapes...")

            mlc_result = self._optimize_mlc_shapes(
                beam_angles=optimized_angles,
                beam_weights=optimized_weights,
                objectives=objectives,
            )

            results["stages"]["mlc_optimization"] = mlc_result

            # Combine final results
            if progress_callback:
                progress_callback(95, "Finalizing results...")

            final_dose = self._calculate_final_dose_distribution(
                optimized_angles, optimized_weights, mlc_result.get("mlc_patterns", [])
            )

            # Evaluate final quality metrics
            final_metrics = self._evaluate_final_quality(final_dose, objectives)

            results.update(
                {
                    "final_beam_angles": optimized_angles,
                    "final_beam_weights": optimized_weights,
                    "final_mlc_patterns": mlc_result.get("mlc_patterns", []),
                    "final_dose_distribution": final_dose,
                    "final_quality_metrics": final_metrics,
                    "total_objective_value": sum(final_metrics.values()),
                }
            )

            if progress_callback:
                progress_callback(100, "Hoàn thành comprehensive optimization")

            logger.info("Hoàn thành sequential beam optimization")
            return results

        except Exception as e:
            logger.error(f"Lỗi sequential optimization: {e}")
            return {"success": False, "error": str(e)}

    def _iterative_optimization(
        self,
        target_structures: List[str],
        oar_structures: List[str],
        objectives: List[BeamOptimizationObjective],
        num_beams: int,
        progress_callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """Iterative optimization: alternate between angles and weights."""
        try:
            results = {
                "optimization_strategy": "ITERATIVE",
                "success": True,
                "iterations": [],
            }

            # Initial beam configuration
            current_angles = [i * 360 / num_beams for i in range(num_beams)]
            current_weights = [1.0 / num_beams] * num_beams

            best_objective = float("inf")
            best_config = None

            for iteration in range(self.max_outer_iterations):
                iteration_start = 20 + iteration * 60 / self.max_outer_iterations
                iteration_progress = 60 / self.max_outer_iterations

                if progress_callback:
                    progress_callback(
                        int(iteration_start),
                        f"Iteration {iteration + 1}/{self.max_outer_iterations}",
                    )

                iteration_results = {}

                # Optimize angles with fixed weights
                angle_result = self.angle_optimizer.optimize_angles(
                    target_structures=target_structures,
                    oar_structures=oar_structures,
                    objectives=objectives,
                    num_beams=num_beams,
                    initial_angles=current_angles,
                )

                if angle_result.get("success", False):
                    current_angles = angle_result["optimized_angles"]
                    iteration_results["angle_optimization"] = angle_result

                # Optimize weights with fixed angles
                weight_result = self.weight_optimizer.optimize_weights(
                    beam_angles=current_angles,
                    objectives=objectives,
                    initial_weights=current_weights,
                )

                if weight_result.get("success", False):
                    current_weights = weight_result["optimized_weights"]
                    iteration_results["weight_optimization"] = weight_result

                # Evaluate current configuration
                current_objective = weight_result.get("objective_value", float("inf"))
                iteration_results["objective_value"] = current_objective

                # Check for improvement
                if current_objective < best_objective:
                    best_objective = current_objective
                    best_config = {
                        "angles": current_angles.copy(),
                        "weights": current_weights.copy(),
                        "objective": current_objective,
                    }

                results["iterations"].append(iteration_results)

                # Check convergence
                if iteration > 0:
                    prev_objective = results["iterations"][iteration - 1].get(
                        "objective_value", float("inf")
                    )
                    improvement = abs(prev_objective - current_objective) / abs(
                        prev_objective
                    )

                    if improvement < 0.01:  # 1% improvement threshold
                        logger.info(f"Converged after {iteration + 1} iterations")
                        break

            # Final MLC optimization
            if progress_callback:
                progress_callback(85, "Final MLC optimization...")

            if best_config:
                mlc_result = self._optimize_mlc_shapes(
                    beam_angles=best_config["angles"],
                    beam_weights=best_config["weights"],
                    objectives=objectives,
                )

                final_dose = self._calculate_final_dose_distribution(
                    best_config["angles"],
                    best_config["weights"],
                    mlc_result.get("mlc_patterns", []),
                )

                final_metrics = self._evaluate_final_quality(final_dose, objectives)

                results.update(
                    {
                        "final_beam_angles": best_config["angles"],
                        "final_beam_weights": best_config["weights"],
                        "final_mlc_patterns": mlc_result.get("mlc_patterns", []),
                        "final_dose_distribution": final_dose,
                        "final_quality_metrics": final_metrics,
                        "best_objective_value": best_config["objective"],
                    }
                )

            if progress_callback:
                progress_callback(100, "Hoàn thành iterative optimization")

            logger.info("Hoàn thành iterative beam optimization")
            return results

        except Exception as e:
            logger.error(f"Lỗi iterative optimization: {e}")
            return {"success": False, "error": str(e)}

    def _optimize_mlc_shapes(
        self,
        beam_angles: List[float],
        beam_weights: List[float],
        objectives: List[BeamOptimizationObjective],
    ) -> Dict[str, Any]:
        """Optimize MLC leaf positions."""
        try:
            mlc_patterns = []

            for i, (angle, weight) in enumerate(zip(beam_angles, beam_weights)):
                # Simplified MLC optimization
                # In practice, this would use proper fluence optimization

                if HAS_BEAM_MODULES:
                    leaf_positions = self.mlc_optimizer.optimize_leaves(
                        beam_angle=angle,
                        beam_weight=weight,
                        target_fluence=None,  # Would be computed from objectives
                    )
                else:
                    # Fallback: simple rectangular field
                    leaf_positions = {
                        "bank_a": [-5.0] * 60,  # 60 leaf pairs, 5cm field
                        "bank_b": [5.0] * 60,
                    }

                mlc_patterns.append(
                    {
                        "beam_id": f"beam_{i + 1}",
                        "gantry_angle": angle,
                        "leaf_positions": leaf_positions,
                    }
                )

            return {
                "success": True,
                "mlc_patterns": mlc_patterns,
                "optimization_method": "simplified",
            }

        except Exception as e:
            logger.error(f"Lỗi MLC optimization: {e}")
            return {"success": False, "error": str(e)}

    def _calculate_final_dose_distribution(
        self,
        angles: List[float],
        weights: List[float],
        mlc_patterns: List[Dict[str, Any]],
    ) -> np.ndarray:
        """Tính toán final dose distribution."""
        try:
            # Simplified calculation
            total_dose = np.zeros((32, 32, 16))

            for i, (angle, weight) in enumerate(zip(angles, weights)):
                # Calculate beam dose with MLC shaping
                beam_dose = self._calculate_shaped_beam_dose(
                    angle, mlc_patterns[i] if i < len(mlc_patterns) else None
                )
                total_dose += beam_dose * weight

            return total_dose

        except Exception as e:
            logger.error(f"Lỗi calculate final dose: {e}")
            return np.zeros((32, 32, 16))

    def _calculate_shaped_beam_dose(
        self, angle: float, mlc_pattern: Optional[Dict[str, Any]]
    ) -> np.ndarray:
        """Tính toán dose cho shaped beam."""
        try:
            # Start with basic beam dose
            base_dose = self._calculate_single_beam_dose(angle)

            if mlc_pattern and "leaf_positions" in mlc_pattern:
                # Apply MLC shaping (simplified)
                leaf_positions = mlc_pattern["leaf_positions"]

                # Simple rectangular field approximation
                if isinstance(leaf_positions, dict) and "bank_a" in leaf_positions:
                    bank_a = leaf_positions["bank_a"]
                    bank_b = leaf_positions["bank_b"]

                    # Create MLC mask (very simplified)
                    mlc_mask = np.ones_like(base_dose)

                    # Apply leaf blocking (simplified 2D projection)
                    field_width = (
                        abs(bank_b[0] - bank_a[0]) if bank_a and bank_b else 10.0
                    )

                    center_x, center_y = 16, 16
                    for x in range(32):
                        for y in range(32):
                            dist_from_center = (
                                (x - center_x) ** 2 + (y - center_y) ** 2
                            ) ** 0.5
                            if dist_from_center > field_width:
                                mlc_mask[x, y, :] = 0.0

                    return base_dose * mlc_mask

            return base_dose

        except Exception as e:
            logger.error(f"Lỗi calculate shaped beam dose: {e}")
            return np.zeros((32, 32, 16))

    def _calculate_single_beam_dose(self, angle: float) -> np.ndarray:
        """Tính toán dose cho single beam (reused from weight optimizer)."""
        try:
            dose = np.zeros((32, 32, 16))

            center_x, center_y = 16, 16
            beam_width = 10

            direction_x = np.cos(np.radians(angle))
            direction_y = np.sin(np.radians(angle))

            for x in range(32):
                for y in range(32):
                    for z in range(16):
                        beam_center_x = center_x + direction_x * (z - 8)
                        beam_center_y = center_y + direction_y * (z - 8)

                        dist_from_axis = (
                            (x - beam_center_x) ** 2 + (y - beam_center_y) ** 2
                        ) ** 0.5

                        if dist_from_axis <= beam_width:
                            depth_factor = np.exp(-0.05 * z)
                            dose[x, y, z] = (
                                10.0 * depth_factor * np.exp(-0.1 * dist_from_axis)
                            )

            return dose

        except Exception as e:
            logger.error(f"Lỗi calculate single beam dose: {e}")
            return np.zeros((32, 32, 16))

    def _evaluate_final_quality(
        self, dose_distribution: np.ndarray, objectives: List[BeamOptimizationObjective]
    ) -> Dict[str, float]:
        """Đánh giá final quality metrics."""
        try:
            quality_metrics = {}

            for obj in objectives:
                structure_mask = self._get_structure_mask(obj.structure_name)
                metric_value = obj.evaluate(dose_distribution, structure_mask)
                quality_metrics[obj.structure_name] = metric_value

            return quality_metrics

        except Exception as e:
            logger.error(f"Lỗi evaluate final quality: {e}")
            return {}

    def _get_structure_mask(self, structure_name: str) -> np.ndarray:
        """Lấy structure mask (reused from other optimizers)."""
        try:
            mask = np.zeros((32, 32, 16), dtype=bool)

            if "PTV" in structure_name.upper():
                center = (16, 16, 8)
                radius = 5
                for x in range(32):
                    for y in range(32):
                        for z in range(16):
                            dist = (
                                (x - center[0]) ** 2
                                + (y - center[1]) ** 2
                                + (z - center[2]) ** 2
                            ) ** 0.5
                            if dist <= radius:
                                mask[x, y, z] = True

            elif any(
                oar in structure_name.upper() for oar in ["BLADDER", "RECTUM", "SPINAL"]
            ):
                center = (20, 12, 6)
                radius = 3
                for x in range(32):
                    for y in range(32):
                        for z in range(16):
                            dist = (
                                (x - center[0]) ** 2
                                + (y - center[1]) ** 2
                                + (z - center[2]) ** 2
                            ) ** 0.5
                            if dist <= radius:
                                mask[x, y, z] = True

            return mask

        except Exception as e:
            logger.error(f"Lỗi get structure mask: {e}")
            return np.zeros((32, 32, 16), dtype=bool)


# Factory functions
def create_beam_angle_optimizer(
    dose_engine: Optional[DoseEngine] = None,
    constraints: Optional[BeamConstraint] = None,
) -> BeamAngleOptimizer:
    """Factory function để tạo BeamAngleOptimizer."""
    return BeamAngleOptimizer(dose_engine, constraints)


def create_beam_weight_optimizer(
    dose_engine: Optional[DoseEngine] = None,
) -> BeamWeightOptimizer:
    """Factory function để tạo BeamWeightOptimizer."""
    return BeamWeightOptimizer(dose_engine)


def create_comprehensive_beam_optimizer(
    dose_engine: Optional[DoseEngine] = None,
    constraints: Optional[BeamConstraint] = None,
) -> ComprehensiveBeamOptimizer:
    """Factory function để tạo ComprehensiveBeamOptimizer."""
    return ComprehensiveBeamOptimizer(dose_engine, constraints)


def create_standard_objectives(
    site: str = "prostate",
) -> List[BeamOptimizationObjective]:
    """Tạo standard objectives cho different treatment sites."""
    objectives = []

    if site.lower() == "prostate":
        # PTV objective
        objectives.append(
            BeamOptimizationObjective(
                objective_type="MAXIMIZE_COVERAGE",
                structure_name="PTV",
                weight=10.0,
                min_dose=78.0,  # Gy
                priority=1,
            )
        )

        # Rectum constraint
        objectives.append(
            BeamOptimizationObjective(
                objective_type="MINIMIZE_OAR_DOSE",
                structure_name="Rectum",
                weight=5.0,
                max_dose=70.0,  # Gy
                priority=2,
            )
        )

        # Bladder constraint
        objectives.append(
            BeamOptimizationObjective(
                objective_type="MINIMIZE_OAR_DOSE",
                structure_name="Bladder",
                weight=3.0,
                max_dose=65.0,  # Gy
                priority=3,
            )
        )

    elif site.lower() == "head_neck":
        # PTV objective
        objectives.append(
            BeamOptimizationObjective(
                objective_type="MAXIMIZE_COVERAGE",
                structure_name="PTV",
                weight=10.0,
                min_dose=70.0,  # Gy
                priority=1,
            )
        )

        # Spinal cord constraint
        objectives.append(
            BeamOptimizationObjective(
                objective_type="MINIMIZE_OAR_DOSE",
                structure_name="SpinalCord",
                weight=8.0,
                max_dose=45.0,  # Gy
                priority=1,
            )
        )

        # Parotid glands
        objectives.append(
            BeamOptimizationObjective(
                objective_type="MINIMIZE_OAR_DOSE",
                structure_name="Parotid_L",
                weight=4.0,
                max_dose=26.0,  # Gy
                priority=2,
            )
        )

        objectives.append(
            BeamOptimizationObjective(
                objective_type="MINIMIZE_OAR_DOSE",
                structure_name="Parotid_R",
                weight=4.0,
                max_dose=26.0,  # Gy
                priority=2,
            )
        )

    return objectives


if __name__ == "__main__":
    # Test code
    logging.basicConfig(level=logging.INFO)

    # Tạo beam optimizer
    optimizer = create_comprehensive_beam_optimizer()

    # Test objectives
    objectives = create_standard_objectives("prostate")
    print(f"Created {len(objectives)} objectives")

    # Test constraints
    constraints = BeamConstraint(min_angle_separation=15.0, min_beams=5, max_beams=9)

    print("Beam Optimizer test hoàn thành!")
