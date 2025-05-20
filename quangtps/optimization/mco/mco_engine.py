#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Multi-criteria optimization (MCO) engine.

This module provides the backend functionality for multi-criteria optimization,
including generation of Pareto-optimal plans, interpolation between plans,
and navigation of the Pareto surface.
"""

import os
import json
import time
import logging
import copy
import uuid
from typing import Dict, List, Optional, Tuple, Union, Any, Set, Callable

import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
import pandas as pd
from datetime import datetime

from quangtps.core.types import Plan, DoseGrid, Treatment, Structure
from quangtps.planning.optimization import PlanOptimizer
from quangtps.optimization.objectives import (
    Objective,
    ObjectiveResult,
    ObjectiveFunction,
    ObjectiveType,
)
from quangtps.optimization.constraints import Constraint
from quangtps.optimization.optimizer_factory import create_optimizer
from quangtps.treatment.techniques.imrt import IMRTTreatment
from quangtps.treatment.techniques.vmat import VMATTreatment
from quangtps.core.logging import get_logger
from quangtps.core.patient import Patient
from quangtps.evaluation.dvh.dvh_calculation import DVHCalculator, calculate_dvh_metrics
from quangtps.planning.plan import Plan
from quangtps.optimization.optimizers.optimizer_factory import OptimizerFactory
from quangtps.dose.dose_calculation import DoseEngine
from quangtps.optimization.optimizer import Optimizer, OptimizationParameters
from quangtps.optimization.objectives.objective_factory import ObjectiveFactory
from quangtps.core.utils import get_timestamp, create_directory_if_not_exists
from quangtps.core.exceptions import OptimizationError
from quangtps.optimization.mco.pareto_surface import ParetoSurface, ParetoSolution
from quangtps.optimization.constraints import ConstraintCollection, ConstraintBase
from quangtps.optimization.mco.pareto_navigator import ParetoNavigator

logger = get_logger(__name__)


@dataclass
class MCOTradeoffObjective:
    """
    Defines an objective for MCO trade-off exploration.

    This represents a single objective function that can be used in multi-criteria
    optimization to generate and navigate the Pareto front.
    """

    objective_id: str
    structure_id: str
    structure_name: str
    objective_type: (
        str  # e.g., "MinDose", "MaxDose", "MeanDose", "DoseAtVolume", "VolumeAtDose"
    )
    parameters: Dict[str, Any] = field(default_factory=dict)
    priority: int = 1  # Priority (1-3, with 1 being highest)
    weight: float = 1.0  # Current weight for weighted-sum optimization
    value: float = 0.0  # Current value of this objective
    min_value: float = 0.0  # Minimum achievable value
    max_value: float = 0.0  # Maximum achievable value
    is_active: bool = True  # Whether this objective is active in the current trade-off

    def to_optimizer_objective(self) -> Objective:
        """Convert to an optimizer objective."""
        factory = ObjectiveFactory()
        return factory.create_objective(
            objective_type=self.objective_type,
            structure_id=self.structure_id,
            parameters=self.parameters,
            weight=self.weight,
        )

    def calculate_value(self, plan: Plan) -> float:
        """
        Calculate the current value of this objective for a given plan.

        Args:
            plan: The plan to evaluate

        Returns:
            The current value of this objective
        """
        structure = None
        for s in plan.structure_set.structures:
            if s.id == self.structure_id:
                structure = s
                break

        if not structure:
            logger.warning(
                f"Structure {self.structure_id} not found for objective {self.objective_id}"
            )
            return 0.0

        metrics = calculate_dvh_metrics(structure, plan.dose)

        if self.objective_type == "MinDose":
            return metrics.get("min_dose", 0.0)
        elif self.objective_type == "MaxDose":
            return metrics.get("max_dose", 0.0)
        elif self.objective_type == "MeanDose":
            return metrics.get("mean_dose", 0.0)
        elif self.objective_type == "DoseAtVolume":
            volume = self.parameters.get("volume", 50)
            return metrics.get(f"D{volume}", 0.0)
        elif self.objective_type == "VolumeAtDose":
            dose = self.parameters.get("dose", 50)
            return metrics.get(f"V{dose}", 0.0)
        else:
            logger.warning(
                f"Unknown objective type {self.objective_type} for {self.objective_id}"
            )
            return 0.0

    def update_value(self, plan: Plan):
        """Update the value of this objective based on the given plan."""
        self.value = self.calculate_value(plan)


@dataclass
class MCOSolution:
    """
    Represents a single solution in the MCO Pareto front.

    This includes the plan, objective values, and weights used to generate this solution.
    """

    plan: Plan
    objective_values: Dict[str, float]  # objective_id -> value
    weights: Dict[str, float]  # objective_id -> weight
    solution_id: str = ""  # Unique identifier for this solution

    def __post_init__(self):
        if not self.solution_id:
            self.solution_id = f"sol_{id(self)}"


class MCOEngine:
    """
    Multi-Criteria Optimization (MCO) engine.

    This class provides the capability to generate and navigate Pareto-optimal
    treatment plans using multi-criteria optimization, similar to Eclipse's MCO.
    """

    def __init__(
        self,
        patient: Optional[Patient] = None,
        plan: Optional[Plan] = None,
        name: str = "MCO Session",
    ):
        """
        Initialize the MCO engine.

        Args:
            patient: The patient, default is None
            plan: The base plan to start optimization from, default is None
            name: The name of the MCO session, default is "MCO Session"
        """
        self.patient = patient
        self.base_plan = plan
        self.name = name
        self.objectives = {}  # Dictionary of objective functions and weights
        self.constraints = ConstraintCollection()  # Constraints
        self.pareto_surface = ParetoSurface(name=name)  # Pareto surface
        self.pareto_navigator = None  # Pareto navigator
        self.optimization_parameters = {}  # Optimization parameters
        self.current_solution = None  # Current Pareto solution
        self.optimization_history = []  # Optimization history
        self.session_id = str(uuid.uuid4())
        self.timestamp = get_timestamp()
        self.metadata = {}

    def add_objective(
        self,
        name: str,
        objective_function: Callable,
        weight: float = 1.0,
        is_target: bool = False,
    ):
        """
        Add an objective to the MCO problem.

        Args:
            name: The name of the objective
            objective_function: The objective function
            weight: The weight of the objective, default is 1.0
            is_target: True if this is a target objective (PTV), False if it's a protection objective (OAR), default is False
        """
        self.objectives[name] = {
            "function": objective_function,
            "weight": weight,
            "is_target": is_target,
        }

    def add_constraint(self, constraint: ConstraintBase):
        """
        Add a constraint to the MCO problem.

        Args:
            constraint: The constraint to add
        """
        self.constraints.add_constraint(constraint)

    def set_optimization_parameters(self, params: Dict[str, Any]):
        """
        Set the optimization parameters.

        Args:
            params: Dictionary of optimization parameters
        """
        self.optimization_parameters = params

    def generate_pareto_surface(
        self,
        parameter_ranges: Dict[str, Tuple[float, float]],
        num_samples: int = 100,
        max_iterations: int = 10,
    ) -> ParetoSurface:
        """
        Generate the Pareto surface by sampling the parameter space.

        Parameters
        ----------
        parameter_ranges : Dict[str, Tuple[float, float]]
            The range of each parameter (min, max)
        num_samples : int, optional
            Initial number of samples, default is 100
        max_iterations : int, optional
            Maximum number of iterations, default is 10

        Returns
        -------
        ParetoSurface
            The generated Pareto surface
        """
        try:
            # Combined objective function to evaluate parameters
            def optimization_function(params):
                # Calculate the value of each objective
                objective_values = {}
                for name, obj_info in self.objectives.items():
                    objective_values[name] = obj_info["function"](params)
                return objective_values

            # Generate Pareto surface
            self.pareto_surface.generate_pareto_set(
                optimization_function=optimization_function,
                parameter_ranges=parameter_ranges,
                objective_names=list(self.objectives.keys()),
                num_samples=num_samples,
                max_iterations=max_iterations,
            )

            # Create Pareto navigator
            self.pareto_navigator = ParetoNavigator(self.pareto_surface)
            self.pareto_navigator.set_plan_generator(self.generate_plan_from_solution)

            return self.pareto_surface

        except Exception as e:
            logger.error(f"Error generating Pareto surface: {str(e)}")
            raise OptimizationError(f"Cannot generate Pareto surface: {str(e)}")

    def generate_plan_from_solution(self, solution: ParetoSolution) -> Optional[Plan]:
        """
        Generate a treatment plan from a Pareto solution.

        Parameters
        ----------
        solution : ParetoSolution
            The Pareto solution

        Returns
        -------
        Optional[Plan]
            The generated plan or None if unable to generate
        """
        if not self.base_plan:
            logger.warning("No base plan to generate new plan")
            return None

        try:
            # Clone the base plan
            new_plan = self.base_plan.clone()

            # Set optimization parameters from the solution
            for param_name, param_value in solution.parameters.items():
                # Apply only parameters defined in the base plan
                if hasattr(new_plan, param_name):
                    setattr(new_plan, param_name, param_value)
                elif param_name in new_plan.parameters:
                    new_plan.parameters[param_name] = param_value

            # Update plan metadata
            new_plan.metadata["mco_solution_id"] = str(uuid.uuid4())
            new_plan.metadata["mco_session_id"] = self.session_id
            new_plan.metadata["mco_timestamp"] = get_timestamp()
            new_plan.metadata["mco_objective_values"] = solution.objective_values

            # Set name for the plan
            objective_str = "_".join(
                [
                    f"{name}={value:.2f}"
                    for name, value in list(solution.objective_values.items())[
                        :2
                    ]  # Take only the first two objectives
                ]
            )
            new_plan.name = f"{self.base_plan.name}_MCO_{objective_str}"

            return new_plan

        except Exception as e:
            logger.error(f"Error generating plan from Pareto solution: {str(e)}")
            return None

    def navigate_pareto_surface(
        self, objective_weights: Optional[Dict[str, float]] = None
    ) -> Optional[ParetoSolution]:
        """
        Navigate through the Pareto surface to find the optimal solution.

        Parameters
        ----------
        objective_weights : Optional[Dict[str, float]], optional
            Objective weights, default is None (use current weights)

        Returns
        -------
        Optional[ParetoSolution]
            The found Pareto solution
        """
        if not self.pareto_navigator:
            logger.warning("Pareto navigator not created")
            return None

        # Set weights if provided
        if objective_weights:
            self.pareto_navigator.set_objective_weights(objective_weights)
        else:
            # Use weights from objectives
            weights = {
                name: obj_info["weight"] for name, obj_info in self.objectives.items()
            }
            self.pareto_navigator.set_objective_weights(weights)

        # Find the optimal solution based on weights
        solution = self.pareto_navigator.select_solution_by_weights()
        if solution:
            self.current_solution = solution

        return solution

    def select_solution_by_objectives(
        self, objectives: Dict[str, Dict[str, Any]], weights: Dict[str, float] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Chọn giải pháp tốt nhất dựa trên các mục tiêu và trọng số.

        Parameters
        ----------
        objectives : Dict[str, Dict[str, Any]]
            Từ điển các mục tiêu, với cấu trúc {structure_name: {objective_type: params}}
        weights : Dict[str, float], optional
            Trọng số cho mỗi mục tiêu, theo cấu trúc {objective_id: weight}

        Returns
        -------
        Optional[Dict[str, Any]]
            Giải pháp tốt nhất, hoặc None nếu không tìm thấy
        """
        if not self.pareto_surface or not self.pareto_surface.solutions:
            logger.warning("No Pareto solutions available to select from")
            return None

        try:
            # Tạo vector trọng số nếu không được cung cấp
            if weights is None:
                weights = {obj_id: 1.0 for obj_id in objectives.keys()}

            # Tính điểm cho mỗi giải pháp
            solution_scores = {}

            for sol_id, solution in self.pareto_surface.solutions.items():
                score = 0.0

                # Tính điểm dựa trên mức độ đáp ứng mục tiêu
                for structure, structure_objectives in objectives.items():
                    if structure not in solution.objective_values:
                        continue

                    for obj_type, params in structure_objectives.items():
                        # Lấy trọng số cho mục tiêu này
                        weight = weights.get(f"{structure}_{obj_type}", 1.0)

                        # Tính điểm cho mục tiêu cụ thể
                        obj_score = self._calculate_objective_score(
                            solution.objective_values[structure], obj_type, params
                        )

                        # Cộng vào điểm tổng
                        score += obj_score * weight

                solution_scores[sol_id] = score

            # Chọn giải pháp có điểm cao nhất
            if not solution_scores:
                logger.warning("Không thể tính điểm cho bất kỳ giải pháp nào")
                return None

            best_solution_id = max(solution_scores, key=solution_scores.get)
            logger.info(
                f"Đã chọn giải pháp {best_solution_id} với điểm {solution_scores[best_solution_id]}"
            )

            return self.pareto_surface.solutions.get(best_solution_id)

        except Exception as e:
            logger.error(f"Lỗi khi chọn giải pháp theo mục tiêu: {e}")
            # Trả về giải pháp đầu tiên trong trường hợp lỗi
            if self.pareto_surface.solutions:
                first_solution_id = next(iter(self.pareto_surface.solutions))
                logger.warning(f"Trả về giải pháp {first_solution_id} do lỗi xử lý")
                return self.pareto_surface.solutions.get(first_solution_id)
            return None

    def get_current_plan(self) -> Optional[Plan]:
        """
        Get the current treatment plan based on the current Pareto solution.

        Returns
        -------
        Optional[Plan]
            The current treatment plan
        """
        if not self.current_solution:
            logger.warning("No current Pareto solution")
            return None

        return self.generate_plan_from_solution(self.current_solution)

    def visualize_pareto_front(
        self,
        obj_x: str,
        obj_y: str,
        obj_z: Optional[str] = None,
        save_path: Optional[str] = None,
    ):
        """
        Visualize the Pareto front with 2 or 3 objectives.

        Parameters
        ----------
        obj_x : str
            Name of the objective for the X axis
        obj_y : str
            Name of the objective for the Y axis
        obj_z : Optional[str], optional
            Name of the objective for the Z axis (3D plot), default is None
        save_path : Optional[str], optional
            Path to save the image, default is None
        """
        if not self.pareto_surface:
            logger.warning("No Pareto surface to visualize")
            return

        self.pareto_surface.visualize(
            obj_x, obj_y, obj_z, self.current_solution, save_path
        )

    def visualize_navigation(
        self,
        obj_x: str,
        obj_y: str,
        obj_z: Optional[str] = None,
        show_history: bool = True,
        save_path: Optional[str] = None,
    ):
        """
        Visualize the Pareto navigation process.

        Parameters
        ----------
        obj_x : str
            Name of the objective for the X axis
        obj_y : str
            Name of the objective for the Y axis
        obj_z : Optional[str], optional
            Name of the objective for the Z axis (3D plot), default is None
        show_history : bool, optional
            Show navigation history, default is True
        save_path : Optional[str], optional
            Path to save the image, default is None
        """
        if not self.pareto_navigator:
            logger.warning("Pareto navigator not created")
            return

        self.pareto_navigator.visualize_navigation(
            obj_x, obj_y, obj_z, show_history, save_path
        )

    def analyze_tradeoffs(self) -> Dict[str, Dict[str, float]]:
        """
        Analyze trade-offs between objectives.

        Returns
        -------
        Dict[str, Dict[str, float]]
            Trade-off matrix between objectives
        """
        if not self.pareto_navigator:
            logger.warning("Pareto navigator not created")
            return {}

        obj_names = list(self.objectives.keys())
        tradeoffs = {}

        # Calculate trade-offs between each pair of objectives
        for i, obj1 in enumerate(obj_names):
            tradeoffs[obj1] = {}
            for j, obj2 in enumerate(obj_names):
                if i != j:
                    coeff, _ = self.pareto_navigator.get_tradeoff_analysis(obj1, obj2)
                    tradeoffs[obj1][obj2] = coeff

        return tradeoffs

    def save_session(self, filepath: Optional[str] = None) -> str:
        """
        Save the MCO session to file.

        Parameters
        ----------
        filepath : Optional[str], optional
            File path to save, default is None (auto-generate)

        Returns
        -------
        str
            Saved file path
        """
        if filepath is None:
            # Create default directory
            mco_dir = os.path.join(os.getcwd(), "data", "mco_sessions")
            os.makedirs(mco_dir, exist_ok=True)

            # Create file name based on time
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"mco_session_{self.name.replace(' ', '_')}_{timestamp}.json"
            filepath = os.path.join(mco_dir, filename)

        try:
            # Save session data
            session_data = {
                "name": self.name,
                "session_id": self.session_id,
                "timestamp": self.timestamp,
                "metadata": self.metadata,
                "optimization_parameters": self.optimization_parameters,
            }

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(session_data, f, indent=2)

            # Save Pareto surface
            if self.pareto_surface:
                pareto_path = os.path.splitext(filepath)[0] + "_pareto.pkl"
                self.pareto_surface.save(pareto_path)

            # Save navigation session
            if self.pareto_navigator:
                nav_path = os.path.splitext(filepath)[0] + "_navigation.csv"
                self.pareto_navigator.save_navigation_session(
                    nav_path, include_pareto=False
                )

            logger.info(f"Saved MCO session to {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"Error saving MCO session: {str(e)}")
            raise OptimizationError(f"Cannot save MCO session: {str(e)}")

    @classmethod
    def load_session(cls, filepath: str) -> "MCOEngine":
        """
        Load the MCO session from file.

        Parameters
        ----------
        filepath : str
            File path to load

        Returns
        -------
        MCOEngine
            Loaded MCOEngine object
        """
        try:
            # Load session data
            with open(filepath, "r", encoding="utf-8") as f:
                session_data = json.load(f)

            # Create new MCOEngine object
            engine = cls(name=session_data.get("name", "Loaded MCO Session"))

            # Update properties
            engine.session_id = session_data.get("session_id", str(uuid.uuid4()))
            engine.timestamp = session_data.get("timestamp", get_timestamp())
            engine.metadata = session_data.get("metadata", {})
            engine.optimization_parameters = session_data.get(
                "optimization_parameters", {}
            )

            # Load Pareto surface
            pareto_path = os.path.splitext(filepath)[0] + "_pareto.pkl"
            if os.path.exists(pareto_path):
                engine.pareto_surface = ParetoSurface.load(pareto_path)

                # Create Pareto navigator
                engine.pareto_navigator = ParetoNavigator(engine.pareto_surface)
                engine.pareto_navigator.set_plan_generator(
                    engine.generate_plan_from_solution
                )

                # Load navigation session
                nav_path = os.path.splitext(filepath)[0] + "_navigation.csv"
                if os.path.exists(nav_path):
                    pass  # TODO: Load navigation history

            logger.info(f"Loaded MCO session from {filepath}")
            return engine

        except Exception as e:
            logger.error(f"Error loading MCO session: {str(e)}")
            raise OptimizationError(f"Cannot load MCO session: {str(e)}")

    def _calculate_objective_score(
        self, metrics: Dict[str, Any], obj_type: str, params: Dict[str, Any]
    ) -> float:
        """
        Tính điểm cho một mục tiêu cụ thể dựa trên các tham số và giá trị thực tế.

        Parameters
        ----------
        metrics : Dict[str, Any]
            Các metrics của cấu trúc
        obj_type : str
            Loại mục tiêu (ví dụ: 'max_dose', 'mean_dose', 'dvh')
        params : Dict[str, Any]
            Tham số cho mục tiêu

        Returns
        -------
        float
            Điểm số của mục tiêu (càng cao càng tốt)
        """
        try:
            # Xử lý các loại mục tiêu khác nhau
            if obj_type == "max_dose":
                target = params.get("dose", 0.0)
                actual = metrics.get("max_dose", 0.0)

                # Nếu là OAR, thì thấp hơn mục tiêu là tốt
                if params.get("is_oar", True):
                    return 100.0 if actual <= target else 100.0 * (target / actual)
                # Nếu là PTV, thì gần với mục tiêu là tốt
                else:
                    return 100.0 / (1.0 + abs(actual - target) / target)

            elif obj_type == "mean_dose":
                target = params.get("dose", 0.0)
                actual = metrics.get("mean_dose", 0.0)

                # Nếu là OAR, thì thấp hơn mục tiêu là tốt
                if params.get("is_oar", True):
                    return 100.0 if actual <= target else 100.0 * (target / actual)
                # Nếu là PTV, thì gần với mục tiêu là tốt
                else:
                    return 100.0 / (1.0 + abs(actual - target) / target)

            elif obj_type == "dvh":
                dose = params.get("dose", 0.0)
                volume = params.get("volume", 0.0)
                relation = params.get("relation", "less")  # "less" hoặc "more"

                # Lấy giá trị DVH thực tế
                dvh_data = metrics.get("dvh", {})
                actual_volume = None

                # Tìm điểm DVH gần nhất
                if dvh_data and "doses" in dvh_data and "volumes" in dvh_data:
                    doses = dvh_data["doses"]
                    volumes = dvh_data["volumes"]

                    # Tìm điểm gần nhất với dose
                    closest_idx = min(
                        range(len(doses)), key=lambda i: abs(doses[i] - dose)
                    )
                    actual_volume = volumes[closest_idx]

                if actual_volume is None:
                    return 0.0

                # Tính điểm
                if relation == "less":  # V20Gy < 30% (OAR)
                    return (
                        100.0
                        if actual_volume <= volume
                        else 100.0 * (volume / actual_volume)
                    )
                else:  # V95% > 98% (PTV)
                    return (
                        100.0
                        if actual_volume >= volume
                        else 100.0 * (actual_volume / volume)
                    )

            # Các loại mục tiêu khác
            else:
                logger.warning(f"Loại mục tiêu không được hỗ trợ: {obj_type}")
                return 0.0

        except Exception as e:
            logger.error(f"Lỗi khi tính điểm mục tiêu {obj_type}: {e}")
            return 0.0


def create_mco_engine(
    patient: Optional[Patient] = None,
    plan: Optional[Plan] = None,
    name: str = "MCO Session",
) -> MCOEngine:
    """
    Create and configure a new MCOEngine object.

    Parameters
    ----------
    patient : Optional[Patient], optional
        The patient, default is None
    plan : Optional[Plan], optional
        The base plan to start from, default is None
    name : str, optional
        The name of the MCO session, default is "MCO Session"

    Returns
    -------
    MCOEngine
        The configured MCOEngine object
    """
    return MCOEngine(patient=patient, plan=plan, name=name)


if __name__ == "__main__":
    # Test code
    import sys
    import matplotlib.pyplot as plt

    # Create a dummy plan and objectives
    class DummyPlan:
        def __init__(self):
            self.name = "Test Plan"
            self.id = "test_plan_1"

        def clone(self):
            return DummyPlan()

    class DummyObjective:
        def __init__(self, name):
            self.name = name

        def evaluate(self, fluence):
            # Return a dummy result
            return 0.5

    # Create a test engine
    plan = DummyPlan()
    objectives = {
        "ptv_coverage": DummyObjective("ptv_coverage"),
        "oar_sparing": DummyObjective("oar_sparing"),
    }

    engine = MCOEngine()

    # Test saving and loading
    engine.solutions = {
        "Solution_1": MCOSolution(
            id="Solution_1",
            objective_values={"ptv_coverage": 1.0, "oar_sparing": 0.0},
            plan=plan,
            weight_vector={"ptv_coverage": 1.0, "oar_sparing": 0.0},
        ),
        "Solution_2": MCOSolution(
            id="Solution_2",
            objective_values={"ptv_coverage": 0.0, "oar_sparing": 1.0},
            plan=plan,
            weight_vector={"ptv_coverage": 0.0, "oar_sparing": 1.0},
        ),
    }

    engine.save_current_solution("Test Solution 1")
    print("Saved current solution")

    engine.current_solution_id = None
    engine.load_solutions("test_solutions.json")
    print(f"Loaded {len(engine.solutions)} solutions")
