"""
Robust optimization implementation for handling setup and range uncertainties.

This module provides classes and functions for robust treatment plan optimization,
which takes into account positional setup uncertainties and range uncertainties
for particle therapy to ensure treatment plans are resilient to various uncertainties.
"""

import os
import time
import logging
import numpy as np
import scipy.optimize as sopt
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum, auto

from quangtps.core.types import Plan, Structure, Dose, DoseGrid
from quangtps.core.exceptions import OptimizationError
from quangtps.core.logging import get_logger
from quangtps.dose.dose_calculation import DoseCalculator
from quangtps.optimization.optimizer import Optimizer
from quangtps.optimization.objectives import ObjectiveFunction, PlanningObjectives
from quangtps.optimization.constraints import ConstraintFunction
from quangtps.optimization.methods.objective_based import ObjectiveBasedOptimizer

# Configure logging
logger = get_logger(__name__)


@dataclass
class UncertaintyScenario:
    """Scenario for robust optimization including uncertainty parameters."""

    name: str
    parameters: Dict[str, Any]
    weight: float = 1.0

    def __str__(self) -> str:
        """Return string representation of scenario."""
        param_str = ", ".join([f"{k}={v}" for k, v in self.parameters.items()])
        return f"Scenario '{self.name}': {param_str} (weight={self.weight})"


class RobustOptimizer:
    """
    Optimizer for robust treatment planning.

    This optimizer takes into account uncertainties in patient setup and
    range (for particle therapy) to create plans that are resilient to these
    variations.
    """

    def __init__(
        self,
        plan: Plan,
        objectives: PlanningObjectives,
        dose_calculator: DoseCalculator,
    ):
        """
        Initialize robust optimizer.

        Parameters
        ----------
        plan : Plan
            Treatment plan to optimize
        objectives : PlanningObjectives
            Planning objectives
        dose_calculator : DoseCalculator
            Dose calculation engine
        """
        self.plan = plan
        self.objectives = objectives
        self.dose_calculator = dose_calculator

        # Nominal (reference) scenario
        self.nominal_scenario = UncertaintyScenario(
            name="nominal", parameters={}, weight=1.0
        )

        # List of uncertainty scenarios
        self.scenarios = []

        # Optimization parameters
        self.parameters = {
            "max_iterations": 100,
            "convergence_threshold": 1e-4,
            "setup_uncertainty": 3.0,  # mm
            "range_uncertainty": 3.5,  # percentage
            "worst_case": True,  # Use worst-case vs expected value
            "scenario_sampling": "corners",  # 'corners', 'random', 'hybrid'
            "iteration_callback": None,
        }

        # Optimization state
        self.current_iteration = 0
        self.objective_values = []
        self.nominal_optimizer = None
        self.scenario_optimizers = {}

    def set_parameter(self, name: str, value: Any) -> None:
        """
        Set optimization parameter.

        Parameters
        ----------
        name : str
            Parameter name
        value : Any
            Parameter value
        """
        if name in self.parameters:
            self.parameters[name] = value
        else:
            logger.warning(f"Unknown parameter: {name}")

    def add_scenario(
        self,
        structures: Dict[str, Structure],
        weight: float = 1.0,
        name: Optional[str] = None,
    ) -> None:
        """
        Add a custom scenario with different structures.

        Parameters
        ----------
        structures : Dict[str, Structure]
            Dictionary of structures for this scenario
        weight : float, optional
            Scenario weight, by default 1.0
        name : Optional[str], optional
            Scenario name, by default None
        """
        if name is None:
            name = f"scenario_{len(self.scenarios) + 1}"

        # Create scenario
        scenario = UncertaintyScenario(
            name=name, parameters={"structures": structures}, weight=weight
        )

        self.scenarios.append(scenario)

    def generate_standard_scenarios(
        self,
        setup_uncertainty: Optional[float] = None,
        range_uncertainty: Optional[float] = None,
    ) -> None:
        """
        Generate standard uncertainty scenarios based on setup and range uncertainties.

        Parameters
        ----------
        setup_uncertainty : Optional[float], optional
            Setup uncertainty in mm, by default None
        range_uncertainty : Optional[float], optional
            Range uncertainty in percentage, by default None
        """
        # Use provided values or defaults from parameters
        setup_unc = (
            setup_uncertainty
            if setup_uncertainty is not None
            else self.parameters["setup_uncertainty"]
        )
        range_unc = (
            range_uncertainty
            if range_uncertainty is not None
            else self.parameters["range_uncertainty"]
        )

        # Clear existing scenarios
        self.scenarios = []

        # Determine sampling strategy
        strategy = self.parameters["scenario_sampling"]

        if strategy == "corners":
            # Generate scenarios at the corners of the uncertainty space
            for x_shift in [-setup_unc, 0, setup_unc]:
                for y_shift in [-setup_unc, 0, setup_unc]:
                    for z_shift in [-setup_unc, 0, setup_unc]:
                        # Skip nominal scenario (0,0,0)
                        if x_shift == 0 and y_shift == 0 and z_shift == 0:
                            continue

                        # Create setup parameters
                        scenario_params = {
                            "setup_x": x_shift,
                            "setup_y": y_shift,
                            "setup_z": z_shift,
                        }

                        # Add range uncertainty if applicable (for particle therapy)
                        use_range = (
                            range_unc > 0
                            and hasattr(self.plan, "modality")
                            and getattr(self.plan, "modality", "").lower()
                            in ["proton", "carbon"]
                        )

                        if use_range:
                            for r_shift in [-range_unc, range_unc]:
                                r_params = scenario_params.copy()
                                r_params["range"] = r_shift
                                name = self._make_scenario_name(r_params)
                                self.scenarios.append(
                                    UncertaintyScenario(
                                        name=name, parameters=r_params, weight=1.0
                                    )
                                )
                        else:
                            name = self._make_scenario_name(scenario_params)
                            self.scenarios.append(
                                UncertaintyScenario(
                                    name=name, parameters=scenario_params, weight=1.0
                                )
                            )

        elif strategy == "random":
            # Generate random scenarios within the uncertainty space
            num_scenarios = 10  # Number of random scenarios to generate

            for i in range(num_scenarios):
                x_shift = np.random.uniform(-setup_unc, setup_unc)
                y_shift = np.random.uniform(-setup_unc, setup_unc)
                z_shift = np.random.uniform(-setup_unc, setup_unc)

                scenario_params = {
                    "setup_x": x_shift,
                    "setup_y": y_shift,
                    "setup_z": z_shift,
                }

                # Add range uncertainty if applicable
                use_range = (
                    range_unc > 0
                    and hasattr(self.plan, "modality")
                    and getattr(self.plan, "modality", "").lower()
                    in ["proton", "carbon"]
                )

                if use_range:
                    r_shift = np.random.uniform(-range_unc, range_unc)
                    scenario_params["range"] = r_shift

                name = f"random_{i + 1}"
                self.scenarios.append(
                    UncertaintyScenario(
                        name=name, parameters=scenario_params, weight=1.0
                    )
                )

        elif strategy == "hybrid":
            # Generate a mix of corner and random scenarios
            # First, generate key corners
            for axis in ["x", "y", "z"]:
                for shift in [-setup_unc, setup_unc]:
                    scenario_params = {
                        "setup_x": shift if axis == "x" else 0,
                        "setup_y": shift if axis == "y" else 0,
                        "setup_z": shift if axis == "z" else 0,
                    }
                    name = self._make_scenario_name(scenario_params)
                    self.scenarios.append(
                        UncertaintyScenario(
                            name=name, parameters=scenario_params, weight=1.0
                        )
                    )

            # Then add some random scenarios
            num_random = 5
            for i in range(num_random):
                x_shift = np.random.uniform(-setup_unc, setup_unc)
                y_shift = np.random.uniform(-setup_unc, setup_unc)
                z_shift = np.random.uniform(-setup_unc, setup_unc)

                scenario_params = {
                    "setup_x": x_shift,
                    "setup_y": y_shift,
                    "setup_z": z_shift,
                }

                name = f"random_{i + 1}"
                self.scenarios.append(
                    UncertaintyScenario(
                        name=name, parameters=scenario_params, weight=1.0
                    )
                )

        else:
            logger.warning(f"Unknown scenario sampling strategy: {strategy}")

        logger.info(
            f"Generated {len(self.scenarios)} scenarios for robust optimization"
        )

    def _make_scenario_name(self, params: Dict[str, Any]) -> str:
        """
        Create a descriptive name for a scenario based on its parameters.

        Parameters
        ----------
        params : Dict[str, Any]
            Dictionary of scenario parameters

        Returns
        -------
        str
            Descriptive scenario name
        """
        parts = []

        if "setup_x" in params:
            x = params["setup_x"]
            parts.append(f"X{x:+.1f}")

        if "setup_y" in params:
            y = params["setup_y"]
            parts.append(f"Y{y:+.1f}")

        if "setup_z" in params:
            z = params["setup_z"]
            parts.append(f"Z{z:+.1f}")

        if "range" in params:
            r = params["range"]
            parts.append(f"R{r:+.1f}%")

        return "_".join(parts)

    def calculate_scenario_doses(self) -> Dict[str, DoseGrid]:
        """
        Calculate dose for all scenarios.

        Returns
        -------
        Dict[str, DoseGrid]
            Dictionary mapping scenario names to dose grids
        """
        doses = {}

        # Calculate nominal scenario dose
        logger.info("Calculating dose for nominal scenario")
        nominal_dose = self.dose_calculator.calculate_dose(self.plan)
        doses[self.nominal_scenario.name] = nominal_dose

        # Calculate doses for all other scenarios
        for scenario in self.scenarios:
            logger.info(f"Calculating dose for scenario: {scenario.name}")

            # Apply scenario-specific modifications
            modified_plan = self._apply_scenario_to_plan(self.plan, scenario)

            # Calculate dose
            scenario_dose = self.dose_calculator.calculate_dose(modified_plan)
            doses[scenario.name] = scenario_dose

        return doses

    def _apply_scenario_to_plan(
        self, plan: Plan, scenario: UncertaintyScenario
    ) -> Plan:
        """
        Apply scenario uncertainty to a plan.

        Parameters
        ----------
        plan : Plan
            Original plan
        scenario : UncertaintyScenario
            Uncertainty scenario to apply

        Returns
        -------
        Plan
            Modified plan with scenario applied
        """
        # Create a copy of the plan to modify
        modified_plan = plan.copy()

        # Apply setup uncertainties
        if (
            "setup_x" in scenario.parameters
            or "setup_y" in scenario.parameters
            or "setup_z" in scenario.parameters
        ):
            x_shift = scenario.parameters.get("setup_x", 0)
            y_shift = scenario.parameters.get("setup_y", 0)
            z_shift = scenario.parameters.get("setup_z", 0)

            # Apply isocenter shift (this is a simplified example)
            if hasattr(modified_plan, "isocenter"):
                current_iso = modified_plan.isocenter
                new_iso = (
                    current_iso[0] + x_shift,
                    current_iso[1] + y_shift,
                    current_iso[2] + z_shift,
                )
                modified_plan.isocenter = new_iso

        # Apply range uncertainty (for particle therapy)
        if "range" in scenario.parameters:
            range_factor = 1.0 + scenario.parameters["range"] / 100.0

            # Apply range scaling to all beams in the plan
            if hasattr(modified_plan, "beams"):
                for beam in modified_plan.beams:
                    if hasattr(beam, "energy"):
                        beam.energy *= range_factor

        # Apply custom structures if provided
        if "structures" in scenario.parameters:
            modified_plan.structures = scenario.parameters["structures"]

        return modified_plan

    def optimize(self) -> Dict:
        """
        Run robust optimization.

        Returns
        -------
        Dict
            Optimization results
        """
        logger.info("Starting robust optimization")

        # Reset state
        self.current_iteration = 0
        self.objective_values = []

        # Create initial plan parameters vector
        initial_params = self._plan_to_params(self.plan)

        # Initialize optimizers for each scenario
        self._initialize_optimizers()

        # Define objective function for optimizer
        def objective_function(params):
            # Update plan with new parameters
            self._params_to_plan(params, self.plan)

            # Calculate objective value for all scenarios
            obj_value = self._calculate_composite_objective(
                self.nominal_optimizer, self.scenario_optimizers
            )

            # Track progress
            self.objective_values.append(obj_value)

            # Call iteration callback if provided
            if self.parameters["iteration_callback"] is not None:
                self.parameters["iteration_callback"](self.current_iteration, obj_value)

            self.current_iteration += 1

            return obj_value

        # Run optimization
        try:
            max_iterations = self.parameters["max_iterations"]
            convergence_threshold = self.parameters["convergence_threshold"]

            result = sopt.minimize(
                objective_function,
                initial_params,
                method="L-BFGS-B",
                options={
                    "maxiter": max_iterations,
                    "ftol": convergence_threshold,
                    "disp": True,
                },
            )

            # Apply final parameters to plan
            self._params_to_plan(result.x, self.plan)

            # Create a final plan
            final_plan = self.plan.copy()

            # Calculate final objective value
            best_objective_value = result.fun

            # Log results
            logger.info(
                f"Robust optimization completed in {self.current_iteration} iterations"
            )
            logger.info(f"Final objective value: {best_objective_value}")

        except Exception as e:
            logger.error(f"Error during optimization: {e}")
            raise OptimizationError(f"Robust optimization failed: {e}")

        # Return results
        return {
            "plan": final_plan,
            "objective_values": self.objective_values,
            "final_objective_value": best_objective_value,
            "iterations": self.current_iteration + 1,
            "success": True,
        }

    def _calculate_composite_objective(
        self,
        nominal_optimizer: ObjectiveBasedOptimizer,
        scenario_optimizers: Dict[str, ObjectiveBasedOptimizer],
    ) -> float:
        """
        Calculate composite objective value across all scenarios.

        Parameters
        ----------
        nominal_optimizer : ObjectiveBasedOptimizer
            Optimizer for nominal scenario
        scenario_optimizers : Dict[str, ObjectiveBasedOptimizer]
            Optimizers for all other scenarios

        Returns
        -------
        float
            Composite objective value
        """
        # Get objective value for nominal scenario
        nominal_value = nominal_optimizer.calculate_objective_value()

        # Get objective values for all other scenarios
        scenario_values = {}
        for name, optimizer in scenario_optimizers.items():
            scenario_values[name] = optimizer.calculate_objective_value()

        # Calculate composite value based on strategy
        if self.parameters["worst_case"]:
            # Worst-case optimization
            return max(
                [nominal_value * self.nominal_scenario.weight]
                + [
                    v * self.scenarios[i].weight
                    for i, (_, v) in enumerate(scenario_values.items())
                ]
            )
        else:
            # Weighted average
            total_weight = self.nominal_scenario.weight + sum(
                s.weight for s in self.scenarios
            )
            weighted_sum = nominal_value * self.nominal_scenario.weight

            for i, (_, value) in enumerate(scenario_values.items()):
                weighted_sum += value * self.scenarios[i].weight

            return weighted_sum / total_weight

    def _initialize_optimizers(self) -> None:
        """Initialize optimizers for nominal and all scenarios."""
        # Create optimizer for nominal scenario
        self.nominal_optimizer = ObjectiveBasedOptimizer(
            plan=self.plan,
            objectives=self.objectives,
            dose_calculator=self.dose_calculator,
        )

        # Create optimizers for all other scenarios
        self.scenario_optimizers = {}
        for scenario in self.scenarios:
            # Apply scenario to plan
            modified_plan = self._apply_scenario_to_plan(self.plan, scenario)

            # Create optimizer for this scenario
            self.scenario_optimizers[scenario.name] = ObjectiveBasedOptimizer(
                plan=modified_plan,
                objectives=self.objectives,
                dose_calculator=self.dose_calculator,
            )

    def _plan_to_params(self, plan: Plan) -> np.ndarray:
        """
        Convert plan to optimization parameters vector.

        Parameters
        ----------
        plan : Plan
            Treatment plan

        Returns
        -------
        np.ndarray
            Parameter vector
        """
        # This is a simplified implementation
        # In a real system, you would extract all relevant plan parameters

        params = []

        # Extract beam weights
        if hasattr(plan, "beams"):
            for beam in plan.beams:
                if hasattr(beam, "weight"):
                    params.append(beam.weight)

        # If no parameters extracted, use a dummy parameter
        if not params:
            params = [1.0]

        return np.array(params)

    def _params_to_plan(self, params: np.ndarray, plan: Plan) -> None:
        """
        Update plan with optimization parameters.

        Parameters
        ----------
        params : np.ndarray
            Parameter vector
        plan : Plan
            Plan to update
        """
        # This is a simplified implementation
        # In a real system, you would update all relevant plan parameters

        param_idx = 0

        # Update beam weights
        if hasattr(plan, "beams"):
            for beam in plan.beams:
                if hasattr(beam, "weight") and param_idx < len(params):
                    beam.weight = params[param_idx]
                    param_idx += 1

    def _calculate_d95(self, structure_name: str, dose_grid: DoseGrid) -> float:
        """
        Calculate D95 for a structure.

        Parameters
        ----------
        structure_name : str
            Name of the structure
        dose_grid : DoseGrid
            Dose grid to calculate from

        Returns
        -------
        float
            D95 value in Gy
        """
        # This is a simplified implementation
        # In a real system, you would use the DVH module
        return 0.0  # Placeholder

    def get_current_plan(self) -> Plan:
        """
        Get the current plan.

        Returns
        -------
        Plan
            The current plan
        """
        return self.plan

    def plot_objective_history(self, figsize=(10, 6)):
        """
        Plot the history of objective values.

        Parameters
        ----------
        figsize : tuple, optional
            Figure size

        Returns
        -------
        matplotlib.figure.Figure
            The figure
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(range(1, len(self.objective_values) + 1), self.objective_values, "o-")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Composite Objective Value")
        ax.set_title("Robust Optimization Progress")
        ax.grid(True)

        # Add best value marker
        if self.objective_values:
            best_idx = np.argmin(self.objective_values)
            best_value = self.objective_values[best_idx]
            ax.plot(
                best_idx + 1,
                best_value,
                "ro",
                markersize=10,
                label=f"Best: {best_value:.4f}",
            )
            ax.legend()

        return fig


def create_robust_objective(
    base_objective: ObjectiveFunction, weight_factor: float = 1.0, priority: int = 1
) -> ObjectiveFunction:
    """
    Create a robust version of an objective function.

    Parameters
    ----------
    base_objective : ObjectiveFunction
        Base objective function
    weight_factor : float, optional
        Factor to apply to weight, by default 1.0
    priority : int, optional
        Priority level, by default 1

    Returns
    -------
    ObjectiveFunction
        Robust objective function
    """
    # Clone the objective
    robust_obj = base_objective.copy()

    # Update weight and priority
    robust_obj.weight *= weight_factor
    robust_obj.priority = priority

    # Add robustness flag
    robust_obj.is_robust = True

    return robust_obj


def optimize_robust_plan(
    plan: Plan,
    objectives: PlanningObjectives,
    dose_calculator: DoseCalculator,
    setup_uncertainty: float = 3.0,
    range_uncertainty: float = 3.5,
    max_iterations: int = 100,
    scenario_sampling: str = "corners",
) -> Dict:
    """
    Optimize a plan with robustness considerations.

    Parameters
    ----------
    plan : Plan
        Treatment plan to optimize
    objectives : PlanningObjectives
        Planning objectives
    dose_calculator : DoseCalculator
        Dose calculation engine
    setup_uncertainty : float, optional
        Setup uncertainty in mm, by default 3.0
    range_uncertainty : float, optional
        Range uncertainty in percent, by default 3.5
    max_iterations : int, optional
        Maximum number of iterations, by default 100
    scenario_sampling : str, optional
        Scenario sampling strategy ('corners', 'random', 'hybrid'), by default 'corners'

    Returns
    -------
    Dict
        Optimization results
    """
    # Create optimizer
    optimizer = RobustOptimizer(plan, objectives, dose_calculator)

    # Set parameters
    optimizer.set_parameter("max_iterations", max_iterations)
    optimizer.set_parameter("setup_uncertainty", setup_uncertainty)
    optimizer.set_parameter("range_uncertainty", range_uncertainty)
    optimizer.set_parameter("scenario_sampling", scenario_sampling)

    # Run optimization
    return optimizer.optimize()
