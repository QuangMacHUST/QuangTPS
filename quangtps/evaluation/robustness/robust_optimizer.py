"""
Robust optimization module for interfacing with the optimization module.

This module provides a unified interface to the robust optimization
functionality in the optimization module, ensuring consistency and
providing additional utility functions.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Union

from quangtps.core.plan import Plan
from quangtps.core.dose import Dose
from quangtps.core.structure import Structure
from quangtps.core.types import DoseGrid
from quangtps.core.logging import get_logger
from quangtps.dose.calculation import DoseCalculator
from quangtps.optimization.objectives import ObjectiveFunction, PlanningObjectives
from quangtps.optimization.constraints import ConstraintFunction
from quangtps.optimization.methods.robust_optimizer import (
    RobustOptimizer as OptRobustOptimizer,
    UncertaintyScenario,
    create_robust_objective,
    optimize_robust_plan as opt_optimize_robust_plan,
)
from .robustness_analyzer import RobustnessAnalyzer, RobustnessResult

logger = get_logger(__name__)


class RobustOptimizer:
    """
    Wrapper for robust optimization that integrates with the evaluation module.

    This class acts as a bridge between the robust optimization functionality
    in the optimization module and the robustness analysis in the evaluation module.
    """

    def __init__(
        self,
        plan: Plan,
        objectives: PlanningObjectives,
        dose_calculator: DoseCalculator,
        structures: Dict[str, Structure] = None,
    ):
        """
        Initialize robust optimizer wrapper.

        Parameters
        ----------
        plan : Plan
            Plan to optimize
        objectives : PlanningObjectives
            Planning objectives
        dose_calculator : DoseCalculator
            Dose calculation engine
        structures : Dict[str, Structure], optional
            Dictionary of structures, by default None
        """
        self.plan = plan
        self.objectives = objectives
        self.dose_calculator = dose_calculator
        self.structures = structures or {}

        # Initialize the underlying optimizer
        self.optimizer = OptRobustOptimizer(plan, objectives, dose_calculator)

        # Initialize analyzer
        self.analyzer = None
        if hasattr(plan, "dose_grid"):
            self.analyzer = RobustnessAnalyzer(
                plan=plan, structures=self.structures, dose_grid=plan.dose_grid
            )

        # Callback function for optimization progress
        self.progress_callback = None

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
        self.optimizer.set_parameter(name, value)

    def set_progress_callback(self, callback):
        """
        Set callback function for optimization progress.

        Parameters
        ----------
        callback : function
            Callback function that takes iteration number as argument
        """
        self.progress_callback = callback

        # Set up iteration callback on underlying optimizer
        def iteration_callback(iteration, obj_value):
            if self.progress_callback is not None:
                self.progress_callback(iteration)

        self.optimizer.parameters["iteration_callback"] = iteration_callback

    def generate_standard_scenarios(
        self, setup_uncertainty: float = 3.0, range_uncertainty: float = 3.5
    ) -> None:
        """
        Generate standard uncertainty scenarios.

        Parameters
        ----------
        setup_uncertainty : float, optional
            Setup uncertainty in mm, by default 3.0
        range_uncertainty : float, optional
            Range uncertainty in percent, by default 3.5
        """
        self.optimizer.generate_standard_scenarios(
            setup_uncertainty=setup_uncertainty, range_uncertainty=range_uncertainty
        )

        # Also set up for analyzer if available
        if self.analyzer is not None:
            self.analyzer.set_setup_uncertainty(setup_uncertainty)
            self.analyzer.set_range_uncertainty(range_uncertainty)

    def optimize(self) -> Tuple[Plan, Optional[RobustnessResult]]:
        """
        Run robust optimization and analysis.

        Returns
        -------
        Tuple[Plan, Optional[RobustnessResult]]
            The optimized plan and robustness analysis results (if available)
        """
        logger.info("Starting robust optimization...")
        opt_result = self.optimizer.optimize()
        optimized_plan = opt_result.get("plan", self.plan)

        robustness_result = None
        if self.analyzer is not None and hasattr(optimized_plan, "dose_grid"):
            logger.info("Starting robustness analysis of optimized plan...")
            try:
                self.analyzer = RobustnessAnalyzer(
                    plan=optimized_plan,
                    structures=self.structures,
                    dose_grid=optimized_plan.dose_grid,
                )
                robustness_result = self.analyzer.analyze()
            except Exception as e:
                logger.error(f"Error analyzing optimized plan: {e}")

        return optimized_plan, robustness_result

    def add_constraint(self, constraint: ConstraintFunction) -> None:
        """
        Add constraint to optimization.

        Parameters
        ----------
        constraint : ConstraintFunction
            Constraint function
        """
        self.optimizer.add_constraint(constraint)

    def get_robustness_metrics(self) -> Dict:
        """
        Get robustness metrics for current plan.

        Returns
        -------
        Dict
            Dictionary of robustness metrics
        """
        if hasattr(self.optimizer, "get_robustness_metrics"):
            return self.optimizer.get_robustness_metrics()
        return {}

    def add_scenario(
        self,
        structures: Dict[str, Structure],
        weight: float = 1.0,
        name: Optional[str] = None,
    ) -> None:
        """
        Add custom scenario with different anatomical structures.

        Parameters
        ----------
        structures : Dict[str, Structure]
            Dictionary of structures for this scenario
        weight : float, optional
            Scenario weight, by default 1.0
        name : Optional[str], optional
            Scenario name, by default None
        """
        self.optimizer.add_scenario(structures, weight, name)


def optimize_robust_plan(
    plan: Plan,
    objectives: PlanningObjectives,
    dose_calculator: DoseCalculator,
    structures: Dict[str, Structure] = None,
    setup_uncertainty: float = 3.0,
    range_uncertainty: float = 3.5,
    progress_callback=None,
) -> Tuple[Plan, Optional[RobustnessResult]]:
    """
    Optimize a plan considering setup and range uncertainties.

    This is a convenience function that combines the functionality of the
    RobustOptimizer class into a single function call.

    Parameters
    ----------
    plan : Plan
        Plan to optimize
    objectives : PlanningObjectives
        Planning objectives
    dose_calculator : DoseCalculator
        Dose calculation engine
    structures : Dict[str, Structure], optional
        Dictionary of structures, by default None
    setup_uncertainty : float, optional
        Setup uncertainty in mm, by default 3.0
    range_uncertainty : float, optional
        Range uncertainty in percent, by default 3.5
    progress_callback : function, optional
        Callback function for optimization progress, by default None

    Returns
    -------
    Tuple[Plan, Optional[RobustnessResult]]
        The optimized plan and robustness analysis results (if available)
    """
    optimizer = RobustOptimizer(plan, objectives, dose_calculator, structures)
    optimizer.generate_standard_scenarios(setup_uncertainty, range_uncertainty)

    if progress_callback is not None:
        optimizer.set_progress_callback(progress_callback)

    return optimizer.optimize()
