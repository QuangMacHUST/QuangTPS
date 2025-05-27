#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module defining simulated annealing optimization for radiotherapy treatment planning.

This module implements the Simulated Annealing algorithm which uses a
probabilistic approach to finding global optima, especially useful for
treatment plan optimization when the objective function has many local minima.
"""

import numpy as np
import time
import random
import math
import logging
from typing import Dict, List, Tuple, Optional, Union, Any, Callable

# Import the base classes defined in gradient_descent.py
from quangtps.optimization.gradient_descent import (
    OptimizationSolver,
    OptimizationResult,
    Constraint,
)

logger = logging.getLogger(__name__)


class SimulatedAnnealing(OptimizationSolver):
    """
    Simulated Annealing optimization algorithm for treatment planning.

    Simulated annealing is a probabilistic technique for approximating the global
    optimum of a given function, particularly useful for avoiding local minima.
    """

    def __init__(
        self,
        initial_temp: float = 100.0,
        final_temp: float = 0.1,
        cooling_rate: float = 0.95,
        max_iterations: int = 1000,
        max_iterations_at_temp: int = 50,
    ):
        """
        Initialize Simulated Annealing optimizer.

        Parameters
        ----------
        initial_temp : float
            Initial temperature for annealing
        final_temp : float
            Final temperature where the algorithm terminates
        cooling_rate : float
            Rate at which temperature decreases (0-1)
        max_iterations : int
            Maximum number of total iterations
        max_iterations_at_temp : int
            Maximum iterations at each temperature
        """
        super().__init__()
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.cooling_rate = cooling_rate
        self.max_iterations = max_iterations
        self.max_iterations_at_temp = max_iterations_at_temp
        self.temperature = initial_temp
        self.iteration_count = 0
        self.best_objective = float("inf")
        self.best_params = None

        logger.info(
            f"Initialized Simulated Annealing optimizer with initial_temp={initial_temp}, "
            f"final_temp={final_temp}, cooling_rate={cooling_rate}"
        )

    def initialize(self, params: Dict[str, Any]) -> bool:
        """
        Initialize the optimizer with parameters.

        Parameters
        ----------
        params : Dict[str, Any]
            Parameters for the optimizer

        Returns
        -------
        bool
            True if successful
        """
        if "initial_temp" in params:
            self.initial_temp = params["initial_temp"]
        if "final_temp" in params:
            self.final_temp = params["final_temp"]
        if "cooling_rate" in params:
            self.cooling_rate = params["cooling_rate"]
        if "max_iterations" in params:
            self.max_iterations = params["max_iterations"]
        if "max_iterations_at_temp" in params:
            self.max_iterations_at_temp = params["max_iterations_at_temp"]

        self.temperature = self.initial_temp
        self.iteration_count = 0

        return True

    def optimize(
        self,
        objective_function: Callable[[np.ndarray], float],
        initial_parameters: np.ndarray,
        parameter_bounds: Optional[List[Tuple[float, float]]] = None,
        constraints: List[Constraint] = None,
        callback: Callable[[int, np.ndarray, float], None] = None,
    ) -> OptimizationResult:
        """
        Optimize using simulated annealing.

        Parameters
        ----------
        objective_function : Callable
            Function to minimize
        initial_parameters : np.ndarray
            Starting point for optimization
        parameter_bounds : List[Tuple[float, float]], optional
            Bounds for each parameter (min, max)
        constraints : List[Constraint], optional
            List of constraints to apply
        callback : Callable, optional
            Function to call after each iteration with (iteration, params, obj_value)

        Returns
        -------
        OptimizationResult
            Results of the optimization
        """
        start_time = time.time()

        # Initialize parameters and history
        current_params = initial_parameters.copy()
        current_objective = objective_function(current_params)

        # Best so far
        best_params = current_params.copy()
        best_objective = current_objective

        # Setup bounds if not provided
        if parameter_bounds is None:
            parameter_bounds = [
                (-np.inf, np.inf) for _ in range(len(initial_parameters))
            ]

        # History tracking
        objective_history = [current_objective]
        param_history = [current_params.copy()]
        temp_history = [self.temperature]

        # Main optimization loop
        iteration = 0
        total_iterations = 0
        converged = False
        message = "Maximum iterations reached"

        # Continue until temperature is too low or max iterations reached
        while (
            self.temperature > self.final_temp
            and total_iterations < self.max_iterations
        ):
            # For each temperature, try multiple iterations
            for _ in range(self.max_iterations_at_temp):
                total_iterations += 1
                iteration += 1

                # Generate a neighbor solution
                neighbor_params = self._generate_neighbor(
                    current_params, parameter_bounds
                )

                # Apply constraints if any
                valid_neighbor = True
                if constraints:
                    for constraint in constraints:
                        if constraint.is_hard and not constraint.evaluate(
                            neighbor_params
                        ):
                            valid_neighbor = False
                            break

                # Skip invalid neighbors
                if not valid_neighbor:
                    continue

                # Evaluate neighbor
                neighbor_objective = objective_function(neighbor_params)

                # Calculate acceptance probability
                if neighbor_objective < current_objective:
                    # Always accept better solutions
                    accept_probability = 1.0
                else:
                    # Metropolis criterion for accepting worse solutions
                    delta = neighbor_objective - current_objective
                    accept_probability = math.exp(-delta / self.temperature)

                # Accept the neighbor based on probability
                if random.random() < accept_probability:
                    current_params = neighbor_params.copy()
                    current_objective = neighbor_objective

                    # Update best solution if improved
                    if current_objective < best_objective:
                        best_params = current_params.copy()
                        best_objective = current_objective

                # Track history
                objective_history.append(current_objective)
                param_history.append(current_params.copy())
                temp_history.append(self.temperature)

                # Call the callback if provided
                if callback:
                    callback(total_iterations, current_params, current_objective)

                # Check for early convergence
                if iteration > 100:
                    # If no improvement for a while, assume convergence
                    if abs(objective_history[-1] - objective_history[-50]) < 1e-6:
                        converged = True
                        message = (
                            "Converged: no significant improvement for many iterations"
                        )
                        break

            # Cool down the temperature
            self.temperature *= self.cooling_rate

            # Break if converged
            if converged:
                break

        # Create result object
        result = OptimizationResult(
            best_params, best_objective, total_iterations, converged, message
        )

        # Add additional information
        result.execution_time = time.time() - start_time
        result.history = {
            "objective": objective_history,
            "parameters": param_history,
            "temperature": temp_history,
        }

        logger.info(
            f"Simulated annealing optimization completed: {message}, "
            f"iterations={total_iterations}, final_objective={best_objective:.6f}"
        )

        return result

    def _generate_neighbor(
        self, params: np.ndarray, bounds: List[Tuple[float, float]]
    ) -> np.ndarray:
        """
        Generate a neighboring solution by perturbing the current solution.

        Parameters
        ----------
        params : np.ndarray
            Current parameter values
        bounds : List[Tuple[float, float]]
            Bounds for each parameter

        Returns
        -------
        np.ndarray
            New neighbor solution
        """
        # Create a copy of current parameters
        neighbor = params.copy()

        # Choose a random parameter to modify
        idx = random.randint(0, len(params) - 1)

        # The perturbation size is relative to the temperature
        perturbation = np.random.normal(0, self.temperature * 0.1)

        # Apply perturbation
        neighbor[idx] += perturbation

        # Enforce bounds
        min_val, max_val = bounds[idx]
        neighbor[idx] = max(min_val, min(max_val, neighbor[idx]))

        return neighbor

    def get_parameters(self) -> Dict[str, Any]:
        """
        Get the current parameters of the optimizer.

        Returns
        -------
        Dict[str, Any]
            Dictionary of optimizer parameters
        """
        return {
            "initial_temp": self.initial_temp,
            "final_temp": self.final_temp,
            "cooling_rate": self.cooling_rate,
            "max_iterations": self.max_iterations,
            "max_iterations_at_temp": self.max_iterations_at_temp,
            "current_temp": self.temperature,
            "iteration_count": self.iteration_count,
            "best_objective": self.best_objective
            if hasattr(self, "best_objective")
            else None,
        }

    def __str__(self) -> str:
        """
        String representation of the optimizer.

        Returns
        -------
        str
            String representation
        """
        return (
            f"SimulatedAnnealing(initial_temp={self.initial_temp}, "
            f"final_temp={self.final_temp}, "
            f"cooling_rate={self.cooling_rate}, "
            f"max_iterations={self.max_iterations})"
        )


# Alias for backward compatibility
SimulatedAnnealingOptimizer = SimulatedAnnealing
