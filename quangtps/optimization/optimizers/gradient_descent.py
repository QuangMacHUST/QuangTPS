#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Gradient Descent Optimizer for QuangTPS.

This module implements gradient descent optimization algorithms for
treatment plan optimization.
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Any, Callable, Tuple
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class GradientDescentOptimizer:
    """
    Gradient Descent optimizer cho tối ưu hóa kế hoạch xạ trị.

    Implements various gradient descent variants including:
    - Standard gradient descent
    - Momentum-based gradient descent
    - Adam optimizer
    - RMSprop
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        momentum: float = 0.9,
        variant: str = "adam",
        max_iterations: int = 1000,
        tolerance: float = 1e-6,
        **kwargs,
    ):
        """
        Khởi tạo Gradient Descent optimizer.

        Parameters
        ----------
        learning_rate : float
            Tốc độ học (learning rate)
        momentum : float
            Hệ số momentum cho momentum-based methods
        variant : str
            Biến thể gradient descent ("standard", "momentum", "adam", "rmsprop")
        max_iterations : int
            Số lần lặp tối đa
        tolerance : float
            Tolerance cho convergence
        """
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.variant = variant.lower()
        self.max_iterations = max_iterations
        self.tolerance = tolerance

        # Optimizer state
        self.current_iteration = 0
        self.best_objective_value = float("inf")
        self.convergence_history = []

        # Adam/RMSprop parameters
        self.beta1 = kwargs.get("beta1", 0.9)
        self.beta2 = kwargs.get("beta2", 0.999)
        self.epsilon = kwargs.get("epsilon", 1e-8)

        # State variables for momentum/adam
        self.velocity = None
        self.momentum_buffer = None
        self.adam_m = None  # First moment estimate
        self.adam_v = None  # Second moment estimate

        logger.info(f"Initialized Gradient Descent optimizer (variant: {self.variant})")

    def optimize(
        self,
        objectives: List[Any],
        initial_parameters: np.ndarray,
        constraints: Optional[List[Any]] = None,
        progress_callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Thực hiện tối ưu hóa gradient descent.

        Parameters
        ----------
        objectives : List[Any]
            Danh sách các objective functions
        initial_parameters : np.ndarray
            Tham số khởi tạo
        constraints : List[Any], optional
            Danh sách constraints (chưa implement)
        progress_callback : Callable, optional
            Callback function cho progress updates

        Returns
        -------
        Dict[str, Any]
            Kết quả tối ưu hóa
        """
        logger.info(f"Starting {self.variant} optimization")

        # Initialize parameters
        current_params = initial_parameters.copy()
        param_shape = current_params.shape

        # Initialize optimizer state
        self._initialize_state(param_shape)

        # Initialize best tracking
        self.best_objective_value = float("inf")
        self.convergence_history = []
        best_params = initial_parameters.copy()  # Khởi tạo best_params

        # Main optimization loop
        for iteration in range(self.max_iterations):
            self.current_iteration = iteration

            # Evaluate objectives and compute gradients
            objective_value, gradient = self._evaluate_objectives_and_gradients(
                objectives, current_params
            )

            # Track best solution
            if objective_value < self.best_objective_value:
                self.best_objective_value = objective_value
                best_params = current_params.copy()

            # Store convergence info
            convergence_info = {
                "iteration": iteration,
                "objective_value": objective_value,
                "gradient_norm": np.linalg.norm(gradient),
                "learning_rate": self.learning_rate,
            }
            self.convergence_history.append(convergence_info)

            # Check convergence
            if self._check_convergence(gradient):
                logger.info(f"Converged at iteration {iteration}")
                break

            # Update parameters
            current_params = self._update_parameters(current_params, gradient)

            # Apply constraints if provided
            if constraints:
                current_params = self._apply_constraints(current_params, constraints)

            # Progress callback
            if progress_callback and iteration % 10 == 0:
                progress = (iteration / self.max_iterations) * 100
                progress_callback(
                    progress, f"Iteration {iteration}: obj = {objective_value:.6f}"
                )

        # Compile results
        results = {
            "optimal_parameters": best_params,
            "optimal_objective_value": self.best_objective_value,
            "iterations": self.current_iteration + 1,
            "convergence_history": self.convergence_history,
            "converged": iteration < self.max_iterations - 1,
            "final_gradient_norm": np.linalg.norm(gradient),
            "optimizer_variant": self.variant,
        }

        logger.info(
            f"Optimization completed. Best objective: {self.best_objective_value:.6f}"
        )
        return results

    def _initialize_state(self, param_shape: Tuple[int, ...]) -> None:
        """Initialize optimizer state variables."""
        if self.variant == "momentum":
            self.momentum_buffer = np.zeros(param_shape)
        elif self.variant == "adam":
            self.adam_m = np.zeros(param_shape)
            self.adam_v = np.zeros(param_shape)
        elif self.variant == "rmsprop":
            self.velocity = np.zeros(param_shape)

    def _evaluate_objectives_and_gradients(
        self, objectives: List[Any], parameters: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """Evaluate objectives and compute gradients."""
        total_objective = 0.0
        total_gradient = np.zeros_like(parameters)

        for objective in objectives:
            try:
                # Evaluate objective
                obj_value = objective.evaluate(parameters)
                total_objective += obj_value

                # Compute gradient (numerical if no analytical gradient)
                if hasattr(objective, "gradient"):
                    grad = objective.gradient(parameters)
                else:
                    grad = self._numerical_gradient(objective, parameters)

                total_gradient += grad

            except Exception as e:
                logger.warning(f"Error evaluating objective {objective}: {e}")
                continue

        return total_objective, total_gradient

    def _numerical_gradient(
        self, objective: Any, parameters: np.ndarray, epsilon: float = 1e-8
    ) -> np.ndarray:
        """Compute numerical gradient using finite differences."""
        gradient = np.zeros_like(parameters)
        flat_params = parameters.flatten()

        for i in range(len(flat_params)):
            # Forward difference
            params_plus = flat_params.copy()
            params_plus[i] += epsilon

            params_minus = flat_params.copy()
            params_minus[i] -= epsilon

            # Reshape and evaluate
            obj_plus = objective.evaluate(params_plus.reshape(parameters.shape))
            obj_minus = objective.evaluate(params_minus.reshape(parameters.shape))

            # Central difference
            gradient.flat[i] = (obj_plus - obj_minus) / (2 * epsilon)

        return gradient

    def _update_parameters(
        self, parameters: np.ndarray, gradient: np.ndarray
    ) -> np.ndarray:
        """Update parameters based on chosen variant."""
        if self.variant == "standard":
            return parameters - self.learning_rate * gradient

        elif self.variant == "momentum":
            self.momentum_buffer = (
                self.momentum * self.momentum_buffer + self.learning_rate * gradient
            )
            return parameters - self.momentum_buffer

        elif self.variant == "adam":
            # Adam optimizer
            self.adam_m = self.beta1 * self.adam_m + (1 - self.beta1) * gradient
            self.adam_v = self.beta2 * self.adam_v + (1 - self.beta2) * (gradient**2)

            # Bias correction
            m_corrected = self.adam_m / (1 - self.beta1 ** (self.current_iteration + 1))
            v_corrected = self.adam_v / (1 - self.beta2 ** (self.current_iteration + 1))

            return parameters - self.learning_rate * m_corrected / (
                np.sqrt(v_corrected) + self.epsilon
            )

        elif self.variant == "rmsprop":
            # RMSprop optimizer
            self.velocity = self.beta2 * self.velocity + (1 - self.beta2) * (
                gradient**2
            )
            return parameters - self.learning_rate * gradient / (
                np.sqrt(self.velocity) + self.epsilon
            )

        else:
            logger.warning(
                f"Unknown variant {self.variant}, using standard gradient descent"
            )
            return parameters - self.learning_rate * gradient

    def _check_convergence(self, gradient: np.ndarray) -> bool:
        """Check if optimization has converged."""
        gradient_norm = np.linalg.norm(gradient)

        # Gradient-based convergence
        if gradient_norm < self.tolerance:
            return True

        # Objective value convergence
        if len(self.convergence_history) > 10:
            recent_changes = np.abs(np.diff(self.convergence_history[-10:]))
            if np.max(recent_changes) < self.tolerance:
                return True

        return False

    def _apply_constraints(
        self, parameters: np.ndarray, constraints: List[Any]
    ) -> np.ndarray:
        """Apply constraints to parameters (placeholder)."""
        # TODO: Implement constraint handling
        logger.warning("Constraint handling not yet implemented")
        return parameters

    def reset(self) -> None:
        """Reset optimizer state."""
        self.current_iteration = 0
        self.best_objective_value = float("inf")
        self.convergence_history = []
        self.velocity = None
        self.momentum_buffer = None
        self.adam_m = None
        self.adam_v = None

        logger.info("Optimizer state reset")


# Factory function
def create_gradient_descent_optimizer(**kwargs) -> GradientDescentOptimizer:
    """Create a gradient descent optimizer with specified parameters."""
    return GradientDescentOptimizer(**kwargs)


__all__ = ["GradientDescentOptimizer", "create_gradient_descent_optimizer"]
