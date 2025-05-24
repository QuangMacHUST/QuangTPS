#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module tối ưu hóa nâng cao cho QuangTPS.

Module này cung cấp các thuật toán tối ưu hóa tiên tiến cho lập kế hoạch xạ trị,
bao gồm gradient descent, evolutionary algorithms, và các phương pháp hybrid.
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass
from enum import Enum
import time
import json
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

# Try importing optimization libraries
try:
    from scipy.optimize import minimize, differential_evolution

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    logger.warning("SciPy not available - limited optimization algorithms")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim

    HAS_PYTORCH = True
except ImportError:
    HAS_PYTORCH = False
    logger.warning("PyTorch not available - no GPU acceleration")

try:
    from numba import jit, cuda

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    logger.warning("Numba not available - no JIT compilation")


class OptimizationAlgorithm(Enum):
    """Các thuật toán tối ưu hóa có sẵn."""

    GRADIENT_DESCENT = "gradient_descent"
    ADAM = "adam"
    LBFGS = "lbfgs"
    DIFFERENTIAL_EVOLUTION = "differential_evolution"
    GENETIC_ALGORITHM = "genetic_algorithm"
    SIMULATED_ANNEALING = "simulated_annealing"
    PARTICLE_SWARM = "particle_swarm"
    HYBRID_GA_SA = "hybrid_ga_sa"


class ConvergenceCriteria(Enum):
    """Tiêu chí hội tụ."""

    OBJECTIVE_CHANGE = "objective_change"
    GRADIENT_NORM = "gradient_norm"
    MAX_ITERATIONS = "max_iterations"
    TIME_LIMIT = "time_limit"
    COMBINED = "combined"


@dataclass
class OptimizationParameters:
    """Tham số tối ưu hóa."""

    # General parameters
    algorithm: OptimizationAlgorithm = OptimizationAlgorithm.ADAM
    max_iterations: int = 1000
    tolerance: float = 1e-6
    time_limit: float = 3600.0  # seconds

    # Learning rate parameters
    learning_rate: float = 0.01
    learning_rate_decay: float = 0.95
    min_learning_rate: float = 1e-6

    # Population-based algorithms
    population_size: int = 50
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8

    # Simulated annealing
    initial_temperature: float = 100.0
    cooling_rate: float = 0.95
    min_temperature: float = 0.01

    # Particle swarm
    w: float = 0.729  # inertia weight
    c1: float = 1.494  # cognitive parameter
    c2: float = 1.494  # social parameter

    # Advanced parameters
    use_gpu: bool = False
    use_adaptive_learning: bool = True
    save_history: bool = True
    verbose: int = 1


@dataclass
class OptimizationResult:
    """Kết quả tối ưu hóa."""

    success: bool
    optimal_parameters: np.ndarray
    optimal_objective: float
    iterations: int
    time_elapsed: float
    convergence_history: List[float]
    gradient_history: List[float]
    message: str
    algorithm_used: OptimizationAlgorithm


class ObjectiveFunction(ABC):
    """Base class cho objective functions."""

    @abstractmethod
    def evaluate(self, parameters: np.ndarray) -> float:
        """Evaluate objective function."""
        pass

    @abstractmethod
    def gradient(self, parameters: np.ndarray) -> np.ndarray:
        """Compute gradient of objective function."""
        pass

    def hessian(self, parameters: np.ndarray) -> np.ndarray:
        """Compute Hessian matrix (optional)."""
        # Default finite difference approximation
        eps = 1e-8
        n = len(parameters)
        hess = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                x_pp = parameters.copy()
                x_pm = parameters.copy()
                x_mp = parameters.copy()
                x_mm = parameters.copy()

                x_pp[i] += eps
                x_pp[j] += eps

                x_pm[i] += eps
                x_pm[j] -= eps

                x_mp[i] -= eps
                x_mp[j] += eps

                x_mm[i] -= eps
                x_mm[j] -= eps

                hess[i, j] = (
                    self.evaluate(x_pp)
                    - self.evaluate(x_pm)
                    - self.evaluate(x_mp)
                    + self.evaluate(x_mm)
                ) / (4 * eps * eps)

        return hess


class DoseObjectiveFunction(ObjectiveFunction):
    """
    Objective function cho tối ưu hóa phân bố liều.

    Tích hợp các objectives từ DVH constraints, dose constraints,
    và biological models.
    """

    def __init__(self, dose_calculator, structures, objectives, constraints):
        """
        Initialize dose objective function.

        Parameters
        ----------
        dose_calculator : object
            Dose calculation engine
        structures : dict
            Structure masks and information
        objectives : list
            List of optimization objectives
        constraints : list
            List of constraints
        """
        self.dose_calculator = dose_calculator
        self.structures = structures
        self.objectives = objectives
        self.constraints = constraints

        # Cache for dose calculation
        self._last_parameters = None
        self._last_dose = None

    def evaluate(self, parameters: np.ndarray) -> float:
        """Evaluate total objective function."""
        # Calculate dose distribution
        dose_distribution = self._calculate_dose(parameters)

        # Evaluate all objectives
        total_objective = 0.0

        for objective in self.objectives:
            try:
                obj_value = self._evaluate_single_objective(
                    objective, dose_distribution
                )
                total_objective += objective.weight * obj_value
            except Exception as e:
                logger.warning(f"Error evaluating objective {objective.name}: {str(e)}")
                total_objective += 1e6  # Penalty for failed evaluation

        # Add constraint penalties
        penalty = self._evaluate_constraints(dose_distribution)
        total_objective += penalty

        return total_objective

    def gradient(self, parameters: np.ndarray) -> np.ndarray:
        """Compute gradient using finite differences."""
        eps = 1e-6
        gradient = np.zeros_like(parameters)
        f0 = self.evaluate(parameters)

        for i in range(len(parameters)):
            params_plus = parameters.copy()
            params_plus[i] += eps
            f_plus = self.evaluate(params_plus)
            gradient[i] = (f_plus - f0) / eps

        return gradient

    def _calculate_dose(self, parameters: np.ndarray) -> np.ndarray:
        """Calculate dose distribution for given parameters."""
        # Use cache if parameters haven't changed
        if self._last_parameters is not None and np.allclose(
            parameters, self._last_parameters
        ):
            return self._last_dose

        # Calculate new dose distribution
        try:
            dose = self.dose_calculator.calculate(parameters)
            self._last_parameters = parameters.copy()
            self._last_dose = dose.copy()
            return dose
        except Exception as e:
            logger.error(f"Dose calculation failed: {str(e)}")
            # Return zero dose as fallback
            return np.zeros((64, 64, 32))

    def _evaluate_single_objective(
        self, objective, dose_distribution: np.ndarray
    ) -> float:
        """Evaluate a single objective."""
        structure_name = objective.structure_name

        if structure_name not in self.structures:
            logger.warning(f"Structure {structure_name} not found")
            return 0.0

        structure_mask = self.structures[structure_name]
        structure_dose = dose_distribution[structure_mask]

        if objective.type == "mean_dose":
            target_dose = objective.target_value
            actual_dose = np.mean(structure_dose)
            return (actual_dose - target_dose) ** 2

        elif objective.type == "max_dose":
            target_dose = objective.target_value
            actual_dose = np.max(structure_dose)
            if actual_dose > target_dose:
                return (actual_dose - target_dose) ** 2
            return 0.0

        elif objective.type == "dvh_constraint":
            dose_level = objective.dose_level
            volume_limit = objective.volume_limit
            exceeding_volume = np.sum(structure_dose > dose_level) / len(structure_dose)
            if exceeding_volume > volume_limit:
                return (exceeding_volume - volume_limit) ** 2
            return 0.0

        else:
            logger.warning(f"Unknown objective type: {objective.type}")
            return 0.0

    def _evaluate_constraints(self, dose_distribution: np.ndarray) -> float:
        """Evaluate constraint penalties."""
        penalty = 0.0

        for constraint in self.constraints:
            violation = self._check_constraint(constraint, dose_distribution)
            if violation > 0:
                penalty += constraint.penalty_weight * violation**2

        return penalty

    def _check_constraint(self, constraint, dose_distribution: np.ndarray) -> float:
        """Check single constraint violation."""
        # Implementation depends on constraint type
        return 0.0


class AdvancedOptimizer:
    """
    Advanced optimization engine với multiple algorithms.

    Supports gradient-based, evolutionary, and hybrid optimization methods
    with GPU acceleration when available.
    """

    def __init__(self, parameters: OptimizationParameters = None):
        """Initialize optimizer."""
        self.params = parameters or OptimizationParameters()
        self.history = []
        self.best_solution = None
        self.best_objective = float("inf")

        # Check for GPU availability
        self.use_gpu = self.params.use_gpu and HAS_PYTORCH and torch.cuda.is_available()
        if self.use_gpu:
            self.device = torch.device("cuda")
            logger.info("Using GPU acceleration for optimization")
        else:
            self.device = torch.device("cpu")

    def optimize(
        self,
        objective_function: ObjectiveFunction,
        initial_parameters: np.ndarray,
        bounds: List[Tuple[float, float]] = None,
    ) -> OptimizationResult:
        """
        Run optimization with specified algorithm.

        Parameters
        ----------
        objective_function : ObjectiveFunction
            Function to optimize
        initial_parameters : np.ndarray
            Starting point for optimization
        bounds : List[Tuple[float, float]], optional
            Parameter bounds

        Returns
        -------
        OptimizationResult
            Optimization results
        """
        start_time = time.time()

        try:
            if self.params.algorithm == OptimizationAlgorithm.GRADIENT_DESCENT:
                result = self._gradient_descent(
                    objective_function, initial_parameters, bounds
                )
            elif self.params.algorithm == OptimizationAlgorithm.ADAM:
                result = self._adam_optimizer(
                    objective_function, initial_parameters, bounds
                )
            elif self.params.algorithm == OptimizationAlgorithm.LBFGS:
                result = self._lbfgs_optimizer(
                    objective_function, initial_parameters, bounds
                )
            elif self.params.algorithm == OptimizationAlgorithm.DIFFERENTIAL_EVOLUTION:
                result = self._differential_evolution(
                    objective_function, initial_parameters, bounds
                )
            elif self.params.algorithm == OptimizationAlgorithm.GENETIC_ALGORITHM:
                result = self._genetic_algorithm(
                    objective_function, initial_parameters, bounds
                )
            elif self.params.algorithm == OptimizationAlgorithm.SIMULATED_ANNEALING:
                result = self._simulated_annealing(
                    objective_function, initial_parameters, bounds
                )
            elif self.params.algorithm == OptimizationAlgorithm.PARTICLE_SWARM:
                result = self._particle_swarm(
                    objective_function, initial_parameters, bounds
                )
            else:
                raise ValueError(
                    f"Unknown optimization algorithm: {self.params.algorithm}"
                )

            result.time_elapsed = time.time() - start_time
            return result

        except Exception as e:
            logger.error(f"Optimization failed: {str(e)}")
            return OptimizationResult(
                success=False,
                optimal_parameters=initial_parameters,
                optimal_objective=float("inf"),
                iterations=0,
                time_elapsed=time.time() - start_time,
                convergence_history=[],
                gradient_history=[],
                message=f"Optimization failed: {str(e)}",
                algorithm_used=self.params.algorithm,
            )

    def _gradient_descent(
        self,
        objective_function: ObjectiveFunction,
        initial_parameters: np.ndarray,
        bounds: List[Tuple[float, float]] = None,
    ) -> OptimizationResult:
        """Standard gradient descent optimization."""
        parameters = initial_parameters.copy()
        learning_rate = self.params.learning_rate
        convergence_history = []
        gradient_history = []

        for iteration in range(self.params.max_iterations):
            # Evaluate objective and gradient
            obj_value = objective_function.evaluate(parameters)
            gradient = objective_function.gradient(parameters)

            # Store history
            convergence_history.append(obj_value)
            gradient_history.append(np.linalg.norm(gradient))

            # Check convergence
            if np.linalg.norm(gradient) < self.params.tolerance:
                message = f"Converged after {iteration} iterations (gradient norm)"
                break

            # Update parameters
            parameters -= learning_rate * gradient

            # Apply bounds if specified
            if bounds:
                parameters = self._apply_bounds(parameters, bounds)

            # Adaptive learning rate
            if self.params.use_adaptive_learning and iteration > 10:
                recent_change = abs(convergence_history[-1] - convergence_history[-10])
                if recent_change < self.params.tolerance:
                    learning_rate *= self.params.learning_rate_decay
                    learning_rate = max(learning_rate, self.params.min_learning_rate)

            # Progress reporting
            if self.params.verbose and iteration % 100 == 0:
                logger.info(
                    f"Iteration {iteration}: Objective = {obj_value:.6f}, "
                    f"Gradient norm = {np.linalg.norm(gradient):.6f}"
                )
        else:
            message = f"Max iterations ({self.params.max_iterations}) reached"

        return OptimizationResult(
            success=True,
            optimal_parameters=parameters,
            optimal_objective=convergence_history[-1],
            iterations=len(convergence_history),
            time_elapsed=0.0,  # Will be set by caller
            convergence_history=convergence_history,
            gradient_history=gradient_history,
            message=message,
            algorithm_used=OptimizationAlgorithm.GRADIENT_DESCENT,
        )

    def _adam_optimizer(
        self,
        objective_function: ObjectiveFunction,
        initial_parameters: np.ndarray,
        bounds: List[Tuple[float, float]] = None,
    ) -> OptimizationResult:
        """Adam optimization algorithm."""
        parameters = initial_parameters.copy()
        m = np.zeros_like(parameters)  # First moment
        v = np.zeros_like(parameters)  # Second moment
        beta1, beta2 = 0.9, 0.999
        epsilon = 1e-8

        convergence_history = []
        gradient_history = []

        for iteration in range(self.params.max_iterations):
            # Evaluate objective and gradient
            obj_value = objective_function.evaluate(parameters)
            gradient = objective_function.gradient(parameters)

            # Store history
            convergence_history.append(obj_value)
            gradient_history.append(np.linalg.norm(gradient))

            # Check convergence
            if np.linalg.norm(gradient) < self.params.tolerance:
                message = f"Converged after {iteration} iterations (gradient norm)"
                break

            # Update biased first and second moment estimates
            m = beta1 * m + (1 - beta1) * gradient
            v = beta2 * v + (1 - beta2) * gradient**2

            # Compute bias-corrected moment estimates
            m_hat = m / (1 - beta1 ** (iteration + 1))
            v_hat = v / (1 - beta2 ** (iteration + 1))

            # Update parameters
            parameters -= self.params.learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

            # Apply bounds if specified
            if bounds:
                parameters = self._apply_bounds(parameters, bounds)

            # Progress reporting
            if self.params.verbose and iteration % 100 == 0:
                logger.info(f"Iteration {iteration}: Objective = {obj_value:.6f}")
        else:
            message = f"Max iterations ({self.params.max_iterations}) reached"

        return OptimizationResult(
            success=True,
            optimal_parameters=parameters,
            optimal_objective=convergence_history[-1],
            iterations=len(convergence_history),
            time_elapsed=0.0,
            convergence_history=convergence_history,
            gradient_history=gradient_history,
            message=message,
            algorithm_used=OptimizationAlgorithm.ADAM,
        )

    def _lbfgs_optimizer(
        self,
        objective_function: ObjectiveFunction,
        initial_parameters: np.ndarray,
        bounds: List[Tuple[float, float]] = None,
    ) -> OptimizationResult:
        """L-BFGS optimization using SciPy."""
        if not HAS_SCIPY:
            raise ValueError("SciPy required for L-BFGS optimization")

        convergence_history = []

        def callback(x):
            obj_val = objective_function.evaluate(x)
            convergence_history.append(obj_val)
            if self.params.verbose and len(convergence_history) % 10 == 0:
                logger.info(
                    f"Iteration {len(convergence_history)}: Objective = {obj_val:.6f}"
                )

        result = minimize(
            objective_function.evaluate,
            initial_parameters,
            method="L-BFGS-B",
            jac=objective_function.gradient,
            bounds=bounds,
            options={
                "maxiter": self.params.max_iterations,
                "ftol": self.params.tolerance,
                "gtol": self.params.tolerance,
            },
            callback=callback,
        )

        return OptimizationResult(
            success=result.success,
            optimal_parameters=result.x,
            optimal_objective=result.fun,
            iterations=result.nit,
            time_elapsed=0.0,
            convergence_history=convergence_history,
            gradient_history=[],
            message=result.message,
            algorithm_used=OptimizationAlgorithm.LBFGS,
        )

    def _differential_evolution(
        self,
        objective_function: ObjectiveFunction,
        initial_parameters: np.ndarray,
        bounds: List[Tuple[float, float]] = None,
    ) -> OptimizationResult:
        """Differential evolution optimization."""
        if not HAS_SCIPY:
            raise ValueError("SciPy required for differential evolution")

        if bounds is None:
            # Create default bounds
            bounds = [(-10, 10) for _ in range(len(initial_parameters))]

        convergence_history = []

        def callback(x, convergence):
            obj_val = objective_function.evaluate(x)
            convergence_history.append(obj_val)
            if self.params.verbose and len(convergence_history) % 10 == 0:
                logger.info(
                    f"Generation {len(convergence_history)}: Best = {obj_val:.6f}"
                )

        result = differential_evolution(
            objective_function.evaluate,
            bounds,
            maxiter=self.params.max_iterations,
            popsize=self.params.population_size,
            tol=self.params.tolerance,
            callback=callback,
        )

        return OptimizationResult(
            success=result.success,
            optimal_parameters=result.x,
            optimal_objective=result.fun,
            iterations=result.nit,
            time_elapsed=0.0,
            convergence_history=convergence_history,
            gradient_history=[],
            message=result.message,
            algorithm_used=OptimizationAlgorithm.DIFFERENTIAL_EVOLUTION,
        )

    def _genetic_algorithm(
        self,
        objective_function: ObjectiveFunction,
        initial_parameters: np.ndarray,
        bounds: List[Tuple[float, float]] = None,
    ) -> OptimizationResult:
        """Custom genetic algorithm implementation."""
        if bounds is None:
            bounds = [(-10, 10) for _ in range(len(initial_parameters))]

        # Initialize population
        pop_size = self.params.population_size
        n_params = len(initial_parameters)
        population = np.random.uniform(
            [b[0] for b in bounds], [b[1] for b in bounds], (pop_size, n_params)
        )

        # Include initial parameters in population
        population[0] = initial_parameters

        convergence_history = []
        best_individual = None
        best_fitness = float("inf")

        for generation in range(self.params.max_iterations // pop_size):
            # Evaluate fitness
            fitness = np.array([objective_function.evaluate(ind) for ind in population])

            # Track best
            gen_best_idx = np.argmin(fitness)
            if fitness[gen_best_idx] < best_fitness:
                best_fitness = fitness[gen_best_idx]
                best_individual = population[gen_best_idx].copy()

            convergence_history.append(best_fitness)

            # Selection (tournament)
            new_population = []
            for _ in range(pop_size):
                # Tournament selection
                tournament_size = 3
                tournament_indices = np.random.choice(
                    pop_size, tournament_size, replace=False
                )
                tournament_fitness = fitness[tournament_indices]
                winner_idx = tournament_indices[np.argmin(tournament_fitness)]
                new_population.append(population[winner_idx].copy())

            population = np.array(new_population)

            # Crossover and mutation
            for i in range(0, pop_size - 1, 2):
                if np.random.random() < self.params.crossover_rate:
                    # Single-point crossover
                    crossover_point = np.random.randint(1, n_params)
                    (
                        population[i, crossover_point:],
                        population[i + 1, crossover_point:],
                    ) = (
                        population[i + 1, crossover_point:].copy(),
                        population[i, crossover_point:].copy(),
                    )

                # Mutation
                for j in range(2):
                    individual = population[i + j]
                    for k in range(n_params):
                        if np.random.random() < self.params.mutation_rate:
                            mutation_strength = 0.1 * (bounds[k][1] - bounds[k][0])
                            individual[k] += np.random.normal(0, mutation_strength)
                            individual[k] = np.clip(
                                individual[k], bounds[k][0], bounds[k][1]
                            )

            # Progress reporting
            if self.params.verbose and generation % 10 == 0:
                logger.info(
                    f"Generation {generation}: Best fitness = {best_fitness:.6f}"
                )

        return OptimizationResult(
            success=True,
            optimal_parameters=best_individual,
            optimal_objective=best_fitness,
            iterations=len(convergence_history),
            time_elapsed=0.0,
            convergence_history=convergence_history,
            gradient_history=[],
            message="Genetic algorithm completed",
            algorithm_used=OptimizationAlgorithm.GENETIC_ALGORITHM,
        )

    def _simulated_annealing(
        self,
        objective_function: ObjectiveFunction,
        initial_parameters: np.ndarray,
        bounds: List[Tuple[float, float]] = None,
    ) -> OptimizationResult:
        """Simulated annealing optimization."""
        current_solution = initial_parameters.copy()
        current_objective = objective_function.evaluate(current_solution)

        best_solution = current_solution.copy()
        best_objective = current_objective

        temperature = self.params.initial_temperature
        convergence_history = []

        for iteration in range(self.params.max_iterations):
            # Generate neighbor solution
            step_size = 0.1 * temperature / self.params.initial_temperature
            candidate = current_solution + np.random.normal(
                0, step_size, len(current_solution)
            )

            # Apply bounds
            if bounds:
                candidate = self._apply_bounds(candidate, bounds)

            # Evaluate candidate
            candidate_objective = objective_function.evaluate(candidate)

            # Accept or reject
            delta = candidate_objective - current_objective
            if delta < 0 or np.random.random() < np.exp(-delta / temperature):
                current_solution = candidate
                current_objective = candidate_objective

                # Update best
                if candidate_objective < best_objective:
                    best_solution = candidate.copy()
                    best_objective = candidate_objective

            # Cool down
            temperature *= self.params.cooling_rate
            temperature = max(temperature, self.params.min_temperature)

            convergence_history.append(best_objective)

            # Progress reporting
            if self.params.verbose and iteration % 100 == 0:
                logger.info(
                    f"Iteration {iteration}: Best = {best_objective:.6f}, "
                    f"Temperature = {temperature:.6f}"
                )

        return OptimizationResult(
            success=True,
            optimal_parameters=best_solution,
            optimal_objective=best_objective,
            iterations=len(convergence_history),
            time_elapsed=0.0,
            convergence_history=convergence_history,
            gradient_history=[],
            message="Simulated annealing completed",
            algorithm_used=OptimizationAlgorithm.SIMULATED_ANNEALING,
        )

    def _particle_swarm(
        self,
        objective_function: ObjectiveFunction,
        initial_parameters: np.ndarray,
        bounds: List[Tuple[float, float]] = None,
    ) -> OptimizationResult:
        """Particle swarm optimization."""
        if bounds is None:
            bounds = [(-10, 10) for _ in range(len(initial_parameters))]

        # Initialize swarm
        swarm_size = self.params.population_size
        n_params = len(initial_parameters)

        # Particle positions and velocities
        positions = np.random.uniform(
            [b[0] for b in bounds], [b[1] for b in bounds], (swarm_size, n_params)
        )
        velocities = np.random.uniform(-1, 1, (swarm_size, n_params))

        # Include initial parameters
        positions[0] = initial_parameters

        # Personal best positions and values
        personal_best_positions = positions.copy()
        personal_best_values = np.array(
            [objective_function.evaluate(p) for p in positions]
        )

        # Global best
        global_best_idx = np.argmin(personal_best_values)
        global_best_position = personal_best_positions[global_best_idx].copy()
        global_best_value = personal_best_values[global_best_idx]

        convergence_history = []
        w = self.params.w
        c1 = self.params.c1
        c2 = self.params.c2

        for iteration in range(self.params.max_iterations):
            for i in range(swarm_size):
                # Update velocity
                r1, r2 = np.random.random(n_params), np.random.random(n_params)
                velocities[i] = (
                    w * velocities[i]
                    + c1 * r1 * (personal_best_positions[i] - positions[i])
                    + c2 * r2 * (global_best_position - positions[i])
                )

                # Update position
                positions[i] += velocities[i]

                # Apply bounds
                positions[i] = self._apply_bounds(positions[i], bounds)

                # Evaluate new position
                current_value = objective_function.evaluate(positions[i])

                # Update personal best
                if current_value < personal_best_values[i]:
                    personal_best_values[i] = current_value
                    personal_best_positions[i] = positions[i].copy()

                    # Update global best
                    if current_value < global_best_value:
                        global_best_value = current_value
                        global_best_position = positions[i].copy()

            convergence_history.append(global_best_value)

            # Progress reporting
            if self.params.verbose and iteration % 50 == 0:
                logger.info(
                    f"Iteration {iteration}: Global best = {global_best_value:.6f}"
                )

        return OptimizationResult(
            success=True,
            optimal_parameters=global_best_position,
            optimal_objective=global_best_value,
            iterations=len(convergence_history),
            time_elapsed=0.0,
            convergence_history=convergence_history,
            gradient_history=[],
            message="Particle swarm optimization completed",
            algorithm_used=OptimizationAlgorithm.PARTICLE_SWARM,
        )

    def _apply_bounds(
        self, parameters: np.ndarray, bounds: List[Tuple[float, float]]
    ) -> np.ndarray:
        """Apply parameter bounds."""
        bounded_params = parameters.copy()
        for i, (lower, upper) in enumerate(bounds):
            bounded_params[i] = np.clip(bounded_params[i], lower, upper)
        return bounded_params

    def save_optimization_history(self, filepath: str):
        """Save optimization history to file."""
        history_data = {
            "convergence_history": self.history,
            "parameters": self.params.__dict__,
            "best_objective": self.best_objective,
            "best_solution": self.best_solution.tolist()
            if self.best_solution is not None
            else None,
        }

        with open(filepath, "w") as f:
            json.dump(history_data, f, indent=2)

        logger.info(f"Optimization history saved to {filepath}")

    def load_optimization_history(self, filepath: str):
        """Load optimization history from file."""
        with open(filepath, "r") as f:
            history_data = json.load(f)

        self.history = history_data["convergence_history"]
        self.best_objective = history_data["best_objective"]
        if history_data["best_solution"] is not None:
            self.best_solution = np.array(history_data["best_solution"])

        logger.info(f"Optimization history loaded from {filepath}")


# GPU-accelerated optimization functions
if HAS_PYTORCH:

    class TorchObjectiveFunction(nn.Module):
        """PyTorch-based objective function for GPU acceleration."""

        def __init__(self, dose_calculator, structures, objectives):
            super().__init__()
            self.dose_calculator = dose_calculator
            self.structures = structures
            self.objectives = objectives

        def forward(self, parameters):
            """Forward pass for objective evaluation."""
            # Convert parameters to dose distribution
            dose = self.dose_calculator(parameters)

            # Evaluate objectives
            total_loss = torch.tensor(0.0, device=parameters.device)

            for objective in self.objectives:
                loss = self._evaluate_objective_torch(objective, dose, parameters)
                total_loss += objective.weight * loss

            return total_loss

        def _evaluate_objective_torch(self, objective, dose, parameters):
            """Evaluate single objective using PyTorch tensors."""
            # Implementation depends on objective type
            return torch.tensor(0.0, device=parameters.device)


def create_advanced_optimizer(
    algorithm: OptimizationAlgorithm = OptimizationAlgorithm.ADAM,
    use_gpu: bool = False,
    **kwargs,
) -> AdvancedOptimizer:
    """
    Factory function to create advanced optimizer.

    Parameters
    ----------
    algorithm : OptimizationAlgorithm
        Optimization algorithm to use
    use_gpu : bool
        Whether to use GPU acceleration
    **kwargs
        Additional parameters for OptimizationParameters

    Returns
    -------
    AdvancedOptimizer
        Configured optimizer instance
    """
    params = OptimizationParameters(algorithm=algorithm, use_gpu=use_gpu, **kwargs)
    return AdvancedOptimizer(params)
