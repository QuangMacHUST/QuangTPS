#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module defining gradient descent optimization for radiotherapy treatment planning.

This module implements the Gradient Descent algorithm that can be used
for optimizing the treatment planning parameters and objectives.
"""

import numpy as np
import time
import logging
from typing import Dict, List, Tuple, Optional, Union, Any, Callable

# Define base classes if they don't exist
class OptimizationResult:
    """
    Class to store optimization results.
    """
    def __init__(self, 
                parameters: np.ndarray,
                objective_value: float,
                num_iterations: int,
                converged: bool,
                message: str = ""
                ):
        self.parameters = parameters
        self.objective_value = objective_value
        self.num_iterations = num_iterations
        self.converged = converged
        self.message = message
        self.execution_time = 0.0
        self.history = []

class Constraint:
    """Base class for optimization constraints."""
    def __init__(self, name: str = "", is_hard: bool = False):
        self.name = name
        self.is_hard = is_hard
    
    def evaluate(self, params: np.ndarray) -> bool:
        """Evaluate if constraint is satisfied."""
        return True

class OptimizationSolver:
    """
    Base class for optimization solvers.
    """
    def __init__(self, **kwargs):
        """Initialize the optimizer."""
        pass
    
    def initialize(self, params: Dict[str, Any]) -> bool:
        """Initialize the optimizer with parameters."""
        return True
    
    def optimize(self, 
                objective_function: Callable[[np.ndarray], float], 
                initial_parameters: np.ndarray,
                **kwargs) -> OptimizationResult:
        """
        Optimize the objective function.
        
        Parameters
        ----------
        objective_function : Callable
            Function to minimize
        initial_parameters : np.ndarray
            Starting point for optimization
        
        Returns
        -------
        OptimizationResult
            Results of the optimization
        """
        return OptimizationResult(
            initial_parameters,
            objective_function(initial_parameters),
            0,
            False,
            "Base optimizer does not implement optimize()"
        )

logger = logging.getLogger(__name__)

class GradientDescent(OptimizationSolver):
    """
    Gradient Descent optimization algorithm for treatment planning.
    
    This implementation includes momentum and adaptive learning rate options.
    """
    
    def __init__(self, 
                learning_rate: float = 0.01, 
                max_iterations: int = 1000, 
                convergence_threshold: float = 1e-6,
                momentum: float = 0.9,
                use_adaptive_rate: bool = True):
        """
        Initialize Gradient Descent optimizer.
        
        Parameters
        ----------
        learning_rate : float
            Initial learning rate
        max_iterations : int
            Maximum number of iterations
        convergence_threshold : float
            Threshold for convergence determination
        momentum : float
            Momentum coefficient (0 = no momentum)
        use_adaptive_rate : bool
            Whether to use adaptive learning rate
        """
        super().__init__()
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.momentum = momentum
        self.use_adaptive_rate = use_adaptive_rate
        self.velocity = None
        self.iteration_count = 0
        self.best_objective = float('inf')
        self.best_params = None
        
        logger.info(f"Initialized Gradient Descent optimizer with learning_rate={learning_rate}, "
                   f"max_iterations={max_iterations}, convergence_threshold={convergence_threshold}")
    
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
        if 'learning_rate' in params:
            self.learning_rate = params['learning_rate']
        if 'max_iterations' in params:
            self.max_iterations = params['max_iterations']
        if 'convergence_threshold' in params:
            self.convergence_threshold = params['convergence_threshold']
        if 'momentum' in params:
            self.momentum = params['momentum']
        if 'use_adaptive_rate' in params:
            self.use_adaptive_rate = params['use_adaptive_rate']
        
        self.velocity = None
        self.iteration_count = 0
        
        return True
    
    def optimize(self, 
                objective_function: Callable[[np.ndarray], float], 
                gradient_function: Callable[[np.ndarray], np.ndarray],
                initial_parameters: np.ndarray,
                constraints: List[Constraint] = None,
                callback: Callable[[int, np.ndarray, float], None] = None) -> OptimizationResult:
        """
        Optimize using gradient descent.
        
        Parameters
        ----------
        objective_function : Callable
            Function to minimize
        gradient_function : Callable
            Function to compute gradient of objective function
        initial_parameters : np.ndarray
            Starting point for optimization
        constraints : List[Constraint]
            List of constraints to apply
        callback : Callable
            Function to call after each iteration with (iteration, params, obj_value)
        
        Returns
        -------
        OptimizationResult
            Results of the optimization
        """
        start_time = time.time()
        
        # Initialize parameters and history
        params = initial_parameters.copy()
        self.velocity = np.zeros_like(params)
        objective_history = []
        param_history = []
        
        # Best so far
        best_params = params.copy()
        best_objective = objective_function(params)
        
        # Optimization loop
        converged = False
        message = "Maximum iterations reached"
        
        for iteration in range(self.max_iterations):
            self.iteration_count = iteration
            
            # Current objective value
            obj_value = objective_function(params)
            objective_history.append(obj_value)
            param_history.append(params.copy())
            
            # Call the callback if provided
            if callback:
                callback(iteration, params, obj_value)
            
            # Check for new best
            if obj_value < best_objective:
                best_objective = obj_value
                best_params = params.copy()
            
            # Compute gradient
            gradient = gradient_function(params)
            
            # Check for convergence
            if np.linalg.norm(gradient) < self.convergence_threshold:
                converged = True
                message = "Converged: gradient norm below threshold"
                break
            
            # Compute velocity with momentum
            self.velocity = self.momentum * self.velocity - self.learning_rate * gradient
            
            # Update parameters
            params = params + self.velocity
            
            # Apply constraints if any
            if constraints:
                for constraint in constraints:
                    if constraint.is_hard and not constraint.evaluate(params):
                        # Project back to feasible region or apply penalty
                        # This is a simple placeholder implementation
                        params = best_params.copy()
                        break
            
            # Check for convergence by objective values
            if iteration > 10:
                diff = abs(objective_history[-1] - objective_history[-2]) / (abs(objective_history[-2]) + 1e-10)
                if diff < self.convergence_threshold:
                    converged = True
                    message = "Converged: objective value change below threshold"
                    break
            
            # Adaptive learning rate (simple implementation)
            if self.use_adaptive_rate and iteration > 0:
                if objective_history[-1] > objective_history[-2]:
                    # Objective got worse, decrease learning rate
                    self.learning_rate *= 0.5
                else:
                    # Objective improved, carefully increase learning rate
                    self.learning_rate *= 1.05
        
        # Create result object
        result = OptimizationResult(
            best_params,
            best_objective,
            self.iteration_count + 1,
            converged,
            message
        )
        
        # Add additional information
        result.execution_time = time.time() - start_time
        result.history = {"objective": objective_history, "parameters": param_history}
        
        logger.info(f"Gradient descent optimization completed: {message}, "
                   f"iterations={self.iteration_count+1}, final_objective={best_objective:.6f}")
        
        return result
    
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get the current parameters of the optimizer.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary of optimizer parameters
        """
        return {
            "learning_rate": self.learning_rate,
            "max_iterations": self.max_iterations,
            "convergence_threshold": self.convergence_threshold,
            "momentum": self.momentum,
            "use_adaptive_rate": self.use_adaptive_rate,
            "iteration_count": self.iteration_count,
            "best_objective": self.best_objective if hasattr(self, "best_objective") else None
        }
    
    def __str__(self) -> str:
        """
        String representation of the optimizer.
        
        Returns
        -------
        str
            String representation
        """
        return (f"GradientDescent(learning_rate={self.learning_rate}, "
                f"max_iterations={self.max_iterations}, "
                f"convergence_threshold={self.convergence_threshold}, "
                f"momentum={self.momentum}, "
                f"use_adaptive_rate={self.use_adaptive_rate})") 