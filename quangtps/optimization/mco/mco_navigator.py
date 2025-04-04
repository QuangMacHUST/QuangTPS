import logging
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Union
import uuid
import copy
import time

from quangtps.optimization.mco.trade_off import TradeOffExplorer
from quangtps.optimization.mco.pareto_surface import ParetoSurface, ParetoSolution
from quangtps.optimization.objectives import ObjectiveFunction, ObjectiveType, Objective
from quangtps.planning.plan import Plan
from quangtps.optimization.optimizer import PlanOptimizer, Optimizer
from quangtps.dose.dose_calculation import DoseCalculator
from quangtps.core.types import Structure
from quangtps.optimization.constraints import Constraint
from quangtps.core.logging import get_logger

logger = get_logger(__name__)

class MCONavigator:
    """
    Multi-Criteria Optimization Navigator.
    
    This class manages the MCO workflow by:
    1. Generating Pareto-optimal plans
    2. Allowing navigation between plans
    3. Providing interpolation between plans
    4. Supporting slider-based exploration of the Pareto surface
    
    The MCO workflow follows these steps:
    1. Define objectives and constraints
    2. Generate Pareto-optimal plans
    3. Navigate the Pareto surface using sliders
    4. Select the final plan
    """
    
    def __init__(self, plan: Plan):
        """
        Initialize the MCO Navigator.
        
        Args:
            plan: The base plan for MCO exploration
        """
        self.base_plan = plan
        self.objectives: List[ObjectiveFunction] = []
        self.constraints: List[Constraint] = []
        self.solutions: List[ParetoSolution] = []
        self.pareto_surface = ParetoSurface()
        self.trade_off_explorer = TradeOffExplorer()
        self.current_weights: Dict[int, float] = {}
        self.current_solution: Optional[ParetoSolution] = None
        self.current_solution_index: int = -1
        self.optimizer = PlanOptimizer()
        self.dose_calculator = DoseCalculator()
        
        # Tracking the generation process
        self.generation_in_progress = False
        self.generation_progress = 0.0
        self.generation_message = ""
        
        # Caching the interpolated plans
        self.interpolated_plans: Dict[str, Plan] = {}
        
        logger.info("MCO Navigator initialized")
    
    def set_objectives(self, objectives: List[ObjectiveFunction]):
        """
        Set the objectives for MCO.
        
        Args:
            objectives: List of objective functions
        """
        self.objectives = objectives
        self.trade_off_explorer.set_objectives(objectives)
        logger.info(f"Set {len(objectives)} objectives for MCO")
    
    def generate_pareto_plans(self, num_plans: int = 10) -> bool:
        """
        Generate Pareto-optimal plans.
        
        Args:
            num_plans: Number of Pareto-optimal plans to generate
            
        Returns:
            True if generation was successful, False otherwise
        """
        if self.generation_in_progress:
            logger.warning("Pareto plan generation already in progress")
            return False
        
        self.generation_in_progress = True
        self.generation_progress = 0.0
        self.generation_message = "Initializing..."
        
        try:
            logger.info(f"Generating {num_plans} Pareto-optimal plans")
            
            # Clear existing solutions
            self.solutions = []
            
            # Generate the base solution first
            base_solution = self._optimize_base_plan()
            if base_solution:
                self.solutions.append(base_solution)
                self.current_solution = base_solution
            
            # Generate alternative solutions
            self.generation_message = "Generating alternative plans..."
            
            for i in range(1, num_plans):
                # Update progress
                self.generation_progress = i / num_plans
                
                # Generate a new plan with different objective weights
                weight_factor = i / (num_plans - 1)
                solution = self._generate_alternative_solution(weight_factor)
                
                if solution:
                    self.solutions.append(solution)
            
            # Build the Pareto surface from the solutions
            self.pareto_surface.build_from_solutions(self.solutions)
            
            # Analyze trade-offs
            self.trade_off_explorer.analyze_trade_offs([sol.plan for sol in self.solutions])
            
            # Set first solution as current
            if self.solutions:
                self.current_solution_index = 0
                self.current_solution = self.solutions[0]
            
            # Reset navigation weights
            self._reset_navigation_weights()
            
            self.generation_in_progress = False
            self.generation_progress = 1.0
            
            logger.info(f"Successfully generated {len(self.solutions)} Pareto-optimal plans")
            return True
            
        except Exception as e:
            logger.error(f"Error generating Pareto-optimal plans: {str(e)}")
            self.generation_in_progress = False
            return False
            
        finally:
            self.generation_in_progress = False
            self.generation_progress = 1.0
            self.generation_message = "Complete"
    
    def _optimize_base_plan(self) -> Optional[ParetoSolution]:
        """
        Optimize the base plan to create a Pareto-optimal solution.
        
        Returns:
            A ParetoSolution if successful, None otherwise
        """
        try:
            # Create a copy of the base plan
            optimized_plan = self.base_plan.create_copy("MCO_Base")
            
            # Run optimization with default weights
            self.optimizer.set_plan(optimized_plan)
            self.optimizer.set_objectives(self.objectives)
            optimization_result = self.optimizer.optimize()
            
            if not optimization_result:
                logger.error("Base plan optimization failed")
                return None
            
            # Calculate dose
            self.dose_calculator.calculate(optimized_plan)
            
            # Evaluate objectives
            objective_values = {}
            for obj in self.objectives:
                value = obj.evaluate(optimized_plan)
                objective_values[obj.name] = value
            
            # Create a Pareto solution
            solution = ParetoSolution(
                plan=optimized_plan,
                objective_values=objective_values,
                weight_vector=np.ones(len(self.objectives)) / len(self.objectives)
            )
            
            return solution
            
        except Exception as e:
            logger.error(f"Error optimizing base plan: {str(e)}")
            return None
    
    def _generate_alternative_solution(self, weight_factor: float) -> Optional[ParetoSolution]:
        """
        Generate an alternative Pareto-optimal solution.
        
        Args:
            weight_factor: Factor to adjust weights (0-1)
            
        Returns:
            A ParetoSolution if successful, None otherwise
        """
        try:
            # Create a copy of the base plan
            plan_name = f"MCO_Alt_{int(weight_factor * 100)}"
            alt_plan = self.base_plan.create_copy(plan_name)
            
            # Generate alternative weights
            weight_vector = self._generate_weights(weight_factor)
            
            # Set weighted objectives
            weighted_objectives = []
            for i, obj in enumerate(self.objectives):
                weighted_obj = obj.clone()
                weighted_obj.weight = weight_vector[i]
                weighted_objectives.append(weighted_obj)
            
            # Run optimization
            self.optimizer.set_plan(alt_plan)
            self.optimizer.set_objectives(weighted_objectives)
            optimization_result = self.optimizer.optimize()
            
            if not optimization_result:
                logger.error(f"Alternative plan optimization failed for {plan_name}")
                return None
            
            # Calculate dose
            self.dose_calculator.calculate(alt_plan)
            
            # Evaluate objectives
            objective_values = {}
            for obj in self.objectives:
                value = obj.evaluate(alt_plan)
                objective_values[obj.name] = value
            
            # Create a Pareto solution
            solution = ParetoSolution(
                plan=alt_plan,
                objective_values=objective_values,
                weight_vector=weight_vector
            )
            
            return solution
            
        except Exception as e:
            logger.error(f"Error generating alternative solution: {str(e)}")
            return None
    
    def _generate_weights(self, factor: float) -> np.ndarray:
        """
        Generate a set of weights for objectives.
        
        Args:
            factor: A factor (0-1) to adjust the weights
            
        Returns:
            Array of weights for each objective
        """
        n_objectives = len(self.objectives)
        
        # Create varied weights
        weights = np.ones(n_objectives)
        
        # Adjust weights based on factor
        for i in range(n_objectives):
            phase = (i / n_objectives) * 2 * np.pi
            weights[i] = 1.0 + np.sin(phase + factor * 2 * np.pi) * 0.8
        
        # Normalize weights to sum to 1
        weights = weights / np.sum(weights)
        
        return weights
    
    def interpolate(self, weight_coefficients: Dict[int, float]) -> Optional[Plan]:
        """
        Interpolate between Pareto-optimal plans.
        
        Args:
            weight_coefficients: Dictionary mapping solution indices to weights
                                (should sum to 1.0)
            
        Returns:
            Interpolated plan if successful, None otherwise
        """
        if not self.solutions:
            logger.error("No Pareto-optimal solutions available for interpolation")
            return None
        
        # Check if weights sum to approximately 1.0
        weight_sum = sum(weight_coefficients.values())
        if abs(weight_sum - 1.0) > 1e-5:
            logger.error(f"Weight coefficients must sum to 1.0, got {weight_sum}")
            return None
        
        # Create a hash for the interpolation to use as cache key
        weights_hash = self._hash_weights(weight_coefficients)
        
        # Check if we've already computed this interpolation
        if weights_hash in self.interpolated_plans:
            return self.interpolated_plans[weights_hash]
        
        try:
            # Create a new interpolated plan
            plan_name = f"MCO_Interpolated_{weights_hash[:8]}"
            interpolated_plan = self.base_plan.create_copy(plan_name)
            
            # Get fluence maps from each solution and interpolate
            for fluence_map in interpolated_plan.get_fluence_maps():
                # Reset fluence to zero
                fluence_map.set_zero()
                
                # Interpolate fluence from each solution
                for sol_idx, weight in weight_coefficients.items():
                    if sol_idx < 0 or sol_idx >= len(self.solutions):
                        logger.error(f"Invalid solution index: {sol_idx}")
                        continue
                    
                    # Get corresponding fluence map from the solution
                    solution_plan = self.solutions[sol_idx].plan
                    solution_fluence = solution_plan.get_fluence_map_by_id(fluence_map.id)
                    
                    if solution_fluence:
                        # Add weighted contribution
                        fluence_map.add_weighted(solution_fluence, weight)
            
            # Calculate dose for the interpolated plan
            self.dose_calculator.calculate(interpolated_plan)
            
            # Cache the interpolated plan
            self.interpolated_plans[weights_hash] = interpolated_plan
            
            # Update current solution
            interpolated_objective_values = {}
            for obj in self.objectives:
                value = obj.evaluate(interpolated_plan)
                interpolated_objective_values[obj.name] = value
            
            # Create a ParetoSolution for the interpolated plan
            self.current_solution = ParetoSolution(
                plan=interpolated_plan,
                objective_values=interpolated_objective_values,
                weight_vector=np.array(list(weight_coefficients.values()))
            )
            
            return interpolated_plan
            
        except Exception as e:
            logger.error(f"Error interpolating plans: {str(e)}")
            return None
    
    def _hash_weights(self, weights: Dict[int, float]) -> str:
        """
        Create a hash string from weight coefficients.
        
        Args:
            weights: Dictionary mapping solution indices to weights
            
        Returns:
            Hash string representing the weights
        """
        weight_str = "_".join([f"{idx}_{w:.3f}" for idx, w in sorted(weights.items())])
        return weight_str
    
    def get_objective_range(self, objective_name: str) -> Tuple[float, float]:
        """
        Get the range of values for an objective across all Pareto solutions.
        
        Args:
            objective_name: Name of the objective
            
        Returns:
            Tuple of (min_value, max_value)
        """
        if not self.solutions:
            return (0.0, 0.0)
        
        values = [s.objective_values.get(objective_name, 0.0) for s in self.solutions]
        return (min(values), max(values))
    
    def apply_current_solution(self) -> bool:
        """
        Apply the current solution to the base plan.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.current_solution:
            logger.error("No current solution to apply")
            return False
        
        try:
            # Create a new plan based on the current solution
            final_plan = self.current_solution.plan.create_copy("MCO_Final")
            
            # Set as the new base plan
            self.base_plan = final_plan
            
            return True
            
        except Exception as e:
            logger.error(f"Error applying current solution: {str(e)}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the MCO navigator.
        
        Returns:
            Dictionary with status information
        """
        return {
            'generation_in_progress': self.generation_in_progress,
            'generation_progress': self.generation_progress,
            'generation_message': self.generation_message,
            'num_solutions': len(self.solutions),
            'has_current_solution': self.current_solution is not None,
            'current_solution_index': self.current_solution_index,
            'objectives': [obj.name for obj in self.objectives] if hasattr(self, 'objectives') else []
        }
    
    def get_objective_ids(self) -> List[str]:
        """Get IDs of all objectives."""
        return [obj.id for obj in self.objectives]
        
    def get_objective_names(self) -> List[str]:
        """Get names of all objectives."""
        return [obj.name for obj in self.objectives]
        
    def clear_solutions(self):
        """Clear all generated solutions and reset state."""
        self.solutions = []
        self.current_solution = None
        self.current_solution_index = -1
        self.current_weights = {}
        self.interpolated_plans = {}
        
    def _reset_navigation_weights(self):
        """Reset navigation weights to initial state."""
        self.current_weights = {}
        
        # If we have solutions, set up equal weights
        if not self.pareto_surface.is_empty():
            num_solutions = len(self.pareto_surface.solutions)
            if num_solutions > 0:
                weight = 1.0 / num_solutions
                self.current_weights = {i: weight for i in range(num_solutions)} 