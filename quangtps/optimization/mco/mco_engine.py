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
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

from quangtps.core.types import Plan, DoseGrid, Treatment
from quangtps.planning.optimization import PlanOptimizer
from quangtps.optimization.objectives import Objective, ObjectiveResult
from quangtps.optimization.constraints import Constraint
from quangtps.optimization.optimizer_factory import create_optimizer
from quangtps.treatment.techniques.imrt import IMRTTreatment
from quangtps.treatment.techniques.vmat import VMATTreatment
from quangtps.core.logging import get_logger

logger = get_logger(__name__)


class ParetoSolution:
    """
    Class representing a single Pareto-optimal solution.
    
    A solution consists of:
    - Objective weights used to generate the solution
    - Resulting fluence/control points
    - Resulting dose distribution
    - Objective values
    - Any additional metadata
    """
    
    def __init__(self, 
                 weights: Dict[str, float], 
                 objective_values: Dict[str, float] = None,
                 fluence_map: np.ndarray = None,
                 control_points: List[Dict] = None,
                 dose_grid: DoseGrid = None,
                 metadata: Dict[str, Any] = None):
        """
        Initialize a Pareto solution.
        
        Args:
            weights: Dictionary mapping objective names to weights
            objective_values: Dictionary mapping objective names to achieved values
            fluence_map: Fluence map for IMRT plans (if applicable)
            control_points: Control points for VMAT plans (if applicable)
            dose_grid: Resulting dose distribution
            metadata: Additional metadata about the solution
        """
        self.weights = weights
        self.objective_values = objective_values or {}
        self.fluence_map = fluence_map
        self.control_points = control_points
        self.dose_grid = dose_grid
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict:
        """Convert the solution to a dictionary for serialization."""
        result = {
            'weights': self.weights,
            'objective_values': self.objective_values,
            'metadata': self.metadata
        }
        
        # We don't serialize the fluence map, control points, or dose grid
        # as these can be very large. Instead, we just note if they exist.
        result['has_fluence_map'] = self.fluence_map is not None
        result['has_control_points'] = self.control_points is not None
        result['has_dose_grid'] = self.dose_grid is not None
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ParetoSolution':
        """Create a solution from a dictionary representation."""
        # Note: This only loads the metadata, weights, and objective values
        # The fluence map, control points, and dose grid must be loaded separately
        return cls(
            weights=data.get('weights', {}),
            objective_values=data.get('objective_values', {}),
            metadata=data.get('metadata', {})
        )


class MCOEngine:
    """
    Engine for multi-criteria optimization.
    
    This class provides methods for generating Pareto-optimal plans,
    interpolating between plans, and navigating the Pareto surface.
    """
    
    def __init__(self, plan: Plan, objectives: Dict[str, Objective], constraints: List[Constraint] = None):
        """
        Initialize the MCO engine.
        
        Args:
            plan: Treatment plan to optimize
            objectives: Dictionary mapping objective names to Objective objects
            constraints: List of Constraint objects (optional)
        """
        self.plan = plan
        self.objectives = objectives
        self.constraints = constraints or []
        self.solutions: List[ParetoSolution] = []
        self.prepared = False
        self.optimizer = None
        
        # Store the original plan in case we need to reset
        self._original_plan = copy.deepcopy(plan)
    
    def prepare(self) -> bool:
        """
        Prepare the optimizer for MCO.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create an optimizer appropriate for this plan
            self.optimizer = create_optimizer(self.plan, list(self.objectives.values()), self.constraints)
            
            if not self.optimizer:
                logger.error("Failed to create optimizer")
                return False
            
            # Initialize the optimizer
            success = self.optimizer.initialize()
            if not success:
                logger.error("Failed to initialize optimizer")
                return False
            
            self.prepared = True
            return True
        except Exception as e:
            logger.error(f"Error preparing MCO engine: {e}", exc_info=True)
            return False
    
    def generate_anchor_plans(self, num_anchors: Optional[int] = None) -> List[ParetoSolution]:
        """
        Generate anchor plans, which optimize a single objective at a time.
        
        Args:
            num_anchors: Number of anchor plans to generate (if None, generates one per objective)
        
        Returns:
            List of ParetoSolution objects
        """
        if not self.prepared:
            logger.error("MCO engine not prepared")
            return []
        
        # Clear existing solutions
        self.solutions = []
        
        # Determine which objectives to use as anchors
        objective_names = list(self.objectives.keys())
        if num_anchors is not None and num_anchors < len(objective_names):
            # Select a subset of objectives if requested
            import random
            selected_names = random.sample(objective_names, num_anchors)
        else:
            # Use all objectives
            selected_names = objective_names
        
        # Generate a plan for each selected objective
        for name in selected_names:
            # Set weight to 1.0 for this objective, 0.0 for others
            weights = {n: 1.0 if n == name else 0.0 for n in objective_names}
            
            # Optimize the plan
            solution = self._optimize_with_weights(weights)
            
            if solution:
                self.solutions.append(solution)
                logger.info(f"Generated anchor plan for {name}")
            else:
                logger.warning(f"Failed to generate anchor plan for {name}")
        
        return self.solutions
    
    def generate_balanced_plan(self) -> Optional[ParetoSolution]:
        """
        Generate a balanced plan using equal weights for all objectives.
        
        Returns:
            ParetoSolution object if successful, None otherwise
        """
        if not self.prepared:
            logger.error("MCO engine not prepared")
            return None
        
        # Set equal weights for all objectives
        objective_names = list(self.objectives.keys())
        weight = 1.0 / len(objective_names)
        weights = {name: weight for name in objective_names}
        
        # Optimize the plan
        solution = self._optimize_with_weights(weights)
        
        if solution:
            self.solutions.append(solution)
            logger.info("Generated balanced plan")
            return solution
        else:
            logger.warning("Failed to generate balanced plan")
            return None
    
    def generate_pareto_surface(self, num_points: int = 10, method: str = 'weight_sampling') -> List[ParetoSolution]:
        """
        Generate a set of Pareto-optimal plans to form a Pareto surface.
        
        Args:
            num_points: Number of points to generate
            method: Method to use for generating points ('weight_sampling', 
                   'constraint_sampling', or 'normal_constraint')
        
        Returns:
            List of ParetoSolution objects
        """
        if not self.prepared:
            logger.error("MCO engine not prepared")
            return []
        
        # Generate points based on the chosen method
        if method == 'weight_sampling':
            return self._generate_by_weight_sampling(num_points)
        elif method == 'constraint_sampling':
            return self._generate_by_constraint_sampling(num_points)
        elif method == 'normal_constraint':
            return self._generate_by_normal_constraint(num_points)
        else:
            logger.error(f"Unknown method: {method}")
            return []
    
    def _generate_by_weight_sampling(self, num_points: int) -> List[ParetoSolution]:
        """
        Generate Pareto-optimal plans by sampling weights.
        
        Args:
            num_points: Number of points to generate
        
        Returns:
            List of ParetoSolution objects
        """
        objective_names = list(self.objectives.keys())
        n_objectives = len(objective_names)
        
        # Generate random weights
        for _ in range(num_points):
            # Generate random weights that sum to 1
            raw_weights = np.random.random(n_objectives)
            normalized_weights = raw_weights / np.sum(raw_weights)
            
            # Convert to dictionary
            weights = {name: float(normalized_weights[i]) for i, name in enumerate(objective_names)}
            
            # Check if we already have a solution with similar weights
            if self._is_similar_to_existing(weights):
                continue
            
            # Optimize the plan
            solution = self._optimize_with_weights(weights)
            
            if solution:
                self.solutions.append(solution)
                logger.info(f"Generated plan with weights: {weights}")
            else:
                logger.warning(f"Failed to generate plan with weights: {weights}")
        
        return self.solutions
    
    def _generate_by_constraint_sampling(self, num_points: int) -> List[ParetoSolution]:
        """
        Generate Pareto-optimal plans by constraint sampling.
        
        This method optimizes one objective while constraining others.
        
        Args:
            num_points: Number of points to generate
        
        Returns:
            List of ParetoSolution objects
        """
        # TODO: Implement constraint sampling method
        logger.warning("Constraint sampling not yet implemented, falling back to weight sampling")
        return self._generate_by_weight_sampling(num_points)
    
    def _generate_by_normal_constraint(self, num_points: int) -> List[ParetoSolution]:
        """
        Generate Pareto-optimal plans using the normal constraint method.
        
        This advanced method systematically explores the Pareto surface.
        
        Args:
            num_points: Number of points to generate
        
        Returns:
            List of ParetoSolution objects
        """
        # TODO: Implement normal constraint method
        logger.warning("Normal constraint method not yet implemented, falling back to weight sampling")
        return self._generate_by_weight_sampling(num_points)
    
    def _is_similar_to_existing(self, weights: Dict[str, float], threshold: float = 0.1) -> bool:
        """
        Check if a set of weights is similar to an existing solution.
        
        Args:
            weights: Dictionary mapping objective names to weights
            threshold: Similarity threshold (L2 norm)
        
        Returns:
            True if similar to an existing solution, False otherwise
        """
        objective_names = list(self.objectives.keys())
        
        # Convert weights to array for comparison
        weights_array = np.array([weights.get(name, 0.0) for name in objective_names])
        
        for solution in self.solutions:
            solution_weights = np.array([solution.weights.get(name, 0.0) for name in objective_names])
            
            # Calculate L2 norm (Euclidean distance)
            distance = np.linalg.norm(weights_array - solution_weights)
            
            if distance < threshold:
                return True
        
        return False
    
    def _optimize_with_weights(self, weights: Dict[str, float]) -> Optional[ParetoSolution]:
        """
        Optimize the plan with the given weights.
        
        Args:
            weights: Dictionary mapping objective names to weights
        
        Returns:
            ParetoSolution object if successful, None otherwise
        """
        if not self.prepared:
            logger.error("MCO engine not prepared")
            return None
        
        try:
            # Make a copy of the plan to avoid modifying the original
            current_plan = copy.deepcopy(self._original_plan)
            
            # Set the weights in the optimizer
            for name, weight in weights.items():
                if name in self.objectives:
                    self.optimizer.set_objective_weight(self.objectives[name], weight)
            
            # Record start time
            start_time = time.time()
            
            # Run the optimization
            result = self.optimizer.optimize()
            
            # Record end time
            end_time = time.time()
            optimization_time = end_time - start_time
            
            if not result.success:
                logger.warning(f"Optimization failed: {result.message}")
                return None
            
            # Extract results
            fluence_map = result.fluence_map if hasattr(result, 'fluence_map') else None
            control_points = result.control_points if hasattr(result, 'control_points') else None
            
            # Calculate the dose distribution
            dose_grid = self._calculate_dose(current_plan, fluence_map, control_points)
            
            # Evaluate all objectives on the result
            objective_values = self._evaluate_objectives(fluence_map, control_points, dose_grid)
            
            # Create metadata
            metadata = {
                'optimization_time': optimization_time,
                'iterations': result.iterations if hasattr(result, 'iterations') else None,
                'interpolated': False
            }
            
            # Create and return the solution
            solution = ParetoSolution(
                weights=weights,
                objective_values=objective_values,
                fluence_map=fluence_map,
                control_points=control_points,
                dose_grid=dose_grid,
                metadata=metadata
            )
            
            return solution
        except Exception as e:
            logger.error(f"Error optimizing with weights: {e}", exc_info=True)
            return None
    
    def _calculate_dose(self, plan: Plan, fluence_map: np.ndarray = None, 
                      control_points: List[Dict] = None) -> Optional[DoseGrid]:
        """
        Calculate the dose distribution for a plan.
        
        Args:
            plan: Treatment plan
            fluence_map: Fluence map for IMRT plans (if applicable)
            control_points: Control points for VMAT plans (if applicable)
        
        Returns:
            DoseGrid object if successful, None otherwise
        """
        try:
            # Update the treatment with the optimization results
            treatment = plan.get_treatment()
            
            if isinstance(treatment, IMRTTreatment) and fluence_map is not None:
                treatment.set_fluence_map(fluence_map)
            elif isinstance(treatment, VMATTreatment) and control_points is not None:
                treatment.set_control_points(control_points)
            
            # Calculate the dose
            dose_calc = treatment.get_dose_calculator()
            dose_grid = dose_calc.calculate()
            
            return dose_grid
        except Exception as e:
            logger.error(f"Error calculating dose: {e}", exc_info=True)
            return None
    
    def _evaluate_objectives(self, fluence_map: np.ndarray = None, 
                           control_points: List[Dict] = None,
                           dose_grid: DoseGrid = None) -> Dict[str, float]:
        """
        Evaluate all objectives on the given result.
        
        Args:
            fluence_map: Fluence map for IMRT plans (if applicable)
            control_points: Control points for VMAT plans (if applicable)
            dose_grid: Dose distribution (if already calculated)
        
        Returns:
            Dictionary mapping objective names to achieved values
        """
        objective_values = {}
        
        try:
            for name, objective in self.objectives.items():
                # Determine what to pass to the objective for evaluation
                if hasattr(objective, 'evaluate_dose') and dose_grid is not None:
                    result = objective.evaluate_dose(dose_grid)
                elif fluence_map is not None:
                    result = objective.evaluate(fluence_map)
                elif control_points is not None:
                    # Some objectives might have special methods for control points
                    if hasattr(objective, 'evaluate_control_points'):
                        result = objective.evaluate_control_points(control_points)
                    else:
                        continue
                else:
                    continue
                
                # Store the result
                if isinstance(result, ObjectiveResult):
                    objective_values[name] = result.value
                elif isinstance(result, (int, float)):
                    objective_values[name] = float(result)
        except Exception as e:
            logger.error(f"Error evaluating objectives: {e}", exc_info=True)
        
        return objective_values
    
    def navigate(self, slider_values: Dict[str, float]) -> Optional[ParetoSolution]:
        """
        Navigate to a new solution based on slider values.
        
        This method attempts to find an existing solution or interpolate between
        existing solutions to match the desired weights.
        
        Args:
            slider_values: Dictionary mapping objective names to desired weights
        
        Returns:
            ParetoSolution object if successful, None otherwise
        """
        if not self.solutions:
            logger.error("No solutions available for navigation")
            return None
        
        try:
            # Normalize the slider values
            total = sum(slider_values.values())
            if total == 0:
                logger.error("All slider values are zero")
                return None
            
            normalized_values = {name: weight / total for name, weight in slider_values.items()}
            
            # Check if we have an exact match
            for solution in self.solutions:
                if all(abs(solution.weights.get(name, 0.0) - normalized_values.get(name, 0.0)) < 0.01 
                       for name in self.objectives.keys()):
                    logger.info("Found exact match for weights")
                    return solution
            
            # Find the closest solution
            closest_solution, closest_distance = self._find_closest_solution(normalized_values)
            
            # If very close, just return the closest solution
            if closest_distance < 0.1:
                logger.info(f"Found close match for weights (distance: {closest_distance:.4f})")
                return closest_solution
            
            # Otherwise, try to interpolate
            interpolated_solution = self._interpolate_solution(normalized_values)
            
            if interpolated_solution:
                logger.info("Created interpolated solution")
                # Add the interpolated solution to our list
                self.solutions.append(interpolated_solution)
                return interpolated_solution
            
            # If interpolation fails, re-optimize with the desired weights
            logger.info("Interpolation failed, re-optimizing with desired weights")
            solution = self._optimize_with_weights(normalized_values)
            
            if solution:
                self.solutions.append(solution)
                return solution
            
            # If all else fails, return the closest solution
            logger.warning("Re-optimization failed, returning closest solution")
            return closest_solution
        except Exception as e:
            logger.error(f"Error navigating to new solution: {e}", exc_info=True)
            return None
    
    def _find_closest_solution(self, weights: Dict[str, float]) -> Tuple[ParetoSolution, float]:
        """
        Find the solution with the closest weights.
        
        Args:
            weights: Dictionary mapping objective names to weights
        
        Returns:
            Tuple of (closest solution, distance)
        """
        objective_names = list(self.objectives.keys())
        
        # Convert weights to array for comparison
        weights_array = np.array([weights.get(name, 0.0) for name in objective_names])
        
        closest_solution = None
        closest_distance = float('inf')
        
        for solution in self.solutions:
            solution_weights = np.array([solution.weights.get(name, 0.0) for name in objective_names])
            
            # Calculate L2 norm (Euclidean distance)
            distance = np.linalg.norm(weights_array - solution_weights)
            
            if distance < closest_distance:
                closest_distance = distance
                closest_solution = solution
        
        return closest_solution, closest_distance
    
    def _interpolate_solution(self, weights: Dict[str, float]) -> Optional[ParetoSolution]:
        """
        Interpolate a new solution based on existing solutions.
        
        This method uses Delaunay triangulation and barycentric interpolation
        to create a new solution.
        
        Args:
            weights: Dictionary mapping objective names to weights
        
        Returns:
            ParetoSolution object if successful, None otherwise
        """
        if len(self.solutions) < 3:
            # Need at least 3 points for triangulation
            return None
        
        try:
            from scipy.spatial import Delaunay
            from scipy.interpolate import LinearNDInterpolator
            
            objective_names = list(self.objectives.keys())
            
            # Extract weight vectors and results from solutions
            points = []
            fluence_values = []
            control_point_values = []
            has_fluence = False
            has_control_points = False
            
            for solution in self.solutions:
                # Extract weights
                point = [solution.weights.get(name, 0.0) for name in objective_names]
                points.append(point)
                
                # Extract fluence or control points
                if solution.fluence_map is not None:
                    has_fluence = True
                    fluence_values.append(solution.fluence_map.flatten())
                elif solution.control_points is not None:
                    has_control_points = True
                    # Convert control points to a flat array for interpolation
                    control_point_array = self._control_points_to_array(solution.control_points)
                    control_point_values.append(control_point_array)
            
            # Convert to numpy arrays
            points = np.array(points)
            
            # Create target point
            target = np.array([weights.get(name, 0.0) for name in objective_names])
            
            # Check if target point is inside the convex hull
            try:
                tri = Delaunay(points)
                simplex_index = tri.find_simplex(target)
                if simplex_index == -1:
                    # Target is outside the convex hull
                    return None
            except Exception as e:
                logger.warning(f"Delaunay triangulation failed: {e}")
                return None
            
            # Perform interpolation
            if has_fluence:
                fluence_values = np.array(fluence_values)
                fluence_interp = LinearNDInterpolator(points, fluence_values)
                interpolated_fluence_flat = fluence_interp(target)
                
                if interpolated_fluence_flat is None or np.any(np.isnan(interpolated_fluence_flat)):
                    return None
                
                # Reshape back to original shape
                fluence_shape = self.solutions[0].fluence_map.shape
                interpolated_fluence = interpolated_fluence_flat.reshape(fluence_shape)
                
                # Create the interpolated solution
                solution = self._create_interpolated_solution(weights, fluence_map=interpolated_fluence)
                return solution
            elif has_control_points:
                control_point_values = np.array(control_point_values)
                cp_interp = LinearNDInterpolator(points, control_point_values)
                interpolated_cp_flat = cp_interp(target)
                
                if interpolated_cp_flat is None or np.any(np.isnan(interpolated_cp_flat)):
                    return None
                
                # Convert back to control points
                interpolated_control_points = self._array_to_control_points(
                    interpolated_cp_flat, self.solutions[0].control_points
                )
                
                # Create the interpolated solution
                solution = self._create_interpolated_solution(weights, control_points=interpolated_control_points)
                return solution
            
            return None
        except Exception as e:
            logger.error(f"Error interpolating solution: {e}", exc_info=True)
            return None
    
    def _control_points_to_array(self, control_points: List[Dict]) -> np.ndarray:
        """
        Convert control points to a flat array for interpolation.
        
        Args:
            control_points: List of control point dictionaries
        
        Returns:
            Flat numpy array
        """
        # This is a simplified implementation that assumes control points
        # have a consistent structure. In practice, this would need to be
        # more robust.
        result = []
        
        for cp in control_points:
            # Extract leaf positions and MU weight
            if 'leaf_positions' in cp:
                result.extend(cp['leaf_positions'])
            if 'mu_weight' in cp:
                result.append(cp['mu_weight'])
            if 'gantry_angle' in cp:
                result.append(cp['gantry_angle'])
            if 'collimator_angle' in cp:
                result.append(cp['collimator_angle'])
        
        return np.array(result)
    
    def _array_to_control_points(self, array: np.ndarray, 
                               template: List[Dict]) -> List[Dict]:
        """
        Convert a flat array back to control points.
        
        Args:
            array: Flat numpy array
            template: Template control points for structure
        
        Returns:
            List of control point dictionaries
        """
        # This is a simplified implementation that assumes control points
        # have a consistent structure. In practice, this would need to be
        # more robust.
        result = []
        index = 0
        
        for template_cp in template:
            cp = {}
            
            # Extract leaf positions
            if 'leaf_positions' in template_cp:
                num_leaves = len(template_cp['leaf_positions'])
                cp['leaf_positions'] = array[index:index+num_leaves].tolist()
                index += num_leaves
            
            # Extract MU weight
            if 'mu_weight' in template_cp:
                cp['mu_weight'] = array[index]
                index += 1
            
            # Extract gantry angle
            if 'gantry_angle' in template_cp:
                cp['gantry_angle'] = array[index]
                index += 1
            
            # Extract collimator angle
            if 'collimator_angle' in template_cp:
                cp['collimator_angle'] = array[index]
                index += 1
            
            # Copy any other fields
            for key, value in template_cp.items():
                if key not in cp:
                    cp[key] = value
            
            result.append(cp)
        
        return result
    
    def _create_interpolated_solution(self, weights: Dict[str, float],
                                    fluence_map: np.ndarray = None,
                                    control_points: List[Dict] = None) -> Optional[ParetoSolution]:
        """
        Create an interpolated solution.
        
        Args:
            weights: Dictionary mapping objective names to weights
            fluence_map: Interpolated fluence map (if applicable)
            control_points: Interpolated control points (if applicable)
        
        Returns:
            ParetoSolution object if successful, None otherwise
        """
        try:
            # Make a copy of the plan
            current_plan = copy.deepcopy(self._original_plan)
            
            # Calculate the dose
            dose_grid = self._calculate_dose(current_plan, fluence_map, control_points)
            
            if dose_grid is None:
                return None
            
            # Evaluate objectives
            objective_values = self._evaluate_objectives(fluence_map, control_points, dose_grid)
            
            # Create metadata
            metadata = {
                'interpolated': True
            }
            
            # Create and return the solution
            solution = ParetoSolution(
                weights=weights,
                objective_values=objective_values,
                fluence_map=fluence_map,
                control_points=control_points,
                dose_grid=dose_grid,
                metadata=metadata
            )
            
            return solution
        except Exception as e:
            logger.error(f"Error creating interpolated solution: {e}", exc_info=True)
            return None
    
    def accept_current_solution(self) -> Optional[Plan]:
        """
        Accept the current solution and apply it to the plan.
        
        Returns:
            Updated Plan object if successful, None otherwise
        """
        # Find the current solution (the most recently added one)
        if not self.solutions:
            logger.error("No solutions available")
            return None
        
        current_solution = self.solutions[-1]
        
        try:
            # Make a copy of the original plan
            updated_plan = copy.deepcopy(self._original_plan)
            
            # Update the treatment with the solution
            treatment = updated_plan.get_treatment()
            
            if isinstance(treatment, IMRTTreatment) and current_solution.fluence_map is not None:
                treatment.set_fluence_map(current_solution.fluence_map)
            elif isinstance(treatment, VMATTreatment) and current_solution.control_points is not None:
                treatment.set_control_points(current_solution.control_points)
            
            # Calculate and store the dose
            if current_solution.dose_grid is not None:
                updated_plan.set_dose_grid(current_solution.dose_grid)
            else:
                # Calculate the dose
                dose_calc = treatment.get_dose_calculator()
                dose_grid = dose_calc.calculate()
                updated_plan.set_dose_grid(dose_grid)
            
            # Store the weights and objective values as metadata
            metadata = updated_plan.get_metadata() or {}
            metadata['mco_weights'] = current_solution.weights
            metadata['mco_objective_values'] = current_solution.objective_values
            metadata['mco_solution_metadata'] = current_solution.metadata
            updated_plan.set_metadata(metadata)
            
            return updated_plan
        except Exception as e:
            logger.error(f"Error accepting solution: {e}", exc_info=True)
            return None
    
    def save_solutions(self, filename: str) -> bool:
        """
        Save the solutions to a file.
        
        Args:
            filename: Path to save to
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Convert solutions to dictionaries
            solution_dicts = [solution.to_dict() for solution in self.solutions]
            
            # Create parent directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
            
            # Save to file
            with open(filename, 'w') as f:
                json.dump({
                    'version': 1,
                    'objective_names': list(self.objectives.keys()),
                    'plan_id': self.plan.id if hasattr(self.plan, 'id') else None,
                    'solutions': solution_dicts
                }, f, indent=2)
            
            return True
        except Exception as e:
            logger.error(f"Error saving solutions: {e}", exc_info=True)
            return False
    
    def load_solutions(self, filename: str) -> bool:
        """
        Load solutions from a file.
        
        Args:
            filename: Path to load from
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load from file
            with open(filename, 'r') as f:
                data = json.load(f)
            
            # Check version
            version = data.get('version', 0)
            if version != 1:
                logger.error(f"Unsupported version: {version}")
                return False
            
            # Check plan ID
            plan_id = data.get('plan_id')
            if plan_id is not None and hasattr(self.plan, 'id') and plan_id != self.plan.id:
                logger.warning(f"Plan ID mismatch: {plan_id} != {self.plan.id}")
            
            # Load solutions
            self.solutions = [ParetoSolution.from_dict(s) for s in data.get('solutions', [])]
            
            return True
        except Exception as e:
            logger.error(f"Error loading solutions: {e}", exc_info=True)
            return False
    
    def plot_pareto_front(self, x_objective: str, y_objective: str, 
                        return_fig: bool = False) -> Optional[plt.Figure]:
        """
        Plot the Pareto front for two objectives.
        
        Args:
            x_objective: Name of objective for x-axis
            y_objective: Name of objective for y-axis
            return_fig: If True, return the matplotlib Figure object
        
        Returns:
            matplotlib Figure object if return_fig is True, None otherwise
        """
        if not self.solutions:
            logger.error("No solutions available")
            return None
        
        try:
            # Extract values
            x_values = []
            y_values = []
            
            for solution in self.solutions:
                if x_objective in solution.objective_values and y_objective in solution.objective_values:
                    x_values.append(solution.objective_values[x_objective])
                    y_values.append(solution.objective_values[y_objective])
            
            if not x_values:
                logger.error(f"No values found for objectives {x_objective} and {y_objective}")
                return None
            
            # Create the plot
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Plot points
            ax.scatter(x_values, y_values, c='blue', s=50)
            
            # Add labels and title
            x_label = x_objective.replace("_", " ").title()
            y_label = y_objective.replace("_", " ").title()
            ax.set_xlabel(f"{x_label}")
            ax.set_ylabel(f"{y_label}")
            ax.set_title(f"Pareto Front: {x_label} vs {y_label}")
            ax.grid(True)
            
            if return_fig:
                return fig
            else:
                plt.show()
                return None
        except Exception as e:
            logger.error(f"Error plotting Pareto front: {e}", exc_info=True)
            return None
    
    def reset(self):
        """Reset the MCO engine, clearing all solutions."""
        self.solutions = []
        self.prepared = False
        self.optimizer = None
        self.plan = copy.deepcopy(self._original_plan)


def create_mco_engine(plan: Plan, objectives: Dict[str, Objective], 
                     constraints: List[Constraint] = None) -> MCOEngine:
    """
    Create an MCO engine for the given plan.
    
    Args:
        plan: Treatment plan to optimize
        objectives: Dictionary mapping objective names to Objective objects
        constraints: List of Constraint objects (optional)
    
    Returns:
        MCOEngine object
    """
    return MCOEngine(plan, objectives, constraints)


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
        'ptv_coverage': DummyObjective('ptv_coverage'),
        'oar_sparing': DummyObjective('oar_sparing')
    }
    
    engine = MCOEngine(plan, objectives)
    
    # Test saving and loading
    engine.solutions = [
        ParetoSolution({'ptv_coverage': 1.0, 'oar_sparing': 0.0}, {'ptv_coverage': 0.9, 'oar_sparing': 0.5}),
        ParetoSolution({'ptv_coverage': 0.0, 'oar_sparing': 1.0}, {'ptv_coverage': 0.7, 'oar_sparing': 0.9})
    ]
    
    engine.save_solutions("test_solutions.json")
    print("Saved solutions")
    
    engine.solutions = []
    engine.load_solutions("test_solutions.json")
    print(f"Loaded {len(engine.solutions)} solutions") 