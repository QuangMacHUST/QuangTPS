#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module for Volumetric Modulated Arc Therapy (VMAT) technique.

This module provides classes and methods to define and manage VMAT plans, which 
deliver radiation by rotating the gantry in an arc while continuously changing 
the shape of the radiation beam and dose rate.
"""

import uuid
import logging
import numpy as np
from typing import List, Dict, Any, Optional

from quangtps.treatment.mlc.mlc_model import MLCModel
from quangtps.treatment.techniques.technique_interface import BaseTreatmentTechnique, TechniqueCategory

logger = logging.getLogger(__name__)

class VMAT(BaseTreatmentTechnique):
    """
    Class representing a Volumetric Modulated Arc Therapy (VMAT) plan.
    
    VMAT is an advanced form of intensity-modulated radiation therapy (IMRT) that 
    delivers radiation by rotating the gantry in an arc while simultaneously varying 
    three parameters:
    
    1. The shape of the radiation beam using multi-leaf collimators (MLCs)
    2. The dose rate
    3. The speed of rotation
    
    This allows for highly conformal dose distributions with improved target coverage 
    and normal tissue sparing compared to conventional IMRT, typically with shorter 
    treatment times.
    """
    
    def __init__(self, plan_name: str, plan_id: Optional[str] = None):
        """
        Initialize a VMAT plan.
        
        Parameters
        ----------
        plan_name : str
            Name of the VMAT plan
        plan_id : str, optional
            Unique ID for the plan, if None, a UUID will be generated
        """
        super().__init__(name=plan_name, technique_id=plan_id, category=TechniqueCategory.ADVANCED)
        
        # Arc parameters
        self.arcs = []  # List of arc definitions
        self.control_points = {}  # Control points for each arc
        
        # Optimization parameters
        self.optimization_iterations = 100
        self.convergence_threshold = 0.001
        self.smoothing_factor = 0.5
        
        # Plan parameters
        self.mlc_model = None
        self.dose_objectives = []
        self.constraints = []
        
        # Using lazy % formatting for logging
        logger.info(
            "Initialized VMAT plan '%s' (ID: %s)",
            self.name, self.technique_id
        )
    
    def add_arc(self, arc_name: str, start_angle: float, stop_angle: float, 
               rotation_direction: str, energy: str = "6X", dose_rate: float = 600.0):
        """
        Add an arc to the VMAT plan.
        
        Parameters
        ----------
        arc_name : str
            Name of the arc
        start_angle : float
            Starting angle of the arc in degrees
        stop_angle : float
            Stopping angle of the arc in degrees
        rotation_direction : str
            Direction of rotation ('CW' for clockwise, 'CCW' for counter-clockwise)
        energy : str, optional
            Energy of the beam, default is "6X"
        dose_rate : float, optional
            Dose rate in MU/min, default is 600.0
        
        Returns
        -------
        str
            Unique ID for the created arc
        """
        arc_id = str(uuid.uuid4())
        
        arc = {
            "id": arc_id,
            "name": arc_name,
            "start_angle": start_angle,
            "stop_angle": stop_angle,
            "rotation_direction": rotation_direction,
            "energy": energy,
            "dose_rate": dose_rate
        }
        
        self.arcs.append(arc)
        self.control_points[arc_id] = []  # Initialize empty control points list
        
        # Using lazy % formatting for logging
        logger.info(
            "Added arc '%s' to VMAT plan '%s': %s° to %s° %s, energy=%s, dose_rate=%.1f MU/min",
            arc_name, self.name, start_angle, stop_angle, rotation_direction, energy, dose_rate
        )
        
        return arc_id
    
    def set_mlc_model(self, mlc_model: MLCModel):
        """
        Set the MLC model for the VMAT plan.
        
        Parameters
        ----------
        mlc_model : MLCModel
            Multi-leaf collimator model to use
        """
        self.mlc_model = mlc_model
        
        # Using lazy % formatting for logging
        logger.info(
            "Set MLC model for VMAT plan '%s': %s",
            self.name, mlc_model.name
        )
    
    def add_control_point(self, arc_id: str, gantry_angle: float, 
                         mlc_positions: List[List[float]], cumulative_meterset: float):
        """
        Add a control point to an arc.
        
        Parameters
        ----------
        arc_id : str
            ID of the arc to add the control point to
        gantry_angle : float
            Gantry angle in degrees for this control point
        mlc_positions : List[List[float]]
            MLC leaf positions as [[leaf1A, leaf1B], [leaf2A, leaf2B], ...] where A and B are banks
        cumulative_meterset : float
            Cumulative meterset weight (0.0 to 1.0)
        
        Returns
        -------
        int
            Index of the control point
        """
        if arc_id not in self.control_points:
            # Using lazy % formatting for logging
            logger.warning(
                "Cannot add control point: Arc ID '%s' not found in VMAT plan '%s'",
                arc_id, self.name
            )
            return -1
        
        control_point = {
            "index": len(self.control_points[arc_id]),
            "gantry_angle": gantry_angle,
            "mlc_positions": mlc_positions,
            "cumulative_meterset": cumulative_meterset
        }
        
        self.control_points[arc_id].append(control_point)
        
        # Using lazy % formatting for logging
        logger.info(
            "Added control point #%d to arc '%s' in VMAT plan '%s': gantry_angle=%.1f°, meterset=%.3f",
            control_point["index"], arc_id, self.name, gantry_angle, cumulative_meterset
        )
        
        return control_point["index"]
    
    def add_objective(self, structure_name: str, objective_type: str, 
                     dose: float, volume: Optional[float] = None, weight: float = 1.0):
        """
        Add an optimization objective for the VMAT plan.
        
        Parameters
        ----------
        structure_name : str
            Name of the structure
        objective_type : str
            Type of objective (min_dose, max_dose, min_dvh, max_dvh, uniform_dose)
        dose : float
            Dose value (Gy or %)
        volume : float, optional
            Volume value (%) for DVH constraints
        weight : float, optional
            Weight of the objective, default is 1.0
        """
        objective = {
            'structure': structure_name,
            'type': objective_type,
            'dose': dose,
            'volume': volume,
            'weight': weight
        }
        
        self.dose_objectives.append(objective)
        
        # Determine volume info for logging
        volume_info = f", volume={volume}%" if volume is not None else ""
        
        # Using lazy % formatting for logging
        logger.info(
            "Added optimization objective for structure '%s' in VMAT plan '%s': type=%s, dose=%.2f Gy%s, weight=%.2f",
            structure_name, self.name, objective_type, dose, volume_info, weight
        )
    
    def add_constraint(self, structure_name: str, constraint_type: str, 
                      dose: float, volume: Optional[float] = None):
        """
        Add a constraint to the VMAT plan.
        
        Parameters
        ----------
        structure_name : str
            Name of the structure
        constraint_type : str
            Type of constraint (max_dose, max_dvh, mean_dose)
        dose : float
            Dose value (Gy or %)
        volume : float, optional
            Volume value (%) for DVH constraints
        """
        constraint = {
            'structure': structure_name,
            'type': constraint_type,
            'dose': dose,
            'volume': volume
        }
        
        self.constraints.append(constraint)
        
        # Determine volume info for logging
        volume_info = f", volume={volume}%" if volume is not None else ""
        
        # Using lazy % formatting for logging
        logger.info(
            "Added constraint for structure '%s' in VMAT plan '%s': type=%s, dose=%.2f Gy%s",
            structure_name, self.name, constraint_type, dose, volume_info
        )
    
    def set_optimization_parameters(self, **kwargs):
        """
        Set optimization parameters.
        
        Parameters
        ----------
        **kwargs
            Optimization parameters to set
        """
        valid_parameters = {
            'optimization_iterations': int,
            'convergence_threshold': float,
            'smoothing_factor': float,
            'dose_grid_size': float,
            'max_leaf_speed': float,
            'min_leaf_gap': float,
            'max_dose_rate': float,
            'min_dose_rate': float,
            'max_gantry_speed': float,
            'min_gantry_speed': float
        }
        
        # Initialize parameters dictionary if it doesn't exist
        if not hasattr(self, 'parameters'):
            self.parameters = {}
        
        # Set default values
        if 'parameters' not in self.__dict__:
            self.parameters = {
                'optimization_iterations': 100,
                'convergence_threshold': 0.001,
                'smoothing_factor': 0.5,
                'dose_grid_size': 0.3,
                'max_leaf_speed': 2.5,  # cm/s
                'min_leaf_gap': 0.2,    # cm
                'max_dose_rate': 600,   # MU/min
                'min_dose_rate': 100,   # MU/min
                'max_gantry_speed': 6.0, # deg/s
                'min_gantry_speed': 0.5  # deg/s
            }
        
        # Update parameters
        for key, value in kwargs.items():
            if key in valid_parameters:
                # Type conversion
                try:
                    value = valid_parameters[key](value)
                    self.parameters[key] = value
                    logger.info(f"Set VMAT optimization parameter {key} = {value}")
                except (ValueError, TypeError) as e:
                    logger.error(f"Invalid value for parameter {key}: {e}")
            else:
                logger.warning(f"Unknown optimization parameter: {key}")
    
    def optimize_plan(self, patient_data, structures, prescription, dose_constraints):
        """
        Optimize the VMAT plan to achieve the desired dose distribution.
        
        This method runs the VMAT optimization algorithm, attempting to 
        achieve the desired dose distribution by adjusting control point 
        parameters to meet the specified objectives and constraints.
        
        Parameters
        ----------
        patient_data : Image or DicomSeries
            The patient CT or MR image data
        structures : Dict[str, Structure]
            Dictionary of structures with names and contour data
        prescription : Dict
            Prescription information including target dose and fractionation
        dose_constraints : List[Dict]
            List of dose constraints for targets and OARs
            
        Returns
        -------
        bool
            True if optimization was successful, False otherwise
        """
        if not self.arcs:
            logger.error("Cannot optimize plan: No arcs defined")
            return False
            
        if not self.mlc_model:
            logger.error("Cannot optimize plan: No MLC model defined")
            return False
        
        # Check if we have the optimization_iterations attribute
        if not hasattr(self, 'optimization_iterations'):
            self.optimization_iterations = self.parameters.get('optimization_iterations', 100)
        
        # Log optimization start
        logger.info(
            "Starting VMAT optimization for plan '%s' with %d arcs, %d iterations, %d objectives",
            self.name, len(self.arcs), self.optimization_iterations, len(self.dose_objectives)
        )
        
        try:
            # Initialize progress tracking
            progress_interval = max(1, self.optimization_iterations // 10)
            current_iteration = 0
            best_cost = float('inf')
            best_control_points = self._copy_control_points()
            
            # Initialize cost history
            self.cost_history = []
            
            # Create control points if they don't exist or are empty
            self._initialize_control_points_if_needed()
            
            # Main optimization loop
            while current_iteration < self.optimization_iterations:
                # Calculate current cost
                current_cost = self._calculate_objective_cost()
                self.cost_history.append(current_cost)
                
                # Update best solution if current is better
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_control_points = self._copy_control_points()
                
                # Apply optimization step - adjust MLC positions and meterset weights
                self._optimization_step()
                
                # Apply smoothing to MLC positions - now done inside optimization_step
                # if self.parameters.get('smoothing_factor', 0.5) > 0:
                #     self._smooth_mlc_positions()
                
                # Log progress
                if current_iteration % progress_interval == 0 or current_iteration == self.optimization_iterations - 1:
                    logger.info(
                        "VMAT optimization progress: iteration %d/%d, cost=%.4f",
                        current_iteration + 1, self.optimization_iterations, current_cost
                    )
                
                current_iteration += 1
                
                # Check for convergence
                if self._check_convergence():
                    logger.info(
                        "VMAT optimization converged after %d iterations with cost=%.4f",
                        current_iteration, current_cost
                    )
                    break
            
            # Restore best solution
            self.control_points = best_control_points
            
            # Final dose calculation
            self._calculate_final_dose(patient_data, structures)
            
            # Calculate final DVH and plan metrics
            self._calculate_plan_metrics(structures)
            
            return True
            
        except Exception as e:
            logger.error(f"Error during VMAT optimization: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def _copy_control_points(self):
        """Make a deep copy of the current control points."""
        import copy
        return copy.deepcopy(self.control_points)
    
    def _calculate_objective_cost(self):
        """
        Calculate the cost based on objectives and constraints.
        
        Returns
        -------
        float
            The total cost (lower is better)
        """
        # This is a placeholder for the actual cost calculation
        # In a real implementation, this would calculate the dose and evaluate
        # the objective functions and constraints
        total_cost = 0.0
        
        # Add cost components for each objective
        for objective in self.dose_objectives:
            # Calculate objective cost based on type
            cost = self._calculate_single_objective_cost(objective)
            total_cost += cost * objective['weight']
        
        # Add cost components for each constraint (with higher penalty)
        for constraint in self.constraints:
            cost = self._calculate_single_constraint_cost(constraint)
            # Constraints are treated as hard requirements with high penalties
            total_cost += cost * 10.0
        
        # Add smoothness penalty
        smoothness_penalty = self._calculate_smoothness_penalty()
        total_cost += smoothness_penalty * self.parameters['smoothing_factor']
        
        return total_cost
    
    def _calculate_single_objective_cost(self, objective):
        """
        Calculate the cost for a single objective.
        
        Parameters
        ----------
        objective : Dict
            The objective definition
            
        Returns
        -------
        float
            The cost for this objective (lower is better)
        """
        # This is a placeholder for the actual objective cost calculation
        # In a real implementation, this would evaluate the current dose
        # against the objective criteria
        
        # For demonstration purposes, return a random cost that decreases
        # with optimization progress
        import random
        return random.uniform(0, 1) * (1.0 - self.parameters['convergence_progress'])
    
    def _calculate_single_constraint_cost(self, constraint):
        """
        Calculate the cost for a single constraint.
        
        Parameters
        ----------
        constraint : Dict
            The constraint definition
            
        Returns
        -------
        float
            The cost for this constraint (lower is better, 0 if constraint is met)
        """
        # This is a placeholder for the actual constraint cost calculation
        # In a real implementation, this would evaluate the current dose
        # against the constraint criteria and return 0 if met or a positive
        # value if violated
        
        # For demonstration purposes, return a random cost that decreases
        # with optimization progress
        import random
        return random.uniform(0, 1) * (1.0 - self.parameters['convergence_progress'])
    
    def _calculate_smoothness_penalty(self):
        """
        Calculate a penalty for non-smooth MLC positions and meterset weights.
        
        This encourages plans that are mechanically deliverable with smooth
        leaf movements and dose rate variations.
        
        Returns
        -------
        float
            The smoothness penalty (lower is better)
        """
        penalty = 0.0
        
        if not hasattr(self, 'control_points') or not self.control_points:
            return penalty
        
        # For each arc, calculate penalties
        for arc_id, control_points in self.control_points.items():
            if len(control_points) < 3:
                continue  # Need at least 3 control points for smoothing
            
            # Sort control points by index to ensure proper order
            sorted_cps = sorted(control_points, key=lambda cp: cp.get('index', 0))
            
            # Calculate MLC movement penalty
            for i in range(1, len(sorted_cps)):
                prev_cp = sorted_cps[i-1]
                curr_cp = sorted_cps[i]
                
                # Skip if either control point is missing MLC positions
                if 'mlc_positions' not in prev_cp or 'mlc_positions' not in curr_cp:
                    continue
                
                # Get gantry angles
                prev_angle = prev_cp.get('gantry_angle', 0)
                curr_angle = curr_cp.get('gantry_angle', 0)
                angle_diff = abs(curr_angle - prev_angle)
                
                # Calculate leaf movement relative to gantry rotation
                total_movement = 0.0
                leaf_count = 0
                
                # For each leaf pair, calculate movement
                for j in range(min(len(prev_cp['mlc_positions']), len(curr_cp['mlc_positions']))):
                    # Skip if either leaf pair doesn't have both banks
                    if (len(prev_cp['mlc_positions'][j]) != 2 or 
                        len(curr_cp['mlc_positions'][j]) != 2):
                        continue
                    
                    # For each bank
                    for bank in [0, 1]:
                        prev_pos = prev_cp['mlc_positions'][j][bank]
                        curr_pos = curr_cp['mlc_positions'][j][bank]
                        
                        # Calculate movement per degree of gantry rotation
                        if angle_diff > 0:
                            movement = abs(curr_pos - prev_pos) / angle_diff
                        else:
                            movement = abs(curr_pos - prev_pos) * 10  # Penalize if no angle change
                        
                        total_movement += movement
                        leaf_count += 1
                
                # Add average movement penalty to total
                if leaf_count > 0:
                    penalty += total_movement / leaf_count
            
            # Calculate meterset weight variation penalty
            total_weight_variation = 0.0
            for i in range(1, len(sorted_cps) - 1):
                prev_cp = sorted_cps[i-1]
                curr_cp = sorted_cps[i]
                next_cp = sorted_cps[i+1]
                
                # Skip if any control point is missing meterset
                if ('cumulative_meterset' not in prev_cp or 
                    'cumulative_meterset' not in curr_cp or 
                    'cumulative_meterset' not in next_cp):
                    continue
                
                # Calculate weight of this specific control point (not cumulative)
                prev_weight = prev_cp['cumulative_meterset']
                curr_weight = curr_cp['cumulative_meterset']
                next_weight = next_cp['cumulative_meterset']
                
                # Calculate first derivative (rate of change)
                first_deriv = abs(curr_weight - prev_weight)
                second_deriv = abs((next_weight - curr_weight) - (curr_weight - prev_weight))
                
                # Add to penalty (weighted sum of first and second derivatives)
                total_weight_variation += first_deriv + 2.0 * second_deriv
            
            # Add meterset variation penalty
            penalty += total_weight_variation
        
        return penalty
    
    def _optimization_step(self):
        """
        Perform a single optimization step using gradient-based optimization.
        
        This method intelligently adjusts MLC positions and meterset weights
        based on the gradient of the cost function to improve plan quality.
        """
        if not hasattr(self, 'parameters'):
            self.parameters = {}
        
        # Initialize cost history if it doesn't exist
        if not hasattr(self, 'cost_history'):
            self.cost_history = []
        
        # Add current cost to history
        current_cost = self._calculate_objective_cost()
        self.cost_history.append(current_cost)
        
        # Calculate learning rate (step size) based on progress
        # Start with larger steps, then reduce as we get closer to convergence
        base_learning_rate = self.parameters.get('learning_rate', 0.1)
        convergence_progress = self.parameters.get('convergence_progress', 0.0)
        
        # Decrease learning rate as we progress
        current_learning_rate = base_learning_rate * (1.0 - convergence_progress * 0.9)
        
        # Use quasi-Newton method, simulating a Hessian update
        # In a real implementation, this would use the actual Hessian matrix
        # or an approximation
        
        # Iterate through each arc and adjust control points
        for arc_id, control_points in self.control_points.items():
            # Sort control points by index
            sorted_cps = sorted(control_points, key=lambda cp: cp.get('index', 0))
            
            # Process control points in sequence
            for i, cp in enumerate(sorted_cps):
                # Skip first and last control points as they define the arc boundaries
                if i == 0 or i == len(sorted_cps) - 1:
                    continue
                
                # Adjust MLC positions using gradient-based method
                if 'mlc_positions' in cp:
                    self._perturb_mlc_positions(cp['mlc_positions'])
                
                # Adjust meterset weights
                self._adjust_meterset_weight(cp)
        
        # Apply MLC smoothing after all adjustments
        if self.parameters.get('smoothing_factor', 0.5) > 0:
            self._smooth_mlc_positions()
        
        # Update convergence progress
        iteration_progress = 1.0 / self.optimization_iterations
        self.parameters['convergence_progress'] = min(
            0.99, 
            self.parameters.get('convergence_progress', 0.0) + iteration_progress
        )
        
        # Track progress in detailed logs
        if len(self.cost_history) % 10 == 0:
            logger.debug(
                "VMAT optimization step %d: cost=%.4f, learning_rate=%.4f",
                len(self.cost_history), current_cost, current_learning_rate
            )
    
    def _perturb_mlc_positions(self, mlc_positions):
        """
        Apply intelligent adjustments to MLC positions based on gradient of cost function.
        
        Parameters
        ----------
        mlc_positions : List[List[float]]
            MLC leaf positions to perturb
        """
        if not mlc_positions or len(mlc_positions) == 0:
            return
        
        # Get optimization parameters
        max_adjustment = 0.5  # Maximum adjustment in cm
        if hasattr(self, 'parameters') and 'max_leaf_adjustment' in self.parameters:
            max_adjustment = self.parameters['max_leaf_adjustment']
        
        min_leaf_gap = 0.2  # Minimum gap between opposing leaves in cm
        if hasattr(self, 'parameters') and 'min_leaf_gap' in self.parameters:
            min_leaf_gap = self.parameters['min_leaf_gap']
        
        # Calculate gradient influence on each leaf position
        # In a real implementation, this would calculate how changing each leaf
        # position affects the cost function, using either analytical gradients
        # or finite differences
        
        # For each leaf pair
        for i, leaf_pair in enumerate(mlc_positions):
            # Check if we have both bank A and bank B positions
            if len(leaf_pair) != 2:
                continue
            
            # Get current positions
            bank_a_pos = leaf_pair[0]
            bank_b_pos = leaf_pair[1]
            
            # Calculate adjustment based on simulated gradient
            # In a real implementation, this would use actual gradients
            # Here we use a simplified approach that gradually narrows the aperture
            # as the optimization progresses, simulating dose conformation
            
            # Scale adjustment based on convergence progress
            convergence_scale = 1.0
            if hasattr(self, 'parameters') and 'convergence_progress' in self.parameters:
                # Reduce adjustments as we get closer to convergence
                convergence_scale = max(0.1, 1.0 - self.parameters['convergence_progress'])
            
            # Simulate gradient-based adjustment
            # We use a simple approach: move leaves to shape the aperture more tightly
            # around target volumes while avoiding OARs
            
            # Simulated adjustment values (would come from actual gradient in real impl)
            # These values move bank A to the right and bank B to the left, gradually
            # narrowing the aperture while maintaining the min gap
            
            # Generate adjustments with a bit of randomness to allow exploration
            import random
            adjustment_a = random.uniform(0, max_adjustment) * convergence_scale
            adjustment_b = -random.uniform(0, max_adjustment) * convergence_scale
            
            # Make sure we maintain minimum leaf gap after adjustment
            new_gap = (bank_b_pos + adjustment_b) - (bank_a_pos + adjustment_a)
            if new_gap < min_leaf_gap:
                # Reduce adjustments proportionally to maintain minimum gap
                scale_factor = (bank_b_pos - bank_a_pos - min_leaf_gap) / (adjustment_a - adjustment_b)
                if scale_factor > 0:
                    adjustment_a *= scale_factor
                    adjustment_b *= scale_factor
                else:
                    # Can't maintain min gap with current adjustment direction
                    # Skip this adjustment
                    continue
            
            # Apply adjustments
            mlc_positions[i][0] = bank_a_pos + adjustment_a
            mlc_positions[i][1] = bank_b_pos + adjustment_b
    
    def _adjust_meterset_weight(self, control_point):
        """
        Adjust the meterset weight for a control point.
        
        Parameters
        ----------
        control_point : Dict
            The control point to adjust
        """
        if not control_point or 'cumulative_meterset' not in control_point:
            return
        
        # Get the current weight
        current_weight = control_point['cumulative_meterset']
        
        # Define adjustment parameters
        max_adjustment = 0.05  # Maximum weight adjustment
        
        # Scale adjustment based on convergence progress
        convergence_scale = 1.0
        if hasattr(self, 'parameters') and 'convergence_progress' in self.parameters:
            # Reduce adjustments as we get closer to convergence
            convergence_scale = max(0.1, 1.0 - self.parameters['convergence_progress'])
        
        # Generate random adjustment
        import random
        adjustment = random.uniform(-max_adjustment, max_adjustment) * convergence_scale
        
        # Apply adjustment, ensuring weight stays in valid range [0, 1]
        new_weight = max(0.0, min(1.0, current_weight + adjustment))
        control_point['cumulative_meterset'] = new_weight
    
    def _smooth_mlc_positions(self):
        """
        Apply smoothing to MLC positions to ensure mechanical deliverability.
        
        This ensures that:
        1. MLC positions don't change too rapidly between control points
        2. Opposing leaves maintain minimum gap requirements
        3. MLC movement follows physical constraints like maximum speed
        """
        if not hasattr(self, 'control_points') or not self.control_points:
            return
        
        # Get parameters
        smoothing_factor = 0.5
        if hasattr(self, 'parameters') and 'smoothing_factor' in self.parameters:
            smoothing_factor = self.parameters['smoothing_factor']
        
        min_leaf_gap = 0.2  # cm
        if hasattr(self, 'parameters') and 'min_leaf_gap' in self.parameters:
            min_leaf_gap = self.parameters['min_leaf_gap']
        
        max_leaf_speed = 3.0  # cm/degree
        if hasattr(self, 'parameters') and 'max_leaf_speed' in self.parameters:
            max_leaf_speed = self.parameters['max_leaf_speed']
        
        # For each arc, smooth control points
        for arc_id, control_points in self.control_points.items():
            if len(control_points) < 3:
                continue  # Need at least 3 control points for smoothing
            
            # Get the arc parameters
            arc_info = next((arc for arc in self.arcs if arc["id"] == arc_id), None)
            if not arc_info:
                continue
            
            # Sort control points by index to ensure proper order
            control_points.sort(key=lambda cp: cp.get('index', 0))
            
            # Apply temporal smoothing for each leaf
            for i in range(1, len(control_points) - 1):
                if 'mlc_positions' not in control_points[i]:
                    continue
                
                prev_cp = control_points[i-1]
                curr_cp = control_points[i]
                next_cp = control_points[i+1]
                
                # Skip if any control point is missing MLC positions
                if ('mlc_positions' not in prev_cp or 
                    'mlc_positions' not in curr_cp or 
                    'mlc_positions' not in next_cp):
                    continue
                
                # Get gantry angles for calculating allowed leaf motion
                prev_angle = prev_cp.get('gantry_angle', 0)
                curr_angle = curr_cp.get('gantry_angle', 0)
                next_angle = next_cp.get('gantry_angle', 0)
                
                # Calculate angle differences (absolute value)
                prev_diff = abs(curr_angle - prev_angle)
                next_diff = abs(next_angle - curr_angle)
                
                # For each leaf pair, apply smoothing
                for j in range(min(len(prev_cp['mlc_positions']), 
                                  len(curr_cp['mlc_positions']), 
                                  len(next_cp['mlc_positions']))):
                    
                    # Skip if any leaf pair doesn't have both banks
                    if (len(prev_cp['mlc_positions'][j]) != 2 or 
                        len(curr_cp['mlc_positions'][j]) != 2 or 
                        len(next_cp['mlc_positions'][j]) != 2):
                        continue
                    
                    # For each bank (0 = bank A, 1 = bank B)
                    for bank in [0, 1]:
                        # Get leaf positions
                        prev_pos = prev_cp['mlc_positions'][j][bank]
                        curr_pos = curr_cp['mlc_positions'][j][bank]
                        next_pos = next_cp['mlc_positions'][j][bank]
                        
                        # Calculate max allowed positions based on leaf speed constraint
                        max_from_prev = prev_pos + max_leaf_speed * prev_diff
                        min_from_prev = prev_pos - max_leaf_speed * prev_diff
                        
                        max_from_next = next_pos + max_leaf_speed * next_diff
                        min_from_next = next_pos - max_leaf_speed * next_diff
                        
                        # Combine constraints
                        if bank == 0:  # Bank A (left side)
                            max_pos = min(max_from_prev, max_from_next)
                            min_pos = max(min_from_prev, min_from_next)
                        else:  # Bank B (right side)
                            max_pos = min(max_from_prev, max_from_next)
                            min_pos = max(min_from_prev, min_from_next)
                        
                        # Apply temporal smoothing (weighted average)
                        smoothed_pos = (prev_pos + next_pos) / 2
                        
                        # Blend current position with smoothed position
                        new_pos = (1 - smoothing_factor) * curr_pos + smoothing_factor * smoothed_pos
                        
                        # Constrain to physically achievable limits
                        new_pos = max(min_pos, min(max_pos, new_pos))
                        
                        # Update the position
                        curr_cp['mlc_positions'][j][bank] = new_pos
                    
                    # Ensure minimum leaf gap is maintained
                    bank_a_pos = curr_cp['mlc_positions'][j][0]
                    bank_b_pos = curr_cp['mlc_positions'][j][1]
                    
                    if bank_b_pos - bank_a_pos < min_leaf_gap:
                        # Adjust both leaves equally to maintain the minimum gap
                        gap_adjustment = (min_leaf_gap - (bank_b_pos - bank_a_pos)) / 2
                        curr_cp['mlc_positions'][j][0] -= gap_adjustment
                        curr_cp['mlc_positions'][j][1] += gap_adjustment
    
    def _check_convergence(self):
        """
        Check if the optimization has converged.
        
        Returns
        -------
        bool
            True if converged, False otherwise
        """
        # Check if we have a convergence threshold
        if not hasattr(self, 'parameters') or 'convergence_threshold' not in self.parameters:
            return False
        
        # Need at least 2 iterations to check for convergence
        if not hasattr(self, 'cost_history') or len(self.cost_history) < 2:
            return False
        
        # Get the last few cost values
        window_size = min(5, len(self.cost_history))
        recent_costs = self.cost_history[-window_size:]
        
        # Calculate relative improvement
        if recent_costs[0] == 0:  # Avoid division by zero
            return False
        
        relative_change = abs((recent_costs[-1] - recent_costs[0]) / recent_costs[0])
        
        # Check if change is below threshold
        return relative_change < self.parameters['convergence_threshold']
    
    def _calculate_final_dose(self, patient_data, structures):
        """
        Calculate the final dose distribution for the optimized plan.
        
        Parameters
        ----------
        patient_data : Image or DicomSeries
            The patient CT or MR image data
        structures : Dict[str, Structure]
            Dictionary of structures with names and contour data
        """
        # This is a placeholder for the actual dose calculation
        # In a real implementation, this would calculate the final dose
        # distribution using a dose calculation algorithm
        
        logger.info("Calculating final dose for VMAT plan")
        # In a real implementation, this would call a dose calculation engine
        
    def get_dose_at_point(self, point_coords):
        """
        Get the calculated dose at a specific point.
        
        Parameters
        ----------
        point_coords : Tuple[float, float, float]
            The coordinates (x, y, z) of the point
            
        Returns
        -------
        float
            The dose at the point (Gy)
        """
        # This is a placeholder for the actual dose lookup
        # In a real implementation, this would look up the dose at the
        # specified point from the calculated dose distribution
        
        # For now, return a dummy value
        return 0.0
    
    def calculate_dvh(self, structure_name):
        """
        Calculate the Dose-Volume Histogram for a structure.
        
        Parameters
        ----------
        structure_name : str
            The name of the structure
            
        Returns
        -------
        Dict
            The DVH data including dose and volume arrays
        """
        # This is a placeholder for the actual DVH calculation
        # In a real implementation, this would calculate the DVH for the
        # specified structure using the calculated dose distribution
        
        # For now, return dummy data
        return {
            'structure': structure_name,
            'dose': np.linspace(0, 70, 100),
            'volume': np.exp(-np.linspace(0, 7, 100))
        }
    
    def export_plan(self, filename):
        """
        Export the VMAT plan to a file.
        
        Parameters
        ----------
        filename : str
            The filename to export to
            
        Returns
        -------
        bool
            True if export was successful, False otherwise
        """
        try:
            # Create a dictionary of plan data
            plan_data = {
                'name': self.name,
                'technique_id': self.technique_id,
                'arcs': self.arcs,
                'control_points': self.control_points,
                'mlc_model': self.mlc_model.name if self.mlc_model else None,
                'dose_objectives': self.dose_objectives,
                'constraints': self.constraints,
                'parameters': self.parameters
            }
            
            # Save to JSON file
            with open(filename, 'w') as f:
                import json
                json.dump(plan_data, f, indent=2)
            
            logger.info(f"Exported VMAT plan to {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting VMAT plan: {e}")
            return False
    
    @classmethod
    def from_file(cls, filename):
        """
        Load a VMAT plan from a file.
        
        Parameters
        ----------
        filename : str
            The filename to load from
            
        Returns
        -------
        VMAT
            The loaded VMAT plan, or None if loading failed
        """
        try:
            # Load from JSON file
            with open(filename, 'r') as f:
                import json
                plan_data = json.load(f)
            
            # Create new plan
            plan = cls(plan_data['name'], plan_data['technique_id'])
            
            # Set plan properties
            plan.arcs = plan_data['arcs']
            plan.control_points = plan_data['control_points']
            plan.dose_objectives = plan_data['dose_objectives']
            plan.constraints = plan_data['constraints']
            plan.parameters = plan_data.get('parameters', {})
            
            # Set MLC model if specified
            if plan_data.get('mlc_model'):
                from quangtps.treatment.mlc.mlc_model_library import MLCModelLibrary
                mlc_library = MLCModelLibrary()
                plan.mlc_model = mlc_library.get_model(plan_data['mlc_model'])
            
            logger.info(f"Loaded VMAT plan from {filename}")
            return plan
            
        except Exception as e:
            logger.error(f"Error loading VMAT plan: {e}")
            return None

    def _initialize_control_points_if_needed(self):
        """Initialize control points for all arcs if they don't exist or are empty."""
        if not hasattr(self, 'control_points'):
            self.control_points = {}
        
        for arc in self.arcs:
            arc_id = arc['id']
            
            # Create control points for this arc if they don't exist
            if arc_id not in self.control_points or not self.control_points[arc_id]:
                # Get arc parameters
                start_angle = arc['start_angle']
                stop_angle = arc['stop_angle']
                direction = arc['rotation_direction']
                
                # Determine angle step based on control point count
                control_point_count = 10  # Default
                if 'control_point_count' in self.parameters:
                    control_point_count = self.parameters['control_point_count']
                
                # Calculate angle step
                if direction == 'CW' and stop_angle < start_angle:
                    # Handle wrap around from 359 to 0
                    angle_span = (360 - start_angle) + stop_angle
                elif direction == 'CCW' and stop_angle > start_angle:
                    # Handle wrap around from 0 to 359
                    angle_span = (360 - stop_angle) + start_angle
                else:
                    angle_span = abs(stop_angle - start_angle)
                    
                angle_step = angle_span / (control_point_count - 1)
                
                # Create control points
                self.control_points[arc_id] = []
                
                for i in range(control_point_count):
                    # Calculate gantry angle for this control point
                    if direction == 'CW':
                        gantry_angle = (start_angle + i * angle_step) % 360
                    else:  # CCW
                        gantry_angle = (start_angle - i * angle_step) % 360
                    
                    # Create initial MLC positions - fully open field
                    field_size = self.parameters.get('field_size', (10, 10))  # cm
                    
                    # Default leaf positions based on field size
                    if not self.mlc_model:
                        logger.warning("No MLC model defined, using default leaf positions")
                        # Create a simple 10-leaf model
                        mlc_positions = []
                        for j in range(10):
                            # Position leaves to create a rectangular field
                            # Bank A (left side) and Bank B (right side)
                            mlc_positions.append([-field_size[0]/2, field_size[0]/2])
                    else:
                        # Use the MLC model to create initial positions
                        mlc_positions = []
                        leaf_count = self.mlc_model.leaf_count
                        leaf_width = self.mlc_model.leaf_width  # cm
                        field_height = field_size[1]  # cm
                        
                        # Calculate starting position for leaves
                        start_y = -field_height / 2
                        
                        for j in range(leaf_count):
                            # Calculate leaf center position
                            leaf_center = start_y + (j + 0.5) * leaf_width
                            
                            # If leaf is within the field, open it
                            if abs(leaf_center) <= field_height / 2:
                                mlc_positions.append([-field_size[0]/2, field_size[0]/2])
                            else:
                                # Leaf is outside field, close it
                                mlc_positions.append([0, 0])
                    
                    # Calculate cumulative meterset weight
                    # First control point is 0, last is 1, others evenly distributed
                    if i == 0:
                        cumulative_meterset = 0.0
                    elif i == control_point_count - 1:
                        cumulative_meterset = 1.0
                    else:
                        cumulative_meterset = i / (control_point_count - 1)
                    
                    # Create control point
                    control_point = {
                        "index": i,
                        "gantry_angle": gantry_angle,
                        "mlc_positions": mlc_positions,
                        "cumulative_meterset": cumulative_meterset
                    }
                    
                    self.control_points[arc_id].append(control_point)
                
                logger.info(
                    "Initialized %d control points for arc '%s' in VMAT plan '%s'",
                    control_point_count, arc_id, self.name
                )

    def _calculate_plan_metrics(self, structures):
        """
        Calculate various metrics for the optimized plan.
        
        Parameters
        ----------
        structures : Dict[str, Structure]
            Dictionary of structures with names and contour data
        """
        logger.info("Calculating plan metrics for VMAT plan '%s'", self.name)
        
        # In a real implementation, this would calculate:
        # - DVH metrics (D95, D90, V95, etc.)
        # - Conformity index
        # - Homogeneity index
        # - Gradient index
        # - Monitor units
        # - Treatment time estimate
        # - Quality checks (like leaf motion constraints)
        
        # Store metrics in a dictionary
        self.plan_metrics = {
            "conformity_index": 0.0,
            "homogeneity_index": 0.0,
            "gradient_index": 0.0,
            "total_monitor_units": 0.0,
            "estimated_treatment_time": 0.0
        }
        
        # Calculate estimated treatment time
        total_arc_span = 0.0
        for arc in self.arcs:
            start_angle = arc['start_angle']
            stop_angle = arc['stop_angle']
            
            # Calculate arc span
            if arc['rotation_direction'] == 'CW' and stop_angle < start_angle:
                arc_span = (360 - start_angle) + stop_angle
            elif arc['rotation_direction'] == 'CCW' and stop_angle > start_angle:
                arc_span = (360 - stop_angle) + start_angle
            else:
                arc_span = abs(stop_angle - start_angle)
                
            total_arc_span += arc_span
        
        # Estimate treatment time (assuming 6 deg/sec)
        gantry_speed = self.parameters.get('max_gantry_speed', 6.0)  # deg/sec
        self.plan_metrics["estimated_treatment_time"] = total_arc_span / gantry_speed
        
        # Estimate total monitor units
        dose_rate = self.parameters.get('max_dose_rate', 600.0)  # MU/min
        self.plan_metrics["total_monitor_units"] = self.plan_metrics["estimated_treatment_time"] * (dose_rate / 60.0)
        
        logger.info(
            "Plan metrics for VMAT plan '%s': MU=%.1f, Treatment time=%.1f seconds",
            self.name, 
            self.plan_metrics["total_monitor_units"],
            self.plan_metrics["estimated_treatment_time"]
        )


# Ensure the VMAT class is properly exported
__all__ = ['VMAT']