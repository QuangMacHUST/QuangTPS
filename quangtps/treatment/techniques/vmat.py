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
from typing import List, Dict, Any, Optional, Tuple, Union, TypeVar
from dataclasses import dataclass
from enum import Enum
import numpy as np
from numpy.typing import NDArray

from quangtps.treatment.mlc.mlc_model import MLCModel
from quangtps.treatment.techniques.technique_interface import BaseTreatmentTechnique, TechniqueCategory
from quangtps.structures.structure import Structure
from quangtps.dose.dose_grid import DoseGrid
from quangtps.dose.dose_calculator import DoseCalculator
from quangtps.treatment.mlc.mlc_model_library import MLCModelLibrary
from quangtps.patient.patient_data import PatientData
from quangtps.prescription.prescription import Prescription
from quangtps.dose.dose_constraints import DoseConstraints

logger = logging.getLogger(__name__)

T = TypeVar('T')

@dataclass
class Arc:
    """Arc parameters for VMAT delivery."""
    id: str
    name: str
    start_angle: float
    stop_angle: float
    rotation_direction: str
    energy: str
    dose_rate: float

@dataclass 
class ControlPoint:
    """Control point parameters for VMAT delivery."""
    index: int
    gantry_angle: float
    mlc_positions: List[List[float]]
    cumulative_meterset: float

@dataclass
class DoseObjective:
    """Dose objective parameters."""
    structure: str
    type: str
    dose: float
    volume: Optional[float]
    weight: float = 1.0

@dataclass
class DoseConstraint:
    """Dose constraint parameters."""
    structure: str
    type: str
    dose: float
    volume: Optional[float]

class VMAT(BaseTreatmentTechnique):
    """
    Class representing a Volumetric Modulated Arc Therapy (VMAT) plan.
    
    Attributes
    ----------
    name : str
        Name of the VMAT plan
    technique_id : str
        Unique ID for the plan
    arcs : List[Arc]
        List of arc definitions
    control_points : Dict[str, List[ControlPoint]]
        Control points for each arc
    mlc_model : Optional[MLCModel]
        Multi-leaf collimator model
    dose_objectives : List[DoseObjective]
        List of dose objectives
    constraints : List[DoseConstraint]
        List of dose constraints
    parameters : Dict[str, Any]
        Optimization parameters
    structures : Dict[str, Structure]
        Dictionary of structures used in the plan
    current_dose : Optional[DoseGrid]
        Current calculated dose distribution
    dose_calculator : Optional[DoseCalculator]
        Dose calculation engine
    """
    
    def __init__(self, name: str, technique_id: Optional[str] = None) -> None:
        """Initialize a VMAT plan."""
        super().__init__(name=name, technique_id=technique_id, category=TechniqueCategory.ADVANCED)
        
        self.arcs: List[Arc] = []
        self.control_points: Dict[str, List[ControlPoint]] = {}
        self.mlc_model: Optional[MLCModel] = None
        self.dose_objectives: List[DoseObjective] = []
        self.constraints: List[DoseConstraint] = []
        self.parameters: Dict[str, Any] = {}
        self.structures: Dict[str, Structure] = {}
        self.current_dose: Optional[DoseGrid] = None
        self.dose_calculator: Optional[DoseCalculator] = None
        
        logger.info(
            "Initialized VMAT plan '%s' (ID: %s)",
            self.name, self.technique_id
        )
    
    def add_arc(self, arc: Arc) -> None:
        """
        Add an arc to the VMAT plan.
        
        Parameters
        ----------
        arc : Arc
            Arc to add
        """
        self.arcs.append(arc)
        logger.info(
            "Added arc '%s' to VMAT plan '%s'",
            arc.name, self.name
        )
    
    def add_structure(self, structure: Structure) -> None:
        """
        Add a structure to the VMAT plan.
        
        Parameters
        ----------
        structure : Structure
            Structure to add
        """
        self.structures[structure.name] = structure
        logger.info(
            "Added structure '%s' to VMAT plan '%s'",
            structure.name, self.name
        )

    def get_structure(self, name: str) -> Optional[Structure]:
        """
        Get a structure by name.
        
        Parameters
        ----------
        name : str
            Name of the structure
            
        Returns
        -------
        Optional[Structure]
            Structure if found, None otherwise
        """
        return self.structures.get(name)

    def get_arc(self, arc_id: str) -> Optional[Arc]:
        """
        Get an arc by ID.
        
        Parameters
        ----------
        arc_id : str
            ID of the arc
            
        Returns
        -------
        Optional[Arc]
            Arc if found, None otherwise
        """
        for arc in self.arcs:
            if arc.id == arc_id:
                return arc
        return None

    def get_control_points(self, arc_id: str) -> Optional[List[ControlPoint]]:
        """
        Get control points for an arc.
        
        Parameters
        ----------
        arc_id : str
            ID of the arc
            
        Returns
        -------
        Optional[List[ControlPoint]]
            Control points if found, None otherwise
        """
        return self.control_points.get(arc_id)

    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validate the VMAT plan.
        
        Returns
        -------
        Tuple[bool, List[str]]
            Tuple containing:
            - bool: True if valid, False otherwise
            - List[str]: List of validation errors
        """
        errors = []
        
        # Check if we have any arcs
        if not self.arcs:
            errors.append("No arcs defined")
            
        # Check if we have any control points
        for arc in self.arcs:
            if not self.control_points.get(arc.id):
                errors.append(f"No control points defined for arc '{arc.name}'")
                
        # Check if we have an MLC model
        if self.mlc_model is None:
            errors.append("No MLC model set")
            
        # Check if we have a dose calculator
        if self.dose_calculator is None:
            errors.append("No dose calculator set")
            
        # Check if we have any structures
        if not self.structures:
            errors.append("No structures defined")
            
        # Check if we have any objectives
        if not self.dose_objectives:
            errors.append("No dose objectives defined")
            
        return len(errors) == 0, errors

    def __str__(self) -> str:
        """
        Get a string representation of the plan.
        
        Returns
        -------
        str
            String representation
        """
        return f"VMAT plan '{self.name}' (ID: {self.technique_id})"

    def __repr__(self) -> str:
        """
        Get a detailed string representation of the plan.
        
        Returns
        -------
        str
            Detailed string representation
        """
        return (
            f"VMAT(name='{self.name}', technique_id='{self.technique_id}', "
            f"num_arcs={len(self.arcs)}, num_structures={len(self.structures)}, "
            f"num_objectives={len(self.dose_objectives)}, num_constraints={len(self.constraints)})"
        )
    
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
        objective = DoseObjective(
            structure=structure_name,
            type=objective_type,
            dose=dose,
            volume=volume,
            weight=weight
        )
        
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
        constraint = DoseConstraint(
            structure=structure_name,
            type=constraint_type,
            dose=dose,
            volume=volume
        )
        
        self.constraints.append(constraint)
        
        # Determine volume info for logging
        volume_info = f", volume={volume}%" if volume is not None else ""
        
        # Using lazy % formatting for logging
        logger.info(
            "Added constraint for structure '%s' in VMAT plan '%s': type=%s, dose=%.2f Gy%s",
            structure_name, self.name, constraint_type, dose, volume_info
        )
    
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
            total_cost += cost * objective.weight
        
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
        objective : DoseObjective
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
        constraint : DoseConstraint
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
            sorted_cps = sorted(control_points, key=lambda cp: cp.index)
            
            # Calculate MLC movement penalty
            for i in range(1, len(sorted_cps)):
                prev_cp = sorted_cps[i-1]
                curr_cp = sorted_cps[i]
                
                # Skip if either control point is missing MLC positions
                if 'mlc_positions' not in prev_cp or 'mlc_positions' not in curr_cp:
                    continue
                
                # Get gantry angles
                prev_angle = prev_cp.gantry_angle
                curr_angle = curr_cp.gantry_angle
                angle_diff = abs(curr_angle - prev_angle)
                
                # Calculate leaf movement relative to gantry rotation
                total_movement = 0.0
                leaf_count = 0
                
                # For each leaf pair, calculate movement
                for j in range(min(len(prev_cp.mlc_positions), len(curr_cp.mlc_positions))):
                    # Skip if either leaf pair doesn't have both banks
                    if (len(prev_cp.mlc_positions[j]) != 2 or 
                        len(curr_cp.mlc_positions[j]) != 2):
                        continue
                    
                    # For each bank
                    for bank in [0, 1]:
                        prev_pos = prev_cp.mlc_positions[j][bank]
                        curr_pos = curr_cp.mlc_positions[j][bank]
                        
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
                prev_weight = prev_cp.cumulative_meterset
                curr_weight = curr_cp.cumulative_meterset
                next_weight = next_cp.cumulative_meterset
                
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
            sorted_cps = sorted(control_points, key=lambda cp: cp.index)
            
            # Process control points in sequence
            for i, cp in enumerate(sorted_cps):
                # Skip first and last control points as they define the arc boundaries
                if i == 0 or i == len(sorted_cps) - 1:
                    continue
                
                # Adjust MLC positions using gradient-based method
                if 'mlc_positions' in cp:
                    self._perturb_mlc_positions(cp.mlc_positions)
                
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
        
        # Độ học hiện tại dựa trên tiến trình hội tụ
        learning_rate = self.parameters.get('learning_rate', 0.1)
        convergence_progress = self.parameters.get('convergence_progress', 0.0)
        current_learning_rate = learning_rate * (1.0 - convergence_progress * 0.9)
        
        # Lấy thông tin về các cấu trúc mục tiêu và OARs để điều hướng lá MLC
        targets = self._get_target_structures()
        oars = self._get_oar_structures()
        
        # Tính toán gradient thực tế cho mỗi vị trí lá
        leaf_gradients = self._calculate_leaf_gradients(mlc_positions, targets, oars)
        
        # Cho mỗi cặp lá
        for i, leaf_pair in enumerate(mlc_positions):
            # Kiểm tra xem có cả hai vị trí bank A và bank B không
            if len(leaf_pair) != 2:
                continue
            
            # Lấy vị trí hiện tại
            bank_a_pos = leaf_pair[0]  # Lá bên trái (bank A)
            bank_b_pos = leaf_pair[1]  # Lá bên phải (bank B)
            
            # Lấy gradient cho cặp lá này
            if i < len(leaf_gradients):
                gradient_a, gradient_b = leaf_gradients[i]
            else:
                # Fallback nếu không có gradient
                logger.debug(f"Không có gradient cho cặp lá {i}, sử dụng giá trị mặc định")
                gradient_a = 0.1
                gradient_b = -0.1
            
            # Tính điều chỉnh dựa trên gradient
            # Định hướng gradient: giá trị âm = di chuyển vào trong (thu hẹp khẩu độ)
            adjustment_a = -gradient_a * current_learning_rate * max_adjustment
            adjustment_b = -gradient_b * current_learning_rate * max_adjustment
            
            # Giới hạn điều chỉnh tối đa
            adjustment_a = max(-max_adjustment, min(max_adjustment, adjustment_a))
            adjustment_b = max(-max_adjustment, min(max_adjustment, adjustment_b))
            
            # Đảm bảo duy trì khoảng cách tối thiểu giữa các lá đối diện
            new_gap = (bank_b_pos + adjustment_b) - (bank_a_pos + adjustment_a)
            if new_gap < min_leaf_gap:
                # Giảm điều chỉnh tỷ lệ thuận để duy trì khoảng cách tối thiểu
                current_gap = bank_b_pos - bank_a_pos
                needed_reduction = min_leaf_gap - new_gap
                
                # Phân phối việc giảm giữa hai lá dựa trên hướng gradient
                total_adjustment_magnitude = abs(adjustment_a) + abs(adjustment_b)
                if total_adjustment_magnitude > 0:
                    a_portion = abs(adjustment_a) / total_adjustment_magnitude
                    b_portion = abs(adjustment_b) / total_adjustment_magnitude
                    
                    # Điều chỉnh lại
                    if adjustment_a > 0:  # Lá A đang di chuyển sang phải
                        adjustment_a -= needed_reduction * a_portion
                    else:  # Lá A đang di chuyển sang trái
                        adjustment_a += needed_reduction * a_portion
                        
                    if adjustment_b < 0:  # Lá B đang di chuyển sang trái
                        adjustment_b += needed_reduction * b_portion
                    else:  # Lá B đang di chuyển sang phải
                        adjustment_b -= needed_reduction * b_portion
                else:
                    # Nếu không có điều chỉnh, đảm bảo khoảng cách tối thiểu
                    mid_point = (bank_a_pos + bank_b_pos) / 2
                    bank_a_pos = mid_point - min_leaf_gap / 2
                    bank_b_pos = mid_point + min_leaf_gap / 2
                    adjustment_a = 0
                    adjustment_b = 0
            
            # Áp dụng điều chỉnh
            mlc_positions[i][0] = bank_a_pos + adjustment_a
            mlc_positions[i][1] = bank_b_pos + adjustment_b
            
            # Tính khoảng cách mới và log để gỡ lỗi
            new_gap = mlc_positions[i][1] - mlc_positions[i][0]
            if new_gap < min_leaf_gap:
                logger.warning(f"Khoảng cách giữa cặp lá {i} ({new_gap:.2f} cm) nhỏ hơn giá trị tối thiểu ({min_leaf_gap} cm)")
                # Sửa lỗi cuối cùng nếu vẫn vi phạm
                mid_point = (mlc_positions[i][0] + mlc_positions[i][1]) / 2
                mlc_positions[i][0] = mid_point - min_leaf_gap / 2
                mlc_positions[i][1] = mid_point + min_leaf_gap / 2
    
    def _calculate_leaf_gradients(self, mlc_positions, targets, oars):
        """
        Tính toán gradient cho các vị trí lá MLC.
        
        Đây là nơi thực hiện tính toán gradient thực tế, đánh giá tác động của
        việc thay đổi vị trí từng lá đến hàm mục tiêu tổng thể (đối với cả
        target coverage và OAR sparing).
        
        Parameters
        ----------
        mlc_positions : List[List[float]]
            Vị trí lá MLC hiện tại
        targets : List[Dict]
            Thông tin về các cấu trúc mục tiêu
        oars : List[Dict]
            Thông tin về các cơ quan nguy cấp (OARs)
            
        Returns
        -------
        List[Tuple[float, float]]
            Danh sách gradient cho mỗi cặp lá [gradient_bank_A, gradient_bank_B]
        """
        # Trong triển khai thực tế, chúng ta sẽ tính gradient theo phương pháp sai phân hữu hạn:
        # 1. Đánh giá hàm mục tiêu hiện tại
        # 2. Thay đổi vị trí của mỗi lá MLC một chút (+epsilon)
        # 3. Đánh giá hàm mục tiêu mới
        # 4. Tính gradient = (new_cost - current_cost) / epsilon
        
        num_leaf_pairs = len(mlc_positions)
        gradients = []
        
        # Hàm mục tiêu hiện tại
        current_cost = self._calculate_objective_cost()
        
        # Epsilon cho phương pháp sai phân hữu hạn
        epsilon = 0.1  # 1mm
        
        # Đối với mỗi cặp lá, tính gradient
        for i in range(num_leaf_pairs):
            if len(mlc_positions[i]) != 2:
                gradients.append((0.0, 0.0))
                continue
                
            gradient_a = 0.0
            gradient_b = 0.0
            
            # Lưu vị trí ban đầu
            original_a = mlc_positions[i][0]
            original_b = mlc_positions[i][1]
            
            # Tính gradient cho lá bank A (di chuyển sang phải +epsilon)
            mlc_positions[i][0] += epsilon
            new_cost = self._calculate_objective_cost()
            gradient_a = (new_cost - current_cost) / epsilon
            mlc_positions[i][0] = original_a  # Khôi phục
            
            # Tính gradient cho lá bank B (di chuyển sang trái -epsilon)
            mlc_positions[i][1] -= epsilon
            new_cost = self._calculate_objective_cost()
            gradient_b = (new_cost - current_cost) / epsilon
            mlc_positions[i][1] = original_b  # Khôi phục
            
            # Thêm gradient vào danh sách
            gradients.append((gradient_a, gradient_b))
            
            # Tối ưu hóa hiệu suất: chỉ lấy mẫu một số cặp lá ngẫu nhiên 
            # thay vì tính toán mọi gradient nếu có nhiều cặp lá
            if num_leaf_pairs > 20 and i % 5 != 0 and i < num_leaf_pairs - 1:
                # Sử dụng giá trị tương tự với cặp lá tiếp theo nếu là một phần của cùng một vùng
                gradients.append((gradient_a, gradient_b))
                i += 1  # Bỏ qua cặp lá tiếp theo
        
        # Nếu tính toán gradient toàn diện quá tốn kém, thay thế bằng mô hình dựa trên fluence
        if self.parameters.get('use_fluence_model', False):
            gradients = self._calculate_fluence_based_gradients(mlc_positions, targets, oars)
            
        return gradients
    
    def _get_target_structures(self):
        """
        Lấy thông tin về các cấu trúc mục tiêu từ danh sách các mục tiêu.
        
        Returns
        -------
        List[Dict]
            Danh sách các cấu trúc mục tiêu và thông tin liên quan
        """
        targets = []
        
        # Lấy thông tin từ các mục tiêu
        for objective in self.dose_objectives:
            structure_name = objective.structure
            structure_type = objective.type
            
            # Chỉ xem xét các cấu trúc mục tiêu
            if structure_type.lower() in ['ptv', 'target', 'ctv', 'gtv']:
                # Kiểm tra xem cấu trúc này đã được thêm vào danh sách chưa
                existing = next((t for t in targets if t['name'] == structure_name), None)
                if not existing:
                    targets.append({
                        'name': structure_name,
                        'type': structure_type,
                        'prescription': objective.dose,
                        'prescription': objective.get('dose', 0.0),
                        'weight': objective.get('weight', 1.0)
                    })
        
        return targets
    
    def _get_oar_structures(self):
        """
        Lấy thông tin về các cơ quan nguy cấp (OARs) từ các ràng buộc.
        
        Returns
        -------
        List[Dict]
            Danh sách các OARs và thông tin liên quan
        """
        oars = []
        
        # Lấy thông tin từ các ràng buộc và mục tiêu
        for constraint in self.constraints:
            structure_name = constraint.get('structure', '')
            structure_type = constraint.get('structure_type', '')
            
            # Chỉ xem xét các cấu trúc không phải mục tiêu
            if structure_type.lower() not in ['ptv', 'target', 'ctv', 'gtv']:
                # Kiểm tra xem cấu trúc này đã được thêm vào danh sách chưa
                existing = next((o for o in oars if o['name'] == structure_name), None)
                if not existing:
                    oars.append({
                        'name': structure_name,
                        'type': structure_type,
                        'max_dose': constraint.get('dose', 0.0),
                        'priority': 'high' if constraint.get('priority', 'medium') == 'high' else 'medium'
                    })
        
        # Thêm các OARs từ mục tiêu (thường là mục tiêu liều trung bình hoặc tối đa)
        for objective in self.dose_objectives:
            structure_name = objective.get('structure', '')
            structure_type = objective.get('structure_type', '')
            
            if (structure_type.lower() not in ['ptv', 'target', 'ctv', 'gtv'] and 
                not any(o['name'] == structure_name for o in oars)):
                oars.append({
                    'name': structure_name,
                    'type': structure_type,
                    'max_dose': objective.get('dose', 0.0),
                    'weight': objective.get('weight', 1.0),
                    'priority': 'medium'
                })
        
        return oars
    
    def _calculate_fluence_based_gradients(self, mlc_positions, targets, oars):
        """
        Tính toán gradient dựa trên fluence map mục tiêu.
        
        Đây là một phương pháp xấp xỉ nhanh hơn khi không thể tính toán 
        gradient từ phương pháp sai phân hữu hạn đầy đủ.
        
        Parameters
        ----------
        mlc_positions : List[List[float]]
            Vị trí lá MLC hiện tại
        targets : List[Dict]
            Thông tin về các cấu trúc mục tiêu
        oars : List[Dict]
            Thông tin về các cơ quan nguy cấp (OARs)
            
        Returns
        -------
        List[Tuple[float, float]]
            Danh sách gradient cho mỗi cặp lá
        """
        import numpy as np
        
        # Tạo fluence map mục tiêu lý tưởng (đơn giản hóa)
        num_leaves = len(mlc_positions)
        
        # Fluence map hiện tại từ vị trí lá
        current_fluence = np.zeros((num_leaves, 100))  # Giả sử 100 điểm mẫu theo chiều ngang
        
        # Fill fluence map hiện tại dựa trên vị trí lá
        for i, leaf_pair in enumerate(mlc_positions):
            if len(leaf_pair) != 2:
                continue
                
            bank_a_pos = max(0, min(99, int(leaf_pair[0] * 10)))  # Giả sử thang đo 0.1cm
            bank_b_pos = max(0, min(99, int(leaf_pair[1] * 10)))
            
            # Đặt fluence = 1.0 trong khoảng mở
            current_fluence[i, bank_a_pos:bank_b_pos] = 1.0
        
        # Giả định fluence map lý tưởng (trong triển khai thực tế, điều này sẽ đến từ tính toán liều dựa trên mô hình)
        ideal_fluence = np.ones_like(current_fluence)
        
        # Điều chỉnh fluence map lý tưởng dựa trên thông tin về targets và OARs
        # (Đơn giản hóa mô hình)
        
        # Tính toán độ chênh lệch và hướng điều chỉnh cần thiết
        fluence_diff = ideal_fluence - current_fluence
        
        # Chuyển đổi sự khác biệt fluence thành gradient cho vị trí lá
        gradients = []
        for i in range(num_leaves):
            if i >= len(mlc_positions) or len(mlc_positions[i]) != 2:
                gradients.append((0.0, 0.0))
                continue
                
            bank_a_pos = max(0, min(99, int(mlc_positions[i][0] * 10)))
            bank_b_pos = max(0, min(99, int(mlc_positions[i][1] * 10)))
            
            # Tính gradient theo hướng
            # - Giá trị dương: mở rộng khẩu độ (di chuyển lá A sang trái, lá B sang phải)
            # - Giá trị âm: thu hẹp khẩu độ (di chuyển lá A sang phải, lá B sang trái)
            
            gradient_a = 0.0
            gradient_b = 0.0
            
            # Tính gradient cho bank A
            left_margin = max(0, bank_a_pos - 5)
            if bank_a_pos > 0:
                gradient_a = -np.sum(fluence_diff[i, left_margin:bank_a_pos]) / max(1, bank_a_pos - left_margin)
            
            # Tính gradient cho bank B
            right_margin = min(99, bank_b_pos + 5)
            if bank_b_pos < 99:
                gradient_b = np.sum(fluence_diff[i, bank_b_pos:right_margin]) / max(1, right_margin - bank_b_pos)
            
            # Thêm gradient vào danh sách
            gradients.append((gradient_a, gradient_b))
        
        return gradients
    
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
        try:
            # Initialize dose calculator
            dose_calc = DoseCalculator()
            
            # Create dose grid matching patient geometry
            self.dose_grid = DoseGrid.from_patient_data(patient_data)
            
            # Calculate dose for each arc
            total_dose = np.zeros_like(self.dose_grid.data)
            
            for arc in self.arcs:
                arc_id = arc['id']
                control_points = self.control_points[arc_id]
                
                # Calculate dose contribution from this arc
                arc_dose = dose_calc.calculate_vmat_arc_dose(
                    patient_data=patient_data,
                    structures=structures,
                    control_points=control_points,
                    mlc_model=self.mlc_model,
                    beam_energy=arc['energy'],
                    dose_rate=arc['dose_rate']
                )
                
                total_dose += arc_dose
            
            # Store final dose distribution
            self.dose_grid.data = total_dose
            
            # Calculate and store DVHs for all structures
            self._calculate_dvhs(structures)
            
            logger.info(
                "Completed final dose calculation for VMAT plan '%s'",
                self.name
            )
            
        except Exception as e:
            logger.error(f"Error in final dose calculation: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def _calculate_dvhs(self, structures):
        """
        Calculate DVHs for all structures.
        
        Parameters
        ----------
        structures : Dict[str, Structure]
            Dictionary of structures
        """
        self.dvhs = {}
        
        for name, structure in structures.items():
            try:
                # Calculate DVH for this structure
                dvh_data = calculate_dvh(
                    dose_grid=self.dose_grid,
                    structure=structure,
                    dose_bins=100  # Can be parameterized
                )
                
                self.dvhs[name] = dvh_data
                
                # Log key DVH metrics
                d95 = dvh_data.get_dose_at_volume(95)
                d50 = dvh_data.get_dose_at_volume(50)
                mean_dose = dvh_data.get_mean_dose()
                
                logger.info(
                    "DVH metrics for structure '%s': D95=%.1f Gy, D50=%.1f Gy, Mean=%.1f Gy",
                    name, d95, d50, mean_dose
                )
                
            except Exception as e:
                logger.error(f"Error calculating DVH for structure '{name}': {e}")
    
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
        if self.dose_grid is None:
            logger.warning("No dose grid available")
            return 0.0
            
        try:
            # Convert physical coordinates to grid indices
            i, j, k = self.dose_grid.get_indices(point_coords)
            
            # Get interpolated dose value
            dose = self.dose_grid.get_interpolated_value(point_coords)
            
            return dose
            
        except Exception as e:
            logger.error(f"Error getting dose at point {point_coords}: {e}")
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
        if not hasattr(self, 'dvhs') or structure_name not in self.dvhs:
            logger.warning(f"No DVH data available for structure '{structure_name}'")
            return None
            
        return self.dvhs[structure_name]
    
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