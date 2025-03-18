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
    
    def set_optimization_parameters(self, iterations: int, threshold: float, smoothing: float):
        """
        Set optimization parameters for the VMAT plan.
        
        Parameters
        ----------
        iterations : int
            Maximum number of iterations
        threshold : float
            Convergence threshold
        smoothing : float
            Smoothing factor for fluence maps and control points
        """
        self.optimization_iterations = iterations
        self.convergence_threshold = threshold
        self.smoothing_factor = smoothing
        
        # Using lazy % formatting for logging
        logger.info(
            "Set optimization parameters for VMAT plan '%s': iterations=%d, threshold=%.6f, smoothing=%.2f",
            self.name, iterations, threshold, smoothing
        )
    
    def optimize(self):
        """
        Optimize the VMAT plan.
        
        This is a complex process that involves:
        - Initial fluence map optimization
        - Converting fluence maps to MLC sequences
        - Optimizing MLC sequences for deliverability
        - Calculating final dose distribution
        
        Returns
        -------
        bool
            True if optimization was successful, False otherwise
        """
        if not self.arcs:
            # Using lazy % formatting for logging
            logger.warning(
                "Cannot optimize VMAT plan '%s': No arcs defined",
                self.name
            )
            return False
        
        if not self.mlc_model:
            # Using lazy % formatting for logging
            logger.warning(
                "Cannot optimize VMAT plan '%s': No MLC model defined",
                self.name
            )
            return False
        
        if not self.dose_objectives:
            # Using lazy % formatting for logging
            logger.warning(
                "Cannot optimize VMAT plan '%s': No optimization objectives defined",
                self.name
            )
            return False
        
        # Using lazy % formatting for logging
        logger.info(
            "Starting optimization for VMAT plan '%s' with %d arcs and %d objectives",
            self.name, len(self.arcs), len(self.dose_objectives)
        )
        
        # For demonstration purposes, we'll create some simulated control points
        for arc in self.arcs:
            arc_id = arc["id"]
            start_angle = arc["start_angle"]
            stop_angle = arc["stop_angle"]
            direction = arc["rotation_direction"]
            
            # Determine angle increment based on direction
            if direction == "CW":
                if stop_angle < start_angle:
                    stop_angle += 360.0
                angle_diff = stop_angle - start_angle
            else:  # CCW
                if stop_angle > start_angle:
                    start_angle += 360.0
                angle_diff = start_angle - stop_angle
                
            # Create simulated control points
            num_control_points = 20  # Example: 20 control points per arc
            angle_step = angle_diff / (num_control_points - 1)
            
            # Clear existing control points
            self.control_points[arc_id] = []
            
            for i in range(num_control_points):
                # Calculate gantry angle for this control point
                if direction == "CW":
                    angle = (start_angle + i * angle_step) % 360.0
                else:  # CCW
                    angle = (start_angle - i * angle_step) % 360.0
                
                # Create simulated MLC positions (just for demonstration)
                num_leaves = 60  # Assuming 60 leaf pairs
                mlc_positions = []
                
                for j in range(num_leaves):
                    # Simple sinusoidal pattern for demonstration
                    center = 0.0
                    width = 5.0 + 5.0 * np.sin(i * np.pi / 10.0 + j * np.pi / 30.0)
                    leaf_a = center - width / 2.0
                    leaf_b = center + width / 2.0
                    mlc_positions.append([leaf_a, leaf_b])
                
                # Calculate cumulative meterset
                meterset = i / (num_control_points - 1)
                
                # Add the control point
                self.add_control_point(arc_id, angle, mlc_positions, meterset)
        
        # Using lazy % formatting for logging
        logger.info(
            "Completed optimization for VMAT plan '%s'. Created %d control points across %d arcs.",
            self.name, 
            sum(len(control_points) for control_points in self.control_points.values()),
            len(self.arcs)
        )
        
        return True
    
    def calculate_delivery_time(self):
        """
        Calculate the estimated delivery time for the VMAT plan.
        
        Returns
        -------
        float
            Estimated delivery time in minutes
        """
        if not self.arcs:
            # Using lazy % formatting for logging
            logger.warning(
                "Cannot calculate delivery time for VMAT plan '%s': No arcs defined",
                self.name
            )
            return 0.0
        
        total_time = 0.0
        
        # Setup time
        setup_time = 5.0  # minutes
        total_time += setup_time
        
        # Time for each arc
        for arc in self.arcs:
            arc_id = arc["id"]
            start_angle = arc["start_angle"]
            stop_angle = arc["stop_angle"]
            direction = arc["rotation_direction"]
            dose_rate = arc["dose_rate"]  # MU/min
            
            # Calculate arc angle span
            if direction == "CW":
                if stop_angle < start_angle:
                    stop_angle += 360.0
                angle_span = stop_angle - start_angle
            else:  # CCW
                if stop_angle > start_angle:
                    start_angle += 360.0
                angle_span = start_angle - stop_angle
            
            # Estimate MUs for this arc (based on control points if available)
            total_mu = 0.0
            if arc_id in self.control_points and self.control_points[arc_id]:
                # Get the last control point's cumulative meterset
                last_meterset = self.control_points[arc_id][-1]["cumulative_meterset"]
                # Assume a typical VMAT arc uses 200-600 MU
                total_mu = last_meterset * 400.0  # Just an estimate
            else:
                # Default estimate if no control points
                total_mu = 400.0
            
            # Calculate arc time based on gantry rotation and dose rate
            # VMAT gantry typically rotates at 4-6 degrees per second
            gantry_speed = 5.0  # degrees per second
            rotation_time = angle_span / gantry_speed / 60.0  # minutes
            
            # Delivery time is the maximum of rotation time and MU delivery time
            mu_delivery_time = total_mu / dose_rate  # minutes
            
            # Add the arc delivery time to total time
            total_time += max(rotation_time, mu_delivery_time)
        
        # Time between arcs if multiple arcs
        if len(self.arcs) > 1:
            between_arc_time = 0.5  # minutes
            total_time += between_arc_time * (len(self.arcs) - 1)
        
        # Using lazy % formatting for logging
        logger.info(
            "Estimated delivery time for VMAT plan '%s': %.2f minutes",
            self.name, total_time
        )
        
        return total_time
    
    def to_dict(self):
        """
        Convert the VMAT plan to a dictionary.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary containing all VMAT plan information
        """
        result = super().to_dict()
        
        # Add VMAT-specific information
        result.update({
            'technique_type': 'VMAT',
            'arcs': self.arcs,
            'optimization_parameters': {
                'iterations': self.optimization_iterations,
                'convergence_threshold': self.convergence_threshold,
                'smoothing_factor': self.smoothing_factor
            },
            'dose_objectives': self.dose_objectives,
            'constraints': self.constraints
        })
        
        # Don't include full control points as they can be large
        # Just include summary information
        control_points_summary = {}
        for arc_id, points in self.control_points.items():
            control_points_summary[arc_id] = {
                'count': len(points),
                'has_data': len(points) > 0
            }
        
        result['control_points_summary'] = control_points_summary
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """
        Create a VMAT plan from a dictionary.
        
        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary containing VMAT plan data
            
        Returns
        -------
        VMAT
            VMAT plan object
        
        Raises
        ------
        ValueError
            If the dictionary does not contain valid VMAT data
        """
        # Check for valid VMAT data
        if data.get('technique_type') != 'VMAT':
            raise ValueError("Dictionary does not contain valid VMAT data")
        
        # Create basic VMAT plan
        vmat = cls(
            plan_name=data.get('name', 'VMAT Plan'),
            plan_id=data.get('id')
        )
        
        # Set optimization parameters
        if 'optimization_parameters' in data:
            params = data['optimization_parameters']
            vmat.set_optimization_parameters(
                params.get('iterations', 100),
                params.get('convergence_threshold', 0.001),
                params.get('smoothing_factor', 0.5)
            )
        
        # Add arcs
        if 'arcs' in data and isinstance(data['arcs'], list):
            for arc in data['arcs']:
                # Rather than using add_arc which generates a new ID, we want to preserve the original IDs
                vmat.arcs.append(arc)
                vmat.control_points[arc['id']] = []
                
                # Using lazy % formatting for logging
                logger.info(
                    "Restored arc '%s' from dictionary to VMAT plan '%s'",
                    arc.get('name', 'Unknown'), vmat.name
                )
        
        # Add objectives and constraints
        if 'dose_objectives' in data and isinstance(data['dose_objectives'], list):
            for obj in data['dose_objectives']:
                vmat.add_objective(
                    obj.get('structure', ''),
                    obj.get('type', ''),
                    obj.get('dose', 0.0),
                    obj.get('volume'),
                    obj.get('weight', 1.0)
                )
                
        if 'constraints' in data and isinstance(data['constraints'], list):
            for con in data['constraints']:
                vmat.add_constraint(
                    con.get('structure', ''),
                    con.get('type', ''),
                    con.get('dose', 0.0),
                    con.get('volume')
                )
        
        # Note: Control points are not restored from dictionary as they can be large
        # They would need to be regenerated via optimization or loaded from a separate source
        
        return vmat


# Ensure the VMAT class is properly exported
__all__ = ['VMAT']