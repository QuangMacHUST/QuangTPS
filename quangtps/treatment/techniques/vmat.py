
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
from typing import List, Dict, Any, Optional, Tuple, Union

from quangtps.treatment.beams.beam import Beam
from quangtps.treatment.machine.linac import Linac
from quangtps.treatment.fractionation import Fractionation
from quangtps.treatment.mlc.mlc_model import MLCModel

logger = logging.getLogger(__name__)

class VMAT:
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
            Unique ID of the plan. If not provided, a new ID will be generated.
        """
        self.plan_name = plan_name
        self.plan_id = plan_id if plan_id else str(uuid.uuid4())
        
        # Basic plan attributes
        self.technique_type = "VMAT"
        self.description = ""
        self.status = "DRAFT"  # DRAFT, APPROVED, DELIVERED, ARCHIVED
        
        # Treatment machine - VMAT requires a machine with MLC and rotational capabilities
        self.treatment_machine: Optional[Linac] = None
        
        # VMAT-specific attributes
        self.beams: List[Beam] = []
        self.arcs: List[Dict[str, Any]] = []
        self.mlc: Optional[MLCModel] = None
        self.fractionation: Optional[Fractionation] = None
        
        # Optimization parameters
        self.optimization_objectives: List[Dict[str, Any]] = []
        self.optimization_constraints: List[Dict[str, Any]] = []
        self.dose_volume_objectives: List[Dict[str, Any]] = []
        
        # Plan evaluation
        self.plan_quality_metrics: Dict[str, float] = {}
        
        logger.info(f"Created new VMAT plan: {plan_name} (ID: {self.plan_id})")
    
    def add_arc(self, beam: Beam, start_angle: float, stop_angle: float, 
                rotation_direction: str = "CW", collimator_angle: float = 0.0,
                dose_rate: float = 600.0, gantry_speed: float = 6.0) -> str:
        """
        Add an arc to the VMAT plan.
        
        Parameters
        ----------
        beam : Beam
            The beam to use for this arc
        start_angle : float
            Starting angle of the arc in degrees
        stop_angle : float
            Stopping angle of the arc in degrees
        rotation_direction : str, optional
            Direction of rotation ('CW' for clockwise or 'CCW' for counter-clockwise)
        collimator_angle : float, optional
            Angle of the collimator in degrees
        dose_rate : float, optional
            Dose rate in MU/min
        gantry_speed : float, optional
            Gantry rotation speed in deg/sec
            
        Returns
        -------
        str
            ID of the created arc
        """
        arc_id = str(uuid.uuid4())
        
        # Create arc dictionary
        arc = {
            "id": arc_id,
            "beam_id": beam.beam_id,
            "start_angle": start_angle,
            "stop_angle": stop_angle,
            "rotation_direction": rotation_direction,
            "collimator_angle": collimator_angle,
            "dose_rate": dose_rate,
            "gantry_speed": gantry_speed,
            "control_points": self._generate_control_points(start_angle, stop_angle, rotation_direction)
        }
        
        # Add beam to plan if not already added
        if beam not in self.beams:
            self.beams.append(beam)
        
        # Add arc to plan
        self.arcs.append(arc)
        
        logger.info(f"Added arc {arc_id} to VMAT plan {self.plan_name}: "
                   f"{start_angle}° to {stop_angle}° ({rotation_direction})")
        
        return arc_id
    
    def _generate_control_points(self, start_angle: float, stop_angle: float, 
                               rotation_direction: str, num_points: int = 90) -> List[Dict[str, Any]]:
        """
        Generate control points for the arc.
        
        Parameters
        ----------
        start_angle : float
            Starting angle of the arc in degrees
        stop_angle : float
            Stopping angle of the arc in degrees
        rotation_direction : str
            Direction of rotation ('CW' or 'CCW')
        num_points : int, optional
            Number of control points to generate
            
        Returns
        -------
        List[Dict[str, Any]]
            List of control points
        """
        control_points = []
        
        # Handle angle calculation based on rotation direction
        if rotation_direction == "CW":
            if stop_angle < start_angle:
                stop_angle += 360
            angles = np.linspace(start_angle, stop_angle, num_points)
        else:  # CCW
            if start_angle < stop_angle:
                start_angle += 360
            angles = np.linspace(start_angle, stop_angle, num_points)
        
        # Create control points at each angle
        for i, angle in enumerate(angles):
            # Normalize angle to 0-360
            norm_angle = angle % 360
            
            # Create control point
            control_point = {
                "index": i,
                "gantry_angle": norm_angle,
                "cumulative_meterset_weight": i / (num_points - 1),  # Normalized from 0 to 1
                "mlc_positions": None,  # To be filled during optimization
                "jaw_positions": None   # To be filled during optimization
            }
            
            control_points.append(control_point)
        
        return control_points
    
    def set_fractionation(self, fractionation: Fractionation) -> None:
        """
        Set the fractionation scheme for the VMAT plan.
        
        Parameters
        ----------
        fractionation : Fractionation
            Fractionation scheme
        """
        self.fractionation = fractionation
        logger.info(f"Set fractionation for VMAT plan {self.plan_name}: "
                   f"{fractionation.num_fractions} fractions, "
                   f"{fractionation.dose_per_fraction} Gy per fraction")
    
    def set_treatment_machine(self, machine: Linac) -> None:
        """
        Set the treatment machine for the VMAT plan.
        
        Parameters
        ----------
        machine : Linac
            Treatment machine
        
        Raises
        ------
        ValueError
            If the machine does not support VMAT delivery
        """
        # Check if the machine supports VMAT
        if not machine.has_mlc:
            raise ValueError(f"Machine {machine.name} does not have MLC, required for VMAT")
        
        if not machine.supports_vmat:
            raise ValueError(f"Machine {machine.name} does not support VMAT delivery")
        
        self.treatment_machine = machine
        self.mlc = machine.mlc
        logger.info(f"Set treatment machine for VMAT plan {self.plan_name}: {machine.name}")
    
    def add_optimization_objective(self, structure_name: str, objective_type: str, 
                                  dose: float, volume: Optional[float] = None, 
                                  weight: float = 1.0) -> None:
        """
        Add an optimization objective for the VMAT plan.
        
        Parameters
        ----------
        structure_name : str
            Name of the structure
        objective_type : str
            Type of objective (e.g., 'MAX_DOSE', 'MIN_DOSE', 'MAX_DVH', 'MIN_DVH')
        dose : float
            Dose value in Gy
        volume : float, optional
            Volume percentage (0-100) for DVH objectives
        weight : float, optional
            Weight of the objective in optimization
        """
        objective = {
            "structure": structure_name,
            "type": objective_type,
            "dose": dose,
            "volume": volume,
            "weight": weight
        }
        
        self.optimization_objectives.append(objective)
        logger.info(f"Added optimization objective to VMAT plan {self.plan_name}: "
                   f"{objective_type} for {structure_name}")
    
    def calculate_dose(self) -> np.ndarray:
        """
        Calculate the dose distribution for the VMAT plan.
        
        Returns
        -------
        np.ndarray
            3D dose distribution array
        """
        # This would implement a complex dose calculation algorithm
        # For now, we'll return a placeholder
        logger.info(f"Calculating dose for VMAT plan {self.plan_name}")
        return np.zeros((100, 100, 100))  # Placeholder
    
    def optimize_plan(self, max_iterations: int = 100) -> bool:
        """
        Optimize the VMAT plan based on the defined objectives.
        
        Parameters
        ----------
        max_iterations : int, optional
            Maximum number of optimization iterations
            
        Returns
        -------
        bool
            True if optimization was successful, False otherwise
        """
        if not self.treatment_machine:
            logger.error("Cannot optimize VMAT plan: No treatment machine defined")
            return False
        
        if not self.arcs:
            logger.error("Cannot optimize VMAT plan: No arcs defined")
            return False
        
        if not self.optimization_objectives:
            logger.warning("Optimizing VMAT plan without optimization objectives")
        
        # This would implement a complex optimization algorithm
        # For now, we'll just log the process
        logger.info(f"Optimizing VMAT plan {self.plan_name} with {max_iterations} iterations")
        
        # Simulate optimization
        for i in range(max_iterations):
            # Update MLC positions for each control point
            for arc in self.arcs:
                for cp in arc["control_points"]:
                    # In a real implementation, this would adjust MLC positions
                    # based on optimization algorithm
                    pass
            
            # Log progress every 10 iterations
            if i % 10 == 0:
                logger.info(f"VMAT optimization iteration {i}/{max_iterations}")
        
        logger.info(f"Completed VMAT plan optimization for {self.plan_name}")
        return True
    
    def evaluate_plan(self) -> Dict[str, float]:
        """
        Evaluate the quality of the VMAT plan.
        
        Returns
        -------
        Dict[str, float]
            Dictionary of plan quality metrics
        """
        # This would implement various plan quality metrics
        # For now, we'll return placeholder values
        metrics = {
            "conformity_index": 0.95,
            "homogeneity_index": 1.05,
            "gradient_index": 3.2,
            "monitor_units": sum(b.monitor_units for b in self.beams if hasattr(b, 'monitor_units')),
            "treatment_time": self._estimate_treatment_time()
        }
        
        self.plan_quality_metrics = metrics
        logger.info(f"Evaluated VMAT plan {self.plan_name}: CI={metrics['conformity_index']:.2f}, HI={metrics['homogeneity_index']:.2f}")
        
        return metrics
    
    def _estimate_treatment_time(self) -> float:
        """
        Estimate the treatment time for the VMAT plan.
        
        Returns
        -------
        float
            Estimated treatment time in minutes
        """
        total_time = 0.0
        
        # Add time for each arc
        for arc in self.arcs:
            # Calculate arc angle difference
            start = arc["start_angle"]
            stop = arc["stop_angle"]
            
            # Handle wraparound
            if arc["rotation_direction"] == "CW" and stop < start:
                stop += 360
            elif arc["rotation_direction"] == "CCW" and start < stop:
                start += 360
            
            angle_diff = abs(stop - start)
            
            # Calculate time based on gantry speed
            gantry_speed = arc["gantry_speed"]  # deg/sec
            arc_time = angle_diff / gantry_speed / 60  # minutes
            
            total_time += arc_time
        
        # Add setup time (typical)
        setup_time = 2.0  # minutes
        
        return total_time + setup_time
    
    def export_to_dicom(self, output_dir: str) -> bool:
        """
        Export the VMAT plan to DICOM RT Plan format.
        
        Parameters
        ----------
        output_dir : str
            Directory to save the DICOM files
            
        Returns
        -------
        bool
            True if export was successful, False otherwise
        """
        # This would implement DICOM RT Plan export
        # For now, we'll just log the export
        logger.info(f"Exporting VMAT plan {self.plan_name} to DICOM RT Plan in {output_dir}")
        return True
    
    def __str__(self) -> str:
        """Return string representation of the VMAT plan."""
        return f"VMAT Plan: {self.plan_name} (ID: {self.plan_id})"