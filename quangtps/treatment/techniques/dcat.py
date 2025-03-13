
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module for Dynamic Conformal Arc Therapy (DCAT) technique.

This module provides classes and methods to define and manage DCAT plans.
"""

import uuid
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

from quangtps.treatment.beams.beam import Beam
from quangtps.treatment.machine.linac import Linac
from quangtps.treatment.fractionation import Fractionation

logger = logging.getLogger(__name__)


class DCAT:
    """
    Class representing a Dynamic Conformal Arc Therapy (DCAT) plan.
    
    DCAT is a radiotherapy technique that uses the multi-leaf collimator (MLC)
    to dynamically conform to the target volume as the gantry rotates around
    the patient. It is primarily used for small, spherical targets.
    """
    
    def __init__(self, plan_name: str, plan_id: Optional[str] = None):
        """
        Initialize a DCAT plan.
        
        Parameters
        ----------
        plan_name : str
            Name of the DCAT plan
        plan_id : str, optional
            Unique ID of the plan. If not provided, a new ID will be generated.
        """
        self.plan_name = plan_name
        self.plan_id = plan_id if plan_id else str(uuid.uuid4())
        
        # Basic plan attributes
        self.description = ""
        self.status = "DRAFT"  # DRAFT, APPROVED, DELIVERED, ARCHIVED
        
        # Treatment machine
        self.treatment_machine: Optional[Linac] = None
        
        # DCAT-specific attributes
        self.arcs: List[Dict[str, Any]] = []
        self.fractionation: Optional[Fractionation] = None
        
        # MLC config
        self.margin = 5.0  # mm
        self.leaf_adjustment_frequency = 10  # Degrees
        
        # Plan evaluation
        self.plan_quality_metrics: Dict[str, float] = {}
        
        logger.info(f"Created new DCAT plan: {plan_name} (ID: {self.plan_id})")
    
    def add_arc(self, start_angle: float, stop_angle: float, collimator_angle: float = 0.0,
               couch_angle: float = 0.0, beam_energy: str = "6X") -> None:
        """
        Add an arc to the DCAT plan.
        
        Parameters
        ----------
        start_angle : float
            Start angle of the arc in degrees
        stop_angle : float
            Stop angle of the arc in degrees
        collimator_angle : float, optional
            Collimator angle in degrees
        couch_angle : float, optional
            Couch angle in degrees
        beam_energy : str, optional
            Beam energy (e.g. "6X", "10X")
        """
        arc = {
            "arc_id": str(uuid.uuid4()),
            "start_angle": start_angle,
            "stop_angle": stop_angle,
            "collimator_angle": collimator_angle,
            "couch_angle": couch_angle,
            "beam_energy": beam_energy,
            "control_points": []
        }
        
        self.arcs.append(arc)
        
        logger.info(f"Added arc to DCAT plan {self.plan_name}: "
                   f"start={start_angle}°, stop={stop_angle}°, "
                   f"collimator={collimator_angle}°, couch={couch_angle}°")
    
    def set_fractionation(self, fractionation: Fractionation) -> None:
        """
        Set the fractionation scheme for the DCAT plan.
        
        Parameters
        ----------
        fractionation : Fractionation
            Fractionation scheme
        """
        self.fractionation = fractionation
        logger.info(f"Set fractionation for DCAT plan {self.plan_name}: "
                   f"{fractionation.num_fractions} fractions, "
                   f"{fractionation.dose_per_fraction} Gy per fraction")
    
    def set_treatment_machine(self, machine: Linac) -> None:
        """
        Set the treatment machine for the DCAT plan.
        
        Parameters
        ----------
        machine : Linac
            Treatment machine
            
        Raises
        ------
        ValueError
            If the machine does not support DCAT
        """
        if not machine.supports_conformal_arc:
            raise ValueError(f"Machine {machine.name} does not support DCAT")
        
        self.treatment_machine = machine
        logger.info(f"Set treatment machine for DCAT plan {self.plan_name}: {machine.name}")
    
    def set_margin(self, margin: float) -> None:
        """
        Set the margin around the target volume for MLC shaping.
        
        Parameters
        ----------
        margin : float
            Margin in mm
        """
        self.margin = margin
        logger.info(f"Set margin for DCAT plan {self.plan_name}: {margin} mm")
    
    def set_leaf_adjustment_frequency(self, frequency: int) -> None:
        """
        Set the frequency of MLC leaf adjustments during arc delivery.
        
        Parameters
        ----------
        frequency : int
            Frequency in degrees (e.g., adjust MLC every 10 degrees)
        """
        self.leaf_adjustment_frequency = frequency
        logger.info(f"Set leaf adjustment frequency for DCAT plan {self.plan_name}: {frequency}°")
    
    def generate_control_points(self) -> None:
        """
        Generate control points for all arcs in the DCAT plan.
        
        This method calculates the MLC positions at various gantry angles according
        to the beam's eye view of the target.
        """
        for i, arc in enumerate(self.arcs):
            start_angle = arc["start_angle"]
            stop_angle = arc["stop_angle"]
            
            # Calculate step size based on leaf adjustment frequency
            step = self.leaf_adjustment_frequency
            
            # Determine direction (clockwise or counterclockwise)
            if start_angle <= stop_angle:
                angles = list(range(int(start_angle), int(stop_angle) + 1, step))
            else:
                angles = list(range(int(start_angle), int(stop_angle) - 1, -step))
            
            # Ensure stop angle is included
            if angles[-1] != stop_angle:
                angles.append(stop_angle)
            
            # Generate control points
            control_points = []
            for j, angle in enumerate(angles):
                # In a real implementation, this would calculate the MLC positions
                # based on the projection of the target at this gantry angle
                # For now, we'll create dummy MLC positions
                mlc_positions = self._create_dummy_mlc_positions()
                
                control_point = {
                    "index": j,
                    "gantry_angle": angle,
                    "cumulative_meterset_weight": j / (len(angles) - 1),
                    "mlc_positions": mlc_positions
                }
                
                control_points.append(control_point)
            
            self.arcs[i]["control_points"] = control_points
            
            logger.info(f"Generated {len(control_points)} control points for arc {i+1} in DCAT plan {self.plan_name}")
    
    def _create_dummy_mlc_positions(self) -> Dict[str, List[float]]:
        """
        Create dummy MLC positions for testing.
        
        Returns
        -------
        Dict[str, List[float]]
            Dictionary with leaf bank positions
        """
        # Assuming a standard MLC with 60 leaf pairs
        num_leaves = 60
        
        # Create simple leaf positions that form a rectangular aperture
        bank_a = [-20.0] * num_leaves
        bank_b = [20.0] * num_leaves
        
        return {
            "bank_a": bank_a,
            "bank_b": bank_b
        }
    
    def calculate_dose(self) -> np.ndarray:
        """
        Calculate the dose distribution for the DCAT plan.
        
        Returns
        -------
        np.ndarray
            3D dose distribution array
        """
        # This would implement a dose calculation algorithm
        # For now, we'll return a placeholder
        logger.info(f"Calculating dose for DCAT plan {self.plan_name}")
        return np.zeros((100, 100, 100))  # Placeholder
    
    def evaluate_plan(self) -> Dict[str, float]:
        """
        Evaluate the quality of the DCAT plan.
        
        Returns
        -------
        Dict[str, float]
            Dictionary of plan quality metrics
        """
        # Calculate plan quality metrics
        metrics = {
            "conformity_index": 0.9,         # Placeholder value
            "gradient_index": 3.0,           # Placeholder value
            "coverage": 0.98,                # Placeholder value (98%)
            "maximum_dose": 105.0,           # % of prescription dose
            "minimum_dose": 95.0             # % of prescription dose
        }
        
        self.plan_quality_metrics = metrics
        
        logger.info(f"Evaluated DCAT plan {self.plan_name}: "
                   f"coverage={metrics['coverage']:.2f}, "
                   f"CI={metrics['conformity_index']:.2f}")
        
        return metrics
    
    def export_to_dicom(self, output_dir: str) -> bool:
        """
        Export the DCAT plan to DICOM RT Plan format.
        
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
        logger.info(f"Exporting DCAT plan {self.plan_name} to DICOM RT Plan in {output_dir}")
        return True
    
    def __str__(self) -> str:
        """Return string representation of the DCAT plan."""
        return f"DCAT Plan: {self.plan_name} (ID: {self.plan_id})"