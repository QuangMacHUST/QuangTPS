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
from quangtps.treatment.machine.treatment_machine import TreatmentMachine
from quangtps.treatment.fractionation import Fractionation
from quangtps.treatment.techniques.technique_interface import BaseTreatmentTechnique, TechniqueCategory

logger = logging.getLogger(__name__)


class DCAT(BaseTreatmentTechnique):
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
        super().__init__(
            name=plan_name,
            technique_id=plan_id,
            category=TechniqueCategory.ADVANCED
        )
        
        # Basic plan attributes
        self.description = ""
        self.status = "DRAFT"  # DRAFT, APPROVED, DELIVERED, ARCHIVED
        
        # DCAT-specific attributes
        self.arcs: List[Dict[str, Any]] = []
        
        # MLC config
        self.margin = 5.0  # mm
        self.leaf_adjustment_frequency = 10  # Degrees
        
        # Plan evaluation
        self.plan_quality_metrics: Dict[str, float] = {}
        
        logger.info(f"Created new DCAT plan: {plan_name} (ID: {self.technique_id})")
    
    def get_name(self) -> str:
        """
        Get the name of the technique.
        
        Returns
        -------
        str
            The name of the technique
        """
        return self.name
    
    def get_id(self) -> str:
        """
        Get the unique identifier of the technique.
        
        Returns
        -------
        str
            The technique ID
        """
        return self.technique_id
    
    def get_category(self) -> TechniqueCategory:
        """
        Get the category of the technique.
        
        Returns
        -------
        TechniqueCategory
            The technique category
        """
        return self.category
    
    def add_beam(self, beam: Beam) -> None:
        """
        Add a beam to the DCAT plan.
        
        Parameters
        ----------
        beam : Beam
            The beam to add
        """
        if beam not in self.beams:
            self.beams.append(beam)
            logger.info(f"Added beam {beam.beam_name} to DCAT plan {self.name}")
    
    def get_beams(self) -> List[Beam]:
        """
        Get the list of beams in the DCAT plan.
        
        Returns
        -------
        List[Beam]
            List of beams in the plan
        """
        return self.beams
    
    def add_arc(self, start_angle: float, stop_angle: float, collimator_angle: float = 0.0,
               couch_angle: float = 0.0, beam_energy: str = "6X") -> str:
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
            
        Returns
        -------
        str
            ID of the created arc
        """
        arc_id = str(uuid.uuid4())
        
        arc = {
            "arc_id": arc_id,
            "start_angle": start_angle,
            "stop_angle": stop_angle,
            "collimator_angle": collimator_angle,
            "couch_angle": couch_angle,
            "beam_energy": beam_energy,
            "control_points": []
        }
        
        self.arcs.append(arc)
        
        logger.info(f"Added arc to DCAT plan {self.name}: "
                   f"start={start_angle}°, stop={stop_angle}°, "
                   f"collimator={collimator_angle}°, couch={couch_angle}°")
        
        return arc_id
    
    def set_fractionation(self, fractionation: Fractionation) -> None:
        """
        Set the fractionation scheme for the DCAT plan.
        
        Parameters
        ----------
        fractionation : Fractionation
            Fractionation scheme
        """
        self.fractionation = fractionation
        logger.info(f"Set fractionation for DCAT plan {self.name}: "
                   f"{fractionation.num_fractions} fractions, "
                   f"{fractionation.dose_per_fraction} Gy per fraction")
    
    def set_machine(self, machine: TreatmentMachine) -> None:
        """
        Set the treatment machine for the DCAT plan.
        
        Parameters
        ----------
        machine : TreatmentMachine
            Treatment machine
            
        Raises
        ------
        ValueError
            If the machine does not support DCAT
        """
        if not isinstance(machine, Linac):
            raise ValueError(f"DCAT requires a Linac treatment machine, got {type(machine).__name__}")
            
        if not machine.supports_conformal_arc:
            raise ValueError(f"Machine {machine.name} does not support DCAT")
        
        self.machine = machine
        logger.info(f"Set treatment machine for DCAT plan {self.name}: {machine.name}")
    
    def set_margin(self, margin: float) -> None:
        """
        Set the margin around the target volume for MLC shaping.
        
        Parameters
        ----------
        margin : float
            Margin in mm
        """
        self.margin = margin
        logger.info(f"Set margin for DCAT plan {self.name}: {margin} mm")
    
    def set_leaf_adjustment_frequency(self, frequency: int) -> None:
        """
        Set the frequency of MLC leaf adjustments during arc delivery.
        
        Parameters
        ----------
        frequency : int
            Frequency in degrees (e.g., adjust MLC every 10 degrees)
        """
        self.leaf_adjustment_frequency = frequency
        logger.info(f"Set leaf adjustment frequency for DCAT plan {self.name}: {frequency}°")
    
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
            
            logger.info(f"Generated {len(control_points)} control points for arc {i+1} in DCAT plan {self.name}")
    
    def _create_dummy_mlc_positions(self) -> List[Tuple[float, float]]:
        """
        Create dummy MLC positions for demonstration purposes.
        
        Returns
        -------
        List[Tuple[float, float]]
            List of (left, right) positions for each MLC leaf pair
        """
        # Simulate a 60-leaf MLC (30 pairs)
        num_leaf_pairs = 30
        field_size = 10.0  # cm
        
        # Create a leaf pattern that approximates a circle
        mlc_positions = []
        for i in range(num_leaf_pairs):
            # Distance from central leaf pair (leaf pair 15)
            distance = abs(i - num_leaf_pairs // 2) / (num_leaf_pairs // 2)
            
            # Field width decreases as we move away from central axis
            width = field_size * np.sqrt(1 - distance ** 2)
            
            # Center the opening
            left = -width / 2
            right = width / 2
            
            mlc_positions.append((left, right))
        
        return mlc_positions
    
    def calculate_dose(self) -> Dict[str, np.ndarray]:
        """
        Calculate the dose distribution for the DCAT plan.
        
        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary with dose array and metadata
        """
        # This would implement a dose calculation algorithm for DCAT
        # For now, return a dummy result
        logger.info(f"Calculating dose for DCAT plan {self.name}")
        
        # Create dummy 3D dose array (100x100x100)
        dose_array = np.zeros((100, 100, 100))
        
        # Add some dummy dose values
        for i, arc in enumerate(self.arcs):
            center = 50
            radius = 10
            
            # Create a sphere of dose
            for x in range(center - radius, center + radius):
                for y in range(center - radius, center + radius):
                    for z in range(center - radius, center + radius):
                        if (x - center) ** 2 + (y - center) ** 2 + (z - center) ** 2 <= radius ** 2:
                            # Distance from center determines dose value
                            dist = np.sqrt((x - center) ** 2 + (y - center) ** 2 + (z - center) ** 2)
                            dose_array[x, y, z] += (1 - dist / radius) * 70.0  # Gy
        
        result = {
            "dose_array": dose_array,
            "dimensions": (100, 100, 100),
            "spacing": (0.3, 0.3, 0.3),  # cm
            "origin": (-15, -15, -15)    # cm
        }
        
        return result
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the DCAT plan to a dictionary.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary representation of the plan
        """
        # Start with the base technique dictionary
        result = super().to_dict()
        
        # Add DCAT-specific attributes
        result.update({
            "description": self.description,
            "status": self.status,
            "arcs": self.arcs,
            "margin": self.margin,
            "leaf_adjustment_frequency": self.leaf_adjustment_frequency,
            "plan_quality_metrics": self.plan_quality_metrics
        })
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DCAT':
        """
        Create a DCAT plan from a dictionary.
        
        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary with plan data
            
        Returns
        -------
        DCAT
            DCAT instance
        """
        plan = cls(
            plan_name=data["name"],
            plan_id=data["technique_id"]
        )
        
        # Restore basic attributes
        plan.description = data.get("description", "")
        plan.status = data.get("status", "DRAFT")
        
        # Restore DCAT-specific attributes
        if "arcs" in data:
            plan.arcs = data["arcs"]
        
        if "margin" in data:
            plan.margin = data["margin"]
        
        if "leaf_adjustment_frequency" in data:
            plan.leaf_adjustment_frequency = data["leaf_adjustment_frequency"]
        
        if "plan_quality_metrics" in data:
            plan.plan_quality_metrics = data["plan_quality_metrics"]
        
        # Restore common components (machine, fractionation, beams) if present
        if "machine" in data and data["machine"]:
            from quangtps.treatment.machine.machine_factory import MachineFactory
            machine_factory = MachineFactory()
            machine = machine_factory.create_from_dict(data["machine"])
            plan.set_machine(machine)
        
        if "fractionation" in data and data["fractionation"]:
            fractionation = Fractionation.from_dict(data["fractionation"])
            plan.set_fractionation(fractionation)
        
        if "beams" in data and data["beams"]:
            from quangtps.treatment.beams.beam_factory import BeamFactory
            beam_factory = BeamFactory()
            for beam_data in data["beams"]:
                beam = beam_factory.create_from_dict(beam_data)
                plan.add_beam(beam)
        
        return plan


# Ensure class is exported correctly
__all__ = ['DCAT']