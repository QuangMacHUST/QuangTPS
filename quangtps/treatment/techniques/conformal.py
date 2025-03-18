#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module for 3D Conformal Radiation Therapy (3D-CRT) technique.

This module provides classes and methods to define and manage 3D conformal radiation therapy plans.
"""

import logging
from typing import List, Dict, Any, Optional

from quangtps.treatment.beams.beam import Beam
from quangtps.treatment.beams.beam_modifiers import Block, Wedge
from quangtps.treatment.machine.linac import Linac
from quangtps.treatment.machine.treatment_machine import TreatmentMachine
from quangtps.treatment.fractionation import Fractionation
from quangtps.treatment.techniques.technique_interface import BaseTreatmentTechnique, TechniqueCategory

logger = logging.getLogger(__name__)

class Conformal3DRT(BaseTreatmentTechnique):
    """
    Class representing a 3D Conformal Radiation Therapy plan.
    
    3D-CRT is a radiation therapy technique that uses 3D imaging to precisely conform
    the radiation dose to the shape of the tumor while minimizing dose to surrounding
    healthy tissues.
    """
    
    def __init__(self, name: str, technique_id: Optional[str] = None):
        """
        Initialize a 3D-CRT plan.
        
        Parameters
        ----------
        name : str
            Name of the 3D-CRT plan
        technique_id : str, optional
            Unique ID of the plan. If not provided, a new ID will be generated.
        """
        super().__init__(
            name=name,
            technique_id=technique_id,
            category=TechniqueCategory.STANDARD
        )
        
        # 3D-CRT specific attributes
        self.description = ""
        self.status = "DRAFT"  # DRAFT, APPROVED, DELIVERED, ARCHIVED
        
        # Target structures
        self.target_structures: List[str] = []  # List of ROI IDs
        
        # Critical structures
        self.critical_structures: List[str] = []  # List of ROI IDs
        
        # Plan evaluation metrics
        self.evaluation_metrics: Dict[str, Any] = {}
        
        # Additional information
        self.metadata: Dict[str, Any] = {}
        
        logger.info("Initialized 3D-CRT plan: %s (ID: %s)", name, self.technique_id)
    
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
        Add a beam to the 3D-CRT plan.
        
        Parameters
        ----------
        beam : Beam
            The beam to add to the plan
        """
        self.beams.append(beam)
        logger.info("Added beam '%s' to plan '%s'", beam.beam_name, self.name)
    
    def get_beams(self) -> List[Beam]:
        """
        Get all beams in the plan.
        
        Returns
        -------
        List[Beam]
            List of beams in the plan
        """
        return self.beams
    
    def remove_beam(self, beam_id: str) -> bool:
        """
        Remove a beam from the 3D-CRT plan.
        
        Parameters
        ----------
        beam_id : str
            ID of the beam to remove
            
        Returns
        -------
        bool
            True if removal was successful, False if beam was not found
        """
        for i, beam in enumerate(self.beams):
            if beam.beam_id == beam_id:
                self.beams.pop(i)
                logger.info("Removed beam with ID '%s' from plan '%s'", beam_id, self.name)
                return True
        logger.warning("Beam with ID '%s' not found in plan '%s'", beam_id, self.name)
        return False
    
    def set_fractionation(self, fractionation: Fractionation) -> None:
        """
        Set the fractionation for the 3D-CRT plan.
        
        Parameters
        ----------
        fractionation : Fractionation
            The fractionation scheme
        """
        self.fractionation = fractionation
        logger.info("Set fractionation to %s Gy in %s fractions for plan '%s'", 
                   fractionation.total_dose, fractionation.num_fractions, self.name)
    
    def set_prescription(self, total_dose: float, num_fractions: int) -> None:
        """
        Set the prescription for the 3D-CRT plan.
        
        Parameters
        ----------
        total_dose : float
            Total prescribed dose in Gy
        num_fractions : int
            Number of fractions
        """
        self.fractionation = Fractionation(total_dose, num_fractions)
        logger.info("Set prescription to %s Gy in %s fractions for plan '%s'", 
                   total_dose, num_fractions, self.name)
    
    def set_machine(self, machine: TreatmentMachine) -> None:
        """
        Set the treatment machine for the 3D-CRT plan.
        
        Parameters
        ----------
        machine : TreatmentMachine
            The treatment machine to use
        """
        if not isinstance(machine, Linac):
            raise ValueError("3D-CRT requires a Linac treatment machine")
        
        self.machine = machine
        logger.info("Set treatment machine to '%s' for plan '%s'", machine.name, self.name)
    
    def add_target_structure(self, structure_id: str) -> None:
        """
        Add a target structure to the 3D-CRT plan.
        
        Parameters
        ----------
        structure_id : str
            ID of the target structure
        """
        if structure_id not in self.target_structures:
            self.target_structures.append(structure_id)
            logger.info("Added target structure '%s' to plan '%s'", structure_id, self.name)
    
    def add_critical_structure(self, structure_id: str) -> None:
        """
        Add a critical structure to the 3D-CRT plan.
        
        Parameters
        ----------
        structure_id : str
            ID of the critical structure
        """
        if structure_id not in self.critical_structures:
            self.critical_structures.append(structure_id)
            logger.info("Added critical structure '%s' to plan '%s'", structure_id, self.name)
    
    def create_conformal_block(self, beam: Beam, target_structure_id: str, margin: float = 0.5) -> Block:
        """
        Create a conformal block for a beam based on a target structure.
        
        Parameters
        ----------
        beam : Beam
            The beam to create the block for
        target_structure_id : str
            ID of the target structure to conform to
        margin : float, optional
            Margin around the target structure in cm
            
        Returns
        -------
        Block
            The created conformal block
        """
        # In a real implementation, this would project the structure onto the beam's eye view
        # and create a block with appropriate margins
        
        # For demonstration, create a simple rectangular block
        block = Block(f"Conformal Block for {target_structure_id}")
        
        # Add some dummy contour points (in a real implementation, these would be derived from the structure)
        block.set_contour([
            (-5.0, -5.0),
            (5.0, -5.0),
            (5.0, 5.0),
            (-5.0, 5.0),
            (-5.0, -5.0)
        ])
        
        # Add the block to the beam
        beam.add_modifier(block)
        
        logger.info("Created conformal block for beam '%s' based on structure '%s'", 
                   beam.beam_name, target_structure_id)
        return block
    
    def calculate_beam_weights(self, method: str = "equal") -> Dict[str, float]:
        """
        Calculate beam weights for the 3D-CRT plan.
        
        Parameters
        ----------
        method : str, optional
            Method to use for weight calculation, one of:
            - 'equal': Equal weights to all beams
            - 'inverse': Weights inversely proportional to path length
            - 'custom': Custom weighting (requires additional parameters)
            
        Returns
        -------
        Dict[str, float]
            Dictionary mapping beam IDs to weights
        """
        weights = {}
        num_beams = len(self.beams)
        
        if num_beams == 0:
            logger.warning("No beams in plan '%s', cannot calculate weights", self.name)
            return weights
        
        if method == "equal":
            # Equal weights to all beams
            weight = 1.0 / num_beams
            for beam in self.beams:
                weights[beam.beam_id] = weight
                beam.set_weight(weight)
        
        elif method == "inverse":
            # Weights inversely proportional to path length (simplified example)
            # In a real implementation, this would use actual path length through the patient
            total_inverse = 0
            beam_inverses = []
            
            # Calculate inverse values
            for beam in self.beams:
                # Dummy path length based on beam angle for demonstration
                path_length = 20.0  # Default path length in cm
                inverse = 1.0 / path_length
                total_inverse += inverse
                beam_inverses.append(inverse)
            
            # Normalize weights
            for i, beam in enumerate(self.beams):
                weight = beam_inverses[i] / total_inverse
                weights[beam.beam_id] = weight
                beam.set_weight(weight)
        
        else:
            logger.warning("Unsupported weight calculation method: %s", method)
            # Default to equal weights
            weight = 1.0 / num_beams
            for beam in self.beams:
                weights[beam.beam_id] = weight
                beam.set_weight(weight)
        
        logger.info("Calculated beam weights for plan '%s' using method '%s'", self.name, method)
        return weights
    
    def add_wedge(self, beam: Beam, wedge: Wedge) -> None:
        """
        Add a wedge to a beam in the 3D-CRT plan.
        
        Parameters
        ----------
        beam : Beam
            The beam to add the wedge to
        wedge : Wedge
            The wedge to add
        """
        beam.add_modifier(wedge)
        logger.info("Added wedge '%s' to beam '%s' in plan '%s'", 
                   wedge.name, beam.beam_name, self.name)
    
    def optimize_beam_angles(self, num_beams: int = 3, method: str = "equiangular") -> List[float]:
        """
        Optimize beam angles for the 3D-CRT plan.
        
        Parameters
        ----------
        num_beams : int, optional
            Number of beams to use
        method : str, optional
            Method to use for angle optimization:
            - 'equiangular': Evenly spaced beams
            - 'custom': Custom angle optimization (requires additional parameters)
            
        Returns
        -------
        List[float]
            List of optimized beam angles
        """
        angles = []
        
        if method == "equiangular":
            # Create evenly spaced beams
            start_angle = 0
            angle_step = 360 / num_beams
            
            for i in range(num_beams):
                angle = (start_angle + i * angle_step) % 360
                angles.append(angle)
                
                # Create a new beam with this angle if requested
                beam_name = f"Beam_{i+1}"
                beam = Beam(beam_name=beam_name)
                beam.geometry.gantry_angle = angle
                self.add_beam(beam)
        
        else:
            logger.warning("Unsupported beam angle optimization method: %s", method)
        
        logger.info("Optimized beam angles for plan '%s' using method '%s'", self.name, method)
        return angles
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the 3D-CRT plan to a dictionary.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary representation of the plan
        """
        return {
            "id": self.technique_id,
            "name": self.name,
            "type": "3D-CRT",
            "category": self.category.value,
            "description": self.description,
            "status": self.status,
            "beams": [beam.to_dict() for beam in self.beams],
            "fractionation": self.fractionation.to_dict() if self.fractionation else None,
            "machine": self.machine.name if self.machine else None,
            "target_structures": self.target_structures,
            "critical_structures": self.critical_structures,
            "evaluation_metrics": self.evaluation_metrics,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Conformal3DRT':
        """
        Create a 3D-CRT plan from a dictionary.
        
        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary containing plan data
            
        Returns
        -------
        Conformal3DRT
            The created 3D-CRT plan
        """
        plan = cls(name=data["name"], technique_id=data["id"])
        
        plan.description = data.get("description", "")
        plan.status = data.get("status", "DRAFT")
        
        # Load beams
        from quangtps.treatment.beams.beam import Beam
        for beam_data in data.get("beams", []):
            beam = Beam.from_dict(beam_data)
            plan.beams.append(beam)
        
        # Load fractionation
        if "fractionation" in data and data["fractionation"]:
            from quangtps.treatment.fractionation import Fractionation
            plan.fractionation = Fractionation.from_dict(data["fractionation"])
        
        # Load structures
        plan.target_structures = data.get("target_structures", [])
        plan.critical_structures = data.get("critical_structures", [])
        
        # Load other attributes
        plan.evaluation_metrics = data.get("evaluation_metrics", {})
        plan.metadata = data.get("metadata", {})
        
        return plan


# Export the class
__all__ = ['Conformal3DRT']