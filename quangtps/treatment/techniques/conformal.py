#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module for 3D Conformal Radiation Therapy (3D-CRT) technique.

This module provides classes and methods to define and manage 3D conformal radiation therapy plans.
"""

import uuid
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union

from quangtps.treatment.beams.beam import Beam
from quangtps.treatment.beams.beam_modifiers import Block, Wedge
from quangtps.treatment.machine.linac import Linac
from quangtps.treatment.fractionation import Fractionation

logger = logging.getLogger(__name__)

class Conformal3DRT:
    """
    Class representing a 3D Conformal Radiation Therapy plan.
    
    3D-CRT is a radiation therapy technique that uses 3D imaging to precisely conform
    the radiation dose to the shape of the tumor while minimizing dose to surrounding
    healthy tissues.
    """
    
    def __init__(self, plan_name: str, plan_id: Optional[str] = None):
        """
        Initialize a 3D-CRT plan.
        
        Parameters
        ----------
        plan_name : str
            Name of the 3D-CRT plan
        plan_id : str, optional
            Unique ID of the plan. If not provided, a new ID will be generated.
        """
        self.plan_name = plan_name
        self.plan_id = plan_id if plan_id else str(uuid.uuid4())
        
        # Basic plan attributes
        self.technique_type = "3D-CRT"
        self.description = ""
        self.status = "DRAFT"  # DRAFT, APPROVED, DELIVERED, ARCHIVED
        
        # Treatment machine
        self.treatment_machine: Optional[Linac] = None
        
        # Beams in the plan
        self.beams: List[Beam] = []
        
        # Prescription
        self.prescription_dose = 0.0  # Total dose in Gy
        self.prescription_fractions = 0  # Number of fractions
        self.fractionation: Optional[Fractionation] = None
        
        # Target structures
        self.target_structures = []  # List of ROI IDs
        
        # Critical structures
        self.critical_structures = []  # List of ROI IDs
        
        # Plan evaluation metrics
        self.evaluation_metrics = {}
        
        # Additional information
        self.metadata = {}
    
    def add_beam(self, beam: Beam):
        """
        Add a beam to the 3D-CRT plan.
        
        Parameters
        ----------
        beam : Beam
            The beam to add to the plan
        """
        self.beams.append(beam)
        logger.info(f"Added beam '{beam.beam_name}' to plan '{self.plan_name}'")
    
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
                logger.info(f"Removed beam with ID '{beam_id}' from plan '{self.plan_name}'")
                return True
        logger.warning(f"Beam with ID '{beam_id}' not found in plan '{self.plan_name}'")
        return False
    
    def set_prescription(self, total_dose: float, num_fractions: int):
        """
        Set the prescription for the 3D-CRT plan.
        
        Parameters
        ----------
        total_dose : float
            Total prescribed dose in Gy
        num_fractions : int
            Number of fractions
        """
        self.prescription_dose = total_dose
        self.prescription_fractions = num_fractions
        self.fractionation = Fractionation(total_dose, num_fractions)
        logger.info(f"Set prescription to {total_dose} Gy in {num_fractions} fractions for plan '{self.plan_name}'")
    
    def set_treatment_machine(self, machine: Linac):
        """
        Set the treatment machine for the 3D-CRT plan.
        
        Parameters
        ----------
        machine : Linac
            The treatment machine to use
        """
        self.treatment_machine = machine
        logger.info(f"Set treatment machine to '{machine.machine_name}' for plan '{self.plan_name}'")
    
    def add_target_structure(self, structure_id: str):
        """
        Add a target structure to the 3D-CRT plan.
        
        Parameters
        ----------
        structure_id : str
            ID of the target structure
        """
        if structure_id not in self.target_structures:
            self.target_structures.append(structure_id)
            logger.info(f"Added target structure '{structure_id}' to plan '{self.plan_name}'")
    
    def add_critical_structure(self, structure_id: str):
        """
        Add a critical structure to the 3D-CRT plan.
        
        Parameters
        ----------
        structure_id : str
            ID of the critical structure
        """
        if structure_id not in self.critical_structures:
            self.critical_structures.append(structure_id)
            logger.info(f"Added critical structure '{structure_id}' to plan '{self.plan_name}'")
    
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
        
        logger.info(f"Created conformal block for beam '{beam.beam_name}' based on structure '{target_structure_id}'")
        return block
    
    def calculate_beam_weights(self, method: str = "equal"):
        """
        Calculate beam weights for the 3D-CRT plan.
        
        Parameters
        ----------
        method : str, optional
            Method to use for weight calculation ("equal", "distance", or "custom")
        """
        if not self.beams:
            logger.warning(f"No beams in plan '{self.plan_name}' to calculate weights for")
            return
        
        if method == "equal":
            # Equal weighting for all beams
            weight = 1.0 / len(self.beams)
            for beam in self.beams:
                beam.weight = weight
            logger.info(f"Applied equal weighting to all beams in plan '{self.plan_name}'")
        
        elif method == "distance":
            # This would be a more complex algorithm based on distances in a real implementation
            # Just use equal weighting for demonstration
            weight = 1.0 / len(self.beams)
            for beam in self.beams:
                beam.weight = weight
            logger.info(f"Applied distance-based weighting to all beams in plan '{self.plan_name}'")
        
        elif method == "custom":
            # Custom weighting would be implemented here
            logger.info(f"Custom weighting not implemented yet for plan '{self.plan_name}'")
        
        else:
            logger.warning(f"Unknown weighting method '{method}'. Using equal weighting.")
            weight = 1.0 / len(self.beams)
            for beam in self.beams:
                beam.weight = weight
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the 3D-CRT plan to a dictionary.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary containing the plan information
        """
        return {
            "plan_id": self.plan_id,
            "plan_name": self.plan_name,
            "technique_type": self.technique_type,
            "description": self.description,
            "status": self.status,
            "treatment_machine": self.treatment_machine.machine_id if self.treatment_machine else None,
            "beams": [beam.beam_id for beam in self.beams],
            "prescription_dose": self.prescription_dose,
            "prescription_fractions": self.prescription_fractions,
            "target_structures": self.target_structures,
            "critical_structures": self.critical_structures,
            "evaluation_metrics": self.evaluation_metrics,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], beam_dict: Dict[str, Beam] = None, 
                 machine_dict: Dict[str, Linac] = None) -> 'Conformal3DRT':
        """
        Create a Conformal3DRT object from a dictionary.
        
        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary containing the plan information
        beam_dict : Dict[str, Beam], optional
            Dictionary mapping beam IDs to Beam objects
        machine_dict : Dict[str, Linac], optional
            Dictionary mapping machine IDs to Linac objects
            
        Returns
        -------
        Conformal3DRT
            The created Conformal3DRT object
        """
        plan = cls(
            plan_name=data["plan_name"],
            plan_id=data["plan_id"]
        )
        
        plan.technique_type = data["technique_type"]
        plan.description = data["description"]
        plan.status = data["status"]
        
        # Add treatment machine if available
        if data["treatment_machine"] and machine_dict:
            machine_id = data["treatment_machine"]
            if machine_id in machine_dict:
                plan.treatment_machine = machine_dict[machine_id]
        
        # Add beams if available
        if beam_dict:
            for beam_id in data["beams"]:
                if beam_id in beam_dict:
                    plan.beams.append(beam_dict[beam_id])
        
        plan.prescription_dose = data["prescription_dose"]
        plan.prescription_fractions = data["prescription_fractions"]
        if plan.prescription_dose > 0 and plan.prescription_fractions > 0:
            plan.fractionation = Fractionation(plan.prescription_dose, plan.prescription_fractions)
        
        plan.target_structures = data["target_structures"]
        plan.critical_structures = data["critical_structures"]
        plan.evaluation_metrics = data["evaluation_metrics"]
        plan.metadata = data["metadata"]
        
        return plan