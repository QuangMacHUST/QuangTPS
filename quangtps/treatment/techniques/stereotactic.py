#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module for Stereotactic treatment techniques (SRS/SBRT).

This module provides classes and methods to define and manage Stereotactic Radiosurgery (SRS)
and Stereotactic Body Radiation Therapy (SBRT) plans.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional

from quangtps.treatment.beams.beam import Beam
from quangtps.treatment.machine.linac import Linac
from quangtps.treatment.machine.treatment_machine import TreatmentMachine
from quangtps.treatment.fractionation import Fractionation
from quangtps.treatment.techniques.technique_interface import BaseTreatmentTechnique, TechniqueCategory

logger = logging.getLogger(__name__)

class StereotacticBase(BaseTreatmentTechnique):
    """Base class for stereotactic treatments (SRS/SBRT)."""
    
    def __init__(self, name: str, technique_id: Optional[str] = None, technique_type: str = "Stereotactic", 
                 category: TechniqueCategory = TechniqueCategory.ADVANCED):
        """
        Initialize a stereotactic treatment plan.
        
        Parameters
        ----------
        name : str
            Name of the plan
        technique_id : str, optional
            ID for the technique. If not provided, a new ID will be generated
        technique_type : str, optional
            Type of stereotactic technique (e.g., "SRS", "SBRT")
        category : TechniqueCategory, optional
            Category of the technique
        """
        super().__init__(
            name=name,
            technique_id=technique_id,
            category=category
        )
        
        # Basic plan attributes
        self.technique_type = technique_type
        self.description = ""
        self.status = "DRAFT"  # DRAFT, APPROVED, DELIVERED, ARCHIVED
        
        # Stereotactic-specific attributes
        self.prescription_isodose: float = 80.0  # Default: 80% isodose line
        self.immobilization_device: str = ""
        self.stereotactic_frame: Optional[str] = None
        self.margin_recipe: Dict[str, float] = {
            "GTV_to_CTV": 0.0,  # mm
            "CTV_to_PTV": 1.0,  # mm - Usually minimal PTV margins in stereotactic treatments
        }
        
        # Plan evaluation
        self.plan_quality_metrics: Dict[str, float] = {}
        
        logger.info(f"Created new {technique_type} plan: {name} (ID: {self.technique_id})")
    
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
        Add a beam to the stereotactic plan.
        
        Parameters
        ----------
        beam : Beam
            The beam to add
        """
        if beam not in self.beams:
            self.beams.append(beam)
            logger.info(f"Added beam {beam.beam_name} to {self.technique_type} plan {self.name}")
    
    def get_beams(self) -> List[Beam]:
        """
        Get the list of beams in the stereotactic plan.
        
        Returns
        -------
        List[Beam]
            List of beams in the plan
        """
        return self.beams
    
    def set_fractionation(self, fractionation: Fractionation) -> None:
        """
        Set the fractionation scheme for the stereotactic plan.
        
        Parameters
        ----------
        fractionation : Fractionation
            Fractionation scheme
        """
        self.fractionation = fractionation
        logger.info(f"Set fractionation for {self.technique_type} plan {self.name}: "
                   f"{fractionation.num_fractions} fractions, "
                   f"{fractionation.dose_per_fraction} Gy per fraction")
    
    def set_prescription_isodose(self, isodose_percent: float) -> None:
        """
        Set the prescription isodose line.
        
        Parameters
        ----------
        isodose_percent : float
            Prescription isodose as a percentage (e.g., 80.0 for 80%)
            
        Raises
        ------
        ValueError
            If isodose is outside the valid range (50% - 100%)
        """
        if not 50 <= isodose_percent <= 100:
            raise ValueError(f"Prescription isodose must be between 50% and 100%, got {isodose_percent}%")
        
        self.prescription_isodose = isodose_percent
        logger.info(f"Set prescription isodose for {self.technique_type} plan {self.name}: {isodose_percent}%")
    
    def set_machine(self, machine: TreatmentMachine) -> None:
        """
        Set the treatment machine for the stereotactic plan.
        
        Parameters
        ----------
        machine : TreatmentMachine
            Treatment machine
            
        Raises
        ------
        ValueError
            If the machine does not support stereotactic treatments
        """
        if not isinstance(machine, Linac):
            raise ValueError(f"Stereotactic treatments require a Linac treatment machine, got {type(machine).__name__}")
            
        if not getattr(machine, 'supports_stereotactic', False):
            raise ValueError(f"Machine {machine.name} does not support stereotactic treatments")
        
        self.machine = machine
        logger.info(f"Set treatment machine for {self.technique_type} plan {self.name}: {machine.name}")
    
    def set_immobilization(self, device: str, frame: Optional[str] = None) -> None:
        """
        Set immobilization details for the stereotactic plan.
        
        Parameters
        ----------
        device : str
            Immobilization device name
        frame : str, optional
            Stereotactic frame name
        """
        self.immobilization_device = device
        self.stereotactic_frame = frame
        
        logger.info(f"Set immobilization for {self.technique_type} plan {self.name}: {device}")
        if frame:
            logger.info(f"Set stereotactic frame for {self.technique_type} plan {self.name}: {frame}")
    
    def calculate_dose(self) -> Dict[str, Any]:
        """
        Calculate the dose distribution for the stereotactic plan.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary with dose array and metadata
        """
        logger.info(f"Calculating dose for {self.technique_type} plan {self.name}")
        
        # Create dummy 3D dose array (100x100x100)
        dose_array = np.zeros((100, 100, 100))
        
        # Add dummy dose distribution that mimics a stereotactic plan (sharp fall-off)
        center = 50
        radius = 10
        
        # Create a sphere of dose with sharp fall-off
        for x in range(center - radius - 10, center + radius + 10):
            for y in range(center - radius - 10, center + radius + 10):
                for z in range(center - radius - 10, center + radius + 10):
                    if 0 <= x < 100 and 0 <= y < 100 and 0 <= z < 100:
                        # Calculate distance from center
                        dist = np.sqrt((x - center) ** 2 + (y - center) ** 2 + (z - center) ** 2)
                        
                        # Dose inside target volume
                        if dist <= radius:
                            dose_array[x, y, z] = 20.0  # Gy
                        else:
                            # Steep dose gradient outside target
                            fall_off = 3.0  # mm
                            if dist <= radius + fall_off:
                                dose_factor = ((radius + fall_off) - dist) / fall_off
                                dose_array[x, y, z] = 20.0 * (dose_factor ** 2)
        
        result = {
            "dose_array": dose_array,
            "dimensions": (100, 100, 100),
            "spacing": (0.2, 0.2, 0.2),  # cm
            "origin": (-10, -10, -10),   # cm
            "technique_type": self.technique_type,
            "prescription_isodose": self.prescription_isodose
        }
        
        return result
    
    def evaluate_plan(self) -> Dict[str, float]:
        """
        Evaluate the quality of the stereotactic plan.
        
        Returns
        -------
        Dict[str, float]
            Dictionary of plan quality metrics
        """
        metrics = {
            "conformity_index": 1.2,     # Placeholder value
            "gradient_index": 3.0,       # Placeholder value
            "homogeneity_index": 1.25,   # Placeholder value
            "paddick_ci": 0.85,          # Placeholder value
            "coverage": 0.98             # Placeholder value (98%)
        }
        
        self.plan_quality_metrics = metrics
        logger.info(f"Evaluated {self.technique_type} plan {self.name}")
        return metrics
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the stereotactic plan to a dictionary.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary representation of the plan
        """
        # Start with the base technique dictionary
        result = super().to_dict()
        
        # Add stereotactic-specific attributes
        result.update({
            "technique_type": self.technique_type,
            "description": self.description,
            "status": self.status,
            "prescription_isodose": self.prescription_isodose,
            "immobilization_device": self.immobilization_device,
            "stereotactic_frame": self.stereotactic_frame,
            "margin_recipe": self.margin_recipe,
            "plan_quality_metrics": self.plan_quality_metrics
        })
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StereotacticBase':
        """
        Create a stereotactic plan from a dictionary.
        
        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary with plan data
            
        Returns
        -------
        StereotacticBase
            Stereotactic plan instance
        """
        plan = cls(
            name=data["name"],
            technique_id=data["technique_id"],
            technique_type=data.get("technique_type", "Stereotactic")
        )
        
        # Restore basic attributes
        plan.description = data.get("description", "")
        plan.status = data.get("status", "DRAFT")
        
        # Restore stereotactic-specific attributes
        if "prescription_isodose" in data:
            plan.prescription_isodose = data["prescription_isodose"]
            
        if "immobilization_device" in data:
            plan.immobilization_device = data["immobilization_device"]
            
        if "stereotactic_frame" in data:
            plan.stereotactic_frame = data["stereotactic_frame"]
            
        if "margin_recipe" in data:
            plan.margin_recipe = data["margin_recipe"]
            
        if "plan_quality_metrics" in data:
            plan.plan_quality_metrics = data["plan_quality_metrics"]
        
        # Restore common components if present
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


class SRS(StereotacticBase):
    """Class representing a Stereotactic Radiosurgery (SRS) plan."""
    
    def __init__(self, name: str, technique_id: Optional[str] = None):
        """
        Initialize an SRS plan.
        
        Parameters
        ----------
        name : str
            Name of the plan
        technique_id : str, optional
            ID for the technique. If not provided, a new ID will be generated
        """
        super().__init__(
            name=name, 
            technique_id=technique_id, 
            technique_type="SRS",
            category=TechniqueCategory.ADVANCED
        )
        
        # SRS-specific attributes
        self.target_location = "Intracranial"  # SRS is typically used for brain lesions
        self.typical_fraction_count = 1  # SRS is typically single-fraction
        self.prescription_isodose = 80.0  # Typical SRS prescription isodose
        
        # Common dose constraints for critical structures
        self.dose_constraints = {
            "brainstem": {"max_dose": 12.0},
            "optic_chiasm": {"max_dose": 8.0},
            "optic_nerve": {"max_dose": 8.0},
            "normal_brain": {"volume_dose": {"volume": 10.0, "dose": 12.0}}  # V12Gy < 10cc
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the SRS plan to a dictionary.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary representation of the plan
        """
        # Start with base class dictionary
        result = super().to_dict()
        
        # Add SRS-specific attributes
        result.update({
            "target_location": self.target_location,
            "typical_fraction_count": self.typical_fraction_count,
            "dose_constraints": self.dose_constraints
        })
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SRS':
        """
        Create an SRS plan from a dictionary.
        
        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary with plan data
            
        Returns
        -------
        SRS
            SRS plan instance
        """
        plan = super(SRS, cls).from_dict(data)
        
        # Restore SRS-specific attributes
        if "target_location" in data:
            plan.target_location = data["target_location"]
            
        if "typical_fraction_count" in data:
            plan.typical_fraction_count = data["typical_fraction_count"]
            
        if "dose_constraints" in data:
            plan.dose_constraints = data["dose_constraints"]
        
        return plan


class SBRT(StereotacticBase):
    """Class representing a Stereotactic Body Radiation Therapy (SBRT) plan."""
    
    def __init__(self, name: str, technique_id: Optional[str] = None):
        """
        Initialize an SBRT plan.
        
        Parameters
        ----------
        name : str
            Name of the plan
        technique_id : str, optional
            ID for the technique. If not provided, a new ID will be generated
        """
        super().__init__(
            name=name, 
            technique_id=technique_id, 
            technique_type="SBRT",
            category=TechniqueCategory.ADVANCED
        )
        
        # SBRT-specific attributes
        self.target_location = "Extracranial"  # SBRT is for body (non-brain) lesions
        self.typical_fraction_count = 3  # SBRT typically uses 3-5 fractions
        self.prescription_isodose = 80.0  # Typical SBRT prescription isodose
        self.respiratory_motion_management = None  # Optional motion management technique
        self.dose_constraints = {}  # Will be populated based on treatment site
    
    def set_respiratory_management(self, technique: str, details: Dict[str, Any] = None) -> None:
        """
        Set respiratory motion management technique.
        
        Parameters
        ----------
        technique : str
            Name of the respiratory motion management technique
        details : Dict[str, Any], optional
            Additional details about the technique
        """
        if details is None:
            details = {}
        
        self.respiratory_motion_management = {
            "technique": technique,
            "details": details
        }
        
        logger.info(f"Set respiratory motion management for SBRT plan {self.name}: {technique}")
    
    def setup_organ_constraints(self, site: str) -> None:
        """
        Set up common dose constraints based on treatment site.
        
        Parameters
        ----------
        site : str
            Treatment site (e.g., "lung", "liver", "spine")
        """
        constraints = {}
        
        if site.lower() == "lung":
            constraints = {
                "spinal_cord": {"max_dose": 18.0},
                "lungs_minus_itv": {"volume_dose": {"volume": 10.0, "dose": 20.0}}  # V20Gy < 10%
            }
        elif site.lower() == "liver":
            constraints = {
                "spinal_cord": {"max_dose": 18.0},
                "liver_minus_gtv": {"volume_dose": {"volume": 700.0, "dose": 15.0}}  # At least 700cc < 15Gy
            }
        elif site.lower() == "spine":
            constraints = {
                "spinal_cord": {"max_dose": 14.0, "volume_dose": {"volume": 0.35, "dose": 10.0}}
            }
        else:
            logger.warning(f"No predefined constraints available for site: {site}")
        
        self.dose_constraints = constraints
        logger.info(f"Set up standard {site} constraints for SBRT plan {self.name}")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the SBRT plan to a dictionary.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary representation of the plan
        """
        # Start with base class dictionary
        result = super().to_dict()
        
        # Add SBRT-specific attributes
        result.update({
            "target_location": self.target_location,
            "typical_fraction_count": self.typical_fraction_count,
            "respiratory_motion_management": self.respiratory_motion_management,
            "dose_constraints": self.dose_constraints
        })
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SBRT':
        """
        Create an SBRT plan from a dictionary.
        
        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary with plan data
            
        Returns
        -------
        SBRT
            SBRT plan instance
        """
        plan = super(SBRT, cls).from_dict(data)
        
        # Restore SBRT-specific attributes
        if "target_location" in data:
            plan.target_location = data["target_location"]
            
        if "typical_fraction_count" in data:
            plan.typical_fraction_count = data["typical_fraction_count"]
            
        if "respiratory_motion_management" in data:
            plan.respiratory_motion_management = data["respiratory_motion_management"]
            
        if "dose_constraints" in data:
            plan.dose_constraints = data["dose_constraints"]
        
        return plan


# Ensure classes are exported correctly
__all__ = ['StereotacticBase', 'SRS', 'SBRT']