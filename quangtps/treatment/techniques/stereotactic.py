#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module for Stereotactic treatment techniques (SRS/SBRT).

This module provides classes and methods to define and manage Stereotactic Radiosurgery (SRS)
and Stereotactic Body Radiation Therapy (SBRT) plans.
"""

import uuid
import logging
import numpy as np
from typing import List, Dict, Any, Optional

from quangtps.treatment.beams.beam import Beam
from quangtps.treatment.machine.linac import Linac
from quangtps.treatment.fractionation import Fractionation

logger = logging.getLogger(__name__)

class StereotacticBase:
    """Base class for stereotactic treatments (SRS/SBRT)."""
    
    def __init__(self, plan_name: str, plan_id: Optional[str] = None, technique_type: str = "Stereotactic"):
        """Initialize a stereotactic treatment plan."""
        self.plan_name = plan_name
        self.plan_id = plan_id if plan_id else str(uuid.uuid4())
        
        # Basic plan attributes
        self.technique_type = technique_type
        self.description = ""
        self.status = "DRAFT"  # DRAFT, APPROVED, DELIVERED, ARCHIVED
        
        # Treatment machine
        self.treatment_machine: Optional[Linac] = None
        
        # Stereotactic-specific attributes
        self.beams: List[Beam] = []
        self.prescription_isodose: float = 80.0  # Default: 80% isodose line
        self.fractionation: Optional[Fractionation] = None
        self.immobilization_device: str = ""
        self.stereotactic_frame: Optional[str] = None
        self.margin_recipe: Dict[str, float] = {
            "GTV_to_CTV": 0.0,  # mm
            "CTV_to_PTV": 1.0,  # mm - Usually minimal PTV margins in stereotactic treatments
        }
        
        # Plan evaluation
        self.plan_quality_metrics: Dict[str, float] = {}
        
        logger.info(f"Created new {technique_type} plan: {plan_name} (ID: {self.plan_id})")
    
    def add_beam(self, beam: Beam) -> None:
        """Add a beam to the stereotactic plan."""
        if beam not in self.beams:
            self.beams.append(beam)
            logger.info(f"Added beam {beam.beam_name} to {self.technique_type} plan {self.plan_name}")
    
    def set_fractionation(self, fractionation: Fractionation) -> None:
        """Set the fractionation scheme for the stereotactic plan."""
        self.fractionation = fractionation
        logger.info(f"Set fractionation for {self.technique_type} plan {self.plan_name}: "
                   f"{fractionation.num_fractions} fractions, "
                   f"{fractionation.dose_per_fraction} Gy per fraction")
    
    def set_prescription_isodose(self, isodose_percent: float) -> None:
        """Set the prescription isodose line."""
        if not 50 <= isodose_percent <= 100:
            raise ValueError(f"Prescription isodose must be between 50% and 100%, got {isodose_percent}%")
        
        self.prescription_isodose = isodose_percent
        logger.info(f"Set prescription isodose for {self.technique_type} plan {self.plan_name}: {isodose_percent}%")
    
    def set_treatment_machine(self, machine: Linac) -> None:
        """Set the treatment machine for the stereotactic plan."""
        if not machine.supports_stereotactic:
            raise ValueError(f"Machine {machine.name} does not support stereotactic treatments")
        
        self.treatment_machine = machine
        logger.info(f"Set treatment machine for {self.technique_type} plan {self.plan_name}: {machine.name}")
    
    def set_immobilization(self, device: str, frame: Optional[str] = None) -> None:
        """Set immobilization details for the stereotactic plan."""
        self.immobilization_device = device
        self.stereotactic_frame = frame
        
        logger.info(f"Set immobilization for {self.technique_type} plan {self.plan_name}: {device}")
        if frame:
            logger.info(f"Set stereotactic frame for {self.technique_type} plan {self.plan_name}: {frame}")
    
    def calculate_dose(self) -> np.ndarray:
        """Calculate the dose distribution for the stereotactic plan."""
        logger.info(f"Calculating dose for {self.technique_type} plan {self.plan_name}")
        return np.zeros((100, 100, 100))  # Placeholder
    
    def evaluate_plan(self) -> Dict[str, float]:
        """Evaluate the quality of the stereotactic plan."""
        metrics = {
            "conformity_index": 1.2,     # Placeholder value
            "gradient_index": 3.0,       # Placeholder value
            "homogeneity_index": 1.25,   # Placeholder value
            "paddick_ci": 0.85,          # Placeholder value
            "coverage": 0.98             # Placeholder value (98%)
        }
        
        self.plan_quality_metrics = metrics
        logger.info(f"Evaluated {self.technique_type} plan {self.plan_name}")
        return metrics
    
    def __str__(self) -> str:
        """Return string representation of the stereotactic plan."""
        return f"{self.technique_type} Plan: {self.plan_name} (ID: {self.plan_id})"


class SRS(StereotacticBase):
    """Class representing a Stereotactic Radiosurgery (SRS) plan."""
    
    def __init__(self, plan_name: str, plan_id: Optional[str] = None):
        """Initialize an SRS plan."""
        super().__init__(plan_name, plan_id, technique_type="SRS")
        
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


class SBRT(StereotacticBase):
    """Class representing a Stereotactic Body Radiation Therapy (SBRT) plan."""
    
    def __init__(self, plan_name: str, plan_id: Optional[str] = None):
        """Initialize an SBRT plan."""
        super().__init__(plan_name, plan_id, technique_type="SBRT")
        
        # SBRT-specific attributes
        self.target_location = "Extracranial"  # SBRT is for body (non-brain) lesions
        self.typical_fraction_count = 3  # SBRT typically uses 3-5 fractions
        self.prescription_isodose = 80.0  # Typical SBRT prescription isodose
        self.respiratory_motion_management = None  # Optional motion management technique
    
    def set_respiratory_management(self, technique: str, details: Dict[str, Any] = None) -> None:
        """Set respiratory motion management technique."""
        if details is None:
            details = {}
        
        self.respiratory_motion_management = {
            "technique": technique,
            "details": details
        }
        
        logger.info(f"Set respiratory motion management for SBRT plan {self.plan_name}: {technique}")
    
    def setup_organ_constraints(self, site: str) -> None:
        """Set up common dose constraints based on treatment site."""
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
        logger.info(f"Set up standard {site} constraints for SBRT plan {self.plan_name}")