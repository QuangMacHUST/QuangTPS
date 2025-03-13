#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module for Carbon Ion Therapy techniques.

This module provides classes for configuring and managing Carbon Ion Therapy,
a form of particle therapy that uses carbon ions for cancer treatment.
"""

import logging
from typing import Dict, Any, List, Optional
from enum import Enum

from quangtps.treatment.beams.beam import Beam
from quangtps.treatment.machine.carbon_ion import CarbonIonMachine
from quangtps.treatment.fractionation import Fractionation

logger = logging.getLogger(__name__)

class CarbonIonDeliveryTechnique(str, Enum):
    """Enum for different Carbon Ion delivery techniques."""
    PASSIVE_SCATTERING = "PASSIVE_SCATTERING"  # Traditional passive scattering method
    ACTIVE_SCANNING = "ACTIVE_SCANNING"  # Pencil beam scanning without intensity modulation
    IMCT = "IMCT"  # Intensity Modulated Carbon Therapy (using pencil beam scanning)
    HYBRID = "HYBRID"  # Combination of techniques

class CarbonIonPlanningMethod(str, Enum):
    """Enum for different Carbon Ion treatment planning methods."""
    SINGLE_FIELD_UNIFORM_DOSE = "SINGLE_FIELD_UNIFORM_DOSE"  # SFUD
    MULTI_FIELD_OPTIMIZATION = "MULTI_FIELD_OPTIMIZATION"  # MFO
    ROBUST_OPTIMIZATION = "ROBUST_OPTIMIZATION"  # With robustness parameters
    BIOLOGICAL_OPTIMIZATION = "BIOLOGICAL_OPTIMIZATION"  # Using RBE models
    BRAGG_PEAK_BOOST = "BRAGG_PEAK_BOOST"  # Boost using the Bragg peak

class CarbonIonTherapy:
    """
    Class for Carbon Ion Therapy treatment technique.
    
    Carbon Ion Therapy uses carbon ions for cancer treatment, offering
    higher linear energy transfer (LET) and relative biological effectiveness (RBE)
    compared to proton therapy, potentially making it more effective for
    radioresistant tumors.
    """
    
    def __init__(self, 
                 name: str, 
                 technique: CarbonIonDeliveryTechnique = CarbonIonDeliveryTechnique.ACTIVE_SCANNING,
                 planning_method: CarbonIonPlanningMethod = CarbonIonPlanningMethod.MULTI_FIELD_OPTIMIZATION,
                 carbon_id: Optional[str] = None):
        """
        Initialize a Carbon Ion Therapy treatment.
        
        Parameters
        ----------
        name : str
            Name of the treatment plan
        technique : CarbonIonDeliveryTechnique
            Delivery technique to use
        planning_method : CarbonIonPlanningMethod
            Planning method to use
        carbon_id : str, optional
            Unique ID for the plan
        """
        self.name = name
        self.carbon_id = carbon_id or f"carbon_{name.lower().replace(' ', '_')}"
        self.technique = technique
        self.planning_method = planning_method
        
        # Carbon Ion-specific attributes
        self.beams: List[Beam] = []
        self.machine: Optional[CarbonIonMachine] = None
        self.fractionation: Optional[Fractionation] = None
        
        # Treatment parameters
        self.total_dose = 60.0  # GyE (Gray Equivalent)
        self.dose_prescription_type = "PHYSICAL"  # PHYSICAL or BIOLOGICAL
        self.rbe_model = "MIXED_LEM"  # Local Effect Model variant
        self.alpha_beta_ratio = 2.0  # For RBE calculations (Gy)
        self.target_structures = []  # List of target structure IDs
        self.oar_structures = []  # List of organ-at-risk structure IDs
        self.optimization_objectives = []  # List of optimization objectives
        
        # Robustness parameters
        self.robust_optimization = False
        self.setup_uncertainty = 2.0  # mm
        self.range_uncertainty = 3.5  # % of range
        
    def set_fractionation(self, fractionation: Fractionation):
        """
        Set fractionation scheme.
        
        Parameters
        ----------
        fractionation : Fractionation
            Fractionation scheme
        """
        self.fractionation = fractionation
        self.total_dose = fractionation.total_dose
        
    def set_machine(self, machine: CarbonIonMachine):
        """
        Set treatment machine.
        
        Parameters
        ----------
        machine : CarbonIonMachine
            Carbon ion accelerator for treatment
        """
        self.machine = machine
        
    def add_beam(self, beam: Beam):
        """
        Add a beam to the Carbon Ion plan.
        
        Parameters
        ----------
        beam : Beam
            Beam to add
        """
        self.beams.append(beam)
        
    def set_planning_method(self, method: CarbonIonPlanningMethod):
        """
        Set the planning method.
        
        Parameters
        ----------
        method : CarbonIonPlanningMethod
            Planning method to use
        """
        self.planning_method = method
        
        # Update parameters based on planning method
        if method == CarbonIonPlanningMethod.ROBUST_OPTIMIZATION:
            self.robust_optimization = True
        else:
            self.robust_optimization = False
            
    def set_rbe_parameters(self, model: str, alpha_beta_ratio: float):
        """
        Set RBE calculation parameters.
        
        Parameters
        ----------
        model : str
            RBE model (e.g., "MIXED_LEM", "KANAI", "MKM")
        alpha_beta_ratio : float
            Alpha/beta ratio for the target tissue (Gy)
        """
        self.rbe_model = model
        self.alpha_beta_ratio = alpha_beta_ratio
        
    def set_robust_parameters(self, setup_uncertainty: float, range_uncertainty: float):
        """
        Set robust optimization parameters.
        
        Parameters
        ----------
        setup_uncertainty : float
            Setup uncertainty (mm)
        range_uncertainty : float
            Range uncertainty (% of range)
        """
        self.setup_uncertainty = setup_uncertainty
        self.range_uncertainty = range_uncertainty
        self.robust_optimization = True
        
    def set_target_structures(self, structure_ids: List[str]):
        """
        Set target structures.
        
        Parameters
        ----------
        structure_ids : List[str]
            List of structure IDs
        """
        self.target_structures = structure_ids
        
    def set_oar_structures(self, structure_ids: List[str]):
        """
        Set organs-at-risk structures.
        
        Parameters
        ----------
        structure_ids : List[str]
            List of structure IDs
        """
        self.oar_structures = structure_ids
        
    def add_optimization_objective(self, objective: Dict[str, Any]):
        """
        Add an optimization objective.
        
        Parameters
        ----------
        objective : Dict[str, Any]
            Optimization objective
        """
        self.optimization_objectives.append(objective)
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert Carbon Ion Therapy plan to dictionary.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary representation
        """
        return {
            "name": self.name,
            "carbon_id": self.carbon_id,
            "technique": self.technique,
            "planning_method": self.planning_method,
            "total_dose": self.total_dose,
            "dose_prescription_type": self.dose_prescription_type,
            "rbe_model": self.rbe_model,
            "alpha_beta_ratio": self.alpha_beta_ratio,
            "target_structures": self.target_structures,
            "oar_structures": self.oar_structures,
            "optimization_objectives": self.optimization_objectives,
            "robust_optimization": self.robust_optimization,
            "setup_uncertainty": self.setup_uncertainty,
            "range_uncertainty": self.range_uncertainty,
            "fractionation": self.fractionation.to_dict() if self.fractionation else None,
            "machine": self.machine.machine_id if self.machine else None,
            "beams": [beam.to_dict() for beam in self.beams]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CarbonIonTherapy':
        """
        Create Carbon Ion Therapy plan from dictionary.
        
        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary with plan data
            
        Returns
        -------
        CarbonIonTherapy
            Carbon Ion Therapy plan instance
        """
        plan = cls(
            name=data["name"],
            technique=data["technique"],
            planning_method=data["planning_method"],
            carbon_id=data["carbon_id"]
        )
        
        plan.total_dose = data["total_dose"]
        plan.dose_prescription_type = data["dose_prescription_type"]
        plan.rbe_model = data["rbe_model"]
        plan.alpha_beta_ratio = data["alpha_beta_ratio"]
        plan.target_structures = data["target_structures"]
        plan.oar_structures = data["oar_structures"]
        plan.optimization_objectives = data["optimization_objectives"]
        plan.robust_optimization = data["robust_optimization"]
        plan.setup_uncertainty = data["setup_uncertainty"]
        plan.range_uncertainty = data["range_uncertainty"]
        
        return plan