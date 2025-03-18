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
from quangtps.treatment.machine.treatment_machine import TreatmentMachine
from quangtps.treatment.fractionation import Fractionation
from quangtps.treatment.techniques.technique_interface import BaseTreatmentTechnique, TechniqueCategory

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

class CarbonIonTherapy(BaseTreatmentTechnique):
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
        super().__init__(
            name=name,
            technique_id=carbon_id,
            category=TechniqueCategory.PARTICLE
        )
        
        self.delivery_technique = technique
        self.planning_method = planning_method
        
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
        
        logger.info(f"Created new Carbon Ion Therapy plan: {name} (ID: {self.technique_id})")
        
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
    
    def set_fractionation(self, fractionation: Fractionation) -> None:
        """
        Set fractionation scheme.
        
        Parameters
        ----------
        fractionation : Fractionation
            Fractionation scheme
        """
        self.fractionation = fractionation
        self.total_dose = fractionation.total_dose
        logger.info(f"Set fractionation for Carbon Ion plan {self.name}: "
                   f"{fractionation.num_fractions} fractions, "
                   f"{fractionation.dose_per_fraction} GyE per fraction")
        
    def set_machine(self, machine: TreatmentMachine) -> None:
        """
        Set treatment machine.
        
        Parameters
        ----------
        machine : TreatmentMachine
            Treatment machine for carbon ion therapy
        """
        if not isinstance(machine, CarbonIonMachine):
            raise ValueError(f"Carbon Ion therapy requires a CarbonIonMachine, got {type(machine).__name__}")
            
        self.machine = machine
        logger.info(f"Set treatment machine for Carbon Ion plan {self.name}: {machine.name}")
        
    def add_beam(self, beam: Beam) -> None:
        """
        Add a beam to the Carbon Ion plan.
        
        Parameters
        ----------
        beam : Beam
            Beam to add
        """
        if beam not in self.beams:
            self.beams.append(beam)
            logger.info(f"Added beam {beam.beam_name} to Carbon Ion plan {self.name}")
    
    def get_beams(self) -> List[Beam]:
        """
        Get the list of beams in the Carbon Ion plan.
        
        Returns
        -------
        List[Beam]
            List of beams in the plan
        """
        return self.beams
        
    def set_planning_method(self, method: CarbonIonPlanningMethod) -> None:
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
        
        logger.info(f"Set planning method for Carbon Ion plan {self.name}: {method.value}")
            
    def set_rbe_parameters(self, model: str, alpha_beta_ratio: float) -> None:
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
        logger.info(f"Set RBE parameters for Carbon Ion plan {self.name}: "
                   f"model={model}, α/β={alpha_beta_ratio} Gy")
        
    def set_robust_parameters(self, setup_uncertainty: float, range_uncertainty: float) -> None:
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
        logger.info(f"Set robust parameters for Carbon Ion plan {self.name}: "
                   f"setup={setup_uncertainty} mm, range={range_uncertainty}%")
        
    def set_target_structures(self, structure_ids: List[str]) -> None:
        """
        Set target structures.
        
        Parameters
        ----------
        structure_ids : List[str]
            List of structure IDs
        """
        self.target_structures = structure_ids
        logger.info(f"Set {len(structure_ids)} target structures for Carbon Ion plan {self.name}")
        
    def set_oar_structures(self, structure_ids: List[str]) -> None:
        """
        Set organs-at-risk structures.
        
        Parameters
        ----------
        structure_ids : List[str]
            List of structure IDs
        """
        self.oar_structures = structure_ids
        logger.info(f"Set {len(structure_ids)} OAR structures for Carbon Ion plan {self.name}")
        
    def add_optimization_objective(self, objective: Dict[str, Any]) -> None:
        """
        Add an optimization objective.
        
        Parameters
        ----------
        objective : Dict[str, Any]
            Optimization objective
        """
        self.optimization_objectives.append(objective)
        logger.info(f"Added optimization objective for Carbon Ion plan {self.name}: "
                   f"structure={objective.get('structure', 'unknown')}, "
                   f"type={objective.get('type', 'unknown')}")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert Carbon Ion plan to a dictionary.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary representation of the plan
        """
        # Start with the base technique dictionary
        result = super().to_dict()
        
        # Add Carbon Ion-specific attributes
        result.update({
            "delivery_technique": self.delivery_technique.value,
            "planning_method": self.planning_method.value,
            "dose_prescription_type": self.dose_prescription_type,
            "rbe_model": self.rbe_model,
            "alpha_beta_ratio": self.alpha_beta_ratio,
            "target_structures": self.target_structures,
            "oar_structures": self.oar_structures,
            "optimization_objectives": self.optimization_objectives,
            "robust_optimization": self.robust_optimization,
            "setup_uncertainty": self.setup_uncertainty,
            "range_uncertainty": self.range_uncertainty
        })
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CarbonIonTherapy':
        """
        Create a Carbon Ion plan from a dictionary.
        
        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary with plan data
            
        Returns
        -------
        CarbonIonTherapy
            CarbonIonTherapy instance
        """
        # Create delivery technique enum from string
        delivery_technique = CarbonIonDeliveryTechnique(data.get("delivery_technique", 
                                                        CarbonIonDeliveryTechnique.ACTIVE_SCANNING.value))
        
        # Create planning method enum from string
        planning_method = CarbonIonPlanningMethod(data.get("planning_method", 
                                                 CarbonIonPlanningMethod.MULTI_FIELD_OPTIMIZATION.value))
        
        # Create basic plan
        plan = cls(
            name=data["name"],
            technique=delivery_technique,
            planning_method=planning_method,
            carbon_id=data["technique_id"]
        )
        
        # Restore specific parameters
        if "dose_prescription_type" in data:
            plan.dose_prescription_type = data["dose_prescription_type"]
            
        if "rbe_model" in data:
            plan.rbe_model = data["rbe_model"]
            
        if "alpha_beta_ratio" in data:
            plan.alpha_beta_ratio = data["alpha_beta_ratio"]
            
        if "target_structures" in data:
            plan.target_structures = data["target_structures"]
            
        if "oar_structures" in data:
            plan.oar_structures = data["oar_structures"]
            
        if "optimization_objectives" in data:
            plan.optimization_objectives = data["optimization_objectives"]
            
        if "robust_optimization" in data:
            plan.robust_optimization = data["robust_optimization"]
            
        if "setup_uncertainty" in data:
            plan.setup_uncertainty = data["setup_uncertainty"]
            
        if "range_uncertainty" in data:
            plan.range_uncertainty = data["range_uncertainty"]
        
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