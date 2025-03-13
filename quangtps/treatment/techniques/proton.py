
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module for Proton Therapy treatment techniques.

This module provides classes and methods to define and manage Proton Therapy
treatment planning including Passive Scattering and Pencil Beam Scanning techniques.
"""

import uuid
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

from quangtps.treatment.beams.beam import Beam
from quangtps.treatment.machine.proton import ProtonMachine
from quangtps.treatment.fractionation import Fractionation

logger = logging.getLogger(__name__)


class ProtonTherapy:
    """Base class for proton therapy treatment planning."""
    
    def __init__(self, plan_name: str, plan_id: Optional[str] = None, technique_type: str = "Proton"):
        """
        Initialize a proton therapy treatment plan.
        
        Parameters
        ----------
        plan_name : str
            Name of the proton therapy plan
        plan_id : str, optional
            Unique ID of the plan. If not provided, a new ID will be generated.
        technique_type : str, optional
            Specific type of proton therapy (e.g., "PBS", "Passive")
        """
        self.plan_name = plan_name
        self.plan_id = plan_id if plan_id else str(uuid.uuid4())
        
        # Basic plan attributes
        self.technique_type = technique_type
        self.description = ""
        self.status = "DRAFT"  # DRAFT, APPROVED, DELIVERED, ARCHIVED
        
        # Treatment machine
        self.treatment_machine: Optional[ProtonMachine] = None
        
        # Proton-specific attributes
        self.beams: List[Beam] = []
        self.fractionation: Optional[Fractionation] = None
        self.robustness_settings: Dict[str, Any] = {
            "setup_uncertainty": 3.0,  # mm
            "range_uncertainty": 3.5,  # % of nominal range
            "scenarios": ["nominal", "setup_x+", "setup_x-", "setup_y+", "setup_y-", 
                          "setup_z+", "setup_z-", "range+", "range-"]
        }
        self.margin_recipe: Dict[str, float] = {
            "GTV_to_CTV": 0.0,  # mm
            "CTV_to_PTV": 5.0,  # mm - default margin for proton therapy
        }
        
        # Plan evaluation
        self.plan_quality_metrics: Dict[str, float] = {}
        self.robustness_evaluation: Dict[str, Dict[str, float]] = {}
        
        logger.info(f"Created new {technique_type} plan: {plan_name} (ID: {self.plan_id})")
    
    def add_beam(self, beam: Beam) -> None:
        """
        Add a beam to the proton therapy plan.
        
        Parameters
        ----------
        beam : Beam
            The beam to add
        """
        if beam not in self.beams:
            self.beams.append(beam)
            logger.info(f"Added beam {beam.beam_name} to {self.technique_type} plan {self.plan_name}")
    
    def set_fractionation(self, fractionation: Fractionation) -> None:
        """
        Set the fractionation scheme for the proton therapy plan.
        
        Parameters
        ----------
        fractionation : Fractionation
            Fractionation scheme
        """
        self.fractionation = fractionation
        logger.info(f"Set fractionation for {self.technique_type} plan {self.plan_name}: "
                   f"{fractionation.num_fractions} fractions, "
                   f"{fractionation.dose_per_fraction} Gy(RBE) per fraction")
    
    def set_treatment_machine(self, machine: ProtonMachine) -> None:
        """
        Set the treatment machine for the proton therapy plan.
        
        Parameters
        ----------
        machine : ProtonMachine
            Treatment machine
        """
        self.treatment_machine = machine
        logger.info(f"Set treatment machine for {self.technique_type} plan {self.plan_name}: {machine.name}")
    
    def set_robustness_settings(self, setup_uncertainty: float = 3.0, range_uncertainty: float = 3.5) -> None:
        """
        Set robustness settings for plan optimization and evaluation.
        
        Parameters
        ----------
        setup_uncertainty : float, optional
            Setup uncertainty in mm
        range_uncertainty : float, optional
            Range uncertainty as percentage of nominal range
        """
        self.robustness_settings = {
            "setup_uncertainty": setup_uncertainty,
            "range_uncertainty": range_uncertainty,
            "scenarios": ["nominal", "setup_x+", "setup_x-", "setup_y+", "setup_y-", 
                          "setup_z+", "setup_z-", "range+", "range-"]
        }
        
        logger.info(f"Set robustness settings for {self.technique_type} plan {self.plan_name}: "
                   f"setup = {setup_uncertainty} mm, range = {range_uncertainty}%")
    
    def calculate_dose(self, robust: bool = True) -> Dict[str, np.ndarray]:
        """
        Calculate the dose distribution for the proton therapy plan.
        
        Parameters
        ----------
        robust : bool, optional
            If True, calculate dose for all robustness scenarios
            
        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary of 3D dose distributions for each scenario
        """
        # This would implement a complex dose calculation algorithm
        # For now, we'll return a placeholder
        logger.info(f"Calculating {'robust ' if robust else ''}dose for {self.technique_type} plan {self.plan_name}")
        
        result = {"nominal": np.zeros((100, 100, 100))}  # Placeholder
        
        if robust:
            # Add dose distributions for robustness scenarios
            for scenario in self.robustness_settings["scenarios"]:
                if scenario != "nominal":
                    result[scenario] = np.zeros((100, 100, 100))  # Placeholder
        
        return result
    
    def evaluate_plan(self, robust: bool = True) -> Dict[str, Any]:
        """
        Evaluate the quality of the proton therapy plan.
        
        Parameters
        ----------
        robust : bool, optional
            If True, evaluate plan robustness
            
        Returns
        -------
        Dict[str, Any]
            Dictionary of plan quality metrics
        """
        # Calculate plan quality metrics
        metrics = {
            "homogeneity_index": 1.05,       # Placeholder value
            "conformity_index": 0.95,        # Placeholder value
            "coverage": 0.98,                # Placeholder value (98%)
            "maximum_dose": 105.0,           # % of prescription dose
            "average_dose": 101.5,           # % of prescription dose
            "minimum_dose": 95.0             # % of prescription dose
        }
        
        self.plan_quality_metrics = metrics
        
        if robust:
            # Calculate robustness metrics
            robust_metrics = {}
            for scenario in self.robustness_settings["scenarios"]:
                robust_metrics[scenario] = {
                    "coverage": 0.95 if scenario != "nominal" else 0.98,  # Placeholder values
                    "maximum_dose": 107.0 if scenario != "nominal" else 105.0,
                    "minimum_dose": 90.0 if scenario != "nominal" else 95.0
                }
            
            self.robustness_evaluation = robust_metrics
            
            # Include worst-case scenario metrics
            metrics["worst_case_coverage"] = min(scenario["coverage"] for scenario in robust_metrics.values())
            metrics["worst_case_max_dose"] = max(scenario["maximum_dose"] for scenario in robust_metrics.values())
            metrics["worst_case_min_dose"] = min(scenario["minimum_dose"] for scenario in robust_metrics.values())
        
        logger.info(f"Evaluated {self.technique_type} plan {self.plan_name}: "
                   f"coverage={metrics['coverage']:.2f}, "
                   f"HI={metrics['homogeneity_index']:.2f}")
        
        return metrics
    
    def __str__(self) -> str:
        """Return string representation of the proton therapy plan."""
        return f"{self.technique_type} Plan: {self.plan_name} (ID: {self.plan_id})"


class PencilBeamScanning(ProtonTherapy):
    """
    Class representing a Pencil Beam Scanning (PBS) proton therapy plan.
    
    PBS is a modern proton therapy technique that uses magnetically scanned 
    narrow proton beams ("pencil beams") to precisely target the tumor volume.
    It provides superior dose conformity compared to passive scattering.
    """
    
    def __init__(self, plan_name: str, plan_id: Optional[str] = None):
        """
        Initialize a PBS proton therapy plan.
        
        Parameters
        ----------
        plan_name : str
            Name of the PBS plan
        plan_id : str, optional
            Unique ID of the plan. If not provided, a new ID will be generated.
        """
        super().__init__(plan_name, plan_id, technique_type="PBS")
        
        # PBS-specific attributes
        self.spot_map: Dict[str, List[Tuple[float, float, float, float]]] = {}  # Key: beam_id, Value: List of (x, y, energy, weight) tuples
        self.optimization_type = "robust"  # "robust" or "conventional"
        self.energy_layers: Dict[str, List[float]] = {}  # Key: beam_id, Value: List of energies
        self.scanning_pattern = "continuous"  # "continuous" or "discrete" or "line"
        self.layer_spacing = 5.0  # mm water-equivalent pathlength
        self.spot_spacing = 5.0  # mm at isocenter
        
        # PBS optimization objectives
        self.objectives = []
    
    def set_spot_spacing(self, spot_spacing: float) -> None:
        """
        Set the spot spacing for PBS plan.
        
        Parameters
        ----------
        spot_spacing : float
            Spot spacing in mm at isocenter
        """
        self.spot_spacing = spot_spacing
        logger.info(f"Set spot spacing for PBS plan {self.plan_name}: {spot_spacing} mm")
    
    def set_layer_spacing(self, layer_spacing: float) -> None:
        """
        Set the energy layer spacing for PBS plan.
        
        Parameters
        ----------
        layer_spacing : float
            Layer spacing in mm water-equivalent pathlength
        """
        self.layer_spacing = layer_spacing
        logger.info(f"Set layer spacing for PBS plan {self.plan_name}: {layer_spacing} mm WEL")
    
    def set_scanning_pattern(self, pattern: str) -> None:
        """
        Set the scanning pattern for PBS plan.
        
        Parameters
        ----------
        pattern : str
            Scanning pattern, one of "continuous", "discrete", or "line"
            
        Raises
        ------
        ValueError
            If pattern is not recognized
        """
        valid_patterns = ["continuous", "discrete", "line"]
        if pattern not in valid_patterns:
            raise ValueError(f"Scanning pattern must be one of {valid_patterns}, got {pattern}")
        
        self.scanning_pattern = pattern
        logger.info(f"Set scanning pattern for PBS plan {self.plan_name}: {pattern}")
    
    def set_optimization_type(self, opt_type: str) -> None:
        """
        Set the optimization type for PBS plan.
        
        Parameters
        ----------
        opt_type : str
            Optimization type, one of "robust" or "conventional"
            
        Raises
        ------
        ValueError
            If opt_type is not recognized
        """
        valid_types = ["robust", "conventional"]
        if opt_type not in valid_types:
            raise ValueError(f"Optimization type must be one of {valid_types}, got {opt_type}")
        
        self.optimization_type = opt_type
        logger.info(f"Set optimization type for PBS plan {self.plan_name}: {opt_type}")
    
    def add_optimization_objective(self, structure: str, objective_type: str, dose: float, 
                                  volume: Optional[float] = None, weight: float = 1.0) -> None:
        """
        Add an optimization objective for the PBS plan.
        
        Parameters
        ----------
        structure : str
            Name of the structure
        objective_type : str
            Type of objective (e.g., "max_dose", "min_dose", "min_dvh", "max_dvh", "mean_dose")
        dose : float
            Dose value in Gy(RBE)
        volume : float, optional
            Volume value in percentage (for DVH objectives)
        weight : float, optional
            Weight of the objective
        """
        objective = {
            "structure": structure,
            "type": objective_type,
            "dose": dose,
            "weight": weight
        }
        
        if volume is not None and objective_type in ["min_dvh", "max_dvh"]:
            objective["volume"] = volume
        
        self.objectives.append(objective)
        
        logger.info(f"Added optimization objective for PBS plan {self.plan_name}: "
                   f"{structure}, {objective_type}, {dose} Gy(RBE)")
    
    def generate_spot_map(self) -> Dict[str, List[Tuple[float, float, float, float]]]:
        """
        Generate the spot map for all beams in the PBS plan.
        
        Returns
        -------
        Dict[str, List[Tuple[float, float, float, float]]]
            Spot map for each beam (x, y, energy, weight)
        """
        # This would be a complex algorithm to generate spot positions, energies, and weights
        # For now, we'll generate a dummy spot map
        logger.info(f"Generating spot map for PBS plan {self.plan_name}")
        
        for beam in self.beams:
            # Generate dummy spot map
            num_layers = 10
            num_spots_per_layer = 100
            
            # Create dummy energy layers
            energies = [100.0 + i * 10.0 for i in range(num_layers)]
            self.energy_layers[beam.beam_id] = energies
            
            # Create dummy spot map
            spots = []
            for energy in energies:
                for i in range(int(np.sqrt(num_spots_per_layer))):
                    for j in range(int(np.sqrt(num_spots_per_layer))):
                        x = -25.0 + i * 5.0  # -25 to 25 mm
                        y = -25.0 + j * 5.0  # -25 to 25 mm
                        weight = 1.0  # Initial weight
                        spots.append((x, y, energy, weight))
            
            self.spot_map[beam.beam_id] = spots
        
        return self.spot_map
    
    def optimize_spot_weights(self) -> None:
        """
        Optimize spot weights for the PBS plan.
        
        This would implement an optimization algorithm to determine the optimal
        spot weights to meet the planning objectives.
        """
        logger.info(f"Optimizing spot weights for PBS plan {self.plan_name}")
        
        if not self.spot_map:
            self.generate_spot_map()
        
        # In a real implementation, this would run an optimization algorithm
        # For now, we'll just assign random weights to the spots
        for beam_id, spots in self.spot_map.items():
            updated_spots = []
            for x, y, energy, _ in spots:
                # Assign a random weight between 0 and 2
                weight = np.random.random() * 2.0
                updated_spots.append((x, y, energy, weight))
            
            self.spot_map[beam_id] = updated_spots
        
        logger.info(f"Completed spot weight optimization for PBS plan {self.plan_name}")


class PassiveScattering(ProtonTherapy):
    """
    Class representing a Passive Scattering proton therapy plan.
    
    Passive scattering is a traditional proton therapy technique that uses
    scattering devices to spread out the proton beam and a range compensator
    to conform the dose to the distal edge of the target.
    """
    
    def __init__(self, plan_name: str, plan_id: Optional[str] = None):
        """
        Initialize a Passive Scattering proton therapy plan.
        
        Parameters
        ----------
        plan_name : str
            Name of the passive scattering plan
        plan_id : str, optional
            Unique ID of the plan. If not provided, a new ID will be generated.
        """
        super().__init__(plan_name, plan_id, technique_type="Passive")
        
        # Passive scattering-specific attributes
        self.range_compensators: Dict[str, Any] = {}  # Key: beam_id, Value: compensator details
        self.apertures: Dict[str, Any] = {}  # Key: beam_id, Value: aperture details
        self.smear_margins: Dict[str, float] = {}  # Key: beam_id, Value: smear margin in mm
        self.range_modulation: Dict[str, Tuple[float, float]] = {}  # Key: beam_id, Value: (modulation width, modulation center) in mm
        
    def add_aperture(self, beam_id: str, aperture_data: Dict[str, Any]) -> None:
        """
        Add an aperture for a beam in the passive scattering plan.
        
        Parameters
        ----------
        beam_id : str
            ID of the beam
        aperture_data : Dict[str, Any]
            Aperture details including contour points, material, thickness, etc.
        """
        self.apertures[beam_id] = aperture_data
        logger.info(f"Added aperture for beam {beam_id} in passive scattering plan {self.plan_name}")
    
    def add_range_compensator(self, beam_id: str, compensator_data: Dict[str, Any]) -> None:
        """
        Add a range compensator for a beam in the passive scattering plan.
        
        Parameters
        ----------
        beam_id : str
            ID of the beam
        compensator_data : Dict[str, Any]
            Compensator details including thickness map, material, etc.
        """
        self.range_compensators[beam_id] = compensator_data
        logger.info(f"Added range compensator for beam {beam_id} in passive scattering plan {self.plan_name}")
    
    def set_smear_margin(self, beam_id: str, margin: float) -> None:
        """
        Set the smear margin for a beam in the passive scattering plan.
        
        Parameters
        ----------
        beam_id : str
            ID of the beam
        margin : float
            Smear margin in mm
        """
        self.smear_margins[beam_id] = margin
        logger.info(f"Set smear margin for beam {beam_id} in passive scattering plan {self.plan_name}: {margin} mm")
    
    def set_range_modulation(self, beam_id: str, width: float, center: float) -> None:
        """
        Set the range modulation for a beam in the passive scattering plan.
        
        Parameters
        ----------
        beam_id : str
            ID of the beam
        width : float
            Modulation width in mm
        center : float
            Modulation center in mm
        """
        self.range_modulation[beam_id] = (width, center)
        logger.info(f"Set range modulation for beam {beam_id} in passive scattering plan {self.plan_name}: "
                   f"width = {width} mm, center = {center} mm")
    
    def design_range_compensator(self, beam_id: str) -> None:
        """
        Design a range compensator for a beam in the passive scattering plan.
        
        Parameters
        ----------
        beam_id : str
            ID of the beam
        """
        logger.info(f"Designing range compensator for beam {beam_id} in passive scattering plan {self.plan_name}")
        
        # In a real implementation, this would design a range compensator
        # based on patient anatomy and beam properties
        # For now, we'll create a dummy compensator
        compensator_data = {
            "material": "Lucite",
            "max_thickness": 50.0,  # mm
            "grid_size": (50, 50),  # 50x50 grid
            "pixel_size": 2.0,  # mm
            "thickness_map": np.random.rand(50, 50) * 50.0  # Random thickness map
        }
        
        self.add_range_compensator(beam_id, compensator_data)
    
    def design_aperture(self, beam_id: str) -> None:
        """
        Design an aperture for a beam in the passive scattering plan.
        
        Parameters
        ----------
        beam_id : str
            ID of the beam
        """
        logger.info(f"Designing aperture for beam {beam_id} in passive scattering plan {self.plan_name}")
        
        # In a real implementation, this would design an aperture
        # based on patient anatomy and beam properties
        # For now, we'll create a dummy aperture
        aperture_data = {
            "material": "Brass",
            "thickness": 60.0,  # mm
            "contour_points": [(x, y) for x in range(-30, 31, 10) for y in range(-30, 31, 10)],
            "margin": 5.0  # mm
        }
        
        self.add_aperture(beam_id, aperture_data)