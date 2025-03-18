#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module for Pencil Beam Scanning (PBS) techniques.

This module provides classes for configuring and managing Pencil Beam Scanning,
which is an advanced form of proton therapy that uses a narrow proton beam
to precisely deliver dose to the target volume.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from enum import Enum
import numpy as np

from quangtps.treatment.beams.beam import Beam
from quangtps.treatment.machine.treatment_machine import TreatmentMachine
from quangtps.treatment.fractionation import Fractionation
from quangtps.treatment.techniques.technique_interface import BaseTreatmentTechnique, TechniqueCategory

logger = logging.getLogger(__name__)

class SpotDeliveryPattern(str, Enum):
    """Enum for different spot delivery patterns in PBS."""
    REGULAR_GRID = "REGULAR_GRID"        # Regular grid pattern
    ADAPTIVE_GRID = "ADAPTIVE_GRID"      # Adaptive grid with variable spacing
    CONTOUR_SCANNING = "CONTOUR_SCANNING" # Contour-based scanning
    
class OptimizationStrategy(str, Enum):
    """Enum for optimization strategies in PBS."""
    SINGLE_FIELD = "SINGLE_FIELD"        # Single-field optimization (SFO)
    MULTI_FIELD = "MULTI_FIELD"          # Multi-field optimization (MFO)
    ROBUST = "ROBUST"                    # Robust optimization

class SpotRepresentation(str, Enum):
    """Enum for spot representation methods."""
    GAUSSIAN = "GAUSSIAN"                # Gaussian spot model
    DOUBLE_GAUSSIAN = "DOUBLE_GAUSSIAN"  # Double Gaussian spot model
    MEASURED = "MEASURED"                # Measured spot profile

class PencilBeamScanning(BaseTreatmentTechnique):
    """
    Class for Pencil Beam Scanning (PBS) technique.
    
    PBS is an advanced form of proton therapy that uses a narrow proton beam
    to "paint" radiation dose layer by layer throughout the target volume.
    It allows for more precise dose distribution and better sparing of 
    normal tissues compared to traditional proton scattering techniques.
    """
    
    def __init__(self, 
                 name: str,
                 delivery_pattern: SpotDeliveryPattern = SpotDeliveryPattern.REGULAR_GRID,
                 optimization_strategy: OptimizationStrategy = OptimizationStrategy.MULTI_FIELD,
                 spot_representation: SpotRepresentation = SpotRepresentation.GAUSSIAN,
                 technique_id: Optional[str] = None):
        """
        Initialize a Pencil Beam Scanning treatment.
        
        Parameters
        ----------
        name : str
            Name of the PBS treatment
        delivery_pattern : SpotDeliveryPattern
            Pattern for spot delivery
        optimization_strategy : OptimizationStrategy
            Strategy for optimization
        spot_representation : SpotRepresentation
            Method for spot representation
        technique_id : str, optional
            Unique ID for the PBS treatment
        """
        super().__init__(
            name=name, 
            technique_id=technique_id, 
            category=TechniqueCategory.ADVANCED
        )
        
        self.delivery_pattern = delivery_pattern
        self.optimization_strategy = optimization_strategy
        self.spot_representation = spot_representation
        
        # PBS-specific parameters
        self.spot_size = 5.0  # mm, sigma of the Gaussian spot at isocenter
        self.spot_spacing = 5.0  # mm, spacing between spots
        self.energy_layers = []  # List of energy layers
        self.spot_map = {}  # Map of spots for each energy layer
        self.layer_spacing = 5.0  # mm, spacing between energy layers
        
        # Robustness parameters
        self.setup_uncertainty = 3.0  # mm
        self.range_uncertainty = 3.5  # percent
        self.robust_scenarios = []  # List of robust scenarios
        
        # Dose calculation parameters
        self.dose_grid_resolution = (2.0, 2.0, 2.0)  # mm
        
        # Planning parameters
        self.target_dose_homogeneity = 3.0  # percent
        self.target_coverage = 95.0  # percent
        self.optimization_parameters = {
            "objective_function": "LEAST_SQUARES",
            "max_iterations": 100,
            "convergence_threshold": 0.001
        }
        
        logger.info(f"Initialized Pencil Beam Scanning treatment: {name} (ID: {self.technique_id})")
    
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
    
    def set_spot_parameters(self, spot_size: float, spot_spacing: float):
        """
        Set spot parameters.
        
        Parameters
        ----------
        spot_size : float
            Spot size in mm (sigma)
        spot_spacing : float
            Spot spacing in mm
        """
        self.spot_size = spot_size
        self.spot_spacing = spot_spacing
        logger.info(f"Set spot parameters for treatment {self.name}: "
                   f"size={spot_size} mm, spacing={spot_spacing} mm")
    
    def set_energy_layer_parameters(self, layer_spacing: float):
        """
        Set energy layer parameters.
        
        Parameters
        ----------
        layer_spacing : float
            Spacing between energy layers in mm
        """
        self.layer_spacing = layer_spacing
        logger.info(f"Set energy layer spacing to {layer_spacing} mm for treatment {self.name}")
    
    def add_energy_layer(self, energy_mev: float, spots: List[Tuple[float, float, float]]):
        """
        Add an energy layer with spots.
        
        Parameters
        ----------
        energy_mev : float
            Energy in MeV
        spots : List[Tuple[float, float, float]]
            List of spots as (x, y, weight) tuples
        """
        self.energy_layers.append(energy_mev)
        self.spot_map[energy_mev] = spots
        logger.info(f"Added energy layer at {energy_mev} MeV with {len(spots)} spots "
                   f"for treatment {self.name}")
    
    def set_robust_parameters(self, setup_uncertainty: float, range_uncertainty: float):
        """
        Set parameters for robust optimization.
        
        Parameters
        ----------
        setup_uncertainty : float
            Setup uncertainty in mm
        range_uncertainty : float
            Range uncertainty in percent
        """
        self.setup_uncertainty = setup_uncertainty
        self.range_uncertainty = range_uncertainty
        
        # Generate robust scenarios
        if self.optimization_strategy == OptimizationStrategy.ROBUST:
            self._generate_robust_scenarios()
            
        logger.info(f"Set robust parameters for treatment {self.name}: "
                   f"setup uncertainty={setup_uncertainty} mm, "
                   f"range uncertainty={range_uncertainty}%")
    
    def _generate_robust_scenarios(self):
        """
        Generate robust scenarios based on setup and range uncertainties.
        """
        # Nominal scenario
        scenarios = [{"name": "nominal", "setup_shift": (0, 0, 0), "range_shift": 0.0}]
        
        # Setup uncertainty scenarios
        for axis in ["x", "y", "z"]:
            for direction in [-1, 1]:
                shift = [0, 0, 0]
                if axis == "x":
                    shift[0] = direction * self.setup_uncertainty
                elif axis == "y":
                    shift[1] = direction * self.setup_uncertainty
                else:
                    shift[2] = direction * self.setup_uncertainty
                    
                scenario_name = f"{axis}{'+' if direction > 0 else '-'}"
                scenarios.append({
                    "name": scenario_name,
                    "setup_shift": tuple(shift),
                    "range_shift": 0.0
                })
        
        # Range uncertainty scenarios
        for direction in [-1, 1]:
            range_shift = direction * self.range_uncertainty
            scenario_name = f"range{'+' if direction > 0 else '-'}"
            scenarios.append({
                "name": scenario_name,
                "setup_shift": (0, 0, 0),
                "range_shift": range_shift
            })
            
        self.robust_scenarios = scenarios
        logger.info(f"Generated {len(scenarios)} robust scenarios for treatment {self.name}")
    
    def set_optimization_parameters(self, 
                                   objective_function: str, 
                                   max_iterations: int, 
                                   convergence_threshold: float):
        """
        Set optimization parameters.
        
        Parameters
        ----------
        objective_function : str
            Objective function type
        max_iterations : int
            Maximum number of iterations
        convergence_threshold : float
            Convergence threshold
        """
        self.optimization_parameters = {
            "objective_function": objective_function,
            "max_iterations": max_iterations,
            "convergence_threshold": convergence_threshold
        }
        logger.info(f"Set optimization parameters for treatment {self.name}: "
                   f"objective={objective_function}, "
                   f"max_iterations={max_iterations}, "
                   f"threshold={convergence_threshold}")
    
    def optimize_spot_weights(self):
        """
        Optimize the spot weights to achieve the desired dose distribution.
        This is a placeholder for the actual optimization algorithm.
        
        Returns
        -------
        bool
            True if optimization was successful, False otherwise
        """
        # This would be a complex optimization algorithm in a real implementation
        # For now, we'll just simulate success
        logger.info(f"Optimizing spot weights for treatment {self.name} "
                   f"with strategy {self.optimization_strategy}")
        
        # Simulated optimization result
        return True
    
    def calculate_spot_positions(self, 
                                target_volume: np.ndarray, 
                                spacing_override: Optional[float] = None):
        """
        Calculate optimal spot positions based on target volume.
        This is a placeholder for the actual spot position calculation algorithm.
        
        Parameters
        ----------
        target_volume : np.ndarray
            3D array representing the target volume
        spacing_override : float, optional
            Override the default spot spacing
            
        Returns
        -------
        Dict[float, List[Tuple[float, float, float]]]
            Dictionary mapping energy levels to lists of spot positions and weights
        """
        logger.info(f"Calculating spot positions for treatment {self.name} "
                   f"with pattern {self.delivery_pattern}")
        
        # Use the specified spacing or the default
        spacing = spacing_override or self.spot_spacing
        
        # This would be a complex algorithm in a real implementation
        # For now, we'll just return an empty dictionary
        return {}
    
    def set_fractionation(self, fractionation: Fractionation) -> None:
        """
        Set the fractionation for the PBS treatment.
        
        Parameters
        ----------
        fractionation : Fractionation
            The fractionation scheme
        """
        self.fractionation = fractionation
        logger.info(f"Set fractionation to {fractionation.total_dose} Gy in {fractionation.num_fractions} "
                   f"fractions for PBS treatment '{self.name}'")
    
    def set_machine(self, machine: TreatmentMachine) -> None:
        """
        Set the treatment machine for the PBS treatment.
        
        Parameters
        ----------
        machine : TreatmentMachine
            The treatment machine to use
        """
        self.machine = machine
        logger.info(f"Set treatment machine to {machine.name} for PBS treatment '{self.name}'")
        
        # Verify machine compatibility
        if not hasattr(machine, 'supports_pbs') or not machine.supports_pbs:
            logger.warning(f"Machine {machine.name} may not support PBS, which could cause issues")
    
    def add_beam(self, beam: Beam) -> None:
        """
        Add a beam to the PBS plan.
        
        Parameters
        ----------
        beam : Beam
            The beam to add to the plan
        """
        if beam not in self.beams:
            self.beams.append(beam)
            logger.info(f"Added beam {beam.beam_id} to PBS treatment '{self.name}'")
    
    def get_beams(self) -> List[Beam]:
        """
        Get all beams in the PBS plan.
        
        Returns
        -------
        List[Beam]
            List of beams in the plan
        """
        return self.beams
    
    def estimate_delivery_time(self) -> float:
        """
        Estimate the treatment delivery time.
        
        Returns
        -------
        float
            Estimated delivery time in seconds
        """
        # Simple model for delivery time
        # Assuming 10ms per spot switching, 1s per energy layer switching
        total_spots = sum(len(spots) for spots in self.spot_map.values())
        total_layers = len(self.energy_layers)
        
        spot_switching_time = total_spots * 0.01  # seconds
        layer_switching_time = total_layers * 1.0  # seconds
        
        # Adding fixed setup time
        setup_time = 60.0  # seconds
        
        return setup_time + spot_switching_time + layer_switching_time
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert PBS to dictionary.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary representation
        """
        return {
            "id": self.technique_id,
            "name": self.name,
            "category": self.category.value,
            "delivery_pattern": self.delivery_pattern,
            "optimization_strategy": self.optimization_strategy,
            "spot_representation": self.spot_representation,
            "spot_size": self.spot_size,
            "spot_spacing": self.spot_spacing,
            "energy_layers": self.energy_layers,
            "layer_spacing": self.layer_spacing,
            "setup_uncertainty": self.setup_uncertainty,
            "range_uncertainty": self.range_uncertainty,
            "dose_grid_resolution": self.dose_grid_resolution,
            "target_dose_homogeneity": self.target_dose_homogeneity,
            "target_coverage": self.target_coverage,
            "optimization_parameters": self.optimization_parameters
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PencilBeamScanning':
        """
        Create PBS from dictionary.
        
        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary with PBS data
            
        Returns
        -------
        PencilBeamScanning
            PBS instance
        """
        pbs = cls(
            name=data["name"],
            delivery_pattern=SpotDeliveryPattern(data["delivery_pattern"]),
            optimization_strategy=OptimizationStrategy(data["optimization_strategy"]),
            spot_representation=SpotRepresentation(data["spot_representation"]),
            technique_id=data["id"]
        )
        
        # Set PBS-specific parameters
        pbs.spot_size = data.get("spot_size", 5.0)
        pbs.spot_spacing = data.get("spot_spacing", 5.0)
        pbs.energy_layers = data.get("energy_layers", [])
        pbs.layer_spacing = data.get("layer_spacing", 5.0)
        pbs.setup_uncertainty = data.get("setup_uncertainty", 3.0)
        pbs.range_uncertainty = data.get("range_uncertainty", 3.5)
        pbs.dose_grid_resolution = data.get("dose_grid_resolution", (2.0, 2.0, 2.0))
        pbs.target_dose_homogeneity = data.get("target_dose_homogeneity", 3.0)
        pbs.target_coverage = data.get("target_coverage", 95.0)
        pbs.optimization_parameters = data.get("optimization_parameters", {
            "objective_function": "LEAST_SQUARES",
            "max_iterations": 100,
            "convergence_threshold": 0.001
        })
        
        # Generate robust scenarios if using robust optimization
        if pbs.optimization_strategy == OptimizationStrategy.ROBUST:
            pbs._generate_robust_scenarios()
        
        return pbs


# Ensure proper exports
__all__ = ['PencilBeamScanning', 'SpotDeliveryPattern', 'OptimizationStrategy', 'SpotRepresentation']
