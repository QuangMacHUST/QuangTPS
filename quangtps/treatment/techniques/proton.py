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
from typing import List, Dict, Any, Optional, Tuple, Union

from quangtps.treatment.beams.beam import Beam
from quangtps.treatment.machine.proton import ProtonMachine
from quangtps.treatment.machine.treatment_machine import TreatmentMachine
from quangtps.treatment.fractionation import Fractionation
from quangtps.treatment.techniques.technique_interface import BaseTreatmentTechnique, TechniqueCategory
from quangtps.treatment.techniques.treatment_technique import TreatmentTechnique

logger = logging.getLogger(__name__)


class Proton(TreatmentTechnique):
    """
    Lớp đại diện cho kỹ thuật xạ trị proton.
    
    Kỹ thuật xạ trị proton sử dụng chùm proton để điều trị ung thư,
    với lợi thế chính là phân bố liều theo đỉnh Bragg, nơi hầu hết
    năng lượng được giải phóng ở cuối quãng đường của chùm tia.
    """
    
    def __init__(self, technique_name: str = "Proton"):
        """
        Khởi tạo kỹ thuật xạ trị proton.
        
        Parameters
        ----------
        technique_name : str, optional
            Tên của kỹ thuật, mặc định là "Proton"
        """
        super().__init__(technique_name)
        self.delivery_method = None  # PBS (Pencil Beam Scanning), US (Uniform Scanning), DS (Double Scattering)
        self.energy_range = (70, 250)  # MeV, mặc định
        self.range_modulation = None  # Modulation width (g/cm^2)
        self.spot_size = None  # mm, cho PBS
        self.spot_spacing = None  # mm, cho PBS
        self.layer_spacing = None  # mm, cho PBS
        self.has_range_shifter = False
        self.range_shifter_thickness = None  # mm water equivalent
    
    def set_delivery_method(self, method: str):
        """
        Thiết lập phương pháp phân phối chùm tia proton.
        
        Parameters
        ----------
        method : str
            Phương pháp phân phối: "PBS", "US", "DS"
        """
        valid_methods = ["PBS", "US", "DS"]
        if method not in valid_methods:
            logger.warning(f"Phương pháp phân phối không hợp lệ: {method}. Phải là một trong {valid_methods}")
            return
        
        self.delivery_method = method
        logger.info(f"Đã thiết lập phương pháp phân phối proton: {method}")
    
    def set_energy_range(self, min_energy: float, max_energy: float):
        """
        Thiết lập phạm vi năng lượng proton.
        
        Parameters
        ----------
        min_energy : float
            Năng lượng tối thiểu (MeV)
        max_energy : float
            Năng lượng tối đa (MeV)
        """
        if min_energy <= 0 or max_energy <= 0 or min_energy >= max_energy:
            logger.warning(f"Phạm vi năng lượng không hợp lệ: {min_energy}-{max_energy} MeV")
            return
        
        self.energy_range = (min_energy, max_energy)
        logger.info(f"Đã thiết lập phạm vi năng lượng proton: {min_energy}-{max_energy} MeV")
    
    def set_range_modulation(self, modulation_width: float):
        """
        Thiết lập độ rộng điều chế phạm vi.
        
        Parameters
        ----------
        modulation_width : float
            Độ rộng điều chế (g/cm^2)
        """
        if modulation_width <= 0:
            logger.warning(f"Độ rộng điều chế không hợp lệ: {modulation_width} g/cm^2")
            return
        
        self.range_modulation = modulation_width
        logger.info(f"Đã thiết lập độ rộng điều chế: {modulation_width} g/cm^2")
    
    def configure_pbs(self, spot_size: float, spot_spacing: float, layer_spacing: float):
        """
        Cấu hình thông số cho phương pháp PBS (Pencil Beam Scanning).
        
        Parameters
        ----------
        spot_size : float
            Kích thước điểm (mm)
        spot_spacing : float
            Khoảng cách giữa các điểm (mm)
        layer_spacing : float
            Khoảng cách giữa các lớp (mm)
        """
        if self.delivery_method != "PBS":
            logger.warning("Không thể cấu hình PBS khi phương pháp phân phối không phải là PBS")
            return
        
        if spot_size <= 0 or spot_spacing <= 0 or layer_spacing <= 0:
            logger.warning("Thông số PBS không hợp lệ")
            return
        
        self.spot_size = spot_size
        self.spot_spacing = spot_spacing
        self.layer_spacing = layer_spacing
        logger.info(f"Đã cấu hình PBS với kích thước điểm: {spot_size} mm, "
                   f"khoảng cách điểm: {spot_spacing} mm, khoảng cách lớp: {layer_spacing} mm")
    
    def add_range_shifter(self, thickness: float):
        """
        Thêm range shifter để điều chỉnh phạm vi chùm tia.
        
        Parameters
        ----------
        thickness : float
            Độ dày của range shifter (mm water equivalent)
        """
        if thickness <= 0:
            logger.warning(f"Độ dày range shifter không hợp lệ: {thickness} mm")
            return
        
        self.has_range_shifter = True
        self.range_shifter_thickness = thickness
        logger.info(f"Đã thêm range shifter với độ dày: {thickness} mm water equivalent")
    
    def remove_range_shifter(self):
        """Loại bỏ range shifter."""
        self.has_range_shifter = False
        self.range_shifter_thickness = None
        logger.info("Đã loại bỏ range shifter")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Chuyển đổi thông tin kỹ thuật xạ trị proton thành dictionary.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary chứa thông tin kỹ thuật
        """
        data = super().to_dict()
        data.update({
            "delivery_method": self.delivery_method,
            "energy_range": self.energy_range,
            "range_modulation": self.range_modulation,
            "spot_size": self.spot_size,
            "spot_spacing": self.spot_spacing,
            "layer_spacing": self.layer_spacing,
            "has_range_shifter": self.has_range_shifter,
            "range_shifter_thickness": self.range_shifter_thickness
        })
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Proton':
        """
        Tạo đối tượng Proton từ dictionary.
        
        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary chứa thông tin kỹ thuật
            
        Returns
        -------
        Proton
            Đối tượng Proton
        """
        technique = cls(data.get("technique_name", "Proton"))
        
        # Thiết lập các thuộc tính
        if "delivery_method" in data:
            technique.set_delivery_method(data["delivery_method"])
        
        if "energy_range" in data and isinstance(data["energy_range"], tuple) and len(data["energy_range"]) == 2:
            technique.set_energy_range(data["energy_range"][0], data["energy_range"][1])
        
        if "range_modulation" in data and data["range_modulation"] is not None:
            technique.set_range_modulation(data["range_modulation"])
        
        if (data.get("delivery_method") == "PBS" and
            "spot_size" in data and "spot_spacing" in data and "layer_spacing" in data):
            technique.configure_pbs(
                data["spot_size"],
                data["spot_spacing"],
                data["layer_spacing"]
            )
        
        if data.get("has_range_shifter", False) and "range_shifter_thickness" in data:
            technique.add_range_shifter(data["range_shifter_thickness"])
        
        return technique


class ProtonTherapy(BaseTreatmentTechnique):
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
        super().__init__(
            name=plan_name,
            technique_id=plan_id,
            category=TechniqueCategory.PARTICLE
        )
        
        # Basic plan attributes
        self.technique_type = technique_type
        self.description = ""
        self.status = "DRAFT"  # DRAFT, APPROVED, DELIVERED, ARCHIVED
        
        # Proton-specific attributes
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
        
        logger.info(f"Created new {technique_type} plan: {plan_name} (ID: {self.technique_id})")
    
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
        Add a beam to the proton therapy plan.
        
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
        Get the list of beams in the proton therapy plan.
        
        Returns
        -------
        List[Beam]
            List of beams in the plan
        """
        return self.beams
    
    def set_fractionation(self, fractionation: Fractionation) -> None:
        """
        Set the fractionation scheme for the proton therapy plan.
        
        Parameters
        ----------
        fractionation : Fractionation
            Fractionation scheme
        """
        self.fractionation = fractionation
        logger.info(f"Set fractionation for {self.technique_type} plan {self.name}: "
                   f"{fractionation.num_fractions} fractions, "
                   f"{fractionation.dose_per_fraction} Gy(RBE) per fraction")
    
    def set_machine(self, machine: TreatmentMachine) -> None:
        """
        Set the treatment machine for the proton therapy plan.
        
        Parameters
        ----------
        machine : TreatmentMachine
            Treatment machine
        """
        if not isinstance(machine, ProtonMachine):
            raise ValueError(f"Proton therapy requires a ProtonMachine, got {type(machine).__name__}")
            
        self.machine = machine
        logger.info(f"Set treatment machine for {self.technique_type} plan {self.name}: {machine.name}")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the proton therapy plan to a dictionary.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary representation of the plan
        """
        # Start with the base technique dictionary
        result = super().to_dict()
        
        # Add proton-specific attributes
        result.update({
            "technique_type": self.technique_type,
            "description": self.description,
            "status": self.status,
            "robustness_settings": self.robustness_settings,
            "margin_recipe": self.margin_recipe,
            "plan_quality_metrics": self.plan_quality_metrics,
            "robustness_evaluation": self.robustness_evaluation
        })
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProtonTherapy':
        """
        Create a proton therapy plan from a dictionary.
        
        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary with plan data
            
        Returns
        -------
        ProtonTherapy
            ProtonTherapy instance
        """
        plan = cls(
            plan_name=data["name"],
            plan_id=data["technique_id"],
            technique_type=data.get("technique_type", "Proton")
        )
        
        # Restore basic attributes
        plan.description = data.get("description", "")
        plan.status = data.get("status", "DRAFT")
        
        # Restore proton-specific attributes
        if "robustness_settings" in data:
            plan.robustness_settings = data["robustness_settings"]
        
        if "margin_recipe" in data:
            plan.margin_recipe = data["margin_recipe"]
        
        if "plan_quality_metrics" in data:
            plan.plan_quality_metrics = data["plan_quality_metrics"]
        
        if "robustness_evaluation" in data:
            plan.robustness_evaluation = data["robustness_evaluation"]
        
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
        logger.info(f"Calculating {'robust ' if robust else ''}dose for {self.technique_type} plan {self.name}")
        
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
        
        logger.info(f"Evaluated {self.technique_type} plan {self.name}: "
                   f"coverage={metrics['coverage']:.2f}, "
                   f"HI={metrics['homogeneity_index']:.2f}")
        
        return metrics
    
    def __str__(self) -> str:
        """Return string representation of the proton therapy plan."""
        return f"{self.technique_type} Plan: {self.name} (ID: {self.technique_id})"


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
        logger.info(f"Set spot spacing for PBS plan {self.name}: {spot_spacing} mm")
    
    def set_layer_spacing(self, layer_spacing: float) -> None:
        """
        Set the energy layer spacing for PBS plan.
        
        Parameters
        ----------
        layer_spacing : float
            Layer spacing in mm water-equivalent pathlength
        """
        self.layer_spacing = layer_spacing
        logger.info(f"Set layer spacing for PBS plan {self.name}: {layer_spacing} mm WEL")
    
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
        logger.info(f"Set scanning pattern for PBS plan {self.name}: {pattern}")
    
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
        logger.info(f"Set optimization type for PBS plan {self.name}: {opt_type}")
    
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
        
        logger.info(f"Added optimization objective for PBS plan {self.name}: "
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
        logger.info(f"Generating spot map for PBS plan {self.name}")
        
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
        logger.info(f"Optimizing spot weights for PBS plan {self.name}")
        
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
        
        logger.info(f"Completed spot weight optimization for PBS plan {self.name}")


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
        logger.info(f"Added aperture for beam {beam_id} in passive scattering plan {self.name}")
    
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
        logger.info(f"Added range compensator for beam {beam_id} in passive scattering plan {self.name}")
    
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
        logger.info(f"Set smear margin for beam {beam_id} in passive scattering plan {self.name}: {margin} mm")
    
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
        logger.info(f"Set range modulation for beam {beam_id} in passive scattering plan {self.name}: "
                   f"width = {width} mm, center = {center} mm")
    
    def design_range_compensator(self, beam_id: str) -> None:
        """
        Design a range compensator for a beam in the passive scattering plan.
        
        Parameters
        ----------
        beam_id : str
            ID of the beam
        """
        logger.info(f"Designing range compensator for beam {beam_id} in passive scattering plan {self.name}")
        
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
        logger.info(f"Designing aperture for beam {beam_id} in passive scattering plan {self.name}")
        
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