#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module for Electron Therapy techniques.

This module provides classes and methods for defining electron therapy
treatment techniques, including standard electron therapy and specialized
variants like total skin electron therapy.
"""

import logging
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np

# Treatment technique imports
from quangtps.treatment.techniques.technique_interface import BaseTreatmentTechnique, TechniqueCategory
from quangtps.treatment.beams.beam import Beam, BeamType
from quangtps.treatment.beams.beam_modifiers import Bolus
from quangtps.treatment.fractionation import Fractionation
from quangtps.treatment.machine.treatment_machine import TreatmentMachine
from quangtps.treatment.techniques.treatment_technique import TreatmentTechnique

logger = logging.getLogger(__name__)


class ElectronApplicationTechnique(str, Enum):
    """Enum for different electron therapy application techniques."""
    STANDARD = "STANDARD"  # Standard electron therapy
    CUSTOM_CUTOUT = "CUSTOM_CUTOUT"  # Shaped aperture using custom cutout
    SKIN_COLLIMATION = "SKIN_COLLIMATION"  # Surface collimation
    TOTAL_SKIN = "TOTAL_SKIN"  # Total skin electron therapy (TSET)
    INTRAOPERATIVE = "INTRAOPERATIVE"  # Intraoperative electron therapy (IOERT)
    BOLUS = "BOLUS"  # With bolus material to modify surface dose


class ElectronTherapy(BaseTreatmentTechnique):
    """
    Class for Electron Therapy treatment technique.
    
    Electron therapy uses electron beams for treating superficial tumors
    with rapid dose falloff after reaching a certain depth.
    """
    
    def __init__(
        self,
        name: str = "Electron Therapy",
        technique_id: Optional[str] = None,
        technique: ElectronApplicationTechnique = ElectronApplicationTechnique.STANDARD,
        energy: Optional[int] = None
    ):
        """
        Initialize an electron therapy technique.
        
        Parameters
        ----------
        name : str, optional
            Name of the treatment
        technique_id : str, optional
            Unique ID for the technique
        technique : ElectronApplicationTechnique, optional
            Application technique (standard, inverted, etc.)
        energy : int, optional
            Electron energy to use
        """
        super().__init__(
            name=name,
            technique_id=technique_id,
            category=TechniqueCategory.CONVENTIONAL
        )
        
        # Treatment technique details
        self.technique = technique
        self.energy = energy
        
        # Applicator and bolus settings
        self.applicator_size_cm = None
        self.bolus_thickness_mm = None
        self.bolus_material = "tissue-equivalent"
        
        # Custom cutout details (if applicable)
        self.cutout_shape = None  # Could be "CIRCULAR", "RECTANGULAR", "CUSTOM"
        self.cutout_dimensions = {}  # e.g. {"width": 5, "height": 5} for rectangular
        
        # Treatment planning details
        self.target_depth_mm = None
        self.prescription_isodose = 90.0  # Default to 90% isodose
        
        # Machine information
        self.machine: Optional[TreatmentMachine] = None
        
        # Beam geometry parameters
        self.gantry_angle = 0.0
        self.collimator_angle = 0.0
        self.ssd_cm = 100.0  # Standard SSD
        
        # Beams
        self.beams: List[Beam] = []
        
        # Fractionation scheme
        self.fractionation: Optional[Fractionation] = None
        self.total_dose = None
        
        # Handle potential None value for energy 
        if self.energy:
            energy_str = self.energy
        else:
            energy_str = "None"
        
        logger.info(
            f"Initialized ElectronTherapy '{self.name}' with technique {self.technique.value}, energy {energy_str} MeV"
        )
    
    def get_name(self) -> str:
        """
        Get the name of the technique.
        
        Returns
        -------
        str
            Technique name
        """
        return self.name
    
    def get_category(self) -> TechniqueCategory:
        """
        Get the category of the technique.
        
        Returns
        -------
        TechniqueCategory
            Category enum value (CONVENTIONAL)
        """
        return self.category
    
    def set_energy(self, energy: int) -> None:
        """
        Set electron energy for treatment.
        
        Parameters
        ----------
        energy : int
            Electron energy in MeV
        """
        self.energy = energy
        logger.info(f"Set energy to {self.energy} MeV for treatment '{self.name}'")
        
    def set_applicator(self, size_cm: int) -> None:
        """
        Set electron applicator size.
        
        Parameters
        ----------
        size_cm : int
            Applicator size in centimeters (typically 6, 10, 14, 20, 25)
        """
        self.applicator_size_cm = size_cm
        logger.info(f"Set applicator size to {size_cm} cm for treatment '{self.name}'")
        
    def set_cutout_shape(self, shape: str) -> None:
        """
        Set custom cutout shape description.
        
        Parameters
        ----------
        shape : str
            Description of the cutout shape
        """
        self.cutout_shape = shape
        logger.info(f"Set cutout shape to '{shape}' for treatment '{self.name}'")
        
    def set_ssd(self, ssd_cm: float) -> None:
        """
        Set source-to-surface distance (SSD).
        
        Parameters
        ----------
        ssd_cm : float
            Source-to-surface distance in centimeters
        """
        self.ssd_cm = ssd_cm
        logger.info(f"Set SSD to {ssd_cm} cm for treatment '{self.name}'")
        
    def set_bolus(self, thickness_mm: float, material: str = "tissue-equivalent") -> None:
        """
        Add bolus material to the treatment.
        
        Parameters
        ----------
        thickness_mm : float
            Thickness of bolus in millimeters
        material : str, optional
            Bolus material type
        """
        self.bolus_thickness_mm = thickness_mm
        self.bolus_material = material
        logger.info(
            f"Added {thickness_mm} mm {material} bolus to treatment '{self.name}'"
        )
        
    def set_target_depth(self, depth_mm: float) -> None:
        """
        Set target depth for electron treatment.
        
        Parameters
        ----------
        depth_mm : float
            Target depth in millimeters
        """
        self.target_depth_mm = depth_mm
        logger.info(f"Set target depth to {depth_mm} mm for treatment '{self.name}'")
        
    def set_prescription_isodose(self, isodose_percent: float) -> None:
        """
        Set the prescription isodose line (typically 80-90% for electrons).
        
        Parameters
        ----------
        isodose_percent : float
            Isodose line as percentage (typically 80-90% for electrons)
        """
        self.prescription_isodose = isodose_percent
        logger.info(f"Set prescription isodose to {isodose_percent}% for treatment '{self.name}'")
        
    def add_beam(self, beam: Beam) -> None:
        """
        Add a beam to the electron treatment.
        
        Parameters
        ----------
        beam : Beam
            The beam to add
        """
        self.beams.append(beam)
        logger.info(f"Added beam '{beam.beam_name}' to treatment '{self.name}'")
        
    def set_machine(self, machine: TreatmentMachine) -> None:
        """
        Set the treatment machine for this electron therapy.
        
        Parameters
        ----------
        machine : TreatmentMachine
            The treatment machine to use
        """
        self.machine = machine
        logger.info(f"Set treatment machine '{machine.name}' for electron therapy '{self.name}'")
        
    def set_cutout(self, shape: str, dimensions: Dict[str, Any]) -> None:
        """
        Set cutout shape and dimensions for electron applicator.
        
        This is a higher-level method that handles custom cutout
        details and propagates them to all beams.
        
        Parameters
        ----------
        shape : str
            Shape descriptor ('CIRCULAR', 'RECTANGULAR', 'CUSTOM')
        dimensions : Dict[str, Any]
            Dimensions of the cutout (e.g., diameter, width, height, etc.)
        """
        self.cutout_shape = shape
        self.cutout_dimensions = dimensions
        
        # Update any existing beams with the cutout information
        for beam in self.beams:
            if hasattr(beam, 'set_custom_cutout'):
                beam.set_custom_cutout(shape, dimensions)
        
        logger.info(f"Set cutout shape '{shape}' with dimensions {dimensions} for treatment '{self.name}'")
        
    def set_fractionation(self, fractionation: Fractionation) -> None:
        """
        Set fractionation scheme for electron treatment.
        
        Parameters
        ----------
        fractionation : Fractionation
            Fractionation scheme details
        """
        self.fractionation = fractionation
        self.total_dose = fractionation.total_dose
        
        logger.info(
            f"Set fractionation to {fractionation.total_dose} Gy in {fractionation.num_fractions} fractions for electron treatment '{self.name}'"
        )
        
    def get_beams(self) -> List[Beam]:
        """
        Get all beams in the electron treatment.
        
        Returns
        -------
        List[Beam]
            List of beams in the treatment
        """
        return self.beams
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert electron therapy to dictionary.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary representation
        """
        # Handle potential None values
        if self.energy:
            energy_value = self.energy
        else:
            energy_value = None
        
        if self.machine:
            machine_name = self.machine.name
        else:
            machine_name = None
        
        if self.fractionation:
            fractionation_dict = self.fractionation.to_dict()
        else:
            fractionation_dict = None
        
        return {
            "id": self.technique_id,
            "name": self.name,
            "type": "ELECTRON",
            "category": self.category.value,
            "technique": self.technique.value,
            "energy": energy_value,
            "applicator_size_cm": self.applicator_size_cm,
            "target_depth_mm": self.target_depth_mm,
            "prescription_isodose": self.prescription_isodose,
            "bolus_thickness_mm": self.bolus_thickness_mm,
            "bolus_material": self.bolus_material,
            "cutout_shape": self.cutout_shape,
            "cutout_dimensions": self.cutout_dimensions,
            "beams": [beam.to_dict() for beam in self.beams],
            "fractionation": fractionation_dict,
            "machine": machine_name
        }
    
    def generate_standard_beam(self) -> Beam:
        """
        Generate a standard electron beam based on current parameters.
        
        Returns
        -------
        Beam
            Configured electron beam
        
        Raises
        ------
        ValueError
            If required parameters are not set
        """
        if self.energy is None:
            raise ValueError("Energy must be set before generating standard beam")
        
        if self.applicator_size_cm is None:
            raise ValueError("Applicator size must be set before generating standard beam")
        
        # Create a beam name based on parameters
        beam_name = f"E-{self.energy}MeV-{self.applicator_size_cm}cm"
        if self.cutout_shape:
            beam_name += f"-{self.cutout_shape}"
        
        # Create the beam
        beam = Beam(beam_name=beam_name)
        beam.beam_type = BeamType.ELECTRON
        
        # Set beam physical parameters
        beam.set_energy(self.energy)
        
        # Set beam geometry
        if hasattr(beam.geometry, 'gantry_angle'):
            beam.geometry.gantry_angle = self.gantry_angle
        
        if hasattr(beam.geometry, 'collimator_angle'):
            beam.geometry.collimator_angle = self.collimator_angle
            
        if hasattr(beam.geometry, 'source_surface_distance'):
            beam.geometry.source_surface_distance = self.ssd_cm * 10.0  # Convert to mm
        
        # Add bolus if specified
        if self.bolus_thickness_mm is not None and self.bolus_thickness_mm > 0:
            # Convert from mm to cm
            thickness_cm = self.bolus_thickness_mm / 10.0
            # Create bolus with name and thickness
            bolus = Bolus(name=f"Bolus-{beam_name}", thickness=thickness_cm)
            beam.modifiers.append(bolus)
        
        # Set treatment machine if available
        if self.machine:
            beam.set_treatment_machine(self.machine)
        
        logger.info(
            f"Generated standard electron beam '{beam_name}' with energy {self.energy} MeV and applicator size {self.applicator_size_cm} cm"
        )
        
        return beam
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ElectronTherapy':
        """
        Create an ElectronTherapy instance from a dictionary.
        
        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary containing technique data
            
        Returns
        -------
        ElectronTherapy
            Initialized technique instance
        """
        # Extract energy from data if present
        energy_value = data.get('energy')
        energy = energy_value if energy_value else None
        
        # Create base therapy
        therapy = cls(
            name=data.get('name', 'Electron Therapy'),
            technique_id=data.get('technique_id'),
            technique=ElectronApplicationTechnique(data.get('technique', ElectronApplicationTechnique.STANDARD.value)),
            energy=energy
        )
        
        # Set additional properties if present in the data
        if 'applicator_size_cm' in data:
            therapy.set_applicator(data['applicator_size_cm'])
            
        if 'cutout_shape' in data:
            therapy.set_cutout_shape(data['cutout_shape'])
            
        if 'ssd_cm' in data:
            therapy.set_ssd(data['ssd_cm'])
            
        if 'bolus_thickness_mm' in data and data['bolus_thickness_mm'] > 0:
            therapy.set_bolus(
                data['bolus_thickness_mm'],
                data.get('bolus_material', 'tissue-equivalent')
            )
            
        return therapy

class Electron(TreatmentTechnique):
    """
    Lớp đại diện cho kỹ thuật xạ trị electron.
    
    Kỹ thuật xạ trị electron sử dụng chùm electron để điều trị các 
    khối u nông, gần bề mặt cơ thể.
    """
    
    def __init__(self, technique_name: str = "Electron"):
        """
        Khởi tạo kỹ thuật xạ trị electron.
        
        Parameters
        ----------
        technique_name : str, optional
            Tên của kỹ thuật, mặc định là "Electron"
        """
        super().__init__(technique_name)
        self.energy_range = (4, 20)  # MeV, mặc định
        self.current_energy = None   # MeV
        self.applicator_size = None  # cm x cm
        self.custom_cutout = False
        self.cutout_shape = None
        self.bolus_thickness = 0.0   # cm
        self.ssd = 100.0             # Source-to-Surface Distance (cm)