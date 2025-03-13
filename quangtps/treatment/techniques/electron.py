#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module for Electron Therapy techniques.

This module provides classes for configuring and managing Electron Therapy, 
which uses electron beams for treating superficial tumors.
"""

import logging
from typing import Dict, Any, List, Optional
from enum import Enum

from quangtps.treatment.beams.beam import Beam
from quangtps.treatment.machine.linac import Linac
from quangtps.treatment.fractionation import Fractionation

logger = logging.getLogger(__name__)

class ElectronApplicationTechnique(str, Enum):
    """Enum for different electron therapy application techniques."""
    STANDARD = "STANDARD"  # Standard electron applicator
    CUSTOM_CUTOUT = "CUSTOM_CUTOUT"  # Standard applicator with custom cutout
    SKIN_COLLIMATION = "SKIN_COLLIMATION"  # Additional collimation directly on skin
    TOTAL_SKIN = "TOTAL_SKIN"  # Total skin electron therapy
    INTRAOPERATIVE = "INTRAOPERATIVE"  # Intraoperative electron therapy (IOERT)
    BOLUS = "BOLUS"  # With bolus material to modify surface dose

class ElectronEnergy(str, Enum):
    """Enum for standard clinical electron energies in MeV."""
    E4 = "4"
    E6 = "6"
    E9 = "9"
    E12 = "12"
    E15 = "15"
    E18 = "18"
    E20 = "20"
    E22 = "22"
    
class ElectronTherapy:
    """
    Class for Electron Therapy technique.
    
    Electron Therapy uses electron beams for treating superficial tumors up to 
    approximately 5cm deep, depending on the energy used. It provides a high
    surface dose and rapid dose fall-off beyond the target depth.
    """
    
    def __init__(self, 
                 name: str, 
                 technique: ElectronApplicationTechnique = ElectronApplicationTechnique.STANDARD,
                 energy: ElectronEnergy = ElectronEnergy.E9):
        """
        Initialize an Electron Therapy treatment.
        
        Parameters
        ----------
        name : str
            Name of the electron treatment
        technique : ElectronApplicationTechnique
            Application technique
        energy : ElectronEnergy
            Electron energy in MeV
        """
        self.name = name
        self.technique = technique
        self.energy = energy
        
        # Treatment parameters
        self.beams: List[Beam] = []
        self.fractionation: Optional[Fractionation] = None
        self.machine: Optional[Linac] = None
        self.total_dose: Optional[float] = None
        self.target_depth_mm: float = 0.0  # Target depth in mm
        self.prescription_isodose: float = 90.0  # Default prescription isodose (%)
        
        # Electron-specific parameters
        self.applicator_size_cm: Optional[int] = None  # Applicator size in cm
        self.ssd_cm: float = 100.0  # Source-to-surface distance in cm
        self.use_bolus: bool = False
        self.bolus_thickness_mm: float = 0.0
        self.bolus_material: str = "TISSUE_EQUIVALENT"
        self.cutout_shape: Optional[str] = None
        self.cutout_dimensions: Optional[Dict[str, float]] = None
        
    def set_energy(self, energy: ElectronEnergy):
        """
        Set the electron beam energy.
        
        Parameters
        ----------
        energy : ElectronEnergy
            Electron energy in MeV
        """
        self.energy = energy
        logger.info(f"Set electron energy to {energy} MeV for treatment {self.name}")
        
    def set_applicator(self, size_cm: int):
        """
        Set the electron applicator size.
        
        Parameters
        ----------
        size_cm : int
            Applicator size in cm (typically 6, 10, 14, 20, or 25)
        """
        self.applicator_size_cm = size_cm
        logger.info(f"Set applicator size to {size_cm} cm for treatment {self.name}")
        
    def set_custom_cutout(self, shape: str, dimensions: Dict[str, float]):
        """
        Set a custom cutout for the electron beam.
        
        Parameters
        ----------
        shape : str
            Shape of the cutout (e.g., "RECTANGULAR", "CIRCULAR", "IRREGULAR")
        dimensions : Dict[str, float]
            Dimensions of the cutout
        """
        self.technique = ElectronApplicationTechnique.CUSTOM_CUTOUT
        self.cutout_shape = shape
        self.cutout_dimensions = dimensions
        logger.info(f"Set custom {shape} cutout for treatment {self.name}")
        
    def set_bolus(self, thickness_mm: float, material: str = "TISSUE_EQUIVALENT"):
        """
        Set bolus parameters.
        
        Parameters
        ----------
        thickness_mm : float
            Bolus thickness in mm
        material : str
            Bolus material type
        """
        self.use_bolus = True
        self.bolus_thickness_mm = thickness_mm
        self.bolus_material = material
        self.technique = ElectronApplicationTechnique.BOLUS
        logger.info(f"Set {thickness_mm} mm {material} bolus for treatment {self.name}")
        
    def set_ssd(self, ssd_cm: float):
        """
        Set the source-to-surface distance.
        
        Parameters
        ----------
        ssd_cm : float
            SSD in cm
        """
        self.ssd_cm = ssd_cm
        logger.info(f"Set SSD to {ssd_cm} cm for treatment {self.name}")
        
    def set_target_depth(self, depth_mm: float):
        """
        Set the target depth.
        
        Parameters
        ----------
        depth_mm : float
            Target depth in mm
        """
        self.target_depth_mm = depth_mm
        
        # Recommend appropriate energy based on target depth
        if depth_mm < 15:
            recommended_energy = ElectronEnergy.E6
        elif depth_mm < 30:
            recommended_energy = ElectronEnergy.E9
        elif depth_mm < 40:
            recommended_energy = ElectronEnergy.E12
        elif depth_mm < 50:
            recommended_energy = ElectronEnergy.E15
        elif depth_mm < 60:
            recommended_energy = ElectronEnergy.E18
        else:
            recommended_energy = ElectronEnergy.E20
            
        if self.energy != recommended_energy:
            logger.info(
                f"Based on target depth of {depth_mm} mm, recommended energy is "
                f"{recommended_energy} MeV (current: {self.energy} MeV)"
            )
            
    def set_prescription_isodose(self, isodose_percent: float):
        """
        Set the prescription isodose.
        
        Parameters
        ----------
        isodose_percent : float
            Prescription isodose in percentage (typically 80-90%)
        """
        self.prescription_isodose = isodose_percent
        logger.info(f"Set prescription to {isodose_percent}% isodose for treatment {self.name}")
        
    def add_beam(self, beam: Beam):
        """
        Add a beam to the electron treatment.
        
        Parameters
        ----------
        beam : Beam
            Electron beam
        """
        self.beams.append(beam)
        logger.info(f"Added beam {beam.name} to electron treatment {self.name}")
        
    def set_machine(self, machine: Linac):
        """
        Set the treatment machine.
        
        Parameters
        ----------
        machine : Linac
            Treatment machine (linear accelerator)
        """
        # Verify machine has electron capabilities
        if not machine.has_electron_mode:
            logger.error(f"Machine {machine.name} does not support electron therapy")
            raise ValueError(f"Machine {machine.name} does not support electron therapy")
            
        self.machine = machine
        logger.info(f"Set machine to {machine.name} for electron treatment {self.name}")
        
    def set_fractionation(self, fractionation: Fractionation):
        """
        Set the fractionation scheme.
        
        Parameters
        ----------
        fractionation : Fractionation
            Fractionation scheme
        """
        self.fractionation = fractionation
        self.total_dose = fractionation.total_dose
        logger.info(
            f"Set fractionation to {fractionation.total_dose}Gy in "
            f"{fractionation.number_of_fractions} fractions for treatment {self.name}"
        )
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert electron treatment to dictionary.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary representation
        """
        return {
            "name": self.name,
            "technique": self.technique,
            "energy": self.energy,
            "total_dose": self.total_dose,
            "target_depth_mm": self.target_depth_mm,
            "prescription_isodose": self.prescription_isodose,
            "beams": [beam.name for beam in self.beams],
            "machine": self.machine.name if self.machine else None,
            "fractionation": self.fractionation.to_dict() if self.fractionation else None,
            "applicator_size_cm": self.applicator_size_cm,
            "ssd_cm": self.ssd_cm,
            "use_bolus": self.use_bolus,
            "bolus_thickness_mm": self.bolus_thickness_mm,
            "bolus_material": self.bolus_material,
            "cutout_shape": self.cutout_shape,
            "cutout_dimensions": self.cutout_dimensions
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ElectronTherapy':
        """
        Create electron treatment from dictionary.
        
        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary with treatment data
            
        Returns
        -------
        ElectronTherapy
            Electron therapy instance
        """
        treatment = cls(
            name=data["name"],
            technique=data["technique"],
            energy=data["energy"]
        )
        
        treatment.total_dose = data["total_dose"]
        treatment.target_depth_mm = data["target_depth_mm"]
        treatment.prescription_isodose = data["prescription_isodose"]
        treatment.applicator_size_cm = data["applicator_size_cm"]
        treatment.ssd_cm = data["ssd_cm"]
        treatment.use_bolus = data["use_bolus"]
        treatment.bolus_thickness_mm = data["bolus_thickness_mm"]
        treatment.bolus_material = data["bolus_material"]
        treatment.cutout_shape = data["cutout_shape"]
        treatment.cutout_dimensions = data["cutout_dimensions"]
        
        return treatment