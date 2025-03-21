#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module for FLASH Radiotherapy techniques.

This module provides classes for configuring and managing FLASH Radiotherapy,
a novel treatment approach that uses ultra-high dose rates to reduce side effects
on normal tissues while maintaining tumor control.
"""

import logging
from enum import Enum
from typing import Dict, Optional, Any

from quangtps.treatment.beams.beam import Beam
from quangtps.treatment.fractionation import Fractionation
from quangtps.treatment.machine.treatment_machine import TreatmentMachine
from quangtps.treatment.techniques.technique_interface import BaseTreatmentTechnique, TechniqueCategory

logger = logging.getLogger(__name__)

class FLASHMode(str, Enum):
    """Enum representing different FLASH treatment modes."""
    ELECTRON = "ELECTRON"  # FLASH treatment with electrons
    PHOTON = "PHOTON"      # FLASH treatment with photons
    PROTON = "PROTON"      # FLASH treatment with protons

class FLASHTherapy(BaseTreatmentTechnique):
    """
    Class for FLASH Radiotherapy technique.
    
    FLASH Radiotherapy is characterized by ultra-high dose rates (>40 Gy/s),
    which have been shown to spare normal tissues while maintaining tumor control,
    potentially revolutionizing radiation therapy.
    """
    
    def __init__(self, 
                 name: str = "Default FLASH",
                 technique_id: Optional[str] = None,
                 mode: FLASHMode = FLASHMode.ELECTRON):
        """
        Initialize a FLASH Radiotherapy treatment.
        
        Parameters
        ----------
        name : str, optional
            Name of the FLASH treatment
        technique_id : str, optional
            Unique ID for the FLASH treatment
        mode : FLASHMode, optional
            Treatment mode for FLASH delivery
        """
        super().__init__(
            name=name,
            technique_id=technique_id,
            category=TechniqueCategory.ADVANCED
        )
        
        self.mode = mode
        self.dose_rate = 40.0  # Gy/s, minimum >40 Gy/s to achieve FLASH effect
        self.pulse_duration = 0.1  # s
        self.time_between_pulses = 0.001  # s
        self.total_dose = 0.0  # Total dose (Gy)
        self.metadata = {}
        
        logger.info("Created new FLASH Radiotherapy plan: %s (ID: %s, Mode: %s)", 
                   name, self.technique_id, mode.value)
    
    def get_name(self) -> str:
        """
        Get the name of the treatment technique.
        
        Returns
        -------
        str
            Treatment technique name
        """
        return self.name
    
    def get_id(self) -> str:
        """
        Get the ID of the treatment technique.
        
        Returns
        -------
        str
            Treatment technique ID
        """
        return self.technique_id
    
    def get_category(self) -> TechniqueCategory:
        """
        Get the category of the treatment technique.
        
        Returns
        -------
        TechniqueCategory
            Treatment technique category
        """
        return self.category
    
    def add_beam(self, beam: Beam) -> None:
        """
        Add a beam to the FLASH treatment plan.
        
        Parameters
        ----------
        beam : Beam
            Beam to add to the plan
        """
        if beam not in self.beams:
            self.beams.append(beam)
            logger.info("Added beam to FLASH plan: %s", beam.name)
        else:
            logger.warning("Beam %s already exists in FLASH plan", beam.name)
    
    def configure_fractionation(self, fractionation: Fractionation) -> None:
        """
        Configure the fractionation scheme for the FLASH treatment.
        
        Parameters
        ----------
        fractionation : Fractionation
            Fractionation scheme to use
        """
        self.fractionation = fractionation
        self.total_dose = fractionation.total_dose
        logger.info("Configured fractionation for FLASH plan: %s", 
                   fractionation.name)
    
    def set_machine(self, machine: TreatmentMachine) -> None:
        """
        Set the treatment machine for FLASH delivery.
        
        Parameters
        ----------
        machine : TreatmentMachine
            Treatment machine to use for FLASH delivery
        """
        # Check if machine supports FLASH
        if not hasattr(machine, 'supports_flash') or not machine.supports_flash:
            logger.warning("Machine %s may not support FLASH dose rates", 
                          machine.name)
            
        self.machine = machine
        logger.info("Set treatment machine for FLASH plan: %s", 
                   machine.name)
    
    def set_dose_rate(self, dose_rate: float) -> None:
        """
        Set the dose rate for FLASH delivery.
        
        Parameters
        ----------
        dose_rate : float
            Dose rate in Gy/s
        """
        if dose_rate < 40.0:
            logger.warning("Dose rate of %.2f Gy/s may be below FLASH threshold (40 Gy/s)", 
                          dose_rate)
        
        self.dose_rate = dose_rate
        logger.info("Set FLASH dose rate to %.2f Gy/s", dose_rate)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the FLASH treatment to a dictionary.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary representation of the FLASH treatment
        """
        return {
            'id': self.technique_id,
            'name': self.name,
            'technique_type': 'FLASH',
            'category': self.category.value,
            'mode': self.mode.value,
            'dose_rate': self.dose_rate,
            'pulse_duration': self.pulse_duration,
            'time_between_pulses': self.time_between_pulses,
            'total_dose': self.total_dose,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FLASHTherapy':
        """
        Create a FLASH treatment from a dictionary.
        
        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary containing FLASH treatment data
            
        Returns
        -------
        FLASHTherapy
            FLASH treatment created from the dictionary
        """
        if data.get('technique_type') != 'FLASH':
            raise ValueError("Dictionary does not contain valid FLASH data")
            
        flash = cls(
            name=data.get('name', 'Default FLASH'),
            technique_id=data.get('id'),
            mode=FLASHMode(data.get('mode', 'ELECTRON'))
        )
        
        flash.dose_rate = data.get('dose_rate', 40.0)
        flash.pulse_duration = data.get('pulse_duration', 0.1)
        flash.time_between_pulses = data.get('time_between_pulses', 0.001)
        flash.total_dose = data.get('total_dose', 0.0)
        flash.metadata = data.get('metadata', {})
        
        return flash

# Add alias for backwards compatibility
FLASHRadiotherapy = FLASHTherapy

# Ensure class is exported correctly
__all__ = ['FLASHMode', 'FLASHTherapy']