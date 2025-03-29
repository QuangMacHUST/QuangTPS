#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module defining FLASH radiotherapy techniques.

FLASH radiotherapy uses ultra-high dose rates (>40 Gy/s) to deliver the prescribed
dose in an extremely short time, potentially reducing normal tissue toxicity
while maintaining tumor control.
"""

from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import logging

from quangtps.treatment.techniques.treatment_technique import TreatmentTechnique

logger = logging.getLogger(__name__)


class FLASH(TreatmentTechnique):
    """
    Class representing FLASH radiotherapy technique.
    
    FLASH radiotherapy delivers very high dose rates (>40 Gy/s) in short pulses,
    exploiting the FLASH effect which appears to spare normal tissues while
    maintaining tumor control.
    """
    
    def __init__(self, technique_name: str = "FLASH"):
        """
        Initialize FLASH radiotherapy technique.
        
        Parameters
        ----------
        technique_name : str, optional
            Name of the technique, default is "FLASH"
        """
        super().__init__(technique_name)
        self.dose_rate = None  # Gy/s
        self.pulse_duration = None  # ms
        self.is_electron = True  # Default is electron FLASH
        self.is_proton = False
        self.is_photon = False
        self.minimum_dose_rate = 40.0  # Gy/s, minimum dose rate for FLASH effect
        self.delivery_mode = None  # Pulsed or continuous
        self.energy = None  # MeV
        self.field_size = None  # cm²
        self.pulse_repetition_rate = None  # Hz
        self.total_delivery_time = None  # s
    
    def set_beam_type(self, electron: bool = False, proton: bool = False, photon: bool = False):
        """
        Set the beam type for FLASH delivery.
        
        Parameters
        ----------
        electron : bool, optional
            Whether to use electron FLASH, default is False
        proton : bool, optional
            Whether to use proton FLASH, default is False
        photon : bool, optional
            Whether to use photon FLASH, default is False
        """
        if not any([electron, proton, photon]):
            logger.warning("At least one beam type must be selected")
            return
        
        self.is_electron = electron
        self.is_proton = proton
        self.is_photon = photon
        
        beam_types = []
        if electron:
            beam_types.append("electron")
        if proton:
            beam_types.append("proton")
        if photon:
            beam_types.append("photon")
        
        logger.info(f"Set FLASH beam type to: {', '.join(beam_types)}")
    
    def set_dose_rate(self, dose_rate: float):
        """
        Set the dose rate for FLASH delivery.
        
        Parameters
        ----------
        dose_rate : float
            Dose rate (Gy/s)
        """
        if dose_rate < self.minimum_dose_rate:
            logger.warning(f"Dose rate {dose_rate} Gy/s is below minimum recommended for FLASH effect ({self.minimum_dose_rate} Gy/s)")
            
        self.dose_rate = dose_rate
        logger.info(f"Set FLASH dose rate to {dose_rate} Gy/s")
    
    def set_pulse_parameters(self, pulse_duration: float, repetition_rate: float):
        """
        Set pulse parameters for pulsed FLASH delivery.
        
        Parameters
        ----------
        pulse_duration : float
            Duration of each pulse (ms)
        repetition_rate : float
            Pulse repetition rate (Hz)
        """
        if pulse_duration <= 0 or repetition_rate <= 0:
            logger.warning("Pulse parameters must be positive")
            return
        
        self.pulse_duration = pulse_duration
        self.pulse_repetition_rate = repetition_rate
        self.delivery_mode = "Pulsed"
        
        logger.info(f"Set FLASH pulse parameters: duration={pulse_duration} ms, repetition rate={repetition_rate} Hz")
    
    def set_continuous_delivery(self):
        """Set FLASH delivery mode to continuous."""
        self.delivery_mode = "Continuous"
        self.pulse_duration = None
        self.pulse_repetition_rate = None
        logger.info("Set FLASH delivery mode to continuous")
    
    def set_energy(self, energy: float):
        """
        Set beam energy for FLASH.
        
        Parameters
        ----------
        energy : float
            Beam energy (MeV)
        """
        if energy <= 0:
            logger.warning(f"Energy must be positive, got {energy} MeV")
            return
        
        self.energy = energy
        logger.info(f"Set FLASH beam energy to {energy} MeV")
    
    def set_field_size(self, field_size: float):
        """
        Set field size for FLASH.
        
        Parameters
        ----------
        field_size : float
            Field size (cm²)
        """
        if field_size <= 0:
            logger.warning(f"Field size must be positive, got {field_size} cm²")
            return
        
        self.field_size = field_size
        logger.info(f"Set FLASH field size to {field_size} cm²")
    
    def estimate_delivery_time(self, prescribed_dose: float):
        """
        Estimate total delivery time based on prescribed dose and dose rate.
        
        Parameters
        ----------
        prescribed_dose : float
            Prescribed dose (Gy)
        
        Returns
        -------
        float or None
            Estimated delivery time (s) or None if parameters are missing
        """
        if self.dose_rate is None or prescribed_dose <= 0:
            logger.warning("Cannot estimate delivery time: missing dose rate or invalid prescribed dose")
            return None
        
        self.total_delivery_time = prescribed_dose / self.dose_rate
        logger.info(f"Estimated FLASH delivery time: {self.total_delivery_time:.6f} s for {prescribed_dose} Gy")
        return self.total_delivery_time
    
    def is_valid_flash(self) -> bool:
        """
        Check if current parameters constitute valid FLASH therapy.
        
        Returns
        -------
        bool
            True if parameters meet FLASH criteria, False otherwise
        """
        if self.dose_rate is None or self.dose_rate < self.minimum_dose_rate:
            logger.warning(f"Dose rate must be at least {self.minimum_dose_rate} Gy/s for FLASH effect")
            return False
        
        if not any([self.is_electron, self.is_proton, self.is_photon]):
            logger.warning("No beam type selected")
            return False
        
        if self.energy is None:
            logger.warning("Energy not specified")
            return False
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert FLASH radiotherapy information to a dictionary.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary containing technique information
        """
        data = super().to_dict()
        data.update({
            "dose_rate": self.dose_rate,
            "pulse_duration": self.pulse_duration,
            "is_electron": self.is_electron,
            "is_proton": self.is_proton,
            "is_photon": self.is_photon,
            "minimum_dose_rate": self.minimum_dose_rate,
            "delivery_mode": self.delivery_mode,
            "energy": self.energy,
            "field_size": self.field_size,
            "pulse_repetition_rate": self.pulse_repetition_rate,
            "total_delivery_time": self.total_delivery_time
        })
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FLASH':
        """
        Create a FLASH object from a dictionary.
        
        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary containing technique information
            
        Returns
        -------
        FLASH
            FLASH object
        """
        technique = cls(data.get("technique_name", "FLASH"))
        
        # Set beam type
        technique.set_beam_type(
            electron=data.get("is_electron", False),
            proton=data.get("is_proton", False),
            photon=data.get("is_photon", False)
        )
        
        # Set other parameters
        if "dose_rate" in data:
            technique.set_dose_rate(data["dose_rate"])
            
        if "energy" in data:
            technique.set_energy(data["energy"])
            
        if "field_size" in data:
            technique.set_field_size(data["field_size"])
            
        # Set delivery mode
        if data.get("delivery_mode") == "Pulsed" and "pulse_duration" in data and "pulse_repetition_rate" in data:
            technique.set_pulse_parameters(data["pulse_duration"], data["pulse_repetition_rate"])
        else:
            technique.set_continuous_delivery()
            
        # Set delivery time if available
        technique.total_delivery_time = data.get("total_delivery_time")
        
        return technique


# Alias for backward compatibility
class FLASHRadiotherapy(FLASH):
    """
    Class representing FLASH radiotherapy technique (alias for FLASH).
    
    This class is provided for backward compatibility with existing code that 
    may reference FLASHRadiotherapy instead of FLASH.
    """
    
    def __init__(self, technique_name: str = "FLASH Radiotherapy"):
        """
        Initialize FLASH radiotherapy technique.
        
        Parameters
        ----------
        technique_name : str, optional
            Name of the technique, default is "FLASH Radiotherapy"
        """
        super().__init__(technique_name)
        logger.info("FLASHRadiotherapy initialized (alias for FLASH)")

# Add FLASHTherapy as an alias for FLASH
class FLASHTherapy(FLASH):
    """
    Alias for FLASH class for backward compatibility.
    """
    def __init__(self, technique_name: str = "FLASH Therapy"):
        """
        Initialize FLASHTherapy, which is an alias for FLASH.
        
        Parameters
        ----------
        technique_name : str, optional
            Name of the technique, default is "FLASH Therapy"
        """
        super().__init__(technique_name)
        logger.info("FLASHTherapy initialized (alias for FLASH)")