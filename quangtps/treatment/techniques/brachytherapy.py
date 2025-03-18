#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module for Brachytherapy techniques.

This module provides classes for configuring and managing Brachytherapy treatments,
which involve placing radioactive sources in or near the target tissue.
"""

import logging
from typing import Dict, Any, List, Optional, Union, Tuple
from enum import Enum

from quangtps.treatment.beams.beam import Beam
from quangtps.treatment.machine.treatment_machine import TreatmentMachine
from quangtps.treatment.fractionation import Fractionation
from quangtps.treatment.techniques.technique_interface import BaseTreatmentTechnique, TechniqueCategory

logger = logging.getLogger(__name__)

class BrachytherapyType(str, Enum):
    """Enum for different brachytherapy types."""
    LDR = "LDR"  # Low Dose Rate
    HDR = "HDR"  # High Dose Rate
    PDR = "PDR"  # Pulsed Dose Rate

class ApplicationMethod(str, Enum):
    """Enum for different application methods."""
    INTERSTITIAL = "INTERSTITIAL"  # Radioactive sources inserted into tissue
    INTRACAVITARY = "INTRACAVITARY"  # Sources placed in body cavities
    SURFACE = "SURFACE"  # Sources placed on the surface
    INTRALUMINAL = "INTRALUMINAL"  # Sources placed inside tubular structures

class RadioactiveIsotope(str, Enum):
    """Enum for common radioactive isotopes used in brachytherapy."""
    IR_192 = "IR_192"  # Iridium-192
    CS_137 = "CS_137"  # Cesium-137
    I_125 = "I_125"  # Iodine-125
    PD_103 = "PD_103"  # Palladium-103
    CO_60 = "CO_60"  # Cobalt-60

class Brachytherapy(BaseTreatmentTechnique):
    """
    Class for Brachytherapy technique.
    
    Brachytherapy is a form of radiotherapy where radioactive sources are placed
    inside or next to the treatment area. It allows very high doses to be delivered
    to the tumor while reducing exposure to surrounding healthy tissues.
    """
    
    def __init__(self, 
                 name: str,
                 brachy_type: BrachytherapyType = BrachytherapyType.HDR,
                 application: ApplicationMethod = ApplicationMethod.INTRACAVITARY,
                 isotope: RadioactiveIsotope = RadioactiveIsotope.IR_192,
                 technique_id: Optional[str] = None):
        """
        Initialize a Brachytherapy treatment.
        
        Parameters
        ----------
        name : str
            Name of the brachytherapy treatment
        brachy_type : BrachytherapyType
            Type of brachytherapy (LDR, HDR, PDR)
        application : ApplicationMethod
            Method of source application
        isotope : RadioactiveIsotope
            Radioactive isotope used
        technique_id : str, optional
            Unique ID for the brachytherapy treatment
        """
        super().__init__(
            name=name, 
            technique_id=technique_id, 
            category=TechniqueCategory.SPECIAL
        )
        
        self.brachy_type = brachy_type
        self.application = application
        self.isotope = isotope
        
        # Brachytherapy-specific parameters
        self.dose_specification_point: Optional[Tuple[float, float, float]] = None
        self.dose_rate: Optional[float] = None  # Gy/h
        self.implant_duration: Optional[float] = None  # Hours for LDR/PDR
        self.source_activity: Optional[float] = None  # Ci or mCi
        self.source_dwell_positions: List[Dict[str, Any]] = []
        self.source_dwell_times: List[float] = []
        self.applicator_type: Optional[str] = None
        self.reference_air_kerma_rate: Optional[float] = None  # cGy·cm²/h
        
        # Quality assurance
        self.pre_treatment_imaging: Optional[str] = None
        
        logger.info(f"Initialized Brachytherapy treatment: {name} (ID: {self.technique_id})")
    
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
    
    def set_isotope(self, isotope: RadioactiveIsotope, activity: float):
        """
        Set the radioactive isotope and its activity.
        
        Parameters
        ----------
        isotope : RadioactiveIsotope
            Radioactive isotope
        activity : float
            Source activity (in Ci or mCi)
        """
        self.isotope = isotope
        self.source_activity = activity
        logger.info(f"Set isotope to {isotope} with activity {activity} for treatment {self.name}")
        
        # Update dose rate based on isotope and activity
        self._calculate_dose_rate()
    
    def _calculate_dose_rate(self):
        """
        Calculate the dose rate based on isotope and activity.
        This is a simplified calculation and should be replaced with
        proper dosimetric models in a production environment.
        """
        # Simplified dose rate calculation - these would be replaced with proper models
        if self.isotope == RadioactiveIsotope.IR_192:
            self.dose_rate = self.source_activity * 4.0  # Example conversion
        elif self.isotope == RadioactiveIsotope.CS_137:
            self.dose_rate = self.source_activity * 3.0
        elif self.isotope == RadioactiveIsotope.I_125:
            self.dose_rate = self.source_activity * 1.5
        elif self.isotope == RadioactiveIsotope.PD_103:
            self.dose_rate = self.source_activity * 1.0
        elif self.isotope == RadioactiveIsotope.CO_60:
            self.dose_rate = self.source_activity * 5.0
    
    def set_application_method(self, method: ApplicationMethod, applicator_type: Optional[str] = None):
        """
        Set the application method and applicator type.
        
        Parameters
        ----------
        method : ApplicationMethod
            Method of source application
        applicator_type : str, optional
            Type of applicator used
        """
        self.application = method
        self.applicator_type = applicator_type
        logger.info(f"Set application method to {method} with applicator {applicator_type} "
                   f"for treatment {self.name}")
    
    def set_dose_specification_point(self, point: Tuple[float, float, float]):
        """
        Set the dose specification point in 3D space.
        
        Parameters
        ----------
        point : Tuple[float, float, float]
            3D coordinates of the dose specification point
        """
        self.dose_specification_point = point
        logger.info(f"Set dose specification point to {point} for treatment {self.name}")
    
    def add_dwell_position(self, 
                           position: Tuple[float, float, float], 
                           time: float, 
                           channel: Optional[int] = None):
        """
        Add a source dwell position with its dwell time.
        
        Parameters
        ----------
        position : Tuple[float, float, float]
            3D coordinates of the dwell position
        time : float
            Dwell time in seconds
        channel : int, optional
            Channel number for multiple-channel treatments
        """
        dwell_data = {
            "position": position,
            "time": time,
            "channel": channel
        }
        self.source_dwell_positions.append(dwell_data)
        self.source_dwell_times.append(time)
        logger.info(f"Added dwell position at {position} with time {time}s "
                   f"to channel {channel} for treatment {self.name}")
    
    def set_implant_duration(self, duration: float):
        """
        Set the implant duration for LDR or PDR treatments.
        
        Parameters
        ----------
        duration : float
            Duration in hours
        """
        if self.brachy_type == BrachytherapyType.HDR:
            logger.warning(f"Setting implant duration for HDR treatment {self.name} - "
                          f"this is not typically needed for HDR")
        
        self.implant_duration = duration
        logger.info(f"Set implant duration to {duration} hours for treatment {self.name}")
    
    def calculate_total_dose(self) -> float:
        """
        Calculate the total dose based on type, dose rate, and time.
        
        Returns
        -------
        float
            Total dose in Gy
        """
        if self.brachy_type == BrachytherapyType.HDR:
            # HDR dose is usually prescribed directly
            if self.fractionation:
                return self.fractionation.total_dose
            return 0.0
        
        elif self.brachy_type in [BrachytherapyType.LDR, BrachytherapyType.PDR]:
            # LDR/PDR dose depends on dose rate and implant duration
            if self.dose_rate is not None and self.implant_duration is not None:
                return self.dose_rate * self.implant_duration
            return 0.0
    
    def set_fractionation(self, fractionation: Fractionation) -> None:
        """
        Set the fractionation for the brachytherapy treatment.
        
        Parameters
        ----------
        fractionation : Fractionation
            The fractionation scheme
        """
        self.fractionation = fractionation
        logger.info(f"Set fractionation to {fractionation.total_dose} Gy in {fractionation.num_fractions} "
                   f"fractions for brachytherapy '{self.name}'")
    
    def set_machine(self, machine: TreatmentMachine) -> None:
        """
        Set the treatment machine for the brachytherapy treatment.
        In brachytherapy, this typically refers to the afterloader.
        
        Parameters
        ----------
        machine : TreatmentMachine
            The treatment machine to use
        """
        self.machine = machine
        logger.info(f"Set treatment machine to {machine.name} for brachytherapy '{self.name}'")
    
    def add_beam(self, beam: Beam) -> None:
        """
        Add a beam to the brachytherapy plan. In brachytherapy, this is typically
        used for modeling the radioactive source.
        
        Parameters
        ----------
        beam : Beam
            The beam to add to the plan
        """
        if beam not in self.beams:
            self.beams.append(beam)
            logger.info(f"Added beam {beam.beam_id} to brachytherapy '{self.name}'")
    
    def get_beams(self) -> List[Beam]:
        """
        Get all beams in the brachytherapy plan.
        
        Returns
        -------
        List[Beam]
            List of beams in the plan
        """
        return self.beams
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert Brachytherapy to dictionary.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary representation
        """
        return {
            "id": self.technique_id,
            "name": self.name,
            "category": self.category.value,
            "brachy_type": self.brachy_type,
            "application": self.application,
            "isotope": self.isotope,
            "dose_specification_point": self.dose_specification_point,
            "dose_rate": self.dose_rate,
            "implant_duration": self.implant_duration,
            "source_activity": self.source_activity,
            "source_dwell_positions": self.source_dwell_positions,
            "source_dwell_times": self.source_dwell_times,
            "applicator_type": self.applicator_type,
            "reference_air_kerma_rate": self.reference_air_kerma_rate,
            "pre_treatment_imaging": self.pre_treatment_imaging
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Brachytherapy':
        """
        Create Brachytherapy from dictionary.
        
        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary with Brachytherapy data
            
        Returns
        -------
        Brachytherapy
            Brachytherapy instance
        """
        brachy = cls(
            name=data["name"],
            brachy_type=BrachytherapyType(data["brachy_type"]),
            application=ApplicationMethod(data["application"]),
            isotope=RadioactiveIsotope(data["isotope"]),
            technique_id=data["id"]
        )
        
        # Set brachytherapy-specific parameters
        brachy.dose_specification_point = data.get("dose_specification_point")
        brachy.dose_rate = data.get("dose_rate")
        brachy.implant_duration = data.get("implant_duration")
        brachy.source_activity = data.get("source_activity")
        brachy.source_dwell_positions = data.get("source_dwell_positions", [])
        brachy.source_dwell_times = data.get("source_dwell_times", [])
        brachy.applicator_type = data.get("applicator_type")
        brachy.reference_air_kerma_rate = data.get("reference_air_kerma_rate")
        brachy.pre_treatment_imaging = data.get("pre_treatment_imaging")
        
        return brachy


# Ensure proper exports
__all__ = ['Brachytherapy', 'BrachytherapyType', 'ApplicationMethod', 'RadioactiveIsotope']
