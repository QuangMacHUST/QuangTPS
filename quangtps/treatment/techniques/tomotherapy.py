#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module for Tomotherapy treatment techniques.

This module provides classes for configuring and managing Tomotherapy treatments,
which is a specialized form of intensity-modulated radiation therapy (IMRT) that combines
CT imaging with a radiation therapy system.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum

from quangtps.treatment.beams.beam import Beam
from quangtps.treatment.machine.treatment_machine import TreatmentMachine
from quangtps.treatment.fractionation import Fractionation
from quangtps.treatment.techniques.technique_interface import BaseTreatmentTechnique, TechniqueCategory

logger = logging.getLogger(__name__)

class TomoplanType(str, Enum):
    """Enum for different types of Tomotherapy plans."""
    HELICAL = "HELICAL"  # Helical Tomotherapy - radiation delivered in a spiral pattern
    DIRECT = "DIRECT"    # Direct/TomoDirect - radiation delivered from discrete angles
    
class PitchSetting(str, Enum):
    """Enum for pitch settings in Tomotherapy."""
    FINE = "FINE"           # Fine pitch (~0.25-0.3)
    STANDARD = "STANDARD"   # Standard pitch (~0.3-0.4)
    COARSE = "COARSE"       # Coarse pitch (~0.4-0.5)
    
class FieldWidth(str, Enum):
    """Enum for field width settings in Tomotherapy."""
    NARROW = "NARROW"       # 1.0 cm
    MEDIUM = "MEDIUM"       # 2.5 cm
    WIDE = "WIDE"           # 5.0 cm

class Tomotherapy(BaseTreatmentTechnique):
    """
    Class for Tomotherapy treatment technique.
    
    Tomotherapy is an advanced form of radiation therapy that combines IMRT with
    image-guided radiation therapy (IGRT). It delivers radiation in a helical pattern
    or from discrete angles (TomoDirect) while the patient moves through the machine.
    """
    
    def __init__(self, 
                 name: str,
                 plan_type: TomoplanType = TomoplanType.HELICAL,
                 pitch: PitchSetting = PitchSetting.STANDARD,
                 field_width: FieldWidth = FieldWidth.MEDIUM,
                 technique_id: Optional[str] = None):
        """
        Initialize a Tomotherapy treatment.
        
        Parameters
        ----------
        name : str
            Name of the tomotherapy treatment
        plan_type : TomoplanType
            Type of tomotherapy plan (helical or direct)
        pitch : PitchSetting
            Pitch setting for helical delivery
        field_width : FieldWidth
            Field width setting
        technique_id : str, optional
            Unique ID for the tomotherapy treatment
        """
        super().__init__(
            name=name, 
            technique_id=technique_id, 
            category=TechniqueCategory.ADVANCED
        )
        
        self.plan_type = plan_type
        self.pitch_setting = pitch
        self.field_width_setting = field_width
        
        # Numerical values based on settings
        self.pitch_value = self._get_pitch_value(pitch)
        self.field_width_value = self._get_field_width_value(field_width)
        
        # Tomotherapy-specific parameters
        self.modulation_factor = 2.0
        self.gantry_period = 20.0  # seconds per rotation
        self.mvct_scan = True      # Use MVCT scan for positioning
        self.image_registration_method = "AUTOMATIC_THEN_MANUAL"
        self.angles = []           # For TomoDirect
        
        # Dose calculation parameters
        self.dose_grid_resolution = (0.3, 0.3, 0.3)  # cm
        self.calculation_parameters = {
            "optimization_type": "BEAMLET",  # Alternative: FULL_SCATTER
            "convergence_mode": "STANDARD",  # Alternative: HIGH_ACCURACY
            "iterations": 150
        }
        
        logger.info(f"Initialized Tomotherapy treatment: {name} (ID: {self.technique_id})")
    
    def _get_pitch_value(self, pitch_setting: PitchSetting) -> float:
        """
        Convert pitch setting to numerical value.
        
        Parameters
        ----------
        pitch_setting : PitchSetting
            Pitch setting
            
        Returns
        -------
        float
            Numerical pitch value
        """
        if pitch_setting == PitchSetting.FINE:
            return 0.287
        elif pitch_setting == PitchSetting.STANDARD:
            return 0.430
        elif pitch_setting == PitchSetting.COARSE:
            return 0.502
        return 0.430  # Default standard pitch
    
    def _get_field_width_value(self, field_width: FieldWidth) -> float:
        """
        Convert field width setting to numerical value.
        
        Parameters
        ----------
        field_width : FieldWidth
            Field width setting
            
        Returns
        -------
        float
            Numerical field width value in cm
        """
        if field_width == FieldWidth.NARROW:
            return 1.0
        elif field_width == FieldWidth.MEDIUM:
            return 2.5
        elif field_width == FieldWidth.WIDE:
            return 5.0
        return 2.5  # Default medium field width
    
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
    
    def set_plan_type(self, plan_type: TomoplanType):
        """
        Set the tomotherapy plan type.
        
        Parameters
        ----------
        plan_type : TomoplanType
            Type of tomotherapy plan
        """
        self.plan_type = plan_type
        logger.info(f"Set plan type to {plan_type} for treatment {self.name}")
        
        # Clear angles if switching to helical
        if plan_type == TomoplanType.HELICAL:
            self.angles = []
    
    def set_delivery_parameters(self, 
                               pitch: Optional[PitchSetting] = None, 
                               field_width: Optional[FieldWidth] = None,
                               modulation_factor: Optional[float] = None):
        """
        Set delivery parameters for tomotherapy.
        
        Parameters
        ----------
        pitch : PitchSetting, optional
            Pitch setting
        field_width : FieldWidth, optional
            Field width setting
        modulation_factor : float, optional
            Modulation factor
        """
        if pitch is not None:
            self.pitch_setting = pitch
            self.pitch_value = self._get_pitch_value(pitch)
            
        if field_width is not None:
            self.field_width_setting = field_width
            self.field_width_value = self._get_field_width_value(field_width)
            
        if modulation_factor is not None:
            self.modulation_factor = modulation_factor
            
        logger.info(f"Set delivery parameters for treatment {self.name}: "
                   f"pitch={self.pitch_value}, field width={self.field_width_value} cm, "
                   f"modulation factor={self.modulation_factor}")
    
    def set_direct_angles(self, angles: List[float]):
        """
        Set gantry angles for TomoDirect delivery.
        
        Parameters
        ----------
        angles : List[float]
            List of gantry angles in degrees
        """
        if self.plan_type != TomoplanType.DIRECT:
            self.plan_type = TomoplanType.DIRECT
            logger.info(f"Changed plan type to DIRECT for treatment {self.name}")
            
        self.angles = angles
        logger.info(f"Set TomoDirect angles to {angles} for treatment {self.name}")
    
    def set_mvct_parameters(self, use_mvct: bool, registration_method: str):
        """
        Set MVCT imaging parameters.
        
        Parameters
        ----------
        use_mvct : bool
            Whether to use MVCT for positioning
        registration_method : str
            Image registration method
        """
        self.mvct_scan = use_mvct
        self.image_registration_method = registration_method
        logger.info(f"Set MVCT parameters for treatment {self.name}: "
                   f"use MVCT={use_mvct}, registration={registration_method}")
    
    def set_calculation_parameters(self, 
                                  optimization_type: str, 
                                  convergence_mode: str, 
                                  iterations: int,
                                  dose_grid_resolution: Optional[Tuple[float, float, float]] = None):
        """
        Set dose calculation parameters.
        
        Parameters
        ----------
        optimization_type : str
            Type of optimization algorithm
        convergence_mode : str
            Convergence mode
        iterations : int
            Number of iterations
        dose_grid_resolution : Tuple[float, float, float], optional
            Resolution of dose grid in cm
        """
        self.calculation_parameters = {
            "optimization_type": optimization_type,
            "convergence_mode": convergence_mode,
            "iterations": iterations
        }
        
        if dose_grid_resolution is not None:
            self.dose_grid_resolution = dose_grid_resolution
            
        logger.info(f"Set calculation parameters for treatment {self.name}: "
                   f"optimization={optimization_type}, "
                   f"convergence={convergence_mode}, "
                   f"iterations={iterations}, "
                   f"grid={self.dose_grid_resolution}")
    
    def set_fractionation(self, fractionation: Fractionation) -> None:
        """
        Set the fractionation for the tomotherapy treatment.
        
        Parameters
        ----------
        fractionation : Fractionation
            The fractionation scheme
        """
        self.fractionation = fractionation
        logger.info(f"Set fractionation to {fractionation.total_dose} Gy in {fractionation.num_fractions} "
                   f"fractions for tomotherapy '{self.name}'")
    
    def set_machine(self, machine: TreatmentMachine) -> None:
        """
        Set the treatment machine for the tomotherapy treatment.
        This would typically be a specialized tomotherapy machine.
        
        Parameters
        ----------
        machine : TreatmentMachine
            The treatment machine to use
        """
        self.machine = machine
        logger.info(f"Set treatment machine to {machine.name} for tomotherapy '{self.name}'")
        
        # Verify machine compatibility
        if not hasattr(machine, 'is_tomotherapy') or not machine.is_tomotherapy:
            logger.warning(f"Machine {machine.name} is not a tomotherapy machine, which may cause issues")
    
    def add_beam(self, beam: Beam) -> None:
        """
        Add a beam to the tomotherapy plan. In Tomotherapy, this represents
        a projection or beamlet rather than a conventional beam.
        
        Parameters
        ----------
        beam : Beam
            The beam to add to the plan
        """
        if beam not in self.beams:
            self.beams.append(beam)
            logger.info(f"Added beam/projection {beam.beam_id} to tomotherapy '{self.name}'")
    
    def get_beams(self) -> List[Beam]:
        """
        Get all beams in the tomotherapy plan.
        
        Returns
        -------
        List[Beam]
            List of beams in the plan
        """
        return self.beams
    
    def estimate_treatment_time(self) -> float:
        """
        Estimate the treatment delivery time.
        
        Returns
        -------
        float
            Estimated treatment time in minutes
        """
        if self.plan_type == TomoplanType.HELICAL:
            # Estimate based on pitch, field width, and gantry period
            if self.fractionation is None or self.fractionation.dose_per_fraction <= 0:
                return 0.0
                
            # Rough estimation for a typical treatment length
            treatment_length_cm = 20.0  # cm
            
            # Number of rotations needed
            rotations = treatment_length_cm / (self.field_width_value * self.pitch_value)
            
            # Time per rotation
            time_per_rotation = self.gantry_period  # seconds
            
            # Total time in minutes plus setup time
            return (rotations * time_per_rotation / 60.0) + 5.0  # minutes
        
        elif self.plan_type == TomoplanType.DIRECT:
            # For TomoDirect, estimate based on number of angles
            if not self.angles:
                return 0.0
                
            # Rough estimation - 2 minutes per angle plus setup time
            return (len(self.angles) * 2.0) + 5.0  # minutes
            
        return 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert Tomotherapy to dictionary.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary representation
        """
        return {
            "id": self.technique_id,
            "name": self.name,
            "category": self.category.value,
            "plan_type": self.plan_type,
            "pitch_setting": self.pitch_setting,
            "pitch_value": self.pitch_value,
            "field_width_setting": self.field_width_setting,
            "field_width_value": self.field_width_value,
            "modulation_factor": self.modulation_factor,
            "gantry_period": self.gantry_period,
            "mvct_scan": self.mvct_scan,
            "image_registration_method": self.image_registration_method,
            "angles": self.angles,
            "dose_grid_resolution": self.dose_grid_resolution,
            "calculation_parameters": self.calculation_parameters
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Tomotherapy':
        """
        Create Tomotherapy from dictionary.
        
        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary with Tomotherapy data
            
        Returns
        -------
        Tomotherapy
            Tomotherapy instance
        """
        tomo = cls(
            name=data["name"],
            plan_type=TomoplanType(data["plan_type"]),
            pitch=PitchSetting(data["pitch_setting"]),
            field_width=FieldWidth(data["field_width_setting"]),
            technique_id=data["id"]
        )
        
        # Set tomotherapy-specific parameters
        tomo.modulation_factor = data.get("modulation_factor", 2.0)
        tomo.gantry_period = data.get("gantry_period", 20.0)
        tomo.mvct_scan = data.get("mvct_scan", True)
        tomo.image_registration_method = data.get("image_registration_method", "AUTOMATIC_THEN_MANUAL")
        tomo.angles = data.get("angles", [])
        tomo.dose_grid_resolution = data.get("dose_grid_resolution", (0.3, 0.3, 0.3))
        tomo.calculation_parameters = data.get("calculation_parameters", {
            "optimization_type": "BEAMLET",
            "convergence_mode": "STANDARD",
            "iterations": 150
        })
        
        return tomo


# Ensure proper exports
__all__ = ['Tomotherapy', 'TomoplanType', 'PitchSetting', 'FieldWidth']
