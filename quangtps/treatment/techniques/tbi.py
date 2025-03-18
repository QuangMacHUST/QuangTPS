#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module for Total Body Irradiation (TBI) and Total Skin Irradiation (TSI) techniques.

This module provides classes for configuring and managing TBI and TSI
treatment techniques used in radiotherapy.
"""

import logging
from typing import Dict, Any, List, Optional
from enum import Enum

from quangtps.treatment.beams.beam import Beam
from quangtps.treatment.machine.linac import Linac
from quangtps.treatment.machine.treatment_machine import TreatmentMachine
from quangtps.treatment.fractionation import Fractionation
from quangtps.treatment.techniques.technique_interface import (
    BaseTreatmentTechnique, TechniqueCategory
)

logger = logging.getLogger(__name__)

class TBITechnique(str, Enum):
    """Enum for TBI delivery techniques."""
    STATIONARY = "STATIONARY"  # Patient remains stationary
    TRANSLATIONAL = "TRANSLATIONAL"  # Patient moves laterally
    ROTATIONAL = "ROTATIONAL"  # Patient rotates

class TSITechnique(str, Enum):
    """Enum for TSI delivery techniques."""
    STANFORD = "STANFORD"  # Stanford technique
    YALE = "YALE"  # Yale technique
    ROTARY = "ROTARY"  # Rotary technique
    TRANSLATIONAL = "TRANSLATIONAL"  # Translational technique

class TBI(BaseTreatmentTechnique):
    """
    Class for Total Body Irradiation (TBI) technique.
    
    TBI involves delivery of radiation to the entire body, primarily
    used for conditioning regimens before bone marrow transplantation.
    """
    
    def __init__(self, 
                 name: str, 
                 technique: TBITechnique = TBITechnique.STATIONARY,
                 tbi_id: Optional[str] = None):
        """
        Initialize a TBI treatment.
        
        Parameters
        ----------
        name : str
            Name of the TBI treatment
        technique : TBITechnique
            Delivery technique
        tbi_id : str, optional
            Unique ID for the TBI treatment
        """
        super().__init__(
            name=name, 
            technique_id=tbi_id, 
            category=TechniqueCategory.SPECIAL
        )
        self.technique = technique
        
        # TBI-specific attributes
        self.extended_ssd = True  # Use extended source-to-surface distance
        self.beam_spoiler = True  # Use beam spoiler
        self.compensators = True  # Use compensators for dose homogeneity
        self.lung_blocks = False  # Use lung blocks
        
        # Treatment parameters
        self.total_dose = 12.0  # Gy
        self.dose_rate = 0.1  # Gy/min (typically low dose rate)
        self.ssd = 400.0  # Source-to-surface distance (cm)
        
        logger.info("Initialized TBI treatment: %s (ID: %s)", name, self.technique_id)
    
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
    
    def set_technique(self, technique: TBITechnique) -> None:
        """
        Set TBI delivery technique.
        
        Parameters
        ----------
        technique : TBITechnique
            Delivery technique
        """
        self.technique = technique
        logger.info("Set TBI technique to %s for plan '%s'", technique.value, self.name)
    
    def set_treatment_parameters(self, total_dose: float, dose_rate: float, ssd: float) -> None:
        """
        Set treatment parameters.
        
        Parameters
        ----------
        total_dose : float
            Total dose in Gy
        dose_rate : float
            Dose rate in Gy/min
        ssd : float
            Source-to-surface distance in cm
        """
        self.total_dose = total_dose
        self.dose_rate = dose_rate
        self.ssd = ssd
        
        # Update fractionation based on total dose
        if self.fractionation:
            self.fractionation.total_dose = total_dose
        
        logger.info("Set TBI parameters: %f Gy at %f Gy/min, SSD=%f cm for '%s'", total_dose, dose_rate, ssd, self.name)
    
    def set_beam_modifiers(self, 
                          extended_ssd: bool, 
                          beam_spoiler: bool, 
                          compensators: bool, 
                          lung_blocks: bool) -> None:
        """
        Set beam modifiers.
        
        Parameters
        ----------
        extended_ssd : bool
            Use extended source-to-surface distance
        beam_spoiler : bool
            Use beam spoiler
        compensators : bool
            Use compensators for dose homogeneity
        lung_blocks : bool
            Use lung blocks
        """
        self.extended_ssd = extended_ssd
        self.beam_spoiler = beam_spoiler
        self.compensators = compensators
        self.lung_blocks = lung_blocks
        
        logger.info(
            "Set beam modifiers for TBI '%s': extended SSD=%s, beam spoiler=%s, compensators=%s, lung blocks=%s", 
            self.name, extended_ssd, beam_spoiler, compensators, lung_blocks
        )
    
    def set_fractionation(self, fractionation: Fractionation) -> None:
        """
        Set the fractionation for the TBI treatment.
        
        Parameters
        ----------
        fractionation : Fractionation
            The fractionation scheme
        """
        self.fractionation = fractionation
        
        # Update total dose to match fractionation
        self.total_dose = fractionation.total_dose
        
        logger.info("Set fractionation to %f Gy in %d fractions for TBI '%s'", fractionation.total_dose, fractionation.num_fractions, self.name)
    
    def set_machine(self, machine: TreatmentMachine) -> None:
        """
        Set the treatment machine for the TBI treatment.
        
        Parameters
        ----------
        machine : TreatmentMachine
            The treatment machine to use
        """
        if not isinstance(machine, Linac):
            raise ValueError("TBI requires a Linac treatment machine")
        
        self.machine = machine
        logger.info("Set treatment machine to '%s' for TBI '%s'", machine.name, self.name)
    
    def add_beam(self, beam: Beam) -> None:
        """
        Add a beam to the TBI plan.
        
        Parameters
        ----------
        beam : Beam
            The beam to add to the plan
        """
        self.beams.append(beam)
        logger.info("Added beam '%s' to TBI plan '%s'", beam.beam_name, self.name)
    
    def get_beams(self) -> List[Beam]:
        """
        Get all beams in the TBI plan.
        
        Returns
        -------
        List[Beam]
            List of beams in the plan
        """
        return self.beams
    
    def generate_standard_beams(self) -> List[Beam]:
        """
        Generate standard beams for the selected technique.
        
        Returns
        -------
        List[Beam]
            List of generated beams
        """
        beams = []
        
        if not self.machine:
            logger.warning("No treatment machine set, cannot generate beams")
            return beams
        
        if self.technique == TBITechnique.STATIONARY:
            # AP/PA setup
            ap_beam = Beam(beam_name=f"{self.name}_AP")
            ap_beam.set_energy(6)  # 6 MV typically used
            ap_beam.geometry.gantry_angle = 0  # AP
            ap_beam.geometry.field_size = (40, 40)  # Large field
            ap_beam.geometry.ssd = self.ssd
            beams.append(ap_beam)
            
            pa_beam = Beam(beam_name=f"{self.name}_PA")
            pa_beam.set_energy(6)
            pa_beam.geometry.gantry_angle = 180  # PA
            pa_beam.geometry.field_size = (40, 40)
            pa_beam.geometry.ssd = self.ssd
            beams.append(pa_beam)
            
        elif self.technique == TBITechnique.TRANSLATIONAL:
            # Lateral fields with patient translation
            right_beam = Beam(beam_name=f"{self.name}_Right")
            right_beam.set_energy(6)
            right_beam.geometry.gantry_angle = 270  # Right lateral
            right_beam.geometry.field_size = (40, 20)  # Tall, narrow field
            right_beam.geometry.ssd = self.ssd
            beams.append(right_beam)
            
            left_beam = Beam(beam_name=f"{self.name}_Left")
            left_beam.set_energy(6)
            left_beam.geometry.gantry_angle = 90  # Left lateral
            left_beam.geometry.field_size = (40, 20)
            left_beam.geometry.ssd = self.ssd
            beams.append(left_beam)
            
        elif self.technique == TBITechnique.ROTATIONAL:
            # Single arc beam
            arc_beam = Beam(beam_name=f"{self.name}_Arc")
            arc_beam.set_energy(6)
            arc_beam.geometry.gantry_angle = 0  # Start angle
            arc_beam.geometry.gantry_stop_angle = 360  # End angle
            arc_beam.geometry.field_size = (40, 40)
            arc_beam.geometry.ssd = self.ssd
            beams.append(arc_beam)
        
        # Add beams to the plan
        for beam in beams:
            self.add_beam(beam)
            
        return beams
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert TBI to dictionary.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary representation
        """
        return {
            "id": self.technique_id,
            "name": self.name,
            "type": "TBI",
            "category": self.category.value,
            "technique": self.technique.value,
            "extended_ssd": self.extended_ssd,
            "beam_spoiler": self.beam_spoiler,
            "compensators": self.compensators,
            "lung_blocks": self.lung_blocks,
            "total_dose": self.total_dose,
            "dose_rate": self.dose_rate,
            "ssd": self.ssd,
            "beams": [beam.to_dict() for beam in self.beams],
            "fractionation": self.fractionation.to_dict() if self.fractionation else None,
            "machine": self.machine.name if self.machine else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TBI':
        """
        Create TBI from dictionary.
        
        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary with TBI data
            
        Returns
        -------
        TBI
            TBI instance
        """
        # Create the technique enum from string
        technique_value = data.get("technique", TBITechnique.STATIONARY.value)
        technique = TBITechnique(technique_value)
        
        # Create TBI instance
        tbi = cls(
            name=data["name"],
            technique=technique,
            tbi_id=data["id"]
        )
        
        # Set beam modifiers
        tbi.set_beam_modifiers(
            extended_ssd=data.get("extended_ssd", True),
            beam_spoiler=data.get("beam_spoiler", True),
            compensators=data.get("compensators", True),
            lung_blocks=data.get("lung_blocks", False)
        )
        
        # Set treatment parameters
        tbi.set_treatment_parameters(
            total_dose=data.get("total_dose", 12.0),
            dose_rate=data.get("dose_rate", 0.1),
            ssd=data.get("ssd", 400.0)
        )
        
        # Load beams
        from quangtps.treatment.beams.beam import Beam
        for beam_data in data.get("beams", []):
            beam = Beam.from_dict(beam_data)
            tbi.beams.append(beam)
        
        # Load fractionation
        if "fractionation" in data and data["fractionation"]:
            from quangtps.treatment.fractionation import Fractionation
            tbi.fractionation = Fractionation.from_dict(data["fractionation"])
        
        return tbi


class TSI(BaseTreatmentTechnique):
    """
    Class for Total Skin Irradiation (TSI) technique.
    
    TSI involves delivery of radiation to the entire skin surface,
    primarily used for treating cutaneous lymphomas.
    """
    
    def __init__(self, 
                 name: str, 
                 technique: TSITechnique = TSITechnique.STANFORD,
                 tsi_id: Optional[str] = None):
        """
        Initialize a TSI treatment.
        
        Parameters
        ----------
        name : str
            Name of the TSI treatment
        technique : TSITechnique
            Delivery technique
        tsi_id : str, optional
            Unique ID for the TSI treatment
        """
        super().__init__(
            name=name, 
            technique_id=tsi_id, 
            category=TechniqueCategory.SPECIAL
        )
        self.technique = technique
        
        # TSI-specific attributes
        self.beam_positions = 6  # Typically six dual fields
        self.use_degrader = True  # Beam energy degrader
        self.use_screen = True  # Scatter screen
        
        # Treatment parameters
        self.total_dose = 36.0  # Gy
        self.dose_rate = 4.0  # Gy/min
        self.ssd = 300.0  # Source-to-surface distance (cm)
        self.gantry_angles = [0, 60, 120, 180, 240, 300]  # Standard angles
        
        logger.info("Initialized TSI treatment: %s (ID: %s)", name, self.technique_id)
    
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
        
    def set_technique(self, technique: TSITechnique) -> None:
        """
        Set TSI delivery technique.
        
        Parameters
        ----------
        technique : TSITechnique
            Delivery technique
        """
        self.technique = technique
        logger.info("Set TSI technique to %s for plan '%s'", technique.value, self.name)
        
    def set_treatment_parameters(self, total_dose: float, dose_rate: float, ssd: float) -> None:
        """
        Set treatment parameters.
        
        Parameters
        ----------
        total_dose : float
            Total dose in Gy
        dose_rate : float
            Dose rate in Gy/min
        ssd : float
            Source-to-surface distance in cm
        """
        self.total_dose = total_dose
        self.dose_rate = dose_rate
        self.ssd = ssd
        
        # Update fractionation based on total dose
        if self.fractionation:
            self.fractionation.total_dose = total_dose
            
        logger.info("Set TSI parameters: %f Gy at %f Gy/min, SSD=%f cm for '%s'", total_dose, dose_rate, ssd, self.name)
    
    def set_gantry_angles(self, angles: List[float]) -> None:
        """
        Set gantry angles for TSI.
        
        Parameters
        ----------
        angles : List[float]
            List of gantry angles in degrees
        """
        self.gantry_angles = angles
        logger.info("Set gantry angles for TSI '%s': %s", self.name, angles)
    
    def set_beam_modifiers(self, positions: int, use_degrader: bool, use_screen: bool) -> None:
        """
        Set beam modifiers for TSI.
        
        Parameters
        ----------
        positions : int
            Number of beam positions
        use_degrader : bool
            Whether to use beam energy degrader
        use_screen : bool
            Whether to use scatter screen
        """
        self.beam_positions = positions
        self.use_degrader = use_degrader
        self.use_screen = use_screen
        
        logger.info("Set beam modifiers for TSI '%s': positions=%d, degrader=%s, screen=%s", 
                   self.name, positions, use_degrader, use_screen)
    
    def set_fractionation(self, fractionation: Fractionation) -> None:
        """
        Set the fractionation for the TSI treatment.
        
        Parameters
        ----------
        fractionation : Fractionation
            The fractionation scheme
        """
        self.fractionation = fractionation
        
        # Update total dose to match fractionation
        self.total_dose = fractionation.total_dose
        
        logger.info("Set fractionation to %f Gy in %d fractions for TSI '%s'", fractionation.total_dose, fractionation.num_fractions, self.name)
    
    def set_machine(self, machine: TreatmentMachine) -> None:
        """
        Set the treatment machine for the TSI treatment.
        
        Parameters
        ----------
        machine : TreatmentMachine
            The treatment machine to use
        """
        if not isinstance(machine, Linac):
            raise ValueError("TSI requires a Linac treatment machine")
        
        self.machine = machine
        logger.info("Set treatment machine to '%s' for TSI '%s'", machine.name, self.name)
    
    def add_beam(self, beam: Beam) -> None:
        """
        Add a beam to the TSI plan.
        
        Parameters
        ----------
        beam : Beam
            The beam to add to the plan
        """
        self.beams.append(beam)
        logger.info("Added beam '%s' to TSI plan '%s'", beam.beam_name, self.name)
    
    def get_beams(self) -> List[Beam]:
        """
        Get all beams in the TSI plan.
        
        Returns
        -------
        List[Beam]
            List of beams in the plan
        """
        return self.beams
    
    def generate_standard_beams(self) -> List[Beam]:
        """
        Generate standard beams for the selected technique.
        
        Returns
        -------
        List[Beam]
            List of generated beams
        """
        beams = []
        
        if not self.machine:
            logger.warning("No treatment machine set, cannot generate beams")
            return beams
        
        if self.technique == TSITechnique.STANFORD:
            # Stanford technique: 6 dual field technique (12 beams)
            for i, angle in enumerate(self.gantry_angles):
                # Anterior oblique beam
                beam1 = Beam(beam_name=f"{self.name}_{i+1}a")
                beam1.set_energy(6)  # Typically 6 MeV electrons
                beam1.geometry.gantry_angle = angle
                beam1.geometry.field_size = (40, 40)
                beam1.geometry.ssd = self.ssd
                beams.append(beam1)
                
                # Posterior oblique beam
                beam2 = Beam(beam_name=f"{self.name}_{i+1}b")
                beam2.set_energy(6)
                beam2.geometry.gantry_angle = (angle + 180) % 360
                beam2.geometry.field_size = (40, 40)
                beam2.geometry.ssd = self.ssd
                beams.append(beam2)
                
        elif self.technique == TSITechnique.YALE:
            # Yale technique: Similar to Stanford but with specific setup
            for i, angle in enumerate([0, 60, 120, 180, 240, 300]):
                beam = Beam(beam_name=f"{self.name}_Yale_{i+1}")
                beam.set_energy(6)
                beam.geometry.gantry_angle = angle
                beam.geometry.field_size = (36, 36)
                beam.geometry.ssd = self.ssd
                beams.append(beam)
                
        elif self.technique == TSITechnique.ROTARY:
            # Rotary platform technique
            beam = Beam(beam_name=f"{self.name}_Rotary")
            beam.set_energy(6)
            beam.geometry.gantry_angle = 90  # Horizontal beam
            beam.geometry.field_size = (40, 40)
            beam.geometry.ssd = self.ssd
            beams.append(beam)
            
        elif self.technique == TSITechnique.TRANSLATIONAL:
            # Translational technique
            beam = Beam(beam_name=f"{self.name}_Translational")
            beam.set_energy(6)
            beam.geometry.gantry_angle = 90  # Horizontal beam
            beam.geometry.field_size = (40, 10)  # Narrow field
            beam.geometry.ssd = self.ssd
            beams.append(beam)
            
        # Add beams to the plan
        for beam in beams:
            self.add_beam(beam)
            
        return beams
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert TSI to dictionary.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary representation
        """
        return {
            "id": self.technique_id,
            "name": self.name,
            "type": "TSI",
            "category": self.category.value,
            "technique": self.technique.value,
            "beam_positions": self.beam_positions,
            "use_degrader": self.use_degrader,
            "use_screen": self.use_screen,
            "total_dose": self.total_dose,
            "dose_rate": self.dose_rate,
            "ssd": self.ssd,
            "gantry_angles": self.gantry_angles,
            "beams": [beam.to_dict() for beam in self.beams],
            "fractionation": self.fractionation.to_dict() if self.fractionation else None,
            "machine": self.machine.name if self.machine else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TSI':
        """
        Create TSI from dictionary.
        
        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary with TSI data
            
        Returns
        -------
        TSI
            TSI instance
        """
        # Create the technique enum from string
        technique_value = data.get("technique", TSITechnique.STANFORD.value)
        technique = TSITechnique(technique_value)
        
        # Create TSI instance
        tsi = cls(
            name=data["name"],
            technique=technique,
            tsi_id=data["id"]
        )
        
        # Set beam modifiers
        tsi.set_beam_modifiers(
            positions=data.get("beam_positions", 6),
            use_degrader=data.get("use_degrader", True),
            use_screen=data.get("use_screen", True)
        )
        
        # Set treatment parameters
        tsi.set_treatment_parameters(
            total_dose=data.get("total_dose", 36.0),
            dose_rate=data.get("dose_rate", 4.0),
            ssd=data.get("ssd", 300.0)
        )
        
        # Set gantry angles
        if "gantry_angles" in data:
            tsi.set_gantry_angles(data["gantry_angles"])
        
        # Load beams
        from quangtps.treatment.beams.beam import Beam
        for beam_data in data.get("beams", []):
            beam = Beam.from_dict(beam_data)
            tsi.beams.append(beam)
        
        # Load fractionation
        if "fractionation" in data and data["fractionation"]:
            from quangtps.treatment.fractionation import Fractionation
            tsi.fractionation = Fractionation.from_dict(data["fractionation"])
        
        return tsi

# Ensure proper exports
__all__ = ['TBI', 'TSI', 'TBITechnique', 'TSITechnique']