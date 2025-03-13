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
from quangtps.treatment.fractionation import Fractionation

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

class TBI:
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
        self.name = name
        self.tbi_id = tbi_id or f"tbi_{name.lower().replace(' ', '_')}"
        self.technique = technique
        
        # TBI-specific attributes
        self.beams: List[Beam] = []
        self.machine: Optional[Linac] = None
        self.fractionation: Optional[Fractionation] = None
        self.extended_ssd = True  # Extended SSD is typically used in TBI
        self.lung_blocks = False  # Whether lung blocks are used
        self.spoilers = True  # Acrylic spoilers for dose buildup
        self.compensators = False  # Tissue compensators
        
        # Treatment parameters
        self.total_dose = 12.0  # Gy
        self.dose_rate = 0.1  # Gy/min, typically low
        self.ssd = 300.0  # Source-to-surface distance (cm)
        
    def set_fractionation(self, fractionation: Fractionation):
        """
        Set fractionation scheme for TBI.
        
        Parameters
        ----------
        fractionation : Fractionation
            Fractionation scheme
        """
        self.fractionation = fractionation
        self.total_dose = fractionation.total_dose
        
    def set_machine(self, machine: Linac):
        """
        Set treatment machine.
        
        Parameters
        ----------
        machine : Linac
            Linear accelerator for treatment
        """
        self.machine = machine
        
    def add_beam(self, beam: Beam):
        """
        Add a beam to the TBI plan.
        
        Parameters
        ----------
        beam : Beam
            Beam to add
        """
        self.beams.append(beam)
        
    def set_lung_blocks(self, use_blocks: bool):
        """
        Set whether to use lung blocks.
        
        Parameters
        ----------
        use_blocks : bool
            Whether to use lung blocks
        """
        self.lung_blocks = use_blocks
        
    def set_compensators(self, use_compensators: bool):
        """
        Set whether to use tissue compensators.
        
        Parameters
        ----------
        use_compensators : bool
            Whether to use tissue compensators
        """
        self.compensators = use_compensators
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert TBI to dictionary.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary representation
        """
        return {
            "name": self.name,
            "tbi_id": self.tbi_id,
            "technique": self.technique,
            "total_dose": self.total_dose,
            "dose_rate": self.dose_rate,
            "ssd": self.ssd,
            "extended_ssd": self.extended_ssd,
            "lung_blocks": self.lung_blocks,
            "spoilers": self.spoilers,
            "compensators": self.compensators,
            "fractionation": self.fractionation.to_dict() if self.fractionation else None,
            "machine": self.machine.machine_id if self.machine else None,
            "beams": [beam.to_dict() for beam in self.beams]
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
        tbi = cls(
            name=data["name"],
            technique=data["technique"],
            tbi_id=data["tbi_id"]
        )
        
        tbi.total_dose = data["total_dose"]
        tbi.dose_rate = data["dose_rate"]
        tbi.ssd = data["ssd"]
        tbi.extended_ssd = data["extended_ssd"]
        tbi.lung_blocks = data["lung_blocks"]
        tbi.spoilers = data["spoilers"]
        tbi.compensators = data["compensators"]
        
        return tbi


class TSI:
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
        self.name = name
        self.tsi_id = tsi_id or f"tsi_{name.lower().replace(' ', '_')}"
        self.technique = technique
        
        # TSI-specific attributes
        self.beams: List[Beam] = []
        self.machine: Optional[Linac] = None
        self.fractionation: Optional[Fractionation] = None
        self.beam_positions = 6  # Typically six dual fields
        self.use_degrader = True  # Beam energy degrader
        self.use_screen = True  # Scatter screen
        
        # Treatment parameters
        self.total_dose = 36.0  # Gy
        self.dose_rate = 4.0  # Gy/min
        self.ssd = 300.0  # Source-to-surface distance (cm)
        self.gantry_angles = [0, 60, 120, 180, 240, 300]  # Standard angles
        
    def set_fractionation(self, fractionation: Fractionation):
        """
        Set fractionation scheme for TSI.
        
        Parameters
        ----------
        fractionation : Fractionation
            Fractionation scheme
        """
        self.fractionation = fractionation
        self.total_dose = fractionation.total_dose
        
    def set_machine(self, machine: Linac):
        """
        Set treatment machine.
        
        Parameters
        ----------
        machine : Linac
            Linear accelerator for treatment
        """
        self.machine = machine
        
    def set_technique(self, technique: TSITechnique):
        """
        Set TSI delivery technique.
        
        Parameters
        ----------
        technique : TSITechnique
            Delivery technique
        """
        self.technique = technique
        
        # Update parameters based on technique
        if technique == TSITechnique.STANFORD:
            self.beam_positions = 6
            self.gantry_angles = [0, 60, 120, 180, 240, 300]
        elif technique == TSITechnique.YALE:
            self.beam_positions = 8
            self.gantry_angles = [0, 45, 90, 135, 180, 225, 270, 315]
        elif technique == TSITechnique.ROTARY:
            self.beam_positions = 1  # One rotational field
            self.gantry_angles = [0]  # Starting angle
            
    def add_beam(self, beam: Beam):
        """
        Add a beam to the TSI plan.
        
        Parameters
        ----------
        beam : Beam
            Beam to add
        """
        self.beams.append(beam)
        
    def generate_standard_beams(self):
        """
        Generate standard beams for the selected technique.
        
        Returns
        -------
        List[Beam]
            List of generated beams
        """
        if self.machine is None:
            logger.error("Machine must be set before generating beams")
            return []
            
        # Implementation would create beams based on the technique
        # This is a placeholder
        beams = []
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
            "name": self.name,
            "tsi_id": self.tsi_id,
            "technique": self.technique,
            "total_dose": self.total_dose,
            "dose_rate": self.dose_rate,
            "ssd": self.ssd,
            "beam_positions": self.beam_positions,
            "use_degrader": self.use_degrader,
            "use_screen": self.use_screen,
            "gantry_angles": self.gantry_angles,
            "fractionation": self.fractionation.to_dict() if self.fractionation else None,
            "machine": self.machine.machine_id if self.machine else None,
            "beams": [beam.to_dict() for beam in self.beams]
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
        tsi = cls(
            name=data["name"],
            technique=data["technique"],
            tsi_id=data["tsi_id"]
        )
        
        tsi.total_dose = data["total_dose"]
        tsi.dose_rate = data["dose_rate"]
        tsi.ssd = data["ssd"]
        tsi.beam_positions = data["beam_positions"]
        tsi.use_degrader = data["use_degrader"]
        tsi.use_screen = data["use_screen"]
        tsi.gantry_angles = data["gantry_angles"]
        
        return tsi