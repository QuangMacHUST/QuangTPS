"""
Beams Module

Contains the Beam class and related functionality.
"""

from typing import Dict, List, Optional, Any, Tuple, Union
import numpy as np
from enum import Enum


class BeamType(Enum):
    """Enumeration of beam types."""
    STATIC = "STATIC"  # Static beam
    ARC = "ARC"  # Arc beam
    CONFORMAL_ARC = "CONFORMAL_ARC"  # Conformal arc
    IMRT = "IMRT"  # Intensity-modulated radiation therapy
    VMAT = "VMAT"  # Volumetric modulated arc therapy
    ELECTRON = "ELECTRON"  # Electron beam
    PROTON = "PROTON"  # Proton beam


class MachineType(Enum):
    """Enumeration of treatment machine types."""
    LINAC = "LINAC"  # Linear accelerator
    CYBERKNIFE = "CYBERKNIFE"  # CyberKnife
    TOMOTHERAPY = "TOMOTHERAPY"  # TomoTherapy
    GAMMA_KNIFE = "GAMMA_KNIFE"  # Gamma Knife
    PROTON = "PROTON"  # Proton therapy system


class Beam:
    """
    Class representing a treatment beam.
    
    Attributes:
        id: Unique identifier for the beam
        name: Name of the beam
        type: Type of beam (static, arc, etc.)
        machine: Treatment machine
        energy: Beam energy
        gantry_angle: Gantry angle in degrees
        collimator_angle: Collimator angle in degrees
        couch_angle: Couch angle in degrees
        isocenter: Isocenter coordinates (x, y, z)
        weight: Beam weight in MU or relative weight
        field_size: Field size (width, height)
    """
    
    def __init__(self, id: str, name: str, type: Union[BeamType, str] = BeamType.STATIC,
                machine: Union[MachineType, str] = MachineType.LINAC,
                energy: str = "6X"):
        """
        Initialize a beam.
        
        Args:
            id: Unique identifier for the beam
            name: Name of the beam
            type: Type of beam
            machine: Treatment machine
            energy: Beam energy (e.g., "6X", "10X", "6FFF")
        """
        self.id = id
        self.name = name
        
        # Ensure type is an enum
        if isinstance(type, str):
            try:
                self.type = BeamType(type)
            except ValueError:
                self.type = BeamType.STATIC
        else:
            self.type = type
        
        # Ensure machine is an enum
        if isinstance(machine, str):
            try:
                self.machine = MachineType(machine)
            except ValueError:
                self.machine = MachineType.LINAC
        else:
            self.machine = machine
        
        self.energy = energy
        
        # Beam geometry
        self.gantry_angle = 0.0
        self.collimator_angle = 0.0
        self.couch_angle = 0.0
        self.isocenter = (0.0, 0.0, 0.0)
        self.field_size = (10.0, 10.0)  # Width, height in cm
        
        # For arc beams
        self.arc_start_angle = 0.0
        self.arc_stop_angle = 0.0
        self.arc_direction = "CW"  # Clockwise or counterclockwise
        
        # Beam weight
        self.weight = 100.0  # Monitor units (MU) or relative weight
        
        # For IMRT and VMAT beams
        self.control_points = []
    
    def set_angles(self, gantry: float, collimator: float, couch: float):
        """
        Set the beam angles.
        
        Args:
            gantry: Gantry angle in degrees
            collimator: Collimator angle in degrees
            couch: Couch angle in degrees
        """
        self.gantry_angle = gantry
        self.collimator_angle = collimator
        self.couch_angle = couch
    
    def set_isocenter(self, x: float, y: float, z: float):
        """
        Set the isocenter coordinates.
        
        Args:
            x: X coordinate in mm
            y: Y coordinate in mm
            z: Z coordinate in mm
        """
        self.isocenter = (x, y, z)
    
    def set_field_size(self, width: float, height: float):
        """
        Set the field size.
        
        Args:
            width: Field width in cm
            height: Field height in cm
        """
        self.field_size = (width, height)
    
    def set_arc_params(self, start_angle: float, stop_angle: float, direction: str = "CW"):
        """
        Set arc parameters.
        
        Args:
            start_angle: Arc start angle in degrees
            stop_angle: Arc stop angle in degrees
            direction: Arc direction ("CW" or "CCW")
        """
        if self.type not in [BeamType.ARC, BeamType.CONFORMAL_ARC, BeamType.VMAT]:
            raise ValueError("Arc parameters can only be set for arc-type beams")
        
        self.arc_start_angle = start_angle
        self.arc_stop_angle = stop_angle
        self.arc_direction = direction
    
    def set_weight(self, weight: float):
        """
        Set the beam weight.
        
        Args:
            weight: Beam weight in MU or relative weight
        """
        self.weight = weight
    
    def __str__(self) -> str:
        """
        Get a string representation of the beam.
        
        Returns:
            String representation
        """
        if self.type in [BeamType.ARC, BeamType.CONFORMAL_ARC, BeamType.VMAT]:
            return f"{self.name}: {self.type.value} ({self.energy}) from {self.arc_start_angle}° to {self.arc_stop_angle}° ({self.arc_direction})"
        else:
            return f"{self.name}: {self.type.value} ({self.energy}) at gantry {self.gantry_angle}°" 