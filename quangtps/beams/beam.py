"""
Beam module for QuangTPS.

This module defines the Beam, BeamType and BeamSet classes used in radiotherapy treatment planning.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
import logging
import uuid
import copy
import numpy as np

logger = logging.getLogger(__name__)

class BeamType(Enum):
    """Beam type enumeration."""
    STATIC = "static"
    ARC = "arc"
    DYNAMIC = "dynamic"
    CONFORMAL = "conformal"
    IMRT = "imrt"
    VMAT = "vmat"
    ELECTRON = "electron"
    PROTON = "proton"

class Beam:
    """
    Beam class for representing a radiotherapy treatment beam.
    
    Attributes:
        id (str): Unique identifier for the beam
        name (str): Name of the beam
        type (BeamType): Type of beam
        energy (float): Energy in MV (for photons) or MeV (for electrons)
        gantry_angle (float): Gantry angle in degrees
        collimator_angle (float): Collimator angle in degrees
        couch_angle (float): Couch angle in degrees
        field_size (Tuple[float, float]): Field size in cm (width, height)
        isocenter (Tuple[float, float, float]): Isocenter position in mm (x, y, z)
        weight (float): Relative weight of this beam in the plan
        mlc (Optional[np.ndarray]): MLC positions (if applicable)
        modifiers (Dict[str, Any]): Beam modifiers (wedges, blocks, etc.)
    """
    
    def __init__(self, name: str = "", beam_type: BeamType = BeamType.STATIC):
        """
        Initialize a new Beam.
        
        Args:
            name: Name of the beam
            beam_type: Type of beam
        """
        self.id = f"b_{uuid.uuid4().hex[:8]}"
        self.name = name
        self.type = beam_type
        self.energy = 6.0  # Default to 6MV
        self.gantry_angle = 0.0
        self.collimator_angle = 0.0
        self.couch_angle = 0.0
        self.field_size = (10.0, 10.0)  # Default to 10x10 cm
        self.isocenter = (0.0, 0.0, 0.0)
        self.weight = 1.0
        self.mlc = None
        self.modifiers = {}
        self.sad = 1000.0  # Source-to-axis distance in mm (default: 100cm)
        self.technique = "STATIC"
        self.props = {}
        
    def set_mlc(self, mlc_positions: np.ndarray):
        """
        Set MLC positions for this beam.
        
        Args:
            mlc_positions: Array of MLC positions
        """
        self.mlc = mlc_positions
        
    def set_modifier(self, modifier_type: str, modifier_data: Any):
        """
        Add a modifier to this beam.
        
        Args:
            modifier_type: Type of modifier (e.g., "wedge", "block")
            modifier_data: Data for the modifier
        """
        self.modifiers[modifier_type] = modifier_data
        
    def create_copy(self, new_name: Optional[str] = None) -> 'Beam':
        """
        Create a copy of this beam.
        
        Args:
            new_name: Optional new name for the copied beam
            
        Returns:
            A new Beam instance with the same properties
        """
        if new_name is None:
            new_name = f"{self.name}_copy"
            
        new_beam = Beam(new_name, self.type)
        new_beam.energy = self.energy
        new_beam.gantry_angle = self.gantry_angle
        new_beam.collimator_angle = self.collimator_angle
        new_beam.couch_angle = self.couch_angle
        new_beam.field_size = self.field_size
        new_beam.isocenter = self.isocenter
        new_beam.weight = self.weight
        
        if self.mlc is not None:
            new_beam.mlc = self.mlc.copy()
            
        new_beam.modifiers = copy.deepcopy(self.modifiers)
        new_beam.sad = self.sad
        new_beam.technique = self.technique
        new_beam.props = copy.deepcopy(self.props)
        
        return new_beam
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert beam to a dictionary.
        
        Returns:
            Dictionary representation of the beam
        """
        beam_dict = {
            'id': self.id,
            'name': self.name,
            'type': self.type.value if isinstance(self.type, BeamType) else self.type,
            'energy': self.energy,
            'gantry_angle': self.gantry_angle,
            'collimator_angle': self.collimator_angle,
            'couch_angle': self.couch_angle,
            'field_size': self.field_size,
            'isocenter': self.isocenter,
            'weight': self.weight,
            'sad': self.sad,
            'technique': self.technique
        }
        
        # Add modifiers
        if self.modifiers:
            beam_dict['modifiers'] = self.modifiers
            
        # Add props
        if self.props:
            beam_dict['props'] = self.props
            
        return beam_dict
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Beam':
        """
        Create a beam from a dictionary.
        
        Args:
            data: Dictionary representation of a beam
            
        Returns:
            New Beam instance
        """
        beam_type_value = data.get('type', 'static')
        try:
            beam_type = BeamType(beam_type_value)
        except ValueError:
            logger.warning(f"Unknown beam type: {beam_type_value}, using STATIC")
            beam_type = BeamType.STATIC
            
        beam = cls(name=data.get('name', ''), beam_type=beam_type)
        beam.id = data.get('id', beam.id)
        beam.energy = data.get('energy', beam.energy)
        beam.gantry_angle = data.get('gantry_angle', beam.gantry_angle)
        beam.collimator_angle = data.get('collimator_angle', beam.collimator_angle)
        beam.couch_angle = data.get('couch_angle', beam.couch_angle)
        beam.field_size = data.get('field_size', beam.field_size)
        beam.isocenter = data.get('isocenter', beam.isocenter)
        beam.weight = data.get('weight', beam.weight)
        beam.sad = data.get('sad', beam.sad)
        beam.technique = data.get('technique', beam.technique)
        
        # Add modifiers
        if 'modifiers' in data:
            beam.modifiers = data['modifiers']
            
        # Add props
        if 'props' in data:
            beam.props = data['props']
            
        return beam
        
    def __str__(self) -> str:
        return f"Beam({self.name}, type={self.type}, energy={self.energy}, gantry={self.gantry_angle}°)"
        
    def __repr__(self) -> str:
        return self.__str__()

class BeamSet:
    """
    BeamSet class for managing a collection of beams.
    
    Attributes:
        id (str): Unique identifier for the beam set
        name (str): Name of the beam set
        beams (Dict[str, Beam]): Dictionary of beams indexed by ID
        prescription_dose (float): Prescribed dose in Gy
        technique (str): Technique name (e.g., "VMAT", "IMRT")
        props (Dict): Additional properties
    """
    
    def __init__(self, name: str = ""):
        """
        Initialize a new BeamSet.
        
        Args:
            name: Name of the beam set
        """
        self.id = f"bs_{uuid.uuid4().hex[:8]}"
        self.name = name
        self.beams = {}
        self.prescription_dose = 2.0  # Default to 2 Gy
        self.technique = "STATIC"
        self.props = {}
        
    def add_beam(self, beam: Beam) -> None:
        """
        Add a beam to this beam set.
        
        Args:
            beam: Beam object to add
        """
        if not isinstance(beam, Beam):
            raise TypeError("beam must be an instance of Beam")
            
        if beam.id in self.beams:
            logger.warning(f"Beam with ID {beam.id} already exists in set {self.name}, overwriting")
            
        self.beams[beam.id] = beam
        logger.info(f"Added beam {beam.name} to set {self.name}")
        
    def get_beam(self, beam_id: str) -> Optional[Beam]:
        """
        Get a beam by ID.
        
        Args:
            beam_id: ID of the beam to retrieve
            
        Returns:
            Beam if found, None otherwise
        """
        return self.beams.get(beam_id)
        
    def get_beam_by_name(self, name: str) -> Optional[Beam]:
        """
        Get a beam by name.
        
        Args:
            name: Name of the beam to retrieve
            
        Returns:
            First beam with matching name if found, None otherwise
        """
        for beam in self.beams.values():
            if beam.name == name:
                return beam
        return None
        
    def remove_beam(self, beam_id: str) -> bool:
        """
        Remove a beam from this beam set.
        
        Args:
            beam_id: ID of the beam to remove
            
        Returns:
            True if the beam was removed, False if not found
        """
        if beam_id in self.beams:
            beam = self.beams[beam_id]
            del self.beams[beam_id]
            logger.info(f"Removed beam {beam.name} from set {self.name}")
            return True
        else:
            logger.warning(f"Beam with ID {beam_id} not found in set {self.name}")
            return False
            
    def normalize_weights(self) -> None:
        """
        Normalize beam weights to sum to 1.0.
        """
        total_weight = sum(beam.weight for beam in self.beams.values())
        if total_weight > 0:
            for beam in self.beams.values():
                beam.weight = beam.weight / total_weight
        else:
            # If all weights are 0, set equal weights
            weight = 1.0 / len(self.beams) if len(self.beams) > 0 else 0
            for beam in self.beams.values():
                beam.weight = weight
                
    def set_equal_weights(self) -> None:
        """
        Set equal weights for all beams.
        """
        weight = 1.0 / len(self.beams) if len(self.beams) > 0 else 0
        for beam in self.beams.values():
            beam.weight = weight
            
    def get_beam_ids(self) -> List[str]:
        """
        Get a list of all beam IDs in this set.
        
        Returns:
            List of beam IDs
        """
        return list(self.beams.keys())
        
    def get_beam_names(self) -> List[str]:
        """
        Get a list of all beam names in this set.
        
        Returns:
            List of beam names
        """
        return [b.name for b in self.beams.values()]
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert beam set to a dictionary.
        
        Returns:
            Dictionary representation of the beam set
        """
        beam_set_dict = {
            'id': self.id,
            'name': self.name,
            'prescription_dose': self.prescription_dose,
            'technique': self.technique,
            'beams': [beam.to_dict() for beam in self.beams.values()]
        }
        
        # Add props
        if self.props:
            beam_set_dict['props'] = self.props
            
        return beam_set_dict
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BeamSet':
        """
        Create a beam set from a dictionary.
        
        Args:
            data: Dictionary representation of a beam set
            
        Returns:
            New BeamSet instance
        """
        beam_set = cls(name=data.get('name', ''))
        beam_set.id = data.get('id', beam_set.id)
        beam_set.prescription_dose = data.get('prescription_dose', beam_set.prescription_dose)
        beam_set.technique = data.get('technique', beam_set.technique)
        
        # Add beams
        if 'beams' in data:
            for beam_data in data['beams']:
                beam = Beam.from_dict(beam_data)
                beam_set.add_beam(beam)
                
        # Add props
        if 'props' in data:
            beam_set.props = data['props']
            
        return beam_set
        
    def __len__(self) -> int:
        return len(self.beams)
        
    def __iter__(self):
        return iter(self.beams.values())
        
    def __str__(self) -> str:
        return f"BeamSet({self.name}, beams={len(self.beams)}, technique={self.technique})"
        
    def __repr__(self) -> str:
        return self.__str__() 