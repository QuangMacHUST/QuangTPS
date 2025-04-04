"""
Beam modifiers module for QuangTPS.

This module defines classes for various beam modifiers used in radiotherapy,
such as wedges, blocks, boluses, and compensators.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
import logging
import math
import uuid
import numpy as np

logger = logging.getLogger(__name__)

class WedgeType(Enum):
    """Wedge type enumeration."""
    PHYSICAL = "physical"
    DYNAMIC = "dynamic"
    ENHANCED_DYNAMIC = "enhanced_dynamic"
    VIRTUAL = "virtual"

class WedgeOrientation(Enum):
    """Wedge orientation enumeration."""
    IN = "in"
    OUT = "out"
    LEFT = "left"
    RIGHT = "right"

class Wedge:
    """
    Wedge class for representing a beam wedge modifier.
    
    Attributes:
        id (str): Unique identifier for the wedge
        name (str): Name of the wedge
        type (WedgeType): Type of wedge
        angle (float): Wedge angle in degrees
        orientation (WedgeOrientation): Orientation of the wedge
        factor (float): Wedge factor
    """
    
    def __init__(self, name: str = "", wedge_type: WedgeType = WedgeType.PHYSICAL):
        """
        Initialize a new Wedge.
        
        Args:
            name: Name of the wedge
            wedge_type: Type of wedge
        """
        self.id = f"w_{uuid.uuid4().hex[:8]}"
        self.name = name
        self.type = wedge_type
        self.angle = 15.0  # Default: 15 degrees
        self.orientation = WedgeOrientation.IN  # Default: IN
        self.factor = 1.0  # Default: No attenuation
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert wedge to a dictionary.
        
        Returns:
            Dictionary representation of the wedge
        """
        return {
            'id': self.id,
            'name': self.name,
            'type': self.type.value if isinstance(self.type, WedgeType) else self.type,
            'angle': self.angle,
            'orientation': self.orientation.value if isinstance(self.orientation, WedgeOrientation) else self.orientation,
            'factor': self.factor
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Wedge':
        """
        Create a wedge from a dictionary.
        
        Args:
            data: Dictionary representation of a wedge
            
        Returns:
            New Wedge instance
        """
        # Parse wedge type
        wedge_type_value = data.get('type', 'physical')
        try:
            wedge_type = WedgeType(wedge_type_value)
        except ValueError:
            logger.warning(f"Unknown wedge type: {wedge_type_value}, using PHYSICAL")
            wedge_type = WedgeType.PHYSICAL
            
        # Create wedge
        wedge = cls(name=data.get('name', ''), wedge_type=wedge_type)
        wedge.id = data.get('id', wedge.id)
        wedge.angle = data.get('angle', wedge.angle)
        
        # Parse orientation
        orientation_value = data.get('orientation', 'in')
        try:
            wedge.orientation = WedgeOrientation(orientation_value)
        except ValueError:
            logger.warning(f"Unknown wedge orientation: {orientation_value}, using IN")
            wedge.orientation = WedgeOrientation.IN
            
        wedge.factor = data.get('factor', wedge.factor)
        
        return wedge

class Block:
    """
    Block class for representing a beam block modifier.
    
    Attributes:
        id (str): Unique identifier for the block
        name (str): Name of the block
        contour (np.ndarray): 2D array of points defining the block contour
        thickness (float): Thickness of the block in cm
        ssd (float): Source-to-surface distance in mm
        transmission_factor (float): Transmission factor of the block
    """
    
    def __init__(self, name: str = ""):
        """
        Initialize a new Block.
        
        Args:
            name: Name of the block
        """
        self.id = f"bl_{uuid.uuid4().hex[:8]}"
        self.name = name
        self.contour = np.array([])  # Empty contour
        self.thickness = 7.5  # Default: 7.5 cm (standard for Cerrobend blocks)
        self.ssd = 950.0  # Default: 95 cm
        self.transmission_factor = 0.05  # Default: 5% transmission
        
    def set_contour(self, contour: np.ndarray):
        """
        Set the block contour.
        
        Args:
            contour: 2D array of points defining the block contour
        """
        self.contour = contour
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert block to a dictionary.
        
        Returns:
            Dictionary representation of the block
        """
        return {
            'id': self.id,
            'name': self.name,
            'contour': self.contour.tolist() if isinstance(self.contour, np.ndarray) else self.contour,
            'thickness': self.thickness,
            'ssd': self.ssd,
            'transmission_factor': self.transmission_factor
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Block':
        """
        Create a block from a dictionary.
        
        Args:
            data: Dictionary representation of a block
            
        Returns:
            New Block instance
        """
        block = cls(name=data.get('name', ''))
        block.id = data.get('id', block.id)
        
        # Parse contour
        contour_data = data.get('contour', [])
        if contour_data:
            block.contour = np.array(contour_data)
            
        block.thickness = data.get('thickness', block.thickness)
        block.ssd = data.get('ssd', block.ssd)
        block.transmission_factor = data.get('transmission_factor', block.transmission_factor)
        
        return block

class Bolus:
    """
    Bolus class for representing a beam bolus modifier.
    
    Attributes:
        id (str): Unique identifier for the bolus
        name (str): Name of the bolus
        thickness (float): Thickness of the bolus in cm
        material (str): Material of the bolus
        contour (np.ndarray): 2D array of points defining the bolus contour
        ssd (float): Source-to-surface distance in mm
    """
    
    def __init__(self, name: str = ""):
        """
        Initialize a new Bolus.
        
        Args:
            name: Name of the bolus
        """
        self.id = f"bo_{uuid.uuid4().hex[:8]}"
        self.name = name
        self.thickness = 1.0  # Default: 1 cm
        self.material = "water"  # Default: water-equivalent material
        self.contour = np.array([])  # Empty contour
        self.ssd = 950.0  # Default: 95 cm
        
    def set_contour(self, contour: np.ndarray):
        """
        Set the bolus contour.
        
        Args:
            contour: 2D array of points defining the bolus contour
        """
        self.contour = contour
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert bolus to a dictionary.
        
        Returns:
            Dictionary representation of the bolus
        """
        return {
            'id': self.id,
            'name': self.name,
            'thickness': self.thickness,
            'material': self.material,
            'contour': self.contour.tolist() if isinstance(self.contour, np.ndarray) else self.contour,
            'ssd': self.ssd
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Bolus':
        """
        Create a bolus from a dictionary.
        
        Args:
            data: Dictionary representation of a bolus
            
        Returns:
            New Bolus instance
        """
        bolus = cls(name=data.get('name', ''))
        bolus.id = data.get('id', bolus.id)
        bolus.thickness = data.get('thickness', bolus.thickness)
        bolus.material = data.get('material', bolus.material)
        
        # Parse contour
        contour_data = data.get('contour', [])
        if contour_data:
            bolus.contour = np.array(contour_data)
            
        bolus.ssd = data.get('ssd', bolus.ssd)
        
        return bolus

class Compensator:
    """
    Compensator class for representing a beam compensator modifier.
    
    Attributes:
        id (str): Unique identifier for the compensator
        name (str): Name of the compensator
        thickness_map (np.ndarray): 2D array of thickness values in cm
        material (str): Material of the compensator
        resolution (Tuple[float, float]): Resolution of the thickness map in mm/pixel
        ssd (float): Source-to-surface distance in mm
    """
    
    def __init__(self, name: str = ""):
        """
        Initialize a new Compensator.
        
        Args:
            name: Name of the compensator
        """
        self.id = f"c_{uuid.uuid4().hex[:8]}"
        self.name = name
        self.thickness_map = np.array([])  # Empty thickness map
        self.material = "brass"  # Default: brass material
        self.resolution = (5.0, 5.0)  # Default: 5 mm/pixel
        self.ssd = 950.0  # Default: 95 cm
        
    def set_thickness_map(self, thickness_map: np.ndarray):
        """
        Set the compensator thickness map.
        
        Args:
            thickness_map: 2D array of thickness values in cm
        """
        self.thickness_map = thickness_map
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert compensator to a dictionary.
        
        Returns:
            Dictionary representation of the compensator
        """
        return {
            'id': self.id,
            'name': self.name,
            'thickness_map': self.thickness_map.tolist() if isinstance(self.thickness_map, np.ndarray) else self.thickness_map,
            'material': self.material,
            'resolution': self.resolution,
            'ssd': self.ssd
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Compensator':
        """
        Create a compensator from a dictionary.
        
        Args:
            data: Dictionary representation of a compensator
            
        Returns:
            New Compensator instance
        """
        compensator = cls(name=data.get('name', ''))
        compensator.id = data.get('id', compensator.id)
        
        # Parse thickness map
        thickness_map_data = data.get('thickness_map', [])
        if thickness_map_data:
            compensator.thickness_map = np.array(thickness_map_data)
            
        compensator.material = data.get('material', compensator.material)
        compensator.resolution = data.get('resolution', compensator.resolution)
        compensator.ssd = data.get('ssd', compensator.ssd)
        
        return compensator 