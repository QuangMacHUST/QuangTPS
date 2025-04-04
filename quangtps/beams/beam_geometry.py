"""
Beam geometry module for QuangTPS.

This module defines the BeamGeometry class and related enums for managing
the geometric properties of radiotherapy beams.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
import logging
import math
import numpy as np

logger = logging.getLogger(__name__)

class GantryDirection(Enum):
    """Gantry rotation direction."""
    CLOCKWISE = "CW"
    COUNTER_CLOCKWISE = "CCW"

class CollimatorDirection(Enum):
    """Collimator rotation direction."""
    CLOCKWISE = "CW"
    COUNTER_CLOCKWISE = "CCW"

class CouchDirection(Enum):
    """Couch rotation direction."""
    CLOCKWISE = "CW"
    COUNTER_CLOCKWISE = "CCW"

class BeamGeometry:
    """
    BeamGeometry class for managing geometric properties of radiotherapy beams.
    
    Attributes:
        source_position (Tuple[float, float, float]): Source position in mm
        isocenter (Tuple[float, float, float]): Isocenter position in mm
        gantry_angle (float): Gantry angle in degrees
        collimator_angle (float): Collimator angle in degrees
        couch_angle (float): Couch angle in degrees
        sad (float): Source-to-axis distance in mm
        field_size (Tuple[float, float]): Field size in cm (width, height)
        effective_field_size (Tuple[float, float]): Effective field size in cm (width, height)
    """
    
    def __init__(self):
        """Initialize a new BeamGeometry."""
        self.source_position = (0.0, 0.0, 1000.0)  # Default: 100cm above isocenter
        self.isocenter = (0.0, 0.0, 0.0)  # Default: at origin
        self.gantry_angle = 0.0  # Default: 0 degrees (anterior)
        self.collimator_angle = 0.0  # Default: 0 degrees
        self.couch_angle = 0.0  # Default: 0 degrees
        self.sad = 1000.0  # Default: 100cm
        self.field_size = (100.0, 100.0)  # Default: 10x10 cm (in mm)
        self.effective_field_size = (100.0, 100.0)  # Default: same as field_size
        
    def set_angles(self, gantry: float, collimator: float, couch: float) -> None:
        """
        Set all angles at once.
        
        Args:
            gantry: Gantry angle in degrees
            collimator: Collimator angle in degrees
            couch: Couch angle in degrees
        """
        self.gantry_angle = gantry
        self.collimator_angle = collimator
        self.couch_angle = couch
        self._update_source_position()
        
    def set_gantry_angle(self, angle: float) -> None:
        """
        Set the gantry angle.
        
        Args:
            angle: Gantry angle in degrees
        """
        self.gantry_angle = angle
        self._update_source_position()
        
    def set_collimator_angle(self, angle: float) -> None:
        """
        Set the collimator angle.
        
        Args:
            angle: Collimator angle in degrees
        """
        self.collimator_angle = angle
        
    def set_couch_angle(self, angle: float) -> None:
        """
        Set the couch angle.
        
        Args:
            angle: Couch angle in degrees
        """
        self.couch_angle = angle
        
    def set_isocenter(self, isocenter: Tuple[float, float, float]) -> None:
        """
        Set the isocenter position.
        
        Args:
            isocenter: Isocenter position in mm (x, y, z)
        """
        self.isocenter = isocenter
        self._update_source_position()
        
    def set_sad(self, sad: float) -> None:
        """
        Set the source-to-axis distance.
        
        Args:
            sad: Source-to-axis distance in mm
        """
        self.sad = sad
        self._update_source_position()
        
    def set_field_size(self, width: float, height: float) -> None:
        """
        Set the field size.
        
        Args:
            width: Field width in cm
            height: Field height in cm
        """
        # Convert to mm
        self.field_size = (width * 10.0, height * 10.0)
        self.effective_field_size = self.field_size
        
    def _update_source_position(self) -> None:
        """Update the source position based on isocenter, SAD, and gantry angle."""
        # Convert gantry angle to radians
        gantry_rad = math.radians(self.gantry_angle)
        
        # Calculate source position
        x = self.isocenter[0] - self.sad * math.sin(gantry_rad)
        y = self.isocenter[1]
        z = self.isocenter[2] + self.sad * math.cos(gantry_rad)
        
        self.source_position = (x, y, z)
        
    def get_beam_direction(self) -> Tuple[float, float, float]:
        """
        Get the beam direction vector.
        
        Returns:
            Normalized beam direction vector
        """
        # Calculate direction vector from source to isocenter
        dx = self.isocenter[0] - self.source_position[0]
        dy = self.isocenter[1] - self.source_position[1]
        dz = self.isocenter[2] - self.source_position[2]
        
        # Normalize
        length = math.sqrt(dx*dx + dy*dy + dz*dz)
        if length > 0:
            return (dx/length, dy/length, dz/length)
        else:
            return (0.0, 0.0, 1.0)  # Default to beam pointing along z-axis
            
    def get_collimator_x_direction(self) -> Tuple[float, float, float]:
        """
        Get the X direction of the collimator.
        
        Returns:
            X direction vector in patient coordinates
        """
        # First get beam direction
        beam_dir = self.get_beam_direction()
        
        # Define default X direction (perpendicular to beam direction)
        if abs(beam_dir[2]) < 0.99:  # Not pointing along Z axis
            x_dir = np.cross([0, 0, 1], beam_dir)
        else:  # Pointing along Z axis
            x_dir = [1, 0, 0]
            
        # Normalize
        x_dir = x_dir / np.linalg.norm(x_dir)
        
        # Rotate by collimator angle
        coll_rad = math.radians(self.collimator_angle)
        y_dir = np.cross(beam_dir, x_dir)
        
        x_rotated = x_dir * math.cos(coll_rad) + y_dir * math.sin(coll_rad)
        
        return tuple(x_rotated)
        
    def get_collimator_y_direction(self) -> Tuple[float, float, float]:
        """
        Get the Y direction of the collimator.
        
        Returns:
            Y direction vector in patient coordinates
        """
        # Get beam direction and X direction
        beam_dir = self.get_beam_direction()
        x_dir = self.get_collimator_x_direction()
        
        # Y direction is cross product of beam direction and X direction
        y_dir = np.cross(beam_dir, x_dir)
        
        return tuple(y_dir)
        
    def get_beam_divergence(self) -> Tuple[float, float]:
        """
        Get the beam divergence angles.
        
        Returns:
            Tuple of (x_divergence, y_divergence) in radians
        """
        # Calculate divergence from field size and SAD
        x_divergence = math.atan(self.field_size[0] / (2 * self.sad))
        y_divergence = math.atan(self.field_size[1] / (2 * self.sad))
        
        return (x_divergence, y_divergence)
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert beam geometry to a dictionary.
        
        Returns:
            Dictionary representation of the beam geometry
        """
        return {
            'source_position': self.source_position,
            'isocenter': self.isocenter,
            'gantry_angle': self.gantry_angle,
            'collimator_angle': self.collimator_angle,
            'couch_angle': self.couch_angle,
            'sad': self.sad,
            'field_size': self.field_size,
            'effective_field_size': self.effective_field_size
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BeamGeometry':
        """
        Create a beam geometry from a dictionary.
        
        Args:
            data: Dictionary representation of a beam geometry
            
        Returns:
            New BeamGeometry instance
        """
        geom = cls()
        geom.isocenter = data.get('isocenter', geom.isocenter)
        geom.gantry_angle = data.get('gantry_angle', geom.gantry_angle)
        geom.collimator_angle = data.get('collimator_angle', geom.collimator_angle)
        geom.couch_angle = data.get('couch_angle', geom.couch_angle)
        geom.sad = data.get('sad', geom.sad)
        geom.field_size = data.get('field_size', geom.field_size)
        geom.effective_field_size = data.get('effective_field_size', geom.effective_field_size)
        
        # Source position might be included or we can recalculate it
        if 'source_position' in data:
            geom.source_position = data['source_position']
        else:
            geom._update_source_position()
            
        return geom 