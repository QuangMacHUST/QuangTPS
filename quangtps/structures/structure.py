"""
Structure module for QuangTPS.

This module defines the Structure class used in radiotherapy treatment planning.
A Structure represents a contoured region of interest, such as a target volume
or an organ at risk.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import logging
import uuid
import numpy as np

logger = logging.getLogger(__name__)

class Structure:
    """
    Structure class for representing contoured regions of interest.
    
    Attributes:
        id (str): Unique identifier for the structure
        name (str): Name of the structure
        type (str): Type of structure (e.g., "PTV", "OAR", "BODY", etc.)
        color (Tuple[int, int, int]): RGB color for displaying the structure
        mask (np.ndarray): 3D boolean array representing the structure mask
        visible (bool): Whether the structure is visible in the UI
        opacity (float): Opacity for displaying the structure (0.0-1.0)
        props (Dict): Additional properties
    """
    
    def __init__(self, name: str = "", structure_type: str = "OTHER"):
        """
        Initialize a new Structure.
        
        Args:
            name: Name of the structure
            structure_type: Type of structure (e.g., "PTV", "OAR", "BODY")
        """
        self.id = f"s_{uuid.uuid4().hex[:8]}"
        self.name = name
        self.type = structure_type
        self.color = (255, 0, 0)  # Default to red
        self.mask = None
        self.visible = True
        self.opacity = 0.5
        self.props = {}
        
    def set_mask(self, mask: np.ndarray):
        """
        Set the mask for this structure.
        
        Args:
            mask: 3D boolean array representing the structure
        """
        if not isinstance(mask, np.ndarray):
            raise TypeError("Mask must be a numpy ndarray")
        
        if mask.dtype != bool:
            logger.warning(f"Converting mask for {self.name} to boolean")
            mask = mask.astype(bool)
            
        self.mask = mask
        logger.info(f"Mask set for structure {self.name}, shape: {mask.shape}")
        
    def get_volume(self, voxel_size: Tuple[float, float, float] = (1.0, 1.0, 1.0)) -> float:
        """
        Calculate the volume of the structure.
        
        Args:
            voxel_size: Size of each voxel in mm (dx, dy, dz)
            
        Returns:
            Volume in cubic centimeters (cc)
        """
        if self.mask is None:
            logger.warning(f"Can't calculate volume for {self.name}: mask is None")
            return 0.0
            
        voxel_volume_mm3 = voxel_size[0] * voxel_size[1] * voxel_size[2]
        volume_mm3 = np.sum(self.mask) * voxel_volume_mm3
        volume_cc = volume_mm3 / 1000.0  # Convert to cc
        
        return volume_cc
        
    def get_centroid(self, voxel_size: Tuple[float, float, float] = (1.0, 1.0, 1.0)) -> Tuple[float, float, float]:
        """
        Calculate the centroid of the structure.
        
        Args:
            voxel_size: Size of each voxel in mm (dx, dy, dz)
            
        Returns:
            Centroid coordinates (x, y, z) in mm
        """
        if self.mask is None or not np.any(self.mask):
            logger.warning(f"Can't calculate centroid for {self.name}: mask is None or empty")
            return (0.0, 0.0, 0.0)
            
        indices = np.where(self.mask)
        x = np.mean(indices[0]) * voxel_size[0]
        y = np.mean(indices[1]) * voxel_size[1]
        z = np.mean(indices[2]) * voxel_size[2]
        
        return (x, y, z)
        
    def create_copy(self, new_name: Optional[str] = None) -> 'Structure':
        """
        Create a copy of this structure.
        
        Args:
            new_name: Optional new name for the copied structure
            
        Returns:
            A new Structure instance with the same properties
        """
        if new_name is None:
            new_name = f"{self.name}_copy"
            
        new_struct = Structure(new_name, self.type)
        new_struct.color = self.color
        if self.mask is not None:
            new_struct.mask = self.mask.copy()
        new_struct.visible = self.visible
        new_struct.opacity = self.opacity
        new_struct.props = self.props.copy()
        
        return new_struct
        
    def __str__(self) -> str:
        return f"Structure({self.name}, type={self.type})"
        
    def __repr__(self) -> str:
        return self.__str__() 