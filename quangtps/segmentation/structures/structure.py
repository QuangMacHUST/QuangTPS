#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Structure Module for QuangTPS.

This module provides the Structure class for representing anatomical structures
and tumors in radiotherapy treatment planning. Each structure contains contour
information, properties, and methods for manipulation and analysis.
"""

import logging
import uuid
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass, field, asdict
import SimpleITK as sitk

from quangtps.core.exceptions import ValidationError
from quangtps.segmentation.structures.structure_templates import StructureTemplate

logger = logging.getLogger(__name__)


class Structure:
    """
    Class representing a radiotherapy structure (anatomical structure or tumor).
    
    A Structure contains 3D contour information, properties, and methods for 
    manipulating and analyzing the structure. It can be used to represent 
    targets (like PTV, CTV, GTV) or organs at risk (OARs).
    """
    
    def __init__(self, 
                 name: str,
                 contour_data: Optional[np.ndarray] = None,
                 color: Tuple[int, int, int] = (255, 0, 0),
                 structure_id: Optional[str] = None,
                 description: str = "",
                 structure_type: str = "TARGET",
                 properties: Dict[str, Any] = None):
        """
        Initialize a Structure.
        
        Parameters
        ----------
        name : str
            Name of the structure
        contour_data : Optional[np.ndarray], optional
            3D binary mask of the structure, by default None
        color : Tuple[int, int, int], optional
            RGB color for visualization, by default (255, 0, 0)
        structure_id : Optional[str], optional
            Unique ID for the structure, by default None (will generate UUID)
        description : str, optional
            Description of the structure, by default ""
        structure_type : str, optional
            Type of structure (TARGET, ORGAN_AT_RISK, etc.), by default "TARGET"
        properties : Dict[str, Any], optional
            Additional properties, by default None
        """
        self.name = name
        self.description = description
        self.color = color
        self.structure_id = structure_id if structure_id else str(uuid.uuid4())
        self.structure_type = structure_type
        self.properties = properties or {}
        
        # For visualization
        self.alpha = 0.5  # Transparency
        self.line_width = 2  # Line width for contour drawing
        
        # Dose constraints for planning
        self.dose_constraints = {}
        
        # Contour data - 3D binary mask
        self._contour_data = None
        if contour_data is not None:
            self.set_contour_data(contour_data)
        
        # Metadata
        self.origin = (0, 0, 0)  # Origin in world coordinates
        self.spacing = (1, 1, 1)  # Voxel spacing in mm
        self.direction = np.eye(3).flatten().tolist()  # Direction cosines
    
    def set_contour_data(self, contour_data: np.ndarray):
        """
        Set the contour data for this structure.
        
        Parameters
        ----------
        contour_data : np.ndarray
            3D binary mask of the structure
            
        Raises
        ------
        ValidationError
            If the contour data is not a valid binary mask
        """
        if not isinstance(contour_data, np.ndarray):
            raise ValidationError("Contour data must be a numpy array")
        
        if contour_data.ndim != 3:
            raise ValidationError("Contour data must be a 3D array")
        
        # Ensure binary mask (0 and 1 values only)
        unique_values = np.unique(contour_data)
        if not np.all(np.isin(unique_values, [0, 1])):
            logger.warning("Non-binary values found in contour data. Converting to binary mask.")
            contour_data = (contour_data > 0).astype(np.uint8)
        
        self._contour_data = contour_data
    
    def get_contour_data(self) -> Optional[np.ndarray]:
        """
        Get the contour data for this structure.
        
        Returns
        -------
        Optional[np.ndarray]
            3D binary mask of the structure, or None if not set
        """
        return self._contour_data
    
    def get_volume(self, voxel_spacing: Optional[Tuple[float, float, float]] = None) -> float:
        """
        Calculate the volume of the structure in cubic millimeters.
        
        Parameters
        ----------
        voxel_spacing : Optional[Tuple[float, float, float]], optional
            Voxel spacing in mm, by default None (uses structure spacing)
            
        Returns
        -------
        float
            Volume in cubic millimeters (mm³)
            
        Raises
        ------
        ValidationError
            If contour data is not set
        """
        if self._contour_data is None:
            raise ValidationError("Cannot calculate volume: contour data not set")
        
        spacing = voxel_spacing or self.spacing
        voxel_volume = spacing[0] * spacing[1] * spacing[2]
        
        # Count non-zero voxels and multiply by voxel volume
        return np.count_nonzero(self._contour_data) * voxel_volume
    
    def get_center_of_mass(self) -> Tuple[float, float, float]:
        """
        Calculate the center of mass of the structure.
        
        Returns
        -------
        Tuple[float, float, float]
            Center of mass coordinates (x, y, z)
            
        Raises
        ------
        ValidationError
            If contour data is not set
        """
        if self._contour_data is None:
            raise ValidationError("Cannot calculate center of mass: contour data not set")
        
        # Use SciPy's center of mass calculation
        from scipy import ndimage
        center = ndimage.center_of_mass(self._contour_data)
        
        # Convert to world coordinates
        x = self.origin[0] + center[0] * self.spacing[0]
        y = self.origin[1] + center[1] * self.spacing[1]
        z = self.origin[2] + center[2] * self.spacing[2]
        
        return (x, y, z)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the structure to a dictionary.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary representation of the structure
        """
        result = {
            'name': self.name,
            'description': self.description,
            'color': self.color,
            'structure_id': self.structure_id,
            'structure_type': self.structure_type,
            'properties': self.properties,
            'alpha': self.alpha,
            'line_width': self.line_width,
            'dose_constraints': self.dose_constraints,
            'origin': self.origin,
            'spacing': self.spacing,
            'direction': self.direction
        }
        
        # We don't include contour_data in the dict as it's too large
        # Caller needs to handle this separately if needed
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], contour_data: Optional[np.ndarray] = None) -> 'Structure':
        """
        Create a structure from a dictionary.
        
        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary containing structure data
        contour_data : Optional[np.ndarray], optional
            3D binary mask of the structure, by default None
            
        Returns
        -------
        Structure
            New structure instance
        """
        structure = cls(
            name=data.get('name', 'Unnamed Structure'),
            contour_data=contour_data,
            color=data.get('color', (255, 0, 0)),
            structure_id=data.get('structure_id'),
            description=data.get('description', ''),
            structure_type=data.get('structure_type', 'TARGET'),
            properties=data.get('properties', {})
        )
        
        # Set additional properties
        structure.alpha = data.get('alpha', 0.5)
        structure.line_width = data.get('line_width', 2)
        structure.dose_constraints = data.get('dose_constraints', {})
        structure.origin = data.get('origin', (0, 0, 0))
        structure.spacing = data.get('spacing', (1, 1, 1))
        structure.direction = data.get('direction', np.eye(3).flatten().tolist())
        
        return structure
    
    @classmethod
    def from_template(cls, template: StructureTemplate, contour_data: Optional[np.ndarray] = None) -> 'Structure':
        """
        Create a structure from a template.
        
        Parameters
        ----------
        template : StructureTemplate
            Template to create structure from
        contour_data : Optional[np.ndarray], optional
            3D binary mask of the structure, by default None
            
        Returns
        -------
        Structure
            New structure instance based on the template
        """
        structure = cls(
            name=template.name,
            contour_data=contour_data,
            color=template.color,
            description=template.description,
            properties=template.properties.copy()
        )
        
        structure.alpha = template.alpha
        structure.line_width = template.line_width
        structure.dose_constraints = template.dose_constraints.copy()
        
        return structure
    
    def resample(self, target_spacing: Tuple[float, float, float], 
                 interpolation: str = 'nearest') -> 'Structure':
        """
        Resample the structure to a different voxel spacing.
        
        Parameters
        ----------
        target_spacing : Tuple[float, float, float]
            Target voxel spacing in mm
        interpolation : str, optional
            Interpolation method ('nearest', 'linear', etc.), by default 'nearest'
            
        Returns
        -------
        Structure
            Resampled structure
            
        Raises
        ------
        ValidationError
            If contour data is not set
        """
        if self._contour_data is None:
            raise ValidationError("Cannot resample: contour data not set")
        
        # Convert to SimpleITK image for resampling
        sitk_image = sitk.GetImageFromArray(self._contour_data.astype(np.uint8))
        sitk_image.SetSpacing(self.spacing)
        sitk_image.SetOrigin(self.origin)
        sitk_image.SetDirection(self.direction)
        
        # Calculate new size
        current_size = sitk_image.GetSize()
        new_size = [
            int(round(current_size[0] * self.spacing[0] / target_spacing[0])),
            int(round(current_size[1] * self.spacing[1] / target_spacing[1])),
            int(round(current_size[2] * self.spacing[2] / target_spacing[2]))
        ]
        
        # Set up the resampler
        if interpolation == 'nearest':
            interp_method = sitk.sitkNearestNeighbor
        elif interpolation == 'linear':
            interp_method = sitk.sitkLinear
        else:
            interp_method = sitk.sitkNearestNeighbor  # Default to nearest neighbor
        
        resampler = sitk.ResampleImageFilter()
        resampler.SetSize(new_size)
        resampler.SetOutputSpacing(target_spacing)
        resampler.SetOutputOrigin(self.origin)
        resampler.SetOutputDirection(self.direction)
        resampler.SetInterpolator(interp_method)
        resampler.SetDefaultPixelValue(0)
        
        # Perform resampling
        resampled_image = resampler.Execute(sitk_image)
        resampled_array = sitk.GetArrayFromImage(resampled_image)
        
        # Create new structure with resampled data
        new_structure = Structure(
            name=self.name,
            contour_data=resampled_array,
            color=self.color,
            structure_id=self.structure_id,
            description=self.description,
            structure_type=self.structure_type,
            properties=self.properties.copy()
        )
        
        new_structure.alpha = self.alpha
        new_structure.line_width = self.line_width
        new_structure.dose_constraints = self.dose_constraints.copy()
        new_structure.origin = self.origin
        new_structure.spacing = target_spacing
        new_structure.direction = self.direction
        
        return new_structure
    
    def __str__(self) -> str:
        """
        Get string representation of the structure.
        
        Returns
        -------
        str
            String representation including name and type
        """
        return f"Structure({self.name}, type={self.structure_type})"
    
    def __repr__(self) -> str:
        """
        Get detailed string representation of the structure.
        
        Returns
        -------
        str
            Detailed string representation including ID and data shape
        """
        shape_str = "None" if self._contour_data is None else str(self._contour_data.shape)
        return f"Structure(name='{self.name}', id='{self.structure_id}', type='{self.structure_type}', shape={shape_str})"
