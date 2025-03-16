"""
Core data types for QuangTPS.

This module defines the fundamental data types used throughout the QuangTPS
radiotherapy treatment planning system, providing consistent data structures
for all modules.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Union, Any
import SimpleITK as sitk
import numpy as np


@dataclass
class DoseGrid:
    """
    A three-dimensional dose grid.
    
    This class represents a 3D dose distribution, including its geometric
    properties and metadata.
    """
    
    data: np.ndarray  # 3D array of dose values (Gy)
    spacing: Tuple[float, float, float]  # Voxel spacing in mm
    origin: Tuple[float, float, float]  # Grid origin in mm
    direction: Optional[np.ndarray] = None  # Direction cosine matrix (3x3)
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata
    
    def to_sitk(self) -> sitk.Image:
        """
        Convert to SimpleITK image.
        
        Returns:
            SimpleITK image representation of the dose grid
        """
        img = sitk.GetImageFromArray(self.data.astype(np.float32))
        img.SetSpacing(self.spacing)
        img.SetOrigin(self.origin)
        
        if self.direction is not None:
            img.SetDirection(self.direction.flatten())
            
        # Set metadata
        for key, value in self.metadata.items():
            if isinstance(value, str):
                img.SetMetaData(key, value)
            else:
                img.SetMetaData(key, str(value))
        
        return img
    
    @classmethod
    def from_sitk(cls, image: sitk.Image) -> 'DoseGrid':
        """
        Create from SimpleITK image.
        
        Args:
            image: SimpleITK image containing dose data
            
        Returns:
            DoseGrid instance
        """
        data = sitk.GetArrayFromImage(image)
        spacing = image.GetSpacing()
        origin = image.GetOrigin()
        direction = np.array(image.GetDirection()).reshape(3, 3)
        
        # Extract metadata
        metadata = {}
        for key in image.GetMetaDataKeys():
            metadata[key] = image.GetMetaData(key)
        
        return cls(
            data=data,
            spacing=spacing,
            origin=origin,
            direction=direction,
            metadata=metadata
        )
    
    def get_dose_at_point(self, point: Tuple[float, float, float]) -> float:
        """
        Get dose value at a specific point in 3D space.
        
        Args:
            point: 3D coordinates (x, y, z) in mm
            
        Returns:
            Interpolated dose value at the point (Gy)
        """
        # Convert point to grid indices
        ix = (point[0] - self.origin[0]) / self.spacing[0]
        iy = (point[1] - self.origin[1]) / self.spacing[1]
        iz = (point[2] - self.origin[2]) / self.spacing[2]
        
        # Check if point is within grid bounds
        if (0 <= ix < self.data.shape[2] - 1 and 
            0 <= iy < self.data.shape[1] - 1 and 
            0 <= iz < self.data.shape[0] - 1):
            
            # Trilinear interpolation
            ix_floor, iy_floor, iz_floor = int(ix), int(iy), int(iz)
            ix_ceil, iy_ceil, iz_ceil = ix_floor + 1, iy_floor + 1, iz_floor + 1
            
            # Interpolation weights
            wx = ix - ix_floor
            wy = iy - iy_floor
            wz = iz - iz_floor
            
            # Interpolate
            dose = (
                self.data[iz_floor, iy_floor, ix_floor] * (1 - wx) * (1 - wy) * (1 - wz) +
                self.data[iz_floor, iy_floor, ix_ceil] * wx * (1 - wy) * (1 - wz) +
                self.data[iz_floor, iy_ceil, ix_floor] * (1 - wx) * wy * (1 - wz) +
                self.data[iz_floor, iy_ceil, ix_ceil] * wx * wy * (1 - wz) +
                self.data[iz_ceil, iy_floor, ix_floor] * (1 - wx) * (1 - wy) * wz +
                self.data[iz_ceil, iy_floor, ix_ceil] * wx * (1 - wy) * wz +
                self.data[iz_ceil, iy_ceil, ix_floor] * (1 - wx) * wy * wz +
                self.data[iz_ceil, iy_ceil, ix_ceil] * wx * wy * wz
            )
            
            return dose
        else:
            # Point is outside grid
            return 0.0
    
    def resample_to_image(self, reference_image: sitk.Image) -> 'DoseGrid':
        """
        Resample the dose grid to match a reference image.
        
        Args:
            reference_image: Image with desired geometry
            
        Returns:
            Resampled dose grid
        """
        # Convert to SimpleITK image
        dose_image = self.to_sitk()
        
        # Create resampler
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(reference_image)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(0.0)
        
        # Resample dose
        resampled_image = resampler.Execute(dose_image)
        
        # Return new dose grid
        return DoseGrid.from_sitk(resampled_image)


@dataclass
class BeamParameters:
    """
    Parameters describing a radiation beam.
    
    This class contains all parameters needed to define a radiation beam
    for dose calculation.
    """
    
    beam_name: str  # Beam name/ID
    beam_type: str  # 'photon', 'electron', 'proton', etc.
    nominal_energy: float  # Nominal energy in MeV
    isocenter: Tuple[float, float, float]  # Isocenter position in mm
    gantry_angle: float  # Gantry angle in degrees
    collimator_angle: float = 0.0  # Collimator angle in degrees
    couch_angle: float = 0.0  # Couch angle in degrees
    field_size: Tuple[float, float] = (100.0, 100.0)  # Field size in mm
    sad: Optional[float] = 1000.0  # Source-axis distance in mm
    ssd: Optional[float] = None  # Source-surface distance in mm
    
    # MLC configuration for IMRT/VMAT
    mlc_positions: Optional[List[Tuple[float, float]]] = None  # Leaf positions in mm
    
    # Wedge filter
    wedge_angle: Optional[float] = None  # Wedge angle in degrees
    wedge_orientation: Optional[float] = None  # Wedge orientation in degrees
    
    # Monitor units
    monitor_units: float = 100.0  # Monitor units (MU)
    
    # Dose grid normalization
    dose_grid_normalization: Optional[float] = None  # Normalization factor for dose grid
    
    # Beam modifiers
    applicator_id: Optional[str] = None  # Electron applicator ID
    applicator_size: Optional[float] = None  # Electron applicator size in mm
    bolus_thickness: Optional[float] = None  # Bolus thickness in mm
    
    # Additional parameters for specific beam types
    additional_parameters: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize derived parameters."""
        # Calculate SSD if not provided
        if self.ssd is None and self.sad is not None:
            # In a real implementation, this would calculate SSD based on patient surface
            # For now, use a default value
            self.ssd = self.sad - 50.0  # Typical patient thickness
    
    def get_beam_direction(self) -> np.ndarray:
        """
        Get the beam direction vector.
        
        Returns:
            3D unit vector pointing from source to isocenter
        """
        # Convert angles to radians
        gantry_rad = np.radians(self.gantry_angle)
        couch_rad = np.radians(self.couch_angle)
        
        # Calculate beam direction
        # At gantry=0, beam points in +z direction
        direction = np.array([
            np.sin(gantry_rad),
            0,
            np.cos(gantry_rad)
        ])
        
        # Apply couch rotation
        if abs(couch_rad) > 1e-6:
            # Rotation around z-axis
            couch_cos = np.cos(couch_rad)
            couch_sin = np.sin(couch_rad)
            
            direction = np.array([
                direction[0] * couch_cos - direction[1] * couch_sin,
                direction[0] * couch_sin + direction[1] * couch_cos,
                direction[2]
            ])
        
        return direction
    
    def get_source_position(self) -> np.ndarray:
        """
        Get the source position.
        
        Returns:
            3D position of the radiation source
        """
        # Get beam direction
        direction = self.get_beam_direction()
        
        # Calculate source position
        source_pos = np.array(self.isocenter) - direction * self.sad
        
        return source_pos
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary.
        
        Returns:
            Dictionary representation of beam parameters
        """
        return {
            'beam_name': self.beam_name,
            'beam_type': self.beam_type,
            'nominal_energy': self.nominal_energy,
            'isocenter': self.isocenter,
            'gantry_angle': self.gantry_angle,
            'collimator_angle': self.collimator_angle,
            'couch_angle': self.couch_angle,
            'field_size': self.field_size,
            'sad': self.sad,
            'ssd': self.ssd,
            'monitor_units': self.monitor_units,
            'wedge_angle': self.wedge_angle,
            'wedge_orientation': self.wedge_orientation,
            'applicator_id': self.applicator_id,
            'applicator_size': self.applicator_size,
            'bolus_thickness': self.bolus_thickness,
            'additional_parameters': self.additional_parameters
        }
