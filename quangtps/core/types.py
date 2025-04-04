"""
Core data types for QuangTPS.

This module defines the fundamental data types used throughout the QuangTPS
radiotherapy treatment planning system, providing consistent data structures
for all modules.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Union, Any, TYPE_CHECKING
from enum import Enum, auto
import SimpleITK as sitk
import numpy as np
import uuid
from datetime import datetime, date
import os
import sys
import enum

# Use TYPE_CHECKING to avoid circular imports
if TYPE_CHECKING:
    from quangtps.evaluation.dvh.dvh_analysis import DVHAnalysis

# Define PatientStatus directly here to avoid circular imports
class PatientStatus(str, Enum):
    """Trạng thái của bệnh nhân."""
    ACTIVE = "Active"
    PLANNED = "Planned"
    ON_TREATMENT = "On Treatment"
    COMPLETED = "Completed"
    ON_HOLD = "On Hold"
    ARCHIVED = "Archived"
    DECEASED = "Deceased"
    UNKNOWN = "Unknown"


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


class BeamEnergyType(Enum):
    """Loại năng lượng của chùm tia"""
    PHOTON = "photon"
    ELECTRON = "electron"
    PROTON = "proton"
    NEUTRON = "neutron"
    CARBON = "carbon"
    UNKNOWN = "unknown"


class TechniqueType(Enum):
    """Loại kỹ thuật xạ trị."""
    CONFORMAL = "3D-CRT"
    IMRT = "IMRT"
    VMAT = "VMAT"
    SRS = "SRS"
    SBRT = "SBRT"
    ELECTRON = "Electron"
    UNKNOWN = "Unknown"


class BeamType(Enum):
    """Loại chùm tia xạ trị."""
    STATIC = "static"
    ARC = "arc"
    DYNAMIC = "dynamic"
    

class StructureType(Enum):
    """Loại cấu trúc."""
    PTV = "ptv"
    CTV = "ctv"
    GTV = "gtv"
    OAR = "oar"
    EXTERNAL = "external"
    IMPLANT = "implant"
    COUCH = "couch"
    BOLUS = "bolus"
    SUPPORT = "support"
    ISOCENTER = "isocenter"
    MARKER = "marker"
    CONTRAST = "contrast"
    CAVITY = "cavity"
    UNDEFINED = "undefined"
    

class PatientPosition(Enum):
    """Vị trí của bệnh nhân."""
    HFS = "HFS"  # Head First-Supine
    HFP = "HFP"  # Head First-Prone
    FFS = "FFS"  # Feet First-Supine
    FFP = "FFP"  # Feet First-Prone
    HFDR = "HFDR"  # Head First-Decubitus Right
    HFDL = "HFDL"  # Head First-Decubitus Left
    FFDR = "FFDR"  # Feet First-Decubitus Right
    FFDL = "FFDL"  # Feet First-Decubitus Left
    UNKNOWN = "Unknown"


class ImageModality(Enum):
    """Các loại hình thức hình ảnh."""
    CT = "CT"
    MRI = "MR"
    PET = "PT"
    RTDOSE = "RTDOSE"
    CBCT = "CBCT"
    RTPLAN = "RTPLAN"
    RTSTRUCT = "RTSTRUCT"
    RTIMAGE = "RTIMAGE"
    US = "US"
    UNKNOWN = "UNKNOWN"
    

class Orientation(Enum):
    """Hướng của hình ảnh."""
    AXIAL = "axial"
    SAGITTAL = "sagittal"
    CORONAL = "coronal"
    OBLIQUE = "oblique"
    

class BinaryOperation(Enum):
    """Các phép toán binary."""
    AND = "and"
    OR = "or"
    SUB = "sub"
    XOR = "xor"
    

class DoseUnit(Enum):
    """Đơn vị liều."""
    GY = "Gy"
    CGY = "cGy"
    

class VolumeUnit(Enum):
    """Đơn vị thể tích."""
    CC = "cm³"
    ML = "ml"
    

class LengthUnit(Enum):
    """Đơn vị độ dài."""
    MM = "mm"
    CM = "cm"
    

class TimeUnit(Enum):
    """Đơn vị thời gian."""
    S = "s"
    MIN = "min"
    H = "h"
    

class BeamStatus(Enum):
    """Trạng thái của chùm tia."""
    PLANNING = "planning"
    APPROVED = "approved"
    DELIVERED = "delivered"
    INTERRUPTED = "interrupted"
    CANCELED = "canceled"
    

class PlanStatus(Enum):
    """Trạng thái của kế hoạch."""
    PLANNING = "planning"
    APPROVED = "approved"
    DELIVERED = "delivered"
    INTERRUPTED = "interrupted"
    CANCELED = "canceled"
    

class TreatmentStatus(Enum):
    """Trạng thái của điều trị."""
    PLANNING = "planning"
    READY = "ready"
    ONGOING = "ongoing"
    COMPLETED = "completed"
    INTERRUPTED = "interrupted"
    CANCELED = "canceled"
    

class FractionStatus(Enum):
    """Trạng thái của phân liều."""
    PLANNED = "planned"
    DELIVERED = "delivered"
    PARTIAL = "partial"
    CANCELED = "canceled"
    

class TaskStatus(Enum):
    """Trạng thái của nhiệm vụ."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"
    

class RoleType(Enum):
    """Loại vai trò người dùng."""
    ADMIN = "admin"
    PHYSICIST = "physicist"
    PHYSICIAN = "physician"
    THERAPIST = "therapist"
    DOSIMETRIST = "dosimetrist"
    RESEARCHER = "researcher"
    GUEST = "guest"


class DataType:
    """Base class for data types with serialization capabilities."""
    
    def __init__(self):
        """Initialize the data type."""
        pass
    
    def to_dict(self) -> Dict:
        """
        Convert the object to a dictionary for serialization.
        
        Returns:
            Dict: Dictionary representation of the object
        """
        # Get all attributes that don't start with underscore
        attrs = {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        
        # Handle special types
        for key, value in attrs.items():
            if isinstance(value, Enum):
                attrs[key] = value.value
            elif isinstance(value, np.ndarray):
                attrs[key] = value.tolist()
            elif isinstance(value, (datetime, date)):
                attrs[key] = value.isoformat()
            elif hasattr(value, 'to_dict') and callable(getattr(value, 'to_dict')):
                attrs[key] = value.to_dict()
        
        return attrs
    
    def from_dict(self, data: Dict) -> None:
        """
        Update the object from a dictionary.
        
        Args:
            data: Dictionary with attribute values
        """
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def validate(self) -> bool:
        """
        Validate the object.
        
        Returns:
            bool: True if valid, False otherwise
        """
        return True


@dataclass
class Patient(DataType):
    """
    Patient information.
    
    This class represents a patient in the radiation therapy planning system,
    including demographic and medical information.
    """
    patient_id: str  # Hospital ID of the patient
    name: str  # Full name of the patient
    birth_date: Optional[Union[date, str]] = None  # Birth date
    gender: Optional[str] = None  # M, F, O (Other)
    
    # Additional demographics and medical information
    height: Optional[float] = None  # Height in cm
    weight: Optional[float] = None  # Weight in kg
    allergies: List[str] = field(default_factory=list)  # List of allergies
    conditions: List[str] = field(default_factory=list)  # List of medical conditions
    
    # Treatment-related information
    diagnosis: Optional[str] = None  # Diagnosis/reason for treatment
    diagnosis_date: Optional[Union[date, str]] = None  # Date of diagnosis
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional information
    
    def __post_init__(self):
        """Process values after initialization."""
        # Ensure patient_id is a string
        self.patient_id = str(self.patient_id)
        
        # Convert date strings to date objects if needed
        if isinstance(self.birth_date, str):
            try:
                self.birth_date = datetime.fromisoformat(self.birth_date).date()
            except ValueError:
                pass  # Keep as string if invalid format
        
        if isinstance(self.diagnosis_date, str):
            try:
                self.diagnosis_date = datetime.fromisoformat(self.diagnosis_date).date()
            except ValueError:
                pass  # Keep as string if invalid format
    
    def get_age(self) -> Optional[int]:
        """
        Calculate the patient's age.
        
        Returns:
            Optional[int]: Age in years, or None if birth date is not set
        """
        if not self.birth_date:
            return None
        
        if isinstance(self.birth_date, str):
            return None  # Can't calculate from string
        
        today = date.today()
        age = today.year - self.birth_date.year
        
        # Adjust age if birthday hasn't occurred yet this year
        if (today.month, today.day) < (self.birth_date.month, self.birth_date.day):
            age -= 1
            
        return age
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation of patient
        """
        data = super().to_dict()
        
        # Ensure dates are converted to strings
        if isinstance(data.get('birth_date'), date):
            data['birth_date'] = data['birth_date'].isoformat()
        
        if isinstance(data.get('diagnosis_date'), date):
            data['diagnosis_date'] = data['diagnosis_date'].isoformat()
            
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Patient':
        """
        Create from dictionary.
        
        Args:
            data: Dictionary with patient data
            
        Returns:
            Patient: New patient object
        """
        # Create a copy of the data to avoid modifying the original
        data_copy = data.copy()
        
        # Handle nested structures
        if 'allergies' in data_copy and not isinstance(data_copy['allergies'], list):
            data_copy['allergies'] = []
        
        if 'conditions' in data_copy and not isinstance(data_copy['conditions'], list):
            data_copy['conditions'] = []
            
        if 'metadata' in data_copy and not isinstance(data_copy['metadata'], dict):
            data_copy['metadata'] = {}
            
        # Create patient object
        return cls(**data_copy)

