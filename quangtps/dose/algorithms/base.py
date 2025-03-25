"""
Base module for dose calculation algorithms.

This module provides abstract base classes for all dose calculation
algorithms in the QuangTPS system, ensuring a consistent interface.
"""

import abc
import SimpleITK as sitk
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import time
import logging
from dataclasses import dataclass

from quangtps.core.exceptions import DoseCalculationError, ValidationError
from quangtps.core.types import DoseGrid, BeamParameters
from quangtps.imaging.image import Image
from quangtps.planning.beam import Beam

logger = logging.getLogger(__name__)


@dataclass
class DoseCalculationResult:
    """
    Result of a dose calculation.
    
    This class stores the dose distribution and associated metadata
    from a dose calculation.
    """
    
    dose: Image  # Dose distribution as Image object
    algorithm_name: str  # Name of the algorithm used
    calculation_time: float  # Calculation time in seconds
    additional_data: Dict[str, Any] = None  # Additional data (e.g., uncertainty)
    
    def __post_init__(self):
        """Initialize additional data if not provided."""
        if self.additional_data is None:
            self.additional_data = {}


class DoseCalculationAlgorithm(abc.ABC):
    """
    Abstract base class for dose calculation algorithms.
    
    All dose calculation algorithms in QuangTPS should inherit from this
    class and implement its abstract methods.
    """
    
    def __init__(self, name: str):
        """
        Initialize the dose calculation algorithm.
        
        Args:
            name: Algorithm name
        """
        self.name = name
        self.version = "1.0"
        self.beam_model = None
        self.parameters = {}
        
    def validate_inputs(self, ct_image: Image, beam: Beam) -> None:
        """
        Validate inputs for dose calculation.
        
        Parameters
        ----------
        ct_image : Image
            CT image for dose calculation
        beam : Beam
            Treatment beam
            
        Raises
        ------
        ValidationError
            If inputs are invalid
        """
        # Validate CT image
        if ct_image is None:
            raise ValidationError("CT image cannot be None")
            
        if not isinstance(ct_image, Image):
            raise ValidationError(f"Expected Image object, got {type(ct_image)}")
            
        if not ct_image.data.any():
            raise ValidationError("CT image has no data")
            
        if len(ct_image.data.shape) != 3:
            raise ValidationError(f"CT image must be 3D, got shape {ct_image.data.shape}")
            
        # Check CT numbers are in valid range
        if np.min(ct_image.data) < -1024 or np.max(ct_image.data) > 3071:
            raise ValidationError("CT numbers out of valid range [-1024, 3071]")
            
        # Validate beam
        if beam is None:
            raise ValidationError("Beam cannot be None")
            
        if not isinstance(beam, Beam):
            raise ValidationError(f"Expected Beam object, got {type(beam)}")
            
        if not beam.isocenter:
            raise ValidationError("Beam has no isocenter")
            
        if not beam.gantry_angle and beam.gantry_angle != 0:
            raise ValidationError("Beam has no gantry angle")
            
        if not beam.field_size or any(s <= 0 for s in beam.field_size):
            raise ValidationError(f"Invalid field size: {beam.field_size}")
            
        # Validate beam model
        if self.beam_model is None:
            raise ValidationError("No beam model set for dose calculation")
    
    def set_beam_model(self, beam_model: Any) -> None:
        """
        Set beam model for dose calculation.
        
        Parameters
        ----------
        beam_model : Any
            Beam model containing beam data
        """
        self.beam_model = beam_model
        logger.info(f"Set beam model for {self.name} algorithm")
    
    def set_parameter(self, name: str, value: Any) -> None:
        """
        Set calculation parameter.
        
        Parameters
        ----------
        name : str
            Parameter name
        value : Any
            Parameter value
        """
        self.parameters[name] = value
        logger.debug(f"Set parameter {name}={value} for {self.name} algorithm")
    
    def get_parameter(self, name: str, default: Any = None) -> Any:
        """
        Get calculation parameter.
        
        Parameters
        ----------
        name : str
            Parameter name
        default : Any, optional
            Default value if parameter not found
            
        Returns
        -------
        Any
            Parameter value
        """
        return self.parameters.get(name, default)
    
    @abc.abstractmethod
    def calculate(self, ct_image: Image, beam: Beam) -> DoseCalculationResult:
        """
        Calculate dose distribution.
        
        Parameters
        ----------
        ct_image : Image
            CT image for dose calculation
        beam : Beam
            Treatment beam
            
        Returns
        -------
        DoseCalculationResult
            Calculated dose and metadata
            
        Raises
        ------
        DoseCalculationError
            If dose calculation fails
        ValidationError
            If inputs are invalid
        """
        pass
    
    def get_name(self) -> str:
        """Get algorithm name."""
        return self.name
    
    def get_version(self) -> str:
        """Get algorithm version."""
        return self.version
    
    def get_description(self) -> str:
        """Get algorithm description."""
        return "Base dose calculation algorithm"
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get all calculation parameters."""
        return self.parameters.copy()
    
    def _convert_ct_to_density(self, ct_image: Image) -> np.ndarray:
        """
        Convert CT numbers to electron density.
        
        Parameters
        ----------
        ct_image : Image
            CT image
            
        Returns
        -------
        np.ndarray
            Electron density map
            
        Raises
        ------
        ValueError
            If conversion fails
        """
        try:
            # Simple linear conversion (should be replaced with proper calibration curve)
            density = (ct_image.data + 1000) / 1000
            density = np.clip(density, 0, None)  # Ensure non-negative
            return density
            
        except Exception as e:
            raise ValueError(f"Failed to convert CT to density: {str(e)}")
    
    def _initialize_dose_grid(self, ct_image: Image) -> np.ndarray:
        """
        Initialize dose calculation grid.
        
        Parameters
        ----------
        ct_image : Image
            Reference CT image
            
        Returns
        -------
        np.ndarray
            Zero-initialized dose grid
            
        Raises
        ------
        ValueError
            If grid initialization fails
        """
        try:
            return np.zeros_like(ct_image.data)
            
        except Exception as e:
            raise ValueError(f"Failed to initialize dose grid: {str(e)}")
            
    def _validate_calculation_completed(self, dose: np.ndarray) -> None:
        """
        Validate calculated dose distribution.
        
        Parameters
        ----------
        dose : np.ndarray
            Calculated dose distribution
            
        Raises
        ------
        DoseCalculationError
            If dose is invalid
        """
        if dose is None:
            raise DoseCalculationError("Dose calculation returned None")
            
        if not dose.any():
            raise DoseCalculationError("Calculated dose is all zeros")
            
        if not np.isfinite(dose).all():
            raise DoseCalculationError("Dose contains invalid values (inf/nan)")
            
        if np.min(dose) < 0:
            raise DoseCalculationError("Dose contains negative values") 