"""
Base module for dose calculation algorithms.

This module provides abstract base classes for all dose calculation
algorithms in the QuangTPS system, ensuring a consistent interface.
"""

import abc
import SimpleITK as sitk
from typing import Dict, List, Tuple, Optional, Union, Any
import time
import logging
from dataclasses import dataclass

from ..core.types import DoseGrid, BeamParameters

logger = logging.getLogger(__name__)


@dataclass
class DoseCalculationResult:
    """
    Result of a dose calculation.
    
    This class stores the dose distribution and associated metadata
    from a dose calculation.
    """
    
    dose: sitk.Image  # Dose distribution as a SimpleITK image
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
        
    @abc.abstractmethod
    def calculate(self, ct_image: sitk.Image, structures: Dict[str, sitk.Image],
                 beam_parameters: BeamParameters) -> DoseCalculationResult:
        """
        Calculate dose distribution.
        
        Args:
            ct_image: CT image used for material and density information
            structures: Dictionary of structure masks (target, OARs)
            beam_parameters: Parameters describing the beam setup
            
        Returns:
            Calculated dose distribution and additional information
        """
        pass
    
    def get_name(self) -> str:
        """
        Get algorithm name.
        
        Returns:
            Algorithm name
        """
        return self.name
    
    def get_version(self) -> str:
        """
        Get algorithm version.
        
        Returns:
            Algorithm version
        """
        return self.version
    
    def get_description(self) -> str:
        """
        Get algorithm description.
        
        Returns:
            Description of the algorithm
        """
        return "Base dose calculation algorithm"
    
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get algorithm parameters.
        
        Returns:
            Dictionary of parameter names and values
        """
        return {}
    
    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        Set algorithm parameters.
        
        Args:
            parameters: Dictionary of parameter names and values
        """
        pass
    
    def validate_input(self, ct_image: sitk.Image, structures: Dict[str, sitk.Image],
                      beam_parameters: BeamParameters) -> bool:
        """
        Validate input data for dose calculation.
        
        Args:
            ct_image: CT image
            structures: Structure masks
            beam_parameters: Beam parameters
            
        Returns:
            True if input is valid, False otherwise
        """
        # Check if CT image is valid
        if ct_image is None:
            logger.error("CT image is None")
            return False
        
        if ct_image.GetDimension() != 3:
            logger.error(f"CT image has wrong dimension: {ct_image.GetDimension()}")
            return False
        
        # Check if structures are valid
        if structures is None:
            logger.warning("Structures dictionary is None")
        else:
            for name, structure in structures.items():
                if structure is None:
                    logger.warning(f"Structure {name} is None")
                    continue
                
                if structure.GetDimension() != 3:
                    logger.error(f"Structure {name} has wrong dimension: {structure.GetDimension()}")
                    return False
                
                # Check if structure has same size as CT
                if structure.GetSize() != ct_image.GetSize():
                    logger.error(f"Structure {name} has different size than CT image")
                    return False
        
        # Check if beam parameters are valid
        if beam_parameters is None:
            logger.error("Beam parameters are None")
            return False
        
        if beam_parameters.isocenter is None:
            logger.error("Beam isocenter is None")
            return False
        
        if beam_parameters.nominal_energy is None or beam_parameters.nominal_energy <= 0:
            logger.error(f"Invalid beam energy: {beam_parameters.nominal_energy}")
            return False
        
        return True
