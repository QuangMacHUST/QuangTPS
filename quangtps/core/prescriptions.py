"""
Prescriptions Module

Contains the Prescription class and related functionality.
"""

from typing import Dict, List, Optional, Any, Tuple
from enum import Enum


class PrescriptionType(Enum):
    """Enumeration of prescription types."""
    DOSE_TO_VOLUME = "DOSE_TO_VOLUME"  # Dose to a percentage of volume
    VOLUME_AT_DOSE = "VOLUME_AT_DOSE"  # Percentage of volume at a dose
    MEAN_DOSE = "MEAN_DOSE"  # Mean dose to the target
    MIN_DOSE = "MIN_DOSE"  # Minimum dose to the target
    MAX_DOSE = "MAX_DOSE"  # Maximum dose to the target


class Prescription:
    """
    Class representing a treatment prescription.
    
    Attributes:
        id: Unique identifier for the prescription
        target_id: ID of the target structure
        dose: Prescription dose in Gy
        fractions: Number of fractions
        prescription_type: Type of prescription (dose to volume, etc.)
        volume_percent: For DOSE_TO_VOLUME, the volume percentage
        dose_percent: For VOLUME_AT_DOSE, the dose percentage
    """
    
    def __init__(self, id: str, target_id: str, dose: float, fractions: int,
                prescription_type: PrescriptionType = PrescriptionType.DOSE_TO_VOLUME,
                volume_percent: float = 95.0,
                dose_percent: float = 100.0):
        """
        Initialize a prescription.
        
        Args:
            id: Unique identifier for the prescription
            target_id: ID of the target structure
            dose: Prescription dose in Gy
            fractions: Number of fractions
            prescription_type: Type of prescription
            volume_percent: For DOSE_TO_VOLUME, the volume percentage
            dose_percent: For VOLUME_AT_DOSE, the dose percentage
        """
        self.id = id
        self.target_id = target_id
        self.dose = dose
        self.fractions = fractions
        
        # Ensure prescription_type is an enum
        if isinstance(prescription_type, str):
            try:
                self.prescription_type = PrescriptionType(prescription_type)
            except ValueError:
                self.prescription_type = PrescriptionType.DOSE_TO_VOLUME
        else:
            self.prescription_type = prescription_type
        
        self.volume_percent = volume_percent
        self.dose_percent = dose_percent
    
    @property
    def dose_per_fraction(self) -> float:
        """
        Get the dose per fraction in Gy.
        
        Returns:
            Dose per fraction in Gy
        """
        if self.fractions > 0:
            return self.dose / self.fractions
        return 0.0
    
    def __str__(self) -> str:
        """
        Get a string representation of the prescription.
        
        Returns:
            String representation
        """
        if self.prescription_type == PrescriptionType.DOSE_TO_VOLUME:
            return f"{self.dose} Gy to {self.volume_percent}% of target in {self.fractions} fractions"
        elif self.prescription_type == PrescriptionType.VOLUME_AT_DOSE:
            return f"{self.volume_percent}% of target to receive {self.dose_percent}% of {self.dose} Gy in {self.fractions} fractions"
        elif self.prescription_type == PrescriptionType.MEAN_DOSE:
            return f"Mean dose of {self.dose} Gy to target in {self.fractions} fractions"
        elif self.prescription_type == PrescriptionType.MIN_DOSE:
            return f"Minimum dose of {self.dose} Gy to target in {self.fractions} fractions"
        elif self.prescription_type == PrescriptionType.MAX_DOSE:
            return f"Maximum dose of {self.dose} Gy to target in {self.fractions} fractions"
        else:
            return f"{self.dose} Gy in {self.fractions} fractions" 