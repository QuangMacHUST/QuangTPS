"""
Treatment Plan Model

Contains the core Plan class and related functionality.
"""

from typing import Dict, List, Optional, Any, Tuple, Union
import numpy as np
from datetime import datetime

# Forward reference imports to avoid circular imports
from quangtps.core.structures import Structure
from quangtps.core.beams import Beam
from quangtps.core.prescriptions import Prescription

# Import DVHData type but not the actual implementation
# We'll import it inside the methods where it's needed
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from quangtps.evaluation.dvh.dvh_data import DVHData

class Plan:
    """
    Class representing a treatment plan.
    
    A plan consists of a prescription, a set of beams, and reference to
    a set of structures.
    
    Attributes:
        id: Unique identifier for the plan
        name: Name of the plan
        patient_id: ID of the patient this plan belongs to
        prescription: Prescription for the plan
        beams: List of beams in the plan
        structures: Dictionary of structures referenced by the plan
        date_created: Date the plan was created
        date_modified: Date the plan was last modified
    """
    
    def __init__(self, id: str, name: str, patient_id: str):
        """
        Initialize a plan.
        
        Args:
            id: Unique identifier for the plan
            name: Name of the plan
            patient_id: ID of the patient this plan belongs to
        """
        self.id = id
        self.name = name
        self.patient_id = patient_id
        
        # Core plan components
        self.prescription = None
        self.beams: List[Beam] = []
        self.structures: Dict[str, Structure] = {}
        
        # Metadata
        self.date_created = datetime.now()
        self.date_modified = datetime.now()
        
        # Optimization settings
        self.optimization_settings = {}
        
        # Calculation results
        self.dose_grid = None
        self.dvh_data = {}
        
        # Status
        self.is_calculated = False
        self.calculation_progress = 0.0
    
    def add_structure(self, structure: Structure):
        """
        Add a structure to the plan.
        
        Args:
            structure: The structure to add
        """
        self.structures[structure.id] = structure
    
    def get_structure(self, structure_id: str) -> Optional[Structure]:
        """
        Get a structure by ID.
        
        Args:
            structure_id: ID of the structure to get
            
        Returns:
            The structure if found, None otherwise
        """
        return self.structures.get(structure_id)
    
    def get_structures(self) -> List[Structure]:
        """
        Get all structures in the plan.
        
        Returns:
            List of all structures
        """
        return list(self.structures.values())
    
    def add_beam(self, beam: Beam):
        """
        Add a beam to the plan.
        
        Args:
            beam: The beam to add
        """
        self.beams.append(beam)
    
    def remove_beam(self, beam_id: str) -> bool:
        """
        Remove a beam from the plan.
        
        Args:
            beam_id: ID of the beam to remove
            
        Returns:
            True if the beam was removed, False if not found
        """
        for i, beam in enumerate(self.beams):
            if beam.id == beam_id:
                del self.beams[i]
                return True
        return False
    
    def get_beam(self, beam_id: str) -> Optional[Beam]:
        """
        Get a beam by ID.
        
        Args:
            beam_id: ID of the beam to get
            
        Returns:
            The beam if found, None otherwise
        """
        for beam in self.beams:
            if beam.id == beam_id:
                return beam
        return None
    
    def get_beams(self) -> List[Beam]:
        """
        Get all beams in the plan.
        
        Returns:
            List of all beams
        """
        return self.beams
    
    def set_prescription(self, prescription: Prescription):
        """
        Set the prescription for the plan.
        
        Args:
            prescription: The prescription to set
        """
        self.prescription = prescription
    
    def get_dvh_data(self, structure_id: str) -> Optional['DVHData']:
        """
        Get DVH data for a specific structure.
        
        Args:
            structure_id: ID of the structure
            
        Returns:
            DVH data object or None if not available
        """
        # In a real implementation, this would retrieve pre-calculated DVH data
        # or calculate it on-the-fly
        
        # For demonstration, create some sample DVH data
        structure = self.get_structure(structure_id)
        if not structure:
            return None
        
        # Import here to avoid circular imports
        from quangtps.evaluation.dvh.dvh_data import DVHData, DVHCurve
        
        # Create dummy DVH data
        if "PTV" in structure.name:
            # For PTV, high coverage curve
            dose_bins = np.linspace(0, 70, 100)
            volume_bins = np.zeros_like(dose_bins)
            
            # Calculate volume for each dose bin based on prescription
            rx_dose = self.prescription.dose
            for i, dose in enumerate(dose_bins):
                if dose <= 0.95 * rx_dose:
                    volume_bins[i] = 100  # 100% volume below 95% of rx
                elif dose <= rx_dose:
                    # Linear falloff from 100% to 95%
                    ratio = (dose - 0.95 * rx_dose) / (0.05 * rx_dose)
                    volume_bins[i] = 100 - 5 * ratio
                elif dose <= 1.05 * rx_dose:
                    # Linear falloff from 95% to 0%
                    ratio = (dose - rx_dose) / (0.05 * rx_dose)
                    volume_bins[i] = 95 - 95 * ratio
                else:
                    volume_bins[i] = 0  # No volume above 105% of rx
                    
            max_dose = 1.05 * rx_dose
            mean_dose = 0.98 * rx_dose
            min_dose = 0.95 * rx_dose
            
        elif any(oar in structure.name for oar in ["Lung", "Heart", "Liver", "Kidney", "Spinal", "Brain", "Cord"]):
            # For OAR, typical sparing curve
            dose_bins = np.linspace(0, 70, 100)
            volume_bins = np.zeros_like(dose_bins)
            
            # Calculate volume for each dose bin
            rx_dose = self.prescription.dose
            for i, dose in enumerate(dose_bins):
                if dose <= 0.1 * rx_dose:
                    volume_bins[i] = 100  # 100% volume at low dose
                else:
                    # Exponential falloff
                    volume_bins[i] = 100 * np.exp(-3 * dose / rx_dose)
                    
            max_dose = 0.7 * rx_dose
            mean_dose = 0.25 * rx_dose
            min_dose = 0
            
        else:
            # For other structures, generic curve
            dose_bins = np.linspace(0, 70, 100)
            volume_bins = np.zeros_like(dose_bins)
            
            # Calculate volume for each dose bin
            rx_dose = self.prescription.dose
            for i, dose in enumerate(dose_bins):
                if dose <= 0.5 * rx_dose:
                    volume_bins[i] = 100  # 100% volume at low dose
                else:
                    # Linear falloff
                    volume_bins[i] = max(0, 100 * (1 - dose / rx_dose))
                    
            max_dose = 0.9 * rx_dose
            mean_dose = 0.5 * rx_dose
            min_dose = 0
        
        # Create cumulative DVH curve
        cumulative = DVHCurve(
            dose_bins=dose_bins,
            volume_bins=volume_bins,
            is_cumulative=True
        )
        
        # Create differential DVH curve (simple derivative of cumulative)
        diff_volume = np.zeros_like(volume_bins)
        diff_volume[1:] = np.diff(volume_bins)
        diff_volume[0] = volume_bins[0]
        
        differential = DVHCurve(
            dose_bins=dose_bins,
            volume_bins=diff_volume,
            is_cumulative=False
        )
        
        # Create DVH data object
        dvh_data = DVHData(
            structure_id=structure_id,
            structure_name=structure.name,
            structure_volume=structure.volume,
            cumulative=cumulative,
            differential=differential,
            max_dose=max_dose,
            mean_dose=mean_dose,
            min_dose=min_dose
        )
        
        return dvh_data
    
    def get_dose(self) -> Optional[np.ndarray]:
        """
        Get the 3D dose array for this plan.
        
        Returns:
            3D numpy array of dose values or None if not available
        """
        # In a real implementation, this would return the actual dose grid
        # For demonstration, return a simple 3D array
        
        if not hasattr(self, '_dose_grid'):
            # Create a sample dose grid
            # Shape: 100x100x50 (x, y, z)
            shape = (100, 100, 50)
            self._dose_grid = np.zeros(shape)
            
            # Create a simple dose distribution - highest in the center
            center = np.array([shape[0]/2, shape[1]/2, shape[2]/2])
            rx_dose = self.prescription.dose
            
            # Generate dose for each voxel
            for x in range(shape[0]):
                for y in range(shape[1]):
                    for z in range(shape[2]):
                        # Distance from center
                        dist = np.sqrt((x - center[0])**2 + 
                                      (y - center[1])**2 + 
                                      (z - center[2])**2)
                        
                        # Max distance for normalization
                        max_dist = np.sqrt(center[0]**2 + center[1]**2 + center[2]**2)
                        
                        # Dose falls off with distance from center
                        self._dose_grid[x, y, z] = rx_dose * (1 - 0.9 * (dist / max_dist))
        
        return self._dose_grid 