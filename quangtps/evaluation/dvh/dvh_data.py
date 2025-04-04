import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple

from quangtps.structures.structure import Structure
from quangtps.dose.dose_distribution import DoseDistribution
from quangtps.structures.structure_set import StructureSet

logger = logging.getLogger(__name__)

class DVHCurve:
    """
    Represents a single DVH curve for a structure.
    Contains dose and volume data, along with metadata about the curve.
    """
    
    def __init__(self, structure_id: str, structure_name: str = ""):
        self.structure_id = structure_id
        self.structure_name = structure_name
        
        # Actual DVH data
        self.dose_data = np.array([])  # Dose values in Gy
        self.volume_data = np.array([])  # Volume values in percentage (0-100)
        
        # Metadata
        self.is_cumulative = True  # Whether the curve is cumulative (True) or differential (False)
        self.total_volume = 0.0  # Total volume of the structure in cc
        self.min_dose = 0.0  # Minimum dose in Gy
        self.max_dose = 0.0  # Maximum dose in Gy
        self.mean_dose = 0.0  # Mean dose in Gy
        self.median_dose = 0.0  # Median dose (D50) in Gy
        self.std_dose = 0.0  # Standard deviation of dose in Gy
        
        # Dose coverage metrics
        self.d_metrics: Dict[float, float] = {}  # Dx values (e.g., D95 = dose to 95% of volume)
        self.v_metrics: Dict[float, float] = {}  # Vx values (e.g., V20 = volume receiving 20 Gy)
    
    def set_data(self, dose_data: np.ndarray, volume_data: np.ndarray, is_cumulative: bool = True):
        """
        Set the actual dose and volume data for the curve.
        
        Args:
            dose_data: Array of dose values in Gy
            volume_data: Array of volume values in percentage (0-100)
            is_cumulative: Whether the data represents a cumulative DVH (True) or differential DVH (False)
        """
        if len(dose_data) != len(volume_data):
            raise ValueError("Dose and volume arrays must have the same length")
        
        self.dose_data = np.array(dose_data)
        self.volume_data = np.array(volume_data)
        self.is_cumulative = is_cumulative
        
        # Calculate basic statistics if data is available
        if len(dose_data) > 0:
            self._calculate_basic_stats()
    
    def _calculate_basic_stats(self):
        """Calculate basic DVH statistics"""
        if len(self.dose_data) == 0:
            return
        
        # Basic dose statistics
        self.min_dose = np.min(self.dose_data)
        self.max_dose = np.max(self.dose_data)
        
        # For mean and median, we need to consider the volume weights
        # This is a simplified calculation - a proper implementation would
        # use the full dose distribution
        if self.is_cumulative:
            # For cumulative DVH, we first convert to differential
            diff_volume = np.zeros_like(self.volume_data)
            diff_volume[:-1] = np.diff(self.volume_data[::-1])[::-1]
            diff_volume[-1] = self.volume_data[-1]
            
            # Normalize weights
            weights = diff_volume / np.sum(diff_volume) if np.sum(diff_volume) > 0 else np.zeros_like(diff_volume)
            
            # Calculate weighted statistics
            self.mean_dose = np.sum(self.dose_data * weights)
            
            # Median dose (D50) - find dose at 50% volume
            if self.is_cumulative:
                # Find index where volume is closest to 50%
                idx = np.argmin(np.abs(self.volume_data - 50.0))
                self.median_dose = self.dose_data[idx]
            
            # Standard deviation
            self.std_dose = np.sqrt(np.sum(weights * (self.dose_data - self.mean_dose)**2))
        else:
            # For differential DVH, we can use the raw data
            # Normalize weights
            weights = self.volume_data / np.sum(self.volume_data) if np.sum(self.volume_data) > 0 else np.zeros_like(self.volume_data)
            
            # Calculate weighted statistics
            self.mean_dose = np.sum(self.dose_data * weights)
            
            # Convert to cumulative for median
            cum_volume = np.cumsum(self.volume_data[::-1])[::-1]
            cum_volume = 100.0 * cum_volume / cum_volume[0] if cum_volume[0] > 0 else np.zeros_like(cum_volume)
            
            # Find index where cumulative volume is closest to 50%
            idx = np.argmin(np.abs(cum_volume - 50.0))
            self.median_dose = self.dose_data[idx]
            
            # Standard deviation
            self.std_dose = np.sqrt(np.sum(weights * (self.dose_data - self.mean_dose)**2))
    
    def calculate_d_metric(self, volume_percent: float) -> float:
        """
        Calculate Dx - the dose received by x% of the volume.
        
        Args:
            volume_percent: The volume percentage (0-100)
            
        Returns:
            The dose in Gy
        """
        if len(self.dose_data) == 0 or len(self.volume_data) == 0:
            return 0.0
        
        # Ensure we have a cumulative DVH
        if not self.is_cumulative:
            # Convert differential to cumulative
            cum_volume = np.cumsum(self.volume_data[::-1])[::-1]
            cum_volume = 100.0 * cum_volume / cum_volume[0] if cum_volume[0] > 0 else np.zeros_like(cum_volume)
            dose_values = self.dose_data
        else:
            cum_volume = self.volume_data
            dose_values = self.dose_data
        
        # Find index where volume is closest to requested percentage
        idx = np.argmin(np.abs(cum_volume - volume_percent))
        
        # Store result in cache
        self.d_metrics[volume_percent] = dose_values[idx]
        
        return dose_values[idx]
    
    def calculate_v_metric(self, dose: float) -> float:
        """
        Calculate Vx - the volume (in %) receiving at least x Gy.
        
        Args:
            dose: The dose threshold in Gy
            
        Returns:
            The volume percentage (0-100)
        """
        if len(self.dose_data) == 0 or len(self.volume_data) == 0:
            return 0.0
        
        # Ensure we have a cumulative DVH
        if not self.is_cumulative:
            # Convert differential to cumulative
            cum_volume = np.cumsum(self.volume_data[::-1])[::-1]
            cum_volume = 100.0 * cum_volume / cum_volume[0] if cum_volume[0] > 0 else np.zeros_like(cum_volume)
            dose_values = self.dose_data
        else:
            cum_volume = self.volume_data
            dose_values = self.dose_data
        
        # Find the volume at the given dose
        # Interpolate if necessary
        if dose <= np.min(dose_values):
            volume = 100.0  # Minimum dose covers 100% of volume
        elif dose >= np.max(dose_values):
            volume = 0.0  # No volume receives more than the maximum dose
        else:
            # Find indices where dose is just below and above the threshold
            idx_below = np.max(np.where(dose_values <= dose)[0])
            idx_above = np.min(np.where(dose_values >= dose)[0])
            
            if idx_below == idx_above:
                volume = cum_volume[idx_below]
            else:
                # Linear interpolation
                dose_below = dose_values[idx_below]
                dose_above = dose_values[idx_above]
                volume_below = cum_volume[idx_below]
                volume_above = cum_volume[idx_above]
                
                # Interpolate
                volume = volume_below + (volume_above - volume_below) * (dose - dose_below) / (dose_above - dose_below)
        
        # Store result in cache
        self.v_metrics[dose] = volume
        
        return volume


class DVHData:
    """
    Container for DVH curves for multiple structures.
    Includes methods for calculating and retrieving DVH metrics.
    """
    
    def __init__(self):
        self.curves: Dict[str, DVHCurve] = {}  # Map of structure_id to DVHCurve
        self.structures: Dict[str, Structure] = {}  # Map of structure_id to Structure
        
        # Plan-level information
        self.prescription_dose = 0.0  # Prescription dose in Gy
        self.plan_name = ""
        self.patient_id = ""
        self.date_calculated = ""
        
        # Calculation parameters
        self.bin_width = 0.1  # Dose bin width in Gy
        self.calculation_grid_size = 0.0  # Calculation grid size in mm
    
    def add_curve(self, curve: DVHCurve, structure: Optional[Structure] = None):
        """
        Add a DVH curve for a structure.
        
        Args:
            curve: The DVH curve to add
            structure: The corresponding structure object (optional)
        """
        self.curves[curve.structure_id] = curve
        
        if structure:
            self.structures[curve.structure_id] = structure
    
    def get_curve(self, structure_id: str) -> Optional[DVHCurve]:
        """Get the DVH curve for a structure by ID"""
        return self.curves.get(structure_id)
    
    def get_structure(self, structure_id: str) -> Optional[Structure]:
        """Get the structure by ID"""
        return self.structures.get(structure_id)
    
    def get_structure_ids(self) -> List[str]:
        """Get list of all structure IDs with DVH curves"""
        return list(self.curves.keys())
    
    def calculate_d_metric(self, structure_id: str, volume_percent: float) -> float:
        """
        Calculate Dx for a structure - the dose received by x% of the volume.
        
        Args:
            structure_id: The ID of the structure
            volume_percent: The volume percentage (0-100)
            
        Returns:
            The dose in Gy, or 0 if structure not found
        """
        curve = self.get_curve(structure_id)
        if not curve:
            return 0.0
        
        # Check if we've already calculated this metric
        if volume_percent in curve.d_metrics:
            return curve.d_metrics[volume_percent]
        
        # Calculate and return
        return curve.calculate_d_metric(volume_percent)
    
    def calculate_v_metric(self, structure_id: str, dose: float) -> float:
        """
        Calculate Vx for a structure - the volume (in %) receiving at least x Gy.
        
        Args:
            structure_id: The ID of the structure
            dose: The dose threshold in Gy
            
        Returns:
            The volume percentage (0-100), or 0 if structure not found
        """
        curve = self.get_curve(structure_id)
        if not curve:
            return 0.0
        
        # Check if we've already calculated this metric
        if dose in curve.v_metrics:
            return curve.v_metrics[dose]
        
        # Calculate and return
        return curve.calculate_v_metric(dose)
    
    def get_mean_dose(self, structure_id: str) -> float:
        """Get the mean dose for a structure in Gy"""
        curve = self.get_curve(structure_id)
        if not curve:
            return 0.0
        
        return curve.mean_dose
    
    def get_min_dose(self, structure_id: str) -> float:
        """Get the minimum dose for a structure in Gy"""
        curve = self.get_curve(structure_id)
        if not curve:
            return 0.0
        
        return curve.min_dose
    
    def get_max_dose(self, structure_id: str) -> float:
        """Get the maximum dose for a structure in Gy"""
        curve = self.get_curve(structure_id)
        if not curve:
            return 0.0
        
        return curve.max_dose
    
    def get_median_dose(self, structure_id: str) -> float:
        """Get the median dose (D50) for a structure in Gy"""
        return self.calculate_d_metric(structure_id, 50.0)
    
    def get_total_volume(self, structure_id: str) -> float:
        """Get the total volume of a structure in cc"""
        curve = self.get_curve(structure_id)
        if not curve:
            return 0.0
        
        return curve.total_volume
    
    def calculate_homogeneity_index(self, target_id: str) -> float:
        """
        Calculate the Homogeneity Index (HI) for a target structure.
        HI = (D2 - D98) / D50
        
        Args:
            target_id: The ID of the target structure
            
        Returns:
            Homogeneity Index value, or 0 if calculation fails
        """
        d2 = self.calculate_d_metric(target_id, 2.0)
        d98 = self.calculate_d_metric(target_id, 98.0)
        d50 = self.calculate_d_metric(target_id, 50.0)
        
        if d50 == 0:
            return 0.0
        
        return (d2 - d98) / d50
    
    def calculate_conformity_index(self, target_id: str, reference_dose: float) -> float:
        """
        Calculate the Conformity Index (CI) for a target structure.
        CI = (V_ref / V_target) * (V_ref / V_body_ref)
        
        Args:
            target_id: The ID of the target structure
            reference_dose: The reference isodose value in Gy
            
        Returns:
            Conformity Index value, or 0 if calculation fails
        """
        # This is a simplified version - a full implementation would need body contour
        # and would calculate the V_body_ref (volume of the reference isodose in the body)
        target_volume = self.get_total_volume(target_id)
        if target_volume == 0:
            return 0.0
        
        # Calculate volume of target receiving at least reference_dose
        v_ref_percent = self.calculate_v_metric(target_id, reference_dose)
        v_ref = target_volume * v_ref_percent / 100.0
        
        # Simplified CI (assuming V_ref = V_body_ref, which is not generally true)
        return (v_ref / target_volume)
    
    @staticmethod
    def from_structure_set_and_dose(
        structure_set: StructureSet, 
        dose: DoseDistribution,
        dose_bin_width: float = 0.1
    ) -> 'DVHData':
        """
        Create a DVHData object from a structure set and dose distribution.
        This is a static factory method that creates and populates a DVHData object.
        
        Args:
            structure_set: The structure set
            dose: The dose distribution
            dose_bin_width: The dose bin width in Gy
            
        Returns:
            A populated DVHData object
        """
        # Create a new DVHData object
        dvh_data = DVHData()
        dvh_data.bin_width = dose_bin_width
        
        # Set plan-level information if available
        if hasattr(dose, 'plan_name'):
            dvh_data.plan_name = dose.plan_name
        
        if hasattr(dose, 'prescription_dose'):
            dvh_data.prescription_dose = dose.prescription_dose
        
        # Set calculation grid size if available
        if hasattr(dose, 'grid_size'):
            dvh_data.calculation_grid_size = dose.grid_size
        
        # In a real implementation, this would calculate actual DVH curves
        # from the dose grid and structure masks
        # For this example, we'll create some synthetic data
        
        for structure in structure_set.structures:
            # Create a new curve
            curve = DVHCurve(structure.id, structure.name)
            
            # Set the volume
            curve.total_volume = structure.get_volume()
            
            # Create synthetic dose and volume data
            # In a real implementation, this would be calculated from actual dose and structure
            curve = DVHData._create_synthetic_dvh_curve(structure, dose, curve)
            
            # Add curve to DVHData
            dvh_data.add_curve(curve, structure)
        
        return dvh_data
    
    @staticmethod
    def _create_synthetic_dvh_curve(
        structure: Structure, 
        dose: DoseDistribution,
        curve: DVHCurve
    ) -> DVHCurve:
        """
        Create a synthetic DVH curve for testing purposes.
        In a real implementation, this would be calculated from the dose grid and structure mask.
        
        Args:
            structure: The structure object
            dose: The dose distribution
            curve: The DVH curve object to populate
            
        Returns:
            The populated DVH curve
        """
        # Create synthetic dose and volume data based on structure type
        min_dose = 0.0
        max_dose = 80.0
        
        # Number of points in the DVH curve
        num_points = 100
        
        # Create dose array
        dose_data = np.linspace(min_dose, max_dose, num_points)
        
        # Create volume array based on structure type
        if "PTV" in structure.name or "CTV" in structure.name or "GTV" in structure.name:
            # Target structures have high coverage
            d50 = 60.0  # Median dose
            d98 = 57.0  # Dose to 98% of volume
            
            # Create sigmoidal curve for cumulative DVH
            volume_data = 100.0 / (1.0 + np.exp((dose_data - d50) / (d98 - d50) * 10.0))
        elif any(oar in structure.name for oar in ["Lung", "Heart", "Liver", "Kidney", "Spinal", "Brain", "Cord"]):
            # OARs have lower dose
            d50 = 20.0  # Median dose
            d90 = 10.0  # Dose to 90% of volume
            
            # Create sigmoidal curve for cumulative DVH
            volume_data = 100.0 / (1.0 + np.exp((dose_data - d50) / (d90 - d50) * 5.0))
        else:
            # Other structures have variable dose
            d50 = 30.0  # Median dose
            d90 = 15.0  # Dose to 90% of volume
            
            # Create sigmoidal curve for cumulative DVH
            volume_data = 100.0 / (1.0 + np.exp((dose_data - d50) / (d90 - d50) * 3.0))
        
        # Set the data in the curve
        curve.set_data(dose_data, volume_data, is_cumulative=True)
        
        return curve 