"""
DVH Data

This module provides the DVHData class for storing and manipulating
Dose-Volume Histogram (DVH) data for radiotherapy structure evaluation.
"""

import numpy as np
from typing import List, Optional, Dict, Any, Tuple
import json
import logging

from quangtps.core.logging import get_logger

logger = get_logger(__name__)

class DVHData:
    """
    Class for storing and manipulating DVH data for a structure.
    
    Attributes:
        structure_id (str): Unique identifier for the structure
        dose_bins (List[float]): Dose bin values, typically in Gy
        cumulative_volume (List[float]): Cumulative volume at each dose bin
        differential_volume (List[float]): Differential volume at each dose bin
        total_volume (float): Total volume of the structure in cc
        min_dose (float): Minimum dose to the structure
        max_dose (float): Maximum dose to the structure
        mean_dose (float): Mean dose to the structure
        median_dose (float): Median dose (D50) to the structure
        dose_unit (str): Unit for dose values ("Gy" or "%")
        volume_unit (str): Unit for volume values ("cc" or "%")
    """
    
    def __init__(self, structure_id: str):
        """
        Initialize a new DVH data object.
        
        Args:
            structure_id: Unique identifier for the structure
        """
        self.structure_id = structure_id
        self.dose_bins: List[float] = []
        self.cumulative_volume: List[float] = []
        self.differential_volume: List[float] = []
        self.total_volume: float = 0.0
        self.min_dose: float = 0.0
        self.max_dose: float = 0.0
        self.mean_dose: float = 0.0
        self.median_dose: float = 0.0
        self.dose_unit: str = "Gy"
        self.volume_unit: str = "%"
    
    @classmethod
    def from_raw_data(cls, structure_id: str, dose_bins: List[float], 
                     cumulative_volume: List[float], total_volume: float) -> 'DVHData':
        """
        Create a DVH data object from raw data arrays.
        
        Args:
            structure_id: Unique identifier for the structure
            dose_bins: Array of dose bin values (typically in Gy)
            cumulative_volume: Array of cumulative volume values at each dose bin
            total_volume: Total volume of the structure in cc
            
        Returns:
            A new DVHData object
        """
        dvh = cls(structure_id)
        
        # Validate inputs
        if len(dose_bins) != len(cumulative_volume):
            raise ValueError("Dose bins and cumulative volume arrays must have the same length")
        
        # Ensure all arrays are numpy arrays
        dose_bins_np = np.array(dose_bins, dtype=float)
        cumulative_volume_np = np.array(cumulative_volume, dtype=float)
        
        # Store raw data
        dvh.dose_bins = dose_bins_np.tolist()
        dvh.cumulative_volume = cumulative_volume_np.tolist()
        dvh.total_volume = float(total_volume) if total_volume else 0.0
        
        # Calculate differential volume
        if len(dose_bins_np) > 1:
            # Calculate differential volume by taking the negative derivative of the cumulative DVH
            diff_vol = -np.diff(cumulative_volume_np)
            # Append a zero at the end to match length of dose_bins
            diff_vol = np.append(diff_vol, 0.0)
            dvh.differential_volume = diff_vol.tolist()
        else:
            dvh.differential_volume = [0.0]
        
        # Calculate statistics
        if len(dose_bins_np) > 0 and len(cumulative_volume_np) > 0:
            dvh.min_dose = float(dose_bins_np[0])
            dvh.max_dose = float(dose_bins_np[-1]) if len(dose_bins_np) > 1 else dvh.min_dose
            
            # Calculate mean dose
            if len(dvh.differential_volume) > 1:
                diff_vol_np = np.array(dvh.differential_volume)
                dvh.mean_dose = float(np.sum(dose_bins_np * diff_vol_np) / np.sum(diff_vol_np)) if np.sum(diff_vol_np) > 0 else 0.0
            
            # Calculate median dose (D50)
            dvh.median_dose = dvh.get_dose_at_volume(50.0) if dvh.total_volume > 0 else 0.0
        
        # Set units
        dvh.dose_unit = "Gy"
        dvh.volume_unit = "%"  # Default to percentage
        
        return dvh
    
    def get_dose_at_volume(self, volume_percent: float) -> float:
        """
        Get the dose value at a given volume percentage (Dx).
        
        Args:
            volume_percent: Volume percentage (0-100)
            
        Returns:
            Dose value at the specified volume percentage
        """
        if not self.dose_bins or not self.cumulative_volume:
            return 0.0
            
        # Ensure volume is in the range [0, 100]
        volume_percent = max(0.0, min(100.0, volume_percent))
        
        # Convert volume percent to the appropriate unit if necessary
        if self.volume_unit == "cc" and self.total_volume > 0:
            volume_val = volume_percent * self.total_volume / 100.0
        else:
            volume_val = volume_percent
        
        # Interpolate to find dose at the given volume
        try:
            cum_vol_np = np.array(self.cumulative_volume)
            dose_bins_np = np.array(self.dose_bins)
            
            # For Dx, we need to find where cumulative volume = x%
            idx = np.where(cum_vol_np <= volume_val)[0]
            
            if len(idx) == 0:
                # If no values are less than volume_val, return minimum dose
                return self.min_dose
            
            if len(idx) == len(cum_vol_np):
                # If all values are less than volume_val, return maximum dose
                return self.max_dose
            
            # Find bounding indices
            i_lower = idx[-1]
            i_upper = i_lower + 1 if i_lower < len(cum_vol_np) - 1 else i_lower
            
            # If we're already exactly at the volume value, return the dose
            if cum_vol_np[i_lower] == volume_val:
                return float(dose_bins_np[i_lower])
            
            # Interpolate between bounding indices
            v_lower = cum_vol_np[i_lower]
            v_upper = cum_vol_np[i_upper]
            d_lower = dose_bins_np[i_lower]
            d_upper = dose_bins_np[i_upper]
            
            # Linear interpolation
            if v_upper != v_lower:
                dose = d_lower + (d_upper - d_lower) * (volume_val - v_lower) / (v_upper - v_lower)
            else:
                dose = d_lower
                
            return float(dose)
            
        except Exception as e:
            logger.error(f"Error calculating dose at volume {volume_percent}%: {str(e)}")
            return 0.0
    
    def get_volume_at_dose(self, dose: float) -> float:
        """
        Get the volume percentage receiving at least the given dose (Vx).
        
        Args:
            dose: Dose value (in the unit specified by dose_unit)
            
        Returns:
            Volume percentage receiving at least the specified dose
        """
        if not self.dose_bins or not self.cumulative_volume:
            return 0.0
        
        # Handle out-of-range doses
        if dose <= self.min_dose:
            return 100.0
        if dose > self.max_dose:
            return 0.0
        
        # Interpolate to find volume at the given dose
        try:
            cum_vol_np = np.array(self.cumulative_volume)
            dose_bins_np = np.array(self.dose_bins)
            
            # For Vx, we need to find the cumulative volume at dose x
            idx = np.where(dose_bins_np <= dose)[0]
            
            if len(idx) == 0:
                # If no values are less than dose, return 100%
                return 100.0
            
            if len(idx) == len(dose_bins_np):
                # If all values are less than dose, return 0%
                return 0.0
            
            # Find bounding indices
            i_lower = idx[-1]
            i_upper = i_lower + 1 if i_lower < len(dose_bins_np) - 1 else i_lower
            
            # If we're already exactly at the dose value, return the volume
            if dose_bins_np[i_lower] == dose:
                return float(cum_vol_np[i_lower])
            
            # Interpolate between bounding indices
            d_lower = dose_bins_np[i_lower]
            d_upper = dose_bins_np[i_upper]
            v_lower = cum_vol_np[i_lower]
            v_upper = cum_vol_np[i_upper]
            
            # Linear interpolation
            if d_upper != d_lower:
                volume = v_lower + (v_upper - v_lower) * (dose - d_lower) / (d_upper - d_lower)
            else:
                volume = v_lower
                
            return float(volume)
            
        except Exception as e:
            logger.error(f"Error calculating volume at dose {dose} Gy: {str(e)}")
            return 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert DVH data to a dictionary for serialization.
        
        Returns:
            Dictionary representation of the DVH data
        """
        return {
            'structure_id': self.structure_id,
            'dose_bins': self.dose_bins,
            'cumulative_volume': self.cumulative_volume,
            'differential_volume': self.differential_volume,
            'total_volume': self.total_volume,
            'min_dose': self.min_dose,
            'max_dose': self.max_dose,
            'mean_dose': self.mean_dose,
            'median_dose': self.median_dose,
            'dose_unit': self.dose_unit,
            'volume_unit': self.volume_unit
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DVHData':
        """
        Create a DVH data object from a dictionary.
        
        Args:
            data: Dictionary containing DVH data
            
        Returns:
            A new DVHData object
        """
        dvh = cls(data['structure_id'])
        
        dvh.dose_bins = data.get('dose_bins', [])
        dvh.cumulative_volume = data.get('cumulative_volume', [])
        dvh.differential_volume = data.get('differential_volume', [])
        dvh.total_volume = data.get('total_volume', 0.0)
        dvh.min_dose = data.get('min_dose', 0.0)
        dvh.max_dose = data.get('max_dose', 0.0)
        dvh.mean_dose = data.get('mean_dose', 0.0)
        dvh.median_dose = data.get('median_dose', 0.0)
        dvh.dose_unit = data.get('dose_unit', 'Gy')
        dvh.volume_unit = data.get('volume_unit', '%')
        
        return dvh
    
    def to_json(self) -> str:
        """
        Convert DVH data to a JSON string.
        
        Returns:
            JSON string representation of the DVH data
        """
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str: str) -> 'DVHData':
        """
        Create a DVH data object from a JSON string.
        
        Args:
            json_str: JSON string containing DVH data
            
        Returns:
            A new DVHData object
        """
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def resample(self, bin_count: int = 100) -> 'DVHData':
        """
        Resample the DVH data to a specific number of dose bins.
        
        Args:
            bin_count: Number of dose bins in the resampled DVH
            
        Returns:
            A new DVHData object with resampled data
        """
        if not self.dose_bins or bin_count <= 0:
            return self
        
        # Create new dose bins
        new_dose_bins = np.linspace(self.min_dose, self.max_dose, bin_count)
        
        # Interpolate cumulative volume
        cum_vol_interp = np.interp(
            new_dose_bins,
            self.dose_bins,
            self.cumulative_volume
        )
        
        # Create resampled DVH
        resampled_dvh = DVHData.from_raw_data(
            self.structure_id,
            new_dose_bins.tolist(),
            cum_vol_interp.tolist(),
            self.total_volume
        )
        
        # Copy over units
        resampled_dvh.dose_unit = self.dose_unit
        resampled_dvh.volume_unit = self.volume_unit
        
            dose_bins: Array of dose values in Gy
            volume_bins: Array of volume values in % (0-100)
            is_cumulative: Whether this is a cumulative curve
        """
        self.dose_bins = dose_bins
        self.volume_bins = volume_bins
        self.is_cumulative = is_cumulative
        
        # Ensure arrays are the same length
        if len(dose_bins) != len(volume_bins):
            raise ValueError("Dose and volume arrays must be the same length")


class DVHData:
    """
    Class representing DVH data for a structure.
    
    Attributes:
        structure_id: Identifier for the structure
        structure_name: Name of the structure
        structure_volume: Volume of the structure in cc
        cumulative: Cumulative DVH curve
        differential: Differential DVH curve
        max_dose: Maximum dose to the structure in Gy
        mean_dose: Mean dose to the structure in Gy
        min_dose: Minimum dose to the structure in Gy
    """
    
    def __init__(self, 
                structure_id: str,
                structure_name: str,
                structure_volume: float,
                cumulative: DVHCurve,
                differential: DVHCurve,
                max_dose: float,
                mean_dose: float,
                min_dose: float):
        """
        Initialize DVH data.
        
        Args:
            structure_id: Identifier for the structure
            structure_name: Name of the structure
            structure_volume: Volume of the structure in cc
            cumulative: Cumulative DVH curve
            differential: Differential DVH curve
            max_dose: Maximum dose to the structure in Gy
            mean_dose: Mean dose to the structure in Gy
            min_dose: Minimum dose to the structure in Gy
        """
        self.structure_id = structure_id
        self.structure_name = structure_name
        self.structure_volume = structure_volume
        self.cumulative = cumulative
        self.differential = differential
        self.max_dose = max_dose
        self.mean_dose = mean_dose
        self.min_dose = min_dose
    
    def get_dose_at_volume(self, volume_percent: float) -> float:
        """
        Get the dose at a specified volume percentage.
        
        Args:
            volume_percent: Volume percentage (0-100)
            
        Returns:
            Dose in Gy at the specified volume
        """
        # Ensure we're using the cumulative curve
        dose_bins = self.cumulative.dose_bins
        volume_bins = self.cumulative.volume_bins
        
        # Find the index where volume <= volume_percent
        # For cumulative DVH, higher volume means lower dose
        for i in range(len(volume_bins)):
            if volume_bins[i] <= volume_percent:
                if i == 0:
                    return dose_bins[0]
                else:
                    # Linear interpolation between points
                    v1, v2 = volume_bins[i-1], volume_bins[i]
                    d1, d2 = dose_bins[i-1], dose_bins[i]
                    
                    # Handle case where volumes are the same
                    if v1 == v2:
                        return d1
                    
                    # Interpolate
                    ratio = (volume_percent - v1) / (v2 - v1)
                    return d1 + ratio * (d2 - d1)
        
        # If volume is lower than any in the curve, return max dose
        return dose_bins[-1]
    
    def get_volume_at_dose(self, dose: float) -> float:
        """
        Get the volume percentage at a specified dose.
        
        Args:
            dose: Dose in Gy
            
        Returns:
            Volume percentage (0-100) at the specified dose
        """
        # Ensure we're using the cumulative curve
        dose_bins = self.cumulative.dose_bins
        volume_bins = self.cumulative.volume_bins
        
        # Find the index where dose >= specified dose
        for i in range(len(dose_bins)):
            if dose_bins[i] >= dose:
                if i == 0:
                    return volume_bins[0]
                else:
                    # Linear interpolation between points
                    d1, d2 = dose_bins[i-1], dose_bins[i]
                    v1, v2 = volume_bins[i-1], volume_bins[i]
                    
                    # Handle case where doses are the same
                    if d1 == d2:
                        return v1
                    
                    # Interpolate
                    ratio = (dose - d1) / (d2 - d1)
                    return v1 + ratio * (v2 - v1)
        
        # If dose is higher than any in the curve, return 0 volume
        return 0.0


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