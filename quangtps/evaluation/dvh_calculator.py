#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DVH Calculator for QuangTPS.

This module provides functionality for calculating dose-volume histograms
and related metrics for radiotherapy treatment plans.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any

from quangtps.dose.dose import Dose
from quangtps.structures.structure_set import StructureSet, Structure

# Define constants
DEFAULT_DOSE_BINS = 100  # Number of dose bins for DVH calculation
MIN_DOSE_STEP = 0.01  # Minimum dose step (Gy)

# Set up logging
logger = logging.getLogger(__name__)


class DVHPoint:
    """Simple class to represent a point on a DVH curve."""

    def __init__(self, dose: float, volume: float):
        """
        Initialize a DVH point.

        Parameters
        ----------
        dose : float
            Dose value in Gy
        volume : float
            Volume value in percentage (0-100)
        """
        self.dose = dose
        self.volume = volume

    def __str__(self) -> str:
        """String representation of the DVH point."""
        return f"DVH Point: {self.dose:.2f} Gy, {self.volume:.2f}%"


class DVHData:
    """
    Class to hold DVH data for a structure.

    Attributes
    ----------
    structure_name : str
        Name of the structure
    doses : List[float]
        List of dose values (Gy)
    volumes_cum : List[float]
        List of cumulative volume values (%)
    volumes_diff : List[float]
        List of differential volume values (%)
    metrics : Dict[str, float]
        Dictionary of DVH metrics
    """

    def __init__(self, structure_name: str = ""):
        """
        Initialize DVH data.

        Parameters
        ----------
        structure_name : str, optional
            Name of the structure, by default ""
        """
        self.structure_name = structure_name
        self.doses = []
        self.volumes_cum = []
        self.volumes_diff = []
        self.metrics = {}

    def add_point(self, dose: float, volume_cum: float, volume_diff: float):
        """
        Add a point to the DVH data.

        Parameters
        ----------
        dose : float
            Dose value in Gy
        volume_cum : float
            Cumulative volume value in percentage (0-100)
        volume_diff : float
            Differential volume value in percentage (0-100)
        """
        self.doses.append(dose)
        self.volumes_cum.append(volume_cum)
        self.volumes_diff.append(volume_diff)

    def get_metrics(self) -> Dict[str, float]:
        """
        Get DVH metrics.

        Returns
        -------
        Dict[str, float]
            Dictionary of DVH metrics
        """
        return self.metrics

    def calculate_metrics(self):
        """Calculate common DVH metrics."""
        # Ensure that we have data
        if not self.doses or not self.volumes_cum:
            return

        # Convert to numpy arrays for easier calculations
        doses = np.array(self.doses)
        volumes_cum = np.array(self.volumes_cum)

        # Calculate min, max, and mean dose
        # Min dose is the dose at which the cumulative volume is (nearly) 100%
        idx_min = np.argmax(volumes_cum >= 99.5)
        self.metrics["min_dose"] = doses[idx_min] if idx_min < len(doses) else doses[0]

        # Max dose is the highest dose
        self.metrics["max_dose"] = doses[-1] if doses.size > 0 else 0.0

        # Mean dose is the area under the differential DVH
        if len(self.doses) > 1 and len(self.volumes_diff) > 1:
            # Calculate the mean dose as the weighted average
            vol_fractions = np.array(self.volumes_diff) / 100.0  # Convert to fractions
            self.metrics["mean_dose"] = np.sum(doses * vol_fractions) / np.sum(
                vol_fractions
            )
        else:
            self.metrics["mean_dose"] = 0.0

        # Calculate dose to volume metrics (D95, D90, D50, D5)
        vol_points = [95, 90, 50, 5]
        dose_to_volume = {}

        for vol in vol_points:
            # Find the dose that covers vol% of the volume
            idx = np.argmin(np.abs(volumes_cum - vol))
            dose_to_volume[vol] = doses[idx] if idx < len(doses) else 0.0

        self.metrics["dose_to_volume"] = dose_to_volume

        # Calculate volume at dose metrics (V100%, V95%, V90%, etc.)
        # First, determine the prescription dose (or use max dose if not available)
        prescription_dose = self.metrics.get(
            "prescribed_dose", self.metrics["max_dose"]
        )

        dose_points = [100, 95, 90, 80, 50, 20]  # Percentage of prescription dose
        volume_at_dose = {}

        for dose_pct in dose_points:
            dose = prescription_dose * dose_pct / 100.0
            # Find the volume receiving at least this dose
            idx = np.argmin(np.abs(doses - dose))
            if idx < len(volumes_cum):
                volume_at_dose[dose] = volumes_cum[idx]
            else:
                volume_at_dose[dose] = 0.0

        self.metrics["volume_at_dose"] = volume_at_dose

        # Calculate heterogeneity index (HI) - higher values indicate more heterogeneous dose
        # HI = (D5 - D95) / Dmean
        if "mean_dose" in self.metrics and self.metrics["mean_dose"] > 0:
            d5 = dose_to_volume[5]
            d95 = dose_to_volume[95]
            hi = (d5 - d95) / self.metrics["mean_dose"]
            self.metrics["homogeneity_index"] = hi

        # Calculate conformity index (CI) - only if we have prescription dose
        # CI = V100% / Total volume (ideally 1.0)
        if "prescribed_dose" in self.metrics:
            prescribed_dose = self.metrics["prescribed_dose"]
            idx = np.argmin(np.abs(doses - prescribed_dose))
            if idx < len(volumes_cum):
                v100 = volumes_cum[idx]
                self.metrics["conformity_index"] = v100 / 100.0  # Normalize to 0-1
            else:
                self.metrics["conformity_index"] = 0.0


class DVHCalculator:
    """
    Calculator for dose-volume histograms.

    Calculates DVH data and metrics for radiotherapy treatment plans.
    """

    def __init__(self, num_bins: int = DEFAULT_DOSE_BINS):
        """
        Initialize the DVH calculator.

        Parameters
        ----------
        num_bins : int, optional
            Number of dose bins for DVH calculation, by default DEFAULT_DOSE_BINS
        """
        self.num_bins = num_bins

    def calculate_dvh(
        self, dose: Dose, structure_set: StructureSet
    ) -> Dict[str, Dict[str, Any]]:
        """
        Calculate DVH for all structures in the structure set.

        Parameters
        ----------
        dose : Dose
            Dose distribution
        structure_set : StructureSet
            Set of structures

        Returns
        -------
        Dict[str, Dict[str, Any]]
            Dictionary of DVH data for each structure
        """
        result = {}

        # Ensure we have valid dose and structures
        if not dose or not structure_set or not structure_set.structures:
            logger.warning("No dose or structures available for DVH calculation")
            return result

        # Get dose array and grid information
        dose_array = dose.dose_data
        if dose_array is None or dose_array.size == 0:
            logger.warning("Empty dose data")
            return result

        # Get dose range for bins
        max_dose = np.max(dose_array)
        if max_dose <= 0:
            logger.warning("No positive dose values found")
            return result

        # Create dose bins
        bin_width = max(max_dose / self.num_bins, MIN_DOSE_STEP)
        dose_bins = np.arange(0, max_dose + bin_width, bin_width)

        # Calculate DVH for each structure
        for structure in structure_set.structures:
            try:
                # Skip if structure has no ROI data
                if not structure.roi_data:
                    continue

                # Get structure mask
                structure_mask = structure.get_mask(
                    dose.dimensions, dose.voxel_size, dose.origin
                )

                # Skip if mask is empty
                if structure_mask is None or np.sum(structure_mask) == 0:
                    continue

                # Extract doses in structure
                structure_doses = dose_array[structure_mask > 0]

                # Skip if no doses
                if structure_doses.size == 0:
                    continue

                # Calculate histogram
                hist, bin_edges = np.histogram(structure_doses, bins=dose_bins)

                # Calculate DVH data
                dvh_data = self._calculate_dvh_data(
                    structure, hist, bin_edges, structure_doses
                )

                # Store result
                result[structure.name] = dvh_data

            except Exception as e:
                logger.error(
                    f"Error calculating DVH for structure {structure.name}: {e}"
                )

        return result

    def _calculate_dvh_data(
        self,
        structure: Structure,
        hist: np.ndarray,
        bin_edges: np.ndarray,
        structure_doses: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Calculate DVH data for a structure.

        Parameters
        ----------
        structure : Structure
            Structure to calculate DVH for
        hist : np.ndarray
            Histogram of doses
        bin_edges : np.ndarray
            Bin edges for histogram
        structure_doses : np.ndarray
            Doses within the structure

        Returns
        -------
        Dict[str, Any]
            DVH data for the structure
        """
        # Convert histogram to percent
        total_voxels = structure_doses.size
        hist_percent = (hist / total_voxels) * 100.0

        # Calculate cumulative histogram (reverse sum)
        cum_hist_percent = np.cumsum(hist_percent[::-1])[::-1]

        # Get dose bins (use center of bins)
        doses = (bin_edges[:-1] + bin_edges[1:]) / 2.0

        # Create result dictionary
        result = {
            "name": structure.name,
            "doses": doses.tolist(),
            "volumes_diff": hist_percent.tolist(),
            "volumes_cum": cum_hist_percent.tolist(),
            "volume": structure.volume,
            "mean_dose": np.mean(structure_doses),
            "max_dose": np.max(structure_doses),
            "min_dose": np.min(structure_doses),
        }

        # Calculate dose metrics
        dose_to_volume = {}
        for volume in [95, 90, 50, 5]:
            # Interpolate to find dose covering volume% of structure
            try:
                idx = np.argmin(np.abs(cum_hist_percent - volume))
                dose_to_volume[volume] = doses[idx]
            except (ValueError, IndexError):
                dose_to_volume[volume] = 0.0

        result["dose_to_volume"] = dose_to_volume

        # Calculate volume metrics for various dose levels
        volume_at_dose = {}
        dose_levels = np.arange(0, np.max(structure_doses) + 5, 5)
        for dose_level in dose_levels:
            try:
                idx = np.argmin(np.abs(doses - dose_level))
                volume_at_dose[dose_level] = cum_hist_percent[idx]
            except (ValueError, IndexError):
                volume_at_dose[dose_level] = 0.0

        result["volume_at_dose"] = volume_at_dose

        return result

    def save_dvh_to_csv(
        self,
        dvh_data: Dict[str, Dict[str, Any]],
        file_path: str,
        is_cumulative: bool = True,
    ) -> bool:
        """
        Save DVH data to CSV file.

        Parameters
        ----------
        dvh_data : Dict[str, Dict[str, Any]]
            DVH data for structures
        file_path : str
            Path to save CSV file
        is_cumulative : bool, optional
            Whether to save cumulative or differential DVH, by default True

        Returns
        -------
        bool
            True if successful, False otherwise
        """
        try:
            with open(file_path, "w") as f:
                # Write header
                dvh_type = "Cumulative" if is_cumulative else "Differential"
                f.write(f"# {dvh_type} DVH Data\n")
                f.write("#\n")

                # Write dose header
                f.write("Dose (Gy)")
                for structure_name in sorted(dvh_data.keys()):
                    f.write(f",{structure_name} (%)")
                f.write("\n")

                # Get all dose bins
                all_doses = []
                for structure_data in dvh_data.values():
                    if "doses" in structure_data:
                        all_doses.append(structure_data["doses"])

                # Use the most detailed dose bins
                max_len_idx = (
                    np.argmax([len(d) for d in all_doses]) if all_doses else -1
                )
                if max_len_idx >= 0:
                    doses = all_doses[max_len_idx]
                else:
                    return False

                # Write data rows
                for i, dose in enumerate(doses):
                    f.write(f"{dose:.3f}")

                    for structure_name in sorted(dvh_data.keys()):
                        structure_data = dvh_data[structure_name]
                        volumes = (
                            structure_data["volumes_cum"]
                            if is_cumulative
                            else structure_data["volumes_diff"]
                        )

                        # Interpolate if necessary
                        if i < len(volumes):
                            f.write(f",{volumes[i]:.3f}")
                        else:
                            f.write(",0.000")

                    f.write("\n")

            return True

        except Exception as e:
            logger.error(f"Error saving DVH to CSV: {e}")
            return False


def test_dvh_calculator():
    """Test function for DVH calculator."""
    # Create a simple test dose distribution
    dose_array = np.zeros((50, 50, 50))
    dose_array[10:40, 10:40, 10:40] = 50.0  # 50 Gy to central region

    # Create a simple test structure
    from quangtps.structures.structure import Structure

    structure = Structure("PTV")
    structure.roi_data = np.zeros((50, 50, 50), dtype=bool)
    structure.roi_data[15:35, 15:35, 15:35] = True

    # Create dose object
    from quangtps.dose.dose import Dose

    dose = Dose()
    dose.dose_data = dose_array
    dose.dimensions = dose_array.shape
    dose.voxel_size = (1.0, 1.0, 1.0)
    dose.origin = (0.0, 0.0, 0.0)

    # Create structure set
    from quangtps.structures.structure_set import StructureSet

    structure_set = StructureSet()
    structure_set.add_structure(structure)

    # Calculate DVH
    calculator = DVHCalculator()
    dvh_data = calculator.calculate_dvh(dose, structure_set)

    # Print results
    for structure_name, data in dvh_data.items():
        print(f"Structure: {structure_name}")
        print(f"  Mean dose: {data['mean_dose']:.2f} Gy")
        print(f"  Max dose: {data['max_dose']:.2f} Gy")
        print(f"  Min dose: {data['min_dose']:.2f} Gy")
        print(f"  D95: {data['dose_to_volume'][95]:.2f} Gy")
        print(f"  V50: {data['volume_at_dose'][50.0]:.2f}%")

    return dvh_data


if __name__ == "__main__":
    test_dvh_calculator()
