#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module for analyzing and managing DVH (Dose Volume Histogram) data.

This module provides classes and functions for calculating, analyzing, and visualizing
DVH data for radiotherapy treatment planning.
"""

import logging
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import time
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import copy
import datetime
from quangtps.core.structures import StructureSet
from quangtps.core.utils import create_unique_id  # Changed from utilities to utils

from quangtps.segmentation.structures.structure import Structure

logger = logging.getLogger(__name__)


class DVHCurve:
    """
    Class representing a single DVH curve (cumulative or differential).

    Attributes:
        structure_id: Identifier for the structure
        structure_name: Name of the structure
        dose_bins: Array of dose values in Gy
        volume_bins: Array of volume values in % (0-100)
        is_cumulative: Whether this is a cumulative curve
        total_volume: Volume of the structure in cc
        min_dose: Minimum dose to the structure in Gy
        max_dose: Maximum dose to the structure in Gy
        mean_dose: Mean dose to the structure in Gy
        d_metrics: Dictionary of cached Dx metrics
        v_metrics: Dictionary of cached Vx metrics
    """

    def __init__(
        self,
        structure_id: str,
        structure_name: str,
        dose_bins: List[float] = None,
        volume_bins: List[float] = None,
        is_cumulative: bool = True,
    ):
        """
        Initialize a DVH curve.

        Args:
            structure_id: Identifier for the structure
            structure_name: Name of the structure
            dose_bins: Array of dose values in Gy (optional)
            volume_bins: Array of volume values in % (0-100) (optional)
            is_cumulative: Whether this is a cumulative curve
        """
        self.structure_id = structure_id
        self.structure_name = structure_name
        self.dose_bins = dose_bins or []
        self.volume_bins = volume_bins or []
        self.is_cumulative = is_cumulative

        # Additional attributes
        self.total_volume = 0.0
        self.min_dose = 0.0
        self.max_dose = 0.0
        self.mean_dose = 0.0

        # Cache for metrics
        self.d_metrics = {}  # Cache for Dx metrics
        self.v_metrics = {}  # Cache for Vx metrics

    def set_data(
        self,
        dose_bins: List[float],
        volume_bins: List[float],
        is_cumulative: bool = True,
    ):
        """
        Set the data for this curve.

        Args:
            dose_bins: Array of dose values in Gy
            volume_bins: Array of volume values in % (0-100)
            is_cumulative: Whether this is a cumulative curve
        """
        # Ensure arrays are the same length
        if len(dose_bins) != len(volume_bins):
            raise ValueError("Dose and volume arrays must be the same length")

        self.dose_bins = dose_bins
        self.volume_bins = volume_bins
        self.is_cumulative = is_cumulative

        # Update derived values
        if len(dose_bins) > 0:
            self.min_dose = min(dose_bins)
            self.max_dose = max(dose_bins)

            # Calculate mean dose (approximated from the differential DVH)
            if not is_cumulative and len(dose_bins) > 1:
                # For differential DVH, mean dose is weighted sum of bin doses
                total_vol = sum(volume_bins)
                if total_vol > 0:
                    self.mean_dose = (
                        sum(d * v for d, v in zip(dose_bins, volume_bins)) / total_vol
                    )

            # Clear cached metrics
            self.d_metrics = {}
            self.v_metrics = {}

    def calculate_d_metric(self, volume_percent: float) -> float:
        """
        Calculate Dx - the dose received by x% of the volume.

        Args:
            volume_percent: Volume percentage (0-100)

        Returns:
            Dose in Gy at specified volume
        """
        # Ensure we're using the cumulative curve data
        if not self.is_cumulative:
            # If we have differential data, convert it to cumulative first
            # This is a placeholder - actual implementation would depend on your data structure
            logger.warning("Calculating D-metric on differential DVH may be inaccurate")

        # Ensure volume is in the range [0, 100]
        volume_percent = max(0.0, min(100.0, volume_percent))

        # Find the index where volume <= volume_percent
        # For cumulative DVH, higher volume means lower dose
        for i in range(len(self.volume_bins)):
            if self.volume_bins[i] <= volume_percent:
                if i == 0:
                    return self.dose_bins[0]
                else:
                    # Linear interpolation between points
                    v1, v2 = self.volume_bins[i - 1], self.volume_bins[i]
                    d1, d2 = self.dose_bins[i - 1], self.dose_bins[i]

                    # Handle case where volumes are the same
                    if v1 == v2:
                        return d1

                    # Interpolate
                    ratio = (volume_percent - v1) / (v2 - v1)
                    dose = d1 + ratio * (d2 - d1)

                    # Cache the result
                    self.d_metrics[volume_percent] = dose
                    return dose

        # If volume is lower than any in the curve, return max dose
        return self.max_dose

    def calculate_v_metric(self, dose: float) -> float:
        """
        Calculate Vx - the volume (in %) receiving at least x Gy.

        Args:
            dose: Dose threshold in Gy

        Returns:
            Volume percentage (0-100) receiving at least the specified dose
        """
        # Ensure we're using the cumulative curve data
        if not self.is_cumulative:
            # If we have differential data, convert it to cumulative first
            logger.warning("Calculating V-metric on differential DVH may be inaccurate")

        # Find the index where dose >= specified dose
        for i in range(len(self.dose_bins)):
            if self.dose_bins[i] >= dose:
                if i == 0:
                    return self.volume_bins[0]
                else:
                    # Linear interpolation between points
                    d1, d2 = self.dose_bins[i - 1], self.dose_bins[i]
                    v1, v2 = self.volume_bins[i - 1], self.volume_bins[i]

                    # Handle case where doses are the same
                    if d1 == d2:
                        return v1

                    # Interpolate
                    ratio = (dose - d1) / (d2 - d1)
                    volume = v1 + ratio * (v2 - v1)

                    # Cache the result
                    self.v_metrics[dose] = volume
                    return volume

        # If dose is higher than any in the curve, return 0 volume
        return 0.0


class DVHData:
    """
    Container for DVH curves for multiple structures.
    Includes methods for calculating and retrieving DVH metrics.
    """

    def __init__(self):
        """Initialize DVH data container."""
        self.curves: Dict[str, DVHCurve] = {}  # Map of structure_id to DVHCurve
        self.structures: Dict[str, Structure] = {}  # Map of structure_id to Structure

        # Plan-level information
        self.prescription_dose = 0.0  # Prescription dose in Gy
        self.plan_name = ""
        self.patient_id = ""
        self.date_calculated = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Calculation parameters
        self.bin_width = 0.1  # Dose bin width in Gy
        self.calculation_grid_size = 0.0  # Calculation grid size in mm

    @classmethod
    def from_raw_data(
        cls,
        structure_id: str,
        dose_bins: List[float],
        cumulative_volume: List[float],
        total_volume: float,
        structure_name: str = "",
    ) -> "DVHData":
        """
        Create a DVH data object from raw data arrays.

        Args:
            structure_id: Unique identifier for the structure
            dose_bins: Array of dose bin values (typically in Gy)
            cumulative_volume: Array of cumulative volume values at each dose bin
            total_volume: Total volume of the structure in cc
            structure_name: Name of the structure (optional)

        Returns:
            A new DVHData object with a single curve
        """
        # Create a new DVHData object
        dvh_data = cls()

        # Validate inputs
        if len(dose_bins) != len(cumulative_volume):
            raise ValueError(
                "Dose bins and cumulative volume arrays must have the same length"
            )

        # Create a new curve
        curve = DVHCurve(structure_id, structure_name or structure_id)
        curve.set_data(dose_bins, cumulative_volume, is_cumulative=True)
        curve.total_volume = total_volume

        # Calculate statistics if not already set
        if curve.min_dose == 0 and len(dose_bins) > 0:
            curve.min_dose = min(dose_bins)
        if curve.max_dose == 0 and len(dose_bins) > 0:
            curve.max_dose = max(dose_bins)

        # Calculate mean dose from differential volume if possible
        if curve.mean_dose == 0 and len(dose_bins) > 1:
            # Create differential volume from cumulative
            diff_vol = [-np.diff([100.0] + cumulative_volume)]
            # Dose values for differential (midpoints of bins)
            diff_dose = [
                (dose_bins[i] + dose_bins[i + 1]) / 2 for i in range(len(dose_bins) - 1)
            ]
            # Calculate mean dose
            if sum(diff_vol) > 0:
                curve.mean_dose = sum(d * v for d, v in zip(diff_dose, diff_vol)) / sum(
                    diff_vol
                )

        # Add curve to DVHData
        dvh_data.add_curve(curve)

        return dvh_data

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

    def calculate_conformity_index(
        self, target_id: str, reference_dose: float
    ) -> float:
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
        return v_ref / target_volume

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert DVH data to a dictionary for serialization.

        Returns:
            Dictionary representation of the DVH data
        """
        curves_dict = {}
        for structure_id, curve in self.curves.items():
            curves_dict[structure_id] = {
                "structure_id": curve.structure_id,
                "structure_name": curve.structure_name,
                "dose_bins": curve.dose_bins,
                "volume_bins": curve.volume_bins,
                "is_cumulative": curve.is_cumulative,
                "total_volume": curve.total_volume,
                "min_dose": curve.min_dose,
                "max_dose": curve.max_dose,
                "mean_dose": curve.mean_dose,
            }

        return {
            "curves": curves_dict,
            "prescription_dose": self.prescription_dose,
            "plan_name": self.plan_name,
            "patient_id": self.patient_id,
            "date_calculated": self.date_calculated,
            "bin_width": self.bin_width,
            "calculation_grid_size": self.calculation_grid_size,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DVHData":
        """
        Create a DVH data object from a dictionary.

        Args:
            data: Dictionary containing DVH data

        Returns:
            A new DVHData object
        """
        dvh_data = cls()

        dvh_data.prescription_dose = data.get("prescription_dose", 0.0)
        dvh_data.plan_name = data.get("plan_name", "")
        dvh_data.patient_id = data.get("patient_id", "")
        dvh_data.date_calculated = data.get("date_calculated", "")
        dvh_data.bin_width = data.get("bin_width", 0.1)
        dvh_data.calculation_grid_size = data.get("calculation_grid_size", 0.0)

        # Load curves
        curves_dict = data.get("curves", {})
        for structure_id, curve_data in curves_dict.items():
            curve = DVHCurve(
                structure_id=curve_data.get("structure_id", structure_id),
                structure_name=curve_data.get("structure_name", structure_id),
                dose_bins=curve_data.get("dose_bins", []),
                volume_bins=curve_data.get("volume_bins", []),
                is_cumulative=curve_data.get("is_cumulative", True),
            )
            curve.total_volume = curve_data.get("total_volume", 0.0)
            curve.min_dose = curve_data.get("min_dose", 0.0)
            curve.max_dose = curve_data.get("max_dose", 0.0)
            curve.mean_dose = curve_data.get("mean_dose", 0.0)

            dvh_data.add_curve(curve)

        return dvh_data

    def to_json(self) -> str:
        """
        Convert DVH data to a JSON string.

        Returns:
            JSON string representation of the DVH data
        """
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> "DVHData":
        """
        Create a DVH data object from a JSON string.

        Args:
            json_str: JSON string containing DVH data

        Returns:
            A new DVHData object
        """
        data = json.loads(json_str)
        return cls.from_dict(data)

    def resample(self, bin_count: int = 100) -> "DVHData":
        """
        Resample the DVH data to a specific number of dose bins.

        Args:
            bin_count: Number of dose bins in the resampled DVH

        Returns:
            A new DVHData object with resampled data
        """
        # Create a new DVHData object
        resampled_dvh = DVHData()

        # Copy plan-level information
        resampled_dvh.prescription_dose = self.prescription_dose
        resampled_dvh.plan_name = self.plan_name
        resampled_dvh.patient_id = self.patient_id
        resampled_dvh.date_calculated = self.date_calculated
        resampled_dvh.bin_width = self.bin_width
        resampled_dvh.calculation_grid_size = self.calculation_grid_size

        # Resample each curve
        for structure_id, curve in self.curves.items():
            if not curve.dose_bins or bin_count <= 0:
                resampled_dvh.add_curve(copy.deepcopy(curve))
                continue

            # Create new dose bins
            new_dose_bins = np.linspace(curve.min_dose, curve.max_dose, bin_count)

            # Interpolate cumulative volume
            cum_vol_interp = np.interp(
                new_dose_bins, curve.dose_bins, curve.volume_bins
            )

            # Create resampled curve
            resampled_curve = DVHCurve(
                structure_id=curve.structure_id,
                structure_name=curve.structure_name,
                dose_bins=new_dose_bins.tolist(),
                volume_bins=cum_vol_interp.tolist(),
                is_cumulative=curve.is_cumulative,
            )

            # Copy over statistics
            resampled_curve.total_volume = curve.total_volume
            resampled_curve.min_dose = curve.min_dose
            resampled_curve.max_dose = curve.max_dose
            resampled_curve.mean_dose = curve.mean_dose

            # Add to resampled DVH
            resampled_dvh.add_curve(resampled_curve)

        return resampled_dvh
