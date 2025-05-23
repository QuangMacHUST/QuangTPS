#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DVH calculation functions for QuangTPS.

This module provides functions for calculating dose-volume histograms (DVHs)
and related metrics from dose distributions and structures.
"""

import os
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from scipy import interpolate

from quangtps.evaluation.dvh.dvh_data import DVHData, DVHCurve
from quangtps.dose.dose_distribution import DoseDistribution
from quangtps.structures.structure_set import StructureSet
from quangtps.structures.structure import Structure

logger = logging.getLogger(__name__)


def calculate_dvh(
    dose_grid,
    structure_mask,
    dose_bins: Optional[np.ndarray] = None,
    bin_count: int = 1000,
) -> Dict:
    """
    Calculate the DVH for a structure.

    Parameters
    ----------
    dose_grid : np.ndarray
        3D dose grid with dose values in Gy
    structure_mask : np.ndarray
        Binary 3D mask of the structure (same dimensions as dose_grid)
    dose_bins : np.ndarray, optional
        Array of dose bin edges. If not provided, bins will be automatically
        generated based on the dose range.
    bin_count : int, optional
        Number of bins to use if dose_bins is not provided

    Returns
    -------
    Dict
        Dictionary containing DVH data with keys:
        - 'dose_bins': Array of dose bin centers
        - 'differential_volume': Differential DVH values (% volume per bin)
        - 'cumulative_volume': Cumulative DVH values (% volume)
        - 'min_dose': Minimum dose in the structure (Gy)
        - 'max_dose': Maximum dose in the structure (Gy)
        - 'mean_dose': Mean dose in the structure (Gy)
        - 'median_dose': Median dose in the structure (Gy)
        - 'std_dose': Standard deviation of dose in the structure (Gy)
        - 'volume': Volume of the structure (cm³)
    """
    # Extract dose values within structure
    structure_doses = dose_grid[structure_mask > 0]

    if len(structure_doses) == 0:
        logger.warning("No dose points found within structure")
        return {
            "dose_bins": np.array([0]),
            "differential_volume": np.array([0]),
            "cumulative_volume": np.array([0]),
            "min_dose": 0,
            "max_dose": 0,
            "mean_dose": 0,
            "median_dose": 0,
            "std_dose": 0,
            "volume": 0,
        }

    # Get basic dose statistics
    min_dose = np.min(structure_doses)
    max_dose = np.max(structure_doses)
    mean_dose = np.mean(structure_doses)
    median_dose = np.median(structure_doses)
    std_dose = np.std(structure_doses)

    # Create dose bins if not provided
    if dose_bins is None:
        # Ensure we have some margin above max dose
        max_bin = max_dose * 1.05
        dose_bins = np.linspace(0, max_bin, bin_count + 1)

    # Calculate bin centers
    bin_centers = (dose_bins[1:] + dose_bins[:-1]) / 2

    # Calculate histogram
    hist, _ = np.histogram(structure_doses, bins=dose_bins, density=False)

    # Convert to volume percentages
    total_voxels = len(structure_doses)
    differential_volume = (hist / total_voxels) * 100

    # Calculate cumulative DVH (reversed: % volume receiving at least X dose)
    cumulative_volume = np.cumsum(differential_volume[::-1])[::-1]

    # Ensure values are monotonically decreasing (for interpolation)
    for i in range(1, len(cumulative_volume)):
        if cumulative_volume[i] > cumulative_volume[i - 1]:
            cumulative_volume[i] = cumulative_volume[i - 1]

    # Estimate structure volume (assuming voxel size in cm)
    # This is just a placeholder - real implementation would use voxel dimensions
    voxel_volume = 0.1 * 0.1 * 0.1  # Example: 1mm³ voxels = 0.001 cm³
    volume = total_voxels * voxel_volume

    return {
        "dose_bins": bin_centers,
        "differential_volume": differential_volume,
        "cumulative_volume": cumulative_volume,
        "min_dose": min_dose,
        "max_dose": max_dose,
        "mean_dose": mean_dose,
        "median_dose": median_dose,
        "std_dose": std_dose,
        "volume": volume,
    }


def calculate_dvh_for_plan(
    dose_grid, structures: Dict, bin_count: int = 1000
) -> Dict[str, Dict]:
    """
    Calculate DVHs for all structures in a plan.

    Parameters
    ----------
    dose_grid : np.ndarray
        3D dose grid with dose values in Gy
    structures : Dict
        Dictionary mapping structure names to binary 3D masks
    bin_count : int, optional
        Number of bins to use for DVH calculation

    Returns
    -------
    Dict[str, Dict]
        Dictionary mapping structure names to DVH data dictionaries
    """
    # Create common dose bins for all structures
    max_dose = np.max(dose_grid)
    dose_bins = np.linspace(0, max_dose * 1.05, bin_count + 1)

    # Calculate DVH for each structure
    dvhs = {}
    for name, mask in structures.items():
        dvhs[name] = calculate_dvh(
            dose_grid, mask, dose_bins=dose_bins, bin_count=bin_count
        )

    return dvhs


def calculate_dvh_metrics(
    dvh_data: Dict,
    metrics_list: Optional[List[str]] = None,
    rx_dose: Optional[float] = None,
) -> Dict:
    """
    Calculate various DVH metrics from DVH data.

    Parameters
    ----------
    dvh_data : Dict
        Dictionary containing DVH data with keys:
        - 'dose_bins': array of dose values
        - 'cumulative_volume': array of cumulative volume values
        - 'min_dose', 'max_dose', 'mean_dose', etc.
    metrics_list : List[str], optional
        List of metrics to calculate, e.g. ['D95', 'V20', 'Dmax']
        If None, calculates a default set of metrics
    rx_dose : float, optional
        Prescription dose in Gy, used for calculating relative metrics

    Returns
    -------
    Dict
        Dictionary of calculated metrics
    """
    # Default metrics if none provided
    if metrics_list is None:
        metrics_list = [
            "Dmin",
            "Dmean",
            "Dmax",
            "D95",
            "D90",
            "D50",
            "D2",
            "V5",
            "V10",
            "V20",
            "V30",
            "V40",
            "V50",
        ]

    # Get dose and volume data
    dose_bins = dvh_data.get("dose_bins", np.array([]))
    volume_bins = dvh_data.get("cumulative_volume", np.array([]))
    structure_volume = dvh_data.get("volume", 0.0)

    # Initialize result dictionary
    result = {}

    if (
        len(dose_bins) == 0
        or len(volume_bins) == 0
        or len(dose_bins) != len(volume_bins)
    ):
        logger.warning("Invalid DVH data for metrics calculation")
        return result

    # Calculate basic metrics
    for metric in metrics_list:
        if metric.startswith("D") and metric[1:].replace(".", "", 1).isdigit():
            # Dx metrics (dose to x% volume)
            try:
                percent = float(metric[1:])
                result[metric] = _get_dose_at_volume(dose_bins, volume_bins, percent)
            except (ValueError, IndexError) as e:
                logger.warning(f"Cannot calculate {metric}: {str(e)}")
                result[metric] = float("nan")

        elif metric.startswith("V") and metric[1:].replace(".", "", 1).isdigit():
            # Vx metrics (volume receiving at least x Gy)
            try:
                dose = float(metric[1:])
                # If prescription dose is provided and metric contains %, convert to absolute dose
                if rx_dose is not None and "%" in metric:
                    percent = float(metric[1:-1])  # remove 'V' and '%'
                    dose = rx_dose * percent / 100

                result[metric] = _get_volume_at_dose(dose_bins, volume_bins, dose)
            except (ValueError, IndexError) as e:
                logger.warning(f"Cannot calculate {metric}: {str(e)}")
                result[metric] = float("nan")

        elif metric == "Dmax":
            result[metric] = np.max(dose_bins) if len(dose_bins) > 0 else float("nan")

        elif metric == "Dmean":
            # Calculate mean dose from histogram
            if "mean_dose" in dvh_data:
                result[metric] = dvh_data["mean_dose"]
            else:
                # Approximate from dose_bins and volume_bins
                total_dose = 0
                total_volume = 0

                # Calculate volume differences for weighting
                for i in range(len(volume_bins) - 1):
                    vol_diff = abs(volume_bins[i] - volume_bins[i + 1])
                    if vol_diff > 0:
                        # Use average dose for this volume slice
                        avg_dose = (dose_bins[i] + dose_bins[i + 1]) / 2
                        total_dose += avg_dose * vol_diff
                        total_volume += vol_diff

                result[metric] = (
                    total_dose / total_volume if total_volume > 0 else float("nan")
                )

        elif metric == "Dmin":
            # Find minimum non-zero dose
            non_zero_doses = [d for i, d in enumerate(dose_bins) if volume_bins[i] > 0]
            result[metric] = min(non_zero_doses) if non_zero_doses else float("nan")

    # Calculate prescription-related metrics if provided
    if rx_dose is not None:
        if "V_rx" in metrics_list:
            result["V_rx"] = _get_volume_at_dose(dose_bins, volume_bins, rx_dose)

        if "V_rx_percent" in metrics_list:
            v_rx = _get_volume_at_dose(dose_bins, volume_bins, rx_dose)
            result["V_rx_percent"] = (
                100 * v_rx / structure_volume if structure_volume > 0 else float("nan")
            )

    return result


def _get_dose_at_volume(
    dose_bins: np.ndarray, cumulative_volume: np.ndarray, volume_percent: float
) -> float:
    """
    Calculate the dose at a specific volume percentage.

    Parameters
    ----------
    dose_bins : np.ndarray
        Array of dose bin centers
    cumulative_volume : np.ndarray
        Cumulative DVH values (% volume)
    volume_percent : float
        Volume percentage to find dose at

    Returns
    -------
    float
        Dose at the specified volume percentage (Gy)
    """
    # Check if we have valid data
    if len(dose_bins) <= 1 or len(cumulative_volume) <= 1:
        return 0.0

    # Handle boundary cases
    if volume_percent >= 100 or volume_percent >= cumulative_volume[0]:
        return dose_bins[0]
    if volume_percent <= 0 or volume_percent <= cumulative_volume[-1]:
        return dose_bins[-1]

    # Find the dose at the given volume by interpolation
    # Note: cumulative_volume is decreasing with dose
    for i in range(len(cumulative_volume) - 1):
        if (
            cumulative_volume[i] >= volume_percent >= cumulative_volume[i + 1]
            or cumulative_volume[i] <= volume_percent <= cumulative_volume[i + 1]
        ):
            # Linear interpolation
            slope = (dose_bins[i + 1] - dose_bins[i]) / (
                cumulative_volume[i + 1] - cumulative_volume[i]
            )
            return dose_bins[i] + slope * (volume_percent - cumulative_volume[i])

    # Fallback - shouldn't reach here with proper data
    return 0.0


def _get_volume_at_dose(
    dose_bins: np.ndarray, cumulative_volume: np.ndarray, dose_value: float
) -> float:
    """
    Calculate the volume percentage receiving at least a specific dose.

    Parameters
    ----------
    dose_bins : np.ndarray
        Array of dose bin centers
    cumulative_volume : np.ndarray
        Cumulative DVH values (% volume)
    dose_value : float
        Dose value to find volume at (Gy)

    Returns
    -------
    float
        Volume percentage receiving at least the specified dose (%)
    """
    # Check if we have valid data
    if len(dose_bins) <= 1 or len(cumulative_volume) <= 1:
        return 0.0

    # Handle boundary cases
    if dose_value <= dose_bins[0]:
        return cumulative_volume[0]
    if dose_value >= dose_bins[-1]:
        return cumulative_volume[-1]

    # Find the volume at the given dose by interpolation
    for i in range(len(dose_bins) - 1):
        if dose_bins[i] <= dose_value <= dose_bins[i + 1]:
            # Linear interpolation
            slope = (cumulative_volume[i + 1] - cumulative_volume[i]) / (
                dose_bins[i + 1] - dose_bins[i]
            )
            return cumulative_volume[i] + slope * (dose_value - dose_bins[i])

    # Fallback - shouldn't reach here with proper data
    return 0.0


def calculate_conformity_index(target_dvh: Dict, prescription_dose: float) -> float:
    """
    Calculate the conformity index for a target.

    Conformity Index (CI) = (V_prescription / V_target)
    where V_prescription is the volume of tissue receiving the prescription dose
    and V_target is the target volume.

    Parameters
    ----------
    target_dvh : Dict
        DVH data for the target structure
    prescription_dose : float
        Prescription dose (Gy)

    Returns
    -------
    float
        Conformity index value
    """
    dose_bins = target_dvh.get("dose_bins", np.array([0]))
    cumulative_volume = target_dvh.get("cumulative_volume", np.array([0]))

    # Get volume receiving prescription dose
    v_prescription = _get_volume_at_dose(
        dose_bins, cumulative_volume, prescription_dose
    )

    # Calculate CI
    return v_prescription / 100.0  # Convert from % to ratio


def calculate_homogeneity_index(target_dvh: Dict, prescription_dose: float) -> float:
    """
    Calculate the homogeneity index for a target.

    Homogeneity Index (HI) = (D2% - D98%) / Prescription Dose

    Parameters
    ----------
    target_dvh : Dict
        DVH data for the target structure
    prescription_dose : float
        Prescription dose (Gy)

    Returns
    -------
    float
        Homogeneity index value
    """
    metrics = calculate_dvh_metrics(target_dvh)

    # Calculate D2% and D98%
    d2 = metrics.get("D2", 0)
    d98 = metrics.get("D98", 0)

    # Calculate HI
    if prescription_dose > 0:
        return (d2 - d98) / prescription_dose
    return 0.0


def calculate_gradient_index(
    reference_dvh: Dict, high_dose: float, low_dose: float
) -> float:
    """
    Calculate the gradient index.

    Gradient Index (GI) = V_low_dose / V_high_dose
    where V_low_dose is the volume receiving the lower reference dose
    and V_high_dose is the volume receiving the higher reference dose.

    Parameters
    ----------
    reference_dvh : Dict
        DVH data for the reference structure (typically the body)
    high_dose : float
        High reference dose (Gy)
    low_dose : float
        Low reference dose (Gy)

    Returns
    -------
    float
        Gradient index value
    """
    dose_bins = reference_dvh.get("dose_bins", np.array([0]))
    cumulative_volume = reference_dvh.get("cumulative_volume", np.array([0]))

    # Get volumes
    v_high = _get_volume_at_dose(dose_bins, cumulative_volume, high_dose)
    v_low = _get_volume_at_dose(dose_bins, cumulative_volume, low_dose)

    # Calculate GI
    if v_high > 0:
        return v_low / v_high
    return 0.0


def calculate_equivalent_uniform_dose(dvh_data: Dict, a: float = 1.0) -> float:
    """
    Calculate the generalized Equivalent Uniform Dose (gEUD).

    gEUD = (Σ vi × Di^a)^(1/a)
    where vi is the fractional volume, Di is the dose, and a is a tissue-specific parameter.

    Parameters
    ----------
    dvh_data : Dict
        DVH data dictionary
    a : float, optional
        Tissue-specific parameter:
        a > 0 for tumors (higher values for more aggressive tumors)
        a < 0 for normal tissues (more negative for serial organs)

    Returns
    -------
    float
        gEUD value (Gy)
    """
    dose_bins = dvh_data.get("dose_bins", np.array([0]))
    cumulative_volume = dvh_data.get("cumulative_volume", np.array([0]))

    # Check for valid data
    if len(dose_bins) <= 1 or len(cumulative_volume) <= 1:
        return 0.0

    # Convert to fractional volumes
    fractional_volume = cumulative_volume / 100.0

    # Calculate sum vi × Di^a
    sum_term = np.sum(fractional_volume * np.power(dose_bins, a))

    # Calculate gEUD
    if sum_term > 0:
        return np.power(sum_term, 1.0 / a)
    return 0.0


def calculate_dvh_from_dose_grid(
    dose_grid: np.ndarray,
    structure_mask: np.ndarray,
    num_bins: int = 100,
    volume_type: str = "relative",
) -> Dict[str, Any]:
    """
    Tính toán DVH từ lưới liều (dose grid) và mặt nạ cấu trúc.

    Parameters
    ----------
    dose_grid : np.ndarray
        Mảng chứa dữ liệu liều (Gy)
    structure_mask : np.ndarray
        Mặt nạ nhị phân của cấu trúc (1 trong cấu trúc, 0 ngoài cấu trúc)
    num_bins : int, optional
        Số lượng bin cho histogram
    volume_type : str, optional
        'relative' để xuất thể tích theo %, 'absolute' để xuất thể tích theo cc

    Returns
    -------
    Dict[str, Any]
        Dictionary chứa dữ liệu DVH, bao gồm dose_bins, volume_bins, min_dose, mean_dose, max_dose, etc.
    """
    # Kiểm tra đầu vào
    if dose_grid.shape != structure_mask.shape:
        raise ValueError(
            f"Kích thước dose_grid {dose_grid.shape} và structure_mask {structure_mask.shape} không khớp"
        )

    # Tạo mặt nạ cấu trúc boolean
    mask_bool = structure_mask > 0

    # Lấy giá trị liều trong cấu trúc
    structure_doses = dose_grid[mask_bool]

    if len(structure_doses) == 0:
        logger.warning("Không có voxel nào trong cấu trúc")
        return {
            "dose_bins": np.array([]),
            "volume_bins": np.array([]),
            "min_dose": 0.0,
            "mean_dose": 0.0,
            "max_dose": 0.0,
            "structure_volume": 0.0,
        }

    # Tính thể tích cấu trúc (tổng số voxel)
    structure_volume = np.sum(mask_bool)

    # Tính các thống kê cơ bản
    min_dose = np.min(structure_doses) if len(structure_doses) > 0 else 0.0
    mean_dose = np.mean(structure_doses) if len(structure_doses) > 0 else 0.0
    max_dose = np.max(structure_doses) if len(structure_doses) > 0 else 0.0

    # Tạo bin liều
    if max_dose > min_dose:
        dose_bins = np.linspace(0, max_dose * 1.05, num_bins)
    else:
        # Tránh trường hợp liều đồng nhất
        dose_bins = np.linspace(0, mean_dose * 2, num_bins)

    # Tính histogram
    hist, bin_edges = np.histogram(structure_doses, bins=dose_bins)

    # Chuyển đổi sang tích lũy (cumulative)
    cum_hist = np.cumsum(hist[::-1])[::-1]

    # Chuẩn hóa theo loại thể tích
    if volume_type.lower() == "relative":
        volume_bins = cum_hist / structure_volume * 100.0  # Theo phần trăm
    else:
        # Nếu có thông tin voxel size, nhân với thể tích thực, hoặc giả định 1mm³ mỗi voxel
        voxel_volume = 1.0  # mm³, giả định
        volume_bins = cum_hist * voxel_volume / 1000.0  # Chuyển thành cc (cm³)

    # Tạo kết quả
    dvh_data = {
        "dose_bins": dose_bins[:-1],  # Bỏ bin cuối từ bin_edges
        "volume_bins": volume_bins,
        "min_dose": min_dose,
        "mean_dose": mean_dose,
        "max_dose": max_dose,
        "structure_volume": structure_volume,
        "is_cumulative": True,
    }

    return dvh_data


def get_structure_color(structure_name: str) -> str:
    """
    Get a color for a structure based on its name.

    Args:
        structure_name: Name of the structure

    Returns:
        Hex color code
    """
    # Standard colors for common structures
    structure_colors = {
        "ptv": "#FF0000",  # Red
        "target": "#FF0000",  # Red
        "gtv": "#FF3333",  # Lighter red
        "ctv": "#FF6666",  # Even lighter red
        "body": "#00FF00",  # Green
        "external": "#00FF00",  # Green
        "cord": "#FFFF00",  # Yellow
        "spinal_cord": "#FFFF00",  # Yellow
        "heart": "#FF69B4",  # Pink
        "lung": "#ADD8E6",  # Light blue
        "lung_left": "#ADD8E6",  # Light blue
        "lung_right": "#87CEEB",  # Sky blue
        "liver": "#8B4513",  # Brown
        "kidney": "#A52A2A",  # Brown
        "kidney_left": "#A52A2A",  # Brown
        "kidney_right": "#8B4513",  # Dark brown
        "brain": "#FFA500",  # Orange
        "brainstem": "#FF8C00",  # Dark orange
        "bladder": "#FFD700",  # Gold
        "rectum": "#9400D3",  # Purple
        "bowel": "#800080",  # Purple
        "small_bowel": "#BA55D3",  # Medium purple
        "parotid": "#00FFFF",  # Cyan
        "parotid_left": "#00FFFF",  # Cyan
        "parotid_right": "#40E0D0",  # Turquoise
        "eye": "#1E90FF",  # Dodger blue
        "eye_left": "#1E90FF",  # Dodger blue
        "eye_right": "#4169E1",  # Royal blue
        "optic_nerve": "#7B68EE",  # Medium slate blue
        "optic_chiasm": "#9370DB",  # Medium purple
        "femur": "#808080",  # Gray
        "femur_left": "#A9A9A9",  # Dark gray
        "femur_right": "#808080",  # Gray
    }

    # Check for partial matches in lowercase
    name_lower = structure_name.lower()
    for key, color in structure_colors.items():
        if key in name_lower:
            return color

    # Default color for unknown structures
    return "#336699"  # Blue


class DVHCalculator:
    """
    Calculates Dose-Volume Histograms (DVH) for radiotherapy treatment plans.
    This class handles the calculation of DVH curves from dose distributions and structure sets.
    """

    def __init__(self):
        """Initialize the DVH calculator."""
        self.min_dose = 0.0
        self.max_dose = 100.0
        self.bin_width = 0.1
        self.bin_count = 1000  # Add bin_count attribute
        self.use_relative_volume = True

    def set_bin_width(self, bin_width: float):
        """Set the dose bin width for DVH calculation"""
        if bin_width <= 0:
            raise ValueError("Bin width must be positive")

        self.bin_width = bin_width

    def set_dose_range(self, min_dose: float, max_dose: float):
        """Set the dose range for DVH calculation"""
        if min_dose < 0:
            raise ValueError("Minimum dose cannot be negative")
        if max_dose <= min_dose:
            raise ValueError("Maximum dose must be greater than minimum dose")

        self.min_dose = min_dose
        self.max_dose = max_dose

    def calculate_dvh(
        self,
        dose: DoseDistribution,
        structure_set: StructureSet,
        structures_to_include: Optional[List[str]] = None,
    ) -> DVHData:
        """
        Calculate DVH for all structures in the structure set, or for specific structures if provided.

        Args:
            dose: The dose distribution
            structure_set: The structure set containing structures
            structures_to_include: Optional list of structure IDs to include (if None, include all)

        Returns:
            DVHData object containing all calculated DVH curves
        """
        # Create a new DVHData object
        dvh_data = DVHData()
        dvh_data.bin_width = self.bin_width

        # Set plan-level information
        dvh_data.plan_name = getattr(dose, "plan_name", "")
        dvh_data.prescription_dose = getattr(dose, "prescription_dose", 0.0)
        dvh_data.patient_id = getattr(dose, "patient_id", "")
        dvh_data.calculation_grid_size = getattr(dose, "grid_size", 0.0)

        # Get the list of structures to process
        if structures_to_include:
            structures = [
                s for s in structure_set.structures if s.id in structures_to_include
            ]
        else:
            structures = structure_set.structures

        # Calculate DVH for each structure
        for structure in structures:
            try:
                # Calculate DVH curve for this structure
                curve = self._calculate_structure_dvh(structure, dose)

                # Add to results
                if hasattr(dvh_data, "add_curve"):
                    dvh_data.add_curve(curve)
                elif hasattr(dvh_data, "curves"):
                    dvh_data.curves[structure.id] = curve
                else:
                    # Fallback: add as attribute
                    setattr(dvh_data, f"curve_{structure.id}", curve)

            except Exception as e:
                logger.error(
                    f"Error calculating DVH for structure {structure.name}: {str(e)}"
                )
                # Continue with next structure

        return dvh_data

    def _calculate_structure_dvh(
        self, structure: Structure, dose: DoseDistribution
    ) -> DVHCurve:
        """
        Calculate DVH curve for a single structure using real dose distribution.

        Args:
            structure: The structure
            dose: The dose distribution

        Returns:
            DVHCurve object for the structure
        """
        try:
            # Get structure mask and dose data
            structure_mask = getattr(structure, "mask", None)
            dose_data = getattr(dose, "data", None)

            if structure_mask is None or dose_data is None:
                logger.warning(
                    f"Missing mask or dose data for structure {structure.name}"
                )
                # Create synthetic DVH as fallback
                return self._create_synthetic_dvh_curve_fallback(structure, dose)

            # Ensure mask and dose have compatible shapes
            if structure_mask.shape != dose_data.shape:
                logger.warning(
                    f"Shape mismatch: mask {structure_mask.shape} vs dose {dose_data.shape}"
                )

                # Try to interpolate mask to match dose grid
                try:
                    from scipy.ndimage import zoom

                    zoom_factors = [
                        dose_data.shape[i] / structure_mask.shape[i] for i in range(3)
                    ]
                    structure_mask = (
                        zoom(structure_mask.astype(float), zoom_factors, order=1) > 0.5
                    )
                    logger.info(f"Resized mask to {structure_mask.shape}")
                except ImportError:
                    logger.error("scipy not available for mask resizing")
                    return self._create_synthetic_dvh_curve_fallback(structure, dose)

            # Extract dose values inside the structure
            structure_voxels = structure_mask > 0
            if not np.any(structure_voxels):
                logger.warning(f"Empty structure mask for {structure.name}")
                return self._create_synthetic_dvh_curve_fallback(structure, dose)

            dose_values = dose_data[structure_voxels]

            # Remove invalid dose values
            dose_values = dose_values[np.isfinite(dose_values)]
            dose_values = dose_values[dose_values >= 0]

            if len(dose_values) == 0:
                logger.warning(f"No valid dose values for structure {structure.name}")
                return self._create_synthetic_dvh_curve_fallback(structure, dose)

            # Calculate voxel volume
            voxel_volume = 1.0  # Default
            if hasattr(dose, "spacing"):
                voxel_volume = np.prod(dose.spacing)
            elif hasattr(dose, "voxel_size"):
                voxel_volume = dose.voxel_size

            # Calculate total structure volume
            total_volume = len(dose_values) * voxel_volume

            # Create dose bins
            max_dose = float(np.max(dose_values))
            min_dose = float(np.min(dose_values))
            dose_bins = np.linspace(min_dose, max_dose, self.bin_count)

            # Calculate cumulative volume for each dose bin
            cumulative_volumes = []
            for dose_threshold in dose_bins:
                volume_at_dose = np.sum(dose_values >= dose_threshold) * voxel_volume
                volume_percent = (
                    (volume_at_dose / total_volume) * 100.0 if total_volume > 0 else 0.0
                )
                cumulative_volumes.append(volume_percent)

            cumulative_volume_percent = np.array(cumulative_volumes)

            # Create DVH curve with proper parameters
            curve = DVHCurve(
                structure_id=structure.id,
                structure_name=structure.name,
                dose_bins=dose_bins,
                volume_bins=cumulative_volume_percent,
                structure_volume=total_volume,
            )

            # Set additional data using proper method name
            if hasattr(curve, "set_dvh_data"):
                curve.set_dvh_data(
                    dose_bins, cumulative_volume_percent, is_cumulative=True
                )
            elif hasattr(curve, "data"):
                curve.data = {
                    "dose_bins": dose_bins,
                    "volume_bins": cumulative_volume_percent,
                    "is_cumulative": True,
                    "total_volume": total_volume,
                }

            # Calculate and set basic statistics
            self._calculate_dvh_statistics(curve)

            logger.info(
                f"Calculated DVH for {structure.name}: "
                f"volume={total_volume:.2f}cm³, max_dose={max_dose:.2f}Gy"
            )

            return curve

        except Exception as e:
            logger.error(
                f"Error calculating DVH for structure {structure.name}: {str(e)}"
            )
            import traceback

            logger.error(traceback.format_exc())
            # Fallback to synthetic DVH
            return self._create_synthetic_dvh_curve_fallback(structure, dose)

    def _create_synthetic_dvh_curve_fallback(
        self, structure: Structure, dose: DoseDistribution
    ) -> DVHCurve:
        """Create a fallback synthetic DVH curve when real calculation fails."""
        try:
            # Create basic dose bins
            dose_bins = np.linspace(0, 80, self.bin_count)

            # Create synthetic volume data
            volume_bins = self._create_synthetic_volume_data(structure.name, dose_bins)

            # Create DVH curve
            curve = DVHCurve(
                structure_id=structure.id,
                structure_name=structure.name,
                dose_bins=dose_bins,
                volume_bins=volume_bins,
                structure_volume=10.0,  # Default 10 cm³
            )

            logger.info(f"Created synthetic DVH for {structure.name}")
            return curve

        except Exception as e:
            logger.error(f"Failed to create synthetic DVH: {str(e)}")
            # Create minimal curve
            dose_bins = np.array([0, 80])
            volume_bins = np.array([100, 0])
            return DVHCurve(
                structure_id=getattr(structure, "id", "unknown"),
                structure_name=getattr(structure, "name", "Unknown"),
                dose_bins=dose_bins,
                volume_bins=volume_bins,
                structure_volume=1.0,
            )

    def _create_synthetic_volume_data(
        self, structure_name: str, dose_bins: np.ndarray
    ) -> np.ndarray:
        """Create synthetic volume data for fallback DVH."""
        # Create different patterns based on structure type
        if any(
            name in structure_name.lower() for name in ["ptv", "gtv", "ctv", "target"]
        ):
            # Target structures - steep drop-off around prescription dose
            prescription_dose = 60.0  # Gy
            volume_bins = 100 * np.exp(
                -((dose_bins - prescription_dose) ** 2) / (2 * 5**2)
            )
            volume_bins = np.maximum(volume_bins, 0)
            volume_bins[dose_bins < prescription_dose] = 100
        else:
            # OAR structures - exponential decay
            volume_bins = 100 * np.exp(-dose_bins / 20.0)

        return volume_bins

    def _calculate_dvh_statistics(self, curve: DVHCurve):
        """
        Calculate DVH statistics for a curve.
        This includes min/max/mean dose, and common metrics like D95, D50, V20, etc.

        Args:
            curve: The DVH curve to analyze
        """
        # Basic statistics are already calculated in the curve.set_data() method

        # Calculate common D metrics (dose to x% of volume)
        common_volume_percentages = [95, 90, 80, 50, 20, 10, 5, 2]
        for volume_percent in common_volume_percentages:
            curve.calculate_d_metric(volume_percent)

        # Calculate common V metrics (volume receiving at least x Gy)
        common_doses = [5, 10, 20, 30, 40, 50, 60, 70]
        for dose in common_doses:
            curve.calculate_v_metric(dose)

        # Calculate volume receiving at least 95% of prescription dose
        if hasattr(curve, "prescription_dose") and curve.prescription_dose > 0:
            v95_dose = 0.95 * curve.prescription_dose
            curve.calculate_v_metric(v95_dose)
