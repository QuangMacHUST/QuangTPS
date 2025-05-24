"""
QuangTPS DVH Metrics Module

Module tính toán các metrics liên quan đến Dose Volume Histogram.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from scipy import interpolate, integrate

logger = logging.getLogger(__name__)


@dataclass
class DVHCurve:
    """Biểu diễn một DVH curve."""

    dose_bins: np.ndarray
    volume_data: np.ndarray
    structure_name: str = ""
    volume_type: str = "cumulative"  # cumulative or differential
    total_volume: float = 0.0


@dataclass
class DVHStatistics:
    """Thống kê DVH."""

    mean_dose: float = 0.0
    median_dose: float = 0.0
    modal_dose: float = 0.0
    std_dose: float = 0.0
    integral_dose: float = 0.0

    # Dose statistics
    d_max: float = 0.0
    d_min: float = 0.0
    d_mean: float = 0.0

    # Volume statistics at dose levels
    v_5: float = 0.0  # Volume receiving >= 5 Gy
    v_10: float = 0.0  # Volume receiving >= 10 Gy
    v_20: float = 0.0  # Volume receiving >= 20 Gy
    v_30: float = 0.0  # Volume receiving >= 30 Gy

    # Dose statistics at volume levels
    d_95: float = 0.0  # Dose to 95% volume
    d_50: float = 0.0  # Dose to 50% volume
    d_5: float = 0.0  # Dose to 5% volume
    d_2: float = 0.0  # Dose to 2% volume


def create_dvh_from_dose_distribution(
    dose_distribution: np.ndarray,
    structure_mask: np.ndarray,
    dose_bins: Optional[np.ndarray] = None,
    structure_name: str = "",
    volume_type: str = "cumulative",
) -> DVHCurve:
    """
    Tạo DVH curve từ phân phối liều và mask cấu trúc.

    Args:
        dose_distribution: Array phân phối liều
        structure_mask: Mask cấu trúc
        dose_bins: Bins liều (optional)
        structure_name: Tên cấu trúc
        volume_type: Loại volume (cumulative/differential)

    Returns:
        DVHCurve: DVH curve
    """
    try:
        # Lấy liều trong structure
        structure_doses = dose_distribution[structure_mask > 0]

        if len(structure_doses) == 0:
            return DVHCurve(
                dose_bins=np.array([]),
                volume_data=np.array([]),
                structure_name=structure_name,
                volume_type=volume_type,
                total_volume=0.0,
            )

        # Tạo dose bins nếu không có
        if dose_bins is None:
            max_dose = np.max(structure_doses)
            dose_bins = np.linspace(0, max_dose, 1000)

        # Tính volume data
        if volume_type == "cumulative":
            # Cumulative DVH
            volume_data = np.array(
                [
                    np.sum(structure_doses >= dose) / len(structure_doses) * 100.0
                    for dose in dose_bins
                ]
            )
        else:
            # Differential DVH
            hist, _ = np.histogram(structure_doses, bins=dose_bins)
            volume_data = hist / len(structure_doses) * 100.0
            dose_bins = dose_bins[:-1]  # Remove last bin edge

        total_volume = len(structure_doses)

        return DVHCurve(
            dose_bins=dose_bins,
            volume_data=volume_data,
            structure_name=structure_name,
            volume_type=volume_type,
            total_volume=total_volume,
        )

    except Exception as e:
        logger.error(f"Error creating DVH from dose distribution: {e}")
        return DVHCurve(
            dose_bins=np.array([]),
            volume_data=np.array([]),
            structure_name=structure_name,
            volume_type=volume_type,
            total_volume=0.0,
        )


def calculate_dose_at_volume(dvh_curve: DVHCurve, volume_percent: float) -> float:
    """
    Tính dose tại volume percent cụ thể (Dx%).

    Args:
        dvh_curve: DVH curve
        volume_percent: Phần trăm volume (0-100)

    Returns:
        float: Dose value (Gy)
    """
    try:
        if len(dvh_curve.dose_bins) == 0 or len(dvh_curve.volume_data) == 0:
            return 0.0

        if dvh_curve.volume_type != "cumulative":
            logger.warning("Dose at volume calculation works best with cumulative DVH")

        # Tìm dose tương ứng với volume percent
        valid_indices = dvh_curve.volume_data >= volume_percent

        if not np.any(valid_indices):
            return 0.0

        # Interpolation để tìm dose chính xác
        if np.all(valid_indices):
            return float(dvh_curve.dose_bins[0])

        # Find crossing point
        crossing_idx = np.where(dvh_curve.volume_data < volume_percent)[0]
        if len(crossing_idx) == 0:
            return float(dvh_curve.dose_bins[-1])

        idx = crossing_idx[0]
        if idx == 0:
            return float(dvh_curve.dose_bins[0])

        # Linear interpolation
        x1, x2 = dvh_curve.dose_bins[idx - 1], dvh_curve.dose_bins[idx]
        y1, y2 = dvh_curve.volume_data[idx - 1], dvh_curve.volume_data[idx]

        if y1 == y2:
            dose = x1
        else:
            dose = x1 + (volume_percent - y1) * (x2 - x1) / (y2 - y1)

        return float(dose)

    except Exception as e:
        logger.error(f"Error calculating dose at volume: {e}")
        return 0.0


def calculate_volume_at_dose(dvh_curve: DVHCurve, dose_level: float) -> float:
    """
    Tính volume percent tại dose level cụ thể (Vx%).

    Args:
        dvh_curve: DVH curve
        dose_level: Mức dose (Gy)

    Returns:
        float: Volume percent (%)
    """
    try:
        if len(dvh_curve.dose_bins) == 0 or len(dvh_curve.volume_data) == 0:
            return 0.0

        if dvh_curve.volume_type != "cumulative":
            logger.warning("Volume at dose calculation works best with cumulative DVH")

        # Tìm volume tương ứng với dose level
        valid_indices = dvh_curve.dose_bins <= dose_level

        if not np.any(valid_indices):
            return 0.0

        if np.all(valid_indices):
            return float(dvh_curve.volume_data[-1])

        # Find crossing point
        crossing_idx = np.where(dvh_curve.dose_bins > dose_level)[0]
        if len(crossing_idx) == 0:
            return float(dvh_curve.volume_data[-1])

        idx = crossing_idx[0]
        if idx == 0:
            return float(dvh_curve.volume_data[0])

        # Linear interpolation
        x1, x2 = dvh_curve.dose_bins[idx - 1], dvh_curve.dose_bins[idx]
        y1, y2 = dvh_curve.volume_data[idx - 1], dvh_curve.volume_data[idx]

        if x1 == x2:
            volume = y1
        else:
            volume = y1 + (dose_level - x1) * (y2 - y1) / (x2 - x1)

        return float(volume)

    except Exception as e:
        logger.error(f"Error calculating volume at dose: {e}")
        return 0.0


def calculate_mean_dose_from_dvh(dvh_curve: DVHCurve) -> float:
    """
    Tính mean dose từ DVH curve.

    Args:
        dvh_curve: DVH curve

    Returns:
        float: Mean dose (Gy)
    """
    try:
        if len(dvh_curve.dose_bins) == 0 or len(dvh_curve.volume_data) == 0:
            return 0.0

        if dvh_curve.volume_type == "cumulative":
            # Convert to differential for integration
            diff_volumes = np.diff(dvh_curve.volume_data)
            diff_doses = dvh_curve.dose_bins[1:]

            # Ensure we have valid data
            if len(diff_volumes) == 0:
                return 0.0

            # Mean dose = integral of dose * volume / total volume
            total_dose_volume = np.sum(diff_doses * np.abs(diff_volumes))
            total_volume = np.sum(np.abs(diff_volumes))

            if total_volume == 0:
                return 0.0

            mean_dose = total_dose_volume / total_volume

        else:
            # Differential DVH
            total_dose_volume = np.sum(dvh_curve.dose_bins * dvh_curve.volume_data)
            total_volume = np.sum(dvh_curve.volume_data)

            if total_volume == 0:
                return 0.0

            mean_dose = total_dose_volume / total_volume

        return float(mean_dose)

    except Exception as e:
        logger.error(f"Error calculating mean dose from DVH: {e}")
        return 0.0


def calculate_integral_dose_from_dvh(
    dvh_curve: DVHCurve, voxel_volume: float = 1.0
) -> float:
    """
    Tính integral dose từ DVH curve.

    Args:
        dvh_curve: DVH curve
        voxel_volume: Thể tích voxel (ml)

    Returns:
        float: Integral dose (Gy·ml)
    """
    try:
        if len(dvh_curve.dose_bins) == 0 or len(dvh_curve.volume_data) == 0:
            return 0.0

        mean_dose = calculate_mean_dose_from_dvh(dvh_curve)
        total_volume_ml = dvh_curve.total_volume * voxel_volume

        integral_dose = mean_dose * total_volume_ml

        return float(integral_dose)

    except Exception as e:
        logger.error(f"Error calculating integral dose from DVH: {e}")
        return 0.0


def calculate_dvh_statistics(
    dvh_curve: DVHCurve,
    custom_dose_levels: Optional[List[float]] = None,
    custom_volume_levels: Optional[List[float]] = None,
) -> DVHStatistics:
    """
    Tính toán comprehensive DVH statistics.

    Args:
        dvh_curve: DVH curve
        custom_dose_levels: Custom dose levels for Vx calculation
        custom_volume_levels: Custom volume levels for Dx calculation

    Returns:
        DVHStatistics: Comprehensive DVH statistics
    """
    try:
        if len(dvh_curve.dose_bins) == 0 or len(dvh_curve.volume_data) == 0:
            return DVHStatistics()

        # Basic statistics
        mean_dose = calculate_mean_dose_from_dvh(dvh_curve)

        # For detailed statistics, we need to reconstruct dose distribution
        # This is an approximation from DVH
        if dvh_curve.volume_type == "cumulative":
            # Approximate median from cumulative DVH
            median_dose = calculate_dose_at_volume(dvh_curve, 50.0)

            # Max and min dose
            d_max = dvh_curve.dose_bins[np.where(dvh_curve.volume_data > 0)[0][-1]]
            d_min = (
                dvh_curve.dose_bins[np.where(dvh_curve.volume_data >= 100.0)[0][0]]
                if np.any(dvh_curve.volume_data >= 100.0)
                else 0.0
            )
        else:
            # For differential DVH
            weighted_doses = dvh_curve.dose_bins * dvh_curve.volume_data
            total_weight = np.sum(dvh_curve.volume_data)

            if total_weight > 0:
                median_dose = np.sum(weighted_doses) / total_weight
            else:
                median_dose = 0.0

            d_max = dvh_curve.dose_bins[-1] if len(dvh_curve.dose_bins) > 0 else 0.0
            d_min = dvh_curve.dose_bins[0] if len(dvh_curve.dose_bins) > 0 else 0.0

        # Standard dose levels for volume calculation
        dose_levels = custom_dose_levels or [5.0, 10.0, 20.0, 30.0]
        volume_levels = custom_volume_levels or [95.0, 50.0, 5.0, 2.0]

        # Calculate Vx values
        v_values = {}
        for dose in dose_levels:
            v_values[f"v_{int(dose)}"] = calculate_volume_at_dose(dvh_curve, dose)

        # Calculate Dx values
        d_values = {}
        for volume in volume_levels:
            d_values[f"d_{int(volume)}"] = calculate_dose_at_volume(dvh_curve, volume)

        return DVHStatistics(
            mean_dose=mean_dose,
            median_dose=median_dose,
            modal_dose=median_dose,  # Approximation
            std_dose=0.0,  # Difficult to calculate from DVH
            integral_dose=calculate_integral_dose_from_dvh(dvh_curve),
            d_max=d_max,
            d_min=d_min,
            d_mean=mean_dose,
            v_5=v_values.get("v_5", 0.0),
            v_10=v_values.get("v_10", 0.0),
            v_20=v_values.get("v_20", 0.0),
            v_30=v_values.get("v_30", 0.0),
            d_95=d_values.get("d_95", 0.0),
            d_50=d_values.get("d_50", 0.0),
            d_5=d_values.get("d_5", 0.0),
            d_2=d_values.get("d_2", 0.0),
        )

    except Exception as e:
        logger.error(f"Error calculating DVH statistics: {e}")
        return DVHStatistics()


def compare_dvh_curves(
    dvh1: DVHCurve,
    dvh2: DVHCurve,
    dose_tolerance: float = 2.0,  # Gy
    volume_tolerance: float = 2.0,  # %
) -> Dict[str, Any]:
    """
    So sánh hai DVH curves.

    Args:
        dvh1: DVH curve 1
        dvh2: DVH curve 2
        dose_tolerance: Tolerance cho dose (Gy)
        volume_tolerance: Tolerance cho volume (%)

    Returns:
        Dict[str, Any]: Kết quả so sánh
    """
    try:
        comparison = {
            "structures": [dvh1.structure_name, dvh2.structure_name],
            "mean_dose_diff": 0.0,
            "max_dose_diff": 0.0,
            "volume_differences": {},
            "dose_differences": {},
            "within_tolerance": True,
            "max_volume_deviation": 0.0,
            "max_dose_deviation": 0.0,
        }

        # Calculate statistics for both DVH curves
        stats1 = calculate_dvh_statistics(dvh1)
        stats2 = calculate_dvh_statistics(dvh2)

        # Compare mean and max doses
        comparison["mean_dose_diff"] = abs(stats1.mean_dose - stats2.mean_dose)
        comparison["max_dose_diff"] = abs(stats1.d_max - stats2.d_max)

        # Compare volumes at standard dose levels
        dose_levels = [5.0, 10.0, 20.0, 30.0, 40.0, 50.0]
        max_volume_dev = 0.0

        for dose in dose_levels:
            v1 = calculate_volume_at_dose(dvh1, dose)
            v2 = calculate_volume_at_dose(dvh2, dose)
            volume_diff = abs(v1 - v2)

            comparison["volume_differences"][f"V{int(dose)}"] = volume_diff
            max_volume_dev = max(max_volume_dev, volume_diff)

        # Compare doses at standard volume levels
        volume_levels = [95.0, 50.0, 5.0, 2.0]
        max_dose_dev = 0.0

        for volume in volume_levels:
            d1 = calculate_dose_at_volume(dvh1, volume)
            d2 = calculate_dose_at_volume(dvh2, volume)
            dose_diff = abs(d1 - d2)

            comparison["dose_differences"][f"D{int(volume)}"] = dose_diff
            max_dose_dev = max(max_dose_dev, dose_diff)

        comparison["max_volume_deviation"] = max_volume_dev
        comparison["max_dose_deviation"] = max_dose_dev

        # Check if within tolerance
        comparison["within_tolerance"] = (
            max_volume_dev <= volume_tolerance and max_dose_dev <= dose_tolerance
        )

        return comparison

    except Exception as e:
        logger.error(f"Error comparing DVH curves: {e}")
        return {
            "structures": [dvh1.structure_name, dvh2.structure_name],
            "error": str(e),
        }


def interpolate_dvh_curve(
    dvh_curve: DVHCurve, new_dose_bins: np.ndarray, interpolation_method: str = "linear"
) -> DVHCurve:
    """
    Interpolate DVH curve lên dose bins mới.

    Args:
        dvh_curve: DVH curve gốc
        new_dose_bins: Dose bins mới
        interpolation_method: Phương pháp interpolation

    Returns:
        DVHCurve: DVH curve đã được interpolate
    """
    try:
        if len(dvh_curve.dose_bins) == 0 or len(dvh_curve.volume_data) == 0:
            return DVHCurve(
                dose_bins=new_dose_bins,
                volume_data=np.zeros_like(new_dose_bins),
                structure_name=dvh_curve.structure_name,
                volume_type=dvh_curve.volume_type,
                total_volume=dvh_curve.total_volume,
            )

        # Create interpolation function
        if interpolation_method == "cubic":
            interp_func = interpolate.interp1d(
                dvh_curve.dose_bins,
                dvh_curve.volume_data,
                kind="cubic",
                bounds_error=False,
                fill_value=(dvh_curve.volume_data[0], dvh_curve.volume_data[-1]),
            )
        else:
            interp_func = interpolate.interp1d(
                dvh_curve.dose_bins,
                dvh_curve.volume_data,
                kind="linear",
                bounds_error=False,
                fill_value=(dvh_curve.volume_data[0], dvh_curve.volume_data[-1]),
            )

        # Interpolate to new dose bins
        new_volume_data = interp_func(new_dose_bins)

        return DVHCurve(
            dose_bins=new_dose_bins,
            volume_data=new_volume_data,
            structure_name=dvh_curve.structure_name,
            volume_type=dvh_curve.volume_type,
            total_volume=dvh_curve.total_volume,
        )

    except Exception as e:
        logger.error(f"Error interpolating DVH curve: {e}")
        return dvh_curve


def convert_cumulative_to_differential(dvh_curve: DVHCurve) -> DVHCurve:
    """
    Chuyển đổi cumulative DVH thành differential DVH.

    Args:
        dvh_curve: Cumulative DVH curve

    Returns:
        DVHCurve: Differential DVH curve
    """
    try:
        if dvh_curve.volume_type != "cumulative":
            logger.warning("DVH curve is not cumulative")
            return dvh_curve

        if len(dvh_curve.dose_bins) < 2:
            return dvh_curve

        # Calculate differential volumes
        diff_volumes = -np.diff(
            dvh_curve.volume_data
        )  # Negative because cumulative decreases
        diff_dose_bins = dvh_curve.dose_bins[:-1]  # Remove last bin

        # Ensure positive values
        diff_volumes = np.maximum(diff_volumes, 0.0)

        return DVHCurve(
            dose_bins=diff_dose_bins,
            volume_data=diff_volumes,
            structure_name=dvh_curve.structure_name,
            volume_type="differential",
            total_volume=dvh_curve.total_volume,
        )

    except Exception as e:
        logger.error(f"Error converting cumulative to differential DVH: {e}")
        return dvh_curve


def convert_differential_to_cumulative(dvh_curve: DVHCurve) -> DVHCurve:
    """
    Chuyển đổi differential DVH thành cumulative DVH.

    Args:
        dvh_curve: Differential DVH curve

    Returns:
        DVHCurve: Cumulative DVH curve
    """
    try:
        if dvh_curve.volume_type != "differential":
            logger.warning("DVH curve is not differential")
            return dvh_curve

        # Calculate cumulative volumes by reverse integration
        cumulative_volumes = np.cumsum(dvh_curve.volume_data[::-1])[::-1]

        return DVHCurve(
            dose_bins=dvh_curve.dose_bins,
            volume_data=cumulative_volumes,
            structure_name=dvh_curve.structure_name,
            volume_type="cumulative",
            total_volume=dvh_curve.total_volume,
        )

    except Exception as e:
        logger.error(f"Error converting differential to cumulative DVH: {e}")
        return dvh_curve


__all__ = [
    "DVHCurve",
    "DVHStatistics",
    "create_dvh_from_dose_distribution",
    "calculate_dose_at_volume",
    "calculate_volume_at_dose",
    "calculate_mean_dose_from_dvh",
    "calculate_integral_dose_from_dvh",
    "calculate_dvh_statistics",
    "compare_dvh_curves",
    "interpolate_dvh_curve",
    "convert_cumulative_to_differential",
    "convert_differential_to_cumulative",
]
