"""
QuangTPS Dose Metrics Module

Module tính toán các metrics liều xạ trị.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from scipy import stats
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DoseStatistics:
    """Thống kê liều cơ bản."""

    mean_dose: float = 0.0
    median_dose: float = 0.0
    min_dose: float = 0.0
    max_dose: float = 0.0
    std_dose: float = 0.0
    percentile_95: float = 0.0
    percentile_5: float = 0.0
    volume_ml: float = 0.0


def calculate_dose_statistics(
    dose_distribution: np.ndarray,
    structure_mask: Optional[np.ndarray] = None,
    voxel_volume: float = 1.0,
) -> DoseStatistics:
    """
    Tính toán thống kê liều cơ bản.

    Args:
        dose_distribution: Array phân phối liều
        structure_mask: Mask cấu trúc (optional)
        voxel_volume: Thể tích voxel (ml)

    Returns:
        DoseStatistics: Thống kê liều
    """
    try:
        if structure_mask is not None:
            # Apply structure mask
            valid_doses = dose_distribution[structure_mask > 0]
            total_volume = np.sum(structure_mask > 0) * voxel_volume
        else:
            valid_doses = dose_distribution.flatten()
            total_volume = len(valid_doses) * voxel_volume

        # Remove NaN and infinite values
        valid_doses = valid_doses[np.isfinite(valid_doses)]

        if len(valid_doses) == 0:
            return DoseStatistics()

        return DoseStatistics(
            mean_dose=float(np.mean(valid_doses)),
            median_dose=float(np.median(valid_doses)),
            min_dose=float(np.min(valid_doses)),
            max_dose=float(np.max(valid_doses)),
            std_dose=float(np.std(valid_doses)),
            percentile_95=float(np.percentile(valid_doses, 95)),
            percentile_5=float(np.percentile(valid_doses, 5)),
            volume_ml=total_volume,
        )

    except Exception as e:
        logger.error(f"Error calculating dose statistics: {e}")
        return DoseStatistics()


def calculate_dose_volume_at_level(
    dose_distribution: np.ndarray,
    dose_level: float,
    structure_mask: Optional[np.ndarray] = None,
    voxel_volume: float = 1.0,
) -> float:
    """
    Tính thể tích nhận liều >= dose_level.

    Args:
        dose_distribution: Array phân phối liều
        dose_level: Mức liều (Gy)
        structure_mask: Mask cấu trúc (optional)
        voxel_volume: Thể tích voxel (ml)

    Returns:
        float: Thể tích (ml)
    """
    try:
        if structure_mask is not None:
            # Apply structure mask
            valid_doses = dose_distribution[structure_mask > 0]
        else:
            valid_doses = dose_distribution.flatten()

        # Count voxels with dose >= dose_level
        volume_voxels = np.sum(valid_doses >= dose_level)

        return float(volume_voxels * voxel_volume)

    except Exception as e:
        logger.error(f"Error calculating dose volume at level: {e}")
        return 0.0


def calculate_dose_at_volume(
    dose_distribution: np.ndarray,
    volume_percent: float,
    structure_mask: Optional[np.ndarray] = None,
) -> float:
    """
    Tính liều tại volume percent cụ thể (Dx%).

    Args:
        dose_distribution: Array phân phối liều
        volume_percent: Phần trăm volume (0-100)
        structure_mask: Mask cấu trúc (optional)

    Returns:
        float: Giá trị liều (Gy)
    """
    try:
        if structure_mask is not None:
            # Apply structure mask
            valid_doses = dose_distribution[structure_mask > 0]
        else:
            valid_doses = dose_distribution.flatten()

        # Remove zero and negative doses
        valid_doses = valid_doses[valid_doses > 0]

        if len(valid_doses) == 0:
            return 0.0

        # Sort doses in descending order
        sorted_doses = np.sort(valid_doses)[::-1]

        # Calculate index for volume percent
        volume_index = int((volume_percent / 100.0) * len(sorted_doses))
        volume_index = min(volume_index, len(sorted_doses) - 1)

        return float(sorted_doses[volume_index])

    except Exception as e:
        logger.error(f"Error calculating dose at volume: {e}")
        return 0.0


def calculate_volume_at_dose(
    dose_distribution: np.ndarray,
    dose_level: float,
    structure_mask: Optional[np.ndarray] = None,
) -> float:
    """
    Tính phần trăm volume nhận liều >= dose_level (Vx%).

    Args:
        dose_distribution: Array phân phối liều
        dose_level: Mức liều (Gy)
        structure_mask: Mask cấu trúc (optional)

    Returns:
        float: Phần trăm volume (%)
    """
    try:
        if structure_mask is not None:
            # Apply structure mask
            valid_doses = dose_distribution[structure_mask > 0]
            total_volume = np.sum(structure_mask > 0)
        else:
            valid_doses = dose_distribution.flatten()
            total_volume = len(valid_doses)

        if total_volume == 0:
            return 0.0

        # Count voxels with dose >= dose_level
        volume_at_dose = np.sum(valid_doses >= dose_level)

        # Calculate percentage
        volume_percent = (volume_at_dose / total_volume) * 100.0

        return float(volume_percent)

    except Exception as e:
        logger.error(f"Error calculating volume at dose: {e}")
        return 0.0


def calculate_mean_dose(
    dose_distribution: np.ndarray, structure_mask: Optional[np.ndarray] = None
) -> float:
    """
    Tính liều trung bình.

    Args:
        dose_distribution: Array phân phối liều
        structure_mask: Mask cấu trúc (optional)

    Returns:
        float: Liều trung bình (Gy)
    """
    try:
        if structure_mask is not None:
            # Apply structure mask
            valid_doses = dose_distribution[structure_mask > 0]
        else:
            valid_doses = dose_distribution.flatten()

        # Remove NaN and infinite values
        valid_doses = valid_doses[np.isfinite(valid_doses)]

        if len(valid_doses) == 0:
            return 0.0

        return float(np.mean(valid_doses))

    except Exception as e:
        logger.error(f"Error calculating mean dose: {e}")
        return 0.0


def calculate_max_dose(
    dose_distribution: np.ndarray, structure_mask: Optional[np.ndarray] = None
) -> float:
    """
    Tính liều tối đa.

    Args:
        dose_distribution: Array phân phối liều
        structure_mask: Mask cấu trúc (optional)

    Returns:
        float: Liều tối đa (Gy)
    """
    try:
        if structure_mask is not None:
            # Apply structure mask
            valid_doses = dose_distribution[structure_mask > 0]
        else:
            valid_doses = dose_distribution.flatten()

        # Remove NaN and infinite values
        valid_doses = valid_doses[np.isfinite(valid_doses)]

        if len(valid_doses) == 0:
            return 0.0

        return float(np.max(valid_doses))

    except Exception as e:
        logger.error(f"Error calculating max dose: {e}")
        return 0.0


def calculate_min_dose(
    dose_distribution: np.ndarray, structure_mask: Optional[np.ndarray] = None
) -> float:
    """
    Tính liều tối thiểu.

    Args:
        dose_distribution: Array phân phối liều
        structure_mask: Mask cấu trúc (optional)

    Returns:
        float: Liều tối thiểu (Gy)
    """
    try:
        if structure_mask is not None:
            # Apply structure mask
            valid_doses = dose_distribution[structure_mask > 0]
        else:
            valid_doses = dose_distribution.flatten()

        # Remove NaN and infinite values and zeros
        valid_doses = valid_doses[np.isfinite(valid_doses) & (valid_doses > 0)]

        if len(valid_doses) == 0:
            return 0.0

        return float(np.min(valid_doses))

    except Exception as e:
        logger.error(f"Error calculating min dose: {e}")
        return 0.0


def calculate_dose_gradient(
    dose_distribution: np.ndarray, spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> np.ndarray:
    """
    Tính gradient liều 3D.

    Args:
        dose_distribution: Array phân phối liều 3D
        spacing: Khoảng cách voxel (x, y, z)

    Returns:
        np.ndarray: Magnitude của gradient
    """
    try:
        # Calculate gradient in all 3 dimensions
        grad_z, grad_y, grad_x = np.gradient(
            dose_distribution, spacing[2], spacing[1], spacing[0]
        )

        # Calculate magnitude
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)

        return gradient_magnitude

    except Exception as e:
        logger.error(f"Error calculating dose gradient: {e}")
        return np.zeros_like(dose_distribution)


def calculate_dose_conformity_index(
    dose_distribution: np.ndarray,
    target_mask: np.ndarray,
    prescription_dose: float,
    isodose_level: float = 0.95,
) -> float:
    """
    Tính chỉ số conformity (CI).

    Args:
        dose_distribution: Array phân phối liều
        target_mask: Mask mục tiêu
        prescription_dose: Liều chỉ định
        isodose_level: Mức isodose (fraction of prescription)

    Returns:
        float: Conformity index
    """
    try:
        isodose_value = prescription_dose * isodose_level

        # Volume nhận isodose
        v_isodose = np.sum(dose_distribution >= isodose_value)

        # Volume target nhận isodose
        v_target_isodose = np.sum(
            (dose_distribution >= isodose_value) & (target_mask > 0)
        )

        if v_target_isodose == 0:
            return 0.0

        # CI = (Volume target trong isodose)^2 / (Volume target * Volume isodose)
        v_target = np.sum(target_mask > 0)

        if v_target == 0 or v_isodose == 0:
            return 0.0

        ci = (v_target_isodose**2) / (v_target * v_isodose)

        return float(ci)

    except Exception as e:
        logger.error(f"Error calculating conformity index: {e}")
        return 0.0


def calculate_dose_homogeneity_index(
    dose_distribution: np.ndarray, target_mask: np.ndarray, prescription_dose: float
) -> float:
    """
    Tính chỉ số homogeneity (HI).

    Args:
        dose_distribution: Array phân phối liều
        target_mask: Mask mục tiêu
        prescription_dose: Liều chỉ định

    Returns:
        float: Homogeneity index
    """
    try:
        # Liều trong target
        target_doses = dose_distribution[target_mask > 0]

        if len(target_doses) == 0:
            return 0.0

        # D2% và D98%
        d2 = np.percentile(target_doses, 98)
        d98 = np.percentile(target_doses, 2)

        # HI = (D2% - D98%) / D_prescription
        hi = (d2 - d98) / prescription_dose

        return float(hi)

    except Exception as e:
        logger.error(f"Error calculating homogeneity index: {e}")
        return 0.0


def calculate_dose_coverage(
    dose_distribution: np.ndarray,
    target_mask: np.ndarray,
    prescription_dose: float,
    coverage_level: float = 0.95,
) -> float:
    """
    Tính độ phủ liều của target.

    Args:
        dose_distribution: Array phân phối liều
        target_mask: Mask mục tiêu
        prescription_dose: Liều chỉ định
        coverage_level: Mức độ phủ (fraction of prescription)

    Returns:
        float: Coverage percentage (%)
    """
    try:
        target_doses = dose_distribution[target_mask > 0]

        if len(target_doses) == 0:
            return 0.0

        coverage_dose = prescription_dose * coverage_level
        covered_voxels = np.sum(target_doses >= coverage_dose)
        total_voxels = len(target_doses)

        coverage_percent = (covered_voxels / total_voxels) * 100.0

        return float(coverage_percent)

    except Exception as e:
        logger.error(f"Error calculating dose coverage: {e}")
        return 0.0


def calculate_integral_dose(
    dose_distribution: np.ndarray,
    structure_mask: Optional[np.ndarray] = None,
    voxel_volume: float = 1.0,
) -> float:
    """
    Tính integral dose (tổng liều * thể tích).

    Args:
        dose_distribution: Array phân phối liều
        structure_mask: Mask cấu trúc (optional)
        voxel_volume: Thể tích voxel (ml)

    Returns:
        float: Integral dose (Gy·ml)
    """
    try:
        if structure_mask is not None:
            # Apply structure mask
            valid_doses = dose_distribution[structure_mask > 0]
        else:
            valid_doses = dose_distribution.flatten()

        # Remove NaN and infinite values
        valid_doses = valid_doses[np.isfinite(valid_doses)]

        if len(valid_doses) == 0:
            return 0.0

        integral_dose = np.sum(valid_doses) * voxel_volume

        return float(integral_dose)

    except Exception as e:
        logger.error(f"Error calculating integral dose: {e}")
        return 0.0


__all__ = [
    "DoseStatistics",
    "calculate_dose_statistics",
    "calculate_dose_volume_at_level",
    "calculate_dose_at_volume",
    "calculate_volume_at_dose",
    "calculate_mean_dose",
    "calculate_max_dose",
    "calculate_min_dose",
    "calculate_dose_gradient",
    "calculate_dose_conformity_index",
    "calculate_dose_homogeneity_index",
    "calculate_dose_coverage",
    "calculate_integral_dose",
]

# Tạo alias để tương thích với import
calculate_conformity_index = calculate_dose_conformity_index
calculate_homogeneity_index = calculate_dose_homogeneity_index
calculate_coverage_index = calculate_dose_coverage
