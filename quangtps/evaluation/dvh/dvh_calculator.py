#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module for calculating Dose-Volume Histograms (DVH) in the QuangTPS radiotherapy treatment planning system.

This module provides utilities for calculating and analyzing dose-volume histograms
which are essential for evaluating radiotherapy treatment plans.
"""

import numpy as np
import SimpleITK as sitk
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from dataclasses import dataclass, field
from enum import Enum

from quangtps.dose.dose_grid import DoseGrid
from quangtps.core.logging import get_logger

logger = get_logger(__name__)


class DVHType(Enum):
    """Loại DVH."""

    CUMULATIVE = "cumulative"
    DIFFERENTIAL = "differential"


class VolumeUnits(Enum):
    """Đơn vị thể tích."""

    PERCENT = "percent"
    CC = "cc"


class DoseUnits(Enum):
    """Đơn vị liều."""

    GY = "Gy"
    CGY = "cGy"
    PERCENT = "percent"


@dataclass
class DVHPoint:
    """
    Represents a point in a Dose-Volume Histogram.

    Attributes
    ----------
    dose : float
        Dose value in Gy
    volume : float
        Volume in cc or as a percentage of the structure volume
    """

    dose: float
    volume: float


@dataclass
class DVHData:
    """
    Container for DVH data.

    Attributes
    ----------
    dose_bins : np.ndarray
        Array of dose values
    volume_bins : np.ndarray
        Array of volume values corresponding to dose_bins
    structure_name : str
        Name of the structure
    structure_volume : float
        Total volume of the structure in cc
    max_dose : float
        Maximum dose in the structure
    mean_dose : float
        Mean dose in the structure
    min_dose : float
        Minimum dose in the structure (in non-zero voxels)
    d_x : Dict[float, float]
        Dictionary of Dx values (dose in Gy covering x% of the volume)
    v_x : Dict[float, float]
        Dictionary of Vx values (volume in cc or % receiving at least x Gy)
    is_cumulative : bool
        Whether the DVH is cumulative or differential
    """

    dose_bins: np.ndarray
    volume_bins: np.ndarray
    structure_name: str
    structure_volume: float
    max_dose: float = 0.0
    mean_dose: float = 0.0
    min_dose: float = 0.0
    d_x: Optional[Dict[float, float]] = None
    v_x: Optional[Dict[float, float]] = None
    is_cumulative: bool = True

    def __post_init__(self):
        """Initialize dictionaries if not provided."""
        if self.d_x is None:
            self.d_x = {}
        if self.v_x is None:
            self.v_x = {}

        # Compute basic statistics if not already computed
        if self.max_dose == 0.0 and len(self.dose_bins) > 0:
            self.max_dose = np.max(self.dose_bins)

        # Min dose is minimum non-zero dose
        if self.min_dose == 0.0 and len(self.dose_bins) > 0:
            non_zero_doses = self.dose_bins[self.dose_bins > 0]
            if len(non_zero_doses) > 0:
                self.min_dose = np.min(non_zero_doses)

    @classmethod
    def from_raw_data(
        cls,
        structure_name: str,
        dose_bins: List[float],
        volume_bins: List[float],
        structure_volume: float,
        is_cumulative: bool = True,
    ) -> "DVHData":
        """
        Tạo DVHData từ raw data.

        Parameters
        ----------
        structure_name : str
            Tên cấu trúc
        dose_bins : List[float]
            Danh sách liều
        volume_bins : List[float]
            Danh sách thể tích
        structure_volume : float
            Tổng thể tích cấu trúc
        is_cumulative : bool
            DVH tích lũy hay vi phân

        Returns
        -------
        DVHData
            Đối tượng DVHData
        """
        return cls(
            dose_bins=np.array(dose_bins),
            volume_bins=np.array(volume_bins),
            structure_name=structure_name,
            structure_volume=structure_volume,
            is_cumulative=is_cumulative,
        )

    def get_dx(self, percent_volume: float) -> float:
        """
        Get the dose (Gy) that covers a specified percentage of the structure volume.

        Parameters
        ----------
        percent_volume : float
            Percentage of volume (0-100)

        Returns
        -------
        float
            Dose in Gy covering percent_volume of the structure
        """
        if not self.is_cumulative:
            raise ValueError("Dx values are only defined for cumulative DVHs")

        if percent_volume in self.d_x:
            return self.d_x[percent_volume]

        if percent_volume < 0 or percent_volume > 100:
            raise ValueError("Percentage must be between 0 and 100")

        if len(self.dose_bins) == 0:
            return 0.0

        # Normalize volume bins to percent if needed
        vol_percent = self.volume_bins
        if np.max(vol_percent) > 1.1:  # Already in percent (0-100)
            pass
        else:  # In fraction (0-1), convert to percent
            vol_percent = vol_percent * 100

        # For cumulative DVH, volume typically decreases with dose
        # Interpolate to find dose at the specified volume
        if vol_percent[0] < vol_percent[-1]:  # If not decreasing, reverse
            dose_interp = np.interp(percent_volume, vol_percent, self.dose_bins)
        else:
            dose_interp = np.interp(
                percent_volume, vol_percent[::-1], self.dose_bins[::-1]
            )

        # Cache result
        self.d_x[percent_volume] = dose_interp
        return dose_interp

    def get_vx(self, dose: float, percent: bool = True) -> float:
        """
        Get the volume receiving at least the specified dose.

        Parameters
        ----------
        dose : float
            Dose threshold in Gy
        percent : bool
            If True, return volume as percentage of structure volume
            If False, return absolute volume in cc

        Returns
        -------
        float
            Volume (in cc or %) receiving at least the specified dose
        """
        if not self.is_cumulative:
            raise ValueError("Vx values are only defined for cumulative DVHs")

        # Generate a key that combines dose and units
        key = (dose, percent)
        if key in self.v_x:
            return self.v_x[key]

        if len(self.dose_bins) == 0:
            return 0.0

        # Find the volume receiving at least the specified dose
        # For cumulative DVH, interpolate dose vs volume curve
        if np.max(self.volume_bins) > 1.1:  # In percent (0-100)
            vol_bins = self.volume_bins
        else:  # In fraction (0-1)
            vol_bins = (
                self.volume_bins * 100
                if percent
                else self.volume_bins * self.structure_volume
            )

        # Interpolate to find volume at the specified dose
        if vol_bins[0] > vol_bins[-1]:  # Decreasing with dose (typical cumulative DVH)
            volume_interp = np.interp(dose, self.dose_bins, vol_bins)
        else:  # If not decreasing, reverse
            volume_interp = np.interp(dose, self.dose_bins[::-1], vol_bins[::-1])

        # Cache result
        self.v_x[key] = volume_interp
        return volume_interp


class DVHCalculator:
    """
    Calculator for Dose-Volume Histograms (DVH).

    This class provides methods for calculating and analyzing DVHs
    for radiation therapy structures.
    """

    def __init__(self, num_bins: int = 1000):
        """
        Initialize DVH calculator.

        Parameters
        ----------
        num_bins : int
            Number of bins to use for the DVH calculation
        """
        self.num_bins = num_bins

    def calculate_dvh(
        self, dose_image: sitk.Image, roi_mask: sitk.Image, cumulative: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate DVH for a single structure.

        Parameters
        ----------
        dose_image : sitk.Image
            Dose distribution as a SimpleITK image
        roi_mask : sitk.Image
            Binary mask of the structure of interest
        cumulative : bool
            If True, calculate cumulative DVH; otherwise, differential

        Returns
        -------
        tuple
            Dose bins and volume bins
        """
        # Make sure images have the same size and spacing
        if dose_image.GetSize() != roi_mask.GetSize():
            raise ValueError("Dose image and ROI mask must have the same size")

        # Convert to numpy arrays
        dose_array = sitk.GetArrayFromImage(dose_image)
        mask_array = sitk.GetArrayFromImage(roi_mask)

        # Extract doses within the ROI
        roi_doses = dose_array[mask_array > 0]

        if len(roi_doses) == 0:
            logger.warning("No voxels in ROI mask")
            return np.array([]), np.array([])

        # Calculate min and max dose
        min_dose = np.min(roi_doses)
        max_dose = np.max(roi_doses)

        # Create dose bins
        if min_dose == max_dose:
            dose_bins = np.array([min_dose])
            vol_bins = np.array([1.0])
        else:
            # Create histogram
            dose_bins = np.linspace(0, max_dose * 1.05, self.num_bins)
            hist, bin_edges = np.histogram(roi_doses, bins=dose_bins)

            # Calculate volume in each bin
            voxel_volume = np.prod(dose_image.GetSpacing()) / 1000  # in cc
            vol_bins = hist * voxel_volume

            # Use bin centers for dose bins
            dose_bins = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Convert to cumulative if requested
        if cumulative:
            vol_bins = np.cumsum(vol_bins[::-1])[::-1]

        # Normalize volume to percentage of total
        total_volume = len(roi_doses) * np.prod(dose_image.GetSpacing()) / 1000  # in cc
        vol_bins = vol_bins / total_volume

        return dose_bins, vol_bins

    def calculate_dvh_data(
        self,
        dose_image: sitk.Image,
        roi_mask: sitk.Image,
        structure_name: str,
        cumulative: bool = True,
    ) -> DVHData:
        """
        Calculate DVH data structure with comprehensive metrics.

        Parameters
        ----------
        dose_image : sitk.Image
            Dose distribution as a SimpleITK image
        roi_mask : sitk.Image
            Binary mask of the structure of interest
        structure_name : str
            Name of the structure
        cumulative : bool
            If True, calculate cumulative DVH; otherwise, differential

        Returns
        -------
        DVHData
            Complete DVH data structure with metrics
        """
        # Calculate basic DVH
        dose_bins, volume_bins = self.calculate_dvh(dose_image, roi_mask, cumulative)

        # Calculate structure volume
        mask_array = sitk.GetArrayFromImage(roi_mask)
        voxel_volume = np.prod(dose_image.GetSpacing()) / 1000  # in cc
        structure_volume = np.sum(mask_array > 0) * voxel_volume

        # Calculate dose statistics
        dose_array = sitk.GetArrayFromImage(dose_image)
        roi_doses = dose_array[mask_array > 0]

        if len(roi_doses) == 0:
            logger.warning(f"No voxels in structure: {structure_name}")
            max_dose = 0.0
            mean_dose = 0.0
            min_dose = 0.0
        else:
            max_dose = np.max(roi_doses)
            mean_dose = np.mean(roi_doses)
            # Min dose is minimum non-zero dose
            non_zero_doses = roi_doses[roi_doses > 0]
            min_dose = np.min(non_zero_doses) if len(non_zero_doses) > 0 else 0.0

        # Create DVH data object
        dvh_data = DVHData(
            dose_bins=dose_bins,
            volume_bins=volume_bins,
            structure_name=structure_name,
            structure_volume=structure_volume,
            max_dose=max_dose,
            mean_dose=mean_dose,
            min_dose=min_dose,
            is_cumulative=cumulative,
        )

        # Pre-calculate common D-x values
        for percent in [1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 98, 99]:
            if len(dose_bins) > 0:
                dvh_data.d_x[percent] = dvh_data.get_dx(percent)

        # Pre-calculate common V-x values (both absolute and percentage)
        dose_steps = np.linspace(0, max_dose, 20) if max_dose > 0 else [0]
        for dose in dose_steps:
            if len(dose_bins) > 0:
                dvh_data.v_x[(dose, True)] = dvh_data.get_vx(dose, True)
                dvh_data.v_x[(dose, False)] = dvh_data.get_vx(dose, False)

        return dvh_data

    def calculate_multiple_dvhs(
        self,
        dose_image: sitk.Image,
        structures: Dict[str, sitk.Image],
        cumulative: bool = True,
    ) -> Dict[str, DVHData]:
        """
        Calculate DVH data for multiple structures.

        Parameters
        ----------
        dose_image : sitk.Image
            Dose distribution as a SimpleITK image
        structures : dict
            Dictionary of structure masks with names as keys
        cumulative : bool
            If True, calculate cumulative DVH; otherwise, differential

        Returns
        -------
        dict
            Dictionary of DVH data with structure names as keys
        """
        result = {}
        for name, mask in structures.items():
            logger.info(f"Calculating DVH for structure: {name}")
            result[name] = self.calculate_dvh_data(dose_image, mask, name, cumulative)

        return result

    def get_common_metrics(self, dvh_data: DVHData) -> Dict[str, float]:
        """
        Extract common DVH metrics from a DVH data object.

        Parameters
        ----------
        dvh_data : DVHData
            DVH data object

        Returns
        -------
        dict
            Dictionary of common metrics
        """
        metrics = {
            "Structure": dvh_data.structure_name,
            "Volume (cc)": dvh_data.structure_volume,
            "Max Dose (Gy)": dvh_data.max_dose,
            "Mean Dose (Gy)": dvh_data.mean_dose,
            "Min Dose (Gy)": dvh_data.min_dose,
        }

        # Add D-x values
        for percent in [1, 2, 5, 50, 90, 95, 98, 99]:
            if percent in dvh_data.d_x:
                metrics[f"D{percent}% (Gy)"] = dvh_data.d_x[percent]

        # Add V-x values
        for dose_level in [5, 10, 20, 30, 40, 50, 60, 70, 80]:
            if (dose_level, True) in dvh_data.v_x:
                metrics[f"V{dose_level}Gy (%)"] = dvh_data.v_x[(dose_level, True)]

        return metrics


class DVHMetrics:
    """
    Static methods for computing DVH-based metrics used in plan evaluation.
    """

    @staticmethod
    def calculate_conformity_index(
        target_dvh: DVHData, prescription_dose: float
    ) -> float:
        """
        Calculate the RTOG Conformity Index.

        CI = V_RI / TV

        Where:
        - V_RI is the volume of the reference isodose
        - TV is the target volume

        Parameters
        ----------
        target_dvh : DVHData
            DVH data for the target structure
        prescription_dose : float
            Prescription dose in Gy

        Returns
        -------
        float
            Conformity Index
        """
        if target_dvh.structure_volume <= 0:
            raise ValueError("Target volume must be positive")

        # Get volume receiving at least the prescription dose
        v_ri = target_dvh.get_vx(prescription_dose, False)  # in cc

        # Calculate CI
        ci = v_ri / target_dvh.structure_volume

        return ci

    @staticmethod
    def calculate_homogeneity_index(
        target_dvh: DVHData, prescription_dose: float
    ) -> float:
        """
        Calculate the Homogeneity Index (HI).

        HI = (D2% - D98%) / D50%

        Where:
        - D2% is the dose to 2% of the target volume
        - D98% is the dose to 98% of the target volume
        - D50% is the dose to 50% of the target volume

        Parameters
        ----------
        target_dvh : DVHData
            DVH data for the target structure
        prescription_dose : float
            Prescription dose in Gy (used for validation)

        Returns
        -------
        float
            Homogeneity Index
        """
        # Get dose values
        d2 = target_dvh.get_dx(2)
        d98 = target_dvh.get_dx(98)
        d50 = target_dvh.get_dx(50)

        if d50 <= 0:
            logger.warning("D50% is zero or negative, cannot calculate HI")
            return float("inf")

        # Calculate HI
        hi = (d2 - d98) / d50

        return hi

    @staticmethod
    def calculate_gradient_index(
        target_dvh: DVHData,
        reference_dvh: DVHData,
        prescription_dose: float,
        lower_dose: float,
    ) -> float:
        """
        Calculate the Gradient Index (GI).

        GI = V_R1 / V_R2

        Where:
        - V_R1 is the volume of the lower isodose (e.g., 50% of prescription)
        - V_R2 is the volume of the prescription isodose

        Parameters
        ----------
        target_dvh : DVHData
            DVH data for the target structure
        reference_dvh : DVHData
            DVH data for the reference structure (e.g., whole body or external)
        prescription_dose : float
            Prescription dose in Gy
        lower_dose : float
            Lower dose value in Gy (typically 50% of prescription dose)

        Returns
        -------
        float
            Gradient Index
        """
        # Get volumes
        v_r2 = reference_dvh.get_vx(prescription_dose, False)  # in cc
        v_r1 = reference_dvh.get_vx(lower_dose, False)  # in cc

        if v_r2 <= 0:
            logger.warning(
                "Volume receiving prescription dose is zero, cannot calculate GI"
            )
            return float("inf")

        # Calculate GI
        gi = v_r1 / v_r2

        return gi


def calculate_dvh(
    dose_grid: np.ndarray, structure_mask: np.ndarray, num_bins: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Tính toán histogram thể tích liều đơn giản cho một cấu trúc.

    Parameters
    ----------
    dose_grid : np.ndarray
        Mảng 3D chứa giá trị liều (Gy)
    structure_mask : np.ndarray
        Mảng 3D chứa mặt nạ nhị phân của cấu trúc
    num_bins : int, optional
        Số bin sử dụng cho histogram, mặc định là 100

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Trả về mảng liều và mảng thể tích tương ứng
    """
    # Kiểm tra đầu vào
    if dose_grid is None or structure_mask is None:
        logger.error("Lỗi: dose_grid hoặc structure_mask là None")
        return np.array([0]), np.array([0])

    # Kiểm tra kích thước có khớp nhau không
    if dose_grid.shape != structure_mask.shape:
        logger.error(
            f"Lỗi: Kích thước không khớp: dose_grid {dose_grid.shape}, structure_mask {structure_mask.shape}"
        )
        return np.array([0]), np.array([0])

    # Kiểm tra tính hợp lệ của num_bins
    if num_bins <= 0:
        logger.warning(
            f"Giá trị num_bins không hợp lệ: {num_bins}, sử dụng giá trị mặc định 100"
        )
        num_bins = 100

    try:
        # Lấy các giá trị liều trong cấu trúc
        mask_indices = structure_mask > 0
        if not np.any(mask_indices):
            logger.warning("Không có voxel nào trong cấu trúc")
            return np.array([0]), np.array([0])

        doses_in_structure = dose_grid[mask_indices]
        num_voxels = len(doses_in_structure)

        # Kiểm tra xem có voxel nào trong cấu trúc không
        if num_voxels == 0:
            logger.warning("Không có voxel nào trong cấu trúc sau khi áp dụng mask")
            return np.array([0]), np.array([0])

        # Kiểm tra xem liều có hợp lệ không
        if np.all(doses_in_structure == 0):
            logger.warning("Tất cả các giá trị liều trong cấu trúc đều bằng 0")
            return np.array([0]), np.array([0])

        # Xác định phạm vi liều
        min_dose = 0.0  # Bắt đầu từ 0 Gy
        max_dose = np.max(doses_in_structure)

        # Thêm margin để đảm bảo hiển thị trực quan
        max_dose = max_dose * 1.05 if max_dose > 0 else 0.1

        # Kiểm tra xem liều có hợp lệ không
        if np.isnan(max_dose) or np.isinf(max_dose) or max_dose <= 0:
            logger.warning(f"Giá trị liều không hợp lệ: max_dose = {max_dose}")
            return np.array([0]), np.array([0])

        # Tạo bin liều
        dose_bins = np.linspace(min_dose, max_dose, num_bins + 1)
        bin_centers = (dose_bins[1:] + dose_bins[:-1]) / 2

        # Tính histogram vi phân - tối ưu cho bộ nhớ sử dụng bincount nếu có thể
        if (
            len(doses_in_structure) > 1e6
        ):  # Nếu có nhiều voxel, sử dụng phương pháp tiết kiệm bộ nhớ
            hist, _ = np.histogram(doses_in_structure, bins=dose_bins)
        else:
            hist, _ = np.histogram(doses_in_structure, bins=dose_bins)

        # Chuẩn hóa thể tích thành phần trăm
        vol_percent = (hist / num_voxels) * 100.0

        # Tính DVH tích lũy (từ cao đến thấp)
        cumulative_volume = np.zeros_like(bin_centers)
        for i in range(len(bin_centers)):
            # Thể tích nhận được liều ít nhất bằng bin_centers[i]
            cumulative_volume[i] = np.sum(vol_percent[i:])

        return bin_centers, cumulative_volume

    except Exception as e:
        logger.error(f"Lỗi khi tính DVH: {str(e)}")
        import traceback

        logger.debug(traceback.format_exc())
        return np.array([0]), np.array([0])


def calculate_dvh_from_3d_data(
    dose_data: np.ndarray,
    structure_mask: np.ndarray,
    voxel_size: Tuple[float, float, float],
    structure_id: str,
    num_bins: int = 100,
) -> Optional[DVHData]:
    """
    Calculate DVH data from raw 3D dose data and structure mask.

    Args:
        dose_data: 3D numpy array containing dose values (in Gy)
        structure_mask: 3D binary mask of the structure (same shape as dose_data)
        voxel_size: Voxel size in cm (x, y, z)
        structure_id: ID of the structure
        num_bins: Number of dose bins to use

    Returns:
        DVHData object containing the calculated DVH, or None if calculation fails
    """
    try:
        # Validate inputs
        if dose_data is None or structure_mask is None:
            logger.error("Dose data or structure mask is None")
            return None

        if dose_data.shape != structure_mask.shape:
            logger.error(
                f"Shape mismatch: dose shape {dose_data.shape}, mask shape {structure_mask.shape}"
            )
            return None

        # Calculate structure volume in cc
        voxel_volume = voxel_size[0] * voxel_size[1] * voxel_size[2]  # cm³
        num_voxels = np.sum(structure_mask > 0)
        total_volume = num_voxels * voxel_volume

        # Extract dose values within the structure
        dose_values = dose_data[structure_mask > 0]

        # If no voxels in structure, return empty DVH
        if len(dose_values) == 0:
            logger.warning(f"No dose values found in structure {structure_id}")
            return DVHData.from_raw_data(structure_id, [0.0], [0.0], total_volume)

        # Determine dose range
        min_dose = 0.0  # Start from 0 Gy
        max_dose = np.max(dose_values) * 1.05  # Add 5% margin for visualization

        # Create dose bins
        dose_bins = np.linspace(min_dose, max_dose, num_bins)

        # Calculate differential DVH
        hist, edges = np.histogram(dose_values, bins=dose_bins)

        # Convert histogram counts to volume
        if total_volume > 0 and len(dose_values) > 0:
            # Convert counts to percentage of total structure volume
            volume_percent = hist / len(dose_values) * 100.0
        else:
            volume_percent = np.zeros_like(hist, dtype=float)

        # Calculate cumulative DVH (reverse sum)
        cumulative_volume = np.zeros_like(dose_bins, dtype=float)
        cumulative_volume[:-1] = 100.0 - np.cumsum(volume_percent)

        # Create DVH data object
        dvh = DVHData.from_raw_data(
            structure_id, dose_bins.tolist(), cumulative_volume.tolist(), total_volume
        )

        # Set units
        dvh.dose_unit = "Gy"
        dvh.volume_unit = "%"

        # Calculate additional statistics
        dvh.min_dose = float(np.min(dose_values))
        dvh.max_dose = float(np.max(dose_values))
        dvh.mean_dose = float(np.mean(dose_values))

        # Calculate median dose
        if len(dose_values) > 0:
            dvh.median_dose = float(np.median(dose_values))

        return dvh

    except Exception as e:
        logger.error(f"Error calculating DVH from 3D data: {str(e)}")
        return None
