#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module phân tích Gamma cho đảm bảo chất lượng trong xạ trị.

Module này cung cấp các công cụ để thực hiện phân tích Gamma - một phương pháp
được sử dụng rộng rãi để so sánh định lượng giữa phân bố liều tính toán và đo đạc.
Phân tích Gamma cung cấp thông tin về sự khác biệt về liều và không gian giữa hai phân bố.
"""

import numpy as np
import SimpleITK as sitk
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
import matplotlib.pyplot as plt
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class GammaParameters:
    """
    Tham số cho phân tích Gamma.

    Attributes
    ----------
    dose_threshold : float
        Ngưỡng % khác biệt liều chấp nhận được (ví dụ: 3%)
    distance_threshold : float
        Ngưỡng mm khác biệt khoảng cách chấp nhận được (ví dụ: 3mm)
    global_normalization : bool
        Sử dụng chuẩn hóa toàn cục (True) hoặc cục bộ (False)
    normalization_value : Optional[float]
        Giá trị chuẩn hóa tùy chỉnh (nếu không phải là None)
    dose_threshold_type : str
        Loại ngưỡng liều: 'relative' hoặc 'absolute'
    dose_threshold_unit : str
        Đơn vị ngưỡng liều (%, Gy, cGy)
    max_gamma : float
        Giá trị Gamma tối đa để tính toán (tối ưu hóa)
    min_dose_percent : float
        Phần trăm liều tối thiểu để đánh giá (% của liều tối đa)
    """

    dose_threshold: float = 3.0
    distance_threshold: float = 3.0
    global_normalization: bool = True
    normalization_value: Optional[float] = None
    dose_threshold_type: str = "relative"
    dose_threshold_unit: str = "%"
    max_gamma: float = 5.0
    min_dose_percent: float = 10.0


class GammaAnalysis:
    """
    Thực hiện phân tích Gamma giữa hai phân bố liều.
    """

    def __init__(self, parameters: GammaParameters = None):
        """
        Khởi tạo phân tích Gamma với các tham số cụ thể.

        Parameters
        ----------
        parameters : GammaParameters, optional
            Tham số phân tích Gamma, mặc định sẽ sử dụng 3%/3mm
        """
        self.parameters = parameters if parameters is not None else GammaParameters()
        self.reference_dose = None
        self.evaluated_dose = None
        self.gamma_map = None
        self.passing_rate = None
        self.histogram = None
        self.is_computed = False

    def set_reference_dose(self, dose: Union[np.ndarray, sitk.Image]):
        """
        Đặt phân bố liều tham chiếu (tính toán).

        Parameters
        ----------
        dose : Union[np.ndarray, sitk.Image]
            Phân bố liều tham chiếu
        """
        if isinstance(dose, sitk.Image):
            self.reference_dose = sitk.GetArrayFromImage(dose)
        else:
            self.reference_dose = dose.copy()

        self.is_computed = False

    def set_evaluated_dose(self, dose: Union[np.ndarray, sitk.Image]):
        """
        Đặt phân bố liều cần đánh giá (đo đạc).

        Parameters
        ----------
        dose : Union[np.ndarray, sitk.Image]
            Phân bố liều cần đánh giá
        """
        if isinstance(dose, sitk.Image):
            self.evaluated_dose = sitk.GetArrayFromImage(dose)
        else:
            self.evaluated_dose = dose.copy()

        self.is_computed = False

    def compute(
        self, mask: Optional[Union[np.ndarray, sitk.Image]] = None
    ) -> np.ndarray:
        """
        Tính toán bản đồ Gamma.

        Parameters
        ----------
        mask : Union[np.ndarray, sitk.Image], optional
            Mask chỉ ra vùng cần phân tích (nếu None, phân tích toàn bộ)

        Returns
        -------
        np.ndarray
            Bản đồ Gamma

        Raises
        ------
        ValueError
            Nếu phân bố liều chưa được đặt hoặc các thông số không hợp lệ
        """
        if self.reference_dose is None or self.evaluated_dose is None:
            raise ValueError(
                "Phải đặt cả phân bố liều tham chiếu và đánh giá trước khi tính toán"
            )

        if self.reference_dose.shape != self.evaluated_dose.shape:
            raise ValueError(
                "Phân bố liều tham chiếu và đánh giá phải có cùng kích thước"
            )

        # Chuẩn bị mask
        if mask is None:
            eval_mask = np.ones_like(self.reference_dose, dtype=bool)
        elif isinstance(mask, sitk.Image):
            eval_mask = sitk.GetArrayFromImage(mask).astype(bool)
        else:
            eval_mask = mask.astype(bool)

        # Áp dụng ngưỡng liều tối thiểu
        if self.parameters.min_dose_percent > 0:
            max_dose = np.max(self.reference_dose)
            min_dose_threshold = max_dose * self.parameters.min_dose_percent / 100.0
            eval_mask = np.logical_and(
                eval_mask, self.reference_dose >= min_dose_threshold
            )

        # Chuẩn bị tham số
        dose_threshold = self.parameters.dose_threshold
        distance_threshold = self.parameters.distance_threshold
        max_gamma = self.parameters.max_gamma

        # Chuẩn hóa liều nếu cần
        if self.parameters.dose_threshold_type == "relative":
            if self.parameters.global_normalization:
                if self.parameters.normalization_value is not None:
                    normalization = self.parameters.normalization_value
                else:
                    normalization = np.max(self.reference_dose)
            else:
                normalization = 1.0  # Chuẩn hóa cục bộ sẽ được xử lý trong vòng lặp
        else:
            normalization = 1.0  # Không chuẩn hóa cho ngưỡng tuyệt đối

        # Khởi tạo bản đồ Gamma
        gamma_map = np.ones_like(self.reference_dose) * max_gamma

        # Tạo lưới tọa độ
        y_indices, x_indices, z_indices = np.indices(self.reference_dose.shape)

        # Tính toán Gamma cho mỗi điểm trong mask
        for idx in np.argwhere(eval_mask):
            i, j, k = idx[0], idx[1], idx[2]

            ref_dose = self.reference_dose[i, j, k]

            if self.parameters.global_normalization:
                dose_diff_threshold = dose_threshold * normalization / 100.0
            else:
                dose_diff_threshold = dose_threshold * ref_dose / 100.0

            # Tính khoảng cách và chênh lệch liều cho mọi điểm lân cận
            spatial_distance = np.sqrt(
                (y_indices - i) ** 2 + (x_indices - j) ** 2 + (z_indices - k) ** 2
            )

            dose_difference = np.abs(self.evaluated_dose - ref_dose)

            # Tính Gamma cho điểm hiện tại
            gamma_values = np.sqrt(
                (spatial_distance / distance_threshold) ** 2
                + (dose_difference / dose_diff_threshold) ** 2
            )

            # Lấy giá trị Gamma nhỏ nhất
            gamma_map[i, j, k] = np.min(gamma_values)

        self.gamma_map = gamma_map

        # Tính tỷ lệ vượt qua (passing rate)
        if np.sum(eval_mask) > 0:
            self.passing_rate = (
                np.sum(gamma_map[eval_mask] <= 1.0) / np.sum(eval_mask) * 100.0
            )
        else:
            self.passing_rate = 0.0

        # Tính histogram
        self.histogram = np.histogram(
            gamma_map[eval_mask], bins=100, range=(0, max_gamma)
        )

        self.is_computed = True

        return gamma_map

    def get_passing_rate(self) -> float:
        """
        Lấy tỷ lệ vượt qua (passing rate) của phân tích Gamma.

        Returns
        -------
        float
            Tỷ lệ vượt qua (%) - phần trăm điểm có gamma <= 1.0
        """
        if not self.is_computed:
            raise ValueError("Phải gọi compute() trước khi lấy tỷ lệ vượt qua")

        return self.passing_rate

    def get_gamma_map(self) -> np.ndarray:
        """
        Lấy bản đồ Gamma.

        Returns
        -------
        np.ndarray
            Bản đồ Gamma
        """
        if not self.is_computed:
            raise ValueError("Phải gọi compute() trước khi lấy bản đồ Gamma")

        return self.gamma_map

    def plot_gamma_histogram(self, ax=None, figsize=(10, 6)):
        """
        Vẽ histogram của các giá trị Gamma.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Trục để vẽ, nếu None sẽ tạo mới
        figsize : tuple, optional
            Kích thước hình (theo inch) nếu tạo mới

        Returns
        -------
        matplotlib.axes.Axes
            Trục đã vẽ
        """
        if not self.is_computed:
            raise ValueError("Phải gọi compute() trước khi vẽ histogram")

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        counts, bin_edges = self.histogram
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        ax.bar(bin_centers, counts, width=bin_centers[1] - bin_centers[0], alpha=0.7)
        ax.axvline(x=1.0, color="r", linestyle="--", label="Gamma = 1.0")

        ax.set_xlabel("Gamma Value")
        ax.set_ylabel("Frequency")
        ax.set_title(f"Gamma Histogram (Pass Rate: {self.passing_rate:.2f}%)")
        ax.legend()

        return ax

    def plot_gamma_map(
        self,
        slice_idx=None,
        ax=None,
        figsize=(10, 8),
        colormap="viridis",
        show_colorbar=True,
    ):
        """
        Vẽ bản đồ Gamma.

        Parameters
        ----------
        slice_idx : int, optional
            Chỉ số lát cắt cần vẽ, nếu None sẽ chọn lát cắt giữa
        ax : matplotlib.axes.Axes, optional
            Trục để vẽ, nếu None sẽ tạo mới
        figsize : tuple, optional
            Kích thước hình (theo inch) nếu tạo mới
        colormap : str, optional
            Bảng màu sử dụng
        show_colorbar : bool, optional
            Hiển thị thang màu

        Returns
        -------
        matplotlib.axes.Axes
            Trục đã vẽ
        """
        if not self.is_computed:
            raise ValueError("Phải gọi compute() trước khi vẽ bản đồ Gamma")

        if slice_idx is None:
            slice_idx = self.gamma_map.shape[0] // 2

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        gamma_slice = self.gamma_map[slice_idx, :, :]
        im = ax.imshow(
            gamma_slice, cmap=colormap, vmin=0, vmax=min(2.0, self.parameters.max_gamma)
        )

        if show_colorbar:
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label("Gamma Value")

        ax.set_title(
            f"Gamma Map (Slice {slice_idx}, Pass Rate: {self.passing_rate:.2f}%)"
        )

        return ax

    def __str__(self) -> str:
        """Biểu diễn chuỗi của phân tích Gamma."""
        if not self.is_computed:
            return "Phân tích Gamma (chưa tính toán)"

        return (
            f"Phân tích Gamma ({self.parameters.dose_threshold}{self.parameters.dose_threshold_unit}/"
            f"{self.parameters.distance_threshold}mm): Pass Rate = {self.passing_rate:.2f}%"
        )


def perform_gamma_analysis(
    reference_dose: Union[np.ndarray, sitk.Image],
    evaluated_dose: Union[np.ndarray, sitk.Image],
    dose_threshold: float = 3.0,
    distance_threshold: float = 3.0,
    mask: Optional[Union[np.ndarray, sitk.Image]] = None,
    global_normalization: bool = True,
    min_dose_percent: float = 10.0,
) -> Tuple[np.ndarray, float]:
    """
    Thực hiện phân tích Gamma giữa hai phân bố liều.

    Parameters
    ----------
    reference_dose : Union[np.ndarray, sitk.Image]
        Phân bố liều tham chiếu (tính toán)
    evaluated_dose : Union[np.ndarray, sitk.Image]
        Phân bố liều cần đánh giá (đo đạc)
    dose_threshold : float, optional
        Ngưỡng % khác biệt liều chấp nhận được
    distance_threshold : float, optional
        Ngưỡng mm khác biệt khoảng cách chấp nhận được
    mask : Union[np.ndarray, sitk.Image], optional
        Mask chỉ ra vùng cần phân tích
    global_normalization : bool, optional
        Sử dụng chuẩn hóa toàn cục
    min_dose_percent : float, optional
        Phần trăm liều tối thiểu để đánh giá

    Returns
    -------
    Tuple[np.ndarray, float]
        Bản đồ Gamma và tỷ lệ vượt qua (%)
    """
    params = GammaParameters(
        dose_threshold=dose_threshold,
        distance_threshold=distance_threshold,
        global_normalization=global_normalization,
        min_dose_percent=min_dose_percent,
    )

    analyzer = GammaAnalysis(parameters=params)
    analyzer.set_reference_dose(reference_dose)
    analyzer.set_evaluated_dose(evaluated_dose)

    gamma_map = analyzer.compute(mask)
    passing_rate = analyzer.get_passing_rate()

    return gamma_map, passing_rate


def calculate_gamma_index_3d(
    reference_dose: np.ndarray,
    evaluated_dose: np.ndarray,
    dose_threshold: float = 3.0,
    distance_threshold: float = 3.0,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    mask: Optional[np.ndarray] = None,
    global_normalization: bool = True,
) -> np.ndarray:
    """
    Tính toán gamma index 3D giữa hai phân bố liều.

    Parameters
    ----------
    reference_dose : np.ndarray
        Phân bố liều tham chiếu (3D array)
    evaluated_dose : np.ndarray
        Phân bố liều cần đánh giá (3D array)
    dose_threshold : float, optional
        Ngưỡng % khác biệt liều, mặc định 3.0%
    distance_threshold : float, optional
        Ngưỡng mm khác biệt khoảng cách, mặc định 3.0mm
    spacing : Tuple[float, float, float], optional
        Khoảng cách voxel (mm), mặc định (1.0, 1.0, 1.0)
    mask : np.ndarray, optional
        Mask vùng đánh giá
    global_normalization : bool, optional
        Sử dụng chuẩn hóa toàn cục, mặc định True

    Returns
    -------
    np.ndarray
        Bản đồ gamma index 3D
    """
    try:
        # Kiểm tra input
        if reference_dose.shape != evaluated_dose.shape:
            raise ValueError("Reference và evaluated dose phải có cùng kích thước")

        # Tạo mask mặc định nếu không có
        if mask is None:
            mask = np.ones_like(reference_dose, dtype=bool)

        # Khởi tạo gamma map
        gamma_map = np.ones_like(reference_dose) * 999.0

        # Chuẩn hóa
        if global_normalization:
            normalization = np.max(reference_dose)
        else:
            normalization = 1.0

        dose_diff_threshold = dose_threshold * normalization / 100.0

        # Tạo lưới tọa độ với spacing
        z_coords, y_coords, x_coords = np.indices(reference_dose.shape)
        z_coords = z_coords * spacing[2]
        y_coords = y_coords * spacing[1]
        x_coords = x_coords * spacing[0]

        # Tính gamma cho từng voxel trong mask
        for idx in np.argwhere(mask):
            z, y, x = idx[0], idx[1], idx[2]

            ref_dose = reference_dose[z, y, x]

            # Tính khoảng cách không gian
            spatial_distance = np.sqrt(
                ((z_coords - z_coords[z, y, x]) ** 2)
                + ((y_coords - y_coords[z, y, x]) ** 2)
                + ((x_coords - x_coords[z, y, x]) ** 2)
            )

            # Tính chênh lệch liều
            dose_difference = np.abs(evaluated_dose - ref_dose)

            # Tính gamma
            if not global_normalization:
                dose_diff_threshold = dose_threshold * ref_dose / 100.0

            gamma_values = np.sqrt(
                (spatial_distance / distance_threshold) ** 2
                + (dose_difference / dose_diff_threshold) ** 2
            )

            # Lấy gamma nhỏ nhất
            gamma_map[z, y, x] = np.min(gamma_values)

        logger.info(
            f"Tính toán gamma 3D hoàn thành: {dose_threshold}%/{distance_threshold}mm"
        )
        return gamma_map

    except Exception as e:
        logger.error(f"Lỗi tính toán gamma index 3D: {e}")
        return np.ones_like(reference_dose) * 999.0


def calculate_gamma_3d(
    reference_dose: np.ndarray,
    evaluated_dose: np.ndarray,
    distance_mm: float = 3.0,
    dose_percent: float = 3.0,
    **kwargs,
) -> np.ndarray:
    """
    Alias function cho calculate_gamma_index_3d với tên ngắn gọn hơn.

    Parameters
    ----------
    reference_dose : np.ndarray
        Phân bố liều tham chiếu
    evaluated_dose : np.ndarray
        Phân bố liều cần đánh giá
    distance_mm : float, optional
        Ngưỡng khoảng cách (mm), mặc định 3.0
    dose_percent : float, optional
        Ngưỡng liều (%), mặc định 3.0
    **kwargs
        Các tham số khác

    Returns
    -------
    np.ndarray
        Bản đồ gamma index
    """
    return calculate_gamma_index_3d(
        reference_dose=reference_dose,
        evaluated_dose=evaluated_dose,
        dose_threshold=dose_percent,
        distance_threshold=distance_mm,
        **kwargs,
    )


# Export list
__all__ = [
    "GammaParameters",
    "GammaAnalysis",
    "perform_gamma_analysis",
    "calculate_gamma_index_3d",
    "calculate_gamma_3d",
]
