#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module so sánh liều (Dose Comparison) cho đảm bảo chất lượng trong xạ trị.

Module này cung cấp các công cụ để so sánh phân bố liều tính toán và đo đạc 
thông qua các phương pháp khác nhau như chênh lệch liều tuyệt đối, chênh lệch liều 
tương đối, DTA (Distance-to-Agreement), và các phương pháp kết hợp khác.
"""

import numpy as np
import SimpleITK as sitk
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
import matplotlib.pyplot as plt
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ComparisonMetricType(str, Enum):
    """Loại phép đo so sánh liều."""
    ABSOLUTE_DIFFERENCE = "absolute_difference"
    RELATIVE_DIFFERENCE = "relative_difference"
    DTA = "dta"  # Distance to agreement
    SIGNED_RELATIVE_DIFFERENCE = "signed_relative_difference"
    SIGNED_ABSOLUTE_DIFFERENCE = "signed_absolute_difference"


@dataclass
class DoseComparisonParameters:
    """
    Tham số cho so sánh liều.
    
    Attributes
    ----------
    reference_normalization : float
        Giá trị chuẩn hóa cho liều tham chiếu, thường là liều tối đa
    evaluated_normalization : float
        Giá trị chuẩn hóa cho liều đánh giá, thường là liều tối đa
    min_dose_percent : float
        Phần trăm liều tối thiểu để đánh giá (% của liều tối đa)
    threshold : float
        Ngưỡng chỉ định phần trăm khác biệt được chấp nhận
    search_distance : float
        Khoảng cách tìm kiếm tối đa (mm) cho phân tích DTA
    interpolation_method : str
        Phương pháp nội suy ("linear", "nearest", "spline")
    """
    reference_normalization: Optional[float] = None
    evaluated_normalization: Optional[float] = None
    min_dose_percent: float = 10.0
    threshold: float = 3.0
    search_distance: float = 5.0
    interpolation_method: str = "linear"


class DoseComparison:
    """
    So sánh hai phân bố liều bằng nhiều phương pháp khác nhau.
    """
    
    def __init__(self, parameters: DoseComparisonParameters = None):
        """
        Khởi tạo đối tượng so sánh liều.
        
        Parameters
        ----------
        parameters : DoseComparisonParameters, optional
            Tham số so sánh liều, mặc định sẽ sử dụng giá trị mặc định
        """
        self.parameters = parameters if parameters is not None else DoseComparisonParameters()
        self.reference_dose = None
        self.evaluated_dose = None
        self.reference_physical_points = None
        self.evaluated_physical_points = None
        self.reference_spacing = None
        self.evaluated_spacing = None
        self.comparison_results = {}
        self.is_computed = False
    
    def set_reference_dose(self, dose: Union[np.ndarray, sitk.Image], spacing: Optional[Tuple[float, float, float]] = None):
        """
        Đặt phân bố liều tham chiếu (tính toán).
        
        Parameters
        ----------
        dose : Union[np.ndarray, sitk.Image]
            Phân bố liều tham chiếu
        spacing : Tuple[float, float, float], optional
            Khoảng cách voxel (mm) nếu sử dụng np.ndarray, không cần nếu sử dụng sitk.Image
        """
        if isinstance(dose, sitk.Image):
            self.reference_dose = sitk.GetArrayFromImage(dose)
            self.reference_spacing = dose.GetSpacing()[::-1]  # SimpleITK và numpy có thứ tự khác nhau (XYZ vs ZYX)
        else:
            self.reference_dose = dose.copy()
            if spacing is None:
                raise ValueError("Phải cung cấp spacing khi sử dụng ndarray")
            self.reference_spacing = spacing
        
        # Tạo lưới tọa độ vật lý
        shape = self.reference_dose.shape
        x = np.arange(0, shape[2]) * self.reference_spacing[2]
        y = np.arange(0, shape[1]) * self.reference_spacing[1]
        z = np.arange(0, shape[0]) * self.reference_spacing[0]
        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        self.reference_physical_points = np.stack([xx, yy, zz], axis=-1)
        
        # Đặt chuẩn hóa liều tham chiếu nếu chưa được đặt
        if self.parameters.reference_normalization is None:
            self.parameters.reference_normalization = np.max(self.reference_dose)
        
        self.is_computed = False
        
    def set_evaluated_dose(self, dose: Union[np.ndarray, sitk.Image], spacing: Optional[Tuple[float, float, float]] = None):
        """
        Đặt phân bố liều cần đánh giá (đo đạc).
        
        Parameters
        ----------
        dose : Union[np.ndarray, sitk.Image]
            Phân bố liều cần đánh giá
        spacing : Tuple[float, float, float], optional
            Khoảng cách voxel (mm) nếu sử dụng np.ndarray, không cần nếu sử dụng sitk.Image
        """
        if isinstance(dose, sitk.Image):
            self.evaluated_dose = sitk.GetArrayFromImage(dose)
            self.evaluated_spacing = dose.GetSpacing()[::-1]  # SimpleITK và numpy có thứ tự khác nhau (XYZ vs ZYX)
        else:
            self.evaluated_dose = dose.copy()
            if spacing is None:
                raise ValueError("Phải cung cấp spacing khi sử dụng ndarray")
            self.evaluated_spacing = spacing
        
        # Tạo lưới tọa độ vật lý
        shape = self.evaluated_dose.shape
        x = np.arange(0, shape[2]) * self.evaluated_spacing[2]
        y = np.arange(0, shape[1]) * self.evaluated_spacing[1]
        z = np.arange(0, shape[0]) * self.evaluated_spacing[0]
        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        self.evaluated_physical_points = np.stack([xx, yy, zz], axis=-1)
        
        # Đặt chuẩn hóa liều đánh giá nếu chưa được đặt
        if self.parameters.evaluated_normalization is None:
            self.parameters.evaluated_normalization = np.max(self.evaluated_dose)
        
        self.is_computed = False
    
    def _compute_absolute_difference(self, mask: np.ndarray) -> np.ndarray:
        """
        Tính toán chênh lệch liều tuyệt đối giữa hai phân bố.
        
        Parameters
        ----------
        mask : np.ndarray
            Mask chỉ ra vùng cần phân tích
            
        Returns
        -------
        np.ndarray
            Bản đồ chênh lệch liều tuyệt đối (đơn vị theo liều gốc)
        """
        return np.abs(self.evaluated_dose - self.reference_dose)
    
    def _compute_relative_difference(self, mask: np.ndarray) -> np.ndarray:
        """
        Tính toán chênh lệch liều tương đối giữa hai phân bố.
        
        Parameters
        ----------
        mask : np.ndarray
            Mask chỉ ra vùng cần phân tích
            
        Returns
        -------
        np.ndarray
            Bản đồ chênh lệch liều tương đối (%)
        """
        # Tránh chia cho 0
        safe_ref_dose = np.copy(self.reference_dose)
        safe_ref_dose[safe_ref_dose == 0] = 1e-10
        
        return np.abs(self.evaluated_dose - self.reference_dose) / self.parameters.reference_normalization * 100.0
    
    def _compute_signed_absolute_difference(self, mask: np.ndarray) -> np.ndarray:
        """
        Tính toán chênh lệch liều tuyệt đối có dấu.
        
        Parameters
        ----------
        mask : np.ndarray
            Mask chỉ ra vùng cần phân tích
            
        Returns
        -------
        np.ndarray
            Bản đồ chênh lệch liều tuyệt đối có dấu (đơn vị theo liều gốc)
        """
        return self.evaluated_dose - self.reference_dose
    
    def _compute_signed_relative_difference(self, mask: np.ndarray) -> np.ndarray:
        """
        Tính toán chênh lệch liều tương đối có dấu.
        
        Parameters
        ----------
        mask : np.ndarray
            Mask chỉ ra vùng cần phân tích
            
        Returns
        -------
        np.ndarray
            Bản đồ chênh lệch liều tương đối có dấu (%)
        """
        # Tránh chia cho 0
        safe_ref_dose = np.copy(self.reference_dose)
        safe_ref_dose[safe_ref_dose == 0] = 1e-10
        
        return (self.evaluated_dose - self.reference_dose) / self.parameters.reference_normalization * 100.0
    
    def _compute_dta(self, mask: np.ndarray) -> np.ndarray:
        """
        Tính toán DTA (Distance to Agreement).
        
        Parameters
        ----------
        mask : np.ndarray
            Mask chỉ ra vùng cần phân tích
            
        Returns
        -------
        np.ndarray
            Bản đồ DTA (mm)
        """
        # Khởi tạo bản đồ DTA
        dta_map = np.ones_like(self.reference_dose) * self.parameters.search_distance
        
        # Tạo lưới gradient liều
        grad_x, grad_y, grad_z = np.gradient(self.reference_dose, 
                                             self.reference_spacing[0], 
                                             self.reference_spacing[1], 
                                             self.reference_spacing[2])
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
        
        # Chỉ tính DTA cho các điểm có gradient khác 0
        valid_mask = np.logical_and(mask, gradient_magnitude > 0)
        
        # Thuật toán bước nhỏ để tìm DTA
        search_increment = min(self.reference_spacing) / 2.0
        max_steps = int(self.parameters.search_distance / search_increment) + 1
        
        for idx in np.argwhere(valid_mask):
            i, j, k = idx
            ref_value = self.reference_dose[i, j, k]
            
            # Tính gradient tại điểm
            grad_vector = np.array([grad_x[i, j, k], grad_y[i, j, k], grad_z[i, j, k]])
            grad_norm = np.linalg.norm(grad_vector)
            if grad_norm < 1e-10:
                continue
            
            grad_unit_vector = grad_vector / grad_norm
            
            # Tìm kiếm theo gradient
            min_distance = self.parameters.search_distance
            
            for step in range(1, max_steps + 1):
                for direction in [1, -1]:  # Tìm cả hai hướng
                    distance = step * search_increment * direction
                    
                    # Tính tọa độ điểm tìm kiếm
                    offset = grad_unit_vector * distance
                    search_point = np.array([i, j, k]) + offset
                    
                    # Kiểm tra xem điểm có nằm trong miền không
                    if (search_point[0] < 0 or search_point[0] >= self.reference_dose.shape[0] or
                        search_point[1] < 0 or search_point[1] >= self.reference_dose.shape[1] or
                        search_point[2] < 0 or search_point[2] >= self.reference_dose.shape[2]):
                        continue
                    
                    # Lấy mẫu giá trị liều tại điểm tìm kiếm bằng nội suy
                    i_floor, i_ceil = int(search_point[0]), min(int(search_point[0]) + 1, self.evaluated_dose.shape[0] - 1)
                    j_floor, j_ceil = int(search_point[1]), min(int(search_point[1]) + 1, self.evaluated_dose.shape[1] - 1)
                    k_floor, k_ceil = int(search_point[2]), min(int(search_point[2]) + 1, self.evaluated_dose.shape[2] - 1)
                    
                    i_frac = search_point[0] - i_floor
                    j_frac = search_point[1] - j_floor
                    k_frac = search_point[2] - k_floor
                    
                    interpolated_value = (
                        self.evaluated_dose[i_floor, j_floor, k_floor] * (1 - i_frac) * (1 - j_frac) * (1 - k_frac) +
                        self.evaluated_dose[i_ceil, j_floor, k_floor] * i_frac * (1 - j_frac) * (1 - k_frac) +
                        self.evaluated_dose[i_floor, j_ceil, k_floor] * (1 - i_frac) * j_frac * (1 - k_frac) +
                        self.evaluated_dose[i_floor, j_floor, k_ceil] * (1 - i_frac) * (1 - j_frac) * k_frac +
                        self.evaluated_dose[i_ceil, j_ceil, k_floor] * i_frac * j_frac * (1 - k_frac) +
                        self.evaluated_dose[i_ceil, j_floor, k_ceil] * i_frac * (1 - j_frac) * k_frac +
                        self.evaluated_dose[i_floor, j_ceil, k_ceil] * (1 - i_frac) * j_frac * k_frac +
                        self.evaluated_dose[i_ceil, j_ceil, k_ceil] * i_frac * j_frac * k_frac
                    )
                    
                    # Kiểm tra xem có bằng giá trị tham chiếu không
                    if abs(interpolated_value - ref_value) < self.parameters.threshold * self.parameters.reference_normalization / 100.0:
                        actual_distance = abs(distance)
                        if actual_distance < min_distance:
                            min_distance = actual_distance
                            break  # Đã tìm thấy điểm gần nhất
            
            dta_map[i, j, k] = min_distance
        
        return dta_map
    
    def compute(self, metric_types: List[ComparisonMetricType], mask: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Tính toán các chỉ số so sánh liều.
        
        Parameters
        ----------
        metric_types : List[ComparisonMetricType]
            Danh sách các chỉ số so sánh cần tính
        mask : np.ndarray, optional
            Mask chỉ ra vùng cần phân tích
            
        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary chứa các bản đồ kết quả, khóa là tên chỉ số
        """
        if self.reference_dose is None or self.evaluated_dose is None:
            raise ValueError("Phải đặt cả phân bố liều tham chiếu và đánh giá trước khi tính toán")
        
        if self.reference_dose.shape != self.evaluated_dose.shape:
            raise ValueError("Phân bố liều tham chiếu và đánh giá phải có cùng kích thước")
        
        # Chuẩn bị mask
        if mask is None:
            eval_mask = np.ones_like(self.reference_dose, dtype=bool)
        else:
            eval_mask = mask.astype(bool)
        
        # Áp dụng ngưỡng liều tối thiểu
        if self.parameters.min_dose_percent > 0:
            max_dose = np.max(self.reference_dose)
            min_dose_threshold = max_dose * self.parameters.min_dose_percent / 100.0
            eval_mask = np.logical_and(eval_mask, self.reference_dose >= min_dose_threshold)
        
        # Tính toán các chỉ số được yêu cầu
        results = {}
        
        for metric_type in metric_types:
            if metric_type == ComparisonMetricType.ABSOLUTE_DIFFERENCE:
                results[metric_type] = self._compute_absolute_difference(eval_mask)
            elif metric_type == ComparisonMetricType.RELATIVE_DIFFERENCE:
                results[metric_type] = self._compute_relative_difference(eval_mask)
            elif metric_type == ComparisonMetricType.SIGNED_ABSOLUTE_DIFFERENCE:
                results[metric_type] = self._compute_signed_absolute_difference(eval_mask)
            elif metric_type == ComparisonMetricType.SIGNED_RELATIVE_DIFFERENCE:
                results[metric_type] = self._compute_signed_relative_difference(eval_mask)
            elif metric_type == ComparisonMetricType.DTA:
                results[metric_type] = self._compute_dta(eval_mask)
        
        self.comparison_results = results
        self.is_computed = True
        
        return results
    
    def get_passing_rate(self, metric_type: ComparisonMetricType, threshold: Optional[float] = None) -> float:
        """
        Lấy tỷ lệ vượt qua (passing rate) của phép đo so sánh.
        
        Parameters
        ----------
        metric_type : ComparisonMetricType
            Loại phép đo so sánh
        threshold : float, optional
            Ngưỡng chấp nhận, nếu None sẽ sử dụng ngưỡng mặc định
            
        Returns
        -------
        float
            Tỷ lệ vượt qua (%)
        """
        if not self.is_computed or metric_type not in self.comparison_results:
            raise ValueError(f"Phải gọi compute() với {metric_type} trước khi lấy tỷ lệ vượt qua")
        
        if threshold is None:
            threshold = self.parameters.threshold
        
        result_map = self.comparison_results[metric_type]
        
        # Áp dụng ngưỡng liều tối thiểu
        max_dose = np.max(self.reference_dose)
        min_dose_threshold = max_dose * self.parameters.min_dose_percent / 100.0
        eval_mask = self.reference_dose >= min_dose_threshold
        
        # Tính tỷ lệ vượt qua
        if metric_type == ComparisonMetricType.ABSOLUTE_DIFFERENCE:
            passing_points = result_map <= threshold
        elif metric_type == ComparisonMetricType.RELATIVE_DIFFERENCE:
            passing_points = result_map <= threshold
        elif metric_type == ComparisonMetricType.SIGNED_ABSOLUTE_DIFFERENCE:
            passing_points = np.abs(result_map) <= threshold
        elif metric_type == ComparisonMetricType.SIGNED_RELATIVE_DIFFERENCE:
            passing_points = np.abs(result_map) <= threshold
        elif metric_type == ComparisonMetricType.DTA:
            passing_points = result_map <= threshold
        
        # Áp dụng mask
        valid_points = np.logical_and(eval_mask, ~np.isnan(result_map))
        passing_points = np.logical_and(passing_points, valid_points)
        
        if np.sum(valid_points) > 0:
            passing_rate = np.sum(passing_points) / np.sum(valid_points) * 100.0
        else:
            passing_rate = 0.0
        
        return passing_rate
    
    def plot_comparison(self, metric_type: ComparisonMetricType, slice_idx=None, ax=None, 
                        figsize=(10, 8), colormap='jet', show_colorbar=True,
                        vmin=None, vmax=None, title=None):
        """
        Vẽ kết quả so sánh liều.
        
        Parameters
        ----------
        metric_type : ComparisonMetricType
            Loại phép đo so sánh
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
        vmin, vmax : float, optional
            Giới hạn giá trị cho thang màu
        title : str, optional
            Tiêu đề đồ thị
            
        Returns
        -------
        matplotlib.axes.Axes
            Trục đã vẽ
        """
        if not self.is_computed or metric_type not in self.comparison_results:
            raise ValueError(f"Phải gọi compute() với {metric_type} trước khi vẽ")
        
        result_map = self.comparison_results[metric_type]
        
        if slice_idx is None:
            slice_idx = result_map.shape[0] // 2
        
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        
        result_slice = result_map[slice_idx, :, :]
        
        # Đặt giới hạn thang màu mặc định
        if vmin is None and vmax is None:
            if metric_type == ComparisonMetricType.DTA:
                vmin, vmax = 0, self.parameters.search_distance
            elif metric_type == ComparisonMetricType.ABSOLUTE_DIFFERENCE:
                vmin, vmax = 0, np.percentile(result_map[~np.isnan(result_map)], 95)
            elif metric_type == ComparisonMetricType.RELATIVE_DIFFERENCE:
                vmin, vmax = 0, np.percentile(result_map[~np.isnan(result_map)], 95)
            elif metric_type in [ComparisonMetricType.SIGNED_ABSOLUTE_DIFFERENCE, 
                                 ComparisonMetricType.SIGNED_RELATIVE_DIFFERENCE]:
                abs_max = np.percentile(np.abs(result_map[~np.isnan(result_map)]), 95)
                vmin, vmax = -abs_max, abs_max
        
        im = ax.imshow(result_slice, cmap=colormap, vmin=vmin, vmax=vmax)
        
        if show_colorbar:
            cbar = plt.colorbar(im, ax=ax)
            
            if metric_type == ComparisonMetricType.ABSOLUTE_DIFFERENCE:
                cbar.set_label('Absolute Difference (Gy)')
            elif metric_type == ComparisonMetricType.RELATIVE_DIFFERENCE:
                cbar.set_label('Relative Difference (%)')
            elif metric_type == ComparisonMetricType.SIGNED_ABSOLUTE_DIFFERENCE:
                cbar.set_label('Signed Absolute Difference (Gy)')
            elif metric_type == ComparisonMetricType.SIGNED_RELATIVE_DIFFERENCE:
                cbar.set_label('Signed Relative Difference (%)')
            elif metric_type == ComparisonMetricType.DTA:
                cbar.set_label('Distance to Agreement (mm)')
        
        # Đặt tiêu đề
        if title is None:
            passing_rate = self.get_passing_rate(metric_type)
            if metric_type == ComparisonMetricType.ABSOLUTE_DIFFERENCE:
                title = f'Absolute Difference (Slice {slice_idx}, Pass Rate: {passing_rate:.2f}%)'
            elif metric_type == ComparisonMetricType.RELATIVE_DIFFERENCE:
                title = f'Relative Difference (Slice {slice_idx}, Pass Rate: {passing_rate:.2f}%)'
            elif metric_type == ComparisonMetricType.SIGNED_ABSOLUTE_DIFFERENCE:
                title = f'Signed Absolute Difference (Slice {slice_idx}, Pass Rate: {passing_rate:.2f}%)'
            elif metric_type == ComparisonMetricType.SIGNED_RELATIVE_DIFFERENCE:
                title = f'Signed Relative Difference (Slice {slice_idx}, Pass Rate: {passing_rate:.2f}%)'
            elif metric_type == ComparisonMetricType.DTA:
                title = f'Distance to Agreement (Slice {slice_idx}, Pass Rate: {passing_rate:.2f}%)'
        
        ax.set_title(title)
        
        return ax


def compare_dose_distributions(
    reference_dose: Union[np.ndarray, sitk.Image],
    evaluated_dose: Union[np.ndarray, sitk.Image],
    metric_type: ComparisonMetricType = ComparisonMetricType.RELATIVE_DIFFERENCE,
    threshold: float = 3.0,
    min_dose_percent: float = 10.0,
    mask: Optional[Union[np.ndarray, sitk.Image]] = None
) -> Tuple[np.ndarray, float]:
    """
    So sánh hai phân bố liều.
    
    Parameters
    ----------
    reference_dose : Union[np.ndarray, sitk.Image]
        Phân bố liều tham chiếu
    evaluated_dose : Union[np.ndarray, sitk.Image]
        Phân bố liều cần đánh giá
    metric_type : ComparisonMetricType, optional
        Loại phép đo so sánh
    threshold : float, optional
        Ngưỡng chấp nhận (% hoặc mm)
    min_dose_percent : float, optional
        Phần trăm liều tối thiểu để đánh giá
    mask : Union[np.ndarray, sitk.Image], optional
        Mask chỉ ra vùng cần phân tích
        
    Returns
    -------
    Tuple[np.ndarray, float]
        Bản đồ kết quả và tỷ lệ vượt qua (%)
    """
    params = DoseComparisonParameters(
        threshold=threshold,
        min_dose_percent=min_dose_percent
    )
    
    comparator = DoseComparison(parameters=params)
    comparator.set_reference_dose(reference_dose)
    comparator.set_evaluated_dose(evaluated_dose)
    
    if isinstance(mask, sitk.Image):
        mask_array = sitk.GetArrayFromImage(mask)
    else:
        mask_array = mask
    
    results = comparator.compute([metric_type], mask_array)
    passing_rate = comparator.get_passing_rate(metric_type)
    
    return results[metric_type], passing_rate