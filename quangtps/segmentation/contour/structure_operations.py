#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module Structure Operations cho QuangTPS

Cung cấp các thao tác nâng cao trên cấu trúc như Boolean Operations,
phân tích và tái cấu trúc contour, và các thao tác chỉnh sửa khối lượng.
"""

import os
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union, Any, Set
from enum import Enum
from datetime import datetime

try:
    from skimage import measure, morphology, filters, segmentation

    SKIMAGE_AVAILABLE = True
except ImportError as e:
    logging.error(f"Không thể import skimage: {e}")
    SKIMAGE_AVAILABLE = False

from quangtps.core.logging import get_logger

logger = get_logger(__name__)


class BooleanOperation(str, Enum):
    """Enum cho các phép toán Boolean giữa các cấu trúc."""

    UNION = "UNION"  # Phép hợp
    INTERSECTION = "INTERSECTION"  # Phép giao
    SUBTRACTION = "SUBTRACTION"  # Phép trừ
    EXCLUSIVE_OR = "EXCLUSIVE_OR"  # Phép XOR


class StructureOperations:
    """
    Lớp cung cấp các thao tác trên cấu trúc nâng cao.

    Bao gồm phép Boolean (AND, OR, NOT, XOR), làm mượt, lọc contour,
    và thao tác tạo cấu trúc phụ thuộc.
    """

    @staticmethod
    def boolean_operation(
        structure_a_contours: Dict[int, List[np.ndarray]],
        structure_b_contours: Dict[int, List[np.ndarray]],
        operation: BooleanOperation,
        pixel_spacing: Tuple[float, float] = (1.0, 1.0),
    ) -> Dict[int, List[np.ndarray]]:
        """
        Thực hiện phép toán Boolean giữa hai cấu trúc.

        Parameters
        ----------
        structure_a_contours : Dict[int, List[np.ndarray]]
            Contours của cấu trúc A, dạng {slice_index: [contour1, contour2, ...]}
        structure_b_contours : Dict[int, List[np.ndarray]]
            Contours của cấu trúc B, dạng {slice_index: [contour1, contour2, ...]}
        operation : BooleanOperation
            Loại phép toán Boolean (UNION, INTERSECTION, SUBTRACTION, EXCLUSIVE_OR)
        pixel_spacing : Tuple[float, float], optional
            Khoảng cách pixel theo mm, by default (1.0, 1.0)

        Returns
        -------
        Dict[int, List[np.ndarray]]
            Contours kết quả, dạng {slice_index: [contour1, contour2, ...]}
        """
        if not SKIMAGE_AVAILABLE:
            logger.error("Không thể thực hiện phép Boolean: thiếu thư viện skimage")
            return {}

        # Kết quả contours
        result_contours = {}

        # Tìm tất cả các slice có ít nhất một cấu trúc
        all_slices = set(
            list(structure_a_contours.keys()) + list(structure_b_contours.keys())
        )

        for slice_idx in all_slices:
            # Lấy các contours cho slice này
            contours_a = structure_a_contours.get(slice_idx, [])
            contours_b = structure_b_contours.get(slice_idx, [])

            # Nếu contours_a rỗng và contours_b rỗng, bỏ qua slice này
            if not contours_a and not contours_b:
                continue

            # Nếu contours_a rỗng, xử lý tùy thuộc loại phép toán
            if not contours_a:
                if (
                    operation == BooleanOperation.UNION
                    or operation == BooleanOperation.EXCLUSIVE_OR
                ):
                    result_contours[slice_idx] = contours_b.copy()
                continue

            # Nếu contours_b rỗng, xử lý tùy thuộc loại phép toán
            if not contours_b:
                if operation in (
                    BooleanOperation.UNION,
                    BooleanOperation.SUBTRACTION,
                    BooleanOperation.EXCLUSIVE_OR,
                ):
                    result_contours[slice_idx] = contours_a.copy()
                continue

            # Tìm bounding box bao phủ cả hai cấu trúc
            all_points = []
            for contour in contours_a + contours_b:
                all_points.extend(contour)

            if not all_points:
                continue

            all_points = np.vstack(all_points)
            min_x, min_y = np.floor(np.min(all_points, axis=0)).astype(int) - 10
            max_x, max_y = np.ceil(np.max(all_points, axis=0)).astype(int) + 10

            width = max_x - min_x
            height = max_y - min_y

            # Tạo mask cho cấu trúc A
            mask_a = np.zeros((height, width), dtype=np.uint8)
            for contour in contours_a:
                normalized_contour = contour.copy()
                normalized_contour[:, 0] -= min_x
                normalized_contour[:, 1] -= min_y

                # Vẽ và điền đầy contour
                rr, cc = StructureOperations._draw_polygon(
                    normalized_contour, (height, width)
                )
                if len(rr) > 0 and len(cc) > 0:
                    mask_a[rr, cc] = 1

            # Tạo mask cho cấu trúc B
            mask_b = np.zeros((height, width), dtype=np.uint8)
            for contour in contours_b:
                normalized_contour = contour.copy()
                normalized_contour[:, 0] -= min_x
                normalized_contour[:, 1] -= min_y

                # Vẽ và điền đầy contour
                rr, cc = StructureOperations._draw_polygon(
                    normalized_contour, (height, width)
                )
                if len(rr) > 0 and len(cc) > 0:
                    mask_b[rr, cc] = 1

            # Thực hiện phép toán Boolean
            result_mask = np.zeros((height, width), dtype=np.uint8)

            if operation == BooleanOperation.UNION:
                result_mask = np.logical_or(mask_a, mask_b).astype(np.uint8)
            elif operation == BooleanOperation.INTERSECTION:
                result_mask = np.logical_and(mask_a, mask_b).astype(np.uint8)
            elif operation == BooleanOperation.SUBTRACTION:
                result_mask = np.logical_and(mask_a, np.logical_not(mask_b)).astype(
                    np.uint8
                )
            elif operation == BooleanOperation.EXCLUSIVE_OR:
                result_mask = np.logical_xor(mask_a, mask_b).astype(np.uint8)

            # Chuyển đổi mask kết quả thành contours
            contours = measure.find_contours(result_mask, 0.5)

            # Lọc các contours nhỏ
            min_contour_size = 5  # Số điểm tối thiểu trong contour
            filtered_contours = [c for c in contours if len(c) >= min_contour_size]

            # Chuyển đổi contours về hệ tọa độ ban đầu
            result_contours_slice = []
            for contour in filtered_contours:
                # Hoán đổi x, y và chuyển về hệ tọa độ ban đầu
                adjusted_contour = np.fliplr(contour)
                adjusted_contour[:, 0] += min_x
                adjusted_contour[:, 1] += min_y
                result_contours_slice.append(adjusted_contour)

            # Thêm vào kết quả
            if result_contours_slice:
                result_contours[slice_idx] = result_contours_slice

        return result_contours

    @staticmethod
    def _draw_polygon(
        contour: np.ndarray, shape: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Vẽ polygon từ contour và trả về các tọa độ được điền.

        Parameters
        ----------
        contour : np.ndarray
            Các điểm của contour
        shape : Tuple[int, int]
            Kích thước (height, width) của mask

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Tọa độ hàng, cột của các điểm trong polygon
        """
        try:
            # Sử dụng skimage.draw.polygon nếu có
            from skimage import draw

            rr, cc = draw.polygon(contour[:, 1], contour[:, 0])

            # Lọc các điểm nằm ngoài mask
            valid_indices = (rr >= 0) & (rr < shape[0]) & (cc >= 0) & (cc < shape[1])
            rr, cc = rr[valid_indices], cc[valid_indices]

            return rr, cc
        except ImportError:
            # Fallback nếu không có skimage
            return np.array([]), np.array([])

    @staticmethod
    def extract_largest_connected_component(
        structure_contours: Dict[int, List[np.ndarray]],
        pixel_spacing: Tuple[float, float] = (1.0, 1.0),
    ) -> Dict[int, List[np.ndarray]]:
        """
        Trích xuất thành phần liên thông lớn nhất từ contours của cấu trúc.

        Parameters
        ----------
        structure_contours : Dict[int, List[np.ndarray]]
            Contours của cấu trúc, dạng {slice_index: [contour1, contour2, ...]}
        pixel_spacing : Tuple[float, float], optional
            Khoảng cách pixel theo mm, by default (1.0, 1.0)

        Returns
        -------
        Dict[int, List[np.ndarray]]
            Contours của thành phần liên thông lớn nhất
        """
        if not SKIMAGE_AVAILABLE:
            logger.error(
                "Không thể trích xuất thành phần liên thông: thiếu thư viện skimage"
            )
            return structure_contours

        result_contours = {}

        for slice_idx, contours in structure_contours.items():
            if not contours:
                continue

            # Tìm bounding box cho toàn bộ contours
            all_points = []
            for contour in contours:
                all_points.extend(contour)

            all_points = np.vstack(all_points)
            min_x, min_y = np.floor(np.min(all_points, axis=0)).astype(int) - 10
            max_x, max_y = np.ceil(np.max(all_points, axis=0)).astype(int) + 10

            width = max_x - min_x
            height = max_y - min_y

            # Tạo mask
            mask = np.zeros((height, width), dtype=np.uint8)
            for contour in contours:
                normalized_contour = contour.copy()
                normalized_contour[:, 0] -= min_x
                normalized_contour[:, 1] -= min_y

                # Vẽ và điền đầy contour
                rr, cc = StructureOperations._draw_polygon(
                    normalized_contour, (height, width)
                )
                if len(rr) > 0 and len(cc) > 0:
                    mask[rr, cc] = 1

            # Dán nhãn các thành phần liên thông
            labeled_mask, num_components = measure.label(
                mask, connectivity=2, return_num=True
            )

            if num_components == 0:
                continue

            # Tìm thành phần lớn nhất
            component_sizes = np.bincount(labeled_mask.ravel())[1:]  # Bỏ qua nền (0)
            largest_component = (
                np.argmax(component_sizes) + 1
            )  # +1 vì nhãn bắt đầu từ 1

            # Tạo mask chỉ với thành phần lớn nhất
            largest_mask = (labeled_mask == largest_component).astype(np.uint8)

            # Chuyển đổi mask thành contours
            large_contours = measure.find_contours(largest_mask, 0.5)

            # Lọc các contours nhỏ
            min_contour_size = 5  # Số điểm tối thiểu trong contour
            filtered_contours = [
                c for c in large_contours if len(c) >= min_contour_size
            ]

            # Chuyển đổi contours về hệ tọa độ ban đầu
            result_contours_slice = []
            for contour in filtered_contours:
                # Hoán đổi x, y và chuyển về hệ tọa độ ban đầu
                adjusted_contour = np.fliplr(contour)
                adjusted_contour[:, 0] += min_x
                adjusted_contour[:, 1] += min_y
                result_contours_slice.append(adjusted_contour)

            # Thêm vào kết quả
            if result_contours_slice:
                result_contours[slice_idx] = result_contours_slice

        return result_contours

    @staticmethod
    def smooth_contours(
        structure_contours: Dict[int, List[np.ndarray]],
        smoothing_factor: float = 0.5,
        method: str = "gaussian",
        pixel_spacing: Tuple[float, float] = (1.0, 1.0),
    ) -> Dict[int, List[np.ndarray]]:
        """
        Làm mượt contours của cấu trúc.

        Parameters
        ----------
        structure_contours : Dict[int, List[np.ndarray]]
            Contours của cấu trúc, dạng {slice_index: [contour1, contour2, ...]}
        smoothing_factor : float, optional
            Hệ số làm mượt, by default 0.5
        method : str, optional
            Phương pháp làm mượt ('gaussian', 'morphological', 'fourier'), by default "gaussian"
        pixel_spacing : Tuple[float, float], optional
            Khoảng cách pixel theo mm, by default (1.0, 1.0)

        Returns
        -------
        Dict[int, List[np.ndarray]]
            Contours đã được làm mượt
        """
        if not SKIMAGE_AVAILABLE:
            logger.error("Không thể làm mượt contours: thiếu thư viện skimage")
            return structure_contours

        result_contours = {}

        for slice_idx, contours in structure_contours.items():
            if not contours:
                continue

            # Xử lý cho từng contour
            smoothed_contours = []
            for contour in contours:
                if len(contour) < 5:  # Quá ít điểm, không làm mượt
                    smoothed_contours.append(contour)
                    continue

                if method == "gaussian":
                    # Làm mượt bằng Gaussian filter
                    sigma = max(0.1, smoothing_factor * 2.0)

                    # Tách x và y
                    x = contour[:, 0]
                    y = contour[:, 1]

                    # Áp dụng filter
                    x_smooth = filters.gaussian(x, sigma=sigma, mode="wrap")
                    y_smooth = filters.gaussian(y, sigma=sigma, mode="wrap")

                    # Kết hợp lại
                    smoothed_contour = np.column_stack([x_smooth, y_smooth])

                elif method == "morphological":
                    # Làm mượt bằng phương pháp morphological
                    # Tạo mask từ contour
                    # Tìm bounding box
                    min_x, min_y = np.floor(np.min(contour, axis=0)).astype(int) - 10
                    max_x, max_y = np.ceil(np.max(contour, axis=0)).astype(int) + 10

                    width = max_x - min_x
                    height = max_y - min_y

                    # Tạo mask
                    mask = np.zeros((height, width), dtype=np.uint8)

                    # Chuẩn hóa contour
                    normalized_contour = contour.copy()
                    normalized_contour[:, 0] -= min_x
                    normalized_contour[:, 1] -= min_y

                    # Vẽ và điền đầy contour
                    rr, cc = StructureOperations._draw_polygon(
                        normalized_contour, (height, width)
                    )
                    if len(rr) > 0 and len(cc) > 0:
                        mask[rr, cc] = 1

                    # Áp dụng phép mở và đóng để làm mượt
                    radius = int(max(1, smoothing_factor * 3))
                    selem = morphology.disk(radius)

                    # Mở trước, sau đó đóng
                    mask_smooth = morphology.binary_opening(mask, selem)
                    mask_smooth = morphology.binary_closing(mask_smooth, selem)

                    # Chuyển đổi mask thành contours
                    smooth_contours = measure.find_contours(mask_smooth, 0.5)

                    # Lấy contour dài nhất
                    if not smooth_contours:
                        smoothed_contours.append(
                            contour
                        )  # Giữ nguyên nếu không tìm được contour
                        continue

                    longest_contour = max(smooth_contours, key=len)

                    # Chuyển về hệ tọa độ ban đầu
                    smoothed_contour = np.fliplr(longest_contour)
                    smoothed_contour[:, 0] += min_x
                    smoothed_contour[:, 1] += min_y

                elif method == "fourier":
                    # Làm mượt bằng phương pháp Fourier
                    n_points = len(contour)
                    n_keep = max(5, int(n_points * (1 - smoothing_factor)))

                    # Tách x và y
                    x = contour[:, 0]
                    y = contour[:, 1]

                    # Chuyển đổi Fourier
                    x_fft = np.fft.fft(x)
                    y_fft = np.fft.fft(y)

                    # Loại bỏ tần số cao
                    x_fft[n_keep // 2 : -n_keep // 2] = 0
                    y_fft[n_keep // 2 : -n_keep // 2] = 0

                    # Chuyển đổi ngược lại
                    x_smooth = np.real(np.fft.ifft(x_fft))
                    y_smooth = np.real(np.fft.ifft(y_fft))

                    # Kết hợp lại
                    smoothed_contour = np.column_stack([x_smooth, y_smooth])

                else:
                    # Không làm mượt
                    smoothed_contour = contour

                smoothed_contours.append(smoothed_contour)

            # Thêm vào kết quả
            if smoothed_contours:
                result_contours[slice_idx] = smoothed_contours

        return result_contours

    @staticmethod
    def filter_small_contours(
        structure_contours: Dict[int, List[np.ndarray]],
        min_area_mm2: float = 1.0,
        pixel_spacing: Tuple[float, float] = (1.0, 1.0),
    ) -> Dict[int, List[np.ndarray]]:
        """
        Lọc bỏ các contour nhỏ từ structure.

        Parameters
        ----------
        structure_contours : Dict[int, List[np.ndarray]]
            Contours của cấu trúc, dạng {slice_index: [contour1, contour2, ...]}
        min_area_mm2 : float, optional
            Diện tích tối thiểu tính theo mm², by default 1.0
        pixel_spacing : Tuple[float, float], optional
            Khoảng cách pixel theo mm, by default (1.0, 1.0)

        Returns
        -------
        Dict[int, List[np.ndarray]]
            Contours sau khi lọc
        """
        if not SKIMAGE_AVAILABLE:
            logger.error("Không thể lọc contours: thiếu thư viện skimage")
            return structure_contours

        result_contours = {}

        # Chuyển đổi min_area_mm2 sang pixel
        min_area_px = min_area_mm2 / (pixel_spacing[0] * pixel_spacing[1])

        for slice_idx, contours in structure_contours.items():
            if not contours:
                continue

            # Lọc contours dựa trên diện tích
            filtered_contours = []
            for contour in contours:
                # Tính diện tích
                try:
                    area = 0.0
                    n = len(contour)

                    if n < 3:  # Không đủ điểm để tạo thành đa giác
                        continue

                    # Công thức tính diện tích đa giác
                    j = n - 1
                    for i in range(n):
                        area += (contour[j, 0] + contour[i, 0]) * (
                            contour[j, 1] - contour[i, 1]
                        )
                        j = i

                    area = abs(area) / 2.0

                    if area >= min_area_px:
                        filtered_contours.append(contour)

                except Exception as e:
                    logger.error(f"Lỗi khi tính diện tích contour: {e}")
                    filtered_contours.append(contour)  # Giữ nguyên contour nếu có lỗi

            # Thêm vào kết quả
            if filtered_contours:
                result_contours[slice_idx] = filtered_contours

        return result_contours

    @staticmethod
    def interpolate_contours(
        structure_contours: Dict[int, List[np.ndarray]],
        new_slice_indices: List[int],
        method: str = "linear",
    ) -> Dict[int, List[np.ndarray]]:
        """
        Nội suy contours giữa các slice.

        Parameters
        ----------
        structure_contours : Dict[int, List[np.ndarray]]
            Contours của cấu trúc, dạng {slice_index: [contour1, contour2, ...]}
        new_slice_indices : List[int]
            Các slice index cần nội suy
        method : str, optional
            Phương pháp nội suy ('linear', 'nearest'), by default "linear"

        Returns
        -------
        Dict[int, List[np.ndarray]]
            Contours sau khi nội suy
        """
        # Sao chép contours ban đầu
        result_contours = {k: v.copy() for k, v in structure_contours.items()}

        # Lấy các slice index hiện có
        existing_slices = sorted(list(structure_contours.keys()))

        if len(existing_slices) < 2:
            logger.warning("Cần ít nhất 2 slice để nội suy")
            return result_contours

        # Nội suy cho từng slice mới
        for new_idx in new_slice_indices:
            # Tìm các slice gần nhất
            lower_idx = None
            upper_idx = None

            for i, idx in enumerate(existing_slices):
                if idx < new_idx:
                    lower_idx = idx
                if idx > new_idx and upper_idx is None:
                    upper_idx = idx
                    break

            # Nếu không tìm được cả hai slice trên và dưới, bỏ qua
            if lower_idx is None or upper_idx is None:
                continue

            # Lấy contours tại các slice gần nhất
            lower_contours = structure_contours[lower_idx]
            upper_contours = structure_contours[upper_idx]

            # Nếu một trong hai không có contours, bỏ qua
            if not lower_contours or not upper_contours:
                continue

            # Lấy contour lớn nhất từ mỗi slice (đơn giản hóa vấn đề)
            lower_contour = max(lower_contours, key=len)
            upper_contour = max(upper_contours, key=len)

            # Chuẩn hóa các contour có cùng số điểm
            n_points = min(len(lower_contour), len(upper_contour))
            if n_points < 5:
                continue

            # Đơn giản hóa contours để có cùng số điểm
            from scipy.interpolate import interp1d

            # Tạo tham số hóa
            t_lower = np.linspace(0, 1, len(lower_contour))
            t_upper = np.linspace(0, 1, len(upper_contour))
            t_new = np.linspace(0, 1, n_points)

            # Nội suy các contour với cùng số điểm
            lower_x = interp1d(t_lower, lower_contour[:, 0], kind="linear")(t_new)
            lower_y = interp1d(t_lower, lower_contour[:, 1], kind="linear")(t_new)

            upper_x = interp1d(t_upper, upper_contour[:, 0], kind="linear")(t_new)
            upper_y = interp1d(t_upper, upper_contour[:, 1], kind="linear")(t_new)

            # Tính hệ số nội suy
            alpha = (new_idx - lower_idx) / (upper_idx - lower_idx)

            # Nội suy tuyến tính
            if method == "linear":
                new_x = lower_x + alpha * (upper_x - lower_x)
                new_y = lower_y + alpha * (upper_y - lower_y)
            else:  # nearest
                if alpha < 0.5:
                    new_x, new_y = lower_x, lower_y
                else:
                    new_x, new_y = upper_x, upper_y

            # Tạo contour mới
            new_contour = np.column_stack([new_x, new_y])

            # Thêm vào kết quả
            result_contours[new_idx] = [new_contour]

        return result_contours


def get_boolean_result_name(
    operation: BooleanOperation, name_a: str, name_b: str
) -> str:
    """
    Tạo tên cho cấu trúc kết quả từ phép toán Boolean.

    Parameters
    ----------
    operation : BooleanOperation
        Loại phép toán Boolean
    name_a : str
        Tên cấu trúc A
    name_b : str
        Tên cấu trúc B

    Returns
    -------
    str
        Tên gợi ý cho cấu trúc kết quả
    """
    if operation == BooleanOperation.UNION:
        return f"{name_a}_OR_{name_b}"
    elif operation == BooleanOperation.INTERSECTION:
        return f"{name_a}_AND_{name_b}"
    elif operation == BooleanOperation.SUBTRACTION:
        return f"{name_a}_SUB_{name_b}"
    elif operation == BooleanOperation.EXCLUSIVE_OR:
        return f"{name_a}_XOR_{name_b}"
    else:
        return f"{name_a}_{name_b}_result"
