#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module Contour Smoothing cho QuangTPS

Cung cấp các thuật toán làm mượt contour và cấu trúc giải phẫu
để giảm các đường viền răng cưa và cải thiện chất lượng phân đoạn.
"""

import numpy as np
import logging
from typing import List, Tuple, Optional, Union, Dict, Any
from scipy import ndimage
from skimage import measure, morphology, filters

try:
    # Nếu có PyTorch, thêm tối ưu GPU
    import torch
    from torch.nn import functional as F

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from quangtps.core.logging import get_logger

logger = get_logger(__name__)


def apply_smoothing(structure, smoothing_factor: float = 0.5):
    """
    Áp dụng làm mượt cho một structure.

    Args:
        structure: Structure cần làm mượt
        smoothing_factor: Hệ số làm mượt (0.0 - 1.0), 0 = không làm mượt, 1 = làm mượt tối đa

    Returns:
        Structure: Structure đã được làm mượt
    """
    if not hasattr(structure, "mask") or structure.mask is None:
        logger.warning(f"Structure {structure.name} không có mask để làm mượt")
        return structure

    # Copy structure để không thay đổi cấu trúc gốc
    try:
        import copy

        smoothed_structure = copy.deepcopy(structure)
    except:
        logger.warning("Deep copy không khả dụng, thay đổi structure gốc")
        smoothed_structure = structure

    # Áp dụng làm mượt lên mask
    try:
        smoothed_mask = smooth_binary_mask(structure.mask, strength=smoothing_factor)
        smoothed_structure.mask = smoothed_mask

        # Cập nhật contours từ mask mới nếu cần
        if hasattr(smoothed_structure, "update_contours_from_mask"):
            smoothed_structure.update_contours_from_mask()
        elif hasattr(smoothed_structure, "contours"):
            # Nếu không có phương thức cập nhật, tạo contours mới
            smoothed_structure.contours = create_contours_from_mask(smoothed_mask)

        logger.info(
            f"Đã làm mượt structure {structure.name} với hệ số {smoothing_factor}"
        )
    except Exception as e:
        logger.error(f"Lỗi khi làm mượt structure {structure.name}: {e}")

    return smoothed_structure


def smooth_binary_mask(
    mask: np.ndarray, strength: float = 0.5, method: str = "gaussian"
) -> np.ndarray:
    """
    Làm mượt mask nhị phân 3D.

    Args:
        mask: Mask nhị phân 3D
        strength: Cường độ làm mượt (0.0 - 1.0)
        method: Phương pháp làm mượt ('gaussian', 'morphological', 'fourier')

    Returns:
        np.ndarray: Mask đã được làm mượt
    """
    # Đảm bảo mask là nhị phân
    binary_mask = mask.astype(bool)

    # Giới hạn cường độ làm mượt
    strength = max(0.0, min(1.0, strength))

    if method == "gaussian":
        # Sử dụng lọc Gaussian
        sigma = 0.8 * strength  # Sigma tỷ lệ với cường độ làm mượt
        smoothed = ndimage.gaussian_filter(binary_mask.astype(float), sigma=sigma)
        # Chuyển về mask nhị phân với ngưỡng 0.5
        result = smoothed > 0.5

    elif method == "morphological":
        # Các toán tử hình thái học
        if strength < 0.1:
            # Với cường độ thấp, không làm gì
            return binary_mask

        # Size của cấu trúc phần tử dựa trên cường độ
        element_size = int(max(1, strength * 3))

        # Tạo cấu trúc phần tử hình cầu
        element = morphology.ball(element_size)

        # Áp dụng toán tử mở (erosion + dilation) để loại bỏ nhiễu
        opened = morphology.binary_opening(binary_mask, element)

        # Áp dụng toán tử đóng (dilation + erosion) để lấp lỗ nhỏ
        result = morphology.binary_closing(opened, element)

    elif method == "fourier":
        # Làm mượt trong không gian Fourier
        if strength < 0.01:
            return binary_mask

        # Chuyển đổi sang không gian Fourier
        fourier = np.fft.fftn(binary_mask.astype(float))

        # Tạo bộ lọc low-pass
        shape = binary_mask.shape
        x, y, z = np.ogrid[: shape[0], : shape[1], : shape[2]]
        center = (shape[0] // 2, shape[1] // 2, shape[2] // 2)

        # Tính khoảng cách từ tâm
        dist = np.sqrt(
            (x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2
        )

        # Tính bán kính ngắt
        cutoff = (1.0 - strength) * np.max(dist)

        # Tạo bộ lọc low-pass Butterworth
        n = 2  # Bậc của bộ lọc
        low_pass = 1.0 / (1.0 + (dist / cutoff) ** (2 * n))

        # Áp dụng bộ lọc
        filtered = fourier * low_pass

        # Chuyển về không gian thực
        smoothed = np.real(np.fft.ifftn(filtered))

        # Áp dụng ngưỡng
        result = smoothed > 0.5

    else:
        # Nếu phương pháp không được hỗ trợ, trả về mask gốc
        logger.warning(f"Phương pháp làm mượt '{method}' không được hỗ trợ")
        return binary_mask

    return result.astype(bool)


def smooth_contour(
    contour_points: np.ndarray, smoothing_factor: float = 0.5
) -> np.ndarray:
    """
    Làm mượt đường viền 2D được biểu diễn bởi tập hợp các điểm.

    Args:
        contour_points: Mảng các điểm (x, y) biểu diễn đường viền
        smoothing_factor: Hệ số làm mượt (0.0 - 1.0)

    Returns:
        np.ndarray: Mảng các điểm đã được làm mượt
    """
    if len(contour_points) < 3:
        return contour_points

    # Làm mượt bằng lọc thấp qua (moving average)
    smoothed = np.copy(contour_points).astype(float)

    # Kích thước cửa sổ dựa trên cường độ làm mượt và số điểm
    n_points = len(contour_points)
    window_size = max(3, int(n_points * smoothing_factor * 0.2))

    # Nếu cửa sổ lớn hơn số điểm, giảm kích thước
    if window_size >= n_points:
        window_size = max(3, n_points // 3)

    # Đảm bảo window_size là số lẻ
    if window_size % 2 == 0:
        window_size += 1

    # Tham số cho bộ lọc Savitzky-Golay
    polyorder = min(window_size - 1, 3)

    try:
        # Sử dụng bộ lọc Savitzky-Golay (cho kết quả tốt hơn moving average)
        from scipy.signal import savgol_filter

        # Tách tọa độ x và y
        x = contour_points[:, 0]
        y = contour_points[:, 1]

        # Làm mượt riêng cho x và y
        x_smoothed = savgol_filter(x, window_size, polyorder, mode="wrap")
        y_smoothed = savgol_filter(y, window_size, polyorder, mode="wrap")

        # Kết hợp lại
        smoothed = np.column_stack((x_smoothed, y_smoothed))
    except ImportError:
        # Nếu không có scipy, sử dụng moving average đơn giản
        half_window = window_size // 2

        # Làm mượt với điều kiện biên tuần hoàn
        for i in range(n_points):
            # Tạo chỉ số tuần hoàn
            indices = [(i + j) % n_points for j in range(-half_window, half_window + 1)]
            # Tính trung bình
            smoothed[i] = np.mean(contour_points[indices], axis=0)

    return smoothed


def create_contours_from_mask(mask: np.ndarray) -> List[np.ndarray]:
    """
    Tạo các đường viền từ mask 3D.

    Args:
        mask: Mask nhị phân 3D

    Returns:
        List[np.ndarray]: Danh sách các đường viền theo từng lát cắt
    """
    contours = []

    # Tạo đường viền cho từng lát cắt
    for z in range(mask.shape[0]):
        slice_mask = mask[z]
        if not np.any(slice_mask):
            continue  # Bỏ qua lát cắt trống

        # Tìm các đường viền trong lát cắt
        slice_contours = measure.find_contours(slice_mask.astype(float), 0.5)

        # Thêm thông tin z vào mỗi đường viền
        for contour in slice_contours:
            # Chuyển đổi từ (row, col) sang (x, y, z)
            points = np.column_stack(
                (contour[:, 1], contour[:, 0], np.full(len(contour), z))
            )
            contours.append(points)

    return contours


def smooth_surface_mesh(
    vertices: np.ndarray, faces: np.ndarray, iterations: int = 3
) -> np.ndarray:
    """
    Làm mượt lưới 3D bề mặt sử dụng thuật toán Laplacian Smoothing.

    Args:
        vertices: Mảng các đỉnh (x, y, z)
        faces: Mảng các mặt (i, j, k) chỉ ra chỉ số của 3 đỉnh tạo thành một mặt
        iterations: Số lần lặp thuật toán làm mượt

    Returns:
        np.ndarray: Mảng các đỉnh đã được làm mượt
    """
    smoothed_vertices = np.copy(vertices)

    # Tạo danh sách láng giềng cho mỗi đỉnh
    neighbors = [[] for _ in range(len(vertices))]

    # Xác định láng giềng từ các mặt
    for face in faces:
        i, j, k = face
        neighbors[i].extend([j, k])
        neighbors[j].extend([i, k])
        neighbors[k].extend([i, j])

    # Loại bỏ các láng giềng trùng lặp
    neighbors = [list(set(n)) for n in neighbors]

    # Áp dụng thuật toán làm mượt Laplacian
    for _ in range(iterations):
        new_vertices = np.copy(smoothed_vertices)

        # Cập nhật từng đỉnh
        for i, vertex_neighbors in enumerate(neighbors):
            if not vertex_neighbors:
                continue

            # Tính vị trí trung bình của các láng giềng
            neighbor_positions = smoothed_vertices[vertex_neighbors]
            neighbor_mean = np.mean(neighbor_positions, axis=0)

            # Di chuyển đỉnh hiện tại 50% về phía vị trí trung bình
            new_vertices[i] = 0.5 * smoothed_vertices[i] + 0.5 * neighbor_mean

        smoothed_vertices = new_vertices

    return smoothed_vertices


def optimize_contours(structure, simplify: bool = True, smooth: bool = True):
    """
    Tối ưu hóa contours của một structure bằng cách đơn giản hóa và làm mượt.

    Args:
        structure: Structure cần tối ưu
        simplify: Có đơn giản hóa contours không
        smooth: Có làm mượt contours không

    Returns:
        Structure: Structure đã được tối ưu
    """
    if not hasattr(structure, "contours") or not structure.contours:
        logger.warning(f"Structure {structure.name} không có contours để tối ưu")
        return structure

    # Copy structure để không thay đổi structure gốc
    try:
        import copy

        optimized = copy.deepcopy(structure)
    except:
        logger.warning("Deep copy không khả dụng, thay đổi structure gốc")
        optimized = structure

    # Đơn giản hóa contours nếu cần
    if simplify:
        try:
            from skimage.measure import approximate_polygon

            new_contours = []
            for contour in optimized.contours:
                # Đơn giản hóa đường viền, giữ z không đổi
                z_value = contour[0, 2] if contour.shape[1] > 2 else 0

                # Tách tọa độ x, y
                xy_contour = contour[:, :2]

                # Đơn giản hóa đường viền 2D
                tolerance = 0.5  # Dung sai tối đa
                simplified = approximate_polygon(xy_contour, tolerance)

                # Thêm lại tọa độ z
                if contour.shape[1] > 2:
                    z_column = np.full((simplified.shape[0], 1), z_value)
                    simplified = np.hstack((simplified, z_column))

                new_contours.append(simplified)

            optimized.contours = new_contours
        except Exception as e:
            logger.error(f"Lỗi khi đơn giản hóa contours: {e}")

    # Làm mượt contours nếu cần
    if smooth:
        try:
            new_contours = []
            for contour in optimized.contours:
                # Chỉ làm mượt nếu có đủ điểm
                if len(contour) > 5:
                    # Tách tọa độ x, y
                    xy_contour = contour[:, :2]

                    # Làm mượt đường viền 2D
                    smoothed_xy = smooth_contour(xy_contour, 0.3)

                    # Thêm lại tọa độ z
                    if contour.shape[1] > 2:
                        z_column = contour[:, 2:]
                        smoothed = np.hstack((smoothed_xy, z_column))
                    else:
                        smoothed = smoothed_xy

                    new_contours.append(smoothed)
                else:
                    new_contours.append(contour)

            optimized.contours = new_contours
        except Exception as e:
            logger.error(f"Lỗi khi làm mượt contours: {e}")

    return optimized
