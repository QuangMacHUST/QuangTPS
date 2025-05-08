#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module hiển thị liều.

Module này cung cấp các lớp và hàm để hiển thị phân phối liều trong 2D và 3D.
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
import os

# Import matplotlib và xử lý ngoại lệ
try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.colors import LinearSegmentedColormap

    # Kiểm tra xem matplotlib có phương thức register_cmap không
    HAS_MATPLOTLIB = True
    MATPLOTLIB_HAS_REGISTER_CMAP = hasattr(plt.cm, "register_cmap")
except ImportError:
    HAS_MATPLOTLIB = False
    MATPLOTLIB_HAS_REGISTER_CMAP = False

    # Tạo các lớp giả lắp để tránh lỗi linter
    class DummyLinearSegmentedColormap:
        def __init__(self, *args, **kwargs):
            pass

        @classmethod
        def from_list(cls, name, colors_list):
            return cls()

    class DummyColors:
        pass

    class DummyPlot:
        @staticmethod
        def register_cmap(name, cmap):
            pass

        class cm:
            @staticmethod
            def register_cmap(name, cmap):
                pass

    LinearSegmentedColormap = DummyLinearSegmentedColormap
    mcolors = DummyColors
    plt = DummyPlot

# Import OpenCV và xử lý ngoại lệ
try:
    import cv2

    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

    # Tạo module giả lắp cv2
    class cv2:
        # Constants
        RETR_EXTERNAL = 0
        CHAIN_APPROX_SIMPLE = 1

        @staticmethod
        def findContours(image, mode, method):
            """Mô phỏng tìm contour khi không có OpenCV"""
            logger.warning("OpenCV không khả dụng, sử dụng phương thức dự phòng")
            # Trả về danh sách rỗng
            return [], None


logger = logging.getLogger(__name__)

# Các bản đồ màu mặc định
DEFAULT_COLORMAPS = {
    "Eclipse": {
        "description": "Bản đồ màu tương tự Eclipse TPS",
        "colors": [
            (0.0, (0.0, 0.0, 1.0)),  # Dark blue for low dose
            (0.25, (0.0, 1.0, 1.0)),  # Cyan
            (0.5, (0.0, 1.0, 0.0)),  # Green
            (0.7, (1.0, 1.0, 0.0)),  # Yellow
            (0.9, (1.0, 0.5, 0.0)),  # Orange
            (1.0, (1.0, 0.0, 0.0)),  # Red for high dose
        ],
    },
    "Hot": {
        "description": "Bản đồ màu nóng",
        "colors": [
            (0.0, (0.0, 0.0, 0.0)),  # Black for low dose
            (0.25, (0.5, 0.0, 0.0)),  # Dark red
            (0.5, (1.0, 0.0, 0.0)),  # Red
            (0.75, (1.0, 0.5, 0.0)),  # Orange
            (1.0, (1.0, 1.0, 0.0)),  # Yellow for high dose
        ],
    },
    "Rainbow": {
        "description": "Bản đồ màu cầu vồng",
        "colors": [
            (0.0, (0.5, 0.0, 1.0)),  # Purple for low dose
            (0.2, (0.0, 0.0, 1.0)),  # Blue
            (0.4, (0.0, 1.0, 1.0)),  # Cyan
            (0.6, (0.0, 1.0, 0.0)),  # Green
            (0.8, (1.0, 1.0, 0.0)),  # Yellow
            (1.0, (1.0, 0.0, 0.0)),  # Red for high dose
        ],
    },
    "Gray": {
        "description": "Bản đồ màu xám",
        "colors": [
            (0.0, (0.0, 0.0, 0.0)),  # Black for low dose
            (1.0, (1.0, 1.0, 1.0)),  # White for high dose
        ],
    },
}


def get_colormap(
    name: str = "Eclipse", register: bool = True
) -> LinearSegmentedColormap:
    """
    Lấy bản đồ màu từ tên.

    Args:
        name: Tên bản đồ màu
        register: Có đăng ký bản đồ màu với matplotlib không

    Returns:
        Đối tượng LinearSegmentedColormap
    """
    if not HAS_MATPLOTLIB:
        logger.warning("Matplotlib không có sẵn, không thể tạo colormap")
        return None

    if name not in DEFAULT_COLORMAPS:
        logger.warning(f"Bản đồ màu {name} không tồn tại, sử dụng Eclipse thay thế")
        name = "Eclipse"

    cmap_def = DEFAULT_COLORMAPS[name]
    colors = cmap_def["colors"]

    # Tạo danh sách các điểm màu
    positions = [x[0] for x in colors]
    rgb_colors = [x[1] for x in colors]

    # Tạo colormap
    cmap = LinearSegmentedColormap.from_list(name, list(zip(positions, rgb_colors)))

    # Đăng ký với matplotlib nếu cần
    if register and MATPLOTLIB_HAS_REGISTER_CMAP:
        try:
            if hasattr(plt.cm, "register_cmap"):
                plt.cm.register_cmap(name=name, cmap=cmap)
            else:
                # Phương thức cũ hơn
                plt.register_cmap(name=name, cmap=cmap)
        except Exception as e:
            logger.warning(f"Không thể đăng ký colormap {name}: {e}")

    return cmap


def get_eclipse_colormap() -> Dict[int, Tuple[float, float, float]]:
    """
    Lấy bản đồ màu Eclipse cho các mức isodose.

    Returns:
        Từ điển ánh xạ mức % sang màu RGB
    """
    return {
        100: (1.0, 0.0, 0.0),  # Đỏ
        95: (1.0, 0.3, 0.0),  # Cam đỏ
        90: (1.0, 0.5, 0.0),  # Cam
        80: (1.0, 0.8, 0.0),  # Vàng cam
        70: (1.0, 1.0, 0.0),  # Vàng
        60: (0.5, 1.0, 0.0),  # Vàng lục
        50: (0.0, 1.0, 0.0),  # Lục
        40: (0.0, 1.0, 0.5),  # Lục lam
        30: (0.0, 1.0, 1.0),  # Lam
        20: (0.0, 0.5, 1.0),  # Lam xanh
        10: (0.0, 0.0, 1.0),  # Xanh
    }


def get_color_for_dose_value(
    value: float, max_dose: float, colormap_name: str = "Eclipse"
) -> Tuple[float, float, float]:
    """
    Lấy màu cho giá trị liều cụ thể.

    Args:
        value: Giá trị liều
        max_dose: Giá trị liều tối đa (dùng để chuẩn hóa)
        colormap_name: Tên bản đồ màu

    Returns:
        Màu RGB dưới dạng tuple (r, g, b)
    """
    if not HAS_MATPLOTLIB:
        # Trả về màu mặc định nếu không có matplotlib
        return (0.0, 0.0, 1.0)

    # Lấy colormap
    cmap = get_colormap(colormap_name)

    if cmap is None:
        # Sử dụng màu tùy thuộc vào giá trị liều nếu không có colormap
        norm_value = min(max(value / max_dose, 0.0), 1.0)
        if norm_value < 0.2:
            return (0.0, 0.0, 1.0)  # Xanh thẫm
        elif norm_value < 0.4:
            return (0.0, 1.0, 1.0)  # Lam
        elif norm_value < 0.6:
            return (0.0, 1.0, 0.0)  # Lục
        elif norm_value < 0.8:
            return (1.0, 1.0, 0.0)  # Vàng
        else:
            return (1.0, 0.0, 0.0)  # Đỏ

    # Chuẩn hóa giá trị
    norm_value = min(max(value / max_dose, 0.0), 1.0)

    # Lấy màu
    return cmap(norm_value)[:3]


def show_dose_2d(
    dose_data: np.ndarray,
    slice_idx: int = None,
    axis: int = 0,
    colormap: str = "Eclipse",
    alpha: float = 0.7,
    ax: Optional["plt.Axes"] = None,
    contour: bool = False,
    show_colorbar: bool = True,
    dose_range: Tuple[float, float] = None,
    mask: Optional[np.ndarray] = None,
    background: Optional[np.ndarray] = None,
    background_cmap: str = "gray",
):
    """
    Hiển thị phân phối liều 2D.

    Args:
        dose_data: Mảng 3D chứa dữ liệu liều
        slice_idx: Chỉ số lát cắt. Nếu None, sẽ lấy lát cắt giữa
        axis: Trục hiển thị (0: axial, 1: coronal, 2: sagittal)
        colormap: Tên bản đồ màu cho liều
        alpha: Độ trong suốt (0-1)
        ax: Trục matplotlib để vẽ
        contour: Có vẽ đường đồng liều không
        show_colorbar: Có hiển thị thanh màu không
        dose_range: Khoảng liều hiển thị (min, max)
        mask: Mặt nạ để chỉ hiển thị liều trong vùng mask
        background: Hình ảnh nền CT/MRI
        background_cmap: Bản đồ màu cho hình ảnh nền

    Returns:
        Đối tượng AxesImage
    """
    if not HAS_MATPLOTLIB:
        logger.error("Matplotlib không có sẵn, không thể hiển thị liều")
        return None

    if dose_data is None or len(dose_data.shape) != 3:
        logger.error("Dữ liệu liều không hợp lệ")
        return None

    # Xác định lát cắt
    if slice_idx is None:
        slice_idx = dose_data.shape[axis] // 2

    # Lấy lát cắt theo trục
    if axis == 0:
        dose_slice = dose_data[slice_idx, :, :]
        xlabel, ylabel = "X", "Y"
    elif axis == 1:
        dose_slice = dose_data[:, slice_idx, :]
        xlabel, ylabel = "X", "Z"
    else:  # axis == 2
        dose_slice = dose_data[:, :, slice_idx]
        xlabel, ylabel = "Y", "Z"

    # Áp dụng mask nếu có
    if mask is not None:
        mask_slice = None
        if axis == 0:
            if mask.shape[0] > slice_idx:
                mask_slice = mask[slice_idx, :, :]
        elif axis == 1:
            if mask.shape[1] > slice_idx:
                mask_slice = mask[:, slice_idx, :]
        else:  # axis == 2
            if mask.shape[2] > slice_idx:
                mask_slice = mask[:, :, slice_idx]

        if mask_slice is not None:
            dose_slice = np.where(mask_slice, dose_slice, 0)

    # Tạo axes nếu cần
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    # Lấy khoảng liều
    if dose_range is None:
        vmin, vmax = 0, np.max(dose_data)
    else:
        vmin, vmax = dose_range

    # Vẽ hình ảnh nền nếu có
    if background is not None:
        bg_slice = None
        if axis == 0 and background.shape[0] > slice_idx:
            bg_slice = background[slice_idx, :, :]
        elif axis == 1 and background.shape[1] > slice_idx:
            bg_slice = background[:, slice_idx, :]
        elif axis == 2 and background.shape[2] > slice_idx:
            bg_slice = background[:, :, slice_idx]

        if bg_slice is not None:
            ax.imshow(bg_slice, cmap=background_cmap, aspect="equal")

    # Vẽ phân phối liều
    if contour:
        # Vẽ đường đồng liều
        contour_levels = np.linspace(vmin, vmax, 10)
        im = ax.contour(dose_slice, levels=contour_levels, cmap=colormap, alpha=alpha)
    else:
        # Vẽ phân phối liều dạng colorwash
        im = ax.imshow(
            dose_slice, cmap=colormap, vmin=vmin, vmax=vmax, alpha=alpha, aspect="equal"
        )

    # Thêm thanh màu
    if show_colorbar:
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Liều (Gy)")

    # Thêm nhãn
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"Phân phối liều - Lát cắt {slice_idx}")

    return im


def create_dose_dvh(
    dose_data: np.ndarray,
    structure_masks: Dict[str, np.ndarray],
    structure_colors: Dict[str, Tuple[float, float, float]] = None,
    dose_bins: int = 100,
    ax: Optional["plt.Axes"] = None,
    relative_volume: bool = True,
    relative_dose: bool = False,
    prescription: float = None,
    grid_on: bool = True,
):
    """
    Tạo biểu đồ DVH (Dose Volume Histogram).

    Args:
        dose_data: Mảng 3D chứa dữ liệu liều (Gy)
        structure_masks: Từ điển các mask cấu trúc {tên: mask}
        structure_colors: Từ điển màu cấu trúc {tên: màu}
        dose_bins: Số khoảng liều
        ax: Trục matplotlib để vẽ
        relative_volume: Sử dụng thể tích tương đối (%)
        relative_dose: Sử dụng liều tương đối (% của liều chỉ định)
        prescription: Liều chỉ định (Gy) cho đồ thị liều tương đối
        grid_on: Có hiển thị lưới không

    Returns:
        Đối tượng Axes
    """
    if not HAS_MATPLOTLIB:
        logger.error("Matplotlib không có sẵn, không thể tạo DVH")
        return None

    if dose_data is None or len(dose_data.shape) != 3:
        logger.error("Dữ liệu liều không hợp lệ")
        return None

    if not structure_masks:
        logger.error("Không có cấu trúc để tạo DVH")
        return None

    # Tạo axes nếu cần
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    # Xác định khoảng liều
    dose_max = np.max(dose_data)
    if dose_max <= 0:
        logger.error("Giá trị liều tối đa không hợp lệ")
        return ax

    if relative_dose and prescription is None:
        logger.warning("Không có liều chỉ định, sử dụng liều tuyệt đối")
        relative_dose = False

    dose_edges = np.linspace(0, dose_max * 1.05, dose_bins + 1)

    # Tính DVH cho từng cấu trúc
    for name, mask in structure_masks.items():
        if mask.shape != dose_data.shape:
            logger.warning(f"Kích thước mask không khớp với dữ liệu liều cho {name}")
            continue

        # Lấy giá trị liều trong cấu trúc
        structure_dose = dose_data[mask > 0]

        if len(structure_dose) == 0:
            logger.warning(f"Không có voxel trong cấu trúc {name}")
            continue

        # Tính histogram
        hist, edges = np.histogram(structure_dose, bins=dose_edges)

        # Tính thể tích tích lũy
        cum_vol = np.cumsum(hist[::-1])[::-1]

        # Chuyển đổi sang % nếu cần
        if relative_volume:
            cum_vol = cum_vol / cum_vol[0] * 100 if cum_vol[0] > 0 else cum_vol

        # Chuyển đổi liều sang % nếu cần
        if relative_dose and prescription:
            dose_points = edges[:-1] / prescription * 100
        else:
            dose_points = edges[:-1]

        # Lấy màu cấu trúc
        color = None
        if structure_colors and name in structure_colors:
            color = structure_colors[name]

        # Vẽ đường DVH
        label = f"{name} ({cum_vol[0]:.1f} cc)" if not relative_volume else name
        ax.plot(dose_points, cum_vol, label=label, color=color)

    # Thiết lập trục và nhãn
    if relative_volume:
        ax.set_ylabel("Thể tích (%)")
    else:
        ax.set_ylabel("Thể tích (cc)")

    if relative_dose:
        ax.set_xlabel("Liều (% của liều chỉ định)")
    else:
        ax.set_xlabel("Liều (Gy)")

    ax.set_title("Biểu đồ Dose-Volume Histogram (DVH)")
    ax.set_xlim(
        0,
        dose_edges[-1]
        if not relative_dose
        else dose_edges[-1] / prescription * 100
        if prescription
        else 100,
    )
    ax.set_ylim(0, 105 if relative_volume else None)

    # Thêm lưới
    if grid_on:
        ax.grid(True, linestyle="--", alpha=0.7)

    # Thêm chú thích
    ax.legend(loc="best")

    return ax


def colorwash_to_contour(
    dose_data: np.ndarray,
    isodose_levels: List[float],
    axis: int = 0,
    slice_idx: int = None,
):
    """
    Chuyển đổi phân phối liều sang các đường đồng liều.

    Args:
        dose_data: Mảng 3D chứa dữ liệu liều
        isodose_levels: Danh sách các mức liều (Gy)
        axis: Trục hiển thị (0: axial, 1: coronal, 2: sagittal)
        slice_idx: Chỉ số lát cắt. Nếu None, sẽ lấy lát cắt giữa

    Returns:
        Danh sách các đường contour cho mỗi mức liều
    """
    if dose_data is None or len(dose_data.shape) != 3:
        logger.error("Dữ liệu liều không hợp lệ")
        return None

    # Xác định lát cắt
    if slice_idx is None:
        slice_idx = dose_data.shape[axis] // 2

    # Lấy lát cắt theo trục
    if axis == 0:
        dose_slice = dose_data[slice_idx, :, :]
    elif axis == 1:
        dose_slice = dose_data[:, slice_idx, :]
    else:  # axis == 2
        dose_slice = dose_data[:, :, slice_idx]

    # Tìm contour cho mỗi mức liều
    contours_by_level = {}

    for level in isodose_levels:
        # Tạo mask cho mức liều
        mask = (dose_slice >= level).astype(np.uint8)

        # Tìm contour
        try:
            if HAS_CV2:
                # Sử dụng OpenCV (nhanh hơn)
                contours_by_level[level] = cv2.findContours(
                    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )[0]
            else:
                # Sử dụng scikit-image nếu có
                try:
                    import skimage.measure as measure

                    contours = measure.find_contours(mask, 0.5)
                    contours_by_level[level] = contours
                except ImportError:
                    logger.error(
                        "Cần cài đặt OpenCV hoặc scikit-image để chuyển đổi sang contour"
                    )
                    contours_by_level[level] = []
        except Exception as e:
            logger.error(f"Lỗi khi tìm contour: {e}")
            contours_by_level[level] = []

    return contours_by_level
