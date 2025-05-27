#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module quản lý màu sắc và colormap cho QuangTPS.
Cung cấp các công cụ để chuyển đổi màu và tạo colormap tùy chỉnh.
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class ColorMap:
    """
    Lớp quản lý bản đồ màu và chuyển đổi giữa các định dạng màu khác nhau.
    """

    PREDEFINED_MAPS = {
        "eclipse": {
            100: (1.0, 0.0, 0.0),  # Đỏ
            95: (1.0, 0.5, 0.0),  # Cam
            90: (1.0, 1.0, 0.0),  # Vàng
            80: (0.0, 1.0, 0.0),  # Lục
            70: (0.0, 1.0, 1.0),  # Xanh ngọc
            60: (0.0, 0.5, 1.0),  # Xanh dương nhạt
            50: (0.0, 0.0, 1.0),  # Xanh dương
            40: (0.5, 0.0, 1.0),  # Tím nhạt
            30: (1.0, 0.0, 1.0),  # Hồng
            20: (0.7, 0.7, 0.7),  # Xám nhạt
            10: (0.5, 0.5, 0.5),  # Xám đậm
        },
        "rainbow": {
            100: (1.0, 0.0, 0.0),  # Đỏ
            95: (1.0, 0.2, 0.0),  # Đỏ đậm
            90: (1.0, 0.5, 0.0),  # Cam
            80: (1.0, 0.8, 0.0),  # Cam vàng
            70: (1.0, 1.0, 0.0),  # Vàng
            60: (0.5, 1.0, 0.0),  # Vàng lục
            50: (0.0, 1.0, 0.0),  # Lục
            40: (0.0, 1.0, 0.5),  # Lục lam
            30: (0.0, 1.0, 1.0),  # Lam
            20: (0.0, 0.5, 1.0),  # Dương
            10: (0.0, 0.0, 1.0),  # Xanh đậm
        },
        "hot": {
            100: (1.0, 0.0, 0.0),  # Đỏ
            90: (1.0, 0.2, 0.0),
            80: (1.0, 0.4, 0.0),
            70: (1.0, 0.6, 0.0),
            60: (1.0, 0.8, 0.0),
            50: (1.0, 1.0, 0.0),  # Vàng
            40: (1.0, 1.0, 0.25),
            30: (1.0, 1.0, 0.5),
            20: (1.0, 1.0, 0.75),
            10: (1.0, 1.0, 1.0),  # Trắng
        },
    }

    @staticmethod
    def rgb_to_hex(rgb: Tuple[float, float, float]) -> str:
        """
        Chuyển đổi màu RGB (0-1) sang chuỗi hex.

        Parameters:
        -----------
        rgb : tuple
            Tuple (r, g, b) với các giá trị từ 0-1.

        Returns:
        --------
        str
            Chuỗi hex theo định dạng #RRGGBB.
        """
        r, g, b = [int(x * 255) for x in rgb]
        return f"#{r:02x}{g:02x}{b:02x}"

    @staticmethod
    def hex_to_rgb(hex_color: str) -> Tuple[float, float, float]:
        """
        Chuyển đổi chuỗi hex sang RGB (0-1).

        Parameters:
        -----------
        hex_color : str
            Chuỗi hex theo định dạng #RRGGBB hoặc #RGB.

        Returns:
        --------
        tuple
            Tuple (r, g, b) với các giá trị từ 0-1.
        """
        hex_color = hex_color.lstrip("#")
        if len(hex_color) == 3:
            # #RGB format
            r, g, b = [int(c + c, 16) for c in hex_color]
        else:
            # #RRGGBB format
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)

        return (r / 255.0, g / 255.0, b / 255.0)

    @staticmethod
    def rgba_to_rgb(
        rgba: Tuple[float, float, float, float],
        bg_color: Tuple[float, float, float] = (1, 1, 1),
    ) -> Tuple[float, float, float]:
        """
        Chuyển đổi RGBA sang RGB bằng cách alpha blend với màu nền.

        Parameters:
        -----------
        rgba : tuple
            Tuple (r, g, b, a) với các giá trị từ 0-1.
        bg_color : tuple, optional
            Màu nền, mặc định là trắng (1, 1, 1).

        Returns:
        --------
        tuple
            Tuple RGB (r, g, b) với các giá trị từ 0-1.
        """
        r, g, b, a = rgba
        bg_r, bg_g, bg_b = bg_color

        r = r * a + bg_r * (1 - a)
        g = g * a + bg_g * (1 - a)
        b = b * a + bg_b * (1 - a)

        return (r, g, b)

    @classmethod
    def create_colormap(
        cls, color_map_name: str = "eclipse"
    ) -> Dict[int, Tuple[float, float, float]]:
        """
        Tạo colormap từ tên colormap có sẵn.

        Parameters:
        -----------
        color_map_name : str
            Tên colormap ("eclipse", "rainbow", "hot")

        Returns:
        --------
        dict
            Dictionary ánh xạ mức dose % sang màu RGB
        """
        return cls.PREDEFINED_MAPS.get(color_map_name, cls.PREDEFINED_MAPS["eclipse"])

    @classmethod
    def interpolate_color(
        cls,
        value: float,
        min_val: float,
        max_val: float,
        color1: Tuple[float, float, float],
        color2: Tuple[float, float, float],
    ) -> Tuple[float, float, float]:
        """
        Nội suy màu giữa hai màu dựa trên giá trị.

        Parameters:
        -----------
        value : float
            Giá trị cần nội suy
        min_val, max_val : float
            Giá trị min, max
        color1, color2 : tuple
            Màu bắt đầu và kết thúc

        Returns:
        --------
        tuple
            Màu RGB đã nội suy
        """
        if max_val == min_val:
            return color1

        t = (value - min_val) / (max_val - min_val)
        t = max(0.0, min(1.0, t))  # Clamp to [0, 1]

        r = color1[0] + t * (color2[0] - color1[0])
        g = color1[1] + t * (color2[1] - color1[1])
        b = color1[2] + t * (color2[2] - color1[2])

        return (r, g, b)

    @classmethod
    def get_dose_color(
        cls, dose_percent: float, colormap_name: str = "eclipse"
    ) -> Tuple[float, float, float]:
        """
        Lấy màu cho mức liều dựa trên phần trăm dose.

        Parameters:
        -----------
        dose_percent : float
            Phần trăm liều (0-100+)
        colormap_name : str
            Tên colormap

        Returns:
        --------
        tuple
            Màu RGB cho mức liều
        """
        colormap = cls.create_colormap(colormap_name)

        # Tìm hai mức liều gần nhất để nội suy
        levels = sorted(colormap.keys())

        # Trường hợp đặc biệt
        if dose_percent <= levels[0]:
            return colormap[levels[0]]
        if dose_percent >= levels[-1]:
            return colormap[levels[-1]]

        # Tìm khoảng để nội suy
        for i in range(len(levels) - 1):
            if levels[i] <= dose_percent <= levels[i + 1]:
                return cls.interpolate_color(
                    dose_percent,
                    levels[i],
                    levels[i + 1],
                    colormap[levels[i]],
                    colormap[levels[i + 1]],
                )

        return colormap[levels[-1]]


class ColorUtils:
    """
    Các tiện ích để làm việc với màu sắc.
    """

    @staticmethod
    def create_gradient_colormap(
        colors: List[Tuple[float, float, float]], n_colors: int = 256
    ) -> np.ndarray:
        """
        Tạo gradient colormap từ danh sách màu.

        Parameters:
        -----------
        colors : list
            Danh sách màu RGB
        n_colors : int
            Số màu trong colormap kết quả

        Returns:
        --------
        np.ndarray
            Mảng màu với shape (n_colors, 3)
        """
        if len(colors) < 2:
            raise ValueError("Cần ít nhất 2 màu để tạo gradient")

        n_segments = len(colors) - 1
        colors_per_segment = n_colors // n_segments

        result = []
        for i in range(n_segments):
            start_color = np.array(colors[i])
            end_color = np.array(colors[i + 1])

            segment_colors = []
            for j in range(colors_per_segment):
                t = j / (colors_per_segment - 1) if colors_per_segment > 1 else 0
                color = start_color + t * (end_color - start_color)
                segment_colors.append(color)

            result.extend(segment_colors)

        # Đảm bảo có đúng n_colors
        while len(result) < n_colors:
            result.append(colors[-1])

        return np.array(result[:n_colors])

    @staticmethod
    def adjust_brightness(
        color: Tuple[float, float, float], factor: float
    ) -> Tuple[float, float, float]:
        """
        Điều chỉnh độ sáng của màu.

        Parameters:
        -----------
        color : tuple
            Màu RGB gốc
        factor : float
            Hệ số điều chỉnh (1.0 = không đổi, >1.0 = sáng hơn, <1.0 = tối hơn)

        Returns:
        --------
        tuple
            Màu RGB đã điều chỉnh
        """
        r, g, b = color
        r = max(0.0, min(1.0, r * factor))
        g = max(0.0, min(1.0, g * factor))
        b = max(0.0, min(1.0, b * factor))
        return (r, g, b)

    @staticmethod
    def get_contrasting_color(
        color: Tuple[float, float, float],
    ) -> Tuple[float, float, float]:
        """
        Lấy màu tương phản (đen hoặc trắng) phù hợp với màu nền.

        Parameters:
        -----------
        color : tuple
            Màu nền RGB

        Returns:
        --------
        tuple
            Màu tương phản (đen hoặc trắng)
        """
        # Tính độ sáng theo công thức luminance
        r, g, b = color
        luminance = 0.299 * r + 0.587 * g + 0.114 * b

        # Trả về đen hoặc trắng tùy theo độ sáng
        return (0.0, 0.0, 0.0) if luminance > 0.5 else (1.0, 1.0, 1.0)

    @staticmethod
    def blend_colors(
        color1: Tuple[float, float, float],
        color2: Tuple[float, float, float],
        alpha: float,
    ) -> Tuple[float, float, float]:
        """
        Pha trộn hai màu với hệ số alpha.

        Parameters:
        -----------
        color1, color2 : tuple
            Hai màu RGB để pha trộn
        alpha : float
            Hệ số pha trộn (0.0 = color1, 1.0 = color2)

        Returns:
        --------
        tuple
            Màu RGB đã pha trộn
        """
        alpha = max(0.0, min(1.0, alpha))
        r = color1[0] * (1 - alpha) + color2[0] * alpha
        g = color1[1] * (1 - alpha) + color2[1] * alpha
        b = color1[2] * (1 - alpha) + color2[2] * alpha
        return (r, g, b)


# Hàm tiện ích
def get_eclipse_colormap() -> Dict[int, Tuple[float, float, float]]:
    """
    Lấy colormap Eclipse mặc định.

    Returns:
    --------
    dict
        Dictionary ánh xạ mức dose % sang màu RGB
    """
    return ColorMap.PREDEFINED_MAPS["eclipse"]


def create_matplotlib_colormap(
    colormap_name: str = "eclipse",
) -> mcolors.LinearSegmentedColormap:
    """
    Tạo matplotlib colormap từ colormap QuangTPS.

    Parameters:
    -----------
    colormap_name : str
        Tên colormap

    Returns:
    --------
    matplotlib.colors.LinearSegmentedColormap
        Matplotlib colormap
    """
    try:
        colormap = ColorMap.create_colormap(colormap_name)

        # Chuyển đổi sang format matplotlib
        colors = []
        positions = []

        levels = sorted(colormap.keys())
        for level in levels:
            positions.append(level / 100.0)  # Normalize to 0-1
            colors.append(colormap[level])

        # Tạo colormap
        cmap = mcolors.LinearSegmentedColormap.from_list(
            f"quangtps_{colormap_name}", list(zip(positions, colors))
        )

        return cmap

    except Exception as e:
        logger.error(f"Không thể tạo matplotlib colormap: {e}")
        # Fallback về colormap mặc định
        try:
            return plt.cm.get_cmap("jet")
        except:
            # Nếu jet không khả dụng, dùng colormap basic nhất
            return plt.cm.get_cmap("viridis")


# Export
__all__ = [
    "ColorMap",
    "ColorUtils",
    "get_eclipse_colormap",
    "create_matplotlib_colormap",
]
