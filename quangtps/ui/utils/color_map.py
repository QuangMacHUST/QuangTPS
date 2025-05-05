#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module cung cấp các tiện ích để quản lý màu sắc và bản đồ màu trong QuangTPS.
Bao gồm chuyển đổi giữa các định dạng màu và các bản đồ màu được xác định trước.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import get_cmap


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
        "grayscale": {
            100: (0.0, 0.0, 0.0),  # Đen
            90: (0.1, 0.1, 0.1),
            80: (0.2, 0.2, 0.2),
            70: (0.3, 0.3, 0.3),
            60: (0.4, 0.4, 0.4),
            50: (0.5, 0.5, 0.5),
            40: (0.6, 0.6, 0.6),
            30: (0.7, 0.7, 0.7),
            20: (0.8, 0.8, 0.8),
            10: (0.9, 0.9, 0.9),  # Trắng
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
    def rgb_to_hex(rgb):
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
    def hex_to_rgb(hex_color):
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
    def rgba_to_rgb(rgba, bg_color=(1, 1, 1)):
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

    @staticmethod
    def get_colormap_from_matplotlib(name, levels=10):
        """
        Tạo bản đồ màu từ colormap của Matplotlib.

        Parameters:
        -----------
        name : str
            Tên của colormap Matplotlib (ví dụ: 'jet', 'viridis', 'plasma').
        levels : int, optional
            Số lượng mức màu, mặc định là 10.

        Returns:
        --------
        dict
            Dictionary ánh xạ mức -> màu RGB.
        """
        try:
            cmap = get_cmap(name)
            colormap = {}

            for i in range(levels):
                level = 100 - (i * (100 / (levels - 1)) if levels > 1 else 0)
                level = round(level, 1)  # Làm tròn để tránh lỗi số thực
                colormap[level] = cmap(i / (levels - 1) if levels > 1 else 0)[:3]

            return colormap
        except:
            # Trả về bản đồ màu mặc định nếu lỗi
            return ColorMap.PREDEFINED_MAPS["eclipse"]

    @staticmethod
    def get_dose_color(dose_percentage, colormap=None):
        """
        Lấy màu tương ứng với phần trăm liều từ bản đồ màu.

        Parameters:
        -----------
        dose_percentage : float
            Phần trăm liều (0-100+).
        colormap : dict, optional
            Bản đồ màu, mặc định là Eclipse.

        Returns:
        --------
        tuple
            Màu RGB (r, g, b) với các giá trị từ 0-1.
        """
        if colormap is None:
            colormap = ColorMap.PREDEFINED_MAPS["eclipse"]

        # Nếu có chính xác mức liều trong bản đồ màu
        if dose_percentage in colormap:
            return colormap[dose_percentage]

        # Tìm hai mức gần nhất để nội suy
        levels = sorted(colormap.keys())

        # Nếu ngoài phạm vi, sử dụng mức đầu hoặc cuối
        if dose_percentage >= max(levels):
            return colormap[max(levels)]
        elif dose_percentage <= min(levels):
            return colormap[min(levels)]

        # Tìm hai mức để nội suy
        for i in range(len(levels) - 1):
            if (
                levels[i] >= dose_percentage >= levels[i + 1]
                or levels[i] <= dose_percentage <= levels[i + 1]
            ):
                lower_level = min(levels[i], levels[i + 1])
                upper_level = max(levels[i], levels[i + 1])
                break
        else:
            # Không tìm thấy khoảng phù hợp
            return colormap[min(levels, key=lambda x: abs(x - dose_percentage))]

        # Nội suy màu
        lower_color = colormap[lower_level]
        upper_color = colormap[upper_level]

        # Tính toán hệ số nội suy
        t = (dose_percentage - lower_level) / (upper_level - lower_level)

        # Nội suy tuyến tính
        r = lower_color[0] + t * (upper_color[0] - lower_color[0])
        g = lower_color[1] + t * (upper_color[1] - lower_color[1])
        b = lower_color[2] + t * (upper_color[2] - lower_color[2])

        return (r, g, b)

    @staticmethod
    def generate_vtk_lookup_table(colormap=None, alpha=1.0):
        """
        Tạo bảng tra cứu màu VTK từ bản đồ màu.

        Parameters:
        -----------
        colormap : dict, optional
            Bản đồ màu, mặc định là Eclipse.
        alpha : float, optional
            Giá trị alpha, mặc định là 1.0.

        Returns:
        --------
        vtkLookupTable
            Bảng tra cứu màu VTK.
        """
        try:
            import vtk

            if colormap is None:
                colormap = ColorMap.PREDEFINED_MAPS["eclipse"]

            # Tạo lookup table
            lut = vtk.vtkLookupTable()

            # Số lượng mục
            n_colors = 256
            lut.SetNumberOfTableValues(n_colors)

            # Phạm vi giá trị
            min_level = min(colormap.keys())
            max_level = max(colormap.keys())
            lut.SetTableRange(min_level, max_level)

            # Thiết lập màu cho từng mức
            for i in range(n_colors):
                level = min_level + (i / (n_colors - 1)) * (max_level - min_level)
                color = ColorMap.get_dose_color(level, colormap)
                lut.SetTableValue(i, color[0], color[1], color[2], alpha)

            lut.Build()
            return lut
        except ImportError:
            print("VTK không được cài đặt, không thể tạo bảng tra cứu màu VTK.")
            return None

    @staticmethod
    def plot_colormap(colormap=None, title="Bản đồ màu"):
        """
        Vẽ biểu đồ bản đồ màu.

        Parameters:
        -----------
        colormap : dict, optional
            Bản đồ màu, mặc định là Eclipse.
        title : str, optional
            Tiêu đề biểu đồ, mặc định là "Bản đồ màu".

        Returns:
        --------
        matplotlib.figure.Figure
            Đối tượng figure Matplotlib.
        """
        if colormap is None:
            colormap = ColorMap.PREDEFINED_MAPS["eclipse"]

        fig, ax = plt.subplots(figsize=(10, 2))

        # Các mức và màu
        levels = sorted(colormap.keys(), reverse=True)
        colors = [colormap[level] for level in levels]

        # Vẽ các ô màu
        for i, (level, color) in enumerate(zip(levels, colors)):
            ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=color))
            ax.text(
                i + 0.5,
                0.5,
                f"{level}%",
                ha="center",
                va="center",
                color="black" if sum(color) > 1.5 else "white",
            )

        ax.set_xlim(0, len(levels))
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title)

        plt.tight_layout()
        return fig


def get_eclipse_colormap():
    """
    Hàm tiện ích để lấy bản đồ màu Eclipse.

    Returns:
    --------
    dict
        Bản đồ màu Eclipse.
    """
    return ColorMap.PREDEFINED_MAPS["eclipse"].copy()


def get_rainbow_colormap():
    """
    Hàm tiện ích để lấy bản đồ màu Rainbow.

    Returns:
    --------
    dict
        Bản đồ màu Rainbow.
    """
    return ColorMap.PREDEFINED_MAPS["rainbow"].copy()


def get_hot_colormap():
    """
    Hàm tiện ích để lấy bản đồ màu Hot.

    Returns:
    --------
    dict
        Bản đồ màu Hot.
    """
    return ColorMap.PREDEFINED_MAPS["hot"].copy()


def get_grayscale_colormap():
    """
    Hàm tiện ích để lấy bản đồ màu Grayscale.

    Returns:
    --------
    dict
        Bản đồ màu Grayscale.
    """
    return ColorMap.PREDEFINED_MAPS["grayscale"].copy()


def get_colormap_from_name(name):
    """
    Lấy bản đồ màu dựa trên tên.

    Parameters:
    -----------
    name : str
        Tên bản đồ màu ('eclipse', 'rainbow', 'hot', 'grayscale' hoặc tên Matplotlib).

    Returns:
    --------
    dict
        Bản đồ màu tương ứng.
    """
    if name.lower() in ColorMap.PREDEFINED_MAPS:
        return ColorMap.PREDEFINED_MAPS[name.lower()].copy()
    else:
        # Thử lấy từ Matplotlib
        try:
            return ColorMap.get_colormap_from_matplotlib(name)
        except:
            # Trả về mặc định nếu không tìm thấy
            return get_eclipse_colormap()


# Test standalone
if __name__ == "__main__":
    # Vẽ các bản đồ màu được xác định trước
    for name in ColorMap.PREDEFINED_MAPS:
        fig = ColorMap.plot_colormap(
            ColorMap.PREDEFINED_MAPS[name], f"Bản đồ màu {name}"
        )
        plt.figure(fig.number)
        plt.show()

    # Thử một số bản đồ màu từ Matplotlib
    for name in ["viridis", "plasma", "inferno", "magma", "jet"]:
        try:
            cmap = ColorMap.get_colormap_from_matplotlib(name)
            fig = ColorMap.plot_colormap(cmap, f"Bản đồ màu {name} (Matplotlib)")
            plt.figure(fig.number)
            plt.show()
        except:
            print(f"Không thể tạo bản đồ màu {name} từ Matplotlib")
