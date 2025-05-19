#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module khởi tạo giao diện người dùng cho QuangTPS.

Module này cung cấp các hàm và lớp cơ bản cho khởi tạo và quản lý
giao diện người dùng đồ họa cho hệ thống QuangTPS.
"""

import os
import sys
import logging
from typing import Dict, List, Optional, Tuple, Union, Any, Type

# Khởi tạo logger
logger = logging.getLogger(__name__)

# Thử import PyQt5
try:
    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtCore import QObject
    from PyQt5.QtGui import QColor

    HAS_PYQT = True
    logger.debug("PyQt5 được import thành công.")
except ImportError:
    HAS_PYQT = False
    logger.warning("PyQt5 không khả dụng. UI sẽ hoạt động ở chế độ giả lập.")

    # Tạo các lớp giả
    class QApplication:
        @staticmethod
        def instance():
            return None

    class QObject:
        def __init__(self, *args, **kwargs):
            pass

    class pyqtSignal:
        def __init__(self, *args, **kwargs):
            pass

        def emit(self, *args, **kwargs):
            pass

    QColor = None

# Thử import matplotlib và cmaps thông dụng
try:
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    HAS_MATPLOTLIB = True
    logger.debug("Matplotlib được import thành công.")
except ImportError:
    HAS_MATPLOTLIB = False
    logger.warning("Matplotlib không khả dụng. Một số tính năng sẽ bị hạn chế.")

# Singleton instance cho UI signals
_ui_signals = None


def initialize_ui(use_eclipse_theme: bool = True) -> Optional[QApplication]:
    """
    Khởi tạo ứng dụng UI và các thành phần cần thiết.

    Parameters
    ----------
    use_eclipse_theme : bool, optional
        Nếu True, áp dụng theme Eclipse, mặc định là True

    Returns
    -------
    Optional[QApplication]
        Instance của QApplication hoặc None nếu PyQt không khả dụng
    """
    if not HAS_PYQT:
        logger.error(
            "PyQt5 không khả dụng. Không thể khởi tạo giao diện người dùng đồ họa."
        )
        return None

    # Khởi tạo QApplication nếu chưa có
    app = QApplication.instance()
    if app is None:
        try:
            app = QApplication(sys.argv)
        except Exception as e:
            logger.error(f"Không thể khởi tạo QApplication: {str(e)}")
            return None

    # Khởi tạo UI Signals
    global _ui_signals
    if _ui_signals is None:
        _ui_signals = UISignals()

    # Áp dụng Eclipse theme nếu cần
    if use_eclipse_theme:
        try:
            from quangtps.ui.eclipse_style_theme import apply_eclipse_theme

            apply_eclipse_theme(app)
            logger.info("Đã áp dụng Eclipse theme thành công.")
        except ImportError:
            logger.warning(
                "Không thể import eclipse_style_theme. Sử dụng giao diện mặc định."
            )
        except Exception as e:
            logger.error(f"Lỗi khi áp dụng Eclipse theme: {str(e)}")

    return app


def get_colormap_for_display(name: str, fallback_name: str = None) -> Dict:
    """
    Lấy colormap cho hiển thị với cơ chế dự phòng nhiều lớp.

    Hàm này sẽ thử lấy colormap theo thứ tự ưu tiên sau:
    1. Colormap Eclipse-style tùy chỉnh (nếu có)
    2. Colormap có tên được chỉ định
    3. Colormap dự phòng được chỉ định
    4. Colormap mặc định ('viridis' hoặc tương tự)
    5. Colormap nhị phân đơn giản nếu tất cả đều không khả dụng

    Parameters
    ----------
    name : str
        Tên của colormap cần lấy
    fallback_name : str, optional
        Tên của colormap dự phòng, mặc định là None

    Returns
    -------
    Dict
        Dict với các khóa cần thiết cho colormap
        (Nhằm chuẩn hóa output bất kể colormap nào được sử dụng)
    """
    result = {
        "name": name,
        "colors": None,  # Mảng màu RGB/RGBA
        "source": None,  # 'eclipse', 'matplotlib', 'custom', 'binary'
        "is_discrete": False,
        "is_fallback": False,
    }

    # Thử lấy Eclipse colormap (ưu tiên cao nhất)
    try:
        from quangtps.ui.eclipse_style_theme import get_eclipse_colormap

        eclipse_cmap = get_eclipse_colormap(name)
        if eclipse_cmap:
            result["colors"] = eclipse_cmap
            result["source"] = "eclipse"
            logger.debug(f"Đã sử dụng Eclipse colormap '{name}'")
            return result
    except Exception as e:
        logger.debug(f"Không thể lấy Eclipse colormap: {str(e)}")

    # Thử lấy từ matplotlib
    if HAS_MATPLOTLIB:
        try:
            # Thử lấy colormap được yêu cầu
            if hasattr(cm, name):
                cmap = getattr(cm, name)
                result["colors"] = [cmap(i) for i in range(256)]
                result["source"] = "matplotlib"
                logger.debug(f"Đã sử dụng matplotlib colormap '{name}'")
                return result

            # Thử lấy colormap dự phòng
            if fallback_name and hasattr(cm, fallback_name):
                result["is_fallback"] = True
                cmap = getattr(cm, fallback_name)
                result["colors"] = [cmap(i) for i in range(256)]
                result["name"] = fallback_name
                result["source"] = "matplotlib"
                logger.debug(f"Sử dụng matplotlib fallback colormap '{fallback_name}'")
                return result

            # Sử dụng viridis nếu có
            if hasattr(cm, "viridis"):
                result["is_fallback"] = True
                result["name"] = "viridis"
                cmap = cm.viridis
                result["colors"] = [cmap(i) for i in range(256)]
                result["source"] = "matplotlib"
                logger.debug(f"Sử dụng matplotlib default colormap 'viridis'")
                return result

            # Cuối cùng thử 'jet' hoặc 'hot'
            for default_cmap in ["jet", "hot", "plasma", "inferno"]:
                if hasattr(cm, default_cmap):
                    result["is_fallback"] = True
                    result["name"] = default_cmap
                    cmap = getattr(cm, default_cmap)
                    result["colors"] = [cmap(i) for i in range(256)]
                    result["source"] = "matplotlib"
                    logger.debug(
                        f"Sử dụng matplotlib fallback colormap '{default_cmap}'"
                    )
                    return result
        except Exception as e:
            logger.warning(f"Lỗi khi lấy matplotlib colormap: {str(e)}")

    # Tạo colormap nhị phân đơn giản (cực kỳ dự phòng)
    result["is_fallback"] = True
    result["name"] = "binary"
    result["source"] = "binary"
    result["is_discrete"] = True

    # Tạo colormap nhị phân đỏ-xanh đơn giản
    n_colors = 256
    colors = []
    for i in range(n_colors):
        if i < n_colors // 2:
            # Gradient từ trong suốt đến xanh
            alpha = (i / (n_colors // 2)) * 0.8
            colors.append((0, 0, 1, alpha))
        else:
            # Gradient từ xanh đến đỏ
            red = (i - n_colors // 2) / (n_colors // 2)
            blue = 1 - red
            colors.append((red, 0, blue, 0.8))

    result["colors"] = colors
    logger.debug("Sử dụng binary fallback colormap")

    return result


class UISignals(QObject):
    """Lớp tín hiệu chung cho giao diện người dùng."""

    dose_updated = pyqtSignal()  # Phát khi phân phối liều được cập nhật
    structures_changed = pyqtSignal()  # Phát khi cấu trúc thay đổi
    plan_loaded = pyqtSignal(object)  # Phát khi kế hoạch được tải
    plan_saved = pyqtSignal(str)  # Phát khi kế hoạch được lưu

    # Các tín hiệu khác
    ui_theme_changed = pyqtSignal(str)  # Phát khi theme giao diện thay đổi
    status_message = pyqtSignal(str, int)  # Phát để hiển thị thông báo trạng thái
    calculation_started = pyqtSignal(str)  # Phát khi bắt đầu tính toán (với mô tả)
    calculation_progress = pyqtSignal(
        int, str
    )  # Phát để cập nhật tiến độ (phần trăm, mô tả)
    calculation_finished = pyqtSignal(
        bool, str, object
    )  # Phát khi tính toán kết thúc (thành công, mô tả, kết quả)

    # Tín hiệu cho tab selection
    tab_changed = pyqtSignal(str)  # Phát khi người dùng chuyển tab (với tên tab)

    # Tín hiệu cho module DVH
    dvh_calculated = pyqtSignal(dict)  # Phát khi DVH được tính toán


def get_ui_signals() -> Optional[UISignals]:
    """
    Lấy instance của UISignals.

    Returns
    -------
    Optional[UISignals]
        Instance của UISignals hoặc None nếu chưa khởi tạo
    """
    global _ui_signals
    if _ui_signals is None:
        if HAS_PYQT:
            _ui_signals = UISignals()
    return _ui_signals


def create_color_from_hex(hex_color: str) -> Union[QColor, Tuple[int, int, int, int]]:
    """
    Tạo đối tượng màu từ mã hex.

    Parameters
    ----------
    hex_color : str
        Mã màu hex (ví dụ: "#FF5733")

    Returns
    -------
    Union[QColor, Tuple[int, int, int, int]]
        Đối tượng QColor nếu PyQt khả dụng, hoặc tuple (r,g,b,a)
    """
    hex_color = hex_color.lstrip("#")

    # Parse hex color
    if len(hex_color) == 3:  # Short form (#RGB)
        r = int(hex_color[0] + hex_color[0], 16)
        g = int(hex_color[1] + hex_color[1], 16)
        b = int(hex_color[2] + hex_color[2], 16)
        a = 255
    elif len(hex_color) == 4:  # Short form with alpha (#RGBA)
        r = int(hex_color[0] + hex_color[0], 16)
        g = int(hex_color[1] + hex_color[1], 16)
        b = int(hex_color[2] + hex_color[2], 16)
        a = int(hex_color[3] + hex_color[3], 16)
    elif len(hex_color) == 6:  # Normal form (#RRGGBB)
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        a = 255
    elif len(hex_color) == 8:  # Normal form with alpha (#RRGGBBAA)
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        a = int(hex_color[6:8], 16)
    else:
        # Invalid hex color
        r, g, b, a = 0, 0, 0, 255

    # Trả về QColor nếu có, nếu không thì tuple
    if HAS_PYQT and QColor:
        return QColor(r, g, b, a)
    else:
        return (r, g, b, a)


def create_placeholder_widget(
    message: str = "Chức năng này chưa được triển khai", parent=None
):
    """
    Tạo widget placeholder cho các tính năng chưa hoàn thiện.

    Parameters
    ----------
    message : str, optional
        Thông báo hiển thị trên widget
    parent : QWidget, optional
        Widget cha

    Returns
    -------
    QWidget
        Widget placeholder
    """
    if not HAS_PYQT:
        return None

    try:
        from PyQt5.QtWidgets import QLabel, QVBoxLayout, QWidget
        from PyQt5.QtCore import Qt

        widget = QWidget(parent)
        layout = QVBoxLayout(widget)

        label = QLabel(message)
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("color: #888; font-style: italic;")

        layout.addWidget(label)

        return widget
    except Exception as e:
        logger.error(f"Không thể tạo placeholder widget: {str(e)}")
        return None


# Xuất các lớp và hàm quan trọng
__all__ = [
    "initialize_ui",
    "get_colormap_for_display",
    "get_ui_signals",
    "create_color_from_hex",
    "create_placeholder_widget",
    "UISignals",
]
