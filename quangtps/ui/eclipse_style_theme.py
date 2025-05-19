#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module giao diện Eclipse-style cho QuangTPS.

Module này cung cấp stylesheets, colormaps và các thành phần giao diện
mô phỏng giao diện Eclipse TPS của Varian.
"""

import logging
import os
from typing import Dict, List, Optional, Tuple, Any, Union

logger = logging.getLogger(__name__)

# Thử import PyQt5
try:
    from PyQt5.QtWidgets import (
        QApplication,
        QWidget,
        QPushButton,
        QLabel,
        QComboBox,
        QTabWidget,
        QMainWindow,
    )
    from PyQt5.QtGui import QColor, QPalette, QFont
    from PyQt5.QtCore import Qt

    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False
    logger.warning("PyQt5 không khả dụng - chức năng tạo Eclipse theme bị hạn chế")

# Constants cho Eclipse theme
ECLIPSE_PRIMARY_COLOR = "#2D5B86"  # Xanh dương đậm của Eclipse
ECLIPSE_SECONDARY_COLOR = "#5C87B2"  # Xanh dương nhạt
ECLIPSE_ACCENT_COLOR = "#E27025"  # Cam của Varian
ECLIPSE_BG_COLOR = "#F5F5F5"  # Màu nền xám nhạt
ECLIPSE_TEXT_COLOR = "#333333"  # Màu chữ
ECLIPSE_HEADER_BG = "#E6E6E6"  # Màu nền header
ECLIPSE_ACTIVE_TAB_COLOR = "#FFFFFF"  # Màu nền tab đang chọn
ECLIPSE_INACTIVE_TAB_COLOR = "#E6E6E6"  # Màu nền tab không chọn

# Colormaps cho Eclipse-style
ECLIPSE_DOSE_COLORMAP = {
    0.0: (0.0, 0.0, 0.68, 0.3),  # Xanh dương đậm ở mức 0%
    0.1: (0.0, 0.0, 0.85, 0.5),  # Xanh dương nhạt ở mức 10%
    0.3: (0.0, 0.8, 0.8, 0.6),  # Xanh lá nhạt ở mức 30%
    0.5: (0.0, 0.9, 0.0, 0.7),  # Xanh lá ở mức 50%
    0.7: (0.9, 0.9, 0.0, 0.8),  # Vàng ở mức 70%
    0.9: (0.9, 0.45, 0.0, 0.9),  # Cam ở mức 90%
    1.0: (0.95, 0.0, 0.0, 1.0),  # Đỏ ở mức 100%
    1.1: (1.0, 0.0, 1.0, 1.0),  # Tím ở mức >100%
}

ECLIPSE_STRUCTURE_COLORS = {
    "PTV": (0.85, 0.45, 0.1, 0.8),  # Cam đặc trưng của Eclipse cho PTV
    "GTV": (0.9, 0.0, 0.0, 0.8),  # Đỏ cho GTV
    "CTV": (0.8, 0.6, 0.0, 0.7),  # Vàng cam cho CTV
    "BODY": (0.2, 0.5, 0.8, 0.3),  # Xanh nhạt cho BODY
    # Màu cho các cơ quan nguy cấp
    "SPINAL_CORD": (1.0, 1.0, 0.0, 0.8),  # Vàng
    "BRAINSTEM": (0.0, 0.8, 0.0, 0.7),  # Xanh lá
    "HEART": (0.8, 0.0, 0.0, 0.7),  # Đỏ
    "LUNG_RIGHT": (0.0, 0.6, 0.9, 0.5),  # Xanh da trời
    "LUNG_LEFT": (0.0, 0.7, 0.8, 0.5),  # Xanh da trời nhạt hơn
    "PAROTID_RIGHT": (0.8, 0.6, 0.8, 0.6),  # Tím nhạt
    "PAROTID_LEFT": (0.9, 0.7, 0.9, 0.6),  # Tím nhạt hơn
    "OPTIC_CHIASM": (1.0, 0.8, 0.0, 0.8),  # Vàng cam
    "BLADDER": (1.0, 1.0, 0.0, 0.7),  # Vàng
    "RECTUM": (0.7, 0.4, 0.0, 0.7),  # Nâu
    "FEMUR_HEAD_R": (0.0, 0.8, 0.8, 0.7),  # Xanh ngọc
    "FEMUR_HEAD_L": (0.0, 0.9, 0.9, 0.7),  # Xanh ngọc nhạt hơn
}

# Eclipse-style stylesheet
ECLIPSE_STYLESHEET = (
    """
QMainWindow {
    background-color: """
    + ECLIPSE_BG_COLOR
    + """;
    color: """
    + ECLIPSE_TEXT_COLOR
    + """;
}

QWidget {
    background-color: """
    + ECLIPSE_BG_COLOR
    + """;
    color: """
    + ECLIPSE_TEXT_COLOR
    + """;
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: 9pt;
}

QToolBar {
    background-color: """
    + ECLIPSE_HEADER_BG
    + """;
    border-bottom: 1px solid #D0D0D0;
    spacing: 2px;
}

QPushButton {
    background-color: #F0F0F0;
    color: """
    + ECLIPSE_TEXT_COLOR
    + """;
    border: 1px solid #C0C0C0;
    border-radius: 3px;
    padding: 4px 12px;
    min-height: 20px;
}

QPushButton:hover {
    background-color: #E3E3E3;
    border-color: #A0A0A0;
}

QPushButton:pressed {
    background-color: #D6D6D6;
}

QPushButton:disabled {
    background-color: #F8F8F8;
    color: #B0B0B0;
    border-color: #D8D8D8;
}

QPushButton[default=true] {
    background-color: """
    + ECLIPSE_PRIMARY_COLOR
    + """;
    color: white;
    border-color: """
    + ECLIPSE_SECONDARY_COLOR
    + """;
}

QPushButton[default=true]:hover {
    background-color: #3A6C9D;
}

QComboBox {
    background-color: white;
    selection-background-color: """
    + ECLIPSE_SECONDARY_COLOR
    + """;
    border: 1px solid #C0C0C0;
    border-radius: 3px;
    padding: 2px 18px 2px 4px;
    min-height: 20px;
}

QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 16px;
    border-left: 1px solid #C0C0C0;
}

QLineEdit {
    background-color: white;
    border: 1px solid #C0C0C0;
    border-radius: 3px;
    padding: 2px 4px;
    selection-background-color: """
    + ECLIPSE_SECONDARY_COLOR
    + """;
}

QTabWidget::pane {
    border: 1px solid #D0D0D0;
    border-top-width: 0px;
    background-color: white;
}

QTabBar::tab {
    background-color: """
    + ECLIPSE_INACTIVE_TAB_COLOR
    + """;
    border: 1px solid #D0D0D0;
    border-bottom: none;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    padding: 5px 10px;
    margin-right: 2px;
    min-width: 80px;
}

QTabBar::tab:selected {
    background-color: """
    + ECLIPSE_ACTIVE_TAB_COLOR
    + """;
    border-bottom: none;
}

QTabBar::tab:hover:!selected {
    background-color: #EFEFEF;
}

QTableView {
    border: 1px solid #D0D0D0;
    background-color: white;
    alternate-background-color: #F9F9F9;
    gridline-color: #E0E0E0;
}

QHeaderView::section {
    background-color: """
    + ECLIPSE_HEADER_BG
    + """;
    border: 1px solid #D0D0D0;
    padding: 4px;
}

QTreeView {
    border: 1px solid #D0D0D0;
    background-color: white;
    alternate-background-color: #FAFAFA;
}

QTreeView::item {
    padding: 2px;
    border: none;
}

QTreeView::item:selected {
    background-color: """
    + ECLIPSE_SECONDARY_COLOR
    + """;
    color: white;
}

QScrollBar:vertical {
    border: 1px solid #E0E0E0;
    background: white;
    width: 14px;
    margin: 15px 0 15px 0;
}

QScrollBar::handle:vertical {
    background: #C0C0C0;
    min-height: 20px;
    border-radius: 4px;
    margin: 2px;
}

QScrollBar::handle:vertical:hover {
    background: #A0A0A0;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    border: 1px solid #E0E0E0;
    background: #F0F0F0;
    height: 14px;
    subcontrol-origin: margin;
}

QScrollBar::add-line:vertical {
    subcontrol-position: bottom;
}

QScrollBar::sub-line:vertical {
    subcontrol-position: top;
}

QListView {
    border: 1px solid #D0D0D0;
    background-color: white;
}

QListView::item:selected {
    background-color: """
    + ECLIPSE_SECONDARY_COLOR
    + """;
    color: white;
}

QMenu {
    background-color: white;
    border: 1px solid #D0D0D0;
}

QMenu::item {
    padding: 5px 30px 5px 20px;
    border: 1px solid transparent;
}

QMenu::item:selected {
    background-color: """
    + ECLIPSE_SECONDARY_COLOR
    + """;
    color: white;
}

QMenu::separator {
    height: 1px;
    background-color: #D0D0D0;
    margin: 4px 0px;
}

QMenuBar {
    background-color: """
    + ECLIPSE_HEADER_BG
    + """;
    color: """
    + ECLIPSE_TEXT_COLOR
    + """;
}

QMenuBar::item {
    background: transparent;
    padding: 5px 10px;
}

QMenuBar::item:selected {
    background-color: #DADADA;
}

QGroupBox {
    border: 1px solid #D0D0D0;
    border-radius: 4px;
    margin-top: 10px;
    padding-top: 8px;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 5px;
}

QStatusBar {
    background-color: """
    + ECLIPSE_HEADER_BG
    + """;
    color: """
    + ECLIPSE_TEXT_COLOR
    + """;
}

QCheckBox::indicator {
    width: 15px;
    height: 15px;
}

QRadioButton::indicator {
    width: 14px;
    height: 14px;
}
"""
)


def apply_eclipse_theme(app: Optional[QApplication] = None) -> bool:
    """
    Áp dụng Eclipse-style theme cho ứng dụng.

    Parameters
    ----------
    app : Optional[QApplication]
        Đối tượng QApplication để áp dụng theme. Nếu None,
        sẽ sử dụng QApplication.instance().

    Returns
    -------
    bool
        True nếu áp dụng thành công, False nếu thất bại
    """
    if not PYQT_AVAILABLE:
        logger.error("PyQt5 không khả dụng. Không thể áp dụng Eclipse theme.")
        return False

    try:
        # Lấy instance hiện tại nếu app là None
        if app is None:
            app = QApplication.instance()
            if app is None:
                logger.error("Không tìm thấy QApplication instance.")
                return False

        # Áp dụng stylesheet chính
        app.setStyleSheet(ECLIPSE_STYLESHEET)

        # Thiết lập font mặc định
        default_font = QFont("Segoe UI", 9)
        app.setFont(default_font)

        # Thiết lập palette
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(ECLIPSE_BG_COLOR))
        palette.setColor(QPalette.WindowText, QColor(ECLIPSE_TEXT_COLOR))
        palette.setColor(QPalette.Base, QColor("white"))
        palette.setColor(QPalette.AlternateBase, QColor("#F9F9F9"))
        palette.setColor(QPalette.ToolTipBase, QColor("white"))
        palette.setColor(QPalette.ToolTipText, QColor(ECLIPSE_TEXT_COLOR))
        palette.setColor(QPalette.Text, QColor(ECLIPSE_TEXT_COLOR))
        palette.setColor(QPalette.Button, QColor("#F0F0F0"))
        palette.setColor(QPalette.ButtonText, QColor(ECLIPSE_TEXT_COLOR))
        palette.setColor(QPalette.Highlight, QColor(ECLIPSE_SECONDARY_COLOR))
        palette.setColor(QPalette.HighlightedText, QColor("white"))
        palette.setColor(QPalette.Link, QColor(ECLIPSE_PRIMARY_COLOR))
        palette.setColor(QPalette.LinkVisited, QColor("#9370DB"))
        app.setPalette(palette)

        logger.info("Eclipse-style theme đã được áp dụng thành công.")
        return True

    except Exception as e:
        logger.error(f"Lỗi khi áp dụng Eclipse theme: {str(e)}")
        return False


def get_eclipse_colormap(
    name: str = "dose",
) -> Dict[float, Tuple[float, float, float, float]]:
    """
    Trả về colormap kiểu Eclipse cho hiển thị dữ liệu.

    Parameters
    ----------
    name : str
        Loại colormap, có thể là "dose" cho hiển thị liều hoặc
        tên cấu trúc cụ thể, mặc định là "dose"

    Returns
    -------
    Dict[float, Tuple[float, float, float, float]]
        Colormap định dạng cặp giá trị và màu RGBA
    """
    # Colormap cho liều
    if name.lower() == "dose":
        return ECLIPSE_DOSE_COLORMAP

    # Colormap cho cấu trúc cụ thể
    if name.upper() in ECLIPSE_STRUCTURE_COLORS:
        # Đối với cấu trúc, trả về một colormap đơn giản với một màu
        color = ECLIPSE_STRUCTURE_COLORS[name.upper()]
        return {0.0: (0, 0, 0, 0), 1.0: color}

    # Trường hợp không tìm thấy, tạo màu ngẫu nhiên nhưng ổn định
    import hashlib

    # Tạo màu dựa trên hash của tên
    hash_obj = hashlib.md5(name.encode())
    hash_int = int(hash_obj.hexdigest(), 16)

    # Tạo màu RGB từ hash, đảm bảo màu không quá tối hoặc quá sáng
    r = ((hash_int & 0xFF0000) >> 16) / 255.0
    g = ((hash_int & 0x00FF00) >> 8) / 255.0
    b = (hash_int & 0x0000FF) / 255.0

    # Đảm bảo màu không quá tối
    r = 0.3 + (r * 0.7)
    g = 0.3 + (g * 0.7)
    b = 0.3 + (b * 0.7)

    # Trả về colormap với màu được tạo
    return {0.0: (0, 0, 0, 0), 1.0: (r, g, b, 0.7)}


def create_eclipse_widget_style(widget_type: str) -> str:
    """
    Tạo CSS style cho một loại widget cụ thể theo phong cách Eclipse.

    Parameters
    ----------
    widget_type : str
        Loại widget, ví dụ: "button", "table", "tab", "tree"

    Returns
    -------
    str
        CSS style cho widget
    """
    widget_type = widget_type.lower()

    if widget_type == "button":
        return """
        QPushButton {
            background-color: #F0F0F0;
            color: #333333;
            border: 1px solid #C0C0C0;
            border-radius: 3px;
            padding: 4px 12px;
            min-height: 20px;
        }
        QPushButton:hover {
            background-color: #E3E3E3;
            border-color: #A0A0A0;
        }
        QPushButton:pressed {
            background-color: #D6D6D6;
        }
        QPushButton:disabled {
            background-color: #F8F8F8;
            color: #B0B0B0;
            border-color: #D8D8D8;
        }
        """

    elif widget_type == "primary_button":
        return (
            """
        QPushButton {
            background-color: """
            + ECLIPSE_PRIMARY_COLOR
            + """;
            color: white;
            border: 1px solid """
            + ECLIPSE_SECONDARY_COLOR
            + """;
            border-radius: 3px;
            padding: 4px 12px;
            min-height: 20px;
        }
        QPushButton:hover {
            background-color: #3A6C9D;
        }
        QPushButton:pressed {
            background-color: #1F4A75;
        }
        QPushButton:disabled {
            background-color: #A9BCD3;
            color: #E6E6E6;
            border-color: #A9BCD3;
        }
        """
        )

    elif widget_type == "table":
        return (
            """
        QTableView {
            border: 1px solid #D0D0D0;
            background-color: white;
            alternate-background-color: #F9F9F9;
            gridline-color: #E0E0E0;
        }
        QHeaderView::section {
            background-color: """
            + ECLIPSE_HEADER_BG
            + """;
            border: 1px solid #D0D0D0;
            padding: 4px;
            font-weight: bold;
        }
        QTableView::item:selected {
            background-color: """
            + ECLIPSE_SECONDARY_COLOR
            + """;
            color: white;
        }
        """
        )

    elif widget_type == "tab":
        return (
            """
        QTabWidget::pane {
            border: 1px solid #D0D0D0;
            border-top-width: 0px;
            background-color: white;
        }
        QTabBar::tab {
            background-color: """
            + ECLIPSE_INACTIVE_TAB_COLOR
            + """;
            border: 1px solid #D0D0D0;
            border-bottom: none;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
            padding: 5px 10px;
            margin-right: 2px;
            min-width: 80px;
        }
        QTabBar::tab:selected {
            background-color: """
            + ECLIPSE_ACTIVE_TAB_COLOR
            + """;
            border-bottom: none;
        }
        QTabBar::tab:hover:!selected {
            background-color: #EFEFEF;
        }
        """
        )

    elif widget_type == "tree":
        return (
            """
        QTreeView {
            border: 1px solid #D0D0D0;
            background-color: white;
            alternate-background-color: #FAFAFA;
        }
        QTreeView::item {
            padding: 3px;
            border: none;
        }
        QTreeView::item:selected {
            background-color: """
            + ECLIPSE_SECONDARY_COLOR
            + """;
            color: white;
        }
        QTreeView::branch {
            background: transparent;
        }
        """
        )

    # Trả về rỗng nếu không tìm thấy loại widget
    return ""


def get_eclipse_icon(icon_name: str) -> str:
    """
    Trả về đường dẫn đến icon Eclipse-style.

    Parameters
    ----------
    icon_name : str
        Tên icon cần lấy

    Returns
    -------
    str
        Đường dẫn đến file icon
    """
    # Đường dẫn đến thư mục icon
    icon_dir = os.path.join(os.path.dirname(__file__), "icons", "new_icons")

    # Các đuôi file phổ biến cho icon
    extensions = [".png", ".svg", ".ico"]

    # Kiểm tra file với các đuôi
    for ext in extensions:
        icon_path = os.path.join(icon_dir, f"{icon_name}{ext}")
        if os.path.exists(icon_path):
            return icon_path

    # Trả về None nếu không tìm thấy
    return None


# Class EncapsulateWidget để cung cấp widget với style Eclipse
class EclipseStyleWidget:
    """
    Lớp utility để tạo và style widget theo phong cách Eclipse.
    """

    @staticmethod
    def button(text: str, is_primary: bool = False) -> QPushButton:
        """
        Tạo button với style Eclipse.

        Parameters
        ----------
        text : str
            Text hiển thị trên button
        is_primary : bool
            True nếu là button chính (màu xanh), False nếu là button thường

        Returns
        -------
        QPushButton
            Button đã được style
        """
        if not PYQT_AVAILABLE:
            logger.error("PyQt5 không khả dụng. Không thể tạo button.")
            return None

        button = QPushButton(text)

        if is_primary:
            button.setStyleSheet(create_eclipse_widget_style("primary_button"))
            button.setProperty("default", True)
        else:
            button.setStyleSheet(create_eclipse_widget_style("button"))

        return button

    @staticmethod
    def make_tab_widget() -> QTabWidget:
        """
        Tạo tab widget với style Eclipse.

        Returns
        -------
        QTabWidget
            Tab widget đã được style
        """
        if not PYQT_AVAILABLE:
            logger.error("PyQt5 không khả dụng. Không thể tạo tab widget.")
            return None

        tab_widget = QTabWidget()
        tab_widget.setStyleSheet(create_eclipse_widget_style("tab"))
        return tab_widget


# Thông tin phiên bản
__version__ = "0.7.8"
