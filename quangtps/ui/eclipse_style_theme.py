#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module eclipse_style_theme cho QuangTPS.

Module này cung cấp các hàm và tiện ích để áp dụng giao diện theo phong cách
Eclipse của Varian cho các widget trong hệ thống.
"""

import logging
import os
from typing import Dict, List, Optional, Tuple, Any, Union

logger = logging.getLogger(__name__)

# Kiểm tra PyQt
try:
    from PyQt5.QtWidgets import (
        QApplication,
        QWidget,
        QPushButton,
        QLabel,
        QComboBox,
        QTabWidget,
        QMainWindow,
        QStyleFactory,
        QGraphicsDropShadowEffect,
        QFrame,
        QToolBar,
        QMenu,
    )
    from PyQt5.QtCore import Qt, QSize, QPointF
    from PyQt5.QtGui import (
        QPalette,
        QColor,
        QFont,
        QBrush,
        QLinearGradient,
        QPixmap,
        QPainter,
        QPen,
        QIcon,
        QPolygonF,
    )

    HAS_PYQT = True
except ImportError:
    logger.warning("PyQt5 không khả dụng. Các chức năng stylesheet sẽ bị giới hạn.")
    HAS_PYQT = False

# Constants cho Eclipse theme
ECLIPSE_PRIMARY_COLOR = "#2D5B86"  # Xanh dương đậm của Eclipse
ECLIPSE_SECONDARY_COLOR = "#5C87B2"  # Xanh dương nhạt
ECLIPSE_ACCENT_COLOR = "#E27025"  # Cam của Varian
ECLIPSE_BG_COLOR = "#F5F5F5"  # Màu nền xám nhạt
ECLIPSE_TEXT_COLOR = "#333333"  # Màu chữ
ECLIPSE_HEADER_BG = "#E6E6E6"  # Màu nền header
ECLIPSE_ACTIVE_TAB_COLOR = "#FFFFFF"  # Màu nền tab đang chọn
ECLIPSE_INACTIVE_TAB_COLOR = "#E6E6E6"  # Màu nền tab không chọn

# Định nghĩa ECLIPSE_COLORS cho sử dụng trong các hàm style
ECLIPSE_COLORS = {
    "primary": ECLIPSE_PRIMARY_COLOR,
    "secondary": ECLIPSE_SECONDARY_COLOR,
    "accent": ECLIPSE_ACCENT_COLOR,
    "background": ECLIPSE_BG_COLOR,
    "text": ECLIPSE_TEXT_COLOR,
    "header": ECLIPSE_HEADER_BG,
    "tab_active": ECLIPSE_ACTIVE_TAB_COLOR,
    "tab_inactive": ECLIPSE_INACTIVE_TAB_COLOR,
    "text_light": "#FFFFFF",  # Màu chữ sáng
    "border": "#E0E0E0",  # Màu đường viền
    "ptv": "#E27025",  # Màu PTV kiểu Eclipse
    "oar": "#5C87B2",  # Màu OAR kiểu Eclipse
    "isodose": "#2D8659",  # Màu đường isodose
}

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
    if not HAS_PYQT:
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


def create_eclipse_widget_style(
    widget_type: str = "default", custom_colors: Dict[str, str] = None
) -> str:
    """
    Tạo stylesheet theo phong cách Eclipse cho widget.

    Parameters
    ----------
    widget_type : str
        Loại widget cần áp dụng style (default, button, table, tab, etc.)
    custom_colors : Dict[str, str], optional
        Tùy chỉnh màu sắc nếu cần

    Returns
    -------
    str
        Stylesheet CSS
    """
    if not HAS_PYQT:
        return ""

    colors = ECLIPSE_COLORS.copy()
    if custom_colors:
        colors.update(custom_colors)

    # Stylesheet chung cho tất cả các widget
    base_style = f"""
    QWidget {{
        font-family: 'Segoe UI', 'Open Sans', Arial, sans-serif;
        color: {colors["text"]};
        background-color: {colors["background"]};
    }}
    """

    # Stylesheet riêng theo loại widget
    if widget_type == "button":
        return (
            base_style
            + f"""
        QPushButton {{
            background-color: {colors["primary"]};
            color: {colors["text_light"]};
            border: none;
            border-radius: 3px;
            padding: 6px 12px;
            font-weight: 500;
        }}

        QPushButton:hover {{
            background-color: {colors["secondary"]};
        }}

        QPushButton:pressed {{
            background-color: {colors["accent"]};
        }}

        QPushButton:disabled {{
            background-color: #B0BEC5;
            color: #78909C;
        }}
        """
        )
    elif widget_type == "table":
        return (
            base_style
            + f"""
        QTableWidget {{
            border: 1px solid #E0E0E0;
            gridline-color: #E0E0E0;
            selection-background-color: {colors["accent"]};
            selection-color: {colors["text_light"]};
        }}

        QTableWidget::item {{
            padding: 4px;
            border-bottom: 1px solid #F5F5F5;
        }}

        QHeaderView::section {{
            background-color: #EEEEEE;
            font-weight: bold;
            padding: 4px;
            border: none;
            border-right: 1px solid #E0E0E0;
            border-bottom: 1px solid #E0E0E0;
        }}
        """
        )
    elif widget_type == "tab":
        return (
            base_style
            + f"""
        QTabWidget::pane {{
            border: 1px solid #E0E0E0;
            border-top: 0px;
            border-radius: 0px 0px 3px 3px;
        }}

        QTabBar::tab {{
            background-color: #EEEEEE;
            color: {colors["text"]};
            padding: 6px 12px;
            border: 1px solid #E0E0E0;
            border-bottom: none;
            border-top-left-radius: 3px;
            border-top-right-radius: 3px;
            margin-right: 2px;
        }}

        QTabBar::tab:selected {{
            background-color: {colors["primary"]};
            color: {colors["text_light"]};
            border: 1px solid {colors["primary"]};
            border-bottom: none;
        }}

        QTabBar::tab:!selected:hover {{
            background-color: #E0E0E0;
        }}
        """
        )
    elif widget_type == "dvh":
        return (
            base_style
            + f"""
        QGroupBox {{
            border: 1px solid #E0E0E0;
            border-radius: 3px;
            margin-top: 0.5em;
            padding-top: 1em;
            font-weight: bold;
        }}

        QGroupBox::title {{
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px 0 5px;
        }}

        QComboBox {{
            border: 1px solid #E0E0E0;
            border-radius: 3px;
            padding: 3px 5px;
            background-color: white;
            selection-background-color: {colors["primary"]};
        }}

        QComboBox::drop-down {{
            subcontrol-origin: padding;
            subcontrol-position: top right;
            width: 15px;
            border-left: 1px solid #E0E0E0;
        }}

        QCheckBox {{
            spacing: 5px;
        }}

        QCheckBox::indicator {{
            width: 15px;
            height: 15px;
        }}

        QCheckBox::indicator:unchecked {{
            border: 1px solid #E0E0E0;
            background-color: white;
            border-radius: 3px;
        }}

        QCheckBox::indicator:checked {{
            border: 1px solid {colors["primary"]};
            background-color: {colors["primary"]};
            border-radius: 3px;
        }}
        """
        )
    else:
        # Default style
        return base_style


def apply_eclipse_style_theme(widget: "QWidget", widget_type: str = "default") -> None:
    """
    Áp dụng theme Eclipse cho widget.

    Parameters
    ----------
    widget : QWidget
        Widget cần áp dụng theme
    widget_type : str, optional
        Loại widget để áp dụng style phù hợp
    """
    if not HAS_PYQT:
        return

    # Áp dụng stylesheet
    widget.setStyleSheet(create_eclipse_widget_style(widget_type))


def get_structure_color(structure_type: str) -> Tuple[int, int, int]:
    """
    Lấy màu sắc mặc định cho cấu trúc dựa vào loại cấu trúc.

    Parameters
    ----------
    structure_type : str
        Loại cấu trúc ("PTV", "OAR", "OTHER")

    Returns
    -------
    Tuple[int, int, int]
        Màu sắc RGB
    """
    structure_type = structure_type.upper() if isinstance(structure_type, str) else ""

    if structure_type == "PTV" or structure_type.startswith("PTV"):
        return (229, 57, 53)  # Đỏ
    elif structure_type == "OAR" or "OAR" in structure_type:
        return (30, 136, 229)  # Xanh dương
    else:
        return (141, 110, 99)  # Nâu


# Thông tin phiên bản
__version__ = "0.7.8"


# Apply enhanced Eclipse theme with better icons and visual hierarchy
def apply_eclipse_theme(widget):
    """Áp dụng theme Eclipse nâng cao với icons và visual hierarchy tốt hơn."""
    if not widget:
        return

    enhanced_style = """
    /* Main Window */
    QMainWindow {
        background-color: #2B2B2B;
        color: #CCCCCC;
        border: none;
    }

    /* Menu Bar */
    QMenuBar {
        background-color: #3C3C3C;
        color: #CCCCCC;
        border: none;
        padding: 2px;
    }

    QMenuBar::item {
        background-color: transparent;
        padding: 6px 12px;
        border-radius: 4px;
    }

    QMenuBar::item:selected {
        background-color: #4A90E2;
        color: white;
    }

    QMenuBar::item:pressed {
        background-color: #357ABD;
    }

    /* Menu */
    QMenu {
        background-color: #3C3C3C;
        color: #CCCCCC;
        border: 1px solid #555555;
        border-radius: 4px;
        padding: 4px;
    }

    QMenu::item {
        padding: 8px 25px 8px 20px;
        border-radius: 4px;
        margin: 1px;
    }

    QMenu::item:selected {
        background-color: #4A90E2;
        color: white;
    }

    QMenu::separator {
        height: 1px;
        background-color: #555555;
        margin: 4px 0px;
    }

    QMenu::indicator {
        width: 16px;
        height: 16px;
        margin-left: 5px;
    }

    /* Tool Bar */
    QToolBar {
        background-color: #3C3C3C;
        border: 1px solid #555555;
        border-radius: 4px;
        padding: 2px;
        spacing: 3px;
    }

    QToolBar::handle {
        background-color: #555555;
        width: 10px;
        margin: 4px 2px;
        border-radius: 2px;
    }

    QToolButton {
        background-color: transparent;
        border: 1px solid transparent;
        border-radius: 4px;
        padding: 6px;
        margin: 1px;
        min-width: 24px;
        min-height: 24px;
    }

    QToolButton:hover {
        background-color: #4A90E2;
        border-color: #4A90E2;
        color: white;
    }

    QToolButton:pressed {
        background-color: #357ABD;
        border-color: #357ABD;
    }

    QToolButton:checked {
        background-color: #4A90E2;
        border-color: #4A90E2;
        color: white;
    }

    /* Status Bar */
    QStatusBar {
        background-color: #3C3C3C;
        color: #CCCCCC;
        border-top: 1px solid #555555;
        padding: 4px;
    }

    QStatusBar::item {
        border: none;
        padding: 2px 8px;
    }

    /* Tab Widget */
    QTabWidget::pane {
        border: 1px solid #555555;
        background-color: #2B2B2B;
        border-radius: 4px;
    }

    QTabBar::tab {
        background-color: #3C3C3C;
        color: #CCCCCC;
        padding: 10px 20px;
        margin-right: 2px;
        border-top-left-radius: 6px;
        border-top-right-radius: 6px;
        border: 1px solid #555555;
        border-bottom: none;
        min-width: 100px;
    }

    QTabBar::tab:selected {
        background-color: #4A90E2;
        color: white;
        border-color: #4A90E2;
    }

    QTabBar::tab:hover:!selected {
        background-color: #4C4C4C;
    }

    QTabBar::close-button {
        image: url(icons/close.png);
        subcontrol-position: right;
    }

    QTabBar::close-button:hover {
        background-color: #FF6B6B;
        border-radius: 2px;
    }

    /* Splitter */
    QSplitter::handle {
        background-color: #555555;
        border: 1px solid #666666;
    }

    QSplitter::handle:horizontal {
        width: 6px;
        border-radius: 3px;
    }

    QSplitter::handle:vertical {
        height: 6px;
        border-radius: 3px;
    }

    QSplitter::handle:hover {
        background-color: #4A90E2;
    }

    /* Dock Widget */
    QDockWidget {
        background-color: #2B2B2B;
        color: #CCCCCC;
        border: 1px solid #555555;
        border-radius: 4px;
    }

    QDockWidget::title {
        background-color: #3C3C3C;
        color: #CCCCCC;
        padding: 8px;
        border-bottom: 1px solid #555555;
        font-weight: bold;
    }

    QDockWidget::close-button, QDockWidget::float-button {
        background-color: transparent;
        border: 1px solid transparent;
        border-radius: 3px;
        padding: 2px;
    }

    QDockWidget::close-button:hover, QDockWidget::float-button:hover {
        background-color: #4A90E2;
        border-color: #4A90E2;
    }

    /* Tree Widget */
    QTreeWidget, QTreeView {
        background-color: #2B2B2B;
        color: #CCCCCC;
        border: 1px solid #555555;
        border-radius: 4px;
        selection-background-color: #4A90E2;
        selection-color: white;
        outline: none;
    }

    QTreeWidget::item, QTreeView::item {
        padding: 4px;
        border: none;
        border-radius: 2px;
    }

    QTreeWidget::item:selected, QTreeView::item:selected {
        background-color: #4A90E2;
        color: white;
    }

    QTreeWidget::item:hover, QTreeView::item:hover {
        background-color: #4C4C4C;
    }

    QTreeWidget::branch:has-siblings:!adjoins-item {
        border-image: url(icons/vline.png) 0;
    }

    QTreeWidget::branch:has-siblings:adjoins-item {
        border-image: url(icons/branch-more.png) 0;
    }

    QTreeWidget::branch:!has-children:!has-siblings:adjoins-item {
        border-image: url(icons/branch-end.png) 0;
    }

    QTreeWidget::branch:has-children:!has-siblings:closed,
    QTreeWidget::branch:closed:has-children:has-siblings {
        border-image: none;
        image: url(icons/branch-closed.png);
    }

    QTreeWidget::branch:open:has-children:!has-siblings,
    QTreeWidget::branch:open:has-children:has-siblings {
        border-image: none;
        image: url(icons/branch-open.png);
    }

    /* List Widget */
    QListWidget {
        background-color: #2B2B2B;
        color: #CCCCCC;
        border: 1px solid #555555;
        border-radius: 4px;
        selection-background-color: #4A90E2;
        selection-color: white;
        outline: none;
    }

    QListWidget::item {
        padding: 6px;
        border: none;
        border-radius: 2px;
    }

    QListWidget::item:selected {
        background-color: #4A90E2;
        color: white;
    }

    QListWidget::item:hover {
        background-color: #4C4C4C;
    }

    /* Table Widget */
    QTableWidget, QTableView {
        background-color: #2B2B2B;
        color: #CCCCCC;
        border: 1px solid #555555;
        border-radius: 4px;
        selection-background-color: #4A90E2;
        selection-color: white;
        gridline-color: #555555;
        outline: none;
    }

    QHeaderView::section {
        background-color: #3C3C3C;
        color: #CCCCCC;
        padding: 8px;
        border: 1px solid #555555;
        border-radius: 0px;
        font-weight: bold;
    }

    QHeaderView::section:hover {
        background-color: #4A90E2;
        color: white;
    }

    /* Scroll Bar */
    QScrollBar:vertical {
        background-color: #3C3C3C;
        width: 14px;
        border-radius: 7px;
        margin: 0px;
    }

    QScrollBar::handle:vertical {
        background-color: #555555;
        border-radius: 7px;
        min-height: 20px;
        margin: 1px;
    }

    QScrollBar::handle:vertical:hover {
        background-color: #4A90E2;
    }

    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
        border: none;
        background: none;
    }

    QScrollBar:horizontal {
        background-color: #3C3C3C;
        height: 14px;
        border-radius: 7px;
        margin: 0px;
    }

    QScrollBar::handle:horizontal {
        background-color: #555555;
        border-radius: 7px;
        min-width: 20px;
        margin: 1px;
    }

    QScrollBar::handle:horizontal:hover {
        background-color: #4A90E2;
    }

    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
        border: none;
        background: none;
    }

    /* Progress Bar */
    QProgressBar {
        background-color: #3C3C3C;
        border: 1px solid #555555;
        border-radius: 6px;
        text-align: center;
        color: #CCCCCC;
        font-weight: bold;
    }

    QProgressBar::chunk {
        background-color: #4A90E2;
        border-radius: 5px;
        margin: 1px;
    }

    /* Group Box */
    QGroupBox {
        color: #CCCCCC;
        border: 2px solid #555555;
        border-radius: 6px;
        margin-top: 1ex;
        padding-top: 15px;
        font-weight: bold;
    }

    QGroupBox::title {
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 8px 0 8px;
        color: #4A90E2;
        font-weight: bold;
    }

    /* Input Widgets */
    QLineEdit, QTextEdit, QPlainTextEdit {
        background-color: #3C3C3C;
        color: #CCCCCC;
        border: 1px solid #555555;
        border-radius: 4px;
        padding: 6px;
        selection-background-color: #4A90E2;
        selection-color: white;
    }

    QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {
        border: 2px solid #4A90E2;
        background-color: #2B2B2B;
    }

    QLineEdit:hover, QTextEdit:hover, QPlainTextEdit:hover {
        border-color: #4A90E2;
    }

    /* Combo Box */
    QComboBox {
        background-color: #3C3C3C;
        color: #CCCCCC;
        border: 1px solid #555555;
        border-radius: 4px;
        padding: 6px 10px;
        min-width: 6em;
    }

    QComboBox:focus {
        border: 2px solid #4A90E2;
    }

    QComboBox:hover {
        border-color: #4A90E2;
    }

    QComboBox::drop-down {
        border: none;
        width: 20px;
    }

    QComboBox::down-arrow {
        image: url(icons/down-arrow.png);
        width: 12px;
        height: 12px;
    }

    QComboBox QAbstractItemView {
        background-color: #3C3C3C;
        color: #CCCCCC;
        border: 1px solid #555555;
        border-radius: 4px;
        selection-background-color: #4A90E2;
        selection-color: white;
        outline: none;
    }

    /* Spin Box */
    QSpinBox, QDoubleSpinBox {
        background-color: #3C3C3C;
        color: #CCCCCC;
        border: 1px solid #555555;
        border-radius: 4px;
        padding: 6px;
    }

    QSpinBox:focus, QDoubleSpinBox:focus {
        border: 2px solid #4A90E2;
    }

    QSpinBox::up-button, QDoubleSpinBox::up-button {
        background-color: #555555;
        border: none;
        border-radius: 2px;
        margin: 1px;
    }

    QSpinBox::down-button, QDoubleSpinBox::down-button {
        background-color: #555555;
        border: none;
        border-radius: 2px;
        margin: 1px;
    }

    QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
    QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
        background-color: #4A90E2;
    }

    /* Buttons */
    QPushButton {
        background-color: #3C3C3C;
        color: #CCCCCC;
        border: 1px solid #555555;
        border-radius: 4px;
        padding: 8px 16px;
        font-weight: bold;
        min-width: 80px;
    }

    QPushButton:hover {
        background-color: #4A90E2;
        border-color: #4A90E2;
        color: white;
    }

    QPushButton:pressed {
        background-color: #357ABD;
        border-color: #357ABD;
    }

    QPushButton:default {
        border: 2px solid #4A90E2;
        background-color: #4A90E2;
        color: white;
    }

    QPushButton:disabled {
        background-color: #2B2B2B;
        color: #666666;
        border-color: #444444;
    }

    /* Check Box */
    QCheckBox {
        color: #CCCCCC;
        spacing: 8px;
    }

    QCheckBox::indicator {
        width: 16px;
        height: 16px;
        border: 1px solid #555555;
        border-radius: 3px;
        background-color: #3C3C3C;
    }

    QCheckBox::indicator:hover {
        border-color: #4A90E2;
    }

    QCheckBox::indicator:checked {
        background-color: #4A90E2;
        border-color: #4A90E2;
        image: url(icons/check.png);
    }

    /* Radio Button */
    QRadioButton {
        color: #CCCCCC;
        spacing: 8px;
    }

    QRadioButton::indicator {
        width: 16px;
        height: 16px;
        border: 1px solid #555555;
        border-radius: 8px;
        background-color: #3C3C3C;
    }

    QRadioButton::indicator:hover {
        border-color: #4A90E2;
    }

    QRadioButton::indicator:checked {
        background-color: #4A90E2;
        border-color: #4A90E2;
        border-width: 4px;
        border-style: solid;
    }

    /* Slider */
    QSlider::groove:horizontal {
        background-color: #3C3C3C;
        height: 6px;
        border-radius: 3px;
    }

    QSlider::handle:horizontal {
        background-color: #4A90E2;
        border: 1px solid #357ABD;
        width: 16px;
        margin: -6px 0;
        border-radius: 8px;
    }

    QSlider::handle:horizontal:hover {
        background-color: #5BA0F2;
    }

    QSlider::sub-page:horizontal {
        background-color: #4A90E2;
        border-radius: 3px;
    }

    /* Date Edit */
    QDateEdit, QTimeEdit, QDateTimeEdit {
        background-color: #3C3C3C;
        color: #CCCCCC;
        border: 1px solid #555555;
        border-radius: 4px;
        padding: 6px;
    }

    QDateEdit:focus, QTimeEdit:focus, QDateTimeEdit:focus {
        border: 2px solid #4A90E2;
    }

    QDateEdit::drop-down, QTimeEdit::drop-down, QDateTimeEdit::drop-down {
        background-color: #555555;
        border: none;
        border-radius: 2px;
        margin: 1px;
    }

    QDateEdit::drop-down:hover, QTimeEdit::drop-down:hover, QDateTimeEdit::drop-down:hover {
        background-color: #4A90E2;
    }

    /* Calendar Widget */
    QCalendarWidget {
        background-color: #2B2B2B;
        color: #CCCCCC;
        border: 1px solid #555555;
        border-radius: 4px;
    }

    QCalendarWidget QAbstractItemView {
        background-color: #2B2B2B;
        color: #CCCCCC;
        selection-background-color: #4A90E2;
        selection-color: white;
    }

    /* Tool Tip */
    QToolTip {
        background-color: #3C3C3C;
        color: #CCCCCC;
        border: 1px solid #555555;
        border-radius: 4px;
        padding: 6px;
        font-size: 12px;
    }

    /* Frame */
    QFrame {
        border: 1px solid #555555;
        border-radius: 4px;
        background-color: #2B2B2B;
    }

    /* Label */
    QLabel {
        color: #CCCCCC;
        background-color: transparent;
    }

    QLabel[accessibleName="title"] {
        color: #4A90E2;
        font-weight: bold;
        font-size: 14px;
    }

    QLabel[accessibleName="subtitle"] {
        color: #AAAAAA;
        font-size: 12px;
    }
    """

    try:
        widget.setStyleSheet(enhanced_style)
        logger.info("Eclipse-style theme đã được áp dụng thành công.")
    except Exception as e:
        logger.error(f"Lỗi khi áp dụng Eclipse theme: {e}")


def create_eclipse_icon(icon_type: str, size: int = 16):
    """Tạo icons theo phong cách Eclipse."""
    from PyQt5.QtGui import QPixmap, QPainter, QPen, QBrush, QColor
    from PyQt5.QtCore import Qt

    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)

    # Eclipse blue color
    eclipse_blue = QColor("#4A90E2")

    if icon_type == "patient":
        # Draw patient icon (person)
        painter.setPen(QPen(eclipse_blue, 2))
        painter.setBrush(QBrush(eclipse_blue))
        # Head
        painter.drawEllipse(size // 4, 2, size // 2, size // 3)
        # Body
        painter.drawRect(size // 3, size // 2, size // 3, size // 2)

    elif icon_type == "plan":
        # Draw plan icon (document with lines)
        painter.setPen(QPen(eclipse_blue, 1))
        painter.setBrush(QBrush(eclipse_blue.lighter(150)))
        painter.drawRect(2, 2, size - 4, size - 4)
        painter.setPen(QPen(eclipse_blue, 1))
        for i in range(3):
            y = 4 + i * 3
            painter.drawLine(4, y, size - 4, y)

    elif icon_type == "structure":
        # Draw structure icon (polygon)
        painter.setPen(QPen(eclipse_blue, 2))
        painter.setBrush(QBrush(eclipse_blue.lighter(180)))

        # Tạo QPolygonF từ list các QPointF
        points = QPolygonF(
            [
                QPointF(size // 4, size // 4),
                QPointF(3 * size // 4, size // 4),
                QPointF(size - 2, 3 * size // 4),
                QPointF(size // 2, size - 2),
                QPointF(2, 3 * size // 4),
            ]
        )
        painter.drawPolygon(points)

    elif icon_type == "dose":
        # Draw dose icon (gradient circles)
        center = size // 2
        for i in range(3):
            radius = (i + 1) * size // 8
            alpha = 255 - i * 60
            color = QColor(eclipse_blue)
            color.setAlpha(alpha)
            painter.setBrush(QBrush(color))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(
                center - radius, center - radius, 2 * radius, 2 * radius
            )

    elif icon_type == "new":
        # Draw new/add icon (plus)
        painter.setPen(QPen(eclipse_blue, 3, Qt.SolidLine, Qt.RoundCap))
        # Horizontal line
        painter.drawLine(4, size // 2, size - 4, size // 2)
        # Vertical line
        painter.drawLine(size // 2, 4, size // 2, size - 4)

    elif icon_type == "open":
        # Draw open icon (folder)
        painter.setPen(QPen(eclipse_blue, 1))
        painter.setBrush(QBrush(eclipse_blue.lighter(150)))
        # Folder body
        painter.drawRect(2, size // 3, size - 4, 2 * size // 3 - 2)
        # Folder tab
        painter.drawRect(2, size // 4, size // 2, size // 6)

    elif icon_type == "save":
        # Draw save icon (disk)
        painter.setPen(QPen(eclipse_blue, 1))
        painter.setBrush(QBrush(eclipse_blue.lighter(150)))
        painter.drawRect(2, 2, size - 4, size - 4)
        # Disk slot
        painter.setBrush(QBrush(eclipse_blue))
        painter.drawRect(4, 3, size - 8, 2)

    elif icon_type == "calculate":
        # Draw calculate icon (calculator)
        painter.setPen(QPen(eclipse_blue, 1))
        painter.setBrush(QBrush(eclipse_blue.lighter(150)))
        painter.drawRect(3, 2, size - 6, size - 4)
        # Buttons
        painter.setBrush(QBrush(eclipse_blue))
        for i in range(2):
            for j in range(2):
                x = 5 + j * 3
                y = 6 + i * 3
                painter.drawRect(x, y, 2, 2)

    painter.end()
    return pixmap
