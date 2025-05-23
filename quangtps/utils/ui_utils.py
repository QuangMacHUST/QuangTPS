#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module tiện ích UI cho QuangTPS.

Module này cung cấp các hàm tiện ích giúp tạo và định dạng các thành phần UI
theo phong cách Eclipse.
"""

import logging
from typing import Optional, Union, Dict, Any, List, Tuple
import os

# Khởi tạo logger trước khi import PyQt5
logger = logging.getLogger(__name__)

# Kiểm tra PyQt5 availability
HAS_PYQT5 = False
try:
    from PyQt5.QtWidgets import QWidget, QApplication, QStyleFactory
    from PyQt5.QtCore import QSize, Qt, QRect
    from PyQt5.QtGui import (
        QIcon,
        QPixmap,
        QColor,
        QPalette,
        QPainter,
        QBrush,
        QPen,
        QFont,
    )

    HAS_PYQT5 = True
    logger.info("PyQt5 được tải thành công")
except ImportError as e:
    logger.warning(f"PyQt5 không khả dụng: {e}")

    # Tạo fallback classes
    class QWidget:
        def __init__(self, *args, **kwargs):
            pass

        def setStyleSheet(self, *args, **kwargs):
            pass

        def style(self):
            return None

    class Qt:
        transparent = 0
        white = 1
        black = 2

    class QIcon:
        def __init__(self, *args, **kwargs):
            pass

    class QPixmap:
        def __init__(self, *args, **kwargs):
            pass

        def fill(self, *args, **kwargs):
            pass

    class QPainter:
        def __init__(self, *args, **kwargs):
            pass

        def setRenderHint(self, *args, **kwargs):
            pass

        def setPen(self, *args, **kwargs):
            pass

        def setBrush(self, *args, **kwargs):
            pass

        def drawEllipse(self, *args, **kwargs):
            pass

        def drawRect(self, *args, **kwargs):
            pass

        def drawText(self, *args, **kwargs):
            pass

        def end(self):
            pass

        Antialiasing = 1

    class QColor:
        def __init__(self, *args, **kwargs):
            pass

    class QPen:
        def __init__(self, *args, **kwargs):
            pass

    class QFont:
        def __init__(self, *args, **kwargs):
            pass

        def setPointSize(self, *args, **kwargs):
            pass

        def setBold(self, *args, **kwargs):
            pass

    class QBrush:
        def __init__(self, *args, **kwargs):
            pass


def create_eclipse_icon(
    icon_name: str, custom_render: bool = True, size: int = 32, colors: Dict = None
) -> Optional["QIcon"]:
    """
    Tạo icon theo phong cách Eclipse với khả năng render tùy chỉnh.

    Parameters
    ----------
    icon_name : str
        Tên icon cần tạo
    custom_render : bool
        Có sử dụng custom rendering hay không
    size : int
        Kích thước icon (pixels)
    colors : Dict
        Dictionary màu sắc tùy chỉnh

    Returns
    -------
    QIcon hoặc None
        Icon được tạo hoặc None nếu không thể tạo
    """
    if not HAS_PYQT5:
        logger.warning("PyQt5 không khả dụng, không thể tạo icon")
        return None

    try:
        # Default colors với Eclipse theme
        default_colors = {
            "primary": "#4A90E2",
            "secondary": "#CCCCCC",
            "background": "#2B2B2B",
            "accent": "#F5A623",
            "danger": "#D0021B",
        }

        if colors:
            default_colors.update(colors)

        # Tạo pixmap trong suốt
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.transparent)

        if not custom_render:
            return QIcon(pixmap)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)

        # Icon-specific rendering
        if icon_name == "plan":
            # Icon kế hoạch xạ trị - target with crosshairs
            painter.setPen(QPen(QColor(default_colors["primary"]), 2))
            center = size // 2
            radius = (size - 6) // 2

            # Vẽ mục tiêu (target)
            painter.drawEllipse(
                center - radius, center - radius, radius * 2, radius * 2
            )
            painter.drawEllipse(
                center - radius // 2, center - radius // 2, radius, radius
            )

            # Vẽ crosshairs
            painter.drawLine(3, center, size - 3, center)
            painter.drawLine(center, 3, center, size - 3)

        elif icon_name == "dose":
            # Icon liều - gradient circles
            colors = [
                default_colors["danger"],
                default_colors["accent"],
                default_colors["primary"],
            ]
            for i, color in enumerate(colors):
                radius = (size - 4) // 2 - i * 4
                if radius > 0:
                    painter.setPen(QPen(QColor(color), 2))
                    center = size // 2
                    painter.drawEllipse(
                        center - radius, center - radius, radius * 2, radius * 2
                    )

        elif icon_name == "structure":
            # Icon cấu trúc - overlapping polygons
            painter.setPen(QPen(QColor(default_colors["secondary"]), 2))
            painter.setBrush(QBrush(QColor(default_colors["primary"])))

            # Vẽ polygon đơn giản
            points = [
                (size // 4, size // 4),
                (3 * size // 4, size // 4),
                (3 * size // 4, 3 * size // 4),
                (size // 4, 3 * size // 4),
            ]
            painter.drawRect(size // 4, size // 4, size // 2, size // 2)

        elif icon_name == "beam":
            # Icon chùm tia - rays from center
            painter.setPen(QPen(QColor(default_colors["accent"]), 2))
            center = size // 2

            # Vẽ các tia từ trung tâm
            for angle in range(0, 360, 45):
                import math

                rad = math.radians(angle)
                x1 = center + int(4 * math.cos(rad))
                y1 = center + int(4 * math.sin(rad))
                x2 = center + int((size // 2 - 4) * math.cos(rad))
                y2 = center + int((size // 2 - 4) * math.sin(rad))
                painter.drawLine(x1, y1, x2, y2)

        elif icon_name == "optimization":
            # Icon tối ưu hóa - ascending bars
            painter.setPen(QPen(QColor(default_colors["primary"]), 1))
            painter.setBrush(QBrush(QColor(default_colors["primary"])))

            # Vẽ các thanh tăng dần
            bar_width = size // 6
            for i in range(5):
                x = 2 + i * (bar_width + 1)
                height = (i + 1) * (size - 4) // 5
                y = size - 2 - height
                painter.drawRect(x, y, bar_width, height)

        elif icon_name == "evaluation":
            # Icon đánh giá - check mark
            painter.setPen(QPen(QColor(default_colors["primary"]), 3))

            # Vẽ dấu check
            check_points = [
                (size // 4, size // 2),
                (size // 2, 3 * size // 4),
                (3 * size // 4, size // 4),
            ]
            painter.drawLine(
                check_points[0][0],
                check_points[0][1],
                check_points[1][0],
                check_points[1][1],
            )
            painter.drawLine(
                check_points[1][0],
                check_points[1][1],
                check_points[2][0],
                check_points[2][1],
            )

        elif icon_name == "patient":
            # Icon bệnh nhân - simplified person
            painter.setPen(QPen(QColor(default_colors["secondary"]), 2))
            painter.setBrush(QBrush(QColor(default_colors["secondary"])))

            # Vẽ đầu
            head_radius = size // 6
            center = size // 2
            painter.drawEllipse(
                center - head_radius,
                size // 4 - head_radius,
                head_radius * 2,
                head_radius * 2,
            )

            # Vẽ thân
            painter.drawRect(center - size // 8, size // 2, size // 4, size // 3)

        elif icon_name == "machine":
            # Icon máy xạ trị - simplified linac
            painter.setPen(QPen(QColor(default_colors["secondary"]), 2))

            # Gantry (oval)
            painter.drawEllipse(4, 4, size - 8, size - 8)

            # Head
            painter.setPen(QPen(QColor(default_colors["primary"]), 3))
            center = size // 2
            painter.drawRect(center - 3, 6, 6, size // 3)

        else:
            # Default icon - simple circle
            painter.setPen(QPen(QColor(default_colors["primary"]), 2))
            painter.drawEllipse(4, 4, size - 8, size - 8)

        painter.end()
        return QIcon(pixmap)

    except Exception as e:
        logger.error(f"Lỗi tạo icon {icon_name}: {e}")
        return None


def apply_eclipse_theme(widget: "QWidget") -> None:
    """
    Áp dụng Eclipse theme cho widget (deprecated - sử dụng apply_eclipse_style_theme).

    Parameters
    ----------
    widget : QWidget
        Widget cần áp dụng theme
    """
    logger.warning("apply_eclipse_theme deprecated, sử dụng apply_eclipse_style_theme")
    apply_eclipse_style_theme(widget)


def apply_eclipse_style_theme(widget):
    """
    Áp dụng theme Eclipse cho widget.

    Parameters
    ----------
    widget : QWidget
        Widget cần áp dụng theme
    """
    if not HAS_PYQT5:
        logger.warning("PyQt5 không khả dụng, không thể áp dụng theme")
        return

    try:
        if not isinstance(widget, QWidget):
            logger.warning(
                f"Widget {widget} không phải QWidget, không thể áp dụng theme"
            )
            return

        # Eclipse dark theme stylesheet
        eclipse_style = """
        QWidget {
            background-color: #2B2B2B;
            color: #CCCCCC;
            font-family: 'Segoe UI', Arial, sans-serif;
            font-size: 9pt;
        }

        QTabWidget::pane {
            border: 1px solid #555555;
            background-color: #2B2B2B;
        }

        QTabBar::tab {
            background-color: #3C3C3C;
            color: #CCCCCC;
            padding: 8px 16px;
            margin-right: 2px;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
        }

        QTabBar::tab:selected {
            background-color: #4A90E2;
            color: white;
        }

        QTabBar::tab:hover {
            background-color: #484848;
        }

        QPushButton {
            background-color: #3C3C3C;
            border: 1px solid #555555;
            color: #CCCCCC;
            padding: 6px 12px;
            border-radius: 3px;
            min-height: 18px;
        }

        QPushButton:hover {
            background-color: #4A90E2;
            border-color: #4A90E2;
        }

        QPushButton:pressed {
            background-color: #357ABD;
        }

        QPushButton:disabled {
            background-color: #2B2B2B;
            color: #777777;
            border-color: #444444;
        }

        QComboBox {
            background-color: #3C3C3C;
            border: 1px solid #555555;
            color: #CCCCCC;
            padding: 4px 8px;
            border-radius: 3px;
            min-height: 18px;
        }

        QComboBox:hover {
            border-color: #4A90E2;
        }

        QComboBox::drop-down {
            border: none;
            width: 20px;
        }

        QComboBox::down-arrow {
            image: none;
            border-left: 4px solid transparent;
            border-right: 4px solid transparent;
            border-top: 6px solid #CCCCCC;
            margin-right: 6px;
        }

        QListWidget {
            background-color: #2B2B2B;
            border: 1px solid #555555;
            color: #CCCCCC;
            selection-background-color: #4A90E2;
        }

        QListWidget::item {
            padding: 4px;
            border-bottom: 1px solid #404040;
        }

        QListWidget::item:selected {
            background-color: #4A90E2;
        }

        QListWidget::item:hover {
            background-color: #484848;
        }

        QTableWidget {
            background-color: #2B2B2B;
            border: 1px solid #555555;
            color: #CCCCCC;
            selection-background-color: #4A90E2;
            gridline-color: #404040;
        }

        QTableWidget::item {
            padding: 4px;
            border-bottom: 1px solid #404040;
        }

        QTableWidget::item:selected {
            background-color: #4A90E2;
        }

        QHeaderView::section {
            background-color: #3C3C3C;
            color: #CCCCCC;
            padding: 6px;
            border: 1px solid #555555;
            font-weight: bold;
        }

        QLineEdit {
            background-color: #3C3C3C;
            border: 1px solid #555555;
            color: #CCCCCC;
            padding: 4px 8px;
            border-radius: 3px;
            selection-background-color: #4A90E2;
        }

        QLineEdit:focus {
            border-color: #4A90E2;
        }

        QTextEdit {
            background-color: #2B2B2B;
            border: 1px solid #555555;
            color: #CCCCCC;
            selection-background-color: #4A90E2;
        }

        QScrollBar:vertical {
            background-color: #2B2B2B;
            width: 16px;
            border: none;
        }

        QScrollBar::handle:vertical {
            background-color: #555555;
            border-radius: 8px;
            min-height: 20px;
            margin: 2px;
        }

        QScrollBar::handle:vertical:hover {
            background-color: #666666;
        }

        QScrollBar::add-line:vertical,
        QScrollBar::sub-line:vertical {
            height: 0px;
        }

        QScrollBar:horizontal {
            background-color: #2B2B2B;
            height: 16px;
            border: none;
        }

        QScrollBar::handle:horizontal {
            background-color: #555555;
            border-radius: 8px;
            min-width: 20px;
            margin: 2px;
        }

        QScrollBar::handle:horizontal:hover {
            background-color: #666666;
        }

        QScrollBar::add-line:horizontal,
        QScrollBar::sub-line:horizontal {
            width: 0px;
        }

        QProgressBar {
            background-color: #2B2B2B;
            border: 1px solid #555555;
            border-radius: 3px;
            text-align: center;
            color: #CCCCCC;
        }

        QProgressBar::chunk {
            background-color: #4A90E2;
            border-radius: 2px;
        }

        QSpinBox, QDoubleSpinBox {
            background-color: #3C3C3C;
            border: 1px solid #555555;
            color: #CCCCCC;
            padding: 4px 8px;
            border-radius: 3px;
        }

        QSpinBox:focus, QDoubleSpinBox:focus {
            border-color: #4A90E2;
        }

        QSpinBox::up-button, QDoubleSpinBox::up-button {
            background-color: #3C3C3C;
            border: 1px solid #555555;
            border-bottom: none;
        }

        QSpinBox::down-button, QDoubleSpinBox::down-button {
            background-color: #3C3C3C;
            border: 1px solid #555555;
            border-top: none;
        }

        QCheckBox {
            color: #CCCCCC;
            spacing: 8px;
        }

        QCheckBox::indicator {
            width: 16px;
            height: 16px;
            background-color: #3C3C3C;
            border: 1px solid #555555;
            border-radius: 3px;
        }

        QCheckBox::indicator:checked {
            background-color: #4A90E2;
            border-color: #4A90E2;
        }

        QRadioButton {
            color: #CCCCCC;
            spacing: 8px;
        }

        QRadioButton::indicator {
            width: 16px;
            height: 16px;
            background-color: #3C3C3C;
            border: 1px solid #555555;
            border-radius: 8px;
        }

        QRadioButton::indicator:checked {
            background-color: #4A90E2;
            border-color: #4A90E2;
        }

        QMenuBar {
            background-color: #2B2B2B;
            color: #CCCCCC;
            border-bottom: 1px solid #555555;
        }

        QMenuBar::item {
            background-color: transparent;
            padding: 6px 12px;
        }

        QMenuBar::item:selected {
            background-color: #4A90E2;
        }

        QMenu {
            background-color: #3C3C3C;
            color: #CCCCCC;
            border: 1px solid #555555;
        }

        QMenu::item {
            padding: 6px 20px;
        }

        QMenu::item:selected {
            background-color: #4A90E2;
        }

        QToolBar {
            background-color: #2B2B2B;
            border: 1px solid #555555;
            spacing: 2px;
        }

        QToolButton {
            background-color: transparent;
            border: none;
            color: #CCCCCC;
            padding: 4px;
            border-radius: 3px;
        }

        QToolButton:hover {
            background-color: #4A90E2;
        }

        QToolButton:pressed {
            background-color: #357ABD;
        }

        QStatusBar {
            background-color: #2B2B2B;
            color: #CCCCCC;
            border-top: 1px solid #555555;
        }

        QSplitter::handle {
            background-color: #555555;
        }

        QSplitter::handle:horizontal {
            width: 3px;
        }

        QSplitter::handle:vertical {
            height: 3px;
        }

        QGroupBox {
            color: #CCCCCC;
            border: 1px solid #555555;
            border-radius: 5px;
            margin-top: 10px;
            font-weight: bold;
        }

        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px 0 5px;
        }
        """

        widget.setStyleSheet(eclipse_style)
        logger.debug(f"Áp dụng Eclipse theme cho {type(widget).__name__}")

    except Exception as e:
        logger.error(f"Lỗi áp dụng Eclipse theme: {e}")


def create_eclipse_themed_widget(widget_class, *args, **kwargs):
    """
    Tạo widget với Eclipse theme được áp dụng sẵn.

    Parameters
    ----------
    widget_class : class
        Class của widget cần tạo
    *args, **kwargs
        Arguments cho constructor của widget

    Returns
    -------
    Widget
        Widget đã được tạo và áp dụng theme, hoặc fallback widget
    """
    if not HAS_PYQT5:
        logger.warning("PyQt5 không khả dụng, trả về fallback widget")
        return FallbackWidget()

    try:
        widget = widget_class(*args, **kwargs)
        apply_eclipse_style_theme(widget)
        return widget
    except Exception as e:
        logger.error(f"Lỗi tạo widget {widget_class.__name__}: {e}")
        return FallbackWidget()


class FallbackWidget:
    """
    Widget dự phòng khi PyQt5 không khả dụng.
    """

    def __init__(self, *args, **kwargs):
        self.visible = True
        self.enabled = True

    def show(self):
        self.visible = True

    def hide(self):
        self.visible = False

    def setEnabled(self, enabled):
        self.enabled = enabled

    def setStyleSheet(self, style):
        pass

    def __getattr__(self, name):
        # Trả về function giả cho bất kỳ method nào
        def dummy_method(*args, **kwargs):
            logger.debug(f"FallbackWidget dummy method called: {name}")
            return None

        return dummy_method
