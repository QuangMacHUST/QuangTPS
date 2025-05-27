#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Eclipse Theme Manager for QuangTPS

Quản lý themes theo phong cách Eclipse TPS với dark/light mode
và customization options tương tự Eclipse IDE.
"""

import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

try:
    from PyQt5.QtWidgets import QWidget, QApplication
    from PyQt5.QtCore import QSettings, pyqtSignal, QObject
    from PyQt5.QtGui import QColor, QPalette, QFont

    _PYQT_AVAILABLE = True
except ImportError:
    _PYQT_AVAILABLE = False
    logging.warning("PyQt5 không khả dụng. Theme manager sẽ không hoạt động.")

logger = logging.getLogger(__name__)


class ThemeType(Enum):
    """Các loại theme có sẵn"""

    DARK_ECLIPSE = "dark_eclipse"
    LIGHT_ECLIPSE = "light_eclipse"
    VARIAN_ECLIPSE = "varian_eclipse"
    HIGH_CONTRAST = "high_contrast"
    CUSTOM = "custom"


@dataclass
class ColorScheme:
    """Định nghĩa color scheme cho theme"""

    # Background colors
    background_primary: str
    background_secondary: str
    background_tertiary: str

    # Text colors
    text_primary: str
    text_secondary: str
    text_disabled: str

    # Accent colors
    accent_primary: str
    accent_secondary: str

    # Status colors
    success: str
    warning: str
    error: str
    info: str

    # Border colors
    border_primary: str
    border_secondary: str

    # Special colors
    selection: str
    hover: str
    focus: str

    # Medical imaging colors
    dose_high: str
    dose_medium: str
    dose_low: str
    structure_ptv: str
    structure_oar: str


class EclipseThemeManager(QObject):
    """Manager cho Eclipse-style themes"""

    theme_changed = pyqtSignal(str)  # Emit khi theme thay đổi

    def __init__(self):
        super().__init__()
        self.settings = QSettings("QuangTPS", "ThemeManager")
        self.current_theme = ThemeType.DARK_ECLIPSE
        self.custom_colors = {}

        # Initialize predefined themes
        self.themes = self._initialize_themes()

        # Load saved theme
        self._load_saved_theme()

    def _initialize_themes(self) -> Dict[ThemeType, ColorScheme]:
        """Khởi tạo các theme có sẵn"""
        themes = {}

        # Dark Eclipse Theme (default)
        themes[ThemeType.DARK_ECLIPSE] = ColorScheme(
            background_primary="#2B2B2B",
            background_secondary="#3C3C3C",
            background_tertiary="#4A4A4A",
            text_primary="#CCCCCC",
            text_secondary="#AAAAAA",
            text_disabled="#666666",
            accent_primary="#4A90E2",
            accent_secondary="#357ABD",
            success="#7ED321",
            warning="#F5A623",
            error="#D0021B",
            info="#4A90E2",
            border_primary="#555555",
            border_secondary="#444444",
            selection="#4A90E2",
            hover="#5BA0F2",
            focus="#6BB0FF",
            dose_high="#FF0000",
            dose_medium="#FFFF00",
            dose_low="#0000FF",
            structure_ptv="#FF6B6B",
            structure_oar="#4ECDC4",
        )

        # Light Eclipse Theme
        themes[ThemeType.LIGHT_ECLIPSE] = ColorScheme(
            background_primary="#F5F5F5",
            background_secondary="#FFFFFF",
            background_tertiary="#E8E8E8",
            text_primary="#333333",
            text_secondary="#666666",
            text_disabled="#AAAAAA",
            accent_primary="#0E5A8A",
            accent_secondary="#1A6EAF",
            success="#2E7D0E",
            warning="#A0530E",
            error="#C5000F",
            info="#0E5A8A",
            border_primary="#CCCCCC",
            border_secondary="#DDDDDD",
            selection="#0E5A8A",
            hover="#1A6EAF",
            focus="#2E7EC9",
            dose_high="#CC0000",
            dose_medium="#CCCC00",
            dose_low="#0000CC",
            structure_ptv="#E85A5A",
            structure_oar="#42A5A1",
        )

        # Varian Eclipse Style
        themes[ThemeType.VARIAN_ECLIPSE] = ColorScheme(
            background_primary="#1E1E1E",
            background_secondary="#2D2D30",
            background_tertiary="#37373D",
            text_primary="#F1F1F1",
            text_secondary="#CCCCCC",
            text_disabled="#808080",
            accent_primary="#007ACC",
            accent_secondary="#1177BB",
            success="#4EC9B0",
            warning="#FFD700",
            error="#F44747",
            info="#569CD6",
            border_primary="#464647",
            border_secondary="#3E3E42",
            selection="#264F78",
            hover="#2A5A8A",
            focus="#3B6AA0",
            dose_high="#FF4444",
            dose_medium="#FFFF44",
            dose_low="#4444FF",
            structure_ptv="#FF7B7B",
            structure_oar="#5EDCD4",
        )

        # High Contrast Theme (for accessibility)
        themes[ThemeType.HIGH_CONTRAST] = ColorScheme(
            background_primary="#000000",
            background_secondary="#111111",
            background_tertiary="#222222",
            text_primary="#FFFFFF",
            text_secondary="#DDDDDD",
            text_disabled="#888888",
            accent_primary="#00FFFF",
            accent_secondary="#00CCCC",
            success="#00FF00",
            warning="#FFFF00",
            error="#FF0000",
            info="#00FFFF",
            border_primary="#FFFFFF",
            border_secondary="#CCCCCC",
            selection="#00FFFF",
            hover="#44FFFF",
            focus="#88FFFF",
            dose_high="#FF0000",
            dose_medium="#FFFF00",
            dose_low="#0000FF",
            structure_ptv="#FF00FF",
            structure_oar="#00FFFF",
        )

        return themes

    def apply_theme(self, theme_type: ThemeType, widget: Optional[QWidget] = None):
        """Áp dụng theme cho widget hoặc toàn bộ application"""
        try:
            if theme_type not in self.themes:
                logger.error(f"Theme {theme_type} không tồn tại")
                return False

            color_scheme = self.themes[theme_type]
            stylesheet = self._generate_stylesheet(color_scheme)

            if widget:
                widget.setStyleSheet(stylesheet)
            else:
                # Áp dụng cho toàn bộ application
                app = QApplication.instance()
                if app:
                    app.setStyleSheet(stylesheet)

            self.current_theme = theme_type
            self._save_current_theme()
            self.theme_changed.emit(theme_type.value)

            logger.info(f"Đã áp dụng theme: {theme_type.value}")
            return True

        except Exception as e:
            logger.error(f"Lỗi khi áp dụng theme: {e}")
            return False

    def _generate_stylesheet(self, colors: ColorScheme) -> str:
        """Tạo stylesheet từ color scheme"""
        return f"""
        /* Main Application */
        QMainWindow {{
            background-color: {colors.background_primary};
            color: {colors.text_primary};
        }}

        /* Widgets */
        QWidget {{
            background-color: {colors.background_primary};
            color: {colors.text_primary};
            selection-background-color: {colors.selection};
        }}

        /* Dock Widgets */
        QDockWidget {{
            background-color: {colors.background_secondary};
            border: 1px solid {colors.border_primary};
            titlebar-close-icon: none;
            titlebar-normal-icon: none;
        }}

        QDockWidget::title {{
            background-color: {colors.background_secondary};
            color: {colors.text_primary};
            padding: 6px;
            border: none;
            font-weight: bold;
        }}

        QDockWidget::title:hover {{
            background-color: {colors.hover};
        }}

        /* Tabs */
        QTabWidget::pane {{
            border: 1px solid {colors.border_primary};
            background-color: {colors.background_primary};
        }}

        QTabBar::tab {{
            background-color: {colors.background_secondary};
            color: {colors.text_secondary};
            padding: 8px 16px;
            margin-right: 2px;
            border-top-left-radius: 3px;
            border-top-right-radius: 3px;
        }}

        QTabBar::tab:selected {{
            background-color: {colors.accent_primary};
            color: white;
            font-weight: bold;
        }}

        QTabBar::tab:hover:!selected {{
            background-color: {colors.hover};
            color: {colors.text_primary};
        }}

        /* Menu Bar */
        QMenuBar {{
            background-color: {colors.background_secondary};
            color: {colors.text_primary};
            border: none;
            padding: 2px;
        }}

        QMenuBar::item {{
            background-color: transparent;
            padding: 4px 8px;
        }}

        QMenuBar::item:selected {{
            background-color: {colors.accent_primary};
            color: white;
        }}

        QMenu {{
            background-color: {colors.background_secondary};
            color: {colors.text_primary};
            border: 1px solid {colors.border_primary};
        }}

        QMenu::item:selected {{
            background-color: {colors.accent_primary};
        }}

        /* Tool Bar */
        QToolBar {{
            background-color: {colors.background_secondary};
            border: 1px solid {colors.border_secondary};
            spacing: 2px;
            padding: 2px;
        }}

        QToolButton {{
            background-color: transparent;
            border: 1px solid transparent;
            padding: 4px;
            margin: 1px;
        }}

        QToolButton:hover {{
            background-color: {colors.hover};
            border: 1px solid {colors.border_primary};
        }}

        QToolButton:pressed {{
            background-color: {colors.accent_primary};
            color: white;
        }}

        /* Buttons */
        QPushButton {{
            background-color: {colors.background_secondary};
            border: 1px solid {colors.border_primary};
            color: {colors.text_primary};
            padding: 6px 16px;
            min-width: 60px;
            border-radius: 3px;
        }}

        QPushButton:hover {{
            background-color: {colors.hover};
            border-color: {colors.accent_primary};
        }}

        QPushButton:pressed {{
            background-color: {colors.accent_primary};
            color: white;
        }}

        QPushButton:disabled {{
            background-color: {colors.background_tertiary};
            color: {colors.text_disabled};
            border-color: {colors.border_secondary};
        }}

        /* Input Fields */
        QLineEdit, QTextEdit, QPlainTextEdit {{
            background-color: {colors.background_primary};
            border: 1px solid {colors.border_primary};
            color: {colors.text_primary};
            padding: 4px;
            border-radius: 2px;
        }}

        QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {{
            border-color: {colors.focus};
            background-color: {colors.background_secondary};
        }}

        /* Combo Box */
        QComboBox {{
            background-color: {colors.background_secondary};
            border: 1px solid {colors.border_primary};
            color: {colors.text_primary};
            padding: 4px 8px;
            min-width: 100px;
        }}

        QComboBox:hover {{
            border-color: {colors.hover};
        }}

        QComboBox::drop-down {{
            border: none;
            background-color: {colors.background_tertiary};
        }}

        /* List Widget */
        QListWidget {{
            background-color: {colors.background_primary};
            border: 1px solid {colors.border_primary};
            color: {colors.text_primary};
            alternate-background-color: {colors.background_secondary};
        }}

        QListWidget::item:selected {{
            background-color: {colors.selection};
            color: white;
        }}

        QListWidget::item:hover {{
            background-color: {colors.hover};
        }}

        /* Tree Widget */
        QTreeWidget {{
            background-color: {colors.background_primary};
            border: 1px solid {colors.border_primary};
            color: {colors.text_primary};
            alternate-background-color: {colors.background_secondary};
        }}

        QTreeWidget::item:selected {{
            background-color: {colors.selection};
            color: white;
        }}

        QTreeWidget::item:hover {{
            background-color: {colors.hover};
        }}

        /* Status Bar */
        QStatusBar {{
            background-color: {colors.background_secondary};
            color: {colors.text_primary};
            border-top: 1px solid {colors.border_primary};
            padding: 2px;
        }}

        /* Progress Bar */
        QProgressBar {{
            background-color: {colors.background_tertiary};
            border: 1px solid {colors.border_primary};
            text-align: center;
            color: {colors.text_primary};
            border-radius: 2px;
        }}

        QProgressBar::chunk {{
            background-color: {colors.accent_primary};
            border-radius: 2px;
        }}

        /* Splitter */
        QSplitter::handle {{
            background-color: {colors.border_primary};
        }}

        QSplitter::handle:hover {{
            background-color: {colors.accent_primary};
        }}

        /* Scroll Bar */
        QScrollBar:vertical {{
            background-color: {colors.background_tertiary};
            width: 12px;
            border: none;
        }}

        QScrollBar::handle:vertical {{
            background-color: {colors.border_primary};
            border-radius: 6px;
            min-height: 20px;
        }}

        QScrollBar::handle:vertical:hover {{
            background-color: {colors.accent_primary};
        }}

        /* Group Box */
        QGroupBox {{
            font-weight: bold;
            border: 1px solid {colors.border_primary};
            border-radius: 3px;
            margin-top: 6px;
            padding-top: 6px;
        }}

        QGroupBox::title {{
            subcontrol-origin: margin;
            left: 8px;
            padding: 0 4px 0 4px;
            color: {colors.text_primary};
        }}

        /* Specific Medical UI Elements */
        .dose-high {{
            background-color: {colors.dose_high};
            color: white;
        }}

        .dose-medium {{
            background-color: {colors.dose_medium};
            color: black;
        }}

        .dose-low {{
            background-color: {colors.dose_low};
            color: white;
        }}

        .structure-ptv {{
            color: {colors.structure_ptv};
            font-weight: bold;
        }}

        .structure-oar {{
            color: {colors.structure_oar};
        }}

        /* Status indicators */
        .status-success {{
            color: {colors.success};
            font-weight: bold;
        }}

        .status-warning {{
            color: {colors.warning};
            font-weight: bold;
        }}

        .status-error {{
            color: {colors.error};
            font-weight: bold;
        }}

        .status-info {{
            color: {colors.info};
        }}
        """

    def get_color(self, color_name: str, theme_type: Optional[ThemeType] = None) -> str:
        """Lấy màu từ theme hiện tại"""
        if theme_type is None:
            theme_type = self.current_theme

        if theme_type in self.themes:
            color_scheme = self.themes[theme_type]
            return getattr(color_scheme, color_name, "#FFFFFF")

        return "#FFFFFF"

    def customize_color(self, color_name: str, color_value: str):
        """Tùy chỉnh màu sắc"""
        self.custom_colors[color_name] = color_value
        self._save_custom_colors()

    def get_available_themes(self) -> Dict[ThemeType, str]:
        """Lấy danh sách themes có sẵn"""
        return {
            ThemeType.DARK_ECLIPSE: "Dark Eclipse (Default)",
            ThemeType.LIGHT_ECLIPSE: "Light Eclipse",
            ThemeType.VARIAN_ECLIPSE: "Varian Eclipse Style",
            ThemeType.HIGH_CONTRAST: "High Contrast",
            ThemeType.CUSTOM: "Custom Theme",
        }

    def _save_current_theme(self):
        """Lưu theme hiện tại"""
        try:
            self.settings.setValue("current_theme", self.current_theme.value)
        except Exception as e:
            logger.error(f"Lỗi khi lưu theme: {e}")

    def _load_saved_theme(self):
        """Load theme đã lưu"""
        try:
            saved_theme = self.settings.value(
                "current_theme", ThemeType.DARK_ECLIPSE.value
            )
            for theme_type in ThemeType:
                if theme_type.value == saved_theme:
                    self.current_theme = theme_type
                    break
        except Exception as e:
            logger.error(f"Lỗi khi load theme: {e}")

    def _save_custom_colors(self):
        """Lưu custom colors"""
        try:
            for color_name, color_value in self.custom_colors.items():
                self.settings.setValue(f"custom_color_{color_name}", color_value)
        except Exception as e:
            logger.error(f"Lỗi khi lưu custom colors: {e}")

    def apply_medical_coloring(self, widget: QWidget, element_type: str):
        """Áp dụng màu sắc chuyên biệt cho các element y tế"""
        try:
            colors = self.themes[self.current_theme]

            if element_type == "dose_high":
                widget.setStyleSheet(
                    f"background-color: {colors.dose_high}; color: white;"
                )
            elif element_type == "dose_medium":
                widget.setStyleSheet(
                    f"background-color: {colors.dose_medium}; color: black;"
                )
            elif element_type == "dose_low":
                widget.setStyleSheet(
                    f"background-color: {colors.dose_low}; color: white;"
                )
            elif element_type == "structure_ptv":
                widget.setStyleSheet(
                    f"color: {colors.structure_ptv}; font-weight: bold;"
                )
            elif element_type == "structure_oar":
                widget.setStyleSheet(f"color: {colors.structure_oar};")

        except Exception as e:
            logger.error(f"Lỗi khi áp dụng medical coloring: {e}")


# Global theme manager instance
_theme_manager = None


def get_theme_manager() -> EclipseThemeManager:
    """Lấy theme manager instance (singleton)"""
    global _theme_manager
    if _theme_manager is None:
        _theme_manager = EclipseThemeManager()
    return _theme_manager


def apply_eclipse_theme(
    theme_type: ThemeType = ThemeType.DARK_ECLIPSE, widget: Optional[QWidget] = None
):
    """Convenience function để áp dụng Eclipse theme"""
    if not _PYQT_AVAILABLE:
        logger.warning("PyQt5 không khả dụng. Không thể áp dụng theme.")
        return False

    theme_manager = get_theme_manager()
    return theme_manager.apply_theme(theme_type, widget)


def get_theme_color(color_name: str, theme_type: Optional[ThemeType] = None) -> str:
    """Convenience function để lấy màu từ theme"""
    theme_manager = get_theme_manager()
    return theme_manager.get_color(color_name, theme_type)
