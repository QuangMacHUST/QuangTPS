"""
Module quản lý theme Eclipse-style cho QuangTPS UI.

Provides comprehensive Eclipse-style theming including:
- Dark/Light theme switching
- Professional color schemes
- Icon management
- Widget styling
- Layout optimization
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from pathlib import Path

# PyQt5 imports with fallbacks
try:
    from PyQt5.QtWidgets import QApplication, QWidget
    from PyQt5.QtCore import QSettings, pyqtSignal, QObject
    from PyQt5.QtGui import QColor, QPalette, QFont, QIcon, QPixmap

    HAS_PYQT5 = True
except ImportError:
    HAS_PYQT5 = False

    # Fallback classes
    class QObject:
        pass

    class QColor:
        def __init__(self, *args):
            pass

    class QPalette:
        pass


logger = logging.getLogger(__name__)


@dataclass
class EclipseColors:
    """Eclipse color scheme definition."""

    # Background colors
    background_primary: str = "#2B2B2B"
    background_secondary: str = "#3C3C3C"
    background_tertiary: str = "#4F4F4F"

    # Text colors
    text_primary: str = "#CCCCCC"
    text_secondary: str = "#AAAAAA"
    text_disabled: str = "#666666"

    # Accent colors
    accent_primary: str = "#4A90E2"
    accent_secondary: str = "#5BA0F2"
    accent_hover: str = "#6BB0FF"

    # Status colors
    success: str = "#7CB342"
    warning: str = "#F5A623"
    error: str = "#D0021B"
    info: str = "#2196F3"

    # Border colors
    border_primary: str = "#555555"
    border_secondary: str = "#666666"
    border_focus: str = "#4A90E2"

    # Selection colors
    selection_background: str = "#4A90E2"
    selection_text: str = "#FFFFFF"


class EclipseThemeManager(QObject):
    """Manager cho Eclipse-style themes."""

    theme_changed = pyqtSignal(str) if HAS_PYQT5 else None

    def __init__(self):
        if HAS_PYQT5:
            super().__init__()

        self.current_theme = "dark"
        self.themes = {}
        self.settings = QSettings("QuangTPS", "ThemeManager") if HAS_PYQT5 else None

        # Initialize default themes
        self._initialize_default_themes()

        # Load saved theme
        self._load_saved_theme()

        logger.info(f"EclipseThemeManager initialized with theme: {self.current_theme}")

    def _initialize_default_themes(self):
        """Khởi tạo default themes."""

        # Dark theme (Eclipse default)
        self.themes["dark"] = EclipseColors()

        # Light theme
        self.themes["light"] = EclipseColors(
            background_primary="#FFFFFF",
            background_secondary="#F5F5F5",
            background_tertiary="#E0E0E0",
            text_primary="#333333",
            text_secondary="#666666",
            text_disabled="#AAAAAA",
            border_primary="#CCCCCC",
            border_secondary="#DDDDDD",
        )

        # High contrast theme
        self.themes["high_contrast"] = EclipseColors(
            background_primary="#000000",
            background_secondary="#1A1A1A",
            background_tertiary="#333333",
            text_primary="#FFFFFF",
            text_secondary="#CCCCCC",
            text_disabled="#888888",
            accent_primary="#00FF00",
            border_primary="#FFFFFF",
        )

    def _load_saved_theme(self):
        """Load saved theme từ settings."""
        if self.settings:
            saved_theme = self.settings.value("current_theme", "dark")
            if saved_theme in self.themes:
                self.current_theme = saved_theme

    def set_theme(self, theme_name: str):
        """Đặt theme hiện tại."""
        if theme_name not in self.themes:
            logger.warning(f"Theme '{theme_name}' not found")
            return False

        self.current_theme = theme_name

        # Save to settings
        if self.settings:
            self.settings.setValue("current_theme", theme_name)

        # Emit signal
        if self.theme_changed:
            self.theme_changed.emit(theme_name)

        logger.info(f"Theme changed to: {theme_name}")
        return True

    def get_current_colors(self) -> EclipseColors:
        """Lấy colors của theme hiện tại."""
        return self.themes[self.current_theme]

    def get_stylesheet(self, widget_type: str = "general") -> str:
        """Tạo stylesheet cho widget type."""
        colors = self.get_current_colors()

        if widget_type == "main_window":
            return self._get_main_window_stylesheet(colors)
        elif widget_type == "button":
            return self._get_button_stylesheet(colors)
        elif widget_type == "tab_widget":
            return self._get_tab_widget_stylesheet(colors)
        elif widget_type == "tree_widget":
            return self._get_tree_widget_stylesheet(colors)
        elif widget_type == "table_widget":
            return self._get_table_widget_stylesheet(colors)
        elif widget_type == "menu":
            return self._get_menu_stylesheet(colors)
        elif widget_type == "toolbar":
            return self._get_toolbar_stylesheet(colors)
        else:
            return self._get_general_stylesheet(colors)

    def _get_main_window_stylesheet(self, colors: EclipseColors) -> str:
        """Main window stylesheet."""
        return f"""
        QMainWindow {{
            background-color: {colors.background_primary};
            color: {colors.text_primary};
        }}

        QWidget {{
            background-color: {colors.background_primary};
            color: {colors.text_primary};
            font-family: "Segoe UI", Arial, sans-serif;
            font-size: 9pt;
        }}

        QStatusBar {{
            background-color: {colors.background_secondary};
            border-top: 1px solid {colors.border_primary};
            color: {colors.text_secondary};
        }}
        """

    def _get_button_stylesheet(self, colors: EclipseColors) -> str:
        """Button stylesheet."""
        return f"""
        QPushButton {{
            background-color: {colors.background_secondary};
            border: 1px solid {colors.border_primary};
            color: {colors.text_primary};
            padding: 6px 12px;
            border-radius: 3px;
            font-weight: normal;
        }}

        QPushButton:hover {{
            background-color: {colors.accent_primary};
            border-color: {colors.accent_secondary};
        }}

        QPushButton:pressed {{
            background-color: {colors.accent_hover};
        }}

        QPushButton:disabled {{
            background-color: {colors.background_tertiary};
            color: {colors.text_disabled};
            border-color: {colors.border_secondary};
        }}
        """

    def _get_tab_widget_stylesheet(self, colors: EclipseColors) -> str:
        """Tab widget stylesheet."""
        return f"""
        QTabWidget::pane {{
            border: 1px solid {colors.border_primary};
            background-color: {colors.background_primary};
        }}

        QTabBar::tab {{
            background-color: {colors.background_secondary};
            color: {colors.text_primary};
            padding: 8px 16px;
            margin-right: 2px;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
        }}

        QTabBar::tab:selected {{
            background-color: {colors.accent_primary};
            color: {colors.selection_text};
        }}

        QTabBar::tab:hover {{
            background-color: {colors.accent_secondary};
        }}
        """

    def _get_tree_widget_stylesheet(self, colors: EclipseColors) -> str:
        """Tree widget stylesheet."""
        return f"""
        QTreeWidget {{
            background-color: {colors.background_primary};
            color: {colors.text_primary};
            border: 1px solid {colors.border_primary};
            selection-background-color: {colors.selection_background};
            selection-color: {colors.selection_text};
        }}

        QTreeWidget::item {{
            padding: 4px;
            border-bottom: 1px solid {colors.background_secondary};
        }}

        QTreeWidget::item:hover {{
            background-color: {colors.background_secondary};
        }}

        QTreeWidget::item:selected {{
            background-color: {colors.selection_background};
        }}
        """

    def _get_table_widget_stylesheet(self, colors: EclipseColors) -> str:
        """Table widget stylesheet."""
        return f"""
        QTableWidget {{
            background-color: {colors.background_primary};
            color: {colors.text_primary};
            border: 1px solid {colors.border_primary};
            gridline-color: {colors.border_secondary};
            selection-background-color: {colors.selection_background};
        }}

        QHeaderView::section {{
            background-color: {colors.background_secondary};
            color: {colors.text_primary};
            padding: 6px;
            border: 1px solid {colors.border_primary};
            font-weight: bold;
        }}
        """

    def _get_menu_stylesheet(self, colors: EclipseColors) -> str:
        """Menu stylesheet."""
        return f"""
        QMenuBar {{
            background-color: {colors.background_secondary};
            color: {colors.text_primary};
            border-bottom: 1px solid {colors.border_primary};
        }}

        QMenuBar::item {{
            padding: 6px 12px;
        }}

        QMenuBar::item:selected {{
            background-color: {colors.accent_primary};
        }}

        QMenu {{
            background-color: {colors.background_secondary};
            color: {colors.text_primary};
            border: 1px solid {colors.border_primary};
        }}

        QMenu::item {{
            padding: 6px 20px;
        }}

        QMenu::item:selected {{
            background-color: {colors.accent_primary};
        }}
        """

    def _get_toolbar_stylesheet(self, colors: EclipseColors) -> str:
        """Toolbar stylesheet."""
        return f"""
        QToolBar {{
            background-color: {colors.background_secondary};
            border: 1px solid {colors.border_primary};
            spacing: 2px;
        }}

        QToolButton {{
            background-color: transparent;
            border: 1px solid transparent;
            padding: 4px;
            border-radius: 3px;
        }}

        QToolButton:hover {{
            background-color: {colors.accent_primary};
            border-color: {colors.accent_secondary};
        }}

        QToolButton:pressed {{
            background-color: {colors.accent_hover};
        }}
        """

    def _get_general_stylesheet(self, colors: EclipseColors) -> str:
        """General stylesheet."""
        return f"""
        QLineEdit {{
            background-color: {colors.background_primary};
            color: {colors.text_primary};
            border: 1px solid {colors.border_primary};
            padding: 4px;
            border-radius: 3px;
        }}

        QLineEdit:focus {{
            border-color: {colors.border_focus};
        }}

        QComboBox {{
            background-color: {colors.background_secondary};
            color: {colors.text_primary};
            border: 1px solid {colors.border_primary};
            padding: 4px;
            border-radius: 3px;
        }}

        QSpinBox, QDoubleSpinBox {{
            background-color: {colors.background_primary};
            color: {colors.text_primary};
            border: 1px solid {colors.border_primary};
            padding: 4px;
            border-radius: 3px;
        }}

        QCheckBox {{
            color: {colors.text_primary};
        }}

        QRadioButton {{
            color: {colors.text_primary};
        }}

        QGroupBox {{
            color: {colors.text_primary};
            border: 1px solid {colors.border_primary};
            border-radius: 5px;
            margin-top: 10px;
            font-weight: bold;
        }}

        QGroupBox::title {{
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px 0 5px;
        }}
        """

    def apply_theme_to_application(self):
        """Áp dụng theme cho toàn bộ application."""
        if not HAS_PYQT5:
            return

        app = QApplication.instance()
        if not app:
            return

        # Set application stylesheet
        stylesheet = self.get_stylesheet("main_window")
        stylesheet += self.get_stylesheet("button")
        stylesheet += self.get_stylesheet("tab_widget")
        stylesheet += self.get_stylesheet("tree_widget")
        stylesheet += self.get_stylesheet("table_widget")
        stylesheet += self.get_stylesheet("menu")
        stylesheet += self.get_stylesheet("toolbar")
        stylesheet += self.get_stylesheet("general")

        app.setStyleSheet(stylesheet)

        # Set application palette
        self._set_application_palette()

        logger.info(f"Applied {self.current_theme} theme to application")

    def _set_application_palette(self):
        """Đặt application palette."""
        if not HAS_PYQT5:
            return

        app = QApplication.instance()
        if not app:
            return

        colors = self.get_current_colors()
        palette = QPalette()

        # Window colors
        palette.setColor(QPalette.Window, QColor(colors.background_primary))
        palette.setColor(QPalette.WindowText, QColor(colors.text_primary))

        # Base colors
        palette.setColor(QPalette.Base, QColor(colors.background_primary))
        palette.setColor(QPalette.AlternateBase, QColor(colors.background_secondary))

        # Text colors
        palette.setColor(QPalette.Text, QColor(colors.text_primary))
        palette.setColor(QPalette.BrightText, QColor(colors.text_primary))

        # Button colors
        palette.setColor(QPalette.Button, QColor(colors.background_secondary))
        palette.setColor(QPalette.ButtonText, QColor(colors.text_primary))

        # Highlight colors
        palette.setColor(QPalette.Highlight, QColor(colors.selection_background))
        palette.setColor(QPalette.HighlightedText, QColor(colors.selection_text))

        app.setPalette(palette)

    def get_icon_path(self, icon_name: str, theme_variant: str = None) -> str:
        """Lấy đường dẫn icon theo theme."""
        if theme_variant is None:
            theme_variant = "dark" if self.current_theme == "dark" else "light"

        # Base icon directory
        icon_dir = Path(__file__).parent.parent / "icons"

        # Try theme-specific icon first
        theme_icon_path = icon_dir / theme_variant / f"{icon_name}.png"
        if theme_icon_path.exists():
            return str(theme_icon_path)

        # Fallback to general icon
        general_icon_path = icon_dir / f"{icon_name}.png"
        if general_icon_path.exists():
            return str(general_icon_path)

        # Return empty string if not found
        logger.warning(f"Icon '{icon_name}' not found")
        return ""

    def create_icon(self, icon_name: str, size: int = 16) -> QIcon:
        """Tạo QIcon từ icon name."""
        if not HAS_PYQT5:
            return None

        icon_path = self.get_icon_path(icon_name)
        if icon_path:
            return QIcon(icon_path)
        else:
            # Create placeholder icon
            pixmap = QPixmap(size, size)
            pixmap.fill(QColor(self.get_current_colors().accent_primary))
            return QIcon(pixmap)


# Global theme manager instance
_theme_manager = None


def get_theme_manager() -> EclipseThemeManager:
    """Lấy global theme manager instance."""
    global _theme_manager
    if _theme_manager is None:
        _theme_manager = EclipseThemeManager()
    return _theme_manager


def apply_eclipse_theme(widget: QWidget = None):
    """Áp dụng Eclipse theme cho widget hoặc toàn bộ app."""
    theme_manager = get_theme_manager()

    if widget is None:
        # Apply to entire application
        theme_manager.apply_theme_to_application()
    else:
        # Apply to specific widget
        if HAS_PYQT5:
            stylesheet = theme_manager.get_stylesheet("general")
            widget.setStyleSheet(stylesheet)
