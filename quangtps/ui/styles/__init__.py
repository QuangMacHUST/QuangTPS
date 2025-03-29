#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
UI styling utilities for QuangTPS.
"""

import os
from typing import Dict, Optional
from PyQt5.QtGui import QIcon, QColor, QPixmap

# Define color constants used throughout the application
class Colors:
    """Color constants for consistent UI styling"""
    PRIMARY = "#2c5aa0"  # Eclipse-like blue
    SECONDARY = "#5b9bd5"
    BACKGROUND = "#f0f0f0"
    DARK_TEXT = "#333333"
    LIGHT_TEXT = "#ffffff"
    WARNING = "#e69138"
    ERROR = "#cc0000"
    SUCCESS = "#6aa84f"
    INFO = "#3d85c6"
    
    # Structure colors
    PTV = "#ff0000"  # Red
    CTV = "#ff9999"  # Light red
    OAR = "#00aa00"  # Green
    NORMAL_TISSUE = "#aaaaaa"  # Gray
    BODY = "#ffcc00"  # Yellow
    
    # Structure type to color mapping
    STRUCTURE_COLORS = {
        "PTV": PTV,
        "CTV": CTV,
        "GTV": "#ff6600",  # Orange
        "OAR": OAR,
        "ORGAN": "#00ccff",  # Light blue
        "NORMAL": NORMAL_TISSUE,
        "BODY": BODY,
        "EXTERNAL": "#ffcc00",  # Yellow
        "SUPPORT": "#cc99ff",  # Purple
        "BOLUS": "#ff99cc",  # Pink
        "REFERENCE": "#ffff00"  # Bright yellow
    }
    
    @classmethod
    def get_structure_color(cls, structure_type: str) -> str:
        """Get color for a structure type."""
        return cls.STRUCTURE_COLORS.get(structure_type.upper(), "#3366ff")  # Default blue


def get_icon(name: str) -> QIcon:
    """
    Get an icon from the icons directory.
    
    Parameters
    ----------
    name : str
        Icon name without extension
        
    Returns
    -------
    QIcon
        The icon
    """
    icon_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "icons", f"{name}.png")
    if os.path.exists(icon_path):
        return QIcon(icon_path)
    
    # Fallback to new_icons directory
    new_icon_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "icons", "new_icons", f"{name}.png")
    if os.path.exists(new_icon_path):
        return QIcon(new_icon_path)
    
    # Create a colored square as placeholder icon
    pixmap = QPixmap(32, 32)
    pixmap.fill(QColor("#3366cc"))
    return QIcon(pixmap)


def load_stylesheet() -> str:
    """
    Load the main application stylesheet.
    
    Returns
    -------
    str
        The stylesheet content
    """
    stylesheet_path = os.path.join(os.path.dirname(__file__), "main_style.qss")
    if os.path.exists(stylesheet_path):
        with open(stylesheet_path, "r") as f:
            return f.read()
    return "" 