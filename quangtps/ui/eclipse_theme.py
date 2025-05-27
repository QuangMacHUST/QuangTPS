#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Eclipse Theme Module for QuangTPS
"""

import logging

logger = logging.getLogger(__name__)


class EclipseColors:
    """Màu sắc theo phong cách Eclipse"""

    BACKGROUND = "#2B2B2B"
    PANEL = "#3C3C3C"
    BORDER = "#555555"
    TEXT = "#CCCCCC"
    ACCENT = "#4A90E2"
    WARNING = "#F5A623"
    ERROR = "#D0021B"
    SUCCESS = "#7ED321"
    TAB_ACTIVE = "#4A90E2"
    TAB_INACTIVE = "#2B2B2B"


def get_eclipse_stylesheet():
    """Lấy stylesheet Eclipse"""
    return f"""
    QMainWindow {{
        background-color: {EclipseColors.BACKGROUND};
        color: {EclipseColors.TEXT};
    }}

    QTabWidget::pane {{
        border: 1px solid {EclipseColors.BORDER};
        background-color: {EclipseColors.BACKGROUND};
    }}

    QTabBar::tab {{
        background-color: {EclipseColors.TAB_INACTIVE};
        color: {EclipseColors.TEXT};
        padding: 8px 16px;
        margin-right: 2px;
    }}

    QTabBar::tab:selected {{
        background-color: {EclipseColors.TAB_ACTIVE};
        color: white;
    }}

    QPushButton {{
        background-color: {EclipseColors.PANEL};
        border: 1px solid {EclipseColors.BORDER};
        color: {EclipseColors.TEXT};
        padding: 5px 10px;
    }}

    QPushButton:hover {{
        background-color: {EclipseColors.ACCENT};
        color: white;
    }}
    """


def apply_eclipse_theme(widget):
    """Áp dụng Eclipse theme cho widget"""
    try:
        widget.setStyleSheet(get_eclipse_stylesheet())
        logger.debug("Đã áp dụng Eclipse theme")
    except Exception as e:
        logger.error(f"Lỗi khi áp dụng Eclipse theme: {e}")
