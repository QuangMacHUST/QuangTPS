#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module tiện ích UI cho QuangTPS.

Module này cung cấp các hàm tiện ích giúp tạo và định dạng các thành phần UI
theo phong cách Eclipse.
"""

import os
import logging
from typing import Optional, Union, Dict, Any, List, Tuple

try:
    from PyQt5.QtWidgets import QWidget, QApplication, QStyleFactory
    from PyQt5.QtCore import QSize, Qt
    from PyQt5.QtGui import QIcon, QPixmap, QColor, QPalette

    HAS_QT = True
except ImportError:
    HAS_QT = False
    logging.warning("PyQt5 không khả dụng. Các chức năng UI sẽ bị giới hạn.")

logger = logging.getLogger(__name__)


def create_eclipse_icon(icon_name: str) -> Optional["QIcon"]:
    """
    Tạo biểu tượng theo phong cách Eclipse cho các nút.

    Function này tìm kiếm biểu tượng trong thư mục icons của QuangTPS và
    trả về QIcon tương ứng. Các biểu tượng được tổ chức theo phong cách của
    Eclipse, với các icon chuẩn cho các chức năng phổ biến.

    Parameters
    ----------
    icon_name : str
        Tên của biểu tượng cần tạo (không bao gồm phần mở rộng)

    Returns
    -------
    Optional[QIcon]
        Đối tượng QIcon nếu tìm thấy biểu tượng, None nếu không tìm thấy
        hoặc PyQt không khả dụng
    """
    if not HAS_QT:
        return None

    # Các đường dẫn có thể chứa biểu tượng
    icon_paths = [
        os.path.join("quangtps", "ui", "icons", f"{icon_name}.png"),
        os.path.join("quangtps", "ui", "icons", "new_icons", f"{icon_name}.png"),
        os.path.join("quangtps", "ui", "icons", "eclipse", f"{icon_name}.png"),
        os.path.join("ui", "icons", f"{icon_name}.png"),
        os.path.join("icons", f"{icon_name}.png"),
    ]

    # Kiểm tra từng đường dẫn
    for path in icon_paths:
        if os.path.exists(path):
            return QIcon(path)

    # Nếu không tìm thấy, tạo biểu tượng mặc định dựa trên tên
    try:
        # Ánh xạ tên biểu tượng phổ biến sang biểu tượng có sẵn trong Qt
        icon_map = {
            "new": QApplication.style().standardIcon(QApplication.style().SP_FileIcon),
            "open": QApplication.style().standardIcon(
                QApplication.style().SP_DirOpenIcon
            ),
            "save": QApplication.style().standardIcon(
                QApplication.style().SP_DriveFDIcon
            ),
            "delete": QApplication.style().standardIcon(
                QApplication.style().SP_TrashIcon
            ),
            "cancel": QApplication.style().standardIcon(
                QApplication.style().SP_DialogCancelButton
            ),
            "apply": QApplication.style().standardIcon(
                QApplication.style().SP_DialogApplyButton
            ),
            "ok": QApplication.style().standardIcon(
                QApplication.style().SP_DialogOkButton
            ),
            "help": QApplication.style().standardIcon(
                QApplication.style().SP_DialogHelpButton
            ),
            "refresh": QApplication.style().standardIcon(
                QApplication.style().SP_BrowserReload
            ),
            "calculate": QApplication.style().standardIcon(
                QApplication.style().SP_ComputerIcon
            ),
            "settings": QApplication.style().standardIcon(
                QApplication.style().SP_FileDialogDetailedView
            ),
            "edit": QApplication.style().standardIcon(
                QApplication.style().SP_FileDialogContentsView
            ),
            "view": QApplication.style().standardIcon(
                QApplication.style().SP_FileDialogListView
            ),
            "generate": QApplication.style().standardIcon(
                QApplication.style().SP_ArrowRight
            ),
            "analysis": QApplication.style().standardIcon(
                QApplication.style().SP_FileDialogInfoView
            ),
            "optimize": QApplication.style().standardIcon(
                QApplication.style().SP_MediaPlay
            ),
            "kbp": QApplication.style().standardIcon(
                QApplication.style().SP_DialogApplyButton
            ),
        }

        if icon_name in icon_map:
            return icon_map[icon_name]
        else:
            # Nếu không có trong ánh xạ, sử dụng biểu tượng mặc định
            return QApplication.style().standardIcon(
                QApplication.style().SP_TitleBarMenuButton
            )
    except Exception as e:
        logger.warning(f"Không thể tạo biểu tượng {icon_name}: {e}")
        return None


def apply_eclipse_theme(widget: "QWidget") -> None:
    """
    Áp dụng theme kiểu Eclipse cho widget.

    Parameters
    ----------
    widget : QWidget
        Widget cần áp dụng theme
    """
    if not HAS_QT:
        return

    try:
        # Thiết lập style cho widget
        if "Fusion" in QStyleFactory.keys():
            QApplication.setStyle(QStyleFactory.create("Fusion"))

        # Tạo palette màu kiểu Eclipse
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(240, 240, 240))
        palette.setColor(QPalette.WindowText, QColor(0, 0, 0))
        palette.setColor(QPalette.Base, QColor(255, 255, 255))
        palette.setColor(QPalette.AlternateBase, QColor(245, 245, 245))
        palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 220))
        palette.setColor(QPalette.ToolTipText, QColor(0, 0, 0))
        palette.setColor(QPalette.Text, QColor(0, 0, 0))
        palette.setColor(QPalette.Button, QColor(240, 240, 240))
        palette.setColor(QPalette.ButtonText, QColor(0, 0, 0))
        palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))

        # Áp dụng palette
        widget.setPalette(palette)
    except Exception as e:
        logger.warning(f"Không thể áp dụng theme Eclipse: {e}")
