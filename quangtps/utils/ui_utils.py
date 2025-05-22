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

    HAS_QT = True
except ImportError:
    HAS_QT = False
    logging.warning("PyQt5 không khả dụng. Các chức năng UI sẽ bị giới hạn.")

logger = logging.getLogger(__name__)


def create_eclipse_icon(
    icon_name: str, custom_render: bool = True, size: int = 32, colors: Dict = None
) -> Optional["QIcon"]:
    """
    Tạo biểu tượng theo phong cách Eclipse cho các nút.

    Function này tìm kiếm biểu tượng trong thư mục icons của QuangTPS và
    trả về QIcon tương ứng. Các biểu tượng được tổ chức theo phong cách của
    Eclipse, với các icon chuẩn cho các chức năng phổ biến.

    Nếu custom_render=True và icon_name là "kbp", "rapidplan" hoặc các loại đặc biệt khác,
    tạo biểu tượng tùy chỉnh với QPainter.

    Parameters
    ----------
    icon_name : str
        Tên của biểu tượng cần tạo (không bao gồm phần mở rộng)
    custom_render : bool, optional
        Nếu True, sử dụng QPainter để render các biểu tượng đặc biệt như kbp hay rapidplan
    size : int, optional
        Kích thước của biểu tượng (mặc định: 32px)
    colors : Dict, optional
        Từ điển chứa màu sắc tùy chỉnh cho biểu tượng

    Returns
    -------
    Optional[QIcon]
        Đối tượng QIcon nếu tìm thấy biểu tượng, None nếu không tìm thấy
        hoặc PyQt không khả dụng
    """
    if not HAS_QT:
        logger.warning("PyQt không khả dụng, không thể tạo biểu tượng.")
        return None

    # Sử dụng màu sắc mặc định nếu không được cung cấp
    if colors is None:
        colors = {
            "kbp": {
                "background": QColor(41, 128, 185),  # Xanh dương
                "border": QColor(52, 152, 219),
                "text": QColor(255, 255, 255),  # Trắng
                "graphic": QColor(255, 255, 255),  # Trắng
            },
            "rapidplan": {
                "background": QColor(155, 89, 182),  # Tím
                "border": QColor(142, 68, 173),
                "text": QColor(255, 255, 255),  # Trắng
                "graphic": QColor(255, 255, 255),  # Trắng
            },
            "default": {
                "background": QColor(52, 152, 219),  # Xanh dương nhạt
                "border": QColor(41, 128, 185),
                "text": QColor(255, 255, 255),  # Trắng
                "graphic": QColor(255, 255, 255),  # Trắng
            },
        }

    # Tạo biểu tượng tùy chỉnh với QPainter nếu được yêu cầu
    if custom_render:
        if icon_name.lower() in ["kbp", "rapidplan", "mcmc", "mco_navigator"]:
            try:
                # Tạo pixmap trống
                pixmap = QPixmap(size, size)
                pixmap.fill(Qt.transparent)

                # Tạo painter
                painter = QPainter(pixmap)
                painter.setRenderHint(QPainter.Antialiasing, True)

                # Lấy màu sắc phù hợp
                color_set = colors.get(icon_name.lower(), colors["default"])

                # Vẽ biểu tượng tùy thuộc vào loại
                if icon_name.lower() == "kbp":
                    # Vẽ biểu tượng KBP (hình tròn với chữ K bên trong)
                    # Vẽ hình tròn với màu nền
                    brush = QBrush(color_set["background"])
                    painter.setBrush(brush)

                    # Viền
                    pen = QPen(color_set["border"], 1)
                    painter.setPen(pen)

                    # Vẽ hình tròn
                    painter.drawEllipse(4, 4, size - 8, size - 8)

                    # Vẽ chữ "K" bên trong
                    font = painter.font()
                    font.setBold(True)
                    font.setPointSize(10) if hasattr(
                        font, "setPointSize"
                    ) else font.setPixelSize(12)
                    painter.setFont(font)

                    pen = QPen(color_set["text"])
                    painter.setPen(pen)
                    painter.drawText(QRect(0, 0, size, size), Qt.AlignCenter, "K")

                    # Vẽ đường cong đại diện cho dữ liệu
                    pen = QPen(color_set["graphic"], 2)
                    painter.setPen(pen)
                    painter.drawLine(8, size / 2 + 4, size - 8, size / 2 - 4)

                elif icon_name.lower() == "rapidplan":
                    # Vẽ biểu tượng RapidPlan
                    brush = QBrush(color_set["background"])
                    painter.setBrush(brush)

                    # Viền
                    pen = QPen(color_set["border"], 1)
                    painter.setPen(pen)

                    # Vẽ hình tròn
                    painter.drawEllipse(4, 4, size - 8, size - 8)

                    # Vẽ chữ "R" bên trong
                    font = painter.font()
                    font.setBold(True)
                    font.setPointSize(10) if hasattr(
                        font, "setPointSize"
                    ) else font.setPixelSize(12)
                    painter.setFont(font)

                    pen = QPen(color_set["text"])
                    painter.setPen(pen)
                    painter.drawText(QRect(0, 0, size, size), Qt.AlignCenter, "R")

                elif icon_name.lower() == "mcmc":
                    # Vẽ biểu tượng Monte Carlo model comparison
                    brush = QBrush(QColor(46, 204, 113))  # Xanh lá
                    painter.setBrush(brush)

                    # Viền
                    pen = QPen(QColor(39, 174, 96), 1)
                    painter.setPen(pen)

                    # Vẽ hình tròn
                    painter.drawEllipse(4, 4, size - 8, size - 8)

                    # Vẽ chữ "MC" bên trong
                    font = painter.font()
                    font.setBold(True)
                    font.setPointSize(9) if hasattr(
                        font, "setPointSize"
                    ) else font.setPixelSize(11)
                    painter.setFont(font)

                    pen = QPen(Qt.white)
                    painter.setPen(pen)
                    painter.drawText(QRect(0, 0, size, size), Qt.AlignCenter, "MC")

                elif icon_name.lower() == "mco_navigator":
                    # Vẽ biểu tượng MCO Navigator
                    brush = QBrush(QColor(230, 126, 34))  # Cam
                    painter.setBrush(brush)

                    # Viền
                    pen = QPen(QColor(211, 84, 0), 1)
                    painter.setPen(pen)

                    # Vẽ hình tròn
                    painter.drawEllipse(4, 4, size - 8, size - 8)

                    # Vẽ chữ "MCO" bên trong
                    font = painter.font()
                    font.setBold(True)
                    font.setPointSize(7) if hasattr(
                        font, "setPointSize"
                    ) else font.setPixelSize(9)
                    painter.setFont(font)

                    pen = QPen(Qt.white)
                    painter.setPen(pen)
                    painter.drawText(QRect(0, 0, size, size), Qt.AlignCenter, "MCO")

                painter.end()

                # Tạo QIcon từ pixmap
                return QIcon(pixmap)
            except Exception as e:
                logger.error(f"Lỗi khi tạo biểu tượng tùy chỉnh {icon_name}: {e}")
                # Nếu lỗi, tiếp tục với phương pháp tìm kiếm file

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
