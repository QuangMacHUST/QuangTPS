#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module cung cấp các widget dùng chung cho QuangTPS.

Module này định nghĩa các widget tùy chỉnh có thể được sử dụng
trong nhiều phần khác nhau của ứng dụng.
"""

import logging
import os
from typing import Optional, Callable, Dict, List, Tuple, Any

from PyQt5.QtCore import Qt, QSize, pyqtSignal, QRect, QPropertyAnimation, QEvent
from PyQt5.QtGui import QIcon, QColor, QPainter, QPixmap
from PyQt5.QtWidgets import (
    QPushButton, QWidget, QColorDialog, QVBoxLayout, QHBoxLayout,
    QFrame, QScrollArea, QToolButton, QSizePolicy, QStyle
)

logger = logging.getLogger(__name__)


class IconButton(QPushButton):
    """
    Nút với biểu tượng và kiểu dáng tùy chỉnh.
    
    Class này mở rộng QPushButton để cung cấp một nút với biểu tượng
    và kiểu dáng tùy chỉnh phù hợp với giao diện của QuangTPS.
    """
    
    def __init__(self, icon_name: str = None, tooltip: str = None, parent=None):
        """
        Khởi tạo nút biểu tượng.
        
        Parameters
        ----------
        icon_name : str, optional
            Tên biểu tượng hoặc đường dẫn đến file biểu tượng
        tooltip : str, optional
            Tooltip hiển thị khi di chuột qua nút
        parent : QWidget, optional
            Widget cha
        """
        super().__init__(parent)
        
        # Thiết lập kiểu dáng
        self.setFlat(True)
        self.setCursor(Qt.PointingHandCursor)
        
        # Thiết lập biểu tượng nếu được cung cấp
        if icon_name:
            self.set_icon(icon_name)
        
        # Thiết lập tooltip nếu được cung cấp
        if tooltip:
            self.setToolTip(tooltip)
    
    def set_icon(self, icon_name: str, size: int = 24):
        """
        Thiết lập biểu tượng cho nút.
        
        Parameters
        ----------
        icon_name : str
            Tên biểu tượng hoặc đường dẫn đến file biểu tượng
        size : int, optional
            Kích thước biểu tượng, mặc định là 24px
        """
        # Kiểm tra xem icon_name là đường dẫn file hay tên biểu tượng
        if os.path.isfile(icon_name):
            icon = QIcon(icon_name)
        else:
            # Sử dụng biểu tượng từ style của Qt
            icon = self.style().standardIcon(getattr(QStyle, icon_name, QStyle.SP_CustomBase))
            
            # Nếu không tìm thấy biểu tượng trong style, sử dụng biểu tượng mặc định
            if icon.isNull():
                logger.warning(f"Icon '{icon_name}' not found, using default icon")
                icon = self.style().standardIcon(QStyle.SP_CustomBase)
        
        self.setIcon(icon)
        self.setIconSize(QSize(size, size))


class ColorButton(QPushButton):
    """
    Nút chọn màu.
    
    Class này cung cấp một nút cho phép người dùng chọn màu
    từ hộp thoại chọn màu.
    """
    
    # Tín hiệu phát ra khi màu thay đổi
    colorChanged = pyqtSignal(QColor)
    
    def __init__(self, color: QColor = None, parent=None):
        """
        Khởi tạo nút chọn màu.
        
        Parameters
        ----------
        color : QColor, optional
            Màu ban đầu, mặc định là màu đen
        parent : QWidget, optional
            Widget cha
        """
        super().__init__(parent)
        
        # Màu hiện tại
        self._color = color or QColor(0, 0, 0)
        
        # Thiết lập kiểu dáng
        self.setFixedSize(30, 30)
        self.setCursor(Qt.PointingHandCursor)
        
        # Kết nối sự kiện nhấp chuột
        self.clicked.connect(self._on_button_clicked)
        
        # Cập nhật hiển thị màu
        self._update_button_color()
    
    def color(self) -> QColor:
        """
        Lấy màu hiện tại.
        
        Returns
        -------
        QColor
            Màu hiện tại
        """
        return self._color
    
    def setColor(self, color: QColor):
        """
        Thiết lập màu mới.
        
        Parameters
        ----------
        color : QColor
            Màu mới
        """
        if color != self._color:
            self._color = color
            self._update_button_color()
            self.colorChanged.emit(self._color)
    
    def _update_button_color(self):
        """Cập nhật hiển thị màu trên nút."""
        # Tạo pixmap với màu hiện tại
        pixmap = QPixmap(self.size())
        pixmap.fill(self._color)
        
        # Vẽ viền
        painter = QPainter(pixmap)
        painter.setPen(Qt.black)
        painter.drawRect(0, 0, pixmap.width() - 1, pixmap.height() - 1)
        painter.end()
        
        # Thiết lập pixmap làm biểu tượng
        self.setIcon(QIcon(pixmap))
        self.setIconSize(self.size())
    
    def _on_button_clicked(self):
        """Xử lý sự kiện khi nút được nhấp."""
        # Hiển thị hộp thoại chọn màu
        color = QColorDialog.getColor(self._color, self, "Chọn màu")
        
        # Nếu người dùng chọn màu (không nhấn Cancel)
        if color.isValid():
            self.setColor(color)


class CollapsibleBox(QWidget):
    """
    Widget hộp có thể mở rộng/thu gọn.
    
    Class này cung cấp một widget hộp có thể mở rộng/thu gọn
    để hiển thị/ẩn nội dung bên trong.
    """
    
    def __init__(self, title: str = "", parent=None):
        """
        Khởi tạo hộp có thể mở rộng/thu gọn.
        
        Parameters
        ----------
        title : str, optional
            Tiêu đề của hộp
        parent : QWidget, optional
            Widget cha
        """
        super().__init__(parent)
        
        # Thiết lập layout
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        
        # Nút mở rộng/thu gọn
        self.toggle_button = QToolButton()
        self.toggle_button.setStyleSheet("QToolButton { border: none; }")
        self.toggle_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.toggle_button.setArrowType(Qt.RightArrow)
        self.toggle_button.setText(title)
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(False)
        
        # Kết nối sự kiện
        self.toggle_button.clicked.connect(self._on_toggle)
        
        # Thiết lập header
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.addWidget(self.toggle_button)
        header_layout.addStretch()
        
        # Khung nội dung
        self.content_area = QScrollArea()
        self.content_area.setFrameShape(QFrame.NoFrame)
        self.content_area.setWidgetResizable(True)
        
        # Widget chứa nội dung
        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_area.setWidget(self.content_widget)
        
        # Thêm các widget vào layout chính
        self.main_layout.addLayout(header_layout)
        self.main_layout.addWidget(self.content_area)
        
        # Ban đầu, nội dung bị ẩn
        self.content_area.setMaximumHeight(0)
        self.content_area.setMinimumHeight(0)
        
        # Animation cho việc mở rộng/thu gọn
        self.animation = QPropertyAnimation(self.content_area, b"maximumHeight")
        self.animation.setDuration(300)  # 300ms
        
        # Thiết lập kích thước
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
    
    def _on_toggle(self, checked: bool):
        """
        Xử lý sự kiện khi nút mở rộng/thu gọn được nhấp.
        
        Parameters
        ----------
        checked : bool
            Trạng thái mới của nút (True: đã chọn, False: chưa chọn)
        """
        # Thay đổi kiểu mũi tên
        arrow_type = Qt.DownArrow if checked else Qt.RightArrow
        self.toggle_button.setArrowType(arrow_type)
        
        # Thiết lập animation
        if checked:
            # Mở rộng nội dung
            self.animation.setStartValue(0)
            self.animation.setEndValue(self.content_widget.sizeHint().height())
        else:
            # Thu gọn nội dung
            self.animation.setStartValue(self.content_area.height())
            self.animation.setEndValue(0)
        
        # Bắt đầu animation
        self.animation.start()
    
    def setContentLayout(self, layout):
        """
        Thiết lập layout cho nội dung.
        
        Parameters
        ----------
        layout : QLayout
            Layout mới cho nội dung
        """
        # Xóa layout cũ
        QWidget().setLayout(self.content_layout)
        
        # Thiết lập layout mới
        self.content_layout = layout
        self.content_widget.setLayout(self.content_layout)
        
        # Mặc định là đã mở rộng
        self.toggle_button.setChecked(True)
        self.toggle_button.setArrowType(Qt.DownArrow)
        self.animation.setStartValue(0)
        self.animation.setEndValue(self.content_widget.sizeHint().height())
        self.animation.start()
    
    def addWidget(self, widget):
        """
        Thêm widget vào nội dung.
        
        Parameters
        ----------
        widget : QWidget
            Widget cần thêm
        """
        self.content_layout.addWidget(widget)
    
    def setTitle(self, title: str):
        """
        Thiết lập tiêu đề cho hộp.
        
        Parameters
        ----------
        title : str
            Tiêu đề mới
        """
        self.toggle_button.setText(title)
