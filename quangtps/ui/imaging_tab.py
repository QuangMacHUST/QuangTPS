#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module tạo tab imaging cho QuangTPS.

Module này triển khai giao diện tab imaging, tích hợp tất cả 
các tính năng hiển thị hình ảnh và công cụ contour.
"""

import os
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                           QPushButton, QTabWidget, QSplitter, QGroupBox,
                           QListWidget, QListWidgetItem, QSlider, QComboBox,
                           QFileDialog, QMessageBox, QToolBar, QAction,
                           QScrollArea, QFrame, QColorDialog, QToolButton,
                           QButtonGroup, QMenu, QCheckBox, QSpinBox, QDoubleSpinBox)
from PyQt5.QtCore import Qt, QSize, pyqtSignal, pyqtSlot, QDir, QPoint, QRect
from PyQt5.QtGui import QIcon, QPixmap, QColor, QPalette, QPainter, QPen, QBrush

from quangtps.ui.image_display import ImageSliceWidget, ImageControlWidget
from quangtps.ui.dicom_loader import DicomLoader, DicomSeries
from quangtps.ui.base_contour_tool import ContourToolManager, Contour, ContourCollection
from quangtps.ui.freehand_contour_tool import FreehandContourTool, BrushContourTool
from quangtps.ui.geometric_contour_tool import CircleContourTool, RectangleContourTool
from quangtps.ui.threshold_contour_tool import ThresholdContourTool, AutoThresholdContourTool

logger = logging.getLogger(__name__)


class ContourControlWidget(QWidget):
    """Widget điều khiển contour cho QuangTPS."""
    
    # Định nghĩa các tín hiệu
    tool_changed = pyqtSignal(str)
    contour_added = pyqtSignal(str, QColor)
    contour_deleted = pyqtSignal(str)
    contour_selected = pyqtSignal(str)
    contour_visibility_changed = pyqtSignal(str, bool)
    contour_color_changed = pyqtSignal(str, QColor)
    
    def __init__(self, parent=None):
        """
        Khởi tạo widget điều khiển contour.
        
        Parameters
        ----------
        parent : QWidget
            Widget cha
        """
        super().__init__(parent)
        
        self.contours = ContourCollection()
        self.tool_manager = None
        self.active_contour_name = None
        
        self._init_ui()
    
    def _init_ui(self):
        """Khởi tạo giao diện người dùng."""
        # Layout chính
        main_layout = QVBoxLayout(self)
        
        # === Phần công cụ contour ===
        tools_group = QGroupBox("Công cụ contour")
        tools_layout = QVBoxLayout(tools_group)
        
        # Nút công cụ
        tools_buttons_layout = QHBoxLayout()
        
        # Tạo các nút công cụ
        self.tool_buttons = {}
        for tool_name, icon_name, tooltip in [
            ("Freehand", "pen.png", "Vẽ tự do"),
            ("Brush", "brush.png", "Vẽ bằng cọ"),
            ("Circle", "circle.png", "Vẽ hình tròn"),
            ("Rectangle", "rectangle.png", "Vẽ hình chữ nhật"),
            ("Threshold", "threshold.png", "Vẽ dựa trên ngưỡng"),
            ("AutoThreshold", "auto_threshold.png", "Ngưỡng tự động"),
        ]:
            button = QToolButton()
            button.setToolTip(tooltip)
            button.setCheckable(True)
            
            # Thiết lập icon nếu có
            icon_path = os.path.join(os.path.dirname(__file__), 'icons', icon_name)
            if os.path.exists(icon_path):
                button.setIcon(QIcon(icon_path))
            else:
                button.setText(tool_name)
            
            button.clicked.connect(lambda checked, name=tool_name: self._on_tool_changed(name))
            tools_buttons_layout.addWidget(button)
            self.tool_buttons[tool_name] = button
        
        tools_layout.addLayout(tools_buttons_layout)
        
        # Thêm các cài đặt công cụ (sẽ được điền khi công cụ được chọn)
        self.tool_settings_widget = QWidget()
        self.tool_settings_layout = QVBoxLayout(self.tool_settings_widget)
        tools_layout.addWidget(self.tool_settings_widget)
        
        # === Phần quản lý contour ===
        contour_group = QGroupBox("Quản lý contour")
        contour_layout = QVBoxLayout(contour_group)
        
        # Nút thêm/xóa contour
        contour_buttons_layout = QHBoxLayout()
        
        self.add_contour_button = QPushButton("Thêm")
        self.add_contour_button.clicked.connect(self._on_add_contour)
        contour_buttons_layout.addWidget(self.add_contour_button)
        
        self.delete_contour_button = QPushButton("Xóa")
        self.delete_contour_button.clicked.connect(self._on_delete_contour)
        contour_buttons_layout.addWidget(self.delete_contour_button)
        
        contour_layout.addLayout(contour_buttons_layout)
        
        # Danh sách contour
        self.contour_list = QListWidget()
        self.contour_list.itemClicked.connect(self._on_contour_selected)
        contour_layout.addWidget(self.contour_list)
        
        # === Thêm vào layout chính ===
        main_layout.addWidget(tools_group)
        main_layout.addWidget(contour_group)
        main_layout.addStretch(1)
        
        # Thiết lập công cụ mặc định
        if "Freehand" in self.tool_buttons:
            self.tool_buttons["Freehand"].setChecked(True)
    
    def set_tool_manager(self, tool_manager):
        """
        Thiết lập trình quản lý công cụ.
        
        Parameters
        ----------
        tool_manager : ContourToolManager
            Trình quản lý công cụ contour
        """
        self.tool_manager = tool_manager
    
    def _on_tool_changed(self, tool_name):
        """
        Xử lý khi thay đổi công cụ.
        
        Parameters
        ----------
        tool_name : str
            Tên công cụ
        """
        # Bỏ chọn tất cả các nút khác
        for name, button in self.tool_buttons.items():
            if name != tool_name:
                button.setChecked(False)
        
        # Phát tín hiệu thay đổi công cụ
        self.tool_changed.emit(tool_name)
        
        # Cập nhật giao diện cài đặt công cụ
        self._update_tool_settings(tool_name)
    
    def _update_tool_settings(self, tool_name):
        """
        Cập nhật giao diện cài đặt cho công cụ đã chọn.
        
        Parameters
        ----------
        tool_name : str
            Tên công cụ
        """
        # Xóa các widget cũ
        for i in reversed(range(self.tool_settings_layout.count())):
            widget = self.tool_settings_layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()
        
        # Thêm widget cài đặt mới cho công cụ đã chọn
        if self.tool_manager:
            tool = self.tool_manager.get_tool(tool_name)
            if tool and hasattr(tool, 'get_settings_widget'):
                settings_widget = tool.get_settings_widget()
                if settings_widget:
                    self.tool_settings_layout.addWidget(settings_widget)
    
    def _on_add_contour(self):
        """Thêm contour mới."""
        # Tạo tên contour mới
        index = 1
        while f"Contour {index}" in self.contours.get_names():
            index += 1
        
        name = f"Contour {index}"
        
        # Tạo màu ngẫu nhiên
        color = QColor(
            np.random.randint(0, 200),
            np.random.randint(0, 200), 
            np.random.randint(0, 200)
        )
        
        # Thêm contour vào danh sách
        self.add_contour(name, color)
    
    def add_contour(self, name, color):
        """
        Thêm contour vào danh sách.
        
        Parameters
        ----------
        name : str
            Tên contour
        color : QColor
            Màu của contour
        """
        # Kiểm tra nếu tên đã tồn tại
        if name in self.contours.get_names():
            return
        
        # Thêm vào tập hợp contour
        self.contours.add_contour(name, color)
        
        # Thêm vào danh sách hiển thị
        item = QListWidgetItem(name)
        item.setData(Qt.UserRole, name)
        
        # Thiết lập màu hiển thị
        pixmap = QPixmap(16, 16)
        pixmap.fill(color)
        item.setIcon(QIcon(pixmap))
        
        self.contour_list.addItem(item)
        
        # Chọn contour vừa thêm
        self.contour_list.setCurrentItem(item)
        self.active_contour_name = name
        
        # Phát tín hiệu
        self.contour_added.emit(name, color)
    
    def add_contour_from_points(self, name, points, slice_idx, color=None):
        """
        Thêm contour từ danh sách điểm.
        
        Parameters
        ----------
        name : str
            Tên contour
        points : List[Tuple[int, int]]
            Danh sách các điểm (x, y)
        slice_idx : int
            Chỉ số lát cắt
        color : QColor, optional
            Màu của contour
        """
        # Nếu contour chưa tồn tại, tạo mới
        if name not in self.contours.get_names():
            if color is None:
                color = QColor(
                    np.random.randint(0, 200),
                    np.random.randint(0, 200), 
                    np.random.randint(0, 200)
                )
            self.add_contour(name, color)
        
        # Cập nhật contour
        self.contours.add_contour_points(name, points, slice_idx)
    
    def _on_delete_contour(self):
        """Xóa contour đã chọn."""
        item = self.contour_list.currentItem()
        if not item:
            return
        
        name = item.data(Qt.UserRole)
        
        # Xóa khỏi tập hợp contour
        self.contours.remove_contour(name)
        
        # Xóa khỏi danh sách hiển thị
        row = self.contour_list.row(item)
        self.contour_list.takeItem(row)
        
        # Cập nhật contour đang hoạt động
        if self.active_contour_name == name:
            self.active_contour_name = None
            if self.contour_list.count() > 0:
                self.contour_list.setCurrentRow(0)
                new_item = self.contour_list.currentItem()
                if new_item:
                    self.active_contour_name = new_item.data(Qt.UserRole)
        
        # Phát tín hiệu
        self.contour_deleted.emit(name)
    
    def _on_contour_selected(self, item):
        """
        Xử lý khi chọn contour.
        
        Parameters
        ----------
        item : QListWidgetItem
            Item được chọn
        """
        name = item.data(Qt.UserRole)
        self.active_contour_name = name
        
        # Phát tín hiệu
        self.contour_selected.emit(name)
    
    def set_contour_visibility(self, name, visible):
        """
        Thiết lập khả năng hiển thị của contour.
        
        Parameters
        ----------
        name : str
            Tên contour
        visible : bool
            True nếu hiển thị, False nếu ẩn
        """
        contour = self.contours.get_contour(name)
        if contour:
            contour.visible = visible
            
            # Phát tín hiệu
            self.contour_visibility_changed.emit(name, visible)
    
    def set_contour_color(self, name, color):
        """
        Thiết lập màu của contour.
        
        Parameters
        ----------
        name : str
            Tên contour
        color : QColor
            Màu mới
        """
        contour = self.contours.get_contour(name)
        if contour:
            contour.color = color
            
            # Cập nhật icon trong danh sách
            for i in range(self.contour_list.count()):
                item = self.contour_list.item(i)
                if item.data(Qt.UserRole) == name:
                    pixmap = QPixmap(16, 16)
                    pixmap.fill(color)
                    item.setIcon(QIcon(pixmap))
                    break
            
            # Phát tín hiệu
            self.contour_color_changed.emit(name, color)


class ImagingTab(QWidget):
    """Tab hiển thị hình ảnh và công cụ contour cho QuangTPS."""
    
    def __init__(self, parent=None):
        """
        Khởi tạo tab imaging.
        
        Parameters
        ----------
        parent : QWidget
            Widget cha
        """
        super().__init__(parent)
        
        self.dicom_loader = DicomLoader()
        self.current_series = None
        self.contour_tool_manager = ContourToolManager()
        
        self._init_ui()
        self._init_tools()
        self._connect_signals()
    
    def _init_ui(self):
        """Khởi tạo giao diện người dùng."""
        # Layout chính
        main_layout = QHBoxLayout(self)
        
        # Tạo splitter chính
        main_splitter = QSplitter(Qt.Horizontal)
        
        # === Panel trái - Hiển thị và điều khiển hình ảnh ===
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        # Widget hiển thị lát cắt
        self.image_widget = ImageSliceWidget()
        
        # Widget điều khiển hình ảnh
        self.image_control = ImageControlWidget()
        
        # Thêm vào layout trái
        left_layout.addWidget(self.image_widget, 4)
        left_layout.addWidget(self.image_control, 1)
        
        # === Panel phải - Công cụ contour và điều khiển ===
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        # Nút tải DICOM
        load_button = QPushButton("Tải DICOM")
        load_button.clicked.connect(self._on_load_dicom)
        
        # Widget điều khiển contour
        self.contour_control = ContourControlWidget()
        
        # Thêm vào layout phải
        right_layout.addWidget(load_button)
        right_layout.addWidget(self.contour_control)
        
        # Thêm các panel vào splitter
        main_splitter.addWidget(left_panel)
        main_splitter.addWidget(right_panel)
        
        # Thiết lập tỷ lệ khởi tạo
        main_splitter.setSizes([700, 300])
        
        # Thêm vào layout chính
        main_layout.addWidget(main_splitter)
    
    def _init_tools(self):
        """Khởi tạo các công cụ contour."""
        # Thiết lập widget hiển thị cho tool manager
        self.contour_tool_manager.set_image_widget(self.image_widget)
        
        # Thêm các công cụ contour
        self.contour_tool_manager.add_tool(FreehandContourTool())
        self.contour_tool_manager.add_tool(BrushContourTool())
        self.contour_tool_manager.add_tool(CircleContourTool())
        self.contour_tool_manager.add_tool(RectangleContourTool())
        self.contour_tool_manager.add_tool(ThresholdContourTool())
        self.contour_tool_manager.add_tool(AutoThresholdContourTool())
        
        # Thiết lập quản lý công cụ cho widget điều khiển
        self.contour_control.set_tool_manager(self.contour_tool_manager)
        
        # Thiết lập công cụ mặc định
        self.contour_tool_manager.set_active_tool("Freehand")
    
    def _connect_signals(self):
        """Kết nối các tín hiệu."""
        # Kết nối tín hiệu thay đổi công cụ
        self.contour_control.tool_changed.connect(self.contour_tool_manager.set_active_tool)
        
        # Kết nối tín hiệu sự kiện chuột từ image widget đến tool manager
        self.image_widget.mouse_pressed.connect(self.contour_tool_manager.mouse_press)
        self.image_widget.mouse_moved.connect(self.contour_tool_manager.mouse_move)
        self.image_widget.mouse_released.connect(self.contour_tool_manager.mouse_release)
        self.image_widget.key_pressed.connect(self.contour_tool_manager.key_press)
        self.image_widget.key_released.connect(self.contour_tool_manager.key_release)
        
        # Kết nối tín hiệu từ các công cụ contour
        for tool_name in ["Freehand", "Brush", "Circle", "Rectangle", "Threshold", "AutoThreshold"]:
            tool = self.contour_tool_manager.get_tool(tool_name)
            if tool:
                tool.contour_created.connect(self.contour_control.add_contour_from_points)
                tool.contour_updated.connect(self.contour_control.add_contour_from_points)
        
        # Kết nối tín hiệu điều khiển hình ảnh
        self.image_control.brightness_changed.connect(self.image_widget.set_brightness)
        self.image_control.contrast_changed.connect(self.image_widget.set_contrast)
        self.image_control.slice_changed.connect(self.image_widget.set_slice_index)
    
    def _on_load_dicom(self):
        """Xử lý khi tải DICOM."""
        # Hiển thị hộp thoại chọn thư mục
        dicom_dir = QFileDialog.getExistingDirectory(
            self, "Chọn thư mục DICOM", QDir.homePath(),
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )
        
        if not dicom_dir:
            return
        
        try:
            # Tải các series DICOM
            series_list = self.dicom_loader.load_directory(dicom_dir)
            
            if not series_list:
                QMessageBox.warning(self, "Lỗi", "Không tìm thấy dữ liệu DICOM trong thư mục đã chọn.")
                return
            
            # Sử dụng series đầu tiên
            self.current_series = series_list[0]
            
            # Hiển thị series
            self._display_current_series()
            
        except Exception as e:
            QMessageBox.critical(self, "Lỗi", f"Lỗi khi tải DICOM: {str(e)}")
            logger.exception("Lỗi khi tải DICOM")
    
    def _display_current_series(self):
        """Hiển thị series DICOM hiện tại."""
        if not self.current_series:
            return
        
        # Thiết lập dữ liệu hình ảnh cho widget hiển thị
        self.image_widget.set_image_data(self.current_series.pixel_data)
        
        # Thiết lập thông tin slice cho widget điều khiển
        self.image_control.set_slice_count(self.current_series.num_slices)
        
        # Cập nhật thông tin series
        self._update_series_info()
    
    def _update_series_info(self):
        """Cập nhật thông tin series DICOM."""
        if not self.current_series:
            return
        
        # Trong phiên bản thực tế, bạn có thể hiển thị thông tin series
        # như ModalityDescription, PatientName, v.v.
        pass
    
    def set_image_data(self, image_data, slice_info=None):
        """
        Thiết lập dữ liệu hình ảnh từ bên ngoài.
        
        Parameters
        ----------
        image_data : np.ndarray
            Dữ liệu hình ảnh 3D
        slice_info : Dict
            Thông tin bổ sung về lát cắt
        """
        # Thiết lập dữ liệu hình ảnh cho widget hiển thị
        self.image_widget.set_image_data(image_data)
        
        # Thiết lập thông tin slice cho widget điều khiển
        if image_data is not None:
            self.image_control.set_slice_count(image_data.shape[0])
        
        # Các xử lý khác nếu cần
    
    def get_contours(self):
        """
        Lấy tất cả các contour.
        
        Returns
        -------
        ContourCollection
            Tập hợp các contour
        """
        return self.contour_control.contours
    
    def get_active_contour_name(self):
        """
        Lấy tên contour đang hoạt động.
        
        Returns
        -------
        str
            Tên contour hoặc None nếu không có
        """
        return self.contour_control.active_contour_name
    
    def get_current_slice_index(self):
        """
        Lấy chỉ số lát cắt hiện tại.
        
        Returns
        -------
        int
            Chỉ số lát cắt
        """
        return self.image_widget.slice_idx if self.image_widget else 0
