#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module hiển thị hình ảnh cho QuangTPS.

Module này cung cấp các thành phần giao diện để hiển thị và thao tác với 
hình ảnh y tế từ nhiều phương thức chụp như CT, MRI, PET... với khả năng
hiển thị đa mặt phẳng, 3D, và các công cụ đo lường, phân tích.
"""

import os
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QSlider,
    QComboBox, QCheckBox, QGroupBox, QToolButton, QMenu, QAction,
    QSplitter, QTabWidget, QToolBar, QSpinBox, QDoubleSpinBox,
    QScrollArea, QFrame, QSizePolicy, QFormLayout, QRadioButton
)
from PyQt5.QtCore import Qt, QSize, pyqtSignal, pyqtSlot, QRectF, QPoint
from PyQt5.QtGui import (
    QImage, QPixmap, QPainter, QColor, QPen, QBrush, QFont, QTransform,
    QMouseEvent, QWheelEvent, QKeyEvent
)

from quangtps.core.logging import get_logger
from quangtps.imaging.image import Image
from quangtps.imaging.structures import Structure, StructureSet
from quangtps.ui.image_display import ImageDisplay

logger = get_logger(__name__)

class ImageViewer(QWidget):
    """
    Widget chính để hiển thị và thao tác với hình ảnh y tế.
    
    Cung cấp chức năng hiển thị đa mặt phẳng, công cụ đo lường, 
    điều chỉnh cửa sổ, chồng hình, và hiển thị cấu trúc.
    """
    
    # Tín hiệu
    position_changed = pyqtSignal(int, int, int)  # x, y, z coordinates
    window_level_changed = pyqtSignal(int, int)   # window width, window level
    
    def __init__(self, parent=None):
        """Khởi tạo ImageViewer."""
        super().__init__(parent)
        
        # Dữ liệu hình ảnh
        self.primary_image = None  # Hình ảnh chính (CT, MRI...)
        self.secondary_images = []  # Hình ảnh phụ (PET, MRI...)
        self.structure_set = None   # Tập hợp cấu trúc
        self.dose_grid = None       # Lưới liều
        
        # Trạng thái hiển thị
        self.view_mode = "MPR"  # MPR (Multi-Planar Reconstruction) hoặc 3D
        self.window_width = 500
        self.window_level = 40
        self.overlay_opacity = 0.7
        self.current_position = [0, 0, 0]  # Vị trí hiện tại trong không gian 3D
        self.current_tool = "pan"  # Công cụ hiện tại (pan, zoom, window, measure...)
        
        # Khởi tạo giao diện
        self._init_ui()
    
    def _init_ui(self):
        """Khởi tạo giao diện người dùng."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Thanh công cụ chính
        self.toolbar = QToolBar("Công cụ hình ảnh")
        main_layout.addWidget(self.toolbar)
        
        # Nút chọn chế độ xem
        view_mode_combo = QComboBox()
        view_mode_combo.addItems(["MPR", "3D"])
        view_mode_combo.setCurrentText(self.view_mode)
        view_mode_combo.currentTextChanged.connect(self._change_view_mode)
        self.toolbar.addWidget(QLabel("Chế độ xem:"))
        self.toolbar.addWidget(view_mode_combo)
        self.toolbar.addSeparator()
        
        # Các nút công cụ
        tools = [
            ("Pan", "pan", "Di chuyển hình ảnh"),
            ("Zoom", "zoom", "Phóng to/thu nhỏ"),
            ("Window", "window", "Điều chỉnh cửa sổ"),
            ("Measure", "measure", "Đo khoảng cách"),
            ("Angle", "angle", "Đo góc"),
            ("Annotation", "annotation", "Chú thích")
        ]
        
        self.tool_buttons = {}
        for name, tool_id, tooltip in tools:
            btn = QToolButton()
            btn.setText(name)
            btn.setCheckable(True)
            btn.setToolTip(tooltip)
            btn.clicked.connect(lambda checked, t=tool_id: self._set_current_tool(t))
            self.toolbar.addWidget(btn)
            self.tool_buttons[tool_id] = btn
            
        # Chọn nút Pan mặc định
        self.tool_buttons["pan"].setChecked(True)
        
        self.toolbar.addSeparator()
        
        # Điều khiển cửa sổ
        self.toolbar.addWidget(QLabel("WW:"))
        self.ww_spin = QSpinBox()
        self.ww_spin.setRange(1, 4000)
        self.ww_spin.setValue(self.window_width)
        self.ww_spin.valueChanged.connect(self._on_window_width_changed)
        self.toolbar.addWidget(self.ww_spin)
        
        self.toolbar.addWidget(QLabel("WL:"))
        self.wl_spin = QSpinBox()
        self.wl_spin.setRange(-1000, 3000)
        self.wl_spin.setValue(self.window_level)
        self.wl_spin.valueChanged.connect(self._on_window_level_changed)
        self.toolbar.addWidget(self.wl_spin)
        
        # Preset cửa sổ
        preset_btn = QToolButton()
        preset_btn.setText("Presets")
        preset_menu = QMenu()
        
        presets = [
            ("Brain", 80, 40),
            ("Lung", 1500, -600),
            ("Abdomen", 400, 50),
            ("Bone", 2000, 500),
            ("Mediastinum", 350, 50)
        ]
        
        for name, ww, wl in presets:
            action = QAction(name, self)
            action.triggered.connect(lambda checked, w=ww, l=wl: self._set_window(w, l))
            preset_menu.addAction(action)
            
        preset_btn.setMenu(preset_menu)
        preset_btn.setPopupMode(QToolButton.InstantPopup)
        self.toolbar.addWidget(preset_btn)
        
        # Khu vực hiển thị chính
        self.splitter = QSplitter(Qt.Vertical)
        main_layout.addWidget(self.splitter, 1)
        
        # Khu vực hiển thị MPR
        self.mpr_widget = QWidget()
        self.mpr_layout = QHBoxLayout(self.mpr_widget)
        self.mpr_layout.setContentsMargins(0, 0, 0, 0)
        
        # Tạo 3 màn hình hiển thị cho 3 mặt phẳng: Axial, Sagittal, Coronal
        self.axial_view = ImageDisplay()
        self.axial_view.setMinimumSize(300, 300)
        self.axial_view.set_title("Axial")
        
        self.sagittal_view = ImageDisplay()
        self.sagittal_view.setMinimumSize(300, 300)
        self.sagittal_view.set_title("Sagittal")
        
        self.coronal_view = ImageDisplay()
        self.coronal_view.setMinimumSize(300, 300)
        self.coronal_view.set_title("Coronal")
        
        # Thêm các màn hình vào layout
        self.mpr_layout.addWidget(self.axial_view)
        self.mpr_layout.addWidget(self.sagittal_view)
        self.mpr_layout.addWidget(self.coronal_view)
        
        # Khu vực hiển thị 3D
        self.view_3d = QWidget()
        self.view_3d_layout = QVBoxLayout(self.view_3d)
        self.view_3d_layout.addWidget(QLabel("Hiển thị 3D (Chưa triển khai)"))
        
        # Thêm các widget vào splitter
        self.splitter.addWidget(self.mpr_widget)
        
        # Thanh trượt và thông tin
        info_widget = QWidget()
        info_layout = QHBoxLayout(info_widget)
        
        # Thanh trượt lát cắt
        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.setRange(0, 100)
        self.slice_slider.setValue(50)
        self.slice_slider.valueChanged.connect(self._on_slice_changed)
        
        # Label thông tin
        self.info_label = QLabel("Slice: 50/100 | Pos: (0, 0, 0) | HU: -")
        
        info_layout.addWidget(QLabel("Lát cắt:"))
        info_layout.addWidget(self.slice_slider, 1)
        info_layout.addWidget(self.info_label)
        
        main_layout.addWidget(info_widget)
        
        # Hiển thị chế độ MPR mặc định
        self._update_view_mode()
        
        # Kết nối sự kiện giữa các chế độ xem
        self.axial_view.mouse_position_changed.connect(self._update_position_info)
        self.sagittal_view.mouse_position_changed.connect(self._update_position_info)
        self.coronal_view.mouse_position_changed.connect(self._update_position_info)
    
    def load_image(self, image: Image):
        """
        Tải một hình ảnh y tế vào trình xem.
        
        Args:
            image: Đối tượng Image chứa dữ liệu hình ảnh y tế
        """
        self.primary_image = image
        
        # Cập nhật thanh trượt lát cắt
        if image and image.data is not None:
            depth = image.data.shape[0]
            self.slice_slider.setRange(0, depth - 1)
            self.slice_slider.setValue(depth // 2)
            
            # Thiết lập cửa sổ mặc định dựa trên hình ảnh
            if image.modality == "CT":
                self._set_window(1500, 500)  # CT default
            elif image.modality == "MR":
                self._set_window(1000, 500)  # MR default
            elif image.modality == "PT":
                self._set_window(25000, 12500)  # PET default
            
            # Cập nhật hiển thị
            self._update_displays()
        
    def load_secondary_image(self, image: Image):
        """
        Tải một hình ảnh thứ cấp để chồng hình.
        
        Args:
            image: Đối tượng Image thứ cấp (PET, MRI...)
        """
        self.secondary_images.append(image)
        self._update_displays()
    
    def load_structure_set(self, structure_set: StructureSet):
        """
        Tải một tập hợp cấu trúc để hiển thị contour.
        
        Args:
            structure_set: Đối tượng StructureSet
        """
        self.structure_set = structure_set
        self._update_displays()
    
    def load_dose_grid(self, dose_grid):
        """
        Tải lưới liều để hiển thị phân bố liều.
        
        Args:
            dose_grid: Đối tượng DoseGrid
        """
        self.dose_grid = dose_grid
        self._update_displays()
    
    def _change_view_mode(self, mode: str):
        """
        Thay đổi chế độ xem giữa MPR và 3D.
        
        Args:
            mode: Chế độ xem mới (MPR hoặc 3D)
        """
        self.view_mode = mode
        self._update_view_mode()
    
    def _update_view_mode(self):
        """Cập nhật giao diện dựa trên chế độ xem hiện tại."""
        # Xóa các widget hiện tại khỏi splitter
        while self.splitter.count() > 0:
            self.splitter.widget(0).setParent(None)
        
        # Thêm widget phù hợp với chế độ xem
        if self.view_mode == "MPR":
            self.splitter.addWidget(self.mpr_widget)
        else:  # 3D mode
            self.splitter.addWidget(self.view_3d)
    
    def _set_current_tool(self, tool: str):
        """
        Thiết lập công cụ hiện tại.
        
        Args:
            tool: ID của công cụ (pan, zoom, window...)
        """
        self.current_tool = tool
        
        # Cập nhật trạng thái nút
        for tool_id, button in self.tool_buttons.items():
            button.setChecked(tool_id == tool)
        
        # Cập nhật chế độ công cụ cho các màn hình hiển thị
        self.axial_view.set_tool_mode(tool)
        self.sagittal_view.set_tool_mode(tool)
        self.coronal_view.set_tool_mode(tool)
    
    def _on_window_width_changed(self, value: int):
        """
        Xử lý sự kiện khi giá trị độ rộng cửa sổ thay đổi.
        
        Args:
            value: Giá trị độ rộng cửa sổ mới
        """
        self.window_width = value
        self._update_window_level()
    
    def _on_window_level_changed(self, value: int):
        """
        Xử lý sự kiện khi giá trị mức cửa sổ thay đổi.
        
        Args:
            value: Giá trị mức cửa sổ mới
        """
        self.window_level = value
        self._update_window_level()
    
    def _set_window(self, width: int, level: int):
        """
        Thiết lập cả độ rộng và mức cửa sổ cùng lúc.
        
        Args:
            width: Độ rộng cửa sổ mới
            level: Mức cửa sổ mới
        """
        self.window_width = width
        self.window_level = level
        
        # Cập nhật giá trị trong spin box
        self.ww_spin.setValue(width)
        self.wl_spin.setValue(level)
        
        self._update_window_level()
    
    def _update_window_level(self):
        """Cập nhật cửa sổ cho tất cả các màn hình."""
        self.axial_view.set_window(self.window_width, self.window_level)
        self.sagittal_view.set_window(self.window_width, self.window_level)
        self.coronal_view.set_window(self.window_width, self.window_level)
        
        # Phát tín hiệu
        self.window_level_changed.emit(self.window_width, self.window_level)
    
    def _on_slice_changed(self, value: int):
        """
        Xử lý sự kiện khi vị trí lát cắt thay đổi.
        
        Args:
            value: Chỉ số lát cắt mới
        """
        if not self.primary_image or self.primary_image.data is None:
            return
            
        # Cập nhật vị trí hiện tại
        self.current_position[2] = value
        
        # Cập nhật hiển thị
        self._update_displays()
        
        # Cập nhật thông tin hiển thị
        self._update_info_label()
    
    def _update_displays(self):
        """Cập nhật tất cả các màn hình hiển thị."""
        if not self.primary_image or self.primary_image.data is None:
            return
            
        # Lấy dữ liệu hình ảnh và vị trí hiện tại
        image_data = self.primary_image.data
        x, y, z = self.current_position
        
        # Giới hạn vị trí trong phạm vi hình ảnh
        z = min(max(0, z), image_data.shape[0] - 1)
        x = min(max(0, x), image_data.shape[1] - 1)
        y = min(max(0, y), image_data.shape[2] - 1)
        
        # Cập nhật vị trí hiện tại
        self.current_position = [x, y, z]
        
        # Hiển thị lát cắt Axial (z)
        axial_slice = image_data[z, :, :]
        self.axial_view.set_image(axial_slice)
        self.axial_view.set_crosshair(x, y)
        
        # Hiển thị lát cắt Sagittal (x)
        if x < image_data.shape[1]:
            sagittal_slice = image_data[:, x, :]
            self.sagittal_view.set_image(np.flipud(sagittal_slice.T))
            self.sagittal_view.set_crosshair(z, image_data.shape[2] - y - 1)
        
        # Hiển thị lát cắt Coronal (y)
        if y < image_data.shape[2]:
            coronal_slice = image_data[:, :, y]
            self.coronal_view.set_image(np.flipud(coronal_slice.T))
            self.coronal_view.set_crosshair(z, image_data.shape[1] - x - 1)
        
        # Hiển thị cấu trúc nếu có
        if self.structure_set is not None:
            # TODO: Hiển thị contour trên các lát cắt
            pass
        
        # Hiển thị liều nếu có
        if self.dose_grid is not None:
            # TODO: Hiển thị phân bố liều
            pass
        
        # Cập nhật cửa sổ
        self._update_window_level()
    
    def _update_position_info(self, x: int, y: int, value: float):
        """
        Cập nhật thông tin vị trí chuột và giá trị HU.
        
        Args:
            x: Tọa độ x trên màn hình hiển thị
            y: Tọa độ y trên màn hình hiển thị
            value: Giá trị HU tại vị trí chuột
        """
        # Cập nhật nhãn thông tin
        self.info_label.setText(f"Pos: ({x}, {y}) | HU: {value:.1f}")
    
    def _update_info_label(self):
        """Cập nhật nhãn thông tin với vị trí hiện tại."""
        if not self.primary_image or self.primary_image.data is None:
            return
            
        x, y, z = self.current_position
        depth = self.primary_image.data.shape[0]
        
        # Lấy giá trị HU tại vị trí hiện tại nếu nằm trong phạm vi
        try:
            hu_value = self.primary_image.data[z, x, y]
        except:
            hu_value = "-"
            
        self.info_label.setText(f"Slice: {z+1}/{depth} | Pos: ({x}, {y}, {z}) | HU: {hu_value}")
        
        # Phát tín hiệu thay đổi vị trí
        self.position_changed.emit(x, y, z)
