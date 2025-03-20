#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module hiển thị 3D cho dữ liệu hình ảnh y tế trong QuangTPS.

Module này cung cấp các lớp và chức năng để hiển thị dữ liệu hình ảnh 3D như CT, MRI, CBCT,
và các contour cấu trúc trong không gian 3D sử dụng VTK.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any

# VTK imports
try:
    import vtk
    from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
    VTK_AVAILABLE = True
except ImportError:
    VTK_AVAILABLE = False
    
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QPushButton, QComboBox, QCheckBox, QSlider,
    QGroupBox, QFormLayout, QColorDialog, QSpinBox,
    QDoubleSpinBox, QMessageBox, QFrame, QSplitter
)
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot, QSize
from PyQt5.QtGui import QColor

logger = logging.getLogger(__name__)


class VolumeRenderingWidget(QWidget):
    """Widget để hiển thị dữ liệu hình ảnh 3D và các contour trong không gian 3D."""
    
    def __init__(self, parent=None):
        """
        Khởi tạo widget hiển thị khối 3D.
        
        Parameters
        ----------
        parent : QWidget, optional
            Widget cha
        """
        super().__init__(parent)
        
        self.vtk_widget = None
        self.renderer = None
        self.render_window = None
        self.interactor = None
        
        self.volume_actor = None
        self.contour_actors = {}
        self.dose_actor = None
        
        self.image_data = None
        self.spacing = (1.0, 1.0, 1.0)
        self.origin = (0.0, 0.0, 0.0)
        
        self.presets = {
            "CT-Bones": {"color": [(0.0, 0.0, 0.0, 0.0), 
                                 (0.5, 0.9, 0.9, 0.9), 
                                 (1.0, 1.0, 1.0, 1.0)],
                       "opacity": [(0.0, 0.0), 
                                 (0.8, 0.0), 
                                 (0.9, 0.15), 
                                 (1.0, 0.3)],
                       "window": [400, 1500]},
            "CT-Soft Tissue": {"color": [(0.0, 0.0, 0.0, 0.0), 
                                       (0.7, 0.5, 0.25, 0.125), 
                                       (1.0, 1.0, 0.9, 0.8)],
                             "opacity": [(0.0, 0.0), 
                                       (0.55, 0.0), 
                                       (0.7, 0.2), 
                                       (1.0, 0.8)],
                             "window": [50, 400]},
            "CT-Lungs": {"color": [(0.0, 0.0, 0.0, 0.0), 
                                 (0.5, 0.3, 0.3, 0.3), 
                                 (1.0, 1.0, 1.0, 1.0)],
                       "opacity": [(0.0, 0.0), 
                                 (0.15, 0.0), 
                                 (0.3, 0.1), 
                                 (1.0, 0.5)],
                       "window": [-400, 400]},
            "MRI": {"color": [(0.0, 0.0, 0.0, 0.0), 
                           (0.5, 0.5, 0.5, 0.5), 
                           (1.0, 1.0, 1.0, 1.0)],
                 "opacity": [(0.0, 0.0), 
                           (0.2, 0.0), 
                           (0.4, 0.3), 
                           (1.0, 0.8)],
                 "window": [50, 300]}
        }
        
        # Màu cho các contour
        self.contour_default_colors = {
            "PTV": (1.0, 0.0, 0.0),      # Đỏ
            "CTV": (1.0, 0.5, 0.0),      # Cam
            "GTV": (1.0, 1.0, 0.0),      # Vàng
            "OAR": (0.0, 1.0, 0.0),      # Xanh lá
            "Body": (0.0, 0.0, 1.0),     # Xanh dương
            "Spinal Cord": (1.0, 0.0, 1.0),  # Tím
            "Lung": (0.0, 1.0, 1.0),     # Xanh ngọc
            "Heart": (0.8, 0.2, 0.2),    # Đỏ sẫm
            "Brain": (0.5, 0.5, 0.5)     # Xám
        }
        
        # Kiểm tra VTK
        if not VTK_AVAILABLE:
            logger.error("VTK không khả dụng. Vui lòng cài đặt VTK để sử dụng tính năng hiển thị 3D.")
            self._init_error_ui()
        else:
            self._init_ui()
    
    def _init_error_ui(self):
        """Khởi tạo giao diện lỗi khi không có VTK."""
        layout = QVBoxLayout(self)
        
        error_label = QLabel(
            "Không thể tải thư viện VTK. Vui lòng cài đặt VTK để sử dụng tính năng hiển thị 3D." 
            "\n\nCài đặt bằng lệnh: pip install vtk"
        )
        error_label.setStyleSheet("color: red;")
        error_label.setAlignment(Qt.AlignCenter)
        
        layout.addWidget(error_label)
    
    def _init_ui(self):
        """Khởi tạo giao diện người dùng."""
        if not VTK_AVAILABLE:
            self._init_error_ui()
            return
            
        # Layout chính
        main_layout = QHBoxLayout(self)
        
        # VTK widget
        self.vtk_widget = QVTKRenderWindowInteractor(self)
        
        # Điều khiển hiển thị
        control_widget = QWidget()
        control_layout = QVBoxLayout(control_widget)
        
        # Nhóm điều khiển hiển thị khối
        volume_group = QGroupBox("Điều khiển hiển thị khối")
        volume_layout = QFormLayout(volume_group)
        
        # Chọn preset
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(self.presets.keys())
        self.preset_combo.currentIndexChanged.connect(self._preset_changed)
        volume_layout.addRow("Preset:", self.preset_combo)
        
        # Điều khiển độ trong suốt
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(50)
        self.opacity_slider.valueChanged.connect(self._opacity_changed)
        volume_layout.addRow("Độ trong suốt:", self.opacity_slider)
        
        # Cửa sổ hiển thị
        self.window_level_spin = QSpinBox()
        self.window_level_spin.setRange(-1000, 3000)
        self.window_level_spin.setValue(50)
        self.window_level_spin.valueChanged.connect(self._window_level_changed)
        volume_layout.addRow("Mức cửa sổ:", self.window_level_spin)
        
        self.window_width_spin = QSpinBox()
        self.window_width_spin.setRange(1, 4000)
        self.window_width_spin.setValue(400)
        self.window_width_spin.valueChanged.connect(self._window_width_changed)
        volume_layout.addRow("Độ rộng cửa sổ:", self.window_width_spin)
        
        # Hiển thị contour
        contour_group = QGroupBox("Contour")
        contour_layout = QVBoxLayout(contour_group)
        
        self.contour_list = QComboBox()
        self.contour_list.addItem("Tất cả")
        contour_layout.addWidget(self.contour_list)
        
        contour_options = QHBoxLayout()
        
        self.contour_opacity_slider = QSlider(Qt.Horizontal)
        self.contour_opacity_slider.setRange(0, 100)
        self.contour_opacity_slider.setValue(70)
        self.contour_opacity_slider.valueChanged.connect(self._contour_opacity_changed)
        contour_options.addWidget(QLabel("Độ trong suốt:"))
        contour_options.addWidget(self.contour_opacity_slider)
        
        self.contour_color_btn = QPushButton("Màu")
        self.contour_color_btn.clicked.connect(self._contour_color_picker)
        contour_options.addWidget(self.contour_color_btn)
        
        contour_layout.addLayout(contour_options)
        
        # Các nút điều khiển
        buttons_layout = QHBoxLayout()
        
        self.reset_view_btn = QPushButton("Đặt lại góc nhìn")
        self.reset_view_btn.clicked.connect(self._reset_view)
        buttons_layout.addWidget(self.reset_view_btn)
        
        self.capture_image_btn = QPushButton("Chụp ảnh")
        self.capture_image_btn.clicked.connect(self._capture_view)
        buttons_layout.addWidget(self.capture_image_btn)
        
        # Thêm các nhóm vào layout điều khiển
        control_layout.addWidget(volume_group)
        control_layout.addWidget(contour_group)
        control_layout.addLayout(buttons_layout)
        control_layout.addStretch()
        
        # Thiết lập tỷ lệ kích thước
        main_layout.addWidget(self.vtk_widget, 4)
        main_layout.addWidget(control_widget, 1)
        
        # Thiết lập VTK renderer
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.1, 0.1, 0.1)
        
        self.render_window = self.vtk_widget.GetRenderWindow()
        self.render_window.AddRenderer(self.renderer)
        
        self.interactor = self.render_window.GetInteractor()
        self.interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
        self.interactor.Initialize()