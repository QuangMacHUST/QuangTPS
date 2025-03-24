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
    from vtk.util import numpy_support
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

    def _preset_changed(self, index):
        """Xử lý khi thay đổi preset hiển thị khối."""
        preset_name = self.preset_combo.currentText()
        if preset_name in self.presets and self.volume_property is not None:
            # Áp dụng preset
            self.apply_preset(preset_name)
            self._update_render()
    
    def _opacity_changed(self, value):
        """Xử lý khi thay đổi độ trong suốt."""
        if self.volume_property is not None:
            opacity = value / 100.0
            # Cập nhật hàm opacity
            self._update_opacity(opacity)
            self._update_render()
    
    def _window_level_changed(self, value):
        """Xử lý khi thay đổi mức cửa sổ."""
        if self.volume_property is not None:
            self.window_level = value
            self._update_transfer_function()
            self._update_render()
    
    def _window_width_changed(self, value):
        """Xử lý khi thay đổi độ rộng cửa sổ."""
        if self.volume_property is not None:
            self.window_width = value
            self._update_transfer_function()
            self._update_render()
    
    def _update_render(self):
        """Cập nhật hiển thị."""
        if self.renderer:
            self.render_window.Render()
    
    def _update_opacity(self, opacity):
        """Cập nhật độ trong suốt cho volume rendering."""
        if not self.volume_property:
            return
            
        # Lấy hàm opacity hiện tại
        opacity_transfer_function = self.volume_property.GetScalarOpacity()
        
        # Điều chỉnh điểm kiểm soát
        points = []
        for i in range(opacity_transfer_function.GetSize()):
            val = opacity_transfer_function.GetValue(i)
            pos = opacity_transfer_function.GetPoint(i)[0]
            # Điều chỉnh giá trị opacity mới dựa trên opactiy tổng thể
            new_val = val * opacity
            points.append((pos, new_val))
        
        # Xóa tất cả điểm hiện tại
        opacity_transfer_function.RemoveAllPoints()
        
        # Thêm lại các điểm với giá trị opacity mới
        for pos, val in points:
            opacity_transfer_function.AddPoint(pos, val)
    
    def _update_transfer_function(self):
        """Cập nhật hàm chuyển đổi dựa trên window level và width."""
        if not self.volume_property or not hasattr(self, 'window_level') or not hasattr(self, 'window_width'):
            return
            
        # Tính toán range dựa trên window level và width
        min_val = self.window_level - self.window_width / 2
        max_val = self.window_level + self.window_width / 2
        
        # Lấy color transfer function
        color_function = self.volume_property.GetRGBTransferFunction()
        
        # Xóa tất cả điểm
        color_function.RemoveAllPoints()
        
        # Thêm các điểm màu mới dựa trên range mới
        color_function.AddRGBPoint(min_val, 0.0, 0.0, 0.0)  # Đen cho giá trị nhỏ nhất
        color_function.AddRGBPoint((min_val + max_val) / 2, 0.5, 0.5, 0.5)  # Xám cho giá trị trung bình
        color_function.AddRGBPoint(max_val, 1.0, 1.0, 1.0)  # Trắng cho giá trị lớn nhất
    
    def apply_preset(self, preset_name):
        """Áp dụng preset được chọn."""
        if preset_name not in self.presets or not self.volume_property:
            return
            
        preset = self.presets[preset_name]
        
        # Cập nhật window level và width từ preset
        if "window" in preset:
            self.window_level = preset["window"][0]
            self.window_width = preset["window"][1]
            
            # Cập nhật giá trị trên UI
            self.window_level_spin.blockSignals(True)
            self.window_level_spin.setValue(self.window_level)
            self.window_level_spin.blockSignals(False)
            
            self.window_width_spin.blockSignals(True)
            self.window_width_spin.setValue(self.window_width)
            self.window_width_spin.blockSignals(False)
        
        # Cập nhật color function
        if "color" in preset:
            color_function = self.volume_property.GetRGBTransferFunction()
            color_function.RemoveAllPoints()
            
            # Thêm các điểm màu từ preset
            min_val = self.window_level - self.window_width / 2
            max_val = self.window_level + self.window_width / 2
            range_size = max_val - min_val
            
            for point in preset["color"]:
                relative_pos, r, g, b = point
                absolute_pos = min_val + relative_pos * range_size
                color_function.AddRGBPoint(absolute_pos, r, g, b)
        
        # Cập nhật opacity function
        if "opacity" in preset:
            opacity_function = self.volume_property.GetScalarOpacity()
            opacity_function.RemoveAllPoints()
            
            # Thêm các điểm opacity từ preset
            for point in preset["opacity"]:
                relative_pos, opacity = point
                absolute_pos = min_val + relative_pos * range_size
                opacity_function.AddPoint(absolute_pos, opacity)
        
        # Cập nhật hiển thị
        self._update_render()

    def _contour_opacity_changed(self, value):
        """Xử lý khi thay đổi độ trong suốt của contour."""
        if not self.contour_actors:
            return
            
        opacity = value / 100.0
        selected = self.contour_list.currentText()
        
        if selected == "Tất cả":
            # Áp dụng cho tất cả các contour
            for actor in self.contour_actors.values():
                actor.GetProperty().SetOpacity(opacity)
        else:
            # Áp dụng cho contour được chọn
            if selected in self.contour_actors:
                self.contour_actors[selected].GetProperty().SetOpacity(opacity)
                
        self._update_render()
    
    def _contour_color_picker(self):
        """Mở hộp thoại chọn màu cho contour."""
        selected = self.contour_list.currentText()
        if selected == "Tất cả" or selected not in self.contour_actors:
            return
            
        # Lấy màu hiện tại
        current_color = self.contour_actors[selected].GetProperty().GetColor()
        initial_color = QColor(
            int(current_color[0] * 255),
            int(current_color[1] * 255),
            int(current_color[2] * 255)
        )
        
        # Mở hộp thoại chọn màu
        color = QColorDialog.getColor(initial_color, self, f"Chọn màu cho {selected}")
        
        if color.isValid():
            # Áp dụng màu mới
            self.contour_actors[selected].GetProperty().SetColor(
                color.red() / 255.0,
                color.green() / 255.0,
                color.blue() / 255.0
            )
            self._update_render()
    
    def _reset_view(self):
        """Đặt lại góc nhìn về trạng thái ban đầu."""
        if self.renderer:
            self.renderer.ResetCamera()
            self._update_render()
    
    def _capture_view(self):
        """Chụp ảnh hiện tại của cảnh 3D."""
        if not self.render_window:
            return
            
        # Tạo đối tượng chụp ảnh
        window_to_image_filter = vtk.vtkWindowToImageFilter()
        window_to_image_filter.SetInput(self.render_window)
        window_to_image_filter.SetInputBufferTypeToRGB()
        window_to_image_filter.ReadFrontBufferOff()
        window_to_image_filter.Update()
        
        # Lưu ảnh dưới dạng PNG
        import tempfile
        import os
        from PyQt5.QtWidgets import QFileDialog
        
        # Mở hộp thoại lưu file
        default_path = os.path.join(tempfile.gettempdir(), "quangtps_volume_render.png")
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Lưu ảnh", default_path, "PNG Files (*.png);;JPEG Files (*.jpg)"
        )
        
        if file_path:
            # Xác định định dạng dựa trên phần mở rộng
            if file_path.lower().endswith(".jpg") or file_path.lower().endswith(".jpeg"):
                writer = vtk.vtkJPEGWriter()
            else:
                writer = vtk.vtkPNGWriter()
                if not file_path.lower().endswith(".png"):
                    file_path += ".png"
            
            writer.SetFileName(file_path)
            writer.SetInputConnection(window_to_image_filter.GetOutputPort())
            writer.Write()
            
            logger.info(f"Đã lưu ảnh hiển thị 3D vào: {file_path}")

    def load_volume_data(self, image, window_level=None, window_width=None, preset=None):
        """
        Tải dữ liệu hình ảnh và hiển thị trong cửa sổ 3D.
        
        Parameters
        ----------
        image : Image hoặc ndarray
            Dữ liệu hình ảnh cần hiển thị
        window_level : int, optional
            Mức cửa sổ để hiển thị
        window_width : int, optional
            Độ rộng cửa sổ để hiển thị
        preset : str, optional
            Preset hiển thị mặc định
        """
        if not VTK_AVAILABLE or not self.renderer:
            logger.error("VTK không khả dụng hoặc renderer chưa được khởi tạo")
            return False
            
        try:
            # Xử lý dữ liệu đầu vào
            if hasattr(image, 'data') and hasattr(image, 'spacing') and hasattr(image, 'origin'):
                # Nếu là đối tượng Image
                data = image.data
                self.spacing = image.spacing
                self.origin = image.origin
                direction = image.direction if hasattr(image, 'direction') else None
            else:
                # Nếu là ndarray
                data = image
                
            # Tạo VTK image data
            vtk_data = vtk.vtkImageData()
            vtk_data.SetDimensions(data.shape[2], data.shape[1], data.shape[0])
            vtk_data.SetSpacing(self.spacing)
            vtk_data.SetOrigin(self.origin)
            
            # Set direction matrix if available
            if direction is not None:
                if hasattr(vtk_data, 'SetDirectionMatrix'):
                    matrix = vtk.vtkMatrix4x4()
                    for i in range(3):
                        for j in range(3):
                            matrix.SetElement(i, j, direction[i, j])
                    vtk_data.SetDirectionMatrix(matrix)
            
            # Chuyển đổi dữ liệu numpy thành VTK
            flat_data = data.ravel(order='F').astype('float32')
            vtk_array = numpy_support.numpy_to_vtk(flat_data, deep=True)
            
            # Gán dữ liệu vào VTK image
            vtk_data.GetPointData().SetScalars(vtk_array)
            
            # Lưu trữ data
            self.image_data = vtk_data
            
            # Tạo volume mapper
            volume_mapper = vtk.vtkGPUVolumeRayCastMapper()
            volume_mapper.SetInputData(vtk_data)
            
            # Tạo volume property
            self.volume_property = vtk.vtkVolumeProperty()
            self.volume_property.ShadeOn()
            self.volume_property.SetInterpolationTypeToLinear()
            
            # Tạo color và opacity transfer functions
            color_function = vtk.vtkColorTransferFunction()
            opacity_function = vtk.vtkPiecewiseFunction()
            
            # Áp dụng preset mặc định nếu có
            if preset and preset in self.presets:
                self.preset_combo.setCurrentText(preset)
            else:
                # Áp dụng preset mặc định dựa trên dữ liệu
                # Nếu data có giá trị trong khoảng HU của CT, sử dụng preset CT
                if data.min() < -500 and data.max() > 500:
                    self.preset_combo.setCurrentText("CT-Soft Tissue")
                else:
                    self.preset_combo.setCurrentText("MRI")
            
            # Thiết lập window level và width
            if window_level is not None and window_width is not None:
                self.window_level = window_level
                self.window_width = window_width
                
                self.window_level_spin.blockSignals(True)
                self.window_level_spin.setValue(window_level)
                self.window_level_spin.blockSignals(False)
                
                self.window_width_spin.blockSignals(True)
                self.window_width_spin.setValue(window_width)
                self.window_width_spin.blockSignals(False)
            else:
                # Tự động tính window level và width dựa trên histogram
                p5 = np.percentile(data, 5)
                p95 = np.percentile(data, 95)
                
                self.window_level = int((p5 + p95) / 2)
                self.window_width = int(p95 - p5)
                
                self.window_level_spin.blockSignals(True)
                self.window_level_spin.setValue(self.window_level)
                self.window_level_spin.blockSignals(False)
                
                self.window_width_spin.blockSignals(True)
                self.window_width_spin.setValue(self.window_width)
                self.window_width_spin.blockSignals(False)
            
            # Áp dụng preset từ combobox
            self.apply_preset(self.preset_combo.currentText())
            
            # Thiết lập volume property
            self.volume_property.SetColor(color_function)
            self.volume_property.SetScalarOpacity(opacity_function)
            
            # Tạo volume actor
            volume = vtk.vtkVolume()
            volume.SetMapper(volume_mapper)
            volume.SetProperty(self.volume_property)
            
            # Xóa actor cũ nếu có
            if self.volume_actor:
                self.renderer.RemoveVolume(self.volume_actor)
                
            # Gán actor mới
            self.volume_actor = volume
            self.renderer.AddVolume(self.volume_actor)
            
            # Reset camera
            self.renderer.ResetCamera()
            self.render_window.Render()
            
            logger.info("Đã tải dữ liệu hình ảnh 3D vào VolumeRenderingWidget")
            return True
            
        except Exception as e:
            logger.error(f"Lỗi khi tải dữ liệu hình ảnh 3D: {str(e)}", exc_info=True)
            return False
            
    def add_contour(self, name, contour_data, color=None):
        """
        Thêm contour vào cảnh 3D.
        
        Parameters
        ----------
        name : str
            Tên của contour
        contour_data : vtkPolyData hoặc list of points
            Dữ liệu của contour
        color : tuple, optional
            Màu RGB, giá trị từ 0 đến 1
        
        Returns
        -------
        bool
            True nếu thành công, False nếu thất bại
        """
        if not VTK_AVAILABLE or not self.renderer:
            logger.error("VTK không khả dụng hoặc renderer chưa được khởi tạo")
            return False
            
        try:
            # Tạo polydata nếu đầu vào là list of points
            if not isinstance(contour_data, vtk.vtkPolyData):
                # Tạo polydata từ điểm
                points = vtk.vtkPoints()
                for i, point in enumerate(contour_data):
                    points.InsertPoint(i, point)
                    
                # Tạo cell array
                cells = vtk.vtkCellArray()
                for i in range(len(contour_data) - 1):
                    line = vtk.vtkLine()
                    line.GetPointIds().SetId(0, i)
                    line.GetPointIds().SetId(1, i + 1)
                    cells.InsertNextCell(line)
                
                # Tạo line từ điểm cuối đến điểm đầu
                if len(contour_data) > 2:
                    line = vtk.vtkLine()
                    line.GetPointIds().SetId(0, len(contour_data) - 1)
                    line.GetPointIds().SetId(1, 0)
                    cells.InsertNextCell(line)
                
                # Tạo polydata
                polydata = vtk.vtkPolyData()
                polydata.SetPoints(points)
                polydata.SetLines(cells)
            else:
                polydata = contour_data
            
            # Xác định màu
            if color is None:
                # Sử dụng màu mặc định cho contour
                if name in self.contour_default_colors:
                    color = self.contour_default_colors[name]
                else:
                    # Tạo màu ngẫu nhiên
                    import random
                    color = (random.random(), random.random(), random.random())
            
            # Tạo mapper
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(polydata)
            
            # Tạo actor
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(color)
            actor.GetProperty().SetOpacity(0.7)  # Độ trong suốt mặc định
            
            # Xóa actor cũ nếu có
            if name in self.contour_actors:
                self.renderer.RemoveActor(self.contour_actors[name])
                
            # Gán actor mới
            self.contour_actors[name] = actor
            self.renderer.AddActor(actor)
            
            # Thêm tên vào combobox nếu chưa có
            items = [self.contour_list.itemText(i) for i in range(self.contour_list.count())]
            if name not in items[1:]:  # Bỏ qua "Tất cả"
                self.contour_list.addItem(name)
            
            # Render lại
            self.render_window.Render()
            
            logger.info(f"Đã thêm contour '{name}' vào cảnh 3D")
            return True
            
        except Exception as e:
            logger.error(f"Lỗi khi thêm contour: {str(e)}", exc_info=True)
            return False