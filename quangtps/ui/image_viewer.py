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
from PyQt5.QtCore import Qt, QSize, pyqtSignal, pyqtSlot, QRectF, QPoint, QEvent
from PyQt5.QtGui import (
    QImage, QPixmap, QPainter, QColor, QPen, QBrush, QFont, QTransform,
    QMouseEvent, QKeyEvent, QWheelEvent
)

try:
    from PyQt5.QtDataVisualization import Q3DScatter
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    logging.warning("PyQt5.QtDataVisualization không khả dụng. Chức năng 3D sẽ bị giới hạn.")

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
        self.view_mode = "4-View"  # "4-View" (3 mặt phẳng + 3D) hoặc "Single"
        self.window_width = 500
        self.window_level = 40
        self.overlay_opacity = 0.7
        self.current_position = [0, 0, 0]  # Vị trí hiện tại trong không gian 3D
        self.current_tool = "pan"  # Công cụ hiện tại (pan, zoom, window...)
        
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
        view_mode_combo.addItems(["4-View", "Axial", "Sagittal", "Coronal", "3D"])
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
        
        # Khu vực hiển thị chính - Sử dụng grid layout cho hiển thị linh hoạt
        self.view_container = QWidget()
        main_layout.addWidget(self.view_container, 1)
        
        # Layout chính chứa các view
        self.views_layout = QHBoxLayout(self.view_container)
        self.views_layout.setContentsMargins(0, 0, 0, 0)
        
        # Tạo widget chứa các view 2D
        self.mpr_widget = QWidget()
        self.mpr_layout = QVBoxLayout(self.mpr_widget)
        self.mpr_layout.setContentsMargins(0, 0, 0, 0)
        
        # Tạo grid layout cho 3 view 2D
        self.grid_layout = QHBoxLayout()
        self.mpr_layout.addLayout(self.grid_layout)
        
        # Tạo 3 màn hình hiển thị cho 3 mặt phẳng: Axial, Sagittal, Coronal
        self.axial_view = ImageDisplay()
        self.axial_view.setMinimumSize(250, 250)
        self.axial_view.set_title("Axial")
        
        self.sagittal_view = ImageDisplay()
        self.sagittal_view.setMinimumSize(250, 250)
        self.sagittal_view.set_title("Sagittal")
        
        self.coronal_view = ImageDisplay()
        self.coronal_view.setMinimumSize(250, 250)
        self.coronal_view.set_title("Coronal")
        
        # Tạo view 3D
        self.view_3d = QWidget()
        self.view_3d_layout = QVBoxLayout(self.view_3d)
        self.view_3d.setMinimumSize(250, 250)
        self.view_3d_layout.addWidget(QLabel("Hiển thị 3D"))
        
        # Thêm view 3D
        if VISUALIZATION_AVAILABLE:
            try:
                self.scatter = Q3DScatter()
                self.scatter_widget = QWidget.createWindowContainer(self.scatter)
                self.scatter_widget.setMinimumSize(250, 250)
                self.view_3d_layout.addWidget(self.scatter_widget)
            except Exception as e:
                logger.error("Lỗi khi tạo view 3D: %s", str(e))
                self.view_3d_layout.addWidget(QLabel("Không thể hiển thị 3D: " + str(e)))
        else:
            self.view_3d_layout.addWidget(QLabel("Chức năng 3D không khả dụng"))
        
        # Thêm các view vào grid
        self.grid_layout.addWidget(self.axial_view)
        self.grid_layout.addWidget(self.sagittal_view)
        self.views_layout.addWidget(self.mpr_widget, 2)
        self.views_layout.addWidget(self.view_3d, 1)
        
        # Widget ở dưới cùng cho coronal view
        self.bottom_widget = QWidget()
        self.bottom_layout = QVBoxLayout(self.bottom_widget)
        self.bottom_layout.setContentsMargins(0, 0, 0, 0)
        self.bottom_layout.addWidget(self.coronal_view)
        self.mpr_layout.addWidget(self.bottom_widget)
        
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
        
        # Kết nối sự kiện giữa các chế độ xem
        self.axial_view.mouse_position_changed.connect(self._update_position_info)
        self.sagittal_view.mouse_position_changed.connect(self._update_position_info)
        self.coronal_view.mouse_position_changed.connect(self._update_position_info)
        
        # Kết nối sự kiện wheel từ các view đến hàm xử lý thay đổi lát cắt
        self.axial_view.installEventFilter(self)
        self.sagittal_view.installEventFilter(self)
        self.coronal_view.installEventFilter(self)
    
    def load_image(self, image: Image):
        """
        Tải một hình ảnh y tế vào trình xem.
        
        Args:
            image: Đối tượng Image chứa dữ liệu hình ảnh y tế
        """
        self.primary_image = image
        
        # Cập nhật thanh trượt lát cắt
        if image is not None and hasattr(image, 'data') and image.data is not None and image.data.size > 0:
            depth = image.data.shape[0]
            self.slice_slider.setRange(0, depth - 1)
            self.slice_slider.setValue(depth // 2)
            
            # Thiết lập cửa sổ mặc định dựa trên hình ảnh
            if hasattr(image, 'modality') and image.modality == "CT":
                self._set_window(1500, 500)  # CT default
            elif hasattr(image, 'modality') and image.modality == "MR":
                self._set_window(1000, 500)  # MR default
            elif hasattr(image, 'modality') and image.modality == "PT":
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
        Thay đổi chế độ xem giữa các mode hiển thị.
        
        Args:
            mode: Chế độ xem mới (4-View, Axial, Sagittal, Coronal, 3D)
        """
        self.view_mode = mode
        self._update_view_mode()
    
    def _update_view_mode(self):
        """Cập nhật giao diện dựa trên chế độ xem hiện tại."""
        # Ẩn tất cả các view trước
        self.axial_view.setVisible(False)
        self.sagittal_view.setVisible(False)
        self.coronal_view.setVisible(False)
        self.view_3d.setVisible(False)
        
        # Hiển thị theo chế độ được chọn
        if self.view_mode == "4-View":
            self.mpr_widget.setVisible(True)
            self.view_3d.setVisible(True)
            self.axial_view.setVisible(True)
            self.sagittal_view.setVisible(True)
            self.coronal_view.setVisible(True)
        elif self.view_mode == "Axial":
            self.mpr_widget.setVisible(True)
            self.view_3d.setVisible(False)
            self.axial_view.setVisible(True)
            self.sagittal_view.setVisible(False)
            self.coronal_view.setVisible(False)
        elif self.view_mode == "Sagittal":
            self.mpr_widget.setVisible(True)
            self.view_3d.setVisible(False)
            self.axial_view.setVisible(False)
            self.sagittal_view.setVisible(True)
            self.coronal_view.setVisible(False)
        elif self.view_mode == "Coronal":
            self.mpr_widget.setVisible(True)
            self.view_3d.setVisible(False)
            self.axial_view.setVisible(False)
            self.sagittal_view.setVisible(False)
            self.coronal_view.setVisible(True)
        elif self.view_mode == "3D":
            self.mpr_widget.setVisible(False)
            self.view_3d.setVisible(True)
    
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
        if self.primary_image is None or not hasattr(self.primary_image, 'data') or self.primary_image.data is None or self.primary_image.data.size == 0:
            return
            
        # Cập nhật vị trí hiện tại
        self.current_position[2] = value
        
        # Cập nhật hiển thị
        self._update_displays()
        
        # Cập nhật thông tin hiển thị
        self._update_info_label()
    
    def _update_displays(self):
        """Cập nhật tất cả các màn hình hiển thị."""
        if self.primary_image is None or not hasattr(self.primary_image, 'data') or self.primary_image.data is None or self.primary_image.data.size == 0:
            logger.warning("Không thể cập nhật hiển thị: Không có dữ liệu hình ảnh hợp lệ")
            return
            
        try:
            # Lấy dữ liệu hình ảnh và vị trí hiện tại
            image_data = self.primary_image.data
            x, y, z = self.current_position
            
            # Kiểm tra dữ liệu hình ảnh có hợp lệ không
            if not isinstance(image_data, np.ndarray):
                logger.error(f"Dữ liệu hình ảnh không phải là mảng NumPy: {type(image_data)}")
                return
                
            # Kiểm tra kích thước mảng có hợp lệ
            if image_data.ndim < 2:
                logger.error(f"Dữ liệu hình ảnh không đủ chiều để hiển thị: {image_data.ndim}D")
                return

            logger.debug(f"Cập nhật hiển thị với dữ liệu hình ảnh kích thước {image_data.shape}, vị trí hiện tại: [{z}, {y}, {x}]")
            
            # Giới hạn vị trí trong phạm vi hình ảnh
            if len(image_data.shape) >= 3:
                z = min(max(0, z), image_data.shape[0] - 1) if image_data.shape[0] > 0 else 0
                if len(image_data.shape) > 1:
                    y = min(max(0, y), image_data.shape[1] - 1) if image_data.shape[1] > 0 else 0
                    if len(image_data.shape) > 2:
                        x = min(max(0, x), image_data.shape[2] - 1) if image_data.shape[2] > 0 else 0
            
            # Cập nhật vị trí hiện tại sau khi giới hạn
            self.current_position = [x, y, z]
            logger.debug(f"Vị trí sau khi giới hạn: [{z}, {y}, {x}]")
            
            # Cập nhật giá trị của thanh trượt lát cắt
            if hasattr(self, 'slice_slider') and self.slice_slider is not None:
                if self.slice_slider.value() != z:
                    self.slice_slider.blockSignals(True)
                    self.slice_slider.setValue(z)
                    self.slice_slider.blockSignals(False)
                
            # Lấy lát cắt hình ảnh
            try:
                # Xác định slice dựa trên số chiều của mảng
                if image_data.ndim == 2:  # Ảnh 2D
                    axial_slice = image_data
                    sagittal_slice = None
                    coronal_slice = None
                    logger.debug("Dữ liệu 2D: Chỉ hiển thị lát cắt axial")
                elif image_data.ndim == 3:  # Ảnh 3D
                    # Kiểm tra shape hợp lệ
                    if 0 in image_data.shape:
                        logger.error(f"Kích thước dữ liệu hình ảnh không hợp lệ: {image_data.shape}")
                        return
                        
                    # Lát cắt Axial (Z)
                    if 0 <= z < image_data.shape[0]:
                        axial_slice = image_data[z, :, :]
                        logger.debug(f"Lát cắt Axial tại z={z}, kích thước: {axial_slice.shape}")
                    else:
                        logger.warning(f"Chỉ số z={z} nằm ngoài phạm vi [0, {image_data.shape[0]-1}]")
                        z = max(0, min(image_data.shape[0]-1, z))  # Giới hạn lại z
                        axial_slice = image_data[z, :, :] if image_data.shape[0] > 0 else None
                        self.current_position[2] = z  # Cập nhật vị trí sau khi giới hạn
                    
                    # Lát cắt Sagittal (X)
                    if 0 <= x < image_data.shape[2]:
                        try:
                            sagittal_slice = image_data[:, :, x]
                            logger.debug(f"Lát cắt Sagittal tại x={x}, kích thước: {sagittal_slice.shape}")
                        except IndexError as e:
                            logger.error(f"Lỗi khi lấy lát cắt Sagittal: {str(e)}")
                            x = max(0, min(image_data.shape[2]-1, x))  # Giới hạn lại x
                            try:
                                sagittal_slice = image_data[:, :, x] if image_data.shape[2] > 0 else None
                                self.current_position[0] = x  # Cập nhật vị trí sau khi giới hạn
                            except Exception:
                                sagittal_slice = None
                    else:
                        logger.warning(f"Chỉ số x={x} nằm ngoài phạm vi [0, {image_data.shape[2]-1}]")
                        x = max(0, min(image_data.shape[2]-1, x))  # Giới hạn lại x
                        try:
                            sagittal_slice = image_data[:, :, x] if image_data.shape[2] > 0 else None
                            self.current_position[0] = x  # Cập nhật vị trí sau khi giới hạn
                        except Exception:
                            sagittal_slice = None
                    
                    # Lát cắt Coronal (Y)
                    if 0 <= y < image_data.shape[1]:
                        try:
                            coronal_slice = image_data[:, y, :]
                            logger.debug(f"Lát cắt Coronal tại y={y}, kích thước: {coronal_slice.shape}")
                        except IndexError as e:
                            logger.error(f"Lỗi khi lấy lát cắt Coronal: {str(e)}")
                            y = max(0, min(image_data.shape[1]-1, y))  # Giới hạn lại y
                            try:
                                coronal_slice = image_data[:, y, :] if image_data.shape[1] > 0 else None
                                self.current_position[1] = y  # Cập nhật vị trí sau khi giới hạn
                            except Exception:
                                coronal_slice = None
                    else:
                        logger.warning(f"Chỉ số y={y} nằm ngoài phạm vi [0, {image_data.shape[1]-1}]")
                        y = max(0, min(image_data.shape[1]-1, y))  # Giới hạn lại y
                        try:
                            coronal_slice = image_data[:, y, :] if image_data.shape[1] > 0 else None
                            self.current_position[1] = y  # Cập nhật vị trí sau khi giới hạn
                        except Exception:
                            coronal_slice = None
                else:  # Ảnh >= 4D (hiếm gặp)
                    logger.warning(f"Dữ liệu hình ảnh có {image_data.ndim} chiều, chỉ hỗ trợ tối đa 3 chiều")
                    # Lấy lát cắt từ 3 chiều đầu tiên
                    try:
                        if z < image_data.shape[0]:
                            axial_slice = image_data[z, :, :, 0]  # Lấy slice đầu tiên ở chiều thứ 4
                        else:
                            z = max(0, min(image_data.shape[0]-1, z))  # Giới hạn lại z
                            axial_slice = image_data[z, :, :, 0] if image_data.shape[0] > 0 else None
                            self.current_position[2] = z  # Cập nhật vị trí sau khi giới hạn
                            
                        if x < image_data.shape[2]:
                            sagittal_slice = image_data[:, :, x, 0]
                        else:
                            x = max(0, min(image_data.shape[2]-1, x))  # Giới hạn lại x
                            sagittal_slice = image_data[:, :, x, 0] if image_data.shape[2] > 0 else None
                            self.current_position[0] = x  # Cập nhật vị trí sau khi giới hạn
                            
                        if y < image_data.shape[1]:
                            coronal_slice = image_data[:, y, :, 0]
                        else:
                            y = max(0, min(image_data.shape[1]-1, y))  # Giới hạn lại y
                            coronal_slice = image_data[:, y, :, 0] if image_data.shape[1] > 0 else None
                            self.current_position[1] = y  # Cập nhật vị trí sau khi giới hạn
                    except IndexError as e:
                        logger.error(f"Lỗi truy cập chiều không tồn tại: {str(e)}")
                        # Fallback to safe values
                        axial_slice = image_data[0, :, :, 0] if image_data.shape[0] > 0 else None
                        sagittal_slice = None
                        coronal_slice = None
            except IndexError as e:
                logger.error(f"Lỗi chỉ số khi lấy lát cắt: z={z}, y={y}, x={x}, shape={image_data.shape}, lỗi={str(e)}")
                # Thử lấy giá trị an toàn
                try:
                    # Kiểm tra lại các chỉ số
                    safe_z = min(max(0, z), image_data.shape[0]-1) if image_data.shape[0] > 0 else 0
                    safe_y = min(max(0, y), image_data.shape[1]-1) if len(image_data.shape) > 1 and image_data.shape[1] > 0 else 0
                    safe_x = min(max(0, x), image_data.shape[2]-1) if len(image_data.shape) > 2 and image_data.shape[2] > 0 else 0
                    
                    # Cập nhật vị trí hiện tại với các chỉ số an toàn
                    self.current_position = [safe_x, safe_y, safe_z]
                    logger.info(f"Đã điều chỉnh vị trí sang giá trị an toàn: [{safe_z}, {safe_y}, {safe_x}]")
                    
                    # Lấy lát cắt với các chỉ số an toàn
                    axial_slice = image_data[safe_z, :, :] if image_data.shape[0] > 0 else None
                    sagittal_slice = image_data[:, :, safe_x] if len(image_data.shape) > 2 and image_data.shape[2] > 0 else None
                    coronal_slice = image_data[:, safe_y, :] if len(image_data.shape) > 1 and image_data.shape[1] > 0 else None
                    logger.info("Đã lấy lát cắt mặc định thay thế với chỉ số an toàn")
                except Exception as e2:
                    logger.error(f"Lỗi khi thử lấy lát cắt an toàn: {str(e2)}")
                    # Fallback to simplest case
                    try:
                        axial_slice = image_data[0, :, :] if image_data.shape[0] > 0 else None
                    except Exception:
                        axial_slice = None
                    sagittal_slice = None
                    coronal_slice = None
            except Exception as e:
                logger.error(f"Lỗi không xác định khi lấy lát cắt: {str(e)}")
                import traceback
                logger.debug(traceback.format_exc())
                axial_slice = None
                sagittal_slice = None
                coronal_slice = None
            
            # Cập nhật hiển thị nếu có lát cắt hợp lệ
            try:
                # Kiểm tra lát cắt trước khi cập nhật hiển thị
                if axial_slice is not None and isinstance(axial_slice, np.ndarray) and axial_slice.size > 0:
                    self.axial_view.set_image(axial_slice, plane="axial")
                    logger.debug("Đã cập nhật hiển thị lát cắt Axial")
                else:
                    logger.warning("Không thể hiển thị lát cắt Axial: dữ liệu không hợp lệ")
                    
                if sagittal_slice is not None and isinstance(sagittal_slice, np.ndarray) and sagittal_slice.size > 0:
                    self.sagittal_view.set_image(sagittal_slice, plane="sagittal")
                    logger.debug("Đã cập nhật hiển thị lát cắt Sagittal")
                else:
                    logger.warning("Không thể hiển thị lát cắt Sagittal: dữ liệu không hợp lệ")
                    
                if coronal_slice is not None and isinstance(coronal_slice, np.ndarray) and coronal_slice.size > 0:
                    self.coronal_view.set_image(coronal_slice, plane="coronal")
                    logger.debug("Đã cập nhật hiển thị lát cắt Coronal")
                else:
                    logger.warning("Không thể hiển thị lát cắt Coronal: dữ liệu không hợp lệ")
            except Exception as e:
                logger.error(f"Lỗi khi cập nhật hiển thị lát cắt: {str(e)}")
                import traceback
                logger.debug(traceback.format_exc())
                
            # Cập nhật thông tin vị trí
            try:
                # Kiểm tra vị trí nằm trong phạm vi hợp lệ
                x, y, z = self.current_position  # Lấy lại giá trị đã được cập nhật
                if (0 <= z < image_data.shape[0] and
                    0 <= y < image_data.shape[1] and
                    0 <= x < image_data.shape[2]):
                    value = image_data[z, y, x]
                    self._update_position_info(x, y, value)
                    logger.debug(f"Giá trị tại vị trí [{z},{y},{x}] = {value}")
                else:
                    logger.warning(f"Vị trí [{z},{y},{x}] nằm ngoài phạm vi dữ liệu")
                    # Hiển thị giá trị NaN hoặc 0 cho các vị trí không hợp lệ
                    self._update_position_info(x, y, float('nan'))
            except IndexError as e:
                logger.error(f"Lỗi truy cập vị trí [{z}, {y}, {x}] trong mảng shape={image_data.shape}: {str(e)}")
                self._update_position_info(x, y, 0)
            except Exception as e:
                logger.error(f"Lỗi không xác định khi cập nhật thông tin vị trí: {str(e)}")
                self._update_position_info(x, y, 0)
                
        except Exception as e:
            logger.error(f"Lỗi khi cập nhật hiển thị: {str(e)}")
            import traceback
            logger.debug(traceback.format_exc())
    
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
        if self.primary_image is None or not hasattr(self.primary_image, 'data') or self.primary_image.data is None:
            return
            
        x, y, z = self.current_position
        depth = self.primary_image.data.shape[0]
        
        # Lấy giá trị HU tại vị trí hiện tại nếu nằm trong phạm vi
        try:
            hu_value = self.primary_image.data[z, x, y]
        except IndexError:
            hu_value = "-"
            
        self.info_label.setText(f"Slice: {z+1}/{depth} | Pos: ({x}, {y}, {z}) | HU: {hu_value}")
        
        # Phát tín hiệu thay đổi vị trí
        self.position_changed.emit(x, y, z)
    
    def eventFilter(self, source, event):
        """
        Lọc sự kiện từ các widget con để xử lý sự kiện wheel từ các màn hình hiển thị.
        
        Args:
            source: Đối tượng phát sự kiện
            event: Sự kiện được phát
        
        Returns:
            bool: True nếu sự kiện đã được xử lý, False nếu không
        """
        if event.type() == QEvent.Wheel and self.primary_image is not None and hasattr(self.primary_image, 'data') and self.primary_image.data is not None:
            delta = event.angleDelta().y()
            step = 1 if delta < 0 else -1  # Đảo ngược hướng để phù hợp với thói quen cuộn
            
            if source == self.axial_view:
                # Thay đổi lát cắt Axial (z)
                z = self.current_position[2]
                new_z = max(0, min(z + step, self.primary_image.data.shape[0] - 1))
                if new_z != z:
                    self.current_position[2] = new_z
                    self.slice_slider.setValue(new_z)  # Cập nhật giá trị thanh trượt
                    self._update_displays()
                    self._update_info_label()
                return True
                
            elif source == self.sagittal_view:
                # Thay đổi lát cắt Sagittal (x)
                x = self.current_position[0]
                new_x = max(0, min(x + step, self.primary_image.data.shape[1] - 1))
                if new_x != x:
                    self.current_position[0] = new_x
                    self._update_displays()
                    self._update_info_label()
                return True
                
            elif source == self.coronal_view:
                # Thay đổi lát cắt Coronal (y)
                y = self.current_position[1]
                new_y = max(0, min(y + step, self.primary_image.data.shape[2] - 1))
                if new_y != y:
                    self.current_position[1] = new_y
                    self._update_displays()
                    self._update_info_label()
                return True
                
        return super().eventFilter(source, event)
