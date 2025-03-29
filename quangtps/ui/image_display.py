#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module cung cấp các widget hiển thị hình ảnh y tế cho QuangTPS.

Module này bao gồm các widget để hiển thị, điều khiển và tương tác với 
hình ảnh y tế (CT, MRI, PET, v.v.) cũng như các contour cấu trúc.
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, 
    QPushButton, QComboBox, QGroupBox, QFormLayout,
    QDoubleSpinBox, QScrollArea, QSplitter, QCheckBox, 
    QRadioButton, QButtonGroup, QFrame, QGridLayout
)
from PyQt5.QtCore import Qt, pyqtSignal, QRectF, QPointF
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QBrush, QFont, QPainterPath

try:
    import matplotlib
    matplotlib.use('Qt5Agg')
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

logger = logging.getLogger(__name__)


class ImageSliceWidget(QWidget):
    """Widget để hiển thị một lát cắt 2D từ tập dữ liệu 3D."""
    
    # Tín hiệu được phát khi vị trí con trỏ thay đổi
    position_changed = pyqtSignal(int, int, int)  # slice_idx, x, y
    mouse_pressed = pyqtSignal(int, int, int, Qt.MouseButton)  # slice_idx, x, y, button
    mouse_moved = pyqtSignal(int, int, int)  # slice_idx, x, y
    mouse_released = pyqtSignal(int, int, int, Qt.MouseButton)  # slice_idx, x, y, button
    key_pressed = pyqtSignal(int)  # key
    key_released = pyqtSignal(int)  # key
    
    def __init__(self, parent=None):
        """
        Khởi tạo widget hiển thị lát cắt hình ảnh.
        
        Parameters
        ----------
        parent : QWidget, optional
            Widget cha
        """
        super().__init__(parent)
        
        # Dữ liệu hình ảnh
        self.image_data = None  # Dữ liệu hình ảnh 3D (z, y, x)
        self.slice_idx = 0  # Chỉ số lát cắt hiện tại
        self.view_plane = 'axial'  # Mặt phẳng xem: 'axial', 'coronal', 'sagittal'
        self.hounsfield_window = (0, 2000)  # Cửa sổ Hounsfield (min, max)
        self.zoom_factor = 1.0  # Hệ số zoom
        self.pan_offset = (0, 0)  # Độ dịch chuyển (dx, dy)
        
        # Contour
        self.contours = {}  # Dict của các contour: {name: [(slice_idx, contour_points), ...]}
        self.contour_colors = {}  # Dict màu sắc contour: {name: QColor}
        self.contour_visibility = {}  # Dict hiển thị contour: {name: bool}
        
        # Dữ liệu liều lượng
        self.dose_data = None
        
        # UI
        self._init_ui()
        
        # Kích thước tối thiểu
        self.setMinimumSize(300, 300)
    
    def _init_ui(self):
        """Khởi tạo giao diện người dùng."""
        # Layout chính
        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)
        
        # Frame hiển thị hình ảnh
        self.image_frame = QLabel()
        self.image_frame.setAlignment(Qt.AlignCenter)
        self.image_frame.setStyleSheet("background-color: black;")
        self.layout().addWidget(self.image_frame)
        
        # Thiết lập màu mặc định cho contour
        self.default_contour_colors = {
            'tumor': QColor(255, 0, 0, 128),  # Đỏ bán trong suốt
            'ptv': QColor(255, 165, 0, 128),  # Cam bán trong suốt
            'ctv': QColor(255, 255, 0, 128),  # Vàng bán trong suốt
            'oar': QColor(0, 255, 0, 128),    # Xanh lá bán trong suốt
            'body': QColor(0, 0, 255, 128)    # Xanh dương bán trong suốt
        }
    
    def set_image_data(self, image_data, plane=None):
        """
        Thiết lập dữ liệu hình ảnh 3D.
        
        Parameters
        ----------
        image_data : ndarray
            Dữ liệu hình ảnh 3D (z, y, x) kiểu numpy array
        plane : str, optional
            Mặt phẳng xem ('axial', 'coronal', 'sagittal')
        """
        self.image_data = image_data
        
        # Set the view plane if provided
        if plane is not None and plane in ['axial', 'coronal', 'sagittal']:
            self.view_plane = plane
        elif not hasattr(self, 'view_plane'):
            self.view_plane = 'axial'  # Default plane
        
        if image_data is not None:
            max_slice = image_data.shape[0] - 1
            self.slice_idx = max_slice // 2  # Mặc định slice giữa
        else:
            self.slice_idx = 0
        
        self._update_display()
    
    def set_hounsfield_window(self, window_min, window_max):
        """
        Thiết lập cửa sổ Hounsfield.
        
        Parameters
        ----------
        window_min : int
            Giá trị Hounsfield tối thiểu
        window_max : int
            Giá trị Hounsfield tối đa
        """
        self.hounsfield_window = (window_min, window_max)
        self._update_display()
    
    def set_view_plane(self, plane):
        """
        Thiết lập mặt phẳng xem.
        
        Parameters
        ----------
        plane : str
            Mặt phẳng xem ('axial', 'coronal', 'sagittal')
        """
        if plane in ['axial', 'coronal', 'sagittal']:
            self.view_plane = plane
            
            # Reset slice_idx to the middle of the new plane
            if self.image_data is not None:
                if plane == 'axial':
                    max_slice = self.image_data.shape[0] - 1
                elif plane == 'coronal':
                    max_slice = self.image_data.shape[1] - 1
                else:  # sagittal
                    max_slice = self.image_data.shape[2] - 1
                
                self.slice_idx = max_slice // 2
            
            self._update_display()
    
    def set_slice_idx(self, idx):
        """
        Thiết lập chỉ số lát cắt.
        
        Parameters
        ----------
        idx : int
            Chỉ số lát cắt mới
        """
        if self.image_data is None:
            return
            
        # Xác định giới hạn chỉ số dựa trên mặt phẳng
        if self.view_plane == 'axial':
            max_slice = self.image_data.shape[0] - 1
        elif self.view_plane == 'coronal':
            max_slice = self.image_data.shape[1] - 1
        else:  # sagittal
            max_slice = self.image_data.shape[2] - 1
        
        # Đảm bảo chỉ số trong phạm vi hợp lệ
        idx = max(0, min(idx, max_slice))
        
        if idx != self.slice_idx:
            self.slice_idx = idx
            self._update_display()
    
    def set_slice_index(self, idx):
        """
        Bí danh cho phương thức set_slice_idx để đảm bảo tính tương thích.
        
        Parameters
        ----------
        idx : int
            Chỉ số lát cắt mới
        """
        self.set_slice_idx(idx)
    
    def set_zoom(self, factor):
        """
        Thiết lập hệ số zoom.
        
        Parameters
        ----------
        factor : float
            Hệ số zoom mới
        """
        self.zoom_factor = max(0.1, factor)
        self._update_display()
    
    def set_pan(self, dx, dy):
        """
        Thiết lập độ dịch chuyển.
        
        Parameters
        ----------
        dx : int
            Độ dịch chuyển theo trục x
        dy : int
            Độ dịch chuyển theo trục y
        """
        self.pan_offset = (dx, dy)
        self._update_display()
    
    def add_contour(self, name, contour_points, slice_idx=None, color=None):
        """
        Thêm contour vào một lát cắt cụ thể.
        
        Parameters
        ----------
        name : str
            Tên contour
        contour_points : list
            Danh sách các điểm contour [(x1, y1), (x2, y2), ...]
        slice_idx : int, optional
            Chỉ số lát cắt (mặc định là lát cắt hiện tại)
        color : QColor, optional
            Màu sắc contour (mặc định theo loại contour)
        """
        if slice_idx is None:
            slice_idx = self.slice_idx
        
        # Thêm contour vào danh sách
        if name not in self.contours:
            self.contours[name] = []
            
            # Thiết lập màu và khả năng hiển thị
            if color is None:
                if name.lower() in self.default_contour_colors:
                    color = self.default_contour_colors[name.lower()]
                else:
                    # Màu ngẫu nhiên nếu không có màu mặc định
                    import random
                    color = QColor(
                        random.randint(0, 255),
                        random.randint(0, 255),
                        random.randint(0, 255),
                        128
                    )
            
            self.contour_colors[name] = color
            self.contour_visibility[name] = True
        
        # Thêm hoặc cập nhật contour cho lát cắt cụ thể
        contour_updated = False
        for i, (idx, points) in enumerate(self.contours[name]):
            if idx == slice_idx:
                self.contours[name][i] = (slice_idx, contour_points)
                contour_updated = True
                break
        
        if not contour_updated:
            self.contours[name].append((slice_idx, contour_points))
        
        self._update_display()
    
    def remove_contour(self, name, slice_idx=None):
        """
        Xóa contour.
        
        Parameters
        ----------
        name : str
            Tên contour
        slice_idx : int, optional
            Chỉ số lát cắt (nếu là None, xóa tất cả các contour có tên đã cho)
        """
        if name not in self.contours:
            return
        
        if slice_idx is None:
            # Xóa tất cả contour có tên đã cho
            del self.contours[name]
            if name in self.contour_colors:
                del self.contour_colors[name]
            if name in self.contour_visibility:
                del self.contour_visibility[name]
        else:
            # Xóa contour cụ thể trên một lát cắt
            self.contours[name] = [(idx, points) for idx, points in self.contours[name] if idx != slice_idx]
            if not self.contours[name]:
                del self.contours[name]
                if name in self.contour_colors:
                    del self.contour_colors[name]
                if name in self.contour_visibility:
                    del self.contour_visibility[name]
        
        self._update_display()
    
    def set_contour_visibility(self, name, visible):
        """
        Thiết lập khả năng hiển thị của contour.
        
        Parameters
        ----------
        name : str
            Tên contour
        visible : bool
            Trạng thái hiển thị
        """
        if name in self.contour_visibility:
            self.contour_visibility[name] = visible
            self._update_display()
    
    def set_contour_color(self, name, color):
        """
        Thiết lập màu sắc của contour.
        
        Parameters
        ----------
        name : str
            Tên contour
        color : QColor
            Màu sắc mới
        """
        if name in self.contour_colors:
            self.contour_colors[name] = color
            self._update_display()
    
    def set_dose_data(self, dose_data, colormap='jet', alpha=0.7):
        """
        Thiết lập dữ liệu liều lượng xạ trị để hiển thị chồng lên hình ảnh nền.
        
        Parameters
        ----------
        dose_data : ndarray
            Dữ liệu liều lượng 2D (y, x) kiểu numpy array
        colormap : str, optional
            Bảng màu để hiển thị (mặc định: 'jet')
        alpha : float, optional
            Độ trong suốt của lớp hiển thị liều (mặc định: 0.7)
        """
        if dose_data is None:
            self.dose_data = None
            self._update_display()
            return
            
        # Lưu dữ liệu liều lượng vào bộ nhớ
        if len(dose_data.shape) == 2:
            # Nếu là lát cắt 2D, tạo tập dữ liệu 3D với 1 lát cắt
            self.dose_data = np.expand_dims(dose_data, axis=0)
        else:
            # Nếu đã là dữ liệu 3D thì sử dụng trực tiếp
            self.dose_data = dose_data
            
        # Lưu các thông số hiển thị
        self.dose_colormap = colormap
        self.dose_alpha = alpha
        
        # Thiết lập chỉ số lát cắt mặc định nếu chưa được thiết lập
        if not hasattr(self, 'slice_idx'):
            self.slice_idx = 0
        
        # Cập nhật hiển thị
        self._update_display()
    
    def _get_current_slice(self):
        """
        Lấy lát cắt hiện tại dựa trên mặt phẳng xem.
        
        Returns
        -------
        ndarray
            Lát cắt 2D (y, x) hiện tại
        """
        if self.image_data is None:
            return None
        
        # Xác định giới hạn chỉ số dựa trên mặt phẳng
        if self.view_plane == 'axial':
            # Lát cắt ngang (z cố định)
            if 0 <= self.slice_idx < self.image_data.shape[0]:
                return self.image_data[self.slice_idx, :, :]
        elif self.view_plane == 'coronal':
            # Lát cắt coronal (y cố định)
            if 0 <= self.slice_idx < self.image_data.shape[1]:
                return self.image_data[:, self.slice_idx, :]
        else:  # sagittal
            # Lát cắt sagittal (x cố định)
            if 0 <= self.slice_idx < self.image_data.shape[2]:
                return self.image_data[:, :, self.slice_idx]
        
        return None
    
    def _update_display(self):
        """Cập nhật hiển thị hình ảnh và contour."""
        slice_data = self._get_current_slice()
        if slice_data is None:
            # Hiển thị hình ảnh trống
            self.image_frame.setPixmap(QPixmap())
            return
        
        # Chuyển đổi dữ liệu lát cắt thành hình ảnh QImage
        h, w = slice_data.shape
        
        # Áp dụng cửa sổ Hounsfield
        window_min, window_max = self.hounsfield_window
        normalized_data = np.clip(slice_data, window_min, window_max)
        normalized_data = (normalized_data - window_min) / (window_max - window_min) * 255
        normalized_data = normalized_data.astype(np.uint8)
        
        # Tạo QImage
        q_image = QImage(normalized_data.data, w, h, w, QImage.Format_Grayscale8)
        
        # Tạo QPixmap từ QImage
        pixmap = QPixmap.fromImage(q_image)
        
        # Áp dụng zoom
        if self.zoom_factor != 1.0:
            new_width = int(pixmap.width() * self.zoom_factor)
            new_height = int(pixmap.height() * self.zoom_factor)
            pixmap = pixmap.scaled(new_width, new_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        # Vẽ contour lên pixmap
        if self.contours:
            # Tạo bản sao để vẽ lên
            painter = QPainter(pixmap)
            
            # Vẽ các contour cho lát cắt hiện tại
            for name, contour_list in self.contours.items():
                if not self.contour_visibility.get(name, True):
                    continue
                
                for idx, points in contour_list:
                    if idx == self.slice_idx:
                        color = self.contour_colors.get(name, QColor(255, 0, 0, 128))
                        pen = QPen(color)
                        pen.setWidth(2)
                        painter.setPen(pen)
                        
                        # Vẽ đường biên
                        path = QPainterPath()
                        if points is not None and len(points) > 0:
                            # Áp dụng zoom và pan cho các điểm contour
                            zoomed_points = []
                            for x, y in points:
                                zoomed_x = x * self.zoom_factor + self.pan_offset[0]
                                zoomed_y = y * self.zoom_factor + self.pan_offset[1]
                                zoomed_points.append((zoomed_x, zoomed_y))
                            
                            # Bắt đầu đường biên
                            path.moveTo(zoomed_points[0][0], zoomed_points[0][1])
                            
                            # Thêm các điểm còn lại
                            for x, y in zoomed_points[1:]:
                                path.lineTo(x, y)
                            
                            # Đóng đường biên
                            path.lineTo(zoomed_points[0][0], zoomed_points[0][1])
                            
                            # Vẽ đường biên
                            painter.drawPath(path)
                            
                            # Tô màu contour (nếu cần)
                            brush = QBrush(QColor(color.red(), color.green(), color.blue(), 40))
                            painter.fillPath(path, brush)
            
            painter.end()
        
        # Vẽ dữ liệu liều lượng lên pixmap
        if self.dose_data is not None:
            # Tạo bản sao để vẽ lên
            painter = QPainter(pixmap)
            
            # Vẽ dữ liệu liều lượng cho lát cắt hiện tại
            if self.slice_idx >= 0 and self.slice_idx < self.dose_data.shape[0]:
                dose_slice = self.dose_data[self.slice_idx, :, :]
                if dose_slice.size > 0 and dose_slice.ndim >= 2:
                    dose_min, dose_max = np.min(dose_slice), np.max(dose_slice)
                    dose_range = dose_max - dose_min
                    
                    # Áp dụng colormap
                    if dose_range > 0:
                        dose_normalized = (dose_slice - dose_min) / dose_range
                    else:
                        dose_normalized = np.zeros_like(dose_slice)
                    
                    # Vẽ dữ liệu liều lượng
                    for y in range(dose_slice.shape[0]):
                        for x in range(dose_slice.shape[1]):
                            dose_value = dose_normalized[y, x]
                            # Chuyển đổi sang giá trị int trước khi tạo QColor
                            r = int(255 * dose_value)
                            color = QColor(r, 0, 0)  # Màu đỏ cho liều lượng
                            painter.setPen(QPen(color))
                            # Chuyển tọa độ về kiểu int trước khi vẽ
                            px = int(x * self.zoom_factor + self.pan_offset[0])
                            py = int(y * self.zoom_factor + self.pan_offset[1])
                            painter.drawPoint(px, py)
            
            painter.end()
        
        # Hiển thị pixmap
        self.image_frame.setPixmap(pixmap)
    
    def update_display(self):
        """
        Bí danh công khai cho phương thức _update_display.
        Cập nhật hiển thị hình ảnh dựa trên dữ liệu hiện tại.
        """
        self._update_display()
    
    def mousePressEvent(self, event):
        """
        Xử lý sự kiện khi nhấn chuột.
        
        Parameters
        ----------
        event : QMouseEvent
            Sự kiện chuột
        """
        if self.image_data is None:
            return
        
        # Lấy tọa độ chuột
        pos = event.pos()
        
        # Chuyển đổi tọa độ chuột thành tọa độ hình ảnh
        frame_pos = self.image_frame.mapFrom(self, pos)
        pixmap_pos = self.image_frame.mapFromParent(frame_pos)
        
        # Tính toán tọa độ thực tế trên hình ảnh (bỏ qua zoom và pan)
        image_x = int((pixmap_pos.x() - self.pan_offset[0]) / self.zoom_factor)
        image_y = int((pixmap_pos.y() - self.pan_offset[1]) / self.zoom_factor)
        
        # Phát tín hiệu với vị trí
        self.position_changed.emit(self.slice_idx, image_x, image_y)
        self.mouse_pressed.emit(self.slice_idx, image_x, image_y, event.button())
    
    def mouseMoveEvent(self, event):
        """
        Xử lý sự kiện di chuyển chuột.
        
        Parameters
        ----------
        event : QMouseEvent
            Sự kiện chuột
        """
        if self.image_data is None:
            return
        
        # Lấy tọa độ chuột
        pos = event.pos()
        
        # Chuyển đổi tọa độ chuột thành tọa độ hình ảnh
        frame_pos = self.image_frame.mapFrom(self, pos)
        pixmap_pos = self.image_frame.mapFromParent(frame_pos)
        
        # Tính toán tọa độ thực tế trên hình ảnh (bỏ qua zoom và pan)
        image_x = int((pixmap_pos.x() - self.pan_offset[0]) / self.zoom_factor)
        image_y = int((pixmap_pos.y() - self.pan_offset[1]) / self.zoom_factor)
        
        # Phát tín hiệu với vị trí
        self.mouse_moved.emit(self.slice_idx, image_x, image_y)
    
    def mouseReleaseEvent(self, event):
        """
        Xử lý sự kiện khi chuột được thả ra.
        
        Parameters
        ----------
        event : QMouseEvent
            Sự kiện chuột
        """
        if self.image_data is None:
            return
        
        # Lấy tọa độ chuột
        pos = event.pos()
        
        # Chuyển đổi tọa độ chuột thành tọa độ hình ảnh
        frame_pos = self.image_frame.mapFrom(self, pos)
        pixmap_pos = self.image_frame.mapFromParent(frame_pos)
        
        # Tính toán tọa độ thực tế trên hình ảnh (bỏ qua zoom và pan)
        image_x = int((pixmap_pos.x() - self.pan_offset[0]) / self.zoom_factor)
        image_y = int((pixmap_pos.y() - self.pan_offset[1]) / self.zoom_factor)
        
        # Phát tín hiệu với vị trí
        self.mouse_released.emit(self.slice_idx, image_x, image_y, event.button())
    
    def keyPressEvent(self, event):
        """
        Xử lý sự kiện nhấn phím.
        
        Parameters
        ----------
        event : QKeyEvent
            Sự kiện phím
        """
        self.key_pressed.emit(event.key())
    
    def keyReleaseEvent(self, event):
        """
        Xử lý sự kiện thả phím.
        
        Parameters
        ----------
        event : QKeyEvent
            Sự kiện phím
        """
        self.key_released.emit(event.key())
    
    def set_brightness(self, value):
        """
        Thiết lập độ sáng cho hình ảnh.
        
        Parameters
        ----------
        value : int
            Giá trị độ sáng
        """
        # Điều chỉnh cửa sổ Hounsfield dựa trên độ sáng
        window_min, window_max = self.hounsfield_window
        window_center = (window_min + window_max) // 2
        
        # Áp dụng độ sáng bằng cách thay đổi tâm cửa sổ Hounsfield
        new_center = window_center + value
        window_width = window_max - window_min
        
        # Thiết lập cửa sổ Hounsfield mới
        self.set_hounsfield_window(new_center - window_width // 2, new_center + window_width // 2)
    
    def set_contrast(self, value):
        """
        Thiết lập độ tương phản cho hình ảnh.
        
        Parameters
        ----------
        value : int
            Giá trị độ tương phản
        """
        # Điều chỉnh cửa sổ Hounsfield dựa trên độ tương phản
        window_min, window_max = self.hounsfield_window
        window_center = (window_min + window_max) // 2
        window_width = window_max - window_min
        
        # Áp dụng độ tương phản bằng cách thay đổi độ rộng cửa sổ Hounsfield
        # Giá trị value càng cao, độ tương phản càng thấp (cửa sổ rộng hơn)
        # Giá trị value càng thấp, độ tương phản càng cao (cửa sổ hẹp hơn)
        new_width = max(10, window_width + value * 10)  # Đảm bảo cửa sổ không quá hẹp
        
        # Thiết lập cửa sổ Hounsfield mới
        self.set_hounsfield_window(window_center - new_width // 2, window_center + new_width // 2)
    
    def set_background_data(self, image_data):
        """
        Thiết lập dữ liệu nền (hình ảnh giải phẫu) cho hiển thị.
        
        Parameters
        ----------
        image_data : ndarray
            Dữ liệu hình ảnh 2D (y, x) kiểu numpy array
        """
        if image_data is None:
            return
            
        # Lưu dữ liệu hình ảnh nền vào bộ nhớ
        if len(image_data.shape) == 2:
            # Nếu là lát cắt 2D, tạo tập dữ liệu 3D với 1 lát cắt
            self.image_data = np.expand_dims(image_data, axis=0)
        else:
            # Nếu đã là dữ liệu 3D thì sử dụng trực tiếp
            self.image_data = image_data
            
        # Thiết lập chỉ số lát cắt mặc định
        self.slice_idx = 0
        
        # Cập nhật hiển thị
        self._update_display()


class ImageControlWidget(QWidget):
    """Widget để điều khiển hiển thị hình ảnh."""
    
    # Tín hiệu
    view_changed = pyqtSignal(str)  # 'axial', 'coronal', 'sagittal'
    slice_changed = pyqtSignal(int)
    window_changed = pyqtSignal(int, int)  # min, max
    brightness_changed = pyqtSignal(int)  # brightness value
    contrast_changed = pyqtSignal(int)  # contrast value
    
    def __init__(self, parent=None):
        """
        Khởi tạo widget điều khiển hình ảnh.
        
        Parameters
        ----------
        parent : QWidget, optional
            Widget cha
        """
        super().__init__(parent)
        
        # Dữ liệu
        self.max_slice = 0
        
        # UI
        self._init_ui()
    
    def _init_ui(self):
        """Khởi tạo giao diện người dùng."""
        # Layout chính
        self.setLayout(QVBoxLayout())
        
        # Nhóm điều khiển mặt phẳng
        self.plane_group = QGroupBox("Mặt phẳng")
        self.plane_layout = QHBoxLayout(self.plane_group)
        
        # Radio buttons mặt phẳng
        self.plane_group_buttons = QButtonGroup(self)
        
        self.axial_radio = QRadioButton("Axial")
        self.axial_radio.setChecked(True)
        self.axial_radio.toggled.connect(self._plane_changed)
        self.plane_group_buttons.addButton(self.axial_radio)
        self.plane_layout.addWidget(self.axial_radio)
        
        self.coronal_radio = QRadioButton("Coronal")
        self.coronal_radio.toggled.connect(self._plane_changed)
        self.plane_group_buttons.addButton(self.coronal_radio)
        self.plane_layout.addWidget(self.coronal_radio)
        
        self.sagittal_radio = QRadioButton("Sagittal")
        self.sagittal_radio.toggled.connect(self._plane_changed)
        self.plane_group_buttons.addButton(self.sagittal_radio)
        self.plane_layout.addWidget(self.sagittal_radio)
        
        self.layout().addWidget(self.plane_group)
        
        # Nhóm điều khiển lát cắt
        self.slice_group = QGroupBox("Lát cắt")
        self.slice_layout = QVBoxLayout(self.slice_group)
        
        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.setRange(0, 100)
        self.slice_slider.setValue(50)
        self.slice_slider.valueChanged.connect(self._slice_changed)
        self.slice_layout.addWidget(self.slice_slider)
        
        self.slice_label = QLabel("Lát cắt: 50/100")
        self.slice_label.setAlignment(Qt.AlignCenter)
        self.slice_layout.addWidget(self.slice_label)
        
        self.layout().addWidget(self.slice_group)
        
        # Nhóm điều khiển cửa sổ Hounsfield
        self.window_group = QGroupBox("Cửa sổ Hounsfield")
        self.window_layout = QFormLayout(self.window_group)
        
        self.window_min_spin = QDoubleSpinBox()
        self.window_min_spin.setRange(-1000, 3000)
        self.window_min_spin.setValue(0)
        self.window_min_spin.valueChanged.connect(self._window_changed)
        self.window_layout.addRow("Min:", self.window_min_spin)
        
        self.window_max_spin = QDoubleSpinBox()
        self.window_max_spin.setRange(-1000, 3000)
        self.window_max_spin.setValue(2000)
        self.window_max_spin.valueChanged.connect(self._window_changed)
        self.window_layout.addRow("Max:", self.window_max_spin)
        
        # Presets
        self.preset_layout = QHBoxLayout()
        
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(["Tùy chỉnh", "Bộ lọc phổi", "Bộ lọc xương", "Bộ lọc mô mềm"])
        self.preset_combo.currentIndexChanged.connect(self._preset_changed)
        self.preset_layout.addWidget(QLabel("Bộ lọc:"))
        self.preset_layout.addWidget(self.preset_combo)
        
        self.window_layout.addRow(self.preset_layout)
        
        self.layout().addWidget(self.window_group)
        
        # Nhóm điều khiển contour
        self.contour_group = QGroupBox("Contour")
        self.contour_layout = QVBoxLayout(self.contour_group)
        
        # Checkbox hiển thị contour
        self.show_contours_check = QCheckBox("Hiển thị contour")
        self.show_contours_check.setChecked(True)
        self.contour_layout.addWidget(self.show_contours_check)
        
        self.layout().addWidget(self.contour_group)
    
    def set_max_slice(self, max_slice):
        """
        Thiết lập số lượng lát cắt tối đa.
        
        Parameters
        ----------
        max_slice : int
            Số lượng lát cắt tối đa
        """
        self.max_slice = max_slice
        self.slice_slider.setRange(0, max_slice)
        self.slice_slider.setValue(max_slice // 2)
        self._update_slice_label()
    
    def set_current_slice(self, slice_idx):
        """
        Thiết lập lát cắt hiện tại.
        
        Parameters
        ----------
        slice_idx : int
            Chỉ số lát cắt
        """
        self.slice_slider.setValue(slice_idx)
    
    def _plane_changed(self):
        """Xử lý sự kiện khi mặt phẳng thay đổi."""
        if self.axial_radio.isChecked():
            self.view_changed.emit('axial')
        elif self.coronal_radio.isChecked():
            self.view_changed.emit('coronal')
        else:
            self.view_changed.emit('sagittal')
    
    def _slice_changed(self, value):
        """
        Xử lý sự kiện khi lát cắt thay đổi.
        
        Parameters
        ----------
        value : int
            Giá trị lát cắt mới
        """
        self._update_slice_label()
        self.slice_changed.emit(value)
    
    def _update_slice_label(self):
        """Cập nhật nhãn hiển thị lát cắt."""
        self.slice_label.setText(f"Lát cắt: {self.slice_slider.value()}/{self.max_slice}")
    
    def _window_changed(self):
        """Xử lý sự kiện khi cửa sổ Hounsfield thay đổi."""
        window_min = int(self.window_min_spin.value())
        window_max = int(self.window_max_spin.value())
        
        # Đảm bảo min < max
        if window_min >= window_max:
            window_max = window_min + 1
            self.window_max_spin.setValue(window_max)
        
        self.window_changed.emit(window_min, window_max)
    
    def _preset_changed(self, index):
        """
        Xử lý sự kiện khi preset thay đổi.
        
        Parameters
        ----------
        index : int
            Chỉ số preset
        """
        # Thiết lập các giá trị cửa sổ Hounsfield dựa trên preset
        if index == 1:  # Bộ lọc phổi
            self.window_min_spin.setValue(-1000)
            self.window_max_spin.setValue(200)
        elif index == 2:  # Bộ lọc xương
            self.window_min_spin.setValue(250)
            self.window_max_spin.setValue(3000)
        elif index == 3:  # Bộ lọc mô mềm
            self.window_min_spin.setValue(-125)
            self.window_max_spin.setValue(225)
        
        # Không cần thiết lập giá trị khi index == 0 (Tùy chỉnh)


class ImageDisplay(QWidget):
    """
    Widget hiển thị hình ảnh y tế với các công cụ tương tác.
    
    Cung cấp hiển thị cơ bản của hình ảnh y tế, điều khiển cửa sổ,
    zoom, pan, và các chức năng tương tác khác.
    """
    
    # Tín hiệu
    window_changed = pyqtSignal(int, int)  # window width, window level
    position_changed = pyqtSignal(int, int)  # x, y coordinates
    mouse_position_changed = pyqtSignal(int, int, float)  # x, y, giá trị tại vị trí
    
    # Thêm các tín hiệu cần thiết cho tương tác với công cụ contour
    mouse_pressed = pyqtSignal(int, int, int, Qt.MouseButton)  # slice_idx, x, y, button
    mouse_moved = pyqtSignal(int, int, int)  # slice_idx, x, y
    mouse_released = pyqtSignal(int, int, int, Qt.MouseButton)  # slice_idx, x, y, button
    key_pressed = pyqtSignal(int)  # key
    key_released = pyqtSignal(int)  # key
    
    def __init__(self, parent=None):
        """Khởi tạo ImageDisplay."""
        super().__init__(parent)
        
        # Dữ liệu
        self.image_data = None
        self.window_width = 400
        self.window_level = 50
        self.zoom_factor = 1.0
        self.pan_offset = (0, 0)
        self.current_tool = "pan"  # pan, zoom, measure, window
        self.title = ""  # Thêm biến title để lưu tiêu đề
        self.handle_wheel_event = False  # Mặc định không xử lý sự kiện wheel, để xử lý ở cấp cao hơn
        
        # UI setup
        self._init_ui()
        
        # Kích thước tối thiểu
        self.setMinimumSize(200, 200)
        
    def _init_ui(self):
        """Khởi tạo giao diện người dùng."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Frame hiển thị hình ảnh
        self.image_frame = QLabel(self)
        self.image_frame.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_frame.setStyleSheet("background-color: black;")
        layout.addWidget(self.image_frame)
        
        self.setLayout(layout)
        
    def set_image(self, image_data, window_center=None, window_width=None, plane=None):
        """
        Thiết lập dữ liệu hình ảnh mới.
        
        Parameters
        ----------
        image_data : ndarray
            Dữ liệu hình ảnh 3D (z, y, x)
        window_center : int, optional
            Tâm cửa sổ Hounsfield
        window_width : int, optional
            Độ rộng cửa sổ Hounsfield
        plane : str, optional
            Mặt phẳng xem ('axial', 'coronal', 'sagittal')
        """
        # Thiết lập dữ liệu hình ảnh
        self.image_data = image_data
        
        # Lưu thông tin mặt phẳng xem nếu được cung cấp
        if plane is not None and plane in ['axial', 'coronal', 'sagittal']:
            self.view_plane = plane
        elif not hasattr(self, 'view_plane'):
            self.view_plane = 'axial'  # Mặt phẳng mặc định
        
        # Thiết lập cửa sổ nếu được cung cấp
        if window_center is not None and window_width is not None:
            window_min = window_center - window_width // 2
            window_max = window_center + window_width // 2
            self.set_window(window_max - window_min, (window_max + window_min) // 2)
        
        # Cập nhật hiển thị
        self._update_display()
        
    def set_window(self, width, level):
        """
        Thiết lập cửa sổ hiển thị.
        
        Args:
            width: Chiều rộng cửa sổ
            level: Tâm cửa sổ
        """
        self.window_width = width
        self.window_level = level
        self._update_display()
        
        # Phát tín hiệu
        self.window_changed.emit(width, level)
        
    def set_zoom(self, factor):
        """
        Thiết lập mức độ zoom.
        
        Args:
            factor: Hệ số zoom
        """
        self.zoom_factor = max(0.1, factor)
        self._update_display()
        
    def set_pan(self, dx, dy):
        """
        Thiết lập độ dịch chuyển.
        
        Args:
            dx: Độ dịch chuyển theo trục x
            dy: Độ dịch chuyển theo trục y
        """
        self.pan_offset = (dx, dy)
        self._update_display()
        
    def _update_display(self):
        """Cập nhật hiển thị hình ảnh."""
        if self.image_data is None:
            # Hiển thị tiêu đề trên nền đen nếu không có hình ảnh
            if self.title:
                try:
                    blank_image = np.zeros((200, 200), dtype=np.uint8)
                    h, w = blank_image.shape
                    qimage = QImage(blank_image.data, w, h, w, QImage.Format_Grayscale8)
                    pixmap = QPixmap.fromImage(qimage)
                    
                    # Vẽ tiêu đề lên pixmap
                    painter = QPainter(pixmap)
                    painter.setPen(QPen(Qt.white))
                    font = QFont()
                    font.setBold(True)
                    font.setPointSize(12)
                    painter.setFont(font)
                    painter.drawText(pixmap.rect(), Qt.AlignCenter, self.title)
                    painter.end()
                    
                    self.image_frame.setPixmap(pixmap)
                    logger.info(f"Hiển thị tiêu đề '{self.title}' trên nền đen")
                except Exception as e:
                    logger.error(f"Lỗi khi hiển thị tiêu đề trên nền đen: {str(e)}")
            else:
                logger.debug("Không có dữ liệu hình ảnh và tiêu đề để hiển thị")
            return
        
        try:
            # Kiểm tra tính hợp lệ của dữ liệu hình ảnh
            if not isinstance(self.image_data, np.ndarray):
                logger.error(f"Dữ liệu hình ảnh không phải là mảng NumPy: {type(self.image_data)}")
                if self.title:
                    # Hiển thị thông báo lỗi
                    blank_image = np.zeros((200, 200), dtype=np.uint8)
                    h, w = blank_image.shape
                    qimage = QImage(blank_image.data, w, h, w, QImage.Format_Grayscale8)
                    pixmap = QPixmap.fromImage(qimage)
                    painter = QPainter(pixmap)
                    painter.setPen(QPen(Qt.red))
                    font = QFont()
                    font.setBold(True)
                    font.setPointSize(10)
                    painter.setFont(font)
                    error_message = f"{self.title if self.title else 'Error'}\nLỗi: Dữ liệu không hợp lệ"
                    painter.drawText(pixmap.rect(), Qt.AlignCenter, error_message)
                    painter.end()
                    self.image_frame.setPixmap(pixmap)
                return
            
            if self.image_data.size == 0:
                logger.error("Dữ liệu hình ảnh rỗng")
                # Hiển thị thông báo lỗi tương tự
                if self.title:
                    blank_image = np.zeros((200, 200), dtype=np.uint8)
                    h, w = blank_image.shape
                    qimage = QImage(blank_image.data, w, h, w, QImage.Format_Grayscale8)
                    pixmap = QPixmap.fromImage(qimage)
                    painter = QPainter(pixmap)
                    painter.setPen(QPen(Qt.red))
                    font = QFont()
                    font.setBold(True)
                    font.setPointSize(10)
                    painter.setFont(font)
                    error_message = f"{self.title if self.title else 'Error'}\nLỗi: Dữ liệu rỗng"
                    painter.drawText(pixmap.rect(), Qt.AlignCenter, error_message)
                    painter.end()
                    self.image_frame.setPixmap(pixmap)
                return
            
            if self.image_data.ndim < 2:
                logger.error(f"Dữ liệu hình ảnh cần ít nhất 2 chiều, nhưng có {self.image_data.ndim} chiều")
                # Hiển thị thông báo lỗi tương tự
                if self.title:
                    blank_image = np.zeros((200, 200), dtype=np.uint8)
                    h, w = blank_image.shape
                    qimage = QImage(blank_image.data, w, h, w, QImage.Format_Grayscale8)
                    pixmap = QPixmap.fromImage(qimage)
                    painter = QPainter(pixmap)
                    painter.setPen(QPen(Qt.red))
                    font = QFont()
                    font.setBold(True)
                    font.setPointSize(10)
                    painter.setFont(font)
                    error_message = f"{self.title if self.title else 'Error'}\nLỗi: Chiều không hợp lệ ({self.image_data.ndim}D)"
                    painter.drawText(pixmap.rect(), Qt.AlignCenter, error_message)
                    painter.end()
                    self.image_frame.setPixmap(pixmap)
                return
            
            # Ghi log thông tin hình ảnh
            logger.debug(f"Đang hiển thị hình ảnh: shape={self.image_data.shape}, "
                       f"dtype={self.image_data.dtype}, "
                       f"min={self.image_data.min()}, max={self.image_data.max()}, "
                       f"window: {self.window_level}±{self.window_width//2}")
            
            # Áp dụng cửa sổ
            min_val = self.window_level - self.window_width // 2
            max_val = self.window_level + self.window_width // 2
            
            # Đảm bảo min_val < max_val
            if min_val >= max_val:
                logger.warning(f"Window không hợp lệ: min={min_val}, max={max_val}, điều chỉnh để có khoảng cách tối thiểu")
                min_val = self.window_level - 1
                max_val = self.window_level + 1
            
            # Clip giá trị trong khoảng [min_val, max_val]
            try:
                # Đảm bảo dữ liệu là kiểu số thực để tránh overflow
                if np.issubdtype(self.image_data.dtype, np.integer) and (min_val < np.iinfo(self.image_data.dtype).min or max_val > np.iinfo(self.image_data.dtype).max):
                    logger.debug(f"Chuyển đổi dữ liệu từ {self.image_data.dtype} sang float32 để tránh tràn số")
                    image_data_float = self.image_data.astype(np.float32)
                    clipped = np.clip(image_data_float, min_val, max_val)
                else:
                    clipped = np.clip(self.image_data, min_val, max_val)
            except Exception as e:
                logger.error(f"Lỗi khi clip dữ liệu hình ảnh: {str(e)}")
                # Thử sửa lỗi bằng cách chuyển đổi kiểu dữ liệu
                try:
                    logger.debug("Thử chuyển đổi sang float32 để xử lý")
                    image_data_float = self.image_data.astype(np.float32)
                    clipped = np.clip(image_data_float, min_val, max_val)
                    logger.info("Đã chuyển đổi thành công sang float32 để xử lý")
                except Exception as e2:
                    logger.error(f"Vẫn không thể clip sau khi chuyển đổi: {str(e2)}")
                    # Hiển thị thông báo lỗi
                    if self.title:
                        blank_image = np.zeros((200, 200), dtype=np.uint8)
                        h, w = blank_image.shape
                        qimage = QImage(blank_image.data, w, h, w, QImage.Format_Grayscale8)
                        pixmap = QPixmap.fromImage(qimage)
                        painter = QPainter(pixmap)
                        painter.setPen(QPen(Qt.red))
                        painter.drawText(pixmap.rect(), Qt.AlignCenter, f"{self.title}\nLỗi xử lý dữ liệu")
                        painter.end()
                        self.image_frame.setPixmap(pixmap)
                    return
            
            # Chuẩn hóa về khoảng [0, 255]
            try:
                if max_val > min_val:
                    normalized = ((clipped - min_val) / (max_val - min_val) * 255).astype(np.uint8)
                else:
                    normalized = np.zeros_like(clipped, dtype=np.uint8)
                    logger.warning("Khoảng cửa sổ quá nhỏ, tạo ảnh đen")
            except Exception as e:
                logger.error(f"Lỗi khi chuẩn hóa dữ liệu hình ảnh: {str(e)}")
                # Thử phương pháp chuẩn hóa khác
                try:
                    # Kiểm tra giá trị tối đa và tối thiểu
                    if np.isfinite(clipped).all():
                        # Nếu tất cả giá trị hữu hạn
                        range_val = max_val - min_val
                        if range_val > 0:
                            normalized = np.round(((clipped - min_val) / range_val) * 255).astype(np.uint8)
                        else:
                            normalized = np.zeros_like(clipped, dtype=np.uint8)
                            logger.warning("Khoảng cửa sổ bằng 0, tạo ảnh đen")
                    else:
                        # Xử lý giá trị NaN/Inf
                        logger.warning("Dữ liệu có giá trị NaN/Inf, thay thế bằng 0")
                        clipped_finite = np.copy(clipped)
                        clipped_finite[~np.isfinite(clipped_finite)] = 0
                        range_val = max_val - min_val
                        if range_val > 0:
                            normalized = np.round(((clipped_finite - min_val) / range_val) * 255).astype(np.uint8)
                        else:
                            normalized = np.zeros_like(clipped_finite, dtype=np.uint8)
                    
                    logger.info("Đã sử dụng phương pháp chuẩn hóa thay thế")
                except Exception as e2:
                    logger.error(f"Vẫn không thể chuẩn hóa: {str(e2)}")
                    # Hiển thị thông báo lỗi
                    if self.title:
                        blank_image = np.zeros((200, 200), dtype=np.uint8)
                        h, w = blank_image.shape
                        qimage = QImage(blank_image.data, w, h, w, QImage.Format_Grayscale8)
                        pixmap = QPixmap.fromImage(qimage)
                        painter = QPainter(pixmap)
                        painter.setPen(QPen(Qt.red))
                        painter.drawText(pixmap.rect(), Qt.AlignCenter, f"{self.title}\nLỗi chuẩn hóa")
                        painter.end()
                        self.image_frame.setPixmap(pixmap)
                    return
            
            # Kiểm tra kích thước dữ liệu
            h, w = normalized.shape
            if h <= 0 or w <= 0:
                logger.error(f"Kích thước hình ảnh không hợp lệ: {h}x{w}")
                # Hiển thị thông báo lỗi
                if self.title:
                    blank_image = np.zeros((200, 200), dtype=np.uint8)
                    h, w = blank_image.shape
                    qimage = QImage(blank_image.data, w, h, w, QImage.Format_Grayscale8)
                    pixmap = QPixmap.fromImage(qimage)
                    painter = QPainter(pixmap)
                    painter.setPen(QPen(Qt.red))
                    painter.drawText(pixmap.rect(), Qt.AlignCenter, f"{self.title}\nKích thước không hợp lệ")
                    painter.end()
                    self.image_frame.setPixmap(pixmap)
                return
            
            # Chuyển sang định dạng QImage
            try:
                # Sao chép dữ liệu để đảm bảo nó liên tục trong bộ nhớ
                normalized_copy = normalized.copy(order='C')
                qimage = QImage(normalized_copy.data, w, h, w, QImage.Format_Grayscale8)
                pixmap = QPixmap.fromImage(qimage)
                
                # Kiểm tra xem QImage có hợp lệ không
                if qimage.isNull():
                    raise ValueError("QImage tạo ra là null")
                    
                # Kiểm tra kích thước QPixmap
                if pixmap.width() <= 0 or pixmap.height() <= 0:
                    raise ValueError(f"QPixmap có kích thước không hợp lệ: {pixmap.width()}x{pixmap.height()}")
                
            except Exception as e:
                logger.error(f"Lỗi khi tạo QImage/QPixmap: {str(e)}")
                # Thử phương pháp khác
                try:
                    logger.debug("Thử tạo QImage với phương pháp thay thế")
                    # Đảm bảo dữ liệu liên tục và đúng thứ tự byte
                    data_copy = np.ascontiguousarray(normalized)
                    
                    # Tạo QImage trực tiếp từ mảng NumPy
                    bytes_per_line = w
                    qimage = QImage(data_copy.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
                    
                    # Tạo bản sao của QImage để đảm bảo dữ liệu được sao chép
                    qimage = qimage.copy()
                    
                    # Tạo QPixmap từ QImage
                    pixmap = QPixmap.fromImage(qimage)
                    
                    logger.info("Đã tạo thành công QImage/QPixmap bằng phương pháp thay thế")
                except Exception as e2:
                    logger.error(f"Vẫn không thể tạo QImage/QPixmap: {str(e2)}")
                    # Hiển thị thông báo lỗi
                    if self.title:
                        blank_image = np.zeros((200, 200), dtype=np.uint8)
                        h, w = blank_image.shape
                        qimage = QImage(blank_image.data, w, h, w, QImage.Format_Grayscale8)
                        pixmap = QPixmap.fromImage(qimage)
                        painter = QPainter(pixmap)
                        painter.setPen(QPen(Qt.red))
                        painter.drawText(pixmap.rect(), Qt.AlignCenter, f"{self.title}\nLỗi hiển thị")
                        painter.end()
                        self.image_frame.setPixmap(pixmap)
                    return
            
            # Áp dụng zoom nếu cần
            if self.zoom_factor != 1.0:
                w_zoomed = int(w * self.zoom_factor)
                h_zoomed = int(h * self.zoom_factor)
                if w_zoomed > 0 and h_zoomed > 0:
                    try:
                        pixmap = pixmap.scaled(w_zoomed, h_zoomed, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    except Exception as e:
                        logger.error(f"Lỗi khi áp dụng zoom: {str(e)}")
                        # Thử với phương pháp không làm mượt
                        try:
                            pixmap = pixmap.scaled(w_zoomed, h_zoomed, Qt.KeepAspectRatio, Qt.FastTransformation)
                            logger.info("Đã áp dụng zoom với phương pháp FastTransformation")
                        except Exception:
                            logger.warning("Không thể áp dụng zoom, tiếp tục mà không có zoom")
            
            # Vẽ tiêu đề nếu có
            if self.title:
                try:
                    painter = QPainter(pixmap)
                    painter.setPen(QPen(Qt.white))
                    
                    # Tạo font đậm cho tiêu đề
                    font = QFont()
                    font.setBold(True)
                    font.setPointSize(10)
                    painter.setFont(font)
                    
                    # Vẽ nền chữ bán trong suốt
                    rect = QRectF(5, 5, pixmap.width() - 10, 25)
                    painter.fillRect(rect, QBrush(QColor(0, 0, 0, 128)))
                    
                    # Vẽ chữ
                    painter.drawText(rect, Qt.AlignCenter, self.title)
                    painter.end()
                except Exception as e:
                    logger.warning(f"Lỗi khi vẽ tiêu đề: {str(e)}")
                    # Tiếp tục mà không có tiêu đề
            
            # Hiển thị pixmap
            self.image_frame.setPixmap(pixmap)
            
            logger.debug("Hiển thị hình ảnh thành công")
            
        except Exception as e:
            logger.error(f"Lỗi không xác định khi cập nhật hiển thị hình ảnh: {str(e)}")
            import traceback
            logger.debug(traceback.format_exc())
            
            # Hiển thị thông báo lỗi
            try:
                blank_image = np.zeros((200, 200), dtype=np.uint8)
                h, w = blank_image.shape
                qimage = QImage(blank_image.data, w, h, w, QImage.Format_Grayscale8)
                pixmap = QPixmap.fromImage(qimage)
                painter = QPainter(pixmap)
                painter.setPen(QPen(Qt.red))
                font = QFont()
                font.setBold(True)
                font.setPointSize(10)
                painter.setFont(font)
                error_message = f"{self.title if self.title else 'Error'}\nLỗi hiển thị hình ảnh"
                painter.drawText(pixmap.rect(), Qt.AlignCenter, error_message)
                painter.end()
                self.image_frame.setPixmap(pixmap)
            except Exception:
                logger.error("Không thể hiển thị thông báo lỗi")
    
    def mousePressEvent(self, event):
        """Xử lý sự kiện nhấn chuột."""
        # Lưu vị trí bắt đầu
        self.last_pos = event.pos()
        
        # Chuyển đổi tọa độ từ vị trí màn hình sang vị trí trong ảnh
        img_x, img_y = self._screen_to_image_coords(event.pos().x(), event.pos().y())
        
        # Xử lý công cụ hiện tại
        if self.current_tool == "pan":
            # Không làm gì đặc biệt, chỉ lưu vị trí bắt đầu
            pass
        elif self.current_tool == "window":
            # Lưu giá trị cửa sổ ban đầu
            self.initial_window_width = self.window_width
            self.initial_window_center = self.window_level
        
        # Phát tín hiệu cho các công cụ contour
        slice_idx = 0  # Giả sử slice_idx là 0, cần cập nhật theo dữ liệu thực tế
        if hasattr(self, 'current_slice_idx'):
            slice_idx = self.current_slice_idx
        self.mouse_pressed.emit(slice_idx, img_x, img_y, event.button())

    def mouseMoveEvent(self, event):
        """Xử lý sự kiện di chuyển chuột."""
        if not self.last_pos:
            self.last_pos = event.pos()
            return
            
        # Chuyển đổi tọa độ từ vị trí màn hình sang vị trí trong ảnh
        img_x, img_y = self._screen_to_image_coords(event.pos().x(), event.pos().y())
        
        # Tính toán vị trí
        dx = event.x() - self.last_pos.x()
        dy = event.y() - self.last_pos.y()
        
        # Xử lý dựa trên công cụ hiện tại
        if event.buttons() & Qt.LeftButton:
            if self.current_tool == "pan":
                # Xử lý pan
                self.pan_offset = (self.pan_offset[0] + dx, self.pan_offset[1] + dy)
                self._update_display()
            elif self.current_tool == "window":
                # Xử lý thay đổi cửa sổ
                # dx thay đổi width, dy thay đổi center
                new_width = max(1, self.initial_window_width + dx)
                new_center = self.initial_window_center - dy
                self.set_window(new_width, new_center)
        
        # Phát tín hiệu cho các công cụ contour
        slice_idx = 0  # Giả sử slice_idx là 0, cần cập nhật theo dữ liệu thực tế
        if hasattr(self, 'current_slice_idx'):
            slice_idx = self.current_slice_idx
        self.mouse_moved.emit(slice_idx, img_x, img_y)
        
        # Cập nhật vị trí
        self.last_pos = event.pos()
        
        # Thông báo vị trí chuột
        self.mouse_position_changed.emit(img_x, img_y, self._get_pixel_value(img_x, img_y))

    def mouseReleaseEvent(self, event):
        """Xử lý sự kiện thả chuột."""
        # Chuyển đổi tọa độ từ vị trí màn hình sang vị trí trong ảnh
        img_x, img_y = self._screen_to_image_coords(event.pos().x(), event.pos().y())
        
        # Phát tín hiệu cho các công cụ contour
        slice_idx = 0  # Giả sử slice_idx là 0, cần cập nhật theo dữ liệu thực tế
        if hasattr(self, 'current_slice_idx'):
            slice_idx = self.current_slice_idx
        self.mouse_released.emit(slice_idx, img_x, img_y, event.button())
        
        self.last_pos = None
        
    def keyPressEvent(self, event):
        """Xử lý sự kiện nhấn phím."""
        # Phát tín hiệu cho các công cụ contour
        self.key_pressed.emit(event.key())
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        """Xử lý sự kiện thả phím."""
        # Phát tín hiệu cho các công cụ contour
        self.key_released.emit(event.key())
        super().keyReleaseEvent(event)

    def _screen_to_image_coords(self, x, y):
        """Chuyển đổi tọa độ màn hình sang tọa độ trong ảnh."""
        if not hasattr(self, 'image_rect') or not self.image_rect:
            return 0, 0
            
        # Tọa độ tương đối trong hình chữ nhật hiển thị
        rel_x = (x - self.image_rect.x()) / self.image_rect.width()
        rel_y = (y - self.image_rect.y()) / self.image_rect.height()
        
        # Chuyển đổi sang tọa độ trong ảnh
        if self.image_data is not None:
            img_width = self.image_data.shape[1] if len(self.image_data.shape) > 1 else 1
            img_height = self.image_data.shape[0] if len(self.image_data.shape) > 0 else 1
            
            img_x = int(rel_x * img_width)
            img_y = int(rel_y * img_height)
            
            # Đảm bảo tọa độ nằm trong khoảng hợp lệ
            img_x = max(0, min(img_x, img_width - 1))
            img_y = max(0, min(img_y, img_height - 1))
            
            return img_x, img_y
        
        return 0, 0
        
    def _get_pixel_value(self, x, y):
        """Lấy giá trị pixel tại vị trí x, y."""
        if self.image_data is not None and 0 <= y < self.image_data.shape[0] and 0 <= x < self.image_data.shape[1]:
            return float(self.image_data[y, x])
        return 0.0

    def wheelEvent(self, event):
        """Xử lý sự kiện lăn chuột."""
        # Chỉ xử lý sự kiện wheel nếu handle_wheel_event = True
        if self.handle_wheel_event:
            # Xử lý zoom
            delta = event.angleDelta().y()
            if delta > 0:
                self.set_zoom(self.zoom_factor * 1.1)
            elif delta < 0:
                self.set_zoom(self.zoom_factor / 1.1)
            event.accept()
        else:
            # Nếu không xử lý, để event được truyền lên parent widget
            event.ignore()
        super().wheelEvent(event)
            
    def resizeEvent(self, event):
        """Xử lý sự kiện thay đổi kích thước."""
        super().resizeEvent(event)
        self._update_display()
        
    def set_tool(self, tool_name):
        """
        Thiết lập công cụ hiện tại.
        
        Args:
            tool_name: Tên công cụ ('pan', 'zoom', 'window', 'measure')
        """
        self.current_tool = tool_name
        
        # Nếu công cụ là zoom, bật xử lý sự kiện wheel
        if tool_name == "zoom":
            self.handle_wheel_event = True
        else:
            self.handle_wheel_event = False
        
    def set_title(self, title):
        """
        Thiết lập tiêu đề cho màn hình hiển thị.
        
        Args:
            title: Tiêu đề hiển thị
        """
        self.title = title
        self._update_display()
        
# Kiểm tra các module cần thiết
try:
    import pydicom
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False

try:
    import SimpleITK as sitk
    SITK_AVAILABLE = True
except ImportError:
    SITK_AVAILABLE = False
