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
    
    def set_image_data(self, image_data):
        """
        Thiết lập dữ liệu hình ảnh 3D.
        
        Parameters
        ----------
        image_data : ndarray
            Dữ liệu hình ảnh 3D (z, y, x) kiểu numpy array
        """
        self.image_data = image_data
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
                        if points and len(points) > 0:
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
            dose_slice = self.dose_data[self.slice_idx, :, :]
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
        
    def set_image(self, image_data, window_center=None, window_width=None):
        """
        Thiết lập hình ảnh hiển thị.
        
        Args:
            image_data: Mảng NumPy 2D chứa dữ liệu hình ảnh
            window_center: Tâm cửa sổ (level)
            window_width: Chiều rộng cửa sổ
        """
        self.image_data = image_data
        
        if window_center is not None:
            self.window_level = window_center
        
        if window_width is not None:
            self.window_width = window_width
            
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
            return
            
        # Áp dụng cửa sổ
        min_val = self.window_level - self.window_width // 2
        max_val = self.window_level + self.window_width // 2
        
        # Clip giá trị trong khoảng [min_val, max_val]
        clipped = np.clip(self.image_data, min_val, max_val)
        
        # Chuẩn hóa về khoảng [0, 255]
        if max_val > min_val:
            normalized = ((clipped - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        else:
            normalized = np.zeros_like(clipped, dtype=np.uint8)
        
        # Chuyển sang định dạng RGB
        h, w = normalized.shape
        qimage = QImage(normalized.data, w, h, w, QImage.Format_Grayscale8)
        
        # Tạo pixmap từ qimage
        pixmap = QPixmap.fromImage(qimage)
        
        # Áp dụng zoom nếu cần
        if self.zoom_factor != 1.0:
            w_zoomed = int(w * self.zoom_factor)
            h_zoomed = int(h * self.zoom_factor)
            if w_zoomed > 0 and h_zoomed > 0:
                pixmap = pixmap.scaled(w_zoomed, h_zoomed, Qt.KeepAspectRatio)
        
        # Vẽ tiêu đề nếu có
        if self.title:
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
        
        # Hiển thị pixmap
        self.image_frame.setPixmap(pixmap)
        
    def mousePressEvent(self, event):
        """Xử lý sự kiện nhấn chuột."""
        super().mousePressEvent(event)
        
        if event.button() == Qt.LeftButton:
            # Lấy vị trí trong tọa độ ảnh
            pos = event.pos()
            x = pos.x()
            y = pos.y()
            
            # TODO: Chuyển tọa độ screen sang tọa độ ảnh
            
            # Phát tín hiệu thay đổi vị trí
            self.position_changed.emit(x, y)
            
    def mouseMoveEvent(self, event):
        """Xử lý sự kiện di chuyển chuột."""
        super().mouseMoveEvent(event)
        
        # Lấy vị trí chuột
        pos = event.pos()
        x, y = pos.x(), pos.y()
        
        # Hiệu chỉnh theo zoom và pan
        if self.image_data is not None and self.zoom_factor > 0:
            # Điều chỉnh tọa độ để tính toán vị trí thực trong hình ảnh
            x_img = int((x - self.pan_offset[0]) / self.zoom_factor)
            y_img = int((y - self.pan_offset[1]) / self.zoom_factor)
            
            # Kiểm tra nếu tọa độ nằm trong phạm vi của hình ảnh
            if 0 <= x_img < self.image_data.shape[1] and 0 <= y_img < self.image_data.shape[0]:
                # Lấy giá trị pixel tại vị trí đó
                pixel_value = float(self.image_data[y_img, x_img])
                # Phát tín hiệu với vị trí và giá trị pixel
                self.mouse_position_changed.emit(x_img, y_img, pixel_value)
            else:
                # Nếu nằm ngoài vùng hình ảnh, phát tín hiệu với giá trị không hợp lệ
                self.mouse_position_changed.emit(x_img, y_img, float('nan'))
        
        if event.buttons() & Qt.LeftButton:
            # Xử lý theo công cụ hiện tại
            if self.current_tool == "pan":
                # TODO: Xử lý pan
                pass
            elif self.current_tool == "window":
                # TODO: Xử lý thay đổi cửa sổ
                pass
            
    def wheelEvent(self, event):
        """Xử lý sự kiện lăn chuột."""
        super().wheelEvent(event)
        
        # Xử lý zoom
        delta = event.angleDelta().y()
        if delta > 0:
            self.set_zoom(self.zoom_factor * 1.1)
        elif delta < 0:
            self.set_zoom(self.zoom_factor / 1.1)
            
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
