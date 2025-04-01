#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MPR Viewer Module

Module này cung cấp các widget để hiển thị MPR (Multi-Planar Reconstruction),
cho phép người dùng xem hình ảnh theo ba mặt cắt: Axial, Sagittal, và Coronal.
"""

import os
import sys
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Set

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QSplitter, QSlider, QFrame, QGridLayout, QSpinBox, QSizePolicy
)
from PyQt5.QtGui import QColor, QImage, QPixmap, QPainter, QPen
from PyQt5.QtCore import Qt, pyqtSignal, QSize, QPoint, QRect

logger = logging.getLogger(__name__)

class SliceView(QWidget):
    """
    Widget hiển thị một mặt cắt từ dữ liệu 3D.
    """
    
    sliceChanged = pyqtSignal(int)  # Tín hiệu khi slice thay đổi
    mouseClicked = pyqtSignal(int, int)  # Tín hiệu khi chuột được nhấp (x, y)
    
    def __init__(self, title="Slice View", parent=None):
        super().__init__(parent)
        self.title = title
        self.slice_data = None
        self.pixmap = None
        self.structures = []
        self.window_center = 128  # Window center mặc định
        self.window_width = 256   # Window width mặc định
        self.zoom_factor = 1.0    # Hệ số zoom
        self.pan_offset_x = 0     # Offset x cho pan
        self.pan_offset_y = 0     # Offset y cho pan
        
        self.setMinimumSize(200, 200)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        self._init_ui()
    
    def _init_ui(self):
        """Khởi tạo giao diện người dùng."""
        layout = QVBoxLayout(self)
        
        # Tiêu đề
        self.title_label = QLabel(self.title)
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("font-weight: bold; color: white; background-color: #2c3e50;")
        
        # Widget hiển thị hình ảnh
        self.image_frame = QFrame()
        self.image_frame.setFrameShape(QFrame.Box)
        self.image_frame.setLineWidth(1)
        self.image_frame.setStyleSheet("background-color: black; border: 1px solid #666;")
        
        layout.addWidget(self.title_label)
        layout.addWidget(self.image_frame, 1)
        
        # Vẽ hình ảnh trong sự kiện paintEvent
    
    def set_slice_data(self, slice_data):
        """Thiết lập dữ liệu slice và cập nhật hiển thị."""
        if slice_data is None:
            return
        
        self.slice_data = slice_data.copy()
        self._update_display()
    
    def set_window_level(self, center, width):
        """Thiết lập cửa sổ hiển thị (window/level)."""
        self.window_center = center
        self.window_width = width
        self._update_display()
    
    def set_zoom(self, factor):
        """Thiết lập hệ số zoom."""
        self.zoom_factor = max(0.1, factor)
        self._update_display()
    
    def set_pan(self, offset_x, offset_y):
        """Thiết lập vị trí pan."""
        self.pan_offset_x = offset_x
        self.pan_offset_y = offset_y
        self._update_display()
    
    def _update_display(self):
        """Cập nhật hiển thị slice với window/level và zoom hiện tại."""
        if self.slice_data is None:
            return
        
        # Áp dụng window/level
        low = self.window_center - self.window_width / 2
        high = self.window_center + self.window_width / 2
        
        # Chuẩn hóa và chuyển đổi thành hình ảnh 8-bit
        img_normalized = np.clip(self.slice_data, low, high)
        img_normalized = ((img_normalized - low) / (high - low) * 255).astype(np.uint8)
        
        # Tạo QImage từ mảng NumPy
        height, width = img_normalized.shape
        bytes_per_line = width
        q_img = QImage(img_normalized.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        
        # Tạo QPixmap từ QImage
        self.pixmap = QPixmap.fromImage(q_img)
        
        # Kích hoạt repaint để vẽ lại widget
        self.update()
    
    def paintEvent(self, event):
        """Xử lý sự kiện vẽ để hiển thị slice và cấu trúc."""
        super().paintEvent(event)
        
        if self.pixmap is None:
            return
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Vẽ hình ảnh slice
        target_rect = self.image_frame.contentsRect()
        
        # Áp dụng zoom và pan
        if self.zoom_factor != 1.0:
            # Tính toán kích thước zoom
            zoom_width = int(self.pixmap.width() * self.zoom_factor)
            zoom_height = int(self.pixmap.height() * self.zoom_factor)
            
            # Tính toán vị trí để giữ hình ảnh ở giữa
            x_offset = (target_rect.width() - zoom_width) // 2 + self.pan_offset_x
            y_offset = (target_rect.height() - zoom_height) // 2 + self.pan_offset_y
            
            # Tạo rectangle cho hình ảnh đã zoom
            target_rect = Qt.QRect(
                target_rect.x() + x_offset,
                target_rect.y() + y_offset,
                zoom_width,
                zoom_height
            )
        
        # Vẽ pixmap
        painter.drawPixmap(target_rect, self.pixmap)
        
        # Vẽ contours cấu trúc (nếu có)
        # TODO: Cài đặt vẽ cấu trúc
        
        painter.end()
    
    def mousePressEvent(self, event):
        """Xử lý sự kiện chuột được nhấn."""
        if event.button() == Qt.LeftButton:
            pos = event.pos()
            # Chuyển đổi vị trí chuột sang tọa độ trong hình ảnh
            # TODO: Tính toán tọa độ dựa trên zoom và pan
            self.mouseClicked.emit(pos.x(), pos.y())
        
        super().mousePressEvent(event)


class MPRViewer(QWidget):
    """
    Widget hiển thị MPR với ba mặt cắt: Axial, Sagittal, và Coronal.
    """
    
    sliceChanged = pyqtSignal(str, int)  # Tín hiệu khi slice thay đổi (plane, index)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image_data = None
        self.structure_set = None
        self.current_tool = None
        self.tool_options = {}
        
        # Slice indexes
        self.axial_slice = 0
        self.sagittal_slice = 0
        self.coronal_slice = 0
        
        # Plane being currently operated on
        self.active_plane = "Axial"
        
        self._init_ui()
    
    def _init_ui(self):
        """Khởi tạo giao diện người dùng MPR."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Tạo splitter chính để chia màn hình
        self.main_splitter = QSplitter(Qt.Vertical)
        
        # Tạo splitter trên với hai viewport ngang nhau
        self.top_splitter = QSplitter(Qt.Horizontal)
        
        # Các view cho từng mặt cắt
        self.axial_view = SliceView("Axial")
        self.sagittal_view = SliceView("Sagittal")
        self.coronal_view = SliceView("Coronal")
        
        # Thêm views vào splitters
        self.top_splitter.addWidget(self.axial_view)
        self.top_splitter.addWidget(self.sagittal_view)
        
        self.main_splitter.addWidget(self.top_splitter)
        self.main_splitter.addWidget(self.coronal_view)
        
        # Thiết lập kích thước ban đầu
        self.top_splitter.setSizes([300, 300])
        self.main_splitter.setSizes([400, 200])
        
        main_layout.addWidget(self.main_splitter)
        
        # Kết nối tín hiệu
        self.axial_view.sliceChanged.connect(lambda idx: self._on_slice_changed("Axial", idx))
        self.sagittal_view.sliceChanged.connect(lambda idx: self._on_slice_changed("Sagittal", idx))
        self.coronal_view.sliceChanged.connect(lambda idx: self._on_slice_changed("Coronal", idx))
        
        self.axial_view.mouseClicked.connect(lambda x, y: self._on_mouse_clicked("Axial", x, y))
        self.sagittal_view.mouseClicked.connect(lambda x, y: self._on_mouse_clicked("Sagittal", x, y))
        self.coronal_view.mouseClicked.connect(lambda x, y: self._on_mouse_clicked("Coronal", x, y))
    
    def set_image_data(self, image_data):
        """Thiết lập dữ liệu hình ảnh 3D và cập nhật tất cả các views."""
        if image_data is None:
            logger.warning("Dữ liệu hình ảnh là None, không thể hiển thị")
            return
        
        try:
            # Đảm bảo image_data là mảng 3D
            if len(image_data.shape) != 3:
                logger.error(f"Dữ liệu hình ảnh phải là mảng 3D, không phải {len(image_data.shape)}D")
                return
            
            self.image_data = image_data
            
            # Lấy kích thước hình ảnh
            depth, height, width = image_data.shape
            
            # Thiết lập slice mặc định ở giữa mỗi chiều
            self.axial_slice = depth // 2
            self.sagittal_slice = width // 2
            self.coronal_slice = height // 2
            
            # Cập nhật các views
            self._update_axial_view()
            self._update_sagittal_view()
            self._update_coronal_view()
            
            logger.info(f"Đã tải hình ảnh 3D có kích thước {image_data.shape}")
        
        except Exception as e:
            logger.error(f"Lỗi khi thiết lập dữ liệu hình ảnh: {e}")
    
    def set_structure_set(self, structure_set):
        """Thiết lập bộ cấu trúc để hiển thị trên các mặt cắt."""
        self.structure_set = structure_set
        
        # TODO: Cập nhật hiển thị cấu trúc trên mỗi view
        logger.info(f"Đã thiết lập bộ cấu trúc với {len(structure_set.structures) if structure_set else 0} cấu trúc")
    
    def set_tool(self, tool_name, options=None):
        """Thiết lập công cụ hiện tại và các tùy chọn."""
        self.current_tool = tool_name
        self.tool_options = options or {}
        logger.info(f"Đã thiết lập công cụ: {tool_name} với các tùy chọn: {options}")
    
    def set_active_plane(self, plane):
        """Thiết lập mặt cắt đang hoạt động."""
        if plane in ["Axial", "Sagittal", "Coronal"]:
            self.active_plane = plane
            logger.info(f"Đã thiết lập mặt cắt hoạt động: {plane}")
        else:
            logger.warning(f"Mặt cắt không hợp lệ: {plane}")
    
    def set_slice(self, slice_index, plane=None):
        """Thiết lập slice hiện tại cho một mặt cắt và cập nhật view."""
        if plane is None:
            plane = self.active_plane
        
        if self.image_data is None:
            return
        
        # Kiểm tra và đặt slice mới
        depth, height, width = self.image_data.shape
        
        if plane == "Axial":
            if 0 <= slice_index < depth:
                self.axial_slice = slice_index
                self._update_axial_view()
                # Cập nhật các đường cắt trên các view khác
                self._update_sagittal_view()
                self._update_coronal_view()
        
        elif plane == "Sagittal":
            if 0 <= slice_index < width:
                self.sagittal_slice = slice_index
                self._update_sagittal_view()
                # Cập nhật các đường cắt trên các view khác
                self._update_axial_view()
                self._update_coronal_view()
        
        elif plane == "Coronal":
            if 0 <= slice_index < height:
                self.coronal_slice = slice_index
                self._update_coronal_view()
                # Cập nhật các đường cắt trên các view khác
                self._update_axial_view()
                self._update_sagittal_view()
        
        # Phát tín hiệu slice đã thay đổi
        self.sliceChanged.emit(plane, slice_index)
    
    def get_max_slice(self, plane=None):
        """Lấy số slice tối đa cho một mặt cắt."""
        if plane is None:
            plane = self.active_plane
        
        if self.image_data is None:
            return 0
        
        depth, height, width = self.image_data.shape
        
        if plane == "Axial":
            return depth - 1
        elif plane == "Sagittal":
            return width - 1
        elif plane == "Coronal":
            return height - 1
        
        return 0
    
    def get_current_slice(self, plane=None):
        """Lấy slice hiện tại cho một mặt cắt."""
        if plane is None:
            plane = self.active_plane
        
        if plane == "Axial":
            return self.axial_slice
        elif plane == "Sagittal":
            return self.sagittal_slice
        elif plane == "Coronal":
            return self.coronal_slice
        
        return 0
    
    def _update_axial_view(self):
        """Cập nhật view Axial với slice hiện tại."""
        if self.image_data is None:
            return
        
        try:
            # Lấy slice axial
            axial_slice_data = self.image_data[self.axial_slice, :, :]
            self.axial_view.set_slice_data(axial_slice_data)
        except Exception as e:
            logger.error(f"Lỗi khi cập nhật view Axial: {e}")
    
    def _update_sagittal_view(self):
        """Cập nhật view Sagittal với slice hiện tại."""
        if self.image_data is None:
            return
        
        try:
            # Lấy slice sagittal (chuyển vị để hiển thị đúng chiều)
            sagittal_slice_data = self.image_data[:, :, self.sagittal_slice].transpose()
            self.sagittal_view.set_slice_data(sagittal_slice_data)
        except Exception as e:
            logger.error(f"Lỗi khi cập nhật view Sagittal: {e}")
    
    def _update_coronal_view(self):
        """Cập nhật view Coronal với slice hiện tại."""
        if self.image_data is None:
            return
        
        try:
            # Lấy slice coronal (chuyển vị để hiển thị đúng chiều)
            coronal_slice_data = self.image_data[:, self.coronal_slice, :].transpose()
            self.coronal_view.set_slice_data(coronal_slice_data)
        except Exception as e:
            logger.error(f"Lỗi khi cập nhật view Coronal: {e}")
    
    def _on_slice_changed(self, plane, slice_index):
        """Xử lý khi slice thay đổi trong một view."""
        self.set_slice(slice_index, plane)
    
    def _on_mouse_clicked(self, plane, x, y):
        """Xử lý khi chuột được nhấp trên một view."""
        # TODO: Cập nhật các slice dựa trên vị trí chuột trên các mặt cắt khác
        pass


def create_sample_data(size=100):
    """
    Tạo dữ liệu hình ảnh 3D mẫu để kiểm thử MPR viewer.
    Tạo một khối 3D với một khối cầu ở giữa.
    
    Args:
        size: Kích thước của mảng 3D (size x size x size)
    
    Returns:
        Mảng numpy 3D với một khối cầu ở giữa
    """
    # Tạo mảng 3D với giá trị 0
    data = np.zeros((size, size, size), dtype=np.float32)
    
    # Tạo một khối cầu ở giữa
    center = size // 2
    radius = size // 4
    
    # Lặp qua tất cả các voxel để kiểm tra xem chúng có nằm trong khối cầu không
    for x in range(size):
        for y in range(size):
            for z in range(size):
                # Tính khoảng cách từ voxel đến tâm
                distance = np.sqrt((x - center) ** 2 + (y - center) ** 2 + (z - center) ** 2)
                
                # Nếu voxel nằm trong khối cầu, đặt giá trị cao hơn
                if distance < radius:
                    data[z, y, x] = 200
    
    # Thêm một vài cấu trúc để dữ liệu thực tế hơn
    # Thêm một hình trụ dọc
    for z in range(size):
        for y in range(size):
            for x in range(center - radius // 3, center + radius // 3):
                data[z, y, x] = 150
    
    # Thêm một hình trụ ngang
    for z in range(size):
        for y in range(center - radius // 3, center + radius // 3):
            for x in range(size):
                data[z, y, x] = 100
    
    # Thêm nhiễu ngẫu nhiên
    noise = np.random.normal(0, 10, (size, size, size))
    data += noise
    
    # Đảm bảo tất cả các giá trị nằm trong phạm vi [0, 255]
    data = np.clip(data, 0, 255)
    
    return data


if __name__ == "__main__":
    # Mã kiểm thử MPR viewer
    import sys
    from PyQt5.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    
    # Tạo dữ liệu mẫu
    sample_data = create_sample_data(100)
    
    # Tạo và hiển thị MPR viewer
    viewer = MPRViewer()
    viewer.set_image_data(sample_data)
    viewer.setWindowTitle("MPR Viewer Demo")
    viewer.resize(800, 600)
    viewer.show()
    
    sys.exit(app.exec_()) 