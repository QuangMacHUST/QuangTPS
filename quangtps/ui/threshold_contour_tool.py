#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module cung cấp công cụ tạo contour dựa trên ngưỡng giá trị pixel.

Module này triển khai công cụ cho phép người dùng tạo contour dựa trên ngưỡng 
giá trị pixel (HU) trong hình ảnh y tế, hỗ trợ các chức năng như seed-based
region growing và thresholding tự động.
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import matplotlib.path as mpath
from scipy import ndimage

from PyQt5.QtCore import Qt, QPointF, QRectF
from PyQt5.QtGui import QPainter, QPen, QColor, QPainterPath

from quangtps.ui.base_contour_tool import ContourTool

logger = logging.getLogger(__name__)


def find_contours_from_mask(mask, simplify=True, tolerance=1.0):
    """
    Tìm đường viền từ một mask nhị phân.
    
    Parameters
    ----------
    mask : np.ndarray
        Mảng nhị phân 2D (0 và 1)
    simplify : bool
        Có đơn giản hóa contour không
    tolerance : float
        Dung sai cho việc đơn giản hóa
    
    Returns
    -------
    List[List[Tuple[int, int]]]
        Danh sách các contour, mỗi contour là danh sách các điểm (x, y)
    """
    # Kiểm tra xem có scipy.ndimage.label hay không
    # Sử dụng thuật toán đơn giản để tìm contour
    
    from skimage import measure
    
    try:
        # Sử dụng skimage.measure.find_contours nếu có thể
        contours = measure.find_contours(mask, 0.5)
        
        # Chuyển đổi định dạng contour
        result = []
        for contour in contours:
            # Đảo ngược x, y vì skimage trả về (row, col) thay vì (x, y)
            contour_points = [(int(c[1]), int(c[0])) for c in contour]
            
            # Đơn giản hóa nếu cần
            if simplify and len(contour_points) > 10:
                # Ở đây chúng ta có thể sử dụng thuật toán Douglas-Peucker
                # Nhưng để đơn giản, chỉ lấy mẫu các điểm
                step = max(1, len(contour_points) // (100 * int(tolerance)))
                contour_points = contour_points[::step]
            
            # Thêm điểm đầu vào cuối để đóng contour
            if contour_points and contour_points[0] != contour_points[-1]:
                contour_points.append(contour_points[0])
            
            result.append(contour_points)
        
        return result
    
    except ImportError:
        # Phương pháp thay thế nếu không có skimage
        logger.warning("skimage not found, using alternative contour finding method")
        
        # Tạo một mask lớn hơn để tìm đường viền
        padded_mask = np.pad(mask, pad_width=1, mode='constant', constant_values=0)
        
        # Tìm đường viền bằng cách tìm sự chuyển đổi từ 0 sang 1 hoặc từ 1 sang 0
        contours = []
        visited = np.zeros_like(padded_mask, dtype=bool)
        
        rows, cols = padded_mask.shape
        
        for r in range(1, rows-1):
            for c in range(1, cols-1):
                if padded_mask[r, c] == 1 and not visited[r, c]:
                    # Tìm thấy một điểm contour
                    contour = []
                    current_r, current_c = r, c
                    
                    # Hướng: 0=right, 1=down, 2=left, 3=up
                    direction = 0
                    
                    while True:
                        visited[current_r, current_c] = True
                        contour.append((current_c-1, current_r-1))  # Điều chỉnh vị trí về mask gốc
                        
                        # Thử các hướng theo chiều kim đồng hồ
                        found_next = False
                        for i in range(4):
                            next_dir = (direction + i) % 4
                            
                            if next_dir == 0:  # Right
                                next_r, next_c = current_r, current_c + 1
                            elif next_dir == 1:  # Down
                                next_r, next_c = current_r + 1, current_c
                            elif next_dir == 2:  # Left
                                next_r, next_c = current_r, current_c - 1
                            else:  # Up
                                next_r, next_c = current_r - 1, current_c
                            
                            if padded_mask[next_r, next_c] == 1 and not visited[next_r, next_c]:
                                current_r, current_c = next_r, next_c
                                direction = next_dir
                                found_next = True
                                break
                        
                        if not found_next or len(contour) > 1000:  # Giới hạn kích thước contour
                            break
                    
                    # Đóng contour
                    if contour and contour[0] != contour[-1]:
                        contour.append(contour[0])
                    
                    contours.append(contour)
        
        return contours


def region_growing(img, seed, threshold_min, threshold_max):
    """
    Thuật toán region growing cho segmentation.
    
    Parameters
    ----------
    img : np.ndarray
        Hình ảnh đầu vào
    seed : Tuple[int, int]
        Điểm seed (x, y)
    threshold_min : float
        Ngưỡng dưới
    threshold_max : float
        Ngưỡng trên
    
    Returns
    -------
    np.ndarray
        Mask nhị phân của vùng tìm được
    """
    # Đảm bảo seed trong phạm vi hình ảnh
    x, y = seed
    if x < 0 or y < 0 or x >= img.shape[1] or y >= img.shape[0]:
        return np.zeros_like(img, dtype=bool)
    
    # Bắt đầu từ seed
    mask = np.zeros_like(img, dtype=bool)
    checked = np.zeros_like(img, dtype=bool)
    
    # Lấy giá trị tại điểm seed
    seed_value = img[y, x]
    
    # Ngưỡng tuyệt đối
    t_min = threshold_min
    t_max = threshold_max
    
    # Stack để BFS
    stack = [(x, y)]
    mask[y, x] = True
    checked[y, x] = True
    
    # Các hướng di chuyển (4-connected)
    dx = [1, 0, -1, 0]
    dy = [0, 1, 0, -1]
    
    # Thực hiện BFS
    while stack:
        x, y = stack.pop(0)
        
        for i in range(4):
            nx, ny = x + dx[i], y + dy[i]
            
            # Kiểm tra biên
            if nx < 0 or ny < 0 or nx >= img.shape[1] or ny >= img.shape[0]:
                continue
            
            # Kiểm tra đã duyệt chưa
            if checked[ny, nx]:
                continue
            
            # Đánh dấu đã duyệt
            checked[ny, nx] = True
            
            # Kiểm tra giá trị
            if t_min <= img[ny, nx] <= t_max:
                mask[ny, nx] = True
                stack.append((nx, ny))
    
    return mask


def threshold_segmentation(img, threshold_min, threshold_max):
    """
    Phân đoạn hình ảnh bằng cách áp dụng ngưỡng.
    
    Parameters
    ----------
    img : np.ndarray
        Hình ảnh đầu vào
    threshold_min : float
        Ngưỡng dưới
    threshold_max : float
        Ngưỡng trên
    
    Returns
    -------
    np.ndarray
        Mask nhị phân của vùng tìm được
    """
    # Áp dụng ngưỡng đơn giản
    mask = (img >= threshold_min) & (img <= threshold_max)
    
    # Lọc nhiễu với phép toán đóng (closing)
    mask = ndimage.binary_closing(mask)
    
    # Loại bỏ các vùng nhỏ
    labeled, num_features = ndimage.label(mask)
    if num_features > 1:
        sizes = ndimage.sum(mask, labeled, range(1, num_features + 1))
        mask_sizes = sizes > 20  # Lọc các vùng có ít hơn 20 pixel
        mask = mask_sizes[labeled - 1]
    
    return mask


class ThresholdContourTool(ContourTool):
    """Công cụ tạo contour dựa trên ngưỡng giá trị pixel."""
    
    def __init__(self):
        """Khởi tạo công cụ tạo contour dựa trên ngưỡng."""
        super().__init__("Threshold")
        
        # Dữ liệu ngưỡng
        self.threshold_min = -100  # Ngưỡng dưới mặc định
        self.threshold_max = 100   # Ngưỡng trên mặc định
        self.preview_points = []   # Điểm xem trước
        self.seed_point = None     # Điểm seed
        self.use_region_growing = True  # Sử dụng region growing hay toàn bộ ảnh
        self.preview_color = QColor(255, 255, 0, 200)  # Màu xem trước (vàng bán trong suốt)
        self.final_color = QColor(255, 0, 0, 200)  # Màu cuối cùng (đỏ bán trong suốt)
    
    def set_thresholds(self, min_val: float, max_val: float):
        """
        Thiết lập ngưỡng giá trị.
        
        Parameters
        ----------
        min_val : float
            Ngưỡng dưới
        max_val : float
            Ngưỡng trên
        """
        self.threshold_min = min_val
        self.threshold_max = max_val
        
        # Cập nhật preview nếu có seed point
        if self.seed_point:
            self._update_preview()
    
    def set_use_region_growing(self, enable: bool):
        """
        Thiết lập việc sử dụng region growing.
        
        Parameters
        ----------
        enable : bool
            Bật/tắt region growing
        """
        self.use_region_growing = enable
        
        # Cập nhật preview nếu có seed point
        if self.seed_point:
            self._update_preview()
    
    def mouse_press(self, pos: Tuple[int, int], button: int):
        """
        Xử lý sự kiện khi nhấn chuột.
        
        Parameters
        ----------
        pos : Tuple[int, int]
            Vị trí chuột (x, y)
        button : int
            Nút chuột (Qt.LeftButton, Qt.RightButton, v.v.)
        """
        if not self.active or not self.image_widget:
            return
        
        if button == Qt.LeftButton:
            # Thiết lập điểm seed mới
            self.seed_point = pos
            self._update_preview()
        elif button == Qt.RightButton:
            # Áp dụng contour
            if self.seed_point and self.preview_points:
                self._finalize_contour()
            
            # Hủy seed point
            self.seed_point = None
            self.preview_points = []
            self.update_preview()
    
    def key_press(self, key: int):
        """
        Xử lý sự kiện khi nhấn phím.
        
        Parameters
        ----------
        key : int
            Mã phím
        """
        if not self.active:
            return
        
        if key == Qt.Key_Escape:
            # Hủy seed point
            self.seed_point = None
            self.preview_points = []
            self.update_preview()
        elif key == Qt.Key_Enter or key == Qt.Key_Return:
            # Áp dụng contour
            if self.seed_point and self.preview_points:
                self._finalize_contour()
                
                # Hủy seed point
                self.seed_point = None
                self.preview_points = []
                self.update_preview()
    
    def set_preview_color(self, color: QColor):
        """
        Thiết lập màu xem trước.
        
        Parameters
        ----------
        color : QColor
            Màu sắc
        """
        self.preview_color = color
    
    def set_final_color(self, color: QColor):
        """
        Thiết lập màu cuối cùng.
        
        Parameters
        ----------
        color : QColor
            Màu sắc
        """
        self.final_color = color
    
    def paint(self, painter: QPainter):
        """
        Vẽ contour tạm thời lên hình ảnh.
        
        Parameters
        ----------
        painter : QPainter
            Đối tượng QPainter để vẽ
        """
        if not self.active or not self.preview_points:
            return
        
        # Thiết lập bút vẽ
        pen = QPen(self.preview_color)
        pen.setWidth(2)
        painter.setPen(pen)
        
        # Vẽ contour
        for contour in self.preview_points:
            if len(contour) > 1:
                path = QPainterPath()
                path.moveTo(contour[0][0], contour[0][1])
                
                for x, y in contour[1:]:
                    path.lineTo(x, y)
                
                painter.drawPath(path)
        
        # Vẽ seed point nếu có
        if self.seed_point:
            painter.setPen(QPen(Qt.white, 1))
            painter.drawEllipse(QPointF(self.seed_point[0], self.seed_point[1]), 5, 5)
    
    def _update_preview(self):
        """Cập nhật preview contour dựa trên ngưỡng và seed point."""
        if not self.image_widget or not self.seed_point:
            return
        
        # Lấy hình ảnh hiện tại từ widget
        img_data = self.image_widget.get_image_data()
        if img_data is None:
            return
        
        # Tạo mask
        if self.use_region_growing:
            # Sử dụng region growing từ seed point
            mask = region_growing(img_data, (self.seed_point[0], self.seed_point[1]), 
                                self.threshold_min, self.threshold_max)
        else:
            # Sử dụng ngưỡng trên toàn bộ ảnh
            mask = threshold_segmentation(img_data, self.threshold_min, self.threshold_max)
        
        # Tìm contour từ mask
        self.preview_points = find_contours_from_mask(mask)
        
        # Cập nhật hiển thị
        self.update_preview()
    
    def _finalize_contour(self):
        """Hoàn thành contour và phát tín hiệu."""
        if not self.preview_points:
            return
        
        # Lấy lát cắt hiện tại
        slice_idx = self.image_widget.slice_idx if self.image_widget else 0
        
        # Chọn contour lớn nhất
        if self.preview_points:
            largest_contour = max(self.preview_points, key=lambda x: len(x))
            
            # Phát tín hiệu với contour mới
            self.contour_created.emit(self.contour_name, largest_contour, slice_idx)
    
    def apply_to_current_slice(self):
        """Áp dụng contour vào lát cắt hiện tại."""
        if self.seed_point and self.preview_points:
            self._finalize_contour()


class AutoThresholdContourTool(ThresholdContourTool):
    """Công cụ tạo contour tự động dựa trên ngưỡng và phân tích histogram."""
    
    def __init__(self):
        """Khởi tạo công cụ tạo contour tự động."""
        super().__init__()
        
        self.name = "AutoThreshold"
        self.use_otsu = True  # Sử dụng thuật toán Otsu
    
    def set_use_otsu(self, enable: bool):
        """
        Thiết lập việc sử dụng thuật toán Otsu.
        
        Parameters
        ----------
        enable : bool
            Bật/tắt Otsu
        """
        self.use_otsu = enable
        
        # Cập nhật preview nếu có seed point
        if self.seed_point:
            self._update_preview()
    
    def _update_preview(self):
        """Cập nhật preview contour dựa trên ngưỡng tự động."""
        if not self.image_widget or not self.seed_point:
            return
        
        # Lấy hình ảnh hiện tại từ widget
        img_data = self.image_widget.get_image_data()
        if img_data is None:
            return
        
        # Tự động tính toán ngưỡng
        if self.use_otsu:
            # Sử dụng thuật toán Otsu
            try:
                from skimage.filters import threshold_otsu
                threshold = threshold_otsu(img_data)
                self.threshold_min = threshold - 50
                self.threshold_max = threshold + 50
            except ImportError:
                # Nếu không có skimage, sử dụng ngưỡng đơn giản
                self.threshold_min = np.percentile(img_data, 25)
                self.threshold_max = np.percentile(img_data, 75)
        else:
            # Sử dụng phân tích histogram đơn giản
            p_low, p_high = np.percentile(img_data, [20, 80])
            self.threshold_min = p_low
            self.threshold_max = p_high
        
        # Gọi phương thức của lớp cha để tạo contour
        super()._update_preview()
    
    def mouse_press(self, pos: Tuple[int, int], button: int):
        """
        Xử lý sự kiện khi nhấn chuột.
        
        Parameters
        ----------
        pos : Tuple[int, int]
            Vị trí chuột (x, y)
        button : int
            Nút chuột (Qt.LeftButton, Qt.RightButton, v.v.)
        """
        # Gọi phương thức của lớp cha
        super().mouse_press(pos, button)
