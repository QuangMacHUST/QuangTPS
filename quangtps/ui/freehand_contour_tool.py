#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module cung cấp công cụ vẽ contour tự do cho QuangTPS.

Module này triển khai công cụ cho phép người dùng vẽ contour bằng cách 
kéo chuột tự do trên hình ảnh. Hỗ trợ các chức năng như làm mịn đường viền 
và tự động đóng contour.
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union

from PyQt5.QtCore import Qt, QPointF, QRectF
from PyQt5.QtGui import QPainter, QPen, QColor, QPainterPath

from quangtps.ui.base_contour_tool import ContourTool

logger = logging.getLogger(__name__)


def simplify_path(points: List[Tuple[int, int]], tolerance: float = 2.0) -> List[Tuple[int, int]]:
    """
    Đơn giản hóa đường dẫn bằng thuật toán Douglas-Peucker.
    
    Parameters
    ----------
    points : List[Tuple[int, int]]
        Danh sách các điểm đầu vào
    tolerance : float
        Dung sai cho việc đơn giản hóa
    
    Returns
    -------
    List[Tuple[int, int]]
        Danh sách các điểm đã đơn giản hóa
    """
    if len(points) <= 2:
        return points
    
    # Tính khoảng cách từ một điểm đến đường thẳng
    def point_line_distance(point, line_start, line_end):
        if line_start == line_end:
            return np.sqrt((point[0] - line_start[0])**2 + (point[1] - line_start[1])**2)
        
        # Tính khoảng cách từ điểm đến đường thẳng
        line_length = np.sqrt((line_end[0] - line_start[0])**2 + (line_end[1] - line_start[1])**2)
        
        # Nếu đường thẳng có độ dài 0, trả về khoảng cách đến điểm đầu
        if line_length == 0:
            return np.sqrt((point[0] - line_start[0])**2 + (point[1] - line_start[1])**2)
        
        # Tính véc-tơ chỉ phương
        line_dir_x = (line_end[0] - line_start[0]) / line_length
        line_dir_y = (line_end[1] - line_start[1]) / line_length
        
        # Tính véc-tơ từ điểm đầu đến điểm cần tính
        vec_x = point[0] - line_start[0]
        vec_y = point[1] - line_start[1]
        
        # Tính độ dài chiếu
        projection = vec_x * line_dir_x + vec_y * line_dir_y
        
        # Tính điểm chiếu
        if projection < 0:
            closest_x, closest_y = line_start
        elif projection > line_length:
            closest_x, closest_y = line_end
        else:
            closest_x = line_start[0] + projection * line_dir_x
            closest_y = line_start[1] + projection * line_dir_y
        
        # Tính khoảng cách
        return np.sqrt((point[0] - closest_x)**2 + (point[1] - closest_y)**2)
    
    # Tìm điểm xa nhất
    dmax = 0
    index = 0
    end = len(points) - 1
    
    for i in range(1, end):
        d = point_line_distance(points[i], points[0], points[end])
        if d > dmax:
            index = i
            dmax = d
    
    # Nếu khoảng cách tối đa lớn hơn dung sai, đệ quy chia và đơn giản hóa
    if dmax > tolerance:
        # Đệ quy cho nửa đầu
        first_half = simplify_path(points[:index+1], tolerance)
        # Đệ quy cho nửa sau
        second_half = simplify_path(points[index:], tolerance)
        
        # Nối kết quả, loại bỏ điểm trùng lặp
        return first_half[:-1] + second_half
    else:
        # Dưới dung sai, chỉ giữ hai điểm đầu và cuối
        return [points[0], points[end]]


class FreehandContourTool(ContourTool):
    """Công cụ vẽ contour tự do bằng chuột."""
    
    def __init__(self):
        """Khởi tạo công cụ vẽ contour tự do."""
        super().__init__("Freehand")
        
        # Dữ liệu contour
        self.points = []  # Danh sách các điểm khi đang vẽ
        self.preview_points = []  # Điểm xem trước (điểm đóng)
        self.is_drawing = False  # Đang vẽ hay không
        self.simplify = True  # Có đơn giản hóa contour hay không
        self.auto_close = True  # Tự động đóng contour
        self.min_points = 3  # Số điểm tối thiểu để tạo contour
        self.tolerance = 2.0  # Dung sai cho việc đơn giản hóa
        self.preview_color = QColor(255, 255, 0, 200)  # Màu xem trước (vàng bán trong suốt)
        self.final_color = QColor(255, 0, 0, 200)  # Màu cuối cùng (đỏ bán trong suốt)
    
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
            # Bắt đầu vẽ mới
            self.is_drawing = True
            self.points = [pos]
            self.preview_points = []
            self.update_preview()
        elif button == Qt.RightButton:
            # Hoàn thành contour (nếu đang vẽ)
            if self.is_drawing and len(self.points) >= self.min_points:
                self._finalize_contour()
            
            # Hủy vẽ
            self.is_drawing = False
            self.points = []
            self.preview_points = []
            self.update_preview()
    
    def mouse_move(self, pos: Tuple[int, int], buttons: int):
        """
        Xử lý sự kiện khi di chuyển chuột.
        
        Parameters
        ----------
        pos : Tuple[int, int]
            Vị trí chuột (x, y)
        buttons : int
            Các nút chuột đang được nhấn (Qt.LeftButton, Qt.RightButton, v.v.)
        """
        if not self.active or not self.image_widget or not self.is_drawing:
            return
        
        # Thêm điểm mới nếu đang nhấn chuột trái
        if buttons & Qt.LeftButton:
            # Kiểm tra xem điểm mới có trùng với điểm cuối cùng không
            if self.points and (pos[0] == self.points[-1][0] and pos[1] == self.points[-1][1]):
                return
            
            self.points.append(pos)
            
            # Cập nhật preview để hiển thị đường vẽ
            if self.auto_close and len(self.points) >= 3:
                # Thêm điểm đầu tiên vào cuối để đóng contour
                self.preview_points = [self.points[0]]
            else:
                self.preview_points = []
            
            self.update_preview()
    
    def mouse_release(self, pos: Tuple[int, int], button: int):
        """
        Xử lý sự kiện khi thả chuột.
        
        Parameters
        ----------
        pos : Tuple[int, int]
            Vị trí chuột (x, y)
        button : int
            Nút chuột (Qt.LeftButton, Qt.RightButton, v.v.)
        """
        if not self.active or not self.image_widget:
            return
        
        if button == Qt.LeftButton and self.is_drawing:
            # Hoàn thành contour nếu có đủ điểm
            if len(self.points) >= self.min_points:
                self._finalize_contour()
            
            # Hủy vẽ
            self.is_drawing = False
            self.points = []
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
            # Hủy vẽ
            self.is_drawing = False
            self.points = []
            self.preview_points = []
            self.update_preview()
    
    def set_simplify(self, enable: bool):
        """
        Thiết lập việc đơn giản hóa contour.
        
        Parameters
        ----------
        enable : bool
            Bật/tắt đơn giản hóa
        """
        self.simplify = enable
    
    def set_auto_close(self, enable: bool):
        """
        Thiết lập việc tự động đóng contour.
        
        Parameters
        ----------
        enable : bool
            Bật/tắt tự động đóng
        """
        self.auto_close = enable
    
    def set_tolerance(self, tolerance: float):
        """
        Thiết lập dung sai cho việc đơn giản hóa.
        
        Parameters
        ----------
        tolerance : float
            Dung sai
        """
        self.tolerance = max(0.5, tolerance)
    
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
        if not self.active or not self.is_drawing or not self.points:
            return
        
        # Vẽ contour đang vẽ
        if len(self.points) > 1:
            # Thiết lập bút vẽ
            pen = QPen(self.preview_color)
            pen.setWidth(2)
            painter.setPen(pen)
            
            # Vẽ đường nối các điểm
            path = QPainterPath()
            path.moveTo(self.points[0][0], self.points[0][1])
            
            for x, y in self.points[1:]:
                path.lineTo(x, y)
            
            # Vẽ điểm đóng nếu có
            if self.preview_points:
                for x, y in self.preview_points:
                    path.lineTo(x, y)
            
            painter.drawPath(path)
            
            # Vẽ các điểm (nodes)
            painter.setPen(QPen(Qt.white, 1))
            for x, y in self.points:
                painter.drawEllipse(QPointF(x, y), 3, 3)
    
    def _finalize_contour(self):
        """Hoàn thành contour và phát tín hiệu."""
        if not self.points or len(self.points) < self.min_points:
            return
        
        # Tạo contour hoàn chỉnh
        final_points = list(self.points)
        
        # Đóng contour nếu cần
        if self.auto_close and final_points[0] != final_points[-1]:
            final_points.append(final_points[0])
        
        # Đơn giản hóa nếu được yêu cầu
        if self.simplify and len(final_points) > 3:
            final_points = simplify_path(final_points, self.tolerance)
        
        # Lấy lát cắt hiện tại
        slice_idx = self.image_widget.slice_idx if self.image_widget else 0
        
        # Phát tín hiệu với contour mới
        self.contour_created.emit(self.contour_name, final_points, slice_idx)
    
    def apply_to_current_slice(self):
        """Áp dụng contour vào lát cắt hiện tại."""
        if self.is_drawing and len(self.points) >= self.min_points:
            self._finalize_contour()


# Variant class for contour with a brush - lớp biến thể cho vẽ contour với brush
class BrushContourTool(FreehandContourTool):
    """Công cụ vẽ contour bằng brush."""
    
    def __init__(self):
        """Khởi tạo công cụ vẽ contour bằng brush."""
        super().__init__()
        
        self.name = "Brush"
        self.brush_size = 10  # Kích thước brush mặc định
        self.brush_points = []  # Điểm brush
        self.brush_mode = "Circle"  # Chế độ brush: "Circle" hoặc "Square"
    
    def set_brush_size(self, size: int):
        """
        Thiết lập kích thước brush.
        
        Parameters
        ----------
        size : int
            Kích thước brush
        """
        self.brush_size = max(1, size)
    
    def set_brush_mode(self, mode: str):
        """
        Thiết lập chế độ brush.
        
        Parameters
        ----------
        mode : str
            Chế độ brush ("Circle" hoặc "Square")
        """
        if mode in ["Circle", "Square"]:
            self.brush_mode = mode
    
    def mouse_move(self, pos: Tuple[int, int], buttons: int):
        """
        Xử lý sự kiện khi di chuyển chuột.
        
        Parameters
        ----------
        pos : Tuple[int, int]
            Vị trí chuột (x, y)
        buttons : int
            Các nút chuột đang được nhấn (Qt.LeftButton, Qt.RightButton, v.v.)
        """
        if not self.active or not self.image_widget:
            return
        
        # Cập nhật vị trí brush để hiển thị
        self.brush_points = [pos]
        
        # Thêm điểm vào contour nếu đang vẽ
        if self.is_drawing and (buttons & Qt.LeftButton):
            # Kiểm tra xem điểm mới có trùng với điểm cuối cùng không
            if not self.points or (pos[0] != self.points[-1][0] or pos[1] != self.points[-1][1]):
                self.points.append(pos)
                self.update_preview()
    
    def paint(self, painter: QPainter):
        """
        Vẽ brush và contour tạm thời lên hình ảnh.
        
        Parameters
        ----------
        painter : QPainter
            Đối tượng QPainter để vẽ
        """
        # Vẽ contour đang vẽ
        super().paint(painter)
        
        # Vẽ brush cursor
        if self.active and self.brush_points:
            # Thiết lập bút vẽ
            pen = QPen(self.preview_color)
            pen.setWidth(1)
            painter.setPen(pen)
            
            # Vẽ brush
            for x, y in self.brush_points:
                if self.brush_mode == "Circle":
                    painter.drawEllipse(QPointF(x, y), self.brush_size, self.brush_size)
                else:  # Square
                    rect = QRectF(x - self.brush_size, y - self.brush_size, 
                                 2 * self.brush_size, 2 * self.brush_size)
                    painter.drawRect(rect)
    
    def _finalize_contour(self):
        """Hoàn thành contour và phát tín hiệu."""
        # Tạo contour từ đường viền của các điểm brush
        # Đây là một giải pháp đơn giản, trong thực tế cần thuật toán phức tạp hơn
        
        # Gọi phương thức của lớp cha
        super()._finalize_contour()
