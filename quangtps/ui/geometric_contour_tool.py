#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module cung cấp các công cụ vẽ contour hình học cho QuangTPS.

Module này triển khai các công cụ cho phép người dùng vẽ contour 
với các hình dạng hình học cơ bản như hình tròn, hình chữ nhật, 
hình đa giác và hình elip.
"""

import logging
import math
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union

from PyQt5.QtCore import Qt, QPointF, QRectF
from PyQt5.QtGui import QPainter, QPen, QColor, QPainterPath

from quangtps.ui.base_contour_tool import ContourTool

logger = logging.getLogger(__name__)


def generate_circle_points(center: Tuple[int, int], radius: float, num_points: int = 60) -> List[Tuple[int, int]]:
    """
    Tạo danh sách điểm cho hình tròn.
    
    Parameters
    ----------
    center : Tuple[int, int]
        Tọa độ tâm (x, y)
    radius : float
        Bán kính
    num_points : int
        Số điểm sử dụng để tạo hình tròn
    
    Returns
    -------
    List[Tuple[int, int]]
        Danh sách các điểm tạo thành hình tròn
    """
    points = []
    
    for i in range(num_points):
        angle = 2 * math.pi * i / num_points
        x = int(center[0] + radius * math.cos(angle))
        y = int(center[1] + radius * math.sin(angle))
        points.append((x, y))
    
    # Thêm điểm đầu tiên vào cuối để đóng contour
    points.append(points[0])
    
    return points


def generate_rectangle_points(top_left: Tuple[int, int], bottom_right: Tuple[int, int]) -> List[Tuple[int, int]]:
    """
    Tạo danh sách điểm cho hình chữ nhật.
    
    Parameters
    ----------
    top_left : Tuple[int, int]
        Tọa độ góc trên bên trái (x, y)
    bottom_right : Tuple[int, int]
        Tọa độ góc dưới bên phải (x, y)
    
    Returns
    -------
    List[Tuple[int, int]]
        Danh sách các điểm tạo thành hình chữ nhật
    """
    x1, y1 = top_left
    x2, y2 = bottom_right
    
    points = [
        (x1, y1),  # Top-left
        (x2, y1),  # Top-right
        (x2, y2),  # Bottom-right
        (x1, y2),  # Bottom-left
        (x1, y1)   # Close the rectangle
    ]
    
    return points


def generate_ellipse_points(center: Tuple[int, int], rx: float, ry: float, num_points: int = 60) -> List[Tuple[int, int]]:
    """
    Tạo danh sách điểm cho hình elip.
    
    Parameters
    ----------
    center : Tuple[int, int]
        Tọa độ tâm (x, y)
    rx : float
        Bán kính theo trục x
    ry : float
        Bán kính theo trục y
    num_points : int
        Số điểm sử dụng để tạo hình elip
    
    Returns
    -------
    List[Tuple[int, int]]
        Danh sách các điểm tạo thành hình elip
    """
    points = []
    
    for i in range(num_points):
        angle = 2 * math.pi * i / num_points
        x = int(center[0] + rx * math.cos(angle))
        y = int(center[1] + ry * math.sin(angle))
        points.append((x, y))
    
    # Thêm điểm đầu tiên vào cuối để đóng contour
    points.append(points[0])
    
    return points


class CircleContourTool(ContourTool):
    """Công cụ vẽ contour hình tròn."""
    
    def __init__(self):
        """Khởi tạo công cụ vẽ contour hình tròn."""
        super().__init__("Circle")
        
        # Dữ liệu contour
        self.center = None  # Tâm hình tròn
        self.current_point = None  # Điểm hiện tại khi vẽ
        self.is_drawing = False  # Đang vẽ hay không
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
            # Thiết lập tâm hình tròn
            self.center = pos
            self.current_point = pos
            self.is_drawing = True
            self.update_preview()
        elif button == Qt.RightButton:
            # Hủy vẽ
            self.is_drawing = False
            self.center = None
            self.current_point = None
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
        
        # Cập nhật điểm hiện tại để tính bán kính
        self.current_point = pos
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
        if not self.active or not self.image_widget or not self.is_drawing:
            return
        
        if button == Qt.LeftButton:
            # Hoàn thành hình tròn
            self.current_point = pos
            self._finalize_contour()
            
            # Hủy vẽ
            self.is_drawing = False
            self.center = None
            self.current_point = None
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
            self.center = None
            self.current_point = None
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
        Vẽ hình tròn tạm thời lên hình ảnh.
        
        Parameters
        ----------
        painter : QPainter
            Đối tượng QPainter để vẽ
        """
        if not self.active or not self.is_drawing or not self.center or not self.current_point:
            return
        
        # Tính bán kính
        dx = self.current_point[0] - self.center[0]
        dy = self.current_point[1] - self.center[1]
        radius = math.sqrt(dx*dx + dy*dy)
        
        # Thiết lập bút vẽ
        pen = QPen(self.preview_color)
        pen.setWidth(2)
        painter.setPen(pen)
        
        # Vẽ hình tròn
        painter.drawEllipse(QPointF(self.center[0], self.center[1]), radius, radius)
        
        # Vẽ tâm và điểm bán kính
        painter.setPen(QPen(Qt.white, 1))
        painter.drawEllipse(QPointF(self.center[0], self.center[1]), 3, 3)
        painter.drawEllipse(QPointF(self.current_point[0], self.current_point[1]), 3, 3)
        painter.drawLine(self.center[0], self.center[1], self.current_point[0], self.current_point[1])
    
    def _finalize_contour(self):
        """Hoàn thành contour và phát tín hiệu."""
        if not self.center or not self.current_point:
            return
        
        # Tính bán kính
        dx = self.current_point[0] - self.center[0]
        dy = self.current_point[1] - self.center[1]
        radius = math.sqrt(dx*dx + dy*dy)
        
        # Tạo danh sách điểm hình tròn
        points = generate_circle_points(self.center, radius)
        
        # Lấy lát cắt hiện tại
        slice_idx = self.image_widget.slice_idx if self.image_widget else 0
        
        # Phát tín hiệu với contour mới
        self.contour_created.emit(self.contour_name, points, slice_idx)
    
    def apply_to_current_slice(self):
        """Áp dụng contour vào lát cắt hiện tại."""
        if self.is_drawing and self.center and self.current_point:
            self._finalize_contour()


class RectangleContourTool(ContourTool):
    """Công cụ vẽ contour hình chữ nhật."""
    
    def __init__(self):
        """Khởi tạo công cụ vẽ contour hình chữ nhật."""
        super().__init__("Rectangle")
        
        # Dữ liệu contour
        self.start_point = None  # Điểm bắt đầu
        self.current_point = None  # Điểm hiện tại khi vẽ
        self.is_drawing = False  # Đang vẽ hay không
        self.preview_color = QColor(255, 255, 0, 200)  # Màu xem trước (vàng bán trong suốt)
        self.final_color = QColor(255, 0, 0, 200)  # Màu cuối cùng (đỏ bán trong suốt)
        self.square_mode = False  # Chế độ vẽ hình vuông
    
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
            # Thiết lập điểm bắt đầu
            self.start_point = pos
            self.current_point = pos
            self.is_drawing = True
            self.update_preview()
        elif button == Qt.RightButton:
            # Hủy vẽ
            self.is_drawing = False
            self.start_point = None
            self.current_point = None
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
        
        # Cập nhật điểm hiện tại
        if self.square_mode:
            # Chế độ vẽ hình vuông - giữ tỷ lệ 1:1
            dx = abs(pos[0] - self.start_point[0])
            dy = abs(pos[1] - self.start_point[1])
            side = max(dx, dy)
            
            # Tính toán vị trí mới dựa trên hướng kéo
            new_x = self.start_point[0] + side if pos[0] >= self.start_point[0] else self.start_point[0] - side
            new_y = self.start_point[1] + side if pos[1] >= self.start_point[1] else self.start_point[1] - side
            
            self.current_point = (new_x, new_y)
        else:
            # Chế độ vẽ hình chữ nhật thông thường
            self.current_point = pos
        
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
        if not self.active or not self.image_widget or not self.is_drawing:
            return
        
        if button == Qt.LeftButton:
            # Cập nhật điểm cuối cùng trước khi hoàn thành
            if self.square_mode:
                # Chế độ vẽ hình vuông - giữ tỷ lệ 1:1
                dx = abs(pos[0] - self.start_point[0])
                dy = abs(pos[1] - self.start_point[1])
                side = max(dx, dy)
                
                # Tính toán vị trí mới dựa trên hướng kéo
                new_x = self.start_point[0] + side if pos[0] >= self.start_point[0] else self.start_point[0] - side
                new_y = self.start_point[1] + side if pos[1] >= self.start_point[1] else self.start_point[1] - side
                
                self.current_point = (new_x, new_y)
            else:
                self.current_point = pos
            
            # Hoàn thành hình chữ nhật
            self._finalize_contour()
            
            # Hủy vẽ
            self.is_drawing = False
            self.start_point = None
            self.current_point = None
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
            self.start_point = None
            self.current_point = None
            self.update_preview()
        elif key == Qt.Key_Shift:
            # Bật chế độ hình vuông
            self.square_mode = True
            if self.is_drawing and self.start_point and self.current_point:
                # Cập nhật hình dạng hiện tại
                dx = abs(self.current_point[0] - self.start_point[0])
                dy = abs(self.current_point[1] - self.start_point[1])
                side = max(dx, dy)
                
                # Tính toán vị trí mới dựa trên hướng kéo
                new_x = self.start_point[0] + side if self.current_point[0] >= self.start_point[0] else self.start_point[0] - side
                new_y = self.start_point[1] + side if self.current_point[1] >= self.start_point[1] else self.start_point[1] - side
                
                self.current_point = (new_x, new_y)
                self.update_preview()
    
    def key_release(self, key: int):
        """
        Xử lý sự kiện khi thả phím.
        
        Parameters
        ----------
        key : int
            Mã phím
        """
        if not self.active:
            return
        
        if key == Qt.Key_Shift:
            # Tắt chế độ hình vuông
            self.square_mode = False
    
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
        Vẽ hình chữ nhật tạm thời lên hình ảnh.
        
        Parameters
        ----------
        painter : QPainter
            Đối tượng QPainter để vẽ
        """
        if not self.active or not self.is_drawing or not self.start_point or not self.current_point:
            return
        
        # Thiết lập bút vẽ
        pen = QPen(self.preview_color)
        pen.setWidth(2)
        painter.setPen(pen)
        
        # Tính toán tọa độ của hình chữ nhật
        x1, y1 = self.start_point
        x2, y2 = self.current_point
        
        # Vẽ hình chữ nhật
        rect = QRectF(min(x1, x2), min(y1, y2), abs(x2-x1), abs(y2-y1))
        painter.drawRect(rect)
        
        # Vẽ các điểm góc
        painter.setPen(QPen(Qt.white, 1))
        painter.drawEllipse(QPointF(x1, y1), 3, 3)
        painter.drawEllipse(QPointF(x2, y2), 3, 3)
        painter.drawEllipse(QPointF(x1, y2), 3, 3)
        painter.drawEllipse(QPointF(x2, y1), 3, 3)
    
    def _finalize_contour(self):
        """Hoàn thành contour và phát tín hiệu."""
        if not self.start_point or not self.current_point:
            return
        
        # Tính toán tọa độ của hình chữ nhật
        x1, y1 = self.start_point
        x2, y2 = self.current_point
        
        # Tạo danh sách điểm hình chữ nhật
        top_left = (min(x1, x2), min(y1, y2))
        bottom_right = (max(x1, x2), max(y1, y2))
        points = generate_rectangle_points(top_left, bottom_right)
        
        # Lấy lát cắt hiện tại
        slice_idx = self.image_widget.slice_idx if self.image_widget else 0
        
        # Phát tín hiệu với contour mới
        self.contour_created.emit(self.contour_name, points, slice_idx)
    
    def apply_to_current_slice(self):
        """Áp dụng contour vào lát cắt hiện tại."""
        if self.is_drawing and self.start_point and self.current_point:
            self._finalize_contour()
