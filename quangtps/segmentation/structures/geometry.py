#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module định nghĩa các đối tượng hình học cho phân đoạn trong QuangTPS.

Module này cung cấp các lớp cơ bản như Point (điểm), Contour (đường bao),
và các hàm phụ trợ để thao tác với chúng.
"""

import math
import numpy as np
from typing import List, Tuple, Union, Optional, Dict, Any
from dataclasses import dataclass
import uuid

@dataclass
class Point:
    """
    Đối tượng đại diện cho một điểm trong không gian 3D.
    
    Attributes:
        x (float): Tọa độ x
        y (float): Tọa độ y
        z (float): Tọa độ z
    """
    x: float
    y: float
    z: float = 0.0
    
    def to_tuple(self) -> Tuple[float, float, float]:
        """
        Chuyển đổi thành tuple.
        
        Returns:
            Tuple[float, float, float]: (x, y, z)
        """
        return (self.x, self.y, self.z)
    
    def to_list(self) -> List[float]:
        """
        Chuyển đổi thành list.
        
        Returns:
            List[float]: [x, y, z]
        """
        return [self.x, self.y, self.z]
    
    def to_dict(self) -> Dict[str, float]:
        """
        Chuyển đổi thành dictionary.
        
        Returns:
            Dict[str, float]: {"x": x, "y": y, "z": z}
        """
        return {"x": self.x, "y": self.y, "z": self.z}
    
    def to_numpy(self) -> np.ndarray:
        """
        Chuyển đổi thành numpy array.
        
        Returns:
            np.ndarray: shape (3,) chứa [x, y, z]
        """
        return np.array([self.x, self.y, self.z])
    
    def distance_to(self, other: 'Point') -> float:
        """
        Tính khoảng cách Euclidean đến điểm khác.
        
        Parameters:
            other (Point): Điểm cần tính khoảng cách đến
            
        Returns:
            float: Khoảng cách Euclidean
        """
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2 + (self.z - other.z)**2)
    
    def distance_to_xy(self, other: 'Point') -> float:
        """
        Tính khoảng cách Euclidean trên mặt phẳng xy đến điểm khác.
        
        Parameters:
            other (Point): Điểm cần tính khoảng cách đến
            
        Returns:
            float: Khoảng cách Euclidean trên mặt phẳng xy
        """
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
    
    @classmethod
    def from_tuple(cls, coords: Tuple[float, float, float]) -> 'Point':
        """
        Tạo đối tượng Point từ tuple.
        
        Parameters:
            coords (Tuple[float, float, float]): Tuple (x, y, z)
            
        Returns:
            Point: Đối tượng Point mới
        """
        return cls(x=coords[0], y=coords[1], z=coords[2] if len(coords) > 2 else 0.0)
    
    @classmethod
    def from_list(cls, coords: List[float]) -> 'Point':
        """
        Tạo đối tượng Point từ list.
        
        Parameters:
            coords (List[float]): List [x, y, z]
            
        Returns:
            Point: Đối tượng Point mới
        """
        return cls(x=coords[0], y=coords[1], z=coords[2] if len(coords) > 2 else 0.0)
    
    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'Point':
        """
        Tạo đối tượng Point từ dictionary.
        
        Parameters:
            data (Dict[str, float]): Dictionary chứa "x", "y", và tùy chọn "z"
            
        Returns:
            Point: Đối tượng Point mới
        """
        return cls(x=data["x"], y=data["y"], z=data.get("z", 0.0))
    
    @classmethod
    def from_numpy(cls, array: np.ndarray) -> 'Point':
        """
        Tạo đối tượng Point từ numpy array.
        
        Parameters:
            array (np.ndarray): Array shape (3,) hoặc (2,)
            
        Returns:
            Point: Đối tượng Point mới
        """
        return cls(x=array[0], y=array[1], z=array[2] if len(array) > 2 else 0.0)


class Contour:
    """
    Đối tượng đại diện cho một đường bao (contour) trong không gian 3D.
    
    Attributes:
        points (List[Point]): Danh sách các điểm trong đường bao
        z (float): Giá trị z của đường bao (giả sử tất cả các điểm có cùng z)
        closed (bool): Nếu True, đường bao được coi là khép kín (điểm đầu = điểm cuối)
        id (str): ID duy nhất của đường bao
    """
    
    def __init__(self, points: Optional[List[Point]] = None, z: float = 0.0, 
                closed: bool = True, contour_id: Optional[str] = None):
        """
        Khởi tạo đường bao.
        
        Parameters:
            points (Optional[List[Point]]): Danh sách các điểm
            z (float): Giá trị z
            closed (bool): Nếu True, đường bao được coi là khép kín
            contour_id (Optional[str]): ID duy nhất, tự động tạo nếu không cung cấp
        """
        self.points = points or []
        self.z = z
        self.closed = closed
        self.id = contour_id or str(uuid.uuid4())
        
        # Đảm bảo tất cả các điểm có cùng giá trị z
        for point in self.points:
            point.z = z
    
    def add_point(self, point: Point):
        """
        Thêm một điểm vào đường bao.
        
        Parameters:
            point (Point): Điểm cần thêm
        """
        point.z = self.z  # Đảm bảo điểm mới có cùng z
        self.points.append(point)
    
    def close(self):
        """Đóng đường bao bằng cách thêm điểm đầu tiên vào cuối nếu cần."""
        if not self.closed or len(self.points) < 2:
            return
            
        first_point = self.points[0]
        last_point = self.points[-1]
        
        # Nếu điểm đầu và cuối không trùng nhau, thêm điểm đầu vào cuối
        if first_point.distance_to_xy(last_point) > 1e-6:  # Sai số nhỏ
            self.points.append(Point(first_point.x, first_point.y, self.z))
    
    def to_numpy(self) -> np.ndarray:
        """
        Chuyển đổi thành numpy array.
        
        Returns:
            np.ndarray: shape (n, 3) chứa các điểm [x, y, z]
        """
        return np.array([p.to_list() for p in self.points])
    
    def to_list(self) -> List[List[float]]:
        """
        Chuyển đổi thành list.
        
        Returns:
            List[List[float]]: Danh sách các điểm [[x, y, z], ...]
        """
        return [p.to_list() for p in self.points]
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Chuyển đổi thành dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary chứa thông tin đường bao
        """
        return {
            "id": self.id,
            "z": self.z,
            "closed": self.closed,
            "points": [p.to_dict() for p in self.points]
        }
    
    def get_length(self) -> float:
        """
        Tính chiều dài đường bao.
        
        Returns:
            float: Tổng chiều dài của tất cả các đoạn
        """
        total_length = 0.0
        for i in range(len(self.points) - 1):
            total_length += self.points[i].distance_to_xy(self.points[i+1])
            
        # Thêm đoạn cuối nếu đường bao khép kín
        if self.closed and len(self.points) > 1:
            total_length += self.points[-1].distance_to_xy(self.points[0])
            
        return total_length
    
    def get_area(self) -> float:
        """
        Tính diện tích của đường bao (giả sử đường bao là đa giác đơn trên mặt phẳng xy).
        
        Returns:
            float: Diện tích đa giác
        """
        if len(self.points) < 3:
            return 0.0
            
        # Đảm bảo đường bao khép kín
        self.close()
        
        # Sử dụng công thức Shoelace (Gauss's area formula)
        area = 0.0
        for i in range(len(self.points) - 1):
            p1 = self.points[i]
            p2 = self.points[i+1]
            area += (p1.x * p2.y - p2.x * p1.y)
            
        return abs(area) / 2.0
    
    def is_point_inside(self, point: Point) -> bool:
        """
        Kiểm tra xem một điểm có nằm trong đường bao hay không (trên mặt phẳng xy).
        
        Parameters:
            point (Point): Điểm cần kiểm tra
            
        Returns:
            bool: True nếu điểm nằm trong đường bao
        """
        if len(self.points) < 3:
            return False
            
        # Đảm bảo đường bao khép kín
        self.close()
        
        # Sử dụng thuật toán ray casting
        x, y = point.x, point.y
        inside = False
        
        for i in range(len(self.points) - 1):
            p1 = self.points[i]
            p2 = self.points[i+1]
            
            if ((p1.y > y) != (p2.y > y)) and (x < (p2.x - p1.x) * (y - p1.y) / (p2.y - p1.y) + p1.x):
                inside = not inside
                
        return inside
    
    @classmethod
    def from_numpy(cls, array: np.ndarray, z: float = 0.0, closed: bool = True) -> 'Contour':
        """
        Tạo đối tượng Contour từ numpy array.
        
        Parameters:
            array (np.ndarray): Array shape (n, 3) hoặc (n, 2)
            z (float): Giá trị z nếu array có shape (n, 2)
            closed (bool): Nếu True, đường bao được coi là khép kín
            
        Returns:
            Contour: Đối tượng Contour mới
        """
        points = []
        for i in range(array.shape[0]):
            if array.shape[1] == 3:
                points.append(Point(array[i, 0], array[i, 1], array[i, 2]))
            else:
                points.append(Point(array[i, 0], array[i, 1], z))
                
        return cls(points=points, z=z, closed=closed)
    
    @classmethod
    def from_list(cls, points_list: List[List[float]], z: float = 0.0, closed: bool = True) -> 'Contour':
        """
        Tạo đối tượng Contour từ list.
        
        Parameters:
            points_list (List[List[float]]): List các điểm [[x, y, z], ...]
            z (float): Giá trị z nếu các điểm không có z
            closed (bool): Nếu True, đường bao được coi là khép kín
            
        Returns:
            Contour: Đối tượng Contour mới
        """
        points = []
        for point_data in points_list:
            if len(point_data) == 3:
                points.append(Point(point_data[0], point_data[1], point_data[2]))
            else:
                points.append(Point(point_data[0], point_data[1], z))
                
        return cls(points=points, z=z, closed=closed)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Contour':
        """
        Tạo đối tượng Contour từ dictionary.
        
        Parameters:
            data (Dict[str, Any]): Dictionary chứa thông tin đường bao
            
        Returns:
            Contour: Đối tượng Contour mới
        """
        points = [Point.from_dict(p) for p in data.get("points", [])]
        return cls(
            points=points,
            z=data.get("z", 0.0),
            closed=data.get("closed", True),
            contour_id=data.get("id")
        )
