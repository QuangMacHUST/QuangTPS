#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module hệ tọa độ không gian 3D.

Module này cung cấp các lớp và hàm để quản lý hệ tọa độ trong không gian 3D,
phục vụ cho việc chuyển đổi giữa các hệ tọa độ khác nhau trong xạ trị.
"""

import numpy as np
from typing import Tuple, List, Union, Optional
from dataclasses import dataclass
from enum import Enum, auto

class CoordinateSystemType(Enum):
    """Loại hệ tọa độ sử dụng trong xạ trị."""
    DICOM = auto()      # Hệ tọa độ DICOM (LPS: Left, Posterior, Superior)
    IEC = auto()        # Hệ tọa độ IEC (máy điều trị)
    VARIAN = auto()     # Hệ tọa độ Varian
    ELEKTA = auto()     # Hệ tọa độ Elekta
    LOCAL = auto()      # Hệ tọa độ cục bộ (tùy chỉnh)

@dataclass
class Point3D:
    """Điểm trong không gian 3D."""
    x: float
    y: float
    z: float
    
    def as_array(self) -> np.ndarray:
        """Chuyển đổi thành mảng numpy."""
        return np.array([self.x, self.y, self.z])
    
    def distance_to(self, other: 'Point3D') -> float:
        """Tính khoảng cách đến một điểm khác."""
        return np.linalg.norm(self.as_array() - other.as_array())
    
    def __add__(self, other: 'Point3D') -> 'Point3D':
        """Phép cộng hai điểm."""
        return Point3D(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other: 'Point3D') -> 'Point3D':
        """Phép trừ hai điểm."""
        return Point3D(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __mul__(self, scalar: float) -> 'Point3D':
        """Phép nhân điểm với một số."""
        return Point3D(self.x * scalar, self.y * scalar, self.z * scalar)
    
    def __truediv__(self, scalar: float) -> 'Point3D':
        """Phép chia điểm cho một số."""
        return Point3D(self.x / scalar, self.y / scalar, self.z / scalar)

@dataclass
class Vector3D:
    """Vector trong không gian 3D."""
    x: float
    y: float
    z: float
    
    def as_array(self) -> np.ndarray:
        """Chuyển đổi thành mảng numpy."""
        return np.array([self.x, self.y, self.z])
    
    def magnitude(self) -> float:
        """Tính độ lớn của vector."""
        return np.linalg.norm(self.as_array())
    
    def normalize(self) -> 'Vector3D':
        """Chuẩn hóa vector (độ lớn = 1)."""
        mag = self.magnitude()
        if mag > 0:
            return Vector3D(self.x / mag, self.y / mag, self.z / mag)
        return Vector3D(0, 0, 0)
    
    def dot(self, other: 'Vector3D') -> float:
        """Tích vô hướng với vector khác."""
        return np.dot(self.as_array(), other.as_array())
    
    def cross(self, other: 'Vector3D') -> 'Vector3D':
        """Tích có hướng với vector khác."""
        result = np.cross(self.as_array(), other.as_array())
        return Vector3D(result[0], result[1], result[2])
    
    def angle(self, other: 'Vector3D') -> float:
        """Tính góc giữa hai vector (radian)."""
        dot_product = self.dot(other)
        magnitudes = self.magnitude() * other.magnitude()
        if magnitudes > 0:
            # Xử lý lỗi số học do làm tròn
            cos_angle = max(min(dot_product / magnitudes, 1.0), -1.0)
            return np.arccos(cos_angle)
        return 0.0
    
    def __add__(self, other: 'Vector3D') -> 'Vector3D':
        """Phép cộng hai vector."""
        return Vector3D(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other: 'Vector3D') -> 'Vector3D':
        """Phép trừ hai vector."""
        return Vector3D(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __mul__(self, scalar: float) -> 'Vector3D':
        """Phép nhân vector với một số."""
        return Vector3D(self.x * scalar, self.y * scalar, self.z * scalar)
    
    def __truediv__(self, scalar: float) -> 'Vector3D':
        """Phép chia vector cho một số."""
        return Vector3D(self.x / scalar, self.y / scalar, self.z / scalar)

class CoordinateSystem:
    """
    Hệ tọa độ trong không gian 3D.
    
    Lớp này đại diện cho một hệ tọa độ trong không gian 3D, với gốc tọa độ
    và ba trục cơ sở. Cho phép chuyển đổi điểm và vector giữa các hệ tọa độ khác nhau.
    """
    
    def __init__(self, 
                origin: Optional[Point3D] = None, 
                x_axis: Optional[Vector3D] = None, 
                y_axis: Optional[Vector3D] = None, 
                z_axis: Optional[Vector3D] = None,
                cs_type: CoordinateSystemType = CoordinateSystemType.LOCAL):
        """
        Khởi tạo hệ tọa độ.
        
        Args:
            origin: Gốc tọa độ. Mặc định là (0, 0, 0).
            x_axis: Trục X. Mặc định là (1, 0, 0).
            y_axis: Trục Y. Mặc định là (0, 1, 0).
            z_axis: Trục Z. Mặc định là (0, 0, 1).
            cs_type: Loại hệ tọa độ. Mặc định là LOCAL.
        """
        # Gốc tọa độ
        self.origin = origin if origin else Point3D(0, 0, 0)
        
        # Trục cơ sở
        self.x_axis = x_axis.normalize() if x_axis else Vector3D(1, 0, 0)
        self.y_axis = y_axis.normalize() if y_axis else Vector3D(0, 1, 0)
        self.z_axis = z_axis.normalize() if z_axis else Vector3D(0, 0, 1)
        
        # Loại hệ tọa độ
        self.cs_type = cs_type
        
        # Ma trận chuyển đổi (từ hệ tọa độ cục bộ sang hệ tọa độ toàn cục)
        self._update_transform_matrix()
    
    def _update_transform_matrix(self):
        """Cập nhật ma trận chuyển đổi."""
        # Ma trận xoay (3x3)
        self.rotation_matrix = np.array([
            [self.x_axis.x, self.y_axis.x, self.z_axis.x],
            [self.x_axis.y, self.y_axis.y, self.z_axis.y],
            [self.x_axis.z, self.y_axis.z, self.z_axis.z]
        ])
        
        # Ma trận dịch chuyển (3x1)
        self.translation_vector = self.origin.as_array()
        
        # Ma trận chuyển đổi đồng nhất (4x4)
        self.transform_matrix = np.eye(4)
        self.transform_matrix[:3, :3] = self.rotation_matrix
        self.transform_matrix[:3, 3] = self.translation_vector
        
        # Ma trận nghịch đảo
        self.inverse_transform_matrix = np.linalg.inv(self.transform_matrix)
    
    def set_origin(self, origin: Point3D):
        """Đặt gốc tọa độ mới."""
        self.origin = origin
        self._update_transform_matrix()
    
    def set_axes(self, x_axis: Vector3D, y_axis: Vector3D, z_axis: Vector3D):
        """Đặt các trục cơ sở mới."""
        self.x_axis = x_axis.normalize()
        self.y_axis = y_axis.normalize()
        self.z_axis = z_axis.normalize()
        self._update_transform_matrix()
    
    def local_to_global(self, point: Union[Point3D, Vector3D]) -> Union[Point3D, Vector3D]:
        """
        Chuyển đổi điểm/vector từ hệ tọa độ cục bộ sang hệ tọa độ toàn cục.
        
        Args:
            point: Điểm hoặc vector trong hệ tọa độ cục bộ.
            
        Returns:
            Điểm hoặc vector đã chuyển đổi sang hệ tọa độ toàn cục.
        """
        is_point = isinstance(point, Point3D)
        
        # Chuyển đổi sang mảng numpy
        if is_point:
            point_array = np.append(point.as_array(), 1)  # Điểm đồng nhất
        else:
            point_array = np.append(point.as_array(), 0)  # Vector đồng nhất
        
        # Áp dụng phép biến đổi
        result_array = np.dot(self.transform_matrix, point_array)
        
        # Chuyển đổi kết quả về kiểu ban đầu
        if is_point:
            return Point3D(result_array[0], result_array[1], result_array[2])
        else:
            return Vector3D(result_array[0], result_array[1], result_array[2])
    
    def global_to_local(self, point: Union[Point3D, Vector3D]) -> Union[Point3D, Vector3D]:
        """
        Chuyển đổi điểm/vector từ hệ tọa độ toàn cục sang hệ tọa độ cục bộ.
        
        Args:
            point: Điểm hoặc vector trong hệ tọa độ toàn cục.
            
        Returns:
            Điểm hoặc vector đã chuyển đổi sang hệ tọa độ cục bộ.
        """
        is_point = isinstance(point, Point3D)
        
        # Chuyển đổi sang mảng numpy
        if is_point:
            point_array = np.append(point.as_array(), 1)  # Điểm đồng nhất
        else:
            point_array = np.append(point.as_array(), 0)  # Vector đồng nhất
        
        # Áp dụng phép biến đổi nghịch đảo
        result_array = np.dot(self.inverse_transform_matrix, point_array)
        
        # Chuyển đổi kết quả về kiểu ban đầu
        if is_point:
            return Point3D(result_array[0], result_array[1], result_array[2])
        else:
            return Vector3D(result_array[0], result_array[1], result_array[2])
    
    def transform_to(self, target_cs: 'CoordinateSystem', 
                   point: Union[Point3D, Vector3D]) -> Union[Point3D, Vector3D]:
        """
        Chuyển đổi điểm/vector từ hệ tọa độ hiện tại sang hệ tọa độ đích.
        
        Args:
            target_cs: Hệ tọa độ đích.
            point: Điểm hoặc vector trong hệ tọa độ hiện tại.
            
        Returns:
            Điểm hoặc vector đã chuyển đổi sang hệ tọa độ đích.
        """
        # Chuyển đổi từ hệ tọa độ hiện tại sang hệ tọa độ toàn cục
        global_point = self.local_to_global(point)
        
        # Chuyển đổi từ hệ tọa độ toàn cục sang hệ tọa độ đích
        return target_cs.global_to_local(global_point)
    
    @classmethod
    def create_dicom_cs(cls) -> 'CoordinateSystem':
        """
        Tạo hệ tọa độ DICOM (LPS: Left, Posterior, Superior).
        
        DICOM định nghĩa:
        - Trục X dương hướng sang trái bệnh nhân
        - Trục Y dương hướng về phía sau bệnh nhân
        - Trục Z dương hướng lên trên (đỉnh đầu)
        
        Returns:
            Hệ tọa độ DICOM.
        """
        return cls(
            origin=Point3D(0, 0, 0),
            x_axis=Vector3D(1, 0, 0),
            y_axis=Vector3D(0, 1, 0),
            z_axis=Vector3D(0, 0, 1),
            cs_type=CoordinateSystemType.DICOM
        )
    
    @classmethod
    def create_iec_cs(cls) -> 'CoordinateSystem':
        """
        Tạo hệ tọa độ IEC (máy điều trị).
        
        IEC định nghĩa:
        - Trục X dương hướng sang phải khi nhìn từ gantry
        - Trục Y dương hướng lên trên khi nhìn từ gantry
        - Trục Z dương hướng về phía gantry
        
        Returns:
            Hệ tọa độ IEC.
        """
        return cls(
            origin=Point3D(0, 0, 0),
            x_axis=Vector3D(1, 0, 0),
            y_axis=Vector3D(0, 1, 0),
            z_axis=Vector3D(0, 0, 1),
            cs_type=CoordinateSystemType.IEC
        )
    
    @classmethod
    def dicom_to_iec(cls, point: Union[Point3D, Vector3D]) -> Union[Point3D, Vector3D]:
        """
        Chuyển đổi trực tiếp từ hệ tọa độ DICOM sang hệ tọa độ IEC.
        
        Phép chuyển đổi từ DICOM (LPS) sang IEC:
        - X_IEC = -X_DICOM
        - Y_IEC = -Z_DICOM
        - Z_IEC = Y_DICOM
        
        Args:
            point: Điểm hoặc vector trong hệ tọa độ DICOM.
            
        Returns:
            Điểm hoặc vector đã chuyển đổi sang hệ tọa độ IEC.
        """
        is_point = isinstance(point, Point3D)
        
        if is_point:
            return Point3D(-point.x, -point.z, point.y)
        else:
            return Vector3D(-point.x, -point.z, point.y)
    
    @classmethod
    def iec_to_dicom(cls, point: Union[Point3D, Vector3D]) -> Union[Point3D, Vector3D]:
        """
        Chuyển đổi trực tiếp từ hệ tọa độ IEC sang hệ tọa độ DICOM.
        
        Phép chuyển đổi từ IEC sang DICOM (LPS):
        - X_DICOM = -X_IEC
        - Y_DICOM = Z_IEC
        - Z_DICOM = -Y_IEC
        
        Args:
            point: Điểm hoặc vector trong hệ tọa độ IEC.
            
        Returns:
            Điểm hoặc vector đã chuyển đổi sang hệ tọa độ DICOM.
        """
        is_point = isinstance(point, Point3D)
        
        if is_point:
            return Point3D(-point.x, point.z, -point.y)
        else:
            return Vector3D(-point.x, point.z, -point.y)

def create_acs_from_markers(origin_marker: Point3D, 
                          x_marker: Point3D, 
                          y_marker: Point3D) -> CoordinateSystem:
    """
    Tạo hệ tọa độ từ ba điểm đánh dấu.
    
    Args:
        origin_marker: Điểm gốc (điểm đánh dấu số 1).
        x_marker: Điểm đánh dấu số 2, dùng để xác định hướng của trục X.
        y_marker: Điểm đánh dấu số 3, dùng để xác định mặt phẳng XY.
        
    Returns:
        Hệ tọa độ được tạo từ ba điểm đánh dấu.
    """
    # Tạo trục X
    x_vec = Vector3D(
        x_marker.x - origin_marker.x,
        x_marker.y - origin_marker.y,
        x_marker.z - origin_marker.z
    )
    
    # Tạo vector từ origin đến y_marker
    temp_vec = Vector3D(
        y_marker.x - origin_marker.x,
        y_marker.y - origin_marker.y,
        y_marker.z - origin_marker.z
    )
    
    # Tạo trục Z bằng tích có hướng của x_vec và temp_vec
    z_vec = x_vec.cross(temp_vec)
    
    # Tạo trục Y bằng tích có hướng của z_vec và x_vec
    y_vec = z_vec.cross(x_vec)
    
    # Chuẩn hóa các vector
    x_vec = x_vec.normalize()
    y_vec = y_vec.normalize()
    z_vec = z_vec.normalize()
    
    # Tạo hệ tọa độ
    return CoordinateSystem(
        origin=origin_marker,
        x_axis=x_vec,
        y_axis=y_vec,
        z_axis=z_vec
    )

def rotate_around_axis(point: Union[Point3D, Vector3D], 
                      axis: Vector3D, 
                      angle: float,
                      pivot: Optional[Point3D] = None) -> Union[Point3D, Vector3D]:
    """
    Xoay một điểm hoặc vector quanh một trục.
    
    Args:
        point: Điểm hoặc vector cần xoay.
        axis: Trục xoay (vector đơn vị).
        angle: Góc xoay (radian).
        pivot: Điểm trung tâm xoay. Mặc định là gốc tọa độ.
        
    Returns:
        Điểm hoặc vector sau khi xoay.
    """
    # Chuẩn hóa trục xoay
    axis = axis.normalize()
    
    # Xác định điểm trung tâm xoay
    if pivot is None:
        pivot = Point3D(0, 0, 0)
    
    is_point = isinstance(point, Point3D)
    
    # Chuyển đổi sang mảng numpy
    if is_point:
        # Dịch chuyển về gốc tọa độ
        point_array = np.array([point.x - pivot.x, point.y - pivot.y, point.z - pivot.z])
    else:
        point_array = np.array([point.x, point.y, point.z])
    
    # Ma trận xoay theo công thức Rodrigues
    a = axis.x
    b = axis.y
    c = axis.z
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    
    rotation_matrix = np.array([
        [cos_angle + a*a*(1-cos_angle),    a*b*(1-cos_angle) - c*sin_angle, a*c*(1-cos_angle) + b*sin_angle],
        [b*a*(1-cos_angle) + c*sin_angle,  cos_angle + b*b*(1-cos_angle),   b*c*(1-cos_angle) - a*sin_angle],
        [c*a*(1-cos_angle) - b*sin_angle,  c*b*(1-cos_angle) + a*sin_angle, cos_angle + c*c*(1-cos_angle)]
    ])
    
    # Áp dụng phép xoay
    rotated_array = np.dot(rotation_matrix, point_array)
    
    # Chuyển đổi kết quả về kiểu ban đầu
    if is_point:
        # Dịch chuyển trở lại
        return Point3D(
            rotated_array[0] + pivot.x,
            rotated_array[1] + pivot.y,
            rotated_array[2] + pivot.z
        )
    else:
        return Vector3D(rotated_array[0], rotated_array[1], rotated_array[2])

if __name__ == "__main__":
    # Ví dụ sử dụng
    # Tạo điểm và vector
    p1 = Point3D(1, 2, 3)
    p2 = Point3D(4, 5, 6)
    v1 = Vector3D(1, 0, 0)
    
    # Tính khoảng cách giữa hai điểm
    distance = p1.distance_to(p2)
    print(f"Khoảng cách giữa p1 và p2: {distance}")
    
    # Tạo hệ tọa độ
    cs1 = CoordinateSystem(
        origin=Point3D(1, 1, 1),
        x_axis=Vector3D(1, 0, 0),
        y_axis=Vector3D(0, 1, 0),
        z_axis=Vector3D(0, 0, 1)
    )
    
    # Chuyển đổi điểm từ hệ tọa độ cục bộ sang hệ tọa độ toàn cục
    global_p = cs1.local_to_global(p1)
    print(f"Điểm p1 trong hệ tọa độ toàn cục: ({global_p.x}, {global_p.y}, {global_p.z})")
    
    # Chuyển đổi điểm từ hệ tọa độ toàn cục sang hệ tọa độ cục bộ
    local_p = cs1.global_to_local(global_p)
    print(f"Điểm global_p trong hệ tọa độ cục bộ: ({local_p.x}, {local_p.y}, {local_p.z})")
    
    # Chuyển đổi từ DICOM sang IEC
    dicom_point = Point3D(10, 20, 30)  # Điểm trong hệ tọa độ DICOM
    iec_point = CoordinateSystem.dicom_to_iec(dicom_point)
    print(f"Điểm DICOM ({dicom_point.x}, {dicom_point.y}, {dicom_point.z}) trong hệ tọa độ IEC: ({iec_point.x}, {iec_point.y}, {iec_point.z})")
    
    # Xoay điểm quanh một trục
    rotated_point = rotate_around_axis(p1, Vector3D(0, 0, 1), np.pi/2)
    print(f"Điểm p1 sau khi xoay 90 độ quanh trục Z: ({rotated_point.x}, {rotated_point.y}, {rotated_point.z})") 