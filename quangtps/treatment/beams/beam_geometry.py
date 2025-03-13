#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module quản lý hình học của chùm tia xạ trị (Beam Geometry).

Module này cung cấp các lớp và phương thức để định nghĩa và quản lý
hình học của chùm tia xạ trị, bao gồm các thông số góc, khoảng cách,
và điểm tham chiếu.
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from enum import Enum

logger = logging.getLogger(__name__)

class CoordinateSystem(str, Enum):
    """Enum đại diện cho các hệ tọa độ."""
    PATIENT = "PATIENT"  # Hệ tọa độ bệnh nhân
    MACHINE = "MACHINE"  # Hệ tọa độ máy
    BEV = "BEV"          # Hệ tọa độ Beam's Eye View
    IEC = "IEC"          # Hệ tọa độ IEC
    DICOM = "DICOM"      # Hệ tọa độ DICOM


class GantryDirection(str, Enum):
    """Enum đại diện cho hướng quay của gantry."""
    CW = "CW"    # Theo chiều kim đồng hồ (Clockwise)
    CCW = "CCW"  # Ngược chiều kim đồng hồ (Counter-Clockwise)


class CollimatorDirection(str, Enum):
    """Enum đại diện cho hướng quay của collimator."""
    CW = "CW"    # Theo chiều kim đồng hồ (Clockwise)
    CCW = "CCW"  # Ngược chiều kim đồng hồ (Counter-Clockwise)


class CouchDirection(str, Enum):
    """Enum đại diện cho hướng quay của bàn điều trị."""
    CW = "CW"    # Theo chiều kim đồng hồ (Clockwise)
    CCW = "CCW"  # Ngược chiều kim đồng hồ (Counter-Clockwise)


class BeamGeometry:
    """
    Lớp đại diện cho hình học của một chùm tia xạ trị.
    
    Lớp này chứa thông tin về hình học của chùm tia xạ trị, bao gồm các
    thông số góc, khoảng cách, và điểm tham chiếu.
    """
    
    def __init__(self):
        """
        Khởi tạo một đối tượng BeamGeometry với các giá trị mặc định.
        """
        # Các góc
        self.gantry_angle = 0.0  # Góc gantry (độ)
        self.gantry_direction = GantryDirection.CW
        
        self.collimator_angle = 0.0  # Góc collimator (độ)
        self.collimator_direction = CollimatorDirection.CW
        
        self.couch_angle = 0.0  # Góc bàn điều trị (độ)
        self.couch_direction = CouchDirection.CW
        
        # Khoảng cách
        self.source_surface_distance = 100.0  # Khoảng cách từ nguồn đến bề mặt (SSD) (cm)
        self.source_axis_distance = 100.0  # Khoảng cách từ nguồn đến trục quay (SAD) (cm)
        
        # Tọa độ isocenter
        self.isocenter = (0.0, 0.0, 0.0)  # x, y, z (cm)
        
        # Kích thước trường
        self.field_size = (10.0, 10.0)  # Kích thước trường tại isocenter (cm)
        self.effective_field_size = (10.0, 10.0)  # Kích thước trường hiệu dụng (cm)
        
        # Dịch chuyển collimator
        self.collimator_x1 = -5.0  # Vị trí cạnh X1 của collimator (cm)
        self.collimator_x2 = 5.0   # Vị trí cạnh X2 của collimator (cm)
        self.collimator_y1 = -5.0  # Vị trí cạnh Y1 của collimator (cm)
        self.collimator_y2 = 5.0   # Vị trí cạnh Y2 của collimator (cm)
        
        # Thông tin bổ sung
        self.metadata = {}
    
    def set_gantry_angle(self, angle: float, direction: GantryDirection = GantryDirection.CW):
        """
        Thiết lập góc gantry.
        
        Parameters
        ----------
        angle : float
            Góc gantry (độ)
        direction : GantryDirection, optional
            Hướng quay của gantry
        """
        self.gantry_angle = angle
        self.gantry_direction = direction
    
    def set_collimator_angle(self, angle: float, direction: CollimatorDirection = CollimatorDirection.CW):
        """
        Thiết lập góc collimator.
        
        Parameters
        ----------
        angle : float
            Góc collimator (độ)
        direction : CollimatorDirection, optional
            Hướng quay của collimator
        """
        self.collimator_angle = angle
        self.collimator_direction = direction
    
    def set_couch_angle(self, angle: float, direction: CouchDirection = CouchDirection.CW):
        """
        Thiết lập góc bàn điều trị.
        
        Parameters
        ----------
        angle : float
            Góc bàn điều trị (độ)
        direction : CouchDirection, optional
            Hướng quay của bàn điều trị
        """
        self.couch_angle = angle
        self.couch_direction = direction
    
    def set_isocenter(self, x: float, y: float, z: float):
        """
        Thiết lập tọa độ isocenter.
        
        Parameters
        ----------
        x : float
            Tọa độ x của isocenter (cm)
        y : float
            Tọa độ y của isocenter (cm)
        z : float
            Tọa độ z của isocenter (cm)
        """
        self.isocenter = (x, y, z)
    
    def set_field_size(self, width: float, height: float):
        """
        Thiết lập kích thước trường tại isocenter.
        
        Parameters
        ----------
        width : float
            Chiều rộng trường (cm)
        height : float
            Chiều cao trường (cm)
        """
        self.field_size = (width, height)
        self.collimator_x1 = -width / 2
        self.collimator_x2 = width / 2
        self.collimator_y1 = -height / 2
        self.collimator_y2 = height / 2
        self.effective_field_size = (width, height)
    
    def set_collimator_positions(self, x1: float, x2: float, y1: float, y2: float):
        """
        Thiết lập vị trí các cạnh của collimator.
        
        Parameters
        ----------
        x1 : float
            Vị trí cạnh X1 của collimator (cm)
        x2 : float
            Vị trí cạnh X2 của collimator (cm)
        y1 : float
            Vị trí cạnh Y1 của collimator (cm)
        y2 : float
            Vị trí cạnh Y2 của collimator (cm)
        """
        self.collimator_x1 = x1
        self.collimator_x2 = x2
        self.collimator_y1 = y1
        self.collimator_y2 = y2
        self.field_size = (abs(x2 - x1), abs(y2 - y1))
        self.effective_field_size = self.field_size
    
    def set_source_surface_distance(self, ssd: float):
        """
        Thiết lập khoảng cách từ nguồn đến bề mặt (SSD).
        
        Parameters
        ----------
        ssd : float
            Khoảng cách từ nguồn đến bề mặt (cm)
        """
        self.source_surface_distance = ssd
    
    def set_source_axis_distance(self, sad: float):
        """
        Thiết lập khoảng cách từ nguồn đến trục quay (SAD).
        
        Parameters
        ----------
        sad : float
            Khoảng cách từ nguồn đến trục quay (cm)
        """
        self.source_axis_distance = sad
    
    def get_beam_eye_view_coordinates(self, point: Tuple[float, float, float]) -> Tuple[float, float]:
        """
        Chuyển đổi tọa độ từ hệ tọa độ bệnh nhân sang hệ tọa độ Beam's Eye View (BEV).
        
        Parameters
        ----------
        point : Tuple[float, float, float]
            Tọa độ điểm trong hệ tọa độ bệnh nhân (cm)
            
        Returns
        -------
        Tuple[float, float]
            Tọa độ điểm trong hệ tọa độ BEV (cm)
        """
        # Tính toán vector từ isocenter đến điểm
        x, y, z = point
        iso_x, iso_y, iso_z = self.isocenter
        dx = x - iso_x
        dy = y - iso_y
        dz = z - iso_z
        
        # Tính toán góc trong hệ tọa độ radian
        gantry_rad = np.radians(self.gantry_angle)
        collimator_rad = np.radians(self.collimator_angle)
        couch_rad = np.radians(self.couch_angle)
        
        # Áp dụng phép biến đổi cho góc gantry
        dx_gantry = dx * np.cos(gantry_rad) + dz * np.sin(gantry_rad)
        dy_gantry = dy
        dz_gantry = -dx * np.sin(gantry_rad) + dz * np.cos(gantry_rad)
        
        # Áp dụng phép biến đổi cho góc collimator
        dx_collimator = dx_gantry * np.cos(collimator_rad) - dy_gantry * np.sin(collimator_rad)
        dy_collimator = dx_gantry * np.sin(collimator_rad) + dy_gantry * np.cos(collimator_rad)
        
        # Tính toán tọa độ BEV
        magnification = self.source_axis_distance / (self.source_axis_distance - dz_gantry)
        bev_x = dx_collimator * magnification
        bev_y = dy_collimator * magnification
        
        return (bev_x, bev_y)
    
    def get_source_position(self) -> Tuple[float, float, float]:
        """
        Lấy tọa độ của nguồn trong hệ tọa độ bệnh nhân.
        
        Returns
        -------
        Tuple[float, float, float]
            Tọa độ của nguồn (cm)
        """
        iso_x, iso_y, iso_z = self.isocenter
        
        # Tính toán góc trong hệ tọa độ radian
        gantry_rad = np.radians(self.gantry_angle)
        
        # Tính toán vị trí nguồn dựa trên góc gantry và SAD
        source_x = iso_x - self.source_axis_distance * np.sin(gantry_rad)
        source_y = iso_y
        source_z = iso_z - self.source_axis_distance * np.cos(gantry_rad)
        
        return (source_x, source_y, source_z)
    
    def calculate_effective_field_size(self, modifiers: List[Any] = None) -> Tuple[float, float]:
        """
        Tính toán kích thước trường hiệu dụng sau khi áp dụng các bộ điều chỉnh.
        
        Parameters
        ----------
        modifiers : List[Any], optional
            Danh sách các bộ điều chỉnh
            
        Returns
        -------
        Tuple[float, float]
            Kích thước trường hiệu dụng (cm)
        """
        if modifiers is None:
            return self.field_size
        
        width, height = self.field_size
        
        # Tính toán ảnh hưởng của các bộ điều chỉnh
        for modifier in modifiers:
            if hasattr(modifier, 'get_effective_size_change'):
                mod_width, mod_height = modifier.get_effective_size_change()
                width = max(0, width - mod_width)
                height = max(0, height - mod_height)
        
        self.effective_field_size = (width, height)
        return self.effective_field_size
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Chuyển đổi thông tin hình học chùm tia thành dictionary.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary chứa thông tin hình học chùm tia
        """
        return {
            "gantry_angle": self.gantry_angle,
            "gantry_direction": self.gantry_direction.value,
            "collimator_angle": self.collimator_angle,
            "collimator_direction": self.collimator_direction.value,
            "couch_angle": self.couch_angle,
            "couch_direction": self.couch_direction.value,
            "source_surface_distance": self.source_surface_distance,
            "source_axis_distance": self.source_axis_distance,
            "isocenter": self.isocenter,
            "field_size": self.field_size,
            "effective_field_size": self.effective_field_size,
            "collimator_x1": self.collimator_x1,
            "collimator_x2": self.collimator_x2,
            "collimator_y1": self.collimator_y1,
            "collimator_y2": self.collimator_y2,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BeamGeometry':
        """
        Tạo đối tượng BeamGeometry từ dictionary.
        
        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary chứa thông tin hình học chùm tia
            
        Returns
        -------
        BeamGeometry
            Đối tượng BeamGeometry
        """
        geometry = cls()
        
        # Cập nhật các thuộc tính
        geometry.gantry_angle = data["gantry_angle"]
        geometry.gantry_direction = GantryDirection(data["gantry_direction"])
        
        geometry.collimator_angle = data["collimator_angle"]
        geometry.collimator_direction = CollimatorDirection(data["collimator_direction"])
        
        geometry.couch_angle = data["couch_angle"]
        geometry.couch_direction = CouchDirection(data["couch_direction"])
        
        geometry.source_surface_distance = data["source_surface_distance"]
        geometry.source_axis_distance = data["source_axis_distance"]
        
        geometry.isocenter = data["isocenter"]
        geometry.field_size = data["field_size"]
        geometry.effective_field_size = data["effective_field_size"]
        
        geometry.collimator_x1 = data["collimator_x1"]
        geometry.collimator_x2 = data["collimator_x2"]
        geometry.collimator_y1 = data["collimator_y1"]
        geometry.collimator_y2 = data["collimator_y2"]
        
        geometry.metadata = data.get("metadata", {})
        
        return geometry