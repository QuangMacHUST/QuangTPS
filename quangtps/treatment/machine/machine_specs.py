#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module quản lý thông số kỹ thuật của máy xạ trị.

Module này cung cấp các lớp và phương thức để định nghĩa và quản lý
các thông số kỹ thuật của các loại máy gia tốc khác nhau.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

class MachineSpecification:
    """
    Lớp đại diện cho thông số kỹ thuật của một máy xạ trị.
    
    Lớp này chứa thông tin về các thông số kỹ thuật của một máy xạ trị,
    như giới hạn gantry, bàn và collimator, tốc độ di chuyển, và các thông số khác.
    """
    
    def __init__(self):
        """
        Khởi tạo thông số kỹ thuật của máy xạ trị với các giá trị mặc định.
        """
        # Giới hạn góc (độ)
        self.gantry_angle_limits = (0, 360)  # Min, Max
        self.collimator_angle_limits = (0, 360)  # Min, Max
        self.couch_angle_limits = (0, 360)  # Min, Max
        
        # Giới hạn kích thước trường (cm)
        self.field_size_limits = (0.5, 40.0)  # Min, Max
        
        # Tốc độ di chuyển (độ/giây hoặc cm/giây)
        self.gantry_rotation_speed = 6.0  # độ/giây
        self.collimator_rotation_speed = 6.0  # độ/giây
        self.couch_rotation_speed = 3.0  # độ/giây
        self.mlc_leaf_speed = 3.0  # cm/giây
        self.jaw_speed = 2.0  # cm/giây
        
        # Giới hạn tốc độ liều (MU/phút)
        self.dose_rate_limits = (100, 600)  # Min, Max
        
        # Các thông số khác
        self.max_mu = 9999  # Đơn vị monitor tối đa cho một trường
        self.max_energy = 15  # MV hoặc MeV
        self.available_energies = []  # Danh sách các năng lượng có sẵn
        self.tolerance_tables = {}  # Bảng dung sai cho từng tham số
        
        # Thông tin bổ sung
        self.metadata = {}
    
    def set_gantry_angle_limits(self, min_angle: float, max_angle: float):
        """
        Thiết lập giới hạn góc gantry.
        
        Parameters
        ----------
        min_angle : float
            Góc gantry tối thiểu (độ)
        max_angle : float
            Góc gantry tối đa (độ)
        """
        self.gantry_angle_limits = (min_angle, max_angle)
    
    def set_collimator_angle_limits(self, min_angle: float, max_angle: float):
        """
        Thiết lập giới hạn góc collimator.
        
        Parameters
        ----------
        min_angle : float
            Góc collimator tối thiểu (độ)
        max_angle : float
            Góc collimator tối đa (độ)
        """
        self.collimator_angle_limits = (min_angle, max_angle)
    
    def set_couch_angle_limits(self, min_angle: float, max_angle: float):
        """
        Thiết lập giới hạn góc bàn.
        
        Parameters
        ----------
        min_angle : float
            Góc bàn tối thiểu (độ)
        max_angle : float
            Góc bàn tối đa (độ)
        """
        self.couch_angle_limits = (min_angle, max_angle)
    
    def set_field_size_limits(self, min_size: float, max_size: float):
        """
        Thiết lập giới hạn kích thước trường.
        
        Parameters
        ----------
        min_size : float
            Kích thước trường tối thiểu (cm)
        max_size : float
            Kích thước trường tối đa (cm)
        """
        self.field_size_limits = (min_size, max_size)
    
    def set_dose_rate_limits(self, min_rate: float, max_rate: float):
        """
        Thiết lập giới hạn tốc độ liều.
        
        Parameters
        ----------
        min_rate : float
            Tốc độ liều tối thiểu (MU/phút)
        max_rate : float
            Tốc độ liều tối đa (MU/phút)
        """
        self.dose_rate_limits = (min_rate, max_rate)
    
    def add_energy(self, energy: float):
        """
        Thêm năng lượng vào danh sách năng lượng có sẵn.
        
        Parameters
        ----------
        energy : float
            Năng lượng (MV hoặc MeV)
        """
        if energy not in self.available_energies:
            self.available_energies.append(energy)
            self.available_energies.sort()
    
    def add_tolerance_table(self, parameter: str, tolerance_values: Dict[str, float]):
        """
        Thêm bảng dung sai cho một tham số.
        
        Parameters
        ----------
        parameter : str
            Tên tham số
        tolerance_values : Dict[str, float]
            Giá trị dung sai cho tham số
        """
        self.tolerance_tables[parameter] = tolerance_values
    
    def get_tolerance(self, parameter: str, tolerance_type: str = "standard") -> Optional[float]:
        """
        Lấy giá trị dung sai cho một tham số và loại dung sai.
        
        Parameters
        ----------
        parameter : str
            Tên tham số
        tolerance_type : str, optional
            Loại dung sai
            
        Returns
        -------
        Optional[float]
            Giá trị dung sai, None nếu không tìm thấy
        """
        if parameter in self.tolerance_tables:
            return self.tolerance_tables[parameter].get(tolerance_type)
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Chuyển đổi thông số kỹ thuật thành dictionary.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary chứa thông số kỹ thuật
        """
        return {
            "gantry_angle_limits": self.gantry_angle_limits,
            "collimator_angle_limits": self.collimator_angle_limits,
            "couch_angle_limits": self.couch_angle_limits,
            "field_size_limits": self.field_size_limits,
            "gantry_rotation_speed": self.gantry_rotation_speed,
            "collimator_rotation_speed": self.collimator_rotation_speed,
            "couch_rotation_speed": self.couch_rotation_speed,
            "mlc_leaf_speed": self.mlc_leaf_speed,
            "jaw_speed": self.jaw_speed,
            "dose_rate_limits": self.dose_rate_limits,
            "max_mu": self.max_mu,
            "max_energy": self.max_energy,
            "available_energies": self.available_energies,
            "tolerance_tables": self.tolerance_tables,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MachineSpecification':
        """
        Tạo đối tượng MachineSpecification từ dictionary.
        
        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary chứa thông số kỹ thuật
            
        Returns
        -------
        MachineSpecification
            Đối tượng MachineSpecification
        """
        specs = cls()
        
        # Cập nhật các thuộc tính
        specs.gantry_angle_limits = data["gantry_angle_limits"]
        specs.collimator_angle_limits = data["collimator_angle_limits"]
        specs.couch_angle_limits = data["couch_angle_limits"]
        specs.field_size_limits = data["field_size_limits"]
        specs.gantry_rotation_speed = data["gantry_rotation_speed"]
        specs.collimator_rotation_speed = data["collimator_rotation_speed"]
        specs.couch_rotation_speed = data["couch_rotation_speed"]
        specs.mlc_leaf_speed = data["mlc_leaf_speed"]
        specs.jaw_speed = data["jaw_speed"]
        specs.dose_rate_limits = data["dose_rate_limits"]
        specs.max_mu = data["max_mu"]
        specs.max_energy = data["max_energy"]
        specs.available_energies = data["available_energies"]
        specs.tolerance_tables = data["tolerance_tables"]
        specs.metadata = data["metadata"]
        
        return specs