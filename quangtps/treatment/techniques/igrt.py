#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module cho kỹ thuật xạ trị điều hướng hình ảnh (Image-Guided Radiation Therapy - IGRT).

Module này cung cấp các lớp và phương thức để mô phỏng và thực hiện
kỹ thuật xạ trị điều hướng hình ảnh, giúp định vị chính xác mục tiêu
điều trị và theo dõi sự thay đổi trong quá trình điều trị.
"""

import logging
from enum import Enum
from typing import Dict, Optional, Any
import uuid

logger = logging.getLogger(__name__)

class IGRTImageType(str, Enum):
    """Enum đại diện cho các loại hình ảnh IGRT."""
    CBCT = "CBCT"          # Cone Beam CT
    PORTAL = "PORTAL"      # Portal Imaging
    ULTRASOUND = "ULTRASOUND"  # Ultrasound Imaging
    MRI = "MRI"            # MRI-Guided
    SURFACE = "SURFACE"    # Surface Guided
    MARKER = "MARKER"      # Fiducial Marker Tracking

class IGRT:
    """
    Lớp đại diện cho kỹ thuật xạ trị điều hướng hình ảnh (IGRT).
    
    Lớp này cung cấp các phương thức để thiết lập, mô phỏng và đánh giá
    các kỹ thuật IGRT trong quá trình xạ trị.
    """
    
    def __init__(self, 
                 igrt_id: Optional[str] = None,
                 name: str = "Default IGRT",
                 image_type: IGRTImageType = IGRTImageType.CBCT):
        """
        Khởi tạo một đối tượng IGRT.
        
        Parameters
        ----------
        igrt_id : str, optional
            ID duy nhất cho kỹ thuật IGRT
        name : str, optional
            Tên mô tả cho kỹ thuật IGRT
        image_type : IGRTImageType, optional
            Loại hình ảnh sử dụng cho IGRT
        """
        self.igrt_id = igrt_id if igrt_id else str(uuid.uuid4())
        self.name = name
        self.image_type = image_type
        self.frequency = "Daily"  # Tần suất thực hiện IGRT: Daily, Weekly, etc.
        self.registration_method = "Automatic"  # Phương pháp đăng ký hình ảnh
        self.setup_tolerances = (3.0, 3.0, 3.0)  # Dung sai thiết lập (mm) cho x, y, z
        self.correction_strategy = "Online"  # Chiến lược hiệu chỉnh: Online, Offline
        self.metadata = {}
    
    def set_frequency(self, frequency: str):
        """
        Thiết lập tần suất thực hiện IGRT.
        
        Parameters
        ----------
        frequency : str
            Tần suất thực hiện (ví dụ: "Daily", "Weekly", "First 3 fractions", etc.)
        """
        self.frequency = frequency
    
    def set_registration_method(self, method: str):
        """
        Thiết lập phương pháp đăng ký hình ảnh.
        
        Parameters
        ----------
        method : str
            Phương pháp đăng ký hình ảnh (ví dụ: "Automatic", "Manual", "Hybrid")
        """
        self.registration_method = method
    
    def set_setup_tolerances(self, x_tolerance: float, y_tolerance: float, z_tolerance: float):
        """
        Thiết lập dung sai thiết lập.
        
        Parameters
        ----------
        x_tolerance : float
            Dung sai theo trục x (mm)
        y_tolerance : float
            Dung sai theo trục y (mm)
        z_tolerance : float
            Dung sai theo trục z (mm)
        """
        self.setup_tolerances = (x_tolerance, y_tolerance, z_tolerance)
    
    def set_correction_strategy(self, strategy: str):
        """
        Thiết lập chiến lược hiệu chỉnh.
        
        Parameters
        ----------
        strategy : str
            Chiến lược hiệu chỉnh (ví dụ: "Online", "Offline", "Adaptive")
        """
        self.correction_strategy = strategy
    
    def add_metadata(self, key: str, value: Any):
        """
        Thêm metadata cho kỹ thuật IGRT.
        
        Parameters
        ----------
        key : str
            Khóa metadata
        value : Any
            Giá trị metadata
        """
        self.metadata[key] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Chuyển đổi đối tượng IGRT thành dictionary.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary chứa thông tin của đối tượng IGRT
        """
        return {
            "igrt_id": self.igrt_id,
            "name": self.name,
            "image_type": self.image_type.value,
            "frequency": self.frequency,
            "registration_method": self.registration_method,
            "setup_tolerances": self.setup_tolerances,
            "correction_strategy": self.correction_strategy,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'IGRT':
        """
        Tạo đối tượng IGRT từ dictionary.
        
        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary chứa thông tin của đối tượng IGRT
            
        Returns
        -------
        IGRT
            Đối tượng IGRT mới
        """
        igrt = cls(
            igrt_id=data["igrt_id"],
            name=data["name"],
            image_type=IGRTImageType(data["image_type"])
        )
        
        igrt.frequency = data["frequency"]
        igrt.registration_method = data["registration_method"]
        igrt.setup_tolerances = data["setup_tolerances"]
        igrt.correction_strategy = data["correction_strategy"]
        igrt.metadata = data["metadata"]
        
        return igrt