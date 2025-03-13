#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module cho kỹ thuật xạ trị FLASH (FLASH Radiotherapy).

Module này cung cấp các lớp và phương thức để mô phỏng và thực hiện
kỹ thuật xạ trị FLASH, một phương pháp điều trị mới sử dụng tốc độ 
liều cực cao để giảm thiểu tác dụng phụ trên mô lành.
"""

import logging
from enum import Enum
from typing import Dict, Optional, Any
import uuid

logger = logging.getLogger(__name__)

class FLASHMode(str, Enum):
    """Enum đại diện cho các chế độ điều trị FLASH."""
    ELECTRON = "ELECTRON"  # Điều trị FLASH với electron
    PHOTON = "PHOTON"      # Điều trị FLASH với photon
    PROTON = "PROTON"      # Điều trị FLASH với proton

class FLASHRadiotherapy:
    """
    Lớp đại diện cho kỹ thuật xạ trị FLASH.
    
    Lớp này cung cấp các phương thức để thiết lập và mô phỏng
    kỹ thuật xạ trị FLASH, đặc trưng bởi tốc độ liều cực cao.
    """
    
    def __init__(self, 
                 flash_id: Optional[str] = None,
                 name: str = "Default FLASH",
                 mode: FLASHMode = FLASHMode.ELECTRON):
        """
        Khởi tạo một đối tượng FLASH Radiotherapy.
        
        Parameters
        ----------
        flash_id : str, optional
            ID duy nhất cho kỹ thuật FLASH
        name : str, optional
            Tên mô tả cho kỹ thuật FLASH
        mode : FLASHMode, optional
            Chế độ điều trị FLASH
        """
        self.flash_id = flash_id if flash_id else str(uuid.uuid4())
        self.name = name
        self.mode = mode
        self.dose_rate = 40.0  # Gy/s, tối thiểu > 40 Gy/s để đạt hiệu ứng FLASH
        self.pulse_duration = 0.1  # s
        self.time_between_pulses = 0.001  # s
        self.total_dose = 0.0  # Tổng liều (Gy)
        self.metadata = {}
    
    def set_dose_rate(self, dose_rate: float):
        """
        Thiết lập tốc độ liều cho FLASH.
        
        Parameters
        ----------
        dose_rate : float
            Tốc độ liều (Gy/s)
        """
        self.dose_rate = dose_rate
    
    def set_pulse_parameters(self, duration: float, time_between: float):
        """
        Thiết lập thông số xung.
        
        Parameters
        ----------
        duration : float
            Thời gian xung (s)
        time_between : float
            Thời gian giữa các xung (s)
        """
        self.pulse_duration = duration
        self.time_between_pulses = time_between
    
    def set_total_dose(self, dose: float):
        """
        Thiết lập tổng liều.
        
        Parameters
        ----------
        dose : float
            Tổng liều (Gy)
        """
        self.total_dose = dose
    
    def add_metadata(self, key: str, value: Any):
        """
        Thêm metadata cho kỹ thuật FLASH.
        
        Parameters
        ----------
        key : str
            Khóa metadata
        value : Any
            Giá trị metadata
        """
        self.metadata[key] = value
    
    def calculate_delivery_time(self) -> float:
        """
        Tính toán thởi gian điều trị.
        
        Returns
        -------
        float
            Thời gian điều trị ước tính (s)
        """
        # Số lượng xung cần thiết
        num_pulses = self.total_dose / (self.dose_rate * self.pulse_duration)
        
        # Thời gian điều trị = (thời gian xung + thời gian giữa các xung) * số xung
        delivery_time = (self.pulse_duration + self.time_between_pulses) * num_pulses
        
        return delivery_time
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Chuyển đổi đối tượng FLASH Radiotherapy thành dictionary.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary chứa thông tin của đối tượng FLASH
        """
        return {
            "flash_id": self.flash_id,
            "name": self.name,
            "mode": self.mode.value,
            "dose_rate": self.dose_rate,
            "pulse_duration": self.pulse_duration,
            "time_between_pulses": self.time_between_pulses,
            "total_dose": self.total_dose,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FLASHRadiotherapy':
        """
        Tạo đối tượng FLASH Radiotherapy từ dictionary.
        
        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary chứa thông tin của đối tượng FLASH
            
        Returns
        -------
        FLASHRadiotherapy
            Đối tượng FLASH Radiotherapy mới
        """
        flash = cls(
            flash_id=data["flash_id"],
            name=data["name"],
            mode=FLASHMode(data["mode"])
        )
        
        flash.dose_rate = data["dose_rate"]
        flash.pulse_duration = data["pulse_duration"]
        flash.time_between_pulses = data["time_between_pulses"]
        flash.total_dose = data["total_dose"]
        flash.metadata = data["metadata"]
        
        return flash