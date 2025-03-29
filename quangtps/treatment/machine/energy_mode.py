#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module định nghĩa các mode năng lượng cho máy xạ trị.
"""

from enum import Enum, auto
from typing import Dict, Any, Optional, List, Union

class EnergyType(str, Enum):
    """Enum đại diện cho loại năng lượng."""
    PHOTON = "Photon"
    ELECTRON = "Electron"
    PROTON = "Proton"
    CARBON_ION = "Carbon Ion"
    NEUTRON = "Neutron"
    COBALT_60 = "Cobalt-60"

class DoseRateUnit(str, Enum):
    """Enum đại diện cho đơn vị tốc độ liều."""
    MU_PER_MIN = "MU/min"
    CGY_PER_MIN = "cGy/min"
    GY_PER_MIN = "Gy/min"

class EnergyMode:
    """
    Lớp đại diện cho một chế độ năng lượng của máy xạ trị.
    
    Lớp này chứa thông tin về loại năng lượng, giá trị năng lượng,
    tốc độ liều và các tham số khác liên quan đến một chế độ năng lượng cụ thể.
    """
    
    def __init__(self, energy_type: EnergyType, energy_value: float, 
                 unit: str = "MV", dose_rate: float = None,
                 dose_rate_unit: DoseRateUnit = DoseRateUnit.MU_PER_MIN,
                 ssd: float = 100.0):
        """
        Khởi tạo một chế độ năng lượng.
        
        Parameters
        ----------
        energy_type : EnergyType
            Loại năng lượng (photon, electron, v.v.)
        energy_value : float
            Giá trị năng lượng
        unit : str, optional
            Đơn vị năng lượng, mặc định là "MV"
        dose_rate : float, optional
            Tốc độ liều, mặc định là None
        dose_rate_unit : DoseRateUnit, optional
            Đơn vị tốc độ liều, mặc định là MU_PER_MIN
        ssd : float, optional
            Khoảng cách nguồn-bề mặt chuẩn (cm), mặc định là 100.0
        """
        self.energy_type = energy_type
        self.energy_value = energy_value
        self.unit = unit
        self.dose_rate = dose_rate
        self.dose_rate_unit = dose_rate_unit
        self.ssd = ssd
        self.output_factor = 1.0
        self.tpr2010 = None  # Tỷ lệ mô-phantom ở độ sâu 20cm/10cm
        self.pdd10 = None    # Phần trăm liều sâu ở độ sâu 10cm
        self.quality_index = None  # Chỉ số chất lượng chùm tia
        self.metadata = {}
    
    @property
    def name(self) -> str:
        """Tên hiển thị của chế độ năng lượng."""
        return f"{self.energy_type.value} {self.energy_value} {self.unit}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Chuyển đổi thông tin chế độ năng lượng thành dictionary."""
        return {
            "energy_type": self.energy_type.value,
            "energy_value": self.energy_value,
            "unit": self.unit,
            "dose_rate": self.dose_rate,
            "dose_rate_unit": self.dose_rate_unit.value,
            "ssd": self.ssd,
            "output_factor": self.output_factor,
            "tpr2010": self.tpr2010,
            "pdd10": self.pdd10,
            "quality_index": self.quality_index,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EnergyMode':
        """Tạo một đối tượng EnergyMode từ dictionary."""
        mode = cls(
            energy_type=EnergyType(data["energy_type"]),
            energy_value=data["energy_value"],
            unit=data["unit"],
            dose_rate=data["dose_rate"],
            dose_rate_unit=DoseRateUnit(data["dose_rate_unit"]),
            ssd=data["ssd"]
        )
        
        mode.output_factor = data.get("output_factor", 1.0)
        mode.tpr2010 = data.get("tpr2010")
        mode.pdd10 = data.get("pdd10")
        mode.quality_index = data.get("quality_index")
        mode.metadata = data.get("metadata", {})
        
        return mode
    
    def __str__(self) -> str:
        """Biểu diễn chuỗi của đối tượng EnergyMode."""
        return self.name
    
    def __repr__(self) -> str:
        """Biểu diễn chi tiết của đối tượng EnergyMode."""
        return f"EnergyMode({self.energy_type.value}, {self.energy_value} {self.unit})" 