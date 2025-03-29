#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module định nghĩa kỹ thuật xạ trị ion carbon.
"""

from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import logging

from quangtps.treatment.techniques.treatment_technique import TreatmentTechnique

logger = logging.getLogger(__name__)


class Carbon(TreatmentTechnique):
    """
    Lớp đại diện cho kỹ thuật xạ trị ion carbon.
    
    Kỹ thuật xạ trị ion carbon sử dụng chùm ion carbon nặng để điều trị ung thư,
    mang lại khả năng kiểm soát liều cao và hiệu quả sinh học tốt hơn so với 
    proton hoặc photon.
    """
    
    def __init__(self, technique_name: str = "Carbon Ion"):
        """
        Khởi tạo kỹ thuật xạ trị ion carbon.
        
        Parameters
        ----------
        technique_name : str, optional
            Tên của kỹ thuật, mặc định là "Carbon Ion"
        """
        super().__init__(technique_name)
        self.delivery_method = None  # PBS (Pencil Beam Scanning), US (Uniform Scanning), RS (Raster Scanning)
        self.energy_range = (100, 400)  # MeV/u, mặc định
        self.rbe = 2.5  # Relative Biological Effectiveness, mặc định
        self.spot_size = None  # mm, cho PBS
        self.spot_spacing = None  # mm, cho PBS
        self.layer_spacing = None  # mm, cho PBS
        self.scanning_pattern = None  # Continuous, Discrete
        self.has_range_shifter = False
        self.range_shifter_thickness = None  # mm water equivalent
    
    def set_delivery_method(self, method: str):
        """
        Thiết lập phương pháp phân phối chùm tia ion carbon.
        
        Parameters
        ----------
        method : str
            Phương pháp phân phối: "PBS", "US", "RS"
        """
        valid_methods = ["PBS", "US", "RS"]
        if method not in valid_methods:
            logger.warning(f"Phương pháp phân phối không hợp lệ: {method}. Phải là một trong {valid_methods}")
            return
        
        self.delivery_method = method
        logger.info(f"Đã thiết lập phương pháp phân phối ion carbon: {method}")
    
    def set_energy_range(self, min_energy: float, max_energy: float):
        """
        Thiết lập phạm vi năng lượng ion carbon.
        
        Parameters
        ----------
        min_energy : float
            Năng lượng tối thiểu (MeV/u)
        max_energy : float
            Năng lượng tối đa (MeV/u)
        """
        if min_energy <= 0 or max_energy <= 0 or min_energy >= max_energy:
            logger.warning(f"Phạm vi năng lượng không hợp lệ: {min_energy}-{max_energy} MeV/u")
            return
        
        self.energy_range = (min_energy, max_energy)
        logger.info(f"Đã thiết lập phạm vi năng lượng ion carbon: {min_energy}-{max_energy} MeV/u")
    
    def set_rbe(self, rbe: float):
        """
        Thiết lập RBE (Relative Biological Effectiveness).
        
        Parameters
        ----------
        rbe : float
            Giá trị RBE
        """
        if rbe <= 1.0:
            logger.warning(f"RBE không hợp lệ: {rbe}. RBE của ion carbon phải > 1.0")
            return
        
        self.rbe = rbe
        logger.info(f"Đã thiết lập RBE: {rbe}")
    
    def configure_pbs(self, spot_size: float, spot_spacing: float, layer_spacing: float, scanning_pattern: str = "Discrete"):
        """
        Cấu hình thông số cho phương pháp PBS (Pencil Beam Scanning).
        
        Parameters
        ----------
        spot_size : float
            Kích thước điểm (mm)
        spot_spacing : float
            Khoảng cách giữa các điểm (mm)
        layer_spacing : float
            Khoảng cách giữa các lớp (mm)
        scanning_pattern : str, optional
            Mẫu quét: "Continuous" hoặc "Discrete", mặc định là "Discrete"
        """
        if self.delivery_method != "PBS":
            logger.warning("Không thể cấu hình PBS khi phương pháp phân phối không phải là PBS")
            return
        
        if spot_size <= 0 or spot_spacing <= 0 or layer_spacing <= 0:
            logger.warning("Thông số PBS không hợp lệ")
            return
        
        if scanning_pattern not in ["Continuous", "Discrete"]:
            logger.warning(f"Mẫu quét không hợp lệ: {scanning_pattern}. Phải là 'Continuous' hoặc 'Discrete'")
            return
        
        self.spot_size = spot_size
        self.spot_spacing = spot_spacing
        self.layer_spacing = layer_spacing
        self.scanning_pattern = scanning_pattern
        logger.info(f"Đã cấu hình PBS với kích thước điểm: {spot_size} mm, "
                   f"khoảng cách điểm: {spot_spacing} mm, khoảng cách lớp: {layer_spacing} mm, "
                   f"mẫu quét: {scanning_pattern}")
    
    def add_range_shifter(self, thickness: float):
        """
        Thêm range shifter để điều chỉnh phạm vi chùm tia.
        
        Parameters
        ----------
        thickness : float
            Độ dày của range shifter (mm water equivalent)
        """
        if thickness <= 0:
            logger.warning(f"Độ dày range shifter không hợp lệ: {thickness} mm")
            return
        
        self.has_range_shifter = True
        self.range_shifter_thickness = thickness
        logger.info(f"Đã thêm range shifter với độ dày: {thickness} mm water equivalent")
    
    def remove_range_shifter(self):
        """Loại bỏ range shifter."""
        self.has_range_shifter = False
        self.range_shifter_thickness = None
        logger.info("Đã loại bỏ range shifter")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Chuyển đổi thông tin kỹ thuật xạ trị ion carbon thành dictionary.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary chứa thông tin kỹ thuật
        """
        data = super().to_dict()
        data.update({
            "delivery_method": self.delivery_method,
            "energy_range": self.energy_range,
            "rbe": self.rbe,
            "spot_size": self.spot_size,
            "spot_spacing": self.spot_spacing,
            "layer_spacing": self.layer_spacing,
            "scanning_pattern": self.scanning_pattern,
            "has_range_shifter": self.has_range_shifter,
            "range_shifter_thickness": self.range_shifter_thickness
        })
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Carbon':
        """
        Tạo đối tượng Carbon từ dictionary.
        
        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary chứa thông tin kỹ thuật
            
        Returns
        -------
        Carbon
            Đối tượng Carbon
        """
        technique = cls(data.get("technique_name", "Carbon Ion"))
        
        # Thiết lập các thuộc tính
        if "delivery_method" in data:
            technique.set_delivery_method(data["delivery_method"])
        
        if "energy_range" in data and isinstance(data["energy_range"], tuple) and len(data["energy_range"]) == 2:
            technique.set_energy_range(data["energy_range"][0], data["energy_range"][1])
        
        if "rbe" in data:
            technique.set_rbe(data["rbe"])
        
        if (data.get("delivery_method") == "PBS" and
            "spot_size" in data and "spot_spacing" in data and "layer_spacing" in data):
            technique.configure_pbs(
                data["spot_size"],
                data["spot_spacing"],
                data["layer_spacing"],
                data.get("scanning_pattern", "Discrete")
            )
        
        if data.get("has_range_shifter", False) and "range_shifter_thickness" in data:
            technique.add_range_shifter(data["range_shifter_thickness"])
        
        return technique


# Lớp đồng nghĩa với Carbon để đảm bảo tính tương thích ngược
class CarbonIonTherapy(Carbon):
    """
    Lớp đại diện cho kỹ thuật xạ trị ion carbon (đồng nghĩa với lớp Carbon).
    
    Lớp này được cung cấp để đảm bảo tính tương thích với mã nguồn cũ sử dụng
    tên CarbonIonTherapy thay vì Carbon.
    """
    
    def __init__(self, technique_name: str = "Carbon Ion Therapy"):
        """
        Khởi tạo kỹ thuật xạ trị ion carbon.
        
        Parameters
        ----------
        technique_name : str, optional
            Tên của kỹ thuật, mặc định là "Carbon Ion Therapy"
        """
        super().__init__(technique_name)