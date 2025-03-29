#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module định nghĩa lớp Linac (máy gia tốc tuyến tính).
"""

import os
import json
import logging
import datetime
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from enum import Enum

from quangtps.treatment.machine.accelerator import Accelerator
from quangtps.treatment.machine.machine_specs import MachineSpecification
from quangtps.treatment.mlc.mlc_model import MLCModel
from quangtps.treatment.machine.machine_type import MachineType
from quangtps.treatment.machine.machine_status import MachineStatus
from quangtps.treatment.machine.energy_mode import EnergyMode

logger = logging.getLogger(__name__)

class Linac(Accelerator):
    """
    Lớp đại diện cho một máy gia tốc tuyến tính (Linear Accelerator - Linac).
    
    Linac là loại máy xạ trị phổ biến nhất, sử dụng sóng điện từ tần số cao
    để tạo ra các chùm tia phát tán qua một cấu trúc gia tốc để tạo ra chùm
    tia có năng lượng cao có thể nhắm vào khối u.
    """
    
    def __init__(self, name: str, machine_id: str, 
                 manufacturer: str = None, model: str = None,
                 installation_date: datetime.date = None,
                 status: MachineStatus = MachineStatus.OPERATIONAL,
                 energy_modes: List[EnergyMode] = None,
                 max_dose_rate: float = None):
        """
        Khởi tạo một máy gia tốc tuyến tính.
        
        Parameters
        ----------
        name : str
            Tên của máy
        machine_id : str
            ID duy nhất của máy
        manufacturer : str, optional
            Nhà sản xuất, mặc định là None
        model : str, optional
            Model, mặc định là None
        installation_date : datetime.date, optional
            Ngày lắp đặt, mặc định là None
        status : MachineStatus, optional
            Trạng thái hiện tại, mặc định là OPERATIONAL
        energy_modes : List[EnergyMode], optional
            Các chế độ năng lượng, mặc định là None
        max_dose_rate : float, optional
            Tốc độ liều tối đa (MU/min), mặc định là None
        """
        super().__init__(name, machine_id, MachineType.LINAC,
                        manufacturer, model, installation_date, status)
        
        self.energy_modes = energy_modes or []
        self.max_dose_rate = max_dose_rate
        self.mlc = None
        self.jaw = None
        
        # Cập nhật hình học 3D dựa trên model cụ thể
        self._update_model_specific_geometry()
        
    def _update_model_specific_geometry(self):
        """
        Cập nhật thông tin hình học 3D dựa trên model cụ thể của máy Linac.
        """
        # Cập nhật thông tin dựa trên model
        if self.model and self.manufacturer:
            model_lower = self.model.lower()
            manufacturer_lower = self.manufacturer.lower()
            
            # Varian TrueBeam
            if "truebeam" in model_lower and "varian" in manufacturer_lower:
                self.update_geometry("gantry", {
                    "radius": 650,
                    "head_dimensions": np.array([320, 220, 510])
                })
                self.update_geometry("collimator", {
                    "dimensions": np.array([220, 220, 160]),
                    "distance_to_isocenter": 520
                })
                self.update_geometry("couch", {
                    "top_dimensions": np.array([2300, 70, 570]),
                    "rotation_center": np.array([0, -270, 0])
                })
            
            # Elekta Versa HD
            elif "versa" in model_lower and "elekta" in manufacturer_lower:
                self.update_geometry("gantry", {
                    "radius": 680,
                    "head_dimensions": np.array([340, 230, 530])
                })
                self.update_geometry("collimator", {
                    "dimensions": np.array([240, 240, 170]),
                    "distance_to_isocenter": 550
                })
                self.update_geometry("couch", {
                    "top_dimensions": np.array([2400, 75, 580]),
                    "rotation_center": np.array([0, -280, 0])
                })
            
            # Siemens Artiste
            elif "artiste" in model_lower and "siemens" in manufacturer_lower:
                self.update_geometry("gantry", {
                    "radius": 660,
                    "head_dimensions": np.array([330, 220, 520])
                })
                self.update_geometry("collimator", {
                    "dimensions": np.array([230, 230, 165]),
                    "distance_to_isocenter": 530
                })
                self.update_geometry("couch", {
                    "top_dimensions": np.array([2350, 72, 560]),
                    "rotation_center": np.array([0, -275, 0])
                })
    
    def get_collision_geometry(self) -> Dict[str, Any]:
        """
        Lấy thông tin hình học 3D cho kiểm tra va chạm, cụ thể cho từng model Linac.
        
        Returns
        -------
        Dict[str, Any]
            Thông tin hình học 3D của máy Linac
        """
        # Lấy thông tin hình học cơ bản
        geometry = self.get_geometry()
        
        # Nếu có MLC, cập nhật thông tin collimator
        if self.mlc:
            mlc_geometry = {
                "type": self.mlc.model if hasattr(self.mlc, "model") else "Generic MLC",
                "leaf_width": self.mlc.leaf_width if hasattr(self.mlc, "leaf_width") else 5.0,  # mm
                "number_of_leaves": self.mlc.number_of_leaves if hasattr(self.mlc, "number_of_leaves") else 120,
                "max_field_size": self.mlc.max_field_size if hasattr(self.mlc, "max_field_size") else [400, 400],  # mm
                "collision_boundary": np.array([
                    geometry["collimator"]["dimensions"][0] + 50,  # Thêm biên an toàn
                    geometry["collimator"]["dimensions"][1],
                    geometry["collimator"]["dimensions"][2] + 50
                ])
            }
            
            # Cập nhật thông tin vào geometry
            geometry["mlc"] = mlc_geometry
        
        # Bổ sung thông tin vật liệu và mật độ cho tính toán tương tác vật lý
        geometry["materials"] = {
            "gantry_head": {
                "density": 7.8,  # g/cm³ (thép)
                "material": "Steel"
            },
            "collimator": {
                "density": 19.3,  # g/cm³ (tungsten)
                "material": "Tungsten"
            },
            "couch_top": {
                "density": 1.2,  # g/cm³ (carbon fiber)
                "material": "Carbon Fiber"
            },
            "couch_base": {
                "density": 7.8,  # g/cm³ (thép)
                "material": "Steel"
            }
        }
        
        # Bổ sung thông tin về các phụ kiện
        geometry["accessories"] = {
            "wedges": [],
            "blocks": [],
            "applicators": [],
            "cones": []
        }
        
        # Nếu có wedge, thêm vào danh sách phụ kiện
        if hasattr(self, "wedges") and self.wedges:
            for wedge in self.wedges:
                wedge_info = {
                    "id": wedge.id if hasattr(wedge, "id") else "Unknown",
                    "angle": wedge.angle if hasattr(wedge, "angle") else 45,  # độ
                    "dimensions": np.array([150, 50, 150]) if not hasattr(wedge, "dimensions") else wedge.dimensions,  # mm
                    "distance_to_isocenter": 300  # mm
                }
                geometry["accessories"]["wedges"].append(wedge_info)
        
        return geometry