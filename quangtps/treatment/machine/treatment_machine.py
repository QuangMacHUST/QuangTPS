#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module này định nghĩa các lớp cơ bản cho máy xạ trị.
"""

import os
import uuid
import json
import logging
import datetime
import numpy as np
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple, Union

from quangtps.treatment.machine.machine_specs import MachineSpecification
from quangtps.treatment.machine.machine_type import MachineType
from quangtps.treatment.machine.machine_status import MachineStatus

logger = logging.getLogger(__name__)


class MachineStatus(str, Enum):
    """Enum đại diện cho trạng thái của máy xạ trị."""
    OPERATIONAL = "Operational"
    MAINTENANCE = "Under Maintenance"
    CALIBRATION = "Under Calibration"
    OFFLINE = "Offline"
    QA_TEST = "Quality Assurance Testing"
    COMMISSIONING = "Commissioning"
    DECOMMISSIONED = "Decommissioned"


class MachineType(str, Enum):
    """Enum đại diện cho các loại máy xạ trị."""
    LINAC = "Linear Accelerator"
    PROTON = "Proton Therapy System"
    CARBON_ION = "Carbon Ion Therapy System"
    GAMMA_KNIFE = "Gamma Knife"
    CYBERKNIFE = "CyberKnife"
    TOMOTHERAPY = "TomoTherapy"
    MR_LINAC = "MR-Linac"
    ORTHOVOLTAGE = "Orthovoltage"
    BRACHYTHERAPY = "Brachytherapy"
    DIAGNOSTIC = "Diagnostic"
    SIMULATOR = "Simulator"


class MaintenanceRecord:
    """Lớp đại diện cho một bản ghi bảo trì của máy xạ trị."""
    
    def __init__(self, date: datetime.date, description: str, duration_hours: float, 
                 performed_by: str, parts_replaced: List[str] = None, 
                 status: MachineStatus = MachineStatus.MAINTENANCE):
        """
        Khởi tạo một bản ghi bảo trì.
        
        Parameters
        ----------
        date : datetime.date
            Ngày bảo trì
        description : str
            Mô tả về hoạt động bảo trì
        duration_hours : float
            Thời gian bảo trì (giờ)
        performed_by : str
            Người thực hiện bảo trì
        parts_replaced : List[str], optional
            Danh sách các bộ phận được thay thế, mặc định là None
        status : MachineStatus, optional
            Trạng thái máy trong quá trình bảo trì, mặc định là MAINTENANCE
        """
        self.date = date
        self.description = description
        self.duration_hours = duration_hours
        self.performed_by = performed_by
        self.parts_replaced = parts_replaced or []
        self.status = status
        self.notes = ""
        
    def to_dict(self) -> Dict[str, Any]:
        """Chuyển đổi bản ghi bảo trì thành dictionary."""
        return {
            "date": self.date.isoformat(),
            "description": self.description,
            "duration_hours": self.duration_hours,
            "performed_by": self.performed_by,
            "parts_replaced": self.parts_replaced,
            "status": self.status.value,
            "notes": self.notes
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MaintenanceRecord':
        """Tạo một bản ghi bảo trì từ dictionary."""
        record = cls(
            date=datetime.date.fromisoformat(data["date"]),
            description=data["description"],
            duration_hours=data["duration_hours"],
            performed_by=data["performed_by"],
            parts_replaced=data["parts_replaced"],
            status=MachineStatus(data["status"])
        )
        record.notes = data.get("notes", "")
        return record


class TreatmentMachine:
    """Lớp cơ bản đại diện cho một máy xạ trị."""
    
    def __init__(self, name: str, machine_id: str, machine_type: MachineType,
                 manufacturer: str = None, model: str = None, 
                 installation_date: datetime.date = None,
                 status: MachineStatus = MachineStatus.OPERATIONAL):
        """
        Khởi tạo một máy xạ trị.
        
        Parameters
        ----------
        name : str
            Tên của máy
        machine_id : str
            ID duy nhất của máy
        machine_type : MachineType
            Loại máy xạ trị
        manufacturer : str, optional
            Nhà sản xuất, mặc định là None
        model : str, optional
            Model, mặc định là None
        installation_date : datetime.date, optional
            Ngày lắp đặt, mặc định là None
        status : MachineStatus, optional
            Trạng thái hiện tại, mặc định là OPERATIONAL
        """
        self.name = name
        self.machine_id = machine_id
        self.machine_type = machine_type
        self.manufacturer = manufacturer
        self.model = model
        self.installation_date = installation_date
        self.status = status
        self.maintenance_records = []  # Danh sách các bản ghi bảo trì
        self.usage_records = []  # Danh sách các bản ghi sử dụng
        self.accessories = []  # Danh sách các phụ kiện
        self.calibration_data = {}  # Dữ liệu hiệu chuẩn
        self.notes = ""
        self.metadata = {}
        
        # Thông tin hình học 3D cho kiểm tra va chạm
        self.geometry = self._initialize_geometry()
    
    def update_status(self, status: MachineStatus, reason: str = ""):
        """
        Cập nhật trạng thái của máy.
        
        Parameters
        ----------
        status : MachineStatus
            Trạng thái mới
        reason : str, optional
            Lý do thay đổi trạng thái, mặc định là ""
        """
        old_status = self.status
        self.status = status
        logger.info(f"Trạng thái máy {self.name} đã được cập nhật từ {old_status.value} thành {status.value}. Lý do: {reason}")
    
    def add_maintenance_record(self, record: MaintenanceRecord):
        """
        Thêm một bản ghi bảo trì mới.
        
        Parameters
        ----------
        record : MaintenanceRecord
            Bản ghi bảo trì cần thêm
        """
        self.maintenance_records.append(record)
        self.maintenance_records.sort(key=lambda x: x.date, reverse=True)
        logger.info(f"Đã thêm bản ghi bảo trì mới cho máy {self.name} vào ngày {record.date.isoformat()}")
    
    def get_maintenance_history(self, start_date: datetime.date = None, 
                               end_date: datetime.date = None) -> List[MaintenanceRecord]:
        """
        Lấy lịch sử bảo trì của máy trong khoảng thời gian cụ thể.
        
        Parameters
        ----------
        start_date : datetime.date, optional
            Ngày bắt đầu, mặc định là None (không giới hạn)
        end_date : datetime.date, optional
            Ngày kết thúc, mặc định là None (không giới hạn)
            
        Returns
        -------
        List[MaintenanceRecord]
            Danh sách các bản ghi bảo trì
        """
        if start_date is None and end_date is None:
            return self.maintenance_records
        
        filtered_records = []
        for record in self.maintenance_records:
            if start_date and record.date < start_date:
                continue
            if end_date and record.date > end_date:
                continue
            filtered_records.append(record)
        
        return filtered_records
    
    def to_dict(self) -> Dict[str, Any]:
        """Chuyển đổi thông tin máy thành dictionary."""
        return {
            "name": self.name,
            "machine_id": self.machine_id,
            "machine_type": self.machine_type.value,
            "manufacturer": self.manufacturer,
            "model": self.model,
            "installation_date": self.installation_date.isoformat() if self.installation_date else None,
            "status": self.status.value,
            "maintenance_records": [record.to_dict() for record in self.maintenance_records],
            "notes": self.notes,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TreatmentMachine':
        """Tạo một đối tượng máy xạ trị từ dictionary."""
        machine = cls(
            name=data["name"],
            machine_id=data["machine_id"],
            machine_type=MachineType(data["machine_type"]),
            manufacturer=data["manufacturer"],
            model=data["model"],
            installation_date=datetime.date.fromisoformat(data["installation_date"]) if data.get("installation_date") else None,
            status=MachineStatus(data["status"])
        )
        
        # Thêm các bản ghi bảo trì
        for record_data in data.get("maintenance_records", []):
            record = MaintenanceRecord.from_dict(record_data)
            machine.add_maintenance_record(record)
        
        machine.notes = data.get("notes", "")
        machine.metadata = data.get("metadata", {})
        
        return machine

    def _initialize_geometry(self) -> Dict[str, Any]:
        """
        Khởi tạo thông tin hình học 3D mặc định cho máy xạ trị.
        
        Returns
        -------
        Dict[str, Any]
            Thông tin hình học 3D của máy
        """
        return {
            "gantry": {
                "center": np.array([0, 0, 0]),  # Tâm quay của gantry
                "radius": 600,  # mm - Bán kính gantry
                "head_dimensions": np.array([300, 200, 500]),  # mm - Kích thước đầu gantry (x, y, z)
                "rotation_axis": np.array([0, 1, 0])  # Trục quay của gantry (theo trục y)
            },
            "collimator": {
                "dimensions": np.array([200, 200, 150]),  # mm - Kích thước collimator (x, y, z)
                "rotation_axis": np.array([0, 0, 1]),  # Trục quay của collimator (theo trục z)
                "distance_to_isocenter": 500  # mm - Khoảng cách từ tâm collimator đến isocenter
            },
            "couch": {
                "top_dimensions": np.array([2200, 70, 550]),  # mm - Kích thước mặt bàn (x, y, z)
                "base_dimensions": np.array([600, 800, 300]),  # mm - Kích thước đế bàn (x, y, z)
                "vertical_range": [-400, 400],  # mm - Phạm vi chuyển động theo chiều dọc
                "lateral_range": [-200, 200],  # mm - Phạm vi chuyển động theo chiều ngang
                "longitudinal_range": [-1000, 1000],  # mm - Phạm vi chuyển động theo chiều dọc
                "rotation_center": np.array([0, -250, 0]),  # Tâm quay của bàn (mm)
                "rotation_axis": np.array([0, 0, 1])  # Trục quay của bàn (theo trục z)
            },
            "detector_panel": {
                "dimensions": np.array([400, 30, 400]),  # mm - Kích thước panel (x, y, z)
                "distance_to_isocenter": 500  # mm - Khoảng cách từ tâm panel đến isocenter
            }
        }
    
    def get_geometry(self) -> Dict[str, Any]:
        """
        Lấy thông tin hình học 3D của máy xạ trị.
        
        Returns
        -------
        Dict[str, Any]
            Thông tin hình học 3D của máy
        """
        return self.geometry
    
    def update_geometry(self, component: str, values: Dict[str, Any]) -> None:
        """
        Cập nhật thông tin hình học 3D của một thành phần máy xạ trị.
        
        Parameters
        ----------
        component : str
            Tên thành phần cần cập nhật (gantry, collimator, couch, detector_panel)
        values : Dict[str, Any]
            Các giá trị cần cập nhật
        """
        if component in self.geometry:
            self.geometry[component].update(values)
            logger.info(f"Đã cập nhật thông tin hình học 3D cho thành phần '{component}'")
        else:
            logger.error(f"Không tìm thấy thành phần '{component}' trong thông tin hình học 3D")
