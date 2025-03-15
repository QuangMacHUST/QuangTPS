#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module quản lý máy xạ trị.

Module này cung cấp các lớp và phương thức để quản lý các máy xạ trị,
bao gồm thông tin cơ bản, trạng thái, lịch bảo trì, và lịch sử sử dụng.
"""

import logging
import datetime
import uuid
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple, Union

from quangtps.treatment.machine.machine_specs import MachineSpecification

logger = logging.getLogger(__name__)


class MachineType(str, Enum):
    """Enum đại diện cho loại máy xạ trị."""
    LINAC = "Linear Accelerator"
    TOMO = "Tomotherapy"
    CYBER_KNIFE = "CyberKnife"
    GAMMA_KNIFE = "Gamma Knife"
    PROTON = "Proton Therapy"
    ORTHOVOLTAGE = "Orthovoltage"
    BRACHYTHERAPY = "Brachytherapy"
    INTRAOPERATIVE = "Intraoperative"
    OTHER = "Other"


class MachineStatus(str, Enum):
    """Enum đại diện cho trạng thái của máy xạ trị."""
    OPERATIONAL = "Operational"
    MAINTENANCE = "Maintenance"
    CALIBRATION = "Calibration"
    FAULT = "Fault"
    OFFLINE = "Offline"
    STANDBY = "Standby"


class MaintenanceType(str, Enum):
    """Enum đại diện cho loại bảo trì."""
    ROUTINE = "Routine Maintenance"
    PREVENTIVE = "Preventive Maintenance"
    CORRECTIVE = "Corrective Maintenance"
    CALIBRATION = "Calibration"
    QA = "Quality Assurance"
    INSPECTION = "Inspection"
    UPGRADE = "Upgrade"
    OTHER = "Other"


class MaintenanceRecord:
    """
    Lớp đại diện cho một bản ghi bảo trì.
    
    Lớp này chứa thông tin về một lần bảo trì, bao gồm thời gian, loại bảo trì,
    mô tả, và kết quả.
    """
    
    def __init__(
        self,
        maintenance_type: MaintenanceType,
        start_time: datetime.datetime,
        end_time: Optional[datetime.datetime] = None,
        description: str = "",
        technician: str = ""
    ):
        """
        Khởi tạo một bản ghi bảo trì.
        
        Parameters
        ----------
        maintenance_type : MaintenanceType
            Loại bảo trì
        start_time : datetime.datetime
            Thời gian bắt đầu
        end_time : datetime.datetime, optional
            Thời gian kết thúc
        description : str, optional
            Mô tả
        technician : str, optional
            Kỹ thuật viên thực hiện
        """
        self.record_id = str(uuid.uuid4())
        self.maintenance_type = maintenance_type
        self.start_time = start_time
        self.end_time = end_time
        self.description = description
        self.technician = technician
        self.results = ""
        self.parts_replaced = []
        self.tests_performed = []
        self.attachments = []
        self.metadata = {}
    
    def complete(self, end_time: datetime.datetime, results: str):
        """
        Hoàn thành bảo trì.
        
        Parameters
        ----------
        end_time : datetime.datetime
            Thời gian kết thúc
        results : str
            Kết quả bảo trì
        """
        self.end_time = end_time
        self.results = results
    
    def add_part_replaced(self, part_name: str, part_number: str, serial_number: str = ""):
        """
        Thêm thông tin về phụ tùng đã thay thế.
        
        Parameters
        ----------
        part_name : str
            Tên phụ tùng
        part_number : str
            Mã phụ tùng
        serial_number : str, optional
            Số sê-ri
        """
        self.parts_replaced.append({
            "part_name": part_name,
            "part_number": part_number,
            "serial_number": serial_number,
            "replacement_time": datetime.datetime.now().isoformat()
        })
    
    def add_test_performed(self, test_name: str, test_result: str, test_value: Any = None):
        """
        Thêm thông tin về kiểm tra đã thực hiện.
        
        Parameters
        ----------
        test_name : str
            Tên kiểm tra
        test_result : str
            Kết quả kiểm tra
        test_value : Any, optional
            Giá trị kiểm tra
        """
        self.tests_performed.append({
            "test_name": test_name,
            "test_result": test_result,
            "test_value": test_value,
            "test_time": datetime.datetime.now().isoformat()
        })
    
    def add_attachment(self, file_name: str, file_path: str, description: str = ""):
        """
        Thêm tập tin đính kèm.
        
        Parameters
        ----------
        file_name : str
            Tên tập tin
        file_path : str
            Đường dẫn tập tin
        description : str, optional
            Mô tả
        """
        self.attachments.append({
            "file_name": file_name,
            "file_path": file_path,
            "description": description,
            "upload_time": datetime.datetime.now().isoformat()
        })
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Chuyển đổi thông tin bản ghi bảo trì thành dictionary.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary chứa thông tin bản ghi bảo trì
        """
        return {
            "record_id": self.record_id,
            "maintenance_type": self.maintenance_type.value,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "description": self.description,
            "technician": self.technician,
            "results": self.results,
            "parts_replaced": self.parts_replaced,
            "tests_performed": self.tests_performed,
            "attachments": self.attachments,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MaintenanceRecord':
        """
        Tạo đối tượng MaintenanceRecord từ dictionary.
        
        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary chứa thông tin bản ghi bảo trì
            
        Returns
        -------
        MaintenanceRecord
            Đối tượng MaintenanceRecord
        """
        record = cls(
            maintenance_type=MaintenanceType(data["maintenance_type"]),
            start_time=datetime.datetime.fromisoformat(data["start_time"]),
            end_time=datetime.datetime.fromisoformat(data["end_time"]) if data["end_time"] else None,
            description=data["description"],
            technician=data["technician"]
        )
        
        record.record_id = data["record_id"]
        record.results = data["results"]
        record.parts_replaced = data["parts_replaced"]
        record.tests_performed = data["tests_performed"]
        record.attachments = data["attachments"]
        record.metadata = data["metadata"]
        
        return record


class UsageRecord:
    """
    Lớp đại diện cho một bản ghi sử dụng.
    
    Lớp này chứa thông tin về một lần sử dụng máy xạ trị, bao gồm thời gian,
    bệnh nhân, kế hoạch điều trị, và các thông số kỹ thuật.
    """
    
    def __init__(
        self,
        start_time: datetime.datetime,
        end_time: Optional[datetime.datetime] = None,
        patient_id: str = "",
        treatment_id: str = "",
        operator: str = ""
    ):
        """
        Khởi tạo một bản ghi sử dụng.
        
        Parameters
        ----------
        start_time : datetime.datetime
            Thời gian bắt đầu
        end_time : datetime.datetime, optional
            Thời gian kết thúc
        patient_id : str, optional
            ID của bệnh nhân
        treatment_id : str, optional
            ID của điều trị
        operator : str, optional
            Người vận hành
        """
        self.record_id = str(uuid.uuid4())
        self.start_time = start_time
        self.end_time = end_time
        self.patient_id = patient_id
        self.treatment_id = treatment_id
        self.operator = operator
        self.beam_on_time = 0.0  # Thời gian bật chùm tia (giây)
        self.delivered_mu = 0.0  # Số MU đã phát
        self.energy_used = []    # Năng lượng đã sử dụng
        self.gantry_angles = []  # Các góc gantry đã sử dụng
        self.field_sizes = []    # Kích thước trường đã sử dụng
        self.notes = ""
        self.metadata = {}
    
    def complete(self, end_time: datetime.datetime, beam_on_time: float, delivered_mu: float):
        """
        Hoàn thành bản ghi sử dụng.
        
        Parameters
        ----------
        end_time : datetime.datetime
            Thời gian kết thúc
        beam_on_time : float
            Thời gian bật chùm tia (giây)
        delivered_mu : float
            Số MU đã phát
        """
        self.end_time = end_time
        self.beam_on_time = beam_on_time
        self.delivered_mu = delivered_mu
    
    def add_energy(self, energy: float):
        """
        Thêm năng lượng đã sử dụng.
        
        Parameters
        ----------
        energy : float
            Năng lượng (MV hoặc MeV)
        """
        if energy not in self.energy_used:
            self.energy_used.append(energy)
    
    def add_gantry_angle(self, angle: float):
        """
        Thêm góc gantry đã sử dụng.
        
        Parameters
        ----------
        angle : float
            Góc gantry (độ)
        """
        if angle not in self.gantry_angles:
            self.gantry_angles.append(angle)
    
    def add_field_size(self, size: Tuple[float, float]):
        """
        Thêm kích thước trường đã sử dụng.
        
        Parameters
        ----------
        size : Tuple[float, float]
            Kích thước trường (cm x cm)
        """
        if size not in self.field_sizes:
            self.field_sizes.append(size)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Chuyển đổi thông tin bản ghi sử dụng thành dictionary.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary chứa thông tin bản ghi sử dụng
        """
        return {
            "record_id": self.record_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "patient_id": self.patient_id,
            "treatment_id": self.treatment_id,
            "operator": self.operator,
            "beam_on_time": self.beam_on_time,
            "delivered_mu": self.delivered_mu,
            "energy_used": self.energy_used,
            "gantry_angles": self.gantry_angles,
            "field_sizes": self.field_sizes,
            "notes": self.notes,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UsageRecord':
        """
        Tạo đối tượng UsageRecord từ dictionary.
        
        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary chứa thông tin bản ghi sử dụng
            
        Returns
        -------
        UsageRecord
            Đối tượng UsageRecord
        """
        record = cls(
            start_time=datetime.datetime.fromisoformat(data["start_time"]),
            end_time=datetime.datetime.fromisoformat(data["end_time"]) if data["end_time"] else None,
            patient_id=data["patient_id"],
            treatment_id=data["treatment_id"],
            operator=data["operator"]
        )
        
        record.record_id = data["record_id"]
        record.beam_on_time = data["beam_on_time"]
        record.delivered_mu = data["delivered_mu"]
        record.energy_used = data["energy_used"]
        record.gantry_angles = data["gantry_angles"]
        record.field_sizes = data["field_sizes"]
        record.notes = data["notes"]
        record.metadata = data["metadata"]
        
        return record


class TreatmentMachine:
    """
    Lớp đại diện cho một máy xạ trị.
    
    Lớp này chứa thông tin về một máy xạ trị, bao gồm thông tin cơ bản,
    thông số kỹ thuật, trạng thái, lịch bảo trì, và lịch sử sử dụng.
    """
    
    def __init__(
        self,
        machine_id: str,
        name: str,
        machine_type: MachineType,
        manufacturer: str,
        model: str,
        location: str,
        specifications: Optional[MachineSpecification] = None
    ):
        """
        Khởi tạo một máy xạ trị.
        
        Parameters
        ----------
        machine_id : str
            ID của máy
        name : str
            Tên máy
        machine_type : MachineType
            Loại máy
        manufacturer : str
            Nhà sản xuất
        model : str
            Model
        location : str
            Vị trí
        specifications : MachineSpecification, optional
            Thông số kỹ thuật
        """
        self.machine_id = machine_id
        self.name = name
        self.machine_type = machine_type
        self.manufacturer = manufacturer
        self.model = model
        self.location = location
        self.status = MachineStatus.OPERATIONAL
        self.specifications = specifications or MachineSpecification()
        
        # Thông tin bổ sung
        self.serial_number = ""
        self.installation_date = None
        self.last_calibration_date = None
        self.next_calibration_date = None
        self.next_maintenance_date = None
        self.operating_hours = 0.0
        self.lifetime_dose = 0.0  # Tổng liều suốt đời (MU)
        
        # Lịch sử bảo trì và sử dụng
        self.maintenance_records: List[MaintenanceRecord] = []
        self.usage_records: List[UsageRecord] = []
        
        # Thông tin khác
        self.description = ""
        self.notes = ""
        self.metadata = {}
    
    def update_status(self, status: MachineStatus, reason: str = ""):
        """
        Cập nhật trạng thái của máy.
        
        Parameters
        ----------
        status : MachineStatus
            Trạng thái mới
        reason : str, optional
            Lý do thay đổi trạng thái
        """
        self.status = status
        self.notes += f"{datetime.datetime.now().isoformat()}: Status changed to {status.value}. Reason: {reason}\n"
    
    def add_maintenance_record(self, record: MaintenanceRecord):
        """
        Thêm bản ghi bảo trì.
        
        Parameters
        ----------
        record : MaintenanceRecord
            Bản ghi bảo trì
        """
        self.maintenance_records.append(record)
        self.maintenance_records.sort(key=lambda r: r.start_time, reverse=True)
        
        # Cập nhật ngày bảo trì cuối cùng và ngày bảo trì tiếp theo
        if record.maintenance_type in [MaintenanceType.CALIBRATION, MaintenanceType.QA]:
            self.last_calibration_date = record.start_time.date()
            self.next_calibration_date = self.last_calibration_date + datetime.timedelta(days=90)  # Mặc định là 3 tháng
    
    def add_usage_record(self, record: UsageRecord):
        """
        Thêm bản ghi sử dụng.
        
        Parameters
        ----------
        record : UsageRecord
            Bản ghi sử dụng
        """
        self.usage_records.append(record)
        self.usage_records.sort(key=lambda r: r.start_time, reverse=True)
        
        # Cập nhật số giờ vận hành và tổng liều
        if record.end_time:
            duration = (record.end_time - record.start_time).total_seconds() / 3600.0  # Giờ
            self.operating_hours += duration
            self.lifetime_dose += record.delivered_mu
    
    def get_maintenance_records(
        self,
        start_date: Optional[datetime.date] = None,
        end_date: Optional[datetime.date] = None,
        maintenance_type: Optional[MaintenanceType] = None
    ) -> List[MaintenanceRecord]:
        """
        Lấy danh sách bản ghi bảo trì.
        
        Parameters
        ----------
        start_date : datetime.date, optional
            Ngày bắt đầu
        end_date : datetime.date, optional
            Ngày kết thúc
        maintenance_type : MaintenanceType, optional
            Loại bảo trì
            
        Returns
        -------
        List[MaintenanceRecord]
            Danh sách bản ghi bảo trì
        """
        records = self.maintenance_records
        
        if start_date:
            records = [r for r in records if r.start_time.date() >= start_date]
            
        if end_date:
            records = [r for r in records if r.start_time.date() <= end_date]
            
        if maintenance_type:
            records = [r for r in records if r.maintenance_type == maintenance_type]
            
        return records
    
    def get_usage_records(
        self,
        start_date: Optional[datetime.date] = None,
        end_date: Optional[datetime.date] = None,
        patient_id: Optional[str] = None
    ) -> List[UsageRecord]:
        """
        Lấy danh sách bản ghi sử dụng.
        
        Parameters
        ----------
        start_date : datetime.date, optional
            Ngày bắt đầu
        end_date : datetime.date, optional
            Ngày kết thúc
        patient_id : str, optional
            ID của bệnh nhân
            
        Returns
        -------
        List[UsageRecord]
            Danh sách bản ghi sử dụng
        """
        records = self.usage_records
        
        if start_date:
            records = [r for r in records if r.start_time.date() >= start_date]
            
        if end_date:
            records = [r for r in records if r.start_time.date() <= end_date]
            
        if patient_id:
            records = [r for r in records if r.patient_id == patient_id]
            
        return records
    
    def calculate_uptime(
        self,
        start_date: Optional[datetime.date] = None,
        end_date: Optional[datetime.date] = None
    ) -> float:
        """
        Tính tỷ lệ hoạt động của máy.
        
        Parameters
        ----------
        start_date : datetime.date, optional
            Ngày bắt đầu
        end_date : datetime.date, optional
            Ngày kết thúc
            
        Returns
        -------
        float
            Tỷ lệ hoạt động (0-1)
        """
        if not start_date:
            start_date = datetime.date.today() - datetime.timedelta(days=30)  # Mặc định là 30 ngày trước
            
        if not end_date:
            end_date = datetime.date.today()
            
        # Tổng số giờ trong khoảng thời gian
        total_hours = (end_date - start_date).days * 24.0
        
        # Tổng số giờ bảo trì
        maintenance_hours = 0.0
        for record in self.get_maintenance_records(start_date, end_date):
            if record.end_time:
                # Tính thời gian bảo trì nằm trong khoảng thời gian
                start = max(record.start_time, datetime.datetime.combine(start_date, datetime.time.min))
                end = min(record.end_time, datetime.datetime.combine(end_date, datetime.time.max))
                
                if start < end:
                    maintenance_hours += (end - start).total_seconds() / 3600.0
        
        # Tỷ lệ hoạt động
        if total_hours > 0:
            return 1.0 - (maintenance_hours / total_hours)
        return 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Chuyển đổi thông tin máy xạ trị thành dictionary.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary chứa thông tin máy xạ trị
        """
        return {
            "machine_id": self.machine_id,
            "name": self.name,
            "machine_type": self.machine_type.value,
            "manufacturer": self.manufacturer,
            "model": self.model,
            "location": self.location,
            "status": self.status.value,
            "specifications": self.specifications.to_dict(),
            "serial_number": self.serial_number,
            "installation_date": self.installation_date.isoformat() if self.installation_date else None,
            "last_calibration_date": self.last_calibration_date.isoformat() if self.last_calibration_date else None,
            "next_calibration_date": self.next_calibration_date.isoformat() if self.next_calibration_date else None,
            "next_maintenance_date": self.next_maintenance_date.isoformat() if self.next_maintenance_date else None,
            "operating_hours": self.operating_hours,
            "lifetime_dose": self.lifetime_dose,
            "maintenance_records": [r.to_dict() for r in self.maintenance_records],
            "usage_records": [r.to_dict() for r in self.usage_records],
            "description": self.description,
            "notes": self.notes,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TreatmentMachine':
        """
        Tạo đối tượng TreatmentMachine từ dictionary.
        
        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary chứa thông tin máy xạ trị
            
        Returns
        -------
        TreatmentMachine
            Đối tượng TreatmentMachine
        """
        from quangtps.treatment.machine.machine_specs import MachineSpecification
        
        specifications = MachineSpecification.from_dict(data["specifications"])
        
        machine = cls(
            machine_id=data["machine_id"],
            name=data["name"],
            machine_type=MachineType(data["machine_type"]),
            manufacturer=data["manufacturer"],
            model=data["model"],
            location=data["location"],
            specifications=specifications
        )
        
        machine.status = MachineStatus(data["status"])
        machine.serial_number = data["serial_number"]
        
        if data["installation_date"]:
            machine.installation_date = datetime.date.fromisoformat(data["installation_date"])
        
        if data["last_calibration_date"]:
            machine.last_calibration_date = datetime.date.fromisoformat(data["last_calibration_date"])
        
        if data["next_calibration_date"]:
            machine.next_calibration_date = datetime.date.fromisoformat(data["next_calibration_date"])
        
        if data["next_maintenance_date"]:
            machine.next_maintenance_date = datetime.date.fromisoformat(data["next_maintenance_date"])
        
        machine.operating_hours = data["operating_hours"]
        machine.lifetime_dose = data["lifetime_dose"]
        
        machine.maintenance_records = [MaintenanceRecord.from_dict(r) for r in data["maintenance_records"]]
        machine.usage_records = [UsageRecord.from_dict(r) for r in data["usage_records"]]
        
        machine.description = data["description"]
        machine.notes = data["notes"]
        machine.metadata = data["metadata"]
        
        return machine
