#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module quản lý đơn liều xạ trị trong QuangTPS.

Module này cung cấp các lớp và phương thức để mô tả và quản lý các đơn liều xạ trị,
bao gồm thông tin về liều kê đơn, phân đoạn và các ràng buộc liều.
"""

import logging
import copy
from typing import Dict, List, Optional, Any, Tuple, Union
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)


class FractionationType(Enum):
    """Enum cho các loại phân đoạn."""
    STANDARD = "Standard"
    HYPOFRACTIONATED = "Hypofractionated"
    HYPERFRACTIONATED = "Hyperfractionated"
    SRS = "Stereotactic Radiosurgery"
    SBRT = "Stereotactic Body Radiation Therapy"
    CUSTOM = "Custom"


class PrescriptionStatus(Enum):
    """Enum cho trạng thái đơn liều."""
    DRAFT = "Draft"
    APPROVED = "Approved"
    COMPLETED = "Completed"
    CANCELLED = "Cancelled"


class Fractionation:
    """
    Lớp quản lý thông tin phân đoạn xạ trị.
    
    Lớp này mô tả cách phân chia liều xạ trị thành các phân đoạn (fractions),
    bao gồm số lượng phân đoạn và liều mỗi phân đoạn.
    """
    
    def __init__(
        self,
        num_fractions: int = 1,
        dose_per_fraction: float = 2.0,
        fractionation_type: FractionationType = FractionationType.STANDARD,
        schedule: Optional[List[datetime]] = None
    ):
        """
        Khởi tạo một đối tượng phân đoạn.
        
        Parameters
        ----------
        num_fractions : int
            Số lượng phân đoạn
        dose_per_fraction : float
            Liều mỗi phân đoạn (Gy)
        fractionation_type : FractionationType
            Loại phân đoạn
        schedule : List[datetime], optional
            Lịch trình cho các phân đoạn
        """
        self.num_fractions = num_fractions
        self.dose_per_fraction = dose_per_fraction
        self.fractionation_type = fractionation_type
        self.schedule = schedule if schedule else []
        self.parameters = {}  # Dictionary lưu trữ các tham số bổ sung
        
    @property
    def total_dose(self) -> float:
        """Tổng liều (Gy) của tất cả các phân đoạn."""
        return self.num_fractions * self.dose_per_fraction
    
    def set_schedule(self, dates: List[datetime]):
        """
        Đặt lịch trình cho các phân đoạn.
        
        Parameters
        ----------
        dates : List[datetime]
            Danh sách ngày điều trị
        """
        if len(dates) != self.num_fractions:
            logger.warning(f"Số lượng ngày ({len(dates)}) khác với số phân đoạn ({self.num_fractions})")
        
        self.schedule = dates
        
    def set_parameter(self, key: str, value: Any):
        """
        Đặt một tham số bổ sung.
        
        Parameters
        ----------
        key : str
            Tên tham số
        value : Any
            Giá trị tham số
        """
        self.parameters[key] = value
        
    def get_parameter(self, key: str, default: Any = None) -> Any:
        """
        Lấy giá trị của một tham số.
        
        Parameters
        ----------
        key : str
            Tên tham số
        default : Any, optional
            Giá trị mặc định nếu tham số không tồn tại
            
        Returns
        -------
        Any
            Giá trị của tham số
        """
        return self.parameters.get(key, default)
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Chuyển đổi đối tượng phân đoạn thành dictionary.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary chứa thông tin phân đoạn
        """
        return {
            'num_fractions': self.num_fractions,
            'dose_per_fraction': self.dose_per_fraction,
            'fractionation_type': self.fractionation_type.value,
            'schedule': [dt.isoformat() for dt in self.schedule] if self.schedule else [],
            'parameters': self.parameters
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Fractionation':
        """
        Tạo đối tượng Fractionation từ dictionary.
        
        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary chứa thông tin phân đoạn
            
        Returns
        -------
        Fractionation
            Đối tượng phân đoạn
        """
        fractionation = cls(
            num_fractions=data.get('num_fractions', 1),
            dose_per_fraction=data.get('dose_per_fraction', 2.0),
            fractionation_type=FractionationType(data.get('fractionation_type', 'STANDARD'))
        )
        
        # Phục hồi lịch trình
        if 'schedule' in data and data['schedule']:
            fractionation.schedule = [datetime.fromisoformat(dt) for dt in data['schedule']]
            
        # Phục hồi các tham số
        if 'parameters' in data:
            fractionation.parameters = data['parameters']
            
        return fractionation
        
    def __str__(self) -> str:
        """Biểu diễn chuỗi của đối tượng phân đoạn."""
        return f"{self.num_fractions} x {self.dose_per_fraction} Gy ({self.fractionation_type.value})"
        
    def copy(self) -> 'Fractionation':
        """
        Tạo một bản sao của đối tượng phân đoạn.
        
        Returns
        -------
        Fractionation
            Bản sao của đối tượng phân đoạn
        """
        return copy.deepcopy(self)


class DosePrescription:
    """
    Lớp đơn liều cho một cấu trúc cụ thể.
    
    Lớp này mô tả liều kê đơn cho một cấu trúc cụ thể, bao gồm liều mục tiêu,
    ràng buộc liều và các thông tin bổ sung.
    """
    
    def __init__(
        self,
        structure_id: str,
        structure_name: str,
        prescribed_dose: float,
        is_target: bool = True,
        priority: int = 1
    ):
        """
        Khởi tạo một đối tượng đơn liều.
        
        Parameters
        ----------
        structure_id : str
            ID của cấu trúc
        structure_name : str
            Tên của cấu trúc
        prescribed_dose : float
            Liều kê đơn (Gy)
        is_target : bool
            True nếu cấu trúc là mục tiêu, False nếu là cơ quan nguy cấp
        priority : int
            Độ ưu tiên (thấp hơn = ưu tiên cao hơn)
        """
        self.structure_id = structure_id
        self.structure_name = structure_name
        self.prescribed_dose = prescribed_dose
        self.is_target = is_target
        self.priority = priority
        
        self.min_dose = None  # Liều tối thiểu (Gy)
        self.max_dose = None  # Liều tối đa (Gy)
        self.coverage = None  # Độ phủ mong muốn (% thể tích)
        self.constraints = []  # Các ràng buộc liều
        self.parameters = {}  # Dictionary lưu trữ các tham số bổ sung
        
    def set_dose_range(self, min_dose: Optional[float] = None, max_dose: Optional[float] = None):
        """
        Đặt dải liều mục tiêu.
        
        Parameters
        ----------
        min_dose : float, optional
            Liều tối thiểu (Gy)
        max_dose : float, optional
            Liều tối đa (Gy)
        """
        self.min_dose = min_dose
        self.max_dose = max_dose
        
    def set_coverage(self, coverage: float):
        """
        Đặt độ phủ mong muốn.
        
        Parameters
        ----------
        coverage : float
            Độ phủ mong muốn (% thể tích)
        """
        self.coverage = coverage
        
    def add_constraint(self, constraint_type: str, dose_value: Optional[float] = None, 
                      volume_value: Optional[float] = None, priority: int = 1):
        """
        Thêm một ràng buộc liều.
        
        Parameters
        ----------
        constraint_type : str
            Loại ràng buộc
        dose_value : float, optional
            Giá trị liều (Gy)
        volume_value : float, optional
            Giá trị thể tích (%)
        priority : int
            Độ ưu tiên (thấp hơn = ưu tiên cao hơn)
        """
        constraint = {
            'type': constraint_type,
            'dose_value': dose_value,
            'volume_value': volume_value,
            'priority': priority
        }
        
        self.constraints.append(constraint)
        
    def set_parameter(self, key: str, value: Any):
        """
        Đặt một tham số bổ sung.
        
        Parameters
        ----------
        key : str
            Tên tham số
        value : Any
            Giá trị tham số
        """
        self.parameters[key] = value
        
    def get_parameter(self, key: str, default: Any = None) -> Any:
        """
        Lấy giá trị của một tham số.
        
        Parameters
        ----------
        key : str
            Tên tham số
        default : Any, optional
            Giá trị mặc định nếu tham số không tồn tại
            
        Returns
        -------
        Any
            Giá trị của tham số
        """
        return self.parameters.get(key, default)
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Chuyển đổi đối tượng đơn liều thành dictionary.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary chứa thông tin đơn liều
        """
        return {
            'structure_id': self.structure_id,
            'structure_name': self.structure_name,
            'prescribed_dose': self.prescribed_dose,
            'is_target': self.is_target,
            'priority': self.priority,
            'min_dose': self.min_dose,
            'max_dose': self.max_dose,
            'coverage': self.coverage,
            'constraints': self.constraints,
            'parameters': self.parameters
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DosePrescription':
        """
        Tạo đối tượng DosePrescription từ dictionary.
        
        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary chứa thông tin đơn liều
            
        Returns
        -------
        DosePrescription
            Đối tượng đơn liều
        """
        prescription = cls(
            structure_id=data.get('structure_id', ''),
            structure_name=data.get('structure_name', ''),
            prescribed_dose=data.get('prescribed_dose', 0.0),
            is_target=data.get('is_target', True),
            priority=data.get('priority', 1)
        )
        
        # Phục hồi các thuộc tính khác
        prescription.min_dose = data.get('min_dose')
        prescription.max_dose = data.get('max_dose')
        prescription.coverage = data.get('coverage')
        
        # Phục hồi các ràng buộc
        if 'constraints' in data:
            prescription.constraints = data['constraints']
            
        # Phục hồi các tham số
        if 'parameters' in data:
            prescription.parameters = data['parameters']
            
        return prescription
        
    def __str__(self) -> str:
        """Biểu diễn chuỗi của đối tượng đơn liều."""
        return f"{self.structure_name}: {self.prescribed_dose} Gy"
        
    def copy(self) -> 'DosePrescription':
        """
        Tạo một bản sao của đối tượng đơn liều.
        
        Returns
        -------
        DosePrescription
            Bản sao của đối tượng đơn liều
        """
        return copy.deepcopy(self)


class Prescription:
    """
    Lớp quản lý đơn liều tổng thể cho một kế hoạch xạ trị.
    
    Lớp này cung cấp các phương thức để quản lý đơn liều cho nhiều cấu trúc,
    cũng như phân đoạn và các thông tin bổ sung về đơn liều.
    """
    
    def __init__(
        self,
        prescription_id: str,
        name: str,
        status: PrescriptionStatus = PrescriptionStatus.DRAFT,
        created_date: Optional[datetime] = None,
        approved_date: Optional[datetime] = None
    ):
        """
        Khởi tạo một đối tượng đơn liều tổng thể.
        
        Parameters
        ----------
        prescription_id : str
            ID duy nhất của đơn liều
        name : str
            Tên đơn liều
        status : PrescriptionStatus
            Trạng thái đơn liều
        created_date : datetime, optional
            Ngày tạo đơn liều
        approved_date : datetime, optional
            Ngày phê duyệt đơn liều
        """
        self.prescription_id = prescription_id
        self.name = name
        self.status = status
        self.created_date = created_date if created_date else datetime.now()
        self.approved_date = approved_date
        
        self.fractionation = Fractionation()
        self.dose_prescriptions = {}  # Dict[str, DosePrescription]
        self.physician = ""
        self.comments = ""
        self.parameters = {}  # Dictionary lưu trữ các tham số bổ sung
        
    def set_fractionation(self, fractionation: Fractionation):
        """
        Đặt phân đoạn cho đơn liều.
        
        Parameters
        ----------
        fractionation : Fractionation
            Đối tượng phân đoạn
        """
        self.fractionation = fractionation
        
    def add_dose_prescription(self, dose_prescription: DosePrescription):
        """
        Thêm một đơn liều cho một cấu trúc.
        
        Parameters
        ----------
        dose_prescription : DosePrescription
            Đối tượng đơn liều cho cấu trúc
        """
        self.dose_prescriptions[dose_prescription.structure_id] = dose_prescription
        
    def get_dose_prescription(self, structure_id: str) -> Optional[DosePrescription]:
        """
        Lấy đơn liều cho một cấu trúc.
        
        Parameters
        ----------
        structure_id : str
            ID của cấu trúc
            
        Returns
        -------
        DosePrescription, optional
            Đối tượng đơn liều cho cấu trúc, hoặc None nếu không tồn tại
        """
        return self.dose_prescriptions.get(structure_id)
        
    def remove_dose_prescription(self, structure_id: str) -> bool:
        """
        Xóa đơn liều cho một cấu trúc.
        
        Parameters
        ----------
        structure_id : str
            ID của cấu trúc
            
        Returns
        -------
        bool
            True nếu xóa thành công, False nếu cấu trúc không tồn tại
        """
        if structure_id in self.dose_prescriptions:
            del self.dose_prescriptions[structure_id]
            return True
        return False
        
    def get_target_prescriptions(self) -> List[DosePrescription]:
        """
        Lấy danh sách đơn liều cho các cấu trúc mục tiêu.
        
        Returns
        -------
        List[DosePrescription]
            Danh sách đơn liều cho các cấu trúc mục tiêu
        """
        return [dp for dp in self.dose_prescriptions.values() if dp.is_target]
        
    def get_oar_prescriptions(self) -> List[DosePrescription]:
        """
        Lấy danh sách đơn liều cho các cơ quan nguy cấp.
        
        Returns
        -------
        List[DosePrescription]
            Danh sách đơn liều cho các cơ quan nguy cấp
        """
        return [dp for dp in self.dose_prescriptions.values() if not dp.is_target]
        
    def set_status(self, status: PrescriptionStatus):
        """
        Đặt trạng thái của đơn liều.
        
        Parameters
        ----------
        status : PrescriptionStatus
            Trạng thái mới
        """
        self.status = status
        
        # Cập nhật ngày phê duyệt nếu chuyển sang trạng thái Approved
        if status == PrescriptionStatus.APPROVED and not self.approved_date:
            self.approved_date = datetime.now()
            
    def set_physician(self, physician: str):
        """
        Đặt bác sĩ điều trị.
        
        Parameters
        ----------
        physician : str
            Tên bác sĩ điều trị
        """
        self.physician = physician
        
    def set_comments(self, comments: str):
        """
        Đặt ghi chú cho đơn liều.
        
        Parameters
        ----------
        comments : str
            Ghi chú về đơn liều
        """
        self.comments = comments
        
    def set_parameter(self, key: str, value: Any):
        """
        Đặt một tham số bổ sung.
        
        Parameters
        ----------
        key : str
            Tên tham số
        value : Any
            Giá trị tham số
        """
        self.parameters[key] = value
        
    def get_parameter(self, key: str, default: Any = None) -> Any:
        """
        Lấy giá trị của một tham số.
        
        Parameters
        ----------
        key : str
            Tên tham số
        default : Any, optional
            Giá trị mặc định nếu tham số không tồn tại
            
        Returns
        -------
        Any
            Giá trị của tham số
        """
        return self.parameters.get(key, default)
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Chuyển đổi đối tượng đơn liều tổng thể thành dictionary.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary chứa thông tin đơn liều tổng thể
        """
        return {
            'prescription_id': self.prescription_id,
            'name': self.name,
            'status': self.status.value,
            'created_date': self.created_date.isoformat() if self.created_date else None,
            'approved_date': self.approved_date.isoformat() if self.approved_date else None,
            'fractionation': self.fractionation.to_dict(),
            'dose_prescriptions': {k: v.to_dict() for k, v in self.dose_prescriptions.items()},
            'physician': self.physician,
            'comments': self.comments,
            'parameters': self.parameters
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Prescription':
        """
        Tạo đối tượng Prescription từ dictionary.
        
        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary chứa thông tin đơn liều tổng thể
            
        Returns
        -------
        Prescription
            Đối tượng đơn liều tổng thể
        """
        created_date = None
        if 'created_date' in data and data['created_date']:
            created_date = datetime.fromisoformat(data['created_date'])
            
        approved_date = None
        if 'approved_date' in data and data['approved_date']:
            approved_date = datetime.fromisoformat(data['approved_date'])
            
        prescription = cls(
            prescription_id=data.get('prescription_id', ''),
            name=data.get('name', ''),
            status=PrescriptionStatus(data.get('status', 'DRAFT')),
            created_date=created_date,
            approved_date=approved_date
        )
        
        # Phục hồi phân đoạn
        if 'fractionation' in data:
            prescription.fractionation = Fractionation.from_dict(data['fractionation'])
            
        # Phục hồi các đơn liều cấu trúc
        if 'dose_prescriptions' in data:
            for struct_id, dp_data in data['dose_prescriptions'].items():
                prescription.add_dose_prescription(DosePrescription.from_dict(dp_data))
                
        # Phục hồi các thuộc tính khác
        prescription.physician = data.get('physician', '')
        prescription.comments = data.get('comments', '')
        
        # Phục hồi các tham số
        if 'parameters' in data:
            prescription.parameters = data['parameters']
            
        return prescription
        
    def __str__(self) -> str:
        """Biểu diễn chuỗi của đối tượng đơn liều tổng thể."""
        targets = self.get_target_prescriptions()
        if targets:
            primary_target = sorted(targets, key=lambda x: x.priority)[0]
            return f"{self.name}: {primary_target.prescribed_dose} Gy in {self.fractionation.num_fractions} fractions"
        return f"{self.name}: {self.fractionation.num_fractions} fractions"
        
    def copy(self) -> 'Prescription':
        """
        Tạo một bản sao của đối tượng đơn liều tổng thể.
        
        Returns
        -------
        Prescription
            Bản sao của đối tượng đơn liều tổng thể
        """
        return copy.deepcopy(self)
