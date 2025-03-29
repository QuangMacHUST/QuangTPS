#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module quản lý đơn liều xạ trị trong QuangTPS.

Module này cung cấp các lớp và phương thức để mô tả và quản lý các đơn liều xạ trị,
bao gồm thông tin về liều kê đơn, phân đoạn và các ràng buộc liều.
"""

import logging
import copy
import json
import os
from typing import Dict, List, Optional, Any, Tuple, Union
from enum import Enum
from datetime import datetime

from quangtps.core.services import ServiceRegistry
from quangtps.core.constants import PRESCRIPTION_TYPES, DOSE_UNITS
from quangtps.database.prescription_db import PrescriptionDB as PrescriptionDatabase

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


class StructurePrescription:
    """
    Lớp quản lý đơn liều và các ràng buộc liều cho một cấu trúc cụ thể.
    
    Lớp này mở rộng thông tin từ DosePrescription với các dữ liệu chi tiết hơn về cấu trúc,
    bao gồm thông tin về mật độ mô, các ràng buộc sinh học, và các đặc tính đặc trưng khác.
    """
    
    def __init__(
        self,
        structure_id: str,
        structure_name: str,
        prescribed_dose: float,
        is_target: bool = True,
        priority: int = 1,
        structure_type: str = ""
    ):
        """
        Khởi tạo một đối tượng đơn liều cấu trúc.
        
        Parameters
        ----------
        structure_id : str
            ID của cấu trúc
        structure_name : str
            Tên hiển thị của cấu trúc
        prescribed_dose : float
            Liều kê đơn (Gy)
        is_target : bool
            Cấu trúc có phải là mục tiêu
        priority : int
            Mức độ ưu tiên của cấu trúc
        structure_type : str
            Loại cấu trúc (GTV, CTV, PTV, OAR, v.v.)
        """
        self.structure_id = structure_id
        self.structure_name = structure_name
        self.structure_type = structure_type
        self.prescribed_dose = prescribed_dose
        self.is_target = is_target
        self.priority = priority
        
        # Đặc tính sinh học và vật lý của cấu trúc
        self.alpha_beta_ratio = 10.0 if is_target else 3.0  # Hệ số alpha/beta
        self.density_override = None  # Ghi đè mật độ mô (g/cm³)
        self.cell_sensitivity = 0.0  # Độ nhạy của tế bào với bức xạ
        
        # Thông tin chi tiết về cấu trúc
        self.volume = 0.0  # Thể tích (cm³)
        self.organ_type = ""  # Loại cơ quan (serial, parallel, v.v.)
        self.biological_effect = 0.0  # Hiệu ứng sinh học
        
        # Các ràng buộc liều
        self.min_dose_constraint = None  # Optional[float]
        self.max_dose_constraint = None  # Optional[float]
        self.mean_dose_constraint = None  # Optional[float]
        self.dvh_constraints = []  # List of (dose, volume) tuples
        
        # Thông tin bổ sung
        self.metadata = {}  # Dict[str, Any]
    
    def set_biological_parameters(
        self,
        alpha_beta_ratio: float,
        cell_sensitivity: Optional[float] = None
    ):
        """
        Đặt các tham số sinh học cho cấu trúc.
        
        Parameters
        ----------
        alpha_beta_ratio : float
            Tỷ lệ alpha/beta cho mô
        cell_sensitivity : float, optional
            Độ nhạy của tế bào với bức xạ
        """
        self.alpha_beta_ratio = alpha_beta_ratio
        if cell_sensitivity is not None:
            self.cell_sensitivity = cell_sensitivity
    
    def set_density_override(self, density: Optional[float] = None):
        """
        Đặt ghi đè mật độ mô.
        
        Parameters
        ----------
        density : float, optional
            Mật độ mô (g/cm³), None để hủy ghi đè
        """
        self.density_override = density
    
    def set_structure_volume(self, volume: float):
        """
        Đặt thể tích của cấu trúc.
        
        Parameters
        ----------
        volume : float
            Thể tích (cm³)
        """
        self.volume = volume
    
    def set_organ_type(self, organ_type: str):
        """
        Đặt loại cơ quan.
        
        Parameters
        ----------
        organ_type : str
            Loại cơ quan (serial, parallel, v.v.)
        """
        self.organ_type = organ_type
    
    def set_dose_constraints(
        self,
        min_dose: Optional[float] = None,
        max_dose: Optional[float] = None,
        mean_dose: Optional[float] = None
    ):
        """
        Đặt các ràng buộc liều cơ bản.
        
        Parameters
        ----------
        min_dose : float, optional
            Liều tối thiểu (Gy)
        max_dose : float, optional
            Liều tối đa (Gy)
        mean_dose : float, optional
            Liều trung bình (Gy)
        """
        if min_dose is not None:
            self.min_dose_constraint = min_dose
        if max_dose is not None:
            self.max_dose_constraint = max_dose
        if mean_dose is not None:
            self.mean_dose_constraint = mean_dose
    
    def add_dvh_constraint(self, dose: float, volume: float):
        """
        Thêm một ràng buộc DVH.
        
        Parameters
        ----------
        dose : float
            Liều (Gy)
        volume : float
            Thể tích (%)
        """
        self.dvh_constraints.append((dose, volume))
    
    def clear_dvh_constraints(self):
        """Xóa tất cả các ràng buộc DVH."""
        self.dvh_constraints = []
    
    def set_metadata(self, key: str, value: Any):
        """
        Đặt một trường metadata.
        
        Parameters
        ----------
        key : str
            Tên trường
        value : Any
            Giá trị trường
        """
        self.metadata[key] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Chuyển đổi đối tượng đơn liều cấu trúc thành dictionary.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary chứa thông tin đơn liều cấu trúc
        """
        return {
            'structure_id': self.structure_id,
            'structure_name': self.structure_name,
            'structure_type': self.structure_type,
            'prescribed_dose': self.prescribed_dose,
            'is_target': self.is_target,
            'priority': self.priority,
            'alpha_beta_ratio': self.alpha_beta_ratio,
            'density_override': self.density_override,
            'cell_sensitivity': self.cell_sensitivity,
            'volume': self.volume,
            'organ_type': self.organ_type,
            'biological_effect': self.biological_effect,
            'min_dose_constraint': self.min_dose_constraint,
            'max_dose_constraint': self.max_dose_constraint,
            'mean_dose_constraint': self.mean_dose_constraint,
            'dvh_constraints': self.dvh_constraints,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StructurePrescription':
        """
        Tạo đối tượng StructurePrescription từ dictionary.
        
        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary chứa thông tin đơn liều cấu trúc
            
        Returns
        -------
        StructurePrescription
            Đối tượng đơn liều cấu trúc
        """
        obj = cls(
            structure_id=data['structure_id'],
            structure_name=data['structure_name'],
            prescribed_dose=data['prescribed_dose'],
            is_target=data.get('is_target', True),
            priority=data.get('priority', 1),
            structure_type=data.get('structure_type', '')
        )
        
        # Khôi phục các tham số sinh học và vật lý
        obj.alpha_beta_ratio = data.get('alpha_beta_ratio', 10.0 if obj.is_target else 3.0)
        obj.density_override = data.get('density_override')
        obj.cell_sensitivity = data.get('cell_sensitivity', 0.0)
        
        # Khôi phục thông tin chi tiết
        obj.volume = data.get('volume', 0.0)
        obj.organ_type = data.get('organ_type', '')
        obj.biological_effect = data.get('biological_effect', 0.0)
        
        # Khôi phục ràng buộc liều
        obj.min_dose_constraint = data.get('min_dose_constraint')
        obj.max_dose_constraint = data.get('max_dose_constraint')
        obj.mean_dose_constraint = data.get('mean_dose_constraint')
        obj.dvh_constraints = data.get('dvh_constraints', [])
        
        # Khôi phục metadata
        obj.metadata = data.get('metadata', {})
        
        return obj
    
    def __str__(self) -> str:
        """Biểu diễn chuỗi của đối tượng đơn liều cấu trúc."""
        status = "Mục tiêu" if self.is_target else "Cơ quan nguy cấp"
        return f"{self.structure_name} [{self.structure_id}] - {status}: {self.prescribed_dose} Gy"
    
    def copy(self) -> 'StructurePrescription':
        """
        Tạo một bản sao của đối tượng đơn liều cấu trúc.
        
        Returns
        -------
        StructurePrescription
            Bản sao của đối tượng đơn liều cấu trúc
        """
        return copy.deepcopy(self)


class DoseConstraint:
    """Represents a dose constraint for a structure in a treatment plan."""
    
    CONSTRAINT_TYPES = [
        "D_MAX", "D_MIN", "D_MEAN", 
        "D_X", "V_X",
        "MAX_DVH", "MIN_DVH",
        "CONFORMITY", "HOMOGENEITY", "GRADIENT"
    ]
    
    PRIORITIES = ["REQUIRED", "PRIORITY_HIGH", "PRIORITY_MEDIUM", "PRIORITY_LOW"]
    
    def __init__(
        self, 
        structure_name: str,
        constraint_type: str,
        dose_value: float = None,
        volume_value: float = None,
        dose_unit: str = "Gy",
        volume_unit: str = "%",
        priority: str = "PRIORITY_MEDIUM",
        achieved: bool = False,
        evaluation_value: float = None,
        description: str = None
    ):
        """Initialize a dose constraint.
        
        Args:
            structure_name: Name of the structure
            constraint_type: Type of the constraint (D_MAX, D_MIN, V_X, etc.)
            dose_value: Dose value for the constraint
            volume_value: Volume value for the constraint
            dose_unit: Unit for dose (Gy, cGy, etc.)
            volume_unit: Unit for volume (%, cc)
            priority: Priority of the constraint
            achieved: Whether the constraint is achieved
            evaluation_value: Actual value from evaluation
            description: Human-readable description of the constraint
        """
        self.structure_name = structure_name
        
        if constraint_type not in self.CONSTRAINT_TYPES:
            logger.warning(f"Unknown constraint type: {constraint_type}. Using D_MAX.")
            self.constraint_type = "D_MAX"
        else:
            self.constraint_type = constraint_type
            
        self.dose_value = dose_value
        self.volume_value = volume_value
        self.dose_unit = dose_unit if dose_unit in DOSE_UNITS else "Gy"
        self.volume_unit = volume_unit
        
        if priority not in self.PRIORITIES:
            logger.warning(f"Unknown priority: {priority}. Using PRIORITY_MEDIUM.")
            self.priority = "PRIORITY_MEDIUM"
        else:
            self.priority = priority
            
        self.achieved = achieved
        self.evaluation_value = evaluation_value
        self.description = description or self._generate_description()
        
    def _generate_description(self) -> str:
        """Generate a human-readable description of the constraint."""
        if self.constraint_type == "D_MAX":
            return f"Maximum dose to {self.structure_name} < {self.dose_value} {self.dose_unit}"
        elif self.constraint_type == "D_MIN":
            return f"Minimum dose to {self.structure_name} > {self.dose_value} {self.dose_unit}"
        elif self.constraint_type == "D_MEAN":
            return f"Mean dose to {self.structure_name} < {self.dose_value} {self.dose_unit}"
        elif self.constraint_type.startswith("D_"):
            # D95, D90, etc.
            volume = self.constraint_type.split("_")[1]
            return f"Dose to {volume}% of {self.structure_name} > {self.dose_value} {self.dose_unit}"
        elif self.constraint_type.startswith("V_"):
            # V20Gy, V5Gy, etc.
            dose = self.constraint_type.split("_")[1]
            return f"Volume of {self.structure_name} receiving {dose}{self.dose_unit} < {self.volume_value}{self.volume_unit}"
        else:
            return f"{self.constraint_type} constraint for {self.structure_name}"
    
    def to_dict(self) -> Dict:
        """Convert the constraint to a dictionary."""
        return {
            "structure_name": self.structure_name,
            "constraint_type": self.constraint_type,
            "dose_value": self.dose_value,
            "volume_value": self.volume_value,
            "dose_unit": self.dose_unit,
            "volume_unit": self.volume_unit,
            "priority": self.priority,
            "achieved": self.achieved,
            "evaluation_value": self.evaluation_value,
            "description": self.description
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'DoseConstraint':
        """Create a constraint from a dictionary."""
        return cls(
            structure_name=data.get("structure_name"),
            constraint_type=data.get("constraint_type"),
            dose_value=data.get("dose_value"),
            volume_value=data.get("volume_value"),
            dose_unit=data.get("dose_unit", "Gy"),
            volume_unit=data.get("volume_unit", "%"),
            priority=data.get("priority", "PRIORITY_MEDIUM"),
            achieved=data.get("achieved", False),
            evaluation_value=data.get("evaluation_value"),
            description=data.get("description")
        )
    
    def evaluate(self, dvh_data: Dict) -> Tuple[bool, float]:
        """Evaluate the constraint against DVH data.
        
        Args:
            dvh_data: Dictionary containing DVH data for structures
            
        Returns:
            Tuple of (constraint met, actual value)
        """
        # This is a placeholder. In a real implementation, this would
        # evaluate the constraint against actual DVH data.
        if self.structure_name not in dvh_data:
            logger.warning(f"Structure {self.structure_name} not found in DVH data")
            return False, None
        
        # Implementation depends on the specific DVH data format
        # For now, just return placeholder values
        return True, 0.0


class ClinicalGoal:
    """Represents a clinical goal for a treatment plan, including multiple constraints."""
    
    def __init__(
        self,
        name: str,
        description: str = None,
        constraints: List[DoseConstraint] = None,
        achieved: bool = False
    ):
        """Initialize a clinical goal.
        
        Args:
            name: Name of the clinical goal
            description: Description of the goal
            constraints: List of dose constraints for the goal
            achieved: Whether the goal is achieved
        """
        self.name = name
        self.description = description or name
        self.constraints = constraints or []
        self.achieved = achieved
        
    def add_constraint(self, constraint: DoseConstraint):
        """Add a constraint to the clinical goal."""
        self.constraints.append(constraint)
        
    def remove_constraint(self, index: int):
        """Remove a constraint from the clinical goal."""
        if 0 <= index < len(self.constraints):
            del self.constraints[index]
            
    def to_dict(self) -> Dict:
        """Convert the clinical goal to a dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "constraints": [c.to_dict() for c in self.constraints],
            "achieved": self.achieved
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ClinicalGoal':
        """Create a clinical goal from a dictionary."""
        constraints = [DoseConstraint.from_dict(c) for c in data.get("constraints", [])]
        return cls(
            name=data.get("name", ""),
            description=data.get("description"),
            constraints=constraints,
            achieved=data.get("achieved", False)
        )
    
    def evaluate(self, dvh_data: Dict) -> bool:
        """Evaluate all constraints in the clinical goal.
        
        Args:
            dvh_data: Dictionary containing DVH data for structures
            
        Returns:
            Whether the goal is achieved (all required constraints met)
        """
        required_constraints_met = True
        
        for constraint in self.constraints:
            met, value = constraint.evaluate(dvh_data)
            constraint.achieved = met
            constraint.evaluation_value = value
            
            if not met and constraint.priority == "REQUIRED":
                required_constraints_met = False
                
        self.achieved = required_constraints_met
        return required_constraints_met


class PrescriptionTemplate:
    """Template for prescription parameters for a specific treatment site."""
    
    def __init__(
        self,
        name: str,
        site: str,
        technique: str,
        prescription_type: str = "STANDARD",
        dose: float = None,
        fractions: int = None,
        targets: Dict[str, Dict] = None,
        clinical_goals: List[ClinicalGoal] = None,
        description: str = None,
        version: str = "1.0",
        last_modified: datetime = None
    ):
        """Initialize a prescription template.
        
        Args:
            name: Template name
            site: Treatment site (e.g., "Lung", "Prostate")
            technique: Treatment technique (e.g., "IMRT", "VMAT")
            prescription_type: Type of prescription (e.g., "STANDARD", "SIB")
            dose: Reference dose in Gy
            fractions: Number of fractions
            targets: Dictionary of target structures and their prescribed doses
            clinical_goals: List of clinical goals
            description: Template description
            version: Template version
            last_modified: Last modification date
        """
        self.name = name
        self.site = site
        self.technique = technique
        
        if prescription_type not in PRESCRIPTION_TYPES:
            logger.warning(f"Unknown prescription type: {prescription_type}. Using STANDARD.")
            self.prescription_type = "STANDARD"
        else:
            self.prescription_type = prescription_type
            
        self.dose = dose
        self.fractions = fractions
        self.targets = targets or {}
        self.clinical_goals = clinical_goals or []
        self.description = description or f"{site} {technique} Template"
        self.version = version
        self.last_modified = last_modified or datetime.now()
        
    def to_dict(self) -> Dict:
        """Convert the template to a dictionary."""
        return {
            "name": self.name,
            "site": self.site,
            "technique": self.technique,
            "prescription_type": self.prescription_type,
            "dose": self.dose,
            "fractions": self.fractions,
            "targets": self.targets,
            "clinical_goals": [g.to_dict() for g in self.clinical_goals],
            "description": self.description,
            "version": self.version,
            "last_modified": self.last_modified.isoformat() if self.last_modified else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'PrescriptionTemplate':
        """Create a template from a dictionary."""
        clinical_goals = [ClinicalGoal.from_dict(g) for g in data.get("clinical_goals", [])]
        last_modified = None
        if data.get("last_modified"):
            try:
                last_modified = datetime.fromisoformat(data["last_modified"])
            except (ValueError, TypeError):
                last_modified = datetime.now()
                
        return cls(
            name=data.get("name", ""),
            site=data.get("site", ""),
            technique=data.get("technique", ""),
            prescription_type=data.get("prescription_type", "STANDARD"),
            dose=data.get("dose"),
            fractions=data.get("fractions"),
            targets=data.get("targets", {}),
            clinical_goals=clinical_goals,
            description=data.get("description"),
            version=data.get("version", "1.0"),
            last_modified=last_modified
        )


class Prescription:
    """Represents a prescription for a treatment plan."""
    
    def __init__(
        self,
        id: int = None,
        patient_id: str = None,
        plan_id: int = None,
        prescription_type: str = "STANDARD",
        site: str = None,
        technique: str = None,
        dose: float = None,
        fractions: int = None,
        dose_per_fraction: float = None,
        targets: Dict[str, Dict] = None,
        clinical_goals: List[ClinicalGoal] = None,
        creation_date: datetime = None,
        last_modified: datetime = None,
        description: str = None,
        notes: str = None,
        template_name: str = None
    ):
        """Initialize a prescription.
        
        Args:
            id: Prescription ID
            patient_id: Patient ID
            plan_id: Plan ID
            prescription_type: Type of prescription
            site: Treatment site
            technique: Treatment technique
            dose: Reference dose in Gy
            fractions: Number of fractions
            dose_per_fraction: Dose per fraction in Gy
            targets: Dictionary of target structures and their prescribed doses
            clinical_goals: List of clinical goals
            creation_date: Creation date
            last_modified: Last modification date
            description: Prescription description
            notes: Additional notes
            template_name: Name of the template used, if any
        """
        self.id = id
        self.patient_id = patient_id
        self.plan_id = plan_id
        
        if prescription_type not in PRESCRIPTION_TYPES:
            logger.warning(f"Unknown prescription type: {prescription_type}. Using STANDARD.")
            self.prescription_type = "STANDARD"
        else:
            self.prescription_type = prescription_type
            
        self.site = site
        self.technique = technique
        self.dose = dose
        self.fractions = fractions
        
        # Calculate dose per fraction if not provided
        if dose_per_fraction is None and dose is not None and fractions is not None and fractions > 0:
            self.dose_per_fraction = dose / fractions
        else:
            self.dose_per_fraction = dose_per_fraction
            
        self.targets = targets or {}
        self.clinical_goals = clinical_goals or []
        self.creation_date = creation_date or datetime.now()
        self.last_modified = last_modified or datetime.now()
        self.description = description or f"Prescription for {site}" if site else "New Prescription"
        self.notes = notes
        self.template_name = template_name
        
    def to_dict(self) -> Dict:
        """Convert the prescription to a dictionary."""
        return {
            "id": self.id,
            "patient_id": self.patient_id,
            "plan_id": self.plan_id,
            "prescription_type": self.prescription_type,
            "site": self.site,
            "technique": self.technique,
            "dose": self.dose,
            "fractions": self.fractions,
            "dose_per_fraction": self.dose_per_fraction,
            "targets": self.targets,
            "clinical_goals": [g.to_dict() for g in self.clinical_goals],
            "creation_date": self.creation_date.isoformat() if self.creation_date else None,
            "last_modified": self.last_modified.isoformat() if self.last_modified else None,
            "description": self.description,
            "notes": self.notes,
            "template_name": self.template_name
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Prescription':
        """Create a prescription from a dictionary."""
        clinical_goals = [ClinicalGoal.from_dict(g) for g in data.get("clinical_goals", [])]
        
        creation_date = None
        if data.get("creation_date"):
            try:
                creation_date = datetime.fromisoformat(data["creation_date"])
            except (ValueError, TypeError):
                creation_date = datetime.now()
                
        last_modified = None
        if data.get("last_modified"):
            try:
                last_modified = datetime.fromisoformat(data["last_modified"])
            except (ValueError, TypeError):
                last_modified = datetime.now()
                
        return cls(
            id=data.get("id"),
            patient_id=data.get("patient_id"),
            plan_id=data.get("plan_id"),
            prescription_type=data.get("prescription_type", "STANDARD"),
            site=data.get("site"),
            technique=data.get("technique"),
            dose=data.get("dose"),
            fractions=data.get("fractions"),
            dose_per_fraction=data.get("dose_per_fraction"),
            targets=data.get("targets", {}),
            clinical_goals=clinical_goals,
            creation_date=creation_date,
            last_modified=last_modified,
            description=data.get("description"),
            notes=data.get("notes"),
            template_name=data.get("template_name")
        )
    
    def add_clinical_goal(self, goal: ClinicalGoal):
        """Add a clinical goal to the prescription."""
        self.clinical_goals.append(goal)
        self.last_modified = datetime.now()
        
    def remove_clinical_goal(self, index: int):
        """Remove a clinical goal from the prescription."""
        if 0 <= index < len(self.clinical_goals):
            del self.clinical_goals[index]
            self.last_modified = datetime.now()
            
    def add_target(self, name: str, dose: float, dose_unit: str = "Gy", volume: float = 100, volume_unit: str = "%"):
        """Add a target to the prescription."""
        self.targets[name] = {
            "dose": dose,
            "dose_unit": dose_unit,
            "volume": volume,
            "volume_unit": volume_unit
        }
        self.last_modified = datetime.now()
        
    def remove_target(self, name: str):
        """Remove a target from the prescription."""
        if name in self.targets:
            del self.targets[name]
            self.last_modified = datetime.now()
            
    def update_from_template(self, template: PrescriptionTemplate):
        """Update prescription from a template."""
        self.site = template.site
        self.technique = template.technique
        self.prescription_type = template.prescription_type
        self.dose = template.dose
        self.fractions = template.fractions
        
        if self.dose is not None and self.fractions is not None and self.fractions > 0:
            self.dose_per_fraction = self.dose / self.fractions
            
        # Add new targets, keeping existing ones
        for name, details in template.targets.items():
            self.targets[name] = details.copy()
            
        # Replace clinical goals with template's goals
        self.clinical_goals = [ClinicalGoal.from_dict(g.to_dict()) for g in template.clinical_goals]
        
        self.template_name = template.name
        self.last_modified = datetime.now()
        
    def save(self):
        """Save the prescription to the database."""
        db = ServiceRegistry.get("PrescriptionDatabase")
        if db:
            if self.id is None:
                # New prescription
                prescription_id = db.create_prescription(self.to_dict())
                if prescription_id:
                    self.id = prescription_id
                    logger.info(f"Created new prescription with ID {self.id}")
                    return True
                else:
                    logger.error("Failed to create prescription")
                    return False
            else:
                # Update existing prescription
                success = db.update_prescription(self.id, self.to_dict())
                if success:
                    logger.info(f"Updated prescription with ID {self.id}")
                    return True
                else:
                    logger.error(f"Failed to update prescription with ID {self.id}")
                    return False
        else:
            logger.error("PrescriptionDatabase service not available")
            return False
            
    @classmethod
    def load(cls, prescription_id: int) -> Optional['Prescription']:
        """Load a prescription from the database."""
        db = ServiceRegistry.get("PrescriptionDatabase")
        if db:
            data = db.get_prescription(prescription_id)
            if data:
                return cls.from_dict(data)
            else:
                logger.warning(f"Prescription with ID {prescription_id} not found")
                return None
        else:
            logger.error("PrescriptionDatabase service not available")
            return None
            
    @classmethod
    def load_for_plan(cls, plan_id: int) -> Optional['Prescription']:
        """Load a prescription for a specific plan."""
        db = ServiceRegistry.get("PrescriptionDatabase")
        if db:
            data = db.get_prescription_by_plan(plan_id)
            if data:
                return cls.from_dict(data)
            else:
                logger.info(f"No prescription found for plan ID {plan_id}")
                return None
        else:
            logger.error("PrescriptionDatabase service not available")
            return None
    
    @classmethod
    def load_for_patient(cls, patient_id: str) -> List['Prescription']:
        """Load all prescriptions for a patient."""
        db = ServiceRegistry.get("PrescriptionDatabase")
        if db:
            data_list = db.get_prescriptions_by_patient(patient_id)
            return [cls.from_dict(data) for data in data_list]
        else:
            logger.error("PrescriptionDatabase service not available")
            return []


class PrescriptionTemplateManager:
    """Manages prescription templates."""
    
    def __init__(self, templates_dir: str = None):
        """Initialize the template manager.
        
        Args:
            templates_dir: Directory containing template files
        """
        self.templates_dir = templates_dir or os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "data",
            "clinical_protocols"
        )
        self.templates = {}
        self._load_templates()
        
    def _load_templates(self):
        """Load templates from the templates directory."""
        if not os.path.exists(self.templates_dir):
            os.makedirs(self.templates_dir, exist_ok=True)
            logger.info(f"Created templates directory: {self.templates_dir}")
            return
        
        for filename in os.listdir(self.templates_dir):
            if filename.endswith(".json"):
                try:
                    with open(os.path.join(self.templates_dir, filename), "r") as f:
                        data = json.load(f)
                        template = PrescriptionTemplate.from_dict(data)
                        self.templates[template.name] = template
                except Exception as e:
                    logger.error(f"Error loading template from {filename}: {str(e)}")
                    
        logger.info(f"Loaded {len(self.templates)} templates")
    
    def get_template(self, name: str) -> Optional[PrescriptionTemplate]:
        """Get a template by name."""
        return self.templates.get(name)
    
    def get_templates_by_site(self, site: str) -> List[PrescriptionTemplate]:
        """Get all templates for a specific site."""
        return [t for t in self.templates.values() if t.site.lower() == site.lower()]
    
    def get_templates_by_technique(self, technique: str) -> List[PrescriptionTemplate]:
        """Get all templates for a specific technique."""
        return [t for t in self.templates.values() if t.technique.lower() == technique.lower()]
    
    def get_all_templates(self) -> List[PrescriptionTemplate]:
        """Get all templates."""
        return list(self.templates.values())
    
    def get_all_sites(self) -> List[str]:
        """Get all unique treatment sites."""
        return sorted(set(t.site for t in self.templates.values()))
    
    def save_template(self, template: PrescriptionTemplate) -> bool:
        """Save a template to disk."""
        try:
            if not os.path.exists(self.templates_dir):
                os.makedirs(self.templates_dir, exist_ok=True)
                
            filename = f"{template.name.replace(' ', '_')}.json"
            filepath = os.path.join(self.templates_dir, filename)
            
            with open(filepath, "w") as f:
                json.dump(template.to_dict(), f, indent=2)
                
            self.templates[template.name] = template
            logger.info(f"Saved template to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving template {template.name}: {str(e)}")
            return False
    
    def delete_template(self, name: str) -> bool:
        """Delete a template."""
        template = self.templates.get(name)
        if not template:
            logger.warning(f"Template '{name}' not found")
            return False
        
        try:
            filename = f"{name.replace(' ', '_')}.json"
            filepath = os.path.join(self.templates_dir, filename)
            
            if os.path.exists(filepath):
                os.remove(filepath)
                
            del self.templates[name]
            logger.info(f"Deleted template '{name}'")
            return True
        except Exception as e:
            logger.error(f"Error deleting template '{name}': {str(e)}")
            return False
    
    def create_standard_templates(self):
        """Create a set of standard templates for common treatment sites."""
        # Lung SBRT template
        lung_sbrt = PrescriptionTemplate(
            name="Lung SBRT",
            site="Lung",
            technique="SBRT",
            prescription_type="STANDARD",
            dose=50.0,
            fractions=5,
            targets={
                "PTV": {"dose": 50.0, "dose_unit": "Gy", "volume": 95, "volume_unit": "%"}
            }
        )
        
        # Add clinical goals
        ptv_coverage = ClinicalGoal(name="PTV Coverage", description="PTV dose coverage")
        ptv_coverage.add_constraint(DoseConstraint(
            structure_name="PTV",
            constraint_type="D_95",
            dose_value=47.5,
            dose_unit="Gy",
            priority="REQUIRED"
        ))
        
        lung_constraint = ClinicalGoal(name="Lung Dose", description="Normal lung dose limits")
        lung_constraint.add_constraint(DoseConstraint(
            structure_name="Lung_L",
            constraint_type="V_20Gy",
            volume_value=10,
            volume_unit="%",
            priority="PRIORITY_HIGH"
        ))
        lung_constraint.add_constraint(DoseConstraint(
            structure_name="Lung_R",
            constraint_type="V_20Gy",
            volume_value=10,
            volume_unit="%",
            priority="PRIORITY_HIGH"
        ))
        
        spinal_cord = ClinicalGoal(name="Spinal Cord", description="Spinal cord maximum dose")
        spinal_cord.add_constraint(DoseConstraint(
            structure_name="SpinalCord",
            constraint_type="D_MAX",
            dose_value=30,
            dose_unit="Gy",
            priority="REQUIRED"
        ))
        
        lung_sbrt.clinical_goals = [ptv_coverage, lung_constraint, spinal_cord]
        self.save_template(lung_sbrt)
        
        # Prostate IMRT template
        prostate_imrt = PrescriptionTemplate(
            name="Prostate IMRT",
            site="Prostate",
            technique="IMRT",
            prescription_type="STANDARD",
            dose=78.0,
            fractions=39,
            targets={
                "PTV": {"dose": 78.0, "dose_unit": "Gy", "volume": 95, "volume_unit": "%"}
            }
        )
        
        # Add clinical goals for prostate
        ptv_coverage = ClinicalGoal(name="PTV Coverage", description="PTV dose coverage")
        ptv_coverage.add_constraint(DoseConstraint(
            structure_name="PTV",
            constraint_type="D_95",
            dose_value=74.1,
            dose_unit="Gy",
            priority="REQUIRED"
        ))
        
        rectum_constraint = ClinicalGoal(name="Rectum Dose", description="Rectum dose limits")
        rectum_constraint.add_constraint(DoseConstraint(
            structure_name="Rectum",
            constraint_type="V_70Gy",
            volume_value=15,
            volume_unit="%",
            priority="PRIORITY_HIGH"
        ))
        rectum_constraint.add_constraint(DoseConstraint(
            structure_name="Rectum",
            constraint_type="V_50Gy",
            volume_value=50,
            volume_unit="%",
            priority="PRIORITY_MEDIUM"
        ))
        
        bladder_constraint = ClinicalGoal(name="Bladder Dose", description="Bladder dose limits")
        bladder_constraint.add_constraint(DoseConstraint(
            structure_name="Bladder",
            constraint_type="V_70Gy",
            volume_value=25,
            volume_unit="%",
            priority="PRIORITY_HIGH"
        ))
        
        prostate_imrt.clinical_goals = [ptv_coverage, rectum_constraint, bladder_constraint]
        self.save_template(prostate_imrt)
        
        # Head and Neck IMRT template
        hn_imrt = PrescriptionTemplate(
            name="Head and Neck IMRT",
            site="Head and Neck",
            technique="IMRT",
            prescription_type="SIB",  # Simultaneous Integrated Boost
            dose=70.0,
            fractions=35,
            targets={
                "PTV_High": {"dose": 70.0, "dose_unit": "Gy", "volume": 95, "volume_unit": "%"},
                "PTV_Intermediate": {"dose": 63.0, "dose_unit": "Gy", "volume": 95, "volume_unit": "%"},
                "PTV_Low": {"dose": 56.0, "dose_unit": "Gy", "volume": 95, "volume_unit": "%"}
            }
        )
        
        # Add clinical goals for head and neck
        ptv_high = ClinicalGoal(name="PTV High Coverage", description="High dose PTV coverage")
        ptv_high.add_constraint(DoseConstraint(
            structure_name="PTV_High",
            constraint_type="D_95",
            dose_value=66.5,
            dose_unit="Gy",
            priority="REQUIRED"
        ))
        
        ptv_int = ClinicalGoal(name="PTV Intermediate Coverage", description="Intermediate dose PTV coverage")
        ptv_int.add_constraint(DoseConstraint(
            structure_name="PTV_Intermediate",
            constraint_type="D_95",
            dose_value=59.85,
            dose_unit="Gy",
            priority="REQUIRED"
        ))
        
        ptv_low = ClinicalGoal(name="PTV Low Coverage", description="Low dose PTV coverage")
        ptv_low.add_constraint(DoseConstraint(
            structure_name="PTV_Low",
            constraint_type="D_95",
            dose_value=53.2,
            dose_unit="Gy",
            priority="REQUIRED"
        ))
        
        parotid = ClinicalGoal(name="Parotid Sparing", description="Parotid dose limits")
        parotid.add_constraint(DoseConstraint(
            structure_name="Parotid_L",
            constraint_type="D_MEAN",
            dose_value=26,
            dose_unit="Gy",
            priority="PRIORITY_HIGH"
        ))
        parotid.add_constraint(DoseConstraint(
            structure_name="Parotid_R",
            constraint_type="D_MEAN",
            dose_value=26,
            dose_unit="Gy",
            priority="PRIORITY_HIGH"
        ))
        
        spinal_cord = ClinicalGoal(name="Spinal Cord", description="Spinal cord maximum dose")
        spinal_cord.add_constraint(DoseConstraint(
            structure_name="SpinalCord",
            constraint_type="D_MAX",
            dose_value=45,
            dose_unit="Gy",
            priority="REQUIRED"
        ))
        
        brainstem = ClinicalGoal(name="Brainstem", description="Brainstem maximum dose")
        brainstem.add_constraint(DoseConstraint(
            structure_name="Brainstem",
            constraint_type="D_MAX",
            dose_value=54,
            dose_unit="Gy",
            priority="REQUIRED"
        ))
        
        hn_imrt.clinical_goals = [ptv_high, ptv_int, ptv_low, parotid, spinal_cord, brainstem]
        self.save_template(hn_imrt)
        
        logger.info("Created standard prescription templates")
