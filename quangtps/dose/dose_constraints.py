#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module quản lý các ràng buộc liều lượng.

Module này cung cấp các lớp và hàm để định nghĩa và quản lý
các ràng buộc liều lượng cho kế hoạch điều trị xạ trị.
"""

import uuid
import logging
from typing import Dict, List, Optional, Any, Tuple, Union, cast, Callable, Protocol, TypedDict, TypeVar, Sequence
from dataclasses import dataclass, field
from enum import Enum, auto
import os
import numpy as np
import json

from quangtps.prescription.prescription import PriorityLevel, DoseUnit, VolumeUnit
from quangtps.optimization.constraints import (
    MaxDoseConstraint,
    MinDoseConstraint,
    MeanDoseConstraint,
    DoseVolumeConstraint as OptimizationDoseVolumeConstraint,
    ConstraintBase
)
from quangtps.core.types import DoseGrid
from quangtps.dose.dose_grid import DoseGrid as DoseGridClass

logger = logging.getLogger(__name__)

class ConstraintType(Enum):
    """Các loại ràng buộc."""
    DOSE_VOLUME = "DOSE_VOLUME"  # Ràng buộc liều-thể tích
    MAX_DOSE = "MAX_DOSE"  # Liều tối đa
    MIN_DOSE = "MIN_DOSE"  # Liều tối thiểu
    MEAN_DOSE = "MEAN_DOSE"  # Liều trung bình
    CONFORMITY = "CONFORMITY"  # Độ phù hợp
    HOMOGENEITY = "HOMOGENEITY"  # Độ đồng nhất
    GRADIENT = "GRADIENT"  # Độ dốc
    CUSTOM = "CUSTOM"  # Tùy chỉnh

class ConstraintDirection(Enum):
    """Hướng của ràng buộc."""
    UPPER = "UPPER"  # Giới hạn trên (≤)
    LOWER = "LOWER"  # Giới hạn dưới (≥)
    EQUAL = "EQUAL"  # Bằng (=)

class EvaluationMetrics(TypedDict, total=False):
    """Type definition for the evaluation metrics dictionary."""
    pass

class MetricsProvider(Protocol):
    """Protocol for objects that provide metrics."""
    def get_metrics(self) -> Dict[str, Dict[str, float]]: ...

class EvaluationProvider(Protocol):
    """Protocol for objects that provide evaluation."""
    def get_evaluation(self) -> MetricsProvider: ...

class DoseConstraintProtocol(Protocol):
    """Protocol for dose constraints."""
    priority: 'ConstraintPriority'  # Use string literal type annotation to resolve forward reference
    def to_dict(self) -> Dict[str, Any]: ...

class ConstraintPriority(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class DVHPoint(TypedDict):
    dose: float
    volume: float

class DVHAnalysis(TypedDict):
    dose_metrics: Dict[str, float]
    volume_metrics: Dict[str, float]
    gradient_metrics: Dict[str, float]
    conformity_metrics: Dict[str, float]

class ConstraintStatus(TypedDict):
    name: str
    priority: str
    type: str
    value: float
    is_met: bool
    actual_value: float
    difference: float
    dvh_analysis: Optional[Dict[str, Any]]

class ProtocolData(TypedDict):
    name: str
    version: str
    constraints: List[Dict[str, Any]]

@dataclass
class DoseVolumeConstraint:
    """
    Ràng buộc liều-thể tích.
    
    Attributes
    ----------
    id : str
        ID duy nhất
    name : str
        Tên ràng buộc
    structure_id : str
        ID của cấu trúc
    structure_name : str
        Tên của cấu trúc
    constraint_type : ConstraintType
        Loại ràng buộc
    dose_value : float
        Giá trị liều
    volume_value : float
        Giá trị thể tích
    dose_unit : DoseUnit
        Đơn vị liều
    volume_unit : VolumeUnit
        Đơn vị thể tích
    direction : ConstraintDirection
        Hướng ràng buộc
    priority : PriorityLevel
        Mức độ ưu tiên
    is_met : bool
        Đã đạt được chưa
    weight : float
        Trọng số cho tối ưu hóa
    description : str
        Mô tả
    metadata : Dict[str, Any]
        Thông tin bổ sung
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    structure_id: str = ""
    structure_name: str = ""
    constraint_type: ConstraintType = ConstraintType.DOSE_VOLUME
    dose_value: float = 0.0
    volume_value: float = 0.0
    dose_unit: DoseUnit = DoseUnit.GY
    volume_unit: VolumeUnit = VolumeUnit.PERCENT
    direction: ConstraintDirection = ConstraintDirection.UPPER
    priority: PriorityLevel = PriorityLevel.MEDIUM
    is_met: bool = False
    weight: float = 1.0
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Xử lý sau khi khởi tạo."""
        # Tạo tên nếu trống
        if not self.name:
            self.name = self._generate_name()
            
        # Tạo mô tả nếu trống
        if not self.description:
            self.description = self._generate_description()
        
        # Chuyển đổi các enum từ chuỗi nếu cần
        if isinstance(self.constraint_type, str):
            try:
                self.constraint_type = ConstraintType(self.constraint_type)
            except ValueError:
                self.constraint_type = ConstraintType.DOSE_VOLUME
                
        if isinstance(self.dose_unit, str):
            try:
                self.dose_unit = DoseUnit(self.dose_unit)
            except ValueError:
                self.dose_unit = DoseUnit.GY
                
        if isinstance(self.volume_unit, str):
            try:
                self.volume_unit = VolumeUnit(self.volume_unit)
            except ValueError:
                self.volume_unit = VolumeUnit.PERCENT
                
        if isinstance(self.direction, str):
            try:
                self.direction = ConstraintDirection(self.direction)
            except ValueError:
                self.direction = ConstraintDirection.UPPER
                
        if isinstance(self.priority, str):
            try:
                self.priority = PriorityLevel[self.priority]
            except KeyError:
                self.priority = PriorityLevel.MEDIUM
    
    def _generate_name(self) -> str:
        """
        Tạo tên tự động.
        
        Returns
        -------
        str
            Tên dưới dạng văn bản
        """
        if self.constraint_type == ConstraintType.DOSE_VOLUME:
            return f"D{self.volume_value}{self.volume_unit.value} {self.structure_name}"
        elif self.constraint_type == ConstraintType.MAX_DOSE:
            return f"D_max {self.structure_name}"
        elif self.constraint_type == ConstraintType.MIN_DOSE:
            return f"D_min {self.structure_name}"
        elif self.constraint_type == ConstraintType.MEAN_DOSE:
            return f"D_mean {self.structure_name}"
        elif self.constraint_type == ConstraintType.CONFORMITY:
            return f"CI {self.structure_name}"
        elif self.constraint_type == ConstraintType.HOMOGENEITY:
            return f"HI {self.structure_name}"
        elif self.constraint_type == ConstraintType.GRADIENT:
            return f"GI {self.structure_name}"
        else:
            return f"Constraint {self.structure_name}"
    
    def _generate_description(self) -> str:
        """
        Tạo mô tả tự động.
        
        Returns
        -------
        str
            Mô tả dưới dạng văn bản
        """
        if self.constraint_type == ConstraintType.DOSE_VOLUME:
            if self.direction == ConstraintDirection.UPPER:
                return f"V{self.dose_value}{self.dose_unit.value} ≤ {self.volume_value}{self.volume_unit.value} cho {self.structure_name}"
            else:
                return f"V{self.dose_value}{self.dose_unit.value} ≥ {self.volume_value}{self.volume_unit.value} cho {self.structure_name}"
                
        elif self.constraint_type == ConstraintType.MAX_DOSE:
            return f"D_max ≤ {self.dose_value} {self.dose_unit.value} cho {self.structure_name}"
            
        elif self.constraint_type == ConstraintType.MIN_DOSE:
            return f"D_min ≥ {self.dose_value} {self.dose_unit.value} cho {self.structure_name}"
            
        elif self.constraint_type == ConstraintType.MEAN_DOSE:
            if self.direction == ConstraintDirection.UPPER:
                return f"D_mean ≤ {self.dose_value} {self.dose_unit.value} cho {self.structure_name}"
            else:
                return f"D_mean ≥ {self.dose_value} {self.dose_unit.value} cho {self.structure_name}"
                
        elif self.constraint_type == ConstraintType.CONFORMITY:
            return f"Chỉ số phù hợp (CI) cho {self.structure_name}"
            
        elif self.constraint_type == ConstraintType.HOMOGENEITY:
            return f"Chỉ số đồng nhất (HI) cho {self.structure_name}"
            
        elif self.constraint_type == ConstraintType.GRADIENT:
            return f"Chỉ số độ dốc (GI) cho {self.structure_name}"
            
        else:
            return f"Ràng buộc cho {self.structure_name}"
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Chuyển đổi thành dictionary.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary chứa dữ liệu của ràng buộc
        """
        return {
            "id": self.id,
            "name": self.name,
            "structure_id": self.structure_id,
            "structure_name": self.structure_name,
            "constraint_type": self.constraint_type.value,
            "dose_value": self.dose_value,
            "volume_value": self.volume_value,
            "dose_unit": self.dose_unit.value,
            "volume_unit": self.volume_unit.value,
            "direction": self.direction.value,
            "priority": self.priority.name,
            "is_met": self.is_met,
            "weight": self.weight,
            "description": self.description,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DoseVolumeConstraint':
        """
        Tạo từ dictionary.
        
        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary chứa dữ liệu
            
        Returns
        -------
        DoseVolumeConstraint
            Đối tượng mới
        """
        return cls(**data)
    
    def is_satisfied(self, value: float) -> bool:
        """
        Kiểm tra xem ràng buộc có thỏa mãn với giá trị đã cho hay không.
        
        Parameters
        ----------
        value : float
            Giá trị cần kiểm tra
            
        Returns
        -------
        bool
            True nếu ràng buộc được thỏa mãn, False nếu không
        """
        if self.direction == ConstraintDirection.UPPER:
            return value <= self.dose_value if self.constraint_type in [ConstraintType.MAX_DOSE, ConstraintType.MEAN_DOSE] else value <= self.volume_value
        elif self.direction == ConstraintDirection.LOWER:
            return value >= self.dose_value if self.constraint_type in [ConstraintType.MIN_DOSE, ConstraintType.MEAN_DOSE] else value >= self.volume_value
        elif self.direction == ConstraintDirection.EQUAL:
            return abs(value - (self.dose_value if self.constraint_type in [ConstraintType.MAX_DOSE, ConstraintType.MIN_DOSE, ConstraintType.MEAN_DOSE] else self.volume_value)) < 1e-6
        else:
            return False

class DoseConstraints:
    """
    Tập hợp các ràng buộc liều lượng cho một kế hoạch điều trị.
    """
    
    def __init__(self, constraints: List[DoseVolumeConstraint] = None):
        """
        Khởi tạo.
        
        Parameters
        ----------
        constraints : List[DoseVolumeConstraint], optional
            Danh sách các ràng buộc ban đầu
        """
        self.constraints: Dict[str, List[DoseVolumeConstraint]] = {}
        self.metadata: Dict[str, Any] = {}
        if constraints:
            for constraint in constraints:
                self.add_constraint(constraint)
    
    def add_constraint(self, constraint: DoseVolumeConstraint) -> None:
        """
        Thêm một ràng buộc.
        
        Parameters
        ----------
        constraint : DoseVolumeConstraint
            Ràng buộc cần thêm
        """
        if constraint.structure_name not in self.constraints:
            self.constraints[constraint.structure_name] = []
        self.constraints[constraint.structure_name].append(constraint)
    
    def add_constraints(self, constraints: List[DoseVolumeConstraint]) -> None:
        """
        Thêm nhiều ràng buộc.
        
        Parameters
        ----------
        constraints : List[DoseVolumeConstraint]
            Danh sách các ràng buộc cần thêm
        """
        for constraint in constraints:
            self.add_constraint(constraint)
    
    def get_constraint(self, constraint_id: str) -> Optional[DoseVolumeConstraint]:
        """
        Lấy một ràng buộc theo ID.
        
        Parameters
        ----------
        constraint_id : str
            ID của ràng buộc
            
        Returns
        -------
        Optional[DoseVolumeConstraint]
            Ràng buộc nếu tìm thấy, None nếu không
        """
        for structure_name, constraints in self.constraints.items():
            for constraint in constraints:
                if constraint.id == constraint_id:
                    return constraint
        return None
    
    def get_constraints_by_structure(self, structure_id: str) -> List[DoseVolumeConstraint]:
        """
        Lấy các ràng buộc cho một cấu trúc.
        
        Parameters
        ----------
        structure_id : str
            ID của cấu trúc
            
        Returns
        -------
        List[DoseVolumeConstraint]
            Danh sách các ràng buộc cho cấu trúc đó
        """
        return self.constraints.get(structure_id, [])
    
    def get_constraints_by_type(self, constraint_type: ConstraintType) -> List[DoseVolumeConstraint]:
        """
        Lấy các ràng buộc theo loại.
        
        Parameters
        ----------
        constraint_type : ConstraintType
            Loại ràng buộc
            
        Returns
        -------
        List[DoseVolumeConstraint]
            Danh sách các ràng buộc có loại đó
        """
        return [constraint for structure_name, constraints in self.constraints.items() for constraint in constraints if constraint.constraint_type == constraint_type]
    
    def get_constraints_by_priority(self, priority: PriorityLevel) -> List[DoseVolumeConstraint]:
        """
        Lấy các ràng buộc theo mức độ ưu tiên.
        
        Parameters
        ----------
        priority : PriorityLevel
            Mức độ ưu tiên
            
        Returns
        -------
        List[DoseVolumeConstraint]
            Danh sách các ràng buộc có mức độ ưu tiên đó
        """
        return [constraint for structure_name, constraints in self.constraints.items() for constraint in constraints if constraint.priority == priority]
    
    def remove_constraint(self, constraint_id: str) -> bool:
        """
        Xóa một ràng buộc.
        
        Parameters
        ----------
        constraint_id : str
            ID của ràng buộc cần xóa
            
        Returns
        -------
        bool
            True nếu xóa thành công, False nếu không tìm thấy
        """
        for structure_name, constraints in self.constraints.items():
            for i, constraint in enumerate(constraints):
                if constraint.id == constraint_id:
                    constraints.pop(i)
                    return True
        return False
    
    def evaluate_constraints(self, evaluation_data: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Đánh giá tất cả các ràng buộc dựa trên dữ liệu đánh giá đã cho.
        
        Parameters
        ----------
        evaluation_data : Dict[str, Dict[str, Any]]
            Dữ liệu đánh giá, dưới dạng {structure_name: {metric_name: value}}
            
        Returns
        -------
        Dict[str, Any]
            Kết quả đánh giá: {
                "constraints_met": int,  # Số lượng ràng buộc đã đạt được
                "total_constraints": int,  # Tổng số ràng buộc
                "percent_met": float,  # Phần trăm ràng buộc đã đạt được
                "constraints_status": List[ConstraintStatus],  # Thông tin chi tiết về từng ràng buộc
                "constraints_by_priority": Dict[str, int],  # Số lượng ràng buộc theo mức độ ưu tiên
                "constraints_met_by_priority": Dict[str, int],  # Số lượng ràng buộc đã đạt được theo mức độ ưu tiên
                "dvh_analysis": Dict[str, DVHAnalysis],  # Phân tích DVH cho từng cấu trúc
                "plan_metrics": Dict[str, float],  # Các chỉ số tổng thể của kế hoạch
                "warnings": List[str],  # Các cảnh báo về ràng buộc
                "recommendations": List[str]  # Các đề xuất cải thiện kế hoạch
            }
        """
        results: Dict[str, Any] = {
            "constraints_met": 0,
            "total_constraints": len(self.constraints),
            "percent_met": 0.0,
            "constraints_status": [],
            "constraints_by_priority": {p.name: 0 for p in PriorityLevel},
            "constraints_met_by_priority": {p.name: 0 for p in PriorityLevel},
            "dvh_analysis": {},
            "plan_metrics": {},
            "warnings": [],
            "recommendations": []
        }
        
        # Đánh giá từng ràng buộc
        for constraint in self.constraints:
            if not isinstance(constraint, (OptimizationDoseVolumeConstraint, MaxDoseConstraint, MinDoseConstraint, MeanDoseConstraint)):
                continue
            
            structure_name = constraint.structure_name
            if structure_name not in evaluation_data:
                results["warnings"].append(f"Không tìm thấy dữ liệu cho cấu trúc {structure_name}")
                continue
            
            structure_data = evaluation_data[structure_name]
            results["constraints_by_priority"][constraint.priority.name] += 1
            
            # Phân tích DVH
            dvh_analysis: Optional[DVHAnalysis] = None
            if "dvh" in structure_data:
                dvh_points: List[DVHPoint] = [
                    {"dose": float(p["dose"]), "volume": float(p["volume"])} 
                    for p in structure_data["dvh"]
                ]
                dvh = DVH(dvh_points)
                dvh_analysis = {
                    "dose_metrics": {
                        "D2": dvh.get_dx(2),
                        "D50": dvh.get_dx(50),
                        "D95": dvh.get_dx(95),
                        "D98": dvh.get_dx(98)
                    },
                    "volume_metrics": {
                        "V95": dvh.get_vx(95),
                        "V100": dvh.get_vx(100),
                        "V105": dvh.get_vx(105)
                    },
                    "gradient_metrics": {
                        "R50": dvh.get_r50(),
                        "R100": dvh.get_r100()
                    },
                    "conformity_metrics": {
                        "CI": dvh.get_conformity_index(),
                        "HI": dvh.get_homogeneity_index()
                    }
                }
                results["dvh_analysis"][structure_name] = dvh_analysis
            
            # Đánh giá ràng buộc
            if isinstance(constraint, OptimizationDoseVolumeConstraint):
                is_met, violation = constraint.evaluate(
                    dose_grid=DoseGridClass(structure_data["dose_grid"]),
                    structure_mask=structure_data["structure_mask"]
                )
                actual_value = constraint.dose
                difference = violation
            else:
                is_met, violation = constraint.evaluate(
                    dose_grid=DoseGridClass(structure_data["dose_grid"]),
                    structure_mask=structure_data["structure_mask"]
                )
                actual_value = constraint.dose_limit
                difference = violation
            
            if is_met:
                results["constraints_met"] += 1
                results["constraints_met_by_priority"][constraint.priority.name] += 1
            
            # Lưu kết quả đánh giá cho ràng buộc này
            constraint_status: ConstraintStatus = {
                "name": constraint.name,
                "priority": constraint.priority.name,
                "type": constraint.constraint_type,
                "value": float(actual_value),
                "is_met": is_met,
                "actual_value": float(actual_value),
                "difference": float(difference),
                "dvh_analysis": dvh_analysis
            }
            results["constraints_status"].append(constraint_status)
        
        # Tính phần trăm ràng buộc đã đạt được
        if results["total_constraints"] > 0:
            results["percent_met"] = (results["constraints_met"] / results["total_constraints"]) * 100
        
        return results
    
    def evaluate_against_plan(self, plan: Any) -> Dict[str, Any]:
        """
        Đánh giá các ràng buộc đối với một kế hoạch điều trị.
        
        Parameters
        ----------
        plan : Any
            Kế hoạch điều trị cần đánh giá
            
        Returns
        -------
        Dict[str, Any]
            Kết quả đánh giá như từ hàm evaluate_constraints
        """
        try:
            # Khởi tạo dictionary cho dữ liệu đánh giá
            evaluation_data: Dict[str, Dict[str, float]] = {}
            
            # Kiểm tra và xử lý plan dựa trên kiểu của nó
            try:
                # Động import để tránh circular imports
                from quangtps.evaluation.plan_evaluation import PlanEvaluation
                
                if isinstance(plan, PlanEvaluation):
                    # Nếu plan là PlanEvaluation
                    if hasattr(plan, 'get_metrics') and callable(getattr(plan, 'get_metrics')):
                        metrics_fn = getattr(plan, 'get_metrics')
                        evaluation_data = metrics_fn()
                else:
                    # Xử lý khi plan không phải là PlanEvaluation
                    if hasattr(plan, 'get_evaluation') and callable(getattr(plan, 'get_evaluation')):
                        # Lấy evaluation từ plan
                        get_eval_fn = getattr(plan, 'get_evaluation')
                        evaluation = get_eval_fn()
                        
                        if hasattr(evaluation, 'get_metrics') and callable(getattr(evaluation, 'get_metrics')):
                            metrics_fn = getattr(evaluation, 'get_metrics')
                            evaluation_data = metrics_fn()
                    else:
                        # Tạo PlanEvaluation từ plan
                        try:
                            # Use the correct parameter name for PlanEvaluation
                            evaluation = PlanEvaluation(plan)
                            
                            if hasattr(evaluation, 'get_metrics') and callable(getattr(evaluation, 'get_metrics')):
                                metrics_fn = getattr(evaluation, 'get_metrics')
                                evaluation_data = metrics_fn()
                        except Exception as e:
                            logging.warning(f"Could not create PlanEvaluation from plan: {e}")
            except ImportError:
                logging.warning("Could not import PlanEvaluation, skipping plan evaluation")
                
            # Đánh giá các ràng buộc với dữ liệu đã có
            return self.evaluate_constraints(evaluation_data)
            
        except Exception as e:
            logging.error(f"Error evaluating constraints against plan: {e}")
            # Trả về một dictionary trống trong trường hợp lỗi
            return {
                "constraints_met": 0,
                "total_constraints": 0,
                "percent_met": 0.0,
                "constraints_status": [],
                "constraints_by_priority": {},
                "constraints_met_by_priority": {},
                "dvh_analysis": {},
                "plan_metrics": {},
                "warnings": [],
                "recommendations": [],
                "error": str(e)
            }
    
    def export_to_clinical_protocol(self, file_path: str) -> None:
        """
        Xuất ràng buộc ra file protocol lâm sàng.
        
        Parameters
        ----------
        file_path : str
            Đường dẫn đến file protocol
        """
        try:
            # Tạo dữ liệu protocol
            protocol_data: ProtocolData = {
                "name": "Clinical Protocol",
                "version": "1.0",
                "constraints": [constraint.to_dict() for constraint in self.constraints]
            }
            
            # Đảm bảo thư mục tồn tại
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Lưu file
            with open(file_path, 'w') as f:
                json.dump(protocol_data, f, indent=4)
            
        except Exception as e:
            raise ValueError(f"Lỗi khi xuất protocol: {str(e)}")

    def generate_constraints_report(self, evaluation_result: Optional[Dict[str, Any]] = None, include_html: bool = False) -> Dict[str, Any]:
        """
        Tạo báo cáo về các ràng buộc liều, tùy chọn bao gồm kết quả đánh giá.
        
        Parameters
        ----------
        evaluation_result : Optional[Dict[str, Any]], optional
            Kết quả đánh giá từ hàm evaluate_constraints, by default None
        include_html : bool, optional
            Có tạo báo cáo HTML không, by default False
            
        Returns
        -------
        Dict[str, Any]
            Báo cáo về các ràng buộc
        """
        # Đếm tổng số ràng buộc
        total_constraints = len(self.constraints)
        
        # Thống kê theo mức độ ưu tiên
        constraints_by_priority: Dict[str, int] = {}
        for constraint in self.constraints:
            # Ensure constraint is properly typed for access to priority attribute
            constraint_obj = cast(DoseConstraintProtocol, constraint)
            priority_name = constraint_obj.priority.name
            if priority_name in constraints_by_priority:
                constraints_by_priority[priority_name] += 1
            else:
                constraints_by_priority[priority_name] = 1
        
        # Thống kê theo loại ràng buộc
        constraints_by_type: Dict[str, int] = {}
        for constraint in self.constraints:
            constraint_type = constraint.__class__.__name__
            if constraint_type in constraints_by_type:
                constraints_by_type[constraint_type] += 1
            else:
                constraints_by_type[constraint_type] = 1
        
        # Khởi tạo báo cáo
        report_data: Dict[str, Any] = {
            "total_constraints": total_constraints,
            "constraints_by_priority": constraints_by_priority,
            "constraints_by_type": constraints_by_type,
            "constraints": []
        }
        
        # Thêm thông tin chi tiết về từng ràng buộc
        for constraint in self.constraints:
            constraint_obj = cast(DoseConstraintProtocol, constraint)
            constraint_dict = constraint_obj.to_dict()
            # Ensure report_data["constraints"] is treated as a list
            if isinstance(report_data["constraints"], list):
                report_data["constraints"].append(constraint_dict)
        
        # Nếu có kết quả đánh giá, thêm vào báo cáo
        if evaluation_result:
            report_data.update(evaluation_result)
        
        # Tạo báo cáo HTML nếu được yêu cầu
        if include_html:
            report_data["html_report"] = self._generate_html_report(report_data)
        
        return report_data
        
    def import_from_clinical_protocol(self, file_path: str) -> None:
        """
        Nhập ràng buộc từ file protocol lâm sàng.
        
        Parameters
        ----------
        file_path : str
            Đường dẫn đến file protocol
        """
        try:
            with open(file_path, 'r') as f:
                protocol_data: ProtocolData = json.load(f)
                
            # Xóa các ràng buộc hiện tại
            self.constraints.clear()
            
            # Thêm các ràng buộc mới
            for constraint_data in protocol_data["constraints"]:
                constraint = self._create_constraint_from_dict(constraint_data)
                if constraint:
                    self.constraints.append(constraint)
                
        except FileNotFoundError:
            raise ValueError(f"Không tìm thấy file protocol: {file_path}")
        except json.JSONDecodeError:
            raise ValueError(f"File protocol không đúng định dạng JSON: {file_path}")
        except Exception as e:
            raise ValueError(f"Lỗi khi nhập protocol: {str(e)}")

    def _generate_html_report(self, report_data: Dict[str, Any]) -> str:
        """
        Tạo báo cáo HTML dựa trên dữ liệu báo cáo.
        
        Parameters
        ----------
        report_data : Dict[str, Any]
            Dữ liệu báo cáo
            
        Returns
        -------
        str
            Báo cáo HTML
        """
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1, h2 { color: #333; }
                table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                .passed { color: green; }
                .failed { color: red; }
                .warning { color: orange; }
                .info { background-color: #f8f9fa; padding: 10px; border-radius: 5px; margin-bottom: 15px; }
            </style>
            <title>Dose Constraints Report</title>
        </head>
        <body>
            <h1>Dose Constraints Report</h1>
        """
        
        # Add summary information
        html += f"""
            <div class="info">
                <p>Total Constraints: {report_data.get('total_constraints', 0)}</p>
        """
        
        if 'constraints_met' in report_data:
            html += f"""
                <p>Constraints Met: {report_data.get('constraints_met', 0)} / {report_data.get('total_constraints', 0)} 
                ({report_data.get('percent_met', 0):.1f}%)</p>
            """
            
        html += """
            </div>
        """
        
        # Add constraints by priority
        if 'constraints_by_priority' in report_data:
            html += """
                <h2>Constraints by Priority</h2>
                <table>
                    <tr>
                        <th>Priority</th>
                        <th>Count</th>
                    </tr>
            """
            
            for priority, count in report_data.get('constraints_by_priority', {}).items():
                html += f"""
                    <tr>
                        <td>{priority}</td>
                        <td>{count}</td>
                    </tr>
                """
                
            html += """
                </table>
            """
        
        # Add constraints by type
        if 'constraints_by_type' in report_data:
            html += """
                <h2>Constraints by Type</h2>
                <table>
                    <tr>
                        <th>Type</th>
                        <th>Count</th>
                    </tr>
            """
            
            for constraint_type, count in report_data.get('constraints_by_type', {}).items():
                html += f"""
                    <tr>
                        <td>{constraint_type}</td>
                        <td>{count}</td>
                    </tr>
                """
                
            html += """
                </table>
            """
        
        # Add detailed constraint status if available
        if 'constraints_status' in report_data:
            html += """
                <h2>Detailed Constraint Status</h2>
                <table>
                    <tr>
                        <th>Structure</th>
                        <th>Constraint</th>
                        <th>Priority</th>
                        <th>Status</th>
                        <th>Value</th>
                    </tr>
            """
            
            for constraint_status in report_data.get('constraints_status', []):
                if isinstance(constraint_status, dict):
                    status_class = ''
                    if constraint_status.get('met'):
                        status_class = 'passed'
                    else:
                        status_class = 'failed'
                    
                    html += f"""
                        <tr>
                            <td>{constraint_status.get('structure', '')}</td>
                            <td>{constraint_status.get('description', '')}</td>
                            <td>{constraint_status.get('priority', '')}</td>
                            <td class="{status_class}">{constraint_status.get('met', False) and 'Pass' or 'Fail'}</td>
                            <td>{constraint_status.get('actual_value', '')}</td>
                        </tr>
                    """
                
            html += """
                </table>
            """
        
        # Close HTML
        html += """
        </body>
        </html>
        """
        
        return html

    def _calculate_dx(self, dvh: 'DVH', x: float) -> float:
        """Tính giá trị Dx từ DVH."""
        return dvh.get_dx(x)

    def _calculate_vx(self, dvh: 'DVH', x: float) -> float:
        """Tính giá trị Vx từ DVH."""
        return dvh.get_vx(x)

    def _calculate_r50(self, dvh: 'DVH') -> float:
        """Tính chỉ số R50 từ DVH."""
        return dvh.get_r50()

    def _calculate_r100(self, dvh: 'DVH') -> float:
        """Tính chỉ số R100 từ DVH."""
        return dvh.get_r100()

    def _calculate_conformity_index(self, dvh: 'DVH') -> float:
        """Tính chỉ số phù hợp từ DVH."""
        return dvh.get_conformity_index()

    def _calculate_homogeneity_index(self, dvh: 'DVH') -> float:
        """Tính chỉ số đồng nhất từ DVH."""
        return dvh.get_homogeneity_index()

    def _calculate_priority_score(self, results: Dict[str, Any], priority: str) -> float:
        """Tính điểm cho một mức độ ưu tiên cụ thể."""
        total = results["constraints_by_priority"].get(priority, 0)
        met = results["constraints_met_by_priority"].get(priority, 0)
        return (met / total * 100.0) if total > 0 else 0.0

@dataclass
class DVH:
    def __init__(self, points: List[DVHPoint]):
        self.points = sorted(points, key=lambda x: x["dose"])
        self.doses = np.array([p["dose"] for p in self.points])
        self.volumes = np.array([p["volume"] for p in self.points])
    
    def get_dx(self, x: float) -> float:
        """Tính giá trị Dx từ DVH."""
        if x < 0 or x > 100:
            raise ValueError("x must be between 0 and 100")
        target_volume = 100 - x
        return np.interp(target_volume, self.volumes, self.doses)
    
    def get_vx(self, x: float) -> float:
        """Tính giá trị Vx từ DVH."""
        return np.interp(x, self.doses, self.volumes)
    
    def get_r50(self) -> float:
        """Tính chỉ số R50 từ DVH."""
        v50 = self.get_vx(50)
        v100 = self.get_vx(100)
        return v50 / v100 if v100 > 0 else 0.0
    
    def get_r100(self) -> float:
        """Tính chỉ số R100 từ DVH."""
        v100 = self.get_vx(100)
        v50 = self.get_vx(50)
        return v100 / v50 if v50 > 0 else 0.0
    
    def get_conformity_index(self) -> float:
        """Tính chỉ số phù hợp từ DVH."""
        v100 = self.get_vx(100)
        v50 = self.get_vx(50)
        return v100 / v50 if v50 > 0 else 0.0
    
    def get_homogeneity_index(self) -> float:
        """Tính chỉ số đồng nhất từ DVH."""
        d2 = self.get_dx(2)
        d98 = self.get_dx(98)
        return (d2 - d98) / d98 if d98 > 0 else 0.0