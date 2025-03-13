#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module quản lý kế hoạch điều trị (Plan).

Module này cung cấp các lớp và phương thức để quản lý kế hoạch điều trị,
bao gồm các thông tin về bệnh nhân, loại kế hoạch, và các thông số điều trị.
"""

import uuid
import logging
import datetime
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Tuple

from quangtps.treatment.beams.beam import Beam
from quangtps.planning.beam import BeamArrangement
from quangtps.planning.optimization import OptimizationSettings
from quangtps.planning.evaluation import PlanEvaluation
from quangtps.planning.prescription import Prescription

logger = logging.getLogger(__name__)


class PlanType(str, Enum):
    """Enum cho các loại kế hoạch điều trị."""
    DEFINITIVE = "Definitive"           # Điều trị triệt căn
    PALLIATIVE = "Palliative"           # Điều trị triệu chứng
    ADJUVANT = "Adjuvant"               # Điều trị bổ trợ
    NEOADJUVANT = "Neoadjuvant"         # Điều trị tân bổ trợ
    SALVAGE = "Salvage"                 # Điều trị cứu vãn
    BOOST = "Boost"                     # Điều trị tăng cường
    CUSTOM = "Custom"                   # Tùy chỉnh


class PlanStatus(str, Enum):
    """Enum cho các trạng thái của kế hoạch điều trị."""
    DRAFT = "Draft"                     # Bản nháp
    PLANNING = "Planning"               # Đang lập kế hoạch
    OPTIMIZATION = "Optimization"       # Đang tối ưu hóa
    CALCULATION = "Calculation"         # Đang tính toán
    REVIEW = "Review"                   # Đang xem xét
    APPROVED = "Approved"               # Đã phê duyệt
    DELIVERED = "Delivered"             # Đã thực hiện
    COMPLETED = "Completed"             # Đã hoàn thành
    ARCHIVED = "Archived"               # Đã lưu trữ
    DISCARDED = "Discarded"             # Đã hủy bỏ


class Plan:
    """
    Lớp đại diện cho một kế hoạch điều trị.
    
    Lớp này chứa thông tin về một kế hoạch điều trị bao gồm thông tin bệnh nhân,
    loại kế hoạch, trạng thái, và các thông số liên quan đến kế hoạch điều trị.
    """
    
    def __init__(
        self, 
        plan_name: str, 
        patient_id: str,
        plan_id: Optional[str] = None,
        plan_type: PlanType = PlanType.DEFINITIVE,
        status: PlanStatus = PlanStatus.DRAFT
    ):
        """
        Khởi tạo một kế hoạch điều trị.
        
        Parameters
        ----------
        plan_name : str
            Tên của kế hoạch điều trị
        patient_id : str
            ID của bệnh nhân
        plan_id : str, optional
            ID duy nhất của kế hoạch điều trị. Nếu không cung cấp, một ID mới sẽ được tạo.
        plan_type : PlanType, optional
            Loại kế hoạch điều trị
        status : PlanStatus, optional
            Trạng thái của kế hoạch điều trị
        """
        self.plan_name = plan_name
        self.patient_id = patient_id
        self.plan_id = plan_id if plan_id else str(uuid.uuid4())
        self.plan_type = plan_type
        self.status = status
        
        # Thông tin cơ bản
        self.description = ""
        self.notes = ""
        self.created_date = datetime.datetime.now()
        self.last_modified = datetime.datetime.now()
        self.created_by = ""
        self.approved_by = ""
        self.approval_date = None
        
        # Thông tin lâm sàng
        self.diagnosis = ""
        self.diagnosis_code = ""
        self.site = ""
        self.laterality = ""  # Left, Right, Bilateral, N/A
        
        # Kỹ thuật và thiết bị
        self.technique = ""  # 3DCRT, IMRT, VMAT, SRS, SBRT, etc.
        self.machine_id = ""
        self.energy = ""  # 6MV, 10MV, Mixed, etc.
        
        # Các thành phần của kế hoạch
        self.beam_arrangement = None
        self.prescription = None
        self.optimization_settings = None
        self.evaluation = None
        
        # Dữ liệu tính toán
        self.dose_grid = None
        self.structures = {}
        self.isodose_levels = [95, 90, 80, 70, 60, 50, 40, 30, 20, 10]
        
        # Trạng thái tính toán
        self.calculation_complete = False
        self.calculation_time = None
        self.calculation_status = ""
        
    def set_plan_type(self, plan_type: PlanType):
        """
        Đặt loại kế hoạch điều trị.
        
        Parameters
        ----------
        plan_type : PlanType
            Loại kế hoạch điều trị
        """
        self.plan_type = plan_type
        self.last_modified = datetime.datetime.now()
    
    def set_status(self, status: PlanStatus):
        """
        Đặt trạng thái của kế hoạch điều trị.
        
        Parameters
        ----------
        status : PlanStatus
            Trạng thái của kế hoạch điều trị
        """
        self.status = status
        self.last_modified = datetime.datetime.now()
        
        # Cập nhật các trường liên quan nếu trạng thái là Approved
        if status == PlanStatus.APPROVED:
            self.approval_date = datetime.datetime.now()
    
    def set_beam_arrangement(self, beam_arrangement: BeamArrangement):
        """
        Đặt sắp xếp chùm tia cho kế hoạch.
        
        Parameters
        ----------
        beam_arrangement : BeamArrangement
            Đối tượng sắp xếp chùm tia
        """
        self.beam_arrangement = beam_arrangement
        self.last_modified = datetime.datetime.now()
    
    def set_prescription(self, prescription: Prescription):
        """
        Đặt đơn điều trị cho kế hoạch.
        
        Parameters
        ----------
        prescription : Prescription
            Đối tượng đơn điều trị
        """
        self.prescription = prescription
        self.last_modified = datetime.datetime.now()
    
    def set_optimization_settings(self, settings: OptimizationSettings):
        """
        Đặt thiết lập tối ưu hóa cho kế hoạch.
        
        Parameters
        ----------
        settings : OptimizationSettings
            Đối tượng thiết lập tối ưu hóa
        """
        self.optimization_settings = settings
        self.last_modified = datetime.datetime.now()
    
    def calculate_dose(self) -> bool:
        """
        Tính toán phân bố liều cho kế hoạch.
        
        Returns
        -------
        bool
            True nếu tính toán thành công, False nếu không
        """
        if not self.beam_arrangement:
            logger.error("Không thể tính toán liều - sắp xếp chùm tia chưa được thiết lập")
            return False
        
        if not self.prescription:
            logger.error("Không thể tính toán liều - đơn điều trị chưa được thiết lập")
            return False
        
        start_time = datetime.datetime.now()
        
        try:
            # Giả lập quá trình tính toán liều
            # Trong thực tế, sẽ gọi đến engine tính toán liều
            self.calculation_status = "Calculating..."
            self.status = PlanStatus.CALCULATION
            
            # Giả lập: tạo ra một grid liều giả
            # Thực tế: sẽ tính toán liều dựa trên các thông số vật lý
            
            # Đánh dấu tính toán hoàn thành
            self.calculation_complete = True
            self.calculation_time = (datetime.datetime.now() - start_time).total_seconds()
            self.calculation_status = "Complete"
            self.status = PlanStatus.REVIEW
            
            return True
        except Exception as e:
            logger.error(f"Lỗi khi tính toán liều: {str(e)}")
            self.calculation_status = f"Error: {str(e)}"
            return False
    
    def evaluate(self) -> Optional[PlanEvaluation]:
        """
        Đánh giá kế hoạch điều trị.
        
        Returns
        -------
        Optional[PlanEvaluation]
            Đối tượng đánh giá kế hoạch nếu thành công, None nếu không
        """
        if not self.calculation_complete:
            logger.error("Không thể đánh giá kế hoạch - tính toán liều chưa hoàn thành")
            return None
        
        try:
            # Tạo đối tượng đánh giá kế hoạch
            self.evaluation = PlanEvaluation(
                dose_grid=self.dose_grid,
                structures=self.structures,
                prescription=self.prescription
            )
            
            # Tính toán DVH và các chỉ số chất lượng
            self.evaluation.calculate_dvh()
            self.evaluation.calculate_quality_metrics()
            
            return self.evaluation
        except Exception as e:
            logger.error(f"Lỗi khi đánh giá kế hoạch: {str(e)}")
            return None
    
    def approve(self, approver: str) -> bool:
        """
        Phê duyệt kế hoạch điều trị.
        
        Parameters
        ----------
        approver : str
            Tên người phê duyệt
            
        Returns
        -------
        bool
            True nếu phê duyệt thành công, False nếu không
        """
        if self.status != PlanStatus.REVIEW:
            logger.error(f"Không thể phê duyệt kế hoạch - trạng thái hiện tại là {self.status}")
            return False
        
        if not self.calculation_complete:
            logger.error("Không thể phê duyệt kế hoạch - tính toán liều chưa hoàn thành")
            return False
        
        if not self.evaluation:
            logger.error("Không thể phê duyệt kế hoạch - đánh giá kế hoạch chưa được thực hiện")
            return False
        
        # Cập nhật trạng thái và thông tin phê duyệt
        self.status = PlanStatus.APPROVED
        self.approved_by = approver
        self.approval_date = datetime.datetime.now()
        self.last_modified = datetime.datetime.now()
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Chuyển đổi đối tượng kế hoạch điều trị thành dictionary.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary chứa thông tin kế hoạch điều trị
        """
        result = {
            "plan_id": self.plan_id,
            "plan_name": self.plan_name,
            "patient_id": self.patient_id,
            "plan_type": self.plan_type.value,
            "status": self.status.value,
            "description": self.description,
            "notes": self.notes,
            "created_date": self.created_date.isoformat(),
            "last_modified": self.last_modified.isoformat(),
            "created_by": self.created_by,
            "approved_by": self.approved_by,
            "approval_date": self.approval_date.isoformat() if self.approval_date else None,
            "diagnosis": self.diagnosis,
            "diagnosis_code": self.diagnosis_code,
            "site": self.site,
            "laterality": self.laterality,
            "technique": self.technique,
            "machine_id": self.machine_id,
            "energy": self.energy,
            "isodose_levels": self.isodose_levels,
            "calculation_complete": self.calculation_complete,
            "calculation_time": self.calculation_time,
            "calculation_status": self.calculation_status
        }
        
        # Thêm các thành phần phức tạp nếu có
        if self.beam_arrangement:
            result["beam_arrangement"] = self.beam_arrangement.to_dict()
        
        if self.prescription:
            result["prescription"] = self.prescription.to_dict()
        
        if self.optimization_settings:
            result["optimization_settings"] = self.optimization_settings.to_dict()
        
        if self.evaluation:
            result["evaluation"] = self.evaluation.to_dict()
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Plan':
        """
        Tạo đối tượng Plan từ dictionary.
        
        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary chứa thông tin kế hoạch điều trị
            
        Returns
        -------
        Plan
            Đối tượng Plan được tạo từ dữ liệu
        """
        # Tạo đối tượng Plan cơ bản
        plan = cls(
            plan_name=data.get("plan_name", ""),
            patient_id=data.get("patient_id", ""),
            plan_id=data.get("plan_id"),
            plan_type=PlanType(data.get("plan_type", PlanType.DEFINITIVE.value)),
            status=PlanStatus(data.get("status", PlanStatus.DRAFT.value))
        )
        
        # Cập nhật các thông tin cơ bản
        plan.description = data.get("description", "")
        plan.notes = data.get("notes", "")
        plan.created_by = data.get("created_by", "")
        plan.approved_by = data.get("approved_by", "")
        plan.diagnosis = data.get("diagnosis", "")
        plan.diagnosis_code = data.get("diagnosis_code", "")
        plan.site = data.get("site", "")
        plan.laterality = data.get("laterality", "")
        plan.technique = data.get("technique", "")
        plan.machine_id = data.get("machine_id", "")
        plan.energy = data.get("energy", "")
        plan.isodose_levels = data.get("isodose_levels", [95, 90, 80, 70, 60, 50, 40, 30, 20, 10])
        plan.calculation_complete = data.get("calculation_complete", False)
        plan.calculation_time = data.get("calculation_time")
        plan.calculation_status = data.get("calculation_status", "")
        
        # Chuyển đổi các trường datetime
        if "created_date" in data:
            plan.created_date = datetime.datetime.fromisoformat(data["created_date"])
        
        if "last_modified" in data:
            plan.last_modified = datetime.datetime.fromisoformat(data["last_modified"])
        
        if "approval_date" in data and data["approval_date"]:
            plan.approval_date = datetime.datetime.fromisoformat(data["approval_date"])
        
        # Tái tạo các thành phần phức tạp nếu có
        if "beam_arrangement" in data:
            from quangtps.planning.beam import BeamArrangement
            plan.beam_arrangement = BeamArrangement.from_dict(data["beam_arrangement"])
        
        if "prescription" in data:
            from quangtps.planning.prescription import Prescription
            plan.prescription = Prescription.from_dict(data["prescription"])
        
        if "optimization_settings" in data:
            from quangtps.planning.optimization import OptimizationSettings
            plan.optimization_settings = OptimizationSettings.from_dict(data["optimization_settings"])
        
        if "evaluation" in data:
            from quangtps.planning.evaluation import PlanEvaluation
            plan.evaluation = PlanEvaluation.from_dict(data["evaluation"])
        
        return plan