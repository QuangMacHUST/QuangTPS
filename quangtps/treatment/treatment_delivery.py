#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module quản lý việc thực hiện điều trị (Treatment Delivery).

Module này cung cấp các lớp và phương thức để quản lý quá trình thực hiện điều trị xạ trị,
bao gồm lập lịch điều trị, theo dõi tiến trình, và quản lý các phiên điều trị.
"""

import logging
import datetime
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple, Union
import uuid
import calendar
import math

from quangtps.treatment.fractionation import Fractionation
from quangtps.treatment.plan import TreatmentPlan
from quangtps.core.patient import Patient

logger = logging.getLogger(__name__)


class TreatmentStatus(str, Enum):
    """Enum đại diện cho trạng thái của điều trị."""
    SCHEDULED = "Scheduled"      # Đã lên lịch
    IN_PROGRESS = "In Progress"  # Đang thực hiện
    COMPLETED = "Completed"      # Đã hoàn thành
    INTERRUPTED = "Interrupted"  # Bị gián đoạn
    CANCELLED = "Cancelled"      # Đã hủy
    ON_HOLD = "On Hold"          # Tạm hoãn


class FractionStatus(str, Enum):
    """Enum đại diện cho trạng thái của một phân đoạn."""
    SCHEDULED = "Scheduled"      # Đã lên lịch
    DELIVERED = "Delivered"      # Đã thực hiện
    MISSED = "Missed"            # Đã bỏ lỡ
    CANCELLED = "Cancelled"      # Đã hủy
    MODIFIED = "Modified"        # Đã thay đổi


class TreatmentFraction:
    """
    Lớp đại diện cho một phân đoạn điều trị.
    
    Lớp này chứa thông tin về một phân đoạn điều trị cụ thể, bao gồm
    ngày điều trị, liều đã thực hiện, và trạng thái của phân đoạn.
    """
    
    def __init__(
        self,
        fraction_number: int,
        scheduled_date: Optional[datetime.date] = None,
        prescribed_dose: float = 0.0
    ):
        """
        Khởi tạo một phân đoạn điều trị.
        
        Parameters
        ----------
        fraction_number : int
            Số thứ tự của phân đoạn
        scheduled_date : datetime.date, optional
            Ngày lên lịch cho phân đoạn
        prescribed_dose : float, optional
            Liều kê đơn cho phân đoạn (Gy)
        """
        self.fraction_id = str(uuid.uuid4())
        self.fraction_number = fraction_number
        self.scheduled_date = scheduled_date
        self.actual_date = None
        self.prescribed_dose = prescribed_dose
        self.delivered_dose = 0.0
        self.status = FractionStatus.SCHEDULED
        
        # Thông tin bổ sung
        self.notes = ""
        self.operator = ""
        self.machine_id = ""
        self.setup_deviations = {"x": 0.0, "y": 0.0, "z": 0.0, "pitch": 0.0, "roll": 0.0, "yaw": 0.0}
        self.imaging_performed = []
        self.qa_results = {}
        self.metadata = {}
    
    def deliver(
        self,
        date: datetime.date,
        delivered_dose: float,
        operator: str,
        machine_id: str,
        setup_deviations: Optional[Dict[str, float]] = None,
        imaging_performed: Optional[List[str]] = None,
        qa_results: Optional[Dict[str, Any]] = None
    ):
        """
        Đánh dấu phân đoạn đã được thực hiện.
        
        Parameters
        ----------
        date : datetime.date
            Ngày thực hiện phân đoạn
        delivered_dose : float
            Liều đã thực hiện (Gy)
        operator : str
            Người thực hiện phân đoạn
        machine_id : str
            ID của máy điều trị
        setup_deviations : Dict[str, float], optional
            Độ lệch thiết lập (mm và độ)
        imaging_performed : List[str], optional
            Các hình ảnh đã thực hiện
        qa_results : Dict[str, Any], optional
            Kết quả QA
        """
        self.actual_date = date
        self.delivered_dose = delivered_dose
        self.operator = operator
        self.machine_id = machine_id
        self.status = FractionStatus.DELIVERED
        
        if setup_deviations:
            self.setup_deviations.update(setup_deviations)
        
        if imaging_performed:
            self.imaging_performed = imaging_performed
        
        if qa_results:
            self.qa_results = qa_results
    
    def cancel(self, reason: str):
        """
        Hủy phân đoạn điều trị.
        
        Parameters
        ----------
        reason : str
            Lý do hủy
        """
        self.status = FractionStatus.CANCELLED
        self.notes += f"Cancelled: {reason}"
    
    def miss(self, reason: str):
        """
        Đánh dấu phân đoạn đã bị bỏ lỡ.
        
        Parameters
        ----------
        reason : str
            Lý do bỏ lỡ
        """
        self.status = FractionStatus.MISSED
        self.notes += f"Missed: {reason}"
    
    def modify(self, new_dose: float, reason: str):
        """
        Thay đổi liều kê đơn cho phân đoạn.
        
        Parameters
        ----------
        new_dose : float
            Liều mới (Gy)
        reason : str
            Lý do thay đổi
        """
        self.prescribed_dose = new_dose
        self.status = FractionStatus.MODIFIED
        self.notes += f"Modified: {reason}. New dose: {new_dose}Gy"
    
    def reschedule(self, new_date: datetime.date, reason: str):
        """
        Lên lịch lại cho phân đoạn.
        
        Parameters
        ----------
        new_date : datetime.date
            Ngày mới
        reason : str
            Lý do lên lịch lại
        """
        self.scheduled_date = new_date
        self.notes += f"Rescheduled: {reason}. New date: {new_date}"
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Chuyển đổi thông tin phân đoạn thành dictionary.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary chứa thông tin phân đoạn
        """
        return {
            "fraction_id": self.fraction_id,
            "fraction_number": self.fraction_number,
            "scheduled_date": self.scheduled_date.isoformat() if self.scheduled_date else None,
            "actual_date": self.actual_date.isoformat() if self.actual_date else None,
            "prescribed_dose": self.prescribed_dose,
            "delivered_dose": self.delivered_dose,
            "status": self.status.value,
            "notes": self.notes,
            "operator": self.operator,
            "machine_id": self.machine_id,
            "setup_deviations": self.setup_deviations,
            "imaging_performed": self.imaging_performed,
            "qa_results": self.qa_results,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TreatmentFraction':
        """
        Tạo đối tượng TreatmentFraction từ dictionary.
        
        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary chứa thông tin phân đoạn
            
        Returns
        -------
        TreatmentFraction
            Đối tượng TreatmentFraction
        """
        fraction = cls(
            fraction_number=data["fraction_number"],
            scheduled_date=datetime.date.fromisoformat(data["scheduled_date"]) if data["scheduled_date"] else None,
            prescribed_dose=data["prescribed_dose"]
        )
        
        fraction.fraction_id = data["fraction_id"]
        fraction.actual_date = datetime.date.fromisoformat(data["actual_date"]) if data["actual_date"] else None
        fraction.delivered_dose = data["delivered_dose"]
        fraction.status = FractionStatus(data["status"])
        fraction.notes = data["notes"]
        fraction.operator = data["operator"]
        fraction.machine_id = data["machine_id"]
        fraction.setup_deviations = data["setup_deviations"]
        fraction.imaging_performed = data["imaging_performed"]
        fraction.qa_results = data["qa_results"]
        fraction.metadata = data["metadata"]
        
        return fraction


class TreatmentCourse:
    """
    Lớp đại diện cho một đợt điều trị.
    
    Lớp này chứa thông tin về một đợt điều trị hoàn chỉnh, bao gồm kế hoạch điều trị,
    các phân đoạn, và trạng thái của đợt điều trị.
    """
    
    def __init__(
        self,
        patient_id: str,
        plan_id: str,
        course_name: str,
        fractionation: Fractionation
    ):
        """
        Khởi tạo một đợt điều trị.
        
        Parameters
        ----------
        patient_id : str
            ID của bệnh nhân
        plan_id : str
            ID của kế hoạch điều trị
        course_name : str
            Tên của đợt điều trị
        fractionation : Fractionation
            Thông tin phân đoạn
        """
        self.course_id = str(uuid.uuid4())
        self.patient_id = patient_id
        self.plan_id = plan_id
        self.course_name = course_name
        self.fractionation = fractionation
        self.status = TreatmentStatus.SCHEDULED
        
        # Thời gian điều trị
        self.start_date = None
        self.end_date = None
        self.expected_completion_date = None
        
        # Danh sách phân đoạn
        self.fractions = []
        
        # Thông tin bổ sung
        self.description = ""
        self.notes = ""
        self.referring_physician = ""
        self.radiation_oncologist = ""
        self.created_date = datetime.datetime.now()
        self.last_modified = datetime.datetime.now()
        self.created_by = ""
        self.metadata = {}
    
    def generate_fractions(self, start_date: datetime.date, days_of_week: Optional[List[int]] = None):
        """
        Tạo các phân đoạn điều trị dựa trên thông tin phân đoạn và lịch điều trị.
        
        Parameters
        ----------
        start_date : datetime.date
            Ngày bắt đầu điều trị
        days_of_week : List[int], optional
            Các ngày trong tuần để điều trị (0=Monday, 6=Sunday).
            Mặc định là [0, 1, 2, 3, 4] (Thứ 2 đến Thứ 6).
        """
        if days_of_week is None:
            days_of_week = [0, 1, 2, 3, 4]  # Thứ 2 đến Thứ 6
        
        self.start_date = start_date
        self.fractions = []
        
        num_fractions = self.fractionation.num_fractions
        dose_per_fraction = self.fractionation.dose_per_fraction
        
        current_date = start_date
        fraction_number = 1
        
        while fraction_number <= num_fractions:
            # Kiểm tra xem ngày hiện tại có phải là ngày điều trị không
            if current_date.weekday() in days_of_week:
                # Tạo phân đoạn mới
                fraction = TreatmentFraction(
                    fraction_number=fraction_number,
                    scheduled_date=current_date,
                    prescribed_dose=dose_per_fraction
                )
                
                self.fractions.append(fraction)
                fraction_number += 1
            
            current_date += datetime.timedelta(days=1)
        
        self.expected_completion_date = current_date - datetime.timedelta(days=1)
        self.last_modified = datetime.datetime.now()
    
    def update_fraction_status(
        self,
        fraction_number: int,
        status: FractionStatus,
        **kwargs
    ):
        """
        Cập nhật trạng thái của một phân đoạn.
        
        Parameters
        ----------
        fraction_number : int
            Số thứ tự của phân đoạn
        status : FractionStatus
            Trạng thái mới
        **kwargs : dict
            Các thông số bổ sung
        """
        for fraction in self.fractions:
            if fraction.fraction_number == fraction_number:
                if status == FractionStatus.DELIVERED:
                    fraction.deliver(
                        date=kwargs.get("date", datetime.date.today()),
                        delivered_dose=kwargs.get("delivered_dose", fraction.prescribed_dose),
                        operator=kwargs.get("operator", ""),
                        machine_id=kwargs.get("machine_id", ""),
                        setup_deviations=kwargs.get("setup_deviations"),
                        imaging_performed=kwargs.get("imaging_performed"),
                        qa_results=kwargs.get("qa_results")
                    )
                elif status == FractionStatus.CANCELLED:
                    fraction.cancel(reason=kwargs.get("reason", "No reason provided"))
                elif status == FractionStatus.MISSED:
                    fraction.miss(reason=kwargs.get("reason", "No reason provided"))
                elif status == FractionStatus.MODIFIED:
                    fraction.modify(
                        new_dose=kwargs.get("new_dose", fraction.prescribed_dose),
                        reason=kwargs.get("reason", "No reason provided")
                    )
                
                self.last_modified = datetime.datetime.now()
                self._update_course_status()
                return True
        
        logger.warning(f"Phân đoạn {fraction_number} không tồn tại trong đợt điều trị {self.course_id}")
        return False
    
    def _update_course_status(self):
        """Cập nhật trạng thái của đợt điều trị dựa trên trạng thái của các phân đoạn."""
        # Kiểm tra nếu tất cả các phân đoạn đã hoàn thành
        all_completed = all(
            f.status == FractionStatus.DELIVERED for f in self.fractions
        )
        
        if all_completed:
            self.status = TreatmentStatus.COMPLETED
            self.end_date = max(f.actual_date for f in self.fractions if f.actual_date)
            return
        
        # Kiểm tra nếu tất cả các phân đoạn đã bị hủy
        all_cancelled = all(
            f.status == FractionStatus.CANCELLED for f in self.fractions
        )
        
        if all_cancelled:
            self.status = TreatmentStatus.CANCELLED
            return
        
        # Kiểm tra xem có phân đoạn nào đã được thực hiện không
        any_delivered = any(
            f.status == FractionStatus.DELIVERED for f in self.fractions
        )
        
        if any_delivered:
            self.status = TreatmentStatus.IN_PROGRESS
            return
        
        # Mặc định là SCHEDULED
        self.status = TreatmentStatus.SCHEDULED
    
    def get_delivered_dose(self) -> float:
        """
        Tính tổng liều đã thực hiện.
        
        Returns
        -------
        float
            Tổng liều đã thực hiện (Gy)
        """
        return sum(f.delivered_dose for f in self.fractions if f.status == FractionStatus.DELIVERED)
    
    def get_remaining_dose(self) -> float:
        """
        Tính liều còn lại cần thực hiện.
        
        Returns
        -------
        float
            Liều còn lại (Gy)
        """
        return self.fractionation.total_dose - self.get_delivered_dose()
    
    def get_progress(self) -> float:
        """
        Tính tiến độ điều trị.
        
        Returns
        -------
        float
            Tiến độ điều trị (0-1)
        """
        total_dose = self.fractionation.total_dose
        if total_dose == 0:
            return 0.0
            
        return self.get_delivered_dose() / total_dose
    
    def get_completed_fractions(self) -> int:
        """
        Đếm số phân đoạn đã hoàn thành.
        
        Returns
        -------
        int
            Số phân đoạn đã hoàn thành
        """
        return sum(1 for f in self.fractions if f.status == FractionStatus.DELIVERED)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Chuyển đổi thông tin đợt điều trị thành dictionary.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary chứa thông tin đợt điều trị
        """
        return {
            "course_id": self.course_id,
            "patient_id": self.patient_id,
            "plan_id": self.plan_id,
            "course_name": self.course_name,
            "fractionation": self.fractionation.to_dict(),
            "status": self.status.value,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "expected_completion_date": self.expected_completion_date.isoformat() if self.expected_completion_date else None,
            "fractions": [f.to_dict() for f in self.fractions],
            "description": self.description,
            "notes": self.notes,
            "referring_physician": self.referring_physician,
            "radiation_oncologist": self.radiation_oncologist,
            "created_date": self.created_date.isoformat(),
            "last_modified": self.last_modified.isoformat(),
            "created_by": self.created_by,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TreatmentCourse':
        """
        Tạo đối tượng TreatmentCourse từ dictionary.
        
        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary chứa thông tin đợt điều trị
            
        Returns
        -------
        TreatmentCourse
            Đối tượng TreatmentCourse
        """
        from quangtps.treatment.fractionation import Fractionation
        
        fractionation = Fractionation.from_dict(data["fractionation"])
        
        course = cls(
            patient_id=data["patient_id"],
            plan_id=data["plan_id"],
            course_name=data["course_name"],
            fractionation=fractionation
        )
        
        course.course_id = data["course_id"]
        course.status = TreatmentStatus(data["status"])
        
        if data["start_date"]:
            course.start_date = datetime.date.fromisoformat(data["start_date"])
        
        if data["end_date"]:
            course.end_date = datetime.date.fromisoformat(data["end_date"])
        
        if data["expected_completion_date"]:
            course.expected_completion_date = datetime.date.fromisoformat(data["expected_completion_date"])
        
        course.fractions = [TreatmentFraction.from_dict(f) for f in data["fractions"]]
        course.description = data["description"]
        course.notes = data["notes"]
        course.referring_physician = data["referring_physician"]
        course.radiation_oncologist = data["radiation_oncologist"]
        course.created_date = datetime.datetime.fromisoformat(data["created_date"])
        course.last_modified = datetime.datetime.fromisoformat(data["last_modified"])
        course.created_by = data["created_by"]
        course.metadata = data["metadata"]
        
        return course
