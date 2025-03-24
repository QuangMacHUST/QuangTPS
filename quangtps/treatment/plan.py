#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module cầu nối cho TreatmentPlan.

Module này cung cấp các lớp và phương thức để quản lý kế hoạch điều trị,
phù hợp trong bối cảnh module treatment. Module này chủ yếu làm cầu nối
giữa module planning và module treatment.
"""

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional, Type, Union, List
from datetime import datetime

# Use TYPE_CHECKING to avoid circular imports
if TYPE_CHECKING:
    from quangtps.planning.plan import Plan, PlanType, PlanStatus

logger = logging.getLogger(__name__)

# We will do dynamic imports when actually needed
def get_plan_class() -> 'Type[Plan]':
    """Get the Plan class from planning module dynamically"""
    from quangtps.planning.plan import Plan
    return Plan

def get_plan_type_enum() -> 'Type[PlanType]':
    """Get the PlanType enum from planning module dynamically"""
    from quangtps.planning.plan import PlanType
    return PlanType

def get_plan_status_enum() -> 'Type[PlanStatus]':
    """Get the PlanStatus enum from planning module dynamically"""
    from quangtps.planning.plan import PlanStatus
    return PlanStatus

# Define treatment-specific functions that use Plan
def create_treatment_plan(name: str, patient_id: str, **kwargs) -> 'Plan':
    """Create a treatment plan instance"""
    Plan = get_plan_class()
    return Plan(plan_name=name, patient_id=patient_id, **kwargs)

def load_treatment_plan(plan_data: Dict[str, Any]) -> 'Plan':
    """Load a treatment plan from dictionary data"""
    Plan = get_plan_class()
    return Plan.from_dict(plan_data)

class TreatmentPlan:
    """
    Lớp đại diện cho kế hoạch điều trị trong module treatment.
    
    Lớp này bao bọc lớp Plan từ module planning và bổ sung thêm
    các chức năng liên quan đến điều trị.
    """
    
    def __init__(self, plan=None, plan_id=None):
        """
        Khởi tạo đối tượng TreatmentPlan.
        
        Parameters
        ----------
        plan : Plan, optional
            Đối tượng Plan từ module planning, by default None
        plan_id : str, optional
            ID của kế hoạch nếu cần tải từ cơ sở dữ liệu, by default None
        """
        from quangtps.planning.plan import Plan, PlanStatus
        
        self._plan = None
        
        if plan and isinstance(plan, Plan):
            self._plan = plan
        elif plan_id:
            # Tải từ cơ sở dữ liệu nếu cần
            from quangtps.database.plan_db import PlanDB
            db = PlanDB()
            plan_data = db.get_plan(plan_id)
            if plan_data:
                self._plan = Plan.from_dict(plan_data)
        
        # Nếu không có kế hoạch, tạo một kế hoạch mới
        if not self._plan:
            self._plan = Plan(
                plan_name="Unnamed Treatment Plan",
                patient_id="",
                status=PlanStatus.DRAFT
            )
        
        # Thông tin bổ sung cho điều trị
        self.delivery_history = []
        self.active_delivery = None
        self.total_fractions = 0
        self.completed_fractions = 0
        self.fractionation_scheme = None
        self.treatment_machine = None
    
    @property
    def plan_id(self) -> str:
        """ID của kế hoạch."""
        return self._plan.plan_id
    
    @property
    def name(self) -> str:
        """Tên kế hoạch."""
        return self._plan.name
    
    @property
    def patient_id(self) -> str:
        """ID của bệnh nhân."""
        return self._plan.patient_id
    
    @property
    def beams(self) -> List:
        """Danh sách các chùm tia."""
        return self._plan.beams
    
    @property
    def status(self):
        """Trạng thái kế hoạch."""
        return self._plan.status
    
    @property
    def plan_type(self):
        """Loại kế hoạch."""
        return self._plan.plan_type
    
    @property
    def base_plan(self):
        """Trả về đối tượng Plan gốc."""
        return self._plan
    
    def set_fractionation_scheme(self, scheme):
        """
        Thiết lập phân đoạn điều trị.
        
        Parameters
        ----------
        scheme : FractionationScheme
            Đối tượng phân đoạn điều trị
        """
        self.fractionation_scheme = scheme
        self.total_fractions = scheme.total_fractions if scheme else 0
    
    def set_treatment_machine(self, machine):
        """
        Thiết lập máy điều trị.
        
        Parameters
        ----------
        machine : TreatmentMachine
            Đối tượng máy điều trị
        """
        self.treatment_machine = machine
    
    def record_fraction_completion(self, fraction_data):
        """
        Ghi nhận hoàn thành một phân đoạn điều trị.
        
        Parameters
        ----------
        fraction_data : dict
            Dữ liệu về phân đoạn đã hoàn thành
            
        Returns
        -------
        bool
            True nếu ghi nhận thành công
        """
        if not fraction_data.get('fraction_number'):
            logger.error("Missing fraction number in fraction completion data")
            return False
        
        # Thêm vào lịch sử và tăng số phân đoạn đã hoàn thành
        self.delivery_history.append({
            'fraction_number': fraction_data.get('fraction_number'),
            'delivery_time': fraction_data.get('delivery_time', datetime.now()),
            'delivery_data': fraction_data,
        })
        
        self.completed_fractions = max(self.completed_fractions, fraction_data.get('fraction_number'))
        
        # Cập nhật trạng thái nếu đã hoàn thành tất cả phân đoạn
        if self.completed_fractions >= self.total_fractions:
            from quangtps.planning.plan import PlanStatus
            self._plan.set_status(PlanStatus.COMPLETED)
        
        return True
    
    def get_remaining_fractions(self):
        """
        Lấy số phân đoạn còn lại.
        
        Returns
        -------
        int
            Số phân đoạn còn lại cần thực hiện
        """
        return max(0, self.total_fractions - self.completed_fractions)
    
    def to_dict(self):
        """
        Chuyển đổi thành dictionary.
        
        Returns
        -------
        dict
            Dictionary chứa dữ liệu của kế hoạch điều trị
        """
        data = self._plan.to_dict()
        
        # Bổ sung thông tin điều trị
        data.update({
            'treatment_info': {
                'completed_fractions': self.completed_fractions,
                'total_fractions': self.total_fractions,
                'delivery_history': self.delivery_history,
                'treatment_machine': self.treatment_machine.machine_id if self.treatment_machine else None,
                'fractionation_scheme': self.fractionation_scheme.to_dict() if self.fractionation_scheme else None
            }
        })
        
        return data
    
    @classmethod
    def from_dict(cls, data):
        """
        Tạo đối tượng TreatmentPlan từ dictionary.
        
        Parameters
        ----------
        data : dict
            Dictionary chứa dữ liệu
            
        Returns
        -------
        TreatmentPlan
            Đối tượng kế hoạch điều trị
        """
        from quangtps.planning.plan import Plan
        
        # Tạo đối tượng Plan cơ bản
        plan_data = {k: v for k, v in data.items() if k != 'treatment_info'}
        plan = Plan.from_dict(plan_data)
        
        # Tạo đối tượng TreatmentPlan
        treatment_plan = cls(plan=plan)
        
        # Bổ sung thông tin điều trị
        treatment_info = data.get('treatment_info', {})
        treatment_plan.completed_fractions = treatment_info.get('completed_fractions', 0)
        treatment_plan.total_fractions = treatment_info.get('total_fractions', 0)
        treatment_plan.delivery_history = treatment_info.get('delivery_history', [])
        
        # Tải máy điều trị nếu có
        if treatment_info.get('treatment_machine'):
            from quangtps.treatment.machine.treatment_machine import TreatmentMachine
            machine_id = treatment_info.get('treatment_machine')
            treatment_plan.treatment_machine = TreatmentMachine.load(machine_id)
        
        # Tải phân đoạn điều trị nếu có
        if treatment_info.get('fractionation_scheme'):
            from quangtps.treatment.fractionation import FractionationScheme
            scheme_data = treatment_info.get('fractionation_scheme')
            treatment_plan.fractionation_scheme = FractionationScheme.from_dict(scheme_data)
        
        return treatment_plan
