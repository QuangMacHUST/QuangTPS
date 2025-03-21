#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module cầu nối cho TreatmentPlan.

Module này cung cấp các lớp và phương thức để quản lý kế hoạch điều trị,
phù hợp trong bối cảnh module treatment. Module này chủ yếu làm cầu nối
giữa module planning và module treatment.
"""

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional, Type, Union

# Use TYPE_CHECKING to avoid circular imports
if TYPE_CHECKING:
    from quangtps.planning.plan import Plan, PlanType

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

# Define treatment-specific functions that use Plan
def create_treatment_plan(name: str, patient_id: str, **kwargs) -> 'Plan':
    """Create a treatment plan instance"""
    Plan = get_plan_class()
    return Plan(plan_name=name, patient_id=patient_id, **kwargs)

def load_treatment_plan(plan_data: Dict[str, Any]) -> 'Plan':
    """Load a treatment plan from dictionary data"""
    Plan = get_plan_class()
    return Plan.from_dict(plan_data)
