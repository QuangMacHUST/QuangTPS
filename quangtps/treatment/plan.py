#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module cầu nối cho TreatmentPlan.

Module này cung cấp các lớp và phương thức để quản lý kế hoạch điều trị,
phù hợp trong bối cảnh module treatment. Module này chủ yếu làm cầu nối
giữa module planning và module treatment.
"""

import logging
from quangtps.planning.plan import Plan, PlanType

logger = logging.getLogger(__name__)

# Alias for Plan class to be used in treatment modules
TreatmentPlan = Plan
TreatmentPlanType = PlanType
