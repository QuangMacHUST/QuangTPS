#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QuangTPS Eclipse Scripting API (ESAPI)

Module này cung cấp API cho phép tương tác với QuangTPS theo cách tương tự như
Eclipse Scripting API (ESAPI) của Varian, giúp tự động hóa quy trình lâm sàng,
truy vấn dữ liệu và mở rộng chức năng của hệ thống lập kế hoạch xạ trị.
"""

import logging
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Union, Set

__version__ = "0.1.0"
__author__ = "QuangTPS Team"

logger = logging.getLogger(__name__)

# Import các module API
from quangtps.scripts.esapi.application import Application
from quangtps.scripts.esapi.patient import Patient, PatientSummary
from quangtps.scripts.esapi.course import Course
from quangtps.scripts.esapi.plan import PlanSetup, PlanSum
from quangtps.scripts.esapi.beam import Beam, BeamParameters
from quangtps.scripts.esapi.structure import Structure, StructureSet
from quangtps.scripts.esapi.dose import Dose, DVH, DoseValue, DoseValuePresentation
from quangtps.scripts.esapi.optimization import OptimizationObjective

# Xuất các lớp và hàm chính cho người dùng
__all__ = [
    "Application",
    "Patient",
    "PatientSummary",
    "Course",
    "PlanSetup",
    "PlanSum",
    "Beam",
    "BeamParameters",
    "Structure",
    "StructureSet",
    "Dose",
    "DVH",
    "DoseValue",
    "DoseValuePresentation",
    "OptimizationObjective",
    "get_current_application",
]

# Biến ứng dụng singleton
_current_application = None


def get_current_application() -> "Application":
    """
    Lấy đối tượng Application hiện tại.

    Returns
    -------
    Application
        Đối tượng Application đại diện cho phiên QuangTPS hiện tại.
    """
    global _current_application
    if _current_application is None:
        from quangtps.scripts.esapi.application import Application

        _current_application = Application()
    return _current_application
