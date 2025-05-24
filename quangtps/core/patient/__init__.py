#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module chứa các lớp cơ bản cho quản lý bệnh nhân trong QuangTPS.

Module này cung cấp các lớp để quản lý dữ liệu bệnh nhân, hình ảnh, cấu trúc,
và các thành phần cơ bản khác trong hệ thống lập kế hoạch xạ trị.
"""

from quangtps.core.patient.patient import Patient, PatientMetadata
from quangtps.core.patient.image import (
    Image,
    ImageModality,
    ImageOrientation,
    create_empty_image,
)
from quangtps.core.patient.study import Study, StudyMetadata
from quangtps.core.patient.series import Series, SeriesMetadata

# Import Plan từ core module
try:
    from quangtps.core.plan import Plan
except ImportError:
    # Fallback Plan class
    class Plan:
        def __init__(self, id, name, patient_id):
            self.id = id
            self.name = name
            self.patient_id = patient_id
            self.beams = []
            self.structures = {}


# Import approval-related classes - định nghĩa fallback để tránh circular import
from enum import Enum, auto


class ApprovalStatus(Enum):
    DRAFT = auto()
    PENDING = auto()
    APPROVED = auto()
    REJECTED = auto()
    DELIVERED = auto()
    ARCHIVED = auto()


class ApprovalAction(Enum):
    CREATE = auto()
    MODIFY = auto()
    SUBMIT = auto()
    APPROVE = auto()
    REJECT = auto()
    ARCHIVE = auto()
    RESTORE = auto()


class TreatmentPlan:
    def __init__(self, name, patient=None):
        self.name = name
        self.patient = patient
        self.approval_status = ApprovalStatus.DRAFT


__all__ = [
    "Patient",
    "PatientMetadata",
    "Image",
    "ImageModality",
    "ImageOrientation",
    "create_empty_image",
    "Study",
    "StudyMetadata",
    "Series",
    "SeriesMetadata",
    "Plan",
    "ApprovalStatus",
    "ApprovalAction",
    "TreatmentPlan",
]
