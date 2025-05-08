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
]
