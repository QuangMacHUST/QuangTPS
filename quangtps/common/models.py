#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Common models cho QuangTPS.

Module này cung cấp các alias và imports chung cho các model được sử dụng
xuyên suốt hệ thống.
"""

import logging

# Import các models chính từ core
try:
    from quangtps.core.patient.patient import Patient, PatientMetadata
    from quangtps.core.patient.study import Study, StudyMetadata
    from quangtps.core.patient.series import (
        Series,
        SeriesMetadata,
        Modality,
        SeriesStatus,
    )
    from quangtps.core.patient.image import Image, ImageModality

    # Thử import StudyType riêng biệt vì có thể không tồn tại
    try:
        from quangtps.core.patient.study import StudyType
    except ImportError:
        # Tạo StudyType enum nếu không có
        from enum import Enum

        class StudyType(str, Enum):
            STANDARD = "STANDARD"
            RESEARCH = "RESEARCH"
            UNKNOWN = "UNKNOWN"

    # Đánh dấu rằng imports thành công
    MODELS_AVAILABLE = True

except ImportError as e:
    logging.warning(f"Không thể import một số models: {e}")
    MODELS_AVAILABLE = False

    # Tạo dummy classes để tránh lỗi import
    from enum import Enum

    class Patient:
        def __init__(self, *args, **kwargs):
            pass

    class PatientMetadata:
        def __init__(self, *args, **kwargs):
            pass

    class Study:
        def __init__(self, *args, **kwargs):
            pass

    class StudyMetadata:
        def __init__(self, *args, **kwargs):
            pass

    class StudyType(str, Enum):
        STANDARD = "STANDARD"
        RESEARCH = "RESEARCH"
        UNKNOWN = "UNKNOWN"

    class Series:
        def __init__(self, *args, **kwargs):
            pass

    class SeriesMetadata:
        def __init__(self, *args, **kwargs):
            pass

    class Modality(str, Enum):
        CT = "CT"
        MR = "MR"
        UNKNOWN = "UNKNOWN"

    class SeriesStatus(str, Enum):
        ACTIVE = "ACTIVE"
        ARCHIVED = "ARCHIVED"

    class Image:
        def __init__(self, *args, **kwargs):
            pass

    class ImageModality(str, Enum):
        CT = "CT"
        MR = "MR"
        UNKNOWN = "UNKNOWN"


# Import các model liên quan đến structure và contour
try:
    from quangtps.segmentation.contour.contour_data import ContourData, ContourSet

    CONTOUR_MODELS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Không thể import contour models: {e}")
    CONTOUR_MODELS_AVAILABLE = False

    class ContourData:
        def __init__(self, *args, **kwargs):
            pass

    class ContourSet:
        def __init__(self, *args, **kwargs):
            pass


# Import các model liên quan đến dose
try:
    from quangtps.dose.dose_grid import DoseGrid

    DOSE_MODELS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Không thể import dose models: {e}")
    DOSE_MODELS_AVAILABLE = False

    class DoseGrid:
        def __init__(self, *args, **kwargs):
            pass


# Import các model liên quan đến beam và plan
try:
    from quangtps.planning.beam import Beam
    from quangtps.treatment.beams.beam_data import BeamData

    BEAM_MODELS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Không thể import beam models: {e}")
    BEAM_MODELS_AVAILABLE = False

    class Beam:
        def __init__(self, *args, **kwargs):
            pass

    class BeamData:
        def __init__(self, *args, **kwargs):
            pass


# Aliases phổ biến
PatientModel = Patient
StudyModel = Study
SeriesModel = Series
ImageModel = Image
ContourModel = ContourData
DoseModel = DoseGrid
BeamModel = Beam

# Đảm bảo tất cả models được export
__all__ = [
    "Patient",
    "PatientMetadata",
    "Study",
    "StudyMetadata",
    "StudyType",
    "Series",
    "SeriesMetadata",
    "Modality",
    "SeriesStatus",
    "Image",
    "ImageModality",
    "ContourData",
    "ContourSet",
    "DoseGrid",
    "Beam",
    "BeamData",
    # Aliases
    "PatientModel",
    "StudyModel",
    "SeriesModel",
    "ImageModel",
    "ContourModel",
    "DoseModel",
    "BeamModel",
    # Availability flags
    "MODELS_AVAILABLE",
    "CONTOUR_MODELS_AVAILABLE",
    "DOSE_MODELS_AVAILABLE",
    "BEAM_MODELS_AVAILABLE",
]

# Log trạng thái import
logger = logging.getLogger(__name__)
logger.info(
    f"Common models initialized - Core: {MODELS_AVAILABLE}, "
    f"Contour: {CONTOUR_MODELS_AVAILABLE}, "
    f"Dose: {DOSE_MODELS_AVAILABLE}, "
    f"Beam: {BEAM_MODELS_AVAILABLE}"
)
