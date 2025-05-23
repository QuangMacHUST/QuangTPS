#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Patient Data Module

Module này quản lý dữ liệu bệnh nhân cho QuangTPS,
bao gồm thông tin cá nhân, hình ảnh y tế, và các cấu trúc.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import date, datetime
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Import với error handling
try:
    from quangtps.core.patient.patient import Patient
    from quangtps.structures.structure_set import StructureSet
    from quangtps.imaging.image import Image
except ImportError as e:
    logger.warning(f"Không thể import một số module: {e}")

    # Tạo classes giả
    class Patient:
        def __init__(self, *args, **kwargs):
            self.id = "unknown"
            self.name = "Unknown Patient"

    class StructureSet:
        def __init__(self, *args, **kwargs):
            self.structures = []

    class Image:
        def __init__(self, *args, **kwargs):
            self.data = np.zeros((64, 64, 32))


@dataclass
class PatientData:
    """Dữ liệu bệnh nhân toàn diện."""

    # Thông tin bệnh nhân
    patient: Optional[Patient] = None

    # Hình ảnh y tế
    ct_images: List[Image] = field(default_factory=list)
    mr_images: List[Image] = field(default_factory=list)
    pet_images: List[Image] = field(default_factory=list)

    # Cấu trúc
    structure_sets: List[StructureSet] = field(default_factory=list)
    primary_structure_set: Optional[StructureSet] = None

    # Metadata
    study_date: Optional[datetime] = None
    modality: str = "CT"
    institution: str = "Unknown"

    def __post_init__(self):
        """Khởi tạo sau khi tạo instance."""
        if self.study_date is None:
            self.study_date = datetime.now()

    def get_primary_image(self) -> Optional[Image]:
        """Lấy hình ảnh chính (thường là CT)."""
        if self.ct_images:
            return self.ct_images[0]
        elif self.mr_images:
            return self.mr_images[0]
        elif self.pet_images:
            return self.pet_images[0]
        return None

    def get_image_spacing(self) -> Tuple[float, float, float]:
        """Lấy spacing của hình ảnh chính."""
        primary_image = self.get_primary_image()
        if primary_image and hasattr(primary_image, "spacing"):
            return primary_image.spacing
        return (1.0, 1.0, 1.0)  # Default spacing

    def get_image_shape(self) -> Tuple[int, int, int]:
        """Lấy shape của hình ảnh chính."""
        primary_image = self.get_primary_image()
        if primary_image and hasattr(primary_image, "data"):
            return primary_image.data.shape
        return (64, 64, 32)  # Default shape


# Factory function để tương thích
def create_patient_data(**kwargs) -> PatientData:
    """Tạo PatientData instance."""
    return PatientData(**kwargs)
