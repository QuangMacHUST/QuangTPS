#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module quản lý chùm tia trong quá trình lập kế hoạch xạ trị.

Module này chứa các lớp và hàm để quản lý các chùm tia, bao gồm
thông số vật lý, hình học và các thành phần liên quan khác.
"""

from quangtps.treatment.beams.beam import Beam, BeamType
from quangtps.treatment.beams.beam_modifiers import Wedge, Compensator, Block
from quangtps.treatment.beams.beam_geometry import (
    BeamGeometry,
    GantryDirection,
    CollimatorDirection,
    CouchDirection,
)
from quangtps.treatment.beams.beam_library import BeamLibrary, BeamTemplate
from quangtps.treatment.beams.beam_sequence_generator import BeamSequenceGenerator
from quangtps.treatment.beams.beam_data_importer import TrueBeamDataReader, BeamDataType

# Import BeamSet class
try:
    from quangtps.treatment.beams.beam_set import BeamSet
except ImportError:
    # Create BeamSet class if not available
    from typing import List, Dict, Any, Optional
    import uuid

    class BeamSet:
        """
        Tập hợp các chùm tia trong một kế hoạch điều trị.

        BeamSet chứa một tập hợp các chùm tia được sử dụng trong cùng một
        phiên điều trị hoặc phân đoạn của kế hoạch điều trị.
        """

        def __init__(self, name: str = "BeamSet", beam_set_id: Optional[str] = None):
            """
            Khởi tạo BeamSet.

            Parameters
            ----------
            name : str
                Tên của beam set
            beam_set_id : str, optional
                ID duy nhất của beam set
            """
            self.name = name
            self.beam_set_id = beam_set_id if beam_set_id else str(uuid.uuid4())
            self.beams: List[Beam] = []
            self.description = ""
            self.prescription_dose = 0.0  # Gy
            self.number_of_fractions = 1
            self.metadata = {}

        def add_beam(self, beam: Beam) -> None:
            """Thêm chùm tia vào beam set."""
            if beam not in self.beams:
                self.beams.append(beam)

        def remove_beam(self, beam_id: str) -> bool:
            """Xóa chùm tia khỏi beam set."""
            for i, beam in enumerate(self.beams):
                if beam.beam_id == beam_id:
                    self.beams.pop(i)
                    return True
            return False

        def get_beam(self, beam_id: str) -> Optional[Beam]:
            """Lấy chùm tia theo ID."""
            for beam in self.beams:
                if beam.beam_id == beam_id:
                    return beam
            return None

        def get_total_monitor_units(self) -> float:
            """Tính tổng monitor units của tất cả chùm tia."""
            return sum(beam.monitor_units for beam in self.beams)

        def set_prescription(self, dose: float, fractions: int) -> None:
            """Thiết lập đơn thuốc."""
            self.prescription_dose = dose
            self.number_of_fractions = fractions

        def to_dict(self) -> Dict[str, Any]:
            """Chuyển đổi BeamSet thành dictionary."""
            return {
                "name": self.name,
                "beam_set_id": self.beam_set_id,
                "beams": [beam.to_dict() for beam in self.beams],
                "description": self.description,
                "prescription_dose": self.prescription_dose,
                "number_of_fractions": self.number_of_fractions,
                "metadata": self.metadata,
            }


__all__ = [
    "Beam",
    "BeamType",
    "BeamSet",
    "Wedge",
    "Compensator",
    "Block",
    "BeamGeometry",
    "GantryDirection",
    "CollimatorDirection",
    "CouchDirection",
    "BeamLibrary",
    "BeamTemplate",
    "BeamSequenceGenerator",
    "TrueBeamDataReader",
    "BeamDataType",
]
