#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module quản lý chùm tia xạ trị (Radiation Beam).

Module này cung cấp các lớp và phương thức để định nghĩa và quản lý
các chùm tia xạ trị được sử dụng trong kế hoạch điều trị.
"""

import uuid
import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from enum import Enum

from quangtps.treatment.beams.beam_geometry import BeamGeometry
from quangtps.treatment.beams.beam_modifiers import BeamModifier

logger = logging.getLogger(__name__)


class BeamType(str, Enum):
    """Enum đại diện cho các loại chùm tia."""

    PHOTON = "PHOTON"
    ELECTRON = "ELECTRON"
    PROTON = "PROTON"
    CARBON = "CARBON"
    NEUTRON = "NEUTRON"


class BeamStatus(str, Enum):
    """Enum đại diện cho trạng thái của chùm tia."""

    PLANNED = "PLANNED"
    APPROVED = "APPROVED"
    TREATED = "TREATED"
    CANCELLED = "CANCELLED"


class BeamEnergy:
    """Class đại diện cho năng lượng chùm tia."""

    def __init__(self, value: float = 6.0, unit: str = "MV"):
        """
        Khởi tạo BeamEnergy.

        Parameters
        ----------
        value : float
            Giá trị năng lượng
        unit : str
            Đơn vị (MV, MeV, etc.)
        """
        self.value = value
        self.unit = unit

    def __str__(self) -> str:
        return f"{self.value}{self.unit}"

    def __repr__(self) -> str:
        return f"BeamEnergy({self.value}, '{self.unit}')"


class Beam:
    """
    Lớp đại diện cho một chùm tia xạ trị.

    Lớp này chứa thông tin về một chùm tia xạ trị, bao gồm các thông số vật lý,
    hình học và các thông số điều trị khác.
    """

    def __init__(self, beam_name: str, beam_id: Optional[str] = None):
        """
        Khởi tạo một chùm tia xạ trị.

        Parameters
        ----------
        beam_name : str
            Tên của chùm tia
        beam_id : str, optional
            ID duy nhất của chùm tia. Nếu không cung cấp, một ID mới sẽ được tạo.
        """
        self.beam_name = beam_name
        self.beam_id = beam_id if beam_id else str(uuid.uuid4())

        # Các thông số vật lý
        self.beam_type = BeamType.PHOTON
        self.energy = 6.0  # MV hoặc MeV
        self.dose_rate = 600.0  # MU/phút
        self.monitor_units = 100.0  # MU

        # Thông tin hình học
        self.geometry = BeamGeometry()

        # Các thiết bị điều biến (modifiers)
        self.modifiers = []

        # Trọng số và thông số khác
        self.weight = 1.0
        self.status = BeamStatus.PLANNED
        self.description = ""

        # Thông tin bổ sung
        self.metadata = {}

    def set_energy(self, energy: float):
        """
        Thiết lập năng lượng cho chùm tia.

        Parameters
        ----------
        energy : float
            Năng lượng của chùm tia (MV hoặc MeV)
        """
        self.energy = energy

    def set_dose_rate(self, dose_rate: float):
        """
        Thiết lập tốc độ liều cho chùm tia.

        Parameters
        ----------
        dose_rate : float
            Tốc độ liều (MU/phút)
        """
        self.dose_rate = dose_rate

    def set_monitor_units(self, mu: float):
        """
        Thiết lập đơn vị monitor (MU) cho chùm tia.

        Parameters
        ----------
        mu : float
            Đơn vị monitor (MU)
        """
        self.monitor_units = mu

    def set_beam_type(self, beam_type: BeamType):
        """
        Thiết lập loại chùm tia.

        Parameters
        ----------
        beam_type : BeamType
            Loại chùm tia
        """
        self.beam_type = beam_type

    def add_modifier(self, modifier: BeamModifier):
        """
        Thêm một thiết bị điều biến cho chùm tia.

        Parameters
        ----------
        modifier : BeamModifier
            Thiết bị điều biến
        """
        self.modifiers.append(modifier)

    def remove_modifier(self, modifier_id: str) -> bool:
        """
        Xóa một thiết bị điều biến khỏi chùm tia.

        Parameters
        ----------
        modifier_id : str
            ID của thiết bị điều biến cần xóa

        Returns
        -------
        bool
            True nếu xóa thành công, False nếu không tìm thấy
        """
        for i, modifier in enumerate(self.modifiers):
            if modifier.modifier_id == modifier_id:
                self.modifiers.pop(i)
                return True
        return False

    def get_modifier(self, modifier_id: str) -> Optional[BeamModifier]:
        """
        Lấy thông tin về một thiết bị điều biến.

        Parameters
        ----------
        modifier_id : str
            ID của thiết bị điều biến

        Returns
        -------
        Optional[BeamModifier]
            Thiết bị điều biến nếu tìm thấy, None nếu không tìm thấy
        """
        for modifier in self.modifiers:
            if modifier.modifier_id == modifier_id:
                return modifier
        return None

    def set_status(self, status: BeamStatus):
        """
        Thiết lập trạng thái cho chùm tia.

        Parameters
        ----------
        status : BeamStatus
            Trạng thái của chùm tia
        """
        self.status = status

    def set_weight(self, weight: float):
        """
        Thiết lập trọng số cho chùm tia.

        Parameters
        ----------
        weight : float
            Trọng số của chùm tia
        """
        self.weight = weight

    def set_description(self, description: str):
        """
        Thiết lập mô tả cho chùm tia.

        Parameters
        ----------
        description : str
            Mô tả của chùm tia
        """
        self.description = description

    def add_metadata(self, key: str, value: Any):
        """
        Thêm thông tin metadata cho chùm tia.

        Parameters
        ----------
        key : str
            Khóa metadata
        value : Any
            Giá trị metadata
        """
        self.metadata[key] = value

    def to_dict(self) -> Dict[str, Any]:
        """
        Chuyển đổi thông tin chùm tia thành dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary chứa thông tin chùm tia
        """
        return {
            "beam_name": self.beam_name,
            "beam_id": self.beam_id,
            "beam_type": self.beam_type.value,
            "energy": self.energy,
            "dose_rate": self.dose_rate,
            "monitor_units": self.monitor_units,
            "geometry": self.geometry.to_dict(),
            "modifiers": [modifier.to_dict() for modifier in self.modifiers],
            "weight": self.weight,
            "status": self.status.value,
            "description": self.description,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Beam":
        """
        Tạo đối tượng Beam từ dictionary.

        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary chứa thông tin chùm tia

        Returns
        -------
        Beam
            Đối tượng Beam
        """
        beam = cls(beam_name=data["beam_name"], beam_id=data["beam_id"])

        # Cập nhật các thuộc tính
        beam.beam_type = BeamType(data["beam_type"])
        beam.energy = data["energy"]
        beam.dose_rate = data["dose_rate"]
        beam.monitor_units = data["monitor_units"]

        # Cập nhật hình học
        from quangtps.treatment.beams.beam_geometry import BeamGeometry

        beam.geometry = BeamGeometry.from_dict(data["geometry"])

        # Cập nhật các thiết bị điều biến
        from quangtps.treatment.beams.beam_modifiers import (
            BeamModifier,
            Wedge,
            Block,
            Bolus,
            Compensator,
        )

        beam.modifiers = []
        for modifier_data in data["modifiers"]:
            if modifier_data["type"] == "WEDGE":
                beam.modifiers.append(Wedge.from_dict(modifier_data))
            elif modifier_data["type"] == "BLOCK":
                beam.modifiers.append(Block.from_dict(modifier_data))
            elif modifier_data["type"] == "BOLUS":
                beam.modifiers.append(Bolus.from_dict(modifier_data))
            elif modifier_data["type"] == "COMPENSATOR":
                beam.modifiers.append(Compensator.from_dict(modifier_data))

        beam.weight = data["weight"]
        beam.status = BeamStatus(data["status"])
        beam.description = data["description"]
        beam.metadata = data["metadata"]

        return beam


class DoseSpecificationPoint:
    """
    Lớp đại diện cho điểm xác định liều cho chùm tia.

    Điểm xác định liều (Dose Specification Point) là điểm mà tại đó
    liều được tính toán và chỉ định, thường là điểm isocenter hoặc
    một điểm cụ thể trong thể tích mục tiêu.
    """

    def __init__(
        self,
        position: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        name: str = "Isocenter",
        dose_value: float = 0.0,
        point_id: Optional[str] = None,
    ):
        """
        Khởi tạo một điểm xác định liều.

        Parameters
        ----------
        position : Tuple[float, float, float], optional
            Tọa độ của điểm trong không gian 3D (mm)
        name : str, optional
            Tên của điểm
        dose_value : float, optional
            Giá trị liều tại điểm (Gy)
        point_id : str, optional
            ID duy nhất của điểm
        """
        self.position = position
        self.name = name
        self.dose_value = dose_value
        self.point_id = point_id if point_id else str(uuid.uuid4())
        self.metadata = {}

    def set_position(self, x: float, y: float, z: float):
        """
        Thiết lập vị trí cho điểm xác định liều.

        Parameters
        ----------
        x : float
            Tọa độ x (mm)
        y : float
            Tọa độ y (mm)
        z : float
            Tọa độ z (mm)
        """
        self.position = (x, y, z)

    def set_dose_value(self, dose_value: float):
        """
        Thiết lập giá trị liều cho điểm xác định liều.

        Parameters
        ----------
        dose_value : float
            Giá trị liều (Gy)
        """
        self.dose_value = dose_value

    def to_dict(self) -> Dict[str, Any]:
        """
        Chuyển đổi thông tin điểm xác định liều thành dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary chứa thông tin điểm xác định liều
        """
        return {
            "position": self.position,
            "name": self.name,
            "dose_value": self.dose_value,
            "point_id": self.point_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DoseSpecificationPoint":
        """
        Tạo đối tượng DoseSpecificationPoint từ dictionary.

        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary chứa thông tin điểm xác định liều

        Returns
        -------
        DoseSpecificationPoint
            Đối tượng DoseSpecificationPoint
        """
        point = cls(
            position=data["position"],
            name=data["name"],
            dose_value=data["dose_value"],
            point_id=data["point_id"],
        )
        point.metadata = data.get("metadata", {})
        return point
