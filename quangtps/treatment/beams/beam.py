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


class BeamParameters:
    """
    Lớp đại diện cho các tham số của chùm tia xạ trị.

    Lớp này chứa các thông số vật lý và kỹ thuật của chùm tia
    được sử dụng trong tính toán liều và tối ưu hóa kế hoạch.
    """

    def __init__(
        self,
        energy: float = 6.0,
        dose_rate: float = 600.0,
        monitor_units: float = 100.0,
        gantry_angle: float = 0.0,
        collimator_angle: float = 0.0,
        couch_angle: float = 0.0,
        isocenter: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        field_size_x: float = 10.0,
        field_size_y: float = 10.0,
        beam_type: BeamType = BeamType.PHOTON,
    ):
        """
        Khởi tạo BeamParameters.

        Parameters
        ----------
        energy : float, optional
            Năng lượng chùm tia (MV hoặc MeV), mặc định 6.0
        dose_rate : float, optional
            Tốc độ liều (MU/phút), mặc định 600.0
        monitor_units : float, optional
            Đơn vị monitor (MU), mặc định 100.0
        gantry_angle : float, optional
            Góc gantry (độ), mặc định 0.0
        collimator_angle : float, optional
            Góc collimator (độ), mặc định 0.0
        couch_angle : float, optional
            Góc bàn (độ), mặc định 0.0
        isocenter : Tuple[float, float, float], optional
            Tọa độ isocenter (x, y, z), mặc định (0.0, 0.0, 0.0)
        field_size_x : float, optional
            Kích thước trường theo X (cm), mặc định 10.0
        field_size_y : float, optional
            Kích thước trường theo Y (cm), mặc định 10.0
        beam_type : BeamType, optional
            Loại chùm tia, mặc định PHOTON
        """
        self.energy = energy
        self.dose_rate = dose_rate
        self.monitor_units = monitor_units
        self.gantry_angle = gantry_angle
        self.collimator_angle = collimator_angle
        self.couch_angle = couch_angle
        self.isocenter = isocenter
        self.field_size_x = field_size_x
        self.field_size_y = field_size_y
        self.beam_type = beam_type

        # Thông số bổ sung
        self.weight = 1.0
        self.description = ""
        self.metadata = {}

    def set_geometry(self, gantry: float, collimator: float, couch: float):
        """
        Thiết lập góc hình học của chùm tia.

        Parameters
        ----------
        gantry : float
            Góc gantry (độ)
        collimator : float
            Góc collimator (độ)
        couch : float
            Góc bàn (độ)
        """
        self.gantry_angle = gantry
        self.collimator_angle = collimator
        self.couch_angle = couch

    def set_field_size(self, x: float, y: float):
        """
        Thiết lập kích thước trường chiếu.

        Parameters
        ----------
        x : float
            Kích thước theo X (cm)
        y : float
            Kích thước theo Y (cm)
        """
        self.field_size_x = x
        self.field_size_y = y

    def set_isocenter(self, x: float, y: float, z: float):
        """
        Thiết lập tọa độ isocenter.

        Parameters
        ----------
        x : float
            Tọa độ X (cm)
        y : float
            Tọa độ Y (cm)
        z : float
            Tọa độ Z (cm)
        """
        self.isocenter = (x, y, z)

    def to_dict(self) -> Dict[str, Any]:
        """
        Chuyển đổi BeamParameters thành dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary chứa thông tin BeamParameters
        """
        return {
            "energy": self.energy,
            "dose_rate": self.dose_rate,
            "monitor_units": self.monitor_units,
            "gantry_angle": self.gantry_angle,
            "collimator_angle": self.collimator_angle,
            "couch_angle": self.couch_angle,
            "isocenter": self.isocenter,
            "field_size_x": self.field_size_x,
            "field_size_y": self.field_size_y,
            "beam_type": self.beam_type.value,
            "weight": self.weight,
            "description": self.description,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BeamParameters":
        """
        Tạo BeamParameters từ dictionary.

        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary chứa thông tin BeamParameters

        Returns
        -------
        BeamParameters
            Instance BeamParameters được tạo từ dictionary
        """
        beam_params = cls(
            energy=data.get("energy", 6.0),
            dose_rate=data.get("dose_rate", 600.0),
            monitor_units=data.get("monitor_units", 100.0),
            gantry_angle=data.get("gantry_angle", 0.0),
            collimator_angle=data.get("collimator_angle", 0.0),
            couch_angle=data.get("couch_angle", 0.0),
            isocenter=data.get("isocenter", (0.0, 0.0, 0.0)),
            field_size_x=data.get("field_size_x", 10.0),
            field_size_y=data.get("field_size_y", 10.0),
            beam_type=BeamType(data.get("beam_type", "PHOTON")),
        )

        beam_params.weight = data.get("weight", 1.0)
        beam_params.description = data.get("description", "")
        beam_params.metadata = data.get("metadata", {})

        return beam_params

    def __str__(self) -> str:
        return f"BeamParameters(energy={self.energy}MV, mu={self.monitor_units})"

    def __repr__(self) -> str:
        return (
            f"BeamParameters(energy={self.energy}, dose_rate={self.dose_rate}, "
            f"monitor_units={self.monitor_units}, gantry={self.gantry_angle}°)"
        )


class PhotonBeam(Beam):
    """
    Lớp đại diện cho chùm tia photon.

    Lớp này kế thừa từ Beam và cung cấp các tính năng đặc biệt
    cho chùm tia photon trong xạ trị.
    """

    def __init__(
        self, beam_name: str, energy: float = 6.0, beam_id: Optional[str] = None
    ):
        """
        Khởi tạo chùm tia photon.

        Parameters
        ----------
        beam_name : str
            Tên chùm tia
        energy : float, optional
            Năng lượng photon (MV), mặc định 6.0
        beam_id : str, optional
            ID duy nhất của chùm tia
        """
        super().__init__(beam_name, beam_id)

        # Thiết lập loại chùm tia là photon
        self.beam_type = BeamType.PHOTON
        self.energy = energy

        # Các thông số đặc biệt cho photon
        self.flattening_filter = True  # Có sử dụng flattening filter
        self.wedge_angle = 0.0  # Góc wedge (độ)
        self.wedge_orientation = 0.0  # Hướng wedge (độ)

        # Thông số MLC (Multi-Leaf Collimator)
        self.mlc_positions = None  # Vị trí các lá MLC
        self.jaw_positions = [10.0, 10.0, 10.0, 10.0]  # X1, X2, Y1, Y2 (cm)

        logger.info(f"Khởi tạo PhotonBeam '{beam_name}' với năng lượng {energy}MV")

    def set_flattening_filter(self, use_filter: bool):
        """
        Thiết lập sử dụng flattening filter.

        Parameters
        ----------
        use_filter : bool
            True để sử dụng flattening filter, False cho FFF (Flattening Filter Free)
        """
        self.flattening_filter = use_filter
        logger.info(f"Flattening filter: {'ON' if use_filter else 'OFF (FFF)'}")

    def set_wedge(self, angle: float, orientation: float = 0.0):
        """
        Thiết lập wedge cho chùm tia.

        Parameters
        ----------
        angle : float
            Góc wedge (độ)
        orientation : float, optional
            Hướng wedge (độ), mặc định 0.0
        """
        self.wedge_angle = angle
        self.wedge_orientation = orientation
        logger.info(f"Wedge: {angle}° tại hướng {orientation}°")

    def set_jaw_positions(self, x1: float, x2: float, y1: float, y2: float):
        """
        Thiết lập vị trí các jaw.

        Parameters
        ----------
        x1 : float
            Vị trí jaw X1 (cm)
        x2 : float
            Vị trí jaw X2 (cm)
        y1 : float
            Vị trí jaw Y1 (cm)
        y2 : float
            Vị trí jaw Y2 (cm)
        """
        self.jaw_positions = [x1, x2, y1, y2]
        field_size_x = abs(x2 - x1)
        field_size_y = abs(y2 - y1)
        logger.info(f"Jaw positions: X1={x1}, X2={x2}, Y1={y1}, Y2={y2}")
        logger.info(f"Field size: {field_size_x} x {field_size_y} cm²")

    def set_mlc_positions(self, positions: List[List[float]]):
        """
        Thiết lập vị trí MLC.

        Parameters
        ----------
        positions : List[List[float]]
            Danh sách vị trí MLC, mỗi phần tử là [leaf_A, leaf_B]
        """
        self.mlc_positions = positions
        logger.info(f"MLC positions set với {len(positions)} cặp lá")

    def get_field_size(self) -> Tuple[float, float]:
        """
        Lấy kích thước trường chiếu.

        Returns
        -------
        Tuple[float, float]
            Kích thước trường (width, height) trong cm
        """
        if self.jaw_positions:
            x1, x2, y1, y2 = self.jaw_positions
            width = abs(x2 - x1)
            height = abs(y2 - y1)
            return (width, height)
        else:
            return (0.0, 0.0)

    def get_beam_quality(self) -> str:
        """
        Lấy chất lượng chùm tia.

        Returns
        -------
        str
            Mô tả chất lượng chùm tia
        """
        filter_status = "FF" if self.flattening_filter else "FFF"
        return f"{self.energy}MV-{filter_status}"

    def calculate_monitor_units(
        self, prescribed_dose: float, calibration_factor: float = 1.0
    ) -> float:
        """
        Tính toán Monitor Units cần thiết.

        Parameters
        ----------
        prescribed_dose : float
            Liều kê đơn (cGy)
        calibration_factor : float, optional
            Hệ số hiệu chuẩn, mặc định 1.0

        Returns
        -------
        float
            Số Monitor Units cần thiết
        """
        # Công thức đơn giản: MU = Dose / (Dose_rate * Calibration_factor)
        # Trong thực tế sẽ phức tạp hơn với các hệ số hiệu chính
        mu = prescribed_dose / calibration_factor
        self.monitor_units = mu

        logger.info(f"Calculated MU: {mu:.2f} for dose {prescribed_dose} cGy")
        return mu

    def validate_beam_parameters(self) -> Tuple[bool, List[str]]:
        """
        Kiểm tra tính hợp lệ của các tham số chùm tia.

        Returns
        -------
        Tuple[bool, List[str]]
            (is_valid, error_messages)
        """
        errors = []

        # Kiểm tra năng lượng
        valid_energies = [4, 6, 10, 15, 18, 23]  # MV
        if self.energy not in valid_energies:
            errors.append(
                f"Năng lượng {self.energy}MV không hợp lệ. Các giá trị hợp lệ: {valid_energies}"
            )

        # Kiểm tra góc wedge
        if self.wedge_angle < 0 or self.wedge_angle > 60:
            errors.append(f"Góc wedge {self.wedge_angle}° nằm ngoài phạm vi 0-60°")

        # Kiểm tra jaw positions
        if self.jaw_positions:
            x1, x2, y1, y2 = self.jaw_positions
            if x1 >= x2 or y1 >= y2:
                errors.append("Vị trí jaw không hợp lệ")

            field_size_x = abs(x2 - x1)
            field_size_y = abs(y2 - y1)

            if field_size_x > 40 or field_size_y > 40:
                errors.append(
                    f"Kích thước trường {field_size_x}x{field_size_y} cm² vượt quá giới hạn 40x40 cm²"
                )

        # Kiểm tra monitor units
        if self.monitor_units <= 0:
            errors.append("Monitor Units phải lớn hơn 0")

        if self.monitor_units > 9999:
            errors.append("Monitor Units vượt quá giới hạn 9999 MU")

        return (len(errors) == 0, errors)

    def to_dict(self) -> Dict[str, Any]:
        """
        Chuyển đổi PhotonBeam thành dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary chứa thông tin PhotonBeam
        """
        base_dict = super().to_dict()

        photon_dict = {
            "flattening_filter": self.flattening_filter,
            "wedge_angle": self.wedge_angle,
            "wedge_orientation": self.wedge_orientation,
            "mlc_positions": self.mlc_positions,
            "jaw_positions": self.jaw_positions,
            "beam_quality": self.get_beam_quality(),
            "field_size": self.get_field_size(),
        }

        base_dict.update(photon_dict)
        return base_dict

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PhotonBeam":
        """
        Tạo PhotonBeam từ dictionary.

        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary chứa thông tin PhotonBeam

        Returns
        -------
        PhotonBeam
            Instance PhotonBeam được tạo từ dictionary
        """
        beam = cls(
            beam_name=data.get("beam_name", "Photon Beam"),
            energy=data.get("energy", 6.0),
            beam_id=data.get("beam_id"),
        )

        # Thiết lập các thuộc tính từ base class
        beam.dose_rate = data.get("dose_rate", 600.0)
        beam.monitor_units = data.get("monitor_units", 100.0)
        beam.weight = data.get("weight", 1.0)
        beam.status = BeamStatus(data.get("status", "PLANNED"))
        beam.description = data.get("description", "")
        beam.metadata = data.get("metadata", {})

        # Thiết lập geometry
        if "geometry" in data:
            beam.geometry = BeamGeometry.from_dict(data["geometry"])

        # Thiết lập các thuộc tính photon
        beam.flattening_filter = data.get("flattening_filter", True)
        beam.wedge_angle = data.get("wedge_angle", 0.0)
        beam.wedge_orientation = data.get("wedge_orientation", 0.0)
        beam.mlc_positions = data.get("mlc_positions")
        beam.jaw_positions = data.get("jaw_positions", [10.0, 10.0, 10.0, 10.0])

        return beam

    def __str__(self) -> str:
        field_size = self.get_field_size()
        return (
            f"PhotonBeam('{self.beam_name}', {self.get_beam_quality()}, "
            f"{field_size[0]}x{field_size[1]}cm², {self.monitor_units}MU)"
        )

    def __repr__(self) -> str:
        return f"PhotonBeam('{self.beam_name}', {self.energy}MV)"


class ElectronBeam(Beam):
    """
    Lớp đại diện cho chùm tia electron.

    Kế thừa từ lớp Beam và bổ sung các tính năng đặc trưng cho electron.
    """

    def __init__(
        self, beam_name: str, energy: float = 6.0, beam_id: Optional[str] = None
    ):
        """
        Khởi tạo chùm tia electron.

        Parameters
        ----------
        beam_name : str
            Tên chùm tia
        energy : float, optional
            Năng lượng electron (MeV), mặc định 6.0
        beam_id : str, optional
            ID duy nhất của chùm tia
        """
        super().__init__(beam_name, beam_id)

        # Thiết lập loại chùm tia
        self.beam_type = BeamType.ELECTRON
        self.energy = energy  # MeV cho electron

        # Các thông số đặc trưng cho electron
        self.applicator_size = 10.0  # cm
        self.insert_thickness = 0.0  # cm (độ dày insert)
        self.ssd = 100.0  # cm (Source-to-Surface Distance)
        self.range_shifter_thickness = 0.0  # cm

        # Thông số vật lý electron
        self.practical_range = self._calculate_practical_range()
        self.therapeutic_range = self._calculate_therapeutic_range()

        logger.info(f"Khởi tạo ElectronBeam: {beam_name}, {energy} MeV")

    def _calculate_practical_range(self) -> float:
        """
        Tính toán practical range của electron.

        Returns
        -------
        float
            Practical range (cm)
        """
        # Công thức empirical cho practical range
        if self.energy <= 2.5:
            return 0.31 * self.energy - 0.15
        else:
            return 0.22 * self.energy + 0.77

    def _calculate_therapeutic_range(self) -> float:
        """
        Tính toán therapeutic range (R90).

        Returns
        -------
        float
            Therapeutic range (cm)
        """
        # R90 ~ 0.3 * practical_range
        return 0.3 * self.practical_range

    def set_applicator_size(self, size: float):
        """
        Thiết lập kích thước applicator.

        Parameters
        ----------
        size : float
            Kích thước applicator (cm)
        """
        self.applicator_size = size
        logger.info(f"Đặt applicator size: {size} cm")

    def set_insert_thickness(self, thickness: float):
        """
        Thiết lập độ dày insert.

        Parameters
        ----------
        thickness : float
            Độ dày insert (cm)
        """
        self.insert_thickness = thickness
        logger.info(f"Đặt insert thickness: {thickness} cm")

    def set_ssd(self, ssd: float):
        """
        Thiết lập Source-to-Surface Distance.

        Parameters
        ----------
        ssd : float
            SSD (cm)
        """
        self.ssd = ssd
        logger.info(f"Đặt SSD: {ssd} cm")

    def set_range_shifter(self, thickness: float):
        """
        Thiết lập range shifter.

        Parameters
        ----------
        thickness : float
            Độ dày range shifter (cm)
        """
        self.range_shifter_thickness = thickness
        # Cập nhật lại range khi có range shifter
        effective_energy = self.energy * (1 - 0.1 * thickness)  # Simplified model
        self.practical_range = self._calculate_practical_range()
        self.therapeutic_range = self._calculate_therapeutic_range()
        logger.info(f"Đặt range shifter: {thickness} cm")

    def get_field_size(self) -> float:
        """
        Lấy kích thước trường chiếu.

        Returns
        -------
        float
            Kích thước trường (cm)
        """
        return self.applicator_size

    def calculate_monitor_units(
        self,
        prescribed_dose: float,
        calibration_factor: float = 1.0,
        depth: float = None,
    ) -> float:
        """
        Tính toán Monitor Units cho electron.

        Parameters
        ----------
        prescribed_dose : float
            Liều kê đơn (Gy)
        calibration_factor : float, optional
            Hệ số hiệu chuẩn, mặc định 1.0
        depth : float, optional
            Độ sâu tính toán (cm)

        Returns
        -------
        float
            Monitor Units (MU)
        """
        if depth is None:
            depth = self.therapeutic_range

        # Simplified calculation cho electron
        # Trong thực tế cần dùng bảng lookup hoặc model phức tạp hơn
        output_factor = self._get_output_factor()
        percent_depth_dose = self._get_percent_depth_dose(depth)

        mu = (prescribed_dose * 100) / (
            output_factor * percent_depth_dose * calibration_factor
        )

        self.monitor_units = mu
        logger.info(f"Tính toán MU: {mu:.2f} cho {prescribed_dose} Gy")
        return mu

    def _get_output_factor(self) -> float:
        """
        Lấy output factor cho applicator size.

        Returns
        -------
        float
            Output factor
        """
        # Simplified output factor lookup
        # Thực tế cần dùng measured data
        output_factors = {6: 0.86, 10: 1.00, 15: 1.05, 20: 1.08, 25: 1.10}

        # Linear interpolation
        sizes = sorted(output_factors.keys())
        if self.applicator_size <= sizes[0]:
            return output_factors[sizes[0]]
        elif self.applicator_size >= sizes[-1]:
            return output_factors[sizes[-1]]
        else:
            # Simple interpolation
            for i in range(len(sizes) - 1):
                if sizes[i] <= self.applicator_size <= sizes[i + 1]:
                    ratio = (self.applicator_size - sizes[i]) / (
                        sizes[i + 1] - sizes[i]
                    )
                    return output_factors[sizes[i]] + ratio * (
                        output_factors[sizes[i + 1]] - output_factors[sizes[i]]
                    )

        return 1.0  # Default

    def _get_percent_depth_dose(self, depth: float) -> float:
        """
        Tính percent depth dose tại độ sâu cho trước.

        Parameters
        ----------
        depth : float
            Độ sâu (cm)

        Returns
        -------
        float
            Percent depth dose (%)
        """
        # Simplified PDD model cho electron
        if depth <= 0:
            return 100.0
        elif depth >= self.practical_range:
            return 5.0  # Bremsstrahlung tail
        else:
            # Simplified curve fit
            normalized_depth = depth / self.practical_range
            if normalized_depth <= 0.3:
                return 100.0  # Surface region
            else:
                # Linear decrease từ R_th đến R_p
                return 90.0 * (1 - (normalized_depth - 0.3) / 0.7) + 5.0

    def validate_beam_parameters(self) -> Tuple[bool, List[str]]:
        """
        Kiểm tra tính hợp lệ của các tham số chùm electron.

        Returns
        -------
        Tuple[bool, List[str]]
            (True nếu hợp lệ, danh sách lỗi nếu có)
        """
        errors = []

        # Kiểm tra năng lượng
        if not (4.0 <= self.energy <= 25.0):
            errors.append("Năng lượng electron phải trong khoảng 4-25 MeV")

        # Kiểm tra applicator size
        if not (5.0 <= self.applicator_size <= 30.0):
            errors.append("Kích thước applicator phải trong khoảng 5-30 cm")

        # Kiểm tra SSD
        if not (95.0 <= self.ssd <= 110.0):
            errors.append("SSD phải trong khoảng 95-110 cm")

        # Kiểm tra MU
        if not (10.0 <= self.monitor_units <= 999.0):
            errors.append("Monitor Units phải trong khoảng 10-999 MU")

        return len(errors) == 0, errors

    def to_dict(self) -> Dict[str, Any]:
        """
        Chuyển đổi sang dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary chứa thông tin chùm electron
        """
        data = super().to_dict()
        data.update(
            {
                "applicator_size": self.applicator_size,
                "insert_thickness": self.insert_thickness,
                "ssd": self.ssd,
                "range_shifter_thickness": self.range_shifter_thickness,
                "practical_range": self.practical_range,
                "therapeutic_range": self.therapeutic_range,
            }
        )
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ElectronBeam":
        """
        Tạo ElectronBeam từ dictionary.

        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary chứa thông tin chùm electron

        Returns
        -------
        ElectronBeam
            Instance ElectronBeam được tạo
        """
        beam = cls(
            beam_name=data.get("beam_name", ""),
            energy=data.get("energy", 6.0),
            beam_id=data.get("beam_id"),
        )

        # Thiết lập các thông số electron
        beam.applicator_size = data.get("applicator_size", 10.0)
        beam.insert_thickness = data.get("insert_thickness", 0.0)
        beam.ssd = data.get("ssd", 100.0)
        beam.range_shifter_thickness = data.get("range_shifter_thickness", 0.0)

        # Thiết lập các thông số chung
        beam.dose_rate = data.get("dose_rate", 600.0)
        beam.monitor_units = data.get("monitor_units", 100.0)
        beam.weight = data.get("weight", 1.0)
        beam.description = data.get("description", "")

        if "metadata" in data:
            beam.metadata = data["metadata"].copy()

        return beam

    def __str__(self) -> str:
        return f"ElectronBeam '{self.beam_name}': {self.energy} MeV, {self.applicator_size} cm"

    def __repr__(self) -> str:
        return f"ElectronBeam('{self.beam_name}', {self.energy}MeV)"


class ProtonBeam(Beam):
    """
    Lớp đại diện cho chùm tia proton.

    Kế thừa từ lớp Beam và bổ sung các tính năng đặc trưng cho proton therapy.
    """

    def __init__(
        self, beam_name: str, energy: float = 150.0, beam_id: Optional[str] = None
    ):
        """
        Khởi tạo chùm tia proton.

        Parameters
        ----------
        beam_name : str
            Tên chùm tia
        energy : float, optional
            Năng lượng proton (MeV), mặc định 150.0
        beam_id : str, optional
            ID duy nhất của chùm tia
        """
        super().__init__(beam_name, beam_id)

        # Thiết lập loại chùm tia
        self.beam_type = BeamType.PROTON
        self.energy = energy  # MeV cho proton

        # Các thông số đặc trưng cho proton
        self.range_in_water = self._calculate_range_in_water()  # cm
        self.bragg_peak_width = 2.0  # cm (SOBP width)
        self.modulation_width = 0.0  # cm
        self.snout_position = 30.0  # cm
        self.range_shifter_thickness = 0.0  # cm water equivalent

        # Scanning parameters (for pencil beam scanning)
        self.spot_size_x = 5.0  # mm (sigma)
        self.spot_size_y = 5.0  # mm (sigma)
        self.spot_spacing_x = 2.5  # mm
        self.spot_spacing_y = 2.5  # mm

        # Delivery technique
        self.delivery_technique = (
            "PBS"  # PBS (Pencil Beam Scanning) or PS (Passive Scattering)
        )

        logger.info(f"Khởi tạo ProtonBeam: {beam_name}, {energy} MeV")

    def _calculate_range_in_water(self) -> float:
        """
        Tính toán range của proton trong nước.

        Returns
        -------
        float
            Range trong nước (cm)
        """
        # Công thức Geant4-based empirical cho proton range
        # R = 0.31 * E^1.8 (cho E < 100 MeV)
        # R = 0.325 * E - 2.2 (cho E >= 100 MeV)
        if self.energy < 100.0:
            return 0.31 * (self.energy**1.8) / 10.0  # Convert to cm
        else:
            return (0.325 * self.energy - 2.2) / 10.0  # Convert to cm

    def set_energy(self, energy: float):
        """
        Thiết lập năng lượng proton và cập nhật range.

        Parameters
        ----------
        energy : float
            Năng lượng proton (MeV)
        """
        self.energy = energy
        self.range_in_water = self._calculate_range_in_water()
        logger.info(
            f"Đặt năng lượng proton: {energy} MeV, range: {self.range_in_water:.2f} cm"
        )

    def set_modulation_width(self, width: float):
        """
        Thiết lập độ rộng modulation (SOBP).

        Parameters
        ----------
        width : float
            Độ rộng modulation (cm)
        """
        self.modulation_width = width
        logger.info(f"Đặt modulation width: {width} cm")

    def set_range_shifter(self, thickness: float):
        """
        Thiết lập range shifter.

        Parameters
        ----------
        thickness : float
            Độ dày range shifter (cm water equivalent)
        """
        self.range_shifter_thickness = thickness
        logger.info(f"Đặt range shifter: {thickness} cm WE")

    def set_snout_position(self, position: float):
        """
        Thiết lập vị trí snout.

        Parameters
        ----------
        position : float
            Vị trí snout (cm từ isocenter)
        """
        self.snout_position = position
        logger.info(f"Đặt snout position: {position} cm")

    def set_spot_size(self, sigma_x: float, sigma_y: float):
        """
        Thiết lập kích thước spot cho PBS.

        Parameters
        ----------
        sigma_x : float
            Sigma X của spot (mm)
        sigma_y : float
            Sigma Y của spot (mm)
        """
        self.spot_size_x = sigma_x
        self.spot_size_y = sigma_y
        logger.info(f"Đặt spot size: {sigma_x}x{sigma_y} mm")

    def set_spot_spacing(self, spacing_x: float, spacing_y: float):
        """
        Thiết lập khoảng cách giữa các spot.

        Parameters
        ----------
        spacing_x : float
            Khoảng cách X giữa các spot (mm)
        spacing_y : float
            Khoảng cách Y giữa các spot (mm)
        """
        self.spot_spacing_x = spacing_x
        self.spot_spacing_y = spacing_y
        logger.info(f"Đặt spot spacing: {spacing_x}x{spacing_y} mm")

    def set_delivery_technique(self, technique: str):
        """
        Thiết lập kỹ thuật phân phối.

        Parameters
        ----------
        technique : str
            Kỹ thuật phân phối ("PBS" hoặc "PS")
        """
        if technique in ["PBS", "PS"]:
            self.delivery_technique = technique
            logger.info(f"Đặt delivery technique: {technique}")
        else:
            logger.warning(f"Kỹ thuật không hỗ trợ: {technique}")

    def calculate_monitor_units(
        self,
        prescribed_dose: float,
        calibration_factor: float = 1.0,
        depth: float = None,
    ) -> float:
        """
        Tính toán Monitor Units cho proton.

        Parameters
        ----------
        prescribed_dose : float
            Liều kê đơn (Gy)
        calibration_factor : float, optional
            Hệ số hiệu chuẩn, mặc định 1.0
        depth : float, optional
            Độ sâu tính toán (cm)

        Returns
        -------
        float
            Monitor Units (MU)
        """
        if depth is None:
            depth = self.range_in_water * 0.8  # 80% of range (approximate Bragg peak)

        # Simplified calculation cho proton
        # Trong thực tế cần dùng commissioning data và beam model
        if self.delivery_technique == "PBS":
            # PBS typically uses spot weights
            base_mu = prescribed_dose * 100.0 / calibration_factor
        else:
            # Passive scattering
            base_mu = prescribed_dose * 150.0 / calibration_factor

        # Apply range-based correction
        range_factor = self.range_in_water / 20.0  # Normalize to 20cm range
        mu = base_mu * range_factor

        self.monitor_units = mu
        logger.info(f"Tính toán MU cho proton: {mu:.2f} cho {prescribed_dose} Gy")
        return mu

    def get_distal_range(self) -> float:
        """
        Lấy distal range (R90).

        Returns
        -------
        float
            Distal range (cm)
        """
        return self.range_in_water - self.range_shifter_thickness

    def get_proximal_range(self) -> float:
        """
        Lấy proximal range.

        Returns
        -------
        float
            Proximal range (cm)
        """
        distal_range = self.get_distal_range()
        return max(0.0, distal_range - self.modulation_width)

    def validate_beam_parameters(self) -> Tuple[bool, List[str]]:
        """
        Kiểm tra tính hợp lệ của các tham số chùm proton.

        Returns
        -------
        Tuple[bool, List[str]]
            (True nếu hợp lệ, danh sách lỗi nếu có)
        """
        errors = []

        # Kiểm tra năng lượng
        if not (50.0 <= self.energy <= 250.0):
            errors.append("Năng lượng proton phải trong khoảng 50-250 MeV")

        # Kiểm tra modulation width
        if self.modulation_width < 0:
            errors.append("Modulation width không thể âm")

        if self.modulation_width > self.range_in_water:
            errors.append("Modulation width không thể lớn hơn range")

        # Kiểm tra spot size cho PBS
        if self.delivery_technique == "PBS":
            if not (1.0 <= self.spot_size_x <= 20.0):
                errors.append("Spot size X phải trong khoảng 1-20 mm")
            if not (1.0 <= self.spot_size_y <= 20.0):
                errors.append("Spot size Y phải trong khoảng 1-20 mm")

        # Kiểm tra snout position
        if not (10.0 <= self.snout_position <= 50.0):
            errors.append("Snout position phải trong khoảng 10-50 cm")

        return len(errors) == 0, errors

    def to_dict(self) -> Dict[str, Any]:
        """
        Chuyển đổi sang dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary chứa thông tin chùm proton
        """
        data = super().to_dict()
        data.update(
            {
                "range_in_water": self.range_in_water,
                "bragg_peak_width": self.bragg_peak_width,
                "modulation_width": self.modulation_width,
                "snout_position": self.snout_position,
                "range_shifter_thickness": self.range_shifter_thickness,
                "spot_size_x": self.spot_size_x,
                "spot_size_y": self.spot_size_y,
                "spot_spacing_x": self.spot_spacing_x,
                "spot_spacing_y": self.spot_spacing_y,
                "delivery_technique": self.delivery_technique,
            }
        )
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProtonBeam":
        """
        Tạo ProtonBeam từ dictionary.

        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary chứa thông tin chùm proton

        Returns
        -------
        ProtonBeam
            Instance ProtonBeam được tạo
        """
        beam = cls(
            beam_name=data.get("beam_name", ""),
            energy=data.get("energy", 150.0),
            beam_id=data.get("beam_id"),
        )

        # Thiết lập các thông số proton
        beam.modulation_width = data.get("modulation_width", 0.0)
        beam.snout_position = data.get("snout_position", 30.0)
        beam.range_shifter_thickness = data.get("range_shifter_thickness", 0.0)
        beam.spot_size_x = data.get("spot_size_x", 5.0)
        beam.spot_size_y = data.get("spot_size_y", 5.0)
        beam.spot_spacing_x = data.get("spot_spacing_x", 2.5)
        beam.spot_spacing_y = data.get("spot_spacing_y", 2.5)
        beam.delivery_technique = data.get("delivery_technique", "PBS")

        # Thiết lập các thông số chung
        beam.dose_rate = data.get("dose_rate", 600.0)
        beam.monitor_units = data.get("monitor_units", 100.0)
        beam.weight = data.get("weight", 1.0)
        beam.description = data.get("description", "")

        if "metadata" in data:
            beam.metadata = data["metadata"].copy()

        return beam

    def __str__(self) -> str:
        return f"ProtonBeam '{self.beam_name}': {self.energy} MeV, {self.delivery_technique}, Range: {self.range_in_water:.1f} cm"

    def __repr__(self) -> str:
        return f"ProtonBeam('{self.beam_name}', {self.energy}MeV)"
