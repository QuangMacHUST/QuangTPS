"""
QuangTPS Beam Data Module

Module quản lý dữ liệu chùm tia xạ trị.
Cung cấp các class và function để xử lý thông tin chùm tia.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class BeamType(Enum):
    """Enum cho các loại chùm tia."""

    PHOTON = "photon"
    ELECTRON = "electron"
    PROTON = "proton"
    NEUTRON = "neutron"
    MIXED = "mixed"


class DeliveryTechnique(Enum):
    """Enum cho các kỹ thuật phân phối."""

    CONFORMAL_3D = "3d_conformal"
    IMRT = "imrt"
    VMAT = "vmat"
    STEREOTACTIC = "stereotactic"
    TOMOTHERAPY = "tomotherapy"


@dataclass
class CollimatorSettings:
    """Cài đặt collimator."""

    # Jaw settings (mm)
    jaw_x1: float = -100.0
    jaw_x2: float = 100.0
    jaw_y1: float = -100.0
    jaw_y2: float = 100.0

    # Collimator angle (degrees)
    collimator_angle: float = 0.0

    # MLC settings
    mlc_positions: Optional[List[Tuple[float, float]]] = None  # (leaf_a, leaf_b) pairs

    def __post_init__(self):
        """Validate collimator settings."""
        if self.jaw_x1 >= self.jaw_x2:
            raise ValueError("jaw_x1 phải nhỏ hơn jaw_x2")
        if self.jaw_y1 >= self.jaw_y2:
            raise ValueError("jaw_y1 phải nhỏ hơn jaw_y2")

        # Normalize collimator angle to 0-360 degrees
        self.collimator_angle = self.collimator_angle % 360.0


@dataclass
class BeamGeometry:
    """Hình học chùm tia."""

    # Gantry angle (degrees)
    gantry_angle: float = 0.0

    # Couch angle (degrees)
    couch_angle: float = 0.0

    # Patient support angle (degrees)
    patient_support_angle: float = 0.0

    # Isocenter position (mm)
    isocenter: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Source to axis distance (mm)
    sad: float = 1000.0

    # Source to skin distance (mm)
    ssd: Optional[float] = None

    def __post_init__(self):
        """Validate beam geometry."""
        # Normalize angles to 0-360 degrees
        self.gantry_angle = self.gantry_angle % 360.0
        self.couch_angle = self.couch_angle % 360.0
        self.patient_support_angle = self.patient_support_angle % 360.0

        if self.sad <= 0:
            raise ValueError("SAD phải lớn hơn 0")


@dataclass
class DoseRateSettings:
    """Cài đặt tốc độ liều."""

    # Dose rate (MU/min)
    dose_rate: float = 600.0

    # Monitor units
    monitor_units: float = 100.0

    # Cumulative MU weight
    cumulative_mu_weight: float = 1.0

    # Meterset rate (1/min)
    meterset_rate: Optional[float] = None

    def __post_init__(self):
        """Validate dose rate settings."""
        if self.dose_rate <= 0:
            raise ValueError("Dose rate phải lớn hơn 0")
        if self.monitor_units < 0:
            raise ValueError("Monitor units không thể âm")
        if not (0.0 <= self.cumulative_mu_weight <= 1.0):
            raise ValueError("Cumulative MU weight phải từ 0-1")


@dataclass
class BeamData:
    """
    Class chính để quản lý dữ liệu chùm tia xạ trị.

    Attributes:
        beam_id: ID duy nhất của chùm tia
        beam_name: Tên chùm tia
        beam_type: Loại chùm tia (photon, electron, etc.)
        energy: Năng lượng chùm tia (MV hoặc MeV)
        delivery_technique: Kỹ thuật phân phối
        geometry: Hình học chùm tia
        collimator: Cài đặt collimator
        dose_rate: Cài đặt tốc độ liều
        weight: Trọng số chùm tia
        is_enabled: Chùm tia có được kích hoạt không
        creation_timestamp: Thời gian tạo
        metadata: Metadata bổ sung
    """

    # Basic beam information
    beam_id: str = ""
    beam_name: str = ""
    beam_type: BeamType = BeamType.PHOTON
    energy: str = "6MV"
    delivery_technique: DeliveryTechnique = DeliveryTechnique.CONFORMAL_3D

    # Beam components
    geometry: BeamGeometry = field(default_factory=BeamGeometry)
    collimator: CollimatorSettings = field(default_factory=CollimatorSettings)
    dose_rate: DoseRateSettings = field(default_factory=DoseRateSettings)

    # Beam control
    weight: float = 1.0
    is_enabled: bool = True

    # VMAT specific
    arc_start_angle: Optional[float] = None
    arc_stop_angle: Optional[float] = None
    arc_direction: str = "CW"  # CW or CCW

    # Control points (for VMAT/IMRT)
    control_points: List[Dict[str, Any]] = field(default_factory=list)

    # Metadata
    creation_timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate beam data after initialization."""
        if not self.beam_id:
            self.beam_id = f"beam_{id(self)}"
        if not self.beam_name:
            self.beam_name = f"Beam {self.beam_id}"

        if self.weight < 0:
            raise ValueError("Beam weight không thể âm")

        # Validate VMAT specific parameters
        if self.delivery_technique == DeliveryTechnique.VMAT:
            if self.arc_start_angle is None or self.arc_stop_angle is None:
                logger.warning("VMAT beam thiếu arc angles")

    def get_gantry_angle(self) -> float:
        """Lấy gantry angle."""
        return self.geometry.gantry_angle

    def set_gantry_angle(self, angle: float):
        """Đặt gantry angle."""
        self.geometry.gantry_angle = angle % 360.0

    def get_collimator_angle(self) -> float:
        """Lấy collimator angle."""
        return self.collimator.collimator_angle

    def set_collimator_angle(self, angle: float):
        """Đặt collimator angle."""
        self.collimator.collimator_angle = angle % 360.0

    def get_monitor_units(self) -> float:
        """Lấy monitor units."""
        return self.dose_rate.monitor_units

    def set_monitor_units(self, mu: float):
        """Đặt monitor units."""
        if mu < 0:
            raise ValueError("Monitor units không thể âm")
        self.dose_rate.monitor_units = mu

    def get_jaw_positions(self) -> Tuple[float, float, float, float]:
        """Lấy vị trí jaw (x1, x2, y1, y2)."""
        return (
            self.collimator.jaw_x1,
            self.collimator.jaw_x2,
            self.collimator.jaw_y1,
            self.collimator.jaw_y2,
        )

    def set_jaw_positions(self, x1: float, x2: float, y1: float, y2: float):
        """Đặt vị trí jaw."""
        if x1 >= x2:
            raise ValueError("jaw_x1 phải nhỏ hơn jaw_x2")
        if y1 >= y2:
            raise ValueError("jaw_y1 phải nhỏ hơn jaw_y2")

        self.collimator.jaw_x1 = x1
        self.collimator.jaw_x2 = x2
        self.collimator.jaw_y1 = y1
        self.collimator.jaw_y2 = y2

    def get_mlc_positions(self) -> Optional[List[Tuple[float, float]]]:
        """Lấy vị trí MLC."""
        return self.collimator.mlc_positions

    def set_mlc_positions(self, positions: List[Tuple[float, float]]):
        """Đặt vị trí MLC."""
        # Validate MLC positions
        for i, (leaf_a, leaf_b) in enumerate(positions):
            if leaf_a > leaf_b:
                raise ValueError(f"MLC leaf pair {i}: leaf_a phải <= leaf_b")

        self.collimator.mlc_positions = positions

    def get_field_size(self) -> Tuple[float, float]:
        """Tính kích thước field (width, height) in mm."""
        width = self.collimator.jaw_x2 - self.collimator.jaw_x1
        height = self.collimator.jaw_y2 - self.collimator.jaw_y1
        return (width, height)

    def get_beam_area(self) -> float:
        """Tính diện tích chùm tia (cm²)."""
        width, height = self.get_field_size()
        # Convert mm² to cm²
        area_cm2 = (width * height) / 100.0
        return area_cm2

    def add_control_point(self, control_point: Dict[str, Any]):
        """Thêm control point cho IMRT/VMAT."""
        required_fields = ["gantry_angle", "collimator_angle", "mu_weight"]
        for field in required_fields:
            if field not in control_point:
                raise ValueError(f"Control point thiếu field: {field}")

        self.control_points.append(control_point)

    def get_control_points_count(self) -> int:
        """Lấy số lượng control points."""
        return len(self.control_points)

    def is_arc_beam(self) -> bool:
        """Kiểm tra xem có phải arc beam không."""
        return self.delivery_technique == DeliveryTechnique.VMAT

    def is_modulated_beam(self) -> bool:
        """Kiểm tra xem có phải modulated beam không."""
        return self.delivery_technique in [
            DeliveryTechnique.IMRT,
            DeliveryTechnique.VMAT,
            DeliveryTechnique.TOMOTHERAPY,
        ]

    def calculate_delivery_time(self) -> float:
        """Tính thời gian phân phối (phút)."""
        if self.dose_rate.dose_rate <= 0:
            return 0.0

        # Basic calculation: MU / dose_rate
        delivery_time = self.dose_rate.monitor_units / self.dose_rate.dose_rate

        # Add overhead for modulated beams
        if self.is_modulated_beam():
            # Add 20% overhead for beam modulation
            delivery_time *= 1.2

        return delivery_time

    def validate(self) -> List[str]:
        """Validate beam data và trả về danh sách warnings."""
        warnings = []

        # Check energy format
        if not (self.energy.endswith("MV") or self.energy.endswith("MeV")):
            warnings.append(f"Energy format không chuẩn: {self.energy}")

        # Check monitor units
        if self.dose_rate.monitor_units == 0:
            warnings.append("Monitor units = 0")
        elif self.dose_rate.monitor_units > 1000:
            warnings.append(f"Monitor units cao: {self.dose_rate.monitor_units}")

        # Check field size
        width, height = self.get_field_size()
        if width <= 0 or height <= 0:
            warnings.append("Field size không hợp lệ")
        elif width > 400 or height > 400:  # 40cm x 40cm max
            warnings.append(f"Field size lớn: {width}x{height}mm")

        # Check arc parameters for VMAT
        if self.is_arc_beam():
            if self.arc_start_angle is None or self.arc_stop_angle is None:
                warnings.append("VMAT beam thiếu arc angles")
            elif abs(self.arc_stop_angle - self.arc_start_angle) < 10:
                warnings.append("Arc length quá nhỏ")

        return warnings

    def to_dict(self) -> Dict[str, Any]:
        """Chuyển đổi beam data thành dictionary."""
        return {
            "beam_id": self.beam_id,
            "beam_name": self.beam_name,
            "beam_type": self.beam_type.value,
            "energy": self.energy,
            "delivery_technique": self.delivery_technique.value,
            "geometry": {
                "gantry_angle": self.geometry.gantry_angle,
                "couch_angle": self.geometry.couch_angle,
                "patient_support_angle": self.geometry.patient_support_angle,
                "isocenter": self.geometry.isocenter,
                "sad": self.geometry.sad,
                "ssd": self.geometry.ssd,
            },
            "collimator": {
                "jaw_x1": self.collimator.jaw_x1,
                "jaw_x2": self.collimator.jaw_x2,
                "jaw_y1": self.collimator.jaw_y1,
                "jaw_y2": self.collimator.jaw_y2,
                "collimator_angle": self.collimator.collimator_angle,
                "mlc_positions": self.collimator.mlc_positions,
            },
            "dose_rate": {
                "dose_rate": self.dose_rate.dose_rate,
                "monitor_units": self.dose_rate.monitor_units,
                "cumulative_mu_weight": self.dose_rate.cumulative_mu_weight,
                "meterset_rate": self.dose_rate.meterset_rate,
            },
            "weight": self.weight,
            "is_enabled": self.is_enabled,
            "arc_start_angle": self.arc_start_angle,
            "arc_stop_angle": self.arc_stop_angle,
            "arc_direction": self.arc_direction,
            "control_points": self.control_points,
            "creation_timestamp": self.creation_timestamp.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BeamData":
        """Tạo BeamData từ dictionary."""
        try:
            # Create geometry
            geom_data = data.get("geometry", {})
            geometry = BeamGeometry(
                gantry_angle=geom_data.get("gantry_angle", 0.0),
                couch_angle=geom_data.get("couch_angle", 0.0),
                patient_support_angle=geom_data.get("patient_support_angle", 0.0),
                isocenter=tuple(geom_data.get("isocenter", (0.0, 0.0, 0.0))),
                sad=geom_data.get("sad", 1000.0),
                ssd=geom_data.get("ssd"),
            )

            # Create collimator settings
            coll_data = data.get("collimator", {})
            collimator = CollimatorSettings(
                jaw_x1=coll_data.get("jaw_x1", -100.0),
                jaw_x2=coll_data.get("jaw_x2", 100.0),
                jaw_y1=coll_data.get("jaw_y1", -100.0),
                jaw_y2=coll_data.get("jaw_y2", 100.0),
                collimator_angle=coll_data.get("collimator_angle", 0.0),
                mlc_positions=coll_data.get("mlc_positions"),
            )

            # Create dose rate settings
            dose_data = data.get("dose_rate", {})
            dose_rate = DoseRateSettings(
                dose_rate=dose_data.get("dose_rate", 600.0),
                monitor_units=dose_data.get("monitor_units", 100.0),
                cumulative_mu_weight=dose_data.get("cumulative_mu_weight", 1.0),
                meterset_rate=dose_data.get("meterset_rate"),
            )

            # Parse timestamp
            timestamp_str = data.get("creation_timestamp")
            if timestamp_str:
                creation_timestamp = datetime.fromisoformat(timestamp_str)
            else:
                creation_timestamp = datetime.now()

            # Create BeamData
            beam = cls(
                beam_id=data.get("beam_id", ""),
                beam_name=data.get("beam_name", ""),
                beam_type=BeamType(data.get("beam_type", "photon")),
                energy=data.get("energy", "6MV"),
                delivery_technique=DeliveryTechnique(
                    data.get("delivery_technique", "3d_conformal")
                ),
                geometry=geometry,
                collimator=collimator,
                dose_rate=dose_rate,
                weight=data.get("weight", 1.0),
                is_enabled=data.get("is_enabled", True),
                arc_start_angle=data.get("arc_start_angle"),
                arc_stop_angle=data.get("arc_stop_angle"),
                arc_direction=data.get("arc_direction", "CW"),
                control_points=data.get("control_points", []),
                creation_timestamp=creation_timestamp,
                metadata=data.get("metadata", {}),
            )

            return beam

        except Exception as e:
            logger.error(f"Error creating BeamData from dict: {e}")
            raise

    def copy(self) -> "BeamData":
        """Tạo bản copy của beam data."""
        return BeamData.from_dict(self.to_dict())

    def __str__(self) -> str:
        """String representation."""
        return (
            f"BeamData(id={self.beam_id}, name={self.beam_name}, "
            f"type={self.beam_type.value}, energy={self.energy}, "
            f"gantry={self.geometry.gantry_angle:.1f}°, "
            f"MU={self.dose_rate.monitor_units:.1f})"
        )

    def __repr__(self) -> str:
        """Detailed representation."""
        return str(self)


def create_beam_data(
    beam_name: str,
    beam_type: BeamType = BeamType.PHOTON,
    energy: str = "6MV",
    gantry_angle: float = 0.0,
    monitor_units: float = 100.0,
    **kwargs,
) -> BeamData:
    """
    Factory function để tạo BeamData.

    Args:
        beam_name: Tên chùm tia
        beam_type: Loại chùm tia
        energy: Năng lượng
        gantry_angle: Góc gantry
        monitor_units: Monitor units
        **kwargs: Các tham số bổ sung

    Returns:
        BeamData: Instance của BeamData
    """
    geometry = BeamGeometry(gantry_angle=gantry_angle)
    dose_rate = DoseRateSettings(monitor_units=monitor_units)

    return BeamData(
        beam_name=beam_name,
        beam_type=beam_type,
        energy=energy,
        geometry=geometry,
        dose_rate=dose_rate,
        **kwargs,
    )


def create_vmat_beam(
    beam_name: str,
    arc_start_angle: float,
    arc_stop_angle: float,
    arc_direction: str = "CW",
    energy: str = "6MV",
    monitor_units: float = 200.0,
    **kwargs,
) -> BeamData:
    """
    Factory function để tạo VMAT beam.

    Args:
        beam_name: Tên chùm tia
        arc_start_angle: Góc bắt đầu arc
        arc_stop_angle: Góc kết thúc arc
        arc_direction: Hướng quay (CW/CCW)
        energy: Năng lượng
        monitor_units: Monitor units
        **kwargs: Các tham số bổ sung

    Returns:
        BeamData: VMAT beam instance
    """
    geometry = BeamGeometry(gantry_angle=arc_start_angle)
    dose_rate = DoseRateSettings(monitor_units=monitor_units)

    return BeamData(
        beam_name=beam_name,
        beam_type=BeamType.PHOTON,
        energy=energy,
        delivery_technique=DeliveryTechnique.VMAT,
        geometry=geometry,
        dose_rate=dose_rate,
        arc_start_angle=arc_start_angle,
        arc_stop_angle=arc_stop_angle,
        arc_direction=arc_direction,
        **kwargs,
    )


def create_imrt_beam(
    beam_name: str,
    gantry_angle: float,
    energy: str = "6MV",
    monitor_units: float = 150.0,
    mlc_positions: Optional[List[Tuple[float, float]]] = None,
    **kwargs,
) -> BeamData:
    """
    Factory function để tạo IMRT beam.

    Args:
        beam_name: Tên chùm tia
        gantry_angle: Góc gantry
        energy: Năng lượng
        monitor_units: Monitor units
        mlc_positions: Vị trí MLC
        **kwargs: Các tham số bổ sung

    Returns:
        BeamData: IMRT beam instance
    """
    geometry = BeamGeometry(gantry_angle=gantry_angle)
    dose_rate = DoseRateSettings(monitor_units=monitor_units)
    collimator = CollimatorSettings(mlc_positions=mlc_positions)

    return BeamData(
        beam_name=beam_name,
        beam_type=BeamType.PHOTON,
        energy=energy,
        delivery_technique=DeliveryTechnique.IMRT,
        geometry=geometry,
        collimator=collimator,
        dose_rate=dose_rate,
        **kwargs,
    )


# Compatibility alias
Beam = BeamData


__all__ = [
    "BeamData",
    "BeamType",
    "DeliveryTechnique",
    "BeamGeometry",
    "CollimatorSettings",
    "DoseRateSettings",
    "create_beam_data",
    "create_vmat_beam",
    "create_imrt_beam",
    "Beam",  # Compatibility alias
]
