#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Core Types Module

Module này định nghĩa các types và classes cơ bản được sử dụng
trong toàn bộ hệ thống QuangTPS.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, date
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Dose:
    """Thông tin liều xạ trị."""

    dose_grid: np.ndarray
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    units: str = "Gy"

    # Metadata
    calculation_algorithm: str = ""
    calculation_time: Optional[datetime] = None
    grid_size: Optional[Tuple[int, int, int]] = None

    def __post_init__(self):
        """Xử lý sau khi khởi tạo."""
        if self.grid_size is None:
            self.grid_size = self.dose_grid.shape

    def get_max_dose(self) -> float:
        """Lấy liều tối đa."""
        return float(np.max(self.dose_grid))

    def get_mean_dose(self) -> float:
        """Lấy liều trung bình."""
        return float(np.mean(self.dose_grid))

    def get_dose_at_point(self, x: float, y: float, z: float) -> float:
        """Lấy liều tại một điểm cụ thể."""
        # Convert physical coordinates to grid indices
        i = int((x - self.origin[0]) / self.spacing[0])
        j = int((y - self.origin[1]) / self.spacing[1])
        k = int((z - self.origin[2]) / self.spacing[2])

        # Check bounds
        if (
            0 <= i < self.grid_size[0]
            and 0 <= j < self.grid_size[1]
            and 0 <= k < self.grid_size[2]
        ):
            return float(self.dose_grid[i, j, k])
        else:
            return 0.0


@dataclass
class Patient:
    """Thông tin bệnh nhân."""

    patient_id: str
    patient_name: str = ""
    birth_date: Optional[date] = None
    sex: str = "O"  # M, F, O

    # Medical information
    medical_record_number: str = ""
    referring_physician: str = ""

    # Physical properties
    height: Optional[float] = None  # cm
    weight: Optional[float] = None  # kg

    # Treatment information
    diagnosis: str = ""
    treatment_site: str = ""

    # System information
    created_date: datetime = field(default_factory=datetime.now)
    last_modified: datetime = field(default_factory=datetime.now)

    def get_age(self) -> Optional[int]:
        """Tính tuổi bệnh nhân."""
        if self.birth_date:
            today = date.today()
            return (
                today.year
                - self.birth_date.year
                - (
                    (today.month, today.day)
                    < (self.birth_date.month, self.birth_date.day)
                )
            )
        return None

    def get_bsa(self) -> Optional[float]:
        """Tính diện tích bề mặt cơ thể (BSA) theo công thức DuBois."""
        if self.height and self.weight:
            return 0.007184 * (self.weight**0.425) * (self.height**0.725)
        return None

    def __str__(self) -> str:
        return f"Patient({self.patient_id}: {self.patient_name})"


class TreatmentType(Enum):
    """Loại điều trị xạ trị."""

    EXTERNAL_BEAM = "external_beam"
    BRACHYTHERAPY = "brachytherapy"
    STEREOTACTIC = "stereotactic"
    PROTON = "proton"
    ELECTRON = "electron"


class TreatmentTechnique(Enum):
    """Kỹ thuật điều trị."""

    CONFORMAL_3D = "3d_conformal"
    IMRT = "imrt"
    VMAT = "vmat"
    SBRT = "sbrt"
    SRS = "srs"
    PROTON_THERAPY = "proton"
    ELECTRON_THERAPY = "electron"
    HDR_BRACHYTHERAPY = "hdr_brachy"
    LDR_BRACHYTHERAPY = "ldr_brachy"


class TreatmentStatus(Enum):
    """Trạng thái điều trị."""

    PLANNING = "planning"
    APPROVED = "approved"
    IN_TREATMENT = "in_treatment"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    ON_HOLD = "on_hold"


class DoseUnit(Enum):
    """Đơn vị liều."""

    GY = "Gy"
    CGY = "cGy"
    MU = "MU"  # Monitor Unit
    PERCENT = "percent"


class VolumeUnit(Enum):
    """Đơn vị thể tích."""

    CC = "cc"  # Cubic centimeter
    ML = "ml"  # Milliliter (same as cc)
    PERCENT = "percent"
    LITER = "liter"


class BeamEnergyType(Enum):
    """Loại năng lượng chùm tia."""

    PHOTON_6MV = "6MV"
    PHOTON_10MV = "10MV"
    PHOTON_15MV = "15MV"
    PHOTON_18MV = "18MV"
    PHOTON_6FFF = "6FFF"
    PHOTON_10FFF = "10FFF"
    ELECTRON_6MEV = "6MeV"
    ELECTRON_9MEV = "9MeV"
    ELECTRON_12MEV = "12MeV"
    ELECTRON_15MEV = "15MeV"
    ELECTRON_18MEV = "18MeV"
    PROTON = "PROTON"


class TechniqueType(Enum):
    """Loại kỹ thuật điều trị."""

    CONFORMAL_3D = "3D_CONFORMAL"
    IMRT = "IMRT"
    VMAT = "VMAT"
    SBRT = "SBRT"
    SRS = "SRS"
    ELECTRON = "ELECTRON"
    PROTON = "PROTON"
    BRACHYTHERAPY = "BRACHYTHERAPY"


class StructureType(Enum):
    """Loại cấu trúc giải phẫu."""

    # Target structures
    PTV = "PTV"  # Planning Target Volume
    CTV = "CTV"  # Clinical Target Volume
    GTV = "GTV"  # Gross Target Volume
    ITV = "ITV"  # Internal Target Volume

    # Organs at Risk
    OAR = "OAR"
    ORGAN = "ORGAN"
    CRITICAL_STRUCTURE = "CRITICAL_STRUCTURE"

    # Support structures
    EXTERNAL = "EXTERNAL"
    BODY = "BODY"
    BOLUS = "BOLUS"
    COUCH = "COUCH"
    AVOIDANCE = "AVOIDANCE"

    # Planning structures
    PRV = "PRV"  # Planning Risk Volume
    OPTIMIZATION = "OPTIMIZATION"
    DOSE_REGION = "DOSE_REGION"

    # Other
    MARKER = "MARKER"
    REGISTRATION = "REGISTRATION"
    CONTRAST_AGENT = "CONTRAST_AGENT"
    CAVITY = "CAVITY"
    BRACHY_CHANNEL = "BRACHY_CHANNEL"
    BRACHY_ACCESSORY = "BRACHY_ACCESSORY"
    BRACHY_SRC_APP = "BRACHY_SRC_APP"
    OTHER = "OTHER"


@dataclass
class ImageProperties:
    """Thuộc tính hình ảnh y tế."""

    modality: str = "CT"  # CT, MR, PET, etc.
    pixel_spacing: Tuple[float, float] = (1.0, 1.0)  # mm
    slice_thickness: float = 1.0  # mm
    image_orientation: Tuple[float, ...] = field(
        default_factory=lambda: (1, 0, 0, 0, 1, 0)
    )
    image_position: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Window/Level settings
    window_center: Optional[float] = None
    window_width: Optional[float] = None

    # Acquisition parameters
    kvp: Optional[float] = None
    mas: Optional[float] = None
    slice_location: Optional[float] = None


@dataclass
class StructureInfo:
    """Thông tin cơ bản của structure."""

    name: str
    id: str
    roi_number: Optional[int] = None
    structure_type: str = "OTHER"
    color: Tuple[float, float, float] = (1.0, 0.0, 0.0)
    visible: bool = True

    # Geometric properties
    volume: float = 0.0  # cm³
    centroid: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Clinical properties
    priority: int = 3  # 1 = highest, 4 = lowest
    is_target: bool = False
    is_oar: bool = False


@dataclass
class DoseInfo:
    """Thông tin liều xạ trị."""

    # Grid properties
    grid_shape: Tuple[int, int, int] = (100, 100, 50)
    grid_spacing: Tuple[float, float, float] = (2.0, 2.0, 3.0)
    grid_origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Dose properties
    dose_unit: str = "Gy"
    dose_type: str = "PHYSICAL"  # PHYSICAL, EFFECTIVE, ERROR
    summation_type: str = "PLAN"  # PLAN, BEAM, BRACHY, etc.

    # Statistics
    max_dose: float = 0.0
    min_dose: float = 0.0
    mean_dose: float = 0.0


@dataclass
class BeamInfo:
    """Thông tin chùm tia."""

    beam_number: int
    beam_name: str
    beam_type: str = "STATIC"  # STATIC, DYNAMIC
    radiation_type: str = "PHOTON"  # PHOTON, ELECTRON, PROTON

    # Geometry
    gantry_angle: float = 0.0  # degrees
    collimator_angle: float = 0.0  # degrees
    couch_angle: float = 0.0  # degrees

    # Energy and dose
    nominal_energy: float = 6.0  # MV or MeV
    dose_rate: float = 400.0  # MU/min
    meterset: float = 100.0  # MU

    # Beam limiting device
    jaw_positions: Optional[Tuple[float, float, float, float]] = None  # X1, X2, Y1, Y2

    # Treatment machine
    treatment_machine_name: str = "TrueBeam"

    # Status
    is_setup_beam: bool = False
    beam_on: bool = False


@dataclass
class PlanInfo:
    """Thông tin kế hoạch điều trị."""

    plan_id: str
    plan_name: str
    plan_label: Optional[str] = None
    plan_description: str = ""

    # Treatment information
    treatment_type: TreatmentType = TreatmentType.EXTERNAL_BEAM
    treatment_technique: TreatmentTechnique = TreatmentTechnique.VMAT
    treatment_intent: str = "CURATIVE"

    # Prescription
    prescribed_dose: float = 0.0  # Gy
    number_of_fractions: int = 1
    dose_per_fraction: float = 0.0  # Gy

    # Planning
    planning_system: str = "QuangTPS"
    planner_name: str = ""
    physicist_name: str = ""
    physician_name: str = ""

    # Status and dates
    status: TreatmentStatus = TreatmentStatus.PLANNING
    created_date: datetime = field(default_factory=datetime.now)
    approved_date: Optional[datetime] = None

    # References
    referenced_structures: List[str] = field(default_factory=list)
    referenced_images: List[str] = field(default_factory=list)


class Plan:
    """
    Lớp đại diện cho kế hoạch điều trị.

    Plan chứa tất cả thông tin về một kế hoạch điều trị bao gồm
    thông tin bệnh nhân, prescription, beams, structures và dose.
    """

    def __init__(self, plan_id: str, plan_name: str, **kwargs):
        """Khởi tạo Plan."""
        self.plan_info = PlanInfo(plan_id=plan_id, plan_name=plan_name, **kwargs)

        # Components
        self.beams: List[BeamInfo] = []
        self.structures: List[StructureInfo] = []
        self.dose_info: Optional[DoseInfo] = None

        # Images
        self.primary_image: Optional["Image"] = None
        self.secondary_images: List["Image"] = []

        # Dose data
        self.dose_grid: Optional["DoseGrid"] = None

        # Metadata
        self.metadata: Dict[str, Any] = {}

        logger.debug(f"Tạo Plan: {plan_id} - {plan_name}")

    def add_beam(self, beam_info: BeamInfo):
        """Thêm beam vào plan."""
        self.beams.append(beam_info)
        self.plan_info.referenced_structures.append(f"BEAM_{beam_info.beam_number}")

    def add_structure(self, structure_info: StructureInfo):
        """Thêm structure vào plan."""
        self.structures.append(structure_info)
        self.plan_info.referenced_structures.append(structure_info.id)

    def set_dose_grid(self, dose_grid: "DoseGrid"):
        """Đặt dose grid cho plan."""
        self.dose_grid = dose_grid

        # Update dose info
        if hasattr(dose_grid, "get_shape"):
            shape = dose_grid.get_shape()
        else:
            shape = (100, 100, 50)

        if hasattr(dose_grid, "get_spacing"):
            spacing = dose_grid.get_spacing()
        else:
            spacing = (2.0, 2.0, 3.0)

        if hasattr(dose_grid, "get_origin"):
            origin = dose_grid.get_origin()
        else:
            origin = (0.0, 0.0, 0.0)

        self.dose_info = DoseInfo(
            grid_shape=shape, grid_spacing=spacing, grid_origin=origin
        )

    def get_summary(self) -> Dict[str, Any]:
        """Lấy summary của plan."""
        return {
            "plan_id": self.plan_info.plan_id,
            "plan_name": self.plan_info.plan_name,
            "treatment_type": self.plan_info.treatment_type.value,
            "treatment_technique": self.plan_info.treatment_technique.value,
            "prescribed_dose": self.plan_info.prescribed_dose,
            "number_of_fractions": self.plan_info.number_of_fractions,
            "number_of_beams": len(self.beams),
            "number_of_structures": len(self.structures),
            "status": self.plan_info.status.value,
            "created_date": self.plan_info.created_date.isoformat(),
        }

    def __str__(self) -> str:
        return f"Plan({self.plan_info.plan_id}: {self.plan_info.plan_name})"

    def __repr__(self) -> str:
        return self.__str__()


class Treatment:
    """
    Lớp đại diện cho một course điều trị hoàn chỉnh.

    Treatment có thể chứa nhiều plans và quản lý toàn bộ
    quá trình điều trị của bệnh nhân.
    """

    def __init__(
        self, treatment_id: str, patient_id: str, course_id: Optional[str] = None
    ):
        """Khởi tạo Treatment."""
        self.treatment_id = treatment_id
        self.patient_id = patient_id
        self.course_id = course_id or f"C1_{treatment_id}"

        # Plans
        self.plans: List[Plan] = []
        self.active_plan: Optional[Plan] = None

        # Treatment information
        self.treatment_site: str = ""
        self.diagnosis: str = ""
        self.stage: str = ""
        self.treatment_intent: str = "CURATIVE"

        # Prescription information
        self.total_dose: float = 0.0  # Gy
        self.dose_per_fraction: float = 0.0  # Gy
        self.number_of_fractions: int = 0

        # Status tracking
        self.status = TreatmentStatus.PLANNING
        self.start_date: Optional[date] = None
        self.end_date: Optional[date] = None

        # Images
        self.simulation_ct: Optional["Image"] = None
        self.planning_images: List["Image"] = []
        self.verification_images: List["Image"] = []

        # Clinical team
        self.physician: str = ""
        self.physicist: str = ""
        self.therapist: str = ""

        logger.debug(f"Tạo Treatment: {treatment_id} cho patient {patient_id}")

    def add_plan(self, plan: Plan) -> None:
        """Thêm plan vào treatment."""
        self.plans.append(plan)

        # Set as active plan if it's the first one
        if self.active_plan is None:
            self.active_plan = plan

        logger.debug(
            f"Thêm plan {plan.plan_info.plan_id} vào treatment {self.treatment_id}"
        )

    def set_active_plan(self, plan_id: str) -> bool:
        """Đặt plan làm active plan."""
        for plan in self.plans:
            if plan.plan_info.plan_id == plan_id:
                self.active_plan = plan
                logger.debug(f"Đặt plan {plan_id} làm active plan")
                return True

        logger.warning(f"Không tìm thấy plan {plan_id}")
        return False

    def get_plan_by_id(self, plan_id: str) -> Optional[Plan]:
        """Tìm plan theo ID."""
        for plan in self.plans:
            if plan.plan_info.plan_id == plan_id:
                return plan
        return None

    def calculate_total_prescription(self) -> Tuple[float, int]:
        """Tính tổng prescription từ tất cả plans."""
        total_dose = 0.0
        total_fractions = 0

        for plan in self.plans:
            total_dose += plan.plan_info.prescribed_dose
            total_fractions += plan.plan_info.number_of_fractions

        return total_dose, total_fractions

    def get_treatment_summary(self) -> Dict[str, Any]:
        """Lấy summary của treatment."""
        total_dose, total_fractions = self.calculate_total_prescription()

        return {
            "treatment_id": self.treatment_id,
            "patient_id": self.patient_id,
            "course_id": self.course_id,
            "treatment_site": self.treatment_site,
            "diagnosis": self.diagnosis,
            "treatment_intent": self.treatment_intent,
            "total_dose": total_dose,
            "total_fractions": total_fractions,
            "number_of_plans": len(self.plans),
            "active_plan": self.active_plan.plan_info.plan_id
            if self.active_plan
            else None,
            "status": self.status.value,
            "physician": self.physician,
            "physicist": self.physicist,
        }

    def __str__(self) -> str:
        return f"Treatment({self.treatment_id}: {self.treatment_site})"

    def __repr__(self) -> str:
        return self.__str__()


class Structure:
    """
    Lớp đại diện cho cấu trúc giải phẫu đơn giản.

    Đây là version đơn giản cho compatibility.
    """

    def __init__(self, name: str = "", structure_type: str = "OTHER"):
        """Khởi tạo Structure."""
        self.name = name
        self.type = structure_type
        self.id = f"struct_{hash(name)}"
        self.color = (1.0, 0.0, 0.0)
        self.visible = True
        self.mask = None

    def __str__(self) -> str:
        return f"Structure({self.name})"


class DoseGrid:
    """
    Lớp đại diện cho lưới liều đơn giản.

    Đây là version đơn giản cho compatibility.
    """

    def __init__(self, grid_data=None, origin=None, spacing=None):
        """Khởi tạo DoseGrid."""
        self.grid_data = grid_data if grid_data is not None else np.zeros((50, 50, 30))
        self.origin = origin or (0.0, 0.0, 0.0)
        self.spacing = spacing or (2.0, 2.0, 3.0)

    def get_shape(self):
        """Lấy shape của grid."""
        return self.grid_data.shape

    def get_spacing(self):
        """Lấy spacing của grid."""
        return self.spacing

    def get_origin(self):
        """Lấy origin của grid."""
        return self.origin

    def __str__(self) -> str:
        return f"DoseGrid(shape={self.get_shape()})"


class Image:
    """
    Lớp đại diện cho hình ảnh y tế đơn giản.

    Đây là version đơn giản cho compatibility.
    """

    def __init__(self, image_data=None, properties=None):
        """Khởi tạo Image."""
        self.image_data = image_data if image_data is not None else np.zeros((100, 100))
        self.properties = properties or ImageProperties()

    def get_shape(self):
        """Lấy shape của image."""
        return self.image_data.shape

    def __str__(self) -> str:
        return f"Image(shape={self.get_shape()}, modality={self.properties.modality})"


class BeamParameters:
    """
    Lớp chứa các tham số của chùm tia.

    Đây là version đơn giản cho compatibility.
    """

    def __init__(self):
        """Khởi tạo BeamParameters."""
        self.energy = 6.0  # MV
        self.dose_rate = 600.0  # MU/min
        self.gantry_angle = 0.0  # degrees
        self.collimator_angle = 0.0  # degrees
        self.couch_angle = 0.0  # degrees
        self.monitor_units = 100.0  # MU
        self.weight = 1.0

        # Beam geometry
        self.jaw_x1 = -5.0  # cm
        self.jaw_x2 = 5.0  # cm
        self.jaw_y1 = -5.0  # cm
        self.jaw_y2 = 5.0  # cm

        # Machine parameters
        self.machine_name = "TrueBeam"
        self.technique = "STATIC"

    def __str__(self) -> str:
        return f"BeamParameters(energy={self.energy}MV, mu={self.monitor_units})"


# Factory functions
def create_plan(plan_id: str, plan_name: str, **kwargs) -> Plan:
    """Factory function để tạo Plan."""
    return Plan(plan_id=plan_id, plan_name=plan_name, **kwargs)


def create_treatment(treatment_id: str, patient_id: str, **kwargs) -> Treatment:
    """Factory function để tạo Treatment."""
    return Treatment(treatment_id=treatment_id, patient_id=patient_id, **kwargs)


def create_beam_info(beam_number: int, beam_name: str, **kwargs) -> BeamInfo:
    """Factory function để tạo BeamInfo."""
    return BeamInfo(beam_number=beam_number, beam_name=beam_name, **kwargs)


def create_structure_info(name: str, structure_id: str, **kwargs) -> StructureInfo:
    """Factory function để tạo StructureInfo."""
    return StructureInfo(name=name, id=structure_id, **kwargs)


# Type aliases cho compatibility
TreatmentPlan = Plan
PlanningImageSet = List[Image]
StructureSet = List[Structure]
BeamSet = List[BeamInfo]
