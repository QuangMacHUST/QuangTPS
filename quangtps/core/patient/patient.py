#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module quản lý thông tin bệnh nhân.

Module cung cấp mô hình dữ liệu toàn diện và các phương thức để quản lý
thông tin bệnh nhân trong hệ thống lập kế hoạch xạ trị QuangTPS.
"""

import uuid
import json
import os
import logging
from datetime import datetime, date
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum, auto

from quangtps.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PatientMetadata:
    """Metadata bổ sung cho bệnh nhân."""

    original_id: str = ""  # ID gốc từ hệ thống cũ
    import_source: str = ""  # Nguồn import dữ liệu
    data_version: str = "1.0"  # Phiên bản dữ liệu
    quality_flags: List[str] = field(default_factory=list)  # Các cờ chất lượng dữ liệu
    custom_fields: Dict[str, Any] = field(default_factory=dict)  # Các trường tùy chỉnh

    def to_dict(self) -> Dict[str, Any]:
        """Chuyển đổi sang dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PatientMetadata":
        """Tạo đối tượng từ dictionary."""
        if not data:
            return cls()
        return cls(
            original_id=data.get("original_id", ""),
            import_source=data.get("import_source", ""),
            data_version=data.get("data_version", "1.0"),
            quality_flags=data.get("quality_flags", []),
            custom_fields=data.get("custom_fields", {}),
        )


class PatientGender(str, Enum):
    """Giới tính của bệnh nhân."""

    MALE = "male"
    FEMALE = "female"
    OTHER = "other"
    UNKNOWN = "unknown"


class PatientStatus(str, Enum):
    """Trạng thái điều trị của bệnh nhân."""

    ACTIVE = "active"  # Đang tham gia điều trị
    PLANNED = "planned"  # Đã lập kế hoạch nhưng chưa bắt đầu điều trị
    ON_TREATMENT = "on_treatment"  # Đang trong quá trình điều trị
    COMPLETED = "completed"  # Đã hoàn thành điều trị
    ON_HOLD = "on_hold"  # Tạm hoãn điều trị
    ARCHIVED = "archived"  # Đã lưu trữ (không còn điều trị tích cực)
    DECEASED = "deceased"  # Bệnh nhân đã mất
    UNKNOWN = "unknown"  # Không xác định


class TreatmentIntent(str, Enum):
    """Mục đích điều trị."""

    CURATIVE = "curative"  # Mục đích điều trị khỏi bệnh
    PALLIATIVE = "palliative"  # Mục đích giảm nhẹ triệu chứng
    ADJUVANT = "adjuvant"  # Điều trị bổ trợ sau phẫu thuật
    NEOADJUVANT = "neoadjuvant"  # Điều trị trước phẫu thuật
    PROPHYLACTIC = "prophylactic"  # Điều trị phòng ngừa
    RESEARCH = "research"  # Nghiên cứu
    UNKNOWN = "unknown"  # Không xác định


@dataclass
class InsuranceInfo:
    """Thông tin bảo hiểm của bệnh nhân."""

    provider: str = ""  # Nhà cung cấp bảo hiểm
    policy_number: str = ""  # Số hợp đồng bảo hiểm
    group_number: str = ""  # Số nhóm
    coverage_start: Optional[date] = None  # Ngày bắt đầu bảo hiểm
    coverage_end: Optional[date] = None  # Ngày kết thúc bảo hiểm
    primary_holder: bool = True  # Bệnh nhân có phải là chủ hợp đồng không
    relationship: str = ""  # Quan hệ với chủ hợp đồng (nếu không phải chủ)
    notes: str = ""  # Ghi chú

    def to_dict(self) -> Dict[str, Any]:
        """Chuyển đổi sang dictionary."""
        data = asdict(self)
        if self.coverage_start:
            data["coverage_start"] = self.coverage_start.isoformat()
        if self.coverage_end:
            data["coverage_end"] = self.coverage_end.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InsuranceInfo":
        """Tạo đối tượng từ dictionary."""
        if not data:
            return cls()

        # Xử lý ngày tháng
        coverage_start = None
        if "coverage_start" in data and data["coverage_start"]:
            try:
                coverage_start = datetime.fromisoformat(data["coverage_start"]).date()
            except (ValueError, TypeError):
                pass

        coverage_end = None
        if "coverage_end" in data and data["coverage_end"]:
            try:
                coverage_end = datetime.fromisoformat(data["coverage_end"]).date()
            except (ValueError, TypeError):
                pass

        return cls(
            provider=data.get("provider", ""),
            policy_number=data.get("policy_number", ""),
            group_number=data.get("group_number", ""),
            coverage_start=coverage_start,
            coverage_end=coverage_end,
            primary_holder=data.get("primary_holder", True),
            relationship=data.get("relationship", ""),
            notes=data.get("notes", ""),
        )


@dataclass
class Physician:
    """Thông tin về bác sĩ."""

    id: str = ""  # ID bác sĩ
    name: str = ""  # Tên bác sĩ
    specialization: str = ""  # Chuyên môn
    institution: str = ""  # Cơ sở y tế
    phone: str = ""  # Số điện thoại
    email: str = ""  # Email
    notes: str = ""  # Ghi chú

    def to_dict(self) -> Dict[str, Any]:
        """Chuyển đổi sang dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Physician":
        """Tạo đối tượng từ dictionary."""
        if not data:
            return cls()
        return cls(
            id=data.get("id", ""),
            name=data.get("name", ""),
            specialization=data.get("specialization", ""),
            institution=data.get("institution", ""),
            phone=data.get("phone", ""),
            email=data.get("email", ""),
            notes=data.get("notes", ""),
        )


@dataclass
class DiagnosisInfo:
    """Thông tin chẩn đoán và bệnh lý."""

    primary_diagnosis: str = ""  # Chẩn đoán chính
    diagnosis_code: str = ""  # Mã ICD-10
    stage: str = ""  # Giai đoạn bệnh
    grade: str = ""  # Độ ác tính
    diagnosis_date: Optional[date] = None  # Ngày chẩn đoán
    site: str = ""  # Vị trí điều trị
    laterality: str = ""  # Bên trái/phải/hai bên
    histology: str = ""  # Mô bệnh học
    secondary_diagnoses: List[str] = field(default_factory=list)  # Chẩn đoán thứ cấp
    notes: str = ""  # Ghi chú

    def to_dict(self) -> Dict[str, Any]:
        """Chuyển đổi sang dictionary."""
        data = asdict(self)
        if self.diagnosis_date:
            data["diagnosis_date"] = self.diagnosis_date.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DiagnosisInfo":
        """Tạo đối tượng từ dictionary."""
        if not data:
            return cls()

        # Xử lý ngày tháng
        diagnosis_date = None
        if "diagnosis_date" in data and data["diagnosis_date"]:
            try:
                diagnosis_date = datetime.fromisoformat(data["diagnosis_date"]).date()
            except (ValueError, TypeError):
                pass

        return cls(
            primary_diagnosis=data.get("primary_diagnosis", ""),
            diagnosis_code=data.get("diagnosis_code", ""),
            stage=data.get("stage", ""),
            grade=data.get("grade", ""),
            diagnosis_date=diagnosis_date,
            site=data.get("site", ""),
            laterality=data.get("laterality", ""),
            histology=data.get("histology", ""),
            secondary_diagnoses=data.get("secondary_diagnoses", []),
            notes=data.get("notes", ""),
        )


@dataclass
class TreatmentProtocol:
    """Thông tin về protocol điều trị được chỉ định."""

    id: str = ""  # ID protocol
    name: str = ""  # Tên protocol
    description: str = ""  # Mô tả
    total_dose: float = 0.0  # Tổng liều (Gy)
    fraction_dose: float = 0.0  # Liều mỗi phân đoạn (Gy)
    num_fractions: int = 0  # Số phân đoạn
    frequency: str = ""  # Tần suất (ví dụ: hàng ngày, 3 lần/tuần)
    technique: str = ""  # Kỹ thuật điều trị
    modality: str = ""  # Phương thức (photon, electron, proton...)
    energy: str = ""  # Năng lượng (6MV, 15MV, ...)
    notes: str = ""  # Ghi chú

    def to_dict(self) -> Dict[str, Any]:
        """Chuyển đổi sang dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TreatmentProtocol":
        """Tạo đối tượng từ dictionary."""
        if not data:
            return cls()
        return cls(
            id=data.get("id", ""),
            name=data.get("name", ""),
            description=data.get("description", ""),
            total_dose=float(data.get("total_dose", 0.0)),
            fraction_dose=float(data.get("fraction_dose", 0.0)),
            num_fractions=int(data.get("num_fractions", 0)),
            frequency=data.get("frequency", ""),
            technique=data.get("technique", ""),
            modality=data.get("modality", ""),
            energy=data.get("energy", ""),
            notes=data.get("notes", ""),
        )


@dataclass
class TreatmentCourse:
    """Thông tin về một đợt điều trị."""

    id: str = ""  # ID đợt điều trị
    name: str = ""  # Tên đợt điều trị
    start_date: Optional[date] = None  # Ngày bắt đầu
    end_date: Optional[date] = None  # Ngày kết thúc
    intent: TreatmentIntent = TreatmentIntent.UNKNOWN  # Mục đích điều trị
    status: str = ""  # Trạng thái
    protocol: Optional[TreatmentProtocol] = None  # Protocol điều trị
    notes: str = ""  # Ghi chú
    plan_ids: List[str] = field(default_factory=list)  # Danh sách ID kế hoạch

    def to_dict(self) -> Dict[str, Any]:
        """Chuyển đổi sang dictionary."""
        data = {
            "id": self.id,
            "name": self.name,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "intent": self.intent.value
            if isinstance(self.intent, Enum)
            else self.intent,
            "status": self.status,
            "protocol": self.protocol.to_dict() if self.protocol else None,
            "notes": self.notes,
            "plan_ids": self.plan_ids,
        }
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TreatmentCourse":
        """Tạo đối tượng từ dictionary."""
        if not data:
            return cls()

        # Xử lý ngày tháng
        start_date = None
        if "start_date" in data and data["start_date"]:
            try:
                start_date = datetime.fromisoformat(data["start_date"]).date()
            except (ValueError, TypeError):
                pass

        end_date = None
        if "end_date" in data and data["end_date"]:
            try:
                end_date = datetime.fromisoformat(data["end_date"]).date()
            except (ValueError, TypeError):
                pass

        # Xử lý protocol
        protocol = None
        if "protocol" in data and data["protocol"]:
            protocol = TreatmentProtocol.from_dict(data["protocol"])

        # Xử lý intent
        try:
            intent = TreatmentIntent(data.get("intent", "unknown"))
        except ValueError:
            intent = TreatmentIntent.UNKNOWN

        return cls(
            id=data.get("id", ""),
            name=data.get("name", ""),
            start_date=start_date,
            end_date=end_date,
            intent=intent,
            status=data.get("status", ""),
            protocol=protocol,
            notes=data.get("notes", ""),
            plan_ids=data.get("plan_ids", []),
        )


@dataclass
class MedicalHistory:
    """Lịch sử y tế của bệnh nhân."""

    previous_treatments: List[str] = field(default_factory=list)  # Điều trị trước đây
    surgeries: List[str] = field(default_factory=list)  # Phẫu thuật
    allergies: List[str] = field(default_factory=list)  # Dị ứng
    medications: List[str] = field(default_factory=list)  # Thuốc đang sử dụng
    family_history: str = ""  # Tiền sử gia đình
    comorbidities: List[str] = field(default_factory=list)  # Bệnh đồng mắc
    notes: str = ""  # Ghi chú

    def to_dict(self) -> Dict[str, Any]:
        """Chuyển đổi sang dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MedicalHistory":
        """Tạo đối tượng từ dictionary."""
        if not data:
            return cls()
        return cls(
            previous_treatments=data.get("previous_treatments", []),
            surgeries=data.get("surgeries", []),
            allergies=data.get("allergies", []),
            medications=data.get("medications", []),
            family_history=data.get("family_history", ""),
            comorbidities=data.get("comorbidities", []),
            notes=data.get("notes", ""),
        )


@dataclass
class Patient:
    """
    Mô hình toàn diện cho bệnh nhân trong hệ thống QuangTPS.

    Lớp này quản lý tất cả thông tin liên quan đến bệnh nhân,
    bao gồm thông tin cá nhân, y tế và điều trị.
    """

    # Thông tin cơ bản
    id: str  # ID bệnh nhân - bắt buộc
    name: str  # Tên bệnh nhân - bắt buộc
    birth_date: date  # Ngày sinh - bắt buộc
    gender: PatientGender = PatientGender.UNKNOWN  # Giới tính
    mrn: str = ""  # Medical Record Number
    patient_id: str = ""  # ID bệnh nhân tại bệnh viện (khác với id trong hệ thống)
    ssn: str = ""  # Số an sinh xã hội (Social Security Number)
    nationality: str = ""  # Quốc tịch
    ethnicity: str = ""  # Dân tộc
    language: str = ""  # Ngôn ngữ chính

    # Thông tin liên hệ
    address: str = ""  # Địa chỉ
    city: str = ""  # Thành phố
    state: str = ""  # Tỉnh/Bang
    postal_code: str = ""  # Mã bưu điện
    country: str = ""  # Quốc gia
    phone: str = ""  # Số điện thoại
    mobile: str = ""  # Số di động
    email: str = ""  # Email
    emergency_contact: str = ""  # Liên hệ khẩn cấp
    emergency_phone: str = ""  # Số điện thoại khẩn cấp

    # Thông tin y tế
    height_cm: float = 0.0  # Chiều cao (cm)
    weight_kg: float = 0.0  # Cân nặng (kg)
    bsa: float = 0.0  # Diện tích cơ thể (Body Surface Area, m²)
    blood_type: str = ""  # Nhóm máu

    # Thông tin chi tiết
    medical_history: MedicalHistory = field(
        default_factory=MedicalHistory
    )  # Lịch sử y tế
    insurance: InsuranceInfo = field(default_factory=InsuranceInfo)  # Bảo hiểm
    diagnosis: DiagnosisInfo = field(
        default_factory=DiagnosisInfo
    )  # Thông tin chẩn đoán
    referring_physician: Physician = field(
        default_factory=Physician
    )  # Bác sĩ giới thiệu
    primary_physician: Physician = field(default_factory=Physician)  # Bác sĩ chính
    radiation_oncologist: Physician = field(
        default_factory=Physician
    )  # Bác sĩ xạ trị ung thư

    # Quản lý điều trị
    treatment_courses: List[TreatmentCourse] = field(
        default_factory=list
    )  # Các đợt điều trị
    status: PatientStatus = PatientStatus.ACTIVE  # Trạng thái điều trị

    # Ghi chú và dữ liệu mở rộng
    notes: str = ""  # Ghi chú chung
    metadata: Dict[str, Any] = field(default_factory=dict)  # Metadata bổ sung

    # Thông tin quản lý
    created_date: datetime = field(default_factory=datetime.now)  # Ngày tạo
    modified_date: Optional[datetime] = None  # Ngày sửa đổi gần nhất
    last_visit: Optional[date] = None  # Ngày thăm khám gần nhất

    def __post_init__(self):
        """Xử lý sau khi khởi tạo."""
        # Nếu ID không được cung cấp, tạo UUID mới
        if not self.id:
            self.id = str(uuid.uuid4())

        # Tính BSA nếu có chiều cao và cân nặng
        if self.height_cm > 0 and self.weight_kg > 0 and self.bsa == 0.0:
            self.calculate_bsa()

    def calculate_bsa(self) -> float:
        """
        Tính diện tích cơ thể (Body Surface Area) theo công thức Dubois.

        BSA = 0.007184 × height(cm)^0.725 × weight(kg)^0.425

        Returns
        -------
        float
            Diện tích cơ thể tính bằng m²
        """
        if self.height_cm <= 0 or self.weight_kg <= 0:
            return 0.0

        self.bsa = round(
            0.007184 * (self.height_cm**0.725) * (self.weight_kg**0.425), 2
        )
        return self.bsa

    def add_treatment_course(self, course: TreatmentCourse) -> None:
        """
        Thêm một đợt điều trị mới.

        Parameters
        ----------
        course : TreatmentCourse
            Đợt điều trị cần thêm
        """
        # Nếu course chưa có ID, tạo mới
        if not course.id:
            course.id = str(uuid.uuid4())

        self.treatment_courses.append(course)
        self.modified_date = datetime.now()

    def get_latest_treatment_course(self) -> Optional[TreatmentCourse]:
        """
        Lấy đợt điều trị gần đây nhất.

        Returns
        -------
        Optional[TreatmentCourse]
            Đợt điều trị gần đây nhất hoặc None nếu không có
        """
        if not self.treatment_courses:
            return None

        # Sắp xếp theo ngày bắt đầu, lấy đợt gần nhất
        sorted_courses = sorted(
            self.treatment_courses,
            key=lambda c: c.start_date if c.start_date else date(1900, 1, 1),
            reverse=True,
        )
        return sorted_courses[0]

    def to_dict(self) -> Dict[str, Any]:
        """
        Chuyển đổi đối tượng thành dictionary để lưu trữ.

        Returns
        -------
        Dict[str, Any]
            Dictionary chứa dữ liệu bệnh nhân
        """
        # Chuẩn bị dữ liệu cơ bản
        data = {
            "id": self.id,
            "name": self.name,
            "birth_date": self.birth_date.isoformat(),
            "gender": self.gender.value
            if isinstance(self.gender, Enum)
            else self.gender,
            "mrn": self.mrn,
            "patient_id": self.patient_id,
            "ssn": self.ssn,
            "nationality": self.nationality,
            "ethnicity": self.ethnicity,
            "language": self.language,
            # Thông tin liên hệ
            "address": self.address,
            "city": self.city,
            "state": self.state,
            "postal_code": self.postal_code,
            "country": self.country,
            "phone": self.phone,
            "mobile": self.mobile,
            "email": self.email,
            "emergency_contact": self.emergency_contact,
            "emergency_phone": self.emergency_phone,
            # Thông tin y tế
            "height_cm": self.height_cm,
            "weight_kg": self.weight_kg,
            "bsa": self.bsa,
            "blood_type": self.blood_type,
            # Thông tin chi tiết (chuyển đổi các đối tượng phức tạp)
            "medical_history": self.medical_history.to_dict(),
            "insurance": self.insurance.to_dict(),
            "diagnosis": self.diagnosis.to_dict(),
            "referring_physician": self.referring_physician.to_dict(),
            "primary_physician": self.primary_physician.to_dict(),
            "radiation_oncologist": self.radiation_oncologist.to_dict(),
            # Đợt điều trị
            "treatment_courses": [
                course.to_dict() for course in self.treatment_courses
            ],
            # Trạng thái
            "status": self.status.value
            if isinstance(self.status, Enum)
            else self.status,
            # Ghi chú và metadata
            "notes": self.notes,
            "metadata": self.metadata,
            # Thông tin quản lý
            "created_date": self.created_date.isoformat(),
            "modified_date": self.modified_date.isoformat()
            if self.modified_date
            else None,
            "last_visit": self.last_visit.isoformat() if self.last_visit else None,
        }

        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Patient":
        """
        Tạo đối tượng Patient từ dictionary.

        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary chứa dữ liệu bệnh nhân

        Returns
        -------
        Patient
            Đối tượng Patient mới
        """
        if not data:
            raise ValueError("Dữ liệu không hợp lệ để tạo bệnh nhân")

        # Kiểm tra các trường bắt buộc
        if "id" not in data or "name" not in data or "birth_date" not in data:
            raise ValueError("Thiếu thông tin bắt buộc (id, name, birth_date)")

        # Xử lý ngày tháng
        try:
            birth_date = datetime.fromisoformat(data["birth_date"]).date()
        except (ValueError, TypeError):
            raise ValueError(
                f"Định dạng ngày sinh không hợp lệ: {data.get('birth_date')}"
            )

        # Xử lý created_date
        created_date = datetime.now()
        if "created_date" in data and data["created_date"]:
            try:
                created_date = datetime.fromisoformat(data["created_date"])
            except (ValueError, TypeError):
                pass

        # Xử lý modified_date
        modified_date = None
        if "modified_date" in data and data["modified_date"]:
            try:
                modified_date = datetime.fromisoformat(data["modified_date"])
            except (ValueError, TypeError):
                pass

        # Xử lý last_visit
        last_visit = None
        if "last_visit" in data and data["last_visit"]:
            try:
                last_visit = datetime.fromisoformat(data["last_visit"]).date()
            except (ValueError, TypeError):
                pass

        # Xử lý gender
        try:
            gender = PatientGender(data.get("gender", "unknown"))
        except ValueError:
            gender = PatientGender.UNKNOWN

        # Xử lý status
        try:
            status = PatientStatus(data.get("status", "unknown"))
        except ValueError:
            status = PatientStatus.UNKNOWN

        # Xử lý các thông tin chi tiết
        medical_history = MedicalHistory.from_dict(data.get("medical_history", {}))
        insurance = InsuranceInfo.from_dict(data.get("insurance", {}))
        diagnosis = DiagnosisInfo.from_dict(data.get("diagnosis", {}))
        referring_physician = Physician.from_dict(data.get("referring_physician", {}))
        primary_physician = Physician.from_dict(data.get("primary_physician", {}))
        radiation_oncologist = Physician.from_dict(data.get("radiation_oncologist", {}))

        # Xử lý đợt điều trị
        treatment_courses = []
        for course_data in data.get("treatment_courses", []):
            try:
                course = TreatmentCourse.from_dict(course_data)
                treatment_courses.append(course)
            except Exception as e:
                logger.error(f"Lỗi khi tạo đợt điều trị: {str(e)}")

        # Tạo đối tượng Patient
        return cls(
            id=data["id"],
            name=data["name"],
            birth_date=birth_date,
            gender=gender,
            mrn=data.get("mrn", ""),
            patient_id=data.get("patient_id", ""),
            ssn=data.get("ssn", ""),
            nationality=data.get("nationality", ""),
            ethnicity=data.get("ethnicity", ""),
            language=data.get("language", ""),
            # Thông tin liên hệ
            address=data.get("address", ""),
            city=data.get("city", ""),
            state=data.get("state", ""),
            postal_code=data.get("postal_code", ""),
            country=data.get("country", ""),
            phone=data.get("phone", ""),
            mobile=data.get("mobile", ""),
            email=data.get("email", ""),
            emergency_contact=data.get("emergency_contact", ""),
            emergency_phone=data.get("emergency_phone", ""),
            # Thông tin y tế
            height_cm=float(data.get("height_cm", 0.0)),
            weight_kg=float(data.get("weight_kg", 0.0)),
            bsa=float(data.get("bsa", 0.0)),
            blood_type=data.get("blood_type", ""),
            # Thông tin chi tiết
            medical_history=medical_history,
            insurance=insurance,
            diagnosis=diagnosis,
            referring_physician=referring_physician,
            primary_physician=primary_physician,
            radiation_oncologist=radiation_oncologist,
            # Đợt điều trị
            treatment_courses=treatment_courses,
            # Trạng thái
            status=status,
            # Ghi chú và metadata
            notes=data.get("notes", ""),
            metadata=data.get("metadata", {}),
            # Thông tin quản lý
            created_date=created_date,
            modified_date=modified_date,
            last_visit=last_visit,
        )

    def save_to_file(self, directory: str) -> str:
        """
        Lưu dữ liệu bệnh nhân vào file JSON.

        Parameters
        ----------
        directory : str
            Thư mục để lưu file

        Returns
        -------
        str
            Đường dẫn đến file đã lưu
        """
        # Tạo thư mục nếu chưa tồn tại
        os.makedirs(directory, exist_ok=True)

        # Cập nhật thời gian sửa đổi
        self.modified_date = datetime.now()

        # Đường dẫn file
        filename = f"{self.id}.json"
        filepath = os.path.join(directory, filename)

        # Chuyển đổi sang JSON và lưu
        data = self.to_dict()
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"Đã lưu thông tin bệnh nhân {self.name} vào {filepath}")
        return filepath

    @classmethod
    def load_from_file(cls, filepath: str) -> "Patient":
        """
        Tải dữ liệu bệnh nhân từ file JSON.

        Parameters
        ----------
        filepath : str
            Đường dẫn đến file

        Returns
        -------
        Patient
            Đối tượng Patient được tải
        """
        # Kiểm tra file tồn tại
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Không tìm thấy file: {filepath}")

        # Đọc dữ liệu
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Tạo đối tượng Patient
        try:
            patient = cls.from_dict(data)
            logger.info(f"Đã tải thông tin bệnh nhân {patient.name} từ {filepath}")
            return patient
        except Exception as e:
            logger.error(f"Lỗi khi tải thông tin bệnh nhân từ {filepath}: {str(e)}")
            raise

    def __str__(self) -> str:
        """
        Chuỗi đại diện cho đối tượng.

        Returns
        -------
        str
            Chuỗi đại diện
        """
        age = datetime.now().year - self.birth_date.year
        return f"{self.name} (ID: {self.id}, {age} tuổi)"

    def get_age(self) -> int:
        """
        Tính tuổi của bệnh nhân.

        Returns
        -------
        int
            Tuổi tính theo năm
        """
        today = date.today()
        age = today.year - self.birth_date.year

        # Kiểm tra xem đã qua sinh nhật trong năm nay chưa
        if (today.month, today.day) < (self.birth_date.month, self.birth_date.day):
            age -= 1

        return age
