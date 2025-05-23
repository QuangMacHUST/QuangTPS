"""
QuangTPS Patient Manager Module

Module quản lý dữ liệu bệnh nhân toàn diện trong hệ thống QuangTPS.
Cung cấp các chức năng CRUD cho patient data, DICOM import/export,
và integration với treatment planning workflow.
"""

import logging
import os
import json
from datetime import datetime, date
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field, asdict
from pathlib import Path

logger = logging.getLogger(__name__)

# Import core types
try:
    from quangtps.core.types import (
        PlanInfo,
        StructureInfo,
        ImageProperties,
        TreatmentType,
        TreatmentStatus,
    )
except ImportError as e:
    logger.warning(f"Không thể import core types: {e}")

    # Fallback definitions
    @dataclass
    class PlanInfo:
        plan_id: str
        plan_name: str
        created_date: datetime = field(default_factory=datetime.now)

    @dataclass
    class StructureInfo:
        name: str
        id: str

    @dataclass
    class ImageProperties:
        modality: str = "CT"

    class TreatmentType:
        EXTERNAL_BEAM = "external_beam"

    class TreatmentStatus:
        PLANNING = "planning"


@dataclass
class PatientDemographics:
    """Thông tin nhân khẩu học của bệnh nhân."""

    patient_id: str
    patient_name: str
    birth_date: Optional[date] = None
    gender: Optional[str] = None  # M, F, O
    address: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    emergency_contact: Optional[str] = None
    emergency_phone: Optional[str] = None

    # Medical information
    medical_record_number: Optional[str] = None
    referring_physician: Optional[str] = None
    primary_oncologist: Optional[str] = None

    def __post_init__(self):
        """Validate patient data."""
        if not self.patient_id or not self.patient_name:
            raise ValueError("Patient ID và Patient Name là bắt buộc")

    @property
    def age(self) -> Optional[int]:
        """Tính tuổi hiện tại của bệnh nhân."""
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

    def get_display_name(self) -> str:
        """Trả về tên hiển thị formatted."""
        return f"{self.patient_name} ({self.patient_id})"


@dataclass
class DiagnosisInfo:
    """Thông tin chẩn đoán và điều trị."""

    primary_diagnosis: str
    icd10_code: Optional[str] = None
    diagnosis_date: Optional[date] = None
    stage: Optional[str] = None
    histology: Optional[str] = None
    grade: Optional[str] = None

    # Treatment intent
    treatment_intent: str = "CURATIVE"  # CURATIVE, PALLIATIVE, ADJUVANT

    # Previous treatments
    previous_surgery: Optional[str] = None
    previous_chemotherapy: Optional[str] = None
    previous_radiation: Optional[str] = None

    # Clinical notes
    clinical_notes: Optional[str] = None

    def __post_init__(self):
        """Validate diagnosis data."""
        if not self.primary_diagnosis:
            raise ValueError("Primary diagnosis là bắt buộc")


@dataclass
class StudyInfo:
    """Thông tin về study (DICOM series)."""

    study_uid: str
    study_date: date
    study_description: str
    modality: str

    # Series information
    series_count: int = 0
    instance_count: int = 0

    # File paths
    dicom_path: Optional[str] = None
    processed_path: Optional[str] = None

    # Metadata
    study_time: Optional[str] = None
    accession_number: Optional[str] = None
    referring_physician: Optional[str] = None

    def __post_init__(self):
        """Validate study data."""
        if not self.study_uid:
            raise ValueError("Study UID là bắt buộc")


@dataclass
class TreatmentCourse:
    """Thông tin khóa điều trị."""

    course_id: str
    course_name: str
    start_date: Optional[date] = None
    end_date: Optional[date] = None

    # Prescription
    total_dose: Optional[float] = None  # Gy
    fractions: Optional[int] = None
    dose_per_fraction: Optional[float] = None  # Gy

    # Plans in this course
    plan_ids: List[str] = field(default_factory=list)

    # Status
    status: str = TreatmentStatus.PLANNING
    completed_fractions: int = 0

    # Clinical information
    treatment_site: Optional[str] = None
    technique: Optional[str] = None

    def __post_init__(self):
        """Validate course data."""
        if not self.course_id or not self.course_name:
            raise ValueError("Course ID và Course Name là bắt buộc")

    @property
    def progress_percentage(self) -> float:
        """Tính % hoàn thành điều trị."""
        if self.fractions and self.fractions > 0:
            return (self.completed_fractions / self.fractions) * 100
        return 0.0

    def is_completed(self) -> bool:
        """Kiểm tra xem khóa điều trị đã hoàn thành chưa."""
        return self.fractions and self.completed_fractions >= self.fractions


class Patient:
    """
    Lớp Patient chính để quản lý tất cả thông tin của một bệnh nhân.
    """

    def __init__(
        self,
        demographics: PatientDemographics,
        diagnosis: Optional[DiagnosisInfo] = None,
    ):
        self.demographics = demographics
        self.diagnosis = diagnosis or DiagnosisInfo(primary_diagnosis="")

        # Collections
        self.studies: Dict[str, StudyInfo] = {}
        self.structures: Dict[str, StructureInfo] = {}
        self.plans: Dict[str, PlanInfo] = {}
        self.courses: Dict[str, TreatmentCourse] = {}

        # Metadata
        self.created_date = datetime.now()
        self.last_modified = datetime.now()
        self.created_by: Optional[str] = None
        self.notes: List[str] = []

        logger.info(f"Tạo bệnh nhân: {self.demographics.get_display_name()}")

    @property
    def patient_id(self) -> str:
        """Patient ID."""
        return self.demographics.patient_id

    @property
    def patient_name(self) -> str:
        """Patient name."""
        return self.demographics.patient_name

    def add_study(self, study: StudyInfo) -> None:
        """Thêm study vào bệnh nhân."""
        self.studies[study.study_uid] = study
        self.last_modified = datetime.now()
        logger.info(f"Thêm study {study.study_description} cho {self.patient_name}")

    def remove_study(self, study_uid: str) -> bool:
        """Xóa study khỏi bệnh nhân."""
        if study_uid in self.studies:
            del self.studies[study_uid]
            self.last_modified = datetime.now()
            logger.info(f"Xóa study {study_uid} từ {self.patient_name}")
            return True
        return False

    def add_structure(self, structure: StructureInfo) -> None:
        """Thêm structure vào bệnh nhân."""
        self.structures[structure.id] = structure
        self.last_modified = datetime.now()
        logger.info(f"Thêm structure {structure.name} cho {self.patient_name}")

    def add_plan(self, plan: PlanInfo) -> None:
        """Thêm plan vào bệnh nhân."""
        self.plans[plan.plan_id] = plan
        self.last_modified = datetime.now()
        logger.info(f"Thêm plan {plan.plan_name} cho {self.patient_name}")

    def add_course(self, course: TreatmentCourse) -> None:
        """Thêm treatment course."""
        self.courses[course.course_id] = course
        self.last_modified = datetime.now()
        logger.info(f"Thêm course {course.course_name} cho {self.patient_name}")

    def get_active_plans(self) -> List[PlanInfo]:
        """Lấy các plan đang active."""
        return [
            plan
            for plan in self.plans.values()
            if hasattr(plan, "status") and plan.status != "CANCELLED"
        ]

    def get_studies_by_modality(self, modality: str) -> List[StudyInfo]:
        """Lấy studies theo modality."""
        return [
            study
            for study in self.studies.values()
            if study.modality.upper() == modality.upper()
        ]

    def get_summary(self) -> Dict[str, Any]:
        """Tạo summary thông tin bệnh nhân."""
        return {
            "patient_info": {
                "id": self.patient_id,
                "name": self.patient_name,
                "age": self.demographics.age,
                "gender": self.demographics.gender,
            },
            "diagnosis": {
                "primary": self.diagnosis.primary_diagnosis,
                "stage": self.diagnosis.stage,
                "treatment_intent": self.diagnosis.treatment_intent,
            },
            "studies_count": len(self.studies),
            "structures_count": len(self.structures),
            "plans_count": len(self.plans),
            "courses_count": len(self.courses),
            "created_date": self.created_date.isoformat(),
            "last_modified": self.last_modified.isoformat(),
        }

    def add_note(self, note: str, author: Optional[str] = None) -> None:
        """Thêm clinical note."""
        timestamp = datetime.now().isoformat()
        formatted_note = f"[{timestamp}]"
        if author:
            formatted_note += f" ({author})"
        formatted_note += f": {note}"

        self.notes.append(formatted_note)
        self.last_modified = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert patient to dictionary."""
        return {
            "demographics": asdict(self.demographics),
            "diagnosis": asdict(self.diagnosis),
            "studies": {uid: asdict(study) for uid, study in self.studies.items()},
            "structures": {
                sid: asdict(struct) for sid, struct in self.structures.items()
            },
            "plans": {pid: asdict(plan) for pid, plan in self.plans.items()},
            "courses": {cid: asdict(course) for cid, course in self.courses.items()},
            "created_date": self.created_date.isoformat(),
            "last_modified": self.last_modified.isoformat(),
            "created_by": self.created_by,
            "notes": self.notes,
        }


class PatientDatabase:
    """
    Database để lưu trữ và quản lý patients.
    """

    def __init__(self, database_path: Optional[str] = None):
        self.database_path = database_path or "data/patient_data"
        self.patients: Dict[str, Patient] = {}

        # Tạo thư mục nếu chưa có
        Path(self.database_path).mkdir(parents=True, exist_ok=True)

        # Load existing patients
        self.load_all_patients()

        logger.info(f"Patient database khởi tạo tại: {self.database_path}")

    def add_patient(self, patient: Patient) -> bool:
        """Thêm patient vào database."""
        try:
            self.patients[patient.patient_id] = patient
            self.save_patient(patient)
            logger.info(f"Thêm patient {patient.patient_name} vào database")
            return True
        except Exception as e:
            logger.error(f"Lỗi thêm patient: {e}")
            return False

    def get_patient(self, patient_id: str) -> Optional[Patient]:
        """Lấy patient theo ID."""
        return self.patients.get(patient_id)

    def remove_patient(self, patient_id: str) -> bool:
        """Xóa patient khỏi database."""
        try:
            if patient_id in self.patients:
                del self.patients[patient_id]

                # Xóa file
                patient_file = os.path.join(self.database_path, f"{patient_id}.json")
                if os.path.exists(patient_file):
                    os.remove(patient_file)

                logger.info(f"Xóa patient {patient_id} khỏi database")
                return True
            return False
        except Exception as e:
            logger.error(f"Lỗi xóa patient: {e}")
            return False

    def search_patients(
        self, query: str, search_fields: Optional[List[str]] = None
    ) -> List[Patient]:
        """Tìm kiếm patients."""
        if not query:
            return list(self.patients.values())

        search_fields = search_fields or [
            "patient_name",
            "patient_id",
            "primary_diagnosis",
        ]
        results = []
        query_lower = query.lower()

        for patient in self.patients.values():
            # Search in patient name and ID
            if (
                query_lower in patient.patient_name.lower()
                or query_lower in patient.patient_id.lower()
            ):
                results.append(patient)
                continue

            # Search in diagnosis
            if (
                hasattr(patient.diagnosis, "primary_diagnosis")
                and query_lower in patient.diagnosis.primary_diagnosis.lower()
            ):
                results.append(patient)
                continue

        return results

    def get_patients_by_date_range(
        self, start_date: date, end_date: date
    ) -> List[Patient]:
        """Lấy patients theo khoảng thời gian tạo."""
        results = []
        for patient in self.patients.values():
            patient_date = patient.created_date.date()
            if start_date <= patient_date <= end_date:
                results.append(patient)
        return results

    def save_patient(self, patient: Patient) -> bool:
        """Lưu patient vào file."""
        try:
            patient_file = os.path.join(
                self.database_path, f"{patient.patient_id}.json"
            )
            with open(patient_file, "w", encoding="utf-8") as f:
                # Convert dates to strings for JSON serialization
                data = patient.to_dict()

                # Handle date serialization
                if isinstance(patient.demographics.birth_date, date):
                    data["demographics"]["birth_date"] = (
                        patient.demographics.birth_date.isoformat()
                    )
                if isinstance(patient.diagnosis.diagnosis_date, date):
                    data["diagnosis"]["diagnosis_date"] = (
                        patient.diagnosis.diagnosis_date.isoformat()
                    )

                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            return True
        except Exception as e:
            logger.error(f"Lỗi lưu patient {patient.patient_id}: {e}")
            return False

    def load_patient(self, patient_file: str) -> Optional[Patient]:
        """Load patient từ file."""
        try:
            with open(patient_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Recreate patient from data
            demographics_data = data["demographics"]

            # Handle date deserialization
            if demographics_data.get("birth_date"):
                demographics_data["birth_date"] = datetime.fromisoformat(
                    demographics_data["birth_date"]
                ).date()

            demographics = PatientDemographics(**demographics_data)

            diagnosis_data = data["diagnosis"]
            if diagnosis_data.get("diagnosis_date"):
                diagnosis_data["diagnosis_date"] = datetime.fromisoformat(
                    diagnosis_data["diagnosis_date"]
                ).date()

            diagnosis = DiagnosisInfo(**diagnosis_data)

            patient = Patient(demographics, diagnosis)

            # Restore collections
            for uid, study_data in data.get("studies", {}).items():
                if study_data.get("study_date"):
                    study_data["study_date"] = datetime.fromisoformat(
                        study_data["study_date"]
                    ).date()
                study = StudyInfo(**study_data)
                patient.add_study(study)

            # Restore metadata
            patient.created_date = datetime.fromisoformat(data["created_date"])
            patient.last_modified = datetime.fromisoformat(data["last_modified"])
            patient.created_by = data.get("created_by")
            patient.notes = data.get("notes", [])

            return patient

        except Exception as e:
            logger.error(f"Lỗi load patient từ {patient_file}: {e}")
            return None

    def load_all_patients(self) -> None:
        """Load tất cả patients từ database."""
        try:
            patient_files = Path(self.database_path).glob("*.json")
            loaded_count = 0

            for patient_file in patient_files:
                patient = self.load_patient(str(patient_file))
                if patient:
                    self.patients[patient.patient_id] = patient
                    loaded_count += 1

            logger.info(f"Loaded {loaded_count} patients từ database")

        except Exception as e:
            logger.error(f"Lỗi load patients: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """Lấy thống kê database."""
        total_patients = len(self.patients)

        # Thống kê theo gender
        gender_stats = {"M": 0, "F": 0, "O": 0, "Unknown": 0}

        # Thống kê theo tuổi
        age_groups = {"<18": 0, "18-65": 0, ">65": 0, "Unknown": 0}

        # Thống kê theo treatment intent
        intent_stats = {"CURATIVE": 0, "PALLIATIVE": 0, "ADJUVANT": 0, "Other": 0}

        for patient in self.patients.values():
            # Gender stats
            gender = patient.demographics.gender or "Unknown"
            if gender in gender_stats:
                gender_stats[gender] += 1
            else:
                gender_stats["Unknown"] += 1

            # Age stats
            age = patient.demographics.age
            if age is None:
                age_groups["Unknown"] += 1
            elif age < 18:
                age_groups["<18"] += 1
            elif age <= 65:
                age_groups["18-65"] += 1
            else:
                age_groups[">65"] += 1

            # Treatment intent stats
            intent = patient.diagnosis.treatment_intent
            if intent in intent_stats:
                intent_stats[intent] += 1
            else:
                intent_stats["Other"] += 1

        return {
            "total_patients": total_patients,
            "gender_distribution": gender_stats,
            "age_distribution": age_groups,
            "treatment_intent_distribution": intent_stats,
            "database_path": self.database_path,
        }


class PatientManager:
    """
    Manager chính để quản lý tất cả operations liên quan đến patient.
    """

    def __init__(self, database_path: Optional[str] = None):
        self.database = PatientDatabase(database_path)
        self.current_patient: Optional[Patient] = None

        logger.info("Patient Manager khởi tạo thành công")

    def create_patient(
        self, patient_id: str, patient_name: str, **kwargs
    ) -> Optional[Patient]:
        """Tạo patient mới."""
        try:
            demographics = PatientDemographics(
                patient_id=patient_id,
                patient_name=patient_name,
                **{
                    k: v
                    for k, v in kwargs.items()
                    if k in PatientDemographics.__dataclass_fields__
                },
            )

            diagnosis_kwargs = {
                k: v
                for k, v in kwargs.items()
                if k in DiagnosisInfo.__dataclass_fields__
            }
            diagnosis = DiagnosisInfo(
                primary_diagnosis=kwargs.get("primary_diagnosis", ""),
                **diagnosis_kwargs,
            )

            patient = Patient(demographics, diagnosis)

            if self.database.add_patient(patient):
                logger.info(f"Tạo patient thành công: {patient.get_display_name()}")
                return patient
            else:
                logger.error(f"Không thể lưu patient {patient_id}")
                return None

        except Exception as e:
            logger.error(f"Lỗi tạo patient: {e}")
            return None

    def set_current_patient(self, patient_id: str) -> bool:
        """Set patient hiện tại."""
        patient = self.database.get_patient(patient_id)
        if patient:
            self.current_patient = patient
            logger.info(f"Set current patient: {patient.get_display_name()}")
            return True
        else:
            logger.warning(f"Không tìm thấy patient {patient_id}")
            return False

    def get_current_patient(self) -> Optional[Patient]:
        """Lấy patient hiện tại."""
        return self.current_patient

    def import_dicom_study(
        self, patient_id: str, dicom_path: str, study_description: Optional[str] = None
    ) -> bool:
        """Import DICOM study cho patient."""
        try:
            patient = self.database.get_patient(patient_id)
            if not patient:
                logger.error(f"Không tìm thấy patient {patient_id}")
                return False

            # TODO: Implement DICOM parsing
            # Tạm thời tạo study info giả
            study = StudyInfo(
                study_uid=f"study_{len(patient.studies) + 1}",
                study_date=date.today(),
                study_description=study_description or "Imported Study",
                modality="CT",
                dicom_path=dicom_path,
            )

            patient.add_study(study)
            self.database.save_patient(patient)

            logger.info(f"Import DICOM study thành công cho {patient.patient_name}")
            return True

        except Exception as e:
            logger.error(f"Lỗi import DICOM: {e}")
            return False

    def export_patient_data(
        self, patient_id: str, export_path: str, format: str = "json"
    ) -> bool:
        """Export patient data."""
        try:
            patient = self.database.get_patient(patient_id)
            if not patient:
                logger.error(f"Không tìm thấy patient {patient_id}")
                return False

            if format.lower() == "json":
                with open(export_path, "w", encoding="utf-8") as f:
                    json.dump(
                        patient.to_dict(), f, indent=2, ensure_ascii=False, default=str
                    )

            logger.info(f"Export patient data thành công: {export_path}")
            return True

        except Exception as e:
            logger.error(f"Lỗi export patient data: {e}")
            return False

    def get_all_patients(self) -> List[Patient]:
        """Lấy tất cả patients."""
        return list(self.database.patients.values())

    def search_patients(self, query: str) -> List[Patient]:
        """Tìm kiếm patients."""
        return self.database.search_patients(query)

    def delete_patient(self, patient_id: str) -> bool:
        """Xóa patient."""
        return self.database.remove_patient(patient_id)

    def get_database_statistics(self) -> Dict[str, Any]:
        """Lấy thống kê database."""
        return self.database.get_statistics()


# Factory functions
def create_patient_manager(database_path: Optional[str] = None) -> PatientManager:
    """Factory function để tạo PatientManager."""
    return PatientManager(database_path)


def create_sample_patient() -> Patient:
    """Tạo sample patient để test."""
    demographics = PatientDemographics(
        patient_id="PT001",
        patient_name="Nguyen Van A",
        birth_date=date(1970, 5, 15),
        gender="M",
        medical_record_number="MRN001",
    )

    diagnosis = DiagnosisInfo(
        primary_diagnosis="Prostate Cancer",
        icd10_code="C61",
        stage="T2N0M0",
        treatment_intent="CURATIVE",
    )

    patient = Patient(demographics, diagnosis)

    # Thêm sample study
    study = StudyInfo(
        study_uid="1.2.3.4.5",
        study_date=date.today(),
        study_description="CT Simulation",
        modality="CT",
    )
    patient.add_study(study)

    return patient


if __name__ == "__main__":
    # Test code
    logging.basicConfig(level=logging.INFO)

    # Tạo patient manager
    pm = create_patient_manager()

    # Tạo sample patient
    patient = create_sample_patient()
    pm.database.add_patient(patient)

    # Test search
    results = pm.search_patients("Nguyen")
    print(f"Tìm thấy {len(results)} patients")

    # Test statistics
    stats = pm.get_database_statistics()
    print(f"Database statistics: {stats}")

    print("Patient Manager test hoàn thành!")
