#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module quản lý Study (Nghiên cứu) và StudyMetadata trong hệ thống QuangTPS.

Study đại diện cho một phiên khám hoặc điều trị của bệnh nhân,
có thể chứa nhiều Series (CT, MR, RTPLAN, etc.)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Dict, List, Optional, Any, Union
from enum import Enum
import uuid

# Tạo logger cho module này
logger = logging.getLogger(__name__)


class StudyType(str, Enum):
    """Loại nghiên cứu y tế."""

    DIAGNOSTIC = "DIAGNOSTIC"  # Chẩn đoán
    TREATMENT_PLANNING = "TREATMENT_PLANNING"  # Lập kế hoạch điều trị
    FOLLOW_UP = "FOLLOW_UP"  # Theo dõi
    VERIFICATION = "VERIFICATION"  # Xác minh
    QUALITY_ASSURANCE = "QA"  # Đảm bảo chất lượng
    RESEARCH = "RESEARCH"  # Nghiên cứu
    OTHER = "OTHER"  # Khác


class StudyStatus(str, Enum):
    """Trạng thái của nghiên cứu."""

    SCHEDULED = "SCHEDULED"  # Đã lên lịch
    IN_PROGRESS = "IN_PROGRESS"  # Đang thực hiện
    COMPLETED = "COMPLETED"  # Hoàn thành
    CANCELLED = "CANCELLED"  # Đã hủy
    FAILED = "FAILED"  # Thất bại
    UNDER_REVIEW = "UNDER_REVIEW"  # Đang xem xét


@dataclass
class StudyMetadata:
    """
    Metadata bổ sung cho Study.

    Chứa các thông tin mở rộng không thuộc DICOM chuẩn.
    """

    # Thông tin phân loại
    priority: str = "NORMAL"  # Mức độ ưu tiên: LOW, NORMAL, HIGH, URGENT
    department: Optional[str] = None  # Khoa thực hiện
    referring_physician: Optional[str] = None  # Bác sĩ chỉ định
    performing_physician: Optional[str] = None  # Bác sĩ thực hiện

    # Thông tin kỹ thuật
    protocol_name: Optional[str] = None  # Tên protocol sử dụng
    scan_options: List[str] = field(default_factory=list)  # Các tùy chọn scan
    contrast_agent: Optional[str] = None  # Chất cản quang
    radiation_dose: Optional[float] = None  # Liều xạ (mGy)

    # Workflow và quality
    approval_status: str = "PENDING"  # PENDING, APPROVED, REJECTED
    quality_score: Optional[float] = None  # Điểm chất lượng (0-100)
    reviewer_notes: List[str] = field(default_factory=list)  # Ghi chú người đánh giá

    # Custom fields cho mở rộng
    custom_fields: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Chuyển đổi metadata thành dictionary."""
        return {
            "priority": self.priority,
            "department": self.department,
            "referring_physician": self.referring_physician,
            "performing_physician": self.performing_physician,
            "protocol_name": self.protocol_name,
            "scan_options": self.scan_options,
            "contrast_agent": self.contrast_agent,
            "radiation_dose": self.radiation_dose,
            "approval_status": self.approval_status,
            "quality_score": self.quality_score,
            "reviewer_notes": self.reviewer_notes,
            "custom_fields": self.custom_fields,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StudyMetadata":
        """Tạo StudyMetadata từ dictionary."""
        return cls(
            priority=data.get("priority", "NORMAL"),
            department=data.get("department"),
            referring_physician=data.get("referring_physician"),
            performing_physician=data.get("performing_physician"),
            protocol_name=data.get("protocol_name"),
            scan_options=data.get("scan_options", []),
            contrast_agent=data.get("contrast_agent"),
            radiation_dose=data.get("radiation_dose"),
            approval_status=data.get("approval_status", "PENDING"),
            quality_score=data.get("quality_score"),
            reviewer_notes=data.get("reviewer_notes", []),
            custom_fields=data.get("custom_fields", {}),
        )


class Study:
    """
    Đại diện cho một Study trong DICOM - một phiên khám/điều trị của bệnh nhân.

    Study có thể chứa nhiều Series (CT, MR, RTPLAN, RTSTRUCT, RTDOSE, etc.)
    và được xác định duy nhất bởi Study Instance UID.
    """

    def __init__(
        self,
        study_instance_uid: str = None,
        study_id: str = None,
        study_date: Union[date, str] = None,
        study_time: str = None,
        accession_number: str = None,
        study_description: str = "",
        study_type: StudyType = StudyType.TREATMENT_PLANNING,
        status: StudyStatus = StudyStatus.SCHEDULED,
        referring_physician: str = "",
        patient_id: str = None,
        metadata: Optional[StudyMetadata] = None,
    ):
        """
        Khởi tạo Study.

        Parameters
        ----------
        study_instance_uid : str, optional
            Study Instance UID theo chuẩn DICOM
        study_id : str, optional
            Study ID (thường là số thứ tự)
        study_date : Union[date, str], optional
            Ngày thực hiện study
        study_time : str, optional
            Thời gian thực hiện study (HHMMSS)
        accession_number : str, optional
            Số đăng ký study
        study_description : str, optional
            Mô tả study
        study_type : StudyType, optional
            Loại study
        status : StudyStatus, optional
            Trạng thái study
        referring_physician : str, optional
            Bác sĩ chỉ định
        patient_id : str, optional
            ID của bệnh nhân
        metadata : StudyMetadata, optional
            Metadata bổ sung
        """

        # DICOM chuẩn
        self.study_instance_uid = study_instance_uid or self._generate_uid()
        self.study_id = study_id or f"STUDY_{datetime.now().strftime('%Y%m%d%H%M%S')}"

        # Xử lý study_date
        if isinstance(study_date, str):
            try:
                self.study_date = datetime.strptime(study_date, "%Y%m%d").date()
            except ValueError:
                try:
                    self.study_date = datetime.strptime(study_date, "%Y-%m-%d").date()
                except ValueError:
                    logger.warning(f"Không thể parse study_date: {study_date}")
                    self.study_date = date.today()
        elif isinstance(study_date, date):
            self.study_date = study_date
        else:
            self.study_date = date.today()

        self.study_time = study_time or datetime.now().strftime("%H%M%S")
        self.accession_number = accession_number or f"ACC{self.study_id}"
        self.study_description = study_description
        self.referring_physician = referring_physician
        self.patient_id = patient_id

        # Enum properties
        self.study_type = (
            study_type
            if isinstance(study_type, StudyType)
            else StudyType.TREATMENT_PLANNING
        )
        self.status = (
            status if isinstance(status, StudyStatus) else StudyStatus.SCHEDULED
        )

        # Series container
        self.series = {}  # Dict[series_instance_uid, Series]

        # Metadata
        self.metadata = metadata or StudyMetadata()

        # Audit trail
        self.created_date = datetime.now()
        self.modified_date = datetime.now()
        self.created_by = None
        self.modified_by = None

        logger.info(f"Tạo mới Study: {self.study_id} ({self.study_instance_uid})")

    def _generate_uid(self) -> str:
        """Tạo Study Instance UID theo chuẩn DICOM."""
        # Sử dụng prefix của QuangTPS
        base_uid = "1.2.826.0.1.3680043.8.498.1"  # Example root
        unique_part = str(uuid.uuid4().int)[:16]  # 16 digits
        return f"{base_uid}.{unique_part}"

    def add_series(self, series) -> None:
        """
        Thêm Series vào Study.

        Parameters
        ----------
        series : Series
            Series object cần thêm
        """
        if hasattr(series, "series_instance_uid"):
            self.series[series.series_instance_uid] = series
            series.study_instance_uid = self.study_instance_uid
            logger.info(
                f"Thêm Series {series.series_instance_uid} vào Study {self.study_id}"
            )
        else:
            logger.error("Series object phải có series_instance_uid")
            raise ValueError("Invalid series object")

    def remove_series(self, series_instance_uid: str) -> bool:
        """
        Xóa Series khỏi Study.

        Parameters
        ----------
        series_instance_uid : str
            UID của Series cần xóa

        Returns
        -------
        bool
            True nếu xóa thành công
        """
        if series_instance_uid in self.series:
            del self.series[series_instance_uid]
            logger.info(f"Xóa Series {series_instance_uid} khỏi Study {self.study_id}")
            return True
        return False

    def get_series(self, series_instance_uid: str):
        """
        Lấy Series theo UID.

        Parameters
        ----------
        series_instance_uid : str
            UID của Series

        Returns
        -------
        Series or None
            Series object hoặc None nếu không tìm thấy
        """
        return self.series.get(series_instance_uid)

    def get_series_by_modality(self, modality: str) -> List:
        """
        Lấy tất cả Series theo modality.

        Parameters
        ----------
        modality : str
            Modality cần tìm (CT, MR, RTPLAN, etc.)

        Returns
        -------
        List[Series]
            Danh sách các Series
        """
        result = []
        for series in self.series.values():
            if hasattr(series, "modality") and series.modality == modality:
                result.append(series)
        return result

    def get_study_datetime(self) -> datetime:
        """
        Lấy datetime của study.

        Returns
        -------
        datetime
            Datetime kết hợp từ study_date và study_time
        """
        try:
            time_part = datetime.strptime(self.study_time, "%H%M%S").time()
            return datetime.combine(self.study_date, time_part)
        except ValueError:
            # Fallback nếu study_time không đúng format
            return datetime.combine(self.study_date, datetime.min.time())

    def update_metadata(self, **kwargs) -> None:
        """
        Cập nhật metadata của study.

        Parameters
        ----------
        **kwargs
            Các field metadata cần cập nhật
        """
        for key, value in kwargs.items():
            if hasattr(self.metadata, key):
                setattr(self.metadata, key, value)
                logger.info(
                    f"Cập nhật metadata.{key} = {value} cho Study {self.study_id}"
                )

        self.modified_date = datetime.now()

    def add_reviewer_note(self, note: str, reviewer: str = None) -> None:
        """
        Thêm ghi chú đánh giá.

        Parameters
        ----------
        note : str
            Nội dung ghi chú
        reviewer : str, optional
            Người đánh giá
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        reviewer_info = f" ({reviewer})" if reviewer else ""
        full_note = f"[{timestamp}]{reviewer_info}: {note}"

        self.metadata.reviewer_notes.append(full_note)
        logger.info(f"Thêm ghi chú cho Study {self.study_id}: {note}")

    def set_status(self, status: StudyStatus, note: str = None) -> None:
        """
        Cập nhật trạng thái study.

        Parameters
        ----------
        status : StudyStatus
            Trạng thái mới
        note : str, optional
            Ghi chú kèm theo
        """
        old_status = self.status
        self.status = status
        self.modified_date = datetime.now()

        status_note = f"Thay đổi trạng thái: {old_status.value} → {status.value}"
        if note:
            status_note += f". {note}"

        self.add_reviewer_note(status_note)
        logger.info(f"Study {self.study_id}: {status_note}")

    def get_summary(self) -> Dict[str, Any]:
        """
        Lấy thông tin tóm tắt study.

        Returns
        -------
        Dict[str, Any]
            Dictionary chứa thông tin tóm tắt
        """
        series_count = len(self.series)
        modalities = set()

        for series in self.series.values():
            if hasattr(series, "modality"):
                modalities.add(series.modality)

        return {
            "study_id": self.study_id,
            "study_instance_uid": self.study_instance_uid,
            "patient_id": self.patient_id,
            "study_date": self.study_date.isoformat(),
            "study_description": self.study_description,
            "study_type": self.study_type.value,
            "status": self.status.value,
            "series_count": series_count,
            "modalities": list(modalities),
            "referring_physician": self.referring_physician,
            "priority": self.metadata.priority,
            "approval_status": self.metadata.approval_status,
        }

    def to_dict(self) -> Dict[str, Any]:
        """
        Chuyển đổi Study thành dictionary để serialization.

        Returns
        -------
        Dict[str, Any]
            Dictionary representation
        """
        return {
            "study_instance_uid": self.study_instance_uid,
            "study_id": self.study_id,
            "study_date": self.study_date.isoformat(),
            "study_time": self.study_time,
            "accession_number": self.accession_number,
            "study_description": self.study_description,
            "study_type": self.study_type.value,
            "status": self.status.value,
            "referring_physician": self.referring_physician,
            "patient_id": self.patient_id,
            "metadata": self.metadata.to_dict(),
            "created_date": self.created_date.isoformat(),
            "modified_date": self.modified_date.isoformat(),
            "created_by": self.created_by,
            "modified_by": self.modified_by,
            "series_count": len(self.series),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Study":
        """
        Tạo Study từ dictionary.

        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary chứa dữ liệu study

        Returns
        -------
        Study
            Study object được tạo
        """
        # Parse dates
        study_date = None
        if "study_date" in data:
            try:
                study_date = datetime.fromisoformat(data["study_date"]).date()
            except (ValueError, TypeError):
                study_date = date.today()

        # Parse enums
        study_type = StudyType.TREATMENT_PLANNING
        if "study_type" in data:
            try:
                study_type = StudyType(data["study_type"])
            except ValueError:
                pass

        status = StudyStatus.SCHEDULED
        if "status" in data:
            try:
                status = StudyStatus(data["status"])
            except ValueError:
                pass

        # Parse metadata
        metadata = StudyMetadata()
        if "metadata" in data:
            metadata = StudyMetadata.from_dict(data["metadata"])

        study = cls(
            study_instance_uid=data.get("study_instance_uid"),
            study_id=data.get("study_id"),
            study_date=study_date,
            study_time=data.get("study_time"),
            accession_number=data.get("accession_number"),
            study_description=data.get("study_description", ""),
            study_type=study_type,
            status=status,
            referring_physician=data.get("referring_physician", ""),
            patient_id=data.get("patient_id"),
            metadata=metadata,
        )

        # Parse timestamps
        if "created_date" in data:
            try:
                study.created_date = datetime.fromisoformat(data["created_date"])
            except (ValueError, TypeError):
                pass

        if "modified_date" in data:
            try:
                study.modified_date = datetime.fromisoformat(data["modified_date"])
            except (ValueError, TypeError):
                pass

        study.created_by = data.get("created_by")
        study.modified_by = data.get("modified_by")

        return study

    def __str__(self) -> str:
        """String representation của Study."""
        return (
            f"Study(ID={self.study_id}, UID={self.study_instance_uid[:16]}..., "
            f"Date={self.study_date}, Type={self.study_type.value}, "
            f"Status={self.status.value}, Series={len(self.series)})"
        )

    def __repr__(self) -> str:
        """Detailed representation của Study."""
        return (
            f"Study(study_instance_uid='{self.study_instance_uid}', "
            f"study_id='{self.study_id}', study_date={self.study_date}, "
            f"study_type={self.study_type}, status={self.status})"
        )


def create_sample_study(patient_id: str = "TEST001") -> Study:
    """
    Tạo sample study cho testing.

    Parameters
    ----------
    patient_id : str, optional
        ID của bệnh nhân

    Returns
    -------
    Study
        Sample study object
    """
    study = Study(
        study_id="STUDY_20241223_001",
        study_date=date.today(),
        study_description="Treatment Planning Study",
        study_type=StudyType.TREATMENT_PLANNING,
        status=StudyStatus.IN_PROGRESS,
        referring_physician="Dr. Nguyen Van A",
        patient_id=patient_id,
    )

    # Thêm metadata mẫu
    study.update_metadata(
        priority="HIGH",
        department="Radiation Oncology",
        protocol_name="IMRT Head & Neck",
        approval_status="APPROVED",
    )

    study.add_reviewer_note("Study created for testing purposes", "System")

    return study


if __name__ == "__main__":
    # Test basic functionality
    logging.basicConfig(level=logging.INFO)

    print("Testing Study module...")

    # Tạo study mẫu
    study = create_sample_study()
    print(f"Created: {study}")

    # Test serialization
    study_dict = study.to_dict()
    restored_study = Study.from_dict(study_dict)
    print(f"Serialization test: {restored_study.study_id == study.study_id}")

    # Test status change
    study.set_status(StudyStatus.COMPLETED, "All series processed successfully")

    # Test summary
    summary = study.get_summary()
    print(f"Summary: {summary}")

    print("Study module testing completed!")
