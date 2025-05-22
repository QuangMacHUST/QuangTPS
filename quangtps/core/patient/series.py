#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module quản lý Series và SeriesMetadata trong hệ thống QuangTPS.

Series đại diện cho một chuỗi ảnh y tế (CT, MR, RTPLAN, RTSTRUCT, etc.)
thuộc về một Study.
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, date, time
from typing import Dict, List, Optional, Any, Union, Tuple
from enum import Enum
import uuid

# Tạo logger cho module này
logger = logging.getLogger(__name__)


class Modality(str, Enum):
    """Các loại modality DICOM."""

    # Imaging modalities
    CT = "CT"  # Computed Tomography
    MR = "MR"  # Magnetic Resonance
    US = "US"  # Ultrasound
    XA = "XA"  # X-Ray Angiography
    CR = "CR"  # Computed Radiography
    DX = "DX"  # Digital Radiography
    PET = "PT"  # Positron Emission Tomography
    NM = "NM"  # Nuclear Medicine

    # RT modalities
    RTPLAN = "RTPLAN"  # RT Treatment Plan
    RTDOSE = "RTDOSE"  # RT Dose
    RTSTRUCT = "RTSTRUCT"  # RT Structure Set
    RTIMAGE = "RTIMAGE"  # RT Image
    RTRECORD = "RTRECORD"  # RT Treatment Record

    # Secondary capture
    SC = "SC"  # Secondary Capture
    OT = "OT"  # Other


class SeriesStatus(str, Enum):
    """Trạng thái của Series."""

    PENDING = "PENDING"  # Đang chờ xử lý
    IMPORTING = "IMPORTING"  # Đang import
    READY = "READY"  # Sẵn sàng sử dụng
    PROCESSING = "PROCESSING"  # Đang xử lý
    COMPLETED = "COMPLETED"  # Hoàn thành
    ERROR = "ERROR"  # Có lỗi
    ARCHIVED = "ARCHIVED"  # Đã lưu trữ


@dataclass
class SeriesMetadata:
    """
    Metadata bổ sung cho Series.

    Chứa các thông tin mở rộng không thuộc DICOM chuẩn.
    """

    # Image acquisition
    acquisition_date: Optional[str] = None  # Ngày chụp
    acquisition_time: Optional[str] = None  # Thời gian chụp
    acquisition_number: Optional[int] = None  # Số thứ tự acquisition

    # Technical parameters
    slice_thickness: Optional[float] = None  # Độ dày lát cắt (mm)
    pixel_spacing: List[float] = field(default_factory=list)  # Khoảng cách pixel [x, y]
    image_orientation: List[float] = field(default_factory=list)  # Hướng ảnh
    image_position: List[float] = field(default_factory=list)  # Vị trí ảnh

    # Protocol information
    protocol_name: Optional[str] = None  # Tên protocol
    scanning_sequence: Optional[str] = None  # Chuỗi quét
    sequence_variant: Optional[str] = None  # Biến thể sequence
    scan_options: List[str] = field(default_factory=list)  # Tùy chọn quét

    # Quality and validation
    image_quality: Optional[str] = None  # Chất lượng ảnh
    quality_score: Optional[float] = None  # Điểm chất lượng (0-100)
    validation_status: str = "PENDING"  # PENDING, VALIDATED, REJECTED
    validation_notes: List[str] = field(default_factory=list)  # Ghi chú validation

    # Processing history
    processing_steps: List[str] = field(
        default_factory=list
    )  # Các bước xử lý đã thực hiện
    algorithms_applied: List[str] = field(default_factory=list)  # Thuật toán đã áp dụng

    # Custom fields cho mở rộng
    custom_fields: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Chuyển đổi metadata thành dictionary."""
        return {
            "acquisition_date": self.acquisition_date,
            "acquisition_time": self.acquisition_time,
            "acquisition_number": self.acquisition_number,
            "slice_thickness": self.slice_thickness,
            "pixel_spacing": self.pixel_spacing,
            "image_orientation": self.image_orientation,
            "image_position": self.image_position,
            "protocol_name": self.protocol_name,
            "scanning_sequence": self.scanning_sequence,
            "sequence_variant": self.sequence_variant,
            "scan_options": self.scan_options,
            "image_quality": self.image_quality,
            "quality_score": self.quality_score,
            "validation_status": self.validation_status,
            "validation_notes": self.validation_notes,
            "processing_steps": self.processing_steps,
            "algorithms_applied": self.algorithms_applied,
            "custom_fields": self.custom_fields,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SeriesMetadata":
        """Tạo SeriesMetadata từ dictionary."""
        return cls(
            acquisition_date=data.get("acquisition_date"),
            acquisition_time=data.get("acquisition_time"),
            acquisition_number=data.get("acquisition_number"),
            slice_thickness=data.get("slice_thickness"),
            pixel_spacing=data.get("pixel_spacing", []),
            image_orientation=data.get("image_orientation", []),
            image_position=data.get("image_position", []),
            protocol_name=data.get("protocol_name"),
            scanning_sequence=data.get("scanning_sequence"),
            sequence_variant=data.get("sequence_variant"),
            scan_options=data.get("scan_options", []),
            image_quality=data.get("image_quality"),
            quality_score=data.get("quality_score"),
            validation_status=data.get("validation_status", "PENDING"),
            validation_notes=data.get("validation_notes", []),
            processing_steps=data.get("processing_steps", []),
            algorithms_applied=data.get("algorithms_applied", []),
            custom_fields=data.get("custom_fields", {}),
        )


class Series:
    """
    Đại diện cho một Series trong DICOM - một chuỗi ảnh y tế.

    Series chứa nhiều Instance (ảnh đơn lẻ) và thuộc về một Study.
    """

    def __init__(
        self,
        series_instance_uid: str = None,
        series_number: int = None,
        series_date: Union[date, str] = None,
        series_time: Union[time, str] = None,
        series_description: str = "",
        modality: Modality = Modality.CT,
        status: SeriesStatus = SeriesStatus.PENDING,
        study_instance_uid: str = None,
        patient_id: str = None,
        metadata: Optional[SeriesMetadata] = None,
    ):
        """
        Khởi tạo Series.

        Parameters
        ----------
        series_instance_uid : str, optional
            Series Instance UID theo chuẩn DICOM
        series_number : int, optional
            Số thứ tự series
        series_date : Union[date, str], optional
            Ngày tạo series
        series_time : Union[time, str], optional
            Thời gian tạo series
        series_description : str, optional
            Mô tả series
        modality : Modality, optional
            Loại modality
        status : SeriesStatus, optional
            Trạng thái series
        study_instance_uid : str, optional
            Study Instance UID chủ
        patient_id : str, optional
            ID của bệnh nhân
        metadata : SeriesMetadata, optional
            Metadata bổ sung
        """

        # DICOM chuẩn
        self.series_instance_uid = series_instance_uid or self._generate_uid()
        self.series_number = series_number or 1

        # Xử lý series_date
        if isinstance(series_date, str):
            try:
                self.series_date = datetime.strptime(series_date, "%Y%m%d").date()
            except ValueError:
                try:
                    self.series_date = datetime.strptime(series_date, "%Y-%m-%d").date()
                except ValueError:
                    logger.warning(f"Không thể parse series_date: {series_date}")
                    self.series_date = date.today()
        elif isinstance(series_date, date):
            self.series_date = series_date
        else:
            self.series_date = date.today()

        # Xử lý series_time
        if isinstance(series_time, str):
            try:
                self.series_time = datetime.strptime(series_time, "%H%M%S").time()
            except ValueError:
                try:
                    self.series_time = datetime.strptime(series_time, "%H:%M:%S").time()
                except ValueError:
                    logger.warning(f"Không thể parse series_time: {series_time}")
                    self.series_time = datetime.now().time()
        elif isinstance(series_time, time):
            self.series_time = series_time
        else:
            self.series_time = datetime.now().time()

        self.series_description = series_description
        self.study_instance_uid = study_instance_uid
        self.patient_id = patient_id

        # Enum properties
        self.modality = modality if isinstance(modality, Modality) else Modality.CT
        self.status = (
            status if isinstance(status, SeriesStatus) else SeriesStatus.PENDING
        )

        # Instance container
        self.instances = {}  # Dict[sop_instance_uid, Instance]

        # Image data (nếu đã load)
        self.image_data = None
        self.image_spacing = None
        self.image_origin = None
        self.image_orientation = None

        # Metadata
        self.metadata = metadata or SeriesMetadata()

        # Audit trail
        self.created_date = datetime.now()
        self.modified_date = datetime.now()
        self.created_by = None
        self.modified_by = None

        logger.info(
            f"Tạo mới Series: {self.series_number} ({self.modality.value}) - {self.series_instance_uid}"
        )

    def _generate_uid(self) -> str:
        """Tạo Series Instance UID theo chuẩn DICOM."""
        # Sử dụng prefix của QuangTPS
        base_uid = "1.2.826.0.1.3680043.8.498.2"  # Example root for series
        unique_part = str(uuid.uuid4().int)[:16]  # 16 digits
        return f"{base_uid}.{unique_part}"

    def add_instance(self, instance) -> None:
        """
        Thêm Instance vào Series.

        Parameters
        ----------
        instance : Instance
            Instance object cần thêm
        """
        if hasattr(instance, "sop_instance_uid"):
            self.instances[instance.sop_instance_uid] = instance
            instance.series_instance_uid = self.series_instance_uid
            logger.info(
                f"Thêm Instance {instance.sop_instance_uid} vào Series {self.series_number}"
            )
        else:
            logger.error("Instance object phải có sop_instance_uid")
            raise ValueError("Invalid instance object")

    def remove_instance(self, sop_instance_uid: str) -> bool:
        """
        Xóa Instance khỏi Series.

        Parameters
        ----------
        sop_instance_uid : str
            UID của Instance cần xóa

        Returns
        -------
        bool
            True nếu xóa thành công
        """
        if sop_instance_uid in self.instances:
            del self.instances[sop_instance_uid]
            logger.info(
                f"Xóa Instance {sop_instance_uid} khỏi Series {self.series_number}"
            )
            return True
        return False

    def get_instance(self, sop_instance_uid: str):
        """
        Lấy Instance theo UID.

        Parameters
        ----------
        sop_instance_uid : str
            UID của Instance

        Returns
        -------
        Instance or None
            Instance object hoặc None nếu không tìm thấy
        """
        return self.instances.get(sop_instance_uid)

    def get_instance_count(self) -> int:
        """
        Lấy số lượng Instance trong Series.

        Returns
        -------
        int
            Số lượng Instance
        """
        return len(self.instances)

    def set_image_data(
        self,
        image_data: np.ndarray,
        spacing: Tuple[float, float, float] = None,
        origin: Tuple[float, float, float] = None,
        orientation: Tuple[float, ...] = None,
    ) -> None:
        """
        Thiết lập dữ liệu ảnh cho Series.

        Parameters
        ----------
        image_data : np.ndarray
            Dữ liệu ảnh 3D
        spacing : Tuple[float, float, float], optional
            Khoảng cách voxel (x, y, z)
        origin : Tuple[float, float, float], optional
            Vị trí gốc (x, y, z)
        orientation : Tuple[float, ...], optional
            Hướng ảnh (6 hoặc 9 giá trị)
        """
        self.image_data = image_data
        self.image_spacing = spacing or (1.0, 1.0, 1.0)
        self.image_origin = origin or (0.0, 0.0, 0.0)
        self.image_orientation = orientation or (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)

        # Cập nhật metadata
        if spacing and len(spacing) >= 2:
            self.metadata.pixel_spacing = list(spacing[:2])
        if spacing and len(spacing) >= 3:
            self.metadata.slice_thickness = spacing[2]
        if origin:
            self.metadata.image_position = list(origin)
        if orientation:
            self.metadata.image_orientation = list(orientation)

        self.modified_date = datetime.now()
        logger.info(
            f"Thiết lập image data cho Series {self.series_number}: {image_data.shape}"
        )

    def get_image_data(self) -> Optional[np.ndarray]:
        """
        Lấy dữ liệu ảnh.

        Returns
        -------
        np.ndarray or None
            Dữ liệu ảnh hoặc None nếu chưa có
        """
        return self.image_data

    def get_image_properties(self) -> Dict[str, Any]:
        """
        Lấy thuộc tính ảnh.

        Returns
        -------
        Dict[str, Any]
            Dictionary chứa các thuộc tính ảnh
        """
        properties = {
            "spacing": self.image_spacing,
            "origin": self.image_origin,
            "orientation": self.image_orientation,
            "shape": self.image_data.shape if self.image_data is not None else None,
            "dtype": str(self.image_data.dtype)
            if self.image_data is not None
            else None,
        }
        return properties

    def get_series_datetime(self) -> datetime:
        """
        Lấy datetime của series.

        Returns
        -------
        datetime
            Datetime kết hợp từ series_date và series_time
        """
        return datetime.combine(self.series_date, self.series_time)

    def update_metadata(self, **kwargs) -> None:
        """
        Cập nhật metadata của series.

        Parameters
        ----------
        **kwargs
            Các field metadata cần cập nhật
        """
        for key, value in kwargs.items():
            if hasattr(self.metadata, key):
                setattr(self.metadata, key, value)
                logger.info(
                    f"Cập nhật metadata.{key} = {value} cho Series {self.series_number}"
                )

        self.modified_date = datetime.now()

    def add_processing_step(self, step: str, algorithm: str = None) -> None:
        """
        Thêm bước xử lý vào lịch sử.

        Parameters
        ----------
        step : str
            Mô tả bước xử lý
        algorithm : str, optional
            Tên thuật toán được sử dụng
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        full_step = f"[{timestamp}] {step}"

        self.metadata.processing_steps.append(full_step)

        if algorithm and algorithm not in self.metadata.algorithms_applied:
            self.metadata.algorithms_applied.append(algorithm)

        logger.info(f"Thêm bước xử lý cho Series {self.series_number}: {step}")

    def set_status(self, status: SeriesStatus, note: str = None) -> None:
        """
        Cập nhật trạng thái series.

        Parameters
        ----------
        status : SeriesStatus
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

        self.add_processing_step(status_note)
        logger.info(f"Series {self.series_number}: {status_note}")

    def validate_series(self) -> Tuple[bool, List[str]]:
        """
        Validate tính hợp lệ của series.

        Returns
        -------
        Tuple[bool, List[str]]
            (is_valid, list_of_errors)
        """
        errors = []

        # Kiểm tra UID
        if not self.series_instance_uid:
            errors.append("Series Instance UID không được để trống")

        # Kiểm tra modality
        if not self.modality:
            errors.append("Modality không được để trống")

        # Kiểm tra số lượng instance cho imaging modality
        if self.modality in [Modality.CT, Modality.MR] and len(self.instances) == 0:
            errors.append(f"Series {self.modality.value} phải có ít nhất 1 instance")

        # Kiểm tra dữ liệu ảnh nếu có
        if self.image_data is not None:
            if len(self.image_data.shape) != 3:
                errors.append("Image data phải là mảng 3D")

            if self.image_spacing and len(self.image_spacing) != 3:
                errors.append("Image spacing phải có 3 giá trị (x, y, z)")

        is_valid = len(errors) == 0

        # Cập nhật validation status
        if is_valid:
            self.metadata.validation_status = "VALIDATED"
        else:
            self.metadata.validation_status = "REJECTED"
            self.metadata.validation_notes = errors

        return is_valid, errors

    def get_summary(self) -> Dict[str, Any]:
        """
        Lấy thông tin tóm tắt series.

        Returns
        -------
        Dict[str, Any]
            Dictionary chứa thông tin tóm tắt
        """
        image_shape = self.image_data.shape if self.image_data is not None else None

        return {
            "series_number": self.series_number,
            "series_instance_uid": self.series_instance_uid,
            "patient_id": self.patient_id,
            "study_instance_uid": self.study_instance_uid,
            "series_date": self.series_date.isoformat(),
            "series_time": self.series_time.isoformat(),
            "series_description": self.series_description,
            "modality": self.modality.value,
            "status": self.status.value,
            "instance_count": len(self.instances),
            "image_shape": image_shape,
            "image_spacing": self.image_spacing,
            "validation_status": self.metadata.validation_status,
            "protocol_name": self.metadata.protocol_name,
        }

    def to_dict(self) -> Dict[str, Any]:
        """
        Chuyển đổi Series thành dictionary để serialization.

        Returns
        -------
        Dict[str, Any]
            Dictionary representation
        """
        return {
            "series_instance_uid": self.series_instance_uid,
            "series_number": self.series_number,
            "series_date": self.series_date.isoformat(),
            "series_time": self.series_time.isoformat(),
            "series_description": self.series_description,
            "modality": self.modality.value,
            "status": self.status.value,
            "study_instance_uid": self.study_instance_uid,
            "patient_id": self.patient_id,
            "metadata": self.metadata.to_dict(),
            "image_spacing": self.image_spacing,
            "image_origin": self.image_origin,
            "image_orientation": self.image_orientation,
            "created_date": self.created_date.isoformat(),
            "modified_date": self.modified_date.isoformat(),
            "created_by": self.created_by,
            "modified_by": self.modified_by,
            "instance_count": len(self.instances),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Series":
        """
        Tạo Series từ dictionary.

        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary chứa dữ liệu series

        Returns
        -------
        Series
            Series object được tạo
        """
        # Parse dates and times
        series_date = None
        if "series_date" in data:
            try:
                series_date = datetime.fromisoformat(data["series_date"]).date()
            except (ValueError, TypeError):
                series_date = date.today()

        series_time = None
        if "series_time" in data:
            try:
                series_time = datetime.fromisoformat(
                    f"2000-01-01T{data['series_time']}"
                ).time()
            except (ValueError, TypeError):
                series_time = datetime.now().time()

        # Parse enums
        modality = Modality.CT
        if "modality" in data:
            try:
                modality = Modality(data["modality"])
            except ValueError:
                pass

        status = SeriesStatus.PENDING
        if "status" in data:
            try:
                status = SeriesStatus(data["status"])
            except ValueError:
                pass

        # Parse metadata
        metadata = SeriesMetadata()
        if "metadata" in data:
            metadata = SeriesMetadata.from_dict(data["metadata"])

        series = cls(
            series_instance_uid=data.get("series_instance_uid"),
            series_number=data.get("series_number", 1),
            series_date=series_date,
            series_time=series_time,
            series_description=data.get("series_description", ""),
            modality=modality,
            status=status,
            study_instance_uid=data.get("study_instance_uid"),
            patient_id=data.get("patient_id"),
            metadata=metadata,
        )

        # Set image properties
        series.image_spacing = data.get("image_spacing")
        series.image_origin = data.get("image_origin")
        series.image_orientation = data.get("image_orientation")

        # Parse timestamps
        if "created_date" in data:
            try:
                series.created_date = datetime.fromisoformat(data["created_date"])
            except (ValueError, TypeError):
                pass

        if "modified_date" in data:
            try:
                series.modified_date = datetime.fromisoformat(data["modified_date"])
            except (ValueError, TypeError):
                pass

        series.created_by = data.get("created_by")
        series.modified_by = data.get("modified_by")

        return series

    def __str__(self) -> str:
        """String representation của Series."""
        return (
            f"Series(Number={self.series_number}, UID={self.series_instance_uid[:16]}..., "
            f"Modality={self.modality.value}, Status={self.status.value}, "
            f"Instances={len(self.instances)})"
        )

    def __repr__(self) -> str:
        """Detailed representation của Series."""
        return (
            f"Series(series_instance_uid='{self.series_instance_uid}', "
            f"series_number={self.series_number}, modality={self.modality}, "
            f"status={self.status})"
        )


def create_sample_series(
    modality: Modality = Modality.CT, study_uid: str = None, patient_id: str = "TEST001"
) -> Series:
    """
    Tạo sample series cho testing.

    Parameters
    ----------
    modality : Modality, optional
        Loại modality
    study_uid : str, optional
        Study Instance UID
    patient_id : str, optional
        ID của bệnh nhân

    Returns
    -------
    Series
        Sample series object
    """
    series = Series(
        series_number=1,
        series_date=date.today(),
        series_time=datetime.now().time(),
        series_description=f"Sample {modality.value} Series",
        modality=modality,
        status=SeriesStatus.READY,
        study_instance_uid=study_uid,
        patient_id=patient_id,
    )

    # Thêm metadata mẫu cho CT
    if modality == Modality.CT:
        series.update_metadata(
            slice_thickness=3.0,
            pixel_spacing=[0.5, 0.5],
            protocol_name="Chest CT with Contrast",
            scanning_sequence="HELICAL",
            image_quality="EXCELLENT",
        )

        # Tạo sample image data
        sample_data = np.random.randint(-1000, 1000, (50, 512, 512), dtype=np.int16)
        series.set_image_data(
            sample_data, spacing=(0.5, 0.5, 3.0), origin=(0.0, 0.0, 0.0)
        )

    series.add_processing_step("Series created for testing purposes", "Testing")

    return series


if __name__ == "__main__":
    # Test basic functionality
    logging.basicConfig(level=logging.INFO)

    print("Testing Series module...")

    # Tạo series mẫu
    series = create_sample_series()
    print(f"Created: {series}")

    # Test validation
    is_valid, errors = series.validate_series()
    print(f"Validation: {'✓' if is_valid else '✗'} {errors}")

    # Test serialization
    series_dict = series.to_dict()
    restored_series = Series.from_dict(series_dict)
    print(
        f"Serialization test: {restored_series.series_number == series.series_number}"
    )

    # Test status change
    series.set_status(SeriesStatus.COMPLETED, "Processing finished successfully")

    # Test summary
    summary = series.get_summary()
    print(f"Summary: {summary}")

    print("Series module testing completed!")
