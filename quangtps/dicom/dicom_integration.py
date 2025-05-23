"""
QuangTPS DICOM Integration Module

Module tích hợp DICOM toàn diện cho hệ thống QuangTPS.
Cung cấp import/export DICOM, validation, conversion và integration
với treatment planning workflow.
"""

import logging
import os
import tempfile
import shutil
from datetime import datetime, date
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Import DICOM processing libraries với fallback
HAS_PYDICOM = False
HAS_SIMPLEITK = False
HAS_NUMPY = False

try:
    import pydicom
    from pydicom.dataset import Dataset, FileDataset
    from pydicom.uid import generate_uid, ImplicitVRLittleEndian

    HAS_PYDICOM = True
    logger.info("PyDICOM được tải thành công")
except ImportError as e:
    logger.warning(f"PyDICOM không khả dụng: {e}")

    # Fallback class
    class Dataset:
        def __init__(self):
            self.data = {}

        def __setattr__(self, name, value):
            if name == "data":
                super().__setattr__(name, value)
            else:
                self.data[name] = value

        def __getattr__(self, name):
            return self.data.get(name, None)


try:
    import SimpleITK as sitk

    HAS_SIMPLEITK = True
    logger.info("SimpleITK được tải thành công")
except ImportError as e:
    logger.warning(f"SimpleITK không khả dụng: {e}")

    # Fallback module
    class sitk:
        @staticmethod
        def ReadImage(path):
            return None

        @staticmethod
        def WriteImage(image, path):
            pass


try:
    import numpy as np

    HAS_NUMPY = True
except ImportError as e:
    logger.warning(f"NumPy không khả dụng: {e}")

    # Fallback
    class np:
        @staticmethod
        def array(data):
            return data

        @staticmethod
        def zeros(shape):
            return [[0 for _ in range(shape[1])] for _ in range(shape[0])]


@dataclass
class DicomSeriesInfo:
    """Thông tin về DICOM series."""

    series_uid: str
    series_description: str
    modality: str
    series_number: int
    instance_count: int

    # File paths
    file_paths: List[str] = field(default_factory=list)

    # Metadata
    acquisition_date: Optional[date] = None
    acquisition_time: Optional[str] = None
    slice_thickness: Optional[float] = None
    pixel_spacing: Optional[Tuple[float, float]] = None

    # Image properties
    image_orientation: Optional[List[float]] = None
    image_position: Optional[List[float]] = None
    rows: Optional[int] = None
    columns: Optional[int] = None

    def __post_init__(self):
        """Validate series data."""
        if not self.series_uid:
            raise ValueError("Series UID là bắt buộc")


@dataclass
class DicomStudyInfo:
    """Thông tin về DICOM study."""

    study_uid: str
    study_description: str
    study_date: date
    patient_id: str
    patient_name: str

    # Series trong study
    series: Dict[str, DicomSeriesInfo] = field(default_factory=dict)

    # Metadata
    study_time: Optional[str] = None
    accession_number: Optional[str] = None
    referring_physician: Optional[str] = None
    patient_birth_date: Optional[date] = None
    patient_sex: Optional[str] = None

    def __post_init__(self):
        """Validate study data."""
        if not self.study_uid or not self.patient_id:
            raise ValueError("Study UID và Patient ID là bắt buộc")

    @property
    def total_instances(self) -> int:
        """Tổng số instances trong study."""
        return sum(series.instance_count for series in self.series.values())

    def get_series_by_modality(self, modality: str) -> List[DicomSeriesInfo]:
        """Lấy series theo modality."""
        return [
            series
            for series in self.series.values()
            if series.modality.upper() == modality.upper()
        ]


class DicomValidator:
    """Validator cho DICOM files và datasets."""

    @staticmethod
    def validate_dicom_file(file_path: str) -> Tuple[bool, List[str]]:
        """
        Validate DICOM file.

        Returns
        -------
        Tuple[bool, List[str]]
            (is_valid, error_messages)
        """
        if not HAS_PYDICOM:
            return False, ["PyDICOM không khả dụng"]

        errors = []

        try:
            # Kiểm tra file tồn tại
            if not os.path.exists(file_path):
                errors.append(f"File không tồn tại: {file_path}")
                return False, errors

            # Đọc DICOM file
            try:
                ds = pydicom.dcmread(file_path, force=True)
            except Exception as e:
                errors.append(f"Không thể đọc DICOM file: {e}")
                return False, errors

            # Kiểm tra required tags
            required_tags = [
                "StudyInstanceUID",
                "SeriesInstanceUID",
                "SOPInstanceUID",
                "Modality",
            ]

            for tag in required_tags:
                if not hasattr(ds, tag) or not getattr(ds, tag):
                    errors.append(f"Thiếu required tag: {tag}")

            # Kiểm tra Patient ID nếu có
            if hasattr(ds, "PatientID") and not ds.PatientID:
                errors.append("Patient ID trống")

            # Kiểm tra modality hợp lệ
            valid_modalities = [
                "CT",
                "MR",
                "PT",
                "RTIMAGE",
                "RTDOSE",
                "RTPLAN",
                "RTSTRUCT",
            ]
            if hasattr(ds, "Modality") and ds.Modality not in valid_modalities:
                errors.append(f"Modality không hỗ trợ: {ds.Modality}")

            # Kiểm tra pixel data cho image modalities
            image_modalities = ["CT", "MR", "PT", "RTIMAGE"]
            if (
                hasattr(ds, "Modality")
                and ds.Modality in image_modalities
                and not hasattr(ds, "PixelData")
            ):
                errors.append("Image modality nhưng thiếu pixel data")

        except Exception as e:
            errors.append(f"Lỗi validate DICOM: {e}")

        return len(errors) == 0, errors

    @staticmethod
    def validate_rt_structure_set(ds: Dataset) -> Tuple[bool, List[str]]:
        """Validate RT Structure Set."""
        errors = []

        if not hasattr(ds, "Modality") or ds.Modality != "RTSTRUCT":
            errors.append("Không phải RT Structure Set")
            return False, errors

        # Kiểm tra ROI Contour Sequence
        if not hasattr(ds, "ROIContourSequence"):
            errors.append("Thiếu ROI Contour Sequence")

        # Kiểm tra Structure Set ROI Sequence
        if not hasattr(ds, "StructureSetROISequence"):
            errors.append("Thiếu Structure Set ROI Sequence")

        return len(errors) == 0, errors

    @staticmethod
    def validate_rt_plan(ds: Dataset) -> Tuple[bool, List[str]]:
        """Validate RT Plan."""
        errors = []

        if not hasattr(ds, "Modality") or ds.Modality != "RTPLAN":
            errors.append("Không phải RT Plan")
            return False, errors

        # Kiểm tra Beam Sequence
        if not hasattr(ds, "BeamSequence"):
            errors.append("Thiếu Beam Sequence")

        # Kiểm tra Fraction Group Sequence
        if not hasattr(ds, "FractionGroupSequence"):
            errors.append("Thiếu Fraction Group Sequence")

        return len(errors) == 0, errors


class DicomParser:
    """Parser để extract thông tin từ DICOM files."""

    def __init__(self):
        self.validator = DicomValidator()

    def parse_dicom_directory(self, directory_path: str) -> Dict[str, DicomStudyInfo]:
        """
        Parse tất cả DICOM files trong directory.

        Returns
        -------
        Dict[str, DicomStudyInfo]
            Dictionary mapping study_uid -> study_info
        """
        if not HAS_PYDICOM:
            logger.error("PyDICOM không khả dụng")
            return {}

        studies = {}

        try:
            # Tìm tất cả DICOM files
            dicom_files = []
            for root, dirs, files in os.walk(directory_path):
                for file in files:
                    file_path = os.path.join(root, file)

                    # Kiểm tra extension (có thể không có .dcm)
                    if self._is_dicom_file(file_path):
                        dicom_files.append(file_path)

            logger.info(f"Tìm thấy {len(dicom_files)} DICOM files")

            # Parse từng file
            for file_path in dicom_files:
                try:
                    study_info = self.parse_dicom_file(file_path)
                    if study_info:
                        study_uid = study_info.study_uid

                        if study_uid not in studies:
                            studies[study_uid] = study_info
                        else:
                            # Merge series vào existing study
                            existing_study = studies[study_uid]
                            for series_uid, series_info in study_info.series.items():
                                if series_uid not in existing_study.series:
                                    existing_study.series[series_uid] = series_info
                                else:
                                    # Merge file paths
                                    existing_study.series[series_uid].file_paths.extend(
                                        series_info.file_paths
                                    )
                                    existing_study.series[
                                        series_uid
                                    ].instance_count += 1

                except Exception as e:
                    logger.warning(f"Lỗi parse file {file_path}: {e}")
                    continue

            # Update instance counts
            for study in studies.values():
                for series in study.series.values():
                    series.instance_count = len(series.file_paths)

            logger.info(f"Parse thành công {len(studies)} studies")
            return studies

        except Exception as e:
            logger.error(f"Lỗi parse DICOM directory: {e}")
            return {}

    def parse_dicom_file(self, file_path: str) -> Optional[DicomStudyInfo]:
        """Parse single DICOM file."""
        if not HAS_PYDICOM:
            return None

        try:
            # Validate file
            is_valid, errors = self.validator.validate_dicom_file(file_path)
            if not is_valid:
                logger.warning(f"DICOM file không hợp lệ {file_path}: {errors}")
                # Vẫn cố gắng parse nếu chỉ có warning

            # Read DICOM
            ds = pydicom.dcmread(file_path, force=True)

            # Extract study info
            study_uid = ds.get("StudyInstanceUID", "")
            study_description = ds.get("StudyDescription", "Unknown Study")

            # Parse date
            study_date_str = ds.get("StudyDate", "")
            try:
                study_date = datetime.strptime(study_date_str, "%Y%m%d").date()
            except:
                study_date = date.today()

            patient_id = ds.get("PatientID", "Unknown")
            patient_name = ds.get("PatientName", "Unknown")

            # Create study info
            study_info = DicomStudyInfo(
                study_uid=study_uid,
                study_description=study_description,
                study_date=study_date,
                patient_id=patient_id,
                patient_name=str(patient_name),
                study_time=ds.get("StudyTime"),
                accession_number=ds.get("AccessionNumber"),
                referring_physician=ds.get("ReferringPhysicianName"),
                patient_sex=ds.get("PatientSex"),
            )

            # Parse patient birth date
            birth_date_str = ds.get("PatientBirthDate", "")
            if birth_date_str:
                try:
                    study_info.patient_birth_date = datetime.strptime(
                        birth_date_str, "%Y%m%d"
                    ).date()
                except:
                    pass

            # Extract series info
            series_uid = ds.get("SeriesInstanceUID", "")
            series_description = ds.get("SeriesDescription", "Unknown Series")
            modality = ds.get("Modality", "Unknown")
            series_number = ds.get("SeriesNumber", 0)

            # Extract image properties
            slice_thickness = ds.get("SliceThickness")
            pixel_spacing = ds.get("PixelSpacing")
            if pixel_spacing:
                pixel_spacing = tuple(float(x) for x in pixel_spacing)

            image_orientation = ds.get("ImageOrientationPatient")
            if image_orientation:
                image_orientation = [float(x) for x in image_orientation]

            image_position = ds.get("ImagePositionPatient")
            if image_position:
                image_position = [float(x) for x in image_position]

            # Create series info
            series_info = DicomSeriesInfo(
                series_uid=series_uid,
                series_description=series_description,
                modality=modality,
                series_number=int(series_number) if series_number else 0,
                instance_count=1,
                file_paths=[file_path],
                slice_thickness=float(slice_thickness) if slice_thickness else None,
                pixel_spacing=pixel_spacing,
                image_orientation=image_orientation,
                image_position=image_position,
                rows=ds.get("Rows"),
                columns=ds.get("Columns"),
            )

            # Add series to study
            study_info.series[series_uid] = series_info

            return study_info

        except Exception as e:
            logger.error(f"Lỗi parse DICOM file {file_path}: {e}")
            return None

    def _is_dicom_file(self, file_path: str) -> bool:
        """Kiểm tra xem file có phải DICOM không."""
        if not HAS_PYDICOM:
            return False

        try:
            # Kiểm tra extension
            ext = Path(file_path).suffix.lower()
            if ext in [".dcm", ".dicom"]:
                return True

            # Kiểm tra magic number (nhanh)
            with open(file_path, "rb") as f:
                f.seek(128)  # Skip preamble
                dicm = f.read(4)
                if dicm == b"DICM":
                    return True

            # Thử đọc với pydicom (chậm hơn)
            try:
                pydicom.dcmread(file_path, stop_before_pixels=True)
                return True
            except:
                return False

        except:
            return False


class DicomConverter:
    """Converter để chuyển đổi DICOM data sang format khác."""

    def __init__(self):
        self.parser = DicomParser()

    def dicom_to_numpy(self, dicom_files: List[str]) -> Optional[np.ndarray]:
        """Convert DICOM series thành numpy array."""
        if not HAS_PYDICOM or not HAS_NUMPY:
            logger.error("PyDICOM hoặc NumPy không khả dụng")
            return None

        try:
            # Sort files theo instance number hoặc position
            sorted_files = self._sort_dicom_files(dicom_files)

            # Đọc first file để lấy dimensions
            first_ds = pydicom.dcmread(sorted_files[0])
            rows = first_ds.Rows
            cols = first_ds.Columns

            # Tạo 3D array
            volume = np.zeros((len(sorted_files), rows, cols))

            for i, file_path in enumerate(sorted_files):
                ds = pydicom.dcmread(file_path)

                # Apply rescale slope/intercept nếu có
                pixel_array = ds.pixel_array.astype(np.float32)

                if hasattr(ds, "RescaleSlope") and hasattr(ds, "RescaleIntercept"):
                    slope = float(ds.RescaleSlope)
                    intercept = float(ds.RescaleIntercept)
                    pixel_array = pixel_array * slope + intercept

                volume[i] = pixel_array

            return volume

        except Exception as e:
            logger.error(f"Lỗi convert DICOM to numpy: {e}")
            return None

    def dicom_to_sitk(self, dicom_files: List[str]) -> Optional["sitk.Image"]:
        """Convert DICOM series thành SimpleITK Image."""
        if not HAS_SIMPLEITK:
            logger.error("SimpleITK không khả dụng")
            return None

        try:
            # Sort files
            sorted_files = self._sort_dicom_files(dicom_files)

            # Sử dụng SimpleITK ImageSeriesReader
            reader = sitk.ImageSeriesReader()
            reader.SetFileNames(sorted_files)
            reader.MetaDataDictionaryArrayUpdateOn()
            reader.LoadPrivateTagsOn()

            image = reader.Execute()
            return image

        except Exception as e:
            logger.error(f"Lỗi convert DICOM to SimpleITK: {e}")
            return None

    def _sort_dicom_files(self, dicom_files: List[str]) -> List[str]:
        """Sort DICOM files theo thứ tự đúng."""
        if not HAS_PYDICOM:
            return dicom_files

        try:
            file_info = []

            for file_path in dicom_files:
                try:
                    ds = pydicom.dcmread(file_path, stop_before_pixels=True)

                    # Thử instance number trước
                    sort_key = ds.get("InstanceNumber", 0)

                    # Nếu không có instance number, dùng image position
                    if sort_key == 0 and hasattr(ds, "ImagePositionPatient"):
                        pos = ds.ImagePositionPatient
                        sort_key = float(pos[2])  # Z coordinate

                    file_info.append((sort_key, file_path))

                except Exception as e:
                    logger.warning(f"Lỗi sort file {file_path}: {e}")
                    file_info.append((0, file_path))

            # Sort theo sort_key
            file_info.sort(key=lambda x: x[0])

            return [file_path for _, file_path in file_info]

        except Exception as e:
            logger.error(f"Lỗi sort DICOM files: {e}")
            return dicom_files


class DicomExporter:
    """Exporter để tạo DICOM files từ internal data."""

    def __init__(self):
        self.temp_dir = tempfile.mkdtemp(prefix="quangtps_dicom_")

    def __del__(self):
        """Cleanup temp directory."""
        try:
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
        except:
            pass

    def export_rt_dose(
        self,
        dose_array: np.ndarray,
        reference_dicom: str,
        output_path: str,
        dose_units: str = "GY",
    ) -> bool:
        """Export dose array as RT Dose DICOM."""
        if not HAS_PYDICOM or not HAS_NUMPY:
            logger.error("PyDICOM hoặc NumPy không khả dụng")
            return False

        try:
            # Đọc reference DICOM để lấy metadata
            ref_ds = pydicom.dcmread(reference_dicom)

            # Tạo RT Dose dataset
            ds = Dataset()

            # Required metadata
            ds.Modality = "RTDOSE"
            ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.481.2"  # RT Dose Storage
            ds.SOPInstanceUID = generate_uid()
            ds.StudyInstanceUID = ref_ds.get("StudyInstanceUID", generate_uid())
            ds.SeriesInstanceUID = generate_uid()
            ds.InstanceNumber = 1

            # Patient info
            ds.PatientName = ref_ds.get("PatientName", "QuangTPS Patient")
            ds.PatientID = ref_ds.get("PatientID", "QTP001")
            ds.PatientBirthDate = ref_ds.get("PatientBirthDate", "")
            ds.PatientSex = ref_ds.get("PatientSex", "")

            # Study info
            ds.StudyDate = datetime.now().strftime("%Y%m%d")
            ds.StudyTime = datetime.now().strftime("%H%M%S")
            ds.StudyDescription = "QuangTPS Dose Calculation"

            # Series info
            ds.SeriesDate = ds.StudyDate
            ds.SeriesTime = ds.StudyTime
            ds.SeriesDescription = "Dose Distribution"
            ds.SeriesNumber = 1

            # Image info
            ds.Rows = dose_array.shape[1]
            ds.Columns = dose_array.shape[2]
            ds.NumberOfFrames = dose_array.shape[0]

            # Dose specific tags
            ds.DoseUnits = dose_units
            ds.DoseType = "PHYSICAL"
            ds.DoseSummationType = "PLAN"

            # Grid scaling
            dose_max = np.max(dose_array)
            if dose_max > 0:
                ds.DoseGridScaling = dose_max / 65535.0  # Scale to 16-bit
            else:
                ds.DoseGridScaling = 1.0

            # Convert dose to pixel data
            if ds.DoseGridScaling > 0:
                pixel_array = (dose_array / ds.DoseGridScaling).astype(np.uint16)
            else:
                pixel_array = dose_array.astype(np.uint16)

            ds.PixelData = pixel_array.tobytes()
            ds.BitsAllocated = 16
            ds.BitsStored = 16
            ds.HighBit = 15
            ds.PixelRepresentation = 0

            # Spatial info (copy từ reference nếu có)
            if hasattr(ref_ds, "PixelSpacing"):
                ds.PixelSpacing = ref_ds.PixelSpacing
            if hasattr(ref_ds, "SliceThickness"):
                ds.SliceThickness = ref_ds.SliceThickness
            if hasattr(ref_ds, "ImageOrientationPatient"):
                ds.ImageOrientationPatient = ref_ds.ImageOrientationPatient
            if hasattr(ref_ds, "ImagePositionPatient"):
                ds.ImagePositionPatient = ref_ds.ImagePositionPatient

            # Write file
            file_meta = Dataset()
            file_meta.MediaStorageSOPClassUID = ds.SOPClassUID
            file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
            file_meta.ImplementationClassUID = generate_uid()
            file_meta.TransferSyntaxUID = ImplicitVRLittleEndian

            file_ds = FileDataset(
                output_path,
                ds,
                file_meta=file_meta,
                preamble=b"\0" * 128,
                is_implicit_VR=True,
            )

            file_ds.save_as(output_path)

            logger.info(f"Exported RT Dose DICOM: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Lỗi export RT Dose: {e}")
            return False

    def export_rt_plan(self, plan_data: Dict[str, Any], output_path: str) -> bool:
        """Export treatment plan as RT Plan DICOM."""
        if not HAS_PYDICOM:
            logger.error("PyDICOM không khả dụng")
            return False

        try:
            # TODO: Implement RT Plan export
            # Tạm thời placeholder
            logger.info(f"RT Plan export placeholder: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Lỗi export RT Plan: {e}")
            return False


class DicomIntegration:
    """
    Main integration class cho DICOM operations.
    """

    def __init__(self, temp_dir: Optional[str] = None):
        self.parser = DicomParser()
        self.converter = DicomConverter()
        self.exporter = DicomExporter()
        self.validator = DicomValidator()

        self.temp_dir = temp_dir or tempfile.mkdtemp(prefix="quangtps_dicom_")
        Path(self.temp_dir).mkdir(parents=True, exist_ok=True)

        logger.info(f"DICOM Integration khởi tạo tại: {self.temp_dir}")

    def import_dicom_directory(
        self, directory_path: str, patient_id: Optional[str] = None
    ) -> Dict[str, DicomStudyInfo]:
        """
        Import tất cả DICOM files từ directory.

        Parameters
        ----------
        directory_path : str
            Path đến directory chứa DICOM files
        patient_id : str, optional
            Filter theo patient ID cụ thể

        Returns
        -------
        Dict[str, DicomStudyInfo]
            Dictionary mapping study_uid -> study_info
        """
        try:
            studies = self.parser.parse_dicom_directory(directory_path)

            # Filter theo patient ID nếu có
            if patient_id:
                filtered_studies = {}
                for study_uid, study_info in studies.items():
                    if study_info.patient_id == patient_id:
                        filtered_studies[study_uid] = study_info
                studies = filtered_studies

            logger.info(f"Import {len(studies)} studies từ {directory_path}")
            return studies

        except Exception as e:
            logger.error(f"Lỗi import DICOM directory: {e}")
            return {}

    def validate_dicom_directory(self, directory_path: str) -> Dict[str, Any]:
        """
        Validate tất cả DICOM files trong directory.

        Returns
        -------
        Dict[str, Any]
            Validation results
        """
        results = {
            "total_files": 0,
            "valid_files": 0,
            "invalid_files": 0,
            "errors": [],
            "warnings": [],
        }

        try:
            # Tìm tất cả files
            all_files = []
            for root, dirs, files in os.walk(directory_path):
                for file in files:
                    all_files.append(os.path.join(root, file))

            results["total_files"] = len(all_files)

            # Validate từng file
            for file_path in all_files:
                is_valid, errors = self.validator.validate_dicom_file(file_path)

                if is_valid:
                    results["valid_files"] += 1
                else:
                    results["invalid_files"] += 1
                    results["errors"].extend([f"{file_path}: {err}" for err in errors])

            logger.info(
                f"Validation results: {results['valid_files']}/{results['total_files']} valid"
            )
            return results

        except Exception as e:
            logger.error(f"Lỗi validate directory: {e}")
            results["errors"].append(str(e))
            return results

    def convert_series_to_volume(
        self, series_info: DicomSeriesInfo
    ) -> Optional[np.ndarray]:
        """
        Convert DICOM series thành volume data.

        Parameters
        ----------
        series_info : DicomSeriesInfo
            Series information

        Returns
        -------
        np.ndarray or None
            Volume data as 3D numpy array
        """
        try:
            if not series_info.file_paths:
                logger.error("Không có file paths trong series")
                return None

            volume = self.converter.dicom_to_numpy(series_info.file_paths)

            if volume is not None:
                logger.info(
                    f"Convert series {series_info.series_description} thành volume: {volume.shape}"
                )

            return volume

        except Exception as e:
            logger.error(f"Lỗi convert series to volume: {e}")
            return None

    def get_series_metadata(self, series_info: DicomSeriesInfo) -> Dict[str, Any]:
        """Lấy metadata của series."""
        metadata = {
            "series_uid": series_info.series_uid,
            "series_description": series_info.series_description,
            "modality": series_info.modality,
            "series_number": series_info.series_number,
            "instance_count": series_info.instance_count,
            "slice_thickness": series_info.slice_thickness,
            "pixel_spacing": series_info.pixel_spacing,
            "dimensions": None,
            "file_size_mb": 0,
        }

        try:
            # Tính tổng file size
            total_size = 0
            for file_path in series_info.file_paths:
                if os.path.exists(file_path):
                    total_size += os.path.getsize(file_path)
            metadata["file_size_mb"] = total_size / (1024 * 1024)

            # Lấy dimensions từ first file
            if series_info.file_paths and HAS_PYDICOM:
                first_file = series_info.file_paths[0]
                ds = pydicom.dcmread(first_file, stop_before_pixels=True)

                if hasattr(ds, "Rows") and hasattr(ds, "Columns"):
                    metadata["dimensions"] = (
                        len(series_info.file_paths),
                        ds.Rows,
                        ds.Columns,
                    )

        except Exception as e:
            logger.warning(f"Lỗi lấy metadata: {e}")

        return metadata

    def cleanup(self):
        """Cleanup temp files."""
        try:
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                logger.info("Cleanup DICOM temp files")
        except Exception as e:
            logger.warning(f"Lỗi cleanup: {e}")


# Factory function
def create_dicom_integration(temp_dir: Optional[str] = None) -> DicomIntegration:
    """Factory function để tạo DicomIntegration."""
    return DicomIntegration(temp_dir)


# Utility functions
def get_dicom_capabilities() -> Dict[str, bool]:
    """Kiểm tra khả năng DICOM của hệ thống."""
    return {
        "pydicom_available": HAS_PYDICOM,
        "simpleitk_available": HAS_SIMPLEITK,
        "numpy_available": HAS_NUMPY,
        "full_capability": HAS_PYDICOM and HAS_SIMPLEITK and HAS_NUMPY,
    }


def check_dicom_requirements() -> Tuple[bool, List[str]]:
    """Kiểm tra requirements cho DICOM processing."""
    missing = []

    if not HAS_PYDICOM:
        missing.append("pydicom")
    if not HAS_SIMPLEITK:
        missing.append("SimpleITK")
    if not HAS_NUMPY:
        missing.append("numpy")

    return len(missing) == 0, missing


if __name__ == "__main__":
    # Test code
    logging.basicConfig(level=logging.INFO)

    # Kiểm tra capabilities
    caps = get_dicom_capabilities()
    print(f"DICOM Capabilities: {caps}")

    has_requirements, missing = check_dicom_requirements()
    if not has_requirements:
        print(f"Thiếu requirements: {missing}")
    else:
        print("Tất cả requirements đã sẵn sàng")

        # Tạo DICOM integration
        dicom_integration = create_dicom_integration()
        print("DICOM Integration test hoàn thành!")
