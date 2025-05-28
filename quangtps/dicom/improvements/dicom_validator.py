"""
Module validation DICOM để đảm bảo tính toàn vẹn dữ liệu.

Provides comprehensive DICOM validation including:
- DICOM standard compliance checking
- Data integrity validation
- Structure consistency verification
- Dose data validation
- Plan validation
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

# DICOM imports with fallbacks
try:
    import pydicom
    from pydicom import Dataset
    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False
    # Fallback classes
    class Dataset:
        pass

logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Kết quả validation DICOM."""

    is_valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    info: List[str] = field(default_factory=list)

    # Detailed results
    modality_valid: bool = True
    structure_valid: bool = True
    dose_valid: bool = True
    plan_valid: bool = True

    # Statistics
    total_files: int = 0
    valid_files: int = 0

    def add_error(self, message: str):
        """Thêm error message."""
        self.errors.append(message)
        self.is_valid = False
        logger.error(f"DICOM Validation Error: {message}")

    def add_warning(self, message: str):
        """Thêm warning message."""
        self.warnings.append(message)
        logger.warning(f"DICOM Validation Warning: {message}")

    def add_info(self, message: str):
        """Thêm info message."""
        self.info.append(message)
        logger.info(f"DICOM Validation Info: {message}")

class DICOMValidator:
    """Validator cho DICOM files và datasets."""

    def __init__(self):
        self.required_tags = {
            'CT': ['StudyInstanceUID', 'SeriesInstanceUID', 'SOPInstanceUID',
                   'ImagePositionPatient', 'ImageOrientationPatient', 'PixelSpacing'],
            'RTSTRUCT': ['StudyInstanceUID', 'SeriesInstanceUID', 'SOPInstanceUID',
                        'StructureSetROISequence', 'ROIContourSequence'],
            'RTDOSE': ['StudyInstanceUID', 'SeriesInstanceUID', 'SOPInstanceUID',
                      'DoseGridScaling', 'PixelData', 'ImagePositionPatient'],
            'RTPLAN': ['StudyInstanceUID', 'SeriesInstanceUID', 'SOPInstanceUID',
                      'BeamSequence', 'FractionGroupSequence']
        }

        self.modality_checks = {
            'CT': self._validate_ct,
            'RTSTRUCT': self._validate_rtstruct,
            'RTDOSE': self._validate_rtdose,
            'RTPLAN': self._validate_rtplan
        }

        logger.info("DICOMValidator initialized")

    def validate_file(self, file_path: str) -> ValidationResult:
        """Validate single DICOM file."""
        result = ValidationResult()
        result.total_files = 1

        if not HAS_PYDICOM:
            result.add_error("PyDICOM not available for validation")
            return result

        try:
            # Read DICOM file
            dataset = pydicom.dcmread(file_path, force=True)

            # Basic validation
            if not self._validate_basic_dicom(dataset, result):
                return result

            # Modality-specific validation
            modality = getattr(dataset, 'Modality', None)
            if modality in self.modality_checks:
                self.modality_checks[modality](dataset, result)
            else:
                result.add_warning(f"Unknown modality: {modality}")

            if result.is_valid:
                result.valid_files = 1
                result.add_info(f"File {file_path} is valid")

        except Exception as e:
            result.add_error(f"Failed to read DICOM file {file_path}: {str(e)}")

        return result

    def validate_series(self, file_paths: List[str]) -> ValidationResult:
        """Validate DICOM series."""
        result = ValidationResult()
        result.total_files = len(file_paths)

        if not file_paths:
            result.add_error("No files provided for validation")
            return result

        datasets = []
        valid_count = 0

        # Validate individual files
        for file_path in file_paths:
            file_result = self.validate_file(file_path)

            # Merge results
            result.errors.extend(file_result.errors)
            result.warnings.extend(file_result.warnings)
            result.info.extend(file_result.info)

            if file_result.is_valid:
                valid_count += 1
                try:
                    dataset = pydicom.dcmread(file_path, force=True)
                    datasets.append(dataset)
                except:
                    pass

        result.valid_files = valid_count

        # Series consistency validation
        if datasets:
            self._validate_series_consistency(datasets, result)

        # Overall validity
        if result.errors:
            result.is_valid = False

        return result

    def validate_study(self, study_datasets: Dict[str, List]) -> ValidationResult:
        """Validate complete study với multiple series."""
        result = ValidationResult()

        # Count total files
        total_files = sum(len(series) for series in study_datasets.values())
        result.total_files = total_files

        if total_files == 0:
            result.add_error("No datasets provided for study validation")
            return result

        # Validate each series
        valid_count = 0
        for modality, datasets in study_datasets.items():
            if datasets:
                # Convert to file paths if needed
                if isinstance(datasets[0], str):
                    series_result = self.validate_series(datasets)
                else:
                    # Direct dataset validation
                    series_result = self._validate_dataset_series(datasets)

                # Merge results
                result.errors.extend(series_result.errors)
                result.warnings.extend(series_result.warnings)
                result.info.extend(series_result.info)
                valid_count += series_result.valid_files

        result.valid_files = valid_count

        # Study-level validation
        self._validate_study_consistency(study_datasets, result)

        # Overall validity
        if result.errors:
            result.is_valid = False

        return result

    def _validate_basic_dicom(self, dataset: Dataset, result: ValidationResult) -> bool:
        """Basic DICOM validation."""
        try:
            # Check if it's a valid DICOM file
            if not hasattr(dataset, 'file_meta'):
                result.add_error("Not a valid DICOM file - missing file meta information")
                return False

            # Check required basic tags
            required_basic = ['StudyInstanceUID', 'SeriesInstanceUID', 'SOPInstanceUID']
            for tag in required_basic:
                if not hasattr(dataset, tag):
                    result.add_error(f"Missing required tag: {tag}")
                    return False

            # Check modality
            if not hasattr(dataset, 'Modality'):
                result.add_error("Missing Modality tag")
                return False

            return True

        except Exception as e:
            result.add_error(f"Basic DICOM validation failed: {str(e)}")
            return False

    def _validate_ct(self, dataset: Dataset, result: ValidationResult):
        """Validate CT dataset."""
        modality = getattr(dataset, 'Modality', '')
        if modality != 'CT':
            result.add_error(f"Expected CT modality, got {modality}")
            return

        # Check required CT tags
        required_tags = self.required_tags['CT']
        for tag in required_tags:
            if not hasattr(dataset, tag):
                result.add_error(f"CT: Missing required tag {tag}")

        # Check image data
        if hasattr(dataset, 'PixelData'):
            try:
                pixel_array = dataset.pixel_array
                if pixel_array.size == 0:
                    result.add_error("CT: Empty pixel data")
                else:
                    result.add_info(f"CT: Image size {pixel_array.shape}")
            except Exception as e:
                result.add_error(f"CT: Cannot read pixel data - {str(e)}")
        else:
            result.add_error("CT: Missing pixel data")

        # Check geometry
        if hasattr(dataset, 'ImagePositionPatient') and hasattr(dataset, 'ImageOrientationPatient'):
            try:
                position = dataset.ImagePositionPatient
                orientation = dataset.ImageOrientationPatient
                if len(position) != 3 or len(orientation) != 6:
                    result.add_error("CT: Invalid geometry information")
            except:
                result.add_error("CT: Cannot parse geometry information")

    def _validate_rtstruct(self, dataset: Dataset, result: ValidationResult):
        """Validate RT Structure Set."""
        modality = getattr(dataset, 'Modality', '')
        if modality != 'RTSTRUCT':
            result.add_error(f"Expected RTSTRUCT modality, got {modality}")
            return

        # Check required RTSTRUCT tags
        required_tags = self.required_tags['RTSTRUCT']
        for tag in required_tags:
            if not hasattr(dataset, tag):
                result.add_error(f"RTSTRUCT: Missing required tag {tag}")

        # Validate structure data
        if hasattr(dataset, 'StructureSetROISequence'):
            roi_sequence = dataset.StructureSetROISequence
            result.add_info(f"RTSTRUCT: Found {len(roi_sequence)} ROIs")

            # Check each ROI
            for i, roi in enumerate(roi_sequence):
                if not hasattr(roi, 'ROINumber'):
                    result.add_error(f"RTSTRUCT: ROI {i} missing ROINumber")
                if not hasattr(roi, 'ROIName'):
                    result.add_warning(f"RTSTRUCT: ROI {i} missing ROIName")

        # Validate contour data
        if hasattr(dataset, 'ROIContourSequence'):
            contour_sequence = dataset.ROIContourSequence
            for i, contour in enumerate(contour_sequence):
                if hasattr(contour, 'ContourSequence'):
                    contour_count = len(contour.ContourSequence)
                    result.add_info(f"RTSTRUCT: ROI {i} has {contour_count} contours")

    def _validate_rtdose(self, dataset: Dataset, result: ValidationResult):
        """Validate RT Dose."""
        modality = getattr(dataset, 'Modality', '')
        if modality != 'RTDOSE':
            result.add_error(f"Expected RTDOSE modality, got {modality}")
            return

        # Check required RTDOSE tags
        required_tags = self.required_tags['RTDOSE']
        for tag in required_tags:
            if not hasattr(dataset, tag):
                result.add_error(f"RTDOSE: Missing required tag {tag}")

        # Validate dose data
        if hasattr(dataset, 'PixelData'):
            try:
                dose_array = dataset.pixel_array
                if dose_array.size == 0:
                    result.add_error("RTDOSE: Empty dose data")
                else:
                    # Check dose scaling
                    if hasattr(dataset, 'DoseGridScaling'):
                        scaling = float(dataset.DoseGridScaling)
                        max_dose = np.max(dose_array) * scaling
                        result.add_info(f"RTDOSE: Max dose {max_dose:.2f} Gy")
                    else:
                        result.add_error("RTDOSE: Missing DoseGridScaling")
            except Exception as e:
                result.add_error(f"RTDOSE: Cannot read dose data - {str(e)}")

        # Check dose units
        if hasattr(dataset, 'DoseUnits'):
            units = dataset.DoseUnits
            if units not in ['GY', 'RELATIVE']:
                result.add_warning(f"RTDOSE: Unusual dose units: {units}")
        else:
            result.add_error("RTDOSE: Missing DoseUnits")

    def _validate_rtplan(self, dataset: Dataset, result: ValidationResult):
        """Validate RT Plan."""
        modality = getattr(dataset, 'Modality', '')
        if modality != 'RTPLAN':
            result.add_error(f"Expected RTPLAN modality, got {modality}")
            return

        # Check required RTPLAN tags
        required_tags = self.required_tags['RTPLAN']
        for tag in required_tags:
            if not hasattr(dataset, tag):
                result.add_error(f"RTPLAN: Missing required tag {tag}")

        # Validate beam sequence
        if hasattr(dataset, 'BeamSequence'):
            beam_sequence = dataset.BeamSequence
            result.add_info(f"RTPLAN: Found {len(beam_sequence)} beams")

            for i, beam in enumerate(beam_sequence):
                if not hasattr(beam, 'BeamNumber'):
                    result.add_error(f"RTPLAN: Beam {i} missing BeamNumber")
                if not hasattr(beam, 'BeamName'):
                    result.add_warning(f"RTPLAN: Beam {i} missing BeamName")

        # Validate fraction groups
        if hasattr(dataset, 'FractionGroupSequence'):
            fraction_groups = dataset.FractionGroupSequence
            result.add_info(f"RTPLAN: Found {len(fraction_groups)} fraction groups")

    def _validate_series_consistency(self, datasets: List[Dataset], result: ValidationResult):
        """Validate consistency within a series."""
        if len(datasets) < 2:
            return

        # Check StudyInstanceUID consistency
        study_uids = set()
        series_uids = set()
        modalities = set()

        for dataset in datasets:
            if hasattr(dataset, 'StudyInstanceUID'):
                study_uids.add(dataset.StudyInstanceUID)
            if hasattr(dataset, 'SeriesInstanceUID'):
                series_uids.add(dataset.SeriesInstanceUID)
            if hasattr(dataset, 'Modality'):
                modalities.add(dataset.Modality)

        if len(study_uids) > 1:
            result.add_error(f"Series contains multiple studies: {len(study_uids)}")

        if len(series_uids) > 1:
            result.add_warning(f"Multiple series UIDs in same series: {len(series_uids)}")

        if len(modalities) > 1:
            result.add_error(f"Series contains multiple modalities: {modalities}")

        # CT-specific consistency checks
        if 'CT' in modalities and len(datasets) > 1:
            self._validate_ct_series_consistency(datasets, result)

    def _validate_ct_series_consistency(self, datasets: List[Dataset], result: ValidationResult):
        """Validate CT series consistency."""
        positions = []
        spacings = []

        for dataset in datasets:
            if hasattr(dataset, 'ImagePositionPatient'):
                positions.append(dataset.ImagePositionPatient)
            if hasattr(dataset, 'PixelSpacing'):
                spacings.append(dataset.PixelSpacing)

        # Check pixel spacing consistency
        if spacings:
            first_spacing = spacings[0]
            for spacing in spacings[1:]:
                if abs(spacing[0] - first_spacing[0]) > 0.01 or abs(spacing[1] - first_spacing[1]) > 0.01:
                    result.add_warning("CT series has inconsistent pixel spacing")
                    break

        # Check slice spacing
        if len(positions) > 2:
            z_positions = [pos[2] for pos in positions]
            z_positions.sort()

            spacings = [z_positions[i+1] - z_positions[i] for i in range(len(z_positions)-1)]
            avg_spacing = np.mean(spacings)

            for spacing in spacings:
                if abs(spacing - avg_spacing) > 0.1:  # 0.1mm tolerance
                    result.add_warning("CT series has irregular slice spacing")
                    break

    def _validate_study_consistency(self, study_datasets: Dict[str, List], result: ValidationResult):
        """Validate consistency across study."""
        # Extract all datasets
        all_datasets = []
        for datasets in study_datasets.values():
            if isinstance(datasets[0], str):
                # File paths - skip for now
                continue
            all_datasets.extend(datasets)

        if len(all_datasets) < 2:
            return

        # Check StudyInstanceUID consistency
        study_uids = set()
        for dataset in all_datasets:
            if hasattr(dataset, 'StudyInstanceUID'):
                study_uids.add(dataset.StudyInstanceUID)

        if len(study_uids) > 1:
            result.add_error(f"Study contains multiple StudyInstanceUIDs: {len(study_uids)}")

        # Check for required modalities
        modalities = set()
        for modality in study_datasets.keys():
            if study_datasets[modality]:  # Non-empty
                modalities.add(modality)

        if 'CT' not in modalities:
            result.add_warning("Study missing CT images")

        result.add_info(f"Study contains modalities: {', '.join(modalities)}")

    def _validate_dataset_series(self, datasets: List[Dataset]) -> ValidationResult:
        """Validate series of datasets directly."""
        result = ValidationResult()
        result.total_files = len(datasets)

        valid_count = 0
        for i, dataset in enumerate(datasets):
            try:
                if self._validate_basic_dicom(dataset, result):
                    valid_count += 1

                    # Modality-specific validation
                    modality = getattr(dataset, 'Modality', None)
                    if modality in self.modality_checks:
                        self.modality_checks[modality](dataset, result)
            except Exception as e:
                result.add_error(f"Dataset {i} validation failed: {str(e)}")

        result.valid_files = valid_count

        # Series consistency
        if valid_count > 1:
            self._validate_series_consistency(datasets, result)

        return result

def validate_dicom_file(file_path: str) -> ValidationResult:
    """Convenience function để validate single DICOM file."""
    validator = DICOMValidator()
    return validator.validate_file(file_path)

def validate_dicom_series(file_paths: List[str]) -> ValidationResult:
    """Convenience function để validate DICOM series."""
    validator = DICOMValidator()
    return validator.validate_series(file_paths)

def validate_dicom_study(study_datasets: Dict[str, List]) -> ValidationResult:
    """Convenience function để validate DICOM study."""
    validator = DICOMValidator()
    return validator.validate_study(study_datasets)