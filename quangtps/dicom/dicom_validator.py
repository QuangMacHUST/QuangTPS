"""
Xác thực dữ liệu DICOM.

Module này cung cấp các công cụ để xác thực tính hợp lệ của dữ liệu DICOM,
đảm bảo dữ liệu có đủ thông tin cần thiết và đúng định dạng.
"""

import logging
import pydicom
from pydicom.errors import InvalidDicomError
from typing import List, Dict, Any, Union, Optional, Tuple, Set

from quangtps.core.exceptions import DicomError, ValidationError
from quangtps.core.constants import Constants

logger = logging.getLogger(__name__)

class DicomValidator:
    """Lớp xác thực tính hợp lệ của dữ liệu DICOM"""
    
    @staticmethod
    def validate_ct_dataset(dataset: pydicom.dataset.FileDataset) -> bool:
        """
        Xác thực tính hợp lệ của dataset CT.
        
        Parameters
        ----------
        dataset : pydicom.dataset.FileDataset
            Dataset DICOM CT cần xác thực
            
        Returns
        -------
        bool
            True nếu dataset hợp lệ
            
        Raises
        ------
        ValidationError
            Nếu dataset không hợp lệ
        """
        # Kiểm tra modality
        if not hasattr(dataset, 'Modality') or dataset.Modality != 'CT':
            raise ValidationError("Dataset is not a CT dataset")
        
        # Kiểm tra các thuộc tính bắt buộc
        required_attributes = [
            'PatientID',
            'PatientName',
            'StudyInstanceUID',
            'SeriesInstanceUID',
            'SOPInstanceUID',
            'PixelSpacing',
            'ImagePositionPatient',
            'ImageOrientationPatient',
            'SliceThickness',
            'RescaleIntercept',
            'RescaleSlope'
        ]
        
        missing_attributes = []
        for attr in required_attributes:
            if not hasattr(dataset, attr):
                missing_attributes.append(attr)
        
        if missing_attributes:
            raise ValidationError(f"Missing required attributes: {', '.join(missing_attributes)}")
        
        # Kiểm tra dữ liệu hình ảnh
        if not hasattr(dataset, 'pixel_array'):
            raise ValidationError("CT dataset does not contain image data")
        
        return True
    
    @staticmethod
    def validate_rt_structure_dataset(dataset: pydicom.dataset.FileDataset) -> bool:
        """
        Xác thực tính hợp lệ của dataset RT Structure.
        
        Parameters
        ----------
        dataset : pydicom.dataset.FileDataset
            Dataset DICOM RT Structure cần xác thực
            
        Returns
        -------
        bool
            True nếu dataset hợp lệ
            
        Raises
        ------
        ValidationError
            Nếu dataset không hợp lệ
        """
        # Kiểm tra modality
        if not hasattr(dataset, 'Modality') or dataset.Modality != 'RTSTRUCT':
            raise ValidationError("Dataset is not an RT Structure dataset")
        
        # Kiểm tra các thuộc tính bắt buộc
        required_attributes = [
            'PatientID',
            'PatientName',
            'StudyInstanceUID',
            'SeriesInstanceUID',
            'SOPInstanceUID',
            'StructureSetLabel',
            'StructureSetROISequence',
            'ROIContourSequence'
        ]
        
        missing_attributes = []
        for attr in required_attributes:
            if not hasattr(dataset, attr):
                missing_attributes.append(attr)
        
        if missing_attributes:
            raise ValidationError(f"Missing required attributes: {', '.join(missing_attributes)}")
        
        # Kiểm tra bổ sung cho StructureSetROISequence và ROIContourSequence
        try:
            # Kiểm tra StructureSetROISequence
            if len(dataset.StructureSetROISequence) == 0:
                raise ValidationError("StructureSetROISequence is empty")
            
            # Kiểm tra ROIContourSequence
            if len(dataset.ROIContourSequence) == 0:
                raise ValidationError("ROIContourSequence is empty")
            
            # Kiểm tra tham chiếu ROI
            roi_numbers = set()
            for roi in dataset.StructureSetROISequence:
                if not hasattr(roi, 'ROINumber'):
                    raise ValidationError("ROINumber missing in StructureSetROISequence")
                roi_numbers.add(roi.ROINumber)
            
            for roi_contour in dataset.ROIContourSequence:
                if not hasattr(roi_contour, 'ReferencedROINumber'):
                    raise ValidationError("ReferencedROINumber missing in ROIContourSequence")
                if roi_contour.ReferencedROINumber not in roi_numbers:
                    raise ValidationError(f"ReferencedROINumber {roi_contour.ReferencedROINumber} not found in StructureSetROISequence")
        
        except Exception as e:
            if isinstance(e, ValidationError):
                raise
            else:
                raise ValidationError(f"Error validating RT Structure sequences: {str(e)}")
        
        return True
    
    @staticmethod
    def validate_rt_dose_dataset(dataset: pydicom.dataset.FileDataset) -> bool:
        """
        Xác thực tính hợp lệ của dataset RT Dose.
        
        Parameters
        ----------
        dataset : pydicom.dataset.FileDataset
            Dataset DICOM RT Dose cần xác thực
            
        Returns
        -------
        bool
            True nếu dataset hợp lệ
            
        Raises
        ------
        ValidationError
            Nếu dataset không hợp lệ
        """
        # Kiểm tra modality
        if not hasattr(dataset, 'Modality') or dataset.Modality != 'RTDOSE':
            raise ValidationError("Dataset is not an RT Dose dataset")
        
        # Kiểm tra các thuộc tính bắt buộc
        required_attributes = [
            'PatientID',
            'PatientName',
            'StudyInstanceUID',
            'SeriesInstanceUID',
            'SOPInstanceUID',
            'DoseGridScaling',
            'DoseUnits',
            'DoseSummationType'
        ]
        
        missing_attributes = []
        for attr in required_attributes:
            if not hasattr(dataset, attr):
                missing_attributes.append(attr)
        
        if missing_attributes:
            raise ValidationError(f"Missing required attributes: {', '.join(missing_attributes)}")
        
        # Kiểm tra dữ liệu liều
        if not hasattr(dataset, 'pixel_array'):
            raise ValidationError("RT Dose dataset does not contain dose data")
        
        return True
    
    @staticmethod
    def validate_rt_plan_dataset(dataset: pydicom.dataset.FileDataset) -> bool:
        """
        Xác thực tính hợp lệ của dataset RT Plan.
        
        Parameters
        ----------
        dataset : pydicom.dataset.FileDataset
            Dataset DICOM RT Plan cần xác thực
            
        Returns
        -------
        bool
            True nếu dataset hợp lệ
            
        Raises
        ------
        ValidationError
            Nếu dataset không hợp lệ
        """
        # Kiểm tra modality
        if not hasattr(dataset, 'Modality') or dataset.Modality != 'RTPLAN':
            raise ValidationError("Dataset is not an RT Plan dataset")
        
        # Kiểm tra các thuộc tính bắt buộc
        required_attributes = [
            'PatientID',
            'PatientName',
            'StudyInstanceUID',
            'SeriesInstanceUID',
            'SOPInstanceUID',
            'RTPlanLabel'
        ]
        
        missing_attributes = []
        for attr in required_attributes:
            if not hasattr(dataset, attr):
                missing_attributes.append(attr)
        
        if missing_attributes:
            raise ValidationError(f"Missing required attributes: {', '.join(missing_attributes)}")
        
        # Kiểm tra BeamSequence (nếu có)
        if hasattr(dataset, 'BeamSequence'):
            for i, beam in enumerate(dataset.BeamSequence):
                if not hasattr(beam, 'BeamNumber'):
                    raise ValidationError(f"BeamNumber missing in beam #{i+1}")
        
        # Kiểm tra DoseReferenceSequence (nếu có)
        if hasattr(dataset, 'DoseReferenceSequence'):
            for i, dose_ref in enumerate(dataset.DoseReferenceSequence):
                if not hasattr(dose_ref, 'DoseReferenceNumber'):
                    raise ValidationError(f"DoseReferenceNumber missing in dose reference #{i+1}")
        
        return True
    
    @staticmethod
    def validate_mr_dataset(dataset: pydicom.dataset.FileDataset) -> bool:
        """
        Xác thực tính hợp lệ của dataset MR.
        
        Parameters
        ----------
        dataset : pydicom.dataset.FileDataset
            Dataset DICOM MR cần xác thực
            
        Returns
        -------
        bool
            True nếu dataset hợp lệ
            
        Raises
        ------
        ValidationError
            Nếu dataset không hợp lệ
        """
        # Kiểm tra modality
        if not hasattr(dataset, 'Modality') or dataset.Modality != 'MR':
            raise ValidationError("Dataset is not an MR dataset")
        
        # Kiểm tra các thuộc tính bắt buộc
        required_attributes = [
            'PatientID',
            'PatientName',
            'StudyInstanceUID',
            'SeriesInstanceUID',
            'SOPInstanceUID',
            'PixelSpacing',
            'ImagePositionPatient',
            'ImageOrientationPatient',
            'SliceThickness',
            'MagneticFieldStrength',
            'RepetitionTime',
            'EchoTime'
        ]
        
        missing_attributes = []
        for attr in required_attributes:
            if not hasattr(dataset, attr):
                missing_attributes.append(attr)
        
        if missing_attributes:
            raise ValidationError(f"Missing required attributes: {', '.join(missing_attributes)}")
        
        # Kiểm tra dữ liệu hình ảnh
        if not hasattr(dataset, 'pixel_array'):
            raise ValidationError("MR dataset does not contain image data")
        
        return True
    
    @staticmethod
    def validate_rt_image_dataset(dataset: pydicom.dataset.FileDataset) -> bool:
        """
        Xác thực tính hợp lệ của dataset RT Image.
        
        Parameters
        ----------
        dataset : pydicom.dataset.FileDataset
            Dataset DICOM RT Image cần xác thực
            
        Returns
        -------
        bool
            True nếu dataset hợp lệ
            
        Raises
        ------
        ValidationError
            Nếu dataset không hợp lệ
        """
        # Kiểm tra modality
        if not hasattr(dataset, 'Modality') or dataset.Modality != 'RTIMAGE':
            raise ValidationError("Dataset is not an RT Image dataset")
        
        # Kiểm tra các thuộc tính bắt buộc
        required_attributes = [
            'PatientID',
            'PatientName',
            'StudyInstanceUID',
            'SeriesInstanceUID',
            'SOPInstanceUID',
            'RTImageLabel',
            'RTImagePlane',
            'RTImagePosition',
            'RadiationMachineName'
        ]
        
        missing_attributes = []
        for attr in required_attributes:
            if not hasattr(dataset, attr):
                missing_attributes.append(attr)
        
        if missing_attributes:
            raise ValidationError(f"Missing required attributes: {', '.join(missing_attributes)}")
        
        # Kiểm tra dữ liệu hình ảnh
        if not hasattr(dataset, 'pixel_array'):
            raise ValidationError("RT Image dataset does not contain image data")
        
        return True
    
    @staticmethod
    def validate_dataset(dataset: pydicom.dataset.FileDataset) -> bool:
        """
        Xác thực tính hợp lệ của dataset DICOM dựa trên loại của nó.
        
        Parameters
        ----------
        dataset : pydicom.dataset.FileDataset
            Dataset DICOM cần xác thực
            
        Returns
        -------
        bool
            True nếu dataset hợp lệ
            
        Raises
        ------
        ValidationError
            Nếu dataset không hợp lệ
        DicomError
            Nếu loại dataset không được hỗ trợ
        """
        if not hasattr(dataset, 'Modality'):
            raise ValidationError("Dataset does not have a Modality attribute")
        
        modality = dataset.Modality
        
        if modality == 'CT':
            return DicomValidator.validate_ct_dataset(dataset)
        elif modality == 'RTSTRUCT':
            return DicomValidator.validate_rt_structure_dataset(dataset)
        elif modality == 'RTDOSE':
            return DicomValidator.validate_rt_dose_dataset(dataset)
        elif modality == 'RTPLAN':
            return DicomValidator.validate_rt_plan_dataset(dataset)
        elif modality == 'MR':
            return DicomValidator.validate_mr_dataset(dataset)
        elif modality == 'RTIMAGE':
            return DicomValidator.validate_rt_image_dataset(dataset)
        else:
            raise DicomError(f"Unsupported modality: {modality}")
    
    @staticmethod
    def validate_dicom_file(file_path: str) -> Tuple[bool, str]:
        """
        Xác thực tính hợp lệ của file DICOM.
        
        Parameters
        ----------
        file_path : str
            Đường dẫn đến file DICOM cần xác thực
            
        Returns
        -------
        Tuple[bool, str]
            (is_valid, message) - Kết quả xác thực và thông báo
        """
        try:
            from quangtps.dicom.dicom_reader import DicomReader
            dataset = DicomReader.read_file(file_path)
            DicomValidator.validate_dataset(dataset)
            return True, "File is valid"
        except ValidationError as e:
            return False, str(e)
        except DicomError as e:
            return False, str(e)
        except Exception as e:
            return False, f"Unexpected error validating DICOM file: {str(e)}"
    
    @staticmethod
    def check_rt_consistency(rt_struct: pydicom.dataset.FileDataset, ct_datasets: List[pydicom.dataset.FileDataset]) -> bool:
        """
        Kiểm tra tính nhất quán giữa RT Structure và bộ dữ liệu CT.
        
        Parameters
        ----------
        rt_struct : pydicom.dataset.FileDataset
            Dataset RT Structure
        ct_datasets : List[pydicom.dataset.FileDataset]
            Danh sách các dataset CT
            
        Returns
        -------
        bool
            True nếu RT Structure và CT nhất quán với nhau
            
        Raises
        ------
        ValidationError
            Nếu phát hiện sự không nhất quán
        """
        if not hasattr(rt_struct, 'ReferencedFrameOfReferenceSequence'):
            raise ValidationError("RT Structure does not have ReferencedFrameOfReferenceSequence")
        
        if len(ct_datasets) == 0:
            raise ValidationError("No CT datasets provided")
        
        # Kiểm tra Study Instance UID
        rt_study_uid = rt_struct.StudyInstanceUID
        for ct in ct_datasets:
            if ct.StudyInstanceUID != rt_study_uid:
                raise ValidationError(f"StudyInstanceUID mismatch: RT Structure ({rt_study_uid}) ≠ CT ({ct.StudyInstanceUID})")
        
        # Kiểm tra Frame of Reference
        frame_of_ref_uid = None
        for frame_of_ref in rt_struct.ReferencedFrameOfReferenceSequence:
            if hasattr(frame_of_ref, 'RTReferencedStudySequence'):
                for rt_ref_study in frame_of_ref.RTReferencedStudySequence:
                    if hasattr(rt_ref_study, 'ReferencedSOPInstanceUID') and rt_ref_study.ReferencedSOPInstanceUID == rt_study_uid:
                        frame_of_ref_uid = frame_of_ref.FrameOfReferenceUID
                        break
        
        if frame_of_ref_uid is None:
            raise ValidationError("Could not find matching Frame of Reference in RT Structure")
        
        for ct in ct_datasets:
            if hasattr(ct, 'FrameOfReferenceUID') and ct.FrameOfReferenceUID != frame_of_ref_uid:
                raise ValidationError(f"FrameOfReferenceUID mismatch: RT Structure ({frame_of_ref_uid}) ≠ CT ({ct.FrameOfReferenceUID})")
        
        return True