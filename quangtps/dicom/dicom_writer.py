"""
Ghi dữ liệu ra file DICOM.
"""

import os
import logging
import datetime
import pydicom
from pydicom.uid import generate_uid

from quangtps.core.exceptions import DicomError, IOError

logger = logging.getLogger(__name__)

class DicomWriter:
    """Lớp xử lý việc ghi dữ liệu ra file DICOM"""
    
    def __init__(self):
        """Khởi tạo DicomWriter"""
        pass
    
    @staticmethod
    def save_file(dataset, file_path):
        """
        Lưu dataset DICOM ra file.
        
        Parameters:
            dataset (pydicom.dataset.FileDataset): Dataset DICOM
            file_path (str): Đường dẫn đến file đích
        
        Raises:
            IOError: Nếu không thể ghi file
            DicomError: Nếu dataset không hợp lệ
        """
        if dataset is None:
            raise DicomError("Cannot save empty DICOM dataset")
        
        # Đảm bảo thư mục tồn tại
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        
        try:
            dataset.save_as(file_path)
            logger.info(f"Saved DICOM file: {file_path}")
        except Exception as e:
            raise IOError(f"Failed to save DICOM file: {str(e)}", file_path=file_path)
    
    @staticmethod
    def create_new_rt_plan(patient_name, patient_id, study_instance_uid=None):
        """
        Tạo DICOM RT Plan mới.
        
        Parameters:
            patient_name (str): Tên bệnh nhân
            patient_id (str): ID bệnh nhân
            study_instance_uid (str, optional): Study Instance UID
        
        Returns:
            pydicom.dataset.FileDataset: Dataset DICOM RT Plan
        """
        # Tạo file meta
        file_meta = pydicom.dataset.FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.481.5'  # RT Plan Storage
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
        
        # Tạo dataset
        ds = pydicom.dataset.FileDataset(
            '', {}, file_meta=file_meta, preamble=b"\0" * 128
        )
        
        # Thêm thẻ bắt buộc
        ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
        ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        
        # Ngày và thời gian
        dt = datetime.datetime.now()
        ds.InstanceCreationDate = dt.strftime('%Y%m%d')
        ds.InstanceCreationTime = dt.strftime('%H%M%S.%f')
        
        # Thông tin bệnh nhân
        ds.PatientName = patient_name
        ds.PatientID = patient_id
        ds.PatientBirthDate = ''
        ds.PatientSex = ''
        
        # Thông tin nghiên cứu
        if study_instance_uid:
            ds.StudyInstanceUID = study_instance_uid
        else:
            ds.StudyInstanceUID = generate_uid()
        
        ds.StudyDate = dt.strftime('%Y%m%d')
        ds.StudyTime = dt.strftime('%H%M%S.%f')
        ds.StudyID = '1'
        ds.Modality = 'RTPLAN'
        
        # Series
        ds.SeriesInstanceUID = generate_uid()
        ds.SeriesNumber = '1'
        
        # Các thẻ RT Plan
        ds.RTPlanLabel = 'PLAN'
        ds.RTPlanName = 'QuangTPS Plan'
        ds.RTPlanDate = dt.strftime('%Y%m%d')
        ds.RTPlanTime = dt.strftime('%H%M%S.%f')
        ds.RTPlanGeometry = 'PATIENT'
        
        # Thiết lập Explicit VR Little Endian cho Transfer Syntax
        ds.is_little_endian = True
        ds.is_implicit_VR = False
        
        return ds
    
    @staticmethod
    def create_new_rt_structure_set(ct_dataset):
        """
        Tạo DICOM RT Structure Set mới dựa trên CT dataset.
        
        Parameters:
            ct_dataset (pydicom.dataset.FileDataset): Dataset DICOM CT
        
        Returns:
            pydicom.dataset.FileDataset: Dataset DICOM RT Structure Set
        """
        # Tạo file meta
        file_meta = pydicom.dataset.FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.481.3'  # RT Structure Set Storage
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
        
        # Tạo dataset
        ds = pydicom.dataset.FileDataset(
            '', {}, file_meta=file_meta, preamble=b"\0" * 128
        )
        
        # Thêm thẻ bắt buộc
        ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
        ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        
        # Ngày và thời gian
        dt = datetime.datetime.now()
        ds.InstanceCreationDate = dt.strftime('%Y%m%d')
        ds.InstanceCreationTime = dt.strftime('%H%M%S.%f')
        
        # Thông tin bệnh nhân từ CT
        ds.PatientName = ct_dataset.PatientName
        ds.PatientID = ct_dataset.PatientID
        if hasattr(ct_dataset, 'PatientBirthDate'):
            ds.PatientBirthDate = ct_dataset.PatientBirthDate
        else:
            ds.PatientBirthDate = ''
        if hasattr(ct_dataset, 'PatientSex'):
            ds.PatientSex = ct_dataset.PatientSex
        else:
            ds.PatientSex = ''
        
        # Thông tin nghiên cứu từ CT
        ds.StudyInstanceUID = ct_dataset.StudyInstanceUID
        ds.StudyDate = ct_dataset.StudyDate
        ds.StudyTime = ct_dataset.StudyTime
        ds.StudyID = ct_dataset.StudyID
        ds.Modality = 'RTSTRUCT'
        
        # Series
        ds.SeriesInstanceUID = generate_uid()
        ds.SeriesNumber = '1'
        
        # Tham chiếu đến CT series
        ds.StructureSetLabel = 'RTSTRUCT'
        ds.StructureSetName = 'QuangTPS Structures'
        ds.StructureSetDate = dt.strftime('%Y%m%d')
        ds.StructureSetTime = dt.strftime('%H%M%S.%f')
        
        # Thiết lập Explicit VR Little Endian cho Transfer Syntax
        ds.is_little_endian = True
        ds.is_implicit_VR = False
        
        return ds