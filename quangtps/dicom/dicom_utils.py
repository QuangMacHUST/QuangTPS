"""
Các tiện ích và hàm phụ trợ cho module DICOM.

Module này cung cấp các công cụ và tiện ích hỗ trợ cho các thao tác với dữ liệu DICOM,
bao gồm các hàm xử lý, chuyển đổi và hỗ trợ khác.
"""

import os
import logging
import numpy as np
import pydicom
from typing import List, Dict, Any, Tuple, Optional, Union
from datetime import datetime

from quangtps.core.exceptions import DicomError, IOError

logger = logging.getLogger(__name__)

def get_dicom_type(dataset: pydicom.dataset.FileDataset) -> str:
    """
    Xác định loại DICOM từ dataset.
    
    Parameters
    ----------
    dataset : pydicom.dataset.FileDataset
        Dataset DICOM cần xác định loại
        
    Returns
    -------
    str
        Loại DICOM ("CT", "MR", "RTSTRUCT", "RTDOSE", "RTPLAN", "PET", "US", "OTHER")
        
    Raises
    ------
    DicomError
        Nếu không thể xác định loại DICOM
    """
    try:
        if not hasattr(dataset, 'Modality'):
            raise DicomError("Dataset does not have Modality attribute")
        
        modality = dataset.Modality
        
        if modality == 'CT':
            return "CT"
        elif modality == 'MR':
            return "MR"
        elif modality == 'RTSTRUCT':
            return "RTSTRUCT"
        elif modality == 'RTDOSE':
            return "RTDOSE"
        elif modality == 'RTPLAN':
            return "RTPLAN"
        elif modality == 'PT':
            return "PET"
        elif modality == 'US':
            return "US"
        else:
            return "OTHER"
    except Exception as e:
        logger.error(f"Error determining DICOM type: {str(e)}")
        raise DicomError(f"Error determining DICOM type: {str(e)}")

def dicom_date_to_string(dicom_date: str, format: str = "%Y-%m-%d") -> str:
    """
    Chuyển đổi định dạng ngày DICOM (YYYYMMDD) thành string.
    
    Parameters
    ----------
    dicom_date : str
        Ngày DICOM dạng YYYYMMDD
    format : str, optional
        Định dạng đầu ra, mặc định là "%Y-%m-%d"
        
    Returns
    -------
    str
        Ngày đã chuyển đổi theo định dạng
    """
    if not dicom_date or len(dicom_date) != 8:
        return ""
    
    try:
        year = int(dicom_date[:4])
        month = int(dicom_date[4:6])
        day = int(dicom_date[6:8])
        date_obj = datetime(year, month, day)
        return date_obj.strftime(format)
    except:
        return ""

def dicom_time_to_string(dicom_time: str, format: str = "%H:%M:%S") -> str:
    """
    Chuyển đổi định dạng thời gian DICOM (HHMMSS.FFFFFF) thành string.
    
    Parameters
    ----------
    dicom_time : str
        Thời gian DICOM dạng HHMMSS.FFFFFF
    format : str, optional
        Định dạng đầu ra, mặc định là "%H:%M:%S"
        
    Returns
    -------
    str
        Thời gian đã chuyển đổi theo định dạng
    """
    if not dicom_time:
        return ""
    
    try:
        # Cắt bỏ phần microsecond nếu có
        if '.' in dicom_time:
            dicom_time = dicom_time.split('.')[0]
        
        if len(dicom_time) < 6:
            return ""
        
        hour = int(dicom_time[:2])
        minute = int(dicom_time[2:4])
        second = int(dicom_time[4:6])
        time_obj = datetime(1900, 1, 1, hour, minute, second)
        return time_obj.strftime(format)
    except:
        return ""

def string_to_dicom_date(date_str: str, input_format: str = "%Y-%m-%d") -> str:
    """
    Chuyển đổi string thành định dạng ngày DICOM (YYYYMMDD).
    
    Parameters
    ----------
    date_str : str
        Chuỗi ngày đầu vào
    input_format : str, optional
        Định dạng của chuỗi đầu vào, mặc định là "%Y-%m-%d"
        
    Returns
    -------
    str
        Ngày dạng DICOM (YYYYMMDD)
    """
    if not date_str:
        return ""
    
    try:
        date_obj = datetime.strptime(date_str, input_format)
        return date_obj.strftime("%Y%m%d")
    except:
        return ""

def string_to_dicom_time(time_str: str, input_format: str = "%H:%M:%S") -> str:
    """
    Chuyển đổi string thành định dạng thời gian DICOM (HHMMSS).
    
    Parameters
    ----------
    time_str : str
        Chuỗi thời gian đầu vào
    input_format : str, optional
        Định dạng của chuỗi đầu vào, mặc định là "%H:%M:%S"
        
    Returns
    -------
    str
        Thời gian dạng DICOM (HHMMSS)
    """
    if not time_str:
        return ""
    
    try:
        time_obj = datetime.strptime(time_str, input_format)
        return time_obj.strftime("%H%M%S")
    except:
        return ""

def get_cleaned_dicom_value(value: Any) -> Any:
    """
    Làm sạch giá trị DICOM, chuyển các đối tượng PersonName thành string.
    
    Parameters
    ----------
    value : Any
        Giá trị DICOM cần làm sạch
        
    Returns
    -------
    Any
        Giá trị đã làm sạch
    """
    from pydicom.valuerep import PersonName
    
    if isinstance(value, PersonName):
        return str(value)
    return value

def extract_patient_info(dataset: pydicom.dataset.FileDataset) -> Dict[str, Any]:
    """
    Trích xuất thông tin bệnh nhân từ dataset DICOM.
    
    Parameters
    ----------
    dataset : pydicom.dataset.FileDataset
        Dataset DICOM
        
    Returns
    -------
    Dict[str, Any]
        Thông tin bệnh nhân
    """
    patient_info = {}
    
    # Thông tin cơ bản
    if hasattr(dataset, 'PatientID'):
        patient_info['PatientID'] = get_cleaned_dicom_value(dataset.PatientID)
    
    if hasattr(dataset, 'PatientName'):
        patient_info['PatientName'] = get_cleaned_dicom_value(dataset.PatientName)
    
    if hasattr(dataset, 'PatientBirthDate'):
        patient_info['PatientBirthDate'] = get_cleaned_dicom_value(dataset.PatientBirthDate)
    
    if hasattr(dataset, 'PatientSex'):
        patient_info['PatientSex'] = get_cleaned_dicom_value(dataset.PatientSex)
    
    # Thông tin thêm
    if hasattr(dataset, 'PatientAge'):
        patient_info['PatientAge'] = get_cleaned_dicom_value(dataset.PatientAge)
    
    if hasattr(dataset, 'PatientWeight'):
        patient_info['PatientWeight'] = get_cleaned_dicom_value(dataset.PatientWeight)
    
    if hasattr(dataset, 'PatientSize'):
        patient_info['PatientSize'] = get_cleaned_dicom_value(dataset.PatientSize)
    
    return patient_info

def extract_study_info(dataset: pydicom.dataset.FileDataset) -> Dict[str, Any]:
    """
    Trích xuất thông tin nghiên cứu từ dataset DICOM.
    
    Parameters
    ----------
    dataset : pydicom.dataset.FileDataset
        Dataset DICOM
        
    Returns
    -------
    Dict[str, Any]
        Thông tin nghiên cứu
    """
    study_info = {}
    
    # Thông tin cơ bản
    if hasattr(dataset, 'StudyInstanceUID'):
        study_info['StudyInstanceUID'] = get_cleaned_dicom_value(dataset.StudyInstanceUID)
    
    if hasattr(dataset, 'StudyDate'):
        study_info['StudyDate'] = get_cleaned_dicom_value(dataset.StudyDate)
        study_info['StudyDateFormatted'] = dicom_date_to_string(study_info['StudyDate'])
    
    if hasattr(dataset, 'StudyTime'):
        study_info['StudyTime'] = get_cleaned_dicom_value(dataset.StudyTime)
        study_info['StudyTimeFormatted'] = dicom_time_to_string(study_info['StudyTime'])
    
    if hasattr(dataset, 'StudyDescription'):
        study_info['StudyDescription'] = get_cleaned_dicom_value(dataset.StudyDescription)
    
    if hasattr(dataset, 'StudyID'):
        study_info['StudyID'] = get_cleaned_dicom_value(dataset.StudyID)
    
    # Thông tin thêm
    if hasattr(dataset, 'AccessionNumber'):
        study_info['AccessionNumber'] = get_cleaned_dicom_value(dataset.AccessionNumber)
    
    if hasattr(dataset, 'ReferringPhysicianName'):
        study_info['ReferringPhysicianName'] = get_cleaned_dicom_value(dataset.ReferringPhysicianName)
    
    return study_info

def extract_series_info(dataset: pydicom.dataset.FileDataset) -> Dict[str, Any]:
    """
    Trích xuất thông tin series từ dataset DICOM.
    
    Parameters
    ----------
    dataset : pydicom.dataset.FileDataset
        Dataset DICOM
        
    Returns
    -------
    Dict[str, Any]
        Thông tin series
    """
    series_info = {}
    
    # Thông tin cơ bản
    if hasattr(dataset, 'SeriesInstanceUID'):
        series_info['SeriesInstanceUID'] = get_cleaned_dicom_value(dataset.SeriesInstanceUID)
    
    if hasattr(dataset, 'SeriesNumber'):
        series_info['SeriesNumber'] = get_cleaned_dicom_value(dataset.SeriesNumber)
    
    if hasattr(dataset, 'SeriesDescription'):
        series_info['SeriesDescription'] = get_cleaned_dicom_value(dataset.SeriesDescription)
    
    if hasattr(dataset, 'SeriesDate'):
        series_info['SeriesDate'] = get_cleaned_dicom_value(dataset.SeriesDate)
        series_info['SeriesDateFormatted'] = dicom_date_to_string(series_info['SeriesDate'])
    
    if hasattr(dataset, 'SeriesTime'):
        series_info['SeriesTime'] = get_cleaned_dicom_value(dataset.SeriesTime)
        series_info['SeriesTimeFormatted'] = dicom_time_to_string(series_info['SeriesTime'])
    
    if hasattr(dataset, 'Modality'):
        series_info['Modality'] = get_cleaned_dicom_value(dataset.Modality)
    
    # Thông tin thêm
    if hasattr(dataset, 'BodyPartExamined'):
        series_info['BodyPartExamined'] = get_cleaned_dicom_value(dataset.BodyPartExamined)
    
    if hasattr(dataset, 'ProtocolName'):
        series_info['ProtocolName'] = get_cleaned_dicom_value(dataset.ProtocolName)
    
    return series_info

def extract_instance_info(dataset: pydicom.dataset.FileDataset) -> Dict[str, Any]:
    """
    Trích xuất thông tin instance từ dataset DICOM.
    
    Parameters
    ----------
    dataset : pydicom.dataset.FileDataset
        Dataset DICOM
        
    Returns
    -------
    Dict[str, Any]
        Thông tin instance
    """
    instance_info = {}
    
    # Thông tin cơ bản
    if hasattr(dataset, 'SOPInstanceUID'):
        instance_info['SOPInstanceUID'] = get_cleaned_dicom_value(dataset.SOPInstanceUID)
    
    if hasattr(dataset, 'SOPClassUID'):
        instance_info['SOPClassUID'] = get_cleaned_dicom_value(dataset.SOPClassUID)
    
    if hasattr(dataset, 'InstanceNumber'):
        instance_info['InstanceNumber'] = get_cleaned_dicom_value(dataset.InstanceNumber)
    
    # Thông tin hình ảnh
    if hasattr(dataset, 'Rows'):
        instance_info['Rows'] = get_cleaned_dicom_value(dataset.Rows)
    
    if hasattr(dataset, 'Columns'):
        instance_info['Columns'] = get_cleaned_dicom_value(dataset.Columns)
    
    if hasattr(dataset, 'PixelSpacing'):
        instance_info['PixelSpacing'] = get_cleaned_dicom_value(dataset.PixelSpacing)
    
    if hasattr(dataset, 'SliceThickness'):
        instance_info['SliceThickness'] = get_cleaned_dicom_value(dataset.SliceThickness)
    
    if hasattr(dataset, 'SliceLocation'):
        instance_info['SliceLocation'] = get_cleaned_dicom_value(dataset.SliceLocation)
    
    if hasattr(dataset, 'ImagePositionPatient'):
        instance_info['ImagePositionPatient'] = get_cleaned_dicom_value(dataset.ImagePositionPatient)
    
    if hasattr(dataset, 'ImageOrientationPatient'):
        instance_info['ImageOrientationPatient'] = get_cleaned_dicom_value(dataset.ImageOrientationPatient)
    
    return instance_info

def sort_dicom_files_by_instance_number(file_paths: List[str]) -> List[str]:
    """
    Sắp xếp danh sách các file DICOM theo số thứ tự instance.
    
    Parameters
    ----------
    file_paths : List[str]
        Danh sách đường dẫn các file DICOM
        
    Returns
    -------
    List[str]
        Danh sách đã sắp xếp
    """
    file_info = []
    for path in file_paths:
        try:
            dataset = pydicom.dcmread(path, force=True, stop_before_pixels=True)
            instance_number = 0
            if hasattr(dataset, 'InstanceNumber'):
                try:
                    instance_number = int(dataset.InstanceNumber)
                except:
                    pass
            
            slice_location = 0
            if hasattr(dataset, 'SliceLocation'):
                try:
                    slice_location = float(dataset.SliceLocation)
                except:
                    pass
            
            file_info.append((path, instance_number, slice_location))
        except Exception as e:
            logger.warning(f"Error reading DICOM file {path}: {str(e)}")
            # Thêm vào cuối danh sách nếu không đọc được
            file_info.append((path, float('inf'), float('inf')))
    
    # Sắp xếp theo InstanceNumber, sau đó theo SliceLocation
    sorted_files = [fi[0] for fi in sorted(file_info, key=lambda x: (x[1], x[2]))]
    return sorted_files

def get_dicom_sop_class_name(sop_class_uid: str) -> str:
    """
    Lấy tên SOP Class từ UID.
    
    Parameters
    ----------
    sop_class_uid : str
        SOP Class UID
        
    Returns
    -------
    str
        Tên SOP Class
    """
    from pydicom.uid import UID_dictionary
    
    if sop_class_uid in UID_dictionary:
        return UID_dictionary[sop_class_uid][0]
    else:
        return "Unknown"

def anonymize_dataset(dataset: pydicom.dataset.FileDataset, 
                      new_patient_id: str = None,
                      new_patient_name: str = None,
                      keep_uids: bool = False) -> pydicom.dataset.FileDataset:
    """
    Ẩn danh dataset DICOM.
    
    Parameters
    ----------
    dataset : pydicom.dataset.FileDataset
        Dataset DICOM cần ẩn danh
    new_patient_id : str, optional
        ID bệnh nhân mới, nếu None sẽ tạo một ID ngẫu nhiên
    new_patient_name : str, optional
        Tên bệnh nhân mới, nếu None sẽ tạo một tên ngẫu nhiên
    keep_uids : bool, optional
        Giữ lại các UID gốc hay không
        
    Returns
    -------
    pydicom.dataset.FileDataset
        Dataset đã ẩn danh
    """
    import uuid
    import copy
    
    # Tạo bản sao để không ảnh hưởng đến dataset gốc
    anon_dataset = copy.deepcopy(dataset)
    
    # Tạo ID và tên mới nếu cần
    if new_patient_id is None:
        new_patient_id = f"ANON{uuid.uuid4().hex[:8].upper()}"
    
    if new_patient_name is None:
        new_patient_name = f"Anonymous^Patient^{uuid.uuid4().hex[:8].upper()}"
    
    # Danh sách các tag cần ẩn danh
    tags_to_anonymize = [
        # Thông tin bệnh nhân
        (0x0010, 0x0010),  # PatientName
        (0x0010, 0x0020),  # PatientID
        (0x0010, 0x0030),  # PatientBirthDate
        (0x0010, 0x0040),  # PatientSex
        (0x0010, 0x1000),  # OtherPatientIDs
        (0x0010, 0x1001),  # OtherPatientNames
        (0x0010, 0x1010),  # PatientAge
        (0x0010, 0x1020),  # PatientSize
        (0x0010, 0x1030),  # PatientWeight
        (0x0010, 0x2160),  # EthnicGroup
        (0x0010, 0x4000),  # PatientComments
        
        # Thông tin nghiên cứu
        (0x0008, 0x0050),  # AccessionNumber
        (0x0008, 0x0090),  # ReferringPhysicianName
        (0x0008, 0x1060),  # NameOfPhysiciansReadingStudy
        (0x0008, 0x1070),  # OperatorsName
        (0x0020, 0x4000),  # ImageComments
        
        # Các thông tin nhận dạng khác
        (0x0008, 0x0080),  # InstitutionName
        (0x0008, 0x0081),  # InstitutionAddress
        (0x0008, 0x0070),  # Manufacturer
        (0x0008, 0x1010),  # StationName
        (0x0008, 0x1040),  # InstitutionalDepartmentName
        (0x0008, 0x1048),  # PhysiciansOfRecord
        (0x0008, 0x1050),  # PerformingPhysicianName
        (0x0088, 0x0140),  # StorageMediaFileSetUID
    ]
    
    # Xóa các tag nhạy cảm
    for tag in tags_to_anonymize:
        if tag in anon_dataset:
            del anon_dataset[tag]
    
    # Thiết lập thông tin mới
    anon_dataset.PatientName = new_patient_name
    anon_dataset.PatientID = new_patient_id
    
    # Tạo UID mới nếu cần
    if not keep_uids:
        import pydicom.uid
        
        # Tạo các UID mới
        anon_dataset.StudyInstanceUID = pydicom.uid.generate_uid()
        anon_dataset.SeriesInstanceUID = pydicom.uid.generate_uid()
        anon_dataset.SOPInstanceUID = pydicom.uid.generate_uid()
        
        # Xử lý các UID khác trong dataset
        if hasattr(anon_dataset, 'FrameOfReferenceUID'):
            anon_dataset.FrameOfReferenceUID = pydicom.uid.generate_uid()
    
    return anon_dataset
