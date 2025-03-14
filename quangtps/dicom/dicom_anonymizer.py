"""
Công cụ ẩn danh dữ liệu DICOM.

Module này cung cấp các chức năng để ẩn danh dữ liệu DICOM,
xóa bỏ thông tin cá nhân và nhạy cảm của bệnh nhân.
"""

import os
import logging
import pydicom
import datetime
import hashlib
import uuid
import random
import re
from typing import List, Dict, Any, Tuple, Optional, Union, Set

from quangtps.core.exceptions import DicomError
from quangtps.dicom.dicom_utils import get_dicom_type

logger = logging.getLogger(__name__)

class DicomAnonymizer:
    """
    Lớp ẩn danh dữ liệu DICOM.
    
    Class này cung cấp các phương thức để ẩn danh dữ liệu DICOM,
    xóa bỏ thông tin cá nhân và nhạy cảm của bệnh nhân.
    """
    
    # Các tag chứa thông tin nhạy cảm cần ẩn danh hoàn toàn
    TAGS_TO_ANONYMIZE = [
        (0x0008, 0x0080),  # InstitutionName
        (0x0008, 0x0081),  # InstitutionAddress
        (0x0008, 0x0090),  # ReferringPhysicianName
        (0x0008, 0x0092),  # ReferringPhysicianAddress
        (0x0008, 0x0094),  # ReferringPhysicianTelephoneNumber
        (0x0008, 0x1048),  # PhysiciansOfRecord
        (0x0008, 0x1050),  # PerformingPhysicianName
        (0x0010, 0x0010),  # PatientName
        (0x0010, 0x0020),  # PatientID
        (0x0010, 0x0030),  # PatientBirthDate
        (0x0010, 0x0032),  # PatientBirthTime
        (0x0010, 0x0040),  # PatientSex
        (0x0010, 0x1000),  # OtherPatientIDs
        (0x0010, 0x1001),  # OtherPatientNames
        (0x0010, 0x1005),  # PatientBirthName
        (0x0010, 0x1010),  # PatientAge
        (0x0010, 0x1040),  # PatientAddress
        (0x0010, 0x1060),  # PatientMotherBirthName
        (0x0010, 0x1080),  # MilitaryRank
        (0x0010, 0x1081),  # BranchOfService
        (0x0010, 0x2150),  # CountryOfResidence
        (0x0010, 0x2152),  # RegionOfResidence
        (0x0010, 0x2154),  # PatientTelephoneNumbers
        (0x0010, 0x2160),  # EthnicGroup
        (0x0010, 0x2180),  # Occupation
        (0x0010, 0x21B0),  # AdditionalPatientHistory
    ]
    
    # Các tag chứa thông tin ngày tháng cần được sửa đổi
    DATE_TAGS = [
        (0x0008, 0x0020),  # StudyDate
        (0x0008, 0x0021),  # SeriesDate
        (0x0008, 0x0022),  # AcquisitionDate
        (0x0008, 0x0023),  # ContentDate
        (0x0008, 0x0024),  # OverlayDate
        (0x0008, 0x0025),  # CurveDate
    ]
    
    # Các tag chứa thông tin thời gian cần được sửa đổi
    TIME_TAGS = [
        (0x0008, 0x0030),  # StudyTime
        (0x0008, 0x0031),  # SeriesTime
        (0x0008, 0x0032),  # AcquisitionTime
        (0x0008, 0x0033),  # ContentTime
        (0x0008, 0x0034),  # OverlayTime
        (0x0008, 0x0035),  # CurveTime
    ]
    
    # Các tag chứa thông tin mô tả có thể chứa thông tin nhạy cảm
    DESCRIPTION_TAGS = [
        (0x0008, 0x1030),  # StudyDescription
        (0x0008, 0x103E),  # SeriesDescription
        (0x0008, 0x1080),  # AdmittingDiagnosesDescription
        (0x0018, 0x1030),  # ProtocolName
        (0x0040, 0x0254),  # PerformedProcedureStepDescription
        (0x0040, 0x0260),  # PerformedProtocolCodeSequence
    ]
    
    # Các tag chứa thông tin định danh cần được giữ lại mối quan hệ
    # nhưng cần thay đổi giá trị
    UID_TAGS = [
        (0x0008, 0x0018),  # SOPInstanceUID
        (0x0020, 0x000D),  # StudyInstanceUID
        (0x0020, 0x000E),  # SeriesInstanceUID
        (0x0020, 0x0052),  # FrameOfReferenceUID
        (0x0088, 0x0140),  # StorageMediaFileSetUID
    ]
    
    def __init__(self):
        """Khởi tạo đối tượng DicomAnonymizer."""
        self.uid_map = {}  # Map từ UID cũ sang UID mới
        self.id_map = {}   # Map từ ID cũ sang ID mới
        self.date_offset = random.randint(-1000, 1000)  # Độ lệch ngày ngẫu nhiên
    
    def anonymize_dataset(self, dataset: pydicom.dataset.FileDataset, 
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
            
        Raises
        ------
        DicomError
            Nếu không thể ẩn danh dataset
        """
        try:
            # Tạo một bản sao của dataset
            anonymized = dataset.copy()
            
            # Loại bỏ thông tin nhạy cảm
            for tag in self.TAGS_TO_ANONYMIZE:
                if tag in anonymized:
                    del anonymized[tag]
            
            # Xử lý các tag ngày tháng
            for tag in self.DATE_TAGS:
                if tag in anonymized:
                    anonymized[tag].value = self._modify_date(anonymized[tag].value)
            
            # Xử lý các tag thời gian
            for tag in self.TIME_TAGS:
                if tag in anonymized:
                    anonymized[tag].value = self._modify_time(anonymized[tag].value)
            
            # Xử lý các tag mô tả
            for tag in self.DESCRIPTION_TAGS:
                if tag in anonymized:
                    anonymized[tag].value = f"ANONYMIZED_{get_dicom_type(dataset)}"
            
            # Xử lý các tag UID
            if not keep_uids:
                for tag in self.UID_TAGS:
                    if tag in anonymized:
                        old_uid = anonymized[tag].value
                        if old_uid in self.uid_map:
                            anonymized[tag].value = self.uid_map[old_uid]
                        else:
                            new_uid = pydicom.uid.generate_uid()
                            self.uid_map[old_uid] = new_uid
                            anonymized[tag].value = new_uid
            
            # Thiết lập ID và tên bệnh nhân mới
            if (0x0010, 0x0020) not in anonymized:
                anonymized.add_new((0x0010, 0x0020), 'LO', '')
            if (0x0010, 0x0010) not in anonymized:
                anonymized.add_new((0x0010, 0x0010), 'PN', '')
            
            if new_patient_id is None:
                # Tạo ID mới hoặc sử dụng ID đã tồn tại
                old_id = dataset.get((0x0010, 0x0020), None)
                if old_id is not None and old_id.value in self.id_map:
                    new_patient_id = self.id_map[old_id.value]
                else:
                    new_patient_id = f"ANONYM{str(uuid.uuid4())[:8].upper()}"
                    if old_id is not None:
                        self.id_map[old_id.value] = new_patient_id
            
            if new_patient_name is None:
                new_patient_name = f"ANONYMOUS^PATIENT^{new_patient_id}"
            
            anonymized.PatientID = new_patient_id
            anonymized.PatientName = new_patient_name
            
            # Thêm thông tin ẩn danh
            anonymized.add_new((0x0012, 0x0062), 'CS', 'YES')  # PatientIdentityRemoved
            anonymized.add_new((0x0012, 0x0063), 'LO', 'QuangTPS Anonymizer')  # DeidentificationMethod
            
            # Đặt nhãn thời gian ẩn danh
            current_date = datetime.datetime.now().strftime('%Y%m%d')
            current_time = datetime.datetime.now().strftime('%H%M%S')
            anonymized.add_new((0x0008, 0x0012), 'DA', current_date)  # InstanceCreationDate
            anonymized.add_new((0x0008, 0x0013), 'TM', current_time)  # InstanceCreationTime
            
            return anonymized
            
        except Exception as e:
            logger.error(f"Error anonymizing DICOM dataset: {str(e)}")
            raise DicomError(f"Error anonymizing DICOM dataset: {str(e)}")
    
    def anonymize_file(self, input_file: str, output_file: str,
                      new_patient_id: str = None,
                      new_patient_name: str = None,
                      keep_uids: bool = False) -> None:
        """
        Ẩn danh file DICOM.
        
        Parameters
        ----------
        input_file : str
            Đường dẫn đến file DICOM đầu vào
        output_file : str
            Đường dẫn đến file DICOM đầu ra
        new_patient_id : str, optional
            ID bệnh nhân mới, nếu None sẽ tạo một ID ngẫu nhiên
        new_patient_name : str, optional
            Tên bệnh nhân mới, nếu None sẽ tạo một tên ngẫu nhiên
        keep_uids : bool, optional
            Giữ lại các UID gốc hay không
            
        Raises
        ------
        DicomError
            Nếu không thể ẩn danh file
        """
        try:
            # Đọc file DICOM
            ds = pydicom.dcmread(input_file)
            
            # Ẩn danh dataset
            anonymized = self.anonymize_dataset(ds, new_patient_id, new_patient_name, keep_uids)
            
            # Tạo thư mục đầu ra nếu chưa tồn tại
            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Lưu file đã ẩn danh
            anonymized.save_as(output_file)
            
            logger.info(f"Anonymized DICOM file: {input_file} -> {output_file}")
            
        except Exception as e:
            logger.error(f"Error anonymizing DICOM file: {str(e)}")
            raise DicomError(f"Error anonymizing DICOM file: {str(e)}")
    
    def anonymize_directory(self, input_dir: str, output_dir: str,
                         new_patient_id: str = None,
                         new_patient_name: str = None,
                         keep_uids: bool = False,
                         file_pattern: str = "*.dcm") -> List[str]:
        """
        Ẩn danh tất cả các file DICOM trong một thư mục.
        
        Parameters
        ----------
        input_dir : str
            Thư mục chứa các file DICOM đầu vào
        output_dir : str
            Thư mục đầu ra
        new_patient_id : str, optional
            ID bệnh nhân mới, nếu None sẽ tạo một ID ngẫu nhiên
        new_patient_name : str, optional
            Tên bệnh nhân mới, nếu None sẽ tạo một tên ngẫu nhiên
        keep_uids : bool, optional
            Giữ lại các UID gốc hay không
        file_pattern : str, optional
            Mẫu tên file để lọc, mặc định là "*.dcm"
            
        Returns
        -------
        List[str]
            Danh sách các file đã ẩn danh
            
        Raises
        ------
        DicomError
            Nếu không thể ẩn danh thư mục
        """
        try:
            import glob
            
            # Tạo thư mục đầu ra nếu chưa tồn tại
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Tìm tất cả các file DICOM trong thư mục đầu vào
            file_pattern_path = os.path.join(input_dir, file_pattern)
            input_files = glob.glob(file_pattern_path)
            
            if not input_files:
                logger.warning(f"No DICOM files found in {input_dir} matching pattern {file_pattern}")
                return []
            
            # Danh sách các file đã ẩn danh
            output_files = []
            
            # Ẩn danh từng file
            for input_file in input_files:
                # Tạo đường dẫn file đầu ra
                rel_path = os.path.relpath(input_file, input_dir)
                output_file = os.path.join(output_dir, rel_path)
                
                # Ẩn danh file
                self.anonymize_file(input_file, output_file, new_patient_id, new_patient_name, keep_uids)
                output_files.append(output_file)
            
            logger.info(f"Anonymized {len(output_files)} DICOM files from {input_dir} to {output_dir}")
            return output_files
            
        except Exception as e:
            logger.error(f"Error anonymizing DICOM directory: {str(e)}")
            raise DicomError(f"Error anonymizing DICOM directory: {str(e)}")
    
    def _modify_date(self, date_str: str) -> str:
        """
        Sửa đổi chuỗi ngày DICOM.
        
        Parameters
        ----------
        date_str : str
            Chuỗi ngày DICOM (YYYYMMDD)
            
        Returns
        -------
        str
            Chuỗi ngày đã sửa đổi
        """
        if not date_str or len(date_str) != 8:
            return date_str
        
        try:
            # Chuyển đổi chuỗi ngày thành đối tượng datetime
            year = int(date_str[:4])
            month = int(date_str[4:6])
            day = int(date_str[6:8])
            date_obj = datetime.datetime(year, month, day)
            
            # Áp dụng độ lệch ngày
            modified_date = date_obj + datetime.timedelta(days=self.date_offset)
            
            # Chuyển đổi trở lại chuỗi
            return modified_date.strftime("%Y%m%d")
            
        except:
            return date_str
    
    def _modify_time(self, time_str: str) -> str:
        """
        Sửa đổi chuỗi thời gian DICOM.
        
        Parameters
        ----------
        time_str : str
            Chuỗi thời gian DICOM (HHMMSS.FFFFFF)
            
        Returns
        -------
        str
            Chuỗi thời gian đã sửa đổi
        """
        if not time_str:
            return time_str
        
        # Giữ lại phần microsecond nếu có
        microsecond = ""
        if '.' in time_str:
            parts = time_str.split('.')
            time_str = parts[0]
            microsecond = f".{parts[1]}"
        
        if len(time_str) < 6:
            return time_str
        
        try:
            # Chuyển đổi chuỗi thời gian thành số giờ, phút, giây
            hour = int(time_str[:2])
            minute = int(time_str[2:4])
            second = int(time_str[4:6])
            
            # Tạo đối tượng thời gian gốc (sử dụng một ngày bất kỳ)
            time_obj = datetime.datetime(1900, 1, 1, hour, minute, second)
            
            # Thay đổi thời gian một cách ngẫu nhiên
            random_seconds = random.randint(-3600, 3600)  # +/- 1 giờ
            modified_time = time_obj + datetime.timedelta(seconds=random_seconds)
            
            # Chuyển đổi trở lại chuỗi
            return modified_time.strftime("%H%M%S") + microsecond
            
        except:
            return time_str
