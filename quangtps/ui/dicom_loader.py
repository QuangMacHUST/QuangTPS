#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module cung cấp chức năng đọc và xử lý file DICOM cho hệ thống QuangTPS.

Module này bao gồm các lớp và hàm để tải, xử lý và tổ chức dữ liệu DICOM
từ các nguồn khác nhau, bao gồm file riêng lẻ và series DICOM.
"""

import os
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from collections import defaultdict

try:
    import pydicom
    from pydicom.errors import InvalidDicomError
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False

try:
    import SimpleITK as sitk
    SITK_AVAILABLE = True
except ImportError:
    SITK_AVAILABLE = False

logger = logging.getLogger(__name__)


class DicomSeries:
    """Lớp đại diện cho một chuỗi các file DICOM."""
    
    def __init__(self, series_id: str = "", description: str = ""):
        """
        Khởi tạo một chuỗi DICOM.
        
        Parameters
        ----------
        series_id : str
            ID của chuỗi DICOM
        description : str
            Mô tả về chuỗi DICOM
        """
        self.series_id = series_id
        self.description = description
        self.files = []  # Danh sách đường dẫn đến các file DICOM trong chuỗi
        self.metadata = {}  # Metadata của chuỗi DICOM
        self.modality = ""  # Dạng hình ảnh (CT, MR, PT, v.v.)
        self.patient_name = ""
        self.patient_id = ""
        self.study_date = ""
        self.study_description = ""
        
        # Dữ liệu hình ảnh
        self.image_data = None  # Dữ liệu 3D
        self.image_position = None  # Vị trí voxel đầu tiên
        self.image_orientation = None  # Hướng của hình ảnh
        self.pixel_spacing = None  # Khoảng cách giữa các pixel
        self.slice_thickness = None  # Độ dày lát cắt
    
    def add_file(self, file_path: str) -> bool:
        """
        Thêm một file DICOM vào chuỗi.
        
        Parameters
        ----------
        file_path : str
            Đường dẫn đến file DICOM
        
        Returns
        -------
        bool
            True nếu thêm thành công, False nếu thất bại
        """
        if not PYDICOM_AVAILABLE:
            logger.error("Không thể thêm file DICOM vì thiếu thư viện pydicom")
            return False
        
        try:
            # Kiểm tra file có phải là DICOM hợp lệ không
            dicom_data = pydicom.dcmread(file_path, force=False)
            
            # Kiểm tra xem file có thuộc series này không
            if hasattr(dicom_data, 'SeriesInstanceUID') and dicom_data.SeriesInstanceUID:
                if not self.series_id:
                    # Nếu series_id chưa được thiết lập, sử dụng ID từ file đầu tiên
                    self.series_id = dicom_data.SeriesInstanceUID
                    
                    # Thiết lập các thông tin khác từ file đầu tiên
                    if hasattr(dicom_data, 'SeriesDescription') and dicom_data.SeriesDescription:
                        self.description = dicom_data.SeriesDescription
                    
                    if hasattr(dicom_data, 'Modality') and dicom_data.Modality:
                        self.modality = dicom_data.Modality
                    
                    if hasattr(dicom_data, 'PatientName') and dicom_data.PatientName:
                        self.patient_name = str(dicom_data.PatientName)
                    
                    if hasattr(dicom_data, 'PatientID') and dicom_data.PatientID:
                        self.patient_id = dicom_data.PatientID
                    
                    if hasattr(dicom_data, 'StudyDate') and dicom_data.StudyDate:
                        self.study_date = dicom_data.StudyDate
                    
                    if hasattr(dicom_data, 'StudyDescription') and dicom_data.StudyDescription:
                        self.study_description = dicom_data.StudyDescription
                
                if dicom_data.SeriesInstanceUID == self.series_id:
                    # File thuộc series này
                    self.files.append(file_path)
                    return True
                else:
                    # File không thuộc series này
                    return False
            else:
                # File không có SeriesInstanceUID
                logger.warning(f"File {file_path} không có SeriesInstanceUID")
                return False
        
        except (InvalidDicomError, Exception) as e:
            logger.error(f"Lỗi khi đọc file DICOM {file_path}: {str(e)}")
            return False
    
    def load_image_data(self) -> bool:
        """
        Tải dữ liệu hình ảnh từ các file DICOM.
        
        Returns
        -------
        bool
            True nếu tải thành công, False nếu thất bại
        """
        if not self.files:
            logger.error("Không có file DICOM nào để tải")
            return False
        
        if not PYDICOM_AVAILABLE:
            logger.error("Không thể tải dữ liệu hình ảnh vì thiếu thư viện pydicom")
            return False
        
        try:
            if SITK_AVAILABLE:
                # Sử dụng SimpleITK để tải dữ liệu (tốt hơn cho các chuỗi lớn)
                return self._load_with_sitk()
            else:
                # Sử dụng pydicom trực tiếp
                return self._load_with_pydicom()
        
        except Exception as e:
            logger.error(f"Lỗi khi tải dữ liệu hình ảnh: {str(e)}")
            return False
    
    def _load_with_sitk(self) -> bool:
        """
        Tải dữ liệu hình ảnh bằng SimpleITK.
        
        Returns
        -------
        bool
            True nếu tải thành công, False nếu thất bại
        """
        try:
            # Đọc chuỗi
            reader = sitk.ImageSeriesReader()
            reader.SetFileNames(self.files)
            image = reader.Execute()
            
            # Chuyển đổi sang numpy array
            self.image_data = sitk.GetArrayFromImage(image)
            
            # Lưu metadata
            self.pixel_spacing = image.GetSpacing()[:2]  # (x, y)
            self.slice_thickness = image.GetSpacing()[2]
            
            # Lưu thông tin vị trí và hướng
            self.image_position = image.GetOrigin()
            self.image_orientation = image.GetDirection()
            
            return True
        
        except Exception as e:
            logger.error(f"Lỗi khi tải dữ liệu với SimpleITK: {str(e)}")
            return False
    
    def _load_with_pydicom(self) -> bool:
        """
        Tải dữ liệu hình ảnh bằng pydicom.
        
        Returns
        -------
        bool
            True nếu tải thành công, False nếu thất bại
        """
        try:
            # Đọc tất cả các file DICOM
            slices = [pydicom.dcmread(file) for file in self.files]
            
            # Sắp xếp các lát cắt theo vị trí
            if hasattr(slices[0], 'ImagePositionPatient') and slices[0].ImagePositionPatient:
                slices.sort(key=lambda s: s.ImagePositionPatient[2])
            elif hasattr(slices[0], 'SliceLocation') and slices[0].SliceLocation:
                slices.sort(key=lambda s: s.SliceLocation)
            else:
                # Không thể sắp xếp theo vị trí
                logger.warning("Không thể sắp xếp các lát cắt theo vị trí")
                return False
            
            # Lấy kích thước pixel và số lát cắt
            pixel_shape = slices[0].pixel_array.shape
            num_slices = len(slices)
            
            # Tạo mảng 3D
            self.image_data = np.zeros((num_slices, *pixel_shape), dtype=np.int16)
            
            # Điền dữ liệu vào mảng 3D
            for i, s in enumerate(slices):
                self.image_data[i, :, :] = s.pixel_array
            
            # Rescale dữ liệu nếu cần thiết (đối với CT)
            if hasattr(slices[0], 'RescaleIntercept') and hasattr(slices[0], 'RescaleSlope'):
                intercept = slices[0].RescaleIntercept
                slope = slices[0].RescaleSlope
                self.image_data = self.image_data * slope + intercept
            
            # Lưu metadata
            if hasattr(slices[0], 'PixelSpacing') and slices[0].PixelSpacing:
                self.pixel_spacing = slices[0].PixelSpacing
            
            if hasattr(slices[0], 'SliceThickness') and slices[0].SliceThickness:
                self.slice_thickness = slices[0].SliceThickness
            
            if hasattr(slices[0], 'ImagePositionPatient') and slices[0].ImagePositionPatient:
                self.image_position = slices[0].ImagePositionPatient
            
            if hasattr(slices[0], 'ImageOrientationPatient') and slices[0].ImageOrientationPatient:
                self.image_orientation = slices[0].ImageOrientationPatient
            
            return True
        
        except Exception as e:
            logger.error(f"Lỗi khi tải dữ liệu với pydicom: {str(e)}")
            return False
    
    def get_metadata_summary(self) -> Dict[str, str]:
        """
        Trả về tóm tắt metadata của chuỗi DICOM.
        
        Returns
        -------
        Dict[str, str]
            Dictionary chứa các thông tin metadata chính
        """
        summary = {
            "Series ID": self.series_id,
            "Description": self.description,
            "Modality": self.modality,
            "Patient Name": self.patient_name,
            "Patient ID": self.patient_id,
            "Study Date": self.study_date,
            "Study Description": self.study_description,
            "Number of Files": str(len(self.files))
        }
        
        if self.image_data is not None:
            summary["Image Dimensions"] = f"{self.image_data.shape}"
        
        if self.pixel_spacing is not None:
            summary["Pixel Spacing"] = f"{self.pixel_spacing}"
        
        if self.slice_thickness is not None:
            summary["Slice Thickness"] = f"{self.slice_thickness}"
        
        return summary
    
    def get_slice(self, index: int, plane: str = 'axial') -> Optional[np.ndarray]:
        """
        Lấy một lát cắt từ dữ liệu 3D.
        
        Parameters
        ----------
        index : int
            Chỉ số lát cắt
        plane : str
            Mặt phẳng ('axial', 'coronal', 'sagittal')
        
        Returns
        -------
        Optional[np.ndarray]
            Dữ liệu lát cắt 2D hoặc None nếu không có dữ liệu
        """
        if self.image_data is None:
            return None
        
        if plane == 'axial':
            if 0 <= index < self.image_data.shape[0]:
                return self.image_data[index, :, :]
        elif plane == 'coronal':
            if 0 <= index < self.image_data.shape[1]:
                return self.image_data[:, index, :]
        elif plane == 'sagittal':
            if 0 <= index < self.image_data.shape[2]:
                return self.image_data[:, :, index]
        
        return None


class DicomLoader:
    """Lớp chịu trách nhiệm tải và quản lý dữ liệu DICOM."""
    
    def __init__(self):
        """Khởi tạo DicomLoader."""
        self.series_list = []  # Danh sách các chuỗi DICOM đã tải
        
        # Kiểm tra các thư viện cần thiết
        self._check_libraries()
    
    def _check_libraries(self):
        """Kiểm tra các thư viện cần thiết đã được cài đặt chưa."""
        if not PYDICOM_AVAILABLE:
            logger.warning("Thư viện pydicom không có sẵn. Một số chức năng có thể không hoạt động.")
        
        if not SITK_AVAILABLE:
            logger.warning("Thư viện SimpleITK không có sẵn. Hiệu suất tải DICOM có thể bị ảnh hưởng.")
    
    def load_dicom_file(self, file_path: str) -> Optional[DicomSeries]:
        """
        Tải một file DICOM và tạo chuỗi mới từ file đó.
        
        Parameters
        ----------
        file_path : str
            Đường dẫn đến file DICOM
        
        Returns
        -------
        Optional[DicomSeries]
            Chuỗi DICOM mới hoặc None nếu tải thất bại
        """
        if not PYDICOM_AVAILABLE:
            logger.error("Không thể tải file DICOM vì thiếu thư viện pydicom")
            return None
        
        try:
            # Tạo chuỗi mới
            series = DicomSeries()
            
            # Thêm file vào chuỗi
            if series.add_file(file_path):
                self.series_list.append(series)
                return series
            else:
                return None
        
        except Exception as e:
            logger.error(f"Lỗi khi tải file DICOM {file_path}: {str(e)}")
            return None
    
    def load_dicom_directory(self, directory_path: str) -> List[DicomSeries]:
        """
        Tải tất cả các file DICOM từ một thư mục và tổ chức thành các chuỗi.
        
        Parameters
        ----------
        directory_path : str
            Đường dẫn đến thư mục chứa các file DICOM
        
        Returns
        -------
        List[DicomSeries]
            Danh sách các chuỗi DICOM đã tải
        """
        if not PYDICOM_AVAILABLE:
            logger.error("Không thể tải thư mục DICOM vì thiếu thư viện pydicom")
            return []
        
        # Danh sách chuỗi mới
        new_series_list = []
        
        try:
            # Dictionary tạm thời lưu trữ các chuỗi theo ID
            series_dict = {}
            
            # Duyệt qua tất cả các file trong thư mục
            for root, _, files in os.walk(directory_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    
                    try:
                        # Kiểm tra file có phải là DICOM hợp lệ không
                        dicom_data = pydicom.dcmread(file_path, force=False)
                        
                        # Kiểm tra xem file có SeriesInstanceUID không
                        if hasattr(dicom_data, 'SeriesInstanceUID') and dicom_data.SeriesInstanceUID:
                            series_id = dicom_data.SeriesInstanceUID
                            
                            # Tạo chuỗi mới nếu cần
                            if series_id not in series_dict:
                                series_description = ""
                                if hasattr(dicom_data, 'SeriesDescription') and dicom_data.SeriesDescription:
                                    series_description = dicom_data.SeriesDescription
                                
                                series_dict[series_id] = DicomSeries(series_id, series_description)
                            
                            # Thêm file vào chuỗi
                            series_dict[series_id].add_file(file_path)
                    
                    except (InvalidDicomError, Exception) as e:
                        # Không phải file DICOM hoặc có lỗi khác
                        logger.debug(f"Bỏ qua file {file_path}: {str(e)}")
                        continue
            
            # Thêm các chuỗi mới vào danh sách
            for series in series_dict.values():
                self.series_list.append(series)
                new_series_list.append(series)
            
            return new_series_list
        
        except Exception as e:
            logger.error(f"Lỗi khi tải thư mục DICOM {directory_path}: {str(e)}")
            return []
    
    def get_series_by_id(self, series_id: str) -> Optional[DicomSeries]:
        """
        Tìm chuỗi DICOM theo ID.
        
        Parameters
        ----------
        series_id : str
            ID của chuỗi DICOM cần tìm
        
        Returns
        -------
        Optional[DicomSeries]
            Chuỗi DICOM hoặc None nếu không tìm thấy
        """
        for series in self.series_list:
            if series.series_id == series_id:
                return series
        
        return None
    
    def clear_series(self):
        """Xóa tất cả các chuỗi DICOM đã tải."""
        self.series_list.clear()


def get_dicom_file_metadata(file_path: str) -> Dict[str, Any]:
    """
    Trích xuất metadata từ file DICOM.
    
    Parameters
    ----------
    file_path : str
        Đường dẫn đến file DICOM
    
    Returns
    -------
    Dict[str, Any]
        Dictionary chứa metadata
    """
    if not PYDICOM_AVAILABLE:
        logger.error("Không thể trích xuất metadata vì thiếu thư viện pydicom")
        return {}
    
    try:
        dicom_data = pydicom.dcmread(file_path, force=True)
        
        metadata = {
            "FileName": os.path.basename(file_path),
            "FilePath": file_path
        }
        
        # Thông tin cơ bản
        if hasattr(dicom_data, 'PatientName') and dicom_data.PatientName:
            metadata["PatientName"] = str(dicom_data.PatientName)
        
        if hasattr(dicom_data, 'PatientID') and dicom_data.PatientID:
            metadata["PatientID"] = dicom_data.PatientID
        
        if hasattr(dicom_data, 'PatientBirthDate') and dicom_data.PatientBirthDate:
            metadata["PatientBirthDate"] = dicom_data.PatientBirthDate
        
        if hasattr(dicom_data, 'PatientSex') and dicom_data.PatientSex:
            metadata["PatientSex"] = dicom_data.PatientSex
        
        # Thông tin nghiên cứu
        if hasattr(dicom_data, 'StudyInstanceUID') and dicom_data.StudyInstanceUID:
            metadata["StudyInstanceUID"] = dicom_data.StudyInstanceUID
        
        if hasattr(dicom_data, 'StudyDate') and dicom_data.StudyDate:
            metadata["StudyDate"] = dicom_data.StudyDate
        
        if hasattr(dicom_data, 'StudyTime') and dicom_data.StudyTime:
            metadata["StudyTime"] = dicom_data.StudyTime
        
        if hasattr(dicom_data, 'StudyDescription') and dicom_data.StudyDescription:
            metadata["StudyDescription"] = dicom_data.StudyDescription
        
        # Thông tin chuỗi
        if hasattr(dicom_data, 'SeriesInstanceUID') and dicom_data.SeriesInstanceUID:
            metadata["SeriesInstanceUID"] = dicom_data.SeriesInstanceUID
        
        if hasattr(dicom_data, 'SeriesNumber') and dicom_data.SeriesNumber:
            metadata["SeriesNumber"] = dicom_data.SeriesNumber
        
        if hasattr(dicom_data, 'SeriesDescription') and dicom_data.SeriesDescription:
            metadata["SeriesDescription"] = dicom_data.SeriesDescription
        
        # Thông tin hình ảnh
        if hasattr(dicom_data, 'Modality') and dicom_data.Modality:
            metadata["Modality"] = dicom_data.Modality
        
        if hasattr(dicom_data, 'Manufacturer') and dicom_data.Manufacturer:
            metadata["Manufacturer"] = dicom_data.Manufacturer
        
        if hasattr(dicom_data, 'InstitutionName') and dicom_data.InstitutionName:
            metadata["InstitutionName"] = dicom_data.InstitutionName
        
        if hasattr(dicom_data, 'PixelSpacing') and dicom_data.PixelSpacing:
            metadata["PixelSpacing"] = dicom_data.PixelSpacing
        
        if hasattr(dicom_data, 'SliceThickness') and dicom_data.SliceThickness:
            metadata["SliceThickness"] = dicom_data.SliceThickness
        
        if hasattr(dicom_data, 'ImagePositionPatient') and dicom_data.ImagePositionPatient:
            metadata["ImagePositionPatient"] = dicom_data.ImagePositionPatient
        
        if hasattr(dicom_data, 'Rows') and dicom_data.Rows:
            metadata["Rows"] = dicom_data.Rows
        
        if hasattr(dicom_data, 'Columns') and dicom_data.Columns:
            metadata["Columns"] = dicom_data.Columns
        
        return metadata
    
    except Exception as e:
        logger.error(f"Lỗi khi trích xuất metadata từ {file_path}: {str(e)}")
        return {"Error": str(e)}
