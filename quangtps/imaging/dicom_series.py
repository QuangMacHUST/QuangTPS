#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module chuỗi DICOM.
"""

import os
import logging
import numpy as np
import SimpleITK as sitk
import pydicom
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path

logger = logging.getLogger(__name__)

class DicomSeries:
    """
    Lớp đại diện cho một chuỗi hình ảnh DICOM.
    
    Lưu trữ dữ liệu hình ảnh 3D và các thông tin liên quan
    từ các file DICOM trong cùng một chuỗi.
    """
    
    def __init__(self, series_uid: str = None, study_uid: str = None, patient_id: str = None):
        """
        Khởi tạo đối tượng DicomSeries.
        
        Parameters
        ----------
        series_uid : str, optional
            UID của chuỗi DICOM
        study_uid : str, optional
            UID của nghiên cứu chứa chuỗi
        patient_id : str, optional
            ID của bệnh nhân
        """
        self.series_uid = series_uid
        self.study_uid = study_uid
        self.patient_id = patient_id
        
        # Dữ liệu hình ảnh
        self.image_data = None
        
        # Thông tin vị trí và kích thước
        self.origin = (0, 0, 0)
        self.spacing = (1, 1, 1)
        self.direction = np.eye(3)
        
        # Metadata
        self.modality = None
        self.series_description = None
        self.series_number = None
        self.acquisition_date = None
        self.acquisition_time = None
        self.manufacturer = None
        self.institution_name = None
        self.slice_thickness = None
        self.num_slices = 0
        
        # Danh sách các file DICOM
        self.files = []
        
        # Các thuộc tính bổ sung
        self.metadata = {}
        
        # SimpleITK Image
        self._sitk_image = None
    
    def load_from_files(self, file_paths: List[str]) -> bool:
        """
        Tải dữ liệu hình ảnh từ danh sách các file DICOM.
        
        Parameters
        ----------
        file_paths : List[str]
            Danh sách đường dẫn đến các file DICOM
            
        Returns
        -------
        bool
            True nếu tải thành công, False nếu thất bại
        """
        if not file_paths:
            logger.error("Không có file DICOM nào được cung cấp")
            return False
        
        # Lưu danh sách file
        self.files = file_paths
        
        # Đầu tiên thử tải bằng SimpleITK
        success = self._load_with_sitk(file_paths)
        
        # Nếu không thành công, thử tải bằng pydicom
        if not success:
            logger.warning(f"Không thể tải file DICOM bằng SimpleITK, thử với pydicom")
            success = self._load_with_pydicom(file_paths)
        
        if not success:
            logger.error("Không thể tải file DICOM với cả SimpleITK và pydicom")
            return False
        
        logger.info(f"Đã tải thành công chuỗi DICOM với {self.num_slices} lát cắt")
        return True
    
    def _load_with_sitk(self, file_paths: List[str]) -> bool:
        """
        Tải chuỗi DICOM bằng SimpleITK.
        
        Parameters
        ----------
        file_paths : List[str]
            Danh sách đường dẫn đến các file DICOM
            
        Returns
        -------
        bool
            True nếu tải thành công, False nếu thất bại
        """
        try:
            # Tạo reader
            reader = sitk.ImageSeriesReader()
            reader.SetFileNames(file_paths)
            
            # Đọc ảnh
            self._sitk_image = reader.Execute()
            
            # Lấy dữ liệu hình ảnh
            self.image_data = sitk.GetArrayFromImage(self._sitk_image)
            
            # Cập nhật thông tin
            self.spacing = self._sitk_image.GetSpacing()
            self.origin = self._sitk_image.GetOrigin()
            self.direction = np.array(self._sitk_image.GetDirection()).reshape(3, 3)
            
            # Cập nhật số lát cắt
            self.num_slices = self.image_data.shape[0]
            
            # Trích xuất metadata từ file đầu tiên
            self._extract_metadata_from_sitk()
            
            return True
            
        except Exception as e:
            logger.error(f"Lỗi khi tải DICOM với SimpleITK: {e}")
            return False
    
    def _load_with_pydicom(self, file_paths: List[str]) -> bool:
        """
        Tải chuỗi DICOM bằng pydicom.
        
        Parameters
        ----------
        file_paths : List[str]
            Danh sách đường dẫn đến các file DICOM
            
        Returns
        -------
        bool
            True nếu tải thành công, False nếu thất bại
        """
        try:
            # Đọc tất cả các file
            slices = []
            for file_path in file_paths:
                try:
                    dcm = pydicom.dcmread(file_path, force=True)
                    slices.append(dcm)
                except Exception as e:
                    logger.warning(f"Không thể đọc file {file_path}: {e}")
            
            if not slices:
                logger.error("Không thể đọc bất kỳ file DICOM nào")
                return False
            
            # Sắp xếp theo vị trí
            try:
                slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
            except Exception as e:
                logger.warning(f"Không thể sắp xếp theo vị trí, sử dụng thứ tự file: {e}")
            
            # Trích xuất metadata từ lát cắt đầu tiên
            self._extract_metadata_from_pydicom(slices[0])
            
            # Tính khoảng cách giữa các lát cắt
            try:
                slice_thickness = slices[0].SliceThickness
            except:
                try:
                    # Tính từ vị trí
                    if len(slices) > 1:
                        slice_thickness = abs(float(slices[1].ImagePositionPatient[2]) - 
                                             float(slices[0].ImagePositionPatient[2]))
                    else:
                        slice_thickness = 1.0
                except:
                    logger.warning("Không thể xác định độ dày lát cắt, sử dụng giá trị mặc định 1.0")
                    slice_thickness = 1.0
            
            # Cập nhật thông tin
            self.slice_thickness = slice_thickness
            
            # Tạo mảng 3D từ các lát cắt
            img_shape = slices[0].pixel_array.shape
            self.image_data = np.zeros((len(slices), img_shape[0], img_shape[1]), dtype=slices[0].pixel_array.dtype)
            
            # Đọc dữ liệu từ từng lát cắt
            for i, dcm in enumerate(slices):
                try:
                    self.image_data[i, :, :] = dcm.pixel_array
                except Exception as e:
                    logger.warning(f"Không thể đọc pixel_array từ lát cắt {i}: {e}")
                    self.image_data[i, :, :] = np.zeros(img_shape, dtype=self.image_data.dtype)
            
            # Áp dụng RescaleSlope và RescaleIntercept nếu có
            try:
                if hasattr(slices[0], 'RescaleSlope') and hasattr(slices[0], 'RescaleIntercept'):
                    slope = float(slices[0].RescaleSlope)
                    intercept = float(slices[0].RescaleIntercept)
                    self.image_data = self.image_data * slope + intercept
            except Exception as e:
                logger.warning(f"Không thể áp dụng rescale: {e}")
            
            # Cập nhật thông tin
            try:
                # Pixel spacing
                ps = slices[0].PixelSpacing
                self.spacing = (float(ps[1]), float(ps[0]), slice_thickness)
                
                # Origin
                pos = slices[0].ImagePositionPatient
                self.origin = (float(pos[0]), float(pos[1]), float(pos[2]))
                
                # Direction (identity matrix nếu không có thông tin)
                self.direction = np.eye(3)
                
            except Exception as e:
                logger.warning(f"Không thể trích xuất thông tin hình học: {e}")
                self.spacing = (1.0, 1.0, slice_thickness)
                self.origin = (0, 0, 0)
            
            # Cập nhật số lát cắt
            self.num_slices = len(slices)
            
            return True
            
        except Exception as e:
            logger.error(f"Lỗi khi tải DICOM với pydicom: {e}")
            return False
    
    def _extract_metadata_from_sitk(self) -> None:
        """
        Trích xuất metadata từ hình ảnh SimpleITK.
        """
        try:
            # Đọc metadata từ file đầu tiên bằng pydicom để lấy thông tin chi tiết
            first_file = self.files[0]
            try:
                dcm = pydicom.dcmread(first_file, force=True)
                
                # Lấy thông tin cơ bản
                self.modality = str(dcm.Modality) if hasattr(dcm, 'Modality') else None
                self.series_description = str(dcm.SeriesDescription) if hasattr(dcm, 'SeriesDescription') else None
                self.series_number = int(dcm.SeriesNumber) if hasattr(dcm, 'SeriesNumber') else None
                self.series_uid = str(dcm.SeriesInstanceUID) if hasattr(dcm, 'SeriesInstanceUID') else self.series_uid
                self.study_uid = str(dcm.StudyInstanceUID) if hasattr(dcm, 'StudyInstanceUID') else self.study_uid
                self.patient_id = str(dcm.PatientID) if hasattr(dcm, 'PatientID') else self.patient_id
                
                # Thông tin thêm
                self.manufacturer = str(dcm.Manufacturer) if hasattr(dcm, 'Manufacturer') else None
                self.institution_name = str(dcm.InstitutionName) if hasattr(dcm, 'InstitutionName') else None
                
                # Thông tin thời gian
                if hasattr(dcm, 'AcquisitionDate') and hasattr(dcm, 'AcquisitionTime'):
                    try:
                        date_str = str(dcm.AcquisitionDate)
                        time_str = str(dcm.AcquisitionTime).split('.')[0]
                        self.acquisition_date = date_str
                        self.acquisition_time = time_str
                    except Exception as e:
                        logger.warning(f"Không thể phân tích thời gian: {e}")
                
                # Độ dày lát cắt
                self.slice_thickness = float(dcm.SliceThickness) if hasattr(dcm, 'SliceThickness') else None
                
                # Các metadata bổ sung
                for elem in dcm:
                    try:
                        tag_name = elem.name
                        if elem.value is not None and tag_name not in ['Pixel Data']:
                            self.metadata[tag_name] = str(elem.value)
                    except:
                        pass
                        
            except Exception as e:
                logger.warning(f"Không thể đọc metadata chi tiết từ file {first_file}: {e}")
                
                # Sử dụng metadata từ SimpleITK
                for key in self._sitk_image.GetMetaDataKeys():
                    self.metadata[key] = self._sitk_image.GetMetaData(key)
                
                # Cố gắng lấy các thông tin cơ bản
                self.modality = self.metadata.get('0008|0060', None)
                self.series_description = self.metadata.get('0008|103e', None)
                self.series_number = self.metadata.get('0020|0011', None)
                self.series_uid = self.metadata.get('0020|000e', self.series_uid)
                self.study_uid = self.metadata.get('0020|000d', self.study_uid)
                self.patient_id = self.metadata.get('0010|0020', self.patient_id)
            
        except Exception as e:
            logger.error(f"Lỗi khi trích xuất metadata: {e}")
    
    def _extract_metadata_from_pydicom(self, dcm: pydicom.dataset.FileDataset) -> None:
        """
        Trích xuất metadata từ đối tượng pydicom.
        
        Parameters
        ----------
        dcm : pydicom.dataset.FileDataset
            Đối tượng DICOM dataset
        """
        try:
            # Lấy thông tin cơ bản
            self.modality = str(dcm.Modality) if hasattr(dcm, 'Modality') else None
            self.series_description = str(dcm.SeriesDescription) if hasattr(dcm, 'SeriesDescription') else None
            self.series_number = int(dcm.SeriesNumber) if hasattr(dcm, 'SeriesNumber') else None
            self.series_uid = str(dcm.SeriesInstanceUID) if hasattr(dcm, 'SeriesInstanceUID') else self.series_uid
            self.study_uid = str(dcm.StudyInstanceUID) if hasattr(dcm, 'StudyInstanceUID') else self.study_uid
            self.patient_id = str(dcm.PatientID) if hasattr(dcm, 'PatientID') else self.patient_id
            
            # Thông tin thêm
            self.manufacturer = str(dcm.Manufacturer) if hasattr(dcm, 'Manufacturer') else None
            self.institution_name = str(dcm.InstitutionName) if hasattr(dcm, 'InstitutionName') else None
            
            # Thông tin thời gian
            if hasattr(dcm, 'AcquisitionDate') and hasattr(dcm, 'AcquisitionTime'):
                try:
                    date_str = str(dcm.AcquisitionDate)
                    time_str = str(dcm.AcquisitionTime).split('.')[0]
                    self.acquisition_date = date_str
                    self.acquisition_time = time_str
                except Exception as e:
                    logger.warning(f"Không thể phân tích thời gian: {e}")
            
            # Độ dày lát cắt
            self.slice_thickness = float(dcm.SliceThickness) if hasattr(dcm, 'SliceThickness') else None
            
            # Các metadata bổ sung
            for elem in dcm:
                try:
                    tag_name = elem.name
                    if elem.value is not None and tag_name not in ['Pixel Data']:
                        self.metadata[tag_name] = str(elem.value)
                except:
                    pass
                    
        except Exception as e:
            logger.error(f"Lỗi khi trích xuất metadata từ pydicom: {e}")
    
    def get_sitk_image(self) -> Optional[sitk.Image]:
        """
        Lấy đối tượng SimpleITK Image từ chuỗi DICOM.
        
        Returns
        -------
        SimpleITK.Image or None
            Đối tượng hình ảnh SimpleITK hoặc None nếu không có
        """
        if self._sitk_image is not None:
            return self._sitk_image
        
        # Tạo từ dữ liệu nếu có
        if self.image_data is not None:
            try:
                img = sitk.GetImageFromArray(self.image_data)
                img.SetSpacing(self.spacing)
                img.SetOrigin(self.origin)
                img.SetDirection(self.direction.flatten())
                
                # Thêm metadata
                for key, value in self.metadata.items():
                    if isinstance(key, str) and isinstance(value, str):
                        img.SetMetaData(key, value)
                
                self._sitk_image = img
                return img
            except Exception as e:
                logger.error(f"Lỗi khi tạo SimpleITK Image: {e}")
                return None
        
        return None
    
    def get_slice(self, index: int, orientation: str = "axial") -> np.ndarray:
        """
        Lấy lát cắt 2D từ khối dữ liệu 3D theo hướng chỉ định.
        
        Parameters
        ----------
        index : int
            Chỉ số của lát cắt
        orientation : str, optional
            Hướng của lát cắt ("axial", "sagittal", "coronal")
            
        Returns
        -------
        np.ndarray
            Dữ liệu lát cắt 2D
        """
        if self.image_data is None:
            logger.warning("Không có dữ liệu hình ảnh")
            return None
        
        try:
            # Chuyển orientation về chữ thường để so sánh
            orientation = orientation.lower()
            
            # Kiểm tra hợp lệ
            if orientation not in ["axial", "sagittal", "coronal"]:
                logger.warning(f"Hướng không hợp lệ: {orientation}, sử dụng axial")
                orientation = "axial"
            
            # Kiểm tra chỉ số
            if not isinstance(index, (int, np.integer)):
                try:
                    index = int(index)
                except:
                    logger.warning(f"Chỉ số không hợp lệ: {index}, sử dụng 0")
                    index = 0
            
            # Lấy lát cắt theo hướng
            if orientation == "axial":
                # Kiểm tra chỉ số nằm trong phạm vi
                if index < 0 or index >= self.image_data.shape[0]:
                    logger.warning(f"Chỉ số lát cắt axial {index} nằm ngoài phạm vi [0, {self.image_data.shape[0]-1}]")
                    index = max(0, min(index, self.image_data.shape[0]-1))
                
                return self.image_data[index, :, :]
                
            elif orientation == "sagittal":
                # Kiểm tra chỉ số nằm trong phạm vi
                if index < 0 or index >= self.image_data.shape[2]:
                    logger.warning(f"Chỉ số lát cắt sagittal {index} nằm ngoài phạm vi [0, {self.image_data.shape[2]-1}]")
                    index = max(0, min(index, self.image_data.shape[2]-1))
                
                return self.image_data[:, :, index]
                
            elif orientation == "coronal":
                # Kiểm tra chỉ số nằm trong phạm vi
                if index < 0 or index >= self.image_data.shape[1]:
                    logger.warning(f"Chỉ số lát cắt coronal {index} nằm ngoài phạm vi [0, {self.image_data.shape[1]-1}]")
                    index = max(0, min(index, self.image_data.shape[1]-1))
                
                return self.image_data[:, index, :]
                
        except Exception as e:
            logger.error(f"Lỗi khi lấy lát cắt {orientation} tại chỉ số {index}: {e}")
            
        return None
    
    def get_value_at_point(self, x: float, y: float, z: float) -> Optional[float]:
        """
        Lấy giá trị tại điểm chỉ định trong khối dữ liệu 3D.
        
        Parameters
        ----------
        x : float
            Tọa độ x trong không gian thực (mm)
        y : float
            Tọa độ y trong không gian thực (mm)
        z : float
            Tọa độ z trong không gian thực (mm)
            
        Returns
        -------
        float or None
            Giá trị tại điểm chỉ định, hoặc None nếu tọa độ không hợp lệ
        """
        if self.image_data is None:
            logger.warning("Không có dữ liệu hình ảnh")
            return None
        
        try:
            # Chuyển tọa độ thực sang tọa độ voxel
            i = int((z - self.origin[2]) / self.spacing[2])
            j = int((y - self.origin[1]) / self.spacing[1])
            k = int((x - self.origin[0]) / self.spacing[0])
            
            # Kiểm tra tọa độ nằm trong phạm vi
            shape = self.image_data.shape
            if (i < 0 or i >= shape[0] or
                j < 0 or j >= shape[1] or
                k < 0 or k >= shape[2]):
                logger.warning(f"Tọa độ ({x}, {y}, {z}) -> ({k}, {j}, {i}) nằm ngoài phạm vi {shape}")
                return None
            
            return float(self.image_data[i, j, k])
            
        except Exception as e:
            logger.error(f"Lỗi khi lấy giá trị tại điểm ({x}, {y}, {z}): {e}")
            return None
    
    def create_from_image(self, image_data: np.ndarray, metadata: Dict[str, Any] = None) -> bool:
        """
        Tạo chuỗi DICOM từ dữ liệu hình ảnh và metadata.
        
        Parameters
        ----------
        image_data : np.ndarray
            Dữ liệu hình ảnh 3D
        metadata : Dict[str, Any], optional
            Metadata của hình ảnh
            
        Returns
        -------
        bool
            True nếu tạo thành công, False nếu thất bại
        """
        if image_data is None:
            logger.error("Không thể tạo chuỗi DICOM từ dữ liệu hình ảnh rỗng")
            return False
        
        try:
            self.image_data = image_data
            
            # Kiểm tra kích thước
            if len(image_data.shape) != 3:
                logger.warning(f"Dữ liệu hình ảnh có kích thước không hợp lệ: {image_data.shape}, cần kích thước 3D")
                return False
            
            # Cập nhật số lát cắt
            self.num_slices = self.image_data.shape[0]
            
            # Cập nhật metadata
            if metadata:
                # Cập nhật metadata
                self.metadata = metadata.copy()
                
                # Cập nhật các thuộc tính cụ thể
                if 'modality' in metadata:
                    self.modality = metadata['modality']
                
                if 'series_description' in metadata:
                    self.series_description = metadata['series_description']
                    
                if 'spacing' in metadata:
                    self.spacing = metadata['spacing']
                
                if 'origin' in metadata:
                    self.origin = metadata['origin']
                
                if 'direction' in metadata:
                    self.direction = metadata['direction']
                
                if 'series_uid' in metadata:
                    self.series_uid = metadata['series_uid']
                    
                if 'study_uid' in metadata:
                    self.study_uid = metadata['study_uid']
                    
                if 'patient_id' in metadata:
                    self.patient_id = metadata['patient_id']
            
            # Tạo hình ảnh SimpleITK
            self._sitk_image = None  # Xóa hình ảnh cũ nếu có
            _ = self.get_sitk_image()  # Tạo hình ảnh mới
            
            logger.info(f"Đã tạo chuỗi DICOM từ dữ liệu hình ảnh {self.image_data.shape}")
            return True
            
        except Exception as e:
            logger.error(f"Lỗi khi tạo chuỗi DICOM từ dữ liệu hình ảnh: {e}")
            return False
    
    def resample(self, new_spacing: Tuple[float, float, float]) -> Optional['DicomSeries']:
        """
        Tái lấy mẫu chuỗi hình ảnh với spacing mới.
        
        Parameters
        ----------
        new_spacing : Tuple[float, float, float]
            Spacing mới (mm)
            
        Returns
        -------
        DicomSeries or None
            Chuỗi hình ảnh đã được tái lấy mẫu hoặc None nếu thất bại
        """
        if self.image_data is None or self._sitk_image is None:
            logger.error("Không có dữ liệu hình ảnh để tái lấy mẫu")
            return None
        
        try:
            # Tính tỷ lệ kích thước
            original_spacing = self.spacing
            original_size = self._sitk_image.GetSize()
            
            new_size = [
                int(round(original_size[0] * original_spacing[0] / new_spacing[0])),
                int(round(original_size[1] * original_spacing[1] / new_spacing[1])),
                int(round(original_size[2] * original_spacing[2] / new_spacing[2]))
            ]
            
            # Tạo bộ lọc resampling
            resampler = sitk.ResampleImageFilter()
            resampler.SetOutputSpacing(new_spacing)
            resampler.SetSize(new_size)
            resampler.SetOutputDirection(self._sitk_image.GetDirection())
            resampler.SetOutputOrigin(self._sitk_image.GetOrigin())
            resampler.SetTransform(sitk.Transform())
            resampler.SetDefaultPixelValue(0)
            resampler.SetInterpolator(sitk.sitkLinear)
            
            # Thực hiện resampling
            resampled_image = resampler.Execute(self._sitk_image)
            
            # Tạo chuỗi mới
            resampled_series = DicomSeries(
                series_uid=self.series_uid,
                study_uid=self.study_uid,
                patient_id=self.patient_id
            )
            
            # Lấy dữ liệu hình ảnh
            resampled_data = sitk.GetArrayFromImage(resampled_image)
            
            # Tạo metadata mới
            new_metadata = self.metadata.copy()
            new_metadata.update({
                'modality': self.modality,
                'series_description': f"{self.series_description} (Resampled)" if self.series_description else "Resampled",
                'spacing': new_spacing,
                'origin': self.origin,
                'direction': self.direction,
                'original_spacing': self.spacing
            })
            
            # Thiết lập dữ liệu cho chuỗi mới
            success = resampled_series.create_from_image(resampled_data, new_metadata)
            
            if not success:
                logger.error("Không thể tạo chuỗi mới từ dữ liệu đã được tái lấy mẫu")
                return None
                
            logger.info(f"Đã tái lấy mẫu chuỗi DICOM từ {self.spacing} thành {new_spacing}")
            return resampled_series
            
        except Exception as e:
            logger.error(f"Lỗi khi tái lấy mẫu chuỗi DICOM: {e}")
            return None
    
    def save_to_files(self, output_dir: str, base_filename: str = "image") -> List[str]:
        """
        Lưu chuỗi DICOM thành các file.
        
        Parameters
        ----------
        output_dir : str
            Thư mục đầu ra
        base_filename : str, optional
            Tên cơ sở cho file (mặc định: "image")
            
        Returns
        -------
        List[str]
            Danh sách đường dẫn đến các file đã lưu
        """
        if self.image_data is None:
            logger.error("Không có dữ liệu hình ảnh để lưu")
            return []
        
        try:
            # Tạo thư mục nếu chưa tồn tại
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            # Tạo writer
            writer = sitk.ImageFileWriter()
            writer.KeepOriginalImageUIDOn()
            
            # Lưu từng lát cắt
            output_files = []
            for i in range(self.num_slices):
                # Tạo tên file
                filename = f"{base_filename}_{i:04d}.dcm"
                filepath = os.path.join(output_dir, filename)
                
                # Trích xuất lát cắt
                slice_data = self.get_slice(i, "axial")
                if slice_data is None:
                    logger.warning(f"Không thể trích xuất lát cắt {i}")
                    continue
                
                # Chuyển thành ảnh SimpleITK
                slice_image = sitk.GetImageFromArray(slice_data)
                slice_image.SetSpacing((self.spacing[0], self.spacing[1]))
                
                # Lưu file
                writer.SetFileName(filepath)
                writer.Execute(slice_image)
                
                output_files.append(filepath)
            
            logger.info(f"Đã lưu {len(output_files)} lát cắt DICOM vào thư mục {output_dir}")
            return output_files
            
        except Exception as e:
            logger.error(f"Lỗi khi lưu chuỗi DICOM thành file: {e}")
            return []
    
    def __str__(self) -> str:
        """Chuỗi đại diện cho đối tượng DicomSeries."""
        return (f"DicomSeries(series_uid={self.series_uid}, "
                f"modality={self.modality}, "
                f"num_slices={self.num_slices}, "
                f"shape={self.image_data.shape if self.image_data is not None else None})")
    
    def __repr__(self) -> str:
        """Chuỗi đại diện chi tiết cho đối tượng DicomSeries."""
        return self.__str__() 