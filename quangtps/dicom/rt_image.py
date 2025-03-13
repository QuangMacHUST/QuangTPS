"""
Xử lý file DICOM RT Image.
"""

import logging
import numpy as np
import pydicom

from quangtps.core.exceptions import DicomError, IOError

logger = logging.getLogger(__name__)

class RTImage:
    """Lớp xử lý dữ liệu DICOM RT Image"""
    
    def __init__(self, rt_image_dataset=None):
        """
        Khởi tạo RTImage.
        
        Parameters:
            rt_image_dataset (pydicom.dataset.FileDataset, optional): Dataset RTIMAGE
        """
        self.dataset = rt_image_dataset
        self.image = None
        self.pixel_spacing = None
        self.image_position_patient = None
        self.image_orientation_patient = None
        self.gantry_angle = None
        self.beam_limiting_device_angle = None
        self.patient_support_angle = None
        
        if rt_image_dataset is not None:
            self._load_image_data()
    
    def _load_image_data(self):
        """Tải dữ liệu hình ảnh từ dataset"""
        if not self.dataset or not hasattr(self.dataset, 'Modality') or self.dataset.Modality != 'RTIMAGE':
            logger.warning("Dataset is not a valid RT Image")
            return
        
        try:
            # Lấy dữ liệu hình ảnh
            if hasattr(self.dataset, 'pixel_array'):
                self.image = self.dataset.pixel_array
            
            # Lấy thông tin không gian
            if hasattr(self.dataset, 'PixelSpacing'):
                self.pixel_spacing = [float(x) for x in self.dataset.PixelSpacing]
            
            if hasattr(self.dataset, 'ImagePositionPatient'):
                self.image_position_patient = [float(x) for x in self.dataset.ImagePositionPatient]
            
            if hasattr(self.dataset, 'ImageOrientationPatient'):
                self.image_orientation_patient = [float(x) for x in self.dataset.ImageOrientationPatient]
            
            # Lấy thông tin góc
            if hasattr(self.dataset, 'GantryAngle'):
                self.gantry_angle = float(self.dataset.GantryAngle)
            
            if hasattr(self.dataset, 'BeamLimitingDeviceAngle'):
                self.beam_limiting_device_angle = float(self.dataset.BeamLimitingDeviceAngle)
            
            if hasattr(self.dataset, 'PatientSupportAngle'):
                self.patient_support_angle = float(self.dataset.PatientSupportAngle)
            
        except Exception as e:
            logger.error(f"Error loading RT Image data: {str(e)}")
            raise DicomError(f"Error loading RT Image data: {str(e)}")
    
    def get_image(self):
        """
        Lấy hình ảnh.
        
        Returns:
            numpy.ndarray: Mảng chứa dữ liệu hình ảnh
        """
        return self.image
    
    def get_angles(self):
        """
        Lấy các góc.
        
        Returns:
            dict: Các góc gantry, collimator, và couch
        """
        return {
            'gantry': self.gantry_angle,
            'collimator': self.beam_limiting_device_angle,
            'couch': self.patient_support_angle
        }
    
    def get_pixel_data(self, window_center=None, window_width=None):
        """
        Lấy dữ liệu pixel đã được áp dụng windowing.
        
        Parameters:
            window_center (float, optional): Trung tâm cửa sổ hiển thị
            window_width (float, optional): Độ rộng của cửa sổ hiển thị
        
        Returns:
            numpy.ndarray: Mảng chứa dữ liệu hình ảnh đã window
        """
        if self.image is None:
            return None
        
        if window_center is None or window_width is None:
            # Sử dụng giá trị mặc định từ DICOM
            if hasattr(self.dataset, 'WindowCenter') and hasattr(self.dataset, 'WindowWidth'):
                window_center = float(self.dataset.WindowCenter)
                window_width = float(self.dataset.WindowWidth)
            else:
                # Tính toán window dựa trên dữ liệu
                min_val = self.image.min()
                max_val = self.image.max()
                window_width = max_val - min_val
                window_center = min_val + window_width / 2
        
        # Áp dụng window
        min_value = window_center - window_width / 2
        max_value = window_center + window_width / 2
        windowed_image = np.clip(self.image, min_value, max_value)
        
        # Chuẩn hóa về [0, 1]
        windowed_image = (windowed_image - min_value) / (max_value - min_value)
        
        return windowed_image
    
    @classmethod
    def from_file(cls, file_path):
        """
        Tạo đối tượng RTImage từ file DICOM.
        
        Parameters:
            file_path (str): Đường dẫn đến file RTIMAGE
        
        Returns:
            RTImage: Đối tượng RTImage
        
        Raises:
            IOError: Nếu file không tồn tại
            DicomError: Nếu file không phải là RTIMAGE hợp lệ
        """
        try:
            from quangtps.dicom.dicom_reader import DicomReader
            dataset = DicomReader.read_file(file_path)
            
            # Kiểm tra loại file
            if hasattr(dataset, 'Modality') and dataset.Modality != 'RTIMAGE':
                raise DicomError(f"File is not an RT Image (Modality: {dataset.Modality})")
            
            return cls(dataset)
        except Exception as e:
            logger.error(f"Error creating RTImage from file: {str(e)}")
            raise
