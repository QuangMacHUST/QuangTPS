"""
Chuyển đổi giữa các định dạng DICOM và các định dạng khác.
"""

import os
import logging
import numpy as np
import pydicom
import SimpleITK as sitk
from PIL import Image
import matplotlib.pyplot as plt
import io

from quangtps.core.exceptions import DicomError, IOError

logger = logging.getLogger(__name__)

class DicomConverter:
    """Lớp chuyển đổi giữa các định dạng DICOM và các định dạng khác"""
    
    @staticmethod
    def dicom_to_numpy(dicom_dataset):
        """
        Chuyển đổi dữ liệu hình ảnh DICOM thành mảng numpy.
        
        Parameters:
            dicom_dataset (pydicom.dataset.FileDataset): Dataset DICOM
        
        Returns:
            tuple: (array, metadata)
                array (numpy.ndarray): Mảng dữ liệu
                metadata (dict): Metadata của hình ảnh
        
        Raises:
            DicomError: Nếu dataset không có dữ liệu hình ảnh
        """
        try:
            if not hasattr(dicom_dataset, 'pixel_array'):
                raise DicomError("DICOM dataset does not contain image data")
            
            # Lấy dữ liệu hình ảnh
            image_data = dicom_dataset.pixel_array
            
            # Áp dụng slope và intercept (nếu có) để chuyển thành giá trị thực
            if hasattr(dicom_dataset, 'RescaleSlope') and hasattr(dicom_dataset, 'RescaleIntercept'):
                slope = float(dicom_dataset.RescaleSlope)
                intercept = float(dicom_dataset.RescaleIntercept)
                image_data = image_data * slope + intercept
            
            # Thu thập metadata
            metadata = {}
            
            # Thông tin không gian
            if hasattr(dicom_dataset, 'PixelSpacing'):
                metadata['pixel_spacing'] = [float(x) for x in dicom_dataset.PixelSpacing]
            
            if hasattr(dicom_dataset, 'ImagePositionPatient'):
                metadata['image_position'] = [float(x) for x in dicom_dataset.ImagePositionPatient]
            
            if hasattr(dicom_dataset, 'ImageOrientationPatient'):
                metadata['image_orientation'] = [float(x) for x in dicom_dataset.ImageOrientationPatient]
            
            # Thông tin window
            if hasattr(dicom_dataset, 'WindowCenter') and hasattr(dicom_dataset, 'WindowWidth'):
                metadata['window_center'] = float(dicom_dataset.WindowCenter)
                metadata['window_width'] = float(dicom_dataset.WindowWidth)
            
            # Thông tin bệnh nhân
            if hasattr(dicom_dataset, 'PatientName'):
                metadata['patient_name'] = str(dicom_dataset.PatientName)
            
            if hasattr(dicom_dataset, 'PatientID'):
                metadata['patient_id'] = dicom_dataset.PatientID
            
            # Thông tin nghiên cứu
            if hasattr(dicom_dataset, 'StudyInstanceUID'):
                metadata['study_instance_uid'] = dicom_dataset.StudyInstanceUID
            
            if hasattr(dicom_dataset, 'SeriesInstanceUID'):
                metadata['series_instance_uid'] = dicom_dataset.SeriesInstanceUID
            
            if hasattr(dicom_dataset, 'SOPInstanceUID'):
                metadata['sop_instance_uid'] = dicom_dataset.SOPInstanceUID
            
            return image_data, metadata
        except Exception as e:
            logger.error(f"Error converting DICOM to numpy: {str(e)}")
            raise DicomError(f"Error converting DICOM to numpy: {str(e)}")
    
    @staticmethod
    def numpy_to_image(array, format='PNG', window_center=None, window_width=None):
        """
        Chuyển đổi mảng numpy thành hình ảnh.
        
        Parameters:
            array (numpy.ndarray): Mảng dữ liệu
            format (str): Định dạng hình ảnh đầu ra ('PNG', 'JPG', etc.)
            window_center (float, optional): Trung tâm cửa sổ hiển thị
            window_width (float, optional): Độ rộng của cửa sổ hiển thị
        
        Returns:
            bytes: Dữ liệu hình ảnh dạng binary
        """
        # Chuẩn hóa mảng dữ liệu về [0, 255]
        if window_center is not None and window_width is not None:
            min_value = window_center - window_width / 2
            max_value = window_center + window_width / 2
            array = np.clip(array, min_value, max_value)
        
        # Chuẩn hóa về [0, 1]
        min_val = array.min()
        max_val = array.max()
        
        if max_val != min_val:
            normalized = (array - min_val) / (max_val - min_val)
        else:
            normalized = np.zeros_like(array)
        
        # Chuyển thành [0, 255] và uint8
        image_data = (normalized * 255).astype(np.uint8)
        
        # Tạo hình ảnh PIL
        image = Image.fromarray(image_data)
        
        # Chuyển thành bytes
        buffer = io.BytesIO()
        image.save(buffer, format=format)
        
        return buffer.getvalue()
    
    @staticmethod
    def dicom_to_sitk(dicom_dataset):
        """
        Chuyển đổi dataset DICOM thành ảnh SimpleITK.
        
        Parameters:
            dicom_dataset (pydicom.dataset.FileDataset): Dataset DICOM
        
        Returns:
            sitk.Image: Ảnh SimpleITK
        
        Raises:
            DicomError: Nếu dataset không có dữ liệu hình ảnh
        """
        try:
            if not hasattr(dicom_dataset, 'pixel_array'):
                raise DicomError("DICOM dataset does not contain image data")
            
            # Lấy dữ liệu hình ảnh
            image_data = dicom_dataset.pixel_array
            
            # Áp dụng slope và intercept (nếu có)
            if hasattr(dicom_dataset, 'RescaleSlope') and hasattr(dicom_dataset, 'RescaleIntercept'):
                slope = float(dicom_dataset.RescaleSlope)
                intercept = float(dicom_dataset.RescaleIntercept)
                image_data = image_data * slope + intercept
            
            # Chuyển đổi dữ liệu thành SimpleITK Image
            sitk_image = sitk.GetImageFromArray(image_data)
            
            # Thiết lập thông tin metadata
            if hasattr(dicom_dataset, 'PixelSpacing'):
                spacing = [float(x) for x in dicom_dataset.PixelSpacing]
                if hasattr(dicom_dataset, 'SliceThickness'):
                    spacing.append(float(dicom_dataset.SliceThickness))
                else:
                    spacing.append(1.0)  # Giá trị mặc định
                sitk_image.SetSpacing(spacing)
            
            if hasattr(dicom_dataset, 'ImagePositionPatient'):
                origin = [float(x) for x in dicom_dataset.ImagePositionPatient]
                sitk_image.SetOrigin(origin)
            
            if hasattr(dicom_dataset, 'ImageOrientationPatient'):
                orientation = [float(x) for x in dicom_dataset.ImageOrientationPatient]
                direction = [
                    orientation[0], orientation[3], 0,
                    orientation[1], orientation[4], 0,
                    orientation[2], orientation[5], 1
                ]
                sitk_image.SetDirection(direction)
            
            return sitk_image
        except Exception as e:
            logger.error(f"Error converting DICOM to SimpleITK: {str(e)}")
            raise DicomError(f"Error converting DICOM to SimpleITK: {str(e)}")
    
    @staticmethod
    def dicom_series_to_volume(dicom_dir, series_id=None):
        """
        Đọc một series DICOM và chuyển đổi thành volume 3D.
        
        Parameters:
            dicom_dir (str): Đường dẫn đến thư mục chứa các file DICOM
            series_id (str, optional): Series Instance UID, nếu None thì lấy series đầu tiên
        
        Returns:
            sitk.Image: Volume 3D dạng SimpleITK Image
        
        Raises:
            DicomError: Nếu không thể đọc series
        """
        try:
            # Tạo đối tượng ImageSeriesReader
            reader = sitk.ImageSeriesReader()
            
            # Lấy danh sách tất cả series IDs
            series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(dicom_dir)
            
            if not series_ids:
                raise DicomError(f"No DICOM series found in {dicom_dir}")
            
            # Nếu không chỉ định series_id, lấy series đầu tiên
            if series_id is None:
                series_id = series_ids[0]
            elif series_id not in series_ids:
                raise DicomError(f"Series {series_id} not found in {dicom_dir}")
            
            # Lấy danh sách file cho series
            dicom_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(dicom_dir, series_id)
            
            # Thiết lập danh sách file và đọc
            reader.SetFileNames(dicom_names)
            volume = reader.Execute()
            
            return volume
        except Exception as e:
            logger.error(f"Error reading DICOM series: {str(e)}")
            raise DicomError(f"Error reading DICOM series: {str(e)}")
    
    @staticmethod
    def volume_to_dicom_series(volume, output_dir, patient_info=None, study_info=None):
        """
        Chuyển đổi volume 3D thành series DICOM.
        
        Parameters:
            volume (sitk.Image): Volume 3D dạng SimpleITK Image
            output_dir (str): Thư mục đầu ra
            patient_info (dict, optional): Thông tin bệnh nhân
            study_info (dict, optional): Thông tin nghiên cứu
        
        Returns:
            bool: True nếu thành công
        
        Raises:
            DicomError: Nếu không thể tạo series
        """
        try:
            # Đảm bảo thư mục đầu ra tồn tại
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Lấy kích thước và số lượng slice
            size = volume.GetSize()
            num_slices = size[2]
            
            # Tạo các tham số DICOM mặc định
            if patient_info is None:
                patient_info = {
                    'name': 'Anonymous',
                    'id': 'Unknown',
                    'birth_date': '',
                    'sex': ''
                }
            
            if study_info is None:
                study_info = {
                    'study_id': '1',
                    'study_uid': pydicom.uid.generate_uid(),
                    'series_uid': pydicom.uid.generate_uid(),
                    'modality': 'CT'
                }
            
            # Tạo image writer
            writer = sitk.ImageFileWriter()
            
            # Tạo các file DICOM cho từng slice
            for i in range(num_slices):
                # Lấy slice
                slice_image = volume[:, :, i]
                
                # Tạo file tạm thời
                temp_file = os.path.join(output_dir, f"slice_{i:04d}.dcm")
                
                # Ghi slice ra file tạm thời
                writer.SetFileName(temp_file)
                writer.Execute(slice_image)
                
                # Đọc file tạm thời và sửa metadata
                ds = pydicom.dcmread(temp_file)
                
                # Thêm thông tin bệnh nhân
                ds.PatientName = patient_info['name']
                ds.PatientID = patient_info['id']
                ds.PatientBirthDate = patient_info['birth_date']
                ds.PatientSex = patient_info['sex']
                
                # Thêm thông tin nghiên cứu
                ds.StudyID = study_info['study_id']
                ds.StudyInstanceUID = study_info['study_uid']
                ds.SeriesInstanceUID = study_info['series_uid']
                ds.Modality = study_info['modality']
                
                # Thêm thông tin slice
                ds.InstanceNumber = i + 1
                ds.SOPInstanceUID = pydicom.uid.generate_uid()
                
                # Ghi file DICOM cuối cùng
                ds.save_as(temp_file)
            
            return True
        except Exception as e:
            logger.error(f"Error creating DICOM series: {str(e)}")
            raise DicomError(f"Error creating DICOM series: {str(e)}")