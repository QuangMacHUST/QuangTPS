"""
Đọc dữ liệu từ file DICOM.
"""

import os
import logging
import numpy as np
import pydicom
from pydicom.errors import InvalidDicomError

from quangtps.core.exceptions import DicomError, IOError

logger = logging.getLogger(__name__)

class DicomReader:
    """Lớp xử lý việc đọc dữ liệu từ file DICOM"""
    
    def __init__(self):
        """Khởi tạo DicomReader"""
        pass
    
    @staticmethod
    def read_file(file_path):
        """
        Đọc file DICOM và trả về đối tượng dataset.
        
        Parameters:
            file_path (str): Đường dẫn đến file DICOM
        
        Returns:
            pydicom.dataset.FileDataset: Dataset DICOM
        
        Raises:
            IOError: Nếu file không tồn tại
            DicomError: Nếu file không phải là DICOM hợp lệ
        """
        if not os.path.exists(file_path):
            raise IOError(f"File not found", file_path=file_path)
        
        try:
            return pydicom.dcmread(file_path)
        except InvalidDicomError:
            raise DicomError(f"Invalid DICOM file: {file_path}")
        except Exception as e:
            raise DicomError(f"Error reading DICOM file: {str(e)}")
    
    @staticmethod
    def read_directory(directory):
        """
        Đọc tất cả các file DICOM trong thư mục.
        
        Parameters:
            directory (str): Đường dẫn đến thư mục
        
        Returns:
            list: Danh sách các dataset DICOM
        
        Raises:
            IOError: Nếu thư mục không tồn tại
        """
        if not os.path.exists(directory):
            raise IOError(f"Directory not found", file_path=directory)
        
        dicom_files = []
        
        for root, _, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    dicom_dataset = DicomReader.read_file(file_path)
                    dicom_files.append(dicom_dataset)
                except (DicomError, IOError) as e:
                    logger.warning(f"Skipping file {file_path}: {str(e)}")
                except Exception as e:
                    logger.error(f"Unexpected error reading {file_path}: {str(e)}")
        
        return dicom_files
    
    @staticmethod
    def get_dicom_type(dicom_dataset):
        """
        Xác định loại DICOM (CT, MR, RTSTRUCT, RTDOSE, RTPLAN,...).
        
        Parameters:
            dicom_dataset (pydicom.dataset.FileDataset): Dataset DICOM
        
        Returns:
            str: Loại DICOM (CT, MR, RTSTRUCT, RTDOSE, RTPLAN,...)
        """
        try:
            modality = dicom_dataset.Modality
            return modality
        except Exception:
            return "UNKNOWN"
    
    @staticmethod
    def sort_ct_slices(ct_datasets):
        """
        Sắp xếp các slices CT theo vị trí z.
        
        Parameters:
            ct_datasets (list): Danh sách các dataset DICOM CT
        
        Returns:
            list: Danh sách các dataset đã sắp xếp
        """
        # Sắp xếp theo vị trí z của slice
        return sorted(ct_datasets, key=lambda x: float(x.ImagePositionPatient[2]))
    
    @staticmethod
    def extract_ct_volume(ct_datasets):
        """
        Tạo thành volume 3D từ các slices CT.
        
        Parameters:
            ct_datasets (list): Danh sách các dataset DICOM CT
        
        Returns:
            tuple: (volume, voxel_size)
                volume (numpy.ndarray): Volume 3D
                voxel_size (tuple): Kích thước voxel (dx, dy, dz)
        """
        # Sắp xếp slices
        sorted_slices = DicomReader.sort_ct_slices(ct_datasets)
        
        if len(sorted_slices) == 0:
            raise DicomError("No CT slices found")
        
        # Lấy kích thước voxel
        first_slice = sorted_slices[0]
        pixel_spacing = first_slice.PixelSpacing
        dx, dy = float(pixel_spacing[0]), float(pixel_spacing[1])
        
        # Tính khoảng cách slice (dz)
        if len(sorted_slices) > 1:
            z_pos = [float(s.ImagePositionPatient[2]) for s in sorted_slices]
            dz = sum([abs(z_pos[i] - z_pos[i-1]) for i in range(1, len(z_pos))]) / (len(z_pos) - 1)
        else:
            dz = 1.0  # Giá trị mặc định
        
        # Tạo volume
        pixel_arrays = [s.pixel_array for s in sorted_slices]
        volume = np.stack(pixel_arrays, axis=0)
        
        # Chuyển HU
        if hasattr(first_slice, 'RescaleIntercept') and hasattr(first_slice, 'RescaleSlope'):
            intercept = first_slice.RescaleIntercept
            slope = first_slice.RescaleSlope
            volume = volume * slope + intercept
        
        return volume, (dx, dy, dz)
