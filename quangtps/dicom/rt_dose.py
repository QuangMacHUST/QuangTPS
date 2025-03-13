"""
Xử lý file DICOM RT Dose.
"""

import numpy as np
import logging
import pydicom

from quangtps.core.exceptions import DicomError, IOError

logger = logging.getLogger(__name__)

class RTDose:
    """Lớp xử lý dữ liệu DICOM RT Dose"""
    
    def __init__(self, rt_dose_dataset=None):
        """
        Khởi tạo RTDose.
        
        Parameters:
            rt_dose_dataset (pydicom.dataset.FileDataset, optional): Dataset RTDOSE
        """
        self.dataset = rt_dose_dataset
        self.dose_grid = None
        self.dose_summation_type = None
        self.dose_units = None
        self.dose_scaling = None
        self.dose_type = None
        self.grid_frame_offset_vector = None
        self.pixel_spacing = None
        self.image_position_patient = None
        
        if rt_dose_dataset is not None:
            self._load_dose_data()
    
    def _load_dose_data(self):
        """Tải dữ liệu liều từ dataset"""
        if not self.dataset or not hasattr(self.dataset, 'Modality') or self.dataset.Modality != 'RTDOSE':
            logger.warning("Dataset is not a valid RT Dose")
            return
        
        try:
            # Lấy thông tin cơ bản về liều
            if hasattr(self.dataset, 'DoseSummationType'):
                self.dose_summation_type = self.dataset.DoseSummationType
            
            if hasattr(self.dataset, 'DoseUnits'):
                self.dose_units = self.dataset.DoseUnits
            
            if hasattr(self.dataset, 'DoseType'):
                self.dose_type = self.dataset.DoseType
            
            if hasattr(self.dataset, 'DoseGridScaling'):
                self.dose_scaling = float(self.dataset.DoseGridScaling)
            
            # Lấy thông tin không gian liều
            if hasattr(self.dataset, 'PixelSpacing'):
                self.pixel_spacing = [float(x) for x in self.dataset.PixelSpacing]
            
            if hasattr(self.dataset, 'ImagePositionPatient'):
                self.image_position_patient = [float(x) for x in self.dataset.ImagePositionPatient]
            
            if hasattr(self.dataset, 'GridFrameOffsetVector'):
                self.grid_frame_offset_vector = [float(x) for x in self.dataset.GridFrameOffsetVector]
            
            # Lấy dữ liệu liều
            if hasattr(self.dataset, 'pixel_array'):
                # Nhân với scaling factor để chuyển thành giá trị liều thực
                if self.dose_scaling is not None:
                    self.dose_grid = self.dataset.pixel_array * self.dose_scaling
                else:
                    self.dose_grid = self.dataset.pixel_array
            
        except Exception as e:
            logger.error(f"Error loading RT Dose data: {str(e)}")
            raise DicomError(f"Error loading RT Dose data: {str(e)}")
    
    def get_dose_grid(self):
        """
        Lấy lưới liều 3D.
        
        Returns:
            numpy.ndarray: Mảng 3D chứa giá trị liều
        """
        return self.dose_grid
    
    def get_dose_at_point(self, x, y, z):
        """
        Lấy giá trị liều tại điểm (x, y, z) trong không gian bệnh nhân.
        
        Parameters:
            x (float): Tọa độ x trong không gian bệnh nhân (mm)
            y (float): Tọa độ y trong không gian bệnh nhân (mm)
            z (float): Tọa độ z trong không gian bệnh nhân (mm)
        
        Returns:
            float: Giá trị liều tại điểm, None nếu điểm nằm ngoài lưới
        
        Raises:
            ValueError: Nếu thông tin không gian không đầy đủ
        """
        if (self.dose_grid is None or self.pixel_spacing is None or 
            self.image_position_patient is None or self.grid_frame_offset_vector is None):
            raise ValueError("Spatial information is not complete")
        
        # Chuyển từ tọa độ bệnh nhân sang chỉ số voxel
        dx, dy = self.pixel_spacing
        ipx, ipy, ipz = self.image_position_patient
        
        # Tính toán chỉ số voxel
        col = int(round((x - ipx) / dx))
        row = int(round((y - ipy) / dy))
        
        # Tìm slice phù hợp
        z_positions = [ipz + offset for offset in self.grid_frame_offset_vector]
        slice_index = np.argmin(np.abs(np.array(z_positions) - z))
        
        # Kiểm tra giới hạn
        if (0 <= row < self.dose_grid.shape[1] and 
            0 <= col < self.dose_grid.shape[2] and 
            0 <= slice_index < self.dose_grid.shape[0]):
            return self.dose_grid[slice_index, row, col]
        else:
            return None
    
    def get_dose_resolution(self):
        """
        Lấy độ phân giải của lưới liều.
        
        Returns:
            tuple: (dx, dy, dz) in mm
        """
        if self.pixel_spacing is None or self.grid_frame_offset_vector is None:
            return None
        
        dx, dy = self.pixel_spacing
        
        # Tính dz trung bình
        if len(self.grid_frame_offset_vector) > 1:
            dz = np.mean(np.diff(self.grid_frame_offset_vector))
        else:
            dz = 1.0  # Giá trị mặc định
        
        return dx, dy, dz
    
    def get_dose_statistics(self, mask=None):
        """
        Tính toán thống kê về liều.
        
        Parameters:
            mask (numpy.ndarray, optional): Mask 3D để giới hạn vùng tính toán
        
        Returns:
            dict: Các giá trị thống kê (min, max, mean, median, etc.)
        """
        if self.dose_grid is None:
            return None
        
        if mask is not None and mask.shape != self.dose_grid.shape:
            logger.warning("Mask and dose grid shapes do not match")
            return None
        
        # Lấy giá trị liều trong mask
        if mask is not None:
            dose_values = self.dose_grid[mask > 0]
        else:
            dose_values = self.dose_grid.flatten()
        
        # Tính toán thống kê
        return {
            'min': np.min(dose_values),
            'max': np.max(dose_values),
            'mean': np.mean(dose_values),
            'median': np.median(dose_values),
            'std': np.std(dose_values)
        }
    
    def get_dose_in_structure(self, structure_mask):
        """
        Lấy phân bố liều trong một cấu trúc.
        
        Parameters:
            structure_mask (numpy.ndarray): Mask 3D của cấu trúc
        
        Returns:
            numpy.ndarray: Mảng 1D chứa giá trị liều trong cấu trúc
        """
        if self.dose_grid is None or structure_mask is None:
            return None
        
        if structure_mask.shape != self.dose_grid.shape:
            logger.warning("Structure mask and dose grid shapes do not match")
            return None
        
        return self.dose_grid[structure_mask > 0]
    
    @classmethod
    def from_file(cls, file_path):
        """
        Tạo đối tượng RTDose từ file DICOM.
        
        Parameters:
            file_path (str): Đường dẫn đến file RTDOSE
        
        Returns:
            RTDose: Đối tượng RTDose
        
        Raises:
            IOError: Nếu file không tồn tại
            DicomError: Nếu file không phải là RTDOSE hợp lệ
        """
        try:
            from quangtps.dicom.dicom_reader import DicomReader
            dataset = DicomReader.read_file(file_path)
            
            # Kiểm tra loại file
            if hasattr(dataset, 'Modality') and dataset.Modality != 'RTDOSE':
                raise DicomError(f"File is not an RT Dose (Modality: {dataset.Modality})")
            
            return cls(dataset)
        except Exception as e:
            logger.error(f"Error creating RTDose from file: {str(e)}")
            raise
