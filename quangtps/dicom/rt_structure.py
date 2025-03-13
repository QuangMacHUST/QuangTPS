"""
Xử lý file DICOM RT Structure.
"""

import numpy as np
import logging
import pydicom
from pydicom.sequence import Sequence

from quangtps.core.exceptions import DicomError
from quangtps.core.constants import Constants

logger = logging.getLogger(__name__)

class RTStructure:
    """Lớp xử lý dữ liệu DICOM RT Structure"""
    
    def __init__(self, rt_struct_dataset=None):
        """
        Khởi tạo RTStructure.
        
        Parameters:
            rt_struct_dataset (pydicom.dataset.FileDataset, optional): Dataset RTSTRUCT
        """
        self.dataset = rt_struct_dataset
        self.structures = {}
        
        if rt_struct_dataset is not None:
            self._load_structures()
    
    def _load_structures(self):
        """Tải thông tin cấu trúc từ dataset"""
        if not self.dataset or not hasattr(self.dataset, 'ROIContourSequence'):
            logger.warning("No ROIContourSequence found in RT Structure dataset")
            return
        
        # Lấy thông tin ROI từ StructureSetROISequence
        roi_info = {}
        if hasattr(self.dataset, 'StructureSetROISequence'):
            for roi in self.dataset.StructureSetROISequence:
                roi_info[roi.ROINumber] = {
                    'name': roi.ROIName,
                    'roi_number': roi.ROINumber
                }
        
        # Lấy thông tin contour từ ROIContourSequence
        for roi_contour in self.dataset.ROIContourSequence:
            roi_number = roi_contour.ReferencedROINumber
            
            if roi_number not in roi_info:
                logger.warning(f"ROI #{roi_number} referenced in ROIContourSequence not found in StructureSetROISequence")
                continue
            
            name = roi_info[roi_number]['name']
            contour_data = []
            
            # Lấy màu
            color = None
            if hasattr(roi_contour, 'ROIDisplayColor'):
                color = [int(c) for c in roi_contour.ROIDisplayColor]
            
            # Lấy dữ liệu contour
            if hasattr(roi_contour, 'ContourSequence'):
                for contour in roi_contour.ContourSequence:
                    if hasattr(contour, 'ContourData'):
                        points = np.array(contour.ContourData).reshape(-1, 3)
                        contour_data.append(points)
            
            # Lưu cấu trúc
            self.structures[name] = {
                'roi_number': roi_number,
                'color': color,
                'contours': contour_data
            }
    
    def get_structure_names(self):
        """
        Lấy danh sách tên các cấu trúc.
        
        Returns:
            list: Danh sách tên cấu trúc
        """
        return list(self.structures.keys())
    
    def get_structure(self, name):
        """
        Lấy dữ liệu cấu trúc theo tên.
        
        Parameters:
            name (str): Tên cấu trúc
        
        Returns:
            dict: Dữ liệu cấu trúc
        
        Raises:
            KeyError: Nếu cấu trúc không tồn tại
        """
        if name not in self.structures:
            raise KeyError(f"Structure '{name}' not found")
        
        return self.structures[name]
    
    def get_contour_points(self, name):
        """
        Lấy tọa độ các điểm contour của cấu trúc.
        
        Parameters:
            name (str): Tên cấu trúc
        
        Returns:
            list: Danh sách các mảng numpy chứa tọa độ các điểm
        
        Raises:
            KeyError: Nếu cấu trúc không tồn tại
        """
        structure = self.get_structure(name)
        return structure['contours']
    
    def add_structure(self, name, contours, color=None, roi_number=None):
        """
        Thêm cấu trúc mới vào dataset.
        
        Parameters:
            name (str): Tên cấu trúc
            contours (list): Danh sách các mảng numpy chứa tọa độ các điểm
            color (list, optional): Màu RGB của cấu trúc
            roi_number (int, optional): Số ROI, tự động nếu để None
        
        Returns:
            bool: True nếu thành công, False nếu thất bại
        """
        if name in self.structures:
            logger.warning(f"Structure '{name}' already exists, overwriting")
        
        # Tự động tạo ROI number nếu cần
        if roi_number is None:
            existing_numbers = [s['roi_number'] for s in self.structures.values()]
            roi_number = 1
            while roi_number in existing_numbers:
                roi_number += 1
        
        # Tự động chọn màu từ constants nếu cần
        if color is None:
            if name.upper() in Constants.STRUCTURE_COLORS:
                color = Constants.STRUCTURE_COLORS[name.upper()]
            else:
                # Màu mặc định là đỏ
                color = [255, 0, 0]
        
        # Lưu cấu trúc mới
        self.structures[name] = {
            'roi_number': roi_number,
            'color': color,
            'contours': contours
        }
        
        # Cập nhật dataset nếu có
        if self.dataset is not None:
            self._update_dataset()
        
        return True
    
    def remove_structure(self, name):
        """
        Xóa cấu trúc khỏi dataset.
        
        Parameters:
            name (str): Tên cấu trúc
        
        Returns:
            bool: True nếu thành công, False nếu thất bại
        
        Raises:
            KeyError: Nếu cấu trúc không tồn tại
        """
        if name not in self.structures:
            raise KeyError(f"Structure '{name}' not found")
        
        del self.structures[name]
        
        # Cập nhật dataset nếu có
        if self.dataset is not None:
            self._update_dataset()
        
        return True
    
    def _update_dataset(self):
        """Cập nhật dataset DICOM với dữ liệu cấu trúc mới"""
        if self.dataset is None:
            logger.warning("No DICOM dataset to update")
            return
        
        # Tạo ROI và Contour sequences
        struct_set_roi_sequence = []
        roi_contour_sequence = []
        
        for name, structure in self.structures.items():
            roi_number = structure['roi_number']
            
            # Tạo ROI trong StructureSetROISequence
            roi_dataset = pydicom.Dataset()
            roi_dataset.ROINumber = roi_number
            roi_dataset.ROIName = name
            roi_dataset.ROIGenerationAlgorithm = 'MANUAL'
            struct_set_roi_sequence.append(roi_dataset)
            
            # Tạo ROI trong ROIContourSequence
            roi_contour_dataset = pydicom.Dataset()
            roi_contour_dataset.ReferencedROINumber = roi_number
            
            # Thêm màu
            if structure['color']:
                roi_contour_dataset.ROIDisplayColor = structure['color']
            
            # Thêm contour data
            contour_sequence = []
            for i, contour in enumerate(structure['contours']):
                contour_dataset = pydicom.Dataset()
                contour_dataset.ContourGeometricType = 'CLOSED_PLANAR'
                contour_dataset.NumberOfContourPoints = len(contour)
                contour_dataset.ContourData = contour.flatten().tolist()
                contour_sequence.append(contour_dataset)
            
            roi_contour_dataset.ContourSequence = Sequence(contour_sequence)
            roi_contour_sequence.append(roi_contour_dataset)
        
        # Cập nhật dataset
        self.dataset.StructureSetROISequence = Sequence(struct_set_roi_sequence)
        self.dataset.ROIContourSequence = Sequence(roi_contour_sequence)
    
    def create_mask_volume(self, structure_name, reference_volume, voxel_size):
        """
        Tạo mask 3D từ contours của cấu trúc.
        
        Parameters:
            structure_name (str): Tên cấu trúc
            reference_volume (numpy.ndarray): Volume tham chiếu (CT, MRI,...)
            voxel_size (tuple): Kích thước voxel (dx, dy, dz)
        
        Returns:
            numpy.ndarray: Mask binary 3D cùng kích thước với reference_volume
        
        Raises:
            KeyError: Nếu cấu trúc không tồn tại
        """
        if structure_name not in self.structures:
            raise KeyError(f"Structure '{structure_name}' not found")
        
        # Tạo mask rỗng
        mask = np.zeros_like(reference_volume, dtype=np.bool_)
        
        # Lấy contours
        contours = self.get_contour_points(structure_name)
        
        # TODO: Implement algorithm to convert contours to 3D mask
        # This requires knowing the reference frame and transformation matrix
        # between world coordinates and voxel coordinates
        
        logger.warning("create_mask_volume is not fully implemented yet")
        return mask
    
    @classmethod
    def from_file(cls, file_path):
        """
        Tạo đối tượng RTStructure từ file DICOM.
        
        Parameters:
            file_path (str): Đường dẫn đến file RTSTRUCT
        
        Returns:
            RTStructure: Đối tượng RTStructure
        
        Raises:
            IOError: Nếu file không tồn tại
            DicomError: Nếu file không phải là RTSTRUCT hợp lệ
        """
        try:
            from quangtps.dicom.dicom_reader import DicomReader
            dataset = DicomReader.read_file(file_path)
            
            # Kiểm tra loại file
            if hasattr(dataset, 'Modality') and dataset.Modality != 'RTSTRUCT':
                raise DicomError(f"File is not an RT Structure Set (Modality: {dataset.Modality})")
            
            return cls(dataset)
        except Exception as e:
            logger.error(f"Error creating RTStructure from file: {str(e)}")
            raise
