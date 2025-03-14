"""
Công cụ xử lý dữ liệu 4D-CT (Four-dimensional Computed Tomography).

Module này cung cấp các chức năng để xử lý dữ liệu 4D-CT trong xạ trị,
bao gồm việc phân loại theo pha hô hấp và quản lý chuyển động nội tạng.
"""

import os
import logging
import numpy as np
import pydicom
import SimpleITK as sitk
from typing import List, Dict, Any, Tuple, Optional, Union, Sequence

from quangtps.core.exceptions import DicomError
from quangtps.dicom.dicom_reader import DicomReader
from quangtps.dicom.dicom_converter import DicomConverter

logger = logging.getLogger(__name__)

class FourDCTProcessor:
    """
    Lớp xử lý dữ liệu 4D-CT.
    
    Class này cung cấp các phương thức để xử lý dữ liệu 4D-CT, bao gồm việc
    tổ chức các phase theo thời gian và phân tích chuyển động cơ quan.
    """
    
    def __init__(self):
        """Khởi tạo đối tượng FourDCTProcessor."""
        self.phase_datasets = {}  # Dict[phase_id, List[FileDataset]]
        self.phase_volumes = {}   # Dict[phase_id, np.ndarray]
        self.phase_spacing = {}   # Dict[phase_id, Tuple[float, float, float]]
        self.reference_phase = None
        self.displacement_fields = {} # Dict[phase_id, sitk.Image]
    
    def load_4dct_data(self, dicom_files: List[str]) -> List[str]:
        """
        Tải dữ liệu 4D-CT từ danh sách các file DICOM.
        
        Parameters
        ----------
        dicom_files : List[str]
            Danh sách các file DICOM 4D-CT
            
        Returns
        -------
        List[str]
            Danh sách các phase đã phát hiện
            
        Raises
        ------
        DicomError
            Nếu không thể tải dữ liệu 4D-CT
        """
        try:
            reader = DicomReader()
            datasets = reader.read_files(dicom_files)
            
            # Kiểm tra xem có phải CT không
            if not datasets or not hasattr(datasets[0], 'Modality') or datasets[0].Modality != 'CT':
                raise DicomError("Files are not CT data")
            
            # Nhóm các slice theo phase
            phase_groups = self._group_slices_by_phase(datasets)
            
            # Tải từng phase
            for phase_id, phase_datasets in phase_groups.items():
                # Sắp xếp các slice theo vị trí
                sorted_datasets = self._sort_slices_by_position(phase_datasets)
                
                # Lưu datasets
                self.phase_datasets[phase_id] = sorted_datasets
                
                # Chuyển đổi thành dữ liệu khối
                converter = DicomConverter()
                volume, spacing, origin, direction = \
                    converter.convert_dicom_to_volumetric_data(sorted_datasets)
                
                # Lưu thông tin
                self.phase_volumes[phase_id] = volume
                self.phase_spacing[phase_id] = spacing
                
                # Chuyển đổi thành đối tượng SimpleITK
                sitk_image = sitk.GetImageFromArray(volume)
                sitk_image.SetSpacing(spacing)
                sitk_image.SetOrigin(origin)
                sitk_image.SetDirection(direction)
            
            # Đặt phase tham chiếu là phase đầu tiên nếu có
            if self.phase_datasets:
                self.reference_phase = list(self.phase_datasets.keys())[0]
            
            logger.info(f"Loaded 4D-CT data: {len(self.phase_datasets)} phases")
            
            return list(self.phase_datasets.keys())
            
        except Exception as e:
            logger.error(f"Error loading 4D-CT data: {str(e)}")
            raise DicomError(f"Error loading 4D-CT data: {str(e)}")
    
    def _group_slices_by_phase(self, datasets: List[pydicom.dataset.FileDataset]) -> Dict[str, List[pydicom.dataset.FileDataset]]:
        """
        Nhóm các slice theo phase dựa trên các tag DICOM.
        
        Parameters
        ----------
        datasets : List[pydicom.dataset.FileDataset]
            Danh sách các dataset DICOM
            
        Returns
        -------
        Dict[str, List[pydicom.dataset.FileDataset]]
            Từ điển ánh xạ từ phase_id đến danh sách các dataset
        """
        phase_groups = {}
        
        for ds in datasets:
            # Tìm thông tin phase
            phase_id = None
            
            # Thứ tự ưu tiên kiểm tra các tag
            if hasattr(ds, 'RespiratoryPhaseIdentifier'):
                phase_id = str(ds.RespiratoryPhaseIdentifier)
            elif hasattr(ds, 'CardiacPhaseIdentifier'):
                phase_id = str(ds.CardiacPhaseIdentifier)
            elif hasattr(ds, 'TemporalPositionIdentifier'):
                phase_id = f"phase_{ds.TemporalPositionIdentifier}"
            elif hasattr(ds, 'AcquisitionTime'):
                # Sử dụng thời gian thu nhận nếu không có thông tin phase
                phase_id = ds.AcquisitionTime
            else:
                # Sử dụng số series nếu không tìm thấy thông tin khác
                phase_id = f"series_{ds.SeriesNumber}"
            
            # Thêm vào nhóm phase tương ứng
            if phase_id not in phase_groups:
                phase_groups[phase_id] = []
            
            phase_groups[phase_id].append(ds)
        
        return phase_groups
    
    def _sort_slices_by_position(self, datasets: List[pydicom.dataset.FileDataset]) -> List[pydicom.dataset.FileDataset]:
        """
        Sắp xếp các slice theo vị trí.
        
        Parameters
        ----------
        datasets : List[pydicom.dataset.FileDataset]
            Danh sách các dataset DICOM
            
        Returns
        -------
        List[pydicom.dataset.FileDataset]
            Danh sách đã sắp xếp
        """
        def get_slice_position(ds):
            if hasattr(ds, 'ImagePositionPatient'):
                # Lấy tọa độ z (thông thường là trục dọc của bệnh nhân)
                return float(ds.ImagePositionPatient[2])
            return 0.0
        
        # Sắp xếp theo vị trí tăng dần
        return sorted(datasets, key=get_slice_position)
    
    def set_reference_phase(self, phase_id: str) -> bool:
        """
        Đặt phase tham chiếu.
        
        Parameters
        ----------
        phase_id : str
            ID của phase tham chiếu
            
        Returns
        -------
        bool
            True nếu thành công, False nếu không tìm thấy phase
        """
        if phase_id in self.phase_datasets:
            self.reference_phase = phase_id
            logger.info(f"Set reference phase to {phase_id}")
            return True
        
        logger.warning(f"Phase {phase_id} not found")
        return False
    
    def calculate_displacement_field(self, target_phase: str = None) -> sitk.Image:
        """
        Tính toán trường biến dạng (displacement field) giữa phase tham chiếu và phase mục tiêu.
        
        Parameters
        ----------
        target_phase : str, optional
            ID của phase mục tiêu, nếu None thì tính toán cho tất cả các phase
            
        Returns
        -------
        sitk.Image
            Trường biến dạng từ phase tham chiếu đến phase mục tiêu
            
        Raises
        ------
        DicomError
            Nếu không thể tính toán trường biến dạng
        """
        if self.reference_phase is None:
            raise DicomError("Reference phase not set")
        
        if self.reference_phase not in self.phase_volumes:
            raise DicomError(f"Reference phase {self.reference_phase} not loaded")
        
        try:
            # Lấy hình ảnh tham chiếu
            ref_volume = self.phase_volumes[self.reference_phase]
            ref_spacing = self.phase_spacing[self.reference_phase]
            
            # Chuyển đổi thành đối tượng SimpleITK
            ref_image = sitk.GetImageFromArray(ref_volume)
            ref_image.SetSpacing(ref_spacing)
            
            if target_phase is not None:
                # Tính toán cho một phase cụ thể
                if target_phase not in self.phase_volumes:
                    raise DicomError(f"Target phase {target_phase} not loaded")
                
                # Lấy hình ảnh mục tiêu
                target_volume = self.phase_volumes[target_phase]
                target_spacing = self.phase_spacing[target_phase]
                
                # Chuyển đổi thành đối tượng SimpleITK
                target_image = sitk.GetImageFromArray(target_volume)
                target_image.SetSpacing(target_spacing)
                
                # Tính toán trường biến dạng
                displacement_field = self._register_images(ref_image, target_image)
                
                self.displacement_fields[target_phase] = displacement_field
                return displacement_field
                
            else:
                # Tính toán cho tất cả các phase
                for phase_id, phase_volume in self.phase_volumes.items():
                    if phase_id == self.reference_phase:
                        continue
                    
                    # Lấy hình ảnh mục tiêu
                    target_spacing = self.phase_spacing[phase_id]
                    
                    # Chuyển đổi thành đối tượng SimpleITK
                    target_image = sitk.GetImageFromArray(phase_volume)
                    target_image.SetSpacing(target_spacing)
                    
                    # Tính toán trường biến dạng
                    displacement_field = self._register_images(ref_image, target_image)
                    
                    self.displacement_fields[phase_id] = displacement_field
                
                # Trả về danh sách các trường biến dạng
                return self.displacement_fields
                
        except Exception as e:
            logger.error(f"Error calculating displacement field: {str(e)}")
            raise DicomError(f"Error calculating displacement field: {str(e)}")
    
    def _register_images(self, reference_image: sitk.Image, target_image: sitk.Image) -> sitk.Image:
        """
        Đăng ký hai ảnh và tính toán trường biến dạng.
        
        Parameters
        ----------
        reference_image : sitk.Image
            Ảnh tham chiếu
        target_image : sitk.Image
            Ảnh mục tiêu
            
        Returns
        -------
        sitk.Image
            Trường biến dạng từ ảnh tham chiếu đến ảnh mục tiêu
        """
        # Tiền xử lý ảnh
        reference_float = sitk.Cast(reference_image, sitk.sitkFloat32)
        target_float = sitk.Cast(target_image, sitk.sitkFloat32)
        
        # Chuẩn hóa cường độ
        reference_float = sitk.Normalize(reference_float)
        target_float = sitk.Normalize(target_float)
        
        # Tạo phép biến đổi giữa hai ảnh sử dụng demons registration
        registration_method = sitk.ImageRegistrationMethod()
        
        # Sử dụng đo lường tương quan
        registration_method.SetMetricAsMeanSquares()
        
        # Sử dụng bộ tối ưu hóa gradient descent
        registration_method.SetOptimizerAsGradientDescent(
            learningRate=1.0, numberOfIterations=50
        )
        
        # Sử dụng BSpline transform với cỡ lưới 8x8x8
        transform_domain_mesh_size = [8] * reference_float.GetDimension()
        initial_transform = sitk.BSplineTransformInitializer(
            reference_float, transform_domain_mesh_size
        )
        
        registration_method.SetInitialTransform(initial_transform)
        
        # Thực hiện đăng ký
        final_transform = registration_method.Execute(reference_float, target_float)
        
        # Tính toán trường biến dạng
        displacement_field = sitk.TransformToDisplacementField(
            final_transform, sitk.sitkVectorFloat64, reference_float.GetSize(),
            reference_float.GetOrigin(), reference_float.GetSpacing(),
            reference_float.GetDirection()
        )
        
        return displacement_field
    
    def create_mid_position_scan(self) -> Tuple[np.ndarray, Tuple[float, float, float]]:
        """
        Tạo một CT ở vị trí giữa (mid-position) từ dữ liệu 4D-CT.
        
        Returns
        -------
        Tuple[np.ndarray, Tuple[float, float, float]]
            CT mid-position và spacing
            
        Raises
        ------
        DicomError
            Nếu không thể tạo CT mid-position
        """
        if not self.phase_volumes:
            raise DicomError("No 4D-CT data loaded")
        
        try:
            # Tính trung bình của tất cả các phase
            all_volumes = list(self.phase_volumes.values())
            mid_position = np.mean(all_volumes, axis=0)
            
            # Lấy spacing từ phase đầu tiên
            first_phase = list(self.phase_spacing.keys())[0]
            spacing = self.phase_spacing[first_phase]
            
            logger.info(f"Created mid-position scan from {len(all_volumes)} phases")
            
            return mid_position, spacing
            
        except Exception as e:
            logger.error(f"Error creating mid-position scan: {str(e)}")
            raise DicomError(f"Error creating mid-position scan: {str(e)}")
    
    def calculate_motion_amplitude(self) -> Dict[str, float]:
        """
        Tính toán biên độ chuyển động giữa các phase 4D-CT.
        
        Returns
        -------
        Dict[str, float]
            Biên độ chuyển động cho mỗi phase (đơn vị: mm)
            
        Raises
        ------
        DicomError
            Nếu không thể tính toán biên độ chuyển động
        """
        if not self.displacement_fields:
            raise DicomError("Displacement fields not calculated. Call calculate_displacement_field() first.")
        
        try:
            motion_amplitudes = {}
            
            for phase_id, displacement_field in self.displacement_fields.items():
                # Chuyển đổi trường biến dạng thành mảng
                displacement_array = sitk.GetArrayFromImage(displacement_field)
                
                # Tính độ lớn của vector dịch chuyển tại mỗi voxel
                magnitude = np.sqrt(np.sum(displacement_array**2, axis=-1))
                
                # Tính giá trị lớn nhất
                max_displacement = np.max(magnitude)
                
                # Lưu kết quả
                motion_amplitudes[phase_id] = float(max_displacement)
            
            return motion_amplitudes
            
        except Exception as e:
            logger.error(f"Error calculating motion amplitude: {str(e)}")
            raise DicomError(f"Error calculating motion amplitude: {str(e)}")
