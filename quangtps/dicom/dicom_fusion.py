"""
Công cụ hỗ trợ fusion (chồng hình) đa phương thức DICOM.

Module này cung cấp các công cụ để thực hiện chồng hình giữa các phương thức
hình ảnh khác nhau như CT-MRI, CT-PET, MRI-PET, giúp cho việc phân tích
và hiển thị dữ liệu y tế toàn diện hơn.
"""

import os
import logging
import numpy as np
import pydicom
from typing import List, Dict, Any, Tuple, Optional, Union
import SimpleITK as sitk

from quangtps.core.exceptions import DicomError, FusionError
from quangtps.dicom.dicom_reader import DicomReader
from quangtps.dicom.dicom_converter import DicomConverter

logger = logging.getLogger(__name__)

class DicomFusion:
    """
    Lớp xử lý chồng hình (fusion) DICOM đa phương thức.
    
    Class này cung cấp các phương thức để đăng ký (registration) và chồng hình
    giữa các bộ dữ liệu DICOM từ các phương thức hình ảnh khác nhau, hỗ trợ
    các tính năng như CT-MRI fusion, CT-PET fusion và MRI-PET fusion.
    """
    
    def __init__(self):
        """Khởi tạo đối tượng DicomFusion."""
        self.primary_image = None
        self.secondary_image = None
        self.registration_transform = None
        self.primary_type = None
        self.secondary_type = None
        self.resampled_secondary = None
        self.fusion_result = None
        self.overlay_alpha = 0.5  # Độ trong suốt mặc định
    
    def load_primary_image(self, dicom_files: List[str]) -> None:
        """
        Tải ảnh chính (primary image) từ các file DICOM.
        
        Parameters
        ----------
        dicom_files : List[str]
            Danh sách các file DICOM
            
        Raises
        ------
        DicomError
            Nếu không thể tải ảnh
        """
        try:
            # Tải các file DICOM
            reader = DicomReader()
            self.primary_dataset = reader.read_files(dicom_files)
            
            # Chuyển đổi thành ảnh SimpleITK
            if hasattr(self.primary_dataset[0], 'Modality'):
                self.primary_type = self.primary_dataset[0].Modality
            else:
                self.primary_type = "UNKNOWN"
            
            self.primary_image = self._convert_dicom_to_sitk(self.primary_dataset)
            logger.info(f"Loaded primary image of type {self.primary_type}")
        except Exception as e:
            logger.error(f"Error loading primary image: {str(e)}")
            raise DicomError(f"Error loading primary image: {str(e)}")
    
    def load_secondary_image(self, dicom_files: List[str]) -> None:
        """
        Tải ảnh thứ cấp (secondary image) từ các file DICOM.
        
        Parameters
        ----------
        dicom_files : List[str]
            Danh sách các file DICOM
            
        Raises
        ------
        DicomError
            Nếu không thể tải ảnh
        """
        try:
            # Tải các file DICOM
            reader = DicomReader()
            self.secondary_dataset = reader.read_files(dicom_files)
            
            # Chuyển đổi thành ảnh SimpleITK
            if hasattr(self.secondary_dataset[0], 'Modality'):
                self.secondary_type = self.secondary_dataset[0].Modality
            else:
                self.secondary_type = "UNKNOWN"
            
            self.secondary_image = self._convert_dicom_to_sitk(self.secondary_dataset)
            logger.info(f"Loaded secondary image of type {self.secondary_type}")
        except Exception as e:
            logger.error(f"Error loading secondary image: {str(e)}")
            raise DicomError(f"Error loading secondary image: {str(e)}")
    
    def _convert_dicom_to_sitk(self, dicom_datasets: List[pydicom.dataset.FileDataset]) -> sitk.Image:
        """
        Chuyển đổi dataset DICOM thành đối tượng SimpleITK Image.
        
        Parameters
        ----------
        dicom_datasets : List[pydicom.dataset.FileDataset]
            Dataset DICOM
            
        Returns
        -------
        sitk.Image
            Đối tượng SimpleITK Image
            
        Raises
        ------
        DicomError
            Nếu không thể chuyển đổi
        """
        try:
            # Trích xuất dữ liệu pixel
            converter = DicomConverter()
            volume_array, spacing, origin, direction = converter.convert_dicom_to_volumetric_data(
                dicom_datasets
            )
            
            # Chuyển đổi thành SimpleITK Image
            image = sitk.GetImageFromArray(volume_array)
            image.SetSpacing(spacing)
            image.SetOrigin(origin)
            image.SetDirection(direction)
            
            return image
        except Exception as e:
            logger.error(f"Error converting DICOM to SimpleITK image: {str(e)}")
            raise DicomError(f"Error converting DICOM to SimpleITK image: {str(e)}")
    
    def register_images(self, method: str = 'rigid') -> None:
        """
        Thực hiện đăng ký (registration) giữa ảnh chính và ảnh thứ cấp.
        
        Parameters
        ----------
        method : str, optional
            Phương pháp đăng ký, có thể là 'rigid', 'affine', hoặc 'deformable'
            
        Raises
        ------
        FusionError
            Nếu không thể thực hiện đăng ký ảnh
        """
        if self.primary_image is None or self.secondary_image is None:
            raise FusionError("Primary and secondary images must be loaded first")
        
        try:
            # Lựa chọn phương pháp đăng ký
            if method == 'rigid':
                registration_method = sitk.ImageRegistrationMethod()
                registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
                registration_method.SetOptimizerAsRegularStepGradientDescent(
                    learningRate=1.0, minStep=0.001, numberOfIterations=200
                )
                registration_method.SetInitialTransform(sitk.CenteredTransformInitializer(
                    self.primary_image, self.secondary_image, sitk.Euler3DTransform(),
                    sitk.CenteredTransformInitializerFilter.GEOMETRY
                ))
                
            elif method == 'affine':
                registration_method = sitk.ImageRegistrationMethod()
                registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
                registration_method.SetOptimizerAsRegularStepGradientDescent(
                    learningRate=1.0, minStep=0.001, numberOfIterations=300
                )
                registration_method.SetInitialTransform(sitk.CenteredTransformInitializer(
                    self.primary_image, self.secondary_image, sitk.AffineTransform(3),
                    sitk.CenteredTransformInitializerFilter.GEOMETRY
                ))
                
            elif method == 'deformable':
                # Đầu tiên, thực hiện đăng ký affine
                affine_transform = sitk.CenteredTransformInitializer(
                    self.primary_image, self.secondary_image, sitk.AffineTransform(3),
                    sitk.CenteredTransformInitializerFilter.GEOMETRY
                )
                
                affine_registration = sitk.ImageRegistrationMethod()
                affine_registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
                affine_registration.SetOptimizerAsRegularStepGradientDescent(
                    learningRate=1.0, minStep=0.001, numberOfIterations=100
                )
                affine_registration.SetInitialTransform(affine_transform)
                
                affine_transform = affine_registration.Execute(
                    sitk.Cast(self.primary_image, sitk.sitkFloat32),
                    sitk.Cast(self.secondary_image, sitk.sitkFloat32)
                )
                
                # Sau đó, thực hiện đăng ký deformable
                displacement_field_transform = sitk.DisplacementFieldTransform(3)
                displacement_field_transform.SetSmoothingGaussianOnUpdate(
                    varianceForUpdateField=0.0, varianceForTotalField=1.5
                )
                
                registration_method = sitk.ImageRegistrationMethod()
                registration_method.SetMetricAsANTSNeighborhoodCorrelation(4)
                registration_method.SetOptimizerAsGradientDescent(
                    learningRate=1.0, numberOfIterations=50, convergenceWindowSize=10
                )
                registration_method.SetInitialTransform(displacement_field_transform, inPlace=True)
                registration_method.SetInterpolator(sitk.sitkLinear)
                registration_method.SetShrinkFactorsPerLevel([4, 2, 1])
                registration_method.SetSmoothingSigmasPerLevel([4, 2, 1])
                
                moving_resampled = sitk.Resample(
                    self.secondary_image, self.primary_image, affine_transform,
                    sitk.sitkLinear, 0.0, self.secondary_image.GetPixelID()
                )
                
                self.registration_transform = registration_method.Execute(
                    sitk.Cast(self.primary_image, sitk.sitkFloat32),
                    sitk.Cast(moving_resampled, sitk.sitkFloat32)
                )
                
                # Kết hợp 2 phép biến đổi
                composite_transform = sitk.CompositeTransform(3)
                composite_transform.AddTransform(self.registration_transform)
                composite_transform.AddTransform(affine_transform)
                
                self.registration_transform = composite_transform
                
                logger.info("Completed deformable registration")
                self._resample_secondary_image()
                return
            
            else:
                raise FusionError(f"Unknown registration method: {method}")
            
            # Thực hiện đăng ký
            self.registration_transform = registration_method.Execute(
                sitk.Cast(self.primary_image, sitk.sitkFloat32),
                sitk.Cast(self.secondary_image, sitk.sitkFloat32)
            )
            
            logger.info(f"Completed {method} registration")
            
            # Resample ảnh thứ cấp theo biến đổi đã tính
            self._resample_secondary_image()
            
        except Exception as e:
            logger.error(f"Error during image registration: {str(e)}")
            raise FusionError(f"Error during image registration: {str(e)}")
    
    def _resample_secondary_image(self) -> None:
        """
        Resample ảnh thứ cấp theo ảnh chính sử dụng biến đổi đã đăng ký.
        
        Raises
        ------
        FusionError
            Nếu không thể resample ảnh
        """
        try:
            if self.registration_transform is None:
                raise FusionError("Registration transform not available")
            
            self.resampled_secondary = sitk.Resample(
                self.secondary_image, self.primary_image, self.registration_transform,
                sitk.sitkLinear, 0.0, self.secondary_image.GetPixelID()
            )
            
            logger.info("Resampled secondary image to primary image space")
        except Exception as e:
            logger.error(f"Error resampling secondary image: {str(e)}")
            raise FusionError(f"Error resampling secondary image: {str(e)}")
    
    def create_fusion(self, alpha: float = None, fusion_method: str = 'overlay') -> np.ndarray:
        """
        Tạo ảnh fusion từ ảnh chính và ảnh thứ cấp đã được resample.
        
        Parameters
        ----------
        alpha : float, optional
            Độ trong suốt của ảnh thứ cấp, từ 0.0 đến 1.0
        fusion_method : str, optional
            Phương pháp fusion, có thể là 'overlay', 'colormap', hoặc 'checkerboard'
            
        Returns
        -------
        np.ndarray
            Ảnh fusion dạng numpy array
            
        Raises
        ------
        FusionError
            Nếu không thể tạo ảnh fusion
        """
        if self.primary_image is None or self.resampled_secondary is None:
            raise FusionError("Primary and resampled secondary images are required")
        
        try:
            # Lấy dữ liệu numpy từ SimpleITK Image
            primary_array = sitk.GetArrayFromImage(self.primary_image)
            secondary_array = sitk.GetArrayFromImage(self.resampled_secondary)
            
            # Chuẩn hóa dữ liệu pixel
            primary_normalized = self._normalize_array(primary_array)
            secondary_normalized = self._normalize_array(secondary_array)
            
            # Cập nhật alpha nếu được cung cấp
            if alpha is not None:
                self.overlay_alpha = max(0.0, min(1.0, alpha))
            
            # Lựa chọn phương pháp fusion
            if fusion_method == 'overlay':
                # Fusion dạng overlay
                fusion_array = (1.0 - self.overlay_alpha) * primary_normalized + self.overlay_alpha * secondary_normalized
            
            elif fusion_method == 'colormap':
                # Fusion với colormap (giả màu)
                # Tạo fusion RGB
                fusion_array = np.zeros((*primary_normalized.shape, 3), dtype=np.float32)
                
                # Ảnh chính hiển thị theo thang độ xám
                fusion_array[..., 0] = primary_normalized
                fusion_array[..., 1] = primary_normalized
                fusion_array[..., 2] = primary_normalized
                
                # Ảnh thứ cấp hiển thị theo màu đặc trưng (ví dụ: PET thường dùng hot colormap)
                if self.secondary_type == 'PT':
                    # Áp dụng hot colormap cho PET (đỏ-vàng)
                    mask = secondary_normalized > 0.05  # Lọc nhiễu
                    fusion_array[mask, 0] = secondary_normalized[mask]
                    fusion_array[mask, 1] = 0.7 * secondary_normalized[mask]
                    fusion_array[mask, 2] = 0.0
                elif self.secondary_type == 'MR':
                    # Áp dụng cool colormap cho MRI (xanh dương)
                    mask = secondary_normalized > 0.1  # Lọc nhiễu
                    fusion_array[mask, 0] = 0.0
                    fusion_array[mask, 1] = 0.5 * secondary_normalized[mask]
                    fusion_array[mask, 2] = secondary_normalized[mask]
                else:
                    # Mặc định: đỏ
                    mask = secondary_normalized > 0.1  # Lọc nhiễu
                    fusion_array[mask, 0] = secondary_normalized[mask]
                    fusion_array[mask, 1] = 0.0
                    fusion_array[mask, 2] = 0.0
                
                # Đảm bảo giá trị nằm trong khoảng [0, 1]
                fusion_array = np.clip(fusion_array, 0.0, 1.0)
            
            elif fusion_method == 'checkerboard':
                # Fusion dạng bàn cờ
                # Tạo mẫu bàn cờ
                pattern_size = 10  # Kích thước ô cờ
                x, y, z = np.indices(primary_array.shape)
                checkerboard = np.mod(x // pattern_size + y // pattern_size + z // pattern_size, 2)
                
                # Áp dụng mẫu bàn cờ
                fusion_array = np.zeros_like(primary_normalized)
                fusion_array[checkerboard == 0] = primary_normalized[checkerboard == 0]
                fusion_array[checkerboard == 1] = secondary_normalized[checkerboard == 1]
            
            else:
                raise FusionError(f"Unknown fusion method: {fusion_method}")
            
            self.fusion_result = fusion_array
            logger.info(f"Created fusion image using {fusion_method} method")
            
            return fusion_array
            
        except Exception as e:
            logger.error(f"Error creating fusion image: {str(e)}")
            raise FusionError(f"Error creating fusion image: {str(e)}")
    
    def _normalize_array(self, array: np.ndarray) -> np.ndarray:
        """
        Chuẩn hóa mảng numpy về khoảng [0, 1].
        
        Parameters
        ----------
        array : np.ndarray
            Mảng đầu vào
            
        Returns
        -------
        np.ndarray
            Mảng đã chuẩn hóa
        """
        min_val = np.min(array)
        max_val = np.max(array)
        
        if max_val == min_val:
            return np.zeros_like(array)
        
        return (array - min_val) / (max_val - min_val)
    
    def get_fusion_slice(self, axis: int = 0, slice_index: int = None) -> np.ndarray:
        """
        Lấy slice từ kết quả fusion.
        
        Parameters
        ----------
        axis : int, optional
            Trục để lấy slice (0=axial, 1=coronal, 2=sagittal)
        slice_index : int, optional
            Chỉ số của slice, nếu None thì lấy slice giữa
            
        Returns
        -------
        np.ndarray
            Slice từ kết quả fusion
            
        Raises
        ------
        FusionError
            Nếu không thể lấy slice
        """
        if self.fusion_result is None:
            raise FusionError("Fusion result not available")
        
        try:
            # Lấy kích thước của kết quả fusion
            shape = self.fusion_result.shape
            
            # Xác định chỉ số slice nếu không được cung cấp
            if slice_index is None:
                if axis == 0:
                    slice_index = shape[0] // 2
                elif axis == 1:
                    slice_index = shape[1] // 2
                elif axis == 2:
                    slice_index = shape[2] // 2
                else:
                    raise FusionError(f"Invalid axis: {axis}")
            
            # Trích xuất slice
            if axis == 0:
                if slice_index < 0 or slice_index >= shape[0]:
                    raise FusionError(f"Slice index out of range: 0-{shape[0]-1}")
                return self.fusion_result[slice_index, :, :]
            elif axis == 1:
                if slice_index < 0 or slice_index >= shape[1]:
                    raise FusionError(f"Slice index out of range: 0-{shape[1]-1}")
                return self.fusion_result[:, slice_index, :]
            elif axis == 2:
                if slice_index < 0 or slice_index >= shape[2]:
                    raise FusionError(f"Slice index out of range: 0-{shape[2]-1}")
                return self.fusion_result[:, :, slice_index]
            else:
                raise FusionError(f"Invalid axis: {axis}")
                
        except Exception as e:
            logger.error(f"Error getting fusion slice: {str(e)}")
            raise FusionError(f"Error getting fusion slice: {str(e)}")
    
    def save_fusion_image(self, output_path: str, axis: int = 0, slice_index: int = None) -> None:
        """
        Lưu slice từ kết quả fusion thành file hình ảnh.
        
        Parameters
        ----------
        output_path : str
            Đường dẫn file đầu ra
        axis : int, optional
            Trục để lấy slice (0=axial, 1=coronal, 2=sagittal)
        slice_index : int, optional
            Chỉ số của slice, nếu None thì lấy slice giữa
            
        Raises
        ------
        FusionError
            Nếu không thể lưu hình ảnh
        """
        try:
            from PIL import Image
            
            # Lấy slice từ kết quả fusion
            slice_data = self.get_fusion_slice(axis, slice_index)
            
            # Kiểm tra kích thước của mảng
            if slice_data.ndim == 2:
                # Dữ liệu grayscale
                img_data = (slice_data * 255).astype(np.uint8)
                img = Image.fromarray(img_data)
            elif slice_data.ndim == 3 and slice_data.shape[2] == 3:
                # Dữ liệu RGB
                img_data = (slice_data * 255).astype(np.uint8)
                img = Image.fromarray(img_data)
            else:
                raise FusionError(f"Unsupported slice data shape: {slice_data.shape}")
            
            # Lưu hình ảnh
            img.save(output_path)
            logger.info(f"Saved fusion image to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving fusion image: {str(e)}")
            raise FusionError(f"Error saving fusion image: {str(e)}")
