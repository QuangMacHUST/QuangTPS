#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module cung cấp các chức năng làm việc với trường vector chuyển dịch (displacement field)
để theo dõi sự thay đổi giải phẫu giữa các hình ảnh.
"""

import os
import numpy as np
import logging
import SimpleITK as sitk
from typing import List, Dict, Tuple, Optional, Union, Any
from dataclasses import dataclass

from quangtps.core.types import Image, Structure, Dose
from quangtps.core.exceptions import DeformationError

logger = logging.getLogger(__name__)

class DisplacementField:
    """
    Lớp biểu diễn trường vector chuyển dịch giữa hai hình ảnh
    """
    
    def __init__(self, field_array: np.ndarray = None, reference_image: Image = None):
        """
        Khởi tạo trường vector chuyển dịch
        
        Parameters
        ----------
        field_array : np.ndarray, optional
            Mảng numpy 4D biểu diễn trường vector chuyển dịch (dim 0 = z, dim 1 = y, dim 2 = x, dim 3 = vector components)
        reference_image : Image, optional
            Hình ảnh tham chiếu để lấy thông tin không gian (spacing, origin, direction)
        """
        self.field_array = field_array
        self.reference_image = reference_image
        self.sitk_field = None
        
        if field_array is not None and reference_image is not None:
            self._initialize_sitk_field()
    
    def _initialize_sitk_field(self):
        """
        Khởi tạo đối tượng trường chuyển dịch SimpleITK từ mảng numpy
        """
        # Chuyển đổi mảng numpy thành đối tượng SimpleITK
        # Lưu ý: SimpleITK vector field có trật tự [x,y,z] trong khi numpy thường là [z,y,x]
        # Cần chuyển đổi trật tự các thành phần vector
        
        # Tạo một mảng mới với trật tự [x,y,z] phù hợp với SimpleITK
        field_sitk_order = np.zeros_like(self.field_array)
        
        # Đảm bảo rằng mảng field_array có đúng kích thước
        if self.field_array.shape[3] == 3:  # 3D vector field
            field_sitk_order[..., 0] = self.field_array[..., 2]  # x component
            field_sitk_order[..., 1] = self.field_array[..., 1]  # y component
            field_sitk_order[..., 2] = self.field_array[..., 0]  # z component
        elif self.field_array.shape[3] == 2:  # 2D vector field
            field_sitk_order[..., 0] = self.field_array[..., 1]  # x component
            field_sitk_order[..., 1] = self.field_array[..., 0]  # y component
        
        # Tạo đối tượng hình ảnh SimpleITK
        self.sitk_field = sitk.GetImageFromArray(field_sitk_order, isVector=True)
        
        # Thiết lập thông tin không gian
        if self.reference_image is not None:
            self.sitk_field.SetSpacing(self.reference_image.spacing)
            self.sitk_field.SetOrigin(self.reference_image.origin)
            self.sitk_field.SetDirection(self.reference_image.direction.flatten())
    
    @classmethod
    def from_sitk_field(cls, sitk_field: sitk.Image, reference_image: Image = None) -> 'DisplacementField':
        """
        Tạo đối tượng DisplacementField từ trường chuyển dịch SimpleITK
        
        Parameters
        ----------
        sitk_field : sitk.Image
            Đối tượng SimpleITK biểu diễn trường chuyển dịch
        reference_image : Image, optional
            Hình ảnh tham chiếu (nếu không có sẽ tạo thông tin không gian từ sitk_field)
            
        Returns
        -------
        DisplacementField
            Đối tượng DisplacementField mới
        """
        # Chuyển đổi từ sitk sang numpy array
        field_array = sitk.GetArrayFromImage(sitk_field)
        
        # Chuyển đổi trật tự từ SimpleITK [x,y,z] sang numpy [z,y,x]
        field_numpy_order = np.zeros_like(field_array)
        
        if field_array.shape[3] == 3:  # 3D vector field
            field_numpy_order[..., 0] = field_array[..., 2]  # z component
            field_numpy_order[..., 1] = field_array[..., 1]  # y component
            field_numpy_order[..., 2] = field_array[..., 0]  # x component
        elif field_array.shape[3] == 2:  # 2D vector field
            field_numpy_order[..., 0] = field_array[..., 1]  # y component
            field_numpy_order[..., 1] = field_array[..., 0]  # x component
        
        # Tạo đối tượng tham chiếu nếu chưa có
        if reference_image is None:
            from ...core.types import Image
            reference_image = Image(
                patient_id="unknown",
                modality="unknown",
                pixel_array=np.zeros(field_array.shape[:3]),
                spacing=sitk_field.GetSpacing(),
                origin=sitk_field.GetOrigin(),
                direction=np.array(sitk_field.GetDirection()).reshape(3, 3)
            )
        
        # Tạo đối tượng DisplacementField
        obj = cls(field_numpy_order, reference_image)
        obj.sitk_field = sitk_field  # Lưu trữ đối tượng SimpleITK gốc
        
        return obj
    
    @classmethod
    def from_registration_result(cls, registration_result, reference_image: Image) -> 'DisplacementField':
        """
        Tạo đối tượng DisplacementField từ kết quả đăng ký hình ảnh
        
        Parameters
        ----------
        registration_result : RegistrationResult
            Kết quả của quá trình đăng ký hình ảnh
        reference_image : Image
            Hình ảnh tham chiếu để tạo trường chuyển dịch
            
        Returns
        -------
        DisplacementField
            Đối tượng DisplacementField mới
        """
        # Tạo đối tượng hình ảnh SimpleITK từ reference_image
        reference_sitk = sitk.GetImageFromArray(reference_image.pixel_array)
        reference_sitk.SetSpacing(reference_image.spacing)
        reference_sitk.SetOrigin(reference_image.origin)
        reference_sitk.SetDirection(reference_image.direction.flatten())
        
        # Tạo trường chuyển dịch từ phép biến đổi
        displacement_field = sitk.TransformToDisplacementField(
            registration_result.transform,
            sitk.sitkVectorFloat64,
            reference_sitk.GetSize(),
            reference_sitk.GetOrigin(),
            reference_sitk.GetSpacing(),
            reference_sitk.GetDirection()
        )
        
        return cls.from_sitk_field(displacement_field, reference_image)
    
    def apply_to_image(self, image: Image) -> Image:
        """
        Áp dụng trường chuyển dịch cho hình ảnh
        
        Parameters
        ----------
        image : Image
            Hình ảnh cần biến đổi
            
        Returns
        -------
        Image
            Hình ảnh đã biến đổi
        """
        if self.sitk_field is None:
            self._initialize_sitk_field()
        
        # Chuyển đổi hình ảnh sang đối tượng SimpleITK
        sitk_image = sitk.GetImageFromArray(image.pixel_array)
        sitk_image.SetSpacing(image.spacing)
        sitk_image.SetOrigin(image.origin)
        sitk_image.SetDirection(image.direction.flatten())
        
        # Tạo đối tượng biến đổi từ trường chuyển dịch
        transform = sitk.DisplacementFieldTransform(self.sitk_field)
        
        # Áp dụng phép biến đổi
        warped_image = sitk.Resample(
            sitk_image, 
            self.reference_image.pixel_array.shape[::-1],  # Kích thước [x,y,z]
            transform, 
            sitk.sitkLinear, 
            0.0, 
            sitk_image.GetPixelID()
        )
        
        # Chuyển đổi trở lại đối tượng Image
        warped_array = sitk.GetArrayFromImage(warped_image)
        
        # Tạo đối tượng Image mới
        transformed_image = Image(
            patient_id=image.patient_id,
            modality=image.modality,
            study_id=image.study_id,
            series_id=image.series_id,
            pixel_array=warped_array,
            spacing=image.spacing,
            origin=image.origin,
            direction=image.direction,
            description=f"Warped from {image.id}"
        )
        
        return transformed_image
    
    def apply_to_dose(self, dose: Dose) -> Dose:
        """
        Áp dụng trường chuyển dịch cho liều
        
        Parameters
        ----------
        dose : Dose
            Đối tượng liều cần biến đổi
            
        Returns
        -------
        Dose
            Đối tượng liều đã biến đổi
        """
        if self.sitk_field is None:
            self._initialize_sitk_field()
        
        # Chuyển đổi mảng liều thành đối tượng SimpleITK
        sitk_dose = sitk.GetImageFromArray(dose.dose_matrix)
        sitk_dose.SetSpacing(dose.spacing)
        sitk_dose.SetOrigin(dose.origin)
        sitk_dose.SetDirection(dose.direction.flatten())
        
        # Tạo đối tượng biến đổi từ trường chuyển dịch
        transform = sitk.DisplacementFieldTransform(self.sitk_field)
        
        # Áp dụng phép biến đổi với nội suy tuyến tính và tổng hợp liều
        warped_dose = sitk.Resample(
            sitk_dose, 
            self.reference_image.pixel_array.shape[::-1],  # Kích thước [x,y,z]
            transform, 
            sitk.sitkLinear, 
            0.0, 
            sitk_dose.GetPixelID()
        )
        
        # Chuyển đổi trở lại mảng numpy
        warped_array = sitk.GetArrayFromImage(warped_dose)
        
        # Tạo đối tượng Dose mới
        transformed_dose = Dose(
            patient_id=dose.patient_id,
            plan_id=dose.plan_id,
            dose_matrix=warped_array,
            spacing=dose.spacing,
            origin=dose.origin,
            direction=dose.direction,
            dose_grid_scaling=dose.dose_grid_scaling,
            dose_units=dose.dose_units,
            dose_type=dose.dose_type,
            dose_summation_type=dose.dose_summation_type,
            description=f"Warped from {dose.id}"
        )
        
        return transformed_dose
    
    def apply_to_structure(self, structure: Structure) -> Structure:
        """
        Áp dụng trường chuyển dịch cho cấu trúc
        
        Parameters
        ----------
        structure : Structure
            Cấu trúc cần biến đổi
            
        Returns
        -------
        Structure
            Cấu trúc đã biến đổi
        """
        if self.sitk_field is None:
            self._initialize_sitk_field()
        
        # Tạo đối tượng biến đổi từ trường chuyển dịch
        transform = sitk.DisplacementFieldTransform(self.sitk_field)
        
        # Tạo một cấu trúc mới
        transformed_structure = Structure(
            patient_id=structure.patient_id,
            name=structure.name,
            type=structure.type,
            color=structure.color,
            description=f"Warped from {structure.id}"
        )
        
        # Áp dụng phép biến đổi cho mỗi contour
        for contour in structure.contours:
            transformed_contour = []
            
            # Áp dụng phép biến đổi cho mỗi điểm trong contour
            for point in contour:
                # Chuyển đổi từ [z,y,x] sang [x,y,z] cho SimpleITK
                sitk_point = [point[2], point[1], point[0]]
                
                # Áp dụng phép biến đổi
                transformed_point = transform.TransformPoint(sitk_point)
                
                # Chuyển đổi trở lại [z,y,x]
                transformed_point = [transformed_point[2], transformed_point[1], transformed_point[0]]
                transformed_contour.append(transformed_point)
            
            transformed_structure.contours.append(transformed_contour)
        
        return transformed_structure
    
    def save(self, filename: str):
        """
        Lưu trường chuyển dịch vào tệp tin
        
        Parameters
        ----------
        filename : str
            Đường dẫn tệp tin để lưu trường chuyển dịch
        """
        if self.sitk_field is None:
            self._initialize_sitk_field()
        
        sitk.WriteImage(self.sitk_field, filename)
    
    @classmethod
    def load(cls, filename: str, reference_image: Image = None) -> 'DisplacementField':
        """
        Tải trường chuyển dịch từ tệp tin
        
        Parameters
        ----------
        filename : str
            Đường dẫn tệp tin chứa trường chuyển dịch
        reference_image : Image, optional
            Hình ảnh tham chiếu
            
        Returns
        -------
        DisplacementField
            Đối tượng DisplacementField đã tải
        """
        try:
            sitk_field = sitk.ReadImage(filename)
            return cls.from_sitk_field(sitk_field, reference_image)
        except Exception as e:
            logger.error(f"Lỗi khi tải trường chuyển dịch từ {filename}: {str(e)}")
            raise DeformationError(f"Không thể tải trường chuyển dịch: {str(e)}")
    
    def compose(self, other: 'DisplacementField') -> 'DisplacementField':
        """
        Kết hợp trường chuyển dịch này với trường chuyển dịch khác
        
        Parameters
        ----------
        other : DisplacementField
            Trường chuyển dịch khác để kết hợp
            
        Returns
        -------
        DisplacementField
            Trường chuyển dịch mới kết hợp hai trường gốc
        """
        if self.sitk_field is None:
            self._initialize_sitk_field()
        
        if other.sitk_field is None:
            other._initialize_sitk_field()
        
        # Kiểm tra tính tương thích
        if (self.sitk_field.GetSize() != other.sitk_field.GetSize() or
            self.sitk_field.GetSpacing() != other.sitk_field.GetSpacing() or
            self.sitk_field.GetOrigin() != other.sitk_field.GetOrigin()):
            raise DeformationError("Không thể kết hợp hai trường chuyển dịch không tương thích")
        
        # Tạo hai phép biến đổi từ trường chuyển dịch
        transform1 = sitk.DisplacementFieldTransform(self.sitk_field)
        transform2 = sitk.DisplacementFieldTransform(other.sitk_field)
        
        # Tạo phép biến đổi kết hợp
        composed_transform = sitk.CompositeTransform(self.sitk_field.GetDimension())
        composed_transform.AddTransform(transform1)
        composed_transform.AddTransform(transform2)
        
        # Tạo trường chuyển dịch mới từ phép biến đổi kết hợp
        composed_field = sitk.TransformToDisplacementField(
            composed_transform,
            sitk.sitkVectorFloat64,
            self.sitk_field.GetSize(),
            self.sitk_field.GetOrigin(),
            self.sitk_field.GetSpacing(),
            self.sitk_field.GetDirection()
        )
        
        return self.from_sitk_field(composed_field, self.reference_image)
    
    def invert(self) -> 'DisplacementField':
        """
        Nghịch đảo trường chuyển dịch
        
        Returns
        -------
        DisplacementField
            Trường chuyển dịch nghịch đảo
        """
        if self.sitk_field is None:
            self._initialize_sitk_field()
        
        # Chuyển đổi trường chuyển dịch thành phép biến đổi
        transform = sitk.DisplacementFieldTransform(self.sitk_field)
        
        try:
            # Tính toán nghịch đảo xấp xỉ của phép biến đổi
            inverse_transform = transform.GetInverse()
            
            # Tạo trường chuyển dịch từ phép biến đổi nghịch đảo
            inverse_field = sitk.TransformToDisplacementField(
                inverse_transform,
                sitk.sitkVectorFloat64,
                self.sitk_field.GetSize(),
                self.sitk_field.GetOrigin(),
                self.sitk_field.GetSpacing(),
                self.sitk_field.GetDirection()
            )
            
            return self.from_sitk_field(inverse_field, self.reference_image)
        except Exception as e:
            logger.error(f"Không thể tính nghịch đảo của trường chuyển dịch: {str(e)}")
            raise DeformationError(f"Không thể tính nghịch đảo của trường chuyển dịch: {str(e)}")
    
    def smooth(self, sigma: float = 1.0) -> 'DisplacementField':
        """
        Làm mịn trường chuyển dịch bằng bộ lọc Gaussian
        
        Parameters
        ----------
        sigma : float, optional
            Độ lệch chuẩn của bộ lọc Gaussian, by default 1.0
            
        Returns
        -------
        DisplacementField
            Trường chuyển dịch đã được làm mịn
        """
        if self.sitk_field is None:
            self._initialize_sitk_field()
        
        # Áp dụng bộ lọc làm mịn Gaussian cho trường chuyển dịch
        smoothed_field = sitk.SmoothingRecursiveGaussian(self.sitk_field, sigma)
        
        return self.from_sitk_field(smoothed_field, self.reference_image)
    
    def get_jacobian_determinant(self) -> np.ndarray:
        """
        Tính định thức Jacobian của trường chuyển dịch
        
        Định thức Jacobian cho biết mức độ giãn nở hoặc co lại cục bộ
        của các tế mô khi áp dụng trường chuyển dịch.
        
        Returns
        -------
        np.ndarray
            Mảng chứa giá trị định thức Jacobian tại mỗi vị trí trong trường chuyển dịch
        """
        if self.sitk_field is None:
            self._initialize_sitk_field()
        
        # Tạo phép biến đổi từ trường chuyển dịch
        transform = sitk.DisplacementFieldTransform(self.sitk_field)
        
        # Tính toán định thức Jacobian
        jacobian_filter = sitk.DisplacementFieldJacobianDeterminantFilter()
        jacobian_image = jacobian_filter.Execute(self.sitk_field)
        
        # Chuyển đổi thành mảng numpy
        jacobian_array = sitk.GetArrayFromImage(jacobian_image)
        
        return jacobian_array
    
    def analyze_deformation(self) -> Dict[str, Any]:
        """
        Phân tích trường chuyển dịch để đánh giá các thay đổi giải phẫu
        
        Returns
        -------
        Dict[str, Any]
            Từ điển chứa các thống kê về trường chuyển dịch
        """
        if self.sitk_field is None:
            self._initialize_sitk_field()
        
        # Tính định thức Jacobian
        jacobian_array = self.get_jacobian_determinant()
        
        # Tính các thống kê về trường chuyển dịch
        field_mag = np.linalg.norm(self.field_array, axis=3)
        
        # Phân tích các khu vực co/giãn
        expansion_regions = jacobian_array > 1.1  # Giãn nở > 10%
        contraction_regions = jacobian_array < 0.9  # Co lại > 10%
        folding_regions = jacobian_array < 0  # Gấp cuộn (không hợp lý về mặt vật lý)
        
        return {
            "max_displacement": np.max(field_mag),
            "mean_displacement": np.mean(field_mag),
            "std_displacement": np.std(field_mag),
            "min_jacobian": np.min(jacobian_array),
            "max_jacobian": np.max(jacobian_array),
            "mean_jacobian": np.mean(jacobian_array),
            "expansion_percentage": np.sum(expansion_regions) / expansion_regions.size * 100,
            "contraction_percentage": np.sum(contraction_regions) / contraction_regions.size * 100,
            "folding_percentage": np.sum(folding_regions) / folding_regions.size * 100,
        }
    
    def visualize(self, slice_index: int = None, downsample: int = 4, scale: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Tạo dữ liệu trực quan hóa cho trường chuyển dịch
        
        Parameters
        ----------
        slice_index : int, optional
            Chỉ số lát cắt để hiển thị, by default None (lấy lát cắt giữa)
        downsample : int, optional
            Hệ số giảm minthumple để hiển thị vector, by default 4
        scale : float, optional
            Hệ số tỷ lệ cho độ dài vector, by default 1.0
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            Tọa độ X, Y và thành phần U, V của trường vector
        """
        if self.field_array is None:
            return None, None, None
        
        # Lấy lát cắt giữa nếu không chỉ định
        if slice_index is None:
            slice_index = self.field_array.shape[0] // 2
        
        # Lấy lát cắt trường vector cho lát cắt chỉ định
        slice_field = self.field_array[slice_index, :, :, :]
        
        # Tạo lưới tọa độ
        y, x = np.mgrid[0:slice_field.shape[0]:downsample, 0:slice_field.shape[1]:downsample]
        
        # Lấy các thành phần vector, giảm minthumple và tỷ lệ
        u = slice_field[::downsample, ::downsample, 2] * scale  # x component
        v = slice_field[::downsample, ::downsample, 1] * scale  # y component
        
        return x, y, u, v 