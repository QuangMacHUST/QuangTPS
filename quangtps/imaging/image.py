#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module quản lý hình ảnh y tế cho QuangTPS.

Module này định nghĩa các lớp và hàm để làm việc với hình ảnh y tế,
bao gồm đọc, xử lý, và chuyển đổi giữa các định dạng khác nhau.
"""

import numpy as np
import SimpleITK as sitk
import logging
from typing import Optional, Tuple, Dict, Any, List, Union
import os

from quangtps.core.logging import get_logger

logger = get_logger(__name__)

class Image:
    """
    Lớp đại diện cho một hình ảnh y tế.
    
    Hỗ trợ các loại hình ảnh y tế phổ biến như CT, MRI, PET, CBCT, etc.
    và cung cấp các phương thức để thao tác với dữ liệu hình ảnh.
    """
    
    def __init__(self, data: Optional[np.ndarray] = None, metadata: Optional[Dict[str, Any]] = None):
        """
        Khởi tạo đối tượng Image.
        
        Args:
            data: Mảng NumPy chứa dữ liệu hình ảnh
            metadata: Dictionary chứa các thông tin mô tả về hình ảnh
        """
        self.data = data
        self.metadata = metadata or {}
        self._sitk_image = None
        
    @property
    def shape(self) -> Tuple[int, ...]:
        """Trả về kích thước dữ liệu hình ảnh."""
        if self.data is not None:
            return self.data.shape
        return (0, 0, 0)
    
    @property
    def dtype(self):
        """Trả về kiểu dữ liệu của hình ảnh."""
        if self.data is not None:
            return self.data.dtype
        return None
    
    @property
    def pixel_spacing(self) -> Tuple[float, float, float]:
        """Trả về khoảng cách giữa các pixel trong hình ảnh (x, y, z)."""
        return (
            self.metadata.get('pixel_spacing_x', 1.0),
            self.metadata.get('pixel_spacing_y', 1.0),
            self.metadata.get('slice_thickness', 1.0)
        )
    
    @property
    def origin(self) -> Tuple[float, float, float]:
        """Trả về tọa độ gốc của hình ảnh."""
        return (
            self.metadata.get('origin_x', 0.0),
            self.metadata.get('origin_y', 0.0),
            self.metadata.get('origin_z', 0.0)
        )
    
    @property
    def direction(self) -> Tuple[float, ...]:
        """Trả về ma trận hướng của hình ảnh."""
        return self.metadata.get('direction', (1, 0, 0, 0, 1, 0, 0, 0, 1))
    
    @property
    def window_center(self) -> int:
        """Trả về giá trị trung tâm cửa sổ mặc định."""
        return self.metadata.get('window_center', 40)
    
    @property
    def window_width(self) -> int:
        """Trả về chiều rộng cửa sổ mặc định."""
        return self.metadata.get('window_width', 400)
    
    @property
    def modality(self) -> str:
        """Trả về loại hình ảnh."""
        return self.metadata.get('modality', 'Unknown')
    
    @classmethod
    def from_sitk(cls, sitk_image: sitk.Image) -> 'Image':
        """
        Tạo đối tượng Image từ một SimpleITK Image.
        
        Args:
            sitk_image: Đối tượng SimpleITK Image
            
        Returns:
            Đối tượng Image mới
        """
        # Chuyển đổi từ SimpleITK sang NumPy
        data = sitk.GetArrayFromImage(sitk_image)
        
        # Lấy thông tin metadata
        size = sitk_image.GetSize()
        spacing = sitk_image.GetSpacing()
        origin = sitk_image.GetOrigin()
        direction = sitk_image.GetDirection()
        
        metadata = {
            'size_x': size[0],
            'size_y': size[1],
            'size_z': size[2] if len(size) > 2 else 1,
            'pixel_spacing_x': spacing[0],
            'pixel_spacing_y': spacing[1],
            'slice_thickness': spacing[2] if len(spacing) > 2 else 1.0,
            'origin_x': origin[0],
            'origin_y': origin[1],
            'origin_z': origin[2] if len(origin) > 2 else 0.0,
            'direction': direction
        }
        
        # Tính giá trị cửa sổ mặc định dựa trên dữ liệu
        if data.size > 0:
            min_val = np.min(data)
            max_val = np.max(data)
            range_val = max_val - min_val
            
            if range_val > 0:
                # Sử dụng giá trị tính toán cho cửa sổ
                metadata['window_width'] = int(range_val)
                metadata['window_center'] = int(min_val + range_val / 2)
            else:
                # Giá trị mặc định cho các hình ảnh như nhau
                metadata['window_width'] = 1
                metadata['window_center'] = int(min_val)
        
        # Tạo đối tượng Image mới
        image = cls(data, metadata)
        image._sitk_image = sitk_image
        
        return image
    
    def to_sitk(self) -> sitk.Image:
        """
        Chuyển đổi Image thành SimpleITK Image.
        
        Returns:
            SimpleITK Image
        """
        if self._sitk_image is not None:
            return self._sitk_image
            
        if self.data is None:
            raise ValueError("Không có dữ liệu để chuyển đổi")
            
        # Chuyển đổi từ NumPy sang SimpleITK
        sitk_image = sitk.GetImageFromArray(self.data)
        
        # Đặt các thông số về không gian
        sitk_image.SetSpacing((
            self.metadata.get('pixel_spacing_x', 1.0),
            self.metadata.get('pixel_spacing_y', 1.0),
            self.metadata.get('slice_thickness', 1.0)
        ))
        
        sitk_image.SetOrigin((
            self.metadata.get('origin_x', 0.0),
            self.metadata.get('origin_y', 0.0),
            self.metadata.get('origin_z', 0.0)
        ))
        
        if 'direction' in self.metadata:
            sitk_image.SetDirection(self.metadata['direction'])
            
        self._sitk_image = sitk_image
        return sitk_image
    
    def resample(self, new_spacing: Tuple[float, float, float] = None, 
                 new_size: Tuple[int, int, int] = None,
                 interpolation: int = sitk.sitkLinear) -> 'Image':
        """
        Thực hiện resampling hình ảnh với độ phân giải mới.
        
        Args:
            new_spacing: Khoảng cách pixel mới (x, y, z)
            new_size: Kích thước mới của hình ảnh (x, y, z)
            interpolation: Phương pháp nội suy, mặc định là tuyến tính
            
        Returns:
            Đối tượng Image mới sau khi resampling
        """
        if self.data is None:
            raise ValueError("Không có dữ liệu để resampling")
            
        # Chuyển đổi sang SimpleITK Image
        sitk_image = self.to_sitk()
        
        # Nếu không chỉ định spacing mới, sử dụng spacing hiện tại
        if new_spacing is None:
            new_spacing = self.pixel_spacing
            
        # Nếu không chỉ định kích thước mới, tính toán dựa trên spacing
        if new_size is None:
            old_size = sitk_image.GetSize()
            old_spacing = sitk_image.GetSpacing()
            
            new_size = [
                int(round(old_size[0] * old_spacing[0] / new_spacing[0])),
                int(round(old_size[1] * old_spacing[1] / new_spacing[1])),
                int(round(old_size[2] * old_spacing[2] / new_spacing[2])) if len(old_size) > 2 else 1
            ]
        
        # Tạo resampler
        resampler = sitk.ResampleImageFilter()
        resampler.SetInterpolator(interpolation)
        resampler.SetOutputSpacing(new_spacing)
        resampler.SetSize(new_size)
        resampler.SetOutputDirection(sitk_image.GetDirection())
        resampler.SetOutputOrigin(sitk_image.GetOrigin())
        resampler.SetTransform(sitk.Transform())
        resampler.SetDefaultPixelValue(sitk.GetArrayFromImage(sitk_image).min())
        
        # Thực hiện resampling
        resampled_image = resampler.Execute(sitk_image)
        
        # Chuyển về Image
        return Image.from_sitk(resampled_image)
    
    def get_slice(self, slice_idx: int, axis: int = 0) -> np.ndarray:
        """
        Lấy một lát cắt của hình ảnh.
        
        Args:
            slice_idx: Chỉ số của lát cắt
            axis: Trục để lấy lát cắt (0: z, 1: y, 2: x)
            
        Returns:
            Mảng NumPy chứa dữ liệu lát cắt
        """
        if self.data is None:
            return None
            
        # Đảm bảo chỉ số nằm trong khoảng hợp lệ
        shape = self.data.shape
        if axis == 0:
            max_idx = shape[0] - 1
        elif axis == 1:
            max_idx = shape[1] - 1
        else:
            max_idx = shape[2] - 1 if len(shape) > 2 else 0
            
        slice_idx = max(0, min(slice_idx, max_idx))
        
        # Lấy lát cắt theo trục tương ứng
        if axis == 0:
            return self.data[slice_idx, :, :]
        elif axis == 1:
            return self.data[:, slice_idx, :]
        else:
            return self.data[:, :, slice_idx] if len(shape) > 2 else self.data
    
    def save(self, filepath: str, compress: bool = True) -> bool:
        """
        Lưu hình ảnh vào file.
        
        Args:
            filepath: Đường dẫn file để lưu
            compress: Có nén file hay không
            
        Returns:
            True nếu lưu thành công, False nếu thất bại
        """
        try:
            # Chuyển đổi sang SimpleITK Image
            sitk_image = self.to_sitk()
            
            # Tạo thư mục nếu chưa tồn tại
            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
            
            # Lưu file
            writer = sitk.ImageFileWriter()
            writer.SetFileName(filepath)
            writer.SetUseCompression(compress)
            writer.Execute(sitk_image)
            
            logger.info(f"Đã lưu hình ảnh thành công vào: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Lỗi khi lưu hình ảnh: {str(e)}")
            return False
    
    @classmethod
    def load(cls, filepath: str) -> 'Image':
        """
        Đọc hình ảnh từ file.
        
        Args:
            filepath: Đường dẫn file để đọc
            
        Returns:
            Đối tượng Image mới
            
        Raises:
            ValueError: Nếu file không tồn tại hoặc không đọc được
        """
        if not os.path.exists(filepath):
            raise ValueError(f"File không tồn tại: {filepath}")
            
        try:
            # Đọc file bằng SimpleITK
            reader = sitk.ImageFileReader()
            reader.SetFileName(filepath)
            sitk_image = reader.Execute()
            
            # Chuyển đổi sang Image
            image = cls.from_sitk(sitk_image)
            
            # Thêm đường dẫn file vào metadata
            image.metadata['filepath'] = filepath
            
            return image
            
        except Exception as e:
            logger.error(f"Lỗi khi đọc file hình ảnh: {str(e)}")
            raise ValueError(f"Không thể đọc file hình ảnh: {str(e)}")
    
    def apply_window(self, window_center: int = None, window_width: int = None) -> np.ndarray:
        """
        Áp dụng cửa sổ và chuyển đổi sang dạng 8-bit cho hiển thị.
        
        Args:
            window_center: Giá trị trung tâm cửa sổ
            window_width: Chiều rộng cửa sổ
            
        Returns:
            Mảng NumPy 8-bit đã áp dụng cửa sổ
        """
        if self.data is None:
            return None
            
        # Sử dụng giá trị mặc định nếu không có đầu vào
        if window_center is None:
            window_center = self.window_center
        if window_width is None:
            window_width = self.window_width
            
        # Tính toán giới hạn cửa sổ
        min_val = window_center - window_width // 2
        max_val = window_center + window_width // 2
        
        # Clip giá trị trong khoảng [min_val, max_val]
        clipped = np.clip(self.data, min_val, max_val)
        
        # Chuẩn hóa về khoảng [0, 255]
        if max_val > min_val:
            normalized = ((clipped - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        else:
            normalized = np.zeros_like(clipped, dtype=np.uint8)
            
        return normalized
