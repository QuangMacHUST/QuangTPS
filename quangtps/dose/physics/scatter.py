"""
Module mô phỏng tán xạ trong quá trình tính toán liều xạ trị.

Module này cung cấp các hàm để tính toán tán xạ quang tử và điện tử trong vật chất.
Tán xạ là hiện tượng quan trọng cần xem xét trong tính toán liều, đặc biệt là ở
các vùng không đồng nhất và tại ranh giới giữa các loại mô khác nhau.
"""

import numpy as np
import logging
from typing import Tuple, Dict, Any, Optional

logger = logging.getLogger(__name__)

def calculate_scatter(terma: np.ndarray,
                     density: np.ndarray,
                     spacing: Tuple[float, float, float],
                     energy: float,
                     kernel_type: str = 'water') -> np.ndarray:
    """
    Tính toán phân bố tán xạ từ TERMA.
    
    Parameters:
        terma (np.ndarray): Mảng TERMA
        density (np.ndarray): Mảng mật độ điện tử
        spacing (tuple): Khoảng cách voxel (mm)
        energy (float): Năng lượng chùm tia (MV)
        kernel_type (str, optional): Loại kernel tán xạ
    
    Returns:
        np.ndarray: Mảng tán xạ
    """
    logger.info(f"Calculating scatter for {energy} MV")
    
    # Lấy kích thước mảng
    depth, height, width = terma.shape
    
    # Tạo kernel tán xạ
    kernel_size = min(31, depth, height, width)
    kernel = calculate_scatter_kernel(kernel_size, spacing, energy, kernel_type)
    
    # Áp dụng kernel tán xạ bằng tích chập
    # Sử dụng tích chập nhanh thông qua FFT
    import scipy.signal
    scatter = scipy.signal.fftconvolve(terma, kernel, mode='same')
    
    # Chia cho mật độ để chuyển đổi từ năng lượng sang liều
    with np.errstate(divide='ignore', invalid='ignore'):
        scatter = np.divide(scatter, density)
        scatter = np.nan_to_num(scatter, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scatter

def calculate_scatter_kernel(size: int,
                            spacing: Tuple[float, float, float],
                            energy: float,
                            kernel_type: str = 'water') -> np.ndarray:
    """
    Tính toán kernel tán xạ.
    
    Parameters:
        size (int): Kích thước kernel (số voxel mỗi chiều)
        spacing (tuple): Khoảng cách voxel (mm)
        energy (float): Năng lượng chùm tia (MV)
        kernel_type (str, optional): Loại kernel tán xạ
    
    Returns:
        np.ndarray: Kernel tán xạ
    """
    # Tạo kernel tán xạ
    # Kích thước kernel phải là lẻ
    if size % 2 == 0:
        size += 1
    
    # Khởi tạo kernel
    kernel = np.zeros((size, size, size), dtype=np.float32)
    
    # Tâm của kernel
    center = size // 2
    
    # Tính bán kính tối đa (mm)
    max_radius = min(spacing) * (size // 2)
    
    # Tạo lưới tọa độ
    x = np.linspace(-center * spacing[0], center * spacing[0], size)
    y = np.linspace(-center * spacing[1], center * spacing[1], size)
    z = np.linspace(-center * spacing[2], center * spacing[2], size)
    xv, yv, zv = np.meshgrid(x, y, z, indexing='ij')
    
    # Tính khoảng cách từ tâm
    rv = np.sqrt(xv**2 + yv**2 + zv**2)
    
    # Khoảng cách từng voxel đến tâm
    for i in range(size):
        for j in range(size):
            for k in range(size):
                r = rv[i, j, k]
                if r < 1e-6:  # Tránh phân kỳ tại r=0
                    continue
                
                # Tính giá trị kernel dựa trên loại và năng lượng
                if kernel_type.lower() == 'water':
                    # Mô hình tán xạ đơn giản cho water
                    # Hàm mô tả phân bố tán xạ dựa trên khoảng cách:
                    # kernel ~ A*exp(-B*r)/r²
                    
                    # Tham số A và B phụ thuộc vào năng lượng
                    A = 1.0
                    if energy <= 6.0:
                        B = 0.047 - 0.0022 * energy
                    else:
                        B = 0.032 - 0.00086 * energy
                    
                    # Tính giá trị kernel
                    kernel[i, j, k] = A * np.exp(-B * r) / (r * r)
                
                elif kernel_type.lower() == 'bone':
                    # Mô hình tán xạ cho xương
                    A = 0.85  # Giảm tán xạ trong xương
                    B = 0.060 - 0.0025 * energy
                    kernel[i, j, k] = A * np.exp(-B * r) / (r * r)
                
                elif kernel_type.lower() == 'lung':
                    # Mô hình tán xạ cho phổi
                    A = 1.20  # Tăng tán xạ trong phổi
                    B = 0.035 - 0.0018 * energy
                    kernel[i, j, k] = A * np.exp(-B * r) / (r * r)
                
                else:
                    # Mặc định: water
                    A = 1.0
                    B = 0.047 - 0.0022 * energy if energy <= 6.0 else 0.032 - 0.00086 * energy
                    kernel[i, j, k] = A * np.exp(-B * r) / (r * r)
    
    # Normalization
    # Tổng của kernel nên là 1 để bảo toàn năng lượng
    voxel_volume = spacing[0] * spacing[1] * spacing[2]
    total = np.sum(kernel) * voxel_volume
    
    if total > 0:
        kernel /= total
    
    return kernel

def calculate_advanced_scatter(terma: np.ndarray,
                              density: np.ndarray,
                              spacing: Tuple[float, float, float],
                              energy: float,
                              material_map: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Tính toán tán xạ nâng cao với nhiều vật liệu.
    
    Parameters:
        terma (np.ndarray): Mảng TERMA
        density (np.ndarray): Mảng mật độ điện tử
        spacing (tuple): Khoảng cách voxel (mm)
        energy (float): Năng lượng chùm tia (MV)
        material_map (np.ndarray, optional): Bản đồ vật liệu
    
    Returns:
        np.ndarray: Mảng tán xạ
    """
    logger.info(f"Calculating advanced scatter for {energy} MV")
    
    # Lấy kích thước mảng
    depth, height, width = terma.shape
    
    # Khởi tạo mảng tán xạ
    scatter = np.zeros_like(terma)
    
    # Tạo các kernel tán xạ cho các loại vật liệu khác nhau
    water_kernel = calculate_scatter_kernel(31, spacing, energy, 'water')
    bone_kernel = calculate_scatter_kernel(31, spacing, energy, 'bone')
    lung_kernel = calculate_scatter_kernel(31, spacing, energy, 'lung')
    
    # Nếu không có material_map, tạo một bản đồ dựa trên mật độ
    if material_map is None:
        material_map = np.zeros_like(density, dtype=np.int32)
        material_map[(density >= 0.9) & (density <= 1.1)] = 0  # water
        material_map[density > 1.1] = 1  # bone
        material_map[density < 0.9] = 2  # lung/air
    
    # Tính tán xạ cho từng loại vật liệu
    import scipy.signal
    
    # Water
    water_mask = (material_map == 0)
    if np.any(water_mask):
        water_terma = terma.copy()
        water_terma[~water_mask] = 0
        water_scatter = scipy.signal.fftconvolve(water_terma, water_kernel, mode='same')
        scatter += water_scatter
    
    # Bone
    bone_mask = (material_map == 1)
    if np.any(bone_mask):
        bone_terma = terma.copy()
        bone_terma[~bone_mask] = 0
        bone_scatter = scipy.signal.fftconvolve(bone_terma, bone_kernel, mode='same')
        scatter += bone_scatter
    
    # Lung
    lung_mask = (material_map == 2)
    if np.any(lung_mask):
        lung_terma = terma.copy()
        lung_terma[~lung_mask] = 0
        lung_scatter = scipy.signal.fftconvolve(lung_terma, lung_kernel, mode='same')
        scatter += lung_scatter
    
    # Chia cho mật độ để chuyển đổi từ năng lượng sang liều
    with np.errstate(divide='ignore', invalid='ignore'):
        scatter = np.divide(scatter, density)
        scatter = np.nan_to_num(scatter, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scatter
