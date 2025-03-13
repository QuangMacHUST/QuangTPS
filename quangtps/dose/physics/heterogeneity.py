"""
Module xử lý không đồng nhất trong tính toán liều xạ trị.

Module này cung cấp các hàm để hiệu chỉnh tính toán liều tại các vùng không đồng nhất,
như ranh giới giữa mô mềm và xương, hoặc mô mềm và phổi/không khí. Hiệu chỉnh không
đồng nhất là một thành phần quan trọng để tăng độ chính xác của tính toán liều trong
các tình huống lâm sàng thực tế.
"""

import numpy as np
import logging
from typing import Tuple, Dict, Any, Optional, Callable

logger = logging.getLogger(__name__)

def apply_heterogeneity_correction(dose: np.ndarray,
                                  density: np.ndarray,
                                  spacing: Tuple[float, float, float],
                                  energy: float,
                                  method: str = 'batho') -> np.ndarray:
    """
    Áp dụng hiệu chỉnh không đồng nhất cho phân bố liều.
    
    Parameters:
        dose (np.ndarray): Mảng liều chưa hiệu chỉnh
        density (np.ndarray): Mảng mật độ điện tử
        spacing (tuple): Khoảng cách voxel (mm)
        energy (float): Năng lượng chùm tia (MV)
        method (str, optional): Phương pháp hiệu chỉnh ('batho', 'epp', 'equivalent_tad')
    
    Returns:
        np.ndarray: Mảng liều sau hiệu chỉnh
    """
    logger.info(f"Applying {method} heterogeneity correction for {energy} MV")
    
    # Kiểm tra phương pháp hiệu chỉnh
    if method.lower() == 'batho':
        return batho_correction(dose, density, spacing, energy)
    elif method.lower() == 'epp':
        return equivalent_path_length_correction(dose, density, spacing, energy)
    elif method.lower() == 'equivalent_tad':
        return equivalent_tad_correction(dose, density, spacing, energy)
    else:
        logger.warning(f"Unknown heterogeneity correction method: {method}. Using batho method.")
        return batho_correction(dose, density, spacing, energy)

def batho_correction(dose: np.ndarray,
                    density: np.ndarray,
                    spacing: Tuple[float, float, float],
                    energy: float) -> np.ndarray:
    """
    Áp dụng hiệu chỉnh không đồng nhất Batho.
    
    Parameters:
        dose (np.ndarray): Mảng liều chưa hiệu chỉnh
        density (np.ndarray): Mảng mật độ điện tử
        spacing (tuple): Khoảng cách voxel (mm)
        energy (float): Năng lượng chùm tia (MV)
    
    Returns:
        np.ndarray: Mảng liều sau hiệu chỉnh
    """
    # Khởi tạo mảng liều hiệu chỉnh
    corrected_dose = np.copy(dose)
    
    # Hệ số suy giảm khối dựa trên năng lượng
    mu_rho_water = get_mu_rho_for_energy(energy)
    
    # Lấy kích thước mảng
    depth, height, width = dose.shape
    
    # Giả định hướng chùm tia dọc theo trục z (từ trên xuống)
    for z in range(depth):
        for y in range(height):
            for x in range(width):
                if density[z, y, x] <= 0:
                    continue  # Bỏ qua các voxel không có mật độ
                
                # Tính hiệu chỉnh Batho
                # CF = (ρ/ρw)^(μ/ρ * d)
                
                # Chiều dài đường dẫn điều trị (mm)
                path_length = z * spacing[2]
                
                # Tính hệ số hiệu chỉnh
                rho_ratio = density[z, y, x]  # Đã là mật độ tương đối so với nước
                correction_factor = rho_ratio ** (mu_rho_water * path_length)
                
                # Áp dụng hiệu chỉnh
                corrected_dose[z, y, x] *= correction_factor
    
    return corrected_dose

def equivalent_path_length_correction(dose: np.ndarray,
                                     density: np.ndarray,
                                     spacing: Tuple[float, float, float],
                                     energy: float) -> np.ndarray:
    """
    Áp dụng hiệu chỉnh không đồng nhất dựa trên đường dẫn tương đương.
    
    Parameters:
        dose (np.ndarray): Mảng liều chưa hiệu chỉnh
        density (np.ndarray): Mảng mật độ điện tử
        spacing (tuple): Khoảng cách voxel (mm)
        energy (float): Năng lượng chùm tia (MV)
    
    Returns:
        np.ndarray: Mảng liều sau hiệu chỉnh
    """
    # Khởi tạo mảng liều hiệu chỉnh
    corrected_dose = np.zeros_like(dose)
    
    # Lấy kích thước mảng
    depth, height, width = dose.shape
    
    # Tính đường dẫn tương đương cho mỗi điểm
    for y in range(height):
        for x in range(width):
            # Tính toán đường dẫn tương đương
            equivalent_path = np.zeros(depth)
            
            # Tích lũy mật độ dọc theo đường dẫn
            for z in range(depth):
                if z == 0:
                    equivalent_path[z] = density[z, y, x] * spacing[2]
                else:
                    equivalent_path[z] = equivalent_path[z-1] + density[z, y, x] * spacing[2]
            
            # Tính liều dựa trên đường dẫn tương đương
            for z in range(depth):
                # Tìm vị trí trong nước với cùng đường dẫn tương đương
                equivalent_depth = equivalent_path[z] / 1.0  # Mật độ nước = 1.0
                
                # Tìm chỉ số voxel tương ứng
                eq_z = int(equivalent_depth / spacing[2])
                
                # Nội suy nếu vị trí không khớp chính xác
                if eq_z < depth - 1:
                    frac = equivalent_depth / spacing[2] - eq_z
                    interp_dose = (1 - frac) * dose[eq_z, y, x] + frac * dose[eq_z + 1, y, x]
                    corrected_dose[z, y, x] = interp_dose
                elif eq_z < depth:
                    corrected_dose[z, y, x] = dose[eq_z, y, x]
                else:
                    # Nếu đường dẫn tương đương vượt quá độ dày, sử dụng giá trị biên
                    corrected_dose[z, y, x] = dose[depth - 1, y, x]
    
    return corrected_dose

def equivalent_tad_correction(dose: np.ndarray,
                             density: np.ndarray,
                             spacing: Tuple[float, float, float],
                             energy: float) -> np.ndarray:
    """
    Áp dụng hiệu chỉnh không đồng nhất dựa trên TAD (Tissue-Air Ratio) tương đương.
    
    Parameters:
        dose (np.ndarray): Mảng liều chưa hiệu chỉnh
        density (np.ndarray): Mảng mật độ điện tử
        spacing (tuple): Khoảng cách voxel (mm)
        energy (float): Năng lượng chùm tia (MV)
    
    Returns:
        np.ndarray: Mảng liều sau hiệu chỉnh
    """
    # Khởi tạo mảng liều hiệu chỉnh
    corrected_dose = np.copy(dose)
    
    # Lấy TAR (Tissue-Air Ratio) cho năng lượng chùm tia
    tar_function = get_tar_function(energy)
    
    # Lấy kích thước mảng
    depth, height, width = dose.shape
    
    # Tính đường dẫn bên trên mỗi điểm
    path_above = np.zeros((depth, height, width))
    
    # Tích lũy đường dẫn từ trên xuống
    for z in range(depth):
        if z == 0:
            path_above[z, :, :] = 0.0
        else:
            # Tích lũy mật độ nhân với khoảng cách
            for i in range(z):
                path_above[z, :, :] += density[i, :, :] * spacing[2]
    
    # Tính liều hiệu chỉnh
    for z in range(depth):
        for y in range(height):
            for x in range(width):
                # Lấy đường dẫn vật lý từ bề mặt đến điểm
                physical_depth = z * spacing[2]
                
                # Lấy đường dẫn tương đương từ bề mặt đến điểm
                equivalent_depth = path_above[z, y, x]
                
                # Tính hiệu chỉnh dựa trên tỉ lệ TAR
                tar_physical = tar_function(physical_depth)
                tar_equivalent = tar_function(equivalent_depth)
                
                # Tránh chia cho 0
                if tar_physical > 0:
                    correction_factor = tar_equivalent / tar_physical
                    corrected_dose[z, y, x] *= correction_factor
    
    return corrected_dose

def get_mu_rho_for_energy(energy: float) -> float:
    """
    Lấy hệ số suy giảm khối (μ/ρ) cho nước ở một năng lượng cụ thể.
    
    Parameters:
        energy (float): Năng lượng chùm tia (MV)
    
    Returns:
        float: Hệ số suy giảm khối (cm²/g)
    """
    # Bảng hệ số suy giảm khối cho nước ở các năng lượng khác nhau
    # Nguồn: NIST Data Gateway
    mu_rho_water = {
        1.0: 0.0706,   # 1 MV
        2.0: 0.0494,   # 2 MV
        4.0: 0.0358,   # 4 MV
        6.0: 0.0297,   # 6 MV
        10.0: 0.0233,  # 10 MV
        15.0: 0.0195,  # 15 MV
        18.0: 0.0180,  # 18 MV
        20.0: 0.0174   # 20 MV
    }
    
    # Nội suy hệ số cho năng lượng cụ thể
    energies = np.array(list(mu_rho_water.keys()))
    mu_rhos = np.array(list(mu_rho_water.values()))
    
    if energy in mu_rho_water:
        return mu_rho_water[energy]
    else:
        return np.interp(energy, energies, mu_rhos)

def get_tar_function(energy: float) -> Callable[[float], float]:
    """
    Lấy hàm tính Tissue-Air Ratio (TAR) cho một năng lượng cụ thể.
    
    Parameters:
        energy (float): Năng lượng chùm tia (MV)
    
    Returns:
        callable: Hàm tính TAR từ độ sâu (mm)
    """
    def tar_function(depth_mm: float) -> float:
        """
        Tính Tissue-Air Ratio (TAR) cho một độ sâu.
        
        Parameters:
            depth_mm (float): Độ sâu (mm)
        
        Returns:
            float: Giá trị TAR
        """
        # Chuyển đổi mm sang cm
        depth_cm = depth_mm / 10.0
        
        # Tham số cho mô hình TAR dựa trên năng lượng
        if energy <= 4.0:
            # Tham số cho năng lượng thấp
            a = 1.0
            b = 0.0044 * energy + 0.008
            c = 0.02
            d = 0.8
        elif energy <= 10.0:
            # Tham số cho năng lượng trung bình
            a = 1.0
            b = 0.0033 * energy + 0.012
            c = 0.015
            d = 0.7
        else:
            # Tham số cho năng lượng cao
            a = 1.0
            b = 0.0022 * energy + 0.023
            c = 0.012
            d = 0.6
        
        # Mô hình TAR: a*exp(-b*d) + c*exp(-d*d)
        return a * np.exp(-b * depth_cm) + c * np.exp(-d * depth_cm)
    
    return tar_function

def detect_heterogeneities(density: np.ndarray) -> np.ndarray:
    """
    Phát hiện vùng không đồng nhất trong mảng mật độ.
    
    Parameters:
        density (np.ndarray): Mảng mật độ điện tử
    
    Returns:
        np.ndarray: Mảng đánh dấu các vùng không đồng nhất
    """
    # Khởi tạo mảng đánh dấu
    heterogeneity_map = np.zeros_like(density, dtype=np.int32)
    
    # Lấy kích thước mảng
    depth, height, width = density.shape
    
    # Định nghĩa các ngưỡng mật độ
    lung_threshold = 0.4  # Mật độ phổi < 0.4
    water_min = 0.9       # Mật độ nước > 0.9
    water_max = 1.1       # Mật độ nước < 1.1
    bone_threshold = 1.2  # Mật độ xương > 1.2
    
    # Đánh dấu các vùng dựa trên mật độ
    # 0: không xác định
    # 1: không khí
    # 2: phổi
    # 3: mô mềm/nước
    # 4: xương
    # 5: implant kim loại
    
    # Không khí
    heterogeneity_map[density < 0.1] = 1
    
    # Phổi
    heterogeneity_map[(0.1 <= density) & (density < lung_threshold)] = 2
    
    # Mô mềm/nước
    heterogeneity_map[(water_min <= density) & (density <= water_max)] = 3
    
    # Xương
    heterogeneity_map[(bone_threshold <= density) & (density < 3.0)] = 4
    
    # Implant kim loại
    heterogeneity_map[density >= 3.0] = 5
    
    # Phát hiện ranh giới không đồng nhất
    # Đánh dấu các voxel nằm gần ranh giới giữa các loại mô khác nhau
    border_map = np.zeros_like(heterogeneity_map, dtype=np.bool_)
    
    for z in range(1, depth - 1):
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                # Kiểm tra 6 voxel xung quanh (mặt tiếp xúc)
                current_type = heterogeneity_map[z, y, x]
                
                if (current_type != heterogeneity_map[z-1, y, x] or
                    current_type != heterogeneity_map[z+1, y, x] or
                    current_type != heterogeneity_map[z, y-1, x] or
                    current_type != heterogeneity_map[z, y+1, x] or
                    current_type != heterogeneity_map[z, y, x-1] or
                    current_type != heterogeneity_map[z, y, x+1]):
                    border_map[z, y, x] = True
    
    # Đánh dấu ranh giới trong heterogeneity_map
    # 10: ranh giới không đồng nhất
    heterogeneity_map[border_map] = 10
    
    return heterogeneity_map
