"""
Module tính toán TERMA (Total Energy Released per unit MAss).

TERMA là tổng năng lượng phát ra trên đơn vị khối lượng tại điểm tương tác
của photon với vật chất. TERMA là bước đầu tiên trong quá trình tính toán
phân bố liều, trước khi áp dụng kernel để mô phỏng sự lan truyền của năng
lượng từ điểm tương tác ban đầu.
"""

import numpy as np
import logging
from typing import Tuple, Dict, Any, Optional

logger = logging.getLogger(__name__)

def calculate_terma(density_array: np.ndarray,
                   spacing: Tuple[float, float, float],
                   beam_energy: float,
                   beam_direction: np.ndarray,
                   beam_mu: float,
                   beam_isocenter: np.ndarray,
                   beam_field_size: Tuple[float, float] = (100.0, 100.0),
                   beam_spectrum: Optional[Dict[float, float]] = None) -> np.ndarray:
    """
    Tính toán TERMA cho một chùm tia.
    
    Parameters:
        density_array (np.ndarray): Mảng mật độ điện tử
        spacing (tuple): Khoảng cách voxel (mm)
        beam_energy (float): Năng lượng chùm tia (MV)
        beam_direction (np.ndarray): Hướng chùm tia
        beam_mu (float): Số MU (Monitor Units)
        beam_isocenter (np.ndarray): Tọa độ tâm (mm)
        beam_field_size (tuple, optional): Kích thước trường (mm)
        beam_spectrum (dict, optional): Phổ năng lượng chùm tia (energy -> weight)
    
    Returns:
        np.ndarray: Mảng TERMA
    """
    logger.info(f"Calculating TERMA for beam with energy {beam_energy} MV")
    
    # Khởi tạo mảng TERMA
    terma = np.zeros_like(density_array, dtype=np.float32)
    
    # Lấy kích thước mảng
    depth, height, width = density_array.shape
    
    # Chuẩn hóa hướng chùm tia
    beam_direction = beam_direction / np.linalg.norm(beam_direction)
    
    # Chuyển đổi tọa độ isocenter từ hệ tọa độ thế giới sang hệ tọa độ voxel
    # Giả định rằng isocenter được định nghĩa trong cùng hệ tọa độ với spacing
    isocenter_voxel = np.array([
        beam_isocenter[2] / spacing[2],  # Z (depth)
        beam_isocenter[1] / spacing[1],  # Y (height)
        beam_isocenter[0] / spacing[0]   # X (width)
    ])
    
    # Tính toán hệ số suy giảm chùm tia (beam attenuation) dựa trên năng lượng
    # Giá trị μ/ρ (hệ số suy giảm khối) cho nước ở các năng lượng khác nhau
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
    
    # Nội suy hệ số suy giảm khối cho năng lượng cụ thể
    energies = np.array(list(mu_rho_water.keys()))
    mu_rhos = np.array(list(mu_rho_water.values()))
    
    mu_rho = np.interp(beam_energy, energies, mu_rhos) if beam_energy not in mu_rho_water else mu_rho_water[beam_energy]
    
    # Tính tỉ lệ liều-MU tại điểm tham chiếu (thường là 100 cGy/MU ở độ sâu cực đại)
    # Đơn giản hóa: giả định rằng 1 MU = 1 cGy tại độ sâu cực đại
    dose_per_mu = 0.01  # Gy/MU (1 cGy = 0.01 Gy)
    
    # Tính toán TERMA cho mỗi voxel
    # Sử dụng ray tracing để theo dõi chùm tia xuyên qua vật thể
    for z in range(depth):
        for y in range(height):
            for x in range(width):
                # Tính khoảng cách từ voxel đến isocenter
                voxel_pos = np.array([z, y, x])
                vec_to_isocenter = voxel_pos - isocenter_voxel
                
                # Tìm khoảng cách dọc theo chùm tia (chiếu vec_to_isocenter lên beam_direction)
                along_beam_dist = np.dot(vec_to_isocenter, beam_direction)
                
                # Tính khoảng cách vuông góc với chùm tia
                perp_vec = vec_to_isocenter - along_beam_dist * beam_direction
                perp_dist = np.linalg.norm(perp_vec)
                
                # Chuyển đổi khoảng cách từ voxel sang mm
                perp_dist_mm = perp_dist * np.mean(spacing)
                along_beam_dist_mm = along_beam_dist * np.mean(spacing)
                
                # Kiểm tra nếu voxel nằm trong trường chùm tia
                field_x, field_y = beam_field_size
                if abs(perp_vec[2]) * spacing[0] > field_x / 2 or abs(perp_vec[1]) * spacing[1] > field_y / 2:
                    continue  # Voxel nằm ngoài trường chùm tia
                
                # Tính procentional depth dose (PDD) dựa trên along_beam_dist
                # Đơn giản hóa: sử dụng mô hình PDD có dạng exp(-μ*d)
                pdd = calculate_pdd(along_beam_dist_mm, beam_energy)
                
                # Tính off-axis ratio (OAR)
                oar = calculate_oar(perp_dist_mm, field_x, field_y, beam_energy)
                
                # Tính TERMA tại voxel này
                terma_value = dose_per_mu * beam_mu * pdd * oar
                
                # TERMA là liều trước khi xem xét lan truyền năng lượng thứ cấp
                # TERMA = μ/ρ * Ψ (fluence)
                # Ở đây, chúng ta tính fluence từ terma_value
                terma[z, y, x] = terma_value * mu_rho * density_array[z, y, x]
    
    # Cấu trúc chùm tia (beam modifiers như MLC, jaw, v.v.) có thể được áp dụng ở đây
    
    return terma

def calculate_pdd(depth: float, energy: float) -> float:
    """
    Tính Percentage Depth Dose (PDD) cho một độ sâu và năng lượng cụ thể.
    
    Parameters:
        depth (float): Độ sâu (mm)
        energy (float): Năng lượng chùm tia (MV)
    
    Returns:
        float: Giá trị PDD
    """
    # Tính tham số cho mô hình PDD
    # Mô hình đơn giản: PDD(d) = e^(-μ₁*d) + b*e^(-μ₂*d)
    
    # Tham số cho độ sâu cực đại dựa trên năng lượng
    # Đơn giản hóa: d_max = 0.6 * energy (cm)
    d_max = 0.6 * energy * 10  # mm
    
    # Tham số suy giảm
    mu_1 = 0.00800 - 0.00033 * energy
    mu_2 = 0.00200 - 0.00013 * energy
    b = 0.3
    
    if depth < d_max:
        # Build-up region
        return (depth / d_max) * np.exp(mu_1 * (depth - d_max))
    else:
        # Beyond d_max
        return np.exp(-mu_1 * (depth - d_max)) + b * np.exp(-mu_2 * (depth - d_max))

def calculate_oar(distance: float, field_x: float, field_y: float, energy: float) -> float:
    """
    Tính Off-Axis Ratio (OAR) cho một khoảng cách ngang.
    
    Parameters:
        distance (float): Khoảng cách ngang từ trục chùm tia (mm)
        field_x (float): Chiều rộng trường (mm)
        field_y (float): Chiều cao trường (mm)
        energy (float): Năng lượng chùm tia (MV)
    
    Returns:
        float: Giá trị OAR
    """
    # Mô hình đơn giản: OAR(r) = exp(-σ*r²)
    # σ phụ thuộc vào năng lượng và kích thước trường
    
    # Tính σ dựa trên năng lượng và kích thước trường
    field_size = np.sqrt(field_x * field_y) / 10  # Convert to cm
    sigma = 0.0015 - 0.00005 * energy + 0.00001 * field_size
    
    # Tính OAR
    return np.exp(-sigma * distance * distance)

def get_beam_spectrum(energy: float) -> Dict[float, float]:
    """
    Lấy phổ năng lượng cho chùm tia xạ trị.
    
    Parameters:
        energy (float): Năng lượng danh định của chùm tia (MV)
    
    Returns:
        dict: Phổ năng lượng (E -> tỉ lệ)
    """
    # Mô hình đơn giản cho phổ năng lượng
    # Thực tế, phổ năng lượng phức tạp hơn và phụ thuộc vào máy xạ trị cụ thể
    
    spectrum = {}
    
    # Tạo phổ từ 0 đến energy * 1.2
    max_energy = energy * 1.2
    num_bins = 20
    
    for i in range(num_bins):
        e = i * max_energy / (num_bins - 1)
        
        # Mô hình phân bố giản đơn
        if e < 0.2 * energy:
            weight = e / (0.2 * energy)
        elif e < energy:
            weight = 1.0
        else:
            weight = np.exp(-(e - energy) / (0.1 * energy))
        
        spectrum[e] = weight
    
    # Chuẩn hóa phổ
    total = sum(spectrum.values())
    for e in spectrum:
        spectrum[e] /= total
    
    return spectrum
