"""
Module tính toán RBE (Relative Biological Effectiveness) trong xạ trị.

RBE là thước đo hiệu quả sinh học tương đối của một loại bức xạ so với bức xạ tham chiếu 
(thường là photon 60Co hoặc 250 kVp X-ray) trong việc tạo ra cùng một hiệu ứng sinh học.
Module này cung cấp các mô hình RBE khác nhau cho xạ trị proton, ion và các loại bức xạ khác.
"""

import numpy as np
import logging
from typing import Tuple, Dict, Optional, Any, List, Union, Callable

logger = logging.getLogger(__name__)

def calculate_rbe(dose_distribution: np.ndarray,
                 particle_type: str,
                 let_distribution: Optional[np.ndarray] = None,
                 alpha_beta_ratio: Union[float, np.ndarray] = 10.0,
                 dose_per_fraction: Union[float, np.ndarray] = 2.0,
                 tissue_type: Union[str, np.ndarray] = 'generic',
                 model: str = 'mcnamara') -> np.ndarray:
    """
    Tính toán phân bố RBE (Relative Biological Effectiveness).
    
    Parameters:
        dose_distribution (np.ndarray): Phân bố liều vật lý (Gy)
        particle_type (str): Loại hạt ('proton', 'carbon', 'helium', 'neutron', 'photon')
        let_distribution (np.ndarray, optional): Phân bố LET (keV/μm)
        alpha_beta_ratio (float or np.ndarray): Tỉ lệ alpha/beta của mô (Gy)
        dose_per_fraction (float or np.ndarray): Liều trên mỗi phân đoạn (Gy)
        tissue_type (str or np.ndarray): Loại mô ('tumor', 'normal', 'generic' hoặc mã mô cụ thể)
        model (str): Mô hình tính RBE
    
    Returns:
        np.ndarray: Phân bố RBE
    """
    logger.info(f"Calculating RBE for {particle_type} using {model} model")
    
    # Khởi tạo phân bố RBE
    rbe_distribution = np.ones_like(dose_distribution)
    
    # Với photon, RBE = 1 theo định nghĩa
    if particle_type.lower() == 'photon':
        return rbe_distribution
    
    # Kiểm tra LET
    if let_distribution is None and particle_type.lower() not in ['photon']:
        logger.warning(f"LET distribution not provided for {particle_type}. Using default RBE values.")
        
        # Gán RBE mặc định cho từng loại hạt
        if particle_type.lower() == 'proton':
            rbe_distribution = np.ones_like(dose_distribution) * 1.1
        elif particle_type.lower() == 'carbon':
            rbe_distribution = np.ones_like(dose_distribution) * 3.0
        elif particle_type.lower() == 'helium':
            rbe_distribution = np.ones_like(dose_distribution) * 2.0
        elif particle_type.lower() == 'neutron':
            rbe_distribution = np.ones_like(dose_distribution) * 2.5
        
        return rbe_distribution
    
    # Chọn mô hình RBE dựa trên loại hạt và yêu cầu
    if particle_type.lower() == 'proton':
        # RBE cho proton
        if model.lower() == 'constant':
            # Mô hình RBE hằng số: 1.1 cho proton theo tiêu chuẩn lâm sàng
            rbe_distribution = np.ones_like(dose_distribution) * 1.1
        
        elif model.lower() == 'mcnamara':
            # Mô hình McNamara: RBE = 1 + (0.843 * LET / (α/β))
            rbe_distribution = mcnamara_model(let_distribution, alpha_beta_ratio)
        
        elif model.lower() == 'wedenberg':
            # Mô hình Wedenberg: RBE = 1 + (0.434 * LET / (α/β))
            rbe_distribution = wedenberg_model(let_distribution, alpha_beta_ratio)
        
        elif model.lower() == 'carlson':
            # Mô hình Carlson: Dựa trên LET và tỉ lệ alpha/beta với hiệu chỉnh cho loại mô
            rbe_distribution = carlson_model(let_distribution, alpha_beta_ratio, tissue_type)
        
        elif model.lower() == 'variable':
            # Mô hình biến đổi theo vị trí: RBE cao hơn tại đỉnh Bragg
            # Giả định let_distribution tương quan với vị trí tương đối dọc theo đường đi
            rbe_distribution = variable_rbe_model(dose_distribution, let_distribution, alpha_beta_ratio)
        
        elif model.lower() == 'repair':
            # Mô hình xem xét khả năng sửa chữa DNA
            rbe_distribution = repair_model(let_distribution, alpha_beta_ratio, dose_per_fraction)
        
        else:
            logger.warning(f"Unknown RBE model: {model} for {particle_type}. Using constant RBE = 1.1")
            rbe_distribution = np.ones_like(dose_distribution) * 1.1
    
    elif particle_type.lower() == 'carbon':
        # RBE cho Carbon ion
        if model.lower() == 'constant':
            # Giá trị RBE hằng số cho ion carbon (đơn giản hóa)
            rbe_distribution = np.ones_like(dose_distribution) * 3.0
        
        elif model.lower() == 'lq':
            # Mô hình Linear-Quadratic cho ion carbon
            rbe_distribution = carbon_lq_model(let_distribution, dose_distribution, alpha_beta_ratio, dose_per_fraction)
        
        elif model.lower() == 'lem':
            # Local Effect Model (phiên bản đơn giản)
            rbe_distribution = local_effect_model(let_distribution, dose_distribution, alpha_beta_ratio)
        
        elif model.lower() == 'mki':
            # Microdosimetric Kinetic Model
            rbe_distribution = mki_model(let_distribution, alpha_beta_ratio, dose_per_fraction)
        
        else:
            logger.warning(f"Unknown RBE model: {model} for {particle_type}. Using constant RBE = 3.0")
            rbe_distribution = np.ones_like(dose_distribution) * 3.0
    
    elif particle_type.lower() == 'helium':
        # RBE cho Helium ion
        if model.lower() == 'constant':
            rbe_distribution = np.ones_like(dose_distribution) * 2.0
        else:
            # Mô hình đơn giản cho Helium: trung gian giữa proton và carbon
            rbe_proton = mcnamara_model(let_distribution, alpha_beta_ratio)
            rbe_carbon = local_effect_model(let_distribution, dose_distribution, alpha_beta_ratio)
            rbe_distribution = (rbe_proton + rbe_carbon) / 2.0
    
    elif particle_type.lower() == 'neutron':
        # RBE cho neutron
        if model.lower() == 'constant':
            # Giá trị RBE hằng số cho neutron
            rbe_distribution = np.ones_like(dose_distribution) * 2.5
        
        elif model.lower() == 'energy_dependent':
            # RBE phụ thuộc vào năng lượng neutron
            # Giả định let_distribution tương quan với năng lượng neutron
            let_max = np.max(let_distribution)
            with np.errstate(divide='ignore', invalid='ignore'):
                relative_let = np.divide(let_distribution, let_max, 
                                      out=np.zeros_like(let_distribution), 
                                      where=let_max > 0)
                
                # RBE cao hơn cho neutron năng lượng thấp
                rbe_distribution = 1.0 + 4.0 * (1.0 - relative_let)
                rbe_distribution = np.clip(rbe_distribution, 1.0, 5.0)
        
        else:
            logger.warning(f"Unknown RBE model: {model} for {particle_type}. Using constant RBE = 2.5")
            rbe_distribution = np.ones_like(dose_distribution) * 2.5
    
    # Giới hạn giá trị RBE hợp lý
    rbe_distribution = np.clip(rbe_distribution, 1.0, 10.0)
    
    # Trả về 1.0 cho vùng không có liều
    rbe_distribution[dose_distribution <= 0.001 * np.max(dose_distribution)] = 1.0
    
    return rbe_distribution

def mcnamara_model(let_distribution: np.ndarray, alpha_beta_ratio: Union[float, np.ndarray]) -> np.ndarray:
    """
    Tính RBE theo mô hình McNamara cho proton.
    
    Parameters:
        let_distribution (np.ndarray): Phân bố LET (keV/μm)
        alpha_beta_ratio (float or np.ndarray): Tỉ lệ alpha/beta của mô (Gy)
    
    Returns:
        np.ndarray: Phân bố RBE
    """
    # Chuyển đổi alpha_beta_ratio thành mảng nếu cần
    if isinstance(alpha_beta_ratio, (int, float)):
        alpha_beta = np.ones_like(let_distribution) * alpha_beta_ratio
    else:
        alpha_beta = alpha_beta_ratio
    
    # Tính RBE theo công thức: RBE = 1 + (0.843 * LET / (α/β))
    with np.errstate(divide='ignore', invalid='ignore'):
        rbe = 1.0 + (0.843 * let_distribution / alpha_beta)
        rbe = np.nan_to_num(rbe, nan=1.0, posinf=10.0, neginf=1.0)
    
    return rbe

def wedenberg_model(let_distribution: np.ndarray, alpha_beta_ratio: Union[float, np.ndarray]) -> np.ndarray:
    """
    Tính RBE theo mô hình Wedenberg cho proton.
    
    Parameters:
        let_distribution (np.ndarray): Phân bố LET (keV/μm)
        alpha_beta_ratio (float or np.ndarray): Tỉ lệ alpha/beta của mô (Gy)
    
    Returns:
        np.ndarray: Phân bố RBE
    """
    # Chuyển đổi alpha_beta_ratio thành mảng nếu cần
    if isinstance(alpha_beta_ratio, (int, float)):
        alpha_beta = np.ones_like(let_distribution) * alpha_beta_ratio
    else:
        alpha_beta = alpha_beta_ratio
    
    # Tính RBE theo công thức: RBE = 1 + (0.434 * LET / (α/β))
    with np.errstate(divide='ignore', invalid='ignore'):
        rbe = 1.0 + (0.434 * let_distribution / alpha_beta)
        rbe = np.nan_to_num(rbe, nan=1.0, posinf=10.0, neginf=1.0)
    
    return rbe

def carlson_model(let_distribution: np.ndarray, 
                 alpha_beta_ratio: Union[float, np.ndarray],
                 tissue_type: Union[str, np.ndarray] = 'generic') -> np.ndarray:
    """
    Tính RBE theo mô hình Carlson cho proton, với xem xét loại mô.
    
    Parameters:
        let_distribution (np.ndarray): Phân bố LET (keV/μm)
        alpha_beta_ratio (float or np.ndarray): Tỉ lệ alpha/beta của mô (Gy)
        tissue_type (str or np.ndarray): Loại mô ('tumor', 'normal', 'generic')
    
    Returns:
        np.ndarray: Phân bố RBE
    """
    # Khởi tạo mảng RBE
    rbe = np.ones_like(let_distribution)
    
    # Chuyển đổi alpha_beta_ratio thành mảng nếu cần
    if isinstance(alpha_beta_ratio, (int, float)):
        alpha_beta = np.ones_like(let_distribution) * alpha_beta_ratio
    else:
        alpha_beta = alpha_beta_ratio
    
    # Xử lý loại mô dạng chuỗi
    if isinstance(tissue_type, str):
        if tissue_type.lower() == 'tumor':
            # Khối u có tỉ lệ chết tế bào cao
            k_value = 0.55
        elif tissue_type.lower() == 'normal':
            # Mô lành cần bảo vệ
            k_value = 0.95
        else:
            # Mặc định
            k_value = 0.75
            
        # Tính RBE
        with np.errstate(divide='ignore', invalid='ignore'):
            rbe = 1.0 + (k_value * let_distribution / alpha_beta)
            rbe = np.nan_to_num(rbe, nan=1.0, posinf=10.0, neginf=1.0)
    
    # Xử lý loại mô dạng mảng (với phân vùng mô khác nhau)
    else:
        # Mặt nạ cho từng loại mô
        tumor_mask = (tissue_type == 'tumor')
        normal_mask = (tissue_type == 'normal')
        generic_mask = ~(tumor_mask | normal_mask)
        
        # Áp dụng công thức cho từng vùng
        with np.errstate(divide='ignore', invalid='ignore'):
            if np.any(tumor_mask):
                rbe[tumor_mask] = 1.0 + (0.55 * let_distribution[tumor_mask] / alpha_beta[tumor_mask])
            
            if np.any(normal_mask):
                rbe[normal_mask] = 1.0 + (0.95 * let_distribution[normal_mask] / alpha_beta[normal_mask])
            
            if np.any(generic_mask):
                rbe[generic_mask] = 1.0 + (0.75 * let_distribution[generic_mask] / alpha_beta[generic_mask])
            
            # Đảm bảo giá trị hợp lệ
            rbe = np.nan_to_num(rbe, nan=1.0, posinf=10.0, neginf=1.0)
    
    return rbe

def variable_rbe_model(dose_distribution: np.ndarray,
                      let_distribution: np.ndarray,
                      alpha_beta_ratio: Union[float, np.ndarray]) -> np.ndarray:
    """
    Mô hình RBE biến đổi theo vị trí dọc theo đường đi của hạt proton.
    
    Parameters:
        dose_distribution (np.ndarray): Phân bố liều
        let_distribution (np.ndarray): Phân bố LET
        alpha_beta_ratio (float or np.ndarray): Tỉ lệ alpha/beta của mô
    
    Returns:
        np.ndarray: Phân bố RBE
    """
    # Chuẩn hóa LET
    let_max = np.max(let_distribution)
    with np.errstate(divide='ignore', invalid='ignore'):
        normalized_let = np.divide(let_distribution, let_max, 
                                 out=np.zeros_like(let_distribution), 
                                 where=let_max > 0)
    
    # Giả định normalized_let tương quan với vị trí tương đối dọc theo đường đi của hạt
    # RBE tăng từ 1.1 ở đầu vào đến 1.7 tại đỉnh Bragg
    rbe = 1.1 + 0.6 * normalized_let
    
    # Đảm bảo giá trị hợp lý
    rbe = np.clip(rbe, 1.0, 1.7)
    
    # Trả về 1.0 cho vùng không có liều
    rbe[dose_distribution <= 0.001 * np.max(dose_distribution)] = 1.0
    
    return rbe

def repair_model(let_distribution: np.ndarray,
                alpha_beta_ratio: Union[float, np.ndarray],
                dose_per_fraction: Union[float, np.ndarray]) -> np.ndarray:
    """
    Mô hình RBE có xem xét khả năng sửa chữa DNA phụ thuộc vào liều.
    
    Parameters:
        let_distribution (np.ndarray): Phân bố LET
        alpha_beta_ratio (float or np.ndarray): Tỉ lệ alpha/beta của mô
        dose_per_fraction (float or np.ndarray): Liều trên mỗi phân đoạn
    
    Returns:
        np.ndarray: Phân bố RBE
    """
    # Chuyển đổi tham số thành mảng nếu cần
    if isinstance(alpha_beta_ratio, (int, float)):
        alpha_beta = np.ones_like(let_distribution) * alpha_beta_ratio
    else:
        alpha_beta = alpha_beta_ratio
        
    if isinstance(dose_per_fraction, (int, float)):
        d = np.ones_like(let_distribution) * dose_per_fraction
    else:
        d = dose_per_fraction
    
    # Tính RBE dựa trên LET, tỉ lệ alpha/beta và liều phân đoạn
    # RBE = (alpha_x + beta_x * d) / (alpha_p * (1 + k*LET) + beta_p * d)
    # Đơn giản hóa: alpha_p = alpha_x * (1 + k*LET), beta_p = beta_x
    # => RBE = (alpha_x + beta_x * d) / (alpha_x * (1 + k*LET) + beta_x * d)
    
    # Tham số
    k = 0.05  # keV/μm^-1
    
    # Tính tỉ lệ phụ thuộc vào khả năng sửa chữa
    with np.errstate(divide='ignore', invalid='ignore'):
        alpha_x = alpha_beta / (1.0 + d)
        beta_x = alpha_x / alpha_beta
        
        numerator = alpha_x + beta_x * d
        denominator = alpha_x * (1.0 + k * let_distribution) + beta_x * d
        
        rbe = np.divide(numerator, denominator, 
                      out=np.ones_like(numerator), 
                      where=denominator > 0)
        
        # Đảm bảo giá trị hợp lý
        rbe = np.nan_to_num(rbe, nan=1.0, posinf=5.0, neginf=1.0)
    
    return rbe

def carbon_lq_model(let_distribution: np.ndarray,
                   dose_distribution: np.ndarray,
                   alpha_beta_ratio: Union[float, np.ndarray],
                   dose_per_fraction: Union[float, np.ndarray]) -> np.ndarray:
    """
    Mô hình Linear-Quadratic cho ion carbon.
    
    Parameters:
        let_distribution (np.ndarray): Phân bố LET
        dose_distribution (np.ndarray): Phân bố liều
        alpha_beta_ratio (float or np.ndarray): Tỉ lệ alpha/beta của mô
        dose_per_fraction (float or np.ndarray): Liều trên mỗi phân đoạn
    
    Returns:
        np.ndarray: Phân bố RBE
    """
    # Chuyển đổi tham số thành mảng nếu cần
    if isinstance(alpha_beta_ratio, (int, float)):
        alpha_beta = np.ones_like(let_distribution) * alpha_beta_ratio
    else:
        alpha_beta = alpha_beta_ratio
        
    if isinstance(dose_per_fraction, (int, float)):
        d = np.ones_like(let_distribution) * dose_per_fraction
    else:
        d = dose_per_fraction
    
    # Tính tham số alpha và beta cho photon
    alpha_x = alpha_beta / (1.0 + d)
    beta_x = alpha_x / alpha_beta
    
    # Tham số cho ion carbon
    # alpha tăng mạnh với LET, beta tăng nhẹ với LET
    alpha_c = alpha_x * (1.0 + 0.15 * let_distribution)
    beta_c = beta_x * (1.0 + 0.03 * let_distribution)
    
    # Tính RBE
    with np.errstate(divide='ignore', invalid='ignore'):
        # Hiệu ứng sinh học của photon
        e_x = alpha_x * d + beta_x * d * d
        
        # Hiệu ứng sinh học của ion carbon
        e_c = alpha_c * d + beta_c * d * d
        
        # RBE là tỉ lệ giữa liều photon và liều ion carbon tạo ra cùng hiệu ứng
        # E_x(d_x) = E_c(d_c) => d_x/d_c = E_c(d_c)/E_x(d_x) = RBE
        rbe = np.divide(e_c, e_x, out=np.ones_like(e_c), where=e_x > 0)
        
        # Đảm bảo giá trị hợp lý
        rbe = np.nan_to_num(rbe, nan=1.0, posinf=10.0, neginf=1.0)
    
    return rbe

def local_effect_model(let_distribution: np.ndarray,
                      dose_distribution: np.ndarray,
                      alpha_beta_ratio: Union[float, np.ndarray]) -> np.ndarray:
    """
    Local Effect Model (LEM) cho ion carbon (phiên bản đơn giản).
    
    Parameters:
        let_distribution (np.ndarray): Phân bố LET
        dose_distribution (np.ndarray): Phân bố liều
        alpha_beta_ratio (float or np.ndarray): Tỉ lệ alpha/beta của mô
    
    Returns:
        np.ndarray: Phân bố RBE
    """
    # Chuyển đổi tham số thành mảng nếu cần
    if isinstance(alpha_beta_ratio, (int, float)):
        alpha_beta = np.ones_like(let_distribution) * alpha_beta_ratio
    else:
        alpha_beta = alpha_beta_ratio
    
    # Tham số cho LEM
    D_t = 30.0  # Gy, liều ngưỡng
    r_nuc = 5.0  # μm, bán kính nhân tế bào
    
    # Tính rbe dựa trên LEM
    with np.errstate(divide='ignore', invalid='ignore'):
        # Tính tham số alpha và beta cho photon
        alpha_x = alpha_beta / (1.0 + dose_distribution)
        beta_x = alpha_x / alpha_beta
        
        # Hiệu ứng sinh học của photon
        e_x = alpha_x * dose_distribution + beta_x * dose_distribution * dose_distribution
        
        # Tính liều lethal events cho ion carbon
        # Đơn giản hóa từ công thức LEM đầy đủ
        let_factor = np.sqrt(let_distribution / 50.0)  # Chuẩn hóa với LET = 50 keV/μm
        e_c = e_x * (1.0 + 4.0 * let_factor)
        
        # RBE
        rbe = np.divide(e_c, e_x, out=np.ones_like(e_c), where=e_x > 0)
        
        # Đảm bảo giá trị hợp lý
        rbe = np.nan_to_num(rbe, nan=1.0, posinf=10.0, neginf=1.0)
    
    return rbe

def mki_model(let_distribution: np.ndarray,
             alpha_beta_ratio: Union[float, np.ndarray],
             dose_per_fraction: Union[float, np.ndarray]) -> np.ndarray:
    """
    Microdosimetric Kinetic Model cho ion carbon.
    
    Parameters:
        let_distribution (np.ndarray): Phân bố LET
        alpha_beta_ratio (float or np.ndarray): Tỉ lệ alpha/beta của mô
        dose_per_fraction (float or np.ndarray): Liều trên mỗi phân đoạn
    
    Returns:
        np.ndarray: Phân bố RBE
    """
    # Chuyển đổi tham số thành mảng nếu cần
    if isinstance(alpha_beta_ratio, (int, float)):
        alpha_beta = np.ones_like(let_distribution) * alpha_beta_ratio
    else:
        alpha_beta = alpha_beta_ratio
        
    if isinstance(dose_per_fraction, (int, float)):
        d = np.ones_like(let_distribution) * dose_per_fraction
    else:
        d = dose_per_fraction
    
    # Tham số cho MKI model
    y_0 = 150.0  # keV/μm
    
    # Tính rbe dựa trên MKI
    with np.errstate(divide='ignore', invalid='ignore'):
        # Tính tham số alpha và beta cho photon
        alpha_x = alpha_beta / (1.0 + d)
        beta_x = alpha_x / alpha_beta
        
        # Tính y* (lineal energy) từ LET
        y_star = 0.8 * let_distribution
        
        # Tính hệ số saturation
        sat_factor = 1.0 - np.exp(-y_star * y_star / (y_0 * y_0))
        
        # Tính tham số alpha và beta cho ion carbon
        alpha_c = alpha_x * (1.0 + (2.0 / alpha_beta) * sat_factor * y_star)
        beta_c = beta_x
        
        # Tính RBE
        e_x = alpha_x * d + beta_x * d * d
        e_c = alpha_c * d + beta_c * d * d
        
        rbe = np.divide(e_c, e_x, out=np.ones_like(e_c), where=e_x > 0)
        
        # Đảm bảo giá trị hợp lý
        rbe = np.nan_to_num(rbe, nan=1.0, posinf=10.0, neginf=1.0)
    
    return rbe

def calculate_rbe_weighted_dose(dose_distribution: np.ndarray,
                              rbe_distribution: np.ndarray) -> np.ndarray:
    """
    Tính toán liều trọng số RBE (RBE-weighted dose).
    
    Parameters:
        dose_distribution (np.ndarray): Phân bố liều vật lý (Gy)
        rbe_distribution (np.ndarray): Phân bố RBE
    
    Returns:
        np.ndarray: Phân bố liều trọng số RBE (Gy(RBE))
    """
    return dose_distribution * rbe_distribution
