"""
Module tính toán LET (Linear Energy Transfer) trong xạ trị.

LET mô tả lượng năng lượng trung bình mà một hạt tích điện truyền cho vật chất 
trên mỗi đơn vị chiều dài của quỹ đạo di chuyển. LET là một tham số quan trọng 
trong xạ trị proton, ion và BNCT, ảnh hưởng đến hiệu quả sinh học của liều xạ.
"""

import numpy as np
import logging
from typing import Dict, Tuple, Optional, Union, List, Any

logger = logging.getLogger(__name__)

def calculate_let(dose_distribution: np.ndarray,
                 fluence_distribution: Optional[np.ndarray] = None,
                 particle_type: str = "proton",
                 energy: Union[float, np.ndarray] = 150.0,
                 material_density: Union[float, np.ndarray] = 1.0,
                 calculation_method: str = "dose_weighted") -> np.ndarray:
    """
    Tính toán phân bố LET (Linear Energy Transfer).
    
    Parameters:
        dose_distribution (np.ndarray): Phân bố liều (Gy)
        fluence_distribution (np.ndarray, optional): Phân bố fluence (hạt/cm²)
        particle_type (str): Loại hạt ("proton", "carbon", "helium", "electron")
        energy (float hoặc np.ndarray): Năng lượng hạt (MeV) hoặc phổ năng lượng
        material_density (float hoặc np.ndarray): Mật độ vật liệu (g/cm³)
        calculation_method (str): Phương pháp tính ("dose_weighted", "track_weighted")
    
    Returns:
        np.ndarray: Phân bố LET (keV/μm)
    """
    logger.info(f"Calculating LET for {particle_type} with method {calculation_method}")
    
    # Kiểm tra loại hạt
    supported_particles = ["proton", "carbon", "helium", "electron"]
    if particle_type.lower() not in supported_particles:
        logger.warning(f"Unsupported particle type: {particle_type}. Defaulting to proton.")
        particle_type = "proton"
    
    # Khởi tạo phân bố LET
    let_distribution = np.zeros_like(dose_distribution)
    
    # Xử lý trường hợp năng lượng đơn
    if isinstance(energy, (int, float)):
        let_distribution = _calculate_let_for_energy(
            dose_distribution,
            fluence_distribution,
            particle_type,
            energy,
            material_density,
            calculation_method
        )
    # Xử lý trường hợp phổ năng lượng
    else:
        if isinstance(energy, np.ndarray) and energy.ndim == 1:
            # Giả định phổ năng lượng dạng [energy1, energy2, ...] với trọng số đều
            weights = np.ones_like(energy) / len(energy)
            let_distribution = np.zeros_like(dose_distribution)
            
            for e, w in zip(energy, weights):
                let_e = _calculate_let_for_energy(
                    dose_distribution,
                    fluence_distribution,
                    particle_type,
                    e,
                    material_density,
                    calculation_method
                )
                let_distribution += w * let_e
                
        elif isinstance(energy, dict) and 'values' in energy and 'weights' in energy:
            # Phổ năng lượng dạng {'values': [e1, e2, ...], 'weights': [w1, w2, ...]}
            energy_values = energy['values']
            energy_weights = energy['weights']
            let_distribution = np.zeros_like(dose_distribution)
            
            for e, w in zip(energy_values, energy_weights):
                let_e = _calculate_let_for_energy(
                    dose_distribution,
                    fluence_distribution,
                    particle_type,
                    e,
                    material_density,
                    calculation_method
                )
                let_distribution += w * let_e
                
        else:
            logger.warning("Energy spectrum format not recognized. Using nominal energy of 150 MeV.")
            let_distribution = _calculate_let_for_energy(
                dose_distribution,
                fluence_distribution,
                particle_type,
                150.0,
                material_density,
                calculation_method
            )
    
    # Gán LET = 0 cho vùng không có liều
    dose_threshold = 0.001 * np.max(dose_distribution)
    let_distribution[dose_distribution < dose_threshold] = 0.0
    
    return let_distribution

def _calculate_let_for_energy(dose_distribution: np.ndarray,
                             fluence_distribution: Optional[np.ndarray],
                             particle_type: str,
                             energy: float,
                             material_density: Union[float, np.ndarray],
                             calculation_method: str) -> np.ndarray:
    """
    Tính LET cho một năng lượng hạt cụ thể.
    
    Parameters:
        dose_distribution (np.ndarray): Phân bố liều
        fluence_distribution (np.ndarray, optional): Phân bố fluence
        particle_type (str): Loại hạt
        energy (float): Năng lượng hạt (MeV)
        material_density (float hoặc np.ndarray): Mật độ vật liệu (g/cm³)
        calculation_method (str): Phương pháp tính LET
    
    Returns:
        np.ndarray: Phân bố LET cho năng lượng cụ thể
    """
    # Khởi tạo mảng LET
    let_distribution = np.zeros_like(dose_distribution)
    
    # Chuyển đổi material_density thành mảng nếu cần
    if isinstance(material_density, (int, float)):
        density = np.ones_like(dose_distribution) * material_density
    else:
        density = material_density
    
    # Tính LET theo loại hạt
    if particle_type.lower() == "proton":
        # Ước tính LET cho proton dựa trên công thức Bethe-Bloch đơn giản hóa
        # LET ∝ (Z²/v²) × f(v)
        # với v là vận tốc tương đối của hạt, Z là điện tích
        
        # Tính range (biến thiên theo E và mật độ)
        range_cm = _calculate_range_for_proton(energy, 1.0)  # range trong nước
        
        # Tính LET đầu vào (LET ở đầu vào của chùm tia)
        let_entrance = _calculate_entrance_let_for_proton(energy)
        
        # Tính phân bố tương đối để ước tính LET
        # Sử dụng đạo hàm của liều để ước tính vị trí tương đối dọc theo đường đi
        if calculation_method.lower() == "dose_weighted":
            # Phương pháp dose-weighted LET
            
            # Ước tính vị trí tương đối dọc theo đường đi dựa trên dose
            # Đây là phương pháp đơn giản, trong thực tế cần mô phỏng Monte Carlo chi tiết
            
            # Chuẩn hóa dose
            max_dose = np.max(dose_distribution)
            if max_dose > 0:
                normalized_dose = dose_distribution / max_dose
            else:
                normalized_dose = np.zeros_like(dose_distribution)
            
            # Tính gradient để ước tính vị trí gần đỉnh Bragg
            # (trong thực tế, điều này phức tạp hơn nhiều)
            rel_position = normalized_dose
            
            # Tính LET: tăng dần theo vị trí tương đối, cao nhất tại đỉnh Bragg
            # LET tăng khoảng 2-3 lần từ đầu vào đến đỉnh Bragg
            let_distribution = let_entrance * (1.0 + 2.0 * rel_position)
            
            # Điều chỉnh theo mật độ vật liệu
            let_distribution = let_distribution * (density / 1.0)
            
        else:  # "track_weighted"
            # Phương pháp track-weighted LET
            if fluence_distribution is not None:
                # Sử dụng thông tin fluence nếu có
                max_fluence = np.max(fluence_distribution)
                if max_fluence > 0:
                    normalized_fluence = fluence_distribution / max_fluence
                else:
                    normalized_fluence = np.zeros_like(fluence_distribution)
                
                rel_position = 1.0 - normalized_fluence  # fluence giảm khi đi sâu vào vật chất
                let_distribution = let_entrance * (1.0 + 2.0 * rel_position)
                let_distribution = let_distribution * (density / 1.0)
            else:
                # Nếu không có thông tin fluence, sử dụng phương pháp tương tự dose-weighted
                logger.warning("Fluence distribution not provided for track-weighted LET calculation. Using dose-based approximation.")
                max_dose = np.max(dose_distribution)
                if max_dose > 0:
                    normalized_dose = dose_distribution / max_dose
                else:
                    normalized_dose = np.zeros_like(dose_distribution)
                
                rel_position = normalized_dose
                let_distribution = let_entrance * (1.0 + 2.0 * rel_position)
                let_distribution = let_distribution * (density / 1.0)
    
    elif particle_type.lower() == "carbon":
        # LET cho carbon ion cao hơn nhiều so với proton cùng range
        # LET ∝ Z²
        let_entrance = _calculate_entrance_let_for_carbon(energy)
        
        max_dose = np.max(dose_distribution)
        if max_dose > 0:
            normalized_dose = dose_distribution / max_dose
        else:
            normalized_dose = np.zeros_like(dose_distribution)
        
        rel_position = normalized_dose
        
        # Carbon có tăng LET nhanh hơn (4-6 lần từ đầu vào đến đỉnh)
        let_distribution = let_entrance * (1.0 + 5.0 * rel_position)
        let_distribution = let_distribution * (density / 1.0)
    
    elif particle_type.lower() == "helium":
        # LET cho helium ion, trung gian giữa proton và carbon
        let_entrance = _calculate_entrance_let_for_helium(energy)
        
        max_dose = np.max(dose_distribution)
        if max_dose > 0:
            normalized_dose = dose_distribution / max_dose
        else:
            normalized_dose = np.zeros_like(dose_distribution)
        
        rel_position = normalized_dose
        
        # Helium có tăng LET khá nhanh (3-4 lần từ đầu vào đến đỉnh)
        let_distribution = let_entrance * (1.0 + 3.0 * rel_position)
        let_distribution = let_distribution * (density / 1.0)
    
    elif particle_type.lower() == "electron":
        # LET cho electron thấp hơn nhiều
        let_entrance = _calculate_entrance_let_for_electron(energy)
        
        # Electron có LET khá đồng đều dọc theo đường đi
        let_distribution = np.ones_like(dose_distribution) * let_entrance
        let_distribution = let_distribution * (density / 1.0)
    
    return let_distribution

def _calculate_range_for_proton(energy: float, density: float = 1.0) -> float:
    """
    Tính range (cm) cho proton dựa trên năng lượng (MeV) và mật độ vật liệu.
    
    Parameters:
        energy (float): Năng lượng proton (MeV)
        density (float): Mật độ vật liệu (g/cm³)
    
    Returns:
        float: Range (cm) trong vật liệu
    """
    # Công thức xấp xỉ: Range (cm) ≈ 0.0022 × E^1.77
    # với E là năng lượng tính bằng MeV, cho nước (density = 1 g/cm³)
    range_water = 0.0022 * (energy ** 1.77)
    
    # Hiệu chỉnh theo mật độ
    range_material = range_water / density
    
    return range_material

def _calculate_entrance_let_for_proton(energy: float) -> float:
    """
    Tính LET đầu vào (keV/μm) cho proton dựa trên năng lượng (MeV).
    
    Parameters:
        energy (float): Năng lượng proton (MeV)
    
    Returns:
        float: LET (keV/μm) tại đầu vào
    """
    # Công thức xấp xỉ dựa trên dữ liệu thực nghiệm
    # LET (keV/μm) ≈ 0.5 + 60 / E^1.4 cho E > 10 MeV
    if energy > 10.0:
        let = 0.5 + 60.0 / (energy ** 1.4)
    else:
        # Với năng lượng thấp, LET cao hơn
        let = 100.0 / np.sqrt(energy)
    
    return let

def _calculate_entrance_let_for_carbon(energy: float) -> float:
    """
    Tính LET đầu vào (keV/μm) cho carbon ion dựa trên năng lượng (MeV).
    
    Parameters:
        energy (float): Năng lượng carbon/nucleon (MeV/u)
    
    Returns:
        float: LET (keV/μm) tại đầu vào
    """
    # Carbon có LET cao hơn proton khoảng 36 lần (Z² = 36) cùng vận tốc
    # Ở đây energy là năng lượng trên nucleon (MeV/u)
    proton_let = _calculate_entrance_let_for_proton(energy)
    carbon_let = proton_let * 36.0
    
    return carbon_let

def _calculate_entrance_let_for_helium(energy: float) -> float:
    """
    Tính LET đầu vào (keV/μm) cho helium ion dựa trên năng lượng (MeV).
    
    Parameters:
        energy (float): Năng lượng helium/nucleon (MeV/u)
    
    Returns:
        float: LET (keV/μm) tại đầu vào
    """
    # Helium có LET cao hơn proton khoảng 4 lần (Z² = 4) cùng vận tốc
    proton_let = _calculate_entrance_let_for_proton(energy)
    helium_let = proton_let * 4.0
    
    return helium_let

def _calculate_entrance_let_for_electron(energy: float) -> float:
    """
    Tính LET đầu vào (keV/μm) cho electron dựa trên năng lượng (MeV).
    
    Parameters:
        energy (float): Năng lượng electron (MeV)
    
    Returns:
        float: LET (keV/μm) tại đầu vào
    """
    # Electron có LET thấp hơn nhiều so với proton
    # Công thức xấp xỉ: LET ≈ 0.2 keV/μm cho E từ 1-30 MeV
    let = 0.2
    
    return let

def calculate_rbe_from_let(let_distribution: np.ndarray, 
                         alpha_beta_ratio: float = 10.0,
                         tissue_type: str = 'generic',
                         model: str = 'mcnamara') -> np.ndarray:
    """
    Tính toán RBE (Relative Biological Effectiveness) từ LET.
    
    Parameters:
        let_distribution (np.ndarray): Phân bố LET (keV/μm)
        alpha_beta_ratio (float): Tỉ lệ alpha/beta của mô (Gy)
        tissue_type (str): Loại mô ('tumor', 'normal', 'generic')
        model (str): Mô hình tính RBE ('mcnamara', 'wedenberg', 'carlson', 'custom')
    
    Returns:
        np.ndarray: Phân bố RBE
    """
    logger.info(f"Calculating RBE from LET using {model} model")
    
    # Khởi tạo phân bố RBE
    rbe_distribution = np.ones_like(let_distribution)
    
    # Tính RBE theo mô hình McNamara
    if model.lower() == 'mcnamara':
        # McNamara model: RBE = 1 + (0.843 * LET / (α/β))
        rbe_distribution = 1.0 + (0.843 * let_distribution / alpha_beta_ratio)
    
    # Mô hình Wedenberg
    elif model.lower() == 'wedenberg':
        # Wedenberg model: RBE = 1 + (0.434 * LET / (α/β))
        rbe_distribution = 1.0 + (0.434 * let_distribution / alpha_beta_ratio)
    
    # Mô hình Carlson
    elif model.lower() == 'carlson':
        # Carlson model with tissue type consideration
        if tissue_type.lower() == 'tumor':
            # Khối u có tỉ lệ chết tế bào cao
            rbe_distribution = 1.0 + (0.55 * let_distribution / alpha_beta_ratio)
        elif tissue_type.lower() == 'normal':
            # Mô lành cần bảo vệ
            rbe_distribution = 1.0 + (0.95 * let_distribution / alpha_beta_ratio)
        else:
            # Mặc định
            rbe_distribution = 1.0 + (0.75 * let_distribution / alpha_beta_ratio)
    
    # Giới hạn giá trị RBE hợp lý (thường RBE < 10 for clinical scenarios)
    rbe_distribution = np.clip(rbe_distribution, 1.0, 10.0)
    
    return rbe_distribution
