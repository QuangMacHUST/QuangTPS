"""
Module mô hình hóa các tương tác bức xạ với vật chất.

Module này mô hình hóa các tương tác vật lý cơ bản của bức xạ ion hóa
(photon, electron, neutron, proton, ion nặng) với vật chất, bao gồm
hiệu ứng quang điện, hiệu ứng Compton, sinh đôi, v.v.
"""

import numpy as np
import logging
from typing import Dict, Any, List, Tuple, Optional
from enum import Enum

logger = logging.getLogger(__name__)

class InteractionType(str, Enum):
    """Loại tương tác bức xạ-vật chất."""
    PHOTOELECTRIC = "photoelectric"
    COMPTON = "compton"
    PAIR_PRODUCTION = "pair_production"
    COHERENT = "coherent"
    PHOTONUCLEAR = "photonuclear"
    IONIZATION = "ionization"
    BREMSSTRAHLUNG = "bremsstrahlung"
    NUCLEAR_ELASTIC = "nuclear_elastic"
    NUCLEAR_INELASTIC = "nuclear_inelastic"
    TOTAL = "total"  # Tổng xác suất tương tác

def get_interaction_probability(energy: float, 
                              material: str, 
                              interaction_type: InteractionType) -> float:
    """
    Tính xác suất xảy ra tương tác cụ thể cho một năng lượng và vật liệu.
    
    Parameters
    ----------
    energy : float
        Năng lượng bức xạ (MeV)
    material : str
        Vật liệu ("water", "bone", "lung", "muscle", "fat", v.v.)
    interaction_type : InteractionType
        Loại tương tác cần tính xác suất
        
    Returns
    -------
    float
        Xác suất tương tác (cm²/g)
    """
    # Các hệ số suy giảm khối cho nước (cm²/g)
    if material.lower() == "water":
        if interaction_type == InteractionType.PHOTOELECTRIC:
            # Hiệu ứng quang điện: σ_pe ∝ Z^4 / E^3.5
            return 0.0022 * (energy ** -3.5)
        
        elif interaction_type == InteractionType.COMPTON:
            # Hiệu ứng Compton: Klein-Nishina (gần đúng)
            # Suy giảm khi E tăng, nhưng chậm hơn hiệu ứng quang điện
            if energy < 0.1:
                return 0.1494
            elif energy < 1.0:
                return 0.1494 * (energy ** -0.5)
            else:
                return 0.1494 * (energy ** -0.3)
        
        elif interaction_type == InteractionType.PAIR_PRODUCTION:
            # Sinh đôi: σ_pp ∝ ln(2E)
            # Chỉ xảy ra khi E > 1.022 MeV
            if energy <= 1.022:
                return 0.0
            else:
                return 0.0028 * np.log(2 * energy)
        
        elif interaction_type == InteractionType.COHERENT:
            # Tán xạ kết hợp (Rayleigh): giảm nhanh với năng lượng tăng
            return 0.0025 * (energy ** -2)
            
        # Tổng xác suất tương tác
        elif interaction_type == InteractionType.TOTAL:
            return (get_interaction_probability(energy, material, InteractionType.PHOTOELECTRIC) +
                   get_interaction_probability(energy, material, InteractionType.COMPTON) +
                   get_interaction_probability(energy, material, InteractionType.PAIR_PRODUCTION) +
                   get_interaction_probability(energy, material, InteractionType.COHERENT))
    
    # TODO: Bổ sung dữ liệu cho các vật liệu khác (xương, phổi, mô, v.v.)
    # Tạm thời sử dụng hệ số điều chỉnh dựa trên mật độ điện tử tương đối
    else:
        relative_density = {
            "bone": 1.85,
            "lung": 0.26,
            "muscle": 1.04,
            "fat": 0.92,
            "air": 0.001
        }.get(material.lower(), 1.0)
        
        # Với hiệu ứng quang điện, xác suất tỷ lệ với Z^4
        if interaction_type == InteractionType.PHOTOELECTRIC:
            z_factor = {
                "bone": 2.8,  # Z hiệu dụng cao hơn nước
                "lung": 0.9,
                "muscle": 1.02,
                "fat": 0.95,
                "air": 0.85
            }.get(material.lower(), 1.0)
            
            # Hiệu ứng quang điện phụ thuộc mạnh vào Z
            z_factor = z_factor ** 4
            return get_interaction_probability(energy, "water", interaction_type) * relative_density * z_factor
        
        # Với các tương tác khác, chủ yếu phụ thuộc vào mật độ điện tử
        return get_interaction_probability(energy, "water", interaction_type) * relative_density

def calculate_interaction_depth(energy: float, 
                              material: str, 
                              num_samples: int = 1000) -> np.ndarray:
    """
    Tính độ sâu tương tác cho photon dựa trên hệ số suy giảm.
    
    Parameters
    ----------
    energy : float
        Năng lượng photon (MeV)
    material : str
        Vật liệu ("water", "bone", "lung", v.v.)
    num_samples : int, optional
        Số lượng mẫu cần tạo
        
    Returns
    -------
    np.ndarray
        Mảng các giá trị độ sâu tương tác (cm)
    """
    # Lấy hệ số suy giảm tuyến tính tổng (cm^-1)
    mu = get_interaction_probability(energy, material, InteractionType.TOTAL)
    
    # Mật độ vật liệu (g/cm³)
    density = {
        "water": 1.0,
        "bone": 1.85, 
        "lung": 0.26,
        "muscle": 1.04,
        "fat": 0.92,
        "air": 0.001
    }.get(material.lower(), 1.0)
    
    # Tính hệ số suy giảm tuyến tính
    mu_linear = mu * density  # cm^-1
    
    # Tạo các mẫu theo phân bố mũ
    random_samples = np.random.random(num_samples)
    depths = -np.log(random_samples) / mu_linear
    
    return depths

def calculate_energy_spectrum_after_attenuation(incident_spectrum: Dict[float, float],
                                              material: str,
                                              thickness: float) -> Dict[float, float]:
    """
    Tính phổ năng lượng sau khi đi qua vật liệu.
    
    Parameters
    ----------
    incident_spectrum : Dict[float, float]
        Phổ năng lượng ban đầu (năng lượng -> cường độ tương đối)
    material : str
        Vật liệu ("water", "bone", "lung", v.v.)
    thickness : float
        Độ dày vật liệu (cm)
        
    Returns
    -------
    Dict[float, float]
        Phổ năng lượng sau khi bị suy giảm
    """
    attenuated_spectrum = {}
    
    # Mật độ vật liệu (g/cm³)
    density = {
        "water": 1.0,
        "bone": 1.85,
        "lung": 0.26, 
        "muscle": 1.04,
        "fat": 0.92,
        "air": 0.001
    }.get(material.lower(), 1.0)
    
    # Tính phổ sau khi bị suy giảm
    for energy, intensity in incident_spectrum.items():
        # Lấy hệ số suy giảm cho năng lượng này
        mu = get_interaction_probability(energy, material, InteractionType.TOTAL)
        
        # Tính hệ số suy giảm tuyến tính
        mu_linear = mu * density  # cm^-1
        
        # Áp dụng định luật Beer-Lambert: I = I₀ × e^(-μ×d)
        attenuated_intensity = intensity * np.exp(-mu_linear * thickness)
        
        attenuated_spectrum[energy] = attenuated_intensity
    
    return attenuated_spectrum

def sample_scattering_angle(energy: float, interaction_type: InteractionType) -> float:
    """
    Lấy mẫu góc tán xạ dựa trên loại tương tác và năng lượng.
    
    Parameters
    ----------
    energy : float
        Năng lượng bức xạ (MeV)
    interaction_type : InteractionType
        Loại tương tác
        
    Returns
    -------
    float
        Góc tán xạ (radian)
    """
    if interaction_type == InteractionType.COMPTON:
        # Sử dụng công thức Klein-Nishina để lấy mẫu góc tán xạ Compton
        # Góc tán xạ phụ thuộc vào năng lượng
        
        # Tham số α = E / (mc²)
        alpha = energy / 0.511  # Năng lượng electron nghỉ = 0.511 MeV
        
        # Lấy mẫu theo phương pháp rejection sampling
        while True:
            # Lấy mẫu cos(θ) đều trong khoảng [-1, 1]
            cos_theta = 2 * np.random.random() - 1
            
            # Công thức Klein-Nishina (xác suất tương đối)
            p = (1 + cos_theta**2) / 2
            
            # Hiệu chỉnh theo năng lượng
            if alpha > 0:
                p *= (1 + (alpha**2 * (1 - cos_theta)**2) / ((1 + cos_theta**2) * (1 + alpha * (1 - cos_theta))))
            
            # Chấp nhận/từ chối mẫu
            if np.random.random() <= p:
                break
        
        # Chuyển đổi cos(θ) sang θ
        theta = np.arccos(cos_theta)
        return theta
    
    elif interaction_type == InteractionType.PHOTOELECTRIC:
        # Hiệu ứng quang điện không thay đổi hướng photon (electron thứ cấp đi theo hướng khác)
        return 0.0
    
    elif interaction_type == InteractionType.PAIR_PRODUCTION:
        # Sinh đôi tạo ra electron và positron, mỗi cái đi theo một phân bố phức tạp
        # Đơn giản hóa: lấy mẫu đều trong không gian
        return np.arccos(2 * np.random.random() - 1)
    
    elif interaction_type == InteractionType.COHERENT:
        # Tán xạ kết hợp (Rayleigh)
        # Đặc trưng bởi góc tán xạ nhỏ
        # Góc tán xạ phụ thuộc vào năng lượng: góc nhỏ hơn ở năng lượng cao
        
        # Lấy mẫu góc nhỏ, giảm khi năng lượng tăng
        max_angle = np.pi / 4 * np.exp(-energy)
        return max_angle * np.sqrt(np.random.random())
    
    # Mặc định: lấy mẫu đều trong không gian
    return np.arccos(2 * np.random.random() - 1)

def calculate_energy_after_compton(energy: float, scattering_angle: float) -> float:
    """
    Tính năng lượng photon sau tán xạ Compton.
    
    Parameters
    ----------
    energy : float
        Năng lượng photon ban đầu (MeV)
    scattering_angle : float
        Góc tán xạ (radian)
        
    Returns
    -------
    float
        Năng lượng photon sau tán xạ (MeV)
    """
    # Công thức Compton: E' = E / (1 + (E/mc²) * (1 - cos(θ)))
    # Với mc² = 0.511 MeV (năng lượng nghỉ của electron)
    
    cos_theta = np.cos(scattering_angle)
    denominator = 1 + (energy / 0.511) * (1 - cos_theta)
    
    return energy / denominator

def get_radiation_yield(energy: float, material: str, particle_type: str) -> float:
    """
    Tính tỷ lệ biến đổi năng lượng động năng thành bức xạ hãm.
    
    Parameters
    ----------
    energy : float
        Năng lượng hạt (MeV)
    material : str
        Vật liệu
    particle_type : str
        Loại hạt ("electron", "proton", "ion")
        
    Returns
    -------
    float
        Tỷ lệ năng lượng biến đổi thành bức xạ hãm
    """
    # Số nguyên tử Z hiệu dụng của vật liệu
    z_eff = {
        "water": 7.42,
        "bone": 13.8,
        "lung": 7.4,
        "muscle": 7.64,
        "fat": 6.46,
        "air": 7.64
    }.get(material.lower(), 7.42)
    
    # Tỷ lệ bức xạ hãm phụ thuộc vào loại hạt
    if particle_type.lower() == "electron":
        # Với electron, tỷ lệ bức xạ tỷ lệ với E×Z
        return z_eff * energy / (1600 + z_eff * energy)
    
    elif particle_type.lower() == "proton":
        # Với proton, tỷ lệ bức xạ rất thấp (tỷ lệ với 1/m²)
        return (z_eff * energy / 1836**2) / 100
    
    elif particle_type.lower() == "ion":
        # Với ion nặng, còn thấp hơn nữa
        return (z_eff * energy / 3672**2) / 1000
    
    # Mặc định: giả định là electron
    return z_eff * energy / (1600 + z_eff * energy)

def calculate_stopping_power(energy: float, material: str, particle_type: str) -> Dict[str, float]:
    """
    Tính stopping power (khả năng mất năng lượng) của hạt tích điện.
    
    Parameters
    ----------
    energy : float
        Năng lượng hạt (MeV)
    material : str
        Vật liệu
    particle_type : str
        Loại hạt ("electron", "proton", "carbon", v.v.)
        
    Returns
    -------
    Dict[str, float]
        Dictionary chứa electronic, nuclear và total stopping power (MeV·cm²/g)
    """
    # Mật độ điện tử tương đối so với nước
    electron_density_ratio = {
        "water": 1.0,
        "bone": 1.85,
        "lung": 0.26,
        "muscle": 1.04,
        "fat": 0.92,
        "air": 0.001
    }.get(material.lower(), 1.0)
    
    # Tính stopping power cho từng loại hạt
    if particle_type.lower() == "electron":
        # Electronic stopping power cho electron (Bethe-Bloch đơn giản hóa)
        # Công thức đơn giản hóa: a + b·ln(E) + c/E
        electronic = (1.02 + 0.18 * np.log(energy) - 0.25 / energy) * electron_density_ratio
        
        # Nuclear stopping power (đóng góp rất nhỏ với electron)
        nuclear = 0.001 * electronic
        
        # Radiative stopping power (quan trọng ở năng lượng cao)
        radiative = get_radiation_yield(energy, material, "electron") * electronic
        
        total = electronic + nuclear + radiative
        
    elif particle_type.lower() == "proton":
        # Electronic stopping power cho proton (Bethe-Bloch)
        # Công thức đơn giản hóa
        if energy < 1.0:
            electronic = (30 * np.sqrt(energy)) * electron_density_ratio
        else:
            electronic = (15 / energy * np.log(0.1 * energy)) * electron_density_ratio
        
        # Nuclear stopping power (quan trọng ở năng lượng thấp)
        nuclear = (0.02 / np.sqrt(energy)) * electron_density_ratio
        
        # Radiative stopping power (không đáng kể với proton)
        radiative = 0.0
        
        total = electronic + nuclear
        
    # Thêm các particle type khác nếu cần
    else:
        # Mặc định: giả định là proton
        logger.warning(f"Không tìm thấy dữ liệu stopping power cho {particle_type}. Sử dụng giá trị proton.")
        return calculate_stopping_power(energy, material, "proton")
    
    return {
        "electronic": electronic,
        "nuclear": nuclear,
        "radiative": radiative,
        "total": total
    }

def calculate_kerma(photon_energy_fluence: float, energy: float, material: str) -> float:
    """
    Tính KERMA (Kinetic Energy Released per unit MAss).
    
    Parameters
    ----------
    photon_energy_fluence : float
        Fluence năng lượng photon (MeV/cm²)
    energy : float
        Năng lượng photon (MeV)
    material : str
        Vật liệu
        
    Returns
    -------
    float
        KERMA (Gray = J/kg)
    """
    # Lấy hệ số hấp thụ năng lượng khối (cm²/g)
    # μ_en/ρ = μ/ρ × (1 - g)
    # Với g là phần năng lượng mất đi do bức xạ hãm
    
    # Lấy hệ số suy giảm khối
    mu_rho = get_interaction_probability(energy, material, InteractionType.TOTAL)
    
    # Tính g (phần năng lượng mất do bức xạ hãm) - giá trị gần đúng
    g = get_radiation_yield(energy, material, "electron")
    
    # Tính hệ số hấp thụ năng lượng khối
    mu_en_rho = mu_rho * (1 - g)
    
    # Chuyển đổi photon_energy_fluence từ MeV/cm² sang J/cm²
    # 1 MeV = 1.602e-13 J
    energy_fluence_J = photon_energy_fluence * 1.602e-13
    
    # Tính KERMA (Gy = J/kg)
    # KERMA = energy_fluence × μ_en/ρ × (J/kg) / (J/cm²) / (cm²/g)
    # 1 g = 0.001 kg
    kerma = energy_fluence_J * mu_en_rho * 0.001 * 100  # Gy
    
    return kerma
