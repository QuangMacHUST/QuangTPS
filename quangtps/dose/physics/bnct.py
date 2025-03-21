"""
Module tính toán liều trong liệu pháp bắt bắt neutron bởi boron (BNCT - Boron Neutron Capture Therapy).

BNCT là phương pháp xạ trị kết hợp dùng thuốc chứa boron-10 cùng với chiếu xạ neutron năng lượng thấp,
tạo ra phản ứng bắt neutron và phân rã alpha tại chỗ, giải phóng năng lượng cao tập trung trong tế bào u.
Module này cung cấp các hàm tính toán thành phần liều khác nhau trong BNCT dựa trên phân bố neutron.
"""

import numpy as np
import logging
from typing import Dict, Tuple, Optional, Union, List, Any

logger = logging.getLogger(__name__)

# Các hằng số vật lý và sinh học
# Các hệ số sinh học tương đối (CBE - Compound Biological Effectiveness) cho từng thành phần liều
CBE_FACTORS = {
    "boron": {
        "tumor": {
            "bpa": 3.8,  # Boronophenylalanine
            "bsh": 2.5,  # Sodium borocaptate
            "mixed": 3.0  # Phối hợp BPA và BSH
        },
        "skin": {
            "bpa": 2.5,
            "bsh": 1.3,
            "mixed": 2.0
        },
        "brain": {
            "bpa": 1.3,
            "bsh": 1.3,
            "mixed": 1.3
        }
    },
    "nitrogen": 3.0,  # RBE cho phản ứng nitrogen capture
    "hydrogen": 3.2,  # RBE cho proton recoil từ neutron nhanh
    "photon": 1.0,    # RBE cho photon (gamma) là 1 theo định nghĩa
}

# Các hệ số kerma (kinetic energy released per unit mass) cho các nguyên tố
# Đơn vị: cGy.cm^2 cho 1 neutron/cm^2
KERMA_FACTORS = {
    "hydrogen": 4.0e-11,   # Kerma factor cho hydrogen recoil
    "nitrogen": 7.8e-12,   # Kerma factor cho nitrogen capture
    "boron": 8.66e-8       # Kerma factor cho boron capture
}

def calculate_bnct_dose(thermal_neutron_flux: np.ndarray,
                       epithermal_neutron_flux: np.ndarray,
                       fast_neutron_flux: np.ndarray,
                       gamma_dose: np.ndarray,
                       boron_concentration: Union[float, np.ndarray],
                       tissue_nitrogen_fraction: Union[float, np.ndarray] = 0.035,
                       tissue_hydrogen_fraction: Union[float, np.ndarray] = 0.10,
                       boron_compound: str = "bpa",
                       tissue_type: Union[str, np.ndarray] = "tumor",
                       irradiation_time: float = 60.0) -> Dict[str, np.ndarray]:
    """
    Tính toán các thành phần liều trong liệu pháp BNCT.
    
    Parameters:
        thermal_neutron_flux (np.ndarray): Phân bố neutron nhiệt (n/cm²/s)
        epithermal_neutron_flux (np.ndarray): Phân bố neutron epithermal (n/cm²/s)
        fast_neutron_flux (np.ndarray): Phân bố neutron nhanh (n/cm²/s)
        gamma_dose (np.ndarray): Phân bố liều gamma (Gy)
        boron_concentration (float hoặc np.ndarray): Nồng độ Boron-10 (ppm hoặc μg/g)
        tissue_nitrogen_fraction (float hoặc np.ndarray): Tỉ lệ khối lượng nitrogen trong mô
        tissue_hydrogen_fraction (float hoặc np.ndarray): Tỉ lệ khối lượng hydrogen trong mô
        boron_compound (str): Loại hợp chất boron ("bpa", "bsh", "mixed")
        tissue_type (str hoặc np.ndarray): Loại mô ("tumor", "skin", "brain")
        irradiation_time (float): Thời gian chiếu xạ (giây)
    
    Returns:
        Dict[str, np.ndarray]: Dictionary chứa các thành phần liều và tổng liều sinh học
    """
    # Kiểm tra đầu vào
    if np.any(thermal_neutron_flux < 0) or np.any(epithermal_neutron_flux < 0) or np.any(fast_neutron_flux < 0):
        logger.warning("Negative neutron flux detected - potential calculation error")
    
    # Chuyển đổi thành mảng nếu cần
    if isinstance(boron_concentration, (int, float)):
        boron_conc = np.ones_like(thermal_neutron_flux) * boron_concentration
    else:
        boron_conc = boron_concentration
        
    if isinstance(tissue_nitrogen_fraction, (int, float)):
        nitrogen_frac = np.ones_like(thermal_neutron_flux) * tissue_nitrogen_fraction
    else:
        nitrogen_frac = tissue_nitrogen_fraction
        
    if isinstance(tissue_hydrogen_fraction, (int, float)):
        hydrogen_frac = np.ones_like(thermal_neutron_flux) * tissue_hydrogen_fraction
    else:
        hydrogen_frac = tissue_hydrogen_fraction
    
    # Tính toán fluence (tích phân flux theo thời gian)
    thermal_fluence = thermal_neutron_flux * irradiation_time
    epithermal_fluence = epithermal_neutron_flux * irradiation_time
    fast_fluence = fast_neutron_flux * irradiation_time
    
    # 1. Tính liều từ phản ứng bắt neutron bởi boron: B-10(n,α)Li-7
    # D_B [Gy] = Φ_thermal [n/cm²] × C_B [μg/g] × K_B [cGy.cm²/n/ppm] / 100
    boron_dose = thermal_fluence * boron_conc * KERMA_FACTORS["boron"] / 100.0
    
    # 2. Tính liều từ phản ứng bắt neutron bởi nitrogen: N-14(n,p)C-14
    nitrogen_dose = thermal_fluence * nitrogen_frac * KERMA_FACTORS["nitrogen"] / 100.0
    
    # 3. Tính liều từ hydrogen recoil protons: H-1(n,n')p
    # Chủ yếu từ neutron nhanh
    hydrogen_dose = fast_fluence * hydrogen_frac * KERMA_FACTORS["hydrogen"] / 100.0
    
    # 4. Liều gamma đã được cung cấp trực tiếp
    
    # Tính liều vật lý tổng
    physical_dose = boron_dose + nitrogen_dose + hydrogen_dose + gamma_dose
    
    # Áp dụng hệ số CBE/RBE cho từng thành phần liều để tính liều sinh học
    # Xử lý tissue_type là chuỗi đơn
    if isinstance(tissue_type, str):
        boron_cbe = CBE_FACTORS["boron"][tissue_type][boron_compound]
        boron_weighted_dose = boron_dose * boron_cbe
    # Xử lý tissue_type là mảng
    else:
        boron_weighted_dose = np.zeros_like(boron_dose)
        for t_type in ["tumor", "skin", "brain"]:
            mask = (tissue_type == t_type)
            if np.any(mask):
                boron_weighted_dose[mask] = boron_dose[mask] * CBE_FACTORS["boron"][t_type][boron_compound]
    
    # Áp dụng RBE cho thành phần khác
    nitrogen_weighted_dose = nitrogen_dose * CBE_FACTORS["nitrogen"]
    hydrogen_weighted_dose = hydrogen_dose * CBE_FACTORS["hydrogen"]
    gamma_weighted_dose = gamma_dose * CBE_FACTORS["photon"]  # RBE của photon = 1
    
    # Tính tổng liều sinh học tương đương
    total_weighted_dose = (boron_weighted_dose + nitrogen_weighted_dose + 
                          hydrogen_weighted_dose + gamma_weighted_dose)
    
    # Trả về tất cả các thành phần liều
    return {
        "boron_dose": boron_dose,
        "nitrogen_dose": nitrogen_dose,
        "hydrogen_dose": hydrogen_dose,
        "gamma_dose": gamma_dose,
        "physical_dose": physical_dose,
        "boron_weighted_dose": boron_weighted_dose,
        "nitrogen_weighted_dose": nitrogen_weighted_dose,
        "hydrogen_weighted_dose": hydrogen_weighted_dose,
        "gamma_weighted_dose": gamma_weighted_dose,
        "total_weighted_dose": total_weighted_dose
    }

def calculate_tumor_to_normal_ratio(tumor_boron_concentration: float,
                                   normal_boron_concentration: float,
                                   tumor_thermal_neutron_flux: np.ndarray = None,
                                   normal_thermal_neutron_flux: np.ndarray = None) -> float:
    """
    Tính tỉ lệ liều giữa u và mô lành (Tumor-to-Normal Ratio - TNR) cho BNCT.
    
    Parameters:
        tumor_boron_concentration (float): Nồng độ Boron-10 trong u (ppm)
        normal_boron_concentration (float): Nồng độ Boron-10 trong mô lành (ppm)
        tumor_thermal_neutron_flux (np.ndarray, optional): Phân bố neutron nhiệt tại u
        normal_thermal_neutron_flux (np.ndarray, optional): Phân bố neutron nhiệt tại mô lành
    
    Returns:
        float: Tỉ lệ liều u/mô lành cho thành phần boron
    """
    # Tỉ lệ nồng độ
    conc_ratio = tumor_boron_concentration / normal_boron_concentration if normal_boron_concentration > 0 else float('inf')
    
    # Nếu không cung cấp phân bố neutron, chỉ dựa vào nồng độ
    if tumor_thermal_neutron_flux is None or normal_thermal_neutron_flux is None:
        return conc_ratio
    
    # Tỉ lệ neutron nhiệt trung bình
    tumor_flux_avg = np.mean(tumor_thermal_neutron_flux)
    normal_flux_avg = np.mean(normal_thermal_neutron_flux)
    flux_ratio = tumor_flux_avg / normal_flux_avg if normal_flux_avg > 0 else 1.0
    
    # TNR dựa trên cả nồng độ boron và phân bố neutron
    return conc_ratio * flux_ratio

def estimate_boron_distribution(blood_boron_concentration: float,
                               tumor_to_blood_ratio: float = 3.5,
                               normal_to_blood_ratio: float = 1.0,
                               tissue_types: np.ndarray = None) -> np.ndarray:
    """
    Ước tính phân bố nồng độ boron trong các mô dựa trên nồng độ boron trong máu.
    
    Parameters:
        blood_boron_concentration (float): Nồng độ boron trong máu (ppm)
        tumor_to_blood_ratio (float): Tỉ lệ nồng độ boron u/máu
        normal_to_blood_ratio (float): Tỉ lệ nồng độ boron mô lành/máu
        tissue_types (np.ndarray): Mảng chứa mã phân loại mô
    
    Returns:
        np.ndarray: Phân bố nồng độ boron ước tính trong mô
    """
    if tissue_types is None:
        logger.warning("No tissue types provided - returning uniform boron concentration")
        return blood_boron_concentration
    
    # Khởi tạo mảng nồng độ boron
    boron_distribution = np.ones_like(tissue_types, dtype=float) * blood_boron_concentration * normal_to_blood_ratio
    
    # Gán nồng độ boron trong khối u
    tumor_mask = (tissue_types == "tumor")
    if np.any(tumor_mask):
        boron_distribution[tumor_mask] = blood_boron_concentration * tumor_to_blood_ratio
    
    return boron_distribution

def calculate_minimum_boron_concentration(target_dose: float,
                                        thermal_neutron_flux: float,
                                        irradiation_time: float = 60.0) -> float:
    """
    Tính nồng độ boron tối thiểu cần thiết để đạt liều mục tiêu.
    
    Parameters:
        target_dose (float): Liều mục tiêu (Gy)
        thermal_neutron_flux (float): Thông lượng neutron nhiệt (n/cm²/s)
        irradiation_time (float): Thời gian chiếu xạ (giây)
    
    Returns:
        float: Nồng độ boron tối thiểu (ppm)
    """
    # Tính fluence
    thermal_fluence = thermal_neutron_flux * irradiation_time
    
    # D_B [Gy] = Φ_thermal [n/cm²] × C_B [μg/g] × K_B [cGy.cm²/n/ppm] / 100
    # => C_B = D_B * 100 / (Φ_thermal * K_B)
    min_concentration = target_dose * 100.0 / (thermal_fluence * KERMA_FACTORS["boron"])
    
    return min_concentration

def calculate_neutron_spectrum_weighted_dose(neutron_spectrum: np.ndarray, 
                                          energy_bins: np.ndarray,
                                          boron_concentration: float,
                                          tissue_type: str = "tumor",
                                          boron_compound: str = "bpa") -> Dict[str, float]:
    """
    Tính liều trọng số dựa trên phổ năng lượng neutron đầy đủ.
    
    Parameters:
        neutron_spectrum (np.ndarray): Phổ neutron theo các khoảng năng lượng (n/cm²/s)
        energy_bins (np.ndarray): Các khoảng năng lượng tương ứng (eV)
        boron_concentration (float): Nồng độ boron (ppm)
        tissue_type (str): Loại mô
        boron_compound (str): Loại hợp chất boron
    
    Returns:
        Dict[str, float]: Dictionary chứa các thành phần liều
    """
    # Phân loại neutron theo năng lượng
    thermal_mask = (energy_bins < 0.5)  # Neutron nhiệt: E < 0.5 eV
    epithermal_mask = (energy_bins >= 0.5) & (energy_bins < 10000)  # Neutron epithermal: 0.5 eV ≤ E < 10 keV
    fast_mask = (energy_bins >= 10000)  # Neutron nhanh: E ≥ 10 keV
    
    # Tính thông lượng neutron cho mỗi vùng năng lượng
    thermal_flux = np.sum(neutron_spectrum[thermal_mask])
    epithermal_flux = np.sum(neutron_spectrum[epithermal_mask])
    fast_flux = np.sum(neutron_spectrum[fast_mask])
    
    # Ước tính liều gamma dựa vào neutron (giả định đơn giản)
    # Gamma dose thường khoảng 2-3 lần liều neutron nhiệt với water phantom
    gamma_dose_rate = thermal_flux * 2.5e-13  # Gy/(n/cm²)
    
    # Tính liều dùng hàm có sẵn
    doses = calculate_bnct_dose(
        thermal_neutron_flux=np.array([thermal_flux]),
        epithermal_neutron_flux=np.array([epithermal_flux]),
        fast_neutron_flux=np.array([fast_flux]),
        gamma_dose=np.array([gamma_dose_rate]),
        boron_concentration=boron_concentration,
        tissue_type=tissue_type,
        boron_compound=boron_compound,
        irradiation_time=1.0  # 1 giây để tính liều tức thời
    )
    
    # Chuyển tất cả giá trị trong dict sang dạng scalar
    return {k: float(v[0]) for k, v in doses.items()}

def estimate_treatment_time(target_dose: float,
                           boron_concentration: float,
                           thermal_neutron_flux: float,
                           tissue_type: str = "tumor",
                           boron_compound: str = "bpa") -> float:
    """
    Ước tính thời gian chiếu xạ cần thiết để đạt liều mục tiêu.
    
    Parameters:
        target_dose (float): Liều mục tiêu (Gy)
        boron_concentration (float): Nồng độ boron (ppm)
        thermal_neutron_flux (float): Thông lượng neutron nhiệt (n/cm²/s)
        tissue_type (str): Loại mô
        boron_compound (str): Loại hợp chất boron
    
    Returns:
        float: Thời gian chiếu xạ ước tính (phút)
    """
    # Tính liều boron tức thời
    boron_dose_rate = (thermal_neutron_flux * boron_concentration * 
                     KERMA_FACTORS["boron"] / 100.0)  # Gy/s
    
    # Tính liều trọng số boron tức thời
    boron_cbe = CBE_FACTORS["boron"][tissue_type][boron_compound]
    boron_weighted_dose_rate = boron_dose_rate * boron_cbe  # Gy-Eq/s
    
    # Ước tính thời gian chiếu xạ (giả định 80% liều từ phản ứng boron)
    # Phần còn lại từ các thành phần khác (nitrogen, hydrogen, gamma)
    if boron_weighted_dose_rate > 0:
        treatment_time = target_dose * 0.8 / boron_weighted_dose_rate  # giây
        treatment_time_minutes = treatment_time / 60.0  # chuyển sang phút
    else:
        treatment_time_minutes = float('inf')
    
    return treatment_time_minutes

def optimize_beam_parameters(target_dose: float,
                            boron_concentration: float,
                            max_normal_dose: float,
                            beam_options: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Tối ưu hóa tham số chùm tia để đạt liều mục tiêu trong u và giảm thiểu liều cho mô lành.
    
    Parameters:
        target_dose (float): Liều mục tiêu cho khối u (Gy)
        boron_concentration (float): Nồng độ boron (ppm)
        max_normal_dose (float): Liều tối đa cho phép cho mô lành (Gy)
        beam_options (List[Dict]): Danh sách các cấu hình chùm tia khả dĩ
    
    Returns:
        Dict: Tham số chùm tia tối ưu
    """
    best_therapeutic_ratio = 0
    best_option = None
    
    for option in beam_options:
        # Trích xuất thông tin chùm tia
        tumor_thermal_flux = option["tumor_thermal_flux"]
        normal_thermal_flux = option["normal_thermal_flux"]
        irradiation_time = option.get("irradiation_time", 60.0)  # mặc định 60 giây
        
        # Tính liều cho u
        tumor_dose = calculate_bnct_dose(
            thermal_neutron_flux=np.array([tumor_thermal_flux]),
            epithermal_neutron_flux=np.array([option.get("tumor_epithermal_flux", 0.0)]),
            fast_neutron_flux=np.array([option.get("tumor_fast_flux", 0.0)]),
            gamma_dose=np.array([option.get("tumor_gamma_dose", 0.0)]),
            boron_concentration=boron_concentration,
            tissue_type="tumor",
            boron_compound=option.get("boron_compound", "bpa"),
            irradiation_time=irradiation_time
        )
        
        # Tính liều cho mô lành
        normal_dose = calculate_bnct_dose(
            thermal_neutron_flux=np.array([normal_thermal_flux]),
            epithermal_neutron_flux=np.array([option.get("normal_epithermal_flux", 0.0)]),
            fast_neutron_flux=np.array([option.get("normal_fast_flux", 0.0)]),
            gamma_dose=np.array([option.get("normal_gamma_dose", 0.0)]),
            boron_concentration=boron_concentration/3.5,  # Giả định nồng độ Boron trong mô lành = 1/3.5 trong u
            tissue_type="brain",  # Giả định mô lành là não
            boron_compound=option.get("boron_compound", "bpa"),
            irradiation_time=irradiation_time
        )
        
        tumor_total_dose = float(tumor_dose["total_weighted_dose"][0])
        normal_total_dose = float(normal_dose["total_weighted_dose"][0])
        
        # Kiểm tra các ràng buộc
        if normal_total_dose <= max_normal_dose and tumor_total_dose >= 0.8 * target_dose:
            # Tính tỉ lệ điều trị (therapeutic ratio)
            therapeutic_ratio = tumor_total_dose / normal_total_dose if normal_total_dose > 0 else float('inf')
            
            # Cập nhật lựa chọn tốt nhất
            if therapeutic_ratio > best_therapeutic_ratio:
                best_therapeutic_ratio = therapeutic_ratio
                best_option = option.copy()
                best_option["estimated_tumor_dose"] = tumor_total_dose
                best_option["estimated_normal_dose"] = normal_total_dose
                best_option["therapeutic_ratio"] = therapeutic_ratio
    
    if best_option is None:
        logger.warning("No beam option meets the constraints")
        return None
    
    return best_option 