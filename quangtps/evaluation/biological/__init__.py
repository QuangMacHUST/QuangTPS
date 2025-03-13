#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module sinh học cho đánh giá kế hoạch xạ trị.

Module này cung cấp các công cụ đánh giá tác động sinh học của kế hoạch điều trị,
bao gồm TCP (Tumor Control Probability), NTCP (Normal Tissue Complication Probability),
chuyển đổi liều tương đương sinh học (EQD2, BED), và hiệu ứng oxy (Oxygen Effect).
"""

from quangtps.evaluation.biological.tcp import (
    calculate_tcp_lq_poisson,
    calculate_tcp_lq_poisson_dvh,
    calculate_tcp_niemierko,
    calculate_tcp_logistic,
    calculate_tcp_webb
)

from quangtps.evaluation.biological.ntcp import (
    calculate_ntcp_lkb,
    calculate_ntcp_relative_seriality,
    calculate_ntcp_logit,
    calculate_ntcp_poisson,
    calculate_cutoff_ntcp,
    calculate_ntcp_for_dvh,
    get_ntcp_constraints
)

from quangtps.evaluation.biological.eqd2 import (
    calculate_eqd2,
    calculate_bed,
    bed_to_eqd2,
    eqd2_to_bed,
    convert_dose_fractionation,
    calculate_eqd2_for_volume,
    calculate_standard_fractionation_equivalent,
    get_alpha_beta_ratio
)

from quangtps.evaluation.biological.eqd2 import EQD2Calculator
from quangtps.evaluation.biological.ntcp import NTCPModels
from quangtps.evaluation.biological.tcp import TCPModels
from quangtps.evaluation.biological.bed import BiologicalEffectiveDose
from quangtps.evaluation.biological.oxygen_effect import OxygenEffect

__all__ = [
    # TCP functions
    'calculate_tcp_lq_poisson',
    'calculate_tcp_lq_poisson_dvh',
    'calculate_tcp_niemierko',
    'calculate_tcp_logistic',
    'calculate_tcp_webb',
    
    # NTCP functions
    'calculate_ntcp_lkb',
    'calculate_ntcp_relative_seriality',
    'calculate_ntcp_logit',
    'calculate_ntcp_poisson',
    'calculate_cutoff_ntcp',
    'calculate_ntcp_for_dvh',
    'get_ntcp_constraints',
    
    # EQD2 and dose conversion functions
    'calculate_eqd2',
    'calculate_bed',
    'bed_to_eqd2',
    'eqd2_to_bed',
    'convert_dose_fractionation',
    'calculate_eqd2_for_volume',
    'calculate_standard_fractionation_equivalent',
    'get_alpha_beta_ratio',
    
    # Main class objects
    'EQD2Calculator',
    'NTCPModels',
    'TCPModels',
    'BiologicalEffectiveDose',
    'OxygenEffect',
    
    # Helper functions defined in __init__
    'calculate_tcp',
    'calculate_ntcp',
    'calculate_bed',
    'calculate_oxygen_effect'
]

def calculate_tcp(dose: float, fractions: int, 
               model: str = 'poisson', 
               parameters: dict = None,
               alpha_beta: float = 10.0) -> float:
    """
    Tính xác suất kiểm soát khối u (TCP).
    
    Parameters
    ----------
    dose : float
        Tổng liều (Gy)
    fractions : int
        Số phân liều
    model : str, optional
        Mô hình tính TCP ('poisson', 'logistic', 'lq_poisson'), mặc định là 'poisson'
    parameters : dict, optional
        Các tham số cho mô hình TCP
    alpha_beta : float, optional
        Tỷ lệ alpha/beta của mô (Gy), mặc định là 10.0
        
    Returns
    -------
    float
        Xác suất kiểm soát khối u (0-1)
    """
    if parameters is None:
        parameters = {}
        
    # Sử dụng giá trị mặc định nếu không được chỉ định
    if model == 'poisson':
        parameters.setdefault('tcd50', 45.0)  # Liều kiểm soát 50% (Gy)
        parameters.setdefault('gamma50', 2.0)  # Độ dốc của đường cong liều-đáp ứng
    elif model == 'logistic':
        parameters.setdefault('tcd50', 45.0)
        parameters.setdefault('k', 4.0)  # Hệ số độ dốc
    elif model == 'lq_poisson':
        parameters.setdefault('alpha', 0.3)  # Hệ số alpha (Gy^-1)
        parameters.setdefault('rho', 10**7)  # Mật độ tế bào nhân ban đầu
        
    # Tính TCP dựa trên mô hình được chọn
    if model == 'poisson':
        return TCPModels.poisson_tcp(dose, fractions, parameters['tcd50'], parameters['gamma50'], alpha_beta)
    elif model == 'logistic':
        return TCPModels.logistic_tcp(dose, fractions, parameters['tcd50'], parameters['k'], alpha_beta)
    elif model == 'lq_poisson':
        return TCPModels.lq_poisson_tcp(dose, fractions, parameters['alpha'], parameters['rho'], alpha_beta)
    else:
        raise ValueError(f"Mô hình '{model}' không được hỗ trợ")

def calculate_ntcp(dose: float, fractions: int,
                model: str = 'lkb',
                parameters: dict = None,
                organ: str = None,
                endpoint: str = None,
                alpha_beta: float = 3.0) -> float:
    """
    Tính xác suất biến chứng mô lành (NTCP).
    
    Parameters
    ----------
    dose : float
        Tổng liều (Gy)
    fractions : int
        Số phân liều
    model : str, optional
        Mô hình tính NTCP ('lkb', 'logistic', 'relative_seriality'), mặc định là 'lkb'
    parameters : dict, optional
        Các tham số cho mô hình NTCP
    organ : str, optional
        Tên cơ quan
    endpoint : str, optional
        Điểm cuối lâm sàng
    alpha_beta : float, optional
        Tỷ lệ alpha/beta của mô (Gy), mặc định là 3.0
        
    Returns
    -------
    float
        Xác suất biến chứng mô lành (0-1)
    """
    if parameters is None:
        parameters = {}
        
    # Nếu chỉ định cơ quan và điểm cuối, lấy tham số từ dữ liệu sẵn có
    if parameters == {} and organ is not None and endpoint is not None:
        parameters = NTCPModels.get_model_parameters(organ, endpoint, model)
        
    # Sử dụng giá trị mặc định nếu không được chỉ định
    if model == 'lkb':
        parameters.setdefault('td50', 50.0)  # Liều gây biến chứng 50% (Gy)
        parameters.setdefault('n', 0.1)  # Thông số thể tích (volume parameter)
        parameters.setdefault('m', 0.3)  # Độ dốc
    elif model == 'logistic':
        parameters.setdefault('td50', 50.0)
        parameters.setdefault('k', 4.0)  # Hệ số độ dốc
    elif model == 'relative_seriality':
        parameters.setdefault('d50', 50.0)  # Liều gây biến chứng 50% (Gy)
        parameters.setdefault('gamma', 2.0)  # Độ dốc của đường cong liều-đáp ứng
        parameters.setdefault('s', 1.0)  # Thông số chuỗi (seriality parameter)
        
    # Chuyển đổi liều sang EQD2 trước
    eqd2_dose = EQD2Calculator.calculate_eqd2(dose, fractions, alpha_beta)
    
    # Tính NTCP dựa trên mô hình được chọn
    if model == 'lkb':
        return NTCPModels.lkb_ntcp(eqd2_dose, parameters['td50'], parameters['n'], parameters['m'])
    elif model == 'logistic':
        return NTCPModels.logistic_ntcp(eqd2_dose, parameters['td50'], parameters['k'])
    elif model == 'relative_seriality':
        # Giả sử v_ref = 1 (toàn bộ cơ quan nhận liều đồng nhất)
        return NTCPModels.relative_seriality_ntcp(eqd2_dose, parameters['d50'], parameters['gamma'], parameters['s'], 1.0)
    else:
        raise ValueError(f"Mô hình '{model}' không được hỗ trợ")

def calculate_bed(dose: float, fractions: int, alpha_beta: float = 10.0) -> float:
    """
    Tính toán liều sinh học hiệu quả (BED).
    
    Parameters
    ----------
    dose : float
        Tổng liều (Gy)
    fractions : int
        Số phân liều
    alpha_beta : float, optional
        Tỷ lệ alpha/beta của mô (Gy), mặc định là 10.0
        
    Returns
    -------
    float
        Liều sinh học hiệu quả (BED), đơn vị Gy
    """
    return BiologicalEffectiveDose.calculate_bed(dose, fractions, alpha_beta)

def calculate_oxygen_effect(dose: float, 
                        oxygen_concentration: float, 
                        hypoxic_fraction: float = 0.0,
                        method: str = 'oer',
                        parameters: dict = None) -> float:
    """
    Tính toán hiệu ứng oxy cho một liều bức xạ nhất định.
    
    Parameters
    ----------
    dose : float
        Liều bức xạ (Gy)
    oxygen_concentration : float
        Nồng độ oxy (mmHg)
    hypoxic_fraction : float, optional
        Phân đoạn tế bào thiếu oxy (0-1), mặc định là 0.0
    method : str, optional
        Phương pháp tính ('oer', 'survival', 'effective_dose'), mặc định là 'oer'
    parameters : dict, optional
        Các tham số bổ sung cho tính toán
        
    Returns
    -------
    float
        Kết quả tính toán tùy thuộc vào phương pháp:
        - 'oer': Tỷ số tăng cường oxy
        - 'survival': Phân số sống sót tế bào
        - 'effective_dose': Liều hiệu quả đã điều chỉnh theo hiệu ứng oxy
    """
    if parameters is None:
        parameters = {}
        
    # Thiết lập các tham số mặc định
    parameters.setdefault('k_m', 3.0)  # Hằng số Michaelis-Menten
    parameters.setdefault('max_oer', 3.0)  # OER tối đa
    parameters.setdefault('alpha', 0.3)  # Hệ số alpha (Gy^-1)
    parameters.setdefault('beta', 0.03)  # Hệ số beta (Gy^-2)
    
    if method == 'oer':
        return OxygenEffect.calculate_oer(
            oxygen_concentration, 
            parameters['k_m'], 
            parameters['max_oer']
        )
    elif method == 'effective_dose':
        return OxygenEffect.calculate_oer_effective_dose(
            dose, 
            oxygen_concentration, 
            parameters['k_m'], 
            parameters['max_oer']
        )
    elif method == 'survival':
        if hypoxic_fraction > 0:
            return OxygenEffect.calculate_hypoxic_fraction_effect(
                dose, 
                hypoxic_fraction, 
                OxygenEffect.calculate_oer(oxygen_concentration, parameters['k_m'], parameters['max_oer']),
                parameters['alpha'],
                parameters['beta']
            )
        else:
            return OxygenEffect.calculate_oxygen_modified_survival(
                dose, 
                oxygen_concentration, 
                parameters['alpha'], 
                parameters['beta'], 
                parameters['k_m'], 
                parameters['max_oer'], 
                parameters['max_oer']
            )
    else:
        raise ValueError(f"Phương pháp '{method}' không được hỗ trợ")
