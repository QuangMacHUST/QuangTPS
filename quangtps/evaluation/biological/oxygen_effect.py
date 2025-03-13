#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module chứa các phương thức tính toán hiệu ứng oxy (Oxygen Effect) trong xạ trị.

Hiệu ứng oxy là hiện tượng tế bào thiếu oxy (hypoxic) ít nhạy cảm với bức xạ ion hóa
hơn so với tế bào đủ oxy (normoxic). Tỷ số tăng cường oxy (OER - Oxygen Enhancement Ratio)
là tỷ lệ giữa liều cần thiết để đạt cùng hiệu ứng sinh học trong điều kiện thiếu oxy
so với điều kiện đủ oxy.
"""

import numpy as np
from typing import Dict, Union, Tuple, List, Optional


class OxygenEffect:
    """
    Lớp tính toán các hiệu ứng liên quan đến oxy trong xạ trị.
    """

    @staticmethod
    def calculate_oer(oxygen_concentration: float, k_m: float = 3.0, 
                    max_oer: float = 3.0) -> float:
        """
        Tính toán tỷ số tăng cường oxy (OER) theo nồng độ oxy.
        
        OER được mô hình hóa bằng phương trình Alper-Howard-Flanders:
        OER = (max_OER * p + K_m) / (p + K_m)
        
        Trong đó:
        - p: nồng độ oxy (mmHg)
        - K_m: nồng độ oxy tại điểm đạt được một nửa hiệu ứng tối đa (mmHg)
        - max_OER: giá trị OER tối đa
        
        Parameters
        ----------
        oxygen_concentration : float
            Nồng độ oxy (mmHg)
        k_m : float, optional
            Hằng số Michaelis-Menten (mmHg), mặc định là 3.0 mmHg
        max_oer : float, optional
            Giá trị OER tối đa, mặc định là 3.0
            
        Returns
        -------
        float
            Tỷ số tăng cường oxy (OER)
            
        References
        ----------
        Alper, T., Howard-Flanders, P. (1956) Role of oxygen in modifying the 
        radiosensitivity of E. coli B. Nature, 178, 978-979.
        """
        if oxygen_concentration < 0:
            raise ValueError("Nồng độ oxy phải không âm")
            
        # Tính OER theo mô hình Alper-Howard-Flanders
        oer = (max_oer * oxygen_concentration + k_m) / (oxygen_concentration + k_m)
        
        return float(oer)

    @staticmethod
    def calculate_oer_effective_dose(physical_dose: float, oxygen_concentration: float, 
                                   k_m: float = 3.0, max_oer: float = 3.0) -> float:
        """
        Tính liều hiệu quả khi xét đến hiệu ứng oxy.
        
        Parameters
        ----------
        physical_dose : float
            Liều vật lý (Gy)
        oxygen_concentration : float
            Nồng độ oxy (mmHg)
        k_m : float, optional
            Hằng số Michaelis-Menten (mmHg), mặc định là 3.0 mmHg
        max_oer : float, optional
            Giá trị OER tối đa, mặc định là 3.0
            
        Returns
        -------
        float
            Liều hiệu quả có tính đến hiệu ứng oxy (Gy)
        """
        if physical_dose < 0:
            raise ValueError("Liều vật lý phải không âm")
            
        # Tính OER
        oer = OxygenEffect.calculate_oer(oxygen_concentration, k_m, max_oer)
        
        # Liều hiệu quả = Liều vật lý / OER
        effective_dose = physical_dose / oer
        
        return float(effective_dose)

    @staticmethod
    def calculate_hypoxic_fraction_effect(dose: float, 
                                       hypoxic_fraction: float, 
                                       oer_hypoxic: float = 3.0,
                                       alpha_normoxic: float = 0.3,
                                       beta_normoxic: float = 0.03) -> float:
        """
        Tính toán hiệu ứng của phân đoạn thiếu oxy trong khối u theo mô hình LQ.
        
        Sự sống sót tế bào được mô hình hóa theo hai quần thể:
        - Quần thể đủ oxy (normoxic): S_norm = exp(-alpha_norm*D - beta_norm*D²)
        - Quần thể thiếu oxy (hypoxic): S_hyp = exp(-alpha_norm*D/OER - beta_norm*(D/OER)²)
        - Tổng quát: S = (1-HF)*S_norm + HF*S_hyp
        
        Trong đó:
        - HF: phân đoạn thiếu oxy (hypoxic fraction)
        - OER: tỷ số tăng cường oxy
        
        Parameters
        ----------
        dose : float
            Liều bức xạ (Gy)
        hypoxic_fraction : float
            Phân đoạn thiếu oxy (0-1)
        oer_hypoxic : float, optional
            Tỷ số tăng cường oxy cho vùng thiếu oxy, mặc định là 3.0
        alpha_normoxic : float, optional
            Hệ số alpha của mô hình LQ cho tế bào đủ oxy (Gy^-1), mặc định là 0.3
        beta_normoxic : float, optional
            Hệ số beta của mô hình LQ cho tế bào đủ oxy (Gy^-2), mặc định là 0.03
            
        Returns
        -------
        float
            Phân số sống sót tế bào sau bức xạ
        """
        if hypoxic_fraction < 0 or hypoxic_fraction > 1:
            raise ValueError("Phân đoạn thiếu oxy phải nằm trong khoảng [0, 1]")
            
        if oer_hypoxic <= 1:
            raise ValueError("OER phải lớn hơn 1")
            
        # Tính phân số sống sót cho tế bào đủ oxy
        survival_normoxic = np.exp(-alpha_normoxic * dose - beta_normoxic * dose**2)
        
        # Tính phân số sống sót cho tế bào thiếu oxy
        alpha_hypoxic = alpha_normoxic / oer_hypoxic
        beta_hypoxic = beta_normoxic / (oer_hypoxic**2)
        survival_hypoxic = np.exp(-alpha_hypoxic * dose - beta_hypoxic * dose**2)
        
        # Tính tổng phân số sống sót
        total_survival = (1 - hypoxic_fraction) * survival_normoxic + hypoxic_fraction * survival_hypoxic
        
        return float(total_survival)

    @staticmethod
    def calculate_hypoxic_dose_modifying_factor(hypoxic_fraction: float, 
                                             oer: float = 3.0) -> float:
        """
        Tính hệ số điều chỉnh liều do thiếu oxy.
        
        Parameters
        ----------
        hypoxic_fraction : float
            Phân đoạn thiếu oxy (0-1)
        oer : float, optional
            Tỷ số tăng cường oxy, mặc định là 3.0
            
        Returns
        -------
        float
            Hệ số điều chỉnh liều (DMF - Dose Modifying Factor)
        """
        if hypoxic_fraction < 0 or hypoxic_fraction > 1:
            raise ValueError("Phân đoạn thiếu oxy phải nằm trong khoảng [0, 1]")
            
        # Tính gần đúng DMF theo công thức đơn giản
        dmf = 1 / ((1 - hypoxic_fraction) + hypoxic_fraction / oer)
        
        return float(dmf)

    @staticmethod
    def calculate_reoxygenation_dynamics(initial_hypoxic_fraction: float,
                                      time_points: np.ndarray,
                                      reoxygenation_half_time: float) -> np.ndarray:
        """
        Mô phỏng động học tái oxy hóa trong quá trình xạ trị phân liều.
        
        Parameters
        ----------
        initial_hypoxic_fraction : float
            Phân đoạn thiếu oxy ban đầu (0-1)
        time_points : np.ndarray
            Mảng các điểm thời gian cần tính (ngày)
        reoxygenation_half_time : float
            Thời gian bán rã tái oxy hóa (ngày)
            
        Returns
        -------
        np.ndarray
            Mảng các giá trị phân đoạn thiếu oxy tại các điểm thời gian
        """
        if initial_hypoxic_fraction < 0 or initial_hypoxic_fraction > 1:
            raise ValueError("Phân đoạn thiếu oxy ban đầu phải nằm trong khoảng [0, 1]")
            
        if reoxygenation_half_time <= 0:
            raise ValueError("Thời gian bán rã tái oxy hóa phải dương")
            
        # Tính phân đoạn thiếu oxy theo thời gian với mô hình suy giảm theo hàm mũ
        hypoxic_fractions = initial_hypoxic_fraction * np.exp(-np.log(2) * time_points / reoxygenation_half_time)
        
        return hypoxic_fractions

    @staticmethod
    def calculate_oxygen_modification_factor(oxygen_concentration: float,
                                          k_m: float = 3.0,
                                          max_oer_alpha: float = 3.0,
                                          max_oer_beta: float = 3.0) -> Dict[str, float]:
        """
        Tính hệ số điều chỉnh oxy cho các tham số alpha và beta trong mô hình LQ.
        
        Parameters
        ----------
        oxygen_concentration : float
            Nồng độ oxy (mmHg)
        k_m : float, optional
            Hằng số Michaelis-Menten (mmHg), mặc định là 3.0 mmHg
        max_oer_alpha : float, optional
            Giá trị OER tối đa cho tham số alpha, mặc định là 3.0
        max_oer_beta : float, optional
            Giá trị OER tối đa cho tham số beta, mặc định là 3.0
            
        Returns
        -------
        Dict[str, float]
            Từ điển chứa hệ số điều chỉnh cho alpha và beta:
            - alpha_mod: hệ số điều chỉnh cho alpha
            - beta_mod: hệ số điều chỉnh cho beta
        """
        if oxygen_concentration < 0:
            raise ValueError("Nồng độ oxy phải không âm")
            
        # Sử dụng mô hình Alper-Howard-Flanders để tính hệ số điều chỉnh
        alpha_mod = (oxygen_concentration + k_m) / (oxygen_concentration + k_m / max_oer_alpha)
        beta_mod = (oxygen_concentration + k_m) / (oxygen_concentration + k_m / max_oer_beta)
        
        return {
            'alpha_mod': float(alpha_mod),
            'beta_mod': float(beta_mod)
        }

    @staticmethod
    def calculate_oxygen_modified_survival(dose: float, 
                                        oxygen_concentration: float,
                                        alpha_aerated: float = 0.3,
                                        beta_aerated: float = 0.03,
                                        k_m: float = 3.0,
                                        max_oer_alpha: float = 3.0,
                                        max_oer_beta: float = 3.0) -> float:
        """
        Tính phân số sống sót tế bào với liều đã điều chỉnh theo hiệu ứng oxy.
        
        Parameters
        ----------
        dose : float
            Liều bức xạ (Gy)
        oxygen_concentration : float
            Nồng độ oxy (mmHg)
        alpha_aerated : float, optional
            Hệ số alpha của mô hình LQ cho tế bào đủ oxy (Gy^-1), mặc định là 0.3
        beta_aerated : float, optional
            Hệ số beta của mô hình LQ cho tế bào đủ oxy (Gy^-2), mặc định là 0.03
        k_m : float, optional
            Hằng số Michaelis-Menten (mmHg), mặc định là 3.0 mmHg
        max_oer_alpha : float, optional
            Giá trị OER tối đa cho tham số alpha, mặc định là 3.0
        max_oer_beta : float, optional
            Giá trị OER tối đa cho tham số beta, mặc định là 3.0
            
        Returns
        -------
        float
            Phân số sống sót tế bào
        """
        if dose < 0:
            raise ValueError("Liều phải không âm")
            
        # Tính hệ số điều chỉnh oxy
        modifiers = OxygenEffect.calculate_oxygen_modification_factor(
            oxygen_concentration, k_m, max_oer_alpha, max_oer_beta
        )
        
        # Điều chỉnh các tham số alpha và beta
        alpha_mod = alpha_aerated * modifiers['alpha_mod']
        beta_mod = beta_aerated * modifiers['beta_mod']
        
        # Tính phân số sống sót theo mô hình LQ
        survival = np.exp(-alpha_mod * dose - beta_mod * dose**2)
        
        return float(survival)

    @staticmethod
    def calculate_sensitizer_effect(dose: float,
                                 sensitizer_enhancement_ratio: float,
                                 alpha: float = 0.3,
                                 beta: float = 0.03) -> float:
        """
        Tính hiệu ứng của tác nhân nhạy cảm oxy trong xạ trị.
        
        Parameters
        ----------
        dose : float
            Liều bức xạ (Gy)
        sensitizer_enhancement_ratio : float
            Tỷ số tăng cường nhạy cảm
        alpha : float, optional
            Hệ số alpha của mô hình LQ (Gy^-1), mặc định là 0.3
        beta : float, optional
            Hệ số beta của mô hình LQ (Gy^-2), mặc định là 0.03
            
        Returns
        -------
        float
            Phân số sống sót tế bào với tác nhân nhạy cảm
        """
        if dose < 0:
            raise ValueError("Liều phải không âm")
            
        if sensitizer_enhancement_ratio < 1:
            raise ValueError("Tỷ số tăng cường nhạy cảm phải lớn hơn hoặc bằng 1")
            
        # Điều chỉnh liều theo tỷ số tăng cường nhạy cảm
        effective_dose = dose * sensitizer_enhancement_ratio
        
        # Tính phân số sống sót
        survival = np.exp(-alpha * effective_dose - beta * effective_dose**2)
        
        return float(survival)

    @staticmethod
    def evaluate_oxygen_status(oxygen_concentration: float) -> Dict[str, Union[float, str]]:
        """
        Đánh giá trạng thái oxy hóa dựa trên nồng độ oxy.
        
        Parameters
        ----------
        oxygen_concentration : float
            Nồng độ oxy (mmHg)
            
        Returns
        -------
        Dict[str, Union[float, str]]
            Từ điển chứa đánh giá trạng thái oxy hóa:
            - status: trạng thái oxy hóa ('anoxic', 'hypoxic', 'normoxic')
            - description: mô tả trạng thái
            - relative_radiosensitivity: độ nhạy cảm bức xạ tương đối
            - oer: tỷ số tăng cường oxy
        """
        if oxygen_concentration < 0:
            raise ValueError("Nồng độ oxy phải không âm")
            
        # Đánh giá trạng thái oxy hóa
        if oxygen_concentration < 0.5:
            status = 'anoxic'
            description = 'Thiếu oxy nghiêm trọng, rất đề kháng với bức xạ'
            relative_radiosensitivity = 0.25  # Giả định 25% so với tế bào đủ oxy
        elif oxygen_concentration < 10:
            status = 'hypoxic'
            description = 'Thiếu oxy, đề kháng với bức xạ'
            # Tính độ nhạy cảm tương đối dựa trên công thức nội suy
            relative_radiosensitivity = 0.25 + (oxygen_concentration - 0.5) * (0.8 - 0.25) / (10 - 0.5)
        else:
            status = 'normoxic'
            description = 'Đủ oxy, nhạy cảm với bức xạ'
            relative_radiosensitivity = 1.0
            
        # Tính OER
        oer = OxygenEffect.calculate_oer(oxygen_concentration)
        
        return {
            'status': status,
            'description': description,
            'relative_radiosensitivity': float(relative_radiosensitivity),
            'oer': float(oer)
        }

    @staticmethod
    def generate_oxygen_effect_report(oxygen_concentration: float, 
                                   dose: float, 
                                   hypoxic_fraction: float = 0.0,
                                   alpha: float = 0.3,
                                   beta: float = 0.03) -> Dict[str, Union[float, str, Dict]]:
        """
        Tạo báo cáo toàn diện về hiệu ứng oxy cho một kế hoạch xạ trị.
        
        Parameters
        ----------
        oxygen_concentration : float
            Nồng độ oxy trung bình (mmHg)
        dose : float
            Liều bức xạ (Gy)
        hypoxic_fraction : float, optional
            Phân đoạn tế bào thiếu oxy, mặc định là 0.0
        alpha : float, optional
            Hệ số alpha của mô hình LQ (Gy^-1), mặc định là 0.3
        beta : float, optional
            Hệ số beta của mô hình LQ (Gy^-2), mặc định là 0.03
            
        Returns
        -------
        Dict[str, Union[float, str, Dict]]
            Báo cáo toàn diện về hiệu ứng oxy
        """
        # Đánh giá trạng thái oxy hóa
        oxygen_status = OxygenEffect.evaluate_oxygen_status(oxygen_concentration)
        
        # Tính OER
        oer = OxygenEffect.calculate_oer(oxygen_concentration)
        
        # Tính liều hiệu quả
        effective_dose = OxygenEffect.calculate_oer_effective_dose(dose, oxygen_concentration)
        
        # Tính hệ số điều chỉnh oxy cho alpha và beta
        modifiers = OxygenEffect.calculate_oxygen_modification_factor(oxygen_concentration)
        
        # Tính phân số sống sót tế bào
        survival = OxygenEffect.calculate_oxygen_modified_survival(
            dose, oxygen_concentration, alpha, beta
        )
        
        # Tính DMF nếu có phân đoạn thiếu oxy
        dmf = 1.0
        adjusted_survival = survival
        if hypoxic_fraction > 0:
            dmf = OxygenEffect.calculate_hypoxic_dose_modifying_factor(hypoxic_fraction, oer)
            adjusted_survival = OxygenEffect.calculate_hypoxic_fraction_effect(
                dose, hypoxic_fraction, oer, alpha, beta
            )
            
        # Diễn giải kết quả
        if oer > 2.5:
            interpretation = "Hiệu ứng oxy đáng kể, xem xét sử dụng tác nhân nhạy cảm hoặc tăng liều"
        elif oer > 1.5:
            interpretation = "Hiệu ứng oxy trung bình, theo dõi và đánh giá lại trong quá trình điều trị"
        else:
            interpretation = "Hiệu ứng oxy không đáng kể, không cần điều chỉnh đặc biệt"
            
        return {
            'oxygen_concentration': float(oxygen_concentration),
            'dose': float(dose),
            'hypoxic_fraction': float(hypoxic_fraction),
            'oxygen_status': oxygen_status,
            'oer': float(oer),
            'effective_dose': float(effective_dose),
            'alpha_modification': float(modifiers['alpha_mod']),
            'beta_modification': float(modifiers['beta_mod']),
            'cell_survival': float(survival),
            'dose_modifying_factor': float(dmf),
            'adjusted_survival': float(adjusted_survival),
            'interpretation': interpretation
        } 