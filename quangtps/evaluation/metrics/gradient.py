#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module chứa các chỉ số gradient (Gradient Index) cho đánh giá kế hoạch xạ trị.
Chỉ số gradient đo lường mức độ giảm liều nhanh xung quanh thể tích mục tiêu, 
đặc biệt quan trọng trong xạ phẫu (SRS) và xạ trị định vị thân (SBRT).
"""

import numpy as np
from typing import Dict, Union, Tuple, List, Optional


class GradientIndices:
    """
    Lớp tính toán các chỉ số gradient để đánh giá độ dốc của phân phối liều xung quanh 
    thể tích mục tiêu. Độ dốc liều cao (giá trị gradient index thấp) giúp bảo vệ mô 
    lành xung quanh khối u.
    """

    @staticmethod
    def gi_paddick(v_half: float, v_ref: float) -> float:
        """
        Tính toán chỉ số gradient Paddick: GI = V_half / V_ref
        
        Trong đó:
        - V_half là thể tích nhận một nửa liều tham chiếu
        - V_ref là thể tích nhận đầy đủ liều tham chiếu (thường là liều kê)
        
        Giá trị thấp hơn cho thấy độ dốc liều tốt hơn.
        Giá trị lý tưởng tiệm cận 1 (không thể đạt được trong thực tế).
        
        Parameters
        ----------
        v_half : float
            Thể tích nhận một nửa liều tham chiếu (cm³)
        v_ref : float
            Thể tích nhận đầy đủ liều tham chiếu (cm³)
            
        Returns
        -------
        float
            Chỉ số gradient Paddick
            
        References
        ----------
        Paddick, I., Lippitz, B. (2006) A simple dose gradient measurement tool to
        complement the conformity index. J Neurosurg., 105 Suppl, 194-201.
        """
        if v_ref <= 0:
            raise ValueError("Thể tích tham chiếu (V_ref) phải lớn hơn 0")
            
        return v_half / v_ref

    @staticmethod
    def gi_siyong(v_half: float, v_ref: float) -> float:
        """
        Tính toán chỉ số gradient dạng khác: GI = (V_half)^(1/3) / (V_ref)^(1/3)
        
        Chỉ số này đo lường tỷ lệ bán kính tương đương giữa thể tích tại nửa liều
        và thể tích tại liều tham chiếu.
        
        Parameters
        ----------
        v_half : float
            Thể tích nhận một nửa liều tham chiếu (cm³)
        v_ref : float
            Thể tích nhận đầy đủ liều tham chiếu (cm³)
            
        Returns
        -------
        float
            Chỉ số gradient dạng bán kính
            
        References
        ----------
        Siyong, K. et al. (2007) A dosimetric comparison of four treatment plans for
        stereotactic radiosurgery of brainstem lesions. Int J Radiat Oncol Biol Phys., 67(5), 1324-1332.
        """
        if v_ref <= 0 or v_half <= 0:
            raise ValueError("Các thể tích phải lớn hơn 0")
            
        return (v_half ** (1/3)) / (v_ref ** (1/3))

    @staticmethod
    def average_dose_gradient(v_ref: float, d_ref: float, r_eff: float) -> float:
        """
        Tính toán độ dốc liều trung bình: ADG = D_ref / r_eff
        
        Trong đó:
        - D_ref là liều tham chiếu
        - r_eff là bán kính hiệu dụng ngoài thể tích tham chiếu (tính từ bề mặt thể tích tham chiếu)
        
        Parameters
        ----------
        v_ref : float
            Thể tích nhận đầy đủ liều tham chiếu (cm³)
        d_ref : float
            Liều tham chiếu (Gy)
        r_eff : float
            Bán kính hiệu dụng ngoài thể tích tham chiếu (cm)
            
        Returns
        -------
        float
            Độ dốc liều trung bình (Gy/cm)
            
        References
        ----------
        Wagner, T.H. et al. (2003) A simple dose-gradient measurement tool to complement
        conformity index. J Neurosurg., 99(Suppl 3), 22-28.
        """
        if r_eff <= 0:
            raise ValueError("Bán kính hiệu dụng phải lớn hơn 0")
            
        return d_ref / r_eff

    @staticmethod
    def maximum_dose_gradient(dose_array: np.ndarray, spacing: Tuple[float, float, float]) -> float:
        """
        Tính toán độ dốc liều tối đa dựa trên dữ liệu liều 3D
        
        Parameters
        ----------
        dose_array : np.ndarray
            Mảng 3D chứa dữ liệu liều
        spacing : Tuple[float, float, float]
            Khoảng cách voxel theo ba trục (mm)
            
        Returns
        -------
        float
            Độ dốc liều tối đa (Gy/mm)
        """
        # Tính gradient theo 3 hướng
        gradient = np.gradient(dose_array, *spacing)
        
        # Tính độ lớn của gradient
        magnitude = np.sqrt(gradient[0]**2 + gradient[1]**2 + gradient[2]**2)
        
        # Trả về giá trị lớn nhất
        return np.max(magnitude)

    @staticmethod
    def volume_ratio_gradient(v1: float, v2: float, d1: float, d2: float) -> float:
        """
        Tính toán chỉ số gradient dựa trên tỷ lệ thể tích giữa hai mức liều: VRG = (V1/V2) / (D1/D2)
        
        Trong đó:
        - V1, V2 là thể tích tại hai mức liều D1, D2
        - D1 > D2
        
        Parameters
        ----------
        v1 : float
            Thể tích tại mức liều D1 (cm³)
        v2 : float
            Thể tích tại mức liều D2 (cm³), với V2 > V1
        d1 : float
            Mức liều cao hơn (Gy hoặc %)
        d2 : float
            Mức liều thấp hơn (Gy hoặc %), với D2 < D1
            
        Returns
        -------
        float
            Chỉ số gradient dựa trên tỷ lệ thể tích
        """
        if v1 <= 0 or v2 <= 0:
            raise ValueError("Các thể tích phải lớn hơn 0")
        if d1 <= d2:
            raise ValueError("D1 phải lớn hơn D2")
        if v1 >= v2:
            raise ValueError("V2 phải lớn hơn V1")
            
        return (v1/v2) / (d1/d2)

    @staticmethod
    def distance_gradient(r_ref: float, r_half: float) -> float:
        """
        Tính toán chỉ số gradient dựa trên khoảng cách: DG = r_half - r_ref
        
        Trong đó:
        - r_ref là bán kính tương đương của thể tích tham chiếu
        - r_half là bán kính tương đương của thể tích nửa liều
        
        Giá trị thấp hơn cho thấy độ dốc liều tốt hơn.
        
        Parameters
        ----------
        r_ref : float
            Bán kính tương đương của thể tích tham chiếu (cm)
        r_half : float
            Bán kính tương đương của thể tích nửa liều (cm)
            
        Returns
        -------
        float
            Chỉ số gradient dựa trên khoảng cách (cm)
        """
        if r_ref <= 0 or r_half <= 0:
            raise ValueError("Các bán kính phải lớn hơn 0")
        if r_half <= r_ref:
            raise ValueError("r_half phải lớn hơn r_ref")
            
        return r_half - r_ref

    @staticmethod
    def calculate_equivalent_radius(volume: float) -> float:
        """
        Tính bán kính tương đương của một thể tích, giả định thể tích là hình cầu
        
        Parameters
        ----------
        volume : float
            Thể tích (cm³)
            
        Returns
        -------
        float
            Bán kính tương đương (cm)
        """
        if volume <= 0:
            raise ValueError("Thể tích phải lớn hơn 0")
            
        return ((3 * volume) / (4 * np.pi)) ** (1/3)

    @staticmethod
    def interpret_gi_paddick(gi_value: float) -> str:
        """
        Diễn giải chỉ số gradient Paddick
        
        Parameters
        ----------
        gi_value : float
            Giá trị chỉ số gradient Paddick
            
        Returns
        -------
        str
            Diễn giải chỉ số
        """
        if gi_value < 3.0:
            return "Rất tốt (< 3.0)"
        elif 3.0 <= gi_value < 3.5:
            return "Tốt (3.0 - 3.5)"
        elif 3.5 <= gi_value < 4.0:
            return "Chấp nhận được (3.5 - 4.0)"
        elif 4.0 <= gi_value < 4.5:
            return "Kém (4.0 - 4.5)"
        else:
            return "Rất kém (≥ 4.5)"

    @staticmethod
    def calculate_all_metrics(v_ref: float, v_half: float, d_ref: float = None,
                             r_eff: float = None) -> Dict[str, float]:
        """
        Tính toán các chỉ số gradient phổ biến và trả về dưới dạng từ điển
        
        Parameters
        ----------
        v_ref : float
            Thể tích nhận đầy đủ liều tham chiếu (cm³)
        v_half : float
            Thể tích nhận một nửa liều tham chiếu (cm³)
        d_ref : float, optional
            Liều tham chiếu (Gy)
        r_eff : float, optional
            Bán kính hiệu dụng ngoài thể tích tham chiếu (cm)
            
        Returns
        -------
        Dict[str, float]
            Từ điển chứa các chỉ số gradient đã tính
        """
        metrics = {}
        
        try:
            # Chỉ số gradient Paddick
            metrics["GI_Paddick"] = GradientIndices.gi_paddick(v_half, v_ref)
            
            # Chỉ số gradient dạng bán kính
            metrics["GI_Siyong"] = GradientIndices.gi_siyong(v_half, v_ref)
            
            # Tính bán kính tương đương
            r_ref = GradientIndices.calculate_equivalent_radius(v_ref)
            r_half = GradientIndices.calculate_equivalent_radius(v_half)
            
            # Chỉ số gradient dựa trên khoảng cách
            metrics["DG"] = GradientIndices.distance_gradient(r_ref, r_half)
            
            # Nếu có thông tin liều và bán kính hiệu dụng
            if d_ref is not None and r_eff is not None:
                metrics["ADG"] = GradientIndices.average_dose_gradient(v_ref, d_ref, r_eff)
                
        except ValueError as e:
            # Thêm thông báo lỗi vào kết quả
            metrics["error"] = str(e)
            
        return metrics

    @staticmethod
    def gi_from_dose_volume_histogram(
        dvh_data: Dict[str, np.ndarray], 
        target_name: str, 
        ref_dose_percent: float = 100.0, 
        half_dose_percent: float = 50.0
    ) -> float:
        """
        Tính toán chỉ số gradient từ dữ liệu DVH
        
        Parameters
        ----------
        dvh_data : Dict[str, np.ndarray]
            Từ điển chứa dữ liệu DVH với khóa là tên cấu trúc,
            giá trị là mảng 2D (liều, thể tích)
        target_name : str
            Tên cấu trúc mục tiêu
        ref_dose_percent : float, optional
            Phần trăm liều tham chiếu (mặc định: 100%)
        half_dose_percent : float, optional
            Phần trăm liều nửa tham chiếu (mặc định: 50%)
            
        Returns
        -------
        float
            Chỉ số gradient Paddick
            
        Raises
        ------
        ValueError
            Nếu cấu trúc mục tiêu không có trong dữ liệu DVH
        """
        if target_name not in dvh_data:
            raise ValueError(f"Cấu trúc mục tiêu '{target_name}' không có trong dữ liệu DVH")
            
        doses, volumes = dvh_data[target_name]
        
        # Tìm thể tích tại liều tham chiếu
        idx_ref = np.argmin(np.abs(doses - ref_dose_percent))
        v_ref = volumes[idx_ref]
        
        # Tìm thể tích tại nửa liều tham chiếu
        idx_half = np.argmin(np.abs(doses - half_dose_percent))
        v_half = volumes[idx_half]
        
        return GradientIndices.gi_paddick(v_half, v_ref)
