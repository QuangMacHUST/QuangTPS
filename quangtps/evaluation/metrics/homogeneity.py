#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module chứa các chỉ số đồng nhất (Homogeneity Index) cho đánh giá kế hoạch xạ trị.
"""

import numpy as np
from typing import Dict, Union, Tuple, List, Optional


class HomogeneityIndices:
    """
    Lớp tính toán các chỉ số đồng nhất (Homogeneity Index) để đánh giá mức độ
    đồng đều của phân phối liều trong thể tích mục tiêu.
    """

    @staticmethod
    def hi_icru83(d2: float, d98: float, d50: float) -> float:
        """
        Tính toán chỉ số đồng nhất theo ICRU Report 83: HI = (D2% - D98%) / D50%
        
        Trong đó:
        - D2% là liều nhận bởi 2% thể tích mục tiêu (gần với liều tối đa)
        - D98% là liều nhận bởi 98% thể tích mục tiêu (gần với liều tối thiểu)
        - D50% là liều nhận bởi 50% thể tích mục tiêu (liều trung vị)
        
        Giá trị lý tưởng = 0, cho thấy phân phối liều hoàn toàn đồng nhất.
        
        Parameters
        ----------
        d2 : float
            Liều nhận bởi 2% thể tích mục tiêu (Gy hoặc %)
        d98 : float
            Liều nhận bởi 98% thể tích mục tiêu (Gy hoặc %)
        d50 : float
            Liều nhận bởi 50% thể tích mục tiêu (Gy hoặc %)
            
        Returns
        -------
        float
            Chỉ số đồng nhất ICRU 83
            
        References
        ----------
        ICRU (2010) Report 83: Prescribing, Recording, and Reporting Photon-Beam 
        Intensity-Modulated Radiation Therapy (IMRT). Journal of the ICRU, 10(1).
        """
        if d50 <= 0:
            raise ValueError("D50% phải lớn hơn 0")
            
        return (d2 - d98) / d50

    @staticmethod
    def hi_icru83_with_d50(d2: float, d98: float, d50: float) -> float:
        """
        Tính toán chỉ số đồng nhất đầy đủ theo ICRU Report 83: HI = (D2% - D98%) / D50%
        
        Parameters
        ----------
        d2 : float
            Liều nhận bởi 2% thể tích mục tiêu (Gy hoặc %)
        d98 : float
            Liều nhận bởi 98% thể tích mục tiêu (Gy hoặc %)
        d50 : float
            Liều nhận bởi 50% thể tích mục tiêu (Gy hoặc %)
            
        Returns
        -------
        float
            Chỉ số đồng nhất ICRU 83
        """
        if d50 <= 0:
            raise ValueError("D50% phải lớn hơn 0")
            
        return (d2 - d98) / d50

    @staticmethod
    def hi_icru62(d_max: float, d_min: float, d_ref: float) -> float:
        """
        Tính toán chỉ số đồng nhất theo ICRU Report 62
        
        Trong đó:
        - D_max là liều tối đa trong thể tích mục tiêu
        - D_min là liều tối thiểu trong thể tích mục tiêu
        - D_ref là liều tham chiếu (thường là liều kê)
        
        Giá trị lý tưởng = 0, cho thấy phân phối liều hoàn toàn đồng nhất.
        
        Parameters
        ----------
        d_max : float
            Liều tối đa trong thể tích mục tiêu (Gy hoặc %)
        d_min : float
            Liều tối thiểu trong thể tích mục tiêu (Gy hoặc %)
        d_ref : float
            Liều tham chiếu (Gy hoặc %)
            
        Returns
        -------
        float
            Chỉ số đồng nhất ICRU 62
            
        References
        ----------
        ICRU (1999) Report 62: Prescribing, Recording, and Reporting Photon Beam Therapy
        (Supplement to ICRU Report 50). Journal of the ICRU.
        """
        if d_ref <= 0:
            raise ValueError("Liều tham chiếu (D_ref) phải lớn hơn 0")
            
        return (d_max - d_min) / d_ref

    @staticmethod
    def hi_rtog(d_max: float, d_ref: float) -> float:
        """
        Tính toán chỉ số đồng nhất theo định nghĩa của RTOG: HI = D_max / D_ref
        
        Trong đó:
        - D_max là liều tối đa trong thể tích mục tiêu
        - D_ref là liều tham chiếu (thường là liều kê)
        
        Giá trị lý tưởng = 1, các giá trị lớn hơn cho thấy tính không đồng nhất tăng lên.
        
        Parameters
        ----------
        d_max : float
            Liều tối đa trong thể tích mục tiêu (Gy hoặc %)
        d_ref : float
            Liều tham chiếu (Gy hoặc %)
            
        Returns
        -------
        float
            Chỉ số đồng nhất RTOG
            
        References
        ----------
        Shaw, E. et al. (1993) Radiation Therapy Oncology Group: radiosurgery quality
        assurance guidelines. Int J Radiat Oncol Biol Phys., 27(5), 1231-1239.
        """
        if d_ref <= 0:
            raise ValueError("Liều tham chiếu (D_ref) phải lớn hơn 0")
            
        return d_max / d_ref

    @staticmethod
    def hi_immordino(d5: float, d95: float) -> float:
        """
        Tính toán chỉ số đồng nhất theo Immordino et al.: HI = D5% / D95%
        
        Trong đó:
        - D5% là liều nhận bởi 5% thể tích mục tiêu
        - D95% là liều nhận bởi 95% thể tích mục tiêu
        
        Giá trị lý tưởng = 1, các giá trị lớn hơn cho thấy tính không đồng nhất tăng lên.
        
        Parameters
        ----------
        d5 : float
            Liều nhận bởi 5% thể tích mục tiêu (Gy hoặc %)
        d95 : float
            Liều nhận bởi 95% thể tích mục tiêu (Gy hoặc %)
            
        Returns
        -------
        float
            Chỉ số đồng nhất Immordino
            
        References
        ----------
        Immordino, L. et al. (2016) Plan quality and homogeneity index for small volume
        single fraction stereotactic radiotherapy. Physica Medica, 32, 261.
        """
        if d95 <= 0:
            raise ValueError("D95% phải lớn hơn 0")
            
        return d5 / d95

    @staticmethod
    def hi_gunderson(d5: float, d95: float, d_ref: float) -> float:
        """
        Tính toán chỉ số đồng nhất theo Gunderson & Tepper: HI = (D5% - D95%) / D_ref
        
        Trong đó:
        - D5% là liều nhận bởi 5% thể tích mục tiêu
        - D95% là liều nhận bởi 95% thể tích mục tiêu
        - D_ref là liều tham chiếu (thường là liều kê)
        
        Giá trị lý tưởng = 0, cho thấy phân phối liều hoàn toàn đồng nhất.
        
        Parameters
        ----------
        d5 : float
            Liều nhận bởi 5% thể tích mục tiêu (Gy hoặc %)
        d95 : float
            Liều nhận bởi 95% thể tích mục tiêu (Gy hoặc %)
        d_ref : float
            Liều tham chiếu (Gy hoặc %)
            
        Returns
        -------
        float
            Chỉ số đồng nhất Gunderson & Tepper
            
        References
        ----------
        Gunderson, L.L., Tepper, J.E. (2015) Clinical Radiation Oncology, 4th Edition.
        Elsevier.
        """
        if d_ref <= 0:
            raise ValueError("Liều tham chiếu (D_ref) phải lớn hơn 0")
            
        return (d5 - d95) / d_ref

    @staticmethod
    def hi_wu(d5: float, d95: float, d_mean: float) -> float:
        """
        Tính toán chỉ số đồng nhất theo Wu et al.: HI = (D5% - D95%) / D_prescription
        Công thức đơn giản hóa thành: HI = (D5% - D95%) / D_mean
        
        Trong đó:
        - D5% là liều nhận bởi 5% thể tích mục tiêu
        - D95% là liều nhận bởi 95% thể tích mục tiêu
        - D_mean là liều trung bình trong thể tích mục tiêu (thay thế D_prescription)
        
        Giá trị lý tưởng = 0, cho thấy phân phối liều hoàn toàn đồng nhất.
        
        Parameters
        ----------
        d5 : float
            Liều nhận bởi 5% thể tích mục tiêu (Gy hoặc %)
        d95 : float
            Liều nhận bởi 95% thể tích mục tiêu (Gy hoặc %)
        d_mean : float
            Liều trung bình (Gy hoặc %)
            
        Returns
        -------
        float
            Chỉ số đồng nhất Wu
            
        References
        ----------
        Wu, Q. et al. (2003) Optimization of intensity-modulated radiotherapy plans based
        on the equivalent uniform dose. Int J Radiat Oncol Biol Phys., 56(1), 224-235.
        """
        if d_mean <= 0:
            raise ValueError("Liều trung bình (D_mean) phải lớn hơn 0")
            
        return (d5 - d95) / d_mean

    @staticmethod
    def hi_wu_with_mean(d5: float, d95: float, d_mean: float) -> float:
        """
        Tính toán chỉ số đồng nhất đầy đủ theo Wu et al. với liều trung bình cụ thể
        
        Parameters
        ----------
        d5 : float
            Liều nhận bởi 5% thể tích mục tiêu (Gy hoặc %)
        d95 : float
            Liều nhận bởi 95% thể tích mục tiêu (Gy hoặc %)
        d_mean : float
            Liều trung bình (Gy hoặc %)
            
        Returns
        -------
        float
            Chỉ số đồng nhất Wu
        """
        if d_mean <= 0:
            raise ValueError("Liều trung bình (D_mean) phải lớn hơn 0")
            
        return (d5 - d95) / d_mean

    @staticmethod
    def hi_kataria(d2: float, d98: float) -> float:
        """
        Tính toán chỉ số đồng nhất theo Kataria et al.: HI = (D2% - D98%) / D50%
        Công thức đơn giản hóa bằng cách bỏ qua D50%
        
        Trong đó:
        - D2% là liều nhận bởi 2% thể tích mục tiêu
        - D98% là liều nhận bởi 98% thể tích mục tiêu
        
        Giá trị thấp hơn cho thấy phân phối liều đồng nhất hơn.
        
        Parameters
        ----------
        d2 : float
            Liều nhận bởi 2% thể tích mục tiêu (Gy hoặc %)
        d98 : float
            Liều nhận bởi 98% thể tích mục tiêu (Gy hoặc %)
            
        Returns
        -------
        float
            Chỉ số đồng nhất Kataria (đơn giản hóa)
            
        References
        ----------
        Kataria, T. et al. (2012) Homogeneity Index: An objective tool for assessment of
        conformal radiation treatments. J Med Phys., 37(4), 207-213.
        """
        return d2 - d98

    @staticmethod
    def hi_kataria_with_d50(d2: float, d98: float, d50: float) -> float:
        """
        Tính toán chỉ số đồng nhất đầy đủ theo Kataria et al.: HI = (D2% - D98%) / D50%
        
        Parameters
        ----------
        d2 : float
            Liều nhận bởi 2% thể tích mục tiêu (Gy hoặc %)
        d98 : float
            Liều nhận bởi 98% thể tích mục tiêu (Gy hoặc %)
        d50 : float
            Liều nhận bởi 50% thể tích mục tiêu (Gy hoặc %)
            
        Returns
        -------
        float
            Chỉ số đồng nhất Kataria (đầy đủ)
        """
        if d50 <= 0:
            raise ValueError("D50% phải lớn hơn 0")
            
        return (d2 - d98) / d50

    @staticmethod
    def calculate_all_metrics(d_max: float, d_min: float, d2: float, d5: float, 
                             d50: float, d95: float, d98: float, d_ref: float, 
                             d_mean: float) -> Dict[str, float]:
        """
        Tính toán tất cả các chỉ số đồng nhất và trả về dưới dạng từ điển
        
        Parameters
        ----------
        d_max : float
            Liều tối đa trong thể tích mục tiêu (Gy hoặc %)
        d_min : float
            Liều tối thiểu trong thể tích mục tiêu (Gy hoặc %)
        d2 : float
            Liều nhận bởi 2% thể tích mục tiêu (Gy hoặc %)
        d5 : float
            Liều nhận bởi 5% thể tích mục tiêu (Gy hoặc %)
        d50 : float
            Liều nhận bởi 50% thể tích mục tiêu (Gy hoặc %)
        d95 : float
            Liều nhận bởi 95% thể tích mục tiêu (Gy hoặc %)
        d98 : float
            Liều nhận bởi 98% thể tích mục tiêu (Gy hoặc %)
        d_ref : float
            Liều tham chiếu (Gy hoặc %)
        d_mean : float
            Liều trung bình (Gy hoặc %)
            
        Returns
        -------
        Dict[str, float]
            Từ điển chứa tất cả các chỉ số đồng nhất đã tính
        """
        try:
            metrics = {
                "HI_ICRU83": HomogeneityIndices.hi_icru83_with_d50(d2, d98, d50),
                "HI_ICRU62": HomogeneityIndices.hi_icru62(d_max, d_min, d_ref),
                "HI_RTOG": HomogeneityIndices.hi_rtog(d_max, d_ref),
                "HI_Immordino": HomogeneityIndices.hi_immordino(d5, d95),
                "HI_Gunderson": HomogeneityIndices.hi_gunderson(d5, d95, d_ref),
                "HI_Wu": HomogeneityIndices.hi_wu_with_mean(d5, d95, d_mean),
                "HI_Kataria": HomogeneityIndices.hi_kataria_with_d50(d2, d98, d50)
            }
            return metrics
        except ValueError as e:
            # Trả về từ điển với thông báo lỗi
            return {"error": str(e)}

    @staticmethod
    def interpret_icru83(hi_value: float) -> str:
        """
        Diễn giải chỉ số đồng nhất ICRU83
        
        Parameters
        ----------
        hi_value : float
            Giá trị chỉ số đồng nhất ICRU83
            
        Returns
        -------
        str
            Diễn giải chỉ số
        """
        if hi_value <= 0.05:
            return "Rất tốt (≤ 0.05)"
        elif 0.05 < hi_value <= 0.1:
            return "Tốt (0.05 - 0.1)"
        elif 0.1 < hi_value <= 0.2:
            return "Chấp nhận được (0.1 - 0.2)"
        elif 0.2 < hi_value <= 0.3:
            return "Kém (0.2 - 0.3)"
        else:
            return "Rất kém (> 0.3)"
    
    @staticmethod
    def interpret_rtog(hi_value: float) -> str:
        """
        Diễn giải chỉ số đồng nhất RTOG
        
        Parameters
        ----------
        hi_value : float
            Giá trị chỉ số đồng nhất RTOG
            
        Returns
        -------
        str
            Diễn giải chỉ số
        """
        if hi_value <= 1.05:
            return "Rất tốt (≤ 1.05)"
        elif 1.05 < hi_value <= 1.1:
            return "Tốt (1.05 - 1.1)"
        elif 1.1 < hi_value <= 1.2:
            return "Chấp nhận được (1.1 - 1.2)"
        elif 1.2 < hi_value <= 1.5:
            return "Kém (1.2 - 1.5)"
        else:
            return "Rất kém (> 1.5)"
