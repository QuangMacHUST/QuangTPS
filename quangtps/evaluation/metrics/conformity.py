#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module chứa các chỉ số đồng dạng (Conformity Index) cho đánh giá kế hoạch xạ trị.
"""

import numpy as np
from typing import Dict, Union, Tuple, List, Optional


class ConformityIndices:
    """
    Lớp tính toán các chỉ số đồng dạng (Conformity Index) để đánh giá mức độ
    phù hợp của phân phối liều với thể tích mục tiêu.
    """

    @staticmethod
    def ci_rtog(v_ri: float, tv: float) -> float:
        """
        Tính toán chỉ số đồng dạng RTOG: CI = V_RI / TV
        
        Trong đó:
        - V_RI là thể tích tham chiếu phủ bởi mức liều quy định (ví dụ: 95% liều)
        - TV là thể tích mục tiêu (Target Volume)
        
        Giá trị lý tưởng = 1
        CI < 1: thiếu phủ liều
        CI > 1: thừa phủ liều
        
        Parameters
        ----------
        v_ri : float
            Thể tích tham chiếu phủ bởi mức liều quy định (cm³)
        tv : float
            Thể tích mục tiêu (cm³)
            
        Returns
        -------
        float
            Chỉ số đồng dạng RTOG
            
        References
        ----------
        Shaw, E. et al. (1993) Radiation Therapy Oncology Group: radiosurgery quality
        assurance guidelines. Int J Radiat Oncol Biol Phys., 27(5), 1231-1239.
        """
        if tv <= 0 or v_ri <= 0:
            raise ValueError("Thể tích mục tiêu (TV) và thể tích tham chiếu (V_RI) phải lớn hơn 0")
        
        return v_ri / tv

    @staticmethod
    def ci_paddick(tv_ri: float, tv: float, v_ri: float) -> float:
        """
        Tính toán chỉ số đồng dạng Paddick: CI = (TV_RI)² / (TV * V_RI)
        
        Trong đó:
        - TV_RI là thể tích mục tiêu phủ bởi mức liều quy định
        - TV là thể tích mục tiêu
        - V_RI là toàn bộ thể tích phủ bởi mức liều quy định
        
        Giá trị lý tưởng = 1
        Giá trị thấp chỉ ra mức độ đồng dạng kém
        
        Parameters
        ----------
        tv_ri : float
            Thể tích mục tiêu phủ bởi mức liều quy định (cm³)
        tv : float
            Thể tích mục tiêu tổng (cm³)
        v_ri : float
            Toàn bộ thể tích phủ bởi mức liều quy định (cm³)
            
        Returns
        -------
        float
            Chỉ số đồng dạng Paddick
            
        References
        ----------
        Paddick, I. (2000) A simple scoring ratio to index the conformity of radiosurgical
        treatment plans. J Neurosurg., 93(Suppl 3), 219-222.
        """
        if tv <= 0 or v_ri <= 0 or tv_ri <= 0:
            raise ValueError("Tất cả các thể tích phải lớn hơn 0")
            
        return (tv_ri ** 2) / (tv * v_ri)

    @staticmethod
    def ci_lomax(tv_ri: float, tv: float, v_ri: float) -> float:
        """
        Tính toán chỉ số đồng dạng Lomax (còn gọi là chỉ số đồng dạng COIN):
        CI = (TV_RI/TV) * (TV_RI/V_RI)
        
        Trong đó:
        - TV_RI là thể tích mục tiêu phủ bởi mức liều quy định
        - TV là thể tích mục tiêu
        - V_RI là toàn bộ thể tích phủ bởi mức liều quy định
        
        Giá trị lý tưởng = 1
        Giá trị thấp chỉ ra mức độ đồng dạng kém
        
        Parameters
        ----------
        tv_ri : float
            Thể tích mục tiêu phủ bởi mức liều quy định (cm³)
        tv : float
            Thể tích mục tiêu tổng (cm³)
        v_ri : float
            Toàn bộ thể tích phủ bởi mức liều quy định (cm³)
            
        Returns
        -------
        float
            Chỉ số đồng dạng Lomax (COIN)
            
        References
        ----------
        Lomax, N.J., Scheib, S.G. (2003) Quantifying the degree of conformity in radiosurgery
        treatment planning. Int J Radiat Oncol Biol Phys., 55(5), 1409-1419.
        """
        if tv <= 0 or v_ri <= 0 or tv_ri <= 0:
            raise ValueError("Tất cả các thể tích phải lớn hơn 0")
            
        return (tv_ri / tv) * (tv_ri / v_ri)

    @staticmethod
    def ci_van_t_riet(tv_ri: float, tv: float, v_ri: float) -> float:
        """
        Tính toán chỉ số đồng dạng van't Riet (còn gọi là chỉ số đồng dạng mới - nCI):
        CI = (TV_RI/TV) * (TV_RI/V_RI)
        
        Lưu ý: Công thức giống Lomax nhưng được đề xuất độc lập
        
        Trong đó:
        - TV_RI là thể tích mục tiêu phủ bởi mức liều quy định
        - TV là thể tích mục tiêu
        - V_RI là toàn bộ thể tích phủ bởi mức liều quy định
        
        Giá trị lý tưởng = 1
        Giá trị thấp chỉ ra mức độ đồng dạng kém
        
        Parameters
        ----------
        tv_ri : float
            Thể tích mục tiêu phủ bởi mức liều quy định (cm³)
        tv : float
            Thể tích mục tiêu tổng (cm³)
        v_ri : float
            Toàn bộ thể tích phủ bởi mức liều quy định (cm³)
            
        Returns
        -------
        float
            Chỉ số đồng dạng van't Riet
            
        References
        ----------
        van't Riet, A. et al. (1997) A conformation number to quantify the degree of
        conformality in brachytherapy and external beam irradiation: application to the
        prostate. Int J Radiat Oncol Biol Phys., 37(3), 731-736.
        """
        if tv <= 0 or v_ri <= 0 or tv_ri <= 0:
            raise ValueError("Tất cả các thể tích phải lớn hơn 0")
            
        return (tv_ri / tv) * (tv_ri / v_ri)

    @staticmethod
    def ci_ohtakara(tv_ri: float, tv: float, v_ri: float) -> float:
        """
        Tính toán chỉ số đồng dạng Ohtakara (mCI):
        CI = (TV_RI/TV + TV_RI/V_RI) / 2
        
        Trong đó:
        - TV_RI là thể tích mục tiêu phủ bởi mức liều quy định
        - TV là thể tích mục tiêu
        - V_RI là toàn bộ thể tích phủ bởi mức liều quy định
        
        Giá trị lý tưởng = 1
        Giá trị thấp chỉ ra mức độ đồng dạng kém
        
        Parameters
        ----------
        tv_ri : float
            Thể tích mục tiêu phủ bởi mức liều quy định (cm³)
        tv : float
            Thể tích mục tiêu tổng (cm³)
        v_ri : float
            Toàn bộ thể tích phủ bởi mức liều quy định (cm³)
            
        Returns
        -------
        float
            Chỉ số đồng dạng Ohtakara
            
        References
        ----------
        Ohtakara, K. et al. (2012) Clinical and statistical analysis of the correlation between
        conformity indices and dosimetric variables for gamma knife radiosurgery.
        Acta Neurochir Suppl., 113, 17-23.
        """
        if tv <= 0 or v_ri <= 0 or tv_ri <= 0:
            raise ValueError("Tất cả các thể tích phải lớn hơn 0")
            
        return (tv_ri / tv + tv_ri / v_ri) / 2

    @staticmethod
    def ci_knoos(tv_ri: float, tv: float) -> float:
        """
        Tính toán chỉ số đồng dạng Knöös (TC - Target Coverage):
        CI = TV_RI / TV
        
        Trong đó:
        - TV_RI là thể tích mục tiêu phủ bởi mức liều quy định
        - TV là thể tích mục tiêu
        
        Giá trị lý tưởng = 1
        Giá trị thấp chỉ ra mức độ phủ mục tiêu kém
        
        Parameters
        ----------
        tv_ri : float
            Thể tích mục tiêu phủ bởi mức liều quy định (cm³)
        tv : float
            Thể tích mục tiêu tổng (cm³)
            
        Returns
        -------
        float
            Chỉ số đồng dạng Knöös (Target Coverage)
            
        References
        ----------
        Knöös, T. et al. (1998) Volumetric and dosimetric evaluation of radiation treatment
        plans: radiation conformity index. Int J Radiat Oncol Biol Phys., 42(5), 1169-1176.
        """
        if tv <= 0 or tv_ri <= 0:
            raise ValueError("Tất cả các thể tích phải lớn hơn 0")
            
        return tv_ri / tv

    @staticmethod
    def ci_dice(tv_ri: float, tv: float, v_ri: float) -> float:
        """
        Tính toán hệ số Dice:
        Dice = 2*TV_RI / (TV + V_RI)
        
        Trong đó:
        - TV_RI là thể tích mục tiêu phủ bởi mức liều quy định
        - TV là thể tích mục tiêu
        - V_RI là toàn bộ thể tích phủ bởi mức liều quy định
        
        Giá trị lý tưởng = 1
        Giá trị thấp chỉ ra mức độ chồng lấn kém
        
        Parameters
        ----------
        tv_ri : float
            Thể tích mục tiêu phủ bởi mức liều quy định (cm³)
        tv : float
            Thể tích mục tiêu tổng (cm³)
        v_ri : float
            Toàn bộ thể tích phủ bởi mức liều quy định (cm³)
            
        Returns
        -------
        float
            Hệ số Dice
            
        References
        ----------
        Dice, L.R. (1945) Measures of the amount of ecologic association between species.
        Ecology, 26(3), 297-302.
        """
        if tv <= 0 or v_ri <= 0 or tv_ri <= 0:
            raise ValueError("Tất cả các thể tích phải lớn hơn 0")
            
        return 2 * tv_ri / (tv + v_ri)

    @staticmethod
    def ci_jaccard(tv_ri: float, tv: float, v_ri: float) -> float:
        """
        Tính toán chỉ số Jaccard:
        Jaccard = TV_RI / (TV + V_RI - TV_RI)
        
        Trong đó:
        - TV_RI là thể tích mục tiêu phủ bởi mức liều quy định
        - TV là thể tích mục tiêu
        - V_RI là toàn bộ thể tích phủ bởi mức liều quy định
        
        Giá trị lý tưởng = 1
        Giá trị thấp chỉ ra mức độ chồng lấn kém
        
        Parameters
        ----------
        tv_ri : float
            Thể tích mục tiêu phủ bởi mức liều quy định (cm³)
        tv : float
            Thể tích mục tiêu tổng (cm³)
        v_ri : float
            Toàn bộ thể tích phủ bởi mức liều quy định (cm³)
            
        Returns
        -------
        float
            Chỉ số Jaccard
            
        References
        ----------
        Jaccard, P. (1901) Étude comparative de la distribution florale dans une portion
        des Alpes et des Jura. Bulletin de la Société Vaudoise des Sciences Naturelles, 37, 547-579.
        """
        if tv <= 0 or v_ri <= 0 or tv_ri <= 0:
            raise ValueError("Tất cả các thể tích phải lớn hơn 0")
            
        # Kiểm tra điều kiện bổ sung
        if (tv + v_ri - tv_ri) <= 0:
            raise ValueError("Phép tính bị lỗi: (TV + V_RI - TV_RI) phải lớn hơn 0")
            
        return tv_ri / (tv + v_ri - tv_ri)

    @staticmethod
    def calculate_all_metrics(tv_ri: float, tv: float, v_ri: float) -> Dict[str, float]:
        """
        Tính toán tất cả các chỉ số đồng dạng và trả về dưới dạng từ điển
        
        Parameters
        ----------
        tv_ri : float
            Thể tích mục tiêu phủ bởi mức liều quy định (cm³)
        tv : float
            Thể tích mục tiêu tổng (cm³)
        v_ri : float
            Toàn bộ thể tích phủ bởi mức liều quy định (cm³)
            
        Returns
        -------
        Dict[str, float]
            Từ điển chứa tất cả các chỉ số đồng dạng đã tính
        """
        try:
            metrics = {
                "CI_RTOG": ConformityIndices.ci_rtog(v_ri, tv),
                "CI_Paddick": ConformityIndices.ci_paddick(tv_ri, tv, v_ri),
                "CI_Lomax": ConformityIndices.ci_lomax(tv_ri, tv, v_ri),
                "CI_vanTRiet": ConformityIndices.ci_van_t_riet(tv_ri, tv, v_ri),
                "CI_Ohtakara": ConformityIndices.ci_ohtakara(tv_ri, tv, v_ri),
                "CI_Knoos": ConformityIndices.ci_knoos(tv_ri, tv),
                "Dice": ConformityIndices.ci_dice(tv_ri, tv, v_ri),
                "Jaccard": ConformityIndices.ci_jaccard(tv_ri, tv, v_ri)
            }
            return metrics
        except ValueError as e:
            # Trả về từ điển với thông báo lỗi
            return {"error": str(e)}

    @staticmethod
    def get_rtog_evaluation(ci_rtog: float) -> str:
        """
        Đánh giá kế hoạch dựa trên chỉ số đồng dạng RTOG
        
        Theo hướng dẫn của RTOG:
        - CI = 1: Phù hợp hoàn hảo
        - 1 < CI < 2: Vi phạm nhỏ, chấp nhận được
        - 2 ≤ CI < 2.5: Vi phạm lớn, cần xem xét kỹ
        - CI ≥ 2.5: Vi phạm không chấp nhận được
        
        Parameters
        ----------
        ci_rtog : float
            Chỉ số đồng dạng RTOG
            
        Returns
        -------
        str
            Đánh giá dựa trên chỉ số RTOG
        """
        if ci_rtog < 1:
            return "Thiếu phủ liều (CI < 1)"
        elif ci_rtog == 1:
            return "Phù hợp hoàn hảo (CI = 1)"
        elif 1 < ci_rtog < 2:
            return "Vi phạm nhỏ, chấp nhận được (1 < CI < 2)"
        elif 2 <= ci_rtog < 2.5:
            return "Vi phạm lớn, cần xem xét kỹ (2 ≤ CI < 2.5)"
        else:
            return "Vi phạm không chấp nhận được (CI ≥ 2.5)"

    @staticmethod
    def interpret_paddick(ci_paddick: float) -> str:
        """
        Diễn giải chỉ số đồng dạng Paddick
        
        Parameters
        ----------
        ci_paddick : float
            Chỉ số đồng dạng Paddick
            
        Returns
        -------
        str
            Diễn giải chỉ số
        """
        if ci_paddick > 0.9:
            return "Rất tốt (> 0.9)"
        elif 0.8 <= ci_paddick <= 0.9:
            return "Tốt (0.8 - 0.9)"
        elif 0.6 <= ci_paddick < 0.8:
            return "Khá (0.6 - 0.8)"
        elif 0.4 <= ci_paddick < 0.6:
            return "Trung bình (0.4 - 0.6)"
        else:
            return "Kém (< 0.4)"
