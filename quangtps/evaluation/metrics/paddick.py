#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module chứa các phương thức tính toán chỉ số Paddick cho đánh giá kế hoạch xạ trị.
Chỉ số Paddick phổ biến trong đánh giá kế hoạch xạ phẫu (SRS).
"""

import numpy as np
from typing import Dict, Union, Tuple, List, Optional


class PaddickIndices:
    """
    Lớp tính toán chỉ số Paddick và các biến thể của nó để đánh giá mức độ
    đồng dạng và chất lượng kế hoạch xạ trị.
    """

    @staticmethod
    def ci_paddick(tv_ri: float, tv: float, v_ri: float) -> float:
        """
        Tính toán chỉ số đồng dạng Paddick (PCI): CI = (TV_RI)² / (TV * V_RI)

        Trong đó:
        - TV_RI là thể tích mục tiêu phủ bởi mức liều quy định
        - TV là thể tích mục tiêu
        - V_RI là toàn bộ thể tích phủ bởi mức liều quy định

        Giá trị lý tưởng = 1 cho thấy sự phủ mục tiêu hoàn hảo mà không có mô lành
        nhận phóng xạ không cần thiết.
        Giá trị thấp chỉ ra mức độ đồng dạng kém.

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

        return (tv_ri**2) / (tv * v_ri)

    @staticmethod
    def coverage_index(tv_ri: float, tv: float) -> float:
        """
        Tính toán chỉ số phủ (Coverage Index): Coverage = TV_RI / TV

        Trong đó:
        - TV_RI là thể tích mục tiêu phủ bởi mức liều quy định
        - TV là thể tích mục tiêu

        Giá trị lý tưởng = 1, cho thấy 100% thể tích mục tiêu được phủ bởi mức liều quy định.

        Parameters
        ----------
        tv_ri : float
            Thể tích mục tiêu phủ bởi mức liều quy định (cm³)
        tv : float
            Thể tích mục tiêu tổng (cm³)

        Returns
        -------
        float
            Chỉ số phủ
        """
        if tv <= 0 or tv_ri <= 0:
            raise ValueError("Tất cả các thể tích phải lớn hơn 0")

        return tv_ri / tv

    @staticmethod
    def selectivity_index(tv_ri: float, v_ri: float) -> float:
        """
        Tính toán chỉ số chọn lọc (Selectivity Index): Selectivity = TV_RI / V_RI

        Trong đó:
        - TV_RI là thể tích mục tiêu phủ bởi mức liều quy định
        - V_RI là toàn bộ thể tích phủ bởi mức liều quy định

        Giá trị lý tưởng = 1, cho thấy không có mô lành nhận liều không cần thiết.

        Parameters
        ----------
        tv_ri : float
            Thể tích mục tiêu phủ bởi mức liều quy định (cm³)
        v_ri : float
            Toàn bộ thể tích phủ bởi mức liều quy định (cm³)

        Returns
        -------
        float
            Chỉ số chọn lọc
        """
        if v_ri <= 0 or tv_ri <= 0:
            raise ValueError("Tất cả các thể tích phải lớn hơn 0")

        return tv_ri / v_ri

    @staticmethod
    def modified_paddick_ci(tv_ri: float, tv: float, v_ri: float) -> float:
        """
        Tính toán chỉ số Paddick sửa đổi (mPCI): mPCI = (TV_RI/TV + TV_RI/V_RI) / 2

        Trong đó:
        - TV_RI là thể tích mục tiêu phủ bởi mức liều quy định
        - TV là thể tích mục tiêu
        - V_RI là toàn bộ thể tích phủ bởi mức liều quy định

        Đây là trung bình của chỉ số phủ và chỉ số chọn lọc.
        Giá trị lý tưởng = 1

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
            Chỉ số Paddick sửa đổi
        """
        if tv <= 0 or v_ri <= 0 or tv_ri <= 0:
            raise ValueError("Tất cả các thể tích phải lớn hơn 0")

        coverage = tv_ri / tv
        selectivity = tv_ri / v_ri

        return (coverage + selectivity) / 2

    @staticmethod
    def gradient_index(v_half: float, v_ri: float) -> float:
        """
        Tính toán chỉ số gradient Paddick: GI = V_half / V_RI

        Trong đó:
        - V_half là thể tích nhận một nửa liều tham chiếu
        - V_RI là thể tích nhận đầy đủ liều tham chiếu

        Giá trị thấp hơn cho thấy độ dốc liều tốt hơn.

        Parameters
        ----------
        v_half : float
            Thể tích nhận một nửa liều tham chiếu (cm³)
        v_ri : float
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
        if v_ri <= 0 or v_half <= 0:
            raise ValueError("Tất cả các thể tích phải lớn hơn 0")

        return v_half / v_ri

    @staticmethod
    def paddick_metrics_report(
        tv_ri: float, tv: float, v_ri: float, v_half: float = None
    ) -> Dict[str, float]:
        """
        Tính toán và báo cáo tất cả các chỉ số Paddick liên quan

        Parameters
        ----------
        tv_ri : float
            Thể tích mục tiêu phủ bởi mức liều quy định (cm³)
        tv : float
            Thể tích mục tiêu tổng (cm³)
        v_ri : float
            Toàn bộ thể tích phủ bởi mức liều quy định (cm³)
        v_half : float, optional
            Thể tích nhận một nửa liều tham chiếu (cm³)

        Returns
        -------
        Dict[str, float]
            Từ điển chứa tất cả các chỉ số Paddick
        """
        try:
            metrics = {
                "Paddick CI": PaddickIndices.ci_paddick(tv_ri, tv, v_ri),
                "Coverage Index": PaddickIndices.coverage_index(tv_ri, tv),
                "Selectivity Index": PaddickIndices.selectivity_index(tv_ri, v_ri),
                "Modified Paddick CI": PaddickIndices.modified_paddick_ci(
                    tv_ri, tv, v_ri
                ),
            }

            # Nếu có thông tin về thể tích nửa liều
            if v_half is not None:
                metrics["Gradient Index"] = PaddickIndices.gradient_index(v_half, v_ri)

            return metrics
        except ValueError as e:
            return {"error": str(e)}

    @staticmethod
    def interpret_paddick_ci(ci_value: float) -> str:
        """
        Diễn giải chỉ số đồng dạng Paddick

        Parameters
        ----------
        ci_value : float
            Giá trị chỉ số đồng dạng Paddick

        Returns
        -------
        str
            Diễn giải chỉ số
        """
        if ci_value > 0.9:
            return "Rất tốt (> 0.9)"
        elif 0.8 <= ci_value <= 0.9:
            return "Tốt (0.8 - 0.9)"
        elif 0.6 <= ci_value < 0.8:
            return "Khá (0.6 - 0.8)"
        elif 0.4 <= ci_value < 0.6:
            return "Trung bình (0.4 - 0.6)"
        else:
            return "Kém (< 0.4)"

    @staticmethod
    def interpret_coverage(coverage_value: float) -> str:
        """
        Diễn giải chỉ số phủ

        Parameters
        ----------
        coverage_value : float
            Giá trị chỉ số phủ

        Returns
        -------
        str
            Diễn giải chỉ số
        """
        if coverage_value >= 0.98:
            return "Tuyệt vời (≥ 0.98)"
        elif 0.95 <= coverage_value < 0.98:
            return "Rất tốt (0.95 - 0.98)"
        elif 0.90 <= coverage_value < 0.95:
            return "Tốt (0.90 - 0.95)"
        elif 0.80 <= coverage_value < 0.90:
            return "Chấp nhận được (0.80 - 0.90)"
        else:
            return "Không đạt (< 0.80)"

    @staticmethod
    def interpret_selectivity(selectivity_value: float) -> str:
        """
        Diễn giải chỉ số chọn lọc

        Parameters
        ----------
        selectivity_value : float
            Giá trị chỉ số chọn lọc

        Returns
        -------
        str
            Diễn giải chỉ số
        """
        if selectivity_value >= 0.9:
            return "Tuyệt vời (≥ 0.9)"
        elif 0.8 <= selectivity_value < 0.9:
            return "Rất tốt (0.8 - 0.9)"
        elif 0.7 <= selectivity_value < 0.8:
            return "Tốt (0.7 - 0.8)"
        elif 0.6 <= selectivity_value < 0.7:
            return "Chấp nhận được (0.6 - 0.7)"
        else:
            return "Kém (< 0.6)"

    @staticmethod
    def interpret_gradient_index(gi_value: float) -> str:
        """
        Diễn giải chỉ số gradient

        Parameters
        ----------
        gi_value : float
            Giá trị chỉ số gradient

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
    def interpret_paddick_metrics(metrics: Dict[str, float]) -> Dict[str, str]:
        """
        Diễn giải tất cả các chỉ số Paddick

        Parameters
        ----------
        metrics : Dict[str, float]
            Từ điển chứa các chỉ số Paddick

        Returns
        -------
        Dict[str, str]
            Từ điển chứa diễn giải của các chỉ số
        """
        interpretations = {}

        if "Paddick CI" in metrics:
            interpretations["Paddick CI"] = PaddickIndices.interpret_paddick_ci(
                metrics["Paddick CI"]
            )

        if "Coverage Index" in metrics:
            interpretations["Coverage Index"] = PaddickIndices.interpret_coverage(
                metrics["Coverage Index"]
            )

        if "Selectivity Index" in metrics:
            interpretations["Selectivity Index"] = PaddickIndices.interpret_selectivity(
                metrics["Selectivity Index"]
            )

        if "Gradient Index" in metrics:
            interpretations["Gradient Index"] = PaddickIndices.interpret_gradient_index(
                metrics["Gradient Index"]
            )

        return interpretations


# Alias function for backward compatibility
def calculate_paddick_indices(
    dose_grid: np.ndarray,
    target_mask: np.ndarray,
    prescription_dose: float,
    dose_level: float = 0.95,
    spacing: tuple = (1.0, 1.0, 1.0),
) -> Dict[str, float]:
    """
    Alias function cho paddick_metrics_report

    Parameters
    ----------
    dose_grid : np.ndarray
        Ma trận liều 3D
    target_mask : np.ndarray
        Mask của target structure
    prescription_dose : float
        Liều kê toa
    dose_level : float, optional
        Mức liều để đánh giá (phần trăm của prescription dose)
    spacing : tuple, optional
        Spacing của voxel (mm)

    Returns
    -------
    Dict[str, float]
        Dictionary chứa các chỉ số Paddick
    """
    # Tính toán các thể tích
    voxel_volume = spacing[0] * spacing[1] * spacing[2] / 1000.0  # cm³

    # Thể tích target
    tv = np.sum(target_mask) * voxel_volume

    # Thể tích phủ bởi mức liều quy định
    dose_threshold = prescription_dose * dose_level
    dose_mask = dose_grid >= dose_threshold
    v_ri = np.sum(dose_mask) * voxel_volume

    # Thể tích target phủ bởi mức liều quy định
    tv_ri = np.sum(target_mask & dose_mask) * voxel_volume

    # Thể tích tại 50% liều
    half_dose_threshold = prescription_dose * 0.5
    half_dose_mask = dose_grid >= half_dose_threshold
    v_half = np.sum(half_dose_mask) * voxel_volume

    # Sử dụng PaddickIndices class để tính toán
    paddick_indices = PaddickIndices()

    return paddick_indices.paddick_metrics_report(tv_ri, tv, v_ri, v_half)


__all__ = ["PaddickIndices", "calculate_paddick_indices"]
