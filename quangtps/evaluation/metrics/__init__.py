#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Các metrics đánh giá trong hệ thống lập kế hoạch QuangTPS.

Module này chứa các chỉ số và công cụ đánh giá khác nhau bao gồm
phân tích gamma, chỉ số đồng nhất, chỉ số đánh giá, và phân tích DVH.
"""

from typing import Dict, List, Any, Optional, Union, Tuple

# Import các lớp và hàm chính
try:
    from .clinical_metrics import calculate_clinical_metrics
except ImportError:
    pass

try:
    from .quality_metrics import calculate_quality_metrics
except ImportError:
    pass

try:
    from .radiobiological import calculate_radiobiological_metrics
except ImportError:
    pass

try:
    from .plan_metrics import (
        calculate_plan_metrics,
        compare_plan_metrics,
        analyze_plan_robustness,
        plot_plan_comparison,
        generate_plan_metrics_report,
    )
except ImportError:
    pass

# Import các module metrics
try:
    from quangtps.evaluation.metrics.gamma_analysis import (
        calculate_gamma_3d,
        calculate_gamma_2d,
        calculate_gamma_3d_gpu,
        gamma_pass_rate,
        get_gamma_statistics,
        analyze_gamma_by_dose_regions,
        plot_gamma_results,
    )

    HAS_GAMMA_MODULE = True

    # Kiểm tra khả năng sử dụng GPU
    try:
        import cupy as cp

        HAS_GAMMA_GPU = True
    except (ImportError, ModuleNotFoundError):
        HAS_GAMMA_GPU = False
except ImportError:
    HAS_GAMMA_MODULE = False
    HAS_GAMMA_GPU = False

from .dose_metrics import *
from .biological_metrics import *
from .plan_quality_metrics import *
from .dvh_metrics import *


# Compatibility functions
def calculate_d_metric(dose_distribution, volume_percent, structure_mask=None):
    """
    Tính toán D metric (dose tại volume percent cụ thể).

    Args:
        dose_distribution: Array phân phối liều
        volume_percent: Phần trăm volume (0-100)
        structure_mask: Mask cấu trúc (optional)

    Returns:
        float: Giá trị dose tại volume percent
    """
    try:
        import numpy as np

        if structure_mask is not None:
            # Apply structure mask
            valid_doses = dose_distribution[structure_mask > 0]
        else:
            valid_doses = dose_distribution.flatten()

        # Remove zero and negative doses
        valid_doses = valid_doses[valid_doses > 0]

        if len(valid_doses) == 0:
            return 0.0

        # Sort doses in descending order
        sorted_doses = np.sort(valid_doses)[::-1]

        # Calculate index for volume percent
        volume_index = int((volume_percent / 100.0) * len(sorted_doses))
        volume_index = min(volume_index, len(sorted_doses) - 1)

        return float(sorted_doses[volume_index])

    except Exception as e:
        print(f"Error calculating D metric: {e}")
        return 0.0


def calculate_v_metric(dose_distribution, dose_level, structure_mask=None):
    """
    Tính toán V metric (volume percent tại dose level cụ thể).

    Args:
        dose_distribution: Array phân phối liều
        dose_level: Mức liều (Gy)
        structure_mask: Mask cấu trúc (optional)

    Returns:
        float: Phần trăm volume tại dose level
    """
    try:
        import numpy as np

        if structure_mask is not None:
            # Apply structure mask
            valid_doses = dose_distribution[structure_mask > 0]
            total_volume = np.sum(structure_mask > 0)
        else:
            valid_doses = dose_distribution.flatten()
            total_volume = len(valid_doses)

        if total_volume == 0:
            return 0.0

        # Count voxels with dose >= dose_level
        volume_at_dose = np.sum(valid_doses >= dose_level)

        # Calculate percentage
        volume_percent = (volume_at_dose / total_volume) * 100.0

        return float(volume_percent)

    except Exception as e:
        print(f"Error calculating V metric: {e}")
        return 0.0


def conformity_index(
    dose_distribution, target_mask, prescription_dose, isodose_level=0.95
):
    """
    Tính chỉ số phù hợp (Conformity Index).

    Parameters
    ----------
    dose_distribution : np.ndarray
        Phân phối liều 3D
    target_mask : np.ndarray
        Mask của cấu trúc target (PTV)
    prescription_dose : float
        Liều kê đơn (Gy)
    isodose_level : float, optional
        Mức isodose để đánh giá, mặc định 0.95 (95%)

    Returns
    -------
    float
        Chỉ số conformity (0-1, 1 là tốt nhất)
    """
    try:
        import numpy as np

        # Tính toán liều threshold
        dose_threshold = prescription_dose * isodose_level

        # Volume target
        target_volume = np.sum(target_mask > 0)
        if target_volume == 0:
            return 0.0

        # Volume nhận liều >= threshold
        volume_receiving_dose = np.sum(dose_distribution >= dose_threshold)
        if volume_receiving_dose == 0:
            return 0.0

        # Volume target nhận liều >= threshold
        target_receiving_dose = np.sum(
            (dose_distribution >= dose_threshold) & (target_mask > 0)
        )

        # Conformity Index = V_target_receiving_dose / V_receiving_dose
        ci = target_receiving_dose / volume_receiving_dose

        return float(min(1.0, max(0.0, ci)))

    except Exception as e:
        print(f"Error calculating conformity index: {e}")
        return 0.0


def homogeneity_index(dose_distribution, target_mask, prescription_dose):
    """
    Tính chỉ số đồng nhất (Homogeneity Index).

    Parameters
    ----------
    dose_distribution : np.ndarray
        Phân phối liều 3D
    target_mask : np.ndarray
        Mask của cấu trúc target (PTV)
    prescription_dose : float
        Liều kê đơn (Gy)

    Returns
    -------
    float
        Chỉ số homogeneity (0-1, càng gần 0 càng đồng nhất)
    """
    try:
        import numpy as np

        # Lấy liều trong target
        target_doses = dose_distribution[target_mask > 0]
        if len(target_doses) == 0:
            return 1.0  # Worst case

        # Tính D2 và D98 (liều tại 2% và 98% volume)
        sorted_doses = np.sort(target_doses)[::-1]  # Descending order

        n_voxels = len(sorted_doses)
        d2_index = int(0.02 * n_voxels)
        d98_index = int(0.98 * n_voxels)

        d2_index = min(d2_index, n_voxels - 1)
        d98_index = min(d98_index, n_voxels - 1)

        d2 = sorted_doses[d2_index]
        d98 = sorted_doses[d98_index]

        # Homogeneity Index = (D2 - D98) / prescription_dose
        if prescription_dose > 0:
            hi = (d2 - d98) / prescription_dose
        else:
            hi = 1.0

        return float(max(0.0, hi))

    except Exception as e:
        print(f"Error calculating homogeneity index: {e}")
        return 1.0


def coverage_index(
    dose_distribution, target_mask, prescription_dose, coverage_level=0.95
):
    """
    Tính chỉ số bao phủ (Coverage Index).

    Parameters
    ----------
    dose_distribution : np.ndarray
        Phân phối liều 3D
    target_mask : np.ndarray
        Mask của cấu trúc target
    prescription_dose : float
        Liều kê đơn (Gy)
    coverage_level : float, optional
        Mức bao phủ yêu cầu, mặc định 0.95 (95%)

    Returns
    -------
    float
        Chỉ số coverage (0-1, 1 là tốt nhất)
    """
    try:
        import numpy as np

        # Tính toán liều threshold
        dose_threshold = prescription_dose * coverage_level

        # Volume target
        target_volume = np.sum(target_mask > 0)
        if target_volume == 0:
            return 0.0

        # Volume target nhận liều >= threshold
        covered_volume = np.sum(
            (dose_distribution >= dose_threshold) & (target_mask > 0)
        )

        # Coverage Index
        coverage = covered_volume / target_volume

        return float(min(1.0, max(0.0, coverage)))

    except Exception as e:
        print(f"Error calculating coverage index: {e}")
        return 0.0


def gradient_index(dose_distribution, target_mask, body_mask, prescription_dose):
    """
    Tính chỉ số gradient (Gradient Index).

    Parameters
    ----------
    dose_distribution : np.ndarray
        Phân phối liều 3D
    target_mask : np.ndarray
        Mask của cấu trúc target
    body_mask : np.ndarray
        Mask của body
    prescription_dose : float
        Liều kê đơn (Gy)

    Returns
    -------
    float
        Chỉ số gradient (>= 1, càng gần 1 càng tốt)
    """
    try:
        import numpy as np

        # Volume nhận 50% liều kê đơn trong body
        dose_50_threshold = prescription_dose * 0.5
        volume_50 = np.sum((dose_distribution >= dose_50_threshold) & (body_mask > 0))

        # Volume target nhận liều kê đơn (sử dụng ngưỡng thấp hơn để tránh zero)
        dose_95_threshold = prescription_dose * 0.95  # Sử dụng 95% thay vì 100%
        target_volume_95 = np.sum(
            (dose_distribution >= dose_95_threshold) & (target_mask > 0)
        )

        # Nếu vẫn không có volume nào, sử dụng toàn bộ target volume
        if target_volume_95 == 0:
            target_volume_95 = np.sum(target_mask > 0)

        # Nếu vẫn bằng 0, trả về giá trị mặc định
        if target_volume_95 == 0:
            return 10.0  # Giá trị kém để chỉ ra lỗi

        # Gradient Index = V50% / V95%_target
        gi = volume_50 / target_volume_95

        # Clamp giá trị trong khoảng hợp lý [1.0, 20.0]
        return float(max(1.0, min(gi, 20.0)))

    except Exception as e:
        print(f"Error calculating gradient index: {e}")
        return 10.0  # Giá trị mặc định thay vì infinity


def calculate_plan_quality_metrics(
    dose_distribution, structure_masks, prescription_dose
):
    """
    Tính toán tất cả các metrics chất lượng kế hoạch.

    Parameters
    ----------
    dose_distribution : np.ndarray
        Phân phối liều 3D
    structure_masks : dict
        Dictionary các mask cấu trúc
    prescription_dose : float
        Liều kê đơn (Gy)

    Returns
    -------
    dict
        Dictionary chứa tất cả metrics
    """
    metrics = {}

    try:
        # Tìm target mask (PTV)
        target_mask = None
        for name, mask in structure_masks.items():
            if "ptv" in name.lower() or "target" in name.lower():
                target_mask = mask
                break

        if target_mask is not None:
            metrics["conformity_index"] = conformity_index(
                dose_distribution, target_mask, prescription_dose
            )
            metrics["homogeneity_index"] = homogeneity_index(
                dose_distribution, target_mask, prescription_dose
            )
            metrics["coverage_index"] = coverage_index(
                dose_distribution, target_mask, prescription_dose
            )

            # Tìm body mask cho gradient index
            body_mask = structure_masks.get("body", structure_masks.get("external"))
            if body_mask is not None:
                metrics["gradient_index"] = gradient_index(
                    dose_distribution, target_mask, body_mask, prescription_dose
                )

        # Tính toán metrics cho các OAR
        for struct_name, struct_mask in structure_masks.items():
            if "ptv" not in struct_name.lower() and "target" not in struct_name.lower():
                # Dose metrics cho OAR
                metrics[f"{struct_name}_mean_dose"] = calculate_d_metric(
                    dose_distribution,
                    50,
                    struct_mask,  # D50 ~ mean dose
                )
                metrics[f"{struct_name}_max_dose"] = calculate_d_metric(
                    dose_distribution,
                    0.1,
                    struct_mask,  # D0.1 ~ max dose
                )

    except Exception as e:
        print(f"Error calculating plan quality metrics: {e}")

    return metrics


def monitor_unit_efficiency(total_mu, prescription_dose, target_volume=None):
    """
    Tính toán hiệu suất Monitor Unit.

    Parameters
    ----------
    total_mu : float
        Tổng số Monitor Units
    prescription_dose : float
        Liều kê đơn (Gy)
    target_volume : float, optional
        Thể tích target (cc)

    Returns
    -------
    float
        Hiệu suất MU (Gy/MU hoặc Gy*cc/MU)
    """
    try:
        if total_mu <= 0:
            return 0.0

        if target_volume is not None and target_volume > 0:
            # MU efficiency per unit volume
            efficiency = (prescription_dose * target_volume) / total_mu
        else:
            # Simple MU efficiency
            efficiency = prescription_dose / total_mu

        return float(efficiency)

    except Exception as e:
        print(f"Error calculating MU efficiency: {e}")
        return 0.0


__all__ = [
    "calculate_clinical_metrics",
    "calculate_quality_metrics",
    "calculate_radiobiological_metrics",
    "calculate_plan_metrics",
    "compare_plan_metrics",
    "analyze_plan_robustness",
    "plot_plan_comparison",
    "generate_plan_metrics_report",
    "calculate_d_metric",
    "calculate_v_metric",
    "conformity_index",
    "homogeneity_index",
    "coverage_index",
    "gradient_index",
    "calculate_plan_quality_metrics",
    "monitor_unit_efficiency",
]

if HAS_GAMMA_MODULE:
    __all__.extend(
        [
            "calculate_gamma_3d",
            "calculate_gamma_2d",
            "gamma_pass_rate",
            "get_gamma_statistics",
            "analyze_gamma_by_dose_regions",
            "plot_gamma_results",
        ]
    )

    # Thêm hàm GPU nếu có hỗ trợ
    if HAS_GAMMA_GPU:
        __all__.append("calculate_gamma_3d_gpu")


def get_available_metrics() -> List[str]:
    """
    Trả về danh sách các metrics đã đăng ký và sẵn sàng sử dụng.

    Returns
    -------
    List[str]
        Danh sách tên các metrics có sẵn
    """
    metrics = []

    if HAS_GAMMA_MODULE:
        metrics.append("gamma_analysis")
        if HAS_GAMMA_GPU:
            metrics.append("gamma_analysis_gpu")

    return metrics
