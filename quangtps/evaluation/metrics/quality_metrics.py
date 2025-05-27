#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module tính toán các chỉ số đánh giá chất lượng kế hoạch xạ trị.

Module này cung cấp các hàm tính toán các chỉ số đánh giá chất lượng kế hoạch
như Conformity Index (CI), Homogeneity Index (HI), Gradient Index (GI), và nhiều
chỉ số khác được sử dụng trong lâm sàng xạ trị.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any

from quangtps.evaluation.dvh.dvh_data import DVHData
from quangtps.dose.dose_grid import DoseGrid

logger = logging.getLogger(__name__)


def calculate_conformity_index(
    dose_grid: DoseGrid,
    target_mask: np.ndarray,
    prescription_dose: float,
    method: str = "paddick",
) -> float:
    """
    Tính toán chỉ số phù hợp (Conformity Index).

    Chỉ số này đánh giá mức độ phù hợp của phân bố liều với cấu trúc đích.

    Args:
        dose_grid: Lưới liều
        target_mask: Mặt nạ nhị phân của cấu trúc đích
        prescription_dose: Liều chỉ định (Gy)
        method: Phương pháp tính ("paddick", "rtog", or "van't riet")

    Returns:
        Giá trị chỉ số phù hợp (CI)
    """
    # Lấy liều và tạo mặt nạ cho vùng nhận liều >= prescription_dose
    dose_array = dose_grid.dose_array
    rx_dose_mask = dose_array >= prescription_dose

    # Tính thể tích cấu trúc đích
    target_volume = np.sum(target_mask)
    if target_volume == 0:
        logger.warning("Thể tích cấu trúc đích bằng 0")
        return 0.0

    # Tính thể tích nhận liều chỉ định
    rx_dose_volume = np.sum(rx_dose_mask)
    if rx_dose_volume == 0:
        logger.warning("Không có thể tích nào nhận liều chỉ định")
        return 0.0

    # Tính thể tích cấu trúc đích nhận liều chỉ định
    target_rx_volume = np.sum(target_mask & rx_dose_mask)

    # Tính CI theo phương pháp được chỉ định
    if method.lower() == "paddick":
        # Công thức Paddick CI = (TV_PIV)² / (TV * PIV)
        # TV_PIV: thể tích đích nhận liều chỉ định
        # TV: thể tích đích
        # PIV: thể tích nhận liều chỉ định
        ci = (target_rx_volume * target_rx_volume) / (target_volume * rx_dose_volume)
    elif method.lower() == "rtog":
        # Công thức RTOG CI = PIV / TV
        ci = rx_dose_volume / target_volume
    elif method.lower() == "van't riet":
        # Công thức van't Riet (tương tự Paddick)
        ci = (target_rx_volume * target_rx_volume) / (target_volume * rx_dose_volume)
    else:
        logger.warning(f"Phương pháp {method} không được hỗ trợ, sử dụng Paddick")
        ci = (target_rx_volume * target_rx_volume) / (target_volume * rx_dose_volume)

    return ci


def calculate_homogeneity_index(
    dose_grid: DoseGrid,
    target_mask: np.ndarray,
    method: str = "icru",
    d_near_min: float = 2.0,
    d_near_max: float = 2.0,
) -> float:
    """
    Tính toán chỉ số đồng nhất (Homogeneity Index).

    Chỉ số này đánh giá mức độ đồng nhất của liều trong cấu trúc đích.

    Args:
        dose_grid: Lưới liều
        target_mask: Mặt nạ nhị phân của cấu trúc đích
        method: Phương pháp tính ("icru", "rtog", "other")
        d_near_min: Giá trị % thể tích gần nhỏ nhất (thường là 2% cho D98%)
        d_near_max: Giá trị % thể tích gần lớn nhất (thường là 2% cho D2%)

    Returns:
        Giá trị chỉ số đồng nhất (HI)
    """
    # Lấy giá trị liều trong cấu trúc đích
    target_doses = dose_grid.dose_array[target_mask]
    if len(target_doses) == 0:
        logger.warning("Không có dữ liệu liều trong cấu trúc đích")
        return 0.0

    # Tính các giá trị liều cần thiết
    d_max = np.max(target_doses)
    d_min = np.min(target_doses)
    d_mean = np.mean(target_doses)
    d_median = np.median(target_doses)

    # Tính D2% và D98% (gần nhất với Dmax và Dmin)
    sorted_doses = np.sort(target_doses)
    total_voxels = len(sorted_doses)
    d_near_min_index = int(d_near_min * total_voxels / 100)
    d_near_max_index = int((100 - d_near_max) * total_voxels / 100)

    # Đảm bảo các chỉ số nằm trong khoảng hợp lệ
    d_near_min_index = max(0, min(d_near_min_index, total_voxels - 1))
    d_near_max_index = max(0, min(d_near_max_index, total_voxels - 1))

    d_near_min_value = sorted_doses[d_near_min_index]
    d_near_max_value = sorted_doses[d_near_max_index]

    # Tính HI theo phương pháp được chỉ định
    if method.lower() == "icru":
        # ICRU 83: HI = (D2% - D98%) / D50%
        if d_median == 0:
            logger.warning("D50% bằng 0, không thể tính HI theo phương pháp ICRU")
            return 0.0
        hi = (d_near_max_value - d_near_min_value) / d_median
    elif method.lower() == "rtog":
        # RTOG: HI = Dmax / Dprescription
        # Quy ước Dprescription = Dmean
        if d_mean == 0:
            logger.warning("Dmean bằng 0, không thể tính HI theo phương pháp RTOG")
            return 0.0
        hi = d_max / d_mean
    elif method.lower() == "other":
        # Công thức khác: HI = (Dmax - Dmin) / Dmean
        if d_mean == 0:
            logger.warning("Dmean bằng 0, không thể tính HI")
            return 0.0
        hi = (d_max - d_min) / d_mean
    else:
        logger.warning(f"Phương pháp {method} không được hỗ trợ, sử dụng ICRU")
        if d_median == 0:
            return 0.0
        hi = (d_near_max_value - d_near_min_value) / d_median

    return hi


def calculate_gradient_index(
    dose_grid: DoseGrid,
    prescription_dose: float,
    low_dose_level: float = 0.5,
) -> float:
    """
    Tính toán chỉ số độ dốc (Gradient Index).

    Chỉ số này đánh giá tốc độ giảm liều xung quanh vùng điều trị.

    Args:
        dose_grid: Lưới liều
        prescription_dose: Liều chỉ định (Gy)
        low_dose_level: Mức liều thấp cho việc tính toán (so với liều chỉ định)

    Returns:
        Giá trị chỉ số độ dốc (GI)
    """
    # Lấy liều
    dose_array = dose_grid.dose_array

    # Tạo mặt nạ cho các vùng liều
    rx_dose_mask = dose_array >= prescription_dose
    low_dose_mask = dose_array >= (prescription_dose * low_dose_level)

    # Tính thể tích các vùng liều
    rx_dose_volume = np.sum(rx_dose_mask)
    low_dose_volume = np.sum(low_dose_mask)

    # Kiểm tra thể tích liều chỉ định
    if rx_dose_volume == 0:
        logger.warning("Không có thể tích nào nhận liều chỉ định")
        return 0.0

    # Tính GI
    gi = low_dose_volume / rx_dose_volume

    return gi


def calculate_target_coverage(
    dose_grid: DoseGrid,
    target_mask: np.ndarray,
    prescription_dose: float,
    coverage_level: float = 0.95,
) -> float:
    """
    Tính toán độ phủ đích (Target Coverage).

    Độ phủ đích là tỷ lệ thể tích cấu trúc đích nhận được ít nhất
    một phần nhất định của liều chỉ định.

    Args:
        dose_grid: Lưới liều
        target_mask: Mặt nạ nhị phân của cấu trúc đích
        prescription_dose: Liều chỉ định (Gy)
        coverage_level: Mức liều cần đánh giá (so với liều chỉ định)

    Returns:
        Giá trị độ phủ đích (TC)
    """
    # Lấy liều
    dose_array = dose_grid.dose_array

    # Tạo mặt nạ cho vùng nhận liều >= coverage_level * prescription_dose
    coverage_dose_mask = dose_array >= (coverage_level * prescription_dose)

    # Tính thể tích cấu trúc đích
    target_volume = np.sum(target_mask)
    if target_volume == 0:
        logger.warning("Thể tích cấu trúc đích bằng 0")
        return 0.0

    # Tính thể tích cấu trúc đích nhận liều đủ
    covered_target_volume = np.sum(target_mask & coverage_dose_mask)

    # Tính TC
    tc = covered_target_volume / target_volume

    return tc


def calculate_conformation_number(
    dose_grid: DoseGrid,
    target_mask: np.ndarray,
    prescription_dose: float,
) -> float:
    """
    Tính toán số phù hợp (Conformation Number).

    Số phù hợp kết hợp độ phủ đích và tính chọn lọc.

    Args:
        dose_grid: Lưới liều
        target_mask: Mặt nạ nhị phân của cấu trúc đích
        prescription_dose: Liều chỉ định (Gy)

    Returns:
        Giá trị số phù hợp (CN)
    """
    # Lấy liều
    dose_array = dose_grid.dose_array

    # Tạo mặt nạ cho vùng nhận liều >= prescription_dose
    rx_dose_mask = dose_array >= prescription_dose

    # Tính thể tích cấu trúc đích
    target_volume = np.sum(target_mask)
    if target_volume == 0:
        logger.warning("Thể tích cấu trúc đích bằng 0")
        return 0.0

    # Tính thể tích nhận liều chỉ định
    rx_dose_volume = np.sum(rx_dose_mask)
    if rx_dose_volume == 0:
        logger.warning("Không có thể tích nào nhận liều chỉ định")
        return 0.0

    # Tính thể tích cấu trúc đích nhận liều chỉ định
    target_rx_volume = np.sum(target_mask & rx_dose_mask)

    # Tính CN
    target_coverage = target_rx_volume / target_volume
    healthy_tissue_sparing = target_rx_volume / rx_dose_volume
    cn = target_coverage * healthy_tissue_sparing

    return cn


def calculate_radiation_conformity_index(
    dose_grid: DoseGrid,
    target_mask: np.ndarray,
    prescription_dose: float,
) -> float:
    """
    Tính toán chỉ số phù hợp xạ trị (Radiation Conformity Index).

    RCI là tỷ lệ giữa thể tích cấu trúc đích nhận liều chỉ định
    và thể tích nhận liều chỉ định.

    Args:
        dose_grid: Lưới liều
        target_mask: Mặt nạ nhị phân của cấu trúc đích
        prescription_dose: Liều chỉ định (Gy)

    Returns:
        Giá trị chỉ số phù hợp xạ trị (RCI)
    """
    # Lấy liều
    dose_array = dose_grid.dose_array

    # Tạo mặt nạ cho vùng nhận liều >= prescription_dose
    rx_dose_mask = dose_array >= prescription_dose

    # Tính thể tích nhận liều chỉ định
    rx_dose_volume = np.sum(rx_dose_mask)
    if rx_dose_volume == 0:
        logger.warning("Không có thể tích nào nhận liều chỉ định")
        return 0.0

    # Tính thể tích cấu trúc đích nhận liều chỉ định
    target_rx_volume = np.sum(target_mask & rx_dose_mask)

    # Tính RCI
    rci = target_rx_volume / rx_dose_volume

    return rci


def calculate_dvh_metrics(
    dvh_data: DVHData,
    structure_id: str,
    metrics: List[str],
) -> Dict[str, float]:
    """
    Tính toán các chỉ số DVH cho một cấu trúc.

    Args:
        dvh_data: Dữ liệu DVH
        structure_id: ID của cấu trúc
        metrics: Danh sách các chỉ số cần tính (ví dụ: ["D95", "V20Gy"])

    Returns:
        Dictionary chứa các chỉ số DVH
    """
    result = {}

    # Lấy đường cong DVH cho cấu trúc
    curve = dvh_data.get_curve(structure_id)
    if curve is None:
        logger.warning(f"Không tìm thấy dữ liệu DVH cho cấu trúc {structure_id}")
        return result

    # Tính từng chỉ số
    for metric in metrics:
        metric = metric.strip()
        if metric.startswith("D") and "%" in metric:
            # Dxx% - liều nhận bởi xx% thể tích
            try:
                volume_percent = float(metric[1:].replace("%", ""))
                result[metric] = curve.calculate_d_metric(volume_percent)
            except (ValueError, IndexError):
                logger.warning(f"Không thể tính chỉ số {metric}")
        elif metric.startswith("D") and "cc" in metric:
            # Dxxcc - liều nhận bởi xx cc thể tích
            try:
                volume_cc = float(metric[1:].replace("cc", ""))
                volume_percent = 100 * volume_cc / curve.total_volume
                result[metric] = curve.calculate_d_metric(volume_percent)
            except (ValueError, IndexError, ZeroDivisionError):
                logger.warning(f"Không thể tính chỉ số {metric}")
        elif metric.startswith("V") and "Gy" in metric:
            # Vxx Gy - thể tích nhận liều >= xx Gy
            try:
                dose = float(metric[1:].replace("Gy", ""))
                result[metric] = curve.calculate_v_metric(dose)
            except (ValueError, IndexError):
                logger.warning(f"Không thể tính chỉ số {metric}")
        elif metric.startswith("V") and "%" in metric:
            # Vxx% - thể tích nhận liều >= xx% liều chỉ định
            try:
                dose_percent = float(metric[1:].replace("%", ""))
                rx_dose = dvh_data.prescription_dose
                dose = rx_dose * dose_percent / 100.0
                result[metric] = curve.calculate_v_metric(dose)
            except (ValueError, IndexError):
                logger.warning(f"Không thể tính chỉ số {metric}")
        elif metric == "mean" or metric == "dmean":
            # Liều trung bình
            result[metric] = dvh_data.get_mean_dose(structure_id)
        elif metric == "min" or metric == "dmin":
            # Liều tối thiểu
            result[metric] = dvh_data.get_min_dose(structure_id)
        elif metric == "max" or metric == "dmax":
            # Liều tối đa
            result[metric] = dvh_data.get_max_dose(structure_id)
        elif metric == "median":
            # Liều trung vị
            result[metric] = dvh_data.get_median_dose(structure_id)
        else:
            logger.warning(f"Chỉ số {metric} không được hỗ trợ")

    return result


def calculate_equivalent_uniform_dose(
    dvh_data: DVHData,
    structure_id: str,
    a: float = 1.0,
) -> float:
    """
    Tính toán liều đồng nhất tương đương (Equivalent Uniform Dose - EUD).

    EUD là liều đồng nhất sẽ gây ra cùng hiệu ứng sinh học như
    phân bố liều không đồng nhất.

    Args:
        dvh_data: Dữ liệu DVH
        structure_id: ID của cấu trúc
        a: Tham số a đặc trưng cho mô (a < 0 cho OAR, a > 0 cho PTV)

    Returns:
        Giá trị EUD (Gy)
    """
    # Lấy đường cong DVH cho cấu trúc
    curve = dvh_data.get_curve(structure_id)
    if curve is None or not curve.dose_bins or not curve.volume_bins:
        logger.warning(f"Không tìm thấy dữ liệu DVH hợp lệ cho cấu trúc {structure_id}")
        return 0.0

    # Chuyển đổi từ DVH tích lũy sang dạng vi phân
    diff_volume = np.zeros_like(curve.volume_bins)
    diff_volume[0] = 100.0 - curve.volume_bins[0]
    for i in range(1, len(curve.volume_bins)):
        diff_volume[i] = curve.volume_bins[i - 1] - curve.volume_bins[i]

    # Chuẩn hóa thể tích vi phân thành phần trăm
    total_diff_volume = np.sum(diff_volume)
    if total_diff_volume == 0:
        logger.warning("Tổng thể tích vi phân bằng 0")
        return 0.0

    normalized_diff_volume = diff_volume / total_diff_volume

    # Tính EUD
    if a == 1:
        # Trường hợp đặc biệt a = 1, EUD = Dmean
        eud = np.sum(curve.dose_bins * normalized_diff_volume)
    else:
        # Công thức tổng quát
        eud = np.power(
            np.sum(normalized_diff_volume * np.power(curve.dose_bins, a)), 1.0 / a
        )

    return eud


def calculate_biological_metrics(
    dvh_data: DVHData,
    structure_info: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, float]]:
    """
    Tính toán các chỉ số sinh học cho các cấu trúc.

    Args:
        dvh_data: Dữ liệu DVH
        structure_info: Dictionary chứa thông tin về các cấu trúc
            {structure_id: {"type": "ptv/oar", "alpha_beta": 3.0, "a": 1.0}}

    Returns:
        Dictionary chứa các chỉ số sinh học cho mỗi cấu trúc
    """
    result = {}

    for structure_id, info in structure_info.items():
        if structure_id not in dvh_data.get_structure_ids():
            logger.warning(f"Không tìm thấy dữ liệu DVH cho cấu trúc {structure_id}")
            continue

        structure_type = info.get("type", "").lower()
        alpha_beta = info.get("alpha_beta", 3.0)  # Giá trị α/β, mặc định 3.0 Gy
        a_param = info.get("a", 1.0)  # Tham số a cho EUD

        # Xác định tham số a dựa trên loại cấu trúc nếu không được chỉ định
        if "a" not in info:
            if structure_type == "ptv" or structure_type == "target":
                a_param = 10.0  # Giá trị điển hình cho PTV
            elif structure_type == "oar":
                a_param = -10.0  # Giá trị điển hình cho OAR

        structure_result = {}

        # Tính EUD
        eud = calculate_equivalent_uniform_dose(dvh_data, structure_id, a_param)
        structure_result["EUD"] = eud

        # Tính các chỉ số sinh học khác dựa trên loại cấu trúc
        if structure_type == "ptv" or structure_type == "target":
            # Tính TCP (Tumor Control Probability) - đơn giản hóa
            # TCP = exp(-exp(-γ(EUD - TCD50)))
            # γ: độ dốc của đường cong liều-đáp ứng (mặc định 2)
            # TCD50: liều kiểm soát u 50% (mặc định = prescription_dose)
            gamma = info.get("gamma", 2.0)
            tcd50 = info.get("tcd50", dvh_data.prescription_dose)

            if tcd50 is not None and gamma > 0:
                tcp = np.exp(-np.exp(-gamma * (eud - tcd50)))
                structure_result["TCP"] = tcp
        elif structure_type == "oar":
            # Tính NTCP (Normal Tissue Complication Probability) - mô hình LKB
            # NTCP = 1/(1 + (TCD50/EUD)^k)
            # k: độ dốc của đường cong liều-đáp ứng (mặc định 4)
            # TD50: liều gây tổn thương 50% (tùy thuộc vào cơ quan)
            k = info.get("k", 4.0)
            td50 = info.get(
                "td50", 50.0
            )  # Giá trị mặc định, cần điều chỉnh theo cơ quan

            if td50 > 0 and k > 0 and eud > 0:
                ntcp = 1.0 / (1.0 + (td50 / eud) ** k)
                structure_result["NTCP"] = ntcp

        result[structure_id] = structure_result

    return result


def calculate_all_quality_metrics(
    dose_grid: DoseGrid,
    targets: Dict[str, np.ndarray],
    oars: Dict[str, np.ndarray],
    prescription_doses: Dict[str, float],
) -> Dict[str, Dict[str, float]]:
    """
    Tính toán toàn bộ các chỉ số chất lượng cho kế hoạch xạ trị.

    Args:
        dose_grid: Lưới liều
        targets: Dictionary chứa mặt nạ cho các cấu trúc đích
        oars: Dictionary chứa mặt nạ cho các cơ quan nguy cấp
        prescription_doses: Dictionary chứa liều chỉ định cho các cấu trúc đích

    Returns:
        Dictionary chứa các chỉ số chất lượng cho mỗi cấu trúc và toàn kế hoạch
    """
    results = {"targets": {}, "oars": {}, "plan": {}}

    # Tính các chỉ số cho từng cấu trúc đích
    for target_name, target_mask in targets.items():
        if target_name not in prescription_doses:
            logger.warning(
                f"Không tìm thấy liều chỉ định cho cấu trúc đích {target_name}"
            )
            continue

        rx_dose = prescription_doses[target_name]
        target_results = {}

        # Tính các chỉ số chất lượng cho cấu trúc đích
        target_results["CI"] = calculate_conformity_index(
            dose_grid, target_mask, rx_dose
        )
        target_results["HI"] = calculate_homogeneity_index(dose_grid, target_mask)
        target_results["GI"] = calculate_gradient_index(dose_grid, rx_dose)
        target_results["TC"] = calculate_target_coverage(
            dose_grid, target_mask, rx_dose
        )
        target_results["CN"] = calculate_conformation_number(
            dose_grid, target_mask, rx_dose
        )
        target_results["RCI"] = calculate_radiation_conformity_index(
            dose_grid, target_mask, rx_dose
        )

        results["targets"][target_name] = target_results

    # Tính các chỉ số tổng thể cho kế hoạch
    if targets and prescription_doses:
        # Lấy cấu trúc đích chính và liều chỉ định tương ứng
        main_target = next(iter(targets))
        main_rx_dose = prescription_doses[main_target]

        # Tính các chỉ số chung cho toàn kế hoạch
        plan_results = {}
        plan_results["Overall_CI"] = np.mean(
            [results["targets"][t]["CI"] for t in results["targets"]]
        )
        plan_results["Overall_HI"] = np.mean(
            [results["targets"][t]["HI"] for t in results["targets"]]
        )
        plan_results["Overall_GI"] = np.mean(
            [results["targets"][t]["GI"] for t in results["targets"]]
        )
        plan_results["Overall_TC"] = np.mean(
            [results["targets"][t]["TC"] for t in results["targets"]]
        )

        results["plan"] = plan_results

    return results


# Alias function for backward compatibility
def calculate_quality_metrics(
    dose_grid: DoseGrid,
    targets: Dict[str, np.ndarray],
    oars: Dict[str, np.ndarray],
    prescription_doses: Dict[str, float],
) -> Dict[str, Dict[str, float]]:
    """
    Alias function cho calculate_all_quality_metrics

    Parameters
    ----------
    dose_grid : DoseGrid
        Lưới liều
    targets : Dict[str, np.ndarray]
        Dictionary chứa mặt nạ của các mục tiêu
    oars : Dict[str, np.ndarray]
        Dictionary chứa mặt nạ của các cơ quan nguy cấp
    prescription_doses : Dict[str, float]
        Dictionary chứa liều kê toa cho mỗi mục tiêu

    Returns
    -------
    Dict[str, Dict[str, float]]
        Dictionary chứa các metrics cho mỗi cấu trúc
    """
    return calculate_all_quality_metrics(dose_grid, targets, oars, prescription_doses)


# Export cho convenience
__all__ = [
    "calculate_conformity_index",
    "calculate_homogeneity_index",
    "calculate_gradient_index",
    "calculate_target_coverage",
    "calculate_conformation_number",
    "calculate_radiation_conformity_index",
    "calculate_dvh_metrics",
    "calculate_equivalent_uniform_dose",
    "calculate_biological_metrics",
    "calculate_all_quality_metrics",
    "calculate_quality_metrics",  # Alias
]
