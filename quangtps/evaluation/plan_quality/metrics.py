#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module chỉ số đánh giá kế hoạch xạ trị.

Module này cung cấp các công cụ tính toán các chỉ số đánh giá
chất lượng kế hoạch xạ trị theo các tiêu chuẩn ICRU và QUANTEC.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Đảm bảo import an toàn các module cần thiết
try:
    from quangtps.evaluation.dvh.dose_volume_histogram import DoseVolumeHistogram
    from quangtps.evaluation.dvh.dvh_metrics import DVHMetrics

    HAS_DVH = True
except ImportError:
    logger.warning(
        "Không thể import module DVH. Tính năng một số metrics sẽ bị giới hạn."
    )
    HAS_DVH = False


def calculate_conformity_index(
    target_volume_mm3: float, prescription_volume_mm3: float
) -> float:
    """
    Tính chỉ số phù hợp (Conformity Index).

    CI = V(prescription) / V(target)
    Lý tưởng: CI = 1.0

    Parameters:
        target_volume_mm3: Thể tích khối u (mm³)
        prescription_volume_mm3: Thể tích nhận liều kê đơn (mm³)

    Returns:
        Chỉ số phù hợp
    """
    if target_volume_mm3 <= 0:
        logger.warning("Thể tích khối u bằng hoặc nhỏ hơn 0.")
        return float("inf")

    return prescription_volume_mm3 / target_volume_mm3


def calculate_paddick_ci(
    target_prescription_volume_mm3: float,
    target_volume_mm3: float,
    prescription_volume_mm3: float,
) -> float:
    """
    Tính chỉ số phù hợp Paddick.

    CI_Paddick = (TV_PIV)² / (TV × PIV)
    Trong đó:
    - TV_PIV: Thể tích khối u nhận liều kê đơn
    - TV: Thể tích khối u
    - PIV: Thể tích nhận liều kê đơn

    Lý tưởng: CI_Paddick = 1.0

    Parameters:
        target_prescription_volume_mm3: Thể tích khối u nhận liều kê đơn (mm³)
        target_volume_mm3: Thể tích khối u (mm³)
        prescription_volume_mm3: Thể tích nhận liều kê đơn (mm³)

    Returns:
        Chỉ số phù hợp Paddick
    """
    if target_volume_mm3 <= 0 or prescription_volume_mm3 <= 0:
        logger.warning("Thể tích bằng hoặc nhỏ hơn 0.")
        return 0.0

    return (target_prescription_volume_mm3**2) / (
        target_volume_mm3 * prescription_volume_mm3
    )


def calculate_gradient_index(
    half_prescription_volume_mm3: float, prescription_volume_mm3: float
) -> float:
    """
    Tính chỉ số dốc (Gradient Index).

    GI = V(50% prescription) / V(prescription)
    Lý tưởng: GI < 3.0

    Parameters:
        half_prescription_volume_mm3: Thể tích nhận 50% liều kê đơn (mm³)
        prescription_volume_mm3: Thể tích nhận liều kê đơn (mm³)

    Returns:
        Chỉ số dốc
    """
    if prescription_volume_mm3 <= 0:
        logger.warning("Thể tích liều kê đơn bằng hoặc nhỏ hơn 0.")
        return float("inf")

    return half_prescription_volume_mm3 / prescription_volume_mm3


def calculate_homogeneity_index(
    d2_gy: float, d98_gy: float, prescription_dose_gy: float
) -> float:
    """
    Tính chỉ số đồng đều (Homogeneity Index) theo ICRU 83.

    HI = (D2% - D98%) / Dprescription
    Lý tưởng: HI = 0

    Parameters:
        d2_gy: Liều tại 2% thể tích (Gy)
        d98_gy: Liều tại 98% thể tích (Gy)
        prescription_dose_gy: Liều kê đơn (Gy)

    Returns:
        Chỉ số đồng đều
    """
    if prescription_dose_gy <= 0:
        logger.warning("Liều kê đơn bằng hoặc nhỏ hơn 0.")
        return float("inf")

    return (d2_gy - d98_gy) / prescription_dose_gy


def calculate_coverage_index(
    target_prescription_volume_mm3: float, target_volume_mm3: float
) -> float:
    """
    Tính chỉ số phủ (Coverage Index).

    CO = V(target,prescription) / V(target)
    Lý tưởng: CO = 1.0

    Parameters:
        target_prescription_volume_mm3: Thể tích khối u nhận liều kê đơn (mm³)
        target_volume_mm3: Thể tích khối u (mm³)

    Returns:
        Chỉ số phủ
    """
    if target_volume_mm3 <= 0:
        logger.warning("Thể tích khối u bằng hoặc nhỏ hơn 0.")
        return 0.0

    return target_prescription_volume_mm3 / target_volume_mm3


def calculate_conformation_number(
    target_prescription_volume_mm3: float,
    target_volume_mm3: float,
    prescription_volume_mm3: float,
) -> float:
    """
    Tính số đồng dạng (Conformation Number).

    CN = (V(target,prescription) / V(target)) × (V(target,prescription) / V(prescription))
    Lý tưởng: CN = 1.0

    Parameters:
        target_prescription_volume_mm3: Thể tích khối u nhận liều kê đơn (mm³)
        target_volume_mm3: Thể tích khối u (mm³)
        prescription_volume_mm3: Thể tích nhận liều kê đơn (mm³)

    Returns:
        Số đồng dạng
    """
    if target_volume_mm3 <= 0 or prescription_volume_mm3 <= 0:
        logger.warning("Thể tích bằng hoặc nhỏ hơn 0.")
        return 0.0

    coverage = target_prescription_volume_mm3 / target_volume_mm3
    conformity = target_prescription_volume_mm3 / prescription_volume_mm3

    return coverage * conformity


def calculate_healthy_tissue_conformity_index(
    target_prescription_volume_mm3: float, prescription_volume_mm3: float
) -> float:
    """
    Tính chỉ số phù hợp mô lành (Healthy Tissue Conformity Index).

    HTCI = V(target,prescription) / V(prescription)
    Lý tưởng: HTCI = 1.0

    Parameters:
        target_prescription_volume_mm3: Thể tích khối u nhận liều kê đơn (mm³)
        prescription_volume_mm3: Thể tích nhận liều kê đơn (mm³)

    Returns:
        Chỉ số phù hợp mô lành
    """
    if prescription_volume_mm3 <= 0:
        logger.warning("Thể tích liều kê đơn bằng hoặc nhỏ hơn 0.")
        return 0.0

    return target_prescription_volume_mm3 / prescription_volume_mm3


def calculate_van_t_riet_index(
    target_prescription_volume_mm3: float,
    target_volume_mm3: float,
    prescription_volume_mm3: float,
    overdose_factor: float = 1.0,
) -> float:
    """
    Tính chỉ số Van't Riet.

    VRI = (TV_PIV / TV) × (TV_PIV / PIV) × overdose_factor
    Trong đó:
    - TV_PIV: Thể tích khối u nhận liều kê đơn
    - TV: Thể tích khối u
    - PIV: Thể tích nhận liều kê đơn
    - overdose_factor: Hệ số điều chỉnh liều vượt quá

    Lý tưởng: VRI = 1.0

    Parameters:
        target_prescription_volume_mm3: Thể tích khối u nhận liều kê đơn (mm³)
        target_volume_mm3: Thể tích khối u (mm³)
        prescription_volume_mm3: Thể tích nhận liều kê đơn (mm³)
        overdose_factor: Hệ số điều chỉnh liều vượt quá

    Returns:
        Chỉ số Van't Riet
    """
    if target_volume_mm3 <= 0 or prescription_volume_mm3 <= 0:
        logger.warning("Thể tích bằng hoặc nhỏ hơn 0.")
        return 0.0

    coverage = target_prescription_volume_mm3 / target_volume_mm3
    conformity = target_prescription_volume_mm3 / prescription_volume_mm3

    return coverage * conformity * overdose_factor


def calculate_radiation_conformity_index(
    target_prescription_volume_mm3: float,
    target_volume_mm3: float,
    prescription_volume_mm3: float,
) -> float:
    """
    Tính chỉ số phù hợp bức xạ (Radiation Conformity Index).

    RCI = TV_PIV / PIV - TV_out / TV
    Trong đó:
    - TV_PIV: Thể tích khối u nhận liều kê đơn
    - PIV: Thể tích nhận liều kê đơn
    - TV_out: Thể tích khối u không nhận liều kê đơn
    - TV: Thể tích khối u

    Lý tưởng: RCI = 1.0

    Parameters:
        target_prescription_volume_mm3: Thể tích khối u nhận liều kê đơn (mm³)
        target_volume_mm3: Thể tích khối u (mm³)
        prescription_volume_mm3: Thể tích nhận liều kê đơn (mm³)

    Returns:
        Chỉ số phù hợp bức xạ
    """
    if prescription_volume_mm3 <= 0 or target_volume_mm3 <= 0:
        logger.warning("Thể tích bằng hoặc nhỏ hơn 0.")
        return 0.0

    # Thể tích khối u không nhận liều kê đơn
    target_out_volume = target_volume_mm3 - target_prescription_volume_mm3

    return (target_prescription_volume_mm3 / prescription_volume_mm3) - (
        target_out_volume / target_volume_mm3
    )


def calculate_modified_gradient_index(
    half_prescription_volume_mm3: float,
    prescription_volume_mm3: float,
    target_volume_mm3: float,
) -> float:
    """
    Tính chỉ số dốc cải tiến (Modified Gradient Index).

    MGI = (V(50% prescription) - V(target)) / V(prescription)

    Parameters:
        half_prescription_volume_mm3: Thể tích nhận 50% liều kê đơn (mm³)
        prescription_volume_mm3: Thể tích nhận liều kê đơn (mm³)
        target_volume_mm3: Thể tích khối u (mm³)

    Returns:
        Chỉ số dốc cải tiến
    """
    if prescription_volume_mm3 <= 0:
        logger.warning("Thể tích liều kê đơn bằng hoặc nhỏ hơn 0.")
        return float("inf")

    return (half_prescription_volume_mm3 - target_volume_mm3) / prescription_volume_mm3


def calculate_equivalent_uniform_dose(
    dvh_data: np.ndarray, volume_fractions: np.ndarray, a_value: float = -10
) -> float:
    """
    Tính liều đồng đều tương đương (Equivalent Uniform Dose).

    EUD = (Σ(v_i × D_i^a))^(1/a)

    Parameters:
        dvh_data: Mảng các giá trị liều (Gy)
        volume_fractions: Mảng các phân đoạn thể tích tương ứng
        a_value: Tham số mô (-10 cho khối u, 1-5 cho mô lành)

    Returns:
        Liều đồng đều tương đương (Gy)
    """
    if len(dvh_data) == 0 or len(volume_fractions) == 0:
        logger.warning("Dữ liệu DVH trống.")
        return 0.0

    # Chuẩn hóa volume_fractions để tổng bằng 1
    volume_fractions_normalized = volume_fractions / np.sum(volume_fractions)

    # Tính EUD
    eud_sum = np.sum(volume_fractions_normalized * np.power(dvh_data, a_value))
    eud = np.power(eud_sum, 1.0 / a_value)

    return eud


def calculate_metrics_from_dvh(
    dvh: Any, prescription_dose_gy: float
) -> Dict[str, float]:
    """
    Tính toán các chỉ số đánh giá từ DVH.

    Parameters:
        dvh: Đối tượng DVH (DoseVolumeHistogram hoặc dict chứa dữ liệu DVH)
        prescription_dose_gy: Liều kê đơn (Gy)

    Returns:
        Dict chứa các chỉ số đánh giá
    """
    # Kiểm tra xem dvh có phải đối tượng DoseVolumeHistogram không
    if HAS_DVH and isinstance(dvh, DoseVolumeHistogram):
        # Lấy dữ liệu từ đối tượng DVH
        doses = dvh.doses
        volumes = dvh.volumes
        structure_name = dvh.structure_name
        structure_type = dvh.structure_type
    else:
        # Giả sử dvh là dict chứa dữ liệu cần thiết
        doses = dvh.get("doses", [])
        volumes = dvh.get("volumes", [])
        structure_name = dvh.get("structure_name", "Unknown")
        structure_type = dvh.get("structure_type", "Unknown")

    # Khởi tạo dict kết quả
    metrics = {}

    try:
        # Tính các chỉ số cơ bản
        if len(doses) > 0 and len(volumes) > 0:
            metrics["min_dose"] = float(np.min(doses))
            metrics["max_dose"] = float(np.max(doses))
            metrics["mean_dose"] = float(np.average(doses, weights=volumes))

            # Tính D95, D98, D50, D2
            dvh_metrics = DVHMetrics(doses, volumes) if HAS_DVH else None

            if dvh_metrics:
                d95 = dvh_metrics.get_dose_at_volume(95.0)
                d98 = dvh_metrics.get_dose_at_volume(98.0)
                d50 = dvh_metrics.get_dose_at_volume(50.0)
                d2 = dvh_metrics.get_dose_at_volume(2.0)

                metrics["D95"] = float(d95)
                metrics["D98"] = float(d98)
                metrics["D50"] = float(d50)
                metrics["D2"] = float(d2)

                # Tính V95, V100, V107 (% thể tích nhận 95%, 100%, 107% liều kê đơn)
                v95 = dvh_metrics.get_volume_at_dose(0.95 * prescription_dose_gy)
                v100 = dvh_metrics.get_volume_at_dose(prescription_dose_gy)
                v107 = dvh_metrics.get_volume_at_dose(1.07 * prescription_dose_gy)

                metrics["V95"] = float(v95)
                metrics["V100"] = float(v100)
                metrics["V107"] = float(v107)

                # Chỉ tính các chỉ số nâng cao cho cấu trúc PTV
                if "ptv" in structure_name.lower() or (
                    structure_type and "ptv" in structure_type.lower()
                ):
                    # Tính chỉ số đồng đều
                    metrics["HI"] = calculate_homogeneity_index(
                        d2, d98, prescription_dose_gy
                    )

                    # Giả định các thể tích cần thiết cho các chỉ số khác
                    # Trong thực tế, cần tính toán chính xác từ phân phối liều 3D
                    metrics["CI"] = v100 / 100.0  # Ước lượng conformity index
            else:
                logger.warning(
                    f"Không thể tính DVH metrics cho cấu trúc {structure_name}"
                )
    except Exception as e:
        logger.error(f"Lỗi khi tính toán chỉ số từ DVH: {e}")

    return metrics


def calculate_all_metrics(
    target_dvh: Any,
    normal_tissue_dvhs: List[Any],
    prescription_dose_gy: float,
    dose_grid: Optional[np.ndarray] = None,
    target_mask: Optional[np.ndarray] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Tính toán tất cả các chỉ số đánh giá cho một kế hoạch.

    Parameters:
        target_dvh: DVH của cấu trúc khối u
        normal_tissue_dvhs: Danh sách DVH của các cấu trúc mô lành
        prescription_dose_gy: Liều kê đơn (Gy)
        dose_grid: Mảng 3D chứa phân phối liều (tùy chọn)
        target_mask: Mảng 3D chứa mask của khối u (tùy chọn)

    Returns:
        Dict chứa tất cả các chỉ số đánh giá
    """
    result = {"target": {}, "normal_tissues": {}, "plan": {}}

    # Tính chỉ số cho khối u
    if target_dvh:
        result["target"] = calculate_metrics_from_dvh(target_dvh, prescription_dose_gy)

    # Tính chỉ số cho các mô lành
    if normal_tissue_dvhs:
        for i, dvh in enumerate(normal_tissue_dvhs):
            if HAS_DVH and isinstance(dvh, DoseVolumeHistogram):
                name = dvh.structure_name
            else:
                name = dvh.get("structure_name", f"OAR_{i + 1}")

            result["normal_tissues"][name] = calculate_metrics_from_dvh(
                dvh, prescription_dose_gy
            )

    # Tính các chỉ số toàn cục nếu có dose_grid và target_mask
    if dose_grid is not None and target_mask is not None:
        try:
            # Tính thể tích nhận liều kê đơn
            prescription_dose_voxels = dose_grid >= prescription_dose_gy
            half_prescription_dose_voxels = dose_grid >= 0.5 * prescription_dose_gy

            # Số voxel trong mỗi vùng
            target_voxels = np.sum(target_mask)
            prescription_dose_voxels_count = np.sum(prescription_dose_voxels)
            half_prescription_dose_voxels_count = np.sum(half_prescription_dose_voxels)

            # Vùng giao
            target_prescription_voxels = np.sum(target_mask & prescription_dose_voxels)

            # Tính các chỉ số
            if target_voxels > 0 and prescription_dose_voxels_count > 0:
                result["plan"]["CI"] = calculate_conformity_index(
                    target_voxels, prescription_dose_voxels_count
                )

                result["plan"]["Paddick_CI"] = calculate_paddick_ci(
                    target_prescription_voxels,
                    target_voxels,
                    prescription_dose_voxels_count,
                )

                result["plan"]["GI"] = calculate_gradient_index(
                    half_prescription_dose_voxels_count, prescription_dose_voxels_count
                )

                result["plan"]["Coverage"] = calculate_coverage_index(
                    target_prescription_voxels, target_voxels
                )

                result["plan"]["CN"] = calculate_conformation_number(
                    target_prescription_voxels,
                    target_voxels,
                    prescription_dose_voxels_count,
                )

                result["plan"]["HTCI"] = calculate_healthy_tissue_conformity_index(
                    target_prescription_voxels, prescription_dose_voxels_count
                )
        except Exception as e:
            logger.error(f"Lỗi khi tính toán các chỉ số từ dose_grid: {e}")

    return result


def evaluate_plan_quality(metrics: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    """
    Đánh giá chất lượng kế hoạch dựa trên các chỉ số.

    Parameters:
        metrics: Dict chứa các chỉ số đánh giá

    Returns:
        Dict chứa đánh giá chất lượng
    """
    evaluation = {"score": 0.0, "grade": "Unknown", "comments": [], "details": {}}

    try:
        # Đánh giá khối u
        target_metrics = metrics.get("target", {})
        if target_metrics:
            # Đánh giá coverage
            d95 = target_metrics.get("D95", 0)
            v95 = target_metrics.get("V95", 0)
            hi = target_metrics.get("HI", float("inf"))

            target_score = 0
            target_comments = []

            # Đánh giá coverage
            if v95 >= 99:
                target_score += 40
                target_comments.append("Coverage xuất sắc (V95 ≥ 99%)")
            elif v95 >= 95:
                target_score += 30
                target_comments.append("Coverage tốt (V95 ≥ 95%)")
            elif v95 >= 90:
                target_score += 20
                target_comments.append("Coverage chấp nhận được (V95 ≥ 90%)")
            else:
                target_comments.append("Coverage kém (V95 < 90%)")

            # Đánh giá đồng đều
            if hi <= 0.05:
                target_score += 40
                target_comments.append("Đồng đều xuất sắc (HI ≤ 0.05)")
            elif hi <= 0.07:
                target_score += 30
                target_comments.append("Đồng đều tốt (HI ≤ 0.07)")
            elif hi <= 0.1:
                target_score += 20
                target_comments.append("Đồng đều chấp nhận được (HI ≤ 0.1)")
            else:
                target_comments.append("Đồng đều kém (HI > 0.1)")

            # Đánh giá hot spots
            v107 = target_metrics.get("V107", 0)
            if v107 <= 1:
                target_score += 20
                target_comments.append("Không có hot spots đáng kể (V107 ≤ 1%)")
            elif v107 <= 2:
                target_score += 15
                target_comments.append("Hot spots trong giới hạn (V107 ≤ 2%)")
            elif v107 <= 5:
                target_score += 10
                target_comments.append("Hot spots chấp nhận được (V107 ≤ 5%)")
            else:
                target_comments.append("Hot spots đáng kể (V107 > 5%)")

            evaluation["details"]["target"] = {
                "score": target_score,
                "comments": target_comments,
            }

        # Đánh giá mô lành
        oar_metrics = metrics.get("normal_tissues", {})
        if oar_metrics:
            oar_score = 100
            oar_comments = []

            # Đánh giá từng cơ quan
            for organ, organ_metrics in oar_metrics.items():
                # Logic đánh giá riêng cho từng loại cơ quan sẽ phức tạp hơn
                # Đây chỉ là ví dụ đơn giản
                mean_dose = organ_metrics.get("mean_dose", 0)
                max_dose = organ_metrics.get("max_dose", 0)

                if "spinal" in organ.lower():
                    if max_dose > 45:
                        oar_score -= 20
                        oar_comments.append(
                            f"{organ}: Vượt giới hạn liều tối đa (Dmax > 45 Gy)"
                        )
                elif "parotid" in organ.lower():
                    if mean_dose > 26:
                        oar_score -= 10
                        oar_comments.append(
                            f"{organ}: Vượt giới hạn liều trung bình (Dmean > 26 Gy)"
                        )

            evaluation["details"]["oars"] = {
                "score": max(0, oar_score),
                "comments": oar_comments,
            }

        # Đánh giá chỉ số toàn cục
        plan_metrics = metrics.get("plan", {})
        if plan_metrics:
            plan_score = 0
            plan_comments = []

            # Đánh giá Conformity
            ci = plan_metrics.get("CI", float("inf"))
            if 0.9 <= ci <= 1.1:
                plan_score += 40
                plan_comments.append("Conformity xuất sắc (0.9 ≤ CI ≤ 1.1)")
            elif 0.8 <= ci <= 1.2:
                plan_score += 30
                plan_comments.append("Conformity tốt (0.8 ≤ CI ≤ 1.2)")
            elif 0.7 <= ci <= 1.3:
                plan_score += 20
                plan_comments.append("Conformity chấp nhận được (0.7 ≤ CI ≤ 1.3)")
            else:
                plan_comments.append("Conformity kém (CI < 0.7 hoặc CI > 1.3)")

            # Đánh giá Gradient
            gi = plan_metrics.get("GI", float("inf"))
            if gi < 3:
                plan_score += 30
                plan_comments.append("Gradient xuất sắc (GI < 3)")
            elif gi < 4:
                plan_score += 20
                plan_comments.append("Gradient tốt (GI < 4)")
            elif gi < 5:
                plan_score += 10
                plan_comments.append("Gradient chấp nhận được (GI < 5)")
            else:
                plan_comments.append("Gradient kém (GI ≥ 5)")

            # Đánh giá Coverage
            coverage = plan_metrics.get("Coverage", 0)
            if coverage >= 0.95:
                plan_score += 30
                plan_comments.append("Coverage xuất sắc (≥ 95%)")
            elif coverage >= 0.9:
                plan_score += 20
                plan_comments.append("Coverage tốt (≥ 90%)")
            elif coverage >= 0.8:
                plan_score += 10
                plan_comments.append("Coverage chấp nhận được (≥ 80%)")
            else:
                plan_comments.append("Coverage kém (< 80%)")

            evaluation["details"]["plan"] = {
                "score": min(100, plan_score),
                "comments": plan_comments,
            }

        # Tính điểm tổng thể
        total_score = 0
        count = 0

        if "target" in evaluation["details"]:
            total_score += evaluation["details"]["target"]["score"]
            count += 1
            evaluation["comments"].extend(evaluation["details"]["target"]["comments"])

        if "oars" in evaluation["details"]:
            total_score += evaluation["details"]["oars"]["score"]
            count += 1
            evaluation["comments"].extend(evaluation["details"]["oars"]["comments"])

        if "plan" in evaluation["details"]:
            total_score += evaluation["details"]["plan"]["score"]
            count += 1
            evaluation["comments"].extend(evaluation["details"]["plan"]["comments"])

        if count > 0:
            evaluation["score"] = total_score / count

        # Xác định xếp hạng
        if evaluation["score"] >= 90:
            evaluation["grade"] = "Excellent"
        elif evaluation["score"] >= 80:
            evaluation["grade"] = "Good"
        elif evaluation["score"] >= 70:
            evaluation["grade"] = "Acceptable"
        elif evaluation["score"] >= 50:
            evaluation["grade"] = "Marginal"
        else:
            evaluation["grade"] = "Poor"

    except Exception as e:
        logger.error(f"Lỗi khi đánh giá chất lượng kế hoạch: {e}")
        evaluation["comments"].append(f"Lỗi đánh giá: {str(e)}")

    return evaluation


# Export
__all__ = [
    "calculate_conformity_index",
    "calculate_paddick_ci",
    "calculate_gradient_index",
    "calculate_homogeneity_index",
    "calculate_coverage_index",
    "calculate_conformation_number",
    "calculate_healthy_tissue_conformity_index",
    "calculate_van_t_riet_index",
    "calculate_radiation_conformity_index",
    "calculate_modified_gradient_index",
    "calculate_equivalent_uniform_dose",
    "calculate_metrics_from_dvh",
    "calculate_all_metrics",
    "evaluate_plan_quality",
]
