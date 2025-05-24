"""
QuangTPS Plan Quality Metrics Module

Module tính toán các chỉ số chất lượng kế hoạch xạ trị.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from scipy import ndimage, stats

logger = logging.getLogger(__name__)


@dataclass
class PlanQualityResults:
    """Kết quả đánh giá chất lượng kế hoạch."""

    conformity_index: float = 0.0
    homogeneity_index: float = 0.0
    gradient_index: float = 0.0
    coverage_index: float = 0.0
    spillage_index: float = 0.0
    paddick_conformity_index: float = 0.0
    new_conformity_index: float = 0.0

    # Target coverage metrics
    target_coverage_95: float = 0.0
    target_coverage_98: float = 0.0
    target_coverage_99: float = 0.0

    # Hot spot analysis
    hotspot_volume_105: float = 0.0
    hotspot_volume_107: float = 0.0
    hotspot_volume_110: float = 0.0

    # Monitor units
    total_monitor_units: float = 0.0
    monitor_unit_efficiency: float = 0.0


def calculate_conformity_index(
    dose_distribution: np.ndarray,
    target_mask: np.ndarray,
    prescription_dose: float,
    isodose_level: float = 0.95,
) -> float:
    """
    Tính Conformity Index (CI) cơ bản.

    Args:
        dose_distribution: Array phân phối liều
        target_mask: Mask cấu trúc mục tiêu
        prescription_dose: Liều chỉ định
        isodose_level: Mức isodose (0-1)

    Returns:
        float: Conformity Index
    """
    try:
        isodose_value = prescription_dose * isodose_level

        # Volume nhận isodose
        v_isodose = np.sum(dose_distribution >= isodose_value)

        # Volume target nhận isodose
        v_target_isodose = np.sum(
            (dose_distribution >= isodose_value) & (target_mask > 0)
        )

        if v_isodose == 0:
            return 0.0

        # CI = V_target_isodose / V_isodose
        ci = v_target_isodose / v_isodose

        return float(np.clip(ci, 0.0, 1.0))

    except Exception as e:
        logger.error(f"Error calculating conformity index: {e}")
        return 0.0


def calculate_paddick_conformity_index(
    dose_distribution: np.ndarray,
    target_mask: np.ndarray,
    prescription_dose: float,
    isodose_level: float = 0.95,
) -> float:
    """
    Tính Paddick Conformity Index (pCI).

    Args:
        dose_distribution: Array phân phối liều
        target_mask: Mask cấu trúc mục tiêu
        prescription_dose: Liều chỉ định
        isodose_level: Mức isodose (0-1)

    Returns:
        float: Paddick Conformity Index
    """
    try:
        isodose_value = prescription_dose * isodose_level

        # Volume nhận isodose
        v_isodose = np.sum(dose_distribution >= isodose_value)

        # Volume target
        v_target = np.sum(target_mask > 0)

        # Volume target nhận isodose
        v_target_isodose = np.sum(
            (dose_distribution >= isodose_value) & (target_mask > 0)
        )

        if v_target == 0 or v_isodose == 0:
            return 0.0

        # pCI = (V_target_isodose)² / (V_target × V_isodose)
        pci = (v_target_isodose**2) / (v_target * v_isodose)

        return float(np.clip(pci, 0.0, 1.0))

    except Exception as e:
        logger.error(f"Error calculating Paddick conformity index: {e}")
        return 0.0


def calculate_new_conformity_index(
    dose_distribution: np.ndarray,
    target_mask: np.ndarray,
    prescription_dose: float,
    isodose_level: float = 0.95,
) -> float:
    """
    Tính New Conformity Index (nCI).

    Args:
        dose_distribution: Array phân phối liều
        target_mask: Mask cấu trúc mục tiêu
        prescription_dose: Liều chỉ định
        isodose_level: Mức isodose (0-1)

    Returns:
        float: New Conformity Index
    """
    try:
        isodose_value = prescription_dose * isodose_level

        # Volume target
        v_target = np.sum(target_mask > 0)

        # Volume target nhận isodose
        v_target_isodose = np.sum(
            (dose_distribution >= isodose_value) & (target_mask > 0)
        )

        # Volume ngoài target nhận isodose
        v_outside_isodose = np.sum(
            (dose_distribution >= isodose_value) & (target_mask == 0)
        )

        if v_target == 0:
            return 0.0

        # nCI = V_target_isodose / V_target - V_outside_isodose / V_target
        nci = (v_target_isodose / v_target) - (v_outside_isodose / v_target)

        return float(np.clip(nci, -1.0, 1.0))

    except Exception as e:
        logger.error(f"Error calculating new conformity index: {e}")
        return 0.0


def calculate_homogeneity_index(
    dose_distribution: np.ndarray, target_mask: np.ndarray, prescription_dose: float
) -> float:
    """
    Tính Homogeneity Index (HI).

    Args:
        dose_distribution: Array phân phối liều
        target_mask: Mask cấu trúc mục tiêu
        prescription_dose: Liều chỉ định

    Returns:
        float: Homogeneity Index
    """
    try:
        # Lấy liều trong target
        target_doses = dose_distribution[target_mask > 0]

        if len(target_doses) == 0:
            return 0.0

        # D2% và D98%
        d2 = np.percentile(target_doses, 98)
        d98 = np.percentile(target_doses, 2)

        # HI = (D2% - D98%) / D_prescription
        hi = (d2 - d98) / prescription_dose

        return float(hi)

    except Exception as e:
        logger.error(f"Error calculating homogeneity index: {e}")
        return 0.0


def calculate_gradient_index(
    dose_distribution: np.ndarray,
    target_mask: np.ndarray,
    prescription_dose: float,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    reference_volume: float = 1.0,  # cm³
) -> float:
    """
    Tính Gradient Index (GI).

    Args:
        dose_distribution: Array phân phối liều
        target_mask: Mask cấu trúc mục tiêu
        prescription_dose: Liều chỉ định
        spacing: Khoảng cách voxel (mm)
        reference_volume: Thể tích tham chiếu (cm³)

    Returns:
        float: Gradient Index
    """
    try:
        # Volume target
        voxel_volume = np.prod(spacing) / 1000.0  # Convert mm³ to cm³
        v_target = np.sum(target_mask > 0) * voxel_volume

        # Volume nhận 50% liều chỉ định
        v_50 = np.sum(dose_distribution >= 0.5 * prescription_dose) * voxel_volume

        # GI = V_50% / V_target
        if v_target == 0:
            return 0.0

        # For small targets, use reference volume
        if v_target < reference_volume:
            gi = v_50 / reference_volume
        else:
            gi = v_50 / v_target

        return float(gi)

    except Exception as e:
        logger.error(f"Error calculating gradient index: {e}")
        return 0.0


def calculate_coverage_index(
    dose_distribution: np.ndarray,
    target_mask: np.ndarray,
    prescription_dose: float,
    coverage_level: float = 0.95,
) -> float:
    """
    Tính Coverage Index (COV).

    Args:
        dose_distribution: Array phân phối liều
        target_mask: Mask cấu trúc mục tiêu
        prescription_dose: Liều chỉ định
        coverage_level: Mức độ phủ (0-1)

    Returns:
        float: Coverage Index (%)
    """
    try:
        target_doses = dose_distribution[target_mask > 0]

        if len(target_doses) == 0:
            return 0.0

        coverage_dose = prescription_dose * coverage_level
        covered_voxels = np.sum(target_doses >= coverage_dose)
        total_voxels = len(target_doses)

        coverage_percent = (covered_voxels / total_voxels) * 100.0

        return float(coverage_percent)

    except Exception as e:
        logger.error(f"Error calculating coverage index: {e}")
        return 0.0


def calculate_spillage_index(
    dose_distribution: np.ndarray,
    target_mask: np.ndarray,
    prescription_dose: float,
    spillage_level: float = 0.5,
) -> float:
    """
    Tính Spillage Index (SI).

    Args:
        dose_distribution: Array phân phối liều
        target_mask: Mask cấu trúc mục tiêu
        prescription_dose: Liều chỉ định
        spillage_level: Mức spillage (0-1)

    Returns:
        float: Spillage Index
    """
    try:
        spillage_dose = prescription_dose * spillage_level

        # Volume ngoài target nhận spillage dose
        v_spillage = np.sum((dose_distribution >= spillage_dose) & (target_mask == 0))

        # Volume target
        v_target = np.sum(target_mask > 0)

        if v_target == 0:
            return 0.0

        # SI = V_spillage / V_target
        si = v_spillage / v_target

        return float(si)

    except Exception as e:
        logger.error(f"Error calculating spillage index: {e}")
        return 0.0


def calculate_hotspot_volumes(
    dose_distribution: np.ndarray,
    target_mask: np.ndarray,
    prescription_dose: float,
    hotspot_levels: List[float] = [1.05, 1.07, 1.10],
) -> Dict[str, float]:
    """
    Tính thể tích các hotspot ở các mức khác nhau.

    Args:
        dose_distribution: Array phân phối liều
        target_mask: Mask cấu trúc mục tiêu
        prescription_dose: Liều chỉ định
        hotspot_levels: List các mức hotspot (fraction of prescription)

    Returns:
        Dict[str, float]: Dictionary các thể tích hotspot
    """
    try:
        hotspot_volumes = {}
        v_target = np.sum(target_mask > 0)

        for level in hotspot_levels:
            hotspot_dose = prescription_dose * level

            # Volume trong target có hotspot
            v_hotspot = np.sum((dose_distribution >= hotspot_dose) & (target_mask > 0))

            # Phần trăm volume
            if v_target > 0:
                hotspot_percent = (v_hotspot / v_target) * 100.0
            else:
                hotspot_percent = 0.0

            hotspot_volumes[f"V{int(level * 100)}"] = float(hotspot_percent)

        return hotspot_volumes

    except Exception as e:
        logger.error(f"Error calculating hotspot volumes: {e}")
        return {}


def calculate_target_coverage_metrics(
    dose_distribution: np.ndarray,
    target_mask: np.ndarray,
    prescription_dose: float,
    coverage_levels: List[float] = [0.95, 0.98, 0.99],
) -> Dict[str, float]:
    """
    Tính metrics độ phủ target ở các mức khác nhau.

    Args:
        dose_distribution: Array phân phối liều
        target_mask: Mask cấu trúc mục tiêu
        prescription_dose: Liều chỉ định
        coverage_levels: List các mức coverage (fraction of prescription)

    Returns:
        Dict[str, float]: Dictionary các coverage metrics
    """
    try:
        coverage_metrics = {}
        target_doses = dose_distribution[target_mask > 0]

        if len(target_doses) == 0:
            return {f"D{int(level * 100)}": 0.0 for level in coverage_levels}

        for level in coverage_levels:
            coverage_dose = prescription_dose * level
            covered_voxels = np.sum(target_doses >= coverage_dose)
            total_voxels = len(target_doses)

            coverage_percent = (covered_voxels / total_voxels) * 100.0
            coverage_metrics[f"V{int(level * 100)}"] = float(coverage_percent)

        return coverage_metrics

    except Exception as e:
        logger.error(f"Error calculating target coverage metrics: {e}")
        return {}


def calculate_monitor_unit_efficiency(
    monitor_units: List[float], target_volume: float, prescription_dose: float
) -> float:
    """
    Tính hiệu suất Monitor Unit.

    Args:
        monitor_units: List MU của các beam
        target_volume: Thể tích target (cm³)
        prescription_dose: Liều chỉ định (Gy)

    Returns:
        float: MU efficiency (MU/Gy/cm³)
    """
    try:
        total_mu = sum(monitor_units)

        if target_volume == 0 or prescription_dose == 0:
            return 0.0

        # MU efficiency = Total MU / (Prescription Dose × Target Volume)
        mu_efficiency = total_mu / (prescription_dose * target_volume)

        return float(mu_efficiency)

    except Exception as e:
        logger.error(f"Error calculating MU efficiency: {e}")
        return 0.0


def calculate_dose_uniformity(
    dose_distribution: np.ndarray, structure_mask: np.ndarray
) -> float:
    """
    Tính độ đồng đều liều.

    Args:
        dose_distribution: Array phân phối liều
        structure_mask: Mask cấu trúc

    Returns:
        float: Dose uniformity (coefficient of variation)
    """
    try:
        structure_doses = dose_distribution[structure_mask > 0]

        if len(structure_doses) == 0:
            return 0.0

        mean_dose = np.mean(structure_doses)
        std_dose = np.std(structure_doses)

        if mean_dose == 0:
            return 0.0

        # Coefficient of variation
        uniformity = std_dose / mean_dose

        return float(uniformity)

    except Exception as e:
        logger.error(f"Error calculating dose uniformity: {e}")
        return 0.0


def calculate_comprehensive_plan_quality(
    dose_distribution: np.ndarray,
    target_mask: np.ndarray,
    prescription_dose: float,
    monitor_units: Optional[List[float]] = None,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> PlanQualityResults:
    """
    Tính toán comprehensive plan quality metrics.

    Args:
        dose_distribution: Array phân phối liều
        target_mask: Mask cấu trúc mục tiêu
        prescription_dose: Liều chỉ định
        monitor_units: List MU của các beam (optional)
        spacing: Khoảng cách voxel (mm)

    Returns:
        PlanQualityResults: Tất cả metrics chất lượng kế hoạch
    """
    try:
        # Basic indices
        ci = calculate_conformity_index(
            dose_distribution, target_mask, prescription_dose
        )
        hi = calculate_homogeneity_index(
            dose_distribution, target_mask, prescription_dose
        )
        gi = calculate_gradient_index(
            dose_distribution, target_mask, prescription_dose, spacing
        )
        cov = calculate_coverage_index(
            dose_distribution, target_mask, prescription_dose
        )
        si = calculate_spillage_index(dose_distribution, target_mask, prescription_dose)

        # Advanced conformity indices
        pci = calculate_paddick_conformity_index(
            dose_distribution, target_mask, prescription_dose
        )
        nci = calculate_new_conformity_index(
            dose_distribution, target_mask, prescription_dose
        )

        # Target coverage at different levels
        coverage_metrics = calculate_target_coverage_metrics(
            dose_distribution, target_mask, prescription_dose, [0.95, 0.98, 0.99]
        )

        # Hotspot analysis
        hotspot_volumes = calculate_hotspot_volumes(
            dose_distribution, target_mask, prescription_dose, [1.05, 1.07, 1.10]
        )

        # Monitor unit efficiency
        total_mu = 0.0
        mu_efficiency = 0.0
        if monitor_units:
            total_mu = sum(monitor_units)
            voxel_volume = np.prod(spacing) / 1000.0  # Convert to cm³
            target_volume = np.sum(target_mask > 0) * voxel_volume
            mu_efficiency = calculate_monitor_unit_efficiency(
                monitor_units, target_volume, prescription_dose
            )

        return PlanQualityResults(
            conformity_index=ci,
            homogeneity_index=hi,
            gradient_index=gi,
            coverage_index=cov,
            spillage_index=si,
            paddick_conformity_index=pci,
            new_conformity_index=nci,
            target_coverage_95=coverage_metrics.get("V95", 0.0),
            target_coverage_98=coverage_metrics.get("V98", 0.0),
            target_coverage_99=coverage_metrics.get("V99", 0.0),
            hotspot_volume_105=hotspot_volumes.get("V105", 0.0),
            hotspot_volume_107=hotspot_volumes.get("V107", 0.0),
            hotspot_volume_110=hotspot_volumes.get("V110", 0.0),
            total_monitor_units=total_mu,
            monitor_unit_efficiency=mu_efficiency,
        )

    except Exception as e:
        logger.error(f"Error calculating comprehensive plan quality: {e}")
        return PlanQualityResults()


def evaluate_plan_quality_score(quality_results: PlanQualityResults) -> float:
    """
    Đánh giá tổng thể chất lượng kế hoạch thành điểm số.

    Args:
        quality_results: Kết quả metrics chất lượng

    Returns:
        float: Điểm chất lượng (0-100)
    """
    try:
        score = 0.0
        weight_sum = 0.0

        # Conformity (weight: 25%)
        if quality_results.conformity_index > 0:
            ci_score = min(quality_results.conformity_index * 100, 100)
            score += ci_score * 0.25
            weight_sum += 0.25

        # Homogeneity (weight: 20%)
        if quality_results.homogeneity_index >= 0:
            # Lower HI is better, ideal < 0.1
            hi_score = max(0, 100 - quality_results.homogeneity_index * 500)
            score += hi_score * 0.20
            weight_sum += 0.20

        # Coverage (weight: 25%)
        if quality_results.coverage_index > 0:
            score += quality_results.coverage_index * 0.25
            weight_sum += 0.25

        # Target coverage V95 (weight: 15%)
        if quality_results.target_coverage_95 > 0:
            score += quality_results.target_coverage_95 * 0.15
            weight_sum += 0.15

        # Hotspot penalty (weight: 10%)
        hotspot_penalty = quality_results.hotspot_volume_110 * 2.0  # Penalize V110
        hotspot_score = max(0, 100 - hotspot_penalty)
        score += hotspot_score * 0.10
        weight_sum += 0.10

        # Spillage penalty (weight: 5%)
        spillage_penalty = min(quality_results.spillage_index * 20, 100)
        spillage_score = max(0, 100 - spillage_penalty)
        score += spillage_score * 0.05
        weight_sum += 0.05

        # Normalize by actual weights used
        if weight_sum > 0:
            score = score / weight_sum
        else:
            score = 0.0

        return float(np.clip(score, 0.0, 100.0))

    except Exception as e:
        logger.error(f"Error evaluating plan quality score: {e}")
        return 0.0


__all__ = [
    "PlanQualityResults",
    "calculate_conformity_index",
    "calculate_paddick_conformity_index",
    "calculate_new_conformity_index",
    "calculate_homogeneity_index",
    "calculate_gradient_index",
    "calculate_coverage_index",
    "calculate_spillage_index",
    "calculate_hotspot_volumes",
    "calculate_target_coverage_metrics",
    "calculate_monitor_unit_efficiency",
    "calculate_dose_uniformity",
    "calculate_comprehensive_plan_quality",
    "evaluate_plan_quality_score",
]
