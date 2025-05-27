#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Robustness Analysis Module

Module này cung cấp các hàm phân tích độ bền vững cho kế hoạch xạ trị.
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)


def calculate_robustness_metrics(
    nominal_dose: np.ndarray,
    scenario_doses: List[np.ndarray],
    structure_masks: Dict[str, np.ndarray],
    prescription_dose: float = 60.0,
) -> Dict[str, Any]:
    """
    Tính toán các chỉ số độ bền vững.

    Parameters
    ----------
    nominal_dose : np.ndarray
        Phân bố liều danh định
    scenario_doses : List[np.ndarray]
        Danh sách phân bố liều cho các kịch bản
    structure_masks : Dict[str, np.ndarray]
        Masks của các cấu trúc
    prescription_dose : float
        Liều kê đơn

    Returns
    -------
    Dict[str, Any]
        Các chỉ số độ bền vững
    """
    try:
        metrics = {}

        for structure_name, mask in structure_masks.items():
            structure_metrics = {}

            # Tính toán cho liều danh định
            nominal_structure_dose = nominal_dose[mask > 0]
            structure_metrics["nominal"] = {
                "mean_dose": np.mean(nominal_structure_dose),
                "max_dose": np.max(nominal_structure_dose),
                "min_dose": np.min(nominal_structure_dose),
                "d95": np.percentile(nominal_structure_dose, 5),
                "d5": np.percentile(nominal_structure_dose, 95),
            }

            # Tính toán cho các kịch bản
            scenario_metrics = []
            for scenario_dose in scenario_doses:
                scenario_structure_dose = scenario_dose[mask > 0]
                scenario_metrics.append(
                    {
                        "mean_dose": np.mean(scenario_structure_dose),
                        "max_dose": np.max(scenario_structure_dose),
                        "min_dose": np.min(scenario_structure_dose),
                        "d95": np.percentile(scenario_structure_dose, 5),
                        "d5": np.percentile(scenario_structure_dose, 95),
                    }
                )

            structure_metrics["scenarios"] = scenario_metrics

            # Tính toán độ biến thiên
            for metric_name in ["mean_dose", "max_dose", "min_dose", "d95", "d5"]:
                values = [s[metric_name] for s in scenario_metrics]
                structure_metrics[f"{metric_name}_variation"] = {
                    "min": np.min(values),
                    "max": np.max(values),
                    "std": np.std(values),
                    "range": np.max(values) - np.min(values),
                }

            metrics[structure_name] = structure_metrics

        logger.info(f"Calculated robustness metrics for {len(metrics)} structures")
        return metrics

    except Exception as e:
        logger.error(f"Error calculating robustness metrics: {e}")
        return {}


def robustness_dvh_bands(
    nominal_dose: np.ndarray,
    scenario_doses: List[np.ndarray],
    structure_masks: Dict[str, np.ndarray],
    dose_bins: Optional[np.ndarray] = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Tính toán DVH bands cho phân tích độ bền vững.

    Parameters
    ----------
    nominal_dose : np.ndarray
        Phân bố liều danh định
    scenario_doses : List[np.ndarray]
        Danh sách phân bố liều cho các kịch bản
    structure_masks : Dict[str, np.ndarray]
        Masks của các cấu trúc
    dose_bins : Optional[np.ndarray]
        Bins liều cho DVH

    Returns
    -------
    Dict[str, Dict[str, np.ndarray]]
        DVH bands cho từng cấu trúc
    """
    try:
        if dose_bins is None:
            max_dose = max(np.max(nominal_dose), max(np.max(d) for d in scenario_doses))
            dose_bins = np.linspace(0, max_dose * 1.1, 100)

        dvh_bands = {}

        for structure_name, mask in structure_masks.items():
            # Tính DVH cho liều danh định
            nominal_structure_dose = nominal_dose[mask > 0]
            nominal_dvh = _calculate_dvh(nominal_structure_dose, dose_bins)

            # Tính DVH cho các kịch bản
            scenario_dvhs = []
            for scenario_dose in scenario_doses:
                scenario_structure_dose = scenario_dose[mask > 0]
                scenario_dvh = _calculate_dvh(scenario_structure_dose, dose_bins)
                scenario_dvhs.append(scenario_dvh)

            # Tính toán bands
            scenario_dvhs = np.array(scenario_dvhs)
            dvh_bands[structure_name] = {
                "dose_bins": dose_bins,
                "nominal": nominal_dvh,
                "min_band": np.min(scenario_dvhs, axis=0),
                "max_band": np.max(scenario_dvhs, axis=0),
                "mean_band": np.mean(scenario_dvhs, axis=0),
                "std_band": np.std(scenario_dvhs, axis=0),
            }

        logger.info(f"Calculated DVH bands for {len(dvh_bands)} structures")
        return dvh_bands

    except Exception as e:
        logger.error(f"Error calculating DVH bands: {e}")
        return {}


def find_worst_case_scenario(
    robustness_results: Dict[str, Any],
    target_structures: List[str] = None,
    metric: str = "d95",
) -> Dict[str, Any]:
    """
    Tìm kịch bản tệ nhất dựa trên các chỉ số.

    Parameters
    ----------
    robustness_results : Dict[str, Any]
        Kết quả phân tích độ bền vững
    target_structures : List[str]
        Danh sách cấu trúc mục tiêu
    metric : str
        Chỉ số để đánh giá

    Returns
    -------
    Dict[str, Any]
        Thông tin kịch bản tệ nhất
    """
    try:
        if not robustness_results:
            return {}

        # Kiểm tra nếu robustness_results không phải dictionary
        if not isinstance(robustness_results, dict):
            logger.warning(
                f"robustness_results is not a dict: {type(robustness_results)}"
            )
            return {"scenario_id": "unknown", "description": "Invalid results format"}

        worst_case = {
            "scenario_id": "worst_case",
            "description": f"Worst case based on {metric}",
            "affected_structures": [],
            "severity_score": 0.0,
        }

        # Tính toán điểm nghiêm trọng
        total_score = 0.0
        structure_count = 0

        for structure_name, structure_data in robustness_results.items():
            if target_structures and structure_name not in target_structures:
                continue

            # Kiểm tra structure_data có phải dictionary không
            if not isinstance(structure_data, dict):
                logger.warning(
                    f"structure_data for {structure_name} is not a dict: {type(structure_data)}"
                )
                continue

            if metric in structure_data:
                variation = structure_data[f"{metric}_variation"]
                severity = variation.get("range", 0.0) / structure_data["nominal"].get(
                    metric, 1.0
                )
                total_score += severity
                structure_count += 1

                if severity > 0.1:  # 10% variation threshold
                    worst_case["affected_structures"].append(
                        {
                            "name": structure_name,
                            "severity": severity,
                            "variation_range": variation.get("range", 0.0),
                        }
                    )

        if structure_count > 0:
            worst_case["severity_score"] = total_score / structure_count

        logger.info(
            f"Found worst case scenario with severity score: {worst_case['severity_score']:.3f}"
        )
        return worst_case

    except Exception as e:
        logger.error(f"Error finding worst case scenario: {e}")
        return {}


def _calculate_dvh(dose_values: np.ndarray, dose_bins: np.ndarray) -> np.ndarray:
    """
    Tính toán DVH từ giá trị liều.

    Parameters
    ----------
    dose_values : np.ndarray
        Giá trị liều
    dose_bins : np.ndarray
        Bins liều

    Returns
    -------
    np.ndarray
        DVH values
    """
    try:
        total_volume = len(dose_values)
        if total_volume == 0:
            return np.zeros_like(dose_bins)

        dvh = np.zeros_like(dose_bins)

        for i, dose_threshold in enumerate(dose_bins):
            volume_above = np.sum(dose_values >= dose_threshold)
            dvh[i] = volume_above / total_volume * 100.0

        return dvh

    except Exception as e:
        logger.error(f"Error calculating DVH: {e}")
        return np.zeros_like(dose_bins)
