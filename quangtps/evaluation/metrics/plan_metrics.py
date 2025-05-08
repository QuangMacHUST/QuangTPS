#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module tính toán các chỉ số đánh giá kế hoạch xạ trị.

Module này cung cấp các chức năng để tính toán và đánh giá các chỉ số
quan trọng của kế hoạch xạ trị, bao gồm các chỉ số lâm sàng, chỉ số chất lượng,
và phân tích sự khác biệt giữa các kế hoạch.
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, Sequence, Set
import pandas as pd
import matplotlib.pyplot as plt

from quangtps.core.types import Plan, Structure
from quangtps.evaluation.dvh.dvh_calculator import DVHCalculator
from quangtps.evaluation.metrics.clinical_metrics import calculate_clinical_metrics
from quangtps.evaluation.metrics.quality_metrics import calculate_quality_metrics
from quangtps.evaluation.metrics.conformity import calculate_conformity_indices
from quangtps.evaluation.metrics.homogeneity import calculate_homogeneity_indices
from quangtps.evaluation.metrics.gradient import calculate_gradient_indices
from quangtps.evaluation.metrics.paddick import calculate_paddick_indices
from quangtps.evaluation.metrics.hotspot import calculate_hotspot_metrics
from quangtps.evaluation.metrics.integral import calculate_integral_metrics
from quangtps.evaluation.metrics.edge import calculate_edge_metrics
from quangtps.evaluation.metrics.biological import calculate_biological_indices
from quangtps.evaluation.metrics.radiobiological import (
    calculate_radiobiological_metrics,
)

logger = logging.getLogger(__name__)


def calculate_plan_metrics(
    plan: Plan,
    structures: Dict[str, Structure] = None,
    metrics_type: List[str] = None,
    reference_dose: float = None,
    include_dvh: bool = True,
) -> Dict[str, Any]:
    """
    Tính toán các chỉ số đánh giá cho một kế hoạch xạ trị.

    Parameters
    ----------
    plan : Plan
        Kế hoạch xạ trị cần đánh giá
    structures : Dict[str, Structure], optional
        Từ điển các cấu trúc, mặc định là None (lấy từ kế hoạch)
    metrics_type : List[str], optional
        Danh sách các loại chỉ số cần tính, mặc định là None (tất cả)
    reference_dose : float, optional
        Liều tham chiếu (Gy), mặc định là None (lấy từ kế hoạch)
    include_dvh : bool, optional
        Có tính DVH không, mặc định là True

    Returns
    -------
    Dict[str, Any]
        Từ điển chứa các chỉ số đánh giá
    """
    # Nếu không cung cấp structures, lấy từ kế hoạch
    if structures is None:
        structures = plan.get_structures()

    # Nếu không cung cấp metrics_type, sử dụng tất cả
    if metrics_type is None:
        metrics_type = [
            "clinical",
            "quality",
            "conformity",
            "homogeneity",
            "gradient",
            "paddick",
            "hotspot",
            "integral",
            "edge",
            "biological",
            "radiobiological",
        ]

    # Nếu không cung cấp reference_dose, lấy từ kế hoạch
    if reference_dose is None:
        reference_dose = plan.get_prescription_dose()

    # Tính toán DVH nếu cần
    dvh_data = None
    if include_dvh:
        dvh_calculator = DVHCalculator()
        dvh_data = dvh_calculator.calculate_dvh(plan, structures)

    # Khởi tạo từ điển chứa kết quả
    results = {
        "plan_id": plan.get_id(),
        "plan_name": plan.get_name(),
        "prescription_dose": reference_dose,
        "structures": list(structures.keys()),
    }

    # Tính toán các chỉ số theo loại
    for metric_type in metrics_type:
        if metric_type == "clinical":
            clinical_metrics = calculate_clinical_metrics(
                plan, structures, reference_dose, dvh_data
            )
            results["clinical"] = clinical_metrics

        elif metric_type == "quality":
            quality_metrics = calculate_quality_metrics(
                plan, structures, reference_dose, dvh_data
            )
            results["quality"] = quality_metrics

        elif metric_type == "conformity":
            conformity_indices = calculate_conformity_indices(
                plan, structures, reference_dose, dvh_data
            )
            results["conformity"] = conformity_indices

        elif metric_type == "homogeneity":
            homogeneity_indices = calculate_homogeneity_indices(
                plan, structures, reference_dose, dvh_data
            )
            results["homogeneity"] = homogeneity_indices

        elif metric_type == "gradient":
            gradient_indices = calculate_gradient_indices(
                plan, structures, reference_dose, dvh_data
            )
            results["gradient"] = gradient_indices

        elif metric_type == "paddick":
            paddick_indices = calculate_paddick_indices(
                plan, structures, reference_dose, dvh_data
            )
            results["paddick"] = paddick_indices

        elif metric_type == "hotspot":
            hotspot_metrics = calculate_hotspot_metrics(
                plan, structures, reference_dose, dvh_data
            )
            results["hotspot"] = hotspot_metrics

        elif metric_type == "integral":
            integral_metrics = calculate_integral_metrics(
                plan, structures, reference_dose, dvh_data
            )
            results["integral"] = integral_metrics

        elif metric_type == "edge":
            edge_metrics = calculate_edge_metrics(
                plan, structures, reference_dose, dvh_data
            )
            results["edge"] = edge_metrics

        elif metric_type == "biological":
            biological_indices = calculate_biological_indices(
                plan, structures, reference_dose, dvh_data
            )
            results["biological"] = biological_indices

        elif metric_type == "radiobiological":
            radiobiological_metrics = calculate_radiobiological_metrics(
                plan, structures, reference_dose, dvh_data
            )
            results["radiobiological"] = radiobiological_metrics

    return results


def compare_plan_metrics(
    plan1_metrics: Dict[str, Any],
    plan2_metrics: Dict[str, Any],
    metrics_to_compare: List[str] = None,
    threshold: float = 0.05,
) -> Dict[str, Any]:
    """
    So sánh các chỉ số đánh giá giữa hai kế hoạch.

    Parameters
    ----------
    plan1_metrics : Dict[str, Any]
        Chỉ số đánh giá của kế hoạch thứ nhất
    plan2_metrics : Dict[str, Any]
        Chỉ số đánh giá của kế hoạch thứ hai
    metrics_to_compare : List[str], optional
        Danh sách các loại chỉ số cần so sánh, mặc định là None (tất cả)
    threshold : float, optional
        Ngưỡng cho sự khác biệt có ý nghĩa (tỷ lệ), mặc định là 0.05 (5%)

    Returns
    -------
    Dict[str, Any]
        Từ điển chứa kết quả so sánh
    """
    # Nếu không cung cấp metrics_to_compare, sử dụng tất cả
    if metrics_to_compare is None:
        metrics_to_compare = [
            "clinical",
            "quality",
            "conformity",
            "homogeneity",
            "gradient",
            "paddick",
            "hotspot",
            "integral",
            "edge",
            "biological",
            "radiobiological",
        ]

    # Khởi tạo từ điển chứa kết quả
    comparison_results = {
        "plan1_id": plan1_metrics.get("plan_id", "Plan 1"),
        "plan2_id": plan2_metrics.get("plan_id", "Plan 2"),
        "metrics_compared": metrics_to_compare,
        "threshold": threshold,
        "results": {},
    }

    # So sánh các chỉ số theo loại
    for metric_type in metrics_to_compare:
        if metric_type in plan1_metrics and metric_type in plan2_metrics:
            metrics1 = plan1_metrics[metric_type]
            metrics2 = plan2_metrics[metric_type]

            # Khởi tạo từ điển chứa kết quả so sánh cho loại chỉ số này
            metric_comparison = {"structures": {}, "overall": {}}

            # So sánh cho từng cấu trúc
            for structure in set(metrics1.keys()).intersection(set(metrics2.keys())):
                if isinstance(metrics1[structure], dict) and isinstance(
                    metrics2[structure], dict
                ):
                    structure_metrics1 = metrics1[structure]
                    structure_metrics2 = metrics2[structure]

                    structure_comparison = {}

                    # So sánh từng chỉ số
                    for metric_name in set(structure_metrics1.keys()).intersection(
                        set(structure_metrics2.keys())
                    ):
                        value1 = structure_metrics1[metric_name]
                        value2 = structure_metrics2[metric_name]

                        if isinstance(value1, (int, float)) and isinstance(
                            value2, (int, float)
                        ):
                            # Tính sự khác biệt tuyệt đối và tương đối
                            abs_diff = value2 - value1
                            rel_diff = 0.0

                            if abs(value1) > 1e-10:  # Tránh chia cho 0
                                rel_diff = abs_diff / abs(value1)

                            # Xác định có phải sự khác biệt có ý nghĩa không
                            is_significant = abs(rel_diff) > threshold

                            # Lưu kết quả
                            structure_comparison[metric_name] = {
                                "value1": value1,
                                "value2": value2,
                                "abs_diff": abs_diff,
                                "rel_diff": rel_diff,
                                "is_significant": is_significant,
                            }

                    # Lưu kết quả so sánh cho cấu trúc này
                    metric_comparison["structures"][structure] = structure_comparison

            # Lưu kết quả so sánh cho loại chỉ số này
            comparison_results["results"][metric_type] = metric_comparison

    return comparison_results


def analyze_plan_robustness(
    plan: Plan,
    structures: Dict[str, Structure],
    perturbed_plans: List[Plan],
    metrics_type: List[str] = None,
    reference_dose: float = None,
) -> Dict[str, Any]:
    """
    Phân tích độ bền vững của kế hoạch dựa trên các kế hoạch bị nhiễu loạn.

    Parameters
    ----------
    plan : Plan
        Kế hoạch gốc cần phân tích
    structures : Dict[str, Structure]
        Từ điển các cấu trúc
    perturbed_plans : List[Plan]
        Danh sách các kế hoạch bị nhiễu loạn
    metrics_type : List[str], optional
        Danh sách các loại chỉ số cần tính, mặc định là None (tất cả)
    reference_dose : float, optional
        Liều tham chiếu (Gy), mặc định là None (lấy từ kế hoạch)

    Returns
    -------
    Dict[str, Any]
        Từ điển chứa kết quả phân tích độ bền vững
    """
    # Tính chỉ số cho kế hoạch gốc
    original_metrics = calculate_plan_metrics(
        plan, structures, metrics_type, reference_dose
    )

    # Tính chỉ số cho các kế hoạch bị nhiễu loạn
    perturbed_metrics = []
    for perturbed_plan in perturbed_plans:
        metrics = calculate_plan_metrics(
            perturbed_plan, structures, metrics_type, reference_dose
        )
        perturbed_metrics.append(metrics)

    # Khởi tạo từ điển chứa kết quả
    robustness_results = {
        "plan_id": plan.get_id(),
        "plan_name": plan.get_name(),
        "num_perturbed_plans": len(perturbed_plans),
        "metrics_type": metrics_type,
        "original_metrics": original_metrics,
        "perturbed_metrics": perturbed_metrics,
        "statistics": {},
        "robustness_indices": {},
    }

    # Tính toán thống kê cho từng loại chỉ số
    for metric_type in metrics_type:
        if metric_type in original_metrics:
            metric_statistics = {}

            # Trích xuất chỉ số từ kế hoạch gốc
            original_metric_data = original_metrics[metric_type]

            # Trích xuất chỉ số từ các kế hoạch bị nhiễu loạn
            perturbed_metric_data = [
                p_metrics[metric_type]
                for p_metrics in perturbed_metrics
                if metric_type in p_metrics
            ]

            # Tính toán thống kê cho từng cấu trúc và chỉ số
            for structure in original_metric_data:
                if isinstance(original_metric_data[structure], dict):
                    structure_metrics = original_metric_data[structure]

                    # Khởi tạo từ điển chỉ số cho cấu trúc này
                    structure_statistics = {}

                    # Tính toán cho từng chỉ số
                    for metric_name in structure_metrics:
                        # Lấy giá trị gốc
                        original_value = structure_metrics[metric_name]

                        # Lấy giá trị từ các kế hoạch bị nhiễu loạn
                        perturbed_values = []
                        for p_data in perturbed_metric_data:
                            if structure in p_data and isinstance(
                                p_data[structure], dict
                            ):
                                if metric_name in p_data[structure]:
                                    perturbed_values.append(
                                        p_data[structure][metric_name]
                                    )

                        # Tính toán thống kê nếu có đủ dữ liệu
                        if len(perturbed_values) > 0 and isinstance(
                            original_value, (int, float)
                        ):
                            # Chuyển đổi sang numpy array
                            perturbed_array = np.array(perturbed_values, dtype=float)

                            # Tính toán thống kê
                            mean_value = np.mean(perturbed_array)
                            std_value = np.std(perturbed_array)
                            min_value = np.min(perturbed_array)
                            max_value = np.max(perturbed_array)
                            median_value = np.median(perturbed_array)

                            # Tính độ bền vững
                            robustness = (
                                1.0 - std_value / abs(mean_value)
                                if abs(mean_value) > 1e-10
                                else 0.0
                            )

                            # Lưu kết quả
                            structure_statistics[metric_name] = {
                                "original": original_value,
                                "mean": mean_value,
                                "std": std_value,
                                "min": min_value,
                                "max": max_value,
                                "median": median_value,
                                "robustness": robustness,
                                "coefficient_of_variation": std_value / abs(mean_value)
                                if abs(mean_value) > 1e-10
                                else float("inf"),
                            }

                    # Lưu thống kê cho cấu trúc này
                    metric_statistics[structure] = structure_statistics

            # Lưu thống kê cho loại chỉ số này
            robustness_results["statistics"][metric_type] = metric_statistics

            # Tính toán chỉ số độ bền vững tổng thể (tổng hợp)
            robustness_indices = {}
            for structure in metric_statistics:
                structure_robustness = []

                for metric_name in metric_statistics[structure]:
                    if "robustness" in metric_statistics[structure][metric_name]:
                        structure_robustness.append(
                            metric_statistics[structure][metric_name]["robustness"]
                        )

                if len(structure_robustness) > 0:
                    robustness_indices[structure] = np.mean(structure_robustness)

            # Lưu chỉ số độ bền vững tổng thể
            robustness_results["robustness_indices"][metric_type] = robustness_indices

    return robustness_results


def plot_plan_comparison(
    plan1_metrics: Dict[str, Any],
    plan2_metrics: Dict[str, Any],
    structure: str,
    metric_type: str,
    output_file: str = None,
    title: str = None,
) -> plt.Figure:
    """
    Tạo biểu đồ so sánh các chỉ số giữa hai kế hoạch cho một cấu trúc và loại chỉ số.

    Parameters
    ----------
    plan1_metrics : Dict[str, Any]
        Chỉ số đánh giá của kế hoạch thứ nhất
    plan2_metrics : Dict[str, Any]
        Chỉ số đánh giá của kế hoạch thứ hai
    structure : str
        Tên cấu trúc cần so sánh
    metric_type : str
        Loại chỉ số cần so sánh
    output_file : str, optional
        Đường dẫn đến file đầu ra, mặc định là None (không lưu)
    title : str, optional
        Tiêu đề biểu đồ, mặc định là None

    Returns
    -------
    plt.Figure
        Đối tượng Figure chứa biểu đồ
    """
    # Kiểm tra dữ liệu đầu vào
    if (
        metric_type not in plan1_metrics
        or metric_type not in plan2_metrics
        or structure not in plan1_metrics[metric_type]
        or structure not in plan2_metrics[metric_type]
    ):
        logger.error(f"Dữ liệu không đủ để so sánh {metric_type} cho {structure}")
        return None

    # Lấy dữ liệu chỉ số
    metrics1 = plan1_metrics[metric_type][structure]
    metrics2 = plan2_metrics[metric_type][structure]

    # Tìm các chỉ số chung
    common_metrics = set(metrics1.keys()).intersection(set(metrics2.keys()))

    # Lọc ra các chỉ số số học
    numeric_metrics = []
    for metric in common_metrics:
        if isinstance(metrics1[metric], (int, float)) and isinstance(
            metrics2[metric], (int, float)
        ):
            numeric_metrics.append(metric)

    if len(numeric_metrics) == 0:
        logger.error(f"Không có chỉ số số học chung cho {metric_type} của {structure}")
        return None

    # Chuẩn bị dữ liệu cho biểu đồ
    metric_names = []
    plan1_values = []
    plan2_values = []

    for metric in numeric_metrics:
        metric_names.append(metric)
        plan1_values.append(metrics1[metric])
        plan2_values.append(metrics2[metric])

    # Tạo biểu đồ
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(metric_names))
    width = 0.35

    plan1_id = plan1_metrics.get("plan_id", "Plan 1")
    plan2_id = plan2_metrics.get("plan_id", "Plan 2")

    rects1 = ax.bar(x - width / 2, plan1_values, width, label=plan1_id)
    rects2 = ax.bar(x + width / 2, plan2_values, width, label=plan2_id)

    # Thêm nhãn, tiêu đề và chú giải
    ax.set_xlabel("Chỉ số")
    ax.set_ylabel("Giá trị")
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"So sánh chỉ số {metric_type} cho {structure}")

    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, rotation=45, ha="right")
    ax.legend()

    # Thêm nhãn giá trị
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.2f}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
            )

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()

    # Lưu biểu đồ nếu cần
    if output_file:
        fig.savefig(output_file, dpi=300, bbox_inches="tight")

    return fig


def generate_plan_metrics_report(
    plan_metrics: Dict[str, Any], format_type: str = "text", output_file: str = None
) -> Union[str, pd.DataFrame]:
    """
    Tạo báo cáo chỉ số đánh giá kế hoạch.

    Parameters
    ----------
    plan_metrics : Dict[str, Any]
        Chỉ số đánh giá kế hoạch
    format_type : str, optional
        Định dạng báo cáo: 'text', 'html', 'csv', 'dataframe', mặc định là 'text'
    output_file : str, optional
        Đường dẫn đến file đầu ra, mặc định là None (không lưu)

    Returns
    -------
    Union[str, pd.DataFrame]
        Báo cáo dưới dạng văn bản hoặc DataFrame
    """
    # Trích xuất thông tin kế hoạch
    plan_id = plan_metrics.get("plan_id", "Unknown")
    plan_name = plan_metrics.get("plan_name", "Unknown")
    prescription_dose = plan_metrics.get("prescription_dose", 0.0)

    # Chuẩn bị DataFrame cho báo cáo
    data = []
    columns = ["Structure", "Metric Type", "Metric Name", "Value"]

    # Thêm dữ liệu vào DataFrame
    for metric_type in plan_metrics:
        if metric_type in ["plan_id", "plan_name", "prescription_dose", "structures"]:
            continue

        metric_data = plan_metrics[metric_type]

        for structure in metric_data:
            if isinstance(metric_data[structure], dict):
                structure_metrics = metric_data[structure]

                for metric_name, value in structure_metrics.items():
                    if isinstance(value, (int, float)):
                        data.append([structure, metric_type, metric_name, value])

    # Tạo DataFrame
    df = pd.DataFrame(data, columns=columns)

    # Tạo báo cáo theo định dạng yêu cầu
    if format_type == "dataframe":
        result = df

        # Lưu báo cáo nếu cần
        if output_file:
            df.to_csv(output_file, index=False)

    elif format_type == "csv":
        result = df.to_csv(index=False)

        # Lưu báo cáo nếu cần
        if output_file:
            with open(output_file, "w") as f:
                f.write(result)

    elif format_type == "html":
        # Tạo báo cáo HTML
        html = f"""
        <html>
        <head>
            <title>Báo cáo chỉ số đánh giá kế hoạch</title>
            <style>
                body {{ font-family: Arial, sans-serif; }}
                h1, h2 {{ color: #2c3e50; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
            </style>
        </head>
        <body>
            <h1>Báo cáo chỉ số đánh giá kế hoạch</h1>
            <p><strong>Kế hoạch ID:</strong> {plan_id}</p>
            <p><strong>Tên kế hoạch:</strong> {plan_name}</p>
            <p><strong>Liều kê đơn:</strong> {prescription_dose} Gy</p>

            <h2>Chỉ số đánh giá</h2>
            {df.to_html(index=False)}
        </body>
        </html>
        """

        result = html

        # Lưu báo cáo nếu cần
        if output_file:
            with open(output_file, "w") as f:
                f.write(result)

    else:  # format_type == 'text'
        # Tạo báo cáo văn bản
        text = f"""
Báo cáo chỉ số đánh giá kế hoạch
================================

Kế hoạch ID: {plan_id}
Tên kế hoạch: {plan_name}
Liều kê đơn: {prescription_dose} Gy

Chỉ số đánh giá:
----------------

"""

        # Nhóm theo loại chỉ số và cấu trúc
        grouped = df.groupby(["Metric Type", "Structure"])

        for (metric_type, structure), group in grouped:
            text += f"{metric_type} - {structure}:\n"

            for _, row in group.iterrows():
                text += f"  {row['Metric Name']}: {row['Value']:.4f}\n"

            text += "\n"

        result = text

        # Lưu báo cáo nếu cần
        if output_file:
            with open(output_file, "w") as f:
                f.write(result)

    return result
