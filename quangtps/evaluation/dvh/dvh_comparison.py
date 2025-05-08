#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module so sánh DVH giữa các kế hoạch xạ trị hoặc giữa DVH kế hoạch và đo đạc.

Module này cung cấp các công cụ để so sánh định lượng các DVH, tính toán
chỉ số khác biệt, và tạo biểu đồ so sánh trực quan.
"""

import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any, Union
from scipy.interpolate import interp1d
import pandas as pd

logger = logging.getLogger(__name__)


# Thêm hàm helper để lấy colormap an toàn
def _get_safe_colormap(cmap_name="tab10", fallback_cmap="jet", num_colors=10):
    """
    Lấy colormap an toàn, với phương án dự phòng nếu không có colormap yêu cầu.

    Parameters
    ----------
    cmap_name : str
        Tên colormap ưu tiên
    fallback_cmap : str
        Tên colormap dự phòng
    num_colors : int
        Số lượng màu cần lấy

    Returns
    -------
    numpy.ndarray
        Mảng các màu RGB
    """
    try:
        # Kiểm tra xem colormap có tồn tại
        cmap = getattr(plt.cm, cmap_name, None)
        if cmap is None:
            logger.warning(
                f"Colormap {cmap_name} không khả dụng, sử dụng {fallback_cmap} thay thế"
            )
            cmap = getattr(plt.cm, fallback_cmap)
        return cmap(np.linspace(0, 1, num_colors))
    except Exception as e:
        logger.warning(f"Lỗi khi lấy colormap: {e}. Sử dụng giải pháp thay thế.")
        # Trường hợp không có colormap nào hoạt động, tạo các màu cơ bản
        base_colors = [
            [0, 0, 1],  # blue
            [0, 0.5, 0],  # green
            [1, 0, 0],  # red
            [0.5, 0, 0.5],  # purple
            [1, 0.5, 0],  # orange
            [0, 0.5, 0.5],  # teal
            [0.5, 0.5, 0],  # olive
            [0, 0, 0.5],  # navy
            [0.5, 0, 0],  # maroon
            [0.5, 0.5, 0.5],  # gray
        ]
        if num_colors <= len(base_colors):
            return np.array(base_colors[:num_colors])
        else:
            # Nếu cần nhiều màu hơn, nhân bản và điều chỉnh độ sáng
            result = []
            for i in range(num_colors):
                color = base_colors[i % len(base_colors)].copy()
                brightness = 0.7 + (i // len(base_colors)) * 0.3
                color = [min(x * brightness, 1.0) for x in color]
                result.append(color)
            return np.array(result)


def compare_dvhs(
    reference_dvhs: Dict[str, Dict[str, List[float]]],
    evaluation_dvhs: Dict[str, Dict[str, List[float]]],
    metrics: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    So sánh DVH giữa kế hoạch tham chiếu và kế hoạch/đo đạc cần đánh giá.

    Parameters
    ----------
    reference_dvhs : Dict[str, Dict[str, List[float]]]
        DVH tham chiếu (thường là từ kế hoạch), định dạng:
        {structure_id: {"dose": [...], "volume": [...]}}
    evaluation_dvhs : Dict[str, Dict[str, List[float]]]
        DVH cần đánh giá (thường là đo đạc), định dạng tương tự reference_dvhs
    metrics : Optional[List[str]], optional
        Danh sách các chỉ số cần tính toán, mặc định là None (tính tất cả)

    Returns
    -------
    Dict[str, Any]
        Kết quả so sánh chứa chỉ số tổng quát và chi tiết cho từng cấu trúc
    """
    if metrics is None:
        metrics = [
            "similarity_index",
            "area_difference",
            "max_dose_diff",
            "mean_dose_diff",
            "d95_diff",
            "d90_diff",
            "d50_diff",
            "v20_diff",
            "v10_diff",
            "v5_diff",
        ]

    results = {"overall_similarity": 0.0, "structures": {}}

    # Tìm các cấu trúc chung
    common_structures = set(reference_dvhs.keys()) & set(evaluation_dvhs.keys())

    if not common_structures:
        logger.warning("Không tìm thấy cấu trúc chung giữa hai DVH")
        return results

    total_similarity = 0.0

    for structure_id in common_structures:
        ref_dvh = reference_dvhs[structure_id]
        eval_dvh = evaluation_dvhs[structure_id]

        # Kiểm tra dữ liệu đầu vào
        if (
            "dose" not in ref_dvh
            or "volume" not in ref_dvh
            or "dose" not in eval_dvh
            or "volume" not in eval_dvh
        ):
            logger.warning(f"Dữ liệu DVH không hợp lệ cho cấu trúc {structure_id}")
            continue

        # Chuyển đổi sang numpy array
        ref_dose = np.array(ref_dvh["dose"])
        ref_volume = np.array(ref_dvh["volume"])
        eval_dose = np.array(eval_dvh["dose"])
        eval_volume = np.array(eval_dvh["volume"])

        # Tạo dữ liệu chuẩn hóa để so sánh
        max_dose = max(np.max(ref_dose), np.max(eval_dose))
        dose_range = np.linspace(0, max_dose, 100)

        # Nội suy cho cả hai DVH
        ref_interp = interp1d(
            ref_dose, ref_volume, bounds_error=False, fill_value=(100, 0)
        )
        eval_interp = interp1d(
            eval_dose, eval_volume, bounds_error=False, fill_value=(100, 0)
        )

        ref_volume_interp = ref_interp(dose_range)
        eval_volume_interp = eval_interp(dose_range)

        # Điền NaN bằng giá trị thích hợp
        ref_volume_interp = np.nan_to_num(ref_volume_interp, nan=0.0)
        eval_volume_interp = np.nan_to_num(eval_volume_interp, nan=0.0)

        # Tính toán các chỉ số
        structure_results = {}

        # Chỉ số tương đồng (1 - khoảng cách Euclidean chuẩn hóa)
        if "similarity_index" in metrics:
            euclidean_distance = np.sqrt(
                np.mean((ref_volume_interp - eval_volume_interp) ** 2)
            )
            similarity = 1.0 - min(1.0, euclidean_distance / 100.0)
            structure_results["similarity_index"] = similarity
            total_similarity += similarity

        # Chênh lệch diện tích dưới đường cong
        if "area_difference" in metrics:
            ref_area = np.trapz(ref_volume_interp, dose_range)
            eval_area = np.trapz(eval_volume_interp, dose_range)
            area_diff_percent = 100 * abs(ref_area - eval_area) / max(ref_area, 1e-10)
            structure_results["area_difference"] = area_diff_percent

        # Chênh lệch liều tối đa
        if "max_dose_diff" in metrics:
            ref_max_dose = np.max(ref_dose)
            eval_max_dose = np.max(eval_dose)
            max_dose_diff = abs(ref_max_dose - eval_max_dose)
            max_dose_diff_percent = 100 * max_dose_diff / max(ref_max_dose, 1e-10)
            structure_results["max_dose_diff"] = max_dose_diff
            structure_results["max_dose_diff_percent"] = max_dose_diff_percent

        # Liều trung bình
        if "mean_dose_diff" in metrics:
            # Tính liều trung bình từ DVH
            ref_mean_dose = np.trapz(ref_volume_interp, dose_range) / 100  # Chuẩn hóa
            eval_mean_dose = np.trapz(eval_volume_interp, dose_range) / 100
            mean_dose_diff = abs(ref_mean_dose - eval_mean_dose)
            mean_dose_diff_percent = 100 * mean_dose_diff / max(ref_mean_dose, 1e-10)
            structure_results["mean_dose_diff"] = mean_dose_diff
            structure_results["mean_dose_diff_percent"] = mean_dose_diff_percent

        # Các chỉ số D95, D90, D50
        if "d95_diff" in metrics:
            ref_d95 = _find_dose_at_volume(ref_dose, ref_volume, 95)
            eval_d95 = _find_dose_at_volume(eval_dose, eval_volume, 95)
            if ref_d95 is not None and eval_d95 is not None:
                d95_diff = abs(ref_d95 - eval_d95)
                d95_diff_percent = 100 * d95_diff / max(ref_d95, 1e-10)
                structure_results["d95_diff"] = d95_diff
                structure_results["d95_diff_percent"] = d95_diff_percent
                structure_results["ref_d95"] = ref_d95
                structure_results["eval_d95"] = eval_d95

        if "d90_diff" in metrics:
            ref_d90 = _find_dose_at_volume(ref_dose, ref_volume, 90)
            eval_d90 = _find_dose_at_volume(eval_dose, eval_volume, 90)
            if ref_d90 is not None and eval_d90 is not None:
                d90_diff = abs(ref_d90 - eval_d90)
                d90_diff_percent = 100 * d90_diff / max(ref_d90, 1e-10)
                structure_results["d90_diff"] = d90_diff
                structure_results["d90_diff_percent"] = d90_diff_percent
                structure_results["ref_d90"] = ref_d90
                structure_results["eval_d90"] = eval_d90

        if "d50_diff" in metrics:
            ref_d50 = _find_dose_at_volume(ref_dose, ref_volume, 50)
            eval_d50 = _find_dose_at_volume(eval_dose, eval_volume, 50)
            if ref_d50 is not None and eval_d50 is not None:
                d50_diff = abs(ref_d50 - eval_d50)
                d50_diff_percent = 100 * d50_diff / max(ref_d50, 1e-10)
                structure_results["d50_diff"] = d50_diff
                structure_results["d50_diff_percent"] = d50_diff_percent
                structure_results["ref_d50"] = ref_d50
                structure_results["eval_d50"] = eval_d50

        # Các chỉ số V20, V10, V5
        if "v20_diff" in metrics:
            ref_v20 = _find_volume_at_dose(ref_dose, ref_volume, 20)
            eval_v20 = _find_volume_at_dose(eval_dose, eval_volume, 20)
            if ref_v20 is not None and eval_v20 is not None:
                v20_diff = abs(ref_v20 - eval_v20)
                structure_results["v20_diff"] = v20_diff
                structure_results["ref_v20"] = ref_v20
                structure_results["eval_v20"] = eval_v20

        if "v10_diff" in metrics:
            ref_v10 = _find_volume_at_dose(ref_dose, ref_volume, 10)
            eval_v10 = _find_volume_at_dose(eval_dose, eval_volume, 10)
            if ref_v10 is not None and eval_v10 is not None:
                v10_diff = abs(ref_v10 - eval_v10)
                structure_results["v10_diff"] = v10_diff
                structure_results["ref_v10"] = ref_v10
                structure_results["eval_v10"] = eval_v10

        if "v5_diff" in metrics:
            ref_v5 = _find_volume_at_dose(ref_dose, ref_volume, 5)
            eval_v5 = _find_volume_at_dose(eval_dose, eval_volume, 5)
            if ref_v5 is not None and eval_v5 is not None:
                v5_diff = abs(ref_v5 - eval_v5)
                structure_results["v5_diff"] = v5_diff
                structure_results["ref_v5"] = ref_v5
                structure_results["eval_v5"] = eval_v5

        # Lưu dữ liệu DVH gốc
        structure_results["plan_dvh"] = {
            "dose": ref_dose.tolist(),
            "volume": ref_volume.tolist(),
        }
        structure_results["measured_dvh"] = {
            "dose": eval_dose.tolist(),
            "volume": eval_volume.tolist(),
        }

        # Lưu kết quả cho cấu trúc này
        results["structures"][structure_id] = structure_results

    # Tính chỉ số tương đồng tổng quát
    if common_structures:
        results["overall_similarity"] = total_similarity / len(common_structures)

    return results


def _find_dose_at_volume(
    doses: np.ndarray, volumes: np.ndarray, volume_percent: float
) -> Optional[float]:
    """
    Tìm liều lượng tại một phần trăm thể tích cho trước từ DVH.

    Parameters
    ----------
    doses : np.ndarray
        Mảng giá trị liều lượng
    volumes : np.ndarray
        Mảng giá trị thể tích tương ứng (%)
    volume_percent : float
        Giá trị phần trăm thể tích cần tìm liều lượng

    Returns
    -------
    Optional[float]
        Giá trị liều lượng tại phần trăm thể tích cho trước, hoặc None nếu không tìm thấy
    """
    try:
        # Kiểm tra đầu vào
        if len(doses) < 2 or len(volumes) < 2:
            return None

        # Đảm bảo dữ liệu được sắp xếp theo liều tăng dần
        sort_idx = np.argsort(doses)
        doses_sorted = doses[sort_idx]
        volumes_sorted = volumes[sort_idx]

        # DVH tích lũy (thường giảm dần theo liều)
        # Đảm bảo thể tích theo thứ tự giảm dần
        if volumes_sorted[0] < volumes_sorted[-1]:
            volumes_sorted = 100 - volumes_sorted

        # Tìm giá trị liều lượng tại phần trăm thể tích
        interp = interp1d(
            volumes_sorted, doses_sorted, bounds_error=False, fill_value=np.nan
        )
        result = float(interp(volume_percent))

        return None if np.isnan(result) else result

    except Exception as e:
        logger.error(f"Lỗi khi tìm liều lượng tại thể tích {volume_percent}%: {str(e)}")
        return None


def _find_volume_at_dose(
    doses: np.ndarray, volumes: np.ndarray, dose_value: float
) -> Optional[float]:
    """
    Tìm phần trăm thể tích tại một liều lượng cho trước từ DVH.

    Parameters
    ----------
    doses : np.ndarray
        Mảng giá trị liều lượng
    volumes : np.ndarray
        Mảng giá trị thể tích tương ứng (%)
    dose_value : float
        Giá trị liều lượng cần tìm thể tích

    Returns
    -------
    Optional[float]
        Giá trị phần trăm thể tích tại liều lượng cho trước, hoặc None nếu không tìm thấy
    """
    try:
        # Kiểm tra đầu vào
        if len(doses) < 2 or len(volumes) < 2:
            return None

        # Đảm bảo dữ liệu được sắp xếp theo liều tăng dần
        sort_idx = np.argsort(doses)
        doses_sorted = doses[sort_idx]
        volumes_sorted = volumes[sort_idx]

        # DVH tích lũy (thường giảm dần theo liều)
        # Đảm bảo thể tích theo thứ tự giảm dần
        if volumes_sorted[0] < volumes_sorted[-1]:
            volumes_sorted = 100 - volumes_sorted

        # Kiểm tra nếu liều lượng nằm ngoài phạm vi
        if dose_value < np.min(doses_sorted) or dose_value > np.max(doses_sorted):
            return None

        # Tìm giá trị thể tích tại liều lượng
        interp = interp1d(
            doses_sorted, volumes_sorted, bounds_error=False, fill_value=np.nan
        )
        result = float(interp(dose_value))

        return None if np.isnan(result) else result

    except Exception as e:
        logger.error(f"Lỗi khi tìm thể tích tại liều {dose_value}: {str(e)}")
        return None


def plot_dvh_comparison(
    reference_dvhs: Dict[str, Dict[str, List[float]]],
    evaluation_dvhs: Dict[str, Dict[str, List[float]]],
    structure_ids: Optional[List[str]] = None,
    reference_label: str = "Plan",
    evaluation_label: str = "Measured",
    output_file: Optional[str] = None,
    show_metrics: bool = True,
) -> plt.Figure:
    """
    Tạo biểu đồ so sánh DVH giữa phân bố tham chiếu và cần đánh giá.

    Parameters
    ----------
    reference_dvhs : Dict[str, Dict[str, List[float]]]
        DVH tham chiếu
    evaluation_dvhs : Dict[str, Dict[str, List[float]]]
        DVH cần đánh giá
    structure_ids : Optional[List[str]], optional
        Danh sách ID cấu trúc cần vẽ, mặc định là None (vẽ tất cả cấu trúc chung)
    reference_label : str, optional
        Nhãn cho DVH tham chiếu, mặc định là "Plan"
    evaluation_label : str, optional
        Nhãn cho DVH cần đánh giá, mặc định là "Measured"
    output_file : Optional[str], optional
        Đường dẫn file đầu ra, mặc định là None (không lưu file)
    show_metrics : bool, optional
        Hiển thị các chỉ số trên biểu đồ, mặc định là True

    Returns
    -------
    plt.Figure
        Đối tượng Figure chứa biểu đồ
    """
    # Tìm các cấu trúc chung
    common_structures = set(reference_dvhs.keys()) & set(evaluation_dvhs.keys())

    if not common_structures:
        logger.warning("Không tìm thấy cấu trúc chung giữa hai DVH")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "Không có dữ liệu để hiển thị", ha="center", va="center")
        return fig

    # Lọc cấu trúc cần vẽ
    if structure_ids is not None:
        structures_to_plot = [s for s in structure_ids if s in common_structures]
        if not structures_to_plot:
            logger.warning("Không tìm thấy cấu trúc được chỉ định trong dữ liệu DVH")
            structures_to_plot = list(common_structures)
    else:
        structures_to_plot = list(common_structures)

    # Tạo biểu đồ
    fig, ax = plt.subplots(figsize=(12, 8))

    # Vùng chú thích
    legend_entries = []

    # Tính toán các chỉ số nếu cần
    metrics_data = {}
    if show_metrics:
        comparison_results = compare_dvhs(reference_dvhs, evaluation_dvhs)

    # Mảng màu cho các cấu trúc - Sử dụng hàm helper để lấy colormap an toàn
    colors = _get_safe_colormap(cmap_name="tab10", num_colors=len(structures_to_plot))

    # Vẽ từng cấu trúc
    for i, structure_id in enumerate(structures_to_plot):
        color = colors[i]

        # Lấy dữ liệu DVH
        ref_dvh = reference_dvhs[structure_id]
        eval_dvh = evaluation_dvhs[structure_id]

        # Kiểm tra dữ liệu
        if (
            "dose" not in ref_dvh
            or "volume" not in ref_dvh
            or "dose" not in eval_dvh
            or "volume" not in eval_dvh
        ):
            logger.warning(f"Dữ liệu DVH không hợp lệ cho cấu trúc {structure_id}")
            continue

        # Vẽ DVH tham chiếu
        (ref_line,) = ax.plot(
            ref_dvh["dose"],
            ref_dvh["volume"],
            linestyle="-",
            color=color,
            linewidth=2,
            label=f"{structure_id} ({reference_label})",
        )

        # Vẽ DVH cần đánh giá
        (eval_line,) = ax.plot(
            eval_dvh["dose"],
            eval_dvh["volume"],
            linestyle="--",
            color=color,
            linewidth=2,
            label=f"{structure_id} ({evaluation_label})",
        )

        legend_entries.extend([ref_line, eval_line])

        # Thêm chỉ số nếu cần
        if show_metrics and structure_id in comparison_results["structures"]:
            metrics_data[structure_id] = comparison_results["structures"][structure_id]

    # Thiết lập biểu đồ
    ax.set_xlabel("Liều lượng (Gy)")
    ax.set_ylabel("Thể tích (%)")
    ax.set_title(f"So sánh DVH: {reference_label} vs {evaluation_label}")
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 105)

    # Thêm chú thích
    ax.legend(
        handles=legend_entries, loc="best", bbox_to_anchor=(1.05, 1), borderaxespad=0
    )

    # Thêm bảng chỉ số nếu cần
    if show_metrics and metrics_data:
        metrics_text = "Chỉ số so sánh:\n"
        metrics_text += (
            f"Tương đồng tổng thể: {comparison_results['overall_similarity']:.2f}\n\n"
        )

        for structure_id, metrics in metrics_data.items():
            if "similarity_index" in metrics:
                metrics_text += f"{structure_id}:\n"
                metrics_text += f"- Tương đồng: {metrics['similarity_index']:.2f}\n"

            if "d95_diff" in metrics and "ref_d95" in metrics and "eval_d95" in metrics:
                metrics_text += f"- D95: {metrics['ref_d95']:.2f} vs {metrics['eval_d95']:.2f} ({metrics['d95_diff']:.2f} Gy)\n"

            if "v20_diff" in metrics and "ref_v20" in metrics and "eval_v20" in metrics:
                metrics_text += f"- V20: {metrics['ref_v20']:.1f}% vs {metrics['eval_v20']:.1f}% ({metrics['v20_diff']:.1f}%)\n"

            metrics_text += "\n"

        # Thêm văn bản vào một vị trí thích hợp
        plt.figtext(1.05, 0.5, metrics_text, va="center", fontsize=9)

    # Điều chỉnh layout
    plt.tight_layout()

    # Lưu file nếu cần
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")

    return fig


def create_dvh_comparison_table(
    reference_dvhs: Dict[str, Dict[str, List[float]]],
    evaluation_dvhs: Dict[str, Dict[str, List[float]]],
    structure_ids: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Tạo bảng so sánh định lượng giữa hai bộ DVH.

    Parameters
    ----------
    reference_dvhs : Dict[str, Dict[str, List[float]]]
        DVH tham chiếu
    evaluation_dvhs : Dict[str, Dict[str, List[float]]]
        DVH cần đánh giá
    structure_ids : Optional[List[str]], optional
        Danh sách ID cấu trúc cần đưa vào bảng, mặc định là None (tất cả cấu trúc chung)

    Returns
    -------
    pd.DataFrame
        Bảng dữ liệu chứa thông tin so sánh
    """
    # Tính toán các chỉ số so sánh
    comparison_results = compare_dvhs(reference_dvhs, evaluation_dvhs)

    # Tìm các cấu trúc chung
    common_structures = set(reference_dvhs.keys()) & set(evaluation_dvhs.keys())

    # Lọc cấu trúc cần đưa vào bảng
    if structure_ids is not None:
        structures_to_include = [s for s in structure_ids if s in common_structures]
        if not structures_to_include:
            logger.warning("Không tìm thấy cấu trúc được chỉ định trong dữ liệu DVH")
            structures_to_include = list(common_structures)
    else:
        structures_to_include = list(common_structures)

    # Tạo dữ liệu cho bảng
    table_data = []

    for structure_id in structures_to_include:
        if structure_id in comparison_results["structures"]:
            metrics = comparison_results["structures"][structure_id]

            row = {
                "Cấu trúc": structure_id,
                "Chỉ số tương đồng": f"{metrics.get('similarity_index', 0):.3f}",
            }

            # Thêm các chỉ số D95, D90, D50
            if "ref_d95" in metrics and "eval_d95" in metrics:
                row["D95 (Kế hoạch)"] = f"{metrics['ref_d95']:.2f}"
                row["D95 (Đo đạc)"] = f"{metrics['eval_d95']:.2f}"
                row["Chênh lệch D95 (Gy)"] = f"{metrics.get('d95_diff', 0):.2f}"

            if "ref_d90" in metrics and "eval_d90" in metrics:
                row["D90 (Kế hoạch)"] = f"{metrics['ref_d90']:.2f}"
                row["D90 (Đo đạc)"] = f"{metrics['eval_d90']:.2f}"
                row["Chênh lệch D90 (Gy)"] = f"{metrics.get('d90_diff', 0):.2f}"

            if "ref_d50" in metrics and "eval_d50" in metrics:
                row["D50 (Kế hoạch)"] = f"{metrics['ref_d50']:.2f}"
                row["D50 (Đo đạc)"] = f"{metrics['eval_d50']:.2f}"
                row["Chênh lệch D50 (Gy)"] = f"{metrics.get('d50_diff', 0):.2f}"

            # Thêm các chỉ số V20, V10, V5
            if "ref_v20" in metrics and "eval_v20" in metrics:
                row["V20 (Kế hoạch)"] = f"{metrics['ref_v20']:.1f}%"
                row["V20 (Đo đạc)"] = f"{metrics['eval_v20']:.1f}%"
                row["Chênh lệch V20 (%)"] = f"{metrics.get('v20_diff', 0):.1f}"

            if "ref_v10" in metrics and "eval_v10" in metrics:
                row["V10 (Kế hoạch)"] = f"{metrics['ref_v10']:.1f}%"
                row["V10 (Đo đạc)"] = f"{metrics['eval_v10']:.1f}%"
                row["Chênh lệch V10 (%)"] = f"{metrics.get('v10_diff', 0):.1f}"

            if "ref_v5" in metrics and "eval_v5" in metrics:
                row["V5 (Kế hoạch)"] = f"{metrics['ref_v5']:.1f}%"
                row["V5 (Đo đạc)"] = f"{metrics['eval_v5']:.1f}%"
                row["Chênh lệch V5 (%)"] = f"{metrics.get('v5_diff', 0):.1f}"

            table_data.append(row)

    # Tạo DataFrame
    if table_data:
        return pd.DataFrame(table_data)
    else:
        # Trả về DataFrame rỗng nếu không có dữ liệu
        return pd.DataFrame(columns=["Cấu trúc", "Chỉ số tương đồng"])
