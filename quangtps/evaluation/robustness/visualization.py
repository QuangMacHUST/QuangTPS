#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Robustness Visualization Module

Module này cung cấp các hàm để trực quan hóa kết quả phân tích độ bền vững.
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

# Fallback cho matplotlib
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.warning("Matplotlib không khả dụng. Visualization sẽ bị giới hạn.")


def plot_robustness_metrics(
    robustness_results: Dict[str, Any],
    structures: Optional[List[str]] = None,
    metrics: Optional[List[str]] = None,
    save_path: Optional[str] = None,
) -> bool:
    """
    Vẽ biểu đồ các chỉ số độ bền vững.

    Parameters
    ----------
    robustness_results : Dict[str, Any]
        Kết quả phân tích độ bền vững
    structures : Optional[List[str]]
        Danh sách cấu trúc cần vẽ
    metrics : Optional[List[str]]
        Danh sách chỉ số cần vẽ
    save_path : Optional[str]
        Đường dẫn lưu file

    Returns
    -------
    bool
        True nếu thành công
    """
    if not HAS_MATPLOTLIB:
        logger.warning("Matplotlib không khả dụng. Không thể vẽ biểu đồ.")
        return False

    try:
        if not robustness_results:
            logger.warning("Không có dữ liệu để vẽ biểu đồ.")
            return False

        if structures is None:
            structures = list(robustness_results.keys())

        if metrics is None:
            metrics = ["mean_dose", "max_dose", "d95"]

        fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 4 * len(metrics)))
        if len(metrics) == 1:
            axes = [axes]

        for i, metric in enumerate(metrics):
            ax = axes[i]

            structure_names = []
            nominal_values = []
            min_values = []
            max_values = []

            for structure in structures:
                if structure in robustness_results:
                    data = robustness_results[structure]
                    if metric in data.get("nominal", {}):
                        structure_names.append(structure)
                        nominal_values.append(data["nominal"][metric])

                        variation = data.get(f"{metric}_variation", {})
                        min_values.append(variation.get("min", data["nominal"][metric]))
                        max_values.append(variation.get("max", data["nominal"][metric]))

            if structure_names:
                x_pos = np.arange(len(structure_names))

                # Vẽ nominal values
                ax.bar(x_pos, nominal_values, alpha=0.7, label="Nominal")

                # Vẽ error bars cho variation
                errors = [
                    [nom - min_val for nom, min_val in zip(nominal_values, min_values)],
                    [max_val - nom for nom, max_val in zip(nominal_values, max_values)],
                ]
                ax.errorbar(
                    x_pos,
                    nominal_values,
                    yerr=errors,
                    fmt="none",
                    capsize=5,
                    capthick=2,
                    color="red",
                    label="Variation Range",
                )

                ax.set_xlabel("Structures")
                ax.set_ylabel(f"{metric} (Gy)")
                ax.set_title(f"Robustness Analysis - {metric}")
                ax.set_xticks(x_pos)
                ax.set_xticklabels(structure_names, rotation=45, ha="right")
                ax.legend()
                ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Đã lưu biểu đồ robustness metrics tại: {save_path}")
        else:
            plt.show()

        plt.close()
        return True

    except Exception as e:
        logger.error(f"Lỗi khi vẽ biểu đồ robustness metrics: {e}")
        return False


def plot_robustness_bands(
    dvh_bands: Dict[str, Dict[str, np.ndarray]],
    structures: Optional[List[str]] = None,
    save_path: Optional[str] = None,
) -> bool:
    """
    Vẽ DVH bands cho phân tích độ bền vững.

    Parameters
    ----------
    dvh_bands : Dict[str, Dict[str, np.ndarray]]
        DVH bands data
    structures : Optional[List[str]]
        Danh sách cấu trúc cần vẽ
    save_path : Optional[str]
        Đường dẫn lưu file

    Returns
    -------
    bool
        True nếu thành công
    """
    if not HAS_MATPLOTLIB:
        logger.warning("Matplotlib không khả dụng. Không thể vẽ biểu đồ.")
        return False

    try:
        if not dvh_bands:
            logger.warning("Không có dữ liệu DVH bands để vẽ.")
            return False

        if structures is None:
            structures = list(dvh_bands.keys())

        fig, ax = plt.subplots(figsize=(12, 8))

        try:
            colors = plt.cm.tab10(np.linspace(0, 1, len(structures)))
        except AttributeError:
            # Fallback khi tab10 không khả dụng
            colors = plt.cm.Set1(np.linspace(0, 1, len(structures)))

        for i, structure in enumerate(structures):
            if structure not in dvh_bands:
                continue

            data = dvh_bands[structure]
            dose_bins = data["dose_bins"]
            nominal = data["nominal"]
            min_band = data["min_band"]
            max_band = data["max_band"]

            color = colors[i]

            # Vẽ nominal DVH
            ax.plot(
                dose_bins,
                nominal,
                color=color,
                linewidth=2,
                label=f"{structure} (Nominal)",
            )

            # Vẽ uncertainty band
            ax.fill_between(
                dose_bins,
                min_band,
                max_band,
                color=color,
                alpha=0.3,
                label=f"{structure} (Uncertainty Band)",
            )

        ax.set_xlabel("Dose (Gy)")
        ax.set_ylabel("Volume (%)")
        ax.set_title("DVH Robustness Bands")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)
        ax.set_ylim(0, 100)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Đã lưu biểu đồ DVH bands tại: {save_path}")
        else:
            plt.show()

        plt.close()
        return True

    except Exception as e:
        logger.error(f"Lỗi khi vẽ DVH bands: {e}")
        return False


def create_robustness_summary_plot(
    robustness_results: Dict[str, Any],
    dvh_bands: Dict[str, Dict[str, np.ndarray]],
    save_path: Optional[str] = None,
) -> bool:
    """
    Tạo biểu đồ tổng hợp cho phân tích độ bền vững.

    Parameters
    ----------
    robustness_results : Dict[str, Any]
        Kết quả phân tích độ bền vững
    dvh_bands : Dict[str, Dict[str, np.ndarray]]
        DVH bands data
    save_path : Optional[str]
        Đường dẫn lưu file

    Returns
    -------
    bool
        True nếu thành công
    """
    if not HAS_MATPLOTLIB:
        logger.warning("Matplotlib không khả dụng. Không thể tạo biểu đồ tổng hợp.")
        return False

    try:
        fig = plt.figure(figsize=(16, 12))

        # DVH bands subplot
        ax1 = plt.subplot(2, 2, (1, 2))
        plot_dvh_bands_subplot(ax1, dvh_bands)

        # Metrics comparison subplot
        ax2 = plt.subplot(2, 2, 3)
        plot_metrics_comparison_subplot(ax2, robustness_results)

        # Variation summary subplot
        ax3 = plt.subplot(2, 2, 4)
        plot_variation_summary_subplot(ax3, robustness_results)

        plt.suptitle("Robustness Analysis Summary", fontsize=16, fontweight="bold")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Đã lưu biểu đồ tổng hợp tại: {save_path}")
        else:
            plt.show()

        plt.close()
        return True

    except Exception as e:
        logger.error(f"Lỗi khi tạo biểu đồ tổng hợp: {e}")
        return False


def plot_dvh_bands_subplot(ax, dvh_bands):
    """Vẽ DVH bands trong subplot."""
    if not dvh_bands:
        ax.text(
            0.5,
            0.5,
            "No DVH data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return

    colors = plt.cm.tab10(np.linspace(0, 1, len(dvh_bands)))

    for i, (structure, data) in enumerate(dvh_bands.items()):
        dose_bins = data["dose_bins"]
        nominal = data["nominal"]
        min_band = data["min_band"]
        max_band = data["max_band"]

        color = colors[i]

        ax.plot(dose_bins, nominal, color=color, linewidth=2, label=structure)
        ax.fill_between(dose_bins, min_band, max_band, color=color, alpha=0.3)

    ax.set_xlabel("Dose (Gy)")
    ax.set_ylabel("Volume (%)")
    ax.set_title("DVH Robustness Bands")
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_metrics_comparison_subplot(ax, robustness_results):
    """Vẽ so sánh metrics trong subplot."""
    if not robustness_results:
        ax.text(
            0.5,
            0.5,
            "No metrics data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return

    structures = list(robustness_results.keys())[:5]  # Limit to 5 structures
    metrics = ["mean_dose", "d95"]

    x = np.arange(len(structures))
    width = 0.35

    for i, metric in enumerate(metrics):
        values = []
        for structure in structures:
            data = robustness_results.get(structure, {})
            nominal = data.get("nominal", {})
            values.append(nominal.get(metric, 0))

        ax.bar(x + i * width, values, width, label=metric)

    ax.set_xlabel("Structures")
    ax.set_ylabel("Dose (Gy)")
    ax.set_title("Dose Metrics Comparison")
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(structures, rotation=45, ha="right")
    ax.legend()


def plot_variation_summary_subplot(ax, robustness_results):
    """Vẽ tổng hợp variation trong subplot."""
    if not robustness_results:
        ax.text(
            0.5,
            0.5,
            "No variation data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return

    structures = []
    variations = []

    for structure, data in robustness_results.items():
        if "mean_dose_variation" in data:
            structures.append(structure)
            variation = data["mean_dose_variation"]
            variations.append(variation.get("std", 0))

    if structures:
        y_pos = np.arange(len(structures))
        ax.barh(y_pos, variations)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(structures)
        ax.set_xlabel("Dose Variation (Gy)")
        ax.set_title("Dose Variation Summary")
        ax.grid(True, alpha=0.3)
