#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module phân tích và đánh giá chất lượng điều trị (Delivery QA).

Module này cung cấp các công cụ toàn diện để thực hiện và đánh giá QA cho việc
điều trị xạ trị, bao gồm phân tích log file, phân tích gamma và so sánh DVH.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any, Union
import tempfile
from datetime import datetime
import json
from pathlib import Path

from quangtps.evaluation.qa.machine_log_analyzer import (
    LogFileAnalyzer,
    LogFileType,
    DeviationType,
    DeviationSeverity,
)
from quangtps.evaluation.qa.deviation_report import create_deviation_report
from quangtps.evaluation.metrics.gamma_analysis import GammaAnalysis
from quangtps.evaluation.dvh.dvh_comparison import (
    compare_dvhs,
    plot_dvh_comparison as dvh_plot_comparison,
)
from quangtps.utils.file_utils import ensure_directory_exists

logger = logging.getLogger(__name__)


# Thêm hàm helper để lấy colormap an toàn (tương tự như trong dvh_comparison.py)
def _get_safe_colormap(cmap_name="tab10", fallback_cmap="jet", num_colors=10):
    """
    Lấy colormap an toàn, với phương án dự phòng nếu không có colormap yêu cầu.
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


class DeliveryQA:
    """
    Lớp quản lý và thực hiện các phân tích QA điều trị.

    Cung cấp một giao diện thống nhất cho nhiều loại phân tích QA khác nhau:
    - Phân tích log file máy điều trị
    - Phân tích gamma của phân bố liều
    - So sánh DVH giữa kế hoạch và thực tế
    - Tạo báo cáo QA tổng hợp
    """

    def __init__(self, plan_data: Optional[Dict[str, Any]] = None):
        """
        Khởi tạo đối tượng DeliveryQA.

        Parameters
        ----------
        plan_data : Optional[Dict[str, Any]], optional
            Dữ liệu kế hoạch điều trị, mặc định là None
        """
        self.plan_data = plan_data
        self.log_analysis_results = {}
        self.gamma_analysis_results = {}
        self.dvh_comparison_results = {}
        self.qa_summary = {}
        self.qa_results_dir = None
        self.tolerance_levels = None
        self.report_path = None

    def set_plan_data(self, plan_data: Dict[str, Any]) -> None:
        """
        Thiết lập dữ liệu kế hoạch điều trị.

        Parameters
        ----------
        plan_data : Dict[str, Any]
            Dữ liệu kế hoạch điều trị
        """
        self.plan_data = plan_data

    def set_tolerance_levels(
        self, tolerance_levels: Dict[str, Dict[str, float]]
    ) -> None:
        """
        Thiết lập ngưỡng dung sai cho các phân tích QA.

        Parameters
        ----------
        tolerance_levels : Dict[str, Dict[str, float]]
            Ngưỡng dung sai theo cấu trúc:
            {
                "mlc_position": {"minor": 0.2, "moderate": 0.5, "major": 1.0, "critical": 3.0},
                "gamma_pass_rate": {"acceptable": 95, "minor": 90, "moderate": 85, "major": 80, "critical": 70},
                ...
            }
        """
        self.tolerance_levels = tolerance_levels

    def analyze_machine_logs(self, log_file_path: str) -> Dict[str, Any]:
        """
        Phân tích log file máy điều trị.

        Parameters
        ----------
        log_file_path : str
            Đường dẫn đến file log máy điều trị

        Returns
        -------
        Dict[str, Any]
            Kết quả phân tích log file
        """
        if not os.path.exists(log_file_path):
            logger.error(f"Không tìm thấy file log: {log_file_path}")
            return {}

        try:
            # Phân tích log file
            results = LogFileAnalyzer.analyze_log_file(
                log_file_path,
                plan_data=self.plan_data,
                tolerance_levels=self.tolerance_levels,
            )

            # Lưu kết quả
            self.log_analysis_results[log_file_path] = results

            # Cập nhật tóm tắt QA
            if "qa_summary" not in self.qa_summary:
                self.qa_summary["qa_summary"] = {}

            self.qa_summary["qa_summary"]["machine_log"] = {
                "pass_rate": results.get("pass_rate", 0),
                "max_deviation": results.get("max_deviation", {}).get("value", 0),
                "critical_deviations": sum(
                    1
                    for d in results.get("deviations", [])
                    if d.get("severity") == "critical"
                ),
                "major_deviations": sum(
                    1
                    for d in results.get("deviations", [])
                    if d.get("severity") == "major"
                ),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

            return results

        except Exception as e:
            logger.error(f"Lỗi khi phân tích log file: {str(e)}")
            import traceback

            traceback.print_exc()
            return {}

    def perform_gamma_analysis(
        self,
        reference_dose: np.ndarray,
        evaluation_dose: np.ndarray,
        dose_percent_threshold: float = 3.0,
        distance_mm_threshold: float = 3.0,
        dose_threshold_percent: float = 10.0,
        registration_method: str = "rigid",
    ) -> Dict[str, Any]:
        """
        Thực hiện phân tích gamma giữa hai phân bố liều.

        Parameters
        ----------
        reference_dose : np.ndarray
            Phân bố liều tham chiếu (từ kế hoạch)
        evaluation_dose : np.ndarray
            Phân bố liều cần đánh giá (đo được)
        dose_percent_threshold : float, optional
            Ngưỡng phần trăm liều cho tiêu chí gamma, mặc định là 3.0
        distance_mm_threshold : float, optional
            Ngưỡng khoảng cách (mm) cho tiêu chí gamma, mặc định là 3.0
        dose_threshold_percent : float, optional
            Ngưỡng liều tối thiểu để xem xét trong phân tích, mặc định là 10.0
        registration_method : str, optional
            Phương pháp đăng ký hình ảnh, mặc định là "rigid"

        Returns
        -------
        Dict[str, Any]
            Kết quả phân tích gamma
        """
        try:
            # Tạo đối tượng phân tích gamma
            gamma_analyzer = GammaAnalysis(
                reference_dose=reference_dose, evaluation_dose=evaluation_dose
            )

            # Thực hiện phân tích
            gamma_result = gamma_analyzer.calculate(
                dose_percent=dose_percent_threshold,
                distance_mm=distance_mm_threshold,
                threshold=dose_threshold_percent,
                registration=registration_method,
            )

            # Lưu kết quả
            result_id = f"gamma_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            self.gamma_analysis_results[result_id] = gamma_result

            # Cập nhật tóm tắt QA
            if "qa_summary" not in self.qa_summary:
                self.qa_summary["qa_summary"] = {}

            # Xác định mức độ nghiêm trọng dựa trên tỉ lệ pass
            pass_rate = gamma_result.get("pass_rate", 0)

            severity = "acceptable"
            if self.tolerance_levels and "gamma_pass_rate" in self.tolerance_levels:
                levels = self.tolerance_levels["gamma_pass_rate"]
                if pass_rate < levels.get("critical", 70):
                    severity = "critical"
                elif pass_rate < levels.get("major", 80):
                    severity = "major"
                elif pass_rate < levels.get("moderate", 85):
                    severity = "moderate"
                elif pass_rate < levels.get("minor", 90):
                    severity = "minor"
                elif pass_rate >= levels.get("acceptable", 95):
                    severity = "acceptable"

            self.qa_summary["qa_summary"]["gamma_analysis"] = {
                "pass_rate": pass_rate,
                "criteria": f"{dose_percent_threshold}%/{distance_mm_threshold}mm",
                "severity": severity,
                "max_gamma": gamma_result.get("max_gamma", 0),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

            return gamma_result

        except Exception as e:
            logger.error(f"Lỗi khi thực hiện phân tích gamma: {str(e)}")
            import traceback

            traceback.print_exc()
            return {}

    def compare_plan_vs_measured_dvhs(
        self, plan_dvhs: Dict[str, Any], measured_dvhs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        So sánh DVH từ kế hoạch với DVH đo được.

        Parameters
        ----------
        plan_dvhs : Dict[str, Any]
            DVH từ kế hoạch, cấu trúc: {structure_id: {dose: [...], volume: [...]}}
        measured_dvhs : Dict[str, Any]
            DVH đo được, cấu trúc tương tự plan_dvhs

        Returns
        -------
        Dict[str, Any]
            Kết quả so sánh DVH
        """
        try:
            # Thực hiện so sánh DVH
            comparison_results = compare_dvhs(plan_dvhs, measured_dvhs)

            # Lưu kết quả
            result_id = f"dvh_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            self.dvh_comparison_results[result_id] = comparison_results

            # Cập nhật tóm tắt QA
            if "qa_summary" not in self.qa_summary:
                self.qa_summary["qa_summary"] = {}

            self.qa_summary["qa_summary"]["dvh_comparison"] = {
                "similarity_index": comparison_results.get("overall_similarity", 0),
                "structures_analyzed": len(comparison_results.get("structures", {})),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

            return comparison_results

        except Exception as e:
            logger.error(f"Lỗi khi so sánh DVH: {str(e)}")
            import traceback

            traceback.print_exc()
            return {}

    def generate_qa_report(self, output_dir: Optional[str] = None) -> Optional[str]:
        """
        Tạo báo cáo QA tổng hợp từ các phân tích đã thực hiện.

        Parameters
        ----------
        output_dir : Optional[str], optional
            Thư mục đầu ra cho báo cáo, mặc định là None (sử dụng thư mục tạm)

        Returns
        -------
        Optional[str]
            Đường dẫn đến file báo cáo, hoặc None nếu có lỗi
        """
        if not self.qa_summary:
            logger.warning("Không có dữ liệu QA để tạo báo cáo")
            return None

        try:
            # Nếu không chỉ định thư mục đầu ra, sử dụng thư mục tạm
            if output_dir is None:
                output_dir = tempfile.mkdtemp(prefix="quangtps_qa_")
                self.qa_results_dir = output_dir
            else:
                os.makedirs(output_dir, exist_ok=True)
                self.qa_results_dir = output_dir

            # Tạo dữ liệu cho báo cáo
            report_data = {
                "title": "Báo cáo QA Điều trị",
                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "plan_name": self.plan_data.get("plan_name", "Unknown")
                if self.plan_data
                else "Unknown",
                "patient_id": self.plan_data.get("patient_id", "Unknown")
                if self.plan_data
                else "Unknown",
                "summary": self.qa_summary,
                "log_analysis": self.log_analysis_results,
                "gamma_analysis": self.gamma_analysis_results,
                "dvh_comparison": self.dvh_comparison_results,
            }

            # Tạo báo cáo HTML sử dụng API mới
            report_path = create_deviation_report(
                data=report_data,
                output_dir=output_dir,
                template_name="delivery_qa_report.html",
            )

            self.report_path = report_path
            return report_path

        except Exception as e:
            logger.error(f"Lỗi khi tạo báo cáo QA: {str(e)}")
            import traceback

            traceback.print_exc()
            return None

    def plot_all_deviations(self, output_dir: Optional[str] = None) -> List[str]:
        """
        Tạo biểu đồ cho tất cả các sai lệch từ phân tích log file.

        Parameters
        ----------
        output_dir : Optional[str], optional
            Thư mục đầu ra cho các biểu đồ, mặc định là None (sử dụng thư mục tạm)

        Returns
        -------
        List[str]
            Danh sách đường dẫn đến các file biểu đồ
        """
        if not self.log_analysis_results:
            logger.warning("Không có dữ liệu phân tích log để tạo biểu đồ")
            return []

        try:
            # Xác định thư mục đầu ra
            if output_dir is None:
                if self.qa_results_dir:
                    output_dir = os.path.join(self.qa_results_dir, "plots")
                else:
                    output_dir = tempfile.mkdtemp(prefix="quangtps_qa_plots_")

            os.makedirs(output_dir, exist_ok=True)

            # Danh sách đường dẫn biểu đồ
            plot_paths = []

            # Duyệt qua từng kết quả phân tích log
            for log_file, results in self.log_analysis_results.items():
                analyzer = results.get("analyzer")

                if analyzer:
                    # Tạo biểu đồ cho các tham số khác nhau
                    for param_type in DeviationType:
                        fig, ax = plt.subplots(figsize=(10, 6))
                        analyzer.plot_deviations(param_type.value, ax=ax)

                        # Lưu biểu đồ
                        plot_name = (
                            f"{os.path.basename(log_file)}_{param_type.value}.png"
                        )
                        plot_path = os.path.join(output_dir, plot_name)
                        fig.savefig(plot_path, dpi=300, bbox_inches="tight")
                        plt.close(fig)

                        plot_paths.append(plot_path)

            return plot_paths

        except Exception as e:
            logger.error(f"Lỗi khi tạo biểu đồ sai lệch: {str(e)}")
            import traceback

            traceback.print_exc()
            return []

    def plot_gamma_results(self, output_dir: Optional[str] = None) -> List[str]:
        """
        Tạo biểu đồ cho kết quả phân tích gamma.

        Parameters
        ----------
        output_dir : Optional[str], optional
            Thư mục đầu ra cho các biểu đồ, mặc định là None (sử dụng thư mục tạm)

        Returns
        -------
        List[str]
            Danh sách đường dẫn đến các file biểu đồ
        """
        if not self.gamma_analysis_results:
            logger.warning("Không có dữ liệu phân tích gamma để tạo biểu đồ")
            return []

        try:
            # Xác định thư mục đầu ra
            if output_dir is None:
                if self.qa_results_dir:
                    output_dir = os.path.join(self.qa_results_dir, "gamma_plots")
                else:
                    output_dir = tempfile.mkdtemp(prefix="quangtps_gamma_plots_")

            os.makedirs(output_dir, exist_ok=True)

            # Danh sách đường dẫn biểu đồ
            plot_paths = []

            # Duyệt qua từng kết quả phân tích gamma
            for result_id, results in self.gamma_analysis_results.items():
                if "gamma_map" in results and isinstance(
                    results["gamma_map"], np.ndarray
                ):
                    # Tạo biểu đồ phân bố gamma
                    gamma_map = results["gamma_map"]

                    # Chọn slice giữa nếu là 3D
                    if gamma_map.ndim == 3:
                        mid_slice = gamma_map.shape[0] // 2
                        gamma_slice = gamma_map[mid_slice]
                    else:
                        gamma_slice = gamma_map

                    # Tạo biểu đồ
                    fig, ax = plt.subplots(figsize=(10, 8))

                    # Hiển thị phân bố gamma
                    im = ax.imshow(gamma_slice, cmap="jet", interpolation="nearest")
                    ax.set_title(
                        f"Phân bố Gamma (Pass rate: {results.get('pass_rate', 0):.1f}%)"
                    )

                    # Thêm colorbar
                    cbar = fig.colorbar(im, ax=ax)
                    cbar.set_label("Gamma value")

                    # Lưu biểu đồ
                    plot_name = f"{result_id}_gamma_map.png"
                    plot_path = os.path.join(output_dir, plot_name)
                    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
                    plt.close(fig)

                    plot_paths.append(plot_path)

                # Tạo biểu đồ histogram gamma
                if "gamma_values" in results and isinstance(
                    results["gamma_values"], list
                ):
                    gamma_values = np.array(results["gamma_values"])
                    gamma_values = gamma_values[~np.isnan(gamma_values)]

                    if len(gamma_values) > 0:
                        fig, ax = plt.subplots(figsize=(10, 6))

                        # Tạo histogram
                        ax.hist(gamma_values, bins=50, alpha=0.7, color="blue")
                        ax.axvline(
                            x=1.0,
                            color="red",
                            linestyle="--",
                            label="Pass/Fail Threshold (Gamma=1.0)",
                        )

                        ax.set_title(
                            f"Histogram Gamma (Pass rate: {results.get('pass_rate', 0):.1f}%)"
                        )
                        ax.set_xlabel("Gamma Value")
                        ax.set_ylabel("Frequency")
                        ax.legend()

                        # Lưu biểu đồ
                        plot_name = f"{result_id}_gamma_histogram.png"
                        plot_path = os.path.join(output_dir, plot_name)
                        fig.savefig(plot_path, dpi=300, bbox_inches="tight")
                        plt.close(fig)

                        plot_paths.append(plot_path)

            return plot_paths

        except Exception as e:
            logger.error(f"Lỗi khi tạo biểu đồ phân tích gamma: {str(e)}")
            import traceback

            traceback.print_exc()
            return []

    def plot_dvh_comparison(self, output_dir: Optional[str] = None) -> List[str]:
        """
        Tạo biểu đồ so sánh DVH giữa kế hoạch và đo đạc.

        Parameters:
            output_dir: Thư mục lưu biểu đồ, nếu None sẽ sử dụng thư mục tạm

        Returns:
            Danh sách đường dẫn tới các biểu đồ đã tạo
        """
        if not self.dvh_comparison_results:
            logger.warning("Chưa có dữ liệu so sánh DVH")
            return []

        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(__file__), "temp")
        ensure_directory_exists(output_dir)

        plan_dvhs = self.dvh_comparison_results.get("plan_dvhs", {})
        measured_dvhs = self.dvh_comparison_results.get("measured_dvhs", {})
        comparison_results = self.dvh_comparison_results.get("comparison_results", {})

        if not plan_dvhs or not measured_dvhs:
            logger.warning("Thiếu dữ liệu DVH kế hoạch hoặc đo đạc")
            return []

        output_files = []

        # Tạo biểu đồ tổng quan cho tất cả các cấu trúc
        try:
            fig, ax = plt.subplots(figsize=(12, 8))

            # Vẽ DVH cho từng cấu trúc
            common_structures = set(plan_dvhs.keys()) & set(measured_dvhs.keys())

            if common_structures:
                # Sử dụng colormap an toàn thay vì plt.cm.tab10
                colors = _get_safe_colormap(
                    cmap_name="tab10", num_colors=len(common_structures)
                )

                for i, structure_id in enumerate(common_structures):
                    if (
                        "dose" in plan_dvhs[structure_id]
                        and "volume" in plan_dvhs[structure_id]
                        and "dose" in measured_dvhs[structure_id]
                        and "volume" in measured_dvhs[structure_id]
                    ):
                        # Sử dụng màu khác nhau cho mỗi cấu trúc
                        color = colors[i % len(colors)]

                        # Vẽ DVH kế hoạch
                        ax.plot(
                            plan_dvhs[structure_id]["dose"],
                            plan_dvhs[structure_id]["volume"],
                            linestyle="-",
                            color=color,
                            linewidth=2,
                            label=f"{structure_id} (Kế hoạch)",
                        )

                        # Vẽ DVH đo được
                        ax.plot(
                            measured_dvhs[structure_id]["dose"],
                            measured_dvhs[structure_id]["volume"],
                            linestyle="--",
                            color=color,
                            linewidth=2,
                            label=f"{structure_id} (Đo đạc)",
                        )

            ax.set_xlabel("Liều lượng (Gy)")
            ax.set_ylabel("Thể tích (%)")
            ax.set_title("So sánh DVH: Kế hoạch vs Đo đạc")
            ax.grid(True, linestyle="--", alpha=0.7)
            ax.set_xlim(left=0)
            ax.set_ylim(0, 105)
            ax.legend(loc="best", bbox_to_anchor=(1, 1))

            # Lưu biểu đồ
            plot_name = f"dvh_comparison.png"
            plot_path = os.path.join(output_dir, plot_name)
            fig.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close(fig)

            output_files.append(plot_path)

            return output_files

        except Exception as e:
            logger.error(f"Lỗi khi tạo biểu đồ so sánh DVH: {str(e)}")
            import traceback

            traceback.print_exc()
            return []
