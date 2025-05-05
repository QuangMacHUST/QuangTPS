#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module báo cáo DVH (Dose-Volume Histogram) cho QuangTPS.

Module này cung cấp các lớp và hàm để tạo báo cáo liều-thể tích chi tiết,
bao gồm các chỉ số và thông số phân tích quan trọng cho đánh giá kế hoạch điều trị.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Any, Optional, Union
import io
import base64
from pathlib import Path
from jinja2 import Environment, FileSystemLoader, select_autoescape

# Import các module QuangTPS cần thiết
try:
    from quangtps.core.patient import Patient
    from quangtps.core.plan import Plan
    from quangtps.evaluation.dvh.dvh_calculator import DVHCalculator
    from quangtps.structures.structure import Structure
    from quangtps.ui.utils.plot_utils import setup_matplotlib_style
    from quangtps.utils.io_utils import create_directory_if_not_exists

    QUANGTPS_AVAILABLE = True
except ImportError:
    QUANGTPS_AVAILABLE = False
    logging.warning(
        "QuangTPS core modules không khả dụng. Chỉ có thể sử dụng các chức năng cơ bản."
    )

# Khởi tạo logger
logger = logging.getLogger(__name__)

# Định nghĩa các hằng số cho báo cáo
DEFAULT_REPORT_TEMPLATE = "dvh_report.html"
TEMPLATE_DIR = Path(__file__).parent / "templates"


# Hàm tạo thư mục nếu không có module io_utils
def _create_directory_if_not_exists(directory_path):
    """Tạo thư mục nếu chưa tồn tại."""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


class DVHReportGenerator:
    """
    Lớp tạo báo cáo DVH chi tiết, bao gồm các thông số lâm sàng và đồ thị.

    Lớp này cung cấp phương thức để tính toán, phân tích và xuất báo cáo DVH
    dưới nhiều định dạng: HTML, PDF, và CSV.
    """

    def __init__(self, output_dir: str = None):
        """
        Khởi tạo DVHReportGenerator.

        Parameters:
            output_dir (str, optional): Thư mục lưu báo cáo đầu ra. Mặc định là thư mục hiện tại.
        """
        self.output_dir = output_dir or os.getcwd()

        # Đảm bảo thư mục đầu ra tồn tại
        if QUANGTPS_AVAILABLE and "create_directory_if_not_exists" in globals():
            create_directory_if_not_exists(self.output_dir)
        else:
            _create_directory_if_not_exists(self.output_dir)

        # Khởi tạo môi trường template Jinja2
        self.jinja_env = Environment(
            loader=FileSystemLoader(TEMPLATE_DIR),
            autoescape=select_autoescape(["html", "xml"]),
        )

        # Thêm các bộ lọc tùy chỉnh
        self.jinja_env.filters["format_date"] = self._format_date
        self.jinja_env.filters["format_number"] = self._format_number

        # Các màu cố định cho các cấu trúc phổ biến
        self.structure_colors = {
            "PTV": "#FF0000",
            "CTV": "#FF8C00",
            "GTV": "#FF4500",
            "HEART": "#FF69B4",
            "LUNG_LEFT": "#87CEFA",
            "LUNG_RIGHT": "#00BFFF",
            "SPINAL_CORD": "#FFFF00",
            "ESOPHAGUS": "#9932CC",
            "LIVER": "#8B4513",
            "KIDNEY_LEFT": "#006400",
            "KIDNEY_RIGHT": "#228B22",
            "BLADDER": "#FFD700",
            "RECTUM": "#A52A2A",
            "BRAIN": "#F5DEB3",
            "BRAIN_STEM": "#DEB887",
            "LENS_LEFT": "#C0C0C0",
            "LENS_RIGHT": "#D3D3D3",
            "OPTIC_CHIASM": "#808080",
        }

        # Khởi tạo DVHCalculator
        if QUANGTPS_AVAILABLE:
            self.dvh_calculator = DVHCalculator()

    def _format_date(self, value):
        """Định dạng ngày tháng."""
        if isinstance(value, datetime):
            return value.strftime("%d/%m/%Y %H:%M")
        return value

    def _format_number(self, value):
        """Định dạng số."""
        if isinstance(value, (int, float)):
            if value == int(value):
                return f"{int(value)}"
            else:
                return f"{value:.2f}"
        return value

    def calculate_dvh_metrics(
        self, dvh_data: Dict[str, Dict[str, Any]], prescription_dose: float = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Tính toán các chỉ số DVH quan trọng cho từng cấu trúc.

        Parameters:
            dvh_data (dict): Dữ liệu DVH (từ điển cấu trúc -> dữ liệu DVH)
            prescription_dose (float, optional): Liều điều trị (Gy), nếu có

        Returns:
            dict: Từ điển chứa các chỉ số DVH cho mỗi cấu trúc
        """
        metrics = {}

        for structure_name, data in dvh_data.items():
            # Bỏ qua grid liều
            if structure_name == "dose_grid" or not data:
                continue

            # Khởi tạo từ điển chỉ số cho cấu trúc
            metrics[structure_name] = {}

            # Trích xuất dữ liệu DVH
            if "differential_dvh" in data:
                ddvh = data["differential_dvh"]
                doses = ddvh["doses"]
                volumes = ddvh["volumes"]
            elif "cumulative_dvh" in data:
                cdvh = data["cumulative_dvh"]
                doses = cdvh["doses"]
                volumes = cdvh["volumes"]
            else:
                # Không có dữ liệu DVH
                continue

            # Tính Dmin, Dmean, Dmax
            if "dose_values" in data:
                dose_values = data["dose_values"]
                metrics[structure_name]["Dmin"] = np.min(dose_values)
                metrics[structure_name]["Dmean"] = np.mean(dose_values)
                metrics[structure_name]["Dmax"] = np.max(dose_values)
            else:
                # Ước tính từ histogram
                total_volume = np.sum(volumes)
                metrics[structure_name]["Dmin"] = np.min(doses)
                metrics[structure_name]["Dmax"] = np.max(doses)
                metrics[structure_name]["Dmean"] = (
                    np.sum(doses * volumes) / total_volume
                )

            # Tính Dx (liều nhận bởi x% thể tích)
            # Giả sử dữ liệu DVH tích lũy đã được sắp xếp theo thứ tự giảm dần của thể tích
            for percent in [5, 10, 50, 90, 95, 98, 99]:
                target_volume = percent / 100.0
                metrics[structure_name][f"D{percent}"] = self._calculate_dose_at_volume(
                    doses, volumes, target_volume
                )

            # Tính Vx (thể tích nhận ít nhất x Gy)
            for dose_level in [5, 10, 20, 30, 40, 50]:
                metrics[structure_name][f"V{dose_level}Gy"] = (
                    self._calculate_volume_at_dose(doses, volumes, dose_level)
                )

            # Nếu có liều điều trị, tính thể tích nhận phần trăm liều đó
            if prescription_dose:
                for percent in [50, 80, 90, 95, 100, 105, 110]:
                    dose_level = prescription_dose * percent / 100.0
                    metrics[structure_name][f"V{percent}%"] = (
                        self._calculate_volume_at_dose(doses, volumes, dose_level)
                    )

            # Thêm chỉ số độ đồng nhất (Homogeneity Index - HI) nếu là PTV
            if "PTV" in structure_name.upper() and prescription_dose:
                d2 = metrics[structure_name]["D2"]
                d98 = metrics[structure_name]["D98"]
                metrics[structure_name]["HI"] = (d2 - d98) / prescription_dose

            # Thêm chỉ số phù hợp (Conformity Index - CI) nếu là PTV và có liều điều trị
            if "PTV" in structure_name.upper() and prescription_dose:
                v95 = metrics[structure_name].get(f"V95%", 0)
                metrics[structure_name]["CI"] = v95 / 100.0  # CI = V95% / VPTV

        return metrics

    def _calculate_dose_at_volume(
        self, doses: np.ndarray, volumes: np.ndarray, target_volume_percent: float
    ) -> float:
        """
        Tính liều tại một phần trăm thể tích cụ thể.

        Parameters:
            doses (np.ndarray): Mảng liều
            volumes (np.ndarray): Mảng thể tích tương ứng (%)
            target_volume_percent (float): Phần trăm thể tích mục tiêu (0-1)

        Returns:
            float: Liều tại phần trăm thể tích đó (Gy)
        """
        try:
            # Chuẩn hóa volumes thành phần trăm
            if np.max(volumes) > 1.1:  # Đã là phần trăm
                normalized_volumes = volumes / 100.0
            else:  # Đã là tỉ lệ (0-1)
                normalized_volumes = volumes

            # Tìm chỉ số gần nhất
            idx = np.argmin(np.abs(normalized_volumes - target_volume_percent))

            # Nội suy tuyến tính nếu có thể
            if idx > 0 and idx < len(normalized_volumes) - 1:
                vol1, vol2 = normalized_volumes[idx - 1], normalized_volumes[idx]
                dose1, dose2 = doses[idx - 1], doses[idx]

                if vol1 != vol2:  # Tránh chia cho 0
                    return dose1 + (dose2 - dose1) * (target_volume_percent - vol1) / (
                        vol2 - vol1
                    )

            return doses[idx]
        except Exception as e:
            logger.error(f"Lỗi khi tính dose-at-volume: {e}")
            return 0.0

    def _calculate_volume_at_dose(
        self, doses: np.ndarray, volumes: np.ndarray, target_dose: float
    ) -> float:
        """
        Tính thể tích nhận ít nhất một liều cụ thể.

        Parameters:
            doses (np.ndarray): Mảng liều
            volumes (np.ndarray): Mảng thể tích tương ứng (%)
            target_dose (float): Liều mục tiêu (Gy)

        Returns:
            float: Phần trăm thể tích nhận ít nhất liều đó (%)
        """
        try:
            # Tìm chỉ số gần nhất
            idx = np.argmin(np.abs(doses - target_dose))

            # Nội suy tuyến tính nếu có thể
            if idx > 0 and idx < len(doses) - 1:
                dose1, dose2 = doses[idx - 1], doses[idx]
                vol1, vol2 = volumes[idx - 1], volumes[idx]

                if dose1 != dose2:  # Tránh chia cho 0
                    volume_at_dose = vol1 + (vol2 - vol1) * (target_dose - dose1) / (
                        dose2 - dose1
                    )

                    # Chuyển đổi về phần trăm nếu cần
                    if np.max(volumes) <= 1.1:  # Là tỉ lệ (0-1)
                        volume_at_dose *= 100.0

                    return volume_at_dose

            # Trả về giá trị không nội suy
            volume_at_dose = volumes[idx]

            # Chuyển đổi về phần trăm nếu cần
            if np.max(volumes) <= 1.1:  # Là tỉ lệ (0-1)
                volume_at_dose *= 100.0

            return volume_at_dose
        except Exception as e:
            logger.error(f"Lỗi khi tính volume-at-dose: {e}")
            return 0.0

    def create_dvh_plot(
        self,
        dvh_data: Dict[str, Dict[str, Any]],
        plan_name: str = None,
        include_differential: bool = False,
        figsize: Tuple[int, int] = (10, 6),
    ) -> plt.Figure:
        """
        Tạo đồ thị DVH từ dữ liệu.

        Parameters:
            dvh_data (dict): Dữ liệu DVH (từ điển cấu trúc -> dữ liệu DVH)
            plan_name (str, optional): Tên kế hoạch điều trị
            include_differential (bool): Có bao gồm đồ thị DVH vi phân hay không
            figsize (tuple): Kích thước đồ thị

        Returns:
            matplotlib.figure.Figure: Đối tượng Figure chứa đồ thị DVH
        """
        # Thiết lập style cho matplotlib
        if "setup_matplotlib_style" in globals():
            setup_matplotlib_style()

        # Tạo đồ thị
        if include_differential:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=figsize)

        # Tiêu đề
        if plan_name:
            fig.suptitle(f"Biểu đồ Liều-Thể tích (DVH) - {plan_name}", fontsize=14)
        else:
            fig.suptitle("Biểu đồ Liều-Thể tích (DVH)", fontsize=14)

        # Vẽ DVH tích lũy
        ax1.set_title("DVH Tích lũy", fontsize=12)
        ax1.set_xlabel("Liều (Gy)", fontsize=10)
        ax1.set_ylabel("Thể tích (%)", fontsize=10)
        ax1.grid(True, linestyle="--", alpha=0.7)

        # Danh sách màu mặc định
        default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        # Vẽ từng đường DVH
        structure_idx = 0
        legend_entries = []

        for structure_name, data in dvh_data.items():
            # Bỏ qua grid liều
            if structure_name == "dose_grid" or not data:
                continue

            # Kiểm tra dữ liệu DVH tích lũy
            if "cumulative_dvh" in data:
                cdvh = data["cumulative_dvh"]
                doses = cdvh["doses"]
                volumes = cdvh["volumes"]

                # Chuyển đổi volumes thành phần trăm nếu cần
                if np.max(volumes) <= 1.1:  # Là tỉ lệ (0-1)
                    volumes = volumes * 100.0

                # Chọn màu cho cấu trúc
                color = self.structure_colors.get(
                    structure_name.upper(),
                    default_colors[structure_idx % len(default_colors)],
                )

                # Vẽ đường DVH tích lũy
                (line,) = ax1.plot(doses, volumes, label=structure_name, color=color)
                legend_entries.append(line)

                # Tăng chỉ số cấu trúc
                structure_idx += 1

        # Thêm hộp chú thích
        ax1.legend(handles=legend_entries, title="Cấu trúc", loc="best")

        # Giới hạn trục y từ 0 đến 100%
        ax1.set_ylim([0, 105])

        # Vẽ DVH vi phân nếu được yêu cầu
        if include_differential:
            ax2.set_title("DVH Vi phân", fontsize=12)
            ax2.set_xlabel("Liều (Gy)", fontsize=10)
            ax2.set_ylabel("Thể tích (%/Gy)", fontsize=10)
            ax2.grid(True, linestyle="--", alpha=0.7)

            structure_idx = 0
            legend_entries = []

            for structure_name, data in dvh_data.items():
                # Bỏ qua grid liều
                if structure_name == "dose_grid" or not data:
                    continue

                # Kiểm tra dữ liệu DVH vi phân
                if "differential_dvh" in data:
                    ddvh = data["differential_dvh"]
                    doses = ddvh["doses"]
                    volumes = ddvh["volumes"]

                    # Chuyển đổi volumes thành phần trăm nếu cần
                    if np.max(volumes) <= 1.1:  # Là tỉ lệ (0-1)
                        volumes = volumes * 100.0

                    # Chọn màu cho cấu trúc
                    color = self.structure_colors.get(
                        structure_name.upper(),
                        default_colors[structure_idx % len(default_colors)],
                    )

                    # Vẽ đường DVH vi phân
                    (line,) = ax2.plot(
                        doses, volumes, label=structure_name, color=color
                    )
                    legend_entries.append(line)

                    # Tăng chỉ số cấu trúc
                    structure_idx += 1

            # Thêm hộp chú thích
            ax2.legend(handles=legend_entries, title="Cấu trúc", loc="best")

        # Điều chỉnh khoảng cách giữa các subplots
        fig.tight_layout(rect=[0, 0, 1, 0.95])

        return fig

    def generate_html_report(
        self,
        patient: Any,
        plan: Any,
        dvh_data: Dict[str, Dict[str, Any]],
        output_file: str = None,
    ) -> str:
        """
        Tạo báo cáo DVH dưới dạng HTML.

        Parameters:
            patient (Any): Đối tượng bệnh nhân
            plan (Any): Đối tượng kế hoạch điều trị
            dvh_data (dict): Dữ liệu DVH
            output_file (str, optional): Đường dẫn file đầu ra

        Returns:
            str: Đường dẫn đến file HTML được tạo
        """
        try:
            # Tạo đồ thị DVH
            dvh_fig = self.create_dvh_plot(dvh_data, getattr(plan, "name", None))

            # Chuyển đồ thị thành chuỗi base64 để nhúng vào HTML
            buf = io.BytesIO()
            dvh_fig.savefig(buf, format="png", dpi=100)
            buf.seek(0)
            dvh_image = base64.b64encode(buf.read()).decode("utf-8")
            plt.close(dvh_fig)

            # Tạo đồ thị DVH vi phân
            dvh_diff_fig = self.create_dvh_plot(
                dvh_data, getattr(plan, "name", None), include_differential=True
            )
            buf = io.BytesIO()
            dvh_diff_fig.savefig(buf, format="png", dpi=100)
            buf.seek(0)
            dvh_diff_image = base64.b64encode(buf.read()).decode("utf-8")
            plt.close(dvh_diff_fig)

            # Lấy liều điều trị từ kế hoạch
            prescription_dose = None
            num_fractions = None
            if hasattr(plan, "prescription") and plan.prescription:
                prescription_dose = getattr(plan.prescription, "total_dose", None)
                num_fractions = getattr(plan.prescription, "num_fractions", None)

            # Tính toán các chỉ số DVH
            dvh_metrics = self.calculate_dvh_metrics(dvh_data, prescription_dose)

            # Chuẩn bị dữ liệu cho template
            template_data = {
                "report_title": f"Báo cáo DVH - {getattr(plan, 'name', 'Không tên')}",
                "generated_date": datetime.now(),
                "report_id": f"DVH_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                "patient": {
                    "id": getattr(patient, "id", "N/A"),
                    "name": getattr(patient, "name", "N/A"),
                },
                "plan": {
                    "id": getattr(plan, "id", "N/A"),
                    "name": getattr(plan, "name", "N/A"),
                    "creation_date": getattr(plan, "creation_date", datetime.now()),
                    "prescription": {
                        "total_dose": prescription_dose,
                        "num_fractions": num_fractions,
                        "dose_per_fraction": prescription_dose / num_fractions
                        if prescription_dose and num_fractions
                        else None,
                        "target_structures": getattr(
                            plan.prescription, "target_structures", []
                        )
                        if hasattr(plan, "prescription")
                        else [],
                    }
                    if hasattr(plan, "prescription")
                    else None,
                },
                "dvh_image": f"data:image/png;base64,{dvh_image}",
                "dvh_cumulative_image": f"data:image/png;base64,{dvh_image}",
                "dvh_differential_image": f"data:image/png;base64,{dvh_diff_image}",
                "structures": [],
                "structure_colors": self.structure_colors,
                "dvh_metrics": dvh_metrics,
            }

            # Thêm thông tin cấu trúc
            for structure_name, data in dvh_data.items():
                if structure_name == "dose_grid" or not data:
                    continue

                volume = 0
                if "volume" in data:
                    volume = data["volume"]
                elif "volumes" in data:
                    volume = np.sum(data["volumes"])

                structure_info = {
                    "name": structure_name,
                    "type": "Target"
                    if "PTV" in structure_name.upper()
                    or "CTV" in structure_name.upper()
                    or "GTV" in structure_name.upper()
                    else "OAR",
                    "volume": volume,
                    "description": "",
                }
                template_data["structures"].append(structure_info)

            # Tải template và render
            template = self.jinja_env.get_template(DEFAULT_REPORT_TEMPLATE)
            html_content = template.render(**template_data)

            # Lưu file nếu được chỉ định
            if output_file:
                output_path = output_file
            else:
                output_path = os.path.join(
                    self.output_dir,
                    f"DVH_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                )

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(html_content)

            logger.info(f"Đã tạo báo cáo HTML: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Lỗi khi tạo báo cáo HTML: {e}")
            raise

    def export_metrics_to_csv(
        self, dvh_metrics: Dict[str, Dict[str, float]], output_file: str = None
    ) -> str:
        """
        Xuất các chỉ số DVH ra file CSV.

        Parameters:
            dvh_metrics (dict): Từ điển chứa các chỉ số DVH cho mỗi cấu trúc
            output_file (str, optional): Đường dẫn file đầu ra

        Returns:
            str: Đường dẫn đến file CSV được tạo
        """
        try:
            # Tạo DataFrame từ các chỉ số
            df = pd.DataFrame()

            # Danh sách tất cả các chỉ số có thể có
            all_metrics = set()
            for structure_metrics in dvh_metrics.values():
                all_metrics.update(structure_metrics.keys())

            # Điền dữ liệu vào DataFrame
            for structure_name, metrics in dvh_metrics.items():
                row_data = {"Structure": structure_name}
                for metric in all_metrics:
                    row_data[metric] = metrics.get(metric, None)

                df = pd.concat([df, pd.DataFrame([row_data])], ignore_index=True)

            # Sắp xếp các cột
            columns = ["Structure"]
            for col in sorted(list(all_metrics)):
                columns.append(col)

            df = df[columns]

            # Lưu file
            if output_file:
                output_path = output_file
            else:
                output_path = os.path.join(
                    self.output_dir,
                    f"DVH_Metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                )

            df.to_csv(output_path, index=False)
            logger.info(f"Đã xuất chỉ số DVH ra file CSV: {output_path}")

            return output_path

        except Exception as e:
            logger.error(f"Lỗi khi xuất chỉ số DVH ra CSV: {e}")
            raise


def create_dvh_report(patient, plan, output_dir=None, format="html"):
    """
    Hàm tiện ích để tạo báo cáo DVH.

    Parameters:
        patient: Đối tượng bệnh nhân
        plan: Đối tượng kế hoạch điều trị
        output_dir (str, optional): Thư mục lưu báo cáo
        format (str): Định dạng báo cáo ('html', 'pdf', 'csv')

    Returns:
        str: Đường dẫn đến file báo cáo được tạo
    """
    if not QUANGTPS_AVAILABLE:
        logger.error("Không thể tạo báo cáo DVH: QuangTPS core modules không khả dụng.")
        return None

    try:
        # Tạo DVHReportGenerator
        report_generator = DVHReportGenerator(output_dir)

        # Tính toán DVH nếu chưa có
        if not hasattr(plan, "dvh_data") or not plan.dvh_data:
            calculator = DVHCalculator()
            # Lấy structure_set từ plan
            structure_set = getattr(plan, "structure_set", None)
            # Lấy dose_grid từ plan
            dose_grid = getattr(plan, "dose_grid", None)

            if structure_set and dose_grid:
                # Tính toán DVH cho tất cả cấu trúc
                dvh_data = {}
                for structure in structure_set.structures:
                    # Tạo roi_mask từ structure
                    roi_mask = (
                        structure.get_mask() if hasattr(structure, "get_mask") else None
                    )
                    if roi_mask is not None:
                        structure_dvh = calculator.calculate_dvh(
                            structure, dose_grid, roi_mask
                        )
                        dvh_data[structure.name] = structure_dvh
            else:
                logger.warning(
                    "Không thể tính toán DVH: Thiếu structure_set hoặc dose_grid"
                )
                dvh_data = {}
        else:
            dvh_data = plan.dvh_data

        # Tạo báo cáo theo định dạng
        if format.lower() == "html":
            return report_generator.generate_html_report(patient, plan, dvh_data)
        elif format.lower() == "csv":
            # Tính toán các chỉ số DVH
            prescription_dose = None
            if hasattr(plan, "prescription") and plan.prescription:
                prescription_dose = getattr(plan.prescription, "total_dose", None)

            dvh_metrics = report_generator.calculate_dvh_metrics(
                dvh_data, prescription_dose
            )
            return report_generator.export_metrics_to_csv(dvh_metrics)
        elif format.lower() == "pdf":
            # TODO: Triển khai xuất PDF
            logger.warning("Xuất báo cáo PDF chưa được triển khai.")
            return None
        else:
            logger.error(f"Định dạng báo cáo không hợp lệ: {format}")
            return None

    except Exception as e:
        logger.error(f"Lỗi khi tạo báo cáo DVH: {e}")
        return None
