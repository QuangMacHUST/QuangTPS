#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module tạo báo cáo sai lệch QA trong QuangTPS.

Module này cung cấp các công cụ để tạo báo cáo HTML và PDF về sai lệch
của các tham số máy điều trị so với kế hoạch.
"""

import os
import io
import base64
import logging
import tempfile
import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

# Thử nhập các thư viện tùy chọn
try:
    import seaborn as sns

    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

try:
    from matplotlib.colors import LinearSegmentedColormap

    HAS_COLORMAP = True
except ImportError:
    HAS_COLORMAP = False

logger = logging.getLogger(__name__)


def ensure_directory_exists(dir_path: str) -> bool:
    """Đảm bảo thư mục tồn tại, tạo nếu chưa có."""
    if not os.path.exists(dir_path):
        try:
            os.makedirs(dir_path)
        except Exception as e:
            logger.error(f"Không thể tạo thư mục {dir_path}: {e}")
            return False
    return True


def create_deviation_report(
    data: Dict[str, Any],
    output_dir: Optional[str] = None,
    template_name: Optional[str] = None,
) -> Optional[str]:
    """
    Tạo báo cáo HTML cho dữ liệu QA.

    Parameters:
        data: Dictionary chứa dữ liệu báo cáo
        output_dir: Thư mục đầu ra, nếu None sẽ sử dụng thư mục tạm
        template_name: Tên mẫu báo cáo cần sử dụng, None để sử dụng mẫu mặc định

    Returns:
        Đường dẫn file báo cáo HTML, None nếu tạo thất bại
    """
    try:
        # Tạo thư mục output nếu chưa có
        if output_dir is None:
            output_dir = tempfile.mkdtemp(prefix="quangtps_qa_report_")
        else:
            if not ensure_directory_exists(output_dir):
                logger.error(f"Không thể tạo thư mục báo cáo: {output_dir}")
                return None

        # Xác định loại báo cáo dựa trên dữ liệu và template
        if template_name == "delivery_qa_report.html":
            # Báo cáo Delivery QA
            html_content = _generate_delivery_qa_report(data)
        else:
            # Báo cáo sai lệch QA mặc định
            deviations = data.get("deviations", [])
            summary = data.get("summary", {})
            log_data = data.get("log_data")

            # Tạo các biểu đồ
            figures = _create_report_figures(deviations, log_data)

            # Tạo nội dung HTML
            html_content = _generate_html_report(deviations, summary, figures)

        # Ghi nội dung ra file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"qa_report_{timestamp}.html"
        report_path = os.path.join(output_dir, report_filename)

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        logger.info(f"Đã tạo báo cáo QA: {report_path}")
        return report_path

    except Exception as e:
        logger.error(f"Lỗi khi tạo báo cáo: {str(e)}")
        import traceback

        traceback.print_exc()
        return None


def _create_report_figures(
    deviations: List[Dict[str, Any]], log_data: Optional[pd.DataFrame] = None
) -> Dict[str, str]:
    """
    Tạo các biểu đồ cho báo cáo sai lệch.

    Parameters:
        deviations: Danh sách các sai lệch
        log_data: Dữ liệu log dạng DataFrame (tùy chọn)

    Returns:
        Dictionary chứa các biểu đồ dạng base64
    """
    figures = {}

    # Biểu đồ phân bố mức độ nghiêm trọng
    figures["severity_distribution"] = _create_severity_distribution_chart(deviations)

    # Biểu đồ sai lệch theo thời gian cho từng loại
    deviation_types = set()
    for dev in deviations:
        deviation_types.add(dev["type"])

    for dev_type in deviation_types:
        figures[f"time_series_{dev_type}"] = _create_time_series_chart(
            deviations, dev_type
        )

    # Biểu đồ heatmap MLC (nếu có data MLC và log_data)
    mlc_deviations = [d for d in deviations if "mlc" in d["type"].lower()]
    if mlc_deviations and log_data is not None and "mlc_positions" in log_data.columns:
        figures["mlc_heatmap"] = _create_mlc_heatmap(mlc_deviations, log_data)

    return figures


def _create_severity_distribution_chart(deviations: List[Dict[str, Any]]) -> str:
    """
    Tạo biểu đồ phân bố mức độ nghiêm trọng.

    Parameters:
        deviations: Danh sách các sai lệch

    Returns:
        Biểu đồ dạng base64 string
    """
    try:
        # Đếm theo mức độ nghiêm trọng
        severity_counts = {}
        for dev in deviations:
            severity = dev.get("severity", "unknown")
            if severity in severity_counts:
                severity_counts[severity] += 1
            else:
                severity_counts[severity] = 1

        # Sắp xếp mức độ nghiêm trọng
        severity_order = [
            "critical",
            "major",
            "moderate",
            "minor",
            "acceptable",
            "unknown",
        ]
        severities = []
        counts = []

        for sev in severity_order:
            if sev in severity_counts:
                severities.append(sev)
                counts.append(severity_counts[sev])

        # Tạo biểu đồ
        fig, ax = plt.subplots(figsize=(10, 6))

        # Tạo mapping màu sắc cho các mức độ nghiêm trọng
        colors = {
            "critical": "red",
            "major": "orange",
            "moderate": "yellow",
            "minor": "lightgreen",
            "acceptable": "green",
            "unknown": "gray",
        }

        # Danh sách màu sắc cho các cột
        bar_colors = [colors.get(sev, "gray") for sev in severities]

        # Vẽ biểu đồ cột
        bars = ax.bar(severities, counts, color=bar_colors)

        # Thêm nhãn số lượng lên đầu mỗi cột
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.1,
                str(int(height)),
                ha="center",
                va="bottom",
            )

        # Đặt tiêu đề và nhãn
        ax.set_title("Phân bố mức độ nghiêm trọng")
        ax.set_xlabel("Mức độ nghiêm trọng")
        ax.set_ylabel("Số lượng")

        # Làm đẹp biểu đồ
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        return _fig_to_base64(fig)

    except Exception as e:
        logger.error(f"Lỗi khi tạo biểu đồ phân bố mức độ nghiêm trọng: {str(e)}")
        return ""


def _create_time_series_chart(deviations: List[Dict[str, Any]], dev_type: str) -> str:
    """
    Tạo biểu đồ sai lệch theo thời gian cho một loại cụ thể.

    Parameters:
        deviations: Danh sách các sai lệch
        dev_type: Loại sai lệch cần vẽ biểu đồ

    Returns:
        Biểu đồ dạng base64 string
    """
    try:
        # Lọc sai lệch theo loại
        type_deviations = [d for d in deviations if d.get("type") == dev_type]

        if not type_deviations:
            return ""

        # Lấy dữ liệu
        timestamps = [d.get("timestamp", i) for i, d in enumerate(type_deviations)]
        values = [d.get("value", 0) for d in type_deviations]
        severity = [d.get("severity", "unknown") for d in type_deviations]

        # Ánh xạ màu sắc cho mức độ nghiêm trọng
        colors = {
            "critical": "red",
            "major": "orange",
            "moderate": "yellow",
            "minor": "lightgreen",
            "acceptable": "green",
            "unknown": "gray",
        }

        # Màu sắc cho mỗi điểm dữ liệu
        point_colors = [colors.get(sev, "gray") for sev in severity]

        # Tạo biểu đồ
        fig, ax = plt.subplots(figsize=(12, 6))

        # Vẽ đường và điểm
        ax.plot(timestamps, values, "b-", alpha=0.5)
        ax.scatter(timestamps, values, c=point_colors, s=50, zorder=5)

        # Lấy đơn vị đo từ deviation đầu tiên
        unit = type_deviations[0].get("unit", "")

        # Đặt tiêu đề và nhãn
        ax.set_title(f"Sai lệch {dev_type} theo thời gian")
        ax.set_xlabel("Thời gian/Mẫu")
        ax.set_ylabel(f"Giá trị ({unit})" if unit else "Giá trị")

        # Thêm đường tham chiếu cho các ngưỡng nếu có
        if "tolerance" in type_deviations[0]:
            tolerance = type_deviations[0]["tolerance"]

            for level, value in tolerance.items():
                if level == "critical":
                    ax.axhline(
                        y=value,
                        color="red",
                        linestyle="--",
                        alpha=0.7,
                        label=f"Critical ({value})",
                    )
                elif level == "major":
                    ax.axhline(
                        y=value,
                        color="orange",
                        linestyle="--",
                        alpha=0.7,
                        label=f"Major ({value})",
                    )
                elif level == "moderate":
                    ax.axhline(
                        y=value,
                        color="yellow",
                        linestyle="--",
                        alpha=0.7,
                        label=f"Moderate ({value})",
                    )
                elif level == "minor":
                    ax.axhline(
                        y=value,
                        color="green",
                        linestyle="--",
                        alpha=0.7,
                        label=f"Minor ({value})",
                    )

        # Thêm chú thích
        ax.legend()

        # Làm đẹp biểu đồ
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        return _fig_to_base64(fig)

    except Exception as e:
        logger.error(f"Lỗi khi tạo biểu đồ sai lệch theo thời gian: {str(e)}")
        return ""


def _get_safe_colormap(name="viridis"):
    """
    Lấy colormap một cách an toàn, với fallback nếu không tìm thấy.

    Parameters:
        name: Tên colormap cần lấy

    Returns:
        Colormap có sẵn
    """
    try:
        return getattr(plt.cm, name, plt.cm.jet)
    except AttributeError:
        return plt.cm.jet


def _create_mlc_heatmap(
    mlc_deviations: List[Dict[str, Any]], log_data: pd.DataFrame
) -> str:
    """
    Tạo biểu đồ heatmap cho sai lệch MLC.

    Parameters:
        mlc_deviations: Danh sách các sai lệch MLC
        log_data: Dữ liệu log dạng DataFrame

    Returns:
        Biểu đồ dạng base64 string
    """
    try:
        # Kiểm tra có thể tạo heatmap không
        if not HAS_COLORMAP:
            logger.warning(
                "Không thể tạo heatmap MLC: Thiếu thư viện matplotlib.colors.LinearSegmentedColormap"
            )
            return ""

        # Tạo ma trận cho heatmap
        max_position = 0
        leaf_positions = {}

        # Tìm số lá MLC và các vị trí
        for dev in mlc_deviations:
            if "position" in dev and ":" in dev["position"]:
                bank, leaf = dev["position"].split(":")
                leaf = int(leaf)
                max_position = max(max_position, leaf)

                key = f"{bank}_{leaf}"
                if key not in leaf_positions:
                    leaf_positions[key] = []

                leaf_positions[key].append(abs(dev["value"]))

        # Nếu không có dữ liệu vị trí cụ thể, thử ước lượng từ log_data
        if not leaf_positions and "mlc_positions" in log_data.columns:
            # Ước lượng từ dữ liệu log
            sample_mlc = log_data["mlc_positions"].iloc[0]
            if isinstance(sample_mlc, (list, np.ndarray)):
                num_leaves = len(sample_mlc)
                max_position = num_leaves - 1
            else:
                # Không thể ước lượng
                return ""

        # Số lượng lá MLC
        num_leaves = max_position + 1

        # Tạo ma trận trung bình sai lệch
        deviation_matrix = np.zeros((2, num_leaves))

        # Điền dữ liệu vào ma trận
        for key, values in leaf_positions.items():
            bank, leaf = key.split("_")
            bank_idx = 0 if bank == "A" else 1
            leaf_idx = int(leaf)

            if values:
                deviation_matrix[bank_idx, leaf_idx] = np.mean(values)

        # Tạo biểu đồ heatmap
        fig, ax = plt.subplots(figsize=(14, 6))

        # Tạo colormap tùy chỉnh: xanh lá (tốt) -> vàng -> đỏ (xấu)
        cmap = LinearSegmentedColormap.from_list(
            "custom_cmap", [(0, "green"), (0.5, "yellow"), (1, "red")]
        )

        # Vẽ heatmap
        im = ax.imshow(deviation_matrix, cmap=cmap, aspect="auto")

        # Thêm colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Sai lệch trung bình (mm)")

        # Đặt nhãn trục
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Bank A", "Bank B"])

        # Đặt nhãn cho leaf số 10, 20, 30, ...
        leaf_tick_indices = list(range(0, num_leaves, 10))
        if leaf_tick_indices and leaf_tick_indices[-1] != num_leaves - 1:
            leaf_tick_indices.append(num_leaves - 1)

        ax.set_xticks(leaf_tick_indices)
        ax.set_xticklabels([str(i) for i in leaf_tick_indices])

        # Đặt tiêu đề
        ax.set_title("Heatmap sai lệch MLC")
        ax.set_xlabel("Số lá MLC")

        # Làm đẹp
        for edge in ["top", "right", "bottom", "left"]:
            ax.spines[edge].set_visible(True)
            ax.spines[edge].set_color("black")
            ax.spines[edge].set_linewidth(0.5)

        return _fig_to_base64(fig)

    except Exception as e:
        logger.error(f"Lỗi khi tạo heatmap MLC: {str(e)}")
        import traceback

        traceback.print_exc()
        return ""


def _generate_delivery_qa_report(data: Dict[str, Any]) -> str:
    """
    Tạo báo cáo HTML cho Delivery QA.

    Parameters:
        data: Dictionary chứa dữ liệu báo cáo

    Returns:
        Nội dung HTML của báo cáo
    """
    try:
        # Lấy dữ liệu
        qa_info = data.get("qa_info", {})
        gamma_results = data.get("gamma_results", {})
        dvh_comparison = data.get("dvh_comparison", {})
        log_analysis = data.get("log_analysis", {})

        # Tạo các phần dữ liệu từ gamma analysis
        gamma_pass_rate = gamma_results.get("pass_rate", 0)
        gamma_criteria = gamma_results.get("criteria", "3%/3mm")
        gamma_image = gamma_results.get("image", "")

        # Tạo các phần dữ liệu từ DVH comparison
        dvh_image = dvh_comparison.get("image", "")
        dvh_metrics = dvh_comparison.get("metrics", {})

        # Tạo các phần dữ liệu từ log analysis
        log_pass_rate = log_analysis.get("pass_rate", 0)
        log_deviations = log_analysis.get("deviations", [])

        # Tạo HTML
        html = f"""
<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Báo cáo QA điều trị</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}

        .report-header {{
            text-align: center;
            margin-bottom: 30px;
        }}

        .report-section {{
            background-color: white;
            padding: 20px;
            margin-bottom: 30px;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}

        h1, h2, h3 {{
            color: #2c3e50;
        }}

        h1 {{
            margin-bottom: 10px;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 20px;
        }}

        th, td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}

        th {{
            background-color: #3498db;
            color: white;
        }}

        tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}

        tr:hover {{
            background-color: #f5f5f5;
        }}

        .summary-box {{
            display: flex;
            flex-wrap: wrap;
            justify-content: space-between;
            margin-bottom: 20px;
        }}

        .summary-item {{
            flex-basis: 48%;
            padding: 15px;
            margin-bottom: 15px;
            background-color: #f8f9fa;
            border-radius: 5px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}

        .summary-value {{
            font-size: 24px;
            font-weight: bold;
            margin: 10px 0;
        }}

        .progress-container {{
            width: 100%;
            background-color: #e9ecef;
            border-radius: 4px;
            margin-top: 10px;
        }}

        .progress-bar {{
            height: 20px;
            border-radius: 4px;
            text-align: center;
            color: white;
            font-weight: bold;
        }}

        .progress-bar.good {{
            background-color: #28a745;
        }}

        .progress-bar.warning {{
            background-color: #ffc107;
            color: #212529;
        }}

        .progress-bar.danger {{
            background-color: #dc3545;
        }}

        .tabs {{
            overflow: hidden;
            background-color: #f1f1f1;
            border-radius: 5px 5px 0 0;
        }}

        .tab {{
            background-color: inherit;
            float: left;
            border: none;
            outline: none;
            cursor: pointer;
            padding: 14px 16px;
            transition: 0.3s;
            font-size: 16px;
        }}

        .tab:hover {{
            background-color: #ddd;
        }}

        .tab.active {{
            background-color: #3498db;
            color: white;
        }}

        .tabcontent {{
            display: none;
            padding: 20px;
            border: 1px solid #ccc;
            border-top: none;
            border-radius: 0 0 5px 5px;
            animation: fadeEffect 1s;
        }}

        @keyframes fadeEffect {{
            from {{opacity: 0;}}
            to {{opacity: 1;}}
        }}

        .chart-container {{
            margin: 20px 0;
            text-align: center;
        }}

        .badge {{
            display: inline-block;
            padding: 5px 10px;
            border-radius: 3px;
            color: white;
            font-weight: bold;
        }}

        .critical {{
            background-color: #dc3545;
        }}

        .major {{
            background-color: #fd7e14;
        }}

        .moderate {{
            background-color: #ffc107;
            color: #212529;
        }}

        .minor {{
            background-color: #28a745;
        }}

        .acceptable {{
            background-color: #20c997;
        }}
    </style>
</head>
<body>
    <div class="report-header">
        <h1>Báo cáo QA điều trị</h1>
        <p>Tạo ngày: {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}</p>
    </div>

    <div class="report-section">
        <h2>Thông tin chung</h2>
        <div class="summary-box">
            <div class="summary-item">
                <h3>Kế hoạch</h3>
                <div class="summary-value">{qa_info.get("plan_name", "N/A")}</div>
            </div>
            <div class="summary-item">
                <h3>Bệnh nhân</h3>
                <div class="summary-value">{qa_info.get("patient_name", "N/A")}</div>
            </div>
            <div class="summary-item">
                <h3>Ngày QA</h3>
                <div class="summary-value">{qa_info.get("qa_date", "N/A")}</div>
            </div>
            <div class="summary-item">
                <h3>Thiết bị QA</h3>
                <div class="summary-value">{qa_info.get("qa_device", "N/A")}</div>
            </div>
        </div>
    </div>

    <div class="report-section">
        <h2>Kết quả QA</h2>

        <div class="tabs">
            <button class="tab active" onclick="openTab(event, 'Gamma')">Phân tích Gamma</button>
            <button class="tab" onclick="openTab(event, 'DVH')">So sánh DVH</button>
            <button class="tab" onclick="openTab(event, 'LogAnalysis')">Phân tích Log File</button>
        </div>

        <div id="Gamma" class="tabcontent" style="display: block;">
            <h3>Phân tích Gamma</h3>
            <div class="summary-box">
                <div class="summary-item">
                    <h3>Tỷ lệ đạt</h3>
                    <div class="summary-value">{gamma_pass_rate:.2f}%</div>
                    <div class="progress-container">
                        <div class="progress-bar"
                             data-rate="{gamma_pass_rate:.2f}"
                             style="width:{gamma_pass_rate}%">{gamma_pass_rate:.2f}%</div>
                    </div>
                </div>
                <div class="summary-item">
                    <h3>Tiêu chí</h3>
                    <div class="summary-value">{gamma_criteria}</div>
                </div>
            </div>
            <div class="chart-container">
                <img src="data:image/png;base64,{gamma_image}" alt="Phân tích Gamma" style="max-width: 100%;">
            </div>
        </div>

        <div id="DVH" class="tabcontent">
            <h3>So sánh DVH</h3>
            <div class="chart-container">
                <img src="data:image/png;base64,{dvh_image}" alt="So sánh DVH" style="max-width: 100%;">
            </div>
            <table>
                <tr>
                    <th>Cấu trúc</th>
                    <th>Chỉ số</th>
                    <th>Kế hoạch</th>
                    <th>Đo lường</th>
                    <th>Sai lệch</th>
                    <th>Trạng thái</th>
                </tr>
        """

        # Thêm các hàng DVH metrics
        for structure, metrics in dvh_metrics.items():
            for metric, values in metrics.items():
                planned = values.get("planned", 0)
                measured = values.get("measured", 0)
                difference = values.get("difference", 0)
                status = values.get("status", "Đạt")

                status_class = "minor" if status == "Đạt" else "major"

                html += f"""
                <tr>
                    <td>{structure}</td>
                    <td>{metric}</td>
                    <td>{planned}</td>
                    <td>{measured}</td>
                    <td>{difference}</td>
                    <td><span class="badge {status_class}">{status}</span></td>
                </tr>
                """

        html += """
            </table>
        </div>

        <div id="LogAnalysis" class="tabcontent">
            <h3>Phân tích Log File</h3>
            <div class="summary-box">
                <div class="summary-item">
                    <h3>Tỷ lệ đạt</h3>
                    <div class="summary-value">{log_pass_rate:.2f}%</div>
                    <div class="progress-container">
                        <div class="progress-bar"
                             data-rate="{log_pass_rate:.2f}"
                             style="width:{log_pass_rate}%">{log_pass_rate:.2f}%</div>
                    </div>
                </div>
            </div>
            <h3>Chi tiết sai lệch</h3>
            <table>
                <tr>
                    <th>STT</th>
                    <th>Loại</th>
                    <th>Mức độ</th>
                    <th>Giá trị tối đa</th>
                    <th>Đơn vị</th>
                </tr>
        """

        # Thêm các hàng log deviations
        for i, dev in enumerate(log_deviations):
            html += f"""
                <tr>
                    <td>{i + 1}</td>
                    <td>{dev.get("type", "")}</td>
                    <td><span class="badge {dev.get("severity", "minor")}">{dev.get("severity", "minor")}</span></td>
                    <td>{dev.get("value", 0):.4f}</td>
                    <td>{dev.get("unit", "")}</td>
                </tr>
            """

        html += """
            </table>
        </div>
    </div>

    <script>
    function openTab(evt, tabName) {
      var i, tabcontent, tablinks;
      tabcontent = document.getElementsByClassName("tabcontent");
      for (i = 0; i < tabcontent.length; i++) {
        tabcontent[i].style.display = "none";
      }
      tablinks = document.getElementsByClassName("tab");
      for (i = 0; i < tablinks.length; i++) {
        tablinks[i].className = tablinks[i].className.replace(" active", "");
      }
      document.getElementById(tabName).style.display = "block";
      evt.currentTarget.className += " active";
    }

    // Xử lý các thanh progress bar
    document.addEventListener('DOMContentLoaded', function() {
        var progressBars = document.querySelectorAll('.progress-bar[data-rate]');
        progressBars.forEach(function(bar) {
            var rate = parseFloat(bar.getAttribute('data-rate'));
            if (rate >= 95) {
                bar.classList.add('good');
            } else if (rate >= 90) {
                bar.classList.add('warning');
            } else {
                bar.classList.add('danger');
            }
        });
    });
    </script>
</body>
</html>
        """

        return html

    except Exception as e:
        logger.error(f"Lỗi khi tạo báo cáo Delivery QA: {str(e)}")
        import traceback

        traceback.print_exc()
        return f"<html><body><h1>Lỗi khi tạo báo cáo</h1><p>{str(e)}</p></body></html>"


def _generate_html_report(
    deviations: List[Dict[str, Any]], summary: Dict[str, Any], figures: Dict[str, str]
) -> str:
    """
    Tạo báo cáo HTML cho dữ liệu sai lệch.

    Parameters:
        deviations: Danh sách các sai lệch
        summary: Tóm tắt phân tích
        figures: Các biểu đồ dạng base64

    Returns:
        Nội dung HTML của báo cáo
    """
    try:
        # Xử lý dữ liệu tóm tắt
        total_deviations = len(deviations)
        critical_deviations = sum(
            1 for d in deviations if d.get("severity") == "critical"
        )
        major_deviations = sum(1 for d in deviations if d.get("severity") == "major")

        # Tỷ lệ đạt
        pass_rate = summary.get("pass_rate", 0)

        # Loại máy
        machine_type = summary.get("machine_type", "Unknown")

        # Ngày giờ phân tích
        analysis_date = summary.get(
            "analysis_date", datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )

        # Tạo báo cáo HTML
        html = f"""
<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Báo cáo phân tích sai lệch</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: #fff;
            padding: 20px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
            border-radius: 5px;
        }}

        h1, h2, h3 {{
            color: #2c3e50;
        }}

        h1 {{
            text-align: center;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }}

        .header-section {{
            display: flex;
            justify-content: space-between;
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #3498db;
            margin-bottom: 20px;
        }}

        .info-group {{
            flex: 1;
        }}

        .info-item {{
            margin-bottom: 5px;
        }}

        .info-label {{
            font-weight: bold;
            margin-right: 5px;
            color: #7f8c8d;
        }}

        .summary-section {{
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
            margin-bottom: 30px;
        }}

        .summary-card {{
            flex: 1;
            min-width: 200px;
            background-color: #fff;
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 15px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            text-align: center;
        }}

        .summary-value {{
            font-size: 24px;
            font-weight: bold;
            margin: 10px 0;
        }}

        .chart-container {{
            margin-bottom: 30px;
            background-color: #fff;
            border-radius: 5px;
            padding: 15px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }}

        .chart-title {{
            margin-bottom: 15px;
            color: #2c3e50;
            border-bottom: 1px solid #eee;
            padding-bottom: 10px;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 20px;
        }}

        th, td {{
            padding: 10px 15px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}

        th {{
            background-color: #3498db;
            color: white;
        }}

        tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}

        tr:hover {{
            background-color: #f5f5f5;
        }}

        .badge {{
            display: inline-block;
            padding: 3px 8px;
            border-radius: 3px;
            font-size: 12px;
            font-weight: bold;
            color: white;
        }}

        .badge.critical {{
            background-color: #e74c3c;
        }}

        .badge.major {{
            background-color: #e67e22;
        }}

        .badge.moderate {{
            background-color: #f39c12;
        }}

        .badge.minor {{
            background-color: #3498db;
        }}

        .badge.acceptable {{
            background-color: #27ae60;
        }}

        .progress-container {{
            width: 100%;
            background-color: #e9ecef;
            border-radius: 4px;
            margin-top: 10px;
        }}

        .progress-bar {{
            height: 20px;
            border-radius: 4px;
            text-align: center;
            color: white;
            font-weight: bold;
            line-height: 20px;
            font-size: 12px;
        }}

        .progress-bar.good {{
            background-color: #27ae60;
        }}

        .progress-bar.warning {{
            background-color: #f39c12;
        }}

        .progress-bar.danger {{
            background-color: #e74c3c;
        }}

        img {{
            max-width: 100%;
            height: auto;
            display: block;
            margin: 0 auto;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Báo cáo phân tích sai lệch máy điều trị</h1>

        <div class="header-section">
            <div class="info-group">
                <div class="info-item">
                    <span class="info-label">Loại máy:</span>
                    <span>{machine_type}</span>
                </div>
                <div class="info-item">
                    <span class="info-label">Tổng số sai lệch:</span>
                    <span>{total_deviations}</span>
                </div>
            </div>
            <div class="info-group">
                <div class="info-item">
                    <span class="info-label">Sai lệch nghiêm trọng:</span>
                    <span>{critical_deviations}</span>
                </div>
                <div class="info-item">
                    <span class="info-label">Sai lệch lớn:</span>
                    <span>{major_deviations}</span>
                </div>
            </div>
            <div class="info-group">
                <div class="info-item">
                    <span class="info-label">Ngày phân tích:</span>
                    <span>{analysis_date}</span>
                </div>
            </div>
        </div>

        <div class="summary-section">
            <div class="summary-card">
                <h3>Tỷ lệ đạt QA</h3>
                <div class="summary-value">{pass_rate:.2f}%</div>
                <div class="progress-container">
                    <div class="progress-bar"
                         data-rate="{pass_rate:.2f}"
                         style="width: {pass_rate}%">{pass_rate:.2f}%</div>
                </div>
            </div>
            <div class="summary-card">
                <h3>Phân bố mức độ</h3>
                <img src="data:image/png;base64,{figures.get("severity_distribution", "")}"
                     alt="Phân bố mức độ nghiêm trọng" style="max-width: 100%; height: auto;">
            </div>
        </div>

        <h2>Chi tiết sai lệch</h2>

        <table>
            <tr>
                <th>STT</th>
                <th>Loại sai lệch</th>
                <th>Mức độ</th>
                <th>Giá trị</th>
                <th>Đơn vị</th>
                <th>Thời điểm</th>
            </tr>
        """

        # Thêm dữ liệu từng sai lệch
        for i, dev in enumerate(
            deviations[:100]
        ):  # Giới hạn 100 sai lệch để tránh file quá lớn
            severity = dev.get("severity", "unknown")
            html += f"""
            <tr>
                <td>{i + 1}</td>
                <td>{dev.get("type", "")}</td>
                <td><span class="badge {severity}">{severity}</span></td>
                <td>{dev.get("value", 0):.4f}</td>
                <td>{dev.get("unit", "")}</td>
                <td>{dev.get("timestamp", "")}</td>
            </tr>
            """

        # Thêm thông báo nếu có quá nhiều sai lệch
        if len(deviations) > 100:
            html += f"""
            <tr>
                <td colspan="6" style="text-align: center; font-style: italic;">
                    ... và {len(deviations) - 100} sai lệch khác (giới hạn hiển thị 100)
                </td>
            </tr>
            """

        html += """
        </table>
        """

        # Thêm các biểu đồ sai lệch theo thời gian
        if any(key.startswith("time_series_") for key in figures.keys()):
            html += """
        <h2>Biểu đồ sai lệch theo thời gian</h2>
            """

            for key, img_data in figures.items():
                if key.startswith("time_series_") and img_data:
                    deviation_type = key.replace("time_series_", "")
                    html += f"""
        <div class="chart-container">
            <h3 class="chart-title">Sai lệch {deviation_type} theo thời gian</h3>
            <img src="data:image/png;base64,{img_data}" alt="Biểu đồ sai lệch {deviation_type}">
        </div>
                    """

        # Thêm heatmap MLC nếu có
        if "mlc_heatmap" in figures and figures["mlc_heatmap"]:
            html += """
        <h2>Heatmap sai lệch MLC</h2>
        <div class="chart-container">
            <img src="data:image/png;base64,{0}" alt="Heatmap sai lệch MLC">
        </div>
            """.format(figures["mlc_heatmap"])

        html += """
    </div>
</body>
</html>
        """

        return html

    except Exception as e:
        logger.error(f"Lỗi khi tạo báo cáo HTML: {str(e)}")
        import traceback

        traceback.print_exc()
        return f"<html><body><h1>Lỗi khi tạo báo cáo</h1><p>{str(e)}</p></body></html>"


def _fig_to_base64(fig):
    """
    Chuyển đổi matplotlib figure thành chuỗi base64.

    Parameters:
        fig: Matplotlib figure

    Returns:
        Chuỗi base64 của hình ảnh
    """
    try:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)  # Đóng figure để giải phóng bộ nhớ
        return img_base64
    except Exception as e:
        logger.error(f"Lỗi khi chuyển đổi hình ảnh sang base64: {str(e)}")
        plt.close(fig)  # Đảm bảo đóng figure ngay cả khi có lỗi
        return ""
