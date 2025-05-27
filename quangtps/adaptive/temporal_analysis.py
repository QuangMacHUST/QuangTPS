#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module phân tích biến đổi giải phẫu theo thời gian trong QuangTPS.

Module này cung cấp các chức năng để phân tích sự thay đổi hình ảnh và cấu trúc
theo thời gian, hỗ trợ cho quá trình lập kế hoạch điều trị thích ứng.
"""

import os
import logging
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Union, Any, Sequence
from enum import Enum, auto

from quangtps.core.types import Patient, Image, Structure, Dose, Plan
from quangtps.core.exceptions import TemporalAnalysisError
from quangtps.imaging.registration import ImageRegistration
from quangtps.segmentation.contour.dice import calculate_dice
from quangtps.evaluation.dvh.dvh_calculator import DVHCalculator
from quangtps.evaluation.metrics.clinical_metrics import ClinicalMetricsCalculator
from quangtps.adaptive.dose_accumulation import DoseAccumulator
from quangtps.adaptive.deformation import DeformableRegistration
from quangtps.core.utils import get_timestamp, create_directory_if_not_exists

logger = logging.getLogger(__name__)


class TemporalChangeMetric(Enum):
    """Các loại phương pháp đo lường sự thay đổi theo thời gian."""

    DICE_COEFFICIENT = auto()  # Hệ số Dice giữa các cấu trúc
    CENTROID_DISTANCE = auto()  # Khoảng cách giữa các tâm của cấu trúc
    VOLUME_CHANGE = auto()  # Thay đổi thể tích
    SURFACE_DISTANCE = auto()  # Khoảng cách bề mặt trung bình
    HAUSDORFF_DISTANCE = auto()  # Khoảng cách Hausdorff
    JACOBIAN_ANALYSIS = auto()  # Phân tích Jacobian từ dòng biến dạng
    DVH_PARAMETER_CHANGE = auto()  # Thay đổi trong tham số DVH
    HU_DISTRIBUTION_CHANGE = auto()  # Thay đổi trong phân phối đơn vị Hounsfield


class TemporalAnalysisResult:
    """Lưu trữ kết quả phân tích biến đổi theo thời gian."""

    def __init__(self, reference_date: datetime.datetime):
        """
        Khởi tạo đối tượng kết quả phân tích.

        Parameters
        ----------
        reference_date : datetime.datetime
            Ngày tham chiếu cơ sở để so sánh
        """
        self.reference_date = reference_date
        self.analysis_date = datetime.datetime.now()
        self.timepoints = []  # Danh sách các mốc thời gian
        self.structure_metrics = {}  # Dict[str, Dict[str, Dict[str, float]]]
        self.image_metrics = {}  # Dict[str, Dict[str, float]]
        self.dose_metrics = {}  # Dict[str, Dict[str, Dict[str, float]]]
        self.deformation_metrics = {}  # Dict[str, Dict[str, float]]

    def add_timepoint(self, date: datetime.datetime, description: str = ""):
        """Thêm mốc thời gian mới vào phân tích."""
        timepoint_info = {
            "date": date,
            "days_from_reference": (date - self.reference_date).days,
            "description": description,
        }
        self.timepoints.append(timepoint_info)

    def add_structure_metric(
        self,
        timepoint_idx: int,
        structure_name: str,
        metric_type: TemporalChangeMetric,
        value: float,
    ):
        """Thêm giá trị đo lường cho một cấu trúc tại một mốc thời gian."""
        if structure_name not in self.structure_metrics:
            self.structure_metrics[structure_name] = {}

        timepoint_str = str(timepoint_idx)
        if timepoint_str not in self.structure_metrics[structure_name]:
            self.structure_metrics[structure_name][timepoint_str] = {}

        self.structure_metrics[structure_name][timepoint_str][metric_type.name] = value

    def add_image_metric(self, timepoint_idx: int, metric_type: str, value: float):
        """Thêm giá trị đo lường cho một hình ảnh tại một mốc thời gian."""
        timepoint_str = str(timepoint_idx)
        if timepoint_str not in self.image_metrics:
            self.image_metrics[timepoint_str] = {}

        self.image_metrics[timepoint_str][metric_type] = value

    def add_dose_metric(
        self, timepoint_idx: int, structure_name: str, metric_name: str, value: float
    ):
        """Thêm giá trị đo lường liều cho một cấu trúc tại một mốc thời gian."""
        if structure_name not in self.dose_metrics:
            self.dose_metrics[structure_name] = {}

        timepoint_str = str(timepoint_idx)
        if timepoint_str not in self.dose_metrics[structure_name]:
            self.dose_metrics[structure_name][timepoint_str] = {}

        self.dose_metrics[structure_name][timepoint_str][metric_name] = value

    def add_deformation_metric(
        self, timepoint_idx: int, metric_name: str, value: float
    ):
        """Thêm giá trị đo lường biến dạng tại một mốc thời gian."""
        timepoint_str = str(timepoint_idx)
        if timepoint_str not in self.deformation_metrics:
            self.deformation_metrics[timepoint_str] = {}

        self.deformation_metrics[timepoint_str][metric_name] = value

    def get_metric_timeseries(
        self, structure_name: str, metric_type: TemporalChangeMetric
    ) -> pd.Series:
        """Lấy chuỗi thời gian cho một đo lường cụ thể của một cấu trúc."""
        if structure_name not in self.structure_metrics:
            return pd.Series()

        timepoints = []
        values = []

        for tp_idx, tp_info in enumerate(self.timepoints):
            tp_str = str(tp_idx)
            if (
                tp_str in self.structure_metrics[structure_name]
                and metric_type.name in self.structure_metrics[structure_name][tp_str]
            ):
                timepoints.append(tp_info["days_from_reference"])
                values.append(
                    self.structure_metrics[structure_name][tp_str][metric_type.name]
                )

        return pd.Series(values, index=timepoints)

    def to_dataframe(self) -> Dict[str, pd.DataFrame]:
        """Chuyển đổi kết quả phân tích thành các DataFrame."""
        result = {}

        # Tạo dataframe cho các số liệu cấu trúc
        structure_dfs = {}
        for structure_name, timepoint_data in self.structure_metrics.items():
            rows = []
            for tp_idx, tp_info in enumerate(self.timepoints):
                tp_str = str(tp_idx)
                if tp_str in timepoint_data:
                    row = {
                        "days_from_reference": tp_info["days_from_reference"],
                        "date": tp_info["date"],
                        "description": tp_info["description"],
                    }
                    row.update(timepoint_data[tp_str])
                    rows.append(row)
            if rows:
                structure_dfs[structure_name] = pd.DataFrame(rows)

        result["structures"] = structure_dfs

        # Tạo dataframe cho các số liệu hình ảnh
        image_rows = []
        for tp_idx, tp_info in enumerate(self.timepoints):
            tp_str = str(tp_idx)
            if tp_str in self.image_metrics:
                row = {
                    "days_from_reference": tp_info["days_from_reference"],
                    "date": tp_info["date"],
                    "description": tp_info["description"],
                }
                row.update(self.image_metrics[tp_str])
                image_rows.append(row)
        if image_rows:
            result["images"] = pd.DataFrame(image_rows)

        # Tạo dataframe cho các số liệu liều
        dose_dfs = {}
        for structure_name, timepoint_data in self.dose_metrics.items():
            rows = []
            for tp_idx, tp_info in enumerate(self.timepoints):
                tp_str = str(tp_idx)
                if tp_str in timepoint_data:
                    row = {
                        "days_from_reference": tp_info["days_from_reference"],
                        "date": tp_info["date"],
                        "description": tp_info["description"],
                    }
                    row.update(timepoint_data[tp_str])
                    rows.append(row)
            if rows:
                dose_dfs[structure_name] = pd.DataFrame(rows)

        result["doses"] = dose_dfs

        return result

    def plot_metric_timeseries(
        self,
        structure_name: str,
        metric_type: TemporalChangeMetric,
        save_path: Optional[str] = None,
    ):
        """Vẽ đồ thị chuỗi thời gian cho một đo lường cụ thể của một cấu trúc."""
        series = self.get_metric_timeseries(structure_name, metric_type)

        if series.empty:
            logger.warning(
                f"Không có dữ liệu để vẽ đồ thị cho {structure_name}, metric: {metric_type.name}"
            )
            return

        plt.figure(figsize=(10, 6))
        plt.plot(series.index, series.values, "o-")
        plt.title(f"{metric_type.name} cho {structure_name} theo thời gian")
        plt.xlabel("Ngày từ tham chiếu")
        plt.ylabel(metric_type.name)
        plt.grid(True)

        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()


class TemporalAnalyzer:
    """Lớp cung cấp các phương pháp phân tích biến đổi giải phẫu theo thời gian."""

    def __init__(
        self,
        default_output_dir: str = "./temporal_analysis_results",
        deformable_registration: Optional[DeformableRegistration] = None,
        rigid_registration: Optional[ImageRegistration] = None,
    ):
        """
        Khởi tạo bộ phân tích biến đổi theo thời gian.

        Parameters
        ----------
        default_output_dir : str, optional
            Thư mục mặc định để lưu kết quả phân tích
        deformable_registration : DeformableRegistration, optional
            Đối tượng đăng ký biến dạng để sử dụng khi phân tích hình ảnh
        rigid_registration : ImageRegistration, optional
            Đối tượng đăng ký cứng để sử dụng trước khi thực hiện đăng ký biến dạng
        """
        self.default_output_dir = default_output_dir
        create_directory_if_not_exists(default_output_dir)

        # Sử dụng các công cụ đăng ký hình ảnh đã được cung cấp hoặc tạo mới
        self.deformable_registration = (
            deformable_registration or DeformableRegistration()
        )
        self.rigid_registration = rigid_registration or ImageRegistration()

        # Khởi tạo các công cụ tính toán
        self.dvh_calculator = DVHCalculator()
        self.metrics_calculator = ClinicalMetricsCalculator()
        self.dose_accumulator = DoseAccumulator()

    def analyze_timepoints(
        self,
        reference_image: Image,
        reference_structures: Dict[str, Structure],
        timepoint_images: List[Image],
        timepoint_structures: List[Dict[str, Structure]],
        timepoint_dates: List[datetime.datetime],
        timepoint_descriptions: Optional[List[str]] = None,
        metrics: Optional[List[TemporalChangeMetric]] = None,
        output_dir: Optional[str] = None,
    ) -> TemporalAnalysisResult:
        """
        Phân tích biến đổi theo thời gian giữa các hình ảnh và cấu trúc.

        Parameters
        ----------
        reference_image : Image
            Hình ảnh tham chiếu cơ sở
        reference_structures : Dict[str, Structure]
            Từ điển cấu trúc tham chiếu
        timepoint_images : List[Image]
            Danh sách các hình ảnh theo thời gian
        timepoint_structures : List[Dict[str, Structure]]
            Danh sách các từ điển cấu trúc theo thời gian
        timepoint_dates : List[datetime.datetime]
            Danh sách các ngày tương ứng với mỗi mốc thời gian
        timepoint_descriptions : Optional[List[str]], optional
            Danh sách mô tả cho mỗi mốc thời gian
        metrics : Optional[List[TemporalChangeMetric]], optional
            Danh sách các đo lường cần phân tích
        output_dir : Optional[str], optional
            Thư mục để lưu kết quả phân tích

        Returns
        -------
        TemporalAnalysisResult
            Kết quả phân tích biến đổi theo thời gian
        """
        # Khởi tạo kết quả phân tích
        reference_date = (
            timepoint_dates[0] if timepoint_dates else datetime.datetime.now()
        )
        result = TemporalAnalysisResult(reference_date)

        # Thiết lập thư mục đầu ra
        if output_dir is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(
                self.default_output_dir, f"temporal_analysis_{timestamp}"
            )
        create_directory_if_not_exists(output_dir)

        # Thiết lập các đo lường mặc định nếu không được cung cấp
        if metrics is None:
            metrics = [
                TemporalChangeMetric.DICE_COEFFICIENT,
                TemporalChangeMetric.CENTROID_DISTANCE,
                TemporalChangeMetric.VOLUME_CHANGE,
                TemporalChangeMetric.SURFACE_DISTANCE,
            ]

        # Thêm các mốc thời gian vào kết quả
        for i, date in enumerate(timepoint_dates):
            description = (
                timepoint_descriptions[i]
                if timepoint_descriptions and i < len(timepoint_descriptions)
                else ""
            )
            result.add_timepoint(date, description)

        # Phân tích từng mốc thời gian
        for tp_idx, (image, structures, date) in enumerate(
            zip(timepoint_images, timepoint_structures, timepoint_dates)
        ):
            logger.info(
                f"Phân tích mốc thời gian {tp_idx + 1}/{len(timepoint_images)}: {date.strftime('%Y-%m-%d')}"
            )

            # Tính toán các số liệu cấu trúc
            self._calculate_structure_metrics(
                reference_structures, structures, result, tp_idx, metrics
            )

            # Tính toán các số liệu hình ảnh
            self._calculate_image_metrics(reference_image, image, result, tp_idx)

            # Lưu hình ảnh kết quả
            self._save_analysis_images(
                reference_image,
                image,
                reference_structures,
                structures,
                output_dir,
                tp_idx,
                date,
            )

        # Tạo báo cáo tổng hợp
        self._generate_analysis_report(result, output_dir)

        return result

    def _calculate_structure_metrics(
        self,
        reference_structures: Dict[str, Structure],
        current_structures: Dict[str, Structure],
        result: TemporalAnalysisResult,
        timepoint_idx: int,
        metrics: List[TemporalChangeMetric],
    ):
        """Tính toán các số liệu cho cấu trúc tại một mốc thời gian."""
        for name, ref_struct in reference_structures.items():
            if name not in current_structures:
                logger.warning(
                    f"Cấu trúc {name} không có tại mốc thời gian {timepoint_idx}"
                )
                continue

            curr_struct = current_structures[name]

            # Tính toán các đo lường khác nhau dựa trên danh sách đo lường đã yêu cầu
            for metric in metrics:
                if metric == TemporalChangeMetric.DICE_COEFFICIENT:
                    dice = calculate_dice(ref_struct, curr_struct)
                    result.add_structure_metric(timepoint_idx, name, metric, dice)

                elif metric == TemporalChangeMetric.VOLUME_CHANGE:
                    # Tính phần trăm thay đổi thể tích so với tham chiếu
                    ref_vol = ref_struct.get_volume()
                    curr_vol = curr_struct.get_volume()
                    pct_change = (
                        ((curr_vol - ref_vol) / ref_vol * 100) if ref_vol > 0 else 0
                    )
                    result.add_structure_metric(timepoint_idx, name, metric, pct_change)

                elif metric == TemporalChangeMetric.CENTROID_DISTANCE:
                    # Tính khoảng cách giữa các tâm cấu trúc
                    ref_centroid = ref_struct.get_centroid()
                    curr_centroid = curr_struct.get_centroid()
                    distance = np.sqrt(
                        np.sum((np.array(ref_centroid) - np.array(curr_centroid)) ** 2)
                    )
                    result.add_structure_metric(timepoint_idx, name, metric, distance)

                elif metric == TemporalChangeMetric.SURFACE_DISTANCE:
                    # Tính khoảng cách bề mặt trung bình
                    # Lưu ý: Cần triển khai hàm tính khoảng cách bề mặt
                    distance = self._calculate_surface_distance(ref_struct, curr_struct)
                    result.add_structure_metric(timepoint_idx, name, metric, distance)

                elif metric == TemporalChangeMetric.HAUSDORFF_DISTANCE:
                    # Tính khoảng cách Hausdorff
                    # Lưu ý: Cần triển khai hàm tính khoảng cách Hausdorff
                    distance = self._calculate_hausdorff_distance(
                        ref_struct, curr_struct
                    )
                    result.add_structure_metric(timepoint_idx, name, metric, distance)

    def _calculate_image_metrics(
        self,
        reference_image: Image,
        current_image: Image,
        result: TemporalAnalysisResult,
        timepoint_idx: int,
    ):
        """Tính toán các số liệu cho hình ảnh tại một mốc thời gian."""
        # Tính tương quan chuẩn hóa chéo giữa các hình ảnh
        try:
            ncc = self._calculate_normalized_cross_correlation(
                reference_image, current_image
            )
            result.add_image_metric(timepoint_idx, "normalized_cross_correlation", ncc)
        except Exception as e:
            logger.error(f"Lỗi khi tính tương quan chuẩn hóa chéo: {e}")

        # Tính chỉ số tương tự cấu trúc (SSIM)
        try:
            ssim = self._calculate_ssim(reference_image, current_image)
            result.add_image_metric(timepoint_idx, "ssim", ssim)
        except Exception as e:
            logger.error(f"Lỗi khi tính chỉ số tương tự cấu trúc: {e}")

        # Tính sự thay đổi trung bình trong các giá trị Hounsfield
        try:
            mean_hu_change = self._calculate_mean_hu_change(
                reference_image, current_image
            )
            result.add_image_metric(timepoint_idx, "mean_hu_change", mean_hu_change)
        except Exception as e:
            logger.error(
                f"Lỗi khi tính sự thay đổi trung bình trong các giá trị Hounsfield: {e}"
            )

    def _save_analysis_images(
        self,
        reference_image: Image,
        current_image: Image,
        reference_structures: Dict[str, Structure],
        current_structures: Dict[str, Structure],
        output_dir: str,
        timepoint_idx: int,
        date: datetime.datetime,
    ):
        """Lưu các hình ảnh phân tích cho một mốc thời gian."""
        # TODO: Triển khai việc lưu hình ảnh phân tích
        timepoint_dir = os.path.join(
            output_dir, f"timepoint_{timepoint_idx}_{date.strftime('%Y%m%d')}"
        )
        create_directory_if_not_exists(timepoint_dir)

        # Lưu hình ảnh chênh lệch
        diff_path = os.path.join(timepoint_dir, "image_difference.png")
        self._save_image_difference(reference_image, current_image, diff_path)

        # Lưu hình ảnh overlay cấu trúc
        for name in reference_structures.keys():
            if name in current_structures:
                structure_path = os.path.join(
                    timepoint_dir, f"structure_{name}_overlay.png"
                )
                self._save_structure_overlay(
                    reference_image,
                    reference_structures[name],
                    current_structures[name],
                    structure_path,
                )

    def _generate_analysis_report(
        self, result: TemporalAnalysisResult, output_dir: str
    ):
        """Tạo báo cáo tổng hợp cho phân tích biến đổi theo thời gian."""
        # Xuất các DataFrame ra file CSV
        dataframes = result.to_dataframe()

        # Lưu các DataFrame cấu trúc
        structure_dir = os.path.join(output_dir, "structure_metrics")
        create_directory_if_not_exists(structure_dir)
        for structure_name, df in dataframes.get("structures", {}).items():
            df.to_csv(
                os.path.join(structure_dir, f"{structure_name}_metrics.csv"),
                index=False,
            )

        # Lưu DataFrame hình ảnh
        if "images" in dataframes:
            dataframes["images"].to_csv(
                os.path.join(output_dir, "image_metrics.csv"), index=False
            )

        # Lưu các DataFrame liều
        dose_dir = os.path.join(output_dir, "dose_metrics")
        create_directory_if_not_exists(dose_dir)
        for structure_name, df in dataframes.get("doses", {}).items():
            df.to_csv(
                os.path.join(dose_dir, f"{structure_name}_dose_metrics.csv"),
                index=False,
            )

        # Tạo báo cáo HTML tổng hợp với các đồ thị
        report_path = os.path.join(output_dir, "temporal_analysis_report.html")
        self._generate_html_report(result, report_path)

    # Các phương thức phụ trợ cho các tính toán khác nhau

    def _calculate_surface_distance(
        self, struct_a: Structure, struct_b: Structure
    ) -> float:
        """Tính khoảng cách bề mặt trung bình giữa hai cấu trúc."""
        # TODO: Triển khai tính toán khoảng cách bề mặt trung bình
        # Đây là một triển khai đơn giản, cần thay thế bằng thuật toán thực tế
        return 0.0

    def _calculate_hausdorff_distance(
        self, struct_a: Structure, struct_b: Structure
    ) -> float:
        """Tính khoảng cách Hausdorff giữa hai cấu trúc."""
        # TODO: Triển khai tính toán khoảng cách Hausdorff
        # Đây là một triển khai đơn giản, cần thay thế bằng thuật toán thực tế
        return 0.0

    def _calculate_normalized_cross_correlation(
        self, image_a: Image, image_b: Image
    ) -> float:
        """Tính tương quan chuẩn hóa chéo giữa hai hình ảnh."""
        # TODO: Triển khai tính toán tương quan chuẩn hóa chéo
        # Đây là một triển khai đơn giản, cần thay thế bằng thuật toán thực tế
        return 0.5

    def _calculate_ssim(self, image_a: Image, image_b: Image) -> float:
        """Tính chỉ số tương tự cấu trúc (SSIM) giữa hai hình ảnh."""
        # TODO: Triển khai tính toán SSIM
        # Đây là một triển khai đơn giản, cần thay thế bằng thuật toán thực tế
        return 0.7

    def _calculate_mean_hu_change(self, image_a: Image, image_b: Image) -> float:
        """Tính sự thay đổi trung bình trong các giá trị Hounsfield."""
        # TODO: Triển khai tính toán sự thay đổi HU trung bình
        # Đây là một triển khai đơn giản, cần thay thế bằng thuật toán thực tế
        return 10.0

    def _save_image_difference(self, image_a: Image, image_b: Image, save_path: str):
        """Lưu hình ảnh hiển thị sự khác biệt giữa hai hình ảnh."""
        # TODO: Triển khai lưu hình ảnh khác biệt
        logger.info(f"Lưu hình ảnh khác biệt vào {save_path}")

    def _save_structure_overlay(
        self, image: Image, struct_a: Structure, struct_b: Structure, save_path: str
    ):
        """Lưu hình ảnh hiển thị overlay của hai cấu trúc trên hình ảnh."""
        # TODO: Triển khai lưu hình ảnh overlay cấu trúc
        logger.info(f"Lưu hình ảnh overlay cấu trúc vào {save_path}")

    def _generate_html_report(self, result: TemporalAnalysisResult, report_path: str):
        """Tạo báo cáo HTML với các đồ thị và bảng dữ liệu."""
        # TODO: Triển khai tạo báo cáo HTML
        logger.info(f"Đã tạo báo cáo HTML tại {report_path}")

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Báo cáo phân tích biến đổi theo thời gian</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1, h2, h3 {{ color: #2c3e50; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        .container {{ margin-bottom: 30px; }}
    </style>
</head>
<body>
    <h1>Báo cáo phân tích biến đổi theo thời gian</h1>
    <p>Ngày phân tích: {result.analysis_date.strftime("%Y-%m-%d %H:%M:%S")}</p>
    <p>Ngày tham chiếu: {result.reference_date.strftime("%Y-%m-%d")}</p>

    <div class="container">
        <h2>Tóm tắt các mốc thời gian</h2>
        <table>
            <tr>
                <th>STT</th>
                <th>Ngày</th>
                <th>Ngày từ tham chiếu</th>
                <th>Mô tả</th>
            </tr>
""")

            # Thêm thông tin các mốc thời gian
            for i, tp in enumerate(result.timepoints):
                f.write(f"""
            <tr>
                <td>{i}</td>
                <td>{tp["date"].strftime("%Y-%m-%d")}</td>
                <td>{tp["days_from_reference"]}</td>
                <td>{tp["description"]}</td>
            </tr>""")

            f.write("""
        </table>
    </div>

    <!-- Thêm các phần khác của báo cáo tại đây -->

</body>
</html>
""")

# Convenience function
def analyze_temporal_changes(
    reference_image: Image,
    reference_structures: Dict[str, Structure],
    timepoint_images: List[Image],
    timepoint_structures: List[Dict[str, Structure]],
    timepoint_dates: List[datetime.datetime],
    output_dir: Optional[str] = None,
    metrics: Optional[List[TemporalChangeMetric]] = None
) -> TemporalAnalysisResult:
    """
    Convenience function để thực hiện phân tích temporal changes

    Parameters
    ----------
    reference_image : Image
        Hình ảnh tham chiếu
    reference_structures : Dict[str, Structure]
        Cấu trúc tham chiếu
    timepoint_images : List[Image]
        Danh sách hình ảnh theo thời gian
    timepoint_structures : List[Dict[str, Structure]]
        Danh sách cấu trúc theo thời gian
    timepoint_dates : List[datetime.datetime]
        Danh sách ngày
    output_dir : str, optional
        Thư mục xuất kết quả
    metrics : List[TemporalChangeMetric], optional
        Danh sách metrics cần tính

    Returns
    -------
    TemporalAnalysisResult
        Kết quả phân tích
    """
    analyzer = TemporalAnalyzer(default_output_dir=output_dir or "./temporal_analysis")

    return analyzer.analyze_timepoints(
        reference_image=reference_image,
        reference_structures=reference_structures,
        timepoint_images=timepoint_images,
        timepoint_structures=timepoint_structures,
        timepoint_dates=timepoint_dates,
        metrics=metrics,
        output_dir=output_dir
    )
