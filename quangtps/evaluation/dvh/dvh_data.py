#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module phân tích DVH (Dose-Volume Histogram) trong QuangTPS.

Module này cung cấp các lớp và phương thức để tính toán, phân tích, và
trực quan hóa biểu đồ liều-thể tích (DVH) cho kế hoạch xạ trị.
"""

import logging
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional, Union
import pandas as pd
import json
import os
from dataclasses import dataclass, field
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

logger = logging.getLogger(__name__)


class DVHCurve:
    """
    Lớp biểu diễn đường cong DVH (Dose-Volume Histogram).

    Lớp này lưu trữ và thao tác với dữ liệu đường cong DVH cho một cấu trúc,
    hỗ trợ tính toán các chỉ số DVH, chuyển đổi giữa DVH tích lũy và vi phân.
    """

    def __init__(
        self,
        structure_id: str,
        structure_name: str,
        dose_bins: np.ndarray,
        volume_bins: np.ndarray,
        structure_volume: float,
        is_cumulative: bool = True,
    ):
        """
        Khởi tạo đường cong DVH.

        Args:
            structure_id: ID của cấu trúc
            structure_name: Tên của cấu trúc
            dose_bins: Mảng chứa các giá trị liều (Gy)
            volume_bins: Mảng chứa các giá trị thể tích (đơn vị tùy thuộc vào is_absolute)
            structure_volume: Thể tích tổng của cấu trúc (cc)
            is_cumulative: True nếu DVH tích lũy, False nếu vi phân
        """
        self.structure_id = structure_id
        self.structure_name = structure_name
        self.dose_bins = np.array(dose_bins, dtype=float)
        self.volume_bins = np.array(volume_bins, dtype=float)
        self.structure_volume = float(structure_volume)
        self.is_cumulative = is_cumulative

        # Điểm dữ liệu có thể tính được
        self._min_dose = None
        self._max_dose = None
        self._mean_dose = None
        self._median_dose = None
        self._d_metrics = {}  # Dictionary lưu trữ Dx metrics (liều ở x% thể tích)
        self._v_metrics = {}  # Dictionary lưu trữ Vx metrics (% thể tích nhận ít nhất x Gy)

        # Khởi tạo các chỉ số
        self._initialize_metrics()

    def _initialize_metrics(self):
        """Tính toán các metrics cơ bản từ dữ liệu DVH."""
        if len(self.dose_bins) == 0 or len(self.volume_bins) == 0:
            logger.warning(
                f"Không có dữ liệu để tính toán metrics cho cấu trúc {self.structure_name}"
            )
            return

        # Chuyển sang DVH tích lũy nếu đang là vi phân để tính toán metrics
        if not self.is_cumulative:
            cumulative_dvh = self.to_cumulative()
            dose_bins = cumulative_dvh.dose_bins
            volume_bins = cumulative_dvh.volume_bins
        else:
            dose_bins = self.dose_bins
            volume_bins = self.volume_bins

        # Tính các giá trị cơ bản
        if len(dose_bins) > 0:
            self._min_dose = min(dose_bins)
            self._max_dose = max(dose_bins)

            # Tính mean dose: cần phải chuyển về vi phân để tính chính xác
            if self.is_cumulative:
                diff_dvh = self.to_differential()
                # Mean dose = tổng(liều * phần thể tích) / tổng thể tích
                self._mean_dose = np.sum(
                    diff_dvh.dose_bins * diff_dvh.volume_bins
                ) / np.sum(diff_dvh.volume_bins)
            else:
                self._mean_dose = np.sum(self.dose_bins * self.volume_bins) / np.sum(
                    self.volume_bins
                )

            # Tính median dose (D50)
            self._median_dose = self.get_dose_at_volume(50.0)

            # Tính các giá trị Dx phổ biến
            for x in [1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 98, 99]:
                self._d_metrics[f"D{x}"] = self.get_dose_at_volume(float(x))

            # Tính các giá trị Vx phổ biến (với khoảng 5 Gy)
            max_d = int(self._max_dose) + 1
            for x in range(0, max_d, 5):
                if x > 0:  # Bỏ qua V0
                    self._v_metrics[f"V{x}"] = self.get_volume_at_dose(float(x))

    def get_volume_at_dose(self, dose: float, unit: str = "percent") -> float:
        """
        Lấy thể tích (%) nhận ít nhất một liều cụ thể.

        Args:
            dose: Giá trị liều (Gy)
            unit: Đơn vị kết quả, "percent" hoặc "cc"

        Returns:
            Phần trăm hoặc thể tích tuyệt đối nhận ít nhất dose
        """
        if len(self.dose_bins) == 0 or len(self.volume_bins) == 0:
            return 0.0

        # Đảm bảo đang làm việc với DVH tích lũy
        if not self.is_cumulative:
            cumulative_dvh = self.to_cumulative()
            dose_bins = cumulative_dvh.dose_bins
            volume_bins = cumulative_dvh.volume_bins
        else:
            dose_bins = self.dose_bins
            volume_bins = self.volume_bins

        # Tìm thể tích ở liều dose
        if dose <= np.min(dose_bins):
            volume_value = np.max(volume_bins)
        elif dose >= np.max(dose_bins):
            volume_value = 0.0
        else:
            # Nội suy tuyến tính để tìm giá trị chính xác
            idx = np.searchsorted(dose_bins, dose)
            if idx == 0:
                volume_value = volume_bins[0]
            else:
                # Nội suy tuyến tính
                x0, x1 = dose_bins[idx - 1], dose_bins[idx]
                y0, y1 = volume_bins[idx - 1], volume_bins[idx]
                volume_value = y0 + (dose - x0) * (y1 - y0) / (x1 - x0)

        # Chuyển đổi đơn vị nếu cần
        if unit.lower() == "cc" and volume_value <= 100.0:
            return volume_value * self.structure_volume / 100.0
        elif unit.lower() == "percent" and volume_value > 100.0:
            return volume_value * 100.0 / self.structure_volume

        return volume_value

    def get_dose_at_volume(self, volume_percent: float) -> float:
        """
        Lấy giá trị liều mà một phần trăm thể tích cụ thể nhận được.

        Args:
            volume_percent: Phần trăm thể tích (0-100)

        Returns:
            Giá trị liều (Gy) mà volume_percent% thể tích nhận được
        """
        if len(self.dose_bins) == 0 or len(self.volume_bins) == 0:
            return 0.0

        # Đảm bảo volume_percent trong khoảng hợp lệ
        volume_percent = max(0.0, min(100.0, volume_percent))

        # Đảm bảo đang làm việc với DVH tích lũy
        if not self.is_cumulative:
            cumulative_dvh = self.to_cumulative()
            dose_bins = cumulative_dvh.dose_bins
            volume_bins = cumulative_dvh.volume_bins
        else:
            dose_bins = self.dose_bins
            volume_bins = self.volume_bins

        # Tìm liều ở volume_percent%
        volume_target = volume_percent

        if volume_target >= np.max(volume_bins):
            return np.min(dose_bins)
        elif volume_target <= np.min(volume_bins):
            return np.max(dose_bins)
        else:
            # Nội suy tuyến tính để tìm giá trị chính xác
            idx = np.searchsorted(volume_bins[::-1], volume_target)
            idx = len(volume_bins) - idx - 1
            if idx < 0:
                return dose_bins[0]
            elif idx >= len(volume_bins) - 1:
                return dose_bins[-1]
            else:
                # Nội suy tuyến tính
                y0, y1 = volume_bins[idx], volume_bins[idx + 1]
                x0, x1 = dose_bins[idx], dose_bins[idx + 1]
                dose_value = x0 + (volume_target - y0) * (x1 - x0) / (y1 - y0)
                return dose_value

    def to_differential(self) -> "DVHCurve":
        """
        Chuyển đổi DVH tích lũy thành vi phân.

        Returns:
            DVHCurve vi phân mới
        """
        if not self.is_cumulative:
            return self  # Đã là vi phân

        if len(self.dose_bins) <= 1:
            return DVHCurve(
                self.structure_id,
                self.structure_name,
                self.dose_bins.copy(),
                self.volume_bins.copy(),
                self.structure_volume,
                is_cumulative=False,
            )

        # Tính toán giá trị vi phân
        diff_volume_bins = np.zeros_like(self.volume_bins)

        # Vi phân = độ khác biệt giữa các điểm liên tiếp
        diff_volume_bins[:-1] = np.abs(np.diff(self.volume_bins))
        diff_volume_bins[-1] = self.volume_bins[-2] - self.volume_bins[-1]

        return DVHCurve(
            self.structure_id,
            self.structure_name,
            self.dose_bins.copy(),
            diff_volume_bins,
            self.structure_volume,
            is_cumulative=False,
        )

    def to_cumulative(self) -> "DVHCurve":
        """
        Chuyển đổi DVH vi phân thành tích lũy.

        Returns:
            DVHCurve tích lũy mới
        """
        if self.is_cumulative:
            return self  # Đã là tích lũy

        if len(self.dose_bins) <= 1:
            return DVHCurve(
                self.structure_id,
                self.structure_name,
                self.dose_bins.copy(),
                self.volume_bins.copy(),
                self.structure_volume,
                is_cumulative=True,
            )

        # Tính toán giá trị tích lũy
        cum_volume_bins = np.zeros_like(self.volume_bins)

        # Tích lũy = tổng từ phải sang trái (liều cao đến thấp)
        total_volume = np.sum(self.volume_bins)
        cum_volume_bins = np.zeros_like(self.volume_bins)
        for i in range(len(self.dose_bins)):
            cum_volume_bins[i] = np.sum(self.volume_bins[i:])

        # Chuẩn hóa về phần trăm nếu cần
        if np.max(cum_volume_bins) > 0:
            cum_volume_bins = cum_volume_bins * 100.0 / total_volume

        return DVHCurve(
            self.structure_id,
            self.structure_name,
            self.dose_bins.copy(),
            cum_volume_bins,
            self.structure_volume,
            is_cumulative=True,
        )

    def normalize_to_dose(self, normalization_dose: float) -> "DVHCurve":
        """
        Chuẩn hóa DVH theo một liều cụ thể.

        Args:
            normalization_dose: Liều chuẩn hóa (Gy)

        Returns:
            DVHCurve chuẩn hóa mới
        """
        if normalization_dose <= 0:
            logger.warning(
                f"Liều chuẩn hóa {normalization_dose} không hợp lệ, không thực hiện chuẩn hóa"
            )
            return self

        normalization_factor = 1.0 / normalization_dose
        normalized_dose_bins = (
            self.dose_bins * normalization_factor * 100.0
        )  # Đổi sang %

        return DVHCurve(
            self.structure_id,
            self.structure_name,
            normalized_dose_bins,
            self.volume_bins.copy(),
            self.structure_volume,
            is_cumulative=self.is_cumulative,
        )

    def get_metrics(self) -> Dict[str, float]:
        """
        Lấy tất cả các chỉ số DVH.

        Returns:
            Dictionary chứa tất cả các chỉ số DVH
        """
        metrics = {
            "min_dose": self._min_dose if self._min_dose is not None else 0.0,
            "max_dose": self._max_dose if self._max_dose is not None else 0.0,
            "mean_dose": self._mean_dose if self._mean_dose is not None else 0.0,
            "median_dose": self._median_dose if self._median_dose is not None else 0.0,
        }

        # Thêm các chỉ số Dx
        metrics.update(self._d_metrics)

        # Thêm các chỉ số Vx
        metrics.update(self._v_metrics)

        return metrics

    def plot(self, ax=None, color=None, label=None, style=None, **kwargs):
        """
        Vẽ đường cong DVH.

        Args:
            ax: Trục matplotlib để vẽ (nếu None, sẽ tạo mới)
            color: Màu sắc của đường
            label: Nhãn cho đường
            style: Kiểu đường (line style)
            **kwargs: Các tham số khác cho plt.plot

        Returns:
            Trục matplotlib đã vẽ
        """
        if len(self.dose_bins) == 0 or len(self.volume_bins) == 0:
            logger.warning(
                f"Không có dữ liệu để vẽ DVH cho cấu trúc {self.structure_name}"
            )
            return ax or plt.gca()

        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))

        # Sử dụng nhãn và màu sắc mặc định nếu không được cung cấp
        if label is None:
            label = self.structure_name

        if style is None:
            style = "-"

        # Vẽ đường cong DVH
        (line,) = ax.plot(
            self.dose_bins,
            self.volume_bins,
            linestyle=style,
            color=color,
            label=label,
            **kwargs,
        )

        # Thiết lập nhãn trục
        ax.set_xlabel("Liều (Gy)")
        if self.is_cumulative:
            ax.set_ylabel("Thể tích (%)")
        else:
            ax.set_ylabel("Thể tích vi phân (%)")

        ax.grid(True, linestyle="--", alpha=0.7)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)

        return ax


class DVHData:
    """
    Lớp quản lý và phân tích dữ liệu DVH cho toàn bộ kế hoạch.

    Lớp này lưu trữ nhiều đường cong DVH cho các cấu trúc khác nhau
    và cung cấp các phương thức để phân tích và trực quan hóa dữ liệu DVH.
    """

    def __init__(self, prescription_dose: float = 0.0):
        """
        Khởi tạo đối tượng DVHData.

        Args:
            prescription_dose: Liều kê toa (Gy)
        """
        self.curves: Dict[str, DVHCurve] = {}  # structure_id -> DVHCurve
        self.prescription_dose = prescription_dose
        self.plan_name = ""
        self.metadata: Dict[str, Any] = {}

    def add_curve(self, curve: DVHCurve):
        """
        Thêm một đường cong DVH.

        Args:
            curve: Đối tượng DVHCurve cần thêm
        """
        self.curves[curve.structure_id] = curve

    def get_curve(self, structure_id: str) -> Optional[DVHCurve]:
        """
        Lấy đường cong DVH cho một cấu trúc cụ thể.

        Args:
            structure_id: ID của cấu trúc

        Returns:
            Đối tượng DVHCurve hoặc None nếu không tìm thấy
        """
        return self.curves.get(structure_id)

    def remove_curve(self, structure_id: str):
        """
        Xóa một đường cong DVH.

        Args:
            structure_id: ID của cấu trúc cần xóa
        """
        if structure_id in self.curves:
            del self.curves[structure_id]

    def normalize(self, dose: float):
        """
        Chuẩn hóa tất cả các đường cong DVH với một liều cụ thể.

        Args:
            dose: Liều chuẩn hóa (Gy)
        """
        if dose <= 0:
            logger.warning(
                f"Liều chuẩn hóa {dose} không hợp lệ, không thực hiện chuẩn hóa"
            )
            return

        for structure_id, curve in self.curves.items():
            self.curves[structure_id] = curve.normalize_to_dose(dose)

    def normalize_to_prescription(self):
        """Chuẩn hóa tất cả các đường cong DVH với liều kê toa."""
        if self.prescription_dose <= 0:
            logger.warning("Liều kê toa không được thiết lập hoặc không hợp lệ")
            return

        self.normalize(self.prescription_dose)

    def get_metrics_table(self) -> pd.DataFrame:
        """
        Tạo bảng chứa các chỉ số DVH cho tất cả các cấu trúc.

        Returns:
            DataFrame chứa các chỉ số DVH
        """
        metrics_data = []

        for structure_id, curve in self.curves.items():
            metrics = curve.get_metrics()
            metrics["structure_id"] = structure_id
            metrics["structure_name"] = curve.structure_name
            metrics["volume_cc"] = curve.structure_volume
            metrics_data.append(metrics)

        if not metrics_data:
            return pd.DataFrame()

        df = pd.DataFrame(metrics_data)

        # Sắp xếp các cột
        first_cols = [
            "structure_id",
            "structure_name",
            "volume_cc",
            "min_dose",
            "max_dose",
            "mean_dose",
            "median_dose",
        ]
        d_cols = sorted([col for col in df.columns if col.startswith("D")])
        v_cols = sorted([col for col in df.columns if col.startswith("V")])
        other_cols = [
            col for col in df.columns if col not in first_cols + d_cols + v_cols
        ]

        return df[first_cols + d_cols + v_cols + other_cols]

    def plot_dvhs(
        self,
        structure_ids: Optional[List[str]] = None,
        cumulative: bool = True,
        colors: Optional[Dict[str, str]] = None,
        figsize: Tuple[int, int] = (12, 8),
        **kwargs,
    ):
        """
        Vẽ biểu đồ DVH cho các cấu trúc đã chọn.

        Args:
            structure_ids: Danh sách ID cấu trúc cần vẽ (None = tất cả)
            cumulative: True nếu vẽ DVH tích lũy, False nếu vẽ vi phân
            colors: Dict ánh xạ structure_id -> màu sắc
            figsize: Kích thước hình (width, height)
            **kwargs: Các tham số khác cho plt.plot

        Returns:
            figure, axes đã vẽ
        """
        if not self.curves:
            logger.warning("Không có đường cong DVH để vẽ")
            fig, ax = plt.subplots(figsize=figsize)
            ax.set_xlabel("Liều (Gy)")
            ax.set_ylabel("Thể tích (%)")
            ax.set_title("Biểu đồ DVH")
            return fig, ax

        # Chọn cấu trúc cần vẽ
        if structure_ids is None:
            structure_ids = list(self.curves.keys())

        # Tạo figure và axes
        fig, ax = plt.subplots(figsize=figsize)

        # Colors mặc định
        default_colors = plt.cm.tab10.colors

        # Vẽ từng đường cong
        for i, structure_id in enumerate(structure_ids):
            if structure_id in self.curves:
                curve = self.curves[structure_id]

                # Chuyển đổi sang dạng tích lũy hoặc vi phân nếu cần
                if cumulative and not curve.is_cumulative:
                    curve = curve.to_cumulative()
                elif not cumulative and curve.is_cumulative:
                    curve = curve.to_differential()

                # Chọn màu sắc
                color = None
                if colors and structure_id in colors:
                    color = colors[structure_id]
                else:
                    color = default_colors[i % len(default_colors)]

                # Vẽ đường cong
                curve.plot(ax=ax, color=color, **kwargs)

        # Thiết lập tiêu đề và legend
        title = "DVH Tích lũy" if cumulative else "DVH Vi phân"
        if self.plan_name:
            title += f" - {self.plan_name}"
        ax.set_title(title)
        ax.legend(loc="best")

        # Hiển thị lưới
        ax.grid(True, linestyle="--", alpha=0.7)

        return fig, ax

    def to_dict(self) -> Dict[str, Any]:
        """
        Chuyển đối tượng DVHData thành từ điển.

        Returns:
            Dictionary biểu diễn đối tượng DVHData
        """
        curves_data = {}
        for structure_id, curve in self.curves.items():
            curves_data[structure_id] = {
                "structure_id": curve.structure_id,
                "structure_name": curve.structure_name,
                "dose_bins": curve.dose_bins.tolist(),
                "volume_bins": curve.volume_bins.tolist(),
                "structure_volume": curve.structure_volume,
                "is_cumulative": curve.is_cumulative,
            }

        return {
            "curves": curves_data,
            "prescription_dose": self.prescription_dose,
            "plan_name": self.plan_name,
            "metadata": self.metadata,
        }

    def save_to_json(self, filepath: str):
        """
        Lưu dữ liệu DVH vào file JSON.

        Args:
            filepath: Đường dẫn đến file JSON
        """
        try:
            data = self.to_dict()

            # Đảm bảo thư mục tồn tại
            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            logger.info(f"Đã lưu dữ liệu DVH vào {filepath}")

        except Exception as e:
            logger.error(f"Lỗi khi lưu dữ liệu DVH: {str(e)}")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DVHData":
        """
        Tạo đối tượng DVHData từ từ điển.

        Args:
            data: Dictionary biểu diễn đối tượng DVHData

        Returns:
            Đối tượng DVHData mới
        """
        dvh_data = cls(prescription_dose=data.get("prescription_dose", 0.0))
        dvh_data.plan_name = data.get("plan_name", "")
        dvh_data.metadata = data.get("metadata", {})

        # Tạo các đường cong DVH
        for structure_id, curve_data in data.get("curves", {}).items():
            curve = DVHCurve(
                structure_id=curve_data["structure_id"],
                structure_name=curve_data["structure_name"],
                dose_bins=np.array(curve_data["dose_bins"]),
                volume_bins=np.array(curve_data["volume_bins"]),
                structure_volume=curve_data["structure_volume"],
                is_cumulative=curve_data["is_cumulative"],
            )
            dvh_data.add_curve(curve)

        return dvh_data

    @classmethod
    def load_from_json(cls, filepath: str) -> "DVHData":
        """
        Tải dữ liệu DVH từ file JSON.

        Args:
            filepath: Đường dẫn đến file JSON

        Returns:
            Đối tượng DVHData mới
        """
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            return cls.from_dict(data)

        except Exception as e:
            logger.error(f"Lỗi khi tải dữ liệu DVH: {str(e)}")
            return cls()


def create_dvh_data(prescription_dose: float = 0.0) -> DVHData:
    """
    Tạo đối tượng DVHData mới.

    Args:
        prescription_dose: Liều kê toa (Gy)

    Returns:
        Đối tượng DVHData mới
    """
    return DVHData(prescription_dose)
