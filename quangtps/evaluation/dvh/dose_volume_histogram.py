#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module Dose Volume Histogram (DVH) cho QuangTPS.

Module này cung cấp các chức năng tính toán, phân tích và hiển thị
Dose Volume Histogram (DVH) cho đánh giá kế hoạch điều trị xạ trị.
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json

logger = logging.getLogger(__name__)

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.warning("Matplotlib not available - DVH plotting will be limited")


class DVHType(str, Enum):
    """Loại DVH."""

    CUMULATIVE = "cumulative"  # DVH tích lũy
    DIFFERENTIAL = "differential"  # DVH vi phân


class VolumeUnits(str, Enum):
    """Đơn vị thể tích cho DVH."""

    PERCENT = "percent"  # Phần trăm
    CC = "cc"  # Centimet khối (cm³)


@dataclass
class DVHPoint:
    """Điểm dữ liệu trong DVH."""

    dose: float  # Liều (Gy)
    volume: float  # Thể tích (% hoặc cc)

    def __str__(self) -> str:
        return f"D={self.dose:.2f}Gy, V={self.volume:.2f}"


@dataclass
class DVHMetrics:
    """Các chỉ số đánh giá từ DVH."""

    # Chỉ số liều tại thể tích cụ thể (Dx)
    d95: Optional[float] = None  # Liều tại 95% thể tích
    d50: Optional[float] = None  # Liều trung vị
    d2: Optional[float] = None  # Liều gần max (D2%)
    d98: Optional[float] = None  # Liều gần min (D98%)
    d_mean: Optional[float] = None  # Liều trung bình
    d_max: Optional[float] = None  # Liều tối đa
    d_min: Optional[float] = None  # Liều tối thiểu

    # Chỉ số thể tích tại liều cụ thể (Vx)
    v_5gy: Optional[float] = None  # Thể tích nhận 5Gy
    v_10gy: Optional[float] = None  # Thể tích nhận 10Gy
    v_20gy: Optional[float] = None  # Thể tích nhận 20Gy
    v_30gy: Optional[float] = None  # Thể tích nhận 30Gy
    v_50gy: Optional[float] = None  # Thể tích nhận 50Gy

    # Chỉ số chất lượng
    conformity_index: Optional[float] = None  # Chỉ số đồng dạng
    homogeneity_index: Optional[float] = None  # Chỉ số đồng nhất
    coverage: Optional[float] = None  # Độ phủ

    # Thể tích cấu trúc
    total_volume: Optional[float] = None  # Tổng thể tích

    def to_dict(self) -> Dict[str, Any]:
        """Chuyển đổi thành dictionary."""
        return {
            "dose_metrics": {
                "D95": self.d95,
                "D50": self.d50,
                "D2": self.d2,
                "D98": self.d98,
                "D_mean": self.d_mean,
                "D_max": self.d_max,
                "D_min": self.d_min,
            },
            "volume_metrics": {
                "V5Gy": self.v_5gy,
                "V10Gy": self.v_10gy,
                "V20Gy": self.v_20gy,
                "V30Gy": self.v_30gy,
                "V50Gy": self.v_50gy,
            },
            "quality_metrics": {
                "CI": self.conformity_index,
                "HI": self.homogeneity_index,
                "Coverage": self.coverage,
            },
            "total_volume": self.total_volume,
        }


@dataclass
class DVHData:
    """Dữ liệu DVH cho một cấu trúc."""

    structure_name: str
    structure_type: str = "UNKNOWN"  # TARGET, OAR, OTHER
    dvh_type: DVHType = DVHType.CUMULATIVE
    volume_units: VolumeUnits = VolumeUnits.PERCENT

    # Dữ liệu DVH
    dose_bins: np.ndarray = field(default_factory=lambda: np.array([]))
    volume_data: np.ndarray = field(default_factory=lambda: np.array([]))

    # Metadata
    structure_volume: float = 0.0  # Thể tích cấu trúc (cc)
    prescription_dose: Optional[float] = None  # Liều kê đơn (Gy)

    # Các chỉ số đã tính
    metrics: Optional[DVHMetrics] = None

    # Styling cho hiển thị
    color: str = "#FF0000"  # Màu mặc định
    line_style: str = "-"  # Kiểu đường
    line_width: float = 2.0  # Độ dày đường

    def __post_init__(self):
        """Xử lý sau khi khởi tạo."""
        if len(self.dose_bins) > 0 and len(self.volume_data) > 0:
            self.calculate_metrics()

    def calculate_metrics(self) -> DVHMetrics:
        """Tính toán các chỉ số DVH."""
        if len(self.dose_bins) == 0 or len(self.volume_data) == 0:
            logger.warning(
                f"Không có dữ liệu DVH để tính toán cho {self.structure_name}"
            )
            return DVHMetrics()

        metrics = DVHMetrics()

        try:
            # Chỉ số liều
            metrics.d_mean = np.average(
                self.dose_bins, weights=self._get_differential_volume()
            )

            # Tìm indices có volume > 0
            valid_indices = self.volume_data > 0
            if np.any(valid_indices):
                metrics.d_max = np.max(self.dose_bins[valid_indices])
                metrics.d_min = np.min(self.dose_bins[valid_indices])
            else:
                metrics.d_max = (
                    np.max(self.dose_bins) if len(self.dose_bins) > 0 else 0.0
                )
                metrics.d_min = (
                    np.min(self.dose_bins) if len(self.dose_bins) > 0 else 0.0
                )

            # Dx metrics (liều tại x% thể tích)
            metrics.d95 = self._get_dose_at_volume(95.0)
            metrics.d50 = self._get_dose_at_volume(50.0)
            metrics.d2 = self._get_dose_at_volume(2.0)
            metrics.d98 = self._get_dose_at_volume(98.0)

            # Vx metrics (thể tích tại x Gy)
            metrics.v_5gy = self._get_volume_at_dose(5.0)
            metrics.v_10gy = self._get_volume_at_dose(10.0)
            metrics.v_20gy = self._get_volume_at_dose(20.0)
            metrics.v_30gy = self._get_volume_at_dose(30.0)
            metrics.v_50gy = self._get_volume_at_dose(50.0)

            # Chỉ số chất lượng
            if self.prescription_dose:
                metrics.coverage = self._get_volume_at_dose(
                    self.prescription_dose * 0.95
                )
                if metrics.d2 and metrics.d98:
                    metrics.homogeneity_index = (
                        metrics.d2 - metrics.d98
                    ) / self.prescription_dose

            metrics.total_volume = self.structure_volume

            self.metrics = metrics

        except Exception as e:
            logger.error(
                f"Lỗi khi tính toán DVH metrics cho {self.structure_name}: {str(e)}"
            )
            self.metrics = DVHMetrics()

        return self.metrics

    def _get_differential_volume(self) -> np.ndarray:
        """Tính thể tích vi phân từ DVH tích lũy."""
        if self.dvh_type == DVHType.DIFFERENTIAL:
            return self.volume_data
        else:
            # Chuyển đổi từ cumulative sang differential
            diff_volume = np.zeros_like(self.volume_data)
            diff_volume[:-1] = self.volume_data[:-1] - self.volume_data[1:]
            diff_volume[-1] = self.volume_data[-1]
            return diff_volume

    def _get_dose_at_volume(self, volume_percent: float) -> Optional[float]:
        """Lấy liều tại phần trăm thể tích cụ thể."""
        if len(self.dose_bins) == 0:
            return None

        try:
            # Tìm chỉ số gần nhất với volume_percent
            idx = np.argmin(np.abs(self.volume_data - volume_percent))
            return float(self.dose_bins[idx])
        except Exception:
            return None

    def _get_volume_at_dose(self, dose_gy: float) -> Optional[float]:
        """Lấy thể tích tại liều cụ thể."""
        if len(self.dose_bins) == 0:
            return None

        try:
            # Interpolation để tìm thể tích tại liều cụ thể
            return float(np.interp(dose_gy, self.dose_bins, self.volume_data))
        except Exception:
            return None

    def to_dict(self) -> Dict[str, Any]:
        """Chuyển đổi thành dictionary."""
        data = {
            "structure_name": self.structure_name,
            "structure_type": self.structure_type,
            "dvh_type": self.dvh_type.value,
            "volume_units": self.volume_units.value,
            "dose_bins": self.dose_bins.tolist() if len(self.dose_bins) > 0 else [],
            "volume_data": self.volume_data.tolist()
            if len(self.volume_data) > 0
            else [],
            "structure_volume": self.structure_volume,
            "prescription_dose": self.prescription_dose,
            "color": self.color,
            "line_style": self.line_style,
            "line_width": self.line_width,
        }

        if self.metrics:
            data["metrics"] = self.metrics.to_dict()

        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DVHData":
        """Tạo từ dictionary."""
        dvh = cls(
            structure_name=data["structure_name"],
            structure_type=data.get("structure_type", "UNKNOWN"),
            dvh_type=DVHType(data.get("dvh_type", "cumulative")),
            volume_units=VolumeUnits(data.get("volume_units", "percent")),
            dose_bins=np.array(data.get("dose_bins", [])),
            volume_data=np.array(data.get("volume_data", [])),
            structure_volume=data.get("structure_volume", 0.0),
            prescription_dose=data.get("prescription_dose"),
            color=data.get("color", "#FF0000"),
            line_style=data.get("line_style", "-"),
            line_width=data.get("line_width", 2.0),
        )

        # Tính lại metrics
        if len(dvh.dose_bins) > 0 and len(dvh.volume_data) > 0:
            dvh.calculate_metrics()

        return dvh


class DVHCalculator:
    """Lớp tính toán DVH từ dữ liệu dose và structure."""

    def __init__(self, dose_bin_width: float = 0.1):
        """
        Khởi tạo DVH calculator.

        Parameters:
            dose_bin_width: Độ rộng bin cho liều (Gy)
        """
        self.dose_bin_width = dose_bin_width

    def calculate_dvh(
        self,
        dose_grid: np.ndarray,
        structure_mask: np.ndarray,
        structure_name: str,
        structure_type: str = "UNKNOWN",
        prescription_dose: Optional[float] = None,
        volume_units: VolumeUnits = VolumeUnits.PERCENT,
        voxel_volume: float = 1.0,
    ) -> DVHData:
        """
        Tính toán DVH từ dose grid và structure mask.

        Parameters:
            dose_grid: Ma trận liều 3D
            structure_mask: Mask cấu trúc 3D (boolean)
            structure_name: Tên cấu trúc
            structure_type: Loại cấu trúc (TARGET, OAR, OTHER)
            prescription_dose: Liều kê đơn
            volume_units: Đơn vị thể tích
            voxel_volume: Thể tích voxel (cc)

        Returns:
            DVHData: Dữ liệu DVH đã tính toán
        """
        try:
            # Kiểm tra input
            if dose_grid.shape != structure_mask.shape:
                raise ValueError("Kích thước dose_grid và structure_mask không khớp")

            # Lấy liều trong cấu trúc
            structure_doses = dose_grid[structure_mask]

            if len(structure_doses) == 0:
                logger.warning(f"Không có voxel nào trong cấu trúc {structure_name}")
                return DVHData(
                    structure_name=structure_name, structure_type=structure_type
                )

            # Tính thể tích cấu trúc
            structure_volume = np.sum(structure_mask) * voxel_volume

            # Tạo bins liều
            max_dose = np.max(structure_doses)
            dose_bins = np.arange(
                0, max_dose + self.dose_bin_width, self.dose_bin_width
            )

            # Tính histogram và DVH tích lũy
            hist, _ = np.histogram(structure_doses, bins=dose_bins)
            cumulative_volume = np.cumsum(hist[::-1])[::-1]  # Reverse cumsum

            # Chuyển đổi đơn vị thể tích
            if volume_units == VolumeUnits.PERCENT:
                total_voxels = len(structure_doses)
                volume_data = (cumulative_volume / total_voxels) * 100.0
            else:  # CC
                volume_data = cumulative_volume * voxel_volume

            # Lấy giá trị trung điểm của bins
            dose_centers = (dose_bins[:-1] + dose_bins[1:]) / 2

            # Tạo DVHData
            dvh_data = DVHData(
                structure_name=structure_name,
                structure_type=structure_type,
                dose_bins=dose_centers,
                volume_data=volume_data,
                structure_volume=structure_volume,
                prescription_dose=prescription_dose,
                volume_units=volume_units,
            )

            logger.info(f"Đã tính toán DVH cho cấu trúc {structure_name}")
            return dvh_data

        except Exception as e:
            logger.error(f"Lỗi khi tính toán DVH cho {structure_name}: {str(e)}")
            return DVHData(structure_name=structure_name, structure_type=structure_type)


class DVHAnalyzer:
    """Lớp phân tích và so sánh DVH."""

    @staticmethod
    def compare_dvh(dvh1: DVHData, dvh2: DVHData) -> Dict[str, Any]:
        """So sánh hai DVH."""
        if not dvh1.metrics or not dvh2.metrics:
            return {"error": "Thiếu metrics để so sánh"}

        comparison = {
            "structure_1": dvh1.structure_name,
            "structure_2": dvh2.structure_name,
            "dose_differences": {},
            "volume_differences": {},
        }

        # So sánh các chỉ số liều
        dose_metrics = ["d95", "d50", "d2", "d98", "d_mean", "d_max"]
        for metric in dose_metrics:
            val1 = getattr(dvh1.metrics, metric)
            val2 = getattr(dvh2.metrics, metric)
            if val1 is not None and val2 is not None:
                comparison["dose_differences"][metric] = val2 - val1

        # So sánh các chỉ số thể tích
        volume_metrics = ["v_5gy", "v_10gy", "v_20gy", "v_30gy", "v_50gy"]
        for metric in volume_metrics:
            val1 = getattr(dvh1.metrics, metric)
            val2 = getattr(dvh2.metrics, metric)
            if val1 is not None and val2 is not None:
                comparison["volume_differences"][metric] = val2 - val1

        return comparison

    @staticmethod
    def evaluate_plan_quality(dvh_list: List[DVHData]) -> Dict[str, Any]:
        """Đánh giá chất lượng kế hoạch từ danh sách DVH."""
        evaluation = {
            "targets": [],
            "oars": [],
            "overall_score": 0.0,
            "warnings": [],
            "recommendations": [],
        }

        target_score = 0.0
        oar_score = 0.0

        for dvh in dvh_list:
            if not dvh.metrics:
                continue

            if dvh.structure_type == "TARGET":
                # Đánh giá target
                target_eval = DVHAnalyzer._evaluate_target(dvh)
                evaluation["targets"].append(target_eval)
                target_score += target_eval.get("score", 0.0)

            elif dvh.structure_type == "OAR":
                # Đánh giá OAR
                oar_eval = DVHAnalyzer._evaluate_oar(dvh)
                evaluation["oars"].append(oar_eval)
                oar_score += oar_eval.get("score", 0.0)

        # Tính điểm tổng
        num_targets = len(evaluation["targets"])
        num_oars = len(evaluation["oars"])

        if num_targets > 0:
            target_score /= num_targets
        if num_oars > 0:
            oar_score /= num_oars

        evaluation["overall_score"] = target_score * 0.6 + oar_score * 0.4

        return evaluation

    @staticmethod
    def _evaluate_target(dvh: DVHData) -> Dict[str, Any]:
        """Đánh giá target từ DVH."""
        evaluation = {
            "structure_name": dvh.structure_name,
            "score": 0.0,
            "criteria": [],
        }

        if not dvh.metrics:
            return evaluation

        score = 0.0

        # Đánh giá coverage
        if dvh.metrics.coverage is not None:
            if dvh.metrics.coverage >= 95.0:
                score += 40.0
                evaluation["criteria"].append("Độ phủ tốt (≥95%)")
            elif dvh.metrics.coverage >= 90.0:
                score += 30.0
                evaluation["criteria"].append("Độ phủ chấp nhận được (≥90%)")
            else:
                evaluation["criteria"].append("Độ phủ kém (<90%)")

        # Đánh giá homogeneity
        if dvh.metrics.homogeneity_index is not None:
            if dvh.metrics.homogeneity_index <= 0.1:
                score += 30.0
                evaluation["criteria"].append("Đồng nhất tốt (HI≤0.1)")
            elif dvh.metrics.homogeneity_index <= 0.15:
                score += 20.0
                evaluation["criteria"].append("Đồng nhất chấp nhận được (HI≤0.15)")
            else:
                evaluation["criteria"].append("Đồng nhất kém (HI>0.15)")

        # Đánh giá conformity
        if dvh.metrics.conformity_index is not None:
            if 0.9 <= dvh.metrics.conformity_index <= 1.1:
                score += 30.0
                evaluation["criteria"].append("Đồng dạng tốt (0.9≤CI≤1.1)")
            elif 0.8 <= dvh.metrics.conformity_index <= 1.2:
                score += 20.0
                evaluation["criteria"].append("Đồng dạng chấp nhận được")
            else:
                evaluation["criteria"].append("Đồng dạng kém")

        evaluation["score"] = score
        return evaluation

    @staticmethod
    def _evaluate_oar(dvh: DVHData) -> Dict[str, Any]:
        """Đánh giá OAR từ DVH."""
        evaluation = {
            "structure_name": dvh.structure_name,
            "score": 0.0,
            "criteria": [],
        }

        if not dvh.metrics:
            return evaluation

        # Đánh giá dựa trên các constraint thông dụng
        score = 100.0  # Bắt đầu với điểm tối đa

        # Giảm điểm nếu vượt constraint
        constraints = DVHAnalyzer._get_oar_constraints(dvh.structure_name.lower())

        for constraint in constraints:
            if constraint["type"] == "max_dose":
                if dvh.metrics.d_max and dvh.metrics.d_max > constraint["value"]:
                    score -= 20.0
                    evaluation["criteria"].append(
                        f"Vượt liều tối đa ({constraint['value']}Gy)"
                    )
            elif constraint["type"] == "volume_dose":
                volume = getattr(dvh.metrics, f"v_{constraint['dose']}gy", None)
                if volume and volume > constraint["value"]:
                    score -= 15.0
                    evaluation["criteria"].append(
                        f"Vượt V{constraint['dose']}Gy ({constraint['value']}%)"
                    )

        evaluation["score"] = max(0.0, score)
        return evaluation

    @staticmethod
    def _get_oar_constraints(structure_name: str) -> List[Dict[str, Any]]:
        """Lấy constraint cho OAR."""
        constraints_db = {
            "lung": [
                {"type": "max_dose", "value": 20.0},
                {"type": "volume_dose", "dose": 20, "value": 30.0},
            ],
            "heart": [
                {"type": "max_dose", "value": 40.0},
                {"type": "volume_dose", "dose": 30, "value": 50.0},
            ],
            "spinal_cord": [{"type": "max_dose", "value": 45.0}],
            "liver": [
                {"type": "max_dose", "value": 30.0},
                {"type": "volume_dose", "dose": 30, "value": 30.0},
            ],
        }

        for organ, constraints in constraints_db.items():
            if organ in structure_name:
                return constraints

        return []


if HAS_MATPLOTLIB:

    class DVHPlotter:
        """Lớp vẽ DVH."""

        @staticmethod
        def plot_dvh_list(
            dvh_list: List[DVHData],
            title: str = "Dose Volume Histogram",
            figure_size: Tuple[int, int] = (10, 6),
            show_grid: bool = True,
            show_legend: bool = True,
        ) -> Figure:
            """Vẽ danh sách DVH."""
            fig, ax = plt.subplots(figsize=figure_size)

            for dvh in dvh_list:
                if len(dvh.dose_bins) > 0 and len(dvh.volume_data) > 0:
                    ax.plot(
                        dvh.dose_bins,
                        dvh.volume_data,
                        color=dvh.color,
                        linestyle=dvh.line_style,
                        linewidth=dvh.line_width,
                        label=dvh.structure_name,
                    )

            ax.set_xlabel("Dose (Gy)")
            ax.set_ylabel(
                "Volume (%)"
                if dvh_list and dvh_list[0].volume_units == VolumeUnits.PERCENT
                else "Volume (cc)"
            )
            ax.set_title(title)

            if show_grid:
                ax.grid(True, alpha=0.3)

            if show_legend and dvh_list:
                ax.legend()

            plt.tight_layout()
            return fig

        @staticmethod
        def plot_dvh_comparison(
            dvh1: DVHData, dvh2: DVHData, title: str = "DVH Comparison"
        ) -> Figure:
            """So sánh hai DVH."""
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # Vẽ DVH
            ax1.plot(
                dvh1.dose_bins,
                dvh1.volume_data,
                color=dvh1.color,
                label=dvh1.structure_name,
                linewidth=2,
            )
            ax1.plot(
                dvh2.dose_bins,
                dvh2.volume_data,
                color=dvh2.color,
                label=dvh2.structure_name,
                linewidth=2,
                linestyle="--",
            )

            ax1.set_xlabel("Dose (Gy)")
            ax1.set_ylabel("Volume (%)")
            ax1.set_title("DVH Curves")
            ax1.grid(True, alpha=0.3)
            ax1.legend()

            # Vẽ metrics comparison
            if dvh1.metrics and dvh2.metrics:
                metrics = ["D95", "D50", "D2", "V20Gy", "V50Gy"]
                values1 = [
                    dvh1.metrics.d95 or 0,
                    dvh1.metrics.d50 or 0,
                    dvh1.metrics.d2 or 0,
                    dvh1.metrics.v_20gy or 0,
                    dvh1.metrics.v_50gy or 0,
                ]
                values2 = [
                    dvh2.metrics.d95 or 0,
                    dvh2.metrics.d50 or 0,
                    dvh2.metrics.d2 or 0,
                    dvh2.metrics.v_20gy or 0,
                    dvh2.metrics.v_50gy or 0,
                ]

                x = np.arange(len(metrics))
                width = 0.35

                ax2.bar(
                    x - width / 2,
                    values1,
                    width,
                    label=dvh1.structure_name,
                    color=dvh1.color,
                    alpha=0.7,
                )
                ax2.bar(
                    x + width / 2,
                    values2,
                    width,
                    label=dvh2.structure_name,
                    color=dvh2.color,
                    alpha=0.7,
                )

                ax2.set_xlabel("Metrics")
                ax2.set_ylabel("Value")
                ax2.set_title("Metrics Comparison")
                ax2.set_xticks(x)
                ax2.set_xticklabels(metrics)
                ax2.legend()
                ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            return fig

else:

    class DVHPlotter:
        """Dummy plotter khi không có matplotlib."""

        @staticmethod
        def plot_dvh_list(*args, **kwargs):
            logger.error("Matplotlib không có sẵn - không thể vẽ DVH")
            return None

        @staticmethod
        def plot_dvh_comparison(*args, **kwargs):
            logger.error("Matplotlib không có sẵn - không thể vẽ DVH")
            return None
