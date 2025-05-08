#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module tính toán chỉ số phù hợp liều xạ trị.

Module này cung cấp các lớp và hàm để tính toán chỉ số phù hợp của liều xạ trị
như Conformity Index (CI), Homogeneity Index (HI), Gradient Index (GI), và các
chỉ số khác dùng trong đánh giá độ phù hợp của kế hoạch xạ trị.
"""

import os
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, NamedTuple
from enum import Enum
import matplotlib.pyplot as plt
from dataclasses import dataclass

from quangtps.dose.dose_grid import DoseGrid
from quangtps.structures.roi import ROI
from quangtps.evaluation.dvh.dvh_calculator import DVHCalculator, DVH

logger = logging.getLogger(__name__)


class ConformityIndexType(Enum):
    """Loại chỉ số phù hợp."""

    RTOG = "rtog"
    PADDICK = "paddick"
    ICRU = "icru"
    LOMAX = "lomax"
    KNOOS = "knoos"
    VAN_T_RIET = "van_t_riet"


@dataclass
class ConformityResult:
    """Kết quả tính toán chỉ số phù hợp."""

    ci_value: float
    """Giá trị chỉ số phù hợp."""

    hi_value: float
    """Giá trị chỉ số đồng nhất."""

    gi_value: float
    """Giá trị chỉ số gradient."""

    target_coverage: float
    """Mức độ bao phủ tổn thương (%)."""

    prescription_isodose_volume: float
    """Thể tích (cc) của isodose tại mức liều chỉ định."""

    target_volume: float
    """Thể tích (cc) của tổn thương."""

    target_min_dose: float
    """Liều tối thiểu trong tổn thương."""

    target_max_dose: float
    """Liều tối đa trong tổn thương."""

    target_mean_dose: float
    """Liều trung bình trong tổn thương."""

    target_median_dose: float
    """Liều trung vị trong tổn thương."""

    #: Dictionary chứa các chỉ số phù hợp khác (nếu có tính)
    other_metrics: Dict[str, float] = None

    def __post_init__(self):
        if self.other_metrics is None:
            self.other_metrics = {}


class ConformityAnalyzer:
    """
    Lớp phân tích chỉ số phù hợp của liều xạ trị.

    Lớp này cung cấp các phương thức để tính toán chỉ số phù hợp và đánh giá
    độ phù hợp của liều xạ trị với tổn thương.
    """

    def __init__(self, dose_grid: DoseGrid, target: ROI):
        """
        Khởi tạo bộ phân tích chỉ số phù hợp.

        Parameters
        ----------
        dose_grid : DoseGrid
            Lưới liều xạ trị
        target : ROI
            Cấu trúc tổn thương cần đánh giá
        """
        self.dose_grid = dose_grid
        self.target = target

        # DVH để tính toán
        self.dvh_calculator = DVHCalculator()
        self.target_dvh = None

        # Kết quả phân tích
        self.result = None

        # Hiệu chuẩn liều theo liều chỉ định
        self.prescription_dose = None

    def calculate_conformity_index(
        self,
        prescription_dose: float,
        ci_type: ConformityIndexType = ConformityIndexType.PADDICK,
        calculate_all: bool = False,
    ) -> ConformityResult:
        """
        Tính toán chỉ số phù hợp.

        Parameters
        ----------
        prescription_dose : float
            Liều chỉ định (Gy hoặc cGy)
        ci_type : ConformityIndexType, optional
            Loại chỉ số phù hợp, mặc định là ConformityIndexType.PADDICK
        calculate_all : bool, optional
            Tính tất cả các loại chỉ số, mặc định là False

        Returns
        -------
        ConformityResult
            Kết quả tính toán chỉ số phù hợp
        """
        self.prescription_dose = prescription_dose

        # Tính DVH cho tổn thương nếu chưa có
        if self.target_dvh is None:
            self.target_dvh = self.dvh_calculator.calculate_dvh_for_structure(
                self.dose_grid, self.target
            )

        # Tổng hợp kết quả
        result = ConformityResult(
            ci_value=0.0,
            hi_value=0.0,
            gi_value=0.0,
            target_coverage=0.0,
            prescription_isodose_volume=0.0,
            target_volume=self.target.volume,
            target_min_dose=0.0,
            target_max_dose=0.0,
            target_mean_dose=0.0,
            target_median_dose=0.0,
            other_metrics={},
        )

        # Cập nhật thông tin liều cho tổn thương
        if self.target_dvh:
            result.target_min_dose = self.target_dvh.get_min_dose()
            result.target_max_dose = self.target_dvh.get_max_dose()
            result.target_mean_dose = self.target_dvh.get_mean_dose()
            result.target_median_dose = self.target_dvh.get_median_dose()

            # Tính độ bao phủ tổn thương
            result.target_coverage = (
                self.target_dvh.get_volume_at_dose(prescription_dose)
                / result.target_volume
                * 100
            )

            # Tính thể tích isodose tại mức liều chỉ định
            result.prescription_isodose_volume = self._calculate_isodose_volume(
                prescription_dose
            )

        # Tính chỉ số phù hợp theo loại chỉ định
        if ci_type == ConformityIndexType.RTOG or calculate_all:
            rtog_ci = self._calculate_rtog_ci(
                prescription_dose,
                result.target_volume,
                result.prescription_isodose_volume,
            )
            if ci_type == ConformityIndexType.RTOG:
                result.ci_value = rtog_ci
            if calculate_all:
                result.other_metrics["ci_rtog"] = rtog_ci

        if ci_type == ConformityIndexType.PADDICK or calculate_all:
            paddick_ci = self._calculate_paddick_ci(
                prescription_dose,
                result.target_volume,
                result.prescription_isodose_volume,
            )
            if ci_type == ConformityIndexType.PADDICK:
                result.ci_value = paddick_ci
            if calculate_all:
                result.other_metrics["ci_paddick"] = paddick_ci

        if ci_type == ConformityIndexType.ICRU or calculate_all:
            icru_ci = self._calculate_icru_ci(
                prescription_dose,
                result.target_volume,
                result.prescription_isodose_volume,
            )
            if ci_type == ConformityIndexType.ICRU:
                result.ci_value = icru_ci
            if calculate_all:
                result.other_metrics["ci_icru"] = icru_ci

        if ci_type == ConformityIndexType.LOMAX or calculate_all:
            lomax_ci = self._calculate_lomax_ci(
                prescription_dose,
                result.target_volume,
                result.prescription_isodose_volume,
            )
            if ci_type == ConformityIndexType.LOMAX:
                result.ci_value = lomax_ci
            if calculate_all:
                result.other_metrics["ci_lomax"] = lomax_ci

        # Tính chỉ số đồng nhất (Homogeneity Index)
        result.hi_value = self._calculate_homogeneity_index(prescription_dose)

        # Tính chỉ số gradient (Gradient Index)
        result.gi_value = self._calculate_gradient_index(prescription_dose)

        # Lưu kết quả
        self.result = result

        return result

    def _calculate_rtog_ci(
        self,
        prescription_dose: float,
        target_volume: float,
        prescription_isodose_volume: float,
    ) -> float:
        """
        Tính chỉ số phù hợp RTOG (Radiation Therapy Oncology Group).

        CI = V_RI / TV

        Trong đó:
        - V_RI: Thể tích nhận ít nhất liều chỉ định
        - TV: Thể tích tổn thương

        Parameters
        ----------
        prescription_dose : float
            Liều chỉ định
        target_volume : float
            Thể tích tổn thương
        prescription_isodose_volume : float
            Thể tích nhận ít nhất liều chỉ định

        Returns
        -------
        float
            Chỉ số phù hợp RTOG
        """
        if target_volume <= 0:
            logger.warning("Thể tích tổn thương bằng 0, không thể tính CI_RTOG")
            return 0.0

        return prescription_isodose_volume / target_volume

    def _calculate_paddick_ci(
        self,
        prescription_dose: float,
        target_volume: float,
        prescription_isodose_volume: float,
    ) -> float:
        """
        Tính chỉ số phù hợp Paddick.

        CI = (TV_RI)^2 / (TV * V_RI)

        Trong đó:
        - TV_RI: Thể tích tổn thương nhận ít nhất liều chỉ định
        - TV: Thể tích tổn thương
        - V_RI: Thể tích nhận ít nhất liều chỉ định

        Parameters
        ----------
        prescription_dose : float
            Liều chỉ định
        target_volume : float
            Thể tích tổn thương
        prescription_isodose_volume : float
            Thể tích nhận ít nhất liều chỉ định

        Returns
        -------
        float
            Chỉ số phù hợp Paddick
        """
        if target_volume <= 0 or prescription_isodose_volume <= 0:
            logger.warning(
                "Thể tích tổn thương hoặc thể tích isodose bằng 0, không thể tính CI_Paddick"
            )
            return 0.0

        # Tính thể tích tổn thương nhận ít nhất liều chỉ định
        target_volume_receiving_prescription = (
            self._calculate_target_volume_receiving_prescription(prescription_dose)
        )

        # Tính chỉ số Paddick
        return (target_volume_receiving_prescription**2) / (
            target_volume * prescription_isodose_volume
        )

    def _calculate_icru_ci(
        self,
        prescription_dose: float,
        target_volume: float,
        prescription_isodose_volume: float,
    ) -> float:
        """
        Tính chỉ số phù hợp ICRU (International Commission on Radiation Units and Measurements).

        CI = V_RI / TV

        Trong đó:
        - V_RI: Thể tích nhận ít nhất liều chỉ định
        - TV: Thể tích tổn thương

        Parameters
        ----------
        prescription_dose : float
            Liều chỉ định
        target_volume : float
            Thể tích tổn thương
        prescription_isodose_volume : float
            Thể tích nhận ít nhất liều chỉ định

        Returns
        -------
        float
            Chỉ số phù hợp ICRU
        """
        # ICRU CI tương đương RTOG CI
        return self._calculate_rtog_ci(
            prescription_dose, target_volume, prescription_isodose_volume
        )

    def _calculate_lomax_ci(
        self,
        prescription_dose: float,
        target_volume: float,
        prescription_isodose_volume: float,
    ) -> float:
        """
        Tính chỉ số phù hợp Lomax.

        CI = (TV_RI / TV) * (TV_RI / V_RI)

        Trong đó:
        - TV_RI: Thể tích tổn thương nhận ít nhất liều chỉ định
        - TV: Thể tích tổn thương
        - V_RI: Thể tích nhận ít nhất liều chỉ định

        Parameters
        ----------
        prescription_dose : float
            Liều chỉ định
        target_volume : float
            Thể tích tổn thương
        prescription_isodose_volume : float
            Thể tích nhận ít nhất liều chỉ định

        Returns
        -------
        float
            Chỉ số phù hợp Lomax
        """
        if target_volume <= 0 or prescription_isodose_volume <= 0:
            logger.warning(
                "Thể tích tổn thương hoặc thể tích isodose bằng 0, không thể tính CI_Lomax"
            )
            return 0.0

        # Tính thể tích tổn thương nhận ít nhất liều chỉ định
        target_volume_receiving_prescription = (
            self._calculate_target_volume_receiving_prescription(prescription_dose)
        )

        # Tính chỉ số Lomax
        return (target_volume_receiving_prescription / target_volume) * (
            target_volume_receiving_prescription / prescription_isodose_volume
        )

    def _calculate_homogeneity_index(self, prescription_dose: float) -> float:
        """
        Tính chỉ số đồng nhất (Homogeneity Index).

        HI = (D2% - D98%) / D50%

        Trong đó:
        - D2%: Liều nhận bởi 2% thể tích tổn thương
        - D98%: Liều nhận bởi 98% thể tích tổn thương
        - D50%: Liều nhận bởi 50% thể tích tổn thương (liều trung vị)

        Parameters
        ----------
        prescription_dose : float
            Liều chỉ định

        Returns
        -------
        float
            Chỉ số đồng nhất
        """
        if self.target_dvh is None:
            logger.warning("Chưa có DVH cho tổn thương, không thể tính HI")
            return 0.0

        d2 = self.target_dvh.get_dose_at_volume_percent(2)
        d98 = self.target_dvh.get_dose_at_volume_percent(98)
        d50 = self.target_dvh.get_dose_at_volume_percent(50)

        if d50 <= 0:
            logger.warning("D50% bằng 0, không thể tính HI")
            return 0.0

        return (d2 - d98) / d50

    def _calculate_gradient_index(self, prescription_dose: float) -> float:
        """
        Tính chỉ số gradient (Gradient Index).

        GI = V50% / V100%

        Trong đó:
        - V50%: Thể tích nhận ít nhất 50% liều chỉ định
        - V100%: Thể tích nhận ít nhất 100% liều chỉ định

        Parameters
        ----------
        prescription_dose : float
            Liều chỉ định

        Returns
        -------
        float
            Chỉ số gradient
        """
        v100 = self._calculate_isodose_volume(prescription_dose)
        v50 = self._calculate_isodose_volume(prescription_dose * 0.5)

        if v100 <= 0:
            logger.warning("V100% bằng 0, không thể tính GI")
            return 0.0

        return v50 / v100

    def _calculate_isodose_volume(self, dose: float) -> float:
        """
        Tính thể tích nhận ít nhất một mức liều cụ thể.

        Parameters
        ----------
        dose : float
            Mức liều cần tính thể tích

        Returns
        -------
        float
            Thể tích (cc) nhận ít nhất mức liều đã cho
        """
        # Tạo mặt nạ cho các voxel có liều >= mức liều chỉ định
        dose_mask = self.dose_grid.data >= dose

        # Tính số voxel có trong mặt nạ
        num_voxels = np.sum(dose_mask)

        # Tính thể tích mỗi voxel (cc)
        voxel_size = np.prod(self.dose_grid.voxel_size) / 1000.0  # mm^3 -> cc

        # Tính tổng thể tích
        return num_voxels * voxel_size

    def _calculate_target_volume_receiving_prescription(
        self, prescription_dose: float
    ) -> float:
        """
        Tính thể tích tổn thương nhận ít nhất liều chỉ định.

        Parameters
        ----------
        prescription_dose : float
            Liều chỉ định

        Returns
        -------
        float
            Thể tích (cc) tổn thương nhận ít nhất liều chỉ định
        """
        if self.target_dvh is None:
            logger.warning(
                "Chưa có DVH cho tổn thương, không thể tính thể tích tổn thương nhận ít nhất liều chỉ định"
            )
            return 0.0

        # Sử dụng DVH để tính thể tích tổn thương nhận ít nhất liều chỉ định
        volume_percent = self.target_dvh.get_volume_at_dose(prescription_dose)

        # Chuyển đổi phần trăm thành thể tích thực
        return (volume_percent / 100.0) * self.target.volume

    def plot_conformity_analysis(
        self, output_path: Optional[str] = None
    ) -> Optional[plt.Figure]:
        """
        Vẽ biểu đồ phân tích chỉ số phù hợp.

        Parameters
        ----------
        output_path : Optional[str], optional
            Đường dẫn lưu biểu đồ, mặc định là None (không lưu)

        Returns
        -------
        Optional[plt.Figure]
            Đối tượng Figure nếu vẽ thành công, None nếu không
        """
        if self.result is None:
            logger.warning("Chưa có kết quả phân tích, không thể vẽ biểu đồ")
            return None

        try:
            fig, axs = plt.subplots(2, 2, figsize=(12, 10))

            # 1. Biểu đồ DVH cho tổn thương
            if self.target_dvh:
                ax = axs[0, 0]
                self.target_dvh.plot(ax=ax)

                # Đánh dấu liều chỉ định
                if self.prescription_dose:
                    ax.axvline(
                        x=self.prescription_dose,
                        color="r",
                        linestyle="--",
                        label=f"Liều chỉ định: {self.prescription_dose:.2f} Gy",
                    )

                ax.set_title("DVH của tổn thương")
                ax.legend()

            # 2. Biểu đồ chỉ số phù hợp
            ax = axs[0, 1]
            metrics = {
                "CI": self.result.ci_value,
                "HI": self.result.hi_value,
                "GI": self.result.gi_value,
                "Coverage (%)": self.result.target_coverage / 100,
            }

            # Thêm các chỉ số khác nếu có
            if self.result.other_metrics:
                for key, value in self.result.other_metrics.items():
                    if key.startswith("ci_"):
                        metrics[key.upper().replace("CI_", "CI ")] = value

            bars = ax.bar(metrics.keys(), metrics.values())

            # Thêm nhãn giá trị
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.01,
                    f"{height:.2f}",
                    ha="center",
                    va="bottom",
                )

            ax.set_ylim(0, max(metrics.values()) * 1.2)
            ax.set_title("Chỉ số phù hợp")

            # 3. Biểu đồ thể tích
            ax = axs[1, 0]
            volumes = {
                "Tổn thương": self.result.target_volume,
                "Isodose chỉ định": self.result.prescription_isodose_volume,
                "Tổn thương bao phủ": self._calculate_target_volume_receiving_prescription(
                    self.prescription_dose
                ),
            }

            bars = ax.bar(volumes.keys(), volumes.values())

            # Thêm nhãn giá trị
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.1,
                    f"{height:.2f} cc",
                    ha="center",
                    va="bottom",
                )

            ax.set_title("Thông tin thể tích")

            # 4. Biểu đồ thông tin liều
            ax = axs[1, 1]
            doses = {
                "Min": self.result.target_min_dose,
                "Mean": self.result.target_mean_dose,
                "Median": self.result.target_median_dose,
                "Max": self.result.target_max_dose,
                "Rx": self.prescription_dose,
            }

            bars = ax.bar(doses.keys(), doses.values())

            # Thêm nhãn giá trị
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.1,
                    f"{height:.2f} Gy",
                    ha="center",
                    va="bottom",
                )

            ax.set_title("Thông tin liều tổn thương")

            # Thông tin tổng quan
            plt.suptitle(f"Phân tích chỉ số phù hợp: {self.target.name}", fontsize=16)
            plt.tight_layout()
            plt.subplots_adjust(top=0.9)

            # Lưu hình nếu có đường dẫn
            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches="tight")
                logger.info(
                    f"Đã lưu biểu đồ phân tích chỉ số phù hợp vào: {output_path}"
                )

            return fig

        except Exception as e:
            logger.error(f"Lỗi khi vẽ biểu đồ phân tích chỉ số phù hợp: {e}")
            import traceback

            traceback.print_exc()
            return None

    def generate_conformity_report(self, output_path: str) -> bool:
        """
        Tạo báo cáo phân tích chỉ số phù hợp.

        Parameters
        ----------
        output_path : str
            Đường dẫn lưu báo cáo

        Returns
        -------
        bool
            True nếu tạo báo cáo thành công, False nếu không
        """
        if self.result is None:
            logger.warning("Chưa có kết quả phân tích, không thể tạo báo cáo")
            return False

        try:
            # Tạo biểu đồ phân tích
            chart_path = output_path.replace(".txt", ".png")
            if output_path.endswith(".txt"):
                self.plot_conformity_analysis(chart_path)

            # Tạo báo cáo văn bản
            with open(output_path, "w") as f:
                f.write(f"PHÂN TÍCH CHỈ SỐ PHÙ HỢP: {self.target.name}\n")
                f.write(f"=========================================\n\n")

                f.write(f"THÔNG TIN CHUNG:\n")
                f.write(f"- Tổn thương: {self.target.name}\n")
                f.write(f"- Thể tích tổn thương: {self.result.target_volume:.2f} cc\n")
                f.write(f"- Liều chỉ định: {self.prescription_dose:.2f} Gy\n\n")

                f.write(f"CHỈ SỐ PHÙ HỢP:\n")
                f.write(f"- Conformity Index (CI): {self.result.ci_value:.4f}\n")
                f.write(f"- Homogeneity Index (HI): {self.result.hi_value:.4f}\n")
                f.write(f"- Gradient Index (GI): {self.result.gi_value:.4f}\n")
                f.write(f"- Coverage: {self.result.target_coverage:.2f}%\n\n")

                if self.result.other_metrics:
                    f.write(f"CHỈ SỐ PHÙ HỢP KHÁC:\n")
                    for key, value in self.result.other_metrics.items():
                        f.write(f"- {key.upper()}: {value:.4f}\n")
                    f.write("\n")

                f.write(f"THÔNG TIN LIỀU:\n")
                f.write(f"- Liều tối thiểu: {self.result.target_min_dose:.2f} Gy\n")
                f.write(f"- Liều trung bình: {self.result.target_mean_dose:.2f} Gy\n")
                f.write(f"- Liều trung vị: {self.result.target_median_dose:.2f} Gy\n")
                f.write(f"- Liều tối đa: {self.result.target_max_dose:.2f} Gy\n\n")

                f.write(f"THÔNG TIN THỂ TÍCH:\n")
                f.write(f"- Thể tích tổn thương: {self.result.target_volume:.2f} cc\n")
                f.write(
                    f"- Thể tích isodose chỉ định: {self.result.prescription_isodose_volume:.2f} cc\n"
                )
                target_volume_rx = self._calculate_target_volume_receiving_prescription(
                    self.prescription_dose
                )
                f.write(
                    f"- Thể tích tổn thương nhận ít nhất liều chỉ định: {target_volume_rx:.2f} cc\n\n"
                )

                f.write(f"ĐÁNH GIÁ:\n")
                # CI evaluation
                f.write(f"- Conformity Index (CI): ")
                if self.result.ci_value >= 0.9:
                    f.write("Tốt (>= 0.9)\n")
                elif self.result.ci_value >= 0.8:
                    f.write("Chấp nhận được (0.8 - 0.9)\n")
                else:
                    f.write("Cần cải thiện (< 0.8)\n")

                # HI evaluation
                f.write(f"- Homogeneity Index (HI): ")
                if self.result.hi_value <= 0.1:
                    f.write("Tốt (<= 0.1)\n")
                elif self.result.hi_value <= 0.2:
                    f.write("Chấp nhận được (0.1 - 0.2)\n")
                else:
                    f.write("Cần cải thiện (> 0.2)\n")

                # GI evaluation
                f.write(f"- Gradient Index (GI): ")
                if self.result.gi_value <= 3.0:
                    f.write("Tốt (<= 3.0)\n")
                elif self.result.gi_value <= 4.0:
                    f.write("Chấp nhận được (3.0 - 4.0)\n")
                else:
                    f.write("Cần cải thiện (> 4.0)\n")

                # Coverage evaluation
                f.write(f"- Coverage: ")
                if self.result.target_coverage >= 95:
                    f.write("Tốt (>= 95%)\n")
                elif self.result.target_coverage >= 90:
                    f.write("Chấp nhận được (90% - 95%)\n")
                else:
                    f.write("Cần cải thiện (< 90%)\n")

                # Tham khảo hình ảnh nếu có
                if output_path.endswith(".txt"):
                    f.write(f"\nBiểu đồ phân tích: {os.path.basename(chart_path)}\n")

            logger.info(f"Đã tạo báo cáo phân tích chỉ số phù hợp: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Lỗi khi tạo báo cáo phân tích chỉ số phù hợp: {e}")
            import traceback

            traceback.print_exc()
            return False
