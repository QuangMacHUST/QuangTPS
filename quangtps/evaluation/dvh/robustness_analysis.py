#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module phân tích độ bền vững DVH.

Module này cung cấp các công cụ để đánh giá độ bền vững (robustness) của kế hoạch
xạ trị qua biểu đồ DVH, phân tích các biến thể liều có thể xảy ra khi có
các nhiễu loạn về thiết lập hoặc chuyển động của bệnh nhân.
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

# Thiết lập logger
logger = logging.getLogger(__name__)


@dataclass
class DVHRobustnessResult:
    """
    Lưu trữ kết quả phân tích độ bền vững DVH.

    Attributes
    ----------
    structure_name : str
        Tên cấu trúc được phân tích
    nominal_dvh : Dict[str, Any]
        DVH danh nghĩa (nominal)
    min_dvh : Dict[str, Any]
        DVH với giá trị thể tích tối thiểu ở mỗi mức liều
    max_dvh : Dict[str, Any]
        DVH với giá trị thể tích tối đa ở mỗi mức liều
    scenario_dvhs : List[Dict[str, Any]]
        Danh sách các DVH ở các kịch bản khác nhau
    metrics_variation : Dict[str, Dict[str, float]]
        Biến động của các chỉ số (D98, D50, V20, v.v.) qua các kịch bản
    """

    structure_name: str
    nominal_dvh: Dict[str, Any]
    min_dvh: Dict[str, Any] = field(default_factory=dict)
    max_dvh: Dict[str, Any] = field(default_factory=dict)
    scenario_dvhs: List[Dict[str, Any]] = field(default_factory=list)
    metrics_variation: Dict[str, Dict[str, float]] = field(default_factory=dict)


class DVHRobustnessAnalyzer:
    """
    Phân tích độ bền vững của DVH trong nhiều kịch bản khác nhau.
    """

    def __init__(self):
        """Khởi tạo DVHRobustnessAnalyzer."""
        self.results = {}  # Dict[str, DVHRobustnessResult]
        self.scenarios = []  # Danh sách tên kịch bản

    def add_scenario(self, name: str, plan_dvhs: Dict[str, Dict[str, Any]]):
        """
        Thêm kịch bản phân tích độ bền vững.

        Parameters
        ----------
        name : str
            Tên kịch bản (ví dụ: "shift_5mm_x", "rotation_2deg")
        plan_dvhs : Dict[str, Dict[str, Any]]
            Dict chứa DVH của các cấu trúc trong kịch bản này
        """
        try:
            # Lưu tên kịch bản
            if name not in self.scenarios:
                self.scenarios.append(name)

            # Thêm DVH của từng cấu trúc vào kết quả
            for structure_name, dvh_data in plan_dvhs.items():
                if structure_name not in self.results:
                    # Tạo kết quả mới nếu chưa có
                    self.results[structure_name] = DVHRobustnessResult(
                        structure_name=structure_name,
                        nominal_dvh=dvh_data,  # Sử dụng dvh đầu tiên làm nominal
                    )

                # Thêm DVH của kịch bản vào danh sách
                self.results[structure_name].scenario_dvhs.append(dvh_data)

            logger.info(f"Đã thêm kịch bản '{name}' vào phân tích độ bền vững")
        except Exception as e:
            logger.error(f"Lỗi khi thêm kịch bản '{name}': {e}")

    def set_nominal_dvhs(self, nominal_dvhs: Dict[str, Dict[str, Any]]):
        """
        Thiết lập DVH danh nghĩa cho các cấu trúc.

        Parameters
        ----------
        nominal_dvhs : Dict[str, Dict[str, Any]]
            Dict chứa DVH danh nghĩa của các cấu trúc
        """
        try:
            for structure_name, dvh_data in nominal_dvhs.items():
                if structure_name in self.results:
                    # Cập nhật nominal DVH nếu đã có kết quả
                    self.results[structure_name].nominal_dvh = dvh_data
                else:
                    # Tạo kết quả mới với nominal DVH
                    self.results[structure_name] = DVHRobustnessResult(
                        structure_name=structure_name, nominal_dvh=dvh_data
                    )
            logger.info("Đã thiết lập DVH danh nghĩa cho các cấu trúc")
        except Exception as e:
            logger.error(f"Lỗi khi thiết lập DVH danh nghĩa: {e}")

    def analyze(self):
        """
        Phân tích độ bền vững dựa trên các kịch bản đã thêm.

        Tính toán:
        - DVH tối thiểu và tối đa
        - Biến động của các chỉ số như D98, D50, V20, v.v.
        - Độ ổn định của từng chỉ số
        """
        try:
            # Duyệt qua từng cấu trúc
            for structure_name, result in self.results.items():
                if not result.scenario_dvhs:
                    logger.warning(f"Không có dữ liệu kịch bản cho {structure_name}")
                    continue

                # Lấy mảng liều từ nominal DVH
                if not result.nominal_dvh or "dose" not in result.nominal_dvh:
                    logger.warning(
                        f"Không có dữ liệu DVH danh nghĩa hợp lệ cho {structure_name}"
                    )
                    continue

                dose_array = np.array(result.nominal_dvh["dose"])
                if dose_array.size == 0:
                    logger.warning(f"Mảng liều trống cho {structure_name}")
                    continue

                # Thu thập các mảng thể tích từ các kịch bản
                volume_arrays = []
                for dvh in result.scenario_dvhs:
                    # Kiểm tra tính hợp lệ của dữ liệu DVH
                    if not dvh or "volume" not in dvh:
                        logger.warning(
                            f"Bỏ qua kịch bản với dữ liệu DVH không hợp lệ cho {structure_name}"
                        )
                        continue

                    # Chuyển đổi thành mảng numpy nếu chưa phải
                    vol_array = np.array(dvh["volume"])
                    if vol_array.size == 0:
                        logger.warning(
                            f"Bỏ qua kịch bản với mảng thể tích trống cho {structure_name}"
                        )
                        continue

                    # Kiểm tra kích thước mảng phải khớp với dose_array
                    if vol_array.size != dose_array.size:
                        logger.warning(
                            f"Bỏ qua kịch bản với kích thước mảng không khớp: "
                            f"liều {dose_array.size}, thể tích {vol_array.size} cho {structure_name}"
                        )
                        continue

                    volume_arrays.append(vol_array)

                # Kiểm tra nếu không có mảng thể tích hợp lệ nào
                if not volume_arrays:
                    logger.warning(
                        f"Không có dữ liệu thể tích hợp lệ nào cho {structure_name}"
                    )
                    continue

                # Tính toán MIN và MAX DVH
                try:
                    # Chuyển thành mảng numpy 2D để dễ tính toán
                    volume_matrix = np.vstack(volume_arrays)

                    # Tính min/max theo cột (theo mỗi mức liều)
                    min_volumes = np.min(volume_matrix, axis=0)
                    max_volumes = np.max(volume_matrix, axis=0)

                    # Lưu kết quả
                    result.min_dvh = {"dose": dose_array, "volume": min_volumes}

                    result.max_dvh = {"dose": dose_array, "volume": max_volumes}

                    # Phân tích biến động của các chỉ số
                    self._analyze_metrics_variation(result, volume_arrays, dose_array)

                except (ValueError, IndexError, TypeError) as e:
                    logger.error(
                        f"Lỗi khi tính toán min/max DVH cho {structure_name}: {str(e)}"
                    )
                    # Tạo dữ liệu trống để tránh lỗi
                    result.min_dvh = {
                        "dose": dose_array,
                        "volume": np.zeros_like(dose_array),
                    }
                    result.max_dvh = {
                        "dose": dose_array,
                        "volume": np.zeros_like(dose_array),
                    }

        except Exception as e:
            logger.error(
                f"Lỗi không mong đợi trong quá trình phân tích độ bền vững: {str(e)}"
            )
            # Tránh crash, cho phép tiếp tục với dữ liệu đã phân tích được

        return self.results

    def _analyze_metrics_variation(
        self,
        result: DVHRobustnessResult,
        volume_arrays: List[np.ndarray],
        dose_array: np.ndarray,
    ):
        """
        Phân tích biến động của các chỉ số DVH qua các kịch bản.

        Parameters
        ----------
        result : DVHRobustnessResult
            Kết quả phân tích độ bền vững để cập nhật
        volume_arrays : List[np.ndarray]
            Danh sách các mảng thể tích từ các kịch bản
        dose_array : np.ndarray
            Mảng chứa các giá trị liều
        """
        try:
            # Kiểm tra tính hợp lệ của mảng dữ liệu
            if not volume_arrays or len(volume_arrays) == 0:
                logger.warning(
                    f"Không có dữ liệu thể tích để phân tích cho {result.structure_name}"
                )
                return

            if dose_array.size == 0:
                logger.warning(f"Mảng liều trống cho {result.structure_name}")
                return

            # Tạo dictionary lưu trữ biến động các chỉ số
            metrics_variation = {}

            # Chiều dài mảng liều phải phù hợp
            if dose_array.size < 10:
                logger.warning(
                    f"Mảng liều quá nhỏ ({dose_array.size} phần tử) cho {result.structure_name}"
                )
                return

            # Phân tích các chỉ số D98, D95, D90, D50, D5, D2
            # (Liều phủ 98%, 95%, 90%, 50%, 5%, 2% thể tích)
            for percent in [98, 95, 90, 50, 5, 2]:
                try:
                    # Tìm chỉ số gần với percent nhất trong mảng thể tích
                    metric_key = f"D{percent}"
                    metric_values = []

                    # Tính chỉ số D cho mỗi mảng thể tích
                    for vol_array in volume_arrays:
                        # Kiểm tra kích thước mảng volume phải khớp với dose_array
                        if vol_array.size != dose_array.size:
                            logger.warning(
                                f"Bỏ qua kịch bản với kích thước mảng không khớp: "
                                f"liều {dose_array.size}, thể tích {vol_array.size}"
                            )
                            continue

                        # Chuẩn hóa phần trăm nếu thể tích không được biểu diễn dưới dạng phần trăm
                        vol_percent = vol_array
                        if (
                            np.max(vol_array) > 1.5
                        ):  # Nếu max > 1.5, giả sử thể tích biểu diễn dưới dạng %
                            # Tìm chỉ số mảng gần với percent nhất
                            idx = np.argmin(np.abs(vol_percent - percent))
                            if idx < dose_array.size:
                                metric_values.append(dose_array[idx])
                        else:  # Thể tích biểu diễn dưới dạng phần trăm (0-1)
                            # Chuyển đổi percent từ % sang 0-1
                            percent_norm = percent / 100.0
                            idx = np.argmin(np.abs(vol_percent - percent_norm))
                            if idx < dose_array.size:
                                metric_values.append(dose_array[idx])

                    # Nếu không có giá trị nào được tính toán, bỏ qua chỉ số này
                    if not metric_values:
                        logger.debug(
                            f"Không thể tính toán {metric_key} cho {result.structure_name}"
                        )
                        continue

                    # Tính toán thống kê của chỉ số
                    metrics_variation[metric_key] = {
                        "min": float(np.min(metric_values)),
                        "max": float(np.max(metric_values)),
                        "mean": float(np.mean(metric_values)),
                        "std": float(np.std(metric_values)),
                        "values": metric_values,
                    }

                    # Thêm đánh giá biên độ dao động
                    min_val = metrics_variation[metric_key]["min"]
                    max_val = metrics_variation[metric_key]["max"]
                    mean_val = metrics_variation[metric_key]["mean"]

                    # Kiểm tra để tránh chia cho 0
                    if mean_val != 0:
                        # Biên độ dao động tương đối (%)
                        amplitude = ((max_val - min_val) / mean_val) * 100.0
                    else:
                        amplitude = 0.0

                    metrics_variation[metric_key]["amplitude"] = float(amplitude)

                except Exception as e:
                    logger.error(
                        f"Lỗi khi tính toán chỉ số {metric_key} cho {result.structure_name}: {str(e)}"
                    )
                    continue

            # Phân tích chỉ số V20Gy, V10Gy
            # (% thể tích nhận >= 20Gy, 10Gy)
            for dose_level in [20, 10, 5]:
                try:
                    metric_key = f"V{dose_level}Gy"
                    metric_values = []

                    # Tính chỉ số V cho mỗi mảng thể tích
                    for vol_array in volume_arrays:
                        # Kiểm tra kích thước
                        if vol_array.size != dose_array.size:
                            continue

                        # Tìm chỉ số gần với dose_level nhất
                        idx = np.argmin(np.abs(dose_array - dose_level))
                        if idx < vol_array.size:
                            metric_values.append(vol_array[idx])

                    # Nếu không có giá trị nào được tính toán, bỏ qua chỉ số này
                    if not metric_values:
                        logger.debug(
                            f"Không thể tính toán {metric_key} cho {result.structure_name}"
                        )
                        continue

                    # Tính toán thống kê của chỉ số
                    metrics_variation[metric_key] = {
                        "min": float(np.min(metric_values)),
                        "max": float(np.max(metric_values)),
                        "mean": float(np.mean(metric_values)),
                        "std": float(np.std(metric_values)),
                        "values": metric_values,
                    }

                    # Thêm đánh giá biên độ dao động
                    min_val = metrics_variation[metric_key]["min"]
                    max_val = metrics_variation[metric_key]["max"]
                    mean_val = metrics_variation[metric_key]["mean"]

                    # Kiểm tra để tránh chia cho 0
                    if mean_val != 0:
                        # Biên độ dao động tương đối (%)
                        amplitude = ((max_val - min_val) / mean_val) * 100.0
                    else:
                        amplitude = 0.0

                    metrics_variation[metric_key]["amplitude"] = float(amplitude)

                except Exception as e:
                    logger.error(
                        f"Lỗi khi tính toán chỉ số {metric_key} cho {result.structure_name}: {str(e)}"
                    )
                    continue

            # Cập nhật metrics_variation trong result
            result.metrics_variation = metrics_variation

        except Exception as e:
            logger.error(
                f"Lỗi khi phân tích biến động chỉ số cho {result.structure_name}: {str(e)}"
            )
            # Đảm bảo metrics_variation luôn được khởi tạo
            result.metrics_variation = {}

    def get_structure_dvhs(self) -> Dict[str, Dict[str, Any]]:
        """
        Lấy thông tin DVH robustness của các cấu trúc.

        Returns
        -------
        Dict[str, Dict[str, Any]]
            Dict chứa thông tin DVH robustness của các cấu trúc
        """
        result = {}

        for structure_name, robustness_result in self.results.items():
            result[structure_name] = {
                "nominal_dvh": robustness_result.nominal_dvh,
                "min_dvh": robustness_result.min_dvh,
                "max_dvh": robustness_result.max_dvh,
                "metrics_variation": robustness_result.metrics_variation,
            }

        return result

    def get_structure_stability(self) -> Dict[str, float]:
        """
        Tính toán độ ổn định tổng thể cho từng cấu trúc.

        Returns
        -------
        Dict[str, float]
            Dict chứa độ ổn định tổng thể cho từng cấu trúc
        """
        stability_scores = {}

        for structure_name, result in self.results.items():
            # Lấy độ ổn định của từng chỉ số
            metric_stability = []
            for metric_data in result.metrics_variation.values():
                if "stability" in metric_data:
                    metric_stability.append(metric_data["stability"])

            # Tính độ ổn định trung bình của cấu trúc
            if metric_stability:
                stability_scores[structure_name] = np.mean(metric_stability)
            else:
                stability_scores[structure_name] = 0.0

        return stability_scores

    def plot_robustness_bands(self, structure_name: str, ax=None) -> Optional[Figure]:
        """
        Vẽ biểu đồ DVH với dải robustness.

        Parameters
        ----------
        structure_name : str
            Tên cấu trúc cần vẽ
        ax : matplotlib.axes.Axes, optional
            Axes để vẽ, nếu None sẽ tạo axes mới

        Returns
        -------
        Optional[Figure]
            Figure nếu tạo mới, None nếu sử dụng axes có sẵn
        """
        if structure_name not in self.results:
            logger.warning(f"Không tìm thấy dữ liệu robustness cho {structure_name}")
            return None

        result = self.results[structure_name]

        # Kiểm tra dữ liệu DVH
        if (
            not result.nominal_dvh
            or not result.min_dvh
            or not result.max_dvh
            or "dose" not in result.nominal_dvh
            or "volume" not in result.nominal_dvh
        ):
            logger.warning(f"Thiếu dữ liệu DVH robustness cho {structure_name}")
            return None

        # Tạo axes mới nếu cần
        fig = None
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        # Vẽ DVH danh nghĩa
        dose = np.array(result.nominal_dvh["dose"])
        volume = np.array(result.nominal_dvh["volume"])
        ax.plot(dose, volume, "k-", linewidth=2, label=f"{structure_name} (nominal)")

        # Vẽ dải robustness
        min_volume = np.array(result.min_dvh["volume"])
        max_volume = np.array(result.max_dvh["volume"])
        ax.fill_between(dose, min_volume, max_volume, alpha=0.3, color="gray")

        # Thiết lập trục
        ax.set_xlabel("Dose (Gy)")
        ax.set_ylabel("Volume (%)")
        ax.set_title(f"DVH Robustness - {structure_name}")
        ax.grid(True)
        ax.legend()

        # Trả về figure nếu tạo mới
        return fig

    def generate_robustness_report(self, structure_name: str) -> Dict[str, Any]:
        """
        Tạo báo cáo độ bền vững cho một cấu trúc.

        Parameters
        ----------
        structure_name : str
            Tên cấu trúc cần tạo báo cáo

        Returns
        -------
        Dict[str, Any]
            Dict chứa thông tin báo cáo độ bền vững
        """
        if structure_name not in self.results:
            logger.warning(f"Không tìm thấy dữ liệu robustness cho {structure_name}")
            return {}

        result = self.results[structure_name]

        # Khởi tạo báo cáo
        report = {
            "structure_name": structure_name,
            "stability_score": 0.0,
            "metrics": {},
            "recommendation": "",
        }

        # Tính độ ổn định tổng thể
        stability_scores = self.get_structure_stability()
        overall_stability = stability_scores.get(structure_name, 0.0)
        report["stability_score"] = overall_stability

        # Thêm thông tin chi tiết các chỉ số
        report["metrics"] = result.metrics_variation

        # Tạo đánh giá và khuyến nghị
        if overall_stability >= 90:
            report["recommendation"] = "Độ bền vững tuyệt vời, không cần điều chỉnh."
        elif overall_stability >= 80:
            report["recommendation"] = (
                "Độ bền vững tốt, có thể xem xét một số cải thiện nhỏ."
            )
        elif overall_stability >= 70:
            report["recommendation"] = (
                "Độ bền vững khá, nên xem xét tối ưu hóa lại kế hoạch."
            )
        else:
            report["recommendation"] = (
                "Độ bền vững thấp, cần tối ưu hóa lại kế hoạch hoặc xem xét cách tiếp cận khác."
            )

        return report


# Hàm tiện ích để tạo mẫu dữ liệu robustness
def create_sample_robustness_data(
    structure_name: str, is_target: bool = False, num_scenarios: int = 5
) -> DVHRobustnessResult:
    """
    Tạo dữ liệu độ bền vững mẫu cho mục đích demo.

    Parameters
    ----------
    structure_name : str
        Tên cấu trúc
    is_target : bool
        True nếu cấu trúc là mục tiêu (PTV), False nếu là OAR
    num_scenarios : int
        Số lượng kịch bản cần tạo

    Returns
    -------
    DVHRobustnessResult
        Dữ liệu độ bền vững mẫu
    """
    # Tạo mảng liều từ 0 đến 80 Gy
    dose = np.linspace(0, 80, 100)

    # Tạo DVH danh nghĩa
    if is_target:
        # Đường cong mục tiêu với vai phải
        volume_nominal = np.ones(100) * 100
        prescription = 70  # Gy
        idx = np.argmin(np.abs(dose - prescription))
        volume_nominal[idx : idx + 5] = np.linspace(100, 0, 5)
        volume_nominal[idx + 5 :] = 0
    else:
        # Đường cong OAR (hàm mũ)
        volume_nominal = 100 * np.exp(-0.1 * dose)

    # Tạo kết quả
    result = DVHRobustnessResult(
        structure_name=structure_name,
        nominal_dvh={"dose": dose.tolist(), "volume": volume_nominal.tolist()},
    )

    # Tạo các kịch bản với nhiễu
    scenario_dvhs = []
    for i in range(num_scenarios):
        if is_target:
            # Biến thể cho mục tiêu: dịch chuyển vai phải
            noise = np.random.normal(0, 2, 100)
            volume = volume_nominal.copy() + noise
            # Dịch chuyển ngẫu nhiên điểm rơi ±3 Gy
            shift = np.random.uniform(-3, 3)
            new_idx = np.argmin(np.abs(dose - (prescription + shift)))
            volume[new_idx : new_idx + 5] = np.linspace(100, 0, 5)
            volume[new_idx + 5 :] = 0
        else:
            # Biến thể cho OAR: thay đổi hệ số alpha
            alpha_var = 0.1 + np.random.uniform(-0.03, 0.03)
            volume = 100 * np.exp(-alpha_var * dose)
            # Thêm nhiễu
            noise = np.random.normal(0, 3, 100)
            volume = volume + noise

        # Đảm bảo giá trị nằm trong khoảng [0, 100]
        volume = np.clip(volume, 0, 100)

        # Thêm vào danh sách kịch bản
        scenario_dvhs.append({"dose": dose.tolist(), "volume": volume.tolist()})

    # Tính DVH min và max
    volume_arrays = [np.array(dvh["volume"]) for dvh in scenario_dvhs]
    volume_min = np.min(volume_arrays, axis=0)
    volume_max = np.max(volume_arrays, axis=0)

    result.min_dvh = {"dose": dose.tolist(), "volume": volume_min.tolist()}

    result.max_dvh = {"dose": dose.tolist(), "volume": volume_max.tolist()}

    result.scenario_dvhs = scenario_dvhs

    # Tạo dữ liệu biến động chỉ số mẫu
    metrics_variation = (
        {
            "D98": {"min": 65, "max": 72, "mean": 68.5, "std": 2.3, "stability": 96.7},
            "D95": {"min": 67, "max": 73, "mean": 70.1, "std": 2.1, "stability": 97.0},
            "D50": {"min": 72, "max": 76, "mean": 74.2, "std": 1.5, "stability": 98.0},
            "D2": {"min": 74, "max": 78, "mean": 76.5, "std": 1.2, "stability": 98.4},
            "V20Gy": {
                "min": 95,
                "max": 100,
                "mean": 98.7,
                "std": 1.8,
                "stability": 98.2,
            },
        }
        if is_target
        else {
            "D2": {"min": 50, "max": 58, "mean": 54.3, "std": 2.7, "stability": 95.0},
            "D50": {"min": 15, "max": 22, "mean": 18.5, "std": 2.5, "stability": 86.5},
            "V20Gy": {
                "min": 45,
                "max": 62,
                "mean": 53.8,
                "std": 5.7,
                "stability": 89.4,
            },
            "V10Gy": {
                "min": 65,
                "max": 78,
                "mean": 72.3,
                "std": 4.5,
                "stability": 93.8,
            },
        }
    )

    result.metrics_variation = metrics_variation

    return result
