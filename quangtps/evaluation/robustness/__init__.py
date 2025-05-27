#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module đánh giá độ bền vững (robustness) cho kế hoạch xạ trị.

Độ bền vững của một kế hoạch xạ trị thể hiện khả năng duy trì phân bố liều
đạt yêu cầu khi có các biến thiên/nhiễu, bao gồm:
- Sai số thiết lập (setup error)
- Biến động nội tại (organ motion)
- Bất định về phạm vi (range uncertainty, đặc biệt với xạ trị proton)
- Thay đổi giải phẫu (anatomical changes)

Module này cung cấp các công cụ để đánh giá độ bền vững của kế hoạch
thông qua phân tích độ nhạy và mô phỏng các kịch bản khác nhau.
"""

import logging
from typing import Dict, List, Any, Optional, Union, Tuple

# Export các module con
from quangtps.evaluation.robustness.uncertainty import (
    setup_error_scenarios,
    range_uncertainty_scenarios,
    fractionation_effect,
)
from quangtps.evaluation.robustness.analysis import (
    calculate_robustness_metrics,
    robustness_dvh_bands,
    find_worst_case_scenario,
)
from quangtps.evaluation.robustness.visualization import (
    plot_robustness_metrics,
    plot_robustness_bands,
)

__all__ = [
    "setup_error_scenarios",
    "range_uncertainty_scenarios",
    "fractionation_effect",
    "calculate_robustness_metrics",
    "robustness_dvh_bands",
    "find_worst_case_scenario",
    "plot_robustness_metrics",
    "plot_robustness_bands",
    "RobustnessAnalyzer",
    "RobustnessResult",
]

logger = logging.getLogger(__name__)


class RobustnessResult:
    """
    Kết quả phân tích độ bền vững cho một kế hoạch xạ trị.

    Đóng gói toàn bộ dữ liệu từ phân tích độ bền vững bao gồm:
    - DVH bands cho từng cấu trúc
    - Dải thông số cho từng chỉ số lâm sàng (min, max, trung bình)
    - Phân tích độ phủ mục tiêu trong các kịch bản khác nhau
    - Phân tích không gian của sự biến thiên liều
    """

    def __init__(self):
        self.structures = {}
        self.dvh_bands = {}
        self.metrics = {}
        self.coverage_data = {}
        self.spatial_data = {}
        self.scenarios = []
        self.scenario_count = 0

    def add_structure_dvh(
        self,
        structure_name: str,
        nominal_dvh: Tuple[List[float], List[float]],
        scenario_dvhs: List[Tuple[List[float], List[float]]],
    ):
        """Thêm dữ liệu DVH cho một cấu trúc cụ thể."""
        self.structures[structure_name] = {
            "nominal": nominal_dvh,
            "scenarios": scenario_dvhs,
        }

    def add_metric(
        self,
        metric_name: str,
        structure_name: str,
        nominal_value: float,
        scenario_values: List[float],
    ):
        """Thêm kết quả chỉ số đánh giá cho một cấu trúc cụ thể."""
        if structure_name not in self.metrics:
            self.metrics[structure_name] = {}

        self.metrics[structure_name][metric_name] = {
            "nominal": nominal_value,
            "min": min(scenario_values),
            "max": max(scenario_values),
            "mean": sum(scenario_values) / len(scenario_values),
            "std": self._calculate_std(scenario_values),
            "values": scenario_values,
        }

    def add_coverage_data(self, structure_name: str, coverage_data: Dict):
        """Thêm dữ liệu độ phủ cho cấu trúc mục tiêu."""
        self.coverage_data[structure_name] = coverage_data

    def add_spatial_data(self, data: Dict):
        """Thêm dữ liệu phân tích không gian."""
        self.spatial_data = data

    def set_scenarios(self, scenarios: List[Dict]):
        """Thiết lập danh sách kịch bản đã được phân tích."""
        self.scenarios = scenarios
        self.scenario_count = len(scenarios)

    def get_structures(self):
        """Lấy danh sách các cấu trúc đã được phân tích."""
        return list(self.structures.keys())

    def get_structure_dvhs(self, structure_name: str):
        """Lấy dữ liệu DVH cho một cấu trúc cụ thể."""
        if structure_name not in self.structures:
            return None
        return self.structures[structure_name]

    def get_evaluation_metrics(self):
        """Lấy tất cả các chỉ số đánh giá."""
        return self.metrics

    def get_target_coverage_data(self):
        """Lấy dữ liệu độ phủ mục tiêu."""
        return self.coverage_data

    def get_spatial_analysis_data(self):
        """Lấy dữ liệu phân tích không gian."""
        return self.spatial_data

    def _calculate_std(self, values: List[float]) -> float:
        """Tính độ lệch chuẩn của một chuỗi giá trị."""
        if len(values) <= 1:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance**0.5


class RobustnessAnalyzer:
    """
    Bộ phân tích độ bền vững cho kế hoạch xạ trị.

    Cung cấp các phương thức để phân tích độ bền vững của kế hoạch xạ trị
    thông qua mô phỏng các kịch bản khác nhau và đánh giá sự biến thiên
    của phân phối liều và các chỉ số lâm sàng.
    """

    def __init__(
        self,
        setup_uncertainty: float = 3.0,  # mm
        range_uncertainty: float = 3.0,  # %
        num_scenarios: int = 7,
    ):
        """
        Khởi tạo bộ phân tích độ bền vững.

        Args:
            setup_uncertainty: Độ không chắc chắn thiết lập bệnh nhân (mm)
            range_uncertainty: Độ không chắc chắn phạm vi (% cho proton/ion)
            num_scenarios: Số lượng kịch bản phân tích
        """
        self.setup_uncertainty = setup_uncertainty
        self.range_uncertainty = range_uncertainty
        self.num_scenarios = num_scenarios
        self.logger = logging.getLogger(__name__)

    def analyze_plan_robustness(self, plan, dose_calculator=None):
        """
        Phân tích độ bền vững của một kế hoạch xạ trị.

        Args:
            plan: Kế hoạch xạ trị cần phân tích
            dose_calculator: Bộ tính toán liều (nếu cần tính lại)

        Returns:
            RobustnessResult: Kết quả phân tích độ bền vững
        """
        result = RobustnessResult()

        try:
            # Tạo các kịch bản
            scenarios = self._generate_scenarios()
            result.set_scenarios(scenarios)

            # Phân tích từng kịch bản
            for scenario in scenarios:
                self._analyze_scenario(plan, scenario, result, dose_calculator)

            # Phân tích tổng hợp
            self._analyze_overall_robustness(result)

            return result

        except Exception as e:
            self.logger.error(f"Lỗi khi phân tích độ bền vững: {str(e)}")
            import traceback

            self.logger.error(traceback.format_exc())
            return None

    def _generate_scenarios(self):
        """Tạo các kịch bản phân tích dựa trên tham số hiện tại."""
        scenarios = []

        # Thêm kịch bản danh nghĩa (nominal)
        scenarios.append(
            {
                "id": "nominal",
                "name": "Nominal",
                "setup_shift": (0.0, 0.0, 0.0),
                "range_scale": 1.0,
            }
        )

        # Thêm các kịch bản dịch chuyển thiết lập (setup shifts)
        setup_shifts = setup_error_scenarios(
            uncertainty=self.setup_uncertainty, num_scenarios=self.num_scenarios
        )

        for i, shift in enumerate(setup_shifts):
            scenarios.append(
                {
                    "id": f"setup_{i + 1}",
                    "name": f"Setup Shift {i + 1}",
                    "setup_shift": shift,
                    "range_scale": 1.0,
                }
            )

        # Thêm các kịch bản về độ không chắc chắn phạm vi (range uncertainties)
        range_scenarios = range_uncertainty_scenarios(
            uncertainty_percent=self.range_uncertainty, num_scenarios=2
        )

        for i, scale in enumerate(range_scenarios):
            scenarios.append(
                {
                    "id": f"range_{i + 1}",
                    "name": f"Range {'+' if scale > 1 else '-'}{abs(scale - 1) * 100:.1f}%",
                    "setup_shift": (0.0, 0.0, 0.0),
                    "range_scale": scale,
                }
            )

        return scenarios

    def _analyze_scenario(self, plan, scenario, result, dose_calculator):
        """
        Phân tích một kịch bản cụ thể.

        Trong phiên bản thực tế, điều này sẽ:
        1. Áp dụng dịch chuyển thiết lập và/hoặc thay đổi phạm vi
        2. Tính toán lại phân phối liều
        3. Đánh giá DVH và các chỉ số lâm sàng
        """
        # Mô phỏng: Trong triển khai thực tế, sẽ tính toán lại liều thực sự
        # Đây chỉ là mã mẫu để minh họa cấu trúc
        pass

    def _analyze_overall_robustness(self, result):
        """
        Phân tích tổng hợp độ bền vững từ tất cả các kịch bản.

        Tính toán các thông số như:
        - Độ chênh lệch tối đa của các chỉ số lâm sàng
        - Băng DVH (độ rộng dải DVH)
        - Xác định các vùng không ổn định nhất
        """
        # Mã mẫu để minh họa cấu trúc
        pass

    def find_worst_case_scenario(
        self, robustness_results, target_structures=None, metric="d95"
    ):
        """
        Tìm kịch bản tệ nhất dựa trên các chỉ số.

        Parameters
        ----------
        robustness_results : Dict
            Kết quả phân tích độ bền vững
        target_structures : List[str], optional
            Danh sách cấu trúc mục tiêu
        metric : str, optional
            Chỉ số để đánh giá, default "d95"

        Returns
        -------
        Dict
            Thông tin kịch bản tệ nhất
        """
        from .analysis import find_worst_case_scenario

        return find_worst_case_scenario(robustness_results, target_structures, metric)
