#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plan Checker Module
================

Module này cung cấp chức năng kiểm tra kế hoạch điều trị tự động,
tương tự như chức năng Plan Checker trong Eclipse của Varian.

Chức năng chính:
- Kiểm tra toàn diện kế hoạch điều trị dựa trên các protocol lâm sàng
- Phân tích mức độ đạt được các mục tiêu lâm sàng
- Cảnh báo về các vấn đề tiềm ẩn và đề xuất cải thiện
- Tạo báo cáo đánh giá kế hoạch chi tiết
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Any
from collections import defaultdict
from datetime import datetime

from quangtps.core.logging import get_logger
from quangtps.evaluation.clinical_goals import (
    ClinicalGoal,
    ClinicalGoalCollection,
    ClinicalGoalTemplate,
    ClinicalGoalManager,
    GoalType,
    GoalOperator,
    GoalPriority,
    GoalResult,
)
from quangtps.evaluation.dvh.dvh_calculator import DVHCalculator
from quangtps.evaluation.plan_evaluation import PlanEvaluation
from quangtps.evaluation.metrics import (
    calculate_conformity_index,
    calculate_homogeneity_index,
    calculate_gradient_index,
    calculate_paddick_ci,
)
from quangtps.core.patient.plan import Plan
from quangtps.core.patient.course import Course

# Khởi tạo logger
logger = get_logger(__name__)


class PlanCheckerResult:
    """
    Kết quả kiểm tra kế hoạch điều trị.

    Attributes:
        structure_name: Tên cấu trúc
        goal_description: Mô tả mục tiêu lâm sàng
        result: Kết quả đánh giá (Passed/Failed/Warning)
        target_value: Giá trị mục tiêu
        achieved_value: Giá trị đạt được
        deviation: Độ lệch giữa giá trị mục tiêu và đạt được
        priority: Mức độ ưu tiên của mục tiêu
    """

    def __init__(
        self,
        structure_name: str,
        goal_description: str,
        result: GoalResult,
        target_value: float,
        achieved_value: float,
        deviation: float,
        priority: GoalPriority,
    ):
        self.structure_name = structure_name
        self.goal_description = goal_description
        self.result = result
        self.target_value = target_value
        self.achieved_value = achieved_value
        self.deviation = deviation
        self.priority = priority

    def __str__(self) -> str:
        """Trả về biểu diễn chuỗi của kết quả kiểm tra."""
        result_str = {
            GoalResult.PASSED: "Đạt",
            GoalResult.FAILED: "Không đạt",
            GoalResult.WARNING: "Cảnh báo",
            GoalResult.NOT_APPLICABLE: "Không áp dụng",
        }[self.result]

        priority_str = {
            GoalPriority.MINOR: "Thấp",
            GoalPriority.MAJOR: "Trung bình",
            GoalPriority.CRITICAL: "Cao",
        }[self.priority]

        return (
            f"{self.structure_name}: {self.goal_description} - {result_str} "
            f"(Mục tiêu: {self.target_value:.2f}, Đạt được: {self.achieved_value:.2f}, "
            f"Chênh lệch: {self.deviation:.2f}%, Ưu tiên: {priority_str})"
        )


class PlanChecker:
    """
    Plan Checker kiểm tra và đánh giá kế hoạch điều trị dựa trên các mục tiêu lâm sàng.

    Chức năng chính:
    1. Kiểm tra kế hoạch dựa trên protocol lâm sàng
    2. Phân tích độ phù hợp của kế hoạch với mục tiêu lâm sàng
    3. Tạo báo cáo đánh giá chi tiết
    4. Cảnh báo về các vấn đề tiềm ẩn trong kế hoạch
    """

    def __init__(self):
        """Khởi tạo Plan Checker."""
        # Các thành phần chính
        self.plan = None
        self.dvh_calculator = None
        self.goal_manager = ClinicalGoalManager()
        self.plan_evaluation = PlanEvaluation()

        # Kết quả kiểm tra
        self.results = []
        self.passed_count = 0
        self.failed_count = 0
        self.warning_count = 0
        self.not_applicable_count = 0

        # Cài đặt và tùy chọn
        self.warning_threshold = 5.0  # % chênh lệch để cảnh báo

        logger.info("Plan Checker đã được khởi tạo")

    def set_plan(self, plan: Plan):
        """
        Thiết lập kế hoạch cần kiểm tra.

        Parameters:
            plan: Kế hoạch điều trị
        """
        self.plan = plan
        self.dvh_calculator = plan.get_dvh_calculator()
        self.plan_evaluation.set_dose_calculator(plan.get_dose_calculator())
        logger.info(f"Đã thiết lập kế hoạch: {plan.name}")

    def load_protocol(self, protocol_name: str) -> ClinicalGoalCollection:
        """
        Tải protocol lâm sàng từ tên.

        Parameters:
            protocol_name: Tên protocol

        Returns:
            Bộ sưu tập các mục tiêu lâm sàng
        """
        template = self.goal_manager.get_template_by_name(protocol_name)
        if template:
            goals = template.apply_to_plan(self.plan)
            logger.info(f"Đã tải protocol: {protocol_name}")
            return goals
        else:
            logger.warning(f"Không tìm thấy protocol: {protocol_name}")
            return ClinicalGoalCollection(name=protocol_name)

    def load_protocol_from_file(self, file_path: str) -> ClinicalGoalCollection:
        """
        Tải protocol lâm sàng từ file.

        Parameters:
            file_path: Đường dẫn đến file protocol

        Returns:
            Bộ sưu tập các mục tiêu lâm sàng
        """
        try:
            goals = ClinicalGoalCollection.load_from_file(file_path)
            logger.info(f"Đã tải protocol từ file: {file_path}")
            return goals
        except Exception as e:
            logger.error(f"Lỗi khi tải protocol từ file {file_path}: {str(e)}")
            return ClinicalGoalCollection(name="Custom Protocol")

    def check_plan(
        self, protocol: Union[str, ClinicalGoalCollection]
    ) -> List[PlanCheckerResult]:
        """
        Kiểm tra kế hoạch dựa trên protocol lâm sàng.

        Parameters:
            protocol: Tên protocol hoặc bộ sưu tập mục tiêu lâm sàng

        Returns:
            Danh sách kết quả kiểm tra
        """
        if not self.plan:
            logger.error("Chưa thiết lập kế hoạch")
            return []

        # Xác định protocol
        if isinstance(protocol, str):
            goals = self.load_protocol(protocol)
        else:
            goals = protocol

        # Đặt lại bộ đếm kết quả
        self.results = []
        self.passed_count = 0
        self.failed_count = 0
        self.warning_count = 0
        self.not_applicable_count = 0

        # Đánh giá các mục tiêu
        goals.evaluate(self.dvh_calculator)

        # Phân tích kết quả
        for goal in goals:
            if goal.result == GoalResult.NOT_APPLICABLE:
                self.not_applicable_count += 1
                continue

            # Tính toán độ lệch phần trăm
            if goal.value != 0:
                deviation = (goal.achieved_value - goal.value) / goal.value * 100
            else:
                deviation = float("inf") if goal.achieved_value > 0 else 0

            # Tạo kết quả kiểm tra
            result = PlanCheckerResult(
                structure_name=goal.structure_name,
                goal_description=str(goal),
                result=goal.result,
                target_value=goal.value,
                achieved_value=goal.achieved_value,
                deviation=deviation,
                priority=goal.priority,
            )

            # Cập nhật bộ đếm
            if goal.result == GoalResult.PASSED:
                self.passed_count += 1
            elif goal.result == GoalResult.FAILED:
                self.failed_count += 1
            elif goal.result == GoalResult.WARNING:
                self.warning_count += 1

            self.results.append(result)

        # Sắp xếp kết quả theo: 1) Không đạt, 2) Cảnh báo, 3) Đạt
        # Và sau đó theo mức độ ưu tiên
        def sort_key(result):
            result_order = {
                GoalResult.FAILED: 0,
                GoalResult.WARNING: 1,
                GoalResult.PASSED: 2,
                GoalResult.NOT_APPLICABLE: 3,
            }
            priority_order = {
                GoalPriority.CRITICAL: 0,
                GoalPriority.MAJOR: 1,
                GoalPriority.MINOR: 2,
            }
            return (result_order[result.result], priority_order[result.priority])

        self.results.sort(key=sort_key)

        logger.info(f"Đã kiểm tra kế hoạch với {len(self.results)} tiêu chí")
        logger.info(
            f"Đạt: {self.passed_count}, Không đạt: {self.failed_count}, "
            f"Cảnh báo: {self.warning_count}, Không áp dụng: {self.not_applicable_count}"
        )

        return self.results

    def generate_report(self, file_path: str = None, format: str = "html") -> str:
        """
        Tạo báo cáo kiểm tra kế hoạch.

        Parameters:
            file_path: Đường dẫn lưu báo cáo
            format: Định dạng báo cáo ("html", "pdf", "txt")

        Returns:
            Đường dẫn đến file báo cáo
        """
        # Kiểm tra kết quả
        if not self.results:
            logger.error("Chưa có kết quả kiểm tra")
            return None

        # Xây dựng báo cáo HTML
        if format.lower() == "html":
            report = self._generate_html_report()
        elif format.lower() == "txt":
            report = self._generate_text_report()
        else:
            logger.error(f"Định dạng {format} chưa được hỗ trợ")
            return None

        # Lưu báo cáo nếu có đường dẫn
        if file_path:
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(report)
                logger.info(f"Đã lưu báo cáo tại: {file_path}")
                return file_path
            except Exception as e:
                logger.error(f"Lỗi khi lưu báo cáo: {str(e)}")
                return None

        return report

    def _generate_html_report(self) -> str:
        """Tạo báo cáo dạng HTML."""
        if not self.plan:
            return "<h1>Lỗi: Chưa thiết lập kế hoạch</h1>"

        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Báo cáo kiểm tra kế hoạch: {self.plan.name}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            color: #333;
            line-height: 1.6;
        }}
        h1, h2, h3 {{
            color: #2c3e50;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin-bottom: 20px;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }}
        th {{
            background-color: #f2f2f2;
        }}
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        .passed {{
            background-color: #d4edda;
            color: #155724;
        }}
        .warning {{
            background-color: #fff3cd;
            color: #856404;
        }}
        .failed {{
            background-color: #f8d7da;
            color: #721c24;
        }}
        .summary {{
            display: flex;
            justify-content: space-around;
            margin: 20px 0;
        }}
        .summary-box {{
            padding: 10px;
            border-radius: 5px;
            text-align: center;
            width: 20%;
        }}
        .passed-box {{
            background-color: #d4edda;
        }}
        .warning-box {{
            background-color: #fff3cd;
        }}
        .failed-box {{
            background-color: #f8d7da;
        }}
        .not-applicable-box {{
            background-color: #e2e3e5;
        }}
    </style>
</head>
<body>
    <h1>Báo cáo kiểm tra kế hoạch điều trị</h1>

    <h2>Thông tin kế hoạch</h2>
    <table>
        <tr>
            <th>Tên kế hoạch</th>
            <td>{self.plan.name}</td>
        </tr>
        <tr>
            <th>Liều điều trị</th>
            <td>{self.plan.prescription.dose} Gy / {self.plan.prescription.fractions} phân liều</td>
        </tr>
        <tr>
            <th>Ngày kiểm tra</th>
            <td>{datetime.now().strftime("%d/%m/%Y %H:%M")}</td>
        </tr>
    </table>

    <h2>Tổng quan kết quả</h2>
    <div class="summary">
        <div class="summary-box passed-box">
            <h3>Đạt</h3>
            <p>{self.passed_count}</p>
        </div>
        <div class="summary-box warning-box">
            <h3>Cảnh báo</h3>
            <p>{self.warning_count}</p>
        </div>
        <div class="summary-box failed-box">
            <h3>Không đạt</h3>
            <p>{self.failed_count}</p>
        </div>
        <div class="summary-box not-applicable-box">
            <h3>Không áp dụng</h3>
            <p>{self.not_applicable_count}</p>
        </div>
    </div>

    <h2>Chi tiết kết quả kiểm tra</h2>
    <table>
        <tr>
            <th>Cấu trúc</th>
            <th>Mục tiêu</th>
            <th>Giá trị mục tiêu</th>
            <th>Giá trị đạt được</th>
            <th>Chênh lệch (%)</th>
            <th>Ưu tiên</th>
            <th>Kết quả</th>
        </tr>
"""

        # Thêm các kết quả vào bảng
        for result in self.results:
            # Xác định lớp CSS cho kết quả
            result_class = {
                GoalResult.PASSED: "passed",
                GoalResult.FAILED: "failed",
                GoalResult.WARNING: "warning",
                GoalResult.NOT_APPLICABLE: "",
            }[result.result]

            # Chuỗi kết quả
            result_str = {
                GoalResult.PASSED: "Đạt",
                GoalResult.FAILED: "Không đạt",
                GoalResult.WARNING: "Cảnh báo",
                GoalResult.NOT_APPLICABLE: "Không áp dụng",
            }[result.result]

            # Mức độ ưu tiên
            priority_str = {
                GoalPriority.MINOR: "Thấp",
                GoalPriority.MAJOR: "Trung bình",
                GoalPriority.CRITICAL: "Cao",
            }[result.priority]

            html += f"""
        <tr class="{result_class}">
            <td>{result.structure_name}</td>
            <td>{result.goal_description}</td>
            <td>{result.target_value:.2f}</td>
            <td>{result.achieved_value:.2f}</td>
            <td>{result.deviation:.2f}%</td>
            <td>{priority_str}</td>
            <td>{result_str}</td>
        </tr>"""

        html += """
    </table>

    <h2>Đề xuất cải thiện</h2>
"""

        # Thêm đề xuất cải thiện cho các mục tiêu không đạt
        failed_results = [r for r in self.results if r.result == GoalResult.FAILED]
        if failed_results:
            html += "<ul>"
            for result in failed_results:
                html += f"""
        <li>
            <b>{result.structure_name}</b>: {result.goal_description} -
            Giá trị hiện tại {result.achieved_value:.2f}, cần đạt {result.target_value:.2f}.
            Cần cải thiện {abs(result.deviation):.2f}%.
        </li>"""
            html += "</ul>"
        else:
            html += "<p>Không có đề xuất cải thiện cần thiết.</p>"

        html += """
</body>
</html>
"""
        return html

    def _generate_text_report(self) -> str:
        """Tạo báo cáo dạng văn bản thuần túy."""
        if not self.plan:
            return "Lỗi: Chưa thiết lập kế hoạch"

        report = [
            "=== BÁO CÁO KIỂM TRA KẾ HOẠCH ĐIỀU TRỊ ===",
            "",
            "== THÔNG TIN KẾ HOẠCH ==",
            f"Tên kế hoạch: {self.plan.name}",
            f"Liều điều trị: {self.plan.prescription.dose} Gy / {self.plan.prescription.fractions} phân liều",
            f"Ngày kiểm tra: {datetime.now().strftime('%d/%m/%Y %H:%M')}",
            "",
            "== TỔNG QUAN KẾT QUẢ ==",
            f"Đạt: {self.passed_count}",
            f"Cảnh báo: {self.warning_count}",
            f"Không đạt: {self.failed_count}",
            f"Không áp dụng: {self.not_applicable_count}",
            "",
            "== CHI TIẾT KẾT QUẢ ==",
        ]

        # Thêm các kết quả chi tiết
        for result in self.results:
            result_str = {
                GoalResult.PASSED: "Đạt",
                GoalResult.FAILED: "Không đạt",
                GoalResult.WARNING: "Cảnh báo",
                GoalResult.NOT_APPLICABLE: "Không áp dụng",
            }[result.result]

            priority_str = {
                GoalPriority.MINOR: "Thấp",
                GoalPriority.MAJOR: "Trung bình",
                GoalPriority.CRITICAL: "Cao",
            }[result.priority]

            report.append(
                f"{result.structure_name}: {result.goal_description} - {result_str}\n"
                f"  Mục tiêu: {result.target_value:.2f}, Đạt được: {result.achieved_value:.2f}\n"
                f"  Chênh lệch: {result.deviation:.2f}%, Ưu tiên: {priority_str}"
            )

        report.append("")
        report.append("== ĐỀ XUẤT CẢI THIỆN ==")

        # Thêm đề xuất cải thiện cho các mục tiêu không đạt
        failed_results = [r for r in self.results if r.result == GoalResult.FAILED]
        if failed_results:
            for result in failed_results:
                report.append(
                    f"- {result.structure_name}: {result.goal_description}\n"
                    f"  Giá trị hiện tại {result.achieved_value:.2f}, cần đạt {result.target_value:.2f}.\n"
                    f"  Cần cải thiện {abs(result.deviation):.2f}%."
                )
        else:
            report.append("Không có đề xuất cải thiện cần thiết.")

        return "\n".join(report)

    def check_for_warnings(self) -> List[str]:
        """
        Kiểm tra các vấn đề tiềm ẩn trong kế hoạch.

        Returns:
            Danh sách các cảnh báo
        """
        if not self.plan:
            return ["Chưa thiết lập kế hoạch"]

        warnings = []

        # Kiểm tra độ phủ PTV
        target_coverage = self.plan_evaluation.get_target_coverage()
        if target_coverage < 95:
            warnings.append(f"Độ phủ PTV thấp: {target_coverage:.2f}% < 95%")

        # Kiểm tra liều tối đa toàn cục
        max_dose = self.plan_evaluation.get_global_max_dose()
        prescription = self.plan.prescription.dose
        if max_dose > 1.1 * prescription:
            warnings.append(
                f"Liều tối đa toàn cục cao: {max_dose:.2f} Gy > {1.1 * prescription:.2f} Gy (110% liều điều trị)"
            )

        # Kiểm tra độ đồng nhất liều
        hi = self.plan_evaluation.get_homogeneity_index()
        if hi > 0.15:
            warnings.append(f"Độ đồng nhất liều thấp: HI = {hi:.2f} > 0.15")

        # Kiểm tra độ phù hợp
        ci = self.plan_evaluation.get_conformity_index()
        if ci < 0.8:
            warnings.append(f"Độ phù hợp thấp: CI = {ci:.2f} < 0.8")

        # Cảnh báo từ kết quả
        for result in self.results:
            if result.result == GoalResult.WARNING:
                warnings.append(
                    f"Gần ngưỡng: {result.structure_name} - {result.goal_description} "
                    f"(Mục tiêu: {result.target_value:.2f}, Đạt được: {result.achieved_value:.2f})"
                )

        return warnings

    def get_summary(self) -> Dict[str, Any]:
        """
        Lấy tóm tắt kết quả kiểm tra.

        Returns:
            Dict chứa tóm tắt kết quả
        """
        return {
            "total": len(self.results),
            "passed": self.passed_count,
            "failed": self.failed_count,
            "warning": self.warning_count,
            "not_applicable": self.not_applicable_count,
            "pass_rate": self.passed_count
            / max(1, self.passed_count + self.failed_count)
            * 100,
        }

    def plot_goal_results(self, file_path: str = None) -> plt.Figure:
        """
        Vẽ biểu đồ kết quả đánh giá mục tiêu.

        Parameters:
            file_path: Đường dẫn lưu biểu đồ

        Returns:
            Figure của matplotlib
        """
        # Tạo biểu đồ tròn tóm tắt kết quả
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        # 1. Biểu đồ tròn tổng quan
        labels = ["Đạt", "Cảnh báo", "Không đạt", "Không áp dụng"]
        sizes = [
            self.passed_count,
            self.warning_count,
            self.failed_count,
            self.not_applicable_count,
        ]
        colors = ["#4CAF50", "#FFC107", "#F44336", "#9E9E9E"]
        explode = (0.1, 0.1, 0.1, 0)

        ax1.pie(
            sizes,
            explode=explode,
            labels=labels,
            colors=colors,
            autopct="%1.1f%%",
            shadow=True,
            startangle=90,
        )
        ax1.axis("equal")
        ax1.set_title("Tổng quan kết quả kiểm tra")

        # 2. Biểu đồ cột phân tích theo mức độ ưu tiên
        # Tổ chức dữ liệu theo mức độ ưu tiên
        priority_results = defaultdict(lambda: {"passed": 0, "warning": 0, "failed": 0})

        for result in self.results:
            priority_name = {
                GoalPriority.CRITICAL: "Cao",
                GoalPriority.MAJOR: "Trung bình",
                GoalPriority.MINOR: "Thấp",
            }[result.priority]

            if result.result == GoalResult.PASSED:
                priority_results[priority_name]["passed"] += 1
            elif result.result == GoalResult.WARNING:
                priority_results[priority_name]["warning"] += 1
            elif result.result == GoalResult.FAILED:
                priority_results[priority_name]["failed"] += 1

        priorities = list(priority_results.keys())
        passed_values = [priority_results[p]["passed"] for p in priorities]
        warning_values = [priority_results[p]["warning"] for p in priorities]
        failed_values = [priority_results[p]["failed"] for p in priorities]

        bar_width = 0.25
        index = np.arange(len(priorities))

        ax2.bar(index, passed_values, bar_width, color="#4CAF50", label="Đạt")
        ax2.bar(
            index + bar_width,
            warning_values,
            bar_width,
            color="#FFC107",
            label="Cảnh báo",
        )
        ax2.bar(
            index + 2 * bar_width,
            failed_values,
            bar_width,
            color="#F44336",
            label="Không đạt",
        )

        ax2.set_xlabel("Mức độ ưu tiên")
        ax2.set_ylabel("Số lượng mục tiêu")
        ax2.set_title("Kết quả theo mức độ ưu tiên")
        ax2.set_xticks(index + bar_width)
        ax2.set_xticklabels(priorities)
        ax2.legend()

        plt.tight_layout()

        if file_path:
            plt.savefig(file_path, dpi=300, bbox_inches="tight")
            logger.info(f"Đã lưu biểu đồ tại: {file_path}")

        return fig

    def reset(self):
        """Đặt lại trạng thái của Plan Checker."""
        self.results = []
        self.passed_count = 0
        self.failed_count = 0
        self.warning_count = 0
        self.not_applicable_count = 0
        logger.info("Đã đặt lại Plan Checker")


# Hàm tiện ích để tích hợp với hệ thống UI
def run_plan_checker(plan, protocol_name=None, protocol_file=None, report_file=None):
    """
    Chạy Plan Checker và trả về kết quả.

    Parameters:
        plan: Plan object cần kiểm tra
        protocol_name: Tên protocol để sử dụng (nếu có)
        protocol_file: File protocol để sử dụng (nếu có)
        report_file: File để lưu báo cáo (nếu có)

    Returns:
        Tuple[List[PlanCheckerResult], Dict[str, Any], str]: Kết quả kiểm tra, tóm tắt, và báo cáo
    """
    checker = PlanChecker()
    checker.set_plan(plan)

    if protocol_file:
        protocol = checker.load_protocol_from_file(protocol_file)
    elif protocol_name:
        protocol = checker.load_protocol(protocol_name)
    else:
        # Mặc định sử dụng protocol phù hợp với vị trí điều trị
        site = plan.get_treatment_site()
        logger.info(f"Phát hiện vị trí điều trị: {site}")
        protocol = checker.load_protocol(site if site else "Default")

    results = checker.check_plan(protocol)
    summary = checker.get_summary()

    if report_file:
        report = checker.generate_report(report_file)
    else:
        report = checker.generate_report()

    return results, summary, report


if __name__ == "__main__":
    # Kiểm thử module
    print("Plan Checker module - Kiểm thử cơ bản")
