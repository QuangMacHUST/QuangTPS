#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module đánh giá chất lượng kế hoạch xạ trị.

Module này cung cấp các công cụ đánh giá chất lượng kế hoạch
theo các tiêu chuẩn lâm sàng và protocol điều trị.
"""

import logging
from typing import Dict, List, Optional, Union, Any
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


class GoalType(Enum):
    """Loại mục tiêu lâm sàng."""

    DOSE_VOLUME = "dose_volume"  # Dx: Liều tại x% thể tích
    VOLUME_DOSE = "volume_dose"  # Vx: Thể tích tại x Gy
    MEAN_DOSE = "mean_dose"  # Liều trung bình
    MAX_DOSE = "max_dose"  # Liều tối đa
    MIN_DOSE = "min_dose"  # Liều tối thiểu
    HOMOGENEITY_INDEX = "homogeneity_index"  # Chỉ số đồng đều
    CONFORMITY_INDEX = "conformity_index"  # Chỉ số phù hợp
    TCP = "tcp"  # Xác suất kiểm soát khối u
    NTCP = "ntcp"  # Xác suất biến chứng mô bình thường


class GoalPriority(Enum):
    """Mức ưu tiên của mục tiêu."""

    CRITICAL = "critical"  # Mục tiêu quan trọng
    IMPORTANT = "important"  # Mục tiêu quan trọng
    OPTIONAL = "optional"  # Mục tiêu tùy chọn
    REFERENCE = "reference"  # Mục tiêu tham khảo


class ComparisonOperator(Enum):
    """Toán tử so sánh."""

    LESS_THAN = "lt"  # <
    LESS_EQUAL = "le"  # <=
    GREATER_THAN = "gt"  # >
    GREATER_EQUAL = "ge"  # >=
    EQUAL = "eq"  # =
    NOT_EQUAL = "ne"  # !=


@dataclass
class ClinicalGoal:
    """
    Định nghĩa một mục tiêu lâm sàng.

    Ví dụ: PTV D95% >= 95% của prescription dose
    """

    structure_name: str  # Tên cấu trúc
    goal_type: GoalType  # Loại mục tiêu
    target_value: float  # Giá trị mục tiêu
    comparison: ComparisonOperator  # Toán tử so sánh
    priority: GoalPriority = GoalPriority.IMPORTANT

    # Thông tin bổ sung
    description: str = ""  # Mô tả mục tiêu
    units: str = "Gy"  # Đơn vị
    tolerance: Optional[float] = None  # Dung sai cho phép

    # Metadata
    protocol_name: str = ""  # Tên protocol
    created_date: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """Xử lý sau khi khởi tạo."""
        if not self.description:
            self.description = self._generate_description()

    def _generate_description(self) -> str:
        """Tạo mô tả tự động."""
        goal_desc = {
            GoalType.DOSE_VOLUME: f"D{self.target_value:.0f}%",
            GoalType.VOLUME_DOSE: f"V{self.target_value:.0f}Gy",
            GoalType.MEAN_DOSE: "Mean Dose",
            GoalType.MAX_DOSE: "Max Dose",
            GoalType.MIN_DOSE: "Min Dose",
            GoalType.HOMOGENEITY_INDEX: "Homogeneity Index",
            GoalType.CONFORMITY_INDEX: "Conformity Index",
            GoalType.TCP: "TCP",
            GoalType.NTCP: "NTCP",
        }

        op_desc = {
            ComparisonOperator.LESS_THAN: "<",
            ComparisonOperator.LESS_EQUAL: "<=",
            ComparisonOperator.GREATER_THAN: ">",
            ComparisonOperator.GREATER_EQUAL: ">=",
            ComparisonOperator.EQUAL: "=",
            ComparisonOperator.NOT_EQUAL: "!=",
        }

        goal_str = goal_desc.get(self.goal_type, str(self.goal_type.value))
        op_str = op_desc.get(self.comparison, str(self.comparison.value))

        return f"{self.structure_name}: {goal_str} {op_str} {self.target_value:.1f} {self.units}"

    def to_dict(self) -> Dict[str, Any]:
        """Chuyển đổi thành dictionary."""
        return {
            "structure_name": self.structure_name,
            "goal_type": self.goal_type.value,
            "target_value": self.target_value,
            "comparison": self.comparison.value,
            "priority": self.priority.value,
            "description": self.description,
            "units": self.units,
            "tolerance": self.tolerance,
            "protocol_name": self.protocol_name,
            "created_date": self.created_date.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClinicalGoal":
        """Tạo từ dictionary."""
        goal = cls(
            structure_name=data["structure_name"],
            goal_type=GoalType(data["goal_type"]),
            target_value=data["target_value"],
            comparison=ComparisonOperator(data["comparison"]),
            priority=GoalPriority(data.get("priority", "important")),
            description=data.get("description", ""),
            units=data.get("units", "Gy"),
            tolerance=data.get("tolerance"),
            protocol_name=data.get("protocol_name", ""),
        )

        # Parse created_date nếu có
        if "created_date" in data:
            try:
                goal.created_date = datetime.fromisoformat(data["created_date"])
            except Exception:
                pass

        return goal


@dataclass
class GoalEvaluation:
    """Kết quả đánh giá một mục tiêu lâm sàng."""

    goal: ClinicalGoal  # Mục tiêu gốc
    actual_value: float  # Giá trị thực tế đạt được
    achieved: bool  # Có đạt mục tiêu không
    deviation: float  # Độ lệch so với mục tiêu

    # Thông tin bổ sung
    confidence: Optional[float] = None  # Độ tin cậy (0-1)
    notes: str = ""  # Ghi chú

    @property
    def achievement_percentage(self) -> float:
        """Tỷ lệ đạt được mục tiêu (%)."""
        if self.goal.target_value == 0:
            return 100.0 if self.achieved else 0.0

        return min(100.0, (self.actual_value / self.goal.target_value) * 100.0)

    @property
    def score(self) -> float:
        """Điểm số đánh giá (0-100)."""
        if self.achieved:
            return 100.0
        else:
            # Tính điểm dựa trên độ lệch
            if self.goal.tolerance and abs(self.deviation) <= self.goal.tolerance:
                return 80.0  # Trong dung sai
            else:
                # Điểm giảm dần theo độ lệch
                max_deviation = self.goal.target_value * 0.5  # 50% target
                normalized_deviation = (
                    min(abs(self.deviation), max_deviation) / max_deviation
                )
                return max(0.0, 80.0 * (1.0 - normalized_deviation))

    def to_dict(self) -> Dict[str, Any]:
        """Chuyển đổi thành dictionary."""
        return {
            "goal": self.goal.to_dict(),
            "actual_value": self.actual_value,
            "achieved": self.achieved,
            "deviation": self.deviation,
            "confidence": self.confidence,
            "notes": self.notes,
            "achievement_percentage": self.achievement_percentage,
            "score": self.score,
        }


@dataclass
class PlanQualityScore:
    """Điểm đánh giá tổng thể chất lượng kế hoạch."""

    overall_score: float  # Điểm tổng thể (0-100)
    target_score: float  # Điểm target structures
    oar_score: float  # Điểm organs at risk

    # Chi tiết đánh giá
    total_goals: int = 0  # Tổng số mục tiêu
    achieved_goals: int = 0  # Số mục tiêu đạt được
    critical_failures: int = 0  # Số mục tiêu critical không đạt

    # Metadata
    evaluation_date: datetime = field(default_factory=datetime.now)
    evaluator: str = "QuangTPS"

    @property
    def achievement_rate(self) -> float:
        """Tỷ lệ đạt mục tiêu (%)."""
        if self.total_goals == 0:
            return 0.0
        return (self.achieved_goals / self.total_goals) * 100.0

    @property
    def grade(self) -> str:
        """Xếp loại chất lượng kế hoạch."""
        if self.overall_score >= 95:
            return "Excellent"
        elif self.overall_score >= 85:
            return "Good"
        elif self.overall_score >= 70:
            return "Acceptable"
        elif self.overall_score >= 50:
            return "Marginal"
        else:
            return "Poor"

    def to_dict(self) -> Dict[str, Any]:
        """Chuyển đổi thành dictionary."""
        return {
            "overall_score": self.overall_score,
            "target_score": self.target_score,
            "oar_score": self.oar_score,
            "total_goals": self.total_goals,
            "achieved_goals": self.achieved_goals,
            "critical_failures": self.critical_failures,
            "achievement_rate": self.achievement_rate,
            "grade": self.grade,
            "evaluation_date": self.evaluation_date.isoformat(),
            "evaluator": self.evaluator,
        }


class PlanQualityEvaluator:
    """Bộ đánh giá chất lượng kế hoạch."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def evaluate_goals(
        self, goals: List[ClinicalGoal], plan_data: Dict[str, Any]
    ) -> List[GoalEvaluation]:
        """
        Đánh giá danh sách các mục tiêu lâm sàng.

        Parameters:
            goals: Danh sách mục tiêu
            plan_data: Dữ liệu kế hoạch (DVH, dose metrics, etc.)

        Returns:
            List các GoalEvaluation
        """
        evaluations = []

        for goal in goals:
            try:
                evaluation = self._evaluate_single_goal(goal, plan_data)
                evaluations.append(evaluation)
            except Exception as e:
                self.logger.error(f"Lỗi đánh giá goal {goal.description}: {e}")
                # Tạo evaluation fallback
                evaluations.append(
                    GoalEvaluation(
                        goal=goal,
                        actual_value=0.0,
                        achieved=False,
                        deviation=0.0,
                        notes=f"Lỗi đánh giá: {str(e)}",
                    )
                )

        return evaluations

    def _evaluate_single_goal(
        self, goal: ClinicalGoal, plan_data: Dict[str, Any]
    ) -> GoalEvaluation:
        """Đánh giá một mục tiêu cụ thể."""

        # Lấy dữ liệu cấu trúc
        structure_data = plan_data.get("structures", {}).get(goal.structure_name)
        if not structure_data:
            raise ValueError(
                f"Không tìm thấy dữ liệu cho cấu trúc {goal.structure_name}"
            )

        # Tính giá trị thực tế dựa trên loại mục tiêu
        if goal.goal_type == GoalType.DOSE_VOLUME:
            actual_value = self._calculate_dose_at_volume(
                structure_data, goal.target_value
            )
        elif goal.goal_type == GoalType.VOLUME_DOSE:
            actual_value = self._calculate_volume_at_dose(
                structure_data, goal.target_value
            )
        elif goal.goal_type == GoalType.MEAN_DOSE:
            actual_value = structure_data.get("mean_dose", 0.0)
        elif goal.goal_type == GoalType.MAX_DOSE:
            actual_value = structure_data.get("max_dose", 0.0)
        elif goal.goal_type == GoalType.MIN_DOSE:
            actual_value = structure_data.get("min_dose", 0.0)
        else:
            # Fallback cho các loại khác
            actual_value = 0.0

        # Kiểm tra điều kiện đạt được
        achieved = self._check_goal_achievement(goal, actual_value)

        # Tính độ lệch
        deviation = actual_value - goal.target_value

        return GoalEvaluation(
            goal=goal, actual_value=actual_value, achieved=achieved, deviation=deviation
        )

    def _calculate_dose_at_volume(
        self, structure_data: Dict[str, Any], volume_percent: float
    ) -> float:
        """Tính liều tại phần trăm thể tích."""
        # Implement dose at volume calculation
        # Placeholder implementation
        return structure_data.get("mean_dose", 0.0)

    def _calculate_volume_at_dose(
        self, structure_data: Dict[str, Any], dose_gy: float
    ) -> float:
        """Tính thể tích tại liều."""
        # Implement volume at dose calculation
        # Placeholder implementation
        return 50.0  # Default 50%

    def _check_goal_achievement(self, goal: ClinicalGoal, actual_value: float) -> bool:
        """Kiểm tra xem mục tiêu có đạt được không."""
        target = goal.target_value

        if goal.comparison == ComparisonOperator.LESS_THAN:
            return actual_value < target
        elif goal.comparison == ComparisonOperator.LESS_EQUAL:
            return actual_value <= target
        elif goal.comparison == ComparisonOperator.GREATER_THAN:
            return actual_value > target
        elif goal.comparison == ComparisonOperator.GREATER_EQUAL:
            return actual_value >= target
        elif goal.comparison == ComparisonOperator.EQUAL:
            tolerance = goal.tolerance or (target * 0.02)  # 2% tolerance
            return abs(actual_value - target) <= tolerance
        elif goal.comparison == ComparisonOperator.NOT_EQUAL:
            tolerance = goal.tolerance or (target * 0.02)
            return abs(actual_value - target) > tolerance

        return False

    def calculate_plan_score(
        self, evaluations: List[GoalEvaluation]
    ) -> PlanQualityScore:
        """Tính điểm tổng thể cho kế hoạch."""

        if not evaluations:
            return PlanQualityScore(overall_score=0.0, target_score=0.0, oar_score=0.0)

        # Phân loại evaluations
        target_evaluations = []
        oar_evaluations = []

        for eval in evaluations:
            structure_name = eval.goal.structure_name.lower()
            if any(target in structure_name for target in ["ptv", "ctv", "gtv"]):
                target_evaluations.append(eval)
            else:
                oar_evaluations.append(eval)

        # Tính điểm target
        target_score = self._calculate_category_score(target_evaluations)

        # Tính điểm OAR
        oar_score = self._calculate_category_score(oar_evaluations)

        # Tính điểm tổng thể (weighted average)
        if target_evaluations and oar_evaluations:
            overall_score = (target_score * 0.6) + (oar_score * 0.4)
        elif target_evaluations:
            overall_score = target_score
        elif oar_evaluations:
            overall_score = oar_score
        else:
            overall_score = 0.0

        # Thống kê
        total_goals = len(evaluations)
        achieved_goals = sum(1 for eval in evaluations if eval.achieved)
        critical_failures = sum(
            1
            for eval in evaluations
            if not eval.achieved and eval.goal.priority == GoalPriority.CRITICAL
        )

        return PlanQualityScore(
            overall_score=overall_score,
            target_score=target_score,
            oar_score=oar_score,
            total_goals=total_goals,
            achieved_goals=achieved_goals,
            critical_failures=critical_failures,
        )

    def _calculate_category_score(self, evaluations: List[GoalEvaluation]) -> float:
        """Tính điểm cho một nhóm evaluations."""
        if not evaluations:
            return 100.0  # Perfect score if no goals

        # Weighted score dựa trên priority
        weights = {
            GoalPriority.CRITICAL: 3.0,
            GoalPriority.IMPORTANT: 2.0,
            GoalPriority.OPTIONAL: 1.0,
            GoalPriority.REFERENCE: 0.5,
        }

        total_weighted_score = 0.0
        total_weight = 0.0

        for eval in evaluations:
            weight = weights.get(eval.goal.priority, 1.0)
            total_weighted_score += eval.score * weight
            total_weight += weight

        if total_weight == 0:
            return 0.0

        return total_weighted_score / total_weight


# Utility functions
def create_standard_goals(protocol_name: str, site: str) -> List[ClinicalGoal]:
    """Tạo các mục tiêu chuẩn cho site điều trị."""
    goals = []

    if site.lower() == "prostate":
        goals.extend(
            [
                ClinicalGoal(
                    structure_name="PTV",
                    goal_type=GoalType.DOSE_VOLUME,
                    target_value=95.0,  # D95%
                    comparison=ComparisonOperator.GREATER_EQUAL,
                    priority=GoalPriority.CRITICAL,
                    description="PTV D95% >= 95% prescription",
                ),
                ClinicalGoal(
                    structure_name="Rectum",
                    goal_type=GoalType.VOLUME_DOSE,
                    target_value=50.0,  # V50Gy
                    comparison=ComparisonOperator.LESS_THAN,
                    priority=GoalPriority.IMPORTANT,
                    description="Rectum V50Gy < 50%",
                ),
                ClinicalGoal(
                    structure_name="Bladder",
                    goal_type=GoalType.VOLUME_DOSE,
                    target_value=50.0,  # V50Gy
                    comparison=ComparisonOperator.LESS_THAN,
                    priority=GoalPriority.IMPORTANT,
                    description="Bladder V50Gy < 50%",
                ),
            ]
        )

    elif site.lower() == "head_neck":
        goals.extend(
            [
                ClinicalGoal(
                    structure_name="PTV",
                    goal_type=GoalType.DOSE_VOLUME,
                    target_value=95.0,
                    comparison=ComparisonOperator.GREATER_EQUAL,
                    priority=GoalPriority.CRITICAL,
                ),
                ClinicalGoal(
                    structure_name="Spinal_Cord",
                    goal_type=GoalType.MAX_DOSE,
                    target_value=45.0,
                    comparison=ComparisonOperator.LESS_THAN,
                    priority=GoalPriority.CRITICAL,
                ),
                ClinicalGoal(
                    structure_name="Parotid_L",
                    goal_type=GoalType.MEAN_DOSE,
                    target_value=26.0,
                    comparison=ComparisonOperator.LESS_THAN,
                    priority=GoalPriority.IMPORTANT,
                ),
            ]
        )

    # Set protocol name for all goals
    for goal in goals:
        goal.protocol_name = protocol_name

    return goals


# Export
__all__ = [
    "GoalType",
    "GoalPriority",
    "ComparisonOperator",
    "ClinicalGoal",
    "GoalEvaluation",
    "PlanQualityScore",
    "PlanQualityEvaluator",
    "create_standard_goals",
]
