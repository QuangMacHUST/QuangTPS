from typing import Dict, List, Optional, Union, Tuple, Any
import numpy as np
import json
import os
from datetime import datetime

# Import các lớp liên quan
try:
    from quangtps.evaluation.protocols.clinical_protocol import ClinicalProtocol
    from quangtps.evaluation.protocols.clinical_goal import ClinicalGoal
    from quangtps.evaluation.plan_quality.plan_quality_score import PlanQualityScore
    from quangtps.evaluation.dvh.dvh_analyzer import DVHAnalyzer
except ImportError:
    # Tạo các lớp giả mạch khi không có class thực
    class ClinicalProtocol:
        def __init__(self, name="", description="", goals=None):
            self.name = name
            self.description = description
            self.goals = goals or []

    class ClinicalGoal:
        def __init__(
            self, structure_id="", type=None, operator=None, value=0, priority=0
        ):
            self.structure_id = structure_id
            self.type = type
            self.operator = operator
            self.value = value
            self.priority = priority

    class PlanQualityScore:
        EXCELLENT = 0
        GOOD = 1
        ACCEPTABLE = 2
        MARGINAL = 3
        POOR = 4
        NOT_EVALUATED = 5

    class DVHAnalyzer:
        def __init__(self):
            pass

        def get_dvh_metric(self, structure_id, metric_type, param=None):
            return 0.0


class GoalResult:
    """
    Kết quả đánh giá một mục tiêu lâm sàng.
    """

    def __init__(self, goal, actual_value=0.0, achieved=False, acceptable=False):
        """
        Khởi tạo kết quả đánh giá mục tiêu.

        Args:
            goal (ClinicalGoal): Mục tiêu lâm sàng đang đánh giá
            actual_value (float): Giá trị thực tế đạt được
            achieved (bool): True nếu đạt mục tiêu, False nếu không
            acceptable (bool): True nếu không đạt nhưng chấp nhận được, False nếu không
        """
        self.goal = goal
        self.actual_value = actual_value
        self.achieved = achieved
        self.acceptable = acceptable
        self.difference = self._calculate_difference()

    def _calculate_difference(self):
        """Tính toán sự khác biệt giữa giá trị mục tiêu và giá trị thực tế."""
        if not hasattr(self.goal, "value"):
            return 0.0

        try:
            # Tính toán chênh lệch tương đối (%)
            if self.goal.value != 0:
                return (
                    (self.actual_value - self.goal.value) / abs(self.goal.value) * 100
                )
            else:
                return 0.0
        except Exception:
            return 0.0

    def __str__(self):
        status = "Đạt" if self.achieved else "Không đạt"
        if not self.achieved and self.acceptable:
            status = "Chấp nhận được"

        return f"Mục tiêu: {self.goal}, Giá trị thực tế: {self.actual_value}, Kết quả: {status}"


class EvaluationResult:
    """
    Kết quả đánh giá kế hoạch xạ trị theo protocol lâm sàng.
    """

    def __init__(self, protocol=None):
        """
        Khởi tạo kết quả đánh giá.

        Args:
            protocol (ClinicalProtocol): Protocol lâm sàng được sử dụng để đánh giá
        """
        self.protocol = protocol
        self.goal_results = []
        self.scores = {"overall": 0.0, "target": 0.0, "oar": 0.0}
        self.metrics = {}
        self.evaluation_date = datetime.now()

    def add_goal_result(self, goal_result):
        """Thêm kết quả đánh giá một mục tiêu vào danh sách."""
        self.goal_results.append(goal_result)

    def set_scores(self, overall_score, target_score, oar_score):
        """Thiết lập điểm đánh giá kế hoạch."""
        self.scores["overall"] = overall_score
        self.scores["target"] = target_score
        self.scores["oar"] = oar_score

    def add_metric(self, name, value):
        """Thêm chỉ số đánh giá kế hoạch."""
        self.metrics[name] = value

    def get_pass_rate(self):
        """Tính tỷ lệ mục tiêu đạt được (%)."""
        if not self.goal_results:
            return 0.0

        achieved = sum(1 for result in self.goal_results if result.achieved)
        return achieved / len(self.goal_results) * 100

    def get_target_pass_rate(self):
        """Tính tỷ lệ mục tiêu PTV đạt được (%)."""
        target_results = [
            r
            for r in self.goal_results
            if hasattr(r.goal, "structure_name")
            and (
                "PTV" in r.goal.structure_name
                or "GTV" in r.goal.structure_name
                or "CTV" in r.goal.structure_name
            )
        ]

        if not target_results:
            return 0.0

        achieved = sum(1 for result in target_results if result.achieved)
        return achieved / len(target_results) * 100

    def get_oar_pass_rate(self):
        """Tính tỷ lệ mục tiêu OAR đạt được (%)."""
        oar_results = [
            r
            for r in self.goal_results
            if hasattr(r.goal, "structure_name")
            and not (
                "PTV" in r.goal.structure_name
                or "GTV" in r.goal.structure_name
                or "CTV" in r.goal.structure_name
            )
        ]

        if not oar_results:
            return 0.0

        achieved = sum(1 for result in oar_results if result.achieved)
        return achieved / len(oar_results) * 100

    def __str__(self):
        return (
            f"Đánh giá kế hoạch theo protocol: {self.protocol.name if self.protocol else 'N/A'}\n"
            f"Số mục tiêu: {len(self.goal_results)}\n"
            f"Tỉ lệ đạt: {self.get_pass_rate():.1f}%\n"
            f"Điểm tổng thể: {self.scores['overall']:.1f}\n"
            f"Điểm PTV: {self.scores['target']:.1f}\n"
            f"Điểm OAR: {self.scores['oar']:.1f}\n"
            f"Ngày đánh giá: {self.evaluation_date.strftime('%d/%m/%Y %H:%M:%S')}"
        )


class PlanQualityEvaluator:
    """
    Đánh giá chất lượng kế hoạch xạ trị dựa trên các protocol lâm sàng.

    Lớp này cung cấp các công cụ để đánh giá kế hoạch xạ trị theo các tiêu chí lâm sàng,
    đánh giá mục tiêu liều và OAR, và tính toán các chỉ số đánh giá kế hoạch.
    """

    def __init__(self, dvh_analyzer=None):
        """
        Khởi tạo evaluator với DVHAnalyzer.

        Args:
            dvh_analyzer (DVHAnalyzer): Bộ phân tích DVH
        """
        self.dvh_analyzer = dvh_analyzer

    def set_dvh_analyzer(self, dvh_analyzer):
        """
        Thiết lập bộ phân tích DVH.

        Args:
            dvh_analyzer (DVHAnalyzer): Bộ phân tích DVH
        """
        self.dvh_analyzer = dvh_analyzer

    def evaluate_with_protocol(self, plan, protocol):
        """
        Đánh giá kế hoạch xạ trị dựa trên protocol lâm sàng.

        Args:
            plan: Kế hoạch xạ trị cần đánh giá
            protocol (ClinicalProtocol): Protocol lâm sàng

        Returns:
            EvaluationResult: Kết quả đánh giá
        """
        if not protocol or not hasattr(protocol, "goals"):
            print("Protocol không hợp lệ hoặc không có mục tiêu")
            return None

        if not self.dvh_analyzer:
            print("Không có DVHAnalyzer, không thể đánh giá kế hoạch")
            return None

        result = EvaluationResult(protocol)

        # Đánh giá từng mục tiêu lâm sàng
        for goal in protocol.goals:
            goal_result = self._evaluate_goal(plan, goal)
            if goal_result:
                result.add_goal_result(goal_result)

        # Tính điểm chất lượng kế hoạch
        self._calculate_scores(result)

        # Thêm các chỉ số đánh giá bổ sung
        self._add_advanced_metrics(result, plan)

        return result

    def _evaluate_goal(self, plan, goal):
        """
        Đánh giá một mục tiêu lâm sàng.

        Args:
            plan: Kế hoạch xạ trị
            goal (ClinicalGoal): Mục tiêu lâm sàng

        Returns:
            GoalResult: Kết quả đánh giá mục tiêu
        """
        if not hasattr(goal, "structure_id") or not hasattr(goal, "type"):
            return None

        # Lấy giá trị thực tế từ DVH
        actual_value = self._get_metric_value(goal, plan)

        # So sánh với giá trị mục tiêu
        achieved = False
        acceptable = False

        if hasattr(goal, "operator") and hasattr(goal, "value"):
            # Đánh giá dựa trên toán tử so sánh
            if goal.operator == "less_than" or goal.operator == "<":
                achieved = actual_value < goal.value
                acceptable = actual_value < goal.value * 1.1  # 10% dung sai

            elif goal.operator == "less_than_or_equal" or goal.operator == "<=":
                achieved = actual_value <= goal.value
                acceptable = actual_value <= goal.value * 1.1

            elif goal.operator == "greater_than" or goal.operator == ">":
                achieved = actual_value > goal.value
                acceptable = actual_value > goal.value * 0.9  # 10% dung sai

            elif goal.operator == "greater_than_or_equal" or goal.operator == ">=":
                achieved = actual_value >= goal.value
                acceptable = actual_value >= goal.value * 0.9

            elif goal.operator == "equal" or goal.operator == "=":
                tolerance = goal.value * 0.05  # 5% dung sai
                achieved = abs(actual_value - goal.value) <= tolerance
                acceptable = abs(actual_value - goal.value) <= tolerance * 2

        return GoalResult(goal, actual_value, achieved, acceptable and not achieved)

    def _get_metric_value(self, goal, plan):
        """
        Lấy giá trị thực tế từ DVH theo loại mục tiêu.

        Args:
            goal (ClinicalGoal): Mục tiêu lâm sàng
            plan: Kế hoạch xạ trị

        Returns:
            float: Giá trị thực tế
        """
        structure_id = goal.structure_id

        try:
            # Ví dụ các kiểu mục tiêu khác nhau
            if hasattr(goal.type, "__str__"):
                goal_type_str = str(goal.type)
            else:
                goal_type_str = goal.type

            if "DOSE_VOLUME" in goal_type_str:
                # Ví dụ: D95% > 50Gy
                param = goal.parameter if hasattr(goal, "parameter") else 95
                return self.dvh_analyzer.get_dvh_metric(
                    structure_id, "dose_at_volume", param
                )

            elif "VOLUME_DOSE" in goal_type_str:
                # Ví dụ: V20Gy < 30%
                param = goal.parameter if hasattr(goal, "parameter") else 20
                return self.dvh_analyzer.get_dvh_metric(
                    structure_id, "volume_at_dose", param
                )

            elif "MEAN_DOSE" in goal_type_str:
                return self.dvh_analyzer.get_dvh_metric(structure_id, "mean_dose")

            elif "MAX_DOSE" in goal_type_str:
                return self.dvh_analyzer.get_dvh_metric(structure_id, "max_dose")

            elif "MIN_DOSE" in goal_type_str:
                return self.dvh_analyzer.get_dvh_metric(structure_id, "min_dose")

            elif "HOMOGENEITY_INDEX" in goal_type_str:
                d2 = self.dvh_analyzer.get_dvh_metric(structure_id, "dose_at_volume", 2)
                d98 = self.dvh_analyzer.get_dvh_metric(
                    structure_id, "dose_at_volume", 98
                )
                d50 = self.dvh_analyzer.get_dvh_metric(
                    structure_id, "dose_at_volume", 50
                )
                if d50 == 0:
                    return 0
                return (d2 - d98) / d50

            elif "CONFORMITY_INDEX" in goal_type_str:
                if not hasattr(goal, "reference_dose") or not hasattr(
                    goal, "target_volume"
                ):
                    return 0
                ref_dose = goal.reference_dose
                target_vol = goal.target_volume
                v_ref = self.dvh_analyzer.get_dvh_metric(
                    structure_id, "volume_at_dose", ref_dose
                )
                if target_vol == 0:
                    return 0
                return v_ref / target_vol

            else:
                # Mục tiêu không được hỗ trợ
                print(f"Kiểu mục tiêu không được hỗ trợ: {goal.type}")
                return 0

        except Exception as e:
            print(f"Lỗi khi lấy giá trị DVH cho {structure_id}: {str(e)}")
            return 0

    def _calculate_scores(self, result):
        """
        Tính điểm chất lượng kế hoạch.

        Args:
            result (EvaluationResult): Kết quả đánh giá
        """
        # Tính điểm dựa trên tỷ lệ mục tiêu đạt được
        target_pass_rate = result.get_target_pass_rate()
        oar_pass_rate = result.get_oar_pass_rate()
        overall_pass_rate = result.get_pass_rate()

        # Chuyển đổi tỷ lệ thành điểm
        result.set_scores(
            self._rate_to_score(overall_pass_rate),
            self._rate_to_score(target_pass_rate),
            self._rate_to_score(oar_pass_rate),
        )

    def _rate_to_score(self, pass_rate):
        """
        Chuyển đổi tỷ lệ đạt thành điểm.

        Args:
            pass_rate (float): Tỷ lệ mục tiêu đạt được (%)

        Returns:
            float: Điểm số (0-100)
        """
        if pass_rate >= 95:
            return 95.0
        elif pass_rate >= 90:
            return 90.0
        elif pass_rate >= 85:
            return 85.0
        elif pass_rate >= 80:
            return 80.0
        elif pass_rate >= 75:
            return 75.0
        elif pass_rate >= 70:
            return 70.0
        elif pass_rate >= 65:
            return 65.0
        elif pass_rate >= 60:
            return 60.0
        elif pass_rate > 0:
            return 50.0
        else:
            return 0.0

    def _add_advanced_metrics(self, result, plan):
        """
        Thêm các chỉ số đánh giá nâng cao.

        Args:
            result (EvaluationResult): Kết quả đánh giá
            plan: Kế hoạch xạ trị
        """
        # Ví dụ thêm một số chỉ số
        try:
            # Chỉ số đồng nhất cho các PTV
            structures = (
                self.dvh_analyzer.get_structures()
                if hasattr(self.dvh_analyzer, "get_structures")
                else []
            )
            for structure in structures:
                if (
                    hasattr(structure, "name")
                    and hasattr(structure, "id")
                    and "PTV" in structure.name
                ):
                    hi = self._calculate_homogeneity_index(structure.id)
                    result.add_metric(f"HI_{structure.name}", hi)

                    ci = self._calculate_conformity_index(structure.id, plan)
                    result.add_metric(f"CI_{structure.name}", ci)

            # Thêm các chỉ số khác theo nhu cầu

        except Exception as e:
            print(f"Lỗi khi tính toán chỉ số nâng cao: {str(e)}")

    def _calculate_homogeneity_index(self, structure_id):
        """
        Tính chỉ số đồng nhất (HI = (D2% - D98%) / D50%).

        Args:
            structure_id: ID của cấu trúc

        Returns:
            float: Chỉ số đồng nhất
        """
        try:
            d2 = self.dvh_analyzer.get_dvh_metric(structure_id, "dose_at_volume", 2)
            d98 = self.dvh_analyzer.get_dvh_metric(structure_id, "dose_at_volume", 98)
            d50 = self.dvh_analyzer.get_dvh_metric(structure_id, "dose_at_volume", 50)

            if d50 == 0:
                return 0

            return (d2 - d98) / d50

        except Exception:
            return 0

    def _calculate_conformity_index(self, structure_id, plan):
        """
        Tính chỉ số phù hợp (CI = V95% / VPTV).

        Args:
            structure_id: ID của cấu trúc
            plan: Kế hoạch xạ trị

        Returns:
            float: Chỉ số phù hợp
        """
        try:
            # Lấy thể tích mục tiêu
            target_volume = 0
            structures = (
                self.dvh_analyzer.get_structures()
                if hasattr(self.dvh_analyzer, "get_structures")
                else []
            )
            for structure in structures:
                if (
                    hasattr(structure, "id")
                    and structure.id == structure_id
                    and hasattr(structure, "volume")
                ):
                    target_volume = structure.volume
                    break

            if target_volume == 0:
                return 0

            # Lấy liều tham chiếu (95% của liều kê toa)
            if hasattr(plan, "prescription") and hasattr(plan.prescription, "dose"):
                ref_dose = plan.prescription.dose * 0.95
            else:
                # Nếu không có liều kê toa, sử dụng D95 của mục tiêu
                ref_dose = self.dvh_analyzer.get_dvh_metric(
                    structure_id, "dose_at_volume", 95
                )

            # Tính thể tích nhận được ít nhất liều tham chiếu
            v_ref = self.dvh_analyzer.get_dvh_metric(
                structure_id, "volume_at_dose", ref_dose
            )

            return v_ref / target_volume

        except Exception as e:
            print(f"Lỗi khi tính CI: {str(e)}")
            return 0

    def export_evaluation_result(self, result, filename, format_type="json"):
        """
        Xuất kết quả đánh giá sang file.

        Args:
            result (EvaluationResult): Kết quả đánh giá
            filename (str): Đường dẫn file xuất
            format_type (str): Định dạng file ('json', 'csv', 'html')

        Returns:
            bool: True nếu thành công, False nếu thất bại
        """
        try:
            if format_type == "json":
                # Xuất dạng JSON
                data = {
                    "protocol": result.protocol.name
                    if result.protocol and hasattr(result.protocol, "name")
                    else "N/A",
                    "evaluation_date": result.evaluation_date.strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                    "scores": result.scores,
                    "metrics": result.metrics,
                    "goal_results": [],
                }

                for goal_result in result.goal_results:
                    goal_data = {
                        "structure": goal_result.goal.structure_name
                        if hasattr(goal_result.goal, "structure_name")
                        else "N/A",
                        "description": goal_result.goal.description
                        if hasattr(goal_result.goal, "description")
                        else "N/A",
                        "actual_value": goal_result.actual_value,
                        "achieved": goal_result.achieved,
                        "acceptable": goal_result.acceptable,
                    }
                    data["goal_results"].append(goal_data)

                # Đảm bảo thư mục đích tồn tại
                os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)

                with open(filename, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)

            elif format_type == "csv":
                # Xuất dạng CSV
                with open(filename, "w", encoding="utf-8") as f:
                    # Thông tin cơ bản
                    f.write(
                        f"Tên protocol,{result.protocol.name if result.protocol and hasattr(result.protocol, 'name') else 'N/A'}\n"
                    )
                    f.write(
                        f"Ngày đánh giá,{result.evaluation_date.strftime('%Y-%m-%d %H:%M:%S')}\n"
                    )
                    f.write(f"Điểm tổng thể,{result.scores['overall']}\n")
                    f.write(f"Điểm PTV,{result.scores['target']}\n")
                    f.write(f"Điểm OAR,{result.scores['oar']}\n\n")

                    # Bảng kết quả mục tiêu
                    f.write("Cấu trúc,Mục tiêu,Giá trị thực tế,Kết quả\n")
                    for goal_result in result.goal_results:
                        structure_name = (
                            goal_result.goal.structure_name
                            if hasattr(goal_result.goal, "structure_name")
                            else "N/A"
                        )
                        description = (
                            goal_result.goal.description
                            if hasattr(goal_result.goal, "description")
                            else "N/A"
                        )
                        actual_value = goal_result.actual_value

                        status = "Đạt" if goal_result.achieved else "Không đạt"
                        if not goal_result.achieved and goal_result.acceptable:
                            status = "Chấp nhận được"

                        f.write(
                            f"{structure_name},{description},{actual_value},{status}\n"
                        )

                    # Chỉ số nâng cao
                    if result.metrics:
                        f.write("\nChỉ số nâng cao\n")
                        f.write("Tên,Giá trị\n")
                        for name, value in result.metrics.items():
                            f.write(f"{name},{value}\n")

            else:
                print(f"Định dạng không được hỗ trợ: {format_type}")
                return False

            return True

        except Exception as e:
            print(f"Lỗi khi xuất kết quả đánh giá: {str(e)}")
            return False
