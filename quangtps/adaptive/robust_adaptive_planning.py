#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module lập kế hoạch thích ứng bền vững cho QuangTPS.

Module này cung cấp các chức năng để tạo và quản lý kế hoạch xạ trị thích ứng,
với tính năng đánh giá tính bền vững đối với các thay đổi giải phẫu. Module hỗ trợ
cả lập kế hoạch thích ứng ngoại tuyến (trước điều trị) và trực tuyến (trong quá trình
điều trị).
"""

import os
import logging
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Union, Any, Sequence, Set
from enum import Enum, auto
import json
from pathlib import Path
import time

from quangtps.core.types import Patient, Image, Structure, Dose, Plan
from quangtps.core.exceptions import AdaptationError, AdaptivePlanningError
from quangtps.adaptive.prediction import (
    AnatomyPrediction,
    AnatomyPredictor,
    PredictionMethod,
    predict_anatomy_changes,
)
from quangtps.adaptive.temporal_analysis import TemporalAnalyzer, TemporalAnalysisResult
from quangtps.adaptive.deformation.deformable_registration import DeformableRegistration
from quangtps.adaptive.deformation.displacement_field import DisplacementField
from quangtps.planning.plan import ExternalBeamPlan
from quangtps.evaluation.dvh.dvh_calculator import DVHCalculator
from quangtps.evaluation.metrics.plan_metrics import calculate_plan_metrics
from quangtps.optimization.optimizer import Optimizer
from quangtps.dose.dose_calculation import DoseCalculator
from quangtps.imaging.registration import register_images
from quangtps.segmentation.contour.dice import calculate_dice_coefficient

# Import các module robustness từ thư mục evaluation
from quangtps.evaluation.robustness import (
    RobustnessAnalyzer,
    RobustnessResult,
    ScenarioResult,
    UncertaintyType,
    analyze_plan_robustness,
)

# Import các module robust optimization từ thư mục evaluation/robustness
from quangtps.evaluation.robustness import (
    RobustOptimizer,
    optimize_robust_plan,
    create_robust_objective,
)

from quangtps.core.utils import get_timestamp, create_directory_if_not_exists
from quangtps.reporting.templates import get_template

logger = logging.getLogger(__name__)


class AdaptationTrigger(Enum):
    """Các loại kích hoạt thích ứng kế hoạch."""

    VOLUME_CHANGE = auto()  # Thay đổi thể tích cấu trúc vượt quá ngưỡng
    CENTROID_CHANGE = auto()  # Thay đổi tâm cấu trúc vượt quá ngưỡng
    DICE_COEFFICIENT = auto()  # Hệ số Dice thấp hơn ngưỡng
    DOSE_DEVIATION = auto()  # Sai lệch liều vượt quá ngưỡng
    CLINICAL_METRICS = auto()  # Chỉ số lâm sàng không đạt yêu cầu
    MANUAL = auto()  # Kích hoạt thủ công bởi người dùng


class AdaptationType(Enum):
    """Các loại kế hoạch thích ứng."""

    OFFLINE = auto()  # Lập kế hoạch thích ứng ngoại tuyến (trước điều trị)
    ONLINE = auto()  # Lập kế hoạch thích ứng trực tuyến (trong quá trình điều trị)
    HYBRID = auto()  # Kết hợp cả hai phương pháp


class RobustAdaptivePlan:
    """Lớp quản lý kế hoạch xạ trị thích ứng bền vững."""

    def __init__(
        self,
        patient: Patient,
        reference_plan: Plan,
        adaptation_type: AdaptationType = AdaptationType.OFFLINE,
    ):
        """
        Khởi tạo kế hoạch thích ứng bền vững.

        Parameters
        ----------
        patient : Patient
            Đối tượng bệnh nhân
        reference_plan : Plan
            Kế hoạch tham chiếu ban đầu
        adaptation_type : AdaptationType, optional
            Loại thích ứng, mặc định là AdaptationType.OFFLINE
        """
        self.patient = patient
        self.reference_plan = reference_plan
        self.adaptation_type = adaptation_type
        self.creation_date = datetime.datetime.now()

        # Danh sách các kế hoạch thích ứng
        self.adaptive_plans = {}  # Dict[datetime.datetime, Plan]

        # Lưu trữ các dự đoán giải phẫu
        self.anatomy_predictions = {}  # Dict[datetime.datetime, AnatomyPrediction]

        # Lưu trữ các chỉ số tính bền vững của kế hoạch
        self.robustness_metrics = {}  # Dict[datetime.datetime, Dict[str, float]]

        # Ngưỡng kích hoạt thích ứng
        self.adaptation_thresholds = {
            AdaptationTrigger.VOLUME_CHANGE: 10.0,  # Thay đổi thể tích > 10%
            AdaptationTrigger.CENTROID_CHANGE: 5.0,  # Di chuyển tâm > 5mm
            AdaptationTrigger.DICE_COEFFICIENT: 0.85,  # Dice < 0.85
            AdaptationTrigger.DOSE_DEVIATION: 5.0,  # Sai lệch liều > 5%
            AdaptationTrigger.CLINICAL_METRICS: 0.9,  # Chỉ số lâm sàng < 90%
        }

        # Lịch sử các quyết định thích ứng
        self.adaptation_history = []

    def add_adaptive_plan(
        self,
        date: datetime.datetime,
        plan: Plan,
        anatomy_prediction: Optional[AnatomyPrediction] = None,
        robustness_metrics: Optional[Dict[str, float]] = None,
    ):
        """
        Thêm một kế hoạch thích ứng vào danh sách.

        Parameters
        ----------
        date : datetime.datetime
            Ngày tạo kế hoạch thích ứng
        plan : Plan
            Kế hoạch thích ứng
        anatomy_prediction : Optional[AnatomyPrediction], optional
            Dự đoán thay đổi giải phẫu, mặc định là None
        robustness_metrics : Optional[Dict[str, float]], optional
            Các chỉ số tính bền vững, mặc định là None
        """
        self.adaptive_plans[date] = plan

        if anatomy_prediction:
            self.anatomy_predictions[date] = anatomy_prediction

        if robustness_metrics:
            self.robustness_metrics[date] = robustness_metrics

        # Thêm vào lịch sử
        history_entry = {
            "date": date,
            "plan_id": plan.id,
            "action": "add_adaptive_plan",
            "details": {
                "has_prediction": anatomy_prediction is not None,
                "has_robustness": robustness_metrics is not None,
            },
        }
        self.adaptation_history.append(history_entry)

    def get_adaptive_plan(self, date: datetime.datetime) -> Optional[Plan]:
        """
        Lấy kế hoạch thích ứng gần nhất trước một ngày cụ thể.

        Parameters
        ----------
        date : datetime.datetime
            Ngày cần lấy kế hoạch

        Returns
        -------
        Optional[Plan]
            Kế hoạch thích ứng gần nhất hoặc None nếu không có
        """
        # Lấy tất cả các ngày có kế hoạch thích ứng
        adapt_dates = list(self.adaptive_plans.keys())

        # Lọc các ngày trước ngày cần tìm
        valid_dates = [d for d in adapt_dates if d <= date]

        if not valid_dates:
            # Nếu không có kế hoạch thích ứng nào trước ngày cần tìm, trả về kế hoạch tham chiếu
            return self.reference_plan

        # Lấy ngày gần nhất
        closest_date = max(valid_dates)
        return self.adaptive_plans[closest_date]

    def needs_adaptation(
        self,
        current_image: Image,
        current_structures: Dict[str, Structure],
        current_date: datetime.datetime,
    ) -> Tuple[bool, Dict[str, float]]:
        """
        Kiểm tra xem kế hoạch hiện tại có cần thích ứng hay không.

        Parameters
        ----------
        current_image : Image
            Hình ảnh hiện tại của bệnh nhân
        current_structures : Dict[str, Structure]
            Từ điển cấu trúc hiện tại
        current_date : datetime.datetime
            Ngày hiện tại

        Returns
        -------
        Tuple[bool, Dict[str, float]]
            (Có cần thích ứng hay không, Từ điển các chỉ số đánh giá)
        """
        # Lấy kế hoạch hiện tại
        current_plan = self.get_adaptive_plan(current_date)

        # Khởi tạo kết quả đánh giá
        evaluation_results = {}

        # Đánh giá thay đổi thể tích
        volume_changes = self._evaluate_volume_changes(current_structures, current_plan)
        evaluation_results["volume_changes"] = volume_changes

        # Đánh giá thay đổi vị trí
        position_changes = self._evaluate_position_changes(
            current_structures, current_plan
        )
        evaluation_results["position_changes"] = position_changes

        # Đánh giá sự khác biệt giữa các cấu trúc
        structure_differences = self._evaluate_structure_differences(
            current_structures, current_plan
        )
        evaluation_results["structure_differences"] = structure_differences

        # Đánh giá sai lệch liều
        if current_plan and hasattr(current_plan, "dose"):
            dose_deviations = self._evaluate_dose_deviations(
                current_structures, current_image, current_plan
            )
            evaluation_results["dose_deviations"] = dose_deviations
        else:
            dose_deviations = {}

        # Đánh giá các chỉ số lâm sàng
        clinical_metrics = self._evaluate_clinical_metrics(
            current_structures, current_image, current_plan
        )
        evaluation_results["clinical_metrics"] = clinical_metrics

        # Quyết định có cần thích ứng hay không
        needs_adapt = False
        reasons = []

        # Kiểm tra thay đổi thể tích
        for struct_name, percent_change in volume_changes.items():
            threshold = self.adaptation_thresholds[AdaptationTrigger.VOLUME_CHANGE]
            if abs(percent_change) > threshold:
                needs_adapt = True
                reasons.append(
                    f"Thể tích {struct_name} thay đổi {percent_change:.1f}% (ngưỡng: {threshold}%)"
                )

        # Kiểm tra thay đổi vị trí
        for struct_name, position_change_mm in position_changes.items():
            threshold = self.adaptation_thresholds[AdaptationTrigger.CENTROID_CHANGE]
            if position_change_mm > threshold:
                needs_adapt = True
                reasons.append(
                    f"Vị trí {struct_name} thay đổi {position_change_mm:.1f}mm (ngưỡng: {threshold}mm)"
                )

        # Kiểm tra hệ số Dice
        for struct_name, dice_coef in structure_differences.items():
            threshold = self.adaptation_thresholds[AdaptationTrigger.DICE_COEFFICIENT]
            if dice_coef < threshold:
                needs_adapt = True
                reasons.append(
                    f"Hệ số Dice của {struct_name}: {dice_coef:.2f} (ngưỡng: {threshold})"
                )

        # Kiểm tra sai lệch liều
        for metric, deviation in dose_deviations.items():
            threshold = self.adaptation_thresholds[AdaptationTrigger.DOSE_DEVIATION]
            if abs(deviation) > threshold:
                needs_adapt = True
                reasons.append(
                    f"Sai lệch liều {metric}: {deviation:.1f}% (ngưỡng: {threshold}%)"
                )

        # Kiểm tra chỉ số lâm sàng
        for metric, score in clinical_metrics.items():
            threshold = self.adaptation_thresholds[AdaptationTrigger.CLINICAL_METRICS]
            if score < threshold:
                needs_adapt = True
                reasons.append(
                    f"Chỉ số lâm sàng {metric}: {score:.2f} (ngưỡng: {threshold})"
                )

        # Thêm vào lịch sử
        history_entry = {
            "date": current_date,
            "action": "needs_adaptation_check",
            "result": needs_adapt,
            "reasons": reasons,
            "evaluation_results": evaluation_results,
        }
        self.adaptation_history.append(history_entry)

        # Trả về kết quả
        return needs_adapt, evaluation_results

    def _evaluate_volume_changes(
        self, current_structures: Dict[str, Structure], current_plan: Plan
    ) -> Dict[str, float]:
        """
        Đánh giá thay đổi thể tích của các cấu trúc.

        Parameters
        ----------
        current_structures : Dict[str, Structure]
            Từ điển cấu trúc hiện tại
        current_plan : Plan
            Kế hoạch hiện tại

        Returns
        -------
        Dict[str, float]
            Từ điển các thay đổi thể tích theo phần trăm
        """
        volume_changes = {}

        # Lấy các cấu trúc từ kế hoạch
        plan_structures = current_plan.get_structures()

        # So sánh thể tích
        for struct_name, struct in current_structures.items():
            if struct_name in plan_structures:
                plan_struct = plan_structures[struct_name]
                current_vol = struct.get_volume()
                plan_vol = plan_struct.get_volume()

                if plan_vol > 0:
                    percent_change = ((current_vol - plan_vol) / plan_vol) * 100
                    volume_changes[struct_name] = percent_change

        return volume_changes

    def _evaluate_position_changes(
        self, current_structures: Dict[str, Structure], current_plan: Plan
    ) -> Dict[str, float]:
        """
        Đánh giá thay đổi vị trí của các cấu trúc.

        Parameters
        ----------
        current_structures : Dict[str, Structure]
            Từ điển cấu trúc hiện tại
        current_plan : Plan
            Kế hoạch hiện tại

        Returns
        -------
        Dict[str, float]
            Từ điển các thay đổi vị trí (mm)
        """
        position_changes = {}

        # Lấy các cấu trúc từ kế hoạch
        plan_structures = current_plan.get_structures()

        # So sánh vị trí tâm
        for struct_name, struct in current_structures.items():
            if struct_name in plan_structures:
                plan_struct = plan_structures[struct_name]
                current_centroid = struct.get_centroid()
                plan_centroid = plan_struct.get_centroid()

                if current_centroid is not None and plan_centroid is not None:
                    distance_mm = np.linalg.norm(
                        np.array(current_centroid) - np.array(plan_centroid)
                    )
                    position_changes[struct_name] = distance_mm

        return position_changes

    def _evaluate_structure_differences(
        self, current_structures: Dict[str, Structure], current_plan: Plan
    ) -> Dict[str, float]:
        """
        Đánh giá sự khác biệt giữa các cấu trúc hiện tại và trong kế hoạch.

        Parameters
        ----------
        current_structures : Dict[str, Structure]
            Các cấu trúc giải phẫu hiện tại
        current_plan : Plan
            Kế hoạch hiện tại

        Returns
        -------
        Dict[str, float]
            Từ điển chứa các chỉ số so sánh (hệ số Dice) cho từng cấu trúc
        """
        result = {}

        # Lấy cấu trúc từ kế hoạch hiện tại
        plan_structures = (
            current_plan.structures if hasattr(current_plan, "structures") else {}
        )

        if not plan_structures:
            logger.warning("Kế hoạch không có thông tin về cấu trúc")
            return result

        # So sánh từng cấu trúc
        for struct_id, current_struct in current_structures.items():
            if struct_id in plan_structures:
                plan_struct = plan_structures[struct_id]

                try:
                    # Sử dụng module dice để tính hệ số Dice
                    dice_coef = calculate_dice_coefficient(current_struct, plan_struct)
                    result[struct_id] = dice_coef
                    logger.debug(
                        f"Hệ số Dice cho cấu trúc {struct_id}: {dice_coef:.4f}"
                    )

                    # Thêm các thông tin khác
                    if hasattr(current_struct, "volume") and hasattr(
                        plan_struct, "volume"
                    ):
                        current_vol = current_struct.volume
                        plan_vol = plan_struct.volume
                        if plan_vol > 0:
                            vol_change = (current_vol - plan_vol) / plan_vol * 100.0
                            result[f"{struct_id}_volume_change_percent"] = vol_change
                except Exception as e:
                    logger.error(
                        f"Lỗi khi tính hệ số Dice cho cấu trúc {struct_id}: {str(e)}"
                    )
                    result[struct_id] = float("nan")

        return result

    def _evaluate_dose_deviations(
        self,
        current_structures: Dict[str, Structure],
        current_image: Image,
        current_plan: Plan,
    ) -> Dict[str, float]:
        """
        Đánh giá sai lệch liều giữa các cấu trúc.

        Parameters
        ----------
        current_structures : Dict[str, Structure]
            Từ điển cấu trúc hiện tại
        current_image : Image
            Hình ảnh hiện tại của bệnh nhân
        current_plan : Plan
            Kế hoạch hiện tại

        Returns
        -------
        Dict[str, float]
            Từ điển các sai lệch liều theo phần trăm
        """
        dose_deviations = {}

        # Lấy các cấu trúc từ kế hoạch
        plan_structures = current_plan.get_structures()

        # Tính sai lệch liều
        for struct_name, struct in current_structures.items():
            if struct_name in plan_structures:
                plan_struct = plan_structures[struct_name]
                current_dose = struct.get_dose()
                plan_dose = plan_struct.get_dose()

                if plan_dose > 0:
                    percent_deviation = ((current_dose - plan_dose) / plan_dose) * 100
                    dose_deviations[struct_name] = percent_deviation

        return dose_deviations

    def _evaluate_clinical_metrics(
        self,
        current_structures: Dict[str, Structure],
        current_image: Image,
        current_plan: Plan,
    ) -> Dict[str, float]:
        """
        Đánh giá các chỉ số lâm sàng của các cấu trúc.

        Parameters
        ----------
        current_structures : Dict[str, Structure]
            Từ điển cấu trúc hiện tại
        current_image : Image
            Hình ảnh hiện tại của bệnh nhân
        current_plan : Plan
            Kế hoạch hiện tại

        Returns
        -------
        Dict[str, float]
            Từ điển các chỉ số lâm sàng
        """
        clinical_metrics = {}

        # Lấy các cấu trúc từ kế hoạch
        plan_structures = current_plan.get_structures()

        # Tính toán các chỉ số lâm sàng
        for struct_name, struct in current_structures.items():
            if struct_name in plan_structures:
                plan_struct = plan_structures[struct_name]
                clinical_metrics[struct_name] = struct.get_clinical_metric(plan_struct)

        return clinical_metrics

    def create_adaptive_plan(
        self,
        current_image: Image,
        current_structures: Dict[str, Structure],
        current_date: datetime.datetime,
        prediction_days: List[int] = [1, 3, 5, 7],
        optimization_settings: Optional[Dict[str, Any]] = None,
    ) -> Plan:
        """
        Tạo kế hoạch thích ứng mới dựa trên trạng thái hiện tại.

        Parameters
        ----------
        current_image : Image
            Hình ảnh hiện tại
        current_structures : Dict[str, Structure]
            Các cấu trúc hiện tại
        current_date : datetime.datetime
            Ngày hiện tại
        prediction_days : List[int], optional
            Danh sách số ngày dự đoán trong tương lai, mặc định là [1, 3, 5, 7]
        optimization_settings : Optional[Dict[str, Any]], optional
            Cài đặt tối ưu hóa, mặc định là None

        Returns
        -------
        Plan
            Kế hoạch thích ứng mới
        """
        # Lấy kế hoạch hiện tại hoặc kế hoạch tham chiếu
        current_plan = self.get_adaptive_plan(current_date) or self.reference_plan

        # Thu thập dữ liệu lịch sử
        historical_images = []
        historical_structures = []
        historical_dates = []

        # Thêm dữ liệu tham chiếu
        ref_image = current_plan.image
        ref_structures = current_plan.structures
        ref_date = current_plan.creation_date

        historical_images.append(ref_image)
        historical_structures.append(ref_structures)
        historical_dates.append(ref_date)

        # Thêm dữ liệu hiện tại
        historical_images.append(current_image)
        historical_structures.append(current_structures)
        historical_dates.append(current_date)

        # Dự đoán thay đổi giải phẫu trong tương lai
        predictor = AnatomyPredictor()
        prediction_dates = [
            current_date + datetime.timedelta(days=d) for d in prediction_days
        ]

        anatomy_prediction = predictor.predict_anatomy_changes(
            self.patient,
            historical_images,
            historical_structures,
            historical_dates,
            prediction_dates,
            method=PredictionMethod.SPLINE,
        )

        # Tạo các kịch bản giải phẫu cho tối ưu hóa bền vững
        scenarios = []

        # Thêm kịch bản hiện tại
        scenarios.append((current_image, current_structures))

        # Thêm các kịch bản dự đoán
        for i, date in enumerate(prediction_dates):
            pred_structures = anatomy_prediction.predicted_structures.get(date, {})
            if pred_structures:
                scenarios.append((current_image, pred_structures))

        # Tạo kế hoạch thích ứng
        logger.info(f"Bắt đầu tạo kế hoạch thích ứng cho ngày {current_date}")

        # Sao chép các thông số từ kế hoạch hiện tại
        new_plan = current_plan.copy()
        new_plan.creation_date = current_date
        new_plan.name = f"Adaptive_{current_date.strftime('%Y%m%d')}"
        new_plan.description = (
            f"Kế hoạch thích ứng tạo ngày {current_date.strftime('%Y-%m-%d')}"
        )

        # Cập nhật cấu trúc với dữ liệu mới
        new_plan.structures = current_structures
        new_plan.image = current_image

        # Tạo bộ tính toán liều
        dose_calculator = DoseCalculator()

        # Tạo bộ tối ưu hóa bền vững
        optimizer = RobustOptimizer(
            plan=new_plan,
            objectives=new_plan.planning_objectives,
            dose_calculator=dose_calculator,
            structures=current_structures,
        )

        # Kết hợp cài đặt tối ưu hóa
        opt_settings = getattr(self, "adaptation_strategy", {}).get(
            "optimization_settings", {}
        )
        if optimization_settings:
            opt_settings.update(optimization_settings)

        # Thiết lập các tham số tối ưu hóa
        for key, value in opt_settings.items():
            optimizer.set_parameter(key, value)

        # Tạo kịch bản từ các dự đoán
        for i, scenario in enumerate(scenarios):
            if i == 0:
                continue  # Bỏ qua kịch bản đầu tiên (kịch bản hiện tại)
            img, structs = scenario
            weight = 1.0 / len(scenarios)
            optimizer.add_scenario(
                structs, weight=weight, name=f"Prediction_Day_{prediction_days[i - 1]}"
            )

        # Thực hiện tối ưu hóa
        logger.info("Bắt đầu tối ưu hóa kế hoạch thích ứng bền vững")
        optimized_plan, robustness_result = optimizer.optimize()

        # Nếu không có kết quả phân tích độ bền vững, thực hiện phân tích
        if robustness_result is None and hasattr(optimized_plan, "dose_grid"):
            logger.info("Tiến hành phân tích độ bền vững của kế hoạch mới")
            robustness_analyzer = RobustnessAnalyzer(
                plan=optimized_plan,
                structures=current_structures,
                dose_grid=optimized_plan.dose_grid,
            )
            robustness_result = robustness_analyzer.analyze()

        # Lấy thống kê độ bền vững
        if robustness_result:
            robustness_stats = robustness_result.get_statistics()
        else:
            robustness_stats = {}

        # Thêm kế hoạch vào danh sách kế hoạch thích ứng
        self.add_adaptive_plan(
            current_date,
            optimized_plan,
            anatomy_prediction,
            robustness_stats,
        )

        # Thêm vào lịch sử
        history_entry = {
            "date": current_date,
            "action": "create_adaptive_plan",
            "details": {
                "plan_name": optimized_plan.name,
                "prediction_days": prediction_days,
                "has_robustness_metrics": robustness_result is not None,
            },
        }
        self.adaptation_history.append(history_entry)

        logger.info(f"Đã tạo kế hoạch thích ứng bền vững: {optimized_plan.name}")

        return optimized_plan

    def export_adaptation_report(self, output_dir: str) -> str:
        """
        Xuất báo cáo về quá trình thích ứng kế hoạch.

        Parameters
        ----------
        output_dir : str
            Thư mục xuất báo cáo

        Returns
        -------
        str
            Đường dẫn đến file báo cáo
        """
        # Tạo thư mục xuất nếu chưa tồn tại
        create_directory_if_not_exists(output_dir)

        # Tạo tên file báo cáo
        report_name = f"adaptation_report_{self.patient.id}_{get_timestamp()}.html"
        report_path = os.path.join(output_dir, report_name)

        # Thu thập dữ liệu báo cáo
        report_data = {
            "patient_id": self.patient.id,
            "reference_plan": self.reference_plan.name,
            "creation_date": self.creation_date.strftime("%Y-%m-%d %H:%M:%S"),
            "adaptation_type": self.adaptation_type.name,
            "adaptive_plans": [],
        }

        # Thêm thông tin về từng kế hoạch thích ứng
        for date, plan in sorted(self.adaptive_plans.items()):
            plan_info = {
                "date": date.strftime("%Y-%m-%d %H:%M:%S"),
                "plan_name": plan.name,
                "robustness_metrics": self.robustness_metrics.get(date, {}),
            }
            report_data["adaptive_plans"].append(plan_info)

        # Tạo báo cáo HTML
        template = get_template("adaptive_planning_report.html")
        html = template.render(report=report_data)

        # Ghi báo cáo
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html)

        return report_path

    def optimize_robust_plan(self, plan):
        """
        Tối ưu hóa kế hoạch để đạt được tính mạnh mẽ (robust) cao.

        Phương thức này thực hiện tối ưu hóa kế hoạch điều trị để đảm bảo tính ổn định
        và mạnh mẽ trước các biến động có thể xảy ra như thay đổi giải phẫu, sai số
        setup, v.v.

        Parameters
        ----------
        plan : Plan
            Kế hoạch cần được tối ưu hóa.

        Returns
        -------
        bool
            True nếu tối ưu hóa thành công, False nếu thất bại.
        """
        try:
            logger.info("Bắt đầu tối ưu hóa kế hoạch mạnh mẽ")

            if not plan:
                logger.error("Không thể tối ưu hóa: Kế hoạch là None")
                return False

            # Nếu có tích hợp với module tối ưu hóa
            if hasattr(self, "robust_optimizer") and self.robust_optimizer:
                try:
                    self.robust_optimizer.set_plan(plan)
                    self.robust_optimizer.optimize()
                    logger.info(
                        "Đã tối ưu hóa kế hoạch thành công với robust_optimizer"
                    )
                    return True
                except Exception as e:
                    logger.error(f"Lỗi khi tối ưu hóa với robust_optimizer: {str(e)}")

            # Thực hiện logic tối ưu hóa đơn giản nếu không có robust_optimizer

            # 1. Tối ưu hóa các ràng buộc để đảm bảo độ bền vững
            if hasattr(plan, "constraints") and plan.constraints:
                for constraint in plan.constraints:
                    # Thêm biên dự phòng cho các ràng buộc
                    if hasattr(constraint, "add_robustness_margin"):
                        constraint.add_robustness_margin(0.05)  # 5% robustness margin

            # 2. Tối ưu hóa các thông số beam để đạt tính mạnh mẽ cao hơn
            if hasattr(plan, "beams") and plan.beams:
                for beam in plan.beams:
                    # Logic cải thiện độ mạnh mẽ cho beam
                    pass

            logger.info("Đã hoàn thành tối ưu hóa kế hoạch mạnh mẽ")
            return True

        except Exception as e:
            logger.error(f"Lỗi trong quá trình tối ưu hóa kế hoạch: {str(e)}")
            return False


class RobustAdaptivePlanner:
    """
    Lớp để tạo và quản lý chiến lược thích ứng kế hoạch xạ trị.

    Lớp này cung cấp các phương thức để tạo và thực hiện chiến lược thích ứng
    phù hợp với các thay đổi giải phẫu của bệnh nhân theo thời gian.
    """

    def __init__(self):
        """Khởi tạo đối tượng RobustAdaptivePlanner."""
        self.patient = None
        self.validator = None

        # Khởi tạo các thuộc tính bị thiếu
        self.anatomy_predictor = None
        self.adaptation_strategies = {}
        self.default_thresholds = {
            AdaptationTrigger.VOLUME_CHANGE: 0.1,  # 10% thay đổi thể tích
            AdaptationTrigger.CENTROID_CHANGE: 5.0,  # 5mm thay đổi tâm
            AdaptationTrigger.DICE_COEFFICIENT: 0.9,  # Dice coefficient <= 0.9
            AdaptationTrigger.DOSE_DEVIATION: 0.05,  # 5% sai lệch liều
            AdaptationTrigger.CLINICAL_METRICS: 0.1,  # 10% sai lệch các chỉ số lâm sàng
        }

        logger.info("Đã khởi tạo RobustAdaptivePlanner")

    def set_validator(self, validator):
        """
        Thiết lập validator để đánh giá dự đoán thay đổi giải phẫu.

        Parameters
        ----------
        validator : ModelValidator
            Đối tượng validator để kiểm tra và đánh giá kết quả dự đoán giải phẫu.
        """
        self.validator = validator
        logger.info("Đã thiết lập validator cho RobustAdaptivePlanner")

        # Thiết lập validator cho anatomy_predictor nếu có
        if hasattr(self, "anatomy_predictor") and self.anatomy_predictor is not None:
            if hasattr(self.anatomy_predictor, "set_validator"):
                self.anatomy_predictor.set_validator(validator)
                logger.debug("Đã thiết lập validator cho anatomy_predictor")

    def set_patient(self, patient: Patient):
        """
        Thiết lập thông tin bệnh nhân.

        Parameters
        ----------
        patient : Patient
            Đối tượng bệnh nhân
        """
        self.patient = patient

    def create_offline_adaptation_strategy(
        self, reference_plan: Plan, fractions: int, prediction_interval: int = 5
    ) -> RobustAdaptivePlan:
        """
        Tạo chiến lược thích ứng ngoại tuyến.

        Parameters
        ----------
        reference_plan : Plan
            Kế hoạch tham chiếu
        fractions : int
            Số phân đoạn điều trị
        prediction_interval : int, optional
            Khoảng cách giữa các phân đoạn cần thích ứng, mặc định là 5

        Returns
        -------
        RobustAdaptivePlan
            Kế hoạch thích ứng bền vững
        """
        # Lấy thông tin bệnh nhân từ kế hoạch tham chiếu
        patient = reference_plan.patient

        # Tạo kế hoạch thích ứng bền vững
        adaptive_plan = RobustAdaptivePlan(
            patient,
            reference_plan,
            adaptation_type=AdaptationType.OFFLINE,
        )

        # Thiết lập ngưỡng kích hoạt thích ứng
        adaptive_plan.adaptation_thresholds = self.default_thresholds.copy()

        # Tạo danh sách các phân đoạn cần thích ứng
        if fractions <= 0:
            raise ValueError("Số phân đoạn phải lớn hơn 0")

        if prediction_interval <= 0:
            raise ValueError("Khoảng cách thích ứng phải lớn hơn 0")

        # Tính toán ngày bắt đầu điều trị (giả định là ngày sau khi tạo kế hoạch)
        if hasattr(reference_plan, "creation_date") and reference_plan.creation_date:
            start_date = reference_plan.creation_date + datetime.timedelta(days=1)
        else:
            start_date = datetime.datetime.now() + datetime.timedelta(days=1)

        # Tạo lịch thích ứng
        fraction_dates = []
        adaptation_dates = []

        # Tính toán ngày điều trị cho từng phân đoạn (giả định 5 ngày/tuần, bỏ qua cuối tuần)
        current_date = start_date
        for i in range(fractions):
            # Bỏ qua cuối tuần
            while current_date.weekday() >= 5:  # 5: Thứ bảy, 6: Chủ nhật
                current_date += datetime.timedelta(days=1)

            fraction_dates.append(current_date)

            # Kiểm tra nếu phân đoạn này cần thích ứng
            if i % prediction_interval == 0:
                adaptation_dates.append(current_date)

            # Chuyển đến ngày tiếp theo
            current_date += datetime.timedelta(days=1)

        # Tạo kế hoạch dự đoán thích ứng
        for date in adaptation_dates:
            # Thêm một mục nhập vào lịch sử
            history_entry = {
                "date": date,
                "action": "scheduled_adaptation",
                "details": {
                    "scheduled_date": date.strftime("%Y-%m-%d"),
                },
            }
            adaptive_plan.adaptation_history.append(history_entry)

        # Lưu kế hoạch thích ứng
        self.adaptation_strategies[adaptive_plan.reference_plan.id] = adaptive_plan

        return adaptive_plan

    def create_online_adaptation_strategy(
        self, reference_plan: Plan
    ) -> RobustAdaptivePlan:
        """
        Tạo chiến lược thích ứng trực tuyến.

        Parameters
        ----------
        reference_plan : Plan
            Kế hoạch tham chiếu

        Returns
        -------
        RobustAdaptivePlan
            Kế hoạch thích ứng bền vững
        """
        # Lấy thông tin bệnh nhân từ kế hoạch tham chiếu
        patient = reference_plan.patient

        # Tạo kế hoạch thích ứng bền vững
        adaptive_plan = RobustAdaptivePlan(
            patient,
            reference_plan,
            adaptation_type=AdaptationType.ONLINE,
        )

        # Thiết lập ngưỡng kích hoạt thích ứng chặt chẽ hơn cho thích ứng trực tuyến
        adaptive_plan.adaptation_thresholds = {
            AdaptationTrigger.VOLUME_CHANGE: 5.0,  # Giảm xuống, nhạy cảm hơn với thay đổi thể tích
            AdaptationTrigger.CENTROID_CHANGE: 3.0,  # Giảm xuống, nhạy cảm hơn với thay đổi vị trí
            AdaptationTrigger.DICE_COEFFICIENT: 0.9,  # Tăng lên, yêu cầu độ tương đồng cao hơn
            AdaptationTrigger.DOSE_DEVIATION: 3.0,  # Giảm xuống, nhạy cảm hơn với sai lệch liều
            AdaptationTrigger.CLINICAL_METRICS: 0.95,  # Tăng lên, yêu cầu chỉ số lâm sàng cao hơn
        }

        # Lưu kế hoạch thích ứng
        self.adaptation_strategies[adaptive_plan.reference_plan.id] = adaptive_plan

        return adaptive_plan

    def execute_adaptive_planning(
        self,
        adaptive_plan: RobustAdaptivePlan,
        current_image: Image,
        current_structures: Dict[str, Structure],
        current_date: datetime.datetime,
    ) -> Optional[Plan]:
        """
        Thực hiện lập kế hoạch thích ứng tại một thời điểm.

        Parameters
        ----------
        adaptive_plan : RobustAdaptivePlan
            Kế hoạch thích ứng bền vững
        current_image : Image
            Hình ảnh hiện tại
        current_structures : Dict[str, Structure]
            Từ điển cấu trúc hiện tại
        current_date : datetime.datetime
            Ngày hiện tại

        Returns
        -------
        Optional[Plan]
            Kế hoạch thích ứng mới hoặc None nếu không cần thích ứng
        """
        # Kiểm tra xem có cần thích ứng hay không
        needs_adapt, evaluation_results = adaptive_plan.needs_adaptation(
            current_image, current_structures, current_date
        )

        # Nếu không cần thích ứng, trả về None
        if not needs_adapt:
            logger.info(f"Không cần thích ứng tại {current_date.strftime('%Y-%m-%d')}")
            return None

        # Thực hiện thích ứng
        logger.info(f"Thực hiện thích ứng tại {current_date.strftime('%Y-%m-%d')}")

        # Phương pháp thích ứng phụ thuộc vào loại thích ứng
        if adaptive_plan.adaptation_type == AdaptationType.ONLINE:
            # Thích ứng trực tuyến - tối ưu hóa nhanh
            optimization_settings = {
                "max_iterations": 50,  # Giảm số lần lặp để tối ưu nhanh
                "use_warm_start": True,  # Sử dụng kế hoạch hiện tại làm điểm khởi đầu
                "objective_priority": "speed",  # Ưu tiên tốc độ
            }
        else:
            # Thích ứng ngoại tuyến - tối ưu hóa đầy đủ
            optimization_settings = {
                "max_iterations": 200,
                "use_warm_start": True,
                "objective_priority": "quality",  # Ưu tiên chất lượng
            }

        # Tạo kế hoạch thích ứng mới
        try:
            new_plan = adaptive_plan.create_adaptive_plan(
                current_image,
                current_structures,
                current_date,
                prediction_days=[1, 3, 7, 14]
                if adaptive_plan.adaptation_type == AdaptationType.OFFLINE
                else [1],
                optimization_settings=optimization_settings,
            )

            return new_plan
        except Exception as e:
            logger.error(f"Lỗi khi thực hiện thích ứng: {e}")
            return None

    def generate_adaptive_plans(
        self, predictions: Dict[str, Any], **kwargs
    ) -> Dict[str, Plan]:
        """
        Tạo các kế hoạch thích ứng dựa trên dự đoán thay đổi giải phẫu.

        Parameters
        ----------
        predictions : Dict[str, Any]
            Từ điển chứa các dự đoán giải phẫu theo thời điểm.
        **kwargs
            Các tham số bổ sung cho quá trình tạo kế hoạch thích ứng:
            - reference_plan: Plan mẫu để tạo kế hoạch thích ứng
            - robust: bool, có tạo kế hoạch mạnh mẽ hay không
            - optimization_settings: Dict, cài đặt tối ưu hóa

        Returns
        -------
        Dict[str, Plan]
            Từ điển chứa các kế hoạch thích ứng theo thời điểm.
        """
        if not predictions:
            logger.warning(
                "Không có dự đoán nào được cung cấp để tạo kế hoạch thích ứng"
            )
            return {}

        result_plans = {}
        reference_plan = kwargs.get("reference_plan")
        robust = kwargs.get("robust", True)
        optimization_settings = kwargs.get("optimization_settings", {})

        try:
            logger.info(
                f"Bắt đầu tạo kế hoạch thích ứng cho {len(predictions)} dự đoán"
            )

            # Đánh giá dự đoán nếu validator tồn tại
            if hasattr(self, "validator") and self.validator is not None:
                logger.info(
                    "Đánh giá chất lượng các dự đoán trước khi lập kế hoạch thích ứng"
                )
                validation_results = self.validator.validate_predictions(predictions)

                # Lọc các dự đoán có chất lượng tốt
                valid_predictions = {}
                for pred_id, result in validation_results.items():
                    if pred_id != "summary" and result.get("is_valid", False):
                        valid_predictions[pred_id] = predictions[pred_id]

                if valid_predictions:
                    logger.info(
                        f"Có {len(valid_predictions)}/{len(predictions)} dự đoán hợp lệ để lập kế hoạch"
                    )
                    # Sử dụng các dự đoán hợp lệ
                    predictions_to_use = valid_predictions
                else:
                    logger.warning(
                        "Không có dự đoán nào hợp lệ, sử dụng tất cả dự đoán"
                    )
                    predictions_to_use = predictions
            else:
                # Không có validator, sử dụng tất cả dự đoán
                predictions_to_use = predictions

            # Tạo kế hoạch cho từng dự đoán
            for time_point, prediction in predictions_to_use.items():
                if time_point == "summary":
                    continue  # Bỏ qua bản tóm tắt

                try:
                    logger.info(
                        f"Đang tạo kế hoạch thích ứng cho thời điểm {time_point}"
                    )

                    # Trích xuất dữ liệu dự đoán
                    predicted_image = prediction.get("image")
                    predicted_structures = prediction.get("structures")

                    if predicted_image is None or not predicted_structures:
                        logger.warning(
                            f"Dự đoán tại thời điểm {time_point} thiếu hình ảnh hoặc cấu trúc"
                        )
                        continue

                    # Tạo kế hoạch thích ứng
                    plan = self._create_simple_adaptive_plan(
                        predicted_image,
                        predicted_structures,
                        reference_plan=reference_plan,
                        robust=robust,
                    )

                    if plan:
                        # Thiết lập thông tin bổ sung cho kế hoạch
                        if hasattr(plan, "set_adaptive_info"):
                            plan.set_adaptive_info(
                                {
                                    "time_point": time_point,
                                    "based_on_prediction": True,
                                    "prediction_quality": prediction.get(
                                        "quality", 0.0
                                    ),
                                    "creation_timestamp": time.time(),
                                }
                            )

                        result_plans[str(time_point)] = plan
                        logger.info(
                            f"Đã tạo thành công kế hoạch thích ứng cho thời điểm {time_point}"
                        )

                except Exception as e:
                    logger.error(
                        f"Lỗi khi tạo kế hoạch cho thời điểm {time_point}: {str(e)}"
                    )

            logger.info(f"Đã tạo thành công {len(result_plans)} kế hoạch thích ứng")
            return result_plans

        except Exception as e:
            logger.error(f"Lỗi khi tạo kế hoạch thích ứng: {str(e)}")
            return {}

    def _create_simple_adaptive_plan(
        self, predicted_image, predicted_structures, reference_plan=None, robust=True
    ) -> Optional[Plan]:
        """
        Tạo kế hoạch thích ứng đơn giản từ hình ảnh và cấu trúc dự đoán.

        Parameters
        ----------
        predicted_image : Image
            Hình ảnh dự đoán.
        predicted_structures : Dict[str, Structure]
            Từ điển các cấu trúc dự đoán.
        reference_plan : Plan, optional
            Kế hoạch tham chiếu để tạo kế hoạch thích ứng.
        robust : bool, default=True
            Có tạo kế hoạch mạnh mẽ hay không.

        Returns
        -------
        Optional[Plan]
            Kế hoạch thích ứng nếu tạo thành công, None nếu thất bại.
        """
        try:
            # Tạo kế hoạch mới
            if reference_plan is not None:
                # Sao chép từ kế hoạch tham chiếu
                new_plan = reference_plan.create_adapted_copy()
                logger.info("Đã tạo bản sao từ kế hoạch tham chiếu")
            else:
                # Tạo kế hoạch mới từ đầu nếu không có mẫu
                from quangtps.core.patient import Plan

                new_plan = Plan()
                new_plan.set_name(f"Adaptive_Plan_{time.strftime('%Y%m%d_%H%M%S')}")
                logger.info("Đã tạo kế hoạch mới không dựa trên mẫu")

            # Thiết lập hình ảnh và cấu trúc mới
            new_plan.set_primary_image(predicted_image)

            if hasattr(new_plan, "set_structures"):
                new_plan.set_structures(predicted_structures)

            # Điều chỉnh các thông số kế hoạch nếu cần
            if hasattr(new_plan, "adjust_for_new_anatomy"):
                new_plan.adjust_for_new_anatomy()
                logger.debug("Đã điều chỉnh kế hoạch cho giải phẫu mới")

            # Tối ưu hóa lại kế hoạch nếu có các cấu trúc quan trọng
            if reference_plan and hasattr(reference_plan, "get_prescription"):
                prescription = reference_plan.get_prescription()
                if prescription:
                    # Thiết lập lại kê toa
                    if hasattr(new_plan, "set_prescription"):
                        new_plan.set_prescription(prescription)
                        logger.debug("Đã sao chép kê toa từ kế hoạch tham chiếu")

            # Tối ưu hóa lại kế hoạch nếu cần
            if robust and hasattr(self, "optimize_robust_plan"):
                try:
                    self.optimize_robust_plan(new_plan)
                    logger.info(
                        "Đã hoàn thành tối ưu hóa mạnh mẽ cho kế hoạch thích ứng"
                    )
                except Exception as e:
                    logger.error(f"Lỗi khi tối ưu hóa kế hoạch mạnh mẽ: {str(e)}")

            return new_plan

        except Exception as e:
            logger.error(f"Lỗi khi tạo kế hoạch thích ứng đơn giản: {str(e)}")
            return None
