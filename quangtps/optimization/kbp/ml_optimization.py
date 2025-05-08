#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module tối ưu hóa dựa trên học máy.

Module này cung cấp các thuật toán tối ưu hóa dựa trên học máy để tự động
tạo và cải thiện kế hoạch xạ trị dựa trên dữ liệu lịch sử.
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
import pickle
import joblib

from quangtps.optimization.kbp.model import KBPModel
from quangtps.optimization.kbp.predictor import KBPPredictor
from quangtps.optimization.optimizer import Optimizer
from quangtps.planning.plan import Plan
from quangtps.structures.roi import ROI
from quangtps.dose.dose_grid import DoseGrid
from quangtps.evaluation.dvh.dvh_calculator import DVHCalculator
from quangtps.evaluation.metrics.quality_metrics import calculate_plan_metrics

logger = logging.getLogger(__name__)


class MLOptimizer:
    """
    Tối ưu hóa kế hoạch xạ trị dựa trên học máy.

    Lớp này sử dụng mô hình học máy để tự động điều chỉnh các mục tiêu tối ưu hóa
    dựa trên phản hồi từ quá trình tối ưu và kết quả kế hoạch trung gian.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        kbp_model: Optional[KBPModel] = None,
        kbp_predictor: Optional[KBPPredictor] = None,
        max_iterations: int = 5,
        convergence_threshold: float = 0.01,
    ):
        """
        Khởi tạo tối ưu hóa ML.

        Parameters
        ----------
        optimizer : Optimizer
            Bộ tối ưu hóa cơ bản để sử dụng
        kbp_model : Optional[KBPModel], optional
            Mô hình KBP để dự đoán mục tiêu, mặc định là None
        kbp_predictor : Optional[KBPPredictor], optional
            Bộ dự đoán KBP, mặc định là None
        max_iterations : int, optional
            Số lần lặp tối đa, mặc định là 5
        convergence_threshold : float, optional
            Ngưỡng hội tụ, mặc định là 0.01
        """
        self.optimizer = optimizer
        self.kbp_model = kbp_model
        self.kbp_predictor = kbp_predictor
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold

        # Trạng thái và lịch sử
        self.iteration = 0
        self.history = []
        self.metrics_history = []
        self.current_score = 0.0
        self.best_score = 0.0
        self.best_objectives = None
        self.best_constraints = None

        # Callback cho theo dõi tiến độ
        self._progress_callback = None

    def register_progress_callback(self, callback):
        """
        Đăng ký callback cho theo dõi tiến độ.

        Parameters
        ----------
        callback : Callable[[float, str], None]
            Callback theo dõi tiến độ
        """
        self._progress_callback = callback

    def _report_progress(self, progress: float, message: str):
        """
        Báo cáo tiến độ thông qua callback.

        Parameters
        ----------
        progress : float
            Tiến độ từ 0 đến 1
        message : str
            Thông điệp tiến độ
        """
        if self._progress_callback:
            self._progress_callback(progress, message)

    def optimize(
        self,
        plan: Plan,
        objectives: List[Dict[str, Any]],
        constraints: List[Dict[str, Any]],
    ) -> Tuple[bool, Plan, List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Tối ưu hóa kế hoạch sử dụng phương pháp ML.

        Parameters
        ----------
        plan : Plan
            Kế hoạch cần tối ưu hóa
        objectives : List[Dict[str, Any]]
            Mục tiêu tối ưu hóa ban đầu
        constraints : List[Dict[str, Any]]
            Ràng buộc tối ưu hóa ban đầu

        Returns
        -------
        Tuple[bool, Plan, List[Dict[str, Any]], List[Dict[str, Any]]]
            Tuple chứa (thành công, kế hoạch, mục tiêu tối ưu, ràng buộc tối ưu)
        """
        self.iteration = 0
        self.history = []
        self.metrics_history = []
        self.best_score = 0.0
        self.best_objectives = objectives.copy()
        self.best_constraints = constraints.copy()

        current_objectives = objectives.copy()
        current_constraints = constraints.copy()

        success = False

        for i in range(self.max_iterations):
            self.iteration = i + 1
            progress_base = i / self.max_iterations

            # Báo cáo tiến độ
            self._report_progress(
                progress_base, f"Bắt đầu lần lặp ML {i + 1}/{self.max_iterations}"
            )

            # Chạy tối ưu hóa với mục tiêu và ràng buộc hiện tại
            optimization_success, optimized_plan = self._run_optimization(
                plan,
                current_objectives,
                current_constraints,
                progress_base,
                progress_base + 0.8 / self.max_iterations,
            )

            if not optimization_success:
                logger.warning(f"Lần lặp {i + 1}: Tối ưu hóa không thành công")
                # Nếu lần đầu tiên không thành công, chúng ta cần dừng
                if i == 0:
                    return False, plan, objectives, constraints
                # Nếu không, tiếp tục với mục tiêu tiếp theo
                continue

            # Đánh giá kế hoạch tối ưu hóa
            plan_metrics = self._evaluate_plan(optimized_plan)
            self.metrics_history.append(plan_metrics)

            # Tính điểm cho kế hoạch
            score = self._calculate_plan_score(plan_metrics)
            self.current_score = score
            self.history.append(
                {
                    "iteration": i + 1,
                    "score": score,
                    "objectives": current_objectives.copy(),
                    "constraints": current_constraints.copy(),
                    "metrics": plan_metrics,
                }
            )

            # Kiểm tra nếu chúng ta có kế hoạch tốt hơn
            if score > self.best_score:
                self.best_score = score
                self.best_objectives = current_objectives.copy()
                self.best_constraints = current_constraints.copy()
                # Cập nhật kế hoạch
                plan = optimized_plan
                success = True

            # Kiểm tra hội tụ
            if (
                i > 0
                and abs(score - self.history[-2]["score"]) < self.convergence_threshold
            ):
                logger.info(f"Hội tụ đạt được sau {i + 1} lần lặp")
                break

            # Cập nhật mục tiêu dựa trên kết quả
            if i < self.max_iterations - 1:  # Không cập nhật trong lần lặp cuối cùng
                current_objectives, current_constraints = self._update_objectives(
                    optimized_plan,
                    current_objectives,
                    current_constraints,
                    plan_metrics,
                )

            # Báo cáo tiến độ
            self._report_progress(
                progress_base + 1.0 / self.max_iterations,
                f"Hoàn thành lần lặp ML {i + 1}/{self.max_iterations}, điểm: {score:.4f}",
            )

        # Báo cáo hoàn thành
        self._report_progress(
            1.0,
            f"Hoàn thành tối ưu hóa ML sau {self.iteration} lần lặp, điểm tốt nhất: {self.best_score:.4f}",
        )

        return success, plan, self.best_objectives, self.best_constraints

    def _run_optimization(
        self,
        plan: Plan,
        objectives: List[Dict[str, Any]],
        constraints: List[Dict[str, Any]],
        progress_start: float,
        progress_end: float,
    ) -> Tuple[bool, Plan]:
        """
        Chạy tối ưu hóa với mục tiêu và ràng buộc cụ thể.

        Parameters
        ----------
        plan : Plan
            Kế hoạch cần tối ưu hóa
        objectives : List[Dict[str, Any]]
            Mục tiêu tối ưu hóa
        constraints : List[Dict[str, Any]]
            Ràng buộc tối ưu hóa
        progress_start : float
            Tiến độ bắt đầu
        progress_end : float
            Tiến độ kết thúc

        Returns
        -------
        Tuple[bool, Plan]
            Tuple chứa (thành công, kế hoạch tối ưu)
        """
        # Tạo bản sao kế hoạch để không ảnh hưởng đến kế hoạch ban đầu
        plan_copy = plan.copy()

        # Cấu hình tối ưu hóa
        self.optimizer.set_objectives(objectives)
        self.optimizer.set_constraints(constraints)

        # Đăng ký callback để theo dõi tiến độ tối ưu hóa
        def optimization_progress(progress, message):
            adjusted_progress = (
                progress_start + (progress_end - progress_start) * progress
            )
            self._report_progress(adjusted_progress, message)

        self.optimizer.register_progress_callback(optimization_progress)

        # Chạy tối ưu hóa
        success = self.optimizer.optimize(plan_copy)

        return success, plan_copy

    def _evaluate_plan(self, plan: Plan) -> Dict[str, Any]:
        """
        Đánh giá kế hoạch và tính các chỉ số chất lượng.

        Parameters
        ----------
        plan : Plan
            Kế hoạch cần đánh giá

        Returns
        -------
        Dict[str, Any]
            Từ điển chứa chỉ số đánh giá
        """
        # Tính toán DVH
        dvh_calculator = DVHCalculator()
        dvh_results = dvh_calculator.calculate_dvh(plan)

        # Tính toán chỉ số chất lượng
        metrics = calculate_plan_metrics(plan, dvh_results)

        return metrics

    def _calculate_plan_score(self, metrics: Dict[str, Any]) -> float:
        """
        Tính điểm tổng hợp cho kế hoạch dựa trên các chỉ số.

        Parameters
        ----------
        metrics : Dict[str, Any]
            Từ điển chứa chỉ số đánh giá

        Returns
        -------
        float
            Điểm tổng hợp (0-1)
        """
        # Trọng số cho mỗi loại chỉ số
        weights = {
            "CI": 0.3,  # Conformity Index
            "HI": 0.3,  # Homogeneity Index
            "GI": 0.2,  # Gradient Index
            "OAR_sparing": 0.2,  # Mức độ bảo vệ cơ quan nguy cấp
        }

        score = 0.0

        # Điểm CI (càng gần 1 càng tốt)
        if "CI" in metrics:
            ci_score = max(0, 1 - abs(metrics["CI"] - 1))
            score += weights["CI"] * ci_score

        # Điểm HI (càng gần 0 càng tốt)
        if "HI" in metrics:
            hi_score = max(0, 1 - metrics["HI"])
            score += weights["HI"] * hi_score

        # Điểm GI (càng thấp càng tốt, thường <= 3 là tốt)
        if "GI" in metrics:
            gi_score = max(0, 1 - metrics["GI"] / 10)
            score += weights["GI"] * gi_score

        # Điểm OAR (phần trăm ràng buộc OAR được đáp ứng)
        if "oar_constraints_met" in metrics and "total_oar_constraints" in metrics:
            if metrics["total_oar_constraints"] > 0:
                oar_score = (
                    metrics["oar_constraints_met"] / metrics["total_oar_constraints"]
                )
                score += weights["OAR_sparing"] * oar_score

        return score

    def _update_objectives(
        self,
        plan: Plan,
        current_objectives: List[Dict[str, Any]],
        current_constraints: List[Dict[str, Any]],
        metrics: Dict[str, Any],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Cập nhật mục tiêu và ràng buộc dựa trên kết quả trung gian.

        Parameters
        ----------
        plan : Plan
            Kế hoạch hiện tại
        current_objectives : List[Dict[str, Any]]
            Mục tiêu hiện tại
        current_constraints : List[Dict[str, Any]]
            Ràng buộc hiện tại
        metrics : Dict[str, Any]
            Chỉ số đánh giá kế hoạch

        Returns
        -------
        Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]
            Tuple chứa (mục tiêu đã cập nhật, ràng buộc đã cập nhật)
        """
        # Sao chép để không ảnh hưởng đến đầu vào
        updated_objectives = current_objectives.copy()
        updated_constraints = current_constraints.copy()

        try:
            # Nếu có sẵn mô hình KBP, sử dụng nó để dự đoán điều chỉnh
            if self.kbp_model and self.kbp_predictor:
                # Trích xuất dữ liệu từ kế hoạch và DVH để dự đoán điều chỉnh
                patient_data = self._extract_features_for_prediction(plan, metrics)

                # Dự đoán điều chỉnh mục tiêu
                objective_adjustments = (
                    self.kbp_predictor.predict_objective_adjustments(
                        patient_data, current_objectives, metrics
                    )
                )

                # Áp dụng điều chỉnh
                for adjustment in objective_adjustments:
                    obj_idx = adjustment.get("objective_idx")
                    if obj_idx is not None and 0 <= obj_idx < len(updated_objectives):
                        for param, value in adjustment.get("adjustments", {}).items():
                            if param in updated_objectives[obj_idx]:
                                updated_objectives[obj_idx][param] = value
            else:
                # Chiến lược điều chỉnh đơn giản nếu không có mô hình KBP

                # 1. Điều chỉnh mục tiêu PTV nếu chỉ số CI không tốt
                if "CI" in metrics and abs(metrics["CI"] - 1.0) > 0.2:
                    # Tìm mục tiêu PTV
                    for i, obj in enumerate(updated_objectives):
                        if obj.get("structure_type") == "PTV":
                            # Tăng trọng số nếu CI quá cao (coverage không đủ)
                            if metrics["CI"] < 0.8:
                                updated_objectives[i]["weight"] = min(
                                    updated_objectives[i].get("weight", 1.0) * 1.2,
                                    100.0,
                                )
                            # Giảm trọng số nếu CI quá thấp (coverage quá mức)
                            elif metrics["CI"] > 1.2:
                                updated_objectives[i]["weight"] = max(
                                    updated_objectives[i].get("weight", 1.0) * 0.9, 0.1
                                )

                # 2. Điều chỉnh ràng buộc OAR nếu ràng buộc không được đáp ứng
                oar_constraints_met = metrics.get("oar_constraints_met", 0)
                total_oar_constraints = metrics.get("total_oar_constraints", 1)

                if oar_constraints_met / total_oar_constraints < 0.8:
                    # Tăng trọng số ràng buộc OAR không được đáp ứng
                    for i, constraint in enumerate(updated_constraints):
                        if constraint.get("structure_type") == "OAR":
                            constraint_id = constraint.get("id")
                            if constraint_id in metrics.get("unmet_constraints", []):
                                updated_constraints[i]["weight"] = min(
                                    updated_constraints[i].get("weight", 1.0) * 1.3,
                                    100.0,
                                )
        except Exception as e:
            logger.error(f"Lỗi khi cập nhật mục tiêu: {e}")

        return updated_objectives, updated_constraints

    def _extract_features_for_prediction(
        self, plan: Plan, metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Trích xuất đặc trưng từ kế hoạch và chỉ số để chuẩn bị dự đoán.

        Parameters
        ----------
        plan : Plan
            Kế hoạch hiện tại
        metrics : Dict[str, Any]
            Chỉ số đánh giá kế hoạch

        Returns
        -------
        Dict[str, Any]
            Đặc trưng để dự đoán điều chỉnh
        """
        features = {
            "metrics": metrics,
            "plan_type": plan.technique if hasattr(plan, "technique") else "UNKNOWN",
            "structures": {},
        }

        # Trích xuất thông tin cấu trúc
        if hasattr(plan, "structures"):
            for structure in plan.structures:
                structure_id = structure.id if hasattr(structure, "id") else "unknown"
                structure_type = (
                    structure.type if hasattr(structure, "type") else "unknown"
                )
                volume = structure.volume if hasattr(structure, "volume") else 0.0

                features["structures"][structure_id] = {
                    "type": structure_type,
                    "volume": volume,
                }

        # Trích xuất thông tin chùm tia
        if hasattr(plan, "beams"):
            features["num_beams"] = len(plan.beams)
            features["beam_modalities"] = [
                beam.modality if hasattr(beam, "modality") else "unknown"
                for beam in plan.beams
            ]

        return features

    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """
        Lấy lịch sử tối ưu hóa.

        Returns
        -------
        List[Dict[str, Any]]
            Lịch sử tối ưu hóa
        """
        return self.history

    def get_best_result(self) -> Dict[str, Any]:
        """
        Lấy kết quả tối ưu hóa tốt nhất.

        Returns
        -------
        Dict[str, Any]
            Kết quả tối ưu hóa tốt nhất
        """
        return {
            "score": self.best_score,
            "objectives": self.best_objectives,
            "constraints": self.best_constraints,
            "iteration": self.iteration,
        }
