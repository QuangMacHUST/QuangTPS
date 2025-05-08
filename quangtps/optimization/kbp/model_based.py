#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module tối ưu hóa dựa trên mô hình (model-based).

Module này cung cấp các thuật toán tối ưu hóa dựa trên mô hình để tự động
xác định tham số tối ưu cho kế hoạch xạ trị.
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from pathlib import Path
import time
import threading

from quangtps.optimization.kbp.model import KBPModel
from quangtps.planning.plan import Plan
from quangtps.structures.roi import ROI
from quangtps.dose.dose_grid import DoseGrid
from quangtps.evaluation.dvh.dvh_calculator import DVHCalculator
from quangtps.evaluation.metrics.quality_metrics import calculate_plan_metrics
from quangtps.optimization.optimizer import Optimizer

logger = logging.getLogger(__name__)


class ModelBasedOptimizer:
    """
    Tối ưu hóa dựa trên mô hình.

    Lớp này sử dụng mô hình để tự động xác định tham số tối ưu hóa
    dựa trên hình học bệnh nhân và mục tiêu lâm sàng.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        model: Optional[KBPModel] = None,
        model_path: Optional[str] = None,
        max_iterations: int = 3,
        refinement_learning_rate: float = 0.1,
        convergence_threshold: float = 0.01,
    ):
        """
        Khởi tạo tối ưu hóa dựa trên mô hình.

        Parameters
        ----------
        optimizer : Optimizer
            Bộ tối ưu hóa cơ bản
        model : Optional[KBPModel], optional
            Mô hình KBP, mặc định là None
        model_path : Optional[str], optional
            Đường dẫn đến mô hình được lưu, mặc định là None
        max_iterations : int, optional
            Số lần lặp tối đa cho điều chỉnh tinh, mặc định là 3
        refinement_learning_rate : float, optional
            Tốc độ học cho điều chỉnh tinh, mặc định là 0.1
        convergence_threshold : float, optional
            Ngưỡng hội tụ, mặc định là 0.01
        """
        self.optimizer = optimizer
        self.model = model
        self.max_iterations = max_iterations
        self.refinement_learning_rate = refinement_learning_rate
        self.convergence_threshold = convergence_threshold

        # Nạp mô hình nếu đường dẫn được cung cấp
        if model_path and not model:
            self.load_model(model_path)

        # Trạng thái và theo dõi
        self.is_optimizing = False
        self.current_progress = 0.0
        self.current_message = ""
        self._progress_callback = None
        self._stop_event = threading.Event()

        # Lịch sử tối ưu hóa
        self.history = []

    def load_model(self, model_path: str) -> bool:
        """
        Nạp mô hình từ đường dẫn.

        Parameters
        ----------
        model_path : str
            Đường dẫn đến mô hình được lưu

        Returns
        -------
        bool
            True nếu nạp thành công, False nếu không
        """
        try:
            if not os.path.exists(model_path):
                logger.error(f"Không tìm thấy mô hình tại: {model_path}")
                return False

            # Tạo mô hình mới và nạp từ đường dẫn
            self.model = KBPModel()
            if self.model.load(model_path):
                logger.info(f"Đã nạp mô hình thành công từ: {model_path}")
                return True
            else:
                logger.error(f"Không thể nạp mô hình từ: {model_path}")
                self.model = None
                return False
        except Exception as e:
            logger.error(f"Lỗi khi nạp mô hình: {e}")
            self.model = None
            return False

    def register_progress_callback(
        self, callback: Callable[[float, str], None]
    ) -> None:
        """
        Đăng ký callback cho theo dõi tiến độ.

        Parameters
        ----------
        callback : Callable[[float, str], None]
            Callback để báo cáo tiến độ
        """
        self._progress_callback = callback

    def _report_progress(self, progress: float, message: str) -> None:
        """
        Báo cáo tiến độ thông qua callback.

        Parameters
        ----------
        progress : float
            Tiến độ từ 0 đến 1
        message : str
            Thông điệp tiến độ
        """
        self.current_progress = progress
        self.current_message = message

        if self._progress_callback:
            self._progress_callback(progress, message)

    def stop_optimization(self) -> None:
        """Yêu cầu dừng quá trình tối ưu hóa đang chạy."""
        self._stop_event.set()
        self.optimizer.stop_optimization()

    def extract_patient_features(self, plan: Plan) -> Dict[str, Any]:
        """
        Trích xuất đặc trưng bệnh nhân từ kế hoạch.

        Parameters
        ----------
        plan : Plan
            Kế hoạch cần trích xuất đặc trưng

        Returns
        -------
        Dict[str, Any]
            Từ điển chứa đặc trưng bệnh nhân
        """
        features = {}

        try:
            # Thông tin cơ bản
            features["plan_type"] = (
                plan.technique if hasattr(plan, "technique") else "UNKNOWN"
            )
            features["site"] = plan.site if hasattr(plan, "site") else "UNKNOWN"

            # Thông tin cấu trúc
            structure_features = {}
            ptv_volumes = []
            oar_distances = {}

            if hasattr(plan, "structures") and plan.structures:
                # Tìm tất cả PTV
                ptv_structures = [
                    s for s in plan.structures if hasattr(s, "type") and s.type == "PTV"
                ]

                # Tính thể tích PTV
                for ptv in ptv_structures:
                    if hasattr(ptv, "volume") and ptv.volume:
                        ptv_volumes.append(ptv.volume)

                # Tìm OARs và tính khoảng cách đến PTV
                oar_structures = [
                    s for s in plan.structures if hasattr(s, "type") and s.type == "OAR"
                ]

                if ptv_structures and oar_structures:
                    main_ptv = ptv_structures[0]  # Sử dụng PTV đầu tiên làm tham chiếu

                    for oar in oar_structures:
                        if hasattr(oar, "id") and hasattr(oar, "center"):
                            oar_id = oar.id

                            # Tính khoảng cách giữa tâm OAR và tâm PTV
                            if hasattr(main_ptv, "center"):
                                distance = np.linalg.norm(
                                    np.array(oar.center) - np.array(main_ptv.center)
                                )
                                oar_distances[oar_id] = distance

            # Thêm thông tin đã tính toán vào đặc trưng
            features["ptv_volume_cc"] = sum(ptv_volumes) if ptv_volumes else 0
            features["num_ptvs"] = len(ptv_volumes)
            features["oar_distances"] = oar_distances

            # Thông tin chùm tia
            if hasattr(plan, "beams") and plan.beams:
                features["num_beams"] = len(plan.beams)
                features["beam_energies"] = [
                    beam.energy if hasattr(beam, "energy") else "unknown"
                    for beam in plan.beams
                ]
        except Exception as e:
            logger.error(f"Lỗi khi trích xuất đặc trưng bệnh nhân: {e}")

        return features

    def predict_optimal_objectives(
        self, plan: Plan
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Dự đoán mục tiêu tối ưu dựa trên mô hình và đặc trưng bệnh nhân.

        Parameters
        ----------
        plan : Plan
            Kế hoạch cần dự đoán mục tiêu

        Returns
        -------
        Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]
            Tuple chứa (mục tiêu dự đoán, ràng buộc dự đoán)
        """
        if not self.model:
            logger.warning("Không có mô hình KBP khả dụng để dự đoán mục tiêu")
            return [], []

        try:
            # Trích xuất đặc trưng bệnh nhân
            patient_features = self.extract_patient_features(plan)

            # Sử dụng mô hình để dự đoán mục tiêu
            predicted_objectives, predicted_constraints = self.model.predict_objectives(
                patient_features
            )

            return predicted_objectives, predicted_constraints
        except Exception as e:
            logger.error(f"Lỗi khi dự đoán mục tiêu tối ưu: {e}")
            return [], []

    def optimize(self, plan: Plan) -> Tuple[bool, Plan]:
        """
        Tối ưu hóa kế hoạch sử dụng phương pháp dựa trên mô hình.

        Parameters
        ----------
        plan : Plan
            Kế hoạch cần tối ưu hóa

        Returns
        -------
        Tuple[bool, Plan]
            Tuple chứa (thành công, kế hoạch tối ưu)
        """
        if not self.model:
            logger.error("Không thể tối ưu hóa: Không có mô hình KBP khả dụng")
            return False, plan

        # Đặt lại sự kiện dừng
        self._stop_event.clear()
        self.is_optimizing = True
        start_time = time.time()

        try:
            # Báo cáo tiến độ bắt đầu
            self._report_progress(0.01, "Bắt đầu tối ưu hóa dựa trên mô hình...")

            # Tạo bản sao kế hoạch để bảo vệ kế hoạch ban đầu
            plan_copy = plan.copy() if hasattr(plan, "copy") else plan

            # 1. Dự đoán mục tiêu tối ưu ban đầu
            self._report_progress(0.05, "Dự đoán mục tiêu tối ưu từ mô hình...")
            objectives, constraints = self.predict_optimal_objectives(plan_copy)

            if not objectives:
                logger.error("Không thể dự đoán mục tiêu tối ưu từ mô hình")
                self.is_optimizing = False
                return False, plan

            # 2. Chạy tối ưu hóa ban đầu
            self._report_progress(
                0.1, "Thực hiện tối ưu hóa ban đầu với mục tiêu dự đoán..."
            )

            # Thiết lập callback tiến độ cho bộ tối ưu hóa
            def optimization_progress(progress, message):
                # Ánh xạ tiến độ 0-1 đến 0.1-0.7
                adjusted_progress = 0.1 + progress * 0.6
                self._report_progress(adjusted_progress, message)

                # Kiểm tra nếu có yêu cầu dừng
                return not self._stop_event.is_set()

            self.optimizer.register_progress_callback(optimization_progress)

            # Thiết lập mục tiêu và ràng buộc
            self.optimizer.set_objectives(objectives)
            self.optimizer.set_constraints(constraints)

            # Chạy tối ưu hóa
            success = self.optimizer.optimize(plan_copy)

            if not success:
                logger.warning("Tối ưu hóa ban đầu không thành công")
                self.is_optimizing = False
                return False, plan

            # 3. Đánh giá kế hoạch ban đầu
            self._report_progress(0.7, "Đánh giá kế hoạch ban đầu...")
            dvh_calculator = DVHCalculator()
            dvh_results = dvh_calculator.calculate_dvh(plan_copy)
            metrics = calculate_plan_metrics(plan_copy, dvh_results)

            # Lưu kết quả ban đầu
            initial_result = {
                "objectives": objectives.copy(),
                "constraints": constraints.copy(),
                "metrics": metrics.copy(),
                "plan": plan_copy,
            }

            self.history = [initial_result]
            best_result = initial_result

            # 4. Điều chỉnh tinh cho cải thiện kế hoạch
            for i in range(self.max_iterations):
                if self._stop_event.is_set():
                    logger.info("Dừng tối ưu hóa theo yêu cầu")
                    break

                iteration_progress_base = 0.7 + (i / self.max_iterations) * 0.25
                self._report_progress(
                    iteration_progress_base,
                    f"Điều chỉnh tinh lần {i + 1}/{self.max_iterations}...",
                )

                # Sao chép mục tiêu và ràng buộc để điều chỉnh
                refined_objectives = objectives.copy()
                refined_constraints = constraints.copy()

                # Điều chỉnh dựa trên kết quả trước đó
                self._refine_objectives_and_constraints(
                    plan_copy, refined_objectives, refined_constraints, metrics
                )

                # Thiết lập mục tiêu và ràng buộc đã điều chỉnh
                self.optimizer.set_objectives(refined_objectives)
                self.optimizer.set_constraints(refined_constraints)

                # Đặt lại callback tiến độ để ánh xạ đúng phạm vi tiến độ
                def refinement_progress(progress, message):
                    # Ánh xạ tiến độ 0-1 đến phạm vi của lần lặp hiện tại
                    iteration_range = 0.25 / self.max_iterations
                    adjusted_progress = (
                        iteration_progress_base + progress * iteration_range
                    )
                    self._report_progress(adjusted_progress, message)

                    # Kiểm tra nếu có yêu cầu dừng
                    return not self._stop_event.is_set()

                self.optimizer.register_progress_callback(refinement_progress)

                # Chạy tối ưu hóa điều chỉnh
                refinement_success = self.optimizer.optimize(plan_copy)

                if not refinement_success:
                    logger.warning(f"Điều chỉnh tinh lần {i + 1} không thành công")
                    continue

                # Đánh giá kế hoạch đã điều chỉnh
                dvh_results = dvh_calculator.calculate_dvh(plan_copy)
                new_metrics = calculate_plan_metrics(plan_copy, dvh_results)

                # Lưu kết quả điều chỉnh
                refinement_result = {
                    "objectives": refined_objectives.copy(),
                    "constraints": refined_constraints.copy(),
                    "metrics": new_metrics.copy(),
                    "plan": plan_copy,
                }

                self.history.append(refinement_result)

                # Cập nhật kế hoạch tốt nhất nếu điểm số cải thiện
                if self._is_better_plan(new_metrics, best_result["metrics"]):
                    best_result = refinement_result

                # Cập nhật mục tiêu và ràng buộc cho lần lặp tiếp theo
                objectives = refined_objectives.copy()
                constraints = refined_constraints.copy()
                metrics = new_metrics.copy()

                # Kiểm tra hội tụ
                if i > 0 and self._check_convergence(
                    self.history[-2]["metrics"], new_metrics
                ):
                    logger.info(f"Đã đạt hội tụ sau {i + 1} lần lặp")
                    break

            # 5. Hoàn thành tối ưu hóa
            elapsed_time = time.time() - start_time
            self._report_progress(
                1.0,
                f"Đã hoàn thành tối ưu hóa dựa trên mô hình trong {elapsed_time:.2f} giây",
            )

            self.is_optimizing = False
            return True, best_result["plan"]

        except Exception as e:
            logger.error(f"Lỗi trong quá trình tối ưu hóa dựa trên mô hình: {e}")
            self.is_optimizing = False
            return False, plan

    def _refine_objectives_and_constraints(
        self,
        plan: Plan,
        objectives: List[Dict[str, Any]],
        constraints: List[Dict[str, Any]],
        metrics: Dict[str, Any],
    ) -> None:
        """
        Điều chỉnh tinh mục tiêu và ràng buộc dựa trên kết quả.

        Parameters
        ----------
        plan : Plan
            Kế hoạch hiện tại
        objectives : List[Dict[str, Any]]
            Mục tiêu cần điều chỉnh
        constraints : List[Dict[str, Any]]
            Ràng buộc cần điều chỉnh
        metrics : Dict[str, Any]
            Chỉ số đánh giá kế hoạch
        """
        try:
            # Điều chỉnh mục tiêu PTV
            ptv_coverage = metrics.get("ptv_coverage", {})

            for i, obj in enumerate(objectives):
                structure_id = obj.get("structure_id")
                structure_type = obj.get("structure_type")
                objective_type = obj.get("type")

                # Điều chỉnh mục tiêu cho PTV
                if structure_type == "PTV":
                    ptv_metrics = ptv_coverage.get(structure_id, {})
                    d95 = ptv_metrics.get("D95", 0)
                    v95 = ptv_metrics.get("V95", 0)
                    dmin = ptv_metrics.get("Dmin", 0)
                    dmax = ptv_metrics.get("Dmax", 0)
                    dmean = ptv_metrics.get("Dmean", 0)

                    # Điều chỉnh mục tiêu dựa trên kết quả
                    if objective_type == "MinDose":
                        # Tăng liều tối thiểu nếu coverage không đủ
                        if v95 < 98:
                            current_dose = obj.get("dose", 0)
                            adjusted_dose = current_dose * (
                                1 - self.refinement_learning_rate * (1 - v95 / 100)
                            )
                            obj["dose"] = max(
                                adjusted_dose, current_dose * 0.9
                            )  # Giới hạn điều chỉnh

                    elif objective_type == "MaxDose":
                        # Giảm liều tối đa nếu quá cao
                        if dmax > 1.15 * dmean:
                            current_dose = obj.get("dose", 0)
                            adjusted_dose = current_dose * (
                                1
                                - self.refinement_learning_rate * (dmax / dmean - 1.15)
                            )
                            obj["dose"] = max(
                                adjusted_dose, current_dose * 0.95
                            )  # Giới hạn điều chỉnh

                    # Điều chỉnh trọng số dựa trên độ quan trọng
                    if v95 < 95:
                        obj["weight"] = min(
                            obj.get("weight", 1.0) * 1.2, 100.0
                        )  # Tăng trọng số

                # Điều chỉnh mục tiêu cho OAR
                elif structure_type == "OAR":
                    oar_constraints_met = metrics.get("oar_constraints_met", 0)
                    total_oar_constraints = metrics.get("total_oar_constraints", 1)

                    # Nếu hầu hết các ràng buộc OAR được đáp ứng, có thể cố gắng giảm liều hơn nữa
                    if oar_constraints_met / total_oar_constraints > 0.9:
                        if objective_type in ["MaxDose", "MaxDVH"]:
                            current_dose = obj.get("dose", 0)
                            obj["dose"] = current_dose * (
                                1 - self.refinement_learning_rate * 0.1
                            )

                    # Nếu một ràng buộc OAR cụ thể không được đáp ứng, tăng trọng số
                    unmet_constraints = metrics.get("unmet_constraints", [])
                    if structure_id in unmet_constraints:
                        obj["weight"] = min(obj.get("weight", 1.0) * 1.3, 100.0)

            # Điều chỉnh tương tự cho ràng buộc
            for i, constraint in enumerate(constraints):
                structure_id = constraint.get("structure_id")
                structure_type = constraint.get("structure_type")
                constraint_type = constraint.get("type")

                # Điều chỉnh ràng buộc dựa trên kết quả tương tự như mục tiêu

        except Exception as e:
            logger.error(f"Lỗi khi điều chỉnh tinh mục tiêu và ràng buộc: {e}")

    def _is_better_plan(
        self, new_metrics: Dict[str, Any], old_metrics: Dict[str, Any]
    ) -> bool:
        """
        Đánh giá xem kế hoạch mới có tốt hơn kế hoạch cũ không.

        Parameters
        ----------
        new_metrics : Dict[str, Any]
            Chỉ số đánh giá kế hoạch mới
        old_metrics : Dict[str, Any]
            Chỉ số đánh giá kế hoạch cũ

        Returns
        -------
        bool
            True nếu kế hoạch mới tốt hơn, False nếu không
        """
        # Trọng số cho các chỉ số quan trọng
        weights = {
            "CI": 0.3,
            "HI": 0.2,
            "ptv_coverage_score": 0.3,
            "oar_sparing_score": 0.2,
        }

        # Tính điểm cho mỗi kế hoạch
        new_score = 0
        old_score = 0

        # Conformity Index (càng gần 1 càng tốt)
        if "CI" in new_metrics and "CI" in old_metrics:
            new_ci_score = 1 - abs(new_metrics["CI"] - 1)
            old_ci_score = 1 - abs(old_metrics["CI"] - 1)
            new_score += weights["CI"] * new_ci_score
            old_score += weights["CI"] * old_ci_score

        # Homogeneity Index (càng nhỏ càng tốt)
        if "HI" in new_metrics and "HI" in old_metrics:
            new_hi_score = 1 / (1 + new_metrics["HI"])
            old_hi_score = 1 / (1 + old_metrics["HI"])
            new_score += weights["HI"] * new_hi_score
            old_score += weights["HI"] * old_hi_score

        # PTV Coverage Score
        if "ptv_coverage_score" in new_metrics and "ptv_coverage_score" in old_metrics:
            new_score += (
                weights["ptv_coverage_score"] * new_metrics["ptv_coverage_score"]
            )
            old_score += (
                weights["ptv_coverage_score"] * old_metrics["ptv_coverage_score"]
            )

        # OAR Sparing Score
        if "oar_sparing_score" in new_metrics and "oar_sparing_score" in old_metrics:
            new_score += weights["oar_sparing_score"] * new_metrics["oar_sparing_score"]
            old_score += weights["oar_sparing_score"] * old_metrics["oar_sparing_score"]

        # Mặc định, nếu không đủ chỉ số để đánh giá
        if new_score == 0 and old_score == 0:
            # Kiểm tra đơn giản dựa trên số ràng buộc OAR được đáp ứng
            new_oar_met = new_metrics.get("oar_constraints_met", 0)
            old_oar_met = old_metrics.get("oar_constraints_met", 0)
            new_oar_total = new_metrics.get("total_oar_constraints", 1)
            old_oar_total = old_metrics.get("total_oar_constraints", 1)

            new_oar_score = new_oar_met / new_oar_total
            old_oar_score = old_oar_met / old_oar_total

            return new_oar_score > old_oar_score

        return new_score > old_score

    def _check_convergence(
        self, old_metrics: Dict[str, Any], new_metrics: Dict[str, Any]
    ) -> bool:
        """
        Kiểm tra xem quá trình tối ưu hóa đã hội tụ chưa.

        Parameters
        ----------
        old_metrics : Dict[str, Any]
            Chỉ số đánh giá kế hoạch cũ
        new_metrics : Dict[str, Any]
            Chỉ số đánh giá kế hoạch mới

        Returns
        -------
        bool
            True nếu đã hội tụ, False nếu chưa
        """
        # Kiểm tra sự thay đổi trong các chỉ số chính
        for key in ["CI", "HI", "GI", "ptv_coverage_score", "oar_sparing_score"]:
            if key in old_metrics and key in new_metrics:
                old_value = old_metrics[key]
                new_value = new_metrics[key]

                # Nếu giá trị thay đổi quá ngưỡng, chưa hội tụ
                if (
                    abs(new_value - old_value) / (abs(old_value) + 1e-10)
                    > self.convergence_threshold
                ):
                    return False

        # Kiểm tra số lượng ràng buộc OAR được đáp ứng
        old_oar_met = old_metrics.get("oar_constraints_met", 0)
        new_oar_met = new_metrics.get("oar_constraints_met", 0)

        if abs(new_oar_met - old_oar_met) > 0:
            return False

        return True

    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """
        Lấy lịch sử tối ưu hóa.

        Returns
        -------
        List[Dict[str, Any]]
            Lịch sử tối ưu hóa
        """
        return self.history

    def get_progress(self) -> Tuple[float, str]:
        """
        Lấy tiến độ hiện tại của quá trình tối ưu hóa.

        Returns
        -------
        Tuple[float, str]
            Tuple chứa (tiến độ, thông điệp)
        """
        return self.current_progress, self.current_message
