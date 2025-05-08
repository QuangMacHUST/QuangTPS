#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module lập kế hoạch thích ứng thời gian thực tự động.

Module này cung cấp các lớp và chức năng để tự động tạo và điều chỉnh kế hoạch
xạ trị trong thời gian thực, dựa trên sự thay đổi của cấu trúc giải phẫu
trong quá trình điều trị.
"""

import os
import logging
import time
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union, Set
from datetime import datetime, timedelta

from quangtps.core.patient import Patient
from quangtps.planning.plan import Plan
from quangtps.adaptive.dose_accumulation import DoseAccumulation
from quangtps.adaptive.deformation.deformation_map import DeformationMap
from quangtps.adaptive.prediction.deformable_anatomy_predictor import (
    DeformableAnatomyPredictor,
)
from quangtps.adaptive.prediction.anatomy_prediction import AnatomyPredictionModel
from quangtps.structures.structure_utils import StructureOperations
from quangtps.optimization.optimizer import OptimizerFactory
from quangtps.core.types import Structure, ImageSeries
from quangtps.core.services import ServiceRegistry

logger = logging.getLogger(__name__)


class RealTimeAdaptivePlanner:
    """
    Lớp quản lý lập kế hoạch thích ứng thời gian thực tự động.

    Lớp này phối hợp việc dự đoán thay đổi giải phẫu, tạo kế hoạch thích ứng,
    và tối ưu hóa kế hoạch trên cơ sở các thay đổi được dự đoán.
    """

    def __init__(self, patient: Patient, reference_plan: Optional[Plan] = None):
        """
        Khởi tạo lớp lập kế hoạch thích ứng thời gian thực.

        Args:
            patient: Đối tượng bệnh nhân.
            reference_plan: Kế hoạch tham chiếu ban đầu (nếu có).
        """
        self.patient = patient
        self.reference_plan = reference_plan
        self.anatomy_predictor = DeformableAnatomyPredictor()
        self.dose_accumulator = DoseAccumulation()
        self.structure_operations = StructureOperations()
        self.prediction_models: Dict[str, AnatomyPredictionModel] = {}
        self.adaptive_plans: List[Plan] = []
        self.prediction_horizon = 5  # Số phiên điều trị dự đoán trước
        self.monitoring_structures: Set[str] = set()  # Các cấu trúc cần theo dõi
        self.adaptation_threshold = 0.05  # Ngưỡng thay đổi thể tích để thích ứng (5%)

        # Kết nối với các dịch vụ cần thiết
        self.plan_service = ServiceRegistry.get_service("PlanService")
        self.optimization_service = ServiceRegistry.get_service("OptimizationService")
        self.dose_service = ServiceRegistry.get_service("DoseService")
        self.image_service = ServiceRegistry.get_service("ImageService")

        # Thiết lập các cấu trúc cần theo dõi mặc định
        if reference_plan:
            self._setup_monitoring_structures()

    def _setup_monitoring_structures(self):
        """Thiết lập các cấu trúc cần theo dõi dựa trên kế hoạch tham chiếu."""
        if not self.reference_plan:
            return

        # Thêm target volumes vào danh sách theo dõi
        for structure in self.reference_plan.get_structures():
            structure_type = structure.type.lower()
            if (
                "target" in structure_type
                or "ptv" in structure_type
                or "ctv" in structure_type
            ):
                self.monitoring_structures.add(structure.id)

        # Thêm các cơ quan quan trọng vào danh sách theo dõi
        critical_organs = [
            "heart",
            "lung",
            "spinalcord",
            "brainstem",
            "kidney",
            "liver",
            "bladder",
            "rectum",
        ]
        for structure in self.reference_plan.get_structures():
            for organ in critical_organs:
                if organ in structure.name.lower():
                    self.monitoring_structures.add(structure.id)

    def set_monitoring_structures(self, structure_ids: List[str]):
        """
        Thiết lập danh sách các cấu trúc cần theo dõi.

        Args:
            structure_ids: Danh sách ID của các cấu trúc cần theo dõi.
        """
        self.monitoring_structures = set(structure_ids)

    def set_adaptation_threshold(self, threshold: float):
        """
        Thiết lập ngưỡng thay đổi thể tích để kích hoạt thích ứng.

        Args:
            threshold: Ngưỡng thay đổi thể tích (0.0-1.0).
        """
        if 0 <= threshold <= 1:
            self.adaptation_threshold = threshold
        else:
            logger.warning(
                f"Ngưỡng thích ứng không hợp lệ: {threshold}. Phải trong khoảng 0-1."
            )

    def set_prediction_horizon(self, days: int):
        """
        Thiết lập số ngày dự đoán trước.

        Args:
            days: Số ngày dự đoán trước.
        """
        if days > 0:
            self.prediction_horizon = days
        else:
            logger.warning(f"Khoảng dự đoán không hợp lệ: {days}. Phải lớn hơn 0.")

    def load_prediction_model(self, structure_id: str, model_path: str) -> bool:
        """
        Tải mô hình dự đoán cho một cấu trúc cụ thể.

        Args:
            structure_id: ID của cấu trúc.
            model_path: Đường dẫn đến mô hình dự đoán.

        Returns:
            True nếu tải thành công, False nếu thất bại.
        """
        try:
            model = AnatomyPredictionModel.load(model_path)
            self.prediction_models[structure_id] = model
            logger.info(
                f"Đã tải mô hình dự đoán cho cấu trúc {structure_id} từ {model_path}"
            )
            return True
        except Exception as e:
            logger.error(f"Lỗi khi tải mô hình dự đoán: {str(e)}")
            return False

    def predict_anatomy_changes(
        self, date: Optional[datetime] = None, days_ahead: Optional[int] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Dự đoán thay đổi cấu trúc giải phẫu theo thời gian.

        Args:
            date: Ngày bắt đầu dự đoán (mặc định là ngày hiện tại).
            days_ahead: Số ngày dự đoán (ghi đè prediction_horizon nếu được cung cấp).

        Returns:
            Dict chứa thông tin dự đoán cho mỗi cấu trúc theo dõi.
        """
        if date is None:
            date = datetime.now()

        horizon = days_ahead if days_ahead is not None else self.prediction_horizon

        # Lấy ảnh và cấu trúc gần nhất
        latest_image = self.patient.get_latest_image_series()
        if not latest_image:
            logger.error("Không thể dự đoán: Không có ảnh mới nhất của bệnh nhân")
            return {}

        predictions = {}

        for structure_id in self.monitoring_structures:
            # Lấy cấu trúc từ ID
            structure = None
            for struct in self.patient.get_structures():
                if struct.id == structure_id:
                    structure = struct
                    break

            if not structure:
                logger.warning(f"Không tìm thấy cấu trúc {structure_id} để dự đoán")
                continue

            # Sử dụng mô hình dự đoán nếu có, nếu không sử dụng dự đoán mặc định
            if structure_id in self.prediction_models:
                model = self.prediction_models[structure_id]
                structure_predictions = []

                # Dự đoán cho mỗi ngày
                for day in range(1, horizon + 1):
                    target_date = date + timedelta(days=day)
                    predicted_volume = model.predict_volume(structure, target_date)
                    predicted_deformation = model.predict_deformation(
                        structure, target_date
                    )

                    structure_predictions.append(
                        {
                            "date": target_date,
                            "volume": predicted_volume,
                            "deformation": predicted_deformation,
                        }
                    )

                predictions[structure_id] = {
                    "structure_name": structure.name,
                    "current_volume": structure.volume,
                    "predictions": structure_predictions,
                }
            else:
                # Dự đoán đơn giản nếu không có mô hình
                self.anatomy_predictor.add_structure(structure)

                structure_predictions = []
                for day in range(1, horizon + 1):
                    target_date = date + timedelta(days=day)
                    predicted_structure = self.anatomy_predictor.predict_structure(
                        structure_id, target_date
                    )
                    if predicted_structure:
                        structure_predictions.append(
                            {
                                "date": target_date,
                                "volume": predicted_structure.volume,
                                "centroid_shift": predicted_structure.get_centroid_shift(
                                    structure
                                ),
                            }
                        )

                predictions[structure_id] = {
                    "structure_name": structure.name,
                    "current_volume": structure.volume,
                    "predictions": structure_predictions,
                }

        return predictions

    def check_adaptation_needed(self, predictions: Dict[str, Dict[str, Any]]) -> bool:
        """
        Kiểm tra xem có cần thích ứng kế hoạch dựa trên dự đoán thay đổi hay không.

        Args:
            predictions: Kết quả dự đoán từ hàm predict_anatomy_changes.

        Returns:
            True nếu cần thích ứng, False nếu không.
        """
        if not predictions:
            return False

        for structure_id, prediction_data in predictions.items():
            structure_predictions = prediction_data.get("predictions", [])
            current_volume = prediction_data.get("current_volume", 0)

            if not structure_predictions or current_volume <= 0:
                continue

            # Kiểm tra các dự đoán
            for pred in structure_predictions:
                predicted_volume = pred.get("volume", 0)
                if (
                    abs(predicted_volume - current_volume) / current_volume
                    > self.adaptation_threshold
                ):
                    logger.info(
                        f"Cần thích ứng: Cấu trúc {structure_id} dự đoán thay đổi {abs(predicted_volume - current_volume) / current_volume:.2%}"
                    )
                    return True

                # Kiểm tra dịch chuyển tâm (nếu có)
                if (
                    "centroid_shift" in pred and pred["centroid_shift"] > 3.0
                ):  # Ngưỡng 3mm
                    logger.info(
                        f"Cần thích ứng: Cấu trúc {structure_id} dự đoán dịch chuyển tâm {pred['centroid_shift']:.2f}mm"
                    )
                    return True

        return False

    def generate_adaptive_plan(
        self, predicted_date: datetime, predictions: Dict[str, Dict[str, Any]]
    ) -> Optional[Plan]:
        """
        Tạo kế hoạch thích ứng dựa trên dự đoán thay đổi cấu trúc.

        Args:
            predicted_date: Ngày dự đoán kế hoạch.
            predictions: Kết quả dự đoán từ hàm predict_anatomy_changes.

        Returns:
            Kế hoạch thích ứng mới hoặc None nếu thất bại.
        """
        if not self.reference_plan or not predictions:
            logger.error(
                "Không thể tạo kế hoạch thích ứng: Thiếu kế hoạch tham chiếu hoặc dự đoán"
            )
            return None

        try:
            # Tạo bản sao kế hoạch tham chiếu
            adaptive_plan = self.reference_plan.clone()
            date_str = predicted_date.strftime("%Y%m%d")
            adaptive_plan.name = f"{self.reference_plan.name}_Adapt_{date_str}"
            adaptive_plan.description = f"Kế hoạch thích ứng tự động ngày {date_str}"

            # Cập nhật cấu trúc dựa trên dự đoán
            for structure_id, prediction_data in predictions.items():
                structure_predictions = prediction_data.get("predictions", [])

                # Tìm dự đoán cho ngày cụ thể
                target_prediction = None
                for pred in structure_predictions:
                    pred_date = pred.get("date")
                    if pred_date and pred_date.date() == predicted_date.date():
                        target_prediction = pred
                        break

                if not target_prediction:
                    continue

                # Cập nhật cấu trúc
                structure = adaptive_plan.get_structure_by_id(structure_id)
                if not structure:
                    continue

                if "deformation" in target_prediction:
                    # Áp dụng biến dạng dự đoán
                    deformation = target_prediction["deformation"]
                    deformed_structure = self.structure_operations.apply_deformation(
                        structure, deformation
                    )
                    if deformed_structure:
                        adaptive_plan.update_structure(deformed_structure)
                elif "centroid_shift" in target_prediction:
                    # Dịch chuyển cấu trúc theo centroid shift
                    shift = target_prediction.get("centroid_shift", 0)
                    if shift > 0:
                        # Tạo vector dịch chuyển đơn giản
                        shift_vector = np.array(
                            [shift / 2, shift / 2, 0]
                        )  # Dịch chuyển theo hướng X và Y
                        shifted_structure = self.structure_operations.shift_structure(
                            structure, shift_vector
                        )
                        if shifted_structure:
                            adaptive_plan.update_structure(shifted_structure)

            # Tối ưu hóa lại kế hoạch
            if self.optimization_service:
                optimizer = OptimizerFactory.create_optimizer("vmat", adaptive_plan)
                if optimizer:
                    # Sử dụng cùng các mục tiêu và ràng buộc từ kế hoạch tham chiếu
                    optimizer.set_objectives(self.reference_plan.get_objectives())
                    optimizer.set_constraints(self.reference_plan.get_constraints())

                    # Tối ưu hóa kế hoạch
                    result = optimizer.optimize()
                    if not result:
                        logger.warning("Tối ưu hóa kế hoạch thích ứng không thành công")

            # Tính toán lại liều
            if self.dose_service:
                self.dose_service.calculate(adaptive_plan)

            # Lưu kế hoạch vào danh sách
            self.adaptive_plans.append(adaptive_plan)

            logger.info(f"Đã tạo kế hoạch thích ứng: {adaptive_plan.name}")
            return adaptive_plan

        except Exception as e:
            logger.error(f"Lỗi khi tạo kế hoạch thích ứng: {str(e)}")
            return None

    def generate_adaptive_plans_sequence(
        self, start_date: datetime = None
    ) -> List[Plan]:
        """
        Tạo chuỗi kế hoạch thích ứng dựa trên dự đoán thay đổi trong khoảng thời gian.

        Args:
            start_date: Ngày bắt đầu tạo kế hoạch (mặc định là ngày hiện tại).

        Returns:
            Danh sách các kế hoạch thích ứng đã tạo.
        """
        if start_date is None:
            start_date = datetime.now()

        adaptive_plans = []
        current_plan = self.reference_plan

        for day in range(1, self.prediction_horizon + 1):
            current_date = start_date + timedelta(days=day)

            # Dự đoán thay đổi dựa trên kế hoạch hiện tại
            predictions = self.predict_anatomy_changes(current_date, 1)

            # Kiểm tra xem có cần thích ứng hay không
            if self.check_adaptation_needed(predictions):
                # Tạo kế hoạch thích ứng
                adaptive_plan = self.generate_adaptive_plan(current_date, predictions)
                if adaptive_plan:
                    adaptive_plans.append(adaptive_plan)
                    current_plan = adaptive_plan  # Sử dụng kế hoạch mới làm tham chiếu

        return adaptive_plans

    def auto_adapt_plan_for_fraction(
        self, fraction_number: int, fraction_image: Optional[ImageSeries] = None
    ) -> Optional[Plan]:
        """
        Tự động thích ứng kế hoạch cho một phiên điều trị dựa trên ảnh kiểm tra.

        Args:
            fraction_number: Số thứ tự phiên điều trị.
            fraction_image: Ảnh kiểm tra của phiên điều trị (nếu có).

        Returns:
            Kế hoạch thích ứng mới hoặc None nếu không cần thích ứng.
        """
        if not self.reference_plan:
            logger.error("Không thể thích ứng: Thiếu kế hoạch tham chiếu")
            return None

        # Nếu không có ảnh kiểm tra, sử dụng dự đoán
        if not fraction_image:
            current_date = datetime.now()
            predictions = self.predict_anatomy_changes(current_date, 1)

            if self.check_adaptation_needed(predictions):
                return self.generate_adaptive_plan(current_date, predictions)
            else:
                logger.info("Không cần thích ứng kế hoạch dựa trên dự đoán")
                return None

        # Nếu có ảnh kiểm tra, sử dụng ảnh thực tế để thích ứng
        try:
            # Đăng ký ảnh mới với ảnh tham chiếu
            reference_image = self.reference_plan.image_series
            if not reference_image:
                logger.error("Không tìm thấy ảnh tham chiếu trong kế hoạch")
                return None

            # Đăng ký ảnh bằng cách sử dụng dịch vụ
            if self.image_service:
                deformation_map = self.image_service.register_images(
                    reference_image, fraction_image
                )
                if not deformation_map:
                    logger.error("Không thể đăng ký ảnh kiểm tra với ảnh tham chiếu")
                    return None

                # Tạo kế hoạch thích ứng mới
                adaptive_plan = self.reference_plan.clone()
                date_str = datetime.now().strftime("%Y%m%d")
                adaptive_plan.name = (
                    f"{self.reference_plan.name}_Adapt_Frac{fraction_number}"
                )
                adaptive_plan.description = (
                    f"Kế hoạch thích ứng tự động cho phiên {fraction_number}"
                )
                adaptive_plan.image_series = fraction_image

                # Biến dạng cấu trúc từ kế hoạch tham chiếu
                structure_changes = {}
                adaptation_needed = False

                for structure_id in self.monitoring_structures:
                    structure = self.reference_plan.get_structure_by_id(structure_id)
                    if not structure:
                        continue

                    # Áp dụng biến dạng lên cấu trúc
                    deformed_structure = self.structure_operations.apply_deformation(
                        structure, deformation_map
                    )
                    if not deformed_structure:
                        continue

                    # Tính phần trăm thay đổi thể tích
                    volume_change = (
                        abs(deformed_structure.volume - structure.volume)
                        / structure.volume
                    )
                    structure_changes[structure_id] = {
                        "original_volume": structure.volume,
                        "adapted_volume": deformed_structure.volume,
                        "volume_change": volume_change,
                    }

                    # Kiểm tra xem có vượt ngưỡng thích ứng không
                    if volume_change > self.adaptation_threshold:
                        adaptation_needed = True

                    # Cập nhật cấu trúc trong kế hoạch
                    adaptive_plan.update_structure(deformed_structure)

                # Nếu không cần thích ứng thì trả về None
                if not adaptation_needed:
                    logger.info("Không cần thích ứng kế hoạch dựa trên ảnh kiểm tra")
                    return None

                # Tối ưu hóa lại kế hoạch
                if self.optimization_service:
                    optimizer = OptimizerFactory.create_optimizer("vmat", adaptive_plan)
                    if optimizer:
                        # Sử dụng cùng các mục tiêu và ràng buộc từ kế hoạch tham chiếu
                        optimizer.set_objectives(self.reference_plan.get_objectives())
                        optimizer.set_constraints(self.reference_plan.get_constraints())

                        # Tối ưu hóa kế hoạch
                        result = optimizer.optimize()
                        if not result:
                            logger.warning(
                                "Tối ưu hóa kế hoạch thích ứng không thành công"
                            )

                # Tính toán lại liều
                if self.dose_service:
                    self.dose_service.calculate(adaptive_plan)

                # Lưu kế hoạch vào danh sách
                self.adaptive_plans.append(adaptive_plan)

                logger.info(
                    f"Đã tạo kế hoạch thích ứng cho phiên {fraction_number}: {adaptive_plan.name}"
                )
                logger.info(f"Thay đổi thể tích các cấu trúc: {structure_changes}")

                return adaptive_plan

        except Exception as e:
            logger.error(f"Lỗi khi tạo kế hoạch thích ứng từ ảnh kiểm tra: {str(e)}")
            return None
