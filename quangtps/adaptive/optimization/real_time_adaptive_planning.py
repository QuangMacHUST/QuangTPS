#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module lập kế hoạch xạ trị thích ứng thời gian thực trong QuangTPS.

Module này cung cấp các chức năng lập kế hoạch thích ứng tự động trong thời gian thực,
dựa trên hình ảnh mới thu được trong từng phiên điều trị, giúp tính toán và điều chỉnh
kế hoạch điều trị một cách nhanh chóng và hiệu quả.
"""

import os
import time
import logging
import datetime
import threading
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union, Any, Callable
from enum import Enum, auto

from quangtps.core.types import Patient, Image, Structure, Plan, Dose
from quangtps.planning.plan import TreatmentPlan
from quangtps.adaptive.adaptive_planning import (
    AdaptivePlanner,
    AdaptiveActionType,
    AnatomicalChangeType,
)
from quangtps.adaptive.prediction.anatomy_prediction import AnatomyPredictor
from quangtps.adaptive.deformation.deformable_registration import DeformableRegistration
from quangtps.dose.dose_calculation import DoseCalculation
from quangtps.optimization.methods.optimizer import OptimizerBase
from quangtps.adaptive.four_d import FourDAnalysis
from quangtps.imaging.registration import ImageRegistration
from quangtps.segmentation.auto.auto_segmentation import AutoSegmenter
from quangtps.common.timer import Timer
from quangtps.core.utils import get_timestamp

logger = logging.getLogger(__name__)


class AdaptationPriority(Enum):
    """Mức độ ưu tiên của việc điều chỉnh kế hoạch."""

    NO_ADAPTATION = auto()  # Không cần điều chỉnh
    LOW = auto()  # Ưu tiên thấp, thay đổi nhỏ
    MEDIUM = auto()  # Ưu tiên trung bình, cần điều chỉnh nhưng không khẩn cấp
    HIGH = auto()  # Ưu tiên cao, cần điều chỉnh ngay
    CRITICAL = auto()  # Ưu tiên cao nhất, thay đổi quan trọng


class AdaptationStatus(Enum):
    """Trạng thái của quá trình lập kế hoạch thích ứng."""

    IDLE = auto()  # Đang ở chế độ chờ
    ANALYZING = auto()  # Đang phân tích
    SEGMENTING = auto()  # Đang phân đoạn
    PLANNING = auto()  # Đang lập kế hoạch
    OPTIMIZING = auto()  # Đang tối ưu hóa
    CALCULATING_DOSE = auto()  # Đang tính toán liều
    EVALUATING = auto()  # Đang đánh giá kế hoạch
    COMPLETED = auto()  # Hoàn thành
    FAILED = auto()  # Thất bại


class RealTimeAdaptiveSession:
    """Lớp quản lý một phiên lập kế hoạch thích ứng thời gian thực."""

    def __init__(
        self,
        patient: Patient,
        original_plan: TreatmentPlan,
        new_image: Image,
        session_id: Optional[str] = None,
    ):
        """
        Khởi tạo phiên lập kế hoạch thích ứng thời gian thực.

        Parameters
        ----------
        patient : Patient
            Đối tượng bệnh nhân
        original_plan : TreatmentPlan
            Kế hoạch điều trị ban đầu
        new_image : Image
            Hình ảnh mới (ví dụ: CBCT) thu được trong phiên điều trị
        session_id : Optional[str], optional
            ID của phiên, mặc định tự động tạo
        """
        self.patient = patient
        self.original_plan = original_plan
        self.new_image = new_image
        self.session_id = session_id or f"rt_adapt_{get_timestamp()}"

        # Thông tin thời gian
        self.start_time = datetime.datetime.now()
        self.end_time = None
        self.timing = {}  # Thời gian cho từng giai đoạn

        # Trạng thái và kết quả
        self.status = AdaptationStatus.IDLE
        self.priority = AdaptationPriority.NO_ADAPTATION
        self.adapted_plan = None
        self.adapted_structures = {}
        self.adapted_dose = None

        # Thông tin đánh giá
        self.evaluation_metrics = {}
        self.anatomical_changes = []
        self.dose_impact = {}

        # Logs và theo dõi
        self.logs = []
        self.progress = 0.0  # Tiến độ từ 0.0 đến 1.0

    def log(self, message: str):
        """Ghi nhật ký cho phiên."""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.logs.append(log_entry)
        logger.info(f"[Phiên {self.session_id}] {message}")

    def set_status(self, status: AdaptationStatus):
        """Cập nhật trạng thái của phiên."""
        self.status = status
        self.log(f"Trạng thái: {status.name}")

    def set_priority(self, priority: AdaptationPriority):
        """Cập nhật mức độ ưu tiên của phiên."""
        self.priority = priority
        self.log(f"Mức độ ưu tiên: {priority.name}")

    def add_anatomical_change(
        self,
        change_type: AnatomicalChangeType,
        structure_id: str,
        details: Dict[str, Any],
    ):
        """Thêm thông tin về thay đổi giải phẫu được phát hiện."""
        change_info = {
            "type": change_type,
            "structure_id": structure_id,
            "details": details,
        }
        self.anatomical_changes.append(change_info)
        self.log(f"Phát hiện thay đổi {change_type.name} trong cấu trúc {structure_id}")

    def set_dose_impact(self, structure_id: str, metric: str, value: float):
        """Cập nhật tác động liều lên một cấu trúc cụ thể."""
        if structure_id not in self.dose_impact:
            self.dose_impact[structure_id] = {}
        self.dose_impact[structure_id][metric] = value

    def track_timing(self, stage: str, elapsed_time: float):
        """Ghi lại thời gian cho một giai đoạn cụ thể."""
        self.timing[stage] = elapsed_time
        self.log(f"Hoàn thành giai đoạn {stage} sau {elapsed_time:.2f} giây")

    def mark_completed(self):
        """Đánh dấu phiên là hoàn thành."""
        self.end_time = datetime.datetime.now()
        elapsed = (self.end_time - self.start_time).total_seconds()
        self.set_status(AdaptationStatus.COMPLETED)
        self.progress = 1.0
        self.log(f"Hoàn thành sau {elapsed:.2f} giây")

    def mark_failed(self, reason: str):
        """Đánh dấu phiên là thất bại."""
        self.end_time = datetime.datetime.now()
        elapsed = (self.end_time - self.start_time).total_seconds()
        self.set_status(AdaptationStatus.FAILED)
        self.log(f"Thất bại sau {elapsed:.2f} giây. Lý do: {reason}")

    def get_elapsed_time(self) -> float:
        """Lấy tổng thời gian đã trôi qua (giây)."""
        end = self.end_time or datetime.datetime.now()
        return (end - self.start_time).total_seconds()

    def to_dict(self) -> Dict[str, Any]:
        """Chuyển đổi thành từ điển để lưu trữ hoặc hiển thị."""
        return {
            "session_id": self.session_id,
            "patient_id": self.patient.id,
            "original_plan_id": self.original_plan.id,
            "new_image_id": self.new_image.id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "status": self.status.name,
            "priority": self.priority.name,
            "adapted_plan_id": self.adapted_plan.id if self.adapted_plan else None,
            "timing": self.timing,
            "anatomical_changes": [
                {
                    "type": c["type"].name,
                    "structure_id": c["structure_id"],
                    "details": c["details"],
                }
                for c in self.anatomical_changes
            ],
            "dose_impact": self.dose_impact,
            "progress": self.progress,
            "logs": self.logs,
        }


class RealTimeAdaptivePlanner:
    """Lớp chính để quản lý và thực hiện quá trình lập kế hoạch thích ứng thời gian thực."""

    def __init__(
        self,
        auto_segmenter: Optional[AutoSegmenter] = None,
        image_registration: Optional[ImageRegistration] = None,
        dose_calculator: Optional[DoseCalculation] = None,
        deformable_registration: Optional[DeformableRegistration] = None,
        optimizer: Optional[OptimizerBase] = None,
        adaptive_planner: Optional[AdaptivePlanner] = None,
        max_threads: int = 4,
    ):
        """
        Khởi tạo bộ lập kế hoạch thích ứng thời gian thực.

        Parameters
        ----------
        auto_segmenter : Optional[AutoSegmenter], optional
            Đối tượng phân đoạn tự động, mặc định là None
        image_registration : Optional[ImageRegistration], optional
            Đối tượng đăng ký hình ảnh, mặc định là None
        dose_calculator : Optional[DoseCalculation], optional
            Đối tượng tính toán liều, mặc định là None
        deformable_registration : Optional[DeformableRegistration], optional
            Đối tượng đăng ký biến dạng, mặc định là None
        optimizer : Optional[OptimizerBase], optional
            Đối tượng tối ưu hóa, mặc định là None
        adaptive_planner : Optional[AdaptivePlanner], optional
            Đối tượng lập kế hoạch thích ứng, mặc định là None
        max_threads : int, optional
            Số luồng tối đa, mặc định là 4
        """
        # Các thành phần cần thiết
        self.auto_segmenter = auto_segmenter or AutoSegmenter()
        self.image_registration = image_registration or ImageRegistration()
        self.dose_calculator = dose_calculator or DoseCalculation()
        self.deformable_registration = (
            deformable_registration or DeformableRegistration()
        )
        self.optimizer = optimizer or OptimizerBase()
        self.adaptive_planner = adaptive_planner or AdaptivePlanner()

        # Các cài đặt
        self.max_threads = max_threads
        self.tolerance_thresholds = {
            "position": 5.0,  # mm
            "volume": 10.0,  # %
            "dvh": 5.0,  # %
            "dice": 0.85,  # Hệ số Dice
        }

        # Theo dõi tiến độ
        self.active_sessions = {}
        self.session_threads = {}

        # Callback function
        self.on_progress_callback = None
        self.on_complete_callback = None

        logger.info("Khởi tạo bộ lập kế hoạch thích ứng thời gian thực")

    def set_callbacks(
        self,
        on_progress: Optional[Callable[[RealTimeAdaptiveSession, float], None]] = None,
        on_complete: Optional[Callable[[RealTimeAdaptiveSession], None]] = None,
    ):
        """
        Thiết lập các hàm callback.

        Parameters
        ----------
        on_progress : Optional[Callable[[RealTimeAdaptiveSession, float], None]], optional
            Hàm được gọi khi tiến độ thay đổi, mặc định là None
        on_complete : Optional[Callable[[RealTimeAdaptiveSession], None]], optional
            Hàm được gọi khi phiên hoàn thành, mặc định là None
        """
        self.on_progress_callback = on_progress
        self.on_complete_callback = on_complete

    def start_adaptation_session(
        self,
        patient: Patient,
        original_plan: TreatmentPlan,
        new_image: Image,
        auto_start: bool = True,
    ) -> RealTimeAdaptiveSession:
        """
        Bắt đầu một phiên lập kế hoạch thích ứng mới.

        Parameters
        ----------
        patient : Patient
            Đối tượng bệnh nhân
        original_plan : TreatmentPlan
            Kế hoạch điều trị ban đầu
        new_image : Image
            Hình ảnh mới thu được
        auto_start : bool, optional
            Tự động bắt đầu phiên, mặc định là True

        Returns
        -------
        RealTimeAdaptiveSession
            Phiên lập kế hoạch thích ứng mới
        """
        # Tạo phiên mới
        session = RealTimeAdaptiveSession(patient, original_plan, new_image)
        self.active_sessions[session.session_id] = session

        # Bắt đầu phiên
        if auto_start:
            self._start_session_thread(session)

        return session

    def _start_session_thread(self, session: RealTimeAdaptiveSession):
        """Bắt đầu luồng xử lý cho phiên."""
        thread = threading.Thread(
            target=self._run_adaptation_process, args=(session,), daemon=True
        )
        self.session_threads[session.session_id] = thread
        thread.start()

    def _run_adaptation_process(self, session: RealTimeAdaptiveSession):
        """
        Tiến trình lập kế hoạch thích ứng chính.

        Parameters
        ----------
        session : RealTimeAdaptiveSession
            Phiên cần xử lý
        """
        try:
            # 1. Đăng ký hình ảnh
            self._update_progress(session, 0.05, AdaptationStatus.ANALYZING)
            with Timer() as timer:
                registration_result = self._register_new_image(session)
            session.track_timing("image_registration", timer.duration)

            if not registration_result:
                session.mark_failed("Không thể đăng ký hình ảnh")
                return

            # 2. Phân đoạn tự động cấu trúc
            self._update_progress(session, 0.15, AdaptationStatus.SEGMENTING)
            with Timer() as timer:
                segmentation_result = self._segment_structures(session)
            session.track_timing("auto_segmentation", timer.duration)

            if not segmentation_result:
                session.mark_failed("Không thể phân đoạn cấu trúc")
                return

            # 3. Phân tích thay đổi giải phẫu
            self._update_progress(session, 0.25, AdaptationStatus.ANALYZING)
            with Timer() as timer:
                analysis_result = self._analyze_anatomical_changes(session)
            session.track_timing("change_analysis", timer.duration)

            # 4. Quyết định chiến lược thích ứng
            self._update_progress(session, 0.35, AdaptationStatus.PLANNING)
            with Timer() as timer:
                adaptation_strategy = self._determine_adaptation_strategy(
                    session, analysis_result
                )
            session.track_timing("strategy_determination", timer.duration)

            # 5. Triển khai lập kế hoạch thích ứng
            self._update_progress(session, 0.45, AdaptationStatus.PLANNING)
            with Timer() as timer:
                planning_result = self._perform_adaptation(session, adaptation_strategy)
            session.track_timing("plan_adaptation", timer.duration)

            if not planning_result:
                session.mark_failed("Không thể tạo kế hoạch thích ứng")
                return

            # 6. Tính toán liều
            self._update_progress(session, 0.65, AdaptationStatus.CALCULATING_DOSE)
            with Timer() as timer:
                dose_result = self._calculate_adapted_dose(session)
            session.track_timing("dose_calculation", timer.duration)

            if not dose_result:
                session.mark_failed("Không thể tính toán liều cho kế hoạch thích ứng")
                return

            # 7. Đánh giá kế hoạch thích ứng
            self._update_progress(session, 0.85, AdaptationStatus.EVALUATING)
            with Timer() as timer:
                evaluation_result = self._evaluate_adaptation(session)
            session.track_timing("plan_evaluation", timer.duration)

            # 8. Hoàn thành phiên
            self._update_progress(session, 1.0, AdaptationStatus.COMPLETED)
            session.mark_completed()

            # Gọi callback hoàn thành
            if self.on_complete_callback:
                self.on_complete_callback(session)

        except Exception as e:
            logger.error(
                f"Lỗi trong quá trình lập kế hoạch thích ứng: {str(e)}", exc_info=True
            )
            session.mark_failed(f"Lỗi không mong muốn: {str(e)}")

    def _update_progress(
        self,
        session: RealTimeAdaptiveSession,
        progress: float,
        status: AdaptationStatus,
    ):
        """Cập nhật tiến độ và trạng thái của phiên."""
        session.progress = progress
        session.set_status(status)

        # Gọi callback tiến độ
        if self.on_progress_callback:
            self.on_progress_callback(session, progress)

    def _register_new_image(self, session: RealTimeAdaptiveSession) -> bool:
        """Đăng ký hình ảnh mới với hình ảnh tham chiếu."""
        try:
            session.log("Bắt đầu đăng ký hình ảnh mới với hình ảnh tham chiếu")

            # Lấy hình ảnh tham chiếu từ kế hoạch ban đầu
            reference_image = session.original_plan.image

            # Thực hiện đăng ký hình ảnh
            registration_result = self.image_registration.register(
                fixed_image=reference_image,
                moving_image=session.new_image,
                method="rigid",  # Đăng ký cứng để bắt đầu
                options={"max_iterations": 200},
            )

            if not registration_result.success:
                session.log(f"Đăng ký hình ảnh thất bại: {registration_result.message}")
                return False

            # Lưu kết quả vào phiên
            session.registration_result = registration_result
            session.log(
                f"Đăng ký hình ảnh thành công. Ma trận chuyển đổi: {registration_result.transform_matrix}"
            )

            return True

        except Exception as e:
            session.log(f"Lỗi khi đăng ký hình ảnh: {str(e)}")
            return False

    def _segment_structures(self, session: RealTimeAdaptiveSession) -> bool:
        """Phân đoạn cấu trúc tự động từ hình ảnh mới."""
        try:
            session.log("Bắt đầu phân đoạn tự động cấu trúc từ hình ảnh mới")

            # Lấy danh sách cấu trúc cần phân đoạn từ kế hoạch ban đầu
            original_structures = session.original_plan.structures

            # Danh sách các cấu trúc cần phân đoạn
            structures_to_segment = []
            for struct in original_structures:
                # Ưu tiên phân đoạn PTV và OAR
                if struct.is_target or struct.is_oar:
                    structures_to_segment.append(struct.id)

            # Thực hiện phân đoạn tự động
            segmentation_result = self.auto_segmenter.segment_multiple(
                image=session.new_image,
                structure_ids=structures_to_segment,
                options={"use_gpu": True, "batch_size": 4},
            )

            if not segmentation_result.success:
                session.log(
                    f"Phân đoạn tự động thất bại: {segmentation_result.message}"
                )
                return False

            # Lưu kết quả vào phiên
            session.adapted_structures = segmentation_result.structures

            # Ghi nhật ký
            successful_structs = [
                s_id for s_id, s in session.adapted_structures.items() if s is not None
            ]
            session.log(
                f"Phân đoạn tự động thành công cho {len(successful_structs)}/{len(structures_to_segment)} cấu trúc"
            )

            return True

        except Exception as e:
            session.log(f"Lỗi khi phân đoạn cấu trúc: {str(e)}")
            return False

    def _analyze_anatomical_changes(
        self, session: RealTimeAdaptiveSession
    ) -> Dict[str, Any]:
        """Phân tích thay đổi giải phẫu giữa hình ảnh mới và hình ảnh tham chiếu."""
        try:
            session.log("Bắt đầu phân tích thay đổi giải phẫu")

            # Lấy cấu trúc ban đầu từ kế hoạch
            original_structures = session.original_plan.structures

            # Lấy cấu trúc mới từ phân đoạn tự động
            adapted_structures = session.adapted_structures

            # Kết quả phân tích
            analysis_result = {
                "position_changes": {},
                "volume_changes": {},
                "dice_coefficients": {},
                "hausdorff_distances": {},
            }

            # So sánh từng cấu trúc
            for struct_id, original_struct in original_structures.items():
                if struct_id not in adapted_structures:
                    continue

                adapted_struct = adapted_structures[struct_id]

                # So sánh thể tích
                original_vol = original_struct.get_volume()
                adapted_vol = adapted_struct.get_volume()
                vol_change_pct = (
                    100.0 * (adapted_vol - original_vol) / original_vol
                    if original_vol > 0
                    else 0.0
                )
                analysis_result["volume_changes"][struct_id] = vol_change_pct

                # So sánh vị trí (trọng tâm)
                original_center = original_struct.get_center()
                adapted_center = adapted_struct.get_center()
                if original_center is not None and adapted_center is not None:
                    distance = np.linalg.norm(
                        np.array(adapted_center) - np.array(original_center)
                    )
                    analysis_result["position_changes"][struct_id] = distance

                # Tính hệ số Dice
                dice = self._calculate_dice(original_struct, adapted_struct)
                analysis_result["dice_coefficients"][struct_id] = dice

                # Tính khoảng cách Hausdorff
                hausdorff = self._calculate_hausdorff(original_struct, adapted_struct)
                analysis_result["hausdorff_distances"][struct_id] = hausdorff

                # Xác định loại thay đổi và thêm vào phiên
                if abs(vol_change_pct) > self.tolerance_thresholds["volume"]:
                    change_type = (
                        AnatomicalChangeType.TARGET_SIZE
                        if original_struct.is_target
                        else AnatomicalChangeType.OAR_SHAPE
                    )
                    session.add_anatomical_change(
                        change_type=change_type,
                        structure_id=struct_id,
                        details={"volume_change_pct": vol_change_pct},
                    )

                if (
                    analysis_result["position_changes"].get(struct_id, 0)
                    > self.tolerance_thresholds["position"]
                ):
                    change_type = (
                        AnatomicalChangeType.TARGET_POSITION
                        if original_struct.is_target
                        else AnatomicalChangeType.OAR_POSITION
                    )
                    session.add_anatomical_change(
                        change_type=change_type,
                        structure_id=struct_id,
                        details={
                            "position_change_mm": analysis_result["position_changes"][
                                struct_id
                            ]
                        },
                    )

            # Ghi nhật ký
            significant_changes = [c["type"].name for c in session.anatomical_changes]
            if significant_changes:
                session.log(
                    f"Phát hiện các thay đổi đáng kể: {', '.join(significant_changes)}"
                )
            else:
                session.log("Không phát hiện thay đổi đáng kể")

            return analysis_result

        except Exception as e:
            session.log(f"Lỗi khi phân tích thay đổi giải phẫu: {str(e)}")
            return {}

    def _calculate_dice(self, struct1: Structure, struct2: Structure) -> float:
        """Tính hệ số Dice giữa hai cấu trúc."""
        try:
            # Lấy các mặt nạ nhị phân
            mask1 = struct1.get_binary_mask()
            mask2 = struct2.get_binary_mask()

            if mask1 is None or mask2 is None:
                return 0.0

            # Tính toán hệ số Dice
            intersection = np.sum(np.logical_and(mask1, mask2))
            dice = (2.0 * intersection) / (np.sum(mask1) + np.sum(mask2))

            return float(dice)
        except Exception as e:
            logger.error(f"Lỗi khi tính hệ số Dice: {str(e)}")
            return 0.0

    def _calculate_hausdorff(self, struct1: Structure, struct2: Structure) -> float:
        """Tính khoảng cách Hausdorff giữa hai cấu trúc."""
        try:
            # Lấy các điểm bề mặt
            surface1 = struct1.get_surface_points()
            surface2 = struct2.get_surface_points()

            if not surface1 or not surface2:
                return 100.0  # Giá trị lớn nếu không thể tính toán

            # Tính toán khoảng cách Hausdorff
            max_distance = 0.0

            for point1 in surface1:
                min_dist_to_surface2 = min(
                    np.linalg.norm(np.array(point1) - np.array(point2))
                    for point2 in surface2
                )
                max_distance = max(max_distance, min_dist_to_surface2)

            for point2 in surface2:
                min_dist_to_surface1 = min(
                    np.linalg.norm(np.array(point2) - np.array(point1))
                    for point1 in surface1
                )
                max_distance = max(max_distance, min_dist_to_surface1)

            return float(max_distance)
        except Exception as e:
            logger.error(f"Lỗi khi tính khoảng cách Hausdorff: {str(e)}")
            return 100.0

    def _determine_adaptation_strategy(
        self, session: RealTimeAdaptiveSession, analysis_result: Dict[str, Any]
    ) -> AdaptiveActionType:
        """Xác định chiến lược thích ứng dựa trên kết quả phân tích."""
        try:
            session.log("Xác định chiến lược thích ứng phù hợp")

            # Thay đổi hình dạng và kích thước lớn nhất
            max_vol_change = max(
                [abs(v) for v in analysis_result.get("volume_changes", {}).values()],
                default=0,
            )

            # Thay đổi vị trí lớn nhất
            max_pos_change = max(
                analysis_result.get("position_changes", {}).values(), default=0
            )

            # Dice thấp nhất (chỉ số tương đồng)
            min_dice = min(
                analysis_result.get("dice_coefficients", {}).values(), default=1.0
            )

            # Xác định loại thích ứng:
            # 1. Nếu thay đổi nhỏ, không cần điều chỉnh
            if (
                max_vol_change < self.tolerance_thresholds["volume"]
                and max_pos_change < self.tolerance_thresholds["position"]
                and min_dice > self.tolerance_thresholds["dice"]
            ):
                session.set_priority(AdaptationPriority.NO_ADAPTATION)
                session.log("Không cần điều chỉnh kế hoạch (thay đổi không đáng kể)")
                return AdaptiveActionType.CONTINUE_TREATMENT

            # 2. Nếu chỉ có thay đổi vị trí, dịch isocenter
            if (
                max_vol_change < self.tolerance_thresholds["volume"]
                and max_pos_change >= self.tolerance_thresholds["position"]
                and min_dice > self.tolerance_thresholds["dice"]
            ):
                session.set_priority(AdaptationPriority.MEDIUM)
                session.log("Cần điều chỉnh vị trí isocenter")
                return AdaptiveActionType.ISOCENTER_SHIFT

            # 3. Nếu có thay đổi kích thước hoặc hình dạng, tối ưu lại kế hoạch
            if (
                max_vol_change >= self.tolerance_thresholds["volume"]
                or min_dice <= self.tolerance_thresholds["dice"]
            ):
                if max_vol_change > 2 * self.tolerance_thresholds["volume"]:
                    # Thay đổi lớn -> lập kế hoạch lại hoàn toàn
                    session.set_priority(AdaptationPriority.HIGH)
                    session.log("Cần lập kế hoạch lại hoàn toàn (thay đổi lớn)")
                    return AdaptiveActionType.COMPLETE_REPLANNING
                else:
                    # Thay đổi vừa phải -> tối ưu lại
                    session.set_priority(AdaptationPriority.MEDIUM)
                    session.log("Cần tối ưu lại kế hoạch")
                    return AdaptiveActionType.REOPTIMIZE

            # Mặc định
            session.set_priority(AdaptationPriority.LOW)
            session.log("Tối ưu lại kế hoạch (mặc định)")
            return AdaptiveActionType.REOPTIMIZE

        except Exception as e:
            session.log(f"Lỗi khi xác định chiến lược thích ứng: {str(e)}")
            session.set_priority(AdaptationPriority.LOW)
            return AdaptiveActionType.REOPTIMIZE

    def _perform_adaptation(
        self, session: RealTimeAdaptiveSession, action_type: AdaptiveActionType
    ) -> bool:
        """Thực hiện thích ứng kế hoạch theo chiến lược đã xác định."""
        try:
            session.log(
                f"Bắt đầu thích ứng kế hoạch theo chiến lược: {action_type.name}"
            )

            # Thực hiện theo từng loại thích ứng
            if action_type == AdaptiveActionType.CONTINUE_TREATMENT:
                # Không điều chỉnh, sử dụng kế hoạch ban đầu
                session.adapted_plan = session.original_plan
                session.log(
                    "Không điều chỉnh kế hoạch, tiếp tục điều trị với kế hoạch hiện tại"
                )
                return True

            elif action_type == AdaptiveActionType.ISOCENTER_SHIFT:
                # Điều chỉnh vị trí isocenter
                return self._adapt_with_isocenter_shift(session)

            elif action_type == AdaptiveActionType.REOPTIMIZE:
                # Tối ưu lại kế hoạch
                return self._adapt_with_reoptimization(session)

            elif action_type == AdaptiveActionType.COMPLETE_REPLANNING:
                # Lập kế hoạch lại hoàn toàn
                return self._adapt_with_complete_replanning(session)

            else:
                session.log(
                    f"Chiến lược thích ứng không được hỗ trợ: {action_type.name}"
                )
                return False

        except Exception as e:
            session.log(f"Lỗi khi thực hiện thích ứng kế hoạch: {str(e)}")
            return False

    def _adapt_with_isocenter_shift(self, session: RealTimeAdaptiveSession) -> bool:
        """Thích ứng kế hoạch bằng cách dịch chuyển isocenter."""
        try:
            session.log("Bắt đầu điều chỉnh vị trí isocenter")

            # Tạo bản sao kế hoạch
            adapted_plan = session.original_plan.clone()

            # Tính toán vector dịch chuyển
            # Ưu tiên dịch chuyển theo sự thay đổi của PTV
            shift_vector = [0, 0, 0]

            for struct_id in session.original_plan.structures:
                if session.original_plan.structures[struct_id].is_target:
                    # Đây là PTV hoặc GTV
                    if struct_id in session.adapted_structures:
                        original_center = session.original_plan.structures[
                            struct_id
                        ].get_center()
                        adapted_center = session.adapted_structures[
                            struct_id
                        ].get_center()

                        if original_center and adapted_center:
                            shift_vector = [
                                adapted_center[0] - original_center[0],
                                adapted_center[1] - original_center[1],
                                adapted_center[2] - original_center[2],
                            ]
                            break

            # Áp dụng dịch chuyển cho tất cả isocenter
            for beam in adapted_plan.beams:
                beam.shift_isocenter(shift_vector)

            # Cập nhật các thông tin khác
            adapted_plan.structures = session.adapted_structures
            adapted_plan.image = session.new_image
            adapted_plan.set_id(f"{session.original_plan.id}_adapted_{get_timestamp()}")

            # Lưu kế hoạch đã điều chỉnh
            session.adapted_plan = adapted_plan

            session.log(
                f"Đã điều chỉnh isocenter với vector dịch chuyển: {shift_vector}"
            )
            return True

        except Exception as e:
            session.log(f"Lỗi khi điều chỉnh isocenter: {str(e)}")
            return False

    def _adapt_with_reoptimization(self, session: RealTimeAdaptiveSession) -> bool:
        """Thích ứng kế hoạch bằng cách tối ưu lại kế hoạch hiện tại."""
        try:
            session.log("Bắt đầu tối ưu lại kế hoạch hiện tại")

            # Tạo bản sao kế hoạch
            adapted_plan = session.original_plan.clone()

            # Cập nhật cấu trúc và hình ảnh
            adapted_plan.structures = session.adapted_structures
            adapted_plan.image = session.new_image
            adapted_plan.set_id(f"{session.original_plan.id}_reopt_{get_timestamp()}")

            # Tạo đối tượng tối ưu hóa
            optimizer = self.optimizer.clone()

            # Thiết lập các thông số tối ưu hóa từ kế hoạch ban đầu
            optimizer.set_objectives_from_plan(session.original_plan)

            # Tối ưu hóa lại kế hoạch
            optimization_result = optimizer.optimize(adapted_plan)

            if not optimization_result.success:
                session.log(f"Tối ưu hóa thất bại: {optimization_result.message}")
                return False

            # Lưu kế hoạch đã tối ưu hóa
            session.adapted_plan = adapted_plan

            session.log(
                f"Đã tối ưu lại kế hoạch. Số vòng lặp: {optimization_result.iterations}, Giá trị mục tiêu: {optimization_result.objective_value:.4f}"
            )
            return True

        except Exception as e:
            session.log(f"Lỗi khi tối ưu lại kế hoạch: {str(e)}")
            return False

    def _adapt_with_complete_replanning(self, session: RealTimeAdaptiveSession) -> bool:
        """Thích ứng kế hoạch bằng cách lập kế hoạch lại hoàn toàn."""
        try:
            session.log("Bắt đầu lập kế hoạch lại hoàn toàn")

            # Sử dụng adaptive planner để hoàn thành tác vụ
            # Vì đây là quá trình phức tạp hơn nên chúng ta sử dụng module chuyên biệt
            adapted_plan = self.adaptive_planner.create_new_plan_from_existing(
                patient=session.patient,
                template_plan=session.original_plan,
                new_image=session.new_image,
                new_structures=session.adapted_structures,
            )

            if not adapted_plan:
                session.log("Không thể tạo kế hoạch mới")
                return False

            # Lưu kế hoạch đã lập
            session.adapted_plan = adapted_plan

            session.log("Đã lập kế hoạch mới thành công")
            return True

        except Exception as e:
            session.log(f"Lỗi khi lập kế hoạch lại: {str(e)}")
            return False

    def _calculate_adapted_dose(self, session: RealTimeAdaptiveSession) -> bool:
        """Tính toán liều cho kế hoạch đã thích ứng."""
        try:
            session.log("Bắt đầu tính toán liều cho kế hoạch đã thích ứng")

            # Tính toán liều
            dose_result = self.dose_calculator.calculate_dose(
                plan=session.adapted_plan,
                algorithm="monte_carlo"
                if session.priority
                in [AdaptationPriority.HIGH, AdaptationPriority.CRITICAL]
                else "collapsed_cone",
                options={"use_gpu": True},
            )

            if not dose_result.success:
                session.log(f"Tính toán liều thất bại: {dose_result.message}")
                return False

            # Lưu liều đã tính toán
            session.adapted_dose = dose_result.dose
            session.adapted_plan.set_dose(dose_result.dose)

            session.log("Đã tính toán liều thành công")
            return True

        except Exception as e:
            session.log(f"Lỗi khi tính toán liều: {str(e)}")
            return False

    def _evaluate_adaptation(self, session: RealTimeAdaptiveSession) -> bool:
        """Đánh giá kế hoạch đã thích ứng."""
        try:
            session.log("Bắt đầu đánh giá kế hoạch đã thích ứng")

            # Tính toán DVH cho kế hoạch mới
            from quangtps.evaluation.dvh.dvh_calculator import DVHCalculator

            dvh_calculator = DVHCalculator()

            # DVH cho kế hoạch ban đầu
            original_dvhs = dvh_calculator.calculate(
                dose=session.original_plan.dose,
                structures=session.original_plan.structures,
            )

            # DVH cho kế hoạch đã thích ứng
            adapted_dvhs = dvh_calculator.calculate(
                dose=session.adapted_dose, structures=session.adapted_structures
            )

            # So sánh và đánh giá
            from quangtps.evaluation.plan_evaluation import PlanEvaluator

            evaluator = PlanEvaluator()

            evaluation_result = evaluator.compare_plans(
                reference_plan=session.original_plan,
                new_plan=session.adapted_plan,
                reference_dvhs=original_dvhs,
                new_dvhs=adapted_dvhs,
            )

            # Lưu kết quả đánh giá
            session.evaluation_metrics = evaluation_result

            # Ghi nhật ký
            for structure_id, metrics in evaluation_result.items():
                for metric_name, value in metrics.items():
                    if (
                        structure_id in session.original_plan.structures
                        and session.original_plan.structures[structure_id].is_target
                    ):
                        session.log(f"PTV {structure_id}, {metric_name}: {value}")
                    elif (
                        structure_id in session.original_plan.structures
                        and session.original_plan.structures[structure_id].is_oar
                    ):
                        session.log(f"OAR {structure_id}, {metric_name}: {value}")

            session.log("Đã hoàn thành đánh giá kế hoạch")
            return True

        except Exception as e:
            session.log(f"Lỗi khi đánh giá kế hoạch: {str(e)}")
            return False

    def get_session(self, session_id: str) -> Optional[RealTimeAdaptiveSession]:
        """Lấy phiên theo ID."""
        return self.active_sessions.get(session_id)

    def cancel_session(self, session_id: str) -> bool:
        """Hủy một phiên đang chạy."""
        if session_id not in self.active_sessions:
            return False

        session = self.active_sessions[session_id]
        session.mark_failed("Phiên bị hủy bởi người dùng")

        # Xóa khỏi danh sách phiên đang hoạt động
        if session_id in self.session_threads:
            # Thread không thể bị kill, nhưng nó sẽ kết thúc khi mark_failed được gọi
            del self.session_threads[session_id]

        return True

    def get_all_sessions(self) -> List[RealTimeAdaptiveSession]:
        """Lấy tất cả các phiên."""
        return list(self.active_sessions.values())

    def cleanup_completed_sessions(self, older_than_hours: int = 24) -> int:
        """Dọn dẹp các phiên đã hoàn thành cũ."""
        now = datetime.datetime.now()
        sessions_to_remove = []

        for session_id, session in self.active_sessions.items():
            if session.status in [AdaptationStatus.COMPLETED, AdaptationStatus.FAILED]:
                if session.end_time:
                    hours_old = (now - session.end_time).total_seconds() / 3600
                    if hours_old > older_than_hours:
                        sessions_to_remove.append(session_id)

        # Xóa các phiên cũ
        for session_id in sessions_to_remove:
            del self.active_sessions[session_id]
            if session_id in self.session_threads:
                del self.session_threads[session_id]

        return len(sessions_to_remove)


# Hàm trợ giúp để tạo một bộ lập kế hoạch thích ứng thời gian thực
def create_real_time_adaptive_planner() -> RealTimeAdaptivePlanner:
    """
    Tạo và cấu hình một bộ lập kế hoạch thích ứng thời gian thực.

    Returns
    -------
    RealTimeAdaptivePlanner
        Đối tượng bộ lập kế hoạch thích ứng đã cấu hình
    """
    from quangtps.segmentation.auto.auto_segmentation import AutoSegmenter
    from quangtps.imaging.registration import ImageRegistration
    from quangtps.dose.dose_calculation import DoseCalculation
    from quangtps.adaptive.deformation.deformable_registration import (
        DeformableRegistration,
    )
    from quangtps.optimization.methods.vmat_optimization import VMATOptimizer
    from quangtps.adaptive.adaptive_planning import AdaptivePlanner

    # Tạo các thành phần
    auto_segmenter = AutoSegmenter()
    image_registration = ImageRegistration()
    dose_calculator = DoseCalculation()
    deformable_registration = DeformableRegistration()
    optimizer = VMATOptimizer()  # Sử dụng VMAT Optimizer mặc định
    adaptive_planner = AdaptivePlanner()

    # Tạo bộ lập kế hoạch thích ứng
    planner = RealTimeAdaptivePlanner(
        auto_segmenter=auto_segmenter,
        image_registration=image_registration,
        dose_calculator=dose_calculator,
        deformable_registration=deformable_registration,
        optimizer=optimizer,
        adaptive_planner=adaptive_planner,
    )

    return planner


if __name__ == "__main__":
    # Mã để chạy thử và kiểm tra module
    logging.basicConfig(level=logging.INFO)
    logger.info("Kiểm tra module real_time_adaptive_planning.py")

    # Tạo bộ lập kế hoạch thích ứng
    planner = create_real_time_adaptive_planner()
    logger.info("Đã tạo bộ lập kế hoạch thích ứng thời gian thực thành công")
