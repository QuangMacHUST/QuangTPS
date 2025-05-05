#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module tạo kế hoạch điều trị tự động cho QuangTPS.

Module này triển khai các thuật toán và quy trình cho phép tạo kế hoạch
điều trị tự động dựa trên các mẫu lâm sàng và quy trình chuẩn hóa.
Mục tiêu là cung cấp kết quả tương đương với Eclipse của Varian.
"""

import os
import logging
import time
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union, Set

from quangtps.core.patient import Patient
from quangtps.core.plan import Plan
from quangtps.optimization.optimizer import PlanOptimizer
from quangtps.segmentation.auto_segmentation import AutoSegmentation
from quangtps.treatment.beams.beam_configuration import BeamConfiguration
from quangtps.treatment.techniques.technique_factory import TechniqueFactory
from quangtps.optimization.objectives import ObjectiveFunction, ObjectiveType
from quangtps.dose.dose_calculation import DoseCalculator
from quangtps.evaluation.metrics.clinical_metrics import ClinicalMetricsCalculator
from quangtps.planning.templates import TemplateManager
from quangtps.core.exceptions import ValidationError
from quangtps.utils.io_utils import create_directory_if_not_exists

logger = logging.getLogger(__name__)


class AutoPlanningEngine:
    """
    Động cơ lập kế hoạch tự động.

    Lớp này quản lý quá trình tạo kế hoạch điều trị tự động từ mẫu đến
    kế hoạch hoàn chỉnh với tối ưu hóa đầy đủ.
    """

    def __init__(
        self, patient: Patient = None, plan: Plan = None, template_path: str = None
    ):
        """
        Khởi tạo động cơ lập kế hoạch tự động.

        Args:
            patient: Đối tượng bệnh nhân (nếu có)
            plan: Kế hoạch hiện tại (nếu có)
            template_path: Đường dẫn đến thư mục chứa mẫu kế hoạch
        """
        self.patient = patient
        self.plan = plan

        # Khởi tạo các thành phần liên quan
        self.template_manager = TemplateManager(template_path)
        self.optimizer = PlanOptimizer()
        self.dose_calculator = DoseCalculator()
        self.beam_configurator = BeamConfiguration()
        self.technique_factory = TechniqueFactory()
        self.metrics_calculator = ClinicalMetricsCalculator()

        # Trạng thái và tiến độ
        self.is_running = False
        self.progress = 0.0
        self.status = "Sẵn sàng"
        self.callbacks = []
        self.template = None

        # Kết quả
        self.optimization_history = []
        self.evaluation_results = None

        # Kiểm tra môi trường
        self._setup_environment()

        logger.info("Đã khởi tạo AutoPlanningEngine")

    def _setup_environment(self):
        """Thiết lập môi trường cho tạo kế hoạch tự động."""
        # Tạo thư mục cache nếu cần
        cache_dir = Path(os.path.expanduser("~/.quangtps/auto_planning_cache"))
        create_directory_if_not_exists(str(cache_dir))

        # Đặt thư mục mẫu mặc định nếu không được chỉ định
        if not self.template_manager.template_directory:
            default_templates = (
                Path(__file__).parent.parent.parent
                / "data"
                / "templates"
                / "auto_planning"
            )
            if default_templates.exists():
                self.template_manager.set_template_directory(str(default_templates))
            else:
                logger.warning(
                    f"Thư mục mẫu mặc định không tồn tại: {default_templates}"
                )

    def set_patient(self, patient: Patient) -> None:
        """
        Thiết lập bệnh nhân hiện tại.

        Args:
            patient: Đối tượng bệnh nhân
        """
        self.patient = patient
        logger.info(f"Đã thiết lập bệnh nhân: {patient.name}")

    def set_plan(self, plan: Plan) -> None:
        """
        Thiết lập kế hoạch hiện tại.

        Args:
            plan: Đối tượng kế hoạch
        """
        self.plan = plan
        logger.info(f"Đã thiết lập kế hoạch: {plan.name}")

    def load_template(self, template_name: str) -> bool:
        """
        Tải mẫu kế hoạch từ tên.

        Args:
            template_name: Tên của mẫu kế hoạch

        Returns:
            True nếu tải thành công, False nếu thất bại
        """
        try:
            self.template = self.template_manager.load_template(template_name)
            if not self.template:
                logger.error(f"Không thể tải mẫu: {template_name}")
                return False

            logger.info(f"Đã tải mẫu kế hoạch: {template_name}")
            return True
        except Exception as e:
            logger.error(f"Lỗi khi tải mẫu kế hoạch: {e}")
            return False

    def add_progress_callback(self, callback) -> None:
        """
        Thêm callback để nhận thông báo tiến độ.

        Args:
            callback: Hàm callback nhận (progress, status)
        """
        if callback not in self.callbacks:
            self.callbacks.append(callback)

    def _update_progress(self, progress: float, status: str = None) -> None:
        """
        Cập nhật tiến độ và gọi các callback.

        Args:
            progress: Tiến độ (0-1)
            status: Thông tin trạng thái
        """
        self.progress = progress
        if status:
            self.status = status

        # Gọi tất cả callback
        for callback in self.callbacks:
            try:
                callback(progress, status)
            except Exception as e:
                logger.error(f"Lỗi trong callback tiến độ: {e}")

    def create_plan(self, plan_name: str, template_name: str) -> Optional[Plan]:
        """
        Tạo kế hoạch mới dựa trên mẫu.

        Args:
            plan_name: Tên kế hoạch
            template_name: Tên mẫu kế hoạch

        Returns:
            Kế hoạch mới nếu thành công, None nếu thất bại
        """
        if not self.patient:
            logger.error("Không có bệnh nhân nào được chọn để tạo kế hoạch")
            return None

        # Tải mẫu kế hoạch
        if not self.load_template(template_name):
            return None

        # Đánh dấu bắt đầu
        self.is_running = True
        start_time = time.time()
        self._update_progress(0.01, "Đang tạo kế hoạch...")

        try:
            # Tạo kế hoạch mới
            self.plan = self.patient.create_plan(plan_name)

            if not self.plan:
                logger.error(f"Không thể tạo kế hoạch mới với tên: {plan_name}")
                return None

            # Áp dụng thiết lập từ mẫu
            self._apply_template_settings()

            # Cập nhật tiến độ
            self._update_progress(0.2, "Đã áp dụng thiết lập mẫu")

            # Hoàn thành
            elapsed = time.time() - start_time
            self._update_progress(1.0, f"Đã tạo kế hoạch từ mẫu trong {elapsed:.1f}s")

            return self.plan

        except Exception as e:
            logger.error(f"Lỗi khi tạo kế hoạch từ mẫu: {e}")
            self._update_progress(1.0, f"Lỗi: {str(e)}")
            return None

        finally:
            self.is_running = False

    def _apply_template_settings(self) -> None:
        """Áp dụng cài đặt từ mẫu vào kế hoạch hiện tại."""
        if not self.template or not self.plan:
            return

        # Thiết lập liều kê toa
        if "prescription" in self.template:
            prescription = self.template["prescription"]
            self.plan.set_prescription(
                dose=prescription.get("dose", 0),
                fractions=prescription.get("fractions", 1),
            )

        # Thiết lập các chùm tia
        if "beams" in self.template:
            for beam_info in self.template["beams"]:
                # Xử lý chùm tia thông thường
                if beam_info.get("type") == "static":
                    self._create_static_beam(beam_info)
                # Xử lý cung VMAT
                elif beam_info.get("type") == "arc":
                    self._create_arc_beam(beam_info)

        # Thiết lập mục tiêu tối ưu
        if "objectives" in self.template:
            for obj_info in self.template["objectives"]:
                self._add_objective(obj_info)

    def _create_static_beam(self, beam_info: Dict[str, Any]) -> None:
        """
        Tạo chùm tia tĩnh từ thông tin.

        Args:
            beam_info: Thông tin chùm tia
        """
        name = beam_info.get("name", f"Beam_{len(self.plan.beams) + 1}")
        gantry = beam_info.get("gantry_angle", 0)
        collimator = beam_info.get("collimator_angle", 0)
        couch = beam_info.get("couch_angle", 0)
        energy = beam_info.get("energy", 6)

        # Thêm chùm tia vào kế hoạch
        self.plan.add_beam(
            name=name,
            gantry_angle=gantry,
            collimator_angle=collimator,
            couch_angle=couch,
            energy=energy,
        )

        logger.info(f"Đã tạo chùm tia tĩnh: {name}")

    def _create_arc_beam(self, beam_info: Dict[str, Any]) -> None:
        """
        Tạo chùm tia dạng cung từ thông tin.

        Args:
            beam_info: Thông tin chùm tia
        """
        name = beam_info.get("name", f"Arc_{len(self.plan.beams) + 1}")
        start_angle = beam_info.get("start_angle", 181)
        stop_angle = beam_info.get("stop_angle", 179)
        collimator = beam_info.get("collimator_angle", 0)
        couch = beam_info.get("couch_angle", 0)
        energy = beam_info.get("energy", 6)

        # Kiểm tra hướng quay
        direction = beam_info.get("direction", "CW")
        is_clockwise = direction.upper() == "CW"

        # Thêm cung VMAT vào kế hoạch
        self.plan.add_arc(
            name=name,
            start_angle=start_angle,
            stop_angle=stop_angle,
            collimator_angle=collimator,
            couch_angle=couch,
            energy=energy,
            is_clockwise=is_clockwise,
        )

        logger.info(f"Đã tạo cung VMAT: {name} ({start_angle}° → {stop_angle}°)")

    def _add_objective(self, obj_info: Dict[str, Any]) -> None:
        """
        Thêm mục tiêu từ thông tin.

        Args:
            obj_info: Thông tin mục tiêu
        """
        structure_name = obj_info.get("structure")
        if not structure_name:
            logger.warning("Không tìm thấy tên cấu trúc trong mục tiêu, bỏ qua")
            return

        # Tìm cấu trúc theo tên
        structure = next(
            (s for s in self.plan.structures if s.name == structure_name), None
        )
        if not structure:
            logger.warning(
                f"Không tìm thấy cấu trúc: {structure_name}, bỏ qua mục tiêu"
            )
            return

        # Xác định loại mục tiêu
        type_str = obj_info.get("type", "").upper()
        if not type_str:
            logger.warning(f"Không tìm thấy loại mục tiêu cho {structure_name}, bỏ qua")
            return

        try:
            obj_type = ObjectiveType[type_str]
        except KeyError:
            logger.warning(f"Loại mục tiêu không hợp lệ: {type_str}, bỏ qua")
            return

        # Tạo và thêm mục tiêu
        parameter = obj_info.get("parameter", 0)
        weight = obj_info.get("weight", 100)

        objective = ObjectiveFunction(
            structure=structure, type=obj_type, parameter=parameter, weight=weight
        )

        self.optimizer.add_objective(objective)
        logger.info(f"Đã thêm mục tiêu {type_str} cho {structure_name}")

    def run_optimization(self, max_iterations: int = 100) -> bool:
        """
        Chạy tối ưu hóa cho kế hoạch.

        Args:
            max_iterations: Số lần lặp tối đa

        Returns:
            True nếu tối ưu hóa thành công, False nếu thất bại
        """
        if not self.plan:
            logger.error("Không có kế hoạch nào được thiết lập để tối ưu hóa")
            return False

        # Đánh dấu bắt đầu
        self.is_running = True
        start_time = time.time()
        self._update_progress(0.01, "Đang bắt đầu tối ưu hóa...")

        try:
            # Thiết lập optimizer
            self.optimizer.set_plan(self.plan)

            # Đặt callback tiến độ
            def progress_callback(iteration, metrics):
                progress = min(iteration / max_iterations, 0.95)
                status = f"Đang tối ưu hóa (lần lặp {iteration}/{max_iterations})"
                self._update_progress(progress, status)

                # Lưu lịch sử tối ưu
                self.optimization_history.append(
                    {"iteration": iteration, "metrics": metrics}
                )

            # Chạy tối ưu hóa
            self.optimizer.set_progress_callback(progress_callback)
            result = self.optimizer.optimize(max_iterations=max_iterations)

            if not result:
                logger.error("Tối ưu hóa thất bại")
                self._update_progress(1.0, "Tối ưu hóa thất bại")
                return False

            # Tính toán liều cuối cùng
            self._update_progress(0.95, "Đang tính toán liều cuối cùng...")
            self.dose_calculator.calculate(self.plan)

            # Hoàn thành
            elapsed = time.time() - start_time
            self._update_progress(1.0, f"Đã hoàn thành tối ưu hóa trong {elapsed:.1f}s")

            return True

        except Exception as e:
            logger.error(f"Lỗi trong quá trình tối ưu hóa: {e}")
            self._update_progress(1.0, f"Lỗi: {str(e)}")
            return False

        finally:
            self.is_running = False

    def evaluate_plan(self) -> Dict[str, Any]:
        """
        Đánh giá kế hoạch hiện tại.

        Returns:
            Kết quả đánh giá
        """
        if not self.plan:
            logger.error("Không có kế hoạch nào để đánh giá")
            return {}

        # Kiểm tra liều đã được tính toán
        if not self.plan.has_dose():
            logger.warning("Kế hoạch chưa có liều, tính toán liều trước khi đánh giá")
            self.dose_calculator.calculate(self.plan)

        try:
            # Đánh giá kế hoạch
            evaluation = self.metrics_calculator.calculate(self.plan)

            # Lưu kết quả
            self.evaluation_results = evaluation

            return evaluation

        except Exception as e:
            logger.error(f"Lỗi khi đánh giá kế hoạch: {e}")
            return {}

    def run_full_auto_planning(
        self, patient: Patient, plan_name: str, template_name: str
    ) -> Optional[Plan]:
        """
        Chạy toàn bộ quy trình tự động từ đầu đến cuối.

        Args:
            patient: Đối tượng bệnh nhân
            plan_name: Tên kế hoạch
            template_name: Tên mẫu kế hoạch

        Returns:
            Kế hoạch đã tối ưu nếu thành công, None nếu thất bại
        """
        # Thiết lập bệnh nhân
        self.set_patient(patient)

        # Tạo kế hoạch từ mẫu
        plan = self.create_plan(plan_name, template_name)
        if not plan:
            logger.error("Không thể tạo kế hoạch từ mẫu")
            return None

        # Chạy tối ưu hóa
        if not self.run_optimization():
            logger.error("Không thể tối ưu hóa kế hoạch")
            return plan

        # Đánh giá kế hoạch
        evaluation = self.evaluate_plan()
        logger.info(f"Đã hoàn thành đánh giá kế hoạch: {len(evaluation)} chỉ số")

        return plan

    def save_template(self, template_name: str, description: str = "") -> bool:
        """
        Lưu kế hoạch hiện tại như một mẫu.

        Args:
            template_name: Tên mẫu mới
            description: Mô tả mẫu

        Returns:
            True nếu lưu thành công, False nếu thất bại
        """
        if not self.plan:
            logger.error("Không có kế hoạch nào để lưu làm mẫu")
            return False

        try:
            # Tạo template từ kế hoạch hiện tại
            template = {
                "name": template_name,
                "description": description,
                "created": datetime.now().isoformat(),
                "prescription": {
                    "dose": self.plan.prescription.dose
                    if hasattr(self.plan, "prescription")
                    else 0,
                    "fractions": self.plan.prescription.fractions
                    if hasattr(self.plan, "prescription")
                    else 1,
                },
                "beams": [],
                "objectives": [],
            }

            # Thêm thông tin chùm tia
            for beam in self.plan.beams:
                if beam.is_arc:
                    beam_info = {
                        "type": "arc",
                        "name": beam.name,
                        "start_angle": beam.start_angle,
                        "stop_angle": beam.stop_angle,
                        "direction": "CW" if beam.is_clockwise else "CCW",
                        "collimator_angle": beam.collimator_angle,
                        "couch_angle": beam.couch_angle,
                        "energy": beam.energy,
                    }
                else:
                    beam_info = {
                        "type": "static",
                        "name": beam.name,
                        "gantry_angle": beam.gantry_angle,
                        "collimator_angle": beam.collimator_angle,
                        "couch_angle": beam.couch_angle,
                        "energy": beam.energy,
                    }

                template["beams"].append(beam_info)

            # Thêm thông tin mục tiêu
            for obj in self.optimizer.get_objectives():
                obj_info = {
                    "structure": obj.structure.name,
                    "type": obj.type.name,
                    "parameter": obj.parameter,
                    "weight": obj.weight,
                }

                template["objectives"].append(obj_info)

            # Lưu template
            return self.template_manager.save_template(template_name, template)

        except Exception as e:
            logger.error(f"Lỗi khi lưu mẫu kế hoạch: {e}")
            return False


class TemplateBasedPlanner:
    """
    Lớp lập kế hoạch dựa trên mẫu bệnh và vị trí điều trị.

    Lớp này cung cấp một giao diện đơn giản để tạo kế hoạch điều trị
    dựa trên các mẫu có sẵn theo vị trí điều trị.
    """

    # Định nghĩa các mẫu vị trí điều trị phổ biến
    SITES = {
        "PROSTATE": {
            "structures": [
                "PTV",
                "Bladder",
                "Rectum",
                "Femoral_Head_L",
                "Femoral_Head_R",
                "Bowel",
            ],
            "technique": "VMAT",
            "typical_dose": 7800,
            "typical_fractions": 39,
        },
        "BREAST": {
            "structures": ["PTV", "Heart", "Lung_L", "Lung_R", "Spinal_Cord"],
            "technique": "IMRT",
            "typical_dose": 5000,
            "typical_fractions": 25,
        },
        "LUNG": {
            "structures": [
                "PTV",
                "Lung_L",
                "Lung_R",
                "Heart",
                "Esophagus",
                "Spinal_Cord",
            ],
            "technique": "VMAT",
            "typical_dose": 6000,
            "typical_fractions": 30,
        },
        "HEAD_NECK": {
            "structures": [
                "PTV",
                "Brainstem",
                "Spinal_Cord",
                "Parotid_L",
                "Parotid_R",
                "Larynx",
                "Oral_Cavity",
            ],
            "technique": "IMRT",
            "typical_dose": 7000,
            "typical_fractions": 35,
        },
        "BRAIN": {
            "structures": [
                "PTV",
                "Brainstem",
                "Optic_Chiasm",
                "Optic_Nerve_L",
                "Optic_Nerve_R",
            ],
            "technique": "IMRT",
            "typical_dose": 6000,
            "typical_fractions": 30,
        },
    }

    def __init__(self, auto_planner: AutoPlanningEngine = None):
        """
        Khởi tạo lớp lập kế hoạch dựa trên mẫu.

        Args:
            auto_planner: Động cơ lập kế hoạch tự động (tùy chọn)
        """
        self.auto_planner = auto_planner or AutoPlanningEngine()
        logger.info("Đã khởi tạo TemplateBasedPlanner")

    @classmethod
    def get_available_sites(cls) -> List[str]:
        """
        Lấy danh sách các vị trí điều trị có sẵn.

        Returns:
            Danh sách tên các vị trí điều trị
        """
        return list(cls.SITES.keys())

    def create_site_based_plan(
        self, patient: Patient, site: str, plan_name: str = None
    ) -> Optional[Plan]:
        """
        Tạo kế hoạch dựa trên vị trí điều trị.

        Args:
            patient: Đối tượng bệnh nhân
            site: Tên vị trí điều trị
            plan_name: Tên kế hoạch (tùy chọn)

        Returns:
            Kế hoạch đã tạo nếu thành công, None nếu thất bại
        """
        # Kiểm tra vị trí hợp lệ
        site = site.upper()
        if site not in self.SITES:
            logger.error(f"Vị trí điều trị không được hỗ trợ: {site}")
            logger.info(f"Các vị trí được hỗ trợ: {', '.join(self.SITES.keys())}")
            return None

        # Tạo tên kế hoạch nếu không được cung cấp
        if not plan_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            plan_name = f"{site}_{timestamp}"

        # Tìm mẫu phù hợp cho vị trí
        template_name = f"{site.lower()}_standard"

        # Chạy quy trình tự động
        plan = self.auto_planner.run_full_auto_planning(
            patient=patient, plan_name=plan_name, template_name=template_name
        )

        return plan

    def suggest_objectives_for_site(self, site: str) -> List[Dict[str, Any]]:
        """
        Gợi ý các mục tiêu tối ưu dựa trên vị trí điều trị.

        Args:
            site: Tên vị trí điều trị

        Returns:
            Danh sách các mục tiêu được gợi ý
        """
        site = site.upper()
        if site not in self.SITES:
            logger.error(f"Vị trí điều trị không được hỗ trợ: {site}")
            return []

        # Gợi ý mục tiêu dựa trên vị trí
        if site == "PROSTATE":
            return [
                {"structure": "PTV", "type": "LOWER", "parameter": 95, "weight": 100},
                {"structure": "PTV", "type": "UPPER", "parameter": 103, "weight": 100},
                {
                    "structure": "Bladder",
                    "type": "UPPER",
                    "parameter": 80,
                    "weight": 80,
                },
                {"structure": "Rectum", "type": "UPPER", "parameter": 75, "weight": 80},
                {
                    "structure": "Femoral_Head_L",
                    "type": "UPPER",
                    "parameter": 50,
                    "weight": 50,
                },
                {
                    "structure": "Femoral_Head_R",
                    "type": "UPPER",
                    "parameter": 50,
                    "weight": 50,
                },
            ]
        elif site == "BREAST":
            return [
                {"structure": "PTV", "type": "LOWER", "parameter": 95, "weight": 100},
                {"structure": "PTV", "type": "UPPER", "parameter": 105, "weight": 100},
                {"structure": "Heart", "type": "MEAN", "parameter": 8, "weight": 80},
                {"structure": "Lung_L", "type": "UPPER", "parameter": 20, "weight": 60},
                {"structure": "Lung_R", "type": "UPPER", "parameter": 5, "weight": 40},
            ]
        elif site == "LUNG":
            return [
                {"structure": "PTV", "type": "LOWER", "parameter": 95, "weight": 100},
                {"structure": "PTV", "type": "UPPER", "parameter": 105, "weight": 100},
                {"structure": "Lung_L", "type": "UPPER", "parameter": 30, "weight": 70},
                {"structure": "Lung_R", "type": "UPPER", "parameter": 30, "weight": 70},
                {"structure": "Heart", "type": "MEAN", "parameter": 15, "weight": 60},
                {
                    "structure": "Spinal_Cord",
                    "type": "UPPER",
                    "parameter": 45,
                    "weight": 150,
                },
            ]
        elif site == "HEAD_NECK":
            return [
                {"structure": "PTV", "type": "LOWER", "parameter": 95, "weight": 100},
                {"structure": "PTV", "type": "UPPER", "parameter": 105, "weight": 100},
                {
                    "structure": "Brainstem",
                    "type": "UPPER",
                    "parameter": 54,
                    "weight": 150,
                },
                {
                    "structure": "Spinal_Cord",
                    "type": "UPPER",
                    "parameter": 45,
                    "weight": 150,
                },
                {
                    "structure": "Parotid_L",
                    "type": "MEAN",
                    "parameter": 26,
                    "weight": 60,
                },
                {
                    "structure": "Parotid_R",
                    "type": "MEAN",
                    "parameter": 26,
                    "weight": 60,
                },
            ]
        else:  # BRAIN
            return [
                {"structure": "PTV", "type": "LOWER", "parameter": 95, "weight": 100},
                {"structure": "PTV", "type": "UPPER", "parameter": 105, "weight": 100},
                {
                    "structure": "Brainstem",
                    "type": "UPPER",
                    "parameter": 54,
                    "weight": 150,
                },
                {
                    "structure": "Optic_Chiasm",
                    "type": "UPPER",
                    "parameter": 54,
                    "weight": 150,
                },
                {
                    "structure": "Optic_Nerve_L",
                    "type": "UPPER",
                    "parameter": 54,
                    "weight": 150,
                },
                {
                    "structure": "Optic_Nerve_R",
                    "type": "UPPER",
                    "parameter": 54,
                    "weight": 150,
                },
            ]
