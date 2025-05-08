#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module lập kế hoạch tự động (Auto-Planning).

Module này cung cấp các công cụ tự động hóa quá trình lập kế hoạch xạ trị,
dựa trên các mẫu lâm sàng và kế hoạch đã được chứng minh hiệu quả trước đó.
"""

import os
import time
import logging
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Set, Callable
from enum import Enum, auto
from dataclasses import dataclass, field
import threading
import copy

from quangtps.core.patient.patient import Patient
from quangtps.optimization.base import OptimizerBase
from quangtps.optimization.objectives import ObjectiveCriteria, ObjectiveFunction
from quangtps.optimization.constraints import ConstraintType, Constraint
from quangtps.optimization.optimizer import Optimizer
from quangtps.optimization.solver import OptimizationSolver
from quangtps.structures.structure_utils.structure_matcher import StructureMatcher
from quangtps.treatment.beams.beam_generator import BeamGenerator
from quangtps.evaluation.metrics.plan_metrics import calculate_plan_metrics
from quangtps.evaluation.dvh.dvh_calculator import DVHCalculator
from quangtps.utils.file_utils import ensure_directory_exists

logger = logging.getLogger(__name__)


class AutoPlanningMode(Enum):
    """Chế độ lập kế hoạch tự động."""

    TEMPLATE = auto()  # Sử dụng mẫu có sẵn
    KNOWLEDGE_BASED = auto()  # Dựa trên cơ sở tri thức
    PROTOCOL_BASED = auto()  # Dựa trên giao thức lâm sàng
    HYBRID = auto()  # Kết hợp


class AutoPlanningTarget(Enum):
    """Mục tiêu của lập kế hoạch tự động."""

    BEAM_SETUP = auto()  # Thiết lập chùm tia
    OPTIMIZATION_OBJECTIVES = auto()  # Mục tiêu tối ưu hóa
    FULL_PLAN = auto()  # Kế hoạch hoàn chỉnh


class PlanTemplate:
    """Mẫu kế hoạch xạ trị."""

    def __init__(
        self,
        name: str,
        site: str,
        description: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Khởi tạo mẫu kế hoạch.

        Parameters
        ----------
        name : str
            Tên mẫu kế hoạch
        site : str
            Vị trí điều trị (ví dụ: "prostate", "lung", "head_neck")
        description : str, optional
            Mô tả về mẫu kế hoạch, mặc định là ""
        metadata : Optional[Dict[str, Any]], optional
            Metadata bổ sung, mặc định là None
        """
        self.name = name
        self.site = site
        self.description = description
        self.metadata = metadata or {}

        # Các tham số liên quan đến chùm tia
        self.technique = "VMAT"  # VMAT, IMRT, 3DCRT, etc.
        self.beam_parameters: Dict[str, Any] = {}
        self.beam_arrangements: List[Dict[str, Any]] = []

        # Arc parameters if VMAT
        self.arc_parameters: Dict[str, Any] = {}

        # Các mục tiêu tối ưu hóa
        self.objectives: List[Dict[str, Any]] = []
        self.constraints: List[Dict[str, Any]] = []

        # Tham số tối ưu hóa
        self.optimization_parameters: Dict[str, Any] = {
            "max_iterations": 100,
            "convergence_tolerance": 1e-4,
            "use_multi_resolution": True,
            "resolution_levels": [8, 4, 2, 1],  # mm
            "priority_weights": {},
        }

        # Các cấu trúc yêu cầu
        self.required_structures: List[str] = []
        self.structure_mappings: Dict[str, List[str]] = {}

        # Các liều lượng tiêu chuẩn
        self.prescription: Dict[str, Any] = {}

    def add_beam_arrangement(self, technique: str, beam_params: Dict[str, Any]) -> None:
        """
        Thêm một cấu hình chùm tia vào mẫu.

        Parameters
        ----------
        technique : str
            Kỹ thuật điều trị (VMAT, IMRT, 3DCRT...)
        beam_params : Dict[str, Any]
            Tham số chùm tia
        """
        self.technique = technique
        arrangement = {"technique": technique, **beam_params}
        self.beam_arrangements.append(arrangement)

    def add_objective(
        self, structure_id: str, objective_type: str, params: Dict[str, Any]
    ) -> None:
        """
        Thêm một mục tiêu tối ưu hóa vào mẫu.

        Parameters
        ----------
        structure_id : str
            ID của cấu trúc
        objective_type : str
            Loại mục tiêu (min_dose, max_dose, mean_dose, dvh...)
        params : Dict[str, Any]
            Tham số cho mục tiêu
        """
        objective = {
            "structure_id": structure_id,
            "type": objective_type,
            "parameters": params,
            "weight": params.get("weight", 1.0),
            "priority": params.get("priority", 1),
        }
        self.objectives.append(objective)

    def add_constraint(
        self, structure_id: str, constraint_type: str, params: Dict[str, Any]
    ) -> None:
        """
        Thêm một ràng buộc vào mẫu.

        Parameters
        ----------
        structure_id : str
            ID của cấu trúc
        constraint_type : str
            Loại ràng buộc
        params : Dict[str, Any]
            Tham số cho ràng buộc
        """
        constraint = {
            "structure_id": structure_id,
            "type": constraint_type,
            "parameters": params,
        }
        self.constraints.append(constraint)

    def set_prescription(
        self, target_id: str, dose: float, fractions: int, percentage: float = 95.0
    ) -> None:
        """
        Thiết lập liều kê đơn cho mẫu.

        Parameters
        ----------
        target_id : str
            ID của cấu trúc mục tiêu
        dose : float
            Tổng liều (Gy)
        fractions : int
            Số phân liều
        percentage : float, optional
            Phần trăm thể tích nhận liều, mặc định là 95.0
        """
        self.prescription = {
            "target_id": target_id,
            "total_dose": dose,
            "fractions": fractions,
            "dose_per_fraction": dose / fractions,
            "percentage": percentage,
        }

    def add_structure_mapping(
        self, template_id: str, possible_matches: List[str]
    ) -> None:
        """
        Thêm ánh xạ cấu trúc giữa mẫu và cấu trúc thực tế.

        Parameters
        ----------
        template_id : str
            ID của cấu trúc trong mẫu
        possible_matches : List[str]
            Danh sách các ID cấu trúc có thể khớp
        """
        self.structure_mappings[template_id] = possible_matches

    def save_to_file(self, filepath: str) -> bool:
        """
        Lưu mẫu kế hoạch vào file.

        Parameters
        ----------
        filepath : str
            Đường dẫn file lưu trữ

        Returns
        -------
        bool
            True nếu lưu thành công, False nếu không
        """
        try:
            # Chuyển đổi mẫu thành dict
            template_dict = {
                "name": self.name,
                "site": self.site,
                "description": self.description,
                "metadata": self.metadata,
                "technique": self.technique,
                "beam_parameters": self.beam_parameters,
                "beam_arrangements": self.beam_arrangements,
                "arc_parameters": self.arc_parameters,
                "objectives": self.objectives,
                "constraints": self.constraints,
                "optimization_parameters": self.optimization_parameters,
                "required_structures": self.required_structures,
                "structure_mappings": self.structure_mappings,
                "prescription": self.prescription,
            }

            # Tạo thư mục nếu cần
            ensure_directory_exists(os.path.dirname(filepath))

            # Lưu vào file JSON
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(template_dict, f, indent=2)

            logger.info(f"Đã lưu mẫu kế hoạch '{self.name}' vào file: {filepath}")
            return True
        except Exception as e:
            logger.error(f"Lỗi khi lưu mẫu kế hoạch: {e}")
            return False

    @classmethod
    def load_from_file(cls, filepath: str) -> "PlanTemplate":
        """
        Tải mẫu kế hoạch từ file.

        Parameters
        ----------
        filepath : str
            Đường dẫn file

        Returns
        -------
        PlanTemplate
            Đối tượng mẫu kế hoạch
        """
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Tạo mẫu kế hoạch từ dữ liệu
            template = cls(
                name=data["name"],
                site=data["site"],
                description=data.get("description", ""),
                metadata=data.get("metadata", {}),
            )

            # Đọc các thuộc tính
            template.technique = data.get("technique", "VMAT")
            template.beam_parameters = data.get("beam_parameters", {})
            template.beam_arrangements = data.get("beam_arrangements", [])
            template.arc_parameters = data.get("arc_parameters", {})
            template.objectives = data.get("objectives", [])
            template.constraints = data.get("constraints", [])
            template.optimization_parameters = data.get("optimization_parameters", {})
            template.required_structures = data.get("required_structures", [])
            template.structure_mappings = data.get("structure_mappings", {})
            template.prescription = data.get("prescription", {})

            logger.info(f"Đã tải mẫu kế hoạch từ file: {filepath}")
            return template
        except Exception as e:
            logger.error(f"Lỗi khi tải mẫu kế hoạch từ file: {e}")
            raise


@dataclass
class AutoPlanningConfig:
    """Cấu hình cho quá trình lập kế hoạch tự động."""

    mode: AutoPlanningMode = AutoPlanningMode.TEMPLATE
    planning_target: AutoPlanningTarget = AutoPlanningTarget.FULL_PLAN
    template_path: Optional[str] = None
    protocol_path: Optional[str] = None
    knowledge_db_path: Optional[str] = None

    structure_matching_threshold: float = 0.7  # Ngưỡng khớp cấu trúc
    use_structure_mapping: bool = True  # Sử dụng ánh xạ cấu trúc

    # Tham số lập kế hoạch beam
    beam_energy: str = "6X"  # Năng lượng mặc định
    beam_technique: str = "VMAT"  # Kỹ thuật mặc định
    max_optimization_iterations: int = 100  # Số lần lặp tối đa

    # Cài đặt báo cáo
    report_progress: bool = True  # Báo cáo tiến độ trong quá trình tối ưu
    create_dvh_report: bool = True  # Tạo báo cáo DVH sau khi lập kế hoạch

    # Cài đặt nâng cao
    auto_refine_objectives: bool = True  # Tự động điều chỉnh mục tiêu
    auto_analyze_oars: bool = True  # Tự động phân tích cơ quan nguy cấp
    multi_criteria_optimization: bool = False  # Sử dụng MCO

    # Metadata bổ sung
    metadata: Dict[str, Any] = field(default_factory=dict)


class KnowledgeBasedPlanningModel:
    """Mô hình lập kế hoạch dựa trên cơ sở tri thức."""

    def __init__(self, model_path: Optional[str] = None):
        """
        Khởi tạo mô hình lập kế hoạch dựa trên cơ sở tri thức.

        Parameters
        ----------
        model_path : Optional[str], optional
            Đường dẫn đến model đã lưu, mặc định là None
        """
        self.model_path = model_path
        self.is_trained = False
        self.site = ""
        self.features = []
        self.target_variables = []
        self.model_metadata = {}

        # Các mô hình cho từng mục tiêu tối ưu
        self.models = {}

        # Dữ liệu huấn luyện
        self.training_data = None

        if model_path and os.path.exists(model_path):
            self.load_model(model_path)

    def train(self, training_data: pd.DataFrame, site: str) -> bool:
        """
        Huấn luyện mô hình với dữ liệu từ các kế hoạch đã được chấp nhận.

        Parameters
        ----------
        training_data : pd.DataFrame
            DataFrame chứa dữ liệu huấn luyện
        site : str
            Vị trí điều trị (ví dụ: "prostate", "lung")

        Returns
        -------
        bool
            True nếu huấn luyện thành công, False nếu không
        """
        try:
            self.site = site
            self.training_data = training_data

            # TODO: Triển khai mô hình học máy thực tế ở đây
            # Ví dụ:
            # 1. Xử lý dữ liệu
            # 2. Tạo mô hình (Random Forest, Neural Network, etc.)
            # 3. Huấn luyện mô hình
            # 4. Đánh giá mô hình

            logger.info(f"Đã huấn luyện mô hình KBP cho vị trí: {site}")

            self.is_trained = True
            return True
        except Exception as e:
            logger.error(f"Lỗi khi huấn luyện mô hình KBP: {e}")
            return False

    def predict_objectives(self, patient_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Dự đoán các mục tiêu tối ưu hóa dựa trên dữ liệu bệnh nhân.

        Parameters
        ----------
        patient_data : Dict[str, Any]
            Dữ liệu bệnh nhân

        Returns
        -------
        List[Dict[str, Any]]
            Danh sách các mục tiêu tối ưu hóa được dự đoán
        """
        if not self.is_trained:
            logger.error("Mô hình chưa được huấn luyện.")
            return []

        try:
            # TODO: Triển khai dự đoán thực tế
            # Ví dụ đơn giản: trả về các mục tiêu giả

            organs_at_risk = patient_data.get("organs_at_risk", [])
            targets = patient_data.get("targets", [])

            predicted_objectives = []

            # Mục tiêu cho các cấu trúc mục tiêu
            for target in targets:
                predicted_objectives.append(
                    {
                        "structure_id": target["id"],
                        "type": "min_dose",
                        "parameters": {
                            "dose": 0.95 * target.get("prescription_dose", 60),
                            "weight": 100,
                            "priority": 1,
                        },
                    }
                )

                predicted_objectives.append(
                    {
                        "structure_id": target["id"],
                        "type": "max_dose",
                        "parameters": {
                            "dose": 1.07 * target.get("prescription_dose", 60),
                            "weight": 100,
                            "priority": 1,
                        },
                    }
                )

            # Mục tiêu cho các cơ quan nguy cấp
            for oar in organs_at_risk:
                # Giả định các mục tiêu dựa trên tên cơ quan
                if "spinal" in oar["id"].lower():
                    predicted_objectives.append(
                        {
                            "structure_id": oar["id"],
                            "type": "max_dose",
                            "parameters": {
                                "dose": 45,
                                "weight": 80,
                                "priority": 2,
                            },
                        }
                    )
                elif "heart" in oar["id"].lower():
                    predicted_objectives.append(
                        {
                            "structure_id": oar["id"],
                            "type": "mean_dose",
                            "parameters": {
                                "dose": 26,
                                "weight": 70,
                                "priority": 2,
                            },
                        }
                    )
                elif "lung" in oar["id"].lower():
                    predicted_objectives.append(
                        {
                            "structure_id": oar["id"],
                            "type": "dvh",
                            "parameters": {
                                "volume": 20,
                                "dose": 20,
                                "direction": "less_than",
                                "weight": 60,
                                "priority": 3,
                            },
                        }
                    )

            return predicted_objectives

        except Exception as e:
            logger.error(f"Lỗi khi dự đoán mục tiêu tối ưu hóa: {e}")
            return []

    def predict_beam_arrangement(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Dự đoán cấu hình chùm tia dựa trên dữ liệu bệnh nhân.

        Parameters
        ----------
        patient_data : Dict[str, Any]
            Dữ liệu bệnh nhân

        Returns
        -------
        Dict[str, Any]
            Cấu hình chùm tia được dự đoán
        """
        if not self.is_trained:
            logger.error("Mô hình chưa được huấn luyện.")
            return {}

        try:
            # TODO: Triển khai dự đoán thực tế
            # Ví dụ đơn giản: trả về cấu hình chùm tia cơ bản

            site = patient_data.get("site", "").lower()

            # Cấu hình mặc định
            beam_arrangement = {"technique": "VMAT", "energy": "6X", "arcs": []}

            # Tùy chỉnh theo vị trí điều trị
            if "prostate" in site:
                beam_arrangement["arcs"] = [
                    {
                        "gantry_start": 180,
                        "gantry_stop": 179.9,
                        "collimator": 15,
                        "couch": 0,
                    },
                    {
                        "gantry_start": 179.9,
                        "gantry_stop": 180,
                        "collimator": 345,
                        "couch": 0,
                    },
                ]
            elif "lung" in site:
                beam_arrangement["arcs"] = [
                    {
                        "gantry_start": 180,
                        "gantry_stop": 0,
                        "collimator": 15,
                        "couch": 0,
                    },
                    {
                        "gantry_start": 0,
                        "gantry_stop": 180,
                        "collimator": 345,
                        "couch": 0,
                    },
                ]
            elif "head_neck" in site:
                beam_arrangement["arcs"] = [
                    {
                        "gantry_start": 180,
                        "gantry_stop": 0,
                        "collimator": 15,
                        "couch": 0,
                    },
                    {
                        "gantry_start": 0,
                        "gantry_stop": 180,
                        "collimator": 345,
                        "couch": 0,
                    },
                ]
            else:
                # Default dual arc
                beam_arrangement["arcs"] = [
                    {
                        "gantry_start": 180,
                        "gantry_stop": 179.9,
                        "collimator": 15,
                        "couch": 0,
                    },
                    {
                        "gantry_start": 179.9,
                        "gantry_stop": 180,
                        "collimator": 345,
                        "couch": 0,
                    },
                ]

            return beam_arrangement

        except Exception as e:
            logger.error(f"Lỗi khi dự đoán cấu hình chùm tia: {e}")
            return {}

    def save_model(self, filepath: str) -> bool:
        """
        Lưu mô hình đã huấn luyện.

        Parameters
        ----------
        filepath : str
            Đường dẫn file lưu trữ

        Returns
        -------
        bool
            True nếu lưu thành công, False nếu không
        """
        if not self.is_trained:
            logger.error("Không thể lưu mô hình chưa được huấn luyện.")
            return False

        try:
            # Tạo thư mục nếu cần
            ensure_directory_exists(os.path.dirname(filepath))

            # Dữ liệu cần lưu
            model_data = {
                "site": self.site,
                "features": self.features,
                "target_variables": self.target_variables,
                "metadata": self.model_metadata,
                "is_trained": self.is_trained,
                # TODO: Lưu các tham số mô hình thực tế
            }

            # Lưu vào file
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(model_data, f, indent=2)

            logger.info(f"Đã lưu mô hình KBP vào: {filepath}")
            return True
        except Exception as e:
            logger.error(f"Lỗi khi lưu mô hình: {e}")
            return False

    def load_model(self, filepath: str) -> bool:
        """
        Tải mô hình đã huấn luyện.

        Parameters
        ----------
        filepath : str
            Đường dẫn file

        Returns
        -------
        bool
            True nếu tải thành công, False nếu không
        """
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                model_data = json.load(f)

            # Đọc thông tin mô hình
            self.site = model_data["site"]
            self.features = model_data["features"]
            self.target_variables = model_data["target_variables"]
            self.model_metadata = model_data["metadata"]
            self.is_trained = model_data["is_trained"]

            # TODO: Tải các tham số mô hình thực tế

            logger.info(f"Đã tải mô hình KBP từ: {filepath}")
            return True
        except Exception as e:
            logger.error(f"Lỗi khi tải mô hình: {e}")
            return False


class AutoPlanner:
    """
    Lớp chính cho lập kế hoạch xạ trị tự động.

    Lớp này điều phối quá trình tự động hóa lập kế hoạch xạ trị,
    bao gồm tạo cấu hình chùm tia, thiết lập mục tiêu tối ưu hóa,
    và thực hiện quá trình tối ưu hóa.
    """

    def __init__(self, patient: Patient, config: Optional[AutoPlanningConfig] = None):
        """
        Khởi tạo AutoPlanner với một bệnh nhân và cấu hình tùy chọn.

        Parameters
        ----------
        patient : Patient
            Đối tượng bệnh nhân
        config : Optional[AutoPlanningConfig], optional
            Cấu hình lập kế hoạch tự động, mặc định là None
        """
        self.patient = patient
        self.config = config or AutoPlanningConfig()

        # Các thành phần giúp quá trình lập kế hoạch
        self.structure_matcher = StructureMatcher()
        self.beam_generator = BeamGenerator()
        self.dvh_calculator = DVHCalculator()

        # Mô hình tri thức nếu sử dụng KBP
        self.kbp_model = None
        if (
            self.config.mode == AutoPlanningMode.KNOWLEDGE_BASED
            and self.config.knowledge_db_path
        ):
            self.kbp_model = KnowledgeBasedPlanningModel(self.config.knowledge_db_path)

        # Mẫu kế hoạch nếu sử dụng lập kế hoạch dựa trên mẫu
        self.template = None
        if self.config.mode == AutoPlanningMode.TEMPLATE and self.config.template_path:
            try:
                self.template = PlanTemplate.load_from_file(self.config.template_path)
            except Exception as e:
                logger.error(f"Không thể tải mẫu kế hoạch: {e}")

        # Ánh xạ cấu trúc giữa mẫu và bệnh nhân hiện tại
        self.structure_mapping = {}

        # Lưu trữ kết quả
        self.beam_arrangement = None
        self.objectives = []
        self.constraints = []
        self.optimization_result = None
        self.generated_plan = None

        # Đăng ký callback tiến độ
        self._progress_callback = None
        self._progress = 0.0
        self._status_message = ""

        # Cờ trạng thái
        self.is_initialized = False
        self.is_calculating = False
        self.is_done = False

    def initialize(self) -> bool:
        """
        Khởi tạo quá trình lập kế hoạch tự động.

        Returns
        -------
        bool
            True nếu khởi tạo thành công, False nếu không
        """
        try:
            self._report_progress(
                0.1, "Đang khởi tạo quá trình lập kế hoạch tự động..."
            )

            # Khởi tạo các thành phần
            if self.config.mode == AutoPlanningMode.TEMPLATE:
                if not self.template:
                    logger.error("Không có mẫu kế hoạch cho chế độ TEMPLATE.")
                    return False

                # Ánh xạ cấu trúc
                if self.config.use_structure_mapping:
                    self._report_progress(0.2, "Đang ánh xạ cấu trúc...")
                    self._map_structures()

            # Khởi tạo KBP model nếu cần
            elif self.config.mode == AutoPlanningMode.KNOWLEDGE_BASED:
                if not self.kbp_model or not self.kbp_model.is_trained:
                    logger.error(
                        "Mô hình KBP không khả dụng hoặc chưa được huấn luyện."
                    )
                    return False

            self.is_initialized = True
            self._report_progress(0.3, "Đã khởi tạo thành công!")
            return True

        except Exception as e:
            logger.error(f"Lỗi khi khởi tạo AutoPlanner: {e}")
            return False

    def _map_structures(self) -> None:
        """
        Ánh xạ cấu trúc giữa mẫu và bệnh nhân hiện tại.
        """
        if not self.template or not self.template.structure_mappings:
            logger.warning("Không có thông tin ánh xạ cấu trúc trong mẫu.")
            return

        patient_structures = self.patient.get_structure_set().get_structures()
        patient_structure_ids = [s.id for s in patient_structures]

        for template_id, possible_matches in self.template.structure_mappings.items():
            # Tìm khớp tốt nhất
            best_match = None
            best_score = 0.0

            for patient_id in patient_structure_ids:
                score = self.structure_matcher.calculate_match_score(
                    template_id, patient_id, aliases=possible_matches
                )

                if (
                    score > best_score
                    and score >= self.config.structure_matching_threshold
                ):
                    best_score = score
                    best_match = patient_id

            if best_match:
                self.structure_mapping[template_id] = best_match
                logger.info(
                    f"Ánh xạ cấu trúc: {template_id} -> {best_match} (score: {best_score:.2f})"
                )
            else:
                logger.warning(f"Không tìm thấy khớp cho cấu trúc mẫu: {template_id}")

    def create_beam_arrangement(self) -> Dict[str, Any]:
        """
        Tạo cấu hình chùm tia dựa trên chế độ lập kế hoạch.

        Returns
        -------
        Dict[str, Any]
            Cấu hình chùm tia
        """
        self._report_progress(0.4, "Đang tạo cấu hình chùm tia...")

        if not self.is_initialized:
            logger.error("AutoPlanner chưa được khởi tạo.")
            return {}

        try:
            # Tạo cấu hình chùm tia dựa trên chế độ
            if self.config.mode == AutoPlanningMode.TEMPLATE and self.template:
                # Lấy từ mẫu
                beam_arrangement = self._create_beams_from_template()
            elif (
                self.config.mode == AutoPlanningMode.KNOWLEDGE_BASED and self.kbp_model
            ):
                # Dự đoán từ mô hình KBP
                patient_data = self._extract_patient_data_for_kbp()
                beam_arrangement = self.kbp_model.predict_beam_arrangement(patient_data)
            else:
                # Tạo cấu hình mặc định
                beam_arrangement = self._create_default_beam_arrangement()

            self.beam_arrangement = beam_arrangement
            self._report_progress(0.5, "Đã tạo cấu hình chùm tia thành công!")
            return beam_arrangement

        except Exception as e:
            logger.error(f"Lỗi khi tạo cấu hình chùm tia: {e}")
            return {}

    def _create_beams_from_template(self) -> Dict[str, Any]:
        """
        Tạo cấu hình chùm tia từ mẫu.

        Returns
        -------
        Dict[str, Any]
            Cấu hình chùm tia
        """
        beam_arrangement = {"technique": self.template.technique, "beams": []}

        if self.template.technique.upper() == "VMAT":
            beam_arrangement["arcs"] = []

            for arc_params in self.template.beam_arrangements:
                if "gantry_start" in arc_params and "gantry_stop" in arc_params:
                    arc = {
                        "gantry_start": arc_params["gantry_start"],
                        "gantry_stop": arc_params["gantry_stop"],
                        "collimator": arc_params.get("collimator", 0),
                        "couch": arc_params.get("couch", 0),
                        "energy": arc_params.get("energy", self.config.beam_energy),
                    }
                    beam_arrangement["arcs"].append(arc)

        elif self.template.technique.upper() == "IMRT":
            for beam_params in self.template.beam_arrangements:
                if "gantry_angle" in beam_params:
                    beam = {
                        "gantry_angle": beam_params["gantry_angle"],
                        "collimator_angle": beam_params.get("collimator_angle", 0),
                        "couch_angle": beam_params.get("couch_angle", 0),
                        "energy": beam_params.get("energy", self.config.beam_energy),
                    }
                    beam_arrangement["beams"].append(beam)

        return beam_arrangement

    def _create_default_beam_arrangement(self) -> Dict[str, Any]:
        """
        Tạo cấu hình chùm tia mặc định.

        Returns
        -------
        Dict[str, Any]
            Cấu hình chùm tia mặc định
        """
        # Mặc định sử dụng VMAT dual arc
        beam_arrangement = {
            "technique": "VMAT",
            "arcs": [
                {
                    "gantry_start": 180,
                    "gantry_stop": 179.9,
                    "collimator": 15,
                    "couch": 0,
                    "energy": self.config.beam_energy,
                },
                {
                    "gantry_start": 179.9,
                    "gantry_stop": 180,
                    "collimator": 345,
                    "couch": 0,
                    "energy": self.config.beam_energy,
                },
            ],
        }

        return beam_arrangement

    def _extract_patient_data_for_kbp(self) -> Dict[str, Any]:
        """
        Trích xuất thông tin bệnh nhân cho mô hình KBP.

        Returns
        -------
        Dict[str, Any]
            Dữ liệu bệnh nhân
        """
        patient_data = {
            "patient_id": self.patient.id,
            "targets": [],
            "organs_at_risk": [],
            "site": "",  # Xác định từ tên bệnh nhân hoặc thông tin khác
        }

        structure_set = self.patient.get_structure_set()
        if structure_set:
            structures = structure_set.get_structures()

            for struct in structures:
                if struct.type.upper() in ["PTV", "CTV", "GTV"]:
                    target_info = {
                        "id": struct.id,
                        "type": struct.type,
                        "volume": struct.get_volume(),
                        "prescription_dose": 0,  # Cập nhật từ thông tin kê đơn
                    }
                    patient_data["targets"].append(target_info)
                else:
                    oar_info = {
                        "id": struct.id,
                        "type": struct.type,
                        "volume": struct.get_volume(),
                    }
                    patient_data["organs_at_risk"].append(oar_info)

        return patient_data

    def create_objectives(self) -> List[Dict[str, Any]]:
        """
        Tạo các mục tiêu tối ưu hóa dựa trên chế độ lập kế hoạch.

        Returns
        -------
        List[Dict[str, Any]]
            Danh sách các mục tiêu tối ưu hóa
        """
        self._report_progress(0.6, "Đang tạo mục tiêu tối ưu hóa...")

        if not self.is_initialized:
            logger.error("AutoPlanner chưa được khởi tạo.")
            return []

        try:
            # Tạo mục tiêu dựa trên chế độ
            if self.config.mode == AutoPlanningMode.TEMPLATE and self.template:
                # Lấy từ mẫu
                objectives = self._create_objectives_from_template()
                constraints = self._create_constraints_from_template()
            elif (
                self.config.mode == AutoPlanningMode.KNOWLEDGE_BASED and self.kbp_model
            ):
                # Dự đoán từ mô hình KBP
                patient_data = self._extract_patient_data_for_kbp()
                objectives = self.kbp_model.predict_objectives(patient_data)
                constraints = []  # TODO: Thêm dự đoán ràng buộc
            else:
                # Tạo mục tiêu mặc định
                objectives, constraints = self._create_default_objectives()

            self.objectives = objectives
            self.constraints = constraints

            self._report_progress(0.7, "Đã tạo mục tiêu tối ưu hóa thành công!")
            return objectives

        except Exception as e:
            logger.error(f"Lỗi khi tạo mục tiêu tối ưu hóa: {e}")
            return []

    def _create_objectives_from_template(self) -> List[Dict[str, Any]]:
        """
        Tạo các mục tiêu tối ưu hóa từ mẫu.

        Returns
        -------
        List[Dict[str, Any]]
            Danh sách các mục tiêu tối ưu hóa
        """
        objectives = []

        for obj in self.template.objectives:
            # Ánh xạ ID cấu trúc nếu cần
            template_structure_id = obj["structure_id"]
            patient_structure_id = self._map_structure_id(template_structure_id)

            if not patient_structure_id:
                logger.warning(
                    f"Bỏ qua mục tiêu cho cấu trúc không tìm thấy: {template_structure_id}"
                )
                continue

            # Tạo mục tiêu mới với ID đã ánh xạ
            new_objective = copy.deepcopy(obj)
            new_objective["structure_id"] = patient_structure_id

            objectives.append(new_objective)

        return objectives

    def _create_constraints_from_template(self) -> List[Dict[str, Any]]:
        """
        Tạo các ràng buộc từ mẫu.

        Returns
        -------
        List[Dict[str, Any]]
            Danh sách các ràng buộc
        """
        constraints = []

        for constr in self.template.constraints:
            # Ánh xạ ID cấu trúc nếu cần
            template_structure_id = constr["structure_id"]
            patient_structure_id = self._map_structure_id(template_structure_id)

            if not patient_structure_id:
                logger.warning(
                    f"Bỏ qua ràng buộc cho cấu trúc không tìm thấy: {template_structure_id}"
                )
                continue

            # Tạo ràng buộc mới với ID đã ánh xạ
            new_constraint = copy.deepcopy(constr)
            new_constraint["structure_id"] = patient_structure_id

            constraints.append(new_constraint)

        return constraints

    def _map_structure_id(self, template_id: str) -> Optional[str]:
        """
        Ánh xạ ID cấu trúc từ mẫu sang ID cấu trúc của bệnh nhân.

        Parameters
        ----------
        template_id : str
            ID cấu trúc trong mẫu

        Returns
        -------
        Optional[str]
            ID cấu trúc của bệnh nhân hoặc None nếu không tìm thấy
        """
        if template_id in self.structure_mapping:
            return self.structure_mapping[template_id]

        # Nếu không có ánh xạ, kiểm tra xem có cấu trúc trùng tên không
        structure_set = self.patient.get_structure_set()
        if structure_set:
            structures = structure_set.get_structures()
            for struct in structures:
                if struct.id == template_id:
                    return struct.id

        return None

    def _create_default_objectives(
        self,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Tạo các mục tiêu và ràng buộc tối ưu hóa mặc định.

        Returns
        -------
        Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]
            Mục tiêu và ràng buộc tối ưu hóa
        """
        objectives = []
        constraints = []

        structure_set = self.patient.get_structure_set()
        if not structure_set:
            return objectives, constraints

        structures = structure_set.get_structures()

        # Tìm các cấu trúc mục tiêu (PTV, CTV, GTV)
        targets = [s for s in structures if s.type.upper() in ["PTV", "CTV", "GTV"]]

        # Tìm các cơ quan nguy cấp
        oars = [
            s
            for s in structures
            if s.type.upper() not in ["PTV", "CTV", "GTV", "EXTERNAL", "BODY"]
        ]

        # Tạo mục tiêu cho các cấu trúc mục tiêu
        for target in targets:
            # Mục tiêu độ đồng nhất (tối thiểu 95% liều kê đơn)
            objectives.append(
                {
                    "structure_id": target.id,
                    "type": "min_dose",
                    "parameters": {
                        "dose": 50.0,  # Liều mặc định
                        "weight": 100,
                        "priority": 1,
                    },
                }
            )

            # Mục tiêu liều tối đa (tối đa 107% liều kê đơn)
            objectives.append(
                {
                    "structure_id": target.id,
                    "type": "max_dose",
                    "parameters": {
                        "dose": 53.5,  # Liều mặc định
                        "weight": 100,
                        "priority": 1,
                    },
                }
            )

        # Tạo mục tiêu cho các cơ quan nguy cấp
        for oar in oars:
            oar_id = oar.id.lower()

            # Các mục tiêu mặc định dựa trên tên cơ quan
            if "spinal" in oar_id or "cord" in oar_id:
                objectives.append(
                    {
                        "structure_id": oar.id,
                        "type": "max_dose",
                        "parameters": {
                            "dose": 45.0,
                            "weight": 80,
                            "priority": 2,
                        },
                    }
                )
            elif "heart" in oar_id:
                objectives.append(
                    {
                        "structure_id": oar.id,
                        "type": "mean_dose",
                        "parameters": {
                            "dose": 26.0,
                            "weight": 70,
                            "priority": 3,
                        },
                    }
                )
            elif "lung" in oar_id:
                objectives.append(
                    {
                        "structure_id": oar.id,
                        "type": "dvh",
                        "parameters": {
                            "volume": 20.0,
                            "dose": 20.0,
                            "direction": "less_than",
                            "weight": 60,
                            "priority": 3,
                        },
                    }
                )
            elif "liver" in oar_id:
                objectives.append(
                    {
                        "structure_id": oar.id,
                        "type": "mean_dose",
                        "parameters": {
                            "dose": 30.0,
                            "weight": 60,
                            "priority": 3,
                        },
                    }
                )
            elif "kidney" in oar_id:
                objectives.append(
                    {
                        "structure_id": oar.id,
                        "type": "mean_dose",
                        "parameters": {
                            "dose": 18.0,
                            "weight": 40,
                            "priority": 4,
                        },
                    }
                )
            elif "bladder" in oar_id:
                objectives.append(
                    {
                        "structure_id": oar.id,
                        "type": "dvh",
                        "parameters": {
                            "volume": 50.0,
                            "dose": 65.0,
                            "direction": "less_than",
                            "weight": 40,
                            "priority": 3,
                        },
                    }
                )
            elif "rectum" in oar_id:
                objectives.append(
                    {
                        "structure_id": oar.id,
                        "type": "dvh",
                        "parameters": {
                            "volume": 50.0,
                            "dose": 60.0,
                            "direction": "less_than",
                            "weight": 50,
                            "priority": 3,
                        },
                    }
                )

        return objectives, constraints

    def optimize_plan(self) -> bool:
        """
        Thực hiện quá trình tối ưu hóa kế hoạch xạ trị.

        Returns
        -------
        bool
            True nếu tối ưu hóa thành công, False nếu không
        """
        self._report_progress(0.8, "Đang tối ưu hóa kế hoạch xạ trị...")

        if not self.is_initialized:
            logger.error("AutoPlanner chưa được khởi tạo.")
            return False

        if not self.beam_arrangement:
            logger.error("Chưa có cấu hình chùm tia.")
            return False

        if not self.objectives:
            logger.error("Chưa có mục tiêu tối ưu hóa.")
            return False

        try:
            # Thiết lập tối ưu hóa
            optimizer = Optimizer()

            # Thiết lập chùm tia
            if self.beam_arrangement["technique"].upper() == "VMAT":
                for arc in self.beam_arrangement.get("arcs", []):
                    optimizer.add_arc(
                        gantry_start=arc["gantry_start"],
                        gantry_stop=arc["gantry_stop"],
                        collimator_angle=arc.get("collimator", 0),
                        couch_angle=arc.get("couch", 0),
                        energy=arc.get("energy", self.config.beam_energy),
                    )
            else:  # IMRT
                for beam in self.beam_arrangement.get("beams", []):
                    optimizer.add_beam(
                        gantry_angle=beam["gantry_angle"],
                        collimator_angle=beam.get("collimator_angle", 0),
                        couch_angle=beam.get("couch_angle", 0),
                        energy=beam.get("energy", self.config.beam_energy),
                    )

            # Đăng ký callback tiến độ
            def progress_callback(progress: float, message: str) -> None:
                # Map [0, 1] to [0.8, 0.9]
                mapped_progress = 0.8 + progress * 0.1
                self._report_progress(mapped_progress, message)

            optimizer.register_progress_callback(progress_callback)

            # Thiết lập mục tiêu tối ưu hóa
            for obj in self.objectives:
                optimizer.add_objective_from_dict(obj)

            # Thiết lập ràng buộc
            for constraint in self.constraints:
                optimizer.add_constraint_from_dict(constraint)

            # Thiết lập tham số tối ưu hóa
            if self.config.mode == AutoPlanningMode.TEMPLATE and self.template:
                optimizer_params = self.template.optimization_parameters
            else:
                optimizer_params = {
                    "max_iterations": self.config.max_optimization_iterations,
                    "convergence_tolerance": 1e-4,
                    "use_multi_resolution": True,
                    "resolution_levels": [8, 4, 2, 1],  # mm
                }

            # Thực hiện tối ưu hóa
            solver = OptimizationSolver(
                patient=self.patient, optimizer=optimizer, **optimizer_params
            )

            result = solver.optimize()
            self.optimization_result = result

            # Tự động điều chỉnh mục tiêu nếu cần
            if self.config.auto_refine_objectives and not result.converged:
                self._report_progress(0.9, "Đang tinh chỉnh mục tiêu tối ưu hóa...")
                self._refine_objectives()
                # Tối ưu hóa lại
                result = solver.optimize()
                self.optimization_result = result

            # Tạo kế hoạch
            self.generated_plan = solver.create_plan()

            self._report_progress(1.0, "Đã hoàn thành tối ưu hóa kế hoạch!")
            self.is_done = True
            return result.converged

        except Exception as e:
            logger.error(f"Lỗi khi tối ưu hóa kế hoạch: {e}")
            return False

    def _refine_objectives(self) -> None:
        """
        Tinh chỉnh mục tiêu tối ưu hóa dựa trên kết quả tối ưu hóa ban đầu.
        """
        if not self.optimization_result:
            return

        # Phân tích kết quả và điều chỉnh trọng số
        for obj in self.objectives:
            structure_id = obj["structure_id"]
            obj_type = obj["type"]

            # Kiểm tra độ hội tụ riêng cho từng mục tiêu
            obj_convergence = self.optimization_result.get_objective_convergence(
                structure_id, obj_type
            )

            if obj_convergence and obj_convergence < 0.5:  # Chưa hội tụ tốt
                # Tăng trọng số
                current_weight = obj["parameters"].get("weight", 1.0)
                obj["parameters"]["weight"] = min(current_weight * 1.5, 100.0)
                logger.info(
                    f"Tăng trọng số cho {obj_type} của {structure_id}: {current_weight} -> {obj['parameters']['weight']}"
                )

            # TODO: Thêm các heuristics khác cho việc tinh chỉnh mục tiêu

    def create_plan(self) -> Any:
        """
        Tạo kế hoạch xạ trị đầy đủ.

        Returns
        -------
        Any
            Kế hoạch xạ trị đã tạo
        """
        if not self.is_initialized:
            if not self.initialize():
                return None

        if not self.beam_arrangement:
            self.create_beam_arrangement()

        if not self.objectives:
            self.create_objectives()

        if not self.optimization_result:
            self.optimize_plan()

        return self.generated_plan

    def evaluate_plan(self) -> Dict[str, Any]:
        """
        Đánh giá kế hoạch xạ trị đã tạo.

        Returns
        -------
        Dict[str, Any]
            Kết quả đánh giá kế hoạch
        """
        if not self.generated_plan:
            logger.error("Chưa có kế hoạch để đánh giá.")
            return {}

        try:
            # Tính toán DVH
            dvh_results = self.dvh_calculator.calculate_dvh(
                patient=self.patient, plan=self.generated_plan
            )

            # Tính toán các chỉ số đánh giá kế hoạch
            metrics = calculate_plan_metrics(
                patient=self.patient, plan=self.generated_plan, dvhs=dvh_results
            )

            # Tạo báo cáo DVH nếu cần
            if self.config.create_dvh_report:
                self._create_dvh_report(dvh_results, metrics)

            return {
                "dvh_results": dvh_results,
                "metrics": metrics,
                "optimization_result": self.optimization_result,
            }

        except Exception as e:
            logger.error(f"Lỗi khi đánh giá kế hoạch: {e}")
            return {}

    def _create_dvh_report(self, dvh_results: Any, metrics: Dict[str, Any]) -> None:
        """
        Tạo báo cáo DVH cho kế hoạch đã tạo.

        Parameters
        ----------
        dvh_results : Any
            Kết quả tính toán DVH
        metrics : Dict[str, Any]
            Các chỉ số đánh giá kế hoạch
        """
        # TODO: Triển khai tạo báo cáo DVH
        pass

    def register_progress_callback(
        self, callback: Callable[[float, str], None]
    ) -> None:
        """
        Đăng ký callback cho cập nhật tiến độ.

        Parameters
        ----------
        callback : Callable[[float, str], None]
            Hàm callback nhận tham số tiến độ và thông báo
        """
        self._progress_callback = callback

    def _report_progress(self, progress: float, message: str) -> None:
        """
        Báo cáo tiến độ thông qua callback.

        Parameters
        ----------
        progress : float
            Tiến độ (0.0 - 1.0)
        message : str
            Thông báo tiến độ
        """
        self._progress = progress
        self._status_message = message

        if self.config.report_progress:
            logger.info(f"Tiến độ: {progress:.1%} - {message}")

        if self._progress_callback:
            self._progress_callback(progress, message)

    def get_progress(self) -> Tuple[float, str]:
        """
        Lấy tiến độ hiện tại.

        Returns
        -------
        Tuple[float, str]
            Tiến độ và thông báo
        """
        return self._progress, self._status_message


def create_clinical_templates() -> List[PlanTemplate]:
    """
    Tạo các mẫu kế hoạch xạ trị lâm sàng phổ biến.

    Returns
    -------
    List[PlanTemplate]
        Danh sách các mẫu kế hoạch
    """
    templates = []

    # Mẫu kế hoạch cho tuyến tiền liệt tuyến
    prostate_template = PlanTemplate(
        name="Prostate VMAT",
        site="prostate",
        description="Mẫu kế hoạch VMAT cho tuyến tiền liệt tuyến",
    )

    # Thiết lập kỹ thuật và chùm tia
    prostate_template.add_beam_arrangement(
        "VMAT",
        {
            "gantry_start": 180,
            "gantry_stop": 179.9,
            "collimator": 15,
            "couch": 0,
            "energy": "6X",
        },
    )
    prostate_template.add_beam_arrangement(
        "VMAT",
        {
            "gantry_start": 179.9,
            "gantry_stop": 180,
            "collimator": 345,
            "couch": 0,
            "energy": "6X",
        },
    )

    # Thêm mục tiêu tối ưu hóa
    prostate_template.add_objective(
        "PTV",
        "min_dose",
        {
            "dose": 74.1,  # 95% của 78 Gy
            "weight": 100,
            "priority": 1,
        },
    )
    prostate_template.add_objective(
        "PTV",
        "max_dose",
        {
            "dose": 83.5,  # 107% của 78 Gy
            "weight": 100,
            "priority": 1,
        },
    )
    prostate_template.add_objective(
        "Rectum",
        "dvh",
        {
            "volume": 50,
            "dose": 50,
            "direction": "less_than",
            "weight": 80,
            "priority": 2,
        },
    )
    prostate_template.add_objective(
        "Bladder",
        "dvh",
        {
            "volume": 50,
            "dose": 65,
            "direction": "less_than",
            "weight": 70,
            "priority": 2,
        },
    )
    prostate_template.add_objective(
        "Femoral_Heads", "max_dose", {"dose": 50, "weight": 50, "priority": 3}
    )

    # Thiết lập kê đơn
    prostate_template.set_prescription("PTV", 78.0, 39, 95.0)

    # Thiết lập ánh xạ cấu trúc
    prostate_template.add_structure_mapping("PTV", ["PTV", "PTV_78", "PTV_Prostate"])
    prostate_template.add_structure_mapping("Rectum", ["Rectum", "RECTUM"])
    prostate_template.add_structure_mapping("Bladder", ["Bladder", "BLADDER"])
    prostate_template.add_structure_mapping(
        "Femoral_Heads", ["Femur_L", "Femur_R", "Femoral_Heads"]
    )

    templates.append(prostate_template)

    # Mẫu kế hoạch cho phổi
    lung_template = PlanTemplate(
        name="Lung SBRT VMAT",
        site="lung",
        description="Mẫu kế hoạch VMAT cho xạ phẫu thân phổi (SBRT)",
    )

    # Thiết lập kỹ thuật và chùm tia
    lung_template.add_beam_arrangement(
        "VMAT",
        {
            "gantry_start": 180,
            "gantry_stop": 0,
            "collimator": 15,
            "couch": 0,
            "energy": "10X-FFF",
        },
    )
    lung_template.add_beam_arrangement(
        "VMAT",
        {
            "gantry_start": 0,
            "gantry_stop": 180,
            "collimator": 345,
            "couch": 0,
            "energy": "10X-FFF",
        },
    )

    # Thêm mục tiêu tối ưu hóa
    lung_template.add_objective(
        "PTV",
        "min_dose",
        {
            "dose": 47.5,  # 95% của 50 Gy
            "weight": 100,
            "priority": 1,
        },
    )
    lung_template.add_objective(
        "PTV",
        "max_dose",
        {
            "dose": 65.0,  # 130% của 50 Gy
            "weight": 100,
            "priority": 1,
        },
    )
    lung_template.add_objective(
        "Lung-PTV",
        "dvh",
        {
            "volume": 20,
            "dose": 20,
            "direction": "less_than",
            "weight": 80,
            "priority": 2,
        },
    )
    lung_template.add_objective(
        "SpinalCord", "max_dose", {"dose": 30, "weight": 90, "priority": 1}
    )
    lung_template.add_objective(
        "Heart", "mean_dose", {"dose": 26, "weight": 70, "priority": 2}
    )

    # Thiết lập kê đơn
    lung_template.set_prescription("PTV", 50.0, 5, 95.0)

    # Thiết lập ánh xạ cấu trúc
    lung_template.add_structure_mapping("PTV", ["PTV", "PTV_50", "PTV_Lung"])
    lung_template.add_structure_mapping("Lung-PTV", ["Lung-PTV", "Lung_Minus_PTV"])
    lung_template.add_structure_mapping(
        "SpinalCord", ["SpinalCord", "SpinalCord_PRV", "Cord"]
    )
    lung_template.add_structure_mapping("Heart", ["Heart", "HEART"])

    templates.append(lung_template)

    # Mẫu kế hoạch cho đầu cổ
    hn_template = PlanTemplate(
        name="Head_Neck VMAT",
        site="head_neck",
        description="Mẫu kế hoạch VMAT cho ung thư đầu cổ",
    )

    # Thiết lập kỹ thuật và chùm tia
    hn_template.add_beam_arrangement(
        "VMAT",
        {
            "gantry_start": 180,
            "gantry_stop": 0,
            "collimator": 15,
            "couch": 0,
            "energy": "6X",
        },
    )
    hn_template.add_beam_arrangement(
        "VMAT",
        {
            "gantry_start": 0,
            "gantry_stop": 180,
            "collimator": 345,
            "couch": 0,
            "energy": "6X",
        },
    )

    # Thêm mục tiêu tối ưu hóa
    hn_template.add_objective(
        "PTV_70",
        "min_dose",
        {
            "dose": 66.5,  # 95% của 70 Gy
            "weight": 100,
            "priority": 1,
        },
    )
    hn_template.add_objective(
        "PTV_70",
        "max_dose",
        {
            "dose": 75.0,  # 107% của 70 Gy
            "weight": 100,
            "priority": 1,
        },
    )
    hn_template.add_objective(
        "PTV_56",
        "min_dose",
        {
            "dose": 53.2,  # 95% của 56 Gy
            "weight": 90,
            "priority": 2,
        },
    )
    hn_template.add_objective(
        "SpinalCord", "max_dose", {"dose": 45, "weight": 90, "priority": 1}
    )
    hn_template.add_objective(
        "Parotid_L", "mean_dose", {"dose": 26, "weight": 60, "priority": 3}
    )
    hn_template.add_objective(
        "Parotid_R", "mean_dose", {"dose": 26, "weight": 60, "priority": 3}
    )

    # Thiết lập kê đơn
    hn_template.set_prescription("PTV_70", 70.0, 35, 95.0)

    # Thiết lập ánh xạ cấu trúc
    hn_template.add_structure_mapping("PTV_70", ["PTV_70", "PTV70", "PTV_High"])
    hn_template.add_structure_mapping("PTV_56", ["PTV_56", "PTV56", "PTV_Low"])
    hn_template.add_structure_mapping("SpinalCord", ["SpinalCord", "Cord", "SC"])
    hn_template.add_structure_mapping(
        "Parotid_L", ["Parotid_L", "Left_Parotid", "Parotid_Left"]
    )
    hn_template.add_structure_mapping(
        "Parotid_R", ["Parotid_R", "Right_Parotid", "Parotid_Right"]
    )

    templates.append(hn_template)

    return templates


def save_templates_to_directory(templates: List[PlanTemplate], directory: str) -> None:
    """
    Lưu danh sách mẫu kế hoạch vào thư mục.

    Parameters
    ----------
    templates : List[PlanTemplate]
        Danh sách các mẫu kế hoạch
    directory : str
        Thư mục đích
    """
    ensure_directory_exists(directory)

    for template in templates:
        filename = f"{template.site}_{template.name.replace(' ', '_')}.json"
        filepath = os.path.join(directory, filename)
        template.save_to_file(filepath)

    logger.info(f"Đã lưu {len(templates)} mẫu kế hoạch vào thư mục: {directory}")


def load_templates_from_directory(directory: str) -> List[PlanTemplate]:
    """
    Tải danh sách mẫu kế hoạch từ thư mục.

    Parameters
    ----------
    directory : str
        Thư mục nguồn

    Returns
    -------
    List[PlanTemplate]
        Danh sách các mẫu kế hoạch
    """
    templates = []

    if not os.path.exists(directory):
        logger.warning(f"Thư mục mẫu kế hoạch không tồn tại: {directory}")
        return templates

    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            filepath = os.path.join(directory, filename)
            try:
                template = PlanTemplate.load_from_file(filepath)
                templates.append(template)
            except Exception as e:
                logger.error(f"Lỗi khi tải mẫu kế hoạch từ file {filepath}: {e}")

    logger.info(f"Đã tải {len(templates)} mẫu kế hoạch từ thư mục: {directory}")
    return templates
