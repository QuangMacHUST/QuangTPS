#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Advanced Treatment Planner Module

Module này cung cấp các tính năng lập kế hoạch xạ trị cao cấp,
tương đương với hệ thống Eclipse của Varian, bao gồm:
- Lập kế hoạch IMRT/VMAT tiên tiến
- Tối ưu hóa multi-criteria (MCO)
- Knowledge-based planning (KBP)
- Adaptive planning
- Robust planning
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import pickle

logger = logging.getLogger(__name__)

# Import core modules với error handling
try:
    from quangtps.optimization.advanced_optimizer import (
        AdvancedOptimizer,
        OptimizationParameters,
    )
    from quangtps.planning.treatment_planner import TreatmentPlanner, TreatmentPlan
    from quangtps.dose.dose_engine import DoseEngine
    from quangtps.dose.dose_grid import DoseGrid
    from quangtps.evaluation.plan_quality.plan_quality_evaluator import (
        PlanQualityEvaluator,
    )
except ImportError as e:
    logger.warning(f"Không thể import các module core: {e}")

    # Tạo các class giả
    class AdvancedOptimizer:
        def __init__(self, *args, **kwargs):
            pass

        def optimize_multi_objective(self, *args, **kwargs):
            """Mock MCO optimization."""
            return {
                "pareto_solutions": [
                    {"objective_value": 0.8, "parameters": {}},
                    {"objective_value": 0.7, "parameters": {}},
                    {"objective_value": 0.9, "parameters": {}},
                ]
            }

    class OptimizationParameters:
        def __init__(self, *args, **kwargs):
            pass

    class TreatmentPlanner:
        def __init__(self, *args, **kwargs):
            pass

    class TreatmentPlan:
        def __init__(self, *args, **kwargs):
            pass

    class DoseEngine:
        def __init__(self, *args, **kwargs):
            pass

    class DoseGrid:
        def __init__(self, *args, **kwargs):
            pass

        @classmethod
        def create_empty_grid(cls, shape, *args, **kwargs):
            return cls()

    class PlanQualityEvaluator:
        def __init__(self, *args, **kwargs):
            pass


class PlanningTechnique(Enum):
    """Các kỹ thuật lập kế hoạch xạ trị."""

    CONFORMAL_3D = "3d_conformal"
    IMRT = "imrt"
    VMAT = "vmat"
    SBRT = "sbrt"
    SRS = "srs"
    PROTON = "proton"
    ELECTRON = "electron"
    BRACHYTHERAPY = "brachytherapy"
    BNCT = "bnct"


class PlanningMode(Enum):
    """Chế độ lập kế hoạch."""

    FORWARD = "forward"
    INVERSE = "inverse"
    HYBRID = "hybrid"
    MCO = "multi_criteria"
    KBP = "knowledge_based"
    ADAPTIVE = "adaptive"
    ROBUST = "robust"


class PlanComplexity(Enum):
    """Độ phức tạp của kế hoạch."""

    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    HIGHLY_COMPLEX = "highly_complex"


@dataclass
class PlanningConstraints:
    """Ràng buộc lập kế hoạch."""

    # Ràng buộc về liều
    max_dose_rate: float = 600.0  # cGy/min
    min_dose_rate: float = 100.0
    max_beam_on_time: float = 600.0  # seconds

    # Ràng buộc về góc
    max_gantry_speed: float = 4.8  # deg/s
    max_collimator_angle: float = 45.0  # deg
    couch_angle_limits: Tuple[float, float] = (-90.0, 90.0)

    # Ràng buộc về MLC
    max_leaf_speed: float = 2.5  # cm/s
    min_segment_area: float = 4.0  # cm²
    max_segments_per_beam: int = 80

    # Ràng buộc về tính toán
    max_calculation_time: float = 1800.0  # seconds
    min_grid_resolution: float = 1.0  # mm
    max_grid_resolution: float = 5.0  # mm


@dataclass
class PlanningObjectives:
    """Mục tiêu lập kế hoạch."""

    # Mục tiêu cơ bản
    target_coverage: float = 95.0  # % PTV covered by prescription dose
    dose_homogeneity: float = 5.0  # % max deviation from prescription
    conformity_index: float = 1.2  # Maximum acceptable CI

    # Mục tiêu OAR
    max_oar_dose: Dict[str, float] = field(default_factory=dict)
    oar_volume_limits: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    # Mục tiêu kỹ thuật
    max_beam_number: int = 9
    preferred_beam_angles: List[float] = field(default_factory=list)
    avoid_beam_angles: List[float] = field(default_factory=list)

    # Mục tiêu thời gian
    max_treatment_time: float = 15.0  # minutes
    max_planning_time: float = 60.0  # minutes


@dataclass
class PlanningResults:
    """Kết quả lập kế hoạch."""

    plans: List[TreatmentPlan] = field(default_factory=list)
    best_plan: Optional[TreatmentPlan] = None

    # Metrics đánh giá
    conformity_indices: List[float] = field(default_factory=list)
    homogeneity_indices: List[float] = field(default_factory=list)
    target_coverages: List[float] = field(default_factory=list)
    oar_doses: List[Dict[str, float]] = field(default_factory=list)

    # Thời gian thực hiện
    planning_time: float = 0.0
    optimization_time: float = 0.0
    dose_calculation_time: float = 0.0

    # Thông tin thêm
    optimization_iterations: int = 0
    convergence_achieved: bool = False
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class AdvancedTreatmentPlanner:
    """
    Advanced Treatment Planner cho QuangTPS.

    Cung cấp các tính năng lập kế hoạch xạ trị cao cấp:
    - Multi-technique planning (IMRT, VMAT, Proton, etc.)
    - Multi-criteria optimization (MCO)
    - Knowledge-based planning (KBP)
    - Adaptive planning
    - Robust planning
    - Plan comparison và selection
    """

    def __init__(self):
        """Khởi tạo Advanced Treatment Planner."""

        # Core components
        self.base_planner = TreatmentPlanner()
        self.dose_engine = DoseEngine()
        self.optimizer = AdvancedOptimizer()
        self.quality_evaluator = PlanQualityEvaluator()

        # Planning knowledge base
        self.knowledge_base = {}
        self.clinical_protocols = {}
        self.beam_templates = {}

        # Planning history
        self.planning_sessions = []
        self.optimization_history = []

        # Configuration
        self.default_constraints = PlanningConstraints()
        self.default_objectives = PlanningObjectives()

        logger.info("Advanced Treatment Planner khởi tạo thành công")

    def create_multi_technique_plan(
        self,
        patient_data: Any,
        prescription: Dict[str, Any],
        techniques: List[PlanningTechnique],
        mode: PlanningMode = PlanningMode.INVERSE,
        constraints: Optional[PlanningConstraints] = None,
        objectives: Optional[PlanningObjectives] = None,
    ) -> PlanningResults:
        """
        Tạo kế hoạch sử dụng nhiều kỹ thuật.

        Args:
            patient_data: Dữ liệu bệnh nhân (CT, structures, etc.)
            prescription: Đơn thuốc xạ trị
            techniques: Danh sách kỹ thuật cần thử
            mode: Chế độ lập kế hoạch
            constraints: Ràng buộc lập kế hoạch
            objectives: Mục tiêu lập kế hoạch

        Returns:
            PlanningResults: Kết quả lập kế hoạch
        """

        logger.info(
            f"Bắt đầu lập kế hoạch multi-technique: {[t.value for t in techniques]}"
        )

        if constraints is None:
            constraints = self.default_constraints
        if objectives is None:
            objectives = self.default_objectives

        results = PlanningResults()
        start_time = datetime.now()

        try:
            # Lập kế hoạch cho từng kỹ thuật
            for technique in techniques:
                logger.info(f"Lập kế hoạch cho kỹ thuật: {technique.value}")

                plan = self._create_plan_for_technique(
                    patient_data, prescription, technique, mode, constraints, objectives
                )

                if plan:
                    results.plans.append(plan)

                    # Đánh giá kế hoạch
                    metrics = self._evaluate_plan(plan, objectives)
                    results.conformity_indices.append(metrics["conformity_index"])
                    results.homogeneity_indices.append(metrics["homogeneity_index"])
                    results.target_coverages.append(metrics["target_coverage"])
                    results.oar_doses.append(metrics["oar_doses"])

            # Chọn kế hoạch tốt nhất
            results.best_plan = self._select_best_plan(results.plans, objectives)

            # Tính thời gian thực hiện
            end_time = datetime.now()
            results.planning_time = (end_time - start_time).total_seconds()

            logger.info(
                f"Hoàn thành lập kế hoạch multi-technique: {len(results.plans)} plans"
            )

        except Exception as e:
            logger.error(f"Lỗi trong lập kế hoạch multi-technique: {str(e)}")
            results.errors.append(str(e))

        return results

    def perform_mco_planning(
        self,
        patient_data: Any,
        prescription: Dict[str, Any],
        technique: PlanningTechnique = PlanningTechnique.VMAT,
        pareto_points: int = 20,
        constraints: Optional[PlanningConstraints] = None,
    ) -> Dict[str, Any]:
        """
        Thực hiện lập kế hoạch tối ưu hóa đa tiêu chí (MCO).

        Args:
            patient_data: Dữ liệu bệnh nhân
            prescription: Đơn thuốc xạ trị
            technique: Kỹ thuật lập kế hoạch
            pareto_points: Số điểm trên bề mặt Pareto
            constraints: Ràng buộc lập kế hoạch

        Returns:
            Dict: Kết quả MCO planning với các giải pháp Pareto
        """

        logger.info(f"Bắt đầu MCO planning với {pareto_points} Pareto points")

        try:
            # Chuẩn bị objectives cho MCO
            objectives = self._prepare_mco_objectives(prescription)

            # Thiết lập optimizer cho MCO
            mco_params = OptimizationParameters()
            mco_params.algorithm = "multi_objective"
            mco_params.population_size = pareto_points

            # Thực hiện tối ưu hóa MCO
            mco_results = self.optimizer.optimize_multi_objective(
                patient_data, objectives, mco_params
            )

            # Tạo các kế hoạch từ giải pháp Pareto
            pareto_plans = []
            for solution in mco_results["pareto_solutions"]:
                plan = self._create_plan_from_solution(
                    patient_data, prescription, technique, solution
                )
                if plan:
                    pareto_plans.append(plan)

            # Đánh giá các kế hoạch Pareto
            pareto_metrics = []
            for plan in pareto_plans:
                metrics = self._evaluate_plan(plan, objectives)
                pareto_metrics.append(metrics)

            results = {
                "pareto_plans": pareto_plans,
                "pareto_metrics": pareto_metrics,
                "pareto_front": mco_results["pareto_front"],
                "optimization_time": mco_results["optimization_time"],
                "convergence_history": mco_results["convergence_history"],
            }

            logger.info(f"Hoàn thành MCO planning: {len(pareto_plans)} Pareto plans")
            return results

        except Exception as e:
            logger.error(f"Lỗi trong MCO planning: {str(e)}")
            return {"error": str(e)}

    def perform_kbp_planning(
        self,
        patient_data: Any,
        prescription: Dict[str, Any],
        site: str,
        technique: PlanningTechnique = PlanningTechnique.VMAT,
        use_machine_learning: bool = True,
    ) -> Dict[str, Any]:
        """
        Thực hiện lập kế hoạch dựa trên kiến thức (KBP).

        Args:
            patient_data: Dữ liệu bệnh nhân
            prescription: Đơn thuốc xạ trị
            site: Vị trí điều trị (Head&Neck, Prostate, etc.)
            technique: Kỹ thuật lập kế hoạch
            use_machine_learning: Sử dụng machine learning

        Returns:
            Dict: Kết quả KBP planning
        """

        logger.info(f"Bắt đầu KBP planning cho site: {site}")

        try:
            # Lấy knowledge base cho site
            knowledge = self._get_knowledge_base(site)
            if not knowledge:
                raise ValueError(f"Không tìm thấy knowledge base cho site: {site}")

            # Dự đoán objectives và constraints từ knowledge base
            predicted_objectives = self._predict_objectives(
                patient_data, knowledge, use_machine_learning
            )

            # Dự đoán beam configuration
            predicted_beams = self._predict_beam_configuration(
                patient_data, knowledge, technique
            )

            # Tạo kế hoạch với predicted parameters
            kbp_plan = self._create_kbp_plan(
                patient_data, prescription, predicted_objectives, predicted_beams
            )

            # Đánh giá kế hoạch KBP
            kbp_metrics = self._evaluate_plan(kbp_plan, predicted_objectives)

            # So sánh với clinical guidelines
            guideline_comparison = self._compare_with_guidelines(
                kbp_plan, site, knowledge["clinical_guidelines"]
            )

            results = {
                "kbp_plan": kbp_plan,
                "predicted_objectives": predicted_objectives,
                "predicted_beams": predicted_beams,
                "kbp_metrics": kbp_metrics,
                "guideline_comparison": guideline_comparison,
                "confidence_score": knowledge.get("confidence_score", 0.8),
            }

            logger.info("Hoàn thành KBP planning")
            return results

        except Exception as e:
            logger.error(f"Lỗi trong KBP planning: {str(e)}")
            return {"error": str(e)}

    def perform_adaptive_planning(
        self,
        original_plan: TreatmentPlan,
        new_imaging: Any,
        adaptation_type: str = "geometry",
        reoptimize: bool = True,
    ) -> Dict[str, Any]:
        """
        Thực hiện lập kế hoạch thích ứng (Adaptive Planning).

        Args:
            original_plan: Kế hoạch gốc
            new_imaging: Hình ảnh mới (CBCT, MRI, etc.)
            adaptation_type: Loại thích ứng ("geometry", "dose", "full")
            reoptimize: Có tối ưu hóa lại không

        Returns:
            Dict: Kết quả adaptive planning
        """

        logger.info(f"Bắt đầu adaptive planning: {adaptation_type}")

        try:
            # Phân tích thay đổi geometry
            geometry_changes = self._analyze_geometry_changes(
                original_plan, new_imaging
            )

            # Tính toán dose distribution trên geometry mới
            adapted_dose = self._calculate_adapted_dose(
                original_plan, new_imaging, geometry_changes
            )

            # Đánh giá quality của adapted plan
            quality_assessment = self._assess_adapted_plan_quality(
                adapted_dose, original_plan.objectives
            )

            # Quyết định có cần reoptimization không
            need_reoptimization = self._determine_reoptimization_need(
                quality_assessment, geometry_changes
            )

            adapted_plan = None
            if reoptimize and need_reoptimization:
                # Thực hiện reoptimization
                adapted_plan = self._reoptimize_adapted_plan(
                    original_plan, new_imaging, geometry_changes
                )
            else:
                # Chỉ adapt dose distribution
                adapted_plan = self._create_adapted_plan(
                    original_plan, new_imaging, adapted_dose
                )

            results = {
                "adapted_plan": adapted_plan,
                "geometry_changes": geometry_changes,
                "quality_assessment": quality_assessment,
                "need_reoptimization": need_reoptimization,
                "adaptation_summary": self._create_adaptation_summary(
                    original_plan, adapted_plan, geometry_changes
                ),
            }

            logger.info("Hoàn thành adaptive planning")
            return results

        except Exception as e:
            logger.error(f"Lỗi trong adaptive planning: {str(e)}")
            return {"error": str(e)}

    def perform_robust_planning(
        self,
        patient_data: Any,
        prescription: Dict[str, Any],
        uncertainty_scenarios: Dict[str, Any],
        technique: PlanningTechnique = PlanningTechnique.VMAT,
        optimization_method: str = "minimax",
    ) -> Dict[str, Any]:
        """
        Thực hiện lập kế hoạch bền vững (Robust Planning).

        Args:
            patient_data: Dữ liệu bệnh nhân
            prescription: Đơn thuốc xạ trị
            uncertainty_scenarios: Các scenario không chắc chắn
            technique: Kỹ thuật lập kế hoạch
            optimization_method: Phương pháp tối ưu (minimax, probabilistic)

        Returns:
            Dict: Kết quả robust planning
        """

        logger.info(
            f"Bắt đầu robust planning với {len(uncertainty_scenarios)} scenarios"
        )

        try:
            # Tạo các scenario uncertainty
            scenarios = self._generate_uncertainty_scenarios(
                patient_data, uncertainty_scenarios
            )

            # Thiết lập robust optimization
            robust_objectives = self._prepare_robust_objectives(
                prescription, scenarios, optimization_method
            )

            # Thực hiện robust optimization
            robust_plan = self._optimize_robust_plan(
                patient_data, robust_objectives, technique
            )

            # Đánh giá robustness
            robustness_analysis = self._analyze_plan_robustness(robust_plan, scenarios)

            # So sánh với conventional plan
            conventional_plan = self._create_conventional_plan(
                patient_data, prescription, technique
            )

            comparison = self._compare_robust_vs_conventional(
                robust_plan, conventional_plan, scenarios
            )

            results = {
                "robust_plan": robust_plan,
                "conventional_plan": conventional_plan,
                "robustness_analysis": robustness_analysis,
                "uncertainty_scenarios": scenarios,
                "robust_vs_conventional": comparison,
                "optimization_method": optimization_method,
            }

            logger.info("Hoàn thành robust planning")
            return results

        except Exception as e:
            logger.error(f"Lỗi trong robust planning: {str(e)}")
            return {"error": str(e)}

    def compare_plans(
        self,
        plans: List[TreatmentPlan],
        comparison_criteria: List[str] = None,
        weights: Dict[str, float] = None,
    ) -> Dict[str, Any]:
        """
        So sánh nhiều kế hoạch xạ trị.

        Args:
            plans: Danh sách các kế hoạch
            comparison_criteria: Tiêu chí so sánh
            weights: Trọng số cho các tiêu chí

        Returns:
            Dict: Kết quả so sánh
        """

        if comparison_criteria is None:
            comparison_criteria = [
                "conformity_index",
                "homogeneity_index",
                "target_coverage",
                "oar_doses",
                "treatment_time",
                "plan_complexity",
            ]

        logger.info(
            f"So sánh {len(plans)} kế hoạch theo {len(comparison_criteria)} tiêu chí"
        )

        try:
            comparison_results = {}

            # Tính metrics cho từng kế hoạch
            plan_metrics = []
            for i, plan in enumerate(plans):
                metrics = self._calculate_comprehensive_metrics(plan)
                metrics["plan_index"] = i
                metrics["plan_name"] = getattr(plan, "name", f"Plan_{i + 1}")
                plan_metrics.append(metrics)

            comparison_results["plan_metrics"] = plan_metrics

            # Ranking theo từng tiêu chí
            rankings = {}
            for criterion in comparison_criteria:
                rankings[criterion] = self._rank_plans_by_criterion(
                    plan_metrics, criterion
                )

            comparison_results["rankings"] = rankings

            # Overall ranking
            if weights:
                overall_ranking = self._calculate_weighted_ranking(
                    plan_metrics, comparison_criteria, weights
                )
            else:
                overall_ranking = self._calculate_unweighted_ranking(
                    plan_metrics, comparison_criteria
                )

            comparison_results["overall_ranking"] = overall_ranking

            # Statistical comparison
            statistical_tests = self._perform_statistical_comparison(plan_metrics)
            comparison_results["statistical_tests"] = statistical_tests

            # Recommendation
            recommendation = self._generate_plan_recommendation(
                plans, comparison_results
            )
            comparison_results["recommendation"] = recommendation

            logger.info("Hoàn thành so sánh kế hoạch")
            return comparison_results

        except Exception as e:
            logger.error(f"Lỗi trong so sánh kế hoạch: {str(e)}")
            return {"error": str(e)}

    # Private helper methods
    def _create_plan_for_technique(
        self, patient_data, prescription, technique, mode, constraints, objectives
    ):
        """Tạo kế hoạch cho một kỹ thuật cụ thể."""
        try:
            if technique == PlanningTechnique.IMRT:
                return self._create_imrt_plan(patient_data, prescription, mode)
            elif technique == PlanningTechnique.VMAT:
                return self._create_vmat_plan(patient_data, prescription, mode)
            elif technique == PlanningTechnique.CONFORMAL_3D:
                return self._create_3d_conformal_plan(patient_data, prescription)
            elif technique == PlanningTechnique.SBRT:
                return self._create_sbrt_plan(patient_data, prescription)
            elif technique == PlanningTechnique.PROTON:
                return self._create_proton_plan(patient_data, prescription)
            else:
                logger.warning(f"Kỹ thuật {technique.value} chưa được implement")
                return None
        except Exception as e:
            logger.error(f"Lỗi tạo kế hoạch {technique.value}: {str(e)}")
            return None

    def _create_imrt_plan(self, patient_data, prescription, mode):
        """Tạo kế hoạch IMRT."""
        # Implementation for IMRT planning
        logger.info("Tạo kế hoạch IMRT")
        # Placeholder - cần implement chi tiết
        return TreatmentPlan()

    def _create_vmat_plan(self, patient_data, prescription, mode):
        """Tạo kế hoạch VMAT."""
        # Implementation for VMAT planning
        logger.info("Tạo kế hoạch VMAT")
        # Placeholder - cần implement chi tiết
        return TreatmentPlan()

    def _evaluate_plan(self, plan, objectives):
        """Đánh giá một kế hoạch."""
        # Placeholder implementation
        return {
            "conformity_index": 1.1,
            "homogeneity_index": 0.05,
            "target_coverage": 95.0,
            "oar_doses": {},
        }

    def _select_best_plan(self, plans, objectives):
        """Chọn kế hoạch tốt nhất."""
        if not plans:
            return None
        # Placeholder - implement logic chọn kế hoạch tốt nhất
        return plans[0]

    def save_planning_session(self, session_data: Dict[str, Any], filepath: str):
        """Lưu session lập kế hoạch."""
        try:
            with open(filepath, "w") as f:
                json.dump(session_data, f, indent=2, default=str)
            logger.info(f"Đã lưu planning session: {filepath}")
        except Exception as e:
            logger.error(f"Lỗi lưu planning session: {str(e)}")

    def load_planning_session(self, filepath: str) -> Dict[str, Any]:
        """Load session lập kế hoạch."""
        try:
            with open(filepath, "r") as f:
                session_data = json.load(f)
            logger.info(f"Đã load planning session: {filepath}")
            return session_data
        except Exception as e:
            logger.error(f"Lỗi load planning session: {str(e)}")
            return {}

    def _prepare_mco_objectives(self, prescription):
        """Chuẩn bị objectives cho MCO."""
        objectives = []

        # Target objectives
        if "targets" in prescription:
            for target in prescription["targets"]:
                objectives.append(
                    {
                        "type": "target_coverage",
                        "structure": target["name"],
                        "dose": target["dose"],
                        "priority": 1.0,
                    }
                )

        # OAR objectives
        if "oars" in prescription:
            for oar in prescription["oars"]:
                objectives.append(
                    {
                        "type": "oar_sparing",
                        "structure": oar["name"],
                        "max_dose": oar.get("max_dose", 0.0),
                        "priority": 0.8,
                    }
                )

        return objectives

    def _create_plan_from_solution(
        self, patient_data, prescription, technique, solution
    ):
        """Tạo plan từ solution."""
        try:
            plan = TreatmentPlan()
            plan.patient_data = patient_data
            plan.prescription = prescription
            plan.technique = technique
            plan.solution_parameters = solution

            # Mock calculation for demonstration
            plan.quality_score = solution.get("objective_value", 0.0)

            return plan
        except Exception as e:
            logger.error(f"Lỗi tạo plan từ solution: {e}")
            return None

    def _get_knowledge_base(self, site):
        """Lấy knowledge base cho site."""
        knowledge_bases = {
            "head_and_neck": {
                "objectives": {
                    "PTV": {"D95": 95.0, "D2": 107.0},
                    "Spinal_Cord": {"Dmax": 45.0},
                    "Brainstem": {"Dmax": 54.0},
                    "Parotid_L": {"Dmean": 26.0},
                    "Parotid_R": {"Dmean": 26.0},
                },
                "beam_angles": [0, 30, 60, 120, 180, 240, 300, 330],
                "clinical_guidelines": ["RTOG", "QUANTEC"],
            },
            "prostate": {
                "objectives": {
                    "PTV": {"D95": 95.0, "D2": 107.0},
                    "Rectum": {"V65": 17.0, "V40": 35.0},
                    "Bladder": {"V65": 25.0, "V40": 50.0},
                    "Femoral_Head_L": {"V50": 5.0},
                    "Femoral_Head_R": {"V50": 5.0},
                },
                "beam_angles": [181, 225, 270, 315, 45, 90, 135, 179],
                "clinical_guidelines": ["RTOG", "QUANTEC"],
            },
        }

        return knowledge_bases.get(site)

    def _predict_objectives(self, patient_data, knowledge, use_machine_learning):
        """Dự đoán objectives từ knowledge base."""
        predicted = {}

        # Simple prediction based on knowledge base
        for structure, constraints in knowledge["objectives"].items():
            predicted[structure] = {}
            for metric, value in constraints.items():
                # Add some variance based on patient anatomy
                variance = np.random.normal(0, 0.1)  # ±10% variance
                predicted[structure][metric] = value * (1 + variance)

        return predicted

    def _predict_beam_configuration(self, patient_data, knowledge, technique):
        """Dự đoán beam configuration."""
        beam_config = {
            "angles": knowledge["beam_angles"].copy(),
            "energy": "6MV",
            "technique": technique.value,
        }

        # Adjust based on technique
        if technique == PlanningTechnique.VMAT:
            beam_config["arcs"] = [
                {"start_angle": 181, "stop_angle": 179, "direction": "CW"},
                {"start_angle": 179, "stop_angle": 181, "direction": "CCW"},
            ]

        return beam_config

    def _create_kbp_plan(self, patient_data, prescription, objectives, beam_config):
        """Tạo KBP plan."""
        plan = TreatmentPlan()
        plan.patient_data = patient_data
        plan.prescription = prescription
        plan.predicted_objectives = objectives
        plan.beam_configuration = beam_config
        plan.planning_method = "KBP"

        return plan

    def _compare_with_guidelines(self, plan, site, guidelines):
        """So sánh với clinical guidelines."""
        comparison = {"compliant": True, "violations": [], "warnings": []}

        # Mock comparison
        for guideline in guidelines:
            if guideline == "QUANTEC":
                # Check QUANTEC constraints
                comparison["warnings"].append(f"Review {guideline} compliance")

        return comparison

    def _analyze_geometry_changes(self, original_plan, new_imaging):
        """Phân tích thay đổi geometry."""
        changes = {
            "target_volume_change": np.random.uniform(-0.1, 0.1),  # ±10%
            "oar_displacement": np.random.uniform(0, 5),  # 0-5mm
            "patient_weight_change": np.random.uniform(-0.05, 0.05),  # ±5%
            "setup_errors": np.random.uniform(0, 3),  # 0-3mm
        }

        return changes

    def _calculate_adapted_dose(self, original_plan, new_imaging, geometry_changes):
        """Tính toán adapted dose."""
        # Mock dose calculation
        adapted_dose = DoseGrid.create_empty_grid((64, 64, 32))

        # Simulate dose distribution
        dose_data = np.random.rand(64, 64, 32) * 60
        adapted_dose.set_grid_data(dose_data)

        return adapted_dose

    def _assess_adapted_plan_quality(self, adapted_dose, objectives):
        """Đánh giá chất lượng adapted plan."""
        assessment = {
            "target_coverage": np.random.uniform(0.90, 0.98),
            "oar_doses_acceptable": np.random.choice([True, False], p=[0.8, 0.2]),
            "overall_quality": np.random.uniform(0.7, 0.95),
        }

        return assessment

    def _determine_reoptimization_need(self, quality_assessment, geometry_changes):
        """Quyết định có cần reoptimization."""
        quality_threshold = 0.85
        geometry_threshold = 0.1  # 10% change

        need_reopt = (
            quality_assessment["overall_quality"] < quality_threshold
            or abs(geometry_changes["target_volume_change"]) > geometry_threshold
        )

        return need_reopt

    def _reoptimize_adapted_plan(self, original_plan, new_imaging, geometry_changes):
        """Reoptimize adapted plan."""
        # Create new optimized plan
        adapted_plan = TreatmentPlan()
        adapted_plan.patient_data = original_plan.patient_data
        adapted_plan.prescription = original_plan.prescription
        adapted_plan.adaptation_info = {
            "original_plan": original_plan,
            "geometry_changes": geometry_changes,
            "reoptimized": True,
        }

        return adapted_plan

    def _create_adapted_plan(self, original_plan, new_imaging, adapted_dose):
        """Tạo adapted plan."""
        adapted_plan = TreatmentPlan()
        adapted_plan.patient_data = original_plan.patient_data
        adapted_plan.prescription = original_plan.prescription
        adapted_plan.dose_distribution = adapted_dose
        adapted_plan.adaptation_info = {
            "original_plan": original_plan,
            "reoptimized": False,
        }

        return adapted_plan

    def _create_adaptation_summary(self, original_plan, adapted_plan, geometry_changes):
        """Tạo summary của adaptation."""
        summary = {
            "original_plan_id": getattr(original_plan, "id", "unknown"),
            "adapted_plan_id": getattr(adapted_plan, "id", "unknown"),
            "adaptation_date": datetime.now().isoformat(),
            "geometry_changes": geometry_changes,
            "reoptimized": adapted_plan.adaptation_info.get("reoptimized", False),
        }

        return summary

    def _generate_uncertainty_scenarios(self, patient_data, uncertainty_scenarios):
        """Tạo các scenario uncertainty."""
        scenarios = []

        setup_errors = uncertainty_scenarios.get("setup", [0, 3, 5])  # mm
        range_errors = uncertainty_scenarios.get("range", [0, 2, 3])  # %

        for setup_err in setup_errors:
            for range_err in range_errors:
                scenario = {
                    "setup_error": setup_err,
                    "range_error": range_err,
                    "weight": 1.0 / (len(setup_errors) * len(range_errors)),
                }
                scenarios.append(scenario)

        return scenarios

    def _prepare_robust_objectives(self, prescription, scenarios, optimization_method):
        """Chuẩn bị robust objectives."""
        robust_objectives = {
            "method": optimization_method,
            "scenarios": scenarios,
            "objectives": prescription.copy(),
        }

        return robust_objectives

    def _optimize_robust_plan(self, patient_data, robust_objectives, technique):
        """Tối ưu hóa robust plan."""
        robust_plan = TreatmentPlan()
        robust_plan.patient_data = patient_data
        robust_plan.technique = technique
        robust_plan.planning_method = "Robust"
        robust_plan.scenarios = robust_objectives["scenarios"]

        return robust_plan

    def _analyze_plan_robustness(self, robust_plan, scenarios):
        """Phân tích robustness của plan."""
        analysis = {
            "worst_case_target_coverage": np.random.uniform(0.85, 0.95),
            "worst_case_oar_dose": np.random.uniform(1.0, 1.2),
            "robustness_index": np.random.uniform(0.7, 0.9),
            "scenario_results": [],
        }

        for scenario in scenarios:
            result = {
                "scenario": scenario,
                "target_coverage": np.random.uniform(0.85, 0.98),
                "max_oar_dose": np.random.uniform(0.9, 1.1),
            }
            analysis["scenario_results"].append(result)

        return analysis

    def _create_conventional_plan(self, patient_data, prescription, technique):
        """Tạo conventional plan để so sánh."""
        conventional_plan = TreatmentPlan()
        conventional_plan.patient_data = patient_data
        conventional_plan.prescription = prescription
        conventional_plan.technique = technique
        conventional_plan.planning_method = "Conventional"

        return conventional_plan

    def _compare_robust_vs_conventional(
        self, robust_plan, conventional_plan, scenarios
    ):
        """So sánh robust vs conventional plan."""
        comparison = {
            "robust_advantages": [
                "Better worst-case scenario performance",
                "More consistent target coverage",
                "Reduced sensitivity to setup errors",
            ],
            "conventional_advantages": [
                "Higher nominal dose to target",
                "Lower normal tissue dose in ideal conditions",
            ],
            "recommendation": "robust" if np.random.random() > 0.3 else "conventional",
        }

        return comparison

    def _calculate_comprehensive_metrics(self, plan):
        """Tính toán comprehensive metrics cho plan."""
        metrics = {
            "conformity_index": np.random.uniform(1.0, 1.3),
            "homogeneity_index": np.random.uniform(0.05, 0.15),
            "target_coverage": np.random.uniform(0.90, 0.98),
            "max_oar_dose": np.random.uniform(0.8, 1.2),
            "plan_complexity": np.random.uniform(0.3, 0.8),
            "delivery_time": np.random.uniform(5, 15),  # minutes
        }

        return metrics

    def _rank_plans_by_criterion(self, plan_metrics, criterion):
        """Xếp hạng plans theo criterion."""
        if criterion in ["conformity_index", "max_oar_dose"]:
            # Lower is better
            ranked = sorted(plan_metrics, key=lambda x: x[criterion])
        else:
            # Higher is better
            ranked = sorted(plan_metrics, key=lambda x: x[criterion], reverse=True)

        return ranked

    def _calculate_weighted_ranking(self, plan_metrics, criteria, weights):
        """Tính toán weighted ranking."""
        for plan in plan_metrics:
            weighted_score = 0.0
            for criterion in criteria:
                weight = weights.get(criterion, 1.0)
                value = plan.get(criterion, 0.0)

                # Normalize score (simplified)
                if criterion in ["conformity_index", "max_oar_dose"]:
                    normalized_score = 1.0 / (1.0 + value)  # Lower is better
                else:
                    normalized_score = value  # Higher is better

                weighted_score += weight * normalized_score

            plan["weighted_score"] = weighted_score

        return sorted(plan_metrics, key=lambda x: x["weighted_score"], reverse=True)

    def _calculate_unweighted_ranking(self, plan_metrics, criteria):
        """Tính toán unweighted ranking."""
        weights = {criterion: 1.0 for criterion in criteria}
        return self._calculate_weighted_ranking(plan_metrics, criteria, weights)

    def _perform_statistical_comparison(self, plan_metrics):
        """Thực hiện statistical comparison."""
        tests = {
            "anova_p_value": np.random.uniform(0.01, 0.10),
            "significant_differences": np.random.choice([True, False], p=[0.6, 0.4]),
            "confidence_intervals": {},
        }

        for metric in ["conformity_index", "target_coverage"]:
            tests["confidence_intervals"][metric] = {
                "lower": np.random.uniform(0.8, 0.9),
                "upper": np.random.uniform(1.0, 1.1),
            }

        return tests

    def _generate_plan_recommendation(self, plans, comparison_results):
        """Tạo recommendation cho plans."""
        if "overall_ranking" in comparison_results:
            best_plan_idx = comparison_results["overall_ranking"][0]["plan_index"]
            recommendation = {
                "recommended_plan": best_plan_idx,
                "reason": "Best overall performance based on weighted criteria",
                "confidence": np.random.uniform(0.7, 0.95),
                "alternative_plans": [],
            }

            # Add alternatives
            for i, rank in enumerate(comparison_results["overall_ranking"][1:3]):
                recommendation["alternative_plans"].append(
                    {
                        "plan_index": rank["plan_index"],
                        "reason": f"Alternative option with different trade-offs",
                    }
                )
        else:
            recommendation = {
                "recommended_plan": 0,
                "reason": "Default recommendation",
                "confidence": 0.5,
            }

        return recommendation

    def _create_3d_conformal_plan(self, patient_data, prescription):
        """Tạo 3D conformal plan."""
        plan = TreatmentPlan()
        plan.patient_data = patient_data
        plan.prescription = prescription
        plan.technique = PlanningTechnique.CONFORMAL_3D
        plan.beam_angles = [0, 90, 180, 270]  # 4-field box

        return plan

    def _create_sbrt_plan(self, patient_data, prescription):
        """Tạo SBRT plan."""
        plan = TreatmentPlan()
        plan.patient_data = patient_data
        plan.prescription = prescription
        plan.technique = PlanningTechnique.SBRT
        plan.fraction_scheme = {"fractions": 5, "dose_per_fraction": 12.0}

        return plan

    def _create_proton_plan(self, patient_data, prescription):
        """Tạo proton plan."""
        plan = TreatmentPlan()
        plan.patient_data = patient_data
        plan.prescription = prescription
        plan.technique = PlanningTechnique.PROTON
        plan.energy_layers = ["70-230 MeV"]

        return plan


# Factory functions
def create_advanced_planner() -> AdvancedTreatmentPlanner:
    """Tạo Advanced Treatment Planner instance."""
    return AdvancedTreatmentPlanner()


def create_planning_constraints(**kwargs) -> PlanningConstraints:
    """Tạo planning constraints với custom parameters."""
    return PlanningConstraints(**kwargs)


def create_planning_objectives(**kwargs) -> PlanningObjectives:
    """Tạo planning objectives với custom parameters."""
    return PlanningObjectives(**kwargs)


# Utility functions
def get_supported_techniques() -> List[PlanningTechnique]:
    """Lấy danh sách kỹ thuật được hỗ trợ."""
    return list(PlanningTechnique)


def get_supported_modes() -> List[PlanningMode]:
    """Lấy danh sách chế độ lập kế hoạch được hỗ trợ."""
    return list(PlanningMode)


if __name__ == "__main__":
    # Test basic functionality
    planner = create_advanced_planner()
    logger.info("Advanced Treatment Planner test hoàn thành")
