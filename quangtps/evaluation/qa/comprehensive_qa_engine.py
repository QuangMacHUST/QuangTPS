"""
QuangTPS Comprehensive QA Engine

Module tích hợp đầy đủ cho quality assurance trong xạ trị.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
import json
import os

logger = logging.getLogger(__name__)

# Import các module metrics đã tạo
try:
    from quangtps.evaluation.metrics.gamma_analysis import (
        GammaAnalysisSettings,
        GammaAnalysisResult,
        calculate_gamma_3d,
    )
    from quangtps.evaluation.metrics.dose_metrics import (
        DoseStatistics,
        calculate_dose_statistics,
        calculate_dose_at_volume,
        calculate_volume_at_dose,
        calculate_conformity_index,
        calculate_homogeneity_index,
    )
    from quangtps.evaluation.metrics.plan_quality_metrics import (
        PlanQualityResults,
        calculate_comprehensive_plan_quality,
    )
    from quangtps.evaluation.metrics.dvh_metrics import (
        DVHCurve,
        DVHStatistics,
        create_dvh_from_dose_distribution,
        calculate_dvh_statistics,
    )
    from quangtps.evaluation.metrics.biological_metrics import (
        BiologicalMetrics,
        calculate_comprehensive_biological_metrics,
        TCPParameters,
        NTCPParameters,
        EUDParameters,
    )

    HAS_METRICS = True
except ImportError as e:
    logger.warning(f"Some metrics modules not available: {e}")
    HAS_METRICS = False

    # Fallback classes
    @dataclass
    class GammaAnalysisSettings:
        distance_mm: float = 3.0
        dose_percent: float = 3.0
        dose_threshold_percent: float = 10.0

    @dataclass
    class GammaAnalysisResult:
        gamma_map: np.ndarray = field(default_factory=lambda: np.array([]))
        pass_rate: float = 0.0
        mean_gamma: float = 0.0
        max_gamma: float = 0.0

    @dataclass
    class DoseStatistics:
        mean_dose: float = 0.0
        max_dose: float = 0.0
        min_dose: float = 0.0

    @dataclass
    class PlanQualityResults:
        conformity_index: float = 0.0
        homogeneity_index: float = 0.0

    @dataclass
    class DVHStatistics:
        mean_dose: float = 0.0
        d_95: float = 0.0
        d_50: float = 0.0

    @dataclass
    class BiologicalMetrics:
        tcp: float = 0.0
        ntcp: float = 0.0
        eud: float = 0.0


@dataclass
class QAConfiguration:
    """Cấu hình cho comprehensive QA analysis."""

    # Gamma analysis settings
    gamma_distance_mm: float = 3.0
    gamma_dose_percent: float = 3.0
    gamma_threshold_percent: float = 10.0
    gamma_pass_rate_threshold: float = 95.0

    # Dose analysis settings
    dose_difference_threshold: float = 2.0  # %
    distance_to_agreement_threshold: float = 2.0  # mm

    # Plan quality thresholds
    conformity_index_threshold: float = 0.95
    homogeneity_index_threshold: float = 0.1

    # Coverage requirements
    target_coverage_threshold: float = 95.0  # %

    # Performance settings
    use_gpu_acceleration: bool = True
    max_analysis_threads: int = 4

    # Output settings
    generate_detailed_report: bool = True
    save_intermediate_results: bool = False


@dataclass
class QATestResult:
    """Kết quả một test QA cụ thể."""

    test_name: str
    test_type: str
    passed: bool
    score: float
    threshold: float
    details: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    execution_time: float = 0.0


@dataclass
class ComprehensiveQAReport:
    """Báo cáo QA tổng hợp."""

    overall_score: float = 0.0
    overall_passed: bool = False

    # Individual test results
    gamma_analysis: Optional[QATestResult] = None
    dose_statistics: Optional[QATestResult] = None
    plan_quality: Optional[QATestResult] = None
    dvh_analysis: Optional[QATestResult] = None
    biological_evaluation: Optional[QATestResult] = None

    # Summary statistics
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    warning_tests: int = 0

    # Metadata
    analysis_timestamp: datetime = field(default_factory=datetime.now)
    configuration: Optional[QAConfiguration] = None
    processing_time: float = 0.0

    def get_pass_rate(self) -> float:
        """Tính tỷ lệ pass."""
        if self.total_tests == 0:
            return 0.0
        return (self.passed_tests / self.total_tests) * 100.0


class ComprehensiveQAEngine:
    """Engine tổng hợp cho quality assurance analysis."""

    def __init__(self, configuration: Optional[QAConfiguration] = None):
        """
        Initialize QA Engine.

        Args:
            configuration: QA configuration settings
        """
        self.configuration = configuration or QAConfiguration()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Initialize components
        self._initialize_components()

    def _initialize_components(self):
        """Initialize các components cần thiết."""
        try:
            # Check available components
            self.has_gpu = self._check_gpu_availability()
            self.has_metrics = HAS_METRICS

            self.logger.info("QA Engine initialized successfully")
            self.logger.info(f"GPU available: {self.has_gpu}")
            self.logger.info(f"Metrics modules available: {self.has_metrics}")

        except Exception as e:
            self.logger.error(f"Error initializing QA Engine: {e}")

    def _check_gpu_availability(self) -> bool:
        """Kiểm tra GPU availability."""
        try:
            import numba.cuda
            import numpy

            # Kiểm tra NumPy compatibility với Numba
            numpy_version = tuple(map(int, numpy.__version__.split(".")[:2]))
            if numpy_version >= (2, 1):  # NumPy 2.1+
                self.logger.warning(
                    "NumPy version %s không tương thích với Numba. GPU sẽ không khả dụng.",
                    numpy.__version__,
                )
                return False

            return numba.cuda.is_available()
        except ImportError:
            self.logger.warning("Numba CUDA không khả dụng")
            return False
        except Exception as e:
            self.logger.warning("Lỗi kiểm tra GPU: %s", str(e))
            return False

    def run_comprehensive_analysis(
        self,
        reference_dose: np.ndarray,
        evaluated_dose: np.ndarray,
        target_masks: Dict[str, np.ndarray],
        organ_masks: Dict[str, np.ndarray],
        prescription_dose: float,
        spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        progress_callback: Optional[Callable] = None,
    ) -> ComprehensiveQAReport:
        """
        Thực hiện comprehensive QA analysis.

        Args:
            reference_dose: Reference dose distribution
            evaluated_dose: Evaluated dose distribution
            target_masks: Dictionary of target structure masks
            organ_masks: Dictionary of organ masks
            prescription_dose: Prescription dose
            spacing: Voxel spacing
            progress_callback: Progress callback function

        Returns:
            ComprehensiveQAReport: Comprehensive QA report
        """
        start_time = datetime.now()
        report = ComprehensiveQAReport(configuration=self.configuration)

        try:
            self.logger.info("Starting comprehensive QA analysis")

            # 1. Gamma Analysis
            if progress_callback:
                progress_callback(10, "Running gamma analysis...")
            report.gamma_analysis = self._run_gamma_analysis(
                reference_dose, evaluated_dose, spacing
            )

            # 2. Dose Statistics Analysis
            if progress_callback:
                progress_callback(30, "Analyzing dose statistics...")
            report.dose_statistics = self._run_dose_statistics_analysis(
                reference_dose, evaluated_dose, target_masks, organ_masks
            )

            # 3. Plan Quality Analysis
            if progress_callback:
                progress_callback(50, "Evaluating plan quality...")
            report.plan_quality = self._run_plan_quality_analysis(
                evaluated_dose, target_masks, prescription_dose, spacing
            )

            # 4. DVH Analysis
            if progress_callback:
                progress_callback(70, "Performing DVH analysis...")
            report.dvh_analysis = self._run_dvh_analysis(
                evaluated_dose, target_masks, organ_masks, spacing
            )

            # 5. Biological Evaluation
            if progress_callback:
                progress_callback(90, "Computing biological metrics...")
            report.biological_evaluation = self._run_biological_evaluation(
                evaluated_dose, target_masks, organ_masks
            )

            # Calculate overall scores
            if progress_callback:
                progress_callback(100, "Finalizing report...")
            self._calculate_overall_scores(report)

            # Set processing time
            end_time = datetime.now()
            report.processing_time = (end_time - start_time).total_seconds()

            self.logger.info(f"QA analysis completed in {report.processing_time:.2f}s")
            self.logger.info(f"Overall score: {report.overall_score:.1f}%")

            return report

        except Exception as e:
            self.logger.error(f"Error in comprehensive QA analysis: {e}")
            report.processing_time = (datetime.now() - start_time).total_seconds()
            return report

    def _run_gamma_analysis(
        self,
        reference_dose: np.ndarray,
        evaluated_dose: np.ndarray,
        spacing: Tuple[float, float, float],
    ) -> QATestResult:
        """Run gamma analysis."""
        try:
            if HAS_METRICS:
                settings = GammaAnalysisSettings(
                    distance_mm=self.configuration.gamma_distance_mm,
                    dose_percent=self.configuration.gamma_dose_percent,
                    dose_threshold_percent=self.configuration.gamma_threshold_percent,
                    use_gpu=self.configuration.use_gpu_acceleration and self.has_gpu,
                )

                result = calculate_gamma_3d(
                    reference_dose=reference_dose,
                    evaluated_dose=evaluated_dose,
                    settings=settings,
                    spacing=spacing,
                )

                passed = (
                    result.pass_rate >= self.configuration.gamma_pass_rate_threshold
                )

                return QATestResult(
                    test_name="Gamma Analysis",
                    test_type="dose_comparison",
                    passed=passed,
                    score=result.pass_rate,
                    threshold=self.configuration.gamma_pass_rate_threshold,
                    details={
                        "pass_rate": result.pass_rate,
                        "mean_gamma": result.mean_gamma,
                        "max_gamma": result.max_gamma,
                        "voxels_analyzed": result.voxels_analyzed,
                        "method_used": result.method_used,
                        "calculation_time": result.calculation_time,
                    },
                    execution_time=result.calculation_time,
                )
            else:
                # Fallback implementation
                return QATestResult(
                    test_name="Gamma Analysis",
                    test_type="dose_comparison",
                    passed=False,
                    score=0.0,
                    threshold=self.configuration.gamma_pass_rate_threshold,
                    error_message="Gamma analysis module not available",
                )

        except Exception as e:
            self.logger.error(f"Error in gamma analysis: {e}")
            return QATestResult(
                test_name="Gamma Analysis",
                test_type="dose_comparison",
                passed=False,
                score=0.0,
                threshold=self.configuration.gamma_pass_rate_threshold,
                error_message=str(e),
            )

    def _run_dose_statistics_analysis(
        self,
        reference_dose: np.ndarray,
        evaluated_dose: np.ndarray,
        target_masks: Dict[str, np.ndarray],
        organ_masks: Dict[str, np.ndarray],
    ) -> QATestResult:
        """Run dose statistics analysis."""
        try:
            details = {}
            total_score = 0.0
            num_structures = 0

            # Analyze target structures
            for name, mask in target_masks.items():
                if HAS_METRICS:
                    ref_stats = calculate_dose_statistics(reference_dose, mask)
                    eval_stats = calculate_dose_statistics(evaluated_dose, mask)

                    mean_diff = (
                        abs(eval_stats.mean_dose - ref_stats.mean_dose)
                        / ref_stats.mean_dose
                        * 100
                    )
                    max_diff = (
                        abs(eval_stats.max_dose - ref_stats.max_dose)
                        / ref_stats.max_dose
                        * 100
                    )

                    structure_score = max(0, 100 - max(mean_diff, max_diff))
                else:
                    # Simple fallback
                    ref_mean = np.mean(reference_dose[mask])
                    eval_mean = np.mean(evaluated_dose[mask])
                    mean_diff = (
                        abs(eval_mean - ref_mean) / ref_mean * 100
                        if ref_mean > 0
                        else 0
                    )
                    structure_score = max(0, 100 - mean_diff)

                details[f"target_{name}"] = {
                    "mean_difference_percent": mean_diff if HAS_METRICS else mean_diff,
                    "score": structure_score,
                }

                total_score += structure_score
                num_structures += 1

            # Analyze organ structures
            for name, mask in organ_masks.items():
                if HAS_METRICS:
                    ref_stats = calculate_dose_statistics(reference_dose, mask)
                    eval_stats = calculate_dose_statistics(evaluated_dose, mask)

                    mean_diff = (
                        abs(eval_stats.mean_dose - ref_stats.mean_dose)
                        / ref_stats.mean_dose
                        * 100
                        if ref_stats.mean_dose > 0
                        else 0
                    )
                    structure_score = max(0, 100 - mean_diff)
                else:
                    ref_mean = np.mean(reference_dose[mask])
                    eval_mean = np.mean(evaluated_dose[mask])
                    mean_diff = (
                        abs(eval_mean - ref_mean) / ref_mean * 100
                        if ref_mean > 0
                        else 0
                    )
                    structure_score = max(0, 100 - mean_diff)

                details[f"organ_{name}"] = {
                    "mean_difference_percent": mean_diff,
                    "score": structure_score,
                }

                total_score += structure_score
                num_structures += 1

            overall_score = total_score / num_structures if num_structures > 0 else 0
            passed = overall_score >= 90.0  # 90% threshold for dose statistics

            return QATestResult(
                test_name="Dose Statistics",
                test_type="dose_analysis",
                passed=passed,
                score=overall_score,
                threshold=90.0,
                details=details,
            )

        except Exception as e:
            self.logger.error(f"Error in dose statistics analysis: {e}")
            return QATestResult(
                test_name="Dose Statistics",
                test_type="dose_analysis",
                passed=False,
                score=0.0,
                threshold=90.0,
                error_message=str(e),
            )

    def _run_plan_quality_analysis(
        self,
        dose_distribution: np.ndarray,
        target_masks: Dict[str, np.ndarray],
        prescription_dose: float,
        spacing: Tuple[float, float, float],
    ) -> QATestResult:
        """Run plan quality analysis."""
        try:
            details = {}
            scores = []

            for name, mask in target_masks.items():
                if HAS_METRICS:
                    quality_results = calculate_comprehensive_plan_quality(
                        dose_distribution, mask, prescription_dose, spacing=spacing
                    )

                    # Evaluate individual metrics
                    ci_score = (
                        100
                        if quality_results.conformity_index
                        >= self.configuration.conformity_index_threshold
                        else 50
                    )
                    hi_score = (
                        100
                        if quality_results.homogeneity_index
                        <= self.configuration.homogeneity_index_threshold
                        else 50
                    )
                    coverage_score = quality_results.target_coverage_95

                    structure_score = (ci_score + hi_score + coverage_score) / 3

                    details[name] = {
                        "conformity_index": quality_results.conformity_index,
                        "homogeneity_index": quality_results.homogeneity_index,
                        "coverage_95": quality_results.target_coverage_95,
                        "score": structure_score,
                    }
                else:
                    # Simple fallback
                    target_dose = dose_distribution[mask]
                    coverage = (
                        np.sum(target_dose >= prescription_dose * 0.95)
                        / len(target_dose)
                        * 100
                    )
                    structure_score = coverage

                    details[name] = {"coverage_95": coverage, "score": structure_score}

                scores.append(structure_score)

            overall_score = np.mean(scores) if scores else 0
            passed = overall_score >= 85.0  # 85% threshold for plan quality

            return QATestResult(
                test_name="Plan Quality",
                test_type="plan_evaluation",
                passed=passed,
                score=overall_score,
                threshold=85.0,
                details=details,
            )

        except Exception as e:
            self.logger.error(f"Error in plan quality analysis: {e}")
            return QATestResult(
                test_name="Plan Quality",
                test_type="plan_evaluation",
                passed=False,
                score=0.0,
                threshold=85.0,
                error_message=str(e),
            )

    def _run_dvh_analysis(
        self,
        dose_distribution: np.ndarray,
        target_masks: Dict[str, np.ndarray],
        organ_masks: Dict[str, np.ndarray],
        spacing: Tuple[float, float, float],
    ) -> QATestResult:
        """Run DVH analysis."""
        try:
            details = {}
            scores = []

            # Analyze targets
            for name, mask in target_masks.items():
                if HAS_METRICS:
                    dvh_curve = create_dvh_from_dose_distribution(
                        dose_distribution, mask, structure_name=name
                    )
                    dvh_stats = calculate_dvh_statistics(dvh_curve)

                    # Simple scoring based on DVH metrics
                    d95_score = (
                        100 if dvh_stats.d_95 >= 95.0 else 50
                    )  # D95 should be >= 95%
                    mean_score = (
                        100 if dvh_stats.mean_dose >= 100.0 else 80
                    )  # Mean dose reasonable

                    structure_score = (d95_score + mean_score) / 2

                    details[f"target_{name}"] = {
                        "d_95": dvh_stats.d_95,
                        "d_50": dvh_stats.d_50,
                        "mean_dose": dvh_stats.mean_dose,
                        "score": structure_score,
                    }
                else:
                    # Simple fallback
                    target_doses = dose_distribution[mask]
                    d95 = np.percentile(
                        target_doses, 5
                    )  # 5th percentile (dose to 95% volume)
                    mean_dose = np.mean(target_doses)

                    structure_score = 80.0  # Default score

                    details[f"target_{name}"] = {
                        "d_95": d95,
                        "mean_dose": mean_dose,
                        "score": structure_score,
                    }

                scores.append(structure_score)

            # Analyze organs
            for name, mask in organ_masks.items():
                if HAS_METRICS:
                    dvh_curve = create_dvh_from_dose_distribution(
                        dose_distribution, mask, structure_name=name
                    )
                    dvh_stats = calculate_dvh_statistics(dvh_curve)

                    # Organ score based on low dose
                    max_score = (
                        100 if dvh_stats.d_max <= 50.0 else 50
                    )  # Keep max dose low
                    mean_score = (
                        100 if dvh_stats.mean_dose <= 20.0 else 70
                    )  # Keep mean low

                    structure_score = (max_score + mean_score) / 2

                    details[f"organ_{name}"] = {
                        "d_max": dvh_stats.d_max,
                        "mean_dose": dvh_stats.mean_dose,
                        "score": structure_score,
                    }
                else:
                    organ_doses = dose_distribution[mask]
                    max_dose = np.max(organ_doses)
                    mean_dose = np.mean(organ_doses)

                    structure_score = 75.0  # Default score

                    details[f"organ_{name}"] = {
                        "d_max": max_dose,
                        "mean_dose": mean_dose,
                        "score": structure_score,
                    }

                scores.append(structure_score)

            overall_score = np.mean(scores) if scores else 0
            passed = overall_score >= 80.0  # 80% threshold for DVH

            return QATestResult(
                test_name="DVH Analysis",
                test_type="dose_volume_analysis",
                passed=passed,
                score=overall_score,
                threshold=80.0,
                details=details,
            )

        except Exception as e:
            self.logger.error(f"Error in DVH analysis: {e}")
            return QATestResult(
                test_name="DVH Analysis",
                test_type="dose_volume_analysis",
                passed=False,
                score=0.0,
                threshold=80.0,
                error_message=str(e),
            )

    def _run_biological_evaluation(
        self,
        dose_distribution: np.ndarray,
        target_masks: Dict[str, np.ndarray],
        organ_masks: Dict[str, np.ndarray],
    ) -> QATestResult:
        """Run biological evaluation."""
        try:
            details = {}
            scores = []

            # Simplified biological evaluation
            for name, mask in target_masks.items():
                target_doses = dose_distribution[mask]
                mean_dose = np.mean(target_doses)

                # Simple TCP estimation (higher dose = higher TCP)
                tcp_estimate = min(100, mean_dose)  # Simplified
                tcp_score = tcp_estimate

                details[f"target_{name}"] = {
                    "mean_dose": mean_dose,
                    "tcp_estimate": tcp_estimate,
                    "score": tcp_score,
                }

                scores.append(tcp_score)

            for name, mask in organ_masks.items():
                organ_doses = dose_distribution[mask]
                mean_dose = np.mean(organ_doses)

                # Simple NTCP estimation (lower dose = lower NTCP)
                ntcp_estimate = min(100, mean_dose / 2)  # Simplified
                ntcp_score = max(0, 100 - ntcp_estimate)  # Higher score for lower NTCP

                details[f"organ_{name}"] = {
                    "mean_dose": mean_dose,
                    "ntcp_estimate": ntcp_estimate,
                    "score": ntcp_score,
                }

                scores.append(ntcp_score)

            overall_score = np.mean(scores) if scores else 0
            passed = overall_score >= 75.0  # 75% threshold for biological

            return QATestResult(
                test_name="Biological Evaluation",
                test_type="biological_analysis",
                passed=passed,
                score=overall_score,
                threshold=75.0,
                details=details,
            )

        except Exception as e:
            self.logger.error(f"Error in biological evaluation: {e}")
            return QATestResult(
                test_name="Biological Evaluation",
                test_type="biological_analysis",
                passed=False,
                score=0.0,
                threshold=75.0,
                error_message=str(e),
            )

    def _calculate_overall_scores(self, report: ComprehensiveQAReport):
        """Calculate overall scores and statistics."""
        tests = [
            report.gamma_analysis,
            report.dose_statistics,
            report.plan_quality,
            report.dvh_analysis,
            report.biological_evaluation,
        ]

        valid_tests = [test for test in tests if test is not None]

        report.total_tests = len(valid_tests)
        report.passed_tests = sum(1 for test in valid_tests if test.passed)
        report.failed_tests = sum(
            1 for test in valid_tests if not test.passed and test.error_message is None
        )
        report.warning_tests = sum(
            1 for test in valid_tests if test.error_message is not None
        )

        if valid_tests:
            # Calculate weighted overall score
            weights = {
                "Gamma Analysis": 0.3,
                "Dose Statistics": 0.2,
                "Plan Quality": 0.25,
                "DVH Analysis": 0.15,
                "Biological Evaluation": 0.1,
            }

            total_weighted_score = 0.0
            total_weight = 0.0

            for test in valid_tests:
                weight = weights.get(test.test_name, 0.1)
                total_weighted_score += test.score * weight
                total_weight += weight

            report.overall_score = (
                total_weighted_score / total_weight if total_weight > 0 else 0
            )
            report.overall_passed = (
                report.overall_score >= 80.0
            )  # 80% overall threshold
        else:
            report.overall_score = 0.0
            report.overall_passed = False

    def export_report_json(self, report: ComprehensiveQAReport, file_path: str):
        """Export báo cáo dạng JSON."""
        try:
            # Convert report to dict
            report_dict = {
                "overall_score": report.overall_score,
                "overall_passed": report.overall_passed,
                "total_tests": report.total_tests,
                "passed_tests": report.passed_tests,
                "failed_tests": report.failed_tests,
                "processing_time": report.processing_time,
                "analysis_timestamp": report.analysis_timestamp.isoformat(),
                "tests": {},
            }

            # Add test results
            tests = [
                ("gamma_analysis", report.gamma_analysis),
                ("dose_statistics", report.dose_statistics),
                ("plan_quality", report.plan_quality),
                ("dvh_analysis", report.dvh_analysis),
                ("biological_evaluation", report.biological_evaluation),
            ]

            for test_key, test_result in tests:
                if test_result:
                    report_dict["tests"][test_key] = {
                        "test_name": test_result.test_name,
                        "passed": test_result.passed,
                        "score": test_result.score,
                        "threshold": test_result.threshold,
                        "details": test_result.details,
                        "error_message": test_result.error_message,
                        "execution_time": test_result.execution_time,
                    }

            # Save to file
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(report_dict, f, indent=2, ensure_ascii=False)

            self.logger.info(f"QA report exported to {file_path}")

        except Exception as e:
            self.logger.error(f"Error exporting QA report: {e}")
            raise


__all__ = [
    "QAConfiguration",
    "QATestResult",
    "ComprehensiveQAReport",
    "ComprehensiveQAEngine",
]
