"""
QuangTPS Plan Quality Assurance Engine

Module đánh giá chất lượng kế hoạch xạ trị toàn diện cho hệ thống QuangTPS.
Cung cấp các công cụ QA từ cơ bản đến nâng cao,
bao gồm phân tích gamma, đánh giá liều, kiểm tra an toàn.
"""

import logging
import os
import json
import numpy as np
import time
import math
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field, asdict
from pathlib import Path
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

# Import scientific libraries
try:
    import numpy as np
    from scipy import ndimage, stats, interpolate
    from scipy.spatial.distance import cdist
    from scipy.optimize import minimize

    HAS_SCIPY = True
    logger.info("NumPy và SciPy được tải thành công")
except ImportError as e:
    logger.warning(f"Scientific libraries không khả dụng: {e}")
    HAS_SCIPY = False

# Import statistical libraries
try:
    import pandas as pd
    from sklearn.metrics import mean_squared_error, mean_absolute_error

    HAS_STATS = True
    logger.info("Statistical libraries được tải thành công")
except ImportError:
    HAS_STATS = False

# Import plotting libraries
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.colors import LinearSegmentedColormap

    HAS_PLOTTING = True
    logger.info("Plotting libraries được tải thành công")
except ImportError:
    HAS_PLOTTING = False

# Import core modules với fallback
try:
    from quangtps.dose.dose_grid import DoseGrid
    from quangtps.evaluation.dvh.dvh_engine import DVHCurve, DVHCalculator
    from quangtps.structures.structure_manager import Structure, StructureManager
    from quangtps.optimization.objectives import OptimizationObjective

    HAS_CORE_MODULES = True
    logger.info("Core modules được tải thành công")
except ImportError as e:
    logger.warning(f"Core modules không khả dụng: {e}")
    HAS_CORE_MODULES = False

    # Fallback classes
    class DoseGrid:
        def __init__(self, *args, **kwargs):
            self.dose_data = np.zeros((64, 64, 32)) if "np" in globals() else None
            self.spacing = (2.0, 2.0, 3.0)

    class DVHCurve:
        def __init__(self, *args, **kwargs):
            self.structure_name = "Unknown"
            self.dose_bins = []
            self.volume_percent = []

    class Structure:
        def __init__(self, *args, **kwargs):
            self.name = "Unknown"
            self.mask = None


class QATestType(Enum):
    """Enum cho các loại test QA."""

    # Dose verification
    GAMMA_ANALYSIS = "gamma_analysis"
    DOSE_DIFFERENCE = "dose_difference"
    DISTANCE_TO_AGREEMENT = "distance_to_agreement"

    # Plan quality metrics
    CONFORMITY_INDEX = "conformity_index"
    HOMOGENEITY_INDEX = "homogeneity_index"
    GRADIENT_INDEX = "gradient_index"

    # DVH analysis
    DVH_COMPARISON = "dvh_comparison"
    DOSE_VOLUME_METRICS = "dose_volume_metrics"

    # Safety checks
    DOSE_LIMITS_CHECK = "dose_limits_check"
    ORGAN_OVERLAP_CHECK = "organ_overlap_check"
    BEAM_GEOMETRY_CHECK = "beam_geometry_check"

    # Statistical analysis
    PLAN_REPRODUCIBILITY = "plan_reproducibility"
    UNCERTAINTY_ANALYSIS = "uncertainty_analysis"

    # Advanced QA
    MACHINE_LOG_ANALYSIS = "machine_log_analysis"
    PATIENT_SPECIFIC_QA = "patient_specific_qa"

    # Comprehensive
    FULL_QA_SUITE = "full_qa_suite"


class QASeverity(Enum):
    """Enum cho mức độ nghiêm trọng của vấn đề QA."""

    PASS = "pass"
    WARNING = "warning"
    MINOR = "minor"
    MAJOR = "major"
    CRITICAL = "critical"
    FAIL = "fail"


@dataclass
class QASettings:
    """Cài đặt cho plan QA engine."""

    # Gamma analysis settings
    gamma_distance_mm: float = 3.0
    gamma_dose_percent: float = 3.0
    gamma_pass_rate_threshold: float = 95.0
    gamma_dose_threshold_percent: float = 10.0

    # Dose difference settings
    dose_difference_threshold_percent: float = 5.0
    dose_difference_threshold_absolute: float = 2.0  # Gy

    # Plan quality thresholds
    conformity_index_min: float = 0.9
    conformity_index_max: float = 1.2
    homogeneity_index_max: float = 0.15
    gradient_index_max: float = 8.0

    # Safety limits
    max_dose_limit_percent: float = 110.0  # % of prescription
    oar_dose_limit_scaling: float = 1.0

    # Statistical settings
    statistical_significance: float = 0.05
    confidence_interval: float = 0.95
    monte_carlo_samples: int = 10000

    # Performance settings
    use_parallel_processing: bool = True
    max_workers: int = 4
    memory_limit_gb: float = 8.0

    # Reporting settings
    generate_detailed_report: bool = True
    include_images: bool = True
    save_intermediate_results: bool = False

    def __post_init__(self):
        """Validate settings."""
        if self.gamma_distance_mm <= 0:
            raise ValueError("Gamma distance phải > 0")
        if not (0 < self.gamma_dose_percent <= 100):
            raise ValueError("Gamma dose percent phải từ 0-100%")
        if not (0 <= self.gamma_pass_rate_threshold <= 100):
            raise ValueError("Gamma pass rate threshold phải từ 0-100%")


@dataclass
class QAResult:
    """Kết quả một test QA cụ thể."""

    test_type: QATestType
    test_name: str
    severity: QASeverity = QASeverity.PASS

    # Test results
    pass_status: bool = True
    measured_value: Optional[float] = None
    expected_value: Optional[float] = None
    tolerance: Optional[float] = None

    # Detailed results
    result_data: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    recommendations: List[str] = field(default_factory=list)

    # Processing info
    calculation_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

    def get_summary(self) -> Dict[str, Any]:
        """Lấy tóm tắt kết quả."""
        return {
            "test_type": self.test_type.value,
            "test_name": self.test_name,
            "severity": self.severity.value,
            "pass_status": self.pass_status,
            "measured_value": self.measured_value,
            "expected_value": self.expected_value,
            "calculation_time": self.calculation_time,
        }


@dataclass
class ComprehensiveQAReport:
    """Báo cáo QA toàn diện."""

    # Overall results
    overall_pass_status: bool = True
    overall_score: float = 100.0  # 0-100
    total_tests: int = 0
    passed_tests: int = 0

    # Individual test results
    qa_results: List[QAResult] = field(default_factory=list)

    # Summary by category
    dose_verification_score: float = 100.0
    plan_quality_score: float = 100.0
    safety_score: float = 100.0

    # Processing info
    total_processing_time: float = 0.0
    report_timestamp: datetime = field(default_factory=datetime.now)

    # Settings used
    settings_used: Optional[QASettings] = None

    def calculate_scores(self):
        """Tính toán các điểm số tổng hợp."""
        if not self.qa_results:
            return

        # Count results by category
        dose_tests = [
            r
            for r in self.qa_results
            if r.test_type
            in [
                QATestType.GAMMA_ANALYSIS,
                QATestType.DOSE_DIFFERENCE,
                QATestType.DISTANCE_TO_AGREEMENT,
            ]
        ]

        quality_tests = [
            r
            for r in self.qa_results
            if r.test_type
            in [
                QATestType.CONFORMITY_INDEX,
                QATestType.HOMOGENEITY_INDEX,
                QATestType.GRADIENT_INDEX,
            ]
        ]

        safety_tests = [
            r
            for r in self.qa_results
            if r.test_type
            in [
                QATestType.DOSE_LIMITS_CHECK,
                QATestType.ORGAN_OVERLAP_CHECK,
                QATestType.BEAM_GEOMETRY_CHECK,
            ]
        ]

        # Calculate category scores
        self.dose_verification_score = self._calculate_category_score(dose_tests)
        self.plan_quality_score = self._calculate_category_score(quality_tests)
        self.safety_score = self._calculate_category_score(safety_tests)

        # Overall score
        self.total_tests = len(self.qa_results)
        self.passed_tests = sum(1 for r in self.qa_results if r.pass_status)

        if self.total_tests > 0:
            pass_rate = self.passed_tests / self.total_tests
            weighted_score = (
                0.4 * self.dose_verification_score
                + 0.3 * self.plan_quality_score
                + 0.3 * self.safety_score
            )
            self.overall_score = pass_rate * weighted_score
            self.overall_pass_status = self.overall_score >= 90.0

    def _calculate_category_score(self, tests: List[QAResult]) -> float:
        """Tính điểm cho một category."""
        if not tests:
            return 100.0

        total_score = 0.0
        for test in tests:
            if test.pass_status:
                score = 100.0
            elif test.severity == QASeverity.WARNING:
                score = 85.0
            elif test.severity == QASeverity.MINOR:
                score = 70.0
            elif test.severity == QASeverity.MAJOR:
                score = 40.0
            else:  # CRITICAL or FAIL
                score = 0.0

            total_score += score

        return total_score / len(tests)

    def get_summary(self) -> Dict[str, Any]:
        """Lấy tóm tắt báo cáo."""
        return {
            "overall_pass_status": self.overall_pass_status,
            "overall_score": self.overall_score,
            "total_tests": self.total_tests,
            "passed_tests": self.passed_tests,
            "dose_verification_score": self.dose_verification_score,
            "plan_quality_score": self.plan_quality_score,
            "safety_score": self.safety_score,
            "total_processing_time": self.total_processing_time,
            "critical_issues": len(
                [r for r in self.qa_results if r.severity == QASeverity.CRITICAL]
            ),
            "major_issues": len(
                [r for r in self.qa_results if r.severity == QASeverity.MAJOR]
            ),
        }


class BaseQATest:
    """
    Base class cho tất cả QA tests.
    """

    def __init__(self, settings: Optional[QASettings] = None):
        self.settings = settings or QASettings()
        self.name = "Base QA Test"
        self.test_type = QATestType.FULL_QA_SUITE

        logger.info(f"{self.name} khởi tạo")

    def run_test(
        self,
        reference_data: Dict[str, Any],
        measured_data: Dict[str, Any],
        progress_callback: Optional[Callable] = None,
    ) -> QAResult:
        """
        Chạy test QA.

        Args:
            reference_data: Dữ liệu tham chiếu
            measured_data: Dữ liệu đo được
            progress_callback: Callback báo cáo tiến trình

        Returns:
            QAResult với kết quả test
        """
        raise NotImplementedError("Subclasses must implement run_test")

    def validate_inputs(
        self, reference_data: Dict[str, Any], measured_data: Dict[str, Any]
    ) -> bool:
        """Validate input data."""
        try:
            if not reference_data or not measured_data:
                logger.error("Reference hoặc measured data trống")
                return False

            return True

        except Exception as e:
            logger.error(f"Lỗi validate inputs: {e}")
            return False


class GammaAnalysisTest(BaseQATest):
    """
    Gamma analysis test cho dose verification.
    """

    def __init__(self, settings: Optional[QASettings] = None):
        super().__init__(settings)
        self.name = "Gamma Analysis Test"
        self.test_type = QATestType.GAMMA_ANALYSIS

        logger.info("Gamma Analysis Test khởi tạo")

    def run_test(
        self,
        reference_data: Dict[str, Any],
        measured_data: Dict[str, Any],
        progress_callback: Optional[Callable] = None,
    ) -> QAResult:
        """Chạy gamma analysis."""
        start_time = time.time()

        try:
            if not self.validate_inputs(reference_data, measured_data):
                raise ValueError("Input validation failed")

            # Extract dose grids
            ref_dose = reference_data.get("dose_grid")
            measured_dose = measured_data.get("dose_grid")

            if ref_dose is None or measured_dose is None:
                raise ValueError("Dose grids không tìm thấy")

            if progress_callback:
                progress_callback(10, "Starting gamma analysis...")

            # Ensure dose grids có cùng shape
            if hasattr(ref_dose, "dose_data") and hasattr(measured_dose, "dose_data"):
                ref_data = np.array(ref_dose.dose_data)
                meas_data = np.array(measured_dose.dose_data)
            else:
                ref_data = np.array(ref_dose)
                meas_data = np.array(measured_dose)

            if ref_data.shape != meas_data.shape:
                logger.warning("Dose grid shapes khác nhau, resizing...")
                meas_data = self._resize_dose_grid(meas_data, ref_data.shape)

            if progress_callback:
                progress_callback(30, "Calculating gamma values...")

            # Calculate gamma
            gamma_map = self._calculate_gamma_3d(
                ref_data,
                meas_data,
                self.settings.gamma_distance_mm,
                self.settings.gamma_dose_percent,
                progress_callback,
            )

            if progress_callback:
                progress_callback(70, "Analyzing gamma results...")

            # Analyze results
            gamma_results = self._analyze_gamma_results(gamma_map, ref_data)

            # Determine pass/fail
            pass_rate = gamma_results["pass_rate"]
            pass_status = pass_rate >= self.settings.gamma_pass_rate_threshold

            severity = QASeverity.PASS if pass_status else QASeverity.MAJOR
            if pass_rate < 80.0:
                severity = QASeverity.CRITICAL
            elif pass_rate < 90.0:
                severity = QASeverity.MAJOR
            elif pass_rate < 95.0:
                severity = QASeverity.MINOR

            recommendations = []
            if not pass_status:
                recommendations.append(
                    f"Gamma pass rate {pass_rate:.1f}% < threshold {self.settings.gamma_pass_rate_threshold}%"
                )
                recommendations.append(
                    "Kiểm tra lại dose calculation và measurement setup"
                )
                recommendations.append("Xem xét tăng tolerance nếu hợp lý")

            result = QAResult(
                test_type=self.test_type,
                test_name=self.name,
                severity=severity,
                pass_status=pass_status,
                measured_value=pass_rate,
                expected_value=self.settings.gamma_pass_rate_threshold,
                tolerance=5.0,  # 5% tolerance
                result_data={
                    "gamma_map": gamma_map,
                    "gamma_statistics": gamma_results,
                    "criteria": f"{self.settings.gamma_distance_mm}mm/{self.settings.gamma_dose_percent}%",
                },
                recommendations=recommendations,
                calculation_time=time.time() - start_time,
            )

            if progress_callback:
                progress_callback(100, "Gamma analysis completed")

            return result

        except Exception as e:
            logger.error(f"Lỗi gamma analysis: {e}")
            return QAResult(
                test_type=self.test_type,
                test_name=self.name,
                severity=QASeverity.FAIL,
                pass_status=False,
                error_message=str(e),
                calculation_time=time.time() - start_time,
            )

    def _calculate_gamma_3d(
        self,
        reference_dose: np.ndarray,
        measured_dose: np.ndarray,
        distance_mm: float,
        dose_percent: float,
        progress_callback: Optional[Callable] = None,
    ) -> np.ndarray:
        """Tính toán gamma 3D."""
        try:
            # Initialize gamma map
            gamma_map = np.full_like(reference_dose, np.inf)

            # Get non-zero dose regions
            dose_threshold = np.max(reference_dose) * (
                self.settings.gamma_dose_threshold_percent / 100.0
            )
            valid_mask = reference_dose >= dose_threshold

            if progress_callback:
                progress_callback(40, "Computing gamma for each voxel...")

            # Calculate gamma for each valid voxel
            valid_indices = np.where(valid_mask)
            total_voxels = len(valid_indices[0])

            for idx, (i, j, k) in enumerate(
                zip(valid_indices[0], valid_indices[1], valid_indices[2])
            ):
                if idx % 1000 == 0 and progress_callback:
                    progress = 40 + (idx / total_voxels) * 25
                    progress_callback(
                        progress, f"Computing gamma: {idx}/{total_voxels}"
                    )

                ref_dose_val = reference_dose[i, j, k]
                meas_dose_val = measured_dose[i, j, k]

                # Calculate gamma tại voxel này
                gamma_val = self._calculate_gamma_single_point(
                    (i, j, k),
                    ref_dose_val,
                    measured_dose,
                    reference_dose,
                    distance_mm,
                    dose_percent,
                )

                gamma_map[i, j, k] = gamma_val

            return gamma_map

        except Exception as e:
            logger.error(f"Lỗi calculate gamma 3D: {e}")
            return (
                np.ones_like(reference_dose) * 10.0
            )  # High gamma values indicate failure

    def _calculate_gamma_single_point(
        self,
        ref_point: Tuple[int, int, int],
        ref_dose: float,
        measured_dose: np.ndarray,
        reference_dose: np.ndarray,
        distance_mm: float,
        dose_percent: float,
    ) -> float:
        """Tính gamma cho một point."""
        try:
            min_gamma = np.inf

            # Search trong sphere around reference point
            search_radius = int(np.ceil(distance_mm / 2.0))  # Assuming 2mm voxel size

            i0, j0, k0 = ref_point
            for di in range(-search_radius, search_radius + 1):
                for dj in range(-search_radius, search_radius + 1):
                    for dk in range(-search_radius, search_radius + 1):
                        i, j, k = i0 + di, j0 + dj, k0 + dk

                        # Check bounds
                        if (
                            0 <= i < measured_dose.shape[0]
                            and 0 <= j < measured_dose.shape[1]
                            and 0 <= k < measured_dose.shape[2]
                        ):
                            # Distance term (assuming 2mm spacing)
                            distance = np.sqrt(
                                (di * 2.0) ** 2 + (dj * 2.0) ** 2 + (dk * 3.0) ** 2
                            )
                            distance_term = distance / distance_mm

                            # Dose difference term
                            dose_diff = abs(measured_dose[i, j, k] - ref_dose)
                            dose_term = dose_diff / (dose_percent * ref_dose / 100.0)

                            # Gamma value
                            gamma = np.sqrt(distance_term**2 + dose_term**2)
                            min_gamma = min(min_gamma, gamma)

            return min_gamma

        except Exception as e:
            logger.error(f"Lỗi calculate gamma single point: {e}")
            return 10.0  # High value indicates failure

    def _analyze_gamma_results(
        self, gamma_map: np.ndarray, reference_dose: np.ndarray
    ) -> Dict[str, Any]:
        """Phân tích kết quả gamma."""
        try:
            # Get valid dose region
            dose_threshold = np.max(reference_dose) * (
                self.settings.gamma_dose_threshold_percent / 100.0
            )
            valid_mask = reference_dose >= dose_threshold

            valid_gamma = gamma_map[valid_mask]
            valid_gamma = valid_gamma[np.isfinite(valid_gamma)]

            if len(valid_gamma) == 0:
                return {
                    "pass_rate": 0.0,
                    "mean_gamma": np.inf,
                    "max_gamma": np.inf,
                    "analyzed_points": 0,
                }

            # Calculate statistics
            pass_count = np.sum(valid_gamma <= 1.0)
            total_count = len(valid_gamma)
            pass_rate = (pass_count / total_count) * 100.0

            return {
                "pass_rate": pass_rate,
                "mean_gamma": float(np.mean(valid_gamma)),
                "max_gamma": float(np.max(valid_gamma)),
                "std_gamma": float(np.std(valid_gamma)),
                "analyzed_points": total_count,
                "passed_points": pass_count,
            }

        except Exception as e:
            logger.error(f"Lỗi analyze gamma results: {e}")
            return {
                "pass_rate": 0.0,
                "mean_gamma": np.inf,
                "max_gamma": np.inf,
                "analyzed_points": 0,
            }

    def _resize_dose_grid(
        self, dose_grid: np.ndarray, target_shape: Tuple[int, int, int]
    ) -> np.ndarray:
        """Resize dose grid để match target shape."""
        try:
            if not HAS_SCIPY:
                logger.warning("SciPy không khả dụng cho resizing")
                return dose_grid

            # Simple zoom-based resizing
            zoom_factors = [t / s for t, s in zip(target_shape, dose_grid.shape)]
            resized_grid = ndimage.zoom(dose_grid, zoom_factors, order=1)

            return resized_grid

        except Exception as e:
            logger.error(f"Lỗi resize dose grid: {e}")
            return dose_grid


class PlanQualityTest(BaseQATest):
    """
    Plan quality test cho các chỉ số CI, HI, GI.
    """

    def __init__(self, settings: Optional[QASettings] = None):
        super().__init__(settings)
        self.name = "Plan Quality Test"
        self.test_type = QATestType.CONFORMITY_INDEX

        logger.info("Plan Quality Test khởi tạo")

    def run_test(
        self,
        reference_data: Dict[str, Any],
        measured_data: Dict[str, Any],
        progress_callback: Optional[Callable] = None,
    ) -> QAResult:
        """Chạy plan quality analysis."""
        start_time = time.time()

        try:
            if not self.validate_inputs(reference_data, measured_data):
                raise ValueError("Input validation failed")

            if progress_callback:
                progress_callback(10, "Analyzing plan quality metrics...")

            # Extract required data
            dose_grid = measured_data.get("dose_grid")
            ptv_structure = measured_data.get("ptv_structure")
            prescription_dose = measured_data.get("prescription_dose", 50.0)  # Gy

            if dose_grid is None or ptv_structure is None:
                raise ValueError("Dose grid hoặc PTV structure không tìm thấy")

            # Extract dose data và PTV mask
            if hasattr(dose_grid, "dose_data"):
                dose_data = np.array(dose_grid.dose_data)
            else:
                dose_data = np.array(dose_grid)

            if hasattr(ptv_structure, "mask"):
                ptv_mask = np.array(ptv_structure.mask)
            else:
                ptv_mask = np.array(ptv_structure)

            if progress_callback:
                progress_callback(30, "Calculating conformity index...")

            # Calculate Conformity Index
            ci = self._calculate_conformity_index(
                dose_data, ptv_mask, prescription_dose
            )

            if progress_callback:
                progress_callback(50, "Calculating homogeneity index...")

            # Calculate Homogeneity Index
            hi = self._calculate_homogeneity_index(
                dose_data, ptv_mask, prescription_dose
            )

            if progress_callback:
                progress_callback(70, "Calculating gradient index...")

            # Calculate Gradient Index
            gi = self._calculate_gradient_index(dose_data, ptv_mask, prescription_dose)

            if progress_callback:
                progress_callback(90, "Evaluating quality metrics...")

            # Evaluate results
            quality_results = {
                "conformity_index": ci,
                "homogeneity_index": hi,
                "gradient_index": gi,
            }

            # Check pass/fail criteria
            ci_pass = (
                self.settings.conformity_index_min
                <= ci
                <= self.settings.conformity_index_max
            )
            hi_pass = hi <= self.settings.homogeneity_index_max
            gi_pass = gi <= self.settings.gradient_index_max

            overall_pass = ci_pass and hi_pass and gi_pass

            # Determine severity
            if overall_pass:
                severity = QASeverity.PASS
            else:
                failed_metrics = []
                if not ci_pass:
                    failed_metrics.append(f"CI={ci:.3f}")
                if not hi_pass:
                    failed_metrics.append(f"HI={hi:.3f}")
                if not gi_pass:
                    failed_metrics.append(f"GI={gi:.1f}")

                if len(failed_metrics) >= 2:
                    severity = QASeverity.MAJOR
                else:
                    severity = QASeverity.MINOR

            recommendations = []
            if not ci_pass:
                recommendations.append(
                    f"Conformity Index {ci:.3f} ngoài range [{self.settings.conformity_index_min}-{self.settings.conformity_index_max}]"
                )
                recommendations.append(
                    "Xem xét tối ưu hóa lại plan để cải thiện conformity"
                )

            if not hi_pass:
                recommendations.append(
                    f"Homogeneity Index {hi:.3f} > {self.settings.homogeneity_index_max}"
                )
                recommendations.append("Cần cải thiện dose homogeneity trong PTV")

            if not gi_pass:
                recommendations.append(
                    f"Gradient Index {gi:.1f} > {self.settings.gradient_index_max}"
                )
                recommendations.append("Dose gradient quá cao, cần tối ưu hóa")

            result = QAResult(
                test_type=self.test_type,
                test_name=self.name,
                severity=severity,
                pass_status=overall_pass,
                measured_value=ci,  # Use CI as primary metric
                expected_value=(
                    self.settings.conformity_index_min
                    + self.settings.conformity_index_max
                )
                / 2,
                tolerance=0.1,
                result_data=quality_results,
                recommendations=recommendations,
                calculation_time=time.time() - start_time,
            )

            if progress_callback:
                progress_callback(100, "Plan quality analysis completed")

            return result

        except Exception as e:
            logger.error(f"Lỗi plan quality test: {e}")
            return QAResult(
                test_type=self.test_type,
                test_name=self.name,
                severity=QASeverity.FAIL,
                pass_status=False,
                error_message=str(e),
                calculation_time=time.time() - start_time,
            )

    def _calculate_conformity_index(
        self, dose_data: np.ndarray, ptv_mask: np.ndarray, prescription_dose: float
    ) -> float:
        """Tính Conformity Index (CI)."""
        try:
            # Volume nhận được prescription dose
            prescription_volume = np.sum(dose_data >= prescription_dose)

            # PTV volume
            ptv_volume = np.sum(ptv_mask > 0)

            # Volume của PTV nhận được prescription dose
            ptv_prescription_volume = np.sum(
                (dose_data >= prescription_dose) & (ptv_mask > 0)
            )

            if ptv_prescription_volume == 0:
                return 0.0

            # Conformity Index = (V_PTV,ref / V_PTV) * (V_PTV,ref / V_ref)
            # Simplified: V_ref / V_PTV,ref
            ci = prescription_volume / ptv_prescription_volume

            return float(ci)

        except Exception as e:
            logger.error(f"Lỗi calculate conformity index: {e}")
            return 0.0

    def _calculate_homogeneity_index(
        self, dose_data: np.ndarray, ptv_mask: np.ndarray, prescription_dose: float
    ) -> float:
        """Tính Homogeneity Index (HI)."""
        try:
            # Extract dose trong PTV
            ptv_doses = dose_data[ptv_mask > 0]

            if len(ptv_doses) == 0:
                return 1.0  # Worst case

            # Calculate D2% và D98%
            d2_percent = np.percentile(ptv_doses, 98)  # Top 2%
            d98_percent = np.percentile(ptv_doses, 2)  # Bottom 2%

            # Homogeneity Index = (D2% - D98%) / D50%
            d50_percent = np.percentile(ptv_doses, 50)

            if d50_percent == 0:
                return 1.0

            hi = (d2_percent - d98_percent) / d50_percent

            return float(hi)

        except Exception as e:
            logger.error(f"Lỗi calculate homogeneity index: {e}")
            return 1.0

    def _calculate_gradient_index(
        self, dose_data: np.ndarray, ptv_mask: np.ndarray, prescription_dose: float
    ) -> float:
        """Tính Gradient Index (GI)."""
        try:
            # Find 50% isodose volume
            dose_50_percent = prescription_dose * 0.5
            volume_50_percent = np.sum(dose_data >= dose_50_percent)

            # Find prescription isodose volume
            volume_prescription = np.sum(dose_data >= prescription_dose)

            if volume_prescription == 0:
                return 100.0  # Very high gradient

            # Gradient Index = V50% / VPrescription
            gi = volume_50_percent / volume_prescription

            return float(gi)

        except Exception as e:
            logger.error(f"Lỗi calculate gradient index: {e}")
            return 100.0


class PlanQAEngine:
    """
    Main Plan QA Engine với comprehensive QA capabilities.
    """

    def __init__(self, settings: Optional[QASettings] = None):
        self.settings = settings or QASettings()

        # Initialize QA tests
        self.qa_tests: Dict[QATestType, BaseQATest] = {}
        self._initialize_qa_tests()

        # Performance monitoring
        self._qa_history: List[ComprehensiveQAReport] = []

        logger.info("Plan QA Engine khởi tạo")

    def _initialize_qa_tests(self):
        """Initialize all available QA tests."""
        try:
            # Gamma analysis
            self.qa_tests[QATestType.GAMMA_ANALYSIS] = GammaAnalysisTest(self.settings)

            # Plan quality
            self.qa_tests[QATestType.CONFORMITY_INDEX] = PlanQualityTest(self.settings)

            logger.info(f"Initialized {len(self.qa_tests)} QA tests")

        except Exception as e:
            logger.error(f"Lỗi initialize QA tests: {e}")

    def get_available_tests(self) -> List[QATestType]:
        """Lấy danh sách tests khả dụng."""
        return list(self.qa_tests.keys())

    def run_comprehensive_qa(
        self,
        reference_data: Dict[str, Any],
        measured_data: Dict[str, Any],
        test_types: Optional[List[QATestType]] = None,
        progress_callback: Optional[Callable] = None,
    ) -> ComprehensiveQAReport:
        """
        Chạy comprehensive QA analysis.
        """
        start_time = time.time()

        try:
            # Use all available tests if not specified
            if test_types is None:
                test_types = self.get_available_tests()

            qa_results = []
            total_tests = len(test_types)

            for i, test_type in enumerate(test_types):
                if progress_callback:
                    overall_progress = (i / total_tests) * 100
                    progress_callback(overall_progress, f"Running {test_type.value}...")

                if test_type in self.qa_tests:
                    # Run individual test
                    test_result = self.qa_tests[test_type].run_test(
                        reference_data, measured_data
                    )
                    qa_results.append(test_result)

                    logger.info(
                        f"Completed {test_type.value}: {'PASS' if test_result.pass_status else 'FAIL'}"
                    )
                else:
                    logger.warning(f"Test {test_type.value} không khả dụng")

            # Create comprehensive report
            report = ComprehensiveQAReport(
                qa_results=qa_results,
                total_processing_time=time.time() - start_time,
                settings_used=self.settings,
            )

            # Calculate scores
            report.calculate_scores()

            # Store in history
            self._qa_history.append(report)

            # Limit history size
            if len(self._qa_history) > 50:
                self._qa_history = self._qa_history[-50:]

            if progress_callback:
                progress_callback(100, "Comprehensive QA completed")

            logger.info(
                f"QA completed: {report.overall_score:.1f}% score, {'PASS' if report.overall_pass_status else 'FAIL'}"
            )

            return report

        except Exception as e:
            logger.error(f"Lỗi comprehensive QA: {e}")

            # Return minimal report
            error_result = QAResult(
                test_type=QATestType.FULL_QA_SUITE,
                test_name="Comprehensive QA",
                severity=QASeverity.FAIL,
                pass_status=False,
                error_message=str(e),
                calculation_time=time.time() - start_time,
            )

            return ComprehensiveQAReport(
                qa_results=[error_result],
                overall_pass_status=False,
                overall_score=0.0,
                total_processing_time=time.time() - start_time,
            )

    def run_single_test(
        self,
        test_type: QATestType,
        reference_data: Dict[str, Any],
        measured_data: Dict[str, Any],
        progress_callback: Optional[Callable] = None,
    ) -> Optional[QAResult]:
        """Chạy một test QA cụ thể."""
        try:
            if test_type not in self.qa_tests:
                logger.error(f"Test {test_type.value} không khả dụng")
                return None

            return self.qa_tests[test_type].run_test(
                reference_data, measured_data, progress_callback
            )

        except Exception as e:
            logger.error(f"Lỗi run single test: {e}")
            return None

    def get_engine_statistics(self) -> Dict[str, Any]:
        """Lấy thống kê performance của engine."""
        try:
            if not self._qa_history:
                return {"total_qa_runs": 0}

            # Calculate statistics
            total_qa_runs = len(self._qa_history)
            total_time = sum(r.total_processing_time for r in self._qa_history)
            avg_time = total_time / total_qa_runs

            # Success rate
            successful_qa = sum(1 for r in self._qa_history if r.overall_pass_status)
            success_rate = successful_qa / total_qa_runs

            # Average scores
            avg_overall_score = np.mean([r.overall_score for r in self._qa_history])
            avg_dose_score = np.mean(
                [r.dose_verification_score for r in self._qa_history]
            )
            avg_quality_score = np.mean(
                [r.plan_quality_score for r in self._qa_history]
            )
            avg_safety_score = np.mean([r.safety_score for r in self._qa_history])

            return {
                "total_qa_runs": total_qa_runs,
                "total_time": total_time,
                "average_time": avg_time,
                "success_rate": success_rate,
                "average_overall_score": float(avg_overall_score),
                "average_dose_score": float(avg_dose_score),
                "average_quality_score": float(avg_quality_score),
                "average_safety_score": float(avg_safety_score),
                "available_tests": [test.value for test in self.get_available_tests()],
                "last_qa": self._qa_history[-1].get_summary()
                if self._qa_history
                else None,
            }

        except Exception as e:
            logger.error(f"Lỗi get engine statistics: {e}")
            return {"error": str(e)}


# Factory functions
def create_qa_engine(settings: Optional[QASettings] = None) -> PlanQAEngine:
    """Factory function để tạo Plan QA Engine."""
    return PlanQAEngine(settings)


def create_qa_test(
    test_type: QATestType, settings: Optional[QASettings] = None
) -> BaseQATest:
    """Factory function để tạo specific QA test."""
    if test_type == QATestType.GAMMA_ANALYSIS:
        return GammaAnalysisTest(settings)
    elif test_type in [
        QATestType.CONFORMITY_INDEX,
        QATestType.HOMOGENEITY_INDEX,
        QATestType.GRADIENT_INDEX,
    ]:
        return PlanQualityTest(settings)
    else:
        raise ValueError(f"Unknown test type: {test_type}")


def create_sample_qa_data() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Tạo sample QA data để test."""
    # Create sample dose grids
    reference_dose = np.zeros((64, 64, 32))
    measured_dose = np.zeros((64, 64, 32))

    # Add some dose distribution
    center = (32, 32, 16)
    for x in range(64):
        for y in range(64):
            for z in range(32):
                distance = np.sqrt(
                    (x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2
                )
                if distance <= 15:
                    reference_dose[x, y, z] = 50.0 * np.exp(-distance / 10.0)
                    # Add some variation for measured dose
                    measured_dose[x, y, z] = reference_dose[x, y, z] * (
                        1.0 + 0.02 * np.random.normal()
                    )

    # Create PTV mask
    ptv_mask = np.zeros((64, 64, 32), dtype=bool)
    for x in range(64):
        for y in range(64):
            for z in range(32):
                distance = np.sqrt(
                    (x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2
                )
                if distance <= 10:
                    ptv_mask[x, y, z] = True

    reference_data = {
        "dose_grid": reference_dose,
        "ptv_structure": ptv_mask,
        "prescription_dose": 50.0,
    }

    measured_data = {
        "dose_grid": measured_dose,
        "ptv_structure": ptv_mask,
        "prescription_dose": 50.0,
    }

    return reference_data, measured_data


if __name__ == "__main__":
    # Test code
    logging.basicConfig(level=logging.INFO)

    # Test QA engine
    qa_engine = create_qa_engine()

    print(
        f"Available QA tests: {[test.value for test in qa_engine.get_available_tests()]}"
    )

    # Create sample data
    reference_data, measured_data = create_sample_qa_data()

    # Test comprehensive QA
    qa_report = qa_engine.run_comprehensive_qa(
        reference_data=reference_data, measured_data=measured_data
    )

    print(f"\nComprehensive QA Results:")
    print(f"  Overall Score: {qa_report.overall_score:.1f}%")
    print(f"  Overall Status: {'PASS' if qa_report.overall_pass_status else 'FAIL'}")
    print(f"  Tests Passed: {qa_report.passed_tests}/{qa_report.total_tests}")
    print(f"  Processing Time: {qa_report.total_processing_time:.1f}s")

    print(f"\nCategory Scores:")
    print(f"  Dose Verification: {qa_report.dose_verification_score:.1f}%")
    print(f"  Plan Quality: {qa_report.plan_quality_score:.1f}%")
    print(f"  Safety: {qa_report.safety_score:.1f}%")

    # Test individual QA results
    print(f"\nIndividual Test Results:")
    for qa_result in qa_report.qa_results:
        print(
            f"  {qa_result.test_name}: {'PASS' if qa_result.pass_status else 'FAIL'} "
            f"({qa_result.severity.value})"
        )
        if qa_result.measured_value is not None:
            print(f"    Measured: {qa_result.measured_value:.3f}")
        if qa_result.recommendations:
            for rec in qa_result.recommendations:
                print(f"    Recommendation: {rec}")

    # Test engine statistics
    stats = qa_engine.get_engine_statistics()
    print(f"\nEngine Statistics: {stats}")

    print("Plan QA Engine test hoàn thành!")
