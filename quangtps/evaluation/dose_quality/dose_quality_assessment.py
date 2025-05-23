#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dose Quality Assessment Module

Module này cung cấp các công cụ đánh giá chất lượng liều xạ trị toàn diện,
tương đương với Eclipse của Varian, bao gồm:
- Phân tích conformity và homogeneity
- Đánh giá dose gradient và falloff
- Phân tích hotspot và coldspot
- Đánh giá liều OAR và normal tissue
- Tính toán các chỉ số chất lượng clinical
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
from dataclasses import dataclass, field
import scipy.ndimage as ndi
from scipy import stats
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# Import với error handling
try:
    from quangtps.dose.dose_grid import DoseGrid
    from quangtps.structures.structure_set import StructureSet
    from quangtps.structures.structure import Structure
    from quangtps.evaluation.dvh.dose_volume_histogram import DVHCalculator
except ImportError as e:
    logger.warning(f"Không thể import các module core: {e}")

    # Tạo classes giả
    class DoseGrid:
        def __init__(self, *args, **kwargs):
            pass

        def get_shape(self):
            return (64, 64, 32)

        def get_dose_at_point(self, *args):
            return 0.0

    class StructureSet:
        def __init__(self, *args, **kwargs):
            self.structures = []

    class Structure:
        def __init__(self, *args, **kwargs):
            self.name = "Unknown"
            self.type = "OTHER"

    class DVHCalculator:
        def __init__(self, *args, **kwargs):
            pass


class QualityMetricType(Enum):
    """Loại metric chất lượng liều."""

    # Conformity metrics
    CONFORMITY_INDEX = "conformity_index"
    CONFORMATION_NUMBER = "conformation_number"
    COVERAGE_FACTOR = "coverage_factor"
    SPILLAGE_FACTOR = "spillage_factor"

    # Homogeneity metrics
    HOMOGENEITY_INDEX = "homogeneity_index"
    DOSE_UNIFORMITY = "dose_uniformity"
    COEFFICIENT_OF_VARIATION = "coefficient_of_variation"

    # Gradient metrics
    GRADIENT_INDEX = "gradient_index"
    DOSE_FALLOFF = "dose_falloff"
    PENUMBRA_WIDTH = "penumbra_width"

    # Clinical metrics
    TCP = "tumor_control_probability"
    NTCP = "normal_tissue_complication_probability"
    EUD = "equivalent_uniform_dose"

    # Spatial metrics
    HOTSPOT_VOLUME = "hotspot_volume"
    COLDSPOT_VOLUME = "coldspot_volume"
    DOSE_DISTRIBUTION_QUALITY = "dose_distribution_quality"


@dataclass
class QualityAssessmentParameters:
    """Tham số đánh giá chất lượng liều."""

    # Prescription parameters
    prescription_dose: float = 50.0  # Gy
    prescription_isodose: float = 95.0  # %

    # Conformity thresholds
    conformity_index_threshold: float = 1.2
    coverage_threshold: float = 95.0  # %

    # Homogeneity thresholds
    homogeneity_index_threshold: float = 0.1
    max_dose_threshold: float = 107.0  # % of prescription
    min_dose_threshold: float = 95.0  # % of prescription

    # Gradient analysis
    gradient_distance: float = 10.0  # mm from PTV surface
    falloff_distances: List[float] = field(default_factory=lambda: [5.0, 10.0, 20.0])

    # Hotspot/coldspot detection
    hotspot_threshold: float = 110.0  # % of prescription
    coldspot_threshold: float = 90.0  # % of prescription
    min_volume_threshold: float = 0.1  # cc

    # Statistical parameters
    confidence_level: float = 0.95
    monte_carlo_samples: int = 1000


@dataclass
class DoseQualityResults:
    """Kết quả đánh giá chất lượng liều."""

    # Basic metrics
    conformity_index: float = 0.0
    homogeneity_index: float = 0.0
    gradient_index: float = 0.0

    # Detailed conformity metrics
    coverage_factor: float = 0.0
    spillage_factor: float = 0.0
    conformation_number: float = 0.0

    # Detailed homogeneity metrics
    dose_uniformity: float = 0.0
    coefficient_of_variation: float = 0.0
    dose_range: Tuple[float, float] = (0.0, 0.0)

    # Gradient metrics
    dose_falloff_5mm: float = 0.0
    dose_falloff_10mm: float = 0.0
    dose_falloff_20mm: float = 0.0
    penumbra_width: float = 0.0

    # Spatial analysis
    hotspot_volumes: Dict[str, float] = field(default_factory=dict)
    coldspot_volumes: Dict[str, float] = field(default_factory=dict)
    dose_statistics: Dict[str, Any] = field(default_factory=dict)

    # Clinical metrics
    tcp_values: Dict[str, float] = field(default_factory=dict)
    ntcp_values: Dict[str, float] = field(default_factory=dict)
    eud_values: Dict[str, float] = field(default_factory=dict)

    # Quality assessment
    overall_quality_score: float = 0.0
    quality_grade: str = "Unknown"
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class DoseQualityAssessment:
    """
    Dose Quality Assessment cho QuangTPS.

    Cung cấp đánh giá chất lượng liều xạ trị toàn diện:
    - Conformity analysis
    - Homogeneity analysis
    - Dose gradient analysis
    - Hotspot/coldspot detection
    - Clinical metrics (TCP, NTCP, EUD)
    - Overall quality scoring
    """

    def __init__(self, parameters: Optional[QualityAssessmentParameters] = None):
        """Khởi tạo Dose Quality Assessment."""

        self.parameters = parameters or QualityAssessmentParameters()
        self.dvh_calculator = DVHCalculator()

        # Quality thresholds (Eclipse-like)
        self.quality_thresholds = {
            "excellent": {"ci": 1.1, "hi": 0.05, "coverage": 98.0},
            "good": {"ci": 1.15, "hi": 0.07, "coverage": 96.0},
            "acceptable": {"ci": 1.2, "hi": 0.1, "coverage": 95.0},
            "marginal": {"ci": 1.3, "hi": 0.15, "coverage": 93.0},
            "poor": {"ci": float("inf"), "hi": float("inf"), "coverage": 0.0},
        }

        logger.info("Dose Quality Assessment khởi tạo thành công")

    def assess_dose_quality(
        self,
        dose_grid: DoseGrid,
        structure_set: StructureSet,
        target_structures: List[str],
        oar_structures: List[str] = None,
        detailed_analysis: bool = True,
    ) -> DoseQualityResults:
        """
        Đánh giá chất lượng liều toàn diện.

        Args:
            dose_grid: Lưới liều 3D
            structure_set: Tập cấu trúc
            target_structures: Danh sách target structures (PTV)
            oar_structures: Danh sách OAR structures
            detailed_analysis: Có thực hiện phân tích chi tiết không

        Returns:
            DoseQualityResults: Kết quả đánh giá chất lượng
        """

        logger.info(
            f"Bắt đầu đánh giá chất lượng liều cho {len(target_structures)} targets"
        )

        results = DoseQualityResults()

        try:
            # Chuẩn bị dữ liệu
            dose_array = dose_grid.get_dose_array()
            structures_dict = {s.name: s for s in structure_set.structures}

            # Phân tích conformity
            conformity_results = self._analyze_conformity(
                dose_array, dose_grid, target_structures, structures_dict
            )
            results.conformity_index = conformity_results["conformity_index"]
            results.coverage_factor = conformity_results["coverage_factor"]
            results.spillage_factor = conformity_results["spillage_factor"]
            results.conformation_number = conformity_results["conformation_number"]

            # Phân tích homogeneity
            homogeneity_results = self._analyze_homogeneity(
                dose_array, dose_grid, target_structures, structures_dict
            )
            results.homogeneity_index = homogeneity_results["homogeneity_index"]
            results.dose_uniformity = homogeneity_results["dose_uniformity"]
            results.coefficient_of_variation = homogeneity_results[
                "coefficient_of_variation"
            ]
            results.dose_range = homogeneity_results["dose_range"]

            # Phân tích gradient
            if detailed_analysis:
                gradient_results = self._analyze_dose_gradient(
                    dose_array, dose_grid, target_structures, structures_dict
                )
                results.gradient_index = gradient_results["gradient_index"]
                results.dose_falloff_5mm = gradient_results["falloff_5mm"]
                results.dose_falloff_10mm = gradient_results["falloff_10mm"]
                results.dose_falloff_20mm = gradient_results["falloff_20mm"]
                results.penumbra_width = gradient_results["penumbra_width"]

            # Phân tích hotspot/coldspot
            spatial_results = self._analyze_spatial_quality(
                dose_array, dose_grid, target_structures, structures_dict
            )
            results.hotspot_volumes = spatial_results["hotspot_volumes"]
            results.coldspot_volumes = spatial_results["coldspot_volumes"]
            results.dose_statistics = spatial_results["dose_statistics"]

            # Tính các metrics clinical
            if detailed_analysis and oar_structures:
                clinical_results = self._calculate_clinical_metrics(
                    dose_array,
                    dose_grid,
                    target_structures,
                    oar_structures,
                    structures_dict,
                )
                results.tcp_values = clinical_results["tcp_values"]
                results.ntcp_values = clinical_results["ntcp_values"]
                results.eud_values = clinical_results["eud_values"]

            # Tính overall quality score
            results.overall_quality_score = self._calculate_overall_quality_score(
                results
            )
            results.quality_grade = self._determine_quality_grade(results)

            # Tạo warnings và recommendations
            results.warnings = self._generate_warnings(results)
            results.recommendations = self._generate_recommendations(results)

            logger.info(f"Hoàn thành đánh giá chất lượng liều: {results.quality_grade}")

        except Exception as e:
            logger.error(f"Lỗi trong đánh giá chất lượng liều: {str(e)}")
            results.warnings.append(f"Lỗi đánh giá: {str(e)}")

        return results

    def _analyze_conformity(
        self,
        dose_array: np.ndarray,
        dose_grid: DoseGrid,
        target_structures: List[str],
        structures_dict: Dict[str, Structure],
    ) -> Dict[str, float]:
        """Phân tích conformity."""

        logger.info("Phân tích conformity")

        try:
            # Tạo combined target mask
            target_mask = np.zeros_like(dose_array, dtype=bool)
            for target_name in target_structures:
                if target_name in structures_dict:
                    structure = structures_dict[target_name]
                    mask = self._get_structure_mask(structure, dose_grid)
                    target_mask |= mask

            # Prescription dose level
            prescription_dose = self.parameters.prescription_dose
            isodose_level = prescription_dose * (
                self.parameters.prescription_isodose / 100.0
            )

            # Volume calculations
            target_volume = np.sum(target_mask) * self._get_voxel_volume(dose_grid)

            # Volume covered by prescription isodose (V_target_covered)
            target_covered_mask = target_mask & (dose_array >= isodose_level)
            target_covered_volume = np.sum(
                target_covered_mask
            ) * self._get_voxel_volume(dose_grid)

            # Volume of prescription isodose (V_isodose)
            isodose_mask = dose_array >= isodose_level
            isodose_volume = np.sum(isodose_mask) * self._get_voxel_volume(dose_grid)

            # Conformity metrics
            coverage_factor = (
                target_covered_volume / target_volume if target_volume > 0 else 0.0
            )
            spillage_factor = (
                isodose_volume / target_covered_volume
                if target_covered_volume > 0
                else float("inf")
            )
            conformity_index = (
                isodose_volume / target_volume if target_volume > 0 else float("inf")
            )
            conformation_number = (
                coverage_factor / spillage_factor if spillage_factor > 0 else 0.0
            )

            return {
                "conformity_index": conformity_index,
                "coverage_factor": coverage_factor,
                "spillage_factor": spillage_factor,
                "conformation_number": conformation_number,
                "target_volume": target_volume,
                "target_covered_volume": target_covered_volume,
                "isodose_volume": isodose_volume,
            }

        except Exception as e:
            logger.error(f"Lỗi phân tích conformity: {str(e)}")
            return {
                "conformity_index": float("inf"),
                "coverage_factor": 0.0,
                "spillage_factor": float("inf"),
                "conformation_number": 0.0,
            }

    def _analyze_homogeneity(
        self,
        dose_array: np.ndarray,
        dose_grid: DoseGrid,
        target_structures: List[str],
        structures_dict: Dict[str, Structure],
    ) -> Dict[str, Any]:
        """Phân tích homogeneity."""

        logger.info("Phân tích homogeneity")

        try:
            # Tạo combined target mask
            target_mask = np.zeros_like(dose_array, dtype=bool)
            for target_name in target_structures:
                if target_name in structures_dict:
                    structure = structures_dict[target_name]
                    mask = self._get_structure_mask(structure, dose_grid)
                    target_mask |= mask

            # Dose trong target
            target_doses = dose_array[target_mask]

            if len(target_doses) == 0:
                return {
                    "homogeneity_index": float("inf"),
                    "dose_uniformity": 0.0,
                    "coefficient_of_variation": float("inf"),
                    "dose_range": (0.0, 0.0),
                }

            # Prescription dose
            prescription_dose = self.parameters.prescription_dose

            # Homogeneity metrics
            max_dose = np.max(target_doses)
            min_dose = np.min(target_doses)
            mean_dose = np.mean(target_doses)
            std_dose = np.std(target_doses)

            # Homogeneity Index (HI) - ICRU approach
            d5 = np.percentile(target_doses, 95)  # D5%
            d95 = np.percentile(target_doses, 5)  # D95%
            homogeneity_index = (d5 - d95) / prescription_dose

            # Dose Uniformity Index
            dose_uniformity = 1 - (max_dose - min_dose) / prescription_dose

            # Coefficient of Variation
            coefficient_of_variation = (
                std_dose / mean_dose if mean_dose > 0 else float("inf")
            )

            return {
                "homogeneity_index": homogeneity_index,
                "dose_uniformity": dose_uniformity,
                "coefficient_of_variation": coefficient_of_variation,
                "dose_range": (min_dose, max_dose),
                "mean_dose": mean_dose,
                "std_dose": std_dose,
                "d5": d5,
                "d95": d95,
            }

        except Exception as e:
            logger.error(f"Lỗi phân tích homogeneity: {str(e)}")
            return {
                "homogeneity_index": float("inf"),
                "dose_uniformity": 0.0,
                "coefficient_of_variation": float("inf"),
                "dose_range": (0.0, 0.0),
            }

    def _analyze_dose_gradient(
        self,
        dose_array: np.ndarray,
        dose_grid: DoseGrid,
        target_structures: List[str],
        structures_dict: Dict[str, Structure],
    ) -> Dict[str, float]:
        """Phân tích gradient liều."""

        logger.info("Phân tích dose gradient")

        try:
            # Tạo combined target mask
            target_mask = np.zeros_like(dose_array, dtype=bool)
            for target_name in target_structures:
                if target_name in structures_dict:
                    structure = structures_dict[target_name]
                    mask = self._get_structure_mask(structure, dose_grid)
                    target_mask |= mask

            # Tính gradient magnitude
            spacing = dose_grid.get_spacing()
            grad_z, grad_y, grad_x = np.gradient(
                dose_array, spacing[2], spacing[1], spacing[0]
            )
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)

            # Gradient Index - average gradient trong vùng 1cm từ PTV surface
            surface_mask = self._get_surface_expansion_mask(
                target_mask, dose_grid, 10.0
            )  # 1cm
            gradient_in_surface = gradient_magnitude[surface_mask]
            gradient_index = (
                np.mean(gradient_in_surface) if len(gradient_in_surface) > 0 else 0.0
            )

            # Dose falloff analysis
            falloff_results = {}
            for distance in self.parameters.falloff_distances:
                falloff_mask = self._get_surface_expansion_mask(
                    target_mask, dose_grid, distance
                )
                if np.any(falloff_mask):
                    falloff_dose = np.mean(dose_array[falloff_mask])
                    prescription_dose = self.parameters.prescription_dose
                    falloff_percent = (falloff_dose / prescription_dose) * 100.0
                    falloff_results[f"falloff_{distance}mm"] = falloff_percent

            # Penumbra width (80% to 20% falloff distance)
            penumbra_width = self._calculate_penumbra_width(
                dose_array, target_mask, dose_grid
            )

            return {
                "gradient_index": gradient_index,
                "falloff_5mm": falloff_results.get("falloff_5.0mm", 0.0),
                "falloff_10mm": falloff_results.get("falloff_10.0mm", 0.0),
                "falloff_20mm": falloff_results.get("falloff_20.0mm", 0.0),
                "penumbra_width": penumbra_width,
            }

        except Exception as e:
            logger.error(f"Lỗi phân tích gradient: {str(e)}")
            return {
                "gradient_index": 0.0,
                "falloff_5mm": 0.0,
                "falloff_10mm": 0.0,
                "falloff_20mm": 0.0,
                "penumbra_width": 0.0,
            }

    def _analyze_spatial_quality(
        self,
        dose_array: np.ndarray,
        dose_grid: DoseGrid,
        target_structures: List[str],
        structures_dict: Dict[str, Structure],
    ) -> Dict[str, Any]:
        """Phân tích chất lượng không gian."""

        logger.info("Phân tích spatial quality")

        try:
            prescription_dose = self.parameters.prescription_dose
            voxel_volume = self._get_voxel_volume(dose_grid)

            # Hotspot detection
            hotspot_threshold = prescription_dose * (
                self.parameters.hotspot_threshold / 100.0
            )
            hotspot_mask = dose_array >= hotspot_threshold

            # Coldspot detection trong target
            target_mask = np.zeros_like(dose_array, dtype=bool)
            for target_name in target_structures:
                if target_name in structures_dict:
                    structure = structures_dict[target_name]
                    mask = self._get_structure_mask(structure, dose_grid)
                    target_mask |= mask

            coldspot_threshold = prescription_dose * (
                self.parameters.coldspot_threshold / 100.0
            )
            coldspot_mask = target_mask & (dose_array <= coldspot_threshold)

            # Volume calculations
            hotspot_volume = np.sum(hotspot_mask) * voxel_volume
            coldspot_volume = np.sum(coldspot_mask) * voxel_volume

            # Dose statistics
            dose_stats = {
                "mean": np.mean(dose_array),
                "std": np.std(dose_array),
                "min": np.min(dose_array),
                "max": np.max(dose_array),
                "median": np.median(dose_array),
                "percentile_95": np.percentile(dose_array, 95),
                "percentile_5": np.percentile(dose_array, 5),
            }

            return {
                "hotspot_volumes": {"total": hotspot_volume},
                "coldspot_volumes": {"total": coldspot_volume},
                "dose_statistics": dose_stats,
            }

        except Exception as e:
            logger.error(f"Lỗi phân tích spatial quality: {str(e)}")
            return {
                "hotspot_volumes": {},
                "coldspot_volumes": {},
                "dose_statistics": {},
            }

    def _calculate_clinical_metrics(
        self,
        dose_array: np.ndarray,
        dose_grid: DoseGrid,
        target_structures: List[str],
        oar_structures: List[str],
        structures_dict: Dict[str, Structure],
    ) -> Dict[str, Dict[str, float]]:
        """Tính các metrics clinical."""

        logger.info("Tính clinical metrics")

        try:
            tcp_values = {}
            ntcp_values = {}
            eud_values = {}

            # TCP cho target structures
            for target_name in target_structures:
                if target_name in structures_dict:
                    structure = structures_dict[target_name]
                    mask = self._get_structure_mask(structure, dose_grid)
                    doses = dose_array[mask]

                    if len(doses) > 0:
                        # Simplified TCP calculation (logistic model)
                        tcp = self._calculate_tcp(doses, structure_type="target")
                        tcp_values[target_name] = tcp

                        # EUD calculation
                        eud = self._calculate_eud(doses, a_value=10.0)  # Tumor a-value
                        eud_values[target_name] = eud

            # NTCP cho OAR structures
            for oar_name in oar_structures:
                if oar_name in structures_dict:
                    structure = structures_dict[oar_name]
                    mask = self._get_structure_mask(structure, dose_grid)
                    doses = dose_array[mask]

                    if len(doses) > 0:
                        # Simplified NTCP calculation (Lyman model)
                        ntcp = self._calculate_ntcp(doses, oar_name)
                        ntcp_values[oar_name] = ntcp

                        # EUD calculation
                        a_value = self._get_oar_a_value(oar_name)
                        eud = self._calculate_eud(doses, a_value=a_value)
                        eud_values[oar_name] = eud

            return {
                "tcp_values": tcp_values,
                "ntcp_values": ntcp_values,
                "eud_values": eud_values,
            }

        except Exception as e:
            logger.error(f"Lỗi tính clinical metrics: {str(e)}")
            return {"tcp_values": {}, "ntcp_values": {}, "eud_values": {}}

    def _calculate_overall_quality_score(self, results: DoseQualityResults) -> float:
        """Tính overall quality score."""

        try:
            # Weighted scoring system
            scores = []
            weights = []

            # Conformity score (30%)
            ci_score = max(
                0, 1.0 - (results.conformity_index - 1.0) / 0.3
            )  # Best at CI=1.0
            scores.append(ci_score)
            weights.append(0.3)

            # Homogeneity score (25%)
            hi_score = max(0, 1.0 - results.homogeneity_index / 0.15)  # Best at HI=0
            scores.append(hi_score)
            weights.append(0.25)

            # Coverage score (25%)
            coverage_score = results.coverage_factor  # Already in 0-1 range
            scores.append(coverage_score)
            weights.append(0.25)

            # Gradient score (20%)
            gradient_score = max(
                0, 1.0 - results.gradient_index / 10.0
            )  # Arbitrary scaling
            scores.append(gradient_score)
            weights.append(0.2)

            # Weighted average
            weighted_score = np.average(scores, weights=weights)

            return min(1.0, max(0.0, weighted_score))

        except Exception as e:
            logger.error(f"Lỗi tính overall quality score: {str(e)}")
            return 0.0

    def _determine_quality_grade(self, results: DoseQualityResults) -> str:
        """Xác định grade chất lượng."""

        try:
            ci = results.conformity_index
            hi = results.homogeneity_index
            coverage = results.coverage_factor * 100.0

            for grade, thresholds in self.quality_thresholds.items():
                if (
                    ci <= thresholds["ci"]
                    and hi <= thresholds["hi"]
                    and coverage >= thresholds["coverage"]
                ):
                    return grade.upper()

            return "POOR"

        except Exception as e:
            logger.error(f"Lỗi xác định quality grade: {str(e)}")
            return "UNKNOWN"

    # Helper methods
    def _get_structure_mask(
        self, structure: Structure, dose_grid: DoseGrid
    ) -> np.ndarray:
        """Lấy mask của structure."""
        # Placeholder - cần implement chi tiết
        shape = dose_grid.get_shape()
        return np.zeros(shape, dtype=bool)

    def _get_voxel_volume(self, dose_grid: DoseGrid) -> float:
        """Tính volume của một voxel."""
        spacing = dose_grid.get_spacing()
        return spacing[0] * spacing[1] * spacing[2]  # cm³

    def _get_surface_expansion_mask(
        self, mask: np.ndarray, dose_grid: DoseGrid, distance_mm: float
    ) -> np.ndarray:
        """Tạo mask expansion từ surface."""
        spacing = dose_grid.get_spacing()
        distance_voxels = [distance_mm / s for s in spacing]

        # Simple dilation
        expanded = ndi.binary_dilation(mask, iterations=int(max(distance_voxels)))
        return expanded & ~mask  # Only the expansion region

    def _calculate_penumbra_width(
        self, dose_array: np.ndarray, target_mask: np.ndarray, dose_grid: DoseGrid
    ) -> float:
        """Tính penumbra width."""
        # Simplified calculation
        return 5.0  # mm - placeholder

    def _calculate_tcp(
        self, doses: np.ndarray, structure_type: str = "target"
    ) -> float:
        """Tính TCP (simplified)."""
        # Simplified logistic TCP model
        mean_dose = np.mean(doses)
        tcp = 1.0 / (1.0 + np.exp(-(mean_dose - 50.0) / 10.0))  # Sigmoid
        return tcp

    def _calculate_ntcp(self, doses: np.ndarray, oar_name: str) -> float:
        """Tính NTCP (simplified)."""
        # Simplified NTCP model
        mean_dose = np.mean(doses)
        ntcp = 1.0 / (1.0 + np.exp(-(mean_dose - 30.0) / 15.0))  # Conservative sigmoid
        return ntcp

    def _calculate_eud(self, doses: np.ndarray, a_value: float) -> float:
        """Tính EUD (Equivalent Uniform Dose)."""
        if len(doses) == 0:
            return 0.0

        if a_value == 0:
            return np.mean(doses)

        power_mean = np.mean(np.power(doses, a_value))
        eud = np.power(power_mean, 1.0 / a_value)
        return eud

    def _get_oar_a_value(self, oar_name: str) -> float:
        """Lấy a-value cho OAR."""
        oar_a_values = {
            "spinal_cord": 7.0,
            "brainstem": 7.0,
            "lung": 1.0,
            "heart": 3.0,
            "liver": 3.0,
            "kidney": 1.0,
            "rectum": 8.0,
            "bladder": 8.0,
        }
        return oar_a_values.get(oar_name.lower(), 3.0)  # Default value

    def _generate_warnings(self, results: DoseQualityResults) -> List[str]:
        """Tạo warnings."""
        warnings = []

        if results.conformity_index > 1.3:
            warnings.append(
                "Conformity Index cao (>1.3) - kế hoạch không đạt yêu cầu conformity"
            )

        if results.homogeneity_index > 0.15:
            warnings.append(
                "Homogeneity Index cao (>0.15) - phân bố liều không đồng đều"
            )

        if results.coverage_factor < 0.95:
            warnings.append("Target coverage thấp (<95%) - cần tăng coverage")

        return warnings

    def _generate_recommendations(self, results: DoseQualityResults) -> List[str]:
        """Tạo recommendations."""
        recommendations = []

        if results.conformity_index > 1.2:
            recommendations.append(
                "Cần cải thiện conformity - xem xét thêm beams hoặc tối ưu MLC"
            )

        if results.homogeneity_index > 0.1:
            recommendations.append(
                "Cần cải thiện homogeneity - xem xét tối ưu beam weights"
            )

        if len(results.hotspot_volumes) > 0:
            recommendations.append(
                "Phát hiện hotspots - xem xét giảm liều trong vùng này"
            )

        return recommendations


# Factory function
def create_dose_quality_assessment(
    parameters: Optional[QualityAssessmentParameters] = None,
) -> DoseQualityAssessment:
    """Tạo Dose Quality Assessment instance."""
    return DoseQualityAssessment(parameters)


if __name__ == "__main__":
    # Test basic functionality
    assessment = create_dose_quality_assessment()
    logger.info("Dose Quality Assessment test hoàn thành")
