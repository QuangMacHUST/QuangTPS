"""
QuangTPS DVH Analysis Engine

Module phân tích Dose Volume Histogram toàn diện cho hệ thống QuangTPS.
Cung cấp tính toán DVH, thống kê liều, đánh giá chỉ số lâm sàng,
và tích hợp với các workflow đánh giá kế hoạch xạ trị.
"""

import logging
import os
import json
import numpy as np
import math
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field, asdict
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor
import time

logger = logging.getLogger(__name__)

# Import scientific libraries với fallback
try:
    import numpy as np
    from scipy import interpolate, integrate
    from scipy.stats import percentileofscore

    HAS_SCIPY = True
    logger.info("NumPy và SciPy được tải thành công")
except ImportError as e:
    logger.warning(f"Scientific libraries không khả dụng: {e}")
    HAS_SCIPY = False

# Import plotting libraries với fallback
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.colors import LinearSegmentedColormap

    HAS_MATPLOTLIB = True
    logger.info("Matplotlib được tải thành công")
except ImportError as e:
    logger.warning(f"Matplotlib không khả dụng: {e}")
    HAS_MATPLOTLIB = False

    # Fallback classes
    class plt:
        @staticmethod
        def figure(*args, **kwargs):
            return None

        @staticmethod
        def show():
            pass


# Import core modules với fallback
try:
    from quangtps.core.geometry.geometry_utils import calculate_volume_from_mask
    from quangtps.core.statistics.dose_statistics import DoseStatistics

    HAS_CORE_MODULES = True
    logger.info("Core modules được tải thành công")
except ImportError as e:
    logger.warning(f"Core modules không khả dụng: {e}")
    HAS_CORE_MODULES = False

    # Fallback functions
    def calculate_volume_from_mask(mask, spacing):
        return np.sum(mask) * np.prod(spacing) / 1000.0  # Convert to cc

    class DoseStatistics:
        def __init__(self, *args, **kwargs):
            pass


@dataclass
class DVHPoint:
    """Điểm dữ liệu trên DVH curve."""

    dose: float  # Gy
    volume_percent: float  # %
    volume_absolute: float  # cc

    def __post_init__(self):
        """Validate DVH point."""
        if self.dose < 0:
            raise ValueError("Dose không thể âm")
        if not (0 <= self.volume_percent <= 100):
            raise ValueError("Volume percent phải từ 0-100%")
        if self.volume_absolute < 0:
            raise ValueError("Volume absolute không thể âm")


@dataclass
class DVHCurve:
    """Đại diện cho một DVH curve hoàn chỉnh."""

    structure_name: str
    dvh_type: str = "CUMULATIVE"  # CUMULATIVE, DIFFERENTIAL

    # DVH data points
    dose_bins: List[float] = field(default_factory=list)  # Gy
    volume_percent: List[float] = field(default_factory=list)  # %
    volume_absolute: List[float] = field(default_factory=list)  # cc

    # Structure properties
    total_volume: float = 0.0  # cc
    structure_color: Tuple[float, float, float] = (1.0, 0.0, 0.0)

    # Statistics
    mean_dose: Optional[float] = None  # Gy
    median_dose: Optional[float] = None  # Gy
    modal_dose: Optional[float] = None  # Gy
    std_dose: Optional[float] = None  # Gy
    min_dose: Optional[float] = None  # Gy
    max_dose: Optional[float] = None  # Gy

    # Clinical metrics
    d95_percent: Optional[float] = None  # D95% (Gy)
    d50_percent: Optional[float] = None  # D50% (Gy)
    d5_percent: Optional[float] = None  # D5% (Gy)
    d2_percent: Optional[float] = None  # D2% (Gy)
    d98_percent: Optional[float] = None  # D98% (Gy)

    # Volume metrics
    v95_percent: Optional[float] = None  # V95% (%)
    v107_percent: Optional[float] = None  # V107% (%)
    v110_percent: Optional[float] = None  # V110% (%)
    v20gy: Optional[float] = None  # V20Gy (%)
    v30gy: Optional[float] = None  # V30Gy (%)

    # Quality indices
    homogeneity_index: Optional[float] = None  # HI
    conformity_index: Optional[float] = None  # CI
    gradient_index: Optional[float] = None  # GI

    def __post_init__(self):
        """Validate và tính toán các metrics."""
        if not self.structure_name:
            raise ValueError("Structure name là bắt buộc")

        if len(self.dose_bins) != len(self.volume_percent):
            raise ValueError("Dose bins và volume percent phải có cùng độ dài")

        if self.dose_bins and self.volume_percent:
            self._calculate_statistics()

    def _calculate_statistics(self):
        """Tính toán các thống kê DVH."""
        try:
            if not self.dose_bins or not self.volume_percent:
                return

            dose_array = np.array(self.dose_bins)
            volume_array = np.array(self.volume_percent)

            # Basic statistics
            if len(dose_array) > 0:
                self.min_dose = float(np.min(dose_array))
                self.max_dose = float(np.max(dose_array))

                # Calculate weighted statistics
                if len(volume_array) > 1:
                    # Differential volume for weighting
                    diff_volume = np.diff(np.append(100, volume_array[::-1]))
                    diff_volume = np.abs(diff_volume)

                    if np.sum(diff_volume) > 0:
                        weights = diff_volume / np.sum(diff_volume)
                        self.mean_dose = float(np.average(dose_array, weights=weights))
                        self.std_dose = float(
                            np.sqrt(
                                np.average(
                                    (dose_array - self.mean_dose) ** 2, weights=weights
                                )
                            )
                        )
                    else:
                        self.mean_dose = float(np.mean(dose_array))
                        self.std_dose = float(np.std(dose_array))
                else:
                    self.mean_dose = float(dose_array[0])
                    self.std_dose = 0.0

                # Calculate Dx percentiles
                self._calculate_dose_percentiles()

                # Calculate Vx metrics
                self._calculate_volume_metrics()

                # Calculate quality indices
                self._calculate_quality_indices()

        except Exception as e:
            logger.error(f"Lỗi tính toán DVH statistics: {e}")

    def _calculate_dose_percentiles(self):
        """Tính toán các dose percentiles (D95%, D50%, etc.)."""
        try:
            dose_array = np.array(self.dose_bins)
            volume_array = np.array(self.volume_percent)

            if len(dose_array) < 2:
                return

            # Interpolate để tìm dose tại volume percentages cụ thể
            if HAS_SCIPY:
                # Đảm bảo volume giảm dần (cumulative DVH)
                if volume_array[0] < volume_array[-1]:
                    volume_array = volume_array[::-1]
                    dose_array = dose_array[::-1]

                # Tạo interpolation function
                if len(np.unique(volume_array)) > 1:
                    interp_func = interpolate.interp1d(
                        volume_array,
                        dose_array,
                        kind="linear",
                        bounds_error=False,
                        fill_value="extrapolate",
                    )

                    # Calculate percentiles
                    self.d95_percent = float(interp_func(95))
                    self.d50_percent = float(interp_func(50))
                    self.d5_percent = float(interp_func(5))
                    self.d2_percent = float(interp_func(2))
                    self.d98_percent = float(interp_func(98))

                    # Median dose estimation
                    self.median_dose = self.d50_percent

        except Exception as e:
            logger.error(f"Lỗi tính toán dose percentiles: {e}")

    def _calculate_volume_metrics(self):
        """Tính toán các volume metrics (V95%, V20Gy, etc.)."""
        try:
            dose_array = np.array(self.dose_bins)
            volume_array = np.array(self.volume_percent)

            if len(dose_array) < 2:
                return

            if HAS_SCIPY:
                # Tạo interpolation function cho volume từ dose
                if len(np.unique(dose_array)) > 1:
                    interp_func = interpolate.interp1d(
                        dose_array,
                        volume_array,
                        kind="linear",
                        bounds_error=False,
                        fill_value=0,
                    )

                    # Calculate volume at prescription doses
                    if self.d95_percent:
                        prescription_dose = (
                            self.d95_percent
                        )  # Assuming prescription ~ D95%
                        self.v95_percent = float(interp_func(0.95 * prescription_dose))
                        self.v107_percent = float(interp_func(1.07 * prescription_dose))
                        self.v110_percent = float(interp_func(1.10 * prescription_dose))

                    # Calculate volume at specific doses
                    self.v20gy = float(interp_func(20))
                    self.v30gy = float(interp_func(30))

        except Exception as e:
            logger.error(f"Lỗi tính toán volume metrics: {e}")

    def _calculate_quality_indices(self):
        """Tính toán các chỉ số chất lượng."""
        try:
            # Homogeneity Index (HI)
            if self.d2_percent and self.d98_percent and self.d50_percent:
                self.homogeneity_index = (
                    self.d2_percent - self.d98_percent
                ) / self.d50_percent

            # Conformity Index (simplified version)
            if self.v95_percent and self.total_volume > 0:
                target_volume_95 = self.total_volume * (self.v95_percent / 100.0)
                if target_volume_95 > 0:
                    self.conformity_index = self.total_volume / target_volume_95

        except Exception as e:
            logger.error(f"Lỗi tính toán quality indices: {e}")

    def get_dose_at_volume(self, volume_percent: float) -> Optional[float]:
        """Lấy dose tại volume percent cụ thể."""
        try:
            if not self.dose_bins or not self.volume_percent:
                return None

            dose_array = np.array(self.dose_bins)
            vol_array = np.array(self.volume_percent)

            if HAS_SCIPY and len(dose_array) > 1:
                # Ensure volume is descending for cumulative DVH
                if vol_array[0] < vol_array[-1]:
                    vol_array = vol_array[::-1]
                    dose_array = dose_array[::-1]

                if len(np.unique(vol_array)) > 1:
                    interp_func = interpolate.interp1d(
                        vol_array,
                        dose_array,
                        kind="linear",
                        bounds_error=False,
                        fill_value=0,
                    )
                    return float(interp_func(volume_percent))

            return None

        except Exception as e:
            logger.error(f"Lỗi get dose at volume: {e}")
            return None

    def get_volume_at_dose(self, dose: float) -> Optional[float]:
        """Lấy volume percent tại dose cụ thể."""
        try:
            if not self.dose_bins or not self.volume_percent:
                return None

            dose_array = np.array(self.dose_bins)
            vol_array = np.array(self.volume_percent)

            if HAS_SCIPY and len(dose_array) > 1:
                if len(np.unique(dose_array)) > 1:
                    interp_func = interpolate.interp1d(
                        dose_array,
                        vol_array,
                        kind="linear",
                        bounds_error=False,
                        fill_value=0,
                    )
                    return float(interp_func(dose))

            return None

        except Exception as e:
            logger.error(f"Lỗi get volume at dose: {e}")
            return None

    def to_differential(self) -> "DVHCurve":
        """Chuyển đổi cumulative DVH thành differential DVH."""
        try:
            if self.dvh_type == "DIFFERENTIAL":
                return self  # Already differential

            diff_curve = DVHCurve(
                structure_name=self.structure_name,
                dvh_type="DIFFERENTIAL",
                structure_color=self.structure_color,
                total_volume=self.total_volume,
            )

            if len(self.volume_percent) > 1:
                # Calculate differential volumes
                dose_bins = self.dose_bins.copy()
                diff_volumes = []

                for i in range(len(self.volume_percent) - 1):
                    diff_vol = abs(self.volume_percent[i] - self.volume_percent[i + 1])
                    diff_volumes.append(diff_vol)

                # Use midpoint doses for differential bins
                diff_doses = []
                for i in range(len(dose_bins) - 1):
                    mid_dose = (dose_bins[i] + dose_bins[i + 1]) / 2.0
                    diff_doses.append(mid_dose)

                diff_curve.dose_bins = diff_doses
                diff_curve.volume_percent = diff_volumes

                # Calculate absolute volumes
                if self.total_volume > 0:
                    diff_curve.volume_absolute = [
                        (vol_pct / 100.0) * self.total_volume
                        for vol_pct in diff_volumes
                    ]

            return diff_curve

        except Exception as e:
            logger.error(f"Lỗi chuyển đổi to differential: {e}")
            return self

    def export_to_dict(self) -> Dict[str, Any]:
        """Export DVH curve to dictionary."""
        return {
            "structure_name": self.structure_name,
            "dvh_type": self.dvh_type,
            "total_volume": self.total_volume,
            "structure_color": self.structure_color,
            "dose_bins": self.dose_bins,
            "volume_percent": self.volume_percent,
            "volume_absolute": self.volume_absolute,
            "statistics": {
                "mean_dose": self.mean_dose,
                "median_dose": self.median_dose,
                "modal_dose": self.modal_dose,
                "std_dose": self.std_dose,
                "min_dose": self.min_dose,
                "max_dose": self.max_dose,
            },
            "clinical_metrics": {
                "d95_percent": self.d95_percent,
                "d50_percent": self.d50_percent,
                "d5_percent": self.d5_percent,
                "d2_percent": self.d2_percent,
                "d98_percent": self.d98_percent,
                "v95_percent": self.v95_percent,
                "v107_percent": self.v107_percent,
                "v110_percent": self.v110_percent,
                "v20gy": self.v20gy,
                "v30gy": self.v30gy,
            },
            "quality_indices": {
                "homogeneity_index": self.homogeneity_index,
                "conformity_index": self.conformity_index,
                "gradient_index": self.gradient_index,
            },
        }


@dataclass
class DVHCalculationSettings:
    """Cài đặt cho tính toán DVH."""

    # Dose binning
    dose_bin_width: float = 0.1  # Gy
    max_dose: Optional[float] = None  # Gy, auto if None
    min_dose: float = 0.0  # Gy

    # Volume calculation
    volume_calculation_method: str = "VOXEL_BASED"  # VOXEL_BASED, CONTOUR_BASED
    interpolation_method: str = "LINEAR"  # LINEAR, CUBIC, NEAREST

    # Accuracy settings
    sub_voxel_sampling: bool = True
    sampling_factor: int = 3  # For sub-voxel sampling

    # Performance settings
    use_parallel_processing: bool = True
    max_workers: int = 4
    chunk_size: int = 1000

    # Quality assurance
    validate_inputs: bool = True
    check_dose_grid_alignment: bool = True

    def __post_init__(self):
        """Validate settings."""
        if self.dose_bin_width <= 0:
            raise ValueError("Dose bin width phải lớn hơn 0")
        if self.min_dose < 0:
            raise ValueError("Min dose không thể âm")
        if self.max_dose is not None and self.max_dose <= self.min_dose:
            raise ValueError("Max dose phải lớn hơn min dose")


class DVHCalculator:
    """
    Calculator chính cho DVH computation.
    """

    def __init__(self, settings: Optional[DVHCalculationSettings] = None):
        self.settings = settings or DVHCalculationSettings()

        # Calculation cache
        self._calculation_cache: Dict[str, DVHCurve] = {}
        self._cache_enabled = True

        # Performance monitoring
        self._calculation_times: List[float] = []
        self._last_calculation_time: Optional[float] = None

        logger.info("DVH Calculator khởi tạo")

    def calculate_dvh(
        self,
        dose_grid: np.ndarray,
        structure_mask: np.ndarray,
        structure_name: str,
        dose_spacing: Tuple[float, float, float] = (2.0, 2.0, 3.0),
        dvh_type: str = "CUMULATIVE",
    ) -> DVHCurve:
        """
        Tính toán DVH cho một cấu trúc.
        """
        start_time = time.time()

        try:
            # Generate cache key
            cache_key = self._generate_cache_key(
                dose_grid, structure_mask, structure_name, dose_spacing, dvh_type
            )

            # Check cache
            if self._cache_enabled and cache_key in self._calculation_cache:
                logger.info(f"DVH cache hit for {structure_name}")
                return self._calculation_cache[cache_key]

            # Validate inputs
            if self.settings.validate_inputs:
                self._validate_inputs(dose_grid, structure_mask, dose_spacing)

            # Calculate total volume
            total_volume = calculate_volume_from_mask(structure_mask, dose_spacing)

            # Extract dose values in structure
            structure_doses = self._extract_structure_doses(dose_grid, structure_mask)

            if len(structure_doses) == 0:
                logger.warning(f"Không có voxel nào trong structure {structure_name}")
                return self._create_empty_dvh(structure_name, total_volume)

            # Create dose bins
            dose_bins = self._create_dose_bins(structure_doses)

            # Calculate DVH curve
            if dvh_type == "CUMULATIVE":
                volume_percents, volume_absolutes = self._calculate_cumulative_dvh(
                    structure_doses, dose_bins, total_volume
                )
            else:  # DIFFERENTIAL
                volume_percents, volume_absolutes = self._calculate_differential_dvh(
                    structure_doses, dose_bins, total_volume
                )

            # Create DVH curve object
            dvh_curve = DVHCurve(
                structure_name=structure_name,
                dvh_type=dvh_type,
                dose_bins=dose_bins.tolist(),
                volume_percent=volume_percents.tolist(),
                volume_absolute=volume_absolutes.tolist(),
                total_volume=total_volume,
                structure_color=self._get_default_color(structure_name),
            )

            # Cache result
            if self._cache_enabled:
                self._calculation_cache[cache_key] = dvh_curve

            # Record calculation time
            calculation_time = time.time() - start_time
            self._calculation_times.append(calculation_time)
            self._last_calculation_time = calculation_time

            logger.info(
                f"DVH calculated for {structure_name} in {calculation_time:.3f}s"
            )
            return dvh_curve

        except Exception as e:
            logger.error(f"Lỗi calculate DVH: {e}")
            return self._create_empty_dvh(structure_name, 0.0)

    def calculate_multiple_dvh(
        self,
        dose_grid: np.ndarray,
        structure_masks: Dict[str, np.ndarray],
        dose_spacing: Tuple[float, float, float] = (2.0, 2.0, 3.0),
        progress_callback: Optional[Callable] = None,
    ) -> Dict[str, DVHCurve]:
        """
        Tính toán DVH cho nhiều cấu trúc.
        """
        try:
            results = {}
            total_structures = len(structure_masks)

            if self.settings.use_parallel_processing and total_structures > 1:
                # Parallel processing
                results = self._calculate_parallel_dvh(
                    dose_grid, structure_masks, dose_spacing, progress_callback
                )
            else:
                # Sequential processing
                for i, (structure_name, mask) in enumerate(structure_masks.items()):
                    if progress_callback:
                        progress = (i / total_structures) * 100
                        progress_callback(
                            progress, f"Calculating DVH: {structure_name}"
                        )

                    dvh_curve = self.calculate_dvh(
                        dose_grid, mask, structure_name, dose_spacing
                    )
                    results[structure_name] = dvh_curve

            if progress_callback:
                progress_callback(100, "DVH calculation completed")

            logger.info(f"Calculated DVH for {len(results)} structures")
            return results

        except Exception as e:
            logger.error(f"Lỗi calculate multiple DVH: {e}")
            return {}

    def _calculate_parallel_dvh(
        self,
        dose_grid: np.ndarray,
        structure_masks: Dict[str, np.ndarray],
        dose_spacing: Tuple[float, float, float],
        progress_callback: Optional[Callable] = None,
    ) -> Dict[str, DVHCurve]:
        """Tính toán DVH song song."""
        try:
            results = {}
            structure_items = list(structure_masks.items())

            def calculate_single_dvh(item):
                structure_name, mask = item
                return structure_name, self.calculate_dvh(
                    dose_grid, mask, structure_name, dose_spacing
                )

            with ThreadPoolExecutor(max_workers=self.settings.max_workers) as executor:
                # Submit all tasks
                futures = [
                    executor.submit(calculate_single_dvh, item)
                    for item in structure_items
                ]

                # Collect results with progress
                for i, future in enumerate(futures):
                    try:
                        structure_name, dvh_curve = future.result()
                        results[structure_name] = dvh_curve

                        if progress_callback:
                            progress = ((i + 1) / len(futures)) * 100
                            progress_callback(progress, f"Completed: {structure_name}")

                    except Exception as e:
                        logger.error(f"Lỗi parallel DVH calculation: {e}")

            return results

        except Exception as e:
            logger.error(f"Lỗi parallel DVH processing: {e}")
            return {}

    def _validate_inputs(
        self,
        dose_grid: np.ndarray,
        structure_mask: np.ndarray,
        dose_spacing: Tuple[float, float, float],
    ):
        """Validate input data."""
        if dose_grid.shape != structure_mask.shape:
            raise ValueError("Dose grid và structure mask phải có cùng shape")

        if len(dose_spacing) != 3:
            raise ValueError("Dose spacing phải có 3 giá trị")

        if any(s <= 0 for s in dose_spacing):
            raise ValueError("Dose spacing phải lớn hơn 0")

        if np.sum(structure_mask) == 0:
            logger.warning("Structure mask trống")

    def _extract_structure_doses(
        self, dose_grid: np.ndarray, structure_mask: np.ndarray
    ) -> np.ndarray:
        """Extract dose values trong structure."""
        try:
            # Simple extraction
            structure_doses = dose_grid[structure_mask > 0]

            # Remove invalid values
            structure_doses = structure_doses[~np.isnan(structure_doses)]
            structure_doses = structure_doses[~np.isinf(structure_doses)]
            structure_doses = structure_doses[structure_doses >= 0]

            return structure_doses

        except Exception as e:
            logger.error(f"Lỗi extract structure doses: {e}")
            return np.array([])

    def _create_dose_bins(self, structure_doses: np.ndarray) -> np.ndarray:
        """Tạo dose bins cho DVH calculation."""
        try:
            if len(structure_doses) == 0:
                return np.array([0.0])

            min_dose = max(self.settings.min_dose, 0.0)
            max_dose = self.settings.max_dose or float(np.max(structure_doses))

            # Ensure reasonable range
            if max_dose <= min_dose:
                max_dose = min_dose + 1.0

            # Create bins
            num_bins = int((max_dose - min_dose) / self.settings.dose_bin_width) + 1
            dose_bins = np.linspace(min_dose, max_dose, num_bins)

            return dose_bins

        except Exception as e:
            logger.error(f"Lỗi create dose bins: {e}")
            return np.linspace(0, 100, 1001)  # Default bins

    def _calculate_cumulative_dvh(
        self, structure_doses: np.ndarray, dose_bins: np.ndarray, total_volume: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Tính toán cumulative DVH."""
        try:
            volume_percents = np.zeros(len(dose_bins))

            for i, dose_level in enumerate(dose_bins):
                # Volume receiving >= dose_level
                volume_above = np.sum(structure_doses >= dose_level)
                volume_percent = (volume_above / len(structure_doses)) * 100.0
                volume_percents[i] = volume_percent

            # Convert to absolute volumes
            volume_absolutes = (volume_percents / 100.0) * total_volume

            return volume_percents, volume_absolutes

        except Exception as e:
            logger.error(f"Lỗi calculate cumulative DVH: {e}")
            return np.zeros(len(dose_bins)), np.zeros(len(dose_bins))

    def _calculate_differential_dvh(
        self, structure_doses: np.ndarray, dose_bins: np.ndarray, total_volume: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Tính toán differential DVH."""
        try:
            volume_percents = np.zeros(len(dose_bins) - 1)

            for i in range(len(dose_bins) - 1):
                dose_min = dose_bins[i]
                dose_max = dose_bins[i + 1]

                # Volume in dose range [dose_min, dose_max)
                volume_in_range = np.sum(
                    (structure_doses >= dose_min) & (structure_doses < dose_max)
                )
                volume_percent = (volume_in_range / len(structure_doses)) * 100.0
                volume_percents[i] = volume_percent

            # Convert to absolute volumes
            volume_absolutes = (volume_percents / 100.0) * total_volume

            # Use midpoint doses for differential bins
            dose_midpoints = (dose_bins[:-1] + dose_bins[1:]) / 2.0

            return volume_percents, volume_absolutes

        except Exception as e:
            logger.error(f"Lỗi calculate differential DVH: {e}")
            return np.zeros(len(dose_bins) - 1), np.zeros(len(dose_bins) - 1)

    def _create_empty_dvh(self, structure_name: str, total_volume: float) -> DVHCurve:
        """Tạo empty DVH curve."""
        return DVHCurve(
            structure_name=structure_name,
            dvh_type="CUMULATIVE",
            dose_bins=[0.0, 1.0],
            volume_percent=[0.0, 0.0],
            volume_absolute=[0.0, 0.0],
            total_volume=total_volume,
            structure_color=self._get_default_color(structure_name),
        )

    def _get_default_color(self, structure_name: str) -> Tuple[float, float, float]:
        """Lấy màu mặc định cho structure."""
        color_map = {
            "PTV": (1.0, 0.0, 0.0),  # Red
            "CTV": (1.0, 0.5, 0.0),  # Orange
            "GTV": (1.0, 0.0, 0.5),  # Pink
            "RECTUM": (0.0, 1.0, 0.0),  # Green
            "BLADDER": (0.0, 0.0, 1.0),  # Blue
            "SPINAL": (1.0, 1.0, 0.0),  # Yellow
            "LUNG": (0.0, 1.0, 1.0),  # Cyan
            "HEART": (1.0, 0.0, 1.0),  # Magenta
        }

        # Check for partial matches
        name_upper = structure_name.upper()
        for key, color in color_map.items():
            if key in name_upper:
                return color

        # Default color based on hash
        hash_value = hash(structure_name) % 7
        default_colors = [
            (0.8, 0.2, 0.2),  # Red-ish
            (0.2, 0.8, 0.2),  # Green-ish
            (0.2, 0.2, 0.8),  # Blue-ish
            (0.8, 0.8, 0.2),  # Yellow-ish
            (0.8, 0.2, 0.8),  # Magenta-ish
            (0.2, 0.8, 0.8),  # Cyan-ish
            (0.6, 0.4, 0.2),  # Brown-ish
        ]

        return default_colors[hash_value]

    def _generate_cache_key(
        self,
        dose_grid: np.ndarray,
        structure_mask: np.ndarray,
        structure_name: str,
        dose_spacing: Tuple[float, float, float],
        dvh_type: str,
    ) -> str:
        """Generate cache key cho DVH calculation."""
        try:
            # Create hash từ các input parameters
            dose_hash = hash(dose_grid.tobytes())
            mask_hash = hash(structure_mask.tobytes())

            cache_key = (
                f"{structure_name}_{dvh_type}_{dose_hash}_{mask_hash}_{dose_spacing}"
            )
            return cache_key

        except Exception:
            # Fallback simple key
            return f"{structure_name}_{dvh_type}_{time.time()}"

    def clear_cache(self):
        """Xóa calculation cache."""
        self._calculation_cache.clear()
        logger.info("DVH calculation cache cleared")

    def get_calculation_statistics(self) -> Dict[str, Any]:
        """Lấy thống kê calculation performance."""
        return {
            "total_calculations": len(self._calculation_times),
            "cache_size": len(self._calculation_cache),
            "last_calculation_time": self._last_calculation_time,
            "average_calculation_time": np.mean(self._calculation_times)
            if self._calculation_times
            else 0,
            "total_time": sum(self._calculation_times),
        }


class DVHAnalyzer:
    """
    Analyzer cho DVH comparison và analysis.
    """

    def __init__(self):
        self.dvh_curves: Dict[str, DVHCurve] = {}

        logger.info("DVH Analyzer khởi tạo")

    def add_dvh_curve(self, dvh_curve: DVHCurve) -> None:
        """Thêm DVH curve vào analyzer."""
        self.dvh_curves[dvh_curve.structure_name] = dvh_curve
        logger.info(f"Added DVH curve: {dvh_curve.structure_name}")

    def compare_dvh_curves(
        self, curve1_name: str, curve2_name: str
    ) -> Optional[Dict[str, Any]]:
        """So sánh hai DVH curves."""
        try:
            if curve1_name not in self.dvh_curves or curve2_name not in self.dvh_curves:
                logger.error("Không tìm thấy DVH curves để so sánh")
                return None

            curve1 = self.dvh_curves[curve1_name]
            curve2 = self.dvh_curves[curve2_name]

            comparison = {
                "curve1_name": curve1_name,
                "curve2_name": curve2_name,
                "statistics_comparison": self._compare_statistics(curve1, curve2),
                "clinical_metrics_comparison": self._compare_clinical_metrics(
                    curve1, curve2
                ),
                "volume_difference": abs(curve1.total_volume - curve2.total_volume),
                "similarity_score": self._calculate_similarity_score(curve1, curve2),
            }

            return comparison

        except Exception as e:
            logger.error(f"Lỗi compare DVH curves: {e}")
            return None

    def _compare_statistics(
        self, curve1: DVHCurve, curve2: DVHCurve
    ) -> Dict[str, float]:
        """So sánh statistics giữa hai curves."""
        stats_comparison = {}

        stats_fields = ["mean_dose", "median_dose", "std_dose", "min_dose", "max_dose"]

        for field in stats_fields:
            val1 = getattr(curve1, field)
            val2 = getattr(curve2, field)

            if val1 is not None and val2 is not None:
                difference = abs(val1 - val2)
                relative_diff = (difference / max(val1, 0.001)) * 100  # % difference
                stats_comparison[f"{field}_difference"] = difference
                stats_comparison[f"{field}_relative_diff_percent"] = relative_diff

        return stats_comparison

    def _compare_clinical_metrics(
        self, curve1: DVHCurve, curve2: DVHCurve
    ) -> Dict[str, float]:
        """So sánh clinical metrics giữa hai curves."""
        metrics_comparison = {}

        metric_fields = [
            "d95_percent",
            "d50_percent",
            "d5_percent",
            "d2_percent",
            "d98_percent",
            "v95_percent",
            "v107_percent",
            "v110_percent",
        ]

        for field in metric_fields:
            val1 = getattr(curve1, field)
            val2 = getattr(curve2, field)

            if val1 is not None and val2 is not None:
                difference = abs(val1 - val2)
                relative_diff = (difference / max(val1, 0.001)) * 100
                metrics_comparison[f"{field}_difference"] = difference
                metrics_comparison[f"{field}_relative_diff_percent"] = relative_diff

        return metrics_comparison

    def _calculate_similarity_score(self, curve1: DVHCurve, curve2: DVHCurve) -> float:
        """Tính toán similarity score giữa hai DVH curves."""
        try:
            if (
                not curve1.dose_bins
                or not curve2.dose_bins
                or not curve1.volume_percent
                or not curve2.volume_percent
            ):
                return 0.0

            # Interpolate both curves to common dose grid
            max_dose = max(max(curve1.dose_bins), max(curve2.dose_bins))
            common_doses = np.linspace(0, max_dose, 100)

            # Get volumes at common doses
            vol1 = []
            vol2 = []

            for dose in common_doses:
                v1 = curve1.get_volume_at_dose(dose)
                v2 = curve2.get_volume_at_dose(dose)

                if v1 is not None and v2 is not None:
                    vol1.append(v1)
                    vol2.append(v2)
                else:
                    vol1.append(0)
                    vol2.append(0)

            if len(vol1) > 0:
                # Calculate correlation coefficient
                if HAS_SCIPY:
                    from scipy.stats import pearsonr

                    correlation, _ = pearsonr(vol1, vol2)
                    return max(0, correlation)  # Return 0 if negative correlation
                else:
                    # Simple similarity based on mean absolute difference
                    vol1_array = np.array(vol1)
                    vol2_array = np.array(vol2)
                    mad = np.mean(np.abs(vol1_array - vol2_array))
                    # Convert to similarity score (0-1)
                    max_vol = max(np.max(vol1_array), np.max(vol2_array))
                    if max_vol > 0:
                        similarity = 1.0 - (mad / max_vol)
                        return max(0, similarity)

            return 0.0

        except Exception as e:
            logger.error(f"Lỗi calculate similarity score: {e}")
            return 0.0

    def get_structure_ranking(
        self, metric: str = "mean_dose"
    ) -> List[Tuple[str, float]]:
        """Xếp hạng structures theo metric cụ thể."""
        try:
            rankings = []

            for name, curve in self.dvh_curves.items():
                value = getattr(curve, metric, None)
                if value is not None:
                    rankings.append((name, value))

            # Sort by metric value (descending)
            rankings.sort(key=lambda x: x[1], reverse=True)

            return rankings

        except Exception as e:
            logger.error(f"Lỗi get structure ranking: {e}")
            return []

    def export_analysis_report(self, output_path: str) -> bool:
        """Export analysis report to JSON."""
        try:
            report = {
                "analysis_date": datetime.now().isoformat(),
                "total_structures": len(self.dvh_curves),
                "structures": {},
                "summary_statistics": self._calculate_summary_statistics(),
            }

            # Export each DVH curve
            for name, curve in self.dvh_curves.items():
                report["structures"][name] = curve.export_to_dict()

            # Write to file
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False, default=str)

            logger.info(f"DVH analysis report exported to: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Lỗi export analysis report: {e}")
            return False

    def _calculate_summary_statistics(self) -> Dict[str, Any]:
        """Tính toán summary statistics cho tất cả curves."""
        try:
            if not self.dvh_curves:
                return {}

            # Collect all statistics
            mean_doses = [
                c.mean_dose for c in self.dvh_curves.values() if c.mean_dose is not None
            ]
            max_doses = [
                c.max_dose for c in self.dvh_curves.values() if c.max_dose is not None
            ]
            total_volumes = [c.total_volume for c in self.dvh_curves.values()]

            summary = {
                "structures_count": len(self.dvh_curves),
                "total_volume_all_structures": sum(total_volumes),
                "mean_dose_range": {
                    "min": min(mean_doses) if mean_doses else 0,
                    "max": max(mean_doses) if mean_doses else 0,
                    "average": np.mean(mean_doses) if mean_doses else 0,
                },
                "max_dose_range": {
                    "min": min(max_doses) if max_doses else 0,
                    "max": max(max_doses) if max_doses else 0,
                    "average": np.mean(max_doses) if max_doses else 0,
                },
            }

            return summary

        except Exception as e:
            logger.error(f"Lỗi calculate summary statistics: {e}")
            return {}


# Factory functions
def create_dvh_calculator(
    settings: Optional[DVHCalculationSettings] = None,
) -> DVHCalculator:
    """Factory function để tạo DVH Calculator."""
    return DVHCalculator(settings)


def create_dvh_analyzer() -> DVHAnalyzer:
    """Factory function để tạo DVH Analyzer."""
    return DVHAnalyzer()


def create_sample_dvh_curve(structure_name: str = "Sample") -> DVHCurve:
    """Tạo sample DVH curve để test."""
    # Create mock dose-volume data
    dose_bins = list(np.linspace(0, 80, 81))  # 0-80 Gy, 1 Gy bins

    # Create realistic DVH shape (decreasing cumulative)
    volume_percent = []
    for dose in dose_bins:
        if dose <= 2:
            vol = 100.0  # Full volume at low dose
        elif dose <= 50:
            # Linear decrease from 100% to 95%
            vol = 100 - (dose - 2) * (5 / 48)
        elif dose <= 78:
            # Steeper decrease for prescription dose region
            vol = 95 - (dose - 50) * (90 / 28)
        else:
            # Minimal volume at high dose
            vol = max(0, 5 - (dose - 78) * 2.5)

        volume_percent.append(vol)

    # Calculate absolute volumes (assume 100 cc total)
    total_volume = 100.0
    volume_absolute = [(v / 100.0) * total_volume for v in volume_percent]

    return DVHCurve(
        structure_name=structure_name,
        dvh_type="CUMULATIVE",
        dose_bins=dose_bins,
        volume_percent=volume_percent,
        volume_absolute=volume_absolute,
        total_volume=total_volume,
        structure_color=(1.0, 0.0, 0.0),  # Red
    )


if __name__ == "__main__":
    # Test code
    logging.basicConfig(level=logging.INFO)

    # Test DVH calculation
    calculator = create_dvh_calculator()

    # Create sample data
    dose_grid = np.random.rand(32, 32, 16) * 60  # Random dose distribution
    structure_mask = np.zeros((32, 32, 16), dtype=bool)
    structure_mask[10:22, 10:22, 6:10] = True  # Small region

    # Calculate DVH
    dvh_curve = calculator.calculate_dvh(dose_grid, structure_mask, "Test Structure")

    print(f"DVH calculated: {dvh_curve.structure_name}")
    print(f"Mean dose: {dvh_curve.mean_dose:.2f} Gy")
    print(f"D95%: {dvh_curve.d95_percent:.2f} Gy")
    print(f"Total volume: {dvh_curve.total_volume:.2f} cc")

    # Test analyzer
    analyzer = create_dvh_analyzer()
    analyzer.add_dvh_curve(dvh_curve)

    # Test sample DVH
    sample_dvh = create_sample_dvh_curve("PTV")
    analyzer.add_dvh_curve(sample_dvh)

    print(f"Analyzer has {len(analyzer.dvh_curves)} DVH curves")

    print("DVH Engine test hoàn thành!")
