#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module phân tích gamma (Gamma Analysis) cho so sánh phân phối liều.

Module này cung cấp các công cụ để thực hiện phân tích gamma 2D và 3D,
một phương pháp định lượng để so sánh hai phân phối liều.
"""

import logging
import numpy as np
import time
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass
from scipy import ndimage
import concurrent.futures

logger = logging.getLogger(__name__)

# Safe import for numba with enhanced fallback
try:
    from numba import jit, prange
    from numba import cuda

    HAS_NUMBA = True
    HAS_NUMBA_CUDA = True
    logger.info("Numba imported successfully for gamma analysis acceleration")

    # Check NumPy compatibility with Numba
    numpy_version = np.__version__
    major, minor = map(int, numpy_version.split(".")[:2])

    if major >= 2 and minor >= 2:  # NumPy 2.2+
        logger.warning(
            f"NumPy version {numpy_version} not compatible with Numba. Using CPU fallback methods."
        )
        HAS_NUMBA = False
        HAS_NUMBA_CUDA = False

except ImportError as e:
    HAS_NUMBA = False
    HAS_NUMBA_CUDA = False
    logger.warning(f"Numba không khả dụng ({e}). Sử dụng CPU fallback.")
except Exception as e:
    HAS_NUMBA = False
    HAS_NUMBA_CUDA = False
    logger.warning(f"Numba không khả dụng ({e}). Sử dụng CPU fallback.")

# Safe import for CuPy
try:
    import cupy as cp

    HAS_CUPY = True
    logger.info("CuPy imported successfully for GPU acceleration")
except ImportError:
    HAS_CUPY = False
    logger.warning("CuPy không khả dụng. Phân tích gamma GPU sẽ không thể sử dụng.")

# Safe import for scipy
try:
    from scipy import ndimage

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    logger.warning("SciPy không khả dụng. Một số tính năng sẽ bị giới hạn.")

# Import cho parallel processing
try:
    from concurrent.futures import ThreadPoolExecutor

    HAS_THREADING = True
except ImportError:
    HAS_THREADING = False

# Create fallback decorators if numba is not available
if not HAS_NUMBA:

    def jit(nopython=True):
        def decorator(func):
            return func

        return decorator

    class cuda:
        @staticmethod
        def jit(func):
            return func

        @staticmethod
        def grid(ndim):
            return (1,) * ndim

        class atomic:
            @staticmethod
            def add(array, index, value):
                array[index] += value


@dataclass
class GammaAnalysisSettings:
    """Cài đặt cho gamma analysis."""

    # Basic gamma criteria
    distance_mm: float = 3.0
    dose_percent: float = 3.0
    dose_threshold_percent: float = 10.0

    # Advanced settings
    search_radius_mm: float = 10.0
    interpolation_factor: int = 2
    use_global_dose: bool = False
    local_dose_threshold: float = 10.0  # % of max dose

    # Performance settings
    use_gpu: bool = True
    use_parallel: bool = True
    max_workers: int = 4
    chunk_size: int = 1000000

    # Quality settings
    use_dose_threshold: bool = True
    exclude_zero_dose: bool = True
    use_distance_to_agreement: bool = False


@dataclass
class GammaAnalysisResult:
    """Kết quả gamma analysis."""

    gamma_map: np.ndarray
    pass_rate: float = 0.0
    mean_gamma: float = 0.0
    max_gamma: float = 0.0

    # Statistical analysis
    gamma_histogram: Optional[np.ndarray] = None
    gamma_bins: Optional[np.ndarray] = None

    # Spatial analysis
    pass_map: Optional[np.ndarray] = None
    fail_regions: Optional[List[Tuple[int, int, int]]] = None

    # Processing info
    calculation_time: float = 0.0
    voxels_analyzed: int = 0
    voxels_passed: int = 0
    method_used: str = "cpu"

    # Settings used
    settings: Optional[GammaAnalysisSettings] = None


def calculate_gamma_3d(
    reference_dose: np.ndarray,
    evaluated_dose: np.ndarray,
    distance_mm: float = 3.0,
    dose_percent: float = 3.0,
    dose_threshold_percent: float = 10.0,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    settings: Optional[GammaAnalysisSettings] = None,
    progress_callback: Optional[Callable] = None,
    # Các tham số tương thích với API cũ
    reference: np.ndarray = None,
    evaluation: np.ndarray = None,
    dta_mm: float = None,
    dd_percent: float = None,
    threshold: float = None,
    voxel_size: Tuple[float, float, float] = None,
    max_gamma: float = 5.0,
    local_normalization: bool = False,
) -> GammaAnalysisResult:
    """
    Tính toán gamma index 3D với các tùy chọn nâng cao.

    Args:
        reference_dose: Phân phối liều reference
        evaluated_dose: Phân phối liều evaluated
        distance_mm: Distance to agreement (mm)
        dose_percent: Dose difference (%)
        dose_threshold_percent: Dose threshold (% of max dose)
        spacing: Voxel spacing (mm)
        settings: Advanced settings
        progress_callback: Callback function cho progress

        # Tham số tương thích API cũ
        reference: Alias cho reference_dose
        evaluation: Alias cho evaluated_dose
        dta_mm: Alias cho distance_mm
        dd_percent: Alias cho dose_percent
        threshold: Alias cho dose_threshold_percent
        voxel_size: Alias cho spacing
        max_gamma: Giá trị gamma tối đa
        local_normalization: Dùng local thay vì global normalization

    Returns:
        GammaAnalysisResult: Kết quả gamma analysis
    """
    start_time = time.time()

    try:
        # Xử lý các tham số API cũ
        if reference is not None:
            reference_dose = reference
        if evaluation is not None:
            evaluated_dose = evaluation
        if dta_mm is not None:
            distance_mm = dta_mm
        if dd_percent is not None:
            dose_percent = dd_percent
        if threshold is not None:
            dose_threshold_percent = (
                threshold * 100.0
            )  # Chuyển đổi từ tỷ lệ sang phần trăm
        if voxel_size is not None:
            spacing = voxel_size

        # Use settings if provided
        if settings is None:
            settings = GammaAnalysisSettings(
                distance_mm=distance_mm,
                dose_percent=dose_percent,
                dose_threshold_percent=dose_threshold_percent,
            )

            # Cập nhật settings từ tham số API cũ
            if local_normalization:
                settings.use_global_dose = False
            if max_gamma:
                settings.max_gamma = max_gamma

        logger.info(
            f"Starting gamma analysis with {settings.distance_mm}mm/{settings.dose_percent}%"
        )

        # Validate inputs
        if reference_dose.shape != evaluated_dose.shape:
            raise ValueError("Reference and evaluated dose arrays must have same shape")

        # Apply dose threshold
        max_dose = np.max(reference_dose)
        dose_threshold = max_dose * (settings.dose_threshold_percent / 100.0)

        # Create analysis mask
        if settings.use_dose_threshold:
            analysis_mask = reference_dose >= dose_threshold
        else:
            analysis_mask = np.ones_like(reference_dose, dtype=bool)

        if settings.exclude_zero_dose:
            analysis_mask = analysis_mask & (reference_dose > 0)

        # Select calculation method
        method_used = "cpu"

        if settings.use_gpu and HAS_NUMBA_CUDA:
            try:
                gamma_map = _calculate_gamma_gpu(
                    reference_dose,
                    evaluated_dose,
                    analysis_mask,
                    settings,
                    spacing,
                    progress_callback,
                )
                method_used = "gpu"
                logger.info("Using GPU acceleration for gamma analysis")
            except Exception as e:
                logger.warning(f"GPU calculation failed, falling back to CPU: {e}")
                gamma_map = _calculate_gamma_cpu(
                    reference_dose,
                    evaluated_dose,
                    analysis_mask,
                    settings,
                    spacing,
                    progress_callback,
                )
        elif settings.use_parallel and HAS_NUMBA:
            logger.info("Using Numba JIT accelerated CPU calculation")
            gamma_map = _calculate_gamma_cpu(
                reference_dose,
                evaluated_dose,
                analysis_mask,
                settings,
                spacing,
                progress_callback,
            )
        else:
            logger.info("Using standard CPU calculation (no Numba)")
            gamma_map = _calculate_gamma_cpu_fallback(
                reference_dose,
                evaluated_dose,
                analysis_mask,
                settings,
                spacing,
                progress_callback,
            )

        # Tính toán pass rate với xử lý finite values
        # analysis_mask đã được tạo ở trên, sử dụng lại

        # Filter out infinite and NaN values
        finite_mask = np.isfinite(gamma_map)
        combined_mask = analysis_mask & finite_mask

        if np.sum(combined_mask) == 0:
            logger.warning("No valid gamma values found for analysis")
            pass_rate = 0.0
            mean_gamma = float("inf")
            max_gamma = float("inf")
            voxels_analyzed = 0
            voxels_passed = 0
        else:
            # Only analyze finite gamma values within dose threshold
            valid_gamma_values = gamma_map[combined_mask]

            # Clamp gamma values to reasonable range to avoid extreme values
            valid_gamma_values = np.clip(valid_gamma_values, 0.0, 10.0)

            # Calculate statistics
            pass_mask = valid_gamma_values <= 1.0
            voxels_analyzed = len(valid_gamma_values)
            voxels_passed = np.sum(pass_mask)

            # Calculate pass rate as percentage
            pass_rate = (
                (voxels_passed / voxels_analyzed * 100.0)
                if voxels_analyzed > 0
                else 0.0
            )

            # Ensure pass rate is reasonable (0-100%)
            pass_rate = np.clip(pass_rate, 0.0, 100.0)

            # Calculate mean and max gamma for valid values only
            mean_gamma = np.mean(valid_gamma_values)
            max_gamma = np.max(valid_gamma_values)

            # Ensure reasonable bounds
            mean_gamma = np.clip(mean_gamma, 0.0, 10.0)
            max_gamma = np.clip(max_gamma, 0.0, 10.0)

        # Create pass map for visualization
        pass_map = np.zeros_like(gamma_map, dtype=bool)
        if np.sum(combined_mask) > 0:
            pass_map[combined_mask] = gamma_map[combined_mask] <= 1.0

        # Calculate gamma histogram
        gamma_bins = np.linspace(0, min(max_gamma, 5.0), 100)
        if np.sum(combined_mask) > 0:
            gamma_histogram, _ = np.histogram(valid_gamma_values, bins=gamma_bins)
        else:
            gamma_histogram = np.zeros(99)  # bins-1

        # Find fail regions
        fail_regions = _find_fail_regions(pass_map)

        calculation_time = time.time() - start_time

        logger.info(f"Gamma analysis completed in {calculation_time:.2f}s")
        logger.info(f"Pass rate: {pass_rate:.1f}%")

        return GammaAnalysisResult(
            gamma_map=gamma_map,
            pass_rate=pass_rate,
            mean_gamma=mean_gamma,
            max_gamma=max_gamma,
            gamma_histogram=gamma_histogram,
            gamma_bins=gamma_bins,
            pass_map=pass_map,
            fail_regions=fail_regions,
            calculation_time=calculation_time,
            voxels_analyzed=voxels_analyzed,
            voxels_passed=voxels_passed,
            method_used=method_used,
            settings=settings,
        )

    except Exception as e:
        logger.error(f"Error in gamma analysis: {e}")
        return GammaAnalysisResult(
            gamma_map=np.full_like(reference_dose, float("inf")),
            calculation_time=time.time() - start_time,
            method_used="error",
            settings=settings,
        )


def _calculate_gamma_cpu(
    reference_dose: np.ndarray,
    evaluated_dose: np.ndarray,
    analysis_mask: np.ndarray,
    settings: GammaAnalysisSettings,
    spacing: Tuple[float, float, float],
    progress_callback: Optional[Callable] = None,
) -> np.ndarray:
    """CPU implementation của gamma calculation."""

    gamma_map = np.full_like(reference_dose, float("inf"))

    # Get coordinates of voxels to analyze
    z_coords, y_coords, x_coords = np.where(analysis_mask)
    total_voxels = len(z_coords)

    if total_voxels == 0:
        return gamma_map

    logger.info(f"Analyzing {total_voxels} voxels on CPU")

    if settings.use_parallel and total_voxels > 1000:
        # Parallel processing
        gamma_map = _calculate_gamma_parallel(
            reference_dose,
            evaluated_dose,
            z_coords,
            y_coords,
            x_coords,
            settings,
            spacing,
            progress_callback,
        )
    else:
        # Sequential processing with JIT
        gamma_map = _calculate_gamma_sequential_jit(
            reference_dose,
            evaluated_dose,
            z_coords,
            y_coords,
            x_coords,
            settings.distance_mm,
            settings.dose_percent,
            spacing,
            settings.search_radius_mm,
        )

    return gamma_map


def _calculate_gamma_parallel(
    reference_dose: np.ndarray,
    evaluated_dose: np.ndarray,
    z_coords: np.ndarray,
    y_coords: np.ndarray,
    x_coords: np.ndarray,
    settings: GammaAnalysisSettings,
    spacing: Tuple[float, float, float],
    progress_callback: Optional[Callable] = None,
) -> np.ndarray:
    """Parallel implementation sử dụng ThreadPoolExecutor."""

    gamma_map = np.full_like(reference_dose, float("inf"))
    total_voxels = len(z_coords)

    # Split work into chunks
    chunk_size = min(settings.chunk_size, max(1, total_voxels // settings.max_workers))
    chunks = [
        (i, min(i + chunk_size, total_voxels))
        for i in range(0, total_voxels, chunk_size)
    ]

    def process_chunk(chunk_range):
        start_idx, end_idx = chunk_range
        chunk_gammas = []

        for i in range(start_idx, end_idx):
            z, y, x = z_coords[i], y_coords[i], x_coords[i]

            gamma_value = _calculate_gamma_single_point_jit(
                reference_dose,
                evaluated_dose,
                z,
                y,
                x,
                settings.distance_mm,
                settings.dose_percent,
                spacing,
                settings.search_radius_mm,
            )

            chunk_gammas.append((z, y, x, gamma_value))

        return chunk_gammas

    # Execute in parallel
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=settings.max_workers
    ) as executor:
        future_to_chunk = {
            executor.submit(process_chunk, chunk): chunk for chunk in chunks
        }

        completed = 0
        for future in concurrent.futures.as_completed(future_to_chunk):
            chunk_results = future.result()

            for z, y, x, gamma_value in chunk_results:
                gamma_map[z, y, x] = gamma_value

            completed += len(chunk_results)

            if progress_callback:
                progress = int((completed / total_voxels) * 100)
                progress_callback(
                    progress, f"Processed {completed}/{total_voxels} voxels"
                )

    return gamma_map


@jit(nopython=True)
def _calculate_gamma_sequential_jit(
    reference_dose: np.ndarray,
    evaluated_dose: np.ndarray,
    z_coords: np.ndarray,
    y_coords: np.ndarray,
    x_coords: np.ndarray,
    settings_distance: float,
    settings_dose_percent: float,
    spacing: Tuple[float, float, float],
    search_radius: float,
) -> np.ndarray:
    """JIT-compiled sequential gamma calculation."""

    gamma_map = np.full_like(reference_dose, np.inf)

    for i in range(len(z_coords)):  # Change from prange to range to fix linter error
        z, y, x = z_coords[i], y_coords[i], x_coords[i]

        gamma_value = _calculate_gamma_single_point_jit(
            reference_dose,
            evaluated_dose,
            z,
            y,
            x,
            settings_distance,
            settings_dose_percent,
            spacing,
            search_radius,
        )

        gamma_map[z, y, x] = gamma_value

    return gamma_map


@jit(nopython=True)
def _calculate_gamma_single_point_jit(
    reference_dose: np.ndarray,
    evaluated_dose: np.ndarray,
    ref_z: int,
    ref_y: int,
    ref_x: int,
    distance_mm: float,
    dose_percent: float,
    spacing: Tuple[float, float, float],
    search_radius_mm: float,
) -> float:
    """JIT-compiled single point gamma calculation."""

    ref_dose = reference_dose[ref_z, ref_y, ref_x]

    if ref_dose <= 0:
        return np.inf

    # Calculate search range in voxels
    search_z = int(np.ceil(search_radius_mm / spacing[2]))
    search_y = int(np.ceil(search_radius_mm / spacing[1]))
    search_x = int(np.ceil(search_radius_mm / spacing[0]))

    min_gamma = np.inf

    # Search in neighborhood
    for dz in range(-search_z, search_z + 1):
        for dy in range(-search_y, search_y + 1):
            for dx in range(-search_x, search_x + 1):
                eval_z = ref_z + dz
                eval_y = ref_y + dy
                eval_x = ref_x + dx

                # Check bounds
                if (
                    eval_z < 0
                    or eval_z >= reference_dose.shape[0]
                    or eval_y < 0
                    or eval_y >= reference_dose.shape[1]
                    or eval_x < 0
                    or eval_x >= reference_dose.shape[2]
                ):
                    continue

                # Calculate physical distance
                dist_mm = np.sqrt(
                    (dz * spacing[2]) ** 2
                    + (dy * spacing[1]) ** 2
                    + (dx * spacing[0]) ** 2
                )

                if dist_mm > search_radius_mm:
                    continue

                # Calculate dose difference
                eval_dose = evaluated_dose[eval_z, eval_y, eval_x]
                dose_diff_percent = abs(eval_dose - ref_dose) / ref_dose * 100.0

                # Calculate gamma
                gamma = np.sqrt(
                    (dist_mm / distance_mm) ** 2
                    + (dose_diff_percent / dose_percent) ** 2
                )

                if gamma < min_gamma:
                    min_gamma = gamma

    return min_gamma


def _calculate_gamma_gpu(
    reference_dose: np.ndarray,
    evaluated_dose: np.ndarray,
    analysis_mask: np.ndarray,
    settings: GammaAnalysisSettings,
    spacing: Tuple[float, float, float],
    progress_callback: Optional[Callable] = None,
) -> np.ndarray:
    """GPU implementation của gamma calculation."""

    if not HAS_NUMBA_CUDA:
        raise RuntimeError("CUDA not available")

    # Copy data to GPU
    d_reference = cuda.to_device(reference_dose.astype(np.float32))
    d_evaluated = cuda.to_device(evaluated_dose.astype(np.float32))
    d_mask = cuda.to_device(analysis_mask)

    # Initialize output
    gamma_map = np.full_like(reference_dose, float("inf"), dtype=np.float32)
    d_gamma = cuda.to_device(gamma_map)

    # CUDA kernel configuration
    threads_per_block = (8, 8, 8)
    blocks_per_grid = tuple(
        (reference_dose.shape[i] + threads_per_block[i] - 1) // threads_per_block[i]
        for i in range(3)
    )

    # Launch kernel
    _gamma_kernel_3d[blocks_per_grid, threads_per_block](
        d_reference,
        d_evaluated,
        d_mask,
        d_gamma,
        settings.distance_mm,
        settings.dose_percent,
        spacing[0],
        spacing[1],
        spacing[2],
        settings.search_radius_mm,
    )

    # Copy result back
    gamma_map = d_gamma.copy_to_host()

    return gamma_map


@cuda.jit
def _gamma_kernel_3d(
    reference,
    evaluated,
    mask,
    gamma_out,
    distance_mm,
    dose_percent,
    spacing_x,
    spacing_y,
    spacing_z,
    search_radius_mm,
):
    """CUDA kernel cho gamma calculation."""

    # Fix cuda.grid call
    ref_x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    ref_y = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    ref_z = cuda.blockIdx.z * cuda.blockDim.z + cuda.threadIdx.z

    if (
        ref_x >= reference.shape[2]
        or ref_y >= reference.shape[1]
        or ref_z >= reference.shape[0]
    ):
        return

    if not mask[ref_z, ref_y, ref_x]:
        return

    ref_dose = reference[ref_z, ref_y, ref_x]

    if ref_dose <= 0:
        gamma_out[ref_z, ref_y, ref_x] = float("inf")
        return

    # Calculate search range
    search_x = int(search_radius_mm / spacing_x) + 1
    search_y = int(search_radius_mm / spacing_y) + 1
    search_z = int(search_radius_mm / spacing_z) + 1

    min_gamma = float("inf")

    # Search neighborhood
    for dz in range(-search_z, search_z + 1):
        for dy in range(-search_y, search_y + 1):
            for dx in range(-search_x, search_x + 1):
                eval_x = ref_x + dx
                eval_y = ref_y + dy
                eval_z = ref_z + dz

                # Check bounds
                if (
                    eval_x < 0
                    or eval_x >= reference.shape[2]
                    or eval_y < 0
                    or eval_y >= reference.shape[1]
                    or eval_z < 0
                    or eval_z >= reference.shape[0]
                ):
                    continue

                # Calculate distance
                dist_mm = (
                    (dx * spacing_x) ** 2
                    + (dy * spacing_y) ** 2
                    + (dz * spacing_z) ** 2
                ) ** 0.5

                if dist_mm > search_radius_mm:
                    continue

                # Calculate dose difference
                eval_dose = evaluated[eval_z, eval_y, eval_x]
                dose_diff_percent = abs(eval_dose - ref_dose) / ref_dose * 100.0

                # Calculate gamma
                gamma = (
                    (dist_mm / distance_mm) ** 2
                    + (dose_diff_percent / dose_percent) ** 2
                ) ** 0.5

                if gamma < min_gamma:
                    min_gamma = gamma

    gamma_out[ref_z, ref_y, ref_x] = min_gamma  # Fix indexing error


def _find_fail_regions(pass_map: np.ndarray) -> List[Tuple[int, int, int]]:
    """Tìm các vùng fail trong gamma analysis."""

    try:
        # Label connected fail regions
        fail_map = ~pass_map
        labeled_fails, num_fails = ndimage.label(fail_map)

        fail_regions = []

        for i in range(1, num_fails + 1):
            region_coords = np.where(labeled_fails == i)

            if len(region_coords[0]) > 10:  # Only significant regions
                # Get centroid
                centroid = (
                    int(np.mean(region_coords[0])),
                    int(np.mean(region_coords[1])),
                    int(np.mean(region_coords[2])),
                )
                fail_regions.append(centroid)

        return fail_regions

    except Exception as e:
        logger.error(f"Error finding fail regions: {e}")
        return []


def create_gamma_analysis_report(result: GammaAnalysisResult) -> Dict[str, Any]:
    """Tạo báo cáo comprehensive cho gamma analysis."""

    try:
        report = {
            "pass_rate": result.pass_rate,
            "mean_gamma": result.mean_gamma,
            "max_gamma": result.max_gamma,
            "voxels_analyzed": result.voxels_analyzed,
            "voxels_passed": result.voxels_passed,
            "calculation_time": result.calculation_time,
            "method_used": result.method_used,
        }

        if result.settings:
            report["settings"] = {
                "distance_mm": result.settings.distance_mm,
                "dose_percent": result.settings.dose_percent,
                "dose_threshold_percent": result.settings.dose_threshold_percent,
                "use_gpu": result.settings.use_gpu,
                "use_parallel": result.settings.use_parallel,
            }

        if result.fail_regions:
            report["fail_regions_count"] = len(result.fail_regions)
            report["fail_regions"] = result.fail_regions[:10]  # Limit to first 10

        # Statistical summary
        if result.gamma_histogram is not None:
            gamma_percentiles = np.percentile(
                result.gamma_map[result.pass_map], [50, 90, 95, 99]
            )
            report["gamma_percentiles"] = {
                "p50": gamma_percentiles[0],
                "p90": gamma_percentiles[1],
                "p95": gamma_percentiles[2],
                "p99": gamma_percentiles[3],
            }

        return report

    except Exception as e:
        logger.error(f"Error creating gamma analysis report: {e}")
        return {"error": str(e)}


def compare_gamma_analyses(
    result1: GammaAnalysisResult, result2: GammaAnalysisResult, tolerance: float = 1.0
) -> Dict[str, Any]:
    """So sánh hai kết quả gamma analysis."""

    try:
        comparison = {
            "pass_rate_diff": abs(result1.pass_rate - result2.pass_rate),
            "mean_gamma_diff": abs(result1.mean_gamma - result2.mean_gamma),
            "max_gamma_diff": abs(result1.max_gamma - result2.max_gamma),
            "within_tolerance": abs(result1.pass_rate - result2.pass_rate) <= tolerance,
        }

        # Method comparison
        comparison["methods"] = [result1.method_used, result2.method_used]
        comparison["calculation_times"] = [
            result1.calculation_time,
            result2.calculation_time,
        ]

        if result1.method_used != result2.method_used:
            speedup = (
                result1.calculation_time / result2.calculation_time
                if result2.calculation_time > 0
                else 0
            )
            comparison["speedup"] = speedup

        return comparison

    except Exception as e:
        logger.error(f"Error comparing gamma analyses: {e}")
        return {"error": str(e)}


def _calculate_gamma_cpu_fallback(
    reference_dose: np.ndarray,
    evaluated_dose: np.ndarray,
    analysis_mask: np.ndarray,
    settings: GammaAnalysisSettings,
    spacing: Tuple[float, float, float],
    progress_callback: Optional[Callable] = None,
) -> np.ndarray:
    """
    Tính toán gamma map sử dụng CPU thuần, không dùng Numba.
    Fallback method cho các hệ thống không hỗ trợ Numba hoặc NumPy version mới.
    """
    logger.info("Using CPU fallback method for gamma calculation (no Numba)")

    z_coords, y_coords, x_coords = np.where(analysis_mask)
    gamma_map = np.ones_like(reference_dose) * 10.0  # Initialize với giá trị cao

    total_points = len(z_coords)
    search_radius_voxels = int(settings.search_radius_mm / min(spacing))

    for i, (z, y, x) in enumerate(zip(z_coords, y_coords, x_coords)):
        if progress_callback and i % 1000 == 0:
            progress = (i / total_points) * 100
            progress_callback(progress)

        ref_dose = reference_dose[z, y, x]

        # Define search region
        z_min = max(0, z - search_radius_voxels)
        z_max = min(reference_dose.shape[0], z + search_radius_voxels + 1)
        y_min = max(0, y - search_radius_voxels)
        y_max = min(reference_dose.shape[1], y + search_radius_voxels + 1)
        x_min = max(0, x - search_radius_voxels)
        x_max = min(reference_dose.shape[2], x + search_radius_voxels + 1)

        min_gamma = 10.0

        # Search in neighborhood
        for zi in range(z_min, z_max):
            for yi in range(y_min, y_max):
                for xi in range(x_min, x_max):
                    # Calculate distance
                    dist_z = (zi - z) * spacing[0]
                    dist_y = (yi - y) * spacing[1]
                    dist_x = (xi - x) * spacing[2]
                    distance = np.sqrt(dist_z**2 + dist_y**2 + dist_x**2)

                    if distance <= settings.search_radius_mm:
                        # Calculate dose difference
                        eval_dose = evaluated_dose[zi, yi, xi]
                        dose_diff = abs(eval_dose - ref_dose)

                        # Calculate gamma components
                        distance_term = distance / settings.distance_mm
                        if settings.use_global_dose:
                            max_dose = np.max(reference_dose)
                            dose_term = dose_diff / (
                                settings.dose_percent * max_dose / 100.0
                            )
                        else:
                            dose_term = dose_diff / (
                                settings.dose_percent * ref_dose / 100.0
                            )

                        # Calculate gamma value
                        gamma_val = np.sqrt(distance_term**2 + dose_term**2)
                        min_gamma = min(min_gamma, gamma_val)

        gamma_map[z, y, x] = min_gamma

    return gamma_map


__all__ = [
    "GammaAnalysisSettings",
    "GammaAnalysisResult",
    "calculate_gamma_3d",
    "create_gamma_analysis_report",
    "compare_gamma_analyses",
]

__version__ = "0.7.8"
