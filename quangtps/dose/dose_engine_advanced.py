"""
QuangTPS Advanced Dose Engine

Module tính toán liều xạ trị nâng cao cho hệ thống QuangTPS.
Cung cấp nhiều thuật toán tính liều từ cơ bản đến chuyên nghiệp,
tối ưu hóa hiệu suất và tích hợp với workflow lập kế hoạch.
"""

import logging
import os
import json
import numpy as np
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field, asdict
from pathlib import Path
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

logger = logging.getLogger(__name__)

# Import scientific libraries với fallback
try:
    import numpy as np
    from scipy import ndimage, interpolate
    from scipy.spatial.distance import cdist

    HAS_SCIPY = True
    logger.info("NumPy và SciPy được tải thành công")
except ImportError as e:
    logger.warning(f"Scientific libraries không khả dụng: {e}")
    HAS_SCIPY = False

# Import GPU libraries nếu có
try:
    import cupy as cp

    HAS_CUPY = True
    logger.info("CuPy GPU support được tải thành công")
except ImportError:
    HAS_CUPY = False

try:
    import pycuda.driver as cuda
    import pycuda.autoinit

    HAS_CUDA = True
    logger.info("PyCUDA được tải thành công")
except ImportError:
    HAS_CUDA = False

if not HAS_CUPY and not HAS_CUDA:
    logger.info("GPU support không khả dụng")

# Import core modules với fallback
try:
    from quangtps.dose.dose_grid import DoseGrid
    from quangtps.beams.beam_data import BeamData
    from quangtps.core.geometry.geometry_utils import transform_coordinates

    HAS_CORE_MODULES = True
    logger.info("Core modules được tải thành công")
except ImportError as e:
    logger.warning(f"Core modules không khả dụng: {e}")
    HAS_CORE_MODULES = False

    # Fallback classes
    class DoseGrid:
        def __init__(self, *args, **kwargs):
            self.shape = (64, 64, 32)
            self.spacing = (2.0, 2.0, 3.0)
            self.origin = (0.0, 0.0, 0.0)
            self.dose_data = np.zeros(self.shape)

    class BeamData:
        def __init__(self, *args, **kwargs):
            self.gantry_angle = 0.0
            self.energy = "6MV"
            self.weight = 1.0


class DoseAlgorithmType(Enum):
    """Enum cho các loại thuật toán tính liều."""

    PENCIL_BEAM = "pencil_beam"
    COLLAPSED_CONE = "collapsed_cone"
    MONTE_CARLO = "monte_carlo"
    MONTE_CARLO_GPU = "monte_carlo_gpu"
    CONVOLUTION_SUPERPOSITION = "convolution_superposition"
    AAA = "anisotropic_analytic"  # Anisotropic Analytic Algorithm
    ACUROS_XB = "acuros_xb"  # Grid-based Boltzmann solver
    FAST_SUPERPOSITION = "fast_superposition"
    RAYTRACING = "raytracing"
    SIMPLE = "simple"  # Simplified for testing


@dataclass
class DoseCalculationSettings:
    """Cài đặt cho tính toán liều."""

    # Algorithm selection
    algorithm: DoseAlgorithmType = DoseAlgorithmType.COLLAPSED_CONE

    # Grid settings
    grid_resolution: Tuple[float, float, float] = (2.0, 2.0, 3.0)  # mm
    calculation_margin: float = 50.0  # mm beyond PTV

    # Accuracy settings
    statistical_uncertainty: float = 0.02  # 2% for Monte Carlo
    max_iterations: int = 1000000  # Monte Carlo histories
    convergence_threshold: float = 0.001

    # Performance settings
    use_gpu: bool = True
    use_parallel_processing: bool = True
    max_workers: int = mp.cpu_count()
    chunk_size: int = 1000

    # Quality assurance
    validate_inputs: bool = True
    heterogeneity_correction: bool = True
    include_scatter: bool = True
    include_beam_hardening: bool = True

    # Advanced options
    beam_modeling_accuracy: str = "HIGH"  # LOW, MEDIUM, HIGH
    leaf_transmission: float = 0.02  # 2% for MLC
    rounded_leaf_ends: bool = True

    def __post_init__(self):
        """Validate settings."""
        if self.statistical_uncertainty <= 0 or self.statistical_uncertainty > 0.1:
            raise ValueError("Statistical uncertainty phải từ 0-10%")
        if self.max_iterations < 1000:
            raise ValueError("Max iterations phải ít nhất 1000")


@dataclass
class DoseCalculationResult:
    """Kết quả tính toán liều."""

    dose_grid: DoseGrid
    calculation_time: float  # seconds
    algorithm_used: DoseAlgorithmType

    # Quality metrics
    statistical_uncertainty_achieved: Optional[float] = None
    convergence_achieved: bool = True
    iterations_used: Optional[int] = None

    # Performance metrics
    gpu_used: bool = False
    memory_used: float = 0.0  # MB
    cpu_cores_used: int = 1

    # Validation metrics
    dose_range: Tuple[float, float] = (0.0, 0.0)  # (min, max) Gy
    total_monitor_units: float = 0.0
    mean_dose: float = 0.0  # Gy

    # Metadata
    calculation_timestamp: datetime = field(default_factory=datetime.now)
    settings_used: Optional[DoseCalculationSettings] = None

    def __post_init__(self):
        """Calculate summary statistics."""
        if hasattr(self.dose_grid, "dose_data"):
            dose_data = np.array(self.dose_grid.dose_data)
            valid_doses = dose_data[dose_data > 0]

            if len(valid_doses) > 0:
                self.dose_range = (
                    float(np.min(valid_doses)),
                    float(np.max(valid_doses)),
                )
                self.mean_dose = float(np.mean(valid_doses))

    def get_summary(self) -> Dict[str, Any]:
        """Lấy tóm tắt kết quả."""
        return {
            "algorithm": self.algorithm_used.value,
            "calculation_time": self.calculation_time,
            "dose_range": self.dose_range,
            "mean_dose": self.mean_dose,
            "convergence_achieved": self.convergence_achieved,
            "gpu_used": self.gpu_used,
            "memory_used": self.memory_used,
            "statistical_uncertainty": self.statistical_uncertainty_achieved,
        }


class BaseDoseCalculator:
    """
    Base class cho tất cả dose calculators.
    """

    def __init__(self, settings: Optional[DoseCalculationSettings] = None):
        self.settings = settings or DoseCalculationSettings()
        self.name = "Base Calculator"

        # Performance monitoring
        self._calculation_count = 0
        self._total_calculation_time = 0.0

        logger.info(f"{self.name} khởi tạo")

    def calculate_dose(
        self,
        beam_data: BeamData,
        patient_geometry: np.ndarray,
        dose_grid: DoseGrid,
        progress_callback: Optional[Callable] = None,
    ) -> DoseCalculationResult:
        """
        Calculate dose distribution.

        Args:
            beam_data: Beam parameters
            patient_geometry: Patient CT data
            dose_grid: Target dose grid
            progress_callback: Progress reporting function

        Returns:
            DoseCalculationResult with dose distribution
        """
        raise NotImplementedError("Subclasses must implement calculate_dose")

    def validate_inputs(
        self, beam_data: BeamData, patient_geometry: np.ndarray, dose_grid: DoseGrid
    ) -> bool:
        """Validate input parameters."""
        try:
            if patient_geometry.size == 0:
                logger.error("Patient geometry trống")
                return False

            if dose_grid.shape[0] == 0:
                logger.error("Dose grid trống")
                return False

            return True

        except Exception as e:
            logger.error(f"Lỗi validate inputs: {e}")
            return False

    def get_performance_stats(self) -> Dict[str, Any]:
        """Lấy thống kê performance."""
        avg_time = self._total_calculation_time / max(self._calculation_count, 1)

        return {
            "calculator_name": self.name,
            "calculation_count": self._calculation_count,
            "total_time": self._total_calculation_time,
            "average_time": avg_time,
            "calculations_per_second": 1.0 / max(avg_time, 0.001),
        }


class PencilBeamCalculator(BaseDoseCalculator):
    """
    Pencil Beam dose calculation algorithm.
    Fast but less accurate for heterogeneous media.
    """

    def __init__(self, settings: Optional[DoseCalculationSettings] = None):
        super().__init__(settings)
        self.name = "Pencil Beam Calculator"

        # Algorithm-specific parameters
        self.beam_kernel_size = 5  # mm
        self.scatter_factor = 0.05
        self.tissue_correction_enabled = True

        logger.info("Pencil Beam Calculator khởi tạo")

    def calculate_dose(
        self,
        beam_data: BeamData,
        patient_geometry: np.ndarray,
        dose_grid: DoseGrid,
        progress_callback: Optional[Callable] = None,
    ) -> DoseCalculationResult:
        """Calculate dose using Pencil Beam algorithm."""
        start_time = time.time()

        try:
            if self.settings.validate_inputs:
                if not self.validate_inputs(beam_data, patient_geometry, dose_grid):
                    raise ValueError("Input validation failed")

            # Initialize dose distribution
            dose_distribution = np.zeros(dose_grid.shape)

            # Calculate primary dose
            if progress_callback:
                progress_callback(20, "Calculating primary dose...")

            primary_dose = self._calculate_primary_dose(
                beam_data, patient_geometry, dose_grid
            )
            dose_distribution += primary_dose

            # Calculate scatter dose
            if self.settings.include_scatter and progress_callback:
                progress_callback(60, "Calculating scatter dose...")

            if self.settings.include_scatter:
                scatter_dose = self._calculate_scatter_dose(
                    beam_data, patient_geometry, dose_grid, primary_dose
                )
                dose_distribution += scatter_dose

            # Apply heterogeneity corrections
            if self.settings.heterogeneity_correction and progress_callback:
                progress_callback(80, "Applying heterogeneity corrections...")

            if self.settings.heterogeneity_correction:
                dose_distribution = self._apply_heterogeneity_correction(
                    dose_distribution, patient_geometry, dose_grid
                )

            # Create result dose grid
            result_dose_grid = DoseGrid(
                grid_data=dose_distribution,
                origin=dose_grid.origin,
                spacing=dose_grid.spacing,
            )

            # Calculate metrics
            calculation_time = time.time() - start_time
            self._calculation_count += 1
            self._total_calculation_time += calculation_time

            if progress_callback:
                progress_callback(100, "Dose calculation completed")

            return DoseCalculationResult(
                dose_grid=result_dose_grid,
                calculation_time=calculation_time,
                algorithm_used=DoseAlgorithmType.PENCIL_BEAM,
                convergence_achieved=True,
                cpu_cores_used=1,
                settings_used=self.settings,
            )

        except Exception as e:
            logger.error(f"Lỗi Pencil Beam calculation: {e}")
            # Return empty result
            empty_grid = DoseGrid(
                grid_data=np.zeros(dose_grid.shape),
                origin=dose_grid.origin,
                spacing=dose_grid.spacing,
            )
            return DoseCalculationResult(
                dose_grid=empty_grid,
                calculation_time=time.time() - start_time,
                algorithm_used=DoseAlgorithmType.PENCIL_BEAM,
                convergence_achieved=False,
            )

    def _calculate_primary_dose(
        self, beam_data: BeamData, patient_geometry: np.ndarray, dose_grid: DoseGrid
    ) -> np.ndarray:
        """Calculate primary dose component."""
        try:
            dose = np.zeros(dose_grid.shape)

            # Simple depth-dose calculation
            # In real implementation, this would use measured beam data

            beam_center_x = dose_grid.shape[0] // 2
            beam_center_y = dose_grid.shape[1] // 2

            # Create beam profile (simplified Gaussian)
            sigma_x = 20  # mm
            sigma_y = 20  # mm

            for z in range(dose_grid.shape[2]):
                depth = z * dose_grid.spacing[2]  # mm

                # Depth dose curve (simplified exponential)
                pdd = 100 * np.exp(-depth / 150)  # cGy

                for x in range(dose_grid.shape[0]):
                    for y in range(dose_grid.shape[1]):
                        # Distance from beam center
                        dx = (x - beam_center_x) * dose_grid.spacing[0]
                        dy = (y - beam_center_y) * dose_grid.spacing[1]

                        # Gaussian profile
                        profile = np.exp(
                            -(dx**2) / (2 * sigma_x**2) - (dy**2) / (2 * sigma_y**2)
                        )

                        # Final dose
                        dose[x, y, z] = pdd * profile * beam_data.weight

            return dose

        except Exception as e:
            logger.error(f"Lỗi calculate primary dose: {e}")
            return np.zeros(dose_grid.shape)

    def _calculate_scatter_dose(
        self,
        beam_data: BeamData,
        patient_geometry: np.ndarray,
        dose_grid: DoseGrid,
        primary_dose: np.ndarray,
    ) -> np.ndarray:
        """Calculate scatter dose component."""
        try:
            # Simplified scatter calculation
            # Real implementation would use convolution with scatter kernels

            if HAS_SCIPY:
                # Use Gaussian filter as approximation
                scatter_dose = (
                    ndimage.gaussian_filter(
                        primary_dose,
                        sigma=3.0,  # mm
                    )
                    * self.scatter_factor
                )
            else:
                # Simple uniform scatter
                scatter_dose = primary_dose * self.scatter_factor

            return scatter_dose

        except Exception as e:
            logger.error(f"Lỗi calculate scatter dose: {e}")
            return np.zeros(dose_grid.shape)

    def _apply_heterogeneity_correction(
        self, dose: np.ndarray, patient_geometry: np.ndarray, dose_grid: DoseGrid
    ) -> np.ndarray:
        """Apply heterogeneity corrections."""
        try:
            corrected_dose = dose.copy()

            # Simple density-based correction
            # Assuming patient_geometry contains HU values

            for x in range(dose.shape[0]):
                for y in range(dose.shape[1]):
                    for z in range(dose.shape[2]):
                        hu_value = (
                            patient_geometry[x, y, z]
                            if patient_geometry.size > 0
                            else 0
                        )

                        # Convert HU to density (simplified)
                        density = 1.0 + hu_value / 1000.0
                        density = max(
                            0.1, min(3.0, density)
                        )  # Clamp to reasonable range

                        # Apply density correction
                        corrected_dose[x, y, z] = dose[x, y, z] / density

            return corrected_dose

        except Exception as e:
            logger.error(f"Lỗi apply heterogeneity correction: {e}")
            return dose


class CollapsedConeCalculator(BaseDoseCalculator):
    """
    Collapsed Cone Convolution algorithm.
    More accurate for heterogeneous media than Pencil Beam.
    """

    def __init__(self, settings: Optional[DoseCalculationSettings] = None):
        super().__init__(settings)
        self.name = "Collapsed Cone Calculator"

        # Algorithm-specific parameters
        self.cone_angle = 90.0  # degrees
        self.kernel_resolution = 2.0  # mm
        self.max_ray_length = 300.0  # mm

        logger.info("Collapsed Cone Calculator khởi tạo")

    def calculate_dose(
        self,
        beam_data: BeamData,
        patient_geometry: np.ndarray,
        dose_grid: DoseGrid,
        progress_callback: Optional[Callable] = None,
    ) -> DoseCalculationResult:
        """Calculate dose using Collapsed Cone algorithm."""
        start_time = time.time()

        try:
            if self.settings.validate_inputs:
                if not self.validate_inputs(beam_data, patient_geometry, dose_grid):
                    raise ValueError("Input validation failed")

            # Initialize dose distribution
            dose_distribution = np.zeros(dose_grid.shape)

            # Calculate primary fluence
            if progress_callback:
                progress_callback(25, "Calculating primary fluence...")

            primary_fluence = self._calculate_primary_fluence(
                beam_data, patient_geometry, dose_grid
            )

            # Convolve with dose kernels
            if progress_callback:
                progress_callback(50, "Convolving with dose kernels...")

            primary_dose = self._convolve_with_kernels(
                primary_fluence, patient_geometry, dose_grid
            )
            dose_distribution += primary_dose

            # Calculate scatter contribution
            if self.settings.include_scatter and progress_callback:
                progress_callback(75, "Calculating scatter contribution...")

            if self.settings.include_scatter:
                scatter_dose = self._calculate_scatter_contribution(
                    primary_fluence, patient_geometry, dose_grid
                )
                dose_distribution += scatter_dose

            # Create result dose grid
            result_dose_grid = DoseGrid(
                grid_data=dose_distribution,
                origin=dose_grid.origin,
                spacing=dose_grid.spacing,
            )

            # Calculate metrics
            calculation_time = time.time() - start_time
            self._calculation_count += 1
            self._total_calculation_time += calculation_time

            if progress_callback:
                progress_callback(100, "Collapsed Cone calculation completed")

            return DoseCalculationResult(
                dose_grid=result_dose_grid,
                calculation_time=calculation_time,
                algorithm_used=DoseAlgorithmType.COLLAPSED_CONE,
                convergence_achieved=True,
                cpu_cores_used=self.settings.max_workers
                if self.settings.use_parallel_processing
                else 1,
                settings_used=self.settings,
            )

        except Exception as e:
            logger.error(f"Lỗi Collapsed Cone calculation: {e}")
            # Return empty result
            empty_grid = DoseGrid(
                grid_data=np.zeros(dose_grid.shape),
                origin=dose_grid.origin,
                spacing=dose_grid.spacing,
            )
            return DoseCalculationResult(
                dose_grid=empty_grid,
                calculation_time=time.time() - start_time,
                algorithm_used=DoseAlgorithmType.COLLAPSED_CONE,
                convergence_achieved=False,
            )

    def _calculate_primary_fluence(
        self, beam_data: BeamData, patient_geometry: np.ndarray, dose_grid: DoseGrid
    ) -> np.ndarray:
        """Calculate primary photon fluence."""
        try:
            fluence = np.zeros(dose_grid.shape)

            # Ray tracing through patient geometry
            beam_center_x = dose_grid.shape[0] // 2
            beam_center_y = dose_grid.shape[1] // 2

            for x in range(dose_grid.shape[0]):
                for y in range(dose_grid.shape[1]):
                    # Calculate attenuation along ray
                    attenuation = self._calculate_ray_attenuation(
                        x, y, patient_geometry, dose_grid
                    )

                    # Distance from beam axis
                    dx = (x - beam_center_x) * dose_grid.spacing[0]
                    dy = (y - beam_center_y) * dose_grid.spacing[1]
                    distance = np.sqrt(dx**2 + dy**2)

                    # Off-axis ratio (simplified)
                    oar = np.exp(-(distance**2) / (2 * 25**2))  # 25mm sigma

                    # Set fluence for all depths
                    for z in range(dose_grid.shape[2]):
                        fluence[x, y, z] = oar * attenuation[z] * beam_data.weight

            return fluence

        except Exception as e:
            logger.error(f"Lỗi calculate primary fluence: {e}")
            return np.zeros(dose_grid.shape)

    def _calculate_ray_attenuation(
        self, x: int, y: int, patient_geometry: np.ndarray, dose_grid: DoseGrid
    ) -> np.ndarray:
        """Calculate attenuation along a ray."""
        try:
            attenuation = np.ones(dose_grid.shape[2])

            if patient_geometry.size == 0:
                return attenuation

            # Cumulative attenuation
            cumulative_attenuation = 0.0

            for z in range(dose_grid.shape[2]):
                if (
                    x < patient_geometry.shape[0]
                    and y < patient_geometry.shape[1]
                    and z < patient_geometry.shape[2]
                ):
                    hu_value = patient_geometry[x, y, z]

                    # Convert HU to linear attenuation coefficient
                    mu = self._hu_to_attenuation_coefficient(hu_value)

                    # Add path length contribution
                    path_length = dose_grid.spacing[2] / 10.0  # Convert mm to cm
                    cumulative_attenuation += mu * path_length

                    # Calculate transmitted fluence
                    attenuation[z] = np.exp(-cumulative_attenuation)

            return attenuation

        except Exception as e:
            logger.error(f"Lỗi calculate ray attenuation: {e}")
            return np.ones(dose_grid.shape[2])

    def _hu_to_attenuation_coefficient(self, hu_value: float) -> float:
        """Convert Hounsfield Unit to linear attenuation coefficient."""
        try:
            # Simplified conversion for 6MV photons
            # Real implementation would use energy-specific data

            if hu_value < -500:  # Air
                return 0.0002  # cm^-1
            elif hu_value < 50:  # Soft tissue
                return 0.02 + (hu_value + 500) * (0.18 - 0.02) / 550
            else:  # Bone
                return 0.18 + (hu_value - 50) * (0.5 - 0.18) / 1500

        except Exception:
            return 0.18  # Default to soft tissue

    def _convolve_with_kernels(
        self, fluence: np.ndarray, patient_geometry: np.ndarray, dose_grid: DoseGrid
    ) -> np.ndarray:
        """Convolve fluence with dose deposition kernels."""
        try:
            dose = np.zeros(dose_grid.shape)

            # Simplified kernel convolution
            # Real implementation would use energy-dependent kernels

            if HAS_SCIPY:
                # Use 3D Gaussian filter as kernel approximation
                kernel_sigma = self.kernel_resolution / dose_grid.spacing[0]
                dose = ndimage.gaussian_filter(fluence, sigma=kernel_sigma)
            else:
                # Simple copy without convolution
                dose = fluence.copy()

            # Scale by energy and beam parameters
            dose *= 0.6  # cGy per fluence unit (simplified)

            return dose

        except Exception as e:
            logger.error(f"Lỗi convolve with kernels: {e}")
            return fluence

    def _calculate_scatter_contribution(
        self, fluence: np.ndarray, patient_geometry: np.ndarray, dose_grid: DoseGrid
    ) -> np.ndarray:
        """Calculate scatter dose contribution."""
        try:
            scatter_dose = np.zeros(dose_grid.shape)

            if HAS_SCIPY:
                # Scatter is approximated as broad Gaussian
                scatter_sigma = 20.0 / dose_grid.spacing[0]  # 20mm sigma
                scatter_dose = (
                    ndimage.gaussian_filter(fluence, sigma=scatter_sigma) * 0.05
                )  # 5% scatter fraction
            else:
                # Simple uniform scatter
                scatter_dose = fluence * 0.05

            return scatter_dose

        except Exception as e:
            logger.error(f"Lỗi calculate scatter contribution: {e}")
            return np.zeros(dose_grid.shape)


class MonteCarloCalculator(BaseDoseCalculator):
    """
    Monte Carlo dose calculation algorithm.
    Most accurate but computationally intensive.
    """

    def __init__(self, settings: Optional[DoseCalculationSettings] = None):
        super().__init__(settings)
        self.name = "Monte Carlo Calculator"

        # Algorithm-specific parameters
        self.particle_cutoff_energy = 0.01  # MeV
        self.transport_physics = "PRESTA"
        self.variance_reduction_enabled = True

        logger.info("Monte Carlo Calculator khởi tạo")

    def calculate_dose(
        self,
        beam_data: BeamData,
        patient_geometry: np.ndarray,
        dose_grid: DoseGrid,
        progress_callback: Optional[Callable] = None,
    ) -> DoseCalculationResult:
        """Calculate dose using Monte Carlo simulation."""
        start_time = time.time()

        try:
            if self.settings.validate_inputs:
                if not self.validate_inputs(beam_data, patient_geometry, dose_grid):
                    raise ValueError("Input validation failed")

            # Initialize dose accumulator
            dose_accumulator = np.zeros(dose_grid.shape)
            variance_accumulator = np.zeros(dose_grid.shape)

            # Run Monte Carlo simulation
            if progress_callback:
                progress_callback(10, "Starting Monte Carlo simulation...")

            histories_completed = 0
            target_histories = self.settings.max_iterations
            batch_size = min(10000, target_histories // 10)

            while histories_completed < target_histories:
                # Run batch of histories
                batch_dose, batch_variance = self._run_mc_batch(
                    batch_size, beam_data, patient_geometry, dose_grid
                )

                # Accumulate results
                dose_accumulator += batch_dose
                variance_accumulator += batch_variance
                histories_completed += batch_size

                # Check convergence
                if histories_completed % (batch_size * 5) == 0:
                    current_uncertainty = self._calculate_uncertainty(
                        dose_accumulator, variance_accumulator, histories_completed
                    )

                    if progress_callback:
                        progress = min(
                            90, (histories_completed / target_histories) * 90
                        )
                        progress_callback(
                            progress,
                            f"MC: {histories_completed}/{target_histories} histories, "
                            f"uncertainty: {current_uncertainty:.2%}",
                        )

                    # Early termination if converged
                    if current_uncertainty < self.settings.statistical_uncertainty:
                        logger.info(
                            f"MC converged early at {histories_completed} histories"
                        )
                        break

            # Normalize dose
            final_dose = dose_accumulator / histories_completed
            final_uncertainty = self._calculate_uncertainty(
                dose_accumulator, variance_accumulator, histories_completed
            )

            # Create result dose grid
            result_dose_grid = DoseGrid(
                grid_data=final_dose, origin=dose_grid.origin, spacing=dose_grid.spacing
            )

            # Calculate metrics
            calculation_time = time.time() - start_time
            self._calculation_count += 1
            self._total_calculation_time += calculation_time

            if progress_callback:
                progress_callback(100, "Monte Carlo calculation completed")

            return DoseCalculationResult(
                dose_grid=result_dose_grid,
                calculation_time=calculation_time,
                algorithm_used=DoseAlgorithmType.MONTE_CARLO,
                statistical_uncertainty_achieved=final_uncertainty,
                convergence_achieved=final_uncertainty
                < self.settings.statistical_uncertainty * 2,
                iterations_used=histories_completed,
                cpu_cores_used=self.settings.max_workers
                if self.settings.use_parallel_processing
                else 1,
                settings_used=self.settings,
            )

        except Exception as e:
            logger.error(f"Lỗi Monte Carlo calculation: {e}")
            # Return empty result
            empty_grid = DoseGrid(
                grid_data=np.zeros(dose_grid.shape),
                origin=dose_grid.origin,
                spacing=dose_grid.spacing,
            )
            return DoseCalculationResult(
                dose_grid=empty_grid,
                calculation_time=time.time() - start_time,
                algorithm_used=DoseAlgorithmType.MONTE_CARLO,
                convergence_achieved=False,
            )

    def _run_mc_batch(
        self,
        batch_size: int,
        beam_data: BeamData,
        patient_geometry: np.ndarray,
        dose_grid: DoseGrid,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run a batch of Monte Carlo histories."""
        try:
            dose_batch = np.zeros(dose_grid.shape)
            variance_batch = np.zeros(dose_grid.shape)

            if self.settings.use_parallel_processing:
                # Parallel processing
                dose_batch, variance_batch = self._run_parallel_mc_batch(
                    batch_size, beam_data, patient_geometry, dose_grid
                )
            else:
                # Sequential processing
                for i in range(batch_size):
                    history_dose = self._simulate_single_history(
                        beam_data, patient_geometry, dose_grid
                    )
                    dose_batch += history_dose
                    variance_batch += history_dose**2

            return dose_batch, variance_batch

        except Exception as e:
            logger.error(f"Lỗi run MC batch: {e}")
            return np.zeros(dose_grid.shape), np.zeros(dose_grid.shape)

    def _run_parallel_mc_batch(
        self,
        batch_size: int,
        beam_data: BeamData,
        patient_geometry: np.ndarray,
        dose_grid: DoseGrid,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run Monte Carlo batch with parallel processing."""
        try:
            dose_batch = np.zeros(dose_grid.shape)
            variance_batch = np.zeros(dose_grid.shape)

            # Divide batch among workers
            histories_per_worker = batch_size // self.settings.max_workers
            remaining_histories = batch_size % self.settings.max_workers

            with ThreadPoolExecutor(max_workers=self.settings.max_workers) as executor:
                futures = []

                # Submit worker tasks
                for worker_id in range(self.settings.max_workers):
                    worker_histories = histories_per_worker
                    if worker_id < remaining_histories:
                        worker_histories += 1

                    if worker_histories > 0:
                        future = executor.submit(
                            self._run_worker_histories,
                            worker_histories,
                            beam_data,
                            patient_geometry,
                            dose_grid,
                        )
                        futures.append(future)

                # Collect results
                for future in futures:
                    try:
                        worker_dose, worker_variance = future.result()
                        dose_batch += worker_dose
                        variance_batch += worker_variance
                    except Exception as e:
                        logger.error(f"Lỗi worker MC: {e}")

            return dose_batch, variance_batch

        except Exception as e:
            logger.error(f"Lỗi parallel MC batch: {e}")
            return np.zeros(dose_grid.shape), np.zeros(dose_grid.shape)

    def _run_worker_histories(
        self,
        num_histories: int,
        beam_data: BeamData,
        patient_geometry: np.ndarray,
        dose_grid: DoseGrid,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run histories for a single worker."""
        try:
            worker_dose = np.zeros(dose_grid.shape)
            worker_variance = np.zeros(dose_grid.shape)

            for i in range(num_histories):
                history_dose = self._simulate_single_history(
                    beam_data, patient_geometry, dose_grid
                )
                worker_dose += history_dose
                worker_variance += history_dose**2

            return worker_dose, worker_variance

        except Exception as e:
            logger.error(f"Lỗi worker histories: {e}")
            return np.zeros(dose_grid.shape), np.zeros(dose_grid.shape)

    def _simulate_single_history(
        self, beam_data: BeamData, patient_geometry: np.ndarray, dose_grid: DoseGrid
    ) -> np.ndarray:
        """Simulate a single particle history."""
        try:
            history_dose = np.zeros(dose_grid.shape)

            # Simplified Monte Carlo simulation
            # Real implementation would use full transport physics

            # Sample initial particle parameters
            x = np.random.uniform(0, dose_grid.shape[0] - 1)
            y = np.random.uniform(0, dose_grid.shape[1] - 1)
            z = 0.0

            # Initial energy (MeV)
            if "6MV" in beam_data.energy:
                energy = np.random.exponential(2.0)  # Mean 2 MeV
            else:
                energy = np.random.exponential(6.0)  # Mean 6 MeV for 18MV

            # Transport particle through geometry
            while energy > self.particle_cutoff_energy and z < dose_grid.shape[2] - 1:
                # Current voxel indices
                ix, iy, iz = int(x), int(y), int(z)

                if (
                    0 <= ix < dose_grid.shape[0]
                    and 0 <= iy < dose_grid.shape[1]
                    and 0 <= iz < dose_grid.shape[2]
                ):
                    # Get material properties
                    if patient_geometry.size > 0:
                        hu_value = patient_geometry[ix, iy, iz]
                        density = 1.0 + hu_value / 1000.0
                        density = max(0.1, min(3.0, density))
                    else:
                        density = 1.0

                    # Sample interaction
                    interaction_prob = density * 0.1  # Simplified

                    if np.random.random() < interaction_prob:
                        # Deposit energy
                        energy_deposited = energy * 0.1  # Simplified
                        history_dose[ix, iy, iz] += energy_deposited
                        energy -= energy_deposited

                        # Scatter particle (simplified)
                        scatter_angle = np.random.exponential(0.1)  # radians
                        x += np.sin(scatter_angle)
                        y += np.cos(scatter_angle) * 0.5

                    # Advance particle
                    step_size = 1.0 / density  # mm
                    z += step_size
                else:
                    break

            # Scale by beam weight
            history_dose *= beam_data.weight

            return history_dose

        except Exception as e:
            logger.error(f"Lỗi simulate single history: {e}")
            return np.zeros(dose_grid.shape)

    def _calculate_uncertainty(
        self,
        dose_accumulator: np.ndarray,
        variance_accumulator: np.ndarray,
        num_histories: int,
    ) -> float:
        """Calculate statistical uncertainty."""
        try:
            if num_histories < 2:
                return 1.0

            # Mean dose
            mean_dose = dose_accumulator / num_histories

            # Variance
            variance = (variance_accumulator / num_histories) - mean_dose**2
            variance = np.maximum(variance, 0)  # Ensure non-negative

            # Standard error
            std_error = np.sqrt(variance / num_histories)

            # Relative uncertainty in high dose region
            high_dose_mask = mean_dose > 0.5 * np.max(mean_dose)
            if np.sum(high_dose_mask) > 0:
                high_dose_uncertainty = (
                    std_error[high_dose_mask] / mean_dose[high_dose_mask]
                )
                uncertainty = np.mean(high_dose_uncertainty[high_dose_uncertainty > 0])
            else:
                uncertainty = 1.0

            return float(uncertainty)

        except Exception as e:
            logger.error(f"Lỗi calculate uncertainty: {e}")
            return 1.0


class AdvancedDoseEngine:
    """
    Advanced Dose Engine với multiple algorithms và automatic selection.
    """

    def __init__(self, settings: Optional[DoseCalculationSettings] = None):
        self.settings = settings or DoseCalculationSettings()

        # Initialize calculators
        self.calculators: Dict[DoseAlgorithmType, BaseDoseCalculator] = {}
        self._initialize_calculators()

        # Performance monitoring
        self._calculation_history: List[DoseCalculationResult] = []

        logger.info("Advanced Dose Engine khởi tạo")

    def _initialize_calculators(self):
        """Initialize all available calculators."""
        try:
            # Always available calculators
            self.calculators[DoseAlgorithmType.PENCIL_BEAM] = PencilBeamCalculator(
                self.settings
            )
            self.calculators[DoseAlgorithmType.COLLAPSED_CONE] = (
                CollapsedConeCalculator(self.settings)
            )
            self.calculators[DoseAlgorithmType.MONTE_CARLO] = MonteCarloCalculator(
                self.settings
            )

            # GPU calculators (if available)
            if HAS_CUPY or HAS_CUDA:
                # Note: GPU Monte Carlo would be implemented separately
                logger.info("GPU support detected - GPU calculators available")

            logger.info(f"Initialized {len(self.calculators)} dose calculators")

        except Exception as e:
            logger.error(f"Lỗi initialize calculators: {e}")

    def get_available_algorithms(self) -> List[DoseAlgorithmType]:
        """Lấy danh sách thuật toán khả dụng."""
        return list(self.calculators.keys())

    def calculate_dose(
        self,
        algorithm: Optional[DoseAlgorithmType] = None,
        beam_data: Optional[BeamData] = None,
        patient_geometry: Optional[np.ndarray] = None,
        dose_grid: Optional[DoseGrid] = None,
        progress_callback: Optional[Callable] = None,
    ) -> Optional[DoseCalculationResult]:
        """
        Calculate dose using specified or automatically selected algorithm.
        """
        try:
            # Use default algorithm if not specified
            if algorithm is None:
                algorithm = self.settings.algorithm

            # Check if algorithm is available
            if algorithm not in self.calculators:
                logger.error(f"Algorithm {algorithm} không khả dụng")
                return None

            # Create default data if not provided
            if beam_data is None:
                beam_data = BeamData()

            if patient_geometry is None:
                patient_geometry = np.ones((64, 64, 32)) * 0  # Air-equivalent

            if dose_grid is None:
                dose_grid = DoseGrid()

            # Get calculator
            calculator = self.calculators[algorithm]

            if progress_callback:
                progress_callback(5, f"Starting {algorithm.value} calculation...")

            # Perform calculation
            result = calculator.calculate_dose(
                beam_data, patient_geometry, dose_grid, progress_callback
            )

            # Store result in history
            self._calculation_history.append(result)

            # Limit history size
            if len(self._calculation_history) > 100:
                self._calculation_history = self._calculation_history[-100:]

            logger.info(f"Dose calculation completed with {algorithm.value}")
            return result

        except Exception as e:
            logger.error(f"Lỗi calculate dose: {e}")
            return None

    def select_optimal_algorithm(
        self, target_accuracy: str = "MEDIUM", time_constraint: Optional[float] = None
    ) -> DoseAlgorithmType:
        """
        Automatically select optimal algorithm based on constraints.
        """
        try:
            if target_accuracy == "LOW" or time_constraint and time_constraint < 60:
                # Fast calculation needed
                return DoseAlgorithmType.PENCIL_BEAM

            elif target_accuracy == "HIGH" or not time_constraint:
                # Accuracy priority
                if DoseAlgorithmType.MONTE_CARLO in self.calculators:
                    return DoseAlgorithmType.MONTE_CARLO
                else:
                    return DoseAlgorithmType.COLLAPSED_CONE

            else:  # MEDIUM accuracy
                # Balanced choice
                return DoseAlgorithmType.COLLAPSED_CONE

        except Exception as e:
            logger.error(f"Lỗi select optimal algorithm: {e}")
            return DoseAlgorithmType.PENCIL_BEAM  # Safe fallback

    def compare_algorithms(
        self,
        algorithms: List[DoseAlgorithmType],
        beam_data: BeamData,
        patient_geometry: np.ndarray,
        dose_grid: DoseGrid,
    ) -> Dict[str, Any]:
        """Compare multiple algorithms on same data."""
        try:
            comparison_results = {}

            for algorithm in algorithms:
                if algorithm in self.calculators:
                    result = self.calculate_dose(
                        algorithm, beam_data, patient_geometry, dose_grid
                    )

                    if result:
                        comparison_results[algorithm.value] = {
                            "calculation_time": result.calculation_time,
                            "mean_dose": result.mean_dose,
                            "dose_range": result.dose_range,
                            "convergence_achieved": result.convergence_achieved,
                            "statistical_uncertainty": result.statistical_uncertainty_achieved,
                        }

            return {
                "algorithm_comparison": comparison_results,
                "fastest_algorithm": min(
                    comparison_results.items(), key=lambda x: x[1]["calculation_time"]
                )[0]
                if comparison_results
                else None,
                "most_accurate": "monte_carlo"
                if "monte_carlo" in comparison_results
                else None,
            }

        except Exception as e:
            logger.error(f"Lỗi compare algorithms: {e}")
            return {}

    def get_engine_statistics(self) -> Dict[str, Any]:
        """Lấy thống kê performance của engine."""
        try:
            if not self._calculation_history:
                return {"total_calculations": 0}

            # Calculate statistics
            total_calculations = len(self._calculation_history)
            total_time = sum(r.calculation_time for r in self._calculation_history)
            avg_time = total_time / total_calculations

            # Algorithm usage
            algorithm_counts = {}
            for result in self._calculation_history:
                alg_name = result.algorithm_used.value
                algorithm_counts[alg_name] = algorithm_counts.get(alg_name, 0) + 1

            # Success rate
            successful_calculations = sum(
                1 for r in self._calculation_history if r.convergence_achieved
            )
            success_rate = successful_calculations / total_calculations

            return {
                "total_calculations": total_calculations,
                "total_time": total_time,
                "average_time": avg_time,
                "success_rate": success_rate,
                "algorithm_usage": algorithm_counts,
                "available_algorithms": [
                    alg.value for alg in self.get_available_algorithms()
                ],
                "last_calculation": self._calculation_history[-1].get_summary()
                if self._calculation_history
                else None,
            }

        except Exception as e:
            logger.error(f"Lỗi get engine statistics: {e}")
            return {"error": str(e)}


# Factory functions
def create_dose_engine(
    settings: Optional[DoseCalculationSettings] = None,
) -> AdvancedDoseEngine:
    """Factory function để tạo Advanced Dose Engine."""
    return AdvancedDoseEngine(settings)


def create_dose_calculator(
    algorithm: DoseAlgorithmType, settings: Optional[DoseCalculationSettings] = None
) -> BaseDoseCalculator:
    """Factory function để tạo specific dose calculator."""
    if algorithm == DoseAlgorithmType.PENCIL_BEAM:
        return PencilBeamCalculator(settings)
    elif algorithm == DoseAlgorithmType.COLLAPSED_CONE:
        return CollapsedConeCalculator(settings)
    elif algorithm == DoseAlgorithmType.MONTE_CARLO:
        return MonteCarloCalculator(settings)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def create_sample_dose_calculation() -> DoseCalculationResult:
    """Tạo sample dose calculation result để test."""
    # Create sample dose grid
    dose_data = np.random.rand(32, 32, 16) * 50  # Random dose 0-50 Gy

    sample_grid = DoseGrid(
        grid_data=dose_data, origin=(0.0, 0.0, 0.0), spacing=(2.0, 2.0, 3.0)
    )

    return DoseCalculationResult(
        dose_grid=sample_grid,
        calculation_time=45.5,
        algorithm_used=DoseAlgorithmType.COLLAPSED_CONE,
        statistical_uncertainty_achieved=0.015,
        convergence_achieved=True,
        iterations_used=50000,
        gpu_used=False,
        memory_used=256.0,
        cpu_cores_used=4,
    )


if __name__ == "__main__":
    # Test code
    logging.basicConfig(level=logging.INFO)

    # Test dose engine
    engine = create_dose_engine()

    print(
        f"Available algorithms: {[alg.value for alg in engine.get_available_algorithms()]}"
    )

    # Test calculation với sample data
    sample_beam = (
        BeamData()
        if HAS_CORE_MODULES
        else type("BeamData", (), {"weight": 1.0, "energy": "6MV"})()
    )
    sample_geometry = np.random.randint(-100, 100, (32, 32, 16))  # HU values
    sample_dose_grid = (
        DoseGrid()
        if HAS_CORE_MODULES
        else type(
            "DoseGrid",
            (),
            {
                "shape": (32, 32, 16),
                "spacing": (2.0, 2.0, 3.0),
                "origin": (0.0, 0.0, 0.0),
            },
        )()
    )

    # Test với Pencil Beam (fastest)
    result = engine.calculate_dose(
        algorithm=DoseAlgorithmType.PENCIL_BEAM,
        beam_data=sample_beam,
        patient_geometry=sample_geometry,
        dose_grid=sample_dose_grid,
    )

    if result:
        print(f"Calculation completed:")
        print(f"  Algorithm: {result.algorithm_used.value}")
        print(f"  Time: {result.calculation_time:.2f}s")
        print(f"  Mean dose: {result.mean_dose:.2f} Gy")
        print(f"  Success: {result.convergence_achieved}")

    # Test engine statistics
    stats = engine.get_engine_statistics()
    print(f"Engine statistics: {stats}")

    print("Advanced Dose Engine test hoàn thành!")
