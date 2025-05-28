"""
GPU-accelerated Gamma Analysis for QuangTPS
Provides high-performance gamma analysis using CUDA and OpenCL.
"""

import numpy as np
import logging
import time
from typing import Optional, Tuple, Callable, Dict, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import threading
import math

# GPU acceleration libraries (with fallbacks)
try:
    import cupy as cp

    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    cp = None

try:
    import pyopencl as cl
    import pyopencl.array as cl_array

    HAS_OPENCL = True
except ImportError:
    HAS_OPENCL = False
    cl = None

try:
    from numba import cuda, njit

    HAS_NUMBA_CUDA = True
except ImportError:
    HAS_NUMBA_CUDA = False
    cuda = None
    njit = None

logger = logging.getLogger(__name__)


@dataclass
class GPUGammaSettings:
    """Cài đặt cho GPU gamma analysis."""

    # Gamma criteria
    distance_mm: float = 3.0
    dose_percent: float = 3.0
    dose_threshold_percent: float = 10.0

    # GPU settings
    preferred_backend: str = "auto"  # auto, cupy, opencl, numba
    max_memory_usage_gb: float = 4.0
    use_fast_mode: bool = True
    chunk_size: int = 1000000

    # Quality settings
    interpolation_factor: int = 1
    search_radius_voxels: int = 10
    max_gamma_value: float = 10.0


@dataclass
class GPUGammaResult:
    """Kết quả GPU gamma analysis."""

    gamma_map: np.ndarray
    pass_rate: float
    mean_gamma: float
    max_gamma: float
    calculation_time: float
    gpu_backend: str
    memory_usage_mb: float

    # Performance metrics
    voxels_per_second: float = 0.0
    speedup_factor: float = 1.0


class GPUGammaAnalyzer:
    """High-performance GPU gamma analyzer."""

    def __init__(self, settings: Optional[GPUGammaSettings] = None):
        """Initialize GPU gamma analyzer."""
        self.settings = settings or GPUGammaSettings()
        self.gpu_backend = None
        self.gpu_context = None
        self.gpu_device = None

        # Initialize GPU backend
        self._initialize_gpu()

        logger.info(f"GPU Gamma Analyzer initialized with backend: {self.gpu_backend}")

    def _initialize_gpu(self):
        """Initialize GPU backend."""
        if self.settings.preferred_backend == "auto":
            # Try backends in order of preference
            if self._try_cupy():
                self.gpu_backend = "cupy"
            elif self._try_numba_cuda():
                self.gpu_backend = "numba_cuda"
            elif self._try_opencl():
                self.gpu_backend = "opencl"
            else:
                self.gpu_backend = "cpu_fallback"
                logger.warning("No GPU backend available, falling back to CPU")
        else:
            # Try specific backend
            backend_methods = {
                "cupy": self._try_cupy,
                "opencl": self._try_opencl,
                "numba": self._try_numba_cuda,
            }

            if self.settings.preferred_backend in backend_methods:
                if backend_methods[self.settings.preferred_backend]():
                    self.gpu_backend = self.settings.preferred_backend
                else:
                    self.gpu_backend = "cpu_fallback"
                    logger.warning(
                        f"Requested backend {self.settings.preferred_backend} not available"
                    )

    def _try_cupy(self) -> bool:
        """Try to initialize CuPy backend."""
        if not HAS_CUPY:
            return False

        try:
            # Test GPU availability
            cp.cuda.runtime.getDeviceCount()
            self.gpu_device = cp.cuda.Device()

            # Test memory allocation
            test_array = cp.zeros(1000, dtype=np.float32)
            del test_array

            logger.info("CuPy backend initialized successfully")
            return True

        except Exception as e:
            logger.warning(f"CuPy initialization failed: {e}")
            return False

    def _try_numba_cuda(self) -> bool:
        """Try to initialize Numba CUDA backend."""
        if not HAS_NUMBA_CUDA:
            return False

        try:
            # Test CUDA availability
            if cuda.is_available():
                cuda.select_device(0)
                logger.info("Numba CUDA backend initialized successfully")
                return True
            else:
                return False

        except Exception as e:
            logger.warning(f"Numba CUDA initialization failed: {e}")
            return False

    def _try_opencl(self) -> bool:
        """Try to initialize OpenCL backend."""
        if not HAS_OPENCL:
            return False

        try:
            # Get OpenCL platforms and devices
            platforms = cl.get_platforms()
            if not platforms:
                return False

            # Use first available GPU device
            for platform in platforms:
                devices = platform.get_devices(cl.device_type.GPU)
                if devices:
                    self.gpu_context = cl.Context(devices=[devices[0]])
                    self.gpu_device = devices[0]
                    logger.info(
                        f"OpenCL backend initialized with device: {devices[0].name}"
                    )
                    return True

            return False

        except Exception as e:
            logger.warning(f"OpenCL initialization failed: {e}")
            return False

    def calculate_gamma_gpu(
        self,
        reference_dose: np.ndarray,
        evaluated_dose: np.ndarray,
        spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        progress_callback: Optional[Callable] = None,
    ) -> GPUGammaResult:
        """Calculate gamma index using GPU acceleration."""

        start_time = time.time()

        # Validate inputs
        if reference_dose.shape != evaluated_dose.shape:
            raise ValueError("Reference and evaluated dose must have same shape")

        # Choose calculation method based on backend
        if self.gpu_backend == "cupy":
            result = self._calculate_gamma_cupy(
                reference_dose, evaluated_dose, spacing, progress_callback
            )
        elif self.gpu_backend == "numba_cuda":
            result = self._calculate_gamma_numba_cuda(
                reference_dose, evaluated_dose, spacing, progress_callback
            )
        elif self.gpu_backend == "opencl":
            result = self._calculate_gamma_opencl(
                reference_dose, evaluated_dose, spacing, progress_callback
            )
        else:
            # CPU fallback
            result = self._calculate_gamma_cpu_fallback(
                reference_dose, evaluated_dose, spacing, progress_callback
            )

        # Calculate performance metrics
        total_voxels = reference_dose.size
        calculation_time = time.time() - start_time
        voxels_per_second = (
            total_voxels / calculation_time if calculation_time > 0 else 0
        )

        result.calculation_time = calculation_time
        result.voxels_per_second = voxels_per_second
        result.gpu_backend = self.gpu_backend

        logger.info(f"GPU gamma analysis completed in {calculation_time:.2f}s")
        logger.info(f"Performance: {voxels_per_second / 1e6:.2f} M voxels/sec")

        return result

    def _calculate_gamma_cupy(
        self,
        reference_dose: np.ndarray,
        evaluated_dose: np.ndarray,
        spacing: Tuple[float, float, float],
        progress_callback: Optional[Callable],
    ) -> GPUGammaResult:
        """Calculate gamma using CuPy."""

        # Transfer data to GPU
        ref_gpu = cp.asarray(reference_dose)
        eval_gpu = cp.asarray(evaluated_dose)

        # Create dose threshold mask
        dose_threshold = cp.max(ref_gpu) * (
            self.settings.dose_threshold_percent / 100.0
        )
        analysis_mask = ref_gpu >= dose_threshold

        # Initialize gamma map
        gamma_map = cp.full_like(ref_gpu, self.settings.max_gamma_value)

        # Get analysis coordinates
        z_coords, y_coords, x_coords = cp.where(analysis_mask)
        total_points = len(z_coords)

        if progress_callback:
            progress_callback(10, f"Processing {total_points} points on GPU")

        # Process in chunks for memory efficiency
        chunk_size = min(self.settings.chunk_size, total_points)

        for i in range(0, total_points, chunk_size):
            end_idx = min(i + chunk_size, total_points)

            # Calculate gamma for this chunk
            self._cupy_gamma_kernel(
                ref_gpu,
                eval_gpu,
                gamma_map,
                z_coords[i:end_idx],
                y_coords[i:end_idx],
                x_coords[i:end_idx],
                spacing,
            )

            if progress_callback and i % (chunk_size * 10) == 0:
                progress = 10 + int((i / total_points) * 80)
                progress_callback(progress, f"Processed {i}/{total_points} points")

        # Transfer result back to CPU
        gamma_map_cpu = cp.asnumpy(gamma_map)
        analysis_mask_cpu = cp.asnumpy(analysis_mask)

        # Calculate statistics
        valid_gamma = gamma_map_cpu[analysis_mask_cpu]
        valid_gamma = valid_gamma[np.isfinite(valid_gamma)]

        if len(valid_gamma) > 0:
            pass_rate = np.sum(valid_gamma <= 1.0) / len(valid_gamma) * 100.0
            mean_gamma = np.mean(valid_gamma)
            max_gamma = np.max(valid_gamma)
        else:
            pass_rate = 0.0
            mean_gamma = float("inf")
            max_gamma = float("inf")

        # Estimate memory usage
        memory_usage_mb = (ref_gpu.nbytes + eval_gpu.nbytes + gamma_map.nbytes) / (
            1024**2
        )

        return GPUGammaResult(
            gamma_map=gamma_map_cpu,
            pass_rate=pass_rate,
            mean_gamma=mean_gamma,
            max_gamma=max_gamma,
            calculation_time=0.0,  # Will be set by caller
            gpu_backend="cupy",
            memory_usage_mb=memory_usage_mb,
        )

    def _cupy_gamma_kernel(
        self,
        ref_dose: cp.ndarray,
        eval_dose: cp.ndarray,
        gamma_map: cp.ndarray,
        z_coords: cp.ndarray,
        y_coords: cp.ndarray,
        x_coords: cp.ndarray,
        spacing: Tuple[float, float, float],
    ):
        """CuPy kernel for gamma calculation."""

        # Custom CUDA kernel for optimized gamma calculation
        gamma_kernel = cp.RawKernel(
            r"""
        extern "C" __global__
        void gamma_3d_kernel(
            const float* ref_dose,
            const float* eval_dose,
            float* gamma_map,
            const int* z_coords,
            const int* y_coords,
            const int* x_coords,
            int num_points,
            int nz, int ny, int nx,
            float spacing_z, float spacing_y, float spacing_x,
            float dose_criterion, float distance_criterion,
            int search_radius
        ) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= num_points) return;

            int z = z_coords[idx];
            int y = y_coords[idx];
            int x = x_coords[idx];

            float ref_val = ref_dose[z * ny * nx + y * nx + x];
            float min_gamma = 10.0f;

            // Search in neighborhood
            for (int dz = -search_radius; dz <= search_radius; dz++) {
                for (int dy = -search_radius; dy <= search_radius; dy++) {
                    for (int dx = -search_radius; dx <= search_radius; dx++) {
                        int nz_new = z + dz;
                        int ny_new = y + dy;
                        int nx_new = x + dx;

                        if (nz_new >= 0 && nz_new < nz &&
                            ny_new >= 0 && ny_new < ny &&
                            nx_new >= 0 && nx_new < nx) {

                            float eval_val = eval_dose[nz_new * ny * nx + ny_new * nx + nx_new];

                            // Calculate dose difference
                            float dose_diff = fabsf(ref_val - eval_val);
                            float dose_term = dose_diff / (dose_criterion * ref_val / 100.0f);

                            // Calculate distance
                            float dist_z = dz * spacing_z;
                            float dist_y = dy * spacing_y;
                            float dist_x = dx * spacing_x;
                            float distance = sqrtf(dist_z*dist_z + dist_y*dist_y + dist_x*dist_x);
                            float dist_term = distance / distance_criterion;

                            // Calculate gamma
                            float gamma_val = sqrtf(dose_term*dose_term + dist_term*dist_term);
                            if (gamma_val < min_gamma) {
                                min_gamma = gamma_val;
                            }
                        }
                    }
                }
            }

            gamma_map[z * ny * nx + y * nx + x] = min_gamma;
        }
        """,
            "gamma_3d_kernel",
        )

        # Launch kernel
        num_points = len(z_coords)
        block_size = 256
        grid_size = (num_points + block_size - 1) // block_size

        gamma_kernel(
            (grid_size,),
            (block_size,),
            (
                ref_dose,
                eval_dose,
                gamma_map,
                z_coords,
                y_coords,
                x_coords,
                num_points,
                ref_dose.shape[0],
                ref_dose.shape[1],
                ref_dose.shape[2],
                spacing[2],
                spacing[1],
                spacing[0],
                self.settings.dose_percent,
                self.settings.distance_mm,
                self.settings.search_radius_voxels,
            ),
        )

    def _calculate_gamma_numba_cuda(
        self,
        reference_dose: np.ndarray,
        evaluated_dose: np.ndarray,
        spacing: Tuple[float, float, float],
        progress_callback: Optional[Callable],
    ) -> GPUGammaResult:
        """Calculate gamma using Numba CUDA."""

        # Numba CUDA kernel
        @cuda.jit
        def numba_gamma_kernel(
            ref_dose, eval_dose, gamma_map, coords, spacing_vals, criteria
        ):
            idx = cuda.grid(ndim=1)
            if idx >= coords.shape[0]:
                return

            # Extract coordinates
            z, y, x = coords[idx, 0], coords[idx, 1], coords[idx, 2]
            ref_dose_val = ref_dose[z, y, x]
            eval_dose_val = eval_dose[z, y, x]

            # Calculate gamma for this voxel
            min_gamma = 10.0
            search_radius = 5

            for dz in range(-search_radius, search_radius + 1):
                for dy in range(-search_radius, search_radius + 1):
                    for dx in range(-search_radius, search_radius + 1):
                        nz, ny, nx = z + dz, y + dy, x + dx

                        # Boundary check
                        if (
                            nz < 0
                            or ny < 0
                            or nx < 0
                            or nz >= eval_dose.shape[0]
                            or ny >= eval_dose.shape[1]
                            or nx >= eval_dose.shape[2]
                        ):
                            continue

                        # Calculate spatial distance
                        spatial_dist = math.sqrt(
                            (dz * spacing_vals[0]) ** 2
                            + (dy * spacing_vals[1]) ** 2
                            + (dx * spacing_vals[2]) ** 2
                        )

                        # Calculate dose difference
                        dose_diff = abs(ref_dose_val - eval_dose[nz, ny, nx])

                        if ref_dose_val > 0:
                            dose_diff_percent = 100.0 * dose_diff / ref_dose_val
                        else:
                            dose_diff_percent = 0.0

                        # Calculate gamma
                        gamma_val = math.sqrt(
                            (spatial_dist / criteria[0]) ** 2
                            + (dose_diff_percent / criteria[1]) ** 2
                        )

                        min_gamma = min(min_gamma, gamma_val)

            gamma_map[z, y, x] = min_gamma

        # Prepare data for GPU
        dose_threshold = np.max(reference_dose) * (
            self.settings.dose_threshold_percent / 100.0
        )
        analysis_mask = reference_dose >= dose_threshold

        # Get coordinates to analyze
        coords = np.column_stack(np.where(analysis_mask))

        # Transfer to GPU
        ref_gpu = cuda.to_device(reference_dose.astype(np.float32))
        eval_gpu = cuda.to_device(evaluated_dose.astype(np.float32))
        gamma_gpu = cuda.to_device(np.full_like(reference_dose, 10.0, dtype=np.float32))
        coords_gpu = cuda.to_device(coords.astype(np.int32))
        spacing_gpu = cuda.to_device(np.array(spacing, dtype=np.float32))
        criteria_gpu = cuda.to_device(
            np.array(
                [self.settings.distance_mm, self.settings.dose_percent],
                dtype=np.float32,
            )
        )

        # Launch kernel
        threads_per_block = 256
        blocks_per_grid = (coords.shape[0] + threads_per_block - 1) // threads_per_block

        numba_gamma_kernel[blocks_per_grid, threads_per_block](
            ref_gpu, eval_gpu, gamma_gpu, coords_gpu, spacing_gpu, criteria_gpu
        )

        # Copy result back
        gamma_map = gamma_gpu.copy_to_host()

        # Calculate statistics
        valid_gamma = gamma_map[analysis_mask]
        valid_gamma = valid_gamma[np.isfinite(valid_gamma)]

        if len(valid_gamma) > 0:
            pass_rate = np.sum(valid_gamma <= 1.0) / len(valid_gamma) * 100.0
            mean_gamma = np.mean(valid_gamma)
            max_gamma = np.max(valid_gamma)
        else:
            pass_rate = 0.0
            mean_gamma = float("inf")
            max_gamma = float("inf")

        memory_usage_mb = (reference_dose.nbytes * 3) / (1024**2)  # Approximate

        return GPUGammaResult(
            gamma_map=gamma_map,
            pass_rate=pass_rate,
            mean_gamma=mean_gamma,
            max_gamma=max_gamma,
            calculation_time=0.0,
            gpu_backend="numba_cuda",
            memory_usage_mb=memory_usage_mb,
        )

    def _calculate_gamma_opencl(
        self,
        reference_dose: np.ndarray,
        evaluated_dose: np.ndarray,
        spacing: Tuple[float, float, float],
        progress_callback: Optional[Callable],
    ) -> GPUGammaResult:
        """Calculate gamma using OpenCL."""

        # OpenCL kernel source
        kernel_source = """
        __kernel void gamma_3d_opencl(
            __global const float* ref_dose,
            __global const float* eval_dose,
            __global float* gamma_map,
            __global const int* coords,
            int num_points,
            int nz, int ny, int nx,
            float spacing_z, float spacing_y, float spacing_x,
            float dose_criterion, float distance_criterion
        ) {
            int idx = get_global_id(0);
            if (idx >= num_points) return;

            int z = coords[idx * 3 + 0];
            int y = coords[idx * 3 + 1];
            int x = coords[idx * 3 + 2];

            float ref_val = ref_dose[z * ny * nx + y * nx + x];
            float min_gamma = 10.0f;

            for (int dz = -5; dz <= 5; dz++) {
                for (int dy = -5; dy <= 5; dy++) {
                    for (int dx = -5; dx <= 5; dx++) {
                        int nz_new = z + dz;
                        int ny_new = y + dy;
                        int nx_new = x + dx;

                        if (nz_new >= 0 && nz_new < nz &&
                            ny_new >= 0 && ny_new < ny &&
                            nx_new >= 0 && nx_new < nx) {

                            float eval_val = eval_dose[nz_new * ny * nx + ny_new * nx + nx_new];

                            float dose_diff = fabs(ref_val - eval_val);
                            float dose_term = dose_diff / (dose_criterion * ref_val / 100.0f);

                            float dist_z = dz * spacing_z;
                            float dist_y = dy * spacing_y;
                            float dist_x = dx * spacing_x;
                            float distance = sqrt(dist_z*dist_z + dist_y*dist_y + dist_x*dist_x);
                            float dist_term = distance / distance_criterion;

                            float gamma_val = sqrt(dose_term*dose_term + dist_term*dist_term);
                            if (gamma_val < min_gamma) {
                                min_gamma = gamma_val;
                            }
                        }
                    }
                }
            }

            gamma_map[z * ny * nx + y * nx + x] = min_gamma;
        }
        """

        # Prepare data
        dose_threshold = np.max(reference_dose) * (
            self.settings.dose_threshold_percent / 100.0
        )
        analysis_mask = reference_dose >= dose_threshold
        coords = np.column_stack(np.where(analysis_mask)).flatten().astype(np.int32)

        # Create OpenCL buffers
        queue = cl.CommandQueue(self.gpu_context)

        ref_buffer = cl.Buffer(
            self.gpu_context,
            cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
            hostbuf=reference_dose.astype(np.float32),
        )
        eval_buffer = cl.Buffer(
            self.gpu_context,
            cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
            hostbuf=evaluated_dose.astype(np.float32),
        )
        gamma_buffer = cl.Buffer(
            self.gpu_context, cl.mem_flags.WRITE_ONLY, reference_dose.nbytes
        )
        coords_buffer = cl.Buffer(
            self.gpu_context,
            cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
            hostbuf=coords,
        )

        # Build and execute kernel
        program = cl.Program(self.gpu_context, kernel_source).build()

        num_points = len(coords) // 3
        program.gamma_3d_opencl(
            queue,
            (num_points,),
            None,
            ref_buffer,
            eval_buffer,
            gamma_buffer,
            coords_buffer,
            np.int32(num_points),
            np.int32(reference_dose.shape[0]),
            np.int32(reference_dose.shape[1]),
            np.int32(reference_dose.shape[2]),
            np.float32(spacing[2]),
            np.float32(spacing[1]),
            np.float32(spacing[0]),
            np.float32(self.settings.dose_percent),
            np.float32(self.settings.distance_mm),
        )

        # Read result
        gamma_map = np.empty_like(reference_dose, dtype=np.float32)
        cl.enqueue_copy(queue, gamma_map, gamma_buffer)

        # Calculate statistics
        valid_gamma = gamma_map[analysis_mask]
        valid_gamma = valid_gamma[np.isfinite(valid_gamma)]

        if len(valid_gamma) > 0:
            pass_rate = np.sum(valid_gamma <= 1.0) / len(valid_gamma) * 100.0
            mean_gamma = np.mean(valid_gamma)
            max_gamma = np.max(valid_gamma)
        else:
            pass_rate = 0.0
            mean_gamma = float("inf")
            max_gamma = float("inf")

        memory_usage_mb = (reference_dose.nbytes * 3) / (1024**2)

        return GPUGammaResult(
            gamma_map=gamma_map,
            pass_rate=pass_rate,
            mean_gamma=mean_gamma,
            max_gamma=max_gamma,
            calculation_time=0.0,
            gpu_backend="opencl",
            memory_usage_mb=memory_usage_mb,
        )

    def _calculate_gamma_cpu_fallback(
        self,
        reference_dose: np.ndarray,
        evaluated_dose: np.ndarray,
        spacing: Tuple[float, float, float],
        progress_callback: Optional[Callable],
    ) -> GPUGammaResult:
        """CPU fallback for gamma calculation."""

        # Import regular gamma analysis as fallback
        try:
            from .gamma_analysis import calculate_gamma_3d

            result = calculate_gamma_3d(
                reference_dose=reference_dose,
                evaluated_dose=evaluated_dose,
                distance_mm=self.settings.distance_mm,
                dose_percent=self.settings.dose_percent,
                dose_threshold_percent=self.settings.dose_threshold_percent,
                spacing=spacing,
            )

            return GPUGammaResult(
                gamma_map=result.gamma_map,
                pass_rate=result.pass_rate,
                mean_gamma=result.mean_gamma,
                max_gamma=result.max_gamma,
                calculation_time=result.calculation_time,
                gpu_backend="cpu_fallback",
                memory_usage_mb=reference_dose.nbytes / (1024**2),
            )

        except ImportError:
            # Final fallback - basic CPU implementation
            logger.warning("Using basic CPU fallback for gamma analysis")

            gamma_map = np.full_like(reference_dose, 10.0)
            dose_threshold = np.max(reference_dose) * (
                self.settings.dose_threshold_percent / 100.0
            )
            analysis_mask = reference_dose >= dose_threshold

            # Very basic gamma calculation
            dose_diff = np.abs(reference_dose - evaluated_dose)
            dose_term = dose_diff / (
                self.settings.dose_percent * reference_dose / 100.0
            )
            gamma_map[analysis_mask] = dose_term[
                analysis_mask
            ]  # Simplified - no distance component

            valid_gamma = gamma_map[analysis_mask]
            pass_rate = (
                np.sum(valid_gamma <= 1.0) / len(valid_gamma) * 100.0
                if len(valid_gamma) > 0
                else 0.0
            )

            return GPUGammaResult(
                gamma_map=gamma_map,
                pass_rate=pass_rate,
                mean_gamma=np.mean(valid_gamma) if len(valid_gamma) > 0 else 0.0,
                max_gamma=np.max(valid_gamma) if len(valid_gamma) > 0 else 0.0,
                calculation_time=0.0,
                gpu_backend="basic_cpu",
                memory_usage_mb=reference_dose.nbytes / (1024**2),
            )


def create_gpu_gamma_analyzer(
    backend: str = "auto", max_memory_gb: float = 4.0, fast_mode: bool = True
) -> GPUGammaAnalyzer:
    """Factory function để tạo GPU gamma analyzer."""

    settings = GPUGammaSettings(
        preferred_backend=backend,
        max_memory_usage_gb=max_memory_gb,
        use_fast_mode=fast_mode,
    )

    return GPUGammaAnalyzer(settings)


def benchmark_gpu_gamma_performance() -> Dict[str, Any]:
    """Benchmark hiệu suất của các GPU backend."""

    # Create test data
    test_size = (64, 64, 32)
    ref_dose = np.random.rand(*test_size) * 60.0
    eval_dose = ref_dose + np.random.normal(0, 2.0, test_size)

    results = {}

    # Test different backends
    backends = ["cupy", "numba", "opencl", "cpu_fallback"]

    for backend in backends:
        try:
            analyzer = create_gpu_gamma_analyzer(backend=backend)

            start_time = time.time()
            result = analyzer.calculate_gamma_gpu(ref_dose, eval_dose)
            end_time = time.time()

            results[backend] = {
                "calculation_time": end_time - start_time,
                "pass_rate": result.pass_rate,
                "voxels_per_second": result.voxels_per_second,
                "memory_usage_mb": result.memory_usage_mb,
                "available": True,
            }

        except Exception as e:
            results[backend] = {"available": False, "error": str(e)}

    return results


if __name__ == "__main__":
    # Test GPU gamma analysis
    print("Testing GPU Gamma Analysis...")

    # Run benchmark
    benchmark_results = benchmark_gpu_gamma_performance()

    print("\nBenchmark Results:")
    for backend, result in benchmark_results.items():
        if result.get("available", False):
            print(
                f"{backend:12}: {result['calculation_time']:.3f}s, "
                f"{result['voxels_per_second'] / 1e6:.2f} MVoxels/s, "
                f"Pass Rate: {result['pass_rate']:.1f}%"
            )
        else:
            print(
                f"{backend:12}: Not available - {result.get('error', 'Unknown error')}"
            )
