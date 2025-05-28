"""
GPU-accelerated Dose Calculation for QuangTPS
High-performance dose calculation using CUDA, OpenCL and parallel computing.
"""

import numpy as np
import logging
import time
import math
import concurrent.futures
from typing import Optional, Tuple, Dict, Any, List, Union
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing
import threading

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
    from numba import cuda, njit, prange

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    cuda = None
    njit = None
    prange = None

# Scientific computing libraries
try:
    from scipy import ndimage
    from scipy.spatial.distance import cdist

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

logger = logging.getLogger(__name__)


@dataclass
class GPUDoseSettings:
    """Cài đặt cho GPU dose calculation."""

    # Algorithm settings
    algorithm_type: str = "collapsed_cone"  # pencil_beam, collapsed_cone, monte_carlo
    accuracy_level: str = "high"  # low, medium, high, ultra
    max_calculation_time: float = 300.0  # seconds

    # GPU settings
    preferred_backend: str = "auto"  # auto, cupy, opencl, numba
    max_memory_usage_gb: float = 8.0
    use_gpu_acceleration: bool = True
    fallback_to_cpu: bool = True

    # Performance settings
    chunk_size: int = 2000000
    num_cpu_cores: int = -1  # -1 = auto detect
    use_multiprocessing: bool = True

    # Quality settings
    voxel_size: Tuple[float, float, float] = (2.0, 2.0, 2.0)
    dose_grid_padding: int = 10
    interpolation_method: str = "linear"

    # Monte Carlo specific
    num_histories: int = 1000000
    statistical_uncertainty: float = 0.02
    variance_reduction: bool = True


@dataclass
class GPUDoseResult:
    """Kết quả GPU dose calculation."""

    dose_grid: np.ndarray
    statistical_uncertainty: Optional[np.ndarray]
    calculation_time: float
    algorithm_used: str
    gpu_backend: str
    memory_usage_mb: float

    # Performance metrics
    voxels_per_second: float = 0.0
    histories_per_second: float = 0.0
    speedup_factor: float = 1.0

    # Quality metrics
    mean_dose: float = 0.0
    max_dose: float = 0.0
    dose_volume_histogram: Optional[Dict] = None


class GPUDoseCalculator:
    """High-performance GPU dose calculator."""

    def __init__(self, settings: Optional[GPUDoseSettings] = None):
        """Initialize GPU dose calculator."""
        self.settings = settings or GPUDoseSettings()
        self.gpu_backend = None
        self.gpu_context = None
        self.gpu_device = None
        self.cpu_cores = self._get_cpu_cores()

        # Initialize GPU backend
        if self.settings.use_gpu_acceleration:
            self._initialize_gpu()
        else:
            self.gpu_backend = "cpu_only"

        logger.info(f"GPU Dose Calculator initialized with backend: {self.gpu_backend}")
        logger.info(
            f"Using {self.cpu_cores} CPU cores for fallback/parallel computation"
        )

    def _get_cpu_cores(self) -> int:
        """Get number of CPU cores to use."""
        if self.settings.num_cpu_cores == -1:
            return max(1, multiprocessing.cpu_count() - 1)  # Reserve 1 core for OS
        else:
            return max(1, min(self.settings.num_cpu_cores, multiprocessing.cpu_count()))

    def _initialize_gpu(self):
        """Initialize GPU backend."""
        if self.settings.preferred_backend == "auto":
            # Try backends in order of preference for dose calculation
            if self._try_cupy():
                self.gpu_backend = "cupy"
            elif self._try_numba():
                self.gpu_backend = "numba"
            elif self._try_opencl():
                self.gpu_backend = "opencl"
            else:
                self.gpu_backend = "cpu_fallback"
                logger.warning("No GPU backend available for dose calculation")
        else:
            # Try specific backend
            backend_methods = {
                "cupy": self._try_cupy,
                "opencl": self._try_opencl,
                "numba": self._try_numba,
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
            # Test GPU availability and memory
            device_count = cp.cuda.runtime.getDeviceCount()
            if device_count == 0:
                return False

            self.gpu_device = cp.cuda.Device()
            memory_info = cp.cuda.runtime.memGetInfo()
            available_memory_gb = memory_info[0] / (1024**3)

            if available_memory_gb < 1.0:  # Need at least 1GB for dose calculation
                logger.warning(f"Insufficient GPU memory: {available_memory_gb:.1f}GB")
                return False

            # Test computation capability
            test_array = cp.random.rand(1000, 1000, dtype=np.float32)
            result = cp.sum(test_array)
            del test_array, result

            logger.info(
                f"CuPy backend initialized - GPU memory: {available_memory_gb:.1f}GB"
            )
            return True

        except Exception as e:
            logger.warning(f"CuPy initialization failed: {e}")
            return False

    def _try_numba(self) -> bool:
        """Try to initialize Numba backend."""
        if not HAS_NUMBA:
            return False

        try:
            if cuda.is_available():
                cuda.select_device(0)
                device = cuda.get_current_device()
                logger.info(f"Numba CUDA initialized - Device: {device.name}")
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
            platforms = cl.get_platforms()
            for platform in platforms:
                devices = platform.get_devices(cl.device_type.GPU)
                if devices:
                    self.gpu_context = cl.Context(devices=[devices[0]])
                    self.gpu_device = devices[0]
                    logger.info(f"OpenCL initialized - Device: {devices[0].name}")
                    return True
            return False

        except Exception as e:
            logger.warning(f"OpenCL initialization failed: {e}")
            return False

    def calculate_dose_gpu(
        self,
        beam_data: Dict[str, Any],
        patient_geometry: np.ndarray,
        dose_grid_shape: Tuple[int, int, int],
        spacing: Tuple[float, float, float],
        progress_callback: Optional[callable] = None,
    ) -> GPUDoseResult:
        """Calculate dose using GPU acceleration."""

        start_time = time.time()

        # Validate inputs
        if not isinstance(patient_geometry, np.ndarray):
            raise ValueError("Patient geometry must be numpy array")

        # Choose calculation method based on algorithm and backend
        if self.settings.algorithm_type == "monte_carlo":
            result = self._calculate_monte_carlo_gpu(
                beam_data, patient_geometry, dose_grid_shape, spacing, progress_callback
            )
        elif self.settings.algorithm_type == "collapsed_cone":
            result = self._calculate_collapsed_cone_gpu(
                beam_data, patient_geometry, dose_grid_shape, spacing, progress_callback
            )
        elif self.settings.algorithm_type == "pencil_beam":
            result = self._calculate_pencil_beam_gpu(
                beam_data, patient_geometry, dose_grid_shape, spacing, progress_callback
            )
        else:
            raise ValueError(f"Unknown algorithm type: {self.settings.algorithm_type}")

        # Calculate performance metrics
        total_voxels = np.prod(dose_grid_shape)
        calculation_time = time.time() - start_time
        voxels_per_second = (
            total_voxels / calculation_time if calculation_time > 0 else 0
        )

        result.calculation_time = calculation_time
        result.voxels_per_second = voxels_per_second
        result.gpu_backend = self.gpu_backend

        # Calculate dose statistics
        if result.dose_grid is not None:
            result.mean_dose = np.mean(result.dose_grid)
            result.max_dose = np.max(result.dose_grid)

        logger.info(f"GPU dose calculation completed in {calculation_time:.2f}s")
        logger.info(f"Performance: {voxels_per_second / 1e6:.2f} M voxels/sec")

        return result

    def _calculate_monte_carlo_gpu(
        self,
        beam_data: Dict[str, Any],
        patient_geometry: np.ndarray,
        dose_grid_shape: Tuple[int, int, int],
        spacing: Tuple[float, float, float],
        progress_callback: Optional[callable],
    ) -> GPUDoseResult:
        """Calculate dose using GPU-accelerated Monte Carlo."""

        if self.gpu_backend == "cupy":
            return self._monte_carlo_cupy(
                beam_data, patient_geometry, dose_grid_shape, spacing, progress_callback
            )
        elif self.gpu_backend == "numba":
            return self._monte_carlo_numba(
                beam_data, patient_geometry, dose_grid_shape, spacing, progress_callback
            )
        else:
            return self._monte_carlo_cpu_parallel(
                beam_data, patient_geometry, dose_grid_shape, spacing, progress_callback
            )

    def _monte_carlo_cupy(
        self,
        beam_data: Dict[str, Any],
        patient_geometry: np.ndarray,
        dose_grid_shape: Tuple[int, int, int],
        spacing: Tuple[float, float, float],
        progress_callback: Optional[callable],
    ) -> GPUDoseResult:
        """CuPy implementation of Monte Carlo dose calculation."""

        import cupy as cp

        start_time = time.time()
        num_histories = self.settings.num_histories

        if progress_callback:
            progress_callback(
                10, f"Starting Monte Carlo with {num_histories} histories"
            )

        # Monte Carlo kernel
        mc_kernel = cp.RawKernel(
            r"""
        extern "C" __global__
        void monte_carlo_transport(
            float* dose_grid,
            float* uncertainty_grid,
            float* geometry,
            float* beam_params,
            int nx, int ny, int nz,
            float dx, float dy, float dz,
            int num_histories,
            int seed
        ) {
            int tid = blockIdx.x * blockDim.x + threadIdx.x;
            if (tid >= num_histories) return;

            // Initialize random state
            curandState state;
            curand_init(seed + tid, 0, 0, &state);

            // Sample energy from spectrum
            float energy = beam_params[0]; // MV

            // Sample initial position and direction
            float x = (curand_uniform(&state) - 0.5f) * 20.0f; // Field size
            float y = (curand_uniform(&state) - 0.5f) * 20.0f;
            float z = 0.0f; // Source position

            float dx_dir = 0.0f; // Parallel beam
            float dy_dir = 0.0f;
            float dz_dir = 1.0f;

            // Transport particle
            float dose_deposited = 0.0f;
            float path_length = 0.0f;

            for (int step = 0; step < 1000; ++step) {
                // Move particle
                x += dx_dir * dz;
                y += dy_dir * dz;
                z += dz_dir * dz;

                // Check bounds
                if (z < 0 || z >= nz * dz) break;

                int ix = (int)((x + nx * dx / 2) / dx);
                int iy = (int)((y + ny * dy / 2) / dy);
                int iz = (int)(z / dz);

                if (ix >= 0 && ix < nx && iy >= 0 && iy < ny && iz >= 0 && iz < nz) {
                    int idx = iz * ny * nx + iy * nx + ix;
                    float density = geometry[idx];

                    // Simple dose deposition
                    float mu = 0.05f; // Linear attenuation coefficient
                    float dose_step = mu * density * energy * expf(-mu * path_length);
                    atomicAdd(&dose_grid[idx], dose_step);

                    path_length += mu * density * dz;
                    energy *= expf(-mu * density * dz);

                    if (energy < 0.01f) break; // Energy cutoff
                }
            }
        }
        """,
            "monte_carlo_transport",
        )

        # Run Monte Carlo simulation in batches
        batch_size = min(10000, num_histories)
        dose_grid = cp.zeros(dose_grid_shape, dtype=np.float32)
        uncertainty_grid = cp.zeros(dose_grid_shape, dtype=np.float32)
        geometry_gpu = cp.asarray(patient_geometry.astype(np.float32))

        for batch in range(0, num_histories, batch_size):
            current_batch_size = min(batch_size, num_histories - batch)

            # Setup beam parameters
            beam_params_gpu = cp.array(
                [
                    beam_data.get("energy", 6.0),  # MV
                    beam_data.get("fluence", 1.0),
                    beam_data.get("gantry_angle", 0.0),
                    beam_data.get("collimator_angle", 0.0),
                ],
                dtype=np.float32,
            )

            # Launch kernel
            block_size = (256,)
            grid_size = ((current_batch_size + block_size[0] - 1) // block_size[0],)

            mc_kernel(
                (grid_size,),
                (block_size,),
                (
                    dose_grid,
                    uncertainty_grid,
                    geometry_gpu,
                    beam_params_gpu,
                    dose_grid_shape[2],
                    dose_grid_shape[1],
                    dose_grid_shape[0],
                    spacing[2],
                    spacing[1],
                    spacing[0],
                    current_batch_size,
                    batch * 12345,
                ),
            )

            if progress_callback and batch % (batch_size * 5) == 0:
                progress = 10 + int((batch / num_histories) * 80)
                progress_callback(
                    progress, f"Processed {batch}/{num_histories} histories"
                )

        # Calculate statistical uncertainty
        dose_grid /= num_histories
        uncertainty_grid = cp.sqrt(uncertainty_grid / num_histories - dose_grid**2)

        # Convert back to CPU
        dose_grid_cpu = cp.asnumpy(dose_grid)
        uncertainty_grid_cpu = cp.asnumpy(uncertainty_grid)

        # Calculate quality metrics
        mean_dose = float(cp.mean(dose_grid))
        max_dose = float(cp.max(dose_grid))

        if progress_callback:
            progress_callback(100, "Monte Carlo calculation completed")

        # Estimate memory usage
        memory_usage_mb = (
            geometry_gpu.nbytes + dose_grid.nbytes + uncertainty_grid.nbytes
        ) / (1024**2)

        return GPUDoseResult(
            dose_grid=dose_grid_cpu,
            statistical_uncertainty=uncertainty_grid_cpu,
            calculation_time=time.time() - start_time,
            algorithm_used="monte_carlo_cupy",
            gpu_backend="cupy",
            memory_usage_mb=memory_usage_mb,
            histories_per_second=num_histories / 1.0,  # Placeholder
        )

    def _monte_carlo_numba(
        self,
        beam_data: Dict[str, Any],
        patient_geometry: np.ndarray,
        dose_grid_shape: Tuple[int, int, int],
        spacing: Tuple[float, float, float],
        progress_callback: Optional[callable],
    ) -> GPUDoseResult:
        """Numba CUDA implementation of Monte Carlo dose calculation."""

        try:
            from numba import cuda
            import numpy as np

            @cuda.jit
            def monte_carlo_numba_kernel(
                dose_grid, geometry, beam_params, spacing_vals, num_histories
            ):
                tid = cuda.grid(ndim=1)
                if tid >= num_histories:
                    return

                # Particle initialization
                x, y, z = (
                    beam_params[0] + tid % 10 * spacing_vals[0],
                    beam_params[1] + (tid // 10) % 10 * spacing_vals[1],
                    beam_params[2],
                )
                energy = beam_params[3]

                # Transport simulation
                for step in range(100):
                    # Convert to grid coordinates
                    ix = int(x / spacing_vals[0])
                    iy = int(y / spacing_vals[1])
                    iz = int(z / spacing_vals[2])

                    if (
                        ix >= 0
                        and iy >= 0
                        and iz >= 0
                        and ix < dose_grid.shape[2]
                        and iy < dose_grid.shape[1]
                        and iz < dose_grid.shape[0]
                    ):
                        density = geometry[iz, iy, ix]
                        dose_step = energy * 0.01 * density
                        cuda.atomic.add(dose_grid, (iz, iy, ix), dose_step)
                        energy *= 0.99  # Energy loss

                        if energy < 0.1:
                            break

                    # Update position
                    x += 0.5
                    y += 0.5
                    z += 1.0

            start_time = time.time()
            num_histories = self.settings.num_histories

            # Prepare data for GPU
            dose_grid = np.zeros(dose_grid_shape, dtype=np.float32)
            geometry_gpu = patient_geometry.astype(np.float32)
            beam_params = np.array(
                [beam_data.get("energy", 6.0), beam_data.get("fluence", 1.0)],
                dtype=np.float32,
            )
            spacing_vals = np.array(spacing, dtype=np.float32)

            # Transfer to GPU
            dose_grid_gpu = cuda.to_device(dose_grid)
            geometry_gpu = cuda.to_device(geometry_gpu)
            beam_params_gpu = cuda.to_device(beam_params)
            spacing_gpu = cuda.to_device(spacing_vals)

            # Launch kernel
            block_size = 256
            grid_size = (num_histories + block_size - 1) // block_size

            monte_carlo_numba_kernel[grid_size, block_size](
                dose_grid_gpu, geometry_gpu, beam_params_gpu, spacing_gpu, num_histories
            )

            # Copy back to CPU
            dose_grid_cpu = dose_grid_gpu.copy_to_host()

            # Normalize
            dose_grid_cpu /= num_histories

            return GPUDoseResult(
                dose_grid=dose_grid_cpu,
                statistical_uncertainty=None,
                calculation_time=time.time() - start_time,
                algorithm_used="monte_carlo_numba",
                gpu_backend="numba",
                memory_usage_mb=dose_grid_cpu.nbytes / (1024**2),
            )

        except Exception as e:
            logger.warning(f"Numba Monte Carlo failed: {e}, falling back to CPU")
            return self._monte_carlo_cpu_parallel(
                beam_data, patient_geometry, dose_grid_shape, spacing, progress_callback
            )

    def _monte_carlo_cpu_parallel(
        self,
        beam_data: Dict[str, Any],
        patient_geometry: np.ndarray,
        dose_grid_shape: Tuple[int, int, int],
        spacing: Tuple[float, float, float],
        progress_callback: Optional[callable],
    ) -> GPUDoseResult:
        """CPU parallel implementation of Monte Carlo dose calculation."""

        def monte_carlo_worker(args):
            start_history, end_history, geometry, beam_params = args
            local_dose = np.zeros(dose_grid_shape, dtype=np.float32)

            energy = beam_params["energy"]

            for history in range(start_history, end_history):
                # Simple Monte Carlo simulation
                x, y, z = 10.0, 10.0, 0.0  # Start position
                current_energy = energy

                nz, ny, nx = dose_grid_shape
                for step in range(100):  # Max steps per history
                    iz = int(z / spacing[0])
                    iy = int(y / spacing[1])
                    ix = int(x / spacing[2])

                    if 0 <= iz < nz and 0 <= iy < ny and 0 <= ix < nx:
                        density = geometry[iz, iy, ix]
                        dose_step = current_energy * 0.01 * density
                        local_dose[iz, iy, ix] += dose_step
                        current_energy *= 0.99

                        if current_energy < 0.1:
                            break

                    z += spacing[0]

            return local_dose

        start_time = time.time()
        num_histories = self.settings.num_histories

        # Split histories among workers
        histories_per_worker = num_histories // self.cpu_cores
        tasks = []

        for i in range(self.cpu_cores):
            start_history = i * histories_per_worker
            end_history = (
                (i + 1) * histories_per_worker
                if i < self.cpu_cores - 1
                else num_histories
            )
            tasks.append((start_history, end_history, patient_geometry, beam_data))

        # Execute in parallel
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=self.cpu_cores
        ) as executor:
            results = list(executor.map(monte_carlo_worker, tasks))

        # Combine results
        final_dose = np.sum(results, axis=0)
        final_dose /= num_histories

        return GPUDoseResult(
            dose_grid=final_dose,
            statistical_uncertainty=None,
            calculation_time=time.time() - start_time,
            algorithm_used="monte_carlo_cpu_parallel",
            gpu_backend="cpu_parallel",
            memory_usage_mb=final_dose.nbytes / (1024**2),
        )

    def _calculate_collapsed_cone_gpu(
        self,
        beam_data: Dict[str, Any],
        patient_geometry: np.ndarray,
        dose_grid_shape: Tuple[int, int, int],
        spacing: Tuple[float, float, float],
        progress_callback: Optional[callable],
    ) -> GPUDoseResult:
        """Calculate dose using GPU-accelerated Collapsed Cone."""

        if self.gpu_backend == "cupy":
            return self._collapsed_cone_cupy(
                beam_data, patient_geometry, dose_grid_shape, spacing, progress_callback
            )
        else:
            return self._collapsed_cone_cpu_parallel(
                beam_data, patient_geometry, dose_grid_shape, spacing, progress_callback
            )

    def _collapsed_cone_cupy(
        self,
        beam_data: Dict[str, Any],
        patient_geometry: np.ndarray,
        dose_grid_shape: Tuple[int, int, int],
        spacing: Tuple[float, float, float],
        progress_callback: Optional[callable],
    ) -> GPUDoseResult:
        """CuPy implementation of Collapsed Cone algorithm."""

        # Transfer data to GPU
        geometry_gpu = cp.asarray(patient_geometry.astype(np.float32))
        dose_grid = cp.zeros(dose_grid_shape, dtype=np.float32)

        if progress_callback:
            progress_callback(20, "Setting up collapsed cone kernels")

        # Collapsed cone kernel
        cc_kernel = cp.RawKernel(
            r"""
        extern "C" __global__
        void collapsed_cone_kernel(
            float* dose_grid,
            const float* geometry,
            const float* fluence_map,
            int grid_x, int grid_y, int grid_z,
            float spacing_x, float spacing_y, float spacing_z,
            float sad, float energy
        ) {
            int gx = blockIdx.x * blockDim.x + threadIdx.x;
            int gy = blockIdx.y * blockDim.y + threadIdx.y;

            if (gx >= grid_x || gy >= grid_y) return;

            // Calculate ray through this pixel
            float source_x = gx * spacing_x - (grid_x * spacing_x) / 2.0f;
            float source_y = gy * spacing_y - (grid_y * spacing_y) / 2.0f;
            float source_z = -sad;

            // Ray direction
            float ray_length = sqrtf(source_x*source_x + source_y*source_y + sad*sad);
            float dx = source_x / ray_length;
            float dy = source_y / ray_length;
            float dz = sad / ray_length;

            // Step through geometry
            float step_size = min(spacing_x, min(spacing_y, spacing_z)) * 0.5f;
            float total_attenuation = 0.0f;

            for (float t = 0; t < grid_z * spacing_z; t += step_size) {
                float x = source_x + dx * t;
                float y = source_y + dy * t;
                float z = dz * t;

                // Convert to grid coordinates
                int ix = (int)((x + grid_x * spacing_x / 2.0f) / spacing_x);
                int iy = (int)((y + grid_y * spacing_y / 2.0f) / spacing_y);
                int iz = (int)(z / spacing_z);

                if (ix >= 0 && ix < grid_x && iy >= 0 && iy < grid_y && iz >= 0 && iz < grid_z) {
                    int geo_idx = iz * grid_y * grid_x + iy * grid_x + ix;
                    float density = geometry[geo_idx];

                    // Calculate attenuation
                    float mu = 0.1f * density * energy / 6.0f; // Simplified
                    total_attenuation += mu * step_size;

                    // Calculate dose deposition
                    float fluence = fluence_map[gy * grid_x + gx];
                    float attenuated_fluence = fluence * expf(-total_attenuation);
                    float dose_rate = mu * attenuated_fluence * density;

                    atomicAdd(&dose_grid[geo_idx], dose_rate * step_size);
                }
            }
        }
        """,
            "collapsed_cone_kernel",
        )

        # Create fluence map
        fluence_shape = (dose_grid_shape[1], dose_grid_shape[0])  # y, x
        fluence_map = cp.ones(fluence_shape, dtype=np.float32)

        # Apply beam shape if available
        if "field_size" in beam_data:
            field_size = beam_data["field_size"]
            center_x, center_y = fluence_shape[1] // 2, fluence_shape[0] // 2
            y_coords, x_coords = cp.mgrid[: fluence_shape[0], : fluence_shape[1]]

            # Create rectangular field
            field_x = field_size[0] / spacing[0]
            field_y = field_size[1] / spacing[1]

            mask = (cp.abs(x_coords - center_x) <= field_x / 2) & (
                cp.abs(y_coords - center_y) <= field_y / 2
            )
            fluence_map = mask.astype(np.float32)

        # Launch collapsed cone kernel
        block_size = (16, 16)
        grid_size_x = (dose_grid_shape[0] + block_size[0] - 1) // block_size[0]
        grid_size_y = (dose_grid_shape[1] + block_size[1] - 1) // block_size[1]

        if progress_callback:
            progress_callback(50, "Running collapsed cone calculation")

        cc_kernel(
            (grid_size_x, grid_size_y),
            block_size,
            (
                dose_grid,
                geometry_gpu,
                fluence_map,
                dose_grid_shape[0],
                dose_grid_shape[1],
                dose_grid_shape[2],
                spacing[0],
                spacing[1],
                spacing[2],
                beam_data.get("sad", 100.0),  # Source-axis distance
                beam_data.get("energy", 6.0),
            ),  # Energy in MV
        )

        cp.cuda.Device().synchronize()

        if progress_callback:
            progress_callback(90, "Finalizing dose calculation")

        # Transfer result back to CPU
        dose_cpu = cp.asnumpy(dose_grid)

        # Estimate memory usage
        memory_usage_mb = (
            geometry_gpu.nbytes + dose_grid.nbytes + fluence_map.nbytes
        ) / (1024**2)

        return GPUDoseResult(
            dose_grid=dose_cpu,
            statistical_uncertainty=None,
            calculation_time=0.0,
            algorithm_used="collapsed_cone_cupy",
            gpu_backend="cupy",
            memory_usage_mb=memory_usage_mb,
        )

    def _calculate_pencil_beam_gpu(
        self,
        beam_data: Dict[str, Any],
        patient_geometry: np.ndarray,
        dose_grid_shape: Tuple[int, int, int],
        spacing: Tuple[float, float, float],
        progress_callback: Optional[callable],
    ) -> GPUDoseResult:
        """Calculate dose using GPU-accelerated Pencil Beam."""

        if self.gpu_backend in ["cupy", "numba"]:
            return self._pencil_beam_gpu_impl(
                beam_data, patient_geometry, dose_grid_shape, spacing, progress_callback
            )
        else:
            return self._pencil_beam_cpu_parallel(
                beam_data, patient_geometry, dose_grid_shape, spacing, progress_callback
            )

    def _pencil_beam_cpu_parallel(
        self,
        beam_data: Dict[str, Any],
        patient_geometry: np.ndarray,
        dose_grid_shape: Tuple[int, int, int],
        spacing: Tuple[float, float, float],
        progress_callback: Optional[callable],
    ) -> GPUDoseResult:
        """CPU parallel implementation for pencil beam."""

        dose_grid = np.zeros(dose_grid_shape, dtype=np.float32)

        # Define pencil beam calculation function
        def calculate_pencil_beam_chunk(args):
            """Calculate dose for a chunk of beamlets."""
            chunk_start, chunk_end, geometry_chunk, beam_params = args

            local_dose = np.zeros_like(geometry_chunk, dtype=np.float32)

            # Simple pencil beam implementation
            for i in range(chunk_start, chunk_end):
                for j in range(geometry_chunk.shape[1]):
                    # Calculate depth dose
                    depth = 0.0
                    for k in range(geometry_chunk.shape[2]):
                        if geometry_chunk[i, j, k] > 0:
                            # Simplified PDD calculation
                            pdd = np.exp(-0.01 * depth) * geometry_chunk[i, j, k]
                            local_dose[i, j, k] += pdd * beam_params["fluence"]
                            depth += spacing[2]

            return local_dose

        # Split work into chunks for parallel processing
        chunk_size = max(1, dose_grid_shape[0] // self.cpu_cores)
        chunks = []

        for i in range(0, dose_grid_shape[0], chunk_size):
            end_i = min(i + chunk_size, dose_grid_shape[0])
            chunks.append((i, end_i, patient_geometry[i:end_i], beam_data))

        # Execute in parallel
        if self.settings.use_multiprocessing and len(chunks) > 1:
            with ProcessPoolExecutor(max_workers=self.cpu_cores) as executor:
                results = list(executor.map(calculate_pencil_beam_chunk, chunks))
        else:
            with ThreadPoolExecutor(max_workers=self.cpu_cores) as executor:
                results = list(executor.map(calculate_pencil_beam_chunk, chunks))

        # Combine results
        chunk_start = 0
        for result in results:
            chunk_end = chunk_start + result.shape[0]
            dose_grid[chunk_start:chunk_end] += result
            chunk_start = chunk_end

        return GPUDoseResult(
            dose_grid=dose_grid,
            statistical_uncertainty=None,
            calculation_time=0.0,
            algorithm_used="pencil_beam_cpu_parallel",
            gpu_backend="cpu_parallel",
            memory_usage_mb=patient_geometry.nbytes / (1024**2),
        )

    def _pencil_beam_gpu_impl(
        self,
        beam_data: Dict[str, Any],
        patient_geometry: np.ndarray,
        dose_grid_shape: Tuple[int, int, int],
        spacing: Tuple[float, float, float],
        progress_callback: Optional[callable],
    ) -> GPUDoseResult:
        """GPU implementation for Pencil Beam algorithm."""

        if self.gpu_backend == "cupy":
            return self._pencil_beam_cupy(
                beam_data, patient_geometry, dose_grid_shape, spacing, progress_callback
            )
        elif self.gpu_backend == "numba":
            return self._pencil_beam_numba(
                beam_data, patient_geometry, dose_grid_shape, spacing, progress_callback
            )
        else:
            return self._pencil_beam_cpu_parallel(
                beam_data, patient_geometry, dose_grid_shape, spacing, progress_callback
            )

    def _pencil_beam_cupy(
        self,
        beam_data: Dict[str, Any],
        patient_geometry: np.ndarray,
        dose_grid_shape: Tuple[int, int, int],
        spacing: Tuple[float, float, float],
        progress_callback: Optional[callable],
    ) -> GPUDoseResult:
        """CuPy implementation for Pencil Beam algorithm."""

        try:
            import cupy as cp

            start_time = time.time()

            # Transfer data to GPU
            geometry_gpu = cp.asarray(patient_geometry.astype(np.float32))
            dose_grid = cp.zeros(dose_grid_shape, dtype=np.float32)

            # Pencil beam parameters
            energy = beam_data.get("energy", 6.0)
            field_size = beam_data.get("field_size", (10.0, 10.0))

            nz, ny, nx = dose_grid_shape
            center_y, center_x = ny // 2, nx // 2

            # Create coordinate grids
            z_coords, y_coords, x_coords = cp.mgrid[0:nz, 0:ny, 0:nx]

            # Calculate field mask
            dy = (y_coords - center_y) * spacing[1]
            dx = (x_coords - center_x) * spacing[2]
            field_mask = (cp.abs(dy) <= field_size[1] / 2) & (
                cp.abs(dx) <= field_size[0] / 2
            )

            # Calculate depth
            depth = z_coords * spacing[0]

            # Simple PDD calculation
            pdd = cp.exp(-0.01 * depth)

            # Calculate dose
            dose_grid = energy * pdd * geometry_gpu * field_mask

            # Apply tissue heterogeneity corrections
            dose_grid *= geometry_gpu  # Density correction

            # Convert back to CPU
            dose_grid_cpu = cp.asnumpy(dose_grid)

            # Calculate quality metrics
            mean_dose = float(cp.mean(dose_grid))
            max_dose = float(cp.max(dose_grid))

            memory_usage_mb = (geometry_gpu.nbytes + dose_grid.nbytes) / (1024**2)

            return GPUDoseResult(
                dose_grid=dose_grid_cpu,
                statistical_uncertainty=None,
                calculation_time=time.time() - start_time,
                algorithm_used="pencil_beam_cupy",
                gpu_backend="cupy",
                memory_usage_mb=memory_usage_mb,
                mean_dose=mean_dose,
                max_dose=max_dose,
            )

        except Exception as e:
            logger.warning(f"CuPy Pencil Beam failed: {e}, falling back to CPU")
            return self._pencil_beam_cpu_parallel(
                beam_data, patient_geometry, dose_grid_shape, spacing, progress_callback
            )

    def _pencil_beam_numba(
        self,
        beam_data: Dict[str, Any],
        patient_geometry: np.ndarray,
        dose_grid_shape: Tuple[int, int, int],
        spacing: Tuple[float, float, float],
        progress_callback: Optional[callable],
    ) -> GPUDoseResult:
        """Numba CUDA implementation for Pencil Beam algorithm."""

        try:
            from numba import cuda

            @cuda.jit
            def pencil_beam_kernel(
                dose_grid, geometry, beam_params, spacing_vals, field_size
            ):
                iz, iy, ix = cuda.grid(ndim=3)
                if (
                    iz >= dose_grid.shape[0]
                    or iy >= dose_grid.shape[1]
                    or ix >= dose_grid.shape[2]
                ):
                    return

                # Get physical coordinates
                x = ix * spacing_vals[0]
                y = iy * spacing_vals[1]
                z = iz * spacing_vals[2]

                # Check if voxel is within beam field
                if x < field_size[0] and y < field_size[1]:
                    # Get density
                    density = geometry[iz, iy, ix]

                    # Simple pencil beam calculation
                    depth = z * density  # Effective depth
                    lateral_distance = math.sqrt(
                        (x - beam_params[0]) ** 2 + (y - beam_params[1]) ** 2
                    )

                    # Depth dose profile (exponential falloff)
                    depth_factor = math.exp(-0.01 * depth)

                    # Lateral profile (Gaussian)
                    lateral_factor = math.exp(
                        -0.5 * (lateral_distance / beam_params[2]) ** 2
                    )

                    # Calculate dose
                    dose_value = beam_params[3] * depth_factor * lateral_factor

                    # Atomic add to dose grid
                    cuda.atomic.add(dose_grid, (iz, iy, ix), dose_value)

            start_time = time.time()

            # Prepare data
            dose_grid = np.zeros(dose_grid_shape, dtype=np.float32)
            beam_params = np.array([beam_data.get("energy", 6.0)], dtype=np.float32)
            spacing_vals = np.array(spacing, dtype=np.float32)
            field_size = np.array(
                beam_data.get("field_size", (10.0, 10.0)), dtype=np.float32
            )

            # Transfer to GPU
            dose_grid_gpu = cuda.to_device(dose_grid)
            geometry_gpu = cuda.to_device(patient_geometry.astype(np.float32))
            beam_params_gpu = cuda.to_device(beam_params)
            spacing_gpu = cuda.to_device(spacing_vals)
            field_size_gpu = cuda.to_device(field_size)

            # Launch kernel
            block_size = (8, 8, 8)
            grid_size = tuple(
                (n + b - 1) // b for n, b in zip(dose_grid_shape, block_size)
            )

            pencil_beam_kernel[grid_size, block_size](
                dose_grid_gpu,
                geometry_gpu,
                beam_params_gpu,
                spacing_gpu,
                field_size_gpu,
            )

            # Copy back
            dose_grid_cpu = dose_grid_gpu.copy_to_host()

            return GPUDoseResult(
                dose_grid=dose_grid_cpu,
                statistical_uncertainty=None,
                calculation_time=time.time() - start_time,
                algorithm_used="pencil_beam_numba",
                gpu_backend="numba",
                memory_usage_mb=dose_grid_cpu.nbytes / (1024**2),
            )

        except Exception as e:
            logger.warning(f"Numba Pencil Beam failed: {e}, falling back to CPU")
            return self._pencil_beam_cpu_parallel(
                beam_data, patient_geometry, dose_grid_shape, spacing, progress_callback
            )

    def _collapsed_cone_cpu_parallel(
        self,
        beam_data: Dict[str, Any],
        patient_geometry: np.ndarray,
        dose_grid_shape: Tuple[int, int, int],
        spacing: Tuple[float, float, float],
        progress_callback: Optional[callable],
    ) -> GPUDoseResult:
        """CPU parallel implementation for Collapsed Cone algorithm."""

        def collapsed_cone_worker(args):
            z_start, z_end, geometry, beam_params = args
            local_dose = np.zeros(
                (z_end - z_start, dose_grid_shape[1], dose_grid_shape[2]),
                dtype=np.float32,
            )

            energy = beam_params["energy"]
            field_size = beam_params.get("field_size", (10.0, 10.0))

            nz, ny, nx = dose_grid_shape
            center_y, center_x = ny // 2, nx // 2

            for z in range(z_start, z_end):
                for y in range(ny):
                    for x in range(nx):
                        # Check if within field
                        dy = (y - center_y) * spacing[1]
                        dx = (x - center_x) * spacing[2]

                        if (
                            abs(dy) <= field_size[1] / 2
                            and abs(dx) <= field_size[0] / 2
                        ):
                            # Simple dose calculation
                            depth = z * spacing[0]
                            pdd = np.exp(-0.01 * depth)  # Simplified PDD
                            dose = energy * pdd * geometry[z, y, x]
                            local_dose[z - z_start, y, x] = dose

            return local_dose, z_start, z_end

        start_time = time.time()

        # Split z-axis among workers
        nz = dose_grid_shape[0]
        z_per_worker = nz // self.cpu_cores
        tasks = []

        for i in range(self.cpu_cores):
            z_start = i * z_per_worker
            z_end = (i + 1) * z_per_worker if i < self.cpu_cores - 1 else nz
            tasks.append((z_start, z_end, patient_geometry, beam_data))

        # Execute in parallel
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=self.cpu_cores
        ) as executor:
            results = list(executor.map(collapsed_cone_worker, tasks))

        # Combine results
        final_dose = np.zeros(dose_grid_shape, dtype=np.float32)
        for local_dose, z_start, z_end in results:
            final_dose[z_start:z_end] = local_dose

        return GPUDoseResult(
            dose_grid=final_dose,
            statistical_uncertainty=None,
            calculation_time=time.time() - start_time,
            algorithm_used="collapsed_cone_cpu_parallel",
            gpu_backend="cpu_parallel",
            memory_usage_mb=final_dose.nbytes / (1024**2),
        )


def create_gpu_dose_calculator(
    algorithm: str = "collapsed_cone",
    backend: str = "auto",
    accuracy: str = "high",
    max_memory_gb: float = 8.0,
) -> GPUDoseCalculator:
    """Factory function để tạo GPU dose calculator."""

    settings = GPUDoseSettings(
        algorithm_type=algorithm,
        preferred_backend=backend,
        accuracy_level=accuracy,
        max_memory_usage_gb=max_memory_gb,
    )

    return GPUDoseCalculator(settings)


def benchmark_dose_calculation_performance() -> Dict[str, Any]:
    """Benchmark hiệu suất của các thuật toán dose calculation."""

    # Create test data
    test_shape = (64, 64, 32)
    patient_geometry = np.random.rand(*test_shape) * 2.0  # Density variation
    spacing = (2.0, 2.0, 2.0)

    beam_data = {
        "energy": 6.0,
        "fluence": 1.0,
        "sad": 100.0,
        "field_size": (10.0, 10.0),
    }

    results = {}
    algorithms = ["pencil_beam", "collapsed_cone"]
    backends = ["cupy", "numba", "cpu_parallel"]

    for algorithm in algorithms:
        for backend in backends:
            try:
                calculator = create_gpu_dose_calculator(
                    algorithm=algorithm, backend=backend, accuracy="medium"
                )

                start_time = time.time()
                result = calculator.calculate_dose_gpu(
                    beam_data=beam_data,
                    patient_geometry=patient_geometry,
                    dose_grid_shape=test_shape,
                    spacing=spacing,
                )
                end_time = time.time()

                key = f"{algorithm}_{backend}"
                results[key] = {
                    "calculation_time": end_time - start_time,
                    "voxels_per_second": result.voxels_per_second,
                    "memory_usage_mb": result.memory_usage_mb,
                    "mean_dose": result.mean_dose,
                    "max_dose": result.max_dose,
                    "available": True,
                }

            except Exception as e:
                key = f"{algorithm}_{backend}"
                results[key] = {"available": False, "error": str(e)}

    return results


if __name__ == "__main__":
    # Test GPU dose calculation
    print("Testing GPU Dose Calculation...")

    # Run benchmark
    benchmark_results = benchmark_dose_calculation_performance()

    print("\nDose Calculation Benchmark Results:")
    print("=" * 80)
    for method, result in benchmark_results.items():
        if result.get("available", False):
            print(
                f"{method:25}: {result['calculation_time']:.3f}s, "
                f"{result['voxels_per_second'] / 1e6:.2f} MVoxels/s, "
                f"Max Dose: {result['max_dose']:.2f}"
            )
        else:
            print(
                f"{method:25}: Not available - {result.get('error', 'Unknown error')}"
            )
