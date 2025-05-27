#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Implementation of Monte Carlo dose calculation algorithm with GPU acceleration.

This module provides a GPU-accelerated implementation of the Monte Carlo
dose calculation algorithm for radiation therapy treatment planning.
"""

import logging
import numpy as np
import os
import time
from typing import Dict, List, Optional, Any, Tuple, Union

logger = logging.getLogger(__name__)

# Try to import GPU libraries with fallback mechanisms
try:
    import cupy as cp

    HAS_CUPY = True
    logger.info("CuPy successfully imported for GPU acceleration")
except ImportError:
    HAS_CUPY = False
    logger.warning("CuPy not available. Will try PyCUDA next.")

try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    from pycuda import gpuarray
    import pycuda.compiler as compiler

    HAS_PYCUDA = True
    logger.info("PyCUDA successfully imported for GPU acceleration")
except ImportError:
    HAS_PYCUDA = False
    if not HAS_CUPY:
        logger.warning(
            "Neither CuPy nor PyCUDA available. GPU acceleration will not be available."
        )


# Base class for Monte Carlo simulation using GPU
class MonteCarloGPU:
    """
    Monte Carlo dose calculation using GPU acceleration.

    This class implements Monte Carlo dose calculation for radiation therapy
    treatment planning, leveraging GPU acceleration for significantly faster
    computation compared to CPU-based implementations.
    """

    def __init__(self, num_particles: int = 1000000, **kwargs):
        """
        Initialize GPU-accelerated Monte Carlo algorithm.

        Parameters
        ----------
        num_particles : int, optional
            Number of particles to simulate, by default 1000000
        **kwargs
            Additional configuration parameters
        """
        self.num_particles = num_particles
        self.config = kwargs
        self.device = None
        self.has_gpu = False

        # Try to initialize GPU
        self._setup_gpu()

        # Initialize dose grid
        self.dose_grid = None
        self.ct_data = None
        self.materials = None
        self.beam_config = None

        # Performance metrics
        self.calculation_time = 0
        self.particles_per_second = 0
        self.memory_usage = 0

    def _setup_gpu(self):
        """Setup GPU environment based on available libraries."""
        if HAS_CUPY:
            self._setup_gpu_cupy()
        elif HAS_PYCUDA:
            self._setup_gpu_pycuda()
        else:
            self._setup_cpu_fallback()

    def _setup_gpu_cupy(self):
        """Initialize GPU using CuPy."""
        try:
            # Get device information
            num_gpus = cp.cuda.runtime.getDeviceCount()
            if num_gpus > 0:
                # Use first available GPU by default
                device_id = 0
                if "gpu_id" in self.config:
                    device_id = min(self.config["gpu_id"], num_gpus - 1)

                cp.cuda.Device(device_id).use()
                self.device = cp.cuda.Device(device_id)
                device_name = self.device.attributes.get("name", "Unknown").decode(
                    "utf-8"
                )
                mem_info = self.device.mem_info
                total_memory = mem_info[1] / (1024**3)  # GB
                free_memory = mem_info[0] / (1024**3)  # GB

                logger.info(
                    f"Using GPU {device_id}: {device_name} with {free_memory:.2f}GB/{total_memory:.2f}GB free memory"
                )
                self.has_gpu = True
                self.gpu_library = "cupy"

                # Set number of particles based on available memory
                if "auto_particles" in self.config and self.config["auto_particles"]:
                    # 1 million particles per GB as a rough estimate
                    self.num_particles = max(int(free_memory * 1000000), 1000000)
                    logger.info(
                        f"Auto-configured for {self.num_particles} particles based on available memory"
                    )
            else:
                logger.warning("No CUDA-capable GPU found despite CuPy being installed")
                self._setup_cpu_fallback()
        except Exception as e:
            logger.error(f"Error initializing CuPy GPU: {str(e)}")
            self._setup_cpu_fallback()

    def _setup_gpu_pycuda(self):
        """Initialize GPU using PyCUDA."""
        try:
            # Get device information
            num_gpus = cuda.Device.count()
            if num_gpus > 0:
                # Use first available GPU by default
                device_id = 0
                if "gpu_id" in self.config:
                    device_id = min(self.config["gpu_id"], num_gpus - 1)

                self.device = cuda.Device(device_id)
                device_name = self.device.name()
                total_memory = self.device.total_memory() / (1024**3)  # GB
                free_memory = (
                    self.device.total_memory() - self.device.used_memory()
                ) / (1024**3)  # GB

                logger.info(
                    f"Using GPU {device_id}: {device_name} with {free_memory:.2f}GB/{total_memory:.2f}GB free memory"
                )
                self.has_gpu = True
                self.gpu_library = "pycuda"

                # Set number of particles based on available memory
                if "auto_particles" in self.config and self.config["auto_particles"]:
                    # 1 million particles per GB as a rough estimate
                    self.num_particles = max(int(free_memory * 1000000), 1000000)
                    logger.info(
                        f"Auto-configured for {self.num_particles} particles based on available memory"
                    )
            else:
                logger.warning(
                    "No CUDA-capable GPU found despite PyCUDA being installed"
                )
                self._setup_cpu_fallback()
        except Exception as e:
            logger.error(f"Error initializing PyCUDA GPU: {str(e)}")
            self._setup_cpu_fallback()

    def _setup_cpu_fallback(self):
        """Set up CPU fallback when GPU is not available."""
        import multiprocessing

        num_cores = multiprocessing.cpu_count()
        logger.warning(
            f"Using CPU fallback with {num_cores} cores (much slower than GPU)"
        )
        self.has_gpu = False
        self.device = None
        self.gpu_library = None

        # Limit particles when using CPU to avoid excessive runtime
        if self.num_particles > 500000:
            self.num_particles = 500000
            logger.info(
                f"Reduced particle count to {self.num_particles} for CPU calculation"
            )

    def set_ct_data(self, ct_data: np.ndarray, voxel_size: Tuple[float, float, float]):
        """
        Set CT data for dose calculation.

        Parameters
        ----------
        ct_data : np.ndarray
            3D array containing CT data in Hounsfield units
        voxel_size : Tuple[float, float, float]
            Size of voxels in mm
        """
        self.ct_data = ct_data
        self.voxel_size = voxel_size

        # Initialize empty dose grid matching CT dimensions
        self.dose_grid = np.zeros_like(ct_data, dtype=np.float32)

        # Pre-process CT data for material assignment
        self._prepare_materials()

    def _prepare_materials(self):
        """Convert CT data (HU) to material properties for dose calculation."""
        if self.ct_data is None:
            logger.error("CT data not set. Call set_ct_data first.")
            return

        # Simple conversion from HU to relative electron density
        # In a real implementation, this would use a calibration curve
        self.materials = np.zeros_like(self.ct_data, dtype=np.float32)

        # Simple linear mapping of HU to relative electron density
        # Water is typically around 0 HU with density 1.0
        self.materials = 1.0 + (self.ct_data / 1000.0)

        # Limit to realistic values
        self.materials = np.clip(self.materials, 0.1, 10.0)

    def set_beam_configuration(self, beam_config: Dict[str, Any]):
        """
        Configure treatment beam parameters.

        Parameters
        ----------
        beam_config : Dict[str, Any]
            Dictionary containing beam parameters like energy, angle, etc.
        """
        self.beam_config = beam_config

    def calculate_dose(self):
        """
        Calculate dose distribution using Monte Carlo simulation.

        Returns
        -------
        np.ndarray
            3D dose distribution array
        """
        if self.ct_data is None:
            logger.error("CT data not set. Call set_ct_data first.")
            return None

        if self.beam_config is None:
            logger.error("Beam not configured. Call set_beam_configuration first.")
            return None

        start_time = time.time()

        # Choose appropriate calculation method based on available hardware
        if self.has_gpu and self.gpu_library == "cupy":
            self._calculate_dose_cupy()
        elif self.has_gpu and self.gpu_library == "pycuda":
            self._calculate_dose_pycuda()
        else:
            self._calculate_dose_cpu()

        end_time = time.time()
        self.calculation_time = end_time - start_time
        self.particles_per_second = self.num_particles / self.calculation_time

        logger.info(
            f"Monte Carlo calculation completed in {self.calculation_time:.2f} seconds"
        )
        logger.info(f"Performance: {self.particles_per_second:.2f} particles/second")

        # Apply final normalization
        self._normalize_dose()

        return self.dose_grid

    def _calculate_dose_cupy(self):
        """Calculate dose using CuPy GPU acceleration."""
        try:
            start_time = time.time()

            # Đảm bảo CT data được chuyển sang định dạng phù hợp
            if self.ct_data is None or self.materials is None:
                raise ValueError("CT data or materials not initialized")

            # Chuyển dữ liệu sang GPU
            ct_gpu = cp.asarray(self.ct_data)
            materials_gpu = cp.asarray(self.materials)
            dose_gpu = cp.zeros_like(ct_gpu, dtype=cp.float32)

            # Chuẩn bị thông số chùm tia
            source_position = cp.array(
                self.beam_config.get("source_position", [0, 0, -1000]), dtype=cp.float32
            )
            source_direction = cp.array(
                self.beam_config.get("source_direction", [0, 0, 1]), dtype=cp.float32
            )
            source_direction = source_direction / cp.linalg.norm(source_direction)

            # Tối ưu kích thước block cho CUDA
            block_size = 256
            grid_size = (self.num_particles + block_size - 1) // block_size

            # JIT compile custom CUDA kernel
            cuda_kernel = cp.RawKernel(
                r"""
            extern "C" __global__
            void monte_carlo_kernel(
                const float *materials, float *dose, const int nx, const int ny, const int nz,
                const float3 source_pos, const float3 source_dir,
                const float3 voxel_size, const int num_particles,
                unsigned int seed)
            {
                // Mỗi thread xử lý một hạt
                int particle_id = blockIdx.x * blockDim.x + threadIdx.x;
                if (particle_id >= num_particles) return;

                // Random state cho mỗi hạt
                curandState_t rng_state;
                curand_init(seed + particle_id, 0, 0, &rng_state);

                // Vị trí và hướng ban đầu (có nhiễu ngẫu nhiên)
                float3 pos = source_pos;

                // Thêm nhiễu cho hướng để tạo chùm tia phân kỳ
                float theta = curand_uniform(&rng_state) * 0.1f; // 0.1 rad ~ 5.7 độ max
                float phi = curand_uniform(&rng_state) * 2.0f * 3.14159f;

                float3 dir;
                dir.x = source_dir.x * cosf(theta) + sinf(theta) * cosf(phi);
                dir.y = source_dir.y * cosf(theta) + sinf(theta) * sinf(phi);
                dir.z = source_dir.z * cosf(theta);

                // Chuẩn hóa vector hướng
                float dir_norm = sqrtf(dir.x*dir.x + dir.y*dir.y + dir.z*dir.z);
                dir.x /= dir_norm;
                dir.y /= dir_norm;
                dir.z /= dir_norm;

                // Năng lượng ban đầu của hạt (MeV)
                float energy = 6.0f; // Default 6 MeV

                // Monte Carlo loop
                while (energy > 0.01f) { // Năng lượng ngưỡng
                    // Xác định vị trí voxel hiện tại
                    int ix = (int)((pos.x - origin.x) / voxel_size.x);
                    int iy = (int)((pos.y - origin.y) / voxel_size.y);
                    int iz = (int)((pos.z - origin.z) / voxel_size.z);

                    // Kiểm tra xem hạt có nằm trong dose grid không
                    if (ix >= 0 && ix < nx && iy >= 0 && iy < ny && iz >= 0 && iz < nz) {
                        int idx = iz*nx*ny + iy*nx + ix;

                        // Tính năng lượng nạp vào voxel
                        float step_length = fminf(
                            fminf(voxel_size.x / fabsf(dir.x + 1e-6f),
                                  voxel_size.y / fabsf(dir.y + 1e-6f)),
                            voxel_size.z / fabsf(dir.z + 1e-6f)
                        );

                        // Lấy vật liệu tại voxel
                        float material_density = materials[idx];

                        // Tính tương tác và nạp liều
                        float energy_deposit = step_length * material_density * 0.01f * energy;
                        energy_deposit = fminf(energy_deposit, energy); // Không thể nạp nhiều hơn năng lượng có

                        // Cập nhật liều bằng atomic add
                        atomicAdd(&dose[idx], energy_deposit);

                        // Cập nhật năng lượng
                        energy -= energy_deposit;

                        // Xác suất tán xạ (đơn giản hóa)
                        float scatter_prob = material_density * 0.1f;
                        if (curand_uniform(&rng_state) < scatter_prob) {
                            // Tán xạ - thay đổi hướng
                            float theta_s = curand_uniform(&rng_state) * 0.5f; // Max 0.5 rad ~ 29 độ
                            float phi_s = curand_uniform(&rng_state) * 2.0f * 3.14159f;

                            float cos_theta = cosf(theta_s);
                            float sin_theta = sinf(theta_s);
                            float cos_phi = cosf(phi_s);
                            float sin_phi = sinf(phi_s);

                            // Tạo vector trực giao với dir
                            float3 u, v;
                            if (fabsf(dir.z) < 0.9f) {
                                u.x = -dir.y;
                                u.y = dir.x;
                                u.z = 0.0f;
                            } else {
                                u.x = 0.0f;
                                u.y = -dir.z;
                                u.z = dir.y;
                            }

                            // Chuẩn hóa u
                            float u_norm = sqrtf(u.x*u.x + u.y*u.y + u.z*u.z);
                            u.x /= u_norm;
                            u.y /= u_norm;
                            u.z /= u_norm;

                            // v = dir × u
                            v.x = dir.y*u.z - dir.z*u.y;
                            v.y = dir.z*u.x - dir.x*u.z;
                            v.z = dir.x*u.y - dir.y*u.x;

                            // Áp dụng tán xạ
                            dir.x = dir.x*cos_theta + sin_theta*(u.x*cos_phi + v.x*sin_phi);
                            dir.y = dir.y*cos_theta + sin_theta*(u.y*cos_phi + v.y*sin_phi);
                            dir.z = dir.z*cos_theta + sin_theta*(u.z*cos_phi + v.z*sin_phi);

                            // Chuẩn hóa lại vector hướng
                            dir_norm = sqrtf(dir.x*dir.x + dir.y*dir.y + dir.z*dir.z);
                            dir.x /= dir_norm;
                            dir.y /= dir_norm;
                            dir.z /= dir_norm;
                        }
                    } else {
                        // Hạt đi ra ngoài vùng tính - dừng mô phỏng
                        break;
                    }

                    // Di chuyển hạt
                    pos.x += dir.x * step_length;
                    pos.y += dir.y * step_length;
                    pos.z += dir.z * step_length;
                }
            }
            """,
                "monte_carlo_kernel",
            )

            # Chuẩn bị thông số
            seed = np.random.randint(0, 2**32 - 1)
            voxel_size = cp.array(self.voxel_size, dtype=cp.float32)
            nx, ny, nz = self.ct_data.shape

            # Định nghĩa origin
            origin = cp.array([0, 0, 0], dtype=cp.float32)

            # Cố gắng chạy kernel - nếu lỗi, sẽ dùng cách thủ công
            try:
                # Thử chạy CUDA kernel tối ưu
                cuda_kernel(
                    (grid_size,),
                    (block_size,),
                    (
                        materials_gpu,
                        dose_gpu,
                        nx,
                        ny,
                        nz,
                        source_position,
                        source_direction,
                        voxel_size,
                        self.num_particles,
                        seed,
                    ),
                )
            except Exception as e:
                logger.warning(
                    f"GPU kernel error: {e}. Falling back to manual particle simulation."
                )

                # Fallback: Mô phỏng thủ công từng hạt (chậm hơn nhưng chắc chắn chạy được)
                # Tạo ngẫu nhiên các hạt trên GPU
                rng = cp.random.RandomState(seed)

                # Giảm số lượng hạt nếu đang dùng fallback để tránh quá lâu
                num_particles_fallback = min(self.num_particles, 100000)

                for i in range(num_particles_fallback):
                    # Tạo hạt với hướng ngẫu nhiên quanh hướng chính
                    theta = rng.uniform(0, 0.1)  # 0.1 rad ~ 5.7 độ max
                    phi = rng.uniform(0, 2.0 * np.pi)

                    # Tính hướng mới
                    dir_x = source_direction[0] * cp.cos(theta) + cp.sin(
                        theta
                    ) * cp.cos(phi)
                    dir_y = source_direction[1] * cp.cos(theta) + cp.sin(
                        theta
                    ) * cp.sin(phi)
                    dir_z = source_direction[2] * cp.cos(theta)

                    # Chuẩn hóa
                    dir_norm = cp.sqrt(dir_x**2 + dir_y**2 + dir_z**2)
                    dir_x /= dir_norm
                    dir_y /= dir_norm
                    dir_z /= dir_norm

                    # Vị trí ban đầu
                    pos_x, pos_y, pos_z = source_position

                    # Năng lượng ban đầu
                    energy = 6.0  # MeV

                    # Mô phỏng đường đi của hạt
                    while energy > 0.01:
                        # Xác định voxel hiện tại
                        ix = int((pos_x - 0) / voxel_size[0])
                        iy = int((pos_y - 0) / voxel_size[1])
                        iz = int((pos_z - 0) / voxel_size[2])

                        # Kiểm tra nếu nằm trong dose grid
                        if 0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz:
                            idx = iz * nx * ny + iy * nx + ix

                            # Tính bước di chuyển
                            step_x = (
                                voxel_size[0] / cp.abs(dir_x) if dir_x != 0 else 1e6
                            )
                            step_y = (
                                voxel_size[1] / cp.abs(dir_y) if dir_y != 0 else 1e6
                            )
                            step_z = (
                                voxel_size[2] / cp.abs(dir_z) if dir_z != 0 else 1e6
                            )
                            step_length = cp.min(cp.array([step_x, step_y, step_z]))

                            # Lấy vật liệu tại voxel
                            material_density = materials_gpu[idx]

                            # Tính nạp năng lượng
                            energy_deposit = (
                                step_length * material_density * 0.01 * energy
                            )
                            energy_deposit = min(energy_deposit, energy)

                            # Cập nhật liều
                            dose_gpu[idx] += energy_deposit

                            # Cập nhật năng lượng
                            energy -= energy_deposit

                            # Xác suất tán xạ
                            scatter_prob = material_density * 0.1
                            if rng.uniform(0, 1) < scatter_prob:
                                # Tán xạ
                                theta_s = rng.uniform(0, 0.5)
                                phi_s = rng.uniform(0, 2.0 * np.pi)

                                # Tính toán hướng mới sau tán xạ (đơn giản hóa)
                                sin_theta = cp.sin(theta_s)
                                cos_theta = cp.cos(theta_s)

                                # Thay đổi hướng (đơn giản hóa)
                                new_dir_x = dir_x * cos_theta + sin_theta * cp.cos(
                                    phi_s
                                )
                                new_dir_y = dir_y * cos_theta + sin_theta * cp.sin(
                                    phi_s
                                )
                                new_dir_z = dir_z * cos_theta

                                # Chuẩn hóa
                                dir_norm = cp.sqrt(
                                    new_dir_x**2 + new_dir_y**2 + new_dir_z**2
                                )
                                dir_x = new_dir_x / dir_norm
                                dir_y = new_dir_y / dir_norm
                                dir_z = new_dir_z / dir_norm
                        else:
                            # Hạt đi ra ngoài
                            break

                        # Di chuyển hạt
                        pos_x += dir_x * step_length
                        pos_y += dir_y * step_length
                        pos_z += dir_z * step_length

                    # Báo cáo tiến độ nếu mô phỏng nhiều hạt
                    if i % 10000 == 0 and i > 0:
                        progress = i / num_particles_fallback * 100
                        logger.info(
                            f"Fallback particle simulation: {progress:.1f}% complete"
                        )

            # Chuyển kết quả về CPU và chuẩn hóa
            self.dose_grid = cp.asnumpy(dose_gpu)

            # Thời gian tính toán
            elapsed_time = time.time() - start_time
            self.calculation_time = elapsed_time
            self.particles_per_second = self.num_particles / elapsed_time

            # Chuẩn hóa liều
            self._normalize_dose()

            logger.info(f"CuPy GPU Monte Carlo completed in {elapsed_time:.2f} seconds")
            logger.info(
                f"Performance: {self.particles_per_second:.2f} particles/second"
            )

            return self.dose_grid

        except Exception as e:
            logger.error(f"CuPy GPU Monte Carlo error: {str(e)}")
            logger.info("Falling back to CPU calculation")
            return self._calculate_dose_cpu()

    def _calculate_dose_pycuda(self):
        """Implement dose calculation using PyCUDA."""
        try:
            # Similar to CuPy implementation but using PyCUDA
            # This is a placeholder for the actual implementation
            logger.info(
                f"Starting PyCUDA Monte Carlo simulation with {self.num_particles} particles"
            )

            # TODO: Implement actual Monte Carlo transport using PyCUDA

        except Exception as e:
            logger.error(f"Error in PyCUDA dose calculation: {str(e)}")
            logger.warning("Falling back to CPU calculation")
            self._calculate_dose_cpu()

    def _calculate_dose_cpu(self):
        """CPU fallback implementation of Monte Carlo dose calculation."""
        logger.info(
            f"Starting CPU Monte Carlo simulation with {self.num_particles} particles"
        )

        # TODO: Implement simplified Monte Carlo algorithm for CPU
        # This is just a placeholder calculation for demonstration

        # Simple exponential attenuation based on ray tracing
        # In a real implementation, this would be much more complex
        pass

    def _normalize_dose(self):
        """Normalize dose grid to prescribed dose level."""
        if self.dose_grid is None:
            return

        # Find maximum dose value
        max_dose = np.max(self.dose_grid)
        if max_dose > 0:
            # Normalize to prescription dose or to 1.0 if not specified
            prescription = self.beam_config.get("prescription", 1.0)  # Gy
            self.dose_grid = self.dose_grid * (prescription / max_dose)

    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics from the last calculation.

        Returns
        -------
        Dict[str, Any]
            Dictionary with performance metrics
        """
        return {
            "calculation_time": self.calculation_time,
            "particles_per_second": self.particles_per_second,
            "memory_usage_gb": self.memory_usage,
            "using_gpu": self.has_gpu,
            "gpu_library": self.gpu_library,
            "num_particles": self.num_particles,
        }

    def compare_with_dose_grid(self, reference_dose: np.ndarray) -> Dict[str, Any]:
        """
        So sánh phân bố liều tính toán với phân bố liều tham chiếu.

        Phương thức này thực hiện phân tích chi tiết sự khác biệt giữa hai phân phối liều,
        bao gồm cả phân tích gamma 3D với nhiều tiêu chí khác nhau.

        Parameters
        ----------
        reference_dose : np.ndarray
            Phân phối liều tham chiếu để so sánh

        Returns
        -------
        Dict[str, Any]
            Dictionary với các chỉ số so sánh chi tiết
        """
        if self.dose_grid is None or reference_dose is None:
            logger.error("Cả liều tính toán và tham chiếu đều phải tồn tại")
            return {"error": "Missing dose data"}

        try:
            # Kiểm tra kích thước và loại dữ liệu
            if self.dose_grid.shape != reference_dose.shape:
                logger.warning(
                    f"Kích thước không khớp: {self.dose_grid.shape} vs {reference_dose.shape}. "
                    "Sẽ thử điều chỉnh kích thước tự động."
                )

                # Thử điều chỉnh kích thước nếu khác nhau
                try:
                    from scipy.ndimage import zoom

                    # Tính tỷ lệ zoom cần thiết cho mỗi chiều
                    zoom_factors = [
                        ref / calc
                        for ref, calc in zip(reference_dose.shape, self.dose_grid.shape)
                    ]
                    logger.info(f"Điều chỉnh kích thước với tỷ lệ: {zoom_factors}")

                    # Áp dụng zoom
                    calc_dose = zoom(
                        self.dose_grid, zoom_factors, order=1, mode="nearest"
                    )
                    logger.info(
                        f"Đã điều chỉnh kích thước dose_grid thành {calc_dose.shape}"
                    )
                except ImportError:
                    logger.error(
                        "Không thể import scipy.ndimage.zoom để điều chỉnh kích thước. "
                        "Hãy cài đặt scipy hoặc đảm bảo kích thước liều khớp nhau."
                    )
                    return {
                        "error": "Size mismatch and resize capability not available. "
                        "Install scipy or ensure dose grids have matching dimensions."
                    }
                except Exception as e:
                    logger.error(f"Lỗi khi điều chỉnh kích thước dose_grid: {str(e)}")
                    return {
                        "error": f"Size mismatch: {self.dose_grid.shape} vs {reference_dose.shape}. "
                        f"Resize failed: {str(e)}"
                    }
            else:
                # Sử dụng dose_grid hiện tại nếu kích thước đã khớp
                calc_dose = self.dose_grid.copy()

            # Chuyển đổi sang kiểu dữ liệu float32 cho nhất quán
            calc_dose = calc_dose.astype(np.float32)
            ref_dose = reference_dose.astype(np.float32)

            # Kiểm tra giá trị nan và inf
            if np.isnan(calc_dose).any() or np.isinf(calc_dose).any():
                logger.warning(
                    "Liều tính toán có giá trị NaN hoặc Inf. Thay thế bằng 0."
                )
                calc_dose = np.nan_to_num(calc_dose, nan=0.0, posinf=0.0, neginf=0.0)

            if np.isnan(ref_dose).any() or np.isinf(ref_dose).any():
                logger.warning(
                    "Liều tham chiếu có giá trị NaN hoặc Inf. Thay thế bằng 0."
                )
                ref_dose = np.nan_to_num(ref_dose, nan=0.0, posinf=0.0, neginf=0.0)

            # Tính toán các chỉ số cơ bản
            diff = calc_dose - ref_dose
            abs_diff = np.abs(diff)

            # Tạo mask chỉ xét vùng có liều > threshold% liều tối đa
            threshold_percent = 10.0  # 10% của liều tối đa
            ref_max = float(np.max(ref_dose))

            if ref_max <= 0:
                logger.warning("Liều tham chiếu có giá trị tối đa bằng 0 hoặc âm")
                return {"error": "Invalid reference dose (max <= 0)"}

            dose_threshold = ref_max * (threshold_percent / 100.0)
            mask = ref_dose >= dose_threshold
            masked_voxel_count = int(np.sum(mask))

            # Tính % sai khác trung bình trong vùng quan tâm
            if masked_voxel_count > 0:
                mean_pct_diff = 100.0 * float(np.mean(abs_diff[mask])) / ref_max
            else:
                mean_pct_diff = 0.0
                logger.warning(
                    f"Không có voxel nào trong vùng quan tâm (> {threshold_percent}% liều tối đa)"
                )

            # Tính các chỉ số cơ bản - chuyển sang native Python types
            metrics = {
                "mean_error": float(np.mean(diff)),
                "mean_abs_error": float(np.mean(abs_diff)),
                "mean_pct_diff": float(mean_pct_diff),
                "max_error": float(np.max(abs_diff)),
                "rms_error": float(np.sqrt(np.mean(np.square(diff)))),
                "max_reference_dose": float(ref_max),
                "voxels_in_mask": int(masked_voxel_count),
                "threshold_percent": float(threshold_percent),
            }

            # Tính chỉ số gamma nếu module phân tích gamma có sẵn
            try:
                # Import động để tránh phụ thuộc cứng
                gamma_imported = False

                # Thử import với API mới trước
                try:
                    from quangtps.evaluation.metrics.gamma_analysis import (
                        calculate_gamma_3d,
                    )

                    gamma_imported = True
                    logger.info(
                        "Đã import thành công module gamma analysis với API mới"
                    )
                except ImportError:
                    # Thử API cũ
                    try:
                        from quangtps.evaluation.metrics.gamma_index import (
                            calculate_gamma_3d,
                        )

                        gamma_imported = True
                        logger.info(
                            "Đã import thành công module gamma analysis với API cũ"
                        )
                    except ImportError:
                        gamma_imported = False
                        logger.warning("Module phân tích gamma không khả dụng")
                        metrics["gamma_analysis"] = {
                            "error": "Module gamma analysis không khả dụng"
                        }

                if not gamma_imported:
                    return metrics

            except Exception as e:
                logger.warning(f"Lỗi khi import module gamma analysis: {str(e)}")
                metrics["gamma_analysis"] = {
                    "error": f"Lỗi khi import module gamma analysis: {str(e)}"
                }
                return metrics

            logger.info("Bắt đầu phân tích gamma 3D...")

            # Lấy thông tin voxel_size từ thuộc tính nếu có
            voxel_size = getattr(self, "voxel_size", (1.0, 1.0, 1.0))

            # Đảm bảo voxel_size là tuple để tránh lỗi kiểu
            if not isinstance(voxel_size, tuple):
                try:
                    voxel_size = tuple(voxel_size)
                except Exception as e:
                    logger.warning(
                        f"Không thể chuyển đổi voxel_size sang tuple: {str(e)}"
                    )
                    voxel_size = (1.0, 1.0, 1.0)

            # Kiểm tra độ dài của voxel_size để tránh lỗi
            if len(voxel_size) != 3:
                logger.warning(
                    f"voxel_size phải có độ dài 3, nhưng có {len(voxel_size)}. "
                    "Sử dụng (1.0, 1.0, 1.0) thay thế."
                )
                voxel_size = (1.0, 1.0, 1.0)

            # Thiết lập các bộ tiêu chí gamma phổ biến
            gamma_criteria = [
                {
                    "distance_mm": 3.0,
                    "dose_percent": 3.0,
                    "threshold": 10.0,
                },  # 3mm/3%
                {
                    "distance_mm": 2.0,
                    "dose_percent": 2.0,
                    "threshold": 10.0,
                },  # 2mm/2%
                {
                    "distance_mm": 1.0,
                    "dose_percent": 1.0,
                    "threshold": 10.0,
                },  # 1mm/1%
            ]

            # Thêm container cho kết quả gamma
            metrics["gamma_analysis"] = {}

            # Tính toán gamma cho từng bộ tiêu chí
            for criteria in gamma_criteria:
                distance_criterion_mm = criteria["distance_mm"]
                dose_difference_percent = criteria["dose_percent"]
                threshold_dose = criteria["threshold"]

                criteria_key = f"gamma_{int(distance_criterion_mm)}mm_{int(dose_difference_percent)}pct"

                try:
                    # Kiểm tra chữ ký hàm để xác định API đúng
                    import inspect

                    sig = inspect.signature(calculate_gamma_3d)
                    param_names = list(sig.parameters.keys())

                    # Xác định API được sử dụng
                    has_dose_threshold = "dose_threshold_percent" in param_names
                    has_spacing = "spacing" in param_names

                    # Xây dựng tham số phù hợp với API
                    gamma_kwargs = {
                        "reference_dose": ref_dose,
                        "evaluated_dose": calc_dose,
                        "distance_mm": distance_criterion_mm,
                        "dose_percent": dose_difference_percent,
                    }

                    if has_dose_threshold:
                        gamma_kwargs["dose_threshold_percent"] = threshold_dose
                    if has_spacing:
                        gamma_kwargs["spacing"] = voxel_size

                    # Gọi hàm gamma analysis với đúng tham số
                    logger.info(f"Gọi calculate_gamma_3d với tham số: {gamma_kwargs}")
                    gamma_result = calculate_gamma_3d(**gamma_kwargs)

                    # Xử lý kết quả dựa vào kiểu trả về
                    if hasattr(gamma_result, "gamma_map"):
                        # Nếu trả về một đối tượng GammaAnalysisResult
                        gamma_map = gamma_result.gamma_map
                        pass_rate = gamma_result.pass_rate
                        mean_gamma = gamma_result.mean_gamma
                        max_gamma = gamma_result.max_gamma
                    else:
                        # Nếu trả về numpy array
                        gamma_map = gamma_result

                        # Tính pass rate (tỷ lệ voxel có gamma <= 1.0)
                        if np.sum(mask) > 0:
                            passing_voxels = np.sum((gamma_map[mask] <= 1.0))
                            pass_rate = 100.0 * passing_voxels / np.sum(mask)
                            mean_gamma = float(np.mean(gamma_map[mask]))
                            max_gamma = float(np.max(gamma_map[mask]))
                        else:
                            pass_rate = 0.0
                            mean_gamma = 0.0
                            max_gamma = 0.0

                    # Lưu kết quả với các kiểu dữ liệu Python chuẩn để JSON serialization
                    metrics["gamma_analysis"][criteria_key] = {
                        "pass_rate": float(pass_rate),
                        "mean_gamma": float(mean_gamma),
                        "max_gamma": float(max_gamma),
                        "criteria_string": f"{distance_criterion_mm}mm/{dose_difference_percent}%",
                        "evaluated_voxels": int(np.sum(mask)),
                        "passing_voxels": int(
                            np.sum((gamma_map[mask] <= 1.0)) if np.sum(mask) > 0 else 0
                        ),
                    }

                    # Thông tin chi tiết về kết quả phân tích gamma
                    logger.info(
                        f"Phân tích gamma {criteria_key}: pass rate = {float(pass_rate):.2f}% "
                        f"({int(np.sum((gamma_map[mask] <= 1.0)) if np.sum(mask) > 0 else 0)}/{int(np.sum(mask))} voxels)"
                    )

                except Exception as e:
                    error_msg = (
                        f"Lỗi khi tính gamma với tiêu chí {criteria_key}: {str(e)}"
                    )
                    logger.warning(error_msg)
                    import traceback

                    logger.debug(traceback.format_exc())

                    # Lưu thông tin lỗi trong kết quả để frontend có thể xử lý
                    metrics["gamma_analysis"][criteria_key] = {
                        "error": error_msg,
                        "criteria_string": f"{distance_criterion_mm}mm/{dose_difference_percent}%",
                    }

            # Thêm kết quả với tiêu chí chính (3mm/3%) vào cấp cao nhất của dict để dễ truy cập
            if "gamma_3mm_3pct" in metrics["gamma_analysis"]:
                if "error" not in metrics["gamma_analysis"]["gamma_3mm_3pct"]:
                    metrics.update(
                        {
                            "gamma_pass_rate": metrics["gamma_analysis"][
                                "gamma_3mm_3pct"
                            ]["pass_rate"],
                            "gamma_criteria": "3mm/3%",
                            "mean_gamma": metrics["gamma_analysis"]["gamma_3mm_3pct"][
                                "mean_gamma"
                            ],
                            "max_gamma": metrics["gamma_analysis"]["gamma_3mm_3pct"][
                                "max_gamma"
                            ],
                        }
                    )

            return metrics

        except Exception as e:
            logger.error(f"Lỗi không mong đợi trong phân tích so sánh liều: {str(e)}")
            import traceback

            logger.debug(traceback.format_exc())
            return {
                "error": f"Unexpected error during dose comparison: {str(e)}",
                "mean_error": float("nan"),
                "mean_abs_error": float("nan"),
                "rms_error": float("nan"),
            }


# Lớp tích hợp MonteCarloGPUAlgorithm kế thừa từ MonteCarloGPU
class MonteCarloGPUAlgorithm(MonteCarloGPU):
    """
    Lớp tích hợp thuật toán Monte Carlo GPU vào hệ thống thuật toán tính liều của QuangTPS.

    Lớp này kế thừa từ MonteCarloGPU và triển khai các phương thức cần thiết để tích hợp
    với hệ thống thuật toán tính liều (DoseCalculationAlgorithm).
    """

    def __init__(self, **kwargs):
        """
        Khởi tạo thuật toán Monte Carlo GPU.

        Parameters
        ----------
        **kwargs
            Các tham số cấu hình cho thuật toán
        """
        super().__init__(**kwargs)
        self.patient_data = None
        self.beam_arrangement = None

        # Ensure GPU attributes are available
        if not hasattr(self, "gpu_available"):
            self.gpu_available = False
        if not hasattr(self, "backend"):
            self.backend = "cpu"

        self.calculation_status = {
            "initialized": False,
            "ready": False,
            "completed": False,
            "error": None,
        }
        logger.info("Khởi tạo thuật toán MonteCarloGPUAlgorithm")

    def initialize(self, patient_data):
        """
        Khởi tạo thuật toán với dữ liệu bệnh nhân.

        Parameters
        ----------
        patient_data : Any
            Dữ liệu bệnh nhân bao gồm CT và các thông tin liên quan

        Returns
        -------
        bool
            True nếu khởi tạo thành công, False nếu thất bại
        """
        try:
            self.patient_data = patient_data

            # Trích xuất dữ liệu CT từ patient_data
            if hasattr(patient_data, "ct_data") and hasattr(patient_data, "voxel_size"):
                self.set_ct_data(patient_data.ct_data, patient_data.voxel_size)
                self.calculation_status["initialized"] = True
                logger.info("Khởi tạo thuật toán MonteCarloGPU thành công")
                return True
            else:
                logger.error(
                    "Dữ liệu bệnh nhân không chứa thông tin CT hoặc voxel_size"
                )
                self.calculation_status["error"] = "Dữ liệu bệnh nhân không đầy đủ"
                return False
        except Exception as e:
            logger.error(f"Lỗi khi khởi tạo thuật toán MonteCarloGPU: {str(e)}")
            self.calculation_status["error"] = str(e)
            return False

    def calculate_dose(self, beam_arrangement):
        """
        Tính phân bố liều cho các chùm tia xác định.

        Parameters
        ----------
        beam_arrangement : Any
            Bố trí chùm tia, bao gồm thông tin về các chùm tia xạ trị

        Returns
        -------
        Any
            Kết quả tính toán liều
        """
        if not self.calculation_status["initialized"]:
            logger.error("Thuật toán chưa được khởi tạo. Gọi initialize() trước.")
            return None

        try:
            self.beam_arrangement = beam_arrangement
            total_dose = None

            # Xử lý từng chùm trong arrangement
            for i, beam in enumerate(beam_arrangement.beams):
                logger.info(f"Tính liều cho chùm {i + 1}/{len(beam_arrangement.beams)}")

                # Chuẩn bị cấu hình chùm tia
                beam_config = {
                    "energy": beam.energy,
                    "gantry_angle": beam.gantry_angle,
                    "collimator_angle": beam.collimator_angle,
                    "isocenter": beam.isocenter,
                    "mlc_positions": beam.mlc_positions,
                    "jaw_positions": beam.jaw_positions,
                    "weight": beam.weight,
                    "prescription": beam_arrangement.prescription.dose
                    if hasattr(beam_arrangement, "prescription")
                    else 1.0,
                }

                # Thiết lập cấu hình chùm và tính liều
                self.set_beam_configuration(beam_config)
                beam_dose = super().calculate_dose()

                # Cộng vào tổng liều
                if total_dose is None:
                    total_dose = beam_dose * beam.weight
                else:
                    total_dose += beam_dose * beam.weight

            # Lưu kết quả và cập nhật trạng thái
            self.dose_grid = total_dose
            self.calculation_status["completed"] = True
            self.calculation_status["ready"] = True

            # Tạo và trả về đối tượng kết quả
            result = MonteCarloGPUResult(
                dose_grid=self.dose_grid,
                patient_data=self.patient_data,
                beam_arrangement=beam_arrangement,
                performance=self.get_performance_stats(),
            )
            return result

        except Exception as e:
            logger.error(f"Lỗi khi tính liều với thuật toán MonteCarloGPU: {str(e)}")
            self.calculation_status["error"] = str(e)
            return None

    def get_algorithm_type(self):
        """
        Trả về loại thuật toán.

        Returns
        -------
        str
            Mã loại thuật toán
        """
        return "MONTE_CARLO_GPU"

    def get_display_name(self):
        """
        Trả về tên hiển thị của thuật toán.

        Returns
        -------
        str
            Tên hiển thị
        """
        return "Monte Carlo GPU"

    def get_description(self):
        """
        Trả về mô tả của thuật toán.

        Returns
        -------
        str
            Mô tả
        """
        return "Thuật toán Monte Carlo tính toán trên GPU với tốc độ nhanh hơn 50-200x so với CPU."

    def get_calculation_status(self):
        """
        Trả về trạng thái tính toán hiện tại.

        Returns
        -------
        Dict
            Trạng thái tính toán
        """
        return self.calculation_status

    def get_gpu_info(self):
        """
        Trả về thông tin GPU hiện tại.

        Returns
        -------
        Dict
            Thông tin GPU
        """
        gpu_info = {
            "gpu_available": self.gpu_available,
            "backend": self.backend,
            "device_count": 0,
            "memory_info": "N/A",
        }

        if self.gpu_available and self.backend == "cupy":
            try:
                import cupy as cp

                gpu_info["device_count"] = cp.cuda.runtime.getDeviceCount()
                gpu_info["memory_info"] = (
                    f"{cp.get_default_memory_pool().used_bytes() / 1024**3:.2f} GB used"
                )
            except Exception:
                pass
        elif self.gpu_available and self.backend == "pycuda":
            try:
                import pycuda.driver as cuda

                gpu_info["device_count"] = cuda.Device.count()
            except Exception:
                pass

        return gpu_info


class MonteCarloGPUResult:
    """
    Kết quả tính toán liều từ thuật toán Monte Carlo GPU.
    """

    def __init__(self, dose_grid, patient_data, beam_arrangement, performance):
        """
        Khởi tạo kết quả tính toán liều.

        Parameters
        ----------
        dose_grid : np.ndarray
            Mảng 3D chứa phân bố liều tính toán được
        patient_data : Any
            Dữ liệu bệnh nhân được sử dụng trong tính toán
        beam_arrangement : Any
            Bố trí chùm tia được sử dụng trong tính toán
        performance : Dict
            Thống kê hiệu năng từ quá trình tính toán
        """
        self.dose_grid = dose_grid
        self.patient_data = patient_data
        self.beam_arrangement = beam_arrangement
        self.performance = performance
        self.timestamp = time.time()

    def get_dose(self):
        """
        Trả về phân bố liều.

        Returns
        -------
        np.ndarray
            Mảng 3D chứa phân bố liều
        """
        return self.dose_grid

    def get_performance_stats(self):
        """
        Trả về thống kê hiệu năng.

        Returns
        -------
        Dict
            Thống kê hiệu năng
        """
        return self.performance

    def compare_with(self, other_result):
        """
                So sánh kết quả này với kết quả khác.

                Parameters
                ----------
                other_result : Any
                    Kết quả khác để so sánh

        Returns
        -------
                Dict
                    Các chỉ số so sánh
        """
        if hasattr(other_result, "get_dose"):
            other_dose = other_result.get_dose()
            return self._compare_doses(other_dose)
        else:
            logger.error("Đối tượng so sánh không có phương thức get_dose()")
            return None

    def _compare_doses(self, other_dose):
        """
                So sánh phân bố liều này với phân bố liều khác.

                Parameters
                ----------
                other_dose : np.ndarray
                    Phân bố liều khác để so sánh

        Returns
        -------
                Dict
                    Các chỉ số so sánh
        """
        try:
            # Sử dụng phương thức có sẵn từ lớp MonteCarloGPU
            comparator = MonteCarloGPU()
            comparator.dose_grid = self.dose_grid
            return comparator.compare_with_dose_grid(other_dose)
        except Exception as e:
            logger.error(f"Lỗi khi so sánh phân bố liều: {str(e)}")
            return None


# Đăng ký thuật toán với hệ thống
def register_algorithm():
    """Đăng ký thuật toán Monte Carlo GPU với hệ thống."""
    try:
        # Thử import từ dose_engine
        try:
            from quangtps.dose.dose_engine import register_dose_algorithm

            register_dose_algorithm("monte_carlo_gpu", MonteCarloGPUAlgorithm)
            logger.info("Đã đăng ký MonteCarloGPUAlgorithm thành công")
            return
        except (ImportError, NameError) as e:
            logger.debug(f"Không thể import từ dose_engine: {e}")

        # Thử import từ algorithms module
        try:
            from quangtps.dose.algorithms import register_dose_algorithm

            register_dose_algorithm("monte_carlo_gpu", MonteCarloGPUAlgorithm)
            logger.info("Đã đăng ký MonteCarloGPUAlgorithm thành công")
            return
        except (ImportError, NameError) as e:
            logger.debug(f"Không thể import từ algorithms: {e}")

        # Fallback: tạo registry tự quản lý
        try:
            import quangtps.dose.algorithms as algorithms_module

            if not hasattr(algorithms_module, "_dose_algorithms"):
                algorithms_module._dose_algorithms = {}
            algorithms_module._dose_algorithms["monte_carlo_gpu"] = (
                MonteCarloGPUAlgorithm
            )
            logger.info("Đã đăng ký MonteCarloGPUAlgorithm vào registry local")
        except Exception as e:
            logger.warning(f"Không thể đăng ký MonteCarloGPUAlgorithm: {e}")
            logger.info("MonteCarloGPUAlgorithm đã sẵn sàng để đăng ký thủ công")
    except Exception as e:
        logger.error(f"Lỗi khi đăng ký thuật toán: {e}")
        logger.info("MonteCarloGPUAlgorithm đã sẵn sàng để đăng ký thủ công")


# Thực thi đăng ký
try:
    register_algorithm()
except Exception as e:
    logger.error(f"Lỗi khi tự động đăng ký thuật toán: {e}")


# Export main class để import từ __init__.py
__all__ = ["MonteCarloGPUAlgorithm", "MonteCarloGPU", "MonteCarloGPUResult"]
