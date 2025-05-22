#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Các kernel GPU cho thuật toán tính liều Monte Carlo.

Module này cung cấp các kernel CUDA và OpenCL được tối ưu hóa
cho thuật toán Monte Carlo trong tính toán liều cho xạ trị.
"""

import logging
import os
import time
from typing import Dict, Any, Optional, List, Union, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Kiểm tra tính khả dụng của các thư viện GPU
HAS_CUPY = False
HAS_PYCUDA = False
HAS_OPENCL = False

try:
    import cupy as cp

    HAS_CUPY = True
    logger.info("Thư viện CuPy đã được nhận diện")
except ImportError:
    logger.debug("Thư viện CuPy không khả dụng")

try:
    import pycuda.driver as drv
    import pycuda.compiler as compiler

    HAS_PYCUDA = True
    logger.info("Thư viện PyCUDA đã được nhận diện")
except ImportError:
    logger.debug("Thư viện PyCUDA không khả dụng")

try:
    import pyopencl as cl

    HAS_OPENCL = True
    logger.info("Thư viện PyOpenCL đã được nhận diện")
except ImportError:
    logger.debug("Thư viện PyOpenCL không khả dụng")

# CUDA kernel code cho tính toán Monte Carlo
CUDA_KERNEL_SOURCE = """
// Hằng số vật lý
#define ELECTRON_REST_MASS 0.511f  // MeV
#define LIGHT_SPEED 299.792458f    // mm/ns
#define PI 3.14159265359f

// Cấu trúc cho ray
typedef struct {
    float3 origin;
    float3 direction;
    float energy;
    int type;  // 0 = photon, 1 = electron
} Ray;

// Cấu trúc cho dữ liệu CT và liều
typedef struct {
    int3 size;
    float3 spacing;
    float3 origin;
} VolumeInfo;

// Kernel cho mô phỏng photon
extern "C" __global__ void simulate_photons(
    float* ct_data,
    float* dose_data,
    float* material_data,
    int num_materials,
    VolumeInfo volume_info,
    float3 source_pos,
    float3* beam_directions,
    float* beam_energies,
    int num_histories_per_thread,
    int seed
) {
    // Lấy chỉ số thread
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    // Khởi tạo trạng thái ngẫu nhiên cho thread
    curandState rand_state;
    curand_init(seed + thread_id, 0, 0, &rand_state);

    // Mô phỏng nhiều hạt
    for (int i = 0; i < num_histories_per_thread; i++) {
        // Chọn hướng và năng lượng chùm tia từ phân phối
        int beam_idx = curand(&rand_state) % num_histories_per_thread;
        float3 dir = beam_directions[beam_idx];
        float energy = beam_energies[beam_idx];

        // Tạo photon ban đầu
        Ray ray;
        ray.origin = source_pos;
        ray.direction = dir;
        ray.energy = energy;
        ray.type = 0;  // photon

        // Theo dõi photon cho đến khi thoát khỏi phantom hoặc năng lượng quá thấp
        while (ray.energy > 0.01f) {  // 10 keV cutoff
            // Kiểm tra nếu photon vẫn trong phantom
            bool inside = check_inside_volume(ray.origin, volume_info);
            if (!inside) break;

            // Lấy chỉ số voxel
            int3 voxel_idx = get_voxel_index(ray.origin, volume_info);
            int flat_idx = flatten_index(voxel_idx, volume_info.size);

            // Lấy thông tin vật liệu cho voxel hiện tại
            int material_id = get_material_id(ct_data[flat_idx]);
            float density = get_density(ct_data[flat_idx]);

            // Lấy các tham số tương tác cho vật liệu và năng lượng
            float mu_total = get_total_attenuation(material_id, ray.energy, material_data, num_materials) * density;
            float mu_compton = get_compton_attenuation(material_id, ray.energy, material_data, num_materials) * density;
            float mu_photoelectric = get_photoelectric_attenuation(material_id, ray.energy, material_data, num_materials) * density;
            float mu_rayleigh = get_rayleigh_attenuation(material_id, ray.energy, material_data, num_materials) * density;

            // Tính khoảng cách đến tương tác tiếp theo
            float distance = -logf(curand_uniform(&rand_state)) / mu_total;

            // Di chuyển photon đến điểm tương tác
            float3 new_pos;
            new_pos.x = ray.origin.x + ray.direction.x * distance;
            new_pos.y = ray.origin.y + ray.direction.y * distance;
            new_pos.z = ray.origin.z + ray.direction.z * distance;
            ray.origin = new_pos;

            // Kiểm tra xem photon còn trong phantom không
            inside = check_inside_volume(ray.origin, volume_info);
            if (!inside) break;

            // Xác định loại tương tác
            float interact_prob = curand_uniform(&rand_state);
            float compton_prob = mu_compton / mu_total;
            float photoelectric_prob = mu_photoelectric / mu_total;
            float rayleigh_prob = mu_rayleigh / mu_total;

            // Cập nhật chỉ số voxel sau khi di chuyển
            voxel_idx = get_voxel_index(ray.origin, volume_info);
            flat_idx = flatten_index(voxel_idx, volume_info.size);

            if (interact_prob < compton_prob) {
                // Tán xạ Compton
                float old_energy = ray.energy;
                simulate_compton_scattering(&ray, &rand_state);

                // Truyền năng lượng cho voxel (năng lượng mất đi)
                float deposited_energy = old_energy - ray.energy;
                atomicAdd(&dose_data[flat_idx], deposited_energy);
            }
            else if (interact_prob < compton_prob + photoelectric_prob) {
                // Hiệu ứng quang điện - toàn bộ năng lượng được hấp thụ
                atomicAdd(&dose_data[flat_idx], ray.energy);
                ray.energy = 0.0f;
                break;
            }
            else if (interact_prob < compton_prob + photoelectric_prob + rayleigh_prob) {
                // Tán xạ Rayleigh - chỉ thay đổi hướng
                simulate_rayleigh_scattering(&ray, &rand_state);
            }
            else {
                // Các loại tương tác khác (pair production, v.v.)
                atomicAdd(&dose_data[flat_idx], ray.energy * 0.5f);  // Giả định 50% năng lượng được hấp thụ
                ray.energy *= 0.5f;
            }

            // Kiểm tra nếu năng lượng quá thấp
            if (ray.energy < 0.01f) {
                atomicAdd(&dose_data[flat_idx], ray.energy);  // Hấp thụ năng lượng còn lại
                break;
            }
        }
    }
}

// Mô phỏng tán xạ Compton
__device__ void simulate_compton_scattering(Ray* ray, curandState* rand_state) {
    // Công thức Klein-Nishina
    float e = ray->energy / ELECTRON_REST_MASS;

    // Lấy góc tán xạ từ phân phối Klein-Nishina
    float cos_theta = sample_klein_nishina(e, rand_state);
    float sin_theta = sqrtf(1.0f - cos_theta*cos_theta);

    // Góc phương vị ngẫu nhiên
    float phi = 2.0f * PI * curand_uniform(rand_state);

    // Cập nhật hướng ray
    float3 new_dir = update_direction(ray->direction, cos_theta, sin_theta, phi);
    ray->direction = new_dir;

    // Cập nhật năng lượng (công thức Compton)
    float old_e = ray->energy;
    ray->energy = old_e / (1.0f + (old_e / ELECTRON_REST_MASS) * (1.0f - cos_theta));
}

// Mô phỏng tán xạ Rayleigh
__device__ void simulate_rayleigh_scattering(Ray* ray, curandState* rand_state) {
    // Góc tán xạ từ hàm phân phối Rayleigh
    float cos_theta = 2.0f * curand_uniform(rand_state) - 1.0f;
    float sin_theta = sqrtf(1.0f - cos_theta*cos_theta);

    // Góc phương vị ngẫu nhiên
    float phi = 2.0f * PI * curand_uniform(rand_state);

    // Cập nhật hướng ray
    float3 new_dir = update_direction(ray->direction, cos_theta, sin_theta, phi);
    ray->direction = new_dir;

    // Năng lượng không thay đổi trong tán xạ Rayleigh
}

// Các hàm tiện ích khác...
__device__ int3 get_voxel_index(float3 position, VolumeInfo volume_info) {
    int3 idx;
    idx.x = (int)((position.x - volume_info.origin.x) / volume_info.spacing.x);
    idx.y = (int)((position.y - volume_info.origin.y) / volume_info.spacing.y);
    idx.z = (int)((position.z - volume_info.origin.z) / volume_info.spacing.z);
    return idx;
}

__device__ bool check_inside_volume(float3 position, VolumeInfo volume_info) {
    int3 idx = get_voxel_index(position, volume_info);
    return (idx.x >= 0 && idx.x < volume_info.size.x &&
            idx.y >= 0 && idx.y < volume_info.size.y &&
            idx.z >= 0 && idx.z < volume_info.size.z);
}

__device__ int flatten_index(int3 idx, int3 size) {
    return idx.x + idx.y * size.x + idx.z * size.x * size.y;
}
"""

# OpenCL kernel code cho tính toán Monte Carlo
OPENCL_KERNEL_SOURCE = """
// Tương tự như CUDA nhưng với cú pháp OpenCL
// Các hằng số vật lý
#define ELECTRON_REST_MASS 0.511f  // MeV
#define LIGHT_SPEED 299.792458f    // mm/ns
#define PI 3.14159265359f

// Cấu trúc cho ray
typedef struct {
    float3 origin;
    float3 direction;
    float energy;
    int type;  // 0 = photon, 1 = electron
} Ray;

// Cấu trúc cho dữ liệu CT và liều
typedef struct {
    int3 size;
    float3 spacing;
    float3 origin;
} VolumeInfo;

// Các prototype
void simulate_compton_scattering(Ray* ray, uint* seed);
void simulate_rayleigh_scattering(Ray* ray, uint* seed);
int3 get_voxel_index(float3 position, VolumeInfo volume_info);
bool check_inside_volume(float3 position, VolumeInfo volume_info);
int flatten_index(int3 idx, int3 size);
float rand_uniform(uint* seed);

// Hàm random đơn giản
float rand_uniform(uint* seed) {
    // Thuật toán LCG đơn giản
    *seed = (*seed * 1664525 + 1013904223) & 0xFFFFFFFF;
    return (float)(*seed) / (float)0xFFFFFFFF;
}

// Kernel chính cho mô phỏng photon
__kernel void simulate_photons(
    __global float* ct_data,
    __global float* dose_data,
    __global float* material_data,
    int num_materials,
    VolumeInfo volume_info,
    float3 source_pos,
    __global float3* beam_directions,
    __global float* beam_energies,
    int num_histories_per_thread,
    int global_seed
) {
    // Lấy chỉ số thread
    int thread_id = get_global_id(0);

    // Khởi tạo hạt giống ngẫu nhiên riêng cho mỗi thread
    uint seed = global_seed + thread_id;

    // Mô phỏng nhiều hạt
    for (int i = 0; i < num_histories_per_thread; i++) {
        // Chọn hướng và năng lượng chùm tia từ phân phối
        int beam_idx = (int)(rand_uniform(&seed) * num_histories_per_thread);
        if (beam_idx >= num_histories_per_thread) beam_idx = num_histories_per_thread - 1;

        float3 dir = beam_directions[beam_idx];
        float energy = beam_energies[beam_idx];

        // Tạo photon ban đầu
        Ray ray;
        ray.origin = source_pos;
        ray.direction = dir;
        ray.energy = energy;
        ray.type = 0;  // photon

        // Theo dõi photon
        while (ray.energy > 0.01f) {  // 10 keV cutoff
            // Kiểm tra nếu photon vẫn trong phantom
            bool inside = check_inside_volume(ray.origin, volume_info);
            if (!inside) break;

            // Lấy chỉ số voxel
            int3 voxel_idx = get_voxel_index(ray.origin, volume_info);
            int flat_idx = flatten_index(voxel_idx, volume_info.size);

            // Xử lý tương tác...
            // [code tương tự như trong CUDA kernel]

            // Tương tự như CUDA, nhưng với atomic_add thay vì atomicAdd:
            atomic_add(&dose_data[flat_idx], ray.energy);
        }
    }
}
"""


class MonteCarloGPUKernels:
    """Class chứa các kernel và tiện ích GPU cho Monte Carlo."""

    # Các hằng số vật lý
    ELECTRON_REST_MASS = 0.511  # MeV
    LIGHT_SPEED = 299.792458  # mm/ns

    def __init__(self):
        """Khởi tạo MonteCarloGPUKernels."""
        self.cuda_module = None
        self.opencl_program = None
        self.opencl_context = None
        self.opencl_queue = None
        self.cuda_context = None
        self.has_cuda = False
        self.has_opencl = False
        self.has_gpu = False
        self.device_name = "CPU fallback"
        self.gpu_backend = None

        # Dữ liệu vật lý
        self.material_data = {}
        self.attenuation_data = {}
        self.cross_section_data = {}

        # Kiểm tra các thư viện có sẵn
        self.available_backends = self._check_available_libraries()
        logger.info(f"Các backend GPU có sẵn: {self.available_backends}")

    def _check_available_libraries(self) -> List[str]:
        """Kiểm tra các thư viện GPU có sẵn.

        Returns
        -------
        List[str]
            Danh sách tên các backend có sẵn
        """
        backends = []

        if HAS_CUPY:
            try:
                # Thử tạo một mảng CuPy để xác nhận GPU có hoạt động không
                test_array = cp.zeros(10)
                backends.append("cupy")
                logger.debug("CuPy (CUDA) khả dụng và đang hoạt động")
            except Exception as e:
                logger.warning(
                    f"CuPy được import nhưng không thể sử dụng GPU: {str(e)}"
                )

        if HAS_PYCUDA:
            try:
                # Kiểm tra PyCUDA có thể truy cập GPU không
                device_count = drv.Device.count()
                if device_count > 0:
                    backends.append("pycuda")
                    logger.debug(f"PyCUDA khả dụng với {device_count} thiết bị")
            except Exception as e:
                logger.warning(
                    f"PyCUDA được import nhưng không thể sử dụng GPU: {str(e)}"
                )

        if HAS_OPENCL:
            try:
                platforms = cl.get_platforms()
                if platforms:
                    backends.append("pyopencl")
                    logger.debug(f"PyOpenCL khả dụng với {len(platforms)} platforms")
            except Exception as e:
                logger.warning(
                    f"PyOpenCL được import nhưng không thể sử dụng: {str(e)}"
                )

        return backends

    def build_cuda_module(self, code: Optional[str] = None) -> bool:
        """Biên dịch CUDA kernel."""
        if not self.has_cuda:
            logger.warning("CUDA không khả dụng, không thể biên dịch module")
            return False

        try:
            cuda_code = code if code is not None else CUDA_KERNEL_SOURCE

            if self.gpu_backend == "cupy":
                # Biên dịch kernel với CuPy
                self.cuda_module = cp.RawModule(code=cuda_code)
                logger.info("Đã biên dịch CUDA module với CuPy")
            elif self.gpu_backend == "pycuda":
                # Biên dịch kernel với PyCUDA
                if HAS_PYCUDA:
                    self.cuda_module = compiler.SourceModule(cuda_code)
                    logger.info("Đã biên dịch CUDA module với PyCUDA")

            return True
        except Exception as e:
            logger.error(f"Lỗi khi biên dịch CUDA module: {str(e)}")
            return False

    def build_opencl_program(self, context, code: Optional[str] = None):
        """Biên dịch OpenCL program."""
        if not self.has_opencl:
            logger.warning("OpenCL không khả dụng, không thể biên dịch program")
            return False

        try:
            opencl_code = code if code is not None else OPENCL_KERNEL_SOURCE

            self.opencl_context = context
            self.opencl_queue = cl.CommandQueue(context)

            # Biên dịch OpenCL program
            self.opencl_program = cl.Program(context, opencl_code).build()
            logger.info("Đã biên dịch OpenCL program thành công")

            return True
        except Exception as e:
            logger.error(f"Lỗi khi biên dịch OpenCL program: {str(e)}")
            return False

    def load_physical_data(self, data_path: Optional[str] = None):
        """
        Nạp dữ liệu vật lý cho mô phỏng (hệ số suy giảm, mặt cắt, v.v.).

        Parameters
        ----------
        data_path : str, optional
            Đường dẫn đến thư mục chứa dữ liệu vật lý
        """
        try:
            if data_path is None or not os.path.exists(data_path):
                # Sử dụng dữ liệu mặc định
                self._load_default_physical_data()
                logger.info("Đã nạp dữ liệu vật lý mặc định")
            else:
                # Nạp dữ liệu vật lý từ file
                try:
                    self._load_physical_data_from_files(data_path)
                    logger.info(f"Đã nạp dữ liệu vật lý từ {data_path}")
                except Exception as e:
                    logger.error(f"Lỗi khi nạp dữ liệu vật lý từ file: {str(e)}")
                    logger.info("Sử dụng dữ liệu vật lý mặc định")
                    self._load_default_physical_data()

            return True
        except Exception as e:
            logger.error(f"Lỗi khi nạp dữ liệu vật lý: {str(e)}")
            return False

    def _load_default_physical_data(self):
        """Nạp dữ liệu vật lý mặc định."""
        # Dữ liệu hệ số suy giảm cho các vật liệu khác nhau
        # Format: {energy: [total, compton, photoelectric, rayleigh]}
        # Energy trong MeV
        self.attenuation_data = {
            # Nước
            "water": {
                0.01: [4.078, 0.0169, 4.061, 0.0],
                0.1: [0.1707, 0.0953, 0.0754, 0.0],
                1.0: [0.0706, 0.0562, 0.0008, 0.0136],
                5.0: [0.0263, 0.0179, 0.00001, 0.0084],
                10.0: [0.0192, 0.0125, 0.000001, 0.0067],
            },
            # Xương
            "bone": {
                0.01: [27.19, 0.0127, 27.18, 0.0],
                0.1: [0.4864, 0.0805, 0.4059, 0.0],
                1.0: [0.1133, 0.0521, 0.0092, 0.052],
                5.0: [0.0412, 0.0167, 0.0001, 0.0244],
                10.0: [0.0289, 0.0115, 0.00001, 0.0174],
            },
            # Mô mềm
            "soft_tissue": {
                0.01: [4.94, 0.0158, 4.923, 0.0],
                0.1: [0.1687, 0.0891, 0.0796, 0.0],
                1.0: [0.0696, 0.0526, 0.0009, 0.0161],
                5.0: [0.0259, 0.0167, 0.00001, 0.0092],
                10.0: [0.0188, 0.0117, 0.000001, 0.0071],
            },
            # Phổi
            "lung": {
                0.01: [1.591, 0.0155, 1.576, 0.0],
                0.1: [0.0661, 0.0371, 0.029, 0.0],
                1.0: [0.0273, 0.0207, 0.0003, 0.0063],
                5.0: [0.0102, 0.0066, 0.000004, 0.0036],
                10.0: [0.0074, 0.0046, 0.0000005, 0.0028],
            },
            # Không khí
            "air": {
                0.01: [0.1493, 0.0005, 0.1488, 0.0],
                0.1: [0.0156, 0.0087, 0.0069, 0.0],
                1.0: [0.0065, 0.0049, 0.0001, 0.0015],
                5.0: [0.0025, 0.0016, 0.000001, 0.0009],
                10.0: [0.0018, 0.0011, 0.0000001, 0.0007],
            },
        }

        # Mật độ cho các vật liệu (g/cm³)
        self.material_data = {
            "water": 1.0,
            "bone": 1.85,
            "soft_tissue": 1.04,
            "lung": 0.3,
            "air": 0.00129,
        }

        # Bảng chuyển đổi HU sang vật liệu
        self.hu_to_material = {
            (-1000, -800): "air",
            (-800, -200): "lung",
            (-200, 50): "soft_tissue",
            (50, 1000): "bone",
            (1000, 3000): "bone",
        }

        logger.info("Đã nạp dữ liệu vật lý mặc định")

    def _load_physical_data_from_files(self, data_path: str):
        """
        Nạp dữ liệu vật lý từ các file.

        Parameters
        ----------
        data_path : str
            Đường dẫn đến thư mục chứa dữ liệu vật lý
        """
        # Kiểm tra xem đường dẫn có tồn tại không
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Đường dẫn không tồn tại: {data_path}")

        # Nạp dữ liệu từ các file
        attenuation_file = os.path.join(data_path, "attenuation.npz")
        material_file = os.path.join(data_path, "materials.npz")

        if os.path.exists(attenuation_file):
            # Nạp dữ liệu suy giảm
            attenuation_data = np.load(attenuation_file, allow_pickle=True)
            self.attenuation_data = attenuation_data["data"].item()
            logger.info("Đã nạp dữ liệu suy giảm từ file")
        else:
            logger.warning(f"Không tìm thấy file dữ liệu suy giảm: {attenuation_file}")
            # Sử dụng dữ liệu mặc định nếu không có file
            self._load_default_physical_data()

        if os.path.exists(material_file):
            # Nạp dữ liệu vật liệu
            material_data = np.load(material_file, allow_pickle=True)
            self.material_data = material_data["data"].item()
            if "hu_to_material" in material_data:
                self.hu_to_material = material_data["hu_to_material"].item()
            logger.info("Đã nạp dữ liệu vật liệu từ file")
        else:
            logger.warning(f"Không tìm thấy file dữ liệu vật liệu: {material_file}")


# Tạo một instance của class để có thể import
kernels = MonteCarloGPUKernels()
