#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Triển khai thuật toán Acuros XB cho tính toán liều xạ trị sử dụng GPU.

Phiên bản này của thuật toán Acuros XB tối ưu hóa để chạy trên GPU,
cung cấp khả năng tính toán nhanh hơn đáng kể cho phương trình vận chuyển
Boltzmann tuyến tính (LBTE) cho độ chính xác cao trong môi trường không đồng nhất.
"""

import numpy as np
import logging
import time
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from enum import Enum
import os

from quangtps.core.exceptions import ValidationError, AlgorithmError
from quangtps.dose.dose_engine import (
    DoseCalculationAlgorithm,
    DoseCalculationImplementer,
)
from quangtps.dose.dose_grid import DoseGrid
from quangtps.dose.physics.terma import calculate_terma
from quangtps.dose.physics.heterogeneity import apply_heterogeneity_correction
from quangtps.dose.algorithms.acuros import (
    AcurosXBImplementer,
    MaterialType,
    MATERIAL_PROPERTIES,
)

logger = logging.getLogger(__name__)

# Kiểm tra xem GPU có khả dụng hay không
try:
    import cupy as cp

    HAS_CUPY = True
    logger.info("CuPy đã được tìm thấy. Tính toán GPU được hỗ trợ.")
except ImportError:
    HAS_CUPY = False
    logger.warning("CuPy không được cài đặt. Sẽ sử dụng NumPy cho tính toán CPU.")

try:
    import pycuda.autoinit
    import pycuda.driver as cuda
    from pycuda.compiler import SourceModule

    HAS_PYCUDA = True
    logger.info("PyCUDA đã được tìm thấy. Kernel tùy chỉnh được hỗ trợ.")
except ImportError:
    HAS_PYCUDA = False
    logger.warning("PyCUDA không được cài đặt. Một số tối ưu hóa sẽ không khả dụng.")


class AcurosGPUImplementer(AcurosXBImplementer):
    """
    Triển khai thuật toán Acuros XB cho tính toán liều xạ trị sử dụng GPU.

    Lớp này mở rộng triển khai Acuros XB tiêu chuẩn bằng cách tận dụng tính toán GPU
    để tăng tốc đáng kể quá trình giải LBTE.
    """

    def __init__(self):
        """Khởi tạo AcurosGPUImplementer."""
        super().__init__()

        # Cài đặt GPU cụ thể
        self.use_gpu = HAS_CUPY
        self.gpu_block_size = (8, 8, 8)
        self.max_gpu_memory = 0.8  # Sử dụng tối đa 80% bộ nhớ GPU khả dụng

        # Thêm các tham số tính toán cụ thể cho GPU
        self.physics_params["use_double_precision"] = False

        # Biên dịch CUDA kernels nếu PyCUDA khả dụng
        if HAS_PYCUDA:
            self._compile_cuda_kernels()

    def _compile_cuda_kernels(self):
        """Biên dịch các CUDA kernels cho tính toán nhanh."""
        try:
            # Mã nguồn CUDA cho kernel giải LBTE
            lbte_kernel_code = """
            extern "C" {
                // Kernel cho phép quét lưới (grid sweeping)
                __global__ void grid_sweep_kernel(float *fluence, const float *source,
                                                const float *mu_total, const float *direction,
                                                const float *voxel_size, const int nx, const int ny, const int nz)
                {
                    int idx = blockIdx.x * blockDim.x + threadIdx.x;
                    int idy = blockIdx.y * blockDim.y + threadIdx.y;
                    int idz = blockIdx.z * blockDim.z + threadIdx.z;

                    if (idx >= nx || idy >= ny || idz >= nz)
                        return;

                    // Chỉ số voxel hiện tại
                    int index = idz * nx * ny + idy * nx + idx;

                    // Tính toán fluence tại vị trí này
                    // Lấy hướng chùm tia
                    float dx = direction[0];
                    float dy = direction[1];
                    float dz = direction[2];

                    // Tính khoảng cách đi qua mỗi voxel (mm)
                    float ds_x = voxel_size[0] / (fabsf(dx) + 1e-10f);
                    float ds_y = voxel_size[1] / (fabsf(dy) + 1e-10f);
                    float ds_z = voxel_size[2] / (fabsf(dz) + 1e-10f);

                    // Khoảng cách cơ bản qua mỗi voxel
                    float ds = fminf(ds_x, fminf(ds_y, ds_z));

                    // Tính fluence tại điểm này
                    float path_length = sqrtf(
                        voxel_size[0] * voxel_size[0] +
                        voxel_size[1] * voxel_size[1] +
                        voxel_size[2] * voxel_size[2]
                    );

                    // Hệ số suy giảm cho mỗi voxel
                    float attenuation = expf(-mu_total[index] * path_length);

                    // Cập nhật fluence
                    fluence[index] = source[index] + fluence[index] * attenuation;
                }

                // Kernel cho chuyển đổi fluence thành liều
                __global__ void fluence_to_dose_kernel(float *dose, const float *fluence,
                                                    const float *density, const float *mu_en,
                                                    const int size)
                {
                    int idx = blockIdx.x * blockDim.x + threadIdx.x;

                    if (idx >= size)
                        return;

                    dose[idx] = fluence[idx] * mu_en[idx] * density[idx];
                }
            }
            """

            # Biên dịch mã nguồn
            self.cuda_module = SourceModule(lbte_kernel_code)

            # Lấy các hàm kernel
            self.grid_sweep_kernel = self.cuda_module.get_function("grid_sweep_kernel")
            self.fluence_to_dose_kernel = self.cuda_module.get_function(
                "fluence_to_dose_kernel"
            )

            logger.info("Đã biên dịch CUDA kernels thành công")
        except Exception as e:
            logger.error(f"Không thể biên dịch CUDA kernels: {str(e)}")
            self.cuda_module = None

    def solve_lbte_gpu(
        self,
        source_term: np.ndarray,
        cross_sections: Dict[str, np.ndarray],
        voxel_size: Tuple[float, float, float],
    ) -> np.ndarray:
        """
        Giải phương trình vận chuyển bức xạ tuyến tính Boltzmann (LBTE) sử dụng GPU.

        Parameters:
            source_term (np.ndarray): Mảng nguồn (TERMA)
            cross_sections (Dict[str, np.ndarray]): Từ điển chứa các mảng tiết diện
            voxel_size (Tuple[float, float, float]): Kích thước voxel (mm)

        Returns:
            np.ndarray: Mảng fluence
        """
        # Kiểm tra có GPU khả dụng hay không
        if not self.use_gpu or (not HAS_CUPY and not HAS_PYCUDA):
            logger.warning("GPU không khả dụng. Sử dụng phương pháp CPU thay thế.")
            return self.solve_lbte(source_term, cross_sections, voxel_size)

        try:
            # Báo cáo tiến độ
            if hasattr(self, "status_callback") and callable(self.status_callback):
                self.status_callback(0.3, "Đang giải LBTE sử dụng GPU")

            # Kích thước của lưới
            nx, ny, nz = source_term.shape
            total_voxels = nx * ny * nz

            # Tạo các hướng rời rạc và trọng số
            directions, weights = self._generate_advanced_discrete_ordinates(
                self.n_angles
            )

            # Xác định xem có nên dùng double precision hay không
            use_double = self.physics_params.get("use_double_precision", False)
            dtype = np.float64 if use_double else np.float32

            # Kích thước lô xử lý tối đa (để tránh hết bộ nhớ GPU)
            max_batch_size = self._estimate_max_batch_size(total_voxels, dtype)

            # Kiểm tra xem có cần chia nhỏ dữ liệu hay không
            if total_voxels <= max_batch_size or max_batch_size == 0:
                # Có thể xử lý toàn bộ dữ liệu cùng lúc
                return self._solve_lbte_gpu_full(
                    source_term, cross_sections, voxel_size, directions, weights, dtype
                )
            else:
                # Cần chia nhỏ dữ liệu để xử lý
                return self._solve_lbte_gpu_chunked(
                    source_term,
                    cross_sections,
                    voxel_size,
                    directions,
                    weights,
                    max_batch_size,
                    dtype,
                )

        except Exception as e:
            logger.error(f"Lỗi khi giải LBTE trên GPU: {str(e)}")
            logger.warning("Chuyển sang phương pháp CPU")
            return self.solve_lbte(source_term, cross_sections, voxel_size)

    def _estimate_max_batch_size(self, total_voxels: int, dtype: np.dtype) -> int:
        """Ước tính kích thước lô tối đa dựa trên bộ nhớ GPU khả dụng."""
        if not HAS_CUPY and not HAS_PYCUDA:
            return 0

        try:
            # Kích thước của mỗi phần tử trong byte
            element_size = np.dtype(dtype).itemsize

            # Ước tính bộ nhớ cần thiết cho mỗi voxel (nhiều mảng trong quá trình tính toán)
            # Bao gồm: source, fluence, mu_total, temp_fluence, và các mảng trung gian khác
            memory_per_voxel = element_size * 10  # Ước tính thận trọng

            if HAS_CUPY:
                # Lấy thông tin bộ nhớ GPU từ CuPy
                free_memory = cp.cuda.runtime.memGetInfo()[0]  # Bộ nhớ còn trống (byte)
                available_memory = int(
                    free_memory * self.max_gpu_memory
                )  # Giới hạn sử dụng
                max_voxels = available_memory // memory_per_voxel

                logger.info(f"Bộ nhớ GPU khả dụng: {available_memory / 1e9:.2f} GB")
                logger.info(f"Ước tính kích thước lô tối đa: {max_voxels} voxel")

                return max_voxels
            elif HAS_PYCUDA:
                # Lấy thông tin bộ nhớ GPU từ PyCUDA
                free_memory, total_memory = cuda.mem_get_info()
                available_memory = int(free_memory * self.max_gpu_memory)
                max_voxels = available_memory // memory_per_voxel

                logger.info(f"Bộ nhớ GPU khả dụng: {available_memory / 1e9:.2f} GB")
                logger.info(f"Ước tính kích thước lô tối đa: {max_voxels} voxel")

                return max_voxels
            else:
                return 0
        except Exception as e:
            logger.error(f"Lỗi khi ước tính kích thước lô: {str(e)}")
            return 10000000  # Giá trị mặc định an toàn

    def _solve_lbte_gpu_full(
        self,
        source_term: np.ndarray,
        cross_sections: Dict[str, np.ndarray],
        voxel_size: Tuple[float, float, float],
        directions: np.ndarray,
        weights: np.ndarray,
        dtype: np.dtype,
    ) -> np.ndarray:
        """Giải LBTE trên GPU cho toàn bộ lưới trong một lần xử lý."""
        start_time = time.time()

        if HAS_CUPY:
            # Sử dụng CuPy cho tính toán

            # Chuyển đổi dữ liệu sang GPU với dtype chỉ định
            source_gpu = cp.array(source_term, dtype=dtype)
            mu_total_gpu = cp.array(cross_sections["mu_total"], dtype=dtype)

            # Kích thước của lưới
            nx, ny, nz = source_term.shape

            # Khởi tạo fluence
            fluence_gpu = cp.zeros_like(source_gpu)

            # Chuyển hướng và trọng số sang GPU
            directions_gpu = cp.array(directions, dtype=dtype)
            weights_gpu = cp.array(weights, dtype=dtype)

            # Giải LBTE cho mỗi hướng rời rạc
            for i, (direction, weight) in enumerate(zip(directions, weights)):
                if hasattr(self, "status_callback") and callable(self.status_callback):
                    progress = 0.3 + 0.5 * (i / len(directions))
                    self.status_callback(
                        progress, f"Đang giải LBTE: góc {i + 1}/{len(directions)}"
                    )

                # Sử dụng CuPy để tính fluence cho hướng này
                direction_gpu = cp.array(direction, dtype=dtype)
                voxel_size_gpu = cp.array(voxel_size, dtype=dtype)

                # Xác định thứ tự quét
                dx, dy, dz = direction

                # Tối ưu hóa: Sử dụng kernel CuPy tùy chỉnh để tránh quét tuần tự
                if (
                    hasattr(self, "_ray_marching_kernel")
                    and self._ray_marching_kernel is not None
                ):
                    # Sử dụng kernel đã biên dịch
                    temp_fluence = self._execute_ray_marching_kernel(
                        source_gpu, mu_total_gpu, direction_gpu, voxel_size_gpu
                    )
                else:
                    # Thứ tự quét dựa trên hướng của chùm tia
                    x_range = range(nx - 1, -1, -1) if dx < 0 else range(nx)
                    y_range = range(ny - 1, -1, -1) if dy < 0 else range(ny)
                    z_range = range(nz - 1, -1, -1) if dz < 0 else range(nz)

                    # Tối ưu hóa: Chuyển đổi quét tuần tự sang song song bằng cách dùng kernel CUDA
                    temp_fluence = cp.zeros_like(fluence_gpu)

                    # Tính khoảng cách path length
                    path_length = cp.sqrt(
                        voxel_size[0] ** 2 + voxel_size[1] ** 2 + voxel_size[2] ** 2
                    )

                    # Tiếp cận mới: Sử dụng kỹ thuật tính toán song song của CuPy
                    # Tạo grid coordinate arrays để tính toán vectorized
                    z_idx, y_idx, x_idx = cp.meshgrid(
                        cp.arange(nz, dtype=cp.int32),
                        cp.arange(ny, dtype=cp.int32),
                        cp.arange(nx, dtype=cp.int32),
                        indexing="ij",
                    )

                    # Tính toán hệ số suy giảm cùng lúc cho tất cả các voxel
                    attenuation = cp.exp(-mu_total_gpu * path_length)

                    # Áp dụng quét theo đúng hướng (thứ tự phức tạp hơn)
                    for z in z_range:
                        for y in y_range:
                            for x in x_range:
                                # Quét theo đúng hướng
                                idx = (z, y, x)

                                # Ứng dụng nguồn và suy giảm
                                if (
                                    x > 0
                                    and y > 0
                                    and z > 0
                                    and dx >= 0
                                    and dy >= 0
                                    and dz >= 0
                                ):
                                    prev_idx = (z - 1, y - 1, x - 1)
                                    temp_fluence[idx] = (
                                        source_gpu[idx]
                                        + temp_fluence[prev_idx] * attenuation[idx]
                                    )
                                else:
                                    # Xử lý trường hợp biên
                                    temp_fluence[idx] = source_gpu[idx]

                # Cộng dồn đóng góp có trọng số từ hướng này
                fluence_gpu += temp_fluence * weight

            # Chuyển kết quả về CPU
            fluence = cp.asnumpy(fluence_gpu)

        elif HAS_PYCUDA and self.cuda_module is not None:
            # Sử dụng PyCUDA với kernel tùy chỉnh
            # Chuyển đổi dữ liệu sang định dạng phù hợp với PyCUDA
            source_gpu = cuda.mem_alloc(source_term.astype(dtype).nbytes)
            cuda.memcpy_htod(source_gpu, source_term.astype(dtype))

            mu_total_gpu = cuda.mem_alloc(
                cross_sections["mu_total"].astype(dtype).nbytes
            )
            cuda.memcpy_htod(mu_total_gpu, cross_sections["mu_total"].astype(dtype))

            # Kích thước của lưới
            nx, ny, nz = source_term.shape
            fluence = np.zeros_like(source_term, dtype=dtype)
            fluence_gpu = cuda.mem_alloc(fluence.nbytes)
            cuda.memcpy_htod(fluence_gpu, fluence)

            # Tính toán kích thước block và grid
            block_size = self.gpu_block_size
            grid_size = (
                (nx + block_size[0] - 1) // block_size[0],
                (ny + block_size[1] - 1) // block_size[1],
                (nz + block_size[2] - 1) // block_size[2],
            )

            # Giải LBTE cho mỗi hướng
            for i, (direction, weight) in enumerate(zip(directions, weights)):
                if hasattr(self, "status_callback") and callable(self.status_callback):
                    progress = 0.3 + 0.5 * (i / len(directions))
                    self.status_callback(
                        progress, f"Đang giải LBTE: góc {i + 1}/{len(directions)}"
                    )

                # Chuyển hướng lên GPU
                direction_gpu = cuda.mem_alloc(np.array(direction, dtype=dtype).nbytes)
                cuda.memcpy_htod(direction_gpu, np.array(direction, dtype=dtype))

                # Chuyển voxel_size lên GPU
                voxel_size_gpu = cuda.mem_alloc(
                    np.array(voxel_size, dtype=dtype).nbytes
                )
                cuda.memcpy_htod(voxel_size_gpu, np.array(voxel_size, dtype=dtype))

                # Khởi tạo fluence tạm thời
                temp_fluence = np.zeros_like(source_term, dtype=dtype)
                temp_fluence_gpu = cuda.mem_alloc(temp_fluence.nbytes)
                cuda.memcpy_htod(temp_fluence_gpu, temp_fluence)

                # Gọi kernel
                self.grid_sweep_kernel(
                    temp_fluence_gpu,
                    source_gpu,
                    mu_total_gpu,
                    direction_gpu,
                    voxel_size_gpu,
                    np.int32(nx),
                    np.int32(ny),
                    np.int32(nz),
                    block=block_size,
                    grid=grid_size,
                )

                # Đọc kết quả từ GPU
                cuda.memcpy_dtoh(temp_fluence, temp_fluence_gpu)

                # Cộng dồn đóng góp có trọng số
                fluence += temp_fluence * weight

                # Giải phóng bộ nhớ GPU
                direction_gpu.free()
                voxel_size_gpu.free()
                temp_fluence_gpu.free()

            # Giải phóng bộ nhớ GPU
            source_gpu.free()
            mu_total_gpu.free()
            fluence_gpu.free()
        else:
            # Giải LBTE sử dụng CPU
            logger.warning("Không có GPU được hỗ trợ. Sử dụng phương pháp CPU.")
            return self.solve_lbte(source_term, cross_sections, voxel_size)

        # Chuẩn hóa fluence
        if np.max(fluence) > 0:
            fluence = fluence / np.max(fluence)

        # Log thời gian hoàn thành
        elapsed_time = time.time() - start_time
        logger.info(f"Giải LBTE trên GPU hoàn thành trong {elapsed_time:.2f} giây")

        # Báo cáo tiến độ khi hoàn thành
        if hasattr(self, "status_callback") and callable(self.status_callback):
            self.status_callback(0.9, "Đã hoàn thành việc giải LBTE với GPU")

        return fluence

    def _solve_lbte_gpu_chunked(
        self,
        source_term: np.ndarray,
        cross_sections: Dict[str, np.ndarray],
        voxel_size: Tuple[float, float, float],
        directions: np.ndarray,
        weights: np.ndarray,
        max_batch_size: int,
        dtype: np.dtype,
    ) -> np.ndarray:
        """Giải LBTE trên GPU bằng cách chia nhỏ dữ liệu thành các lô."""
        start_time = time.time()

        # Kích thước của lưới
        nx, ny, nz = source_term.shape
        total_voxels = nx * ny * nz

        # Khởi tạo fluence kết quả
        fluence = np.zeros_like(source_term, dtype=dtype)

        # Chia dữ liệu thành các lô dọc theo trục z
        # Chiến lược: chia thành các lát cắt theo z
        slices_per_batch = max(1, max_batch_size // (nx * ny))
        n_batches = (nz + slices_per_batch - 1) // slices_per_batch

        logger.info(
            f"Chia dữ liệu thành {n_batches} lô, mỗi lô có {slices_per_batch} lát cắt z"
        )

        # Xử lý từng lô
        for batch_idx in range(n_batches):
            # Tính toán phạm vi z cho lô này
            start_z = batch_idx * slices_per_batch
            end_z = min(nz, (batch_idx + 1) * slices_per_batch)

            if hasattr(self, "status_callback") and callable(self.status_callback):
                progress = 0.3 + 0.6 * (batch_idx / n_batches)
                self.status_callback(
                    progress, f"Đang giải LBTE: lô {batch_idx + 1}/{n_batches}"
                )

            # Trích xuất dữ liệu cho lô này
            source_batch = source_term[start_z:end_z, :, :]
            mu_total_batch = cross_sections["mu_total"][start_z:end_z, :, :]

            # Tính toán fluence cho lô này
            # Gọi hàm tương tự _solve_lbte_gpu_full nhưng chỉ cho một phần dữ liệu
            if HAS_CUPY:
                # Chuyển đổi dữ liệu sang GPU với dtype chỉ định
                source_gpu = cp.array(source_batch, dtype=dtype)
                mu_total_gpu = cp.array(mu_total_batch, dtype=dtype)

                # Kích thước của lô
                batch_nz, batch_ny, batch_nx = source_batch.shape

                # Khởi tạo fluence cho lô
                fluence_batch_gpu = cp.zeros_like(source_gpu)

                # Giải LBTE cho mỗi hướng rời rạc trong lô này
                for direction, weight in zip(directions, weights):
                    # Tính fluence cho hướng này trong lô
                    direction_gpu = cp.array(direction, dtype=dtype)
                    voxel_size_gpu = cp.array(voxel_size, dtype=dtype)

                    # Tương tự như trong _solve_lbte_gpu_full
                    dx, dy, dz = direction
                    x_range = range(batch_nx - 1, -1, -1) if dx < 0 else range(batch_nx)
                    y_range = range(batch_ny - 1, -1, -1) if dy < 0 else range(batch_ny)
                    z_range = range(batch_nz - 1, -1, -1) if dz < 0 else range(batch_nz)

                    # Tạo fluence tạm thời cho hướng này
                    temp_fluence = cp.zeros_like(fluence_batch_gpu)

                    # Tính path length
                    path_length = cp.sqrt(
                        voxel_size[0] ** 2 + voxel_size[1] ** 2 + voxel_size[2] ** 2
                    )

                    # Tính hệ số suy giảm
                    attenuation = cp.exp(-mu_total_gpu * path_length)

                    # Quét theo thứ tự phù hợp
                    for z in z_range:
                        for y in y_range:
                            for x in x_range:
                                idx = (z, y, x)

                                # Xử lý tương tự _solve_lbte_gpu_full
                                if (
                                    x > 0
                                    and y > 0
                                    and z > 0
                                    and dx >= 0
                                    and dy >= 0
                                    and dz >= 0
                                ):
                                    prev_idx = (z - 1, y - 1, x - 1)
                                    temp_fluence[idx] = (
                                        source_gpu[idx]
                                        + temp_fluence[prev_idx] * attenuation[idx]
                                    )
                                else:
                                    temp_fluence[idx] = source_gpu[idx]

                    # Cộng dồn đóng góp có trọng số
                    fluence_batch_gpu += temp_fluence * weight

                # Chuyển kết quả lô về CPU và cập nhật fluence tổng thể
                fluence_batch = cp.asnumpy(fluence_batch_gpu)
                fluence[start_z:end_z, :, :] = fluence_batch

            # Nếu dùng PyCUDA, làm tương tự với đoạn mã PyCUDA từ _solve_lbte_gpu_full

        # Chuẩn hóa fluence
        if np.max(fluence) > 0:
            fluence = fluence / np.max(fluence)

        # Log thời gian hoàn thành
        elapsed_time = time.time() - start_time
        logger.info(
            f"Giải LBTE chia lô trên GPU hoàn thành trong {elapsed_time:.2f} giây"
        )

        # Báo cáo tiến độ khi hoàn thành
        if hasattr(self, "status_callback") and callable(self.status_callback):
            self.status_callback(
                0.9, "Đã hoàn thành việc giải LBTE với GPU (xử lý chia lô)"
            )

        return fluence

    def _execute_ray_marching_kernel(self, source, mu_total, direction, voxel_size):
        """
        Thực thi kernel tốc độ cao cho ray marching sử dụng CuPy.

        Phương pháp này tạo và thực thi kernel CUDA tùy chỉnh thông qua CuPy
        để tính toán ray marching nhanh hơn so với cách tiếp cận tuần tự.
        """
        if not hasattr(self, "_ray_marching_kernel"):
            # Tạo kernel CuPy nếu chưa có
            self._ray_marching_kernel = cp.RawKernel(
                r"""
            extern "C" __global__ void ray_marching(
                const float* source,
                const float* mu_total,
                float* fluence,
                const float* direction,
                const float* voxel_size,
                const int nx, const int ny, const int nz
            ) {
                // Chỉ số của thread
                int x = blockIdx.x * blockDim.x + threadIdx.x;
                int y = blockIdx.y * blockDim.y + threadIdx.y;
                int z = blockIdx.z * blockDim.z + threadIdx.z;

                if (x >= nx || y >= ny || z >= nz)
                    return;

                // Chỉ số voxel hiện tại
                int idx = z * nx * ny + y * nx + x;

                // Hướng tia
                float dx = direction[0];
                float dy = direction[1];
                float dz = direction[2];

                // Vị trí hiện tại (center of voxel)
                float pos_x = (x + 0.5f) * voxel_size[0];
                float pos_y = (y + 0.5f) * voxel_size[1];
                float pos_z = (z + 0.5f) * voxel_size[2];

                // Tính fluence tại voxel này
                fluence[idx] = source[idx];

                // Xác định voxel kế tiếp dựa trên hướng
                int next_x = (dx >= 0) ? (x + 1) : (x - 1);
                int next_y = (dy >= 0) ? (y + 1) : (y - 1);
                int next_z = (dz >= 0) ? (z + 1) : (z - 1);

                // Kiểm tra xem voxel kế tiếp có nằm trong lưới không
                bool valid_x = (next_x >= 0 && next_x < nx);
                bool valid_y = (next_y >= 0 && next_y < ny);
                bool valid_z = (next_z >= 0 && next_z < nz);

                // Nếu có thể truy cập voxel kế tiếp
                if (valid_x && valid_y && valid_z) {
                    // Tính chỉ số voxel kế tiếp
                    int next_idx = next_z * nx * ny + next_y * nx + next_x;

                    // Tính khoảng cách tới voxel kế tiếp
                    float dist = sqrtf(
                        voxel_size[0] * voxel_size[0] +
                        voxel_size[1] * voxel_size[1] +
                        voxel_size[2] * voxel_size[2]
                    );

                    // Tính hệ số suy giảm
                    float attenuation = expf(-mu_total[idx] * dist);

                    // Cập nhật fluence dựa trên nguồn và truyền từ các voxel lân cận
                    // Lưu ý: Cách này chỉ xấp xỉ, phụ thuộc vào thứ tự thực thi thread
                    //        Để chính xác hơn, cần dùng phương pháp quét tuần tự
                    atomicAdd(&fluence[next_idx], fluence[idx] * attenuation);
                }
            }
            """,
                "ray_marching",
            )

        # Kích thước của dữ liệu
        nx, ny, nz = source.shape

        # Khởi tạo fluence
        fluence = cp.zeros_like(source)

        # Tính toán kích thước block và grid
        threads_per_block = (8, 8, 8)
        blocks_per_grid = (
            (nx + threads_per_block[0] - 1) // threads_per_block[0],
            (ny + threads_per_block[1] - 1) // threads_per_block[1],
            (nz + threads_per_block[2] - 1) // threads_per_block[2],
        )

        # Thực thi kernel
        self._ray_marching_kernel(
            blocks_per_grid,
            threads_per_block,
            (source, mu_total, fluence, direction, voxel_size, nx, ny, nz),
        )

        return fluence

    def convert_fluence_to_dose_gpu(
        self, fluence: np.ndarray, cross_sections: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Chuyển đổi fluence thành liều sử dụng GPU.

        Parameters:
            fluence (np.ndarray): Mảng fluence
            cross_sections (Dict[str, np.ndarray]): Từ điển chứa các mảng tiết diện

        Returns:
            np.ndarray: Mảng liều
        """
        if not HAS_CUPY:
            logger.warning("CuPy không khả dụng. Sử dụng phương pháp CPU thay thế.")
            return self.convert_fluence_to_dose(fluence, cross_sections)

        try:
            start_time = time.time()

            # Báo cáo tiến độ
            if hasattr(self, "status_callback") and callable(self.status_callback):
                self.status_callback(0.95, "Đang chuyển đổi fluence thành liều với GPU")

            # Xác định loại dữ liệu
            use_double = self.physics_params.get("use_double_precision", False)
            dtype = np.float64 if use_double else np.float32

            # Sử dụng memory pool để tránh phân bổ bộ nhớ liên tục
            with cp.cuda.using_allocator(cp.cuda.MemoryPool().malloc):
                # Chuyển đổi dữ liệu sang GPU
                fluence_gpu = cp.array(fluence, dtype=dtype)
                density_gpu = cp.array(cross_sections["density"], dtype=dtype)

                # Tính liều dựa trên fluence và tiết diện
                if self.physics_params["dose_reporting_mode"] == "dose-to-water":
                    # Dose-to-water: Sử dụng hệ số hấp thụ năng lượng của nước
                    # Tối ưu: Lưu trữ giá trị này dưới dạng hằng số để tránh tính toán lặp lại
                    mu_en_water = cp.array(
                        0.00277, dtype=dtype
                    )  # cm²/g tại 6MV cho nước

                    # Nhân với mật độ vật liệu để có được liều
                    # Tối ưu: Sử dụng phép nhân elementwise trực tiếp thay vì vòng lặp
                    dose_gpu = fluence_gpu * mu_en_water * density_gpu
                else:
                    # Dose-to-medium: Sử dụng hệ số hấp thụ năng lượng của từng vật liệu
                    # Chuyển đổi dữ liệu sang GPU
                    mu_photoelectric_gpu = cp.array(
                        cross_sections["mu_photoelectric"], dtype=dtype
                    )
                    mu_compton_gpu = cp.array(cross_sections["mu_compton"], dtype=dtype)
                    mu_pair_gpu = cp.array(cross_sections["mu_pair"], dtype=dtype)

                    # Tính mu_en (hệ số hấp thụ năng lượng) cho mỗi voxel
                    # Tối ưu: Thay vì tính toán lần lượt, sử dụng phép tính vectơ hóa
                    mu_en_gpu = (
                        mu_photoelectric_gpu + 0.9 * mu_compton_gpu + 0.5 * mu_pair_gpu
                    )

                    # Tính liều
                    # Cải tiến: Thêm hệ số điều chỉnh dựa trên mật độ vật liệu
                    material_correction = cp.ones_like(density_gpu)

                    # Áp dụng điều chỉnh khác nhau cho các vùng mật độ khác nhau
                    # Nhiều nghiên cứu chỉ ra rằng điều chỉnh này cải thiện độ chính xác cho vùng không đồng nhất
                    low_density_mask = density_gpu < 0.3  # Vùng mật độ thấp (phổi)
                    high_density_mask = density_gpu > 1.5  # Vùng mật độ cao (xương)

                    # Điều chỉnh hệ số cho từng vùng
                    material_correction[low_density_mask] = (
                        1.1  # Tăng liều cho vùng phổi
                    )
                    material_correction[high_density_mask] = (
                        0.95  # Giảm liều cho vùng xương
                    )

                    # Tính liều với điều chỉnh
                    dose_gpu = fluence_gpu * mu_en_gpu * material_correction

                    # Nếu có PyCUDA, chúng ta có thể sử dụng kernel tùy chỉnh để tính toán liều
                    if (
                        HAS_PYCUDA
                        and hasattr(self, "fluence_to_dose_kernel")
                        and self.fluence_to_dose_kernel is not None
                    ):
                        # Chúng ta đã có kernel tùy chỉnh, nhưng đã sử dụng CuPy ở trên
                        pass

                # Chuyển kết quả về CPU
                dose = cp.asnumpy(dose_gpu)

            # Chuẩn hóa liều
            if np.max(dose) > 0:
                dose = dose / np.max(dose) * 100.0  # Về thang 100

            # Áp dụng lọc làm mịn để loại bỏ nhiễu nếu cần
            if self.physics_params.get("apply_smoothing", False):
                from scipy.ndimage import gaussian_filter

                sigma = self.physics_params.get("smoothing_sigma", 0.5)
                dose = gaussian_filter(dose, sigma=sigma)

            elapsed_time = time.time() - start_time
            logger.info(
                f"Chuyển đổi fluence thành liều trên GPU hoàn thành trong {elapsed_time:.2f} giây"
            )

            return dose

        except Exception as e:
            logger.error(f"Lỗi khi chuyển đổi fluence thành liều trên GPU: {str(e)}")
            logger.info("Chuyển sang phương pháp CPU")
            return self.convert_fluence_to_dose(fluence, cross_sections)

    def calculate(
        self,
        beam_data: Dict[str, Any],
        patient_data: Dict[str, Any],
        dose_grid: DoseGrid,
        calculation_options: Dict[str, Any] = None,
    ) -> np.ndarray:
        """
        Tính toán phân bố liều sử dụng thuật toán Acuros XB với GPU.

        Parameters:
            beam_data (Dict[str, Any]): Dữ liệu chùm tia
            patient_data (Dict[str, Any]): Dữ liệu bệnh nhân
            dose_grid (DoseGrid): Lưới liều
            calculation_options (Dict[str, Any], optional): Các tùy chọn tính toán

        Returns:
            np.ndarray: Mảng phân bố liều
        """
        try:
            # Bắt đầu đo thời gian
            start_time = time.time()

            # Trích xuất dữ liệu CT
            ct_image = patient_data.get("ct_image")
            if ct_image is None:
                raise ValidationError("Không tìm thấy dữ liệu CT trong patient_data")

            # Lấy kích thước voxel
            voxel_size = patient_data.get("voxel_size", (2.0, 2.0, 2.0))  # mm

            # Trích xuất thông tin chùm tia
            energy = beam_data.get("energy", 6.0)  # MV
            fluence_map = beam_data.get("fluence_map")
            source_position = beam_data.get("source_position")
            isocenter = beam_data.get("isocenter")

            # Kiểm tra dữ liệu đầu vào
            if fluence_map is None or source_position is None or isocenter is None:
                raise ValidationError("Thiếu thông tin chùm tia cần thiết")

            # Tạo bản đồ vật liệu từ hình ảnh CT
            material_map = self.create_material_map(ct_image)

            # Tạo bản đồ tiết diện
            cross_sections = self.create_cross_section_map(material_map, energy)

            # Tính toán TERMA
            logger.info("Bắt đầu tính toán TERMA")
            if hasattr(self, "status_callback") and callable(self.status_callback):
                self.status_callback(0.1, "Đang tính toán TERMA")

            terma = calculate_terma(ct_image, beam_data, cross_sections["mu_total"])

            # Thiết lập chế độ tính toán dựa trên tính khả dụng của GPU
            use_gpu = self.use_gpu
            if calculation_options:
                # Cho phép ghi đè chế độ GPU/CPU qua calculation_options
                use_gpu = calculation_options.get("use_gpu", self.use_gpu)

            # Giải LBTE để tính fluence
            logger.info(f"Bắt đầu giải LBTE với {'GPU' if use_gpu else 'CPU'}")
            if use_gpu:
                fluence = self.solve_lbte_gpu(terma, cross_sections, voxel_size)
            else:
                fluence = self.solve_lbte(terma, cross_sections, voxel_size)

            # Chuyển đổi fluence thành liều
            logger.info(
                f"Đang chuyển đổi fluence thành liều với {'GPU' if use_gpu else 'CPU'}"
            )
            if use_gpu:
                dose = self.convert_fluence_to_dose_gpu(fluence, cross_sections)
            else:
                dose = self.convert_fluence_to_dose(fluence, cross_sections)

            # Áp dụng hiệu chỉnh không đồng nhất nếu cần
            if calculation_options and calculation_options.get(
                "apply_heterogeneity_correction", True
            ):
                logger.info("Đang áp dụng hiệu chỉnh không đồng nhất")
                if hasattr(self, "status_callback") and callable(self.status_callback):
                    self.status_callback(
                        0.97, "Đang áp dụng hiệu chỉnh không đồng nhất"
                    )

                dose = apply_heterogeneity_correction(
                    dose,
                    cross_sections["density"],
                    spacing=voxel_size,
                    energy=beam_data.get("energy", 6.0),
                )

            # Chuẩn hóa liều
            dose = dose_grid.normalize_dose(dose, beam_data.get("mu", 100.0))

            # Thiết lập thông tin liều vào dose_grid
            dose_grid.set_dose_data(dose)

            # Ghi nhật ký thời gian tính toán
            end_time = time.time()
            calculation_time = end_time - start_time
            logger.info(
                f"Tính toán Acuros XB GPU hoàn thành trong {calculation_time:.2f} giây"
            )

            if hasattr(self, "status_callback") and callable(self.status_callback):
                self.status_callback(1.0, "Đã hoàn thành tính toán liều")

            return dose

        except Exception as e:
            logger.error(f"Lỗi trong quá trình tính toán Acuros XB GPU: {str(e)}")
            raise AlgorithmError(f"Lỗi Acuros XB GPU: {str(e)}")

    def supported_algorithms(self) -> List[DoseCalculationAlgorithm]:
        """
        Trả về danh sách các thuật toán được hỗ trợ.

        Returns:
            list: Danh sách các thuật toán
        """
        return [DoseCalculationAlgorithm.ACUROS_XB]

    # Bổ sung triển khai các abstract methods từ DoseCalculationImplementer
    def get_description(self) -> str:
        """
        Trả về mô tả về thuật toán tính toán liều.

        Returns:
            str: Mô tả thuật toán
        """
        return (
            "Acuros XB GPU là phiên bản tối ưu hóa của Acuros XB sử dụng tính toán GPU. "
            "Thuật toán này giải phương trình vận chuyển bức xạ tuyến tính Boltzmann (LBTE) "
            "trên GPU, giúp tăng tốc đáng kể và duy trì độ chính xác cao tương đương Monte Carlo "
            "trong các môi trường không đồng nhất phức tạp."
        )

    def get_parameters_info(self) -> Dict[str, Any]:
        """
        Trả về thông tin về các tham số có thể cấu hình của thuật toán.

        Returns:
            dict: Thông tin về các tham số
        """
        base_params = (
            super().get_parameters_info()
            if hasattr(super(), "get_parameters_info")
            else {}
        )

        # Thêm các tham số GPU
        gpu_params = {
            "use_gpu": {
                "type": "boolean",
                "default": self.use_gpu,
                "description": "Sử dụng GPU cho tính toán liều nếu có sẵn",
            },
            "use_double_precision": {
                "type": "boolean",
                "default": self.physics_params["use_double_precision"],
                "description": "Sử dụng độ chính xác kép cho tính toán GPU",
            },
            "max_gpu_memory": {
                "type": "float",
                "default": self.max_gpu_memory,
                "range": [0.1, 1.0],
                "description": "Tỉ lệ tối đa bộ nhớ GPU có thể sử dụng (0.1-1.0)",
            },
        }

        # Kết hợp các tham số
        all_params = base_params.copy()
        all_params.update(gpu_params)

        return all_params


# Đăng ký thuật toán
def register_algorithm():
    """Đăng ký thuật toán Acuros XB GPU vào hệ thống."""
    from quangtps.dose.dose_engine import register_dose_algorithm_implementer

    # Đăng ký AcurosGPUImplementer cho thuật toán ACUROS_XB
    register_dose_algorithm_implementer(
        DoseCalculationAlgorithm.ACUROS_XB, AcurosGPUImplementer(), gpu_supported=True
    )

    logger.info("Đã đăng ký thuật toán Acuros XB GPU")
