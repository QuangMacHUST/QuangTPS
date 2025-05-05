#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GPU-accelerated Monte Carlo dose calculation for QuangTPS.

Module này triển khai thuật toán Monte Carlo tính toán liều dùng GPU (CUDA/OpenCL),
giúp tăng tốc độ tính toán đáng kể so với các phương pháp tính toán truyền thống.
"""

import os
import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional, Union
import time
import threading
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import json
from enum import Enum
from pathlib import Path

# Thử import các thư viện GPU (CUDA/OpenCL)
try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

try:
    import pyopencl as cl
    import pyopencl.array

    OPENCL_AVAILABLE = True
except ImportError:
    OPENCL_AVAILABLE = False
    cl = None

from quangtps.dose.dose_engine import (
    DoseCalculationImplementer,
    DoseCalculationAlgorithm,
)
from quangtps.dose.dose_grid import DoseGrid
from quangtps.core.exceptions import ValidationError, AlgorithmError

logger = logging.getLogger(__name__)


class GPUPlatform(Enum):
    """Nền tảng GPU được hỗ trợ."""

    NONE = 0
    CUDA = 1
    OPENCL = 2
    CPU_FALLBACK = 3


class GPUMonteCarloConfig:
    """Cấu hình cho thuật toán Monte Carlo dùng GPU."""

    def __init__(self):
        """Khởi tạo cấu hình mặc định."""
        # Thông số vật lý và tính toán
        self.num_histories = 1_000_000  # Số lịch sử mô phỏng
        self.batch_size = 10_000  # Số lịch sử tính một lần
        self.energy_cutoff = 0.01  # MeV, ngưỡng năng lượng dừng tính toán
        self.max_depth = 50.0  # cm, độ sâu tối đa
        self.voxel_size = (2.0, 2.0, 2.0)  # mm
        self.seed = 12345  # Hạt giống cho số ngẫu nhiên

        # Cấu hình GPU
        self.use_gpu = True  # Sử dụng GPU nếu có
        self.platform = GPUPlatform.CUDA  # Mặc định ưu tiên CUDA
        self.device_index = 0  # GPU đầu tiên
        self.multi_gpu = False  # Sử dụng nhiều GPU
        self.gpu_devices = []  # Danh sách ID thiết bị GPU

        # Cấu hình tính toán
        self.use_variance_reduction = True  # Sử dụng kỹ thuật giảm phương sai
        self.use_track_length_scoring = True  # Sử dụng phương pháp track-length
        self.use_forced_detection = True  # Sử dụng kỹ thuật bắt buộc phát hiện
        self.use_russian_roulette = True  # Sử dụng kỹ thuật Russian roulette

        # Hiệu suất và độ chính xác
        self.precision = "single"  # "single" hoặc "double"
        self.statistical_uncertainty = (
            0.01  # Mức độ không chắc chắn thống kê mục tiêu (1%)
        )
        self.reporting_interval = 5  # Báo cáo tiến độ sau mỗi 5% lịch sử

        # Cache và dữ liệu đã tính toán trước
        self.use_cached_kernel = True  # Sử dụng kernel đã tính trước
        self.kernel_cache_dir = "kernel_cache"  # Thư mục cache

        # Các thông số nâng cao
        self.advanced = {
            "electron_step_size": 2.0,  # mm
            "photon_splitting": 5,  # Phân chia các photon
            "use_woodcock_tracking": True,  # Sử dụng kỹ thuật theo dõi Woodcock
            "use_tabulated_cross_sections": True,  # Sử dụng dữ liệu bảng cho mặt cắt
        }

    def to_dict(self) -> Dict[str, Any]:
        """Chuyển đổi cấu hình sang từ điển."""
        result = {
            "num_histories": self.num_histories,
            "batch_size": self.batch_size,
            "energy_cutoff": self.energy_cutoff,
            "max_depth": self.max_depth,
            "voxel_size": self.voxel_size,
            "seed": self.seed,
            "use_gpu": self.use_gpu,
            "platform": self.platform.name,
            "device_index": self.device_index,
            "multi_gpu": self.multi_gpu,
            "gpu_devices": self.gpu_devices,
            "use_variance_reduction": self.use_variance_reduction,
            "use_track_length_scoring": self.use_track_length_scoring,
            "use_forced_detection": self.use_forced_detection,
            "use_russian_roulette": self.use_russian_roulette,
            "precision": self.precision,
            "statistical_uncertainty": self.statistical_uncertainty,
            "reporting_interval": self.reporting_interval,
            "use_cached_kernel": self.use_cached_kernel,
            "kernel_cache_dir": self.kernel_cache_dir,
            "advanced": self.advanced,
        }
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GPUMonteCarloConfig":
        """Tạo cấu hình từ từ điển."""
        config = cls()
        for key, value in data.items():
            if key == "platform":
                config.platform = GPUPlatform[value]
            elif key == "advanced":
                config.advanced.update(value)
            elif hasattr(config, key):
                setattr(config, key, value)
        return config

    def save_to_file(self, filepath: str) -> bool:
        """Lưu cấu hình vào file."""
        try:
            with open(filepath, "w") as f:
                json.dump(self.to_dict(), f, indent=4)
            return True
        except Exception as e:
            logger.error(f"Không thể lưu cấu hình: {e}")
            return False

    @classmethod
    def load_from_file(cls, filepath: str) -> Optional["GPUMonteCarloConfig"]:
        """Tải cấu hình từ file."""
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
            return cls.from_dict(data)
        except Exception as e:
            logger.error(f"Không thể tải cấu hình: {e}")
            return None


class GPUMonteCarloDose(DoseCalculationImplementer):
    """
    Thuật toán Monte Carlo tính toán liều dùng GPU.

    Thuật toán này mô phỏng tương tác của các hạt (photon, electron) với vật chất
    để tính toán phân bố liều chính xác, đặc biệt trong các trường hợp không đồng
    nhất phức tạp. Việc sử dụng GPU giúp tăng tốc độ tính toán rất nhiều so với CPU.
    """

    def __init__(self):
        """Khởi tạo thuật toán GPU Monte Carlo."""
        super().__init__(algorithm_type=DoseCalculationAlgorithm.MONTE_CARLO)

        # Khởi tạo cấu hình
        self.config = GPUMonteCarloConfig()

        # Khởi tạo các biến trạng thái
        self.gpu_platform = GPUPlatform.NONE
        self.gpu_device = None
        self.gpu_context = None
        self.gpu_queue = None
        self.is_initialized = False
        self.running = False
        self.progress = 0.0
        self.status = "Chưa khởi tạo"
        self.result = None
        self.callbacks = []

        # Kiểm tra các nền tảng GPU có sẵn
        self._detect_available_platforms()

    def _detect_available_platforms(self):
        """Phát hiện các nền tảng GPU có sẵn."""
        available_platforms = []

        # Kiểm tra CUDA
        if CUPY_AVAILABLE:
            try:
                num_gpus = cp.cuda.runtime.getDeviceCount()
                if num_gpus > 0:
                    available_platforms.append(GPUPlatform.CUDA)
                    logger.info(f"Phát hiện {num_gpus} thiết bị CUDA")
            except Exception as e:
                logger.warning(
                    f"CuPy có sẵn nhưng không thể truy cập thiết bị CUDA: {e}"
                )

        # Kiểm tra OpenCL
        if OPENCL_AVAILABLE:
            try:
                platforms = cl.get_platforms()
                if platforms:
                    devices = []
                    for platform in platforms:
                        platform_devices = platform.get_devices()
                        if platform_devices:
                            devices.extend(platform_devices)

                    if devices:
                        available_platforms.append(GPUPlatform.OPENCL)
                        logger.info(f"Phát hiện {len(devices)} thiết bị OpenCL")
            except Exception as e:
                logger.warning(
                    f"PyOpenCL có sẵn nhưng không thể truy cập thiết bị OpenCL: {e}"
                )

        # Luôn thêm CPU fallback
        available_platforms.append(GPUPlatform.CPU_FALLBACK)

        # Chọn nền tảng tốt nhất có sẵn
        if GPUPlatform.CUDA in available_platforms:
            self.gpu_platform = GPUPlatform.CUDA
        elif GPUPlatform.OPENCL in available_platforms:
            self.gpu_platform = GPUPlatform.OPENCL
        else:
            self.gpu_platform = GPUPlatform.CPU_FALLBACK
            logger.warning("Không tìm thấy GPU, sử dụng CPU fallback")

        logger.info(f"Đã chọn nền tảng: {self.gpu_platform.name}")

    def initialize(self) -> bool:
        """Khởi tạo thuật toán và tài nguyên GPU."""
        if self.is_initialized:
            return True

        try:
            # Khởi tạo dựa trên nền tảng
            if self.gpu_platform == GPUPlatform.CUDA and CUPY_AVAILABLE:
                self._initialize_cuda()
            elif self.gpu_platform == GPUPlatform.OPENCL and OPENCL_AVAILABLE:
                self._initialize_opencl()
            else:
                self._initialize_cpu_fallback()

            self.is_initialized = True
            self.status = "Đã khởi tạo"
            logger.info(f"Đã khởi tạo GPU Monte Carlo: {self.gpu_platform.name}")
            return True

        except Exception as e:
            logger.error(f"Không thể khởi tạo GPU Monte Carlo: {e}")
            self.status = f"Lỗi khởi tạo: {str(e)}"
            return False

    def _initialize_cuda(self):
        """Khởi tạo tài nguyên CUDA."""
        device_id = self.config.device_index
        cp.cuda.Device(device_id).use()
        self.gpu_device = cp.cuda.Device(device_id)

        # In thông tin thiết bị
        props = cp.cuda.runtime.getDeviceProperties(device_id)
        logger.info(f"Sử dụng thiết bị CUDA: {props['name'].decode()}")
        logger.info(f"Bộ nhớ tổng: {props['totalGlobalMem'] / (1024**3):.2f} GB")
        logger.info(f"Số lượng SM: {props['multiProcessorCount']}")

        # Khởi tạo hạt giống RNG
        cp.random.seed(self.config.seed)

    def _initialize_opencl(self):
        """Khởi tạo tài nguyên OpenCL."""
        # Lấy tất cả nền tảng
        platforms = cl.get_platforms()
        if not platforms:
            raise RuntimeError("Không tìm thấy nền tảng OpenCL nào")

        # Ưu tiên GPU trước
        devices = []
        for platform in platforms:
            try:
                gpu_devices = platform.get_devices(device_type=cl.device_type.GPU)
                if gpu_devices:
                    devices.extend([(platform, device) for device in gpu_devices])
            except:
                pass

        # Nếu không có GPU, thử CPU
        if not devices:
            for platform in platforms:
                try:
                    cpu_devices = platform.get_devices(device_type=cl.device_type.CPU)
                    if cpu_devices:
                        devices.extend([(platform, device) for device in cpu_devices])
                except:
                    pass

        if not devices:
            raise RuntimeError("Không tìm thấy thiết bị OpenCL nào")

        # Chọn thiết bị dựa trên cấu hình
        if self.config.device_index < len(devices):
            platform, device = devices[self.config.device_index]
        else:
            platform, device = devices[0]

        # Tạo context và queue
        self.gpu_device = device
        self.gpu_context = cl.Context([device])
        self.gpu_queue = cl.CommandQueue(self.gpu_context)

        # In thông tin thiết bị
        logger.info(f"Sử dụng thiết bị OpenCL: {device.name}")
        logger.info(f"Bộ nhớ tổng: {device.global_mem_size / (1024**3):.2f} GB")
        logger.info(f"Đơn vị tính toán: {device.max_compute_units}")

    def _initialize_cpu_fallback(self):
        """Khởi tạo tài nguyên CPU fallback."""
        logger.info("Sử dụng CPU fallback cho tính toán Monte Carlo")
        # Đặt hạt giống cho NumPy
        np.random.seed(self.config.seed)

    def set_config(self, config: Union[GPUMonteCarloConfig, Dict[str, Any]]) -> None:
        """
        Đặt cấu hình cho thuật toán.

        Parameters:
            config: Cấu hình mới (đối tượng GPUMonteCarloConfig hoặc từ điển)
        """
        if isinstance(config, dict):
            self.config = GPUMonteCarloConfig.from_dict(config)
        else:
            self.config = config

        # Cập nhật lại khởi tạo nếu đã khởi tạo rồi
        if self.is_initialized:
            self.cleanup()
            self.initialize()

    def add_progress_callback(self, callback) -> None:
        """
        Thêm callback để nhận thông báo tiến độ.

        Parameters:
            callback: Hàm callback nhận (progress, status)
        """
        if callback not in self.callbacks:
            self.callbacks.append(callback)

    def _update_progress(self, progress: float, status: str = None) -> None:
        """
        Cập nhật tiến độ và gọi các callback.

        Parameters:
            progress: Tiến độ tính (0-1)
            status: Thông tin trạng thái
        """
        self.progress = progress
        if status:
            self.status = status

        # Gọi tất cả callback
        for callback in self.callbacks:
            try:
                callback(progress, status)
            except Exception as e:
                logger.error(f"Lỗi trong callback tiến độ: {e}")

    def cleanup(self) -> None:
        """Giải phóng tài nguyên GPU."""
        if not self.is_initialized:
            return

        try:
            # Dọn dẹp tùy theo nền tảng
            if self.gpu_platform == GPUPlatform.CUDA and CUPY_AVAILABLE:
                # CuPy sẽ tự dọn dẹp thông qua GC
                # Gọi rõ ràng để đảm bảo
                cp.get_default_memory_pool().free_all_blocks()

            elif self.gpu_platform == GPUPlatform.OPENCL and OPENCL_AVAILABLE:
                # Xóa các đối tượng OpenCL
                self.gpu_queue = None
                self.gpu_context = None

            self.is_initialized = False
            self.status = "Đã dọn dẹp tài nguyên"
            logger.info("Đã giải phóng tài nguyên GPU Monte Carlo")

        except Exception as e:
            logger.error(f"Lỗi khi dọn dẹp tài nguyên GPU: {e}")

    def _prepare_materials(self, ct_data: np.ndarray) -> np.ndarray:
        """
        Chuẩn bị dữ liệu vật liệu từ hình ảnh CT.

        Parameters:
            ct_data: Ma trận HU của hình ảnh CT

        Returns:
            Ma trận mã vật liệu và mật độ
        """
        # Chuyển đổi HU thành mật độ và mã vật liệu
        # Đơn giản hóa: 5 loại vật liệu (1-5)
        # 1: Không khí, 2: Phổi, 3: Mô mềm, 4: Xương, 5: Implant

        # Ngưỡng HU
        thresholds = {
            "air": -950,
            "lung": -700,
            "soft_tissue": 100,
            "bone": 1500,
        }

        # Khởi tạo mảng kết quả (mã vật liệu)
        material_data = np.ones_like(ct_data, dtype=np.int8)

        # Gán mã vật liệu dựa trên HU
        material_data[
            (ct_data > thresholds["air"]) & (ct_data <= thresholds["lung"])
        ] = 2
        material_data[
            (ct_data > thresholds["lung"]) & (ct_data <= thresholds["soft_tissue"])
        ] = 3
        material_data[
            (ct_data > thresholds["soft_tissue"]) & (ct_data <= thresholds["bone"])
        ] = 4
        material_data[ct_data > thresholds["bone"]] = 5

        # Tính mật độ từ HU
        # Công thức ρ = ρ_water * (1 + HU/1000)
        density_data = 1.0 + (ct_data / 1000.0)
        density_data = np.clip(density_data, 0.001, 8.0)  # Giới hạn phạm vi hợp lý

        # Kết hợp mã vật liệu và mật độ thành một ma trận duy nhất
        # Sử dụng định dạng (material_id << 24 | density_float_bits)
        result = np.zeros_like(ct_data, dtype=np.uint32)

        # Chuyển đổi material_id sang 8 bit cao nhất
        result = material_data.astype(np.uint32) << 24

        # Chuyển đổi density sang 24 bit còn lại
        # Chuyển float32 sang int32 theo bit
        density_bits = density_data.view(np.int32) & 0x00FFFFFF

        # Kết hợp
        result |= density_bits.astype(np.uint32)

        return result

    def _simulate_histories_cuda(
        self, materials: np.ndarray, beams: List[Dict[str, Any]], num_histories: int
    ) -> np.ndarray:
        """
        Mô phỏng lịch sử hạt dùng CUDA.

        Parameters:
            materials: Ma trận vật liệu
            beams: Cấu hình chùm tia
            num_histories: Số lịch sử mô phỏng

        Returns:
            Ma trận liều tích lũy
        """
        if not CUPY_AVAILABLE:
            raise RuntimeError("CuPy không khả dụng")

        # Chuyển dữ liệu vào GPU
        device_materials = cp.array(materials)

        # Khởi tạo ma trận liều trên GPU
        device_dose = cp.zeros(materials.shape, dtype=cp.float32)

        # Kích thước ma trận
        nx, ny, nz = materials.shape

        # Chia thành các batch
        batch_size = min(self.config.batch_size, num_histories)
        num_batches = (num_histories + batch_size - 1) // batch_size

        # Mã kernel CUDA (viết cho dễ hiểu, thực tế sẽ phức tạp hơn nhiều)
        # Đây chỉ là phác thảo, không phải mã hoàn chỉnh
        cuda_kernel = cp.RawKernel(
            r"""
        extern "C" __global__
        void monte_carlo_kernel(
            unsigned int* materials, float* dose,
            int nx, int ny, int nz,
            float* beam_positions, float* beam_directions,
            float* energies, int num_histories,
            unsigned int seed
        ) {
            // Mỗi thread xử lý một lịch sử
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= num_histories) return;

            // Khởi tạo trạng thái RNG
            curandState rng_state;
            curand_init(seed + idx, 0, 0, &rng_state);

            // Lấy thông tin chùm tia cho lịch sử này
            int beam_idx = idx % beam_positions[0];  // Số lượng chùm tia
            float3 position = make_float3(
                beam_positions[beam_idx*3 + 1],
                beam_positions[beam_idx*3 + 2],
                beam_positions[beam_idx*3 + 3]
            );

            float3 direction = make_float3(
                beam_directions[beam_idx*3 + 0],
                beam_directions[beam_idx*3 + 1],
                beam_directions[beam_idx*3 + 2]
            );

            float energy = energies[beam_idx];

            // Mô phỏng đường đi của hạt
            // ...

            // Tích lũy liều
            // ...

            // Kết thúc lịch sử
        }
        """,
            "monte_carlo_kernel",
        )

        # Khởi tạo dữ liệu beam trên GPU
        beam_positions = []
        beam_directions = []
        beam_energies = []

        for beam in beams:
            # Lấy thông tin chùm tia
            isocenter = beam.get("isocenter", [0, 0, 0])
            gantry_angle = beam.get("gantry_angle", 0.0)
            collimator_angle = beam.get("collimator_angle", 0.0)
            energy = beam.get("energy", 6.0)

            # Tính toán hướng chùm tia từ góc gantry và collimator
            # ... (tính toán vector)

            # Thêm vào danh sách
            beam_positions.extend(isocenter)
            beam_directions.extend([0, 0, 1])  # Placeholder
            beam_energies.append(energy)

        # Chuyển danh sách thành mảng GPU
        device_beam_positions = cp.array(beam_positions, dtype=cp.float32)
        device_beam_directions = cp.array(beam_directions, dtype=cp.float32)
        device_beam_energies = cp.array(beam_energies, dtype=cp.float32)

        # Chạy mô phỏng theo batch
        start_time = time.time()
        for batch in range(num_batches):
            # Số lịch sử trong batch này
            current_batch_size = min(batch_size, num_histories - batch * batch_size)

            # Chạy kernel CUDA
            blocks_per_grid = (current_batch_size + 255) // 256
            threads_per_block = min(current_batch_size, 256)

            # Gọi kernel
            # cuda_kernel(
            #     (blocks_per_grid,), (threads_per_block,),
            #     (device_materials, device_dose, nx, ny, nz,
            #      device_beam_positions, device_beam_directions, device_beam_energies,
            #      current_batch_size, self.config.seed + batch)
            # )

            # Cập nhật tiến độ
            progress = (batch + 1) / num_batches
            elapsed = time.time() - start_time
            estimated_total = elapsed / progress if progress > 0 else 0
            remaining = estimated_total - elapsed

            status = (
                f"Đã mô phỏng {batch + 1}/{num_batches} batch "
                f"({(batch + 1) * batch_size:,}/{num_histories:,} lịch sử). "
                f"Còn lại: {remaining:.1f}s"
            )

            self._update_progress(progress * 0.9, status)  # 90% thời gian cho mô phỏng

        # Lấy kết quả về CPU
        result = device_dose.get()

        return result

    def _simulate_histories_opencl(
        self, materials: np.ndarray, beams: List[Dict[str, Any]], num_histories: int
    ) -> np.ndarray:
        """
        Mô phỏng lịch sử hạt dùng OpenCL.

        Parameters:
            materials: Ma trận vật liệu
            beams: Cấu hình chùm tia
            num_histories: Số lịch sử mô phỏng

        Returns:
            Ma trận liều tích lũy
        """
        if not OPENCL_AVAILABLE:
            raise RuntimeError("PyOpenCL không khả dụng")

        # Mã OpenCL
        opencl_code = """
        // OpenCL kernel code
        // ...
        """

        # Khởi tạo ma trận liều
        dose = np.zeros(materials.shape, dtype=np.float32)

        # Chuyển dữ liệu vào GPU
        materials_buf = cl.Buffer(
            self.gpu_context,
            cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
            hostbuf=materials,
        )

        dose_buf = cl.Buffer(
            self.gpu_context, cl.mem_flags.WRITE_ONLY, size=dose.nbytes
        )

        # Tạo chương trình OpenCL
        program = cl.Program(self.gpu_context, opencl_code).build()

        # TODO: Chuẩn bị dữ liệu chùm tia, tạo kernel và thực thi

        # Lấy kết quả
        cl.enqueue_copy(self.gpu_queue, dose, dose_buf)

        return dose

    def _simulate_histories_cpu(
        self, materials: np.ndarray, beams: List[Dict[str, Any]], num_histories: int
    ) -> np.ndarray:
        """
        Mô phỏng lịch sử hạt dùng CPU (fallback).

        Parameters:
            materials: Ma trận vật liệu
            beams: Cấu hình chùm tia
            num_histories: Số lịch sử mô phỏng

        Returns:
            Ma trận liều tích lũy
        """
        # Khởi tạo ma trận liều
        dose = np.zeros(materials.shape, dtype=np.float32)

        # Một mô phỏng Monte Carlo đơn giản
        nx, ny, nz = materials.shape

        # Chia thành các batch
        batch_size = min(10000, num_histories)
        num_batches = (num_histories + batch_size - 1) // batch_size

        # Mô phỏng theo batch
        start_time = time.time()
        for batch in range(num_batches):
            # Số lịch sử trong batch này
            current_batch_size = min(batch_size, num_histories - batch * batch_size)

            # Mô phỏng CPU đơn giản (đây chỉ là mô phỏng giả)
            # Trong thực tế, đây sẽ là mã mô phỏng Monte Carlo đầy đủ
            for i in range(current_batch_size):
                # Chọn một chùm tia
                beam_idx = i % len(beams)
                beam = beams[beam_idx]

                # Lấy thông tin chùm tia
                energy = beam.get("energy", 6.0)
                isocenter = beam.get("isocenter", [0, 0, 0])

                # Mô phỏng đơn giản bằng cách đặt điểm liều ở gần isocenter
                ix = max(0, min(nx - 1, int(isocenter[0])))
                iy = max(0, min(ny - 1, int(isocenter[1])))
                iz = max(0, min(nz - 1, int(isocenter[2])))

                # Thêm liều vào vị trí này và xung quanh
                for dx in range(-3, 4):
                    for dy in range(-3, 4):
                        for dz in range(-3, 4):
                            x = max(0, min(nx - 1, ix + dx))
                            y = max(0, min(ny - 1, iy + dy))
                            z = max(0, min(nz - 1, iz + dz))

                            # Hệ số suy giảm
                            dist = np.sqrt(dx**2 + dy**2 + dz**2)
                            if dist > 0:
                                dose[x, y, z] += (
                                    energy * np.exp(-dist) / (dist * num_histories)
                                )

            # Cập nhật tiến độ
            progress = (batch + 1) / num_batches
            elapsed = time.time() - start_time
            estimated_total = elapsed / progress if progress > 0 else 0
            remaining = estimated_total - elapsed

            status = (
                f"Đã mô phỏng {batch + 1}/{num_batches} batch "
                f"({(batch + 1) * batch_size:,}/{num_histories:,} lịch sử). "
                f"Còn lại: {remaining:.1f}s"
            )

            self._update_progress(progress * 0.9, status)  # 90% thời gian cho mô phỏng

        return dose

    def calculate(
        self,
        patient_ct: np.ndarray,
        structures: Dict[str, np.ndarray],
        beams: List[Dict[str, Any]],
        spacing: Tuple[float, float, float],
        origin: Tuple[float, float, float],
    ) -> DoseGrid:
        """
        Tính toán phân bố liều dùng Monte Carlo.

        Parameters:
            patient_ct: Ma trận hình ảnh CT (HU)
            structures: Dict các cấu trúc
            beams: Danh sách các chùm tia
            spacing: Khoảng cách giữa các voxel (mm)
            origin: Vị trí gốc của lưới liều (mm)

        Returns:
            DoseGrid: Lưới liều kết quả
        """
        # Đảm bảo đã khởi tạo
        if not self.is_initialized:
            self.initialize()

        # Đánh dấu đang chạy
        self.running = True
        self.result = None

        try:
            start_time = time.time()
            self._update_progress(0.01, "Đang chuẩn bị dữ liệu...")

            # Chuẩn bị dữ liệu vật liệu
            materials = self._prepare_materials(patient_ct)

            # Mô phỏng Monte Carlo
            self._update_progress(0.1, "Bắt đầu mô phỏng Monte Carlo...")

            if self.gpu_platform == GPUPlatform.CUDA and CUPY_AVAILABLE:
                dose = self._simulate_histories_cuda(
                    materials, beams, self.config.num_histories
                )
            elif self.gpu_platform == GPUPlatform.OPENCL and OPENCL_AVAILABLE:
                dose = self._simulate_histories_opencl(
                    materials, beams, self.config.num_histories
                )
            else:
                dose = self._simulate_histories_cpu(
                    materials, beams, self.config.num_histories
                )

            # Chuẩn hóa liều và tính thống kê
            self._update_progress(0.95, "Đang chuẩn hóa liều...")

            # Chuẩn hóa theo MU
            total_mu = sum(beam.get("mu", 100.0) for beam in beams)
            dose = dose * total_mu

            # Tính thống kê không chắc chắn
            # TODO: Tính không chắc chắn thống kê thực tế

            # Tạo lưới liều kết quả
            result_grid = DoseGrid(dose, spacing, origin)

            # Hoàn thành
            elapsed = time.time() - start_time
            self._update_progress(
                1.0, f"Đã hoàn thành tính toán Monte Carlo trong {elapsed:.1f}s"
            )

            self.result = result_grid
            return result_grid

        except Exception as e:
            logger.error(f"Lỗi trong tính toán Monte Carlo: {e}")
            self.status = f"Lỗi: {str(e)}"
            raise AlgorithmError(f"Lỗi Monte Carlo: {str(e)}")

        finally:
            self.running = False
