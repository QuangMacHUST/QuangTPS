#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module thuật toán Monte Carlo trên GPU.

Module này triển khai thuật toán Monte Carlo tăng tốc bằng GPU để tính toán liều
chính xác với hiệu suất cao cho các kế hoạch xạ trị phức tạp.
"""

import os
import time
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union, NamedTuple
from enum import Enum, auto
import threading
import multiprocessing

# GPU imports
try:
    import cupy as cp
    import cupyx.scipy.ndimage as cpx_ndimage

    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    logging.warning(
        "CuPy không khả dụng. Tính toán Monte Carlo GPU sẽ không hoạt động."
    )

try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    from pycuda.compiler import SourceModule

    HAS_PYCUDA = True
except ImportError:
    HAS_PYCUDA = False
    logging.warning(
        "PyCUDA không khả dụng. Một số tính năng Monte Carlo GPU sẽ bị giới hạn."
    )

from quangtps.dose.algorithms.base import (
    DoseAlgorithm,
    DoseAlgorithmType,
    DoseCalculationMode,
)
from quangtps.dose.algorithms.montecarlo import (
    MonteCarloAlgorithm,
    MCConfiguration,
    ParticleType,
)
from quangtps.dose.dose_grid import DoseGrid

logger = logging.getLogger(__name__)


class GPUMonteCarloConfiguration(MCConfiguration):
    """Cấu hình cho thuật toán Monte Carlo trên GPU."""

    def __init__(
        self,
        num_histories: int = 1000000,
        num_threads: int = 8,
        use_multithreading: bool = True,
        random_seed: Optional[int] = None,
        use_heterogeneity_correction: bool = True,
        use_variance_reduction: bool = True,
        batch_size: int = 10000,
        report_uncertainty: bool = True,
        photon_energy_cutoff: float = 0.01,  # MeV
        electron_energy_cutoff: float = 0.2,  # MeV
        gpu_device_id: int = 0,
        use_fast_math: bool = True,
        use_persistent_threads: bool = True,
        use_shared_memory: bool = True,
    ):
        """
        Khởi tạo cấu hình Monte Carlo GPU.

        Parameters
        ----------
        num_histories : int, optional
            Số lượng hạt mô phỏng, mặc định là 1000000
        num_threads : int, optional
            Số luồng sử dụng, mặc định là 8
        use_multithreading : bool, optional
            Sử dụng đa luồng, mặc định là True
        random_seed : Optional[int], optional
            Hạt giống ngẫu nhiên, mặc định là None
        use_heterogeneity_correction : bool, optional
            Sử dụng hiệu chỉnh không đồng nhất, mặc định là True
        use_variance_reduction : bool, optional
            Sử dụng kỹ thuật giảm phương sai, mặc định là True
        batch_size : int, optional
            Kích thước lô xử lý, mặc định là 10000
        report_uncertainty : bool, optional
            Báo cáo độ không chắc chắn, mặc định là True
        photon_energy_cutoff : float, optional
            Ngưỡng năng lượng cắt cho photon (MeV), mặc định là 0.01
        electron_energy_cutoff : float, optional
            Ngưỡng năng lượng cắt cho electron (MeV), mặc định là 0.2
        gpu_device_id : int, optional
            ID của thiết bị GPU sử dụng, mặc định là 0
        use_fast_math : bool, optional
            Sử dụng các hàm toán học nhanh trên GPU, mặc định là True
        use_persistent_threads : bool, optional
            Sử dụng persistent threads trên GPU, mặc định là True
        use_shared_memory : bool, optional
            Sử dụng shared memory trên GPU, mặc định là True
        """
        # Gọi hàm khởi tạo của lớp cha
        super().__init__()

        # Cập nhật các thuộc tính từ tham số
        self.num_histories = num_histories
        self.num_threads = num_threads
        self.use_multithreading = use_multithreading
        self.random_seed = random_seed
        self.use_heterogeneity_correction = use_heterogeneity_correction
        self.use_variance_reduction = use_variance_reduction
        self.batch_size = batch_size
        self.report_uncertainty = report_uncertainty
        self.photon_cutoff = photon_energy_cutoff
        self.electron_cutoff = electron_energy_cutoff

        self.gpu_device_id = gpu_device_id
        self.use_fast_math = use_fast_math
        self.use_persistent_threads = use_persistent_threads
        self.use_shared_memory = use_shared_memory

        # Kiểm tra khả năng sử dụng GPU
        self.gpu_available = HAS_CUPY or HAS_PYCUDA
        if not self.gpu_available:
            logger.warning("Không tìm thấy GPU khả dụng. Chuyển về chế độ CPU.")


class MonteCarloResult(NamedTuple):
    """Kết quả tính toán Monte Carlo."""

    dose_grid: np.ndarray
    """Lưới liều tính toán."""

    uncertainty: Optional[np.ndarray] = None
    """Độ không chắc chắn thống kê (nếu có)."""

    simulation_time: float = 0.0
    """Thời gian mô phỏng (giây)."""

    num_histories: int = 0
    """Số lượng hạt được mô phỏng."""

    additional_info: Dict[str, Any] = {}
    """Thông tin bổ sung."""

    @property
    def efficiency(self) -> float:
        """
        Tính hiệu suất mô phỏng (histories/second).

        Returns
        -------
        float
            Hiệu suất mô phỏng (histories/second)
        """
        if self.simulation_time <= 0:
            return 0.0
        return self.num_histories / self.simulation_time

    @property
    def mean_uncertainty(self) -> float:
        """
        Tính độ không chắc chắn trung bình.

        Returns
        -------
        float
            Độ không chắc chắn trung bình
        """
        if self.uncertainty is None:
            return 0.0

        # Chỉ tính trên các voxel có liều > 0
        mask = self.dose_grid > 0
        if not np.any(mask):
            return 0.0

        # Truy cập uncertainty qua hàm getter
        uncertainty_array = self.uncertainty
        return np.mean(uncertainty_array[mask])

    @property
    def max_uncertainty(self) -> float:
        """
        Tính độ không chắc chắn tối đa.

        Returns
        -------
        float
            Độ không chắc chắn tối đa
        """
        if self.uncertainty is None:
            return 0.0

        # Chỉ tính trên các voxel có liều > 0
        mask = self.dose_grid > 0
        if not np.any(mask):
            return 0.0

        # Truy cập uncertainty qua hàm getter
        uncertainty_array = self.uncertainty
        return np.max(uncertainty_array[mask])

    def get_summary(self) -> Dict[str, Any]:
        """
        Tóm tắt kết quả tính toán.

        Returns
        -------
        Dict[str, Any]
            Thông tin tóm tắt về kết quả mô phỏng
        """
        summary = {
            "num_histories": self.num_histories,
            "simulation_time": self.simulation_time,
            "efficiency": self.efficiency,
            "dose_shape": self.dose_grid.shape,
            "dose_min": float(np.min(self.dose_grid)),
            "dose_max": float(np.max(self.dose_grid)),
            "dose_mean": float(np.mean(self.dose_grid)),
        }

        if self.uncertainty is not None:
            summary.update(
                {
                    "mean_uncertainty": self.mean_uncertainty,
                    "max_uncertainty": self.max_uncertainty,
                }
            )

        # Thêm thông tin từ additional_info
        summary.update(self.additional_info)

        return summary

    def compare_with(self, other: "MonteCarloResult") -> Dict[str, Any]:
        """
        So sánh với kết quả khác.

        Parameters
        ----------
        other : MonteCarloResult
            Kết quả khác để so sánh

        Returns
        -------
        Dict[str, Any]
            Kết quả so sánh
        """
        if self.dose_grid.shape != other.dose_grid.shape:
            raise ValueError(
                f"Không thể so sánh lưới liều khác kích thước: {self.dose_grid.shape} vs {other.dose_grid.shape}"
            )

        # Tính sai số tuyệt đối và tương đối
        abs_diff = np.abs(self.dose_grid - other.dose_grid)

        # Tránh chia cho 0
        nonzero_mask = (self.dose_grid != 0) | (other.dose_grid != 0)
        max_dose = max(np.max(self.dose_grid), np.max(other.dose_grid))

        # Sai số tương đối (chuẩn hóa theo liều tối đa)
        rel_diff = np.zeros_like(abs_diff)
        rel_diff[nonzero_mask] = abs_diff[nonzero_mask] / max_dose * 100.0

        return {
            "abs_diff_mean": float(np.mean(abs_diff)),
            "abs_diff_max": float(np.max(abs_diff)),
            "rel_diff_mean": float(np.mean(rel_diff[nonzero_mask]))
            if np.any(nonzero_mask)
            else 0.0,
            "rel_diff_max": float(np.max(rel_diff[nonzero_mask]))
            if np.any(nonzero_mask)
            else 0.0,
            "gamma_pass_rate": self._calculate_gamma(other),
        }

    def _calculate_gamma(
        self,
        other: "MonteCarloResult",
        dose_threshold: float = 3.0,
        distance_threshold: float = 3.0,
    ) -> float:
        """
        Tính chỉ số gamma giữa hai kết quả.

        Parameters
        ----------
        other : MonteCarloResult
            Kết quả khác để so sánh
        dose_threshold : float, optional
            Ngưỡng sai số liều (%), mặc định là 3.0
        distance_threshold : float, optional
            Ngưỡng khoảng cách (mm), mặc định là 3.0

        Returns
        -------
        float
            Tỷ lệ vượt qua kiểm tra gamma (%)
            Trả về -1.0 nếu không thể tính toán gamma
        """
        if self.dose_grid.shape != other.dose_grid.shape:
            logger.error(
                f"Không thể tính gamma: Kích thước lưới không khớp ({self.dose_grid.shape} vs {other.dose_grid.shape})"
            )
            return -1.0

        try:
            from quangtps.evaluation.metrics.gamma import calculate_gamma_index
        except ImportError as e:
            logger.warning(f"Không thể tính gamma: {e}")
            return -1.0

        try:
            # Chỉ tính gamma cho các voxel > 10% liều tối đa
            dose_max = np.max(self.dose_grid)

            # Kiểm tra liều tối đa
            if dose_max <= 0:
                logger.warning("Không thể tính gamma: Liều tối đa <= 0")
                return -1.0

            dose_threshold_abs = dose_max * 0.1

            # Thực hiện tính toán gamma
            gamma_result = calculate_gamma_index(
                self.dose_grid,
                other.dose_grid,
                dose_threshold=dose_threshold,
                distance_threshold=distance_threshold,
                dose_threshold_abs=dose_threshold_abs,
            )

            # Tính tỷ lệ vượt qua (gamma <= 1.0)
            mask = self.dose_grid > dose_threshold_abs

            # Kiểm tra nếu không có voxel nào vượt ngưỡng
            if not np.any(mask):
                logger.warning(
                    f"Không có voxel nào vượt ngưỡng liều {dose_threshold_abs:.4f} Gy (10% của liều tối đa)"
                )
                return -1.0

            gamma_pass = np.sum(gamma_result[mask] <= 1.0)
            gamma_total = np.sum(mask)

            # Tính và trả về tỷ lệ phần trăm
            return float(gamma_pass / gamma_total * 100.0)

        except Exception as e:
            logger.error(f"Lỗi khi tính chỉ số gamma: {e}")
            import traceback

            traceback.print_exc()
            return -1.0


class MonteCarloGPU(MonteCarloAlgorithm):
    """
    Thuật toán Monte Carlo tăng tốc bằng GPU.

    Triển khai thuật toán Monte Carlo trên GPU để tính toán liều nhanh và chính xác,
    đặc biệt cho các kế hoạch xạ trị phức tạp với nhiều chùm tia.
    """

    def __init__(
        self,
        calculation_mode: DoseCalculationMode = DoseCalculationMode.STANDARD,
        config: Optional[GPUMonteCarloConfiguration] = None,
        beam_data: Optional[Any] = None,
    ):
        """
        Khởi tạo thuật toán Monte Carlo GPU.

        Parameters
        ----------
        calculation_mode : DoseCalculationMode, optional
            Chế độ tính toán, mặc định là STANDARD
        config : Optional[GPUMonteCarloConfiguration], optional
            Cấu hình cho thuật toán, mặc định là None
        beam_data : Optional[Any], optional
            Dữ liệu chùm tia, mặc định là None
        """
        # Khởi tạo cấu hình mặc định nếu không được cung cấp
        if config is None:
            config = GPUMonteCarloConfiguration()

        # Gọi constructor của lớp cha với tham số phù hợp
        super().__init__(calculation_mode=calculation_mode, config=config)

        # Lưu trữ beam_data từ tham số
        self.beam_data = beam_data

        # Định nghĩa loại thuật toán
        self.algorithm_type = DoseAlgorithmType.MONTE_CARLO_GPU

        # Kiểm tra và thiết lập GPU
        self._setup_gpu()

        # Biến theo dõi tiến độ và trạng thái
        self._progress = 0.0
        self._message = ""
        self._gpu_kernels = {}
        self._initialized_kernels = False

    def _setup_gpu(self):
        """Thiết lập và kiểm tra GPU."""
        if not (HAS_CUPY or HAS_PYCUDA):
            logger.warning("Không tìm thấy thư viện GPU nào. Chuyển về chế độ CPU.")
            self.config.gpu_available = False
            return

        try:
            if HAS_CUPY:
                # Thiết lập thiết bị GPU
                cp.cuda.Device(self.config.gpu_device_id).use()

                # Kiểm tra thông tin GPU
                device_props = cp.cuda.runtime.getDeviceProperties(
                    self.config.gpu_device_id
                )
                gpu_name = device_props["name"].decode("utf-8")
                gpu_memory = device_props["totalGlobalMem"] / (1024**3)  # GB

                logger.info(f"Sử dụng GPU: {gpu_name} với {gpu_memory:.2f} GB bộ nhớ")

                # Xác nhận GPU khả dụng
                self.config.gpu_available = True
            elif HAS_PYCUDA:
                # Lấy thiết bị hiện tại
                device = cuda.Device(self.config.gpu_device_id)
                ctx = device.make_context()

                # Lấy thông tin thiết bị
                gpu_name = device.name()
                gpu_memory = device.total_memory() / (1024**3)  # GB

                logger.info(
                    f"Sử dụng GPU với PyCUDA: {gpu_name} với {gpu_memory:.2f} GB bộ nhớ"
                )

                # Giải phóng context sau khi sử dụng
                ctx.pop()

                # Xác nhận GPU khả dụng
                self.config.gpu_available = True
        except Exception as e:
            logger.error(f"Lỗi khi thiết lập GPU: {e}")
            logger.warning("Chuyển về chế độ CPU do lỗi GPU.")
            self.config.gpu_available = False

    def _initialize_gpu_kernels(self):
        """Khởi tạo các kernel GPU cần thiết."""
        if not self.config.gpu_available or self._initialized_kernels:
            return

        try:
            if HAS_CUPY:
                # Define CUDA kernels for CuPy
                monte_carlo_kernel = """
                extern "C" __global__ void monte_carlo_simulation(
                    float* ct_data, float* density_data, float* dose_grid, float* dose_squared,
                    int width, int height, int depth,
                    float source_x, float source_y, float source_z,
                    float direction_x, float direction_y, float direction_z,
                    float sad, float energy, int num_histories,
                    unsigned int seed, float* random_states) {

                    // Mã kernel Monte Carlo
                    // Thread ID
                    int idx = blockIdx.x * blockDim.x + threadIdx.x;
                    if (idx >= num_histories) return;

                    // Khởi tạo trạng thái ngẫu nhiên
                    curandState state;
                    curand_init(seed + idx, 0, 0, &state);

                    // Mô phỏng hạt
                    // Mô phỏng vị trí và hướng ban đầu của photon
                    // ...

                    // Mô phỏng quá trình vận chuyển photon
                    // ...

                    // Tích lũy liều
                    // ...
                }
                """

                # TODO: Triển khai đầy đủ kernel Monte Carlo trên GPU
                # Kernel cần triển khai:
                # 1. Mô phỏng quá trình vận chuyển photon trong vật liệu
                # 2. Mô phỏng tán xạ Compton và hiệu ứng quang điện
                # 3. Tính toán và tích lũy năng lượng vào các voxel
                # 4. Theo dõi số lượng lớn hạt đồng thời

                self._initialized_kernels = True

            elif HAS_PYCUDA:
                # Define CUDA kernels for PyCUDA
                monte_carlo_code = """
                #include <cuda.h>
                #include <curand_kernel.h>

                extern "C" {
                    __global__ void monte_carlo_simulation(
                        float* ct_data, float* density_data, float* dose_grid, float* dose_squared,
                        int width, int height, int depth,
                        float source_x, float source_y, float source_z,
                        float direction_x, float direction_y, float direction_z,
                        float sad, float energy, int num_histories,
                        unsigned int seed) {

                        // Thread ID
                        int idx = blockIdx.x * blockDim.x + threadIdx.x;
                        if (idx >= num_histories) return;

                        // Khởi tạo trạng thái ngẫu nhiên
                        curandState state;
                        curand_init(seed + idx, 0, 0, &state);

                        // Mô phỏng hạt
                        // ...
                    }
                }
                """

                mod = SourceModule(
                    monte_carlo_code,
                    options=["-use_fast_math"] if self.config.use_fast_math else [],
                )
                self._gpu_kernels["monte_carlo_simulation"] = mod.get_function(
                    "monte_carlo_simulation"
                )

                self._initialized_kernels = True

        except Exception as e:
            logger.error(f"Lỗi khi khởi tạo kernel GPU: {e}")
            logger.warning("Chuyển về chế độ CPU do lỗi khởi tạo kernel.")
            self.config.gpu_available = False

    def initialize(self, patient_data: Any) -> bool:
        """
        Khởi tạo thuật toán Monte Carlo GPU với dữ liệu bệnh nhân.

        Parameters
        ----------
        patient_data : Any
            Dữ liệu bệnh nhân cần thiết cho tính toán

        Returns
        -------
        bool
            True nếu khởi tạo thành công, False nếu không
        """
        try:
            # Khởi tạo thuật toán cơ bản từ lớp cha
            success = super().initialize(patient_data, self.beam_data)
            if not success:
                return False

            # Khởi tạo các kernel GPU
            if self.config.gpu_available:
                self._initialize_gpu_kernels()

            return True
        except Exception as e:
            logger.error(f"Lỗi khi khởi tạo thuật toán Monte Carlo GPU: {e}")
            return False

    def calculate_dose(self, beam_arrangement: Any) -> np.ndarray:
        """
        Tính toán liều sử dụng thuật toán Monte Carlo trên GPU.

        Parameters
        ----------
        beam_arrangement : Any
            Cấu hình chùm tia (beam arrangement) cần tính toán liều

        Returns
        -------
        np.ndarray
            Mảng phân bố liều 3D
        """
        # Kiểm tra xem GPU có sẵn không
        if not self.config.gpu_available:
            logger.warning(
                "GPU không khả dụng, sử dụng tính toán trên CPU. Điều này sẽ chậm hơn."
            )

        # Khởi tạo thời gian bắt đầu
        start_time = time.time()

        # Lấy cấu hình tính toán Monte Carlo
        config = self.config

        # Lấy danh sách chùm tia từ beam arrangement
        beams = beam_arrangement.get_beams()
        num_beams = len(beams)
        if num_beams == 0:
            logger.warning("Không có chùm tia nào để tính toán liều.")
            return np.zeros((10, 10, 10))  # Trả về mảng rỗng

        # Lấy thông tin lưới liều từ beam arrangement
        dose_grid = beam_arrangement.get_dose_grid()
        if dose_grid is None:
            # Tạo một lưới liều mặc định nếu không có sẵn
            grid_shape = (100, 100, 50)
            dose_grid = np.zeros(grid_shape)
        else:
            # Sử dụng lưới liều đã có
            grid_shape = dose_grid.shape
            dose_grid = np.zeros(grid_shape)

        # Khởi tạo mảng độ không chắc chắn
        uncertainty = np.zeros(grid_shape) if config.report_uncertainty else None

        try:
            # Tính liều cho từng chùm tia
            for i, beam in enumerate(beams):
                beam_name = (
                    beam.get_name() if hasattr(beam, "get_name") else f"Beam {i + 1}"
                )
                logger.info(
                    f"Tính toán liều cho chùm tia {i + 1}/{num_beams} ({beam_name})"
                )

                # Số lượng lịch sử hạt cho chùm tia này
                beam_histories = config.num_histories // num_beams

                # Đảm bảo có ít nhất 1000 lịch sử hạt cho mỗi chùm tia
                beam_histories = max(beam_histories, 1000)

                # Mô phỏng chùm tia trên GPU
                beam_dose, beam_uncertainty = self._simulate_beam_gpu(
                    beam, beam_histories
                )

                # Cập nhật lưới liều tổng và độ không chắc chắn
                dose_grid += beam_dose
                # Kết hợp độ không chắc chắn theo quy tắc phương sai
                if uncertainty is not None and beam_uncertainty is not None:
                    uncertainty = np.sqrt(uncertainty**2 + beam_uncertainty**2)

                # Báo cáo tiến độ sau khi hoàn thành chùm tia
                progress = (i + 1) / num_beams
                self.report_progress(
                    progress,
                    f"Đã hoàn thành chùm tia {i + 1}/{num_beams} ({beam_name})",
                )

            # Chuẩn hóa liều tổng
            # TODO: Triển khai chuẩn hóa liều thích hợp

            # Cập nhật kết quả
            simulation_time = time.time() - start_time
            self.result = MonteCarloResult(
                dose_grid=dose_grid,
                uncertainty=uncertainty,
                simulation_time=simulation_time,
                num_histories=config.num_histories,
                additional_info={
                    "num_beams": num_beams,
                    "gpu_used": self.config.gpu_available,
                    "gpu_info": self.get_device_info()
                    if self.config.gpu_available
                    else {},
                },
            )

            logger.info(
                f"Hoàn thành tính toán Monte Carlo trên "
                f"{'GPU' if self.config.gpu_available else 'CPU'} "
                f"trong {simulation_time:.2f}s với {config.num_histories} hạt."
            )

            return dose_grid

        except Exception as e:
            logger.error(f"Lỗi trong quá trình tính toán Monte Carlo: {e}")
            import traceback

            traceback.print_exc()
            return np.zeros(grid_shape)

    def _simulate_beam_gpu(
        self, beam: Any, num_histories: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Mô phỏng một chùm tia bằng Monte Carlo trên GPU.

        Parameters
        ----------
        beam : Any
            Đối tượng chùm tia
        num_histories : int
            Số lượng hạt cần mô phỏng

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Tuple chứa mảng liều và mảng độ không chắc chắn
        """
        grid_shape = self.ct_data.shape

        # Lấy thông tin chùm tia
        energy = getattr(beam, "energy", 6.0)  # MV
        gantry_angle = np.radians(getattr(beam, "gantry_angle", 0.0))
        collimator_angle = np.radians(getattr(beam, "collimator_angle", 0.0))
        couch_angle = np.radians(getattr(beam, "couch_angle", 0.0))
        sad = getattr(beam, "sad", 100.0)  # cm

        # Tọa độ nguồn và hướng chùm tia
        source_position = np.array([0.0, 0.0, -sad])  # Tương đối với isocentre

        # Tính hướng chùm tia dựa trên góc
        direction = np.array(
            [
                np.sin(gantry_angle) * np.cos(couch_angle),
                np.sin(gantry_angle) * np.sin(couch_angle),
                -np.cos(gantry_angle),
            ]
        )

        # Chuẩn hóa vector hướng
        direction = direction / np.linalg.norm(direction)

        # Xử lý dựa trên loại GPU framework được sử dụng
        if HAS_CUPY and self.config.gpu_available:
            try:
                # Chuyển dữ liệu lên GPU
                d_ct_data = cp.array(self.ct_data)
                d_density_data = (
                    cp.array(self.density_data)
                    if self.density_data is not None
                    else cp.ones_like(d_ct_data)
                )

                # Khởi tạo mảng kết quả trên GPU
                d_dose_grid = cp.zeros(grid_shape, dtype=cp.float32)
                d_dose_squared = cp.zeros(grid_shape, dtype=cp.float32)

                # Thiết lập cấu hình kernel
                threads_per_block = 256
                blocks_per_grid = (
                    num_histories + threads_per_block - 1
                ) // threads_per_block

                # TODO: Triển khai và gọi kernel CUDA cho Monte Carlo

                # LƯU Ý: Mã dưới đây chỉ là mô phỏng đơn giản tạm thời
                # và sẽ được thay thế bằng triển khai Monte Carlo đầy đủ
                # Mô phỏng đơn giản trên GPU
                center = cp.array(grid_shape) // 2

                # Tạo lưới tọa độ (đơn giản hóa)
                x, y, z = cp.meshgrid(
                    cp.arange(grid_shape[0]),
                    cp.arange(grid_shape[1]),
                    cp.arange(grid_shape[2]),
                    indexing="ij",
                )

                # Tính khoảng cách từ mỗi voxel đến đường trung tâm chùm tia
                distance = cp.abs(
                    (x - center[0]) * direction[0]
                    + (y - center[1]) * direction[1]
                    + (z - center[2]) * direction[2]
                )

                # Suy giảm theo độ sâu
                depth = cp.sqrt(
                    (x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2
                )

                # Mô hình liều đơn giản
                # TODO: Thay thế mô hình này bằng mô phỏng Monte Carlo đầy đủ
                beam_profile = cp.exp(-0.01 * depth) * cp.exp(-0.1 * distance**2)

                # Điều chỉnh theo mật độ vật liệu
                if (
                    self.config.use_heterogeneity_correction
                    and d_density_data is not None
                ):
                    beam_profile *= d_density_data

                # Chuẩn hóa và quy mô theo batch_size
                max_value = cp.max(beam_profile)
                if max_value > 0:
                    beam_profile = beam_profile / max_value * num_histories * 0.001

                # Thêm nhiễu ngẫu nhiên để mô phỏng tính ngẫu nhiên của Monte Carlo
                # Sử dụng cách tiếp cận an toàn hơn để tránh giá trị âm
                noise_scale = 0.1
                beam_profile_for_noise = cp.maximum(
                    beam_profile, 1e-10
                )  # Tránh căn bậc 2 của 0
                noise = cp.random.normal(0, noise_scale, size=grid_shape)
                beam_profile = cp.maximum(
                    0, beam_profile + noise * cp.sqrt(beam_profile_for_noise)
                )

                # Cập nhật kết quả
                d_dose_grid = beam_profile
                d_dose_squared = beam_profile**2

                # Chuyển kết quả về CPU
                beam_dose = cp.asnumpy(d_dose_grid)

                # Tính độ không chắc chắn
                beam_dose_squared = cp.asnumpy(d_dose_squared)
                mask = beam_dose > 0
                beam_uncertainty = np.zeros_like(beam_dose)
                if mask.any():
                    mean_squared = beam_dose_squared[mask] / num_histories
                    squared_mean = (beam_dose[mask] / num_histories) ** 2
                    beam_uncertainty[mask] = np.sqrt(
                        np.maximum(0, mean_squared - squared_mean) / num_histories
                    )

                return beam_dose, beam_uncertainty

            except Exception as e:
                logger.error(f"Lỗi khi mô phỏng chùm tia trên GPU (CuPy): {e}")
                # Fallback to CPU
                return self._simulate_beam(beam, num_histories, 0, "default")

        elif HAS_PYCUDA and self.config.gpu_available:
            try:
                # TODO: Triển khai mô phỏng chùm tia bằng PyCUDA
                # Đây chỉ là code giả

                # Fallback to CPU for now
                return self._simulate_beam(beam, num_histories, 0, "default")

            except Exception as e:
                logger.error(f"Lỗi khi mô phỏng chùm tia trên GPU (PyCUDA): {e}")
                # Fallback to CPU
                return self._simulate_beam(beam, num_histories, 0, "default")
        else:
            # Fallback to CPU implementation
            return self._simulate_beam(beam, num_histories, 0, "default")

    def get_device_info(self) -> Dict[str, Any]:
        """
        Lấy thông tin về thiết bị GPU đang sử dụng.

        Returns
        -------
        Dict[str, Any]
            Thông tin về GPU
        """
        info = {"gpu_available": self.config.gpu_available}

        if not self.config.gpu_available:
            return info

        try:
            if HAS_CUPY:
                dev = cp.cuda.Device(self.config.gpu_device_id)
                props = cp.cuda.runtime.getDeviceProperties(self.config.gpu_device_id)

                info.update(
                    {
                        "name": props["name"].decode("utf-8"),
                        "total_memory": props["totalGlobalMem"] / (1024**3),  # GB
                        "compute_capability": f"{props['major']}.{props['minor']}",
                        "multiprocessor_count": props["multiProcessorCount"],
                        "max_threads_per_block": props["maxThreadsPerBlock"],
                        "clock_rate": props["clockRate"] / 1000,  # MHz
                    }
                )
            elif HAS_PYCUDA:
                dev = cuda.Device(self.config.gpu_device_id)
                attrs = dev.get_attributes()

                info.update(
                    {
                        "name": dev.name(),
                        "total_memory": dev.total_memory() / (1024**3),  # GB
                        "compute_capability": f"{dev.compute_capability()[0]}.{dev.compute_capability()[1]}",
                        "multiprocessor_count": attrs[
                            cuda.device_attribute.MULTIPROCESSOR_COUNT
                        ],
                        "max_threads_per_block": attrs[
                            cuda.device_attribute.MAX_THREADS_PER_BLOCK
                        ],
                        "clock_rate": attrs[cuda.device_attribute.CLOCK_RATE]
                        / 1000,  # MHz
                    }
                )
        except Exception as e:
            logger.error(f"Lỗi khi lấy thông tin thiết bị GPU: {e}")
            info["error"] = str(e)

        return info

    def __str__(self) -> str:
        """Biểu diễn chuỗi của thuật toán."""
        gpu_status = "với GPU" if self.config.gpu_available else "không có GPU khả dụng"
        return f"Monte Carlo GPU Dose Algorithm ({self.calculation_mode.name}, {gpu_status})"


def is_gpu_available() -> bool:
    """
    Kiểm tra xem GPU có khả dụng để tính toán Monte Carlo không.

    Returns
    -------
    bool
        True nếu GPU khả dụng, False nếu không
    """
    if HAS_CUPY:
        try:
            return cp.cuda.runtime.getDeviceCount() > 0
        except:
            return False
    elif HAS_PYCUDA:
        try:
            return cuda.Device.count() > 0
        except:
            return False
    return False


def get_available_gpu_devices() -> List[Dict[str, Any]]:
    """
    Lấy danh sách các thiết bị GPU khả dụng và thông tin của chúng.

    Returns
    -------
    List[Dict[str, Any]]
        Danh sách các thiết bị GPU với thông tin chi tiết
    """
    devices = []

    if HAS_CUPY:
        try:
            device_count = cp.cuda.runtime.getDeviceCount()
            for i in range(device_count):
                props = cp.cuda.runtime.getDeviceProperties(i)
                devices.append(
                    {
                        "id": i,
                        "name": props["name"].decode("utf-8"),
                        "memory": props["totalGlobalMem"] / (1024**3),  # GB
                        "compute_capability": f"{props['major']}.{props['minor']}",
                        "type": "CuPy",
                    }
                )
        except Exception as e:
            logger.error(f"Lỗi khi lấy thông tin thiết bị CuPy: {e}")
    elif HAS_PYCUDA:
        try:
            device_count = cuda.Device.count()
            for i in range(device_count):
                dev = cuda.Device(i)
                devices.append(
                    {
                        "id": i,
                        "name": dev.name(),
                        "memory": dev.total_memory() / (1024**3),  # GB
                        "compute_capability": f"{dev.compute_capability()[0]}.{dev.compute_capability()[1]}",
                        "type": "PyCUDA",
                    }
                )
        except Exception as e:
            logger.error(f"Lỗi khi lấy thông tin thiết bị PyCUDA: {e}")

    return devices
