#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module thuật toán Monte Carlo cho tính toán liều.

Module này triển khai thuật toán Monte Carlo để tính toán liều chính xác cho
các kỹ thuật xạ trị phức tạp, đặc biệt hiệu quả cho các vùng mô không đồng nhất.
"""

import logging
import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
import multiprocessing
from enum import Enum, auto
import os

from quangtps.dose.algorithms.base import (
    DoseAlgorithm,
    DoseAlgorithmType,
    DoseCalculationMode,
)
from quangtps.dose.dose_grid import DoseGrid
from quangtps.core.patient.image import Image

logger = logging.getLogger(__name__)


class ParticleType(Enum):
    """Loại hạt trong mô phỏng Monte Carlo."""

    PHOTON = auto()
    ELECTRON = auto()
    POSITRON = auto()


class InteractionType(Enum):
    """Loại tương tác vật lý được mô phỏng."""

    PHOTOELECTRIC = auto()
    COMPTON = auto()
    PAIR_PRODUCTION = auto()
    BREMSSTRAHLUNG = auto()
    IONIZATION = auto()
    MULTIPLE_SCATTERING = auto()


class MCConfiguration:
    """Cấu hình cho thuật toán Monte Carlo."""

    def __init__(self):
        """Khởi tạo cấu hình mặc định."""
        # Số lượng hạt
        self.num_histories = 1000000

        # Năng lượng cut-off
        self.photon_cutoff = 0.01  # MeV
        self.electron_cutoff = 0.2  # MeV

        # Các loại tương tác cần mô phỏng
        self.simulate_photoelectric = True
        self.simulate_compton = True
        self.simulate_pair_production = True
        self.simulate_bremsstrahlung = True

        # Thông số hiệu suất
        self.use_variance_reduction = True
        self.use_multithreading = True
        self.num_threads = multiprocessing.cpu_count()

        # Thông số vật lý
        self.use_heterogeneity_correction = True
        self.use_density_scaling = True

        # Các thông số đầu ra
        self.report_uncertainty = True
        self.save_intermediate_results = False
        self.intermediate_results_dir = "./mc_results"

        # Giá trị ngẫu nhiên cố định (để tái tạo)
        self.random_seed = None


class MonteCarloResult:
    """Kết quả tính toán Monte Carlo."""

    def __init__(
        self, dose_grid, uncertainty=None, simulation_time=0.0, num_histories=0
    ):
        """Khởi tạo kết quả với liều và thông tin bổ sung."""
        self.dose_grid = dose_grid
        self.uncertainty = uncertainty
        self.simulation_time = simulation_time
        self.num_histories = num_histories
        self.mean_uncertainty = (
            np.mean(uncertainty) if uncertainty is not None else None
        )
        self.max_uncertainty = np.max(uncertainty) if uncertainty is not None else None

    def get_uncertainty_stats(self):
        """Trả về thông tin thống kê về độ không chắc chắn."""
        if self.uncertainty is None:
            return None

        return {
            "mean": self.mean_uncertainty,
            "max": self.max_uncertainty,
            "min": np.min(self.uncertainty),
            "median": np.median(self.uncertainty),
            "std": np.std(self.uncertainty),
        }


class MonteCarloAlgorithm(DoseAlgorithm):
    """
    Thuật toán Monte Carlo để tính toán liều.

    Lớp này triển khai thuật toán Monte Carlo CPU để tính toán liều chính xác,
    đặc biệt hữu ích cho các vùng có mật độ không đồng nhất như phổi hoặc khoang mũi.
    """

    def __init__(
        self,
        calculation_mode: DoseCalculationMode = DoseCalculationMode.STANDARD,
        config: Optional[MCConfiguration] = None,
    ):
        """
        Khởi tạo thuật toán Monte Carlo.

        Args:
            calculation_mode: Chế độ tính toán (FAST, STANDARD, ACCURATE)
            config: Cấu hình cho thuật toán Monte Carlo
        """
        super().__init__(
            algorithm_type=DoseAlgorithmType.MONTE_CARLO,
            calculation_mode=calculation_mode,
            use_heterogeneity_correction=True,
            grid_size=0.25,  # cm
        )

        self.config = config or MCConfiguration()

        # Tự động điều chỉnh thông số dựa trên chế độ tính toán
        if calculation_mode == DoseCalculationMode.FAST:
            self.config.num_histories = min(self.config.num_histories, 100000)
            self.grid_size = 0.5  # cm
        elif calculation_mode == DoseCalculationMode.ACCURATE:
            self.config.num_histories = max(self.config.num_histories, 10000000)
            self.grid_size = 0.2  # cm

        # Dữ liệu hình học và vật liệu
        self.ct_data = None
        self.material_data = None
        self.density_data = None
        self.geometry = None

        # Dữ liệu chùm tia
        self.beam_data = {}

        # Kết quả
        self.result = None
        self.uncertainty_grid = None

        # Các bảng dữ liệu vật lý
        self._init_physics_tables()

    def _init_physics_tables(self):
        """Khởi tạo các bảng dữ liệu vật lý cho mô phỏng Monte Carlo."""
        # Tạo các bảng dữ liệu mặc định
        self.material_tables = {}
        self.cross_sections = {}
        self.stopping_powers = {}

        # TODO: Triển khai việc tải các bảng dữ liệu vật lý đầy đủ
        # Trong triển khai thực tế, đây sẽ là nơi tải dữ liệu từ các file như XCOM
        logger.info("Khởi tạo bảng vật lý đang sử dụng các giá trị mặc định")

    def initialize(self, geometry_data: Any, beam_data: Any) -> bool:
        """
        Khởi tạo thuật toán với dữ liệu hình học và dữ liệu chùm tia.

        Args:
            geometry_data: Dữ liệu hình học (CT, cấu trúc,...)
            beam_data: Dữ liệu chùm tia

        Returns:
            True nếu khởi tạo thành công, False nếu không
        """
        try:
            # Khởi tạo thư mục kết quả trung gian nếu cần
            if self.config.save_intermediate_results:
                os.makedirs(self.config.intermediate_results_dir, exist_ok=True)

            # Lưu trữ dữ liệu
            self.geometry = geometry_data
            self.beam_data = beam_data

            # Trích xuất dữ liệu CT
            if hasattr(geometry_data, "ct_image") and isinstance(
                geometry_data.ct_image, Image
            ):
                self.ct_data = geometry_data.ct_image.get_array()
                self.density_data = self._convert_hu_to_density(self.ct_data)
                self.material_data = self._assign_materials(self.density_data)
            else:
                logger.warning("Không tìm thấy dữ liệu CT hợp lệ")
                return False

            # Kiểm tra và chuẩn bị dữ liệu chùm tia
            if not beam_data or not hasattr(beam_data, "beams") or not beam_data.beams:
                logger.warning("Không tìm thấy dữ liệu chùm tia hợp lệ")
                return False

            # Khởi tạo ngẫu nhiên nếu có seed
            if self.config.random_seed is not None:
                np.random.seed(self.config.random_seed)

            self.is_initialized = True
            return True

        except Exception as e:
            logger.error(f"Lỗi khởi tạo thuật toán Monte Carlo: {e}")
            return False

    def _convert_hu_to_density(self, ct_data: np.ndarray) -> np.ndarray:
        """
        Chuyển đổi giá trị Hounsfield (HU) sang mật độ vật liệu.

        Args:
            ct_data: Mảng giá trị HU

        Returns:
            Mảng mật độ vật liệu (g/cm³)
        """
        # Công thức đơn giản: ρ = 1.0 + HU/1000
        # Với HU = 0 -> nước (ρ = 1.0 g/cm³)
        density = 1.0 + ct_data / 1000.0

        # Giới hạn giá trị hợp lý
        density = np.clip(density, 0.01, 10.0)

        return density

    def _assign_materials(self, density_data: np.ndarray) -> np.ndarray:
        """
        Gán vật liệu dựa trên mật độ.

        Args:
            density_data: Mảng mật độ vật liệu

        Returns:
            Mảng mã vật liệu
        """
        # Triển khai đơn giản với 4 loại vật liệu
        material_map = np.zeros_like(density_data, dtype=np.int8)

        # Phân loại vật liệu dựa trên mật độ
        # 0: Không khí (ρ < 0.2)
        # 1: Phổi (0.2 <= ρ < 0.8)
        # 2: Mô mềm (0.8 <= ρ < 1.2)
        # 3: Xương (ρ >= 1.2)
        material_map[density_data < 0.2] = 0
        material_map[(density_data >= 0.2) & (density_data < 0.8)] = 1
        material_map[(density_data >= 0.8) & (density_data < 1.2)] = 2
        material_map[density_data >= 1.2] = 3

        return material_map

    def calculate_dose(self, beam_arrangement: Any) -> np.ndarray:
        """
        Tính toán phân bố liều cho một cấu hình chùm tia sử dụng Monte Carlo.

        Args:
            beam_arrangement: Cấu hình chùm tia

        Returns:
            Mảng 3D chứa phân bố liều tính toán
        """
        if not self.is_initialized:
            raise RuntimeError("Thuật toán chưa được khởi tạo")

        start_time = time.time()
        self.is_calculating = True

        try:
            # Báo cáo tiến độ ban đầu
            self.report_progress(0.0, "Bắt đầu mô phỏng Monte Carlo...")

            # Lấy thông tin lưới tính toán
            grid_shape = self.ct_data.shape

            # Tạo lưới liều và lưới độ không chắc chắn
            dose_grid = np.zeros(grid_shape)
            uncertainty_grid = np.zeros(grid_shape)

            # Số lượng chùm tia
            beams = beam_arrangement.beams
            num_beams = len(beams)

            # Tổng số hạt cần mô phỏng
            total_histories = self.config.num_histories

            # Phân bổ số hạt theo tỷ lệ với MU của mỗi chùm tia
            if hasattr(beams[0], "meterset") and beams[0].meterset is not None:
                total_mu = sum(beam.meterset for beam in beams)
                histories_per_beam = [
                    int(beam.meterset / total_mu * total_histories) for beam in beams
                ]
            else:
                # Phân bổ đều nếu không có thông tin MU
                histories_per_beam = [total_histories // num_beams] * num_beams
                # Đảm bảo tổng số đúng
                histories_per_beam[-1] += total_histories - sum(histories_per_beam)

            # Mô phỏng từng chùm tia
            for i, beam in enumerate(beams):
                beam_name = getattr(beam, "name", f"Beam_{i + 1}")
                num_histories = histories_per_beam[i]

                # Báo cáo tiến độ
                progress = i / num_beams
                self.report_progress(
                    progress,
                    f"Mô phỏng chùm tia {i + 1}/{num_beams} ({beam_name}) với {num_histories} hạt",
                )

                # Tính toán liều cho chùm tia hiện tại
                beam_dose, beam_uncertainty = self._simulate_beam(beam, num_histories)

                # Cập nhật lưới liều tổng và độ không chắc chắn
                dose_grid += beam_dose
                # Kết hợp độ không chắc chắn theo quy tắc phương sai
                uncertainty_grid = np.sqrt(uncertainty_grid**2 + beam_uncertainty**2)

                # Báo cáo tiến độ sau khi hoàn thành chùm tia
                progress = (i + 1) / num_beams
                self.report_progress(
                    progress,
                    f"Đã hoàn thành chùm tia {i + 1}/{num_beams} ({beam_name})",
                )

            # Chuẩn hóa liều tổng
            # TODO: Triển khai chuẩn hóa liều thích hợp

            # Cập nhật kết quả
            self.result = MonteCarloResult(
                dose_grid=dose_grid,
                uncertainty=uncertainty_grid,
                simulation_time=time.time() - start_time,
                num_histories=total_histories,
            )

            # Ghi nhớ thời gian tính toán
            self.last_calculation_time = time.time() - start_time

            # Báo cáo hoàn thành
            self.report_progress(1.0, "Đã hoàn thành mô phỏng Monte Carlo")
            self.is_calculating = False

            return dose_grid

        except Exception as e:
            self.is_calculating = False
            logger.error(f"Lỗi khi tính toán liều Monte Carlo: {e}")
            raise

    def _simulate_beam(
        self, beam: Any, num_histories: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Mô phỏng một chùm tia bằng Monte Carlo.

        Args:
            beam: Đối tượng chùm tia
            num_histories: Số lượng hạt cần mô phỏng

        Returns:
            Tuple chứa mảng liều và mảng độ không chắc chắn
        """
        grid_shape = self.ct_data.shape
        beam_dose = np.zeros(grid_shape)
        beam_dose_squared = np.zeros_like(beam_dose)  # Để tính độ không chắc chắn

        # Lấy thông tin chùm tia
        energy = getattr(beam, "energy", 6.0)  # MV
        gantry_angle = getattr(beam, "gantry_angle", 0.0)  # độ
        collimator_angle = getattr(beam, "collimator_angle", 0.0)  # độ
        couch_angle = getattr(beam, "couch_angle", 0.0)  # độ
        sad = getattr(beam, "sad", 100.0)  # cm

        # Lấy thông tin trường xạ
        if hasattr(beam, "mlc") and beam.mlc is not None:
            # TODO: Xử lý MLC
            field_type = "MLC"
        elif hasattr(beam, "jaws") and beam.jaws is not None:
            field_type = "JAWS"
            # TODO: Lấy thông tin hàm
        else:
            # Giả định trường vuông mặc định
            field_type = "DEFAULT"
            field_size = (10.0, 10.0)  # cm

        # Batch processing để giảm overhead
        batch_size = min(10000, num_histories)
        num_batches = (num_histories + batch_size - 1) // batch_size

        # Sử dụng đa luồng nếu được yêu cầu
        if self.config.use_multithreading and self.config.num_threads > 1:
            with multiprocessing.Pool(self.config.num_threads) as pool:
                # Tạo các tham số cho mỗi batch
                batch_params = [
                    (
                        batch_size
                        if i < num_batches - 1
                        else num_histories - (num_batches - 1) * batch_size,
                        i,
                        beam,
                        field_type,
                    )
                    for i in range(num_batches)
                ]

                # Chạy mô phỏng song song
                results = pool.map(self._simulate_batch, batch_params)

                # Kết hợp kết quả
                for batch_dose, batch_dose_squared in results:
                    beam_dose += batch_dose
                    beam_dose_squared += batch_dose_squared
        else:
            # Mô phỏng tuần tự
            for i in range(num_batches):
                current_batch_size = (
                    batch_size
                    if i < num_batches - 1
                    else num_histories - (num_batches - 1) * batch_size
                )
                batch_dose, batch_dose_squared = self._simulate_batch(
                    (current_batch_size, i, beam, field_type)
                )
                beam_dose += batch_dose
                beam_dose_squared += batch_dose_squared

                # Báo cáo tiến độ mỗi batch
                sub_progress = (i + 1) / num_batches
                self.report_progress(
                    -1, f"Đã xử lý {i + 1}/{num_batches} batches cho chùm tia hiện tại"
                )

        # Tính độ không chắc chắn
        # σ = sqrt((E[X²] - E[X]²) / N) = sqrt((sum(x²)/N - (sum(x)/N)²))
        mask = beam_dose > 0
        beam_uncertainty = np.zeros_like(beam_dose)
        if mask.any():
            mean_squared = beam_dose_squared[mask] / num_histories
            squared_mean = (beam_dose[mask] / num_histories) ** 2
            beam_uncertainty[mask] = np.sqrt(
                np.maximum(0, mean_squared - squared_mean) / num_histories
            )

        return beam_dose, beam_uncertainty

    def _simulate_batch(self, args) -> Tuple[np.ndarray, np.ndarray]:
        """
        Mô phỏng một batch hạt.

        Args:
            args: Tuple chứa (batch_size, batch_index, beam, field_type)

        Returns:
            Tuple chứa (mảng liều, mảng bình phương liều)
        """
        batch_size, batch_index, beam, field_type = args
        grid_shape = self.ct_data.shape
        batch_dose = np.zeros(grid_shape)
        batch_dose_squared = np.zeros_like(batch_dose)

        # Thiết lập source cho batch này
        if self.config.random_seed is not None:
            # Tạo seed khác nhau cho mỗi batch dựa trên seed gốc
            np.random.seed(self.config.random_seed + batch_index)

        # TODO: Triển khai mô phỏng Monte Carlo chi tiết
        # Đây là nơi để mô phỏng quá trình vận chuyển hạt (particle transport)

        # Mô phỏng giả định đơn giản
        # Trong triển khai thực tế, đây sẽ là một thuật toán phức tạp hơn nhiều

        # Lấy thông tin chùm tia
        energy = getattr(beam, "energy", 6.0)  # MV
        gantry_angle = np.radians(getattr(beam, "gantry_angle", 0.0))

        # Tạo vị trí trung tâm lưới
        center = np.array(grid_shape) // 2

        # Tạo hướng chùm tia dựa trên góc gantry
        direction = np.array([np.sin(gantry_angle), 0, -np.cos(gantry_angle)])

        # Tạo lưới tọa độ (đơn giản hóa)
        x, y, z = np.meshgrid(
            np.arange(grid_shape[0]),
            np.arange(grid_shape[1]),
            np.arange(grid_shape[2]),
            indexing="ij",
        )

        # Tính khoảng cách từ mỗi voxel đến đường trung tâm chùm tia
        # Đây là mô phỏng đơn giản không chính xác
        distance = np.abs(
            (x - center[0]) * direction[0]
            + (y - center[1]) * direction[1]
            + (z - center[2]) * direction[2]
        )

        # Sử dụng mô hình phân bố liều đơn giản cho mô phỏng
        # Suy giảm theo độ sâu
        depth = np.sqrt(
            (x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2
        )
        beam_profile = np.exp(-0.01 * depth) * np.exp(-0.1 * distance**2)

        # Điều chỉnh theo mật độ vật liệu
        if self.density_data is not None and self.config.use_heterogeneity_correction:
            beam_profile *= self.density_data

        # Chuẩn hóa và quy mô theo batch_size
        if np.max(beam_profile) > 0:
            beam_profile = beam_profile / np.max(beam_profile) * batch_size * 0.001

        # Thêm nhiễu ngẫu nhiên để mô phỏng tính ngẫu nhiên của Monte Carlo
        noise = np.random.normal(0, 0.1, size=grid_shape)
        beam_profile = np.maximum(0, beam_profile + noise * np.sqrt(beam_profile))

        # Cập nhật kết quả batch
        batch_dose = beam_profile
        batch_dose_squared = beam_profile**2

        return batch_dose, batch_dose_squared

    def get_uncertainty(self) -> Optional[np.ndarray]:
        """
        Lấy ma trận độ không chắc chắn của tính toán Monte Carlo.

        Returns:
            Ma trận độ không chắc chắn hoặc None nếu chưa tính toán
        """
        if self.result is None:
            return None

        return self.result.uncertainty

    def get_simulation_stats(self) -> Dict[str, Any]:
        """
        Lấy thông tin thống kê về quá trình mô phỏng.

        Returns:
            Dictionary chứa thông tin thống kê
        """
        if self.result is None:
            return {}

        return {
            "simulation_time": self.result.simulation_time,
            "num_histories": self.result.num_histories,
            "efficiency": self.result.num_histories / self.result.simulation_time
            if self.result.simulation_time > 0
            else 0,
            "uncertainty": self.result.get_uncertainty_stats(),
            "algorithm_type": self.algorithm_type.name,
            "calculation_mode": self.calculation_mode.name,
        }

    def __str__(self) -> str:
        """Biểu diễn chuỗi của thuật toán."""
        return f"Monte Carlo Dose Algorithm ({self.calculation_mode.name})"
