#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module base cho các thuật toán tính toán liều.

Module này định nghĩa các lớp cơ sở và các enum cho các thuật toán
tính toán liều trong QuangTPS.
"""

import logging
import numpy as np
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field

from quangtps.dose.dose_grid import DoseGrid

logger = logging.getLogger(__name__)


class DoseAlgorithmType(Enum):
    """Loại thuật toán tính toán liều."""

    NONE = auto()
    GENERIC = auto()
    PDD_SAR = auto()  # Percentage Depth Dose / Scatter-Air Ratio
    CONVOLUTION = auto()
    SUPERPOSITION = auto()
    AAA = auto()  # Anisotropic Analytical Algorithm
    CCC = auto()  # Collapsed Cone Convolution
    ACUROS = auto()  # Acuros XB
    MONTE_CARLO = auto()
    MONTE_CARLO_GPU = auto()  # Monte Carlo tăng tốc bằng GPU
    VMC = auto()  # Voxel Monte Carlo
    ELECTRON_MONTE_CARLO = auto()
    # Thêm các thuật toán khác theo cần thiết


class DoseCalculationMode(Enum):
    """Các chế độ tính toán liều."""

    FAST = auto()  # Tính toán nhanh, độ chính xác thấp
    STANDARD = auto()  # Cân bằng giữa tốc độ và độ chính xác
    ACCURATE = auto()  # Độ chính xác cao, tốc độ chậm
    CLINICAL = auto()  # Phù hợp cho tính toán lâm sàng
    RESEARCH = auto()  # Phù hợp cho nghiên cứu
    CUSTOM = auto()  # Chế độ tùy chỉnh


class DoseCalculationAlgorithm(Enum):
    """Enum tương thích với dose_engine.py cho các thuật toán tính liều."""

    # Mapping tương thích với DoseAlgorithmType
    NONE = "none"
    GENERIC = "generic"
    PDD_SAR = "pdd_sar"
    CONVOLUTION = "convolution"
    SUPERPOSITION = "superposition"
    AAA = "aaa"
    CCC = "ccc"  # Collapsed Cone Convolution
    ACUROS_XB = "acuros_xb"
    MONTE_CARLO = "monte_carlo"
    MONTE_CARLO_GPU = "monte_carlo_gpu"
    VMC = "vmc"
    ELECTRON_MONTE_CARLO = "electron_monte_carlo"

    # Thêm các aliases phổ biến
    COLLAPSED_CONE = "ccc"
    ACUROS = "acuros_xb"

    @classmethod
    def from_algorithm_type(cls, algorithm_type: DoseAlgorithmType):
        """Chuyển đổi từ DoseAlgorithmType sang DoseCalculationAlgorithm."""
        mapping = {
            DoseAlgorithmType.NONE: cls.NONE,
            DoseAlgorithmType.GENERIC: cls.GENERIC,
            DoseAlgorithmType.PDD_SAR: cls.PDD_SAR,
            DoseAlgorithmType.CONVOLUTION: cls.CONVOLUTION,
            DoseAlgorithmType.SUPERPOSITION: cls.SUPERPOSITION,
            DoseAlgorithmType.AAA: cls.AAA,
            DoseAlgorithmType.CCC: cls.CCC,
            DoseAlgorithmType.ACUROS: cls.ACUROS_XB,
            DoseAlgorithmType.MONTE_CARLO: cls.MONTE_CARLO,
            DoseAlgorithmType.MONTE_CARLO_GPU: cls.MONTE_CARLO_GPU,
            DoseAlgorithmType.VMC: cls.VMC,
            DoseAlgorithmType.ELECTRON_MONTE_CARLO: cls.ELECTRON_MONTE_CARLO,
        }
        return mapping.get(algorithm_type, cls.GENERIC)


class DoseAlgorithm(ABC):
    """
    Lớp cơ sở cho các thuật toán tính toán liều.

    Lớp này định nghĩa giao diện cơ bản mà tất cả các thuật toán
    tính toán liều phải triển khai.
    """

    def __init__(
        self,
        algorithm_type: DoseAlgorithmType = DoseAlgorithmType.GENERIC,
        calculation_mode: DoseCalculationMode = DoseCalculationMode.STANDARD,
        use_heterogeneity_correction: bool = True,
        grid_size: float = 0.25,  # cm
    ):
        """
        Khởi tạo thuật toán tính toán liều.

        Args:
            algorithm_type: Loại thuật toán
            calculation_mode: Chế độ tính toán
            use_heterogeneity_correction: Có sử dụng hiệu chỉnh không đồng nhất không
            grid_size: Kích thước lưới tính liều (cm)
        """
        self.algorithm_type = algorithm_type
        self.calculation_mode = calculation_mode
        self.use_heterogeneity_correction = use_heterogeneity_correction
        self.grid_size = grid_size

        # Trạng thái tính toán
        self.is_initialized = False
        self.is_calculating = False
        self.progress = 0.0

        # Lưu trữ kết quả tính toán
        self.last_calculation_time = 0.0
        self.last_calculation_memory = 0.0

        # Callback cho tiến độ
        self.progress_callback = None

    @abstractmethod
    def initialize(self, geometry_data: Any, beam_data: Any) -> bool:
        """
        Khởi tạo thuật toán với dữ liệu hình học và dữ liệu chùm tia.

        Args:
            geometry_data: Dữ liệu hình học (CT, cấu trúc,...)
            beam_data: Dữ liệu chùm tia

        Returns:
            True nếu khởi tạo thành công, False nếu không
        """
        pass

    @abstractmethod
    def calculate_dose(self, beam_arrangement: Any) -> np.ndarray:
        """
        Tính toán phân bố liều cho một cấu hình chùm tia.

        Args:
            beam_arrangement: Cấu hình chùm tia

        Returns:
            Mảng 3D chứa phân bố liều tính toán
        """
        pass

    def set_progress_callback(self, callback: callable) -> None:
        """
        Thiết lập hàm callback để báo cáo tiến độ tính toán.

        Args:
            callback: Hàm callback nhận giá trị tiến độ (0-1) và thông báo
        """
        self.progress_callback = callback

    def report_progress(self, progress: float, message: str = "") -> None:
        """
        Báo cáo tiến độ tính toán.

        Args:
            progress: Giá trị tiến độ (0-1)
            message: Thông báo kèm theo
        """
        self.progress = progress
        if self.progress_callback:
            self.progress_callback(progress, message)

    def get_algorithm_info(self) -> Dict[str, Any]:
        """
        Lấy thông tin về thuật toán.

        Returns:
            Dictionary chứa thông tin về thuật toán
        """
        return {
            "algorithm_type": self.algorithm_type.name,
            "calculation_mode": self.calculation_mode.name,
            "use_heterogeneity_correction": self.use_heterogeneity_correction,
            "grid_size": self.grid_size,
            "last_calculation_time": self.last_calculation_time,
            "last_calculation_memory": self.last_calculation_memory,
        }

    def __str__(self) -> str:
        """Biểu diễn chuỗi của thuật toán."""
        return f"{self.algorithm_type.name} ({self.calculation_mode.name})"


class GenericDoseAlgorithm(DoseAlgorithm):
    """
    Thuật toán tính toán liều tổng quát đơn giản.

    Lớp này triển khai một thuật toán tính toán liều đơn giản
    có thể sử dụng khi không cần độ chính xác cao.
    """

    def __init__(
        self,
        calculation_mode: DoseCalculationMode = DoseCalculationMode.FAST,
        use_heterogeneity_correction: bool = False,
        grid_size: float = 0.5,
    ):
        """
        Khởi tạo thuật toán tính toán liều tổng quát.

        Args:
            calculation_mode: Chế độ tính toán
            use_heterogeneity_correction: Có sử dụng hiệu chỉnh không đồng nhất không
            grid_size: Kích thước lưới tính liều (cm)
        """
        super().__init__(
            algorithm_type=DoseAlgorithmType.GENERIC,
            calculation_mode=calculation_mode,
            use_heterogeneity_correction=use_heterogeneity_correction,
            grid_size=grid_size,
        )

        # Các tham số bổ sung
        self.penumbra_model = "gaussian"
        self.scatter_model = "simple"

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
            # Lưu trữ dữ liệu
            self.geometry_data = geometry_data
            self.beam_data = beam_data

            # TODO: Khởi tạo các tham số cần thiết cho tính toán

            self.is_initialized = True
            return True
        except Exception as e:
            logger.error(f"Lỗi khởi tạo thuật toán tính liều: {e}")
            return False

    def calculate_dose(self, beam_arrangement: Any) -> np.ndarray:
        """
        Tính toán phân bố liều cho một cấu hình chùm tia.

        Args:
            beam_arrangement: Cấu hình chùm tia

        Returns:
            Mảng 3D chứa phân bố liều tính toán
        """
        if not self.is_initialized:
            raise RuntimeError("Thuật toán chưa được khởi tạo")

        try:
            import time

            start_time = time.time()

            self.is_calculating = True

            # Báo cáo tiến độ ban đầu
            self.report_progress(0.0, "Bắt đầu tính toán liều...")

            # TODO: Triển khai thuật toán tính toán liều thực tế
            # Đây chỉ là mã giả định

            # 1. Lấy thông tin lưới tính toán từ dữ liệu hình học
            grid_shape = getattr(self.geometry_data, "grid_shape", (100, 100, 100))

            # 2. Tạo ma trận liều ban đầu với giá trị 0
            dose_matrix = np.zeros(grid_shape)

            # 3. Mô phỏng tính toán liều
            num_beams = len(getattr(beam_arrangement, "beams", [1]))

            for i in range(num_beams):
                # Báo cáo tiến độ
                progress = (i + 0.5) / num_beams
                self.report_progress(
                    progress, f"Đang tính liều cho chùm tia {i + 1}/{num_beams}"
                )

                # TODO: Tính toán liều cho mỗi chùm tia
                # Đây chỉ là mẫu để minh họa

                # Giả lập một phân bố liều đơn giản
                # (Trong triển khai thực tế, đây sẽ là tính toán phức tạp hơn nhiều)
                beam_dose = self._calculate_simple_beam_dose(grid_shape, i)

                # Cộng vào ma trận liều tổng
                dose_matrix += beam_dose

            # 4. Hoàn thành tính toán
            self.is_calculating = False

            # Báo cáo tiến độ hoàn thành
            self.report_progress(1.0, "Đã hoàn thành tính toán liều")

            # Lưu thông tin tính toán
            self.last_calculation_time = time.time() - start_time

            return dose_matrix

        except Exception as e:
            self.is_calculating = False
            logger.error(f"Lỗi khi tính toán liều: {e}")
            raise

    def _calculate_simple_beam_dose(
        self, grid_shape: Tuple[int, int, int], beam_index: int
    ) -> np.ndarray:
        """
        Tính toán phân bố liều đơn giản cho một chùm tia.

        Args:
            grid_shape: Kích thước lưới tính liều
            beam_index: Chỉ số của chùm tia

        Returns:
            Mảng 3D chứa phân bố liều của chùm tia
        """
        # Tạo một mẫu phân bố liều đơn giản
        # (Trong ứng dụng thực tế, đây sẽ là một tính toán phức tạp)

        # Tạo ma trận liều ban đầu với giá trị 0
        dose = np.zeros(grid_shape)

        # Lấy tọa độ trung tâm của lưới
        center = np.array(grid_shape) // 2

        # Tạo lưới tọa độ
        x, y, z = np.meshgrid(
            np.arange(grid_shape[0]),
            np.arange(grid_shape[1]),
            np.arange(grid_shape[2]),
            indexing="ij",
        )

        # Tính khoảng cách từ mỗi điểm đến trung tâm
        distance = np.sqrt(
            (x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2
        )

        # Mô phỏng chùm tia với sự phân rã theo độ sâu
        # và phân bố Gaussian xuyên ngang

        # Tạo hướng chùm tia (đơn giản hóa)
        if beam_index == 0:
            # Chùm tia theo trục X
            depth = np.abs(x - center[0])
            lateral_distance = np.sqrt((y - center[1]) ** 2 + (z - center[2]) ** 2)
        elif beam_index == 1:
            # Chùm tia theo trục Y
            depth = np.abs(y - center[1])
            lateral_distance = np.sqrt((x - center[0]) ** 2 + (z - center[2]) ** 2)
        else:
            # Chùm tia theo hướng khác
            # Góc nghiêng đơn giản
            angle = beam_index * np.pi / 4
            depth = np.abs(
                np.cos(angle) * (x - center[0]) + np.sin(angle) * (y - center[1])
            )
            lateral_distance = np.abs(
                -np.sin(angle) * (x - center[0]) + np.cos(angle) * (y - center[1])
            )

        # Mô phỏng phân rã theo độ sâu (PDD)
        depth_dose = np.exp(-0.05 * depth)

        # Mô phỏng penumbra và sự phân bố ngang (profile)
        sigma = 10.0  # Độ rộng của profile
        lateral_dose = np.exp(-(lateral_distance**2) / (2 * sigma**2))

        # Kết hợp để tạo phân bố liều tổng thể
        dose = depth_dose * lateral_dose

        # Chuẩn hóa liều
        max_dose = np.max(dose)
        if max_dose > 0:
            dose = dose / max_dose

        return dose


@dataclass
class DoseCalculationResult:
    """Kết quả tính toán liều."""

    dose_grid: Any  # DoseGrid object
    calculation_time: float = 0.0  # seconds
    algorithm_used: str = ""

    # Statistics
    max_dose: float = 0.0  # Gy
    min_dose: float = 0.0  # Gy
    mean_dose: float = 0.0  # Gy

    # Metadata
    calculation_parameters: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def is_successful(self) -> bool:
        """Kiểm tra tính toán có thành công không."""
        return len(self.errors) == 0

    def get_summary(self) -> Dict[str, Any]:
        """Lấy summary của kết quả."""
        return {
            "algorithm": self.algorithm_used,
            "calculation_time": self.calculation_time,
            "max_dose": self.max_dose,
            "min_dose": self.min_dose,
            "mean_dose": self.mean_dose,
            "successful": self.is_successful(),
            "warnings_count": len(self.warnings),
            "errors_count": len(self.errors),
        }
