#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module thuật toán Monte Carlo cho tính toán liều - Compatibility module.

Module này là module khả năng tương thích ngược, đơn giản chuyển hướng tới triển khai
mới trong monte_carlo.py với hỗ trợ GPU.
"""

import logging
import warnings
from enum import Enum, auto

logger = logging.getLogger(__name__)

# Phát cảnh báo về module không dùng nữa
warnings.warn(
    "Module 'montecarlo.py' is deprecated and will be removed in the future. "
    "Please use 'monte_carlo.py' instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export các lớp từ monte_carlo.py
from quangtps.dose.algorithms.monte_carlo import (
    MonteCarloAlgorithm,
    DoseCalculationResult,
)


# Định nghĩa lại các enum để đảm bảo khả năng tương thích ngược
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


# Định nghĩa lại lớp MCConfiguration để đảm bảo khả năng tương thích ngược
class MCConfiguration:
    """
    Lớp cấu hình cho thuật toán Monte Carlo - Compatibility class.

    Lớp này chỉ dùng để đảm bảo tương thích ngược. Các thiết lập thực tế
    sẽ được chuyển tới triển khai mới trong MonteCarloAlgorithm.
    """

    def __init__(self):
        """Khởi tạo cấu hình mặc định."""
        # Phát cảnh báo về việc sử dụng lớp không dùng nữa
        warnings.warn(
            "MCConfiguration is deprecated. Please use MonteCarloAlgorithm.set_parameters() instead.",
            DeprecationWarning,
            stacklevel=2,
        )

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
        self.num_threads = -1  # Sử dụng tất cả lõi CPU

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
    """Lớp kết quả tính toán Monte Carlo - Compatibility class."""

    def __init__(
        self, dose_grid, uncertainty=None, simulation_time=0.0, num_histories=0
    ):
        """Khởi tạo kết quả với liều và thông tin bổ sung."""
        # Phát cảnh báo về việc sử dụng lớp không dùng nữa
        warnings.warn(
            "MonteCarloResult is deprecated. Please use DoseCalculationResult instead.",
            DeprecationWarning,
            stacklevel=2,
        )

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


# Import numpy ở cuối để tránh lỗi trong MonteCarloResult
import numpy as np
