#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module quản lý các thuật toán tính liều trong QuangTPS.

Module này cung cấp các lớp và hàm để quản lý và sử dụng các thuật toán
tính toán liều khác nhau, bao gồm Pencil Beam, Collapsed Cone, Monte Carlo, v.v.
"""

import logging
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Type

logger = logging.getLogger(__name__)


# Định nghĩa các lớp cơ sở
class DoseCalculationAlgorithm:
    """Lớp cơ sở cho tất cả các thuật toán tính liều."""

    def __init__(self):
        """Khởi tạo thuật toán tính liều."""
        pass

    def initialize(self, patient_data):
        """Khởi tạo thuật toán với dữ liệu bệnh nhân."""
        pass

    def calculate_dose(self, beam_arrangement):
        """Tính phân bố liều cho các chùm tia xác định."""
        pass


class DoseCalculationResult:
    """Lớp cơ sở cho kết quả tính toán liều."""

    def __init__(self):
        """Khởi tạo kết quả tính liều."""
        pass


# Tạo lớp giả cho các thuật toán không có sẵn
class PencilBeamAlgorithm(DoseCalculationAlgorithm):
    """Cài đặt mặc định cho thuật toán Pencil Beam nếu không có sẵn."""

    def __init__(self):
        """Khởi tạo thuật toán Pencil Beam."""
        super().__init__()
        logger.warning("Sử dụng phiên bản giả của PencilBeamAlgorithm")


class CollapsedConeAlgorithm(DoseCalculationAlgorithm):
    """Cài đặt mặc định cho thuật toán Collapsed Cone nếu không có sẵn."""

    def __init__(self):
        """Khởi tạo thuật toán Collapsed Cone."""
        super().__init__()
        logger.warning("Sử dụng phiên bản giả của CollapsedConeAlgorithm")


class ConvolutionAlgorithm(DoseCalculationAlgorithm):
    """Cài đặt mặc định cho thuật toán Convolution nếu không có sẵn."""

    def __init__(self):
        """Khởi tạo thuật toán Convolution."""
        super().__init__()
        logger.warning("Sử dụng phiên bản giả của ConvolutionAlgorithm")


class MonteCarloAlgorithm(DoseCalculationAlgorithm):
    """Cài đặt mặc định cho thuật toán Monte Carlo nếu không có sẵn."""

    def __init__(self):
        """Khởi tạo thuật toán Monte Carlo."""
        super().__init__()
        logger.warning("Sử dụng phiên bản giả của MonteCarloAlgorithm")


class AAAImplementer(DoseCalculationAlgorithm):
    """Cài đặt mặc định cho thuật toán AAA nếu không có sẵn."""

    def __init__(self):
        """Khởi tạo thuật toán AAA."""
        super().__init__()
        logger.warning("Sử dụng phiên bản giả của AAAImplementer")


class AcurosXBImplementer(DoseCalculationAlgorithm):
    """Cài đặt mặc định cho thuật toán Acuros XB nếu không có sẵn."""

    def __init__(self):
        """Khởi tạo thuật toán Acuros XB."""
        super().__init__()
        logger.warning("Sử dụng phiên bản giả của AcurosXBImplementer")


class MonteCarloGPUAlgorithm(DoseCalculationAlgorithm):
    """Cài đặt mặc định cho thuật toán Monte Carlo GPU nếu không có sẵn."""

    def __init__(self):
        """Khởi tạo thuật toán Monte Carlo GPU."""
        super().__init__()
        logger.warning("Sử dụng phiên bản giả của MonteCarloGPUAlgorithm")

    def get_algorithm_type(self) -> str:
        """Trả về loại thuật toán."""
        return "MONTE_CARLO_GPU"

    def get_display_name(self) -> str:
        """Trả về tên hiển thị của thuật toán."""
        return "Monte Carlo GPU"

    def get_description(self) -> str:
        """Trả về mô tả của thuật toán."""
        return "Thuật toán Monte Carlo tính toán trên GPU để tăng tốc độ."


# Import các thuật toán tính liều
algorithms = {}

# Thử import mỗi thuật toán và xử lý lỗi một cách riêng biệt
try:
    from quangtps.dose.algorithms.pencil_beam import (
        PencilBeamAlgorithm as ActualPencilBeam,
    )

    algorithms["PENCIL_BEAM"] = ActualPencilBeam
    logger.info("Đã import thuật toán PencilBeam thành công")
except ImportError as e:
    logger.warning(f"Không thể import PencilBeamAlgorithm: {str(e)}")
    algorithms["PENCIL_BEAM"] = PencilBeamAlgorithm

try:
    from quangtps.dose.algorithms.collapsed_cone import (
        CollapsedConeAlgorithm as ActualCollapsedCone,
    )

    algorithms["COLLAPSED_CONE"] = ActualCollapsedCone
    logger.info("Đã import thuật toán CollapsedCone thành công")
except ImportError as e:
    logger.warning(f"Không thể import CollapsedConeAlgorithm: {str(e)}")
    algorithms["COLLAPSED_CONE"] = CollapsedConeAlgorithm

try:
    from quangtps.dose.algorithms.convolution import (
        ConvolutionAlgorithm as ActualConvolution,
    )

    algorithms["CONVOLUTION"] = ActualConvolution
    logger.info("Đã import thuật toán Convolution thành công")
except ImportError as e:
    logger.warning(f"Không thể import ConvolutionAlgorithm: {str(e)}")
    algorithms["CONVOLUTION"] = ConvolutionAlgorithm

# Thử import các thuật toán nâng cao với xử lý ngoại lệ
try:
    from quangtps.dose.algorithms.monte_carlo import (
        MonteCarloAlgorithm as ActualMonteCarlo,
    )

    algorithms["MONTE_CARLO"] = ActualMonteCarlo
    logger.info("Đã import thuật toán MonteCarlo thành công")
except ImportError as e:
    logger.warning(f"Không thể import MonteCarloAlgorithm: {str(e)}")
    algorithms["MONTE_CARLO"] = MonteCarloAlgorithm

try:
    from quangtps.dose.algorithms.aaa import AAAImplementer as ActualAAA

    algorithms["AAA"] = ActualAAA
    logger.info("Đã import thuật toán AAA thành công")
except ImportError as e:
    logger.warning(f"Không thể import AAAImplementer: {str(e)}")
    algorithms["AAA"] = AAAImplementer

try:
    from quangtps.dose.algorithms.acuros import AcurosXBImplementer as ActualAcuros

    algorithms["ACUROS_XB"] = ActualAcuros
    logger.info("Đã import thuật toán Acuros XB thành công")
except ImportError as e:
    logger.warning(f"Không thể import AcurosXBImplementer: {str(e)}")
    algorithms["ACUROS_XB"] = AcurosXBImplementer

# Import thuật toán Monte Carlo GPU
try:
    from quangtps.dose.algorithms.improvements.monte_carlo_gpu import (
        MonteCarloGPUAlgorithm as ActualMonteCarloGPU,
    )

    algorithms["MONTE_CARLO_GPU"] = ActualMonteCarloGPU
    logger.info("Đã import thuật toán Monte Carlo GPU thành công")
except ImportError as e:
    logger.warning(f"Không thể import MonteCarloGPUAlgorithm: {str(e)}")
    algorithms["MONTE_CARLO_GPU"] = MonteCarloGPUAlgorithm


class DoseAlgorithmType(Enum):
    """Enum cho các loại thuật toán tính liều."""

    PENCIL_BEAM = auto()
    COLLAPSED_CONE = auto()
    CONVOLUTION = auto()
    MONTE_CARLO = auto()
    AAA = auto()
    ACUROS_XB = auto()
    MONTE_CARLO_GPU = auto()


# Đăng ký các thuật toán có sẵn
AVAILABLE_ALGORITHMS = {
    "pencil_beam": {
        "class": algorithms["PENCIL_BEAM"],
        "name": "Pencil Beam",
        "description": "Fast, simplified dose calculation using pencil beam kernels.",
        "category": "analytical",
    },
    "collapsed_cone": {
        "class": algorithms["COLLAPSED_CONE"],
        "name": "Collapsed Cone Convolution",
        "description": "Intermediate complexity algorithm with heterogeneity correction.",
        "category": "convolution",
    },
    "monte_carlo": {
        "class": algorithms["MONTE_CARLO"],
        "name": "Monte Carlo",
        "description": "High accuracy algorithm using particle simulation.",
        "category": "monte_carlo",
    },
    "convolution": {
        "class": algorithms["CONVOLUTION"],
        "name": "Convolution/Superposition",
        "description": "General convolution algorithm with energy deposition kernels.",
        "category": "convolution",
    },
    "aaa": {
        "class": algorithms["AAA"],
        "name": "Anisotropic Analytical Algorithm (AAA)",
        "description": "Varian AAA algorithm implementation.",
        "category": "analytical",
    },
    "acuros": {
        "class": algorithms["ACUROS_XB"],
        "name": "Acuros XB",
        "description": "Linear Boltzmann transport equation solver.",
        "category": "boltzmann",
    },
    "monte_carlo_gpu": {
        "class": algorithms["MONTE_CARLO_GPU"],
        "name": "Monte Carlo GPU",
        "description": "High performance Monte Carlo algorithm accelerated with GPU.",
        "category": "monte_carlo",
    },
}


def get_algorithm_instance(algorithm_id):
    """
    Get an instance of a dose calculation algorithm.

    Parameters
    ----------
    algorithm_id : str
        Identifier for the algorithm

    Returns
    -------
    DoseCalculationAlgorithm
        Instance of the requested algorithm

    Raises
    ------
    ValueError
        If the algorithm ID is not recognized
    """
    if algorithm_id not in AVAILABLE_ALGORITHMS:
        raise ValueError(f"Unknown algorithm ID: {algorithm_id}")

    algorithm_class = AVAILABLE_ALGORITHMS[algorithm_id]["class"]
    return algorithm_class()


def get_available_algorithms():
    """
    Get a list of available algorithms.

    Returns
    -------
    dict
        Dictionary of available algorithms with metadata
    """
    return AVAILABLE_ALGORITHMS


def get_algorithms_by_category(category):
    """
    Get algorithms filtered by category.

    Parameters
    ----------
    category : str
        Category to filter by

    Returns
    -------
    dict
        Dictionary of algorithms in the specified category
    """
    return {
        alg_id: alg_info
        for alg_id, alg_info in AVAILABLE_ALGORITHMS.items()
        if alg_info["category"] == category
    }


# Define version
__version__ = "1.2.0"
