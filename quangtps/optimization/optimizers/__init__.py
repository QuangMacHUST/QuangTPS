#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module quản lý các optimizer cho tối ưu hóa kế hoạch xạ trị.

Module này cung cấp các thuật toán tối ưu hóa khác nhau để tìm
kế hoạch xạ trị tối ưu cho bệnh nhân.
"""

import logging
from typing import Dict, Any, Optional, List
from enum import Enum

logger = logging.getLogger(__name__)


# Định nghĩa enum cơ bản
class OptimizerType(Enum):
    """Các loại optimizer có sẵn."""

    GRADIENT_DESCENT = "gradient_descent"
    SIMULATED_ANNEALING = "simulated_annealing"
    GENETIC_ALGORITHM = "genetic_algorithm"
    ADVANCED = "advanced"


# Base optimizer class
class BaseOptimizer:
    """Lớp cơ sở cho các optimizer."""

    def __init__(self, name: str = "BaseOptimizer"):
        self.name = name
        self.parameters = {}
        logger.info(f"Khởi tạo optimizer: {name}")

    def optimize(self, objectives, constraints=None):
        """Phương thức tối ưu hóa cơ bản."""
        logger.warning(f"Sử dụng base optimizer {self.name}")
        return None

    def set_parameters(self, parameters: Dict[str, Any]):
        """Thiết lập tham số optimizer."""
        self.parameters.update(parameters)
        logger.info(f"Cập nhật tham số cho {self.name}: {len(parameters)} tham số")


# Import safe với fallback
HAS_OPTIMIZER_FACTORY = False
HAS_GRADIENT_DESCENT = False
HAS_GENETIC_ALGORITHM = False
HAS_SIMULATED_ANNEALING = False
HAS_ADVANCED_OPTIMIZER = False

try:
    from .optimizer_factory import OptimizerFactory as _OptimizerFactory

    OptimizerFactory = _OptimizerFactory
    HAS_OPTIMIZER_FACTORY = True
    logger.info("Imported OptimizerFactory successfully")
except ImportError as e:
    logger.warning(f"Cannot import OptimizerFactory: {e}")

    # Fallback OptimizerFactory
    class OptimizerFactory:
        """Factory fallback cho tạo optimizer."""

        @staticmethod
        def create_optimizer(optimizer_type: OptimizerType, **kwargs):
            """Tạo optimizer dự phòng."""
            logger.warning(f"Tạo fallback optimizer: {optimizer_type.value}")
            return BaseOptimizer(optimizer_type.value)

        @staticmethod
        def get_available_optimizers():
            """Lấy danh sách optimizer có sẵn."""
            return [opt.value for opt in OptimizerType]


try:
    from .gradient_descent import GradientDescentOptimizer as _GradientDescentOptimizer

    GradientDescentOptimizer = _GradientDescentOptimizer
    HAS_GRADIENT_DESCENT = True
    logger.info("Imported GradientDescentOptimizer successfully")
except ImportError as e:
    logger.warning(f"Cannot import GradientDescentOptimizer: {e}")

    # Fallback class
    class GradientDescentOptimizer(BaseOptimizer):
        def __init__(self, **kwargs):
            super().__init__("GradientDescent")


try:
    from .genetic_algorithm import (
        GeneticAlgorithmOptimizer as _GeneticAlgorithmOptimizer,
    )

    GeneticAlgorithmOptimizer = _GeneticAlgorithmOptimizer
    HAS_GENETIC_ALGORITHM = True
    logger.info("Imported GeneticAlgorithmOptimizer successfully")
except ImportError as e:
    logger.warning(f"Cannot import GeneticAlgorithmOptimizer: {e}")

    # Fallback class
    class GeneticAlgorithmOptimizer(BaseOptimizer):
        def __init__(self, **kwargs):
            super().__init__("GeneticAlgorithm")


try:
    from .simulated_annealing import (
        SimulatedAnnealingOptimizer as _SimulatedAnnealingOptimizer,
    )

    SimulatedAnnealingOptimizer = _SimulatedAnnealingOptimizer
    HAS_SIMULATED_ANNEALING = True
    logger.info("Imported SimulatedAnnealingOptimizer successfully")
except ImportError as e:
    logger.warning(f"Cannot import SimulatedAnnealingOptimizer: {e}")

    # Fallback class
    class SimulatedAnnealingOptimizer(BaseOptimizer):
        def __init__(self, **kwargs):
            super().__init__("SimulatedAnnealing")


try:
    from .advanced_optimizer import AdvancedOptimizer as _AdvancedOptimizer

    AdvancedOptimizer = _AdvancedOptimizer
    HAS_ADVANCED_OPTIMIZER = True
    logger.info("Imported AdvancedOptimizer successfully")
except ImportError as e:
    logger.warning(f"Cannot import AdvancedOptimizer: {e}")

    # Fallback class
    class AdvancedOptimizer(BaseOptimizer):
        def __init__(self, **kwargs):
            super().__init__("Advanced")


# Helper functions
def create_optimizer(optimizer_type, **kwargs):
    """Helper function tạo optimizer."""
    try:
        return OptimizerFactory.create_optimizer(optimizer_type, **kwargs)
    except Exception as e:
        logger.error(f"Lỗi khi tạo optimizer {optimizer_type}: {e}")
        return BaseOptimizer(str(optimizer_type))


def get_default_optimizer():
    """Lấy optimizer mặc định."""
    try:
        return create_optimizer(OptimizerType.GRADIENT_DESCENT)
    except Exception as e:
        logger.error(f"Lỗi khi tạo default optimizer: {e}")
        return BaseOptimizer("default")


def list_available_optimizers():
    """Liệt kê các optimizer có sẵn."""
    try:
        if HAS_OPTIMIZER_FACTORY:
            return OptimizerFactory.get_available_optimizers()
        else:
            return [opt.value for opt in OptimizerType]
    except Exception as e:
        logger.error(f"Lỗi khi liệt kê optimizers: {e}")
        return [
            "gradient_descent",
            "simulated_annealing",
            "genetic_algorithm",
            "advanced",
        ]


def get_optimizer_status():
    """Lấy trạng thái các optimizer modules."""
    return {
        "optimizer_factory": HAS_OPTIMIZER_FACTORY,
        "gradient_descent": HAS_GRADIENT_DESCENT,
        "genetic_algorithm": HAS_GENETIC_ALGORITHM,
        "simulated_annealing": HAS_SIMULATED_ANNEALING,
        "advanced_optimizer": HAS_ADVANCED_OPTIMIZER,
    }


# Export list
__all__ = [
    "OptimizerFactory",
    "OptimizerType",
    "BaseOptimizer",
    "GradientDescentOptimizer",
    "GeneticAlgorithmOptimizer",
    "SimulatedAnnealingOptimizer",
    "AdvancedOptimizer",
    "create_optimizer",
    "get_default_optimizer",
    "list_available_optimizers",
    "get_optimizer_status",
]

logger.info(
    f"Module optimizers được khởi tạo thành công. Status: {get_optimizer_status()}"
)
