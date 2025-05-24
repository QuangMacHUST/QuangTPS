#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Optimizer Factory Module

Module này cung cấp factory pattern để tạo và quản lý các optimizer
khác nhau trong hệ thống QuangTPS.
"""

import logging
from typing import Dict, Type, Optional, Any, List
from enum import Enum

logger = logging.getLogger(__name__)


class OptimizerType(Enum):
    """Các loại optimizer có sẵn."""

    GRADIENT_DESCENT = "gradient_descent"
    SIMULATED_ANNEALING = "simulated_annealing"
    GENETIC_ALGORITHM = "genetic_algorithm"
    PARTICLE_SWARM = "particle_swarm"
    NEWTON_METHOD = "newton_method"
    QUASI_NEWTON = "quasi_newton"
    SEQUENTIAL_QUADRATIC = "sequential_quadratic"
    INTERIOR_POINT = "interior_point"
    TRUST_REGION = "trust_region"
    LEVENBERG_MARQUARDT = "levenberg_marquardt"


class BaseOptimizer:
    """
    Lớp cơ sở cho tất cả các optimizer.
    """

    def __init__(self, name: str = "BaseOptimizer"):
        """
        Khởi tạo optimizer cơ sở.

        Parameters
        ----------
        name : str
            Tên của optimizer
        """
        self.name = name
        self.is_initialized = False
        self.parameters = {}
        self.convergence_history = []
        self.current_iteration = 0
        self.max_iterations = 100
        self.tolerance = 1e-6

        logger.info(f"Khởi tạo {self.name} optimizer")

    def initialize(self, **kwargs) -> bool:
        """
        Khởi tạo optimizer với các tham số.

        Parameters
        ----------
        **kwargs
            Các tham số khởi tạo

        Returns
        -------
        bool
            True nếu khởi tạo thành công
        """
        try:
            self.parameters.update(kwargs)
            self.max_iterations = kwargs.get("max_iterations", 100)
            self.tolerance = kwargs.get("tolerance", 1e-6)
            self.is_initialized = True

            logger.info(f"Khởi tạo thành công {self.name} với {len(kwargs)} tham số")
            return True

        except Exception as e:
            logger.error(f"Lỗi khởi tạo {self.name}: {e}")
            return False

    def optimize(self, objective_function, initial_guess, **kwargs) -> Dict[str, Any]:
        """
        Thực hiện tối ưu hóa.

        Parameters
        ----------
        objective_function : callable
            Hàm mục tiêu cần tối ưu hóa
        initial_guess : array-like
            Giá trị khởi tạo
        **kwargs
            Các tham số bổ sung

        Returns
        -------
        Dict[str, Any]
            Kết quả tối ưu hóa
        """
        if not self.is_initialized:
            logger.warning(f"{self.name} chưa được khởi tạo. Sử dụng tham số mặc định.")
            self.initialize()

        try:
            # Giả lập quá trình tối ưu hóa
            self.current_iteration = 0
            self.convergence_history = []

            best_solution = initial_guess
            best_value = objective_function(initial_guess)

            for iteration in range(self.max_iterations):
                self.current_iteration = iteration + 1

                # Giả lập cập nhật solution
                current_value = best_value * (1 - 0.01 * iteration)
                self.convergence_history.append(current_value)

                # Kiểm tra hội tụ
                if iteration > 0:
                    improvement = abs(self.convergence_history[-2] - current_value)
                    if improvement < self.tolerance:
                        logger.info(f"{self.name} hội tụ tại iteration {iteration + 1}")
                        break

                best_value = current_value

            result = {
                "success": True,
                "x": best_solution,
                "fun": best_value,
                "nit": self.current_iteration,
                "convergence_history": self.convergence_history,
                "message": f"Optimization completed with {self.name}",
            }

            logger.info(
                f"{self.name} hoàn thành sau {self.current_iteration} iterations"
            )
            return result

        except Exception as e:
            logger.error(f"Lỗi trong quá trình tối ưu hóa {self.name}: {e}")
            return {
                "success": False,
                "message": f"Optimization failed: {str(e)}",
                "x": initial_guess,
                "fun": float("inf"),
                "nit": self.current_iteration,
            }

    def get_status(self) -> Dict[str, Any]:
        """
        Lấy trạng thái hiện tại của optimizer.

        Returns
        -------
        Dict[str, Any]
            Thông tin trạng thái
        """
        return {
            "name": self.name,
            "is_initialized": self.is_initialized,
            "current_iteration": self.current_iteration,
            "max_iterations": self.max_iterations,
            "tolerance": self.tolerance,
            "parameters": self.parameters.copy(),
        }


class GradientDescentOptimizer(BaseOptimizer):
    """Optimizer sử dụng thuật toán Gradient Descent."""

    def __init__(self):
        super().__init__("Gradient Descent")
        self.learning_rate = 0.01
        self.momentum = 0.0

    def initialize(self, **kwargs) -> bool:
        """Khởi tạo với tham số riêng cho Gradient Descent."""
        self.learning_rate = kwargs.get("learning_rate", 0.01)
        self.momentum = kwargs.get("momentum", 0.0)
        return super().initialize(**kwargs)


class SimulatedAnnealingOptimizer(BaseOptimizer):
    """Optimizer sử dụng thuật toán Simulated Annealing."""

    def __init__(self):
        super().__init__("Simulated Annealing")
        self.initial_temperature = 1000.0
        self.cooling_rate = 0.95

    def initialize(self, **kwargs) -> bool:
        """Khởi tạo với tham số riêng cho Simulated Annealing."""
        self.initial_temperature = kwargs.get("initial_temperature", 1000.0)
        self.cooling_rate = kwargs.get("cooling_rate", 0.95)
        return super().initialize(**kwargs)


class GeneticAlgorithmOptimizer(BaseOptimizer):
    """Optimizer sử dụng thuật toán di truyền."""

    def __init__(self):
        super().__init__("Genetic Algorithm")
        self.population_size = 50
        self.mutation_rate = 0.01
        self.crossover_rate = 0.8

    def initialize(self, **kwargs) -> bool:
        """Khởi tạo với tham số riêng cho Genetic Algorithm."""
        self.population_size = kwargs.get("population_size", 50)
        self.mutation_rate = kwargs.get("mutation_rate", 0.01)
        self.crossover_rate = kwargs.get("crossover_rate", 0.8)
        return super().initialize(**kwargs)


class ParticleSwarmOptimizer(BaseOptimizer):
    """Optimizer sử dụng thuật toán Particle Swarm."""

    def __init__(self):
        super().__init__("Particle Swarm")
        self.swarm_size = 30
        self.inertia_weight = 0.9
        self.cognitive_weight = 2.0
        self.social_weight = 2.0

    def initialize(self, **kwargs) -> bool:
        """Khởi tạo với tham số riêng cho Particle Swarm."""
        self.swarm_size = kwargs.get("swarm_size", 30)
        self.inertia_weight = kwargs.get("inertia_weight", 0.9)
        self.cognitive_weight = kwargs.get("cognitive_weight", 2.0)
        self.social_weight = kwargs.get("social_weight", 2.0)
        return super().initialize(**kwargs)


class OptimizerFactory:
    """
    Factory class để tạo các optimizer.
    """

    _optimizers: Dict[OptimizerType, Type[BaseOptimizer]] = {
        OptimizerType.GRADIENT_DESCENT: GradientDescentOptimizer,
        OptimizerType.SIMULATED_ANNEALING: SimulatedAnnealingOptimizer,
        OptimizerType.GENETIC_ALGORITHM: GeneticAlgorithmOptimizer,
        OptimizerType.PARTICLE_SWARM: ParticleSwarmOptimizer,
    }

    @classmethod
    def create_optimizer(
        cls, optimizer_type: OptimizerType, **kwargs
    ) -> Optional[BaseOptimizer]:
        """
        Tạo optimizer theo loại được chỉ định.

        Parameters
        ----------
        optimizer_type : OptimizerType
            Loại optimizer cần tạo
        **kwargs
            Các tham số khởi tạo

        Returns
        -------
        Optional[BaseOptimizer]
            Instance của optimizer hoặc None nếu lỗi
        """
        try:
            if optimizer_type not in cls._optimizers:
                logger.error(f"Không hỗ trợ optimizer type: {optimizer_type}")
                return None

            optimizer_class = cls._optimizers[optimizer_type]
            optimizer = optimizer_class()

            if kwargs:
                optimizer.initialize(**kwargs)

            logger.info(f"Tạo thành công optimizer: {optimizer.name}")
            return optimizer

        except Exception as e:
            logger.error(f"Lỗi tạo optimizer {optimizer_type}: {e}")
            return None

    @classmethod
    def get_available_optimizers(cls) -> List[OptimizerType]:
        """
        Lấy danh sách các optimizer có sẵn.

        Returns
        -------
        List[OptimizerType]
            Danh sách các optimizer type
        """
        return list(cls._optimizers.keys())

    @classmethod
    def register_optimizer(
        cls, optimizer_type: OptimizerType, optimizer_class: Type[BaseOptimizer]
    ) -> bool:
        """
        Đăng ký optimizer mới.

        Parameters
        ----------
        optimizer_type : OptimizerType
            Loại optimizer
        optimizer_class : Type[BaseOptimizer]
            Class của optimizer

        Returns
        -------
        bool
            True nếu đăng ký thành công
        """
        try:
            if not issubclass(optimizer_class, BaseOptimizer):
                logger.error(f"Optimizer class phải kế thừa từ BaseOptimizer")
                return False

            cls._optimizers[optimizer_type] = optimizer_class
            logger.info(f"Đăng ký thành công optimizer: {optimizer_type}")
            return True

        except Exception as e:
            logger.error(f"Lỗi đăng ký optimizer: {e}")
            return False


# Convenience functions
def create_optimizer(optimizer_type: str, **kwargs) -> Optional[BaseOptimizer]:
    """
    Tạo optimizer từ string type.

    Parameters
    ----------
    optimizer_type : str
        Tên loại optimizer
    **kwargs
        Các tham số khởi tạo

    Returns
    -------
    Optional[BaseOptimizer]
        Instance của optimizer
    """
    try:
        # Chuyển đổi string thành OptimizerType
        if isinstance(optimizer_type, str):
            optimizer_type = optimizer_type.lower()
            for opt_type in OptimizerType:
                if opt_type.value == optimizer_type:
                    return OptimizerFactory.create_optimizer(opt_type, **kwargs)

        logger.error(f"Không tìm thấy optimizer type: {optimizer_type}")
        return None

    except Exception as e:
        logger.error(f"Lỗi tạo optimizer từ string: {e}")
        return None


def get_default_optimizer(**kwargs) -> BaseOptimizer:
    """
    Lấy optimizer mặc định.

    Parameters
    ----------
    **kwargs
        Các tham số khởi tạo

    Returns
    -------
    BaseOptimizer
        Optimizer mặc định (Gradient Descent)
    """
    optimizer = OptimizerFactory.create_optimizer(
        OptimizerType.GRADIENT_DESCENT, **kwargs
    )

    if optimizer is None:
        # Fallback to base optimizer
        optimizer = BaseOptimizer("Default Optimizer")
        optimizer.initialize(**kwargs)

    return optimizer


def list_available_optimizers() -> List[str]:
    """
    Lấy danh sách tên các optimizer có sẵn.

    Returns
    -------
    List[str]
        Danh sách tên optimizer
    """
    return [opt_type.value for opt_type in OptimizerFactory.get_available_optimizers()]
