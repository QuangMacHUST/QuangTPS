#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module for managing core services in QuangTPS.

This module provides the base classes and registry for services
that are used throughout the application.
"""

import logging
import importlib
from typing import Dict, Any, List, Optional, Type, Callable, Union, TypeVar, ClassVar
import sys
import os
import time

from quangtps.core.config import Config

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ServiceBase:
    """Base class for all services in the QuangTPS application."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the service with optional configuration.

        Parameters
        ----------
        config : Optional[Dict[str, Any]]
            Configuration for the service
        """
        self.config = config or {}
        self._initialized = False
        self.name = self.__class__.__name__

    def initialize(self) -> bool:
        """Initialize the service.

        Returns:
            bool: True if initialization was successful, False otherwise.
        """
        if self._initialized:
            logger.warning(f"{self.name} already initialized")
            return True

        logger.info(f"Initializing {self.name}")
        try:
            result = self._initialize()
            if result:
                self._initialized = True
                logger.info(f"{self.name} initialized successfully")
            else:
                logger.error(f"{self.name} initialization failed")
            return result
        except Exception as e:
            logger.error(f"Error initializing {self.name}: {str(e)}")
            return False

    def _initialize(self) -> bool:
        """Initialize the service (to be implemented by subclasses).

        Returns:
            bool: True if initialization was successful, False otherwise.
        """
        return True

    def shutdown(self) -> bool:
        """Shut down the service.

        Returns:
            bool: True if shutdown was successful, False otherwise.
        """
        if not self._initialized:
            logger.warning(f"{self.name} not initialized")
            return True

        logger.info(f"Shutting down {self.name}")
        try:
            result = self._shutdown()
            if result:
                self._initialized = False
                logger.info(f"{self.name} shut down successfully")
            else:
                logger.error(f"{self.name} shutdown failed")
            return result
        except Exception as e:
            logger.error(f"Error shutting down {self.name}: {str(e)}")
            return False

    def _shutdown(self) -> bool:
        """Shut down the service (to be implemented by subclasses).

        Returns:
            bool: True if shutdown was successful, False otherwise.
        """
        return True

    def is_initialized(self) -> bool:
        """Check if the service is initialized.

        Returns:
            bool: True if the service is initialized, False otherwise.
        """
        return self._initialized


class DatabaseService(ServiceBase):
    """Service for managing database connections."""

    def __init__(self):
        """Initialize the database service."""
        super().__init__()
        self._connection = None

    def _initialize(self) -> bool:
        """Initialize the database service.

        Returns:
            bool: True if initialization was successful, False otherwise.
        """
        logger.info("Initializing database service")
        try:
            # Example initialization code
            # This would typically connect to a database
            # and set up the connection
            # self._connection = ...
            return True
        except Exception as e:
            logger.error(f"Error initializing database service: {str(e)}")
            return False

    def _shutdown(self) -> bool:
        """Shut down the database service.

        Returns:
            bool: True if shutdown was successful, False otherwise.
        """
        logger.info("Shutting down database service")
        try:
            # Example shutdown code
            # This would typically close the database connection
            if self._connection is not None:
                # self._connection.close()
                self._connection = None
            return True
        except Exception as e:
            logger.error(f"Error shutting down database service: {str(e)}")
            return False

    def get_connection(self) -> Any:
        """Get the database connection.

        Returns:
            Any: Database connection.
        """
        return self._connection


class DicomService(ServiceBase):
    """Service for managing DICOM operations."""

    def __init__(self):
        """Initialize the DICOM service."""
        super().__init__()
        self._data_directory = None

    def _initialize(self) -> bool:
        """Initialize the DICOM service.

        Returns:
            bool: True if initialization was successful, False otherwise.
        """
        logger.info("Initializing DICOM service")
        try:
            # Example initialization code
            # This would typically set up the DICOM data directory
            # and load any necessary libraries
            # self._data_directory = ...
            return True
        except Exception as e:
            logger.error(f"Error initializing DICOM service: {str(e)}")
            return False

    def _shutdown(self) -> bool:
        """Shut down the DICOM service.

        Returns:
            bool: True if shutdown was successful, False otherwise.
        """
        logger.info("Shutting down DICOM service")
        try:
            # Example shutdown code
            # This would typically clean up any resources
            self._data_directory = None
            return True
        except Exception as e:
            logger.error(f"Error shutting down DICOM service: {str(e)}")
            return False

    def set_data_directory(self, directory: str) -> None:
        """Set the DICOM data directory.

        Args:
            directory (str): Directory path.
        """
        self._data_directory = directory

    def get_data_directory(self) -> Optional[str]:
        """Get the DICOM data directory.

        Returns:
            Optional[str]: Directory path or None if not set.
        """
        return self._data_directory


class DoseCalculationService(ServiceBase):
    """Dịch vụ tính toán liều."""

    def initialize(self) -> bool:
        """Khởi tạo dịch vụ tính toán liều."""
        try:
            # Tải các thuật toán tính liều
            self.algorithms = {}

            # Đảm bảo rằng các thuật toán cơ bản khả dụng
            from quangtps.dose.algorithms import (
                pencil_beam,
                collapsed_cone,
                monte_carlo,
            )

            self.register_algorithm("pencil_beam", pencil_beam.PencilBeamAlgorithm)
            self.register_algorithm(
                "collapsed_cone", collapsed_cone.CollapsedConeAlgorithm
            )
            self.register_algorithm("monte_carlo", monte_carlo.MonteCarloAlgorithm)

            logger.info("Dịch vụ tính toán liều đã khởi tạo thành công")
            self._initialized = True
            return True
        except ImportError as e:
            logger.error(f"Không thể tải thuật toán tính liều: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Không thể khởi tạo dịch vụ tính toán liều: {str(e)}")
            return False

    def register_algorithm(self, name: str, algorithm_class: Type) -> bool:
        """
        Đăng ký thuật toán tính liều mới.

        Parameters
        ----------
        name : str
            Tên của thuật toán
        algorithm_class : Type
            Lớp triển khai thuật toán

        Returns
        -------
        bool
            True nếu đăng ký thành công, False nếu không
        """
        try:
            self.algorithms[name] = algorithm_class
            logger.debug(f"Đã đăng ký thuật toán tính liều: {name}")
            return True
        except Exception as e:
            logger.error(f"Không thể đăng ký thuật toán tính liều '{name}': {str(e)}")
            return False

    def get_algorithm(self, name: str) -> Optional[Type]:
        """
        Lấy lớp thuật toán tính liều theo tên.

        Parameters
        ----------
        name : str
            Tên của thuật toán

        Returns
        -------
        Optional[Type]
            Lớp thuật toán nếu tìm thấy, None nếu không
        """
        return self.algorithms.get(name)

    def get_available_algorithms(self) -> List[str]:
        """
        Trả về danh sách các thuật toán khả dụng.

        Returns
        -------
        List[str]
            Danh sách tên các thuật toán khả dụng
        """
        return list(self.algorithms.keys())


class OptimizationService(ServiceBase):
    """Dịch vụ tối ưu hóa kế hoạch."""

    def initialize(self) -> bool:
        """Khởi tạo dịch vụ tối ưu hóa."""
        try:
            # Tải các thuật toán tối ưu hóa
            self.algorithms = {}

            # Import các thuật toán tối ưu hóa
            from quangtps.optimization import (
                gradient_descent,
                simulated_annealing,
                genetic_algorithm,
            )

            self.register_algorithm(
                "gradient_descent", gradient_descent.GradientDescent
            )
            self.register_algorithm(
                "simulated_annealing", simulated_annealing.SimulatedAnnealing
            )
            self.register_algorithm(
                "genetic_algorithm", genetic_algorithm.GeneticAlgorithm
            )

            logger.info("Dịch vụ tối ưu hóa đã khởi tạo thành công")
            self._initialized = True
            return True
        except ImportError as e:
            logger.error(f"Không thể tải thuật toán tối ưu hóa: {str(e)}")
            # Tiếp tục mà không có tối ưu hóa
            self._initialized = True
            return True
        except Exception as e:
            logger.error(f"Không thể khởi tạo dịch vụ tối ưu hóa: {str(e)}")
            return False

    def register_algorithm(self, name: str, algorithm_class: Type) -> bool:
        """
        Đăng ký thuật toán tối ưu hóa mới.

        Parameters
        ----------
        name : str
            Tên của thuật toán
        algorithm_class : Type
            Lớp triển khai thuật toán

        Returns
        -------
        bool
            True nếu đăng ký thành công, False nếu không
        """
        try:
            self.algorithms[name] = algorithm_class
            logger.debug(f"Đã đăng ký thuật toán tối ưu hóa: {name}")
            return True
        except Exception as e:
            logger.error(f"Không thể đăng ký thuật toán tối ưu hóa '{name}': {str(e)}")
            return False

    def get_algorithm(self, name: str) -> Optional[Type]:
        """
        Lấy lớp thuật toán tối ưu hóa theo tên.

        Parameters
        ----------
        name : str
            Tên của thuật toán

        Returns
        -------
        Optional[Type]
            Lớp thuật toán nếu tìm thấy, None nếu không
        """
        return self.algorithms.get(name)

    def get_available_algorithms(self) -> List[str]:
        """
        Trả về danh sách các thuật toán tối ưu hóa khả dụng.

        Returns
        -------
        List[str]
            Danh sách tên các thuật toán khả dụng
        """
        return list(self.algorithms.keys())


class VisualizationService(ServiceBase):
    """Dịch vụ trực quan hóa."""

    def initialize(self) -> bool:
        """Khởi tạo dịch vụ trực quan hóa."""
        try:
            # Import các thư viện cần thiết
            import vtk
            import matplotlib

            # Đặt backend cho matplotlib
            matplotlib.use("Qt5Agg")

            # Lưu trữ các đối tượng để sử dụng sau
            self.vtk = vtk
            self.matplotlib = matplotlib

            logger.info("Dịch vụ trực quan hóa đã khởi tạo thành công")
            self._initialized = True
            return True
        except ImportError as e:
            logger.error(f"Không thể tải thư viện trực quan hóa: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Không thể khởi tạo dịch vụ trực quan hóa: {str(e)}")
            return False


class ReportingService(ServiceBase):
    """Dịch vụ báo cáo."""

    def initialize(self) -> bool:
        """Khởi tạo dịch vụ báo cáo."""
        try:
            # Import các module báo cáo
            from quangtps.reporting import report_generator, template_manager

            self.report_generator = report_generator.ReportGenerator()
            self.template_manager = template_manager.TemplateManager()

            # Tải các template mặc định
            self.template_manager.load_default_templates()

            logger.info("Dịch vụ báo cáo đã khởi tạo thành công")
            self._initialized = True
            return True
        except ImportError as e:
            logger.error(f"Không thể tải module báo cáo: {str(e)}")
            # Tiếp tục mà không có dịch vụ báo cáo
            self._initialized = True
            return True
        except Exception as e:
            logger.error(f"Không thể khởi tạo dịch vụ báo cáo: {str(e)}")
            return False


class ServiceManager:
    """Quản lý tất cả các dịch vụ trong hệ thống."""

    _instance = None

    def __new__(cls):
        """Tạo một thể hiện duy nhất của lớp (Singleton)."""
        if cls._instance is None:
            cls._instance = super(ServiceManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    @classmethod
    def get_instance(cls):
        """Trả về instance duy nhất của ServiceManager."""
        if cls._instance is None:
            return cls()
        return cls._instance

    def __init__(self):
        """Khởi tạo quản lý dịch vụ."""
        if hasattr(self, "_initialized") and self._initialized:
            return

        self._initialized = True
        self.services = {}
        self.config = Config.get_instance()

    def initialize_services(self) -> bool:
        """
        Khởi tạo tất cả các dịch vụ hệ thống.

        Returns
        -------
        bool
            True nếu tất cả các dịch vụ khởi tạo thành công, False nếu có ít nhất một dịch vụ thất bại
        """
        logger.info("Bắt đầu khởi tạo các dịch vụ hệ thống...")

        # Đăng ký các dịch vụ cốt lõi
        self.register_service("database", DatabaseService())
        self.register_service("dicom", DicomService())
        self.register_service("dose_calculation", DoseCalculationService())
        self.register_service("optimization", OptimizationService())
        self.register_service("visualization", VisualizationService())
        self.register_service("reporting", ReportingService())

        # Khởi tạo tất cả các dịch vụ đã đăng ký
        start_time = time.time()
        success = True

        for name, service in self.services.items():
            try:
                logger.info(f"Đang khởi tạo dịch vụ '{name}'...")
                if not service.initialize():
                    logger.warning(f"Dịch vụ '{name}' không khởi tạo thành công")
                    success = False
            except Exception as e:
                logger.error(f"Lỗi khi khởi tạo dịch vụ '{name}': {str(e)}")
                success = False

        elapsed_time = time.time() - start_time
        logger.info(f"Khởi tạo dịch vụ hoàn tất sau {elapsed_time:.2f} giây")

        return success

    def register_service(self, name: str, service: ServiceBase) -> bool:
        """
        Đăng ký một dịch vụ mới.

        Parameters
        ----------
        name : str
            Tên của dịch vụ
        service : ServiceBase
            Đối tượng dịch vụ cần đăng ký

        Returns
        -------
        bool
            True nếu đăng ký thành công, False nếu không
        """
        if name in self.services:
            logger.warning(f"Dịch vụ '{name}' đã tồn tại và sẽ bị ghi đè")

        self.services[name] = service
        logger.debug(f"Đã đăng ký dịch vụ: {name}")
        return True

    def get_service(self, name: str) -> Optional[ServiceBase]:
        """
        Lấy dịch vụ theo tên.

        Parameters
        ----------
        name : str
            Tên của dịch vụ

        Returns
        -------
        Optional[ServiceBase]
            Đối tượng dịch vụ nếu tìm thấy, None nếu không
        """
        service = self.services.get(name)

        # Khởi tạo dịch vụ nếu chưa được khởi tạo
        if service and not service._initialized:
            service.initialize()

        return service

    def shutdown_services(self) -> bool:
        """
        Tắt tất cả các dịch vụ.

        Returns
        -------
        bool
            True nếu tất cả các dịch vụ tắt thành công, False nếu có ít nhất một dịch vụ thất bại
        """
        logger.info("Bắt đầu tắt các dịch vụ hệ thống...")

        success = True

        # Duyệt qua các dịch vụ theo thứ tự ngược để tắt an toàn
        service_names = list(self.services.keys())
        service_names.reverse()

        for name in service_names:
            service = self.services[name]
            try:
                if service._initialized:
                    logger.info(f"Đang tắt dịch vụ '{name}'...")
                    if not service.shutdown():
                        logger.warning(f"Dịch vụ '{name}' không tắt thành công")
                        success = False
            except Exception as e:
                logger.error(f"Lỗi khi tắt dịch vụ '{name}': {str(e)}")
                success = False

        logger.info("Tắt dịch vụ hoàn tất")

        return success

    def get_available_services(self) -> List[str]:
        """
        Trả về danh sách các dịch vụ đã đăng ký.

        Returns
        -------
        List[str]
            Danh sách tên các dịch vụ
        """
        return list(self.services.keys())


class ServiceRegistry:
    """Registry for services in the QuangTPS application.

    This class is implemented as a singleton to ensure that
    only one instance of the registry exists in the application.
    """

    _instance: ClassVar[Optional["ServiceRegistry"]] = None
    _services: Dict[str, Any]

    def __init__(self):
        """Initialize the service registry."""
        if ServiceRegistry._instance is not None:
            raise RuntimeError(
                "ServiceRegistry is a singleton and should not be instantiated directly"
            )
        self._services = {}

    @classmethod
    def get_instance(cls) -> "ServiceRegistry":
        """Get the singleton instance of the service registry.

        Returns:
            ServiceRegistry: The singleton instance.
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def register(self, name: str, service: Any) -> None:
        """Register a service with the registry.

        Args:
            name (str): Name of the service.
            service (Any): Service instance.
        """
        if name in self._services:
            logger.warning(f"Service {name} already registered, replacing")
        self._services[name] = service
        logger.info(f"Registered service {name}")

    def register_service(self, name: str, service: Any) -> None:
        """Register a service with the registry.

        This method is an alias for register() to ensure compatibility.

        Args:
            name (str): Name of the service.
            service (Any): Service instance.
        """
        self.register(name, service)

    def unregister(self, name: str) -> bool:
        """Unregister a service from the registry.

        Args:
            name (str): Name of the service.

        Returns:
            bool: True if the service was unregistered, False otherwise.
        """
        if name not in self._services:
            logger.warning(f"Service {name} not registered")
            return False
        del self._services[name]
        logger.info(f"Unregistered service {name}")
        return True

    def get(self, service_name_or_class: Union[str, Type[T]]) -> Optional[T]:
        """Get a service by name or class.

        Args:
            service_name_or_class (Union[str, Type[T]]): Name or class of the service.

        Returns:
            Optional[T]: Service instance or None if not found.
        """
        if isinstance(service_name_or_class, str):
            return self._services.get(service_name_or_class)

        # If service_name_or_class is a class, find the first instance of that class
        for service in self._services.values():
            if isinstance(service, service_name_or_class):
                return service

        logger.warning(f"Service {service_name_or_class} not found")
        return None

    def get_service(self, service_name_or_class: Union[str, Type[T]]) -> Optional[T]:
        """Get a service by name or class.

        This method is an alias for get() to ensure compatibility.

        Args:
            service_name_or_class (Union[str, Type[T]]): Name or class of the service.

        Returns:
            Optional[T]: Service instance or None if not found.
        """
        return self.get(service_name_or_class)

    @classmethod
    def get_service_instance(
        cls, service_name_or_class: Union[str, Type[T]]
    ) -> Optional[T]:
        """Class method to get a service by name or class.

        This method is a convenience method that gets the singleton instance
        of the registry and then calls get() on it.

        Args:
            service_name_or_class (Union[str, Type[T]]): Name or class of the service.

        Returns:
            Optional[T]: Service instance or None if not found.
        """
        return cls.get_instance().get(service_name_or_class)

    def initialize_all(self) -> bool:
        """Initialize all registered services.

        Returns:
            bool: True if all services were initialized successfully, False otherwise.
        """
        logger.info("Initializing all services")
        all_success = True
        for name, service in self._services.items():
            if hasattr(service, "initialize"):
                success = service.initialize()
                if not success:
                    logger.error(f"Failed to initialize service {name}")
                    all_success = False
        return all_success

    def shutdown_all(self) -> bool:
        """Shut down all registered services.

        Returns:
            bool: True if all services were shut down successfully, False otherwise.
        """
        logger.info("Shutting down all services")
        all_success = True
        for name, service in self._services.items():
            if hasattr(service, "shutdown"):
                success = service.shutdown()
                if not success:
                    logger.error(f"Failed to shut down service {name}")
                    all_success = False
        return all_success

    def get_all_services(self) -> Dict[str, Any]:
        """Get all registered services.

        Returns:
            Dict[str, Any]: Dictionary of all registered services.
        """
        return self._services.copy()

    def clear(self) -> None:
        """Clear all registered services."""
        logger.info("Clearing all services")
        self._services.clear()

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance of the registry."""
        if cls._instance is not None:
            cls._instance.clear()
            cls._instance = None
        logger.info("Reset service registry")


# Create a singleton instance of the service registry
service_registry = ServiceRegistry.get_instance()

# Create a singleton instance of the service manager
service_manager = ServiceManager.get_instance()
