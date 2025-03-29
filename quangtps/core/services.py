#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module quản lý các dịch vụ cốt lõi của QuangTPS.

Module này cung cấp quản lý trung tâm cho tất cả các dịch vụ hệ thống như 
xử lý DICOM, tính toán liều, truy xuất cơ sở dữ liệu, v.v.
"""

import logging
import importlib
from typing import Dict, Any, List, Optional, Type, Callable, Union
import sys
import os
import time

from quangtps.core.config import Config

logger = logging.getLogger(__name__)

class ServiceBase:
    """Lớp cơ sở cho tất cả các dịch vụ trong hệ thống."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Khởi tạo dịch vụ với cấu hình tùy chọn.
        
        Parameters
        ----------
        config : Optional[Dict[str, Any]]
            Cấu hình cho dịch vụ
        """
        self.config = config or {}
        self.initialized = False
        self.name = self.__class__.__name__
    
    def initialize(self) -> bool:
        """
        Khởi tạo dịch vụ.
        
        Returns
        -------
        bool
            True nếu khởi tạo thành công, False nếu không
        """
        self.initialized = True
        return True
    
    def shutdown(self) -> bool:
        """
        Tắt dịch vụ và giải phóng tài nguyên.
        
        Returns
        -------
        bool
            True nếu tắt thành công, False nếu không
        """
        self.initialized = False
        return True

class DatabaseService(ServiceBase):
    """Dịch vụ quản lý cơ sở dữ liệu."""
    
    def initialize(self) -> bool:
        """Khởi tạo kết nối cơ sở dữ liệu."""
        try:
            from quangtps.database.db_connector import DBConnector
            self.db = DBConnector.get_instance()
            logger.info("Dịch vụ cơ sở dữ liệu đã khởi tạo thành công")
            self.initialized = True
            return True
        except Exception as e:
            logger.error(f"Không thể khởi tạo dịch vụ cơ sở dữ liệu: {str(e)}")
            return False
    
    def shutdown(self) -> bool:
        """Đóng kết nối cơ sở dữ liệu."""
        try:
            if hasattr(self, 'db') and self.db:
                self.db.close()
            self.initialized = False
            logger.info("Dịch vụ cơ sở dữ liệu đã tắt thành công")
            return True
        except Exception as e:
            logger.error(f"Lỗi khi tắt dịch vụ cơ sở dữ liệu: {str(e)}")
            return False

class DicomService(ServiceBase):
    """Dịch vụ quản lý DICOM."""
    
    def initialize(self) -> bool:
        """Khởi tạo dịch vụ DICOM."""
        try:
            import pydicom
            self.lib = pydicom
            # Có thể thêm logic khởi tạo DICOM ở đây, như cấu hình thư mục lưu trữ
            logger.info("Dịch vụ DICOM đã khởi tạo thành công")
            self.initialized = True
            return True
        except ImportError:
            logger.error("Không thể tải thư viện pydicom. Vui lòng cài đặt: pip install pydicom")
            return False
        except Exception as e:
            logger.error(f"Không thể khởi tạo dịch vụ DICOM: {str(e)}")
            return False

class DoseCalculationService(ServiceBase):
    """Dịch vụ tính toán liều."""
    
    def initialize(self) -> bool:
        """Khởi tạo dịch vụ tính toán liều."""
        try:
            # Tải các thuật toán tính liều
            self.algorithms = {}
            
            # Đảm bảo rằng các thuật toán cơ bản khả dụng
            from quangtps.dose.algorithms import pencil_beam, collapsed_cone, monte_carlo
            
            self.register_algorithm("pencil_beam", pencil_beam.PencilBeamAlgorithm)
            self.register_algorithm("collapsed_cone", collapsed_cone.CollapsedConeAlgorithm)
            self.register_algorithm("monte_carlo", monte_carlo.MonteCarloAlgorithm)
            
            logger.info("Dịch vụ tính toán liều đã khởi tạo thành công")
            self.initialized = True
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
            from quangtps.optimization import gradient_descent, simulated_annealing, genetic_algorithm
            
            self.register_algorithm("gradient_descent", gradient_descent.GradientDescent)
            self.register_algorithm("simulated_annealing", simulated_annealing.SimulatedAnnealing)
            self.register_algorithm("genetic_algorithm", genetic_algorithm.GeneticAlgorithm)
            
            logger.info("Dịch vụ tối ưu hóa đã khởi tạo thành công")
            self.initialized = True
            return True
        except ImportError as e:
            logger.error(f"Không thể tải thuật toán tối ưu hóa: {str(e)}")
            # Tiếp tục mà không có tối ưu hóa
            self.initialized = True
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
            matplotlib.use('Qt5Agg')
            
            # Lưu trữ các đối tượng để sử dụng sau
            self.vtk = vtk
            self.matplotlib = matplotlib
            
            logger.info("Dịch vụ trực quan hóa đã khởi tạo thành công")
            self.initialized = True
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
            self.initialized = True
            return True
        except ImportError as e:
            logger.error(f"Không thể tải module báo cáo: {str(e)}")
            # Tiếp tục mà không có dịch vụ báo cáo
            self.initialized = True
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
        if hasattr(self, '_initialized') and self._initialized:
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
        self.register_service('database', DatabaseService())
        self.register_service('dicom', DicomService())
        self.register_service('dose_calculation', DoseCalculationService())
        self.register_service('optimization', OptimizationService())
        self.register_service('visualization', VisualizationService())
        self.register_service('reporting', ReportingService())
        
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
        if service and not service.initialized:
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
                if service.initialized:
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
    """
    Registry for services that provides a global access point for services 
    from any part of the application.
    """
    
    _services = {}
    
    @classmethod
    def register(cls, name, service):
        """
        Register a service with the given name.
        
        Parameters
        ----------
        name : str
            The name of the service
        service : Any
            The service instance
        """
        cls._services[name] = service
        logger.debug(f"Registered service: {name}")
    
    @classmethod
    def get(cls, name):
        """
        Get a service by name.
        
        Parameters
        ----------
        name : str
            The name of the service
            
        Returns
        -------
        Any
            The service instance if found, None otherwise
        """
        return cls._services.get(name)
    
    @classmethod
    def get_service(cls, service_name_or_class):
        """
        Get a service by its name or class. This method can be called with either
        a string name or a class type to provide backward compatibility.
        
        Parameters
        ----------
        service_name_or_class : Union[str, Type]
            The name or class of the service to retrieve
            
        Returns
        -------
        Any
            The service instance if found, None otherwise
        """
        # If service_name_or_class is a string, treat it as a name
        if isinstance(service_name_or_class, str):
            return cls.get(service_name_or_class)
        
        # Otherwise, treat it as a class type
        for service in cls._services.values():
            if isinstance(service, service_name_or_class):
                return service
        return None
    
    @classmethod
    def get_all(cls):
        """
        Get all registered services.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary of service names and instances
        """
        return cls._services
    
    @classmethod
    def remove(cls, name):
        """
        Remove a service by name.
        
        Parameters
        ----------
        name : str
            The name of the service
            
        Returns
        -------
        bool
            True if the service was removed, False if it wasn't found
        """
        if name in cls._services:
            del cls._services[name]
            logger.debug(f"Removed service: {name}")
            return True
        return False
    
    @classmethod
    def clear(cls):
        """Clear all registered services."""
        cls._services.clear()
        logger.debug("Cleared all services")
    
    @classmethod
    def register_service(cls, name, service):
        """
        Alias for register method to maintain backward compatibility.
        
        Parameters
        ----------
        name : str or Type
            The name or class of the service
        service : Any
            The service instance
        """
        if isinstance(name, type):
            # If a class is provided, use the class name
            name = name.__name__
        
        cls.register(name, service)
        logger.debug(f"Registered service with compatibility method: {name}") 