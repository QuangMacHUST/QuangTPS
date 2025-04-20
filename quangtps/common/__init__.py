#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Common utilities and tools for the QuangTPS application.

This package contains common utilities, services, and helper functions
used throughout the application.
"""

import logging

# Import and expose the ServiceRegistry
try:
    from .services import ServiceRegistry, ServiceBase, service_registry, PatientService
except ImportError:
    # Import from core services if common services are not available
    try:
        from quangtps.core.services import ServiceRegistry, ServiceBase

        service_registry = ServiceRegistry.get_instance()

        class PatientService(ServiceBase):
            pass
    except ImportError:
        # Placeholders in case neither module is available
        class ServiceBase:
            pass

        class ServiceRegistry:
            @classmethod
            def get_instance(cls):
                return cls()

            def get_service(self, service_name):
                return None

        class PatientService(ServiceBase):
            pass

        service_registry = ServiceRegistry.get_instance()

# Expose other common utilities
try:
    from .paths import get_base_path, get_data_path, get_config_path
except ImportError:
    # Placeholders for path utilities
    def get_base_path():
        import os

        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def get_data_path():
        import os

        return os.path.join(get_base_path(), "data")

    def get_config_path():
        import os

        return os.path.join(get_base_path(), "config")


logger = logging.getLogger(__name__)
