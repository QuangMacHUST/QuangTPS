#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Package báo cáo và xuất dữ liệu của QuangTPS.

Package này cung cấp các module để tạo báo cáo kế hoạch điều trị, xuất dữ liệu
ra các định dạng khác nhau như PDF, Excel, DICOM, và tích hợp với các hệ thống khác.
"""

try:
    from quangtps.reporting.report_generator import ReportGenerator
except ImportError as e:
    import logging
    logging.getLogger(__name__).error(f"Error importing ReportGenerator: {e}")
    
    # Create a placeholder class
    class ReportGenerator:
        def __init__(self, *args, **kwargs):
            pass
        
        def generate_report(self, *args, **kwargs):
            return None

try:
    from quangtps.reporting.excel_export import ExcelExporter
except ImportError as e:
    import logging
    logging.getLogger(__name__).error(f"Error importing ExcelExporter: {e}")
    
    # Create a placeholder class
    class ExcelExporter:
        def __init__(self, *args, **kwargs):
            pass
        
        def export(self, *args, **kwargs):
            return False

try:
    from quangtps.reporting.dicom_export import DicomExporter
except ImportError as e:
    import logging
    logging.getLogger(__name__).error(f"Error importing DicomExporter: {e}")
    
    # Create a placeholder class
    class DicomExporter:
        def __init__(self, *args, **kwargs):
            pass
        
        def export(self, *args, **kwargs):
            return False

try:
    from quangtps.reporting.template_manager import TemplateManager, ReportTemplate
except ImportError as e:
    import logging
    logging.getLogger(__name__).error(f"Error importing TemplateManager: {e}")
    
    # Create placeholder classes
    class ReportTemplate:
        def __init__(self, *args, **kwargs):
            pass
    
    class TemplateManager:
        def __init__(self, *args, **kwargs):
            pass
        
        def get_template(self, *args, **kwargs):
            return None
        
        def get_all_templates(self, *args, **kwargs):
            return []

# Define get_template_manager function here to fix import error
def get_template_manager():
    """
    Get or create a singleton instance of the TemplateManager.
    
    Returns:
        TemplateManager: The template manager instance
    """
    if not hasattr(get_template_manager, '_instance'):
        get_template_manager._instance = TemplateManager()
    return get_template_manager._instance

__all__ = [
    'ReportGenerator',
    'ExcelExporter',
    'DicomExporter',
    'get_template_manager',
    'TemplateManager',
    'ReportTemplate'
]

# Phiên bản module
__version__ = '0.1.0'
