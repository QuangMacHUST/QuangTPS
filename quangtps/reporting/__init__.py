#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Package báo cáo và xuất dữ liệu của QuangTPS.

Package này cung cấp các module để tạo báo cáo kế hoạch điều trị, xuất dữ liệu
ra các định dạng khác nhau như PDF, Excel, DICOM, và tích hợp với các hệ thống khác.
"""

from quangtps.reporting.report_generator import ReportGenerator
from quangtps.reporting.excel_export import ExcelExporter
from quangtps.reporting.dicom_export import DicomExporter

__all__ = [
    'ReportGenerator',
    'ExcelExporter',
    'DicomExporter'
]

# Phiên bản module
__version__ = '0.1.0'
