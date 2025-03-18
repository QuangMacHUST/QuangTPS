#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module quản lý các đường dẫn trong QuangTPS.

Module này cung cấp các hàm tiện ích để lấy các đường dẫn
mặc định và thư mục lưu trữ dữ liệu ứng dụng.
"""

import os
import sys
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Tên thư mục lưu trữ dữ liệu ứng dụng
APP_DATA_DIR_NAME = "QuangTPS"


def get_app_data_dir() -> str:
    """
    Lấy đường dẫn đến thư mục dữ liệu ứng dụng.
    
    Thư mục này được sử dụng để lưu trữ các tệp tin dữ liệu ứng dụng
    như cấu hình, mô hình, log, và các dữ liệu tạm thời.
    
    Returns
    -------
    str
        Đường dẫn đến thư mục dữ liệu ứng dụng
    """
    # Xác định thư mục dữ liệu người dùng dựa trên hệ điều hành
    if os.name == 'nt':  # Windows
        app_data = os.environ.get('APPDATA', os.path.expanduser('~'))
        app_dir = os.path.join(app_data, APP_DATA_DIR_NAME)
    elif sys.platform == 'darwin':  # macOS
        app_data = os.path.expanduser('~/Library/Application Support')
        app_dir = os.path.join(app_data, APP_DATA_DIR_NAME)
    else:  # Linux và các hệ điều hành khác
        app_data = os.environ.get('XDG_DATA_HOME', os.path.expanduser('~/.local/share'))
        app_dir = os.path.join(app_data, APP_DATA_DIR_NAME)
    
    # Đảm bảo thư mục tồn tại
    os.makedirs(app_dir, exist_ok=True)
    
    logger.debug(f"App data directory: {app_dir}")
    return app_dir


def get_app_config_dir() -> str:
    """
    Lấy đường dẫn đến thư mục cấu hình ứng dụng.
    
    Returns
    -------
    str
        Đường dẫn đến thư mục cấu hình ứng dụng
    """
    config_dir = os.path.join(get_app_data_dir(), "config")
    os.makedirs(config_dir, exist_ok=True)
    return config_dir


def get_models_dir() -> str:
    """
    Lấy đường dẫn đến thư mục lưu trữ mô hình.
    
    Returns
    -------
    str
        Đường dẫn đến thư mục lưu trữ mô hình
    """
    models_dir = os.path.join(get_app_data_dir(), "models")
    os.makedirs(models_dir, exist_ok=True)
    return models_dir


def get_temp_dir() -> str:
    """
    Lấy đường dẫn đến thư mục lưu trữ tệp tin tạm thời.
    
    Returns
    -------
    str
        Đường dẫn đến thư mục lưu trữ tệp tin tạm thời
    """
    temp_dir = os.path.join(get_app_data_dir(), "temp")
    os.makedirs(temp_dir, exist_ok=True)
    return temp_dir


def get_logs_dir() -> str:
    """
    Lấy đường dẫn đến thư mục lưu trữ tệp tin log.
    
    Returns
    -------
    str
        Đường dẫn đến thư mục lưu trữ tệp tin log
    """
    logs_dir = os.path.join(get_app_data_dir(), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    return logs_dir
