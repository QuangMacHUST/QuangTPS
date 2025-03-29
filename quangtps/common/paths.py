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


def get_project_root() -> str:
    """
    Lấy đường dẫn đến thư mục gốc của dự án.
    
    Returns
    -------
    str
        Đường dẫn đến thư mục gốc của dự án
    """
    # Xác định thư mục gốc của dự án
    # Module này nằm trong quangtps/common, nên cần đi lên 2 cấp
    file_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(file_path)))
    return project_root


def get_beam_data_dir() -> str:
    """
    Lấy đường dẫn đến thư mục lưu trữ dữ liệu chùm tia.
    
    Returns
    -------
    str
        Đường dẫn đến thư mục lưu trữ dữ liệu chùm tia
    """
    # Dữ liệu chùm tia được lưu trữ trong thư mục data/beam_data
    beam_data_dir = os.path.join(get_project_root(), "data", "beam_data")
    os.makedirs(beam_data_dir, exist_ok=True)
    return beam_data_dir


def get_icon_path(icon_name: str) -> str:
    """
    Lấy đường dẫn đến một biểu tượng trong thư mục icons.
    
    Parameters
    ----------
    icon_name : str
        Tên tệp tin biểu tượng (có hoặc không có đuôi file)
    
    Returns
    -------
    str
        Đường dẫn đầy đủ đến tệp tin biểu tượng
    """
    # Thêm đuôi .svg nếu không có đuôi tệp
    if not any(icon_name.endswith(ext) for ext in ['.svg', '.png', '.jpg', '.jpeg']):
        icon_name = f"{icon_name}.svg"
    
    # Đầu tiên tìm trong thư mục biểu tượng mới
    icon_path = os.path.join(get_project_root(), "quangtps", "ui", "icons", "new_icons", icon_name)
    if os.path.exists(icon_path):
        return icon_path
    
    # Nếu không tìm thấy, kiểm tra trong thư mục biểu tượng gốc
    icon_path = os.path.join(get_project_root(), "quangtps", "ui", "icons", icon_name)
    if os.path.exists(icon_path):
        return icon_path
    
    # Cuối cùng, thử tìm trong thư mục dữ liệu
    icon_path = os.path.join(get_project_root(), "data", "icons", icon_name)
    if os.path.exists(icon_path):
        return icon_path
    
    # Nếu không tìm thấy, trả về None
    logger.warning(f"Cannot find icon: {icon_name}")
    return None
