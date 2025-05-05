#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module io_utils - Tiện ích hỗ trợ đọc/ghi file cho QuangTPS.

Module này cung cấp các hàm tiện ích để làm việc với file và thư mục,
bao gồm tạo thư mục, kiểm tra sự tồn tại, đọc/ghi dữ liệu và các chức năng IO khác.
"""

import os
import json
import pickle
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Union, Optional, Tuple, BinaryIO

logger = logging.getLogger(__name__)


def create_directory_if_not_exists(directory_path: str) -> bool:
    """
    Tạo thư mục nếu chưa tồn tại.

    Parameters:
        directory_path (str): Đường dẫn thư mục cần tạo

    Returns:
        bool: True nếu thư mục đã tồn tại hoặc đã được tạo thành công, False nếu có lỗi
    """
    try:
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
            logger.info(f"Đã tạo thư mục: {directory_path}")
        return True
    except Exception as e:
        logger.error(f"Không thể tạo thư mục {directory_path}: {str(e)}")
        return False


def save_json(data: Dict[str, Any], file_path: str, indent: int = 4) -> bool:
    """
    Lưu dữ liệu vào file JSON.

    Parameters:
        data (Dict[str, Any]): Dữ liệu cần lưu
        file_path (str): Đường dẫn file đầu ra
        indent (int, optional): Số khoảng trắng thụt đầu dòng

    Returns:
        bool: True nếu lưu thành công, False nếu thất bại
    """
    try:
        # Đảm bảo thư mục tồn tại
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
        logger.debug(f"Đã lưu dữ liệu vào file JSON: {file_path}")
        return True
    except Exception as e:
        logger.error(f"Không thể lưu file JSON {file_path}: {str(e)}")
        return False


def load_json(file_path: str) -> Optional[Dict[str, Any]]:
    """
    Đọc dữ liệu từ file JSON.

    Parameters:
        file_path (str): Đường dẫn file JSON

    Returns:
        Optional[Dict[str, Any]]: Dữ liệu từ file JSON hoặc None nếu có lỗi
    """
    try:
        if not os.path.exists(file_path):
            logger.warning(f"File không tồn tại: {file_path}")
            return None

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.debug(f"Đã đọc dữ liệu từ file JSON: {file_path}")
        return data
    except Exception as e:
        logger.error(f"Không thể đọc file JSON {file_path}: {str(e)}")
        return None


def save_pickle(data: Any, file_path: str) -> bool:
    """
    Lưu dữ liệu vào file pickle.

    Parameters:
        data (Any): Dữ liệu cần lưu
        file_path (str): Đường dẫn file đầu ra

    Returns:
        bool: True nếu lưu thành công, False nếu thất bại
    """
    try:
        # Đảm bảo thư mục tồn tại
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        with open(file_path, "wb") as f:
            pickle.dump(data, f)
        logger.debug(f"Đã lưu dữ liệu vào file pickle: {file_path}")
        return True
    except Exception as e:
        logger.error(f"Không thể lưu file pickle {file_path}: {str(e)}")
        return False


def load_pickle(file_path: str) -> Optional[Any]:
    """
    Đọc dữ liệu từ file pickle.

    Parameters:
        file_path (str): Đường dẫn file pickle

    Returns:
        Optional[Any]: Dữ liệu từ file pickle hoặc None nếu có lỗi
    """
    try:
        if not os.path.exists(file_path):
            logger.warning(f"File không tồn tại: {file_path}")
            return None

        with open(file_path, "rb") as f:
            data = pickle.load(f)
        logger.debug(f"Đã đọc dữ liệu từ file pickle: {file_path}")
        return data
    except Exception as e:
        logger.error(f"Không thể đọc file pickle {file_path}: {str(e)}")
        return None


def safe_file_write(file_path: str, content: str, encoding: str = "utf-8") -> bool:
    """
    Ghi nội dung vào file một cách an toàn bằng cách sử dụng file tạm thời.

    Parameters:
        file_path (str): Đường dẫn file đầu ra
        content (str): Nội dung cần ghi
        encoding (str, optional): Mã hóa ký tự

    Returns:
        bool: True nếu ghi thành công, False nếu thất bại
    """
    try:
        # Đảm bảo thư mục tồn tại
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        # Tạo file tạm
        fd, temp_path = tempfile.mkstemp(prefix="quangtps_", dir=directory)
        with os.fdopen(fd, "w", encoding=encoding) as f:
            f.write(content)

        # Thay thế file cũ bằng file mới
        if os.path.exists(file_path):
            os.remove(file_path)
        shutil.move(temp_path, file_path)

        logger.debug(f"Đã ghi nội dung vào file: {file_path}")
        return True
    except Exception as e:
        logger.error(f"Không thể ghi vào file {file_path}: {str(e)}")
        return False


def get_file_extension(file_path: str) -> str:
    """
    Lấy phần mở rộng của file.

    Parameters:
        file_path (str): Đường dẫn file

    Returns:
        str: Phần mở rộng của file (không có dấu chấm)
    """
    return os.path.splitext(file_path)[1].lstrip(".")


def list_files_by_extension(directory: str, extension: str) -> List[str]:
    """
    Liệt kê các file có phần mở rộng cụ thể trong thư mục.

    Parameters:
        directory (str): Đường dẫn thư mục
        extension (str): Phần mở rộng file cần tìm (không có dấu chấm)

    Returns:
        List[str]: Danh sách đường dẫn các file tìm được
    """
    try:
        if not os.path.exists(directory):
            logger.warning(f"Thư mục không tồn tại: {directory}")
            return []

        extension = extension.lstrip(".")
        result = []
        for file in os.listdir(directory):
            if file.endswith(f".{extension}"):
                result.append(os.path.join(directory, file))
        return result
    except Exception as e:
        logger.error(f"Lỗi khi liệt kê file trong thư mục {directory}: {str(e)}")
        return []


def create_backup(file_path: str, backup_suffix: str = ".bak") -> Optional[str]:
    """
    Tạo bản sao lưu của một file.

    Parameters:
        file_path (str): Đường dẫn file cần sao lưu
        backup_suffix (str, optional): Hậu tố cho file sao lưu

    Returns:
        Optional[str]: Đường dẫn file sao lưu hoặc None nếu thất bại
    """
    try:
        if not os.path.exists(file_path):
            logger.warning(f"File không tồn tại, không thể sao lưu: {file_path}")
            return None

        backup_path = f"{file_path}{backup_suffix}"
        shutil.copy2(file_path, backup_path)
        logger.debug(f"Đã tạo bản sao lưu: {backup_path}")
        return backup_path
    except Exception as e:
        logger.error(f"Không thể tạo bản sao lưu cho {file_path}: {str(e)}")
        return None


def get_application_data_dir(app_name: str = "QuangTPS") -> str:
    """
    Lấy thư mục dữ liệu ứng dụng phù hợp với hệ điều hành.

    Parameters:
        app_name (str, optional): Tên ứng dụng

    Returns:
        str: Đường dẫn thư mục dữ liệu ứng dụng
    """
    home = Path.home()
    if os.name == "posix":  # Linux/Mac
        app_dir = home / f".{app_name.lower()}"
    elif os.name == "nt":  # Windows
        app_data = os.environ.get("APPDATA", str(home / "AppData" / "Roaming"))
        app_dir = Path(app_data) / app_name
    else:
        app_dir = home / app_name

    app_dir.mkdir(parents=True, exist_ok=True)
    return str(app_dir)
