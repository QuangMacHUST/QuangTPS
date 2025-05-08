"""
Module với các tiện ích xử lý file.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Union, List, Tuple

logger = logging.getLogger(__name__)


def get_file_extension(file_path: str) -> str:
    """
    Lấy phần mở rộng của file.

    Parameters:
        file_path: Đường dẫn file

    Returns:
        Phần mở rộng của file (không có dấu chấm, đã được chuyển thành chữ thường)
    """
    _, ext = os.path.splitext(file_path)
    return ext.lower()[1:] if ext else ""


def ensure_directory_exists(dir_path: str) -> bool:
    """
    Đảm bảo thư mục tồn tại, tạo nếu cần.

    Parameters:
        dir_path: Đường dẫn thư mục

    Returns:
        True nếu thư mục tồn tại hoặc đã được tạo thành công, False nếu có lỗi
    """
    try:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        return True
    except Exception as e:
        logger.error(f"Lỗi khi tạo thư mục {dir_path}: {e}")
        return False


def get_file_name(file_path: str, with_extension: bool = True) -> str:
    """
    Lấy tên file từ đường dẫn.

    Parameters:
        file_path: Đường dẫn file
        with_extension: Có bao gồm phần mở rộng hay không

    Returns:
        Tên file
    """
    base_name = os.path.basename(file_path)
    if with_extension:
        return base_name
    else:
        return os.path.splitext(base_name)[0]


def list_files_with_extension(dir_path: str, extension: str) -> List[str]:
    """
    Liệt kê tất cả các file có phần mở rộng cho trước trong thư mục.

    Parameters:
        dir_path: Đường dẫn thư mục
        extension: Phần mở rộng cần tìm (không bao gồm dấu chấm)

    Returns:
        Danh sách đường dẫn đầy đủ đến các file
    """
    if not os.path.exists(dir_path):
        logger.warning(f"Thư mục {dir_path} không tồn tại")
        return []

    extension = extension.lower()
    if extension.startswith("."):
        extension = extension[1:]

    return [
        os.path.join(dir_path, f)
        for f in os.listdir(dir_path)
        if os.path.isfile(os.path.join(dir_path, f))
        and f.lower().endswith(f".{extension}")
    ]


def sanitize_file_name(file_name: str) -> str:
    """
    Làm sạch tên file, loại bỏ các ký tự không hợp lệ.

    Parameters:
        file_name: Tên file cần làm sạch

    Returns:
        Tên file đã được làm sạch
    """
    # Danh sách các ký tự không hợp lệ
    invalid_chars = ["<", ">", ":", '"', "/", "\\", "|", "?", "*"]
    result = file_name
    for char in invalid_chars:
        result = result.replace(char, "_")
    return result
