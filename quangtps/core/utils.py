"""
Các tiện ích và công cụ sử dụng trong toàn bộ hệ thống QuangTPS.
"""

import time
import uuid
import psutil
import hashlib
import platform
import numpy as np
import os
from datetime import datetime
import json
import logging
import zipfile
import tempfile
import re
import shutil
import sys
from typing import Optional, Dict, List, Any, Union, Tuple, Callable
from pathlib import Path

logger = logging.getLogger(__name__)


class Timer:
    """Đo thời gian thực thi của một đoạn code"""

    def __init__(self, name="Task"):
        self.name = name
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        print(f"{self.name} completed in {duration:.4f} seconds")

    def start(self):
        """Bắt đầu đo thời gian"""
        self.start_time = time.time()

    def stop(self):
        """Dừng đo thời gian và trả về thời gian đã trôi qua"""
        self.end_time = time.time()
        return self.end_time - self.start_time

    def elapsed(self):
        """Trả về thời gian đã trôi qua (không dừng bộ hẹn giờ)"""
        if self.start_time is None:
            return 0
        current_time = time.time() if self.end_time is None else self.end_time
        return current_time - self.start_time


def get_memory_usage():
    """Trả về thông tin về việc sử dụng bộ nhớ"""
    process = psutil.Process()
    mem_info = process.memory_info()
    return {
        "rss": mem_info.rss / (1024 * 1024),  # Resident Set Size in MB
        "vms": mem_info.vms / (1024 * 1024),  # Virtual Memory Size in MB
        "percent": process.memory_percent(),
    }


def create_unique_id():
    """Tạo ID duy nhất cho các đối tượng trong hệ thống"""
    return str(uuid.uuid4())


def hash_file(file_path, algorithm="sha256"):
    """Tính hash của một file"""
    hash_obj = hashlib.new(algorithm)
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_obj.update(chunk)
    return hash_obj.hexdigest()


def ensure_directory(directory_path):
    """
    Đảm bảo thư mục tồn tại, nếu không tạo mới thư mục đó.

    Parameters
    ----------
    directory_path : str
        Đường dẫn đến thư mục cần đảm bảo

    Returns
    -------
    str
        Đường dẫn đến thư mục đã được đảm bảo tồn tại
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path, exist_ok=True)
    return directory_path


def get_system_info():
    """Trả về thông tin về hệ thống"""
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "memory_total": psutil.virtual_memory().total / (1024 * 1024 * 1024),  # in GB
        "memory_available": psutil.virtual_memory().available
        / (1024 * 1024 * 1024),  # in GB
        "disk_total": psutil.disk_usage("/").total / (1024 * 1024 * 1024),  # in GB
        "disk_free": psutil.disk_usage("/").free / (1024 * 1024 * 1024),  # in GB
    }


def interpolate_linear(x, y, new_x):
    """Nội suy tuyến tính giữa các điểm"""
    return np.interp(new_x, x, y)


def downsample(data, factor):
    """Giảm lấy mẫu mảng dữ liệu"""
    return data[::factor]


def normalize(data, min_val=0, max_val=1):
    """Chuẩn hóa dữ liệu vào khoảng [min_val, max_val]"""
    data_min = np.min(data)
    data_max = np.max(data)
    if data_max == data_min:
        return np.ones_like(data) * min_val
    normalized = (data - data_min) / (data_max - data_min)
    return normalized * (max_val - min_val) + min_val


def format_date(date_obj=None, format_str="%Y-%m-%d"):
    """Định dạng ngày tháng"""
    if date_obj is None:
        date_obj = datetime.now()
    return date_obj.strftime(format_str)


def get_timestamp() -> str:
    """
    Tạo và trả về một chuỗi timestamp hiện tại theo định dạng
    năm-tháng-ngày-giờ-phút-giây.

    Returns
    -------
    str
        Chuỗi timestamp, ví dụ: '2026-01-25-12-30-45'
    """
    return datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


def create_directory_if_not_exists(directory_path: Union[str, Path]) -> bool:
    """
    Tạo thư mục nếu nó chưa tồn tại.

    Parameters
    ----------
    directory_path : Union[str, Path]
        Đường dẫn thư mục cần tạo

    Returns
    -------
    bool
        True nếu thư mục đã tồn tại hoặc đã được tạo thành công
        False nếu có lỗi xảy ra khi tạo thư mục
    """
    try:
        # Chuyển đổi sang Path đối tượng nếu là chuỗi
        if isinstance(directory_path, str):
            directory_path = Path(directory_path)

        # Tạo thư mục nếu chưa tồn tại
        directory_path.mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"Không thể tạo thư mục {directory_path}: {str(e)}")
        return False


def normalize_path(path: Union[str, Path]) -> Path:
    """
    Chuẩn hóa đường dẫn file hoặc thư mục.

    Parameters
    ----------
    path : Union[str, Path]
        Đường dẫn cần chuẩn hóa

    Returns
    -------
    Path
        Đường dẫn đã được chuẩn hóa
    """
    if isinstance(path, str):
        path = Path(path)

    return path.expanduser().absolute()


def generate_uid() -> str:
    """
    Tạo một chuỗi định danh duy nhất (UUID) với định dạng chuỗi.

    Returns
    -------
    str
        Chuỗi UUID duy nhất
    """
    return str(uuid.uuid4())


def sanitize_filename(filename: str) -> str:
    """
    Làm sạch tên file để loại bỏ các ký tự không hợp lệ.

    Parameters
    ----------
    filename : str
        Tên file cần làm sạch

    Returns
    -------
    str
        Tên file đã được làm sạch
    """
    # Thay thế các ký tự không hợp lệ bằng dấu gạch dưới
    sanitized = re.sub(r'[\\/*?:"<>|]', "_", filename)
    return sanitized


def load_json_file(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Đọc và phân tích nội dung từ file JSON.

    Parameters
    ----------
    file_path : Union[str, Path]
        Đường dẫn đến file JSON

    Returns
    -------
    Dict[str, Any]
        Dữ liệu được đọc từ file JSON

    Raises
    ------
    FileNotFoundError
        Nếu file không tồn tại
    json.JSONDecodeError
        Nếu nội dung file không phải là JSON hợp lệ
    """
    file_path = normalize_path(file_path)

    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json_file(
    data: Dict[str, Any], file_path: Union[str, Path], pretty: bool = True
) -> bool:
    """
    Lưu dữ liệu vào file JSON.

    Parameters
    ----------
    data : Dict[str, Any]
        Dữ liệu cần lưu
    file_path : Union[str, Path]
        Đường dẫn đến file JSON
    pretty : bool, optional
        Nếu True, định dạng JSON để dễ đọc, mặc định là True

    Returns
    -------
    bool
        True nếu lưu thành công, False nếu thất bại
    """
    file_path = normalize_path(file_path)

    try:
        with open(file_path, "w", encoding="utf-8") as f:
            if pretty:
                json.dump(data, f, indent=2, ensure_ascii=False)
            else:
                json.dump(data, f, ensure_ascii=False)
        return True
    except Exception as e:
        logger.error(f"Không thể lưu file JSON {file_path}: {str(e)}")
        return False


def compute_md5(file_path: Union[str, Path]) -> str:
    """
    Tính toán MD5 hash của một file.

    Parameters
    ----------
    file_path : Union[str, Path]
        Đường dẫn đến file

    Returns
    -------
    str
        Chuỗi MD5 hash của file
    """
    file_path = normalize_path(file_path)
    hash_md5 = hashlib.md5()

    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)

    return hash_md5.hexdigest()


def compress_directory(
    dir_path: Union[str, Path], output_path: Union[str, Path]
) -> bool:
    """
    Nén một thư mục thành file zip.

    Parameters
    ----------
    dir_path : Union[str, Path]
        Đường dẫn đến thư mục cần nén
    output_path : Union[str, Path]
        Đường dẫn đến file zip đầu ra

    Returns
    -------
    bool
        True nếu nén thành công, False nếu thất bại
    """
    dir_path = normalize_path(dir_path)
    output_path = normalize_path(output_path)

    try:
        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(dir_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, dir_path)
                    zipf.write(file_path, arcname)

        return True
    except Exception as e:
        logger.error(f"Không thể nén thư mục {dir_path}: {str(e)}")
        return False


def extract_zip(zip_path: Union[str, Path], extract_path: Union[str, Path]) -> bool:
    """
    Giải nén file zip vào thư mục chỉ định.

    Parameters
    ----------
    zip_path : Union[str, Path]
        Đường dẫn đến file zip
    extract_path : Union[str, Path]
        Đường dẫn đến thư mục giải nén

    Returns
    -------
    bool
        True nếu giải nén thành công, False nếu thất bại
    """
    zip_path = normalize_path(zip_path)
    extract_path = normalize_path(extract_path)

    try:
        # Tạo thư mục giải nén nếu chưa tồn tại
        create_directory_if_not_exists(extract_path)

        with zipfile.ZipFile(zip_path, "r") as zipf:
            zipf.extractall(extract_path)

        return True
    except Exception as e:
        logger.error(f"Không thể giải nén file {zip_path}: {str(e)}")
        return False


def get_file_extension(file_path: Union[str, Path]) -> str:
    """
    Lấy phần mở rộng của file.

    Parameters
    ----------
    file_path : Union[str, Path]
        Đường dẫn đến file

    Returns
    -------
    str
        Phần mở rộng của file (không bao gồm dấu chấm)
    """
    file_path = normalize_path(file_path)
    return file_path.suffix.lstrip(".")


def get_temp_directory(prefix: str = "quangtps_") -> Path:
    """
    Tạo và trả về đường dẫn đến thư mục tạm thời.

    Parameters
    ----------
    prefix : str, optional
        Tiền tố cho tên thư mục tạm thời, mặc định là "quangtps_"

    Returns
    -------
    Path
        Đường dẫn đến thư mục tạm thời
    """
    temp_dir = Path(tempfile.mkdtemp(prefix=prefix))
    return temp_dir


def clean_temp_directory(temp_dir: Union[str, Path]) -> bool:
    """
    Xóa thư mục tạm thời và tất cả nội dung của nó.

    Parameters
    ----------
    temp_dir : Union[str, Path]
        Đường dẫn đến thư mục tạm thời cần xóa

    Returns
    -------
    bool
        True nếu xóa thành công, False nếu thất bại
    """
    temp_dir = normalize_path(temp_dir)

    try:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        return True
    except Exception as e:
        logger.error(f"Không thể xóa thư mục tạm thời {temp_dir}: {str(e)}")
        return False


def is_valid_path(path: Union[str, Path]) -> bool:
    """
    Kiểm tra xem đường dẫn có hợp lệ không.

    Parameters
    ----------
    path : Union[str, Path]
        Đường dẫn cần kiểm tra

    Returns
    -------
    bool
        True nếu đường dẫn hợp lệ, False nếu không
    """
    try:
        Path(path)
        return True
    except:
        return False


def get_app_data_directory() -> Path:
    """
    Lấy thư mục dữ liệu ứng dụng tùy thuộc vào hệ điều hành.

    Returns
    -------
    Path
        Đường dẫn đến thư mục dữ liệu ứng dụng
    """
    app_name = "QuangTPS"

    if sys.platform == "win32":
        app_data = os.getenv("APPDATA")
        if app_data:
            return Path(app_data) / app_name
    elif sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / app_name
    else:  # Linux/Unix
        return Path.home() / ".config" / app_name.lower()

    # Fallback if platform-specific directories couldn't be determined
    return Path.home() / f".{app_name.lower()}"


def format_file_size(size_in_bytes: int) -> str:
    """
    Định dạng kích thước file theo đơn vị KB, MB, GB.

    Parameters
    ----------
    size_in_bytes : int
        Kích thước file tính bằng byte

    Returns
    -------
    str
        Chuỗi định dạng kích thước file
    """
    if size_in_bytes < 1024:
        return f"{size_in_bytes} B"
    elif size_in_bytes < 1024 * 1024:
        return f"{size_in_bytes / 1024:.2f} KB"
    elif size_in_bytes < 1024 * 1024 * 1024:
        return f"{size_in_bytes / (1024 * 1024):.2f} MB"
    else:
        return f"{size_in_bytes / (1024 * 1024 * 1024):.2f} GB"
