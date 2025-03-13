"""
Các tiện ích và công cụ sử dụng trong toàn bộ hệ thống QuangTPS.
"""

import time
import uuid
import psutil
import hashlib
import platform
import numpy as np
from datetime import datetime

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
        'rss': mem_info.rss / (1024 * 1024),  # Resident Set Size in MB
        'vms': mem_info.vms / (1024 * 1024),  # Virtual Memory Size in MB
        'percent': process.memory_percent()
    }

def create_unique_id():
    """Tạo ID duy nhất cho các đối tượng trong hệ thống"""
    return str(uuid.uuid4())

def hash_file(file_path, algorithm='sha256'):
    """Tính hash của một file"""
    hash_obj = hashlib.new(algorithm)
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            hash_obj.update(chunk)
    return hash_obj.hexdigest()

def get_system_info():
    """Trả về thông tin về hệ thống"""
    return {
        'platform': platform.platform(),
        'processor': platform.processor(),
        'python_version': platform.python_version(),
        'memory_total': psutil.virtual_memory().total / (1024 * 1024 * 1024),  # in GB
        'memory_available': psutil.virtual_memory().available / (1024 * 1024 * 1024),  # in GB
        'disk_total': psutil.disk_usage('/').total / (1024 * 1024 * 1024),  # in GB
        'disk_free': psutil.disk_usage('/').free / (1024 * 1024 * 1024)  # in GB
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

def format_date(date_obj=None, format_str='%Y-%m-%d'):
    """Định dạng ngày tháng"""
    if date_obj is None:
        date_obj = datetime.now()
    return date_obj.strftime(format_str)
