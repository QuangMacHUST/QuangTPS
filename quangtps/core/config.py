"""
Cấu hình toàn cục cho QuangTPS.
Sử dụng mẫu thiết kế Singleton để đảm bảo chỉ có một instance của Config.
"""

import os
import json
import logging
from pathlib import Path

class Config:
    """Quản lý cấu hình hệ thống"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    @classmethod
    def get_instance(cls):
        """Trả về instance duy nhất của Config"""
        if cls._instance is None:
            return cls()
        return cls._instance
    
    def _initialize(self):
        """Khởi tạo các cấu hình mặc định"""
        self.app_name = "QuangTPS"
        
        # Đường dẫn
        self.root_dir = self._get_root_dir()
        self.data_dir = os.path.join(self.root_dir, "data")
        self.temp_dir = os.path.join(self.root_dir, "temp")
        self.log_dir = os.path.join(self.root_dir, "logs")
        
        # Đảm bảo các thư mục tồn tại
        for dir_path in [self.data_dir, self.temp_dir, self.log_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Cấu hình logging
        self.log_level = logging.INFO
        self.log_file = os.path.join(self.log_dir, "quangtps.log")
        
        # Cấu hình DICOM
        self.dicom_dir = os.path.join(self.data_dir, "dicom")
        os.makedirs(self.dicom_dir, exist_ok=True)
        
        # Cấu hình hiển thị
        self.default_window_width = 400
        self.default_window_level = 40
        
        # Tải cấu hình từ file nếu tồn tại
        self.config_file = os.path.join(self.root_dir, "config.json")
        if os.path.exists(self.config_file):
            self.load_config()
    
    def _get_root_dir(self):
        """Xác định thư mục gốc của ứng dụng"""
        # Giả sử file hiện tại ở quangtps/core/config.py
        # Đi lên 2 cấp để tìm thư mục gốc
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        root_dir = os.path.dirname(parent_dir)
        return root_dir
    
    def save_config(self):
        """Lưu cấu hình hiện tại vào file"""
        config_dict = {
            "log_level": self.log_level,
            "dicom_dir": self.dicom_dir,
            "default_window_width": self.default_window_width,
            "default_window_level": self.default_window_level
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(config_dict, f, indent=4)
    
    def load_config(self):
        """Tải cấu hình từ file"""
        try:
            with open(self.config_file, 'r') as f:
                config_dict = json.load(f)
            
            # Cập nhật các thuộc tính từ file
            for key, value in config_dict.items():
                if hasattr(self, key):
                    setattr(self, key, value)
        except Exception as e:
            print(f"Lỗi khi tải cấu hình: {e}")
    
    def get(self, key, default=None):
        """Lấy giá trị cấu hình theo key"""
        return getattr(self, key, default)
    
    def set(self, key, value):
        """Đặt giá trị cấu hình theo key"""
        setattr(self, key, value)
