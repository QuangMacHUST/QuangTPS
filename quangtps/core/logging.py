"""
Hệ thống logging cho QuangTPS.
"""

import os
import logging
import datetime
from logging.handlers import RotatingFileHandler

from .config import Config

def setup_logger(name="quangtps", log_file=None, level=None):
    """
    Thiết lập logger với tên và cấu hình được chỉ định.
    
    Parameters:
        name (str): Tên của logger
        log_file (str): Đường dẫn đến file log
        level (int): Mức độ log (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    
    Returns:
        logging.Logger: Logger đã cấu hình
    """
    
    # Lấy cấu hình từ Config nếu không được chỉ định
    config = Config.get_instance()
    if log_file is None:
        log_file = config.log_file
    if level is None:
        level = config.log_level
    
    # Tạo logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Xóa các handler hiện có để tránh lặp log
    if logger.handlers:
        logger.handlers = []
    
    # Định dạng log
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Handler console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Handler file (nếu có)
    if log_file:
        # Đảm bảo thư mục chứa file log tồn tại
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # Tạo rotating file handler để tránh file log quá lớn
        file_handler = RotatingFileHandler(
            log_file, maxBytes=10*1024*1024, backupCount=5
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def get_logger(name="quangtps"):
    """
    Lấy logger đã được cấu hình sẵn hoặc tạo mới nếu chưa tồn tại.
    
    Parameters:
        name (str): Tên của logger
    
    Returns:
        logging.Logger: Logger
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logger(name)
    return logger
