"""
Hệ thống logging cho QuangTPS.
"""

import os
import sys
import logging
import locale
from logging.handlers import RotatingFileHandler

from .config import Config

def setup_utf8_console():
    """
    Thiết lập UTF-8 cho console trên Windows.
    
    Hàm này cố gắng thiết lập locale và encoding thích hợp 
    để hiển thị tiếng Việt đúng cách trên Windows.
    """
    if sys.platform == 'win32':
        try:
            # Thử đặt locale cho tiếng Việt
            locale.setlocale(locale.LC_ALL, 'Vietnamese_Vietnam.65001')
        except locale.Error:
            try:
                # Nếu không có locale tiếng Việt, sử dụng UTF-8 chung
                locale.setlocale(locale.LC_ALL, '.65001')
            except locale.Error:
                # Nếu vẫn không được, thử đặt môi trường trực tiếp
                pass
        
        try:
            # Đặt PYTHONIOENCODING
            os.environ['PYTHONIOENCODING'] = 'utf-8'
            
            # Thiết lập utf-8 cho stdout và stderr
            import codecs
            sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
            sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)
            
            # Đặt biến môi trường PYTHONUTF8
            os.environ['PYTHONUTF8'] = '1'
        except Exception as e:
            print(f"Cảnh báo: Không thể thiết lập UTF-8 cho console: {str(e)}")
    
    return True

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
    
    # Thiết lập UTF-8 cho console
    setup_utf8_console()
    
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
    
    # Thêm StreamHandler với xử lý Unicode tốt hơn
    try:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    except Exception as e:
        print(f"Lỗi khi thiết lập console handler: {str(e)}")
    
    # Handler file (nếu có)
    if log_file:
        try:
            # Đảm bảo thư mục chứa file log tồn tại
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            
            # Tạo rotating file handler để tránh file log quá lớn
            file_handler = RotatingFileHandler(
                log_file, maxBytes=10*1024*1024, backupCount=5, 
                encoding='utf-8'  # Chỉ định encoding utf-8 cho file
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            print(f"Lỗi khi thiết lập file handler: {str(e)}")
            # Cố gắng tạo file handler đơn giản hơn nếu rotating handler gặp lỗi
            try:
                simple_file_handler = logging.FileHandler(
                    log_file, encoding='utf-8'
                )
                simple_file_handler.setFormatter(formatter)
                logger.addHandler(simple_file_handler)
            except Exception as e2:
                print(f"Lỗi nghiêm trọng với file logging: {str(e2)}")
    
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
