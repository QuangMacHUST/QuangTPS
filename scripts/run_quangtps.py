#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script khởi động hệ thống QuangTPS.

Script này khởi động hệ thống QuangTPS với logo và màn hình chào,
đồng thời thiết lập các cấu hình cần thiết trước khi chạy.
"""

import os
import sys
import time
import logging
import argparse
from pathlib import Path

def setup_environment():
    """Thiết lập môi trường chạy."""
    # Thêm thư mục gốc vào sys.path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)
    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)
    
    # Thiết lập biến môi trường QUANGTPS_ROOT
    os.environ['QUANGTPS_ROOT'] = root_dir
    
    # Tạo thư mục logs nếu chưa tồn tại
    logs_dir = os.path.join(root_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    # Tạo thư mục data nếu chưa tồn tại
    data_dir = os.path.join(root_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    # Tạo thư mục temp nếu chưa tồn tại
    temp_dir = os.path.join(root_dir, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Tạo các thư mục cần thiết khác
    os.makedirs(os.path.join(data_dir, "beam_data"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "patient_data"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "templates"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "dicom"), exist_ok=True)

def setup_logging(verbose=False):
    """Thiết lập logging."""
    # Xác định mức logging
    log_level = logging.DEBUG if verbose else logging.INFO
    
    # Tên file log
    log_file = os.path.join(
        os.environ.get('QUANGTPS_ROOT', '.'),
        "logs",
        f"quangtps_{time.strftime('%Y%m%d_%H%M%S')}.log"
    )
    
    # Định dạng log
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Thiết lập logging cho console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(logging.Formatter(log_format))
    
    # Thiết lập logging cho file
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(log_level)
    file_handler.setFormatter(logging.Formatter(log_format))
    
    # Cấu hình root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    # Log thông tin khởi động
    logger = logging.getLogger(__name__)
    logger.info(f"Khởi động QuangTPS tại {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Thư mục gốc: {os.environ.get('QUANGTPS_ROOT', '.')}")
    logger.info(f"File log: {log_file}")
    
    return logger

def check_dependencies():
    """Kiểm tra các thư viện phụ thuộc."""
    logger = logging.getLogger(__name__)
    
    # Dictionary ánh xạ tên gói với tên module
    package_to_module = {
        "numpy": "numpy",
        "scipy": "scipy",
        "pandas": "pandas",
        "pydicom": "pydicom",
        "PyQt5": "PyQt5",
        "matplotlib": "matplotlib",
        "scikit-image": "skimage",
        "dicompyler-core": "dicompylercore"
    }
    
    missing_packages = []
    
    for package, module in package_to_module.items():
        try:
            __import__(module)
            logger.debug(f"Thư viện {package} đã được cài đặt")
        except ImportError:
            logger.error(f"Thư viện {package} chưa được cài đặt")
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Thiếu các thư viện: {', '.join(missing_packages)}")
        logger.error("Vui lòng cài đặt các thư viện còn thiếu bằng lệnh:")
        logger.error(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def show_splash_screen():
    """Hiển thị màn hình chào."""
    from PyQt5.QtWidgets import QSplashScreen, QApplication
    from PyQt5.QtGui import QPixmap
    from PyQt5.QtCore import Qt, QTimer
    
    # Tạo ứng dụng QApplication
    app = QApplication(sys.argv)
    splash = None
    
    # Đường dẫn đến file splash screen
    splash_path = os.path.join(
        os.environ.get('QUANGTPS_ROOT', '.'),
        "quangtps",
        "ui",
        "icons",
        "new_icons",
        "splash.png"
    )
    
    # Kiểm tra xem file splash screen có tồn tại không
    if not os.path.exists(splash_path):
        # Sử dụng splash screen mặc định
        splash_path = os.path.join(
            os.environ.get('QUANGTPS_ROOT', '.'),
            "quangtps",
            "ui",
            "icons",
            "splash.png"
        )
    
    # Tạo splash screen nếu tìm thấy file
    if os.path.exists(splash_path):
        # Tạo splash screen
        splash_pixmap = QPixmap(splash_path)
        splash = QSplashScreen(splash_pixmap)
        
        # Thiết lập văn bản
        splash.showMessage(
            "Đang khởi động QuangTPS...",
            Qt.AlignBottom | Qt.AlignCenter,
            Qt.white
        )
        
        # Hiển thị splash screen
        splash.show()
        
        # Đảm bảo splash screen hiển thị
        app.processEvents()
    
    return app, splash

def parse_arguments():
    """Phân tích đối số dòng lệnh."""
    parser = argparse.ArgumentParser(description='Khởi động hệ thống QuangTPS')
    
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Hiển thị thông tin chi tiết')
    
    parser.add_argument('--no-splash', action='store_true',
                       help='Không hiển thị màn hình chào')
    
    parser.add_argument('--console', '-c', action='store_true',
                       help='Chạy ở chế độ console (không có giao diện đồ họa)')
    
    parser.add_argument('--demo', '-d', action='store_true',
                       help='Chạy ở chế độ demo với dữ liệu mẫu')
    
    return parser.parse_args()

def main():
    """Hàm chính của script."""
    # Phân tích đối số
    args = parse_arguments()
    
    # Thiết lập môi trường
    setup_environment()
    
    # Thiết lập logging
    logger = setup_logging(args.verbose)
    
    # Kiểm tra các thư viện phụ thuộc
    if not check_dependencies():
        logger.error("Không thể khởi động QuangTPS do thiếu thư viện")
        return 1
    
    # Hiển thị splash screen nếu không ở chế độ console và không tắt splash screen
    if not args.console and not args.no_splash:
        app, splash = show_splash_screen()
        # Độ trễ để hiển thị splash screen
        from PyQt5.QtCore import QTimer
        QTimer.singleShot(2000, lambda: start_application(app, splash, args))
    else:
        from PyQt5.QtWidgets import QApplication
        app = QApplication(sys.argv)
        start_application(app, None, args)
    
    # Bắt đầu vòng lặp sự kiện
    return app.exec_()

def start_application(app, splash, args):
    """Bắt đầu ứng dụng chính."""
    logger = logging.getLogger(__name__)
    logger.info("Bắt đầu ứng dụng chính")
    
    try:
        # Import main window
        from quangtps.ui.main_window import MainWindow
        
        # Tạo main window
        main_window = MainWindow()
        
        # Tải dữ liệu mẫu nếu ở chế độ demo
        if args.demo:
            logger.info("Chạy ở chế độ demo, đang tải dữ liệu mẫu")
            main_window.load_demo_data()
        
        # Đóng splash screen nếu có
        if splash:
            splash.finish(main_window)
        
        # Hiển thị main window
        main_window.show()
        
        logger.info("Ứng dụng đã khởi động thành công")
        
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        logger.error(f"Lỗi khi khởi động ứng dụng: {str(e)}\n{error_traceback}")
        if splash:
            splash.close()
        
        # Hiển thị thông báo lỗi
        from PyQt5.QtWidgets import QMessageBox
        QMessageBox.critical(
            None,
            "Lỗi khởi động",
            f"Không thể khởi động QuangTPS: {str(e)}"
        )
        
        # Thoát ứng dụng
        app.quit()

if __name__ == "__main__":
    sys.exit(main()) 