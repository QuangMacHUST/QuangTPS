#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QuangTPS - Hệ thống Lập kế hoạch Xạ trị Mã nguồn Mở
Điểm vào chính của ứng dụng
"""

import sys
import os
import argparse
import traceback
from pathlib import Path

# Kiểm tra và cài đặt PyQt5
try:
    from PyQt5.QtWidgets import QApplication, QSplashScreen, QMessageBox
    from PyQt5.QtGui import QPixmap
    from PyQt5.QtCore import Qt, QTimer
except ImportError:
    print("Lỗi: PyQt5 chưa được cài đặt.")
    print("Vui lòng cài đặt bằng lệnh: pip install PyQt5")
    sys.exit(1)

# Thiết lập đường dẫn
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(BASE_DIR)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import các module của QuangTPS
try:
    from quangtps.core.logging import setup_logger, get_logger
    from quangtps.core.config import Config
    from quangtps.ui.main_window import MainWindow
    from quangtps.scripts.batch_processing import batch_process
    from quangtps.scripts.system_check import check_system
except ImportError as e:
    print(f"Lỗi nhập module QuangTPS: {str(e)}")
    print("Vui lòng kiểm tra cài đặt và cấu trúc thư mục.")
    sys.exit(1)

def parse_arguments():
    """Phân tích tham số dòng lệnh"""
    parser = argparse.ArgumentParser(description='QuangTPS - Hệ thống Lập kế hoạch Xạ trị Mã nguồn Mở')
    
    parser.add_argument('--config', type=str, help='Đường dẫn đến file cấu hình')
    parser.add_argument('--log-level', type=str, default='INFO', 
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Mức độ log')
    parser.add_argument('--batch', type=str, help='Chạy chế độ xử lý hàng loạt với file cấu hình')
    parser.add_argument('--check-system', action='store_true', help='Kiểm tra hệ thống và hiển thị thông tin')
    parser.add_argument('--version', action='store_true', help='Hiển thị phiên bản')
    parser.add_argument('--setup', action='store_true', help='Thiết lập và chuẩn bị môi trường')
    
    return parser.parse_args()

def setup_environment():
    """Thiết lập môi trường cần thiết"""
    # Tạo các thư mục cần thiết
    directories = [
        'data',
        'data/beam_data',
        'data/dicom',
        'data/database',
        'data/images',
        'data/structures',
        'data/clinical_protocols',
        'data/machine_data',
        'data/models',
        'data/templates',
        'logs',
        'temp'
    ]
    
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for directory in directories:
        dir_path = os.path.join(root_dir, directory)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
            print(f"Đã tạo thư mục: {dir_path}")
    
    print("Thiết lập môi trường thành công.")
    return True

def show_splash_screen(app):
    """Hiển thị màn hình splash khi khởi động"""
    # Tạo splash screen
    splash_path = os.path.join(os.path.dirname(__file__), "ui", "icons", "new_icons", "splash.png")
    
    # Nếu không tìm thấy file splash, tạo splash screen mặc định
    if not os.path.exists(splash_path):
        from PyQt5.QtGui import QPixmap, QPainter, QColor, QFont
        pixmap = QPixmap(600, 400)
        pixmap.fill(QColor(40, 44, 52))
        painter = QPainter(pixmap)
        painter.setPen(QColor(255, 255, 255))
        font = QFont("Arial", 24)
        painter.setFont(font)
        painter.drawText(pixmap.rect(), Qt.AlignCenter, "QuangTPS\nHệ thống lập kế hoạch xạ trị mở")
        painter.end()
    else:
        pixmap = QPixmap(splash_path)
    
    splash = QSplashScreen(pixmap)
    splash.show()
    app.processEvents()
    
    return splash

def main():
    """Hàm chính để khởi động ứng dụng"""
    args = parse_arguments()
    
    # Thiết lập môi trường nếu được yêu cầu
    if args.setup:
        setup_environment()
        return 0
    
    # Thiết lập logger
    try:
        setup_logger(level=args.log_level)
        logger = get_logger(__name__)
    except Exception as e:
        print(f"Lỗi thiết lập logger: {e}")
        logger = None
    
    # Tải cấu hình
    config_path = args.config if args.config else None
    config = Config.get_instance()
    if config_path:
        # Here we could implement custom config file loading
        pass
    
    # Hiển thị phiên bản nếu được yêu cầu
    if args.version:
        from quangtps.core import __version__
        print(f"QuangTPS phiên bản {__version__}")
        return 0
    
    # Kiểm tra hệ thống nếu được yêu cầu
    if args.check_system:
        check_system()
        return 0
    
    # Chạy chế độ xử lý hàng loạt nếu được chỉ định
    if args.batch:
        batch_config_path = Path(args.batch)
        if logger:
            logger.info("Chạy chế độ xử lý hàng loạt với cấu hình: %s", batch_config_path)
        batch_process(batch_config_path)
        return 0
    
    # Khởi chạy giao diện người dùng
    if logger:
        logger.info("Khởi động QuangTPS...")
    
    try:
        # Kiểm tra thư mục và tạo nếu cần
        setup_environment()
        
        # Khởi tạo ứng dụng
        app = QApplication(sys.argv)
        
        # Hiển thị splash screen
        splash = show_splash_screen(app)
        
        # Tải stylesheet
        style_path = os.path.join(os.path.dirname(__file__), "ui", "styles", "main_style.qss")
        if os.path.exists(style_path):
            with open(style_path, "r", encoding="utf-8") as style_file:
                app.setStyleSheet(style_file.read())
        
        # Khởi tạo cửa sổ chính sau 1 giây
        window = None
        def show_main_window():
            nonlocal window
            try:
                window = MainWindow(config)
                window.show()
                splash.finish(window)
            except Exception as e:
                splash.close()
                error_text = traceback.format_exc()
                QMessageBox.critical(None, "Lỗi khởi động", 
                                    f"Không thể khởi động cửa sổ chính:\n\n{str(e)}")
                if logger:
                    logger.critical("Lỗi khởi động cửa sổ chính: %s", e, exc_info=True)
                sys.exit(1)
        
        QTimer.singleShot(1000, show_main_window)
        
        return app.exec_()
    except Exception as e:
        error_text = traceback.format_exc()
        if logger:
            logger.critical("Lỗi khi khởi động ứng dụng: %s", e, exc_info=True)
        
        # Hiển thị hộp thoại lỗi nếu có thể
        try:
            app = QApplication.instance()
            if not app:
                app = QApplication(sys.argv)
            QMessageBox.critical(None, "Lỗi khởi động", f"Không thể khởi động ứng dụng:\n\n{str(e)}")
        except:
            print(f"Lỗi khởi động ứng dụng: {error_text}")
            
        return 1

if __name__ == "__main__":
    sys.exit(main())