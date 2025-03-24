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
import traceback
from pathlib import Path

# Thiết lập UTF-8 cho console trước khi import bất kỳ thứ gì
if sys.platform == 'win32':
    try:
        import codecs
        os.environ['PYTHONIOENCODING'] = 'utf-8'
        os.environ['PYTHONUTF8'] = '1'
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)
    except Exception as e:
        print(f"Cảnh báo: Không thể thiết lập UTF-8 cho console: {str(e)}")

# Import các module từ thư viện chuẩn
try:
    # Thử import PyQt5 từ module thay vì class cụ thể
    import PyQt5.QtWidgets as QtWidgets
    import PyQt5.QtGui as QtGui
    import PyQt5.QtCore as QtCore
    HAS_PYQT = True
except ImportError as e:
    print(f"CẢNH BÁO: Không thể import module PyQt5: {str(e)}")
    print("Ứng dụng GUI sẽ không hoạt động. Vui lòng cài đặt PyQt5.")
    HAS_PYQT = False

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
    
    # Thiết lập UTF-8 cho console
    try:
        # Sử dụng hàm setup_utf8_console từ module quangtps.core.logging nếu đã import được
        from quangtps.core.logging import setup_utf8_console
        setup_utf8_console()
    except ImportError:
        # Nếu chưa import được (ví dụ khi mới khởi động), sử dụng cài đặt cơ bản
        if sys.platform == 'win32':
            try:
                import codecs
                os.environ['PYTHONIOENCODING'] = 'utf-8'
                os.environ['PYTHONUTF8'] = '1'
                sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
                sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)
            except Exception:
                pass  # Bỏ qua nếu không thực hiện được
    
    # Thiết lập logging cho console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(logging.Formatter(log_format))
    
    # Thiết lập logging cho file
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(log_level)
    file_handler.setFormatter(logging.Formatter(log_format))
    
    # Cấu hình root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Xóa handler hiện có để tránh trùng lặp
    if root_logger.handlers:
        root_logger.handlers = []
        
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
    # Đảm bảo PyQt5 được import
    if not HAS_PYQT:
        return None, None
        
    app = QtWidgets.QApplication(sys.argv)
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
    
    # Nếu không tìm thấy splash.png, tạo splash screen tự động
    if not os.path.exists(splash_path):
        pixmap = QtGui.QPixmap(600, 400)
        pixmap.fill(QtGui.QColor(40, 44, 52))
        
        # Tạo gradient background
        gradient = QtGui.QLinearGradient(0, 0, 0, pixmap.height())
        gradient.setColorAt(0, QtGui.QColor(40, 44, 52))
        gradient.setColorAt(1, QtGui.QColor(30, 34, 42))
        
        painter = QtGui.QPainter(pixmap)
        painter.fillRect(pixmap.rect(), gradient)
        
        # Vẽ tiêu đề
        painter.setPen(QtGui.QColor(255, 255, 255))
        title_font = QtGui.QFont("Arial", 32, QtGui.QFont.Bold)
        painter.setFont(title_font)
        painter.drawText(pixmap.rect().adjusted(0, 50, 0, 0), QtCore.Qt.AlignHCenter, "QuangTPS")
        
        # Vẽ phụ đề
        subtitle_font = QtGui.QFont("Arial", 16)
        painter.setFont(subtitle_font)
        painter.drawText(pixmap.rect().adjusted(0, 120, 0, 0), QtCore.Qt.AlignHCenter, "Hệ thống Lập kế hoạch Xạ trị")
        
        # Vẽ phiên bản
        version_font = QtGui.QFont("Arial", 10)
        painter.setFont(version_font)
        painter.drawText(pixmap.rect().adjusted(0, 180, 0, 0), QtCore.Qt.AlignHCenter, "Phiên bản 1.0.0")
        
        # Hiển thị thời gian
        painter.drawText(pixmap.rect().adjusted(0, 0, -20, -20), QtCore.Qt.AlignRight | QtCore.Qt.AlignBottom, 
                       f"Khởi động: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        painter.end()
        
        # Lưu lại để lần sau dùng
        os.makedirs(os.path.dirname(splash_path), exist_ok=True)
        pixmap.save(splash_path)
        
        splash = QtWidgets.QSplashScreen(pixmap)
    else:
        # Tạo splash screen từ file đã tồn tại
        splash_pixmap = QtGui.QPixmap(splash_path)
        splash = QtWidgets.QSplashScreen(splash_pixmap)
    
    # Thiết lập văn bản
    splash.showMessage(
        "Đang khởi động QuangTPS...",
        QtCore.Qt.AlignBottom | QtCore.Qt.AlignCenter,
        QtGui.QColor(255, 255, 255)
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
    
    parser.add_argument('--debug', action='store_true',
                       help='Chạy ở chế độ debug với thông tin chi tiết')
    
    return parser.parse_args()

def main():
    """Hàm chính của script."""
    # Phân tích đối số
    args = parse_arguments()
    
    # Thiết lập môi trường
    setup_environment()
    
    # Thiết lập logging
    logger = setup_logging(args.verbose or args.debug)
    
    # Kiểm tra các thư viện phụ thuộc
    if not check_dependencies():
        logger.error("Không thể khởi động QuangTPS do thiếu thư viện")
        return 1
    
    # Debug mode
    if args.debug:
        logger.debug("Đang chạy ở chế độ DEBUG")
        # Chế độ debug thì không cần splash screen
        args.no_splash = True
    
    # Hiển thị splash screen nếu không ở chế độ console và không tắt splash screen
    if not args.console and not args.no_splash and HAS_PYQT:
        app, splash = show_splash_screen()
        if app is not None:
            # Độ trễ để hiển thị splash screen (đủ để người dùng thấy)
            QtCore.QTimer.singleShot(2000, lambda: start_application(app, splash, args))
            # Bắt đầu vòng lặp sự kiện
            return app.exec_()
    else:
        if HAS_PYQT:
            app = QtWidgets.QApplication(sys.argv)
            start_application(app, None, args)
            return app.exec_()
        else:
            logger.error("Không thể khởi động giao diện đồ họa do thiếu PyQt5")
            return 1

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
        error_traceback = traceback.format_exc()
        logger.error(f"Lỗi khi khởi động ứng dụng: {str(e)}")
        logger.error(f"Chi tiết lỗi:\n{error_traceback}")
        
        # Hiển thị thông báo lỗi cho người dùng
        try:
            if HAS_PYQT:
                error_msg = QtWidgets.QMessageBox()
                error_msg.setIcon(QtWidgets.QMessageBox.Critical)
                error_msg.setWindowTitle("Lỗi khởi động")
                error_msg.setText("Không thể khởi động QuangTPS")
                error_msg.setDetailedText(f"{str(e)}\n\n{error_traceback}")
                error_msg.exec_()
            else:
                print(f"Lỗi nghiêm trọng: {str(e)}")
                print(error_traceback)
        except Exception as ex:
            print(f"Không thể hiển thị hộp thoại lỗi: {str(ex)}")
            print(f"Lỗi nghiêm trọng: {str(e)}")
            print(error_traceback)

if __name__ == "__main__":
    sys.exit(main()) 