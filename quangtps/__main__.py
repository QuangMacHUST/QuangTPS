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
import time
from pathlib import Path
import logging

# Kiểm tra và cài đặt PyQt5
try:
    from PyQt5.QtWidgets import QApplication, QSplashScreen, QMessageBox
    from PyQt5.QtGui import QPixmap, QPalette, QColor, QIcon
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
    from quangtps.planning.clinical_protocols import ClinicalProtocolManager
    from quangtps.core.services import ServiceRegistry
    from quangtps.administration.rt_admin import RTAdministration, QAManagement
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
    from PyQt5.QtWidgets import QSplashScreen
    from PyQt5.QtGui import QPixmap, QPainter, QColor, QFont, QLinearGradient
    from PyQt5.QtCore import Qt

    splash_path = os.path.join(os.path.dirname(__file__), "ui", "icons", "new_icons", "splash.png")
    
    # Nếu không tìm thấy file splash, tạo splash screen mặc định
    if not os.path.exists(splash_path):
        pixmap = QPixmap(600, 400)
        pixmap.fill(QColor(40, 44, 52))
        
        # Tạo gradient background
        gradient = QLinearGradient(0, 0, 0, pixmap.height())
        gradient.setColorAt(0, QColor(40, 44, 52))
        gradient.setColorAt(1, QColor(30, 34, 42))
        
        painter = QPainter(pixmap)
        painter.fillRect(pixmap.rect(), gradient)
        
        # Vẽ tiêu đề
        painter.setPen(QColor(255, 255, 255))
        title_font = QFont("Arial", 32, QFont.Bold)
        painter.setFont(title_font)
        painter.drawText(pixmap.rect().adjusted(0, 50, 0, 0), Qt.AlignHCenter, "QuangTPS")
        
        # Vẽ phụ đề
        subtitle_font = QFont("Arial", 16)
        painter.setFont(subtitle_font)
        painter.drawText(pixmap.rect().adjusted(0, 120, 0, 0), Qt.AlignHCenter, "Hệ thống Lập kế hoạch Xạ trị")
        
        # Vẽ phiên bản
        version_font = QFont("Arial", 10)
        painter.setFont(version_font)
        painter.drawText(pixmap.rect().adjusted(0, 180, 0, 0), Qt.AlignHCenter, "Phiên bản 1.0.0")
        
        # Hiển thị thời gian
        painter.drawText(pixmap.rect().adjusted(0, 0, -20, -20), Qt.AlignRight | Qt.AlignBottom, 
                       f"Khởi động: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        painter.end()
        
        # Lưu lại để lần sau dùng
        os.makedirs(os.path.dirname(splash_path), exist_ok=True)
        pixmap.save(splash_path)
    else:
        pixmap = QPixmap(splash_path)
    
    splash = QSplashScreen(pixmap)
    splash.showMessage(
        "Đang khởi động QuangTPS...",
        Qt.AlignBottom | Qt.AlignCenter,
        QColor(255, 255, 255)
    )
    
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
        setup_logging()
        logger = get_logger(__name__)
    except Exception as e:
        print(f"Lỗi thiết lập logger: {e}")
        logger = None
    
    # Tải cấu hình
    config_path = args.config if args.config else None
    config = Config.get_instance()
    if config_path:
        try:
            config.load_from_file(config_path)
            if logger:
                logger.info(f"Đã tải cấu hình từ file: {config_path}")
        except Exception as e:
            if logger:
                logger.error(f"Lỗi khi tải file cấu hình: {str(e)}")
            else:
                print(f"Lỗi khi tải file cấu hình: {str(e)}")
    
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
        app.setApplicationName("QuangTPS")
        
        # Cấu hình style sheet nếu có
        try:
            style_path = os.path.join(os.path.dirname(__file__), "ui", "styles", "dark.qss")
            if os.path.exists(style_path):
                with open(style_path, "r", encoding='utf-8') as style_file:
                    app.setStyleSheet(style_file.read())
                    if logger:
                        logger.debug("Đã áp dụng style sheet")
        except Exception as style_error:
            if logger:
                logger.error(f"Lỗi khi tải style sheet: {str(style_error)}")
        
        # Hiển thị splash screen
        splash = show_splash_screen(app)
        
        # Khởi tạo cơ sở dữ liệu
        try:
            from quangtps.database.db_connector import DBConnector
            db = DBConnector.get_instance()
            if logger:
                logger.info("Đã khởi tạo kết nối đến cơ sở dữ liệu")
                
            # Cập nhật thông tin trạng thái
            splash.showMessage(
                "Đang khởi tạo cơ sở dữ liệu...",
                Qt.AlignBottom | Qt.AlignCenter,
                Qt.white
            )
            app.processEvents()
                
        except Exception as db_error:
            if logger:
                logger.error(f"Lỗi khi khởi tạo cơ sở dữ liệu: {str(db_error)}")
            QMessageBox.critical(
                None, 
                "Lỗi Cơ sở dữ liệu",
                f"Không thể khởi tạo cơ sở dữ liệu:\n{str(db_error)}\n\nỨng dụng có thể không hoạt động đúng."
            )
        
        # Khởi tạo các dịch vụ
        try:
            # Cập nhật thông tin trạng thái
            splash.showMessage(
                "Đang khởi tạo các dịch vụ...",
                Qt.AlignBottom | Qt.AlignCenter,
                Qt.white
            )
            app.processEvents()
            
            # Khởi tạo các dịch vụ cốt lõi: DICOM, dose calc, v.v.
            from quangtps.core.services import ServiceManager
            service_manager = ServiceManager.get_instance()
            service_manager.initialize_services()
            
            if logger:
                logger.info("Đã khởi tạo các dịch vụ")
                
        except Exception as service_error:
            if logger:
                logger.error(f"Lỗi khi khởi tạo các dịch vụ: {str(service_error)}")
            QMessageBox.warning(
                None, 
                "Cảnh báo",
                f"Không thể khởi tạo một số dịch vụ:\n{str(service_error)}\n\nMột số tính năng có thể không hoạt động đúng."
            )
        
        # Khởi tạo cửa sổ chính    
        def show_main_window():
            """Hàm hiển thị cửa sổ chính sau khi splash screen đóng"""
            try:
                main_window = MainWindow()
                
                # Load các module & plugin
                main_window.load_plugins()
                
                # Tùy chỉnh cửa sổ
                screen_size = app.primaryScreen().size()
                main_window.resize(int(screen_size.width() * 0.9), int(screen_size.height() * 0.9))
                main_window.show()
                
                # Đóng splash
                splash.finish(main_window)
                
                if logger:
                    logger.info("Ứng dụng đã khởi động thành công")
                
                # Kiểm tra cập nhật nếu được cấu hình
                if config.check_for_updates_on_startup:
                    QTimer.singleShot(5000, main_window.check_for_updates)
            
            except Exception as window_error:
                if logger:
                    logger.error(f"Lỗi khi tạo cửa sổ chính: {str(window_error)}")
                    logger.error(traceback.format_exc())
                
                splash.close()
                
                QMessageBox.critical(
                    None, 
                    "Lỗi Khởi động",
                    f"Không thể khởi tạo cửa sổ chính:\n{str(window_error)}"
                )
                sys.exit(1)
        
        # Trì hoãn hiển thị cửa sổ chính để hiển thị splash
        QTimer.singleShot(1500, show_main_window)
        
        # Initialize application settings
        setup_application(app)
        
        # Initialize core services
        initialize_services()
        
        return app.exec_()
        
    except Exception as e:
        if logger:
            logger.critical(f"Lỗi nghiêm trọng khi khởi động: {str(e)}")
            logger.critical(traceback.format_exc())
        else:
            print(f"Lỗi nghiêm trọng khi khởi động: {str(e)}")
            traceback.print_exc()
            
        QMessageBox.critical(
            None, 
            "Lỗi Khởi động",
            f"Không thể khởi động ứng dụng:\n{str(e)}"
        )
        return 1

def setup_application(app):
    """Configure application settings."""
    # Set application style - use Fusion style with a blue color scheme similar to Eclipse
    app.setStyle("Fusion")
    
    # Configure a blue color palette similar to Eclipse
    palette = app.palette()
    
    # Set blue accent color similar to Eclipse
    blue_accent = QColor(42, 130, 218)
    lighter_blue = QColor(240, 248, 255)
    
    palette.setColor(QPalette.Highlight, blue_accent)
    palette.setColor(QPalette.HighlightedText, Qt.white)
    palette.setColor(QPalette.Link, blue_accent)
    
    # Set background and text colors
    palette.setColor(QPalette.Window, Qt.white)
    palette.setColor(QPalette.WindowText, Qt.black)
    palette.setColor(QPalette.Base, Qt.white)
    palette.setColor(QPalette.AlternateBase, lighter_blue)
    palette.setColor(QPalette.ToolTipBase, lighter_blue)
    palette.setColor(QPalette.ToolTipText, Qt.black)
    palette.setColor(QPalette.Text, Qt.black)
    
    # Set button colors
    palette.setColor(QPalette.Button, QColor(240, 240, 240))
    palette.setColor(QPalette.ButtonText, Qt.black)
    
    app.setPalette(palette)
    
    # Set stylesheet for additional customization
    app.setStyleSheet("""
        QToolBar { border-bottom: 1px solid #cccccc; }
        QStatusBar { border-top: 1px solid #cccccc; }
        QTabWidget::pane { border: 1px solid #cccccc; }
        QTabBar::tab { 
            padding: 6px 12px;
            background-color: #f0f0f0;
            border: 1px solid #cccccc;
            border-bottom: none;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
        }
        QTabBar::tab:selected { 
            background-color: white;
            border-bottom: 1px solid white;
        }
        QTreeView { 
            border: 1px solid #cccccc;
            alternate-background-color: #f7f7f7;
        }
        QHeaderView::section {
            background-color: #f0f0f0;
            padding: 4px;
            border: 1px solid #cccccc;
            border-left: none;
        }
        QTableView {
            gridline-color: #e0e0e0;
            selection-background-color: #2a82da;
            selection-color: white;
        }
        QPushButton {
            padding: 4px 10px;
            border: 1px solid #cccccc;
            border-radius: 2px;
            background-color: #f5f5f5;
        }
        QPushButton:hover {
            background-color: #e0e0e0;
        }
        QPushButton:pressed {
            background-color: #d0d0d0;
        }
    """)
    
    # Set window icon
    icon_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 
        "ui", "icons", "logo.png"
    )
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))
    
    return app

def initialize_services():
    """Initialize and register core services."""
    from quangtps.core.config import Config
    from quangtps.database.patient_db import PatientDB
    from quangtps.database.structure_db import StructureDB
    from quangtps.planning.clinical_protocols import ClinicalProtocolManager
    from quangtps.administration.rt_admin import RTAdministration, QAManagement
    
    # Initialize and register essential services
    config = Config()
    ServiceRegistry.register(Config.__name__, config)
    
    patient_db = PatientDB()
    ServiceRegistry.register(PatientDB.__name__, patient_db)
    
    structure_db = StructureDB()
    ServiceRegistry.register(StructureDB.__name__, structure_db)
    
    protocol_manager = ClinicalProtocolManager()
    ServiceRegistry.register(ClinicalProtocolManager.__name__, protocol_manager)
    
    rt_admin = RTAdministration()
    ServiceRegistry.register(RTAdministration.__name__, rt_admin)
    
    qa_management = QAManagement()
    ServiceRegistry.register(QAManagement.__name__, qa_management)

# Hàm bổ sung để kiểm tra môi trường và phụ thuộc
def check_dependencies():
    """Check for required dependencies."""
    missing_deps = []
    
    try:
        import numpy
    except ImportError:
        missing_deps.append("numpy")
        
    try:
        import pydicom
    except ImportError:
        missing_deps.append("pydicom")
        
    try:
        import matplotlib
    except ImportError:
        missing_deps.append("matplotlib")
        
    try:
        import PyQt5
    except ImportError:
        missing_deps.append("PyQt5")
    
    try:
        import SimpleITK
    except ImportError:
        missing_deps.append("SimpleITK")
    
    try:
        import scipy
    except ImportError:
        missing_deps.append("scipy")
    
    if missing_deps:
        print("Missing required dependencies:")
        for dep in missing_deps:
            print(f"- {dep}")
        return False
    
    return True

def setup_logging():
    """
    Thiết lập ghi nhật ký cho ứng dụng.
    """
    # Get the logger for this module
    logger = logging.getLogger(__name__)
    
    # Tạo thư mục logs nếu chưa tồn tại
    logs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    log_file = os.path.join(logs_dir, 'quangtps.log')
    
    # Định dạng nhật ký
    log_format = '%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # Thiết lập cấu hình ghi nhật ký
    logging.basicConfig(
        level=logging.DEBUG,  # Change to DEBUG level for more verbose logging
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # Thiết lập mức độ ghi nhật ký cho một số module thường gửi quá nhiều thông báo
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    # Don't adjust other loggers - keep them at DEBUG for troubleshooting
    
    logger.info("Đã thiết lập ghi nhật ký. Tệp nhật ký: %s", log_file)

# Điểm vào của ứng dụng
if __name__ == "__main__":
    # Kiểm tra phụ thuộc trước khi khởi động
    if check_dependencies():
        sys.exit(main())