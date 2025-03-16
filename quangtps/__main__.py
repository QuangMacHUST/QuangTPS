"""
QuangTPS - Hệ thống Lập kế hoạch Xạ trị Mã nguồn Mở
Điểm vào chính của ứng dụng
"""

import sys
import argparse
from pathlib import Path

from PyQt5.QtWidgets import QApplication

from quangtps.core.logging import setup_logger, get_logger
from quangtps.core.config import Config
from quangtps.ui.main_window import MainWindow
from quangtps.scripts.batch_processing import batch_process
from quangtps.scripts.system_check import check_system

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
    
    return parser.parse_args()

def main():
    """Hàm chính để khởi động ứng dụng"""
    args = parse_arguments()
    
    # Thiết lập logger
    setup_logger(level=args.log_level)
    logger = get_logger(__name__)
    
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
        logger.info("Chạy chế độ xử lý hàng loạt với cấu hình: %s", batch_config_path)
        batch_process(batch_config_path)
        return 0
    
    # Khởi chạy giao diện người dùng
    logger.info("Khởi động QuangTPS...")
    try:
        app = QApplication(sys.argv)
        window = MainWindow(config)
        window.show()
        return app.exec_()
    except Exception as e:
        logger.critical("Lỗi khi khởi động ứng dụng: %s", e, exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())