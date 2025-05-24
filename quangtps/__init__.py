#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QuangTPS: Hệ thống lập kế hoạch xạ trị mã nguồn mở.

QuangTPS là một hệ thống lập kế hoạch xạ trị mã nguồn mở được thiết kế để hỗ trợ
quá trình lập kế hoạch xạ trị trong điều trị ung thư.

Phát triển bởi một nhóm các chuyên gia vật lý y học và kỹ sư phần mềm,
QuangTPS cung cấp các tính năng từ cơ bản đến nâng cao cho việc lập kế hoạch
xạ trị, bao gồm:

- Phân đoạn cấu trúc giải phẫu
- Lập kế hoạch trị liệu
- Tối ưu hóa kế hoạch
- Tính toán liều
- Đánh giá kế hoạch
- Đảm bảo chất lượng

Phiên bản 0.9.3 cải thiện các chức năng phân tích sinh học và đánh giá độ bền vững
để cung cấp các công cụ nâng cao cho đánh giá kế hoạch xạ trị.
"""

__title__ = "QuangTPS"
__description__ = "Hệ thống lập kế hoạch xạ trị mã nguồn mở"
__version__ = "0.9.3"
__author__ = "Quang Team"
__author_email__ = "quangmacdang@gmail.com"

# Cập nhật phiên bản lên 0.9.2
__version__ = "0.9.2"

# Import các module chính
from quangtps import core
from quangtps import utils
from quangtps import ui

# Thiết lập logging
import os
import logging
from quangtps.utils.logging_config import configure_quangtps_logging

# Thư mục mặc định cho log
LOG_DIR = os.path.join(os.path.expanduser("~"), ".quangtps", "logs")
DEFAULT_LOG_LEVEL = logging.INFO

# Thiết lập logging - sử dụng configure_quangtps_logging thay vì setup_logging
configure_quangtps_logging(log_dir=LOG_DIR, debug=(DEFAULT_LOG_LEVEL == logging.DEBUG))

# Logger cho module này
logger = logging.getLogger(__name__)
logger.info(f"QuangTPS version {__version__} starting up")

__author__ = "QuangTPS Team"
__license__ = "MIT"
__copyright__ = "Copyright 2023, QuangTPS Team"

# Version details
VERSION_MAJOR = 0
VERSION_MINOR = 9
VERSION_PATCH = 2

# Platform detection
import platform

PLATFORM = platform.system().lower()
IS_WINDOWS = PLATFORM == "windows"
IS_LINUX = PLATFORM == "linux"
IS_MACOS = PLATFORM == "darwin"

# Define path constants
import sys
from pathlib import Path

# Get application base dir
if getattr(sys, "frozen", False):
    # We're running in a bundle
    BASE_DIR = Path(sys.executable).parent
else:
    # We're running in a normal Python environment
    BASE_DIR = Path(__file__).parent.parent

# Import essential components
from quangtps.core.types import Plan, Treatment, Structure
from quangtps.core.patient import Patient
from quangtps.core.logging import get_logger, setup_logging

import sys
import time
from pathlib import Path
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger(__name__)


# Define MLCType enum here to avoid import issues
class MLCType(Enum):
    STATIC = "static"
    DYNAMIC = "dynamic"
    STEP_AND_SHOOT = "step_and_shoot"


# Root directory
ROOT_DIR = Path(__file__).parent.absolute()

# Add module directories to path if needed
for module_dir in ["beams", "dose", "structures", "planning", "ui"]:
    module_path = ROOT_DIR / module_dir
    if module_path.exists() and str(module_path) not in sys.path:
        sys.path.insert(0, str(module_path))

# Import key components
try:
    # Core modules
    from quangtps.core import types, exceptions, services

    # Beam-related modules - import directly from treatment.beams to avoid circular imports
    from quangtps.treatment.beams.beam import Beam, BeamType
    from quangtps.treatment.beams.beam_geometry import BeamGeometry
    from quangtps.treatment.beams.beam_modifiers import Wedge, Block, Bolus, Compensator
    from quangtps.planning.beam_set import BeamSet
    from quangtps.planning.mlc import MLC

    # Structure-related modules
    from quangtps.structures.structure import Structure
    from quangtps.structures.structure_set import StructureSet

    # Planning modules
    from quangtps.planning.plan import Plan, PlanCollection

    # Dose calculation
    from quangtps.dose.dose_calculator import DoseCalculator

    # UI components (lazy-loaded to avoid overhead)
    def get_main_window(*args, **kwargs):
        """
        Trả về hoặc tạo mới cửa sổ chính của ứng dụng.

        Parameters
        ----------
        *args
            Các tham số vị trí được chuyển tiếp đến MainWindow.__init__
        **kwargs
            Các tham số từ khóa được chuyển tiếp đến MainWindow.__init__

        Returns
        -------
        MainWindow
            Cửa sổ chính của ứng dụng
        """
        global _main_window

        if _main_window is None:
            try:
                from quangtps.ui.main_window import MainWindow

                _main_window = MainWindow(*args, **kwargs)
            except Exception as e:
                logger.error(f"Lỗi khi khởi tạo cửa sổ chính: {str(e)}")

        return _main_window

    HAS_UI = True

except ImportError as e:
    logger.warning(f"Error importing modules: {str(e)}")
    logger.warning("Some features may not be available")
    HAS_UI = False

# Singleton instances
_app = None
_main_window = None


def get_app():
    """
    Trả về đối tượng QApplication hiện tại hoặc tạo mới nếu chưa có.

    Returns
    -------
    QApplication
        Đối tượng QApplication của PyQt
    """
    global _app

    if _app is None:
        try:
            from quangtps.ui import initialize_ui

            _app = initialize_ui(use_eclipse_theme=True)
        except Exception as e:
            logger.error(f"Lỗi khi khởi tạo QApplication: {str(e)}")

    return _app


def set_main_window(window):
    """
    Đặt cửa sổ chính toàn cục.

    Parameters
    ----------
    window : MainWindow
        Cửa sổ chính mới
    """
    global _main_window
    _main_window = window


def clear_global_instances():
    """
    Xóa các đối tượng toàn cục.
    Hữu ích khi cần khởi động lại ứng dụng.
    """
    global _app, _main_window
    _main_window = None
    # Không xóa _app vì QApplication là singleton


# Define public API
__all__ = [
    # Core
    "services",
    "types",
    "exceptions",
    # Beams
    "Beam",
    "BeamSet",
    "BeamType",
    "BeamGeometry",
    "Wedge",
    "Block",
    "Bolus",
    "Compensator",
    "MLC",
    "MLCType",
    # Structures
    "Structure",
    "StructureSet",
    # Planning
    "Plan",
    "PlanCollection",
    # Dose
    "DoseCalculator",
    # UI
    "get_main_window",
    "start_quangtps",
    # Constants
    "__version__",
    "ROOT_DIR",
]

# Thiết lập hệ thống plugin
from quangtps.plugins import init_plugins


# Hàm khởi động chính
def start_quangtps(apply_theme=True):
    """
    Khởi động QuangTPS với giao diện đồ họa đầy đủ.

    Parameters
    ----------
    apply_theme : bool, optional
        Áp dụng Eclipse-style theme nếu True, mặc định là True

    Returns
    -------
    int
        Mã thoát của ứng dụng hoặc 1 nếu có lỗi
    """
    try:
        # Hiển thị banner khởi động
        show_banner()

        # Log thông tin phiên bản
        logger.info(f"Khởi động QuangTPS phiên bản {__version__}")

        # Khởi tạo ứng dụng
        from quangtps.ui import initialize_ui

        app = initialize_ui(use_eclipse_theme=apply_theme)

        if app is None:
            logger.error("Không thể khởi tạo ứng dụng. Thoát.")
            return 1

        # Hiển thị màn hình splash (nếu có)
        splash = None
        try:
            from PyQt5.QtWidgets import QSplashScreen
            from PyQt5.QtGui import QPixmap
            from PyQt5.QtCore import Qt

            splash_path = os.path.join(
                os.path.dirname(__file__), "ui", "icons", "new_icons", "splash.png"
            )
            if os.path.exists(splash_path):
                splash = QSplashScreen(QPixmap(splash_path))
                splash.show()
                app.processEvents()
                splash.showMessage(
                    "Đang khởi động QuangTPS...",
                    Qt.AlignBottom | Qt.AlignCenter,
                    Qt.white,
                )
        except Exception as e:
            logger.warning(f"Không thể hiển thị màn hình khởi động: {str(e)}")

        # Khởi tạo hệ thống plugin
        logger.info("Đang khởi tạo hệ thống plugin...")
        if splash:
            splash.showMessage(
                "Đang khởi tạo hệ thống plugin...",
                Qt.AlignBottom | Qt.AlignCenter,
                Qt.white,
            )
            app.processEvents()
        init_plugins()

        # Khởi tạo các thành phần cốt lõi
        logger.info("Đang tải các thành phần cốt lõi...")
        if splash:
            splash.showMessage(
                "Đang tải các thành phần cốt lõi...",
                Qt.AlignBottom | Qt.AlignCenter,
                Qt.white,
            )
            app.processEvents()

        # Khởi tạo cửa sổ chính
        logger.info("Đang khởi tạo giao diện người dùng...")
        if splash:
            splash.showMessage(
                "Đang khởi tạo giao diện người dùng...",
                Qt.AlignBottom | Qt.AlignCenter,
                Qt.white,
            )
            app.processEvents()

        main_window = get_main_window()

        # Kiểm tra các thuật toán tính liều khả dụng
        logger.info("Đang kiểm tra các thuật toán tính liều khả dụng...")
        if splash:
            splash.showMessage(
                "Đang kiểm tra các thuật toán khả dụng...",
                Qt.AlignBottom | Qt.AlignCenter,
                Qt.white,
            )
            app.processEvents()

        try:
            from quangtps.dose.algorithms import get_available_algorithms

            algorithms = get_available_algorithms()
            logger.info(f"Các thuật toán tính liều khả dụng: {', '.join(algorithms)}")

            # Kiểm tra GPU
            try:
                from quangtps.dose.algorithms import get_best_available_algorithm

                best_algorithm = get_best_available_algorithm()
                if best_algorithm:
                    logger.info(
                        f"Thuật toán tính liều mặc định: {best_algorithm.get_display_name()}"
                    )
            except Exception as e:
                logger.warning(
                    f"Không thể xác định thuật toán tính liều tốt nhất: {str(e)}"
                )

        except Exception as e:
            logger.warning(f"Không thể kiểm tra các thuật toán tính liều: {str(e)}")

        # Hiển thị cửa sổ chính
        logger.info("Khởi động hoàn tất, hiển thị giao diện chính.")
        main_window.show()

        # Đóng splash screen nếu có
        if splash:
            splash.finish(main_window)

        # Khởi chạy vòng lặp sự kiện
        return app.exec_()

    except Exception as e:
        logger.error(
            f"Lỗi không xử lý được khi khởi động QuangTPS: {str(e)}", exc_info=True
        )
        return 1


# Hiển thị banner khi khởi động
def show_banner():
    """Hiển thị banner khởi động QuangTPS trên terminal."""
    banner = f"""
    ╔═════════════════════════════════════════════════════════╗
    ║                                                         ║
    ║   ██████╗ ██╗   ██╗ █████╗ ███╗   ██╗ ██████╗          ║
    ║  ██╔═══██╗██║   ██║██╔══██╗████╗  ██║██╔════╝          ║
    ║  ██║   ██║██║   ██║███████║██╔██╗ ██║██║  ███╗         ║
    ║  ██║▄▄ ██║██║   ██║██╔══██║██║╚██╗██║██║   ██║         ║
    ║  ╚██████╔╝╚██████╔╝██║  ██║██║ ╚████║╚██████╔╝         ║
    ║   ╚══▀▀═╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═══╝ ╚═════╝          ║
    ║                                                         ║
    ║        Radiation Treatment Planning System              ║
    ║                   v{__version__}                            ║
    ║                                                         ║
    ╚═════════════════════════════════════════════════════════╝
    """

    # Hiệu ứng vẽ banner
    try:
        for line in banner.split("\n"):
            print(line)
            time.sleep(0.01)  # Tạo hiệu ứng vẽ dần
    except:
        # Fallback nếu không thể tạo hiệu ứng
        print(banner)


def configure_logging(log_level=logging.INFO, log_to_file=True):
    """
    Cấu hình hệ thống logging.

    Parameters
    ----------
    log_level : int, optional
        Mức độ log, mặc định là logging.INFO
    log_to_file : bool, optional
        Ghi log vào file nếu True, mặc định là True
    """
    # Thiết lập root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Xóa các handler cũ nếu có
    while root_logger.handlers:
        root_logger.removeHandler(root_logger.handlers[0])

    # Tạo console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Tạo file handler nếu cần
    if log_to_file:
        logs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
        os.makedirs(logs_dir, exist_ok=True)
        log_file = os.path.join(
            logs_dir, f"quangtps_{time.strftime('%Y%m%d_%H%M%S')}.log"
        )

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(
            min(log_level, logging.DEBUG)
        )  # Ghi chi tiết hơn vào file
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

        logger.info(f"Log file được tạo tại: {log_file}")


# Initialize app function
def initialize():
    """
    Initialize the application.

    Returns
    -------
    bool
        True if initialization was successful, False otherwise.
    """
    logger.info("Initializing QuangTPS...")
    try:
        # Verify critical components
        from quangtps.core.verification import verify_critical_components

        if not verify_critical_components():
            logger.error("Failed to verify critical components.")
            return False

        # Setup configuration
        from quangtps.config import load_configuration

        if not load_configuration():
            logger.error("Failed to load configuration.")
            return False

        logger.info("QuangTPS initialized successfully.")
        return True
    except Exception as e:
        logger.error(f"Error during initialization: {str(e)}")
        return False


if __name__ == "__main__":
    sys.exit(start_quangtps())
