#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
QuangTPS - Hệ thống lập kế hoạch xạ trị mã nguồn mở.

Hệ thống lập kế hoạch xạ trị cung cấp đầy đủ các công cụ để tạo,
tối ưu hóa và đánh giá kế hoạch điều trị xạ trị cho bệnh nhân.
"""

__version__ = "0.8.0"
__author__ = "QuangTPS Team"
__license__ = "MIT"
__description__ = "Hệ thống Lập kế hoạch Xạ trị Mã nguồn Mở"

# Import commonly used modules for easier access
from quangtps.core.logging import get_logger, setup_logging

import os
import sys
import time
import logging
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


if __name__ == "__main__":
    sys.exit(start_quangtps())
