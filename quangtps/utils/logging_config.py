#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Logging Configuration for QuangTPS.

This module provides standardized logging configuration for the QuangTPS
radiation therapy treatment planning system.
"""

import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Optional, Union


def setup_logging(
    level: Union[str, int] = logging.INFO,
    log_file: Optional[str] = None,
    console_output: bool = True,
    format_string: Optional[str] = None,
    max_bytes: int = 10485760,  # 10MB
    backup_count: int = 5,
    encoding: str = "utf-8",
) -> logging.Logger:
    """
    Thiết lập cấu hình logging cho QuangTPS.

    Parameters:
        level: Mức độ logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Đường dẫn file log, None = không ghi file
        console_output: Có xuất log ra console không
        format_string: Format string tùy chỉnh
        max_bytes: Kích thước tối đa file log (bytes)
        backup_count: Số lượng file backup
        encoding: Encoding cho file log

    Returns:
        logging.Logger: Logger đã được cấu hình
    """
    # Tạo logger root
    root_logger = logging.getLogger()

    # Xóa handlers cũ nếu có
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Thiết lập level
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    root_logger.setLevel(level)

    # Format mặc định
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    formatter = logging.Formatter(format_string)

    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # File handler với rotation
    if log_file:
        # Tạo thư mục nếu chưa có
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=max_bytes, backupCount=backup_count, encoding=encoding
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Lấy logger với tên cụ thể.

    Parameters:
        name: Tên logger (thường là __name__)

    Returns:
        logging.Logger: Logger instance
    """
    return logging.getLogger(name)


def setup_module_logging(
    module_name: str, level: Union[str, int] = logging.INFO
) -> logging.Logger:
    """
    Thiết lập logging cho một module cụ thể.

    Parameters:
        module_name: Tên module
        level: Mức độ logging

    Returns:
        logging.Logger: Logger cho module
    """
    logger = logging.getLogger(module_name)

    if not logger.handlers:
        # Thiết lập handler nếu chưa có
        handler = logging.StreamHandler()
        if isinstance(level, str):
            level = getattr(logging, level.upper())
        handler.setLevel(level)

        formatter = logging.Formatter(f"%(levelname)s:%(name)s:%(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)

    return logger


def configure_quangtps_logging(
    log_dir: str = "logs", debug: bool = False
) -> logging.Logger:
    """
    Cấu hình logging cho toàn bộ hệ thống QuangTPS.

    Parameters:
        log_dir: Thư mục chứa log files
        debug: Có bật chế độ debug không

    Returns:
        logging.Logger: Main logger
    """
    # Tạo thư mục logs
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)

    # Xác định level
    level = logging.DEBUG if debug else logging.INFO

    # File logs
    main_log = log_path / "quangtps.log"
    error_log = log_path / "quangtps_errors.log"

    # Thiết lập main logger
    main_logger = setup_logging(
        level=level,
        log_file=str(main_log),
        console_output=True,
        format_string="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Error logger riêng
    error_logger = logging.getLogger("quangtps.errors")
    error_handler = logging.handlers.RotatingFileHandler(
        str(error_log),
        maxBytes=5242880,  # 5MB
        backupCount=3,
        encoding="utf-8",
    )
    error_handler.setLevel(logging.ERROR)
    error_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(message)s"
    )
    error_handler.setFormatter(error_formatter)
    error_logger.addHandler(error_handler)

    # Thiết lập một số logger đặc biệt
    # Dose calculation logging
    dose_logger = logging.getLogger("quangtps.dose")
    dose_logger.setLevel(level)

    # Optimization logging
    opt_logger = logging.getLogger("quangtps.optimization")
    opt_logger.setLevel(level)

    # UI logging
    ui_logger = logging.getLogger("quangtps.ui")
    ui_logger.setLevel(level)

    # DICOM logging
    dicom_logger = logging.getLogger("quangtps.dicom")
    dicom_logger.setLevel(level)

    main_logger.info("QuangTPS logging system initialized")
    return main_logger


# Hàm tiện ích cho việc log exceptions
def log_exception(logger: logging.Logger, exc: Exception, context: str = ""):
    """
    Log exception với context.

    Parameters:
        logger: Logger instance
        exc: Exception object
        context: Context string
    """
    if context:
        logger.error(f"{context}: {type(exc).__name__}: {str(exc)}", exc_info=True)
    else:
        logger.error(f"{type(exc).__name__}: {str(exc)}", exc_info=True)


# Decorator để tự động log function calls
def log_function_call(logger: logging.Logger):
    """
    Decorator để log function calls.

    Parameters:
        logger: Logger instance
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
            try:
                result = func(*args, **kwargs)
                logger.debug(f"{func.__name__} completed successfully")
                return result
            except Exception as e:
                log_exception(logger, e, f"Error in {func.__name__}")
                raise

        return wrapper

    return decorator


# Thiết lập mặc định
def init_default_logging():
    """Khởi tạo logging mặc định cho QuangTPS."""
    try:
        return configure_quangtps_logging()
    except Exception as e:
        # Fallback to basic logging if advanced setup fails
        logging.basicConfig(
            level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s"
        )
        logger = logging.getLogger("quangtps")
        logger.warning(f"Could not setup advanced logging: {e}")
        return logger


# Khởi tạo logging khi import module
default_logger = init_default_logging()
