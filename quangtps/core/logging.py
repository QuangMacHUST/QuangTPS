"""
Logging Module for QuangTPS

This module provides logging utilities for the QuangTPS treatment planning
system. It sets up consistent logging throughout the application.
"""

import os
import sys
import logging
import datetime
from pathlib import Path
from typing import Optional, Union, Dict, Any

# Default log format
DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Default log directory
DEFAULT_LOG_DIR = Path.home() / ".quangtps" / "logs"


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[Union[str, Path]] = None,
    log_format: str = DEFAULT_FORMAT,
    log_to_console: bool = True,
    log_to_file: bool = True,
) -> None:
    """
    Set up logging configuration for the application.

    Args:
        level: Logging level (default: INFO)
        log_file: Path to log file (default: auto-generated based on date)
        log_format: Format string for log messages
        log_to_console: Whether to log to console
        log_to_file: Whether to log to file
    """
    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create formatter
    formatter = logging.Formatter(log_format)

    # Add console handler if requested
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        # Set encoding for console output if supported
        if hasattr(sys.stdout, "reconfigure"):
            try:
                sys.stdout.reconfigure(encoding="utf-8")
            except:
                pass
        root_logger.addHandler(console_handler)

    # Add file handler if requested
    if log_to_file:
        # Create log directory if it doesn't exist
        if log_file is None:
            # Generate default log file name based on date and time
            today = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_dir = DEFAULT_LOG_DIR
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / f"quangtps_{today}.log"
        else:
            log_file = Path(log_file)
            log_file.parent.mkdir(parents=True, exist_ok=True)

            # Create file handler with UTF-8 encoding
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)

    # Log system info
    logger = get_logger(__name__)
    logger.info(f"QuangTPS logging initialized at level {logging.getLevelName(level)}")
    if log_to_file:
        logger.info(f"Logging to file: {log_file}")

    logger.debug(f"Python version: {sys.version}")
    logger.debug(f"Platform: {sys.platform}")

    # Log versions of key dependencies
    try:
        import numpy as np

        logger.debug(f"NumPy version: {np.__version__}")
    except ImportError:
        pass

    try:
        from PyQt5.QtCore import QT_VERSION_STR

        logger.debug(f"Qt version: {QT_VERSION_STR}")
    except ImportError:
        pass


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.

    Args:
        name: Logger name (typically __name__ for module-level loggers)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def set_log_level(level: Union[int, str]) -> None:
    """
    Set the logging level for the root logger.

    Args:
        level: Logging level as integer or string
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper())

    logging.getLogger().setLevel(level)
    get_logger(__name__).info(f"Log level changed to {logging.getLevelName(level)}")


def log_exception(
    logger: logging.Logger, exc: Exception, context: Optional[Dict[str, Any]] = None
) -> None:
    """
    Log an exception with optional context information.

    Args:
        logger: Logger instance
        exc: Exception to log
        context: Optional dictionary with context information
    """
    message = f"Exception: {type(exc).__name__}: {str(exc)}"

    if context:
        context_str = ", ".join(f"{k}={v}" for k, v in context.items())
        message += f" [Context: {context_str}]"

    logger.exception(message)


# Initialize default logging configuration
if not logging.getLogger().handlers:
    setup_logging()
