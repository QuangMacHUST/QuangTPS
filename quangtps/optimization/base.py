#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module cơ sở cho các lớp tối ưu hóa.

Module này cung cấp các lớp cơ sở được sử dụng bởi các module tối ưu hóa khác
trong hệ thống QuangTPS.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Callable, Union

logger = logging.getLogger(__name__)


class OptimizerBase(ABC):
    """
    Lớp cơ sở cho các trình tối ưu hóa.

    Lớp này định nghĩa giao diện và phương thức chung cho tất cả
    các thuật toán tối ưu hóa trong QuangTPS.
    """

    def __init__(self):
        """Khởi tạo trình tối ưu hóa cơ sở."""
        self.is_initialized = False
        self.is_optimizing = False
        self.progress_callback = None
        self.iteration = 0
        self.best_score = float("inf")

    @abstractmethod
    def initialize(self) -> bool:
        """
        Khởi tạo trình tối ưu hóa.

        Returns:
            True nếu khởi tạo thành công, False nếu không
        """
        pass

    @abstractmethod
    def optimize(self, **kwargs) -> Any:
        """
        Chạy thuật toán tối ưu hóa.

        Args:
            **kwargs: Các tham số bổ sung cho quá trình tối ưu hóa

        Returns:
            Kết quả tối ưu hóa
        """
        pass

    def set_progress_callback(self, callback: Callable[..., None]) -> None:
        """
        Thiết lập hàm callback để báo cáo tiến độ.

        Args:
            callback: Hàm callback nhận các thông số tiến độ
        """
        self.progress_callback = callback

    def report_progress(self, progress: float, message: str = None) -> None:
        """
        Báo cáo tiến độ thông qua callback nếu có.

        Args:
            progress: Tỷ lệ hoàn thành (0.0 đến 1.0)
            message: Thông báo trạng thái không bắt buộc
        """
        if self.progress_callback:
            self.progress_callback(progress, message)
